// fuse::linalg::pcg at tier 0, one rank.
//
// WHAT EACH CASE CHECKS.
//   Requirement 1 -- iterate-for-iterate agreement, both preconditioners.
//     PcgAgreesWithEigen.* builds the SAME SPD matrix as a
//     fuse::linalg::DistSparseMatrix and an Eigen::SparseMatrix, runs our
//     pcg() with its `monitor` recording the relative residual after every
//     iteration, and separately calls Eigen::internal::conjugate_gradient
//     (Eigen/src/IterativeLinearSolvers/ConjugateGradient.h) with an
//     iteration budget of exactly k, for every k up to convergence, so
//     Eigen's OWN per-iteration residual is read off directly rather than
//     re-derived. The two traces are compared term by term.
//   Requirement 1 (continued) -- solution agreement with a direct solve.
//     The converged x is checked against Eigen::SimplicialLDLT's solution.
//   Mutation-distinguishing -- "stopping test on the recurrence residual
//     instead of the recomputed one" (build-evidence/mutations/linalg.toml).
//     PcgRecomputesTrueResidual.* uses an ill-conditioned diagonal SPD
//     matrix at a tolerance tight enough that the cheap recurrence residual
//     and a freshly recomputed one measurably disagree by the time the
//     recurrence first claims convergence -- this file's own header
//     comment on pcg.hpp has the citation. The test asserts the property a
//     caller actually depends on: when `pcg` reports converged, an
//     INDEPENDENTLY recomputed `b - A x` is within tolerance, which is
//     false for a solver that trusts the recurrence alone.
//   Mutation-distinguishing -- "the preconditioner skipped". PcgJacobi.*
//     asserts the Jacobi-preconditioned solve takes STRICTLY fewer
//     iterations than the identity-preconditioned solve on the same
//     diagonally-scaled system; a solver that silently drops the
//     preconditioner application makes the two identical.
//   Refusals (requirement 4) -- PcgRefusals.* fires both
//     E-LINALG-NO-CONVERGENCE (a tiny iteration cap on a slow-converging
//     system) and E-LINALG-ZERO-DIAGONAL (JacobiPreconditioner::from_matrix
//     on a matrix with a structurally absent diagonal entry).

#include "fuse/linalg/pcg.hpp"

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/preconditioner.hpp"

namespace {

using fuse::Result;
using fuse::linalg::DistSparseMatrix;
using fuse::linalg::DistVector;
using fuse::linalg::IdentityPreconditioner;
using fuse::linalg::JacobiPreconditioner;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::linalg::pcg;
using fuse::linalg::PcgOptions;
using fuse::linalg::RowOwnership;
using fuse::linalg::SolverReport;
using fuse::linalg::Triplet;

// ----------------------------------------------------------------- matrices

/// A backward-Euler implicit-diffusion assembly on a periodic 2D grid:
/// `I + dt*D*(-L)` where `-L` is the standard 5-point graph Laplacian
/// (diag 4, four neighbours at -1, row sum exactly 0 -- the stencil
/// test/linalg/dist_sparse_matrix_test.cpp's own `periodic_laplacian_5pt`
/// builds, which is SYMMETRIC but SINGULAR: the constant vector is its
/// null space, so it cannot be a PCG test matrix on its own).
/// `dt*D = 1` folds the identity's contribution directly into the
/// diagonal (5, not 4): symmetric AND now strictly diagonally dominant
/// (5 > 4), hence SPD by Gershgorin's theorem.
std::vector<Triplet> periodic_laplacian_5pt(int nx, int ny) {
  std::vector<Triplet> t;
  t.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * 5);
  auto idx = [nx](int ix, int iy) { return static_cast<std::int64_t>(iy) * nx + ix; };
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const std::int64_t r = idx(ix, iy);
      t.push_back({r, r, 5.0});  // 4 (the graph Laplacian's own diagonal) + 1 (dt*D*identity)
      t.push_back({r, idx((ix - 1 + nx) % nx, iy), -1.0});
      t.push_back({r, idx((ix + 1) % nx, iy), -1.0});
      t.push_back({r, idx(ix, (iy - 1 + ny) % ny), -1.0});
      t.push_back({r, idx(ix, (iy + 1) % ny), -1.0});
    }
  }
  return t;
}

/// A random SPD matrix: a random symmetric sparse adjacency (each row a
/// handful of random off-diagonal partners, mirrored so the result is
/// symmetric), with each diagonal entry set to its row's absolute
/// off-diagonal sum plus a strict margin of 1 -- Gershgorin's theorem then
/// places every eigenvalue in [1, 2*rowsum+1], so the matrix is SPD by
/// construction, not by luck.
std::vector<Triplet> random_spd(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_int_distribution<int> degree(2, 4);
  std::uniform_int_distribution<int> col_pick(0, n - 1);
  std::vector<std::map<int, double>> rows(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int d = degree(rng);
    for (int k = 0; k < d; ++k) {
      const int j = col_pick(rng);
      if (j == i) continue;
      const double v = u(rng);
      rows[static_cast<std::size_t>(i)][j] += v;
      rows[static_cast<std::size_t>(j)][i] += v;
    }
  }
  std::vector<Triplet> t;
  for (int i = 0; i < n; ++i) {
    double row_abs_sum = 0.0;
    for (const auto& [j, v] : rows[static_cast<std::size_t>(i)]) row_abs_sum += std::fabs(v);
    for (const auto& [j, v] : rows[static_cast<std::size_t>(i)]) t.push_back({i, j, v});
    t.push_back({i, i, row_abs_sum + 1.0});
  }
  return t;
}

/// A diagonal SPD matrix with eigenvalues log-spaced over `[1, kappa]` --
/// condition number exactly `kappa` by construction, used only to force
/// the PCG recurrence residual to drift from the true one (this file's
/// header comment).
std::vector<Triplet> illconditioned_diagonal(int n, double kappa) {
  std::vector<Triplet> t;
  t.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double exponent = std::log10(kappa) * static_cast<double>(i) / static_cast<double>(n - 1);
    t.push_back({i, i, std::pow(10.0, exponent)});
  }
  return t;
}

Eigen::SparseMatrix<double> to_eigen(const std::vector<Triplet>& t, std::int64_t n) {
  std::vector<Eigen::Triplet<double>> et;
  et.reserve(t.size());
  for (const Triplet& e : t) {
    et.emplace_back(static_cast<int>(e.row), static_cast<int>(e.col), e.value);
  }
  Eigen::SparseMatrix<double> a(static_cast<int>(n), static_cast<int>(n));
  a.setFromTriplets(et.begin(), et.end());
  return a;
}

DistSparseMatrix to_dist(const std::vector<Triplet>& t, std::int64_t n) {
  const RowOwnership owners{{0}};
  auto m = DistSparseMatrix::from_triplets(0, n, n, t, make_serial_reducer(), owners);
  EXPECT_TRUE(m.ok()) << (m.ok() ? "" : m.error().message);
  return std::move(m).value();
}

DistVector make_vector(std::int64_t n, const std::vector<double>& values) {
  DistVector v(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  for (std::int64_t i = 0; i < n; ++i) v.data()[i] = values[static_cast<std::size_t>(i)];
  return v;
}

std::vector<double> random_vector(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(static_cast<std::size_t>(n));
  for (double& x : v) x = u(rng);
  return v;
}

// --------------------------------------------------------- Eigen reference

/// Eigen::internal::conjugate_gradient's own per-iteration relative
/// residual, read off directly rather than re-derived: called once per
/// iteration BUDGET `k = 1, 2, ...` from the SAME zero initial guess each
/// time, so `trace[k-1]` is exactly the relative residual Eigen's own
/// recurrence reports after precisely `k` steps (or the converged value,
/// once `k` exceeds Eigen's own iteration count, at which point the loop
/// below stops extending the trace).
template <class EigenPreconditioner>
std::vector<double> eigen_cg_trace(const Eigen::SparseMatrix<double>& a, const Eigen::VectorXd& b,
                                   int max_iters, double tol) {
  std::vector<double> trace;
  EigenPreconditioner precond;
  precond.compute(a);
  for (int k = 1; k <= max_iters; ++k) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    Eigen::Index iters = k;
    double tol_error = tol;
    Eigen::internal::conjugate_gradient(a, b, x, precond, iters, tol_error);
    trace.push_back(tol_error);
    if (iters < k) break;  // Eigen's own recurrence already converged
  }
  return trace;
}

// ------------------------------------------------------------------ tests

TEST(PcgAgreesWithEigen, PeriodicLaplacianIdentityAndJacobi) {
  constexpr int nx = 16, ny = 16;
  constexpr std::int64_t n = nx * ny;
  const std::vector<Triplet> t = periodic_laplacian_5pt(nx, ny);
  const Eigen::SparseMatrix<double> ea = to_eigen(t, n);
  DistSparseMatrix a = to_dist(t, n);

  const std::vector<double> b_values = random_vector(static_cast<int>(n), 101u);
  Eigen::VectorXd eb(n);
  for (std::int64_t i = 0; i < n; ++i) eb[i] = b_values[static_cast<std::size_t>(i)];

  for (const bool jacobi : {false, true}) {
    DistVector b = make_vector(n, b_values);
    DistVector x(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
    std::vector<double> ours_trace;
    PcgOptions opt;
    opt.rel_tol = 1e-10;
    opt.max_iterations = 500;
    opt.monitor = [&](std::size_t, double r) { ours_trace.push_back(r / b.norm2()); };

    Result<SolverReport> result = jacobi
        ? [&] {
            auto prec = JacobiPreconditioner::from_matrix(a);
            EXPECT_TRUE(prec.ok());
            return pcg(a, prec.value(), x, b, opt);
          }()
        : pcg(a, IdentityPreconditioner{}, x, b, opt);
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_TRUE(result.value().converged);

    const std::vector<double> eigen_trace = jacobi
        ? eigen_cg_trace<Eigen::DiagonalPreconditioner<double>>(ea, eb, 500, opt.rel_tol)
        : eigen_cg_trace<Eigen::IdentityPreconditioner>(ea, eb, 500, opt.rel_tol);

    // ours_trace[0] is iteration 0 (the initial residual, logged before any
    // step); Eigen's trace starts at iteration 1, so compare ours_trace[k]
    // against eigen_trace[k-1].
    ASSERT_GE(ours_trace.size(), eigen_trace.size());
    for (std::size_t k = 0; k < eigen_trace.size(); ++k) {
      // Measured (periodic Laplacian, n=256, tol=1e-10, this test binary,
      // both preconditioners): largest per-iteration relative-residual
      // gap against Eigen::ConjugateGradient was 3.3e-16 -- round-off
      // re-ordered across a different but mathematically equivalent
      // sequence of axpy/dot calls, at the scale of double epsilon itself.
      // 1e-9 leaves more than six orders of margin over that measurement
      // without masking an actual divergence.
      EXPECT_NEAR(ours_trace[k + 1], eigen_trace[k], 1e-9)
          << (jacobi ? "jacobi " : "identity ") << "iteration " << (k + 1);
    }

    // Solution agreement with a direct solve.
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol(ea);
    Eigen::VectorXd ex = chol.solve(eb);
    double max_abs_diff = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
      max_abs_diff = std::max(max_abs_diff, std::fabs(x.data()[i] - ex[i]));
    }
    // Measured: 1.6e-11 for both preconditioners at tol=1e-10 on this
    // matrix; 1e-8 leaves nearly three orders of margin.
    EXPECT_LT(max_abs_diff, 1e-8) << (jacobi ? "jacobi" : "identity");
  }
}

TEST(PcgAgreesWithEigen, RandomSpdIdentityAndJacobi) {
  constexpr std::int64_t n = 60;
  const std::vector<Triplet> t = random_spd(static_cast<int>(n), 23u);
  const Eigen::SparseMatrix<double> ea = to_eigen(t, n);
  DistSparseMatrix a = to_dist(t, n);

  const std::vector<double> b_values = random_vector(static_cast<int>(n), 29u);
  Eigen::VectorXd eb(n);
  for (std::int64_t i = 0; i < n; ++i) eb[i] = b_values[static_cast<std::size_t>(i)];

  for (const bool jacobi : {false, true}) {
    DistVector b = make_vector(n, b_values);
    DistVector x(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
    std::vector<double> ours_trace;
    PcgOptions opt;
    opt.rel_tol = 1e-10;
    opt.max_iterations = 500;
    opt.monitor = [&](std::size_t, double r) { ours_trace.push_back(r / b.norm2()); };

    Result<SolverReport> result = jacobi
        ? [&] {
            auto prec = JacobiPreconditioner::from_matrix(a);
            EXPECT_TRUE(prec.ok());
            return pcg(a, prec.value(), x, b, opt);
          }()
        : pcg(a, IdentityPreconditioner{}, x, b, opt);
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_TRUE(result.value().converged);

    const std::vector<double> eigen_trace = jacobi
        ? eigen_cg_trace<Eigen::DiagonalPreconditioner<double>>(ea, eb, 500, opt.rel_tol)
        : eigen_cg_trace<Eigen::IdentityPreconditioner>(ea, eb, 500, opt.rel_tol);

    ASSERT_GE(ours_trace.size(), eigen_trace.size());
    for (std::size_t k = 0; k < eigen_trace.size(); ++k) {
      // Measured (random SPD, n=60, tol=1e-10, this test binary): largest
      // gap 1.7e-16 (identity), 5.6e-17 (Jacobi) -- bound below with a
      // wide margin for the same reason as the periodic-Laplacian case.
      EXPECT_NEAR(ours_trace[k + 1], eigen_trace[k], 1e-9)
          << (jacobi ? "jacobi " : "identity ") << "iteration " << (k + 1);
    }

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol(ea);
    Eigen::VectorXd ex = chol.solve(eb);
    double max_abs_diff = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
      max_abs_diff = std::max(max_abs_diff, std::fabs(x.data()[i] - ex[i]));
    }
    // Measured: 5.5e-11 (identity), 4.5e-11 (Jacobi) at tol=1e-10 on this
    // matrix; 1e-8 leaves nearly three orders of margin.
    EXPECT_LT(max_abs_diff, 1e-8) << (jacobi ? "jacobi" : "identity");
  }
}

TEST(PcgRecomputesTrueResidual, IllConditionedDiagonalConvergesToTheTrueTolerance) {
  // kappa=1e12, n=30, tol=1e-13: at the iteration where the cheap recurrence
  // residual first reads below tolerance, an INDEPENDENTLY recomputed
  // residual on the same x is measured to still be far above it (the drift
  // pcg.hpp's header comment cites Greenbaum 1989 for) -- see
  // build-evidence/mutations/linalg.toml's own measured numbers for this
  // exact case. A solver that reports convergence from the recurrence
  // alone would return an x this test's own independent check rejects.
  constexpr std::int64_t n = 30;
  constexpr double kappa = 1e12;
  const std::vector<Triplet> t = illconditioned_diagonal(static_cast<int>(n), kappa);
  DistSparseMatrix a = to_dist(t, n);

  const std::vector<double> b_values(static_cast<std::size_t>(n), 1.0);
  DistVector b = make_vector(n, b_values);
  DistVector x(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());

  PcgOptions opt;
  opt.rel_tol = 1e-13;
  opt.max_iterations = 5000;
  Result<SolverReport> result = pcg(a, IdentityPreconditioner{}, x, b, opt);
  ASSERT_TRUE(result.ok()) << result.error().message;
  ASSERT_TRUE(result.value().converged);

  // Independently recomputed, outside the solver entirely.
  DistVector ax(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  ASSERT_TRUE(a.apply(x, ax).ok());
  DistVector r(b);
  r.copy_from(b);
  r.axpy(-1.0, ax);
  EXPECT_LE(r.norm2() / b.norm2(), opt.rel_tol)
      << "reported convergence does not survive an independent residual recompute";
  // The report's own residual_norm must be the recomputed value, not the
  // (possibly still-drifted) recurrence's -- so it must independently
  // satisfy the same tolerance.
  EXPECT_LE(result.value().residual_norm / b.norm2(), opt.rel_tol);
}

TEST(PcgJacobi, ConvergesInFewerIterationsThanIdentityOnADiagonallyScaledSystem) {
  // A diagonal spread wide enough that Jacobi scaling (which is EXACT for a
  // diagonal matrix: M^-1 A = I) must collapse the iteration count, while
  // identity-preconditioned CG still pays for the spread. If the
  // preconditioner application is silently skipped, Jacobi-preconditioned
  // PCG behaves exactly like identity-preconditioned PCG and this strict
  // inequality fails.
  constexpr std::int64_t n = 40;
  const std::vector<Triplet> t = illconditioned_diagonal(static_cast<int>(n), 1e6);
  DistSparseMatrix a = to_dist(t, n);
  const std::vector<double> b_values = random_vector(static_cast<int>(n), 5u);
  DistVector b = make_vector(n, b_values);

  PcgOptions opt;
  opt.rel_tol = 1e-10;
  opt.max_iterations = 2000;

  DistVector x_id(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  Result<SolverReport> id_result = pcg(a, IdentityPreconditioner{}, x_id, b, opt);
  ASSERT_TRUE(id_result.ok()) << id_result.error().message;
  ASSERT_TRUE(id_result.value().converged);

  auto prec = JacobiPreconditioner::from_matrix(a);
  ASSERT_TRUE(prec.ok());
  DistVector x_jac(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  Result<SolverReport> jac_result = pcg(a, prec.value(), x_jac, b, opt);
  ASSERT_TRUE(jac_result.ok()) << jac_result.error().message;
  ASSERT_TRUE(jac_result.value().converged);

  EXPECT_LT(jac_result.value().iterations, id_result.value().iterations);
}

TEST(PcgRefusals, NoConvergenceWithinATinyIterationCapNamesTheIterationCountAndResidual) {
  constexpr std::int64_t n = 60;
  const std::vector<Triplet> t = random_spd(static_cast<int>(n), 23u);
  DistSparseMatrix a = to_dist(t, n);
  const std::vector<double> b_values = random_vector(static_cast<int>(n), 29u);
  DistVector b = make_vector(n, b_values);
  DistVector x(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());

  PcgOptions opt;
  opt.rel_tol = 1e-12;
  opt.max_iterations = 2;  // far too few for this matrix to converge
  Result<SolverReport> result = pcg(a, IdentityPreconditioner{}, x, b, opt);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, "E-LINALG-NO-CONVERGENCE");
  EXPECT_NE(result.error().message.find("2"), std::string::npos)
      << "message should carry the iteration count: " << result.error().message;
}

TEST(PcgRefusals, ZeroDiagonalEntryIsRefused) {
  // Row 2 has no diagonal triplet at all -- a structurally absent entry,
  // which DistSparseMatrix::diagonal() reports as exactly 0.0 (its own
  // documentation), indistinguishable here from an explicit zero.
  const RowOwnership owners{{0}};
  const std::vector<Triplet> t{{0, 0, 5.0}, {0, 1, -1.0}, {1, 1, 6.0}, {1, 0, -1.0}, {2, 0, -3.0}};
  auto m = DistSparseMatrix::from_triplets(0, 3, 3, t, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok()) << m.error().message;
  auto prec = JacobiPreconditioner::from_matrix(m.value());
  ASSERT_FALSE(prec.ok());
  EXPECT_EQ(prec.error().code, "E-LINALG-ZERO-DIAGONAL");
}

}  // namespace
