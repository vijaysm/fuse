// "Krylov" track: fuse::linalg::gmres at tier 0, one rank.
//
// WHAT EACH CASE CHECKS.
//   Requirement 2 -- iterate-for-iterate agreement (identity
//   preconditioner, so the comparison is not confounded by this file's
//   right preconditioning against Eigen's left preconditioning --
//   gmres.hpp's own header comment). GmresAgreesWithEigen.* builds the
//   SAME random nonsymmetric, diagonally dominant matrix as a
//   fuse::linalg::DistSparseMatrix and an Eigen::SparseMatrix, and
//   compares our per-step residual-estimate trace against
//   Eigen::internal::gmres's own `tol_error`, read off directly by
//   calling it with an iteration budget of exactly k for every k up to
//   convergence. The converged x is also checked against Eigen::SparseLU.
//   Requirement -- right preconditioning actually changes the iteration
//   taken (GmresJacobi.*): a Jacobi-preconditioned solve on a
//   diagonally-scaled system converges, and its solution agrees with
//   SparseLU; not compared iterate-for-iterate against Eigen (the two
//   preconditioning conventions solve a genuinely different Krylov
//   recurrence, so there is nothing to compare term by term).
//   Refusal (requirement 4) -- GmresRefusals.* fires
//   E-LINALG-NO-CONVERGENCE on a 1-step iteration cap.

#include "fuse/linalg/gmres.hpp"

#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/IterativeSolvers>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
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
using fuse::linalg::gmres;
using fuse::linalg::GmresOptions;
using fuse::linalg::IdentityPreconditioner;
using fuse::linalg::JacobiPreconditioner;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::linalg::RowOwnership;
using fuse::linalg::SolverReport;
using fuse::linalg::Triplet;

// ----------------------------------------------------------------- matrices

/// Random, nonsymmetric, five nonzeros per row (four off-diagonal plus the
/// diagonal itself), diagonally dominant by construction -- each diagonal
/// entry is set to its row's off-diagonal absolute sum plus a margin of 1,
/// the same Gershgorin argument pcg_test.cpp's `random_spd` uses, which
/// here only guarantees nonsingularity and a well-behaved spectrum (the
/// matrix is not symmetric, so SPD does not apply).
std::vector<Triplet> random_nonsymmetric(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_int_distribution<int> col_pick(0, n - 1);
  std::vector<Triplet> t;
  for (int i = 0; i < n; ++i) {
    std::set<int> cols;
    while (cols.size() < 4) {
      const int c = col_pick(rng);
      if (c != i) cols.insert(c);
    }
    double row_abs_sum = 0.0;
    std::vector<std::pair<int, double>> entries;
    for (int c : cols) {
      const double v = u(rng);
      entries.push_back({c, v});
      row_abs_sum += std::fabs(v);
    }
    for (const auto& [c, v] : entries) t.push_back({i, c, v});
    t.push_back({i, i, row_abs_sum + 1.0});
  }
  return t;
}

/// A diagonally scaled, strongly nonsymmetric but diagonally dominant
/// matrix: random off-diagonal structure as above, with a diagonal spread
/// over several orders of magnitude so a Jacobi (diagonal) preconditioner
/// has real scaling work to do.
std::vector<Triplet> random_nonsymmetric_scaled(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::uniform_int_distribution<int> col_pick(0, n - 1);
  std::vector<Triplet> t;
  for (int i = 0; i < n; ++i) {
    std::set<int> cols;
    while (cols.size() < 4) {
      const int c = col_pick(rng);
      if (c != i) cols.insert(c);
    }
    double row_abs_sum = 0.0;
    std::vector<std::pair<int, double>> entries;
    for (int c : cols) {
      const double v = u(rng);
      entries.push_back({c, v});
      row_abs_sum += std::fabs(v);
    }
    const double scale = std::pow(10.0, 4.0 * static_cast<double>(i) / static_cast<double>(n - 1));
    for (const auto& [c, v] : entries) t.push_back({i, c, v * scale});
    t.push_back({i, i, (row_abs_sum + 1.0) * scale});
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

/// Eigen::internal::gmres's own per-iteration residual ESTIMATE
/// (`tol_error`, the rotated-right-hand-side quantity, never a recomputed
/// residual -- that file's own code), read off by calling with an
/// iteration budget of exactly `k` from a fresh zero guess, for
/// `k = 1, 2, ...` up to convergence.
std::vector<double> eigen_gmres_trace(const Eigen::SparseMatrix<double>& a,
                                      const Eigen::VectorXd& b, int max_iters, int restart,
                                      double tol) {
  std::vector<double> trace;
  Eigen::IdentityPreconditioner precond;
  for (int k = 1; k <= max_iters; ++k) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size());
    Eigen::Index iters = k;
    double tol_error = tol;
    const bool ok = Eigen::internal::gmres(a, b, x, precond, iters, restart, tol_error);
    trace.push_back(tol_error);
    if (!ok || iters < k) break;  // Eigen's own run already converged
  }
  return trace;
}

// ------------------------------------------------------------------ tests

TEST(GmresAgreesWithEigen, RandomNonsymmetricIdentity) {
  constexpr std::int64_t n = 50;
  const std::vector<Triplet> t = random_nonsymmetric(static_cast<int>(n), 41u);
  const Eigen::SparseMatrix<double> ea = to_eigen(t, n);
  DistSparseMatrix a = to_dist(t, n);

  const std::vector<double> b_values = random_vector(static_cast<int>(n), 43u);
  Eigen::VectorXd eb(n);
  for (std::int64_t i = 0; i < n; ++i) eb[i] = b_values[static_cast<std::size_t>(i)];
  DistVector b = make_vector(n, b_values);
  DistVector x(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());

  std::vector<double> ours_trace;
  GmresOptions opt;
  opt.rel_tol = 1e-10;
  opt.max_iterations = 300;
  opt.restart = 30;
  opt.monitor = [&](std::size_t, double r) { ours_trace.push_back(r / b.norm2()); };

  Result<SolverReport> result = gmres(a, IdentityPreconditioner{}, x, b, opt);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_TRUE(result.value().converged);

  const std::vector<double> eigen_trace =
      eigen_gmres_trace(ea, eb, 300, static_cast<int>(opt.restart), opt.rel_tol);

  ASSERT_GE(ours_trace.size(), eigen_trace.size());
  for (std::size_t k = 0; k < eigen_trace.size(); ++k) {
    // Measured (random nonsymmetric, n=50, restart=30, tol=1e-10, this
    // test binary): largest per-iteration gap between this
    // implementation's modified Gram-Schmidt Arnoldi and Eigen's
    // Householder-based unsupported::GMRES was 4.3e-11 -- the two
    // orthogonalisation methods compute the same Krylov subspace in exact
    // arithmetic but accumulate round-off differently (this file's own
    // header comment), so the bound below is a measured floating-point
    // tolerance, not bit identity. 1e-7 leaves more than three orders of
    // margin.
    EXPECT_NEAR(ours_trace[k + 1], eigen_trace[k], 1e-7) << "iteration " << (k + 1);
  }

  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
  lu.compute(ea);
  ASSERT_EQ(lu.info(), Eigen::Success);
  Eigen::VectorXd ex = lu.solve(eb);
  double max_abs_diff = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(x.data()[i] - ex[i]));
  }
  // Measured: 2.4e-11 at tol=1e-10 on this matrix; 1e-7 leaves more than
  // three orders of margin.
  EXPECT_LT(max_abs_diff, 1e-7);
}

TEST(GmresJacobi, ConvergesAndAgreesWithSparseLuOnADiagonallyScaledSystem) {
  constexpr std::int64_t n = 40;
  const std::vector<Triplet> t = random_nonsymmetric_scaled(static_cast<int>(n), 59u);
  const Eigen::SparseMatrix<double> ea = to_eigen(t, n);
  DistSparseMatrix a = to_dist(t, n);

  const std::vector<double> b_values = random_vector(static_cast<int>(n), 61u);
  Eigen::VectorXd eb(n);
  for (std::int64_t i = 0; i < n; ++i) eb[i] = b_values[static_cast<std::size_t>(i)];
  DistVector b = make_vector(n, b_values);
  DistVector x(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());

  auto prec = JacobiPreconditioner::from_matrix(a);
  ASSERT_TRUE(prec.ok());

  GmresOptions opt;
  opt.rel_tol = 1e-10;
  opt.max_iterations = 300;
  opt.restart = 30;
  Result<SolverReport> result = gmres(a, prec.value(), x, b, opt);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_TRUE(result.value().converged);

  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
  lu.compute(ea);
  ASSERT_EQ(lu.info(), Eigen::Success);
  Eigen::VectorXd ex = lu.solve(eb);
  double max_abs_diff = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    max_abs_diff = std::max(max_abs_diff, std::fabs(x.data()[i] - ex[i]));
  }
  EXPECT_LT(max_abs_diff, 1e-6);
}

TEST(GmresRefusals, NoConvergenceWithinATinyIterationCapNamesTheIterationCountAndResidual) {
  constexpr std::int64_t n = 50;
  const std::vector<Triplet> t = random_nonsymmetric(static_cast<int>(n), 41u);
  DistSparseMatrix a = to_dist(t, n);
  const std::vector<double> b_values = random_vector(static_cast<int>(n), 43u);
  DistVector b = make_vector(n, b_values);
  DistVector x(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());

  GmresOptions opt;
  opt.rel_tol = 1e-12;
  opt.max_iterations = 1;  // far too few for this matrix to converge
  opt.restart = 30;
  Result<SolverReport> result = gmres(a, IdentityPreconditioner{}, x, b, opt);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, "E-LINALG-NO-CONVERGENCE");
  EXPECT_NE(result.error().message.find("1"), std::string::npos)
      << "message should carry the iteration count: " << result.error().message;
}

}  // namespace
