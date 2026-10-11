// Verifies `fuse_examples::SparseLuBackend` (examples/common/
// sparse_solve_backend.hpp), the one sparse direct solver both FSI-channel's
// `StokesSolver` and Gray-Scott's `MonolithicGrayScott` hold: with
// FUSE_HAVE_CUSPARSE off it IS `Eigen::SparseLU` (same results bit for bit);
// with it on, the numeric refactorisations and solves run in cusolverRf on
// the device, seeded by one cusolverSp host LU. The two call sites only ever
// use analyzePattern/factorize/compute/solve/info, so that is the whole
// contract checked here -- on a nonsymmetric 1D convection-diffusion
// operator (a real pivoting case, not an SPD one) and on a saddle-point
// block (zero diagonal block, the Stokes shape), the two shapes the examples
// actually hand it.

#include <gtest/gtest.h>

#include <Eigen/Sparse>

#include <cmath>
#include <cstdio>
#include <string_view>
#include <vector>

#include "fuse/base/config.hpp"
#include "sparse_solve_backend.hpp"
#include "stokes_solver.hpp"

namespace {

using fuse_examples::SparseLuBackend;
using SpMat = Eigen::SparseMatrix<double>;

// -u'' + a u' on n interior points, Dirichlet ends, h = 1/(n+1). Nonsymmetric
// for a != 0; well conditioned for the sizes used here.
SpMat convection_diffusion(int n, double a) {
  const double h = 1.0 / (n + 1);
  std::vector<Eigen::Triplet<double>> t;
  t.reserve(3 * static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    t.emplace_back(i, i, 2.0 / (h * h));
    if (i > 0)     t.emplace_back(i, i - 1, -1.0 / (h * h) - a / (2 * h));
    if (i < n - 1) t.emplace_back(i, i + 1, -1.0 / (h * h) + a / (2 * h));
  }
  SpMat A(n, n);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

// [K  B^T; B  0] with K = convection_diffusion(n, a) and B a one-row sum
// constraint: the Stokes saddle-point shape, zero diagonal in the last row.
SpMat saddle_point(int n, double a) {
  const SpMat K = convection_diffusion(n, a);
  std::vector<Eigen::Triplet<double>> t;
  for (int k = 0; k < K.outerSize(); ++k) {
    for (SpMat::InnerIterator it(K, k); it; ++it) {
      t.emplace_back(static_cast<int>(it.row()), static_cast<int>(it.col()), it.value());
    }
  }
  for (int i = 0; i < n; ++i) {
    t.emplace_back(n, i, 1.0);
    t.emplace_back(i, n, 1.0);
  }
  SpMat A(n + 1, n + 1);
  A.setFromTriplets(t.begin(), t.end());
  A.makeCompressed();
  return A;
}

Eigen::VectorXd ramp(int n) {
  Eigen::VectorXd b(n);
  for (int i = 0; i < n; ++i) b[i] = 1.0 + 0.5 * i;
  return b;
}

double relative_residual(const SpMat& A, const Eigen::VectorXd& x,
                         const Eigen::VectorXd& b) {
  return (A * x - b).norm() / b.norm();
}

constexpr double kTol = 1e-11;

TEST(SparseSolveBackend, NamesTheBackendItWasBuiltWith) {
  const std::string_view name = SparseLuBackend::name();
  if (FUSE_HAVE_CUSPARSE) {
    EXPECT_EQ(name, "cusolverRf");
  } else {
    EXPECT_EQ(name, "Eigen::SparseLU");
  }
}

TEST(SparseSolveBackend, SolvesANonsymmetricSystemToRoundoff) {
  const int n = 200;
  const SpMat A = convection_diffusion(n, 30.0);
  const Eigen::VectorXd b = ramp(n);

  SparseLuBackend lu;
  lu.analyzePattern(A);
  lu.factorize(A);
  ASSERT_EQ(lu.info(), Eigen::Success);
  const Eigen::VectorXd x = lu.solve(b);
  ASSERT_EQ(lu.info(), Eigen::Success);
  EXPECT_LT(relative_residual(A, x, b), kTol);
}

TEST(SparseSolveBackend, ComputeIsAnalyzePatternThenFactorize) {
  const int n = 150;
  const SpMat A = convection_diffusion(n, -12.0);
  const Eigen::VectorXd b = ramp(n);

  SparseLuBackend lu;
  lu.compute(A);
  ASSERT_EQ(lu.info(), Eigen::Success);
  const Eigen::VectorXd x = lu.solve(b);
  EXPECT_LT(relative_residual(A, x, b), kTol);
}

// The cache case both examples rely on: one analyzePattern, then factorize()
// repeatedly with new values on the same pattern (the Newton Jacobian, the
// dt-dependent CN operator). Each refactorisation must solve ITS matrix.
TEST(SparseSolveBackend, RefactorizesNewValuesOnTheSamePattern) {
  const int n = 180;
  const SpMat A1 = convection_diffusion(n, 5.0);
  const SpMat A2 = convection_diffusion(n, 40.0);
  const SpMat A3 = convection_diffusion(n, -25.0);
  const Eigen::VectorXd b = ramp(n);

  SparseLuBackend lu;
  lu.analyzePattern(A1);
  for (const SpMat* A : {&A1, &A2, &A3}) {
    lu.factorize(*A);
    ASSERT_EQ(lu.info(), Eigen::Success);
    const Eigen::VectorXd x = lu.solve(b);
    ASSERT_EQ(lu.info(), Eigen::Success);
    EXPECT_LT(relative_residual(*A, x, b), kTol);
  }
  // And a solve against A3's factors must NOT satisfy A1 -- i.e. the
  // refactorisation really replaced the numeric values.
  const Eigen::VectorXd x3 = lu.solve(b);
  EXPECT_GT(relative_residual(A1, x3, b), 1e-3);
}

TEST(SparseSolveBackend, SolvesASaddlePointSystemWithAZeroDiagonalBlock) {
  const int n = 120;
  const SpMat A = saddle_point(n, 8.0);
  const Eigen::VectorXd b = ramp(n + 1);

  SparseLuBackend lu;
  lu.compute(A);
  ASSERT_EQ(lu.info(), Eigen::Success);
  const Eigen::VectorXd x = lu.solve(b);
  ASSERT_EQ(lu.info(), Eigen::Success);
  EXPECT_LT(relative_residual(A, x, b), kTol);
}

// THE Stokes saddle point fsi_transient assembles (nx=10, ny=6, L=1, H=0.2,
// mu=0.01: 196 unknowns, 971 nonzeros), copied out of cn_step() through its
// debug_matrix_out hook. The synthetic saddle_point() above is NOT a stand-in
// here: cusolverRf solves it reproducibly, and only this matrix showed the
// jitter that stalled the coupling tests.
SpMat real_stokes_matrix(std::size_t nx = 10, std::size_t ny = 6) {
  const fsi::ChannelParams cp{.nx = nx, .ny = ny, .L = 1.0, .H = 0.2, .mu = 0.01, .rho = 1.0};
  fsi::ChannelBCs bcs;
  bcs.left.u = [](double y, double) { return 4.0 * y * (0.2 - y) / (0.2 * 0.2); };
  bcs.right.outflow = true;
  const fsi::SourceTerms src;
  std::vector<double> u0((cp.nx + 1) * cp.ny, 0.0), v0(cp.nx * (cp.ny + 1), 0.0);
  const fsi::PressurePin pin{cp.nx - 1, 0, 0.0};
  SpMat A;
  (void)fsi::cn_step(cp, bcs, src, /*dt=*/0.01, 0.0, 0.01, u0, v0, pin, &A);
  A.makeCompressed();
  return A;
}

// The coupling iterations both examples drive (IQN-ILS fixed point on the
// interface traction; the energy diagnostic's finite-difference linearisation)
// converge through a DETERMINISTIC rounding floor and stall on one that
// jitters. On the real Stokes matrix cusolverRfSolve alone is not bitwise
// reproducible (~4% of identical-input repeats differ, by ~kappa*eps); the
// backend's refinement step is what suppresses that. 300 repeats, agreement
// to working precision -- this test fails without the refinement step.
TEST(SparseSolveBackend, RepeatedSolvesOfTheRealStokesSystemAgreeToWorkingPrecision) {
  const SpMat A = real_stokes_matrix();
  ASSERT_EQ(A.rows(), 196);
  const Eigen::VectorXd b = ramp(static_cast<int>(A.rows()));

  SparseLuBackend lu;
  lu.compute(A);
  ASSERT_EQ(lu.info(), Eigen::Success);
  const Eigen::VectorXd x0 = lu.solve(b);
  EXPECT_LT(relative_residual(A, x0, b), kTol);
  const double scale = x0.norm();
  double worst = 0.0;
  int differing_solves = 0;
  for (int rep = 0; rep < 300; ++rep) {
    const Eigen::VectorXd x = lu.solve(b);
    const double d = (x - x0).norm() / scale;
    if (d > 0.0) ++differing_solves;
    worst = std::max(worst, d);
  }
  EXPECT_LE(worst, 1e-15) << differing_solves << " of 300 repeat solves differed; worst "
                          << worst << " relative";
}

// Same property with a DIFFERENT right-hand side solved in between each
// repeat -- the pattern every coupling iteration has (a new rhs every sweep).
// The back-to-back version above passed even with the refinement step
// removed; the fsi_transient diagnostic saw ~4% of repeats differ with other
// solves interleaved. This is the case that separates "the solve is random"
// from "the solve depends on state left behind by the previous solve".
TEST(SparseSolveBackend, RepeatedSolvesAgreeWhenOtherSolvesAreInterleaved) {
  const SpMat A = real_stokes_matrix();
  const int n = static_cast<int>(A.rows());
  const Eigen::VectorXd b = ramp(n);
  Eigen::VectorXd b2(n);
  for (int i = 0; i < n; ++i) b2[i] = std::sin(0.37 * i) - 0.2 * i;

  SparseLuBackend lu;
  lu.compute(A);
  ASSERT_EQ(lu.info(), Eigen::Success);
  const Eigen::VectorXd x0 = lu.solve(b);
  const double scale = x0.norm();
  double worst = 0.0;
  int differing_solves = 0;
  for (int rep = 0; rep < 300; ++rep) {
    (void)lu.solve(b2);
    const Eigen::VectorXd x = lu.solve(b);
    const double d = (x - x0).norm() / scale;
    if (d > 0.0) ++differing_solves;
    worst = std::max(worst, d);
  }
  EXPECT_LE(worst, 1e-15) << differing_solves << " of 300 interleaved repeat solves differed; worst "
                          << worst << " relative";
}

// Forward error against a long-double Eigen reference on the real Stokes
// system, for the backend AND for double Eigen::SparseLU -- the two numbers
// the energy-diagnostic budget (finite differences of this solve) actually
// depends on. The backend must not be materially worse than Eigen is.
TEST(SparseSolveBackend, ForwardErrorOnTheRealStokesSystemIsNoWorseThanEigens) {
  const SpMat A = real_stokes_matrix();
  const int n = static_cast<int>(A.rows());
  const Eigen::VectorXd b = ramp(n);

  using LdMat = Eigen::SparseMatrix<long double>;
  const LdMat Ald = A.cast<long double>();
  Eigen::SparseLU<LdMat> ref;
  ref.compute(Ald);
  ASSERT_EQ(ref.info(), Eigen::Success);
  const Eigen::Matrix<long double, Eigen::Dynamic, 1> bld = b.cast<long double>();
  const Eigen::Matrix<long double, Eigen::Dynamic, 1> xref = ref.solve(bld);
  const Eigen::VectorXd xr = xref.cast<double>();

  Eigen::SparseLU<SpMat> eig;
  eig.compute(A);
  const Eigen::VectorXd xe = eig.solve(b);
  SparseLuBackend lu;
  lu.compute(A);
  ASSERT_EQ(lu.info(), Eigen::Success);
  const Eigen::VectorXd xb = lu.solve(b);

  const double fe_eigen = (xe - xr).norm() / xr.norm();
  const double fe_backend = (xb - xr).norm() / xr.norm();
  std::printf("forward error vs long-double reference: Eigen::SparseLU %.3e, %s %.3e\n",
              fe_eigen, SparseLuBackend::name().data(), fe_backend);
  EXPECT_LE(fe_backend, 10.0 * std::max(fe_eigen, 1e-15))
      << "backend forward error " << fe_backend << " vs Eigen " << fe_eigen;
}

// example.fsi_channel's own fluid grid (24x12: 900 unknowns), the size at
// which the first flag-on run of the example produced a non-finite value
// after 19 periods while every 196-unknown case above was clean. Finite,
// accurate against the long-double reference, and reproducible, or the
// backend is not fit for the example whatever the small cases say.
TEST(SparseSolveBackend, SolvesTheExampleSizedStokesSystemAccuratelyAndReproducibly) {
  const SpMat A = real_stokes_matrix(24, 12);
  const int n = static_cast<int>(A.rows());
  ASSERT_EQ(n, 900);
  const Eigen::VectorXd b = ramp(n);

  using LdMat = Eigen::SparseMatrix<long double>;
  const LdMat Ald = A.cast<long double>();
  Eigen::SparseLU<LdMat> ref;
  ref.compute(Ald);
  ASSERT_EQ(ref.info(), Eigen::Success);
  const Eigen::Matrix<long double, Eigen::Dynamic, 1> bld = b.cast<long double>();
  const Eigen::VectorXd xr = Eigen::Matrix<long double, Eigen::Dynamic, 1>(ref.solve(bld)).cast<double>();

  Eigen::SparseLU<SpMat> eig;
  eig.compute(A);
  const double fe_eigen = (Eigen::VectorXd(eig.solve(b)) - xr).norm() / xr.norm();

  SparseLuBackend lu;
  lu.compute(A);
  ASSERT_EQ(lu.info(), Eigen::Success);
  const Eigen::VectorXd x0 = lu.solve(b);
  ASSERT_EQ(lu.info(), Eigen::Success);
  ASSERT_TRUE(x0.allFinite());
  const double fe_backend = (x0 - xr).norm() / xr.norm();
  std::printf("n=900 forward error vs long-double reference: Eigen::SparseLU %.3e, %s %.3e\n",
              fe_eigen, SparseLuBackend::name().data(), fe_backend);
  EXPECT_LE(fe_backend, 10.0 * std::max(fe_eigen, 1e-15));

  double worst = 0.0;
  for (int rep = 0; rep < 100; ++rep) {
    const Eigen::VectorXd x = lu.solve(b);
    ASSERT_TRUE(x.allFinite()) << "rep " << rep;
    worst = std::max(worst, (x - x0).norm() / x0.norm());
  }
  EXPECT_LE(worst, 1e-15) << "repeat solves jitter by " << worst;
}

// compute() a SECOND time on the same object -- what StokesSolver's LU cache
// does whenever dt changes (a rolled-back window re-run at a new dt), and
// what example.fsi_channel hit at its step 7 before diverging. Every case
// above seeds a fresh object once; this one re-seeds an already-seeded one,
// on a new pattern (different grid) and on the same pattern (different dt).
TEST(SparseSolveBackend, RecomputingOnOneObjectAfterASeedStaysCorrect) {
  SparseLuBackend lu;
  const SpMat A1 = real_stokes_matrix(10, 6);
  const SpMat A2 = real_stokes_matrix(24, 12);       // new pattern, new n
  const SpMat A3 = SpMat(1.75 * real_stokes_matrix(24, 12));  // same pattern, new values
  const std::vector<const SpMat*> sequence{&A1, &A2, &A3, &A1};
  for (const SpMat* A : sequence) {
    const Eigen::VectorXd b = ramp(static_cast<int>(A->rows()));
    lu.compute(*A);
    ASSERT_EQ(lu.info(), Eigen::Success) << "n=" << A->rows();
    const Eigen::VectorXd x = lu.solve(b);
    ASSERT_EQ(lu.info(), Eigen::Success) << "n=" << A->rows();
    ASSERT_TRUE(x.allFinite()) << "n=" << A->rows();
    EXPECT_LT(relative_residual(*A, x, b), kTol) << "n=" << A->rows();
  }
}

TEST(SparseSolveBackend, ReportsAStructurallySingularMatrixThroughInfo) {
  // Row 3 is entirely zero: no pivot can exist.
  SpMat A = convection_diffusion(10, 0.0);
  for (int k = 0; k < A.outerSize(); ++k) {
    for (SpMat::InnerIterator it(A, k); it; ++it) {
      if (it.row() == 3) it.valueRef() = 0.0;
    }
  }
  A.prune(0.0);
  A.makeCompressed();

  SparseLuBackend lu;
  lu.compute(A);
  EXPECT_NE(lu.info(), Eigen::Success);
}

// With the flag off the backend must be Eigen::SparseLU itself -- not a
// reimplementation that happens to agree to tolerance. Bitwise against a
// directly-constructed Eigen::SparseLU, so the flag-off build of the two
// examples is provably unchanged by the swap.
TEST(SparseSolveBackend, MatchesEigenSparseLuBitwiseWhenBuiltWithoutCusparse) {
  if (FUSE_HAVE_CUSPARSE) {
    GTEST_SKIP() << "built with cuSPARSE: cusolverRf is not bitwise Eigen";
  }
  const int n = 160;
  const SpMat A = convection_diffusion(n, 17.0);
  const Eigen::VectorXd b = ramp(n);

  Eigen::SparseLU<SpMat> ref;
  ref.compute(A);
  const Eigen::VectorXd xr = ref.solve(b);

  SparseLuBackend lu;
  lu.compute(A);
  const Eigen::VectorXd x = lu.solve(b);
  ASSERT_EQ(x.size(), xr.size());
  for (int i = 0; i < n; ++i) EXPECT_EQ(x[i], xr[i]) << "i=" << i;
}

}  // namespace
