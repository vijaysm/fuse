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
#include <string_view>
#include <vector>

#include "fuse/base/config.hpp"
#include "sparse_solve_backend.hpp"

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
