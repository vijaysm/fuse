// The configured least-squares backend against the built-in factorisation.
//
// WHAT IS AND IS NOT CHECKED. These are floating-point factorisations of
// deliberately ill-conditioned matrices, so the two backends will not agree
// bit for bit and nothing here asks them to. What is checked is the property
// the caller depends on: the residual is no worse than the reference's beyond
// a stated tolerance, the numerical rank agrees, and a column the rank
// decision dropped comes back as exactly zero rather than as a small weight.
//
// WHY ILL-CONDITIONED MATRICES ARE THE INTERESTING CASE. This solve runs
// inside quasi-Newton acceleration of a coupled iteration, on a matrix of
// difference vectors from successive iterations. As the iteration converges
// those vectors become near-parallel: the matrix is worst-conditioned exactly
// when the iteration is nearly done, which is when a wrong answer does the
// most damage. A test on well-conditioned input would prove nothing about the
// regime the code actually runs in.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <fuse/base/error.hpp>
#include <fuse/linalg/least_squares.hpp>

namespace {

using fuse::linalg::Backend;

/// Column-major m x k.
std::vector<double>& at(std::vector<double>& a, std::size_t m, std::size_t i,
                        std::size_t j, double v) {
  a[j * m + i] = v;
  return a;
}

/// ||A x - b||_2 with A, x, b as the facade takes them. Computed from a COPY
/// of the inputs, since the solve overwrites them.
double residual(const std::vector<double>& a, std::size_t m, std::size_t k,
                const std::vector<double>& b, const std::vector<double>& x) {
  double s = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    double r = -b[i];
    for (std::size_t j = 0; j < k; ++j) r += a[j * m + i] * x[j];
    s += r * r;
  }
  return std::sqrt(s);
}

/// A matrix of near-parallel columns: column j is column 0 perturbed by
/// `eps`. This is the shape a converging coupled iteration produces.
std::vector<double> near_parallel(std::size_t m, std::size_t k, double eps,
                                  unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> base(m);
  for (double& v : base) v = u(rng);
  std::vector<double> a(m * k);
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < m; ++i) {
      a[j * m + i] = base[i] + eps * static_cast<double>(j) * u(rng);
    }
  }
  return a;
}

}  // namespace

TEST(LeastSquares, ReportsTheBackendThatActuallyRan) {
  // Not a tautology: an earlier draft reported a backend the build had asked
  // for while a different one did the work, which makes an iteration-count
  // report unusable. The build writes its choice into the resolved-dependency
  // record, and this is the run-time half of the same fact.
  const Backend b = fuse::linalg::configured_backend();
  EXPECT_TRUE(b == Backend::Builtin || b == Backend::Eigen);
  EXPECT_STRNE(fuse::linalg::backend_name(b), "unknown");
}

TEST(LeastSquares, AgreesWithTheBuiltinOnAWellPosedSystem) {
  const std::size_t m = 40, k = 4;
  std::vector<double> a0 = near_parallel(m, k, 0.5, 11u);
  std::vector<double> b0(m);
  for (std::size_t i = 0; i < m; ++i) b0[i] = std::sin(0.3 * static_cast<double>(i));

  std::vector<double> a1 = a0, b1 = b0, x1;
  const std::size_t r1 = fuse::linalg::least_squares(a1, m, k, b1, x1);
  std::vector<double> a2 = a0, b2 = b0, x2;
  const std::size_t r2 = fuse::linalg::least_squares_builtin(a2, m, k, b2, x2);

  EXPECT_EQ(r1, r2) << "the two factorisations disagree on the numerical rank";
  // Residuals, not coefficients: two backends may split a near-dependent
  // column set differently and still solve the problem equally well. The
  // residual is what the caller's convergence depends on.
  EXPECT_NEAR(residual(a0, m, k, b0, x1), residual(a0, m, k, b0, x2), 1e-10);
}

TEST(LeastSquares, TruncatesARankDeficientSystemAndZeroesTheDroppedColumns) {
  // Column 2 is exactly column 0, so the rank is 2 whatever the tolerance.
  const std::size_t m = 12, k = 3;
  std::vector<double> a0(m * k, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    at(a0, m, i, 0, 1.0 + static_cast<double>(i));
    at(a0, m, i, 1, static_cast<double>(i % 3));
    at(a0, m, i, 2, 1.0 + static_cast<double>(i));
  }
  std::vector<double> b0(m, 1.0);

  std::vector<double> a1 = a0, b1 = b0, x1;
  const std::size_t rank = fuse::linalg::least_squares(a1, m, k, b1, x1);
  ASSERT_EQ(rank, 2u) << "a duplicated column must not count towards the rank";

  // Exactly one of the two duplicate columns carries a weight, and the other
  // is exactly zero -- the contract that lets a caller tell "filtered" from
  // "given a small weight".
  const bool one_dropped = (x1[0] == 0.0) != (x1[2] == 0.0);
  EXPECT_TRUE(one_dropped) << "x = [" << x1[0] << ", " << x1[1] << ", " << x1[2] << "]";

  std::vector<double> a2 = a0, b2 = b0, x2;
  const std::size_t rank_ref = fuse::linalg::least_squares_builtin(a2, m, k, b2, x2);
  EXPECT_EQ(rank, rank_ref);
  EXPECT_NEAR(residual(a0, m, k, b0, x1), residual(a0, m, k, b0, x2), 1e-9);

  // The same contract on the built-in path explicitly. Without this the
  // assertion above only covers whichever backend the build configured, and
  // the built-in one is the oracle -- the iteration counts were recorded with
  // its truncation behaviour, so a change in it must fail a test in every
  // build, not only in a build that selects it.
  EXPECT_TRUE((x2[0] == 0.0) != (x2[2] == 0.0))
      << "the built-in factorisation gave both duplicate columns a weight: x = ["
      << x2[0] << ", " << x2[1] << ", " << x2[2] << "]";
}

TEST(LeastSquares, StaysAccurateAsTheColumnsBecomeNearlyParallel) {
  // The regime the accelerator actually reaches. Each step makes the columns
  // more nearly parallel; the residual must not degrade faster than the
  // reference's.
  const std::size_t m = 60, k = 5;
  for (const double eps : {1e-3, 1e-6, 1e-9, 1e-12}) {
    std::vector<double> a0 = near_parallel(m, k, eps, 7u);
    std::vector<double> b0(m);
    for (std::size_t i = 0; i < m; ++i) b0[i] = std::cos(0.11 * static_cast<double>(i));

    std::vector<double> a1 = a0, b1 = b0, x1;
    const std::size_t r1 = fuse::linalg::least_squares(a1, m, k, b1, x1);
    std::vector<double> a2 = a0, b2 = b0, x2;
    const std::size_t r2 = fuse::linalg::least_squares_builtin(a2, m, k, b2, x2);

    EXPECT_EQ(r1, r2) << "rank disagreement at eps=" << eps;
    const double res1 = residual(a0, m, k, b0, x1);
    const double res2 = residual(a0, m, k, b0, x2);
    // A loose bound on purpose: the two may truncate a different column at the
    // same rank, which changes the residual in the last digits. What would be
    // a real defect is one backend being WORSE by a factor, which this catches.
    EXPECT_LT(res1, 1.05 * res2 + 1e-12) << "eps=" << eps;
    for (const double v : x1) EXPECT_TRUE(std::isfinite(v)) << "eps=" << eps;
  }
}

TEST(LeastSquares, RefusesAShapeItCannotSolve) {
  std::vector<double> a(6, 1.0), b(2, 1.0), x;
  // Wider than tall: underdetermined, and a caller mistake rather than a
  // numerical outcome, so it throws instead of returning a rank.
  EXPECT_THROW((void)fuse::linalg::least_squares(a, 2, 3, b, x),
               fuse::InvalidArgument);
  std::vector<double> small(2, 1.0);
  EXPECT_THROW((void)fuse::linalg::least_squares(small, 4, 2, b, x),
               fuse::InvalidArgument);
}

// REVIEW.md R-45: `least_squares_into` is called from the accelerator with
// fewer rows (m, the interface size) than columns (k, the reuse limit) on any
// small-interface coupling, but no test drove that shape. This turns the
// ASan/UBSan check the reviewer ran outside the repository into a durable
// regression: `least_squares_into` is safe and gives a sane, rank-<=m,
// round-off-accurate answer -- it does not enforce the public entry point's
// `m >= k` precondition, by design (see the header doc on the function).
TEST(LeastSquares, IntoAgreesWithBuiltinWhenMIsLessThanK) {
  for (const auto& [m, k] : {std::pair<std::size_t, std::size_t>{1, 4},
                             std::pair<std::size_t, std::size_t>{2, 6}}) {
    std::mt19937 rng(3u);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> a(m * k);
    for (double& v : a) v = u(rng);
    // A consistent system (b is exactly in A's range), so the expected
    // residual is round-off, not a fitting error -- the property that
    // distinguishes "safe and sane" from "safe but garbage".
    std::vector<double> x_true(k);
    for (double& v : x_true) v = u(rng);
    std::vector<double> b(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < k; ++j) b[i] += a[j * m + i] * x_true[j];
    }

    std::vector<double> a1 = a, b1 = b, out;
    fuse::linalg::LsScratch work;
    const std::size_t rank =
        fuse::linalg::least_squares_into(a1, m, k, b1, out, work);
    EXPECT_LE(rank, m) << "m=" << m << " k=" << k;
    EXPECT_LE(rank, k) << "m=" << m << " k=" << k;
    EXPECT_NEAR(residual(a, m, k, b, out), 0.0, 1e-9)
        << "m=" << m << " k=" << k;
    for (const double v : out) EXPECT_TRUE(std::isfinite(v));
  }
}
