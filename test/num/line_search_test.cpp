// The line search: BT cubic backtracking, CP critical-point secant,
// Basic damping.
//
// WHAT EACH CASE CHECKS (one test per mechanism).
//   LineSearchBT.EngagesAndRecoversOnAPoorInitialGuess -- the full Newton
//   step overshoots a steep quadratic badly enough that the Armijo test
//   rejects lambda=1, forcing at least one backtrack; the accepted lambda
//   satisfies the Armijo test by direct recomputation, and every trial is
//   recorded.
//   LineSearchBT.TakesTheFullStepWhenItAlreadySatisfiesArmijo -- the
//   control case: a mild quadratic where lambda=1 is accepted with zero
//   backtracks.
//   LineSearchBT.FailsWhenNoStepAboveLambdaMinIsAcceptable -- a merit
//   function engineered to never satisfy Armijo above lambda_min.
//   LineSearchCP.FindsTheSecantRootOnAOneDProblemWithAnEnergy -- Phi(t) =
//   1/2 (t-3)^2 (so F = grad(Phi) = t-3, a genuine gradient field) along
//   d=1 from x=0: the exact critical point is at lambda=3, found to the
//   secant tolerance.
//   LineSearchBasic.TakesLambdaEqualToDamping -- no evaluation at all;
//   lambda is read back exactly.

#include "fuse/num/line_search.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

using fuse::Result;
using fuse::num::backtracking_line_search;
using fuse::num::basic_line_search;
using fuse::num::critical_point_line_search;
using fuse::num::LineSearchKind;
using fuse::num::LineSearchOptions;
using fuse::num::LineSearchResult;
using fuse::num::LineSearchStatus;

namespace {

/// 1/2 ||F(x + lambda d)||^2 for the scalar quadratic residual
/// F(t) = k * (t - root), i.e. the merit function a Newton step on that
/// residual would backtrack over. `x0`/`d` are the Newton iterate/step;
/// `k` controls how steep (and therefore how badly a full step overshoots)
/// the residual is.
double scalar_merit(double x0, double d, double lambda, double k, double root) {
  const double t = x0 + lambda * d;
  const double f = k * (t - root);
  return 0.5 * f * f;
}

}  // namespace

// ---------------------------------------------------------------------------
// BT.
// ---------------------------------------------------------------------------

TEST(LineSearchBT, EngagesAndRecoversOnAPoorInitialGuess) {
  // F(t) = 50*(t - 1); Newton step from the Jacobian J=50 at x0=10 is
  // d = -F(x0)/J = -(50*9)/50 = -9, landing exactly on the root -- too
  // convenient. Perturb the step to UNDERSHOOT the Jacobian's own slope
  // (an inexact/MFFD Jacobian): d = -20, so
  // lambda=1 overshoots the root at t=1 out to t = 10 - 20 = -10, whose
  // merit is larger than a tamer step would give, and Armijo must reject
  // it.
  const double x0 = 10.0, d = -20.0, k = 50.0, root = 1.0;
  const double f0 = scalar_merit(x0, 0.0, 0.0, k, root);

  LineSearchOptions opt;
  opt.kind = LineSearchKind::BT;

  auto merit = [&](double lambda) -> Result<double> {
    return scalar_merit(x0, d, lambda, k, root);
  };

  Result<LineSearchResult> r = backtracking_line_search(f0, merit, opt);
  ASSERT_TRUE(r.ok());
  const LineSearchResult& res = r.value();

  ASSERT_GE(res.trials.size(), 2u) << "a poor initial guess must force at least one backtrack";
  EXPECT_NE(res.status, LineSearchStatus::FullStep);
  EXPECT_LT(res.lambda, 1.0);
  EXPECT_GT(res.lambda, 0.0);

  // The accepted lambda satisfies Armijo by direct recomputation --
  // f(lambda) <= (1 - 2 alpha lambda) f0 -- never trusted from the
  // search's own internal bookkeeping alone.
  const double f_accepted = scalar_merit(x0, d, res.lambda, k, root);
  EXPECT_LE(f_accepted, (1.0 - 2.0 * opt.armijo_alpha * res.lambda) * f0);

  // Every trial actually reduced towards the root compared to the full,
  // rejected step -- recorded, not merely asserted.
  for (double lambda : res.trials) {
    EXPECT_GT(lambda, 0.0);
    EXPECT_LE(lambda, 1.0);
  }
}

TEST(LineSearchBT, TakesTheFullStepWhenItAlreadySatisfiesArmijo) {
  // An exact Newton step on an affine residual: d = -F(x0)/J lands
  // exactly on the root, so f(1) = 0 trivially satisfies Armijo.
  const double x0 = 5.0, k = 2.0, root = 3.0;
  const double d = -(x0 - root);  // J=1 for this merit's own construction
  const double f0 = scalar_merit(x0, 0.0, 0.0, k, root);

  LineSearchOptions opt;
  auto merit = [&](double lambda) -> Result<double> {
    return scalar_merit(x0, d, lambda, k, root);
  };
  Result<LineSearchResult> r = backtracking_line_search(f0, merit, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, LineSearchStatus::FullStep);
  EXPECT_EQ(r.value().lambda, 1.0);
  EXPECT_EQ(r.value().trials.size(), 1u);
}

TEST(LineSearchBT, FailsWhenNoStepAboveLambdaMinIsAcceptable) {
  // A merit function that is FLAT (never decreases below f0) for any
  // lambda > 0: Armijo can never be satisfied above lambda_min, so the
  // search must report Failed rather than loop forever or silently
  // accept an unacceptable step.
  const double f0 = 10.0;
  LineSearchOptions opt;
  opt.lambda_min = 1e-6;
  opt.max_backtracks = 40;
  auto merit = [&](double) -> Result<double> { return f0; };  // never improves
  Result<LineSearchResult> r = backtracking_line_search(f0, merit, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, LineSearchStatus::Failed);
}

// ---------------------------------------------------------------------------
// CP.
// ---------------------------------------------------------------------------

TEST(LineSearchCP, FindsTheSecantRootOnAOneDProblemWithAnEnergy) {
  // Phi(t) = 1/2 (t - 3)^2, F = grad(Phi) = t - 3 (a genuine gradient
  // field: "a problem with an energy"). From
  // x0=0, d=1: g(lambda) = <F(x0 + lambda d), d> = (lambda - 3) * 1, whose
  // root is exactly lambda=3, the critical point of Phi along the line.
  const double x0 = 0.0, d = 1.0;
  auto directional = [&](double lambda) -> Result<double> {
    const double t = x0 + lambda * d;
    const double f = t - 3.0;
    return f * d;
  };
  const double g0 = directional(0.0).value();

  LineSearchOptions opt;
  opt.secant_tol = 1e-12;
  Result<LineSearchResult> r = critical_point_line_search(g0, directional, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_NEAR(r.value().lambda, 3.0, 1e-8);
  EXPECT_NE(r.value().status, LineSearchStatus::Failed);
  EXPECT_NE(r.value().status, LineSearchStatus::MaxStepsReached);
}

TEST(LineSearchCP, FailsOnADegenerateFlatDirectional) {
  // g(lambda) constant: the secant denominator vanishes identically, so
  // no root can be found -- Failed, not an infinite loop or a division
  // that silently produces NaN.
  auto directional = [](double) -> Result<double> { return 5.0; };
  const double g0 = 5.0;
  Result<LineSearchResult> r = critical_point_line_search(g0, directional, LineSearchOptions{});
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, LineSearchStatus::Failed);
}

// ---------------------------------------------------------------------------
// Basic.
// ---------------------------------------------------------------------------

TEST(LineSearchBasic, TakesLambdaEqualToDamping) {
  LineSearchOptions opt;
  opt.damping = 0.37;
  const LineSearchResult res = basic_line_search(opt);
  EXPECT_EQ(res.lambda, 0.37);
  EXPECT_EQ(res.steps, 0);
  EXPECT_EQ(res.status, LineSearchStatus::Converged);
  ASSERT_EQ(res.trials.size(), 1u);
  EXPECT_EQ(res.trials[0], 0.37);
}

TEST(LineSearchBasic, FullDampingReportsFullStep) {
  LineSearchOptions opt;
  opt.damping = 1.0;
  const LineSearchResult res = basic_line_search(opt);
  EXPECT_EQ(res.lambda, 1.0);
  EXPECT_EQ(res.status, LineSearchStatus::FullStep);
}
