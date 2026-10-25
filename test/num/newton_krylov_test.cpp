// fuse::num::solve(), the native Jacobian-free Newton-Krylov coupler.
//
// WHAT EACH SECTION CHECKS, one per mechanism.
//   Mffd.*            -- (1) the MFFD action vs. the analytic Jacobian on
//                        a quadratic F, to O(h) with h as reported, and
//                        fuse::linalg::taylor_remainder_order() confirming
//                        the analytic Jacobian is order 2.
//   EisenstatWalker.*  -- (3) the formula, the safeguard, and the cap/floor,
//                        recomputed from first principles against the
//                        function's own output.
//   NewtonKrylovLineSearch.* -- (4) BT engages and recovers on a poor
//                        initial guess, with the accepted lambdas recorded.
//   NewtonKrylovConvergence.*/NewtonKrylovDivergence.* -- (5) every named
//                        code, each fired by directly constructing its
//                        condition.
//   NewtonKrylovPreconditioner.* -- (6) the lagged block-Jacobi hook: the
//                        exact block inverse converges GMRES in
//                        <= nblocks iterations per Newton step; identity
//                        takes more.
//   NewtonKrylovRankAgreement.* -- the standing rule: two in-process
//                        partitions, reduced through a genuinely
//                        synchronised (thread-paired) Reducer, reach the
//                        SAME NewtonStatus at the SAME iteration -- and a
//                        decision made from an unreduced local quantity
//                        would deadlock this harness rather than merely
//                        disagree, which is exactly the property being
//                        tested.

#include "fuse/num/newton_krylov.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/derivative_gate.hpp"
#include "fuse/linalg/dist_vector.hpp"

using fuse::Failure;
using fuse::Result;
using fuse::linalg::DistVector;
using fuse::linalg::GhostExchanger;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::linalg::Reducer;
using fuse::linalg::taylor_remainder_order;
using fuse::linalg::TaylorRemainderResult;
using fuse::num::BlockJacobiPreconditioner;
using fuse::num::EisenstatWalkerChoice;
using fuse::num::EisenstatWalkerOptions;
using fuse::num::eisenstat_walker_eta;
using fuse::num::is_converged;
using fuse::num::is_diverged;
using fuse::num::LineSearchKind;
using fuse::num::MffdJacobianAction;
using fuse::num::MffdOptions;
using fuse::num::NewtonKrylovOptions;
using fuse::num::NewtonResult;
using fuse::num::NewtonStatus;
using fuse::num::PreconditionerBlock;
using fuse::num::ResidualFn;
using fuse::num::solve;
using fuse::num::to_cstring;

namespace {

DistVector vec_sized(std::size_t n) {
  return DistVector(n, make_serial_exchanger(), make_serial_reducer());
}

DistVector vec_from(const std::vector<double>& v) {
  DistVector out = vec_sized(v.size());
  std::copy(v.begin(), v.end(), out.data());
  return out;
}

std::vector<double> to_std(const DistVector& v) {
  return std::vector<double>(v.data(), v.data() + v.owned_size());
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. MFFD Jacobian action.
// ---------------------------------------------------------------------------

namespace {

/// F(x)_i = x_i^2 - c_i, J(x) = diag(2 x_i). A quadratic, so the Taylor
/// remainder of F itself is EXACT at second order (no higher terms at
/// all): taylor_remainder_order() must report order ~2 with a floor at
/// round-off, and the MFFD action's own error against the analytic J v
/// must equal h * v_i^2 exactly (up to round-off), since
/// F(x+hv)_i - F(x)_i - h J_i v_i = h^2 v_i^2 algebraically.
ResidualFn quadratic_residual(const std::vector<double>& c) {
  return [c](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) fx.data()[i] = x.data()[i] * x.data()[i] - c[i];
    return Result<void>::success();
  };
}

std::vector<double> quadratic_residual_span(const std::vector<double>& c,
                                            const std::vector<double>& x) {
  std::vector<double> out(c.size());
  for (std::size_t i = 0; i < c.size(); ++i) out[i] = x[i] * x[i] - c[i];
  return out;
}

}  // namespace

TEST(Mffd, ActionAgreesWithAnalyticJacobianAtTheWalkerPerniceStep) {
  const std::vector<double> c = {4.0, 9.0};
  const std::vector<double> x0 = {3.0, 1.0};
  const std::vector<double> v0 = {1.0, 0.5};

  DistVector x = vec_from(x0);
  DistVector fx = vec_sized(2);
  ASSERT_TRUE(quadratic_residual(c)(x, fx).ok());

  MffdOptions mffd_opt;
  MffdJacobianAction mffd(quadratic_residual(c), mffd_opt);
  mffd.set_point(x, fx);

  DistVector v = vec_from(v0);
  DistVector jv = vec_sized(2);
  ASSERT_TRUE(mffd.apply(v, jv).ok());

  ASSERT_EQ(mffd.trace().size(), 1u);
  const double h = mffd.trace().front().h;

  // Walker-Pernice: h = error_rel * sqrt(1 + ||x||) / max(||v||, min_direction_norm).
  const double x_norm = std::sqrt(x0[0] * x0[0] + x0[1] * x0[1]);
  const double v_norm = std::sqrt(v0[0] * v0[0] + v0[1] * v0[1]);
  const double expect_h =
      mffd_opt.error_rel * std::sqrt(1.0 + x_norm) / std::max(v_norm, mffd_opt.min_direction_norm);
  EXPECT_NEAR(h, expect_h, 1e-15 * expect_h);

  // Analytic J v = [2 x0 v0, 2 x1 v1].
  const double analytic0 = 2.0 * x0[0] * v0[0];
  const double analytic1 = 2.0 * x0[1] * v0[1];

  // The MFFD action's own error against the analytic J v is, ALGEBRAICALLY,
  // exactly h * v_i^2 for this quadratic (no higher-order terms at all) --
  // but AT THE WALKER-PERNICE STEP ITSELF, that truncation term and the
  // round-off incurred by subtracting two close evaluations of F and
  // dividing by h are deliberately the SAME order of magnitude (that
  // balance is the entire reason the WP rule has the sqrt(eps_mach) form
  // it does, rather than an arbitrarily small h), so the observed error is
  // O(sqrt(eps_mach)) overall, not a clean O(h) match to the truncation
  // term alone. The bound below is generous for exactly that reason: it
  // confirms the MFFD action is a genuinely good approximation to the
  // analytic Jacobian (distinguishing it from an O(1) wrong-Jacobian bug),
  // not a tight truncation-only estimate.
  const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(jv.data()[0], analytic0, 100.0 * sqrt_eps * std::max(1.0, std::fabs(analytic0)));
  EXPECT_NEAR(jv.data()[1], analytic1, 100.0 * sqrt_eps * std::max(1.0, std::fabs(analytic1)));
}

TEST(Mffd, TaylorRemainderOrderConfirmsTheAnalyticJacobianIsOrderTwo) {
  const std::vector<double> c = {4.0, 9.0};
  const std::vector<double> x0 = {3.0, 1.0};
  const std::vector<double> v0 = {1.0, 0.5};

  auto F = [&](const std::vector<double>& x) { return quadratic_residual_span(c, x); };
  auto Jv = [&](const std::vector<double>& x, const std::vector<double>& v) {
    return std::vector<double>{2.0 * x[0] * v[0], 2.0 * x[1] * v[1]};
  };

  const TaylorRemainderResult res = taylor_remainder_order(F, Jv, x0, v0, /*h0=*/1e-1,
                                                           /*ndecades=*/6);
  ASSERT_FALSE(res.at_floor_everywhere);
  EXPECT_NEAR(res.order, 2.0, 0.2) << "a quadratic residual's Taylor remainder must shrink at "
                                      "order 2 above the round-off floor";
}

TEST(Mffd, EvaluationsCounterAndTraceClearing) {
  const std::vector<double> c = {4.0};
  DistVector x = vec_from(std::vector<double>{3.0});
  DistVector fx = vec_sized(1);
  ASSERT_TRUE(quadratic_residual(c)(x, fx).ok());

  MffdJacobianAction mffd(quadratic_residual(c));
  mffd.set_point(x, fx);
  DistVector v = vec_from(std::vector<double>{1.0});
  DistVector jv = vec_sized(1);
  ASSERT_TRUE(mffd.apply(v, jv).ok());
  ASSERT_TRUE(mffd.apply(v, jv).ok());
  EXPECT_EQ(mffd.evaluations(), 2u);
  EXPECT_EQ(mffd.trace().size(), 2u);
  mffd.clear_trace();
  EXPECT_EQ(mffd.trace().size(), 0u);
  EXPECT_EQ(mffd.evaluations(), 2u) << "clear_trace() must not reset the evaluation counter";
}

TEST(Mffd, ApplyBeforeSetPointIsRefused) {
  MffdJacobianAction mffd(quadratic_residual({4.0}));
  DistVector v = vec_from(std::vector<double>{1.0});
  DistVector jv = vec_sized(1);
  Result<void> r = mffd.apply(v, jv);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-NEWTON-MFFD-NO-POINT");
}

// Correction (a): v = 0 exactly is not a corner case -- fuse::linalg::gmres()
// evaluates its own first matrix-vector product at the initial guess d = 0
// before any Arnoldi step, so apply()'s very first call on every Newton
// step sees this. Without min_direction_norm's zero-division safeguard,
// h would be +inf (finite/0) and x + h*v would be x + inf*0, IEEE 754's
// NaN. The correct action at v = 0 is exactly the zero vector, whatever
// finite h the floor produces (x + h*0 == x identically).
TEST(Mffd, ZeroDirectionIsSafeAndReturnsExactlyZero) {
  const std::vector<double> c = {4.0, 9.0};
  DistVector x = vec_from(std::vector<double>{3.0, 1.0});
  DistVector fx = vec_sized(2);
  ASSERT_TRUE(quadratic_residual(c)(x, fx).ok());

  MffdJacobianAction mffd(quadratic_residual(c));
  mffd.set_point(x, fx);
  DistVector v = vec_sized(2);  // zero by construction (DistVector's own ctor)
  v.set_zero();
  DistVector jv = vec_sized(2);
  ASSERT_TRUE(mffd.apply(v, jv).ok());

  ASSERT_EQ(mffd.trace().size(), 1u);
  const double h = mffd.trace().front().h;
  ASSERT_TRUE(std::isfinite(h));
  EXPECT_GT(h, 0.0);
  EXPECT_EQ(jv.data()[0], 0.0);
  EXPECT_EQ(jv.data()[1], 0.0);
}

// ---------------------------------------------------------------------------
// 3. Eisenstat-Walker forcing.
// ---------------------------------------------------------------------------

TEST(EisenstatWalker, FirstStepUsesEtaInitialForEveryChoiceButConstant) {
  EisenstatWalkerOptions opt;
  opt.eta_initial = 0.3;
  for (EisenstatWalkerChoice c : {EisenstatWalkerChoice::Constant, EisenstatWalkerChoice::Choice1,
                                  EisenstatWalkerChoice::Choice2}) {
    opt.choice = c;
    const double eta0 = eisenstat_walker_eta(opt, /*k=*/0, /*eta_prev=*/0.0, /*norm_fk=*/1.0,
                                             /*norm_fkm1=*/1.0, /*linear_residual_prev=*/-1.0);
    if (c == EisenstatWalkerChoice::Constant) {
      EXPECT_NEAR(eta0, std::clamp(opt.eta_constant, opt.eta_min, opt.eta_max), 1e-15);
    } else {
      EXPECT_NEAR(eta0, opt.eta_initial, 1e-15) << fuse::num::to_cstring(c);
    }
  }
}

TEST(EisenstatWalker, Choice2MatchesTheGammaPowerFormula) {
  EisenstatWalkerOptions opt;
  opt.choice = EisenstatWalkerChoice::Choice2;
  opt.gamma = 0.9;
  opt.alpha = 2.0;
  opt.safeguard_threshold = 0.1;
  // Pick norm_fk/norm_fkm1 so the plain formula is BELOW the safeguard
  // threshold -- gamma*(ratio)^2 = 0.9 * (0.05)^2 = 0.00225 << 0.1, and
  // gamma*eta_prev^alpha = 0.9*0.2^2 = 0.036 < 0.1 too, so the safeguard
  // does not engage and the plain formula is read back exactly.
  const double eta = eisenstat_walker_eta(opt, /*k=*/3, /*eta_prev=*/0.2, /*norm_fk=*/0.05,
                                          /*norm_fkm1=*/1.0, /*linear_residual_prev=*/-1.0);
  const double expect = opt.gamma * std::pow(0.05 / 1.0, opt.alpha);
  EXPECT_NEAR(eta, expect, 1e-14);
}

TEST(EisenstatWalker, OverSolvingSafeguardEngages) {
  EisenstatWalkerOptions opt;
  opt.choice = EisenstatWalkerChoice::Choice2;
  opt.gamma = 0.9;
  opt.alpha = 2.0;
  opt.safeguard_threshold = 0.1;
  // norm_fk/norm_fkm1 tiny -> plain formula ~0, but eta_prev=0.5 makes
  // gamma*eta_prev^alpha = 0.9*0.25 = 0.225 > 0.1: the safeguard must
  // floor eta at 0.225, not let it collapse towards the plain formula's
  // near-zero value.
  const double eta = eisenstat_walker_eta(opt, /*k=*/5, /*eta_prev=*/0.5, /*norm_fk=*/1e-8,
                                          /*norm_fkm1=*/1.0, /*linear_residual_prev=*/-1.0);
  const double safeguard = opt.gamma * std::pow(0.5, opt.alpha);
  EXPECT_NEAR(eta, safeguard, 1e-12);
  EXPECT_GT(safeguard, opt.safeguard_threshold);
}

TEST(EisenstatWalker, CapAndFloorClampTheFormula) {
  EisenstatWalkerOptions opt;
  opt.choice = EisenstatWalkerChoice::Choice2;
  opt.eta_max = 0.9;
  opt.eta_min = 1e-4;
  // norm_fk > norm_fkm1 (the residual GREW on the previous linear step,
  // which can happen inside an otherwise-converging outer Newton
  // iteration) drives the plain formula above 1 -- must clamp at eta_max.
  const double eta_hi = eisenstat_walker_eta(opt, 2, 0.3, /*norm_fk=*/10.0, /*norm_fkm1=*/1.0, -1.0);
  EXPECT_NEAR(eta_hi, opt.eta_max, 1e-15);

  // norm_fk << norm_fkm1 drives the plain formula near zero -- must clamp
  // at eta_min.
  const double eta_lo = eisenstat_walker_eta(opt, 2, 0.0, /*norm_fk=*/1e-12, /*norm_fkm1=*/1.0, -1.0);
  EXPECT_NEAR(eta_lo, opt.eta_min, 1e-15);
}

TEST(EisenstatWalker, Choice1UsesThePreviousLinearResidual) {
  EisenstatWalkerOptions opt;
  opt.choice = EisenstatWalkerChoice::Choice1;
  opt.safeguard_threshold = 1.0;  // disable the safeguard for this check
  const double norm_fk = 0.08, norm_fkm1 = 1.0, linear_residual_prev = 0.1;
  const double eta = eisenstat_walker_eta(opt, 4, 0.3, norm_fk, norm_fkm1, linear_residual_prev);
  const double expect =
      std::clamp(std::fabs(norm_fk - linear_residual_prev) / norm_fkm1, opt.eta_min, opt.eta_max);
  EXPECT_NEAR(eta, expect, 1e-14);
}

// ---------------------------------------------------------------------------
// The atan-vector test problem: F_i(x) = atan(x_i) - atan(c_i), the
// classic line-search-necessity example (a full Newton step from a large
// |x0| overshoots badly because atan saturates). Used for the end-to-end
// driver tests below: convergence/divergence codes, line search
// engagement, the lagged block preconditioner, and rank agreement.
// ---------------------------------------------------------------------------

namespace {

ResidualFn atan_residual(const std::vector<double>& c) {
  return [c](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) {
      fx.data()[i] = std::atan(x.data()[i]) - std::atan(c[i]);
    }
    return Result<void>::success();
  };
}

/// Each coordinate is its own diagonal Jacobian block: the EXACT inverse
/// preconditioner for block i is 1 / (1 / (1 + x_i^2)) = (1 + x_i^2) --
/// but since this preconditioner is rebuilt at a LAGGED point (not
/// necessarily the current x), it reads whatever x the caller's own
/// refresh last captured, exactly the physics-based-but-stale
/// preconditioner the lag option is testing.
std::vector<PreconditionerBlock> exact_block_preconditioner(std::shared_ptr<std::vector<double>> x_at_refresh) {
  std::vector<PreconditionerBlock> blocks;
  for (std::size_t i = 0; i < x_at_refresh->size(); ++i) {
    blocks.push_back(PreconditionerBlock{
        i, 1,
        [x_at_refresh, i](std::span<const double> r, std::span<double> z) -> Result<void> {
          const double xi = (*x_at_refresh)[i];
          z[0] = r[0] * (1.0 + xi * xi);
          return Result<void>::success();
        }});
  }
  return blocks;
}

}  // namespace

// ---------------------------------------------------------------------------
// 4. Line search, exercised end to end through the full Newton driver.
// ---------------------------------------------------------------------------

TEST(NewtonKrylovLineSearch, BtEngagesAndRecoversOnAPoorInitialGuess) {
  const std::vector<double> c = {0.0};
  DistVector x = vec_from(std::vector<double>{20.0});  // far out on atan's saturating tail

  NewtonKrylovOptions opt;
  opt.line_search.kind = LineSearchKind::BT;
  opt.rtol = 1e-10;
  opt.atol = 1e-12;
  opt.max_nonlinear_iterations = 60;

  Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  const NewtonResult& res = r.value();

  ASSERT_FALSE(res.trace.empty());
  bool any_backtrack = false;
  for (const auto& it : res.trace) {
    if (it.line_search_trials.size() > 1) any_backtrack = true;
  }
  EXPECT_TRUE(any_backtrack) << "a full Newton step from x0=20 on atan must be rejected by "
                                "Armijo at least once";
  EXPECT_TRUE(is_converged(res.status)) << to_cstring(res.status);
  EXPECT_NEAR(x.data()[0], 0.0, 1e-6);
}

TEST(NewtonKrylovConvergence, AbsoluteFiresWhenAtolIsTheLooseTest) {
  const std::vector<double> c = {0.3};
  DistVector x = vec_from(std::vector<double>{0.5});
  NewtonKrylovOptions opt;
  opt.atol = 1e-6;
  opt.rtol = 0.0;   // unreachable exactly -> cannot pre-empt atol
  opt.stol = 0.0;   // likewise
  Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::ConvergedAbsolute);
}

TEST(NewtonKrylovConvergence, RelativeFiresWhenAtolIsUnreachable) {
  const std::vector<double> c = {0.3};
  DistVector x = vec_from(std::vector<double>{0.5});
  NewtonKrylovOptions opt;
  opt.atol = 0.0;
  opt.rtol = 1e-6;
  opt.stol = 0.0;
  Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::ConvergedRelative);
}

TEST(NewtonKrylovConvergence, StepFiresOnAQuinticWhoseConvergenceIsLinearNotQuadratic) {
  // F(x) = x^5 (root at 0, multiplicity 5): the exact Newton step is
  // d = -x/5, so ||dx||/||x|| == 0.2 on EVERY step, letting stol=0.25
  // accept immediately while atol/rtol (tight) do not.
  ResidualFn quintic = [](DistVector& x, DistVector& fx) -> Result<void> {
    const double xi = x.data()[0];
    fx.data()[0] = xi * xi * xi * xi * xi;
    return Result<void>::success();
  };
  DistVector x = vec_from(std::vector<double>{0.5});
  NewtonKrylovOptions opt;
  opt.atol = 1e-9;
  opt.rtol = 1e-9;
  opt.stol = 0.25;
  Result<NewtonResult> r = solve(quintic, x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::ConvergedStep);
  EXPECT_EQ(r.value().iterations, 1);
}

TEST(NewtonKrylovDivergence, FunctionCountFiresWhenTheEvaluationBudgetIsTooSmall) {
  const std::vector<double> c = {0.3};
  DistVector x = vec_from(std::vector<double>{0.5});
  NewtonKrylovOptions opt;
  opt.max_function_evaluations = 1;  // only the initial F(x0) evaluation is allowed
  Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::DivergedFunctionCount);
}

TEST(NewtonKrylovDivergence, MaxIterationsFiresWhenTheBudgetIsOneAndNoTestCanFireExactly) {
  const std::vector<double> c = {0.37};
  DistVector x = vec_from(std::vector<double>{5.0});
  NewtonKrylovOptions opt;
  opt.max_nonlinear_iterations = 1;
  opt.atol = 0.0;
  opt.rtol = 0.0;
  opt.stol = 0.0;
  Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::DivergedMaxIterations);
}

TEST(NewtonKrylovDivergence, LineSearchFiresWhenLambdaMinIsSetUnreachablyHigh) {
  const std::vector<double> c = {0.0};
  DistVector x = vec_from(std::vector<double>{20.0});  // forces at least one backtrack (see BT test above)
  NewtonKrylovOptions opt;
  opt.line_search.kind = LineSearchKind::BT;
  opt.line_search.lambda_min = 0.9;  // the first backtrack's candidate is well below this
  Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::DivergedLineSearch);
}

TEST(NewtonKrylovDivergence, LinearSolveFiresWhenGmresIsCappedBelowWhatConvergenceNeeds) {
  // n=6 independent blocks (distinct c_i -> distinct Jacobian diagonal
  // entries) with the IDENTITY preconditioner can need up to 6 GMRES
  // iterations; capping max_iterations at 1 with a tight forcing term
  // leaves GMRES unable to certify convergence.
  const std::vector<double> c = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<double> x0 = {3.0, 3.1, 3.2, 3.3, 3.4, 3.5};
  DistVector x = vec_from(x0);
  NewtonKrylovOptions opt;
  opt.forcing.choice = EisenstatWalkerChoice::Constant;
  opt.forcing.eta_constant = 1e-12;
  opt.forcing.eta_min = 1e-12;
  opt.gmres.max_iterations = 1;
  opt.gmres.restart = 1;
  Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::DivergedLinearSolve);
}

TEST(NewtonKrylovDivergence, FunctionErrorFiresWhenFReturnsAFailure) {
  int calls = 0;
  ResidualFn flaky = [&calls](DistVector& x, DistVector& fx) -> Result<void> {
    ++calls;
    if (calls == 2) return Failure{"E-TEST-FLAKY", "deliberate failure for the test"};
    fx.data()[0] = x.data()[0] - 0.3;
    return Result<void>::success();
  };
  DistVector x = vec_from(std::vector<double>{5.0});
  NewtonKrylovOptions opt;
  Result<NewtonResult> r = solve(flaky, x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::DivergedFunctionError);
}

TEST(NewtonKrylovDivergence, ResidualGrowthFiresWhenBasicDampingOvershootsPastTheSaturatingTail) {
  // Basic line search applies NO merit check at all, so a deliberately
  // oversized damping factor can be used to CONSTRUCT residual growth on
  // demand (each code is tested by constructing the condition), rather than relying on BT ever letting it happen (BT
  // guarantees f(lambda) < f(0) whenever it accepts a step at all, so
  // residual growth past an accepted BT step is not reachable).
  const std::vector<double> c = {0.0};
  DistVector x = vec_from(std::vector<double>{0.1});  // ||F0|| = atan(0.1) = 0.0997, small
  NewtonKrylovOptions opt;
  opt.line_search.kind = LineSearchKind::Basic;
  opt.line_search.damping = 50.0;  // sends x far onto atan's saturating tail
  opt.divergence_tolerance = 2.0;  // ||F_new|| only needs to exceed 2*||F0||
  opt.max_nonlinear_iterations = 1;
  Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().status, NewtonStatus::DivergedResidualGrowth);
}

TEST(NewtonKrylovStatus, RunningIsNeitherConvergedNorDiverged) {
  EXPECT_FALSE(is_converged(NewtonStatus::Running));
  EXPECT_FALSE(is_diverged(NewtonStatus::Running));
}

TEST(NewtonKrylovStatus, EveryTerminalCodeIsExactlyOneOfConvergedOrDiverged) {
  const std::vector<NewtonStatus> terminal = {
      NewtonStatus::ConvergedAbsolute,
      NewtonStatus::ConvergedRelative,
      NewtonStatus::ConvergedStep,
      NewtonStatus::DivergedFunctionCount,
      NewtonStatus::DivergedLineSearch,
      NewtonStatus::DivergedResidualGrowth,
      NewtonStatus::DivergedLinearSolve,
      NewtonStatus::DivergedMaxIterations,
      NewtonStatus::DivergedFunctionError,
  };
  for (NewtonStatus s : terminal) {
    EXPECT_NE(is_converged(s), is_diverged(s)) << to_cstring(s);
    EXPECT_STRNE(to_cstring(s), "?") << "every enumerator must have a named string";
  }
  EXPECT_STRNE(to_cstring(NewtonStatus::Running), "?");
}

// ---------------------------------------------------------------------------
// 6. The lagged, block-Jacobi preconditioner.
// ---------------------------------------------------------------------------

TEST(NewtonKrylovPreconditioner, ExactBlockInverseConvergesGmresWithinBlockCountIterations) {
  const std::vector<double> c = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<double> x0 = {1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
  DistVector x = vec_from(x0);

  auto x_at_refresh = std::make_shared<std::vector<double>>(x0);
  auto refresh = [x_at_refresh](const DistVector& xk, const DistVector&) -> Result<void> {
    *x_at_refresh = to_std(xk);
    return Result<void>::success();
  };
  auto rebuild_precond = [x_at_refresh]() -> Result<BlockJacobiPreconditioner> {
    return BlockJacobiPreconditioner::from_blocks(exact_block_preconditioner(x_at_refresh), 6);
  };
  Result<BlockJacobiPreconditioner> precond0 = rebuild_precond();
  ASSERT_TRUE(precond0.ok()) << precond0.error().message;
  // A mutable holder so the preconditioner-apply closure always reads the
  // MOST RECENTLY refreshed block set, including the one `refresh` above
  // triggers at lag boundaries.
  auto precond_holder = std::make_shared<BlockJacobiPreconditioner>(precond0.value());
  auto refresh_with_rebuild = [=](const DistVector& xk, const DistVector& fxk) -> Result<void> {
    if (Result<void> r = refresh(xk, fxk); !r.ok()) return r;
    Result<BlockJacobiPreconditioner> rebuilt = rebuild_precond();
    if (!rebuilt.ok()) return rebuilt.error();
    *precond_holder = rebuilt.value();
    return Result<void>::success();
  };
  auto precond_apply = [precond_holder](DistVector& r, DistVector& z) -> Result<void> {
    return precond_holder->apply(r, z);
  };

  NewtonKrylovOptions opt;
  opt.preconditioner_lag = 1;  // rebuild every step -- "exact" stays exact at every linearisation
  opt.gmres.max_iterations = 200;
  opt.gmres.restart = 30;

  Result<NewtonResult> r = solve(atan_residual(c), x, precond_apply, refresh_with_rebuild, opt);
  ASSERT_TRUE(r.ok());
  ASSERT_TRUE(is_converged(r.value().status)) << to_cstring(r.value().status);
  for (const auto& it : r.value().trace) {
    EXPECT_LE(it.gmres_iterations, 6u)
        << "iteration " << it.iteration << ": the exact block-diagonal inverse reduces the "
           "preconditioned operator to the identity, which GMRES solves in at most one step per "
           "independent block";
  }
}

TEST(NewtonKrylovPreconditioner, IdentityNeedsMoreGmresIterationsThanTheExactBlockInverse) {
  const std::vector<double> c = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<double> x0 = {1.5, 1.6, 1.7, 1.8, 1.9, 2.0};

  DistVector x_id = vec_from(x0);
  NewtonKrylovOptions opt;
  opt.gmres.max_iterations = 200;
  opt.gmres.restart = 30;
  Result<NewtonResult> r_id = solve(atan_residual(c), x_id, nullptr, nullptr, opt);
  ASSERT_TRUE(r_id.ok());
  ASSERT_TRUE(is_converged(r_id.value().status));

  std::size_t max_gmres_identity = 0;
  for (const auto& it : r_id.value().trace) {
    max_gmres_identity = std::max(max_gmres_identity, it.gmres_iterations);
  }
  EXPECT_GT(max_gmres_identity, 1u)
      << "six blocks with distinct Jacobian entries and no preconditioning should need more "
         "than one GMRES iteration on at least one Newton step";
}

TEST(NewtonKrylovPreconditioner, CoverageGapIsRefused) {
  std::vector<PreconditionerBlock> blocks;
  blocks.push_back(PreconditionerBlock{0, 2, [](std::span<const double> r, std::span<double> z) {
                                         z[0] = r[0];
                                         z[1] = r[1];
                                         return Result<void>::success();
                                       }});
  // Covers [0, 2) only; n=5 leaves a TRAILING gap.
  Result<BlockJacobiPreconditioner> r = BlockJacobiPreconditioner::from_blocks(blocks, 5);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-NEWTON-PRECOND-COVERAGE");
}

TEST(NewtonKrylovPreconditioner, InteriorCoverageGapIsRefused) {
  auto identity = [](std::span<const double> r, std::span<double> z) {
    std::copy(r.begin(), r.end(), z.begin());
    return Result<void>::success();
  };
  std::vector<PreconditionerBlock> blocks;
  // [0, 2) and [3, 5): a gap at [2, 3), caught by the between-blocks check
  // rather than the trailing-coverage check the case above exercises.
  blocks.push_back(PreconditionerBlock{0, 2, identity});
  blocks.push_back(PreconditionerBlock{3, 2, identity});
  Result<BlockJacobiPreconditioner> r = BlockJacobiPreconditioner::from_blocks(blocks, 5);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-NEWTON-PRECOND-COVERAGE");
}

TEST(NewtonKrylovPreconditioner, LagScheduleRefreshesExactlyAtTheDeclaredInterval) {
  // preconditioner_lag=3: refresh() must fire at Newton iteration 0 (always,
  // regardless of lag) and then at iterations 3, 6, ... -- never at 1, 2, 4,
  // 5. A problem that needs enough Newton steps to see several lag
  // boundaries: six independent atan coordinates far from their roots.
  const std::vector<double> c = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::vector<double> x0 = {8.0, 8.1, 8.2, 8.3, 8.4, 8.5};
  DistVector x = vec_from(x0);

  int refresh_calls = 0;
  auto refresh = [&](const DistVector&, const DistVector&) -> Result<void> {
    ++refresh_calls;
    return Result<void>::success();
  };
  auto precond_apply = [](DistVector& r, DistVector& z) -> Result<void> {
    z.copy_from(r);
    return Result<void>::success();
  };

  NewtonKrylovOptions opt;
  opt.preconditioner_lag = 3;
  opt.max_nonlinear_iterations = 12;
  Result<NewtonResult> res = solve(atan_residual(c), x, precond_apply, refresh, opt);
  ASSERT_TRUE(res.ok());
  ASSERT_GE(res.value().trace.size(), 7u) << "need at least iterations 0..6 to see two lag "
                                              "boundaries after the first";

  int expected_refreshes = 0;
  for (std::size_t k = 0; k < res.value().trace.size(); ++k) {
    const bool should_refresh = (k == 0) || (k % 3 == 0);
    EXPECT_EQ(res.value().trace[k].preconditioner_refreshed, should_refresh)
        << "iteration " << k;
    if (should_refresh) ++expected_refreshes;
  }
  EXPECT_EQ(refresh_calls, expected_refreshes);
}

// ---------------------------------------------------------------------------
// Rank agreement: two in-process partitions, each owning half the atan
// vector problem's coordinates, reduced through a REAL thread-synchronised
// Reducer -- every DistVector::sum()/max() call pairs the two sides'
// local contributions via a rendezvous, so a decision made from a locally
// computed (unreduced) quantity would cause the two threads' call
// sequences to diverge and DEADLOCK, rather than merely disagree. A
// generous wait_for() timeout turns that deadlock into a clear test
// failure instead of an unbounded hang.
// ---------------------------------------------------------------------------

namespace {

/// Pairs the k-th `combine()` call from each of exactly two participants.
/// Whichever call arrives SECOND computes the combined result for both.
class RendezvousPair {
 public:
  double combine(double mine, bool is_max) {
    std::unique_lock<std::mutex> lk(mu_);
    const int my_turn = turn_;
    pending_[count_] = mine;
    ++count_;
    if (count_ == 2) {
      result_ = is_max ? std::max(pending_[0], pending_[1]) : (pending_[0] + pending_[1]);
      count_ = 0;
      ++turn_;
      cv_.notify_all();
      return result_;
    }
    const bool advanced = cv_.wait_for(lk, std::chrono::seconds(10),
                                       [&] { return turn_ != my_turn; });
    if (!advanced) {
      // The two sides' call sequences diverged -- exactly the symptom a
      // decision made from an unreduced local quantity would cause.
      ADD_FAILURE() << "RendezvousPair::combine timed out: the two partitions' collective call "
                       "sequences diverged";
      return mine;
    }
    return result_;
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int turn_ = 0;
  int count_ = 0;
  double pending_[2] = {0.0, 0.0};
  double result_ = 0.0;
};

class PairedReducer : public Reducer {
 public:
  PairedReducer(int rank, std::shared_ptr<RendezvousPair> pair) : rank_(rank), pair_(std::move(pair)) {}

  [[nodiscard]] double sum(double local) const override { return pair_->combine(local, false); }
  void sum(double* inout, std::size_t n) const override {
    for (std::size_t i = 0; i < n; ++i) inout[i] = pair_->combine(inout[i], false);
  }
  [[nodiscard]] double max(double local) const override { return pair_->combine(local, true); }
  [[nodiscard]] int rank() const override { return rank_; }
  [[nodiscard]] int size() const override { return 2; }
  void allgather(const double* send, std::size_t n_each, double* recv) const override {
    std::copy(send, send + n_each, recv + static_cast<std::size_t>(rank_) * n_each);
  }

 private:
  int rank_;
  std::shared_ptr<RendezvousPair> pair_;
};

}  // namespace

TEST(NewtonKrylovRankAgreement, TwoPartitionsReachTheSameStatusAtTheSameIteration) {
  // Six coordinates, split 3/3 across two "ranks". Each rank's own block
  // is a self-contained slice of the SAME atan_residual vector problem --
  // F has no cross-rank coupling in VALUE, only in the globally reduced
  // norms every convergence/divergence/EW/line-search decision reads, so
  // the two ranks' control flow (iteration counts, line-search trials,
  // GMRES iterations) is identical by construction UNLESS a decision
  // secretly used an unreduced local quantity -- the scenario the
  // RendezvousPair harness is built to catch by deadlocking instead.
  const std::vector<double> c_all = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  const std::vector<double> x0_all = {3.0, 3.1, 3.2, 3.3, 3.4, 3.5};

  auto pair = std::make_shared<RendezvousPair>();
  NewtonStatus status[2];
  int iterations[2];

  auto run_partition = [&](int rank) {
    const std::size_t lo = rank == 0 ? 0 : 3, hi = rank == 0 ? 3 : 6;
    std::vector<double> c(c_all.begin() + static_cast<std::ptrdiff_t>(lo),
                          c_all.begin() + static_cast<std::ptrdiff_t>(hi));
    std::vector<double> x0(x0_all.begin() + static_cast<std::ptrdiff_t>(lo),
                           x0_all.begin() + static_cast<std::ptrdiff_t>(hi));
    DistVector x(x0.size(), make_serial_exchanger(), std::make_shared<PairedReducer>(rank, pair));
    std::copy(x0.begin(), x0.end(), x.data());

    NewtonKrylovOptions opt;
    opt.atol = 1e-10;
    opt.rtol = 1e-10;
    Result<NewtonResult> r = solve(atan_residual(c), x, nullptr, nullptr, opt);
    ASSERT_TRUE(r.ok());
    status[rank] = r.value().status;
    iterations[rank] = r.value().iterations;
  };

  std::thread t0([&] { run_partition(0); });
  std::thread t1([&] { run_partition(1); });
  t0.join();
  t1.join();

  EXPECT_EQ(status[0], status[1]);
  EXPECT_EQ(iterations[0], iterations[1]);
  EXPECT_TRUE(is_converged(status[0])) << to_cstring(status[0]);
}
