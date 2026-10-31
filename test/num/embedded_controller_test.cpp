// EmbeddedController::propose() is a pure function of (err_norm, h, state)
// (embedded_controller.hpp's own top comment) -- every case here is checked against a
// value computed independently (by hand, cited in the comment) from the SAME formula
// the vendored dopri45_integrator.C/embedded_dirk_integrator.C use, not against a
// number produced by running propose() itself and eyeballing it.

#include "fuse/num/embedded_controller.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

namespace fuse::num {
namespace {

// Hand-computed against dopri45_integrator.C:250-256's own formula:
//   expo1 = 0.2 - beta*0.75 = 0.2 - 0.08*0.75 = 0.14
//   fac11 = err^expo1 = 0.5^0.14
//   fac = max(facc2, min(facc1, fac11/pow(fac_old,beta)/safety))
//       = fac11 / 0.0001^0.08 / 0.9   (fac_old starts at 1e-4, embedded_controller.hpp's
//                                      own ControllerState default)
//   hnew = h/fac
TEST(EmbeddedController, PiAcceptedStepMatchesTheHandComputedDopri45Formula) {
  ControllerOptions opt;
  opt.kind = ControllerKind::PI;
  opt.order = 5;
  opt.embedded_order = 4;
  opt.safety = 0.9;
  opt.beta = 0.08;
  opt.fac_min = 0.2;
  opt.fac_max = 4.0;
  EmbeddedController ctrl(opt);
  ControllerState state;

  const double err = 0.5;
  const double h = 0.1;
  const double expo1 = 0.2 - 0.08 * 0.75;
  const double fac11 = std::pow(err, expo1);
  const double fac = std::max(0.25, std::min(5.0, fac11 / std::pow(1.0e-4, 0.08) / 0.9));
  const double expected_h_next = h / fac;

  Result<StepDecision> r = ctrl.propose(err, h, state);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().outcome, StepOutcome::Accepted);
  EXPECT_NEAR(r.value().h_next, expected_h_next, 1e-12 * std::fabs(expected_h_next));
  EXPECT_DOUBLE_EQ(state.fac_old, std::max(err, 1.0e-4));
}

// A rejected step: err > 1, hnew = h/min(facc1, fac11/safety), fac_old UNCHANGED
// (dopri45_integrator.C:261 only updates facold inside the accepted branch).
TEST(EmbeddedController, PiRejectedStepDoesNotUpdateFacOld) {
  ControllerOptions opt;
  opt.kind = ControllerKind::PI;
  opt.order = 5; opt.embedded_order = 4; opt.safety = 0.9; opt.beta = 0.08;
  opt.fac_min = 0.2; opt.fac_max = 4.0;
  EmbeddedController ctrl(opt);
  ControllerState state;
  state.fac_old = 0.25;  // a value an earlier accepted step would have left

  const double err = 4.0;
  const double h = 0.2;
  const double expo1 = 0.2 - 0.08 * 0.75;
  const double fac11 = std::pow(err, expo1);
  const double expected_h_next = h / std::min(5.0, fac11 / 0.9);

  Result<StepDecision> r = ctrl.propose(err, h, state);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().outcome, StepOutcome::Rejected);
  EXPECT_NEAR(r.value().h_next, expected_h_next, 1e-12 * std::fabs(expected_h_next));
  EXPECT_DOUBLE_EQ(state.fac_old, 0.25) << "a rejection must not touch fac_old";
}

// I-kind, embedded_dirk_integrator.C:352-390's own formula:
//   order_power = 1/(min(order, embedded_order)+1)
//   newdelt = safety * h * (1/err)^order_power
//   accepted: hnew = min(fac_max*h, newdelt); rejected: hnew = max(fac_min*h, newdelt)
TEST(EmbeddedController, IKindMatchesTheHandComputedEmbeddedDirkFormula) {
  ControllerOptions opt;
  opt.kind = ControllerKind::I;
  opt.order = 4; opt.embedded_order = 3; opt.safety = 0.8;
  opt.fac_min = 0.2; opt.fac_max = 4.0;
  EmbeddedController ctrl(opt);
  ControllerState state;

  {
    const double err = 0.3;  // accepted
    const double h = 0.05;
    const double order_power = 1.0 / (std::min(4, 3) + 1);
    const double newdelt = 0.8 * h * std::pow(1.0 / err, order_power);
    const double expected = std::min(4.0 * h, newdelt);
    Result<StepDecision> r = ctrl.propose(err, h, state);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().outcome, StepOutcome::Accepted);
    EXPECT_NEAR(r.value().h_next, expected, 1e-12 * std::fabs(expected));
  }
  {
    const double err = 3.0;  // rejected
    const double h = 0.05;
    const double order_power = 1.0 / (std::min(4, 3) + 1);
    const double newdelt = 0.8 * h * std::pow(1.0 / err, order_power);
    const double expected = std::max(0.2 * h, newdelt);
    Result<StepDecision> r = ctrl.propose(err, h, state);
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().outcome, StepOutcome::Rejected);
    EXPECT_NEAR(r.value().h_next, expected, 1e-12 * std::fabs(expected));
  }
}

// E-INTEGRATOR-NONFINITE-ERROR: a refusal that fires, not a documented intention
// (the report contract's own standard: "a refusal counts as verified only when a test
// makes it fire").
TEST(EmbeddedController, RefusesNonFiniteErrorNorm) {
  ControllerOptions opt;
  EmbeddedController ctrl(opt);
  ControllerState state;

  Result<StepDecision> r_nan = ctrl.propose(std::numeric_limits<double>::quiet_NaN(), 0.1, state);
  ASSERT_FALSE(r_nan.ok());
  EXPECT_EQ(r_nan.error().code, "E-INTEGRATOR-NONFINITE-ERROR");

  Result<StepDecision> r_inf = ctrl.propose(std::numeric_limits<double>::infinity(), 0.1, state);
  ASSERT_FALSE(r_inf.ok());
  EXPECT_EQ(r_inf.error().code, "E-INTEGRATOR-NONFINITE-ERROR");
}

// E-INTEGRATOR-STEP-TOO-SMALL: a rejected step whose own next proposal falls below
// h_min fires the refusal (made to fire, not merely declared).
TEST(EmbeddedController, RefusesStepTooSmallOnARejectionBelowTheFloor) {
  ControllerOptions opt;
  opt.kind = ControllerKind::PI;
  opt.order = 5; opt.embedded_order = 4; opt.safety = 0.9; opt.beta = 0.08;
  opt.fac_min = 0.2; opt.fac_max = 4.0;
  opt.h_min = 1e-6;
  EmbeddedController ctrl(opt);
  ControllerState state;

  // A huge error norm forces a large rejection shrink; with a tiny starting h, the
  // rejected step's own h_next lands below h_min.
  Result<StepDecision> r = ctrl.propose(1.0e30, 1.0e-6, state);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-INTEGRATOR-STEP-TOO-SMALL");
}

// An ACCEPTED step is never refused on h_next alone, however small the controller
// wants the next step to be (embedded_controller.hpp's own doc comment).
TEST(EmbeddedController, AnAcceptedStepIsNeverRefusedOnASmallNextProposal) {
  ControllerOptions opt;
  opt.kind = ControllerKind::I;
  opt.order = 2; opt.embedded_order = 1; opt.safety = 0.8;
  opt.fac_min = 1e-8;  // allow an extreme shrink in the NEXT proposal
  opt.fac_max = 4.0;
  opt.h_min = 1e-3;
  EmbeddedController ctrl(opt);
  ControllerState state;

  Result<StepDecision> r = ctrl.propose(0.01, 1.0, state);  // small err: accepted
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().outcome, StepOutcome::Accepted);
}

TEST(EmbeddedController, ToCstringCoversBothKinds) {
  EXPECT_STREQ(to_cstring(ControllerKind::PI), "PI");
  EXPECT_STREQ(to_cstring(ControllerKind::I), "I");
}

}  // namespace
}  // namespace fuse::num
