#pragma once

#include <algorithm>
#include <cmath>
#include <string>

#include "fuse/base/result.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 1
/// (docs/dev/integrators-design.md Section 4.2).
///
/// The two embedded step-size controllers Section 2 of the Stage A design found living,
/// independently and un-shared, inside two different KARMA integrator drivers --
/// DOPRI45Integrator's own PI (Gustafsson) controller
/// (upstream/karma/src/numerics/temporal/dopri45_integrator.C:198-276) and
/// EmbeddedDIRKIntegrator's own I controller
/// (upstream/karma/src/numerics/temporal/embedded_dirk_integrator.C:328-390) -- as one
/// reusable component, ControllerKind naming which formula runs rather than two
/// hand-inlined copies.
///
/// CONVENTION (task item 4): both kinds take an ALREADY rtol/atol-scaled error norm,
/// err_norm, with the step accepted iff err_norm <= 1 -- the same convention
/// DOPRI45Integrator::core_integrator's own "if (err <= 1.0)" uses (dopri45_integrator.C:
/// 258). The caller (explicit_integrator.hpp's step()) computes err_norm through a
/// norm functor it is given (DistVector::norm2 for distributed state, a plain RMS for
/// serial), so the one reduction every rank needs agrees BY CONSTRUCTION (the design
/// doc's Section 4.6: a DistVector-backed norm already reduces across every
/// participant) -- this header never issues an MPI call of its own and never needs to:
/// propose() is a pure function of the scalar err_norm every rank already agrees on,
/// so every rank that calls it with the same inputs gets the same StepDecision, by
/// construction, which is the project's own standing rule ("a collective path must never
/// return on a locally computed condition") satisfied the only way a pure function can
/// satisfy it: by having no rank-local input to disagree on in the first place.
///
/// EmbeddedDIRKIntegrator's own convention (an absolute LTE_tolerance compared
/// against a raw VecNorm, embedded_dirk_integrator.C:174) is translated into the SAME
/// err_norm <= 1 convention here: the caller forms err_norm = local_error /
/// LTE_tolerance before calling propose(), so ControllerKind::I and
/// ControllerKind::PI share one entry point and one refusal set, rather than the
/// vendored pair's own two different accept/reject thresholds.
namespace fuse::num {

/// PI: DOPRI45Integrator's own Gustafsson/Lund-stabilised controller (Hairer, Norsett &
/// Wanner, Solving ODEs I, II.4's own dopri5.f); carries one step of memory
/// (ControllerState::fac_old) so two consecutive steps' errors both shape the next h.
/// I: EmbeddedDIRKIntegrator's own controller (embedded_dirk_integrator.C:352's own
/// "facerr = pow(LTE_tolerance/local_error, order_power)", the PI extension left
/// commented out there, embedded_dirk_integrator.C:351); stateless step to step.
enum class ControllerKind { PI, I };

[[nodiscard]] inline const char* to_cstring(ControllerKind k) noexcept {
  switch (k) {
    case ControllerKind::PI: return "PI";
    case ControllerKind::I:  return "I";
  }
  return "Unknown";
}

/// order/embedded_order: the SAME two numbers ButcherTableau::order/
/// embedded_order carry (butcher_tableau.hpp) -- a caller builds this from the
/// tableau it is stepping, not from a separately-chosen pair, so the controller's own
/// exponent can never drift from the method it is actually controlling (the defect
/// Section 6.1 of the Stage A design found in DOPRI45Integrator itself, which hardcodes
/// its exponent rather than reading n_secondary_order()).
///
/// safety/fac_min/fac_max/beta: TemporalIntegrator::set_adaptive_params's own
/// defaults (temporal_integrator.C:101-107, "safety=0.8, facr=4.0, facl=0.2") and
/// DOPRI45Integrator's own constructor default (dopri45_integrator.C:34-46,
/// "safetyfac=0.9, beta=0.08 recommended by Hairer") -- two different safety-factor
/// defaults genuinely coexist in the vendored source (Stage A design Section 2); this
/// struct carries ONE value per instance rather than picking a winner, so a caller
/// building a DOPRI45-equivalent controller passes 0.9 and one building a generic
/// embedded-DIRK-equivalent one passes 0.8, each citing which vendored default it reads.
///
/// h_min: NOT a vendored field -- DOPRI45Integrator's own floor
/// (0.1*fabs(h) <= fabs(time)*uround, dopri45_integrator.C:151) needs the absolute
/// simulated time, which this controller deliberately does not carry (Section 4.2 of the
/// design: "a pure function of an already-agreed scalar error norm", not of the
/// integrator's own state) -- the caller derives h_min from its own uround/time
/// once per step and passes the result in, so the refusal below ends up checking the
/// SAME quantity DOPRI45 checks, computed one level up.
struct ControllerOptions {
  ControllerKind kind = ControllerKind::PI;
  int order = 1;
  int embedded_order = 1;
  double safety = 0.9;
  double fac_min = 0.2;
  double fac_max = 4.0;
  double beta = 0.08;
  double h_min = 0.0;
};

/// fac_old: DOPRI45Integrator's own facold, initialised to 1.0e-4
/// (dopri45_integrator.C:131, "double facold = 1.0e-4;") and updated only on an
/// ACCEPTED step (facold = max(err, 1.0e-4), dopri45_integrator.C:261) -- carried by
/// the CALLER across steps (this type has no other state to own), never reset on a
/// rejection, matching the vendored control flow exactly.
struct ControllerState {
  double fac_old = 1.0e-4;
};

enum class StepOutcome { Accepted, Rejected };

struct StepDecision {
  StepOutcome outcome = StepOutcome::Rejected;
  double h_next = 0.0;
};

/// A pure function of (err_norm, h, state) -- Section 4.2 of the design doc,
/// propose(). No vector, no vector norm, no MPI: those are the caller's job
/// (explicit_integrator.hpp), which is exactly what makes this class usable unchanged
/// whether the state is a serial std::vector<double> or a distributed
/// fuse::linalg::DistVector the caller reduced before calling in.
class EmbeddedController {
 public:
  explicit EmbeddedController(ControllerOptions opt) : opt_(opt) {}

  [[nodiscard]] const ControllerOptions& options() const noexcept { return opt_; }

  /// Refuses E-INTEGRATOR-NONFINITE-ERROR when err_norm is NaN or infinite --
  /// KARMA never checks this (Stage A design Section 4.5: an Inf err compares false
  /// to err <= 1.0 in C++ and is silently treated as a REJECT by accident, not by
  /// design); refuses E-INTEGRATOR-STEP-TOO-SMALL when a REJECTED step's own proposed
  /// h_next would already be below opt.h_min -- retrying a step this small is
  /// pointless (it will be rejected again, or is below the precision the problem's own
  /// time scale can resolve), so this is where DOPRI45's own floor
  /// (dopri45_integrator.C:151) is checked, one level up from where DOPRI45 itself
  /// checks it (this header's own top comment). An ACCEPTED step is never refused on
  /// h_next alone: the step that just succeeded is valid regardless of how small the
  /// controller wants the NEXT one to be.
  [[nodiscard]] Result<StepDecision> propose(double err_norm, double h,
                                              ControllerState& state) const {
    if (!std::isfinite(err_norm)) {
      return Failure{"E-INTEGRATOR-NONFINITE-ERROR",
                     "EmbeddedController::propose: error norm is not finite "
                     "(NaN or Inf); KARMA's own dopri45_integrator.C never checks this "
                     "and silently treats it as a rejection by accident"};
    }

    double h_next = 0.0;
    const bool accept = err_norm <= 1.0;

    if (opt_.kind == ControllerKind::PI) {
      // dopri45_integrator.C:130-166 (variable names kept identical for traceability):
      // expo1, facc1, facc2, fac11, fac, hnew.
      const double expo1 = 1.0 / (opt_.embedded_order + 1) - opt_.beta * 0.75;
      const double facc1 = 1.0 / opt_.fac_min;
      const double facc2 = 1.0 / opt_.fac_max;
      const double fac11 = std::pow(err_norm, expo1);
      if (accept) {
        const double fac = std::max(facc2, std::min(facc1, fac11 / std::pow(state.fac_old, opt_.beta) / opt_.safety));
        h_next = h / fac;
        state.fac_old = std::max(err_norm, 1.0e-4);
      } else {
        h_next = h / std::min(facc1, fac11 / opt_.safety);
        // state.fac_old is NOT updated on a rejection (dopri45_integrator.C:261 only
        // assigns facold inside the "if (err <= 1.0)" branch).
      }
    } else {
      // embedded_dirk_integrator.C:328-390, same variable names: order_power, facerr,
      // newdelt. order_power is read from the tableau's OWN order/embedded_order
      // (ControllerOptions), never hardcoded -- the fix for the DOPRI45 metadata defect
      // (Stage A design Section 6.1) generalised to every method that uses this kind.
      const double order_power = 1.0 / (std::min(opt_.order, opt_.embedded_order) + 1);
      const double facerr = std::pow(1.0 / err_norm, order_power);
      const double newdelt = opt_.safety * h * facerr;
      if (accept) {
        h_next = std::min(opt_.fac_max * h, newdelt);
      } else {
        h_next = std::max(opt_.fac_min * h, newdelt);
      }
    }

    if (!accept && opt_.h_min > 0.0 && std::fabs(h_next) < opt_.h_min) {
      return Failure{"E-INTEGRATOR-STEP-TOO-SMALL",
                     "EmbeddedController::propose: rejected step's own next proposal (" +
                     std::to_string(h_next) + ") is below h_min (" +
                     std::to_string(opt_.h_min) + "); retrying would not converge"};
    }

    return StepDecision{accept ? StepOutcome::Accepted : StepOutcome::Rejected, h_next};
  }

 private:
  ControllerOptions opt_;
};

}  // namespace fuse::num
