#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "fuse/base/result.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 3, item 5
/// (docs/dev/integrators-design.md Section 2, mechanism 1; Round 2's own open
/// question 12).
///
/// KARMA's mechanism 1 (`ExplicitRKIntegrator::core_integrator`,
/// upstream/karma/src/numerics/temporal/explicit_rk_integrator.C:213-296) proposes
/// `hnew = safe * dynmax * (prev_dynmax/dynmax)^0.2 * (h/hold)^0.2`, a PI-shaped
/// formula carrying one step of memory (the previous `dynmax`) so consecutive
/// evaluations of the SAME problem callback both shape the next h, the same role
/// `ControllerState::fac_old` plays for `EmbeddedController`'s own PI kind
/// (embedded_controller.hpp).
///
/// ROUND 3 SIMPLIFICATION (recorded as a deviation, open_questions): this class
/// ships the memory-less (I-type, in this project's own ControllerKind naming)
/// half of that formula, `h_next = safety * min_i(scales[i])`, clamped to
/// `[fac_min, fac_max] * h_current` the same way EmbeddedController clamps its own
/// proposal. The PI extension needs a persisted `prev_scale` the same shape as
/// `ControllerState::fac_old`; it is not added here because no test in this round
/// exercises two CONSECUTIVE steps with a time-VARYING dynamical scale (the task's
/// own verification problem, linear decay with scale 1/|lambda|, is constant step
/// to step, for which the PI ratio term is identically 1 and the two formulas
/// agree) -- adding the memory term untested would be exactly the "verify before
/// you optimise" violation Round 1/2 avoided elsewhere. Flagged for the owner.
///
/// `scales[i]` are the PROBLEM's own local dynamical time scales (TemporalProblem::
/// dynamical_time_scales, integrator.hpp), one physical TIME per entry (e.g.
/// 1/|lambda| for a locally-linearised decay/oscillation mode) -- NOT rates, so the
/// MOST RESTRICTIVE (smallest) entry is the one that should bound h, mirroring
/// `dynmax` in KARMA's own formula only insofar as the vendored code's variable
/// name describes the magnitude it tracks, not a rate (Section 2's own prose:
/// "a physics callback, not a truncation-error estimate").
namespace fuse::num {

struct DynamicalControllerOptions {
  /// KARMA's own "safe" (explicit_rk_integrator.C:213's own `safe`, distinct from
  /// EmbeddedController's PI-kind 0.9/I-kind default -- Section 2 notes two
  /// different "default safety factor" values already coexist in that directory).
  double safety = 0.8;
  /// KARMA's own facl/facr (temporal_integrator.C:101-107's own
  /// `set_adaptive_params` defaults), reused verbatim: the proposal is clamped to
  /// [facl, facr] * h_current before being reported, never left unclamped.
  double fac_min = 0.2;
  double fac_max = 4.0;
};

/// A pure function of (scales, h_current) -- same shape as EmbeddedController's own
/// propose(), so a caller (explicit_integrator.hpp) can hold an
/// std::optional<DynamicalTimeScaleController> beside its
/// std::optional<EmbeddedController> and combine the two proposals (Round 3's own
/// combination rule: the minimum of whichever are present, explicit_integrator.hpp's
/// own step()).
class DynamicalTimeScaleController {
 public:
  explicit DynamicalTimeScaleController(DynamicalControllerOptions options = {})
      : options_(options) {}

  /// Returns a refusal (not a silent fallback) if `scales` is empty or every entry
  /// is non-positive -- the same "a refusal counts as verified only when a test
  /// fires it" standard every other named refusal in this track meets
  /// (docs/dev/integrators-design.md Section 4.5).
  template <typename Scalar>
  [[nodiscard]] fuse::Result<Scalar> propose(const std::vector<double>& scales,
                                              Scalar h_current) const {
    double min_scale = 0.0;
    bool have_one = false;
    for (double s : scales) {
      if (s <= 0.0) continue;
      if (!have_one || s < min_scale) {
        min_scale = s;
        have_one = true;
      }
    }
    if (!have_one) {
      return fuse::Failure{"E-DYNAMICAL-CONTROLLER-NO-SCALE",
                            "DynamicalTimeScaleController::propose: every reported "
                            "dynamical time scale was non-positive or the list was "
                            "empty -- this controller refuses rather than propose an "
                            "unbounded or zero step"};
    }
    const double h_cfl = options_.safety * min_scale;
    const double hc = static_cast<double>(h_current);
    const double h_lo = hc * options_.fac_min;
    const double h_hi = hc * options_.fac_max;
    const double clamped = std::clamp(h_cfl, h_lo, h_hi);
    return fuse::Result<Scalar>{static_cast<Scalar>(clamped)};
  }

 private:
  DynamicalControllerOptions options_;
};

}  // namespace fuse::num
