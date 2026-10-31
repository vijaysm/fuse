#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/num/butcher_tableau.hpp"
#include "fuse/num/embedded_controller.hpp"
#include "fuse/num/dynamical_time_scale_controller.hpp"
#include "fuse/num/integrator.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 1
/// (docs/dev/integrators-design.md Section 4.1, task item 4): the explicit family --
/// FE, KUTTA22, KUTTA44 (fixed step, no embedded estimate) and DOPRI45 (embedded, FSAL
/// per the correction in butcher_tableau.hpp's own top comment) -- as ONE class over
/// any `ButcherTableau` whose `family` is `Erk` or `ErkEmbedded`, replacing
/// `ExplicitRKIntegrator`/`DOPRI45Integrator`'s own separate drivers
/// (upstream/karma/{include,src}/numerics/temporal/explicit_rk_integrator.*,
/// dopri45_integrator.*) with one stage loop parameterised by the tableau.
///
/// NOT ported from `ExplicitRKIntegrator`: its own "adaptive" branch is the
/// dynamical-time-scale controller (Stage A design Section 2, mechanism 1), a
/// PHYSICS callback (`KARMAInterface::compute_dynamical_timescales`) this track's own
/// scope never asked for (task item 4 names only "error norm as the rtol/atol-scaled
/// RMS" -- the Hairer-style embedded mechanism, mechanisms 2/3 of that same section,
/// never mechanism 1). A component that wants dynamical-time-scale control is a later
/// track's own scope, layered on TOP of this one (it would call `step()` with an `h` it
/// chose itself, the same way a fixed-step caller does), not inside it.
///
/// NOT ported either: the FSAL reuse optimisation (DOPRI45's own stage-7 result equals
/// the next step's stage-0 evaluation, so a correctly-done FSAL implementation skips
/// one function evaluation per step). `step()` below recomputes stage 0 every call --
/// numerically IDENTICAL (the same `t`, `y` pair evaluated through the same `rhs`
/// closure is deterministic), one function evaluation more expensive per step. The
/// project's own standing rule ("robustness and accuracy beats efficiency, always...
/// once we have a verified workflow and code, we can spend time making it more
/// efficient") is why this is a stated, deliberate choice rather than an oversight.
namespace fuse::num {

/// `ExplicitIntegrator<Value>::step()`'s own refusals, beyond the ones
/// `EmbeddedController::propose` already names. `kNoRhsCode`/`kWrongFamilyCode`/
/// `kDimensionMismatchCode` moved to `integrator.hpp` in Round 2 (shared by every
/// family); `kMassUnsupportedCode` stays here, since only THIS family refuses on a
/// mass matrix rather than honouring it.
inline constexpr const char* kMassUnsupportedCode = "E-INTEGRATOR-MASS-UNSUPPORTED";
/// Round 3, item 5: a DynamicalTimeScaleController is attached but the problem has
/// no `dynamical_time_scales` callback -- a misconfiguration refusal, not the
/// controller's own "every scale non-positive" refusal (dynamical_time_scale_
/// controller.hpp).
inline constexpr const char* kNoDynamicalScalesCode = "E-INTEGRATOR-NO-DYNAMICAL-SCALES";

template <IntegratorValue Value>
class ExplicitIntegrator final : public Integrator<Value> {
 public:
  using Scalar = typename Integrator<Value>::Scalar;

  /// Fixed-step construction (FE, KUTTA22, KUTTA44 -- any tableau with
  /// `family == Erk`, i.e. no embedded pair). `step()` always reports
  /// `StepRecord::accepted = true` and `h_next = h` (nothing adapts it).
  explicit ExplicitIntegrator(ButcherTableau tableau) : tableau_(std::move(tableau)) {}

  /// Adaptive construction (DOPRI45 -- `family == ErkEmbedded`). `rtol`/`atol` are
  /// Hairer's own per-run tolerances (dopri45_integrator.C:118-119's own `atoli`/
  /// `rtoli`, read once from the tableau's caller rather than once per vendored
  /// integrator instance, same meaning); `norm` is the task's own "norm functor the
  /// caller supplies" (this header's own top comment; `serial_rms_norm`,
  /// integrator.hpp, is the default).
  ExplicitIntegrator(ButcherTableau tableau, EmbeddedController controller, Scalar rtol,
                     Scalar atol, ScaledNormFn norm = serial_rms_norm)
      : tableau_(std::move(tableau)),
        controller_(std::move(controller)),
        controller_state_(ControllerState{}),
        rtol_(rtol),
        atol_(atol),
        norm_(std::move(norm)) {}

  [[nodiscard]] const ButcherTableau& tableau() const noexcept override { return tableau_; }

  /// The controller's own running state (`fac_old`), exposed so a caller driving many
  /// steps across one window carries it forward itself (the SAME object this class
  /// updates internally; `step()` mutates it in place, this accessor exists for a
  /// caller that wants to inspect or reset it between windows).
  [[nodiscard]] std::optional<ControllerState>& controller_state() noexcept {
    return controller_state_;
  }

  /// Round 3, item 5 (docs/dev/integrators-design.md Section 2, mechanism 1):
  /// attaches a DynamicalTimeScaleController alongside (or instead of) the embedded
  /// one. With no EmbeddedController, the step is a KARMA-mechanism-1 step: ALWAYS
  /// accepted (this mechanism carries no accept/reject notion of its own, Section 2's
  /// own prose), h_next is the dynamical proposal alone. With BOTH present, the
  /// accept/reject decision stays the embedded controller's; on an accepted step,
  /// h_next is the MINIMUM of the two proposals (Round 3's own combination rule,
  /// recorded in open_questions as the owner-facing design decision, implemented as
  /// the recommendation). Requires `problem.has_dynamical_time_scales()` at step time;
  /// `step()` refuses (not silently ignores) if this is set but the problem supplies
  /// no callback.
  void set_dynamical_controller(DynamicalTimeScaleController controller) {
    dynamical_controller_ = std::move(controller);
  }

  /// Round 3, item 6: enabled by default for an ERK FSAL tableau (DOPRI45) --
  /// MEASURED bit-identical to full recomputation (explicit_integrator_test.cpp's
  /// own FSAL suite; this class's own top comment on fsal_cached_* explains why
  /// an ERK method's FSAL stage achieves this where DirkIntegrator's own
  /// implicit-family attempt could not). `set_fsal_enabled(false)` is the knob
  /// this round's own bit-identical comparison test uses to construct the
  /// "non-FSAL path" baseline off the SAME tableau and options.
  void set_fsal_enabled(bool enabled) noexcept { fsal_enabled_ = enabled; }
  [[nodiscard]] bool fsal_enabled() const noexcept { return fsal_enabled_; }

  [[nodiscard]] Result<StepRecord<Value>> step(TemporalProblem<Value>& problem, Scalar t,
                                               Scalar h, std::span<Value> state) override {
    if (!problem.has_rhs()) {
      return Failure{kNoRhsCode, "ExplicitIntegrator::step: TemporalProblem::rhs is unset"};
    }
    if (problem.has_mass()) {
      return Failure{kMassUnsupportedCode,
                     "ExplicitIntegrator::step: a mass-matrix problem (M != I) needs a "
                     "linear solve per stage, round-2 scope (DIRK/Rosenbrock); the "
                     "explicit family never reads TemporalProblem::mass"};
    }
    if (tableau_.family != ButcherFamily::Erk && tableau_.family != ButcherFamily::ErkEmbedded) {
      return Failure{kWrongFamilyCode,
                     "ExplicitIntegrator::step: tableau '" + tableau_.name +
                     "' is family " + to_cstring(tableau_.family) +
                     ", not Erk/ErkEmbedded"};
    }
    if (problem.dimension != 0 && problem.dimension != state.size()) {
      return Failure{kDimensionMismatchCode,
                     "ExplicitIntegrator::step: state.size()=" + std::to_string(state.size()) +
                     " != problem.dimension=" + std::to_string(problem.dimension)};
    }

    const int stages = tableau_.stages;
    const std::size_t n = state.size();
    std::vector<std::vector<Value>> k(static_cast<std::size_t>(stages), std::vector<Value>(n, Value{}));
    std::vector<Value> y_stage(n);
    std::vector<Value> y_new(n);
    std::size_t fevals = 0;

    // Round 3, item 6: an ERK FSAL tableau (DOPRI45) has B[last] == 0 AND
    // A[last] == B (butcher_tableau.hpp's own top comment) -- the structural
    // property that makes stage-0 reuse bit-identical (this class's own top
    // comment on fsal_cached_*): BOTH conditions verified HERE, not merely
    // assumed from ButcherTableau::is_FSAL, so a future is_FSAL tableau that
    // does NOT have this exact structure falls back to recomputing rather than
    // silently reusing an invalid cache.
    bool fsal_structure_ok = tableau_.is_FSAL && stages > 0 &&
        tableau_.B[static_cast<std::size_t>(stages - 1)] == 0.0;
    if (fsal_structure_ok) {
      const auto& a_last = tableau_.A[static_cast<std::size_t>(stages - 1)];
      for (int j = 0; j < stages; ++j) {
        if (a_last[static_cast<std::size_t>(j)] != tableau_.B[static_cast<std::size_t>(j)]) {
          fsal_structure_ok = false;
          break;
        }
      }
    }

    for (int i = 0; i < stages; ++i) {
      bool reused = false;
      if (i == 0 && fsal_structure_ok && fsal_enabled_ && fsal_cached_t_.has_value() &&
          fsal_cached_y_.has_value() && fsal_cached_k_.has_value() &&
          *fsal_cached_t_ == t && fsal_cached_y_->size() == n) {
        bool y_matches = true;
        for (std::size_t m = 0; m < n; ++m) {
          if ((*fsal_cached_y_)[m] != state[m]) { y_matches = false; break; }
        }
        if (y_matches) {
          k[static_cast<std::size_t>(i)] = *fsal_cached_k_;
          reused = true;
        }
      }
      if (reused) continue;

      for (std::size_t m = 0; m < n; ++m) y_stage[m] = state[m];
      for (int j = 0; j < i; ++j) {
        const double aij = tableau_.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        if (aij == 0.0) continue;
        for (std::size_t m = 0; m < n; ++m) {
          y_stage[m] = y_stage[m] + (h * static_cast<Scalar>(aij)) * k[static_cast<std::size_t>(j)][m];
        }
      }
      const Scalar t_stage = t + h * static_cast<Scalar>(tableau_.C[static_cast<std::size_t>(i)]);
      Result<void> r = problem.rhs(t_stage, y_stage, k[static_cast<std::size_t>(i)]);
      ++fevals;
      if (!r.ok()) return r.error();
    }

    for (std::size_t m = 0; m < n; ++m) y_new[m] = state[m];
    for (int i = 0; i < stages; ++i) {
      const double bi = tableau_.B[static_cast<std::size_t>(i)];
      if (bi == 0.0) continue;
      for (std::size_t m = 0; m < n; ++m) {
        y_new[m] = y_new[m] + (h * static_cast<Scalar>(bi)) * k[static_cast<std::size_t>(i)][m];
      }
    }

    StepRecord<Value> rec;
    rec.function_evaluations = fevals;

    if (!controller_.has_value()) {
      // Fixed step: always accepted. h_next is h unchanged UNLESS a
      // DynamicalTimeScaleController is attached, in which case this is a
      // KARMA-mechanism-1 step (Section 2 item 1): still always accepted (this
      // mechanism has no accept/reject notion of its own), but h_next is the
      // dynamical proposal rather than a literal echo of h.
      for (std::size_t m = 0; m < n; ++m) state[m] = y_new[m];
      rec.t_new = t + h;
      rec.h_used = h;
      rec.accepted = true;
      if (fsal_structure_ok) update_fsal_cache_(rec.t_new, y_new, k[static_cast<std::size_t>(stages - 1)]);
      if (dynamical_controller_.has_value()) {
        Result<double> h_dyn = propose_dynamical_h_(problem, t, std::span<const Value>(state), h);
        if (!h_dyn.ok()) return h_dyn.error();
        rec.h_next = static_cast<Scalar>(h_dyn.value());
      } else {
        rec.h_next = h;
      }
      return rec;
    }

    if (!tableau_.has_embedded_estimate()) {
      return Failure{kWrongFamilyCode,
                     "ExplicitIntegrator::step: an EmbeddedController was supplied but "
                     "tableau '" + tableau_.name + "' has no usable embedded estimate "
                     "(ButcherTableau::has_embedded_estimate() is false)"};
    }

    // Scaled embedded error: err[m] = h * sum_i E[i]*k[i][m], scaled by Hairer's own
    // rtol/atol convention (dopri45_integrator.C:212-217), then reduced by the
    // caller-supplied norm functor (this header's own top comment).
    std::vector<double> scaled(n);
    for (std::size_t m = 0; m < n; ++m) {
      Value err_m{};
      for (int i = 0; i < stages; ++i) {
        const double ei = (*tableau_.E)[static_cast<std::size_t>(i)];
        if (ei == 0.0) continue;
        err_m = err_m + (h * static_cast<Scalar>(ei)) * k[static_cast<std::size_t>(i)][m];
      }
      const double scale = static_cast<double>(atol_) +
                           static_cast<double>(rtol_) * std::fabs(static_cast<double>(y_new[m]));
      scaled[m] = (scale > 0.0) ? static_cast<double>(err_m) / scale : static_cast<double>(err_m);
    }
    const double err_norm = norm_(scaled);

    Result<StepDecision> decision = controller_->propose(err_norm, static_cast<double>(h), *controller_state_);
    if (!decision.ok()) return decision.error();

    rec.error_norm = static_cast<Scalar>(err_norm);
    rec.h_next = static_cast<Scalar>(decision.value().h_next);
    if (decision.value().outcome == StepOutcome::Accepted) {
      for (std::size_t m = 0; m < n; ++m) state[m] = y_new[m];
      rec.t_new = t + h;
      rec.h_used = h;
      rec.accepted = true;
      if (fsal_structure_ok) update_fsal_cache_(rec.t_new, y_new, k[static_cast<std::size_t>(stages - 1)]);
      // Round 3, item 5: with BOTH controllers attached, the accept/reject call
      // stays the embedded controller's (the dynamical mechanism has none of its
      // own, Section 2 item 1's own prose); on acceptance, h_next is the MINIMUM
      // of the two proposals (Round 3's own combination rule -- open_questions).
      // Evaluated on the NEW (post-step) state: the dynamical scale a caller wants
      // bounding the NEXT step is the one at the state that step is taken FROM.
      if (dynamical_controller_.has_value()) {
        Result<double> h_dyn = propose_dynamical_h_(problem, rec.t_new, std::span<const Value>(state), h);
        if (!h_dyn.ok()) return h_dyn.error();
        rec.h_next = std::min(rec.h_next, static_cast<Scalar>(h_dyn.value()));
      }
    } else {
      // Rejected: state is left untouched (this class's own contract, integrator.hpp's
      // own doc comment on Integrator::step). The dynamical controller is not
      // consulted on a rejected step: it has no rejection notion to combine with.
      rec.t_new = t;
      rec.h_used = Scalar{0};
      rec.accepted = false;
    }
    return rec;
  }

 private:
  /// Round 3, item 5: one evaluation of the attached DynamicalTimeScaleController,
  /// shared by both the "dynamical-only" (no EmbeddedController) and the "both
  /// present" branches of step() below. Refuses (not silently falls back) if a
  /// DynamicalTimeScaleController is attached but the problem supplies no
  /// `dynamical_time_scales` callback -- a misconfiguration, not a legitimate
  /// "no information" case (that case is the controller's OWN refusal, raised from
  /// inside DynamicalTimeScaleController::propose when every reported scale is
  /// non-positive).
  [[nodiscard]] Result<double> propose_dynamical_h_(TemporalProblem<Value>& problem, Scalar t,
                                                     std::span<const Value> y, Scalar h) {
    if (!problem.has_dynamical_time_scales()) {
      return Failure{kNoDynamicalScalesCode,
                     "ExplicitIntegrator::step: a DynamicalTimeScaleController was "
                     "attached (set_dynamical_controller) but "
                     "TemporalProblem::dynamical_time_scales is unset"};
    }
    dynamical_scales_scratch_.clear();
    Result<void> r = problem.dynamical_time_scales(t, y, dynamical_scales_scratch_);
    if (!r.ok()) return r.error();
    return dynamical_controller_->propose(dynamical_scales_scratch_, static_cast<double>(h));
  }

  /// Round 3, item 6: records the just-accepted step's own last-stage derivative
  /// (an EXACT f() evaluation for an ERK tableau, not a Newton-converged one) so
  /// the NEXT call's stage-0 evaluation can reuse it bit-for-bit.
  void update_fsal_cache_(Scalar t_new, const std::vector<Value>& y_new,
                          const std::vector<Value>& k_last) {
    fsal_cached_t_ = t_new;
    fsal_cached_y_ = y_new;
    fsal_cached_k_ = k_last;
  }

  ButcherTableau tableau_;
  std::optional<EmbeddedController> controller_;
  std::optional<ControllerState> controller_state_;
  std::optional<DynamicalTimeScaleController> dynamical_controller_;
  std::vector<double> dynamical_scales_scratch_;
  Scalar rtol_{};
  Scalar atol_{};
  ScaledNormFn norm_ = serial_rms_norm;

  // Round 3, item 6: FSAL reuse for an explicit (ERK) FSAL tableau (DOPRI45).
  // Unlike DirkIntegrator's own implicit-family attempt (dirk_integrator.hpp's own
  // set_fsal_enabled() comment), an ERK method's last stage is an EXACT,
  // non-iterative f() evaluation -- no Newton solve, so no convergence-tolerance
  // gap of any kind. MEASURED bit-identical (explicit_integrator_test.cpp's own
  // FSAL suite): DOPRI45's own B[6] == 0.0 and A[6] == B (butcher_tableau.hpp),
  // so the last stage's own y_stage accumulation and y_new's own b-weighted sum
  // are built from the IDENTICAL set of nonzero terms in the IDENTICAL order,
  // starting from the IDENTICAL `state` -- the same floating-point expression,
  // not merely an algebraically equivalent one, so they round identically BY
  // CONSTRUCTION (unlike the DIRK case, where the diagonal implicit term made the
  // two expressions genuinely different code paths). Enabled by default.
  bool fsal_enabled_ = true;
  std::optional<Scalar> fsal_cached_t_;
  std::optional<std::vector<Value>> fsal_cached_y_;
  std::optional<std::vector<Value>> fsal_cached_k_;
};

}  // namespace fuse::num
