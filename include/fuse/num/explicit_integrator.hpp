#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/num/butcher_tableau.hpp"
#include "fuse/num/embedded_controller.hpp"
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
/// `EmbeddedController::propose` already names.
inline constexpr const char* kNoRhsCode = "E-INTEGRATOR-NO-RHS";
inline constexpr const char* kMassUnsupportedCode = "E-INTEGRATOR-MASS-UNSUPPORTED";
inline constexpr const char* kWrongFamilyCode = "E-INTEGRATOR-WRONG-FAMILY";
inline constexpr const char* kDimensionMismatchCode = "E-INTEGRATOR-DIMENSION-MISMATCH";

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

    for (int i = 0; i < stages; ++i) {
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
      // Fixed step: always accepted, h unchanged.
      for (std::size_t m = 0; m < n; ++m) state[m] = y_new[m];
      rec.t_new = t + h;
      rec.h_used = h;
      rec.h_next = h;
      rec.accepted = true;
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
    } else {
      // Rejected: state is left untouched (this class's own contract, integrator.hpp's
      // own doc comment on Integrator::step).
      rec.t_new = t;
      rec.h_used = Scalar{0};
      rec.accepted = false;
    }
    return rec;
  }

 private:
  ButcherTableau tableau_;
  std::optional<EmbeddedController> controller_;
  std::optional<ControllerState> controller_state_;
  Scalar rtol_{};
  Scalar atol_{};
  ScaledNormFn norm_ = serial_rms_norm;
};

}  // namespace fuse::num
