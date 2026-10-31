#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/num/butcher_tableau.hpp"
#include "fuse/num/embedded_controller.hpp"
#include "fuse/num/integrator.hpp"
#include "fuse/num/newton_krylov.hpp"

/// The DIRK/ESDIRK family: "a zero diagonal entry is an explicit stage; stiffly
/// accurate tableaus give FSAL" -- ONE family handles SDIRK (every diagonal nonzero)
/// and ESDIRK (first diagonal zero) alike, since a zero `A[i][i]` makes stage `i`'s own
/// nonlinear solve trivial (it collapses to a plain function evaluation) without any
/// separate code path.
///
/// Stage equation (Hairer & Wanner, Solving ODEs II, IV.6): for stage `i`,
///   stage_y_i = y + h * sum_{j<=i} A[i][j] * k_j
///   M * k_i = f(t_i, stage_y_i),  t_i = t + h * C[i]
/// with `k_i` the unknown when `A[i][i] != 0` (an IMPLICIT stage) and already known by
/// substitution when `A[i][i] == 0` (an EXPLICIT one). The implicit
/// case is solved for `k_i` directly (not `stage_y_i`) via `fuse::num::solve()` -- DIRK
/// stages go through `fuse::num::solve` -- on the
/// residual `F(k_i) = M*k_i - f(t_i, y_known + h*A[i][i]*k_i)`, with `y_known` the
/// ALREADY-COMPUTED `y + h*sum_{j<i} A[i][j] k_j` held fixed for that one stage's own
/// Newton iteration.
///
/// NOT wired: `TemporalProblem::jacobian_action`. `fuse::num::solve()`'s own
/// signature (newton_krylov.hpp) takes a `ResidualFn` ALONE -- mechanism 1 (MFFD) is
/// internal to `solve()` and differences whatever residual function it is given; there
/// is no parameter through which an analytic `jacobian_action` could replace that
/// internal MFFD. Ideally `jacobian_action()` would be used when present, else MFFD;
/// since the DIRK family is built on `fuse::num::solve()` specifically (as opposed to
/// Rosenbrock's own lower-level `gmres`/`MatrixFreeOperator` path,
/// where an analytic Jacobian plugs in naturally), this DIRK integrator always
/// uses `solve()`'s own internal MFFD, regardless of whether `problem.jacobian_action`
/// is set -- an open question (new `solve()` overload, or routing DIRK through
/// the lower-level path when an analytic Jacobian exists) rather than silently assumed
/// resolved.
///
/// `TemporalProblem::dfdt` is NOT called by this family either: a genuine DIRK stage
/// evaluates `f` AT `t_i = t + h*C[i]` directly (no frozen-Jacobian linearisation the
/// way Rosenbrock's own non-autonomous correction needs `h*gamma_i*f_t` for), so `C`'s
/// own role here is exactly "feed the right `t_i` into every stage's own `rhs` call" --
/// the requirement that C defects be visible is
/// answered by `dirk_integrator_test.cpp`'s own non-autonomous problem, not by a dfdt
/// call site this family has no mathematical need for.
///
/// NOT implemented: the FSAL reuse available for stiffly accurate tableaus (the
/// last stage's own `k` IS the next step's own first, explicit stage's `k`, since
/// `t + h*C[last] == t_new` and `stage_y[last] == y_new` for a stiffly accurate
/// tableau). `step()` below recomputes stage 0 every call -- numerically IDENTICAL,
/// one extra function evaluation per step -- for the SAME reason `ExplicitIntegrator`
/// skips the equivalent optimisation for DOPRI45 (robustness and accuracy beat
/// efficiency).
/// Declared as a deviation, not a silent omission.
namespace fuse::num {

inline constexpr const char* kNewtonFailedCode = "E-INTEGRATOR-NEWTON-FAILED";

template <IntegratorValue Value>
class DirkIntegrator final : public Integrator<Value> {
 public:
  using Scalar = typename Integrator<Value>::Scalar;

  /// Fixed-step construction (any tableau with family Dirk/Sdirk -- no embedded pair).
  explicit DirkIntegrator(ButcherTableau tableau, NewtonKrylovOptions newton_opts = {})
      : tableau_(std::move(tableau)), newton_opts_(newton_opts) {}

  /// Adaptive construction (DirkEmbedded/SdirkEmbedded -- e.g. ESDIRK324L2SA,
  /// SDIRK332). Mirrors ExplicitIntegrator's own adaptive constructor exactly (same
  /// controller, same rtol/atol/norm convention).
  DirkIntegrator(ButcherTableau tableau, EmbeddedController controller, Scalar rtol,
                 Scalar atol, ScaledNormFn norm = serial_rms_norm,
                 NewtonKrylovOptions newton_opts = {})
      : tableau_(std::move(tableau)),
        controller_(std::move(controller)),
        controller_state_(ControllerState{}),
        rtol_(rtol),
        atol_(atol),
        norm_(std::move(norm)),
        newton_opts_(newton_opts) {}

  [[nodiscard]] const ButcherTableau& tableau() const noexcept override { return tableau_; }

  [[nodiscard]] std::optional<ControllerState>& controller_state() noexcept {
    return controller_state_;
  }

  /// The last step's own per-stage Newton iteration counts, exposed so a test can
  /// confirm the implicit stages actually ran a nonlinear solve (not merely that the
  /// answer came out numerically right by accident) and so a caller can report solver
  /// cost.
  [[nodiscard]] const std::vector<int>& last_step_newton_iterations() const noexcept {
    return newton_iterations_;
  }

  [[nodiscard]] Result<StepRecord<Value>> step(TemporalProblem<Value>& problem, Scalar t,
                                               Scalar h, std::span<Value> state) override {
    if (!problem.has_rhs()) {
      return Failure{kNoRhsCode, "DirkIntegrator::step: TemporalProblem::rhs is unset"};
    }
    if (tableau_.family != ButcherFamily::Dirk && tableau_.family != ButcherFamily::DirkEmbedded &&
        tableau_.family != ButcherFamily::Sdirk && tableau_.family != ButcherFamily::SdirkEmbedded) {
      return Failure{kWrongFamilyCode,
                     "DirkIntegrator::step: tableau '" + tableau_.name + "' is family " +
                     to_cstring(tableau_.family) + ", not Dirk/DirkEmbedded/Sdirk/SdirkEmbedded"};
    }
    if (problem.dimension != 0 && problem.dimension != state.size()) {
      return Failure{kDimensionMismatchCode,
                     "DirkIntegrator::step: state.size()=" + std::to_string(state.size()) +
                     " != problem.dimension=" + std::to_string(problem.dimension)};
    }

    const int stages = tableau_.stages;
    const std::size_t n = state.size();
    const auto exchanger = fuse::linalg::make_serial_exchanger();
    const auto reducer = fuse::linalg::make_serial_reducer();

    std::vector<std::vector<double>> k(static_cast<std::size_t>(stages), std::vector<double>(n, 0.0));
    newton_iterations_.assign(static_cast<std::size_t>(stages), 0);
    std::size_t fevals = 0;

    for (int i = 0; i < stages; ++i) {
      const double aii = tableau_.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
      std::vector<double> y_known(n);
      for (std::size_t m = 0; m < n; ++m) y_known[m] = static_cast<double>(state[m]);
      for (int j = 0; j < i; ++j) {
        const double aij = tableau_.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        if (aij == 0.0) continue;
        for (std::size_t m = 0; m < n; ++m) {
          y_known[m] += static_cast<double>(h) * aij * k[static_cast<std::size_t>(j)][m];
        }
      }
      const Scalar t_stage = t + h * static_cast<Scalar>(tableau_.C[static_cast<std::size_t>(i)]);

      if (aii == 0.0) {
        // Explicit stage: k_i = f(t_i, y_known) directly, or the
        // linear mass-solve M k_i = f(...) if a mass matrix is present (not exercised
        // by any test problem -- every one uses M = I -- so this path is
        // UNTESTED for M != I and flagged in the design doc as an open question rather
        // than silently assumed correct).
        std::vector<Value> stage_y(n), rhs_val(n);
        for (std::size_t m = 0; m < n; ++m) stage_y[m] = static_cast<Value>(y_known[m]);
        Result<void> r = problem.rhs(t_stage, std::span<const Value>(stage_y), std::span<Value>(rhs_val));
        ++fevals;
        if (!r.ok()) return r.error();
        if (problem.has_mass()) {
          return Failure{kWrongFamilyCode,
                         "DirkIntegrator::step: an explicit (zero-diagonal) stage with a "
                         "mass matrix needs a linear mass-solve that is not "
                         "implemented (no test problem exercises M != I)"};
        }
        for (std::size_t m = 0; m < n; ++m) k[static_cast<std::size_t>(i)][m] = static_cast<double>(rhs_val[m]);
        continue;
      }

      // Implicit stage: solve M*k_i - f(t_i, y_known + h*aii*k_i) = 0 for k_i.
      const double h_aii = static_cast<double>(h) * aii;
      fuse::linalg::DistVector x(n, exchanger, reducer);
      x.set_zero();
      std::size_t stage_fevals = 0;
      auto residual_fn = [&](fuse::linalg::DistVector& xv,
                             fuse::linalg::DistVector& Fx) -> Result<void> {
        std::vector<Value> stage_y(n), rhs_val(n);
        for (std::size_t m = 0; m < n; ++m) {
          stage_y[m] = static_cast<Value>(y_known[m] + h_aii * xv.data()[m]);
        }
        Result<void> r = problem.rhs(t_stage, std::span<const Value>(stage_y), std::span<Value>(rhs_val));
        ++stage_fevals;
        if (!r.ok()) return r.error();
        if (problem.has_mass()) {
          std::vector<Value> xval(n), mass_val(n);
          for (std::size_t m = 0; m < n; ++m) xval[m] = static_cast<Value>(xv.data()[m]);
          Result<void> rm = problem.mass(std::span<const Value>(xval), std::span<Value>(mass_val));
          if (!rm.ok()) return rm.error();
          for (std::size_t m = 0; m < n; ++m) {
            Fx.data()[m] = static_cast<double>(mass_val[m]) - static_cast<double>(rhs_val[m]);
          }
        } else {
          for (std::size_t m = 0; m < n; ++m) Fx.data()[m] = xv.data()[m] - static_cast<double>(rhs_val[m]);
        }
        return Result<void>::success();
      };

      Result<NewtonResult> nr = fuse::num::solve(residual_fn, x, nullptr, nullptr, newton_opts_);
      fevals += stage_fevals;
      if (!nr.ok()) return nr.error();
      newton_iterations_[static_cast<std::size_t>(i)] = nr.value().iterations;
      if (is_diverged(nr.value().status)) {
        return Failure{kNewtonFailedCode,
                       "DirkIntegrator::step: stage " + std::to_string(i) + " of tableau '" +
                       tableau_.name + "' Newton solve diverged (" +
                       std::string(to_cstring(nr.value().status)) + ") after " +
                       std::to_string(nr.value().iterations) + " iterations, final ||F|| = " +
                       std::to_string(nr.value().final_residual_norm)};
      }
      for (std::size_t m = 0; m < n; ++m) k[static_cast<std::size_t>(i)][m] = x.data()[m];
    }

    std::vector<Value> y_new(n);
    for (std::size_t m = 0; m < n; ++m) y_new[m] = state[m];
    for (int i = 0; i < stages; ++i) {
      const double bi = tableau_.B[static_cast<std::size_t>(i)];
      if (bi == 0.0) continue;
      for (std::size_t m = 0; m < n; ++m) {
        y_new[m] = y_new[m] + (h * static_cast<Scalar>(bi)) * static_cast<Value>(k[static_cast<std::size_t>(i)][m]);
      }
    }

    StepRecord<Value> rec;
    rec.function_evaluations = fevals;

    if (!controller_.has_value()) {
      for (std::size_t m = 0; m < n; ++m) state[m] = y_new[m];
      rec.t_new = t + h;
      rec.h_used = h;
      rec.h_next = h;
      rec.accepted = true;
      return rec;
    }

    if (!tableau_.has_embedded_estimate()) {
      return Failure{kWrongFamilyCode,
                     "DirkIntegrator::step: an EmbeddedController was supplied but tableau '" +
                     tableau_.name + "' has no usable embedded estimate"};
    }
    std::vector<double> scaled(n);
    for (std::size_t m = 0; m < n; ++m) {
      double err_m = 0.0;
      for (int i = 0; i < stages; ++i) {
        const double ei = (*tableau_.E)[static_cast<std::size_t>(i)];
        if (ei == 0.0) continue;
        err_m += static_cast<double>(h) * ei * k[static_cast<std::size_t>(i)][m];
      }
      const double scale = static_cast<double>(atol_) +
                           static_cast<double>(rtol_) * std::fabs(static_cast<double>(y_new[m]));
      scaled[m] = (scale > 0.0) ? err_m / scale : err_m;
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
  NewtonKrylovOptions newton_opts_;
  std::vector<int> newton_iterations_;
};

}  // namespace fuse::num
