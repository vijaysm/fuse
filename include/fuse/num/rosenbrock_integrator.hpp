#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/gmres.hpp"
#include "fuse/linalg/operator.hpp"
#include "fuse/linalg/preconditioner.hpp"
#include "fuse/num/butcher_tableau.hpp"
#include "fuse/num/embedded_controller.hpp"
#include "fuse/num/integrator.hpp"
#include "fuse/num/newton_krylov.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 2, task item 3: the Rosenbrock
/// family.
///
/// IMPLEMENTATION NOTE on which equation this header solves (a correction against the
/// task's own prose formula, MEASURED, not a judgement call): the task's literal
/// "(I/(h gamma) - J) k_i = f(y + sum_j a_ij k_j) + sum_j (c_ij/h) k_j" was coded
/// first and found, by direct numerical comparison against the SAME tableau's own
/// independently-validated stability function (`rosenbrock_classical_form()` +
/// `stability_function()`, butcher_tableau_test.cpp -- already proven to match
/// published R(-inf) values for five tableaus), to give a DIFFERENT one-step result
/// than the validated R(z)*y0 prediction on ROS22/Dahlquist (0.96542+0.05012i vs the
/// correct 0.97976+0.02940i at h=0.01) -- confirmed by hand re-deriving both sides in
/// exact arithmetic (recorded in the Round 2 submission's own measured_results, not
/// repeated here). This header instead implements the CLASSICAL Rosenbrock/ROW-method
/// recurrence directly (Hairer & Wanner, Solving ODEs II, IV.7, eq. (7.2)-(7.4)) on
/// the RECOVERED (alpha, Gamma, b) triple (the SAME recovery `rosenbrock_classical_form`
/// already performs for the stability/order tests, extended here to expose `alpha` and
/// the full `Gamma` separately rather than only their sum):
///
///   (I - h*gamma*J) k_i = h * [ f(t_i, y + sum_{j<i} alpha_ij k_j)
///                                + J * sum_{j<i} Gamma_ij k_j
///                                + h * gamma_i * f_t ]   (non-autonomous correction)
///   y_new = y + sum_i b_i k_i
///
/// with `gamma_i = sum_{j<=i} Gamma_ij` (the row sum INCLUDING the diagonal). This
/// recurrence was verified end to end (hand arithmetic in Python, then this header's
/// own compiled code) to reproduce the validated R(z)*y0 prediction exactly at
/// h = 0.01 on ROS22/Dahlquist before being trusted for the order tests.
///
/// J is the Jacobian, held FIXED for the WHOLE step (owner's standing decision, task
/// item 3: "one MatrixFreeOperator set_point plus gmres per stage"): `problem.
/// jacobian_action`, evaluated at `(t, state)`, when present; `fuse::num::
/// MffdJacobianAction` (reused directly from newton_krylov.hpp, wrapping `problem.rhs`
/// at FIXED `t`) otherwise -- task item 2's own "jacobian_action() used when present
/// else MFFD" is satisfied here exactly as asked (DIRK's own header explains why
/// `fuse::num::solve()`'s fixed signature cannot take one there).
///
/// Each stage's own linear solve runs `fuse::linalg::gmres` directly against a
/// `MatrixFreeOperator` built from `I - h*gamma*J`, identity-preconditioned, at
/// `GmresOptions::rel_tol = step_rtol * stage_tolerance_factor` -- owner decision (5),
/// Stage A design Section 7: "stage GMRES rtol = step rtol times a fixed factor
/// starting at 1e-2, verified by the per-method order test, tightened never loosened".
namespace fuse::num {

using fuse::linalg::DistVector;
using fuse::linalg::GmresOptions;
using fuse::linalg::IdentityPreconditioner;
using fuse::linalg::MatrixFreeOperator;

inline constexpr const char* kStageSolveFailedCode = "E-INTEGRATOR-STAGE-SOLVE-FAILED";

/// Recovers `(alpha, Gamma, b)` separately -- `rosenbrock_classical_form()`
/// (butcher_tableau.hpp) performs the SAME inverse map but returns only
/// `M = alpha + Gamma` and `b` (enough for a stability/order check, not enough to run
/// the classical recurrence stage by stage, which needs `alpha` and `Gamma` on their
/// own). Returns `std::nullopt` for a non-Rosenbrock tableau.
struct RosenbrockClassicalTriple {
  std::vector<std::vector<double>> alpha;
  std::vector<std::vector<double>> Gamma;
  std::vector<double> b;
};

[[nodiscard]] inline std::optional<RosenbrockClassicalTriple> rosenbrock_classical_triple(
    const ButcherTableau& t) {
  if (!t.gamma.has_value()) return std::nullopt;
  const std::size_t n = static_cast<std::size_t>(t.stages);
  const double g = (*t.gamma)[0][0];
  std::vector<std::vector<double>> ginv(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      ginv[i][j] = ((i == j) ? 1.0 : -(*t.gamma)[i][j]) / g;
    }
  }
  std::vector<std::vector<double>> gam(n, std::vector<double>(n, 0.0));
  for (std::size_t j = 0; j < n; ++j) {
    gam[j][j] = 1.0 / ginv[j][j];
    for (std::size_t i = j + 1; i < n; ++i) {
      double s = 0.0;
      for (std::size_t k = j; k < i; ++k) s += ginv[i][k] * gam[k][j];
      gam[i][j] = -s / ginv[i][i];
    }
  }
  std::vector<std::vector<double>> alpha(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double a = 0.0;
      for (std::size_t k = 0; k < n; ++k) a += t.A[i][k] * gam[k][j];
      alpha[i][j] = a;
    }
  }
  std::vector<double> b(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += gam[i][j] * t.B[i];
    b[j] = s;
  }
  return RosenbrockClassicalTriple{alpha, gam, b};
}

template <IntegratorValue Value>
class RosenbrockIntegrator final : public Integrator<Value> {
 public:
  using Scalar = typename Integrator<Value>::Scalar;

  RosenbrockIntegrator(ButcherTableau tableau, Scalar step_rtol, double stage_tolerance_factor = 1e-2)
      : tableau_(std::move(tableau)), step_rtol_(step_rtol), stage_tolerance_factor_(stage_tolerance_factor) {}

  RosenbrockIntegrator(ButcherTableau tableau, EmbeddedController controller, Scalar rtol,
                      Scalar atol, ScaledNormFn norm = serial_rms_norm,
                      double stage_tolerance_factor = 1e-2)
      : tableau_(std::move(tableau)),
        controller_(std::move(controller)),
        controller_state_(ControllerState{}),
        rtol_(rtol),
        atol_(atol),
        norm_(std::move(norm)),
        step_rtol_(rtol),
        stage_tolerance_factor_(stage_tolerance_factor) {}

  [[nodiscard]] const ButcherTableau& tableau() const noexcept override { return tableau_; }

  [[nodiscard]] std::optional<ControllerState>& controller_state() noexcept {
    return controller_state_;
  }

  /// Budget of Arnoldi steps for each stage's GMRES solve (GmresOptions::
  /// max_iterations; default 1000, the solver's own). A stage whose solve
  /// exhausts it is refused with E-INTEGRATOR-STAGE-SOLVE-FAILED rather than
  /// folded into the step: the test RosenbrockIntegrator.
  /// StageSolveFailureIsRefusedNotFoldedIntoTheStep fires that refusal with a
  /// budget of one step on a 2x2 system with complex eigenvalues, where no real
  /// one-dimensional Krylov space can reach the stage tolerance (lead,
  /// 2026-10-31; the refusal's mutation row survived round 2 without this).
  void set_stage_gmres_max_iterations(std::size_t n) noexcept { stage_gmres_max_iterations_ = n; }

  [[nodiscard]] const std::vector<std::size_t>& last_step_gmres_iterations() const noexcept {
    return gmres_iterations_;
  }

  [[nodiscard]] Result<StepRecord<Value>> step(TemporalProblem<Value>& problem, Scalar t,
                                               Scalar h, std::span<Value> state) override {
    if (!problem.has_rhs()) {
      return Failure{kNoRhsCode, "RosenbrockIntegrator::step: TemporalProblem::rhs is unset"};
    }
    if (tableau_.family != ButcherFamily::Rosenbrock && tableau_.family != ButcherFamily::RosenbrockEmbedded) {
      return Failure{kWrongFamilyCode,
                     "RosenbrockIntegrator::step: tableau '" + tableau_.name + "' is family " +
                     to_cstring(tableau_.family) + ", not Rosenbrock/RosenbrockEmbedded"};
    }
    const auto triple = rosenbrock_classical_triple(tableau_);
    if (!triple.has_value()) {
      return Failure{kWrongFamilyCode,
                     "RosenbrockIntegrator::step: tableau '" + tableau_.name +
                     "' has no stored gamma matrix (not a Rosenbrock tableau's own data)"};
    }
    if (problem.dimension != 0 && problem.dimension != state.size()) {
      return Failure{kDimensionMismatchCode,
                     "RosenbrockIntegrator::step: state.size()=" + std::to_string(state.size()) +
                     " != problem.dimension=" + std::to_string(problem.dimension)};
    }

    const int stages = tableau_.stages;
    const std::size_t n = state.size();
    const double gamma = (*tableau_.gamma)[0][0];
    const auto& alpha = triple->alpha;
    const auto& Gam = triple->Gamma;
    const auto& b_classical = triple->b;
    const auto exchanger = fuse::linalg::make_serial_exchanger();
    const auto reducer = fuse::linalg::make_serial_reducer();

    DistVector x0(n, exchanger, reducer);
    for (std::size_t m = 0; m < n; ++m) x0.data()[m] = static_cast<double>(state[m]);
    std::vector<Value> f_at_state(n);
    std::size_t fevals = 1;
    {
      Result<void> r = problem.rhs(t, state, std::span<Value>(f_at_state));
      if (!r.ok()) return r.error();
    }
    DistVector fx0(n, exchanger, reducer);
    for (std::size_t m = 0; m < n; ++m) fx0.data()[m] = static_cast<double>(f_at_state[m]);

    std::optional<MffdJacobianAction> mffd;
    auto jacobian_apply = [&](DistVector& v, DistVector& jv) -> Result<void> {
      if (problem.has_jacobian_action()) {
        std::vector<Value> vval(n), jval(n);
        for (std::size_t m = 0; m < n; ++m) vval[m] = static_cast<Value>(v.data()[m]);
        Result<void> r = problem.jacobian_action(t, state, std::span<const Value>(vval), std::span<Value>(jval));
        ++fevals;
        if (!r.ok()) return r.error();
        for (std::size_t m = 0; m < n; ++m) jv.data()[m] = static_cast<double>(jval[m]);
        return Result<void>::success();
      }
      ++fevals;
      return mffd->apply(v, jv);
    };
    if (!problem.has_jacobian_action()) {
      ResidualFn f_frozen_t = [&](DistVector& xv, DistVector& Fx) -> Result<void> {
        std::vector<Value> xval(n), fval(n);
        for (std::size_t m = 0; m < n; ++m) xval[m] = static_cast<Value>(xv.data()[m]);
        Result<void> r = problem.rhs(t, std::span<const Value>(xval), std::span<Value>(fval));
        if (!r.ok()) return r.error();
        for (std::size_t m = 0; m < n; ++m) Fx.data()[m] = static_cast<double>(fval[m]);
        return Result<void>::success();
      };
      mffd.emplace(f_frozen_t);
      mffd->set_point(x0, fx0);
    }

    // (I - h*gamma*J) v
    MatrixFreeOperator lin_op([&](DistVector& v, DistVector& out) -> Result<void> {
      DistVector jv(n, exchanger, reducer);
      Result<void> r = jacobian_apply(v, jv);
      if (!r.ok()) return r.error();
      for (std::size_t m = 0; m < n; ++m) {
        out.data()[m] = v.data()[m] - static_cast<double>(h) * gamma * jv.data()[m];
      }
      return Result<void>::success();
    });

    GmresOptions gmres_opt;
    gmres_opt.rel_tol = static_cast<double>(step_rtol_) * stage_tolerance_factor_;
    gmres_opt.max_iterations = stage_gmres_max_iterations_;

    std::vector<std::vector<double>> k(static_cast<std::size_t>(stages), std::vector<double>(n, 0.0));
    gmres_iterations_.assign(static_cast<std::size_t>(stages), 0);

    for (int i = 0; i < stages; ++i) {
      std::vector<Value> y_arg(n);
      for (std::size_t m = 0; m < n; ++m) y_arg[m] = state[m];
      for (int j = 0; j < i; ++j) {
        const double aij = alpha[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        if (aij != 0.0) {
          for (std::size_t m = 0; m < n; ++m) {
            y_arg[m] = y_arg[m] + static_cast<Scalar>(aij) * static_cast<Value>(k[static_cast<std::size_t>(j)][m]);
          }
        }
      }
      const Scalar t_stage = t + h * static_cast<Scalar>(tableau_.C[static_cast<std::size_t>(i)]);
      std::vector<Value> f_val(n);
      {
        Result<void> r = problem.rhs(t_stage, std::span<const Value>(y_arg), std::span<Value>(f_val));
        ++fevals;
        if (!r.ok()) return r.error();
      }

      std::vector<double> gam_sum(n, 0.0);
      double gamma_i = Gam[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
      for (int j = 0; j < i; ++j) {
        const double gij = Gam[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        gamma_i += gij;
        if (gij != 0.0) {
          for (std::size_t m = 0; m < n; ++m) gam_sum[m] += gij * k[static_cast<std::size_t>(j)][m];
        }
      }
      DistVector gam_sum_vec(n, exchanger, reducer);
      for (std::size_t m = 0; m < n; ++m) gam_sum_vec.data()[m] = gam_sum[m];
      DistVector j_gam_sum(n, exchanger, reducer);
      {
        Result<void> r = jacobian_apply(gam_sum_vec, j_gam_sum);
        if (!r.ok()) return r.error();
      }

      std::vector<double> rhs_vec(n);
      for (std::size_t m = 0; m < n; ++m) {
        rhs_vec[m] = static_cast<double>(h) * (static_cast<double>(f_val[m]) + j_gam_sum.data()[m]);
      }
      if (problem.has_dfdt()) {
        std::vector<Value> dfdt_val(n);
        Result<void> r = problem.dfdt(t, state, std::span<Value>(dfdt_val));
        ++fevals;
        if (!r.ok()) return r.error();
        for (std::size_t m = 0; m < n; ++m) {
          rhs_vec[m] += static_cast<double>(h) * static_cast<double>(h) * gamma_i * static_cast<double>(dfdt_val[m]);
        }
      }

      DistVector rhs_dv(n, exchanger, reducer);
      for (std::size_t m = 0; m < n; ++m) rhs_dv.data()[m] = rhs_vec[m];
      DistVector ki(n, exchanger, reducer);
      ki.set_zero();
      Result<fuse::linalg::SolverReport> gr =
          fuse::linalg::gmres(lin_op, IdentityPreconditioner{}, ki, rhs_dv, gmres_opt);
      if (!gr.ok()) {
        return Failure{kStageSolveFailedCode,
                       "RosenbrockIntegrator::step: stage " + std::to_string(i) + " of tableau '" +
                       tableau_.name + "' GMRES solve failed: " + gr.error().code + ": " +
                       gr.error().message};
      }
      gmres_iterations_[static_cast<std::size_t>(i)] = gr.value().iterations;
      for (std::size_t m = 0; m < n; ++m) k[static_cast<std::size_t>(i)][m] = ki.data()[m];
    }

    std::vector<Value> y_new(n);
    for (std::size_t m = 0; m < n; ++m) y_new[m] = state[m];
    for (int i = 0; i < stages; ++i) {
      const double bi = b_classical[static_cast<std::size_t>(i)];
      if (bi == 0.0) continue;
      for (std::size_t m = 0; m < n; ++m) {
        y_new[m] = y_new[m] + static_cast<Scalar>(bi) * static_cast<Value>(k[static_cast<std::size_t>(i)][m]);
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
                     "RosenbrockIntegrator::step: an EmbeddedController was supplied but tableau '" +
                     tableau_.name + "' has no usable embedded estimate"};
    }
    // E = B - B_hat is on the STORED weights; since y_new = y + sum_i b_classical[i]*
    // k_classical[i] and b_classical = Gamma^T * B_stored (the SAME linear transform
    // this integrator already applied to get b_classical from B_stored), the
    // embedded error in the CLASSICAL basis is E_classical = Gamma^T * E_stored,
    // applied to the SAME k_classical this integrator already computed -- no need to
    // convert k back to the stored basis: err = sum_i E_classical[i] * k_classical[i],
    // exactly mirroring how y_new itself was built from b_classical and k_classical.
    std::vector<double> e_classical(static_cast<std::size_t>(stages), 0.0);
    for (int j = 0; j < stages; ++j) {
      double s = 0.0;
      for (int i = 0; i < stages; ++i) {
        s += Gam[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
             (*tableau_.E)[static_cast<std::size_t>(i)];
      }
      e_classical[static_cast<std::size_t>(j)] = s;
    }
    std::vector<double> scaled(n);
    for (std::size_t m = 0; m < n; ++m) {
      double err_m = 0.0;
      for (int i = 0; i < stages; ++i) {
        const double ei = e_classical[static_cast<std::size_t>(i)];
        if (ei == 0.0) continue;
        err_m += ei * k[static_cast<std::size_t>(i)][m];
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
  Scalar step_rtol_{};
  double stage_tolerance_factor_ = 1e-2;
  std::size_t stage_gmres_max_iterations_ = 1000;
  std::vector<std::size_t> gmres_iterations_;
};

}  // namespace fuse::num
