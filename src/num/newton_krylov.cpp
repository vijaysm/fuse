#include "fuse/num/newton_krylov.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <fmt/format.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/operator.hpp"
#include "fuse/linalg/preconditioner.hpp"

namespace fuse::num {

using fuse::linalg::GmresOptions;
using fuse::linalg::IdentityPreconditioner;
using fuse::linalg::MatrixFreeOperator;
using fuse::linalg::SolverReport;

// ===========================================================================
// 1. MFFD Jacobian action.
// ===========================================================================

MffdJacobianAction::MffdJacobianAction(ResidualFn f, MffdOptions opt)
    : f_(std::move(f)), opt_(opt) {}

void MffdJacobianAction::set_point(const DistVector& x, const DistVector& fx) {
  x_ = x;
  fx_ = fx;
}

Result<void> MffdJacobianAction::apply(DistVector& v, DistVector& jv) {
  if (!x_.has_value() || !fx_.has_value()) {
    return Failure{"E-NEWTON-MFFD-NO-POINT",
                  "MffdJacobianAction::apply: set_point() must be called before apply()"};
  }
  const double x_norm = x_->norm2();
  const double v_norm_raw = v.norm2();
  const double v_norm = std::max(v_norm_raw, opt_.min_direction_norm);
  const double h = opt_.error_rel * std::sqrt(1.0 + x_norm) / v_norm;

  trace_.push_back(MffdRecord{h, x_norm, v_norm_raw});

  // xhv <- x + h v
  DistVector xhv(*x_);
  xhv.axpy(h, v);

  DistVector fxhv(*fx_);
  if (Result<void> r = f_(xhv, fxhv); !r.ok()) return r.error();
  ++evaluations_;

  // jv <- (F(x + h v) - F(x)) / h
  jv.copy_from(fxhv);
  jv.axpy(-1.0, *fx_);
  jv.scale(1.0 / h);
  return Result<void>::success();
}

// ===========================================================================
// 3. Eisenstat-Walker forcing.
// ===========================================================================

const char* to_cstring(EisenstatWalkerChoice c) noexcept {
  switch (c) {
    case EisenstatWalkerChoice::Constant: return "Constant";
    case EisenstatWalkerChoice::Choice1:  return "Choice1";
    case EisenstatWalkerChoice::Choice2:  return "Choice2";
  }
  return "?";
}

double eisenstat_walker_eta(const EisenstatWalkerOptions& opt, int k, double eta_prev,
                            double norm_fk, double norm_fkm1, double linear_residual_prev) {
  if (opt.choice == EisenstatWalkerChoice::Constant) {
    return std::clamp(opt.eta_constant, opt.eta_min, opt.eta_max);
  }
  if (k <= 0) {
    return std::clamp(opt.eta_initial, opt.eta_min, opt.eta_max);
  }

  double eta = opt.eta_initial;
  if (opt.choice == EisenstatWalkerChoice::Choice1) {
    // eta_k = | ||F_k|| - ||F_{k-1} + J_{k-1} d_{k-1}|| | / ||F_{k-1}||
    if (norm_fkm1 > 0.0 && linear_residual_prev >= 0.0) {
      eta = std::fabs(norm_fk - linear_residual_prev) / norm_fkm1;
    }
  } else {
    // Choice 2: eta_k = gamma (||F_k|| / ||F_{k-1}||)^alpha.
    if (norm_fkm1 > 0.0) {
      eta = opt.gamma * std::pow(norm_fk / norm_fkm1, opt.alpha);
    }
  }

  // The over-solving safeguard: eta_k >= gamma * eta_{k-1}^alpha whenever
  // that quantity exceeds safeguard_threshold.
  const double safeguard = opt.gamma * std::pow(eta_prev, opt.alpha);
  if (safeguard > opt.safeguard_threshold) {
    eta = std::max(eta, safeguard);
  }

  return std::clamp(eta, opt.eta_min, opt.eta_max);
}

// ===========================================================================
// 5. Named convergence and divergence tests.
// ===========================================================================

const char* to_cstring(NewtonStatus s) noexcept {
  switch (s) {
    case NewtonStatus::Running:                 return "Running";
    case NewtonStatus::ConvergedAbsolute:       return "ConvergedAbsolute";
    case NewtonStatus::ConvergedRelative:       return "ConvergedRelative";
    case NewtonStatus::ConvergedStep:           return "ConvergedStep";
    case NewtonStatus::DivergedFunctionCount:   return "DivergedFunctionCount";
    case NewtonStatus::DivergedLineSearch:      return "DivergedLineSearch";
    case NewtonStatus::DivergedResidualGrowth:  return "DivergedResidualGrowth";
    case NewtonStatus::DivergedLinearSolve:     return "DivergedLinearSolve";
    case NewtonStatus::DivergedMaxIterations:   return "DivergedMaxIterations";
    case NewtonStatus::DivergedFunctionError:   return "DivergedFunctionError";
  }
  return "?";
}

bool is_diverged(NewtonStatus s) noexcept {
  switch (s) {
    case NewtonStatus::DivergedFunctionCount:
    case NewtonStatus::DivergedLineSearch:
    case NewtonStatus::DivergedResidualGrowth:
    case NewtonStatus::DivergedLinearSolve:
    case NewtonStatus::DivergedMaxIterations:
    case NewtonStatus::DivergedFunctionError:
      return true;
    default:
      return false;
  }
}

bool is_converged(NewtonStatus s) noexcept {
  return s == NewtonStatus::ConvergedAbsolute || s == NewtonStatus::ConvergedRelative ||
         s == NewtonStatus::ConvergedStep;
}

// ===========================================================================
// 6. Lagged, block-Jacobi preconditioner.
// ===========================================================================

Result<BlockJacobiPreconditioner> BlockJacobiPreconditioner::from_blocks(
    std::vector<PreconditionerBlock> blocks, std::size_t n) {
  std::vector<std::pair<std::size_t, std::size_t>> spans;
  spans.reserve(blocks.size());
  for (const auto& b : blocks) spans.emplace_back(b.offset, b.offset + b.length);
  std::sort(spans.begin(), spans.end());

  std::size_t cursor = 0;
  for (const auto& [lo, hi] : spans) {
    if (lo != cursor) {
      return Failure{"E-NEWTON-PRECOND-COVERAGE",
                    fmt::format("BlockJacobiPreconditioner::from_blocks: blocks leave a gap or "
                               "overlap at offset {} (expected the next block to start there, "
                               "n={})",
                               cursor, n)};
    }
    cursor = hi;
  }
  if (cursor != n) {
    return Failure{"E-NEWTON-PRECOND-COVERAGE",
                  fmt::format("BlockJacobiPreconditioner::from_blocks: blocks cover [0, {}) but "
                             "n={}",
                             cursor, n)};
  }
  return BlockJacobiPreconditioner(std::move(blocks), n);
}

Result<void> BlockJacobiPreconditioner::apply(DistVector& x, DistVector& y) const {
  if (x.owned_size() != n_ || y.owned_size() != n_) {
    return Failure{"E-NEWTON-PRECOND-SHAPE",
                  fmt::format("BlockJacobiPreconditioner::apply: x/y owned sizes {}/{} disagree "
                             "with the {} this preconditioner was built for",
                             x.owned_size(), y.owned_size(), n_)};
  }
  for (const auto& b : blocks_) {
    std::span<const double> r(x.data() + b.offset, b.length);
    std::span<double> z(y.data() + b.offset, b.length);
    if (Result<void> s = b.apply(r, z); !s.ok()) return s.error();
  }
  return Result<void>::success();
}

// ===========================================================================
// The driver.
// ===========================================================================

namespace {

/// `0.5 * ||v||_2^2`, the Newton merit function at whatever vector `v`
/// already is.
double half_norm_sq(const DistVector& v) {
  const double n = v.norm2();
  return 0.5 * n * n;
}

}  // namespace

Result<NewtonResult> solve(const ResidualFn& f, DistVector& x,
                           const PreconditionerApplyFn& precond_apply,
                           const RefreshPreconditionerFn& refresh_precond,
                           const NewtonKrylovOptions& opt) {
  NewtonResult result;

  DistVector fx(x);
  fx.set_zero();
  if (Result<void> r = f(x, fx); !r.ok()) {
    result.status = NewtonStatus::DivergedFunctionError;
    return result;
  }
  std::size_t function_evaluations = 1;

  const double norm_f0 = fx.norm2();
  result.initial_residual_norm = norm_f0;
  double norm_f = norm_f0;

  auto finish = [&](NewtonStatus status) {
    result.status = status;
    result.function_evaluations = function_evaluations;
    result.final_residual_norm = norm_f;
    return result;
  };

  if (norm_f0 <= opt.atol) return finish(NewtonStatus::ConvergedAbsolute);

  MffdJacobianAction mffd(f, opt.mffd);
  double eta_prev = opt.forcing.eta_initial;
  double linear_residual_prev = -1.0;
  double norm_f_prev = norm_f0;

  for (int k = 0; k < opt.max_nonlinear_iterations; ++k) {
    NewtonIterationRecord rec;
    rec.iteration = k;
    rec.norm_f = norm_f;

    // -- Preconditioner lag: refresh at k == 0 and every `preconditioner_lag`
    //    steps thereafter.
    if (refresh_precond && (k == 0 || opt.preconditioner_lag <= 0 ||
                            k % opt.preconditioner_lag == 0)) {
      if (Result<void> r = refresh_precond(x, fx); !r.ok()) return r.error();
      rec.preconditioner_refreshed = true;
    }

    // -- Eisenstat-Walker forcing term for this step's linear solve.
    const double eta =
        eisenstat_walker_eta(opt.forcing, k, eta_prev, norm_f, norm_f_prev, linear_residual_prev);
    rec.eta = eta;

    // -- The linear solve: J d = -F, J matrix-free via MFFD, right-
    //    preconditioned.
    mffd.set_point(x, fx);
    mffd.clear_trace();
    MatrixFreeOperator jac_op(
        [&mffd](DistVector& v, DistVector& jv) { return mffd.apply(v, jv); });

    DistVector neg_fx(fx);
    neg_fx.scale(-1.0);
    DistVector d(x);
    d.set_zero();

    GmresOptions gmres_opt = opt.gmres;
    gmres_opt.rel_tol = eta;

    Result<SolverReport> gmres_result{Failure{"", ""}};
    if (precond_apply) {
      MatrixFreeOperator precond_op(precond_apply);
      gmres_result = fuse::linalg::gmres(jac_op, precond_op, d, neg_fx, gmres_opt);
    } else {
      gmres_result = fuse::linalg::gmres(jac_op, IdentityPreconditioner{}, d, neg_fx, gmres_opt);
    }
    function_evaluations += mffd.evaluations();
    rec.mffd_steps = mffd.trace();

    if (!gmres_result.ok()) {
      result.trace.push_back(rec);
      // gmres() itself returns a Failure for exactly two reasons: its own
      // "E-LINALG-NO-CONVERGENCE" (a genuine Krylov stagnation), or
      // whatever Failure the MatrixFreeOperator it called propagated
      // unchanged from `f` itself (MffdJacobianAction::apply passes `f`'s
      // own Failure through verbatim) -- the latter is a function
      // evaluation error, not a linear-solve divergence, and must be
      // reported as such rather than conflated with it.
      if (gmres_result.error().code == "E-LINALG-NO-CONVERGENCE") {
        return finish(NewtonStatus::DivergedLinearSolve);
      }
      return finish(NewtonStatus::DivergedFunctionError);
    }
    const SolverReport& report = gmres_result.value();
    rec.gmres_iterations = report.iterations;
    rec.gmres_residual_norm = report.residual_norm;

    if (function_evaluations > opt.max_function_evaluations) {
      result.trace.push_back(rec);
      return finish(NewtonStatus::DivergedFunctionCount);
    }

    // -- Line search along d.
    const double f0_merit = half_norm_sq(fx);
    DistVector x_trial(x);
    DistVector fx_trial(fx);
    auto eval_at = [&](double lambda, DistVector& xt, DistVector& ft) -> Result<void> {
      xt.copy_from(x);
      xt.axpy(lambda, d);
      if (Result<void> r = f(xt, ft); !r.ok()) return r.error();
      ++function_evaluations;
      return Result<void>::success();
    };

    LineSearchStatus ls_status = LineSearchStatus::FullStep;
    double lambda = 1.0;
    bool function_errored = false;

    if (opt.line_search.kind == LineSearchKind::Basic) {
      LineSearchResult basic = basic_line_search(opt.line_search);
      lambda = basic.lambda;
      ls_status = basic.status;
      rec.line_search_trials = basic.trials;
      if (Result<void> r = eval_at(lambda, x_trial, fx_trial); !r.ok()) function_errored = true;
    } else if (opt.line_search.kind == LineSearchKind::BT) {
      MeritFn merit = [&](double lmb) -> Result<double> {
        DistVector xt(x), ft(fx);
        if (Result<void> r = eval_at(lmb, xt, ft); !r.ok()) return r.error();
        // Keep the most recent trial around: backtracking_line_search()
        // returns immediately after accepting one, so whatever this left
        // behind is the accepted trial on every status except Failed
        // (handled by the explicit re-evaluation below).
        x_trial.copy_from(xt);
        fx_trial.copy_from(ft);
        return half_norm_sq(ft);
      };
      Result<LineSearchResult> ls = backtracking_line_search(f0_merit, merit, opt.line_search);
      if (!ls.ok()) { function_errored = true; }
      else {
        lambda = ls.value().lambda;
        ls_status = ls.value().status;
        rec.line_search_trials = ls.value().trials;
        // x_trial/fx_trial currently hold the LAST trial evaluated, which
        // is the accepted one on every status this function returns
        // (FullStep/Converged/MaxStepsReached all return after the last
        // trial; Failed may not have evaluated the rejected candidate at
        // all, in which case x_trial/fx_trial still hold the previous
        // accepted trial -- re-evaluate explicitly for correctness).
        if (ls_status == LineSearchStatus::Failed) {
          if (Result<void> r = eval_at(lambda, x_trial, fx_trial); !r.ok()) function_errored = true;
        }
      }
    } else {  // CP
      DistVector d_copy(d);
      const double g0 = fx.dot(d_copy);
      DirectionalFn directional = [&](double lmb) -> Result<double> {
        DistVector xt(x), ft(fx);
        if (Result<void> r = eval_at(lmb, xt, ft); !r.ok()) return r.error();
        x_trial.copy_from(xt);
        fx_trial.copy_from(ft);
        return ft.dot(d_copy);
      };
      Result<LineSearchResult> ls = critical_point_line_search(g0, directional, opt.line_search);
      if (!ls.ok()) { function_errored = true; }
      else {
        lambda = ls.value().lambda;
        ls_status = ls.value().status;
        rec.line_search_trials = ls.value().trials;
        if (ls_status == LineSearchStatus::Failed) {
          if (Result<void> r = eval_at(lambda, x_trial, fx_trial); !r.ok()) function_errored = true;
        }
      }
    }

    rec.line_search_status = ls_status;
    rec.line_search_lambda = lambda;

    if (function_errored) {
      result.trace.push_back(rec);
      return finish(NewtonStatus::DivergedFunctionError);
    }
    if (ls_status == LineSearchStatus::Failed || ls_status == LineSearchStatus::MaxStepsReached) {
      result.trace.push_back(rec);
      return finish(NewtonStatus::DivergedLineSearch);
    }
    if (function_evaluations > opt.max_function_evaluations) {
      result.trace.push_back(rec);
      return finish(NewtonStatus::DivergedFunctionCount);
    }

    const double x_norm_prev = x.norm2();
    DistVector step(d);
    step.scale(lambda);
    const double dx_norm = step.norm2();
    const double norm_f_this_step = norm_f;  // the pre-step ||F_k||, for norm_f_prev below

    x.copy_from(x_trial);
    fx.copy_from(fx_trial);
    const double norm_f_new = fx.norm2();

    result.trace.push_back(rec);
    // This Newton step DID update x (every return below this point leaves
    // x at the just-accepted iterate, never a rejected or mid-step
    // value), so it counts as completed regardless of which test below
    // ends the loop.
    result.iterations = k + 1;

    norm_f = norm_f_new;
    if (norm_f_new <= opt.atol) return finish(NewtonStatus::ConvergedAbsolute);
    if (norm_f_new <= opt.rtol * norm_f0) return finish(NewtonStatus::ConvergedRelative);
    if (dx_norm <= opt.stol * x_norm_prev) return finish(NewtonStatus::ConvergedStep);
    if (norm_f_new > opt.divergence_tolerance * norm_f0) {
      return finish(NewtonStatus::DivergedResidualGrowth);
    }

    norm_f_prev = norm_f_this_step;
    eta_prev = eta;
    linear_residual_prev = report.residual_norm;
  }

  return finish(NewtonStatus::DivergedMaxIterations);
}

// ===========================================================================
// S7.2b (PetscSelect track): the Newton substrate registry.
// ===========================================================================

namespace {

/// The registry itself, and `NewtonSubstrate::Native`'s own registration --
/// a FUNCTION-LOCAL static, constructed the first time ANY caller reaches
/// this function (`register_newton_substrate()` or `newton_substrate()`
/// below, both of which call this first), never a static initialiser in
/// this translation unit that depends on link order. The lambda it seeds
/// the map with is a trivial adapter onto `fuse::num::solve()` itself,
/// already defined above in this same file -- Native therefore never
/// needs an explicit `register_newton_substrate(NewtonSubstrate::Native,
/// ...)` call from anywhere.
std::unordered_map<NewtonSubstrate, NewtonSolveFn>& newton_substrate_registry() {
  static std::unordered_map<NewtonSubstrate, NewtonSolveFn> registry = [] {
    std::unordered_map<NewtonSubstrate, NewtonSolveFn> m;
    m[NewtonSubstrate::Native] =
        [](const ResidualFn& f, DistVector& x,
           const PreconditionerApplyFn& precond_apply,
           const RefreshPreconditionerFn& refresh_precond,
           const NewtonKrylovOptions& opt) {
          return solve(f, x, precond_apply, refresh_precond, opt);
        };
    return m;
  }();
  return registry;
}

}  // namespace

void register_newton_substrate(NewtonSubstrate kind, NewtonSolveFn fn) {
  newton_substrate_registry()[kind] = std::move(fn);
}

const NewtonSolveFn* newton_substrate(NewtonSubstrate kind) {
  auto& registry = newton_substrate_registry();
  const auto it = registry.find(kind);
  return it == registry.end() ? nullptr : &it->second;
}

}  // namespace fuse::num
