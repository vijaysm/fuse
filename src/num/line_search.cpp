#include "fuse/num/line_search.hpp"

#include <algorithm>
#include <cmath>

namespace fuse::num {

const char* to_cstring(LineSearchKind k) noexcept {
  switch (k) {
    case LineSearchKind::BT:    return "BT";
    case LineSearchKind::CP:    return "CP";
    case LineSearchKind::Basic: return "Basic";
  }
  return "?";
}

const char* to_cstring(LineSearchStatus s) noexcept {
  switch (s) {
    case LineSearchStatus::FullStep:       return "FullStep";
    case LineSearchStatus::Converged:      return "Converged";
    case LineSearchStatus::MaxStepsReached: return "MaxStepsReached";
    case LineSearchStatus::Failed:         return "Failed";
  }
  return "?";
}

namespace {

/// The quadratic model's minimiser for the FIRST backtrack, derived from
/// `f0`, the inexact-Newton slope estimate `f'(0) = -2 f0`, and the one
/// trial `f(lambda)` available. Dennis & Schnabel Section 6.3.1's quadratic step,
/// specialised to this slope: with `m(t) = f0 - 2 f0 t + c t^2` fit to
/// agree with `f(lambda)` at `t = lambda`, `c = (f(lambda) - f0 +
/// 2 f0 lambda) / lambda^2`, and the minimiser of `m` is `t* = f0 / c`.
double quadratic_backtrack(double f0, double lambda, double f_lambda) {
  const double c = (f_lambda - f0 + 2.0 * f0 * lambda) / (lambda * lambda);
  if (c <= 0.0) {
    // The quadratic model opens downward or is flat: no interior minimiser
    // exists for it. Fall back to the safeguard's own lower multiple of
    // the current trial, the same floor the clamp below would apply.
    return 0.5 * lambda;
  }
  return f0 / c;
}

/// Dennis & Schnabel Section 6.3.1's cubic backtrack: fits a cubic through
/// `f0`, slope `-2 f0`, and the two most recent trials
/// `(lambda1, f1)` (most recent) and `(lambda2, f2)`, and returns the
/// cubic's own interior minimiser.
double cubic_backtrack(double f0, double lambda1, double f1, double lambda2, double f2) {
  const double d1 = (f1 - f0 + 2.0 * f0 * lambda1) / (lambda1 * lambda1);
  const double d2 = (f2 - f0 + 2.0 * f0 * lambda2) / (lambda2 * lambda2);
  const double denom = lambda1 - lambda2;
  const double a = (d1 - d2) / denom;
  const double b = (-lambda2 * d1 + lambda1 * d2) / denom;
  if (a == 0.0) {
    // Degenerates to the quadratic model through (lambda1, f1).
    return quadratic_backtrack(f0, lambda1, f1);
  }
  const double disc = b * b + 6.0 * a * f0;  // b^2 - 3*a*(-2 f0)
  if (disc < 0.0) return 0.5 * lambda1;
  return (-b + std::sqrt(disc)) / (3.0 * a);
}

}  // namespace

Result<LineSearchResult> backtracking_line_search(double f0, const MeritFn& merit,
                                                   const LineSearchOptions& opt) {
  LineSearchResult out;
  if (f0 < 0.0) {
    return Failure{"E-LINESEARCH-SHAPE",
                  "backtracking_line_search: f0 (1/2 ||F(x)||^2) must be >= 0"};
  }
  if (f0 == 0.0) {
    // Already at a root: any step is trivially accepted, the full one by
    // convention.
    out.lambda = 1.0;
    out.status = LineSearchStatus::FullStep;
    out.trials = {1.0};
    return out;
  }

  double lambda = 1.0;
  Result<double> f1r = merit(lambda);
  if (!f1r.ok()) return f1r.error();
  double f_cur = f1r.value();
  out.trials.push_back(lambda);
  ++out.steps;

  // Armijo: f(lambda) <= (1 - 2 alpha lambda) f0.
  if (f_cur <= (1.0 - 2.0 * opt.armijo_alpha * lambda) * f0) {
    out.lambda = lambda;
    out.status = LineSearchStatus::FullStep;
    return out;
  }

  double lambda_prev = lambda, f_prev = f_cur;
  double lambda_prev2 = 0.0, f_prev2 = 0.0;
  bool have_prev2 = false;

  for (int it = 1; it < opt.max_backtracks; ++it) {
    double lambda_new = have_prev2
                             ? cubic_backtrack(f0, lambda_prev, f_prev, lambda_prev2, f_prev2)
                             : quadratic_backtrack(f0, lambda_prev, f_prev);
    // Safeguard: clamp into [safeguard_low, safeguard_high] * lambda_prev,
    // and reject a non-finite or non-positive model minimiser outright.
    if (!std::isfinite(lambda_new) || lambda_new <= 0.0) {
      lambda_new = opt.safeguard_low * lambda_prev;
    }
    lambda_new = std::clamp(lambda_new, opt.safeguard_low * lambda_prev,
                            opt.safeguard_high * lambda_prev);

    if (lambda_new < opt.lambda_min) {
      out.status = LineSearchStatus::Failed;
      out.lambda = lambda_new;
      return out;
    }

    Result<double> fr = merit(lambda_new);
    if (!fr.ok()) return fr.error();
    const double f_new = fr.value();
    out.trials.push_back(lambda_new);
    ++out.steps;

    if (f_new <= (1.0 - 2.0 * opt.armijo_alpha * lambda_new) * f0) {
      out.lambda = lambda_new;
      out.status = LineSearchStatus::Converged;
      return out;
    }

    lambda_prev2 = lambda_prev;
    f_prev2 = f_prev;
    have_prev2 = true;
    lambda_prev = lambda_new;
    f_prev = f_new;
  }

  out.status = LineSearchStatus::MaxStepsReached;
  out.lambda = lambda_prev;
  return out;
}

Result<LineSearchResult> critical_point_line_search(double g0, const DirectionalFn& directional,
                                                     const LineSearchOptions& opt) {
  LineSearchResult out;
  if (g0 == 0.0) {
    out.lambda = 1.0;
    out.status = LineSearchStatus::FullStep;
    out.trials = {1.0};
    return out;
  }

  double lambda0 = 0.0, g_lo = g0;
  double lambda1 = 1.0;
  Result<double> g1r = directional(lambda1);
  if (!g1r.ok()) return g1r.error();
  double g_hi = g1r.value();
  out.trials.push_back(lambda1);
  ++out.steps;

  if (std::fabs(g_hi) <= opt.secant_tol * std::fabs(g0)) {
    out.lambda = lambda1;
    out.status = LineSearchStatus::FullStep;
    return out;
  }

  for (int it = 0; it < opt.max_secant_steps; ++it) {
    const double denom = g_hi - g_lo;
    if (denom == 0.0 || !std::isfinite(denom)) {
      out.status = LineSearchStatus::Failed;
      out.lambda = lambda1;
      return out;
    }
    double lambda_new = lambda1 - g_hi * (lambda1 - lambda0) / denom;
    if (!std::isfinite(lambda_new) || lambda_new <= 0.0) {
      out.status = LineSearchStatus::Failed;
      out.lambda = lambda1;
      return out;
    }

    Result<double> gr = directional(lambda_new);
    if (!gr.ok()) return gr.error();
    const double g_new = gr.value();
    out.trials.push_back(lambda_new);
    ++out.steps;

    if (std::fabs(g_new) <= opt.secant_tol * std::fabs(g0)) {
      out.lambda = lambda_new;
      out.status = LineSearchStatus::Converged;
      return out;
    }

    lambda0 = lambda1;
    g_lo = g_hi;
    lambda1 = lambda_new;
    g_hi = g_new;
  }

  out.status = LineSearchStatus::MaxStepsReached;
  out.lambda = lambda1;
  return out;
}

LineSearchResult basic_line_search(const LineSearchOptions& opt) {
  LineSearchResult out;
  out.lambda = opt.damping;
  out.status = (opt.damping == 1.0) ? LineSearchStatus::FullStep : LineSearchStatus::Converged;
  out.steps = 0;
  out.trials = {opt.damping};
  return out;
}

}  // namespace fuse::num
