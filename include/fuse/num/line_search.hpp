#pragma once

/// S7.2's pluggable line search: three strategies, mirroring the three
/// forms of SNESLineSearch the native Newton-Krylov coupler
/// (fuse/num/newton_krylov.hpp) composes with. Deliberately scalar: every
/// function here takes and returns `double`s and callables over `double
/// lambda`, never a `DistVector` or a residual function -- the step length
/// a line search chooses is a SCALAR along the Newton direction, and the
/// merit/directional VALUE at a trial step is whatever norm or dot product
/// the caller already reduced across every participant before calling in
/// (the caller carries the rank-agreement obligation; this header is pure
/// arithmetic over already-reduced numbers, so it needs no Reducer of its
/// own and is testable with no DistVector at all).
///
/// SNESLINESEARCHBT: cubic backtracking on the merit function
/// `f(lambda) = 1/2 ||F(x + lambda d)||^2`, with the inexact-Newton
/// Armijo test `f(lambda) <= (1 - 2 alpha lambda) f(0)` -- the standard
/// substitution for `f(lambda) <= f(0) + alpha lambda f'(0)` once
/// `f'(0)` is approximated as `-2 f(0)` (exact when `d` solves `J d = -F`
/// to machine precision; `backtracking_line_search()` below never computes
/// a true gradient, the same "no Jacobian-transpose needed" property JFNK
/// itself depends on). Quadratic interpolation picks the first backtrack;
/// cubic interpolation (Dennis & Schnabel, "Numerical Methods for
/// Unconstrained Optimization and Nonlinear Equations", SIAM, 1996,
/// Section 6.3.1) picks every one after, using the two most recent trial points.
///
/// SNESLINESEARCHCP: for a residual that is itself a gradient,
/// `F = grad(Phi)`, the directional derivative of `Phi` along `d` is
/// `g(lambda) = <F(x + lambda d), d>`; a critical point of `Phi` along the
/// line is a ROOT of `g`, found by secant iteration
/// (`critical_point_line_search()` below) rather than backtracking.
///
/// SNESLINESEARCHBASIC: the full step, or a fixed damping factor, with no
/// evaluation of `F` at all (`basic_line_search()` below is a pure
/// function of `LineSearchOptions::damping`).

#include <cstddef>
#include <functional>
#include <vector>

#include "fuse/base/result.hpp"

namespace fuse::num {

enum class LineSearchKind { BT, CP, Basic };

[[nodiscard]] const char* to_cstring(LineSearchKind k) noexcept;

struct LineSearchOptions {
  LineSearchKind kind = LineSearchKind::BT;

  /// BT: the Armijo sufficient-decrease constant (SNESLINESEARCHBT's own
  /// default, `-snes_linesearch_alpha`).
  double armijo_alpha = 1e-4;
  /// BT: a trial step below this is a failure (lambda collapsed to
  /// nothing) rather than another backtrack -- SNESLINESEARCHBT's own
  /// `-snes_linesearch_minlambda`.
  double lambda_min = 1e-12;
  /// BT: the backtrack budget. SNESLINESEARCHBT's own
  /// `-snes_linesearch_max_it` default is lower (40 here matches this
  /// track's own brief).
  int max_backtracks = 40;
  /// BT: every accepted trial is additionally clamped into
  /// `[safeguard_low, safeguard_high] * (previous trial)`, Dennis &
  /// Schnabel's own safeguard against a backtrack that barely moves or
  /// one that collapses to zero in one step.
  double safeguard_low = 0.1;
  double safeguard_high = 0.5;

  /// CP: secant convergence, `|g(lambda)| <= secant_tol * |g(0)|`.
  double secant_tol = 1e-10;
  int max_secant_steps = 40;

  /// Basic: the fixed step taken with no evaluation at all. 1.0 is the
  /// full Newton step; SNESLINESEARCHBASIC's own `-snes_linesearch_damping`
  /// default is the same.
  double damping = 1.0;
};

enum class LineSearchStatus {
  /// BT took the full Newton step (lambda == 1) without a single backtrack.
  FullStep,
  /// BT accepted a step strictly inside (0, 1) after backtracking, or CP's
  /// secant iteration found its root.
  Converged,
  /// BT exhausted max_backtracks, or CP exhausted max_secant_steps, with
  /// no acceptable step found.
  MaxStepsReached,
  /// BT's trial fell below lambda_min, or CP's secant denominator
  /// vanished (g(lambda) stopped changing) -- a degenerate line, not
  /// merely a slow one.
  Failed,
};

[[nodiscard]] const char* to_cstring(LineSearchStatus s) noexcept;

struct LineSearchResult {
  double lambda = 1.0;
  int steps = 0;
  LineSearchStatus status = LineSearchStatus::FullStep;
  /// Every trial lambda, in the order they were tried -- what the task's
  /// own test requirement ("record the accepted lambdas") and
  /// "record the accepted step lengths" read.
  std::vector<double> trials;
};

/// `merit(lambda)` returns `f(lambda) = 1/2 ||F(x + lambda d)||^2`, reduced
/// across every participant by the caller (the same discipline
/// `fuse::linalg::DistVector::norm2()` already bakes into every norm this
/// project computes) before it is handed to this function -- so every
/// participant that calls `backtracking_line_search()` with its own copy
/// of the SAME reduced `f0`/`merit` sequence makes the SAME accept/reject
/// decision at the SAME trial, with no reduction of its own needed here.
using MeritFn = std::function<Result<double>(double lambda)>;

/// `directional(lambda)` returns `g(lambda) = <F(x + lambda d), d>`,
/// likewise already reduced.
using DirectionalFn = std::function<Result<double>(double lambda)>;

/// `f0 = merit(0)`, i.e. `1/2 ||F(x)||^2`, passed in because the caller
/// already has it (it is `F(x)`'s own norm, computed before any line
/// search began) and recomputing it here would be a wasted evaluation.
[[nodiscard]] Result<LineSearchResult> backtracking_line_search(
    double f0, const MeritFn& merit, const LineSearchOptions& opt = {});

/// `g0 = directional(0)`, likewise already known (`<F(x), d>`).
[[nodiscard]] Result<LineSearchResult> critical_point_line_search(
    double g0, const DirectionalFn& directional, const LineSearchOptions& opt = {});

/// No evaluation at all: `lambda = opt.damping`, `status = FullStep` when
/// `damping == 1.0`, `Converged` otherwise (both are successes; the status
/// distinguishes them only for a caller's own reporting).
[[nodiscard]] LineSearchResult basic_line_search(const LineSearchOptions& opt = {});

}  // namespace fuse::num
