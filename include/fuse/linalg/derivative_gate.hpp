#pragma once

/// The derivative gate -- the one check every later "Newton" or
/// "monolithic" claim in this project must cite (docs/dev/derivative-gate.md
/// has the full account: why each bound below is derived rather than chosen,
/// and the measurements that fixed the one free constant each carries).
/// Three independent properties, none of them a tolerance picked by hand:
///
///   1. ADJOINT. For an operator with an `apply` and an `apply_transpose`
///      (or, for an operator this project only ever builds self-adjoint --
///      `fuse::linalg::SchurSlabSolver` over `A = I - coefficient*L_h`, `L_h`
///      symmetric -- the same `apply` used on both sides), `<A x, y>` and
///      `<x, A^T y>` must agree to a bound DERIVED from floating-point
///      summation error: `c * eps * ||A||_est * ||x|| * ||y||`, `||A||_est`
///      estimated by power iteration (`estimate_operator_norm`) so the bound
///      needs no assumption about the operator's sign pattern or structure.
///      `adjoint_gate()` below.
///   2. FINITE-DIFFERENCE CONVERGENCE. For a residual `F` and a claimed
///      Jacobian action `J v`, the Taylor remainder
///      `F(x + h v) - F(x) - h J v` must shrink at the claimed linearisation's
///      order (2, for a correct first derivative) over a decade of `h` ABOVE
///      the round-off floor, and the floor itself -- where the remainder
///      stops decreasing because `F(x+hv) - F(x)` has lost precision to
///      cancellation, not because the Taylor expansion says anything new --
///      must be identified rather than fit through. `taylor_remainder_order()`
///      below.
///   3. ASSEMBLED-VS-MATRIX-FREE AGREEMENT. Where two independent
///      implementations claim the SAME operator (an assembled matrix and a
///      matrix-free kernel), their `apply` must agree on random vectors to a
///      bound of the same derived kind as (1). `matrix_free_agreement()`
///      below.
///
/// WHY HOST-MEMORY SPANS, NOT A TEMPLATE OVER THREE OPERATOR INTERFACES.
/// Every operator kind this gate is run against in
/// test/linalg/derivative_gate_test.cpp -- `fuse::linalg::DistSparseMatrix`
/// (`DistVector`-shaped, dist_sparse_matrix.hpp), `fuse::WeightsOperator`/
/// `fuse::ComposedOperator` (span-shaped, backend/linear_operator.hpp),
/// `fuse::linalg::SchurSlabSolver` (`DistVector`-shaped, `solve()` rather
/// than `apply()`, and no `apply_transpose` of its own at all) -- reduces to
/// the SAME shape once its output is read into a contiguous host buffer:
/// `rows`/`cols` plus two callables, `apply`/`apply_transpose`, each
/// `Result<void>(std::span<const double>, std::span<double>)`. This file
/// works at that one, lowest common shape rather than templating over three
/// unrelated operator interfaces, so the adapter from a `DistVector` or an
/// `Eigen::SparseMatrix` to a plain span is something
/// test/linalg/derivative_gate_test.cpp writes once per operator kind, never
/// something this header needs to know exists. `fuse::adjoint_defect`
/// (backend/linear_operator.hpp) already checks the identical
/// adjoint property for the one operator family it was written for (a
/// RELATIVE defect, floored at a denominator of 1) -- this file does not
/// duplicate it or change it; `adjoint_gate` below is the ABSOLUTE,
/// norm-derived bound this file provides, usable
/// across every operator family in the project, backend::LinearOperator
/// included (test/linalg/derivative_gate_test.cpp runs both checks over the
/// backend cases, so a WeightsOperator is held to both bounds at once).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"

namespace fuse::linalg {

/// `y <- A x` (or `y <- A^T x`), reading `x` and writing `y` in host memory.
/// Every adapter in test/linalg/derivative_gate_test.cpp -- a `DistVector`
/// pair held alongside the span, an `Eigen::SparseMatrix` product, a
/// `fuse::LinearOperator`'s own `apply`/`apply_transpose` -- ends at one of
/// these.
using HostApplyFn = std::function<Result<void>(std::span<const double>, std::span<double>)>;

// ===========================================================================
// 1. The adjoint identity, against a bound derived from the operator's own
//    norm rather than chosen.
// ===========================================================================

/// `||A||_2` (the spectral/operator 2-norm), estimated by power iteration on
/// `A^T A`: `v_0` a fixed-seed random unit vector of length `cols`, then
/// `v_{k+1} <- normalize(A^T (A v_k))` for `max_iterations` steps, with the
/// returned estimate `||A v||` at the final (normalized) `v` -- the
/// Rayleigh-quotient reading the power method converges to the operator's
/// largest singular value. Needs no assumption about `A`'s sign pattern or
/// structure (unlike a row/col-sum bound, which only bounds `||A||_2` for a
/// non-negative matrix -- `DistSparseMatrix`'s own random-nonsymmetric
/// fixture has negative entries by construction, so that bound would not
/// apply to it), and costs only `apply`+`apply_transpose` calls, which every
/// operator this gate runs against already provides (or, for a self-adjoint
/// operator, provides twice over under one name).
///
/// `max_iterations` is small on purpose: this estimate feeds a TOLERANCE
/// (the adjoint and matrix-free-agreement bounds below), not a converged
/// singular value a caller would report as a result in its own right, and a
/// looser estimate only makes those bounds more generous, never less --
/// docs/dev/derivative-gate.md records, for every operator this gate
/// exercises, how many iterations it took the estimate to stop moving by
/// more than 1% (never more than 8, of the 20 this stops at by default).
///
/// Errors, never throws: `E-GATE-NORM-SHAPE` for `rows == 0 || cols == 0`,
/// which a power iteration cannot start from (there is no vector of length
/// zero to normalize); whatever `Failure` `apply`/`apply_transpose` itself
/// returns, passed through unchanged.
[[nodiscard]] inline Result<double> estimate_operator_norm(
    std::size_t rows, std::size_t cols, const HostApplyFn& apply,
    const HostApplyFn& apply_transpose, std::size_t max_iterations = 20,
    unsigned seed = 0x5eed1234u) {
  if (rows == 0 || cols == 0) {
    return Failure{"E-GATE-NORM-SHAPE",
                   "estimate_operator_norm: rows and cols must both be >= 1, got " +
                       std::to_string(rows) + "x" + std::to_string(cols)};
  }
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(cols);
  for (double& e : v) e = u(rng);
  double vnorm = 0.0;
  for (double e : v) vnorm += e * e;
  vnorm = std::sqrt(vnorm);
  if (vnorm == 0.0) v[0] = 1.0, vnorm = 1.0;  // the all-zero draw, astronomically unlikely
  for (double& e : v) e /= vnorm;

  std::vector<double> w(rows, 0.0), u_vec(cols, 0.0);
  double sigma = 0.0;
  for (std::size_t it = 0; it < max_iterations; ++it) {
    if (Result<void> r = apply(v, w); !r.ok()) return r.error();
    if (Result<void> r = apply_transpose(w, u_vec); !r.ok()) return r.error();
    double unorm = 0.0;
    for (double e : u_vec) unorm += e * e;
    unorm = std::sqrt(unorm);
    if (unorm == 0.0) {
      // A v (and hence A^T A v) is exactly zero: the operator annihilates
      // this vector. sigma estimate is 0 from here; stop rather than divide
      // by zero normalizing a vector with nothing left to normalize.
      sigma = 0.0;
      break;
    }
    for (std::size_t i = 0; i < u_vec.size(); ++i) v[i] = u_vec[i] / unorm;
    if (Result<void> r = apply(v, w); !r.ok()) return r.error();
    double wnorm = 0.0;
    for (double e : w) wnorm += e * e;
    sigma = std::sqrt(wnorm);
  }
  return sigma;
}

/// One adjoint-identity measurement: `|<A x, y> - <x, A^T y>|` against
/// `bound = c * eps * norm_est * ||x||_2 * ||y||_2`. `name` is carried
/// through only for a caller's own failure messages (this project's own
/// `EXPECT_LT(..., ...) << c.name` convention, operator_conformance_test.cpp)
/// -- it does not affect the computation.
struct AdjointGateResult {
  double defect = 0.0;     ///< |<Ax,y> - <x,A^Ty>|
  double bound = 0.0;      ///< c * eps * norm_est * ||x|| * ||y||
  double norm_est = 0.0;   ///< the ||A||_est this bound was derived from
  double x_norm = 0.0;
  double y_norm = 0.0;
  bool passed = false;
  std::string name;
};

/// `c` (`kAdjointGateConstant`, docs/dev/derivative-gate.md) is the ONE free
/// constant in the adjoint bound. It is not chosen; it is read off a
/// measurement this file's own test records every time it runs
/// (test/linalg/derivative_gate_test.cpp,
/// `LinalgDerivativeGate.AdjointIdentityHoldsForEveryOperator`, which prints
/// the worst case to stderr and asserts it stays below this constant): over
/// every operator in that file's `adjoint_cases()` (`DistSparseMatrix` at
/// one and two in-process partitions, `WeightsOperator` over three
/// hand-built matrices plus a real consistent-interpolation H plus a real
/// supermesh map plus a real moving-least-squares map, their composition,
/// and `SchurSlabSolver` at one rank and at two in-process partitions), the
/// largest observed ratio `defect / (eps * norm_est * ||x|| * ||y||)`
/// MEASURED (Debug, AppleClang 21, the Eigen linalg
/// backend) was 0.235 -- the one-partition periodic 16x16 Laplacian, the
/// operator with by far the most floating-point terms per entry in the
/// dot products this bound covers. `kAdjointGateConstant = 50` carries
/// roughly 200x headroom over that measurement -- generous on purpose
/// (robustness over a tight bound, this project's own standing rule), while
/// staying many orders of magnitude below the defect a genuine transpose
/// bug produces (`ADeliberatelyWrongAdjointIsDetected`, this file's own
/// control case, measures a mismatched-transpose defect more than 100x its
/// own bound, i.e. an O(1) quantity against an O(1e-13)-scale bound).
inline constexpr double kAdjointGateConstant = 50.0;

[[nodiscard]] inline Result<AdjointGateResult> adjoint_gate(
    std::size_t rows, std::size_t cols, const HostApplyFn& apply,
    const HostApplyFn& apply_transpose, std::span<const double> x,
    std::span<const double> y, double norm_est, double c = kAdjointGateConstant,
    std::string name = {}) {
  if (x.size() != cols || y.size() != rows) {
    return Failure{"E-GATE-ADJOINT-SHAPE",
                   "adjoint_gate: x has " + std::to_string(x.size()) + " entries and y has " +
                       std::to_string(y.size()) + ", but the operator is " +
                       std::to_string(rows) + "x" + std::to_string(cols)};
  }
  std::vector<double> ax(rows, 0.0), aty(cols, 0.0);
  if (Result<void> r = apply(x, ax); !r.ok()) return r.error();
  if (Result<void> r = apply_transpose(y, aty); !r.ok()) return r.error();

  // Straight loops, not a reduction: this compares two SPECIFIC summation
  // orders (y . (A x) vs (A^T y) . x), the same reasoning
  // backend/linear_operator.cpp's own `adjoint_defect` states for the
  // identical choice.
  double lhs = 0.0, rhs = 0.0;
  for (std::size_t i = 0; i < rows; ++i) lhs += y[i] * ax[i];
  for (std::size_t j = 0; j < cols; ++j) rhs += aty[j] * x[j];

  double xnorm = 0.0, ynorm = 0.0;
  for (double e : x) xnorm += e * e;
  for (double e : y) ynorm += e * e;
  xnorm = std::sqrt(xnorm);
  ynorm = std::sqrt(ynorm);

  const double eps = std::numeric_limits<double>::epsilon();
  AdjointGateResult out;
  out.defect = std::abs(lhs - rhs);
  out.norm_est = norm_est;
  out.x_norm = xnorm;
  out.y_norm = ynorm;
  out.bound = c * eps * norm_est * xnorm * ynorm;
  out.passed = out.defect <= out.bound;
  out.name = std::move(name);
  return out;
}

// ===========================================================================
// 2. Finite-difference derivative convergence.
// ===========================================================================

using HostVectorFn = std::function<std::vector<double>(const std::vector<double>&)>;
/// `J(x) v`: the claimed directional derivative at `x` in direction `v`.
using HostJacobianActionFn =
    std::function<std::vector<double>(const std::vector<double>& x, const std::vector<double>& v)>;

/// One `taylor_remainder_order` run: the swept `h` values (largest first),
/// the measured remainder `||F(x+hv) - F(x) - h J(x) v||_2` at each, the
/// fitted order (log-log slope over the points ABOVE the round-off floor),
/// and the floor itself. `order` is `NaN` and `at_floor_everywhere` is `true`
/// when every swept `h` is already at or below the floor -- the dn_heat-
/// shaped affine-map case, where `F` is exactly linear and the remainder is
/// mathematically zero at every `h`, so there is no decreasing trend to fit
/// at all, by construction rather than by bad luck with `h0`.
struct TaylorRemainderResult {
  std::vector<double> h;
  std::vector<double> remainder;
  double order = 0.0;
  double floor = 0.0;
  bool at_floor_everywhere = false;
  std::size_t fit_points = 0;  ///< how many of h/remainder were used in the slope fit
};

/// Sweeps `h = h0 * 10^-k` for `k = 0 .. ndecades` (`ndecades + 1` points,
/// largest `h` first), measures the Taylor remainder at each, DETECTS the
/// round-off plateau (the first `k` at which the remainder fails to drop by
/// at least `plateau_drop` relative to the previous point -- MEASURED,
/// docs/dev/derivative-gate.md: a genuine order-2 remainder drops by ~100x
/// per decade and a genuine order-1 one by ~10x, so `plateau_drop = 0.5`
/// (a mere 2x drop demanded to keep counting as "still converging") never
/// mistakes real convergence for the floor on any case this gate runs, while
/// catching the floor itself, which does not drop at all), and fits the
/// log-log slope over the points strictly before the plateau. Fewer than two
/// such points (the plateau starts at `k <= 1`, including never having
/// dropped at all from the very first step) means there is nothing to fit:
/// `at_floor_everywhere` is set, `order` is `NaN`, and `floor` is the mean of
/// every measured remainder (all of them are the floor in that case).
[[nodiscard]] inline TaylorRemainderResult taylor_remainder_order(
    const HostVectorFn& F, const HostJacobianActionFn& Jv, const std::vector<double>& x,
    const std::vector<double>& v, double h0, int ndecades, double plateau_drop = 0.5) {
  TaylorRemainderResult out;
  const std::size_t n = static_cast<std::size_t>(ndecades) + 1;
  out.h.reserve(n);
  out.remainder.reserve(n);

  const std::vector<double> fx = F(x);
  for (int k = 0; k <= ndecades; ++k) {
    const double h = h0 * std::pow(10.0, -k);
    std::vector<double> xhv(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) xhv[i] = x[i] + h * v[i];
    const std::vector<double> fxhv = F(xhv);
    const std::vector<double> jv = Jv(x, v);
    double rem2 = 0.0;
    for (std::size_t i = 0; i < fx.size(); ++i) {
      const double r = fxhv[i] - fx[i] - h * jv[i];
      rem2 += r * r;
    }
    out.h.push_back(h);
    out.remainder.push_back(std::sqrt(rem2));
  }

  // Plateau: the first k >= 1 where remainder[k] did not drop to at most
  // plateau_drop times remainder[k-1].
  std::size_t plateau_start = n;
  for (std::size_t k = 1; k < n; ++k) {
    const double prev = out.remainder[k - 1];
    const double cur = out.remainder[k];
    const bool still_dropping = prev > 0.0 && cur <= plateau_drop * prev;
    if (!still_dropping) {
      plateau_start = k;
      break;
    }
  }

  if (plateau_start <= 1) {
    out.at_floor_everywhere = true;
    out.order = std::numeric_limits<double>::quiet_NaN();
    double mean = 0.0;
    for (double r : out.remainder) mean += r;
    out.floor = mean / static_cast<double>(n);
    out.fit_points = 0;
    return out;
  }

  // Fit log(remainder) = order * log(h) + const over [0, plateau_start).
  const auto m = static_cast<double>(plateau_start);
  double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
  for (std::size_t k = 0; k < plateau_start; ++k) {
    const double lx = std::log(out.h[k]);
    const double ly = std::log(out.remainder[k]);
    sx += lx;
    sy += ly;
    sxx += lx * lx;
    sxy += lx * ly;
  }
  out.order = (m * sxy - sx * sy) / (m * sxx - sx * sx);
  out.fit_points = plateau_start;
  double floor_mean = 0.0;
  for (std::size_t k = plateau_start; k < n; ++k) floor_mean += out.remainder[k];
  out.floor = floor_mean / static_cast<double>(n - plateau_start);
  return out;
}

// ===========================================================================
// 3. Assembled-vs-matrix-free agreement.
// ===========================================================================

/// `c` for `matrix_free_agreement` (docs/dev/derivative-gate.md): a direct
/// comparison of two `apply`s on the SAME vector, never a dot product, so it
/// carries no second-vector cancellation term at all. MEASURED
/// (test/linalg/derivative_gate_test.cpp,
/// `AssembledDiffusionMatrixAgreesWithHandWrittenStencilApply`, which prints
/// its own max_defect/bound to stderr): the Brusselator diffusion operator
/// at 16x16 (`DistSparseMatrix::apply` vs. the hand-written periodic-stencil
/// kernel) reached a ratio of `defect / (eps * norm_est * ||x||)` of about
/// 0.36 at `c = 20` (i.e. the ratio would be 0.36 even at `c = 1`).
/// `kMatrixFreeAgreementConstant = 20` keeps roughly 55x headroom over that
/// measurement -- the same uniformly generous margin `kAdjointGateConstant`
/// uses, rather than tightening it just because the one measurement allowed
/// it.
inline constexpr double kMatrixFreeAgreementConstant = 20.0;

struct MatrixFreeAgreementResult {
  double max_defect = 0.0;
  double bound = 0.0;
  std::size_t n_vectors = 0;
  bool passed = false;
};

/// Applies both `a` and `b` to `n_vectors` fixed-seed random vectors of
/// length `cols` and requires `||a(x) - b(x)||_2 <= c * eps * norm_est *
/// ||x||_2` for every one of them, reporting the worst (`max_defect`/
/// `bound` are both the worst case's own pair, so `passed` is exactly
/// `max_defect <= bound`).
[[nodiscard]] inline Result<MatrixFreeAgreementResult> matrix_free_agreement(
    std::size_t rows, std::size_t cols, const HostApplyFn& a, const HostApplyFn& b,
    std::size_t n_vectors, double norm_est, double c = kMatrixFreeAgreementConstant,
    unsigned seed = 0xA6ee5eedu) {
  if (rows == 0 || cols == 0) {
    return Failure{"E-GATE-AGREE-SHAPE",
                   "matrix_free_agreement: rows and cols must both be >= 1"};
  }
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const double eps = std::numeric_limits<double>::epsilon();

  MatrixFreeAgreementResult out;
  out.n_vectors = n_vectors;
  double worst_defect = 0.0, worst_bound = 0.0;
  for (std::size_t k = 0; k < n_vectors; ++k) {
    std::vector<double> x(cols);
    for (double& e : x) e = u(rng);
    double xnorm = 0.0;
    for (double e : x) xnorm += e * e;
    xnorm = std::sqrt(xnorm);

    std::vector<double> ya(rows, 0.0), yb(rows, 0.0);
    if (Result<void> r = a(x, ya); !r.ok()) return r.error();
    if (Result<void> r = b(x, yb); !r.ok()) return r.error();
    double d2 = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
      const double d = ya[i] - yb[i];
      d2 += d * d;
    }
    const double defect = std::sqrt(d2);
    const double bound = c * eps * norm_est * xnorm;
    if (defect > worst_defect) {
      worst_defect = defect;
      worst_bound = bound;
    }
  }
  out.max_defect = worst_defect;
  out.bound = worst_bound;
  out.passed = worst_defect <= worst_bound;
  return out;
}

}  // namespace fuse::linalg
