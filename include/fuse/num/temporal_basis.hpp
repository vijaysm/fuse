#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/base/error.hpp"

/// Spec §4.4 / §7.3: the temporal half of a space-time representation. These
/// are the raw basis kernels -- fitting, evaluating and integrating a
/// polynomial or a cubic spline in time -- factored out of `Waveform` so the
/// space-time projection operator (Task 6.3) can reuse them without owning a
/// sample buffer.
///
/// Tier C per spec §3.1: header-only. Everything here is `inline`; there is no
/// translation unit and nothing to add to `src/CMakeLists.txt`.
///
/// CONVENTION -- all of these take *plain doubles* for time. Exact rational
/// time (spec §4.3) lives one level up in `Waveform`, which converts to double
/// only after forming the exact difference `t - window_begin`. Callers should
/// do the same: pass window-local times, not absolute simulated times, or the
/// monomial powers below will lose precision to cancellation.
namespace fuse::num {

namespace detail {

// Gauss-Legendre nodes/weights on [-1, 1]. 25 significant digits so the
// tables are correctly rounded in both double and long double.
inline constexpr double kGaussNodes1[1]   = {0.0};
inline constexpr double kGaussWeights1[1] = {2.0};

inline constexpr double kGaussNodes2[2] = {-0.5773502691896257645091488,
                                           0.5773502691896257645091488};
inline constexpr double kGaussWeights2[2] = {1.0, 1.0};

inline constexpr double kGaussNodes3[3] = {-0.7745966692414833770358531, 0.0,
                                           0.7745966692414833770358531};
inline constexpr double kGaussWeights3[3] = {0.5555555555555555555555556,
                                             0.8888888888888888888888889,
                                             0.5555555555555555555555556};

inline constexpr double kGaussNodes4[4] = {
    -0.8611363115940525752239465, -0.3399810435848562648026658,
    0.3399810435848562648026658, 0.8611363115940525752239465};
inline constexpr double kGaussWeights4[4] = {
    0.3478548451374538573730639, 0.6521451548625461426269361,
    0.6521451548625461426269361, 0.3478548451374538573730639};

}  // namespace detail

/// Gauss-Legendre rule on the reference interval [-1, 1].
struct GaussRule {
  const double* nodes;
  const double* weights;
  int           count;
};

/// `points` must be in [1, 4]; the rule is exact for polynomials of degree
/// <= 2*points - 1.
///
/// Out of range is a THROW, not an assert and not a clamp: the returned
/// struct is a pair of raw pointers, so a silently-wrong or default-
/// constructed rule would be a null dereference in Release (where `assert`
/// compiles out) rather than a diagnosable error. Same reasoning as
/// `Rational`'s unconditional overflow checks.
[[nodiscard]] inline GaussRule gauss_rule(int points) {
  switch (points) {
    case 1: return {detail::kGaussNodes1, detail::kGaussWeights1, 1};
    case 2: return {detail::kGaussNodes2, detail::kGaussWeights2, 2};
    case 3: return {detail::kGaussNodes3, detail::kGaussWeights3, 3};
    case 4: return {detail::kGaussNodes4, detail::kGaussWeights4, 4};
    default:
      FUSE_THROW(InvalidArgument, "{}", 
          "fuse::num::gauss_rule: only 1..4 points are tabulated, got " +
          std::to_string(points));
  }
}

/// Least-squares fit of a polynomial to the pairs (t[i], f[i]), returned as
/// monomial coefficients about `t0`:  p(x) = sum_k c[k] * (x - t0)^k.
///
/// The requested `degree` is CLAMPED to `t.size() - 1`. Above that the normal
/// equations are singular (more unknowns than data), so clamping is the only
/// defined answer; `Waveform` clamps identically before calling, and the two
/// must agree. `c.size()` is therefore `min(degree, n-1) + 1`, which callers
/// must not assume equals `degree + 1`.
///
/// Conditioning: the fit is performed in the scaled variable
/// s = (t - t0) / max_i|t_i - t0| and the coefficients are unscaled afterwards,
/// so the normal-equations matrix depends only on the *shape* of the node
/// distribution and not on the window length.
///
/// Measured, so that nobody over-credits this: the scaling is defensive, not a
/// correctness fix. The window length enters the unscaled matrix as a pure
/// diagonal similarity (entry (r,c) carries a factor h^(r+c)), and Gaussian
/// elimination with partial pivoting is already insensitive to that -- for
/// window lengths from 1e-8 to 1e+8 the *evaluated* polynomial agrees to
/// ~5e-15 relative with and without the scaling. What the scaling does buy is
/// recovery of the high-order coefficients themselves at extreme window
/// lengths (degree 2 at h = 1e-8: c2 comes out 31 scaled versus 293 unscaled,
/// against a true value of 5), which matters to any caller that inspects the
/// coefficients rather than evaluating them.
///
/// Degree is limited in practice because the normal equations square the
/// Vandermonde condition number; that is genuine ill-conditioning, which no
/// scaling removes, and is why `Waveform` caps DG at degree 4. Measured floor
/// for degree 3 on 4 equispaced nodes: ~1e-14, independent of window length.
///
/// Throws `std::invalid_argument` on empty or mismatched inputs or a negative
/// degree, and `std::runtime_error` if the normal equations are exactly
/// singular (duplicate sample times).
///
/// The normal-equations matrix and its elimination depend only on `t`,
/// `degree` and `t0` -- never on `f` -- yet `Waveform::evaluate_dg`/
/// `integrate_dg` called this once per field element with the same `t`,
/// `degree`, `t0` and a different `f` each time (REVIEW.md R-48), repeating
/// the O(n*m^2 + m^3) build-and-eliminate for every element. `fit_polynomial`
/// itself is unchanged -- same signature, same exceptions, same answer -- and
/// is now the one-shot convenience wrapper around the two halves below, which
/// a multi-element caller uses directly to pay the shared cost once.
[[nodiscard]] inline std::vector<double> fit_polynomial(
    const std::vector<double>& t, const std::vector<double>& f, int degree,
    double t0);

/// The `t`-only half of `fit_polynomial`: builds the normal-equations matrix
/// and eliminates it, without ever touching `f`. Reusable across every field
/// element that shares the same sample times, degree and expansion point.
///
/// Throws exactly as `fit_polynomial` does for the checks that do not need
/// `f`: empty `t`, a negative degree, or a singular (duplicate abscissae)
/// system. The `f`-length check is `solve_polynomial_fit`'s, since only it
/// has `f` to check.
struct PolynomialFitPlan {
  std::size_t n = 0;             ///< sample count `solve` must match `f` to.
  std::size_t m = 0;             ///< number of basis functions (deg + 1).
  double      h = 1.0;
  double      t0 = 0.0;
  std::vector<double> phi;       ///< n * m, row i: powers of (t[i]-t0)/h.
  std::vector<double> lu;        ///< m * m, the eliminated normal-equations
                                 ///< matrix (upper triangle is R; the
                                 ///< multipliers used to get there are `mult`).
  std::vector<std::size_t> piv;  ///< piv[k]: the row swapped into position k.
  std::vector<double> mult;      ///< m * m, mult[r*m+k] is the elimination
                                 ///< factor applied to row r at step k.
};

[[nodiscard]] inline PolynomialFitPlan prepare_polynomial_fit(
    const std::vector<double>& t, int degree, double t0) {
  const std::size_t n = t.size();
  if (n == 0) {
    FUSE_THROW(InvalidArgument, "{}", "fuse::num::fit_polynomial: no samples");
  }
  if (degree < 0) {
    FUSE_THROW(InvalidArgument, "{}", 
        "fuse::num::fit_polynomial: negative degree " + std::to_string(degree));
  }
  const std::size_t deg =
      std::min(static_cast<std::size_t>(degree), n - 1);
  const std::size_t m = deg + 1;

  PolynomialFitPlan plan;
  plan.n = n;
  plan.m = m;
  plan.t0 = t0;

  double h = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    h = std::max(h, std::fabs(t[i] - t0));
  }
  if (!(h > 0.0)) h = 1.0;  // single sample at t0, or all samples at t0
  plan.h = h;

  std::vector<double>& phi = plan.phi;
  phi.assign(n * m, 0.0);
  std::vector<double> a(m * m, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double s = (t[i] - t0) / h;
    double       p = 1.0;
    for (std::size_t k = 0; k < m; ++k) {
      phi[i * m + k] = p;
      p *= s;
    }
    for (std::size_t r = 0; r < m; ++r) {
      for (std::size_t c = 0; c < m; ++c) {
        a[r * m + c] += phi[i * m + r] * phi[i * m + c];
      }
    }
  }

  plan.piv.assign(m, 0);
  plan.mult.assign(m * m, 0.0);

  // Gaussian elimination with partial pivoting; m <= 5 in every FUSE use, so
  // LAPACK would be a dependency in the core for a 5x5 solve. Every swap and
  // multiplier is recorded so `solve_polynomial_fit` can replay the identical
  // sequence on a fresh right-hand side without rebuilding `a`.
  for (std::size_t k = 0; k < m; ++k) {
    std::size_t piv  = k;
    double      best = std::fabs(a[k * m + k]);
    for (std::size_t r = k + 1; r < m; ++r) {
      const double v = std::fabs(a[r * m + k]);
      if (v > best) {
        best = v;
        piv  = r;
      }
    }
    if (!(best > 0.0)) {
      FUSE_THROW(InvariantViolation, "{}", 
          "fuse::num::fit_polynomial: singular normal equations "
          "(duplicate sample times?)");
    }
    plan.piv[k] = piv;
    if (piv != k) {
      for (std::size_t c = k; c < m; ++c) std::swap(a[k * m + c], a[piv * m + c]);
    }
    for (std::size_t r = k + 1; r < m; ++r) {
      const double fac = a[r * m + k] / a[k * m + k];
      plan.mult[r * m + k] = fac;
      if (fac == 0.0) continue;
      for (std::size_t c = k; c < m; ++c) a[r * m + c] -= fac * a[k * m + c];
    }
  }
  plan.lu = std::move(a);
  return plan;
}

/// The `f`-only half of `fit_polynomial`, writing into caller-supplied
/// buffers instead of allocating (REVIEW.md R-48, pass 2): `Waveform`'s
/// per-element loops (`evaluate_dg`/`integrate_dg`) call this once per
/// element with `out`/`rhs_scratch` hoisted outside the loop, mirroring
/// `linalg::least_squares_into`'s scratch convention, so a steady-state
/// evaluate/integrate over many elements touches no allocator. `out` is
/// resized to `plan.m` and `rhs_scratch` to `plan.m`; `vector::assign` on a
/// buffer already at that capacity does not reallocate, so only the first
/// call per buffer pair allocates. `f.size()` must equal `plan.n`.
inline void solve_polynomial_fit_into(const PolynomialFitPlan& plan,
                                      const std::vector<double>& f,
                                      std::vector<double>& out,
                                      std::vector<double>& rhs_scratch) {
  if (f.size() != plan.n) {
    FUSE_THROW(InvalidArgument, "{}", 
        "fuse::num::fit_polynomial: t and f have different lengths");
  }
  const std::size_t m = plan.m;
  rhs_scratch.assign(m, 0.0);
  for (std::size_t i = 0; i < plan.n; ++i) {
    for (std::size_t r = 0; r < m; ++r) {
      rhs_scratch[r] += plan.phi[i * m + r] * f[i];
    }
  }

  for (std::size_t k = 0; k < m; ++k) {
    if (plan.piv[k] != k) std::swap(rhs_scratch[k], rhs_scratch[plan.piv[k]]);
    for (std::size_t r = k + 1; r < m; ++r) {
      const double fac = plan.mult[r * m + k];
      if (fac == 0.0) continue;
      rhs_scratch[r] -= fac * rhs_scratch[k];
    }
  }

  out.assign(m, 0.0);
  const std::vector<double>& a = plan.lu;
  for (std::size_t i = m; i-- > 0;) {
    double s = rhs_scratch[i];
    for (std::size_t j = i + 1; j < m; ++j) s -= a[i * m + j] * out[j];
    out[i] = s / a[i * m + i];
  }

  // p(t) = sum c_k s^k with s = (t - t0)/h  ==>  sum (c_k / h^k) (t - t0)^k.
  double hk = 1.0;
  for (std::size_t k = 0; k < m; ++k) {
    out[k] /= hk;
    hk *= plan.h;
  }
}

/// The `f`-only half of `fit_polynomial`: builds the right-hand side and
/// replays `plan`'s recorded swaps/multipliers/back-substitution on it.
/// `f.size()` must equal `plan.n`. Allocating convenience wrapper around
/// `solve_polynomial_fit_into`; a caller in a hot loop should hoist its own
/// `out`/scratch buffers and call that directly instead.
[[nodiscard]] inline std::vector<double> solve_polynomial_fit(
    const PolynomialFitPlan& plan, const std::vector<double>& f) {
  std::vector<double> out;
  std::vector<double> rhs_scratch;
  solve_polynomial_fit_into(plan, f, out, rhs_scratch);
  return out;
}

[[nodiscard]] inline std::vector<double> fit_polynomial(
    const std::vector<double>& t, const std::vector<double>& f, int degree,
    double t0) {
  // N-num-linalg-field-1 (pass 2): on a call with two simultaneous errors
  // (e.g. a wrong `f` length together with a negative degree), the split
  // below runs prepare_polynomial_fit -- which only sees `t`/`degree` -- to
  // completion before solve_polynomial_fit ever sees `f`, so the length
  // check would fire last instead of second. Checking it here, ahead of the
  // split, restores the pre-split order (empty t, then length, then
  // degree) exactly; prepare_polynomial_fit cannot do this check itself
  // because it never receives `f`.
  if (t.size() == 0) {
    FUSE_THROW(InvalidArgument, "{}", "fuse::num::fit_polynomial: no samples");
  }
  if (f.size() != t.size()) {
    FUSE_THROW(InvalidArgument, "{}",
        "fuse::num::fit_polynomial: t and f have different lengths");
  }
  return solve_polynomial_fit(prepare_polynomial_fit(t, degree, t0), f);
}

/// Horner evaluation of the monomial expansion about `t0`. An empty
/// coefficient vector is the zero polynomial.
[[nodiscard]] inline double eval_polynomial(const std::vector<double>& coeffs,
                                            double t, double t0) {
  if (coeffs.empty()) return 0.0;
  const double x   = t - t0;
  double       acc = coeffs.back();
  for (std::size_t i = coeffs.size() - 1; i-- > 0;) acc = acc * x + coeffs[i];
  return acc;
}

/// Exact integral of the fitted polynomial over [a, b] -- analytic, not
/// quadrature, so there is no rule-order caveat to track.
[[nodiscard]] inline double integrate_polynomial(
    const std::vector<double>& coeffs, double a, double b, double t0) {
  const double xa  = a - t0;
  const double xb  = b - t0;
  double       pa  = xa;
  double       pb  = xb;
  double       sum = 0.0;
  for (std::size_t k = 0; k < coeffs.size(); ++k) {
    sum += coeffs[k] * (pb - pa) / static_cast<double>(k + 1);
    pa *= xa;
    pb *= xb;
  }
  return sum;
}

/// Second derivatives ("moments") of the NATURAL cubic spline through
/// (t[i], f[i]), i.e. with S'' == 0 imposed at both ends.
///
/// ACCURACY WARNING, because it is easy to assume otherwise: a natural cubic
/// spline is O(h^4) accurate only in the interior. The natural end condition
/// is wrong whenever f''(t_first) or f''(t_last) is non-zero, and that error
/// is O(h^2) and reaches a couple of intervals into the domain. In the
/// max-norm over the whole sampled range the observed order is therefore 2,
/// not 4. `WaveformOrder.SplineRefinementIsSecondOrderInTheMaxNorm` measures
/// exactly this so nobody claims 4 later.
///
/// `t` must be strictly increasing. Fewer than three samples means there is no
/// interior equation to solve and every moment is zero, which makes
/// `eval_cubic_spline` reduce to linear interpolation (n == 2) or a constant
/// (n == 1) -- the right degeneracies.
///
/// The tridiagonal system's matrix -- `lo`/`di`/`up` below -- depends only on
/// `t`; only the right-hand side `r` depends on `f`. `Waveform::evaluate_spline`/
/// `integrate_spline` called this once per field element with the same `t`
/// and a different `f` each time (REVIEW.md R-48), rebuilding and
/// re-factoring the same tridiagonal matrix every time. `cubic_spline_moments`
/// itself is unchanged; it is now the convenience wrapper around the two
/// halves below, which a multi-element caller uses directly to factor once.
[[nodiscard]] inline std::vector<double> cubic_spline_moments(
    const std::vector<double>& t, const std::vector<double>& f);

/// The `t`-only half of `cubic_spline_moments`: builds and forward-eliminates
/// the tridiagonal moment matrix, without touching `f`. `n < 3` (no interior
/// equation) is recorded as an empty plan; `solve_cubic_spline_moments`
/// checks `plan.n`, not these vectors, to decide the trivial case.
struct SplineMomentPlan {
  std::size_t n = 0;
  std::vector<double> lo;  ///< h0 at each interior row (length n - 2).
  std::vector<double> up;  ///< h1 at each interior row (length n - 2);
                            ///< unmodified by elimination, unlike `di`.
  std::vector<double> di;  ///< diagonal AFTER forward elimination.
  std::vector<double> w;   ///< elimination multiplier at each row (w[0] unused).
};

[[nodiscard]] inline SplineMomentPlan prepare_cubic_spline_moments(
    const std::vector<double>& t) {
  SplineMomentPlan plan;
  plan.n = t.size();
  if (plan.n < 3) return plan;

  const std::size_t k = plan.n - 2;  // unknowns are m[1] .. m[n-2]
  plan.lo.resize(k);
  plan.di.resize(k);
  plan.up.resize(k);
  plan.w.assign(k, 0.0);
  for (std::size_t i = 1; i + 1 < plan.n; ++i) {
    const double h0 = t[i] - t[i - 1];
    const double h1 = t[i + 1] - t[i];
    if (!(h0 > 0.0) || !(h1 > 0.0)) {
      FUSE_THROW(InvalidArgument, "{}", 
          "fuse::num::cubic_spline_moments: sample times must be strictly "
          "increasing");
    }
    plan.lo[i - 1] = h0;
    plan.di[i - 1] = 2.0 * (h0 + h1);
    plan.up[i - 1] = h1;
  }

  // Thomas algorithm, matrix half only. The system is strictly diagonally
  // dominant (2(h0+h1) > h0 + h1), so no pivoting is needed and no pivot can
  // vanish. The multipliers are recorded so `solve_cubic_spline_moments` can
  // apply the identical forward elimination to a fresh right-hand side.
  for (std::size_t i = 1; i < k; ++i) {
    plan.w[i] = plan.lo[i] / plan.di[i - 1];
    plan.di[i] -= plan.w[i] * plan.up[i - 1];
  }
  return plan;
}

/// The `f`-only half of `cubic_spline_moments`, writing into caller-supplied
/// buffers instead of allocating (REVIEW.md R-48, pass 2): `Waveform`'s
/// per-element loops (`evaluate_spline`/`integrate_spline`) call this once
/// per element with `out`/`r_scratch` hoisted outside the loop, mirroring
/// `solve_polynomial_fit_into` above. `out` is resized to `plan.n` and
/// `r_scratch` to `plan.n - 2` (or left untouched when `plan.n < 3`, the
/// no-interior-equation case); `vector::assign`/`resize` on a buffer already
/// at that capacity does not reallocate. `f.size()` must equal `plan.n`.
inline void solve_cubic_spline_moments_into(const SplineMomentPlan& plan,
                                            const std::vector<double>& f,
                                            std::vector<double>& out,
                                            std::vector<double>& r_scratch) {
  if (f.size() != plan.n) {
    FUSE_THROW(InvalidArgument, "{}", 
        "fuse::num::cubic_spline_moments: t and f have different lengths");
  }
  out.assign(plan.n, 0.0);
  if (plan.n < 3) return;

  const std::size_t k = plan.n - 2;
  r_scratch.assign(k, 0.0);
  for (std::size_t i = 1; i + 1 < plan.n; ++i) {
    r_scratch[i - 1] = 6.0 * ((f[i + 1] - f[i]) / plan.up[i - 1] -
                              (f[i] - f[i - 1]) / plan.lo[i - 1]);
  }
  for (std::size_t i = 1; i < k; ++i) r_scratch[i] -= plan.w[i] * r_scratch[i - 1];

  out[k] = r_scratch[k - 1] / plan.di[k - 1];  // out[n-2]
  for (std::size_t i = k - 1; i-- > 0;) {
    out[i + 1] = (r_scratch[i] - plan.up[i] * out[i + 2]) / plan.di[i];
  }
}

/// The `f`-only half of `cubic_spline_moments`: builds the right-hand side
/// and replays `plan`'s recorded forward elimination and back-substitution.
/// `f.size()` must equal `plan.n`. Allocating convenience wrapper around
/// `solve_cubic_spline_moments_into`; a caller in a hot loop should hoist
/// its own `out`/scratch buffers and call that directly instead.
[[nodiscard]] inline std::vector<double> solve_cubic_spline_moments(
    const SplineMomentPlan& plan, const std::vector<double>& f) {
  std::vector<double> out;
  std::vector<double> r_scratch;
  solve_cubic_spline_moments_into(plan, f, out, r_scratch);
  return out;
}

[[nodiscard]] inline std::vector<double> cubic_spline_moments(
    const std::vector<double>& t, const std::vector<double>& f) {
  // N-num-linalg-field-1 (pass 2): same reasoning as fit_polynomial above --
  // check the length here, ahead of the split, so a `t` that is both the
  // wrong length against `f` AND non-monotonic reports the length mismatch
  // first, matching the pre-split order.
  if (f.size() != t.size()) {
    FUSE_THROW(InvalidArgument, "{}",
        "fuse::num::cubic_spline_moments: t and f have different lengths");
  }
  return solve_cubic_spline_moments(prepare_cubic_spline_moments(t), f);
}

/// Evaluate the spline defined by (t, f, m) at `x`. `x` outside [t.front(),
/// t.back()] is clamped to the end interval's cubic, i.e. it EXTRAPOLATES.
/// That is deliberate at this level (it keeps the kernel total), and it is
/// exactly why `Waveform::evaluate` range-checks before calling: extrapolation
/// is `Extrapolate`'s job per spec §5.1, not an interpolator's.
[[nodiscard]] inline double eval_cubic_spline(const std::vector<double>& t,
                                              const std::vector<double>& f,
                                              const std::vector<double>& m,
                                              double                     x) {
  const std::size_t n = t.size();
  if (n == 0) {
    FUSE_THROW(InvalidArgument, "{}", "fuse::num::eval_cubic_spline: no samples");
  }
  if (f.size() != n || m.size() != n) {
    FUSE_THROW(InvalidArgument, "{}", 
        "fuse::num::eval_cubic_spline: t, f and m have different lengths");
  }
  if (n == 1) return f[0];

  std::size_t j  = 0;
  const auto  it = std::upper_bound(t.begin(), t.end(), x);
  if (it == t.begin()) {
    j = 0;
  } else if (it == t.end()) {
    j = n - 2;
  } else {
    j = static_cast<std::size_t>(it - t.begin()) - 1;
  }

  const double h = t[j + 1] - t[j];
  const double A = (t[j + 1] - x) / h;
  const double B = (x - t[j]) / h;
  return A * f[j] + B * f[j + 1] +
         ((A * A * A - A) * m[j] + (B * B * B - B) * m[j + 1]) * h * h / 6.0;
}

/// Integral of the spline over [a, b], EXACT. Each sample interval carries a
/// cubic, and the two-point Gauss-Legendre rule is exact through degree 3, so
/// clipping to [a, b] and applying it per piece has no quadrature error --
/// only rounding. Returns 0 when b <= a.
[[nodiscard]] inline double integrate_cubic_spline(
    const std::vector<double>& t, const std::vector<double>& f,
    const std::vector<double>& m, double a, double b) {
  const std::size_t n = t.size();
  if (n == 0) {
    FUSE_THROW(InvalidArgument, "{}", "fuse::num::integrate_cubic_spline: no samples");
  }
  if (f.size() != n || m.size() != n) {
    FUSE_THROW(InvalidArgument, "{}", 
        "fuse::num::integrate_cubic_spline: t, f and m have different lengths");
  }
  if (!(b > a)) return 0.0;
  if (n == 1) return f[0] * (b - a);

  const GaussRule g   = gauss_rule(2);
  double          sum = 0.0;
  for (std::size_t j = 0; j + 1 < n; ++j) {
    const double lo = std::max(a, t[j]);
    const double hi = std::min(b, t[j + 1]);
    if (!(hi > lo)) continue;
    const double mid  = 0.5 * (lo + hi);
    const double half = 0.5 * (hi - lo);
    for (int q = 0; q < g.count; ++q) {
      sum += g.weights[q] * half *
             eval_cubic_spline(t, f, m, mid + half * g.nodes[q]);
    }
  }
  return sum;
}

}  // namespace fuse::num
