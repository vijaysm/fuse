#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/num/temporal_basis.hpp"

namespace fuse {

/// Spec §4.4: a field sampled at several points across a coupling window plus
/// an interpolation policy. It is the base abstraction for coupled data in
/// time -- **explicit coupling is the degenerate case** (one sample, `Constant`
/// basis; see `is_degenerate()`), not a separate code path -- and it is the
/// carrier the space-time projection of §7.3 reads from.
///
/// The point of the class is that one component samples the waveform on *its*
/// interval and the other evaluates or integrates it at times *it* chooses.
/// Everything below exists to make that safe.
///
/// ---------------------------------------------------------------------------
/// TIME IS EXACT
/// ---------------------------------------------------------------------------
/// Sample times and window bounds are `Rational` (spec §4.3). Nothing here
/// accumulates time in floating point. Conversions to `double` happen only
/// after an exact difference has been formed -- `(t - window_begin)` -- so the
/// basis kernels see *window-local* times and never lose significant digits to
/// a large absolute simulated time. Sample *values* are of course `double`.
///
/// ---------------------------------------------------------------------------
/// WHAT HAPPENS OUTSIDE THE SAMPLED RANGE (read this before using `integrate`)
/// ---------------------------------------------------------------------------
/// A waveform has a *window* `[window_begin(), window_end()]`, fixed by
/// `reset()`, and a *support* `[support_begin(), support_end()]`, determined by
/// the samples actually pushed and by the basis:
///
///   basis      support
///   --------   ----------------------------------------------------------
///   Constant   [t_first, window_end]   zero-order hold: sample i is the
///                                      representation on [t_i, t_{i+1}), and
///                                      the last sample holds to the END OF
///                                      THE WINDOW. That hold *is* the
///                                      `Constant` basis (it is what makes
///                                      `Accumulate`+`Remap` == `temporal =
///                                      Constant`, spec §7.3), and it is what
///                                      makes the one-sample explicit case
///                                      have a well-defined window mean.
///   Linear     [t_first, t_last]
///   DG(p)      [t_first, t_last]
///   Spline(p)  [t_first, t_last]
///
/// `evaluate` and `integrate` REFUSE anything outside the support:
/// `E-WF-OUTSIDE` and `E-WF-RANGE` respectively. In particular nothing here
/// ever extrapolates past `t_last` for `Linear`/`DG`/`Spline`, and nothing ever
/// extrapolates *backwards* before `t_first` for any basis. Silent
/// extrapolation is how a coupled scheme loses its order of accuracy with no
/// error message; per spec §5.1 prediction beyond known data is `Extrapolate`,
/// a different operator with different legality.
///
/// A consequence worth stating: `mean()` integrates over the whole window, so
/// for `Linear`/`DG`/`Spline` it fails with `E-WF-RANGE` unless the samples
/// actually span the window. That is the intended answer -- you cannot average
/// over time you did not sample.
///
/// ---------------------------------------------------------------------------
/// ORDER OF ACCURACY (measured in `waveform_order_test.cpp`)
/// ---------------------------------------------------------------------------
/// Over a window of length h carrying the minimum interpolatory sample count.
/// The right-hand column is what the test actually measured on f = exp(4t)
/// over h = 1/32 -> 1/64, not what the theory says:
///
///                          claimed   measured   integral over the window
///   Constant,  1 sample      1        1.04       2  (2.06)
///   Linear,    2 samples     2        2.05       3  (3.09)
///   DG(1),     2 samples     2        2.05
///   DG(2),     3 samples     3        3.05       5  (5.09)
///   DG(3),     4 samples     4        4.05
///   Spline(3), 4 samples     2        2.08
///
/// Two traps are pinned by tests rather than left to be discovered in Phase
/// 6.5:
///
///  * **DG(p) does not converge under sample refinement.** With the window and
///    `p` fixed, adding samples drives the least-squares fit to the best
///    degree-`p` approximation of f over the window and the error SATURATES.
///    Refining a DG(p) waveform means shortening the window, not sampling it
///    more often.
///  * **A natural cubic spline is second order, not fourth,** wherever the
///    end condition S'' == 0 is felt. That condition is wrong whenever f'' is
///    non-zero at the ends and it contaminates an O(h) layer at each end.
///    Under sample refinement of a fixed window the measured orders are 3.98
///    at interior knot midpoints and 2.00 inside the first interval; with only
///    four samples on a shrinking window every interval is a boundary
///    interval, so the window order is 2.
///
/// ---------------------------------------------------------------------------
/// MEMORY / POINTER VALIDITY
/// ---------------------------------------------------------------------------
/// The value buffer is sized once, in the constructor, to
/// `capacity * elements`, and `times_` is reserved to `capacity`. `push()`
/// therefore never reallocates either vector, so a pointer obtained from
/// `sample_values(k)` stays valid across subsequent `push()` calls. It is
/// invalidated only by destruction; `reset()` leaves it valid but the data it
/// points at will be overwritten by the next `push()`.
///
/// ---------------------------------------------------------------------------
/// ERROR CODES
/// ---------------------------------------------------------------------------
///   E-WF-FULL      push past `capacity`
///   E-WF-ORDER     push with a time not strictly greater than the last
///   E-WF-WINDOW    push outside `[window_begin, window_end]`; `mean` on a
///                  zero-length window
///   E-WF-EMPTY     evaluate/integrate/mean with no samples
///   E-WF-OUTSIDE   evaluate outside the support (see above)
///   E-WF-RANGE     integrate over an interval that is not inside the support,
///                  or with b < a
///   E-WF-NULL      a null buffer where `elements() > 0`
///   E-WF-BASIS     defensive only; the constructor rejects every basis that
///                  the evaluators do not handle
///
/// Constructor and `reset()` argument errors are *programmer* errors and
/// THROW (`std::invalid_argument`); out-of-range sample indices throw
/// `std::out_of_range`. They are exceptions rather than asserts because
/// `assert` compiles out under `-DNDEBUG`, and because `reset()` and the
/// accessors have no `Result` to return.
class Waveform {
 public:
  /// Highest DG degree accepted. The least-squares fit uses normal equations,
  /// which square the Vandermonde condition number; with the internal scaling
  /// in `fit_polynomial` the degree-3 fit reproduces an exact cubic to ~1e-14
  /// independently of the window length (measured), but the loss grows quickly
  /// with degree, and returning a fit that does not deliver its claimed order
  /// is worse than refusing. Task 6.3 should move to an orthogonal basis if it
  /// needs more.
  static constexpr int kMaxDgDegree = 4;

  /// `elements` may be 0 (an empty mesh partition on this rank is legal).
  /// `degree` must match the basis: 0 for `Constant`, 1 for `Linear`,
  /// 0..kMaxDgDegree for `DG`, and exactly 3 for `Spline` (only the natural
  /// cubic is implemented -- accepting `Spline(5)` and quietly building a
  /// cubic would be a silent order loss). `TemporalBasis::None` means "no
  /// temporal representation" (spec §7.3) and is not a waveform.
  ///
  /// The window starts empty at `[0, 0]`; call `reset()` before `push()`.
  Waveform(std::size_t elements, std::size_t capacity, TemporalBasis basis,
           int degree)
      : elements_(elements), capacity_(capacity), basis_(basis),
        degree_(degree) {
    switch (basis) {
      case TemporalBasis::None:
        FUSE_THROW(InvalidArgument, "{}", 
            "Waveform: TemporalBasis::None is not a temporal representation");
      case TemporalBasis::Constant:
        if (degree != 0) {
          FUSE_THROW(InvalidArgument, "{}", 
              "Waveform: Constant basis requires degree 0, got " +
              std::to_string(degree));
        }
        break;
      case TemporalBasis::Linear:
        if (degree != 1) {
          FUSE_THROW(InvalidArgument, "{}", 
              "Waveform: Linear basis requires degree 1, got " +
              std::to_string(degree));
        }
        break;
      case TemporalBasis::DG:
        if (degree < 0 || degree > kMaxDgDegree) {
          FUSE_THROW(InvalidArgument, "{}", 
              "Waveform: DG degree must be in [0, " +
              std::to_string(kMaxDgDegree) + "], got " + std::to_string(degree));
        }
        break;
      case TemporalBasis::Spline:
        if (degree != 3) {
          FUSE_THROW(InvalidArgument, "{}", 
              "Waveform: only the natural cubic spline is implemented; "
              "Spline requires degree 3, got " +
              std::to_string(degree));
        }
        break;
    }
    if (elements != 0 &&
        capacity > std::numeric_limits<std::size_t>::max() / elements) {
      FUSE_THROW(InvalidArgument, "{}", 
          "Waveform: capacity * elements overflows size_t");
    }
    values_.assign(capacity * elements, 0.0);
    times_.reserve(capacity);
  }

  /// Clear the samples and set the window. Capacity, basis and degree are
  /// unchanged, and no buffer is released, so a waveform is reused across
  /// windows without reallocating.
  ///
  /// Throws `std::invalid_argument` if `window_end < window_begin`. A
  /// zero-length window is legal (a single instantaneous sample); `mean()`
  /// then reports `E-WF-WINDOW` rather than dividing by zero.
  void reset(Rational window_begin, Rational window_end) {
    if (window_end < window_begin) {
      FUSE_THROW(InvalidArgument, "{}", 
          "Waveform::reset: window_end (" + window_end.to_string() +
          ") precedes window_begin (" + window_begin.to_string() + ")");
    }
    begin_ = window_begin;
    end_   = window_end;
    times_.clear();
  }

  /// Record one sample. `values` must point at `elements()` doubles; they are
  /// copied. Sample times must be strictly increasing and inside the window.
  /// Neither is clamped: a clamp would turn a scheduling bug into a silently
  /// wrong waveform.
  Result<void> push(Rational t, const double* values) {
    if (times_.size() >= capacity_) {
      return Failure{"E-WF-FULL", "Waveform::push: capacity " +
                                    std::to_string(capacity_) + " exhausted"};
    }
    if (elements_ > 0 && values == nullptr) {
      return Failure{"E-WF-NULL", "Waveform::push: null values buffer"};
    }
    if (t < begin_ || t > end_) {
      return Failure{"E-WF-WINDOW", "Waveform::push: t=" + t.to_string() +
                                      " is outside the window [" +
                                      begin_.to_string() + ", " +
                                      end_.to_string() + "]"};
    }
    if (!times_.empty() && !(t > times_.back())) {
      return Failure{"E-WF-ORDER", "Waveform::push: t=" + t.to_string() +
                                     " does not strictly follow the previous "
                                     "sample time " +
                                     times_.back().to_string()};
    }
    const std::size_t row = times_.size() * elements_;
    for (std::size_t e = 0; e < elements_; ++e) values_[row + e] = values[e];
    times_.push_back(t);
    return Result<void>::success();
  }

  /// Interpolate INSIDE the support (see the class comment). `out` must point
  /// at `elements()` doubles. Beyond the support this returns `E-WF-OUTSIDE`;
  /// prediction past known data is `Extrapolate`'s job (spec §5.1).
  [[nodiscard]] Result<void> evaluate(Rational t, double* out) const {
    if (times_.empty()) {
      return Failure{"E-WF-EMPTY", "Waveform::evaluate: no samples"};
    }
    if (elements_ > 0 && out == nullptr) {
      return Failure{"E-WF-NULL", "Waveform::evaluate: null out buffer"};
    }
    if (t < support_begin() || t > support_end()) {
      return Failure{"E-WF-OUTSIDE",
                   "Waveform::evaluate: t=" + t.to_string() +
                       " is outside the support [" +
                       support_begin().to_string() + ", " +
                       support_end().to_string() + "]"};
    }
    if (elements_ == 0) return Result<void>::success();

    switch (basis_) {
      case TemporalBasis::Constant:
        copy_row(index_at_or_before(t), out);
        return Result<void>::success();
      case TemporalBasis::Linear:
        evaluate_linear(t, out);
        return Result<void>::success();
      case TemporalBasis::DG:
        evaluate_dg(t, out);
        return Result<void>::success();
      case TemporalBasis::Spline:
        evaluate_spline(t, out);
        return Result<void>::success();
      case TemporalBasis::None:
        break;  // rejected in the constructor
    }
    return Failure{"E-WF-BASIS", "Waveform::evaluate: unsupported basis"};
  }

  /// int_a^b f dt, per element -- the primitive the space-time conservation
  /// check of spec §7.4 needs. `[a, b]` must lie inside the support; anything
  /// else is `E-WF-RANGE`, for the same reason `evaluate` refuses to
  /// extrapolate. Exact for `Constant`, `Linear` (piecewise-linear
  /// representation) and `DG` (analytic integral of the fitted polynomial),
  /// and exact for `Spline` up to rounding (two-point Gauss per cubic piece).
  [[nodiscard]] Result<void> integrate(Rational a, Rational b,
                                       double* out) const {
    if (times_.empty()) {
      return Failure{"E-WF-EMPTY", "Waveform::integrate: no samples"};
    }
    if (elements_ > 0 && out == nullptr) {
      return Failure{"E-WF-NULL", "Waveform::integrate: null out buffer"};
    }
    if (b < a) {
      return Failure{"E-WF-RANGE", "Waveform::integrate: b=" + b.to_string() +
                                     " precedes a=" + a.to_string()};
    }
    if (a < support_begin() || b > support_end()) {
      return Failure{"E-WF-RANGE",
                   "Waveform::integrate: [" + a.to_string() + ", " +
                       b.to_string() + "] is not inside the support [" +
                       support_begin().to_string() + ", " +
                       support_end().to_string() + "]"};
    }
    if (elements_ == 0) return Result<void>::success();

    for (std::size_t e = 0; e < elements_; ++e) out[e] = 0.0;
    if (a == b) return Result<void>::success();

    switch (basis_) {
      case TemporalBasis::Constant:
        integrate_constant(a, b, out);
        return Result<void>::success();
      case TemporalBasis::Linear:
        integrate_linear(a, b, out);
        return Result<void>::success();
      case TemporalBasis::DG:
        integrate_dg(a, b, out);
        return Result<void>::success();
      case TemporalBasis::Spline:
        integrate_spline(a, b, out);
        return Result<void>::success();
      case TemporalBasis::None:
        break;  // rejected in the constructor
    }
    return Failure{"E-WF-BASIS", "Waveform::integrate: unsupported basis"};
  }

  /// The window average, `integrate(window_begin, window_end) / (end - begin)`.
  /// `E-WF-WINDOW` on a zero-length window (no division by zero), and
  /// `E-WF-RANGE` when the samples do not span the window.
  [[nodiscard]] Result<void> mean(double* out) const {
    if (times_.empty()) {
      return Failure{"E-WF-EMPTY", "Waveform::mean: no samples"};
    }
    if (elements_ > 0 && out == nullptr) {
      return Failure{"E-WF-NULL", "Waveform::mean: null out buffer"};
    }
    const Rational length = end_ - begin_;
    if (!(length > Rational{0})) {
      return Failure{"E-WF-WINDOW",
                   "Waveform::mean: window [" + begin_.to_string() + ", " +
                       end_.to_string() + "] has zero length"};
    }
    const Result<void> r = integrate(begin_, end_, out);
    if (!r.ok()) return r;
    const double inv = 1.0 / length.to_double();
    for (std::size_t e = 0; e < elements_; ++e) out[e] *= inv;
    return Result<void>::success();
  }

  [[nodiscard]] std::size_t sample_count() const noexcept {
    return times_.size();
  }
  [[nodiscard]] std::size_t   elements() const noexcept { return elements_; }
  [[nodiscard]] std::size_t   capacity() const noexcept { return capacity_; }
  [[nodiscard]] TemporalBasis basis() const noexcept { return basis_; }
  [[nodiscard]] int           degree() const noexcept { return degree_; }

  /// Throws `std::out_of_range` -- an index check that must survive `-DNDEBUG`.
  [[nodiscard]] Rational sample_time(std::size_t i) const {
    check_index(i, "sample_time");
    return times_[i];
  }
  /// Points at `elements()` doubles. See the pointer-validity note above:
  /// stays valid across `push()`.
  [[nodiscard]] const double* sample_values(std::size_t i) const {
    check_index(i, "sample_values");
    return row_ptr(i);
  }

  [[nodiscard]] Rational window_begin() const noexcept { return begin_; }
  [[nodiscard]] Rational window_end() const noexcept { return end_; }

  /// First time at which the waveform is defined. Precondition: at least one
  /// sample; throws otherwise.
  [[nodiscard]] Rational support_begin() const {
    if (times_.empty()) {
      FUSE_THROW(InvalidArgument, "{}", "Waveform::support_begin: no samples");
    }
    return times_.front();
  }
  /// Last time at which the waveform is defined: the end of the window for
  /// `Constant` (zero-order hold), the last sample time otherwise.
  [[nodiscard]] Rational support_end() const {
    if (times_.empty()) {
      FUSE_THROW(InvalidArgument, "{}", "Waveform::support_end: no samples");
    }
    return basis_ == TemporalBasis::Constant ? end_ : times_.back();
  }

  /// Spec §4.4's explicit-coupling case: one sample, constant interpolation.
  [[nodiscard]] bool is_degenerate() const noexcept {
    return times_.size() == 1 && basis_ == TemporalBasis::Constant;
  }

 private:
  void check_index(std::size_t i, const char* what) const {
    if (i >= times_.size()) {
      FUSE_THROW(InvalidArgument, "{}", std::string("Waveform::") + what + ": index " +
                              std::to_string(i) + " >= sample_count " +
                              std::to_string(times_.size()));
    }
  }

  [[nodiscard]] const double* row_ptr(std::size_t i) const noexcept {
    return values_.data() + i * elements_;
  }

  /// Written out rather than calling num::copy: kernels.hpp declares
  /// `extern template copy<double>`, so using it would give this otherwise
  /// header-only tier-C header a link dependency on soma_core.
  void copy_row(std::size_t i, double* out) const noexcept {
    const double* const src = row_ptr(i);
    for (std::size_t e = 0; e < elements_; ++e) out[e] = src[e];
  }

  /// Window-local time as a double. The subtraction is EXACT (Rational); only
  /// the final narrowing is floating point, so a large absolute simulated time
  /// costs nothing in precision. Rational's operator- throws if the exact
  /// difference is not representable in int64/int64, which is the same
  /// contract every other time computation in FUSE has.
  [[nodiscard]] double local(Rational t) const { return (t - begin_).to_double(); }

  /// Index of the last sample with times_[i] <= t. Precondition: t >=
  /// times_.front(), guaranteed by the support check.
  [[nodiscard]] std::size_t index_at_or_before(Rational t) const {
    const auto it = std::upper_bound(times_.begin(), times_.end(), t);
    return static_cast<std::size_t>(it - times_.begin()) - 1;
  }

  /// Index j with times_[j] <= t <= times_[j+1]. Precondition: >= 2 samples
  /// and t inside [t_first, t_last].
  [[nodiscard]] std::size_t bracket_index(Rational t) const {
    const auto it = std::upper_bound(times_.begin(), times_.end(), t);
    if (it == times_.end()) return times_.size() - 2;  // t == t_last
    return static_cast<std::size_t>(it - times_.begin()) - 1;
  }

  void evaluate_linear(Rational t, double* out) const {
    const std::size_t n = times_.size();
    if (n == 1) {
      copy_row(0, out);
      return;
    }
    const std::size_t j  = bracket_index(t);
    const double      dt = (times_[j + 1] - times_[j]).to_double();
    const double      w  = (t - times_[j]).to_double() / dt;
    const double* const lo = row_ptr(j);
    const double* const hi = row_ptr(j + 1);
    for (std::size_t e = 0; e < elements_; ++e) {
      out[e] = (1.0 - w) * lo[e] + w * hi[e];
    }
  }

  [[nodiscard]] std::vector<double> local_times() const {
    std::vector<double> tt(times_.size());
    for (std::size_t i = 0; i < times_.size(); ++i) tt[i] = local(times_[i]);
    return tt;
  }

  void gather(std::size_t e, std::vector<double>& ff) const {
    for (std::size_t i = 0; i < times_.size(); ++i) {
      ff[i] = values_[i * elements_ + e];
    }
  }

  [[nodiscard]] int effective_dg_degree() const noexcept {
    return std::min(degree_, static_cast<int>(times_.size()) - 1);
  }

  void evaluate_dg(Rational t, double* out) const {
    const std::vector<double> tt  = local_times();
    const int                 deg = effective_dg_degree();
    const double              x   = local(t);
    std::vector<double>       ff(times_.size());
    for (std::size_t e = 0; e < elements_; ++e) {
      gather(e, ff);
      out[e] = num::eval_polynomial(num::fit_polynomial(tt, ff, deg, 0.0), x,
                                    0.0);
    }
  }

  void evaluate_spline(Rational t, double* out) const {
    const std::vector<double> tt = local_times();
    const double              x  = local(t);
    std::vector<double>       ff(times_.size());
    for (std::size_t e = 0; e < elements_; ++e) {
      gather(e, ff);
      out[e] = num::eval_cubic_spline(tt, ff, num::cubic_spline_moments(tt, ff),
                                      x);
    }
  }

  void integrate_constant(Rational a, Rational b, double* out) const {
    const std::size_t n = times_.size();
    for (std::size_t i = 0; i < n; ++i) {
      // Sample i holds over [t_i, t_{i+1}); the last one holds to end_.
      const Rational piece_lo = times_[i];
      const Rational piece_hi = (i + 1 < n) ? times_[i + 1] : end_;
      const Rational lo       = std::max(a, piece_lo);
      const Rational hi       = std::min(b, piece_hi);
      if (!(hi > lo)) continue;
      const double        dt  = (hi - lo).to_double();
      const double* const val = row_ptr(i);
      for (std::size_t e = 0; e < elements_; ++e) out[e] += val[e] * dt;
    }
  }

  void integrate_linear(Rational a, Rational b, double* out) const {
    const std::size_t n = times_.size();
    for (std::size_t j = 0; j + 1 < n; ++j) {
      const Rational lo = std::max(a, times_[j]);
      const Rational hi = std::min(b, times_[j + 1]);
      if (!(hi > lo)) continue;
      const double span = (times_[j + 1] - times_[j]).to_double();
      const double wlo  = (lo - times_[j]).to_double() / span;
      const double whi  = (hi - times_[j]).to_double() / span;
      const double dt   = (hi - lo).to_double();
      const double* const p0 = row_ptr(j);
      const double* const p1 = row_ptr(j + 1);
      for (std::size_t e = 0; e < elements_; ++e) {
        const double flo = (1.0 - wlo) * p0[e] + wlo * p1[e];
        const double fhi = (1.0 - whi) * p0[e] + whi * p1[e];
        out[e] += 0.5 * (flo + fhi) * dt;  // exact: the piece is linear
      }
    }
  }

  void integrate_dg(Rational a, Rational b, double* out) const {
    const std::vector<double> tt  = local_times();
    const int                 deg = effective_dg_degree();
    const double              xa  = local(a);
    const double              xb  = local(b);
    std::vector<double>       ff(times_.size());
    for (std::size_t e = 0; e < elements_; ++e) {
      gather(e, ff);
      out[e] = num::integrate_polynomial(num::fit_polynomial(tt, ff, deg, 0.0),
                                         xa, xb, 0.0);
    }
  }

  void integrate_spline(Rational a, Rational b, double* out) const {
    const std::vector<double> tt = local_times();
    const double              xa = local(a);
    const double              xb = local(b);
    std::vector<double>       ff(times_.size());
    for (std::size_t e = 0; e < elements_; ++e) {
      gather(e, ff);
      out[e] = num::integrate_cubic_spline(
          tt, ff, num::cubic_spline_moments(tt, ff), xa, xb);
    }
  }

  std::size_t           elements_;
  std::size_t           capacity_;
  TemporalBasis         basis_;
  int                   degree_;
  Rational              begin_{0}, end_{0};
  std::vector<Rational> times_;
  std::vector<double>   values_;  // capacity_ * elements_, row per sample
};

}  // namespace fuse
