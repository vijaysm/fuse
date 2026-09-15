#pragma once

#include <cmath>
#include <cstddef>

#include "soma/base/ids.hpp"

/// Spec §3.1 tier C and §5.1: the primitive numeric kernels the plan
/// interpreter dispatches to.
///
/// Floating-point accumulation order (standing convention for this header):
/// every reduction here -- `sum`, `weighted_sum`, `norm_l2`, `norm_l2_diff`,
/// and the running `accumulate_into` -- accumulates strictly sequentially in
/// ascending index order into a single scalar of type `T`. That makes each
/// one bit-reproducible for a given input buffer and length, run to run and
/// build to build, and `Kernels.SumAccumulatesInAscendingIndexOrder` pins it.
/// It is NOT invariant under reassociation: a partitioned or vectorised
/// reduction over the same values will generally give a different last bit,
/// so partial sums computed over different decompositions (rank counts,
/// thread counts, block sizes) must not be compared for equality. None of
/// these kernels compensates (no Kahan/Neumaier); a global conservation check
/// that needs decomposition independence has to arrange that itself.
///
/// The kernels take no responsibility for aliasing: `out` must not overlap an
/// input unless the operation is documented as in-place.
namespace soma::num {

template <class T>
void copy(const T* src, T* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
}

/// Running reduction. sample_index == 0 initialises the accumulator, so `acc`
/// need not be pre-zeroed and a stale accumulator cannot leak into a new
/// window. Sum/Mean/Integral add in call order, i.e. in sample order.
template <class T>
void accumulate_into(AccumOp op, const T* sample, T* acc, std::size_t n,
                     std::size_t sample_index) {
  const bool first = (sample_index == 0);
  for (std::size_t i = 0; i < n; ++i) {
    switch (op) {
      case AccumOp::Instant:
        acc[i] = sample[i];
        break;
      case AccumOp::Sum:
      case AccumOp::Mean:
      case AccumOp::Integral:
        acc[i] = first ? sample[i] : acc[i] + sample[i];
        break;
      case AccumOp::Min:
        acc[i] = first ? sample[i] : (sample[i] < acc[i] ? sample[i] : acc[i]);
        break;
      case AccumOp::Max:
        acc[i] = first ? sample[i] : (sample[i] > acc[i] ? sample[i] : acc[i]);
        break;
    }
  }
}

/// Mean divides by the sample count; everything else is already final.
/// Integral is finalised by accumulate_finalize_integral instead. A zero
/// sample count leaves the accumulator untouched rather than dividing by 0.
template <class T>
void accumulate_finalize(AccumOp op, T* acc, std::size_t n,
                         std::size_t sample_count) {
  if (op != AccumOp::Mean || sample_count == 0) return;
  const T inv = T{1} / static_cast<T>(sample_count);
  for (std::size_t i = 0; i < n; ++i) acc[i] *= inv;
}

/// Finalises an AccumOp::Integral accumulator. `dt` is the interval *between
/// samples*, not the window length: `accumulate_into` has already formed
/// sum_k y_k, and this scales it to the left-rectangle quadrature
/// (sum_k y_k) * dt, whose units are [field] x [time]. A caller with
/// non-uniform sample spacing must weight the samples itself.
template <class T>
void accumulate_finalize_integral(T* acc, std::size_t n, T dt) {
  for (std::size_t i = 0; i < n; ++i) acc[i] *= dt;
}

/// out = omega*current + (1 - omega)*previous
template <class T>
void relax(T omega, const T* previous, const T* current, T* out,
           std::size_t n) {
  const T one_minus = T{1} - omega;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = omega * current[i] + one_minus * previous[i];
  }
}

/// `history` points at the NEWEST sample; the sample before it starts
/// `stride` elements earlier, and each sample holds `n` elements.
/// order 0: y_k          order 1: 2y_k - y_{k-1}
/// order 2: 3y_k - 3y_{k-1} + y_{k-2}
/// The order is clamped to [0, 2]; in particular a negative order degrades to
/// a copy rather than reading before `history`. The caller must guarantee
/// that `order` samples of history actually exist before `history`.
template <class T>
void extrapolate(int order, const T* history, std::size_t stride, T* out,
                 std::size_t n) {
  const int k = order < 0 ? 0 : (order > 2 ? 2 : order);
  // Form the older samples as pointer arithmetic on the caller's buffer.
  // `history[i - stride]` would instead compute history + (SIZE_MAX - ...)
  // whenever i < stride: pointer overflow, which UBSan reports and which is
  // only accidentally correct on wrapping targets.
  const T* const older1 = (k >= 1) ? history - stride : nullptr;
  const T* const older2 = (k >= 2) ? history - 2 * stride : nullptr;

  for (std::size_t i = 0; i < n; ++i) {
    const T y0 = history[i];
    switch (k) {
      case 0:
        out[i] = y0;
        break;
      case 1:
        out[i] = T{2} * y0 - older1[i];
        break;
      default:
        out[i] = T{3} * y0 - T{3} * older1[i] + older2[i];
        break;
    }
  }
}

/// Sequential in ascending index order; see the header note on accumulation.
template <class T>
[[nodiscard]] T sum(const T* v, std::size_t n) {
  T s{};
  for (std::size_t i = 0; i < n; ++i) s += v[i];
  return s;
}

/// Conservation checking uses this with MeshView::measure as weights, so the
/// result is a true integral over the region: `v` carries the intensive field
/// value per element and `w` the element measure (area/volume), giving
/// units [field] x [measure]. Sequential in ascending index order.
template <class T>
[[nodiscard]] T weighted_sum(const T* v, const T* w, std::size_t n) {
  T s{};
  for (std::size_t i = 0; i < n; ++i) s += v[i] * w[i];
  return s;
}

/// Sequential in ascending index order; no scaling is applied to avoid
/// overflow, so very large inputs can overflow the squared accumulator.
template <class T>
[[nodiscard]] T norm_l2(const T* v, std::size_t n) {
  T s{};
  for (std::size_t i = 0; i < n; ++i) s += v[i] * v[i];
  return std::sqrt(s);
}

template <class T>
[[nodiscard]] T norm_linf(const T* v, std::size_t n) {
  T m{};
  for (std::size_t i = 0; i < n; ++i) {
    const T a = v[i] < T{} ? -v[i] : v[i];
    if (a > m) m = a;
  }
  return m;
}

/// Sequential in ascending index order; see the header note on accumulation.
template <class T>
[[nodiscard]] T norm_l2_diff(const T* a, const T* b, std::size_t n) {
  T s{};
  for (std::size_t i = 0; i < n; ++i) {
    const T d = a[i] - b[i];
    s += d * d;
  }
  return std::sqrt(s);
}

/// NaN/Inf detection. Correct only under IEEE semantics: building this header
/// with -ffast-math / -ffinite-math-only lets the compiler fold it to false.
template <class T>
[[nodiscard]] bool has_nonfinite(const T* v, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) return true;
  }
  return false;
}

// Spec §3.1: instantiate the common cases once, keep the templates available.
extern template void   copy<double>(const double*, double*, std::size_t);
extern template void   copy<float>(const float*, float*, std::size_t);
extern template double norm_l2<double>(const double*, std::size_t);
extern template float  norm_l2<float>(const float*, std::size_t);
extern template double weighted_sum<double>(const double*, const double*,
                                            std::size_t);

}  // namespace soma::num
