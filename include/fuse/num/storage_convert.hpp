#pragma once

/// Conversion of a Storage field (binary32, bfloat16, binary16, posit16, posit32) to and
/// from binary64, as kernels on a Kokkos execution space.
///
/// `widen` is exact. `narrow` rounds each value once, to nearest with ties to
/// even (`value_traits<T>::from_binary64`), which is the only rounding a value
/// takes between the component that wrote it and the component that reads it
/// back in the narrow format: everything between runs in binary64.
///
/// `narrow_measured` also reports what the rounding did, as a reduction over
/// the same pass: the largest relative change, and how many finite values
/// became an infinity (they did not fit the format's range) or a zero (they
/// were below half its smallest subnormal). The relative change of a value in
/// the format's normal range is at most the unit roundoff 2^-p (p = 24 for
/// binary32, 11 for binary16, 8 for bfloat16); a value that rounds to a
/// subnormal can change by more, and the report does not hide it.
///
/// A posit has no infinity and no zero to fall to, so a value beyond its largest
/// (or below its smallest positive) value is clamped to it, silently as far as the
/// data goes; the report counts such a value as an overflow (an underflow). The
/// precision of a posit is that of the value one at best and falls by a bit for
/// every factor of 2^(2^es) away from it, so `max_relative_change` is the figure to
/// read, not 2^-p.
///
/// Both are Space-generic and take unmanaged views, like the kernels in
/// fuse/num/space_kernels.hpp; the labels stay under the small-string limit
/// that file explains. The functors are named types for the reason given there.

#include <cstddef>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "fuse/model/value_traits.hpp"
#include "fuse/num/space_kernels.hpp"

namespace fuse::num {

template <class Space, class T>
using StorageSpanOf = Kokkos::View<T*, typename Space::memory_space,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <class Space, class T>
using ConstStorageSpanOf = Kokkos::View<const T*, typename Space::memory_space,
                                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/// What one narrowing pass did. All fields are 0 for an empty or exactly
/// representable input.
struct RoundingReport {
  std::int64_t values = 0;     ///< finite values examined
  std::int64_t overflow = 0;   ///< finite values that became an infinity
  std::int64_t underflow = 0;  ///< nonzero values that became a zero
  double max_relative_change = 0.0;  ///< max |rounded - x| / |x| over the rest
};

template <class Space, class T>
struct WidenFunctor {
  ConstStorageSpanOf<Space, T> in;
  SpanOf<Space> out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    out(i) = value_traits<T>::to_binary64(in(i));
  }
};

template <class Space, class T>
struct NarrowFunctor {
  ConstSpanOf<Space> in;
  StorageSpanOf<Space, T> out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    out(i) = value_traits<T>::from_binary64(in(i));
  }
};

template <class Space, class T>
struct NarrowMeasuredFunctor {
  using value_type = RoundingReport;
  ConstSpanOf<Space> in;
  StorageSpanOf<Space, T> out;

  KOKKOS_INLINE_FUNCTION void init(value_type& r) const { r = RoundingReport{}; }
  KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const {
    dst.values += src.values;
    dst.overflow += src.overflow;
    dst.underflow += src.underflow;
    if (src.max_relative_change > dst.max_relative_change) {
      dst.max_relative_change = src.max_relative_change;
    }
  }
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i, value_type& r) const {
    const double x = in(i);
    const T y = value_traits<T>::from_binary64(x);
    out(i) = y;
    if (!Kokkos::isfinite(x)) return;
    ++r.values;
    const double w = value_traits<T>::to_binary64(y);
    if constexpr (requires { T::saturates; }) {
      // clamped to the largest or the smallest positive value
      const double a = Kokkos::abs(x);
      if (a > T::maxpos().to_double()) {
        ++r.overflow;
        return;
      }
      if (a != 0.0 && a < T::minpos().to_double()) {
        ++r.underflow;
        return;
      }
    }
    if (!Kokkos::isfinite(w)) {
      ++r.overflow;
      return;
    }
    if (x == 0.0) return;
    if (w == 0.0) {
      ++r.underflow;
      return;
    }
    const double rel = Kokkos::abs(w - x) / Kokkos::abs(x);
    if (rel > r.max_relative_change) r.max_relative_change = rel;
  }
};

/// `out(i) = binary64(in(i))`, exact.
template <class Space, class T>
void widen(ConstStorageSpanOf<Space, T> in, SpanOf<Space> out) {
  Kokkos::parallel_for("fuse::widen", Kokkos::RangePolicy<Space>(0, out.extent(0)),
                       WidenFunctor<Space, T>{in, out});
}

/// `out(i) = T(in(i))`, rounded to nearest with ties to even.
template <class Space, class T>
void narrow(ConstSpanOf<Space> in, StorageSpanOf<Space, T> out) {
  Kokkos::parallel_for("fuse::narrow", Kokkos::RangePolicy<Space>(0, out.extent(0)),
                       NarrowFunctor<Space, T>{in, out});
}

/// `narrow` plus the report; fences, because it returns a host value.
template <class Space, class T>
[[nodiscard]] RoundingReport narrow_measured(ConstSpanOf<Space> in,
                                             StorageSpanOf<Space, T> out) {
  RoundingReport report;
  Kokkos::parallel_reduce("fuse::narrow_m", Kokkos::RangePolicy<Space>(0, out.extent(0)),
                          NarrowMeasuredFunctor<Space, T>{in, out}, report);
  return report;
}

}  // namespace fuse::num
