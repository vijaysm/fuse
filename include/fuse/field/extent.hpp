#pragma once

/// How a slice of a field is spelled.
///
/// The spelling follows `std::mdspan`'s `submdspan` rather than inventing one,
/// because a reader who knows `mdspan` should not have to learn a second
/// convention for the same idea, and because "what does `2:end` mean here" is
/// a question no coupling library should make anyone ask:
///
///     f.slice(fuse::all, fuse::all)          // everything
///     f.slice({2, 5}, fuse::all)             // elements 2,3,4; all components
///     f.slice({2, fuse::end}, fuse::all)     // element 2 to the last
///     f.slice(7, fuse::all)                  // element 7 only
///
/// Half-open, like every other range in C++ and like `mdspan`: `{2, 5}` is
/// three elements. An `Extent` that is reversed or reaches past the end is a
/// REFUSAL carrying the offending bounds, not a clamp -- clamping turns a
/// caller's arithmetic mistake into a smaller-than-intended loop that still
/// produces numbers.

#include <cstddef>

namespace fuse {

/// `fuse::end` -- the position one past the last element, usable only as the
/// upper bound of an `Extent`. A tag rather than a sentinel value so that
/// `{2, fuse::end}` cannot be confused with `{2, SIZE_MAX}`, and so that
/// arithmetic on it does not compile.
struct EndTag {};
inline constexpr EndTag end{};

/// A half-open range of indices, `[lo, hi)`.
struct Extent {
  std::size_t lo = 0;
  std::size_t hi = 0;
  /// When set, `hi` is whatever the extent of the dimension turns out to be.
  /// Resolved against the view, so `{2, fuse::end}` stays correct after an
  /// earlier stage of the chain has already narrowed the dimension.
  bool hi_is_end = false;

  constexpr Extent() = default;
  constexpr Extent(std::size_t l, std::size_t h) noexcept : lo(l), hi(h) {}
  constexpr Extent(std::size_t l, EndTag) noexcept
      : lo(l), hi(0), hi_is_end(true) {}
  /// A bare index is the one-element range containing it, which is what
  /// `submdspan(m, 7, full_extent)` means.
  constexpr Extent(std::size_t i) noexcept : lo(i), hi(i + 1) {}  // NOLINT: implicit by design
  constexpr Extent(int i) noexcept
      : lo(static_cast<std::size_t>(i)),
        hi(static_cast<std::size_t>(i) + 1) {}  // NOLINT: implicit by design
};

/// `fuse::all` -- the whole dimension. Deliberately an `Extent` rather than a
/// distinct tag type: it means exactly `{0, end}`, and giving it its own type
/// would double every overload that accepts a dimension spec.
inline constexpr Extent all{0, end};

}  // namespace fuse
