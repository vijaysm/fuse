#pragma once

/// Restart track (docs/dev/restart.md), Stage B Slice 1b: `fuse::bfloat16`,
/// a FUSE storage type for Google's bfloat16 ("brain floating point")
/// format -- 1 sign bit, 8 exponent bits (the SAME width and bias as IEEE
/// 754-2019 binary32), 7 explicit significand bits (8 total with the
/// implicit leading one). No C++ type on this project's measured compiler
/// backs it: Apple clang 21.0.0/arm64 has neither `__bf16` nor
/// `std::bfloat16_t` (`<stdfloat>` itself is absent) -- `fuse::bfloat16` is
/// therefore a ONE-`uint16_t` bit-pattern wrapper, trivially copyable and
/// standard layout (so `std::span<const std::byte>` reinterpretation,
/// `fuse/model/restart.hpp`'s own typed `put_field<T>()`/`get_field<T>()`
/// front end, is well-defined), with explicit `to_float()`/`from_float()`
/// conversions this file owns and `test/model/bfloat16_test.cpp` proves
/// exhaustively over every one of the 65536 possible bit patterns.
///
/// ROUNDING. `from_float()` rounds to nearest, ties to even (the IEEE
/// 754-2019 default rounding attribute) -- never truncation (which would
/// silently bias every rounded value downward in magnitude). A NaN input
/// maps to a quiet bfloat16 NaN (the sign and payload are not preserved
/// bit-for-bit; only "is this a NaN" is preserved, since bfloat16's 7
/// significand bits cannot carry float's 23-bit NaN payload in general).
/// `to_float()` (bfloat16 -> float) is always EXACT: every bfloat16 value
/// is, by construction, already representable in float's wider format
/// (8 exponent bits identically, 7-of-23 significand bits, the rest
/// implicitly zero) -- this is `fuse::is_exact_widening(BFloat16,
/// Binary32)`'s own underlying fact, restated here as what makes it true.
///
/// FROM binary64. `from_double()` rounds a binary64 value to bfloat16 ONCE,
/// to nearest with ties to even (fuse/model/small_float.hpp), including the
/// overflow to infinity and the gradual underflow that bfloat16's range
/// implies. Rounding to binary32 first and then to bfloat16 is not the same:
/// a binary64 value just above a bfloat16 midpoint can round DOWN onto the
/// midpoint in binary32, and the tie then goes to the even neighbour.
/// `to_double()` is exact.
///
/// DEVICE. Every member is `KOKKOS_INLINE_FUNCTION`, uses only integer
/// operations and `Kokkos::bit_cast`, and never throws, so the type can be
/// converted inside a kernel on any execution space (test/num/storage_test.cpp
/// compares the device results with the host's over every bit pattern).

#include <cstdint>
#include <type_traits>

#include "fuse/base/portable.hpp"
#include "fuse/model/small_float.hpp"

namespace fuse {

class bfloat16 {
 public:
  bfloat16() = default;

  using format = detail::SmallFloat<8, 7>;

  /// Constructs from the raw 16-bit pattern directly -- used by tests that
  /// need to drive a SPECIFIC bit pattern (including every one of the
  /// 65536 possible patterns, exhaustively) without going through
  /// `from_float()`'s own rounding.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static bfloat16 from_bits(
      std::uint16_t bits) noexcept {
    bfloat16 v;
    v.bits_ = bits;
    return v;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION std::uint16_t bits() const noexcept {
    return bits_;
  }

  /// Exact (see file header comment) -- a plain widen of the 16-bit
  /// pattern into float's 32-bit layout, never a value-level rounding.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION float to_float() const noexcept {
    return Kokkos::bit_cast<float>(static_cast<std::uint32_t>(bits_) << 16);
  }

  /// Exact: every bfloat16 value is a binary64 value.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION double to_double() const noexcept {
    return format::widen(bits_);
  }

  /// Round to nearest, ties to even, once (see the file header comment).
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static bfloat16 from_double(double x) noexcept {
    return from_bits(static_cast<std::uint16_t>(format::narrow(x)));
  }

  /// Round to nearest, ties to even; quiet-NaN-preserving (see file header
  /// comment).
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static bfloat16 from_float(float f) noexcept {
    const auto bits = Kokkos::bit_cast<std::uint32_t>(f);

    // NaN: force a canonical quiet bfloat16 NaN. (Exponent all-ones,
    // mantissa nonzero; the sign bit of the ORIGINAL NaN is preserved,
    // matching how every mainstream bfloat16 cast -- XLA, PyTorch's
    // native bfloat16 -- treats sign on NaN, even though a NaN's sign has
    // no numerical meaning.)
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {  // NaN: exponent all ones, fraction not zero
      const std::uint16_t sign = static_cast<std::uint16_t>(bits >> 16) &
                                 0x8000u;
      return from_bits(static_cast<std::uint16_t>(sign | 0x7FC0u));
    }

    // Round-to-nearest-even over the low 16 bits being discarded.
    // `bits`'s low 16 bits are float's low 16 significand bits; bit 15 of
    // THOSE (bit 15 overall) is the round bit, bits 0-14 are the sticky
    // remainder.
    const std::uint32_t round_bit = (bits >> 15) & 1u;
    const std::uint32_t sticky = bits & 0x7FFFu;
    const std::uint32_t lsb_of_kept = (bits >> 16) & 1u;
    std::uint32_t rounded = bits >> 16;  // truncate to the top 16 bits
    if (round_bit != 0 && (sticky != 0 || lsb_of_kept != 0)) {
      // Round up; this can carry into (and correctly through) the
      // exponent field, including turning a finite value into infinity
      // at the top of the range -- `rounded` is a full 32-bit quantity
      // here specifically so that carry has somewhere to go before the
      // final truncation to 16 bits below.
      rounded += 1;
    }
    return from_bits(static_cast<std::uint16_t>(rounded));
  }

  /// Equality of the bit patterns, not of the numbers: +0 and -0 differ and a
  /// NaN equals itself.
  friend KOKKOS_INLINE_FUNCTION bool operator==(bfloat16 a, bfloat16 b) noexcept {
    return a.bits_ == b.bits_;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator!=(bfloat16 a, bfloat16 b) noexcept {
    return !(a == b);
  }

 private:
  std::uint16_t bits_ = 0;
};

static_assert(sizeof(bfloat16) == 2, "fuse::bfloat16 must be exactly 2 bytes "
                                     "-- it is written to disk as one "
                                     "uint16_t bit pattern verbatim");
static_assert(std::is_trivially_copyable_v<bfloat16>);
static_assert(std::is_standard_layout_v<bfloat16>);

}  // namespace fuse
