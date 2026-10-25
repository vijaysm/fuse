#pragma once

/// Conversion between binary64 and a small IEEE-style binary format, by integer
/// operations on the bit pattern, callable from host and device code.
///
/// THE FORMAT. A format with `EB` exponent bits and `FB` explicit fraction bits
/// has one sign bit, an exponent field of bias 2^(EB-1) - 1, gradual underflow
/// (subnormals), infinities and NaNs, exactly as IEEE 754-2019 defines binary16
/// (EB = 5, FB = 10) and binary32 (EB = 8, FB = 23); bfloat16 (EB = 8, FB = 7)
/// has the layout of binary32 with a shorter fraction. The pattern is held in
/// the low `1 + EB + FB` bits of a std::uint32_t.
///
/// narrow() ROUNDS ONCE. It rounds the binary64 value directly to the format,
/// to nearest with ties to even, so the result is the correctly rounded value
/// whatever the operand. Going through binary32 first is not equivalent:
/// rounding to nearest twice can move a value across a midpoint of the target
/// format (a binary64 value just above a midpoint that first rounds down onto
/// it, and the tie then goes to the even neighbour). A value too large for the
/// format becomes an infinity of its sign, as IEEE 754 prescribes for rounding
/// to nearest; a value too small becomes a subnormal or a zero of its sign. A
/// NaN becomes the canonical quiet NaN of its sign (the payload is not kept:
/// the format may not have room for it).
///
/// widen() IS EXACT. Every value of a format with EB <= 8 and FB <= 52 is a
/// binary64 value. A signaling NaN is returned as a quiet NaN with the payload
/// shifted into place.
///
/// The functions never throw, never read memory, and never look at the
/// floating-point environment (no `std::isnan`, no libm).

#include <cstdint>

#include "fuse/base/portable.hpp"

namespace fuse::detail {

template <int EB, int FB>
struct SmallFloat {
  static_assert(EB >= 2 && EB <= 8, "the exponent range must lie inside binary64's");
  static_assert(FB >= 1 && FB <= 23, "the fraction must fit a 32-bit pattern with the exponent");
  static_assert(1 + EB + FB <= 32);

  static constexpr int exponent_bits = EB;
  static constexpr int fraction_bits = FB;
  static constexpr int bias = (1 << (EB - 1)) - 1;
  /// Unbiased exponent of the smallest normal number and of the largest finite one.
  static constexpr int min_exponent = 1 - bias;
  static constexpr int max_exponent = bias;
  /// Significand precision p, the implicit bit included; the unit roundoff of
  /// rounding to nearest is 2^-p.
  static constexpr int precision = FB + 1;

  static constexpr std::uint32_t exponent_all_ones = ((1u << EB) - 1u) << FB;
  static constexpr std::uint32_t quiet_bit = 1u << (FB - 1);

  /// 2^k as a binary64 value, for -1022 <= k <= 1023.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double pow2(int k) noexcept {
    return Kokkos::bit_cast<double>(static_cast<std::uint64_t>(k + 1023) << 52);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION static std::uint32_t narrow(double x) noexcept {
    const auto b = Kokkos::bit_cast<std::uint64_t>(x);
    const std::uint32_t sign = static_cast<std::uint32_t>(b >> 63) << (EB + FB);
    const int e = static_cast<int>((b >> 52) & 0x7FFu);
    const std::uint64_t m = b & 0xFFFFFFFFFFFFFull;
    if (e == 0x7FF) {
      return m == 0 ? (sign | exponent_all_ones)
                    : (sign | exponent_all_ones | quiet_bit);
    }
    // Zero, and the binary64 subnormals: below 2^-1022, which is below half the
    // smallest subnormal of every format this template accepts.
    if (e == 0) return sign;

    const int exponent = e - 1023;
    const std::uint64_t significand = (1ull << 52) | m;  // 53 bits, value = significand * 2^(exponent - 52)
    if (exponent > max_exponent) return sign | exponent_all_ones;

    // How many low bits of the significand do not fit: a normal number of the
    // format keeps FB + 1 bits; a subnormal one keeps fewer, one fewer for each
    // step the exponent is below min_exponent.
    const bool normal = exponent >= min_exponent;
    const int shift = (52 - FB) + (normal ? 0 : (min_exponent - exponent));
    if (shift >= 54) return sign;  // below half the smallest subnormal: rounds to zero

    std::uint64_t q = significand >> shift;
    const std::uint64_t rem = significand & ((1ull << shift) - 1ull);
    const std::uint64_t half = 1ull << (shift - 1);
    if (rem > half || (rem == half && (q & 1ull) != 0)) ++q;

    if (normal) {
      // q is in [2^FB, 2^(FB+1)]; a carry to 2^(FB+1) adds one to the exponent
      // field and leaves a zero fraction, and at max_exponent that is exactly
      // the infinity encoding.
      return sign + (static_cast<std::uint32_t>(exponent + bias) << FB) +
             static_cast<std::uint32_t>(q - (1ull << FB));
    }
    // A subnormal: q is the fraction field, and q == 2^FB is the encoding of
    // the smallest normal number.
    return sign | static_cast<std::uint32_t>(q);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double widen(std::uint32_t bits) noexcept {
    const std::uint64_t sign = static_cast<std::uint64_t>((bits >> (EB + FB)) & 1u) << 63;
    const std::uint32_t ef = (bits >> FB) & ((1u << EB) - 1u);
    const std::uint32_t f = bits & ((1u << FB) - 1u);
    if (ef == (1u << EB) - 1u) {
      const std::uint64_t payload = f == 0 ? 0ull
                                           : ((static_cast<std::uint64_t>(f) << (52 - FB)) | (1ull << 51));
      return Kokkos::bit_cast<double>(sign | (0x7FFull << 52) | payload);
    }
    if (ef == 0) {
      // Zero or a subnormal: f * 2^(min_exponent - FB), exact in binary64.
      const double v = static_cast<double>(f) * pow2(min_exponent - FB);
      return Kokkos::bit_cast<double>(sign | Kokkos::bit_cast<std::uint64_t>(v));
    }
    return Kokkos::bit_cast<double>(
        sign | (static_cast<std::uint64_t>(static_cast<int>(ef) - bias + 1023) << 52) |
        (static_cast<std::uint64_t>(f) << (52 - FB)));
  }
};

}  // namespace fuse::detail
