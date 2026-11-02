#pragma once

/// The conversion between the bit pattern of a posit<N, 2> (the 2022 Posit Standard) and
/// binary64, as two functions on integers: no class, no libm, nothing but the bit
/// operations of `fuse/base/portable.hpp`, so that the restart reader can use them
/// without taking Kokkos' execution spaces in. `fuse::num::posit` (fuse/num/posit.hpp)
/// is the type built on them, and its header describes the format and the rounding.
///
///   decode(bits)            the value of the pattern, exactly (a NaN for NaR)
///   encode(x, direction)    the pattern of the posit nearest to x, a tie going to the even
///                           pattern; `direction` is the sign of (exact value - x) when x is
///                           a rounded result, and decides a tie only (0: x is exact)

#include <cstdint>

#include "fuse/base/portable.hpp"

namespace fuse::num::detail {

template <int N, int ES>
struct PositCodec {
  static_assert((N == 16 || N == 32) && ES == 2);
  /// maxpos is 2^max_scale and minpos is 2^-max_scale.
  static constexpr int max_scale = (N - 2) * (1 << ES);
  static constexpr std::uint32_t nar_bits = 1u << (N - 1);
  static constexpr std::uint32_t mask = static_cast<std::uint32_t>((std::uint64_t{1} << N) - 1u);

  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double quiet_nan() noexcept {
    return Kokkos::bit_cast<double>(0x7FF8000000000000ull);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double decode(std::uint32_t bits) noexcept {
    bits &= mask;
    if (bits == 0) return 0.0;
    if (bits == nar_bits) return quiet_nan();
    const bool negative = (bits >> (N - 1)) != 0;
    // The pattern of the magnitude: N - 1 bits below a zero sign bit.
    const std::uint32_t magnitude = negative ? ((0u - bits) & mask) : bits;
    // The N - 1 bits after the sign, left-aligned in 32 bits, zero padded.
    const std::uint32_t body = magnitude << (33 - N);
    const bool ones = (body >> 31) != 0;
    const int run = ones ? static_cast<int>(Kokkos::countl_zero(static_cast<std::uint32_t>(~body)))
                         : static_cast<int>(Kokkos::countl_zero(body));
    const int k = ones ? run - 1 : -run;
    // The regime is the run and the bit that ends it, unless the run fills the pattern.
    const int regime_bits = run + 1 < N - 1 ? run + 1 : N - 1;
    const std::uint32_t rest = body << regime_bits;
    const std::uint32_t e = rest >> (32 - ES);
    const std::uint32_t fraction = rest << ES;  // left-aligned, zero padded
    const int scale = k * (1 << ES) + static_cast<int>(e);
    const std::uint64_t mantissa = static_cast<std::uint64_t>(fraction) << 20;
    return Kokkos::bit_cast<double>((negative ? (1ull << 63) : 0ull) |
                                    (static_cast<std::uint64_t>(scale + 1023) << 52) | mantissa);
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION static std::uint32_t encode(double x, int direction) noexcept {
    const std::uint64_t xb = Kokkos::bit_cast<std::uint64_t>(x);
    const bool negative = (xb >> 63) != 0;
    const std::uint64_t ab = xb & 0x7FFFFFFFFFFFFFFFull;
    if (ab == 0) return 0u;
    if (ab >= 0x7FF0000000000000ull) return nar_bits;
    const int biased = static_cast<int>(ab >> 52);
    const int scale = biased - 1023;
    std::uint32_t magnitude;
    if (biased == 0 || scale < -max_scale) {
      magnitude = 1;  // below minpos (a subnormal binary64 is far below it)
    } else if (scale >= max_scale) {
      magnitude = nar_bits - 1u;  // maxpos
    } else {
      const int k = scale >> ES;  // floor division
      const std::uint64_t e = static_cast<std::uint64_t>(scale & ((1 << ES) - 1));
      const std::uint64_t payload = (e << 52) | (ab & 0x000FFFFFFFFFFFFFull);  // ES + 52 bits
      int regime_len;
      std::uint32_t regime;
      if (k >= 0) {
        regime_len = k + 2;  // k + 1 ones and a zero
        regime = ((1u << (k + 1)) - 1u) << 1;
      } else {
        regime_len = -k + 1;  // -k zeros and a one
        regime = 1u;
      }
      const int kept = (N - 1) - regime_len;  // payload bits that fit, 0 or more
      const int dropped = (ES + 52) - kept;
      std::uint32_t pattern = (regime << kept) | static_cast<std::uint32_t>(payload >> dropped);
      const std::uint64_t tail = payload & ((1ull << dropped) - 1ull);
      const std::uint64_t half = 1ull << (dropped - 1);
      const bool guard = (tail & half) != 0;
      const bool sticky = (tail & (half - 1ull)) != 0;
      const int toward = negative ? -direction : direction;  // of the magnitude
      bool up = false;
      if (guard) up = sticky || toward > 0 || (toward == 0 && (pattern & 1u) != 0);
      if (up) ++pattern;
      magnitude = pattern;
    }
    return negative ? ((0u - magnitude) & mask) : magnitude;
  }
};

}  // namespace fuse::num::detail
