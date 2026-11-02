#pragma once

/// `fuse::num::posit<N, ES>`: the posit format of the 2022 Posit Standard
/// (Gustafson and Yonemoto 2017 for the format, the Posit Working Group's Standard
/// for Posit Arithmetic of March 2022 for the rounding), as the bit pattern in an
/// unsigned integer: `posit<16, 2>` (`posit16`) in 16 bits and `posit<32, 2>`
/// (`posit32`) in 32. A posit is a Storage type (fuse/model/value_traits.hpp): it
/// is widened exactly to binary64 at a coupling boundary, computed on there, and
/// rounded once on the way back, so a field in 16 bits keeps its precision where
/// the values are near one and loses it, gradually, far from one.
///
/// THE FORMAT. A pattern is one sign bit and N - 1 more bits. A negative value is
/// stored as the two's complement of the pattern of its magnitude, so the patterns
/// order like signed integers, `0` is zero, and the one pattern with only the sign
/// bit set (`1 << (N - 1)`) is NaR ("not a real"), the one exception value, which
/// stands for every NaN and infinity. After the sign come the regime (a run of
/// equal bits ended by the opposite bit, if there is room for it), then up to ES
/// exponent bits, then the fraction. A run of m ones is the regime k = m - 1, a run
/// of m zeros is k = -m, and the value is
///
///     (-1)^s * 2^(k * 2^ES + e) * (1 + f / 2^(number of fraction bits))
///
/// with the exponent e and the fraction f read from the bits that are left (a
/// missing exponent or fraction bit is zero). For ES = 2 that is 16^k * 2^e * (1 + f).
/// The largest value `maxpos` is 2^((N - 2) 2^ES) and the smallest positive value
/// `minpos` its reciprocal: 2^56 and 2^-56 for posit16, 2^120 and 2^-120 for
/// posit32. The value one has the most fraction bits, N - 3 - ES, and the count
/// falls by one for every step of 2^(2^ES) away from it.
///
/// ROUNDING (`from_double`). The real number is rounded to the nearest posit, a tie
/// going to the pattern that ends in a zero, and the boundary between two adjacent
/// posits is the value of the (N + 1)-bit pattern that is the lower one with a one
/// appended. Where the rounding falls on a fraction bit that is the arithmetic
/// midpoint, as for IEEE formats; where it falls on an exponent or a regime bit it
/// is the geometric one, which is what makes the rounding of the pattern as an
/// integer monotone. A nonzero value never becomes zero and a finite value never
/// becomes NaR: below `minpos` the result is `minpos` and above `maxpos` it is
/// `maxpos` (saturation, no overflow to infinity). Zero is exact, and NaN and the
/// infinities are NaR.
///
/// ARITHMETIC. `+`, `-`, `*`, `/` and `sqrt` are correctly rounded: the exact
/// result of the operation on the two exact binary64 values of the operands is
/// found as a binary64 value and the sign of its error (a two-sum, a two-product
/// or a fused multiply-add residual, fuse/num/eft.hpp), and the rounding above
/// takes the sign of the error as a sticky bit, so the result is rounded once. An
/// operation with a NaR operand, a division by zero and the square root of a
/// negative value are NaR. Comparisons are those of the patterns as signed
/// integers: NaR is equal to itself and less than every real value (the
/// standard's order).
///
/// Every function is callable from host and device code, throws nothing, and uses
/// integer operations on the pattern and on the binary64 bits; no libm function
/// other than `sqrt` and `fma`, which are correctly rounded on every execution
/// space.

#include <cstdint>
#include <string_view>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "fuse/model/element_type.hpp"
#include "fuse/model/value_traits.hpp"
#include "fuse/num/eft.hpp"
#include "fuse/num/posit_bits.hpp"

namespace fuse::num {

namespace detail {

template <int N>
struct PositBits;
template <>
struct PositBits<16> {
  using type = std::uint16_t;
  using signed_type = std::int16_t;
};
template <>
struct PositBits<32> {
  using type = std::uint32_t;
  using signed_type = std::int32_t;
};

}  // namespace detail

template <int N, int ES>
class posit {
  static_assert((N == 16 || N == 32) && ES == 2,
                "fuse::num::posit supports the formats of the 2022 Posit Standard that FUSE "
                "stores: posit<16, 2> and posit<32, 2>");

 public:
  using Codec = detail::PositCodec<N, ES>;
  using bits_type = typename detail::PositBits<N>::type;
  using signed_bits_type = typename detail::PositBits<N>::signed_type;

  /// A conversion to this format never overflows to an infinity or flushes to zero: it clamps
  /// to the largest or smallest value (the rounding report of fuse/num/storage_convert.hpp
  /// counts those as overflow and underflow).
  static constexpr bool saturates = true;
  static constexpr int nbits = N;
  static constexpr int exponent_bits = ES;
  /// maxpos is 2^max_scale and minpos is 2^-max_scale.
  static constexpr int max_scale = Codec::max_scale;
  /// The most fraction bits a posit has (the value one has them): N - 3 - ES.
  static constexpr int max_fraction_bits = N - 3 - ES;

  posit() = default;

  [[nodiscard]] KOKKOS_INLINE_FUNCTION static posit from_bits(bits_type bits) noexcept {
    posit p;
    p.bits_ = bits;
    return p;
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bits_type bits() const noexcept { return bits_; }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION static posit zero() noexcept { return from_bits(0); }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static posit nar() noexcept { return from_bits(nar_bits()); }
  /// The smallest positive value, 2^-max_scale.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static posit minpos() noexcept { return from_bits(1); }
  /// The largest value, 2^max_scale.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static posit maxpos() noexcept {
    return from_bits(static_cast<bits_type>(nar_bits() - 1u));
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool is_nar() const noexcept { return bits_ == nar_bits(); }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool is_zero() const noexcept { return bits_ == 0; }
  /// Negative and NaR (NaR is the most negative pattern).
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool is_negative() const noexcept {
    return static_cast<signed_bits_type>(bits_) < 0;
  }

  /// Exact: every posit is a binary64 value (at most 2^-120 to 2^120 with 28 bits). NaR
  /// is a NaN.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION double to_double() const noexcept {
    return Codec::decode(bits_);
  }

  /// Round to nearest, ties to the even pattern, once; saturating; NaR for NaN and the
  /// infinities.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static posit from_double(double x) noexcept {
    return round_directed(x, 0);
  }

  /// Rounds `x` as if the real number being rounded were slightly above (`direction` > 0)
  /// or below (`direction` < 0) `x`, or exactly `x` (0). It is the rounding of an
  /// operation whose exact result is `x` plus an error too small to move it past a
  /// boundary other than the one `x` sits on: only a tie is decided by it.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static posit round_directed(double x,
                                                                   int direction) noexcept {
    return from_bits(static_cast<bits_type>(Codec::encode(x, direction)));
  }

  /// The twos complement of the pattern, computed in the top bits of a 32-bit word. The plain
  /// 16-bit form (`0u - bits_`, truncated) is miscompiled by nvcc 12.5 for the pattern of NaR:
  /// when the result is both compared with 0x8000 and stored, the comparison is false although
  /// the stored bits are 0x8000 (at -O0 and -O3). test/num/posit_test.cpp compares every
  /// operation with the host on the device legs, which is what found it.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION posit operator-() const noexcept {
    return from_bits(static_cast<bits_type>((0u - (static_cast<std::uint32_t>(bits_) << (32 - N))) >> (32 - N)));
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION posit operator+() const noexcept { return *this; }

  friend KOKKOS_INLINE_FUNCTION posit abs(posit a) noexcept {
    return a.is_negative() ? -a : a;
  }

  friend KOKKOS_INLINE_FUNCTION posit operator+(posit a, posit b) noexcept {
    if (a.is_nar() || b.is_nar()) return nar();
    // The binary64 sum is exact, or its operands differ in scale by so many binades
    // that the exact sum cannot lie on the far side of a boundary between two posits from
    // the binary64 sum (a posit has at most N - 5 fraction bits, fewer away from one), so
    // rounding the binary64 sum once is the correct rounding. The products and quotients
    // below have no such property and carry the sign of their error.
    // test/num/posit_test.cpp checks the sums of operands at every distance in scale and
    // those that land on or next to a boundary.
    return from_double(a.to_double() + b.to_double());
  }
  friend KOKKOS_INLINE_FUNCTION posit operator-(posit a, posit b) noexcept {
    return a + (-b);
  }
  friend KOKKOS_INLINE_FUNCTION posit operator*(posit a, posit b) noexcept {
    if (a.is_nar() || b.is_nar()) return nar();
    double err;
    const double product = eft::two_prod(a.to_double(), b.to_double(), err);
    return round_directed(product, sign_of(err));
  }
  friend KOKKOS_INLINE_FUNCTION posit operator/(posit a, posit b) noexcept {
    if (a.is_nar() || b.is_nar() || b.is_zero()) return nar();
    if (a.is_zero()) return zero();
    const double x = a.to_double();
    const double y = b.to_double();
    const double q = x / y;
    // x = q * y + r exactly, so the quotient is q + r / y.
    const double r = Kokkos::fma(-q, y, x);
    return round_directed(q, sign_of(r) * (y < 0.0 ? -1 : 1));
  }
  friend KOKKOS_INLINE_FUNCTION posit sqrt(posit a) noexcept {
    if (a.is_nar() || a.is_negative()) return nar();
    if (a.is_zero()) return zero();
    const double x = a.to_double();
    const double s = Kokkos::sqrt(x);
    // x = s * s + r exactly, so the root is s + r / (2 s).
    const double r = Kokkos::fma(-s, s, x);
    return round_directed(s, sign_of(r));
  }

  posit& operator+=(posit b) noexcept { return *this = *this + b; }
  posit& operator-=(posit b) noexcept { return *this = *this - b; }
  posit& operator*=(posit b) noexcept { return *this = *this * b; }
  posit& operator/=(posit b) noexcept { return *this = *this / b; }

  /// Bit equality (NaR equals itself), and the order of the patterns as signed integers.
  friend KOKKOS_INLINE_FUNCTION bool operator==(posit a, posit b) noexcept {
    return a.bits_ == b.bits_;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator!=(posit a, posit b) noexcept {
    return a.bits_ != b.bits_;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator<(posit a, posit b) noexcept {
    return static_cast<signed_bits_type>(a.bits_) < static_cast<signed_bits_type>(b.bits_);
  }
  friend KOKKOS_INLINE_FUNCTION bool operator>(posit a, posit b) noexcept {
    return b < a;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator<=(posit a, posit b) noexcept {
    return !(b < a);
  }
  friend KOKKOS_INLINE_FUNCTION bool operator>=(posit a, posit b) noexcept {
    return !(a < b);
  }

 private:
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr std::uint32_t nar_bits() noexcept {
    return 1u << (N - 1);
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static int sign_of(double v) noexcept {
    return v > 0.0 ? 1 : (v < 0.0 ? -1 : 0);
  }

  bits_type bits_ = 0;
};

using posit16 = posit<16, 2>;
using posit32 = posit<32, 2>;

static_assert(sizeof(posit16) == 2 && sizeof(posit32) == 4);
static_assert(std::is_trivially_copyable_v<posit16> && std::is_trivially_copyable_v<posit32>);
static_assert(std::is_standard_layout_v<posit16> && std::is_standard_layout_v<posit32>);

}  // namespace fuse::num

namespace fuse {

/// A posit is stored in a restart file as the unsigned integer of its pattern, tagged
/// `posit16` or `posit32` (fuse/model/element_type.hpp).
template <>
struct ElementTypeOf<num::posit16> {
  static constexpr ElementType value = ElementType::Posit16;
};
template <>
struct ElementTypeOf<num::posit32> {
  static constexpr ElementType value = ElementType::Posit32;
};

template <int N, int ES>
struct value_traits<num::posit<N, ES>> {
  static constexpr std::string_view name = N == 16 ? "posit16" : "posit32";
  static constexpr FieldValueKind kind = FieldValueKind::Storage;
  using scalar_type = num::posit<N, ES>;
  static constexpr int scalar_count = 1;
  static constexpr int gradient_size = 0;
  /// Exact: every posit is a binary64 value; NaR is a NaN.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double to_binary64(num::posit<N, ES> v) noexcept {
    return v.to_double();
  }
  /// Correctly rounded, to nearest with ties to the even pattern, once (posit.hpp), saturating.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static num::posit<N, ES> from_binary64(double v) noexcept {
    return num::posit<N, ES>::from_double(v);
  }
};

}  // namespace fuse
