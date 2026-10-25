#pragma once

/// `fuse::binary16`: IEEE 754-2019 binary16 ("half precision"), 1 sign bit, 5
/// exponent bits, 10 fraction bits, as a one-`std::uint16_t` bit pattern.
///
/// WHY NOT `_Float16`. The compilers FUSE builds with do not agree on it: nvcc
/// 12.5 fails with an internal error on a translation unit that names
/// `_Float16` in device code, and its availability differs between host
/// compilers. A bit-pattern type with integer conversions is the same type on
/// every host and device compiler, so a binary16 field written by a kernel on a
/// GPU and read on the host holds the same bits. `_Float16` stays available
/// under `FUSE_NATIVE_FLOAT16` (fuse/model/element_type.hpp) for the restart
/// files; test/model/binary16_test.cpp uses it as an independent check of the
/// conversions where the compiler has it.
///
/// The type is the sibling of `fuse::bfloat16` (fuse/model/bfloat16.hpp) and has
/// the same interface: trivially copyable, standard layout, conversions to and
/// from binary32 and binary64 by integer operations on the bit pattern
/// (fuse/model/small_float.hpp), callable from host and device code.
///
/// `from_double()` and `from_float()` round to nearest, ties to even, once; a
/// value of magnitude 65520 or more (the midpoint above the largest finite value,
/// 65504) becomes an infinity, and a value below half the smallest subnormal (2^-25) becomes
/// zero. `to_double()` and `to_float()` are exact. A NaN becomes the quiet NaN
/// 0x7E00 of its sign.
///
/// `==` compares bit patterns, not numbers: +0 and -0 differ, a NaN equals
/// itself. Arithmetic is deliberately absent. A binary16 field is a Storage
/// type (fuse/model/value_traits.hpp): it is widened to binary64 at a
/// coupling boundary, computed on there, and rounded once on the way back.

#include <cstdint>
#include <type_traits>

#include "fuse/base/portable.hpp"
#include "fuse/model/small_float.hpp"

namespace fuse {

class binary16 {
 public:
  binary16() = default;

  using format = detail::SmallFloat<5, 10>;

  [[nodiscard]] KOKKOS_INLINE_FUNCTION static binary16 from_bits(
      std::uint16_t bits) noexcept {
    binary16 v;
    v.bits_ = bits;
    return v;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION std::uint16_t bits() const noexcept {
    return bits_;
  }

  /// Exact.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION double to_double() const noexcept {
    return format::widen(bits_);
  }

  /// Exact: every binary16 value is a binary32 value.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION float to_float() const noexcept {
    return static_cast<float>(to_double());
  }

  /// Round to nearest, ties to even, once.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static binary16 from_double(double x) noexcept {
    return from_bits(static_cast<std::uint16_t>(format::narrow(x)));
  }

  /// Round to nearest, ties to even, once: binary32 -> binary64 is exact, so
  /// this is the correctly rounded value of `f`.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static binary16 from_float(float f) noexcept {
    return from_double(static_cast<double>(f));
  }

  friend KOKKOS_INLINE_FUNCTION bool operator==(binary16 a, binary16 b) noexcept {
    return a.bits_ == b.bits_;
  }
  friend KOKKOS_INLINE_FUNCTION bool operator!=(binary16 a, binary16 b) noexcept {
    return !(a == b);
  }

 private:
  std::uint16_t bits_ = 0;
};

static_assert(sizeof(binary16) == 2);
static_assert(std::is_trivially_copyable_v<binary16>);
static_assert(std::is_standard_layout_v<binary16>);

}  // namespace fuse
