#include "fuse/model/element_type.hpp"

#include <cfloat>

#include <gtest/gtest.h>

/// Stage B Slice 1b (docs/dev/restart.md Section 11). Compile-time trait
/// resolution -- `fuse::kElementTypeOf<T>` for every C++ type the typed
/// restart interface supports.
TEST(ElementType, TraitResolvesEveryCppTypeToItsOwnEnumerator) {
  static_assert(fuse::kElementTypeOf<std::int8_t> == fuse::ElementType::Int8);
  static_assert(fuse::kElementTypeOf<std::int16_t> == fuse::ElementType::Int16);
  static_assert(fuse::kElementTypeOf<std::int32_t> == fuse::ElementType::Int32);
  static_assert(fuse::kElementTypeOf<std::int64_t> == fuse::ElementType::Int64);
  static_assert(fuse::kElementTypeOf<std::uint8_t> == fuse::ElementType::UInt8);
  static_assert(fuse::kElementTypeOf<std::uint16_t> == fuse::ElementType::UInt16);
  static_assert(fuse::kElementTypeOf<std::uint32_t> == fuse::ElementType::UInt32);
  static_assert(fuse::kElementTypeOf<std::uint64_t> == fuse::ElementType::UInt64);
  static_assert(fuse::kElementTypeOf<fuse::bfloat16> == fuse::ElementType::BFloat16);
  static_assert(fuse::kElementTypeOf<float> == fuse::ElementType::Binary32);
  static_assert(fuse::kElementTypeOf<double> == fuse::ElementType::Binary64);
  static_assert(fuse::kElementTypeOf<long double> == fuse::ElementType::LongDouble);
  static_assert(fuse::kElementTypeOf<bool> == fuse::ElementType::Bool);
#if defined(__FLT16_MANT_DIG__)
  static_assert(fuse::kElementTypeOf<_Float16> == fuse::ElementType::Binary16);
#endif
  SUCCEED();
}

/// Measured facts this project's own design relies on (docs/dev/
/// restart.md Section 11's own "MEASURED ON THE OWNER'S COMPILER" note) --
/// pinned here so a future compiler/platform change that would silently
/// alter them is caught by a failing test, not rediscovered by hand.
TEST(ElementType, LongDoubleMetadataMatchesThisPlatformsMeasuredFacts) {
  const fuse::ElementTypeInfo& ld = fuse::element_type_info(fuse::ElementType::LongDouble);
  EXPECT_EQ(ld.byte_size, sizeof(long double));
  EXPECT_EQ(ld.significand_bits, static_cast<std::size_t>(LDBL_MANT_DIG));
#if defined(__APPLE__) && defined(__aarch64__)
  // Apple clang 21.0.0/arm64, measured directly (docs/dev/restart.md
  // Section 11): identical to double.
  EXPECT_EQ(sizeof(long double), 8u);
  EXPECT_EQ(LDBL_MANT_DIG, 53);
  EXPECT_FALSE(ld.explicit_integer_bit);
  const fuse::ElementTypeInfo& d = fuse::element_type_info(fuse::ElementType::Binary64);
  EXPECT_EQ(ld.significand_bits, d.significand_bits);
  EXPECT_EQ(ld.exponent_bits, d.exponent_bits);
  EXPECT_EQ(ld.exponent_bias, d.exponent_bias);
#endif
}

TEST(ElementType, UnknownNameReturnsNulloptRatherThanGuessing) {
  EXPECT_FALSE(fuse::element_type_from_name("quadruple precision").has_value());
  const auto binary64 = fuse::element_type_from_name("binary64");
  ASSERT_TRUE(binary64.has_value());
  EXPECT_EQ(*binary64, fuse::ElementType::Binary64);
}

/// `fuse::is_exact_widening()` -- every case the owner's own design
/// names explicitly, plus the "never signed->unsigned" and exponent-
/// range-containment rules it derives those named cases from.
TEST(ElementType, IsExactWideningMatchesEveryOwnerNamedCase) {
  using fuse::ElementType;
  // Float widenings.
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Binary32, ElementType::Binary64));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::BFloat16, ElementType::Binary32));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::BFloat16, ElementType::Binary64));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Binary16, ElementType::Binary32));
  // Integer widenings.
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Int32, ElementType::Int64));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::UInt32, ElementType::UInt64));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::UInt32, ElementType::Int64));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Bool, ElementType::Int64));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Bool, ElementType::UInt8));
  // int -> float only when the float's significand covers the integer's
  // width.
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Int8, ElementType::BFloat16));
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::Int16, ElementType::BFloat16));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Int16, ElementType::Binary32));
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::Int32, ElementType::Binary32));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Int32, ElementType::Binary64));
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::Int64, ElementType::Binary64));
  // Never signed -> unsigned, at any width.
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::Int8, ElementType::UInt64));
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::Int64, ElementType::UInt64));
  // uint64 is never an exact widen into int64 (exceeds its range).
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::UInt64, ElementType::Int64));
  // Exponent-range containment: bfloat16's exponent range is WIDER than
  // binary16's, so bfloat16 -> binary16 is not an exact widening even
  // though bfloat16's significand (8 bits) is narrower than binary16's
  // (11 bits) -- the exponent check, not the significand check, is what
  // excludes this pair.
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::BFloat16, ElementType::Binary16));
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::Binary16, ElementType::BFloat16));
  // float -> integer and bool -> float: not authorized by the owner's
  // design (docs/dev/restart.md Section 11's own open question).
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::Binary32, ElementType::Int32));
  EXPECT_FALSE(fuse::is_exact_widening(ElementType::Bool, ElementType::Binary32));
}

/// On THIS platform (long double bit-identical to double, see the
/// metadata test above) widening is possible in BOTH directions between
/// them -- a measured, platform-specific fact, not something every
/// platform this project could run on would share (an x86-64 Linux
/// build's 64-bit-significand long double is NOT an exact widening
/// target from binary64, since 53 > ... wait, 53 <= 64 so binary64 ->
/// long double WOULD still be exact there too; only long double ->
/// binary64 would differ by platform, since 64 > 53 fails there).
TEST(ElementType, LongDoubleBinary64WideningOnThisPlatform) {
  using fuse::ElementType;
#if defined(__APPLE__) && defined(__aarch64__)
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::LongDouble, ElementType::Binary64));
  EXPECT_TRUE(fuse::is_exact_widening(ElementType::Binary64, ElementType::LongDouble));
#endif
}
