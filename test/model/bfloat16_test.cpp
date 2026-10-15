#include "fuse/model/bfloat16.hpp"

#include <cmath>
#include <cstring>

#include <gtest/gtest.h>

/// Stage B Slice 1b (docs/dev/restart.md Section 11). Exhaustive --
/// `fuse::bfloat16::from_bits()` enumerates all 65536 possible bit
/// patterns, and every non-NaN one must round-trip through `to_float()`/
/// `from_float()` exactly (the owner's own instruction: "Test the
/// conversion exhaustively over all 65536 bit patterns (bf16->float->bf16
/// identity for every non-NaN pattern, NaN stays NaN)"). A NaN pattern
/// does not round-trip BIT-FOR-BIT (its sign IS preserved, its payload is
/// not, by `fuse::bfloat16::from_float()`'s own design -- see that
/// file's header comment) -- this test checks "stays NaN", exactly the
/// owner's own weaker requirement for that one case.
TEST(BFloat16, ExhaustiveBitToFloatToBitIdentityOverEveryNonNanPattern) {
  int nan_count = 0;
  for (unsigned v = 0; v <= 0xFFFFu; ++v) {
    const fuse::bfloat16 b = fuse::bfloat16::from_bits(static_cast<std::uint16_t>(v));
    const float f = b.to_float();
    if (std::isnan(f)) {
      ++nan_count;
      const fuse::bfloat16 b2 = fuse::bfloat16::from_float(f);
      EXPECT_TRUE(std::isnan(b2.to_float()))
          << "NaN pattern 0x" << std::hex << v << " did not stay NaN";
      continue;
    }
    const fuse::bfloat16 b2 = fuse::bfloat16::from_float(f);
    EXPECT_EQ(b2.bits(), b.bits())
        << "bits=0x" << std::hex << v << " -> float " << f << " -> bits=0x"
        << std::hex << b2.bits();
  }
  // bfloat16's exponent field is 8 bits (256 values); exactly 2 of those
  // (all-ones) are reserved for Inf/NaN, and of the (sign x mantissa)
  // combinations under that exponent, every nonzero-mantissa one is a
  // NaN: 2 signs * (2^7 - 1) nonzero 7-bit mantissas = 254.
  EXPECT_EQ(nan_count, 254);
}

/// Round to nearest, ties to even -- the owner's own named rounding rule.
/// Constructed directly from float bit patterns (not `bfloat16::
/// from_bits()`, which does no rounding at all) so each case drives
/// `from_float()`'s real rounding path on a value exactly halfway
/// between two adjacent bfloat16 values.
TEST(BFloat16, RoundToNearestEvenAtExactTies) {
  // 1.0's bfloat16 mantissa is 0000000 (even) -- a tie should round DOWN
  // (stay at the even value), per ties-to-even.
  {
    std::uint32_t bits = 0x3F800000u | 0x00008000u;  // 1.0 + half a bf16 ulp
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    EXPECT_EQ(fuse::bfloat16::from_float(f).bits(), 0x3F80u);
  }
  // mantissa 0000001 (odd) + half ulp -- ties-to-even rounds UP to 0000010.
  {
    std::uint32_t bits = 0x3F800000u | (1u << 16) | 0x00008000u;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    EXPECT_EQ(fuse::bfloat16::from_float(f).bits(), 0x3F82u);
  }
}

/// Rounding that carries out of the significand's top bit must carry into
/// the exponent, including turning the largest finite bfloat16 into
/// infinity -- the standard IEEE "rounding overflow" case, and the
/// mutation row (bf16 rounding that truncates instead of rounding) this
/// test is specifically built to kill (docs/dev/restart.md Section 11).
TEST(BFloat16, RoundingOverflowAtTheTopOfTheRangeBecomesInfinity) {
  const std::uint32_t bits = 0x7F7FFFFFu;  // near FLT_MAX, rounds up
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  const fuse::bfloat16 b = fuse::bfloat16::from_float(f);
  EXPECT_EQ(b.bits(), 0x7F80u);
  EXPECT_TRUE(std::isinf(b.to_float()));
  EXPECT_GT(b.to_float(), 0.0f);
}

TEST(BFloat16, ToFloatIsAlwaysExactWidening) {
  // Every bfloat16 value, by construction (same exponent width/bias as
  // float, significand a strict subset), widens to float with no value
  // change at all -- this is what `fuse::is_exact_widening(BFloat16,
  // Binary32)` asserts; here it is checked directly against a handful of
  // representative values rather than re-deriving the general claim.
  for (std::uint16_t bits : {0x3F80u, 0x4000u, 0xBFC0u, 0x0001u, 0x7F7Fu}) {
    const fuse::bfloat16 b = fuse::bfloat16::from_bits(bits);
    const float f = b.to_float();
    std::uint32_t fbits;
    std::memcpy(&fbits, &f, sizeof(fbits));
    EXPECT_EQ(fbits, static_cast<std::uint32_t>(bits) << 16)
        << "bits=0x" << std::hex << bits;
  }
}
