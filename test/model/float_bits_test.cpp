#include "fuse/model/float_bits.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

using fuse::ElementType;
using fuse::ElementTypeInfo;
using fuse::element_type_info;
namespace fd = fuse::detail;

namespace {

/// Builds the x86-64 extended-precision `long double` layout BY HAND,
/// from raw bytes -- the owner's own instruction (docs/dev/restart.md
/// Section 11, item 6): "construct an x86-64-style 80-bit long double
/// dataset on this Mac by building the file type by hand ... from raw
/// bytes". 64-bit EXPLICIT significand (no implicit leading one -- x86's
/// own convention, `H5T_NORM_MSBSET` in HDF5's own vocabulary,
/// `H5Tpublic.h`), 15-bit exponent, bias 16383, 16-byte storage (10
/// meaningful bytes + 6 padding, matching this test's own raw buffers
/// below). NEVER the real `fuse::ElementType::LongDouble` on this build
/// (whose OWN `element_type_info()` reflects THIS Mac's platform, not
/// x86's) -- a hand-built stand-in, exactly as the owner's instruction
/// asks for.
ElementTypeInfo x86_long_double_info() {
  ElementTypeInfo info;
  info.element_type = ElementType::LongDouble;
  info.name = "long double (x86-64 extended, hand-built test fixture)";
  info.byte_size = 16;
  info.is_float = true;
  info.significand_bits = 64;
  info.exponent_bits = 15;
  info.exponent_bias = 16383;
  info.explicit_integer_bit = true;
  return info;
}

/// Packs a 16-byte x86-80-bit-extended raw buffer: `mantissa` (64 bits,
/// explicit integer bit included) at bytes [0,8), `biased_exponent`
/// (15 bits) and `sign` (1 bit) packed into bytes [8,10), the remaining
/// 6 bytes zero padding.
std::array<unsigned char, 16> pack_x86_ld(std::uint64_t mantissa,
                                          std::uint16_t biased_exponent,
                                          bool sign) {
  std::array<unsigned char, 16> buf{};
  std::memcpy(buf.data(), &mantissa, 8);
  std::uint16_t tail = biased_exponent | (sign ? 0x8000u : 0u);
  std::memcpy(buf.data() + 8, &tail, 2);
  return buf;
}

}  // namespace

/// THE owner's own required scenario (docs/dev/restart.md Section 11,
/// item 6): a hand-built x86-64-style 80-bit `long double`, value
/// `1 + 2^-63` (mantissa `0x8000000000000001`, biased exponent 16383 =
/// unbiased 0), narrowed to THIS Mac's own `double`/`long double` (both
/// 53-bit significand, identical layout) -- AllowNarrowing reads it with
/// a reported maximum change equal to an INDEPENDENTLY computed value
/// (`ldexp(1.0, -63)`, derived from first principles: the dropped low 11
/// bits of the 64-bit significand, at unbiased exponent 0, are worth
/// exactly `2^-63` in the source's own units -- computed here with
/// `std::ldexp`, a different code path from `fuse::detail::narrow_float`'s
/// own internal computation, so this is a genuine independent check, not
/// the same arithmetic restated).
TEST(FloatBits, OwnerScenario_X86ExtendedLongDoubleNarrowsToMacDoubleWithReportedDelta) {
  const ElementTypeInfo x86ld = x86_long_double_info();
  const ElementTypeInfo& mac_double = element_type_info(ElementType::Binary64);

  const auto buf = pack_x86_ld(0x8000000000000001ULL, 16383, /*sign=*/false);
  std::span<const std::byte> raw(reinterpret_cast<const std::byte*>(buf.data()),
                                 buf.size());

  // Exact: the stored type is NOT `fuse::ElementType::Binary64` (it is a
  // hand-built stand-in for `LongDouble`, and even the real LongDouble on
  // THIS Mac is a DIFFERENT enumerator from Binary64 by design) -- a
  // caller comparing enumerators (what `fuse::restart_io`'s own
  // `ReadConversion::Exact` path does before ever reaching this file)
  // would refuse. This test exercises the BIT-LEVEL conversion pipeline
  // directly; the end-to-end refusal-wiring equivalent is
  // `test/runtime/restart_test.cpp`'s own
  // `HandBuiltX86LongDoubleExactAndWideningRefuseAllowNarrowingReportsDelta`.
  EXPECT_NE(x86ld.significand_bits, mac_double.significand_bits);

  // Widening: this hand-built fixture's 64-bit significand does not fit
  // Mac double's 53 bits -- not an exact widening. (NOT checked via
  // fuse::is_exact_widening(ElementType::LongDouble, Binary64): that
  // function looks up the GLOBALLY REGISTERED LongDouble metadata --
  // THIS Mac's own 53-bit long double, which genuinely IS an exact
  // widening into Binary64, see ElementTypeTest's own
  // LongDoubleBinary64WideningOnThisPlatform -- never this test's local
  // x86 stand-in. The widening question for THIS fixture is answered
  // directly from its own significand width instead.)
  EXPECT_GT(x86ld.significand_bits, mac_double.significand_bits);

  // AllowNarrowing: decode, round, and report.
  const fd::NarrowingResult nr = fd::narrow_float(raw, x86ld, mac_double);
  ASSERT_TRUE(nr.changed);
  const double independently_computed_delta = std::ldexp(1.0, -63);
  EXPECT_EQ(nr.absolute_change, independently_computed_delta);

  std::array<unsigned char, 8> outbuf{};
  fd::encode_float_bits(
      nr.decoded, mac_double,
      std::span<std::byte>(reinterpret_cast<std::byte*>(outbuf.data()), 8));
  double out_val;
  std::memcpy(&out_val, outbuf.data(), 8);
  EXPECT_EQ(out_val, 1.0);  // rounds DOWN (ties-to-even: kept bit was 0)
}

/// The all-ones-mantissa companion case: `2 - 2^-63`, which rounds UP
/// through a carry into the exponent -- `round_significand()`'s own
/// "renormalise after a rounding carry" branch, exercised end to end
/// through the SAME hand-built x86 layout.
TEST(FloatBits, X86ExtendedAllOnesMantissaRoundsUpThroughExponentCarry) {
  const ElementTypeInfo x86ld = x86_long_double_info();
  const ElementTypeInfo& mac_double = element_type_info(ElementType::Binary64);
  const auto buf = pack_x86_ld(0xFFFFFFFFFFFFFFFFULL, 16383, /*sign=*/false);
  std::span<const std::byte> raw(reinterpret_cast<const std::byte*>(buf.data()),
                                 buf.size());
  const fd::NarrowingResult nr = fd::narrow_float(raw, x86ld, mac_double);
  ASSERT_TRUE(nr.changed);
  std::array<unsigned char, 8> outbuf{};
  fd::encode_float_bits(
      nr.decoded, mac_double,
      std::span<std::byte>(reinterpret_cast<std::byte*>(outbuf.data()), 8));
  double out_val;
  std::memcpy(&out_val, outbuf.data(), 8);
  EXPECT_EQ(out_val, 2.0);
}

/// A value that rounds EXACTLY (no bits lost) reports `changed == false`
/// and `absolute_change == 0.0` -- the mutation row this test guards
/// against is a stub that reports a nonzero or `converted_any == true`
/// unconditionally.
TEST(FloatBits, ExactNarrowingReportsNoChange) {
  const ElementTypeInfo x86ld = x86_long_double_info();
  const ElementTypeInfo& mac_double = element_type_info(ElementType::Binary64);
  const auto buf = pack_x86_ld(0x8000000000000000ULL, 16383, /*sign=*/false);
  std::span<const std::byte> raw(reinterpret_cast<const std::byte*>(buf.data()),
                                 buf.size());
  const fd::NarrowingResult nr = fd::narrow_float(raw, x86ld, mac_double);
  EXPECT_FALSE(nr.changed);
  EXPECT_EQ(nr.absolute_change, 0.0);
}

/// A pure bit decode immediately followed by an encode at the SAME
/// width is the identity, for every real `ElementType` this project's
/// typed restart interface supports -- the general round-trip property
/// `decode_float_bits()`/`encode_float_bits()`'s own doc comment states.
TEST(FloatBits, DecodeThenEncodeAtTheSameWidthIsIdentity) {
  for (const ElementType t : {ElementType::Binary16, ElementType::BFloat16,
                              ElementType::Binary32, ElementType::Binary64,
                              ElementType::LongDouble}) {
    const ElementTypeInfo& info = element_type_info(t);
    std::vector<std::byte> buf(info.byte_size, std::byte{0});
    // A representative nonzero, non-special value: sign=0, a mid-range
    // biased exponent, significand with a mix of set/clear bits.
    fd::DecodedFloat v;
    v.sign = false;
    v.significand = (std::uint64_t{1} << (info.significand_bits - 1)) | 0x5u;
    v.unbiased_exponent = 2;
    fd::encode_float_bits(v, info, buf);
    const fd::DecodedFloat back = fd::decode_float_bits(buf, info);
    EXPECT_EQ(back.sign, v.sign) << info.name;
    EXPECT_EQ(back.significand, v.significand) << info.name;
    EXPECT_EQ(back.unbiased_exponent, v.unbiased_exponent) << info.name;
  }
}

TEST(FloatBits, WideningThroughNarrowFloatIsExactForBinary16ToBinary32) {
  // binary16 bit pattern for 1.5: sign0, exp=01111(15), mantissa=1000000000.
  const std::uint16_t half_bits = 0x3E00;
  std::span<const std::byte> raw(reinterpret_cast<const std::byte*>(&half_bits), 2);
  const ElementTypeInfo& f16 = element_type_info(ElementType::Binary16);
  const ElementTypeInfo& f32 = element_type_info(ElementType::Binary32);
  const fd::NarrowingResult w = fd::narrow_float(raw, f16, f32);
  EXPECT_FALSE(w.changed);
  std::array<unsigned char, 4> outbuf{};
  fd::encode_float_bits(
      w.decoded, f32, std::span<std::byte>(reinterpret_cast<std::byte*>(outbuf.data()), 4));
  float widened;
  std::memcpy(&widened, outbuf.data(), 4);
  EXPECT_EQ(widened, 1.5f);
}

/// `round_significand()`'s own tie-to-even rule, isolated from any
/// particular `ElementType` -- the identical rule `fuse::bfloat16::
/// from_float()` uses, tested again here at a different width (8 -> 4
/// bits) to confirm the rule itself, not one format's own
/// implementation of it.
TEST(FloatBits, RoundSignificandTiesToEven) {
  // 0b10000000 (8 bits) narrowed to 4 bits: top 4 bits 0b1000 (even,
  // LSB=0), dropped 0b0000 exactly half (0b1000 of 8 dropped bits is the
  // tie boundary for drop=4) -- actually choose a cleaner tie: kept=
  // 0b1010 (even), dropped bits are exactly half of 2^drop.
  const fd::RoundedSignificand r1 =
      fd::round_significand(0b10100000u, 8, 4);  // kept=1010(even), drop exactly half
  EXPECT_EQ(r1.significand, 0b1010u);
  EXPECT_EQ(r1.exponent_increment, 0);

  const fd::RoundedSignificand r2 =
      fd::round_significand(0b10111000u, 8, 4);  // kept=1011(odd), dropped=1000 (exactly half) -> round up to even 1100
  EXPECT_EQ(r2.significand, 0b1100u);
  EXPECT_EQ(r2.exponent_increment, 0);

  // Overflow: all-ones kept + round up carries out of the top bit.
  const fd::RoundedSignificand r3 = fd::round_significand(0b11111111u, 8, 4);
  EXPECT_EQ(r3.significand, 0b1000u);  // renormalised
  EXPECT_EQ(r3.exponent_increment, 1);
}
