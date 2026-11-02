// fuse::binary16 (plans/2026-10-22-universal-number-types.md, stage U3): the
// conversions are checked against an oracle that shares no code with them -- a
// table of the 31744 finite non-negative binary16 values decoded with ldexp,
// searched for the nearest -- at every midpoint, one binary64 step either side
// of it, and over a log-uniform sample, and against the compiler's own _Float16
// where it has one.

#include "fuse/model/binary16.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "fuse/model/element_type.hpp"

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// value of the non-negative finite pattern p (0 <= p <= 0x7BFF), from the
// definition of the format
double decode(unsigned p) {
  const unsigned e = p >> 10, f = p & 0x3FF;
  return e == 0 ? std::ldexp(static_cast<double>(f), -24)
                : std::ldexp(static_cast<double>(1024 + f), static_cast<int>(e) - 25);
}

// the 31744 finite non-negative values, in increasing order (the pattern is
// monotone in the value for a non-negative number)
const std::vector<double>& table() {
  static const std::vector<double> t = [] {
    std::vector<double> v;
    for (unsigned p = 0; p <= 0x7BFF; ++p) v.push_back(decode(p));
    return v;
  }();
  return t;
}

// the correctly rounded pattern of x >= 0 by exhaustive comparison with the
// neighbours in the table; a tie goes to the even pattern
std::uint16_t oracle(double x) {
  if (std::isnan(x)) return 0x7E00;
  if (x >= 65520.0) return 0x7C00;  // the midpoint above 65504 rounds to even: up, to infinity
  const auto& t = table();
  const auto hi = std::lower_bound(t.begin(), t.end(), x);  // first value >= x
  if (hi == t.begin()) return 0;
  const auto lo = hi - 1;
  const double dlo = x - *lo;
  if (hi == t.end()) return static_cast<std::uint16_t>(lo - t.begin());
  const double dhi = *hi - x;
  const auto plo = static_cast<std::uint16_t>(lo - t.begin());
  if (dlo < dhi) return plo;
  if (dhi < dlo) return static_cast<std::uint16_t>(plo + 1);
  return (plo & 1u) == 0 ? plo : static_cast<std::uint16_t>(plo + 1);
}

std::uint16_t oracle_signed(double x) {
  const std::uint16_t sign = std::signbit(x) ? 0x8000u : 0u;
  return static_cast<std::uint16_t>(sign | oracle(std::fabs(x)));
}

std::uint16_t got(double x) { return fuse::binary16::from_double(x).bits(); }

}  // namespace

static_assert(sizeof(fuse::binary16) == 2);
static_assert(std::is_trivially_copyable_v<fuse::binary16>);

TEST(Binary16, EveryPatternWidensExactlyToItsDefinition) {
  int nan_count = 0;
  for (unsigned v = 0; v <= 0xFFFFu; ++v) {
    const fuse::binary16 h = fuse::binary16::from_bits(static_cast<std::uint16_t>(v));
    const unsigned mag = v & 0x7FFFu;
    const double d = h.to_double();
    if (mag > 0x7C00u) {
      ++nan_count;
      EXPECT_TRUE(std::isnan(d)) << std::hex << v;
      EXPECT_TRUE(std::isnan(h.to_float())) << std::hex << v;
      continue;
    }
    const double expect = mag == 0x7C00u ? kInf : decode(mag);
    EXPECT_EQ(d, (v & 0x8000u) ? -expect : expect) << std::hex << v;
    EXPECT_EQ(std::signbit(d), (v & 0x8000u) != 0) << std::hex << v;  // -0 stays -0
    EXPECT_EQ(static_cast<double>(h.to_float()), d) << std::hex << v;
  }
  EXPECT_EQ(nan_count, 2 * 1023);
}

TEST(Binary16, EveryNonNanPatternSurvivesAWidenAndANarrow) {
  for (unsigned v = 0; v <= 0xFFFFu; ++v) {
    if ((v & 0x7FFFu) > 0x7C00u) continue;
    const fuse::binary16 h = fuse::binary16::from_bits(static_cast<std::uint16_t>(v));
    EXPECT_EQ(fuse::binary16::from_double(h.to_double()).bits(), v) << std::hex << v;
    EXPECT_EQ(fuse::binary16::from_float(h.to_float()).bits(), v) << std::hex << v;
  }
}

TEST(Binary16, ANanStaysANanWithItsSign) {
  EXPECT_EQ(got(std::numeric_limits<double>::quiet_NaN()), 0x7E00u);
  EXPECT_EQ(got(-std::numeric_limits<double>::quiet_NaN()), 0xFE00u);
  EXPECT_EQ(got(std::numeric_limits<double>::signaling_NaN()), 0x7E00u);
  EXPECT_EQ(fuse::binary16::from_float(std::numeric_limits<float>::quiet_NaN()).bits(), 0x7E00u);
}

TEST(Binary16, SpecialValuesAndTheEdgesOfTheRange) {
  EXPECT_EQ(got(0.0), 0x0000u);
  EXPECT_EQ(got(-0.0), 0x8000u);
  EXPECT_EQ(got(kInf), 0x7C00u);
  EXPECT_EQ(got(-kInf), 0xFC00u);
  EXPECT_EQ(got(1.0), 0x3C00u);
  EXPECT_EQ(got(65504.0), 0x7BFFu);            // largest finite
  EXPECT_EQ(got(65519.99999999999), 0x7BFFu);  // below the midpoint
  EXPECT_EQ(got(65520.0), 0x7C00u);            // the midpoint ties to the even neighbour: infinity
  EXPECT_EQ(got(-65520.0), 0xFC00u);
  EXPECT_EQ(got(std::numeric_limits<double>::max()), 0x7C00u);
  EXPECT_EQ(got(std::ldexp(1.0, -14)), 0x0400u);  // smallest normal
  EXPECT_EQ(got(std::ldexp(1.0, -24)), 0x0001u);  // smallest subnormal
  EXPECT_EQ(got(std::ldexp(1.0, -25)), 0x0000u);  // half of it: a tie, to the even zero
  EXPECT_EQ(got(std::nextafter(std::ldexp(1.0, -25), 1.0)), 0x0001u);
  EXPECT_EQ(got(-std::ldexp(1.0, -25)), 0x8000u);
  EXPECT_EQ(got(std::numeric_limits<double>::denorm_min()), 0x0000u);
  EXPECT_EQ(got(-std::numeric_limits<double>::denorm_min()), 0x8000u);
  EXPECT_EQ(got(std::numeric_limits<double>::min()), 0x0000u);
}

TEST(Binary16, EveryMidpointAndBothBinary64NeighboursRoundCorrectly) {
  const auto& t = table();
  long checked = 0;
  for (std::size_t i = 0; i + 1 < t.size(); ++i) {
    const double mid = 0.5 * (t[i] + t[i + 1]);  // exact: both have at most 11 significant bits
    const auto even = static_cast<std::uint16_t>((i & 1u) == 0 ? i : i + 1);
    EXPECT_EQ(got(mid), even) << i;
    EXPECT_EQ(got(std::nextafter(mid, 0.0)), static_cast<std::uint16_t>(i)) << i;
    EXPECT_EQ(got(std::nextafter(mid, kInf)), static_cast<std::uint16_t>(i + 1)) << i;
    EXPECT_EQ(got(-mid), static_cast<std::uint16_t>(0x8000u | even)) << i;
    // and the value itself is a fixed point
    EXPECT_EQ(got(t[i]), static_cast<std::uint16_t>(i)) << i;
    checked += 4;
  }
  EXPECT_GT(checked, 100000);
}

TEST(Binary16, AgreesWithTheNearestSearchOverALogUniformSample) {
  std::mt19937_64 rng(20261025);
  std::uniform_real_distribution<double> exponent(-30.0, 17.5);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  for (int n = 0; n < 400000; ++n) {
    double x = std::exp2(exponent(rng));
    if (unit(rng) < 0.5) x = -x;
    ASSERT_EQ(got(x), oracle_signed(x)) << std::hexfloat << x;
  }
}

TEST(Binary16, RoundingTwiceThroughBinary32WouldHaveBeenWrong) {
  // 1 and 1 + 2^-10 are adjacent; their midpoint is 1 + 2^-11. A binary64
  // value one ulp above the midpoint is closer to 1 + 2^-10, and it also
  // rounds to the midpoint in binary32 (24 bits), where the tie goes to the
  // even neighbour, 1.
  const double mid = 1.0 + std::ldexp(1.0, -11);
  const double above = std::nextafter(mid, 2.0);
  ASSERT_NE(above, mid);
  ASSERT_EQ(static_cast<double>(static_cast<float>(above)), mid);
  EXPECT_EQ(got(above), 0x3C01u);
  EXPECT_EQ(fuse::binary16::from_float(static_cast<float>(above)).bits(), 0x3C00u);
}

TEST(Binary16, FloatConversionIsTheDoubleConversionOfTheSameValue) {
  std::mt19937 rng(7);
  for (int n = 0; n < 200000; ++n) {
    const float f = std::bit_cast<float>(static_cast<std::uint32_t>(rng()));
    if (std::isnan(f)) continue;
    ASSERT_EQ(fuse::binary16::from_float(f).bits(), got(static_cast<double>(f)));
  }
}

#if FUSE_NATIVE_FLOAT16
TEST(Binary16, AgreesWithTheCompilersOwnHalfOnRandomBinary64AndBinary32) {
  std::mt19937_64 rng(11);
  std::uniform_real_distribution<double> exponent(-32.0, 18.0);
  for (int n = 0; n < 400000; ++n) {
    double x = std::exp2(exponent(rng)) * (1.0 + std::ldexp(static_cast<double>(rng() & 0xFFFFF), -20));
    if (rng() & 1u) x = -x;
    const _Float16 h = static_cast<_Float16>(x);
    ASSERT_EQ(got(x), std::bit_cast<std::uint16_t>(h)) << std::hexfloat << x;
  }
}
#endif
