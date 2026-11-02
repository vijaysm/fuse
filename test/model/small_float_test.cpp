// fuse::detail::SmallFloat (plans/2026-10-22-universal-number-types.md, stage
// U3): the generic binary64 <-> small-format conversion is instantiated for
// binary32, where the hardware conversion is the independent answer, and for
// bfloat16, where the answer is the exact definition (a bit shift for widening,
// and a table search for narrowing).

#include "fuse/model/small_float.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "fuse/model/bfloat16.hpp"

namespace {

using F32 = fuse::detail::SmallFloat<8, 23>;
using BF = fuse::detail::SmallFloat<8, 7>;

static_assert(F32::bias == 127 && F32::min_exponent == -126 && F32::precision == 24);
static_assert(BF::bias == 127 && BF::min_exponent == -126 && BF::precision == 8);

std::uint32_t hw(double x) { return std::bit_cast<std::uint32_t>(static_cast<float>(x)); }

bool both_nan(std::uint32_t a, std::uint32_t b) {
  const auto isnan = [](std::uint32_t v) { return (v & 0x7FFFFFFFu) > 0x7F800000u; };
  return isnan(a) && isnan(b);
}

}  // namespace

TEST(SmallFloat, Binary32NarrowingEqualsTheHardwareConversion) {
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> exponent(-170.0, 130.0);
  for (int n = 0; n < 500000; ++n) {
    double x = std::exp2(exponent(rng)) * (1.0 + std::ldexp(static_cast<double>(rng() >> 11), -53));
    if (rng() & 1u) x = -x;
    ASSERT_EQ(F32::narrow(x), hw(x)) << std::hexfloat << x;
  }
  // raw binary64 bit patterns, the whole exponent range included
  for (int n = 0; n < 500000; ++n) {
    const double x = std::bit_cast<double>(rng());
    const std::uint32_t a = F32::narrow(x), b = hw(x);
    if (std::isnan(x)) {
      ASSERT_TRUE(both_nan(a, b));
      continue;
    }
    ASSERT_EQ(a, b) << std::hexfloat << x;
  }
}

TEST(SmallFloat, Binary32MidpointsAndTheEdgesAreRoundedLikeTheHardware) {
  // every midpoint between adjacent binary32 values in a few binades, and one
  // binary64 step each side
  for (const std::uint32_t base : {0x00000000u, 0x007FFFF0u, 0x00800000u, 0x3F7FFFF0u,
                                   0x3F800000u, 0x7F7FFFF0u}) {
    for (std::uint32_t k = 0; k < 20; ++k) {
      const std::uint32_t p = base + k;
      if (p >= 0x7F800000u) break;
      const double lo = static_cast<double>(std::bit_cast<float>(p));
      const double hi = static_cast<double>(std::bit_cast<float>(p + 1));
      const double mid = 0.5 * (lo + hi);
      for (const double x : {mid, std::nextafter(mid, 0.0), std::nextafter(mid, 1e300), lo, hi}) {
        for (const double s : {x, -x}) ASSERT_EQ(F32::narrow(s), hw(s)) << std::hexfloat << s;
      }
    }
  }
  const double fmax = static_cast<double>(std::numeric_limits<float>::max());
  const double over = fmax + std::ldexp(1.0, 103);  // the midpoint above float max
  EXPECT_EQ(F32::narrow(over), hw(over));
  EXPECT_EQ(F32::narrow(std::nextafter(over, 0.0)), hw(std::nextafter(over, 0.0)));
  EXPECT_EQ(F32::narrow(std::ldexp(1.0, -150)), hw(std::ldexp(1.0, -150)));  // a tie to zero
  EXPECT_EQ(F32::narrow(std::nextafter(std::ldexp(1.0, -150), 1.0)),
            hw(std::nextafter(std::ldexp(1.0, -150), 1.0)));
}

TEST(SmallFloat, Binary32WideningIsTheHardwareConversionForSampledPatterns) {
  std::mt19937 rng(5);
  for (int n = 0; n < 500000; ++n) {
    const std::uint32_t p = rng();
    const double d = F32::widen(p);
    const float f = std::bit_cast<float>(p);
    if (std::isnan(f)) {
      ASSERT_TRUE(std::isnan(d));
      continue;
    }
    ASSERT_EQ(std::bit_cast<std::uint64_t>(d), std::bit_cast<std::uint64_t>(static_cast<double>(f))) << std::hex << p;
  }
}

TEST(SmallFloat, BFloat16WideningIsTheShiftTheTypeUsesForFloat) {
  for (unsigned v = 0; v <= 0xFFFFu; ++v) {
    const float f = fuse::bfloat16::from_bits(static_cast<std::uint16_t>(v)).to_float();
    const double d = BF::widen(v);
    if (std::isnan(f)) {
      ASSERT_TRUE(std::isnan(d));
      continue;
    }
    ASSERT_EQ(std::bit_cast<std::uint64_t>(d), std::bit_cast<std::uint64_t>(static_cast<double>(f))) << std::hex << v;
  }
}

TEST(SmallFloat, BFloat16NarrowingAgreesWithTheNearestSearch) {
  // the 32640 finite non-negative values, from the definition
  std::vector<double> t;
  for (unsigned p = 0; p <= 0x7F7F; ++p) {
    const unsigned e = p >> 7, f = p & 0x7F;
    t.push_back(e == 0 ? std::ldexp(static_cast<double>(f), -133)
                       : std::ldexp(static_cast<double>(128 + f), static_cast<int>(e) - 134));
  }
  ASSERT_TRUE(std::is_sorted(t.begin(), t.end()));
  const double top = std::ldexp(255.0, 120);  // largest finite, 3.3895e38
  ASSERT_EQ(t.back(), top);
  const auto oracle = [&](double x) -> std::uint32_t {
    if (x >= top + std::ldexp(1.0, 119)) return 0x7F80u;  // the midpoint above the largest, a tie to even: infinity
    const auto hi = std::lower_bound(t.begin(), t.end(), x);
    if (hi == t.begin()) return 0;
    if (hi == t.end()) return 0x7F7Fu;
    const auto lo = hi - 1;
    const double dlo = x - *lo, dhi = *hi - x;
    const auto plo = static_cast<std::uint32_t>(lo - t.begin());
    if (dlo < dhi) return plo;
    if (dhi < dlo) return plo + 1;
    return (plo & 1u) == 0 ? plo : plo + 1;
  };
  for (std::size_t i = 0; i + 1 < t.size(); ++i) {
    const double mid = 0.5 * (t[i] + t[i + 1]);
    ASSERT_EQ(BF::narrow(mid), oracle(mid)) << i;
    ASSERT_EQ(BF::narrow(std::nextafter(mid, 0.0)), static_cast<std::uint32_t>(i)) << i;
    ASSERT_EQ(BF::narrow(std::nextafter(mid, 1e300)), static_cast<std::uint32_t>(i + 1)) << i;
    ASSERT_EQ(BF::narrow(-mid), 0x8000u | oracle(mid)) << i;
  }
  std::mt19937_64 rng(9);
  std::uniform_real_distribution<double> exponent(-140.0, 129.0);
  for (int n = 0; n < 300000; ++n) {
    const double x = std::exp2(exponent(rng)) * (1.0 + std::ldexp(static_cast<double>(rng() >> 11), -53));
    ASSERT_EQ(BF::narrow(x), oracle(x)) << std::hexfloat << x;
  }
}

TEST(SmallFloat, BFloat16FromDoubleAndFromFloatAgreeOnBinary32Values) {
  std::mt19937 rng(13);
  for (int n = 0; n < 1000000; ++n) {
    const float f = std::bit_cast<float>(static_cast<std::uint32_t>(rng()));
    if (std::isnan(f)) continue;
    ASSERT_EQ(fuse::bfloat16::from_double(static_cast<double>(f)).bits(),
              fuse::bfloat16::from_float(f).bits())
        << std::hexfloat << f;
  }
}
