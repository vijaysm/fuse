// fuse::num::dd against exact arithmetic: every operation is compared with the exact result computed with
// expansions (test/num/exact_expansion.hpp), which share no code with the type,
// over random operands of many magnitudes, operands that cancel, and the
// special values. The bit-for-bit comparison with stillwater-sc/universal is
// test/num/dd_universal_test.cpp.

#include "fuse/num/dd.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "exact_expansion.hpp"
#include "fuse_test_main.hpp"

namespace {

using fuse::num::dd;
using fuse::test::Expansion;

static_assert(fuse::CoupledValue<dd>);
static_assert(fuse::value_traits<dd>::kind == fuse::FieldValueKind::Arithmetic);
static_assert(fuse::value_traits<dd>::scalar_count == 2);
static_assert(sizeof(dd) == 16 && alignof(dd) == 8);

Expansion exact(const dd& a) { return Expansion{a.lo, a.hi}; }

constexpr double kU2 = 0x1p-102;  // the bound below which every operation must land

/// |a - b| / |reference|, from the exact difference.
double relative_error(const Expansion& reference, const dd& got) {
  const double ref = fuse::test::approx(reference);
  const double err = fuse::test::approx(fuse::test::sub(reference, exact(got)));
  return ref == 0.0 ? std::fabs(err) : std::fabs(err / ref);
}

/// A normalised dd: |lo| < ulp(hi) / 2.
dd random_dd(std::mt19937_64& rng, int lo_exp, int hi_exp) {
  std::uniform_real_distribution<double> unit(0.5, 1.0);
  std::uniform_real_distribution<double> sym(-1.0, 1.0);
  std::uniform_int_distribution<int> expo(lo_exp, hi_exp);
  double h = std::ldexp(unit(rng), expo(rng));
  if (rng() & 1u) h = -h;
  const double l = sym(rng) * std::ldexp(std::fabs(h), -54);
  return dd(h, l);
}

/// |lo| <= ulp(hi): the form every result has (a result of the accurate sum can
/// have |lo| above half an ulp after a heavy cancellation).
bool normalised(const dd& a) {
  if (!std::isfinite(a.hi)) return true;
  if (a.hi == 0.0) return a.lo == 0.0;
  return std::fabs(a.lo) <= std::nextafter(std::fabs(a.hi), INFINITY) - std::fabs(a.hi);
}

}  // namespace

TEST(DD, ConstructionAndExactConversions) {
  const dd one(1.0);
  EXPECT_EQ(one.hi, 1.0);
  EXPECT_EQ(one.lo, 0.0);
  EXPECT_EQ(one.to_double(), 1.0);

  // from_sum and from_product hold their value exactly
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<double> d(-1e5, 1e5);
  for (int i = 0; i < 20000; ++i) {
    const double a = d(rng), b = d(rng) * 1e-7;
    const dd s = dd::from_sum(a, b);
    EXPECT_EQ(fuse::test::approx(fuse::test::sub(exact(s), Expansion{b, a})), 0.0);
    EXPECT_TRUE(normalised(s));
    const dd p = dd::from_product(a, b);
    EXPECT_EQ(fuse::test::approx(fuse::test::sub(exact(p), fuse::test::mul(Expansion{a}, Expansion{b}))), 0.0);
  }
  // to_double is the nearest binary64 value: 1 + 2^-60 rounds to 1, 1 - 2^-60 to 1
  EXPECT_EQ(dd(1.0, 0x1p-60).to_double(), 1.0);
  EXPECT_EQ(dd(1.0, -0x1p-60).to_double(), 1.0);
  // ... also for limbs that are not in normal form: a lo limb of 0.6875 ulp
  // moves the nearest value one ulp
  const double ulp = 0x1p-52;
  EXPECT_EQ(dd(1.0, 0.6875 * ulp).to_double(), 1.0 + ulp);
  // (below 1 the spacing is ulp/2, so -0.2 ulp is under a quarter of it)
  EXPECT_EQ(dd(1.0, -0.2 * ulp).to_double(), 1.0);
  EXPECT_EQ(dd(1.0, -0.4 * ulp).to_double(), 1.0 - 0.5 * ulp);
}

TEST(DD, AdditionAndSubtractionAreAccurateAcrossMagnitudesAndUnderCancellation) {
  std::mt19937_64 rng(2);
  double worst = 0.0;
  for (int i = 0; i < 50000; ++i) {
    const dd a = random_dd(rng, -40, 40);
    dd b;
    switch (i % 4) {
      case 0: b = random_dd(rng, -40, 40); break;
      case 1: b = random_dd(rng, -60, -20); break;   // very different magnitudes
      case 2: b = -a + random_dd(rng, -100, -60) * 0x1p-20; break;  // cancels to ~2^-80 of a
      default: b = -a; b.lo *= 1.0000001; break;     // cancels to the lo limb
    }
    const dd s = a + b, t = a - b;
    ASSERT_TRUE(normalised(s)) << i << " case " << i % 4;
    const double es = relative_error(fuse::test::add(exact(a), exact(b)), s);
    const double et = relative_error(fuse::test::sub(exact(a), exact(b)), t);
    worst = std::fmax(worst, std::fmax(es, et));
    ASSERT_LE(es, kU2) << i;
    ASSERT_LE(et, kU2) << i;
  }
  RecordProperty("worst_relative_error_log2", static_cast<int>(std::log2(worst)));
}

TEST(DD, MultiplicationIsAccurate) {
  std::mt19937_64 rng(3);
  for (int i = 0; i < 50000; ++i) {
    const dd a = random_dd(rng, -30, 30), b = random_dd(rng, -30, 30);
    const dd p = a * b;
    ASSERT_TRUE(normalised(p)) << i;
    ASSERT_LE(relative_error(fuse::test::mul(exact(a), exact(b)), p), kU2) << i;
  }
  // a double operand is a dd with a zero lo limb
  const dd q = dd(3.0) * 0.1;
  EXPECT_LE(relative_error(fuse::test::mul(Expansion{3.0}, Expansion{0.1}), q), kU2);
}

TEST(DD, DivisionIsAccurate) {
  std::mt19937_64 rng(4);
  for (int i = 0; i < 50000; ++i) {
    const dd a = random_dd(rng, -30, 30), b = random_dd(rng, -30, 30);
    const dd q = a / b;
    ASSERT_TRUE(normalised(q)) << i;
    // q b = a (1 + e): the residual a - q b, from exact arithmetic
    const Expansion residual = fuse::test::sub(exact(a), fuse::test::mul(exact(q), exact(b)));
    const double rel = std::fabs(fuse::test::approx(residual) / fuse::test::approx(exact(a)));
    ASSERT_LE(rel, 2 * kU2) << i;
  }
}

TEST(DD, SquareRootIsAccurateAcrossTheRange) {
  std::mt19937_64 rng(5);
  for (int i = 0; i < 30000; ++i) {
    // the exponent range keeps every product of limbs in the oracle above the
    // subnormal range
    dd a = random_dd(rng, -600, 600);
    if (a.hi < 0) a = -a;
    const dd r = fuse::num::sqrt(a);
    ASSERT_TRUE(normalised(r)) << i;
    // r^2 = a (1 + e): the residual a - r r, from exact arithmetic, relative to a / 2
    const Expansion residual = fuse::test::sub(exact(a), fuse::test::mul(exact(r), exact(r)));
    const double rel = std::fabs(fuse::test::approx(residual) / fuse::test::approx(exact(a)));
    ASSERT_LE(rel, 4 * kU2) << a.hi << " -> " << r.hi;
  }
  EXPECT_EQ(fuse::num::sqrt(dd(4.0)).hi, 2.0);
  EXPECT_EQ(fuse::num::sqrt(dd(4.0)).lo, 0.0);
  EXPECT_EQ(fuse::num::sqrt(dd(0.0)).hi, 0.0);
  EXPECT_TRUE(std::isnan(fuse::num::sqrt(dd(-1.0)).hi));
}

TEST(DD, SquareAgreesWithTheProduct) {
  std::mt19937_64 rng(6);
  for (int i = 0; i < 20000; ++i) {
    const dd a = random_dd(rng, -30, 30);
    EXPECT_LE(relative_error(fuse::test::mul(exact(a), exact(a)), fuse::num::sqr(a)), kU2);
  }
}

TEST(DD, ComparisonsOrderByTheHiLimbThenTheLoLimb) {
  const dd a(1.0, 0x1p-60), b(1.0, 0x1p-61), c(1.0000000000000002, -0x1p-60);
  EXPECT_TRUE(b < a);
  EXPECT_TRUE(a < c);
  EXPECT_TRUE(c > a);
  EXPECT_TRUE(a <= a);
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a == dd(1.0, 0x1p-60));
  const dd nan(std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE(nan < a);
  EXPECT_FALSE(nan > a);
  EXPECT_FALSE(nan == nan);
  EXPECT_EQ(fuse::num::abs(dd(-2.0, -0x1p-60)).hi, 2.0);
  EXPECT_EQ(fuse::num::abs(dd(-2.0, -0x1p-60)).lo, 0x1p-60);
}

TEST(DD, SpecialValuesPropagateAndDivisionByZeroIsIeee) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan((dd(nan) + dd(1.0)).hi));
  EXPECT_TRUE(std::isnan((dd(1.0) * dd(nan)).hi));
  EXPECT_EQ((dd(inf) + dd(1.0)).hi, inf);
  EXPECT_EQ((dd(inf) + dd(1.0)).lo, 0.0);
  EXPECT_TRUE(std::isnan((dd(inf) - dd(inf)).hi));
  EXPECT_EQ((dd(inf) * dd(-2.0)).hi, -inf);
  EXPECT_EQ((dd(1.0) / dd(0.0)).hi, inf);
  EXPECT_EQ((dd(-1.0) / dd(0.0)).hi, -inf);
  EXPECT_TRUE(std::isnan((dd(0.0) / dd(0.0)).hi));
  EXPECT_EQ((dd(1.0) / dd(inf)).hi, 0.0);
  EXPECT_TRUE(std::signbit((dd(-1.0) / dd(inf)).hi));
  EXPECT_TRUE(std::isnan((dd(inf) / dd(inf)).hi));
  // zero and signed zero
  EXPECT_EQ((dd(0.0) + dd(0.0)).hi, 0.0);
  EXPECT_EQ((dd(2.0) - dd(2.0)).hi, 0.0);
  EXPECT_EQ((dd(2.0) - dd(2.0)).lo, 0.0);
}

TEST(DD, TheTraitsDescribeTwoBinary64Limbs) {
  using T = fuse::value_traits<dd>;
  EXPECT_EQ(T::name, "dd");
  EXPECT_EQ(T::gradient_size, 0);
  EXPECT_EQ(T::to_binary64(T::from_binary64(0.1)), 0.1);
  EXPECT_EQ(T::to_binary64(dd(1.0, 0x1p-60)), 1.0);
  EXPECT_EQ(T::to_binary64(dd(1.0, 0.6875 * 0x1p-52)), 1.0 + 0x1p-52);
  const auto t = fuse::field_value_type_of<dd>();
  EXPECT_EQ(t.size, 16u);
  EXPECT_EQ(t.scalar_count, 2);
  EXPECT_EQ(t.scalar_name, "binary64");
  EXPECT_EQ(t.scalar_size, 8u);
}
