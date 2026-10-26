// fuse::num::qd against exact arithmetic (plans/2026-10-22-universal-number-types.md,
// stage U4): every operation is compared with the exact result computed with
// expansions (test/num/exact_expansion.hpp), which share no code with the type,
// over random operands of many magnitudes, operands that cancel, and the special
// values. The bit-for-bit comparison with stillwater-sc/universal is
// test/num/dd_universal_test.cpp.

#include "fuse/num/qd.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>

#include "exact_expansion.hpp"
#include "fuse_test_main.hpp"

namespace {

using fuse::num::dd;
using fuse::num::qd;
using fuse::test::Expansion;

static_assert(fuse::CoupledValue<qd>);
static_assert(fuse::value_traits<qd>::kind == fuse::FieldValueKind::Arithmetic);
static_assert(fuse::value_traits<qd>::scalar_count == 4);
static_assert(sizeof(qd) == 32 && alignof(qd) == 8);

Expansion exact(const qd& a) {
  Expansion e;
  for (int i = 3; i >= 0; --i) e = fuse::test::grow(e, a[i]);
  return e;
}

constexpr double kU4 = 0x1p-206;  // the bound below which every operation must land

/// |a - b| / |reference|, from the exact difference.
double relative_error(const Expansion& reference, const qd& got) {
  const double ref = fuse::test::approx(reference);
  const double err = fuse::test::approx(fuse::test::sub(reference, exact(got)));
  return ref == 0.0 ? std::fabs(err) : std::fabs(err / ref);
}

/// A qd with |x[i+1]| <= 2^-54 |x[i]|: the generators of the format.
qd random_qd(std::mt19937_64& rng, int lo_exp, int hi_exp) {
  std::uniform_real_distribution<double> unit(0.5, 1.0);
  std::uniform_real_distribution<double> sym(-1.0, 1.0);
  std::uniform_int_distribution<int> expo(lo_exp, hi_exp);
  qd a(0.0);
  a[0] = std::ldexp(unit(rng), expo(rng));
  if (rng() & 1u) a[0] = -a[0];
  for (int i = 1; i < 4; ++i) a[i] = sym(rng) * std::ldexp(std::fabs(a[i - 1]), -54);
  return a;
}

/// |x[i+1]| <= ulp(x[i]) for every limb: the form every result has.
bool normalised(const qd& a) {
  if (!std::isfinite(a[0])) return true;
  for (int i = 0; i < 3; ++i) {
    if (a[i] == 0.0) {
      if (a[i + 1] != 0.0) return false;
    } else if (std::fabs(a[i + 1]) > std::nextafter(std::fabs(a[i]), INFINITY) - std::fabs(a[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

TEST(QD, ConstructionAndExactConversions) {
  const qd one(1.0);
  EXPECT_EQ(one[0], 1.0);
  EXPECT_EQ(one[1], 0.0);
  EXPECT_EQ(one[2], 0.0);
  EXPECT_EQ(one[3], 0.0);
  EXPECT_EQ(one.to_double(), 1.0);
  const qd from_dd(dd(1.0, 0x1p-60));
  EXPECT_EQ(from_dd[1], 0x1p-60);
  EXPECT_EQ(from_dd[2], 0.0);
  const qd x(1.0, 0x1p-60, 0x1p-120, 0x1p-180);
  const dd d = x.to_dd();
  EXPECT_EQ(d.hi, 1.0);
  EXPECT_EQ(d.lo, 0x1p-60);
  // negation is exact limb by limb
  const qd n = -x;
  for (int i = 0; i < 4; ++i) EXPECT_EQ(n[i], -x[i]);
  // a power of two scales every limb
  const qd s = fuse::num::ldexp(x, 10);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(s[i], std::ldexp(x[i], 10));
  const qd m = fuse::num::mul_pwr2(x, 0.25);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(m[i], 0.25 * x[i]);
  // to_double is the nearest binary64 value of the whole sum, also for limbs that
  // are not in normal form
  EXPECT_EQ(qd(1.0, 0.6875 * 0x1p-52, 0.0, 0.0).to_double(), 1.0 + 0x1p-52);
  EXPECT_EQ(qd(1.0, -0.2 * 0x1p-52, 0.0, 0.0).to_double(), 1.0);
}

TEST(QD, AdditionAndSubtractionAreAccurateAcrossMagnitudesAndUnderCancellation) {
  std::mt19937_64 rng(12);
  double worst = 0.0;
  for (int i = 0; i < 20000; ++i) {
    const qd a = random_qd(rng, -40, 40);
    qd b(0.0);
    switch (i % 4) {
      case 0: b = random_qd(rng, -40, 40); break;
      case 1: b = random_qd(rng, -80, -20); break;  // very different magnitudes
      case 2: b = -a + random_qd(rng, -150, -100) * 0x1p-20; break;  // cancels to ~2^-120 of a
      default: b = -a; b[3] *= 1.0000001; break;                      // cancels to the last limb
    }
    const qd s = a + b, t = a - b;
    ASSERT_TRUE(normalised(s)) << i << " case " << i % 4;
    const double es = relative_error(fuse::test::add(exact(a), exact(b)), s);
    const double et = relative_error(fuse::test::sub(exact(a), exact(b)), t);
    worst = std::fmax(worst, std::fmax(es, et));
    ASSERT_LE(es, kU4) << i << " case " << i % 4;
    ASSERT_LE(et, kU4) << i << " case " << i % 4;
  }
  RecordProperty("worst_relative_error_log2", static_cast<int>(std::log2(worst)));
}

TEST(QD, MultiplicationIsAccurate) {
  std::mt19937_64 rng(13);
  double worst = 0.0;
  for (int i = 0; i < 10000; ++i) {
    const qd a = random_qd(rng, -30, 30), b = random_qd(rng, -30, 30);
    const qd p = a * b;
    ASSERT_TRUE(normalised(p)) << i;
    const double e = relative_error(fuse::test::mul(exact(a), exact(b)), p);
    worst = std::fmax(worst, e);
    ASSERT_LE(e, kU4) << i;
  }
  RecordProperty("worst_relative_error_log2", static_cast<int>(std::log2(worst)));
  // a double operand is a qd with zero trailing limbs
  const qd q = qd(3.0) * 0.1;
  EXPECT_LE(relative_error(fuse::test::mul(Expansion{3.0}, Expansion{0.1}), q), kU4);
}

TEST(QD, DivisionIsAccurate) {
  std::mt19937_64 rng(14);
  double worst = 0.0;
  for (int i = 0; i < 10000; ++i) {
    const qd a = random_qd(rng, -30, 30), b = random_qd(rng, -30, 30);
    const qd q = a / b;
    ASSERT_TRUE(normalised(q)) << i;
    // q b = a (1 + e): the residual a - q b, from exact arithmetic
    const Expansion residual = fuse::test::sub(exact(a), fuse::test::mul(exact(q), exact(b)));
    const double rel = std::fabs(fuse::test::approx(residual) / fuse::test::approx(exact(a)));
    worst = std::fmax(worst, rel);
    ASSERT_LE(rel, 2 * kU4) << i;
  }
  RecordProperty("worst_relative_error_log2", static_cast<int>(std::log2(worst)));
}

TEST(QD, SquareRootIsAccurateAcrossTheRange) {
  std::mt19937_64 rng(15);
  double worst = 0.0;
  for (int i = 0; i < 10000; ++i) {
    // the exponent range keeps every product of limbs in the oracle (and of the
    // lowest limbs of the result) above the subnormal range
    qd a = random_qd(rng, -300, 300);
    if (a[0] < 0) a = -a;
    const qd r = fuse::num::sqrt(a);
    ASSERT_TRUE(normalised(r)) << i;
    const Expansion residual = fuse::test::sub(exact(a), fuse::test::mul(exact(r), exact(r)));
    const double rel = std::fabs(fuse::test::approx(residual) / fuse::test::approx(exact(a)));
    worst = std::fmax(worst, rel);
    ASSERT_LE(rel, 4 * kU4) << a[0] << " -> " << r[0];
  }
  RecordProperty("worst_relative_error_log2", static_cast<int>(std::log2(worst)));
  EXPECT_EQ(fuse::num::sqrt(qd(4.0))[0], 2.0);
  EXPECT_EQ(fuse::num::sqrt(qd(4.0))[1], 0.0);
  EXPECT_EQ(fuse::num::sqrt(qd(0.0))[0], 0.0);
  EXPECT_TRUE(std::isnan(fuse::num::sqrt(qd(-1.0))[0]));
}

TEST(QD, SquareAgreesWithTheProduct) {
  std::mt19937_64 rng(16);
  for (int i = 0; i < 5000; ++i) {
    const qd a = random_qd(rng, -30, 30);
    EXPECT_LE(relative_error(fuse::test::mul(exact(a), exact(a)), fuse::num::sqr(a)), kU4);
  }
}

TEST(QD, ComparisonsOrderLexicographicallyByLimb) {
  const qd a(1.0, 0x1p-60, 0x1p-120, 0.0), b(1.0, 0x1p-60, 0.0, 0.0), c(1.0, 0x1p-59, -0x1p-120, 0.0);
  EXPECT_TRUE(b < a);
  EXPECT_TRUE(a < c);
  EXPECT_TRUE(c > a);
  EXPECT_TRUE(a <= a);
  EXPECT_TRUE(a >= b);
  EXPECT_TRUE(a != b);
  EXPECT_TRUE(a == qd(1.0, 0x1p-60, 0x1p-120, 0.0));
  const qd nan(std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE(nan < a);
  EXPECT_FALSE(nan > a);
  EXPECT_FALSE(nan == nan);
  const qd m = fuse::num::abs(qd(-2.0, -0x1p-60, 0.0, 0.0));
  EXPECT_EQ(m[0], 2.0);
  EXPECT_EQ(m[1], 0x1p-60);
}

TEST(QD, SpecialValuesPropagateAndDivisionByZeroIsIeee) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(std::isnan((qd(nan) + qd(1.0))[0]));
  EXPECT_TRUE(std::isnan((qd(1.0) * qd(nan))[0]));
  EXPECT_EQ((qd(inf) + qd(1.0))[0], inf);
  EXPECT_TRUE(std::isnan((qd(inf) - qd(inf))[0]));
  EXPECT_EQ((qd(inf) * qd(-2.0))[0], -inf);
  EXPECT_EQ((qd(1.0) / qd(0.0))[0], inf);
  EXPECT_EQ((qd(-1.0) / qd(0.0))[0], -inf);
  EXPECT_TRUE(std::isnan((qd(0.0) / qd(0.0))[0]));
  EXPECT_EQ((qd(1.0) / qd(inf))[0], 0.0);
  EXPECT_TRUE(std::isnan((qd(inf) / qd(inf))[0]));
  EXPECT_EQ((qd(0.0) + qd(0.0))[0], 0.0);
  const qd z = qd(2.0) - qd(2.0);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(z[i], 0.0);
}

TEST(QD, TheTraitsDescribeFourBinary64Limbs) {
  using T = fuse::value_traits<qd>;
  EXPECT_EQ(T::name, "qd");
  EXPECT_EQ(T::gradient_size, 0);
  EXPECT_EQ(T::to_binary64(T::from_binary64(0.1)), 0.1);
  EXPECT_EQ(T::to_binary64(qd(1.0, 0x1p-60, 0.0, 0.0)), 1.0);
  const auto t = fuse::field_value_type_of<qd>();
  EXPECT_EQ(t.size, 32u);
  EXPECT_EQ(t.scalar_count, 4);
  EXPECT_EQ(t.scalar_name, "binary64");
  EXPECT_EQ(t.scalar_size, 8u);
}
