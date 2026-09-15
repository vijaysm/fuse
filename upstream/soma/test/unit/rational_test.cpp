#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

#include <soma/base/rational.hpp>

using soma::gcd_rational;
using soma::Rational;

TEST(Rational, NormalisesOnConstruction) {
  constexpr Rational r{6, 8};
  static_assert(r.num() == 3);
  static_assert(r.den() == 4);
  EXPECT_EQ(r.to_string(), "3/4");
}

TEST(Rational, PutsSignOnNumerator) {
  constexpr Rational r{1, -2};
  static_assert(r.num() == -1);
  static_assert(r.den() == 2);
}

TEST(Rational, IntegerPrintsWithoutDenominator) {
  EXPECT_EQ(Rational{5}.to_string(), "5");
  EXPECT_TRUE(Rational(4, 2).is_integer());
}

TEST(Rational, Arithmetic) {
  constexpr Rational a{1, 3};
  constexpr Rational b{1, 7};
  static_assert((a + b) == Rational(10, 21));
  static_assert((a - b) == Rational(4, 21));
  static_assert((a * b) == Rational(1, 21));
  static_assert((a / b) == Rational(7, 3));
}

TEST(Rational, Ordering) {
  EXPECT_LT(Rational(1, 3), Rational(1, 2));
  EXPECT_EQ(Rational(2, 4), Rational(1, 2));
  EXPECT_GT(Rational(3, 4), Rational(2, 3));
}

// The property that matters for §8.2: no drift over long runs.
TEST(Rational, NoDriftOverManySteps) {
  const Rational dt{1, 3};
  Rational t{0};
  for (int i = 0; i < 1'000'000; ++i) t += dt;
  EXPECT_EQ(t, Rational(1'000'000, 3));
  EXPECT_EQ(t.den(), 3);  // still exact, not renormalised to something huge
}

TEST(Rational, LcmDenominatorGivesFinestCommonTick) {
  EXPECT_EQ(Rational::lcm_denominator(Rational(1, 3), Rational(1, 7)),
            Rational(1, 21));
  EXPECT_EQ(Rational::lcm_denominator(Rational(1, 4), Rational(1, 2)),
            Rational(1, 4));
}

// gcd_rational(a,b) is the largest g with a = i*g and b = j*g for integers i,j.
TEST(Rational, GcdRationalIsCommonDivisor) {
  EXPECT_EQ(gcd_rational(Rational(1, 3), Rational(1, 7)), Rational(1, 21));
  EXPECT_EQ(gcd_rational(Rational(2, 3), Rational(4, 3)), Rational(2, 3));
  EXPECT_EQ(gcd_rational(Rational(3), Rational(2)), Rational(1));
}

TEST(Rational, ToDouble) {
  EXPECT_DOUBLE_EQ(Rational(1, 4).to_double(), 0.25);
}

// Operands exceed sqrt(INT64_MAX) (~3.037e9), so a.num*b.den and b.num*a.den
// (~9.61e18 each) both overflow int64_t (INT64_MAX ~9.223e18) under naive
// 64-bit cross-multiplication. Verified empirically against the pre-fix
// header (git rev ae75011): compiled under UBSan, this exact case trips
// "signed integer overflow: 3100000000 * 3100000002 cannot be represented
// in type 'std::int64_t'" at the old `operator<=>`'s cross-multiply line,
// and produces the wrong (wrapped) answer without a sanitizer. With the
// __int128 intermediate the comparison is exact and untriggered by UBSan.
TEST(Rational, ComparisonNeeds128BitIntermediate) {
  constexpr std::int64_t N = 3'100'000'000LL;  // > sqrt(INT64_MAX)
  EXPECT_LT(Rational(N, N + 1), Rational(N + 1, N + 2));
}

// D1 > sqrt(INT64_MAX), so operator+='s cross term o.num_ * den_ =
// (D1 - 1) * D1 (~9.61e18) overflows int64_t (INT64_MAX ~9.223e18) under
// naive 64-bit arithmetic. Verified empirically against the pre-fix header:
// UBSan reports "signed integer overflow: 3099999999 * 3100000000 cannot be
// represented" at the old operator+='s numerator line, and without a
// sanitizer the naive build silently computes num=den=-8836744073709551616,
// which is not 1 — i.e. this test genuinely FAILS (EXPECT_EQ mismatch)
// against the pre-fix header. With the __int128 intermediate the sum
// reduces exactly to 1.
TEST(Rational, AdditionWithOverflowingIntermediateNormalisesExactly) {
  constexpr std::int64_t kBigDen = 3'100'000'000LL;  // > sqrt(INT64_MAX)
  const Rational a{1, kBigDen};
  const Rational b{kBigDen - 1, kBigDen};
  EXPECT_EQ(a + b, Rational(1));
}

// Four pairwise-coprime primes, each > sqrt(INT64_MAX), so every pairwise
// product formed by operator*= (~9.61e18) overflows int64_t under naive
// 64-bit arithmetic, yet the chain p1/p2 * p2/p3 * p3/p4 * p4/p1 telescopes
// to exactly 1. Verified empirically against the pre-fix header: UBSan
// reports "signed integer overflow: 3100000027 * 3100010033 cannot be
// represented" at the old operator*='s first multiply, and without a
// sanitizer the naive build computes num=den=-8836650701708739105 (not 1)
// — i.e. this test genuinely FAILS against the pre-fix header. With the
// __int128 intermediate every step reduces exactly, ending at 1.
TEST(Rational, ChainedMultiplyAcrossUnrelatedDenominatorsStaysExact) {
  constexpr std::int64_t p1 = 3'100'000'027LL;
  constexpr std::int64_t p2 = 3'100'010'033LL;
  constexpr std::int64_t p3 = 3'100'020'083LL;
  constexpr std::int64_t p4 = 3'100'030'093LL;
  Rational r{p1, p2};
  r *= Rational(p2, p3);
  r *= Rational(p3, p4);
  r *= Rational(p4, p1);
  EXPECT_EQ(r, Rational(1));
}

// Pins the chosen INT64_MIN handling: constructing Rational(INT64_MIN, 1)
// must not be UB (the old code computed -num_ and num_ < 0 ? -num_ : num_
// directly in int64_t, which is UB at INT64_MIN). Since gcd(|INT64_MIN|, 1)
// == 1, the value passes through unchanged and unreduced.
TEST(Rational, Int64MinNumeratorIsPreservedWithoutUB) {
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  const Rational r{kMin, 1};
  EXPECT_EQ(r.num(), kMin);
  EXPECT_EQ(r.den(), 1);
}

// Same hazard, but through a negative denominator: normalise() used to do
// `den_ = -den_` in int64_t, UB when den_ == INT64_MIN (its negation, 2^63,
// doesn't even fit in int64_t). Numerator 2 is chosen so gcd(2, |INT64_MIN|)
// == 2, exercising the reduction path too. Expected result: sign moves to
// the numerator, magnitude divides by 2: -1 / 4611686018427387904.
TEST(Rational, Int64MinDenominatorNormalisesWithoutUB) {
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  const Rational r{2, kMin};
  EXPECT_EQ(r.num(), -1);
  EXPECT_EQ(r.den(), 4611686018427387904LL);
}

// gcd_rational(Rational(INT64_MIN,1), Rational(INT64_MIN,1)) has true gcd
// |INT64_MIN| == 2^63, which does not fit in int64_t (max is 2^63 - 1).
// Every member operator throws std::logic_error rather than silently narrow
// an unrepresentable reduced value; gcd_rational must honour the same
// contract instead of letting static_cast<int64_t>(2^63) wrap to INT64_MIN.
// Enforcement is unconditional (detail::check_fits_i64 throws regardless of
// -DNDEBUG), so EXPECT_THROW is the honest assertion here -- unlike
// EXPECT_DEATH, it verifies the guard survives Release, not just that the
// process crashed somehow. Confirmed this test fails against the pre-fix,
// assert()-only header when built with -DNDEBUG (Release): the death test's
// subject returns normally instead of dying, since assert() compiles out.
TEST(Rational, GcdRationalThrowsOnUnrepresentableGcd) {
  constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
  EXPECT_THROW((void)gcd_rational(Rational(kMin, 1), Rational(kMin, 1)),
               std::logic_error);
}

// The ordinary, representable case must keep working: a gcd that is large
// (2^62, still comfortably within int64_t) must not spuriously trip the new
// guard.
TEST(Rational, GcdRationalHandlesLargeButRepresentableGcd) {
  constexpr std::int64_t kHalfRange = 4'611'686'018'427'387'904LL;  // 2^62
  EXPECT_EQ(gcd_rational(Rational(kHalfRange, 1), Rational(kHalfRange, 1)),
            Rational(kHalfRange, 1));
}

// The DENOMINATOR lcm is the other half of the same contract, and it was the
// half that was missing: std::lcm's result is undefined when it does not fit
// the common type. Measured against the pre-fix header, these two coprime
// denominators (each ~3.037e9, exact lcm 9223372037000249951, int64_t max
// 9223372036854775807) made std::lcm wrap to a negative int64_t, which
// Rational's constructor normalised into num=-1, den=9223372036709301665 --
// a NEGATIVE rational returned as the gcd of two positive ones, with no
// diagnostic. ClockSet::finest_interval() calls gcd_rational in a fold, so
// that negative interval would have propagated straight into §8.3 slab
// decomposition. Unrepresentable must be an error, not a wrapped answer.
TEST(Rational, GcdRationalThrowsOnUnrepresentableDenominatorLcm) {
  EXPECT_THROW(
      (void)gcd_rational(Rational(1, 3'037'000'493LL), Rational(1, 3'037'000'507LL)),
      std::logic_error);
}

// The companion positive case: large coprime denominators whose lcm still
// fits must produce the exact lcm, not a spurious rejection.
TEST(Rational, GcdRationalIsExactForLargeRepresentableDenominatorLcm) {
  const Rational g = gcd_rational(Rational(1, 2'147'483'647LL),   // 2^31 - 1
                                  Rational(1, 2'147'483'629LL));  // prime
  EXPECT_EQ(g.num(), 1);
  EXPECT_EQ(g.den(), 2'147'483'647LL * 2'147'483'629LL);  // 4.612e18 < int64max
  EXPECT_GT(g.den(), 0);
}

// A zero denominator makes int64_t division UB downstream; the constructor
// must reject it unconditionally (not just via assert(), which -DNDEBUG
// compiles out).
TEST(Rational, ZeroDenominatorConstructorThrows) {
  EXPECT_THROW((void)Rational(1, 0), std::logic_error);
}

// Same reasoning as the constructor guard: dividing by a zero Rational is
// UB downstream (o.num_ used as an int64_t divisor via the cross-multiply),
// and must throw regardless of build configuration.
TEST(Rational, DivisionByZeroRationalThrows) {
  Rational a{1, 2};
  EXPECT_THROW(a /= Rational(0), std::logic_error);
}
