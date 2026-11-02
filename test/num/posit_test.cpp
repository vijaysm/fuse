// fuse::num::posit<16, 2> and posit<32, 2> against the definition of the format
// (plans/2026-10-22-universal-number-types.md, stage U5). Nothing here uses the type's own
// decoding or rounding as its reference: `posit_ref_value` (test/num/posit_reference.hpp) decodes a pattern bit by
// bit from the definition (any width, so it also gives the boundary between two adjacent posits,
// which is the value of the one-bit-wider pattern with a one appended), and the rounding of
// a real number is found by locating it among those boundaries with exact arithmetic
// (test/num/exact_expansion.hpp). The bit-for-bit comparison with stillwater-sc/universal
// is test/num/posit_universal_test.cpp.
//
//   * every 16-bit pattern decodes to the value of the definition, and rounds back to itself
//   * at every boundary between adjacent posits, and a binary64 step either side of it,
//     the rounding is the one the definition gives (ties to the even pattern)
//   * random binary64 values round as the definition says; saturation and the special values
//   * + - * / sqrt are the correctly rounded result of the exact operation on the exact
//     values (operands of all kinds for the 16-bit format, random ones for both)
//   * the comparison, negation and absolute value of the patterns
//   * on every enabled execution space the same functions give the same bits

#include "fuse/num/posit.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "exact_expansion.hpp"
#include "posit_reference.hpp"
#include "fuse/num/storage_convert.hpp"
#include "fuse_test_main.hpp"

namespace {

using fuse::num::posit16;
using fuse::num::posit32;
using fuse::test::Expansion;
using fuse::test::posit_ref_value;

static_assert(fuse::CoupledValue<posit16> && fuse::CoupledValue<posit32>);
static_assert(fuse::value_traits<posit16>::kind == fuse::FieldValueKind::Storage);
static_assert(fuse::value_traits<posit32>::kind == fuse::FieldValueKind::Storage);

// ---- the definition ----------------------------------------------------------------

template <class P>
constexpr int kN = P::nbits;

template <class P>
std::uint64_t maxpos_bits() {
  return (std::uint64_t{1} << (kN<P> - 1)) - 1;
}

/// The boundary between the positive patterns u and u + 1: the value of the (n + 1)-bit pattern
/// u followed by a one.
template <class P>
double boundary(std::uint64_t u) {
  return posit_ref_value(kN<P> + 1, 2, 2 * u + 1);
}

int sign_of(const Expansion& e) {
  for (std::size_t i = e.size(); i-- > 0;) {
    if (e[i] > 0.0) return 1;
    if (e[i] < 0.0) return -1;
  }
  return 0;
}

/// The magnitude pattern the definition rounds an exact real value to. `approx` is a binary64
/// near the value (only used to find the neighbourhood), `compare(t)` is the sign of
/// |value| - t, computed exactly.
template <class P, class Cmp>
std::uint64_t ref_round_magnitude(double approx, Cmp compare) {
  const std::uint64_t maxpos = maxpos_bits<P>();
  // the largest pattern whose value is at most the approximation
  std::uint64_t lo = 0, hi = maxpos;
  while (lo < hi) {
    const std::uint64_t mid = lo + (hi - lo + 1) / 2;
    if (posit_ref_value(kN<P>, 2, mid) <= approx) lo = mid; else hi = mid - 1;
  }
  // `above(u)`: the value is past the boundary above u, or on it with u odd (it ties to u + 1)
  const auto above = [&](std::uint64_t u) {
    const int c = compare(boundary<P>(u));
    return c > 0 || (c == 0 && (u & 1u) != 0);
  };
  std::uint64_t r = lo > 2 ? lo - 2 : 1;
  if (r > 1) EXPECT_TRUE(above(r - 1)) << "neighbourhood too narrow";
  while (r < maxpos && above(r)) ++r;
  return r;
}

template <class P>
typename P::bits_type pattern_of_sign(bool negative, std::uint64_t magnitude) {
  const std::uint64_t mask = (std::uint64_t{1} << kN<P>) - 1;
  return static_cast<typename P::bits_type>(negative ? ((~magnitude + 1) & mask) : magnitude);
}

/// The posit the definition gives for the exact value whose sign is `s` and for which
/// `compare_abs(t)` is the sign of |value| - t.
template <class P, class Cmp>
P ref_round(int s, double approx, Cmp compare_abs) {
  if (s == 0) return P::zero();
  const std::uint64_t m = ref_round_magnitude<P>(std::fabs(approx), compare_abs);
  return P::from_bits(pattern_of_sign<P>(s < 0, m));
}

/// The definition's rounding of one binary64 value.
template <class P>
P ref_from_double(double x) {
  if (std::isnan(x) || std::isinf(x)) return P::nar();
  if (x == 0.0) return P::zero();
  const double a = std::fabs(x);
  return ref_round<P>(x < 0 ? -1 : 1, x, [&](double t) { return a < t ? -1 : (a > t ? 1 : 0); });
}

template <class P>
std::uint64_t bits_of(P p) {
  return static_cast<std::uint64_t>(p.bits());
}

template <class P>
std::vector<P> sample_patterns(std::uint64_t seed, std::size_t count) {
  std::mt19937_64 rng(seed);
  std::vector<P> v;
  const std::uint64_t mask = (std::uint64_t{1} << kN<P>) - 1;
  for (std::size_t i = 0; i < count; ++i) v.push_back(P::from_bits(static_cast<typename P::bits_type>(rng() & mask)));
  // the extremes and the neighbourhood of one
  for (std::uint64_t u : {0ull, 1ull, 2ull, 3ull, 4ull}) {
    v.push_back(P::from_bits(static_cast<typename P::bits_type>(u)));
    v.push_back(P::from_bits(static_cast<typename P::bits_type>(maxpos_bits<P>() - u)));
    v.push_back(P::from_bits(static_cast<typename P::bits_type>((mask + 1 - u) & mask)));
  }
  const std::uint64_t one = std::uint64_t{1} << (kN<P> - 2) ;  // the pattern 0100...0 is the value one
  for (std::int64_t d = -3; d <= 3; ++d) {
    v.push_back(P::from_bits(static_cast<typename P::bits_type>(one + d)));
    v.push_back(P::from_bits(static_cast<typename P::bits_type>((mask + 1 - (one + d)) & mask)));
  }
  v.push_back(P::nar());
  return v;
}

// ---- decoding ------------------------------------------------------------------------

TEST(Posit, ConstantsAndSizes) {
  EXPECT_EQ(sizeof(posit16), 2u);
  EXPECT_EQ(sizeof(posit32), 4u);
  EXPECT_EQ(posit16::maxpos().to_double(), std::ldexp(1.0, 56));
  EXPECT_EQ(posit16::minpos().to_double(), std::ldexp(1.0, -56));
  EXPECT_EQ(posit32::maxpos().to_double(), std::ldexp(1.0, 120));
  EXPECT_EQ(posit32::minpos().to_double(), std::ldexp(1.0, -120));
  EXPECT_EQ(posit16::nar().bits(), 0x8000u);
  EXPECT_EQ(posit32::nar().bits(), 0x80000000u);
  EXPECT_TRUE(posit16::nar().is_nar());
  EXPECT_TRUE(std::isnan(posit16::nar().to_double()));
  EXPECT_EQ(posit16::zero().to_double(), 0.0);
  EXPECT_EQ(posit16::from_double(1.0).bits(), 0x4000u);
  EXPECT_EQ(posit32::from_double(1.0).bits(), 0x40000000u);
  EXPECT_EQ(posit16::from_double(-1.0).bits(), 0xC000u);
  EXPECT_EQ(posit16::from_double(0.5).bits(), 0x3800u);   // k = -1, e = 3
  EXPECT_EQ(posit16::from_double(2.0).bits(), 0x4800u);   // k = 0, e = 1
  EXPECT_EQ(posit16::from_double(16.0).bits(), 0x6000u);  // k = 1, e = 0
  EXPECT_EQ(fuse::value_traits<posit16>::name, "posit16");
  EXPECT_EQ(fuse::value_traits<posit32>::name, "posit32");
  EXPECT_EQ(fuse::ElementTypeOf<posit16>::value, fuse::ElementType::Posit16);
  EXPECT_EQ(fuse::ElementTypeOf<posit32>::value, fuse::ElementType::Posit32);
}

TEST(Posit, EveryPosit16PatternDecodesLikeTheDefinitionAndRoundsBackToItself) {
  double previous = -std::numeric_limits<double>::infinity();
  for (std::uint32_t p = 0; p <= 0xFFFFu; ++p) {
    const posit16 v = posit16::from_bits(static_cast<std::uint16_t>(p));
    const double want = posit_ref_value(16, 2, p);
    const double got = v.to_double();
    if (v.is_nar()) {
      ASSERT_TRUE(std::isnan(got));
      ASSERT_EQ(posit16::from_double(got).bits(), 0x8000u);
      continue;
    }
    ASSERT_EQ(got, want) << "pattern " << p;
    ASSERT_EQ(posit16::from_double(got).bits(), p) << "pattern " << p;
    // the patterns, as signed integers, are in the order of the values
    if (p != 0x8001u) {  // the step over NaR
      ASSERT_GT(got, previous) << "pattern " << p;
    }
    previous = got;
  }
}

TEST(Posit, ASampleOfPosit32PatternsDecodesLikeTheDefinitionAndRoundsBackToItself) {
  for (const posit32 v : sample_patterns<posit32>(11, 300000)) {
    if (v.is_nar()) {
      ASSERT_TRUE(std::isnan(v.to_double()));
      continue;
    }
    const double got = v.to_double();
    ASSERT_EQ(got, posit_ref_value(32, 2, v.bits())) << v.bits();
    ASSERT_EQ(posit32::from_double(got).bits(), v.bits()) << v.bits();
  }
}

// ---- rounding --------------------------------------------------------------------------

template <class P>
void check_boundary(std::uint64_t u) {
  const double t = boundary<P>(u);
  const std::uint64_t even = (u & 1u) ? u + 1 : u;
  ASSERT_EQ(bits_of(P::from_double(t)), even) << "tie at the boundary above " << u;
  ASSERT_EQ(bits_of(P::from_double(std::nextafter(t, 0.0))), u) << "just below the boundary above " << u;
  ASSERT_EQ(bits_of(P::from_double(std::nextafter(t, std::numeric_limits<double>::infinity()))), u + 1)
      << "just above the boundary above " << u;
  ASSERT_EQ(bits_of(P::from_double(posit_ref_value(kN<P>, 2, u))), u);
  // a negative value is the two's complement of the magnitude's pattern
  ASSERT_EQ(bits_of(P::from_double(-t)), static_cast<std::uint64_t>(pattern_of_sign<P>(true, even)));
  ASSERT_EQ(bits_of(P::from_double(-std::nextafter(t, 0.0))),
            static_cast<std::uint64_t>(pattern_of_sign<P>(true, u)));
}

TEST(Posit, Posit16RoundsAtEveryBoundaryAsTheDefinitionSays) {
  for (std::uint64_t u = 1; u < maxpos_bits<posit16>(); ++u) check_boundary<posit16>(u);
}

TEST(Posit, Posit32RoundsAtTheBoundariesOfASampleAsTheDefinitionSays) {
  std::vector<std::uint64_t> us;
  for (std::uint64_t u = 1; u < 20000; ++u) us.push_back(u);
  for (std::uint64_t u = maxpos_bits<posit32>() - 20000; u < maxpos_bits<posit32>(); ++u) us.push_back(u);
  std::mt19937_64 rng(5);
  for (int i = 0; i < 200000; ++i) us.push_back(1 + rng() % (maxpos_bits<posit32>() - 1));
  for (const std::uint64_t u : us) check_boundary<posit32>(u);
}

template <class P>
void check_random_doubles(std::uint64_t seed, int count) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> exponent(-(P::max_scale + 6.0), P::max_scale + 6.0);
  for (int i = 0; i < count; ++i) {
    const double mant = 1.0 + std::ldexp(static_cast<double>(rng() >> 11), -53);
    double x = std::ldexp(mant, static_cast<int>(std::floor(exponent(rng))));
    if (rng() & 1u) x = -x;
    ASSERT_EQ(bits_of(P::from_double(x)), bits_of(ref_from_double<P>(x))) << std::hexfloat << x;
  }
}

TEST(Posit, RandomBinary64ValuesRoundAsTheDefinitionSays) {
  check_random_doubles<posit16>(1, 30000);
  check_random_doubles<posit32>(2, 20000);
}

template <class P>
void check_saturation_and_specials() {
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(P::from_double(0.0).bits(), 0u);
  EXPECT_EQ(P::from_double(-0.0).bits(), 0u);
  EXPECT_TRUE(P::from_double(inf).is_nar());
  EXPECT_TRUE(P::from_double(-inf).is_nar());
  EXPECT_TRUE(P::from_double(std::numeric_limits<double>::quiet_NaN()).is_nar());
  const double maxpos = P::maxpos().to_double(), minpos = P::minpos().to_double();
  // a finite value never becomes NaR or zero: it saturates
  for (const double x : {maxpos, maxpos * 1.5, maxpos * 1e10, std::numeric_limits<double>::max(), 1e300}) {
    EXPECT_EQ(P::from_double(x).bits(), P::maxpos().bits()) << x;
    EXPECT_EQ(P::from_double(-x).bits(), (-P::maxpos()).bits()) << x;
  }
  for (const double x : {minpos, minpos * 0.5, minpos * 1e-10, 1e-300, std::numeric_limits<double>::min(),
                         std::numeric_limits<double>::denorm_min()}) {
    EXPECT_EQ(P::from_double(x).bits(), P::minpos().bits()) << x;
    EXPECT_EQ(P::from_double(-x).bits(), (-P::minpos()).bits()) << x;
  }
}

TEST(Posit, FiniteValuesSaturateAndTheOtherSpecialValuesAreNaROrZero) {
  check_saturation_and_specials<posit16>();
  check_saturation_and_specials<posit32>();
}

TEST(Posit, ADirectedTieGoesTheWayTheErrorSays) {
  for (std::uint64_t u = 1; u < maxpos_bits<posit16>(); u += 7) {
    const double t = boundary<posit16>(u);
    EXPECT_EQ(posit16::round_directed(t, 1).bits(), u + 1);
    EXPECT_EQ(posit16::round_directed(t, -1).bits(), u);
    EXPECT_EQ(posit16::round_directed(-t, 1).bits(), (-posit16::from_bits(static_cast<std::uint16_t>(u))).bits());
    EXPECT_EQ(posit16::round_directed(-t, -1).bits(), (-posit16::from_bits(static_cast<std::uint16_t>(u + 1))).bits());
    // not a tie: the direction does not matter
    const double below = std::nextafter(t, 0.0);
    EXPECT_EQ(posit16::round_directed(below, 1).bits(), u);
  }
}

// ---- order, negation, absolute value -------------------------------------------------------

TEST(Posit, OrderNegationAndAbsoluteValueAreThoseOfThePatterns) {
  for (std::uint32_t p = 0; p <= 0xFFFFu; p += 3) {
    const posit16 a = posit16::from_bits(static_cast<std::uint16_t>(p));
    const posit16 neg = -a;
    EXPECT_EQ(neg.bits(), static_cast<std::uint16_t>(0u - p)) << p;
    if (!a.is_nar()) {
      EXPECT_EQ(neg.to_double(), -a.to_double());
      EXPECT_EQ(abs(a).to_double(), std::fabs(a.to_double()));
    } else {
      EXPECT_TRUE(neg.is_nar());
    }
    const posit16 b = posit16::from_bits(static_cast<std::uint16_t>(p * 2654435761u >> 7));
    if (!a.is_nar() && !b.is_nar()) {
      EXPECT_EQ(a < b, a.to_double() < b.to_double());
      EXPECT_EQ(a <= b, a.to_double() <= b.to_double());
      EXPECT_EQ(a > b, a.to_double() > b.to_double());
      EXPECT_EQ(a == b, a.to_double() == b.to_double());
    }
  }
  // NaR equals itself and is below every real value
  EXPECT_TRUE(posit16::nar() == posit16::nar());
  EXPECT_TRUE(posit16::nar() < -posit16::maxpos());
  EXPECT_TRUE(posit32::nar() < -posit32::maxpos());
}

// ---- arithmetic ----------------------------------------------------------------------------

enum class Op { Add, Sub, Mul, Div, Sqrt };

/// The posit the definition gives for the exact result of `op` on the exact values.
template <class P>
P ref_op(Op op, P a, P b) {
  if (a.is_nar() || (op != Op::Sqrt && b.is_nar())) return P::nar();
  const double x = a.to_double(), y = b.to_double();
  const Expansion X{x}, Y{y};
  switch (op) {
    case Op::Add:
    case Op::Sub: {
      const Expansion v = op == Op::Add ? fuse::test::add(X, Y) : fuse::test::sub(X, Y);
      const int s = sign_of(v);
      return ref_round<P>(s, fuse::test::approx(v), [&](double t) {
        const int c = sign_of(fuse::test::sub(v, Expansion{s * t}));
        return s * c;
      });
    }
    case Op::Mul: {
      const Expansion v = fuse::test::mul(X, Y);
      const int s = sign_of(v);
      return ref_round<P>(s, fuse::test::approx(v), [&](double t) {
        return s * sign_of(fuse::test::sub(v, Expansion{s * t}));
      });
    }
    case Op::Div: {
      if (y == 0.0) return P::nar();
      if (x == 0.0) return P::zero();
      const int s = (x < 0 ? -1 : 1) * (y < 0 ? -1 : 1);
      return ref_round<P>(s, x / y, [&](double t) {
        // |x / y| versus t: |x| - t |y|, since y is not zero
        return sign_of(fuse::test::sub(Expansion{std::fabs(x)}, fuse::test::mul(Expansion{t}, Expansion{std::fabs(y)})));
      });
    }
    case Op::Sqrt: {
      if (x < 0.0) return P::nar();
      if (x == 0.0) return P::zero();
      return ref_round<P>(1, std::sqrt(x), [&](double t) {
        // sqrt(x) versus t: x versus t^2
        return sign_of(fuse::test::sub(Expansion{x}, fuse::test::mul(Expansion{t}, Expansion{t})));
      });
    }
  }
  return P::nar();
}

template <class P>
P apply_op(Op op, P a, P b) {
  switch (op) {
    case Op::Add: return a + b;
    case Op::Sub: return a - b;
    case Op::Mul: return a * b;
    case Op::Div: return a / b;
    case Op::Sqrt: return sqrt(a);
  }
  return P::nar();
}

const char* op_name(Op op) {
  switch (op) {
    case Op::Add: return "+";
    case Op::Sub: return "-";
    case Op::Mul: return "*";
    case Op::Div: return "/";
    case Op::Sqrt: return "sqrt";
  }
  return "?";
}

template <class P>
void check_op(Op op, P a, P b) {
  const P got = apply_op(op, a, b);
  const P want = ref_op(op, a, b);
  ASSERT_EQ(got.bits(), want.bits()) << std::hexfloat << a.to_double() << ' ' << op_name(op) << ' ' << b.to_double()
                                     << " (patterns " << std::hex << a.bits() << ", " << b.bits() << ")";
}

template <class P>
std::vector<P> operands(std::uint64_t seed, std::size_t random_count, int exponent_step) {
  std::vector<P> v = sample_patterns<P>(seed, random_count);
  // exact powers of two and the values next to them, in both signs
  for (int e = -P::max_scale; e <= P::max_scale; e += exponent_step) {
    const P p = P::from_double(std::ldexp(1.0, e));
    for (int d = -1; d <= 1; ++d) {
      const P q = P::from_bits(static_cast<typename P::bits_type>(p.bits() + d));
      v.push_back(q);
      v.push_back(-q);
    }
  }
  return v;
}

TEST(Posit, Posit16ArithmeticIsCorrectlyRoundedOnEveryKindOfOperand) {
  const std::vector<posit16> v = operands<posit16>(21, 40, 7);
  for (const Op op : {Op::Add, Op::Sub, Op::Mul, Op::Div}) {
    for (const posit16 a : v) {
      for (const posit16 b : v) check_op(op, a, b);
    }
  }
  for (const posit16 a : operands<posit16>(22, 1000, 3)) check_op(Op::Sqrt, a, a);
  std::mt19937_64 rng(23);
  for (int i = 0; i < 6000; ++i) {
    const posit16 a = posit16::from_bits(static_cast<std::uint16_t>(rng()));
    const posit16 b = posit16::from_bits(static_cast<std::uint16_t>(rng()));
    for (const Op op : {Op::Add, Op::Mul, Op::Div}) check_op(op, a, b);
  }
}

TEST(Posit, Posit32ArithmeticIsCorrectlyRoundedOnRandomOperands) {
  const std::vector<posit32> v = operands<posit32>(31, 10, 13);
  for (const Op op : {Op::Add, Op::Sub, Op::Mul, Op::Div}) {
    for (const posit32 a : v) {
      for (const posit32 b : v) check_op(op, a, b);
    }
  }
  for (const posit32 a : operands<posit32>(32, 600, 5)) check_op(Op::Sqrt, a, a);
  std::mt19937_64 rng(33);
  for (int i = 0; i < 2400; ++i) {
    // operands of similar size (cancellation, products and quotients near one) and of any size
    const posit32 a = posit32::from_bits(static_cast<std::uint32_t>(rng()));
    posit32 b = posit32::from_bits(static_cast<std::uint32_t>(rng()));
    if (i % 3 == 0) b = posit32::from_bits(static_cast<std::uint32_t>(a.bits() + (rng() % 5)));
    if (i % 3 == 1) b = -posit32::from_bits(static_cast<std::uint32_t>(a.bits() + (rng() % 5)));
    for (const Op op : {Op::Add, Op::Mul, Op::Div}) check_op(op, a, b);
  }
}

/// A positive posit of the scale `scale` with random fraction bits (as many as the format has there).
template <class P>
P random_posit_of_scale(std::mt19937_64& rng, int scale) {
  const double mant = 1.0 + std::ldexp(static_cast<double>(rng() >> 11), -53);
  return P::from_double(std::ldexp(mant, scale));
}

/// The sums of two posits whose scales are `distance_lo` to `distance_hi` binades apart, at
/// every second scale of the larger: where binary64 cannot hold the exact sum (a posit32 with its
/// 27 fraction bits needs 26 binades) and just before.
template <class P>
void check_sums_of_operands_apart_in_scale(std::uint64_t seed, int distance_lo, int distance_hi) {
  std::mt19937_64 rng(seed);
  for (int sa = -P::max_scale; sa < P::max_scale; sa += 2) {
    for (int d = distance_lo; d <= distance_hi; ++d) {
      const int sb = sa - d;
      if (sb < -P::max_scale) break;
      const P a = random_posit_of_scale<P>(rng, sa);
      const P b = random_posit_of_scale<P>(rng, sb);
      check_op(Op::Add, a, b);
      check_op(Op::Add, -b, a);
    }
  }
}

/// The sum of a posit `a` and a posit next to the distance from `a` to the boundary above it:
/// the exact sum lands on the boundary or a binary64 step beyond or short of it.
template <class P>
void check_sums_next_to_a_boundary(const std::vector<std::uint64_t>& us) {
  for (const std::uint64_t u : us) {
    const P a = P::from_bits(static_cast<typename P::bits_type>(u));
    const double half = boundary<P>(u) - a.to_double();  // exact: the distance to the boundary above
    const P b = P::from_double(half);
    for (int d = -2; d <= 2; ++d) {
      const P c = P::from_bits(static_cast<typename P::bits_type>(b.bits() + d));
      check_op(Op::Add, a, c);
      check_op(Op::Add, -a, -c);
    }
  }
}

TEST(Posit, SumsOfOperandsAtEveryDistanceInScaleAreCorrectlyRounded) {
  check_sums_of_operands_apart_in_scale<posit16>(41, 30, 64);
  check_sums_of_operands_apart_in_scale<posit32>(42, 20, 64);
}

TEST(Posit, SumsThatLandOnOrNextToABoundaryAreCorrectlyRounded) {
  std::vector<std::uint64_t> us16, us32;
  for (std::uint64_t u = 1; u < maxpos_bits<posit16>(); u += 97) us16.push_back(u);
  std::mt19937_64 rng(43);
  for (int i = 0; i < 1500; ++i) us32.push_back(1 + rng() % (maxpos_bits<posit32>() - 1));
  check_sums_next_to_a_boundary<posit16>(us16);
  check_sums_next_to_a_boundary<posit32>(us32);
}

TEST(Posit, ArithmeticOnNaRAndZeroFollowsTheStandard) {
  const posit16 one = posit16::from_double(1.0), zero = posit16::zero(), nar = posit16::nar();
  EXPECT_TRUE((nar + one).is_nar());
  EXPECT_TRUE((one * nar).is_nar());
  EXPECT_TRUE((one / zero).is_nar());
  EXPECT_TRUE((zero / zero).is_nar());
  EXPECT_EQ((zero / one).bits(), 0u);
  EXPECT_EQ((zero * posit16::maxpos()).bits(), 0u);
  EXPECT_TRUE(sqrt(-one).is_nar());
  EXPECT_TRUE(sqrt(nar).is_nar());
  EXPECT_EQ(sqrt(zero).bits(), 0u);
  EXPECT_EQ((one - one).bits(), 0u);
  // saturation, not overflow, and no underflow to zero
  EXPECT_EQ((posit16::maxpos() * posit16::maxpos()).bits(), posit16::maxpos().bits());
  EXPECT_EQ((posit16::minpos() * posit16::minpos()).bits(), posit16::minpos().bits());
  EXPECT_EQ((posit16::minpos() / posit16::maxpos()).bits(), posit16::minpos().bits());
}

// ---- the same bits on every execution space ------------------------------------------------

template <class Space, class P>
struct ToDoubleFunctor {
  Kokkos::View<const P*, typename Space::memory_space> in;
  Kokkos::View<double*, typename Space::memory_space> out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const { out(i) = in(i).to_double(); }
};

template <class Space, class P>
struct FromDoubleFunctor {
  Kokkos::View<const double*, typename Space::memory_space> in;
  Kokkos::View<P*, typename Space::memory_space> out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const { out(i) = P::from_double(in(i)); }
};

template <class Space, class P>
struct OpFunctor {
  int op;
  Kokkos::View<const P*, typename Space::memory_space> a, b;
  Kokkos::View<P*, typename Space::memory_space> out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    switch (op) {
      case 0: out(i) = a(i) + b(i); break;
      case 1: out(i) = a(i) - b(i); break;
      case 2: out(i) = a(i) * b(i); break;
      case 3: out(i) = a(i) / b(i); break;
      default: out(i) = sqrt(a(i)); break;
    }
  }
};

template <class Space>
class PositOnSpace : public ::testing::Test {
 protected:
  using Mem = typename Space::memory_space;
  template <class T>
  using Vec = Kokkos::View<T*, Mem>;

  template <class T>
  static Vec<T> upload(const std::vector<T>& v) {
    Vec<T> d(Kokkos::view_alloc(std::string("u"), Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }
  template <class T>
  static std::vector<T> download(const Vec<T>& d) {
    auto h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(h, d);
    std::vector<T> v(d.extent(0));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = h(i);
    return v;
  }

  template <class P>
  static void expect_decode_matches(const std::vector<P>& patterns) {
    auto din = upload(patterns);
    Vec<double> dout(Kokkos::view_alloc(std::string("d"), Kokkos::WithoutInitializing), patterns.size());
    Kokkos::parallel_for("posit_decode", Kokkos::RangePolicy<Space>(0, patterns.size()),
                         ToDoubleFunctor<Space, P>{din, dout});
    Kokkos::fence();
    const auto got = download(dout);
    for (std::size_t i = 0; i < patterns.size(); ++i) {
      const double want = patterns[i].to_double();
      if (std::isnan(want)) {
        ASSERT_TRUE(std::isnan(got[i])) << i;
      } else {
        ASSERT_EQ(std::bit_cast<std::uint64_t>(got[i]), std::bit_cast<std::uint64_t>(want)) << i;
      }
    }
  }

  template <class P>
  static void expect_encode_matches(const std::vector<double>& inputs) {
    auto din = upload(inputs);
    Vec<P> dout(Kokkos::view_alloc(std::string("e"), Kokkos::WithoutInitializing), inputs.size());
    Kokkos::parallel_for("posit_encode", Kokkos::RangePolicy<Space>(0, inputs.size()),
                         FromDoubleFunctor<Space, P>{din, dout});
    Kokkos::fence();
    const auto got = download(dout);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      ASSERT_EQ(got[i].bits(), P::from_double(inputs[i]).bits()) << std::hexfloat << inputs[i];
    }
    // through the generic Storage kernels as well
    Vec<P> dout2(Kokkos::view_alloc(std::string("n"), Kokkos::WithoutInitializing), inputs.size());
    fuse::num::narrow<Space, P>(fuse::num::ConstSpanOf<Space>(din.data(), din.extent(0)),
                                fuse::num::StorageSpanOf<Space, P>(dout2.data(), dout2.extent(0)));
    Kokkos::fence();
    const auto got2 = download(dout2);
    for (std::size_t i = 0; i < inputs.size(); ++i) ASSERT_EQ(got2[i].bits(), got[i].bits()) << i;
  }

  template <class P>
  static void expect_ops_match(const std::vector<P>& a, const std::vector<P>& b) {
    auto da = upload(a);
    auto db = upload(b);
    for (int op = 0; op < 5; ++op) {
      Vec<P> dout(Kokkos::view_alloc(std::string("o"), Kokkos::WithoutInitializing), a.size());
      Kokkos::parallel_for("posit_op", Kokkos::RangePolicy<Space>(0, a.size()),
                           OpFunctor<Space, P>{op, da, db, dout});
      Kokkos::fence();
      const auto got = download(dout);
      for (std::size_t i = 0; i < a.size(); ++i) {
        const P want = apply_op(static_cast<Op>(op), a[i], b[i]);
        ASSERT_EQ(got[i].bits(), want.bits()) << "op " << op << " pair " << i;
      }
    }
  }
};

}  // namespace

FUSE_TYPED_TEST_ON_SPACES(PositOnSpace);

TYPED_TEST(PositOnSpace, EveryPosit16PatternDecodesAsOnTheHost) {
  std::vector<posit16> p;
  for (std::uint32_t v = 0; v <= 0xFFFFu; ++v) p.push_back(posit16::from_bits(static_cast<std::uint16_t>(v)));
  TestFixture::template expect_decode_matches<posit16>(p);
}

TYPED_TEST(PositOnSpace, ASampleOfPosit32PatternsDecodesAsOnTheHost) {
  TestFixture::template expect_decode_matches<posit32>(sample_patterns<posit32>(41, 300000));
}

template <class P>
std::vector<double> encoding_inputs(std::uint64_t seed) {
  std::vector<double> in;
  for (std::uint64_t u = 1; u < maxpos_bits<P>(); u += (kN<P> == 16 ? 1 : 49999)) {
    const double t = boundary<P>(u);
    for (const double x : {t, std::nextafter(t, 0.0), std::nextafter(t, 1e300), posit_ref_value(kN<P>, 2, u)}) {
      in.push_back(x);
      in.push_back(-x);
    }
  }
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> e(-(P::max_scale + 6.0), P::max_scale + 6.0);
  for (int i = 0; i < 50000; ++i) {
    const double x = std::exp2(e(rng)) * (1.0 + std::ldexp(static_cast<double>(rng() >> 11), -53));
    in.push_back(rng() & 1u ? -x : x);
  }
  for (const double x : {0.0, -0.0, std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min()}) {
    in.push_back(x);
  }
  return in;
}

TYPED_TEST(PositOnSpace, Posit16RoundsOnTheDeviceAsOnTheHostAtEveryBoundary) {
  TestFixture::template expect_encode_matches<posit16>(encoding_inputs<posit16>(51));
}

TYPED_TEST(PositOnSpace, Posit32RoundsOnTheDeviceAsOnTheHost) {
  TestFixture::template expect_encode_matches<posit32>(encoding_inputs<posit32>(52));
}

TYPED_TEST(PositOnSpace, ArithmeticIsTheSameOnTheDeviceAsOnTheHost) {
  std::mt19937_64 rng(61);
  std::vector<posit16> a16, b16;
  std::vector<posit32> a32, b32;
  for (int i = 0; i < 60000; ++i) {
    a16.push_back(posit16::from_bits(static_cast<std::uint16_t>(rng())));
    b16.push_back(posit16::from_bits(static_cast<std::uint16_t>(rng())));
    a32.push_back(posit32::from_bits(static_cast<std::uint32_t>(rng())));
    b32.push_back(posit32::from_bits(static_cast<std::uint32_t>(rng())));
  }
  for (const posit16 s : sample_patterns<posit16>(62, 0)) { a16.push_back(s); b16.push_back(posit16::from_double(3.0)); }
  TestFixture::template expect_ops_match<posit16>(a16, b16);
  TestFixture::template expect_ops_match<posit32>(a32, b32);
}

TYPED_TEST(PositOnSpace, TheRoundingReportCountsAClampedValueAsAnOverflowOrAnUnderflow) {
  std::vector<double> in;
  std::mt19937_64 rng(71);
  std::uniform_real_distribution<double> e(-9.0, 9.0);  // within the range of posit16 (2^-56 .. 2^56)
  for (int i = 0; i < 20000; ++i) in.push_back(std::exp2(e(rng)));
  in.push_back(std::ldexp(1.0, 57));    // above maxpos: clamped
  in.push_back(-std::ldexp(1.5, 60));   // clamped
  in.push_back(1e300);                  // clamped
  in.push_back(std::ldexp(1.0, -57));   // below minpos: clamped
  in.push_back(1e-300);                 // clamped
  in.push_back(posit16::maxpos().to_double());  // exactly maxpos: not clamped
  in.push_back(posit16::minpos().to_double());  // exactly minpos: not clamped
  in.push_back(0.0);                    // exact
  in.push_back(std::numeric_limits<double>::infinity());  // not finite: not counted
  auto din = TestFixture::upload(in);
  typename TestFixture::template Vec<posit16> dout(
      Kokkos::view_alloc(std::string("r"), Kokkos::WithoutInitializing), in.size());
  const auto r = fuse::num::narrow_measured<TypeParam, posit16>(
      fuse::num::ConstSpanOf<TypeParam>(din.data(), din.extent(0)),
      fuse::num::StorageSpanOf<TypeParam, posit16>(dout.data(), dout.extent(0)));
  EXPECT_EQ(r.values, static_cast<std::int64_t>(in.size()) - 1);
  EXPECT_EQ(r.overflow, 3);
  EXPECT_EQ(r.underflow, 2);
  // values within 2^-9 .. 2^9 have at least 11 - 2 fraction bits, so change by at most 2^-10
  EXPECT_GT(r.max_relative_change, 0.0);
  EXPECT_LE(r.max_relative_change, std::ldexp(1.0, -10));
}
