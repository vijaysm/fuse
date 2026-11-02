// fuse::num::posit<16, 2> and posit<32, 2> against stillwater-sc/universal's
// posit<16, 2> and posit<32, 2> (plans/2026-10-22-universal-number-types.md, stage U5).
// Universal is a second implementation of the same format, measured against, host only
// and never included by FUSE itself; as for the double-double comparison, its headers
// are compiled into a separate program (test/num/universal_oracle.cpp) that this test
// runs on files of operands. The test exists only when the build has
// FUSE_ENABLE_UNIVERSAL=ON and FUSE_UNIVERSAL_INCLUDE_DIR pointing at the include/sw
// directory of a checkout.
//
// Operands and results travel as bit patterns, and every result is compared pattern for
// pattern: the decoding of every 16-bit pattern and of a sample of 32-bit patterns, the
// rounding of binary64 values at and around the boundaries between adjacent posits and
// of random values (saturation included), + - * / and the square root, and the special
// values (zero, NaR). The correctness of the rounding itself does not depend on this
// comparison: posit_test.cpp checks it against the definition.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "fuse/num/posit.hpp"
#include "fuse_test_main.hpp"
#include "posit_reference.hpp"
#include "universal_oracle_client.hpp"

namespace {

using fuse::num::posit16;
using fuse::num::posit32;
using fuse::test::posit_boundary;
using fuse::test::posit_ref_value;
using fuse_test::universal::oracle_results;
using fuse_test::universal::Record;
using fuse_test::universal::Result;

enum Op : std::uint32_t { Add, Sub, Mul, Div, Sqrt, FromDouble = 9, ToDouble = 10 };

const char* op_name(std::uint32_t op) {
  switch (op) {
    case Add: return "+";
    case Sub: return "-";
    case Mul: return "*";
    case Div: return "/";
    case Sqrt: return "sqrt";
    case FromDouble: return "conversion from binary64";
    default: return "conversion to binary64";
  }
}

template <class P>
constexpr std::uint32_t kN = P::nbits;

template <class P>
constexpr std::uint64_t kMask = (std::uint64_t{1} << kN<P>) - 1;

template <class P>
std::uint64_t pattern_of(P p) {
  return static_cast<std::uint64_t>(p.bits());
}

template <class P>
double pattern_value(std::uint64_t pattern) {
  return static_cast<double>(pattern & kMask<P>);
}

template <class P>
P from_pattern(double pattern) {
  return P::from_bits(static_cast<typename P::bits_type>(static_cast<std::uint64_t>(pattern)));
}

template <class P>
Record make(std::uint32_t op, double a, double b = 0.0) {
  return Record{kN<P>, op, {a, 0.0, 0.0, 0.0}, {b, 0.0, 0.0, 0.0}};
}

template <class P>
Record make_patterns(std::uint32_t op, std::uint64_t a, std::uint64_t b = 0) {
  return make<P>(op, pattern_value<P>(a), pattern_value<P>(b));
}

/// What fuse::num computes for a record: a pattern, or the binary64 value of a pattern.
template <class P>
double fuse_result(const Record& r) {
  switch (r.op) {
    case FromDouble: return pattern_value<P>(pattern_of(P::from_double(r.a[0])));
    case ToDouble: return from_pattern<P>(r.a[0]).to_double();
    default: break;
  }
  const P a = from_pattern<P>(r.a[0]), b = from_pattern<P>(r.b[0]);
  P v;
  switch (r.op) {
    case Add: v = a + b; break;
    case Sub: v = a - b; break;
    case Mul: v = a * b; break;
    case Div: v = a / b; break;
    default: v = sqrt(a); break;
  }
  return pattern_value<P>(pattern_of(v));
}

std::string hex(double x) {
  std::ostringstream o;
  o << std::hexfloat << x;
  return o.str();
}

/// Compares every record's fuse::num result with Universal's. A result that Universal
/// reports as NaN (an exception) is the NaR pattern; a conversion to binary64 compares
/// the values, with a NaN matching a NaN. Reports the first differences of each operation.
template <class P>
void expect_same_as_universal(const std::vector<Record>& records) {
  const std::vector<Result> ref = oracle_results(records);
  ASSERT_EQ(ref.size(), records.size());
  const double nar = pattern_value<P>(pattern_of(P::nar()));
  std::size_t differing[11] = {};
  std::size_t shown[11] = {};
  for (std::size_t i = 0; i < records.size(); ++i) {
    const Record& rec = records[i];
    const double mine = fuse_result<P>(rec);
    double theirs = ref[i].r[0];
    if (rec.op != ToDouble && std::isnan(theirs)) theirs = nar;
    const bool same = rec.op == ToDouble ? ((std::isnan(mine) && std::isnan(theirs)) || mine == theirs) : mine == theirs;
    if (same) continue;
    ++differing[rec.op];
    if (shown[rec.op]++ < 3) {
      ADD_FAILURE() << "posit" << kN<P> << " " << op_name(rec.op) << " differs from Universal at record " << i
                    << ": a = " << hex(rec.a[0]) << " b = " << hex(rec.b[0]) << " fuse = " << hex(mine)
                    << " universal = " << hex(theirs);
    }
  }
  for (std::uint32_t op = 0; op < 11; ++op) EXPECT_EQ(differing[op], 0u) << op_name(op) << " on posit" << kN<P>;
}

/// The patterns of a sample: random ones, the extremes, and the neighbourhood of one.
template <class P>
std::vector<std::uint64_t> sample_patterns(std::uint64_t seed, int count) {
  std::mt19937_64 rng(seed);
  std::vector<std::uint64_t> v;
  for (int i = 0; i < count; ++i) v.push_back(rng() & kMask<P>);
  const std::uint64_t maxpos = (std::uint64_t{1} << (kN<P> - 1)) - 1;
  for (std::uint64_t u = 0; u < 6; ++u) {
    v.push_back(u);
    v.push_back(maxpos - u);
    v.push_back((kMask<P> + 1 - u) & kMask<P>);
  }
  const std::uint64_t one = std::uint64_t{1} << (kN<P> - 2);
  for (std::int64_t d = -4; d <= 4; ++d) {
    v.push_back(one + d);
    v.push_back((kMask<P> + 1 - (one + d)) & kMask<P>);
  }
  v.push_back(std::uint64_t{1} << (kN<P> - 1));  // NaR
  return v;
}

template <class P>
std::vector<std::uint64_t> every_pattern() {
  std::vector<std::uint64_t> v;
  for (std::uint64_t u = 0; u <= kMask<P>; ++u) v.push_back(u);
  return v;
}

template <class P>
void decoding_matches(const std::vector<std::uint64_t>& patterns) {
  std::vector<Record> records;
  for (const std::uint64_t u : patterns) records.push_back(make_patterns<P>(ToDouble, u));
  expect_same_as_universal<P>(records);
}

/// Binary64 values at the boundaries between adjacent positive posits (and the exact
/// values, the midpoints of the two values and a binary64 step either side of each),
/// for the magnitude patterns `us`, with both signs.
template <class P>
void conversion_matches(const std::vector<std::uint64_t>& us, int random_count, std::uint64_t seed) {
  const std::uint64_t maxpos = (std::uint64_t{1} << (kN<P> - 1)) - 1;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> xs;
  for (const std::uint64_t u : us) {
    if (u == 0 || u >= maxpos) continue;
    const double t = posit_boundary(kN<P>, 2, u);
    const double v = posit_ref_value(kN<P>, 2, u), w = posit_ref_value(kN<P>, 2, u + 1);
    const double mid = 0.5 * (v + w);
    for (const double x : {t, std::nextafter(t, 0.0), std::nextafter(t, inf), v, w, mid, std::nextafter(mid, 0.0),
                           std::nextafter(mid, inf)}) {
      xs.push_back(x);
      xs.push_back(-x);
    }
  }
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> exponent(-(P::max_scale + 8.0), P::max_scale + 8.0);
  for (int i = 0; i < random_count; ++i) {
    const double mant = 1.0 + std::ldexp(static_cast<double>(rng() >> 11), -53);
    const double x = std::ldexp(mant, static_cast<int>(std::floor(exponent(rng))));
    xs.push_back((rng() & 1u) ? -x : x);
  }
  // the ends of the range, and what is not a finite nonzero number
  const double maxpos_value = posit_ref_value(kN<P>, 2, maxpos);
  for (const double x : {maxpos_value, std::nextafter(maxpos_value, inf), 2.0 * maxpos_value, 1e300, 1e-300,
                         1.0 / maxpos_value, 0.5 / maxpos_value, 0.25 / maxpos_value, 0.0, -0.0,
                         std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max(), inf,
                         std::numeric_limits<double>::quiet_NaN(), 1.0, 0.5, 3.0}) {
    xs.push_back(x);
    xs.push_back(-x);
  }
  std::vector<Record> records;
  for (const double x : xs) records.push_back(make<P>(FromDouble, x));
  expect_same_as_universal<P>(records);
}

template <class P>
void arithmetic_matches(const std::vector<std::uint64_t>& a, const std::vector<std::uint64_t>& b) {
  std::vector<Record> records;
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (const std::uint32_t op : {Add, Sub, Mul, Div}) records.push_back(make_patterns<P>(op, a[i], b[i]));
  }
  expect_same_as_universal<P>(records);
}

/// Universal computes the square root of a posit in binary64 and rounds that to a posit, which
/// is two roundings: when the binary64 root is exactly halfway between two posits and the exact
/// root is not, it takes the even one whichever side the exact root is on. fuse::num::posit
/// rounds the exact root once (posit_test.cpp checks that against exact arithmetic). So the
/// comparison is of two statements: Universal returns the posit nearest the binary64 root, bit for
/// bit, and fuse::num::posit returns the same pattern except at such a tie, where it returns the
/// neighbour on the side the exact root is.
template <class P>
void square_root_matches(const std::vector<std::uint64_t>& a) {
  std::vector<Record> records;
  for (const std::uint64_t u : a) records.push_back(make_patterns<P>(Sqrt, u));
  const std::vector<Result> ref = oracle_results(records);
  ASSERT_EQ(ref.size(), records.size());
  const double nar = pattern_value<P>(pattern_of(P::nar()));
  const double inf = std::numeric_limits<double>::infinity();
  std::size_t differs_from_binary64_root = 0, differs_from_universal_off_a_tie = 0, ties = 0;
  for (std::size_t i = 0; i < records.size(); ++i) {
    const P x = from_pattern<P>(records[i].a[0]);
    const double universal = std::isnan(ref[i].r[0]) ? nar : ref[i].r[0];
    const double mine = pattern_value<P>(pattern_of(sqrt(x)));
    if (x.is_nar() || x.is_negative() || x.is_zero()) {  // no root to round: the rules agree
      EXPECT_EQ(mine, universal) << "record " << i;
      continue;
    }
    const double root = std::sqrt(x.to_double());
    const double twice_rounded = pattern_value<P>(pattern_of(P::from_double(root)));
    if (twice_rounded != universal && differs_from_binary64_root++ < 3) {
      ADD_FAILURE() << "posit" << kN<P> << " sqrt: Universal is not the posit nearest the binary64 root at record " << i
                    << ": a = " << hex(records[i].a[0]) << " universal = " << hex(universal)
                    << " nearest = " << hex(twice_rounded);
    }
    if (mine == universal) continue;
    // the binary64 root is a tie when a step either side of it rounds to different posits
    const bool tie = P::from_double(std::nextafter(root, 0.0)).bits() != P::from_double(std::nextafter(root, inf)).bits();
    if (tie) ++ties;
    if (!tie && differs_from_universal_off_a_tie++ < 3) {
      ADD_FAILURE() << "posit" << kN<P> << " sqrt differs from Universal away from a tie at record " << i
                    << ": a = " << hex(records[i].a[0]) << " fuse = " << hex(mine) << " universal = " << hex(universal);
    }
    EXPECT_EQ(std::fabs(mine - universal), 1.0) << "record " << i;  // the neighbouring pattern
  }
  EXPECT_EQ(differs_from_binary64_root, 0u);
  EXPECT_EQ(differs_from_universal_off_a_tie, 0u);
  ::testing::Test::RecordProperty("sqrt_ties_where_the_two_roundings_differ", static_cast<int>(ties));
}

/// Operand pairs of one run: independent patterns, patterns a few steps apart (the
/// cancellation of a subtraction), and patterns of moderate size (the ordinary range of
/// a field).
template <class P>
std::pair<std::vector<std::uint64_t>, std::vector<std::uint64_t>> operand_pairs(int n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::uint64_t> a, b;
  const std::uint64_t one = std::uint64_t{1} << (kN<P> - 2);
  const std::uint64_t span = std::uint64_t{1} << (kN<P> - 5);  // about 2^-4 to 2^4 around one
  for (int i = 0; i < n; ++i) {
    std::uint64_t x, y;
    switch (i % 3) {
      case 0:
        x = rng() & kMask<P>;
        y = rng() & kMask<P>;
        break;
      case 1:
        x = rng() & kMask<P>;
        y = (x + rng() % 7 - 3) & kMask<P>;
        break;
      default:
        x = one - span + rng() % (2 * span);
        y = one - span + rng() % (2 * span);
        if (rng() & 1u) x = (kMask<P> + 1 - x) & kMask<P>;
        if (rng() & 1u) y = (kMask<P> + 1 - y) & kMask<P>;
        break;
    }
    a.push_back(x);
    b.push_back(y);
  }
  return {a, b};
}

}  // namespace

TEST(PositUniversal, EveryPosit16PatternDecodesAsUniversalDoes) { decoding_matches<posit16>(every_pattern<posit16>()); }

TEST(PositUniversal, ASampleOfPosit32PatternsDecodesAsUniversalDoes) {
  decoding_matches<posit32>(sample_patterns<posit32>(11, 200000));
}

TEST(PositUniversal, Posit16ConvertsFromBinary64AsUniversalDoes) {
  std::vector<std::uint64_t> us;
  for (std::uint64_t u = 1; u < 0x7fff; ++u) us.push_back(u);
  conversion_matches<posit16>(us, 100000, 12);
}

TEST(PositUniversal, Posit32ConvertsFromBinary64AsUniversalDoes) {
  std::vector<std::uint64_t> us;
  for (std::uint64_t u = 1; u < 5000; ++u) us.push_back(u);
  for (std::uint64_t u = 0x7fffffffull - 5000; u < 0x7fffffffull; ++u) us.push_back(u);
  std::mt19937_64 rng(13);
  for (int i = 0; i < 60000; ++i) us.push_back(1 + rng() % 0x7ffffffeull);
  conversion_matches<posit32>(us, 100000, 14);
}

TEST(PositUniversal, Posit16ArithmeticMatchesUniversalPatternForPattern) {
  const auto [a, b] = operand_pairs<posit16>(150000, 21);
  arithmetic_matches<posit16>(a, b);
}

TEST(PositUniversal, Posit32ArithmeticMatchesUniversalPatternForPattern) {
  const auto [a, b] = operand_pairs<posit32>(60000, 22);
  arithmetic_matches<posit32>(a, b);
}

TEST(PositUniversal, SquareRootMatchesUniversalPatternForPattern) {
  std::vector<std::uint64_t> a16;
  for (std::uint64_t u = 0; u <= kMask<posit16>; ++u) a16.push_back(u);  // the negative ones too: NaR
  square_root_matches<posit16>(a16);
  square_root_matches<posit32>(sample_patterns<posit32>(23, 150000));
}

TEST(PositUniversal, ZeroAndNaRMatchUniversalInEveryOperation) {
  const auto special = [](auto tag) {
    using P = decltype(tag);
    const std::uint64_t one = std::uint64_t{1} << (kN<P> - 2);
    return std::vector<std::uint64_t>{0, std::uint64_t{1} << (kN<P> - 1), one, (kMask<P> + 1 - one) & kMask<P>, 1,
                                      (std::uint64_t{1} << (kN<P> - 1)) - 1};
  };
  for (const int width : {16, 32}) {
    std::vector<std::uint64_t> a, b;
    const std::vector<std::uint64_t> v = width == 16 ? special(posit16{}) : special(posit32{});
    for (const std::uint64_t x : v) {
      for (const std::uint64_t y : v) {
        a.push_back(x);
        b.push_back(y);
      }
    }
    if (width == 16) arithmetic_matches<posit16>(a, b); else arithmetic_matches<posit32>(a, b);
  }
}
