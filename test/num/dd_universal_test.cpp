// fuse::num::dd and fuse::num::qd against stillwater-sc/universal's dd and qd
//. Universal is the
// reference implementation the two types are measured against; it is host only and
// never included by FUSE itself. Its headers do not compile under nvcc, so they are
// compiled into a separate program (test/num/universal_oracle.cpp, built with a host
// compiler) that this test runs on files of operands and reads the results back
// from. The test exists only when the build has FUSE_ENABLE_UNIVERSAL=ON and
// FUSE_UNIVERSAL_INCLUDE_DIR pointing at the include/sw directory of a checkout.
//
// The operation sequences of fuse::num reproduce Universal's, so on operands in the
// normal range every result is compared bit for bit (limb by limb; a NaN matches a
// NaN of any payload). Special values are compared the same way.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fuse/num/dd.hpp"
#include "fuse/num/qd.hpp"
#include "fuse_test_main.hpp"
#include "universal_oracle_client.hpp"

namespace {

using fuse::num::dd;
using fuse::num::qd;
using fuse_test::universal::oracle_results;
using fuse_test::universal::Record;
using fuse_test::universal::Result;

enum Op : std::uint32_t { Add, Sub, Mul, Div, Sqrt, AddD, MulD, RSubD, DivD };

const char* op_name(std::uint32_t op) {
  static const char* names[] = {"+", "-", "*", "/", "sqrt", "+ double", "* double", "double -", "/ double"};
  return names[op];
}

std::uint64_t bits(double x) {
  std::uint64_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || bits(a) == bits(b); }

std::string hex(double x) {
  std::ostringstream o;
  o << std::hexfloat << x;
  return o.str();
}

std::string show(const double* p, int n) {
  std::string s = "(";
  for (int i = 0; i < n; ++i) s += (i ? ", " : "") + hex(p[i]);
  return s + ")";
}

Record make(int limbs, std::uint32_t op, const dd& a, const dd& b) {
  Record r{static_cast<std::uint32_t>(limbs), op, {a.hi, a.lo, 0, 0}, {b.hi, b.lo, 0, 0}};
  return r;
}
Record make(int limbs, std::uint32_t op, const qd& a, const qd& b) {
  Record r{static_cast<std::uint32_t>(limbs), op, {a[0], a[1], a[2], a[3]}, {b[0], b[1], b[2], b[3]}};
  return r;
}

/// What fuse::num computes for a record.
Result fuse_result(const Record& r) {
  Result out{{0, 0, 0, 0}};
  if (r.limbs == 2) {
    const dd a(r.a[0], r.a[1]), b(r.b[0], r.b[1]);
    dd v;
    switch (r.op) {
      case Add: v = a + b; break;
      case Sub: v = a - b; break;
      case Mul: v = a * b; break;
      case Div: v = a / b; break;
      case Sqrt: v = fuse::num::sqrt(a); break;
      case AddD: v = a + b.hi; break;
      case MulD: v = a * b.hi; break;
      case RSubD: v = b.hi - a; break;
      default: v = a / b.hi; break;
    }
    out.r[0] = v.hi;
    out.r[1] = v.lo;
  } else {
    const qd a(r.a[0], r.a[1], r.a[2], r.a[3]), b(r.b[0], r.b[1], r.b[2], r.b[3]);
    qd v(0.0);
    switch (r.op) {
      case Add: v = a + b; break;
      case Sub: v = a - b; break;
      case Mul: v = a * b; break;
      case Div: v = a / b; break;
      default: v = fuse::num::sqrt(a); break;
    }
    for (int i = 0; i < 4; ++i) out.r[i] = v[i];
  }
  return out;
}

/// Compares every record's fuse::num result with Universal's; reports the first
/// difference of each operation.
void expect_bitwise(const std::vector<Record>& records) {
  const std::vector<Result> ref = oracle_results(records);
  ASSERT_EQ(ref.size(), records.size());
  bool reported[9] = {false, false, false, false, false, false, false, false, false};
  for (std::size_t i = 0; i < records.size(); ++i) {
    const Record& rec = records[i];
    const Result mine = fuse_result(rec);
    bool ok = true;
    for (std::uint32_t k = 0; k < rec.limbs; ++k) ok = ok && same(mine.r[k], ref[i].r[k]);
    if (!ok && !reported[rec.op]) {
      reported[rec.op] = true;
      ADD_FAILURE() << (rec.limbs == 2 ? "dd " : "qd ") << op_name(rec.op) << " differs from Universal at record " << i
                    << ": a = " << show(rec.a, static_cast<int>(rec.limbs)) << " b = " << show(rec.b, static_cast<int>(rec.limbs))
                    << " fuse = " << show(mine.r, static_cast<int>(rec.limbs))
                    << " universal = " << show(ref[i].r, static_cast<int>(rec.limbs));
    }
  }
}

dd random_dd(std::mt19937_64& rng, int lo_exp, int hi_exp) {
  std::uniform_real_distribution<double> unit(0.5, 1.0), sym(-1.0, 1.0);
  std::uniform_int_distribution<int> expo(lo_exp, hi_exp);
  double h = std::ldexp(unit(rng), expo(rng));
  if (rng() & 1u) h = -h;
  return dd(h, sym(rng) * std::ldexp(std::fabs(h), -54));
}

qd random_qd(std::mt19937_64& rng, int lo_exp, int hi_exp) {
  std::uniform_real_distribution<double> unit(0.5, 1.0), sym(-1.0, 1.0);
  std::uniform_int_distribution<int> expo(lo_exp, hi_exp);
  qd a(0.0);
  a[0] = std::ldexp(unit(rng), expo(rng));
  if (rng() & 1u) a[0] = -a[0];
  for (int i = 1; i < 4; ++i) a[i] = sym(rng) * std::ldexp(std::fabs(a[i - 1]), -54);
  return a;
}

/// The operand pairs of one run: independent, of very different magnitudes, and
/// cancelling to a different depth.
template <class T, class Gen>
std::vector<std::pair<T, T>> operand_pairs(Gen gen, int n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::pair<T, T>> out;
  for (int i = 0; i < n; ++i) {
    const T a = gen(rng, -40, 40);
    T b = a;
    switch (i % 5) {
      case 0: b = gen(rng, -40, 40); break;
      case 1: b = gen(rng, -80, -20); break;
      case 2: b = -a + gen(rng, -100, -60) * 0x1p-20; break;
      case 3: b = -a; break;
      default: b = a; break;
    }
    out.emplace_back(a, b);
  }
  return out;
}

}  // namespace

TEST(DdUniversal, ArithmeticMatchesBitForBitOnRandomOperands) {
  std::vector<Record> records;
  for (const auto& [a, b] : operand_pairs<dd>(random_dd, 100000, 21)) {
    for (const std::uint32_t op : {Add, Sub, Mul, Div}) records.push_back(make(2, op, a, b));
  }
  expect_bitwise(records);
}

TEST(DdUniversal, MixedBinary64OperandsMatchBitForBit) {
  std::mt19937_64 rng(22);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::vector<Record> records;
  for (int i = 0; i < 20000; ++i) {
    const dd a = random_dd(rng, -30, 30);
    const dd b(std::ldexp(unit(rng), 1 + static_cast<int>(rng() % 40) - 20));
    for (const std::uint32_t op : {AddD, MulD, RSubD, DivD}) records.push_back(make(2, op, a, b));
  }
  expect_bitwise(records);
}

TEST(DdUniversal, SquareRootMatchesBitForBit) {
  std::mt19937_64 rng(23);
  std::vector<Record> records;
  for (int i = 0; i < 100000; ++i) {
    dd a = random_dd(rng, -300, 300);
    if (a.hi < 0) a = -a;
    records.push_back(make(2, Sqrt, a, dd(0.0)));
  }
  expect_bitwise(records);
}

TEST(DdUniversal, SpecialValuesMatchBitForBit) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<dd> v = {dd(0.0),  dd(-0.0),  dd(1.0),        dd(-1.0),         dd(0.1),   dd(1.0, 0x1p-60),
                             dd(-3.5, 0x1p-58), dd(1e300), dd(1e-300), dd(inf), dd(-inf), dd(nan)};
  std::vector<Record> records;
  for (const dd& a : v) {
    for (const dd& b : v) {
      // finite operands whose product or quotient stays in range are compared above;
      // here only the special values and the zeros
      const bool special = !std::isfinite(a.hi) || !std::isfinite(b.hi) || a.hi == 0.0 || b.hi == 0.0;
      if (!special) continue;
      for (const std::uint32_t op : {Add, Sub, Mul, Div}) records.push_back(make(2, op, a, b));
    }
  }
  EXPECT_GT(records.size(), 160u);
  expect_bitwise(records);
}

TEST(QdUniversal, ArithmeticMatchesBitForBitOnRandomOperands) {
  std::vector<Record> records;
  for (const auto& [a, b] : operand_pairs<qd>(random_qd, 50000, 31)) {
    for (const std::uint32_t op : {Add, Sub, Mul, Div}) records.push_back(make(4, op, a, b));
  }
  expect_bitwise(records);
}

TEST(QdUniversal, SquareRootMatchesBitForBit) {
  std::mt19937_64 rng(32);
  std::vector<Record> records;
  for (int i = 0; i < 20000; ++i) {
    qd a = random_qd(rng, -200, 200);
    if (a[0] < 0) a = -a;
    records.push_back(make(4, Sqrt, a, qd(0.0)));
  }
  expect_bitwise(records);
}

TEST(QdUniversal, SpecialValuesMatchBitForBit) {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<qd> v = {qd(0.0), qd(-0.0), qd(1.0), qd(-1.0), qd(0.1), qd(1.0, 0x1p-60, 0.0, 0.0),
                             qd(-3.5, 0x1p-58, 0x1p-112, 0.0), qd(1e300), qd(1e-300), qd(inf), qd(-inf), qd(nan)};
  std::vector<Record> records;
  for (const qd& a : v) {
    for (const qd& b : v) {
      const bool special = !std::isfinite(a[0]) || !std::isfinite(b[0]) || a[0] == 0.0 || b[0] == 0.0;
      if (!special) continue;
      for (const std::uint32_t op : {Add, Sub, Mul, Div}) records.push_back(make(4, op, a, b));
    }
  }
  EXPECT_GT(records.size(), 160u);
  expect_bitwise(records);
}
