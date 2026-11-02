// The reference side of test/num/dd_universal_test.cpp and posit_universal_test.cpp:
// stillwater-sc/universal's dd, qd and posit evaluated on operands read from a file. A
// separate program, built with a host C++ compiler and nothing else of FUSE (no
// Kokkos): Universal's headers do not compile under nvcc, which is the compiler of a
// CUDA build, and the tests that compare the results must. (plans/2026-10-22-universal-number-types.md,
// stages U4 and U5.)
//
// Usage: universal_oracle INPUT OUTPUT
// INPUT is a sequence of records, OUTPUT the same number of results, all binary
// (native byte order, binary64 limbs):
//   record: uint32 limbs (2 = dd, 4 = qd, 16 = posit<16,2>, 32 = posit<32,2>), uint32 op,
//           double a[4], double b[4]
//   result: double r[4]
// A dd uses the first two entries of a, b and r. `op` is the enum of the test:
//   0 a+b  1 a-b  2 a*b  3 a/b  4 sqrt(a)
//   5 a+b[0]  6 a*b[0]  7 b[0]-a  8 a/b[0]     (a binary64 operand: Universal's own
//                                              mixed operators; dd and qd only)
// A posit operand is its bit pattern (an integer below 2^16 or 2^32, exact in a
// binary64) in a[0] and b[0], and a posit result is returned the same way in r[0]:
//   0..4 as above on the posits with patterns a[0], b[0]
//   9  the posit nearest the binary64 a[0] (Universal's conversion), as a pattern
//   10 the binary64 value of the posit with pattern a[0] (NaN for NaR)
// An exception thrown by Universal (a square root of a negative number) is written as
// a quiet NaN in every limb.

#include <universal/number/dd/dd.hpp>
#include <universal/number/posit/posit.hpp>
#include <universal/number/qd/qd.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

struct Record {
  std::uint32_t limbs;
  std::uint32_t op;
  double a[4];
  double b[4];
};
struct Result {
  double r[4];
};

namespace su = sw::universal;

template <class T>
T apply(std::uint32_t op, const T& a, const T& b, double bd) {
  switch (op) {
    case 0: return a + b;
    case 1: return a - b;
    case 2: return a * b;
    case 3: return a / b;
    case 4: return su::sqrt(a);
    case 5: return a + bd;
    case 6: return a * bd;
    case 7: return bd - a;
    default: return a / bd;
  }
}

/// A posit<N,2> operand or result as the pattern of its N bits.
template <int N>
Result evaluate_posit(const Record& rec) {
  using P = su::posit<N, 2>;
  constexpr std::uint64_t mask = (N == 32) ? 0xffffffffull : 0xffffull;
  const auto from_pattern = [](double pattern) {
    P p;
    p.setbits(static_cast<std::uint64_t>(pattern));
    return p;
  };
  const auto pattern_of = [](const P& p) {
    return static_cast<double>(static_cast<std::uint64_t>(p.bits().to_ull()) & mask);
  };
  Result out{{0.0, 0.0, 0.0, 0.0}};
  switch (rec.op) {
    case 9: out.r[0] = pattern_of(P(rec.a[0])); break;
    case 10: out.r[0] = static_cast<double>(from_pattern(rec.a[0])); break;
    default: out.r[0] = pattern_of(apply<P>(rec.op, from_pattern(rec.a[0]), from_pattern(rec.b[0]), 0.0)); break;
  }
  return out;
}

Result evaluate(const Record& rec) {
  Result out{{0.0, 0.0, 0.0, 0.0}};
  if (rec.limbs == 16) return evaluate_posit<16>(rec);
  if (rec.limbs == 32) return evaluate_posit<32>(rec);
  if (rec.limbs == 2) {
    const su::dd a(rec.a[0], rec.a[1]), b(rec.b[0], rec.b[1]);
    const su::dd r = apply<su::dd>(rec.op, a, b, rec.b[0]);
    out.r[0] = r.high();
    out.r[1] = r.low();
  } else {
    const su::qd a(rec.a[0], rec.a[1], rec.a[2], rec.a[3]), b(rec.b[0], rec.b[1], rec.b[2], rec.b[3]);
    const su::qd r = apply<su::qd>(rec.op, a, b, rec.b[0]);
    for (int i = 0; i < 4; ++i) out.r[i] = r[i];
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: universal_oracle INPUT OUTPUT\n");
    return 2;
  }
  std::FILE* in = std::fopen(argv[1], "rb");
  if (in == nullptr) return 3;
  std::vector<Record> records;
  Record rec;
  while (std::fread(&rec, sizeof rec, 1, in) == 1) records.push_back(rec);
  std::fclose(in);
  std::FILE* out = std::fopen(argv[2], "wb");
  if (out == nullptr) return 4;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const Record& r : records) {
    Result res{{nan, nan, nan, nan}};
    try {
      res = evaluate(r);
    } catch (...) {
    }
    std::fwrite(&res, sizeof res, 1, out);
  }
  std::fclose(out);
  return 0;
}
