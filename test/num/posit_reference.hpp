// The definition of the posit format, read one bit at a time, for the tests of
// fuse::num::posit (posit_test.cpp, posit_universal_test.cpp). Nothing here uses the
// type's own decoding.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace fuse::test {

/// The value of the n-bit posit with exponent size `es` whose pattern is `pattern`, read
/// from the definition one bit at a time: NaN for NaR.
inline double posit_ref_value(int n, int es, std::uint64_t pattern) {
  const std::uint64_t mask = (std::uint64_t{1} << n) - 1;
  pattern &= mask;
  if (pattern == 0) return 0.0;
  if (pattern == (std::uint64_t{1} << (n - 1))) return std::numeric_limits<double>::quiet_NaN();
  const bool negative = ((pattern >> (n - 1)) & 1u) != 0;
  if (negative) pattern = (~pattern + 1) & mask;
  std::vector<int> b;  // the n - 1 bits after the sign, most significant first
  for (int i = n - 2; i >= 0; --i) b.push_back(static_cast<int>((pattern >> i) & 1u));
  std::size_t pos = 0;
  const int first = b[0];
  int run = 0;
  while (pos < b.size() && b[pos] == first) {
    ++run;
    ++pos;
  }
  const int k = first ? run - 1 : -run;
  if (pos < b.size()) ++pos;  // the bit that ends the regime
  int e = 0;
  for (int i = 0; i < es; ++i) {
    e <<= 1;
    if (pos < b.size()) e |= b[pos++];
  }
  double f = 1.0, w = 0.5;
  while (pos < b.size()) {
    if (b[pos++]) f += w;
    w *= 0.5;
  }
  const double v = std::ldexp(f, k * (1 << es) + e);
  return negative ? -v : v;
}

/// The boundary between the positive n-bit patterns u and u + 1 under round to nearest
/// (2022 Posit Standard): the value of the (n + 1)-bit pattern u followed by a one.
inline double posit_boundary(int n, int es, std::uint64_t u) { return posit_ref_value(n + 1, es, 2 * u + 1); }

}  // namespace fuse::test
