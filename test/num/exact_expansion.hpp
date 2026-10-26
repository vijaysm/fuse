#pragma once

/// Exact arithmetic on sums of binary64 numbers, for tests: the oracle the
/// double-double and quad-double results are measured against. An expansion is a
/// list of doubles whose exact sum is the value; sums and products of expansions
/// are computed with error-free transformations only, so no rounding ever
/// happens (Shewchuk, "Adaptive precision floating-point arithmetic and fast
/// robust geometric predicates", 1997). Nothing here uses fuse::num.

#include <algorithm>
#include <cmath>
#include <vector>

namespace fuse::test {

using Expansion = std::vector<double>;

inline double two_sum_exact(double a, double b, double& err) {
  volatile double s = a + b;
  volatile double bb = s - a;
  err = (a - (s - bb)) + (b - bb);
  return s;
}

inline double two_prod_exact(double a, double b, double& err) {
  volatile double p = a * b;
  err = std::fma(a, b, -p);
  return p;
}

/// e + b, exactly.
inline Expansion grow(const Expansion& e, double b) {
  Expansion h;
  double q = b;
  for (const double ei : e) {
    double hi;
    q = two_sum_exact(q, ei, hi);
    if (hi != 0.0) h.push_back(hi);
  }
  h.push_back(q);
  return h;
}

inline Expansion add(const Expansion& e, const Expansion& f) {
  Expansion h = e;
  for (const double fi : f) h = grow(h, fi);
  return h;
}

inline Expansion negate(Expansion e) {
  for (double& x : e) x = -x;
  return e;
}

inline Expansion sub(const Expansion& e, const Expansion& f) { return add(e, negate(f)); }

/// e * b, exactly.
inline Expansion scale(const Expansion& e, double b) {
  Expansion h;
  for (const double ei : e) {
    double err;
    const double p = two_prod_exact(ei, b, err);
    h = grow(h, err);
    h = grow(h, p);
  }
  return h;
}

inline Expansion mul(const Expansion& e, const Expansion& f) {
  Expansion h;
  for (const double fi : f) h = add(h, scale(e, fi));
  return h;
}

/// The value as a binary64 number to a relative accuracy of about 2^-50: the
/// components summed from the smallest to the largest. Enough to size an error;
/// the exact comparison is made with the expansions themselves.
inline double approx(const Expansion& e) {
  Expansion s = e;
  std::sort(s.begin(), s.end(), [](double a, double b) { return std::fabs(a) < std::fabs(b); });
  long double acc = 0.0L;
  for (const double x : s) acc += static_cast<long double>(x);
  return static_cast<double>(acc);
}

}  // namespace fuse::test
