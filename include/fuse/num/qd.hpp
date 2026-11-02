#pragma once

/// `fuse::num::qd`: quad-double arithmetic, an unevaluated sum of four binary64
/// numbers x[0] + x[1] + x[2] + x[3] with |x[i+1]| <= ulp(x[i]), about 212
/// bits of significand (unit roundoff 2^-212) over binary64's exponent range.
/// Callable from host and device code, never throws, trivially copyable, 32
/// bytes.
///
/// ALGORITHMS. Those of Hida, Li and Bailey, "Algorithms for quad-double
/// precision floating point arithmetic" (15th IEEE Symposium on Computer
/// Arithmetic, 2001): the merge-based accurate addition with quick three-sum
/// accumulation, the multiplication by the ten partial products, division by
/// successive residuals, and a square root by Newton iteration on 1/sqrt, with
/// the renormalisation of five partial sums. The operation sequences are
/// reproduced exactly, so that on normal-range binary64 operands every result is
/// bit for bit what stillwater-sc/universal's `qd` returns
/// (test/num/dd_universal_test.cpp, built with FUSE_ENABLE_UNIVERSAL).
///
/// ACCURACY (test/num/qd_test.cpp checks each against exact arithmetic): + - * /
/// and sqrt are within 2^-206 of the exact result, relative to its magnitude
/// (observed worst case over the tests: 2^-211), for operands in the normal range
/// (the addition also under cancellation).
/// An infinity or a NaN in propagates to x[0] of the result.
///
/// LIMBS. A qd is four consecutive binary64 values, most significant first,
/// which is how a field of them is viewed as scalars and moved by every
/// operation that does not need the arithmetic.

#include <string_view>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "fuse/model/value_traits.hpp"
#include "fuse/num/dd.hpp"
#include "fuse/num/eft.hpp"

namespace fuse::num {

struct qd {
  double x[4] = {0.0, 0.0, 0.0, 0.0};

  constexpr qd() = default;
  /// The binary64 value `a`, exactly.
  KOKKOS_INLINE_FUNCTION constexpr qd(double a) noexcept : x{a, 0.0, 0.0, 0.0} {}  // NOLINT: implicit
  /// A dd, exactly.
  KOKKOS_INLINE_FUNCTION constexpr qd(const dd& a) noexcept : x{a.hi, a.lo, 0.0, 0.0} {}  // NOLINT: implicit
  /// The limbs as given; the caller guarantees they are in canonical form.
  KOKKOS_INLINE_FUNCTION constexpr qd(double a0, double a1, double a2, double a3) noexcept
      : x{a0, a1, a2, a3} {}

  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr double operator[](int i) const noexcept {
    return x[i];
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr double& operator[](int i) noexcept { return x[i]; }

  /// The dd nearest this value, rounded to nearest from the lower limbs.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION dd to_dd() const noexcept {
    return dd(x[0], x[1]) + dd(x[2], x[3]);
  }
  /// The binary64 value nearest this one: the four limbs summed in a dd, whose
  /// two limbs are then added (a result of the arithmetic can have |x[1]| above
  /// half an ulp of x[0], so x[0] alone is not always the nearest value). The
  /// result differs from the correctly rounded sum only when that lies within
  /// 2^-104 (relative) of a midpoint between two binary64 numbers.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION double to_double() const noexcept {
    return to_dd().to_double();
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool isnan() const noexcept {
    return eft::is_nan(x[0]) || eft::is_nan(x[1]) || eft::is_nan(x[2]) || eft::is_nan(x[3]);
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool isinf() const noexcept { return eft::is_inf(x[0]); }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool iszero() const noexcept { return x[0] == 0.0; }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool isneg() const noexcept { return x[0] < 0.0; }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr qd operator-() const noexcept {
    return qd(-x[0], -x[1], -x[2], -x[3]);
  }
};

static_assert(sizeof(qd) == 4 * sizeof(double));
static_assert(alignof(qd) == alignof(double));
static_assert(std::is_trivially_copyable_v<qd>);
static_assert(std::is_standard_layout_v<qd>);

namespace detail {

[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION double absv(double v) noexcept { return v < 0.0 ? -v : v; }

/// Adds `c` to the pair (a, b): if the sum does not fit two doubles, the sum is
/// returned and (a, b) holds the remainder; otherwise 0 is returned and (a, b)
/// holds the sum.
KOKKOS_INLINE_FUNCTION double quick_three_accumulation(double& a, double& b, double c) noexcept {
  double s = eft::two_sum(b, c, b);
  s = eft::two_sum(a, s, a);
  const bool za = (a != 0.0);
  const bool zb = (b != 0.0);
  if (za && zb) return s;
  if (!zb) {
    b = a;
    a = s;
  } else {
    a = s;
  }
  return 0.0;
}

}  // namespace detail

[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator+(const qd& a, const qd& b) noexcept {
  double u, v;
  int i = 0, j = 0, k = 0;
  if (detail::absv(a[i]) > detail::absv(b[j])) {
    u = a[i++];
  } else {
    u = b[j++];
  }
  if (detail::absv(a[i]) > detail::absv(b[j])) {
    v = a[i++];
  } else {
    v = b[j++];
  }
  u = eft::quick_two_sum(u, v, v);

  double c[4] = {0.0, 0.0, 0.0, 0.0};
  while (k < 4) {
    if (i >= 4 && j >= 4) {
      c[k] = u;
      if (k < 3) c[++k] = v;
      break;
    }
    double t;
    if (i >= 4) {
      t = b[j++];
    } else if (j >= 4) {
      t = a[i++];
    } else if (detail::absv(a[i]) > detail::absv(b[j])) {
      t = a[i++];
    } else {
      t = b[j++];
    }
    const double s = detail::quick_three_accumulation(u, v, t);
    if (s != 0.0) c[k++] = s;
  }
  for (k = i; k < 4; k++) c[3] += a[k];
  for (k = j; k < 4; k++) c[3] += b[k];
  detail::renorm4(c[0], c[1], c[2], c[3]);
  return qd(c[0], c[1], c[2], c[3]);
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator-(const qd& a, const qd& b) noexcept {
  return a + (-b);
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator*(const qd& a, const qd& b) noexcept {
  double q0, q1, q2, q3, q4, q5;
  double p0 = eft::two_prod(a[0], b[0], q0);
  double p1 = eft::two_prod(a[0], b[1], q1);
  double p2 = eft::two_prod(a[1], b[0], q2);
  double p3 = eft::two_prod(a[0], b[2], q3);
  double p4 = eft::two_prod(a[1], b[1], q4);
  double p5 = eft::two_prod(a[2], b[0], q5);

  eft::three_sum(p1, p2, q0);
  // six-three sum of p2, q1, q2, p3, p4, p5
  eft::three_sum(p2, q1, q2);
  eft::three_sum(p3, p4, p5);
  // (s0, s1, s2) = (p2, q1, q2) + (p3, p4, p5)
  double t0, t1;
  double s0 = eft::two_sum(p2, p3, t0);
  double s1 = eft::two_sum(q1, p4, t1);
  double s2 = q2 + p5;
  s1 = eft::two_sum(s1, t0, t0);
  s2 += (t0 + t1);

  // O(eps^3) terms
  double q6, q7, q8, q9;
  double p6 = eft::two_prod(a[0], b[3], q6);
  double p7 = eft::two_prod(a[1], b[2], q7);
  double p8 = eft::two_prod(a[2], b[1], q8);
  double p9 = eft::two_prod(a[3], b[0], q9);

  // nine-two sum of q0, s1, q3, q4, q5, p6, p7, p8, p9
  q0 = eft::two_sum(q0, q3, q3);
  q4 = eft::two_sum(q4, q5, q5);
  p6 = eft::two_sum(p6, p7, p7);
  p8 = eft::two_sum(p8, p9, p9);
  t0 = eft::two_sum(q0, q4, t1);
  t1 += (q3 + q5);
  double r1;
  const double r0 = eft::two_sum(p6, p8, r1);
  r1 += (p7 + p9);
  q3 = eft::two_sum(t0, r0, q4);
  q4 += (t1 + r1);
  t0 = eft::two_sum(q3, s1, t1);
  t1 += q4;

  // O(eps^4) terms: nine-one sum
  t1 += eft::mul(a[1], b[3]) + eft::mul(a[2], b[2]) + eft::mul(a[3], b[1]) + q6 + q7 + q8 + q9 + s2;

  detail::renorm5(p0, p1, s0, t0, t1);
  return qd(p0, p1, s0, t0);
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator/(const qd& a, const qd& b) noexcept {
  if (a.isnan()) return a;
  if (b.isnan()) return b;
  if (b.isinf()) {
    if (a.isinf()) return qd(eft::quiet_nan());
    return qd(eft::signbit(a[0]) == eft::signbit(b[0]) ? 0.0 : -0.0);
  }
  if (b.iszero()) {
    if (a.iszero()) return qd(eft::quiet_nan());
    return qd(eft::signbit(a[0]) == eft::signbit(b[0]) ? eft::infinity() : -eft::infinity());
  }
  qd r;
  double q0 = a[0] / b[0];
  r = a - (b * qd(q0));
  double q1 = r[0] / b[0];
  r = r - (b * qd(q1));
  double q2 = r[0] / b[0];
  r = r - (b * qd(q2));
  double q3 = r[0] / b[0];
  r = r - (b * qd(q3));
  double q4 = r[0] / b[0];
  detail::renorm5(q0, q1, q2, q3, q4);
  return qd(q0, q1, q2, q3);
}

KOKKOS_INLINE_FUNCTION qd& operator+=(qd& a, const qd& b) noexcept { return a = a + b; }
KOKKOS_INLINE_FUNCTION qd& operator-=(qd& a, const qd& b) noexcept { return a = a - b; }
KOKKOS_INLINE_FUNCTION qd& operator*=(qd& a, const qd& b) noexcept { return a = a * b; }
KOKKOS_INLINE_FUNCTION qd& operator/=(qd& a, const qd& b) noexcept { return a = a / b; }

[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator+(const qd& a, double b) noexcept { return a + qd(b); }
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator+(double a, const qd& b) noexcept { return qd(a) + b; }
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator-(const qd& a, double b) noexcept { return a - qd(b); }
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator-(double a, const qd& b) noexcept { return qd(a) - b; }
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator*(const qd& a, double b) noexcept { return a * qd(b); }
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator*(double a, const qd& b) noexcept { return qd(a) * b; }
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator/(const qd& a, double b) noexcept { return a / qd(b); }
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd operator/(double a, const qd& b) noexcept { return qd(a) / b; }

/// Order: the limbs from the most significant; a NaN compares false.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator==(const qd& a, const qd& b) noexcept {
  return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const qd& a, const qd& b) noexcept {
  return !(a == b);
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator<(const qd& a, const qd& b) noexcept {
  for (int i = 0; i < 4; ++i) {
    if (a[i] != b[i]) return a[i] < b[i];  // false when either limb is a NaN
  }
  return false;
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator>(const qd& a, const qd& b) noexcept {
  return b < a;
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const qd& a, const qd& b) noexcept {
  return a < b || a == b;
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const qd& a, const qd& b) noexcept {
  return b <= a;
}

/// 2^k times a qd, exactly (while no limb leaves the exponent range).
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd ldexp(const qd& a, int k) noexcept {
  return qd(Kokkos::ldexp(a[0], k), Kokkos::ldexp(a[1], k), Kokkos::ldexp(a[2], k),
            Kokkos::ldexp(a[3], k));
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION qd abs(const qd& a) noexcept { return a[0] < 0.0 ? -a : a; }

/// a * 2^k for a power of two `p`: each limb scaled, exactly.
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd mul_pwr2(const qd& a, double p) noexcept {
  return qd(eft::mul(a[0], p), eft::mul(a[1], p), eft::mul(a[2], p), eft::mul(a[3], p));
}

/// a * a.
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd sqr(const qd& a) noexcept {
  double q0, q1, q2, q3;
  double p0 = eft::two_sqr(a[0], q0);
  double p1 = eft::two_prod(eft::mul(2.0, a[0]), a[1], q1);
  double p2 = eft::two_prod(eft::mul(2.0, a[0]), a[2], q2);
  double p3 = eft::two_sqr(a[1], q3);

  p1 = eft::two_sum(q0, p1, q0);

  q0 = eft::two_sum(q0, q1, q1);
  p2 = eft::two_sum(p2, p3, p3);

  double t0, t1;
  double s0 = eft::two_sum(q0, p2, t0);
  double s1 = eft::two_sum(q1, p3, t1);

  s1 = eft::two_sum(s1, t0, t0);
  t0 += t1;

  s1 = eft::quick_two_sum(s1, t0, t0);
  p2 = eft::quick_two_sum(s0, s1, t1);
  p3 = eft::quick_two_sum(t1, t0, q0);

  double p4 = eft::mul(eft::mul(2.0, a[0]), a[3]);
  double p5 = eft::mul(eft::mul(2.0, a[1]), a[2]);

  p4 = eft::two_sum(p4, p5, p5);
  q2 = eft::two_sum(q2, q3, q3);

  t0 = eft::two_sum(p4, q2, t1);
  t1 = t1 + p5 + q3;

  p3 = eft::two_sum(p3, t0, p4);
  p4 = p4 + q0 + t1;

  detail::renorm5(p0, p1, p2, p3, p4);
  return qd(p0, p1, p2, p3);
}

/// The square root of a non-negative qd; a negative argument gives a NaN.
/// Three Newton steps on 1/sqrt, after scaling the argument into [0.5, 2) by an
/// even power of two (exact).
[[nodiscard]] KOKKOS_INLINE_FUNCTION qd sqrt(const qd& a) noexcept {
  if (a.isneg()) return qd(eft::quiet_nan());
  if (a.iszero()) return qd(0.0);
  if (a.isnan() || a.isinf()) return a;
  int e = 0;
  Kokkos::frexp(a[0], &e);
  const int k = e >> 1;  // floor(e / 2)
  const qd b = ldexp(a, -2 * k);

  qd r = qd(1.0 / Kokkos::sqrt(b[0]));
  const qd h = mul_pwr2(b, 0.5);

  r += ((qd(0.5) - h * sqr(r)) * r);
  r += ((qd(0.5) - h * sqr(r)) * r);
  r += ((qd(0.5) - h * sqr(r)) * r);

  r *= b;
  return ldexp(r, k);
}

}  // namespace fuse::num

namespace fuse {

/// A qd is an Arithmetic value type of four binary64 limbs, like dd.
template <>
struct value_traits<num::qd> {
  static constexpr std::string_view name = "qd";
  static constexpr FieldValueKind kind = FieldValueKind::Arithmetic;
  using scalar_type = double;
  static constexpr int scalar_count = 4;
  static constexpr int gradient_size = 0;
  /// The nearest binary64 value (see qd::to_double).
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double to_binary64(const num::qd& v) noexcept {
    return v.to_double();
  }
  /// Exact.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static num::qd from_binary64(double x) noexcept {
    return num::qd(x);
  }
};

static_assert(CoupledValue<num::qd>);

}  // namespace fuse
