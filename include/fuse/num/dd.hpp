#pragma once

/// `fuse::num::dd`: double-double arithmetic, an unevaluated sum hi + lo of two
/// binary64 numbers with |lo| <= ulp(hi) / 2, about 106 bits of significand
/// (unit roundoff u2 = 2^-106, so a relative error of a few 2^-106 per
/// operation) over binary64's exponent range. Callable from host and device
/// code, never throws, trivially copyable, 16 bytes.
///
/// ALGORITHMS. The error-free transformations in fuse/num/eft.hpp and the
/// double-double and quad-double algorithms of Hida, Li and Bailey, "Algorithms
/// for quad-double precision floating point arithmetic" (15th IEEE Symposium on
/// Computer Arithmetic, 2001) and the QD library built on them; the accurate
/// addition and the multiplication follow the three-sum formulation, and
/// division and the square root follow the quad-double residual forms. Their
/// operation sequences are reproduced exactly, so that on binary64 operands in the
/// normal range every result is bit for bit what stillwater-sc/universal's `dd`
/// returns (test/num/dd_universal_test.cpp, built with FUSE_ENABLE_UNIVERSAL):
/// the oracle of this type, never a dependency of it.
///
/// ACCURACY (test/num/dd_test.cpp checks each against exact arithmetic):
///   + and -    |error| <= 2^-103 |exact|, also under cancellation
///   *          |error| <= 2^-103 |exact|
///   /          |error| <= 2^-102 |exact|
///   sqrt       |error| <= 2^-102 |exact|
/// Results for operands near the ends of the exponent range (the product or the
/// square of a limb overflowing or going subnormal) are not specified, except
/// that an infinity or a NaN in propagates to the hi limb of the result.
///
/// NORMAL FORM. A dd is the exact sum hi + lo. The operations return |lo| of at
/// most one ulp of hi (the accurate addition can leave |lo| between half and one
/// ulp after a heavy cancellation, because it is not re-normalised: that is what
/// makes it agree with Universal bit for bit), and a value built with
/// `from_sum` or `dd(x)` has |lo| of at most half an ulp.
///
/// EXACT CONVERSIONS. `dd(x)` and `dd::from_sum(a, b)`/`dd::from_product(a, b)`
/// hold their value exactly. `to_double()` is fl(hi + lo): the sum of the two
/// limbs rounded once, to nearest with ties to even, is the binary64 value nearest
/// the dd value whatever the form of the limbs, so narrowing a dd to binary64 is
/// one correct rounding.
///
/// LIMBS. A dd is two consecutive binary64 values, hi then lo, which is how a
/// field of them is viewed as scalars (fuse/field/field_ref.hpp) and moved by
/// every operation that does not need the arithmetic.

#include <cstdint>
#include <string_view>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "fuse/model/value_traits.hpp"
#include "fuse/num/eft.hpp"

namespace fuse::num {

struct dd;
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator+(const dd&, const dd&) noexcept;
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator-(const dd&, const dd&) noexcept;
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator*(const dd&, const dd&) noexcept;
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator/(const dd&, const dd&) noexcept;

struct dd {
  double hi = 0.0;
  double lo = 0.0;

  constexpr dd() = default;
  /// The binary64 value `x`, exactly.
  KOKKOS_INLINE_FUNCTION constexpr dd(double x) noexcept : hi(x), lo(0.0) {}  // NOLINT: implicit
  /// The limbs as given. The caller guarantees |l| <= ulp(h) / 2; use
  /// `from_sum` to normalise two arbitrary numbers.
  KOKKOS_INLINE_FUNCTION constexpr dd(double h, double l) noexcept : hi(h), lo(l) {}

  /// a + b, exactly.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static dd from_sum(double a, double b) noexcept {
    double e;
    const double s = eft::two_sum(a, b, e);
    return dd(s, e);
  }
  /// a * b, exactly.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static dd from_product(double a, double b) noexcept {
    double e;
    const double p = eft::two_prod(a, b, e);
    return dd(p, e);
  }

  /// The binary64 value nearest this one: hi + lo rounded once.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr double to_double() const noexcept {
    return hi + lo;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool isnan() const noexcept {
    return eft::is_nan(hi) || eft::is_nan(lo);
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool isinf() const noexcept { return eft::is_inf(hi); }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool iszero() const noexcept { return hi == 0.0; }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION bool isneg() const noexcept { return hi < 0.0; }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr dd operator-() const noexcept {
    return dd(-hi, -lo);
  }
  KOKKOS_INLINE_FUNCTION dd& operator+=(const dd& r) noexcept { return *this = *this + r; }
  KOKKOS_INLINE_FUNCTION dd& operator-=(const dd& r) noexcept { return *this = *this - r; }
  KOKKOS_INLINE_FUNCTION dd& operator*=(const dd& r) noexcept { return *this = *this * r; }
  KOKKOS_INLINE_FUNCTION dd& operator/=(const dd& r) noexcept { return *this = *this / r; }
};

static_assert(sizeof(dd) == 2 * sizeof(double));
static_assert(alignof(dd) == alignof(double));
static_assert(std::is_trivially_copyable_v<dd>);
static_assert(std::is_standard_layout_v<dd>);

namespace detail {

/// Brings four or five partial sums to a canonical quad-double (Hida, Li and
/// Bailey, Algorithm 6, in the form of the QD library). The dd fused
/// multiply-add below needs the five-term form; fuse/num/qd.hpp uses both.
KOKKOS_INLINE_FUNCTION void renorm4(double& a0, double& a1, double& a2, double& a3) noexcept {
  if (eft::is_inf(a0)) return;
  double s0, s1, s2 = 0.0, s3 = 0.0;
  s0 = eft::quick_two_sum(a2, a3, a3);
  s0 = eft::quick_two_sum(a1, s0, a2);
  a0 = eft::quick_two_sum(a0, s0, a1);
  s0 = a0;
  s1 = a1;
  if (s1 != 0.0) {
    s1 = eft::quick_two_sum(s1, a2, s2);
    if (s2 != 0.0) {
      s2 = eft::quick_two_sum(s2, a3, s3);
    } else {
      s1 = eft::quick_two_sum(s1, a3, s2);
    }
  } else {
    s0 = eft::quick_two_sum(s0, a2, s1);
    if (s1 != 0.0) {
      s1 = eft::quick_two_sum(s1, a3, s2);
    } else {
      s0 = eft::quick_two_sum(s0, a3, s1);
    }
  }
  a0 = s0;
  a1 = s1;
  a2 = s2;
  a3 = s3;
}

KOKKOS_INLINE_FUNCTION void renorm5(double& a0, double& a1, double& a2, double& a3,
                                    double& a4) noexcept {
  if (eft::is_inf(a0)) return;
  double s0, s1, s2 = 0.0, s3 = 0.0;
  s0 = eft::quick_two_sum(a3, a4, a4);
  s0 = eft::quick_two_sum(a2, s0, a3);
  s0 = eft::quick_two_sum(a1, s0, a2);
  a0 = eft::quick_two_sum(a0, s0, a1);
  s0 = a0;
  s1 = a1;
  s0 = eft::quick_two_sum(a0, a1, s1);
  if (s1 != 0.0) {
    s1 = eft::quick_two_sum(s1, a2, s2);
    if (s2 != 0.0) {
      s2 = eft::quick_two_sum(s2, a3, s3);
      if (s3 != 0.0) {
        s3 += a4;
      } else {
        s2 += a4;
      }
    } else {
      s1 = eft::quick_two_sum(s1, a3, s2);
      if (s2 != 0.0) {
        s2 = eft::quick_two_sum(s2, a4, s3);
      } else {
        s1 = eft::quick_two_sum(s1, a4, s2);
      }
    }
  } else {
    s0 = eft::quick_two_sum(s0, a2, s1);
    if (s1 != 0.0) {
      s1 = eft::quick_two_sum(s1, a3, s2);
      if (s2 != 0.0) {
        s2 = eft::quick_two_sum(s2, a4, s3);
      } else {
        s1 = eft::quick_two_sum(s1, a4, s2);
      }
    } else {
      s0 = eft::quick_two_sum(s0, a3, s1);
      if (s1 != 0.0) {
        s1 = eft::quick_two_sum(s1, a4, s2);
      } else {
        s0 = eft::quick_two_sum(s0, a4, s1);
      }
    }
  }
  a0 = s0;
  a1 = s1;
  a2 = s2;
  a3 = s3;
}

/// The exact product of two dd as four limbs (powers of eps 0, 1, 2, 3).
KOKKOS_INLINE_FUNCTION void dd_mul_exact(const dd& a, const dd& b, double p[4]) noexcept {
  double p4 = 0.0, p5 = 0.0, p6 = 0.0, p7 = 0.0;
  p[0] = eft::two_prod(a.hi, b.hi, p[1]);
  if (eft::is_finite(p[0])) {
    p[2] = eft::two_prod(a.hi, b.lo, p4);
    p[3] = eft::two_prod(a.lo, b.hi, p5);
    p6 = eft::two_prod(a.lo, b.lo, p7);
    eft::three_sum(p[1], p[2], p[3]);
    eft::three_sum(p4, p5, p6);
    p[2] = eft::two_sum(p[2], p4, p4);
    eft::three_sum(p[3], p4, p5);
    p[3] = eft::two_sum(p[3], p7, p7);
    p4 += (p6 + p7);
    renorm5(p[0], p[1], p[2], p[3], p4);
  } else {
    p[1] = p[2] = p[3] = 0.0;
  }
}

/// Adds a dd to four limbs, renormalising.
KOKKOS_INLINE_FUNCTION void qd_plus_dd(const double a[4], const dd& b, double s[4]) noexcept {
  double t0, t1;
  s[0] = eft::two_sum(a[0], b.hi, t0);
  s[1] = eft::two_sum(a[1], b.lo, t1);
  s[1] = eft::two_sum(s[1], t0, t0);
  s[2] = a[2];
  eft::three_sum(s[2], t0, t1);
  s[3] = eft::two_sum(a[3], t0, t0);
  t0 += t1;
  renorm5(s[0], s[1], s[2], s[3], t0);
}

/// a * b + c with the product held exactly until the sum.
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd dd_fma(const dd& a, const dd& b, const dd& c) noexcept {
  double p[4];
  dd_mul_exact(a, b, p);
  double s[4];
  qd_plus_dd(p, c, s);
  s[0] = eft::two_sum(s[0], s[1] + s[2] + s[3], s[1]);
  return dd(s[0], s[1]);
}

}  // namespace detail

[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator+(const dd& a, const dd& b) noexcept {
  double s2;
  double hi = eft::two_sum(a.hi, b.hi, s2);
  double lo;
  if (eft::is_finite(hi)) {
    double t2, t1 = eft::two_sum(a.lo, b.lo, t2);
    lo = eft::two_sum(s2, t1, t1);
    t1 += t2;
    eft::three_sum(hi, lo, t1);
  } else {
    lo = 0.0;
  }
  return dd(hi, lo);
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator-(const dd& a, const dd& b) noexcept {
  return a + (-b);
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator*(const dd& a, const dd& b) noexcept {
  double p[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // powers of eps in p: 0, 1, 1, 1, 2, 2, 2
  p[0] = eft::two_prod(a.hi, b.hi, p[1]);
  if (eft::is_finite(p[0])) {
    p[2] = eft::two_prod(a.hi, b.lo, p[4]);
    p[3] = eft::two_prod(a.lo, b.hi, p[5]);
    p[6] = eft::mul(a.lo, b.lo);
    eft::three_sum(p[1], p[2], p[3]);
    p[2] += p[4] + p[5] + p[6];
    eft::three_sum(p[0], p[1], p[2]);
    return dd(p[0], p[1]);
  }
  return dd(p[0], 0.0);
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator/(const dd& a, const dd& b) noexcept {
  if (a.isnan()) return a;
  if (b.isnan()) return b;
  if (b.isinf()) {
    if (a.isinf()) return dd(eft::quiet_nan(), 0.0);
    return dd(eft::signbit(a.hi) == eft::signbit(b.hi) ? 0.0 : -0.0, 0.0);
  }
  if (b.iszero()) {
    if (a.iszero()) return dd(eft::quiet_nan(), 0.0);
    const bool same = eft::signbit(a.hi) == eft::signbit(b.hi);
    return dd(same ? eft::infinity()
                   : -eft::infinity(),
              0.0);
  }
  double q1 = a.hi / b.hi;  // the approximate quotient
  if (!eft::is_finite(q1)) return dd(q1, 0.0);
  dd r = detail::dd_fma(dd(-q1), b, a);
  double q2 = r.hi / b.hi;
  r = detail::dd_fma(dd(-q2), b, r);
  double q3 = r.hi / b.hi;
  eft::three_sum(q1, q2, q3);
  return dd(q1, q2);
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator+(const dd& a, double b) noexcept { return a + dd(b); }
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator+(double a, const dd& b) noexcept { return dd(a) + b; }
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator-(const dd& a, double b) noexcept { return a - dd(b); }
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator-(double a, const dd& b) noexcept { return dd(a) - b; }
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator*(const dd& a, double b) noexcept { return a * dd(b); }
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator*(double a, const dd& b) noexcept { return dd(a) * b; }
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator/(const dd& a, double b) noexcept { return a / dd(b); }
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd operator/(double a, const dd& b) noexcept { return dd(a) / b; }

/// Order: the hi limbs, then the lo limbs; a NaN compares false.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator==(const dd& a, const dd& b) noexcept {
  return a.hi == b.hi && a.lo == b.lo;
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator!=(const dd& a, const dd& b) noexcept {
  return !(a == b);
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator<(const dd& a, const dd& b) noexcept {
  return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator>(const dd& a, const dd& b) noexcept {
  return b < a;
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator<=(const dd& a, const dd& b) noexcept {
  return a < b || a == b;
}
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool operator>=(const dd& a, const dd& b) noexcept {
  return b <= a;
}

/// 2^k times a dd, exactly (while no limb leaves the exponent range).
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd ldexp(const dd& a, int k) noexcept {
  return dd(Kokkos::ldexp(a.hi, k), Kokkos::ldexp(a.lo, k));
}

[[nodiscard]] KOKKOS_INLINE_FUNCTION dd abs(const dd& a) noexcept { return a.hi < 0.0 ? -a : a; }

/// a * a.
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd sqr(const dd& a) noexcept {
  if (a.isnan()) return a;
  double p2, p1 = eft::two_sqr(a.hi, p2);
  p2 += eft::mul(eft::mul(2.0, a.hi), a.lo);
  p2 += eft::mul(a.lo, a.lo);
  double s2, s1 = eft::quick_two_sum(p1, p2, s2);
  return dd(s1, s2);
}

/// The square root of a non-negative dd; a negative argument gives a NaN.
/// Karp's trick on the 1/sqrt of the hi limb, after scaling the argument into
/// [0.5, 2) by an even power of two (exact), so no intermediate overflows or
/// goes subnormal wherever the argument sits.
[[nodiscard]] KOKKOS_INLINE_FUNCTION dd sqrt(const dd& a) noexcept {
  if (a.iszero()) return dd{};
  if (a.isneg()) return dd(eft::quiet_nan(), 0.0);
  if (a.isnan() || a.isinf()) return a;
  int e = 0;
  Kokkos::frexp(a.hi, &e);
  const int k = e >> 1;  // floor(e / 2)
  const dd b = ldexp(a, -2 * k);
  const double x = 1.0 / Kokkos::sqrt(b.hi);
  const double bx = eft::mul(b.hi, x);
  const double corr = eft::mul((b - sqr(dd(bx))).hi, eft::mul(x, 0.5));
  return ldexp(dd::from_sum(bx, corr), k);
}

}  // namespace fuse::num

namespace fuse {

/// A dd is an Arithmetic value type: two binary64 limbs that tile it (so a field
/// of them is viewed as scalars, `hi` then `lo`), and an arithmetic of its own.
template <>
struct value_traits<num::dd> {
  static constexpr std::string_view name = "dd";
  static constexpr FieldValueKind kind = FieldValueKind::Arithmetic;
  using scalar_type = double;
  static constexpr int scalar_count = 2;
  static constexpr int gradient_size = 0;
  /// The nearest binary64 value: one correct rounding.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double to_binary64(const num::dd& v) noexcept {
    return v.to_double();
  }
  /// Exact.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static num::dd from_binary64(double x) noexcept {
    return num::dd(x);
  }
};

static_assert(CoupledValue<num::dd>);

}  // namespace fuse
