#pragma once

/// `fuse::num::dual<S, N>`: a forward-mode automatic-differentiation number,
/// the value of a quantity together with its gradient with respect to N
/// parameters.
///
/// WHAT IT IS. A truncated first-order Taylor expansion in N directions: for a
/// quantity f(p) of N parameters seeded with `variable(p0, k)`, the result of
/// any expression built from the operations below holds f(p0) and the N
/// entries of the gradient, the partial derivatives df/dp_k at p0, each exact to
/// rounding (Griewank and Walther, "Evaluating Derivatives", 2nd ed., SIAM
/// 2008, chapter 3, forward mode with N directions). The gradient is
/// propagated by the rule for each operation; no finite difference is involved.
///
/// WHY A FIELD OF THEM CROSSES THE COUPLER UNCHANGED. The value occupies
/// scalar 0 and the gradient scalars 1..N of N+1 consecutive scalars of type
/// S, so a field of `dual` is, to every operation that acts on each scalar
/// separately, a field with N+1 times as many components (`value_traits`
/// below: ScalarArray, gradient_size N). A remap y = W x with W independent of
/// the field maps the gradient of x to the gradient of y exactly, because W is
/// linear. The operations that are not scalar by scalar are named in
/// fuse/model/value_traits.hpp.
///
/// DEVICE USE. Every member and function is KOKKOS_INLINE_FUNCTION and calls
/// only Kokkos's device-safe math, so the type can be used inside a kernel
/// (the build refuses a host-only call from device code, see
/// cmake/FuseDependencies.cmake). Nothing throws.
///
/// COMPARISONS look at the value only: a dual number is ordered as the real
/// number it carries, which is what a branch in a differentiated program
/// needs. `==` is therefore NOT a test that two duals are identical; compare
/// `d.gradient(k)` as well when that is what is meant.
///
/// FUNCTIONS WITH A KINK. `abs` and `pow` follow the one-sided derivative of
/// the branch the value lies on. `abs` at 0 takes the branch of
/// +0 (slope +1); `pow(x, p)` at x = 0 is whatever `pow(0, p - 1)` is.

#include <array>
#include <concepts>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "fuse/model/value_traits.hpp"

namespace fuse::num {

template <std::floating_point S, int N>
  requires(N >= 1)
struct dual {
  using scalar_type = S;
  static constexpr int gradient_size = N;

  /// d[0] is the value, d[1 + k] the gradient in direction k.
  S d[N + 1];

  KOKKOS_INLINE_FUNCTION dual() : d{} {}
  /// A constant: the value `x`, no dependence on any parameter.
  KOKKOS_INLINE_FUNCTION dual(S x) : d{} { d[0] = x; }  // NOLINT: implicit

  /// The value `x` of parameter `k`: its own gradient is 1, all others 0.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static dual variable(S x, int k) {
    dual r;
    r.d[0] = x;
    r.d[1 + k] = S(1);
    return r;
  }

  [[nodiscard]] KOKKOS_INLINE_FUNCTION S value() const { return d[0]; }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION S gradient(int k) const {
    return d[1 + k];
  }

  KOKKOS_INLINE_FUNCTION dual& operator+=(const dual& o) {
    for (int i = 0; i <= N; ++i) d[i] += o.d[i];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION dual& operator-=(const dual& o) {
    for (int i = 0; i <= N; ++i) d[i] -= o.d[i];
    return *this;
  }
  KOKKOS_INLINE_FUNCTION dual& operator*=(const dual& o) {
    const S a = d[0];
    const S b = o.d[0];
    d[0] = a * b;
    for (int k = 1; k <= N; ++k) d[k] = a * o.d[k] + d[k] * b;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION dual& operator/=(const dual& o) {
    const S b = o.d[0];
    const S q = d[0] / b;
    d[0] = q;
    for (int k = 1; k <= N; ++k) d[k] = (d[k] - q * o.d[k]) / b;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION dual& operator*=(S s) {
    for (int i = 0; i <= N; ++i) d[i] *= s;
    return *this;
  }
  KOKKOS_INLINE_FUNCTION dual& operator/=(S s) {
    for (int i = 0; i <= N; ++i) d[i] /= s;
    return *this;
  }
};

template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator-(dual<S, N> a) {
  for (int i = 0; i <= N; ++i) a.d[i] = -a.d[i];
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator+(dual<S, N> a) {
  return a;
}

template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator+(dual<S, N> a,
                                                          const dual<S, N>& b) {
  return a += b;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator-(dual<S, N> a,
                                                          const dual<S, N>& b) {
  return a -= b;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator*(dual<S, N> a,
                                                          const dual<S, N>& b) {
  return a *= b;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator/(dual<S, N> a,
                                                          const dual<S, N>& b) {
  return a /= b;
}

// A plain scalar on either side is a constant. Spelled out rather than left to
// the implicit conversion so that `x * s` scales the gradient by `s` directly,
// without forming a constant's zero gradient and multiplying through it.
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator*(dual<S, N> a,
                                                          std::type_identity_t<S> s) {
  return a *= s;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator*(std::type_identity_t<S> s,
                                                          dual<S, N> a) {
  return a *= s;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator/(dual<S, N> a,
                                                          std::type_identity_t<S> s) {
  return a /= s;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator+(dual<S, N> a,
                                                          std::type_identity_t<S> s) {
  a.d[0] += s;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator+(std::type_identity_t<S> s,
                                                          dual<S, N> a) {
  a.d[0] += s;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator-(dual<S, N> a,
                                                          std::type_identity_t<S> s) {
  a.d[0] -= s;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator-(std::type_identity_t<S> s,
                                                          dual<S, N> a) {
  for (int i = 0; i <= N; ++i) a.d[i] = -a.d[i];
  a.d[0] += s;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> operator/(std::type_identity_t<S> s,
                                                          const dual<S, N>& b) {
  return dual<S, N>(s) / b;
}

// Order by value (see the file header).
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool operator<(const dual<S, N>& a,
                                                    const dual<S, N>& b) {
  return a.d[0] < b.d[0];
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool operator>(const dual<S, N>& a,
                                                    const dual<S, N>& b) {
  return a.d[0] > b.d[0];
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool operator<=(const dual<S, N>& a,
                                                     const dual<S, N>& b) {
  return a.d[0] <= b.d[0];
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool operator>=(const dual<S, N>& a,
                                                     const dual<S, N>& b) {
  return a.d[0] >= b.d[0];
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool operator==(const dual<S, N>& a,
                                                     const dual<S, N>& b) {
  return a.d[0] == b.d[0];
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION bool operator!=(const dual<S, N>& a,
                                                     const dual<S, N>& b) {
  return a.d[0] != b.d[0];
}

// ---- elementary functions: f(a) has value f(a0) and gradient f'(a0) times the gradient of a ----

template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> sqrt(dual<S, N> a) {
  const S r = Kokkos::sqrt(a.d[0]);
  const S two_r = S(2) * r;
  a.d[0] = r;
  for (int k = 1; k <= N; ++k) a.d[k] /= two_r;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> exp(dual<S, N> a) {
  const S e = Kokkos::exp(a.d[0]);
  a.d[0] = e;
  for (int k = 1; k <= N; ++k) a.d[k] *= e;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> log(dual<S, N> a) {
  const S x = a.d[0];
  a.d[0] = Kokkos::log(x);
  for (int k = 1; k <= N; ++k) a.d[k] /= x;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> sin(dual<S, N> a) {
  const S x = a.d[0];
  const S c = Kokkos::cos(x);
  a.d[0] = Kokkos::sin(x);
  for (int k = 1; k <= N; ++k) a.d[k] *= c;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> cos(dual<S, N> a) {
  const S x = a.d[0];
  const S ms = -Kokkos::sin(x);
  a.d[0] = Kokkos::cos(x);
  for (int k = 1; k <= N; ++k) a.d[k] *= ms;
  return a;
}
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> abs(dual<S, N> a) {
  if (a.d[0] < S(0)) {
    for (int i = 0; i <= N; ++i) a.d[i] = -a.d[i];
  }
  return a;
}
/// a^p for a constant exponent p.
template <std::floating_point S, int N>
[[nodiscard]] KOKKOS_INLINE_FUNCTION dual<S, N> pow(dual<S, N> a,
                                                       std::type_identity_t<S> p) {
  const S x = a.d[0];
  const S slope = p * Kokkos::pow(x, p - S(1));
  a.d[0] = Kokkos::pow(x, p);
  for (int k = 1; k <= N; ++k) a.d[k] *= slope;
  return a;
}

namespace detail {

/// "fuse:dual<binary64,3>" built at compile time, so each (S, N) gets a unique
/// `value_traits::name` without a string literal per instantiation.
template <class S, int N>
struct DualName {
  static constexpr std::string_view scalar = fuse::value_traits<S>::name;
  static constexpr auto make() {
    std::array<char, 48> a{};
    std::size_t n = 0;
    auto put = [&](std::string_view sv) {
      for (const char ch : sv) a[n++] = ch;
    };
    put("fuse:dual<");
    put(scalar);
    put(",");
    // N is a small positive integer.
    char digits[12] = {};
    int nd = 0;
    for (int v = N; v > 0; v /= 10) digits[nd++] = static_cast<char>('0' + v % 10);
    while (nd > 0) a[n++] = digits[--nd];
    put(">");
    return std::pair{a, n};
  }
  static constexpr auto storage = make();
  static constexpr std::string_view value{storage.first.data(), storage.second};
};

}  // namespace detail
}  // namespace fuse::num

template <std::floating_point S, int N>
struct fuse::value_traits<fuse::num::dual<S, N>> {
  static constexpr std::string_view name = fuse::num::detail::DualName<S, N>::value;
  static constexpr FieldValueKind kind = FieldValueKind::ScalarArray;
  using scalar_type = S;
  static constexpr int scalar_count = N + 1;
  static constexpr int gradient_size = N;
  [[nodiscard]] static double to_binary64(const fuse::num::dual<S, N>& v) noexcept {
    return value_traits<S>::to_binary64(v.d[0]);
  }
  [[nodiscard]] static fuse::num::dual<S, N> from_binary64(double x) noexcept {
    return fuse::num::dual<S, N>(value_traits<S>::from_binary64(x));
  }
};
