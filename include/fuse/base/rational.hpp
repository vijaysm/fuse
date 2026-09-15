#pragma once

#include <compare>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace fuse {

namespace detail {
// __int128 is a compiler builtin (GCC/Clang), not a standard type, so
// -Wpedantic warns about it unless the extension is explicitly acknowledged.
// This keeps the warning flags intact (per project policy) while still using
// the type as an arithmetic intermediate.
__extension__ typedef __int128 Int128;

/// Shared by Rational's member operators and gcd_rational(): true if a
/// 128-bit value narrows losslessly to int64_t.
constexpr bool fits_i64(Int128 v) noexcept {
  constexpr Int128 kMin = std::numeric_limits<std::int64_t>::min();
  constexpr Int128 kMax = std::numeric_limits<std::int64_t>::max();
  return v >= kMin && v <= kMax;
}

/// Enforced unconditionally, not via assert(): narrowing a value that does
/// not fit int64_t is well-defined-but-wrong under C++20 (modulo
/// wraparound), not a crash, so an assert()-only guard would compile out
/// under -DNDEBUG and let a genuinely unrepresentable time value (e.g. a gcd
/// of |INT64_MIN| == 2^63) silently narrow to the wrong int64_t instead of
/// being rejected. Same reasoning as OperatorDAG::add_node's SSA check.
constexpr void check_fits_i64(Int128 v, const char* what) {
  if (!fits_i64(v)) {
    throw std::logic_error(what);
  }
}
}  // namespace detail

/// Exact rational time. Spec §4.3: floating-point time accumulation would
/// destroy slab decomposition over long runs, so every time quantity in FUSE
/// is a Rational.
///
/// Magnitude invariant: num_/den_ is always kept in lowest terms with
/// den_ > 0 (or den_ == 1 when num_ == 0). Arithmetic never overflows
/// int64_t *before* reduction: cross-multiplied numerator/denominator
/// products are formed in a 128-bit intermediate (`detail::Int128`), gcd
/// reduction happens in that same intermediate, and only the fully-reduced
/// result is narrowed back to int64_t. This matters because FUSE composes
/// coupling intervals with unrelated denominators (1/3, 1/7, 1/11, ...): the
/// scheduler's `den_ *= o.den_` step can transiently exceed 2^63 even though
/// every *value* FUSE ever represents (simulated time in reasonable units)
/// fits comfortably once reduced. Comparisons (`operator<=>`) never reduce,
/// so they are compared directly as 128-bit products with no narrowing at
/// all — a comparison always has a well-defined answer regardless of
/// magnitude.
class Rational {
 public:
  constexpr Rational() = default;
  constexpr Rational(std::int64_t n) : num_(n), den_(1) {}  // NOLINT: implicit

  constexpr Rational(std::int64_t n, std::int64_t d) {
    if (d == 0) {
      throw std::logic_error("Rational::Rational: denominator must be non-zero");
    }
    const Reduced128 r = reduce(n, d);
    detail::check_fits_i64(r.n, "Rational::Rational: numerator overflow after reduction");
    detail::check_fits_i64(r.d, "Rational::Rational: denominator overflow after reduction");
    num_ = static_cast<std::int64_t>(r.n);
    den_ = static_cast<std::int64_t>(r.d);
  }

  [[nodiscard]] constexpr std::int64_t num() const noexcept { return num_; }
  [[nodiscard]] constexpr std::int64_t den() const noexcept { return den_; }

  [[nodiscard]] constexpr bool is_zero() const noexcept { return num_ == 0; }
  [[nodiscard]] constexpr bool is_integer() const noexcept { return den_ == 1; }
  [[nodiscard]] constexpr double to_double() const noexcept {
    return static_cast<double>(num_) / static_cast<double>(den_);
  }

  constexpr Rational& operator+=(const Rational& o) {
    const detail::Int128 n = static_cast<detail::Int128>(num_) * o.den_ +
                              static_cast<detail::Int128>(o.num_) * den_;
    const detail::Int128 d = static_cast<detail::Int128>(den_) * o.den_;
    const Reduced128 r = reduce(n, d);
    detail::check_fits_i64(r.n, "Rational::operator+=: numerator overflow after reduction");
    detail::check_fits_i64(r.d, "Rational::operator+=: denominator overflow after reduction");
    num_ = static_cast<std::int64_t>(r.n);
    den_ = static_cast<std::int64_t>(r.d);
    return *this;
  }
  constexpr Rational& operator-=(const Rational& o) {
    const detail::Int128 n = static_cast<detail::Int128>(num_) * o.den_ -
                              static_cast<detail::Int128>(o.num_) * den_;
    const detail::Int128 d = static_cast<detail::Int128>(den_) * o.den_;
    const Reduced128 r = reduce(n, d);
    detail::check_fits_i64(r.n, "Rational::operator-=: numerator overflow after reduction");
    detail::check_fits_i64(r.d, "Rational::operator-=: denominator overflow after reduction");
    num_ = static_cast<std::int64_t>(r.n);
    den_ = static_cast<std::int64_t>(r.d);
    return *this;
  }
  constexpr Rational& operator*=(const Rational& o) {
    const detail::Int128 n = static_cast<detail::Int128>(num_) * o.num_;
    const detail::Int128 d = static_cast<detail::Int128>(den_) * o.den_;
    const Reduced128 r = reduce(n, d);
    detail::check_fits_i64(r.n, "Rational::operator*=: numerator overflow after reduction");
    detail::check_fits_i64(r.d, "Rational::operator*=: denominator overflow after reduction");
    num_ = static_cast<std::int64_t>(r.n);
    den_ = static_cast<std::int64_t>(r.d);
    return *this;
  }
  constexpr Rational& operator/=(const Rational& o) {
    if (o.num_ == 0) {
      throw std::logic_error("Rational::operator/=: division by zero Rational");
    }
    const detail::Int128 n = static_cast<detail::Int128>(num_) * o.den_;
    const detail::Int128 d = static_cast<detail::Int128>(den_) * o.num_;
    const Reduced128 r = reduce(n, d);
    detail::check_fits_i64(r.n, "Rational::operator/=: numerator overflow after reduction");
    detail::check_fits_i64(r.d, "Rational::operator/=: denominator overflow after reduction");
    num_ = static_cast<std::int64_t>(r.n);
    den_ = static_cast<std::int64_t>(r.d);
    return *this;
  }

  friend constexpr Rational operator+(Rational a, const Rational& b) { return a += b; }
  friend constexpr Rational operator-(Rational a, const Rational& b) { return a -= b; }
  friend constexpr Rational operator*(Rational a, const Rational& b) { return a *= b; }
  friend constexpr Rational operator/(Rational a, const Rational& b) { return a /= b; }

  friend constexpr bool operator==(const Rational& a, const Rational& b) noexcept {
    return a.num_ == b.num_ && a.den_ == b.den_;  // always normalised
  }
  friend constexpr std::strong_ordering operator<=>(const Rational& a,
                                                    const Rational& b) noexcept {
    // Compared directly as 128-bit products: a comparison always has a
    // well-defined answer, so there is nothing to reduce or assert here.
    const detail::Int128 lhs = static_cast<detail::Int128>(a.num_) * b.den_;
    const detail::Int128 rhs = static_cast<detail::Int128>(b.num_) * a.den_;
    return lhs <=> rhs;
  }

  /// Finest tick that divides both denominators: 1/lcm(den(a), den(b)).
  [[nodiscard]] static constexpr Rational lcm_denominator(const Rational& a,
                                                          const Rational& b) {
    return Rational{1, std::lcm(a.den_, b.den_)};
  }

  [[nodiscard]] std::string to_string() const {
    if (den_ == 1) return std::to_string(num_);
    return std::to_string(num_) + "/" + std::to_string(den_);
  }

 private:
  struct Reduced128 {
    detail::Int128 n;
    detail::Int128 d;
  };

  /// Reduce (n, d) to lowest terms entirely in a 128-bit intermediate.
  /// Negation and the gcd's absolute value happen in Int128, so INT64_MIN is
  /// never negated in int64_t — the classic UB this class used to hit.
  /// Callers check the result fits int64_t (detail::check_fits_i64) and
  /// throw if it doesn't; that is a programmer error (a genuinely
  /// unrepresentable time value), not something to silently narrow.
  static constexpr Reduced128 reduce(detail::Int128 n, detail::Int128 d) {
    if (d < 0) {
      n = -n;
      d = -d;
    }
    const detail::Int128 an = n < 0 ? -n : n;
    const detail::Int128 g = std::gcd(an, d);
    if (g > 1) {
      n /= g;
      d /= g;
    }
    if (n == 0) d = 1;
    return {n, d};
  }

  std::int64_t num_ = 0;
  std::int64_t den_ = 1;
};

/// Largest g such that a and b are both integer multiples of g.
/// gcd(a/b, c/d) = gcd(a,c) / lcm(b,d). Used by §8.3 slab decomposition.
[[nodiscard]] constexpr Rational gcd_rational(const Rational& a,
                                              const Rational& b) {
  // Absolute value in Int128 first: negating INT64_MIN directly in int64_t
  // is UB, same hazard as the one fixed in Rational::reduce().
  const detail::Int128 an_wide = a.num();
  const detail::Int128 bn_wide = b.num();
  const detail::Int128 g =
      std::gcd(an_wide < 0 ? -an_wide : an_wide, bn_wide < 0 ? -bn_wide : bn_wide);
  // Same overflow contract as every Rational member operator: |INT64_MIN|
  // == 2^63 does not fit in int64_t (max 2^63 - 1), so narrowing it silently
  // would wrap to INT64_MIN. Enforced unconditionally (see check_fits_i64),
  // consistent with reduce()'s callers -- an assert()-only guard here would
  // vanish under -DNDEBUG and let the wrap through in Release.
  detail::check_fits_i64(g, "gcd_rational: gcd overflow (INT64_MIN input?)");
  // The denominator lcm must also be formed in Int128. std::lcm's result is
  // undefined when it does not fit the common type, and int64_t denominators
  // reach that at roughly 3e9 each: std::lcm(3037000493, 3037000507) wraps to
  // a *negative* int64_t, which Rational's constructor then normalises into a
  // negative rational -- i.e. gcd_rational of two positive intervals silently
  // returns a negative one, and ClockSet::finest_interval() hands that to the
  // slab decomposition. Widen, then apply the same unconditional narrowing
  // check used for the numerator.
  const detail::Int128 ad = a.den();
  const detail::Int128 bd = b.den();
  const detail::Int128 dg = std::gcd(ad, bd);
  const detail::Int128 l = (ad / dg) * bd;  // divide first: no needless growth
  detail::check_fits_i64(l, "gcd_rational: denominator lcm overflow");
  return Rational{static_cast<std::int64_t>(g), static_cast<std::int64_t>(l)};
}

}  // namespace fuse
