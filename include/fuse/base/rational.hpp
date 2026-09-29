#pragma once

#include <compare>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <string>

#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"

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
    FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW", "{}", what);
  }
}
}  // namespace detail

/// The C-ABI shape of a Rational: two int64 in the order every ABI record,
/// the byte writer and the Fortran shim agree on. Layout-identical to the
/// spine's `karma_exact_time` and to the `fuse_time` the coupled-process ABI
/// will declare, so a Rational crosses the boundary as this and nothing else.
struct ExactTimeRecord {
  std::int64_t numerator = 0;
  std::int64_t denominator = 1;
  friend constexpr bool operator==(const ExactTimeRecord&, const ExactTimeRecord&) = default;
};
static_assert(sizeof(ExactTimeRecord) == 16 && alignof(ExactTimeRecord) == 8);

/// Exact rational time. Spec §4.3: floating-point time accumulation would
/// destroy the decomposition into common-refinement intervals over long runs, so every time quantity in FUSE
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
      FUSE_THROW(InvalidArgument, "Rational::Rational: denominator must be non-zero");
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

  /// --- exact arithmetic, in two spellings ---------------------------------
  ///
  /// Each operation is implemented ONCE, in the `try_` form, which reports
  /// failure by returning nothing. The operators are thin wrappers that reject
  /// when it does. Two spellings and one implementation, so the overflow rule
  /// cannot drift between them.
  ///
  /// WHY THE NON-THROWING FORM EXISTS. Several callers must turn an
  /// unrepresentable result into a `Result` or a diagnostic rather than let it
  /// propagate -- `ClockSet::checked_common_period()` is documented never to
  /// throw, and the unit-compatibility stages report an unrepresentable
  /// conversion as a rejection on the field transfer. Those callers used to wrap the
  /// throwing spelling in `try`/`catch`, which made an exception part of the
  /// library's INTERNAL control flow: with `FUSE_ENABLE_EXCEPTIONS=OFF` a
  /// rejection aborts instead of unwinding, so the catch never ran and the
  /// process died inside a test named for the opposite contract
  /// (`Expand.UnrepresentableCommonPeriodIsAnErrorNotAnException`). A path
  /// whose job is to produce a value should never have been throwing.

  /// The one place a 128-bit intermediate becomes a Rational. Reduces first,
  /// then reports whether the reduced result fits -- matching the operators'
  /// contract that overflow is judged AFTER reduction, since 2^64/2^64 is 1.
  [[nodiscard]] static constexpr std::optional<Rational> from_i128(
      detail::Int128 n, detail::Int128 d) noexcept {
    const Reduced128 r = reduce(n, d);
    if (!detail::fits_i64(r.n) || !detail::fits_i64(r.d)) return std::nullopt;
    Rational out;
    out.num_ = static_cast<std::int64_t>(r.n);
    out.den_ = static_cast<std::int64_t>(r.d);
    return out;
  }

  [[nodiscard]] static constexpr std::optional<Rational> try_add(
      const Rational& a, const Rational& b) noexcept {
    return from_i128(static_cast<detail::Int128>(a.num_) * b.den_ +
                         static_cast<detail::Int128>(b.num_) * a.den_,
                     static_cast<detail::Int128>(a.den_) * b.den_);
  }
  [[nodiscard]] static constexpr std::optional<Rational> try_sub(
      const Rational& a, const Rational& b) noexcept {
    return from_i128(static_cast<detail::Int128>(a.num_) * b.den_ -
                         static_cast<detail::Int128>(b.num_) * a.den_,
                     static_cast<detail::Int128>(a.den_) * b.den_);
  }
  [[nodiscard]] static constexpr std::optional<Rational> try_mul(
      const Rational& a, const Rational& b) noexcept {
    return from_i128(static_cast<detail::Int128>(a.num_) * b.num_,
                     static_cast<detail::Int128>(a.den_) * b.den_);
  }
  /// Division by zero reports failure here rather than throwing, because a
  /// caller using this spelling has said it wants an answer it can test. The
  /// operator still throws `InvalidArgument`, which is a different fault from
  /// overflow and stays distinguishable.
  [[nodiscard]] static constexpr std::optional<Rational> try_div(
      const Rational& a, const Rational& b) noexcept {
    if (b.num_ == 0) return std::nullopt;
    return from_i128(static_cast<detail::Int128>(a.num_) * b.den_,
                     static_cast<detail::Int128>(a.den_) * b.num_);
  }

  constexpr Rational& operator+=(const Rational& o) {
    const auto r = try_add(*this, o);
    if (!r) {
      FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW",
                  "Rational::operator+=: overflow after reduction");
    }
    return *this = *r;
  }
  constexpr Rational& operator-=(const Rational& o) {
    const auto r = try_sub(*this, o);
    if (!r) {
      FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW",
                  "Rational::operator-=: overflow after reduction");
    }
    return *this = *r;
  }
  constexpr Rational& operator*=(const Rational& o) {
    const auto r = try_mul(*this, o);
    if (!r) {
      FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW",
                  "Rational::operator*=: overflow after reduction");
    }
    return *this = *r;
  }
  constexpr Rational& operator/=(const Rational& o) {
    if (o.num_ == 0) {
      FUSE_THROW(InvalidArgument, "Rational::operator/=: division by zero Rational");
    }
    const auto r = try_div(*this, o);
    if (!r) {
      FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW",
                  "Rational::operator/=: overflow after reduction");
    }
    return *this = *r;
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
  /// REVIEW.md R-15: std::lcm(a.den_, b.den_) computed directly in int64_t is
  /// the exact overflow try_gcd_rational's own denominator lcm was fixed for
  /// (std::lcm(3037000493, 3037000507) wraps to a negative int64_t, and
  /// Rational{1, <negative>} normalises the sign onto the numerator, so this
  /// silently returned a small NEGATIVE Rational for "the finest tick", a
  /// positive quantity by construction, with no diagnostic). Same Int128
  /// route as try_gcd_rational's denominator lcm, rejecting rather than
  /// admitting a wrapped result.
  [[nodiscard]] static constexpr Rational lcm_denominator(const Rational& a,
                                                          const Rational& b) {
    const detail::Int128 ad = a.den_;
    const detail::Int128 bd = b.den_;
    const detail::Int128 dg = std::gcd(ad, bd);
    const detail::Int128 l = (ad / dg) * bd;  // divide first: no needless growth
    detail::check_fits_i64(l, "Rational::lcm_denominator: denominator lcm overflow");
    return Rational{1, static_cast<std::int64_t>(l)};
  }

  [[nodiscard]] std::string to_string() const {
    if (den_ == 1) return std::to_string(num_);
    return std::to_string(num_) + "/" + std::to_string(den_);
  }

  /// To the ABI record. Always reduced with a positive denominator, because
  /// this object always is.
  [[nodiscard]] constexpr ExactTimeRecord to_record() const noexcept { return {num_, den_}; }

  /// From an ABI record produced by foreign code. The contract says reduced
  /// and positive-denominator; a record that is not is a contract violation
  /// on the producer's side and is rejected rather than repaired (I4) -- a
  /// coupler that quietly reduces 2/4 hides a bug in the code it is coupling.
  [[nodiscard]] static Result<Rational> from_record(ExactTimeRecord rec) {
    if (rec.denominator <= 0) {
      return Failure{"E-BASE-TIME-DENOMINATOR",
                     "exact time record has non-positive denominator " + std::to_string(rec.denominator)};
    }
    const Rational reduced{rec.numerator, rec.denominator};
    if (reduced.num_ != rec.numerator || reduced.den_ != rec.denominator) {
      return Failure{"E-BASE-TIME-UNREDUCED",
                     "exact time record " + std::to_string(rec.numerator) + "/" +
                         std::to_string(rec.denominator) + " is not in lowest terms"};
    }
    return reduced;
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
/// gcd(a/b, c/d) = gcd(a,c) / lcm(b,d). Used by §8.3's decomposition into
/// common-refinement intervals.
/// Largest g such that a and b are both integer multiples of g, or nothing
/// when that value is not representable. The throwing spelling below rejects
/// on the same condition; this one exists for callers that must report rather
/// than propagate (see Rational::try_add).
[[nodiscard]] constexpr std::optional<Rational> try_gcd_rational(
    const Rational& a, const Rational& b) noexcept {
  // Absolute value in Int128 first: negating INT64_MIN directly in int64_t
  // is UB, same hazard as the one fixed in Rational::reduce().
  const detail::Int128 an_wide = a.num();
  const detail::Int128 bn_wide = b.num();
  const detail::Int128 g =
      std::gcd(an_wide < 0 ? -an_wide : an_wide, bn_wide < 0 ? -bn_wide : bn_wide);
  // |INT64_MIN| == 2^63 does not fit in int64_t (max 2^63 - 1), so narrowing
  // it silently would wrap back to INT64_MIN.
  if (!detail::fits_i64(g)) return std::nullopt;
  // The denominator lcm must also be formed in Int128. std::lcm's result is
  // undefined when it does not fit the common type, and int64_t denominators
  // reach that at roughly 3e9 each: std::lcm(3037000493, 3037000507) wraps to
  // a *negative* int64_t, which Rational's constructor then normalises into a
  // negative rational -- i.e. gcd_rational of two positive intervals silently
  // returns a negative one, and ClockSet::finest_interval() hands that to the
  // decomposition into common-refinement intervals.
  const detail::Int128 ad = a.den();
  const detail::Int128 bd = b.den();
  const detail::Int128 dg = std::gcd(ad, bd);
  const detail::Int128 l = (ad / dg) * bd;  // divide first: no needless growth
  if (!detail::fits_i64(l)) return std::nullopt;
  return Rational{static_cast<std::int64_t>(g), static_cast<std::int64_t>(l)};
}

/// gcd(a/b, c/d) = gcd(a,c) / lcm(b,d). Used by §8.3's decomposition into
/// common-refinement intervals.
/// Rejects where try_gcd_rational() reports nothing.
[[nodiscard]] constexpr Rational gcd_rational(const Rational& a,
                                              const Rational& b) {
  const auto g = try_gcd_rational(a, b);
  if (!g) {
    FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW",
                "gcd_rational: gcd or denominator lcm overflow (INT64_MIN input?)");
  }
  return *g;
}


}  // namespace fuse
