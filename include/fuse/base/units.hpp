#pragma once

#include <cstdint>
#include <string>

#include "fuse/base/error.hpp"
#include "fuse/base/rational.hpp"

namespace fuse {

namespace detail {
/// Exponent sums/differences are computed in `int` (promoted from int8_t)
/// and only narrowed back to int8_t once. An exponent outside int8_t is
/// unrepresentable, and an unrepresentable unit must be refused, not wrapped
/// (I4) -- in every build (I6). Upstream had an assert here, which vanished
/// under NDEBUG and let a Release build wrap silently.
constexpr std::int8_t narrow_exponent(int v) {
  if (v < -128 || v > 127) {
    FUSE_REFUSE("E-BASE-UNITS-EXPONENT", "dimension exponent {} does not fit int8_t", v);
  }
  return static_cast<std::int8_t>(v);
}
}  // namespace detail

/// SI dimension exponents. Spec §4.2: exponent vectors compose and compare;
/// unit strings do not.
struct Units {
  std::int8_t length = 0;       // m
  std::int8_t mass = 0;         // kg
  std::int8_t time = 0;         // s
  std::int8_t current = 0;      // A
  std::int8_t temperature = 0;  // K
  std::int8_t amount = 0;       // mol
  std::int8_t luminosity = 0;

  /// Affine part, merged in from the spine's unit type. A quantity's unit is
  /// `scale * (value) + offset` in SI base units, EXACTLY: kilopascal is
  /// `scale = 1000`, degrees Celsius is `scale = 1, offset = 27315/100`. Exact
  /// rationals, not doubles, because a unit conversion applied every coupling
  /// step must not drift, and because the descriptor's canonical bytes must be
  /// reproducible across compilers (I1's reasoning, applied to units).
  ///
  /// Dimensional CHECKING ignores these (`same_dimensions`); a mismatch in them
  /// is a transform to infer, not an incompatibility.
  Rational scale{1};
  Rational offset{0};   // cd

  friend constexpr bool operator==(const Units&, const Units&) = default;

  /// Scales multiply along with the dimensions. An ADDITIVE offset makes the
  /// operation meaningless -- degrees Celsius squared is not a quantity -- so an
  /// affine operand is refused rather than having its offset silently dropped,
  /// which would be a factor error no dimensional check can see.
  [[nodiscard]] constexpr Units operator*(const Units& o) const {
    using detail::narrow_exponent;
    if (!is_linear() || !o.is_linear()) {
      FUSE_REFUSE("E-BASE-UNITS-AFFINE", "cannot multiply an affine unit (offsets {} and {})",
                  offset.to_string(), o.offset.to_string());
    }
    return Units{narrow_exponent(length + o.length),
                 narrow_exponent(mass + o.mass),
                 narrow_exponent(time + o.time),
                 narrow_exponent(current + o.current),
                 narrow_exponent(temperature + o.temperature),
                 narrow_exponent(amount + o.amount),
                 narrow_exponent(luminosity + o.luminosity),
                 scale * o.scale,
                 Rational{0}};
  }

  [[nodiscard]] constexpr Units operator/(const Units& o) const {
    using detail::narrow_exponent;
    if (!is_linear() || !o.is_linear()) {
      FUSE_REFUSE("E-BASE-UNITS-AFFINE", "cannot divide an affine unit (offsets {} and {})",
                  offset.to_string(), o.offset.to_string());
    }
    return Units{narrow_exponent(length - o.length),
                 narrow_exponent(mass - o.mass),
                 narrow_exponent(time - o.time),
                 narrow_exponent(current - o.current),
                 narrow_exponent(temperature - o.temperature),
                 narrow_exponent(amount - o.amount),
                 narrow_exponent(luminosity - o.luminosity),
                 scale / o.scale,
                 Rational{0}};
  }

  /// Exponents only: what "are these the same physical quantity?" means.
  [[nodiscard]] constexpr bool same_dimensions(const Units& o) const noexcept {
    return length == o.length && mass == o.mass && time == o.time && current == o.current &&
           temperature == o.temperature && amount == o.amount && luminosity == o.luminosity;
  }

  /// True when the unit is a pure scaling of SI base units (no additive offset),
  /// which is the only case where multiplication and division are meaningful.
  [[nodiscard]] constexpr bool is_linear() const noexcept { return offset == Rational{0}; }

  [[nodiscard]] constexpr bool is_dimensionless() const noexcept {
    return *this == Units{};
  }

  [[nodiscard]] std::string to_string() const {
    struct Entry { std::int8_t exp; const char* sym; };
    const Entry entries[] = {{mass, "kg"},        {length, "m"},
                             {time, "s"},         {current, "A"},
                             {temperature, "K"},  {amount, "mol"},
                             {luminosity, "cd"}};
    std::string out;
    for (const auto& e : entries) {
      if (e.exp == 0) continue;
      if (!out.empty()) out += "·";  // middle dot
      out += e.sym;
      if (e.exp != 1) {
        out += "^";
        out += std::to_string(static_cast<int>(e.exp));
      }
    }
    return out.empty() ? std::string{"1"} : out;
  }
};

namespace units {

inline constexpr Units kDimensionless{};
inline constexpr Units kMetre{.length = 1};
inline constexpr Units kKilogram{.mass = 1};
inline constexpr Units kSecond{.time = 1};
inline constexpr Units kKelvin{.temperature = 1};
inline constexpr Units kTemperature = kKelvin;

inline constexpr Units kNewton{.length = 1, .mass = 1, .time = -2};
inline constexpr Units kJoule{.length = 2, .mass = 1, .time = -2};
inline constexpr Units kWatt{.length = 2, .mass = 1, .time = -3};
inline constexpr Units kPascal{.length = -1, .mass = 1, .time = -2};
inline constexpr Units kHeatFluxDensity{.mass = 1, .time = -3};  // W/m^2

}  // namespace units
}  // namespace fuse
