#pragma once

#include <cstdint>
#include <string>

#include "fuse/base/error.hpp"

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
  std::int8_t luminosity = 0;   // cd

  friend constexpr bool operator==(const Units&, const Units&) = default;

  [[nodiscard]] constexpr Units operator*(const Units& o) const {
    using detail::narrow_exponent;
    return Units{narrow_exponent(length + o.length),
                 narrow_exponent(mass + o.mass),
                 narrow_exponent(time + o.time),
                 narrow_exponent(current + o.current),
                 narrow_exponent(temperature + o.temperature),
                 narrow_exponent(amount + o.amount),
                 narrow_exponent(luminosity + o.luminosity)};
  }

  [[nodiscard]] constexpr Units operator/(const Units& o) const {
    using detail::narrow_exponent;
    return Units{narrow_exponent(length - o.length),
                 narrow_exponent(mass - o.mass),
                 narrow_exponent(time - o.time),
                 narrow_exponent(current - o.current),
                 narrow_exponent(temperature - o.temperature),
                 narrow_exponent(amount - o.amount),
                 narrow_exponent(luminosity - o.luminosity)};
  }

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
