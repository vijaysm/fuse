#pragma once

#include <string>

#include "fuse/base/rational.hpp"

namespace fuse {

enum class Anchor { WindowStart, WindowEnd, Absolute };

[[nodiscard]] constexpr const char* to_cstring(Anchor a) noexcept {
  switch (a) {
    case Anchor::WindowStart: return "WindowStart";
    case Anchor::WindowEnd:   return "WindowEnd";
    case Anchor::Absolute:    return "Absolute";
  }
  return "Unknown";
}

/// The time at which a value lives: t_ref + offset*dt, from iterate k.
/// Spec §4.3. Every dataflow-graph edge carries one; mismatches are errors
/// raised while compiling into the dataflow graph.
struct TimeLevel {
  Anchor   anchor = Anchor::WindowEnd;
  Rational offset{0};
  int      iterate = 0;  // 0 = current, -1 = previous iterate

  friend bool operator==(const TimeLevel&, const TimeLevel&) = default;

  [[nodiscard]] static TimeLevel current() { return TimeLevel{}; }

  [[nodiscard]] static TimeLevel lagged(int steps) {
    return TimeLevel{Anchor::WindowEnd, Rational{-steps}, 0};
  }

  [[nodiscard]] static TimeLevel previous_iterate() {
    return TimeLevel{Anchor::WindowEnd, Rational{0}, -1};
  }

  [[nodiscard]] std::string to_string() const {
    std::string s = "[";
    s += to_cstring(anchor);
    if (!offset.is_zero()) {
      if (offset > Rational{0}) s += "+";
      s += offset.to_string();
    }
    s += ", k";
    if (iterate != 0) {
      s += (iterate < 0) ? "-" : "+";
      s += std::to_string(iterate < 0 ? -iterate : iterate);
    }
    s += "]";
    return s;
  }
};

/// Strict by default (spec §4.3). A deliberate mismatch is authorised on the
/// field transfer via allow_time_level_mismatch(reason), not by relaxing this.
[[nodiscard]] inline bool compatible(const TimeLevel& produced,
                                     const TimeLevel& required) noexcept {
  return produced == required;
}

}  // namespace fuse
