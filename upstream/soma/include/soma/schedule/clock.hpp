#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "soma/base/ids.hpp"
#include "soma/base/rational.hpp"
#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"

namespace soma {

/// A common period needing more ticks of the finest interval than this is an
/// unusable schedule; see ClockSet::checked_common_period() and Task 3.2's
/// E-SCHED-PERIOD.
inline constexpr std::int64_t kMaxPeriodTicks = 1'000'000;

/// Smallest L > 0 such that both a and b divide L an integer number of times:
/// lcm(num(a), num(b)) / gcd(den(a), den(b)).
///
/// The lcm is formed in __int128 and range-checked, not computed with
/// std::lcm on int64_t: std::lcm's result is undefined if it is not
/// representable, and negating INT64_MIN to take an absolute value is UB.
/// Same contract as gcd_rational() in rational.hpp -- an unrepresentable
/// result throws rather than silently wrapping, and the check is a throw
/// rather than an assert so it survives -DNDEBUG.
[[nodiscard]] constexpr Rational lcm_rational(const Rational& a,
                                              const Rational& b) {
  if (a.is_zero() || b.is_zero()) {
    throw std::logic_error(
        "lcm_rational: the least common multiple of zero is undefined");
  }
  const detail::Int128 an_wide = a.num();
  const detail::Int128 bn_wide = b.num();
  const detail::Int128 an = an_wide < 0 ? -an_wide : an_wide;
  const detail::Int128 bn = bn_wide < 0 ? -bn_wide : bn_wide;
  // an, bn <= 2^63, so an*bn <= 2^126 and the product cannot overflow Int128.
  const detail::Int128 l = an / std::gcd(an, bn) * bn;
  detail::check_fits_i64(l, "lcm_rational: lcm overflow");
  // gcd of two positive denominators is positive and no larger than either,
  // so this narrowing is always safe.
  return Rational{static_cast<std::int64_t>(l), std::gcd(a.den(), b.den())};
}

/// Spec §8.2: one clock per component; there is no global time step.
///
/// now() is computed as start + step*interval, never accumulated, so it is
/// exact at any step and O(1) to query. Ticks are {start + k*interval : k >= 0}
/// -- the step index only ever moves forward from zero.
class Clock {
 public:
  /// The interval must be positive. Enforced with throw, not assert: a clock
  /// with a zero or negative interval has no well-defined next tick, so a
  /// runtime built on one would spin or run backwards, and an assert would
  /// compile out under -DNDEBUG. validate.hpp already rejects such a component
  /// with E-INTERVAL; this is the same invariant at the scheduling layer.
  explicit Clock(Rational interval, Rational start = Rational{0})
      : interval_(interval), start_(start) {
    if (!(interval_ > Rational{0})) {
      throw std::logic_error("Clock::Clock: interval must be positive, got " +
                             interval_.to_string());
    }
  }

  [[nodiscard]] Rational now() const {
    return start_ + Rational{step_} * interval_;
  }
  [[nodiscard]] Rational next() const {
    return start_ + Rational{step_ + 1} * interval_;
  }
  [[nodiscard]] Rational interval() const noexcept { return interval_; }
  [[nodiscard]] Rational start() const noexcept { return start_; }
  [[nodiscard]] std::int64_t step() const noexcept { return step_; }

  void advance() noexcept { ++step_; }
  void reset() noexcept { step_ = 0; }

  /// Is t an exact tick of this clock, i.e. t == start + k*interval for some
  /// integer k >= 0? A grid-aligned time before the start is not a tick: the
  /// clock never occupies it.
  [[nodiscard]] bool at(Rational t) const {
    const Rational q = (t - start_) / interval_;  // interval_ > 0, never zero
    return q.is_integer() && q >= Rational{0};
  }

 private:
  Rational     interval_;
  Rational     start_;
  std::int64_t step_ = 0;
};

/// The partially ordered set of component clocks the runtime advances.
class ClockSet {
 public:
  explicit ClockSet(const CouplingGraph& g) {
    clocks_.reserve(g.component_count());
    for (std::size_t i = 0; i < g.component_count(); ++i) {
      clocks_.emplace_back(
          g.component(ComponentId{static_cast<std::uint32_t>(i)}).interval);
    }
  }

  /// Bounds-checked in every build: .at() throws std::out_of_range.
  [[nodiscard]] const Clock& clock(ComponentId c) const {
    return clocks_.at(c.value());
  }
  [[nodiscard]] Clock& clock(ComponentId c) { return clocks_.at(c.value()); }
  [[nodiscard]] std::size_t size() const noexcept { return clocks_.size(); }

  [[nodiscard]] Rational earliest_next() const {
    Rational best{0};
    bool first = true;
    for (const Clock& c : clocks_) {
      const Rational n = c.next();
      if (first || n < best) {
        best = n;
        first = false;
      }
    }
    return best;
  }

  [[nodiscard]] std::vector<ComponentId> due_at(Rational t) const {
    std::vector<ComponentId> due;
    for (std::size_t i = 0; i < clocks_.size(); ++i) {
      if (clocks_[i].at(t)) {
        due.push_back(ComponentId{static_cast<std::uint32_t>(i)});
      }
    }
    return due;
  }

  /// Spec §8.2: lockstep is the degenerate case and gets a fast path.
  [[nodiscard]] bool is_lockstep() const {
    if (clocks_.empty()) return true;
    const Rational first = clocks_.front().interval();
    return std::all_of(clocks_.begin(), clocks_.end(),
                       [&](const Clock& c) { return c.interval() == first; });
  }

  /// Shortest span after which every clock is simultaneously at a tick.
  /// Throws std::logic_error if the LCM is not representable; callers that
  /// must not throw use checked_common_period() instead.
  [[nodiscard]] Rational common_period() const {
    if (clocks_.empty()) return Rational{1};
    Rational p = clocks_.front().interval();
    for (const Clock& c : clocks_) p = lcm_rational(p, c.interval());
    return p;
  }

  /// Finest interval in the set; the slab grid resolution.
  [[nodiscard]] Rational finest_interval() const {
    if (clocks_.empty()) return Rational{1};
    Rational f = clocks_.front().interval();
    for (const Clock& c : clocks_) f = gcd_rational(f, c.interval());
    return f;
  }

  /// common_period() with the guard the schedule actually needs: a period
  /// spanning more than kMaxPeriodTicks ticks of the finest interval, or one
  /// that is not representable at all, is reported as E-SCHED-PERIOD rather
  /// than handed downstream to build an unusable slab decomposition.
  /// Never throws.
  [[nodiscard]] Result<Rational> checked_common_period() const {
    Rational period{1};
    Rational ticks{1};
    try {
      period = common_period();
      ticks = period / finest_interval();
    } catch (const std::logic_error& e) {
      return Error{"E-SCHED-PERIOD",
                   std::string("common period is not representable: ") +
                       e.what()};
    }
    if (ticks > Rational{kMaxPeriodTicks}) {
      return Error{"E-SCHED-PERIOD",
                   "common period " + period.to_string() + " spans " +
                       ticks.to_string() +
                       " ticks of the finest interval, exceeding the limit of " +
                       std::to_string(kMaxPeriodTicks)};
    }
    return period;
  }

 private:
  std::vector<Clock> clocks_;
};

}  // namespace soma
