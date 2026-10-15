#pragma once

#include <algorithm>
#include <optional>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/base/error.hpp"
#include "fuse/schedule/window.hpp"

namespace fuse {

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
/// Shortest value of which both a and b are integer multiples, or nothing
/// when it is not representable or undefined. Callers that must not throw use
/// this; `lcm_rational` below refuses on the same conditions.
[[nodiscard]] constexpr std::optional<Rational> try_lcm_rational(
    const Rational& a, const Rational& b) noexcept {
  if (a.is_zero() || b.is_zero()) return std::nullopt;
  const detail::Int128 an_wide = a.num();
  const detail::Int128 bn_wide = b.num();
  const detail::Int128 an = an_wide < 0 ? -an_wide : an_wide;
  const detail::Int128 bn = bn_wide < 0 ? -bn_wide : bn_wide;
  // an, bn <= 2^63, so an*bn <= 2^126 and the product cannot overflow Int128.
  // detail::gcd128, not std::gcd: see rational.hpp's gcd128 comment -- nvcc's
  // host pass does not consider __int128 integral, so std::gcd's own
  // static_assert rejects it.
  const detail::Int128 l = an / detail::gcd128(an, bn) * bn;
  if (!detail::fits_i64(l)) return std::nullopt;
  // gcd of two positive denominators is positive and no larger than either,
  // so this narrowing is always safe.
  return Rational{static_cast<std::int64_t>(l), std::gcd(a.den(), b.den())};
}

[[nodiscard]] constexpr Rational lcm_rational(const Rational& a,
                                              const Rational& b) {
  if (a.is_zero() || b.is_zero()) {
    FUSE_THROW(InvalidArgument, "{}",
        "lcm_rational: the least common multiple of zero is undefined");
  }
  const auto l = try_lcm_rational(a, b);
  if (!l) {
    FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW", "lcm_rational: lcm overflow");
  }
  return *l;
}

/// Spec Section 8.2: one clock per component; there is no global time step.
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
      FUSE_THROW(InvalidArgument, "{}","Clock::Clock: interval must be positive, got " +
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
  /// Spec: never a least-common-multiple window (the user's own standing
  /// ruling, KNOWN_ISSUES.md and the plan -- a coupling window exists
  /// precisely so two components' several substeps are coupled through
  /// the temporal interpolant, exactly or conservatively, instead of
  /// waiting for their LCM). A window-scoped member's own ComponentDecl::
  /// interval is its FINE subcycling rate -- read directly from the
  /// graph by `Runner::Impl::advance_window()`'s own `WindowMemberConfig::
  /// interval` (src/runtime/iteration.cpp), never through this class --
  /// and must never also stand in as this member's CLOCK for scheduling
  /// purposes, or `checked_common_period()`'s own LCM reintroduces
  /// exactly the rejected rule: a window whose length does not divide
  /// lcm(every member's own fine interval) would be refused
  /// (E-SCHED-PERIOD) even though `advance_window()` has no such
  /// requirement at all (an `Adjustable` member truncates its last
  /// substep at the window end; a `Fixed` member straddles and replays
  /// -- window_coupler.hpp's own class comment). A window-scoped
  /// member's own CLOCK is instead its region's own WINDOW LENGTH: the
  /// member ticks, from the rest of the compiled schedule's point of
  /// view, once per window instance (`Runner::Impl::build_ticks()`'s own
  /// `window_member_components()` already gives it no ordinary Tick at
  /// all, at any interval -- this is the same exemption, applied to the
  /// clock the PERIOD is computed from, not only to the tick list).
  ///
  /// A `ComponentKind::Coupler` entry gets a `Clock` object too (kept
  /// dense so `ComponentId` indexing into `clocks_` stays valid -- the
  /// `clock(ComponentId)` accessor and `size() == component_count()`
  /// both depend on that), but `coupler_[i]` records that it is one, and
  /// every aggregate below (`common_period()`, `finest_interval()`,
  /// `checked_common_period()`, `is_lockstep()`, `earliest_next()`,
  /// `due_at()`) skips it. docs/dev/coupler-placement.md's own table
  /// claimed this class was "coupler-safe by construction" on the
  /// grounds that nothing ever calls `advance()` on a coupler's clock
  /// (true -- `build_ticks()` never schedules it a `Tick`) -- but every
  /// one of the aggregates folds EVERY clock_'s own `interval()` into
  /// its answer regardless of whether `advance()` is ever called on it,
  /// and `ComponentDecl::interval` defaults to `Rational{1}` (a coupler
  /// declares no interval of its own at all). A coupler added to a
  /// window-scoped graph whose own window is shorter than 1 (true of
  /// the Brusselator's own 1/50, or any window under a second) silently
  /// pulled `lcm(window, 1) = 1` into the compiled period -- the
  /// Brusselator's own window re-ran 500 times before
  /// `is_coupling_ongoing()`'s period/now ever advanced once, since
  /// nothing else in the graph needed that many window instances to
  /// finish a period. Found and fixed this session (Conformance Unit 2)
  /// -- the two coupler tests already in the suite
  /// (window_runner_test.cpp's own WindowChannelRemapRoutedToCoupler
  /// MatchesIdentityClosedForm, layout_driver) never saw it because
  /// their own physics intervals already happen to be exactly 1.
  explicit ClockSet(const CouplingGraph& g) {
    clocks_.reserve(g.component_count());
    coupler_.reserve(g.component_count());
    for (std::uint32_t i = 0; i < g.component_count(); ++i) {
      const ComponentId cid{i};
      const ComponentDecl& comp = g.component(cid);
      Rational interval = comp.interval;
      for (std::uint32_t r = 0; r < g.iteration_region_count(); ++r) {
        const IterationRegionDecl& decl =
            g.iteration_region(IterationRegionIdx{r});
        if (!decl.window_scope.has_value()) continue;
        const std::vector<ComponentId> members = window_region_members(g, decl);
        if (std::find(members.begin(), members.end(), cid) == members.end()) {
          continue;
        }
        const Result<Rational> wl = window_length(g, *decl.window_scope);
        if (!wl.ok()) {
          FUSE_THROW(InvalidArgument, "{}",
                     "ClockSet::ClockSet: component '" + g.component(cid).name +
                         "' belongs to window-scoped region '" + decl.name +
                         "' whose own window length could not be resolved: " +
                         wl.error().code + ": " + wl.error().message);
        }
        interval = wl.value();
        break;  // a component declared in more than one window region is
                // not a configuration anything else here supports either.
      }
      clocks_.emplace_back(interval);
      coupler_.push_back(comp.kind == ComponentKind::Coupler);
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
    for (std::size_t i = 0; i < clocks_.size(); ++i) {
      if (coupler_[i]) continue;  // no clock, by design -- see the ctor's own comment
      const Rational n = clocks_[i].next();
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
      if (coupler_[i]) continue;  // no clock, by design -- see the ctor's own comment
      if (clocks_[i].at(t)) {
        due.push_back(ComponentId{static_cast<std::uint32_t>(i)});
      }
    }
    return due;
  }

  /// Spec Section 8.2: lockstep is the degenerate case and gets a fast path. A
  /// coupler's own default interval (Rational{1}, since it declares
  /// none) never participates -- see the ctor's own comment.
  [[nodiscard]] bool is_lockstep() const {
    std::optional<Rational> first;
    for (std::size_t i = 0; i < clocks_.size(); ++i) {
      if (coupler_[i]) continue;
      if (!first.has_value()) {
        first = clocks_[i].interval();
        continue;
      }
      if (clocks_[i].interval() != *first) return false;
    }
    return true;
  }

  /// Shortest span after which every clock is simultaneously at a tick.
  /// Throws std::logic_error if the LCM is not representable; callers that
  /// must not throw use checked_common_period() instead. A coupler's own
  /// clock never participates -- see the ctor's own comment.
  [[nodiscard]] Rational common_period() const {
    Rational p{1};
    bool any = false;
    for (std::size_t i = 0; i < clocks_.size(); ++i) {
      if (coupler_[i]) continue;
      if (!any) {
        p = clocks_[i].interval();
        any = true;
        continue;
      }
      p = lcm_rational(p, clocks_[i].interval());
    }
    return p;
  }

  /// Finest interval in the set; the slab grid resolution. A coupler's
  /// own clock never participates -- see the ctor's own comment.
  [[nodiscard]] Rational finest_interval() const {
    Rational f{1};
    bool any = false;
    for (std::size_t i = 0; i < clocks_.size(); ++i) {
      if (coupler_[i]) continue;
      if (!any) {
        f = clocks_[i].interval();
        any = true;
        continue;
      }
      f = gcd_rational(f, clocks_[i].interval());
    }
    return f;
  }

  /// common_period() with the guard the schedule actually needs: a period
  /// spanning more than kMaxPeriodTicks ticks of the finest interval, or one
  /// that is not representable at all, is reported as E-SCHED-PERIOD rather
  /// than handed downstream to build an unusable slab decomposition.
  /// Never throws. A coupler's own clock never participates -- see the
  /// ctor's own comment: its default interval (Rational{1}, since it
  /// declares none of its own) must never enter this lcm/gcd, or a
  /// window-scoped graph whose real period is shorter than 1 gets
  /// lcm(period, 1) = 1 folded in silently (measured: the Brusselator's
  /// own 1/50 window, once a coupler was added, re-ran 500 times before
  /// one compiled period elapsed).
  [[nodiscard]] Result<Rational> checked_common_period() const {
    // No try/catch, and that is the point: this function is documented never
    // to throw, and an exception must not be the mechanism by which it keeps
    // that promise. It used to catch `fuse::Error` around the throwing
    // arithmetic, which meant the promise held only in a build WITH
    // exceptions -- with them disabled a refusal aborts instead of unwinding,
    // so the process died here rather than returning the Failure the caller
    // was written to handle. Every step below reports instead.
    Rational period{1};
    Rational finest{1};
    bool any = false;
    for (std::size_t i = 0; i < clocks_.size(); ++i) {
      if (coupler_[i]) continue;
      const Clock& c = clocks_[i];
      if (!any) {
        period = c.interval();
        finest = period;
        any = true;
        continue;
      }
      const auto p = try_lcm_rational(period, c.interval());
      if (!p) {
        return Failure{"E-SCHED-PERIOD",
                       "common period is not representable: the least common "
                       "multiple of the clock intervals overflows"};
      }
      period = *p;
      const auto f = try_gcd_rational(finest, c.interval());
      if (!f) {
        return Failure{"E-SCHED-PERIOD",
                       "the finest interval is not representable: the greatest "
                       "common divisor of the clock intervals overflows"};
      }
      finest = *f;
    }
    const auto maybe_ticks = Rational::try_div(period, finest);
    if (!maybe_ticks) {
      return Failure{"E-SCHED-PERIOD",
                     "common period " + period.to_string() +
                         " is not an expressible number of ticks of the finest "
                         "interval " + finest.to_string()};
    }
    const Rational ticks = *maybe_ticks;
    if (ticks > Rational{kMaxPeriodTicks}) {
      return Failure{"E-SCHED-PERIOD",
                   "common period " + period.to_string() + " spans " +
                       ticks.to_string() +
                       " ticks of the finest interval, exceeding the limit of " +
                       std::to_string(kMaxPeriodTicks)};
    }
    return period;
  }

 private:
  std::vector<Clock> clocks_;
  /// Parallel to `clocks_`: `coupler_[i]` is true when `clocks_[i]` backs a
  /// `ComponentKind::Coupler` entry, which owns no clock of its own by
  /// design (the ctor's own comment) -- kept dense, like `clocks_` itself,
  /// so index `i` always means the same `ComponentId` in both.
  std::vector<bool> coupler_;
};

}  // namespace fuse
