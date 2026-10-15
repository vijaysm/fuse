#include <gtest/gtest.h>

#include <fuse/base/error.hpp>
#include <fuse/schedule/clock.hpp>

#include <algorithm>
#include <stdexcept>

using fuse::Clock;
using fuse::ClockSet;
using fuse::ComponentDecl;
using fuse::ComponentId;
using fuse::CouplingGraph;
using fuse::lcm_rational;
using fuse::Rational;

TEST(Clock, StartsAtZeroStepZero) {
  const Clock c{Rational{1, 3}};
  EXPECT_EQ(c.now(), Rational{0});
  EXPECT_EQ(c.step(), 0);
  EXPECT_EQ(c.interval(), Rational(1, 3));
  EXPECT_EQ(c.next(), Rational(1, 3));
}

TEST(Clock, AdvancesExactly) {
  Clock c{Rational{1, 3}};
  c.advance();
  c.advance();
  c.advance();
  EXPECT_EQ(c.now(), Rational{1});
  EXPECT_EQ(c.step(), 3);
}

// The property spec Section 4.3 exists to guarantee.
TEST(Clock, NoDriftOverAMillionSteps) {
  Clock c{Rational{1, 7}};
  for (int i = 0; i < 1'000'000; ++i) c.advance();
  EXPECT_EQ(c.now(), Rational(1'000'000, 7));
  EXPECT_EQ(c.now().den(), 7);
}

TEST(Clock, NowIsComputedNotAccumulated) {
  // There is no setter for the step, so this cannot literally compare a jumped
  // clock against a stepped one. What it can do is pin the documented closed
  // form now() == start + step*interval against an expression built without
  // any repeated addition, for a start that is not zero.
  Clock a{Rational{1, 3}, Rational{2, 5}};
  for (int i = 0; i < 100; ++i) a.advance();
  EXPECT_EQ(a.step(), 100);
  EXPECT_EQ(a.now(), a.start() + Rational{a.step()} * a.interval());
  EXPECT_EQ(a.now(), Rational(2, 5) + Rational(100, 3));
  EXPECT_EQ(a.next(), a.start() + Rational{a.step() + 1} * a.interval());
}

TEST(Clock, AtTestsTickAlignment) {
  const Clock c{Rational{1, 4}};
  EXPECT_TRUE(c.at(Rational{0}));
  EXPECT_TRUE(c.at(Rational{1, 4}));
  EXPECT_TRUE(c.at(Rational{1}));
  EXPECT_FALSE(c.at(Rational{1, 3}));
}

// Steps are non-negative, so a time before the start is not a tick even when it
// lands on the interval grid.
TEST(Clock, AtIsFalseBeforeTheStart) {
  const Clock c{Rational{1, 2}, Rational{5}};
  EXPECT_TRUE(c.at(Rational{5}));
  EXPECT_TRUE(c.at(Rational{11, 2}));
  EXPECT_FALSE(c.at(Rational{9, 2}));  // grid-aligned but before the start
  EXPECT_FALSE(c.at(Rational{4}));
  EXPECT_FALSE(c.at(Rational{21, 4}));
}

TEST(Clock, ResetReturnsToStart) {
  Clock c{Rational{1, 2}, Rational{5}};
  c.advance();
  c.reset();
  EXPECT_EQ(c.now(), Rational{5});
  EXPECT_EQ(c.step(), 0);
}

// Enforced with throw, not assert: a zero or negative interval has no next
// tick and would hang the runtime, and asserts vanish under -DNDEBUG.
TEST(Clock, RejectsNonPositiveInterval) {
  EXPECT_THROW(((void)Clock{Rational{0}}), fuse::InvalidArgument);
  EXPECT_THROW(((void)Clock{Rational{-1, 3}}), fuse::InvalidArgument);
}

TEST(LcmRational, SmallestCommonMultiple) {
  EXPECT_EQ(lcm_rational(Rational(1, 2), Rational(1, 3)), Rational(1));
  EXPECT_EQ(lcm_rational(Rational(1, 2), Rational(1, 4)), Rational(1, 2));
  EXPECT_EQ(lcm_rational(Rational(2, 3), Rational(1, 2)), Rational(2));
  EXPECT_EQ(lcm_rational(Rational(3), Rational(2)), Rational(6));
}

TEST(LcmRational, IsExactForLargeCoprimeNumerators) {
  // 3037000493 * 3037000507 > 2^63 would overflow a 64-bit product; the
  // computation runs in __int128 and the reduced result still fits.
  EXPECT_THROW((void)lcm_rational(Rational(3037000493LL), Rational(3037000507LL)),
               fuse::Refusal);
  // ... and a product that does fit must come back exact, not wrapped.
  EXPECT_EQ(lcm_rational(Rational(2147483647LL), Rational(2147483629LL)),
            Rational(2147483647LL * 2147483629LL));
}

TEST(LcmRational, RejectsZero) {
  EXPECT_THROW((void)lcm_rational(Rational{0}, Rational{1}), fuse::InvalidArgument);
  EXPECT_THROW((void)lcm_rational(Rational{1}, Rational{0}), fuse::InvalidArgument);
}

namespace {

CouplingGraph two_components(Rational a, Rational b) {
  CouplingGraph g;
  g.add_component(ComponentDecl{.name = "a", .interval = a});
  g.add_component(ComponentDecl{.name = "b", .interval = b});
  return g;
}

}  // namespace

TEST(ClockSet, OneClockPerComponentSeededFromInterval) {
  const ClockSet cs{two_components(Rational{1, 2}, Rational{1, 3})};
  EXPECT_EQ(cs.size(), 2u);
  EXPECT_EQ(cs.clock(ComponentId{0}).interval(), Rational(1, 2));
  EXPECT_EQ(cs.clock(ComponentId{1}).interval(), Rational(1, 3));
}

TEST(ClockSet, EarliestNextPicksTheSoonestTick) {
  const ClockSet cs{two_components(Rational{1, 2}, Rational{1, 3})};
  EXPECT_EQ(cs.earliest_next(), Rational(1, 3));
}

TEST(ClockSet, DueAtListsAlignedComponents) {
  const ClockSet cs{two_components(Rational{1, 2}, Rational{1, 4})};
  const auto due = cs.due_at(Rational{1, 2});
  EXPECT_EQ(due.size(), 2u);  // 1/2 is a tick of both
  const auto due_quarter = cs.due_at(Rational{1, 4});
  ASSERT_EQ(due_quarter.size(), 1u);
  EXPECT_EQ(due_quarter[0], ComponentId{1});
}

TEST(ClockSet, DetectsLockstep) {
  EXPECT_TRUE(ClockSet{two_components(Rational{1}, Rational{1})}.is_lockstep());
  EXPECT_FALSE(
      ClockSet{two_components(Rational{1}, Rational{1, 2})}.is_lockstep());
}

TEST(ClockSet, CommonPeriodIsTheLcm) {
  EXPECT_EQ(
      ClockSet{two_components(Rational{1, 2}, Rational{1, 3})}.common_period(),
      Rational(1));
  EXPECT_EQ(
      ClockSet{two_components(Rational{1, 4}, Rational{1, 2})}.common_period(),
      Rational(1, 2));
}

TEST(ClockSet, FinestIntervalIsTheGcd) {
  EXPECT_EQ(
      ClockSet{two_components(Rational{1, 2}, Rational{1, 3})}.finest_interval(),
      Rational(1, 6));
  EXPECT_EQ(
      ClockSet{two_components(Rational{1, 2}, Rational{1, 4})}.finest_interval(),
      Rational(1, 4));
}

// Bounds-checked in Release too: .at() throws, it is not an assert.
TEST(ClockSet, ClockLookupIsBoundsChecked) {
  const ClockSet cs{two_components(Rational{1}, Rational{1})};
  EXPECT_THROW((void)cs.clock(ComponentId{2}), std::out_of_range);
  EXPECT_THROW((void)cs.clock(ComponentId{}), std::out_of_range);
}

TEST(ClockSet, CheckedCommonPeriodAcceptsAReasonableSchedule) {
  const auto r =
      ClockSet{two_components(Rational{1, 2}, Rational{1, 3})}
          .checked_common_period();
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_EQ(r.value(), Rational{1});
}

// The guard the brief calls for: an LCM needing more than kMaxPeriodTicks ticks
// of the finest interval is an unusable schedule, reported rather than built.
TEST(ClockSet, CheckedCommonPeriodRejectsAnExplosivePeriod) {
  const auto r = ClockSet{two_components(Rational{1, 999983}, Rational{1, 999979})}
                     .checked_common_period();
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-PERIOD");
}

// Overflow inside the LCM is reported as an error, never as an exception
// escaping a Result-returning function.
TEST(ClockSet, CheckedCommonPeriodReportsOverflowAsAnError) {
  const auto r = ClockSet{two_components(Rational(3037000493LL),
                                         Rational(3037000507LL))}
                     .checked_common_period();
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-PERIOD");
}

// Conformance Unit 2, defect found by the lead: ComponentDecl::interval
// defaults to Rational{1} (a coupler declares no interval of its own --
// component.hpp's own comment on CouplerDecl/ComponentKind::Coupler), and
// before this fix every one of ClockSet's own aggregates folded that
// default straight into its answer like any physics component's real
// interval. One physics component at 1/50 plus a coupler must still give
// period/finest_interval 1/50, not lcm(1/50, 1) = 1 -- the 50 window
// instances per period Conformance Unit 2's own tier-0 harness measured.
// MUTATION: reverting ClockSet's own coupler exclusion (deleting the
// `if (coupler_[i]) continue;` guards) makes this fail -- both EXPECT_EQ
// below would instead see Rational{1}.
TEST(ClockSet, ACouplersDefaultIntervalNeverEntersTheLcmOrGcd) {
  CouplingGraph g;
  g.add_component(ComponentDecl{.name = "p", .interval = Rational{1, 50}});
  g.add_coupler(fuse::CouplerDecl{.name = "c", .ranks = fuse::RankSet{0, 1}});
  const ClockSet cs{g};
  ASSERT_EQ(cs.size(), 2u);  // dense: the coupler still gets a Clock slot
  EXPECT_EQ(cs.common_period(), Rational(1, 50));
  EXPECT_EQ(cs.finest_interval(), Rational(1, 50));
  const auto checked = cs.checked_common_period();
  ASSERT_TRUE(checked.ok()) << checked.error().message;
  EXPECT_EQ(checked.value(), Rational(1, 50));
}

// Same defect, is_lockstep(): a single physics component is trivially
// lockstep with itself; a coupler's own default interval (1, which
// differs from 1/50) must not be read as a second, mismatched clock.
TEST(ClockSet, ACouplerDoesNotBreakLockstepDetection) {
  CouplingGraph g;
  g.add_component(ComponentDecl{.name = "p", .interval = Rational{1, 50}});
  g.add_coupler(fuse::CouplerDecl{.name = "c", .ranks = fuse::RankSet{0, 1}});
  const ClockSet cs{g};
  EXPECT_TRUE(cs.is_lockstep());
}

TEST(ClockSet, EmptyGraphIsLockstepWithUnitPeriod) {
  const CouplingGraph g;
  const ClockSet cs{g};
  EXPECT_EQ(cs.size(), 0u);
  EXPECT_TRUE(cs.is_lockstep());
  EXPECT_EQ(cs.common_period(), Rational{1});
  EXPECT_EQ(cs.finest_interval(), Rational{1});
  EXPECT_TRUE(cs.due_at(Rational{0}).empty());
}
