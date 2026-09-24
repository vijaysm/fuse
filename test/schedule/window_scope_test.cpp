// The schedule-time legality gate for the "Time windows" opt-in
// (fuse/schedule/window.hpp), exercised against the SAME two-clock graph
// shape multirate_region_test.cpp probes -- every refusal this file names
// must actually fire (docs/dev/testing.md's "Deliberate refusal" state).

#include <gtest/gtest.h>
#include <fuse/schedule/window.hpp>

#include <string>

using fuse::ComponentDecl;
using fuse::CouplingWindowScope;
using fuse::CouplingGraph;
using fuse::DtPolicy;
using fuse::IterationRegionDecl;
using fuse::Rational;
using fuse::WindowMemberPolicy;
namespace u = fuse::units;

namespace {

// coarse (interval `coarse_interval`) <-> fine (interval `fine_interval`),
// two channels, one iteration region -- mirrors two_clock_region() in
// multirate_region_test.cpp so the two files stay comparable.
struct TwoClockFixture {
  CouplingGraph        graph;
  IterationRegionDecl  region;
};

TwoClockFixture two_clock_fixture(Rational coarse_interval,
                                  Rational fine_interval) {
  TwoClockFixture f;
  CouplingGraph& g = f.graph;
  const auto mc = g.add_mesh(fuse::MeshDecl{.name = "mc"});
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto rc = g.add_region(fuse::Region{.name = "rc", .mesh = mc});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});

  const auto coarse = g.add_component(ComponentDecl{
      .name = "coarse", .ranks = fuse::RankSet{0, 1}, .interval = coarse_interval});
  const auto fine = g.add_component(ComponentDecl{
      .name = "fine", .ranks = fuse::RankSet{1, 1}, .interval = fine_interval});

  const auto fine_out = g.add_field(fine, fuse::FieldSpec{
      .name = "fine_out", .region = rf, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto coarse_in = g.add_field(coarse, fuse::FieldSpec{
      .name = "coarse_in", .region = rc, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto coarse_out = g.add_field(coarse, fuse::FieldSpec{
      .name = "coarse_out", .region = rc, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto fine_in = g.add_field(fine, fuse::FieldSpec{
      .name = "fine_in", .region = rf, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});

  const auto fwd = g.add_channel("fine2coarse", fine_out, coarse_in);
  g.channel(fwd).accumulate(fuse::AccumOp::Mean);
  const auto back = g.add_channel("coarse2fine", coarse_out, fine_in);
  g.channel(back).relax(0.7);

  f.region = IterationRegionDecl{
      .name = "multirate", .channels = {fwd, back}, .max_iterations = 20};
  return f;
}

CouplingWindowScope scope_naming_coarse(int order = 2) {
  CouplingWindowScope s;
  s.window_component = "coarse";
  s.order = order;
  s.policies = {{"coarse", DtPolicy::Adjustable}, {"fine", DtPolicy::Fixed}};
  return s;
}

}  // namespace

TEST(WindowScope, WellFormedScopeIsAccepted) {
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  const auto r = fuse::validate_coupling_window(f.graph, f.region,
                                                scope_naming_coarse());
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

TEST(WindowScope, EmptyRegionIsRefused) {
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  IterationRegionDecl empty_region = f.region;
  empty_region.channels.clear();
  const auto r = fuse::validate_coupling_window(f.graph, empty_region,
                                                scope_naming_coarse());
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-EMPTY");
}

TEST(WindowScope, BothWindowComponentAndExplicitIntervalIsAmbiguous) {
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  CouplingWindowScope s = scope_naming_coarse();
  s.explicit_interval = Rational{2};
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-AMBIGUOUS");
}

TEST(WindowScope, NeitherWindowComponentNorExplicitIntervalIsAmbiguous) {
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  CouplingWindowScope s = scope_naming_coarse();
  s.window_component.reset();
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-AMBIGUOUS");
}

TEST(WindowScope, WindowComponentNotInGraphIsRefused) {
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  CouplingWindowScope s = scope_naming_coarse();
  s.window_component = "nonexistent";
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-COMPONENT");
}

TEST(WindowScope, WindowComponentNotAMemberOfTheRegionIsRefused) {
  TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  // A third, uninvolved component: named as the window, it does not own
  // either end of the region's channels.
  f.graph.add_component(ComponentDecl{
      .name = "bystander", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  CouplingWindowScope s = scope_naming_coarse();
  s.window_component = "bystander";
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-COMPONENT");
}

TEST(WindowScope, NonPositiveExplicitIntervalIsRefused) {
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  CouplingWindowScope s = scope_naming_coarse();
  s.window_component.reset();
  s.explicit_interval = Rational{0};
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-INTERVAL");
}

TEST(WindowScope, OrderBelowTwoIsRefused) {
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  const auto r = fuse::validate_coupling_window(f.graph, f.region,
                                                scope_naming_coarse(1));
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-ORDER");
}

TEST(WindowScope, MemberWithNoPolicyEntryIsRefused) {
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  CouplingWindowScope s = scope_naming_coarse();
  s.policies = {{"coarse", DtPolicy::Adjustable}};  // "fine" missing
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-POLICY");
}

TEST(WindowScope, PolicyNamingANonMemberIsRefused) {
  TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 4});
  f.graph.add_component(ComponentDecl{
      .name = "bystander", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  CouplingWindowScope s = scope_naming_coarse();
  s.policies.push_back({"bystander", DtPolicy::Fixed});
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-COMPONENT");
}

TEST(WindowScope, AdjustableComponentWithNonPositiveIntervalIsRefused) {
  // fine's own interval is 0 -- Adjustable has nothing to shorten.
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{0});
  CouplingWindowScope s;
  s.window_component = "coarse";
  s.policies = {{"coarse", DtPolicy::Adjustable}, {"fine", DtPolicy::Adjustable}};
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-SHORTEN");
}

TEST(WindowScope, WindowLongerThanAWaveformCanHoldIsRefused) {
  // fine ticks 1/1000 of the coarse window -- far more samples per window
  // than the tiny configured capacity can hold.
  const TwoClockFixture f = two_clock_fixture(Rational{1}, Rational{1, 1000});
  CouplingWindowScope s = scope_naming_coarse();
  const auto r = fuse::validate_coupling_window(f.graph, f.region, s,
                                                /*waveform_capacity=*/8);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-CAPACITY");
}

TEST(WindowScope, IncommensurateSevenThirteenWindowLengthIsTheNamedComponent) {
  const TwoClockFixture f = two_clock_fixture(Rational{13}, Rational{7});
  const auto len = fuse::window_length(f.graph, scope_naming_coarse());
  ASSERT_TRUE(len.ok());
  EXPECT_EQ(len.value(), Rational{13});
}

TEST(WindowScope, ExplicitIntervalNeedNotMatchAnyComponent) {
  const TwoClockFixture f = two_clock_fixture(Rational{7}, Rational{5});
  CouplingWindowScope s;
  s.explicit_interval = Rational{11};
  s.policies = {{"coarse", DtPolicy::Fixed}, {"fine", DtPolicy::Fixed}};
  const auto len = fuse::window_length(f.graph, s);
  ASSERT_TRUE(len.ok());
  EXPECT_EQ(len.value(), Rational{11});
}
