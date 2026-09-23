// Tier-0 probe: what happens TODAY when a two-way coupling between two
// components on DIFFERENT clocks is placed inside one iteration region?
//
// This file is a reproducer, not a feature test. It documents the exact
// refusal (or acceptance) `schedule()` produces at ratio m = 1, 2, 4 between
// a coarse component (`interval = 1`) and a fine one (`interval = 1/m`),
// coupled two-way: fine -> coarse (accumulated, so the coarse read is well
// defined) and coarse -> fine (relaxed, so the pair carries a value across
// sweeps rather than being a hard SSA cycle -- the same mechanism
// `Acyclicity.IterationRegionWithACarriedValueBreaksTheCycle` uses at m = 1).
//
// The plan capabilities task (step 1) requires this behaviour recorded
// before anything changes. Do not "fix" this file when waveform iteration
// lands -- update the expectations deliberately, with a comment saying so,
// because `MultiClockRegion.RatioOneMatchesTodaysSingleClockRegion` is the
// equivalence anchor for that later change.

#include <gtest/gtest.h>
#include <fuse/ir/lowering.hpp>
#include <fuse/schedule/plan.hpp>

#include <cstddef>
#include <string>

using fuse::ComponentDecl;
using fuse::CouplingGraph;
using fuse::Rational;
namespace u = fuse::units;

namespace {

// coarse (interval 1) <-> fine (interval 1/m), both channels in one
// iteration region. fine2coarse accumulates (Mean) over the coarse step;
// coarse2fine is relaxed so the region carries a value across sweeps.
CouplingGraph two_clock_region(int m) {
  CouplingGraph g;
  const auto mc = g.add_mesh(fuse::MeshDecl{.name = "mc"});
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto rc = g.add_region(fuse::Region{.name = "rc", .mesh = mc});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});

  const auto coarse = g.add_component(ComponentDecl{
      .name = "coarse", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto fine = g.add_component(ComponentDecl{
      .name = "fine", .ranks = fuse::RankSet{1, 1},
      .interval = Rational{1, m}});

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

  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "multirate", .channels = {fwd, back}, .max_iterations = 20});
  return g;
}

struct ProbeResult {
  bool lowered_ok = false;
  bool scheduled_ok = false;
  std::string diag_codes;   // semicolon-joined
  std::size_t region_count = 0;
};

ProbeResult probe(int m) {
  ProbeResult out;
  const CouplingGraph g = two_clock_region(m);
  const auto lowered = lower(g);
  out.lowered_ok = lowered.ok();
  if (!out.lowered_ok) {
    out.diag_codes = lowered.diags.to_string();
    return out;
  }
  const fuse::ClockSet cs{g};
  const auto r = fuse::schedule(
      g, lowered.dag, cs,
      [](fuse::RegionId, int) -> std::size_t { return 64; });
  out.scheduled_ok = r.ok();
  out.region_count = r.plan.region_count();
  for (const auto& item : r.diags.items()) {
    if (!out.diag_codes.empty()) out.diag_codes += ";";
    out.diag_codes += item.code;
  }
  return out;
}

}  // namespace

// m = 1: the coarse and fine components share one clock, so this is exactly
// today's single-clock region and schedules cleanly. This is the anchor: the
// waveform-iteration change must leave this case's plan unchanged.
TEST(MultirateRegionProbe, RatioOneSchedulesAsASingleClockRegionDoesToday) {
  const ProbeResult r = probe(1);
  ASSERT_TRUE(r.lowered_ok);
  EXPECT_TRUE(r.scheduled_ok) << r.diag_codes;
  EXPECT_EQ(r.region_count, 1u);
}

// m = 2 and m = 4: the fine component ticks inside the coarse component's
// step, so the region's body -- one accumulate-reduction on the coarse clock
// plus one relaxed write on the fine clock -- cannot appear as ONE
// contiguous whole-body run per period once the fine ticks are more than one.
// Recorded exactly, so a later change has a documented "before" to diff
// against: today this is refused with E-SCHED-REGION, not silently accepted
// with a wrong plan.
TEST(MultirateRegionProbe, RatioTwoIsRefusedByTheSchedulerToday) {
  const ProbeResult r = probe(2);
  ASSERT_TRUE(r.lowered_ok) << r.diag_codes;
  EXPECT_FALSE(r.scheduled_ok);
  EXPECT_NE(r.diag_codes.find("E-SCHED-REGION"), std::string::npos)
      << r.diag_codes;
}

TEST(MultirateRegionProbe, RatioFourIsRefusedByTheSchedulerToday) {
  const ProbeResult r = probe(4);
  ASSERT_TRUE(r.lowered_ok) << r.diag_codes;
  EXPECT_FALSE(r.scheduled_ok);
  EXPECT_NE(r.diag_codes.find("E-SCHED-REGION"), std::string::npos)
      << r.diag_codes;
}
