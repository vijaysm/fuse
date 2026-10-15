#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>

using fuse::AccumOp;
using fuse::Channel;
using fuse::ComponentId;
using fuse::CouplingGraph;
using fuse::FieldId;
using fuse::RemapFlags;
using fuse::TemporalBasis;
using fuse::Window;

TEST(Channel, DefaultsToInstantNoRemapNoRelax) {
  const Channel c;
  EXPECT_EQ(c.accumulate.op, AccumOp::Instant);
  EXPECT_EQ(c.remap.flags, RemapFlags::None);
  EXPECT_FALSE(c.relax.enabled);
  EXPECT_FALSE(c.extrapolate.enabled);
  EXPECT_EQ(c.lag, 0);
  EXPECT_EQ(c.spacetime.basis, TemporalBasis::None);
}

TEST(RemapFlags, CombineAndTest) {
  const RemapFlags f = RemapFlags::Conservative | RemapFlags::Monotone;
  EXPECT_TRUE(has(f, RemapFlags::Conservative));
  EXPECT_TRUE(has(f, RemapFlags::Monotone));
  EXPECT_FALSE(has(f, RemapFlags::HighOrder));
}

// This is the exact syntax from spec Section 5.2. ChannelBuilder is obtained only
// through CouplingGraph::channel() (it holds a graph reference + id, not a
// bare Channel&, so it survives channels_ reallocating).
TEST(ChannelBuilder, BuildsTheSpecExample) {
  CouplingGraph g;
  const auto id = g.add_channel("T_wall->T_bc", FieldId{0}, FieldId{1});
  g.channel(id)
      .accumulate(AccumOp::Mean, Window::coupling_period())
      .remap(RemapFlags::Conservative | RemapFlags::Monotone)
      .relax(0.7);

  const Channel& c = g.channel_at(id);
  EXPECT_EQ(c.accumulate.op, AccumOp::Mean);
  EXPECT_EQ(c.accumulate.window.kind, Window::Kind::CouplingPeriod);
  EXPECT_TRUE(has(c.remap.flags, RemapFlags::Conservative));
  EXPECT_TRUE(has(c.remap.flags, RemapFlags::Monotone));
  EXPECT_TRUE(c.relax.enabled);
  EXPECT_DOUBLE_EQ(c.relax.omega, 0.7);
}

// Spec Section 7.3 example.
TEST(ChannelBuilder, BuildsSpacetimeChannel) {
  CouplingGraph g;
  const auto id = g.add_channel("sst->T_surf", FieldId{2}, FieldId{3});
  g.channel(id)
      .spacetime(TemporalBasis::DG, 1, Window::from(ComponentId{0}))
      .conservative();

  const Channel& c = g.channel_at(id);
  EXPECT_EQ(c.spacetime.basis, TemporalBasis::DG);
  EXPECT_EQ(c.spacetime.degree, 1);
  EXPECT_EQ(c.spacetime.window.kind, Window::Kind::FromComponent);
  EXPECT_TRUE(has(c.remap.flags, RemapFlags::Conservative));
}

TEST(ChannelBuilder, LagIsRecorded) {
  CouplingGraph g;
  const auto id = g.add_channel("c", FieldId{}, FieldId{});
  g.channel(id).lag(1);
  EXPECT_EQ(g.channel_at(id).lag, 1);
}

// Spec Section 4.3: the escape hatch must record a reason, never be silent.
TEST(ChannelBuilder, TimeLevelMismatchRequiresAReason) {
  CouplingGraph g;
  const auto id = g.add_channel("c", FieldId{}, FieldId{});
  EXPECT_TRUE(g.channel_at(id).time_level_mismatch_reason.empty());
  g.channel(id).allow_time_level_mismatch("deliberate leapfrog stagger");
  EXPECT_EQ(g.channel_at(id).time_level_mismatch_reason,
            "deliberate leapfrog stagger");
}

TEST(ChannelBuilder, ChainsReturnSelf) {
  CouplingGraph g;
  const auto id = g.add_channel("c", FieldId{}, FieldId{});
  g.channel(id).lag(2).relax(0.5).extrapolate(1);
  const Channel& c = g.channel_at(id);
  EXPECT_EQ(c.lag, 2);
  EXPECT_DOUBLE_EQ(c.relax.omega, 0.5);
  EXPECT_TRUE(c.extrapolate.enabled);
  EXPECT_EQ(c.extrapolate.order, 1);
}

// The reallocation-safety regression test lives in coupling_graph_test.cpp
// (CouplingGraph.ChannelBuilderSurvivesReallocation), since exercising it
// requires driving add_channel() through CouplingGraph.
