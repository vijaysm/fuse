#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>

using fuse::AccumOp;
using fuse::ComponentDecl;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::FieldSpec;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
using fuse::RemapFlags;
namespace u = fuse::units;

namespace {

// Two components, one channel: the smallest useful graph.
CouplingGraph make_pair_graph() {
  CouplingGraph g;
  const auto m_f = g.add_mesh(MeshDecl{.name = "fluid_mesh",
                                       .source = "generated:box:8x8x8"});
  const auto m_s = g.add_mesh(MeshDecl{.name = "solid_mesh",
                                       .source = "generated:box:5x5x5"});
  const auto r_f = g.add_region(Region{.name = "fluid_wall", .mesh = m_f,
                                       .entity_set = "wall", .dimension = 2});
  const auto r_s = g.add_region(Region{.name = "solid_wall", .mesh = m_s,
                                       .entity_set = "wall", .dimension = 2});

  const auto fluid = g.add_component(
      ComponentDecl{.name = "fluid", .ranks = fuse::RankSet{0, 2},
                    .interval = Rational{1, 2}});
  const auto solid = g.add_component(
      ComponentDecl{.name = "solid", .ranks = fuse::RankSet{2, 2},
                    .interval = Rational{1}});

  const auto q_out = g.add_field(
      fluid, FieldSpec{.name = "heat_flux", .region = r_f,
                       .units = u::kHeatFluxDensity,
                       .conservation = Conservation::Extensive,
                       .intent = fuse::Intent::Provides});
  const auto q_in = g.add_field(
      solid, FieldSpec{.name = "q_wall", .region = r_s,
                       .units = u::kHeatFluxDensity,
                       .conservation = Conservation::Extensive,
                       .intent = fuse::Intent::Requires});

  const auto ch = g.add_channel("flux", q_out, q_in);
  g.channel(ch).accumulate(AccumOp::Mean).remap(RemapFlags::Conservative);
  return g;
}

}  // namespace

TEST(CouplingGraph, EmptyGraphHasNoEntities) {
  const CouplingGraph g;
  EXPECT_EQ(g.component_count(), 0u);
  EXPECT_EQ(g.channel_count(), 0u);
}

TEST(CouplingGraph, IdsAreSequentialAndValid) {
  CouplingGraph g;
  const auto a = g.add_mesh(MeshDecl{.name = "a"});
  const auto b = g.add_mesh(MeshDecl{.name = "b"});
  EXPECT_TRUE(a.valid());
  EXPECT_EQ(a.value(), 0u);
  EXPECT_EQ(b.value(), 1u);
  EXPECT_EQ(g.mesh_count(), 2u);
}

TEST(CouplingGraph, BuildsThePairGraph) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_EQ(g.mesh_count(), 2u);
  EXPECT_EQ(g.region_count(), 2u);
  EXPECT_EQ(g.component_count(), 2u);
  EXPECT_EQ(g.field_count(), 2u);
  EXPECT_EQ(g.channel_count(), 1u);
}

TEST(CouplingGraph, FieldsAreOwnedByTheirComponent) {
  const CouplingGraph g = make_pair_graph();
  const auto fluid = g.find_component("fluid").value();
  const auto q = g.find_field(fluid, "heat_flux").value();
  EXPECT_EQ(g.owner_of(q), fluid);
  EXPECT_EQ(g.component(fluid).fields.size(), 1u);
}

TEST(CouplingGraph, FindReturnsNulloptWhenAbsent) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_FALSE(g.find_component("nonexistent").has_value());
  const auto fluid = g.find_component("fluid").value();
  EXPECT_FALSE(g.find_field(fluid, "nope").has_value());
}

TEST(CouplingGraph, ChannelBuilderMutatesStoredChannel) {
  const CouplingGraph g = make_pair_graph();
  const auto& ch = g.channel_at(fuse::ChannelId{0});
  EXPECT_EQ(ch.accumulate.op, AccumOp::Mean);
  EXPECT_TRUE(has(ch.remap.flags, RemapFlags::Conservative));
}

TEST(CouplingGraph, ComponentsKeepIndependentIntervals) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_EQ(g.component(g.find_component("fluid").value()).interval,
            Rational(1, 2));
  EXPECT_EQ(g.component(g.find_component("solid").value()).interval,
            Rational(1));
}

// Regression test: a ChannelBuilder held across intervening add_channel()
// calls must not dangle when channels_ reallocates. Before the fix,
// ChannelBuilder stored a `Channel&` resolved at construction time; this
// test forces at least one reallocation of the backing vector while the
// builder is alive, then mutates through it. Under ASan the pre-fix code
// aborts with a heap-use-after-free; without ASan it silently mutates freed
// memory and the assertions below fail because the write never reaches the
// real, reallocated Channel.
TEST(CouplingGraph, ChannelBuilderSurvivesReallocation) {
  CouplingGraph g;
  const auto first = g.add_channel("first", fuse::FieldId{0}, fuse::FieldId{1});
  auto builder = g.channel(first);  // held, not consumed as a temporary

  // Force at least one reallocation of the std::vector<Channel> backing
  // store while `builder` is still alive.
  for (int i = 0; i < 100; ++i) {
    g.add_channel("filler", fuse::FieldId{0}, fuse::FieldId{1});
  }

  builder.accumulate(AccumOp::Sum).remap(RemapFlags::Conservative);

  const auto& ch = g.channel_at(first);
  EXPECT_EQ(ch.accumulate.op, AccumOp::Sum);
  EXPECT_TRUE(has(ch.remap.flags, RemapFlags::Conservative));
}

// window_sample_capacity(): hand-derived closed-form values, exact
// Rational arithmetic (no std::ceil-on-a-double anywhere in the
// production function). Time windows track (C3): this is the SAME
// function validate_coupling_window()'s E-SCHED-WINDOW-CAPACITY and
// window_lowering.cpp's window_source_type() both call, so a wrong value
// here would silently mis-size a compiled field as well as the schedule-
// time refusal.
TEST(CouplingGraph, WindowSampleCapacityMatchesHandDerivedValues) {
  using fuse::DtPolicy;
  using fuse::window_sample_capacity;

  // window=1, interval=1/4: ratio=4 exact -> 4 substeps + 1 window-start.
  EXPECT_EQ(window_sample_capacity(Rational{1}, Rational{1, 4},
                                   DtPolicy::Adjustable),
            5u);
  // Fixed adds one more for the possible straddle.
  EXPECT_EQ(
      window_sample_capacity(Rational{1}, Rational{1, 4}, DtPolicy::Fixed),
      6u);

  // window=13, interval=7: ratio=13/7, not exact -> ceil=2, +1=3.
  EXPECT_EQ(
      window_sample_capacity(Rational{13}, Rational{7}, DtPolicy::Adjustable),
      3u);
  EXPECT_EQ(
      window_sample_capacity(Rational{13}, Rational{7}, DtPolicy::Fixed), 4u);

  // window=7, interval=13: ratio=7/13 < 1 -> ceil=1, +1=2 (the member
  // takes exactly one, truncated/straddling substep to cover the window).
  EXPECT_EQ(
      window_sample_capacity(Rational{7}, Rational{13}, DtPolicy::Adjustable),
      2u);
  EXPECT_EQ(
      window_sample_capacity(Rational{7}, Rational{13}, DtPolicy::Fixed), 3u);

  // window=1, interval=1/3: ratio=3 exact -> 3+1=4.
  EXPECT_EQ(window_sample_capacity(Rational{1}, Rational{1, 3},
                                   DtPolicy::Adjustable),
            4u);

  // Non-positive interval: never fails, returns the harmless minimum (the
  // caller's own E-SCHED-WINDOW-SHORTEN/E-SCHED-WINDOW-INTERVAL gate is
  // what actually refuses this case).
  EXPECT_EQ(
      window_sample_capacity(Rational{1}, Rational{0}, DtPolicy::Adjustable),
      1u);
}
