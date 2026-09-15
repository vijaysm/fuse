#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>

using soma::AccumOp;
using soma::ComponentDecl;
using soma::Conservation;
using soma::CouplingGraph;
using soma::FieldSpec;
using soma::MeshDecl;
using soma::Rational;
using soma::Region;
using soma::RemapFlags;
namespace u = soma::units;

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
      ComponentDecl{.name = "fluid", .ranks = {.first = 0, .count = 2},
                    .interval = Rational{1, 2}});
  const auto solid = g.add_component(
      ComponentDecl{.name = "solid", .ranks = {.first = 2, .count = 2},
                    .interval = Rational{1}});

  const auto q_out = g.add_field(
      fluid, FieldSpec{.name = "heat_flux", .region = r_f,
                       .units = u::kHeatFluxDensity,
                       .conservation = Conservation::Extensive,
                       .intent = soma::Intent::Provides});
  const auto q_in = g.add_field(
      solid, FieldSpec{.name = "q_wall", .region = r_s,
                       .units = u::kHeatFluxDensity,
                       .conservation = Conservation::Extensive,
                       .intent = soma::Intent::Requires});

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
  const auto& ch = g.channel_at(soma::ChannelId{0});
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
  const auto first = g.add_channel("first", soma::FieldId{0}, soma::FieldId{1});
  auto builder = g.channel(first);  // held, not consumed as a temporary

  // Force at least one reallocation of the std::vector<Channel> backing
  // store while `builder` is still alive.
  for (int i = 0; i < 100; ++i) {
    g.add_channel("filler", soma::FieldId{0}, soma::FieldId{1});
  }

  builder.accumulate(AccumOp::Sum).remap(RemapFlags::Conservative);

  const auto& ch = g.channel_at(first);
  EXPECT_EQ(ch.accumulate.op, AccumOp::Sum);
  EXPECT_TRUE(has(ch.remap.flags, RemapFlags::Conservative));
}
