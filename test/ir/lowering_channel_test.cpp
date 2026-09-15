#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>

#include <vector>

using fuse::AccumOp;
using fuse::ComponentDecl;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::MeshDecl;
using fuse::OpKind;
using fuse::Rational;
using fuse::Region;
using fuse::RemapFlags;
namespace u = fuse::units;

namespace {

struct Pair {
  CouplingGraph g;
  fuse::ChannelId ch;
};

// fluid.heat_flux -> solid.q_wall on distinct regions and rank sets.
Pair make_pair() {
  Pair p;
  auto& g = p.g;
  const auto mf = g.add_mesh(MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(MeshDecl{.name = "ms"});
  const auto rf = g.add_region(Region{.name = "rf", .mesh = mf, .dimension = 2});
  const auto rs = g.add_region(Region{.name = "rs", .mesh = ms, .dimension = 2});
  const auto fluid = g.add_component(ComponentDecl{
      .name = "fluid", .ranks = {.first = 0, .count = 2}, .interval = Rational{1}});
  const auto solid = g.add_component(ComponentDecl{
      .name = "solid", .ranks = {.first = 2, .count = 2}, .interval = Rational{1}});
  const auto s = g.add_field(fluid, FieldSpec{
      .name = "heat_flux", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  const auto t = g.add_field(solid, FieldSpec{
      .name = "q_wall", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Requires});
  p.ch = g.add_channel("flux", s, t);
  return p;
}

std::vector<OpKind> kinds_in_order(const fuse::OperatorDAG& d) {
  std::vector<OpKind> out;
  for (const fuse::NodeId n : d.topological_order()) out.push_back(d.node(n).kind);
  return out;
}

bool contains(const std::vector<OpKind>& v, OpKind k) {
  return std::find(v.begin(), v.end(), k) != v.end();
}

int index_of(const std::vector<OpKind>& v, OpKind k) {
  return static_cast<int>(std::find(v.begin(), v.end(), k) - v.begin());
}

}  // namespace

TEST(ChannelLowering, MinimalChannelIsSourceRemapSink) {
  Pair p = make_pair();
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds_in_order(r.dag);
  EXPECT_TRUE(contains(k, OpKind::Source));
  EXPECT_TRUE(contains(k, OpKind::Sink));
  // Regions differ, so a Remap is emitted even though no flags were set.
  EXPECT_TRUE(contains(k, OpKind::Remap));
  EXPECT_FALSE(contains(k, OpKind::Accumulate));
  EXPECT_FALSE(contains(k, OpKind::Relax));
}

// This is the exact chain from spec §5.2.
TEST(ChannelLowering, SpecExampleEmitsSampleAccumulateRemapRelax) {
  Pair p = make_pair();
  p.g.channel(p.ch)
      .accumulate(AccumOp::Mean)
      .remap(RemapFlags::Conservative | RemapFlags::Monotone)
      .relax(0.7);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds_in_order(r.dag);

  ASSERT_TRUE(contains(k, OpKind::Sample));
  ASSERT_TRUE(contains(k, OpKind::Accumulate));
  ASSERT_TRUE(contains(k, OpKind::Remap));
  ASSERT_TRUE(contains(k, OpKind::Relax));
  // Order is the point of the whole design (spec §2.1).
  EXPECT_LT(index_of(k, OpKind::Sample), index_of(k, OpKind::Accumulate));
  EXPECT_LT(index_of(k, OpKind::Accumulate), index_of(k, OpKind::Remap));
  EXPECT_LT(index_of(k, OpKind::Remap), index_of(k, OpKind::Relax));
}

TEST(ChannelLowering, RemapChangesRegionAndNothingElse) {
  Pair p = make_pair();
  p.g.channel(p.ch).remap(RemapFlags::Conservative);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind != OpKind::Remap) continue;
    const auto& in = r.dag.arc(n.inputs.front()).type;
    const auto& out = r.dag.arc(n.output).type;
    EXPECT_NE(in.region, out.region);
    EXPECT_EQ(in.units, out.units);
    EXPECT_EQ(in.components, out.components);
    EXPECT_EQ(in.conservation, out.conservation);
    return;
  }
  FAIL() << "no Remap node emitted";
}

TEST(ChannelLowering, ExtrapolatePrecedesRemap) {
  Pair p = make_pair();
  p.g.channel(p.ch).extrapolate(1).remap(RemapFlags::Conservative);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds_in_order(r.dag);
  ASSERT_TRUE(contains(k, OpKind::Extrapolate));
  EXPECT_LT(index_of(k, OpKind::Extrapolate), index_of(k, OpKind::Remap));
}

// Spec §4.3: lag(k) is a TimeLevel offset of -k, not a node.
TEST(ChannelLowering, LagShiftsTimeLevelWithoutEmittingANode) {
  Pair p = make_pair();
  p.g.channel(p.ch).lag(1).allow_time_level_mismatch("lag by design");
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  bool found_shifted = false;
  for (std::uint32_t i = 0; i < r.dag.arc_count(); ++i) {
    if (r.dag.arc(fuse::ArcId{i}).type.time.offset == Rational{-1}) {
      found_shifted = true;
    }
  }
  EXPECT_TRUE(found_shifted);
}

TEST(ChannelLowering, AccumulateRecordsItsOpInAttrs) {
  Pair p = make_pair();
  p.g.channel(p.ch).accumulate(AccumOp::Integral);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind == OpKind::Accumulate) {
      EXPECT_EQ(n.attrs.accum, AccumOp::Integral);
      return;
    }
  }
  FAIL() << "no Accumulate node emitted";
}

TEST(ChannelLowering, RelaxRecordsOmega) {
  Pair p = make_pair();
  p.g.channel(p.ch).relax(0.35);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind == OpKind::Relax) {
      EXPECT_DOUBLE_EQ(n.attrs.omega, 0.35);
      return;
    }
  }
  FAIL() << "no Relax node emitted";
}

TEST(ChannelLowering, EveryNodeCarriesItsOriginChannel) {
  Pair p = make_pair();
  p.g.channel(p.ch).accumulate(AccumOp::Mean).relax(0.5);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind == OpKind::Source || n.kind == OpKind::Sink) continue;
    EXPECT_TRUE(n.origin.valid()) << to_cstring(n.kind);
  }
}

// ---------------------------------------------------------------------------
// Regression: a Bidirectional field carries TWO values, and lowering must not
// alias them. See LoweringContext's invariant in include/fuse/ir/lowering.hpp.
// ---------------------------------------------------------------------------

namespace {

// a.a_x --ch0(lag 1)--> b.b_s (Bidirectional) --ch1--> a.a_y
struct BidiGraph {
  CouplingGraph g;
  fuse::FieldId b_s;
};

BidiGraph make_bidi() {
  BidiGraph p;
  auto& g = p.g;
  const auto m = g.add_mesh(MeshDecl{.name = "m"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 2});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = Rational{1}});
  const auto a_x = g.add_field(a, FieldSpec{
      .name = "a_x", .region = r, .units = u::kKelvin,
      .intent = Intent::Provides});
  const auto a_y = g.add_field(a, FieldSpec{
      .name = "a_y", .region = r, .units = u::kKelvin,
      .intent = Intent::Requires});
  p.b_s = g.add_field(b, FieldSpec{
      .name = "b_s", .region = r, .units = u::kKelvin,
      .intent = Intent::Bidirectional});

  const auto ch0 = g.add_channel("ch0_ax_to_bs", a_x, p.b_s);
  g.channel(ch0).lag(1).allow_time_level_mismatch("explicit");
  const auto ch1 = g.add_channel("ch1_bs_to_ay", p.b_s, a_y);
  g.channel(ch1).allow_time_level_mismatch("staggered diagnostics");
  return p;
}

}  // namespace

TEST(ChannelLowering, ChannelSourcedFromABidirectionalFieldReadsItsOwnSource) {
  const BidiGraph p = make_bidi();
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  // Locate b.b_s's Source node and the head of ch1's chain.
  fuse::NodeId source;
  fuse::NodeId ch1_head;
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const fuse::NodeId id{i};
    const auto& n = r.dag.node(id);
    if (n.kind == OpKind::Source && n.label == "b.b_s") source = id;
    if (n.origin.valid() && n.origin.value() == 1 && !ch1_head.valid()) {
      ch1_head = id;
    }
  }
  ASSERT_TRUE(source.valid());
  ASSERT_TRUE(ch1_head.valid()) << "ch1 emitted no node";

  const auto& head = r.dag.node(ch1_head);
  ASSERT_EQ(head.inputs.size(), 1u);
  EXPECT_EQ(head.inputs.front().value(), r.dag.node(source).output.value())
      << "a channel sourced from a Bidirectional field must read the field's "
         "own Source arc, not whatever the incoming channel last wrote to it";

  // Consequences of the aliasing, each of which the acyclicity gate relies on.
  EXPECT_FALSE(r.dag.consumers_of(r.dag.node(source).output).empty())
      << "b.b_s's Source was orphaned, so no dependence runs through "
         "component b";
  EXPECT_EQ(r.dag.arc(head.output).type.time.offset, Rational{0})
      << "ch0's lag offset leaked into ch1's chain";
}

// The other half of the invariant: the Sink still consumes what the channel
// delivered, not the component's own Source.
TEST(ChannelLowering, BidirectionalSinkStillConsumesTheDeliveredValue) {
  const BidiGraph p = make_bidi();
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind != OpKind::Sink || n.label != "b.b_s") continue;
    ASSERT_EQ(n.inputs.size(), 1u);
    const fuse::NodeId producer = r.dag.arc(n.inputs.front()).producer;
    ASSERT_TRUE(producer.valid());
    const auto& prod = r.dag.node(producer);
    EXPECT_TRUE(prod.origin.valid() && prod.origin.value() == 0)
        << "b.b_s's Sink must consume ch0's delivered value, produced by "
        << to_cstring(prod.kind) << " '" << prod.label << "'";
    return;
  }
  FAIL() << "no Sink emitted for b.b_s";
}
