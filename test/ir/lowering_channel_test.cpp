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
      .name = "fluid", .ranks = fuse::RankSet{0, 2}, .interval = Rational{1}});
  const auto solid = g.add_component(ComponentDecl{
      .name = "solid", .ranks = fuse::RankSet{2, 2}, .interval = Rational{1}});
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
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});
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

// ---- shared placements ------------------------------------------------

namespace {

/// Two components with the given placements and one channel between them.
fuse::CouplingGraph shared_pair(fuse::RankSet a, fuse::RankSet b) {
  using namespace fuse;
  CouplingGraph g;
  const auto mesh = g.add_mesh({.name = "m", .source = "generated:cube(2)"});
  const auto reg  = g.add_region({.name = "iface", .mesh = mesh,
                                  .entity_set = "top", .dimension = 2});
  const auto src = g.add_component({.name = "src", .ranks = std::move(a),
                                    .interval = Rational{1}});
  const auto dst = g.add_component({.name = "dst", .ranks = std::move(b),
                                    .interval = Rational{1}});
  const auto out = g.add_field(src, {.name = "q", .region = reg,
                                     .intent = Intent::Provides});
  const auto in  = g.add_field(dst, {.name = "q", .region = reg,
                                     .intent = Intent::Requires});
  g.add_channel("q", out, in);
  return g;
}

std::size_t transports(const fuse::CouplingGraph& g) {
  fuse::Diagnostics d;
  const auto lowered = fuse::lower(g, {});
  std::size_t n = 0;
  for (std::uint32_t i = 0; i < lowered.dag.node_count(); ++i) {
    if (lowered.dag.node(fuse::NodeId{i}).kind == fuse::OpKind::Transport) ++n;
  }
  return n;
}

}  // namespace

TEST(LoweringChannel, IdenticalPlacementsNeedNoDataMove) {
  // Co-located components: every value is already on the rank that wants it.
  EXPECT_EQ(transports(shared_pair(fuse::RankSet{0, 4}, fuse::RankSet{0, 4})), 0u);
  EXPECT_EQ(transports(shared_pair(fuse::RankSet{std::vector<int>{0, 2, 4}},
                                   fuse::RankSet{std::vector<int>{0, 2, 4}})), 0u);
}

TEST(LoweringChannel, PartiallySharedPlacementsStillNeedADataMove) {
  // The case that used to be silently skipped. {0,1,2,3} -> {2,3,4,5} shares
  // 2 and 3, so ranks 4 and 5 can only be fed by a move; suppressing it
  // because the placements touch would deliver nothing to them at all, with
  // no diagnostic. The move is emitted; transport setup then refuses the
  // overlap until send/recv plans exist, which is a failure and not a wrong
  // answer.
  EXPECT_EQ(transports(shared_pair(fuse::RankSet{0, 4},
                                   fuse::RankSet{std::vector<int>{2, 3, 4, 5}})), 1u);
  // One shared rank is enough to have been suppressed before.
  EXPECT_EQ(transports(shared_pair(fuse::RankSet{0, 2},
                                   fuse::RankSet{std::vector<int>{1, 2}})), 1u);
}

TEST(LoweringChannel, DisjointPlacementsNeedADataMove) {
  EXPECT_EQ(transports(shared_pair(fuse::RankSet{0, 2}, fuse::RankSet{2, 2})), 1u);
  // Strided and disjoint: evens to odds.
  EXPECT_EQ(transports(shared_pair(fuse::RankSet{std::vector<int>{0, 2, 4}},
                                   fuse::RankSet{std::vector<int>{1, 3, 5}})), 1u);
}

// ---- a Remap that moves the data itself ----------------------------------

namespace {

/// A nodal-to-cell channel across two placements, with the given method.
fuse::CouplingGraph nodal_pair(fuse::RankSet a, fuse::RankSet b, const char* method,
                               bool nodal_source) {
  using namespace fuse;
  CouplingGraph g;
  const auto m_a = g.add_mesh({.name = "ma", .source = "n/a"});
  const auto m_b = g.add_mesh({.name = "mb", .source = "n/a"});
  const auto r_a = g.add_region({.name = "ra", .mesh = m_a, .dimension = 1});
  const auto r_b = g.add_region({.name = "rb", .mesh = m_b, .dimension = 1});
  const auto src = g.add_component({.name = "src", .ranks = std::move(a), .interval = Rational{1}});
  const auto dst = g.add_component({.name = "dst", .ranks = std::move(b), .interval = Rational{1}});
  FieldSpec fs{.name = "q", .region = r_a, .intent = Intent::Provides};
  FieldSpec ft{.name = "q", .region = r_b, .intent = Intent::Requires};
  if (nodal_source) { fs.disc = Discretization::ContinuousGLL; fs.order = 1; }
  else              { ft.disc = Discretization::ContinuousGLL; ft.order = 1; }
  const auto out = g.add_field(src, fs);
  const auto in  = g.add_field(dst, ft);
  const auto ch = g.add_channel("q", out, in);
  g.channel(ch).remap_method(method);
  return g;
}

struct Shape { std::size_t transports = 0, epochs = 0, remaps = 0; int remap_in_rs = -2, remap_out_rs = -2; };

Shape shape_of(const fuse::CouplingGraph& g) {
  const auto lowered = fuse::lower(g, {});
  Shape s;
  for (std::uint32_t i = 0; i < lowered.dag.node_count(); ++i) {
    const auto& n = lowered.dag.node(fuse::NodeId{i});
    if (n.kind == fuse::OpKind::Transport) ++s.transports;
    if (n.kind == fuse::OpKind::Epoch) ++s.epochs;
    if (n.kind == fuse::OpKind::Remap) {
      ++s.remaps;
      s.remap_in_rs  = lowered.dag.arc(n.inputs.front()).type.rank_set;
      s.remap_out_rs = lowered.dag.arc(n.output).type.rank_set;
    }
  }
  return s;
}

}  // namespace

// fuse/model/remap_kind.hpp. A consistent-interpolation remap on a nodal side
// exchanges the values itself, reading each side where it lives; a Transport
// in front of it would deliver the source values onto the target placement's
// view of the SOURCE mesh -- empty on a disjoint placement -- and the
// operator would then sum nothing. That is what the host measured on
// 2026-09-25: H exact, H^T D delivering zero. So the lowering emits the
// agreement point and a Remap whose output changes placement, and no move.
TEST(LoweringChannel, AConsistentRemapOnANodalSideMovesThePlacementItself) {
  for (const bool nodal_source : {true, false}) {
    const char* method = nodal_source ? "consistent" : "consistent-transpose";
    const Shape s = shape_of(nodal_pair(fuse::RankSet{0, 2}, fuse::RankSet{2, 2}, method, nodal_source));
    EXPECT_EQ(s.transports, 0u) << method;
    EXPECT_EQ(s.epochs, 1u) << method << ": the agreement point stays";
    EXPECT_EQ(s.remaps, 1u) << method;
    EXPECT_EQ(s.remap_in_rs, 0) << method << ": the Remap reads on the source placement";
    EXPECT_EQ(s.remap_out_rs, 1) << method << ": and writes on the target placement";
  }
}

TEST(LoweringChannel, AConsistentRemapWithTheWrongOrientationStillGetsAMove) {
  // "consistent" needs a nodal SOURCE. With the nodal side on the target the
  // predicate is false and the default lowering applies; the runtime then
  // refuses the orientation mismatch by name (E-RT-REMAP-NODAL-PARALLEL).
  const Shape s = shape_of(nodal_pair(fuse::RankSet{0, 2}, fuse::RankSet{2, 2}, "consistent",
                                      /*nodal_source=*/false));
  EXPECT_EQ(s.transports, 1u);
  EXPECT_EQ(s.remap_in_rs, s.remap_out_rs) << "the Transport changed the placement, not the Remap";
}

TEST(LoweringChannel, AConsistentRemapOnOnePlacementNeedsNoMoveAndChangesNothing) {
  const Shape s = shape_of(nodal_pair(fuse::RankSet{0, 4}, fuse::RankSet{0, 4}, "consistent-transpose",
                                      /*nodal_source=*/false));
  EXPECT_EQ(s.transports, 0u);
  EXPECT_EQ(s.epochs, 0u);
  EXPECT_EQ(s.remap_in_rs, s.remap_out_rs);
}

