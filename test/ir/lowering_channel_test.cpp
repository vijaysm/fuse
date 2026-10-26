#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>

#include <utility>
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

// This is the exact chain from spec Section 5.2.
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
  // Order is the point of the whole design (spec Section 2.1).
  EXPECT_LT(index_of(k, OpKind::Sample), index_of(k, OpKind::Accumulate));
  EXPECT_LT(index_of(k, OpKind::Accumulate), index_of(k, OpKind::Remap));
  EXPECT_LT(index_of(k, OpKind::Remap), index_of(k, OpKind::Relax));
}

// Unit 3 (S6 phase 3, Conformance): a LowFive-bound channel's source field
// sets FieldSpec::hdf5_path; lower_channel() (src/ir/rules_channel.cpp)
// must substitute an ExternalDelivery node for the ordinary Source-arc
// read, and the Transport this channel would otherwise need (fluid/
// solid's rank sets overlap but are not identical, so an ordinary
// channel between them DOES emit one -- MinimalChannelIsSourceRemapSink's
// own Pair, reused unchanged here) must not appear at all. ExternalDelivery
// carries the source field's own producer (here, Source) as its ONE input
// -- an ORDERING edge only (OpKind::ExternalDelivery's own comment, op.hpp;
// MEASURED, a later session: severing it entirely let the scheduler place
// this node before the exporting member's own last substep in the SAME
// sweep, reading stale data) -- its execute() case never reads that
// input; the application's own write is what fills its output.
TEST(ChannelLowering, LowFiveBoundChannelEmitsExternalDeliveryNotTransport) {
  Pair p = make_pair();
  const fuse::Channel& ch = p.g.channel_at(p.ch);
  const_cast<FieldSpec&>(p.g.field(ch.source)).hdf5_path =
      "fluid_out.h5/grid/heat_flux";

  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds_in_order(r.dag);

  EXPECT_TRUE(contains(k, OpKind::ExternalDelivery));
  EXPECT_FALSE(contains(k, OpKind::Transport));
  EXPECT_FALSE(contains(k, OpKind::Epoch));
  // Remap still runs -- the regions still differ -- just fed from the
  // externally-delivered arc instead of a Transport's.
  EXPECT_TRUE(contains(k, OpKind::Remap));
  EXPECT_TRUE(contains(k, OpKind::Sink));
  EXPECT_LT(index_of(k, OpKind::ExternalDelivery), index_of(k, OpKind::Remap));
  EXPECT_LT(index_of(k, OpKind::Remap), index_of(k, OpKind::Sink));

  bool found = false;
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind != OpKind::ExternalDelivery) continue;
    found = true;
    // ONE input, an ordering edge onto the source field's own producer
    // (here, Source) -- not a data dependency (execute()'s own
    // ExternalDelivery case, src/runtime/runner.cpp, never reads it);
    // see this test's own top comment and OpKind::ExternalDelivery's
    // own comment (op.hpp) for why severing it is wrong, not simpler.
    ASSERT_EQ(n.inputs.size(), 1u)
        << "ExternalDelivery must carry its source field's own producer "
           "as an ordering input, not zero inputs";
    EXPECT_EQ(r.dag.node(r.dag.arc(n.inputs.front()).producer).kind, OpKind::Source);
    ASSERT_TRUE(n.output.valid());
    // Same shape (region/units/components/conservation/rank_set) as the
    // source field's own producer would have carried -- the rest of the
    // chain (Remap, Sink) must be unable to tell the difference.
    const auto& out_type = r.dag.arc(n.output).type;
    EXPECT_EQ(out_type.units, u::kHeatFluxDensity);
    EXPECT_EQ(out_type.conservation, Conservation::Extensive);
    // The application's own read lands on the CONSUMER's placement (the
    // same reason this channel now needs no Transport at all), not the
    // producer's.
    EXPECT_EQ(out_type.rank_set, static_cast<int>(p.g.owner_of(ch.target).value()));
  }
  EXPECT_TRUE(found);

  // The source field's own ordinary Source node still exists (its
  // component's own export computation is unaffected) but this channel no
  // longer consumes it -- confirmed by there being exactly one Source node
  // and it feeding no Remap/ExternalDelivery/Sink at all in this DAG.
  int source_count = 0;
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    if (r.dag.node(fuse::NodeId{i}).kind == OpKind::Source) ++source_count;
  }
  EXPECT_EQ(source_count, 1);
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

// Spec Section 4.3: lag(k) is a TimeLevel offset of -k, not a node.
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
  p.g.channel(p.ch).accumulate(AccumOp::Mean);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind == OpKind::Accumulate) {
      EXPECT_EQ(n.attrs.accum, AccumOp::Mean);
      EXPECT_EQ(n.attrs.scale, Rational{1}) << "a Mean carries no time scale";
      return;
    }
  }
  FAIL() << "no Accumulate node emitted";
}

// `AccumOp::Integral` delivers the time integral of the source over each
// target step: `(sum of samples) * dt`, dt the interval between samples,
// which is the SOURCE component's own interval because `Sample` runs once
// per source tick. Lowering records that dt exactly on the Accumulate node
// (`OpAttrs::scale`) and types its output arc in [source] x [time], so the
// target field must be declared in those units. The interpreter used to
// deliver the bare sum, in the source's units, with nothing in the plan
// saying so (kernels.hpp's `accumulate_finalize_integral` was never called).
TEST(ChannelLowering, IntegralRecordsTheSourceIntervalAndIntegratesTheUnits) {
  Pair p;
  auto& g = p.g;
  const auto mf = g.add_mesh(MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(MeshDecl{.name = "ms"});
  const auto rf = g.add_region(Region{.name = "rf", .mesh = mf, .dimension = 2});
  const auto rs = g.add_region(Region{.name = "rs", .mesh = ms, .dimension = 2});
  const auto fluid = g.add_component(ComponentDecl{
      .name = "fluid", .ranks = fuse::RankSet{0, 2}, .interval = Rational{1, 4}});
  const auto solid = g.add_component(ComponentDecl{
      .name = "solid", .ranks = fuse::RankSet{2, 2}, .interval = Rational{1}});
  const auto s = g.add_field(fluid, FieldSpec{
      .name = "heat_flux", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  // Energy per area: the flux density's time integral.
  const auto t = g.add_field(solid, FieldSpec{
      .name = "q_wall", .region = rs, .units = u::kHeatFluxDensity * u::kSecond,
      .conservation = Conservation::Extensive, .intent = Intent::Requires});
  p.ch = g.add_channel("flux", s, t);
  g.channel(p.ch).accumulate(AccumOp::Integral).conservative();

  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  bool found = false;
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind != OpKind::Accumulate) continue;
    found = true;
    EXPECT_EQ(n.attrs.accum, AccumOp::Integral);
    EXPECT_EQ(n.attrs.scale, (Rational{1, 4})) << "dt is the source interval";
    EXPECT_TRUE(r.dag.arc(n.output).type.units == u::kHeatFluxDensity * u::kSecond)
        << r.dag.arc(n.output).type.units.to_string();
    EXPECT_TRUE(r.dag.arc(n.inputs.front()).type.units == u::kHeatFluxDensity)
        << r.dag.arc(n.inputs.front()).type.units.to_string();
  }
  EXPECT_TRUE(found) << "no Accumulate node emitted";
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

// ---- coupler placement (design C, plans/2026-09-23-coupler-placement-and-
// migration.md): THE RULE is source -> C -> target, remap on C's
// decomposition of both meshes, any hop between identical placements elided
// by the same `same_as` test as every other hop. -------------------------

namespace {

struct CouplerPair {
  fuse::CouplingGraph g;
  fuse::ComponentId   src_c, dst_c, coupler_c;
  fuse::ChannelId     ch;
};

/// src -[remap_on coupler]-> dst, three independent placements, one
/// channel. Different mesh regions by default (the ordinary coupling
/// shape), so a Remap is always emitted regardless of flags.
CouplerPair make_coupler_pair(fuse::RankSet src_ranks, fuse::RankSet dst_ranks,
                              fuse::RankSet coupler_ranks,
                              const char* method = "",
                              bool nodal_source = false) {
  using namespace fuse;
  CouplerPair p;
  auto& g = p.g;
  const auto m_src = g.add_mesh({.name = "ms"});
  const auto m_dst = g.add_mesh({.name = "md"});
  const auto r_src = g.add_region({.name = "rs", .mesh = m_src, .dimension = 1});
  const auto r_dst = g.add_region({.name = "rd", .mesh = m_dst, .dimension = 1});
  p.src_c = g.add_component({.name = "src", .ranks = std::move(src_ranks),
                             .interval = Rational{1}});
  p.dst_c = g.add_component({.name = "dst", .ranks = std::move(dst_ranks),
                             .interval = Rational{1}});
  p.coupler_c = g.add_coupler({.name = "C", .ranks = std::move(coupler_ranks)});
  FieldSpec fs{.name = "q", .region = r_src, .intent = Intent::Provides};
  FieldSpec ft{.name = "q", .region = r_dst, .intent = Intent::Requires};
  if (*method) {
    if (nodal_source) { fs.disc = Discretization::ContinuousGLL; fs.order = 1; }
    else              { ft.disc = Discretization::ContinuousGLL; ft.order = 1; }
  }
  const auto out = g.add_field(p.src_c, fs);
  const auto in  = g.add_field(p.dst_c, ft);
  p.ch = g.add_channel("q", out, in);
  auto b = g.channel(p.ch).remap_on(p.coupler_c);
  if (*method) b.remap_method(method);
  return p;
}

struct CouplerShape {
  std::size_t transports = 0, epochs = 0, remaps = 0;
  int remap_in_rs = -2, remap_out_rs = -2;
  std::vector<std::pair<int, int>> transport_hops;  // (src_rank_set, dst_rank_set), in order
};

CouplerShape coupler_shape_of(const fuse::CouplingGraph& g) {
  const auto lowered = fuse::lower(g, {});
  CouplerShape s;
  for (const fuse::NodeId n : lowered.dag.topological_order()) {
    const auto& node = lowered.dag.node(n);
    if (node.kind == fuse::OpKind::Transport) {
      ++s.transports;
      s.transport_hops.emplace_back(node.attrs.src_rank_set, node.attrs.dst_rank_set);
    }
    if (node.kind == fuse::OpKind::Epoch) ++s.epochs;
    if (node.kind == fuse::OpKind::Remap) {
      ++s.remaps;
      s.remap_in_rs  = lowered.dag.arc(node.inputs.front()).type.rank_set;
      s.remap_out_rs = lowered.dag.arc(node.output).type.rank_set;
    }
  }
  return s;
}

}  // namespace

TEST(CouplerLowering, DisjointCouplerGetsTwoDataMovesAndOneRemap) {
  const CouplerPair p = make_coupler_pair(fuse::RankSet{0, 2}, fuse::RankSet{4, 2},
                                          fuse::RankSet{8, 1});
  ASSERT_TRUE(fuse::lower(p.g).ok());
  const CouplerShape s = coupler_shape_of(p.g);
  EXPECT_EQ(s.transports, 2u);
  EXPECT_EQ(s.epochs, 2u);
  EXPECT_EQ(s.remaps, 1u);
  // rank-local remap: no placement move OF ITS OWN, so it runs entirely on
  // the coupler's ranks, on both sides.
  EXPECT_EQ(s.remap_in_rs, static_cast<int>(p.coupler_c.value()));
  EXPECT_EQ(s.remap_out_rs, static_cast<int>(p.coupler_c.value()));
  ASSERT_EQ(s.transport_hops.size(), 2u);
  EXPECT_EQ(s.transport_hops[0],
            std::make_pair(static_cast<int>(p.src_c.value()),
                           static_cast<int>(p.coupler_c.value())));
  EXPECT_EQ(s.transport_hops[1],
            std::make_pair(static_cast<int>(p.coupler_c.value()),
                           static_cast<int>(p.dst_c.value())));
}

TEST(CouplerLowering, CouplerEqualToTargetReproducesTodaysIRExactly) {
  // "C = target" means the SAME ComponentId, not merely a same_as RankSet:
  // a distinct Coupler-kind component whose ranks happen to coincide with
  // the target's is still its own placement identity (per-placement
  // weights are keyed by ComponentId, not by RankSet content -- commit
  // a3fb10a), and legitimately elides only the data move, not the Remap's
  // placement. So this graph declares NO separate coupler at all;
  // `remap_on` names the channel's own target component directly, which
  // `validate()`'s E-CHANNEL-REMAP-ON also accepts.
  using namespace fuse;
  CouplingGraph g;
  const auto m_src = g.add_mesh({.name = "ms"});
  const auto m_dst = g.add_mesh({.name = "md"});
  const auto r_src = g.add_region({.name = "rs", .mesh = m_src, .dimension = 1});
  const auto r_dst = g.add_region({.name = "rd", .mesh = m_dst, .dimension = 1});
  const auto src_c = g.add_component({.name = "src", .ranks = RankSet{0, 2},
                                      .interval = Rational{1}});
  const auto dst_c = g.add_component({.name = "dst", .ranks = RankSet{4, 2},
                                      .interval = Rational{1}});
  const auto out = g.add_field(src_c, FieldSpec{.name = "q", .region = r_src,
                                                .intent = Intent::Provides});
  const auto in = g.add_field(dst_c, FieldSpec{.name = "q", .region = r_dst,
                                               .intent = Intent::Requires});
  const auto ch = g.add_channel("q", out, in);
  const auto lowered_plain = lower(g);
  ASSERT_TRUE(lowered_plain.ok()) << lowered_plain.diags.to_string();

  g.channel(ch).remap_on(dst_c);
  const auto lowered_with = lower(g);
  ASSERT_TRUE(lowered_with.ok()) << lowered_with.diags.to_string();

  EXPECT_EQ(dump_text(lowered_with.dag), dump_text(lowered_plain.dag))
      << "C = target's own component id must reproduce today's IR byte for byte";
}

TEST(CouplerLowering, ACouplerColocatedWithTargetStillIdentifiesAsItsOwnPlacement) {
  // The edge case CouplerEqualToTargetReproducesTodaysIRExactly's comment
  // names: a genuinely separate coupler whose RankSet happens to equal the
  // target's. same_as elides the SECOND data move (nothing to move), but
  // the Remap still runs under the coupler's OWN ComponentId, not the
  // target's -- placed_view() and per-placement weights key on identity,
  // not on rank-set content, so a coupler is not "the same placement" as a
  // same-ranks target just because no bytes need to move between them.
  const CouplerPair p = make_coupler_pair(fuse::RankSet{0, 2}, fuse::RankSet{4, 2},
                                          fuse::RankSet{4, 2});  // C's ranks == dst's
  ASSERT_TRUE(fuse::lower(p.g).ok());
  const CouplerShape s = coupler_shape_of(p.g);
  ASSERT_EQ(s.transport_hops.size(), 1u)
      << "only the first hop moves data; C -> target is same_as and elided";
  EXPECT_EQ(s.transport_hops[0],
            std::make_pair(static_cast<int>(p.src_c.value()),
                           static_cast<int>(p.coupler_c.value())));
  EXPECT_EQ(s.remap_in_rs, static_cast<int>(p.coupler_c.value()));
  EXPECT_EQ(s.remap_out_rs, static_cast<int>(p.coupler_c.value()))
      << "the coupler's own identity, even though it shares the target's ranks";
}

TEST(CouplerLowering, RemapOnSourceRunsRemapOnSourceRanksThenTransportsToTarget) {
  const CouplerPair p = make_coupler_pair(fuse::RankSet{0, 2}, fuse::RankSet{4, 2},
                                          fuse::RankSet{0, 2});  // C == src
  ASSERT_TRUE(fuse::lower(p.g).ok());
  const CouplerShape s = coupler_shape_of(p.g);
  // The first hop (source -> C) is elided: C IS the source's placement.
  ASSERT_EQ(s.transport_hops.size(), 1u);
  EXPECT_EQ(s.transport_hops[0],
            std::make_pair(static_cast<int>(p.src_c.value()),
                           static_cast<int>(p.dst_c.value())))
      << "the only move left is C (== source) -> target";
  EXPECT_EQ(s.remaps, 1u);
  EXPECT_EQ(s.remap_in_rs, static_cast<int>(p.src_c.value()))
      << "the Remap ran on the source's own decomposition, before any move";
  EXPECT_EQ(s.remap_out_rs, static_cast<int>(p.src_c.value()));
}

TEST(CouplerLowering, PartiallyOverlappingCouplerStillGetsBothMoves) {
  // C shares rank 2 with src, but is not IDENTICAL to it -- same_as is
  // false, so the first hop is a real move, mirroring
  // LoweringChannel.PartiallySharedPlacementsStillNeedADataMove.
  const CouplerPair p = make_coupler_pair(
      fuse::RankSet{std::vector<int>{0, 1, 2}}, fuse::RankSet{5, 1},
      fuse::RankSet{std::vector<int>{2, 3}});
  ASSERT_TRUE(fuse::lower(p.g).ok());
  const CouplerShape s = coupler_shape_of(p.g);
  EXPECT_EQ(s.transports, 2u);
  EXPECT_EQ(s.epochs, 2u);
}

TEST(CouplerLowering, AConsistentRemapWithACouplerMovesToCThenTransportsToTarget) {
  // The distributed operator IS the source->C move (fuse/model/remap_kind.hpp);
  // the design's second hop, C->target, still needs an ordinary Transport.
  const CouplerPair p = make_coupler_pair(fuse::RankSet{0, 2}, fuse::RankSet{4, 2},
                                          fuse::RankSet{8, 1}, "consistent",
                                          /*nodal_source=*/true);
  ASSERT_TRUE(fuse::lower(p.g).ok());
  const CouplerShape s = coupler_shape_of(p.g);
  EXPECT_EQ(s.remaps, 1u);
  EXPECT_EQ(s.remap_in_rs, static_cast<int>(p.src_c.value()))
      << "the distributed operator reads on the source placement";
  EXPECT_EQ(s.remap_out_rs, static_cast<int>(p.coupler_c.value()))
      << "and writes on the coupler placement, moving the data itself";
  ASSERT_EQ(s.transport_hops.size(), 1u)
      << "only the second hop (C -> target) is an ordinary Transport";
  EXPECT_EQ(s.transport_hops[0],
            std::make_pair(static_cast<int>(p.coupler_c.value()),
                           static_cast<int>(p.dst_c.value())));
  EXPECT_EQ(s.epochs, 2u) << "before-distributed-remap, then after-coupler-transport";
}

TEST(CouplerLowering, RemapOnUnusedWhenChannelWouldNeverRemapReproducesNoCouplerIR) {
  // Same region on both sides (so no region-driven Remap), no flags, and a
  // non-moving method: could_ever_emit_remap is false, so remap_on must be
  // ignored entirely -- one plain Transport straight from source to target,
  // not source->C->target.
  using namespace fuse;
  CouplingGraph g;
  const auto mesh = g.add_mesh({.name = "m"});
  const auto reg  = g.add_region({.name = "r", .mesh = mesh, .dimension = 1});
  const auto src_c = g.add_component({.name = "src", .ranks = RankSet{0, 2},
                                      .interval = Rational{1}});
  const auto dst_c = g.add_component({.name = "dst", .ranks = RankSet{4, 2},
                                      .interval = Rational{1}});
  const auto coupler = g.add_coupler({.name = "C", .ranks = RankSet{8, 1}});
  const auto out = g.add_field(src_c, FieldSpec{.name = "q", .region = reg,
                                                .intent = Intent::Provides});
  const auto in  = g.add_field(dst_c, FieldSpec{.name = "q", .region = reg,
                                                .intent = Intent::Requires});
  const auto ch = g.add_channel("q", out, in);
  g.channel(ch).remap_on(coupler);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  CouplerShape s = coupler_shape_of(g);
  EXPECT_EQ(s.remaps, 0u);
  ASSERT_EQ(s.transport_hops.size(), 1u)
      << "remap_on has no effect here: exactly the no-coupler shape";
  EXPECT_EQ(s.transport_hops[0],
            std::make_pair(static_cast<int>(src_c.value()),
                           static_cast<int>(dst_c.value())));
}

// ---------------------------------------------------------------------------
// Review 2026-10-11 IS-9: a merged field's Sink judged its time-mismatch
// authorization by the FIRST-DECLARED writer, not the one whose contribution
// the merge actually adopted (type_check()'s `n.origin`, set by
// emit_field_sink()'s own "first channel found" scan). Two lagged writers
// into one merged field, with the SECOND-declared one more recent (lag 1)
// than the first (lag 2): the merge's own rule (lower()'s own comment,
// "the combined value's time level is the LATEST of its contributions")
// picks the second writer's level, but the old scan still authorized
// against the first. Below, the first writer IS authorized and the second
// (the real "newest") is NOT -- before the fix this was wrongly admitted;
// after the fix, type_check must see the unauthorized mismatch on the
// writer that actually matters and refuse with E-IR-TYPE.
// ---------------------------------------------------------------------------

namespace {

CouplingGraph two_lagged_writers_merge(bool authorize_first,
                                       bool authorize_second) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m});
  const auto dst = g.add_component(ComponentDecl{
      .name = "dst", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto in = g.add_field(dst, FieldSpec{
      .name = "f_in", .region = r, .units = u::kKelvin,
      .conservation = Conservation::Extensive, .intent = Intent::Requires});

  const auto w1 = g.add_component(ComponentDecl{
      .name = "w1", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto out1 = g.add_field(w1, FieldSpec{
      .name = "f_out", .region = r, .units = u::kKelvin,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  // Declared FIRST, lagged 2 (the OLDER of the two contributions).
  const auto ch1 = g.add_channel("ch1", out1, in);
  g.channel(ch1).lag(2);
  if (authorize_first) {
    g.channel(ch1).allow_time_level_mismatch("writer 1, lag 2");
  }

  const auto w2 = g.add_component(ComponentDecl{
      .name = "w2", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto out2 = g.add_field(w2, FieldSpec{
      .name = "f_out", .region = r, .units = u::kKelvin,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  // Declared SECOND, lagged 1 -- more recent than ch1, so the merge adopts
  // ITS time level as the combined value's (lower()'s own "LATEST of its
  // contributions" rule).
  const auto ch2 = g.add_channel("ch2", out2, in);
  g.channel(ch2).lag(1);
  if (authorize_second) {
    g.channel(ch2).allow_time_level_mismatch("writer 2, lag 1");
  }

  g.merge(in, fuse::MergeOp::Sum);
  return g;
}

}  // namespace

TEST(ChannelLowering,
    MergedFieldAuthorizationFollowsTheActualNewestWriterNotTheFirstDeclared) {
  // The newest (second-declared) writer is unauthorized; the first-declared,
  // older writer is. Before the fix, type_check checked the first-declared
  // channel's flag and wrongly admitted this graph; after the fix it checks
  // the second (actually-newest) channel's flag and refuses it.
  const CouplingGraph g =
      two_lagged_writers_merge(/*authorize_first=*/true,
                               /*authorize_second=*/false);
  const auto r = lower(g);
  ASSERT_FALSE(r.ok())
      << "writer 2's unauthorized mismatch must not be hidden behind "
         "writer 1's authorization";
  bool saw_type_error = false;
  for (const auto& d : r.diags.items()) {
    if (d.code == "E-IR-TYPE") saw_type_error = true;
  }
  EXPECT_TRUE(saw_type_error) << r.diags.to_string();
}

TEST(ChannelLowering, MergedFieldAuthorizedOnTheActualNewestWriterIsAccepted) {
  // The newest (second-declared) writer IS authorized; the first-declared,
  // older one is not (and does not need to be -- its contribution is not
  // the one the merge adopted). Must be accepted.
  const CouplingGraph g =
      two_lagged_writers_merge(/*authorize_first=*/false,
                               /*authorize_second=*/true);
  const auto r = lower(g);
  EXPECT_TRUE(r.ok()) << r.diags.to_string();
}

// ---------------------------------------------------------------------------
// BM-1 (review 2026-10-11): an Extensive-target channel refuses at the
// Conservation stage (connect.hpp's `check_channel()`) whenever its declared
// transfer is not conservative -- a plain copy, here. `Evaluation::finish()`
// (src/model/compatibility.cpp) used to clear the WHOLE operator chain on
// that refusal, including a Transform-phase `UnitScale` step the Unit stage
// had ALREADY authorized (Unit runs before Conservation): a scale-only unit
// mismatch (kPa -> Pa) on such a channel lost its scale and surfaced here as
// a bare, unexplained `E-IR-TYPE` -- the type never converts, so `Sink`'s
// `assignable_from` (op.hpp) sees kilopascals where it required pascals.
// `finish()` now keeps a Transform-phase step across a later-stage refusal,
// so this channel's Scale operator still gets emitted and the type is
// correct by the time `type_check()` runs; the authoring-time refusal itself
// is `validate()`'s job (test/model/validate_test.cpp's own BM-1 tests pin
// that it still fires, as `E-CONN-CONSERVATION`, alongside the inserted-scale
// warning).
// ---------------------------------------------------------------------------

namespace {

CouplingGraph extensive_scale_only_mismatch_graph() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  fuse::Units pascal = u::kPascal;
  fuse::Units kilopascal = pascal;
  kilopascal.scale = Rational{1000};
  const auto src = g.add_field(a, FieldSpec{
      .name = "p_out", .region = r, .units = kilopascal,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  const auto dst = g.add_field(b, FieldSpec{
      .name = "p_in", .region = r, .units = pascal,
      .conservation = Conservation::Extensive, .intent = Intent::Requires});
  // Plain copy: no .conservative(), so the Conservation stage refuses.
  g.add_channel("p_link", src, dst);
  return g;
}

}  // namespace

TEST(ChannelLowering,
    ExtensiveTargetWithAScaleOnlyMismatchGetsItsScaleNotABareTypeError) {
  const CouplingGraph g = extensive_scale_only_mismatch_graph();
  const auto r = lower(g);
  bool saw_type_error = false;
  for (const auto& d : r.diags.items()) {
    if (d.code == "E-IR-TYPE") saw_type_error = true;
  }
  EXPECT_FALSE(saw_type_error) << r.diags.to_string();
  EXPECT_TRUE(r.ok()) << r.diags.to_string();
}

