#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/ir/lowering.hpp>

#include <algorithm>
#include <vector>

using soma::Conservation;
using soma::CouplingGraph;
using soma::OpKind;
using soma::RemapFlags;
namespace u = soma::units;

namespace {

// Same-rank or cross-rank pair, with conservation switchable.
CouplingGraph pair(bool disjoint_ranks, Conservation cons) {
  CouplingGraph g;
  const auto mf = g.add_mesh(soma::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(soma::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(soma::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(soma::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(soma::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 2}});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = disjoint_ranks ? 2 : 0, .count = 2}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = cons, .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = cons, .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).remap(RemapFlags::Conservative);
  return g;
}

std::vector<OpKind> kinds(const soma::OperatorDAG& d) {
  std::vector<OpKind> out;
  for (const soma::NodeId n : d.topological_order()) out.push_back(d.node(n).kind);
  return out;
}

int index_of(const std::vector<OpKind>& v, OpKind k) {
  return static_cast<int>(std::find(v.begin(), v.end(), k) - v.begin());
}

bool contains(const std::vector<OpKind>& v, OpKind k) {
  return std::find(v.begin(), v.end(), k) != v.end();
}

}  // namespace

TEST(DerivedOps, InsertsTransportWhenRankSetsDiffer) {
  const auto r = lower(pair(/*disjoint_ranks=*/true, Conservation::Extensive));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_TRUE(contains(kinds(r.dag), OpKind::Transport));
}

TEST(DerivedOps, NoTransportWhenRankSetsMatch) {
  const auto r = lower(pair(/*disjoint_ranks=*/false, Conservation::Extensive));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_FALSE(contains(kinds(r.dag), OpKind::Transport));
}

TEST(DerivedOps, TransportRecordsSourceAndDestinationRankSets) {
  const auto r = lower(pair(true, Conservation::Extensive));
  ASSERT_TRUE(r.ok());
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind != OpKind::Transport) continue;
    EXPECT_EQ(n.attrs.src_rank_set, 0);
    EXPECT_EQ(n.attrs.dst_rank_set, 1);
    return;
  }
  FAIL() << "no Transport emitted";
}

// v1 policy: remap runs on the target's ranks, so Transport precedes Remap.
TEST(DerivedOps, TransportPrecedesRemap) {
  const auto r = lower(pair(true, Conservation::Extensive));
  ASSERT_TRUE(r.ok());
  const auto k = kinds(r.dag);
  EXPECT_LT(index_of(k, OpKind::Transport), index_of(k, OpKind::Remap));
}

TEST(DerivedOps, InsertsConstrainForExtensiveFields) {
  const auto r = lower(pair(false, Conservation::Extensive));
  ASSERT_TRUE(r.ok());
  const auto k = kinds(r.dag);
  ASSERT_TRUE(contains(k, OpKind::Constrain));
  EXPECT_LT(index_of(k, OpKind::Remap), index_of(k, OpKind::Constrain));
}

TEST(DerivedOps, NoConstrainWhenConservationIsNone) {
  const auto r = lower(pair(false, Conservation::None));
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(contains(kinds(r.dag), OpKind::Constrain));
}

TEST(DerivedOps, AlwaysInsertsDiagnoseAfterRemap) {
  for (const Conservation c : {Conservation::None, Conservation::Extensive}) {
    const auto r = lower(pair(false, c));
    ASSERT_TRUE(r.ok()) << r.diags.to_string();
    const auto k = kinds(r.dag);
    ASSERT_TRUE(contains(k, OpKind::Diagnose)) << to_cstring(c);
    EXPECT_LT(index_of(k, OpKind::Remap), index_of(k, OpKind::Diagnose));
  }
}

TEST(DerivedOps, DiagnoseIsATypePreservingPassThrough) {
  const auto r = lower(pair(false, Conservation::Extensive));
  ASSERT_TRUE(r.ok());
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind != OpKind::Diagnose) continue;
    EXPECT_EQ(r.dag.arc(n.inputs.front()).type, r.dag.arc(n.output).type);
    return;
  }
  FAIL() << "no Diagnose emitted";
}

// The full spec §5.2 chain, end to end.
TEST(DerivedOps, ReproducesTheSpecChainOrder) {
  CouplingGraph g = pair(true, Conservation::Extensive);
  g.channel(soma::ChannelId{0}).accumulate(soma::AccumOp::Mean).relax(0.7);
  const auto r = lower(g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds(r.dag);
  // Sample, Accumulate, Transport, Remap, Constrain, Diagnose, Relax, Transport
  EXPECT_LT(index_of(k, OpKind::Sample),     index_of(k, OpKind::Accumulate));
  EXPECT_LT(index_of(k, OpKind::Accumulate), index_of(k, OpKind::Remap));
  EXPECT_LT(index_of(k, OpKind::Remap),      index_of(k, OpKind::Constrain));
  EXPECT_LT(index_of(k, OpKind::Constrain),  index_of(k, OpKind::Diagnose));
  EXPECT_LT(index_of(k, OpKind::Diagnose),   index_of(k, OpKind::Relax));
}
