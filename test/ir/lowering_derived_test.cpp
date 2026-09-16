#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/ir/lowering.hpp>

#include <algorithm>
#include <vector>

using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::OpKind;
using fuse::RemapFlags;
namespace u = fuse::units;

namespace {

// Same-rank or cross-rank pair, with conservation switchable.
CouplingGraph pair(bool disjoint_ranks, Conservation cons) {
  CouplingGraph g;
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(fuse::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(fuse::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 2}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{disjoint_ranks ? 2 : 0, 2}});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = cons, .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = cons, .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).remap(RemapFlags::Conservative);
  return g;
}

std::vector<OpKind> kinds(const fuse::OperatorDAG& d) {
  std::vector<OpKind> out;
  for (const fuse::NodeId n : d.topological_order()) out.push_back(d.node(n).kind);
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
    const auto& n = r.dag.node(fuse::NodeId{i});
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
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind != OpKind::Diagnose) continue;
    EXPECT_EQ(r.dag.arc(n.inputs.front()).type, r.dag.arc(n.output).type);
    return;
  }
  FAIL() << "no Diagnose emitted";
}

// The full spec §5.2 chain, end to end.
TEST(DerivedOps, ReproducesTheSpecChainOrder) {
  CouplingGraph g = pair(true, Conservation::Extensive);
  g.channel(fuse::ChannelId{0}).accumulate(fuse::AccumOp::Mean).relax(0.7);
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
