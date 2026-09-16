#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/ir/lowering.hpp>

#include <algorithm>

using fuse::CouplingGraph;
using fuse::derive_checkpoint_set;
using fuse::OpKind;
namespace u = fuse::units;

namespace {

CouplingGraph iterated_pair(int max_iters) {
  CouplingGraph g;
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(fuse::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(fuse::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).relax(0.8);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "outer", .channels = {ch}, .max_iterations = max_iters});
  return g;
}

}  // namespace

TEST(IterationRegions, RegionAppearsInTheDag) {
  const auto r = lower(iterated_pair(30));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  ASSERT_EQ(r.dag.region_count(), 1u);
  EXPECT_EQ(r.dag.region_at(fuse::RegionIdx{0}).name, "outer");
  EXPECT_EQ(r.dag.region_at(fuse::RegionIdx{0}).max_iterations, 30);
}

TEST(IterationRegions, BodyContainsTheChannelsNodes) {
  const auto r = lower(iterated_pair(10));
  ASSERT_TRUE(r.ok());
  const auto& region = r.dag.region_at(fuse::RegionIdx{0});
  EXPECT_FALSE(region.body.empty());
  for (const fuse::NodeId n : region.body) {
    EXPECT_TRUE(r.dag.node(n).origin.valid());
  }
}

TEST(IterationRegions, EmitsAResidualPerChannel) {
  const auto r = lower(iterated_pair(10));
  ASSERT_TRUE(r.ok());
  const auto& region = r.dag.region_at(fuse::RegionIdx{0});
  EXPECT_EQ(region.residuals.size(), 1u);

  int residual_nodes = 0;
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    if (r.dag.node(fuse::NodeId{i}).kind == OpKind::Residual) ++residual_nodes;
  }
  EXPECT_EQ(residual_nodes, 1);
}

// Phase 7 Task 7.2 fills this in by dataflow (spec §5.3). `iterated_pair` puts
// .relax(0.8) on its one channel, so the relaxed value is loop-carried: the
// value at sweep k is formed from the value at k-1, which makes its buffer
// state rather than a recomputable transient.
TEST(IterationRegions, CarriedSetHoldsTheRelaxedValue) {
  const auto r = lower(iterated_pair(10));
  ASSERT_TRUE(r.ok());
  const auto& region = r.dag.region_at(fuse::RegionIdx{0});
  ASSERT_EQ(region.carried.size(), 1u) << "one relaxed channel, one carried "
                                          "buffer";
  const fuse::ArcId a = region.carried.front();
  ASSERT_TRUE(a.valid());
  const fuse::NodeId producer = r.dag.arc(a).producer;
  ASSERT_TRUE(producer.valid());
  EXPECT_EQ(r.dag.node(producer).kind, OpKind::Relax);
  // ...and it agrees with the standalone analysis the runner will call.
  const auto set = derive_checkpoint_set(iterated_pair(10), r.dag, region);
  EXPECT_EQ(set.coupler_arcs, region.carried);
}

// The other direction: no relaxation, nothing loop-carried. Not an accident of
// the implementation -- it is the claim that a pure function of live-in values
// is recomputed on a re-sweep rather than saved.
TEST(IterationRegions, CarriedSetIsEmptyWithoutRelaxation) {
  const CouplingGraph plain = [] {
    CouplingGraph h;
    const auto mf = h.add_mesh(fuse::MeshDecl{.name = "mf"});
    const auto ms = h.add_mesh(fuse::MeshDecl{.name = "ms"});
    const auto rf = h.add_region(fuse::Region{.name = "rf", .mesh = mf});
    const auto rs = h.add_region(fuse::Region{.name = "rs", .mesh = ms});
    const auto a = h.add_component(fuse::ComponentDecl{
        .name = "a", .ranks = fuse::RankSet{0, 1}});
    const auto b = h.add_component(fuse::ComponentDecl{
        .name = "b", .ranks = fuse::RankSet{1, 1}});
    const auto s = h.add_field(a, fuse::FieldSpec{
        .name = "f", .region = rf, .units = u::kKelvin,
        .intent = fuse::Intent::Provides});
    const auto t = h.add_field(b, fuse::FieldSpec{
        .name = "f_in", .region = rs, .units = u::kKelvin,
        .intent = fuse::Intent::Requires});
    const auto ch = h.add_channel("c", s, t);
    h.add_iteration_region(fuse::IterationRegionDecl{
        .name = "outer", .channels = {ch}, .max_iterations = 10});
    return h;
  }();
  const auto r = lower(plain);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_TRUE(r.dag.region_at(fuse::RegionIdx{0}).carried.empty());
  // The components still need checkpointing: they are re-advanced over the
  // same interval on every sweep even when no coupler buffer is carried.
  const auto set =
      derive_checkpoint_set(plain, r.dag, r.dag.region_at(fuse::RegionIdx{0}));
  EXPECT_EQ(set.components.size(), 2u);
}

TEST(IterationRegions, DagRemainsAcyclic) {
  const auto r = lower(iterated_pair(10));
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(r.dag.is_acyclic());
}

TEST(IterationRegions, NoRegionsWhenNoneDeclared) {
  CouplingGraph g = iterated_pair(10);
  const CouplingGraph without = [&] {
    CouplingGraph h;
    const auto m = h.add_mesh(fuse::MeshDecl{.name = "m"});
    const auto rr = h.add_region(fuse::Region{.name = "r", .mesh = m});
    const auto c = h.add_component(fuse::ComponentDecl{.name = "solo"});
    h.add_field(c, fuse::FieldSpec{.name = "f", .region = rr,
                                   .intent = fuse::Intent::Provides});
    return h;
  }();
  const auto r = lower(without);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.dag.region_count(), 0u);
}
