#include <gtest/gtest.h>
#include <soma/ir/dag.hpp>

#include <algorithm>
#include <stdexcept>

using soma::ArcId;
using soma::ChannelId;
using soma::NodeId;
using soma::OperatorDAG;
using soma::OpAttrs;
using soma::OpKind;
using soma::RegionId;
using soma::ValueType;

namespace {

// Source -> Accumulate -> Sink
OperatorDAG linear_chain() {
  OperatorDAG d;
  const ValueType t{.region = RegionId{0}};
  const ArcId a0 = d.add_arc(t, "v0");
  const ArcId a1 = d.add_arc(t, "v1");
  const ArcId a2 = d.add_arc(t, "v2");
  d.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "src");
  d.add_node(OpKind::Accumulate, {a0}, a1, OpAttrs{}, ChannelId{0}, "acc");
  d.add_node(OpKind::Sink, {a1}, a2, OpAttrs{}, ChannelId{}, "sink");
  return d;
}

}  // namespace

TEST(OperatorDAG, EmptyDagHasNothing) {
  const OperatorDAG d;
  EXPECT_EQ(d.node_count(), 0u);
  EXPECT_EQ(d.arc_count(), 0u);
  EXPECT_TRUE(d.is_acyclic());
}

TEST(OperatorDAG, RecordsProducerAndConsumers) {
  const OperatorDAG d = linear_chain();
  EXPECT_EQ(d.node_count(), 3u);
  EXPECT_EQ(d.arc_count(), 3u);
  // arc v0 is produced by the Source and consumed by the Accumulate
  EXPECT_EQ(d.arc(ArcId{0}).producer, NodeId{0});
  ASSERT_EQ(d.consumers_of(ArcId{0}).size(), 1u);
  EXPECT_EQ(d.consumers_of(ArcId{0})[0], NodeId{1});
}

TEST(OperatorDAG, TopologicalOrderRespectsDependencies) {
  const OperatorDAG d = linear_chain();
  const auto order = d.topological_order();
  ASSERT_EQ(order.size(), 3u);
  const auto pos = [&](NodeId n) {
    return std::find(order.begin(), order.end(), n) - order.begin();
  };
  EXPECT_LT(pos(NodeId{0}), pos(NodeId{1}));
  EXPECT_LT(pos(NodeId{1}), pos(NodeId{2}));
}

// Golden dumps depend on this: independent nodes must always come out in the
// same order.
TEST(OperatorDAG, TopologicalOrderIsDeterministic) {
  OperatorDAG d;
  const ValueType t{.region = RegionId{0}};
  const ArcId a = d.add_arc(t, "a");
  const ArcId b = d.add_arc(t, "b");
  const ArcId c = d.add_arc(t, "c");
  // Three mutually independent sources.
  d.add_node(OpKind::Source, {}, a, OpAttrs{}, ChannelId{}, "s0");
  d.add_node(OpKind::Source, {}, b, OpAttrs{}, ChannelId{}, "s1");
  d.add_node(OpKind::Source, {}, c, OpAttrs{}, ChannelId{}, "s2");
  const auto first = d.topological_order();
  for (int i = 0; i < 20; ++i) EXPECT_EQ(d.topological_order(), first);
  EXPECT_EQ(first[0], NodeId{0});
  EXPECT_EQ(first[1], NodeId{1});
  EXPECT_EQ(first[2], NodeId{2});
}

TEST(OperatorDAG, DetectsCycle) {
  OperatorDAG d;
  const ValueType t{.region = RegionId{0}};
  const ArcId a = d.add_arc(t, "a");
  const ArcId b = d.add_arc(t, "b");
  d.add_node(OpKind::Relax, {b}, a, OpAttrs{}, ChannelId{}, "n0");
  d.add_node(OpKind::Relax, {a}, b, OpAttrs{}, ChannelId{}, "n1");
  EXPECT_FALSE(d.is_acyclic());
  EXPECT_LT(d.topological_order().size(), d.node_count());
}

TEST(OperatorDAG, StoresIterationRegions) {
  OperatorDAG d = linear_chain();
  const auto idx = d.add_region(OperatorDAG::IterationRegion{
      .name = "outer", .body = {NodeId{1}}, .residuals = {ArcId{1}},
      .carried = {ArcId{0}}, .max_iterations = 30});
  EXPECT_EQ(d.region_count(), 1u);
  EXPECT_EQ(d.region_at(idx).max_iterations, 30);
  EXPECT_EQ(d.region_at(idx).carried.size(), 1u);
}

// The SSA invariant ("every arc has exactly one producer") must be enforced
// in every build configuration, not just when assert() is compiled in.
// A second add_node() targeting an already-produced arc must throw, in
// -DNDEBUG builds too -- a build flag must not be able to turn off the
// core invariant of an SSA IR.
TEST(OperatorDAG, SecondProducerOnSameArcThrows) {
  OperatorDAG d;
  const ValueType t{.region = RegionId{0}};
  const ArcId a0 = d.add_arc(t, "v0");
  d.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "first");
  EXPECT_THROW(
      d.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "second"),
      std::logic_error);
  // The first producer must be left untouched by the rejected second call.
  EXPECT_EQ(d.arc(a0).producer, NodeId{0});
}

// An out-of-range ArcId used as an *input* must be rejected via bounds
// checking that survives -DNDEBUG (assert() alone compiles out under
// -DNDEBUG and this indexes straight into consumers_, a heap-buffer
// overflow -- reproduced under ASan with -DNDEBUG before this fix).
TEST(OperatorDAG, OutOfRangeInputArcIdThrows) {
  OperatorDAG d;
  const ArcId bogus{999};
  EXPECT_THROW(
      d.add_node(OpKind::Sink, {bogus}, ArcId{}, OpAttrs{}, ChannelId{}, "bad"),
      std::out_of_range);
}

// Same as above but for the *output* ArcId -- indexes into arcs_ directly.
TEST(OperatorDAG, OutOfRangeOutputArcIdThrows) {
  OperatorDAG d;
  const ArcId bogus{999};
  EXPECT_THROW(
      d.add_node(OpKind::Source, {}, bogus, OpAttrs{}, ChannelId{}, "bad"),
      std::out_of_range);
}
