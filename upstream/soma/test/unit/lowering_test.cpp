#include <gtest/gtest.h>
#include <soma/ir/dump.hpp>
#include <soma/ir/lowering.hpp>

#include <algorithm>

using soma::ChannelId;
using soma::LoweringContext;
using soma::LoweringRegistry;
using soma::OpKind;

namespace {

int count_kind(const soma::OperatorDAG& d, OpKind k) {
  int n = 0;
  for (std::uint32_t i = 0; i < d.node_count(); ++i) {
    if (d.node(soma::NodeId{i}).kind == k) ++n;
  }
  return n;
}

}  // namespace

TEST(LoweringRegistry, TriesRulesInPriorityOrder) {
  LoweringRegistry reg;
  std::string log;
  reg.add("low", 1, [&](LoweringContext&, ChannelId) {
    log += "low ";
    return false;   // decline
  });
  reg.add("high", 10, [&](LoweringContext&, ChannelId) {
    log += "high ";
    return true;    // handle
  });

  soma::CouplingGraph g;
  soma::OperatorDAG d;
  soma::Diagnostics diags;
  LoweringContext ctx{g, d, diags, {}, {}};
  EXPECT_TRUE(reg.apply_first_match(ctx, ChannelId{0}));
  EXPECT_EQ(log, "high ");   // high ran first and claimed it
}

TEST(LoweringRegistry, FallsThroughWhenRuleDeclines) {
  LoweringRegistry reg;
  std::string log;
  reg.add("first", 10, [&](LoweringContext&, ChannelId) {
    log += "first ";
    return false;
  });
  reg.add("second", 1, [&](LoweringContext&, ChannelId) {
    log += "second ";
    return true;
  });
  soma::CouplingGraph g;
  soma::OperatorDAG d;
  soma::Diagnostics diags;
  LoweringContext ctx{g, d, diags, {}, {}};
  EXPECT_TRUE(reg.apply_first_match(ctx, ChannelId{0}));
  EXPECT_EQ(log, "first second ");
}

TEST(LoweringRegistry, ReportsNoMatch) {
  LoweringRegistry reg;
  soma::CouplingGraph g;
  soma::OperatorDAG d;
  soma::Diagnostics diags;
  LoweringContext ctx{g, d, diags, {}, {}};
  EXPECT_FALSE(reg.apply_first_match(ctx, ChannelId{0}));
}

TEST(LoweringRegistry, BuiltinRegistryIsPopulated) {
  EXPECT_GT(LoweringRegistry::builtin().size(), 0u);
}

// R8 regression: apply_first_match() used to range-iterate rules_ by
// reference while invoking each rule's callback. A callback that calls add()
// on the SAME registry mid-iteration reallocates rules_ (push_back +
// stable_sort), which is a use-after-free if the loop is still walking the
// old storage -- ASan-confirmed. This test registers a rule that does exactly
// that, and returns false so the loop must continue past the reentrant add()
// and (pre-fix) into the reallocated storage.
TEST(LoweringRegistry, ReentrantAddDuringApplyFirstMatchIsSafe) {
  LoweringRegistry reg;
  int first_calls = 0;
  int second_calls = 0;

  reg.add("first", 10, [&](LoweringContext&, ChannelId) {
    ++first_calls;
    // Reentrant registration: reallocates rules_ mid-call to
    // apply_first_match(), below.
    reg.add("second", 1, [&](LoweringContext&, ChannelId) {
      ++second_calls;
      return false;
    });
    return false;  // decline, so the loop continues into the (pre-fix) stale
                   // storage instead of returning immediately.
  });

  soma::CouplingGraph g;
  soma::OperatorDAG d;
  soma::Diagnostics diags;
  LoweringContext ctx{g, d, diags, {}, {}};

  // First call: only "first" existed when the snapshot was taken, so
  // "second" -- added during this very call -- must not fire yet.
  EXPECT_FALSE(reg.apply_first_match(ctx, ChannelId{0}));
  EXPECT_EQ(first_calls, 1);
  EXPECT_EQ(second_calls, 0);
  EXPECT_EQ(reg.size(), 2u);

  // Second call: "second" is now part of the stable snapshot and does fire,
  // proving the reentrant add() took effect for subsequent dispatch.
  EXPECT_FALSE(reg.apply_first_match(ctx, ChannelId{0}));
  EXPECT_EQ(first_calls, 2);
  EXPECT_EQ(second_calls, 1);
}

// The driver must emit boundary nodes even with no channels at all.
TEST(Lowering, EmitsSourceForEveryProvidedField) {
  soma::CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto r = g.add_region(soma::Region{.name = "r", .mesh = m});
  const auto c = g.add_component(soma::ComponentDecl{.name = "a"});
  g.add_field(c, soma::FieldSpec{.name = "f", .region = r,
                                 .intent = soma::Intent::Provides});
  const auto result = lower(g);
  EXPECT_TRUE(result.ok()) << result.diags.to_string();
  EXPECT_EQ(count_kind(result.dag, OpKind::Source), 1);
}

// R15 regression: a Bidirectional field no channel ever targets keeps
// field_arc[f] bound to the Source arc step 1 seeded it with, so step 3's
// Sink for that field deliberately consumes the field's own Source output --
// a degenerate self-copy (see LoweringContext's `field_arc` doc comment in
// include/soma/ir/lowering.hpp, and the cross-reference at the step-3 site in
// src/ir/lowering.cpp). This is documented as deliberate, not a bug.
//
// acyclicity_test.cpp's UnwrittenBidirectionalFieldIsNotASpuriousCycle builds
// the same scenario but only asserts prove_acyclic()'s result; it does not
// look at the DAG at all. This test pins the actual DAG shape: exactly one
// Source, exactly one Sink for the unwritten field, and that the Sink's sole
// input is precisely that Source's output arc -- not zero Sinks (silently
// dropped) and not two (duplicated).
TEST(Lowering, UnwrittenBidirectionalFieldSelfCopiesThroughOneSink) {
  soma::CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto r = g.add_region(soma::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(soma::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}});
  g.add_field(a, soma::FieldSpec{.name = "state", .region = r,
                                 .intent = soma::Intent::Bidirectional});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = r, .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = r, .intent = soma::Intent::Requires});
  g.add_channel("c", s, t);

  const auto result = lower(g);
  ASSERT_TRUE(result.ok()) << result.diags.to_string();

  soma::NodeId source;
  int source_count = 0;
  soma::NodeId sink;
  int sink_count = 0;
  for (std::uint32_t i = 0; i < result.dag.node_count(); ++i) {
    const soma::NodeId id{i};
    const auto& n = result.dag.node(id);
    if (n.label != "a.state") continue;
    if (n.kind == OpKind::Source) { source = id; ++source_count; }
    if (n.kind == OpKind::Sink) { sink = id; ++sink_count; }
  }

  ASSERT_EQ(source_count, 1) << "expected exactly one Source for a.state";
  ASSERT_EQ(sink_count, 1)
      << "expected exactly one Sink for a.state (the self-copy), got "
      << sink_count;

  const auto& sink_node = result.dag.node(sink);
  ASSERT_EQ(sink_node.inputs.size(), 1u);
  EXPECT_EQ(sink_node.inputs.front().value(),
           result.dag.node(source).output.value())
      << "an unwritten Bidirectional field's Sink must consume its own "
         "Source arc (the documented degenerate self-copy), not some other "
         "arc";
}
