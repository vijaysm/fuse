// Unit P of the multirate-window-on-the-Runner track: the ordering proof
// for the subcycling node (OpKind::WindowAdvance, docs/user/glossary.md).
//
// The subcycling node has EMPTY Node::inputs -- like Source -- because a
// real producer/consumer edge from the field transfer that delivers its
// import waveform closes a genuine cycle in the untimed dataflow graph
// (measured directly in an earlier pass of this track: schedule() refused
// with only 2 of 12 nodes topologically orderable). Its import edge is
// instead named by OpAttrs::window_import, metadata only, and causality
// against it is established by TIME (TickPhase) rather than by an edge.
// This file proves the scheduler can actually CHECK that causality holds,
// against a hand-built instruction stream -- deliberately not through
// lower()/expand(), since no lowering rule emits this node yet (that is
// this track's Unit B, not this one).

#include <gtest/gtest.h>
#include <fuse/ir/dag.hpp>
#include <fuse/schedule/plan.hpp>

using fuse::ArcId;
using fuse::Instruction;
using fuse::Node;
using fuse::NodeId;
using fuse::OpAttrs;
using fuse::OpKind;
using fuse::OperatorDAG;
using fuse::Rational;
using fuse::TickPhase;
using fuse::ValueType;

namespace {

// One field transfer's own final node ("writer", any non-WindowAdvance
// kind -- Diagnose here, chosen only because it is a plausible chain tail)
// delivering the arc a subcycling node reads as its import waveform.
struct Fixture {
  OperatorDAG dag;
  NodeId      writer;
  NodeId      subcycler;
  ArcId       import_arc;
  ArcId       export_arc;
};

Fixture build_fixture() {
  Fixture f;
  f.import_arc = f.dag.add_arc(ValueType{}, "import");
  f.writer = f.dag.add_node(OpKind::Diagnose, {}, f.import_arc, OpAttrs{},
                            fuse::ChannelId{}, "writer");
  f.export_arc = f.dag.add_arc(ValueType{}, "export");
  OpAttrs wa_attrs;
  wa_attrs.window_import = f.import_arc;
  f.subcycler = f.dag.add_node(OpKind::WindowAdvance, {}, f.export_arc,
                               wa_attrs, fuse::ChannelId{}, "subcycler");
  return f;
}

Instruction make(NodeId origin, OpKind kind, ArcId out, std::int64_t dst_step,
                 Rational time, TickPhase phase = TickPhase::BeforeTicks) {
  Instruction ins;
  ins.kind = kind;
  ins.out = out;
  ins.origin_node = origin;
  ins.dst_step = dst_step;
  ins.time = time;
  ins.phase = phase;
  return ins;
}

}  // namespace

TEST(WindowCausality, WriterBeforeSubcyclerForTheSameWindowIsAccepted) {
  const Fixture f = build_fixture();
  const std::vector<Instruction> stream = {
      make(f.writer, OpKind::Diagnose, f.import_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::BeforeTicks),
      make(f.subcycler, OpKind::WindowAdvance, f.export_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::AfterTicks),
  };
  const auto r = fuse::detail::check_window_causality(f.dag, stream);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// THE MUTATION Unit P's own acceptance criterion names: swap the phase (in
// effect, swap which of the two instructions the stream places first for
// the same window instance) so the subcycling node reads its import
// waveform before that window's own writer has run. Killed.
TEST(WindowCausality, SwappedOrderForTheSameWindowIsKilled) {
  const Fixture f = build_fixture();
  const std::vector<Instruction> stream = {
      make(f.subcycler, OpKind::WindowAdvance, f.export_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::BeforeTicks),
      make(f.writer, OpKind::Diagnose, f.import_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::AfterTicks),
  };
  const auto r = fuse::detail::check_window_causality(f.dag, stream);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-CAUSALITY");
}

// A second, distinct way the invariant can fail that a coarser "was it
// EVER written before" check (check_dependence_order()'s own style) would
// miss: the writer ran first in the stream, but for a DIFFERENT window
// instance (a stale value from window 0, while the subcycling node is
// asking for window 1). Also killed -- this is exactly why
// check_window_causality() tracks the writer's OWN dst_step rather than a
// plain seen/not-seen flag.
TEST(WindowCausality, WriterForAnEarlierWindowInstanceDoesNotSatisfyALaterOne) {
  const Fixture f = build_fixture();
  const std::vector<Instruction> stream = {
      make(f.writer, OpKind::Diagnose, f.import_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::BeforeTicks),
      make(f.subcycler, OpKind::WindowAdvance, f.export_arc, /*dst_step=*/1,
           Rational{1}, TickPhase::AfterTicks),
  };
  const auto r = fuse::detail::check_window_causality(f.dag, stream);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-CAUSALITY");
}

TEST(WindowCausality, ANonWindowStreamIsUnaffected) {
  // Every existing graph's own instructions have no WindowAdvance node at
  // all; the check must be a pure no-op for them; the arc `written`
  // bookkeeping (last_written_step) never even gets consulted through the
  // WindowAdvance branch.
  OperatorDAG dag;
  const ArcId a = dag.add_arc(ValueType{}, "a");
  const NodeId n = dag.add_node(OpKind::Diagnose, {}, a, OpAttrs{},
                                fuse::ChannelId{}, "n");
  const std::vector<Instruction> stream = {
      make(n, OpKind::Diagnose, a, 0, Rational{0}, TickPhase::BeforeTicks),
  };
  const auto r = fuse::detail::check_window_causality(dag, stream);
  EXPECT_TRUE(r.ok());
}
