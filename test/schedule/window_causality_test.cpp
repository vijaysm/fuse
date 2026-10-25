// The ordering proof
// for the subcycling node (OpKind::WindowAdvance, docs/user/glossary.md).
//
// The subcycling node has EMPTY Node::inputs -- like Source -- because a
// real producer/consumer edge from the field transfer that delivers its
// import waveform closes a genuine cycle in the untimed dataflow graph
// (measured directly: schedule() refuses
// with only 2 of 12 nodes topologically orderable). Its import edge is
// instead named by OpAttrs::window_import, metadata only, and causality
// against it is established by TIME (TickPhase) rather than by an edge.
// This file proves the scheduler can actually CHECK that causality holds,
// against a hand-built instruction stream -- deliberately not through
// lower()/expand()/schedule(), so the mechanical proof stays independent
// of whatever a real compiled graph happens to look like (that
// integration is test/schedule/window_region_golden_test.cpp's own job).

#include <gtest/gtest.h>
#include <fuse/ir/dag.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/schedule/plan.hpp>

using fuse::ArcId;
using fuse::ChannelId;
using fuse::ComponentDecl;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::FieldSpec;
using fuse::Instruction;
using fuse::Intent;
using fuse::MeshDecl;
using fuse::Node;
using fuse::NodeId;
using fuse::OpAttrs;
using fuse::OpKind;
using fuse::OperatorDAG;
using fuse::Rational;
using fuse::RankSet;
using fuse::Region;
using fuse::TickPhase;
using fuse::ValueType;
namespace u = fuse::units;

namespace {

// One field transfer's own final node ("writer", any non-WindowAdvance
// kind -- Diagnose here, chosen only because it is a plausible chain tail)
// delivering the arc a subcycling node reads as its import waveform. `g`
// is an empty graph and every node's origin an invalid ChannelId, so
// window_import_is_lagged() always resolves to false (today's strict
// rule) here -- window_import_is_lagged()'s own laggedness test, and
// the fixtures below that DO want it true, are exercised separately.
struct Fixture {
  CouplingGraph g;
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

// A fixture whose writer's origin names a REAL channel declared
// `.lag(1)`, so window_import_is_lagged() resolves to true for its
// import arc -- exercising the RELAXED (position-independent) rule
// rather than the fixture above's always-strict one.
struct LaggedFixture {
  CouplingGraph g;
  OperatorDAG   dag;
  NodeId        writer;
  NodeId        subcycler;
  ArcId         import_arc;
  ArcId         export_arc;
};

LaggedFixture build_lagged_fixture() {
  LaggedFixture f;
  CouplingGraph& g = f.g;
  const auto m = g.add_mesh(MeshDecl{.name = "m"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = RankSet{1, 1}, .interval = Rational{1}});
  const auto src = g.add_field(a, FieldSpec{
      .name = "a_out", .region = r, .units = u::kKelvin,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  const auto dst = g.add_field(b, FieldSpec{
      .name = "b_in", .region = r, .units = u::kKelvin,
      .conservation = Conservation::Extensive, .intent = Intent::Requires});
  const ChannelId ch = g.add_channel("a2b", src, dst);
  g.channel(ch).lag(1).allow_time_level_mismatch("test fixture");

  f.import_arc = f.dag.add_arc(ValueType{}, "import");
  f.writer = f.dag.add_node(OpKind::Diagnose, {}, f.import_arc, OpAttrs{},
                            ch, "writer");
  f.export_arc = f.dag.add_arc(ValueType{}, "export");
  OpAttrs wa_attrs;
  wa_attrs.window_import = f.import_arc;
  f.subcycler = f.dag.add_node(OpKind::WindowAdvance, {}, f.export_arc,
                               wa_attrs, ch, "subcycler");
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
  const auto r = fuse::detail::check_window_causality(f.g, f.dag, stream);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// THE MUTATION: swap the phase (in
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
  const auto r = fuse::detail::check_window_causality(f.g, f.dag, stream);
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
  const auto r = fuse::detail::check_window_causality(f.g, f.dag, stream);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-CAUSALITY");
}

TEST(WindowCausality, ANonWindowStreamIsUnaffected) {
  // Every existing graph's own instructions have no WindowAdvance node at
  // all; the check must be a pure no-op for them; the arc `written`
  // bookkeeping (last_written_step) never even gets consulted through the
  // WindowAdvance branch.
  CouplingGraph g;
  OperatorDAG dag;
  const ArcId a = dag.add_arc(ValueType{}, "a");
  const NodeId n = dag.add_node(OpKind::Diagnose, {}, a, OpAttrs{},
                                fuse::ChannelId{}, "n");
  const std::vector<Instruction> stream = {
      make(n, OpKind::Diagnose, a, 0, Rational{0}, TickPhase::BeforeTicks),
  };
  const auto r = fuse::detail::check_window_causality(g, dag, stream);
  EXPECT_TRUE(r.ok());
}

// ---- lag-aware relaxation (waveforms travel through the
// ordinary field-transfer rules, and a two-way window needs exactly one
// direction loop-carried, as FSI's own .lag(1) is -- see check_window_
// causality()'s own doc comment, fuse/schedule/plan.hpp).

// A lagged import's writer runs AFTER the read in this SAME static pass
// (the shape one sweep's replay of a loop-carried transfer produces: the
// Runner re-executes the same instructions every sweep, so a writer that
// looks "too late" in this ONE pass will in fact have already run, on the
// PREVIOUS sweep, before this read needs it) -- accepted.
TEST(WindowCausality, LaggedImportWrittenLaterInTheSamePassIsAccepted) {
  const LaggedFixture f = build_lagged_fixture();
  const std::vector<Instruction> stream = {
      make(f.subcycler, OpKind::WindowAdvance, f.export_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::BeforeTicks),
      make(f.writer, OpKind::Diagnose, f.import_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::AfterTicks),
  };
  const auto r = fuse::detail::check_window_causality(f.g, f.dag, stream);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// The UNLAGGED rule stays strict even with the lagged relaxation's
// code path live: an unlagged import's writer running later for the SAME
// window instance is still refused (mirrors SwappedOrderForTheSameWindow
// IsKilled above, using build_fixture()'s own always-unlagged writer).
TEST(WindowCausality, UnlaggedImportWrittenLaterIsStillRejected) {
  const Fixture f = build_fixture();
  const std::vector<Instruction> stream = {
      make(f.subcycler, OpKind::WindowAdvance, f.export_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::BeforeTicks),
      make(f.writer, OpKind::Diagnose, f.import_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::AfterTicks),
  };
  const auto r = fuse::detail::check_window_causality(f.g, f.dag, stream);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-CAUSALITY");
}

// The relaxation is on POSITION only, never on `dst_step`: a lagged
// import's writer for a DIFFERENT window instance does not satisfy a
// later one, or reading the previous WINDOW's value (rather than the
// previous SWEEP's, which the group's own re-execution provides) would
// silently become an explicit, loosely coupled scheme -- exactly the
// accuracy loss this rule's own doc comment says it must not permit.
TEST(WindowCausality, LaggedImportWrittenForADifferentWindowIsRejected) {
  const LaggedFixture f = build_lagged_fixture();
  const std::vector<Instruction> stream = {
      make(f.writer, OpKind::Diagnose, f.import_arc, /*dst_step=*/0,
           Rational{0}, TickPhase::BeforeTicks),
      make(f.subcycler, OpKind::WindowAdvance, f.export_arc, /*dst_step=*/1,
           Rational{1}, TickPhase::AfterTicks),
  };
  const auto r = fuse::detail::check_window_causality(f.g, f.dag, stream);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-WINDOW-CAUSALITY");
}
