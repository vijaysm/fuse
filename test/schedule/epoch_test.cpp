// Rank-agreement points, and the two things the compiler proves about them.
//
// A data move is where ranks that were doing independent work have to meet. If
// one of them failed on the way there, the others are already committed and
// will wait for a message that never comes. The plan therefore carries an
// explicit point after each move where every participating rank learns the same
// outcome. These tests pin that the point is emitted, that it covers exactly
// the ranks the move covered, and that a plan violating either is refused
// rather than produced and run.
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/schedule/plan.hpp>

namespace {

using namespace fuse;

// Two components on DISJOINT ranks, so the channel needs a real data move.
CouplingGraph disjoint_pair() {
  CouplingGraph g;
  const auto m = g.add_mesh({.name = "m"});
  const auto r = g.add_region({.name = "r", .mesh = m});
  const auto a = g.add_component({.name = "a", .ranks = {0, 2}, .interval = Rational{1}});
  const auto b = g.add_component({.name = "b", .ranks = {2, 2}, .interval = Rational{1}});
  const auto s = g.add_field(a, {.name = "q", .region = r, .intent = Intent::Provides});
  const auto t = g.add_field(b, {.name = "q", .region = r, .intent = Intent::Requires});
  (void)g.add_channel("c", s, t);
  return g;
}

// Two components sharing ranks: no move, so nothing to agree about.
CouplingGraph colocated_pair() {
  CouplingGraph g;
  const auto m = g.add_mesh({.name = "m"});
  const auto r = g.add_region({.name = "r", .mesh = m});
  const auto a = g.add_component({.name = "a", .ranks = {0, 4}, .interval = Rational{1}});
  const auto b = g.add_component({.name = "b", .ranks = {0, 4}, .interval = Rational{1}});
  const auto s = g.add_field(a, {.name = "q", .region = r, .intent = Intent::Provides});
  const auto t = g.add_field(b, {.name = "q", .region = r, .intent = Intent::Requires});
  (void)g.add_channel("c", s, t);
  return g;
}

ScheduleResult build(const CouplingGraph& g) {
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const ClockSet cs{g};
  // A stub size: the mesh backend supplies the real one (S5).
  return schedule(g, lowered.dag, cs, [](RegionId) -> std::size_t { return 64; });
}

std::size_t count(const ExecutionPlan& p, OpKind kind) {
  std::size_t n = 0;
  for (const auto& ins : p.instructions()) n += (ins.kind == kind);
  return n;
}

TEST(Epoch, EveryDataMoveIsFollowedByAnAgreementPoint) {
  const auto r = build(disjoint_pair());
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const std::size_t moves = count(r.plan, OpKind::Transport);
  ASSERT_GT(moves, 0u) << "the fixture is supposed to need a data move";
  EXPECT_EQ(count(r.plan, OpKind::Epoch), moves);
}

// Co-located components exchange data without a move, so adding an agreement
// point there would be synchronization the physics never asked for -- the cost
// that makes people distrust a coupler at scale.
TEST(Epoch, NoMoveMeansNoAgreementPoint) {
  const auto r = build(colocated_pair());
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_EQ(count(r.plan, OpKind::Transport), 0u);
  EXPECT_EQ(count(r.plan, OpKind::Epoch), 0u);
}

// The point covers exactly the ranks the move covered. A point covering fewer
// ranks leaves a partner waiting; covering more makes ranks wait for partners
// who are not coming. Both are hangs.
TEST(Epoch, TheAgreementPointCoversTheSameRanksAsItsMove) {
  const auto r = build(disjoint_pair());
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  std::size_t checked = 0;
  const auto& ins = r.plan.instructions();
  for (std::size_t i = 0; i < ins.size(); ++i) {
    if (ins[i].kind != OpKind::Transport) continue;
    for (std::size_t k = i + 1; k < ins.size(); ++k) {
      if (ins[k].kind != OpKind::Epoch || ins[k].n_in == 0 || ins[k].in[0] != ins[i].out) continue;
      EXPECT_EQ(ins[k].attrs.src_rank_set, ins[i].attrs.src_rank_set);
      EXPECT_EQ(ins[k].attrs.dst_rank_set, ins[i].attrs.dst_rank_set);
      EXPECT_EQ(ins[k].time, ins[i].time) << "the agreement belongs to the move's step";
      ++checked;
      break;
    }
  }
  EXPECT_GT(checked, 0u);
}

// Reserved multiscale operators are refused by the scheduler rather than
// scheduled as no-ops: a plan that runs a micro-macro coupling as though the
// closure had been applied is worse than one that will not build.
TEST(Epoch, ReservedMultiscaleOperatorsAreRefused) {
  for (const OpKind kind : {OpKind::Map, OpKind::Reduce, OpKind::Closure}) {
    CouplingGraph g = colocated_pair();
    auto lowered = lower(g);
    ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

    // Splice one in, as a future multiscale lowering rule would.
    const ArcId in = lowered.dag.node(NodeId{0}).output;
    const ArcId out = lowered.dag.add_arc(lowered.dag.arc(in).type, "reserved");
    lowered.dag.add_node(kind, {in}, out, OpAttrs{}, ChannelId{}, "reserved");

    const ClockSet cs{g};
    const auto r = schedule(g, lowered.dag, cs, [](RegionId) -> std::size_t { return 64; });
    EXPECT_FALSE(r.ok()) << to_cstring(kind) << " scheduled as if it worked";
    bool named = false;
    for (const auto& item : r.diags.items()) named = named || item.code == "E-SCHED-MULTISCALE";
    EXPECT_TRUE(named) << r.diags.to_string();
  }
}

}  // namespace
