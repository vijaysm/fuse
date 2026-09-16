#include <gtest/gtest.h>
#include <fuse/model/validate.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/schedule/acyclicity.hpp>
#include <fuse/schedule/expand.hpp>

#include <cstdint>
#include <string>
#include <vector>

using fuse::ComponentDecl;
using fuse::CouplingGraph;
using fuse::Rational;
namespace u = fuse::units;

namespace {

// A <-> B. `lag_on_return` puts a lag on the B->A channel, which makes the
// cycle a legal explicit scheme rather than a deadlock.
CouplingGraph two_way(bool lag_on_return) {
  CouplingGraph g;
  const auto ma = g.add_mesh(fuse::MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(fuse::MeshDecl{.name = "mb"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(b, fuse::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(a, fuse::FieldSpec{
      .name = "a_in", .region = ra, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});

  g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  if (lag_on_return) {
    g.channel(back).lag(1).allow_time_level_mismatch("explicit two-way scheme");
  }
  return g;
}

fuse::CycleReport check(const CouplingGraph& g) {
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  EXPECT_TRUE(td.ok()) << td.error().message;
  return prove_acyclic(g, lowered.dag, td.value());
}

}  // namespace

TEST(Acyclicity, LinearChainIsAcyclic) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = r, .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = r, .intent = fuse::Intent::Requires});
  g.add_channel("c", s, t);

  const auto report = check(g);
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// The hard gate: an unlagged two-way coupling is a genuine deadlock.
TEST(Acyclicity, UnlaggedTwoWayCouplingIsRejected) {
  const auto report = check(two_way(/*lag_on_return=*/false));
  EXPECT_FALSE(report.acyclic);
  EXPECT_FALSE(report.cycle.empty());
  EXPECT_FALSE(report.explanation.empty());
}

// The subtlety: the same topology with a lag is a legal explicit scheme.
TEST(Acyclicity, LaggedTwoWayCouplingIsAccepted) {
  const auto report = check(two_way(/*lag_on_return=*/true));
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

TEST(Acyclicity, ExplanationNamesTheChannelsInvolved) {
  const auto report = check(two_way(false));
  ASSERT_FALSE(report.acyclic);
  const bool names_a_channel =
      report.explanation.find("a2b") != std::string::npos ||
      report.explanation.find("b2a") != std::string::npos;
  EXPECT_TRUE(names_a_channel) << report.explanation;
}

TEST(Acyclicity, EmptyGraphIsAcyclic) {
  const CouplingGraph g;
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok());
  EXPECT_TRUE(prove_acyclic(g, lowered.dag, td.value()).acyclic);
}

// ---------------------------------------------------------------------------
// The message has to be usable, not merely non-empty.
// ---------------------------------------------------------------------------

TEST(Acyclicity, ExplanationNamesBothChannelsAndTheComponentStep) {
  const auto report = check(two_way(false));
  ASSERT_FALSE(report.acyclic);
  EXPECT_NE(report.explanation.find("'a2b'"), std::string::npos)
      << report.explanation;
  EXPECT_NE(report.explanation.find("'b2a'"), std::string::npos)
      << report.explanation;
  // The cycle only closes through the components; say so.
  EXPECT_NE(report.explanation.find("component 'a' step 0"), std::string::npos)
      << report.explanation;
  EXPECT_NE(report.explanation.find("component 'b' step 0"), std::string::npos)
      << report.explanation;
  // And it names the node instances, with their times.
  EXPECT_NE(report.explanation.find("a.a_out"), std::string::npos)
      << report.explanation;
  EXPECT_NE(report.explanation.find("b.b_in"), std::string::npos)
      << report.explanation;
}

TEST(Acyclicity, CycleWitnessIsSelfConsistent) {
  const auto report = check(two_way(false));
  ASSERT_FALSE(report.acyclic);
  EXPECT_EQ(report.cycle.size(), report.cycle_instances.size());
  EXPECT_GE(report.cycle.size(), 2u);
}

TEST(Acyclicity, AcceptedGraphCarriesNoCycleAndNoMessage) {
  const auto report = check(two_way(true));
  ASSERT_TRUE(report.acyclic) << report.explanation;
  EXPECT_TRUE(report.cycle.empty());
  EXPECT_TRUE(report.cycle_instances.empty());
  EXPECT_TRUE(report.explanation.empty());
}

TEST(Acyclicity, LagOnEitherDirectionBreaksTheCycle) {
  // Symmetry: which of the two channels carries the lag must not matter.
  CouplingGraph g = two_way(false);
  g.channel(fuse::ChannelId{0}).lag(1).allow_time_level_mismatch("explicit");
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  const auto report = prove_acyclic(g, lowered.dag, td.value());
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// ---------------------------------------------------------------------------
// Multirate and accumulation: the emission times expand() uses are
// deliberately non-uniform (Sample at a source tick's start, the reduction at
// a target step's end, everything else at a slab's end). The pairing of
// producer instances to consumer instances has to survive that.
// ---------------------------------------------------------------------------

namespace {

// a subcycles 3:1 against b. `accumulate` turns the forward channel into
// Sample-per-source-tick + reduce-per-target-tick.
CouplingGraph multirate(bool lag_on_return, bool accumulate) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = m});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1},
      .interval = Rational{1, 3}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});
  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(b, fuse::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(a, fuse::FieldSpec{
      .name = "a_in", .region = ra, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto fwd = g.add_channel("a2b", a_out, b_in);
  if (accumulate) g.channel(fwd).accumulate(fuse::AccumOp::Mean);
  const auto back = g.add_channel("b2a", b_out, a_in);
  if (lag_on_return) {
    g.channel(back).lag(1).allow_time_level_mismatch("explicit two-way scheme");
  }
  return g;
}

}  // namespace

TEST(Acyclicity, SubcycledTwoWayCouplingIsRejected) {
  const auto report = check(multirate(false, /*accumulate=*/false));
  EXPECT_FALSE(report.acyclic);
  EXPECT_FALSE(report.cycle.empty());
}

TEST(Acyclicity, SubcycledTwoWayCouplingWithLagIsAccepted) {
  const auto report = check(multirate(true, /*accumulate=*/false));
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

TEST(Acyclicity, AccumulatingSubcycledTwoWayCouplingIsRejected) {
  const auto report = check(multirate(false, /*accumulate=*/true));
  ASSERT_FALSE(report.acyclic);
  // The reduction is on the cycle, and so are the Sample instances that fed
  // it: the two run on different clocks and are paired through dst_step.
  EXPECT_NE(report.explanation.find("a2b/accumulate"), std::string::npos)
      << report.explanation;
  EXPECT_NE(report.explanation.find("a2b/sample"), std::string::npos)
      << report.explanation;
}

TEST(Acyclicity, AccumulatingSubcycledTwoWayCouplingWithLagIsAccepted) {
  const auto report = check(multirate(true, /*accumulate=*/true));
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// ---------------------------------------------------------------------------
// Rings longer than two, and the degenerate one-component ring.
// ---------------------------------------------------------------------------

namespace {

// n components in a ring: out_i -> in_{i+1 mod n}. One shared region keeps the
// lowered chain short (Transport only). `lag_on` is the index of the channel
// that carries the lag, or -1 for none.
CouplingGraph ring(int n, int lag_on) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  std::vector<fuse::FieldId> outs;
  std::vector<fuse::FieldId> ins;
  for (int i = 0; i < n; ++i) {
    const auto c = g.add_component(ComponentDecl{
        .name = "c" + std::to_string(i), .ranks = fuse::RankSet{i, 1}});
    outs.push_back(g.add_field(c, fuse::FieldSpec{
        .name = "out", .region = r, .intent = fuse::Intent::Provides}));
    ins.push_back(g.add_field(c, fuse::FieldSpec{
        .name = "in", .region = r, .intent = fuse::Intent::Requires}));
  }
  for (int i = 0; i < n; ++i) {
    const auto ch =
        g.add_channel("ch" + std::to_string(i), outs[i], ins[(i + 1) % n]);
    if (i == lag_on) {
      g.channel(ch).lag(1).allow_time_level_mismatch("explicit ring");
    }
  }
  return g;
}

}  // namespace

TEST(Acyclicity, ThreeComponentRingIsRejected) {
  const auto report = check(ring(3, -1));
  ASSERT_FALSE(report.acyclic);
  EXPECT_NE(report.explanation.find("'ch0'"), std::string::npos)
      << report.explanation;
  EXPECT_NE(report.explanation.find("'ch1'"), std::string::npos)
      << report.explanation;
  EXPECT_NE(report.explanation.find("'ch2'"), std::string::npos)
      << report.explanation;
}

TEST(Acyclicity, ThreeComponentRingWithOneLagIsAccepted) {
  for (int lag_on = 0; lag_on < 3; ++lag_on) {
    const auto report = check(ring(3, lag_on));
    EXPECT_TRUE(report.acyclic)
        << "lag on ch" << lag_on << ": " << report.explanation;
  }
}

// A component that feeds its own input, with nothing in between, is an
// algebraic loop just as much as a two-component one.
TEST(Acyclicity, SelfCouplingInsideOneComponentIsRejected) {
  const auto report = check(ring(1, -1));
  ASSERT_FALSE(report.acyclic) << "a component whose output feeds its own "
                                  "input at the same step is a loop";
  EXPECT_NE(report.explanation.find("component 'c0' step 0"), std::string::npos)
      << report.explanation;
}

TEST(Acyclicity, SelfCouplingWithALagIsAccepted) {
  const auto report = check(ring(1, 0));
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// A Bidirectional field no channel writes lowers to Source -> Sink with
// nothing between. That Sink carries no coupled data; counting it as a
// component input would make every such field a spurious two-cycle against
// its own Source.
TEST(Acyclicity, UnwrittenBidirectionalFieldIsNotASpuriousCycle) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}});
  g.add_field(a, fuse::FieldSpec{.name = "state", .region = r,
                                 .intent = fuse::Intent::Bidirectional});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = r, .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = r, .intent = fuse::Intent::Requires});
  g.add_channel("c", s, t);

  const auto report = check(g);
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// ---------------------------------------------------------------------------
// Iteration regions (spec §5.3), now that Task 7.2 derives
// IterationRegion::carried. The gate learned about `carried`, not about the
// annotation: what breaks a cycle is a previous-iterate value in the IR, and
// declaring a region does not by itself create one.
//
// Both directions are pinned below. The expensive direction is the false
// negative -- an accepted graph that is a genuine algebraic loop -- so three of
// the four tests are on that side.
// ---------------------------------------------------------------------------

// The annotation alone is still not enough, and this is not a limitation to be
// worked around: with no .relax(omega) anywhere in the region, lowering emits
// no node whose value at sweep k reads sweep k-1, `carried` comes back empty,
// and the subgraph is byte-identical to the straight-line one that deadlocks.
TEST(Acyclicity, IterationRegionWithoutACarriedValueDoesNotBreakACycle) {
  CouplingGraph g = two_way(false);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "fsi",
      .channels = {fuse::ChannelId{0}, fuse::ChannelId{1}},
      .max_iterations = 30});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  ASSERT_EQ(lowered.dag.region_count(), 1u);
  EXPECT_TRUE(lowered.dag.region_at(fuse::RegionIdx{0}).carried.empty())
      << "no Relax inside the region, so nothing is loop-carried";

  const auto report = check(g);
  EXPECT_FALSE(report.acyclic);
  // ...and the message says so, rather than recommending a fix that does not
  // work.
  EXPECT_NE(report.explanation.find("carries nothing"), std::string::npos)
      << report.explanation;
}

// The flip. Same topology, same annotation, plus the one thing that makes it a
// Picard iteration rather than a deadlock: a relaxed value on the return
// channel, which Task 7.2 derives as the region's carried buffer.
TEST(Acyclicity, IterationRegionWithACarriedValueBreaksTheCycle) {
  CouplingGraph g = two_way(false);
  g.channel(fuse::ChannelId{1}).relax(0.7);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "fsi",
      .channels = {fuse::ChannelId{0}, fuse::ChannelId{1}},
      .max_iterations = 30});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  ASSERT_EQ(lowered.dag.region_count(), 1u);
  EXPECT_EQ(lowered.dag.region_at(fuse::RegionIdx{0}).carried.size(), 1u);

  const auto report = check(g);
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// The gate is not turned off, only narrowed. Relaxation *outside* any region
// carries nothing: the Relax node's origin channel is not in the region body,
// so `carried` stays empty and the deadlock is still a deadlock.
TEST(Acyclicity, RelaxationOutsideTheRegionDoesNotBreakTheCycle) {
  CouplingGraph g = two_way(false);
  g.channel(fuse::ChannelId{1}).relax(0.7);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "half", .channels = {fuse::ChannelId{0}}, .max_iterations = 30});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  EXPECT_TRUE(lowered.dag.region_at(fuse::RegionIdx{0}).carried.empty());

  const auto report = check(g);
  EXPECT_FALSE(report.acyclic) << report.explanation;
}

// A cycle that leaves the region is not resolved by iterating the region, so
// the cut is refused. a -> b -> c -> a with an iteration region over c->a
// alone: the carried value is delivered to `a`, but `a` sends into a2b, which
// is outside the region, so the runner would never iterate the a->b->c half.
// Accepting this would be the expensive kind of wrong.
TEST(Acyclicity, RelaxedRegionOverPartOfARingIsStillRejected) {
  CouplingGraph g = ring(3, -1);
  ASSERT_EQ(g.channel_count(), 3u);
  const fuse::ChannelId closing{2};  // c -> a
  g.channel(closing).relax(0.7);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "partial", .channels = {closing}, .max_iterations = 30});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  EXPECT_EQ(lowered.dag.region_at(fuse::RegionIdx{0}).carried.size(), 1u)
      << "the value is carried; it is the *cut* that must be refused";

  const auto report = check(g);
  EXPECT_FALSE(report.acyclic) << report.explanation;
}

// The degenerate case the containment guard has to get right: one component
// feeding its own input. The region covers the whole loop -- the component both
// sends into it and receives from it -- so a relaxed self-consistent-field
// iteration over it is legitimate and is accepted.
TEST(Acyclicity, RelaxedSelfCouplingInsideARegionIsAccepted) {
  CouplingGraph g = ring(1, -1);
  ASSERT_EQ(g.channel_count(), 1u);
  g.channel(fuse::ChannelId{0}).relax(0.5);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "scf", .channels = {fuse::ChannelId{0}}, .max_iterations = 50});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  ASSERT_EQ(lowered.dag.region_at(fuse::RegionIdx{0}).carried.size(), 1u);

  const auto report = check(g);
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// A second, independent algebraic loop elsewhere in the graph survives an
// iteration region that legitimately resolves the first one.
TEST(Acyclicity, SecondCycleOutsideTheRegionIsStillRejected) {
  CouplingGraph g = two_way(false);
  g.channel(fuse::ChannelId{1}).relax(0.7);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "fsi",
      .channels = {fuse::ChannelId{0}, fuse::ChannelId{1}},
      .max_iterations = 30});
  // A wholly separate unlagged two-way pair c <-> d.
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "mcd"});
  const auto r = g.add_region(fuse::Region{.name = "rcd", .mesh = m});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{2, 1}, .interval = Rational{1}});
  const auto d = g.add_component(ComponentDecl{
      .name = "d", .ranks = fuse::RankSet{3, 1}, .interval = Rational{1}});
  const auto c_out = g.add_field(c, fuse::FieldSpec{
      .name = "c_out", .region = r, .intent = fuse::Intent::Provides});
  const auto d_in = g.add_field(d, fuse::FieldSpec{
      .name = "d_in", .region = r, .intent = fuse::Intent::Requires});
  const auto d_out = g.add_field(d, fuse::FieldSpec{
      .name = "d_out", .region = r, .intent = fuse::Intent::Provides});
  const auto c_in = g.add_field(c, fuse::FieldSpec{
      .name = "c_in", .region = r, .intent = fuse::Intent::Requires});
  g.add_channel("c2d", c_out, d_in);
  g.add_channel("d2c", d_out, c_in);

  const auto report = check(g);
  EXPECT_FALSE(report.acyclic) << report.explanation;
}

// ---------------------------------------------------------------------------
// Task 3.3 handoff: its instruction order assumes ascending NodeId is a
// topological order of the untimed DAG. Nothing enforces that, so check it
// instead of inheriting it.
// ---------------------------------------------------------------------------

TEST(Acyclicity, EmissionOrderOfEveryExpandedScheduleIsChecked) {
  const std::vector<CouplingGraph> graphs{
      two_way(false),        two_way(true), multirate(false, false),
      multirate(true, true), ring(4, -1),   ring(4, 2)};
  for (const CouplingGraph& g : graphs) {
    const auto report = check(g);
    EXPECT_TRUE(report.emission_order_valid) << report.order_explanation;
  }
}

TEST(Acyclicity, EmissionOrderViolationIsReported) {
  const CouplingGraph g = two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g};
  auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;

  // Sanity: as emitted, the order is fine.
  ASSERT_TRUE(prove_acyclic(g, lowered.dag, td.value()).emission_order_valid);

  // Move the first instruction to the back. It is a Source with live
  // consumers, so every one of them now runs before its producer. (Indices
  // into TimedDag::nodes *are* the emission order; the dependence graph is
  // keyed on (node, instance) and so is unaffected by the move.)
  auto& nodes = td.value().nodes;
  ASSERT_GE(nodes.size(), 2u);
  const fuse::TimedNode first = nodes.front();
  ASSERT_EQ(lowered.dag.node(first.node).kind, fuse::OpKind::Source);
  nodes.erase(nodes.begin());
  nodes.push_back(first);

  const auto report = prove_acyclic(g, lowered.dag, td.value());
  EXPECT_FALSE(report.emission_order_valid);
  EXPECT_FALSE(report.order_explanation.empty());
  // Reordering the emission does not change the dependence graph, so the
  // verdict on cyclicity must be unchanged. The two questions are separate.
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// ---------------------------------------------------------------------------
// A schedule the gate cannot analyse is refused, never waved through.
// ---------------------------------------------------------------------------

TEST(Acyclicity, MalformedTimedDagIsRefusedRatherThanAccepted) {
  const CouplingGraph g = two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  const auto good = expand(g, lowered.dag, cs);
  ASSERT_TRUE(good.ok());
  ASSERT_TRUE(prove_acyclic(g, lowered.dag, good.value()).acyclic);

  {  // Duplicated instance index.
    fuse::TimedDag td = good.value();
    td.nodes.push_back(td.nodes.front());
    const auto report = prove_acyclic(g, lowered.dag, td);
    EXPECT_FALSE(report.acyclic);
    EXPECT_FALSE(report.explanation.empty());
  }
  {  // Node id outside the DAG.
    fuse::TimedDag td = good.value();
    td.nodes.front().node = fuse::NodeId{99999};
    const auto report = prove_acyclic(g, lowered.dag, td);
    EXPECT_FALSE(report.acyclic);
  }
  {  // Instance index out of range.
    fuse::TimedDag td = good.value();
    td.nodes.front().instance = 7;
    const auto report = prove_acyclic(g, lowered.dag, td);
    EXPECT_FALSE(report.acyclic);
  }
}

// ---------------------------------------------------------------------------
// Hand-built DAGs. These reach cases the current lowering rules cannot
// produce, but which the function's contract has to cover.
// ---------------------------------------------------------------------------

namespace {

// A one-component graph, just so rank_set 0 resolves to a real component.
CouplingGraph one_component() {
  CouplingGraph g;
  g.add_component(
      ComponentDecl{.name = "solo", .ranks = fuse::RankSet{0, 1}});
  return g;
}

fuse::ValueType typed(int rank_set, int offset) {
  fuse::ValueType t;
  t.rank_set = rank_set;
  t.time.offset = Rational{offset};
  return t;
}

// One instance per node, at t = 0, in node order.
fuse::TimedDag one_instance_each(const fuse::OperatorDAG& dag) {
  fuse::TimedDag td;
  td.period = Rational{1};
  td.nodes.reserve(dag.node_count());
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    td.nodes.push_back(fuse::TimedNode{fuse::NodeId{i}, Rational{0}, 0, 0, 0});
  }
  return td;
}

}  // namespace

// A lag is one boundary crossing, not a licence to drop every edge downstream
// of it. Lowering marks every arc after the shift with the negative offset, so
// a rule of "drop the edge when the consumed arc's offset is negative" cuts
// the whole tail of the chain. Here the loop B -> C -> D -> B lives entirely
// at level -1: under that rule all three of its edges vanish and the cycle is
// silently accepted. Under "drop only where the consumer reads a level earlier
// than the one it produces" exactly the one edge into A would be dropped (and
// there is none here), so the loop is found.
//
// Confirmed to fail without the fix: replacing is_loop_carried()'s body with
// `return dag.arc(in).type.time.offset < Rational{0};` -- the brief's rule --
// makes this test report acyclic == true.
TEST(Acyclicity, LagDoesNotDropEdgesInsideTheLaggedRegion) {
  using fuse::ArcId;
  using fuse::OpAttrs;
  using fuse::OpKind;

  const CouplingGraph g = one_component();
  fuse::OperatorDAG dag;
  // Arcs first, so nodes may be added in any order: the SSA rule is one
  // producer per arc, not producers before consumers.
  const ArcId a0 = dag.add_arc(typed(0, 0), "a0");   // A -> B, current level
  const ArcId a1 = dag.add_arc(typed(0, -1), "a1");  // B -> C, lagged level
  const ArcId a2 = dag.add_arc(typed(0, -1), "a2");  // C -> D, lagged level
  const ArcId a3 = dag.add_arc(typed(0, -1), "a3");  // D -> B, lagged level
  dag.add_node(OpKind::Source, {}, a0, OpAttrs{}, fuse::ChannelId{}, "A");
  dag.add_node(OpKind::Remap, {a0, a3}, a1, OpAttrs{}, fuse::ChannelId{}, "B");
  dag.add_node(OpKind::Remap, {a1}, a2, OpAttrs{}, fuse::ChannelId{}, "C");
  dag.add_node(OpKind::Remap, {a2}, a3, OpAttrs{}, fuse::ChannelId{}, "D");

  const auto report = prove_acyclic(g, dag, one_instance_each(dag));
  EXPECT_FALSE(report.acyclic)
      << "B -> C -> D -> B is entirely inside the lagged region, so all three "
         "edges are intra-period and the loop is real";
  EXPECT_NE(report.explanation.find("B"), std::string::npos)
      << report.explanation;
}

// `.lag(-1)` is a *lead*: the consumer wants the producer's future value.
// validate() permits it (E-LAG measures std::abs(lag)), so it reaches here.
// Lowering shifts the offset the other way, to +1, so a level-0 value feeds a
// level-+1 consumer. A rule of "cut when the consumer reads an earlier level
// than it produces" fires on that, disconnects the chain from its Source and
// reports the two-way coupling as acyclic -- a false negative.
//
// Confirmed to fail without the fix: dropping the `from.offset < Rational{0}`
// conjunct from is_loop_carried() makes this test report acyclic == true.
TEST(Acyclicity, LeadingChannelIsStillRejectedAsACycle) {
  CouplingGraph g = two_way(false);
  g.channel(fuse::ChannelId{1}).lag(-1).allow_time_level_mismatch("lead");
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  const auto report = prove_acyclic(g, lowered.dag, td.value());
  EXPECT_FALSE(report.acyclic)
      << "a lead is not a lag: it does not reach back across the period "
         "boundary, so it breaks nothing";
}

// The counterpart: a lag that really does cross the boundary is dropped, and
// dropping it is enough to break the loop.
TEST(Acyclicity, LagAtTheBoundaryIsDropped) {
  using fuse::ArcId;
  using fuse::OpAttrs;
  using fuse::OpKind;

  const CouplingGraph g = one_component();
  fuse::OperatorDAG dag;
  const ArcId a0 = dag.add_arc(typed(0, 0), "a0");   // A -> B
  const ArcId a1 = dag.add_arc(typed(0, -1), "a1");  // B -> C, lag introduced
  const ArcId a2 = dag.add_arc(typed(0, 0), "a2");   // C -> A, back to current
  dag.add_node(OpKind::Remap, {a2}, a0, OpAttrs{}, fuse::ChannelId{}, "A");
  dag.add_node(OpKind::Remap, {a0}, a1, OpAttrs{}, fuse::ChannelId{}, "B");
  dag.add_node(OpKind::Remap, {a1}, a2, OpAttrs{}, fuse::ChannelId{}, "C");

  const auto report = prove_acyclic(g, dag, one_instance_each(dag));
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// A DFS that recursed would need one stack frame per node here. The cycle is
// deliberately longer than any plausible stack.
TEST(Acyclicity, VeryDeepCycleIsFoundWithoutRecursingOnIt) {
  using fuse::ArcId;
  using fuse::OpAttrs;
  using fuse::OpKind;

  constexpr std::uint32_t kN = 200'000;
  const CouplingGraph g = one_component();
  fuse::OperatorDAG dag;
  std::vector<ArcId> arcs;
  arcs.reserve(kN);
  for (std::uint32_t i = 0; i < kN; ++i) {
    arcs.push_back(dag.add_arc(typed(0, 0), {}));
  }
  for (std::uint32_t i = 0; i < kN; ++i) {
    dag.add_node(OpKind::Remap, {arcs[(i + kN - 1) % kN]}, arcs[i], OpAttrs{},
                 fuse::ChannelId{}, {});
  }

  const auto report = prove_acyclic(g, dag, one_instance_each(dag));
  ASSERT_FALSE(report.acyclic);
  EXPECT_EQ(report.cycle.size(), kN);
  EXPECT_EQ(report.cycle_instances.size(), kN);
  // The witness is complete, but the prose must not be a megabyte.
  EXPECT_LT(report.explanation.size(), 16u * 1024u) << report.explanation.size();
  EXPECT_NE(report.explanation.find("more steps"), std::string::npos);
}

// The same shape at a scale a real graph could reach, through the real
// lowering and expansion path rather than a hand-built DAG.
TEST(Acyclicity, LargeRingIsRejectedAndAcceptedWithOneLag) {
  EXPECT_FALSE(check(ring(300, -1)).acyclic);
  const auto lagged = check(ring(300, 150));
  EXPECT_TRUE(lagged.acyclic) << lagged.explanation;
}

// ---------------------------------------------------------------------------
// Regression: lowering must not be able to hide a real algebraic loop from the
// gate. Both of these graphs contain a genuine intra-period cycle, and both
// were certified acyclic before the lowering fixes.
// ---------------------------------------------------------------------------

// DEFECT 1. Three components:
//
//   A.a_out --ch0--> C.c_f      C.c_out --ch2--> A.a_in
//
// is A@k -> C@k -> A@k, a real loop, and was correctly rejected. Adding an
// UNRELATED second writer
//
//   B.b_out --ch1--> C.c_f
//
// made the rejection disappear: `lower()` emits one Sink per field bound to
// the LAST writer, so ch0's chain dead-ended with no consumer and C's input
// side no longer depended on a.a_out at all. The graph then cleared the
// authoring gate too, because multi-writer was only a Warning.
namespace {

CouplingGraph multi_writer_hiding_a_loop(bool include_second_writer) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{2, 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(a, fuse::FieldSpec{
      .name = "a_in", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{
      .name = "b_out", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto c_f = g.add_field(c, fuse::FieldSpec{
      .name = "c_f", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto c_out = g.add_field(c, fuse::FieldSpec{
      .name = "c_out", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});

  g.add_channel("ch0_a_to_c", a_out, c_f);
  if (include_second_writer) g.add_channel("ch1_b_to_c", b_out, c_f);
  g.add_channel("ch2_c_to_a", c_out, a_in);
  return g;
}

// A channel between co-located components on one region with no
// accumulate/extrapolate/lag/remap/relax lowers to ZERO nodes: it only
// rebinds the target field's arc. Shadowed by a second writer it leaves no
// node behind, so a check that only looks for a dangling node cannot see it.
CouplingGraph multi_writer_with_a_node_free_channel() {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  // a and c share ranks, so a->c needs no Transport.
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{5, 1}, .interval = Rational{1}});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(a, fuse::FieldSpec{
      .name = "a_in", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{
      .name = "b_out", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto c_f = g.add_field(c, fuse::FieldSpec{
      .name = "c_f", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto c_out = g.add_field(c, fuse::FieldSpec{
      .name = "c_out", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});

  g.add_channel("ch0_a_to_c", a_out, c_f);   // lowers to zero nodes
  g.add_channel("ch1_b_to_c", b_out, c_f);   // shadows it
  g.add_channel("ch2_c_to_a", c_out, a_in);
  return g;
}

}  // namespace

TEST(Acyclicity, SecondWriterCannotHideARealLoop) {
  // Control: the loop without the extra writer.
  ASSERT_FALSE(check(multi_writer_hiding_a_loop(false)).acyclic)
      << "the one-writer control is a genuine A@k -> C@k -> A@k loop";

  // The original hazard: a second, unrelated writer on the same target used to
  // replace the first, deleting the dependence that closed the loop, and the
  // proof then passed. That graph is now refused outright at authoring
  // (MultiWriterIsAnAuthoringError below), so it cannot reach the proof.
  //
  // The harder question is what happens once the user DOES say how to combine
  // the writers, because then the graph compiles and the proof must still see
  // the loop. It does, and for the right reason: the combine consumes every
  // writer, so no declared dependence is lost.
  CouplingGraph g = multi_writer_hiding_a_loop(true);
  fuse::FieldId merged{};
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    if (g.field(fuse::FieldId{i}).name == "c_f") merged = fuse::FieldId{i};
  }
  ASSERT_TRUE(merged.valid());
  g.merge(merged, fuse::MergeOp::Sum);
  ASSERT_FALSE(validate(g).has_errors()) << validate(g).to_string();

  const auto report = check(g);
  EXPECT_FALSE(report.acyclic)
      << "combining the writers must preserve both dependences, so the "
         "A@k -> C@k -> A@k loop is still there\n"
      << report.explanation;
}

TEST(Acyclicity, SecondWriterOnANodeFreeChannelCannotHideARealLoop) {
  // The shadowed channel lowers to no nodes at all, so nothing downstream can
  // notice it went missing -- only counting writers per target field catches
  // it. The graph is refused at authoring; with a declared combine it compiles
  // and the loop is still visible.
  CouplingGraph g = multi_writer_with_a_node_free_channel();
  EXPECT_TRUE(validate(g).has_errors()) << "an undeclared combine must not compile";
}

// The authoring gate must reject the same graph outright: silently dropping
// every writer but the last is not an implementation of spec §5.1's Merge.
TEST(Acyclicity, MultiWriterIsAnAuthoringError) {
  const fuse::Diagnostics d = validate(multi_writer_hiding_a_loop(true));
  EXPECT_TRUE(d.has_errors()) << d.to_string();
  bool found = false;
  for (const auto& item : d.items()) {
    if (item.code == "E-MULTI-WRITER") found = true;
  }
  EXPECT_TRUE(found) << d.to_string();
}

// DEFECT 2. `b.b_s` is Bidirectional: the target of ch0 and the source of ch1.
// Lowering read `field_arc[b_s]`, which ch0 had already rebound, so ch1
// forwarded ch0's value -- complete with ch0's lag offset -- instead of b's
// own Source. b's Source was left with no consumer, the loop through
// component b never closed, and the leaked negative offset then had the
// Sink edge cut by is_loop_carried().
namespace {

// ch0: a.a_x -> b.b_s (lag `lag_in`)   ch1: b.b_s -> a.a_y (lag `lag_out`)
// ch2: a.a_x -> b.b_t (never lagged)
CouplingGraph bidirectional_alias(int lag_in, int lag_out, bool with_ch2) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});

  const auto a_x = g.add_field(a, fuse::FieldSpec{
      .name = "a_x", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto a_y = g.add_field(a, fuse::FieldSpec{
      .name = "a_y", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_s = g.add_field(b, fuse::FieldSpec{
      .name = "b_s", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Bidirectional});
  const auto b_t = g.add_field(b, fuse::FieldSpec{
      .name = "b_t", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});

  const auto ch0 = g.add_channel("ch0_ax_to_bs", a_x, b_s);
  if (lag_in != 0) {
    g.channel(ch0).lag(lag_in).allow_time_level_mismatch("explicit");
  }
  const auto ch1 = g.add_channel("ch1_bs_to_ay", b_s, a_y);
  if (lag_out != 0) {
    g.channel(ch1).lag(lag_out).allow_time_level_mismatch("explicit");
  } else {
    // Present for an unrelated reason. It switches off lowering's E-IR-TYPE
    // guard, which was the only thing accidentally catching the leaked
    // offset; the gate must not depend on that guard.
    g.channel(ch1).allow_time_level_mismatch("staggered diagnostics");
  }
  if (with_ch2) g.add_channel("ch2_ax_to_bt", a_x, b_t);
  return g;
}

}  // namespace

// FALSE NEGATIVE direction. a@k needs b@k (ch1) and b@k needs a@k (ch2): a
// real loop that the lag on ch0 does not break.
TEST(Acyclicity, BidirectionalFieldDoesNotAliasAwayARealLoop) {
  const auto report = check(bidirectional_alias(/*lag_in=*/1, /*lag_out=*/0,
                                                /*with_ch2=*/true));
  EXPECT_FALSE(report.acyclic)
      << "ch1 must read b.b_s's own Source, not the value ch0 delivered\n"
      << report.explanation;
  EXPECT_NE(report.explanation.find("ch1_bs_to_ay"), std::string::npos)
      << report.explanation;
  EXPECT_NE(report.explanation.find("ch2_ax_to_bt"), std::string::npos)
      << report.explanation;
}

// FALSE POSITIVE direction, same root cause. A lead into the Bidirectional
// field (offset +1) and a lag out of it (offset -1) cancelled along the
// spliced chain, so nothing was cut and a legal lagged scheme was rejected.
// With ch1 lagged there is no intra-period path from b back to a, so the
// two-component system is a perfectly ordinary explicit scheme.
TEST(Acyclicity, LeadIntoAndLagOutOfABidirectionalFieldIsAccepted) {
  const auto report = check(bidirectional_alias(/*lag_in=*/-1, /*lag_out=*/1,
                                                /*with_ch2=*/false));
  EXPECT_TRUE(report.acyclic)
      << "the offsets of two independent channels must not cancel by being "
         "spliced into one chain\n"
      << report.explanation;
}
