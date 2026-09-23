// Task 3.7 -- spec §8.2's lockstep fast path.
//
// Lockstep is an *optimisation*, not a second set of semantics. §8.1 makes
// independent per-component intervals the default and lockstep the degenerate
// case, so the fast path is only legitimate if it is unobservable: the plan it
// produces must be the plan the general, slab-based path produces for the same
// graph -- same instructions, in the same order, at the same times, over the
// same buffers -- just computed without touching the slab machinery.
//
// The failure mode this file exists to catch is a fast path that diverges from
// the general one in some corner, so that a graph behaves differently
// depending on whether it happened to be recognised as lockstep. Testing the
// fast path against hand-written expectations cannot catch that. Building each
// graph *both* ways and comparing `ExecutionPlan::to_string()` byte for byte
// can, and that comparison is the backbone here.
//
// It caught a real defect: the version of the fast path this task started from
// emitted every node at `t = period`, which is right for the channel chain and
// wrong for boundary nodes and for the `Sample` of an accumulating channel,
// both of which the general path places at `t = 0`.

#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/schedule/slab.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

using fuse::ChannelId;
using fuse::ComponentDecl;
using fuse::CouplingGraph;
using fuse::ExpandMode;
using fuse::OpKind;
using fuse::Rational;
namespace u = fuse::units;

namespace {

fuse::RegionSizes sizes32() {
  return [](fuse::RegionId, int /*placement*/) -> std::size_t { return 32; };
}

// --- graph builders --------------------------------------------------------
//
// None of these builds a shape hit by the two lowering defects being fixed
// concurrently: no target field has two writers, and no Bidirectional field is
// both a channel target and a channel source.

/// a -> b over one channel. `tweak` decorates the channel.
CouplingGraph pair(Rational dt_a, Rational dt_b,
                   void (*tweak)(CouplingGraph&, ChannelId) = nullptr) {
  CouplingGraph g;
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(fuse::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(fuse::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = dt_a});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = dt_b});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  if (tweak != nullptr) tweak(g, ch);
  return g;
}

void accumulate_mean(CouplingGraph& g, ChannelId c) {
  g.channel(c).accumulate(fuse::AccumOp::Mean);
}
void lag_one(CouplingGraph& g, ChannelId c) {
  g.channel(c).lag(1).allow_time_level_mismatch("explicit scheme");
}
void relax(CouplingGraph& g, ChannelId c) { g.channel(c).relax(0.8); }
void conservative_remap(CouplingGraph& g, ChannelId c) {
  g.channel(c).remap(fuse::RemapFlags::Conservative);
}
void extrapolate(CouplingGraph& g, ChannelId c) { g.channel(c).extrapolate(1); }
void everything(CouplingGraph& g, ChannelId c) {
  g.channel(c)
      .accumulate(fuse::AccumOp::Sum)
      .extrapolate(2)
      .remap(fuse::RemapFlags::Conservative)
      .relax(0.5);
}
void spacetime(CouplingGraph& g, ChannelId c) {
  g.channel(c).spacetime(fuse::TemporalBasis::Constant, 0);
}
void iterate(CouplingGraph& g, ChannelId c) {
  g.channel(c).relax(0.8);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "outer", .channels = {c}, .max_iterations = 7});
}

/// a -> b space-time, a -> c plain, every component on one interval.
///
/// The plain channel is what keeps the fast path *observable*. The general
/// path lays a channel without a `Sample` on the slab decomposition and
/// therefore calls `decompose()` once for it; the fast path never calls it at
/// all. A space-time channel on its own no longer distinguishes the two, since
/// Task 6.3 gave it a `Sample` and put its chain on the two clocks rather than
/// on the slabs -- exactly like an accumulating channel, which
/// `AccumulationDoesNotChangeTheClassification` already notes the counter
/// cannot see.
CouplingGraph spacetime_plus_plain(Rational dt) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = dt});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = dt});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{2, 1}, .interval = dt});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto tb = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto tc = g.add_field(c, fuse::FieldSpec{
      .name = "f_in", .region = r, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  g.channel(g.add_channel("st", s, tb))
      .spacetime(fuse::TemporalBasis::Constant, 0);
  g.add_channel("plain", s, tc);
  return g;
}

/// One component, one field it provides, no channel at all: every node in the
/// lowered DAG is a boundary node, which is the branch the fast path is most
/// likely to get wrong.
CouplingGraph solo(Rational dt) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = dt});
  g.add_field(a, fuse::FieldSpec{.name = "f", .region = r, .units = u::kKelvin,
                                 .intent = fuse::Intent::Provides});
  return g;
}

/// a -> b -> c: three components, two channels, and a Source on b.
CouplingGraph chain3(Rational dt_a, Rational dt_b, Rational dt_c) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = m});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = m});
  const auto rc = g.add_region(fuse::Region{.name = "rc", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = dt_a});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = dt_b});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{2, 1}, .interval = dt_c});
  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(b, fuse::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto c_in = g.add_field(c, fuse::FieldSpec{
      .name = "c_in", .region = rc, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  g.add_channel("a2b", a_out, b_in);
  g.add_channel("b2c", b_out, c_in);
  return g;
}

/// One source fanned out to two targets: two channels sharing a source field,
/// each writing a different target, so neither lowering defect applies.
CouplingGraph fan_out(Rational dt) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = m});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = m});
  const auto rc = g.add_region(fuse::Region{.name = "rc", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = dt});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = dt});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{2, 1}, .interval = dt});
  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(b, fuse::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto c_in = g.add_field(c, fuse::FieldSpec{
      .name = "c_in", .region = rc, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto to_b = g.add_channel("a2b", a_out, b_in);
  g.channel(to_b).accumulate(fuse::AccumOp::Mean);
  g.add_channel("a2c", a_out, c_in);
  return g;
}

// --- the two-ways-and-compare harness --------------------------------------

struct BothWays {
  fuse::ScheduleResult fast;
  fuse::ScheduleResult general;
  std::uint64_t        fast_decompose = 0;
  std::uint64_t        general_decompose = 0;
  std::size_t          node_count = 0;
};

BothWays build_both(const CouplingGraph& g) {
  BothWays out;
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  out.node_count = lowered.dag.node_count();
  const fuse::ClockSet cs{g};

  fuse::reset_decompose_call_count();
  out.fast = schedule(g, lowered.dag, cs, sizes32(), ExpandMode::Auto);
  out.fast_decompose = fuse::decompose_call_count();

  fuse::reset_decompose_call_count();
  out.general = schedule(g, lowered.dag, cs, sizes32(), ExpandMode::ForceGeneral);
  out.general_decompose = fuse::decompose_call_count();
  return out;
}

/// How many channels the general path must decompose: one call per channel
/// that does not accumulate (an accumulating chain is laid out on clock ticks,
/// not on slabs, so it never reaches decompose()).
std::uint64_t expected_general_decompose_calls(const CouplingGraph& g) {
  std::uint64_t n = 0;
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    if (g.channel_at(ChannelId{i}).accumulate.op == fuse::AccumOp::Instant) ++n;
  }
  return n;
}

struct Case {
  std::string   name;
  CouplingGraph graph;
};

/// Every lockstep graph the identity is checked on. Deliberately spans all
/// three of the general path's emission conventions (boundary tick, slab end,
/// accumulate/reduce), several intervals, several channel counts, and an
/// iteration region.
std::vector<Case> lockstep_cases() {
  struct Tweak {
    const char* name;
    void (*fn)(CouplingGraph&, ChannelId);
  };
  const Tweak tweaks[] = {
      {"plain", nullptr},
      {"accumulate", accumulate_mean},
      {"lag", lag_one},
      {"relax", relax},
      {"conservative remap", conservative_remap},
      {"extrapolate", extrapolate},
      {"everything", everything},
      {"iterated", iterate},
  };
  // Integer, unit-fraction, general-fraction, and improper intervals, so that
  // "t = period" and "t = 0" cannot coincide by accident and the period is
  // not always 1.
  const Rational dts[] = {Rational{1}, Rational{1, 2}, Rational{3, 7},
                          Rational{5}};

  std::vector<Case> cases;
  for (const Tweak& tw : tweaks) {
    for (const Rational& dt : dts) {
      cases.push_back({std::string{tw.name} + " dt=" + dt.to_string(),
                       pair(dt, dt, tw.fn)});
    }
  }
  for (const Rational& dt : dts) {
    cases.push_back({"solo boundary-only dt=" + dt.to_string(), solo(dt)});
    cases.push_back({"chain of three dt=" + dt.to_string(),
                     chain3(dt, dt, dt)});
    cases.push_back(
        {"fan out, one leg accumulating dt=" + dt.to_string(), fan_out(dt)});
  }
  return cases;
}

/// The time of the single instruction whose node has this kind.
Rational time_of(const fuse::OperatorDAG& dag, const fuse::ExecutionPlan& plan,
                 OpKind kind) {
  for (const auto& ins : plan.instructions()) {
    if (dag.node(ins.origin_node).kind == kind) return ins.time;
  }
  ADD_FAILURE() << "no instruction of the requested kind";
  return Rational{-1};
}

}  // namespace

// ---------------------------------------------------------------------------
// The backbone: the fast path must be unobservable.
// ---------------------------------------------------------------------------

TEST(Lockstep, FastAndGeneralPathsProduceTheSamePlanByteForByte) {
  for (const Case& c : lockstep_cases()) {
    SCOPED_TRACE(c.name);
    const BothWays b = build_both(c.graph);
    ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
    ASSERT_TRUE(b.general.ok()) << b.general.diags.to_string();
    // The whole artifact: header, period, lockstep flag, every buffer, and
    // every instruction with its time, operands and region markers.
    EXPECT_EQ(b.fast.plan.to_string(), b.general.plan.to_string());
  }
}

/// The comparison above is only worth anything if the two sides really are two
/// algorithms. Pin that the shortcut was taken on one side and not the other.
TEST(Lockstep, TheTwoSidesOfTheComparisonAreDifferentAlgorithms) {
  for (const Case& c : lockstep_cases()) {
    SCOPED_TRACE(c.name);
    const BothWays b = build_both(c.graph);
    ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
    ASSERT_TRUE(b.general.ok()) << b.general.diags.to_string();
    EXPECT_EQ(b.fast_decompose, 0u) << "the fast path decomposed a slab";
    EXPECT_EQ(b.general_decompose,
              expected_general_decompose_calls(c.graph))
        << "the forced-general side did not run the slab machinery, so the "
           "identity check compared the fast path against itself";
  }
}

/// `ExpandMode` is a seam, not a second algorithm: on a graph that is not
/// lockstep, `Auto` never takes the shortcut, so the two modes must agree
/// trivially. If this ever fails, `ForceGeneral` has drifted away from the
/// code the production path actually runs and the identity test above is
/// measuring the wrong thing.
TEST(Lockstep, ForceGeneralIsTheProductionPathOnNonLockstepGraphs) {
  const CouplingGraph cases[] = {
      pair(Rational{1, 2}, Rational{1}),
      pair(Rational{1, 4}, Rational{1}, accumulate_mean),
      pair(Rational{1, 2}, Rational{1, 3}),
      chain3(Rational{1}, Rational{1, 2}, Rational{1}),
  };
  for (const CouplingGraph& g : cases) {
    const BothWays b = build_both(g);
    ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
    ASSERT_TRUE(b.general.ok()) << b.general.diags.to_string();
    EXPECT_EQ(b.fast.plan.to_string(), b.general.plan.to_string());
    EXPECT_FALSE(b.fast.plan.lockstep());
    EXPECT_EQ(b.fast_decompose, b.general_decompose);
  }
}

// ---------------------------------------------------------------------------
// Spec §8.2's two measurable criteria.
// ---------------------------------------------------------------------------

TEST(Lockstep, ExactlyOneInstructionPerDagNode) {
  for (const Case& c : lockstep_cases()) {
    SCOPED_TRACE(c.name);
    const BothWays b = build_both(c.graph);
    ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
    EXPECT_TRUE(b.fast.plan.lockstep());
    EXPECT_EQ(b.fast.plan.instructions().size(), b.node_count);

    // And each node is the origin of exactly one of them.
    std::vector<int> seen(b.node_count, 0);
    for (const auto& ins : b.fast.plan.instructions()) {
      ASSERT_LT(ins.origin_node.value(), b.node_count);
      ++seen[ins.origin_node.value()];
    }
    for (std::size_t i = 0; i < seen.size(); ++i) {
      EXPECT_EQ(seen[i], 1) << "node " << i;
    }
  }
}

TEST(Lockstep, FastPathCallsDecomposeZeroTimes) {
  for (const Case& c : lockstep_cases()) {
    SCOPED_TRACE(c.name);
    fuse::reset_decompose_call_count();
    const auto lowered = lower(c.graph);
    ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
    const fuse::ClockSet cs{c.graph};
    const auto r = schedule(c.graph, lowered.dag, cs, sizes32());
    ASSERT_TRUE(r.ok()) << r.diags.to_string();
    EXPECT_EQ(fuse::decompose_call_count(), 0u)
        << "lockstep must not invoke the slab machinery";
  }
}

TEST(Lockstep, TheCounterItselfCountsSomething) {
  // A counter stuck at zero would make the test above vacuous.
  fuse::reset_decompose_call_count();
  EXPECT_EQ(fuse::decompose_call_count(), 0u);
  const auto d = decompose(Rational{1, 2}, Rational{1}, Rational{1});
  ASSERT_TRUE(d.ok());
  EXPECT_EQ(fuse::decompose_call_count(), 1u);

  fuse::reset_decompose_call_count();
  const auto r = build_both(pair(Rational{1, 2}, Rational{1}));
  ASSERT_TRUE(r.fast.ok());
  EXPECT_GT(r.fast_decompose, 0u);
}

// ---------------------------------------------------------------------------
// Regression: the exact times. A fast path that emits every node at
// `t = period` satisfies both criteria above and is still wrong.
// ---------------------------------------------------------------------------

TEST(Lockstep, BoundaryNodesRunAtTheStartOfThePeriodNotTheEnd) {
  const CouplingGraph g = pair(Rational{1, 2}, Rational{1, 2});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  const auto r = schedule(g, lowered.dag, cs, sizes32());
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  ASSERT_TRUE(r.plan.lockstep());
  EXPECT_EQ(r.plan.period(), Rational(1, 2));

  // The Source is a boundary node: it runs at its component's tick, t = 0.
  EXPECT_EQ(time_of(lowered.dag, r.plan, OpKind::Source), Rational{0});
  // The Sink closes the channel chain: it runs at the end of the only slab.
  EXPECT_EQ(time_of(lowered.dag, r.plan, OpKind::Sink), Rational(1, 2));
}

TEST(Lockstep, AccumulationSamplesAtTheTickStartAndReducesAtItsEnd) {
  const CouplingGraph g = pair(Rational{1}, Rational{1}, accumulate_mean);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  const auto r = schedule(g, lowered.dag, cs, sizes32());
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  ASSERT_TRUE(r.plan.lockstep());

  // Sample runs on the *source* clock, so at the tick's start, exactly as it
  // does under subcycling; only the reduction closes the target step.
  EXPECT_EQ(time_of(lowered.dag, r.plan, OpKind::Sample), Rational{0});
  EXPECT_EQ(time_of(lowered.dag, r.plan, OpKind::Accumulate), Rational{1});
}

TEST(Lockstep, InstructionsAreInNonDecreasingTimeOrder) {
  for (const Case& c : lockstep_cases()) {
    SCOPED_TRACE(c.name);
    const BothWays b = build_both(c.graph);
    ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
    const auto& is = b.fast.plan.instructions();
    for (std::size_t i = 1; i < is.size(); ++i) {
      EXPECT_LE(is[i - 1].time, is[i].time);
    }
  }
}

// ---------------------------------------------------------------------------
// The detection predicate at its boundary. A false positive is the dangerous
// direction: a non-lockstep graph taken for a lockstep one is silently given
// the wrong schedule.
// ---------------------------------------------------------------------------

TEST(LockstepDetection, EqualIntervalsAreLockstep) {
  EXPECT_TRUE(fuse::ClockSet{pair(Rational{1}, Rational{1})}.is_lockstep());
  EXPECT_TRUE(
      fuse::ClockSet{pair(Rational{3, 7}, Rational{3, 7})}.is_lockstep());
  // 2/4 and 1/2 are the same Rational; Rational is always reduced.
  EXPECT_TRUE(fuse::ClockSet{pair(Rational{2, 4}, Rational{1, 2})}.is_lockstep());
}

/// One component off the common interval is enough, wherever it sits.
TEST(LockstepDetection, OneOddIntervalAmongEqualsIsNotLockstep) {
  const CouplingGraph middle = chain3(Rational{1}, Rational{1, 2}, Rational{1});
  const CouplingGraph last = chain3(Rational{1}, Rational{1}, Rational{1, 2});
  const CouplingGraph first = chain3(Rational{1, 2}, Rational{1}, Rational{1});
  EXPECT_FALSE(fuse::ClockSet{middle}.is_lockstep());
  EXPECT_FALSE(fuse::ClockSet{last}.is_lockstep());
  EXPECT_FALSE(fuse::ClockSet{first}.is_lockstep());

  // And the schedule really does take the general path for each of them.
  for (const CouplingGraph* g : {&middle, &last, &first}) {
    const BothWays b = build_both(*g);
    ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
    EXPECT_FALSE(b.fast.plan.lockstep());
    EXPECT_GT(b.fast_decompose, 0u);
  }
}

/// A single component is lockstep: "all intervals equal" is vacuously true of
/// one interval, there is nothing to be out of step with, and the period is
/// that component's interval. Classified lockstep deliberately.
TEST(LockstepDetection, ASingleComponentIsLockstep) {
  const CouplingGraph g = solo(Rational{3, 5});
  EXPECT_TRUE(fuse::ClockSet{g}.is_lockstep());
  const BothWays b = build_both(g);
  ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
  EXPECT_TRUE(b.fast.plan.lockstep());
  EXPECT_EQ(b.fast.plan.period(), Rational(3, 5));
  EXPECT_EQ(b.fast.plan.to_string(), b.general.plan.to_string());
}

/// An empty clock set is lockstep by the same vacuous reading. `expand()`
/// gives it the fallback period of 1 and no instructions. Classified lockstep
/// deliberately; the alternative is a special case with no schedule to differ
/// about.
TEST(LockstepDetection, AnEmptyClockSetIsLockstep) {
  const CouplingGraph empty;
  const fuse::ClockSet cs{empty};
  EXPECT_EQ(cs.size(), 0u);
  EXPECT_TRUE(cs.is_lockstep());

  const auto lowered = lower(empty);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  EXPECT_EQ(lowered.dag.node_count(), 0u);
  const auto fast = expand(empty, lowered.dag, cs, ExpandMode::Auto);
  const auto general = expand(empty, lowered.dag, cs, ExpandMode::ForceGeneral);
  ASSERT_TRUE(fast.ok()) << fast.error().message;
  ASSERT_TRUE(general.ok()) << general.error().message;
  EXPECT_TRUE(fast.value().lockstep);
  EXPECT_TRUE(fast.value().nodes.empty());
  EXPECT_EQ(fast.value().period, general.value().period);
  EXPECT_EQ(fast.value().nodes.size(), general.value().nodes.size());
}

/// `is_lockstep()` compares intervals and ignores `Clock::start`. That is not
/// an oversight the fast path can trip over, because `expand()` ignores the
/// start on *both* paths: every emitted time is `k * interval`, anchored at
/// zero, in the general code as much as in the shortcut. So a non-zero start
/// is classified lockstep, and the two paths still agree exactly.
///
/// (`ClockSet`'s constructor always builds clocks with start 0; a start can
/// only be introduced by assigning a whole `Clock`, as here.)
TEST(LockstepDetection, ANonZeroClockStartDoesNotChangeTheClassification) {
  const CouplingGraph g = pair(Rational{1}, Rational{1});
  fuse::ClockSet cs{g};
  cs.clock(fuse::ComponentId{0}) = fuse::Clock{Rational{1}, Rational{1, 3}};
  ASSERT_EQ(cs.clock(fuse::ComponentId{0}).start(), Rational(1, 3));
  EXPECT_TRUE(cs.is_lockstep());

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto fast = schedule(g, lowered.dag, cs, sizes32(), ExpandMode::Auto);
  const auto general =
      schedule(g, lowered.dag, cs, sizes32(), ExpandMode::ForceGeneral);
  ASSERT_TRUE(fast.ok()) << fast.diags.to_string();
  ASSERT_TRUE(general.ok()) << general.diags.to_string();
  EXPECT_EQ(fast.plan.to_string(), general.plan.to_string());
  // Anchored at zero on both paths, start offset or not.
  EXPECT_EQ(time_of(lowered.dag, fast.plan, OpKind::Source), Rational{0});
}

/// A lag is a `TimeLevel` shift, not an interval, so it cannot make a lockstep
/// graph non-lockstep. Classified lockstep deliberately, and the two paths
/// agree -- neither reads `Channel::lag` when placing instances in time.
TEST(LockstepDetection, ALagDoesNotChangeTheClassification) {
  const CouplingGraph g = pair(Rational{1}, Rational{1}, lag_one);
  EXPECT_TRUE(fuse::ClockSet{g}.is_lockstep());
  const BothWays b = build_both(g);
  ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
  EXPECT_TRUE(b.fast.plan.lockstep());
  EXPECT_EQ(b.fast.plan.to_string(), b.general.plan.to_string());
  EXPECT_EQ(b.fast_decompose, 0u);
}

/// Accumulation puts two nodes of one channel on two different clocks, but
/// both clocks tick once per period here, so it is still lockstep -- and the
/// general path never calls decompose() for it either, which is why the
/// decompose count alone is not enough to tell the two paths apart.
TEST(LockstepDetection, AccumulationDoesNotChangeTheClassification) {
  const CouplingGraph g = pair(Rational{1}, Rational{1}, accumulate_mean);
  EXPECT_TRUE(fuse::ClockSet{g}.is_lockstep());
  const BothWays b = build_both(g);
  ASSERT_TRUE(b.fast.ok()) << b.fast.diags.to_string();
  EXPECT_TRUE(b.fast.plan.lockstep());
  EXPECT_EQ(b.fast.plan.to_string(), b.general.plan.to_string());
  EXPECT_EQ(b.fast_decompose, 0u);
  EXPECT_EQ(b.general_decompose, 0u);
}

/// Spec §8.2 gates the fast path on "all intervals equal **and** no channel is
/// space-time". The clocks alone still say lockstep; the plan must not.
TEST(LockstepDetection, ASpacetimeChannelDisablesTheFastPath) {
  const CouplingGraph g = spacetime_plus_plain(Rational{1});
  EXPECT_TRUE(fuse::ClockSet{g}.is_lockstep())
      << "the clocks are in lockstep; it is the channel that is not";

  fuse::reset_decompose_call_count();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  const auto r = schedule(g, lowered.dag, cs, sizes32());
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_FALSE(r.plan.lockstep());
  // One call, for the plain channel. The fast path would have made zero.
  EXPECT_EQ(fuse::decompose_call_count(), 1u);
}

/// Task 6.3 made the space-time flag a LOWERING change: the channel gains a
/// `Sample` on the source clock and an `Interpolate` carrying the temporal
/// basis on the target clock (spec §7.3's `T`). This test asserted the
/// opposite before 6.3 -- "spacetime changes no lowering rule, so the plans
/// differ only in the `lockstep=` marker" -- which is now false by design, so
/// what it pins instead is that the flag adds exactly that pair and moves
/// nothing else: every other instruction keeps its kind, its order and its
/// time.
TEST(Lockstep, TheSpacetimeFlagAddsOnlyTheSampleInterpolatePair) {
  const auto plan_of = [](const CouplingGraph& g) {
    const auto lowered = lower(g);
    EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
    const fuse::ClockSet cs{g};
    auto r = schedule(g, lowered.dag, cs, sizes32());
    EXPECT_TRUE(r.ok()) << r.diags.to_string();
    return r.plan;
  };
  const auto with_flag =
      plan_of(pair(Rational{1, 2}, Rational{1, 2}, spacetime));
  const auto without = plan_of(pair(Rational{1, 2}, Rational{1, 2}));

  EXPECT_FALSE(with_flag.lockstep());
  EXPECT_TRUE(without.lockstep());
  ASSERT_EQ(with_flag.instructions().size(),
            without.instructions().size() + 2);

  // Exactly one Sample, at the source tick start, and one Interpolate carrying
  // the basis, at the target step end.
  std::size_t samples = 0;
  std::size_t interps = 0;
  for (const auto& ins : with_flag.instructions()) {
    if (ins.kind == OpKind::Sample) {
      ++samples;
      EXPECT_EQ(ins.time, Rational(0));
      EXPECT_EQ(ins.src_step, 0);
    }
    if (ins.kind == OpKind::Interpolate) {
      ++interps;
      EXPECT_EQ(ins.attrs.basis, fuse::TemporalBasis::Constant);
      EXPECT_EQ(ins.time, Rational(1, 2));
      EXPECT_EQ(ins.dst_step, 0);
    }
  }
  EXPECT_EQ(samples, 1u);
  EXPECT_EQ(interps, 1u);

  // Drop the pair and the rest of the stream is unchanged, kind for kind and
  // time for time.
  std::vector<std::pair<OpKind, Rational>> rest;
  for (const auto& ins : with_flag.instructions()) {
    if (ins.kind == OpKind::Sample || ins.kind == OpKind::Interpolate) continue;
    rest.emplace_back(ins.kind, ins.time);
  }
  std::vector<std::pair<OpKind, Rational>> baseline;
  for (const auto& ins : without.instructions()) {
    baseline.emplace_back(ins.kind, ins.time);
  }
  EXPECT_EQ(rest, baseline);
}
