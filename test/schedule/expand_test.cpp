#include <gtest/gtest.h>

#include <fuse/base/error.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/schedule/expand.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

using fuse::ComponentDecl;
using fuse::CouplingGraph;
using fuse::OpKind;
using fuse::Rational;
namespace u = fuse::units;

namespace {

CouplingGraph pair(Rational src_dt, Rational dst_dt, fuse::AccumOp accum) {
  CouplingGraph g;
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(fuse::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(fuse::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = src_dt});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = dst_dt});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  if (accum != fuse::AccumOp::Instant) g.channel(ch).accumulate(accum);
  return g;
}

int count_kind(const fuse::TimedDag& td, const fuse::OperatorDAG& d, OpKind k) {
  int n = 0;
  for (const auto& tn : td.nodes) {
    if (d.node(tn.node).kind == k) ++n;
  }
  return n;
}

std::vector<std::int64_t> field_of_kind(const fuse::TimedDag& td,
                                        const fuse::OperatorDAG& d, OpKind k,
                                        std::int64_t fuse::TimedNode::*member) {
  std::vector<std::int64_t> out;
  for (const auto& tn : td.nodes) {
    if (d.node(tn.node).kind == k) out.push_back(tn.*member);
  }
  return out;
}

}  // namespace

TEST(Expand, LockstepGivesOneInstancePerNode) {
  const CouplingGraph g = pair(Rational{1}, Rational{1}, fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_TRUE(td.value().lockstep);
  EXPECT_EQ(td.value().nodes.size(), lowered.dag.node_count());
}

TEST(Expand, SubcyclingRepeatsTheChain) {
  // Source ticks 4x per target tick.
  const CouplingGraph g = pair(Rational{1, 4}, Rational{1}, fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_FALSE(td.value().lockstep);
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Remap), 4);
}

// Accumulation samples on the source clock and reduces on the target clock.
TEST(Expand, AccumulateSamplesPerSourceTickButReducesPerTargetTick) {
  const CouplingGraph g = pair(Rational{1, 4}, Rational{1}, fuse::AccumOp::Mean);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Sample), 4);
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Accumulate), 1);
}

TEST(Expand, NonNestingIntervalsProduceIntersectedInstances) {
  // 1/2 and 1/3 -> 4 slabs per period of 1.
  const CouplingGraph g = pair(Rational{1, 2}, Rational{1, 3},
                               fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_EQ(td.value().period, Rational{1});
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Remap), 4);
}

TEST(Expand, EveryTimedNodeCarriesAnExactTime) {
  const CouplingGraph g = pair(Rational{1, 3}, Rational{1}, fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok());
  for (const auto& tn : td.value().nodes) {
    EXPECT_GE(tn.time, Rational{0});
    EXPECT_LE(tn.time, td.value().period);
  }
}

TEST(Expand, InstancesAreNumberedFromZero) {
  const CouplingGraph g = pair(Rational{1, 3}, Rational{1}, fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs).value();

  for (std::uint32_t n = 0; n < lowered.dag.node_count(); ++n) {
    std::vector<std::int64_t> instances;
    for (const auto& tn : td.nodes) {
      if (tn.node == fuse::NodeId{n}) instances.push_back(tn.instance);
    }
    if (instances.empty()) continue;
    std::sort(instances.begin(), instances.end());
    for (std::size_t i = 0; i < instances.size(); ++i) {
      EXPECT_EQ(instances[i], static_cast<std::int64_t>(i));
    }
  }
}

TEST(Expand, RejectsExplosiveExpansion) {
  const CouplingGraph g = pair(Rational{1, 999983}, Rational{1, 999979},
                               fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_FALSE(td.ok());
  EXPECT_TRUE(td.error().code == "E-SCHED-PERIOD" ||
              td.error().code == "E-SCHED-EXPLODE");
}

// --- regressions for defects in the task brief's reference implementation ---

// A Sink carries the origin of the channel that wrote it (lowering.cpp step 3),
// so a channel loop keyed on `origin == cid` claims it as well as the boundary
// pass keyed on OpKind. The brief did both and emitted every written Sink
// twice.
TEST(Expand, WrittenSinkIsInstantiatedExactlyOncePerSlab) {
  const CouplingGraph g = pair(Rational{1, 4}, Rational{1}, fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  // The Sink rides its channel's chain, so it appears once per slab, exactly
  // as often as the Remap it consumes -- not once per slab *plus* once per
  // target tick.
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Sink), 4);
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Source), 4);
}

// No node may be silently dropped: the schedule must cover the whole DAG.
TEST(Expand, EveryDagNodeIsInstantiatedAtLeastOnce) {
  const CouplingGraph g = pair(Rational{1, 2}, Rational{1, 3},
                               fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;

  std::vector<bool> seen(lowered.dag.node_count(), false);
  for (const auto& tn : td.value().nodes) seen.at(tn.node.value()) = true;
  for (std::size_t i = 0; i < seen.size(); ++i) {
    EXPECT_TRUE(seen[i]) << "node " << i << " was never scheduled";
  }
}

// A Sink is emitted no earlier than the node that produces its input.
TEST(Expand, SinkNeverPrecedesItsProducer) {
  const CouplingGraph g = pair(Rational{1, 4}, Rational{1}, fuse::AccumOp::Mean);
  const auto lowered = lower(g);
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;

  Rational latest_producer{0};
  Rational earliest_sink = td.value().period;
  bool have_sink = false;
  for (const auto& tn : td.value().nodes) {
    const OpKind k = lowered.dag.node(tn.node).kind;
    if (k == OpKind::Sink) {
      if (!have_sink || tn.time < earliest_sink) earliest_sink = tn.time;
      have_sink = true;
    } else if (k != OpKind::Source && k != OpKind::Sample) {
      latest_producer = std::max(latest_producer, tn.time);
    }
  }
  ASSERT_TRUE(have_sink);
  EXPECT_GE(earliest_sink, latest_producer);
}

// Rational::operator* throws fuse::Refusal when the reduced product does
// not fit int64_t. These intervals are legal and their common period is
// representable (P, four ticks of P/4), but tick 3 is 3P/4 and 3P > 2^63, so
// forming the tick with plain Rational arithmetic threw out of a function
// documented to return a Result.
TEST(Expand, UnrepresentableTickIsAnErrorNotAnException) {
  constexpr std::int64_t kP = 8999999999999999999LL;
  const CouplingGraph g = pair(Rational{kP, 4}, Rational{kP},
                               fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  const fuse::ClockSet cs{g};
  ASSERT_NO_THROW({
    const auto td = expand(g, lowered.dag, cs);
    ASSERT_FALSE(td.ok());
    EXPECT_EQ(td.error().code, "E-SCHED-PERIOD");
  });
}

// ClockSet::common_period() throws when the LCM overflows; only
// checked_common_period() reports it. 3037000493 and 3037000507 are coprime
// and their product exceeds 2^63.
TEST(Expand, UnrepresentableCommonPeriodIsAnErrorNotAnException) {
  const CouplingGraph g = pair(Rational{3037000493LL}, Rational{3037000507LL},
                               fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  const fuse::ClockSet cs{g};
  ASSERT_NO_THROW({
    const auto td = expand(g, lowered.dag, cs);
    ASSERT_FALSE(td.ok());
    EXPECT_EQ(td.error().code, "E-SCHED-PERIOD");
  });
}

// src_dt = 1/3, dst_dt = 1/2 -> period 1, three source ticks (0, 1/3, 2/3) and
// two target steps ([0,1/2), [1/2,1)). The brief hard-coded the "other" step
// index to 0 on both accumulation paths, which is a silent wrong answer as
// soon as the period holds more than one target step.
TEST(Expand, AccumulationRecordsBothStepIndices) {
  const CouplingGraph g = pair(Rational{1, 3}, Rational{1, 2},
                               fuse::AccumOp::Mean);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  ASSERT_EQ(td.value().period, Rational{1});

  EXPECT_EQ(field_of_kind(td.value(), lowered.dag, OpKind::Sample,
                          &fuse::TimedNode::src_step),
            (std::vector<std::int64_t>{0, 1, 2}));
  // Source ticks 0 and 1/3 fall in target step 0; 2/3 falls in target step 1.
  EXPECT_EQ(field_of_kind(td.value(), lowered.dag, OpKind::Sample,
                          &fuse::TimedNode::dst_step),
            (std::vector<std::int64_t>{0, 0, 1}));

  EXPECT_EQ(field_of_kind(td.value(), lowered.dag, OpKind::Accumulate,
                          &fuse::TimedNode::dst_step),
            (std::vector<std::int64_t>{0, 1}));
  // Target step 0 closes at 1/2 and last saw source tick 1; step 1 closes at
  // 1 and last saw source tick 2.
  EXPECT_EQ(field_of_kind(td.value(), lowered.dag, OpKind::Accumulate,
                          &fuse::TimedNode::src_step),
            (std::vector<std::int64_t>{1, 2}));
}

// Times are exact multiples of the interval, never accumulated.
TEST(Expand, TickTimesAreExactMultiples) {
  const CouplingGraph g = pair(Rational{1, 7}, Rational{1, 11},
                               fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_EQ(td.value().period, Rational{1});

  std::vector<Rational> source_times;
  for (const auto& tn : td.value().nodes) {
    if (lowered.dag.node(tn.node).kind == OpKind::Source) {
      source_times.push_back(tn.time);
    }
  }
  ASSERT_EQ(source_times.size(), 7u);
  for (std::size_t k = 0; k < source_times.size(); ++k) {
    // Extra parens: the preprocessor splits macro arguments on the comma in
    // Rational{1, 7}; braces do not nest for it, only parentheses do.
    EXPECT_EQ(source_times[k],
              (Rational{static_cast<std::int64_t>(k)} * Rational{1, 7}));
  }
}

// clocks.clock() is bounds-checked with .at(), which throws; a mismatched
// clock set must come back as an error instead.
TEST(Expand, MismatchedClockSetIsAnErrorNotAnException) {
  const CouplingGraph g = pair(Rational{1}, Rational{1}, fuse::AccumOp::Instant);
  const auto lowered = lower(g);
  CouplingGraph bigger = g;
  bigger.add_component(ComponentDecl{
      .name = "c", .ranks = {.first = 2, .count = 1}, .interval = Rational{1}});
  const fuse::ClockSet cs{bigger};

  ASSERT_NO_THROW({
    const auto td = expand(g, lowered.dag, cs);
    ASSERT_FALSE(td.ok());
    EXPECT_EQ(td.error().code, "E-SCHED-EXPAND");
  });
}

TEST(Expand, SpacetimeChannelIsNotLockstep) {
  CouplingGraph g = pair(Rational{1}, Rational{1}, fuse::AccumOp::Instant);
  g.channel(fuse::ChannelId{0}).spacetime(fuse::TemporalBasis::Linear, 1);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_FALSE(td.value().lockstep);
}

// The accumulating path builds no slab decomposition, so kMaxSlabsPerPeriod
// does not bound it; kMaxTimedNodes is the only thing that does. One Sample
// per source tick (999983) plus one Source per source tick pushes past the
// cap.
TEST(Expand, RejectsExpansionThatExceedsTheTimedNodeCap) {
  const CouplingGraph g = pair(Rational{1, 999983}, Rational{1},
                               fuse::AccumOp::Mean);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_FALSE(td.ok());
  EXPECT_EQ(td.error().code, "E-SCHED-EXPLODE");
}
