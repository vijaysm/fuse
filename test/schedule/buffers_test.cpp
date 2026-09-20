#include <gtest/gtest.h>
#include <fuse/model/validate.hpp>
#include <fuse/base/error.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/schedule/buffers.hpp>
#include <fuse/schedule/expand.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>  // std::size_t limits only; the thrown type is fuse::InvalidArgument (buffers.hpp)
#include <utility>
#include <vector>

using fuse::ComponentDecl;
using fuse::CouplingGraph;
using fuse::Rational;
namespace u = fuse::units;

namespace {

CouplingGraph pair_with_accumulate() {
  CouplingGraph g;
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(fuse::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(fuse::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1, 2}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).accumulate(fuse::AccumOp::Mean);
  return g;
}

/// Two co-located components, optionally coupled through `lag(1)`. The two
/// variants differ *only* in the lag, which makes the pair a clean
/// discriminator for the persistence rule.
CouplingGraph pair_with_optional_lag(bool lagged) {
  CouplingGraph g;
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(fuse::Region{.name = "rs", .mesh = mf});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  if (lagged) g.channel(ch).lag(1).allow_time_level_mismatch("explicit scheme");
  return g;
}

CouplingGraph pair_with_components(int components) {
  CouplingGraph g;
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(fuse::Region{.name = "rs", .mesh = mf});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = rf, .components = components, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = rs, .components = components,
      .units = u::kKelvin, .intent = fuse::Intent::Requires});
  g.add_channel("c", s, t);
  return g;
}

fuse::OperatorDAG lower_ok(const CouplingGraph& g) {
  auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  return std::move(lowered.dag);
}

fuse::TimedDag expand_ok(const CouplingGraph& g, const fuse::OperatorDAG& dag) {
  const fuse::ClockSet cs{g};
  auto td = expand(g, dag, cs);
  EXPECT_TRUE(td.ok()) << td.error().message;
  return td.value();
}

fuse::RegionSizes constant_sizes(std::size_t n) {
  return [n](fuse::RegionId r) {
    EXPECT_TRUE(r.valid()) << "the callback must never see an invalid region";
    return n;
  };
}

fuse::BufferPlan plan_sized(const CouplingGraph& g, std::size_t per_region) {
  const fuse::OperatorDAG dag = lower_ok(g);
  return plan_buffers(dag, expand_ok(g, dag), constant_sizes(per_region));
}

fuse::BufferPlan plan_for(const CouplingGraph& g) {
  // 100 elements per region until a mesh backend exists (Phase 4 Task 4.2).
  return plan_sized(g, 100);
}

}  // namespace

TEST(Buffers, OneDeclarationPerArc) {
  const CouplingGraph g = pair_with_accumulate();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto plan = plan_for(g);
  EXPECT_EQ(plan.buffers.size(), lowered.dag.arc_count());
  // buffers[i] describes ArcId{i}: consumers index the plan by arc.
  for (std::size_t i = 0; i < plan.buffers.size(); ++i) {
    EXPECT_EQ(plan.buffers[i].arc.value(), i);
  }
}

// Spec §4.2: the coupler allocates only its own intermediates.
TEST(Buffers, ComponentOwnedArcsAreNotAllocated) {
  const auto plan = plan_for(pair_with_accumulate());
  int owned = 0;
  for (const auto& b : plan.buffers) {
    if (b.is_component_owned) ++owned;
  }
  EXPECT_GT(owned, 0);
  std::size_t coupler_total = 0;
  for (const auto& b : plan.buffers) {
    if (!b.is_component_owned) coupler_total += b.elements * b.components;
  }
  EXPECT_EQ(plan.total_coupler_elements, coupler_total);
}

// The flag must be *exactly* the Source/Sink outputs, in both directions: a
// registered field wrongly treated as a coupler buffer means the component
// never sees the coupled values, and a coupler intermediate wrongly treated as
// registered means writing through a pointer nobody bound. Both are silent, so
// pin the whole set rather than just its size.
TEST(Buffers, ComponentOwnedIsExactlyTheSourceAndSinkOutputs) {
  const CouplingGraph g = pair_with_accumulate();
  const fuse::OperatorDAG dag = lower_ok(g);
  const auto plan = plan_buffers(dag, expand_ok(g, dag), constant_sizes(100));
  ASSERT_EQ(plan.buffers.size(), dag.arc_count());

  std::vector<char> expect_owned(dag.arc_count(), 0);
  int sources = 0;
  int sinks = 0;
  for (std::size_t i = 0; i < dag.node_count(); ++i) {
    const fuse::Node& n = dag.node(fuse::NodeId{static_cast<std::uint32_t>(i)});
    if (n.kind == fuse::OpKind::Source) ++sources;
    if (n.kind == fuse::OpKind::Sink) ++sinks;
    if (n.kind == fuse::OpKind::Source || n.kind == fuse::OpKind::Sink) {
      ASSERT_TRUE(n.output.valid());
      expect_owned[n.output.value()] = 1;
    }
  }
  EXPECT_EQ(sources, 1);
  EXPECT_EQ(sinks, 1);

  for (std::size_t i = 0; i < plan.buffers.size(); ++i) {
    EXPECT_EQ(plan.buffers[i].is_component_owned, expect_owned[i] != 0)
        << "arc " << i << " (\"" << plan.buffers[i].label << "\")";
  }
  // The intermediates the coupler does own are spec §4.2's list: the waveform
  // sample, the accumulator, transport staging, the remap target, diagnose.
  EXPECT_EQ(plan.total_coupler_elements, 100u * (dag.arc_count() - 2));
}

TEST(Buffers, WaveformArcsArePersistent) {
  const auto plan = plan_for(pair_with_accumulate());
  bool found_persistent = false;
  for (const auto& b : plan.buffers) {
    if (b.is_persistent) found_persistent = true;
  }
  EXPECT_TRUE(found_persistent)
      << "an accumulating channel must have a persistent waveform buffer";
}

// ...and nothing else is, in a graph with no lag. "Some buffer is persistent"
// would also hold for a plan that marked every buffer persistent.
TEST(Buffers, PersistentIsExactlyTheWaveformArcsWhenThereIsNoLag) {
  const CouplingGraph g = pair_with_accumulate();
  const fuse::OperatorDAG dag = lower_ok(g);
  const auto plan = plan_buffers(dag, expand_ok(g, dag), constant_sizes(100));
  ASSERT_EQ(plan.buffers.size(), dag.arc_count());

  int waveforms = 0;
  for (std::size_t i = 0; i < plan.buffers.size(); ++i) {
    const bool is_wave = dag.arc(fuse::ArcId{static_cast<std::uint32_t>(i)})
                             .type.kind == fuse::ValueKind::Waveform;
    if (is_wave) ++waveforms;
    EXPECT_EQ(plan.buffers[i].is_persistent, is_wave)
        << "arc " << i << " (\"" << plan.buffers[i].label << "\")";
  }
  EXPECT_EQ(waveforms, 1) << "Sample produces the one waveform accumulator";
}

// Spec §4.3: lag(k) *is* a TimeLevel offset of -k. The lagged buffer is
// written in period p and read in period p+1, so recycling it inside a period
// makes the consumer read this period's value -- silently turning an explicit
// lagged scheme into an unstable implicit one, with no error at any point.
TEST(Buffers, ArcsReachedThroughALagArePersistent) {
  const CouplingGraph g = pair_with_optional_lag(true);
  ASSERT_FALSE(fuse::validate(g).has_errors()) << fuse::validate(g).to_string();

  const fuse::OperatorDAG dag = lower_ok(g);
  const auto plan = plan_buffers(dag, expand_ok(g, dag), constant_sizes(100));
  ASSERT_EQ(plan.buffers.size(), dag.arc_count());

  int lagged = 0;
  for (std::size_t i = 0; i < plan.buffers.size(); ++i) {
    const fuse::Arc& a = dag.arc(fuse::ArcId{static_cast<std::uint32_t>(i)});
    ASSERT_EQ(a.type.kind, fuse::ValueKind::Instant)
        << "this fixture does not accumulate, so no waveform can mask the "
           "lag rule";
    const bool is_lagged = a.type.time.offset < Rational{0};
    if (is_lagged) {
      ++lagged;
      // The delay is realised in a buffer the coupler owns: a component's
      // registered array holds the current value, not last period's.
      EXPECT_FALSE(plan.buffers[i].is_component_owned)
          << "arc " << i << " (\"" << plan.buffers[i].label << "\")";
    }
    EXPECT_EQ(plan.buffers[i].is_persistent, is_lagged)
        << "arc " << i << " (\"" << plan.buffers[i].label << "\") type "
        << a.type.to_string();
  }
  EXPECT_GT(lagged, 0) << "lag(1) must put a negative offset on some arc";
}

// The control for the test above: the same graph without the lag has no
// persistent buffer at all. Without this, a plan marking everything persistent
// would still pass the positive half of the lag test.
TEST(Buffers, TheSameGraphWithoutALagHasNoPersistentBuffer) {
  const CouplingGraph g = pair_with_optional_lag(false);
  ASSERT_FALSE(fuse::validate(g).has_errors());
  const auto plan = plan_for(g);
  ASSERT_FALSE(plan.buffers.empty());
  for (const auto& b : plan.buffers) {
    EXPECT_FALSE(b.is_persistent) << "arc \"" << b.label << "\"";
  }
}

TEST(Buffers, ElementCountsComeFromTheRegionSizeCallback) {
  const auto plan = plan_for(pair_with_accumulate());
  for (const auto& b : plan.buffers) {
    if (b.is_component_owned) continue;
    EXPECT_EQ(b.elements, 100u);
  }
}

// `elements` describes the region, not who allocates it. Phase 4 has to check
// a component's registered Layout::count against the region it claims to
// cover, which is impossible if the plan zeroes the count for exactly those
// arcs. Allocation is governed by `is_component_owned` alone.
TEST(Buffers, ComponentOwnedArcsStillReportTheirRegionSize) {
  const auto plan = plan_for(pair_with_accumulate());
  int owned = 0;
  for (const auto& b : plan.buffers) {
    if (!b.is_component_owned) continue;
    ++owned;
    EXPECT_EQ(b.elements, 100u) << "arc \"" << b.label << "\"";
  }
  EXPECT_EQ(owned, 2);
}

TEST(Buffers, LabelsAreNonEmpty) {
  const auto plan = plan_for(pair_with_accumulate());
  for (const auto& b : plan.buffers) {
    EXPECT_FALSE(b.label.empty());
  }
}

TEST(Buffers, ComponentCountMultipliesTheTotal) {
  const auto one = plan_sized(pair_with_components(1), 100);
  const auto three = plan_sized(pair_with_components(3), 100);
  ASSERT_EQ(one.buffers.size(), three.buffers.size());
  ASSERT_GT(one.total_coupler_elements, 0u);
  EXPECT_EQ(three.total_coupler_elements, 3 * one.total_coupler_elements);
}

// --- rejected inputs -------------------------------------------------------
// All checked unconditionally: an assert compiles out under Release's
// -O3 -DNDEBUG, and unsigned wraparound is *defined* behaviour, so neither
// ASan nor UBSan would report the size cases either.

TEST(Buffers, RejectsAnOverflowingBufferSize) {
  const CouplingGraph g = pair_with_components(3);
  const fuse::OperatorDAG dag = lower_ok(g);
  const fuse::TimedDag td = expand_ok(g, dag);
  const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
  EXPECT_THROW((void)plan_buffers(dag, td, constant_sizes(huge)),
               fuse::InvalidArgument);
}

TEST(Buffers, RejectsAnOverflowingTotal) {
  const CouplingGraph g = pair_with_components(1);
  const fuse::OperatorDAG dag = lower_ok(g);
  const fuse::TimedDag td = expand_ok(g, dag);
  // Each individual buffer fits; there is more than one coupler buffer, so the
  // running sum does not.
  const std::size_t big = (std::numeric_limits<std::size_t>::max() / 4) * 3;
  EXPECT_THROW((void)plan_buffers(dag, td, constant_sizes(big)), fuse::InvalidArgument);
}

// validate() has no positivity rule for FieldSpec::components, so a graph with
// components = -1 reaches here clean. static_cast<std::size_t>(-1) is 2^64-1
// and the product wraps back to a small, plausible allocation.
TEST(Buffers, RejectsANonPositiveComponentCount) {
  for (const int bad : {0, -1}) {
    const CouplingGraph g = pair_with_components(bad);
    const fuse::OperatorDAG dag = lower_ok(g);
    const fuse::TimedDag td = expand_ok(g, dag);
    EXPECT_THROW((void)plan_buffers(dag, td, constant_sizes(100)),
                 fuse::InvalidArgument)
        << "components = " << bad;
  }
}

TEST(Buffers, RejectsAnEmptyRegionSizesCallback) {
  const CouplingGraph g = pair_with_accumulate();
  const fuse::OperatorDAG dag = lower_ok(g);
  const fuse::TimedDag td = expand_ok(g, dag);
  EXPECT_THROW((void)plan_buffers(dag, td, fuse::RegionSizes{}),
               fuse::InvalidArgument);
}

TEST(Buffers, RejectsAnArcWithNoRegion) {
  fuse::OperatorDAG dag;
  const fuse::ArcId arc = dag.add_arc(fuse::ValueType{}, "unplaced");
  dag.add_node(fuse::OpKind::Source, {}, arc, fuse::OpAttrs{},
               fuse::ChannelId{}, "unplaced");
  EXPECT_THROW((void)plan_buffers(dag, fuse::TimedDag{}, constant_sizes(100)),
               fuse::InvalidArgument);
}

TEST(Buffers, RejectsATimedDagBuiltFromADifferentDag) {
  fuse::OperatorDAG dag;
  const fuse::ArcId arc =
      dag.add_arc(fuse::ValueType{.region = fuse::RegionId{0}}, "x");
  dag.add_node(fuse::OpKind::Source, {}, arc, fuse::OpAttrs{},
               fuse::ChannelId{}, "x");

  fuse::TimedDag td;
  td.nodes.push_back(fuse::TimedNode{.node = fuse::NodeId{7}});
  EXPECT_THROW((void)plan_buffers(dag, td, constant_sizes(100)), fuse::InvalidArgument);

  fuse::TimedDag ok;
  ok.nodes.push_back(fuse::TimedNode{.node = fuse::NodeId{0}});
  EXPECT_NO_THROW((void)plan_buffers(dag, ok, constant_sizes(100)));
}
