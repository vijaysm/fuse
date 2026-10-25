#include <cstdint>
#include <limits>

#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>

using fuse::AccumOp;
using fuse::ComponentDecl;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::FieldSpec;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
using fuse::RemapFlags;
namespace u = fuse::units;

namespace {

// Two components, one channel: the smallest useful graph.
CouplingGraph make_pair_graph() {
  CouplingGraph g;
  const auto m_f = g.add_mesh(MeshDecl{.name = "fluid_mesh",
                                       .source = "generated:box:8x8x8"});
  const auto m_s = g.add_mesh(MeshDecl{.name = "solid_mesh",
                                       .source = "generated:box:5x5x5"});
  const auto r_f = g.add_region(Region{.name = "fluid_wall", .mesh = m_f,
                                       .entity_set = "wall", .dimension = 2});
  const auto r_s = g.add_region(Region{.name = "solid_wall", .mesh = m_s,
                                       .entity_set = "wall", .dimension = 2});

  const auto fluid = g.add_component(
      ComponentDecl{.name = "fluid", .ranks = fuse::RankSet{0, 2},
                    .interval = Rational{1, 2}});
  const auto solid = g.add_component(
      ComponentDecl{.name = "solid", .ranks = fuse::RankSet{2, 2},
                    .interval = Rational{1}});

  const auto q_out = g.add_field(
      fluid, FieldSpec{.name = "heat_flux", .region = r_f,
                       .units = u::kHeatFluxDensity,
                       .conservation = Conservation::Extensive,
                       .intent = fuse::Intent::Provides});
  const auto q_in = g.add_field(
      solid, FieldSpec{.name = "q_wall", .region = r_s,
                       .units = u::kHeatFluxDensity,
                       .conservation = Conservation::Extensive,
                       .intent = fuse::Intent::Requires});

  const auto ch = g.add_channel("flux", q_out, q_in);
  g.channel(ch).accumulate(AccumOp::Mean).remap(RemapFlags::Conservative);
  return g;
}

}  // namespace

TEST(CouplingGraph, EmptyGraphHasNoEntities) {
  const CouplingGraph g;
  EXPECT_EQ(g.component_count(), 0u);
  EXPECT_EQ(g.channel_count(), 0u);
}

TEST(CouplingGraph, IdsAreSequentialAndValid) {
  CouplingGraph g;
  const auto a = g.add_mesh(MeshDecl{.name = "a"});
  const auto b = g.add_mesh(MeshDecl{.name = "b"});
  EXPECT_TRUE(a.valid());
  EXPECT_EQ(a.value(), 0u);
  EXPECT_EQ(b.value(), 1u);
  EXPECT_EQ(g.mesh_count(), 2u);
}

TEST(CouplingGraph, BuildsThePairGraph) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_EQ(g.mesh_count(), 2u);
  EXPECT_EQ(g.region_count(), 2u);
  EXPECT_EQ(g.component_count(), 2u);
  EXPECT_EQ(g.field_count(), 2u);
  EXPECT_EQ(g.channel_count(), 1u);
}

TEST(CouplingGraph, FieldsAreOwnedByTheirComponent) {
  const CouplingGraph g = make_pair_graph();
  const auto fluid = g.find_component("fluid").value();
  const auto q = g.find_field(fluid, "heat_flux").value();
  EXPECT_EQ(g.owner_of(q), fluid);
  EXPECT_EQ(g.component(fluid).fields.size(), 1u);
}

TEST(CouplingGraph, FindReturnsNulloptWhenAbsent) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_FALSE(g.find_component("nonexistent").has_value());
  const auto fluid = g.find_component("fluid").value();
  EXPECT_FALSE(g.find_field(fluid, "nope").has_value());
}

TEST(CouplingGraph, ChannelBuilderMutatesStoredChannel) {
  const CouplingGraph g = make_pair_graph();
  const auto& ch = g.channel_at(fuse::ChannelId{0});
  EXPECT_EQ(ch.accumulate.op, AccumOp::Mean);
  EXPECT_TRUE(has(ch.remap.flags, RemapFlags::Conservative));
}

TEST(CouplingGraph, ComponentsKeepIndependentIntervals) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_EQ(g.component(g.find_component("fluid").value()).interval,
            Rational(1, 2));
  EXPECT_EQ(g.component(g.find_component("solid").value()).interval,
            Rational(1));
}

// Regression test: a ChannelBuilder held across intervening add_channel()
// calls must not dangle when channels_ reallocates. Before the fix,
// ChannelBuilder stored a `Channel&` resolved at construction time; this
// test forces at least one reallocation of the backing vector while the
// builder is alive, then mutates through it. Under ASan the pre-fix code
// aborts with a heap-use-after-free; without ASan it silently mutates freed
// memory and the assertions below fail because the write never reaches the
// real, reallocated Channel.
TEST(CouplingGraph, ChannelBuilderSurvivesReallocation) {
  CouplingGraph g;
  const auto first = g.add_channel("first", fuse::FieldId{0}, fuse::FieldId{1});
  auto builder = g.channel(first);  // held, not consumed as a temporary

  // Force at least one reallocation of the std::vector<Channel> backing
  // store while `builder` is still alive.
  for (int i = 0; i < 100; ++i) {
    g.add_channel("filler", fuse::FieldId{0}, fuse::FieldId{1});
  }

  builder.accumulate(AccumOp::Sum).remap(RemapFlags::Conservative);

  const auto& ch = g.channel_at(first);
  EXPECT_EQ(ch.accumulate.op, AccumOp::Sum);
  EXPECT_TRUE(has(ch.remap.flags, RemapFlags::Conservative));
}

// window_sample_capacity(): hand-derived closed-form values, exact
// Rational arithmetic (no std::ceil-on-a-double anywhere in the
// production function). This is the SAME
// function validate_coupling_window()'s E-SCHED-WINDOW-CAPACITY and
// window_lowering.cpp's window_source_type() both call, so a wrong value
// here would silently mis-size a compiled field as well as the schedule-
// time refusal.
TEST(CouplingGraph, WindowSampleCapacityMatchesHandDerivedValues) {
  using fuse::DtPolicy;
  using fuse::window_sample_capacity;

  // window=1, interval=1/4: ratio=4 exact -> 4 substeps + 1 window-start.
  EXPECT_EQ(window_sample_capacity(Rational{1}, Rational{1, 4},
                                   DtPolicy::Adjustable),
            5u);
  // Fixed adds one more for the possible straddle.
  EXPECT_EQ(
      window_sample_capacity(Rational{1}, Rational{1, 4}, DtPolicy::Fixed),
      6u);

  // window=13, interval=7: ratio=13/7, not exact -> ceil=2, +1=3.
  EXPECT_EQ(
      window_sample_capacity(Rational{13}, Rational{7}, DtPolicy::Adjustable),
      3u);
  EXPECT_EQ(
      window_sample_capacity(Rational{13}, Rational{7}, DtPolicy::Fixed), 4u);

  // window=7, interval=13: ratio=7/13 < 1 -> ceil=1, +1=2 (the member
  // takes exactly one, truncated/straddling substep to cover the window).
  EXPECT_EQ(
      window_sample_capacity(Rational{7}, Rational{13}, DtPolicy::Adjustable),
      2u);
  EXPECT_EQ(
      window_sample_capacity(Rational{7}, Rational{13}, DtPolicy::Fixed), 3u);

  // window=1, interval=1/3: ratio=3 exact -> 3+1=4.
  EXPECT_EQ(window_sample_capacity(Rational{1}, Rational{1, 3},
                                   DtPolicy::Adjustable),
            4u);

  // Non-positive interval: never fails, returns the harmless minimum (the
  // caller's own E-SCHED-WINDOW-SHORTEN/E-SCHED-WINDOW-INTERVAL gate is
  // what actually refuses this case).
  EXPECT_EQ(
      window_sample_capacity(Rational{1}, Rational{0}, DtPolicy::Adjustable),
      1u);
}

// window_sample_times(): a hand-derived check -- B (interval 7, DtPolicy::Fixed), window length 13,
// window index 1 (the SECOND window: T0=13, T1=26) -- {13, 14, 21, 26}:
// T0 itself, the two native grid points (0, 7, 14, 21, 28, ...) strictly
// inside (13, 26), and T1.
TEST(CouplingGraph, WindowSampleTimesMatchesHandDerivedCheck) {
  using fuse::DtPolicy;
  using fuse::window_sample_times;

  const std::vector<Rational> got = window_sample_times(
      Rational{7}, DtPolicy::Fixed, Rational{0}, Rational{13}, 1);
  const std::vector<Rational> want = {Rational{13}, Rational{14},
                                      Rational{21}, Rational{26}};
  EXPECT_EQ(got, want);
}

// window_sample_times() is a pure function of the graph alone (interval,
// policy, member_start, window_length, window_index) -- no runtime
// state -- which is what is meant to let a CONSUMER (reconstructing a
// provider's own grid, Runner::Impl::advance_window()'s own top comment)
// compute the SAME answer the PROVIDER's own subcycling loop does,
// without ever reading the provider's own carried-forward clock. This
// test drives a REAL carried-forward clock the way a genuine Fixed
// subcycling member does -- a straddling step's overshoot carrying into
// the next window, over every window of the 91-unit (lcm(7,13)) period
// -- using a small reference re-implementation of `DtPolicy::Fixed`
// substepping local to this test (never shortening a step, straddling
// past the window end when the native grid does not land exactly on
// it), deliberately NOT `fuse::detail::plan_substeps()` itself (a
// runtime-layer function this model-layer test does not link against),
// so the two computations share no code, only the same specification.
// window_sample_times()'s own header comment states the correspondence
// once there has been a carry-in: its own leading T0 entry has no
// counterpart in the member's real substep grid (dropped below), and
// every remaining entry lines up with the member's own next real
// substep end, one for one.
TEST(CouplingGraph, WindowSampleTimesAgreesWithTheRealCarriedForwardGrid) {
  using fuse::DtPolicy;
  using fuse::window_sample_capacity;
  using fuse::window_sample_times;

  const Rational window_length{13};
  const Rational interval{7};
  const std::size_t capacity =
      window_sample_capacity(window_length, interval, DtPolicy::Fixed);

  Rational clock{0};  // the member's own REAL, carried-forward clock.
  for (std::int64_t w = 0; w < 7; ++w) {  // 91 / 13 = 7 windows / period
    const Rational T0 = Rational(w) * window_length;
    const Rational T1 = T0 + window_length;
    const bool has_carry_in = clock > T0;

    // The member's OWN real grid this window: never shortens a step, so
    // it walks its own native grid from wherever `clock` (this window's
    // own pass_start_clock, in advance_window()'s own vocabulary)
    // already sits, straddling past T1 on whichever step first reaches
    // or exceeds it.
    std::vector<Rational> real_points;
    Rational t = clock;
    while (t < T1) {
      const Rational end = t + interval;
      real_points.push_back(end < T1 ? end : (end > T1 ? T1 : end));
      t = end;
      if (end >= T1) break;  // that step straddled (or landed exactly);
                             // the member's real clock now sits at `end`,
                             // carried into the NEXT window.
    }
    const Rational new_clock = t < T1 ? T1 : t;

    const std::vector<Rational> reconstructed = window_sample_times(
        interval, DtPolicy::Fixed, Rational{0}, window_length, w);
    ASSERT_GE(reconstructed.size(), 2u) << "window " << w;
    ASSERT_LE(reconstructed.size(), capacity)
        << "window " << w << ": window_sample_capacity() must bound every "
           "window_sample_times() this member can ever produce";

    const std::vector<Rational> expected(
        reconstructed.begin() + (has_carry_in ? 2 : 1), reconstructed.end());
    EXPECT_EQ(real_points, expected) << "window " << w;

    clock = new_clock;
  }
}

// ---- the no-exceptions contract ----------------------------------------
//
// Both functions must not compute with the throwing Rational operators
// (+, -, *, /), any of which can FUSE_REFUSE on an unrepresentable
// intermediate -- an uncaught Refusal, or an abort in a
// FUSE_ENABLE_EXCEPTIONS=OFF build, from functions with no Result in
// their signature to report that through. Neither of these inputs is
// anything a legitimate graph would ever produce (every real
// window_length/interval pair already fits comfortably once reduced);
// they exist to confirm the fallback path holds under genuinely
// adversarial input.

TEST(CouplingGraph, WindowSampleCapacityNeverThrowsOnAnUnrepresentableRatio) {
  using fuse::DtPolicy;
  using fuse::window_sample_capacity;
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  const Rational window_length{kMax};
  const Rational interval{1, kMax};
  std::size_t result = 0;
  EXPECT_NO_THROW(
      result = window_sample_capacity(window_length, interval, DtPolicy::Fixed));
  // An unrepresentable ratio reports the capacity as unrepresentably large,
  // so the caller's E-SCHED-WINDOW-CAPACITY gate refuses it (the
  // sentinel is the schedule gate's).
  EXPECT_EQ(result, std::numeric_limits<std::size_t>::max());
}

TEST(CouplingGraph, WindowSampleTimesNeverThrowsOnAnUnrepresentableProduct) {
  using fuse::DtPolicy;
  using fuse::window_sample_times;
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  std::vector<Rational> result;
  EXPECT_NO_THROW(result = window_sample_times(Rational{1}, DtPolicy::Fixed,
                                               Rational{0}, Rational{kMax}, kMax));
  EXPECT_GE(result.size(), 2u)
      << "even when window_index * window_length overflows, the function "
         "must return a well-formed (if degraded) list ending in T1, not "
         "throw";
}
