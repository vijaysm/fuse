#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/base/error.hpp>
#include <fuse/model/descriptor.hpp>
#include <fuse/model/validate.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>

using fuse::AccumOp;
using fuse::ComponentDecl;
using fuse::ComponentId;
using fuse::ComponentKind;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::deserialize;
using fuse::Diagnostics;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
using fuse::Window;
namespace u = fuse::units;

namespace {

bool has_code(const Diagnostics& d, std::string_view code) {
  return std::any_of(d.items().begin(), d.items().end(),
                     [code](const auto& i) { return i.code == code; });
}

// A graph that validates clean; individual tests perturb one thing.
//
// `src`/`dst` are `Conservation::Intensive`, not `Extensive` -- this
// fixture's "flux" channel declares no remap at all (same region on both
// ends), and connect.hpp's `check_channel()` refuses an `Extensive`-target
// channel whose declared transfer is a plain Copy (ConnectTest/ValidateTest
// exercise that refusal directly). None of the tests built on this fixture
// are ABOUT conservation, so `Intensive` keeps them exercising what they
// actually name instead of all failing on an unrelated
// E-CONN-CONSERVATION.
struct Fixture {
  CouplingGraph g;
  fuse::ComponentId fluid, solid;
  fuse::FieldId src, dst;
  fuse::ChannelId ch;

  Fixture() {
    const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:box"});
    const auto r = g.add_region(Region{.name = "wall", .mesh = m,
                                       .entity_set = "w", .dimension = 2});
    fluid = g.add_component(ComponentDecl{.name = "fluid",
                                          .ranks = fuse::RankSet{0, 1},
                                          .interval = Rational{1}});
    solid = g.add_component(ComponentDecl{.name = "solid",
                                          .ranks = fuse::RankSet{1, 1},
                                          .interval = Rational{1}});
    src = g.add_field(fluid, FieldSpec{.name = "q", .region = r,
                                       .units = u::kHeatFluxDensity,
                                       .conservation = Conservation::Intensive,
                                       .intent = Intent::Provides});
    dst = g.add_field(solid, FieldSpec{.name = "q_in", .region = r,
                                       .units = u::kHeatFluxDensity,
                                       .conservation = Conservation::Intensive,
                                       .intent = Intent::Requires});
    ch = g.add_channel("flux", src, dst);
  }
};

}  // namespace

TEST(Validate, CleanGraphHasNoErrors) {
  Fixture f;
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(d.has_errors()) << d.to_string();
}

TEST(Validate, RejectsDuplicateComponentNames) {
  Fixture f;
  f.g.add_component(ComponentDecl{.name = "fluid"});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-DUP-COMP")) << d.to_string();
}

// MPI connect (fuse/model/launch.hpp). These are the
// per-member checks validate() itself owns; the cross-component questions
// (do the launched components' RankSets actually partition the declared
// world, does an appnum collide with another component's) are
// fuse::transport::validate_launch_layout()'s job -- see
// test/transport/connect_test.cpp.
TEST(Validate, RejectsAnMpmdComponentWithNoAppnum) {
  Fixture f;
  f.g.add_component(ComponentDecl{
      .name = "mpmd-c",
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Mpmd}});  // appnum defaults to -1
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LAUNCH-APPNUM")) << d.to_string();
}

TEST(Validate, AcceptsAnMpmdComponentWithANonNegativeAppnum) {
  Fixture f;
  f.g.add_component(ComponentDecl{
      .name = "mpmd-c",
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Mpmd, .appnum = 0}});
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-LAUNCH-APPNUM")) << d.to_string();
}

TEST(Validate, RejectsAConnectComponentWithNoRendezvousTarget) {
  Fixture f;
  f.g.add_component(ComponentDecl{
      .name = "connect-c",
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Connect}});  // no port_file/service_name
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LAUNCH-CONNECT-TARGET")) << d.to_string();
}

TEST(Validate, AcceptsAConnectComponentWithAPortFile) {
  Fixture f;
  f.g.add_component(ComponentDecl{
      .name = "connect-c",
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Connect,
                                 .port_file = "/tmp/fuse-connect.port"}});
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-LAUNCH-CONNECT-TARGET")) << d.to_string();
}

// Adapter host.
TEST(Validate, RejectsAPluginComponentWithNoPluginPath) {
  Fixture f;
  f.g.add_component(ComponentDecl{
      .name = "plugin-c",
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin}});  // plugin_path defaults empty
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LAUNCH-PLUGIN-PATH")) << d.to_string();
}

TEST(Validate, AcceptsAPluginComponentWithAPluginPath) {
  Fixture f;
  f.g.add_component(ComponentDecl{
      .name = "plugin-c",
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin,
                                 .plugin_path = "/opt/fuse-plugins/ideal-gas.so"}});
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-LAUNCH-PLUGIN-PATH")) << d.to_string();
}

TEST(Validate, RejectsUnitMismatch) {
  Fixture f;
  const auto r = f.g.field(f.dst).region;
  const auto bad = f.g.add_field(
      f.solid, FieldSpec{.name = "T_in", .region = r, .units = u::kKelvin,
                         .conservation = Conservation::Intensive,
                         .intent = Intent::Requires});
  f.g.add_channel("bad", f.src, bad);   // W/m^2 -> K
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-UNITS")) << d.to_string();
}

TEST(Validate, RejectsComponentCountMismatch) {
  Fixture f;
  const auto r = f.g.field(f.dst).region;
  const auto vec = f.g.add_field(
      f.solid, FieldSpec{.name = "v_in", .region = r, .components = 3,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Intensive,
                         .intent = Intent::Requires});
  f.g.add_channel("bad_shape", f.src, vec);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-SHAPE")) << d.to_string();
}

// A field with components <= 0 used to validate CLEAN. It then reached
// plan_buffers, where the count multiplies the region size into a
// std::size_t: components == -1 produced a total of 18446744073709551416 --
// a plausible-looking number, not a crash, because unsigned overflow is
// defined to wrap and so is invisible to both assert() and the sanitizer.
// The authoring gate is where this belongs.
TEST(Validate, RejectsNonPositiveComponentCount) {
  for (const int bad : {0, -1}) {
    Fixture f;
    const auto r = f.g.field(f.dst).region;
    f.g.add_field(f.solid,
                  FieldSpec{.name = "degenerate", .region = r,
                            .components = bad, .units = u::kHeatFluxDensity,
                            .conservation = Conservation::Extensive,
                            .intent = Intent::Requires});
    const Diagnostics d = validate(f.g);
    EXPECT_TRUE(has_code(d, "E-SHAPE"))
        << "components=" << bad << " validated clean: " << d.to_string();
  }
}

TEST(Validate, RejectsIntentViolation) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  // A Requires field used as a channel source.
  const auto wrong = f.g.add_field(
      f.fluid, FieldSpec{.name = "wrong", .region = r,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Intensive,
                         .intent = Intent::Requires});
  f.g.add_channel("bad_intent", wrong, f.dst);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-INTENT")) << d.to_string();
}

TEST(Validate, RejectsSelfCoupling) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  const auto other = f.g.add_field(
      f.fluid, FieldSpec{.name = "q2", .region = r,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Intensive,
                         .intent = Intent::Requires});
  f.g.add_channel("self", f.src, other);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-SELF")) << d.to_string();
}

TEST(Validate, RejectsZeroInterval) {
  Fixture f;
  f.g.add_component(ComponentDecl{.name = "broken", .interval = Rational{0}});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-INTERVAL")) << d.to_string();
}

// Spec Section 4.3: OASIS documents lag > coupling period as a deadlock hazard.
TEST(Validate, RejectsLagExceedingWindow) {
  Fixture f;
  f.g.channel(f.ch).lag(5);   // source interval 1, window 1 => 5 > 1
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LAG")) << d.to_string();
}

// `std::abs(ch.lag)` is UB for `ch.lag ==
// INT_MIN` (no positive int counterpart), and `ch.lag` is a raw int32_t
// off the wire with no range check (descriptor.hpp) -- a corrupted
// descriptor could set exactly this value. `ChannelBuilder::lag()` has
// no validation of its own either, so INT_MIN is reachable through the
// ordinary graph-building API too. validate()'s own contract is "every
// problem is reported together", never a throw; this must survive
// without UB and without throwing.
TEST(Validate, LagOfIntMinDoesNotTriggerUndefinedBehavior) {
  Fixture f;
  f.g.channel(f.ch).lag(std::numeric_limits<int>::min());
  Diagnostics d;
  EXPECT_NO_THROW(d = validate(f.g));
  EXPECT_TRUE(has_code(d, "E-LAG")) << d.to_string();
}

// The lag-span multiply must not be the throwing `*` operator, which can
// FUSE_REFUSE out of validate() on overflow -- a throw from a
// Diagnostics-returning function whose whole point is "report every
// problem together", and an abort in a FUSE_ENABLE_EXCEPTIONS=OFF
// build. `try_mul` turns the identical condition into a reported
// diagnostic instead.
TEST(Validate, LagSpanOverflowIsReportedNotThrown) {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:box"});
  const auto r = g.add_region(Region{.name = "wall", .mesh = m,
                                     .entity_set = "w", .dimension = 2});
  const auto fluid = g.add_component(ComponentDecl{.name = "fluid",
                                                    .ranks = fuse::RankSet{0, 1},
                                                    .interval = Rational{kMax}});
  const auto solid = g.add_component(ComponentDecl{.name = "solid",
                                                    .ranks = fuse::RankSet{1, 1},
                                                    .interval = Rational{1}});
  const auto src = g.add_field(fluid, FieldSpec{.name = "q", .region = r,
                                                .units = u::kHeatFluxDensity,
                                                .conservation = Conservation::Intensive,
                                                .intent = Intent::Provides});
  const auto dst = g.add_field(solid, FieldSpec{.name = "q_in", .region = r,
                                                .units = u::kHeatFluxDensity,
                                                .conservation = Conservation::Intensive,
                                                .intent = Intent::Requires});
  const auto ch = g.add_channel("flux", src, dst);
  g.channel(ch).lag(2);  // |lag| * interval = 2 * INT64_MAX overflows
  Diagnostics d;
  EXPECT_NO_THROW(d = validate(g));
  EXPECT_TRUE(has_code(d, "E-LAG")) << d.to_string();
}

TEST(Validate, WarnsOnUnconsumedProvidesField) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  f.g.add_field(f.fluid, FieldSpec{.name = "orphan", .region = r,
                                   .intent = Intent::Provides});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "W-NO-CONSUMER")) << d.to_string();
  EXPECT_FALSE(d.has_errors());  // warning only
}

// Was WarnsOnTwoWritersToOneTarget, at Severity::Warning. Promoted to an
// error, and the severity is the point: `lower()` binds one Sink per field to
// the LAST writer, so every earlier writer's chain dead-ends with no consumer
// and the dependence it declared vanishes from the schedule. A graph with a
// real algebraic loop through a shadowed writer then cleared this gate and
// was certified acyclic by prove_acyclic (232 such graphs in a 19,181-graph
// fuzz sweep). Spec Section 5.1's `Merge` is the feature that would make several
// writers legal; dropping all but one is not an implementation of it.
TEST(Validate, RejectsTwoWritersToOneTarget) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  const auto src2 = f.g.add_field(
      f.fluid, FieldSpec{.name = "q_alt", .region = r,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Intensive,
                         .intent = Intent::Provides});
  f.g.add_channel("second_writer", src2, f.dst);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-MULTI-WRITER")) << d.to_string();
  EXPECT_TRUE(d.has_errors()) << d.to_string();
  // The message must say what to do, and must not imply Merge already works.
  bool explains = false;
  for (const auto& item : d.items()) {
    if (item.code == "E-MULTI-WRITER" &&
        item.message.find("graph.merge(") != std::string::npos) {
      explains = true;  // the message must say what to do about it
    }
  }
  EXPECT_TRUE(explains) << d.to_string();
}

// ... and accepts the same graph once the user says how the two contributions
// combine. The rule is the user's to choose: summing two fluxes and averaging
// two temperatures are different physics.
TEST(Validate, AcceptsTwoWritersOnceAMergeIsDeclared) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  const auto src2 = f.g.add_field(
      f.fluid, FieldSpec{.name = "q_alt", .region = r,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Intensive,
                         .intent = Intent::Provides});
  f.g.add_channel("second_writer", src2, f.dst);
  f.g.merge(f.dst, fuse::MergeOp::Sum);

  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-MULTI-WRITER")) << d.to_string();
  EXPECT_FALSE(d.has_errors()) << d.to_string();
}

// A merge left behind after a channel was retargeted combines nothing, which
// is a mistake worth naming -- but not an error, since the graph still runs.
TEST(Validate, AMergeThatCombinesNothingIsAWarning) {
  Fixture f;
  f.g.merge(f.dst, fuse::MergeOp::Mean);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "W-MERGE-UNUSED")) << d.to_string();
  EXPECT_FALSE(d.has_errors()) << d.to_string();
}

TEST(Validate, RecordsTimeLevelMismatchAsWarning) {
  Fixture f;
  f.g.channel(f.ch).allow_time_level_mismatch("leapfrog by design");
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "W-MISMATCH-OK")) << d.to_string();
  EXPECT_FALSE(d.has_errors());
}

TEST(Validate, ReportsEveryProblemNotJustTheFirst) {
  Fixture f;
  f.g.add_component(ComponentDecl{.name = "fluid"});           // E-DUP-COMP
  f.g.add_component(ComponentDecl{.name = "z", .interval = Rational{0}});  // E-INTERVAL
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-DUP-COMP"));
  EXPECT_TRUE(has_code(d, "E-INTERVAL"));
}

// A FromComponent window naming a component that does not exist must not
// make channel_window()'s g.component(w.component) throw std::out_of_range,
// uncaught, straight out of validate() -- the one
// function whose contract is "accumulate every problem, never abort".
// Id::valid() only tests the 0xFFFFFFFF sentinel, not component_count(), so
// ComponentId{99} on a 2-component graph passed .valid() and reached .at().
TEST(Validate, DirectApiBadWindowComponentDoesNotThrowAndReportsBadRef) {
  Fixture f;
  // lag != 0 is what makes validate() actually evaluate channel_window();
  // without it that code path is never exercised and this test would pass
  // vacuously even against a broken implementation.
  f.g.channel(f.ch)
      .accumulate(AccumOp::Integral, Window::from(ComponentId{99}))
      .lag(1);
  Diagnostics d;
  EXPECT_NO_THROW(d = validate(f.g));
  EXPECT_TRUE(has_code(d, "E-BAD-REF")) << d.to_string();
}

// Same hazard, reached via the other path:
// deserialize() never validates embedded window component ids, so a
// hand-crafted (or corrupted) descriptor byte buffer can carry a
// FromComponent window pointing at a component index that was never
// written. The resulting graph must still validate without throwing.
TEST(Validate, DeserializedGraphWithBadWindowComponentDoesNotThrowAndReportsBadRef) {
  fuse::detail::Writer w;
  w.pod(fuse::kGraphMagic);
  w.pod(fuse::kGraphVersion);
  w.pod(std::uint32_t{1});  // meshes
  w.pod(std::uint32_t{1});  // regions
  w.pod(std::uint32_t{2});  // components
  w.pod(std::uint32_t{2});  // fields
  w.pod(std::uint32_t{1});  // channels
  w.pod(std::uint32_t{0});  // iteration regions
  w.pod(std::uint32_t{0});  // merges

  // mesh 0
  w.str(std::string{"m"});
  w.str(std::string{});
  // region 0 -> mesh 0
  w.str(std::string{"r"});
  w.pod(std::uint32_t{0});
  w.str(std::string{});
  w.pod(std::int32_t{-1});
  // component 0 = "A", on rank 0. Ranks are a COUNT then the list (v5), not
  // {first, count}: a component may be placed on strided or shared ranks.
  w.str(std::string{"A"});
  w.pod(std::uint32_t{1});
  w.pod(std::int32_t{0});
  w.rational(Rational{1});
  w.pod(std::uint8_t{0});  // kind = Physics (v6)
  w.pod(std::uint8_t{0});  // launch.mode = InProcess (v9)
  w.pod(std::int32_t{-1}); // launch.appnum
  w.pod(std::uint8_t{0});  // launch.is_acceptor
  w.str(std::string{});    // launch.port_file
  w.str(std::string{});    // launch.service_name
  w.str(std::string{});    // launch.plugin_path (v10)
  // component 1 = "B", on rank 1
  w.str(std::string{"B"});
  w.pod(std::uint32_t{1});
  w.pod(std::int32_t{1});
  w.rational(Rational{1});
  w.pod(std::uint8_t{0});  // kind = Physics (v6)
  w.pod(std::uint8_t{0});  // launch.mode = InProcess (v9)
  w.pod(std::int32_t{-1}); // launch.appnum
  w.pod(std::uint8_t{0});  // launch.is_acceptor
  w.str(std::string{});    // launch.port_file
  w.str(std::string{});    // launch.service_name
  w.str(std::string{});    // launch.plugin_path (v10)
  // field 0: owner=A, region=0, Provides
  w.str(std::string{"s"});
  w.pod(std::uint32_t{0});
  w.pod(std::uint32_t{0});
  w.pod(std::uint8_t{0});  // disc = FV
  w.pod(std::int32_t{0});
  w.pod(std::int32_t{1});
  w.units(u::kDimensionless);
  w.pod(std::uint8_t{1});  // conservation = Extensive
  w.pod(std::uint8_t{0});  // intent = Provides
  w.pod(std::uint8_t{0});  // hdf5_path presence = unset (v11)
  // field 1: owner=B, region=0, Requires
  w.str(std::string{"t"});
  w.pod(std::uint32_t{1});
  w.pod(std::uint32_t{0});
  w.pod(std::uint8_t{0});
  w.pod(std::int32_t{0});
  w.pod(std::int32_t{1});
  w.units(u::kDimensionless);
  w.pod(std::uint8_t{1});
  w.pod(std::uint8_t{1});  // intent = Requires
  w.pod(std::uint8_t{0});  // hdf5_path presence = unset (v11)
  // channel: field 0 -> field 1, accumulate window names nonexistent
  // component 99 -- this is the byte-level equivalent of the direct-API
  // bug above, and deserialize() does not reject it.
  w.str(std::string{"c"});
  w.pod(std::uint32_t{0});
  w.pod(std::uint32_t{1});
  w.pod(std::uint8_t{0});  // accumulate.op = Instant
  w.window(Window::from(ComponentId{99}));
  w.pod(std::uint32_t{0});  // remap.flags = None
  w.str(std::string{});     // remap.method_override
  w.pod(std::uint32_t{0xFFFFFFFFu});  // remap.remap_on = invalid (v6)
  w.pod(std::uint8_t{0});   // spacetime.basis = None
  w.pod(std::int32_t{0});   // spacetime.degree
  w.window(Window{});       // spacetime.window = CouplingPeriod default
  w.pod(std::uint8_t{0});   // relax.enabled
  w.pod(double{0.0});
  w.pod(std::uint8_t{0});   // extrapolate.enabled
  w.pod(std::int32_t{0});
  // lag != 0 is what makes validate() actually evaluate channel_window();
  // a zero lag would never exercise the pre-fix crash path.
  w.pod(std::int32_t{1});   // lag
  w.str(std::string{});     // time_level_mismatch_reason
  // No merge table: the count is in the header above, written as 0 there.

  const auto restored = deserialize(w.take());
  ASSERT_TRUE(restored.ok()) << restored.error().message;

  Diagnostics d;
  EXPECT_NO_THROW(d = validate(restored.value()));
  EXPECT_TRUE(has_code(d, "E-BAD-REF")) << d.to_string();
}

// ---- merge transform -------------------------------------

// Three outcomes where the two parents had one. Dimensions that differ are an
// incompatibility; an exact zero-offset scale is a conversion the compiler
// inserts and REPORTS; an offset change, or a scale under a policy that forbids
// insertion, is refused.
TEST(Validate, UnitGapHasThreeDistinctOutcomes) {
  const auto graph_with = [](fuse::Units source_units, fuse::Units target_units) {
    fuse::CouplingGraph g;
    const auto a = g.add_component({.name = "a", .ranks = {0, 1}, .interval = fuse::Rational{1}});
    const auto b = g.add_component({.name = "b", .ranks = {1, 1}, .interval = fuse::Rational{1}});
    const auto sf = g.add_field(a, {.name = "q", .units = source_units, .intent = fuse::Intent::Provides});
    const auto tf = g.add_field(b, {.name = "q", .units = target_units, .intent = fuse::Intent::Requires});
    (void)g.add_channel("q", sf, tf);
    return g;
  };
  const auto count = [](const fuse::Diagnostics& d, std::string_view code) {
    std::size_t n = 0;
    for (const auto& item : d.items()) n += (item.code == code);
    return n;
  };

  fuse::Units pascal{.length = -1, .mass = 1, .time = -2};
  fuse::Units kilopascal = pascal;
  kilopascal.scale = fuse::Rational{1000};
  fuse::Units kelvin{.temperature = 1};
  fuse::Units celsius = kelvin;
  celsius.offset = fuse::Rational{27315, 100};

  // Agreeing units: nothing to say.
  const auto same = fuse::validate(graph_with(pascal, pascal));
  EXPECT_EQ(count(same, "E-UNITS"), 0u);
  EXPECT_EQ(count(same, "E-UNITS-SCALE"), 0u);
  EXPECT_EQ(count(same, "I-UNIT-SCALE-INSERTED"), 0u);

  // Exact scale, default policy: inserted, and named. The factor is reported
  // because a user should see an operator they did not write.
  const auto scaled = fuse::validate(graph_with(pascal, kilopascal));
  EXPECT_EQ(count(scaled, "E-UNITS-SCALE"), 0u);
  EXPECT_EQ(count(scaled, "I-UNIT-SCALE-INSERTED"), 1u);
  bool factor_named = false;
  for (const auto& item : scaled.items())
    factor_named = factor_named || item.message.find("1/1000") != std::string::npos ||
                   item.message.find("1000") != std::string::npos;
  EXPECT_TRUE(factor_named);

  // Same gap, policy off: refused rather than silently applied.
  const auto forbidden = fuse::validate(graph_with(pascal, kilopascal), {.allow_automatic_unit_scale = false});
  EXPECT_EQ(count(forbidden, "E-UNITS-SCALE"), 1u);
  EXPECT_EQ(count(forbidden, "I-UNIT-SCALE-INSERTED"), 0u);

  // An OFFSET moves the zero of the scale, so it is refused whatever the policy.
  for (const fuse::ConnectPolicy policy : {fuse::ConnectPolicy{true}, fuse::ConnectPolicy{false}}) {
    const auto affine = fuse::validate(graph_with(kelvin, celsius), policy);
    EXPECT_EQ(count(affine, "E-UNITS-SCALE"), 1u);
    EXPECT_EQ(count(affine, "I-UNIT-SCALE-INSERTED"), 0u);
  }

  // Different dimensions remain an incompatibility, with no conversion offered.
  const auto wrong = fuse::validate(graph_with(pascal, kelvin));
  EXPECT_EQ(count(wrong, "E-UNITS"), 1u);
  EXPECT_EQ(count(wrong, "E-UNITS-SCALE"), 0u);
}

// ---- empty placement ------------------------------------------------------
//
// An empty placement is legal ONLY for a component with no channels. Both
// halves are tested, because the rule is a boundary and a check that refused
// every empty placement would pass a one-sided test while breaking the
// documented way to switch a component off for a run.

namespace {

/// One channel, `a.out -> b.in`, with each side's placement chosen by the
/// caller. `{0, 0}` is a component placed nowhere.
fuse::Diagnostics validate_with_placements(fuse::RankSet a_ranks,
                                           fuse::RankSet b_ranks,
                                           bool with_channel) {
  fuse::CouplingGraph g;
  const fuse::MeshId m = g.add_mesh(fuse::MeshDecl{.name = "m", .source = "test"});
  const fuse::RegionId r =
      g.add_region(fuse::Region{.name = "r", .mesh = m, .dimension = 2});
  const fuse::ComponentId a = g.add_component(fuse::ComponentDecl{
      .name = "a", .ranks = a_ranks, .interval = fuse::Rational{1}});
  const fuse::ComponentId b = g.add_component(fuse::ComponentDecl{
      .name = "b", .ranks = b_ranks, .interval = fuse::Rational{1}});
  const fuse::FieldId af = g.add_field(
      a, fuse::FieldSpec{.name = "out", .region = r, .units = u::kKelvin,
                         .intent = fuse::Intent::Provides});
  const fuse::FieldId bf = g.add_field(
      b, fuse::FieldSpec{.name = "in", .region = r, .units = u::kKelvin,
                         .intent = fuse::Intent::Requires});
  if (with_channel) g.add_channel("link", af, bf);
  return fuse::validate(g);
}

std::size_t count_code(const fuse::Diagnostics& d, std::string_view code) {
  std::size_t n = 0;
  for (const auto& item : d.items()) {
    if (item.code == code && item.severity == fuse::Severity::Error) ++n;
  }
  return n;
}

}  // namespace

TEST(Validate, AChannelOutOfAComponentPlacedNowhereIsRefused) {
  // The source exists on no rank: nothing to read from.
  const auto d = validate_with_placements({0, 0}, {0, 2}, true);
  EXPECT_EQ(count_code(d, "E-PLACEMENT-EMPTY"), 1u)
      << "a channel cannot read from a component that exists on no rank";
}

TEST(Validate, AChannelIntoAComponentPlacedNowhereIsRefused) {
  // The target exists on no rank: nowhere to deliver. Checked separately
  // because a rule written for one side only would pass the other test.
  const auto d = validate_with_placements({0, 2}, {0, 0}, true);
  EXPECT_EQ(count_code(d, "E-PLACEMENT-EMPTY"), 1u)
      << "a channel cannot deliver to a component that exists on no rank";
}

TEST(Validate, BothEndsPlacedNowhereIsReportedOnceForEachEnd) {
  const auto d = validate_with_placements({0, 0}, {0, 0}, true);
  EXPECT_EQ(count_code(d, "E-PLACEMENT-EMPTY"), 2u)
      << "each end is a separate thing for the user to fix";
}

TEST(Validate, AnEmptyPlacementWithNoChannelIsStillLegal) {
  // The other half of the ruling, and the reason the check is per channel
  // rather than per component: switching a component off for one run by
  // giving it an empty placement remains supported (docs/user/placement.md).
  const auto d = validate_with_placements({0, 0}, {0, 2}, false);
  EXPECT_EQ(count_code(d, "E-PLACEMENT-EMPTY"), 0u)
      << "an empty placement is how a component is switched off for a run; "
         "with no channel there is nothing to refuse";
}

TEST(Validate, OrdinaryPlacementsAreNotRefused) {
  const auto d = validate_with_placements({0, 2}, {2, 2}, true);
  EXPECT_EQ(count_code(d, "E-PLACEMENT-EMPTY"), 0u);
}

// ---- coupler placement (design C) --------------------------------------

TEST(Validate, ACouplerOwningFieldsIsRefused) {
  Fixture f;
  const auto coupler = f.g.add_coupler({.name = "C", .ranks = {2, 1}});
  // Nothing in the public API stops a caller from add_field()-ing onto a
  // coupler's ComponentId; validate() is where the misuse is caught.
  (void)f.g.add_field(coupler, FieldSpec{.name = "stray", .region =
                                         f.g.field(f.src).region});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-COUPLER-FIELDS")) << d.to_string();
}

TEST(Validate, ACouplerWithNoFieldsIsClean) {
  Fixture f;
  f.g.add_coupler({.name = "C", .ranks = {2, 1}});
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(d.has_errors()) << d.to_string();
}

TEST(Validate, RemapOnMustNameACouplerOrTheChannelsOwnEndpoint) {
  Fixture f;
  // A THIRD physics component, unrelated to this channel: neither a
  // coupler, nor this channel's own source or target.
  const auto bystander = f.g.add_component(
      ComponentDecl{.name = "bystander", .ranks = {2, 1}});
  f.g.channel(f.ch).remap(fuse::RemapFlags::Conservative).remap_on(bystander);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-CHANNEL-REMAP-ON")) << d.to_string();
}

TEST(Validate, RemapOnACouplerIsAccepted) {
  Fixture f;
  const auto coupler = f.g.add_coupler({.name = "C", .ranks = {2, 1}});
  f.g.channel(f.ch).remap(fuse::RemapFlags::Conservative).remap_on(coupler);
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(d.has_errors()) << d.to_string();
  EXPECT_FALSE(has_code(d, "W-CHANNEL-REMAP-ON-UNUSED")) << d.to_string();
}

TEST(Validate, RemapOnTheChannelsOwnSourceOrTargetIsAccepted) {
  Fixture f;
  f.g.channel(f.ch).remap(fuse::RemapFlags::Conservative).remap_on(f.fluid);
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-CHANNEL-REMAP-ON")) << d.to_string();
  f.g.channel(f.ch).remap_on(f.solid);
  const Diagnostics d2 = validate(f.g);
  EXPECT_FALSE(has_code(d2, "E-CHANNEL-REMAP-ON")) << d2.to_string();
}

// Fixture's "flux" channel has no remap flags and both fields share the
// SAME region (`r`), so nothing about it can ever produce a Remap operator
// (fuse/model/remap_kind.hpp's could_ever_emit_remap): a remap_on declared
// on it is inert.
TEST(Validate, RemapOnWithNoPossibleRemapWarnsUnused) {
  Fixture f;
  const auto coupler = f.g.add_coupler({.name = "C", .ranks = {2, 1}});
  f.g.channel(f.ch).remap_on(coupler);
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(d.has_errors()) << d.to_string();
  EXPECT_TRUE(has_code(d, "W-CHANNEL-REMAP-ON-UNUSED")) << d.to_string();
}

TEST(Validate, RemapOnACouplerPlacedNowhereIsRefusedWhenItWouldBeUsed) {
  Fixture f;
  const auto coupler = f.g.add_coupler({.name = "C", .ranks = fuse::RankSet(0, 0)});
  f.g.channel(f.ch).remap(fuse::RemapFlags::Conservative).remap_on(coupler);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-PLACEMENT-EMPTY")) << d.to_string();
}

// The unused case is the other half of the same ruling as
// AnEmptyPlacementWithNoChannelIsStillLegal above: an empty coupler
// placement that no channel's remap actually routes through is not a
// modelling error.
TEST(Validate, RemapOnACouplerPlacedNowhereIsNotRefusedWhenItWouldNotBeUsed) {
  Fixture f;
  const auto coupler = f.g.add_coupler({.name = "C", .ranks = fuse::RankSet(0, 0)});
  f.g.channel(f.ch).remap_on(coupler);  // no flags -> could never fire
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-PLACEMENT-EMPTY")) << d.to_string();
  EXPECT_TRUE(has_code(d, "W-CHANNEL-REMAP-ON-UNUSED")) << d.to_string();
}

// ---- hdf5_path ------------------------------------------------------------

TEST(Validate, FieldWithHdf5PathButNoLowFiveComponentIsRefused) {
  Fixture f;
  const_cast<FieldSpec&>(f.g.field(f.src)).hdf5_path = "producer_out.h5/grid/q";
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LOWFIVE-NO-COMPONENT")) << d.to_string();
}

TEST(Validate, FieldWithHdf5PathAndALowFiveComponentIsAccepted) {
  Fixture f;
  const_cast<FieldSpec&>(f.g.field(f.src)).hdf5_path = "producer_out.h5/grid/q";
  f.g.add_component(ComponentDecl{
      .name = "interceptor", .ranks = fuse::RankSet{2, 1},
      .interval = Rational{1}, .kind = ComponentKind::Coupler,
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::LowFive}});
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-LOWFIVE-NO-COMPONENT")) << d.to_string();
  EXPECT_FALSE(d.has_errors()) << d.to_string();
}

// A LaunchMode::LowFive component is a placement, the same role an ordinary coupler
// already plays -- it must declare kind=Coupler and own no fields, or
// fuse_initialize()'s own wiring loop (src/abi/participant.cpp) has no
// exemption for it and it becomes a spurious externally-driven
// component with no step call anywhere in a caller's own loop.
TEST(Validate, ALowFiveComponentDeclaringPhysicsKindIsRefused) {
  Fixture f;
  const_cast<FieldSpec&>(f.g.field(f.src)).hdf5_path = "producer_out.h5/grid/q";
  // kind left at its default (Physics) -- the exact mistake this check
  // exists to catch.
  f.g.add_component(ComponentDecl{
      .name = "interceptor", .ranks = fuse::RankSet{2, 1},
      .interval = Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::LowFive}});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LOWFIVE-COMPONENT-KIND")) << d.to_string();
}

TEST(Validate, ALowFiveComponentOwningFieldsIsRefused) {
  Fixture f;
  const_cast<FieldSpec&>(f.g.field(f.src)).hdf5_path = "producer_out.h5/grid/q";
  const auto interceptor = f.g.add_component(ComponentDecl{
      .name = "interceptor", .ranks = fuse::RankSet{2, 1},
      .interval = Rational{1}, .kind = ComponentKind::Coupler,
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::LowFive}});
  // Nothing in the public API stops a caller from add_field()-ing onto a
  // Coupler's ComponentId (ACouplerOwningFieldsIsRefused, above, proves
  // the SAME gap for an ordinary coupler placement) -- declaring
  // kind=Coupler correctly is not, by itself, enough.
  (void)f.g.add_field(interceptor,
                      FieldSpec{.name = "stray", .region = f.g.field(f.src).region});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LOWFIVE-COMPONENT-KIND")) << d.to_string();
}

TEST(Validate, TwoFieldsSharingAnHdf5PathAreRefused) {
  Fixture f;
  f.g.add_component(ComponentDecl{
      .name = "interceptor", .ranks = fuse::RankSet{2, 1},
      .interval = Rational{1}, .kind = ComponentKind::Coupler,
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::LowFive}});
  const_cast<FieldSpec&>(f.g.field(f.src)).hdf5_path = "shared.h5/grid/q";
  const_cast<FieldSpec&>(f.g.field(f.dst)).hdf5_path = "shared.h5/grid/q";
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LOWFIVE-DUPPATH")) << d.to_string();
}

// The wildcard exclusion mirrors LowFiveFieldBindingTable::add's own rule
// (fuse/transport/lowfive.hpp): two wildcard bindings on different real
// files are the ordinary multi-field case, and this layer has no more way
// to know the real file than that one does.
TEST(Validate, WildcardHdf5PathsAreNotCheckedForCollision) {
  Fixture f;
  f.g.add_component(ComponentDecl{
      .name = "interceptor", .ranks = fuse::RankSet{2, 1},
      .interval = Rational{1}, .kind = ComponentKind::Coupler,
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::LowFive}});
  const_cast<FieldSpec&>(f.g.field(f.src)).hdf5_path = "*.h5/grid/q";
  const_cast<FieldSpec&>(f.g.field(f.dst)).hdf5_path = "*.h5/grid/q";
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-LOWFIVE-DUPPATH")) << d.to_string();
}

namespace {

// A two-member (alpha/beta) window-scoped graph on the REAL Brusselator
// example's own dt_u/dt_v/window (examples/brusselator/participants/
// brusselator_graph_tool.cpp): dt_u=1/500, dt_v=3/1000, window=1/50. beta's own back channel
// (b_out -> a_in) is lagged -- required for GaussSeidel (a two-way
// window with no lagged transfer at all is refused, WaveformIteration's
// own comment, fuse/model/iteration.hpp) -- `extra_back_lag` lets a
// caller push the lag past the window on purpose, for the companion
// refusal test below.
[[nodiscard]] CouplingGraph build_real_parameter_window_graph(
    fuse::WaveformIteration wi, int back_lag = 1) {
  using fuse::ChannelId;
  using fuse::ConvergenceCriterion;
  using fuse::CouplingWindowScope;
  using fuse::DtPolicy;
  using fuse::IterationRegionDecl;
  CouplingGraph g;
  const auto ma = g.add_mesh(MeshDecl{.name = "ma", .source = "generated:line:4"});
  const auto mb = g.add_mesh(MeshDecl{.name = "mb", .source = "generated:line:4"});
  const auto ra = g.add_region(Region{.name = "ra", .mesh = ma, .dimension = 1});
  const auto rb = g.add_region(Region{.name = "rb", .mesh = mb, .dimension = 1});
  const auto a = g.add_component(ComponentDecl{
      .name = "alpha", .ranks = fuse::RankSet{0, 1}, .interval = Rational{2, 1000}});
  const auto b = g.add_component(ComponentDecl{
      .name = "beta", .ranks = fuse::RankSet{0, 1}, .interval = Rational{3, 1000}});
  const auto a_out = g.add_field(
      a, FieldSpec{.name = "a_out", .region = ra, .components = 1,
                  .units = u::kKelvin, .intent = Intent::Provides});
  const auto b_in = g.add_field(
      b, FieldSpec{.name = "b_in", .region = rb, .components = 1,
                  .units = u::kKelvin, .intent = Intent::Requires});
  const auto b_out = g.add_field(
      b, FieldSpec{.name = "b_out", .region = rb, .components = 1,
                  .units = u::kHeatFluxDensity, .intent = Intent::Provides});
  const auto a_in = g.add_field(
      a, FieldSpec{.name = "a_in", .region = ra, .components = 1,
                  .units = u::kHeatFluxDensity, .intent = Intent::Requires});
  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const ChannelId back = g.add_channel("b2a", b_out, a_in);
  if (wi == fuse::WaveformIteration::GaussSeidel || back_lag != 1) {
    g.channel(back).lag(back_lag).allow_time_level_mismatch(
        "resolved by the window's own sweep, not a real-time lag");
  }
  CouplingWindowScope scope;
  scope.explicit_interval = Rational{2, 100};  // 1/50
  scope.order = 2;
  scope.policies = {{"alpha", DtPolicy::Adjustable}, {"beta", DtPolicy::Adjustable}};
  scope.iteration = wi;
  ConvergenceCriterion convergence;
  convergence.add(fuse::Converged::Relative(1e-9, "a_out"));
  convergence.add(fuse::Converged::Relative(1e-9, "a_in"));
  g.add_iteration_region(IterationRegionDecl{
      .name = "window", .channels = {fwd, back}, .max_iterations = 60,
      .convergence = convergence, .window_scope = scope});
  return g;
}

}  // namespace

// The real Brusselator parameters (dt_u=1/500, dt_v=3/1000, window=1/50) under
// GaussSeidel -- beta's own back channel lagged by 1*dt_v=3/1000, well
// under the WINDOW's own 1/50, but OVER the target component's own
// interval (alpha's 1/500) that channel_window()'s stale fallback used
// to measure it against. Must validate clean -- no E-LAG, and no OTHER
// error.
TEST(Validate, GaussSeidelWindowLagWithinTheRealWindowValidates) {
  const CouplingGraph g =
      build_real_parameter_window_graph(fuse::WaveformIteration::GaussSeidel);
  const Diagnostics d = validate(g);
  EXPECT_FALSE(has_code(d, "E-LAG")) << d.to_string();
  EXPECT_FALSE(d.has_errors()) << d.to_string();
}

// The companion refusal: a lag genuinely longer than the WINDOW itself
// (not merely longer than some unrelated component's own interval) must
// still be refused -- channel_window()'s fix measures against the
// window-scoped region's own explicit_interval, not against "no check at
// all". back_lag=20 at dt_v=3/1000 spans 20*3/1000=3/50, over the
// window's own 1/50.
TEST(Validate, RejectsLagExceedingTheWindowItself) {
  const CouplingGraph g = build_real_parameter_window_graph(
      fuse::WaveformIteration::GaussSeidel, /*back_lag=*/20);
  const Diagnostics d = validate(g);
  EXPECT_TRUE(has_code(d, "E-LAG")) << d.to_string();
}

// ---------------------------------------------------------------------------
// The Conservation stage reaches validate(). `connect.hpp`'s
// `check_channel()` resolves the declared `TransferKind` rather than
// hardcoding `TransferKind::Copy`, and validate() maps every stage's
// diagnostics, not only Unit's. These three pin that behavior directly: a plain copy into an Extensive field is refused
// with a message naming the field and guarantee; a channel that actually
// declares a conservative transfer is accepted; and a channel that ALSO has
// a scale-only unit mismatch reports both problems rather than losing the
// unit conversion behind the conservation refusal (compatibility_test.cpp's
// `Compatibility.ConservationRules` pins the same rule one layer down, at
// `check_connection()` directly).
// ---------------------------------------------------------------------------

namespace {

/// `fluid`/`solid`, each owning one Conservation::Extensive field over the
/// same region (`r1`) or two different ones (`r1`/`r2`), connected by one
/// channel the caller can still perturb (declare conservative, change
/// units, ...) before calling validate().
struct ExtensiveFixture {
  CouplingGraph g;
  fuse::FieldId src, dst;
  fuse::ChannelId ch;

  explicit ExtensiveFixture(bool same_region) {
    const auto m1 = g.add_mesh(MeshDecl{.name = "m1", .source = "generated:box"});
    const auto r1 = g.add_region(Region{.name = "r1", .mesh = m1,
                                        .entity_set = "w", .dimension = 2});
    const auto r2 = same_region ? r1
                                : g.add_region(Region{.name = "r2", .mesh = m1,
                                                      .entity_set = "w2", .dimension = 2});
    const auto fluid = g.add_component(ComponentDecl{.name = "fluid",
                                                      .ranks = fuse::RankSet{0, 1},
                                                      .interval = Rational{1}});
    const auto solid = g.add_component(ComponentDecl{.name = "solid",
                                                      .ranks = fuse::RankSet{1, 1},
                                                      .interval = Rational{1}});
    src = g.add_field(fluid, FieldSpec{.name = "q", .region = r1,
                                       .units = u::kHeatFluxDensity,
                                       .conservation = Conservation::Extensive,
                                       .intent = Intent::Provides});
    dst = g.add_field(solid, FieldSpec{.name = "q_in", .region = r2,
                                       .units = u::kHeatFluxDensity,
                                       .conservation = Conservation::Extensive,
                                       .intent = Intent::Requires});
    ch = g.add_channel("flux", src, dst);
  }
};

}  // namespace

TEST(Validate, ExtensiveTargetOverAPlainCopyIsRefused) {
  ExtensiveFixture f(/*same_region=*/true);  // no remap declared: a plain copy
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-CONN-CONSERVATION")) << d.to_string();
  bool names_field_and_guarantee = false;
  for (const auto& item : d.items()) {
    if (item.code != "E-CONN-CONSERVATION") continue;
    names_field_and_guarantee = item.message.find("q_in") != std::string::npos &&
                                item.message.find("global") != std::string::npos;
  }
  EXPECT_TRUE(names_field_and_guarantee)
      << "message must name the refused field and the required guarantee: "
      << d.to_string();
}

TEST(Validate, ExtensiveTargetOverAConservativeRemapIsAccepted) {
  ExtensiveFixture f(/*same_region=*/false);
  f.g.channel(f.ch).conservative();
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(has_code(d, "E-CONN-CONSERVATION")) << d.to_string();
  EXPECT_FALSE(d.has_errors()) << d.to_string();
  // `validate()`'s two assertions above would also pass if conservation
  // refusals never reached validate() at all, for the WRONG reason. Asking
  // `check_channel()` directly is what actually discriminates: a hardcoded
  // Copy would report `Unsupported` for this exact graph (ignoring
  // `.conservative()`), whereas `declared_transfer()` resolves the
  // channel's `RemapFlags::Conservative` and the connection compiles.
  const auto connection = check_channel(f.g, f.ch);
  EXPECT_TRUE(connection.compiles())
      << "a channel that actually declares a conservative transfer must "
         "compile through check_channel() -- not merely pass unrelated to "
         "it, which is what the two checks above alone would permit";
}

TEST(Validate,
    ExtensiveTargetWithScaleOnlyMismatchReportsBothConservationAndUnitScale) {
  ExtensiveFixture f(/*same_region=*/true);  // plain copy: Conservation refuses
  // Same dimensions as the fixture's own kHeatFluxDensity, differing only by
  // an exact, zero-offset scale -- the kPa-vs-Pa shape,
  // reproduced here in the fixture's own unit dimension rather than
  // pulling in a second one.
  fuse::Units base_units = u::kHeatFluxDensity;
  fuse::Units scaled_units = base_units;
  scaled_units.scale = Rational{1000};
  const_cast<FieldSpec&>(f.g.field(f.src)).units = scaled_units;
  const_cast<FieldSpec&>(f.g.field(f.dst)).units = base_units;
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-CONN-CONSERVATION")) << d.to_string();
  EXPECT_TRUE(has_code(d, "I-UNIT-SCALE-INSERTED"))
      << "the scale-only conversion must still be reported, not discarded "
         "behind the conservation refusal: "
      << d.to_string();
  EXPECT_FALSE(has_code(d, "E-UNITS")) << d.to_string();
  EXPECT_FALSE(has_code(d, "E-UNITS-SCALE")) << d.to_string();
}
