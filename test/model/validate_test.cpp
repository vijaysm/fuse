#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/descriptor.hpp>
#include <fuse/model/validate.hpp>

#include <algorithm>
#include <string_view>

using fuse::AccumOp;
using fuse::ComponentDecl;
using fuse::ComponentId;
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
                                          .ranks = {.first = 0, .count = 1},
                                          .interval = Rational{1}});
    solid = g.add_component(ComponentDecl{.name = "solid",
                                          .ranks = {.first = 1, .count = 1},
                                          .interval = Rational{1}});
    src = g.add_field(fluid, FieldSpec{.name = "q", .region = r,
                                       .units = u::kHeatFluxDensity,
                                       .conservation = Conservation::Extensive,
                                       .intent = Intent::Provides});
    dst = g.add_field(solid, FieldSpec{.name = "q_in", .region = r,
                                       .units = u::kHeatFluxDensity,
                                       .conservation = Conservation::Extensive,
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

TEST(Validate, RejectsUnitMismatch) {
  Fixture f;
  const auto r = f.g.field(f.dst).region;
  const auto bad = f.g.add_field(
      f.solid, FieldSpec{.name = "T_in", .region = r, .units = u::kKelvin,
                         .conservation = Conservation::Extensive,
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
                         .conservation = Conservation::Extensive,
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
                         .conservation = Conservation::Extensive,
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
                         .conservation = Conservation::Extensive,
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

// Spec §4.3: OASIS documents lag > coupling period as a deadlock hazard.
TEST(Validate, RejectsLagExceedingWindow) {
  Fixture f;
  f.g.channel(f.ch).lag(5);   // source interval 1, window 1 => 5 > 1
  const Diagnostics d = validate(f.g);
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
// fuzz sweep). Spec §5.1's `Merge` is the feature that would make several
// writers legal; dropping all but one is not an implementation of it.
TEST(Validate, RejectsTwoWritersToOneTarget) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  const auto src2 = f.g.add_field(
      f.fluid, FieldSpec{.name = "q_alt", .region = r,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Extensive,
                         .intent = Intent::Provides});
  f.g.add_channel("second_writer", src2, f.dst);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-MULTI-WRITER")) << d.to_string();
  EXPECT_TRUE(d.has_errors()) << d.to_string();
  // The message must say what to do, and must not imply Merge already works.
  bool explains = false;
  for (const auto& item : d.items()) {
    if (item.code == "E-MULTI-WRITER" &&
        item.message.find("not implemented yet") != std::string::npos) {
      explains = true;
    }
  }
  EXPECT_TRUE(explains) << d.to_string();
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

// CRITICAL 1 (post-review fix): a FromComponent window naming a component
// that does not exist used to make channel_window()'s g.component(w.component)
// throw std::out_of_range, uncaught, straight out of validate() -- the one
// function whose contract is "accumulate every problem, never abort".
// Id::valid() only tests the 0xFFFFFFFF sentinel, not component_count(), so
// ComponentId{99} on a 2-component graph passed .valid() and reached .at().
TEST(Validate, DirectApiBadWindowComponentDoesNotThrowAndReportsBadRef) {
  Fixture f;
  // lag != 0 is what makes validate() actually evaluate channel_window();
  // without it the pre-fix code path is never exercised and this test would
  // pass vacuously even against the broken implementation.
  f.g.channel(f.ch)
      .accumulate(AccumOp::Integral, Window::from(ComponentId{99}))
      .lag(1);
  Diagnostics d;
  EXPECT_NO_THROW(d = validate(f.g));
  EXPECT_TRUE(has_code(d, "E-BAD-REF")) << d.to_string();
}

// Same defect, reached via the other path the review called out:
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

  // mesh 0
  w.str(std::string{"m"});
  w.str(std::string{});
  // region 0 -> mesh 0
  w.str(std::string{"r"});
  w.pod(std::uint32_t{0});
  w.str(std::string{});
  w.pod(std::int32_t{-1});
  // component 0 = "A"
  w.str(std::string{"A"});
  w.pod(std::int32_t{0});
  w.pod(std::int32_t{1});
  w.rational(Rational{1});
  // component 1 = "B"
  w.str(std::string{"B"});
  w.pod(std::int32_t{1});
  w.pod(std::int32_t{1});
  w.rational(Rational{1});
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

  const auto restored = deserialize(w.take());
  ASSERT_TRUE(restored.ok()) << restored.error().message;

  Diagnostics d;
  EXPECT_NO_THROW(d = validate(restored.value()));
  EXPECT_TRUE(has_code(d, "E-BAD-REF")) << d.to_string();
}

// ---- merge transform (S2.1) --------------------------------------------

// Dimensions and the affine part are separate verdicts with separate codes: one
// is an incompatibility, the other a conversion not yet inferred.
TEST(Validate, DimensionMismatchAndScaleMismatchAreDistinctVerdicts) {
  const auto graph_with = [](fuse::Units source_units, fuse::Units target_units) {
    fuse::CouplingGraph g;
    const auto a = g.add_component({.name = "a", .ranks = {0, 1}, .interval = fuse::Rational{1}});
    const auto b = g.add_component({.name = "b", .ranks = {1, 1}, .interval = fuse::Rational{1}});
    const auto sf = g.add_field(a, {.name = "q", .units = source_units, .intent = fuse::Intent::Provides});
    const auto tf = g.add_field(b, {.name = "q", .units = target_units, .intent = fuse::Intent::Requires});
    (void)g.add_channel("q", sf, tf);
    return g;
  };
  const auto codes = [](const fuse::Diagnostics& d) {
    std::vector<std::string> out;
    for (const auto& item : d.items()) out.push_back(item.code);
    return out;
  };

  fuse::Units pascal{.mass = 1, .length = -1, .time = -2};
  fuse::Units kilopascal = pascal;
  kilopascal.scale = fuse::Rational{1000};

  const auto same = codes(fuse::validate(graph_with(pascal, pascal)));
  EXPECT_EQ(std::count(same.begin(), same.end(), "E-UNITS"), 0);
  EXPECT_EQ(std::count(same.begin(), same.end(), "E-UNITS-SCALE"), 0);

  const auto scaled = codes(fuse::validate(graph_with(pascal, kilopascal)));
  EXPECT_EQ(std::count(scaled.begin(), scaled.end(), "E-UNITS-SCALE"), 1);
  EXPECT_EQ(std::count(scaled.begin(), scaled.end(), "E-UNITS"), 0);

  const auto wrong = codes(fuse::validate(graph_with(pascal, fuse::Units{.temperature = 1})));
  EXPECT_EQ(std::count(wrong.begin(), wrong.end(), "E-UNITS"), 1);
  EXPECT_EQ(std::count(wrong.begin(), wrong.end(), "E-UNITS-SCALE"), 0);
}
