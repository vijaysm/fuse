#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/descriptor.hpp>
#include <fuse/model/yaml_io.hpp>
#include <fuse/model/freeze_fixture.hpp>

using fuse::AccumOp;
using fuse::ComponentDecl;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::FieldSpec;
using fuse::from_yaml;
using fuse::Intent;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
using fuse::RemapFlags;
using fuse::TemporalBasis;
using fuse::to_yaml;
using fuse::Window;
namespace u = fuse::units;

namespace {

// Deliberately exercises non-default accumulate.window, spacetime.window,
// and remap.method_override -- CRITICAL 2 from the post-1.7 review found
// that to_yaml() silently dropped all three (it never emitted them at all,
// and from_yaml() hardcoded Window::coupling_period() on both window
// paths), and RoundTripsThroughText passed anyway because the pre-fix
// pair_graph() only used defaults. This fixture would have caught it.
CouplingGraph pair_graph() {
  CouplingGraph g;
  const auto mf = g.add_mesh(MeshDecl{.name = "fluid_mesh",
                                      .source = "generated:box:8x8x8"});
  const auto ms = g.add_mesh(MeshDecl{.name = "solid_mesh",
                                      .source = "generated:box:5x5x5"});
  const auto rf = g.add_region(Region{.name = "fluid_wall", .mesh = mf,
                                      .entity_set = "wall", .dimension = 2});
  const auto rs = g.add_region(Region{.name = "solid_wall", .mesh = ms,
                                      .entity_set = "wall", .dimension = 2});
  const auto fluid = g.add_component(ComponentDecl{
      .name = "fluid", .ranks = fuse::RankSet{0, 2},
      .interval = Rational{1, 2}});
  const auto solid = g.add_component(ComponentDecl{
      .name = "solid", .ranks = fuse::RankSet{2, 2},
      .interval = Rational{1}});
  const auto s = g.add_field(fluid, FieldSpec{
      .name = "heat_flux", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  const auto t = g.add_field(solid, FieldSpec{
      .name = "q_wall", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Requires});
  const auto ch = g.add_channel("flux", s, t);
  g.channel(ch)
      .accumulate(AccumOp::Mean, Window::from(fluid))
      .remap(RemapFlags::Conservative | RemapFlags::Monotone)
      .remap_method("mono3")
      .spacetime(TemporalBasis::DG, 2, Window::explicit_length(Rational{2, 5}))
      .relax(0.7);
  return g;
}

}  // namespace

TEST(YamlIo, RoundTripsThroughText) {
  const CouplingGraph original = pair_graph();
  const auto text = to_yaml(original);
  ASSERT_TRUE(text.ok()) << text.error().message;
  const auto restored = from_yaml(text.value());
  ASSERT_TRUE(restored.ok()) << restored.error().message;
  EXPECT_TRUE(structurally_equal(original, restored.value()));
}

TEST(YamlIo, EmittedTextContainsReadableNames) {
  const auto text = to_yaml(pair_graph()).value();
  EXPECT_NE(text.find("fluid_mesh"), std::string::npos);
  EXPECT_NE(text.find("fluid.heat_flux"), std::string::npos);
  EXPECT_NE(text.find("1/2"), std::string::npos);
}

TEST(YamlIo, ReadsTheCheckedInFixture) {
  const auto g = fuse::load_yaml_file(FUSE_TEST_DATA_DIR "/pair.yaml");
  ASSERT_TRUE(g.ok()) << g.error().message;
  EXPECT_EQ(g.value().component_count(), 2u);
  EXPECT_EQ(g.value().channel_count(), 1u);
  EXPECT_TRUE(structurally_equal(pair_graph(), g.value()));
}

// REVIEW.md R-18: to_yaml()'s own signature promises Result<std::string> for
// a reportable failure, but it dereferences several foreign ids (a channel's
// source/target field, an iteration region's channel list, a merge entry's
// field) through CouplingGraph accessors that are plain std::vector::at()
// calls -- and the raw API (add_channel, add_iteration_region, merge) admits
// any id with no bounds check at construction time. Nothing requires
// validate() to run before to_yaml(); this builds a graph via the raw API
// with an out-of-range ChannelId in an iteration region, bypassing
// validate() entirely, and checks to_yaml() returns a Failure rather than
// letting std::out_of_range escape uncaught.
TEST(YamlIo, EmittingAGraphWithADanglingChannelReferenceReturnsFailure) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:box"});
  const auto r = g.add_region(Region{.name = "wall", .mesh = m,
                                     .entity_set = "w", .dimension = 2});
  const auto a = g.add_component(ComponentDecl{.name = "a",
                                                .ranks = fuse::RankSet{0, 1},
                                                .interval = Rational{1}});
  (void)g.add_field(a, FieldSpec{.name = "f", .region = r,
                                 .units = u::kHeatFluxDensity,
                                 .conservation = Conservation::Extensive,
                                 .intent = Intent::Provides});
  // No channels exist, so ChannelId{99} is out of range -- the raw API
  // (iterate(), a thin wrapper over add_iteration_region()) accepts it
  // without complaint.
  g.iterate({fuse::ChannelId{99}}).name("outer").max_iterations(10);

  const auto text = to_yaml(g);
  ASSERT_FALSE(text.ok());
  EXPECT_EQ(text.error().code, "E-YAML-REF");
}

// N-base-model-1 (docs/dev/reviews/2026-09-19/pass2/re-base-model.md): a
// FromComponent window's ComponentId is accepted by ChannelBuilder::
// accumulate()/spacetime() with no bounds check, the same reachability
// R-18 already covers for a channel's source/target field and an
// iteration region's channel list -- but neither window field was
// guarded. Mirrors EmittingAGraphWithADanglingChannelReferenceReturnsFailure
// above, one call site per window.
TEST(YamlIo, EmittingAGraphWithADanglingAccumulateWindowComponentReturnsFailure) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:box"});
  const auto r = g.add_region(Region{.name = "wall", .mesh = m,
                                     .entity_set = "w", .dimension = 2});
  const auto a = g.add_component(ComponentDecl{.name = "a",
                                                .ranks = fuse::RankSet{0, 1},
                                                .interval = Rational{1}});
  const auto s = g.add_field(a, FieldSpec{.name = "s", .region = r,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Provides});
  const auto t = g.add_field(a, FieldSpec{.name = "t", .region = r,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  // Only component "a" exists, so ComponentId{99} is out of range -- the
  // raw builder accepts it without complaint.
  g.channel(ch).accumulate(AccumOp::Mean, Window::from(fuse::ComponentId{99}));

  const auto text = to_yaml(g);
  ASSERT_FALSE(text.ok());
  EXPECT_EQ(text.error().code, "E-YAML-REF");
}

TEST(YamlIo, EmittingAGraphWithADanglingSpacetimeWindowComponentReturnsFailure) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:box"});
  const auto r = g.add_region(Region{.name = "wall", .mesh = m,
                                     .entity_set = "w", .dimension = 2});
  const auto a = g.add_component(ComponentDecl{.name = "a",
                                                .ranks = fuse::RankSet{0, 1},
                                                .interval = Rational{1}});
  const auto s = g.add_field(a, FieldSpec{.name = "s", .region = r,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Provides});
  const auto t = g.add_field(a, FieldSpec{.name = "t", .region = r,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).spacetime(TemporalBasis::DG, 2, Window::from(fuse::ComponentId{99}));

  const auto text = to_yaml(g);
  ASSERT_FALSE(text.ok());
  EXPECT_EQ(text.error().code, "E-YAML-REF");
}

// N-base-model-1: a FieldSpec's RegionId is accepted by add_field() with no
// bounds check (add_field() only asserts the owning ComponentId is in
// range), the same reachability gap as the region->mesh check to_yaml()
// already has for Region::mesh.
TEST(YamlIo, EmittingAGraphWithADanglingFieldRegionReturnsFailure) {
  CouplingGraph g;
  const auto a = g.add_component(ComponentDecl{.name = "a",
                                                .ranks = fuse::RankSet{0, 1},
                                                .interval = Rational{1}});
  // No region exists in g, so RegionId{7} is out of range.
  (void)g.add_field(a, FieldSpec{.name = "f", .region = fuse::RegionId{7},
                                 .units = u::kHeatFluxDensity,
                                 .conservation = Conservation::Extensive,
                                 .intent = Intent::Provides});

  const auto text = to_yaml(g);
  ASSERT_FALSE(text.ok());
  EXPECT_EQ(text.error().code, "E-YAML-REF");
}

TEST(YamlIo, RejectsUnknownFieldReference) {
  const std::string bad = R"(
version: 1
meshes: [{name: m, source: "generated:box"}]
regions: [{name: r, mesh: m, entity_set: w, dimension: 2}]
components:
  - name: a
    ranks: {first: 0, count: 1}
    interval: 1
    fields: [{name: f, region: r, intent: Provides}]
channels:
  - {name: c, source: a.f, target: b.nope}
)";
  const auto g = from_yaml(bad);
  ASSERT_FALSE(g.ok());
  EXPECT_EQ(g.error().code, "E-YAML-REF");
}

TEST(YamlIo, RejectsUnknownUnitSymbol) {
  const std::string bad = R"(
version: 1
meshes: [{name: m, source: "generated:box"}]
regions: [{name: r, mesh: m, entity_set: w, dimension: 2}]
components:
  - name: a
    ranks: {first: 0, count: 1}
    interval: 1
    fields: [{name: f, region: r, units: furlongs, intent: Provides}]
channels: []
)";
  const auto g = from_yaml(bad);
  ASSERT_FALSE(g.ok());
  EXPECT_EQ(g.error().code, "E-YAML-UNITS");
}

// N-base-model-2: R-19 gated a non-positive Units::scale at the binary
// descriptor boundary (Descriptor.RejectsNonPositiveUnitScale) and at
// check_units()/validate_field_semantics(), but from_yaml()'s parse_units()
// parsed "scale" with no sign or zero check at all -- an equally real
// front-end per testing.md's three-leg round-trip. Mirrors
// Descriptor.RejectsNonPositiveUnitScale's two cases through from_yaml()
// instead of deserialize().
TEST(YamlIo, RejectsNonPositiveUnitScale) {
  const auto encode = [](const std::string& scale) {
    return R"(
version: 1
meshes: [{name: m, source: "generated:box"}]
regions: [{name: r, mesh: m, entity_set: w, dimension: 2}]
components:
  - name: a
    ranks: {first: 0, count: 1}
    interval: 1
    fields: [{name: f, region: r, intent: Provides, units: {length: 1, scale: ")" +
           scale + R"("}}]
channels: []
)";
  };
  for (const std::string& bad_scale : {std::string("0"), std::string("-1000")}) {
    const auto g = from_yaml(encode(bad_scale));
    ASSERT_FALSE(g.ok()) << "scale " << bad_scale;
    EXPECT_EQ(g.error().code, "E-YAML-UNITS-SCALE");
  }
}

TEST(YamlIo, RejectsWrongSchemaVersion) {
  const auto g = from_yaml("version: 99\nmeshes: []\n");
  ASSERT_FALSE(g.ok());
  EXPECT_EQ(g.error().code, "E-YAML-VERSION");
}

TEST(YamlIo, ParsesRationalIntervalExactly) {
  const auto g = fuse::load_yaml_file(FUSE_TEST_DATA_DIR "/pair.yaml").value();
  EXPECT_EQ(g.component(g.find_component("fluid").value()).interval,
            Rational(1, 2));
}

// IMPORTANT 3 (post-review fix): Brief 1.7 states "the reader rejects
// unknown keys", but yaml-cpp only reports on keys the caller looks up, so
// nothing enforced that until check_unknown_keys() was added. A misspelled
// key ("sourcee" instead of "source") in a channel block must be rejected,
// not silently ignored (which would otherwise fail later, confusingly, on
// the missing-but-required "source" key instead).
TEST(YamlIo, RejectsUnknownKeyInChannelBlock) {
  const std::string bad = R"(
version: 1
meshes: [{name: m, source: "generated:box"}]
regions: [{name: r, mesh: m, entity_set: w, dimension: 2}]
components:
  - name: a
    ranks: {first: 0, count: 1}
    interval: 1
    fields: [{name: f, region: r, intent: Provides}]
  - name: b
    ranks: {first: 1, count: 1}
    interval: 1
    fields: [{name: g, region: r, intent: Requires}]
channels:
  - {name: c, sourcee: a.f, target: b.g}
)";
  const auto g = from_yaml(bad);
  ASSERT_FALSE(g.ok());
  EXPECT_EQ(g.error().code, "E-YAML-UNKNOWN-KEY");
}

TEST(YamlIo, RejectsUnknownTopLevelKey) {
  const std::string bad = "version: 1\nmeshes: []\ntotally_bogus_key: 42\n";
  const auto g = from_yaml(bad);
  ASSERT_FALSE(g.ok());
  EXPECT_EQ(g.error().code, "E-YAML-UNKNOWN-KEY");
}

// ---- the three-leg round-trip (S2 exit) --------------------------------

// The waist's whole claim is that C++, YAML and the canonical bytes describe the
// same graph. Two legs prove nothing about the third: a front-end could lose a
// member consistently in both directions and its own round-trip would still pass.
// So the comparison is against the BYTES, which are the definition.
// Component index is the placement space, so a coupler authored BETWEEN two
// physics components must come back between them. The writer keeps physics
// components and couplers in two sections, so this is only true because each
// coupler carries its `index`; the first graph that authored a component
// after a coupler (the v10 freeze fixture, S6 phase 4) came back reordered,
// and the fixture was rearranged to hide it instead of the writer being
// fixed. This pins the fix.
// A coupler that is LAUNCHED as its own program -- its own MPMD segment,
// with its own appnum -- must say so in the file, or the program that
// loads the graph on the coupler's own ranks believes it is in-process and
// never joins the collective connect() makes over the whole world. That is
// not a theoretical loss: the Brusselator's MPMD participant rows hung for
// their full 120 s with an empty combined.log on the host (2026-10-08),
// U and V waiting in connect_mpmd()'s MPI_Allgather for a third
// participant whose own executable had decided it had nothing to connect.
// The graph tool had always BUILT the right LaunchSpec; the file was where
// it was lost.
TEST(YamlIo, ALaunchedCouplerKeepsItsLaunchModeAcrossTheRoundTrip) {
  fuse::CouplingGraph g;
  (void)g.add_component({.name = "a", .ranks = {0, 2}, .interval = fuse::Rational{1, 1}});
  const auto hub = g.add_coupler(
      {.name = "hub",
       .ranks = {2, 1},
       .launch = {.mode = fuse::LaunchMode::Mpmd, .appnum = 2}});
  ASSERT_TRUE(hub.valid());

  const auto text = fuse::to_yaml(g);
  ASSERT_TRUE(text.ok()) << text.error().message;
  const auto parsed = fuse::from_yaml(text.value());
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;

  const auto& back = parsed.value().component(hub);
  EXPECT_EQ(back.kind, fuse::ComponentKind::Coupler);
  EXPECT_EQ(back.launch.mode, fuse::LaunchMode::Mpmd)
      << "a coupler reloaded as in-process never calls connect(), and the "
         "programs that do call it then wait for it forever";
  EXPECT_EQ(back.launch.appnum, 2);
  EXPECT_EQ(fuse::serialize(parsed.value()), fuse::serialize(g))
      << "the launch record is part of every component record's own bytes";
}

// An in-process coupler writes no launch block at all, which is what keeps
// a file written before couplers could carry one byte-identical.
TEST(YamlIo, AnInProcessCouplerWritesNoLaunchBlock) {
  fuse::CouplingGraph g;
  (void)g.add_component({.name = "a", .ranks = {0, 2}, .interval = fuse::Rational{1, 1}});
  (void)g.add_coupler({.name = "plain", .ranks = {2, 1}});
  const auto text = fuse::to_yaml(g);
  ASSERT_TRUE(text.ok()) << text.error().message;
  const auto at = text.value().find("plain");
  ASSERT_NE(at, std::string::npos);
  EXPECT_EQ(text.value().find("launch", at), std::string::npos)
      << text.value().substr(at);
}

TEST(YamlIo, ACouplerAuthoredBetweenPhysicsComponentsKeepsItsIndex) {
  fuse::CouplingGraph g;
  (void)g.add_component({.name = "a", .ranks = {0, 2}, .interval = fuse::Rational{1, 1}});
  const auto mid = g.add_coupler({.name = "hub", .ranks = {2, 1}});
  const auto b = g.add_component({.name = "b", .ranks = {3, 2}, .interval = fuse::Rational{1, 1}});
  ASSERT_EQ(mid.value(), 1u);
  ASSERT_EQ(b.value(), 2u);

  const auto text = fuse::to_yaml(g);
  ASSERT_TRUE(text.ok()) << text.error().message;
  const auto parsed = fuse::from_yaml(text.value());
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;

  ASSERT_EQ(parsed.value().component_count(), 3u);
  EXPECT_EQ(parsed.value().component(fuse::ComponentId{1}).name, "hub");
  EXPECT_EQ(parsed.value().component(fuse::ComponentId{1}).kind, fuse::ComponentKind::Coupler);
  EXPECT_EQ(parsed.value().component(fuse::ComponentId{2}).name, "b");
  EXPECT_EQ(fuse::serialize(parsed.value()), fuse::serialize(g))
      << "the round trip moved the coupler, so every placement index after it changed";
}

// A text without `index` on its couplers (hand-written, or written before
// the key existed) appends them after every physics component.
TEST(YamlIo, CouplersWithoutAnIndexAreAppended) {
  fuse::CouplingGraph g;
  (void)g.add_component({.name = "a", .ranks = {0, 2}, .interval = fuse::Rational{1, 1}});
  (void)g.add_coupler({.name = "hub", .ranks = {2, 1}});
  const auto text = fuse::to_yaml(g);
  ASSERT_TRUE(text.ok()) << text.error().message;
  std::string stripped = text.value();
  const auto k = stripped.find("index: 1");
  ASSERT_NE(k, std::string::npos) << text.value();
  stripped.erase(k, std::string("index: 1").size());
  const auto parsed = fuse::from_yaml(stripped);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  ASSERT_EQ(parsed.value().component_count(), 2u);
  EXPECT_EQ(parsed.value().component(fuse::ComponentId{1}).name, "hub");
}

TEST(YamlIo, ThreeLegRoundTripIsStructurallyIdentical) {
  const auto authored = fuse::testing::freeze_fixture();   // leg 1: C++
  const auto direct_bytes = fuse::serialize(authored);

  const auto text = fuse::to_yaml(authored);               // leg 2: YAML
  ASSERT_TRUE(text.ok()) << text.error().message;
  const auto parsed = fuse::from_yaml(text.value());
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;

  const auto via_yaml = fuse::serialize(parsed.value());   // leg 3: bytes
  EXPECT_EQ(via_yaml, direct_bytes)
      << "a graph authored in C++ and the same graph after a YAML round trip do not "
         "serialize identically, so the YAML front-end is losing or inventing a member";

  // And the YAML is text a human can read and edit: names, not indices.
  EXPECT_NE(text.value().find("heat_flux"), std::string::npos);
  EXPECT_NE(text.value().find("fluid"), std::string::npos);
}

// The same property from the other end: the checked-in fixture, read from disk,
// must serialize to the same bytes as a re-emit-and-reparse of itself. This is
// what catches an asymmetric emitter -- one that writes a default it cannot read
// back, or reads a key it never writes.
TEST(YamlIo, FixtureSurvivesAnEmitParseCycleByteForByte) {
  const auto loaded = fuse::load_yaml_file(FUSE_TEST_DATA_DIR "/pair.yaml");
  ASSERT_TRUE(loaded.ok()) << loaded.error().message;
  const auto once = fuse::serialize(loaded.value());

  const auto text = fuse::to_yaml(loaded.value());
  ASSERT_TRUE(text.ok()) << text.error().message;
  const auto again = fuse::from_yaml(text.value());
  ASSERT_TRUE(again.ok()) << again.error().message;

  EXPECT_EQ(fuse::serialize(again.value()), once);
}

// ---------------------------------------------------------------------------
// S7.2: AccelKind::Newton and NewtonConfig. NOT checked via
// structurally_equal()/serialize() (the canonical-bytes descriptor,
// fuse/model/descriptor.hpp, does not cross either requires_residual_
// capability/requires_linearization_capability or newton_config at all --
// out of this track's own ownership, recorded as an open question in its
// report rather than silently assumed covered) -- checked directly against
// the restored IterationRegionDecl's own fields instead, which IS what
// this track's yaml_io.cpp round-trips.
// ---------------------------------------------------------------------------

namespace {

CouplingGraph newton_graph(fuse::NewtonConfig cfg) {
  CouplingGraph g = pair_graph();
  // pair_graph() adds exactly one channel ("flux"), so it is ChannelId{0}.
  g.iterate({fuse::ChannelId{0}}).newton(cfg);
  return g;
}

}  // namespace

TEST(YamlIo, NewtonKindAndConfigSurviveTheTextRoundTrip) {
  fuse::NewtonConfig cfg;
  cfg.line_search_kind = fuse::LineSearchKind::CP;
  cfg.forcing_choice = fuse::EisenstatWalkerChoice::Choice1;
  cfg.atol = 1e-9;
  cfg.rtol = 1e-7;
  cfg.stol = 1e-6;
  cfg.divergence_tolerance = 5e3;
  cfg.max_nonlinear_iterations = 37;
  cfg.max_function_evaluations = 4321;
  cfg.gmres_restart = 17;
  cfg.gmres_max_iterations = 777;
  cfg.preconditioner_lag = 3;
  cfg.use_block_jacobi_preconditioner = true;
  cfg.forcing_gamma = 0.8;
  cfg.eta_min = 2e-4;
  // S7.2b (PetscSelect track): non-default, so the round trip actually
  // exercises reading the field back rather than reconfirming a default
  // both sides happen to agree on.
  cfg.substrate = fuse::NewtonSubstrate::Petsc;

  const CouplingGraph original = newton_graph(cfg);
  const auto text = to_yaml(original);
  ASSERT_TRUE(text.ok()) << text.error().message;
  const auto restored = from_yaml(text.value());
  ASSERT_TRUE(restored.ok()) << restored.error().message;

  const auto& want = original.iteration_region(fuse::IterationRegionIdx{0});
  const auto& got = restored.value().iteration_region(fuse::IterationRegionIdx{0});

  EXPECT_EQ(got.acceleration.kind, fuse::AccelKind::Newton);
  EXPECT_TRUE(got.requires_residual_capability);
  EXPECT_TRUE(got.requires_linearization_capability);

  EXPECT_EQ(got.newton_config.line_search_kind, want.newton_config.line_search_kind);
  EXPECT_EQ(got.newton_config.forcing_choice, want.newton_config.forcing_choice);
  EXPECT_DOUBLE_EQ(got.newton_config.atol, want.newton_config.atol);
  EXPECT_DOUBLE_EQ(got.newton_config.rtol, want.newton_config.rtol);
  EXPECT_DOUBLE_EQ(got.newton_config.stol, want.newton_config.stol);
  EXPECT_DOUBLE_EQ(got.newton_config.divergence_tolerance,
                   want.newton_config.divergence_tolerance);
  EXPECT_EQ(got.newton_config.max_nonlinear_iterations,
           want.newton_config.max_nonlinear_iterations);
  EXPECT_EQ(got.newton_config.max_function_evaluations,
           want.newton_config.max_function_evaluations);
  EXPECT_EQ(got.newton_config.gmres_restart, want.newton_config.gmres_restart);
  EXPECT_EQ(got.newton_config.gmres_max_iterations, want.newton_config.gmres_max_iterations);
  EXPECT_EQ(got.newton_config.preconditioner_lag, want.newton_config.preconditioner_lag);
  EXPECT_EQ(got.newton_config.use_block_jacobi_preconditioner,
           want.newton_config.use_block_jacobi_preconditioner);
  EXPECT_DOUBLE_EQ(got.newton_config.forcing_gamma, want.newton_config.forcing_gamma);
  EXPECT_DOUBLE_EQ(got.newton_config.eta_min, want.newton_config.eta_min);
  EXPECT_EQ(got.newton_config.substrate, want.newton_config.substrate);
  EXPECT_EQ(got.newton_config.substrate, fuse::NewtonSubstrate::Petsc);
}

// An unrecognised newton_config.substrate value must be refused the same
// way an unrecognised line_search_kind/forcing_choice already is (this
// file's own RejectsUnknownKeyInChannelBlock-style discipline, applied to
// enum VALUES rather than keys) -- never silently defaulted to Native,
// which would make a graph author's typo mean something they never asked
// for.
TEST(YamlIo, RejectsUnknownNewtonConfigSubstrate) {
  const CouplingGraph original = newton_graph(fuse::NewtonConfig{});
  const auto text = to_yaml(original);
  ASSERT_TRUE(text.ok()) << text.error().message;
  std::string corrupted = text.value();
  const auto pos = corrupted.find("substrate: Native");
  ASSERT_NE(pos, std::string::npos);
  corrupted.replace(pos, std::string("substrate: Native").size(), "substrate: quantum");
  const auto restored = from_yaml(corrupted);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

// The same auto-trigger the C++ builder applies (IterationBuilder::
// accelerate()) must apply on the YAML read side too: a hand-written YAML
// file that selects Newton but forgets (or omits) the capability keys must
// still come back with requires_residual_capability true, never silently
// trusting the omitted/false value.
TEST(YamlIo, HandWrittenYamlSelectingNewtonGetsTheCapabilityFlagEvenWhenOmitted) {
  const CouplingGraph original = newton_graph(fuse::NewtonConfig{});
  const auto text = to_yaml(original);
  ASSERT_TRUE(text.ok()) << text.error().message;
  // Strip the explicit capability keys and the newton_config block
  // entirely, simulating a hand-written file that never mentions them --
  // only "acceleration: {kind: Newton, ...}" survives.
  std::string stripped = text.value();
  const auto drop_line_containing = [&](const char* needle) {
    std::size_t pos = stripped.find(needle);
    while (pos != std::string::npos) {
      const std::size_t line_start = stripped.rfind('\n', pos) + 1;
      const std::size_t line_end = stripped.find('\n', pos) + 1;
      stripped.erase(line_start, line_end - line_start);
      pos = stripped.find(needle);
    }
  };
  drop_line_containing("requires_residual_capability");
  drop_line_containing("requires_linearization_capability");

  const auto restored = from_yaml(stripped);
  ASSERT_TRUE(restored.ok()) << restored.error().message;
  const auto& got = restored.value().iteration_region(fuse::IterationRegionIdx{0});
  EXPECT_EQ(got.acceleration.kind, fuse::AccelKind::Newton);
  EXPECT_TRUE(got.requires_residual_capability)
      << "selecting Newton must set this even when the key is absent from the file";
}

TEST(YamlIo, UnknownKeyInNewtonConfigBlockIsRefused) {
  const CouplingGraph original = newton_graph(fuse::NewtonConfig{});
  const auto text = to_yaml(original);
  ASSERT_TRUE(text.ok()) << text.error().message;
  std::string mutated = text.value();
  const std::size_t pos = mutated.find("newton_config:");
  ASSERT_NE(pos, std::string::npos);
  const std::size_t insert_at = mutated.find('\n', pos) + 1;
  mutated.insert(insert_at, "      bogus_key: 1\n");

  const auto restored = from_yaml(mutated);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-UNKNOWN-KEY");
}
