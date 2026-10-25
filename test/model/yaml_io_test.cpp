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
// and remap.method_override -- a to_yaml() that silently dropped all three
// (never emitting them, with from_yaml() hardcoding
// Window::coupling_period() on both window paths) would still pass
// RoundTripsThroughText if pair_graph() only used defaults. This fixture
// catches that.
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

// to_yaml()'s own signature promises Result<std::string> for
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

// A FromComponent window's ComponentId is accepted by ChannelBuilder::
// accumulate()/spacetime() with no bounds check, the same reachability
// as a channel's source/target field and an iteration region's channel
// list above, so each window field needs its own guard. Mirrors EmittingAGraphWithADanglingChannelReferenceReturnsFailure
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

// A FieldSpec's RegionId is accepted by add_field() with no
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

// A non-positive Units::scale is gated at the binary descriptor boundary
// (Descriptor.RejectsNonPositiveUnitScale) and at
// check_units()/validate_field_semantics(), and from_yaml()'s parse_units()
// must gate it too -- an equally real front-end per testing.md's
// three-leg round-trip. Mirrors
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

// The reader rejects unknown keys, but yaml-cpp only reports on keys the
// caller looks up, so check_unknown_keys() enforces it. A misspelled
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

// ---- the three-leg round-trip --------------------------------

// The waist's whole claim is that C++, YAML and the canonical bytes describe the
// same graph. Two legs prove nothing about the third: a front-end could lose a
// member consistently in both directions and its own round-trip would still pass.
// So the comparison is against the BYTES, which are the definition.
// Component index is the placement space, so a coupler authored BETWEEN two
// physics components must come back between them. The writer keeps physics
// components and couplers in two sections, so this is only true because each
// coupler carries its `index`; the first graph that authored a component
// after a coupler (the v10 freeze fixture) would come back reordered
// unless the writer preserves it. This pins that.
// A coupler that is LAUNCHED as its own program -- its own MPMD segment,
// with its own appnum -- must say so in the file, or the program that
// loads the graph on the coupler's own ranks believes it is in-process and
// never joins the collective connect() makes over the whole world. That is
// not a theoretical loss: the Brusselator's MPMD participant rows hung for
// their full 120 s with an empty combined.log on the host,
// U and V waiting in connect_mpmd()'s MPI_Allgather for a third
// participant whose own executable had decided it had nothing to connect.
// The graph tool BUILDS the right LaunchSpec; the file is where it would
// be lost.
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
// AccelKind::Newton and NewtonConfig. NOT checked via
// structurally_equal()/serialize() (the canonical-bytes descriptor,
// fuse/model/descriptor.hpp, does not cross either requires_residual_
// capability/requires_linearization_capability or newton_config at all) --
// checked directly against the restored IterationRegionDecl's own fields
// instead, which IS what yaml_io.cpp round-trips.
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
  // Non-default, so the round trip actually
  // exercises reading the field back rather than reconfirming a default
  // both sides happen to agree on.
  cfg.substrate = fuse::NewtonSubstrate::Petsc;
  // Same reasoning, for norm and residual_path.
  cfg.norm = fuse::NewtonNorm::MeasureWeighted;
  cfg.residual_path = fuse::NewtonResidualPath::Direct;

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
  EXPECT_EQ(got.newton_config.norm, want.newton_config.norm);
  EXPECT_EQ(got.newton_config.norm, fuse::NewtonNorm::MeasureWeighted);
  EXPECT_EQ(got.newton_config.residual_path, want.newton_config.residual_path);
  EXPECT_EQ(got.newton_config.residual_path, fuse::NewtonResidualPath::Direct);
}

// An unrecognised newton_config.norm/residual_path value must be refused
// the same way an unrecognised substrate already is (RejectsUnknown
// NewtonConfigSubstrate, immediately above) -- never silently defaulted.
TEST(YamlIo, RejectsUnknownNewtonConfigNorm) {
  const CouplingGraph original = newton_graph(fuse::NewtonConfig{});
  const auto text = to_yaml(original);
  ASSERT_TRUE(text.ok()) << text.error().message;
  std::string corrupted = text.value();
  const auto pos = corrupted.find("norm: Plain");
  ASSERT_NE(pos, std::string::npos);
  corrupted.replace(pos, std::string("norm: Plain").size(), "norm: quantum");
  const auto restored = from_yaml(corrupted);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, RejectsUnknownNewtonConfigResidualPath) {
  const CouplingGraph original = newton_graph(fuse::NewtonConfig{});
  const auto text = to_yaml(original);
  ASSERT_TRUE(text.ok()) << text.error().message;
  std::string corrupted = text.value();
  const auto pos = corrupted.find("residual_path: Advance");
  ASSERT_NE(pos, std::string::npos);
  corrupted.replace(pos, std::string("residual_path: Advance").size(),
                    "residual_path: quantum");
  const auto restored = from_yaml(corrupted);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
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

// ---------------------------------------------------------------------
// from_yaml() must not read a required key with a bare
// `node["key"].as<T>()` -- no presence guard, no surrounding `try` -- since
// a document that LOADS (valid YAML syntax) but omits or misspells a
// required key, or supplies a wrong-typed scalar for one, would make
// yaml-cpp throw a raw `YAML::Exception` out of a function whose contract
// is `Result<CouplingGraph>`. Each case below is a minimal,
// otherwise-valid document with exactly one defect.
// ---------------------------------------------------------------------
namespace {

// A minimal document that loads cleanly through every required key this
// file's from_yaml() reads: one mesh, one region, one component with one
// field, no channels, no iteration regions.
const char* const kMinimalYaml = R"(version: 1
meshes:
  - name: m
regions:
  - name: r
    mesh: m
components:
  - name: c
    fields:
      - name: f
        region: r
channels: []
iteration_regions: []
)";

// Deletes the first line containing `needle` (used to drop a required
// key's own `key: value` line from kMinimalYaml or a derived document).
std::string drop_line(std::string text, const char* needle) {
  const std::size_t pos = text.find(needle);
  if (pos == std::string::npos) return text;
  const std::size_t line_start = text.rfind('\n', pos) + 1;
  const std::size_t line_end = text.find('\n', pos) + 1;
  return text.substr(0, line_start) + text.substr(line_end);
}

}  // namespace

TEST(YamlIo, MinimalDocumentLoadsCleanly) {
  const auto g = from_yaml(kMinimalYaml);
  ASSERT_TRUE(g.ok()) << g.error().message;
  EXPECT_EQ(g.value().mesh_count(), 1u);
  EXPECT_EQ(g.value().region_count(), 1u);
  EXPECT_EQ(g.value().field_count(), 1u);
}

TEST(YamlIo, MissingMeshNameIsRefused) {
  // `drop_line` would remove the whole `- name: m` entry here (it is the
  // mesh's only key), collapsing the list to empty rather than leaving a
  // present-but-keyless mesh entry; replacing with an empty map keeps the
  // entry (and the required-key check reachable) while removing the key.
  std::string text = kMinimalYaml;
  const std::size_t pos = text.find("- name: m");
  text.replace(pos, std::string("- name: m").size(), "- {}");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingRegionMeshIsRefused) {
  const auto restored = from_yaml(drop_line(kMinimalYaml, "mesh: m"));
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingRegionNameIsRefused) {
  const auto restored = from_yaml(drop_line(kMinimalYaml, "name: r"));
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingComponentNameIsRefused) {
  const auto restored = from_yaml(drop_line(kMinimalYaml, "name: c"));
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingFieldNameIsRefused) {
  const auto restored = from_yaml(drop_line(kMinimalYaml, "name: f"));
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingFieldRegionIsRefused) {
  const auto restored = from_yaml(drop_line(kMinimalYaml, "region: r"));
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingCouplerNameIsRefused) {
  std::string text = kMinimalYaml;
  const std::size_t at = text.find("channels: []");
  text.insert(at, "couplers:\n  - ranks: {first: 0, count: 1}\n");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingRanksFirstIsRefused) {
  std::string text = kMinimalYaml;
  const std::size_t at = text.find("fields:");
  text.insert(at, "ranks: {count: 1}\n    ");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingRanksCountIsRefused) {
  std::string text = kMinimalYaml;
  const std::size_t at = text.find("fields:");
  text.insert(at, "ranks: {first: 0}\n    ");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingChannelNameIsRefused) {
  std::string text = kMinimalYaml;
  text = drop_line(text, "channels: []");
  text += "channels:\n  - source: c.f\n    target: c.f\n";
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingChannelSourceIsRefused) {
  std::string text = kMinimalYaml;
  text = drop_line(text, "channels: []");
  text += "channels:\n  - name: ch\n    target: c.f\n";
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingChannelTargetIsRefused) {
  std::string text = kMinimalYaml;
  text = drop_line(text, "channels: []");
  text += "channels:\n  - name: ch\n    source: c.f\n";
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingIterationRegionNameIsRefused) {
  std::string text = kMinimalYaml;
  text = drop_line(text, "iteration_regions: []");
  text += "iteration_regions:\n  - channels: []\n";
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, MissingSpacetimeBasisIsRefused) {
  std::string text = kMinimalYaml;
  text = drop_line(text, "channels: []");
  text += "channels:\n  - name: ch\n    source: c.f\n    target: c.f\n"
          "    spacetime: {degree: 1}\n";
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

// Wrong-typed-scalar classes: a map where a scalar was expected, and a
// string where a number was expected. Both throw a raw
// YAML::BadConversion inside from_yaml(); the outer
// try/catch in from_yaml() (around from_yaml_body()) translates
// whichever one yaml-cpp throws into E-YAML-PARSE.
TEST(YamlIo, WrongTypedScalarMapWhereStringExpectedIsRefused) {
  std::string text = kMinimalYaml;
  // "name" for the mesh entry becomes a map instead of a string scalar.
  const std::size_t pos = text.find("name: m");
  text.replace(pos, std::string("name: m").size(), "name: {a: 1}");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

TEST(YamlIo, WrongTypedScalarStringWhereNumberExpectedIsRefused) {
  std::string text = kMinimalYaml;
  const std::size_t at = text.find("fields:");
  text.insert(at, "ranks: {first: abc, count: 1}\n    ");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

// An out-of-range structural-map exponent (narrow_exponent's own
// bound, fuse/base/units.hpp) must be refused, not silently wrapped into
// a different physical quantity.
TEST(YamlIo, DuplicateMeshNameIsRefused) {
  std::string text = kMinimalYaml;
  const std::size_t at = text.find("regions:");
  text.insert(at, "  - name: m\n");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-DUP");
}

TEST(YamlIo, DuplicateRegionNameIsRefused) {
  std::string text = kMinimalYaml;
  const std::size_t at = text.find("components:");
  text.insert(at, "  - name: r\n    mesh: m\n");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-DUP");
}

TEST(YamlIo, DuplicateChannelNameIsRefused) {
  std::string text = kMinimalYaml;
  text = drop_line(text, "channels: []");
  text += "channels:\n"
          "  - name: ch\n    source: c.f\n    target: c.f\n"
          "  - name: ch\n    source: c.f\n    target: c.f\n";
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-DUP");
}

// A non-positive max_iterations, and a non-finite/non-positive
// convergence tolerance, must be refused at the authoring gate rather
// than surfacing only mid-sweep (ConvergenceMeasure::satisfied()'s own
// FUSE_THROW, fuse/model/iteration.hpp).
TEST(YamlIo, NonPositiveMaxIterationsIsRefused) {
  std::string text = kMinimalYaml;
  text = drop_line(text, "iteration_regions: []");
  text += "iteration_regions:\n  - channels: []\n    name: outer\n    max_iterations: 0\n";
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}

// parse_units()'s scalar branch uses the real unit-expression parser
// semantics.hpp documents as "how a YAML ... caller states units at all",
// not a private table that would refuse the parser's other symbols with
// E-YAML-UNITS. Tests both spellings of the same quantity ("Hz" and the
// structural map it is NOT dimensionally equal to, to show the scalar form
// parses at all) plus a compound expression ("m/s").
TEST(YamlIo, NonPositiveConvergenceToleranceIsRefused) {
  std::string text = kMinimalYaml;
  text = drop_line(text, "iteration_regions: []");
  text += "iteration_regions:\n"
          "  - channels: []\n    name: outer\n"
          "    convergence:\n      - {kind: Relative, tolerance: -1.0, field: f}\n";
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-PARSE");
}
TEST(YamlIo, OutOfRangeUnitExponentIsRefused) {
  std::string text = kMinimalYaml;
  const std::size_t pos = text.find("name: f");
  text.insert(pos + std::string("name: f\n").size(),
             "        units: {length: 300}\n");
  const auto restored = from_yaml(text);
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-YAML-UNITS");
}

// Two meshes/regions/channels with the same name must be refused
// at parse time, not silently resolved to whichever was declared last.
TEST(YamlIo, PreviouslyUnsupportedUnitSymbolsNowParse) {
  std::string text = kMinimalYaml;
  const std::size_t pos = text.find("name: f");
  text.insert(pos + std::string("name: f\n").size(), "        units: Hz\n");
  const auto restored = from_yaml(text);
  ASSERT_TRUE(restored.ok()) << restored.error().message;
  const auto& field = restored.value().field(fuse::FieldId{0});
  EXPECT_EQ(field.units.time, -1) << "Hz = s^-1";
}

TEST(YamlIo, CompoundUnitExpressionParses) {
  std::string text = kMinimalYaml;
  const std::size_t pos = text.find("name: f");
  text.insert(pos + std::string("name: f\n").size(), "        units: m/s\n");
  const auto restored = from_yaml(text);
  ASSERT_TRUE(restored.ok()) << restored.error().message;
  const auto& field = restored.value().field(fuse::FieldId{0});
  EXPECT_EQ(field.units.length, 1);
  EXPECT_EQ(field.units.time, -1);
}

// The 10 symbols the old table DID have must still parse identically
// (same Units value) through the real parser -- the round trip this
// routing change must not disturb.
TEST(YamlIo, PreviouslySupportedUnitSymbolsStillParseTheSameWay) {
  std::string text = kMinimalYaml;
  const std::size_t pos = text.find("name: f");
  text.insert(pos + std::string("name: f\n").size(), "        units: Pa\n");
  const auto restored = from_yaml(text);
  ASSERT_TRUE(restored.ok()) << restored.error().message;
  const auto& field = restored.value().field(fuse::FieldId{0});
  EXPECT_EQ(field.units, u::kPascal);
}

