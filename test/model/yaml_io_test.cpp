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
      .name = "fluid", .ranks = {.first = 0, .count = 2},
      .interval = Rational{1, 2}});
  const auto solid = g.add_component(ComponentDecl{
      .name = "solid", .ranks = {.first = 2, .count = 2},
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
