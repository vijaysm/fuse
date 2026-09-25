#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include <gtest/gtest.h>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/descriptor.hpp>
#include <fuse/model/freeze_fixture.hpp>

using fuse::AccumOp;
using fuse::ComponentDecl;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::deserialize;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
using fuse::RemapFlags;
using fuse::TemporalBasis;
namespace u = fuse::units;

// Exercises every field the descriptor must carry, including the awkward ones:
// rational intervals, spacetime specs, a lag, and a mismatch reason.
CouplingGraph rich_graph() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "mesh_a", .source = "a.h5m"});
  const auto r = g.add_region(Region{.name = "iface", .mesh = m,
                                     .entity_set = "ss1", .dimension = 2});
  const auto a = g.add_component(ComponentDecl{.name = "A",
                                               .ranks = fuse::RankSet{0, 3},
                                               .interval = Rational{1, 3}});
  const auto b = g.add_component(ComponentDecl{.name = "B",
                                               .ranks = fuse::RankSet{3, 5},
                                               .interval = Rational{1, 7}});
  const auto s = g.add_field(a, FieldSpec{.name = "flux", .region = r,
                                          .disc = fuse::Discretization::DiscontinuousGLL,
                                          .order = 3, .components = 3,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Provides});
  const auto t = g.add_field(b, FieldSpec{.name = "flux_in", .region = r,
                                          .disc = fuse::Discretization::FV,
                                          .components = 3,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Requires});
  const auto ch = g.add_channel("A->B", s, t);
  g.channel(ch)
      .accumulate(AccumOp::Integral, fuse::Window::from(a))
      .remap(RemapFlags::Conservative | RemapFlags::HighOrder)
      .remap_method("mono3")
      .spacetime(TemporalBasis::DG, 2, fuse::Window::explicit_length(Rational{2, 5}))
      .relax(0.35)
      .extrapolate(2)
      .lag(0)
      .allow_time_level_mismatch("verified by hand");
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "outer", .channels = {ch}, .max_iterations = 25});
  return g;
}

TEST(Descriptor, HeaderHasMagicAndVersion) {
  const auto bytes = serialize(rich_graph());
  ASSERT_GE(bytes.size(), 8u);
  std::uint32_t magic = 0, version = 0;
  std::memcpy(&magic, bytes.data(), 4);
  std::memcpy(&version, bytes.data() + 4, 4);
  EXPECT_EQ(magic, fuse::kGraphMagic);
  EXPECT_EQ(version, fuse::kGraphVersion);
}

TEST(Descriptor, RoundTripsRichGraph) {
  const CouplingGraph original = rich_graph();
  const auto bytes = serialize(original);
  auto restored = deserialize(bytes);
  ASSERT_TRUE(restored.ok()) << restored.error().message;
  EXPECT_TRUE(structurally_equal(original, restored.value()));
}

TEST(Descriptor, PreservesRationalIntervalsExactly) {
  const auto bytes = serialize(rich_graph());
  const auto g = deserialize(bytes).value();
  EXPECT_EQ(g.component(g.find_component("A").value()).interval, Rational(1, 3));
  EXPECT_EQ(g.component(g.find_component("B").value()).interval, Rational(1, 7));
}

TEST(Descriptor, PreservesChannelSemantics) {
  const auto g = deserialize(serialize(rich_graph())).value();
  const auto& ch = g.channel_at(fuse::ChannelId{0});
  EXPECT_EQ(ch.accumulate.op, AccumOp::Integral);
  EXPECT_EQ(ch.remap.method_override, "mono3");
  EXPECT_EQ(ch.spacetime.basis, TemporalBasis::DG);
  EXPECT_EQ(ch.spacetime.degree, 2);
  EXPECT_EQ(ch.spacetime.window.length, Rational(2, 5));
  EXPECT_DOUBLE_EQ(ch.relax.omega, 0.35);
  EXPECT_EQ(ch.extrapolate.order, 2);
  EXPECT_EQ(ch.time_level_mismatch_reason, "verified by hand");
}

TEST(Descriptor, PreservesIterationRegions) {
  const auto g = deserialize(serialize(rich_graph())).value();
  ASSERT_EQ(g.iteration_region_count(), 1u);
  EXPECT_EQ(g.iteration_region(fuse::IterationRegionIdx{0}).max_iterations, 25);
}

TEST(Descriptor, RejectsTruncatedBuffer) {
  auto bytes = serialize(rich_graph());
  bytes.resize(bytes.size() / 2);
  const auto r = deserialize(bytes);
  EXPECT_FALSE(r.ok());
}

TEST(Descriptor, RejectsWrongMagic) {
  auto bytes = serialize(rich_graph());
  bytes[0] = std::byte{0xFF};
  const auto r = deserialize(bytes);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-MAGIC");
}

TEST(Descriptor, SerializationIsDeterministic) {
  EXPECT_EQ(serialize(rich_graph()), serialize(rich_graph()));
}

// IMPORTANT 5 (post-review): ComponentDecl::regions is harmless today only
// because nothing in the public API populates it (there is no
// add_region_to_component()-style call). Since structurally_equal() is
// defined in terms of serialize(), it cannot detect a field neither side
// writes -- so this pins the choice explicitly: populating `regions`
// directly on a graph must not change what serialize() produces. If Phase 2
// starts populating this field through the graph API, this test should be
// replaced with round-trip coverage, and the writer/reader extended to
// carry it -- see the comment on ComponentDecl::regions.
TEST(Descriptor, RegionsFieldIsIntentionallyNotSerialized) {
  CouplingGraph a = rich_graph();
  CouplingGraph b = rich_graph();
  // Reach into the one component b has and populate `regions` directly
  // (bypassing the public API, since none exists to do this) to prove the
  // byte stream is unaffected either way.
  const_cast<fuse::ComponentDecl&>(b.component(fuse::ComponentId{0}))
      .regions.push_back(fuse::RegionId{0});
  EXPECT_EQ(serialize(a), serialize(b));
}

// ---- merge transforms (S2.1/S2.3) --------------------------------------

// The waist is version 3 with magic "FUSE": two changes from the inherited v2,
// both invisible to a reader that guessed. A v2 buffer must be REFUSED, not
// reinterpreted -- its unit records are two rationals shorter, so every field
// after the first unit would decode as plausible garbage.
TEST(Descriptor, InheritedVersionIsRefusedNotReinterpreted) {
  CouplingGraph g;
  const auto c = g.add_component({.name = "fluid", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  (void)g.add_field(c, {.name = "q", .units = u::kWatt});
  auto bytes = serialize(g);

  auto v2 = bytes;
  fuse::detail::Writer w;
  w.pod(fuse::kGraphMagic);
  w.pod(std::uint32_t{2});
  const auto prefix = w.take();
  std::copy(prefix.begin(), prefix.end(), v2.begin());
  const auto rejected = deserialize(v2);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code, "E-DESC-VERSION");

  auto foreign = bytes;
  foreign[0] = std::byte{0xFF};
  EXPECT_FALSE(deserialize(foreign).ok());
}

// A unit's affine part is part of the unit. Kilopascal and pascal have identical
// dimensions and must not hash alike; that is what the merge added scale for.
TEST(Descriptor, UnitScaleAndOffsetAreInTheCanonicalBytes) {
  const auto encode = [](fuse::Units units) {
    CouplingGraph g;
    const auto c = g.add_component({.name = "fluid", .ranks = {0, 1}, .interval = fuse::Rational{1}});
    (void)g.add_field(c, {.name = "p", .units = units});
    return serialize(g);
  };
  fuse::Units pascal{.length = -1, .mass = 1, .time = -2};
  fuse::Units kilopascal = pascal;
  kilopascal.scale = fuse::Rational{1000};
  fuse::Units celsius{.temperature = 1};
  celsius.offset = fuse::Rational{27315, 100};

  EXPECT_NE(encode(pascal), encode(kilopascal));
  EXPECT_NE(encode(fuse::Units{.temperature = 1}), encode(celsius));
  EXPECT_EQ(encode(kilopascal), encode(kilopascal));

  const auto round_tripped = deserialize(encode(kilopascal));
  ASSERT_TRUE(round_tripped.ok());
  EXPECT_EQ(round_tripped.value().field(fuse::FieldId{0}).units.scale, fuse::Rational{1000});
  EXPECT_TRUE(round_tripped.value().field(fuse::FieldId{0}).units.same_dimensions(pascal));
}

// REVIEW.md R-19: Units has no constructor of its own to gate a non-positive
// scale at (every call site aggregate-initializes it), so deserialize() is
// the narrowest choke point available for the one input in this scope that
// crosses a real trust boundary. Both the zero (annihilator) and negative
// (sign flip) cases must be refused, not silently admitted.
TEST(Descriptor, RejectsNonPositiveUnitScale) {
  const auto encode = [](fuse::Rational scale) {
    CouplingGraph g;
    const auto c = g.add_component({.name = "fluid", .ranks = {0, 1}, .interval = fuse::Rational{1}});
    fuse::Units bad{.length = -1, .mass = 1, .time = -2};
    bad.scale = scale;
    (void)g.add_field(c, {.name = "p", .units = bad});
    return serialize(g);
  };
  for (const fuse::Rational bad_scale : {fuse::Rational{0}, fuse::Rational{-1000}}) {
    const auto r = deserialize(encode(bad_scale));
    ASSERT_FALSE(r.ok()) << "scale " << bad_scale.to_string();
    EXPECT_EQ(r.error().code, "E-DESC-UNITS-SCALE");
  }
}

// REVIEW.md R-04: deserialize() used to read MeasureKind/AccelKind/FilterKind
// straight off the wire with no range check, so an out-of-range byte (a bit
// flip, truncation, or adversarial construction of a descriptor buffer --
// deserialize() is the ABI narrow waist every language binding funnels
// through) round-tripped into a CouplingGraph and only became undefined
// behavior later, at to_yaml()'s unchecked array index. This patches each
// enum byte to an out-of-range value in a known-good buffer, mirroring
// InheritedVersionIsRefusedNotReinterpreted's technique, and relies on the
// iteration region (with exactly one convergence measure, a one-character
// field name, and an empty merge table) being the LAST thing serialize()
// writes, so each enum byte's offset is computable from the end of the
// buffer without re-deriving serialize()'s full layout.
TEST(Descriptor, RejectsCorruptedIterationRegionEnums) {
  CouplingGraph g;
  const auto c = g.add_component(
      {.name = "fluid", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  (void)g.add_field(c, {.name = "q", .units = u::kWatt});
  fuse::IterationRegionDecl d;
  d.name = "outer";
  d.max_iterations = 10;
  d.convergence.add(fuse::ConvergenceMeasure{
      .kind = fuse::MeasureKind::Relative, .tolerance = 1e-6, .field = "q"});
  d.acceleration = fuse::AccelConfig{.kind = fuse::AccelKind::None,
                                     .filter = fuse::FilterKind::None};
  g.add_iteration_region(d);
  const auto bytes = serialize(g);

  // acceleration: kind(1) initial_relaxation(8) omega(8) reuse(4) filter(1)
  // limit(8) scale_fields(1) -- 31 bytes total, and the last thing written
  // (merge_count == 0).
  auto accel_kind = bytes;
  accel_kind[bytes.size() - 31] = std::byte{200};
  EXPECT_EQ(deserialize(accel_kind).error().code, "E-DESC-ENUM")
      << "corrupted acceleration kind byte was not refused";

  auto accel_filter = bytes;
  accel_filter[bytes.size() - 10] = std::byte{200};
  EXPECT_EQ(deserialize(accel_filter).error().code, "E-DESC-ENUM")
      << "corrupted acceleration filter byte was not refused";

  // measure.kind(1) tolerance(8) field(4 + 1, "q") + the 31-byte acceleration
  // block above == 44 bytes after it.
  auto measure_kind = bytes;
  measure_kind[bytes.size() - 45] = std::byte{200};
  EXPECT_EQ(deserialize(measure_kind).error().code, "E-DESC-ENUM")
      << "corrupted convergence measure kind byte was not refused";

  // The unpatched buffer must still read back cleanly, or the offsets above
  // are wrong rather than the deserializer.
  ASSERT_TRUE(deserialize(bytes).ok());
}

// ---- the waist freeze (S2.3) -------------------------------------------

// Every optional member at a NON-DEFAULT value. A round-trip over defaults
// proves nothing about a member: zero bytes written and zero read agree.


// A descriptor is only frozen if a round trip preserves every member it can
// carry -- not just the ones a default-constructed graph happens to set.
TEST(Descriptor, RoundTripPreservesEveryNonDefaultMember) {
  const auto original = fuse::testing::freeze_fixture();
  const auto bytes = fuse::serialize(original);
  const auto read_back = fuse::deserialize(bytes);
  ASSERT_TRUE(read_back.ok()) << read_back.error().message;

  // Structural equality is asserted through the bytes themselves: re-serializing
  // what we read must reproduce the same buffer. That covers every member at
  // once, including any added later, which a field-by-field comparison would not.
  EXPECT_EQ(fuse::serialize(read_back.value()), bytes);

  // And spot-check the members most likely to be silently dropped.
  const auto& f = read_back.value().field(fuse::FieldId{0});
  EXPECT_EQ(f.order, 4);
  EXPECT_EQ(f.components, 3);
  EXPECT_EQ(f.units.scale, fuse::Rational{1000});
  EXPECT_EQ(f.conservation, fuse::Conservation::Extensive);
  EXPECT_EQ(read_back.value().region(fuse::RegionId{0}).dimension, 2);
  EXPECT_EQ(read_back.value().component(fuse::ComponentId{1}).interval, fuse::Rational(1, 3));

  // The coupler-placement track's two new members: a Coupler-kind component
  // (freeze_fixture's "midpoint"), and the "flux" channel's remap_on naming
  // it.
  const auto midpoint = read_back.value().find_component("midpoint");
  ASSERT_TRUE(midpoint.has_value());
  EXPECT_EQ(read_back.value().component(*midpoint).kind, fuse::ComponentKind::Coupler);
  EXPECT_EQ(read_back.value().component(fuse::ComponentId{0}).kind,
            fuse::ComponentKind::Physics);
  bool found_flux = false;
  for (std::uint32_t i = 0; i < read_back.value().channel_count(); ++i) {
    const auto& ch = read_back.value().channel_at(fuse::ChannelId{i});
    if (ch.name != "flux") continue;
    found_flux = true;
    EXPECT_EQ(ch.remap.remap_on, *midpoint);
  }
  EXPECT_TRUE(found_flux);
}

// The frozen bytes. `test/golden/waist-v1.bin` is this descriptor's tag: a
// change to it is an ABI change and must come with a version bump, which is why
// the test compares bytes rather than re-deriving them.
// Every superseded frozen descriptor is kept, and kept REFUSED. The v4 case is
// the sharpest: v4 wrote a component's ranks as {first, count} and v5 writes
// {count, list}, so a v5 reader let loose on v4 bytes would read the rank
// COUNT as a first rank and produce a plausible, wrong placement -- values
// delivered to the wrong ranks, with nothing to indicate it. The version check
// is what turns that into an error.
TEST(Descriptor, SupersededFrozenBytesAreRefusedByThisReader) {
  for (const char* name : {"waist-v1.bin", "waist-v4.bin", "waist-v5.bin"}) {
    const std::filesystem::path old_golden =
        std::filesystem::path(FUSE_TEST_GOLDEN_DIR) / name;
    ASSERT_TRUE(std::filesystem::exists(old_golden)) << old_golden;
    std::ifstream in(old_golden, std::ios::binary);
    const std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
    std::vector<std::byte> bytes(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) bytes[i] = static_cast<std::byte>(raw[i]);
    const auto r = deserialize(bytes);
    ASSERT_FALSE(r.ok()) << name;
    EXPECT_EQ(r.error().code, "E-DESC-VERSION") << name;
  }
}

TEST(Descriptor, TheFrozenV1BytesAreRefusedByThisReader) {
  const std::filesystem::path old_golden =
      std::filesystem::path(FUSE_TEST_GOLDEN_DIR) / "waist-v1.bin";
  ASSERT_TRUE(std::filesystem::exists(old_golden)) << old_golden;
  std::ifstream in(old_golden, std::ios::binary);
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
  std::vector<std::byte> bytes(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) bytes[i] = static_cast<std::byte>(raw[i]);

  const auto r = deserialize(bytes);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-VERSION");
}

TEST(Descriptor, MatchesTheFrozenWaistBytes) {
  const auto bytes = fuse::serialize(fuse::testing::freeze_fixture());
  const std::filesystem::path golden =
      std::filesystem::path(FUSE_TEST_GOLDEN_DIR) / "waist-v6.bin";
  ASSERT_TRUE(std::filesystem::exists(golden)) << golden;

  std::ifstream in(golden, std::ios::binary);
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
  std::vector<std::byte> expected(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) expected[i] = static_cast<std::byte>(raw[i]);

  EXPECT_EQ(bytes, expected)
      << "the canonical descriptor changed. That is an ABI change: bump kGraphVersion, "
         "regenerate the golden file under the new version's name with "
         "fuse_write_golden_descriptor, and keep the old one.";
}
