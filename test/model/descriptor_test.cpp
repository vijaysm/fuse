#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <type_traits>
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

// ComponentDecl::regions is harmless today only
// because nothing in the public API populates it (there is no
// add_region_to_component()-style call). Since structurally_equal() is
// defined in terms of serialize(), it cannot detect a field neither side
// writes -- so this pins the choice explicitly: populating `regions`
// directly on a graph must not change what serialize() produces. If the graph
// API starts populating this field, this test should be
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

// ---- merge transforms --------------------------------------

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

// Units has no constructor of its own to gate a non-positive
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

// deserialize() must range-check MeasureKind/AccelKind/FilterKind read off
// the wire: an out-of-range byte (a bit flip, truncation, or adversarial
// construction of a descriptor buffer -- deserialize() is the ABI narrow
// waist every language binding funnels through) would otherwise round-trip
// into a CouplingGraph and only become undefined behavior later, at
// to_yaml()'s unchecked array index. This patches each
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
  // limit(8) scale_fields(1) -- 31 bytes total. v7 adds a 1-byte
  // window_scope presence flag right after it (0 here: `d` sets none), so
  // the acceleration block sits 32 bytes before the END OF THAT BLOCK's
  // OWN SUCCESSOR. v14 (descriptor.hpp's kGraphVersion history) appends a
  // 164-byte block of its own right after the window_scope flag (two
  // capability bytes, then NewtonConfig's 28 fields: 2 enum bytes, 15
  // doubles, 5 int32s, 2 uint64s, 1 bool byte, 1 enum byte for v13's own
  // `substrate`, 2 more enum bytes for v14's own `norm`/`residual_path` --
  // 2+120+20+16+1+1+2=162, plus the 2 capability bytes = 164), so the
  // acceleration block now sits 32+164 = 196 bytes before the end of this
  // region's record (merge_count == 0, the last thing written).
  auto accel_kind = bytes;
  accel_kind[bytes.size() - 196] = std::byte{200};
  EXPECT_EQ(deserialize(accel_kind).error().code, "E-DESC-ENUM")
      << "corrupted acceleration kind byte was not refused";

  auto accel_filter = bytes;
  accel_filter[bytes.size() - 175] = std::byte{200};
  EXPECT_EQ(deserialize(accel_filter).error().code, "E-DESC-ENUM")
      << "corrupted acceleration filter byte was not refused";

  // measure.kind(1) tolerance(8) field(4 + 1, "q") + the 32-byte
  // acceleration-plus-window-scope-flag block + the 164-byte v14 Newton
  // block above == 45 + 164 = 209 bytes after it.
  auto measure_kind = bytes;
  measure_kind[bytes.size() - 210] = std::byte{200};
  EXPECT_EQ(deserialize(measure_kind).error().code, "E-DESC-ENUM")
      << "corrupted convergence measure kind byte was not refused";

  // v12's own accelerator-kind bound: ordinal 8 is one past AccelKind::
  // Newton (7), the first value still refused after the widened check.
  // Neither v13 nor v14 moved this byte (neither changed AccelKind at all).
  auto newton_kind_out_of_range = bytes;
  newton_kind_out_of_range[bytes.size() - 196] = std::byte{8};
  EXPECT_EQ(deserialize(newton_kind_out_of_range).error().code, "E-DESC-ENUM")
      << "an accelerator kind one past AccelKind::Newton was not refused";

  // v13's own bound check: NewtonConfig::substrate used to be the LAST
  // byte of the per-region record; v14 appends `norm` then
  // `residual_path` after it, so substrate now sits 3 bytes from the end
  // (merge_count == 0 here). Ordinal 2 is one past NewtonSubstrate::Petsc
  // (1).
  auto newton_substrate_out_of_range = bytes;
  newton_substrate_out_of_range[bytes.size() - 3] = std::byte{2};
  EXPECT_EQ(deserialize(newton_substrate_out_of_range).error().code, "E-DESC-ENUM")
      << "a newton_config.substrate one past NewtonSubstrate::Petsc was not refused";

  // v14's own bound checks: `norm` is the second-to-last byte of the
  // per-region record, `residual_path` the last. Ordinal 2 is one past
  // NewtonNorm::MeasureWeighted (1) / NewtonResidualPath::Direct (1).
  auto newton_norm_out_of_range = bytes;
  newton_norm_out_of_range[bytes.size() - 2] = std::byte{2};
  EXPECT_EQ(deserialize(newton_norm_out_of_range).error().code, "E-DESC-ENUM")
      << "a newton_config.norm one past NewtonNorm::MeasureWeighted was not refused";

  auto newton_residual_path_out_of_range = bytes;
  newton_residual_path_out_of_range[bytes.size() - 1] = std::byte{2};
  EXPECT_EQ(deserialize(newton_residual_path_out_of_range).error().code, "E-DESC-ENUM")
      << "a newton_config.residual_path one past NewtonResidualPath::Direct was not refused";

  // The unpatched buffer must still read back cleanly, or the offsets above
  // are wrong rather than the deserializer.
  ASSERT_TRUE(deserialize(bytes).ok());
}

// ---- the waist freeze -------------------------------------------

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

  // The coupler-placement members: a Coupler-kind component
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

  // v7's reason for the version bump: IterationRegionDecl::window_scope.
  // v12 adds a second region (see below), so this fixture now carries two.
  ASSERT_EQ(read_back.value().iteration_region_count(), 2u);
  const auto& region =
      read_back.value().iteration_region(fuse::IterationRegionIdx{0});
  EXPECT_EQ(region.max_iterations, 12);
  ASSERT_TRUE(region.window_scope.has_value());
  const auto& scope = *region.window_scope;
  EXPECT_FALSE(scope.window_component.has_value());
  ASSERT_TRUE(scope.explicit_interval.has_value());
  EXPECT_EQ(*scope.explicit_interval, fuse::Rational(3, 10));
  EXPECT_EQ(scope.order, 3);
  ASSERT_EQ(scope.policies.size(), 2u);
  EXPECT_EQ(scope.policies[0].component, "fluid");
  EXPECT_EQ(scope.policies[0].policy, fuse::DtPolicy::Fixed);
  EXPECT_EQ(scope.policies[1].component, "solid");
  EXPECT_EQ(scope.policies[1].policy, fuse::DtPolicy::Adjustable);
  // v8's reason for the version bump: the default is Jacobi, so a
  // fixture that round-trips this as GaussSeidel proves the byte is
  // read, not just defaulted.
  EXPECT_EQ(scope.iteration, fuse::WaveformIteration::GaussSeidel);

  // v11's reason for the version bump:
  // FieldSpec::hdf5_path and LaunchMode::LowFive.
  ASSERT_TRUE(f.hdf5_path.has_value());
  EXPECT_EQ(*f.hdf5_path, "fluid_out.h5/grid/heat_flux");
  const auto probe = read_back.value().find_component("probe");
  ASSERT_TRUE(probe.has_value());
  EXPECT_EQ(read_back.value().component(*probe).launch.mode,
            fuse::LaunchMode::LowFive);

  // v12's reason for the version bump (the native Jacobian-free
  // Newton-Krylov coupler): the second region's acceleration.kind and its
  // NewtonConfig. Spot-check a value of each serialized type (the two enum
  // bytes, a double from each of the three groups the struct's comment
  // documents reading only for the selected line-search kind, an int, a
  // size_t, and the final bool byte) so a reader that defaults any one of
  // them instead of decoding it is caught here, not just by the frozen
  // bytes matching themselves.
  const auto& newton_region =
      read_back.value().iteration_region(fuse::IterationRegionIdx{1});
  EXPECT_EQ(newton_region.name, "newton");
  ASSERT_EQ(newton_region.acceleration.kind, fuse::AccelKind::Newton);
  EXPECT_TRUE(newton_region.requires_residual_capability);
  EXPECT_TRUE(newton_region.requires_linearization_capability);
  const auto& cfg = newton_region.newton_config;
  EXPECT_EQ(cfg.line_search_kind, fuse::LineSearchKind::CP);
  EXPECT_EQ(cfg.forcing_choice, fuse::EisenstatWalkerChoice::Choice1);
  EXPECT_DOUBLE_EQ(cfg.eta_min, 1e-6);
  EXPECT_DOUBLE_EQ(cfg.secant_tol, 1e-9);
  EXPECT_DOUBLE_EQ(cfg.damping, 0.75);
  EXPECT_EQ(cfg.max_nonlinear_iterations, 9);
  EXPECT_EQ(cfg.gmres_max_iterations, 500u);
  EXPECT_TRUE(cfg.use_block_jacobi_preconditioner);
  // v13's reason for the version bump:
  // NewtonConfig::substrate, appended at the end of the block.
  EXPECT_EQ(cfg.substrate, fuse::NewtonSubstrate::Petsc);
  // v14's reason for the version bump:
  // NewtonConfig::norm/::residual_path, appended after substrate.
  EXPECT_EQ(cfg.norm, fuse::NewtonNorm::MeasureWeighted);
  EXPECT_EQ(cfg.residual_path, fuse::NewtonResidualPath::Direct);
}

// A region with no window_scope (std::nullopt, the default -- every region
// declared before this member existed) must round-trip to exactly that, not
// to an accidentally-empty CouplingWindowScope.
TEST(Descriptor, ANonWindowedIterationRegionRoundTripsWithNoScope) {
  const auto g = deserialize(serialize(rich_graph())).value();
  ASSERT_EQ(g.iteration_region_count(), 1u);
  EXPECT_FALSE(
      g.iteration_region(fuse::IterationRegionIdx{0}).window_scope.has_value());
}

// The wire byte for an unset window_scope is a single zero -- corrupting it
// to an out-of-range DtPolicy byte must be refused like every other enum on
// the wire, not silently admitted.
TEST(Descriptor, RejectsCorruptedWindowScopeDtPolicy) {
  CouplingGraph g;
  const auto c = g.add_component(
      {.name = "fluid", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  const auto f = g.add_field(c, {.name = "q", .units = u::kWatt});
  const auto ch = g.add_channel("self", f, f);
  g.iterate({ch}).name("w");
  {
    fuse::CouplingWindowScope scope;
    scope.explicit_interval = fuse::Rational{1};
    scope.policies.push_back({.component = "fluid", .policy = fuse::DtPolicy::Fixed});
    const_cast<fuse::IterationRegionDecl&>(
        g.iteration_region(fuse::IterationRegionIdx{0}))
        .window_scope = std::move(scope);
  }
  auto bytes = serialize(g);
  // The policy byte used to be the second-to-last byte serialize() writes
  // for this graph: no merges, one policy entry (component name "fluid" =
  // 4 bytes + 4-byte length prefix, then the policy byte), then v8's own
  // trailing WaveformIteration byte after it. v14 appends its own
  // 164-byte block (two capability bytes, then NewtonConfig's 28 fields --
  // see RejectsCorruptedIterationRegionEnums for the field tally) AFTER
  // the window_scope block, unconditionally, so that byte is now 164
  // bytes earlier than the new end of the buffer: 2 + 164 = 166.
  bytes[bytes.size() - 166] = std::byte{200};
  const auto r = deserialize(bytes);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-ENUM");
}

// Same shape, one byte later: the trailing WaveformIteration byte (v8) used
// to be the actual last byte serialize() writes for this graph; v14's own
// 164-byte block (see RejectsCorruptedWindowScopeDtPolicy) now follows it
// unconditionally, so it must be refused 165 bytes before the new end, not
// at the new last byte.
TEST(Descriptor, RejectsCorruptedWindowScopeIteration) {
  CouplingGraph g;
  const auto c = g.add_component(
      {.name = "fluid", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  const auto f = g.add_field(c, {.name = "q", .units = u::kWatt});
  const auto ch = g.add_channel("self", f, f);
  g.iterate({ch}).name("w");
  {
    fuse::CouplingWindowScope scope;
    scope.explicit_interval = fuse::Rational{1};
    scope.policies.push_back({.component = "fluid", .policy = fuse::DtPolicy::Fixed});
    const_cast<fuse::IterationRegionDecl&>(
        g.iteration_region(fuse::IterationRegionIdx{0}))
        .window_scope = std::move(scope);
  }
  auto bytes = serialize(g);
  bytes[bytes.size() - 165] = std::byte{200};
  const auto r = deserialize(bytes);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-ENUM");
}

// descriptor v9: a component's launch mode byte, range-checked like every
// other enum byte on this wire -- set directly via an out-of-range static_cast rather than by
// searching serialized bytes for the right offset, since LaunchMode (like
// every enum class here) accepts an explicit cast to any underlying value.
TEST(Descriptor, RejectsCorruptedLaunchMode) {
  CouplingGraph g;
  fuse::ComponentDecl c{.name = "fluid", .ranks = {0, 1}, .interval = fuse::Rational{1}};
  c.launch.mode = static_cast<fuse::LaunchMode>(200);
  g.add_component(std::move(c));
  const auto bytes = serialize(g);
  const auto r = deserialize(bytes);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-ENUM");
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
  for (const char* name : {"waist-v1.bin", "waist-v4.bin", "waist-v5.bin",
                          "waist-v6.bin", "waist-v7.bin", "waist-v8.bin",
                          "waist-v9.bin", "waist-v10.bin", "waist-v11.bin",
                          "waist-v12.bin", "waist-v13.bin", "waist-v14.bin"}) {
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

// ---- the descriptor byte-order guarantee --------------------------------
//
// The checks below are written against versions 14 (superseded) and 15
// (current).

// The ONLY difference, on this (little-endian) host, between the frozen
// v14 golden's bytes and what this build's serialize() now writes for the
// IDENTICAL fixture graph is the version word -- proving kGraphVersion 15
// changed no field's width or position, only (a) the version tag and (b)
// the GUARANTEE that the bytes no longer depend on the writing host's own
// byte order (invisible here precisely because this host is little-endian,
// exactly as descriptor.hpp's own v15 history entry and bytes.hpp's header
// comment both state).
TEST(Descriptor, V14AndV15BytesOfTheSameGraphDifferOnlyInTheVersionWord) {
  const auto v15_bytes = fuse::serialize(fuse::testing::freeze_fixture());

  const std::filesystem::path v14_golden =
      std::filesystem::path(FUSE_TEST_GOLDEN_DIR) / "waist-v14.bin";
  ASSERT_TRUE(std::filesystem::exists(v14_golden)) << v14_golden;
  std::ifstream in(v14_golden, std::ios::binary);
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
  std::vector<std::byte> v14_bytes(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) v14_bytes[i] = static_cast<std::byte>(raw[i]);

  ASSERT_EQ(v14_bytes.size(), v15_bytes.size())
      << "v14 and v15 bytes differ in length -- the fix changed a field's width";
  std::size_t first_diff = v14_bytes.size();
  std::size_t diff_count = 0;
  for (std::size_t i = 0; i < v14_bytes.size(); ++i) {
    if (v14_bytes[i] != v15_bytes[i]) {
      if (diff_count == 0) first_diff = i;
      ++diff_count;
    }
  }
  EXPECT_EQ(diff_count, 1u) << "exactly one byte should differ between v14 and v15 "
                               "descriptors of the same graph on a little-endian host";
  EXPECT_EQ(first_diff, 4u) << "the differing byte should be the version word's own "
                               "first byte (offset 4, right after the 4-byte magic)";

  // Patch v15's version word down to 14 and confirm the result IS the
  // frozen v14 golden, byte for byte.
  auto v15_relabeled_as_v14 = v15_bytes;
  fuse::detail::Writer w14;
  w14.pod(fuse::kGraphMagic);
  w14.pod(std::uint32_t{14});
  const auto prefix_v14 = w14.take();
  std::copy(prefix_v14.begin(), prefix_v14.end(), v15_relabeled_as_v14.begin());
  EXPECT_EQ(v15_relabeled_as_v14, v14_bytes)
      << "v15 bytes, relabeled back to version 14, do not match the frozen v14 golden";
}

// Decode the superseded version's bytes and the current version's bytes
// for the SAME graph, and show the two decode to the SAME graph. deserialize() itself refuses a v14 buffer by
// version (SupersededFrozenBytesAreRefusedByThisReader, above) -- by
// design, not a limitation this checker works around by inventing a
// second reader. It instead relabels the frozen v14 payload's version
// word to 15 (the EXACT inverse of the previous test's own relabel) and
// hands the result to the one production deserialize(): the previous
// test already proved that relabeling is the ONLY byte that differs
// between the two, so decoding the relabeled v14 payload decodes the
// SAME field bytes freeze_fixture()'s own v15 serialize() produces.
TEST(Descriptor, V14AndV15BytesOfTheSameGraphDecodeIdentically) {
  const auto v15_bytes = fuse::serialize(fuse::testing::freeze_fixture());

  const std::filesystem::path v14_golden =
      std::filesystem::path(FUSE_TEST_GOLDEN_DIR) / "waist-v14.bin";
  ASSERT_TRUE(std::filesystem::exists(v14_golden)) << v14_golden;
  std::ifstream in(v14_golden, std::ios::binary);
  const std::vector<char> raw((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
  std::vector<std::byte> v14_bytes(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) v14_bytes[i] = static_cast<std::byte>(raw[i]);
  ASSERT_EQ(v14_bytes.size(), v15_bytes.size());

  auto v14_relabeled_as_v15 = v14_bytes;
  std::copy(v15_bytes.begin(), v15_bytes.begin() + 8, v14_relabeled_as_v15.begin());

  const auto from_v14 = fuse::deserialize(v14_relabeled_as_v15);
  ASSERT_TRUE(from_v14.ok()) << from_v14.error().message;
  const auto from_v15 = fuse::deserialize(v15_bytes);
  ASSERT_TRUE(from_v15.ok()) << from_v15.error().message;
  EXPECT_TRUE(structurally_equal(from_v14.value(), from_v15.value()))
      << "the v14 golden's payload and the v15 build's own serialize() output "
         "decode to different graphs for the identical fixture";
}

// There is no big-endian host to build or run on here. This test instead
// shows, by construction rather than by executing on real big-endian
// hardware, that `fuse::detail::Writer::pod` (routed through
// `bytes::put_integer`/`bytes::put_signed`) cannot depend on the host's
// native byte order the way a memcpy-based `pod()` would:
// `bytes::put_integer`/`put_signed` build their output from the VALUE through right-shifts and masks -- an operation
// the C++ standard defines on the abstract integer value, not on its
// storage layout -- so a second encoder written the SAME way (shifts, no
// `reinterpret_cast`, no `memcpy`) must agree with `Writer::pod` on every
// host, including one not reachable here. The BYTE-SWAPPING WRITER
// STUB stands in for a memcpy-based `pod()` AS IF it ran on
// a big-endian host: a big-endian host's native memcpy of a value is the
// byte-reverse of this (little-endian) host's own native memcpy of the
// same value, so reversing this host's memcpy output reproduces exactly
// a memcpy-based `pod()` would have written there.
namespace {
template <class T>
std::vector<std::byte> shift_based_reference_encoding(T v) {
  // The canonical encoding, rebuilt here independently of bytes.hpp so
  // this test does not simply check bytes::put_integer against itself:
  // pure integer arithmetic on the VALUE, touching no memory layout.
  std::vector<std::byte> out;
  if constexpr (std::is_floating_point_v<T>) {
    const auto bits = std::bit_cast<std::uint64_t>(v);
    for (std::size_t i = 0; i != sizeof(T); ++i)
      out.push_back(static_cast<std::byte>((bits >> (8 * i)) & 0xffU));
  } else {
    using U = std::make_unsigned_t<T>;
    const auto bits = static_cast<std::uint64_t>(static_cast<U>(v));
    for (std::size_t i = 0; i != sizeof(T); ++i)
      out.push_back(static_cast<std::byte>((bits >> (8 * i)) & 0xffU));
  }
  return out;
}

template <class T>
std::vector<std::byte> memcpy_based_stub_as_if_big_endian(T v) {
  // What a memcpy-based pod() would have written
  // on a big-endian host: the byte-reverse of what it writes HERE (this
  // host's own native memcpy), because flipping every bit of a two's-
  // complement integer's or an IEEE-754 float's byte ORDER is exactly
  // what swapping host endianness does to memcpy's output.
  std::array<std::byte, sizeof(T)> native{};
  std::memcpy(native.data(), &v, sizeof(T));
  std::vector<std::byte> reversed(native.rbegin(), native.rend());
  return reversed;
}
}  // namespace

TEST(Descriptor, PodBytesNoLongerDependOnHostByteOrderViaSwapStub) {
  const std::uint8_t  u8  = 0xAB;
  const std::int32_t  i32 = -100000;
  const std::uint32_t u32 = 0x01020304u;
  const std::uint64_t u64 = 0x0102030405060708ULL;
  const double        f64 = 1.5;  // exact in binary: 0x3FF8000000000000

  const auto check = [](auto value, const char* name) {
    fuse::detail::Writer w;
    w.pod(value);
    const auto actual = w.take();
    const auto reference = shift_based_reference_encoding(value);
    EXPECT_EQ(actual, reference)
        << name << ": Writer::pod no longer matches a pure value-arithmetic "
                   "(host-layout-independent) encoding";

    if (sizeof(value) > 1) {
      const auto old_big_endian_host_would_have_written =
          memcpy_based_stub_as_if_big_endian(value);
      EXPECT_NE(actual, old_big_endian_host_would_have_written)
          << name << ": the byte-swap stub must reproduce the SUPERSEDED bug "
                     "(a big-endian host disagreeing with this one) to be a "
                     "meaningful contrast against the fixed Writer::pod";
    }
  };
  check(u8, "uint8_t");
  check(i32, "int32_t");
  check(u32, "uint32_t");
  check(u64, "uint64_t");
  check(f64, "double");
}

TEST(Descriptor, MatchesTheFrozenWaistBytes) {
  const auto bytes = fuse::serialize(fuse::testing::freeze_fixture());
  const std::filesystem::path golden =
      std::filesystem::path(FUSE_TEST_GOLDEN_DIR) / "waist-v15.bin";
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

namespace {
// A graph small enough that a channel record's offset from the END of the
// buffer is computable by hand (no meshes, no regions, no iteration
// regions, no merges -- so the one channel is the LAST thing
// serialize() writes), mirroring RejectsCorruptedIterationRegionEnums's
// own technique.
CouplingGraph minimal_channel_graph() {
  CouplingGraph g;
  const auto c = g.add_component(
      {.name = "c", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  const auto q = g.add_field(c, {.name = "q"});
  (void)g.add_channel("c", q, q);
  return g;
}
}  // namespace

// Four of the six range-checked enum bytes live in a channel record -- AccumOp, and Window::Kind twice
// (once per accumulate/spacetime window). Offsets computed by hand from
// `minimal_channel_graph()`'s own field sizes (name "c" => a 1-byte
// string, empty remap_method and time_level_mismatch_reason): name(5) +
// source(4) + target(4) + accum_op(1) + accum_window(21) +
// remap_flags(4) + remap_method(4) + remap_on(4) + spacetime_basis(1) +
// spacetime_degree(4) + spacetime_window(21) + relax_enabled(1) +
// relax_omega(8) + extrap_enabled(1) + extrap_order(4) + lag(4) +
// reason(4) = 95 bytes total, so accum_op sits 95-13=82 bytes before the
// end, accum_window's own kind byte 95-14=81, spacetime_basis 95-47=48,
// spacetime_window's own kind byte 95-52=43.
TEST(Descriptor, RejectsCorruptedChannelEnums) {
  const auto bytes = serialize(minimal_channel_graph());
  ASSERT_TRUE(deserialize(bytes).ok()) << "the unpatched buffer must read back cleanly, "
                                          "or the hand-computed offsets below are wrong";

  auto accum_op = bytes;
  accum_op[bytes.size() - 82] = std::byte{200};
  EXPECT_EQ(deserialize(accum_op).error().code, "E-DESC-ENUM")
      << "corrupted accumulate op byte was not refused";

  auto accum_window_kind = bytes;
  accum_window_kind[bytes.size() - 81] = std::byte{200};
  EXPECT_EQ(deserialize(accum_window_kind).error().code, "E-DESC-ENUM")
      << "corrupted accumulate window kind byte was not refused";

  auto spacetime_basis = bytes;
  spacetime_basis[bytes.size() - 48] = std::byte{200};
  EXPECT_EQ(deserialize(spacetime_basis).error().code, "E-DESC-ENUM")
      << "corrupted spacetime basis byte was not refused";

  auto spacetime_window_kind = bytes;
  spacetime_window_kind[bytes.size() - 43] = std::byte{200};
  EXPECT_EQ(deserialize(spacetime_window_kind).error().code, "E-DESC-ENUM")
      << "corrupted spacetime window kind byte was not refused";
}

// The remaining two of the six, Discretization and Conservation,
// plus Intent, all live in the field record, which in
// `minimal_channel_graph()` is immediately followed by the 95-byte channel record above (also the
// end of the buffer): field name "q" (len 1) => name(5) + owner(4) +
// region(4) + disc(1) + order(4) + components(4) + units(39) +
// conservation(1) + intent(1) + hdf5_presence(1) = 64 bytes, so disc
// sits 64+95-13=146 bytes before the end, conservation 64+95-61=98,
// intent 64+95-62=97.
TEST(Descriptor, RejectsCorruptedFieldEnums) {
  const auto bytes = serialize(minimal_channel_graph());

  auto disc = bytes;
  disc[bytes.size() - 146] = std::byte{200};
  EXPECT_EQ(deserialize(disc).error().code, "E-DESC-ENUM")
      << "corrupted discretization byte was not refused";

  auto conservation = bytes;
  conservation[bytes.size() - 98] = std::byte{200};
  EXPECT_EQ(deserialize(conservation).error().code, "E-DESC-ENUM")
      << "corrupted conservation byte was not refused";

  auto intent = bytes;
  intent[bytes.size() - 97] = std::byte{200};
  EXPECT_EQ(deserialize(intent).error().code, "E-DESC-ENUM")
      << "corrupted intent byte was not refused";
}

// An iteration-region channel id off the wire must be bounds-checked, like
// the merge table's field ids a few lines below it in deserialize()
// (E-DESC-MERGE); otherwise the first dereference is a raw
// std::vector::at() in build_iteration_regions()/has_unbroken_cycle(), a
// per-rank throw ahead of later collectives.
// Built directly through the raw API (add_iteration_region() has no
// bounds check of its own either) rather than through a byte patch, since the defect is in what
// deserialize() ACCEPTS, not in corrupting an otherwise-valid buffer.
TEST(Descriptor, RejectsOutOfRangeIterationRegionChannelId) {
  CouplingGraph g;
  const auto c = g.add_component(
      {.name = "c", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  const auto q = g.add_field(c, {.name = "q"});
  (void)g.add_channel("ch", q, q);  // one channel: id 0 is the only valid one
  fuse::IterationRegionDecl d;
  d.name = "outer";
  d.channels.push_back(fuse::ChannelId{99});
  g.add_iteration_region(std::move(d));

  const auto r = deserialize(serialize(g));
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-CHANNEL");
}

// deserialize() must not reserve() an attacker-controlled count before
// reading any of the data it describes -- a truncated buffer claiming a
// huge count would either fail cleanly with E-DESC-TRUNC (if the
// allocator happened to refuse) or throw a raw
// std::bad_alloc/std::length_error out of a Result-returning function.
// Patches the wire count in place (via memcpy, so this is endianness-
// agnostic regardless of the host's byte order) rather than hand-
// computing every preceding byte's value.
TEST(Descriptor, RejectsImpossiblyLargeComponentRankCount) {
  CouplingGraph g;
  (void)g.add_component(
      {.name = "c", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  auto bytes = serialize(g);
  ASSERT_TRUE(deserialize(bytes).ok());

  // Component record (name "c" len 1, 1 rank, no launch strings): name(5)
  // + n_ranks(4) + rank(4) + interval(16) + kind(1) + launch_mode(1) +
  // appnum(4) + is_acceptor(1) + port_file(4) + service_name(4) +
  // plugin_path(4) = 48 bytes, the whole trailing buffer (0 fields, 0
  // channels, 0 iteration regions, 0 merges). n_ranks sits at offset 5
  // within that record, i.e. 48-5=43 bytes before the end.
  const std::uint32_t huge = 0xFFFFFFF0u;
  std::memcpy(bytes.data() + (bytes.size() - 43), &huge, sizeof(huge));

  const auto r = deserialize(bytes);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-TRUNC");
}

// Reader::rational() must not accept and silently renormalize a
// negative-denominator or unreduced wire pair; it refuses at the same trust
// boundary Rational::from_record() refuses the identical shape on
// (E-BASE-TIME-DENOMINATOR/E-BASE-TIME-UNREDUCED). Patches a
// component's interval (the first Rational written after the header and
// the ranks table) to an unreduced pair (2/4) and to a negative
// denominator, checking both are now refused rather than silently
// normalized to 1/2 and -2/-4 => 1/2 respectively.
TEST(Descriptor, RejectsNonCanonicalWireRational) {
  CouplingGraph g;
  (void)g.add_component(
      {.name = "c", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  const auto bytes = serialize(g);
  ASSERT_TRUE(deserialize(bytes).ok());

  // interval sits right after name(5) + n_ranks(4) + rank(4) = 13 bytes
  // in, as two int64_t (num then den), 16 bytes total -- the whole
  // record is 48 bytes (see RejectsImpossiblyLargeComponentRankCount
  // above), so interval starts at offset 48-16-... : num at
  // 48-(13+16)=19 bytes from the end's start, i.e. bytes.size()-35 for
  // num and bytes.size()-27 for den.
  const auto patch_rational = [&](std::int64_t num, std::int64_t den) {
    auto patched = bytes;
    std::memcpy(patched.data() + (patched.size() - 35), &num, sizeof(num));
    std::memcpy(patched.data() + (patched.size() - 27), &den, sizeof(den));
    return patched;
  };

  {
    const auto unreduced = patch_rational(2, 4);
    const auto r = deserialize(unreduced);
    ASSERT_FALSE(r.ok()) << "an unreduced wire rational (2/4) was silently normalized";
    EXPECT_EQ(r.error().code, "E-DESC-TRUNC");
  }
  {
    const auto negative_denominator = patch_rational(-1, -2);
    const auto r = deserialize(negative_denominator);
    ASSERT_FALSE(r.ok()) << "a negative-denominator wire rational was silently normalized";
    EXPECT_EQ(r.error().code, "E-DESC-TRUNC");
  }
}

// The same unchecked-reserve shape as
// RejectsImpossiblyLargeComponentRankCount above, for the iteration
// region's own channel-id count. The region's "outer" name (len 5) and
// an empty channel list are the only fields patched around: header(36)
// + the component record(48) + the field record(64) + the channel
// record(95, all three sized exactly as in the tests above) + the
// region name string (4+5=9) = 252 bytes precede the 4-byte channel
// count this patches.
TEST(Descriptor, RejectsImpossiblyLargeIterationRegionChannelCount) {
  CouplingGraph g;
  const auto c = g.add_component(
      {.name = "c", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  const auto q = g.add_field(c, {.name = "q"});
  (void)g.add_channel("c", q, q);
  fuse::IterationRegionDecl d;
  d.name = "outer";
  g.add_iteration_region(std::move(d));

  auto bytes = serialize(g);
  ASSERT_TRUE(deserialize(bytes).ok());
  ASSERT_GE(bytes.size(), 256u);

  const std::uint32_t huge = 0xFFFFFFF0u;
  std::memcpy(bytes.data() + 252, &huge, sizeof(huge));

  const auto r = deserialize(bytes);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-TRUNC");
}
