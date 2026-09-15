#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/graph/descriptor.hpp>

using soma::AccumOp;
using soma::ComponentDecl;
using soma::Conservation;
using soma::CouplingGraph;
using soma::deserialize;
using soma::FieldSpec;
using soma::Intent;
using soma::MeshDecl;
using soma::Rational;
using soma::Region;
using soma::RemapFlags;
using soma::TemporalBasis;
namespace u = soma::units;

namespace {

// Exercises every field the descriptor must carry, including the awkward ones:
// rational intervals, spacetime specs, a lag, and a mismatch reason.
CouplingGraph rich_graph() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "mesh_a", .source = "a.h5m"});
  const auto r = g.add_region(Region{.name = "iface", .mesh = m,
                                     .entity_set = "ss1", .dimension = 2});
  const auto a = g.add_component(ComponentDecl{.name = "A",
                                               .ranks = {.first = 0, .count = 3},
                                               .interval = Rational{1, 3}});
  const auto b = g.add_component(ComponentDecl{.name = "B",
                                               .ranks = {.first = 3, .count = 5},
                                               .interval = Rational{1, 7}});
  const auto s = g.add_field(a, FieldSpec{.name = "flux", .region = r,
                                          .disc = soma::Discretization::DiscontinuousGLL,
                                          .order = 3, .components = 3,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Provides});
  const auto t = g.add_field(b, FieldSpec{.name = "flux_in", .region = r,
                                          .disc = soma::Discretization::FV,
                                          .components = 3,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Requires});
  const auto ch = g.add_channel("A->B", s, t);
  g.channel(ch)
      .accumulate(AccumOp::Integral, soma::Window::from(a))
      .remap(RemapFlags::Conservative | RemapFlags::HighOrder)
      .remap_method("mono3")
      .spacetime(TemporalBasis::DG, 2, soma::Window::explicit_length(Rational{2, 5}))
      .relax(0.35)
      .extrapolate(2)
      .lag(0)
      .allow_time_level_mismatch("verified by hand");
  g.add_iteration_region(soma::IterationRegionDecl{
      .name = "outer", .channels = {ch}, .max_iterations = 25});
  return g;
}

}  // namespace

TEST(Descriptor, HeaderHasMagicAndVersion) {
  const auto bytes = serialize(rich_graph());
  ASSERT_GE(bytes.size(), 8u);
  std::uint32_t magic = 0, version = 0;
  std::memcpy(&magic, bytes.data(), 4);
  std::memcpy(&version, bytes.data() + 4, 4);
  EXPECT_EQ(magic, soma::kGraphMagic);
  EXPECT_EQ(version, soma::kGraphVersion);
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
  const auto& ch = g.channel_at(soma::ChannelId{0});
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
  EXPECT_EQ(g.iteration_region(soma::IterationRegionIdx{0}).max_iterations, 25);
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
  const_cast<soma::ComponentDecl&>(b.component(soma::ComponentId{0}))
      .regions.push_back(soma::RegionId{0});
  EXPECT_EQ(serialize(a), serialize(b));
}
