// The bridge between the graph's FieldSpec and the semantic model the
// compatibility stages evaluate. Its job is to state what the graph declared and
// NOTHING more: an axis the graph cannot express must stay Unspecified, because
// a defaulted axis passes a check the user never made.
#include "fuse/model/connect.hpp"

#include <gtest/gtest.h>

namespace {

using namespace fuse;

TEST(Connect, DiscretizationMapsToWhereTheValueActuallyLives) {
  EXPECT_EQ(to_semantics({.name = "p", .disc = Discretization::FV}).location,
            SemanticLocation::Element);

  // Order 1 GLL coefficients ARE vertex values; higher order ones are not, and
  // calling them vertex values is what makes a transfer silently wrong.
  EXPECT_EQ(to_semantics({.name = "T", .disc = Discretization::ContinuousGLL, .order = 1}).location,
            SemanticLocation::Vertex);
  EXPECT_EQ(to_semantics({.name = "T", .disc = Discretization::ContinuousGLL, .order = 4}).location,
            SemanticLocation::FiniteElementCoefficient);
  EXPECT_EQ(
      to_semantics({.name = "T", .disc = Discretization::DiscontinuousGLL, .order = 3}).location,
      SemanticLocation::FiniteElementCoefficient);

  // A point cloud is none of the three. Saying nothing is correct.
  EXPECT_EQ(to_semantics({.name = "x", .disc = Discretization::PointCloud}).location,
            SemanticLocation::Unspecified);
}

TEST(Connect, ContinuityIsClaimedOnlyWhereTheDiscretizationImpliesIt) {
  EXPECT_EQ(to_semantics({.name = "T", .disc = Discretization::ContinuousGLL}).continuity,
            SemanticContinuity::Continuous);
  EXPECT_EQ(to_semantics({.name = "T", .disc = Discretization::DiscontinuousGLL}).continuity,
            SemanticContinuity::Discontinuous);
  // Finite volume is neither continuous nor discontinuous in this sense.
  EXPECT_EQ(to_semantics({.name = "p", .disc = Discretization::FV}).continuity,
            SemanticContinuity::Unspecified);
}

TEST(Connect, AxesTheGraphCannotStateStayUnspecified) {
  const FieldSemantics s = to_semantics({.name = "q"});
  EXPECT_EQ(s.frame, SemanticFrame::Unspecified);
  EXPECT_EQ(s.temporal, SemanticTemporal::InstantaneousCommitted)
      << "the only defaulted axis, and it is the semantic model's own default";
  EXPECT_EQ(s.id, "q");
}

TEST(Connect, ShapeFollowsTheComponentCount) {
  EXPECT_EQ(to_semantics({.name = "s", .components = 1}).shape, SemanticShape::Scalar);
  EXPECT_EQ(to_semantics({.name = "v", .components = 3}).shape, SemanticShape::Vector);
  EXPECT_EQ(to_semantics({.name = "t", .components = 9}).shape, SemanticShape::Tensor2);
}

TEST(Connect, IntentBecomesRoleAndExtensiveFieldsRequireConservation) {
  EXPECT_EQ(to_semantics({.name = "q", .intent = Intent::Requires}).role, SemanticRole::Input);
  EXPECT_EQ(to_semantics({.name = "q", .intent = Intent::Provides}).role, SemanticRole::Prognostic);

  EXPECT_EQ(required_conservation({.name = "q", .conservation = Conservation::Extensive}),
            ConservationGuarantee::GlobalConservative);
  EXPECT_EQ(required_conservation({.name = "T", .conservation = Conservation::Intensive}),
            ConservationGuarantee::None);
}

// BM-1 (review 2026-10-11): `declared_transfer()` is what `check_channel()`
// used to replace with a hardcoded `TransferKind::Copy`. These tests pin
// its own contract directly, against a channel's real `RemapSpec` and
// region, independent of `check_connection()`'s stages (those are
// `Compatibility.ConservationRules`'s job, test/model/compatibility_test.cpp).
namespace {

// Two one-field components on their own region by default; a test can move
// `dst`'s region to `r_a` to build a same-region pair, or leave it on `r_b`
// for a region-changing one.
struct TransferFixture {
  CouplingGraph g;
  ChannelId ch;

  explicit TransferFixture(bool same_region) {
    const auto m_a = g.add_mesh(MeshDecl{.name = "m_a", .source = "generated:line:4"});
    const auto m_b = g.add_mesh(MeshDecl{.name = "m_b", .source = "generated:line:4"});
    const auto r_a = g.add_region(Region{.name = "r_a", .mesh = m_a, .dimension = 1});
    const auto r_b = same_region ? r_a
                                 : g.add_region(Region{.name = "r_b", .mesh = m_b, .dimension = 1});
    const auto a = g.add_component(ComponentDecl{.name = "a", .ranks = RankSet{0, 1}, .interval = Rational{1}});
    const auto b = g.add_component(ComponentDecl{.name = "b", .ranks = RankSet{0, 1}, .interval = Rational{1}});
    const auto src = g.add_field(a, FieldSpec{.name = "out", .region = r_a, .intent = Intent::Provides});
    const auto dst = g.add_field(b, FieldSpec{.name = "in", .region = r_b, .intent = Intent::Requires});
    ch = g.add_channel("c", src, dst);
  }
};

}  // namespace

TEST(Connect, DeclaredTransferIsCopyOnlyForNoRemapAndMatchingRegion) {
  TransferFixture f(/*same_region=*/true);
  const auto transfer = declared_transfer(f.g, f.g.channel_at(f.ch));
  EXPECT_EQ(transfer.kind, TransferKind::Copy);
  EXPECT_EQ(transfer.conservation, ConservationGuarantee::None);
}

TEST(Connect, DeclaredTransferIsInterpolationWhenTheRegionChangesWithNoRemapFlags) {
  // lower()'s own `region_changes` condition (src/ir/rules_channel.cpp)
  // emits a real Remap for exactly this shape even with no remap flags at
  // all, so a `Copy` verdict here would be a declaration that disagrees
  // with what actually runs.
  TransferFixture f(/*same_region=*/false);
  const auto transfer = declared_transfer(f.g, f.g.channel_at(f.ch));
  EXPECT_EQ(transfer.kind, TransferKind::Interpolation);
  EXPECT_EQ(transfer.conservation, ConservationGuarantee::None);
}

TEST(Connect, DeclaredTransferIsInterpolationForANonConservativeRemapFlag) {
  TransferFixture f(/*same_region=*/true);
  f.g.channel(f.ch).remap(RemapFlags::Monotone);
  const auto transfer = declared_transfer(f.g, f.g.channel_at(f.ch));
  EXPECT_EQ(transfer.kind, TransferKind::Interpolation);
  EXPECT_EQ(transfer.conservation, ConservationGuarantee::None);
}

TEST(Connect, DeclaredTransferIsConservativeRemapOnlyWhenTheChannelDeclaresIt) {
  TransferFixture f(/*same_region=*/false);
  f.g.channel(f.ch).conservative();
  const auto transfer = declared_transfer(f.g, f.g.channel_at(f.ch));
  EXPECT_EQ(transfer.kind, TransferKind::ConservativeRemap);
  EXPECT_EQ(transfer.conservation, ConservationGuarantee::GlobalConservative);
}

}  // namespace
