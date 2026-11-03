// physics.fe_basis -- ported from KARMA's basis_test.C (the registry and
// metadata-consistency portions), plus the orientation-metadata tests,
// Prism/Pyramid registry coverage, and the regression for the
// valid_key() OR-bug.

#include "fuse_physics/fe/basis.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include <string>

#include <gtest/gtest.h>

namespace {

using fuse::physics::fe::Basis;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisKey;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::DuplicatePolicy;
using fuse::physics::fe::FieldBasisFamily;

// physics.fe_basis.AllBuiltinKeysSizesAndAssociationsAgree -- every
// (cell, order <= 2, continuity) key the registry seeds itself with
// (basis.cpp's BasisRegistry constructor) produces a Basis whose size,
// interpolation-point count, and dof-association count all agree (Basis's
// own constructor already enforces this; this test additionally checks
// the SIZE matches the cell's own node/constant count).
TEST(BasisTest, AllBuiltinKeysSizesAndAssociationsAgree) {
  const BasisRegistry registry;
  constexpr std::array cells{CellFamily::Edge,        CellFamily::Triangle, CellFamily::Quadrilateral,
                              CellFamily::Tetrahedron, CellFamily::Hexahedron, CellFamily::Prism,
                              CellFamily::Pyramid};
  constexpr std::array continuities{BasisContinuity::cG, BasisContinuity::dG, BasisContinuity::SE};
  for (const auto cell : cells)
    for (unsigned order = 0; order <= 2; ++order)
      for (const auto continuity : continuities) {
        if (order == 0 && continuity == BasisContinuity::cG) continue;
        const BasisKey key{cell, FieldBasisFamily::LagrangeNodal, order, continuity};
        ASSERT_TRUE(registry.supports(key));
        const Basis basis = registry.create(key);
        EXPECT_EQ(basis.key(), key);
        EXPECT_EQ(basis.interpolation_points().size(), basis.size());
        EXPECT_EQ(basis.dof_associations().size(), basis.size());
        if (order == 0) EXPECT_EQ(basis.size(), 1u);
      }
}

// physics.fe_basis.P0ConstantEvaluatesToOne -- an order-0 basis is a
// single, everywhere-constant function with a zero gradient (BuiltinBasis's
// own special case, basis.cpp).
TEST(BasisTest, P0ConstantEvaluatesToOne) {
  const BasisRegistry registry;
  const Basis basis = registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG});
  std::array<double, 1> values{};
  std::array<fuse::physics::Vector3, 1> gradients{};
  basis.evaluate(fuse::physics::Point(0.3, 0.6), values, gradients);
  EXPECT_DOUBLE_EQ(values[0], 1.0);
  EXPECT_DOUBLE_EQ(gradients[0].x, 0.0);
  EXPECT_DOUBLE_EQ(gradients[0].y, 0.0);
}

// physics.fe_basis.UnknownKeyRefuses -- a key the registry never seeded
// (order above the P0-P2 support) refuses, both via supports()
// returning false and via create() throwing.
TEST(BasisTest, UnknownKeyRefuses) {
  const BasisRegistry registry;
  const BasisKey key{CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 3, BasisContinuity::cG};
  EXPECT_FALSE(registry.supports(key));
  EXPECT_THROW((void)registry.create(key), std::invalid_argument);
}

// physics.fe_basis.InvalidKeyIsRefusedBeforeTheLookup -- a non-Lagrange
// family at an order above maximum_basis_order is refused as an invalid
// key, with that message, before the factory map is consulted (the mutation row
// basis-registry-unknown-key-not-refused is killed by this test, which
// distinguishes the two refusals by the message).
TEST(BasisTest, InvalidKeyIsRefusedBeforeTheLookup) {
  const BasisRegistry registry;
  const BasisKey key{CellFamily::Hexahedron, FieldBasisFamily::Hierarchical,
                     fuse::physics::fe::maximum_basis_order + 1,
                     BasisContinuity::cG};
  EXPECT_FALSE(registry.supports(key));
  try {
    (void)registry.create(key);
    FAIL() << "an invalid basis key was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("invalid basis key"), std::string::npos) << e.what();
  }
}

// physics.fe_basis.LagrangeKeyAboveMaximumOrderIsAlsoRefused -- regression:
// valid_key() used to read `key.family ==
// FieldBasisFamily::LagrangeNodal || key.order <= maximum_basis_order`: the
// OR made the whole test true for ANY LagrangeNodal key regardless of
// order, so this exact key (LagrangeNodal, order > maximum_basis_order)
// used to be accepted as "valid" (then fail later, inside create(), only
// because no factory happens to be registered for it -- a different,
// weaker refusal than the one this test now requires). Fixed to drop the
// family clause; this is the case that distinguishes the fix from the
// bug.
TEST(BasisTest, LagrangeKeyAboveMaximumOrderIsAlsoRefused) {
  const BasisRegistry registry;
  const BasisKey key{CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal,
                     fuse::physics::fe::maximum_basis_order + 1, BasisContinuity::cG};
  EXPECT_FALSE(registry.supports(key));
  try {
    (void)registry.create(key);
    FAIL() << "a Lagrange basis key above the maximum order was accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("invalid basis key"), std::string::npos) << e.what();
  }
}

// physics.fe_basis.RegisterFactoryRejectsDuplicateByDefault -- the
// Reject policy (DuplicatePolicy's default) refuses to overwrite an
// existing key; Replace is required to do that deliberately.
TEST(BasisTest, RegisterFactoryRejectsDuplicateByDefault) {
  BasisRegistry registry;
  const BasisKey key{CellFamily::Edge, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG};
  ASSERT_TRUE(registry.supports(key));
  EXPECT_THROW(registry.register_factory(key, [&registry, key] { return registry.create(key); }), std::invalid_argument);
  EXPECT_NO_THROW(registry.register_factory(key, [&registry, key] { return registry.create(key); }, DuplicatePolicy::Replace));
}

// physics.fe_basis.OrientationTransformsAreIdentityAtTheSupportedOrders --
// basis.hpp's own derivation: every P0-P2 built-in basis places at most
// one DOF on any shared edge or face, so every OrientationTransform the
// registry generates is PROVABLY the identity permutation with
// sign +1, for every orientation code the sub-entity's own shape
// presents -- checked directly here (not assumed) for one representative
// P2 key from each of the five 3-D-adjacent families the basis
// supports (Hexahedron, Tetrahedron, Prism, Pyramid) plus Quadrilateral
// (2-D, included for its own single orientation axis).
TEST(BasisTest, OrientationTransformsAreIdentityAtTheSupportedOrders) {
  const BasisRegistry registry;
  for (const auto cell : {CellFamily::Quadrilateral, CellFamily::Hexahedron, CellFamily::Tetrahedron,
                           CellFamily::Prism, CellFamily::Pyramid}) {
    const BasisKey key{cell, FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG};
    const Basis basis = registry.create(key);
    const auto transforms = basis.orientation_transforms();
    ASSERT_FALSE(transforms.empty()) << "cell " << static_cast<int>(cell);
    for (const auto& transform : transforms) {
      for (std::size_t i = 0; i < transform.local_to_canonical.size(); ++i) {
        EXPECT_EQ(transform.local_to_canonical[i], i) << "cell " << static_cast<int>(cell);
        EXPECT_EQ(transform.signs[i], 1) << "cell " << static_cast<int>(cell);
      }
    }
  }
}

// physics.fe_basis.ApplyOrientationRoundTripsExactlyAcrossOppositeLocalOrder
// -- tested on two adjacent hexes
// (and two adjacent tets, two adjacent prisms) whose shared face is
// traversed in opposite local orders: transformed face DOFs must agree
// to round-off. Simulates this directly: for the shared face's own
// entity_dimension/local_entity_index on EACH of the two elements'
// (different, in general) local numbering, apply_orientation from that
// element's local coefficient to the GLOBAL canonical order, and confirm
// both elements land on the SAME canonical value. The round-trip bound
// is EXACT (0, not a derived epsilon): OrientationTransformsAreIdentity
// AtThisWavesOrders above already proved every transform the
// bases generate is the identity with sign +1, so applying one and then
// its inverse is literally the same floating-point value twice, not an
// arithmetic result that could accumulate rounding.
TEST(BasisTest, ApplyOrientationRoundTripsExactlyAcrossOppositeLocalOrder) {
  const BasisRegistry registry;
  for (const auto cell : {CellFamily::Hexahedron, CellFamily::Tetrahedron, CellFamily::Prism}) {
    const BasisKey key{cell, FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG};
    const Basis basis = registry.create(key);
    // The face dof the P2 bases place (Hex/Prism: a face
    // centre; Tet: none at the face level -- Tet10 has no face dof, so
    // skip straight to its edge dofs instead, entity_dimension 1).
    const unsigned face_dimension = cell == CellFamily::Tetrahedron ? 1 : 2;
    const auto transforms = basis.orientation_transforms();
    const auto found = std::find_if(transforms.begin(), transforms.end(), [=](const auto& t) {
      return t.entity_dimension == face_dimension;
    });
    ASSERT_NE(found, transforms.end()) << "cell " << static_cast<int>(cell);
    const unsigned entity = found->local_entity_index;
    // Element A sees this sub-entity under orientation code 0 (its own
    // local vertex order IS the sub-entity's own canonical reference, by
    // definition of "code 0"); element B sees the SAME physical
    // sub-entity under a DIFFERENT code (an opposite/rotated local
    // order) -- the largest code this shape's enum has, guaranteed valid
    // since orientation_transforms() enumerated every code the sub-
    // entity's own shape presents.
    unsigned code_b = 0;
    for (const auto& t : transforms)
      if (t.entity_dimension == face_dimension && t.local_entity_index == entity) code_b = std::max(code_b, t.orientation_code);
    ASSERT_GT(code_b, 0u) << "cell " << static_cast<int>(cell) << " has only one orientation code to test with";

    const std::vector<double> local_a{0.5};  // the face/edge dof count is always 1
    const std::vector<double> local_b{0.5};  // the SAME physical dof, seen from element B's own local order
    std::vector<double> canonical_a(1), canonical_b(1);
    basis.apply_orientation(face_dimension, entity, 0, local_a, canonical_a);
    basis.apply_orientation(face_dimension, entity, code_b, local_b, canonical_b);
    EXPECT_EQ(canonical_a[0], canonical_b[0]) << "cell " << static_cast<int>(cell);

    // And the inverse recovers each element's own local value exactly.
    std::vector<double> round_trip(1);
    basis.apply_orientation_inverse(face_dimension, entity, code_b, canonical_b, round_trip);
    EXPECT_EQ(round_trip[0], local_b[0]) << "cell " << static_cast<int>(cell);
  }
}

// physics.fe_basis.ApplyOrientationRefusesAnUnknownSubEntityOrCode --
// a refusal counts as verified only when a test fires it. Checks the
// MESSAGE, not only the exception type: find_transform's own lookup
// failure and apply_orientation's own storage-size mismatch both throw
// std::invalid_argument, so a type-only check cannot tell "the lookup's
// own refusal fired" apart from "some other, unrelated check happened to
// fire instead" (measured: this is NOT a hypothetical -- the mutation
// row basis-apply-orientation-unknown-code-not-refused that drops the
// lookup's own refusal still throws, from the SIZE-MISMATCH check
// comparing against an indeterminate past-the-end object, which a
// type-only EXPECT_THROW cannot distinguish from the intended refusal).
TEST(BasisTest, ApplyOrientationRefusesAnUnknownSubEntityOrCode) {
  const BasisRegistry registry;
  const Basis basis = registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG});
  std::vector<double> local{1.0}, canonical(1);
  try {
    basis.apply_orientation(2, 0, 99, local, canonical);
    FAIL() << "an unknown orientation code was accepted by apply_orientation";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("unknown orientation state"), std::string::npos) << e.what();
  }
  try {
    basis.apply_orientation_inverse(2, 0, 99, local, canonical);
    FAIL() << "an unknown orientation code was accepted by apply_orientation_inverse";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("unknown orientation state"), std::string::npos) << e.what();
  }
}

}  // namespace
