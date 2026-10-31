// physics.fe_basis -- ported from KARMA's basis_test.C (the registry and
// metadata-consistency portions; the orientation-metadata tests are NOT
// ported, see fe_reference_topology_test.cpp's header comment).

#include "fuse_physics/fe/basis.hpp"

#include <array>
#include <stdexcept>

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
  constexpr std::array cells{CellFamily::Edge, CellFamily::Triangle, CellFamily::Quadrilateral,
                              CellFamily::Tetrahedron, CellFamily::Hexahedron};
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
// (order above this wave's P0-P2 support) refuses, both via supports()
// returning false and via create() throwing.
TEST(BasisTest, UnknownKeyRefuses) {
  const BasisRegistry registry;
  const BasisKey key{CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 3, BasisContinuity::cG};
  EXPECT_FALSE(registry.supports(key));
  EXPECT_THROW((void)registry.create(key), std::invalid_argument);
}

// physics.fe_basis.InvalidKeyIsRefusedBeforeTheLookup -- valid_key() accepts
// every Lagrange key whatever its order, so the key above never reaches the
// EARLY refusal; this one does: a non-Lagrange family at an order above
// maximum_basis_order is refused as an invalid key, with that message, before
// the factory map is consulted (lead, 2026-10-31: the mutation row
// basis-registry-unknown-key-not-refused had been declared equivalent because
// no test distinguished the two refusals; this test does, by the message).
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

}  // namespace
