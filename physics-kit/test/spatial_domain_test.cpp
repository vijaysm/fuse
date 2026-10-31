// physics.spatial_domain -- ported from KARMA's karma_domain.h/.C, proving
// the I6 fix (domain.hpp's own header comment: the upstream assert was
// vacuous and never fired) as a real, firing refusal.

#include "fuse_physics/spatial/domain.hpp"

#include <gtest/gtest.h>

namespace {

using fuse::physics::spatial::Domain;
using fuse::physics::spatial::Shape;

// physics.spatial_domain.ConstructsForSupportedShapes -- Edge/Square/Cube
// construct without refusing and report back the extents/symmetry given.
TEST(SpatialDomainTest, ConstructsForSupportedShapes) {
  const Domain cube(3, Shape::Cube, 1.0, 2.0, 3.0, true);
  EXPECT_EQ(cube.dimension(), 3u);
  EXPECT_EQ(cube.shape(), Shape::Cube);
  EXPECT_DOUBLE_EQ(cube.length_x(), 1.0);
  EXPECT_DOUBLE_EQ(cube.length_y(), 2.0);
  EXPECT_DOUBLE_EQ(cube.length_z(), 3.0);
  EXPECT_TRUE(cube.is_symmetric());
}

// physics.spatial_domain.PerAxisSymmetryFlagsAreIndependent -- the
// three-flag constructor sets is_symmetric() only when ALL THREE axes are
// symmetric, matching karma_domain.C's own `_isSymX & _isSymY & _isSymZ`.
TEST(SpatialDomainTest, PerAxisSymmetryFlagsAreIndependent) {
  const Domain mixed(3, Shape::Cube, 1.0, 1.0, 1.0, true, false, true);
  EXPECT_TRUE(mixed.is_symmetric_x());
  EXPECT_FALSE(mixed.is_symmetric_y());
  EXPECT_TRUE(mixed.is_symmetric_z());
  EXPECT_FALSE(mixed.is_symmetric());
}

// physics.spatial_domain.RefusesUnsupportedShapes -- the I6 fix: both
// Hexagon and Unstructured must refuse construction (the upstream
// assert's INTENDED condition, not the vacuous OR it actually wrote).
TEST(SpatialDomainTest, RefusesUnsupportedShapes) {
  EXPECT_THROW(Domain(2, Shape::Hexagon, 1.0, 1.0, 0.0, true), std::invalid_argument);
  EXPECT_THROW(Domain(3, Shape::Unstructured, 1.0, 1.0, 1.0, true), std::invalid_argument);
  EXPECT_THROW(Domain(2, Shape::Hexagon, 1.0, 1.0, 0.0, true, true, true), std::invalid_argument);
}

}  // namespace
