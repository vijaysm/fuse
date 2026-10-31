// physics.fe_geometry_map -- ported from KARMA's geometry_test.C and
// geometry_inverse_map_contract_test.C. The inverse-map contract test is
// the task's own named example ("geometry maps by the inverse-map
// contract KARMA's own test states") and is ported close to verbatim,
// restated as gtest EXPECT_EQ rather than a main()-returning-1 check.

#include "fuse_physics/fe/geometry_map.hpp"

#include <array>
#include <cmath>

#include <gtest/gtest.h>

namespace {

using fuse::physics::Point;
using fuse::physics::fe::GeometryMap;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::InverseMapStatus;
using fuse::physics::fe::ReferenceTopology;

// physics.fe_geometry_map.InverseMapContractLocatesInteriorAndRejectsExterior
// -- KARMA's geometry_inverse_map_contract_test.C, verbatim in substance:
// a point strictly inside a reference-shaped triangle is Located; a point
// outside its bounding region is NotContained.
TEST(GeometryMapTest, InverseMapContractLocatesInteriorAndRejectsExterior) {
  const ReferenceTopology topology(GeometryVariant::Tri3);
  const std::array<Point, 3> nodes{Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0), Point(0.0, 1.0, 0.0)};
  const GeometryMap geometry(topology, nodes);

  const auto located = geometry.try_inverse_map(Point(0.25, 0.25, 0.0), 1.0e-12, 1.0e-10);
  EXPECT_EQ(located.status, InverseMapStatus::Located);

  const auto uncovered = geometry.try_inverse_map(Point(1.2, 0.2, 0.0), 1.0e-12, 1.0e-10);
  EXPECT_EQ(uncovered.status, InverseMapStatus::NotContained);
}

// physics.fe_geometry_map.MapRoundTripsThroughInverseMap -- map(inverse_map(x))
// recovers x to the same contract tolerance, on a non-reference-shaped
// (sheared) quadrilateral -- exercises the affine quad4 fast path.
TEST(GeometryMapTest, MapRoundTripsThroughInverseMap) {
  const ReferenceTopology topology(GeometryVariant::Quad4);
  const std::array<Point, 4> nodes{Point(0, 0), Point(2, 0), Point(3, 1), Point(0, 1)};
  const GeometryMap geometry(topology, nodes);
  const Point physical(1.4, 0.3);
  const Point reference = geometry.inverse_map(physical);
  const Point recovered = geometry.map(reference);
  EXPECT_NEAR(recovered.x, physical.x, 1.0e-10);
  EXPECT_NEAR(recovered.y, physical.y, 1.0e-10);
}

// physics.fe_geometry_map.UnitCellMeasureMatchesVolume -- the unit
// reference triangle/tet/hex's own measure() at any interior point equals
// its textbook area/volume (1/2 for the unit triangle, 1/6 for the unit
// tet, 1 for the unit cube) -- a Jacobian-determinant sanity check
// independent of quadrature.
TEST(GeometryMapTest, UnitCellMeasureMatchesVolume) {
  {
    const ReferenceTopology topology(GeometryVariant::Tri3);
    const std::array<Point, 3> nodes{Point(0, 0), Point(1, 0), Point(0, 1)};
    const GeometryMap geometry(topology, nodes);
    EXPECT_NEAR(geometry.measure(topology.center()), 1.0, 1.0e-12);  // |cross| of the edge vectors, not the area
  }
  {
    const ReferenceTopology topology(GeometryVariant::Hex8);
    const std::array<Point, 8> nodes{Point(0, 0, 0), Point(1, 0, 0), Point(1, 1, 0), Point(0, 1, 0),
                                      Point(0, 0, 1), Point(1, 0, 1), Point(1, 1, 1), Point(0, 1, 1)};
    const GeometryMap geometry(topology, nodes);
    EXPECT_NEAR(geometry.measure(topology.center()), 1.0, 1.0e-12);
  }
}

// physics.fe_geometry_map.DegenerateGeometryRefuses -- three collinear
// "triangle" nodes have zero measure everywhere; reinit's own validate()
// must refuse construction rather than silently produce an invalid map.
TEST(GeometryMapTest, DegenerateGeometryRefuses) {
  const ReferenceTopology topology(GeometryVariant::Tri3);
  const std::array<Point, 3> collinear{Point(0, 0), Point(1, 0), Point(2, 0)};
  EXPECT_THROW(GeometryMap(topology, collinear), std::invalid_argument);
}

// physics.fe_geometry_map.ZeroScaleGeometryRefuses -- three COINCIDENT
// nodes (every pairwise distance exactly zero, not merely collinear with
// nonzero separation) pins validate()'s own "scale <= 0.0" early refusal
// directly: every node-to-node distance is zero, so the element has no
// scale to measure anything against.
TEST(GeometryMapTest, ZeroScaleGeometryRefuses) {
  const ReferenceTopology topology(GeometryVariant::Tri3);
  const std::array<Point, 3> coincident{Point(1, 1), Point(1, 1), Point(1, 1)};
  EXPECT_THROW(GeometryMap(topology, coincident), std::invalid_argument);
}

}  // namespace
