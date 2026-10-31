// physics.fe_reference_topology -- ported from KARMA's basis_test.C and
// geometry_test.C (the reference-cell shape-function properties both
// files check against KARMAReferenceTopology/KARMABasis directly, before
// either touches a mesh). Partition of unity, the Kronecker property at
// nodes, and derivative sums to zero are the three properties
// docs/dev/porting.md's own task brief names for this wave's basis tests.
//
// PORTED SUBSET (deviations: this file does not reproduce every KARMA
// basis_test.C/geometry_test.C case -- see this wave's design_notes for
// the full list of what is deferred, principally the orientation-metadata
// tests, which need the OrientationTransform machinery this wave's
// basis.hpp does not port).

#include "fuse_physics/fe/reference_topology.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {

using fuse::physics::Point;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::ReferenceTopology;

std::vector<GeometryVariant> all_variants() {
  return {GeometryVariant::Edge2,  GeometryVariant::Edge3, GeometryVariant::Tri3, GeometryVariant::Tri6,
          GeometryVariant::Quad4,  GeometryVariant::Quad9, GeometryVariant::Tet4, GeometryVariant::Tet10,
          GeometryVariant::Hex8,   GeometryVariant::Hex27};
}

std::vector<Point> sample_points(CellFamily family) {
  switch (family) {
    case CellFamily::Edge:
      return {Point(0.0), Point(1.0), Point(0.3), Point(0.7123)};
    case CellFamily::Triangle:
      return {Point(0, 0), Point(1, 0), Point(0, 1), Point(.2, .3), Point(1.0 / 3, 1.0 / 3)};
    case CellFamily::Quadrilateral:
      return {Point(0, 0), Point(1, 1), Point(.25, .75), Point(.5, .5)};
    case CellFamily::Tetrahedron:
      return {Point(0, 0, 0), Point(1, 0, 0), Point(.2, .3, .1), Point(.25, .25, .25)};
    case CellFamily::Hexahedron:
      return {Point(0, 0, 0), Point(1, 1, 1), Point(.3, .6, .9), Point(.5, .5, .5)};
  }
  return {};
}

// physics.fe_reference_topology.PartitionOfUnity -- sum_i phi_i(x) == 1 at
// every sampled point, including off-node interior points, for every
// implemented variant.
TEST(ReferenceTopologyTest, PartitionOfUnity) {
  for (const auto variant : all_variants()) {
    const ReferenceTopology topology(variant);
    for (const auto& point : sample_points(topology.family())) {
      const auto values = topology.evaluate_values(point);
      double sum = 0.0;
      for (const double v : values) sum += v;
      EXPECT_NEAR(sum, 1.0, 1.0e-12) << "variant " << static_cast<int>(variant);
    }
  }
}

// physics.fe_reference_topology.KroneckerPropertyAtNodes -- phi_i(node_j)
// == (i == j), the defining property of a nodal Lagrange basis.
TEST(ReferenceTopologyTest, KroneckerPropertyAtNodes) {
  for (const auto variant : all_variants()) {
    const ReferenceTopology topology(variant);
    const auto& nodes = topology.reference_nodes();
    for (std::size_t j = 0; j < nodes.size(); ++j) {
      const auto values = topology.evaluate_values(nodes[j]);
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double expected = (i == j) ? 1.0 : 0.0;
        EXPECT_NEAR(values[i], expected, 1.0e-10) << "variant " << static_cast<int>(variant) << " node " << j;
      }
    }
  }
}

// physics.fe_reference_topology.DerivativesSumToZero -- d/dx sum_i phi_i
// == 0 (the derivative of the partition-of-unity identity), at every
// sampled point.
TEST(ReferenceTopologyTest, DerivativesSumToZero) {
  for (const auto variant : all_variants()) {
    const ReferenceTopology topology(variant);
    for (const auto& point : sample_points(topology.family())) {
      const auto result = topology.evaluate(point);
      Point sum = Point::zero();
      for (const auto& d : result.derivatives) sum += d;
      EXPECT_NEAR(sum.x, 0.0, 1.0e-10) << "variant " << static_cast<int>(variant);
      EXPECT_NEAR(sum.y, 0.0, 1.0e-10) << "variant " << static_cast<int>(variant);
      EXPECT_NEAR(sum.z, 0.0, 1.0e-10) << "variant " << static_cast<int>(variant);
    }
  }
}

// physics.fe_reference_topology.NodeCountMatchesVariant -- the node table
// (reference_topology.cpp's nodes_for) agrees with the element's own
// name (e.g. Hex27 has 27 nodes).
TEST(ReferenceTopologyTest, NodeCountMatchesVariant) {
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Edge2).node_count(), 2u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Edge3).node_count(), 3u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tri3).node_count(), 3u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tri6).node_count(), 6u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Quad4).node_count(), 4u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Quad9).node_count(), 9u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tet4).node_count(), 4u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tet10).node_count(), 10u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Hex8).node_count(), 8u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Hex27).node_count(), 27u);
}

// physics.fe_reference_topology.SideCountsMatchReferenceCell -- side_count()
// matches the textbook facet count for each implemented cell.
TEST(ReferenceTopologyTest, SideCountsMatchReferenceCell) {
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Edge2).side_count(), 2u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tri3).side_count(), 3u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Quad4).side_count(), 4u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tet4).side_count(), 4u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Hex8).side_count(), 6u);
}

// physics.fe_reference_topology.SideTopologyMatchesCorners -- a Hex8 side
// is a Quad4, a Tet4 side is a Tri3 -- the CN-style convention
// side_reference_map/_jacobian rely on.
TEST(ReferenceTopologyTest, SideTopologyMatchesCorners) {
  const ReferenceTopology hex(GeometryVariant::Hex8);
  for (unsigned s = 0; s < hex.side_count(); ++s) EXPECT_EQ(hex.side_topology(s).variant(), GeometryVariant::Quad4);
  const ReferenceTopology tet(GeometryVariant::Tet4);
  for (unsigned s = 0; s < tet.side_count(); ++s) EXPECT_EQ(tet.side_topology(s).variant(), GeometryVariant::Tri3);
}

// physics.fe_reference_topology.EdgeSideTopologyRefuses -- an edge's sides
// are vertices, which have no ReferenceTopology of their own; this must
// refuse rather than return something meaningless. (D-C: "a refusal counts
// as verified only when a test makes it fire".)
TEST(ReferenceTopologyTest, EdgeSideTopologyRefuses) {
  const ReferenceTopology edge(GeometryVariant::Edge2);
  EXPECT_THROW((void)edge.side_topology(0), std::invalid_argument);
}

// physics.fe_reference_topology.SideIndexAtExactUpperBoundRefuses -- the
// boundary itself: side == side_count() (one past the last valid index,
// not merely "large") must refuse from side_nodes/side_topology/
// side_reference_map alike, pinning the ">=" bound exactly (not just "> some
// margin").
TEST(ReferenceTopologyTest, SideIndexAtExactUpperBoundRefuses) {
  const ReferenceTopology tri(GeometryVariant::Tri3);
  const unsigned at_bound = tri.side_count();
  EXPECT_THROW((void)tri.side_nodes(at_bound), std::invalid_argument);
  EXPECT_THROW((void)tri.side_topology(at_bound), std::invalid_argument);
  EXPECT_THROW((void)tri.side_reference_map(at_bound, Point(0.5)), std::invalid_argument);
}

}  // namespace
