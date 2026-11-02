// physics.fe_reference_topology -- ported from KARMA's basis_test.C and
// geometry_test.C (the reference-cell shape-function properties both
// files check against KARMAReferenceTopology/KARMABasis directly, before
// either touches a mesh). Partition of unity, the Kronecker property at
// nodes, and derivative sums to zero are the three properties
// docs/dev/porting.md's own task brief names for this wave's basis tests.
//
// (item 1): Prism6/15/18 and
// Pyramid5/13/14 join the variant list this file's properties are
// checked against, plus the serendipity Quad8 (needed as Prism15's and
// Pyramid13's one quadrilateral side, basis.hpp's own note) -- the same
// three properties hold for a serendipity basis as for a complete one
// (both are still a partition of unity with the Kronecker property), so
// Quad8 is exercised by the same generic loops as every other variant,
// not given special-cased tests. The pyramid's apex gets its own tests
// (ApexValuesAreWellDefinedButDerivativesRefuse) since its basis has a
// genuinely different (removable-limit) evaluation path there.

#include "fuse_physics/fe/reference_topology.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

using fuse::physics::Point;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::ReferenceTopology;

std::vector<GeometryVariant> all_variants() {
  return {GeometryVariant::Edge2,    GeometryVariant::Edge3,    GeometryVariant::Tri3,    GeometryVariant::Tri6,
          GeometryVariant::Quad4,    GeometryVariant::Quad8,    GeometryVariant::Quad9,   GeometryVariant::Tet4,
          GeometryVariant::Tet10,    GeometryVariant::Hex8,     GeometryVariant::Hex27,   GeometryVariant::Prism6,
          GeometryVariant::Prism15,  GeometryVariant::Prism18,  GeometryVariant::Pyramid5,
          GeometryVariant::Pyramid13, GeometryVariant::Pyramid14};
}

// Points used by PartitionOfUnity and DerivativesSumToZero -- evaluate()
// (unlike evaluate_values()) genuinely refuses at the pyramid's exact
// apex (0,0,1) (the path-dependent-derivative carve-out both this file's
// header and reference_topology.hpp's own document), so no pyramid
// sample point here is the apex; ApexValuesAreWellDefinedButDerivativesRefuse
// below covers that point on its own.
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
    case CellFamily::Prism:
      return {Point(0, 0, 0), Point(1, 0, 1), Point(0, 1, .5), Point(.2, .3, .4)};
    case CellFamily::Pyramid:
      // The base corners, a mid-height point near but not at the apex,
      // and the cell centre -- none equal to the exact apex (0,0,1).
      return {Point(-1, -1, 0), Point(1, 1, 0), Point(0, 0, .5), Point(.25, -.25, .75)};
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
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Quad8).node_count(), 8u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Quad9).node_count(), 9u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tet4).node_count(), 4u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tet10).node_count(), 10u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Hex8).node_count(), 8u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Hex27).node_count(), 27u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Prism6).node_count(), 6u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Prism15).node_count(), 15u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Prism18).node_count(), 18u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Pyramid5).node_count(), 5u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Pyramid13).node_count(), 13u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Pyramid14).node_count(), 14u);
}

// physics.fe_reference_topology.SideCountsMatchReferenceCell -- side_count()
// matches the textbook facet count for each implemented cell.
TEST(ReferenceTopologyTest, SideCountsMatchReferenceCell) {
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Edge2).side_count(), 2u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tri3).side_count(), 3u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Quad4).side_count(), 4u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Tet4).side_count(), 4u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Hex8).side_count(), 6u);
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Prism6).side_count(), 5u);   // 3 quad + 2 triangle
  EXPECT_EQ(ReferenceTopology(GeometryVariant::Pyramid5).side_count(), 5u);  // 4 triangle + 1 quad
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

// physics.fe_reference_topology.PrismAndPyramidHaveMixedSideShapes --
// fe_base.hpp's own motivation for carrying two quadrature rules on a
// Face FEBase: a Prism's sides are 3 quadrilaterals then 2 triangles, a
// Pyramid's are 4 triangles then 1 quadrilateral (KARMA's own
// side_table ordering, reference_topology.cpp's own comment).
TEST(ReferenceTopologyTest, PrismAndPyramidHaveMixedSideShapes) {
  const ReferenceTopology prism(GeometryVariant::Prism6);
  for (unsigned s = 0; s < 3; ++s) EXPECT_EQ(prism.side_topology(s).family(), CellFamily::Quadrilateral) << s;
  for (unsigned s = 3; s < 5; ++s) EXPECT_EQ(prism.side_topology(s).family(), CellFamily::Triangle) << s;

  const ReferenceTopology pyramid(GeometryVariant::Pyramid5);
  for (unsigned s = 0; s < 4; ++s) EXPECT_EQ(pyramid.side_topology(s).family(), CellFamily::Triangle) << s;
  EXPECT_EQ(pyramid.side_topology(4).family(), CellFamily::Quadrilateral);
}

// physics.fe_reference_topology.SerendipityQuadSideOfOrder2PrismAndPyramid
// -- Prism15's quad sides and Pyramid13's one quad side are the Quad8
// serendipity face (basis.hpp's own note on why Quad8 exists at all),
// not Quad9: a P2 wedge/pyramid has no face-interior dof to put a ninth
// node at.
TEST(ReferenceTopologyTest, SerendipityQuadSideOfOrder2PrismAndPyramid) {
  const ReferenceTopology prism15(GeometryVariant::Prism15);
  for (unsigned s = 0; s < 3; ++s) EXPECT_EQ(prism15.side_topology(s).variant(), GeometryVariant::Quad8) << s;
  const ReferenceTopology pyramid13(GeometryVariant::Pyramid13);
  EXPECT_EQ(pyramid13.side_topology(4).variant(), GeometryVariant::Quad8);
}

// physics.fe_reference_topology.ApexValuesAreWellDefinedButDerivativesRefuse
// -- the pyramid's apex (0,0,1) is a removable singularity of its
// rational basis (reference_topology.hpp's own documented carve-out):
// evaluate_values() returns the known limit (only the apex node's own
// function is 1) but evaluate() (which also wants the DERIVATIVE, which
// is genuinely path-dependent there) refuses -- a refusal counts as
// verified only when a test fires it (this wave's own rule), so this is
// that test, for all three pyramid variants the apex singularity applies
// to.
TEST(ReferenceTopologyTest, ApexValuesAreWellDefinedButDerivativesRefuse) {
  for (const auto variant : {GeometryVariant::Pyramid5, GeometryVariant::Pyramid13, GeometryVariant::Pyramid14}) {
    const ReferenceTopology pyramid(variant);
    const auto values = pyramid.evaluate_values(Point(0, 0, 1));
    for (std::size_t i = 0; i < values.size(); ++i) EXPECT_NEAR(values[i], i == 4 ? 1.0 : 0.0, 1.0e-14) << i;
    EXPECT_THROW((void)pyramid.evaluate(Point(0, 0, 1)), std::invalid_argument);
  }
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


// physics.fe_reference_topology.EdgeVerticesMatchNodesForMidpoints --
//: fe::edge_vertices' own table (reference_
// topology.hpp's own header comment) must agree with nodes_for()'s
// OWN construction of each order-2 variant's edge-midpoint nodes
// (reference_topology.cpp): edge `e`'s midpoint node, at index
// `node_count(order-1 variant) + e` in the order-2 node list, must
// equal the AVERAGE of the two P1 corner positions edge_vertices(e)
// names, read off the SAME ReferenceTopology's own reference_nodes().
// Tetrahedron, Hexahedron, Prism, Pyramid (the four families with a
// 3-D interior-edge dof this item closes the gap for); Triangle and
// Quadrilateral are covered by SideTopologyMatchesCorners above (their
// "edges" are sides).
void check_edge_midpoints(GeometryVariant order1, GeometryVariant order2) {
  const ReferenceTopology linear(order1);
  const ReferenceTopology quadratic(order2);
  const auto cell = linear.family();
  const unsigned n_corners = linear.node_count();
  const unsigned n_edges = fuse::physics::fe::edge_count(cell);
  for (unsigned e = 0; e < n_edges; ++e) {
    const auto corners = fuse::physics::fe::edge_vertices(cell, e);
    const Point expected = 0.5 * (quadratic.reference_nodes().at(corners[0]) + quadratic.reference_nodes().at(corners[1]));
    const Point actual = quadratic.reference_nodes().at(n_corners + e);
    const Point diff = actual - expected;
    EXPECT_NEAR(diff.x, 0.0, 1e-12) << "edge " << e;
    EXPECT_NEAR(diff.y, 0.0, 1e-12) << "edge " << e;
    EXPECT_NEAR(diff.z, 0.0, 1e-12) << "edge " << e;
  }
}

TEST(ReferenceTopologyTest, EdgeVerticesMatchNodesForMidpointsTet) {
  check_edge_midpoints(GeometryVariant::Tet4, GeometryVariant::Tet10);
}
TEST(ReferenceTopologyTest, EdgeVerticesMatchNodesForMidpointsHex) {
  check_edge_midpoints(GeometryVariant::Hex8, GeometryVariant::Hex27);
}
TEST(ReferenceTopologyTest, EdgeVerticesMatchNodesForMidpointsPrism) {
  check_edge_midpoints(GeometryVariant::Prism6, GeometryVariant::Prism18);
}
TEST(ReferenceTopologyTest, EdgeVerticesMatchNodesForMidpointsPyramid) {
  check_edge_midpoints(GeometryVariant::Pyramid5, GeometryVariant::Pyramid14);
}

// physics.fe_reference_topology.EdgeVertexIndexAtExactUpperBoundRefuses --
// the same named-refusal contract side_topology's own range check
// states (SideIndexAtExactUpperBoundRefuses above), for edge_vertices.
TEST(ReferenceTopologyTest, EdgeVertexIndexAtExactUpperBoundRefuses) {
  EXPECT_THROW((void)fuse::physics::fe::edge_vertices(CellFamily::Tetrahedron, 6), std::invalid_argument);
}

}  // namespace
