// physics.spatial_mesh -- ported from the mesh-topology portions of
// KARMA's mesh_test.C (docs/dev/physics-kit-disposition.md's own
// qualifier: "this wave's tests are deterministic unit tests...mesh
// topology"). Upstream's own mesh_test.C is a 2-rank MOAB/MPI test
// (disposition Appendix B); this wave's native, serial spatial::Mesh has
// neither, so this is a topology-only re-expression against the native
// Hex8 mesh (mesh.hpp's own scope note names the MOAB/MPI gap this leaves
// open).

#include "fuse_physics/spatial/mesh.hpp"

#include <array>

#include <gtest/gtest.h>

#include "fuse_physics/spatial/mesh_utils.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::spatial::EntityHandle;
using fuse::physics::spatial::extract_boundary_faces;
using fuse::physics::spatial::Mesh;

// physics.spatial_mesh.GenerateBoxProducesExpectedCounts -- an nx*ny*nz
// box has nx*ny*nz elements and (nx+1)(ny+1)(nz+1) vertices.
TEST(SpatialMeshTest, GenerateBoxProducesExpectedCounts) {
  Mesh mesh(3);
  mesh.generate_box(2, 3, 4, Point(0, 0, 0), Point(1, 1, 1));
  EXPECT_EQ(mesh.n_elements(), 2u * 3u * 4u);
  EXPECT_EQ(mesh.n_vertices(), 3u * 4u * 5u);
}

// physics.spatial_mesh.InteriorSideHasANeighborSharingItsVertexSet --
// in a 2x1x1 box, element 1's side toward element 2 (side 1, the local
// +x face in reference_topology.cpp's Hex8 side table) has element 2 as
// its neighbor, and the two elements' canonical vertex ids for that
// shared side agree (the SAME property KARMA's own canonical_side_
// vertex_ids exists to let neighbor() compare without caring about local
// orientation).
TEST(SpatialMeshTest, InteriorSideHasANeighborSharingItsVertexSet) {
  Mesh mesh(3);
  mesh.generate_box(2, 1, 1, Point(0, 0, 0), Point(2, 1, 1));
  const auto neighbor = mesh.neighbor(1, 1);
  ASSERT_TRUE(neighbor.has_value());
  EXPECT_EQ(*neighbor, 2u);
  bool found_matching_side = false;
  for (unsigned s = 0; s < mesh.n_sides(2); ++s)
    if (mesh.canonical_side_vertex_ids(2, s) == mesh.canonical_side_vertex_ids(1, 1)) found_matching_side = true;
  EXPECT_TRUE(found_matching_side);
}

// physics.spatial_mesh.EveryOutwardSideOfASingleElementIsBoundary -- a
// single-element mesh has every one of its 6 sides classified boundary
// (is_boundary_side), since there is no second element to be a neighbor.
TEST(SpatialMeshTest, EveryOutwardSideOfASingleElementIsBoundary) {
  Mesh mesh(3);
  mesh.generate_box(1, 1, 1, Point(0, 0, 0), Point(1, 1, 1));
  for (unsigned s = 0; s < mesh.n_sides(1); ++s) EXPECT_TRUE(mesh.is_boundary_side(1, s));
  EXPECT_EQ(extract_boundary_faces(mesh).size(), 6u);
}

// physics.spatial_mesh.TwoElementBoxHasExactlyTenBoundaryFaces -- a 2x1x1
// box has 12 total faces (6 per element) minus the 2 that are shared
// (counted once each, not twice) = 10 boundary faces; the one INTERIOR
// face (shared between the two elements) is excluded from both sides.
TEST(SpatialMeshTest, TwoElementBoxHasExactlyTenBoundaryFaces) {
  Mesh mesh(3);
  mesh.generate_box(2, 1, 1, Point(0, 0, 0), Point(2, 1, 1));
  EXPECT_EQ(extract_boundary_faces(mesh).size(), 10u);
}

// physics.spatial_mesh.BoundingBoxAndCentroidMatchGeometry -- the mesh's
// own bounding_box() and an element's centroid() agree with the box
// extents generate_box was given.
TEST(SpatialMeshTest, BoundingBoxAndCentroidMatchGeometry) {
  Mesh mesh(3);
  mesh.generate_box(2, 2, 2, Point(0, 0, 0), Point(2, 4, 6));
  const auto [lo, hi] = mesh.bounding_box();
  EXPECT_NEAR(lo.x, 0.0, 1.0e-12);
  EXPECT_NEAR(hi.x, 2.0, 1.0e-12);
  EXPECT_NEAR(hi.y, 4.0, 1.0e-12);
  EXPECT_NEAR(hi.z, 6.0, 1.0e-12);
  const Point centroid = mesh.centroid(1);
  EXPECT_NEAR(centroid.x, 0.5, 1.0e-12);
  EXPECT_NEAR(centroid.y, 1.0, 1.0e-12);
  EXPECT_NEAR(centroid.z, 1.5, 1.0e-12);
}

// physics.spatial_mesh.AssignIdsIsIdempotentIdentity -- this native
// mesh's own assign_ids() (mesh.hpp's header comment on why identity is
// the correct numbering here, not a simplification) returns the same
// global/local id before and after a second call.
TEST(SpatialMeshTest, AssignIdsIsIdempotentIdentity) {
  Mesh mesh(3);
  mesh.generate_box(1, 1, 1, Point(0, 0, 0), Point(1, 1, 1));
  mesh.assign_ids();
  const auto first = mesh.global_id(1);
  mesh.assign_ids();
  EXPECT_EQ(mesh.global_id(1), first);
  EXPECT_EQ(mesh.global_id(1), 1u);
}

}  // namespace
