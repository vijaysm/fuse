// physics.spatial_mesh -- ported from the mesh-topology portions of
// KARMA's mesh_test.C.
//
// WAVE 1b (owner instruction 2026-10-31, item 2) closes the MOAB/MPI gap
// wave 1 left open: spatial::Mesh now binds to fuse::backend's MeshView
// (not MOAB directly, per docs/dev/layering.md's "physics-kit reaches
// MOAB only through fuse::backend"), so FromMeshViewReadsOwnershipAndGlobalIds
// below builds a hand-made TWO-"rank" fuse::native::MeshArrays fixture
// (NativeMeshBuilder, owner/is_owned set as if ranks 0 and 1 each loaded
// their own half) and reads it through spatial::Mesh::from_mesh_view --
// entirely at tier 0, no MPI_Init anywhere in this file (the owner's own
// rule quoted in the task: partition-aware queries belong on the public
// surface, exercised without needing MPI to prove it). The genuine
// multi-rank collective test (test/parallel/physics_mesh_mpi_test.cpp,
// RANKS 2 3) is build-only in this sandbox, listed in host_pending.
// generate_2d_box and extract_edge_at_coordinate (karma_mesh_extract_
// edge.C, adapted) are new this wave too.

#include "fuse_physics/spatial/mesh.hpp"

#include <array>
#include <memory>

#include <gtest/gtest.h>

#include "fuse/backend/native_mesh.hpp"
#include "fuse_physics/spatial/mesh_utils.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::spatial::EntityHandle;
using fuse::physics::spatial::extract_boundary_faces;
using fuse::physics::spatial::extract_edge_at_coordinate;
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

// physics.spatial_mesh.Generate2dBoxProducesExpectedCounts -- the 2-D
// structured Quad4 box (owner instruction 2026-10-31, item 2: the source
// karma_mesh_extract_edge.C needs and wave 1 had no 2-D mesh to build):
// an nx*ny box has nx*ny elements and (nx+1)(ny+1) vertices, same
// counting rule as generate_box's own 3-D test above.
TEST(SpatialMeshTest, Generate2dBoxProducesExpectedCounts) {
  Mesh mesh(2);
  mesh.generate_2d_box(3, 2, Point(0, 0), Point(3, 2));
  EXPECT_EQ(mesh.n_elements(), 3u * 2u);
  EXPECT_EQ(mesh.n_vertices(), 4u * 3u);
  EXPECT_EQ(mesh.element_kind(1), fuse::physics::fe::GeometryVariant::Quad4);
}

// physics.spatial_mesh.ExtractEdgeAtCoordinateFindsTheRightLine --
// karma_mesh_extract_edge.C, adapted (mesh_utils.hpp's own note): on a
// 3x2 box over [0,3]x[0,2], the line x == 0 (the left edge) has 3
// vertices and 2 one-dimensional elements; the line x == 1.5 (an
// INTERIOR line, not a boundary -- KARMA's own coordinate filter has no
// boundary concept, mesh_utils.hpp's header comment) still has no
// vertices at all, there being no mesh vertex at that coordinate on this
// box's own 1-unit spacing.
TEST(SpatialMeshTest, ExtractEdgeAtCoordinateFindsTheRightLine) {
  Mesh box(2);
  box.generate_2d_box(3, 2, Point(0, 0), Point(3, 2));
  const Mesh left_edge = extract_edge_at_coordinate(box, 0, 0.0, 1.0e-9);
  EXPECT_EQ(left_edge.n_vertices(), 3u);
  EXPECT_EQ(left_edge.n_elements(), 2u);
  for (EntityHandle e = 1; e <= left_edge.n_elements(); ++e)
    EXPECT_EQ(left_edge.element_kind(e), fuse::physics::fe::GeometryVariant::Edge2);

  const Mesh empty = extract_edge_at_coordinate(box, 0, 1.5, 1.0e-9);
  EXPECT_EQ(empty.n_vertices(), 0u);
  EXPECT_EQ(empty.n_elements(), 0u);
}

// physics.spatial_mesh.ExtractEdgeAtCoordinateRefusesOnA3dMesh -- D-C:
// a refusal counts as verified only when a test fires it.
TEST(SpatialMeshTest, ExtractEdgeAtCoordinateRefusesOnA3dMesh) {
  Mesh volume(3);
  volume.generate_box(1, 1, 1, Point(0, 0, 0), Point(1, 1, 1));
  EXPECT_THROW((void)extract_edge_at_coordinate(volume, 0, 0.0, 1.0e-9), std::invalid_argument);
}

// physics.spatial_mesh.FromMeshViewReadsOwnershipAndGlobalIds --
// spatial::Mesh::from_mesh_view binds to fuse::backend's MeshView
// abstraction, not to a concrete mesh type (docs/dev/layering.md:
// "physics-kit reaches MOAB only through fuse::backend"). Builds a
// TWO-"rank" fuse::native::MeshArrays fixture directly (NativeMeshBuilder,
// owner/is_owned flags set by hand, as if rank 0 owns vertices 0-3/cell 0
// and rank 1 owns vertex 4/cell 1, with vertex 3 a GHOST on rank 0's own
// view of rank 1's cell) -- entirely at tier 0, no MPI_Init: the owner's
// own instruction ("this has to run in parallel...so partition-aware
// queries are part of the public surface") is about the QUERY surface,
// which a single-process NativeMeshView already exercises fully; the
// actual collective behaviour across real ranks is
// test/parallel/physics_mesh_mpi_test.cpp (RANKS 2 3, build-only here).
TEST(SpatialMeshTest, FromMeshViewReadsOwnershipAndGlobalIds) {
  fuse::native::NativeMeshBuilder builder(2);
  // A 1x2 strip of two Quad4 cells sharing an edge (vertices 1,2):
  // 0--1--4
  // |  |  |
  // 3--2--5   (local indices; see the two add_cell calls below)
  const int v0 = builder.add_vertex(100, 0, 0, 0, /*owner=*/0, /*is_owned=*/true);
  const int v1 = builder.add_vertex(101, 1, 0, 0, /*owner=*/0, /*is_owned=*/true);
  const int v2 = builder.add_vertex(102, 1, 1, 0, /*owner=*/0, /*is_owned=*/true);
  const int v3 = builder.add_vertex(103, 0, 1, 0, /*owner=*/0, /*is_owned=*/true);
  // Rank 0's own view of rank 1's far corner: present as a GHOST (owner 1,
  // not owned here) so this rank's cell 1 has all 4 corners available.
  const int v4 = builder.add_vertex(104, 2, 0, 0, /*owner=*/1, /*is_owned=*/false);
  const int v5 = builder.add_vertex(105, 2, 1, 0, /*owner=*/1, /*is_owned=*/false);
  builder.add_cell(fuse::native::CellType::Quad, {v0, v1, v2, v3}, 200, /*owner=*/0, /*is_owned=*/true);
  builder.add_cell(fuse::native::CellType::Quad, {v1, v4, v5, v2}, 201, /*owner=*/1, /*is_owned=*/false);

  const auto built = builder.build();
  ASSERT_TRUE(built.ok()) << built.error().message;
  const auto view = fuse::native::NativeMeshView::create(built.value(), /*my_rank=*/0);
  ASSERT_TRUE(view.ok()) << view.error().message;

  const Mesh mesh = Mesh::from_mesh_view(*view.value());
  EXPECT_EQ(mesh.n_vertices(), 6u);
  EXPECT_EQ(mesh.n_elements(), 2u);
  EXPECT_EQ(mesh.owned_vertex_count(), 4u);   // v0..v3 owned, v4/v5 ghost
  EXPECT_EQ(mesh.owned_element_count(), 1u);  // cell 0 owned, cell 1 ghost

  EXPECT_TRUE(mesh.is_vertex_owned(1));   // v0
  EXPECT_FALSE(mesh.is_vertex_owned(5));  // v4, a ghost
  EXPECT_TRUE(mesh.is_element_owned(1));
  EXPECT_FALSE(mesh.is_element_owned(2));

  EXPECT_EQ(mesh.vertex_global_id(1), 100u);
  EXPECT_EQ(mesh.vertex_global_id(5), 104u);
  EXPECT_EQ(mesh.element_global_id(1), 200u);
  EXPECT_EQ(mesh.element_global_id(2), 201u);

  // The connectivity itself, not only counts and ids: element 1's four
  // corner POSITIONS, read back through element_coordinates, must be
  // v0..v3 in that order -- the thing that would silently read the
  // WRONG vertex (or an out-of-range handle) if from_mesh_view's own
  // 0-based (MeshView) to 1-based (this Mesh) vertex handle shift were
  // ever dropped or off by one.
  std::array<Point, 4> element_1_corners{};
  ASSERT_EQ(mesh.element_coordinates(1, element_1_corners), 4u);
  EXPECT_EQ(element_1_corners[0], Point(0, 0, 0));
  EXPECT_EQ(element_1_corners[1], Point(1, 0, 0));
  EXPECT_EQ(element_1_corners[2], Point(1, 1, 0));
  EXPECT_EQ(element_1_corners[3], Point(0, 1, 0));
}

}  // namespace
