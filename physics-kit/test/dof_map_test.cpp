// physics.dof_map -- fuse_physics::fe::DofMap, adapted from KARMA's
// karma_dof_map.h/.C. See dof_map.hpp's own header comment for how this
// port's identity and numbering diverge from KARMA's own construction.

#include "fuse_physics/fe/dof_map.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#include "fuse/backend/native_mesh.hpp"
#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::Basis;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisKey;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::cell_dimension;
using fuse::physics::fe::cell_family;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::ReferenceTopology;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::EntityHandle;
using fuse::physics::spatial::Mesh;

Variable scalar_p1(std::string name) {
  return Variable{std::move(name), FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
}

// physics.dof_map.SingleVariableNumbersOneDofPerVertex -- on a
// self-generated (every vertex/element owned, identity global ids) 2x1x1
// Hex8 box, a single scalar P1 variable gets exactly one global dof per
// vertex, numbered 0..n_vertices()-1 in ASCENDING GLOBAL-VERTEX-ID order
// (DofKey::operator< sorts by its single corner id for a vertex dof, and
// generate_box's identity numbering makes global id == vertex handle) --
// so the global index of vertex v's dof is EXACTLY v - 1, a fact this
// test holds to exact equality rather than just "distinct and in range".
TEST(DofMapTest, SingleVariableNumbersOneDofPerVertex) {
  Mesh mesh(3);
  mesh.generate_box(2, 1, 1, Point(0, 0, 0), Point(2, 1, 1));
  BasisRegistry registry;
  DofMap dof_map(mesh, {scalar_p1("u")}, registry);

  EXPECT_EQ(dof_map.n_dofs(), static_cast<std::int64_t>(mesh.n_vertices()));
  EXPECT_EQ(dof_map.n_local_dofs(), dof_map.n_dofs());
  EXPECT_EQ(dof_map.first_owned_dof(), 0);
  EXPECT_TRUE(dof_map.ghost_indices().empty());

  for (EntityHandle vertex = 1; vertex <= mesh.n_vertices(); ++vertex) {
    // Find the (unique, since this box's vertices are each a corner of at
    // least one element) element touching this vertex to read its global
    // index back through dof_indices rather than a private accessor.
    bool checked = false;
    for (EntityHandle element = 1; element <= mesh.n_elements() && !checked; ++element) {
      const auto& connectivity = mesh.element_connectivity(element);
      const auto position = std::find(connectivity.begin(), connectivity.end(), vertex);
      if (position == connectivity.end()) continue;
      std::vector<std::int64_t> indices;
      dof_map.dof_indices(element, indices, 0);
      const auto local = static_cast<std::size_t>(std::distance(connectivity.begin(), position));
      EXPECT_EQ(indices.at(local), static_cast<std::int64_t>(mesh.vertex_global_id(vertex)) - 1)
          << "vertex " << vertex;
      checked = true;
    }
    EXPECT_TRUE(checked) << "vertex " << vertex << " touches no element";
  }
}

// physics.dof_map.TwoVariablesOrderVariableMajorThenComponentMinor --
// dof_map.hpp's own stated element ordering (variable-major, then per
// basis association, then component-minor): a scalar "u" (1 component)
// and a vector "velocity" (2 components) on the same P1 Quad4 element
// produce, for that element, u's 4 vertex dofs FIRST, then velocity's 4
// vertices x 2 components = 8 dofs with the x/y pair for each vertex
// adjacent (component-minor).
TEST(DofMapTest, TwoVariablesOrderVariableMajorThenComponentMinor) {
  Mesh mesh(2);
  mesh.generate_2d_box(1, 1, Point(0, 0, 0), Point(1, 1, 0));
  BasisRegistry registry;
  Variable velocity{"velocity", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 2};
  DofMap dof_map(mesh, {scalar_p1("u"), velocity}, registry);

  EXPECT_EQ(dof_map.n_dofs(), static_cast<std::int64_t>(mesh.n_vertices()) * 3);

  std::vector<std::int64_t> all_indices;
  dof_map.dof_indices(1, all_indices);
  ASSERT_EQ(all_indices.size(), 4u + 8u);

  std::vector<std::int64_t> u_indices;
  dof_map.dof_indices(1, u_indices, 0);
  std::vector<std::int64_t> v_indices;
  dof_map.dof_indices(1, v_indices, 1);
  ASSERT_EQ(u_indices.size(), 4u);
  ASSERT_EQ(v_indices.size(), 8u);

  for (std::size_t i = 0; i < u_indices.size(); ++i) EXPECT_EQ(all_indices.at(i), u_indices.at(i));
  for (std::size_t i = 0; i < v_indices.size(); ++i) EXPECT_EQ(all_indices.at(4 + i), v_indices.at(i));

  // component-minor within velocity: entries 0,1 are vertex 0's (x,y),
  // entries 2,3 are vertex 1's (x,y), etc -- no two entries for the SAME
  // vertex are more than one apart.
  for (std::size_t vertex_slot = 0; vertex_slot < 4; ++vertex_slot) {
    EXPECT_NE(v_indices.at(2 * vertex_slot), v_indices.at(2 * vertex_slot + 1));
  }
}

// physics.dof_map.RefusesAGhostDofWithoutACommunicator -- a hand-built
// TWO-"rank" fixture (NativeMeshBuilder, the same construction
// spatial_mesh_test.cpp's own FromMeshViewReadsOwnershipAndGlobalIds
// uses) where this rank's OWNED element straddles the partition
// boundary: two of its four corners are ghost vertices owned by the
// OTHER rank. DofMap's single-rank convenience constructor (no
// communicator) has no way to resolve that ghost's global index, and
// dof_map.hpp's own header comment says so -- refuses by name rather
// than silently numbering the ghost as if it were local.
TEST(DofMapTest, RefusesAGhostDofWithoutACommunicator) {
  fuse::native::NativeMeshBuilder builder(2);
  // 0--1--4
  // |  |  |
  // 3--2--5   (local indices)
  const int v0 = builder.add_vertex(100, 0, 0, 0, /*owner=*/0, /*is_owned=*/true);
  const int v1 = builder.add_vertex(101, 1, 0, 0, /*owner=*/0, /*is_owned=*/true);
  const int v2 = builder.add_vertex(102, 1, 1, 0, /*owner=*/0, /*is_owned=*/true);
  const int v3 = builder.add_vertex(103, 0, 1, 0, /*owner=*/0, /*is_owned=*/true);
  const int v4 = builder.add_vertex(104, 2, 0, 0, /*owner=*/1, /*is_owned=*/false);
  const int v5 = builder.add_vertex(105, 2, 1, 0, /*owner=*/1, /*is_owned=*/false);
  builder.add_cell(fuse::native::CellType::Quad, {v0, v1, v2, v3}, 200, /*owner=*/0, /*is_owned=*/true);
  // THIS rank owns cell 1 too (the one straddling the boundary): v4/v5
  // are ghosts it can see but does not own.
  builder.add_cell(fuse::native::CellType::Quad, {v1, v4, v5, v2}, 201, /*owner=*/0, /*is_owned=*/true);

  const auto built = builder.build();
  ASSERT_TRUE(built.ok()) << built.error().message;
  const auto view = fuse::native::NativeMeshView::create(built.value(), /*my_rank=*/0);
  ASSERT_TRUE(view.ok()) << view.error().message;
  const Mesh mesh = Mesh::from_mesh_view(*view.value());
  ASSERT_EQ(mesh.owned_element_count(), 2u);
  ASSERT_FALSE(mesh.is_vertex_owned(5));  // v4, a ghost

  BasisRegistry registry;
  // The refusal's own code, not merely its type (a type-only EXPECT_THROW
  // would let a mutation that fell through to a DIFFERENT, later
  // std::invalid_argument survive undetected).
  try {
    DofMap dof_map(mesh, {scalar_p1("u")}, registry);
    FAIL() << "a ghost dof without a communicator must be refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("E-PHYSICS-DOFMAP-GHOST-WITHOUT-COMMUNICATOR"), std::string::npos)
        << "refused for another reason: " << e.what();
  }
}

// physics.dof_map.FaceCentreAssociationIndexMatchesSideTopologyOrder --
// CLOSES the open item (docs/dev/physics-kit.md's review):
// cross-check DofAssociation's entity_dimension=2 local_entity_index
// numbering against ReferenceTopology::side_topology()'s own side
// order, for every cell family with a P2 face-centre dof (Hexahedron,
// Prism; Pyramid's base-face dof already named explicitly in
// basis.cpp's own comment). The check is GEOMETRIC and needs no hidden
// table: a face-centre dof's own interpolation point must equal the
// centroid of that side's P1 corner nodes (the side's own
// side_topology(i).reference_nodes(), which are the shared corners by
// construction) -- if dof_associations() ever paired local_entity_index
// i with the WRONG side, this centroid comparison would catch it
// (two different sides of a Hexahedron/Prism/Pyramid have different
// centroids).
void check_face_centres_match_side_topology(GeometryVariant variant, unsigned n_quad_face_centres) {
  // Every face this wave's built-in P2 bases place a centre dof on (the
  // Hexahedron's 6, the Prism's 3 quad side faces, the Pyramid's 1 quad
  // base) is a QUADRILATERAL -- its corner count is 4, read directly from
  // side_nodes(side)'s own "corners first" CN convention
  // (reference_topology.hpp's own comment) rather than through
  // side_topology(side), which for a quadratic cell returns the FACE'S
  // OWN quadratic shape (Quad9) -- averaging all 9 of ITS nodes would
  // happen to also land on the centre by symmetry for this wave's
  // reference elements, but would be the wrong check to rely on; reading
  // only the first 4 (corner) indices out of the TOP cell's own node
  // array is correct independent of that symmetry.
  constexpr std::size_t kCornersPerQuadFace = 4;
  const auto cell = cell_family(variant);
  BasisRegistry registry;
  const Basis basis = registry.create(BasisKey{cell, FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG});
  const ReferenceTopology topology(variant);
  const auto& associations = basis.dof_associations();
  const auto& points = basis.interpolation_points();
  ASSERT_EQ(associations.size(), points.size());

  unsigned checked = 0;
  for (std::size_t i = 0; i < associations.size(); ++i) {
    if (associations[i].entity_dimension != 2) continue;
    const unsigned side = associations[i].local_entity_index;
    const auto node_indices = topology.side_nodes(side);
    ASSERT_GE(node_indices.size(), kCornersPerQuadFace) << "side=" << side;
    Point centroid(0, 0, 0);
    for (std::size_t c = 0; c < kCornersPerQuadFace; ++c)
      centroid = centroid + topology.reference_nodes().at(node_indices.at(c));
    centroid = centroid * (1.0 / static_cast<double>(kCornersPerQuadFace));

    const Point reported = points.at(i);
    EXPECT_NEAR(norm(reported - centroid), 0.0, 1e-12)
        << "variant=" << static_cast<int>(variant) << " side=" << side;
    ++checked;
  }
  EXPECT_EQ(checked, n_quad_face_centres);
}

TEST(DofMapTest, FaceCentreAssociationIndexMatchesSideTopologyOrderHex27) {
  check_face_centres_match_side_topology(GeometryVariant::Hex27, 6);
}
TEST(DofMapTest, FaceCentreAssociationIndexMatchesSideTopologyOrderPrism18) {
  check_face_centres_match_side_topology(GeometryVariant::Prism18, 3);
}
TEST(DofMapTest, FaceCentreAssociationIndexMatchesSideTopologyOrderPyramid14) {
  check_face_centres_match_side_topology(GeometryVariant::Pyramid14, 1);
}

// physics.dof_map.OrientationCodeAgreesBetweenTwoElementsSharingAFace --
// element 1 and element 2 of a 2x1x1 Hex8 box share one quadrilateral
// face; each computes orientation_code() for ITS OWN local view of that
// face, and applying each one's own local_to_canonical_vertices mapping
// to ITS OWN local corner GLOBAL ids must reproduce the SAME canonical
// sequence (the whole point of a canonical form: two elements agree on
// it without either one's local order being privileged).
TEST(DofMapTest, OrientationCodeAgreesBetweenTwoElementsSharingAFace) {
  Mesh mesh(3);
  mesh.generate_box(2, 1, 1, Point(0, 0, 0), Point(2, 1, 1));
  BasisRegistry registry;
  DofMap dof_map(mesh, {scalar_p1("u")}, registry);

  const auto neighbor = mesh.neighbor(1, 1);
  ASSERT_TRUE(neighbor.has_value());
  const EntityHandle other = *neighbor;
  unsigned other_side = 0;
  bool found = false;
  for (unsigned s = 0; s < mesh.n_sides(other); ++s) {
    if (mesh.neighbor(other, s) == EntityHandle{1}) {
      other_side = s;
      found = true;
      break;
    }
  }
  ASSERT_TRUE(found);

  const auto code_1 = dof_map.orientation_code(1, 2, 1);
  const auto code_2 = dof_map.orientation_code(other, 2, other_side);

  using fuse::physics::fe::local_to_canonical_vertices;
  using fuse::physics::fe::QuadrilateralOrientation;

  const auto local_1 = mesh.side_vertices(1, 1);
  const auto local_2 = mesh.side_vertices(other, other_side);
  const auto permutation_1 = local_to_canonical_vertices(static_cast<QuadrilateralOrientation>(code_1));
  const auto permutation_2 = local_to_canonical_vertices(static_cast<QuadrilateralOrientation>(code_2));

  std::array<std::uint64_t, 4> canonical_1{};
  std::array<std::uint64_t, 4> canonical_2{};
  for (unsigned i = 0; i < 4; ++i) {
    canonical_1[permutation_1[i]] = mesh.vertex_global_id(local_1[i]);
    canonical_2[permutation_2[i]] = mesh.vertex_global_id(local_2[i]);
  }
  EXPECT_EQ(canonical_1, canonical_2);
}


// physics.dof_map.P2Hex27DofMapCountsVerticesEdgesFacesAndCells -- a
// 3-D cell's own interior-edge dofs (e.g. a Hex27 edge midpoint) are
// numbered, via ReferenceTopology's own edge-to-corner table and
// Mesh::canonical_edge_vertex_ids (reference_topology.hpp's own header
// comment), so this case is never refused. A 2x1x1 Hex8 box's P2 (Hex27) global dof count
// is the STANDARD structured-grid entity count -- derived, not
// guessed:
//   vertices: (nx+1)(ny+1)(nz+1) = 3*2*2 = 12
//   edges:    nx*(ny+1)*(nz+1) + (nx+1)*ny*(nz+1) + (nx+1)*(ny+1)*nz
//           = 2*2*2 + 3*1*2 + 3*2*1 = 8 + 6 + 6 = 20
//   faces:    (nx+1)*ny*nz + nx*(ny+1)*nz + nx*ny*(nz+1)
//           = 3*1*1 + 2*2*1 + 2*1*2 = 3 + 4 + 4 = 11
//   cells:    nx*ny*nz = 2
// total = 12 + 20 + 11 + 2 = 45 (Hex27's 27th node is a cell-interior
// dof, basis.cpp's own dof_associations(); Tet10 below has none).
TEST(DofMapTest, P2Hex27DofMapCountsVerticesEdgesFacesAndCells) {
  Mesh mesh(3);
  mesh.generate_box(2, 1, 1, Point(0, 0, 0), Point(2, 1, 1));
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry);
  EXPECT_EQ(dof_map.n_dofs(), 45);
}

// physics.dof_map.P2Tet10DofMapNumbersEachSharedEdgeDofOnce -- two Tet4
// elements sharing one triangular FACE (so three of their edges --
// {v0,v1}, {v1,v2}, {v2,v0} -- are shared): element 1 =
// {v0,v1,v2,v3} (apex (0,0,1)), element 2 = {v0,v1,v2,v4} (apex
// (0,0,-1)), both using the SAME local corner order 0,1,2 for the
// shared face, so fe::edge_vertices(Tetrahedron, e)'s own table
// ({0,1},{1,2},{2,0},...) names the SAME three physical edges at local
// edge index 0, 1, 2 in BOTH elements' own connectivity. Tet10 has no
// interior (cell) dof at all (basis.cpp's own dof_associations():
// size() == 4 vertices + 6 edges exactly, nothing left over) -- total
// dof count = 5 vertices + 9 distinct edges (3 shared + 3 unique to
// each element) = 14, and the three SHARED edges' global dof indices
// must agree between the two elements' own dof_indices() (the point of
// this item: a shared edge dof is numbered ONCE, not twice).
TEST(DofMapTest, P2Tet10DofMapNumbersEachSharedEdgeDofOnce) {
  Mesh mesh(3);
  mesh.add_vertex(Point(0, 0, 0));   // v0 (handle 1)
  mesh.add_vertex(Point(1, 0, 0));   // v1 (handle 2)
  mesh.add_vertex(Point(0, 1, 0));   // v2 (handle 3)
  mesh.add_vertex(Point(0, 0, 1));   // v3, element 1's own apex (handle 4)
  mesh.add_vertex(Point(0, 0, -1));  // v4, element 2's own apex (handle 5)
  mesh.add_element(GeometryVariant::Tet4, {1, 2, 3, 4});
  mesh.add_element(GeometryVariant::Tet4, {1, 2, 3, 5});

  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry);
  EXPECT_EQ(dof_map.n_dofs(), 14);

  std::vector<std::int64_t> indices_1;
  std::vector<std::int64_t> indices_2;
  dof_map.dof_indices(1, indices_1, 0);
  dof_map.dof_indices(2, indices_2, 0);
  ASSERT_EQ(indices_1.size(), 10u);
  ASSERT_EQ(indices_2.size(), 10u);

  // Edge positions 4, 5, 6 (after the 4 vertex dofs) are edges {0,1},
  // {1,2}, {2,0} in fe::edge_vertices' own order -- the three SHARED
  // edges, which must carry the SAME global index in both elements.
  for (unsigned e = 0; e < 3; ++e) EXPECT_EQ(indices_1.at(4 + e), indices_2.at(4 + e)) << "shared edge " << e;
  // Edge positions 7, 8, 9 are {0,3}, {1,3}, {2,3} -- each element's
  // own edge to its OWN apex, which must be DISTINCT dofs (element 1's
  // apex v3 and element 2's apex v4 are different vertices).
  for (unsigned e = 3; e < 6; ++e) EXPECT_NE(indices_1.at(4 + e), indices_2.at(4 + e)) << "own edge " << e;
}


// physics.dof_map.D102SharedSideOwnedByTheMinimumCornerOwnerRank --
// tested DIRECTLY against
// fe::shared_sub_entity_owned_here rather than through DofMap's own
// full single-rank constructor: that constructor's "convenience path"
// refuses ANY ghost vertex outright (no communicator to resolve one),
// so a fixture with a genuinely split-owned CORNER vertex always
// throws there regardless of which sub-entity-ownership RULE is under
// test -- the rule itself only matters once a ghost is tolerated and
// resolved through a real communicator (the collective path, build-
// verified only, never run in this sandbox; host_pending has the
// commands). This test instead builds a Mesh via from_mesh_view with
// THREE vertices of known, DIFFERENT owner ranks and checks the rule's
// own three cases directly:
//   * every corner owned by rank 0 (the local rank) -- owned here
//     under BOTH the old and the new rule (no partition question);
//   * corners split 0/1, minimum owner 0 -- owned here
//     (the "every corner owned" rule would have been false, since not EVERY corner
//     is locally owned);
//   * corners split 1/2 (neither is this rank) -- not owned here,
//     correctly, under either rule.
TEST(DofMapTest, D102SharedSideOwnedByTheMinimumCornerOwnerRank) {
  fuse::native::NativeMeshBuilder builder(2);
  const int v0 = builder.add_vertex(100, 0, 0, 0, /*owner=*/0, /*is_owned=*/true);
  const int v1 = builder.add_vertex(101, 1, 0, 0, /*owner=*/0, /*is_owned=*/true);
  const int v2 = builder.add_vertex(102, 2, 0, 0, /*owner=*/1, /*is_owned=*/false);
  const int v3 = builder.add_vertex(103, 3, 0, 0, /*owner=*/2, /*is_owned=*/false);
  // A single triangle so the builder has a valid cell to finalize; its
  // own ownership is irrelevant to this test (shared_sub_entity_owned_
  // here reads only vertex_owner/is_vertex_owned, never element data).
  builder.add_cell(fuse::native::CellType::Triangle, {v0, v1, v2}, 200, /*owner=*/0, /*is_owned=*/true);

  const auto built = builder.build();
  ASSERT_TRUE(built.ok()) << built.error().message;
  const auto view = fuse::native::NativeMeshView::create(built.value(), /*my_rank=*/0);
  ASSERT_TRUE(view.ok()) << view.error().message;
  const Mesh mesh = Mesh::from_mesh_view(*view.value());
  ASSERT_EQ(mesh.vertex_owner(1), 0);  // v0
  ASSERT_EQ(mesh.vertex_owner(2), 0);  // v1
  ASSERT_EQ(mesh.vertex_owner(3), 1);  // v2
  ASSERT_TRUE(mesh.is_vertex_owned(1));
  ASSERT_TRUE(mesh.is_vertex_owned(2));
  ASSERT_FALSE(mesh.is_vertex_owned(3));

  EXPECT_TRUE(fuse::physics::fe::shared_sub_entity_owned_here(mesh, {1, 2}));   // v0,v1: both owner 0
  EXPECT_TRUE(fuse::physics::fe::shared_sub_entity_owned_here(mesh, {1, 3}));   // v0,v2: min(0,1)=0
  EXPECT_FALSE(fuse::physics::fe::shared_sub_entity_owned_here(mesh, {3}));     // v2 alone: owner 1, not here
  EXPECT_FALSE(fuse::physics::fe::shared_sub_entity_owned_here(mesh, {}));      // no corners: owned nowhere
}

}  // namespace
