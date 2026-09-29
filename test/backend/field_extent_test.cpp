// "Nodal fields" track, step 1: the ONE definition of a field's extent from
// (disc, order, mesh entity set) -- mesh_view.hpp's `field_extent`/`region_extent_kind`
// -- and its wiring into `make_region_sizes` (serial_backend.cpp), which every
// caller of RegionSizes (schedule/buffers.cpp's `plan_buffers`, and therefore
// `schedule()`) goes through.
//
// Covers: FV/DiscontinuousGLL order 0 (cell count) and ContinuousGLL order 1
// (vertex count) on 1D (line), 2D (quad AND triangle) and 3D (hexbox AND
// tetbox) generated meshes; the two named rejections (E-EXTENT-DISC for an
// unsupported (disc, order), E-EXTENT-MIXED for one mesh entity set carrying both a
// cell field and a nodal one); and that a mesh entity set with no field on it still
// defaults to the old, cell-based behaviour.

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/error.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace {

using fuse::ComponentDecl;
using fuse::CouplingGraph;
using fuse::Discretization;
using fuse::FieldSpec;
using fuse::MeshDecl;
using fuse::MeshId;
using fuse::Rational;
using fuse::Region;
using fuse::RegionId;
using fuse::field_extent;
using fuse::region_extent_kind;
using fuse::RegionExtentKind;

// ---------------------------------------------------------------------------
// field_extent(): direct, no graph involved
// ---------------------------------------------------------------------------

TEST(FieldExtent, FvAndDiscontinuousGllOrderZeroReadTheCellCount) {
  fuse::SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:line:7"}).ok());
  const auto view = b.view(MeshId{0});
  ASSERT_TRUE(view.ok());

  const auto fv = field_extent(Discretization::FV, 0, *view.value(), 1);
  ASSERT_TRUE(fv.ok());
  EXPECT_EQ(fv.value(), 7u);

  const auto dg0 =
      field_extent(Discretization::DiscontinuousGLL, 0, *view.value(), 1);
  ASSERT_TRUE(dg0.ok());
  EXPECT_EQ(dg0.value(), 7u);
}

TEST(FieldExtent, ContinuousGllOrderOneReadsTheVertexCountNotTheRegionDim) {
  fuse::SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:line:7"}).ok());
  const auto view = b.view(MeshId{0});
  ASSERT_TRUE(view.ok());

  // region_dim = 1 (the line's cells), yet the answer is the VERTEX count:
  // field_extent ignores region_dim entirely for this combination.
  const auto nodal =
      field_extent(Discretization::ContinuousGLL, 1, *view.value(), 1);
  ASSERT_TRUE(nodal.ok());
  EXPECT_EQ(nodal.value(), 8u);  // 7 cells -> 8 vertices
}

TEST(FieldExtent, EveryUnimplementedCombinationRefusesNamedly) {
  fuse::SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:line:4"}).ok());
  const auto view = b.view(MeshId{0});
  ASSERT_TRUE(view.ok());

  for (const auto& [disc, order] :
       {std::pair{Discretization::ContinuousGLL, 0},
        std::pair{Discretization::ContinuousGLL, 2},
        std::pair{Discretization::DiscontinuousGLL, 1},
        std::pair{Discretization::PointCloud, 0},
        std::pair{Discretization::PointCloud, 1}}) {
    const auto r = field_extent(disc, order, *view.value(), 1);
    ASSERT_FALSE(r.ok()) << fuse::to_cstring(disc) << " order " << order;
    EXPECT_EQ(r.error().code, "E-EXTENT-DISC")
        << fuse::to_cstring(disc) << " order " << order;
  }
}

// ---------------------------------------------------------------------------
// region_extent_kind(): what the GRAPH's fields imply for one mesh entity set
// ---------------------------------------------------------------------------

namespace {

/// One mesh entity set on one line mesh, with a single field of `disc`/`order` on it
/// (or none at all when `disc` is left at its default and no field is added).
CouplingGraph one_field_graph(Discretization disc, int order,
                             std::string* field_name = nullptr) {
  CouplingGraph g;
  const MeshId m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:5"});
  const RegionId r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});
  const auto c = g.add_component(
      ComponentDecl{.name = "c", .ranks = {0, 1}, .interval = Rational{1}});
  const auto f = g.add_field(
      c, FieldSpec{.name = "f", .region = r, .disc = disc, .order = order});
  if (field_name != nullptr) *field_name = g.field(f).name;
  return g;
}

}  // namespace

TEST(RegionExtentKind, ARegionWithNoFieldDefaultsToCell) {
  CouplingGraph g;
  const MeshId m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:5"});
  const RegionId r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});
  const auto kind = region_extent_kind(g, r);
  ASSERT_TRUE(kind.ok());
  EXPECT_EQ(kind.value(), RegionExtentKind::Cell);
}

TEST(RegionExtentKind, AnFvFieldMeansCellAndAnOrderOneNodalFieldMeansVertex) {
  {
    const CouplingGraph g = one_field_graph(Discretization::FV, 0);
    const auto kind = region_extent_kind(g, RegionId{0});
    ASSERT_TRUE(kind.ok());
    EXPECT_EQ(kind.value(), RegionExtentKind::Cell);
  }
  {
    const CouplingGraph g =
        one_field_graph(Discretization::DiscontinuousGLL, 0);
    const auto kind = region_extent_kind(g, RegionId{0});
    ASSERT_TRUE(kind.ok());
    EXPECT_EQ(kind.value(), RegionExtentKind::Cell);
  }
  {
    const CouplingGraph g = one_field_graph(Discretization::ContinuousGLL, 1);
    const auto kind = region_extent_kind(g, RegionId{0});
    ASSERT_TRUE(kind.ok());
    EXPECT_EQ(kind.value(), RegionExtentKind::Vertex);
  }
}

TEST(RegionExtentKind, AnUnimplementedFieldDiscretizationRefuses) {
  const CouplingGraph g = one_field_graph(Discretization::ContinuousGLL, 2);
  const auto kind = region_extent_kind(g, RegionId{0});
  ASSERT_FALSE(kind.ok());
  EXPECT_EQ(kind.error().code, "E-EXTENT-DISC");
}

TEST(RegionExtentKind, ACellFieldAndANodalFieldOnOneRegionRefusesAsMixed) {
  CouplingGraph g;
  const MeshId m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:5"});
  const RegionId r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});
  const auto c = g.add_component(
      ComponentDecl{.name = "c", .ranks = {0, 1}, .interval = Rational{1}});
  g.add_field(c, FieldSpec{.name = "cell_field", .region = r,
                          .disc = Discretization::FV, .order = 0});
  g.add_field(c, FieldSpec{.name = "nodal_field", .region = r,
                          .disc = Discretization::ContinuousGLL, .order = 1});
  const auto kind = region_extent_kind(g, r);
  ASSERT_FALSE(kind.ok());
  EXPECT_EQ(kind.error().code, "E-EXTENT-MIXED");
  EXPECT_NE(kind.error().message.find("cell_field"), std::string::npos);
  EXPECT_NE(kind.error().message.find("nodal_field"), std::string::npos);
}

// ---------------------------------------------------------------------------
// make_region_sizes(): end to end, on 1D, 2D (quad + tri) and 3D (hex + tet)
// generated meshes -- both backends this task owns.
// ---------------------------------------------------------------------------

namespace {

/// One mesh entity set on one mesh, with one field of `disc`/`order` declared on it,
/// ready for `make_region_sizes`.
CouplingGraph sizing_graph(const std::string& source, Discretization disc,
                          int order, int region_dim = -1) {
  CouplingGraph g;
  const MeshId m = g.add_mesh(MeshDecl{.name = "m", .source = source});
  const RegionId r =
      g.add_region(Region{.name = "r", .mesh = m, .dimension = region_dim});
  const auto c = g.add_component(
      ComponentDecl{.name = "c", .ranks = {0, 1}, .interval = Rational{1}});
  g.add_field(
      c, FieldSpec{.name = "f", .region = r, .disc = disc, .order = order});
  return g;
}

}  // namespace

TEST(MakeRegionSizesNodal, LineMeshVertexCountIsCellsPlusOne) {
  const CouplingGraph g =
      sizing_graph("generated:line:9", Discretization::ContinuousGLL, 1, 1);
  fuse::SerialMeshBackend b;
  ASSERT_TRUE(b.load(g.mesh(MeshId{0})).ok());
  const fuse::RegionSizes sizes = fuse::make_region_sizes(g, b);
  EXPECT_EQ(sizes(RegionId{0}, -1), 10u);
}

TEST(MakeRegionSizesNodal, BoxMeshVertexCountIsTheStructuredGridFormula) {
  // 3x2x4 cells -> 4x3x5 vertices.
  const CouplingGraph g =
      sizing_graph("generated:box:3x2x4", Discretization::ContinuousGLL, 1, 3);
  fuse::SerialMeshBackend b;
  ASSERT_TRUE(b.load(g.mesh(MeshId{0})).ok());
  const fuse::RegionSizes sizes = fuse::make_region_sizes(g, b);
  EXPECT_EQ(sizes(RegionId{0}, -1), 4u * 3u * 5u);
  // The same mesh entity set's CELL count is unaffected -- region_size() on its own,
  // with no field in view, still answers the way it always has.
  EXPECT_EQ(b.region_size(g.region(RegionId{0})).value(), 3u * 2u * 4u);
}

TEST(MakeRegionSizesNodal, QuadAndTriangleMeshesOnTheSameGridShareVertexCounts) {
  using fuse::native::NativeMeshBackend;
  // 4x3 quads: (4+1)*(3+1) = 20 vertices either way -- a triangle mesh only
  // subdivides each quad's INTERIOR (a shared diagonal), so it introduces no
  // new vertex.
  for (const std::string& kind : {"quad", "tri"}) {
    const CouplingGraph g = sizing_graph("generated:" + kind + ":4x3",
                                        Discretization::ContinuousGLL, 1, 2);
    NativeMeshBackend b(/*rank=*/0, /*num_ranks=*/1);
    ASSERT_TRUE(b.load(g.mesh(MeshId{0})).ok()) << kind;
    const fuse::RegionSizes sizes = fuse::make_region_sizes(g, b);
    EXPECT_EQ(sizes(RegionId{0}, -1), 20u) << kind;
  }

  // Cell counts differ as expected: 12 quads, 24 triangles.
  {
    const CouplingGraph gq =
        sizing_graph("generated:quad:4x3", Discretization::FV, 0, 2);
    NativeMeshBackend bq(0, 1);
    ASSERT_TRUE(bq.load(gq.mesh(MeshId{0})).ok());
    EXPECT_EQ(fuse::make_region_sizes(gq, bq)(RegionId{0}, -1), 12u);
  }
  {
    const CouplingGraph gt =
        sizing_graph("generated:tri:4x3", Discretization::FV, 0, 2);
    NativeMeshBackend bt(0, 1);
    ASSERT_TRUE(bt.load(gt.mesh(MeshId{0})).ok());
    EXPECT_EQ(fuse::make_region_sizes(gt, bt)(RegionId{0}, -1), 24u);
  }
}

TEST(MakeRegionSizesNodal, HexAndTetMeshesOnTheSameGridShareVertexCounts) {
  using fuse::native::NativeMeshBackend;
  // 2x2x3 boxes: (2+1)*(2+1)*(3+1) = 36 vertices either way -- tetbox only
  // splits each hex's INTERIOR (the Kuhn decomposition), introducing no new
  // vertex.
  for (const std::string& kind : {"hexbox", "tetbox"}) {
    const CouplingGraph g = sizing_graph("generated:" + kind + ":2x2x3",
                                        Discretization::ContinuousGLL, 1, 3);
    NativeMeshBackend b(0, 1);
    ASSERT_TRUE(b.load(g.mesh(MeshId{0})).ok()) << kind;
    const fuse::RegionSizes sizes = fuse::make_region_sizes(g, b);
    EXPECT_EQ(sizes(RegionId{0}, -1), 36u) << kind;
  }

  // Cell counts differ as expected: 12 hexes, 72 tets (6 per hex).
  {
    const CouplingGraph gh =
        sizing_graph("generated:hexbox:2x2x3", Discretization::FV, 0, 3);
    NativeMeshBackend bh(0, 1);
    ASSERT_TRUE(bh.load(gh.mesh(MeshId{0})).ok());
    EXPECT_EQ(fuse::make_region_sizes(gh, bh)(RegionId{0}, -1), 12u);
  }
  {
    const CouplingGraph gt =
        sizing_graph("generated:tetbox:2x2x3", Discretization::FV, 0, 3);
    NativeMeshBackend bt(0, 1);
    ASSERT_TRUE(bt.load(gt.mesh(MeshId{0})).ok());
    EXPECT_EQ(fuse::make_region_sizes(gt, bt)(RegionId{0}, -1), 72u);
  }
}

TEST(MakeRegionSizesNodal, RefusesANamedEntitySubsetForANodalField) {
  CouplingGraph g;
  const MeshId m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:box:2x2x2"});
  const RegionId r = g.add_region(
      Region{.name = "r", .mesh = m, .entity_set = "left", .dimension = 2});
  const auto c = g.add_component(
      ComponentDecl{.name = "c", .ranks = {0, 1}, .interval = Rational{1}});
  g.add_field(c, FieldSpec{.name = "f", .region = r,
                          .disc = Discretization::ContinuousGLL, .order = 1});
  fuse::SerialMeshBackend b;
  ASSERT_TRUE(b.load(g.mesh(MeshId{0})).ok());
  const fuse::RegionSizes sizes = fuse::make_region_sizes(g, b);
  EXPECT_THROW((void)sizes(r, -1), fuse::InvariantViolation);
}

TEST(MakeRegionSizesNodal, ThrowsRatherThanGuessingForAnUnimplementedDiscretization) {
  const CouplingGraph g =
      sizing_graph("generated:line:5", Discretization::ContinuousGLL, 2, 1);
  fuse::SerialMeshBackend b;
  ASSERT_TRUE(b.load(g.mesh(MeshId{0})).ok());
  const fuse::RegionSizes sizes = fuse::make_region_sizes(g, b);
  EXPECT_THROW((void)sizes(RegionId{0}, -1), fuse::InvariantViolation);
}

TEST(MakeRegionSizesNodal, ThrowsRatherThanSilentlyPickingOneShapeForAMixedRegion) {
  CouplingGraph g;
  const MeshId m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:5"});
  const RegionId r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});
  const auto c = g.add_component(
      ComponentDecl{.name = "c", .ranks = {0, 1}, .interval = Rational{1}});
  g.add_field(c, FieldSpec{.name = "cell_field", .region = r,
                          .disc = Discretization::FV, .order = 0});
  g.add_field(c, FieldSpec{.name = "nodal_field", .region = r,
                          .disc = Discretization::ContinuousGLL, .order = 1});
  fuse::SerialMeshBackend b;
  ASSERT_TRUE(b.load(g.mesh(MeshId{0})).ok());
  const fuse::RegionSizes sizes = fuse::make_region_sizes(g, b);
  EXPECT_THROW((void)sizes(r, -1), fuse::InvariantViolation);
}

}  // namespace
