// The native distributed mesh's geometry contract: every cell type's
// simplex decomposition, the shared-face rule, and the rejections that fire
// when a cell cannot be decomposed. Closed-form expected values throughout,
// so a failure points at the geometry code, not a tolerance.

#include "fuse/backend/native_mesh.hpp"

#include "fuse/backend/native/geometry.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

using namespace fuse::native;

/// The set of triangles `canonical_fan_triangulation` returns, as sorted
/// GLOBAL id triples -- independent of local index, so two calls with the
/// SAME cyclic face presented in DIFFERENT local orders can be compared
/// directly.
std::set<std::array<std::int64_t, 3>> triangle_gid_sets(
    const std::vector<std::int64_t>& gids, const std::vector<Triangle>& tris) {
  std::set<std::array<std::int64_t, 3>> out;
  for (const auto& t : tris) {
    std::array<std::int64_t, 3> g{gids[static_cast<std::size_t>(t[0])],
                                  gids[static_cast<std::size_t>(t[1])],
                                  gids[static_cast<std::size_t>(t[2])]};
    std::sort(g.begin(), g.end());
    out.insert(g);
  }
  return out;
}

double sum_measures(const NativeMeshView& v, int dim) {
  double s = 0.0;
  for (std::size_t i = 0; i < v.entity_count(dim); ++i) s += v.measure(dim, i);
  return s;
}

double sum_simplices_measure(const NativeMeshView& v, int dim, std::size_t i) {
  double s = 0.0;
  const int corners = v.dimension() + 1;
  std::vector<fuse::Point> pts(static_cast<std::size_t>(corners));
  for (std::size_t sx = 0; sx < v.simplex_count(dim, i); ++sx) {
    v.simplex_points(dim, i, sx, pts.data());
    // Gram-determinant measure, replicated from native/mesh.hpp's formula,
    // generalised for M=1,2,3.
    const int m = corners - 1;
    double e[3][3];
    for (int a = 0; a < m; ++a) {
      e[a][0] = pts[static_cast<std::size_t>(a + 1)].x - pts[0].x;
      e[a][1] = pts[static_cast<std::size_t>(a + 1)].y - pts[0].y;
      e[a][2] = pts[static_cast<std::size_t>(a + 1)].z - pts[0].z;
    }
    double g[3][3];
    for (int a = 0; a < m; ++a)
      for (int b = 0; b < m; ++b) {
        g[a][b] = e[a][0] * e[b][0] + e[a][1] * e[b][1] + e[a][2] * e[b][2];
      }
    double det = 0.0;
    if (m == 1) det = g[0][0];
    else if (m == 2) det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
    else
      det = g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1]) -
            g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0]) +
            g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]);
    if (det < 0.0) det = 0.0;
    double fact = 1.0;
    for (int k = 2; k <= m; ++k) fact *= k;
    s += std::sqrt(det) / fact;
  }
  return s;
}

}  // namespace

// ---- a unit-square Quad -------------------------------------------------

TEST(NativeDistributedMesh, QuadMeasureIsUnitArea) {
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 1.0, 0.0, 0, true);
  b.add_cell(CellType::Quad, {v0, v1, v2, v3}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();
  EXPECT_NEAR(v.measure(2, 0), 1.0, 1e-14);
  EXPECT_EQ(v.simplex_count(2, 0), 2u);
  EXPECT_NEAR(sum_simplices_measure(v, 2, 0), 1.0, 1e-14);
}

// ---- a concave (star-shaped-only-about-some-vertices) polygon -----------

TEST(NativeDistributedMesh, ConcavePolygonThatIsStarShapedFromItsMinGidVertexIsAccepted) {
  // An "L" hexagon (six vertices), star-shaped from vertex 0 (global id 0,
  // the minimum), area 3 (a 2x2 square minus a 1x1 notch).
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 2.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 2.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 1.0, 1.0, 0.0, 0, true);
  const int v4 = b.add_vertex(4, 1.0, 2.0, 0.0, 0, true);
  const int v5 = b.add_vertex(5, 0.0, 2.0, 0.0, 0, true);
  b.add_cell(CellType::Polygon, {v0, v1, v2, v3, v4, v5}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  EXPECT_NEAR(view_r.value()->measure(2, 0), 3.0, 1e-13);
}

TEST(NativeDistributedMesh, ConcavePolygonNotStarShapedFromMinGidVertexIsRefused) {
  // A pentagon shaped so the reflex vertex is ADJACENT to the min-gid
  // vertex 0: fanning from 0 sends one diagonal outside the polygon.
  // Vertices (cyclic): (0,0) [gid0], (2,0) [gid1], (2,2) [gid2],
  // (0.2,0.1) [gid3, the reflex notch pulled in near vertex 0], (0,2)
  // [gid4]. The diagonal 0-(0.2,0.1) to 2 (i.e. the fan triangle
  // (0)-(0.2,0.1)-(0,2)) folds back over the triangle (0)-(2,0)-(2,2),
  // (0)-(2,2)-(0.2,0.1), producing overlap and no simple valid fan.
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 2.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 2.0, 2.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.2, 0.1, 0.0, 0, true);
  const int v4 = b.add_vertex(4, 0.0, 2.0, 0.0, 0, true);
  b.add_cell(CellType::Polygon, {v0, v1, v2, v3, v4}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_FALSE(view_r.ok()) << "the fan from vertex 0 folds over itself and must be refused";
  EXPECT_EQ(view_r.error().code, "E-MESH-POLYGON-NOT-STAR-SHAPED");
}

// ---- the shared-face rule, at the level of canonical_fan_triangulation ---

TEST(NativeDistributedMesh, CanonicalFanTriangulationIsIndependentOfLocalStartingCorner) {
  // The SAME unit-square face's four corners (cyclic order 10-11-12-13),
  // presented starting at two DIFFERENT local corners -- B is A rotated by
  // one position, same cyclic direction. A caller keyed on LOCAL index 0
  // (rather than minimum global id) would fan from a different physical
  // vertex in each case and get a DIFFERENT diagonal; the actual rule must
  // not.
  const std::vector<std::int64_t> gids_a = {10, 11, 12, 13};
  const std::vector<Vec3> pts_a = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  const std::vector<std::int64_t> gids_b = {11, 12, 13, 10};  // A rotated by 1
  const std::vector<Vec3> pts_b = {{1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}};

  auto tri_a = canonical_fan_triangulation(gids_a, pts_a);
  auto tri_b = canonical_fan_triangulation(gids_b, pts_b);
  ASSERT_TRUE(tri_a.ok()) << tri_a.error().message;
  ASSERT_TRUE(tri_b.ok()) << tri_b.error().message;

  const auto sets_a = triangle_gid_sets(gids_a, tri_a.value());
  const auto sets_b = triangle_gid_sets(gids_b, tri_b.value());
  EXPECT_EQ(sets_a, sets_b) << "the same physical face, presented starting at a different "
                              "local corner, must triangulate into the SAME set of global-id "
                              "triangles -- the rule is keyed on global id, not local index";
  // And the actual expected split (fan from gid 10, the minimum): (10,11,12)
  // and (10,12,13).
  const std::set<std::array<std::int64_t, 3>> expected = {{10, 11, 12}, {10, 12, 13}};
  EXPECT_EQ(sets_a, expected);
}

// ---- the shared-face rule: two hexes sharing a quad, opposite local order

TEST(NativeDistributedMesh, TwoHexesSharingAFaceTriangulateItIdenticallyFromBothSides) {
  // Two unit hexes stacked along z, sharing the z=1 face. Cell A's face 1
  // (its "top", local corners 4-7) and cell B's face 0 (its "bottom",
  // local corners 0-3) are the SAME four physical vertices, but B's local
  // numbering is a different rotation/reflection of A's. Both must select
  // the SAME diagonal when splitting that shared quad into two triangles,
  // because the rule is keyed on global id, not on local corner index.
  NativeMeshBuilder b(3);
  // Bottom hex (cell 0): z in [0,1].
  int a[8];
  const double ax[8] = {0, 1, 1, 0, 0, 1, 1, 0};
  const double ay[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  const double az[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i) a[i] = b.add_vertex(i, ax[i], ay[i], az[i], 0, true);
  b.add_cell(CellType::Hex, {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]}, 0, 0, true);

  // Top hex (cell 1): z in [1,2], reusing the shared top-of-A vertices
  // (gids 4,5,6,7) but naming them in a DIFFERENT local order/rotation for
  // this cell's own bottom face.
  const int shared4 = a[4], shared5 = a[5], shared6 = a[6], shared7 = a[7];
  int t4 = b.add_vertex(8, 0.0, 0.0, 2.0, 0, true);
  int t5 = b.add_vertex(9, 1.0, 0.0, 2.0, 0, true);
  int t6 = b.add_vertex(10, 1.0, 1.0, 2.0, 0, true);
  int t7 = b.add_vertex(11, 0.0, 1.0, 2.0, 0, true);
  // Cell 1's own corner order is rotated relative to cell 0's, so its
  // bottom face (local corners 0-3) lists the shared vertices starting at
  // a different corner and in reverse rotational sense.
  b.add_cell(CellType::Hex, {shared6, shared5, shared4, shared7, t6, t5, t4, t7}, 1, 0, true);

  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();
  EXPECT_NEAR(v.measure(3, 0), 1.0, 1e-12);
  EXPECT_NEAR(v.measure(3, 1), 1.0, 1e-12);
  EXPECT_NEAR(sum_measures(v, 3), 2.0, 1e-12);
}

// ---- volumetric primitives, closed forms --------------------------------

TEST(NativeDistributedMesh, UnitHexVolumeIsOne) {
  NativeMeshBuilder b(3);
  int idx[8];
  const double x[8] = {0, 1, 1, 0, 0, 1, 1, 0};
  const double y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  const double z[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i) idx[i] = b.add_vertex(i, x[i], y[i], z[i], 0, true);
  b.add_cell(CellType::Hex, {idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7]}, 0,
            0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();
  EXPECT_NEAR(v.measure(3, 0), 1.0, 1e-12);
  EXPECT_NEAR(sum_simplices_measure(v, 3, 0), 1.0, 1e-12);
}

TEST(NativeDistributedMesh, UnitTetVolumeIsOneSixth) {
  NativeMeshBuilder b(3);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 0.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 0.0, 1.0, 0, true);
  b.add_cell(CellType::Tet, {v0, v1, v2, v3}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  EXPECT_NEAR(view_r.value()->measure(3, 0), 1.0 / 6.0, 1e-15);
}

TEST(NativeDistributedMesh, TriangularPrismVolumeMatchesBaseTimesHeight) {
  // A right prism: base is the right triangle (0,0)-(1,0)-(0,1) (area 0.5),
  // extruded height 1, so its volume is 0.5.
  NativeMeshBuilder b(3);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 0.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 0.0, 1.0, 0, true);
  const int v4 = b.add_vertex(4, 1.0, 0.0, 1.0, 0, true);
  const int v5 = b.add_vertex(5, 0.0, 1.0, 1.0, 0, true);
  b.add_cell(CellType::Prism, {v0, v1, v2, v3, v4, v5}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  EXPECT_NEAR(view_r.value()->measure(3, 0), 0.5, 1e-13);
}

TEST(NativeDistributedMesh, SquarePyramidVolumeMatchesOneThirdBaseTimesHeight) {
  // Unit square base, apex at height 1 above its centre: V = (1/3)*1*1.
  NativeMeshBuilder b(3);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 1.0, 0.0, 0, true);
  const int v4 = b.add_vertex(4, 0.5, 0.5, 1.0, 0, true);
  b.add_cell(CellType::Pyramid, {v0, v1, v2, v3, v4}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  EXPECT_NEAR(view_r.value()->measure(3, 0), 1.0 / 3.0, 1e-13);
}

// ---- a genuine polyhedron: a unit hex expressed via face CSR ------------

TEST(NativeDistributedMesh, UnitHexExpressedAsPolyhedronHasVolumeOne) {
  NativeMeshBuilder b(3);
  int idx[8];
  const double x[8] = {0, 1, 1, 0, 0, 1, 1, 0};
  const double y[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  const double z[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i) idx[i] = b.add_vertex(i, x[i], y[i], z[i], 0, true);
  // The same six faces cell_topology.hpp's Hex table names, given here
  // explicitly (outward-CCW) as a Polyhedron's face-vertex CSR.
  std::vector<std::vector<int>> faces = {
      {idx[0], idx[3], idx[2], idx[1]}, {idx[4], idx[5], idx[6], idx[7]},
      {idx[0], idx[1], idx[5], idx[4]}, {idx[1], idx[2], idx[6], idx[5]},
      {idx[2], idx[3], idx[7], idx[6]}, {idx[3], idx[0], idx[4], idx[7]}};
  b.add_polyhedron_cell(faces, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();
  EXPECT_EQ(v.cell_type_of(0), CellType::Polyhedron);
  EXPECT_NEAR(v.measure(3, 0), 1.0, 1e-12);
}

TEST(NativeDistributedMesh, TwoPolyhedralCellsSharingAFaceDeduplicateIt) {
  // Two unit-hex polyhedra stacked along z, sharing the z=1 face, each
  // built independently from its OWN outward-CCW face list. The builder
  // must detect the shared face and store it once with an orientation flag
  // per side.
  NativeMeshBuilder b(3);
  int a[8];
  const double ax[8] = {0, 1, 1, 0, 0, 1, 1, 0};
  const double ay[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  const double az[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i) a[i] = b.add_vertex(i, ax[i], ay[i], az[i], 0, true);
  std::vector<std::vector<int>> faces_a = {
      {a[0], a[3], a[2], a[1]}, {a[4], a[5], a[6], a[7]}, {a[0], a[1], a[5], a[4]},
      {a[1], a[2], a[6], a[5]}, {a[2], a[3], a[7], a[6]}, {a[3], a[0], a[4], a[7]}};
  b.add_polyhedron_cell(faces_a, 0, 0, true);

  int t[4];
  t[0] = b.add_vertex(8, 0.0, 0.0, 2.0, 0, true);
  t[1] = b.add_vertex(9, 1.0, 0.0, 2.0, 0, true);
  t[2] = b.add_vertex(10, 1.0, 1.0, 2.0, 0, true);
  t[3] = b.add_vertex(11, 0.0, 1.0, 2.0, 0, true);
  // Cell B's bottom face names the SAME shared vertices (a[4..7]) but in
  // the opposite rotational sense from A's top face.
  std::vector<std::vector<int>> faces_b = {
      {a[4], a[5], a[6], a[7]}, {t[0], t[3], t[2], t[1]}, {a[4], t[0], t[1], a[5]},
      {a[5], t[1], t[2], a[6]}, {a[6], t[2], t[3], a[7]}, {a[7], t[3], t[0], a[4]}};
  b.add_polyhedron_cell(faces_b, 1, 0, true);

  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  // The shared face must be stored exactly once: 6 + 6 = 12 face slots
  // referencing at most 11 distinct faces (6 external to A + 5 new for B,
  // since B's bottom reuses A's top).
  EXPECT_EQ(arrays_r.value().face_count(), 11u);

  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();
  EXPECT_NEAR(v.measure(3, 0), 1.0, 1e-12);
  EXPECT_NEAR(v.measure(3, 1), 1.0, 1e-12);
}

// ---- rejections -------------------------------------------------------------

TEST(NativeDistributedMesh, RefusesAVertexIndexOutOfRange) {
  NativeMeshBuilder b(2);
  b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  b.add_vertex(2, 0.0, 1.0, 0.0, 0, true);
  b.add_cell(CellType::Triangle, {0, 1, 7}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_FALSE(arrays_r.ok());
  EXPECT_EQ(arrays_r.error().code, "E-MESH-CONNECTIVITY");
}

TEST(NativeDistributedMesh, RefusesACellWhoseDimensionDisagreesWithTheMeshs) {
  NativeMeshBuilder b(2);  // a 2D mesh
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 0.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 0.0, 1.0, 0, true);
  b.add_cell(CellType::Tet, {v0, v1, v2, v3}, 0, 0, true);  // a 3D cell
  auto arrays_r = b.build();
  ASSERT_FALSE(arrays_r.ok());
  EXPECT_EQ(arrays_r.error().code, "E-MESH-DIM");
}

TEST(NativeDistributedMesh, RefusesAFixedShapeGivenTheWrongCornerCount) {
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  b.add_cell(CellType::Quad, {v0, v1, v2}, 0, 0, true);  // Quad needs 4
  auto arrays_r = b.build();
  ASSERT_FALSE(arrays_r.ok());
  EXPECT_EQ(arrays_r.error().code, "E-MESH-CONNECTIVITY");
}

// ---------------------------------------------------------------------------
// NM-8: the three tests above exercise NativeMeshBuilder::build()'s OWN
// checks; every in-tree producer routes through it, so a direct caller of
// NativeMeshView::create_from_host (the class's own doc: "validates
// connectivity") was the only way to reach either gap below. Not latent
// via any producer in this tree, but the header's claim was false for a
// caller that bypasses the builder, and the unguarded reads this closes
// are both UB (an out-of-range dim fell through to no matching branch in
// the dimension dispatch; a wrong corner count read past a cell's own
// vertex slice).
// ---------------------------------------------------------------------------
TEST(NativeDistributedMesh, CreateFromHostRefusesAMeshWithDimensionOutsideZeroToThree) {
  NativeMeshBuilder b(5);  // no vertices or cells added: build()'s own
                           // per-cell dimension check never runs on an
                           // empty mesh, so dim=5 reaches create_from_host
                           // unchecked without this fix.
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok()) << arrays_r.error().message;
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_FALSE(view_r.ok());
  EXPECT_EQ(view_r.error().code, "E-MESH-DIM");
}

TEST(NativeDistributedMesh, CreateFromHostRefusesAFixedShapeGivenTheWrongCornerCount) {
  // A VALID triangle, built and validated by NativeMeshBuilder::build()
  // -- then its own CSR is hand-corrupted (shrinking the one cell's own
  // corner count from 3 to 2) to reach create_from_host with data the
  // builder itself would have refused, the same way a caller bypassing
  // the builder entirely could.
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  b.add_cell(CellType::Triangle, {v0, v1, v2}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok()) << arrays_r.error().message;
  auto& arrays = arrays_r.value();
  ASSERT_EQ(arrays.cell_vertex_offsets(1) - arrays.cell_vertex_offsets(0), 3);
  arrays.cell_vertex_offsets(1) = arrays.cell_vertex_offsets(0) + 2;  // now "2 corners"

  auto view_r = NativeMeshView::create_from_host(arrays, 0);
  ASSERT_FALSE(view_r.ok());
  EXPECT_EQ(view_r.error().code, "E-MESH-CONNECTIVITY");
}

// ---- vertices are never owned-measured, and owned/ghost accounting ------

TEST(NativeDistributedMesh, OwnedCountExcludesGhosts) {
  NativeMeshBuilder b(1);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 1, false);  // a ghost of rank 1
  const int v2 = b.add_vertex(2, 2.0, 0.0, 0.0, 0, true);
  b.add_cell(CellType::Segment, {v0, v1}, 0, 0, true);
  b.add_cell(CellType::Segment, {v1, v2}, 1, 1, false);  // a ghost cell here
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();
  EXPECT_EQ(v.entity_count(1), 2u);
  EXPECT_EQ(v.owned_count(1), 1u);
  EXPECT_TRUE(v.is_owned(1, 0));
  EXPECT_FALSE(v.is_owned(1, 1));
  EXPECT_EQ(v.owned_count(0), 2u);
}

// ---- device-resident geometry (docs/dev/native-mesh.md, "Execution space")

namespace {

/// Common body for the two tests below: every `device_measure()`/
/// `device_centroid()` entry agrees with the corresponding
/// `measure(dim,i)`/`centroid(dim,i)` call to 1e-14 relative -- the nvcc
/// FMA-contraction tolerance docs/dev/device-resident-arcs.md's
/// reproducibility contract already accepts, and bit-identical on a
/// host-only build where Kokkos::DefaultExecutionSpace IS the host space.
void ExpectDeviceGeometryMatchesHost(const NativeMeshView& v) {
  using DeviceMemSpace = NativeMeshView::DeviceMemSpace;

  // Compile-time: the primary arrays' element type AND memory space are
  // exactly Kokkos::DefaultExecutionSpace::memory_space, not merely
  // "some Kokkos view".
  static_assert(std::is_same_v<typename std::decay_t<decltype(v.device_measure())>::memory_space,
                              Kokkos::DefaultExecutionSpace::memory_space>);
  static_assert(std::is_same_v<typename std::decay_t<decltype(v.device_centroid())>::memory_space,
                              Kokkos::DefaultExecutionSpace::memory_space>);

  // Runtime: whether the host can dereference that memory space directly is
  // exactly "is this build's default execution space a real device" --
  // SpaceAccessibility, not the type alias, is what actually proves it.
  constexpr bool kHostCanTouchDeviceMemoryDirectly =
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, DeviceMemSpace>::accessible;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  EXPECT_FALSE(kHostCanTouchDeviceMemoryDirectly)
      << "a real device backend is compiled in, so the primary geometry arrays "
         "must live in memory the host cannot dereference directly";
#else
  EXPECT_TRUE(kHostCanTouchDeviceMemoryDirectly);
#endif

  ASSERT_EQ(v.device_measure().extent(0), v.cell_count());
  ASSERT_EQ(v.device_centroid().extent(0), v.cell_count());
  auto measure_mirror = Kokkos::create_mirror_view(v.device_measure());
  Kokkos::deep_copy(measure_mirror, v.device_measure());
  auto centroid_mirror = Kokkos::create_mirror_view(v.device_centroid());
  Kokkos::deep_copy(centroid_mirror, v.device_centroid());

  for (std::size_t c = 0; c < v.cell_count(); ++c) {
    SCOPED_TRACE(::testing::Message() << "cell " << c);
    const double host_measure = v.measure(v.dimension(), c);
    const double tol = std::max(1e-14, std::abs(host_measure) * 1e-14);
    EXPECT_NEAR(measure_mirror(c), host_measure, tol);

    const fuse::Point host_centroid = v.centroid(v.dimension(), c);
    EXPECT_NEAR(centroid_mirror(c, 0), host_centroid.x, 1e-13);
    EXPECT_NEAR(centroid_mirror(c, 1), host_centroid.y, 1e-13);
    EXPECT_NEAR(centroid_mirror(c, 2), host_centroid.z, 1e-13);
  }
}

}  // namespace

TEST(NativeDistributedMesh, DeviceGeometryMatchesHostForThreeDimensionalShapes) {
  // One mesh, five different 3D shapes (mixed shapes, one dimension, exactly
  // what "general cell" means here) -- a Hex, a Tet, a Prism, a Pyramid and
  // a Polyhedron twin of the Hex, so the device measure kernel's per-cell
  // simplex count varies cell to cell (6, 1, 3, 2 and 6 simplices).
  NativeMeshBuilder b(3);

  int hex[8];
  const double hx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
  const double hy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  const double hz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i) hex[i] = b.add_vertex(i, hx[i], hy[i], hz[i], 0, true);
  b.add_cell(CellType::Hex, {hex[0], hex[1], hex[2], hex[3], hex[4], hex[5], hex[6], hex[7]}, 0, 0,
            true);

  const int t0 = b.add_vertex(8, 2.0, 0.0, 0.0, 0, true);
  const int t1 = b.add_vertex(9, 3.0, 0.0, 0.0, 0, true);
  const int t2 = b.add_vertex(10, 2.0, 1.0, 0.0, 0, true);
  const int t3 = b.add_vertex(11, 2.0, 0.0, 1.0, 0, true);
  b.add_cell(CellType::Tet, {t0, t1, t2, t3}, 1, 0, true);

  const int p0 = b.add_vertex(12, 4.0, 0.0, 0.0, 0, true);
  const int p1 = b.add_vertex(13, 5.0, 0.0, 0.0, 0, true);
  const int p2 = b.add_vertex(14, 4.0, 1.0, 0.0, 0, true);
  const int p3 = b.add_vertex(15, 4.0, 0.0, 1.0, 0, true);
  const int p4 = b.add_vertex(16, 5.0, 0.0, 1.0, 0, true);
  const int p5 = b.add_vertex(17, 4.0, 1.0, 1.0, 0, true);
  b.add_cell(CellType::Prism, {p0, p1, p2, p3, p4, p5}, 2, 0, true);

  const int q0 = b.add_vertex(18, 6.0, 0.0, 0.0, 0, true);
  const int q1 = b.add_vertex(19, 7.0, 0.0, 0.0, 0, true);
  const int q2 = b.add_vertex(20, 7.0, 1.0, 0.0, 0, true);
  const int q3 = b.add_vertex(21, 6.0, 1.0, 0.0, 0, true);
  const int q4 = b.add_vertex(22, 6.5, 0.5, 1.0, 0, true);
  b.add_cell(CellType::Pyramid, {q0, q1, q2, q3, q4}, 3, 0, true);

  int poly[8];
  const double px[8] = {8, 9, 9, 8, 8, 9, 9, 8};
  const double py[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  const double pz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i) poly[i] = b.add_vertex(23 + i, px[i], py[i], pz[i], 0, true);
  std::vector<std::vector<int>> faces = {
      {poly[0], poly[3], poly[2], poly[1]}, {poly[4], poly[5], poly[6], poly[7]},
      {poly[0], poly[1], poly[5], poly[4]}, {poly[1], poly[2], poly[6], poly[5]},
      {poly[2], poly[3], poly[7], poly[6]}, {poly[3], poly[0], poly[4], poly[7]}};
  b.add_polyhedron_cell(faces, 4, 0, true);

  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  ExpectDeviceGeometryMatchesHost(*view_r.value());
}

TEST(NativeDistributedMesh, DeviceGeometryMatchesHostForTwoDimensionalShapes) {
  // A Quad (2 simplices) and a concave Polygon (4 simplices) in the SAME
  // mesh, so the device measure kernel's per-cell simplex count varies here
  // too, at dimension 2.
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 1.0, 0.0, 0, true);
  b.add_cell(CellType::Quad, {v0, v1, v2, v3}, 0, 0, true);

  const int w0 = b.add_vertex(4, 2.0, 0.0, 0.0, 0, true);
  const int w1 = b.add_vertex(5, 4.0, 0.0, 0.0, 0, true);
  const int w2 = b.add_vertex(6, 4.0, 1.0, 0.0, 0, true);
  const int w3 = b.add_vertex(7, 3.0, 1.0, 0.0, 0, true);
  const int w4 = b.add_vertex(8, 3.0, 2.0, 0.0, 0, true);
  const int w5 = b.add_vertex(9, 2.0, 2.0, 0.0, 0, true);
  b.add_cell(CellType::Polygon, {w0, w1, w2, w3, w4, w5}, 1, 0, true);

  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  ExpectDeviceGeometryMatchesHost(*view_r.value());
}

// ---------------------------------------------------------------------------
// NM-10: a face vertex set `find_or_add_face` cannot reconcile with the one
// it already recorded (same vertex SET, but an order that is neither a
// cyclic match nor its reversal) must be refused, not silently reused under
// whichever orientation the comparison happens to settle on.
// ---------------------------------------------------------------------------
TEST(NativeDistributedMesh, FaceWithSameVertexSetInANonCyclicOrderIsRefused) {
  // A square base {v0,v1,v2,v3} (any cyclic rotation, or its reversal, is
  // the same quad face); a second cell names the identical vertex SET for
  // one of its own faces but in the order {v0,v2,v1,v3} -- a transposition
  // of the middle two entries, neither a rotation of {v0,v1,v2,v3} nor of
  // its reversal {v3,v2,v1,v0}.
  NativeMeshBuilder b(3);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 1.0, 0.0, 0, true);
  const int v4 = b.add_vertex(4, 0.0, 0.0, 1.0, 0, true);

  b.add_polyhedron_cell({{v0, v1, v2, v3}, {v0, v1, v4}, {v1, v2, v4}, {v2, v3, v4}, {v3, v0, v4}},
                        0, 0, true);
  b.add_polyhedron_cell({{v0, v2, v1, v3}, {v0, v2, v4}, {v2, v1, v4}, {v1, v3, v4}, {v3, v0, v4}},
                        1, 0, true);

  auto arrays_r = b.build();
  ASSERT_FALSE(arrays_r.ok());
  EXPECT_EQ(arrays_r.error().code, "E-MESH-POLYGON-FACE-MISMATCH");
}

// ---------------------------------------------------------------------------
// Device-primary storage (docs/dev/native-mesh.md, "Execution space"):
// `device_arrays()` -- not any `std::vector` member -- is this view's
// primary storage, and the host arrays `measure`/`centroid`/`simplex_points`/
// `cell_vertices`/`vertex_x` serve are built, and cached, only the first
// time one of those accessors is actually called.
// ---------------------------------------------------------------------------

TEST(NativeDistributedMesh, DeviceArraysAreThePrimaryStorageAtCompileTime) {
  // Compile-time only: every field of `device_arrays()` lives in
  // NativeMeshView::DeviceMemSpace, not merely "some Kokkos view" -- the
  // same compile-time bar ExpectDeviceGeometryMatchesHost already holds
  // device_measure()/device_centroid() to, above.
  using DeviceMemSpace = NativeMeshView::DeviceMemSpace;
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 1.0, 0.0, 0, true);
  b.add_cell(CellType::Quad, {v0, v1, v2, v3}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();

  static_assert(
      std::is_same_v<typename std::decay_t<decltype(v.device_arrays().vertex_coords)>::memory_space,
                    DeviceMemSpace>);
  static_assert(std::is_same_v<
                typename std::decay_t<decltype(v.device_arrays().cell_vertex_indices)>::memory_space,
                DeviceMemSpace>);
  SUCCEED();
}

TEST(NativeDistributedMesh, NoHostMirrorExistsUntilAHostAccessorIsCalled) {
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 1.0, 0.0, 0, true);
  b.add_cell(CellType::Quad, {v0, v1, v2, v3}, 0, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();

  // Right after construction: validation already ran (it had to, to return
  // an `ok()` Result at all), but nothing has asked for a host-serving
  // array yet.
  EXPECT_EQ(v.host_mirrors_materialized(), 0);

  // The first host accessor call materializes the mirror...
  EXPECT_NEAR(v.measure(2, 0), 1.0, 1e-14);
  EXPECT_EQ(v.host_mirrors_materialized(), 1);

  // ...and every accessor after that (host or otherwise) is served from the
  // SAME cached mirror, not a fresh one.
  EXPECT_NEAR(v.centroid(2, 0).x, 0.5, 1e-14);
  EXPECT_EQ(v.vertex_x(0, 0), 0.0);
  EXPECT_EQ(v.host_mirrors_materialized(), 1);
}

TEST(NativeDistributedMesh, DeviceSimplexCsrMatchesTheValidatedHostDecomposition) {
  // A Hex (6 simplices after the divergence-theorem fan) and a Tet (its own
  // single simplex): device_simplex_offsets()/device_simplex_corners() must
  // describe exactly the same per-cell corner lists simplex_points() does,
  // once the host mirror is materialized.
  NativeMeshBuilder b(3);
  int hex[8];
  const double hx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
  const double hy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  const double hz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (int i = 0; i < 8; ++i) hex[i] = b.add_vertex(i, hx[i], hy[i], hz[i], 0, true);
  b.add_cell(CellType::Hex, {hex[0], hex[1], hex[2], hex[3], hex[4], hex[5], hex[6], hex[7]}, 0, 0,
            true);
  const int t0 = b.add_vertex(8, 2.0, 0.0, 0.0, 0, true);
  const int t1 = b.add_vertex(9, 3.0, 0.0, 0.0, 0, true);
  const int t2 = b.add_vertex(10, 2.0, 1.0, 0.0, 0, true);
  const int t3 = b.add_vertex(11, 2.0, 0.0, 1.0, 0, true);
  b.add_cell(CellType::Tet, {t0, t1, t2, t3}, 1, 0, true);

  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());
  auto view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(view_r.ok()) << view_r.error().message;
  const auto& v = *view_r.value();

  ASSERT_EQ(v.device_simplex_offsets().extent(0), v.cell_count() + 1);
  auto offsets_mirror = Kokkos::create_mirror_view(v.device_simplex_offsets());
  Kokkos::deep_copy(offsets_mirror, v.device_simplex_offsets());
  auto corners_mirror = Kokkos::create_mirror_view(v.device_simplex_corners());
  Kokkos::deep_copy(corners_mirror, v.device_simplex_corners());

  const int corners_per_simplex = v.dimension() + 1;
  for (std::size_t c = 0; c < v.cell_count(); ++c) {
    SCOPED_TRACE(::testing::Message() << "cell " << c);
    const int soff = offsets_mirror(c);
    const int scount = offsets_mirror(c + 1) - soff;
    ASSERT_EQ(static_cast<std::size_t>(scount), v.simplex_count(v.dimension(), c));
    std::vector<fuse::Point> host_corners(static_cast<std::size_t>(corners_per_simplex));
    for (int s = 0; s < scount; ++s) {
      v.simplex_points(v.dimension(), c, static_cast<std::size_t>(s), host_corners.data());
      const int row = soff + s;
      for (int k = 0; k < corners_per_simplex; ++k) {
        EXPECT_NEAR(corners_mirror(row, k, 0), host_corners[static_cast<std::size_t>(k)].x,
                   1e-13);
        EXPECT_NEAR(corners_mirror(row, k, 1), host_corners[static_cast<std::size_t>(k)].y,
                   1e-13);
        EXPECT_NEAR(corners_mirror(row, k, 2), host_corners[static_cast<std::size_t>(k)].z,
                   1e-13);
      }
    }
  }
}
