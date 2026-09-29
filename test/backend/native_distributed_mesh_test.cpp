// The native distributed mesh's geometry contract: every cell type's
// simplex decomposition, the shared-face rule, and the rejections that fire
// when a cell cannot be decomposed. Closed-form expected values throughout,
// so a failure points at the geometry code, not a tolerance.

#include "fuse/backend/native_mesh.hpp"

#include "fuse/backend/native/geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>

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
