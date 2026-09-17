// The native simplicial mesh: the two dimensions, the codimension rules, and
// the measures. Every expected value here is a closed-form area or volume, so
// a failure points at the geometry code rather than at a tolerance.

#include "fuse/backend/native/mesh.hpp"

#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

namespace {

using fuse::native::Mesh;

/// A single simplex from explicit coordinates, with every simplex its own
/// parent cell (the already-simplicial case).
template <int M, int S>
Mesh<M, S> one_simplex(const std::vector<std::array<double, S>>& pts) {
  Mesh<M, S> m("t", pts.size(), 1, 1);
  for (std::size_t i = 0; i < pts.size(); ++i) {
    for (int k = 0; k < S; ++k) m.points()(i, k) = pts[i][k];
  }
  for (int v = 0; v < M + 1; ++v) m.cells()(0, v) = v;
  m.parents()(0) = 0;
  return m;
}

}  // namespace

// ---- the two dimensions and the codimension ----------------------------

TEST(NativeMesh, CodimensionFollowsFromTheTwoDimensions) {
  // The table in the header, asserted rather than described.
  static_assert(Mesh<0, 3>::kCodimension == 3, "point cloud in 3D");
  static_assert(Mesh<1, 3>::kCodimension == 2, "curve in 3D");
  static_assert(Mesh<2, 2>::kCodimension == 0, "planar triangulation");
  static_assert(Mesh<2, 3>::kCodimension == 1, "surface in 3D");
  static_assert(Mesh<3, 3>::kCodimension == 0, "volume mesh");

  // A cell of manifold dimension M is a simplex with M+1 corners.
  static_assert(Mesh<2, 3>::kVertsPerCell == 3, "a triangle has three");
  static_assert(Mesh<3, 3>::kVertsPerCell == 4, "a tetrahedron has four");

  // Exactly the codimension-1 meshes have one normal per cell.
  static_assert(Mesh<2, 3>::kHasUniqueNormal, "a surface in 3D does");
  static_assert(Mesh<1, 2>::kHasUniqueNormal, "an edge in the plane does");
  static_assert(!Mesh<3, 3>::kHasUniqueNormal, "a volume fills space");
  static_assert(!Mesh<2, 2>::kHasUniqueNormal, "so does a planar mesh");
  static_assert(!Mesh<1, 3>::kHasUniqueNormal, "a curve in 3D has infinitely many");
  SUCCEED();
}

// A normal on a mesh that has none is a COMPILE error, not a refusal. This
// documents the intent; uncommenting either line must fail to build.
//
//   Mesh<3,3> v; double n[3]; v.normal(0, n);   // no unique normal
//   Mesh<1,3> c; double n[3]; c.normal(0, n);   // infinitely many
//
// The static_asserts above are what a test can check; that these do not
// compile is checked by the compile-fail leg (docs/dev/testing.md).

// ---- measures, against closed forms ------------------------------------

TEST(NativeMesh, MeasureIsTheManifoldMeasureNotTheSpatialOne) {
  // A 3-4-5 right triangle lying in the plane z = 7 in three-space. Its AREA
  // is 6. A naive determinant of the 3D edge matrix would give zero, which is
  // the mistake the Gram determinant exists to avoid.
  auto tri = one_simplex<2, 3>({{{0.0, 0.0, 7.0}}, {{3.0, 0.0, 7.0}},
                                {{0.0, 4.0, 7.0}}});
  ASSERT_TRUE(tri.finalize().ok());
  EXPECT_NEAR(tri.measure(0), 6.0, 1e-14);

  // ... and its centroid is the mean of the corners, including in z.
  EXPECT_NEAR(tri.centroid(0, 0), 1.0, 1e-14);
  EXPECT_NEAR(tri.centroid(0, 1), 4.0 / 3.0, 1e-14);
  EXPECT_NEAR(tri.centroid(0, 2), 7.0, 1e-14);
}

TEST(NativeMesh, TetrahedronVolumeAndTriangleArea) {
  // The corner tetrahedron of the unit cube: volume 1/6.
  auto tet = one_simplex<3, 3>({{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}},
                                {{0.0, 1.0, 0.0}}, {{0.0, 0.0, 1.0}}});
  ASSERT_TRUE(tet.finalize().ok());
  EXPECT_NEAR(tet.measure(0), 1.0 / 6.0, 1e-15);

  // A planar triangle, codimension 0.
  auto tri2 = one_simplex<2, 2>({{{0.0, 0.0}}, {{2.0, 0.0}}, {{0.0, 5.0}}});
  ASSERT_TRUE(tri2.finalize().ok());
  EXPECT_NEAR(tri2.measure(0), 5.0, 1e-15);

  // A curve segment in 3D: its measure is arc length, 3-4-5 again.
  auto seg = one_simplex<1, 3>({{{1.0, 1.0, 1.0}}, {{4.0, 5.0, 1.0}}});
  ASSERT_TRUE(seg.finalize().ok());
  EXPECT_NEAR(seg.measure(0), 5.0, 1e-15);
}

TEST(NativeMesh, ADegenerateSimplexHasZeroMeasureAndNotANaN) {
  // Three collinear points whose Gram determinant rounds NEGATIVE, so the
  // clamp before the square root is actually exercised. The obvious choice
  // -- (0,0,0), (1,1,1), (2,2,2) -- cancels exactly (3*12 - 6*6 = 0) and
  // therefore proves nothing; the mutation gate caught that. Here
  // det = -8.9e-16, and without the clamp the measure is a NaN that would
  // propagate silently through a conservation sum.
  auto flat = one_simplex<2, 3>({{{0.0, 0.0, 0.0}}, {{0.7, 0.2, 0.1}},
                                 {{2.1, 0.6, 0.3}}});
  ASSERT_TRUE(flat.finalize().ok());
  EXPECT_EQ(flat.measure(0), 0.0);
  EXPECT_FALSE(std::isnan(flat.measure(0)));
}

TEST(NativeMesh, APointCloudHasNoExtent) {
  Mesh<0, 3> pc("pc", 2, 2, 2);
  for (std::size_t i = 0; i < 2; ++i) {
    pc.points()(i, 0) = static_cast<double>(i);
    pc.points()(i, 1) = 0.0;
    pc.points()(i, 2) = 0.0;
    pc.cells()(i, 0) = static_cast<int>(i);
    pc.parents()(i) = i;
  }
  ASSERT_TRUE(pc.finalize().ok());
  // Zero because a point has no extent, which is not a degeneracy.
  EXPECT_EQ(pc.measure(0), 0.0);
  EXPECT_NEAR(pc.centroid(1, 0), 1.0, 1e-15);
}

// ---- the surface normal -------------------------------------------------

TEST(NativeMesh, ASurfaceNormalIsUnitAndPerpendicular) {
  auto tri = one_simplex<2, 3>({{{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}},
                                {{0.0, 1.0, 0.0}}});
  ASSERT_TRUE(tri.finalize().ok());
  double n[3];
  tri.normal(0, n);
  EXPECT_NEAR(std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]), 1.0, 1e-15);
  EXPECT_NEAR(std::abs(n[2]), 1.0, 1e-15) << "the z = 0 plane's normal is +/- z";

  // An edge in the plane: the normal is perpendicular to the edge direction.
  auto edge = one_simplex<1, 2>({{{0.0, 0.0}}, {{2.0, 0.0}}});
  ASSERT_TRUE(edge.finalize().ok());
  double n2[2];
  edge.normal(0, n2);
  EXPECT_NEAR(n2[0] * 2.0 + n2[1] * 0.0, 0.0, 1e-15) << "not perpendicular";
  EXPECT_NEAR(std::sqrt(n2[0] * n2[0] + n2[1] * n2[1]), 1.0, 1e-15);
}

// ---- connectivity and the parent map -----------------------------------

TEST(NativeMesh, RefusesAVertexIndexOutOfRange) {
  Mesh<2, 2> m("m", 3, 1, 1);
  for (std::size_t i = 0; i < 3; ++i) { m.points()(i, 0) = 0.0; m.points()(i, 1) = 0.0; }
  m.cells()(0, 0) = 0;
  m.cells()(0, 1) = 1;
  m.cells()(0, 2) = 7;      // there is no vertex 7
  m.parents()(0) = 0;
  const auto r = m.finalize();
  ASSERT_FALSE(r.ok()) << "an out-of-range corner would read arbitrary memory "
                          "and produce a measure that looks like a number";
  EXPECT_EQ(r.error().code, "E-MESH-CONNECTIVITY");
}

TEST(NativeMesh, RefusesASimplexWhoseParentDoesNotExist) {
  // The parent map is what conservation is accounted over, so a simplex
  // pointing outside it must fail here rather than be dropped from a sum.
  auto m = one_simplex<2, 2>({{{0.0, 0.0}}, {{1.0, 0.0}}, {{0.0, 1.0}}});
  m.parents()(0) = 4;
  const auto r = m.finalize();
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MESH-PARENT");
}

TEST(NativeMesh, ParentCellCountIsWhatConservationBalancesOver) {
  // One hexahedron's worth: six tetrahedra, one parent cell. A conservation
  // sum must balance over 1 cell, not 6 simplices -- confusing the two is the
  // defect the parent map exists to prevent.
  Mesh<3, 3> hex("hex", 8, 6, 1);
  const double c[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                          {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  for (int i = 0; i < 8; ++i) {
    for (int k = 0; k < 3; ++k) hex.points()(i, k) = c[i][k];
  }
  // Kuhn decomposition, the same six paths the structured backend emits.
  const int tets[6][4] = {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6},
                          {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
  for (int t = 0; t < 6; ++t) {
    for (int v = 0; v < 4; ++v) hex.cells()(t, v) = tets[t][v];
    hex.parents()(t) = 0;
  }
  ASSERT_TRUE(hex.finalize().ok());
  EXPECT_EQ(hex.simplex_count(), 6u);
  EXPECT_EQ(hex.parent_cell_count(), 1u);

  double total = 0.0;
  for (std::size_t t = 0; t < hex.simplex_count(); ++t) total += hex.measure(t);
  EXPECT_NEAR(total, 1.0, 1e-14) << "the six tetrahedra must fill the cube";
}
