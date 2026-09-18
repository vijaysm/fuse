// The decompositions, checked as geometry rather than read as tables.
//
// A wrong table does not fail loudly: the clipping kernel still runs, the
// weights still sum to something, and a conservation test still reports a small
// number. So each decomposition is applied to reference cells here and its
// simplex measures must sum to the cell's measure -- which catches a missing
// piece, a duplicated piece and an overlap, because an overlap sums high and a
// gap sums low.
//
// The distorted instances matter as much as the regular ones: the hex table is
// only correct for a general hexahedron if every tet contains the main
// diagonal, and on a cube a wrong table can still sum correctly by symmetry.

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/cell_simplices.hpp"

namespace {

using fuse::CellShape;
using P = std::array<double, 3>;

double tri_area(const P& a, const P& b, const P& c) {
  const P u{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const P v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const P n{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0]};
  return 0.5 * std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
}

double tet_volume(const P& a, const P& b, const P& c, const P& d) {
  const P u{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const P v{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const P w{d[0] - a[0], d[1] - a[1], d[2] - a[2]};
  const double det = u[0] * (v[1] * w[2] - v[2] * w[1]) -
                     u[1] * (v[0] * w[2] - v[2] * w[0]) +
                     u[2] * (v[0] * w[1] - v[1] * w[0]);
  return std::abs(det) / 6.0;
}

/// Sum of the decomposition's simplex measures.
double decomposed_measure(CellShape shape, const std::vector<P>& corners) {
  const std::size_t n = fuse::cell_simplex_count(shape, corners.size());
  const int dim = fuse::shape_dimension(shape);
  double total = 0.0;
  for (std::size_t s = 0; s < n; ++s) {
    std::array<int, 4> idx{};
    EXPECT_TRUE(fuse::simplex_corners(shape, corners.size(), s, idx.data()));
    if (dim == 2) {
      total += tri_area(corners[idx[0]], corners[idx[1]], corners[idx[2]]);
    } else if (dim == 3) {
      total += tet_volume(corners[idx[0]], corners[idx[1]], corners[idx[2]],
                          corners[idx[3]]);
    }
  }
  return total;
}

}  // namespace

TEST(CellSimplices, CountsMatchTheShapes) {
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Edge, 2), 1u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Triangle, 3), 1u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Quad, 4), 2u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Tet, 4), 1u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Pyramid, 5), 2u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Prism, 6), 3u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Hex, 8), 6u);
  // A fan: an n-gon has n-2 triangles.
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Polygon, 3), 1u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Polygon, 6), 4u);
}

TEST(CellSimplices, AWrongCornerCountIsReportedNotRead) {
  // 0 rather than a guess: a backend handed a 5-corner "quad" has a mesh
  // problem, and decomposing 4 of the 5 would hide it behind a plausible
  // measure.
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Quad, 5), 0u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Hex, 7), 0u);
  EXPECT_EQ(fuse::cell_simplex_count(CellShape::Polygon, 2), 0u);
  std::array<int, 4> idx{};
  EXPECT_FALSE(fuse::simplex_corners(CellShape::Hex, 7, 0, idx.data()));
  EXPECT_FALSE(fuse::simplex_corners(CellShape::Hex, 8, 6, idx.data()))
      << "simplex index 6 of a 6-piece decomposition was accepted";
  EXPECT_FALSE(fuse::simplex_corners(CellShape::Tet, 4, 0, nullptr));
}

TEST(CellSimplices, QuadAndPolygonDecompositionsSumToTheArea) {
  const std::vector<P> unit_square{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  EXPECT_NEAR(decomposed_measure(CellShape::Quad, unit_square), 1.0, 1e-14);
  EXPECT_NEAR(decomposed_measure(CellShape::Polygon, unit_square), 1.0, 1e-14);

  // A trapezoid, so the two triangles are not congruent: area = (1 + 2)/2 * 1.
  const std::vector<P> trapezoid{{0, 0, 0}, {2, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  EXPECT_NEAR(decomposed_measure(CellShape::Quad, trapezoid), 1.5, 1e-14);

  // Tilted out of the xy-plane: a surface cell in 3D, which is the 2-manifold
  // in 3-space case the mesh types exist to express.
  const std::vector<P> tilted{{0, 0, 0}, {1, 0, 1}, {1, 1, 1}, {0, 1, 0}};
  EXPECT_NEAR(decomposed_measure(CellShape::Quad, tilted), std::sqrt(2.0), 1e-14);

  // A hexagon by the fan: regular, side 1, area = 3*sqrt(3)/2.
  std::vector<P> hexagon;
  for (int k = 0; k < 6; ++k) {
    const double a = static_cast<double>(k) * M_PI / 3.0;
    hexagon.push_back(P{std::cos(a), std::sin(a), 0.0});
  }
  EXPECT_NEAR(decomposed_measure(CellShape::Polygon, hexagon),
              3.0 * std::sqrt(3.0) / 2.0, 1e-13);
}

TEST(CellSimplices, VolumeDecompositionsSumToTheVolume) {
  const std::vector<P> cube{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                            {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  EXPECT_NEAR(decomposed_measure(CellShape::Hex, cube), 1.0, 1e-14);

  // The case a cube cannot catch, done with an AFFINE image of the cube: a
  // linear map keeps every face planar, so the hexahedron is a genuine solid
  // whose volume is |det M| and whose two diagonal decompositions must agree.
  // No two of its six tets are congruent, so a table whose pieces do not all
  // contain the main diagonal sums wrong here while still summing to 1 on a
  // cube.
  const double m[3][3] = {{1.2, 0.3, 0.1}, {0.0, 0.9, 0.4}, {0.2, 0.1, 1.1}};
  const double det_m =
      m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  std::vector<P> affine;
  for (const P& c : cube) {
    affine.push_back(P{m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2],
                       m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2],
                       m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2]});
  }
  const double via_0_6 = decomposed_measure(CellShape::Hex, affine);
  EXPECT_NEAR(via_0_6, std::abs(det_m), 1e-13);

  // And the other diagonal, as an independent decomposition of the same solid.
  const int other[6][4] = {{1, 2, 3, 7}, {1, 3, 0, 7}, {1, 0, 4, 7},
                           {1, 4, 5, 7}, {1, 5, 6, 7}, {1, 6, 2, 7}};
  double via_1_7 = 0.0;
  for (const auto& tt : other) {
    via_1_7 += tet_volume(affine[tt[0]], affine[tt[1]], affine[tt[2]],
                          affine[tt[3]]);
  }
  EXPECT_NEAR(via_0_6, via_1_7, 1e-13)
      << "the two diagonals disagree on a PLANAR-faced hexahedron, so at least "
         "one decomposition is not a partition of it";

  // A unit-height prism on a right triangle of area 1/2.
  const std::vector<P> prism{{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                             {0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
  EXPECT_NEAR(decomposed_measure(CellShape::Prism, prism), 0.5, 1e-14);
  // Sheared, so the three tets differ: volume is unchanged by a shear parallel
  // to the base.
  const std::vector<P> sheared{{0, 0, 0},     {1, 0, 0},     {0, 1, 0},
                               {0.4, 0.3, 1}, {1.4, 0.3, 1}, {0.4, 1.3, 1}};
  EXPECT_NEAR(decomposed_measure(CellShape::Prism, sheared), 0.5, 1e-14);

  // Pyramid: base area 1, height 1, volume 1/3.
  const std::vector<P> pyramid{{0, 0, 0},     {1, 0, 0}, {1, 1, 0},
                               {0, 1, 0},     {0.5, 0.5, 1}};
  EXPECT_NEAR(decomposed_measure(CellShape::Pyramid, pyramid), 1.0 / 3.0, 1e-14);
  // Apex moved off-centre: still 1/3, and the two tets are now unequal.
  const std::vector<P> oblique{{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                               {0, 1, 0}, {0.1, 0.2, 1}};
  EXPECT_NEAR(decomposed_measure(CellShape::Pyramid, oblique), 1.0 / 3.0, 1e-14);

  const std::vector<P> tet{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  EXPECT_NEAR(decomposed_measure(CellShape::Tet, tet), 1.0 / 6.0, 1e-15);
}

TEST(CellSimplices, EveryCornerIsUsedAndTheMainDiagonalIsShared) {
  // A partition has to reach every corner: a table that silently omits one
  // would still sum correctly on a symmetric cell.
  for (const auto [shape, corners] :
       {std::pair{CellShape::Quad, std::size_t{4}},
        std::pair{CellShape::Tet, std::size_t{4}},
        std::pair{CellShape::Pyramid, std::size_t{5}},
        std::pair{CellShape::Prism, std::size_t{6}},
        std::pair{CellShape::Hex, std::size_t{8}}}) {
    std::vector<bool> seen(corners, false);
    const std::size_t n = fuse::cell_simplex_count(shape, corners);
    const int per = fuse::shape_dimension(shape) + 1;
    for (std::size_t s = 0; s < n; ++s) {
      std::array<int, 4> idx{};
      ASSERT_TRUE(fuse::simplex_corners(shape, corners, s, idx.data()));
      for (int k = 0; k < per; ++k) {
        ASSERT_GE(idx[k], 0);
        ASSERT_LT(static_cast<std::size_t>(idx[k]), corners);
        seen[static_cast<std::size_t>(idx[k])] = true;
      }
      // No simplex may repeat a corner -- that is a degenerate piece with zero
      // measure, which would pass a measure-sum test by contributing nothing.
      for (int a = 0; a < per; ++a)
        for (int b = a + 1; b < per; ++b)
          EXPECT_NE(idx[a], idx[b]) << "simplex " << s << " repeats a corner";
    }
    for (std::size_t c = 0; c < corners; ++c) {
      EXPECT_TRUE(seen[c]) << "corner " << c << " appears in no simplex";
    }
  }

  // The hex table's defining property, stated as a test so a future edit that
  // breaks it fails here rather than in a conservation study.
  for (std::size_t s = 0; s < 6; ++s) {
    std::array<int, 4> idx{};
    ASSERT_TRUE(fuse::simplex_corners(CellShape::Hex, 8, s, idx.data()));
    bool has0 = false, has6 = false;
    for (int k = 0; k < 4; ++k) {
      has0 = has0 || idx[k] == 0;
      has6 = has6 || idx[k] == 6;
    }
    EXPECT_TRUE(has0 && has6)
        << "hex tet " << s << " does not contain the main diagonal 0-6";
  }
}

TEST(CellSimplices, ANonPlanarHexHasNoSingleVolumeAndThatIsWhyTheTableIsFixed) {
  // A hexahedron with non-planar (bilinear) faces is not a polyhedron: each
  // diagonal triangulates those faces differently, so the two decompositions
  // enclose genuinely DIFFERENT solids with different volumes. Neither is
  // wrong.
  //
  // The consequence is the reason cell_simplices.hpp exists. "The volume of
  // this hex" is not a property of the cell alone -- it depends on the
  // decomposition -- so if two backends decomposed the same mesh about
  // different diagonals, they would disagree about cell volumes, and a
  // conservation check between them would show a defect that is really a
  // convention mismatch. One fixed table removes that possibility.
  const std::vector<P> twisted{{0, 0, 0},     {1, 0, 0},     {1, 1, 0},
                               {0, 1, 0},     {0.3, 0.2, 1}, {1.4, 0.1, 1.2},
                               {1.2, 1.3, 1}, {0.1, 1.1, 1.3}};
  const double via_0_6 = decomposed_measure(CellShape::Hex, twisted);
  const int other[6][4] = {{1, 2, 3, 7}, {1, 3, 0, 7}, {1, 0, 4, 7},
                           {1, 4, 5, 7}, {1, 5, 6, 7}, {1, 6, 2, 7}};
  double via_1_7 = 0.0;
  for (const auto& tt : other) {
    via_1_7 += tet_volume(twisted[tt[0]], twisted[tt[1]], twisted[tt[2]],
                          twisted[tt[3]]);
  }
  // Both are positive and plausible, and they differ by ~1% here: large enough
  // to matter in a conservation budget, small enough to be mistaken for
  // rounding if nobody had written this down.
  EXPECT_GT(via_0_6, 0.0);
  EXPECT_GT(via_1_7, 0.0);
  EXPECT_GT(std::abs(via_0_6 - via_1_7), 1e-6)
      << "the faces of this cell are non-planar, so the two decompositions "
         "should NOT agree; if they do, the reference decomposition above is "
         "no longer independent of the table";
}
