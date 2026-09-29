#pragma once

/// How a cell of any supported shape is cut into simplices, as INDEX TABLES.
///
/// WHY TABLES AND NOT GEOMETRY. `MeshView` requires every backend to present
/// its cells as simplices, because the clipping kernel then has one geometric
/// case instead of six. Each backend was free to do that decomposition itself,
/// and the structured backend does -- from grid spacing, without a vertex list.
/// A backend over arbitrary cells (MOAB) cannot work that way: it has a corner
/// list and a shape, and needs to know which corners form which simplex. That
/// mapping is pure combinatorics, identical for every such backend, and wrong
/// in the same way for all of them if each writes its own.
///
/// So this header holds only *which corner indices* form each simplex. It
/// touches no coordinates, needs no mesh, and is therefore testable on its own
/// -- which is how the decompositions below are verified: each is applied to a
/// reference cell and the simplex measures must sum to the cell's measure with
/// no overlap.
///
/// PORTABILITY. Every function is `constexpr` and marked for device use, so a
/// Kokkos kernel can decompose a cell in place rather than calling back to the
/// host for the corner list. That is the reason the tables are values rather
/// than `std::vector`s, and the reason nothing here allocates.
///
/// The corner orderings are the ones MOAB and Exodus use (a 3D cell's first
/// face first, then the opposite face in the same rotational sense), because
/// that is what a backend reading a mesh file actually has. A backend with a
/// different convention must permute before calling, not reinterpret.

#include <cstddef>

#include <Kokkos_Macros.hpp>

namespace fuse {

/// The shapes a cell can have. `Polygon` is the only variable-corner shape; the
/// rest imply their corner count, and a mismatch is a caller error the
/// functions below report by returning 0 rather than by reading past the list.
enum class CellShape {
  Edge,      ///< 2 corners, 1-manifold
  Triangle,  ///< 3
  Quad,      ///< 4
  Polygon,   ///< n >= 3, planar, convex or star-shaped about corner 0
  Tet,       ///< 4
  Pyramid,   ///< 5: quad base 0-3, apex 4
  Prism,     ///< 6: triangle 0-2, opposite triangle 3-5
  Hex,       ///< 8: face 0-3, opposite face 4-7
};

/// The manifold dimension a shape spans: 1 for an edge, 2 for the surface
/// shapes, 3 for the volume ones. A simplex of that dimension has `dim + 1`
/// corners, which is how much room `simplex_corners` needs.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr int shape_dimension(CellShape s) noexcept {
  switch (s) {
    case CellShape::Edge: return 1;
    case CellShape::Triangle:
    case CellShape::Quad:
    case CellShape::Polygon: return 2;
    case CellShape::Tet:
    case CellShape::Pyramid:
    case CellShape::Prism:
    case CellShape::Hex: return 3;
  }
  return 0;
}

/// How many simplices the decomposition produces. 0 means the shape and corner
/// count disagree, which the caller must treat as a rejection -- a silently empty
/// decomposition would look like a legitimately empty cell.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr std::size_t cell_simplex_count(
    CellShape s, std::size_t corners) noexcept {
  switch (s) {
    case CellShape::Edge:     return corners == 2 ? 1 : 0;
    case CellShape::Triangle: return corners == 3 ? 1 : 0;
    case CellShape::Quad:     return corners == 4 ? 2 : 0;
    // A fan about corner 0. Valid for a convex polygon, and for a non-convex
    // one that is star-shaped about corner 0; anything else needs a real
    // triangulation, which is why `Polygon` is documented as that and not as
    // "any polygon".
    case CellShape::Polygon:  return corners >= 3 ? corners - 2 : 0;
    case CellShape::Tet:      return corners == 4 ? 1 : 0;
    case CellShape::Pyramid:  return corners == 5 ? 2 : 0;
    case CellShape::Prism:    return corners == 6 ? 3 : 0;
    case CellShape::Hex:      return corners == 8 ? 6 : 0;
  }
  return 0;
}

/// Writes the corner indices of simplex `s` into `out` (`shape_dimension + 1`
/// entries). Returns false when the shape, corner count or simplex index is out
/// of range, having written nothing.
///
/// The 3D decompositions all cut about a single corner or diagonal, so the
/// pieces share that corner and cover the cell exactly once. They are NOT
/// arbitrary: each is the standard decomposition for its shape, and the
/// measure-sum test in `test/backend/cell_simplices_test.cpp` is what says so.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool simplex_corners(
    CellShape shape, std::size_t corners, std::size_t s, int* out) noexcept {
  if (out == nullptr || s >= cell_simplex_count(shape, corners)) return false;
  switch (shape) {
    case CellShape::Edge:
      out[0] = 0; out[1] = 1;
      return true;
    case CellShape::Triangle:
      out[0] = 0; out[1] = 1; out[2] = 2;
      return true;
    case CellShape::Quad:
      // The 0-2 diagonal. Either diagonal covers the quad; this one is chosen
      // consistently so two backends decomposing the same quad agree, which
      // matters when one mesh is the supermesh oracle for the other.
      out[0] = 0;
      out[1] = static_cast<int>(s + 1);
      out[2] = static_cast<int>(s + 2);
      return true;
    case CellShape::Polygon:
      out[0] = 0;
      out[1] = static_cast<int>(s + 1);
      out[2] = static_cast<int>(s + 2);
      return true;
    case CellShape::Tet:
      out[0] = 0; out[1] = 1; out[2] = 2; out[3] = 3;
      return true;
    case CellShape::Pyramid: {
      // Base 0-1-2-3 split on its 0-2 diagonal, apex 4 on both.
      constexpr int table[2][4] = {{0, 1, 2, 4}, {0, 2, 3, 4}};
      for (int k = 0; k < 4; ++k) out[k] = table[s][k];
      return true;
    }
    case CellShape::Prism: {
      // Three tets about the 0-5 diagonal of the wedge.
      constexpr int table[3][4] = {{0, 1, 2, 5}, {0, 1, 5, 4}, {0, 4, 5, 3}};
      for (int k = 0; k < 4; ++k) out[k] = table[s][k];
      return true;
    }
    case CellShape::Hex: {
      // Six tets about the main diagonal 0-6. Every tet contains both 0 and 6,
      // and the remaining pair walks the six faces not touching that diagonal,
      // so the six are disjoint and fill the hex for ANY hexahedron, not only
      // for a cube -- the property the measure-sum test checks on a distorted
      // one as well as on a unit cube.
      constexpr int table[6][4] = {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6},
                                   {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
      for (int k = 0; k < 4; ++k) out[k] = table[s][k];
      return true;
    }
  }
  return false;
}

}  // namespace fuse
