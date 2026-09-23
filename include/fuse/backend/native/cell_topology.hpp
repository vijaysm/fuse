#pragma once

/// Cell types for the native distributed unstructured mesh, and the fixed
/// per-shape face tables that let every 3D shape (Tet, Pyramid, Prism, Hex)
/// share ONE decomposition code path with the general Polyhedron.
///
/// WHY A SEPARATE ENUM FROM `fuse::CellShape` (cell_simplices.hpp). That
/// table already gives LOCAL-corner-index simplex decompositions for Edge,
/// Triangle, Quad, Polygon, Tet, Pyramid, Prism and Hex -- but "local corner
/// index" is exactly what the native mesh's geometry contract forbids for a
/// face shared between two cells (mesh_view.hpp's simplex contract; see
/// docs/dev/native-mesh.md's "shared-face rule"): two neighbouring cells
/// number their shared face's corners differently, so a decomposition keyed
/// on local index 0 picks a DIFFERENT diagonal on each side of the same
/// physical face. `CellType` adds `Point` and `Polyhedron` (neither of which
/// `CellShape` represents) and `kFaceTable` below names each shape's faces as
/// LOCAL CORNER GROUPS -- which native_mesh.hpp's face triangulator then
/// re-keys by the corners' GLOBAL ids before splitting, so two cells sharing
/// a face always split it identically. `CellShape`'s tables are unaffected
/// and still serve backends (MOAB) that do not need cross-cell consistency.
///
/// CONVENTION. Face corner lists are in the MOAB/Exodus rotational sense
/// (cell_simplices.hpp's own convention): read from OUTSIDE the cell, a
/// face's corners run counter-clockwise. native_mesh.hpp's volume check
/// depends on every table entry honouring this -- get one face backwards and
/// the divergence-theorem cross-check (docs/dev/native-mesh.md) reports the
/// wrong sign and the cell is refused as not star-shaped, which is how a
/// wrong table entry would be caught, not silently accepted.

#include <cstddef>

#include <Kokkos_Macros.hpp>

namespace fuse::native {

/// The cell shapes the native mesh stores. `Polygon` is 2D with n >= 3
/// corners; `Polyhedron` is 3D with an explicit face list (cell-to-face CSR)
/// rather than a fixed corner count. Every other shape implies its corner
/// count via `corner_count` below.
enum class CellType {
  Point,      ///< 0D, 1 corner
  Segment,    ///< 1D, 2 corners
  Triangle,   ///< 2D, 3 corners
  Quad,       ///< 2D, 4 corners
  Polygon,    ///< 2D, n >= 3 corners, general
  Tet,        ///< 3D, 4 corners
  Pyramid,    ///< 3D, 5 corners: quad base 0-3, apex 4
  Prism,      ///< 3D, 6 corners: triangle 0-2, opposite triangle 3-5
  Hex,        ///< 3D, 8 corners: face 0-3, opposite face 4-7
  Polyhedron, ///< 3D, explicit cell-to-face-to-vertex CSR
};

/// The manifold (topological) dimension of a cell type. A mesh holds cells
/// of exactly one dimension -- "mixed types" means mixed SHAPES at that one
/// dimension (tri + quad in 2D; hex + tet + prism + pyramid + polyhedron in
/// 3D), never mixed dimension in one mesh.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr int cell_dimension(CellType t) noexcept {
  switch (t) {
    case CellType::Point: return 0;
    case CellType::Segment: return 1;
    case CellType::Triangle:
    case CellType::Quad:
    case CellType::Polygon: return 2;
    case CellType::Tet:
    case CellType::Pyramid:
    case CellType::Prism:
    case CellType::Hex:
    case CellType::Polyhedron: return 3;
  }
  return -1;
}

/// Fixed corner count for every shape except `Polygon` and `Polyhedron`
/// (both variable; 0 here means "ask the mesh's CSR instead").
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr int fixed_corner_count(CellType t) noexcept {
  switch (t) {
    case CellType::Point: return 1;
    case CellType::Segment: return 2;
    case CellType::Triangle: return 3;
    case CellType::Quad: return 4;
    case CellType::Polygon: return 0;
    case CellType::Tet: return 4;
    case CellType::Pyramid: return 5;
    case CellType::Prism: return 6;
    case CellType::Hex: return 8;
    case CellType::Polyhedron: return 0;
  }
  return 0;
}

/// How many faces a FIXED 3D shape has (never called for `Polyhedron`, whose
/// face count comes from the mesh's own CSR).
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr int fixed_face_count(CellType t) noexcept {
  switch (t) {
    case CellType::Tet: return 4;
    case CellType::Pyramid: return 5;
    case CellType::Prism: return 5;
    case CellType::Hex: return 6;
    default: return 0;
  }
}

/// How many corners face `f` of a fixed 3D shape has.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr int fixed_face_corner_count(
    CellType t, int f) noexcept {
  switch (t) {
    case CellType::Tet: return 3;  // four triangular faces
    case CellType::Pyramid: return (f == 0) ? 4 : 3;  // quad base, 4 triangles
    case CellType::Prism: return (f < 2) ? 3 : 4;     // 2 triangles, 3 quads
    case CellType::Hex: return 4;                     // six quad faces
    default: return 0;
  }
}

/// Writes face `f`'s local corner indices (within the cell's own corner
/// list) into `out`, counter-clockwise as seen from OUTSIDE the cell (see
/// the file header). `out` must have room for `fixed_face_corner_count(t,
/// f)` entries. Returns false for an out-of-range shape or face index,
/// having written nothing.
[[nodiscard]] KOKKOS_INLINE_FUNCTION constexpr bool fixed_face_corners(
    CellType t, int f, int* out) noexcept {
  switch (t) {
    case CellType::Tet: {
      // Corners 0,1,2,3. Faces opposite each corner, outward-CCW.
      constexpr int table[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};
      if (f < 0 || f >= 4) return false;
      for (int k = 0; k < 3; ++k) out[k] = table[f][k];
      return true;
    }
    case CellType::Pyramid: {
      // Quad base 0-1-2-3 (outward normal points AWAY from apex 4), then
      // four triangular sides.
      if (f < 0 || f >= 5) return false;
      if (f == 0) {
        constexpr int base[4] = {0, 3, 2, 1};
        for (int k = 0; k < 4; ++k) out[k] = base[k];
        return true;
      }
      constexpr int side[4][3] = {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
      for (int k = 0; k < 3; ++k) out[k] = side[f - 1][k];
      return true;
    }
    case CellType::Prism: {
      // Triangle 0-1-2, opposite triangle 3-4-5 (0->3, 1->4, 2->5), then
      // three quad sides.
      if (f < 0 || f >= 5) return false;
      if (f == 0) { out[0] = 0; out[1] = 2; out[2] = 1; return true; }
      if (f == 1) { out[0] = 3; out[1] = 4; out[2] = 5; return true; }
      constexpr int side[3][4] = {{0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
      for (int k = 0; k < 4; ++k) out[k] = side[f - 2][k];
      return true;
    }
    case CellType::Hex: {
      // Face 0-1-2-3, opposite face 4-5-6-7 (0->4, 1->5, 2->6, 3->7), then
      // four side quads. This is the MOAB/Exodus hex corner convention
      // (cell_simplices.hpp's comment: "face 0-3, opposite face 4-7").
      if (f < 0 || f >= 6) return false;
      constexpr int table[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
                                   {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}};
      for (int k = 0; k < 4; ++k) out[k] = table[f][k];
      return true;
    }
    default:
      return false;
  }
}

}  // namespace fuse::native
