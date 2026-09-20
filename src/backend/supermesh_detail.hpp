#pragma once

// PRIVATE to src/backend/: not under include/fuse/, because nothing outside
// this translation unit and its whitebox test needs it. Exists so
// test/backend/supermesh_test.cpp can call `inward_plane` directly with a
// manufactured degenerate face (REVIEW.md R-12/BK-M2's "Test to add", which
// the fix pass's own report flagged as impractical through the public
// MeshView grammar alone -- SerialMeshBackend has no explicit point/cell
// source form to build a degenerate cell with). Reached the same way
// test/parallel/comm_graph_test.cpp reaches the transport layer's private
// CrystalRouter: a `target_include_directories(... PRIVATE
// ${CMAKE_SOURCE_DIR}/src)` added for exactly one test target, not a
// widening of supermesh.hpp's public surface.

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/base/result.hpp"

namespace fuse::supermesh_detail {

/// REVIEW.md R-13 (BK-M3): the per-cell coverage audit's refusal, factored
/// out so it is unit-testable on a synthetic ratio directly -- the defect
/// class it guards against (overlap inflated past the cell it came from,
/// R-12's family) cannot be produced through the public MeshView grammar by
/// a well-formed, non-self-intersecting mesh, which is why the original fix
/// pass could only add a mutation (`supermesh-no-over-coverage-audit`) and
/// not a permanent GoogleTest case. Testing the boundary condition and the
/// error shape directly closes that gap without needing to manufacture
/// invalid geometry.
constexpr double kSupermeshCoverageTol = 1e-9;

[[nodiscard]] inline Result<void> audit_coverage(double covered,
                                                  double measure,
                                                  const char* which,
                                                  std::size_t idx, int dim) {
  if (!(measure > 0.0)) return Result<void>::success();
  const double c = covered / measure;
  if (c > 1.0 + kSupermeshCoverageTol) {
    return Failure{
        "E-REMAP-INTX",
        "the supermesh assigns " + std::to_string(c) + " times its own area "
            "to " + std::string(which) + " cell " + std::to_string(idx) +
            " (dimension " + std::to_string(dim) +
            "); overlaps cannot exceed the cell, so the intersection or its "
            "parent attribution is wrong"};
  }
  return Result<void>::success();
}

}  // namespace fuse::supermesh_detail

namespace fuse::supermesh_detail {

namespace detail {
[[nodiscard]] inline double coord(const Point& p, int k) {
  return k == 0 ? p.x : (k == 1 ? p.y : p.z);
}
}  // namespace detail

// --- the geometric kernel --------------------------------------------------
//
// r3d and r2d clip a polytope against half-spaces, keeping the side where
// `dot(n, x) + d >= 0` (verified in their clip loops: a vertex with a negative
// signed distance is the one discarded). So the planes of the CLIPPING simplex
// must point INWARD.
//
// The planes are constructed here rather than with r3d_tet_faces_from_verts,
// and the inward direction is established by TESTING against the opposite
// vertex instead of relying on a vertex-ordering convention. The contract on
// MeshView deliberately does not require an orientation, so there is no
// ordering to rely on -- and an outward-pointing plane would not error, it
// would silently return the complement's measure.

/// Face `i` of a simplex omits vertex `i`. Fills the inward plane and
/// returns whether the face was non-degenerate.
///
/// REVIEW.md R-12: a zero-length edge (2D) or zero-area face (3D) has no
/// normal to give -- `n` comes out `(0,0,0)`, and with it `d == 0`
/// regardless of the vertex used to compute it (`d = -(0 . v[a]) == 0`).
/// r2d/r3d's clip keeps the side where `dot(n, x) + d >= 0`, so an
/// unreported `(0,0,0), 0` plane reads `0 >= 0` -- true for EVERY point,
/// i.e. no constraint at all, which is the opposite of what a degenerate
/// clipping face should impose (zero measure, not the whole target).
/// Returning `false` here lets the caller clip to zero instead, the same
/// way the target polygon's own `twice_area == 0.0` check already does.
template <int Dim>
[[nodiscard]] inline bool inward_plane(const std::array<Point, 4>& v, int i,
                                       double n[3], double& d) {
  using detail::coord;
  const int a = (i + 1) % (Dim + 1);
  const int b = (i + 2) % (Dim + 1);
  if constexpr (Dim == 2) {
    // The face is the edge a-b; a normal is its in-plane perpendicular.
    const double tx = v[b].x - v[a].x;
    const double ty = v[b].y - v[a].y;
    n[0] = -ty;
    n[1] = tx;
    n[2] = 0.0;
  } else {
    const int c = (i + 3) % 4;
    const double e1[3] = {v[b].x - v[a].x, v[b].y - v[a].y, v[b].z - v[a].z};
    const double e2[3] = {v[c].x - v[a].x, v[c].y - v[a].y, v[c].z - v[a].z};
    n[0] = e1[1] * e2[2] - e1[2] * e2[1];
    n[1] = e1[2] * e2[0] - e1[0] * e2[2];
    n[2] = e1[0] * e2[1] - e1[1] * e2[0];
  }
  const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (!(len > 0.0)) return false;
  d = -(n[0] * coord(v[a], 0) + n[1] * coord(v[a], 1) + n[2] * coord(v[a], 2));
  // Make it point at the omitted vertex, which is the one inside.
  const double side = n[0] * coord(v[i], 0) + n[1] * coord(v[i], 1) +
                      n[2] * coord(v[i], 2) + d;
  if (side < 0.0) {
    for (int k = 0; k < 3; ++k) n[k] = -n[k];
    d = -d;
  }
  for (int k = 0; k < 3; ++k) n[k] /= len;
  d /= len;
  return true;
}

}  // namespace fuse::supermesh_detail
