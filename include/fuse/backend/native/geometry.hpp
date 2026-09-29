#pragma once

/// The native mesh's simplex-decomposition geometry, as pure functions over
/// small local arrays -- no Kokkos view, no mesh, so this is testable on its
/// own and reusable for both a 2D polygon cell and one face of a 3D cell.
///
/// THE SHARED-FACE RULE (docs/dev/native-mesh.md). `MeshView`'s contract
/// (mesh_view.hpp) requires a simplex decomposition whose measures sum
/// exactly to the cell's, but says nothing about consistency BETWEEN cells.
/// The native mesh adds that requirement itself, because it is the first
/// backend whose cells share explicit face structure: a face shared by two
/// polyhedra (or two hexes, two prisms, ...) must decompose IDENTICALLY
/// whichever cell asks, or the two cells' tet decompositions disagree along
/// their shared boundary and a conservation sum over both develops a gap or
/// a double-counted sliver there.
///
/// THE RULE CHOSEN: fan the face from its MINIMUM-GLOBAL-ID corner. Two
/// cells sharing a face may store its corner loop starting at a different
/// corner and walking in the opposite rotational sense (one cell's outward
/// is the other's inward), but they store the SAME set of corners with the
/// SAME cyclic adjacency. Rotating a fan's start to a fixed corner is
/// obviously direction-independent-adjacent; the non-obvious fact this
/// relies on is that the fan's OWN triangle set does not depend on which of
/// the two cyclic directions is walked away from that corner either --
/// walking forward from v0 over v1..v_{n-1} gives triangles (v0,v1,v2),
/// (v0,v2,v3), ..., and walking backward gives the same n-2 triangles as
/// vertex SETS, because the diagonal pattern v0-v_i for i = 2..n-2 is
/// symmetric under i -> n-i. So "fan from the min-global-id corner" needs no
/// agreement on direction at all -- only on the corner set, which two
/// records of the same face always share.
///
/// VALIDITY. A fan is only a correct decomposition when it is a fan from a
/// point the polygon is star-shaped about. `canonical_fan_triangulation`
/// checks this the same way for every shape, including non-planar ones
/// embedded in 3D: Newell's formula gives the polygon's signed vector area
/// as an algebraic IDENTITY with the signed sum of the fan's own triangles,
/// for ANY apex, star-shaped or not (it is the same sum regrouped). So a fan
/// that folds back on itself at a reflex corner produces a signed cancellation
/// the UNSIGNED sum (what `simplex_points`/`measure` must report -- see
/// mesh_view.hpp) does not have: `unsigned_sum > |signed_total|` exactly when
/// some triangle's orientation disagrees with the whole. Comparing the two
/// is therefore a correct, general star-shapedness test, not a convexity
/// heuristic -- it is exact for a planar polygon and well defined (though
/// only a "vector area", not a differential-geometric area) for a
/// non-planar one such as a cubed-sphere quad.
///
/// The 3D cell version of the same idea: fan every face (by the rule above)
/// from the CELL's centroid, and cross-check the unsigned tetrahedra sum
/// against the divergence-theorem volume computed from the SAME oriented
/// face triangles from a fixed origin -- again an identity for any origin,
/// so the same "unsigned > |signed|" test catches a cell that is not
/// star-shaped about its own centroid.

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "fuse/base/result.hpp"

namespace fuse::native {

struct Vec3 {
  double x = 0.0, y = 0.0, z = 0.0;
};

[[nodiscard]] inline Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
[[nodiscard]] inline Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
[[nodiscard]] inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
[[nodiscard]] inline double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
[[nodiscard]] inline double norm(const Vec3& a) {
  return std::sqrt(dot(a, a));
}

/// Newell's vector area of an n-gon's corner loop, in the order given.
/// Magnitude is direction-independent; the vector itself flips sign under
/// reversal. Well defined for a non-planar loop too (a bilinear-patch
/// "vector area", not a true differential area), which is what lets the
/// star-shapedness check below apply to a cubed-sphere quad as well as a
/// planar one.
[[nodiscard]] Vec3 newell_vector_area(const std::vector<Vec3>& pts);

/// One triangle of a decomposition, as local indices (0..n-1) into the
/// caller's own point/id arrays -- never as coordinates, so the caller
/// decides whether it wants points, global ids, or both.
using Triangle = std::array<int, 3>;

/// The canonical fan triangulation of an n-gon (n >= 3), keyed by
/// `global_ids` so that two cells presenting the same face (same corner
/// set, same cyclic adjacency, any starting corner, either rotational
/// sense) produce IDENTICAL triangles. `points` and `global_ids` must have
/// the same length and be listed in the polygon's cyclic order.
///
/// Rejects `E-MESH-POLYGON-NOT-STAR-SHAPED` when the fan is not a valid
/// decomposition of this polygon (the unsigned triangle-area sum exceeds
/// the polygon's own Newell magnitude by more than a relative tolerance) --
/// per the file header, this is an exact test, not a heuristic. Rejects
/// `E-MESH-POLYGON-DEGENERATE` for `n < 3` or duplicate global ids.
[[nodiscard]] Result<std::vector<Triangle>> canonical_fan_triangulation(
    const std::vector<std::int64_t>& global_ids, const std::vector<Vec3>& points);

/// One oriented face of a 3D cell, ready for the volume/tet decomposition
/// below: its corners' points and global ids, in the cell's own outward-CCW
/// storage order (cell_topology.hpp's convention).
struct OrientedFace {
  std::vector<Vec3> points;
  std::vector<std::int64_t> global_ids;
};

/// One tetrahedron of a cell's decomposition, as points (never orientation
/// -- mesh_view.hpp's simplex contract does not require one).
struct Tetrahedron {
  Vec3 a, b, c, d;
};

/// Decomposes a 3D cell given as a closed set of outward-oriented faces
/// (`cell_topology.hpp`'s convention: read from outside, a face's corners
/// run CCW) into tetrahedra, fanning each face's canonical triangulation
/// (see above) to `centroid`.
///
/// Validates with the divergence-theorem identity from the file header:
/// the oriented face triangles give a true enclosed volume from ANY origin,
/// so comparing that (using the coordinate origin) against the UNSIGNED sum
/// of the centroid-fanned tetrahedra's volumes is an exact star-shapedness
/// test. Rejects `E-MESH-CELL-NOT-STAR-SHAPED` when they disagree by more
/// than a relative tolerance, and propagates a `E-MESH-POLYGON-*` rejection
/// from a face's own triangulation.
[[nodiscard]] Result<std::vector<Tetrahedron>> canonical_cell_tetrahedra(
    const std::vector<OrientedFace>& faces, const Vec3& centroid);

/// Unsigned volume of a tetrahedron: |scalar triple product| / 6.
[[nodiscard]] double tetrahedron_volume(const Tetrahedron& t);

/// Unsigned area of a triangle: |cross product| / 2.
[[nodiscard]] double triangle_area(const Vec3& a, const Vec3& b, const Vec3& c);

}  // namespace fuse::native
