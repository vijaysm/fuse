#pragma once

// PRIVATE header. Lives under src/, is not installed, and is included by
// exactly two translation units: src/remap/intx_weights.cpp (the generic layer)
// and src/remap/tempest_weights.cpp (the one backend that exists today).
//
// PURPOSE. Task 5.3 added a second way to build remap weights. The seam between
// "resolve the plan, find the meshes, classify the geometry, audit the result"
// and "run a particular numerical method" is drawn here, and it is drawn in
// SOMA's own vocabulary:
//
//   * the mesh crosses as a POLYGON SOUP -- coordinates and connectivity --
//     not as a moab::Interface, a Tempest `Mesh`, or a file;
//   * the geometry crosses as a CLASSIFICATION (`SurfaceKind`), not as an
//     assumption;
//   * the request crosses as ORDER and MONOTONE, which are the two things
//     `RemapPlan` already says.
//
// Nothing here names TempestRemap, a cubed sphere, `ncol`, or a great-circle
// arc. The only backend implemented today happens to be a spherical remapper
// and refuses `SurfaceKind::Plane` and volumes; a planar or volumetric
// high-order backend would implement this same function and the generic layer
// would not change. The sphere-specific parts -- normalising to unit radius,
// orienting faces outward from the origin -- live in the backend, on its own
// copy of the geometry, because they are that backend's requirements.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/remap/remap_plan.hpp"

namespace soma::remap_backend {

/// How the generic layer classified the two surfaces. `Other` covers "not
/// classified", including every mesh of dimension other than 2.
enum class SurfaceKind { Other, Sphere, Plane };

/// A surface as a soup of convex polygons, in the units the mesh was loaded in.
///
/// `node_xyz` is `3 * node_count` doubles. `face_start` is `face_count() + 1`
/// entries, so face `f` uses `face_node[face_start[f] .. face_start[f+1])`, each
/// an index into `node_xyz / 3`. Faces are in the caller's local index order --
/// row `j` and column `i` of the returned matrix mean face `j` of the target
/// soup and face `i` of the source soup, and the caller has arranged for those
/// to be `MeshView` local indices.
///
/// No winding convention is imposed here: a backend that needs one applies it
/// to its own copy. That is deliberate. "Counter-clockwise seen from outside"
/// is only meaningful once you have decided what "outside" is.
struct PolygonSurface {
  std::vector<double>       node_xyz;
  std::vector<std::int32_t> face_start;
  std::vector<std::int32_t> face_node;
  std::string               label;  ///< for error messages

  [[nodiscard]] std::size_t node_count() const noexcept {
    return node_xyz.size() / 3;
  }
  [[nodiscard]] std::size_t face_count() const noexcept {
    return face_start.empty() ? 0 : face_start.size() - 1;
  }
};

/// What the plan asked for, reduced to the two things a numerical method cares
/// about, plus what the generic layer worked out about the geometry.
struct HighOrderRequest {
  /// 1 = piecewise constant, 2 = exact for linears, and so on.
  int order = 1;
  /// Bound preservation is REQUIRED, not preferred. A backend that cannot
  /// deliver it at `order` must refuse, not silently drop the order.
  bool monotone = false;

  SurfaceKind kind = SurfaceKind::Other;
  /// Common radius of both surfaces, when `kind == Sphere`. In the meshes' own
  /// length units.
  double radius = 1.0;
};

/// The matrix, plus everything needed to audit it without re-deriving the
/// geometry. All areas are in the BACKEND's own measure, in the meshes' units
/// (so `radius^2` times whatever it computed on a unit sphere, if it worked
/// there); the generic layer keeps them separate from `MeshView::measure` on
/// purpose, because the two disagree at O(h^2) on a curved surface and
/// pretending otherwise produces a conservation defect nobody can explain.
struct HighOrderResult {
  std::vector<std::int32_t> row, col;  ///< row = target face, col = source face
  std::vector<double>       val;
  std::size_t               n_rows = 0, n_cols = 0;

  std::vector<double> src_area, tgt_area;        ///< parent areas
  std::vector<double> src_covered, tgt_covered;  ///< overlap area attributed

  /// The order the backend actually delivered. A backend must set this to what
  /// it built, and must refuse rather than set it below `request.order`.
  int  order = 1;
  bool monotone = false;

  /// Whatever the backend printed while working. Kept because the only backend
  /// there is reports failure by printing and returning success.
  std::string log;
};

/// True when this build has a working high-order backend. Compiled
/// unconditionally: the stub returns false.
[[nodiscard]] bool high_order_backend_available() noexcept;

/// A short name for messages, e.g. "TempestRemap". Never null.
[[nodiscard]] const char* high_order_backend_name() noexcept;

/// Build the weights. Errors, never throws.
///
/// Declared unconditionally and defined unconditionally, so the generic layer
/// links either way; without the backend it returns `E-REMAP-NOTEMPEST`.
[[nodiscard]] Result<HighOrderResult> build_high_order_weights(
    const HighOrderRequest& request, const PolygonSurface& src,
    const PolygonSurface& tgt);

}  // namespace soma::remap_backend
