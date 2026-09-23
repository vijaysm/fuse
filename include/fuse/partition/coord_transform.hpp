#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>

#include "fuse/partition/types.hpp"

/// Pre-bisection coordinate transforms. Ported from rcb_moab's
/// CoordTransform.{hpp,cpp}. Identity is the default, correct for planar
/// 2D/3D meshes: RCB bisects the centroids as given. Gnomonic is offered,
/// never assumed, for a 2D manifold embedded in 3D that is (locally)
/// sphere-like -- a cubed-sphere or lat/lon mesh, or any mesh close enough
/// to a sphere that RCB on raw (x, y, z) would cut curved "banana slice"
/// partitions across the surface. It projects each centroid onto one of six
/// tangent-plane faces of an inscribed cube first, so RCB then sees genuine
/// 2D planar coordinates within each face.
///
/// PORTED, NOT THE FULL rcb_moab OPTION SET. rcb_moab's coarse-grid path,
/// pole-cap handling, post-pass refinement, and face-aware per-face RCB
/// dispatch (the skew-triggered three-way decision in
/// RCBPartitioner::partition()) are NOT here -- see docs/dev/partitioning.md
/// "What was ported" for the deferral list and why. This header carries only
/// the transform itself: apply it, or don't.
namespace fuse::partition {

enum class CoordTransform { Identity, Gnomonic };

struct GnomonicOptions {
  /// Sphere center in the input mesh's coordinate frame. Defaults to the
  /// origin, as rcb_moab's does.
  double center[3] = {0.0, 0.0, 0.0};

  /// Encoding distance between adjacent cube faces in the projected third
  /// coordinate, so RCB's longest-extent axis picker chooses the face axis
  /// before splitting within a face. Must exceed the projected [-1, 1]
  /// xi/eta range; default 10.0, as rcb_moab's is.
  double face_separation = 10.0;
};

namespace detail {

/// Classify + gnomonically project one point, faithfully reproducing
/// rcb_moab/CoordTransform.cpp::apply_gnomonic_projection's per-point
/// arithmetic (face convention: 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z). A
/// KOKKOS_INLINE_FUNCTION so the same code compiles for host and device.
///
/// Points at the sphere center are degenerate (undefined direction) and
/// project to (0, 0, 0), matching the source's handling.
KOKKOS_INLINE_FUNCTION
void gnomonic_project_point(double x, double y, double z, double cx, double cy, double cz,
                            double face_separation, double& xi_out, double& eta_out,
                            double& z_out) {
  double dx = x - cx;
  double dy = y - cy;
  double dz = z - cz;
  const double r = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
  if (r > 0.0) {
    const double inv = 1.0 / r;
    dx *= inv;
    dy *= inv;
    dz *= inv;
  } else {
    xi_out = 0.0;
    eta_out = 0.0;
    z_out = 0.0;
    return;
  }

  const double ax = dx < 0.0 ? -dx : dx;
  const double ay = dy < 0.0 ? -dy : dy;
  const double az = dz < 0.0 ? -dz : dz;

  int face = 0;
  double xi = 0.0, eta = 0.0;
  if (ax >= ay && ax >= az) {
    face = (dx >= 0.0) ? 0 : 1;
    xi = dy / ax;
    eta = dz / ax;
  } else if (ay >= az) {
    face = (dy >= 0.0) ? 2 : 3;
    xi = dx / ay;
    eta = dz / ay;
  } else {
    face = (dz >= 0.0) ? 4 : 5;
    xi = dx / az;
    eta = dy / az;
  }

  xi_out = xi;
  eta_out = eta;
  z_out = static_cast<double>(face) * face_separation;
}

}  // namespace detail

/// Applies `transform` to `centroids`. Identity returns the SAME view (a
/// shallow alias, no allocation, no kernel launch). Gnomonic allocates a NEW
/// view and fills it via one labelled Kokkos::parallel_for; the caller's
/// `centroids` is never mutated -- unlike rcb_moab's in-place
/// `apply_gnomonic_projection`, which is safe there only because
/// RCBPartitioner::partition() owns a private centroids buffer it can
/// afford to overwrite.
template <class ExecutionSpace>
Kokkos::View<const double* [3], typename ExecutionSpace::memory_space> apply_coord_transform(
    Kokkos::View<const double* [3], typename ExecutionSpace::memory_space> centroids,
    CoordTransform transform, const GnomonicOptions& opts = {}) {
  using MemorySpace = typename ExecutionSpace::memory_space;
  if (transform == CoordTransform::Identity) return centroids;

  const std::size_t n = centroids.extent(0);
  Kokkos::View<double* [3], MemorySpace> projected("fuse::partition::gnomonic_projected", n);
  const double cx = opts.center[0];
  const double cy = opts.center[1];
  const double cz = opts.center[2];
  const double sep = opts.face_separation;
  Kokkos::parallel_for(
      "fuse::partition::apply_gnomonic_projection", Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        double xi = 0.0, eta = 0.0, zc = 0.0;
        detail::gnomonic_project_point(centroids(i, 0), centroids(i, 1), centroids(i, 2), cx, cy,
                                       cz, sep, xi, eta, zc);
        projected(i, 0) = xi;
        projected(i, 1) = eta;
        projected(i, 2) = zc;
      });
  return projected;
}

}  // namespace fuse::partition
