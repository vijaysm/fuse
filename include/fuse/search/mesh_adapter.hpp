#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include "fuse/base/result.hpp"
#include "fuse/search/geometry.hpp"
#include "fuse/search/types.hpp"

/// Builds `fuse::search::Box`es from a mesh WITHOUT depending on
/// `fuse::backend` -- required by this layer's placement (types.hpp's header
/// comment: fuse::search sits BELOW fuse::backend so backend can depend on
/// it, the reverse of fuse::partition's own mesh adapter). `fuse::MeshView`
/// (include/fuse/backend/mesh_view.hpp) lives one layer ABOVE this one, so
/// `#include "fuse/backend/..."` here would be exactly the upward dependency
/// `tools/check-layering.py` forbids.
///
/// The resolution: `element_boxes` below is a template, constrained by the
/// `MeshLike` concept over the STRUCTURAL SHAPE `fuse::MeshView` exposes
/// (`dimension()`, `entity_count()`, `is_owned()`, `simplex_count()`,
/// `simplex_points()`, `centroid()` -- mesh_view.hpp documents each), never
/// naming its concrete type. A caller that already has a concrete
/// `fuse::MeshView&` in scope (fuse::backend itself, fuse::runtime, or a
/// test that links fuse::backend) instantiates this template there; ordinary
/// C++ duck typing makes it work against the real class with no cast, no new
/// virtual dispatch, and no include of mesh_view.hpp from this layer. This is
/// the same shape of resolution `partition::centroids_of` uses for the
/// opposite direction (a function that DOES depend on backend, isolated in
/// its own file so the rest of that layer does not) -- here the dependency
/// is avoided altogether instead, because avoiding it is what the task's
/// placement requires.
namespace fuse::search {

/// `M` need not be `fuse::MeshView` -- it only has to shape-match it. The
/// point type `centroid`/`simplex_points` use is deduced (see
/// `element_boxes`'s `PointT`), not named here, so this concept places no
/// constraint on it beyond having `.x`, `.y`, `.z` (checked at the actual use
/// site by ordinary member access, which is what "duck typed" means).
template <class M>
concept MeshLike = requires(const M& m, int dim, std::size_t i, std::size_t s) {
  { m.dimension() } -> std::convertible_to<int>;
  { m.entity_count(dim) } -> std::convertible_to<std::size_t>;
  { m.is_owned(dim, i) } -> std::convertible_to<bool>;
  { m.simplex_count(dim, i) } -> std::convertible_to<std::size_t>;
  { m.centroid(dim, i) };
  m.simplex_points(dim, i, s, static_cast<std::decay_t<decltype(m.centroid(dim, i))>*>(nullptr));
};

/// One bounding box per OWNED entity of `mesh.dimension()` (mesh_view.hpp:
/// "the top topological dimension"), each grown by `tol`
/// (`geometry.hpp::padded`) so a point on a face shared between two elements
/// is found inside both of their boxes -- this IS the tolerance application
/// point element_boxes commits to; `BoxIndex::build` itself applies none.
///
/// Built from `mesh`'s SIMPLEX decomposition (`simplex_count`/
/// `simplex_points`), not `centroid`/`measure` alone, so a non-convex cell's
/// box is the true bound of its geometry.
///
/// `out_owned_ids[k]` is the LOCAL entity index (`mesh`'s own `i`, meaningful
/// only to `mesh`) `boxes()[k]` was built from -- a caller that needs to map
/// a `BoxIndex` hit back to a mesh entity (rather than a position in the
/// returned vector) reads it from here. Ghost entities are skipped
/// (`is_owned`): an index built for interpolation should not offer a ghost's
/// box, whose values are a copy of another rank's.
///
/// Errors, never throws:
///   * `E-SEARCH-DIM` -- `mesh.dimension()` is not 1, 2 or 3.
///   * `E-SEARCH-NO-GEOMETRY` -- some owned entity's `simplex_count` is 0.
///     mesh_view.hpp: "legal ... the weight builder turns into a refusal
///     naming this backend rather than a silently empty supermesh"; the same
///     posture here, not a guess at what that entity's box should be.
template <MeshLike M>
[[nodiscard]] Result<std::vector<Box>> element_boxes(const M& mesh,
                                                     std::vector<std::size_t>& out_owned_ids,
                                                     const Tolerance& tol = {}) {
  using PointT = std::decay_t<decltype(mesh.centroid(0, std::size_t{0}))>;

  const int dim = mesh.dimension();
  if (dim < 1 || dim > 3) {
    return Failure{"E-SEARCH-DIM", fmt::format("mesh.dimension() == {} is not 1, 2, or 3", dim)};
  }

  out_owned_ids.clear();
  std::vector<Box> boxes;
  const std::size_t n = mesh.entity_count(dim);
  std::vector<PointT> corners;
  for (std::size_t i = 0; i < n; ++i) {
    if (!mesh.is_owned(dim, i)) continue;

    const std::size_t nsimplex = mesh.simplex_count(dim, i);
    if (nsimplex == 0) {
      return Failure{"E-SEARCH-NO-GEOMETRY",
                     fmt::format("entity {} of dimension {} has no simplex decomposition; this "
                                 "mesh view carries no geometry (fuse/backend/mesh_view.hpp)",
                                 i, dim)};
    }

    corners.assign(static_cast<std::size_t>(dim) + 1, PointT{});
    Point lo{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
    Point hi{std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(),
            std::numeric_limits<double>::lowest()};
    for (std::size_t s = 0; s < nsimplex; ++s) {
      mesh.simplex_points(dim, i, s, corners.data());
      for (const PointT& c : corners) {
        lo.x = std::min(lo.x, static_cast<double>(c.x));
        lo.y = std::min(lo.y, static_cast<double>(c.y));
        lo.z = std::min(lo.z, static_cast<double>(c.z));
        hi.x = std::max(hi.x, static_cast<double>(c.x));
        hi.y = std::max(hi.y, static_cast<double>(c.y));
        hi.z = std::max(hi.z, static_cast<double>(c.z));
      }
    }
    boxes.push_back(padded(Box{lo, hi}, tol));
    out_owned_ids.push_back(i);
  }
  return boxes;
}

}  // namespace fuse::search
