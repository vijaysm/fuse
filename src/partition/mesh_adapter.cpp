#include "fuse/partition/mesh_adapter.hpp"

#include <fmt/format.h>

#include "fuse/base/config.hpp"
#include "fuse/partition/detail/collectives.hpp"

namespace fuse::partition {

Result<CentroidData> centroids_of(const MeshView& mesh, CommHandle comm) {
#if !FUSE_HAVE_MPI
  if (comm != kNullComm) {
    return Failure{"E-PART-NO-MPI",
                  "a non-null communicator was supplied but this build has no MPI "
                  "(FUSE_HAVE_MPI=0); rebuild with -DFUSE_ENABLE_MPI=ON"};
  }
#endif

  // Agree `mesh.dimension()` across every rank BEFORE reading a single
  // entity -- max and min both, so disagreement is detectable (not just
  // which value "won"). comm == kNullComm agrees with itself trivially.
  const std::int64_t local_dim = mesh.dimension();
  std::int64_t max_dim = 0, min_dim = 0;
  detail::allreduce_max_i64(&local_dim, &max_dim, 1, comm);
  detail::allreduce_min_i64(&local_dim, &min_dim, 1, comm);

  if (max_dim != min_dim) {
    return Failure{"E-PART-MIXED-DIM",
                  fmt::format("ranks disagree about the mesh's top dimension "
                              "(dimension() ranges from {} to {} across ranks); "
                              "partitioning one rank's answer would silently drop "
                              "the cells implied by another's",
                              min_dim, max_dim)};
  }
  if (max_dim <= 0) {
    return Failure{"E-PART-NO-ENTITIES",
                  fmt::format("mesh.dimension() == {}; no rank has a partitionable mesh loaded",
                              max_dim)};
  }

  const int dim = static_cast<int>(max_dim);
  const std::size_t n_local = mesh.entity_count(dim);
  std::size_t n_owned = 0;
  for (std::size_t i = 0; i < n_local; ++i) {
    if (mesh.is_owned(dim, i)) ++n_owned;
  }

  CentroidData out;
  out.dimension_used = dim;
  out.centroids =
      Kokkos::View<double* [3], Kokkos::HostSpace>("fuse::partition::adapter_centroids", n_owned);
  out.global_ids =
      Kokkos::View<std::int64_t*, Kokkos::HostSpace>("fuse::partition::adapter_gids", n_owned);

  std::size_t k = 0;
  for (std::size_t i = 0; i < n_local; ++i) {
    if (!mesh.is_owned(dim, i)) continue;
    const Point p = mesh.centroid(dim, i);
    out.centroids(k, 0) = p.x;
    out.centroids(k, 1) = p.y;
    out.centroids(k, 2) = p.z;
    out.global_ids(k) = static_cast<std::int64_t>(mesh.global_id(dim, i));
    ++k;
  }
  return out;
}

}  // namespace fuse::partition
