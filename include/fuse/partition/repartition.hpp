#pragma once

/// `fuse::partition::repartition` -- the convenience the native-mesh track
/// asked for: compute each owned cell's centroid, hand them to
/// `rcb_partition`, and `fuse::native::migrate` the mesh onto the result.
/// Header-only (like `rcb.hpp` itself): the only non-template work is
/// `centroids_of` and `fuse::native::migrate`, both already compiled
/// elsewhere in this layer/the one below it.
///
/// This is the ONE file in `fuse::partition` that closes the loop back
/// down into `fuse::backend`'s native mesh for migration, same direction
/// `mesh_adapter.hpp` already depends on it for centroids -- no new
/// layering edge.

#include <memory>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/backend/native_migration.hpp"
#include "fuse/partition/mesh_adapter.hpp"
#include "fuse/partition/rcb.hpp"

namespace fuse::partition {

struct RepartitionOptions {
  /// Forwarded to `rcb_partition` and to `centroids_of`/`migrate`'s
  /// collectives (all three MUST share one comm; `comm` here is the only
  /// copy, `rcb.comm` is overwritten with it in `repartition` so a caller
  /// cannot accidentally pass two different communicators to the two
  /// halves of one call).
  RcbOptions rcb;

  /// See types.hpp. kNullComm (default): Tier 0, one participant.
  CommHandle comm = kNullComm;
};

/// Computes centroids, partitions with `rcb_partition<ExecutionSpace>`
/// (uniform per-cell weight -- a caller wanting load-imbalance-aware
/// weights should call `rcb_partition`/`migrate` directly instead of this
/// convenience wrapper), and migrates `mesh` onto the result.
///
/// CONSTRAINT: `options.rcb.num_parts`, if set explicitly rather than left
/// at its default (0, meaning "the communicator's size"), must equal the
/// communicator's size -- `migrate` refuses `E-MESH-MIGRATE-RANGE` the
/// moment a part id lands outside `[0, num_ranks)`, which is exactly what
/// a mismatched `num_parts` produces. `rcb_partition` and `migrate` have
/// no other coupling: this is the one thing this wrapper's default
/// (leaving `num_parts` at 0) exists to avoid a caller getting wrong.
///
/// Errors: anything `centroids_of` or `migrate` themselves refuse
/// (`E-PART-MIXED-DIM`, `E-PART-NO-ENTITIES`, `E-MESH-MIGRATE-INCOMPLETE`
/// cannot occur here since `dest` is always sized to the owned count,
/// `E-MESH-MIGRATE-RANGE` from the constraint above).
template <class ExecutionSpace = Kokkos::DefaultHostExecutionSpace>
[[nodiscard]] Result<std::unique_ptr<native::NativeMeshView>> repartition(
    const native::NativeMeshView& mesh, RepartitionOptions options) {
  using MemorySpace = typename ExecutionSpace::memory_space;
  options.rcb.comm = options.comm;

  auto cd = centroids_of(mesh, options.comm);
  if (!cd.ok()) return cd.error();
  const std::size_t n = cd.value().centroids.extent(0);

  Kokkos::View<double* [3], MemorySpace> centroids_dev(
      Kokkos::view_alloc("repartition.centroids", Kokkos::WithoutInitializing), n);
  Kokkos::View<std::int64_t*, MemorySpace> gids_dev(
      Kokkos::view_alloc("repartition.gids", Kokkos::WithoutInitializing), n);
  Kokkos::View<Weight*, MemorySpace> weights_dev(
      Kokkos::view_alloc("repartition.weights", Kokkos::WithoutInitializing), n);
  Kokkos::deep_copy(centroids_dev, cd.value().centroids);
  Kokkos::deep_copy(gids_dev, cd.value().global_ids);
  Kokkos::deep_copy(weights_dev, Weight{1});  // uniform weight; see the doc comment above

  auto rcb_result = rcb_partition<ExecutionSpace>(options.rcb, centroids_dev, weights_dev,
                                                   gids_dev);
  auto part_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rcb_result.part_id);

  std::vector<int> dest(n);
  for (std::size_t i = 0; i < n; ++i) dest[i] = part_host(i);

  return native::migrate(mesh, dest, options.comm);
}

}  // namespace fuse::partition
