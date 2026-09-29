#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/base/result.hpp"
#include "fuse/partition/types.hpp"

/// Adapts a `fuse::MeshView` into the `(centroids, global_ids)` pair
/// `rcb_partition` wants: the OWNED entities at `mesh.dimension()` (the
/// mesh's own top topological dimension -- mesh_view.hpp: "1 for a line, 3
/// for a volume mesh"). This is fuse::partition's analogue of rcb_moab's
/// `RCBPartitioner::collect_owned_elements_in`.
///
/// WHY `dimension()`, NOT "scan every dim and take the highest with
/// entities": `MeshView`'s contract already promises a single, unambiguous
/// top dimension -- "entities of LOWER dimension may also exist" (see
/// `entity_count`'s doc). `SerialMeshBackend`'s box mesh is the concrete
/// case: it exposes owned entities at BOTH dim 3 (cells) and dim 2 (the
/// boundary skin) simultaneously, and `dimension() == 3` there, correctly.
/// Scanning dims 1..3 for "any owned entity" would misread the skin as a
/// second top-level dimension and reject every ordinary volumetric mesh --
/// exactly the false positive this design avoids.
///
/// What CAN legitimately go wrong, and is rejected rather than guessed:
/// two ranks disagreeing about `dimension()` -- e.g. a multi-block or
/// heterogeneous mesh where rank 0's local view answers 3 and rank 1's
/// answers 2. Silently taking one rank's answer for the "global" dimension
/// would drop the other rank's cells exactly the way rcb_moab's
/// predecessor to `collect_owned_elements_in` used to. So `dimension()` is
/// Allreduce-agreed (MAX and MIN both, so disagreement is detectable, not
/// just the winning value) BEFORE any entity is read -- never a
/// locally-computed early return (rcb_moab's CLAUDE.md "Global collective
/// discipline").
///
/// This is the ONLY file under fuse::partition that depends on
/// `fuse::backend` (docs/dev/layering.md; `tools/check-layering.py`'s
/// `partition` entry names it explicitly): the core `rcb_partition` in
/// rcb.hpp never mentions `MeshView` and needs only `base`.
namespace fuse::partition {

struct CentroidData {
  /// Owned-only centroids, host space (a `MeshView` is a host-side, virtual
  /// interface; the caller uploads to a device memory space if
  /// `rcb_partition` is called with a device `ExecutionSpace`).
  Kokkos::View<double* [3], Kokkos::HostSpace> centroids;
  Kokkos::View<std::int64_t*, Kokkos::HostSpace> global_ids;
  /// The topological dimension `centroids`/`global_ids` were read from --
  /// `mesh.dimension()`, once agreed.
  int dimension_used = 0;
};

/// COLLECTIVE over `comm` when `comm != kNullComm`: see the header comment
/// above. `comm == kNullComm` (one process) trivially agrees with
/// itself, so the disagreement path below is exercised only under real
/// MPI with two or more ranks holding differently-dimensioned meshes.
///
/// Errors, never a silent guess:
///   * `E-PART-MIXED-DIM` -- ranks disagree about `mesh.dimension()`.
///     Partitioning one rank's answer and dropping the cells implied by
///     another's is exactly the historical rcb_moab defect this rejects
///     instead of repeating.
///   * `E-PART-NO-ENTITIES` -- the agreed dimension is <= 0 (no rank has a
///     meaningful mesh loaded).
Result<CentroidData> centroids_of(const MeshView& mesh, CommHandle comm = kNullComm);

}  // namespace fuse::partition
