#pragma once

/// Repartition migration for the native mesh: moving owned cells (with
/// their full closure) onto a new per-cell destination rank, e.g. the one
/// `fuse::partition::rcb_partition` computed. See docs/dev/native-mesh.md
/// for the protocol and its failure modes (rcb_moab's `MeshMigration.cpp`
/// silently destroyed cells, orphaned vertices, and hung on divergent
/// early returns -- this is written against those specifically).
///
/// STRUCTURED SO THE MPI PART IS ONLY THE EXCHANGE:
///   pack_for_migration(mesh, dest)   -> per-destination byte buffers
///   exchange_migration_buffers(...)  -> received buffers (the ONLY
///                                      function in this header's
///                                      implementation that names an MPI
///                                      type -- native_migration.cpp,
///                                      never a header, same layering rule
///                                      as fuse::partition's
///                                      detail/collectives.hpp)
///   unpack_migrated_mesh(...)       -> a mesh, deduped by global id
/// `migrate` composes exactly these three, which is also what makes Tier 0
/// emulation possible: a test can call `pack_for_migration` for P
/// simulated ranks, hand each destination's buffers directly to
/// `unpack_migrated_mesh` (skipping `exchange_migration_buffers`
/// entirely, since in-process there is nothing to exchange), and check
/// the recombined result -- exactly how the partition track tests RCB
/// against an emulated-rank oracle.
///
/// GPU-AWARE MPI. `native_migration.cpp` probes at configure time (see
/// `cmake/FuseMpiGpuAware.cmake`) whether the MPI library reports
/// CUDA/HIP awareness, and records the result in
/// `fuse::native::mpi_is_gpu_aware()` -- but `exchange_migration_buffers`
/// always stages through host buffers regardless, per the standing
/// priority (robustness before efficiency): a device-resident exchange
/// path is not implemented yet. See docs/dev/native-mesh.md.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "fuse/backend/native_mesh.hpp"

namespace fuse::native {

/// A communicator handle that names no MPI type, so this header stays free
/// of `mpi.h` (the same boundary convention `fuse::partition::CommHandle`
/// uses -- duplicated rather than shared, because `backend` sits BELOW
/// `partition` in the layering order and may not depend on it).
/// `kNullComm` means Tier 0: exactly one participant, no MPI symbol
/// touched.
using CommHandle = int;
inline constexpr CommHandle kNullComm = 0;

/// True if the MPI library this was built against reports CUDA- or
/// HIP-aware buffers (a configure-time probe; always false in a build
/// with no MPI). Informational only: `exchange_migration_buffers` does
/// not act on it yet (see the file header).
[[nodiscard]] bool mpi_is_gpu_aware();

/// Packs each of `mesh`'s OWNED cells whose destination is `dest[i]`
/// (`dest` has exactly `mesh.owned_count(mesh.dimension())` entries, in
/// the same order `MeshView`'s owned-entity iteration would enumerate
/// them: cell index order restricted to `is_owned`) into
/// `result[dest[i]]`, together with that cell's own id/type and its full
/// closure -- every vertex it touches, with coordinates and global id,
/// and (for a Polyhedron) its face structure. `num_ranks` sizes the
/// result.
///
/// Refuses `E-MESH-MIGRATE-INCOMPLETE` when `dest.size()` is not the
/// owned cell count, `E-MESH-MIGRATE-RANGE` when an entry names a
/// destination outside `[0, num_ranks)`. This is the LOCAL half of the
/// collective refusal `migrate` performs -- see its own comment.
[[nodiscard]] Result<std::vector<std::vector<std::byte>>> pack_for_migration(
    const NativeMeshView& mesh, const std::vector<int>& dest, int num_ranks);

/// The only function in this file whose implementation touches MPI: an
/// all-to-all exchange of `send[r]` (this rank's bytes for rank `r`) into
/// `result[s]` (bytes received from rank `s`), sized by the sender's own
/// count (an `MPI_Alltoall` for sizes, then `MPI_Alltoallv` for the
/// payload) -- always staged through host buffers (see the file header).
/// `comm == kNullComm` is Tier 0: `send` must have exactly one entry, and
/// this is the identity (`result = {send[0]}`), touching no MPI symbol.
[[nodiscard]] std::vector<std::vector<std::byte>> exchange_migration_buffers(
    const std::vector<std::vector<std::byte>>& send, CommHandle comm);

/// Unpacks every buffer in `received` (this rank's own share, from
/// `exchange_migration_buffers` or directly from `pack_for_migration` in
/// an emulated Tier-0 test) into one mesh. Vertices and, for a
/// Polyhedron, faces are deduplicated by global id (matching
/// `NativeMeshBuilder`'s own face-dedup rule: the same vertex SET, in
/// either cyclic direction). Every unpacked cell is OWNED by this rank
/// (a cell migrates whole, never as a ghost); vertex ownership is left
/// UNRESOLVED (every vertex tentatively `owned=true` on this rank) --
/// `migrate` resolves it next with a cross-rank collective, because a
/// shared boundary vertex's true owner cannot be decided from one rank's
/// received buffers alone.
[[nodiscard]] Result<MeshArrays<Kokkos::HostSpace>> unpack_migrated_mesh(
    const std::vector<std::vector<std::byte>>& received, int dim, int my_rank);

/// Resolves `arrays`' vertex ownership collectively and in place: the new
/// owner of vertex `v` is the MINIMUM rank, over the whole communicator,
/// among ranks that now hold a copy of `v` (an `MPI_Allgatherv` of every
/// rank's own `(global id, this rank)` pairs, then a local min-by-key --
/// see docs/dev/native-mesh.md's "ownership rule"). `comm == kNullComm`
/// makes every one of this rank's vertices trivially owned by rank 0,
/// touching no MPI symbol.
void resolve_vertex_ownership(MeshArrays<Kokkos::HostSpace>& arrays, CommHandle comm);

/// `pack_for_migration` + `exchange_migration_buffers` +
/// `unpack_migrated_mesh` + `resolve_vertex_ownership`.
///
/// COLLECTIVE REFUSAL. `pack_for_migration`'s two refusals are computed
/// from LOCAL data (this rank's own `dest`), but a caller must never let
/// one rank refuse and return while its neighbours proceed into
/// `exchange_migration_buffers`'s alltoall expecting every rank to
/// participate -- that is rcb_moab's own "divergent early return causing
/// hangs" failure mode (see the file header), reproduced if this
/// function returned the local Result directly. So `migrate` computes
/// its OWN validity locally, then reduces that outcome with every other
/// rank via one `MPI_Allreduce(MIN)` BEFORE calling the exchange -- if
/// any rank is invalid, EVERY rank returns the same refusal, none of
/// them touching the alltoall at all.
[[nodiscard]] Result<std::unique_ptr<NativeMeshView>> migrate(const NativeMeshView& mesh,
                                                              const std::vector<int>& dest,
                                                              CommHandle comm);

}  // namespace fuse::native
