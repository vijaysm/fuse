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
/// `migrate` composes exactly these three, which is also what makes
/// base-build-configuration emulation possible: a test can call `pack_for_migration` for P
/// simulated ranks, hand each destination's buffers directly to
/// `unpack_migrated_mesh` (skipping `exchange_migration_buffers`
/// entirely, since in-process there is nothing to exchange), and check
/// the recombined result -- exactly how the partition track tests RCB
/// against an emulated-rank oracle.
///
/// GPU-AWARE MPI. `fuse::native::mpi_is_gpu_aware()` is a run-time probe
/// (`MPIX_Query_cuda_support()` where OpenMPI's extension declares it,
/// falling back to `cmake/FuseMpiGpuAware.cmake`'s configure-time constant
/// where it does not). `exchange_migration_buffers` here does NOT consult
/// it: this file's own migration exchange always stages through host
/// buffers regardless, per the standing priority (robustness before
/// efficiency) -- a device-resident native-mesh migration path is still
/// future work. See docs/dev/native-mesh.md.
///
/// NOT shared with `fuse::transport::MpiTransport::is_gpu_aware()`
/// (src/transport/mpi_transport.cpp, docs/dev/device-resident-arcs.md
/// Phase 3), which duplicates the identical probe rather than calling this
/// one: `transport`'s library does not link `backend`'s (the CMake link
/// graph is stricter than the header-include direction
/// tools/check-layering.py enforces), the same reason `fuse::partition`'s
/// own `CommHandle` duplicates this file's, rather than including it.

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
/// `kNullComm` (-1) means the base build configuration: exactly one process, no
/// MPI symbol touched. Negative, not 0: OpenMPI's `MPI_Comm_c2f` hands out
/// small sequential NON-NEGATIVE Fortran handles with no reserved
/// always-invalid value (even `MPI_COMM_WORLD` itself can legitimately come
/// back as `0`), so `0` would silently collide with a real communicator and
/// get treated as "no MPI".
using CommHandle = int;
inline constexpr CommHandle kNullComm = -1;

/// True if MPI calls in THIS process may be handed CUDA device pointers
/// directly (a run-time probe; always false in a build with no MPI, and
/// forced false by `FUSE_TRANSPORT_HOST_STAGING=1` regardless of what the
/// probe would otherwise report). `exchange_migration_buffers` itself does
/// not act on it (see the file header); `fuse::transport::MpiTransport`
/// has its own, duplicated copy, not a caller of this one (see the file
/// header).
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
/// Rejects `E-MESH-MIGRATE-INCOMPLETE` when `dest.size()` is not the
/// owned cell count, `E-MESH-MIGRATE-RANGE` when an entry names a
/// destination outside `[0, num_ranks)`. This is the LOCAL half of the
/// collective rejection `migrate` performs -- see its own comment.
[[nodiscard]] Result<std::vector<std::vector<std::byte>>> pack_for_migration(
    const NativeMeshView& mesh, const std::vector<int>& dest, int num_ranks);

/// The only function in this file whose implementation touches MPI: an
/// all-to-all exchange of `send[r]` (this rank's bytes for rank `r`) into
/// `result[s]` (bytes received from rank `s`), sized by the sender's own
/// count (an `MPI_Alltoall` for sizes, then `MPI_Alltoallv` for the
/// payload) -- always staged through host buffers (see the file header).
/// `comm == kNullComm` is the base build configuration: `send` must have exactly one entry, and
/// this is the identity (`result = {send[0]}`), touching no MPI symbol.
[[nodiscard]] std::vector<std::vector<std::byte>> exchange_migration_buffers(
    const std::vector<std::vector<std::byte>>& send, CommHandle comm);

/// Unpacks every buffer in `received` (this rank's own share, from
/// `exchange_migration_buffers` or directly from `pack_for_migration` in
/// an emulated base-build-configuration test) into one mesh. Vertices and, for a
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
/// COLLECTIVE REJECTION. `pack_for_migration`'s two rejections are computed
/// from LOCAL data (this rank's own `dest`), but a caller must never let
/// one rank reject and return while its neighbours proceed into
/// `exchange_migration_buffers`'s alltoall expecting every rank to
/// participate -- that is rcb_moab's own "divergent early return causing
/// hangs" failure mode (see the file header), reproduced if this
/// function returned the local Result directly. So `migrate` computes
/// its OWN validity locally, then reduces that outcome with every other
/// rank via one `MPI_Allreduce(MIN)` BEFORE calling the exchange -- if
/// any rank is invalid, EVERY rank returns the same rejection, none of
/// them touching the alltoall at all. `unpack_migrated_mesh`'s own
/// POST-exchange rebuild is reconciled the identical way, before
/// `resolve_vertex_ownership`'s own Allgather/Allgatherv: a rank whose
/// received buffers it alone cannot rebuild must not return while its
/// peers proceed into that collective either.
[[nodiscard]] Result<std::unique_ptr<NativeMeshView>> migrate(const NativeMeshView& mesh,
                                                              const std::vector<int>& dest,
                                                              CommHandle comm);

}  // namespace fuse::native
