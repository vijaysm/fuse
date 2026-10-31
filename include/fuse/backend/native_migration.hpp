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
/// the recombined result -- exactly how the partition tests exercise RCB
/// against an emulated-rank oracle.
///
/// GPU-AWARE MPI (see docs/dev/native-mesh.md).
/// `fuse::native::mpi_is_gpu_aware()` is a thin
/// wrapper over `fuse::linalg::mpi_is_gpu_aware()` (comm.hpp,
/// src/linalg/mpi_gpu_aware.cpp) -- the SAME run-time probe
/// (`MPIX_Query_cuda_support()` where this MPI's own extension declares
/// it, falling back to `cmake/FuseMpiGpuAware.cmake`'s configure-time
/// constant where it does not), not duplicated between `backend`
/// and `transport`: both call the one copy in `fuse::linalg`, a layer
/// below each of them (tools/check-layering.py) that `backend` already
/// links PRIVATE.
///
/// The `std::vector<std::byte>`-based `exchange_migration_buffers` below
/// is what `migrate`/`build_native_remap_halo` call: it always stages
/// through host buffers. Wiring a GPU-aware replacement all the way
/// through `migrate`/`build_native_remap_halo` needs
/// `unpack_migrated_mesh`/`merge_halo_cells` to stop requiring a
/// host-resident `std::vector<std::byte>` buffer first. The GPU-aware
/// REPLACEMENT is a Space-templated overload, `exchange_migration_buffers<Space>`,
/// taking and returning `Kokkos::View<std::byte*, Space>` buffers --
/// `MPI_Alltoallv` on the raw device pointer when `Space` is a device
/// memory space and the probe above says this MPI build may take one
/// (and `FUSE_TRANSPORT_HOST_STAGING` is not forcing the fallback),
/// staged through host buffers (the SAME two-phase size-then-payload
/// shape the `std::vector` overload already uses) otherwise -- verified
/// byte-for-byte against the `std::vector` overload at 1/2/3 ranks, both
/// settings of `FUSE_TRANSPORT_HOST_STAGING`
/// (test/backend/native_migration_mpi_test.cpp), but NOT called by
/// `pack_for_migration`/`migrate`: the primitive is proven; wiring it
/// end-to-end is future work.

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
///
/// DEVICE-RESIDENT. The wire records are
/// built from `mesh.device_arrays()` via
/// `fuse::native::device::pack_to<false, Space>`
/// (include/fuse/backend/native/mesh_pack_device.hpp) -- no
/// `NativeMeshView` host-serving accessor (`cell_vertices`/`vertex_x`/
/// `cell_faces`) is called, so a device-resident caller's host mirror
/// (`host_mirrors_materialized()`) stays unmaterialized through this
/// call. `dest` is copied to a small device array first (bookkeeping,
/// not mesh geometry); the one remaining host round trip is the packed
/// WIRE BYTES themselves, copied back to this function's own
/// `std::vector<std::byte>` return type for `exchange_migration_
/// buffers` below, which is not yet GPU-aware.
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

/// GPU-aware replacement for the overload above (see this header's own
/// "GPU-AWARE MPI" paragraph for status: proven standalone, not yet
/// called by `migrate`/`build_native_remap_halo`). `send[r]` and the
/// result are `Kokkos::View<std::byte*, Space>` -- the SAME per-
/// destination buffers `fuse::native::device::pack_to`/`pack_fanout`
/// already produce, so a caller on a device-resident mesh never copies
/// its packed buffers to the host before calling this. When `Space` is a
/// device memory space and `fuse::linalg::mpi_is_gpu_aware()` is true,
/// `MPI_Alltoallv` is handed the device buffers' own raw pointers
/// directly; otherwise each buffer is staged to a host
/// `std::vector<std::byte>`, exchanged via the identical two-phase
/// (`MPI_Alltoall` for sizes, `MPI_Alltoallv` for payload) shape the
/// `std::vector` overload uses, and copied back. `comm == kNullComm` is
/// the identity, touching no MPI symbol, matching the `std::vector`
/// overload exactly.
template <class Space>
[[nodiscard]] std::vector<Kokkos::View<std::byte*, Space>> exchange_migration_buffers(
    const std::vector<Kokkos::View<std::byte*, Space>>& send, CommHandle comm);

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
