#pragma once

/// The planar/volumetric remap halo for `fuse::native::NativeMeshBackend`
/// (docs/dev/native-halo.md). `fuse::build_remap_halo`
/// (src/backend/intx_weights.cpp) dispatches here, the same dynamic_cast
/// split it already uses for `MoabMeshBackend`, when the backend handed to
/// it is `NativeMeshBackend`.
///
/// WHAT THIS DOES, in one sentence: each rank forms the bounding boxes of
/// its OWNED target cells, every rank learns every other rank's target
/// boxes, and a rank's OWNED source cell travels (as a COPY, never a move)
/// to every other rank whose target boxes it overlaps -- the SAME
/// construction for a planar (dimension 2) or volumetric (dimension 3)
/// mesh, because an axis-aligned box-box overlap test in three coordinates
/// does not care which topological dimension the cells it bounds are.
///
/// WHY SEPARATE FUNCTIONS, NOT ONE OPAQUE ENTRY POINT. MPI cannot start in
/// this sandbox (docs/dev/testing.md), so `build_native_remap_halo` itself
/// can only be BUILT here, never RUN past one rank. Everything it does
/// beyond the real `MPI_Allgather`/`MPI_Allgatherv`/`exchange_migration_
/// buffers` calls is a PURE function of its inputs -- box overlap, the wire
/// packing, the merge-by-global-id -- and those are exposed here so a
/// tier-0 test can drive them directly with an IN-PROCESS emulation of
/// several ranks in one process, the same technique
/// test/backend/native_migration_test.cpp already uses for
/// `pack_for_migration`/`unpack_migrated_mesh`.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fuse/backend/native_backend.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"  // ComponentId, MeshId

namespace fuse::native {

/// Axis-aligned bounding box in physical (x, y, z) space. A planar mesh's
/// boxes simply have `lo[2] == hi[2]` (both 0 for a mesh in the z = 0
/// plane) -- there is no separate "2D box" type, because the overlap test
/// below needs none.
struct Box3 {
  std::array<double, 3> lo{0.0, 0.0, 0.0};
  std::array<double, 3> hi{0.0, 0.0, 0.0};
};

/// The bounding box of local entity `c` at `mesh.dimension()`, from its own
/// corner vertices (`cell_vertices`/`vertex_x`). Degenerate (lo == hi) for
/// a mesh with no extent in a coordinate, which is correct, not an error:
/// a planar mesh's z-box is a single point at z = 0.
[[nodiscard]] Box3 cell_box(const NativeMeshView& mesh, std::size_t c);

/// PURE, no communication: for each of THIS rank's OWNED source cells
/// (`my_owned_src_boxes`, in ascending LOCAL-INDEX order -- the same order
/// `owned_block_index` below assigns, so `result[i]` lines up with owned
/// source cell `i`), every OTHER rank whose target region (any box in
/// `all_target_boxes` owned by that rank, per `owner_of_target_box`) it
/// overlaps. `my_rank`'s own target boxes are excluded: a cell already
/// local to its own target region needs no copy. Brute force
/// (O(owned source cells x all target cells)) -- exact, no tuning
/// parameter, the right choice at the cell counts this track targets
/// (remap_plan.hpp's own `build_remap_halo` cost note makes the same
/// trade). Each returned list is sorted ascending and has no duplicate.
/// Two boxes that only TOUCH along a face -- an exactly aligned partition
/// boundary, the common case this track's own test fixture exercises --
/// count as NOT overlapping (native_halo.cpp's own `kTouchTol`): their
/// true geometric intersection has zero measure, so a copy of that cell
/// would sit in the supermesh contributing nothing.
[[nodiscard]] std::vector<std::vector<int>> covering_set_destinations(
    const std::vector<Box3>& my_owned_src_boxes, int my_rank,
    const std::vector<Box3>& all_target_boxes, const std::vector<int>& owner_of_target_box);

/// This rank's position among `mesh`'s OWNED cells at `mesh.dimension()`,
/// ascending LOCAL-INDEX order: `result[c]` is that position for local
/// index `c` if `is_owned(dim, c)`, unspecified otherwise. The "owned
/// block" position `fuse::linalg::DistVector`/`GhostPattern` (comm.hpp)
/// already use, computed once so every caller (the covering-set box list,
/// the pack below) agrees with the SAME numbering.
[[nodiscard]] std::vector<std::size_t> owned_block_index(const NativeMeshView& mesh);

/// Packs, for every destination named in `send_by_local_index` (LOCAL
/// cell indices -- every caller here passes only OWNED cells, since only
/// an owned cell has a meaningful `owned_block_index`), a wire record
/// matching `native_migration.cpp`'s own per-cell closure (global id,
/// type, vertex closure by global id + coordinates, and a Polyhedron's
/// face CSR) PLUS a leading owner rank (always `my_rank`: a halo cell is
/// COPIED, never moved, unlike a migrated one) and that cell's own
/// `owned_block_index` entry, so the receiving rank can address it for a
/// `fuse::linalg::GhostPattern` with no second round trip. `result[r]` is
/// what rank `r` receives; hand it to `exchange_migration_buffers` (a real
/// communicator) or concatenate directly across simulated ranks (an
/// in-process test).
[[nodiscard]] std::vector<std::vector<std::byte>> pack_halo_cells(
    const NativeMeshView& mesh, const std::vector<std::vector<std::size_t>>& send_by_local_index,
    const std::vector<std::size_t>& owned_block_index, int my_rank, int num_ranks);

/// `source`'s existing cells and vertices, copied verbatim (ownership and
/// owner rank preserved exactly, including any cell already adopted by an
/// earlier `build_native_remap_halo` call for a DIFFERENT target), PLUS
/// every cell found in `received` (one entry per sending rank, in
/// `pack_halo_cells`'s wire format) whose global id is not already present
/// -- the idempotency and "do not duplicate" rule
/// `MoabMeshBackend::adopt_remap_halo` documents, applied the same way
/// here. New cells are appended NON-OWNED, with the ORIGINAL sender as
/// their `cell_owner_of`, strictly after every existing cell -- so
/// `*owner_rank`/`*owner_owned_index` (parallel outputs, in that same
/// APPEND order) are exactly the new slots' `fuse::linalg::GhostPattern`.
/// Vertices are deduplicated by global id across the WHOLE result (existing
/// and new cells share one vertex numbering), matching
/// `unpack_migrated_mesh`'s own rule.
[[nodiscard]] Result<MeshArrays<Kokkos::HostSpace>> merge_halo_cells(
    const NativeMeshView& source, const std::vector<std::vector<std::byte>>& received,
    int my_rank, std::vector<int>* owner_rank, std::vector<std::int64_t>* owner_owned_index);

/// COLLECTIVE. The full construction: an `MPI_Allgather`/`MPI_Allgatherv`
/// of every rank's OWNED target boxes, `covering_set_destinations`,
/// `pack_halo_cells`, `exchange_migration_buffers`, `merge_halo_cells`, and
/// `backend.commit_remap_halo`. A successful no-op, recorded as BUILT with
/// zero cells, when `backend.communicator_size() <= 1`,
/// `backend.communicator() == kNullComm`, or `source == target` (the
/// identity map: a mesh's cells always overlap their own target region
/// without needing a copy of anything) -- the same three conditions
/// `build_remap_halo`'s MOAB branch short-circuits on, for the same
/// reasons. Returns the number of cells newly adopted (not the halo's
/// total size: a second call for a different `target` on an
/// already-haloed `source` adopts only what that pair still needs).
///
/// Errors, never throws: `E-REMAP-DIM` if `source` and `target`'s views
/// disagree on dimension, `E-MESH-ID`/`E-MESH-PLACEMENT` from resolving
/// either view -- both RANK-LOCAL and reduced with `MPI_Allreduce(MIN)`
/// before either of this function's two collective phases (the box
/// allgather and the cell exchange), so a refusal on one rank is never
/// returned while a peer proceeds into one of them. Separately, whatever
/// `merge_halo_cells`/`NativeMeshView::create_from_host` propagates from a
/// malformed incoming record (`E-MESH-CONNECTIVITY`, `E-MESH-DIM`) can
/// only be known AFTER the cell exchange has already run (it is derived
/// from that exchange's own output) -- this verdict gets its own
/// `MPI_Allreduce(MIN)`, after both collectives, before this function
/// returns, so the CALLER's next collective against the triple this call
/// commits never sees a rank-divergent outcome either. Either class of
/// refusal on one rank gets a plain `E-MESH-HALO-PEER-REFUSED` on every
/// other rank, naming no specific cause because it has none of its own.
[[nodiscard]] Result<std::size_t> build_native_remap_halo(NativeMeshBackend& backend,
                                                          MeshId source, MeshId target, int dim,
                                                          ComponentId placement);

}  // namespace fuse::native
