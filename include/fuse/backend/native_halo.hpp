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
/// parameter, the right choice at the cell counts this targets
/// (remap_plan.hpp's own `build_remap_halo` cost note makes the same
/// trade). Each returned list is sorted ascending and has no duplicate.
/// Two boxes that only TOUCH along a face -- an exactly aligned partition
/// boundary, the common case the test fixtures exercise --
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
///
/// DEVICE-RESIDENT. Built from
/// `mesh.device_arrays()` via `fuse::native::device::pack_fanout<true,
/// Space>` (include/fuse/backend/native/mesh_pack_device.hpp), the
/// many-to-many generalization of `pack_to` a halo copy needs: unlike a
/// migrated cell (one destination), the SAME owned cell may appear in
/// several of `send_by_local_index`'s per-rank lists, so the packing
/// domain is the flattened (cell, destination rank) send list, not the
/// cell set. No host-serving `NativeMeshView` accessor is called; see
/// `pack_for_migration`'s identical note on the one remaining host round
/// trip (the packed wire bytes, not the mesh).
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

/// HALO BRIDGING: the native-backend
/// counterpart to `fuse::build_remap_halo`'s own MOAB cross-placement
/// overload (remap_plan.hpp -- "the OVERLOAD this design needs, kept
/// separate from the one above rather than adding a second placement
/// field to RemapPlan itself"). The single-placement overload above
/// resolves BOTH `source` and `target` through `placement` -- correct for
/// `target` (that IS its own placement), wrong for `source` whenever
/// `source_placement != placement`: `register_default_placements()`
/// (native_backend.cpp) hands every component a placed view of EVERY
/// mesh, generated fresh over that component's OWN `RankSet`, and a
/// component's view of a mesh it does not itself decompose is the EMPTY
/// view documented there ("outside the processor set") -- not a refusal,
/// a legitimately empty one. Resolving `source` through the TARGET's own
/// placement therefore sees zero owned source cells on every rank that
/// actually holds the source mesh's real decomposition (they are outside
/// the TARGET's own RankSet), so the covering-set exchange below adopts
/// NOTHING: `build_native_remap_halo` (the single-placement form) reports
/// success (`commit_remap_halo` still runs and records the triple as
/// built), `has_remap_halo()` then answers true, and the Remap that
/// follows builds a matrix with zero rows contributed from the source
/// side -- a silently empty-matrix failure mode that gets past the
/// refusal guarding against it, because the halo genuinely WAS built, just
/// over the wrong source cells. `parallel.native_remap_halo_runner.{2,3}`
/// measured exactly this: both failed with `remap_weight_nonzeros == 0`
/// despite `initialize()` succeeding.
///
/// `source_placement` resolves ONLY `source`'s own view (`backend.
/// native_view(source, source_placement)`, the difference from the
/// single-placement form); `target` still resolves
/// through `placement`, and the adopted result is still COMMITTED under
/// `placement` (`backend.commit_remap_halo(source, placement, ...)`,
/// unchanged) -- so every existing lookup by the target's own placement
/// (`has_remap_halo`, `native_view`, `placed_view`, `supermesh_side` in
/// intx_weights.cpp) finds it exactly where it already expected to.
/// Only WHICH cells get offered for adoption changes; nothing about
/// where the result lives changes.
///
/// The single-placement overload above now DELEGATES to this one with
/// `source_placement == placement` -- the identity case this function's
/// own two-placement logic already reduces to correctly when the two
/// placements coincide (every existing caller and every existing test
/// of the single-placement form is therefore unaffected).
[[nodiscard]] Result<std::size_t> build_native_remap_halo(NativeMeshBackend& backend,
                                                          MeshId source, MeshId target, int dim,
                                                          ComponentId source_placement,
                                                          ComponentId placement);

// ===========================================================================
// The reverse exchange, the native backend's counterpart
// to `MoabMeshBackend::accumulate_remap_halo` (moab_backend.hpp). The
// forward fill above (`remap_halo_pattern` plugged into
// `fuse::linalg::make_mpi_exchanger`/`GhostExchanger::update`) has no
// reverse; this is it.
// ===========================================================================

/// COLLECTIVE, the reverse of the fill `GhostExchanger::update()` performs
/// along the SAME pattern: `ghosts[g]` is ADDED to its owner's own slot of
/// `owned` (`owned[pattern.remote_local_index[g]]` on rank
/// `pattern.owner_rank[g]`, for the `GhostPattern` `exchanger` was built
/// from), and every ghost slot is then zeroed. Exists for the identical
/// reason `MoabMeshBackend::accumulate_remap_halo` does (its own header
/// comment, moab_backend.hpp): a conservation audit that sums a source
/// cell's weight contributions across every rank holding a copy of it needs
/// the two partial sums -- the owner's own and every halo copy's -- folded
/// into one place, and only their sum is the cell's real coverage.
///
/// Built from `fuse::linalg::PackedExchanger::exchange_packed_adjoint`
/// (comm.hpp) -- the SAME "ADJOINT exchange" primitive
/// `src/backend/nodal_remap_distributed.cpp`'s own H^T scatter-add already
/// uses to fold ghost contributions back to their owners, reached here via
/// `unpack_index()` (this rank's own ghosts, in the order `send_buf` must
/// be packed) and `pack_index()` (this rank's own owned entries, in the
/// order the adjoint call's answer comes back) rather than any new
/// mechanism. Reusing the EXCHANGER object a caller already built for the
/// forward fill, not rebuilding one, is the design comm.hpp's own header
/// comment states ("the pattern itself ... is derived ONCE").
///
/// DEVIATION FROM `MoabMeshBackend::accumulate_remap_halo`'s LITERAL
/// SIGNATURE, stated rather than left to be discovered as an undocumented
/// diff: MOAB's version takes one combined `entity_count(dim) * components`
/// buffer (owned and halo slots together, addressed by MeshView local
/// index) because that is how MOAB's own halo exchange already addresses a
/// field. The native backend's existing `GhostExchanger`/`PackedExchanger`
/// machinery -- the SAME one the forward fill already uses
/// (`native_halo_mpi_test.cpp`'s own `BuildRemapHaloThenBuildWeightsThen
/// ExchangeReproducesALinearFieldToOrderH` test) -- addresses a field as
/// two SEPARATE arrays, `owned` (size `owned_n`) and `ghosts` (size
/// `exchanger.ghost_size()`), the same two arrays `GhostExchanger::update()`
/// itself takes; there is no existing native consumer of a SINGLE
/// MeshView-local-index-addressed buffer for this halo. This function
/// follows that SAME two-array idiom rather than introducing a different
/// one, so `exchange_remap_halo`'s and this function's buffers share the
/// addressing convention everything else in this file already uses.
/// `components` is not a parameter for the identical reason
/// `GhostExchanger::update()` itself has none: a caller needing more than
/// one component calls this once per component, exactly as
/// `native_halo_mpi_test.cpp`'s own forward-fill test calls `update()` once
/// per field it fills.
///
/// A no-op, returning success, when `ghost_n == 0` -- the identical
/// serial/no-halo short-circuit `exchanger.ghost_size() == 0` already
/// reports, correct for both a true one-rank run and a halo that adopted no
/// cells.
///
/// Errors, never throws: `E-MESH-ARG` for a null `owned`/`ghosts` pointer
/// with a nonzero size, or for `ghost_n != exchanger.ghost_size()`;
/// `E-MESH-HALO` if `ghost_n > 0` but `exchanger` exposes no
/// `PackedExchanger` facet (unreachable for a real `make_mpi_exchanger()`
/// result, which always implements both; reachable only from a caller
/// handing this function a `GhostExchanger` of its own that reports ghosts
/// without implementing the adjoint exchange -- a caller/factory mismatch,
/// not a data error); whatever `exchange_packed_adjoint` itself propagates
/// from its own `MPI_Alltoallv`.
[[nodiscard]] Result<void> accumulate_remap_halo(const fuse::linalg::GhostExchanger& exchanger,
                                                 double* owned, std::size_t owned_n,
                                                 double* ghosts, std::size_t ghost_n);

// ===========================================================================
// The conservative coverage check, DISTRIBUTED.
// `build_weights_via_supermesh`'s own `RemapFlags::Conservative` audit
// (intx_weights.cpp) reads `min_src_coverage` off the LOCAL supermesh -- the
// intersections THIS rank's own merged source/target views produced. On a
// partitioned read, a source cell this rank OWNS but whose only intersecting
// target cells live on a peer (adopted there as the peer's own halo copy of
// this cell) shows ZERO local coverage, even though the map covers it fully
// once the peer's own intersection is counted. This is that count: each
// halo copy's own share is folded into its OWNER's own slot (the reverse of
// the forward fill `GhostExchanger::update()` performs, `accumulate_remap_
// halo` above), and the resulting per-owned-cell minimum is then reduced to
// one number over the WHOLE base communicator -- "agree first" -- so every
// rank's Conservative check reads the SAME, now-correct value.
// ===========================================================================

/// `src_covered[c]` is the covered measure (NOT yet divided by the cell's
/// own measure) of LOCAL source cell `c` at `(source, target, dim,
/// placement)` -- `build_supermesh_weights`'s own per-cell coverage sum
/// (supermesh.cpp), recoverable by a caller from the `RemapWeights` it
/// already returned (`val[k] * tgt.measure(dim, row[k])`, summed by
/// `col[k]`) without this function needing a `RemapWeights` of its own.
/// `src_covered.size()` must equal `src_view.entity_count(dim)`, in THIS
/// backend's own owned-cells-first layout (`remap_halo_pattern`'s own
/// header comment: owned cells occupy `[0, owned_count(dim))`, this
/// triple's own ghost block `[owned_count(dim) + remap_halo_ghost_offset(
/// ...), ...)`-- the SAME convention `test/parallel/native_halo_mpi_test.
/// cpp`'s own exchange already relies on).
///
/// Returns the GLOBAL minimum, over every rank's own OWNED source cells (a
/// rank owning none contributes the neutral value `1.0`, never a stray `0`
/// that would make it look un-covered), of `(this cell's own coverage, after
/// folding in every peer's halo copy) / (this cell's own measure)` --
/// IDENTICAL on every rank that calls this, by construction, so a
/// Conservative refusal built from it is agreed before any rank acts on it,
/// never a locally computed condition one rank reaches alone.
///
/// `E-REMAP-PARALLEL` if the halo fold or the final reduction cannot
/// complete (no `FUSE_HAVE_MPI` in this build despite `communicator_size()
/// > 1`, a collective failure resolving the communicator, or the backend's
/// own bookkeeping reports no halo at all for this triple -- the caller is
/// expected to have already checked `has_remap_halo` before calling this,
/// so that last case is a caller bug, not a data error).
[[nodiscard]] Result<double> fold_and_reduce_source_coverage(
    const NativeMeshBackend& backend, MeshId source, MeshId target, int dim, ComponentId placement,
    const MeshView& src_view, const std::vector<double>& src_covered);

}  // namespace fuse::native
