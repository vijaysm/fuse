#pragma once

/// Distributed general consistent interpolation (nodal-remap track, step
/// "Distributed rows and the exact transpose") -- lifts KNOWN_ISSUES D8 for
/// the two consistent methods ("consistent", "consistent-transpose") by
/// building H's rows, and applying H / H^T D, when the nodal source and the
/// evaluation points are each spread across possibly different rank sets.
///
/// THE OWNERSHIP RULE (unchanged from the task's own spec). Each evaluation
/// entity is owned by exactly one rank (`MeshView::is_owned`); only that
/// rank ever holds, or computes, that entity's row of H. This file's
/// `build_distributed_interp_rows` produces exactly one `DistributedRow` per
/// OWNED evaluation entity on the calling rank -- never one per local
/// (owned + ghost) entity, and never a row for an entity this rank does not
/// own.
///
/// THE DESIGN CHOICE THIS FILE MAKES, AND WHY. A scalable implementation
/// would locate each owned evaluation point's containing source cell with
/// `fuse::search::query_points_distributed` (a directory-based distributed
/// BVH query) and exchange only what each pair of ranks actually needs, in a
/// handful of point-to-point rounds. That is NOT what this file does.
/// Instead, EVERY rank first `MPI_Allgatherv`s the WHOLE source nodal
/// mesh's connectivity and coordinates (`gather_global_source`) into an
/// in-memory replica every rank holds in full, and then locates each of its
/// own owned evaluation points against that replica with EXACTLY the same
/// per-cell primitives `general_interp.hpp`'s serial builder already uses
/// and this track already validated (`invert_reference_map`,
/// `basis_weights`, the lowest-global-id tie-break) -- no new distributed
/// geometric algorithm at all. `apply_distributed_H`/`apply_distributed_HT`
/// make the SAME trade for the field VALUES, on every call: one
/// `MPI_Allgatherv` of every rank's owned source values (H), or one
/// `MPI_Allgatherv` of every rank's owned-row partial contributions,
/// followed by an IDENTICAL, index-order-independent local reduction on
/// every rank (H^T D) -- see that function's own header for why this makes
/// the result bit-identical across rank counts with no separate proof
/// needed for "sum in ascending order" bookkeeping: the full term LIST is
/// identical on every rank, so summing it in one fixed order (ascending
/// contributor rank, then ascending local index) gives one answer,
/// independent of which rank asks.
///
/// This is a robustness-over-efficiency trade this project's own standing
/// priority explicitly asks for ("robustness and accuracy beats efficiency
/// always... once we have a verified workflow, we can spend time making it
/// more efficient"): it replicates the WHOLE source mesh's geometry (not the
/// field data of every OTHER field transfer, only this one interpolation's own
/// source mesh entity set) on every rank, which does not scale to a source mesh too
/// large to fit on one rank, and every `apply_*` call is a fresh
/// `MPI_Allgatherv`, not a cached point-to-point plan. Revisiting this for a
/// directory-based, point-to-point implementation once the workflow above is
/// verified is exactly the kind of efficiency work that priority defers, not
/// forbids -- see `open_questions` in this task's report for the concrete
/// next step (reusing `fuse::search::query_points_distributed`, already
/// built and available, for the row-location step only).
///
/// WHAT REMAINS OPEN. `InterfaceDomainPolicy::ClampToBoundary` is not
/// implemented here -- only `Refuse` -- since nothing in this replicated
/// design needs it to be different from the serial builder's own open
/// question on the same policy, and this track's time did not extend to it.
/// `MeshBackend::placed_view` per-placement weight building (D5 item 3) is
/// this file's caller's responsibility, not this file's: every function
/// below takes the two SIDES' already-placed `MeshView`s directly, exactly
/// as `resolve_side`/`build_consistent_weights_for_plan` do for the serial
/// path, and does not itself call `backend.view()`/`backend.placed_view()`.
///
/// WIRED INTO Runner::Impl::resolve_remap since KNOWN_ISSUES.md D8 round 4
/// (REVIEW.md TF-6: this paragraph previously said "not yet," which the
/// 2026-09-25 wiring made false): `Runner` builds one `DistributedInterpRows`
/// per matching nodal Remap instruction at `initialize()` (`runner.cpp`'s
/// `build_nodal_dist_remap`) and applies it allocation-free per coupling
/// cycle, exactly the design this file originally described as pending.
/// The wiring covers the two consistent-interpolation methods this track's
/// scope names -- "consistent" with a nodal SOURCE, "consistent-transpose"
/// with a nodal TARGET -- and only when the orientation matches the method;
/// `resolve_remap` still refuses `E-RT-REMAP-NODAL-PARALLEL` for a
/// nodal-to-nodal channel on more than one rank (neither orientation
/// implemented), for an orientation/method mismatch, and for any other
/// method against a partitioned nodal side (the supermesh conservative
/// methods build FV<->FV weights only; per-placement supermesh weights for
/// a nodal side remain unimplemented, KNOWN_ISSUES D5 item 3) -- each a
/// deliberate, named remaining limitation, not evidence the wiring itself
/// is missing.
///
/// NO BLANKET `#if FUSE_HAVE_MPI` GATE (round 2 change from the first
/// submission): the declarations below are ALWAYS visible, mirroring
/// `fuse::partition::detail::collectives.hpp`'s own convention (its own
/// header comment: "mpi.h appears only in the .cpp, never in a header...
/// comm == kNullComm is a local no-op / identity ... touches no MPI
/// symbol"). `comm_handle` is a `fuse::search::CommHandle`
/// (fuse/search/types.hpp): `fuse::search::kNullComm` (0) means exactly one
/// process -- the base-build-configuration (no-MPI) path this track's Runner test exercises
/// -- and touches no MPI symbol at all; any other value is a REAL
/// communicator (`MPI_Comm_c2f`'d by the caller) and requires
/// `FUSE_HAVE_MPI`, rejecting `E-RT-NODAL-DIST-NOMPI` otherwise. This is what
/// makes the SAME algorithm (and the SAME call sites in runner.cpp)
/// compile and run at the base build configuration as well as under real MPI, rather than needing
/// two separate code paths for the single-rank and multi-rank cases.

#include "fuse/base/config.hpp"

#include <cstddef>
#include <memory>
#include <vector>

#include "fuse/backend/general_interp.hpp"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"  // DistributedRemapPlan::exchanger
#include "fuse/search/types.hpp"

namespace fuse {

/// One evaluation entity this rank OWNS, and its row of H: which source
/// nodal vertices (by GLOBAL id, ascending) contribute, and with what
/// weight.
struct DistributedInterpRow {
  std::size_t local_index = 0;  ///< this rank's own local index, at eval_dim
  std::vector<std::size_t> src_global_ids;  ///< ascending
  std::vector<double> weights;              ///< parallel to src_global_ids
};

struct DistributedInterpRows {
  std::vector<DistributedInterpRow> rows;  ///< one per OWNED evaluation entity
  /// The winning cell's own diagnostic, mirroring `RemapWeights::max_row_defect`
  /// -- `max_j |sum_k weights_jk - 1|` over this rank's own rows only (a
  /// global reduction is the caller's business, exactly like every other
  /// per-rank diagnostic in this codebase).
  double max_row_defect = 0.0;
};

/// Builds H's rows for every entity this rank OWNS at `eval_points`'s
/// dimension `eval_dim`, against the (placed) source nodal mesh
/// `src_nodal`. COLLECTIVE over `comm_handle` when it is a real
/// communicator (`MPI_Comm_c2f()`, the same convention
/// `CommGraph::build`/`accumulate_shared_rows` use) -- every rank of it must
/// call this, including one with zero owned evaluation entities or zero
/// local source cells. `comm_handle == fuse::search::kNullComm` needs no
/// MPI at all (exactly one process; see the file header).
///
/// Rejects, never throws (identically on every rank, since every rank sees
/// the same assembled global mesh): the same codes
/// `build_general_interp_weights` documents (`E-INTERP-NOGEOMETRY`,
/// `E-MESH-NO-CELL-VERTICES`, `E-INTERP-SHAPE`, `E-INTERP-DEGENERATE`,
/// `E-INTERP-NEWTON`, `E-INTERP-DOMAIN` under the default `Refuse` policy --
/// `ClampToBoundary` is not supported here, see the file header), plus
/// `E-RT-NODAL-DIST-MPI` for an MPI call that itself failed,
/// `E-RT-NODAL-DIST-NOMPI` when `comm_handle` names a real communicator but
/// this build of FUSE has no MPI at all, and `E-RT-NODAL-DIST-GID` when a
/// `MeshView` accessor (`global_id`/`centroid`) throws rank-locally while
/// assembling the replicated source mesh -- e.g. `MoabMeshView::global_id`
/// on a partitioned read with unnumbered vertices -- which is caught and
/// turned into this rank-agreed refusal rather than left to escape
/// uncaught on one rank while its peers enter the next collective.
[[nodiscard]] Result<DistributedInterpRows> build_distributed_interp_rows(
    const MeshView& src_nodal, const MeshView& eval_points, int eval_dim,
    search::CommHandle comm_handle);

/// Applies H: `out_owned[k]` (parallel to `rows.rows`, in the SAME order) is
/// `sum_i weights_ki * source_value(src_global_ids_ki)`, gathering every
/// rank's OWNED source values by global id via one `MPI_Allgatherv`
/// (component-major if `components > 1`, matching `apply_weights`'s own
/// convention).
///
/// COLLECTIVE over `comm_handle` when it is a real communicator (a no-MPI
/// local copy when `comm_handle == fuse::search::kNullComm`; see the file
/// header). `local_src_values` is this rank's local
/// `src_nodal` buffer (`src_nodal.entity_count(0) * components` doubles,
/// entity-major); only its OWNED rows are read (a ghost's value is a stale
/// copy of its owner's, per `MeshView`'s own contract, so gathering it too
/// would double-supply the same value under a different rank's name with no
/// guarantee the two agree).
[[nodiscard]] Result<void> apply_distributed_H(const DistributedInterpRows& rows,
                                               const MeshView& src_nodal,
                                               const std::vector<double>& local_src_values,
                                               std::size_t components,
                                               std::vector<double>& out_owned,
                                               search::CommHandle comm_handle);

/// Applies H^T D: scatters each OWNED row's `weights_ki * owned_loads[k]`
/// (already a LOAD, not a density -- see consistent_interp.hpp's "DENSITY
/// VS LOAD"; the caller multiplies by the evaluation side's own cell measure
/// first, exactly as `build_consistent_weights_for_plan`'s serial
/// "consistent-transpose" path does) to source vertex `src_global_ids_ki`,
/// summed there.
///
/// COLLECTIVE over `comm_handle` when it is a real communicator (a no-MPI
/// local copy when `comm_handle == fuse::search::kNullComm`; see the file
/// header). `owned_loads` is `rows.rows.size() *
/// components` doubles, in the SAME order as `rows.rows` (row-major over
/// components, matching `apply_distributed_H`'s output layout). `eval_points`/
/// `eval_dim` are the SAME evaluation view `rows` was built against --
/// needed here (not just at build time) because each contribution is tagged
/// with its OWNING evaluation entity's GLOBAL id before the exchange, and
/// every rank sums the resulting (rank-independent) global list of
/// contributions to one target vertex in ascending order of that global id.
/// THIS, NOT "ascending contributing rank", is what makes the total
/// bit-identical across rank counts: which rank owns a given evaluation
/// entity generally changes with the rank count (the ownership rule is a
/// function of the DECOMPOSITION, not of any global id), so summing in
/// Allgatherv's own by-rank concatenation order would silently reorder the
/// floating-point sum -- and therefore its round-off -- every time the rank
/// count changed, even though the SET of terms contributing to one target
/// vertex is exactly the same set. Sorting by the contributor's global id
/// first removes that dependency entirely.
/// `out_src_nodal_local` is OVERWRITTEN in full (`src_nodal.entity_count(0)
/// * components` doubles, every LOCAL row -- owned or ghost -- gets the same
/// global total for its global id, 0.0 for a vertex no row referenced).
[[nodiscard]] Result<void> apply_distributed_HT(const DistributedInterpRows& rows,
                                                const MeshView& src_nodal,
                                                const MeshView& eval_points, int eval_dim,
                                                const std::vector<double>& owned_loads,
                                                std::size_t components,
                                                std::vector<double>& out_src_nodal_local,
                                                search::CommHandle comm_handle);

/// runtime-exchange track, follow-up round (lead's layering decision):
/// the ONE-TIME resolved exchange `apply_distributed_H`/`apply_distributed_HT`
/// above do NOT use -- those two functions are UNCHANGED (every caller
/// that already depends on their per-call Allgatherv-and-replicate
/// behaviour, including the `comm_handle == fuse::search::kNullComm`
/// single-rank path, keeps it) -- this is the alternative a REAL
/// communicator's caller may build ONCE, at the same point
/// `build_distributed_interp_rows` itself is called, and reuse every
/// period: `apply_distributed_H`'s index resolution (which rank owns
/// each row's `src_global_ids`, and at what LOCAL position) does not
/// change between periods, only the field VALUES gathered through it
/// do, so resolving it once turns the per-period cost from "replicate
/// the whole source mesh's connectivity and rebuild a hash map" into
/// "one `MPI_Alltoallv` of a plain value array" -- the same trade
/// `fuse::transport::CommGraph` already makes for Transport's own
/// channel exchange.
///
/// `exchanger` is `fuse::linalg::make_mpi_exchanger()`'s own pattern,
/// COMPONENT-EXPANDED: ghost slot `(row_offset[r] + k) * components + c`
/// is row `r`'s `k`-th weight's contribution for component `c`, so a
/// SINGLE `GhostExchanger::update()` (or
/// `fuse::backend::DeviceGhostExchanger::update<Space>()`) call moves
/// every component at once, matching `apply_distributed_H`'s own
/// entity-major, component-minor layout convention exactly. `row_offset`
/// is UNEXPANDED (one entry per row, `rows.rows.size() + 1`);
/// `weights_flat` is UNEXPANDED too (one entry per `(row, k)` pair, the
/// same order `row_offset` indexes) -- the caller multiplies by
/// `components` itself wherever a slot or an owned index is actually
/// used, since the weight itself does not vary by component.
///
/// `owned_n` is `src_nodal.entity_count(0) * components`: THIS
/// participant's own local (owned + ghost) nodal buffer size, component-
/// expanded -- the same quantity `apply_distributed_H`'s own
/// `local_src_values.size()` contract already names, so the plan's
/// `exchanger` can be driven directly off a caller's existing per-arc
/// buffer with no repacking into an owned-only compaction.
struct DistributedRemapPlan {
  std::vector<std::size_t> row_offset;  ///< size rows.rows.size() + 1
  std::vector<double> weights_flat;     ///< size row_offset.back(), UNEXPANDED
  std::size_t components = 1;
  std::size_t owned_n = 0;  ///< src_nodal.entity_count(0) * components
  std::shared_ptr<const linalg::GhostExchanger> exchanger;
  /// H^T's own requirement `apply_distributed_HT` already documents --
  /// "every LOCAL row -- owned or ghost -- gets the SAME GLOBAL TOTAL for
  /// its global id" -- that `exchanger` above does not provide by
  /// itself: `exchanger`'s own pattern moves values FROM each row's
  /// source vertices TO the rows that reference them (or, run adjoint,
  /// scatter-adds contributions INTO each source vertex's OWNER), never
  /// back out to every OTHER rank that holds the same vertex as a
  /// GHOST. This second exchanger is that missing broadcast: one ghost
  /// slot per LOCAL ghost entity of `src_nodal` (component-expanded,
  /// same convention as `exchanger`'s own pattern), mapping to that
  /// entity's owner -- built from the SAME global ownership map
  /// `exchanger`'s own pattern resolution already computed, so this is
  /// an second, nearly-free `GhostPattern` derivation, not a second
  /// Allgather. `apply_distributed_H`'s own gather needs no equivalent:
  /// `rows.rows` only ever names entities THIS rank owns on the EVAL
  /// side (`DistributedInterpRow`'s own header comment), so there is no
  /// ghost of the OUTPUT to refresh there.
  std::shared_ptr<const linalg::GhostExchanger> ghost_fill_exchanger;

  /// Lead's determinism follow-up (2026-10-15): the H^T scatter-add's own
  /// FIXED summation order, resolved ONCE here instead of leaving it to
  /// `Kokkos::atomic_add`'s MPI-arrival order -- recovers
  /// `apply_distributed_HT`'s own "bit-identical across rank counts"
  /// guarantee (this struct's original round 2 submission traded that
  /// guarantee away; see KNOWN_ISSUES/report-exchange-2.md for the
  /// history) for this plan-based path too, at a one-time setup cost
  /// instead of paying for it every period.
  ///
  /// Built from `exchanger`'s own `PackedExchanger::pack_index()`
  /// indexing (the ADJOINT exchange's own "answer order",
  /// `exchange_packed_adjoint()`'s own `recv` layout) by an EXTRA
  /// one-time adjoint exchange of each contribution's CONTRIBUTOR
  /// EVALUATION ENTITY'S GLOBAL id -- the identical sort key
  /// `apply_distributed_HT`'s own file-header comment names as the one
  /// that does not depend on the decomposition, unlike the contributing
  /// RANK, which does -- one extra `MPI_Alltoallv` of a `double`-encoded
  /// gid array, paid once at setup, not per period.
  ///
  /// `target_row_offset[t]..target_row_offset[t+1]` (CSR, size
  /// `owned_n + 1`) names every answer slot destined for this rank's
  /// OWNED, component-expanded local position `t`; `sorted_answer_slot`
  /// (size `exchanger`'s own `pack_index().size()`) is a PERMUTATION of
  /// `[0, pack_index().size())`, grouped by `t` and, within each group,
  /// sorted ASCENDING by that slot's contributor global id. The per-
  /// period kernel sums each `t`'s bucket in this FIXED order -- one
  /// thread per `t`, a plain sequential loop over its own bucket, no
  /// atomics -- giving a result that depends only on the SET of terms
  /// (which is itself decomposition-independent), never on how many
  /// ranks produced them or in what order MPI delivered them.
  std::vector<std::size_t> target_row_offset;
  std::vector<std::size_t> sorted_answer_slot;
};

/// Builds the plan above from `rows` (already built by
/// `build_distributed_interp_rows`), `src_nodal` (the SAME nodal
/// `MeshView` that function and `apply_distributed_H`/`HT` take), and
/// `eval_points`/`eval_dim` (the SAME evaluation view and dimension --
/// needed here, not just at apply time, so this function can resolve
/// `target_row_offset`/`sorted_answer_slot` above from each row's own
/// evaluation entity's GLOBAL id).
/// COLLECTIVE over `comm_handle`, called ONCE per matching instruction at
/// setup -- the same discipline `build_distributed_interp_rows` itself
/// already documents. `comm_handle` must be a REAL communicator
/// (`fuse::search::kNullComm` is refused with `E-RT-NODAL-DIST-NOMPI`:
/// a single-rank, no-MPI caller has no index resolution worth
/// amortising -- `apply_distributed_H`/`HT`'s existing null-comm branch,
/// a plain local copy, already touches no MPI symbol and is cheaper than
/// building an `MpiExchanger` for one participant would be).
///
/// Resolves each row's `src_global_ids` to (owner rank, owner's own
/// LOCAL position) via one additional one-time `MPI_Allgatherv` of every
/// rank's own (global id, local index) pairs for its OWNED nodal
/// vertices -- `gather_global_source`'s own vertex gather, minus the
/// coordinates that one carries and this one does not need. Rejects
/// `E-RT-NODAL-DIST-OWNER` (a row names a global id no rank reported
/// owning -- the same code `build_distributed_interp_rows` uses for the
/// identical hazard), `E-RT-NODAL-DIST-GID-RANGE` (a contributor global
/// id exceeds a double's 53-bit mantissa, the same `gid_fits_double`
/// bound every other wire-format gid in this file is checked against),
/// and whatever `fuse::linalg::make_mpi_exchanger` itself asserts true
/// by construction (it cannot fail post-validation; see that function's
/// own contract).
[[nodiscard]] Result<DistributedRemapPlan> build_distributed_remap_plan(
    const DistributedInterpRows& rows, const MeshView& src_nodal, const MeshView& eval_points,
    int eval_dim, std::size_t components, search::CommHandle comm_handle);

}  // namespace fuse
