#pragma once

/// Private to src/runtime/. `Runner::Impl` is shared by exactly two
/// translation units -- runner.cpp, which interprets the flat stream, and
/// iteration.cpp, which drives spec §5.3's iteration regions -- so it lives in
/// a header rather than an anonymous namespace. It is deliberately NOT under
/// include/: nothing outside the runtime may depend on it.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/ir/lowering.hpp"
#include "fuse/num/accelerator.hpp"
#include "fuse/num/space_kernels.hpp"
#include "fuse/model/iteration.hpp"
#include "fuse/backend/nodal_remap_distributed.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/backend/spacetime.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/runtime/window_convergence.hpp"
#include "fuse/runtime/window_coupler.hpp"
#include "fuse/schedule/window.hpp"
#include "resume.hpp"

namespace fuse {
namespace detail {

/// A resolved buffer: either coupler-allocated storage or an alias to a
/// component's own array (spec §4.2).
///
/// `data` is a raw pointer resolved once in initialize(). For a coupler-owned
/// arc it points into `Impl::storage[i]`, which is sized once and never
/// resized afterwards; for a component-owned arc it is the address the
/// component registered, which the FieldRef contract pins for the whole run.
struct Slot {
  double*     data = nullptr;
  std::size_t elements = 0;
  int         components = 1;
  bool        owned_by_component = false;
  bool        persistent = false;

  [[nodiscard]] std::size_t total() const noexcept {
    return elements * static_cast<std::size_t>(components);
  }
};

/// What a `Constrain` compares, resolved at setup.
///
/// Spec §5.2 inserts `Constrain` immediately after a channel's `Remap`, so the
/// quantity it verifies is "did the remap move the integral?": the weighted
/// integral of the value entering the Remap, over the SOURCE region, against
/// the weighted integral of the value leaving it, over the TARGET region.
/// Comparing the Constrain's own input with its own output -- which is a copy
/// of it -- yields exactly zero for every input and is not a check at all.
struct ConstrainInfo {
  std::uint32_t before_arc = 0;      // the Remap's input, on the source region
  std::uint32_t weights_before = 0;  // index into Impl::weight_sets
  std::uint32_t weights_after = 0;
  /// Whether the two integrals are reduced over the base communicator before
  /// they are compared (KNOWN_ISSUES.md D10). True whenever one is configured:
  /// every rank of it walks the same instruction stream, a rank holding
  /// neither region contributes zero, and the totals -- hence the verdict --
  /// come out identical on every rank, so nothing is decided rank-locally.
  bool reduce = false;
};

/// Nodal-remap track, round 2 (KNOWN_ISSUES.md D8): replaces the
/// NodalReduceInfo/NodalReduceMode/reduce_nodal_remap_target() design this
/// struct's own predecessor comment described ("the path the refusal will
/// open"). That design was never wired to a real index (`nodal_reduce_idx`
/// in `resolve_remap()` was set only inside a branch that always returned a
/// `Failure` first) and is deleted below, per this round's explicit
/// instruction to state that decision rather than leave it as unreachable
/// scaffolding; `accumulate_shared_rows`/`copy_owner_shared_rows`
/// (fuse/transport/comm_graph.hpp) are UNCHANGED and untouched by this
/// deletion -- they are general-purpose primitives with their own direct
/// callers in `test/parallel/nodal_field_test.cpp` (a different track's
/// Transport-side ownership tests), not exclusive to the deleted function.
///
/// One `OpKind::Remap` instruction whose target is a nodal region on a
/// component placed on more than one rank, resolved via method `"consistent"`
/// or `"consistent-transpose"` (any other method still refuses
/// `E-RT-REMAP-NODAL-PARALLEL` exactly as before this round -- per-placement
/// weights for the supermesh conservative methods remain unimplemented,
/// KNOWN_ISSUES D5 item 3). Built once in `resolve_remap()`, applied
/// allocation-free (of THIS struct; `apply_distributed_H`/`HT`'s own
/// Allgatherv buffers are a distinct, per-call cost -- see
/// nodal_remap_distributed.hpp) every period from `Resolved::nodal_dist`.
struct NodalDistRemap {
  DistributedInterpRows rows;
  /// The nodal side's (placed) view -- `build_distributed_interp_rows`'s own
  /// `src_nodal` argument, and `apply_distributed_H`'s/`apply_distributed_HT`'s
  /// too. Owned by `backend`, not this struct; stable for the Runner's own
  /// lifetime exactly like `Side::view` (consistent_interp.hpp) and
  /// `remap_weights`'s pointers already assume.
  const MeshView* nodal_view = nullptr;
  /// The evaluation side's (placed) view: for `"consistent"` this is
  /// `resolve_remap`'s own `tgt` (the Remap's output region); for
  /// `"consistent-transpose"` it is `src` (the Remap's INPUT region) --
  /// `apply_nodal_dist_remap()` needs both `nodal_view` and `eval_view`
  /// regardless of direction, since `apply_distributed_HT` reads its
  /// `eval_points` argument at apply time, not just at build time (see that
  /// function's own header comment for why).
  const MeshView* eval_view = nullptr;
  int  eval_dim = 0;
  /// `true` for `"consistent-transpose"` (apply H^T D); `false` for
  /// `"consistent"` (apply H).
  bool transpose = false;
  /// Transpose only: whether the eval (source-of-load) side is ITSELF nodal
  /// (already a load, no measure to multiply by) -- mirrors
  /// `consistent_interp_detail::is_nodal(plan.src_disc, plan.src_order)` in
  /// `build_consistent_weights_for_plan`'s own transpose branch exactly, so
  /// the density-to-load scaling this struct's apply function does at
  /// runtime agrees with what the serial (single-rank) builder bakes into
  /// its weights at build time.
  bool src_is_load = false;
  std::size_t components = 1;
  /// `fuse::search::CommHandle`: `fuse::search::kNullComm` when
  /// `RunnerConfig::base_communicator` is unset (Tier 0, or a single-rank
  /// run with no communicator configured at all -- every entity is locally
  /// "owned" in that case, so the null-comm branch of
  /// `nodal_remap_distributed.cpp`'s `allgatherv_records` is exactly the
  /// right answer, not a stand-in for a real one); otherwise
  /// `*config.base_communicator` UNCHANGED -- the run's WHOLE base
  /// communicator, not a component-scoped split, per this struct's own
  /// build function's comment on why.
  int comm_handle = 0;
};

/// One space-time channel's operator and its window of source history
/// (spec §7.3, Task 6.3). Everything here is sized once in `resolve_spacetime`
/// and never resized afterwards; the sweep loop only indexes it.
///
/// THE OPERATOR. `temporal` is the `T` of `M = T (x) W`, built by
/// `build_temporal_projection` from the two components' intervals and the
/// plan's common period. The spatial half `W` lives on the channel's separate
/// `Remap` node, so what this node applies is `T (x) I`; `identity_*` below is
/// that `I`, materialised in COO so the SAME `apply_spacetime` entry point runs
/// here as in `test/unit/spacetime_test.cpp`, rather than a second, parallel
/// implementation of the tensor product that could drift from it.
///
/// THE HISTORY. `history` is `temporal.n_src_steps` consecutive copies of the
/// projected buffer, source-step-major. The channel's `Sample` writes row
/// `src_step` once per source tick; the `Interpolate` reads whichever rows row
/// `dst_step` of `T` names. It is a plain, fixed-size array and never grows:
/// `component.hpp`'s address stability rule is about registered fields, but a
/// coupler buffer that reallocated mid-run would break the same way.
struct SpacetimeInfo {
  TemporalProjection temporal;
  /// `row_begin[i] .. row_begin[i+1]` is the triplet range of target step `i`
  /// in `temporal`. Precomputed so the sweep does not scan for it.
  std::vector<std::size_t>  row_begin;
  std::vector<double>       history;
  std::vector<std::int32_t> identity_row, identity_col;
  std::vector<double>       identity_val;
  std::size_t               elements = 0;   // spatial entities
  int                       components = 1;
  std::size_t               width = 0;      // elements * components

  [[nodiscard]] SpatialWeightsView identity() const noexcept {
    return SpatialWeightsView{identity_row.data(), identity_col.data(),
                              identity_val.data(), identity_val.size(),
                              elements, elements};
  }
};

/// KNOWN_ISSUES.md D11 (closing) / matrix cell A-SPACES-TRANSPORT: the CSR
/// conversion of one `RemapWeights`' COO triplets, built ONCE alongside the
/// matching entry of `Impl::remap_weights` -- never per period.
///
/// `RemapWeights`' own header comment (fuse/backend/remap_plan.hpp) says
/// `row` is "target entity, ascending" -- true of the conservative supermesh
/// builder (`src/backend/intx_weights.cpp`), but NOT of every builder that
/// returns one: `consistent_interp.hpp`'s `"consistent-transpose"` path
/// builds H^T D by transposing H's own row-sorted triplets element-by-
/// element, so the RESULT's `row` (H's former COLUMN) is sorted by whatever
/// order H's rows happened to name it, not ascending. `build_remap_csr()`
/// (runner.cpp) does not trust the comment: it STABLE-sorts the triplet
/// indices by row before building `row_begin`, which is correct whether or
/// not the input already was sorted (a stable sort of an already-sorted
/// sequence is the identity permutation) and, critically, reproduces the
/// EXACT per-row floating-point accumulation order `apply_weights`' own
/// ascending-k COO scan would have used for that row's own entries --
/// stable sort preserves each row's entries' relative order among
/// THEMSELVES, which is all a per-row sum depends on, regardless of what
/// other rows' entries were interleaved with them in the original array.
///
/// `col` and `val` are therefore both copied here, in the sorted order --
/// `val` cannot be borrowed from `RemapWeights::val` directly the way the
/// single-component `remap_apply`'s test fixture does, because the sort may
/// have reordered it relative to the original array.
///
/// All three views live in `Kokkos::HostSpace` explicitly, not the default
/// execution space's memory space: `Runner::Impl::HostExecSpace` (below) is
/// always host (Serial/OpenMP/Threads), whose `memory_space` IS
/// `Kokkos::HostSpace`, so the two agree without a template parameter here.
struct RemapCsr {
  Kokkos::View<std::size_t*, Kokkos::HostSpace> row_begin;
  Kokkos::View<std::size_t*, Kokkos::HostSpace> col;
  Kokkos::View<double*,      Kokkos::HostSpace> val;
};

/// One `Transport` instruction's execution plan, resolved once at setup.
///
/// WHY `Resolved` HOLDS AN INDEX AND NOT THIS BY VALUE. `remap`, `constrain`
/// and `spacetime` are already std::int32_t indices, and `transport` was the
/// odd one out. transport.hpp:70-77 spells out the cost: `TransportSpec`
/// carries three std::vectors, so keeping one by value in a struct the hot
/// loop copies into `prepare()` is a heap allocation the moment any of them is
/// non-empty -- silently, and only in the configurations that fill them, which
/// are exactly the configurations this plan introduces.
struct TransportPlan {
  /// What `prepare()` is handed. On a rank of the source component this is
  /// `CommGraph::source_spec()`; on a rank of the destination component,
  /// `dest_spec()`; on a rank in neither, an empty spec that still takes part
  /// in every collective with zero counts.
  TransportSpec spec;

  /// The region both ends of this Transport live on. Transport preserves
  /// region -- `rules_channel.cpp` changes only `rank_set` -- so there is one,
  /// not two. Checked, not assumed, in Task 3.
  RegionId region;

  /// False until Task 4 attaches a `CommGraph`: no MPI, no
  /// `base_communicator`, or a `copy` transport. The interpreter then moves
  /// the whole arc, exactly as it did before this struct existed.
  bool has_graph = false;

  /// Matches whose two ends are on this rank: `local_dst[k]` takes the value
  /// at `local_src[k]`. Two parallel vectors rather than
  /// `CommGraph::LocalCopyPair` so this header stays free of the MPI-only
  /// comm_graph.hpp and `Impl` needs no #ifdef.
  std::vector<std::size_t> local_src, local_dst;

  /// Entity index of each `spec.global_ids` entry, in the same order:
  /// `CommGraph::source_indices()` gathering out of the field buffer on the
  /// source side, `dest_indices()` scattering into it on the destination side.
  std::vector<std::size_t> gather, scatter;

  /// True when the plan came from `build_plan_from_peers` (partially shared
  /// placements). Such a plan addresses both buffers BY LOCAL ENTITY INDEX --
  /// the source block on one side, the destination block on the other -- so
  /// `gather`/`scatter` are empty and the hot loop stages differently:
  /// `send` takes the whole source block, and only `peer_written` is copied
  /// back out of `recv`. Before this existed the hot loop ran the matched
  /// path's empty gather and scatter on a peer plan: it sent zeros and
  /// delivered nothing, unnoticed because no Runner test used partial overlap
  /// (KNOWN_ISSUES D5).
  bool from_peers = false;

  /// Peer plans only. How many doubles of the SOURCE block this rank holds
  /// (its own decomposition's extent times components; 0 off the source).
  std::size_t peer_src_total = 0;

  /// Peer plans only. Every destination-local entity index the plan writes --
  /// the receive peers' indices and the same-rank pairs' targets -- sorted and
  /// unique, computed once at setup so the hot loop allocates nothing.
  std::vector<std::size_t> peer_written;

  /// Sized once to `total()` and never resized. BOTH are that size on every
  /// participating rank, including the side that does not use one of them:
  /// `MpiTransport::move` (src/transport/mpi_transport.cpp:709-722) checks a
  /// single `n` against the plan and then rejects a null pointer for EITHER
  /// argument whenever `n != 0`, so handing it an empty vector's null `data()`
  /// on the unused side would be `E-TRANSPORT-NULL`.
  std::vector<double> send, recv;

  // Clamped exactly like MpiTransport::build_plan()'s own p_total
  // (src/transport/mpi_transport.cpp:663-664): a components < 1 spec still
  // sizes send/recv (and therefore the hot loop's own nc clamp in
  // runner.cpp) to at least one component per element, so the two never
  // disagree about how many doubles a "gather" or "scatter" entry is.
  [[nodiscard]] std::size_t total() const noexcept {
    return spec.elements *
           static_cast<std::size_t>(std::max(spec.components, 1));
  }
};

/// Everything an instruction needs that can be worked out at setup, so the
/// hot loop performs no lookup, no arithmetic on ids, and no allocation.
struct Resolved {
  std::size_t   out_total = 0;   // elements * components of the output slot
  AccumOp       accum = AccumOp::Instant;
  /// Index into `Impl::transport_plans`, or -1. Only set for
  /// `OpKind::Transport`.
  std::int32_t  transport = -1;
  std::int32_t  constrain = -1;  // index into Impl::constrains, or -1
  /// Index into `Impl::spacetimes`, or -1. Set on both halves of a space-time
  /// channel: the `Sample` that fills the history and the `Interpolate` that
  /// projects it.
  std::int32_t  spacetime = -1;
  /// Spec §5.4 (Task 5.5). Index into `Impl::remap_weights`, or -1.
  ///
  /// -1 does NOT mean "no remap": it means the PROVED-COINCIDENT identity.
  /// `resolve_remap()` only leaves it at -1 after checking, cell by cell, that
  /// the two regions hold the same entities in the same local order, so the
  /// element-wise copy the interpreter then performs IS the conservative map.
  /// Every other `Remap` carries real weights or the setup failed.
  std::int32_t  remap = -1;
  /// Index into `Impl::nodal_dist_remaps`, or -1 (round 2, replacing the
  /// deleted `nodal_reduce`/`NodalReduceInfo` design -- see
  /// `NodalDistRemap`'s own comment). Set on an `OpKind::Remap` instruction
  /// whose target is a nodal region distributed across more than one rank
  /// AND whose method is `"consistent"`/`"consistent-transpose"`; -1 means
  /// either not nodal, a single-rank placement (nothing is shared, the
  /// Remap's own local answer already IS the answer), or a method this
  /// round does not build distributed rows for (still refused
  /// `E-RT-REMAP-NODAL-PARALLEL`). When set, `execute()`'s `OpKind::Remap`
  /// case calls `apply_nodal_dist_remap()` INSTEAD OF the normal
  /// `remap`-indexed weights-apply -- `remap` is left at -1 for such an
  /// instruction, since the two are mutually exclusive execution paths, not
  /// a base answer plus a correction.
  std::int32_t  nodal_dist = -1;
};

/// One component advance the runtime owes within a period.
struct Tick {
  Rational     time{0};      // period-local, in [0, period)
  Rational     dt{1};        // the component's own interval
  std::size_t  slot = 0;     // index into the ComponentSet
  std::uint32_t order = 0;   // graph ComponentId, the tie-break at equal time
  std::int64_t local_step = 0;
  std::int64_t per_period = 1;  // ticks of this component in one period
};

/// One residual of an iteration region, and the coordinates it contributes to
/// the concatenated iterate vector the accelerator sees.
///
/// The iteration is over `y`, the values FUSE delivers into the participants'
/// `Requires` storage. One sweep evaluates `G(y)`: the components advance from
/// the checkpoint with `y` on their inputs, and the region body transports,
/// remaps and diagnoses their outputs. The residual is `G(y) - y`.
///
///   `value_arc` holds `G(y)` for this channel -- it is the arc the region's
///       `Residual` node reads, i.e. the last coupler-side value of the
///       channel before delivery.
///   `feed_arc` holds `y` -- it is the `Sink`'s output, which IS the
///       component's registered array (spec §4.2), so reading it is reading
///       exactly what the component was advanced with.
///   `sink_instruction` re-delivers an accelerated `y` mid-region. Without it
///       the components would be re-advanced from the same inputs every sweep
///       and the residual would never move.
struct RegionBlock {
  std::uint32_t value_arc = 0;
  std::uint32_t residual_arc = 0;
  std::uint32_t feed_arc = 0;
  std::size_t   sink_instruction = 0;
  std::size_t   offset = 0;
  std::size_t   length = 0;
  std::string   channel_name;
  std::string   source_field;
  std::string   target_field;

  /// 0 when `value_arc` is not a window member's own export arc (an
  /// ordinary, non-window channel, or a window channel this lookup
  /// could not resolve -- resolve_regions()'s own `window_export_meta`
  /// top comment, src/runtime/iteration.cpp); otherwise the exporting
  /// member's own `WindowMemberConfig::capacity`, the per-element
  /// stride `value_arc`'s raw samples are laid out with (sample 0 is
  /// the window-START value, 1..capacity-1 the per-substep END
  /// values). `run_region()`'s own per-block residual computation
  /// strips sample 0 before norming, using this and
  /// `window_export_valid_substeps` below, whenever this is nonzero --
  /// never touching `length`/`offset` themselves, which the
  /// accelerator and the rollback-verification byte-capture both read
  /// over the FULL arc and must keep seeing exactly what
  /// `advance_window()` wrote.
  std::size_t window_export_capacity = 0;
  /// Meaningful only when `window_export_capacity > 0`: the POLICY-
  /// INDEPENDENT count of real substep-end samples, `window_export_
  /// meta`'s own top comment -- `window_sample_capacity(window_length,
  /// interval, DtPolicy::Adjustable) - 1`, never `window_export_
  /// capacity - 1` directly (a `DtPolicy::Fixed` member's own capacity
  /// reserves one more slot than it actually fills on a
  /// non-straddling window instance).
  std::size_t window_export_valid_substeps = 0;
  /// Unit 2b (D19), step (c), CORRECTED (measured against a real host
  /// MPI run of `parallel.window_runner_mpi.{2,3}`, 2026-10-08): `true`
  /// when `value_arc` is NOT a window member's own export (an ordinary
  /// channel, where "owned" is decided by `length > 0` as it always
  /// was), OR when it IS one and the processor set that CONSUMES this
  /// export is hosted here (`WindowMemberConfig::export_consumer_
  /// hosted` -- the field `window_export_meta` is keyed by, step (a)'s
  /// correction). `false` only for a window block whose CONSUMER is NOT
  /// hosted here.
  ///
  /// This is deliberately the CONSUMER's own hosting, not the
  /// EXPORTER's: the compiled delivery chain a window block's residual
  /// measures (Transport/Epoch/Remap/Constrain/Diagnose) executes on
  /// the consumer's own processor set, so that is the one rank whose
  /// local `block_residual`/`block_value` for this block are ever
  /// genuinely written. Storing the EXPORTER's own hosting here (the
  /// pre-correction code) made the exporter's rank -- which never
  /// writes this block's buffer at all -- claim ownership, and the
  /// consumer's rank -- the only rank that does -- disclaim it; see
  /// `WindowMemberConfig::export_consumer_hosted`'s own comment for the
  /// measured host symptom this produced
  /// (`parallel.window_runner_mpi.{2,3}` converging after exactly one
  /// sweep with the partner's own state never updated).
  ///
  /// A non-owning rank's own `block_residual`/`block_value` for this
  /// block are computed from whatever stale buffer sits in `slots[]`
  /// (nothing writes there on that rank) and must never be treated as
  /// "I own zero" the way an ordinary channel's `length == 0` already
  /// is -- `decide_window_convergence()`'s own `WindowBlockSample::
  /// owned` reads this, not `length`, for a window-scoped region.
  bool window_hosted = true;
};

/// Everything one iteration region needs, resolved once in initialize().
/// Every vector here is sized at setup; the sweep loop appends to nothing but
/// `stats.residual_history`, whose capacity is reserved.
struct RegionRuntime {
  std::string name;
  std::size_t index = 0;   // into Impl::regions and Impl::region_stats
  std::size_t enter = 0;   // FIRST pass's first instruction (see `passes`)
  std::size_t exit = 0;    // FIRST pass's last, inclusive
  int         max_iterations = 1;

  /// Time windows track: true for a region whose IterationRegionDecl
  /// carries a `window_scope` (`fuse::CouplingWindowScope`). Changes two
  /// things resolve_regions()/run_region() otherwise assume: an ORDINARY
  /// region's body appears as exactly one contiguous (enter, exit) run
  /// per schedule period (mark_regions(), src/schedule/scheduler.cpp,
  /// stamps one run per period because an ordinary region's own clock
  /// ticks once per period by construction); a WINDOW-scoped region's own
  /// window may tick several times per period (a 7:13 ratio ticks the
  /// window's own clock `period / window_length` times), so its body
  /// appears as that many SEPARATE runs, each needing its own pass
  /// through run_region() with its OWN checkpoint (mirrors
  /// WindowCoupler::sweep_window()'s own save_state() at the top of every
  /// window, not once per period -- see resolve_regions()'s own comment).
  bool window_scoped = false;

  /// Time windows track: every (enter, exit) run mark_regions() stamped
  /// for this region in ONE schedule period, in stream order. Exactly one
  /// entry for an ordinary region (identical to `enter`/`exit` above,
  /// kept for that case so existing code reading them directly still
  /// works); one entry PER WINDOW INSTANCE for a window-scoped region.
  /// step_period() (src/runtime/runner.cpp) reads this to know how many
  /// times to call run_region() in one period and with which bounds --
  /// never `enter`/`exit` directly once a region can have more than one
  /// pass.
  std::vector<std::pair<std::size_t, std::size_t>> passes;

  /// Time windows track (C3): every LAGGED window_import arc this region
  /// owns (`Runner::Impl::window_member_config`'s own `import_lagged`).
  /// `run_region()`'s own sweep loop snapshots each of these into
  /// `Impl::previous` at the TOP of every sweep, BEFORE any instruction
  /// in the sweep's body runs -- so a lagged import reads what was
  /// delivered LAST sweep (Jacobi's/GaussSeidel's lagged direction's own
  /// rule), never this sweep's own fresh delivery, regardless of where
  /// the delivering Sink happens to sit in the instruction stream.
  std::vector<ArcId> window_lagged_import_arcs;

  /// Time windows track: every WindowAdvance node this region owns (one
  /// per window member, `Runner::Impl::window_member_config`'s own keys
  /// restricted to this region). `run_region()`'s own straddle-replay
  /// step, at the top of every window instance's own pass (BEFORE that
  /// instance's checkpoint is taken -- mirrors WindowCoupler::run_window()
  /// calling resolve_pending_straddles() before sweep_window()'s own
  /// save_state()), walks this list to find any member whose PREVIOUS
  /// window instance left `Impl::window_member_state[node].straddle_
  /// pending` set.
  std::vector<NodeId> window_member_nodes;

  ConvergenceCriterion         convergence;
  std::unique_ptr<Accelerator> accel;

  /// S7.2: true for a region whose IterationRegionDecl::acceleration.kind
  /// is AccelKind::Newton. `accel` stays null for such a region --
  /// resolve_regions() never calls make_accelerator() for it, because
  /// Newton is not an Accelerator and make_accelerator() refuses
  /// AccelKind::Newton with E-ACCEL-KIND (src/num/accel_relaxation.cpp
  /// has no `case` for it). run_region() reads this flag, before ever
  /// touching `accel`, to run the Newton-Krylov solve (run_region_newton(),
  /// this file's own sibling in iteration.cpp) instead of the Picard sweep
  /// loop entirely.
  bool newton = false;

  std::vector<RegionBlock> blocks;
  std::size_t              n = 0;  // total length of the concatenated iterate

  /// measure i of `convergence` is tested against block `measure_block[i]`.
  std::vector<std::size_t> measure_block;

  /// ComponentSet indices whose state the region re-advances, from the derived
  /// checkpoint set, and the ticks that re-advance them.
  std::vector<std::size_t> component_slots;
  std::vector<std::size_t> tick_indices;

  /// Loop-carried coupler buffers (`IterationRegion::carried`). Recorded so
  /// that resolve_regions() can CHECK that every one of them is a value or
  /// residual arc the sweep loop keeps in step with the delivered iterate.
  /// They are deliberately not rolled back between sweeps; the reasoning, and
  /// the arithmetic that settles it, is in resolve_regions().
  std::vector<std::uint32_t>       carried_arcs;

  /// Spec §5.3's distinguishing claim over preCICE: the runtime verifies that
  /// the component saved what the analysis said it must, rather than trusting
  /// `supports_rollback()`. These are the `Source` outputs of the checkpointed
  /// components -- the state they write in `advance()` that FUSE can see -- and
  /// the bytes they held when `save_state()` was called.
  std::vector<std::uint32_t>       verify_arcs;
  std::vector<std::vector<double>> verify_save;
  std::vector<std::size_t>         verify_component;  // index into component_slots

  // Scratch, all sized once.
  std::vector<double> x_old, x_new, x_out;         // n
  std::vector<double> block_residual;              // per block, ||G(y) - y||
  std::vector<double> block_value;                 // per block, ||G(y)||
  std::vector<double> block_first;                 // per block, sweep 0's residual
  std::vector<double> residual_norms;              // per convergence measure
  std::vector<double> reference_norms;             // per convergence measure

  /// R-10 (REVIEW.md): the sorted, de-duplicated union of ranks of every
  /// component this region checkpoints -- the same derivation
  /// resolve_epoch_participants() uses for an Epoch, computed once here
  /// instead of once per Epoch instruction because a region has exactly one
  /// participant set for its whole lifetime. Empty (the default) means
  /// "no cross-rank reduction": either no communicator is configured, or the
  /// region's components all sit on one rank, and run_region() computes its
  /// residual purely locally, exactly as it always has.
  std::vector<int> participants;
};

}  // namespace detail

struct Runner::Impl {
  const CouplingGraph& graph;
  const OperatorDAG&   dag;
  ExecutionPlan        plan;
  const MeshBackend&   backend;
  RunnerConfig         config;

  /// Unit 2b (D19): this process's own rank in `config.base_communicator`,
  /// or unset when no base communicator is configured (tier 0, or a
  /// single-rank run) -- the SAME value `build_ticks()` already computed
  /// locally for the ordinary-tick `hosted[]` check (its own comment there
  /// has the full account), now a member so `resolve_regions()`, called
  /// right after `build_ticks()` in `initialize()`, can read the identical
  /// value when resolving each window member's own `WindowMemberConfig::
  /// hosted` below -- one computation, not two that could drift apart.
  std::optional<int>   this_process;

  /// KNOWN_ISSUES.md D11 (closing): the coupling hot loop runs on Kokkos'
  /// DEFAULT HOST execution space -- OpenMP if the build enabled it, Serial
  /// otherwise (`Kokkos::DefaultHostExecutionSpace`'s own definition) --
  /// instead of the hand-rolled serial loops `execute()` used to call
  /// directly. Not templated on `Impl` itself: a device build (A5) is out
  /// of scope for this closing, and every `SpanOf`/`ConstSpanOf` this file
  /// wraps assumes host-accessible memory -- component-owned arrays,
  /// coupler storage, `RemapWeights` triplets -- which is exactly what this
  /// alias's `memory_space` (`Kokkos::HostSpace`) is, for every space it can
  /// resolve to.
  using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

  bool initialized = false;

  std::vector<std::vector<double>> storage;       // coupler-owned, one per arc
  std::vector<detail::Slot>        slots;         // parallel to arcs
  std::vector<std::vector<double>> previous;      // previous iterate, for Relax
  std::vector<std::size_t>         sample_count;  // per arc, for Accumulate
  std::vector<detail::Resolved>    resolved;      // parallel to instructions
  std::vector<detail::Tick>        ticks;         // one period's worth, sorted
  /// Every component's steps, identical on every process. `ticks` is the
  /// subset this process performs (a component is stepped only on its own
  /// processor set); checks whose verdict must agree across processes read
  /// this list instead, so no process rejects a configuration alone.
  std::vector<detail::Tick>        all_ticks;
  std::vector<detail::ConstrainInfo> constrains;
  /// Spec §7.3: one entry per space-time `Interpolate` node. Built once in
  /// resolve_spacetime(); `spacetime_by_node` maps the DAG node id to the
  /// index, so both the `Sample` and the `Interpolate` of one channel resolve
  /// to the same entry.
  std::vector<detail::SpacetimeInfo> spacetimes;
  std::unordered_map<std::uint32_t, std::uint32_t> spacetime_by_node;
  /// Spec §5.4: the spatial half `W`. Built ONCE per distinct `RemapPlan` in
  /// resolve_remap(), which runs inside setup(); `remap_cache` owns the
  /// matrices and hands out pointers that stay valid for its lifetime, so
  /// nothing here is rebuilt, re-sized or re-hashed once the run starts.
  /// Constructing a supermesh is the single most expensive thing this library
  /// does -- doing it per step would not be a slowdown, it would be a
  /// different program -- so the cache is deliberately not consulted below
  /// setup() at all: the hot loop indexes `remap_weights` and nothing else.
  RemapCache                          remap_cache;
  std::vector<const RemapWeights*>    remap_weights;
  /// KNOWN_ISSUES.md D11 (closing): the CSR conversion of the SAME entry of
  /// `remap_weights`, at the same index -- built once, right where that
  /// entry is pushed (resolve_remap()), never per period. See
  /// `detail::RemapCsr`'s own comment. `Resolved::remap` indexes both
  /// vectors together.
  std::vector<detail::RemapCsr>       remap_csr;
  /// One entry per `OpKind::Remap` instruction targeting a distributed nodal
  /// region via `"consistent"`/`"consistent-transpose"` (detail::
  /// NodalDistRemap); `Resolved::nodal_dist` indexes it. Built once in
  /// resolve_remap(), read-only after setup. Holds real `MeshView*`
  /// pointers (owned by `backend`, stable for the Runner's lifetime -- see
  /// `NodalDistRemap::nodal_view`'s own comment) and one already-built
  /// `DistributedInterpRows`, unlike `nodal_reductions` before it (deleted
  /// this round; see `NodalDistRemap`'s own comment for why).
  std::vector<detail::NodalDistRemap> nodal_dist_remaps;
  /// Element measures, expanded to one weight per (element, component) so a
  /// weighted_sum can run straight over a buffer. Built once at setup.
  std::vector<std::vector<double>> weight_sets;
  std::unordered_map<std::uint64_t, std::uint32_t> weights_by_region;
  /// For a component-owned arc that is a `Source` output, the owning graph
  /// component; -1 for everything else (including `Sink` outputs, which the
  /// coupler writes and the component only reads).
  std::vector<std::int32_t>        slot_source_owner;
  std::unique_ptr<Transport>       transport;

  /// One Transport per CommGraph-backed entry of `transport_plans`, at the
  /// same index; the slot is null for a plan without a graph, which moves the
  /// whole arc through `transport` instead.
  ///
  /// Separate instances, not one shared: a Transport carries the plan it was
  /// built for (`MpiTransport::prepare()` refuses a spec other than the
  /// planned one), so two cross-rank channels cannot take turns on a single
  /// object without re-planning at every step. Two independent cross-rank
  /// couplings is the ordinary multiphysics case; before 2026-09-20 the
  /// runner refused any plan with more than one (KNOWN_ISSUES D2).
  ///
  /// Every rank builds the same number in the same order. That is required,
  /// not incidental: `MpiTransport::set_communicator()` duplicates the
  /// communicator and `build_plan()` is collective, so a rank that built a
  /// different set would desynchronise the next collective. `transport_plans`
  /// comes from the plan, which is identical on every rank.
  std::vector<std::unique_ptr<Transport>> graph_transports;
  /// One entry per `OpKind::Transport` instruction, in instruction order.
  /// Built in resolve_instructions(); Task 4 fills in the CommGraph-derived
  /// maps. Never resized after setup.
  /// Row `s` marks every ComponentSet slot that reads, transitively, something
  /// slot `s` provides -- the components whose step would consume a value a
  /// failed step never produced. Square, `components.size()` per side, built
  /// once in setup() from the channels of the coupling graph.
  ///
  /// It exists so that a failing period can keep advancing the components that
  /// are NOT downstream of the failure. Before 2026-09-21 a failure anywhere
  /// stopped every later advance in the period, which for a plan whose ticks
  /// all fall in the trailing loop meant no component advanced at all --
  /// including ones on a channel with no relation to the failure.
  ///
  /// Downstream only, deliberately. A component UPSTREAM of the failure reads
  /// nothing the failed step was supposed to produce, so its own step is as
  /// valid as it would have been in a period that succeeded.
  std::vector<std::vector<char>>     downstream_of;

  /// True for a slot that must not advance for the rest of this period: the
  /// slot the failure came from, plus everything `downstream_of` it. Empty
  /// until a failure occurs. Sized with `downstream_of`.
  std::vector<char>                  advance_blocked;

  std::vector<detail::TransportPlan> transport_plans;
  /// Index into `plan.instructions` of the instruction each plan belongs to,
  /// parallel to `transport_plans`. attach_comm_graphs() needs the instruction
  /// to read its input and output arcs; `Resolved` does not carry them.
  std::vector<std::size_t>           transport_instruction;
  RunStats                         stats;
  std::int64_t                     period_index = 0;

  std::vector<detail::RegionRuntime> regions;
  std::vector<IterationStats>        region_stats;   // parallel to regions
  std::vector<std::string>           region_explain;  // parallel to regions

  /// Time windows track (C3): everything about ONE window-scoped member
  /// that `advance_window()` needs and that never changes across sweeps
  /// or window instances -- resolved ONCE, in `resolve_regions()`, from
  /// the graph alone (`CouplingWindowScope`, `ComponentDecl::interval`),
  /// never from a partner's runtime state. Keyed by the `NodeId` of this
  /// member's OWN `OpKind::WindowAdvance` instruction (stable for the
  /// whole run; `finish_window_regions()`, src/ir/window_lowering.cpp,
  /// builds at most one such node per member per region).
  struct WindowMemberConfig {
    std::size_t component_slot = 0;  ///< this member's own ComponentSet slot
    /// Unit 2b (D19): whether THIS process hosts this member -- resolved
    /// once in `resolve_regions()` from `Runner::Impl::this_process` and
    /// this member's own `ComponentDecl::ranks`, the IDENTICAL formula
    /// `build_ticks()`'s own `hosted[]` already uses for ordinary ticks
    /// (that array's own comment has the full account). `true`
    /// unconditionally when `this_process` is unset (tier 0, or a
    /// single-rank run) -- every existing single-process caller keeps
    /// meaning exactly what it meant before this field existed.
    /// `advance_window()` reads this FIRST, before anything else: a
    /// non-hosted member takes no substeps, calls neither `advance()`
    /// nor `save_state()`/`restore_state()`, and yields no
    /// `ExternalSuspend` -- its own export arc is left untouched here,
    /// for the field transfer (Transport/Remap/Sink) to deliver instead.
    bool        hosted = true;
    /// Unit 2b (D19) correction, measured against a real host MPI run of
    /// `parallel.window_runner_mpi.{2,3}` (2026-10-08): whether the
    /// processor set that CONSUMES this member's own export -- the
    /// channel's TARGET component, resolved the identical way `hosted`
    /// above resolves THIS member's own processor set -- includes
    /// `this_process`. `true` whenever this member exports nothing (no
    /// channel to consume it) or `this_process` is unset, the same
    /// permissive default `hosted` itself uses.
    ///
    /// `window_export_meta` (src/runtime/iteration.cpp) is keyed by
    /// THIS member's own export arc but must answer a question about
    /// the OTHER end of the field transfer: a window-scoped block's
    /// residual is measured on the CONSUMER's own processor set (the
    /// compiled Transport/Epoch/Remap/Constrain/Diagnose chain runs
    /// there, never on the exporter's), so the flag `run_region()`
    /// gates convergence ownership with must say "is the consumer
    /// hosted here", not "is the exporter hosted here". Before this
    /// field existed, `window_export_meta` stored `hosted` (THIS
    /// member's own flag) instead: `test/golden/window_two_member.plan`
    /// shows the a2b block's value arc (%6, `a2b/diagnose`) is reached
    /// from A's export (%0) only through `Transport{0->1} -> Epoch ->
    /// Remap -> Constrain -> Diagnose`, the chain B's own processor set
    /// executes -- so on A's rank, the a2b block was wrongly marked
    /// owned (A's own `hosted` is true there) even though A's rank
    /// never writes %6 at all, and on B's rank it was wrongly marked
    /// NOT owned (A's own `hosted` is false there) even though B's rank
    /// is the one genuine owner. Both ranks disagreed with the truth in
    /// the same wrong direction for the mirrored b2a block. MEASURED
    /// consequence on the host run: `window_export_meta`'s stale,
    /// never-written local buffer on the wrongly-"owning" rank read as
    /// a 0/0 residual/reference pair, which `decide_window_convergence
    /// ()` accepts as converged (`0 <= tol * 0`) on sweep 0 --
    /// `got.iterations_per_window[0] == 1` against a reference of 37,
    /// and every later window the same way (35, 25, 15, 3 vs 1 every
    /// time), with `got.x_b_per_window[0] == 0.2`, its own unmodified
    /// initial value, since no real exchange ever ran to convergence.
    bool        export_consumer_hosted = true;
    /// Unit 2b (D19), step (b): the channel this member's own IMPORT
    /// came through -- `ChannelId{}` (invalid, the default) when this
    /// member imports nothing. Set in `resolve_regions()`'s first pass
    /// alongside `provider_exported_field`/`cfg.required`, read back in
    /// its second pass (after every member's own EXPORT arc has been
    /// primed, this struct's own `cfg.provided` -> `ins.out`) to find
    /// and re-run this channel's own compiled delivery chain --
    /// Transport/Epoch/Interpolate/Sink, whichever of those
    /// window_lowering.cpp's "ordinary per-channel rule" actually
    /// emitted -- exactly once, so this member's own `window_import`
    /// arc is primed THROUGH that chain and never through a direct
    /// `registry.get(provider_exported_field)` read (KNOWN_ISSUES.md
    /// D19's own evidence 3).
    ChannelId   import_channel;
    Rational    interval{1};
    DtPolicy    policy = DtPolicy::Adjustable;
    int         order = 2;
    Rational    window_length{1};
    /// `fuse::window_sample_capacity(window_length, interval, policy)` --
    /// the SAME function that sized this member's own exported arc
    /// (`window_lowering.cpp`'s `window_export_width()`), so a mismatch
    /// between "how wide the compiled field is" and "how many samples
    /// this member believes it owns" cannot happen.
    std::size_t capacity = 1;
    /// Whether THIS member's own IMPORT (not export) is loop-carried,
    /// per `effective_channel_lag()` (fuse/model/coupling_graph.hpp) --
    /// the SAME function `lower()`'s own type-check and `window_import_
    /// is_lagged()` (src/schedule/scheduler.cpp) already use, so
    /// `advance_window()` cannot disagree with the compiled graph about
    /// which direction is lagged. `false` when this member has no
    /// import at all (a pure exporter).
    bool        import_lagged = false;
    ArcId       window_import;  ///< this member's own import arc, or invalid
    /// Unit 3 (S6 phase 3, Conformance), MEASURED this session (traced,
    /// element 0, hexfloat, both channels, period 0 side by side against
    /// the in-process reference -- the lead's own instruction): the SAME
    /// `import_view_arc` resolve_regions()'s own walk-back already
    /// computes for `window_import_field_arc` (this file's own comment
    /// on that walk has the full account) -- the chain's TRUE root, an
    /// ordinary channel's own `window_import` for that kind and a
    /// LowFive-bound one's own `ExternalDelivery` arc for this kind,
    /// ALWAYS holding the application's actual write the instant it
    /// happens, regardless of whether this member's own import CHAIN
    /// (Remap..Sink) has executed yet THIS sweep. `advance_window()`'s
    /// own t=0 history-correction (this file) needs exactly that: under
    /// the scheduling-order fix (OpKind::ExternalDelivery's own comment,
    /// include/fuse/ir/op.hpp), a member scheduled BEFORE its own
    /// import's source_producer (U, in the two-member Brusselator graph,
    /// since V's own WindowAdvance -- and therefore U's own import
    /// chain, which stands after it -- runs AFTER U's) reaches its own
    /// t=0 correction before that chain has ever run this sweep, so
    /// `window_import`'s own buffer (the Sink's downstream COPY) is
    /// still unwritten -- MEASURED: the harness's own window 0 sweep 0
    /// WindowAdvance output for U already diverged from the reference
    /// at substep 1, by roughly 4 in an O(1) quantity, traced directly
    /// to this read. `window_import`, by contrast, is read everywhere
    /// ELSE in this file for the ordinary, LAGGED per-sweep import
    /// (`previous[]`/`slots[]`, both read deliberately one sweep late
    /// for Jacobi/GaussSeidel), which this field must never replace:
    /// only the t=0 correction, which explicitly wants the LIVE value
    /// the instant it exists, uses it. Equal to `window_import` for an
    /// ordinary channel (no walk-back needed there, `import_view_arc`'s
    /// own initializer); invalid when this member imports nothing, the
    /// same as `window_import`.
    ArcId       window_import_delivery_arc;
    /// This member's own IMPORT's own PROVIDER's own `ComponentSet`
    /// slot (resolved the same way `component_slot` above is, from
    /// `graph.owner_of(ch.source)` -- graph-derived, never a partner's
    /// runtime state), or `components.size()` (an out-of-range
    /// sentinel) when this member has no import. Lets `resolve_
    /// pending_window_straddles()`, replaying PROVIDER P's own
    /// overshoot, find every OTHER member's config whose own import
    /// comes from P (`provider_component_slot == P's own
    /// component_slot`) and publish P's real post-replay value into
    /// THAT member's own `window_import` arc -- the one `advance_
    /// window()`'s own predictor actually reads -- rather than into
    /// P's own raw export arc, which the compiled Transport/Remap/Sink
    /// chain has not yet carried anywhere by the time the replay runs.
    std::size_t provider_component_slot = 0;
    std::size_t region_index = 0;  ///< which `regions[]` entry owns this node

    /// This member's own NATIVE (single-sample, per-element) Provides
    /// storage -- resolved directly from the `FieldRegistry`, per this
    /// struct's own top comment. `advance()` writes here every substep;
    /// `advance_window()` copies the result into the correct sample slot
    /// of the K-wide `ins.out` buffer right after. Null for a
    /// consumer-only member (owns no channel's source in this region).
    double*     provided = nullptr;
    /// This member's own native Requires storage, written by
    /// `advance_window()` (the substep's mean) right before each
    /// `advance()` call. Null for a pure exporter.
    double*     required = nullptr;
    /// Element count both pointers above share (`FieldRef::size()`).
    std::size_t length = 0;

    /// The IMPORT's own provider's interval/policy -- needed to interpret
    /// `window_import`'s flat sample array (its sample TIMES are the
    /// provider's own `plan_substeps()` grid, not this member's). Graph-
    /// derived, resolved once alongside everything else above; ignored
    /// when `window_import` is invalid.
    Rational    provider_interval{1};
    DtPolicy    provider_policy = DtPolicy::Adjustable;
    std::size_t provider_capacity = 1;
  };
  std::unordered_map<std::uint32_t, WindowMemberConfig> window_member_config;

  /// Unit 3 (S6 phase 3, Conformance): `Runner::field_window_view()`'s own
  /// lookup tables, keyed by FieldId (not by the WindowAdvance NodeId
  /// `window_member_config` uses) since the ABI caller names a field, not
  /// a node. Populated alongside `window_member_config` itself, in
  /// `resolve_regions()`, from the SAME `exported_field`/`imported_field`
  /// that function already resolves there -- not re-derived. `_arc` is
  /// the arc field_window_view() reads through `buffer()`'s own `slots`;
  /// a field present in neither map is not window-scoped at all (field_
  /// window_view()'s "window_samples = 0" fallback).
  std::unordered_map<std::uint32_t, ArcId> window_export_field_arc;
  std::unordered_map<std::uint32_t, ArcId> window_import_field_arc;

  /// Time windows track: one member's own subcycling state, keyed the
  /// same way as `window_member_config` (the `NodeId` of its own
  /// `OpKind::WindowAdvance` instruction). `fuse::detail::
  /// WindowMemberState` (fuse/runtime/window_coupler.hpp, Unit A)
  /// unchanged -- what `advance_window()` reads and updates every sweep
  /// for THIS member's own trajectory (clock, history, straddle
  /// bookkeeping). Never read by a partner: a partner's own sample TIMES
  /// come from `window_member_config` (the graph alone) and its own
  /// `dst_step`, never from this map -- there is no side channel between
  /// two members' own `WindowMemberState`.
  std::unordered_map<std::uint32_t, detail::WindowMemberState>
      window_member_state;
  /// The ABSOLUTE window instance index (`window_absolute_instance()`,
  /// not the period-relative `Instruction::dst_step` itself) THIS node's
  /// `advance_window()` last ran for, keyed the same way as
  /// `window_member_state`. Absent means "never run yet". A call whose
  /// absolute instance differs from this is the first sweep of a NEW
  /// window instance for this member -- see `advance_window()`'s own
  /// comment on what that triggers (a pending straddle replay, a fresh
  /// checkpoint, and a fresh substep-grid plan) before this sweep's own
  /// work.
  ///
  /// ABSOLUTE, NOT PERIOD-RELATIVE, because `dst_step` restarts at 0 on
  /// every `run_period()` call (the compiled `TimedDag` spans one period
  /// and is replayed verbatim): a window-scoped region with exactly one
  /// window instance per period (`period == window_length`, the common
  /// case) has `dst_step == 0` on EVERY call, which a `dst_step`-only
  /// comparison cannot tell apart from "still the same window instance
  /// as last time" on the second and subsequent `run_period()` calls --
  /// found by examples/brusselator's own Runner driver (`--driver
  /// runner`) needing more than one window, which no existing
  /// window_runner test (always a single `run_period()` call) had
  /// exercised. `window_absolute_instance()`'s own comment has the fix.
  std::unordered_map<std::uint32_t, std::int64_t> window_last_dst_step;
  /// Time windows track (C3): the clock `plan_substeps()` starts EVERY
  /// sweep of the CURRENT pass from -- fixed once, on that pass's own
  /// first sweep (`iteration == 0`), to either `T0` (no carried
  /// straddle) or wherever the previous window instance's own overshoot
  /// left `WindowMemberState::clock`. Read, never written, on every
  /// later sweep of the SAME pass: `run_region()`'s own generic rollback
  /// restores this member to the SAME window-start checkpoint before
  /// every sweep, so every sweep must replan from the SAME starting
  /// clock -- reading `WindowMemberState::clock` directly instead (as an
  /// earlier draft of `advance_window()` did) drifts forward every call,
  /// since that field is ALSO where this sweep's own `final_clock` gets
  /// recorded, and plans a zero-length (or negative) interval on every
  /// sweep after the first.
  std::unordered_map<std::uint32_t, Rational> window_pass_start_clock;

  /// Time windows track: this member's own PROVIDES value at the window's
  /// own start (T0), cached ONCE on the pass's own first sweep
  /// (`first_sweep_of_pass`), keyed the same way as
  /// `window_pass_start_clock`. Read, never written, on every later sweep
  /// of the SAME pass -- the EXACT SAME shape of bug `window_pass_start_
  /// clock`'s own comment documents for the clock, now for the exported
  /// value instead: `run_region()`'s own generic rollback restores this
  /// member's registered state to the window-start checkpoint before
  /// every sweep, but `Component::restore_state()` is contractually only
  /// required to restore what `save_state()` captured (the component's
  /// own hidden state) -- NOT to resynchronize a `Provides` field that is
  /// itself a FUNCTION of that hidden state (e.g. a remap of it), which
  /// stays at whatever the REJECTED sweep's own last `advance()` call
  /// left it holding. Reading `cfg.provided[e]` live at the top of every
  /// sweep -- as `advance_window()`'s own "Sample 0" write used to --
  /// therefore mislabels the PREVIOUS sweep's own end-of-window value as
  /// the window-start one from the second sweep on, exactly the defect
  /// `WindowCoupler::sweep_window()`'s own `start_value[i][e]` (window_
  /// coupler.cpp's own comment: "captured ONCE, right after save_state(),
  /// before any advance() this window can mutate `provided`") was fixed
  /// to remove on the driver side -- found here by a hexfloat trace
  /// against that same driver on the Brusselator's own parameters
  /// (max_window_balance_error 1.686e-03, exactly the pre-fix historical
  /// defect's own signature, docs/user/gray-scott.md's sibling account
  /// and docs/user/brusselator.md's own 2026-09-26 history).
  std::unordered_map<std::uint32_t, std::vector<double>> window_start_value;

  /// KNOWN_ISSUES.md D11 (closing): scratch for the fixed-chunk
  /// deterministic reductions (fuse/num/space_kernels.hpp's
  /// `deterministic_norm_squared`/`deterministic_norm_squared_diff`) that
  /// replace iteration.cpp's block residual/value norm loop. Sized ONCE in
  /// resolve_regions(), from the largest block any region has -- never
  /// per sweep, per iteration.cpp's own header comment (point 4: nothing in
  /// the sweep loop may allocate).
  Kokkos::View<double*, Kokkos::HostSpace> reduction_scratch;

  Impl(const CouplingGraph& g, const OperatorDAG& d, ExecutionPlan p,
       const MeshBackend& b)
      : graph(g), dag(d), plan(std::move(p)), backend(b) {}

  Result<void> setup(ComponentSet& components, FieldRegistry& registry);
  Result<void> resolve_buffers(FieldRegistry& registry);
  Result<void> resolve_instructions();
  /// Spec §6 mode 1. Splits one communicator per component out of
  /// `config.base_communicator` and builds one `CommGraph` per entry of
  /// `transport_plans`. A no-op when `base_communicator` is unset.
  ///
  /// COLLECTIVE over the base communicator, so every rank must call it, in the
  /// same order, with the same plan. The ExecutionPlan is identical on every
  /// rank (it is derived from the graph), but that alone does not get every
  /// rank HERE: resolve_buffers()/resolve_instructions() run first and fail
  /// rank-divergently on each rank's own FieldRegistry. What guarantees every
  /// live rank arrives is the MPI_Allreduce status-agreement checkpoint in
  /// setup(), immediately above the call.
  /// Works out, for each distinct pair of rank sets appearing on an `Epoch`,
  /// which base-communicator ranks take part, so a rank-agreement point
  /// reduces over exactly those and not over the whole job.
  ///
  /// NOT collective, and creates no MPI resource. It reads the graph, which is
  /// identical on every rank, so every rank derives the same lists. That
  /// matters: an earlier version split a communicator per pair and held it for
  /// the Runner's lifetime, which would have required a destructor that calls
  /// `MPI_Comm_free` -- a COLLECTIVE call, and therefore exactly the
  /// asymmetric-destruction hang attach_comm_graphs() is careful to avoid
  /// ("nothing outlives this function"). The host run that exposed it reported
  /// unfreed COMM handles at exit.
  Result<void> resolve_epoch_participants();

  /// Reduces this rank's status over the participants of `ins` and fails if any
  /// of them has failed.
  ///
  /// Without a base communicator there is one participant, so agreement is
  /// trivially satisfied and this is a no-op -- not a silent omission: with one
  /// rank there is nobody to disagree with.
  Result<void> reconcile_epoch(const Instruction& ins);

  /// Set when an instruction fails mid-period. The period loop keeps going
  /// from there but executes ONLY `Epoch` instructions, so the failing rank
  /// still reaches every agreement point its partners are waiting at. Without
  /// that, the failing rank returns immediately and its partners block in the
  /// reduction forever -- the hang the agreement points exist to prevent.
  std::optional<Failure> step_failure;

  /// One entry per distinct (source rank set, destination rank set) pair that
  /// appears on an `Epoch`: the base-communicator ranks that take part, sorted
  /// ascending. Sorted because the reduction schedule is derived from a rank's
  /// POSITION in this list, and every participant must derive the same one.
  ///
  /// Plain rank numbers, no MPI handle, nothing to release. The reduction
  /// (`reconcile_epoch`) is point-to-point on the base communicator with a
  /// dedicated tag, following the same discipline as the crystal router: a
  /// schedule that depends only on the participant list and this rank's place
  /// in it cannot diverge on data, and there is no resource whose lifetime
  /// could outlive the Runner.
  std::unordered_map<std::uint64_t, std::vector<int>> epoch_participants;

  Result<void> attach_comm_graphs();
  [[nodiscard]] Result<void> check_transport_extents() const;
  Result<void> resolve_constrain(const Instruction& ins, detail::Resolved& r);
  /// Spec §5.4 (Task 5.5). Gives one `Remap` instruction its operator: either
  /// real weights from `remap_cache`, or the identity -- but only after
  /// PROVING the two regions are the same cells in the same order. Refuses
  /// everything else, because the only remaining alternative is an identity
  /// copy between meshes that merely agree on a count, which is silently
  /// wrong rather than loudly wrong.
  Result<void> resolve_remap(const Instruction& ins, detail::Resolved& r);
  /// KNOWN_ISSUES.md D11 (closing). Builds the CSR conversion of `weights`'
  /// COO triplets -- see `detail::RemapCsr`'s own comment for the layout --
  /// called from `resolve_remap()` at the point it is about to push a new
  /// entry onto `remap_weights`, so the two vectors stay parallel. Defensive,
  /// not a re-derivation of `remap_cache`'s own correctness: refuses
  /// `E-REMAP-COO` if a triplet's row or column falls outside the matrix's
  /// declared shape, or if `RemapWeights::row` is not actually ascending --
  /// the same invariant `apply_weights` (fuse/backend/remap_plan.hpp) checks
  /// per call and this checks once, at the point the invariant is consumed
  /// to build a prefix scan instead of re-validated every period.
  Result<detail::RemapCsr> build_remap_csr(const RemapWeights& weights);
  /// Nodal-remap track, round 2 (KNOWN_ISSUES.md D8). Called from
  /// `resolve_remap()` in place of the flat `E-RT-REMAP-NODAL-PARALLEL`
  /// refusal, ONLY for a Remap whose target is a nodal region on more than
  /// one rank AND whose method is `"consistent"`/`"consistent-transpose"`.
  /// Resolves the nodal and evaluation `MeshView`s, decides `comm_handle`
  /// from `config.base_communicator` (`fuse::search::kNullComm` when unset),
  /// and calls `fuse::build_distributed_interp_rows` once -- the expensive
  /// part, done here at setup so `apply_nodal_dist_remap` below allocates
  /// nothing PER PERIOD beyond what `apply_distributed_H`/`HT` themselves
  /// need for their own (documented, per-call) Allgatherv buffers. Pushes
  /// one `detail::NodalDistRemap` and returns its index for
  /// `Resolved::nodal_dist`, or a `Failure` (any code
  /// `build_distributed_interp_rows`/`E-INTERP-DISC` documents).
  ///
  /// COLLECTIVE over `comm_handle` exactly when it names a real
  /// communicator (see `NodalDistRemap::comm_handle`'s own comment for why
  /// this is `config.base_communicator` UNCHANGED, not a per-component
  /// split): every rank of it must call this at the same position in the
  /// (identical) instruction stream, same discipline `attach_comm_graphs()`
  /// and the deleted `reduce_nodal_remap_target()` both used.
  Result<std::int32_t> build_nodal_dist_remap(const Instruction& ins,
                                              const std::string& label,
                                              RegionId src, RegionId tgt,
                                              const FieldSpec* sf_ptr,
                                              const FieldSpec* tf_ptr);
  /// Called from the `OpKind::Remap` case of `execute()` INSTEAD OF the
  /// normal weights-apply, for exactly the instructions
  /// `build_nodal_dist_remap()` above marked with `Resolved::nodal_dist >=
  /// 0`. Reads `in_ptr(0)`, calls `apply_distributed_H` or
  /// `apply_distributed_HT` (per `NodalDistRemap::transpose`) against the
  /// already-built rows, and writes the result into `out_ptr()`.
  ///
  /// COLLECTIVE over the stored `comm_handle`, same rule as the build
  /// function above -- every period, not just once.
  Result<void> apply_nodal_dist_remap(const Instruction& ins,
                                      const detail::Resolved& r);
  /// Spec §7.3 (Task 6.3). Builds one `SpacetimeInfo` per space-time
  /// `Interpolate` node and refuses every configuration the interpreter cannot
  /// execute correctly, rather than falling back to a lower-order basis.
  Result<void> resolve_spacetime();
  /// Checks that the instruction stream really does drive the operator built
  /// above: every source step sampled exactly once, every target step
  /// projected exactly once, and no row of `T` reaching past the last source
  /// step available when it is applied.
  Result<void> check_spacetime_schedule();
  Result<std::uint32_t> weights_for(RegionId region, int rank_set,
                                    std::size_t elements,
                                    int components);
  Result<void> build_ticks(ComponentSet& components);

  Result<void> execute(const Instruction& ins, const detail::Resolved& r);

  /// S7.1's Newton-Krylov seam. `f_out = x - G(x)` for region
  /// `region_index`, `G` one Picard sweep of that region's body delivered
  /// with `x` on the interface -- see fuse/runtime/fixed_point_residual.hpp
  /// and src/runtime/fixed_point_residual.cpp (this method's own
  /// definition) for the full save/restore contract and the single-
  /// participant, non-window, non-externally-driven scope it is
  /// restricted to.
  ///
  /// `commit`: S7.2's own addition. `false` (the default, and S7.1's own
  /// original meaning) restores every component and arc this call touched
  /// before returning, so repeated calls at arbitrary `x` are side-effect
  /// free -- what a Newton iteration's own Jacobian-action and line-search
  /// trial evaluations need. `true` skips that restore, leaving the
  /// delivered `x`/advanced state/executed body exactly as this call left
  /// them -- `run_region_newton()`'s own "verification sweep", the one
  /// evaluation whose point (the converged iterate) is meant to become the
  /// region's real, persisted state rather than be undone.
  Result<void> evaluate_region_residual(ComponentSet& components,
                                        std::size_t region_index,
                                        std::span<const double> x,
                                        std::span<double> f_out,
                                        bool commit = false);

  /// S7.2's own eager validation: the SAME three checks
  /// evaluate_region_residual() enforces lazily on its first call
  /// (non-window, single cross-rank participant, no externally driven
  /// member), run once at setup from resolve_regions() so a Newton-
  /// selected region this Runner cannot yet solve is refused with
  /// E-NEWTON-UNSUPPORTED-REGION before the first period runs rather than
  /// on it -- resolve_regions()'s own "fail fast" rule (this file's top
  /// comment, src/runtime/iteration.cpp) applied to this track's own new
  /// region shape. One function so the two call sites (here and
  /// evaluate_region_residual() itself) cannot drift apart on what
  /// "supported" means.
  Result<void> check_region_supports_newton(ComponentSet& components,
                                            std::size_t region_index) const;

  /// S7.2: the Newton-Krylov solve that replaces the Picard sweep loop
  /// entirely for a region with `RegionRuntime::newton` set -- called from
  /// run_region() before any of the Picard-only setup (R.accel->reset(),
  /// the window-scoped checkpoint, the x_old seed) runs, since none of it
  /// applies to this region. See its own definition in
  /// src/runtime/iteration.cpp for the full account: one
  /// fuse::num::solve() call over Runner::region_fixed_point_residual()'s
  /// `evaluate`, then one `commit = true` evaluation at the converged
  /// iterate (the "verification sweep") to make it the region's real,
  /// persisted state and to populate region_stats the same way a Picard
  /// sweep would.
  Result<void> run_region_newton(ComponentSet& components,
                                 detail::RegionRuntime& R);

  /// KNOWN_ISSUES.md D12. A coroutine (see src/runtime/resume.hpp):
  /// suspends via `co_await detail::ExternalSuspend{}`, reporting
  /// `pending_external_request`, when `tick`'s own component is
  /// `is_externally_driven()` -- BEFORE calling its `advance()` (a no-op
  /// for the one such component, `abi_detail::ExternalStepComponent`, so
  /// deferring the call past the suspend changes nothing observable) --
  /// and calls `advance()` directly, with no suspension at all, for every
  /// ordinary (in-process) component: the in-process statement order and
  /// timing are therefore identical to what this function did before it
  /// became a coroutine.
  detail::Task<Result<void>> advance(ComponentSet& components,
                                     const detail::Tick& tick, int iteration);
  detail::Task<Result<void>> step_period(ComponentSet& components);

  /// KNOWN_ISSUES.md D12's fix, driving `step_period()`'s coroutine from
  /// outside one call: creates a new `period_task` if the previous one
  /// ran to completion (or none has started yet), resumes it, and reports
  /// either the `ExternalAdvanceRequest` it suspended at
  /// (`pending_external_request`, consumed here) or `std::nullopt` once
  /// the period itself is over. See runner.hpp's own doc comment on
  /// `Runner::resume_period()`, which forwards straight to this.
  Result<std::optional<ExternalAdvanceRequest>> resume_period(
      ComponentSet& components);

  /// Defect 3's own refinement (Conformance Unit 2): every component the
  /// compiled stream visits at a period's own start (time 0), in the
  /// EXACT order it visits them -- in-process and externally driven
  /// alike, whichever this is called for picks out the one it cares
  /// about. A STATIC property of the plan: `ticks` and `plan.instructions()`
  /// are both built once, at `build_ticks()`/`schedule()` time, and a
  /// period's own start repeats identically every period (the schedule's
  /// own common-period invariant), so there is exactly one answer for the
  /// whole run, not one per period. Mirrors `step_period()`'s own merge
  /// walk (ticks interleaved with the instruction stream, windows walked
  /// by their first instance's own body) structurally, but calls nothing
  /// that executes anything -- no `advance()`, no coroutine, no
  /// `co_await`, no `resume_period()` -- so it carries NONE of
  /// `resume_period()`'s own side effects and is safe to call any number
  /// of times, from any state, including before a period has even
  /// started. See `Runner::components_due_at_period_start()` (runner.hpp)
  /// for why `fuse_is_coupling_ongoing()` needed this.
  [[nodiscard]] std::vector<ComponentId> components_due_at_period_start() const;

  /// Unit 3 (S6 phase 3, Conformance), the lead's own corrected ruling
  /// (2026-10-04, superseding an earlier same-day one): resolves what
  /// `next_component` must report when the cached request is a
  /// delivery proxy (`ExternalAdvanceRequest::is_delivery_due`,
  /// runner.hpp, has the full account of why the proxy's own importer
  /// name is never the right answer) -- the SAME kind of static plan
  /// scan `components_due_at_period_start()` performs for a period's
  /// own start, just starting from a given position in the stream
  /// instead of from its front. `instruction_index` is `ExternalAdvance
  /// Request::delivery_instruction_index`: the `ExternalDelivery`
  /// instruction's own index into `plan.instructions()`.
  ///
  /// Scans TWO sources forward from `instruction_index`, exactly the
  /// two `step_period()`'s own merge walk draws from: ordinary ticks
  /// (`ticks`, already filtered to this process's own hosted
  /// components by `resolve_regions()`'s own caller) with `time`
  /// strictly after this instruction's own `time`, and `WindowAdvance`
  /// instructions later in `plan.instructions()` -- for each, the
  /// OWNING component must be `is_externally_driven()` (read via
  /// `components`, the same way `deliver_external()` reads it) AND
  /// hosted on this process (the SAME `this_process`/`RankSet::
  /// contains()` formula `deliver_external()` and `build_ticks()`'s own
  /// `hosted[]` both use) -- a candidate failing either check is
  /// skipped, not returned, because the NEXT eligible one further out
  /// is still a candidate (unlike `components_due_at_period_start()`,
  /// which returns every period-0 component unfiltered and leaves the
  /// caller to pick the first eligible one: that works there because
  /// everything at time 0 is a genuine tie with no ordering between
  /// them, but a forward scan has a real order to respect, and
  /// stopping at the first in-process one would wrongly ignore the
  /// first EXTERNAL one one step later -- the reason this function
  /// filters internally instead). Whichever of the two sources' earliest
  /// eligible candidate has the smaller `time` wins; a genuine tie
  /// between an ordinary tick and a window boundary is not exercised by
  /// any graph this codebase tests today and is broken toward the
  /// window candidate, arbitrarily but deterministically.
  ///
  /// The `WindowAdvance` scan is bounded, not a plain walk to
  /// `plan.instructions().size()`: the lead's own ruling (2026-10-04),
  /// correcting the SAME day's first version of this function, once a
  /// scheduling-order fix made a delivery instruction's own position
  /// coincide with the END of a region's per-sweep instruction range.
  /// If `instruction_index` lies inside some `RegionRuntime`'s own
  /// `passes` entry (enter, exit), the forward scan stops at that
  /// `exit` and, finding nothing there, WRAPS to that SAME pass's own
  /// `enter` and continues up to (not including) `instruction_index` --
  /// the instruction sequence the sweep loop actually executes if it
  /// loops back for another sweep, never instructions past `exit` that
  /// belong only if the region does NOT loop back, a convergence
  /// decision this static scan cannot make. Only when
  /// `instruction_index` lies outside every region's every pass does
  /// the scan run the plain, unbounded walk to `plan.instructions().
  /// size()`. Returns `std::nullopt` if even that -- or the bounded
  /// scan plus its own wrap -- finds nothing: the caller wraps to
  /// `components_due_at_period_start()`'s own answer for the NEXT
  /// period, since the schedule repeats identically every period (this
  /// remains correct ONLY outside a region, where there is no sweep
  /// loop left to account for). The answer this function returns from
  /// inside a region is therefore conditional -- the component due IF
  /// the iteration continues -- which is exactly why
  /// `ExternalAdvanceRequest::is_delivery_due`'s own comment
  /// (`include/fuse/runtime/runner.hpp`) states it may disagree with
  /// what `fuse_begin_step()` actually does next: only resuming the
  /// Runner for real can resolve whether the region converges. Like
  /// `components_due_at_period_start()`, calls nothing that executes
  /// anything: a pure function of the compiled plan, `ticks`, `regions`,
  /// and the (already-wired, read-only) `components`, safe to call from
  /// any state.
  [[nodiscard]] std::optional<ComponentId> next_real_advance_after(
      const ComponentSet& components, std::size_t instruction_index) const;

  /// The request `advance()` most recently suspended with, valid only
  /// between a `period_task->resume()` that returned `false` and the
  /// `resume_period()` call that reads it out.
  std::optional<ExternalAdvanceRequest> pending_external_request;
  /// The in-flight `step_period()` coroutine, alive across every
  /// suspension until the period it is running completes. Recreated by
  /// `resume_period()` once the previous one is done.
  std::optional<detail::Task<Result<void>>> period_task;
  /// KNOWN_ISSUES.md D12. The handle `resume_period()` must call
  /// `.resume()` on NEXT -- see resume.hpp's own comment on
  /// `ExternalSuspend` for why this is never simply `period_task`'s own
  /// handle after the first suspension. Seeded from `period_task->
  /// raw_handle()` when a period starts; overwritten by `ExternalSuspend::
  /// await_suspend()` every time `advance()` or `advance_window()`
  /// actually suspends, at whatever depth that happens to be.
  std::coroutine_handle<> active_handle;
  /// Per-ComponentSet-slot bookkeeping for `ExternalAdvanceRequest::
  /// requires_checkpoint`/`requires_restore`: set the instant a region's
  /// `save_state()`/`restore_state()` runs for that slot (checkpoint_
  /// regions() and run_region()'s own window-scoped checkpoint, both
  /// runner.cpp/iteration.cpp; run_region()'s restore loop,
  /// iteration.cpp), consumed (read then cleared) the next time `advance()`
  /// above is about to suspend for that slot. Sized to `components.size()`
  /// in build_ticks(); both default to 0 (false) for a component never
  /// checkpointed by any region, which is the common case and matches
  /// today's ABI behaviour (`requires_checkpoint`/`requires_restore`
  /// always 0) for a graph with no iteration region at all.
  std::vector<char> slot_just_checkpointed;
  std::vector<char> slot_just_restored;

  // --- src/runtime/iteration.cpp (spec §5.3) -------------------------------
  /// Builds `regions` and refuses every plan whose shape the sweep loop below
  /// cannot execute correctly. Called from setup().
  /// Time windows track (C3): also resolves each window member's own
  /// native Provides/Requires field pointers directly from `registry`
  /// (`WindowMemberConfig::provided`/`required` below) -- bypassing the
  /// arc system entirely, since a window-scoped export/import has no
  /// ordinary Source/Sink arc for `resolve_buffers()` to have aliased
  /// them onto (`seed_window_source_arcs()` replaces the ordinary Source
  /// node; the K-wide `ins.out` arc is a fresh coupler-owned scratch
  /// buffer, never the component's own registered storage). Exactly the
  /// same registry `WindowMember::provided`/`required` (fuse/runtime/
  /// window_coupler.hpp) documents a caller must bind directly.
  Result<void> resolve_regions(ComponentSet& components,
                               FieldRegistry& registry);
  /// Refuses an instruction outside the region that reads a value the region
  /// re-computes on every sweep.
  Result<void> check_region_isolation(const detail::RegionRuntime& r,
                                      const CheckpointSet& cp);
  /// Checkpoints, at the top of a period, before anything advances. Time
  /// windows track: SKIPS a window-scoped region -- see
  /// `RegionRuntime::window_scoped`'s own comment for why that region's
  /// own checkpoint instead happens once per PASS, inside `run_region()`
  /// itself, not once per period here.
  Result<void> checkpoint_regions(ComponentSet& components);
  /// One PASS of one region: sweep until converged, exhausted, or failed.
  /// `enter`/`exit` are THIS pass's own bounds in the instruction stream
  /// (one entry of `r.passes`) -- explicit, rather than read from
  /// `r.enter`/`r.exit`, because a window-scoped region may run this
  /// several times in one period, once per window instance, each with
  /// different bounds; an ordinary region's caller passes `r.enter`/
  /// `r.exit` directly (its only pass). For a window-scoped region, also
  /// does that window instance's OWN checkpoint (save_state() + the byte-
  /// capture `checkpoint_regions()` does for an ordinary region at
  /// period-top) as the first act, before the sweep loop.
  /// KNOWN_ISSUES.md D12. A coroutine (src/runtime/resume.hpp): every
  /// re-advance this function makes (the `R.tick_indices` loop) goes
  /// through `advance()` above, `co_await`ed, so a region member that is
  /// `is_externally_driven()` suspends there exactly like an ordinary
  /// period-level tick does; every `OpKind::WindowAdvance` dispatch
  /// (`advance_window()` below) is likewise `co_await`ed.
  detail::Task<Result<void>> run_region(ComponentSet& components,
                                        detail::RegionRuntime& r,
                                        std::size_t enter, std::size_t exit);
  /// Mirrors `WindowCoupler::resolve_pending_straddles()`
  /// (window_coupler.cpp) on the Runner: for every window member of `r`
  /// whose `WindowMemberState::straddle_pending` is still set (a
  /// `DtPolicy::Fixed` member's last substep of the window that just
  /// finished crossed that window's own end), replays every substep
  /// since the member's own LAST checkpoint -- restore_state() then
  /// re-advance, reading the now-fully-converged data from the window
  /// that just finished instead of whatever a mid-sweep (possibly
  /// lagged) snapshot held when that substep first ran. Called from
  /// `run_region()` itself, BEFORE its own per-window checkpoint
  /// (`save_state()`): that checkpoint would otherwise overwrite the
  /// very state this replay needs to restore_state() back to (nothing
  /// else saves over it in between, exactly the single-checkpoint
  /// contract `resolve_pending_straddles()`'s own comment documents for
  /// the driver). A no-op for a region with no pending straddle.
  ///
  /// NOT a coroutine, deliberately (KNOWN_ISSUES.md D12's own fix
  /// statement only commits to step_period(), run_region() and the
  /// window-advance path): its replay calls `comp.advance(ctx)` directly,
  /// for a member whose PREVIOUS window instance straddled this one's
  /// boundary, with no `ComponentSet`-tick to suspend through cleanly --
  /// doing that correctly needs its own accounting of which replayed
  /// interval an external caller would be asked to redo, which is outside
  /// this round's scope. Refuses `E-RT-WINDOW-EXTERNAL-STRADDLE` up front
  /// if the member being replayed `is_externally_driven()`, rather than
  /// silently calling its (no-op) `advance()` out of the order an
  /// external caller would see through `fuse_begin_step()`/
  /// `fuse_end_step()`.
  Result<void> resolve_pending_window_straddles(ComponentSet& components,
                                                detail::RegionRuntime& r);
  /// Time windows track: executes exactly one `OpKind::WindowAdvance`
  /// instruction, for exactly one sweep of the region it belongs to.
  /// Called from `run_region()`'s own body loop INSTEAD OF `execute()`
  /// for this one instruction kind -- `execute()` has no `ComponentSet&`
  /// and cannot call `advance()`/`save_state()`/`restore_state()` itself
  /// (see its own `OpKind::WindowAdvance` case, src/runtime/runner.cpp).
  /// `iteration` is `run_region()`'s own sweep index `k`. Detects the
  /// first sweep of a window instance THIS node has not seen before by
  /// comparing `ins.dst_step` against `window_last_dst_step[node]`
  /// itself (no separate flag needed: within one pass, every sweep
  /// k=0,1,2,... calls this with the SAME `dst_step`, so the comparison
  /// is naturally true only once per pass) -- which is when a pending
  /// straddle from the PREVIOUS window instance must be resolved and this
  /// window's own checkpoint/substep grid planned, before any of this
  /// sweep's own work.
  ///
  /// STATUS: stub. Returns E-RT-WINDOW-UNWIRED, same as execute()'s own
  /// former WindowAdvance case (see that case's own comment, updated to
  /// point here) -- the real subcycling implementation is blocked on a
  /// design question flagged to the lead (Jacobi vs Gauss-Seidel exchange
  /// semantics between two window members within one sweep; see this
  /// pass's own host.submit_output).
  /// KNOWN_ISSUES.md D12. A coroutine (src/runtime/resume.hpp): its two
  /// direct `comp.advance(ctx)` calls each suspend first, via
  /// `co_await detail::ExternalSuspend{}`, when `comp.is_externally_
  /// driven()` -- the same rule `advance()` above applies to an ordinary
  /// tick.
  detail::Task<Result<void>> advance_window(ComponentSet& components,
                                            const Instruction& ins,
                                            const detail::Resolved& r,
                                            int iteration);
  /// Unit 3 (S6 phase 3, Conformance): executes exactly one
  /// `OpKind::ExternalDelivery` instruction, for exactly one sweep of the
  /// region it belongs to. Called from `run_region()`'s own body loop
  /// INSTEAD OF `execute()` for this one instruction kind -- mirrors
  /// `advance_window()` immediately above, for the same reason: this is
  /// the one place that both has `components` (to resolve the importing
  /// `ComponentId` to a real `Component&`) and is reachable every time an
  /// `ExternalDelivery` instruction appears inside a window-scoped
  /// region's body (which, today, every one does -- `lower_channel()`
  /// only emits `ExternalDelivery` for a channel whose source field sets
  /// `hdf5_path`, and every such field in this codebase is window-scoped;
  /// an ExternalDelivery reaching `step_period()`'s own top-level loop
  /// outside any region is not handled by this fix and falls through to
  /// `execute()`'s own pre-existing no-op case there, unchanged).
  ///
  /// The importing component's own `ComponentId` is read directly off
  /// `dag.arc(ins.out).type.rank_set` -- `lower_channel()`'s own LowFive
  /// substitution sets it there (`delivered.rank_set =
  /// g.owner_of(ch.target).value()`), so no new `OpAttrs` field is
  /// needed. First checked against hosting, the SAME formula
  /// `build_ticks()`'s own `hosted[]` and `resolve_regions()`'s own
  /// `WindowMemberConfig::hosted` both use (`this_process`): on a
  /// process that does not host the importer, this is a no-op, exactly
  /// as `advance_window()` is for a non-hosted member -- on U's own
  /// process, a channel whose importer is V suspends nothing and writes
  /// nothing, because V's own process handles its own end of that
  /// delivery independently. When the importer IS hosted here and
  /// `is_externally_driven()`, suspends via `co_await detail::
  /// ExternalSuspend{}` exactly like `advance()`/`advance_window()` do,
  /// with `pending_external_request.is_delivery_due = true` and
  /// `component` set to the importer's own name --
  /// `ExternalAdvanceRequest::is_delivery_due`'s own comment
  /// (`include/fuse/runtime/runner.hpp`) has the full ABI-side contract
  /// this sets up: the lead's own ruling (2026-10-04) is that
  /// `fuse_begin_step(C)`, WHATEVER `C` names, resumes the Runner past
  /// EVERY pending delivery proxy it encounters while resuming --
  /// whatever component each one names, not only ones matching `C` --
  /// until it reaches a real advance request, which is the only thing
  /// checked against `C`; `next_component` naming a proxy's own
  /// importer is purely informational, never a claim that importer
  /// must call next. An importer that is NOT externally driven
  /// (in-process) is refused earlier, at `fuse_initialize()`
  /// (`E-ABI-DELIVERY-INPROCESS`, src/abi/participant.cpp) --
  /// unreachable here by construction, so this function does not
  /// re-check it; a validation gap that let one through would silently
  /// become `E-RT-RESUME-INVARIANT` the first time `resume_period()`
  /// tried to resume past it, not a hang, since an in-process component
  /// never calls `fuse_begin_step()` to perform that resume at all.
  /// `instruction_index` is this instruction's own index into
  /// `plan.instructions()` -- `run_region()`'s own loop variable at the
  /// call site -- recorded on the pending request as `ExternalAdvance
  /// Request::delivery_instruction_index` so `next_real_advance_after()`
  /// knows where to start its own forward scan.
  detail::Task<Result<void>> deliver_external(ComponentSet& components,
                                              const Instruction& ins,
                                              std::size_t instruction_index);
  /// `period_index * (this compiled period / cfg.window_length) +
  /// dst_step` -- the window instance index counted from the START OF
  /// THE RUN, never reset between `run_period()` calls, unlike
  /// `Instruction::dst_step` itself (period-relative: the compiled
  /// `TimedDag` spans one period and is replayed verbatim by every call,
  /// so `dst_step` restarts at 0 every time). `window_last_dst_step`'s
  /// own declaration above has the full account of why a dst_step-only
  /// computation of a window's own [T0, T1) is wrong on the second and
  /// later `run_period()` call whenever a region has exactly one window
  /// instance per period -- this is the SAME fix, for every place that
  /// turns a window instance index into an absolute simulated time.
  /// `cfg.window_length` is guaranteed to divide `plan.period()` an
  /// exact integer number of times (`src/schedule/expand.cpp`'s own
  /// `tick_count`/E-SCHED-PERIOD refused the schedule otherwise), so the
  /// division below never has a remainder.
  [[nodiscard]] std::int64_t window_absolute_instance(
      const WindowMemberConfig& cfg, std::int64_t dst_step) const;
  /// R-10 (REVIEW.md): one sweep's cross-rank residual agreement, reached by
  /// every rank in `r.participants` on every sweep -- including one that has
  /// already locally recorded a step_failure, exactly like reconcile_epoch().
  /// `payload` is laid out `[status, (r_value, r_owned, v_value, v_owned) *
  /// r.blocks.size()]`; combined in place via fuse::num::combine() so every
  /// participant ends the call holding the same globally-agreed values.
  /// No-ops (`Result<void>::success()`, `payload` unchanged) when
  /// `r.participants` has at most one rank or this rank is not in it.
  /// Mark `origin` and everything downstream of it as not advancing for the
  /// rest of this period. An out-of-range origin blocks every slot.
  void block_advances_from(std::size_t origin);

  /// False once this period has a failure that `slot` is downstream of.
  [[nodiscard]] bool may_advance(std::size_t slot) const;

  Result<void> reduce_region_residuals(const detail::RegionRuntime& r,
                                       std::vector<double>& payload);

  /// Unit 2b (D19), step (c): gathers every participant's own
  /// `WindowBlockSample` array (one per `r.blocks`) for a window-scoped
  /// region's convergence agreement -- `decide_window_convergence()`'s
  /// own input. An all-to-all point-to-point exchange over `r.
  /// participants`, not a tree (unlike `reduce_region_residuals()`): the
  /// decision needs every participant's OWN contribution individually
  /// (to exclude a non-hosted rank's one block without discarding its
  /// other, legitimately-owned ones), not a single pre-combined value.
  /// `local_ok` is this rank's own `!step_failure.has_value()`, carried
  /// alongside the samples so a locally-failed rank's contribution is
  /// visible to every other participant in the SAME exchange, exactly as
  /// `reduce_region_residuals()`'s own `payload[0]` is. `all_ok_out` is
  /// true only when every gathered rank reports ok. No-ops (`gathered_
  /// out = {mine}`, `all_ok_out = local_ok`) when `r.participants.size()
  /// <= 1` -- the single-process case, nothing to gather.
  Result<void> gather_window_block_samples(
      const detail::RegionRuntime& r, bool local_ok,
      const std::vector<WindowBlockSample>& mine,
      std::vector<std::vector<WindowBlockSample>>& gathered_out,
      bool& all_ok_out);
};

}  // namespace fuse
