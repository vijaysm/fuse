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
};

}  // namespace fuse
