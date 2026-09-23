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

#include "fuse/ir/lowering.hpp"
#include "fuse/num/accelerator.hpp"
#include "fuse/model/iteration.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/backend/spacetime.hpp"
#include "fuse/runtime/runner.hpp"

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
  std::size_t enter = 0;   // first instruction of one pass over the body
  std::size_t exit = 0;    // last, inclusive
  int         max_iterations = 1;

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

  bool initialized = false;

  std::vector<std::vector<double>> storage;       // coupler-owned, one per arc
  std::vector<detail::Slot>        slots;         // parallel to arcs
  std::vector<std::vector<double>> previous;      // previous iterate, for Relax
  std::vector<std::size_t>         sample_count;  // per arc, for Accumulate
  std::vector<detail::Resolved>    resolved;      // parallel to instructions
  std::vector<detail::Tick>        ticks;         // one period's worth, sorted
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
  Result<void> resolve_constrain(const Instruction& ins, detail::Resolved& r);
  /// Spec §5.4 (Task 5.5). Gives one `Remap` instruction its operator: either
  /// real weights from `remap_cache`, or the identity -- but only after
  /// PROVING the two regions are the same cells in the same order. Refuses
  /// everything else, because the only remaining alternative is an identity
  /// copy between meshes that merely agree on a count, which is silently
  /// wrong rather than loudly wrong.
  Result<void> resolve_remap(const Instruction& ins, detail::Resolved& r);
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
  Result<void> advance(ComponentSet& components, const detail::Tick& tick,
                       int iteration);
  Result<void> step_period(ComponentSet& components);

  // --- src/runtime/iteration.cpp (spec §5.3) -------------------------------
  /// Builds `regions` and refuses every plan whose shape the sweep loop below
  /// cannot execute correctly. Called from setup().
  Result<void> resolve_regions(ComponentSet& components);
  /// Refuses an instruction outside the region that reads a value the region
  /// re-computes on every sweep.
  Result<void> check_region_isolation(const detail::RegionRuntime& r,
                                      const CheckpointSet& cp);
  /// Checkpoints, at the top of a period, before anything advances.
  Result<void> checkpoint_regions(ComponentSet& components);
  /// One region: sweep until converged, exhausted, or failed.
  Result<void> run_region(ComponentSet& components, detail::RegionRuntime& r);
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
