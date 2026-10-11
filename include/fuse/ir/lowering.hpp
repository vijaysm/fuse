#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/connect.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/ir/dag.hpp"

namespace fuse {

// Only ever named by reference below, so a declaration is enough. Including
// fuse/runtime/component.hpp and fuse/model/iteration.hpp here would drag the
// field registry and the convergence measures into every translation unit
// that lowers a graph; src/ir/checkpoint.cpp includes them instead.
class ComponentSet;
class ConvergenceCriterion;

/// Threaded through every lowering rule.
///
/// A field has up to TWO arcs, and conflating them is a correctness bug, not a
/// tidiness one. An `Intent::Bidirectional` field is simultaneously the value
/// its own component *produces* and the value a channel *delivers* to it, and
/// those are different values at different points in the schedule. The two
/// maps below keep them apart. The invariant, in full:
///
///   `source_arc[f]` -- the arc carrying what f's OWN COMPONENT produces, i.e.
///       the output of f's `Source` node. Populated in `lower()` step 1 for
///       every `Provides` and `Bidirectional` field and NEVER REBOUND
///       afterwards. This is the only thing a channel *sourced from* f may
///       read. Absent for a pure `Requires` field, which produces nothing.
///
///   `field_arc[f]` -- the arc carrying the value most recently DELIVERED TO f
///       by the lowering of a channel, which is what f's `Sink` consumes and
///       what the derived-operator pass (`insert_derived_ops`) chains through.
///       Rebound by every channel that targets f. Seeded in step 1 from
///       `source_arc[f]` for `Provides`/`Bidirectional` fields so that a
///       `Bidirectional` field no channel writes still lowers to the
///       degenerate `Source -> Sink` self-copy that `prove_acyclic` and
///       `expand()` are written against.
///
/// Why this matters: reading `field_arc[f]` on the source side made a channel
/// out of a `Bidirectional` field forward whatever the *incoming* channel had
/// just written to it, including that channel's lag offset. That accepted real
/// algebraic loops (the component's own `Source` was orphaned, so the loop
/// through the component never closed) and rejected legal lagged schemes (a
/// lead into f and a lag out of f cancelled along the spliced chain). Anything
/// that consumes a field on the producing side must use `source_arc`.
struct LoweringContext {
  const CouplingGraph&                      graph;
  OperatorDAG&                              dag;
  Diagnostics&                              diags;
  std::unordered_map<std::uint32_t, ArcId>  field_arc;
  std::unordered_map<std::uint32_t, ArcId>  source_arc;

  /// Time windows track: `sink_arc[f]` is field `f`'s own `Sink` node's
  /// output arc, populated by `lower()`'s step 3 (one `Sink` per
  /// `Requires`/`Bidirectional` field that has a writer) as each one is
  /// created. Unlike `field_arc[f]` (the value delivered TO the Sink),
  /// this is the Sink's own OUTPUT -- what `finish_window_regions()`
  /// (src/ir/window_lowering.cpp) needs for an importing member's
  /// `OpAttrs::window_import`, which must name the arc the field transfer
  /// DELIVERS on the importer's own mesh and ranks, never the exporter's
  /// raw output directly (that would be a real edge into a node with
  /// empty inputs on the WRONG side of the transfer, and on the wrong
  /// placement besides).
  std::unordered_map<std::uint32_t, ArcId>  sink_arc;

  /// Unit 3 (S6 phase 3, Conformance): the arc THIS channel's own chain
  /// actually begins at, keyed by `ChannelId::value()` -- `source_arc[f]`
  /// for an ordinary channel, but the channel's own `ExternalDelivery` arc
  /// for a LowFive-bound one, since `lower_channel()` gives that node zero
  /// inputs and never reads `source_arc[f]` at all (see its own comment on
  /// why: the value lands on the consumer's placement, not the producer's).
  /// Populated by `lower_channel()` (`src/ir/rules_channel.cpp`) once `cur`
  /// is settled, immediately after the LowFive substitution (or its
  /// absence); read by `insert_derived_ops()`'s `channel_chain()`
  /// (`src/ir/rules_derived.cpp`), which used to start from `source_arc[f]`
  /// unconditionally and so silently found no nodes of this channel's own
  /// origin downstream of a severed `ExternalDelivery` root -- `has_remap`
  /// came back false and both `Constrain` and `Diagnose` went unemitted for
  /// every LowFive-bound channel, contradicting docs/user/lowfive.md's own
  /// stated design. Keyed by channel, not field, because a field can be
  /// read by more than one channel and each has its own entry point.
  std::unordered_map<std::uint32_t, ArcId>  chain_entry_arc;

  /// Every arc that writes a field, in channel declaration order. More than
  /// one is legal exactly when the graph declares how to combine them; the
  /// combine node is emitted once all channels have lowered.
  std::unordered_map<std::uint32_t, std::vector<ArcId>> field_writers;

  /// The same policy the validator used. It must be the same object, or the
  /// compiler can emit a unit conversion the checker forbade (or omit one it
  /// authorized) and nothing would notice: validation and lowering would each
  /// be self-consistent about different rules.
  ConnectPolicy                             connect_policy{};
};

/// Returns true if the rule claimed the channel.
using LoweringRule = std::function<bool(LoweringContext&, ChannelId)>;

/// Spec §5.2: a registry, not a switch. New numerics register a rule; the
/// core is untouched.
class LoweringRegistry {
 public:
  void add(std::string name, int priority, LoweringRule rule) {
    rules_.push_back(Entry{std::move(name), priority, std::move(rule),
                           next_seq_++});
    std::stable_sort(rules_.begin(), rules_.end(),
                     [](const Entry& a, const Entry& b) {
                       if (a.priority != b.priority) return a.priority > b.priority;
                       return a.seq < b.seq;
                     });
  }

  [[nodiscard]] bool apply_first_match(LoweringContext& ctx, ChannelId c) const {
    // Snapshot the callables before invoking any of them: a rule's callback
    // may itself call add() on this same registry (reentrant registration),
    // which push_back()s and stable_sort()s rules_ and can reallocate it.
    // Range-iterating rules_ directly while a callback can reallocate it
    // underneath the loop is a use-after-free -- ASan-confirmed (see R8 in
    // docs/HANDOFF.md, and the ReentrantAddDuringApplyFirstMatchIsSafe
    // regression test in test/unit/lowering_test.cpp). Unreachable today,
    // since every builtin rule registers once before any dispatch runs, but a
    // real hazard for any future plugin-style external registration.
    // Iterating a local copy means a mid-call add() is only visible starting
    // with the NEXT call to apply_first_match, never the one in progress --
    // sufficient here since this only needs to survive reentrant addition,
    // not removal.
    std::vector<LoweringRule> snapshot;
    snapshot.reserve(rules_.size());
    for (const Entry& e : rules_) snapshot.push_back(e.rule);
    for (const LoweringRule& rule : snapshot) {
      if (rule(ctx, c)) return true;
    }
    return false;
  }

  [[nodiscard]] std::size_t size() const noexcept { return rules_.size(); }

  /// The registry the driver uses. Populated on first access.
  static LoweringRegistry& builtin();

 private:
  struct Entry {
    std::string  name;
    int          priority;
    LoweringRule rule;
    int          seq;
  };
  std::vector<Entry> rules_;
  int                next_seq_ = 0;
};

/// Whether this compiler has an operator for a step the connection check can
/// authorize.
///
/// It is a function rather than a `default:` in the emission switch so that it
/// can be enumerated in a test: every value of `OperatorKind` is either emitted
/// or refused, and adding a new one without deciding which is a test failure
/// rather than a silent omission. The checker is allowed to be ahead of the
/// compiler -- it is describing what the physics needs -- but the gap must be
/// visible.
[[nodiscard]] bool ir_can_emit(OperatorKind kind) noexcept;

struct LoweringResult {
  OperatorDAG dag;
  Diagnostics diags;
  [[nodiscard]] bool ok() const { return !diags.has_errors(); }
};

/// Spec §2: CouplingGraph -> OperatorDAG.
/// `policy` must be the one the validator used: the compiler emits the
/// conversions the connection check authorized, so a different policy here
/// would silently compile a plan the checker never approved.
[[nodiscard]] LoweringResult lower(const CouplingGraph& g, ConnectPolicy policy = {});

/// Spec §5.2: inserts the Constrain (from conservation) and Diagnose
/// (always) operators the user did not write, immediately after this
/// channel's Remap. (Transport, the third derived operator, is placed by the
/// channel rule itself in rules_channel.cpp, since v1 hardcodes it
/// immediately before Remap — see that file's comment.) Called from inside
/// lower_channel, between the Remap and Relax stages: the DAG is
/// append-only, so Constrain/Diagnose must be created before any
/// user-authored Relax node that should consume their output.
void insert_derived_ops(LoweringContext& ctx, ChannelId cid);

/// Spec §5.3: turns graph-level IterationRegionDecls into DAG regions with
/// residual arcs, and fills `IterationRegion::carried` from
/// derive_checkpoint_set() below (Phase 7 Task 7.2).
void build_iteration_regions(LoweringContext& ctx);

/// Time windows track: every field a window-scoped iteration region's
/// channels source FROM a member component (`fuse::CouplingWindowScope`,
/// `fuse/model/iteration.hpp`) -- the field an exporting member's own
/// `OpKind::WindowAdvance` node (`finish_window_regions()` below) produces
/// instead. `lower()`'s step 1 must not build an ordinary `Source` node
/// for these; `seed_window_source_arcs()` is what does, in their place.
[[nodiscard]] std::unordered_set<std::uint32_t> window_provided_fields(
    const CouplingGraph& g);

/// Time windows track: every field a window-scoped iteration region's
/// channels deliver TO a member component -- the field whose `Sink` must
/// exist BEFORE `finish_window_regions()` runs, because that member's own
/// `OpKind::WindowAdvance` node's `OpAttrs::window_import` names this
/// Sink's output arc (`ctx.sink_arc`). `lower()`'s step 3 creates this
/// subset of Sinks FIRST, ahead of `build_iteration_regions()`, and every
/// other required field's Sink afterwards, in its ordinary position --
/// so a graph with no window-scoped region (for which this is always
/// empty) sees no change in node creation order at all.
[[nodiscard]] std::unordered_set<std::uint32_t> window_imported_fields(
    const CouplingGraph& g);

/// Time windows track (C3): the `components` count field `f`'s own `Sink`
/// must be built with, when `f` is a window-scoped import (i.e.
/// `window_imported_fields(g)` names it) -- exactly the SAME width its
/// PROVIDER's own exported field carries (the identical K-sample flat
/// array, delivered unchanged; see `window_source_type()`'s own widening
/// in window_lowering.cpp), never the field's own declared
/// `FieldSpec::components` alone. Falls back to `g.field(f).components`
/// (today's behaviour) for a field that is not actually a window import
/// -- callers only use this after checking `window_imported_fields()`.
[[nodiscard]] std::size_t window_import_width(const CouplingGraph& g,
                                              FieldId f);

/// Time windows track: pre-creates the bare output ARC (no producing node
/// yet) for every window-scoped region member that owns a channel's
/// source field, and seeds `ctx.source_arc`/`ctx.field_arc` with it --
/// exactly what `lower()`'s ordinary step 1 (`Source` per provided field)
/// would have done, had the field's value come from a `Source` rather
/// than from this member's own `OpKind::WindowAdvance` node.
///
/// Called from `lower()` right after step 1 and BEFORE step 2 (the
/// ordinary per-channel lowering rule, run UNCHANGED for a window-scoped
/// region's own channels -- the user's chosen design: waveforms travel
/// through the ordinary Remap/Transport/Epoch operations, not around
/// them): that rule reads `ctx.source_arc[[the channel's source field]]`
/// for ANY channel, window-scoped or not, and this is what gives a
/// window-scoped one something real to read.
///
/// Only the ARC is created here, not the WindowAdvance NODE itself:
/// `OperatorDAG` is append-only with no node-mutation API, and that
/// node's OWN `OpAttrs::window_import` cannot be known until the
/// PARTNER channel's `Sink` exists, which is not until `lower()`'s step 3
/// has run -- see `finish_window_regions()`.
void seed_window_source_arcs(LoweringContext& ctx);

/// Time windows track: creates the actual `OpKind::WindowAdvance` node for
/// every member of every window-scoped iteration region, in
/// `members_in_first_seen_order()`'s (a duplicate of `fuse::
/// window_region_members()`, `fuse/schedule/window.hpp` -- `ir` cannot
/// depend on `schedule`, docs/dev/layering.md) order, against the output
/// arc `seed_window_source_arcs()` already created for it.
///
/// Each node has EMPTY `inputs`, like `Source`: this member's own
/// exported field transfer reads this node's output as a REAL edge
/// (`ctx.source_arc`), which is fine -- the node itself has nothing
/// upstream to close a cycle with -- but the reverse, this node reading
/// its PARTNER's output as a real edge, would (measured directly, an
/// earlier pass of this track: schedule() refused with only 2 of 12 nodes
/// topologically orderable). `OpAttrs::window_import` instead names the
/// arc the field transfer INTO this member actually DELIVERS -- that
/// channel's own `Sink` output (`ctx.sink_arc`, populated by `lower()`'s
/// step 3, which must therefore run before this function), on this
/// member's own mesh and ranks, never the partner's raw output directly:
/// metadata only, ordered by TIME (`TickPhase`/`expand()`'s window-
/// instance ticking) rather than an SSA edge, and proved correct by
/// `schedule()`'s own `detail::check_window_causality()`
/// (`fuse/schedule/plan.hpp`), not by this function. A member that owns
/// no channel's source in this region (a consumer-only role) gets no node
/// of its own here; nothing else needs one, since nothing reads such a
/// member's "state" through this mechanism at all.
///
/// Called from `lower()` after step 3 (`Sink` per required field) and
/// before `build_iteration_regions()`, which groups these nodes into the
/// region's body by their `origin` channel exactly as it already does for
/// ordinarily-lowered nodes.
void finish_window_regions(LoweringContext& ctx);

// ---------------------------------------------------------------------------
// Spec §5.3: the derived checkpoint set (Task 7.2).
//
// FUSE derives the checkpoint set rather than asking the user. preCICE pushes
// this onto every adapter via requiresWritingCheckpoint(); here a dataflow
// analysis over the SSA form computes it and the runtime then verifies that
// the component actually saved what the analysis said it must.
//
// THE DIRECTION OF ERROR, which decides every judgement call below.
// Checkpointing too much costs memory and a copy. Checkpointing too little is
// silent wrong physics: a re-swept iterate starts from state the previous
// sweep mutated, so the iteration converges to the fixed point of the wrong
// operator and nothing in the residual history says so. Every rule therefore
// errs towards saving more, and each place it does is marked in
// src/ir/checkpoint.cpp.
// ---------------------------------------------------------------------------

/// What must survive a rejected iterate.
///
/// `coupler_arcs` are the SSA values FUSE itself owns whose definition at
/// sweep k reads sweep k-1, so their buffers are real state rather than
/// recomputable transients. `components` are the participants whose internal
/// state the region re-advances, so they must implement
/// save_state()/restore_state(). `explanation` says why, in the terms the user
/// authored the graph in.
struct CheckpointSet {
  std::vector<ArcId>       coupler_arcs;  // loop-carried coupler buffers
  std::vector<ComponentId> components;    // components whose state must persist
  std::string              explanation;
};

/// The dataflow analysis itself. Deterministic: both vectors come back sorted
/// and deduplicated, so two calls on the same DAG are byte-identical and the
/// set can be diffed between runs.
[[nodiscard]] CheckpointSet derive_checkpoint_set(
    const CouplingGraph& g, const OperatorDAG& dag,
    const OperatorDAG::IterationRegion& region);

/// The verification half of the claim: the analysis says which components must
/// be restorable, and this refuses a run in which one of them is not.
/// `E-ITER-NOROLLBACK` for a component that reports
/// `supports_rollback() == false`, and equally for one the analysis named that
/// is not registered at all — an unverifiable claim is refused, not assumed.
/// Also `E-ITER-NONDETERMINISTIC` (KNOWN_ISSUES.md B3) for one whose
/// `determinism_level() == DeterminismLevel::Nondeterministic`: the sweep
/// loop's convergence test compares successive iterates, and noise in a
/// component is indistinguishable from a residual.
[[nodiscard]] Diagnostics verify_rollback_support(const CouplingGraph& g,
                                                  const CheckpointSet& set,
                                                  const ComponentSet& components);

/// S7.1's Newton-Krylov seam. Over the SAME checkpoint set
/// `verify_rollback_support()` checks -- the Newton coupler re-advances
/// these components' own residuals exactly where Picard would re-advance
/// their state, so the same universe of components applies -- refuses a
/// member that cannot supply what a Newton-solved region needs from it.
///
/// `E-NEWTON-CAPABILITY`, naming the component, when `requires_residual`
/// is true and `Component::capabilities()` does not set
/// `kCapabilityResidual` (fuse::ImplicitComponent is not implemented; see
/// fuse/runtime/component.hpp), or when `requires_linearization` is true
/// and `kCapabilityLinearization` is unset. Both flags false (no Newton
/// solver selected for this region, `IterationRegionDecl`'s own default)
/// makes every call a no-op.
[[nodiscard]] Diagnostics verify_newton_capability_support(
    const CouplingGraph& g, const CheckpointSet& set,
    const ComponentSet& components, bool requires_residual,
    bool requires_linearization);

/// Everything about an iteration region's DECLARATION that can be judged from
/// the graph alone -- no DAG, no `ComponentSet`, no mesh backend:
///
///   * `E-ITER-NOCONVERGE` when `convergence.empty()`. An empty criterion
///     reports satisfied (true is the identity of AND), so the region would
///     stop after one sweep and report that it converged.
///   * `E-ITER-MAXITER` when `max_iterations < 1`, which would run the body
///     zero times.
///
/// Split out of `verify_iteration_region()` below because the C ABI needs
/// exactly this subset: `soma_initialize()` (src/capi/soma_c.cpp) has no
/// `ComponentSet`, so it cannot ask about rollback support, and no `Runner`, so
/// it never reaches `verify_iteration_region()` at all -- yet both states above
/// are exactly as wrong there as they are on the Runner path. Shared rather
/// than duplicated so the two cannot drift apart, and so the diagnostic a C or
/// Fortran caller reads is the same text a C++ caller reads.
[[nodiscard]] Diagnostics verify_iteration_declaration(
    const std::string& region_name, int max_iterations,
    const ConvergenceCriterion& convergence);

/// The names one residual of an iteration region answers to: its channel, and
/// the two fields that channel connects. Spec §5.3 lets a convergence measure
/// name any of the three.
struct MeasureTarget {
  std::string channel;
  std::string source_field;
  std::string target_field;
};

/// Match every measure of `convergence` against `targets`, returning, per
/// measure, the index of the single target it names.
///
/// TWO RULES, both `E-ITER-MEASURE`, and both about a stopping rule that is
/// not the coupling condition:
///   1. A measure must name EXACTLY ONE target. Zero means it is never tested
///      against anything; more than one means it is ambiguous which residual
///      it governs.
///   2. Every target must be named by SOME measure. A region whose criterion
///      covers only part of its coupling can stop with the rest far from its
///      fixed point -- measured, not argued: with one of two channels
///      measured, IQN-ILS drove the measured residual to exactly zero on sweep
///      3 while the unmeasured one stood at 0.8, and the region reported
///      convergence at a 10% error with a monotone residual history.
///
/// Shared by `Runner::Impl::resolve_regions()` (src/runtime/iteration.cpp),
/// whose targets come from the DAG's residual blocks, and by
/// `soma_initialize()` (src/capi/soma_c.cpp), whose targets come from the
/// graph's declared channel list.
///
/// WHY THOSE TWO LISTS AGREE, stated precisely because the obvious reason is
/// not the operative one. `build_iteration_regions()` (src/ir/regions.cpp)
/// loops over the region's declared channels and emits one `Residual` per
/// channel -- but it `continue`s past any channel with no `ctx.field_arc`
/// entry, which would silently make the DAG list shorter than the declared
/// one. That branch is dead for any graph that lowers: `lower_channel()`
/// (src/ir/rules_channel.cpp) binds `ctx.field_arc[ch.target]` unconditionally
/// for every channel it lowers, once before `insert_derived_ops()` and again
/// as its last act. A channel that never lowered would have failed the
/// lowering, which both call sites run first. So the correspondence rests on
/// that unconditional bind, not on the loop shape.
[[nodiscard]] Result<std::vector<std::size_t>> match_convergence_measures(
    const std::string& region_name, const ConvergenceCriterion& convergence,
    const std::vector<MeasureTarget>& targets);

/// Everything that must hold before a region may be iterated. Call this at
/// initialize; it is the single entry point so that rollback support cannot be
/// checked without also checking the convergence measure.
///
/// `E-ITER-NOCONVERGE` when `convergence.empty()`. An empty
/// `ConvergenceCriterion` reports satisfied == true (true is the identity of
/// AND, see num/convergence.hpp), so a region carrying one would stop after a
/// single sweep: an explicit scheme masquerading as a converged implicit one,
/// with a convergence report that says it converged. Also `E-ITER-MAXITER`
/// when `max_iterations < 1`, which would run the body zero times.
/// Then everything verify_rollback_support() checks, and -- when
/// `requires_residual_capability`/`requires_linearization_capability` is
/// true, S7.1's own addition -- everything verify_newton_capability_
/// support() checks. Both default to `false` (no Newton solver selected
/// for this region), so every pre-S7.1 call site is unchanged.
[[nodiscard]] Diagnostics verify_iteration_region(
    const CouplingGraph& g, const OperatorDAG& dag,
    const OperatorDAG::IterationRegion& region,
    const ConvergenceCriterion& convergence, const ComponentSet& components,
    bool requires_residual_capability = false,
    bool requires_linearization_capability = false);

/// Defined across src/ir/rules_channel.cpp and src/ir/rules_derived.cpp.
void register_builtin_rules(LoweringRegistry& reg);

}  // namespace fuse
