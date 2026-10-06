#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/schedule/plan.hpp"
#include "fuse/transport/transport.hpp"

namespace fuse {

struct RunnerConfig {
  std::string transport = "copy";
  bool        check_conservation = true;
  double      conservation_tolerance = 1e-12;
  bool        abort_on_violation = false;
  /// Spec §5.3. An iteration region that reaches `max_iterations` without
  /// meeting its convergence criterion has NOT converged, and the values it
  /// leaves behind are the last iterate, not a converged solution.
  ///
  /// `false` (the default) counts the window in
  /// `RunStats::windows_not_converged` and continues, which is what a
  /// production run that must not lose a job to one stiff window wants; the
  /// counter is the only record, so a run that ignores it is reporting an
  /// unconverged answer as a converged one. `true` fails the period with
  /// `E-RT-NOCONVERGE`.
  bool        abort_on_nonconvergence = false;

  /// Take the distributed nodal-remap path (`build_nodal_dist_remap`,
  /// `apply_distributed_H` / `_HT`) even when both placements sit on one rank,
  /// where the Runner would otherwise use the rank-local weights. The
  /// distributed operator is a correct remap at any participant count,
  /// including one: with no communicator it runs as a single participant.
  /// This exists so the serial tier can exercise the code the MPI fixtures
  /// run -- a one-rank failure of that path is otherwise debuggable only
  /// under a launcher. Not a performance setting; a production run leaves it
  /// false.
  bool        force_distributed_nodal_remap = false;

  /// `MPI_Comm_c2f()` of a communicator whose group contains every rank of
  /// every component in the graph -- `MPI_COMM_WORLD` under spec §6 mode 1,
  /// which is the only integration mode v1 implements. Set it to make
  /// `Transport` actually redistribute; leave it unset and `Transport` moves
  /// whole arcs, which is what `copy` does and what every existing caller
  /// relies on.
  ///
  /// WHY A BASE COMMUNICATOR AND NOT A PER-COMPONENT CALLBACK.
  /// `ComponentDecl::ranks` (include/fuse/model/component.hpp:13) already IS
  /// the graph's statement of who belongs to a component, and
  /// src/ir/rules_channel.cpp already asks it whether a Transport is needed at
  /// all. Splitting each component's communicator out of this one, by that
  /// same field, keeps the lowering and the runtime reading ONE source of
  /// truth. A `std::function<int(ComponentId)>` would be a second one, free to
  /// disagree with the first, and would add C ABI surface -- whereas
  /// `soma_set_communicator` (src/capi/soma_c.cpp:659-667) already stores
  /// exactly this handle and need only forward it.
  ///
  /// LIMITATION, WRITTEN DOWN RATHER THAN DISCOVERED. `RankSet` is a
  /// contiguous range, so the graph cannot currently express arbitrary partial
  /// overlap. `CommGraph` does not care -- it takes a communicator per side
  /// precisely so that every overlap shape is uniform data. When the graph
  /// gains non-contiguous rank sets, only the `MPI_Comm_split` colour changes.
  ///
  /// std::optional because a Fortran communicator handle has no portable
  /// "unset" value to use as a sentinel: 0 is a legal handle.
  std::optional<int> base_communicator;

  /// Unit 2b (D19): overrides `build_ticks()`'s own `this_process`
  /// computation directly, bypassing `base_communicator`/`MPI_Comm_rank`
  /// entirely -- the SAME reason `force_distributed_nodal_remap` above
  /// exists: a tier-0 build (no MPI compiled in at all) can exercise the
  /// per-rank HOSTING decision (ordinary ticks' own `hosted[]`, and
  /// Unit 2b's own `WindowMemberConfig::hosted`) with a specific rank
  /// number, with no real communicator, no launcher, and no `#if
  /// FUSE_HAVE_MPI` branch ever taken. `base_communicator` need not be
  /// set at all when this is -- a test that wants both a pretend "rank 1
  /// of 3" placement AND genuine ordinary-tick filtering sets only this
  /// one field. Left unset (the default) for every production caller,
  /// which continues to derive `this_process` from `base_communicator`
  /// exactly as before this field existed.
  std::optional<int> force_this_process;

  /// Task 8.5's gap-closer. `Transport`'s virtual interface (prepare/move/
  /// name/is_local) has no hook for a backend's one-time, backend-specific
  /// setup -- `MpiTransport::set_communicator`+`build_plan` or
  /// `LowFiveTransport::set_location`+`build_plan` -- because that setup takes
  /// a concrete type (a Fortran communicator handle, a file path), not
  /// anything `Transport&` can express. Without this, `RunnerConfig{.transport
  /// = "lowfive"}` (or "mpi") builds an unconfigured transport that fails at
  /// the first Transport instruction.
  ///
  /// If set, `Runner::initialize()` calls it exactly once, immediately after
  /// the registry constructs the named transport and before any instruction
  /// is resolved, with a mutable reference to that transport. A caller that
  /// needs the concrete type `dynamic_cast`s to it. Left empty (the default)
  /// for `"copy"` and any transport that needs no such setup; empty is a
  /// silent no-op, not an error, so `CopyTransport` and every existing
  /// `RunnerConfig{}` user is unaffected.
  std::function<Result<void>(Transport&)> transport_setup;
};

struct RunStats {
  std::int64_t periods = 0;
  std::int64_t instructions_executed = 0;
  std::int64_t conservation_checks = 0;
  std::int64_t conservation_violations = 0;
  double       worst_conservation_error = 0.0;
  /// Sweeps of an iteration region body, summed over every region and every
  /// period. A period with no region adds nothing.
  std::int64_t total_iterations = 0;
  /// Region-periods that hit `max_iterations` without satisfying the
  /// criterion. Non-zero means some answer in this run is not converged.
  std::int64_t windows_not_converged = 0;
  /// Spec §5.4. How many conservative remap matrices `initialize()` built for
  /// this plan -- one per distinct `RemapPlan`, built at setup and never in
  /// the step loop.
  ///
  /// WHY THIS IS A COUNTER AND NOT A BOOLEAN. `worst_conservation_error` reads
  /// exactly 0 both when a remap was checked and was perfect and when no remap
  /// ran at all, so no error magnitude can distinguish "the weights were used"
  /// from "the weights were never built". A test that wants to assert a REAL
  /// remap happened -- rather than the coincident-mesh identity the runtime
  /// still uses when it can prove the two regions are the same cells in the
  /// same order -- asserts on this.
  std::int64_t remap_weight_builds = 0;
  /// Non-zeros summed over those matrices. `n` for an exact permutation of `n`
  /// cells, more when cells genuinely overlap several partners; 0 when every
  /// `Remap` took the proved-coincident identity path.
  std::int64_t remap_weight_nonzeros = 0;
  /// Times the runtime compared a component's registered `Provides` storage
  /// against the checkpoint it took, after that component's `restore_state()`
  /// (spec §5.3: FUSE verifies that the component saved what the derived set
  /// said it must, rather than trusting `supports_rollback()`). Exposed so the
  /// verification is observable, and therefore testable, rather than asserted
  /// in a comment.
  ///
  /// R16: "registered" is load-bearing, not decorative. Each count is a
  /// `memcmp` against one arc's REGISTERED storage; a component's own
  /// private state that was never bound via `register_fields()` is not
  /// counted here and is not covered by any of these verifications, even
  /// though `save_state()`/`restore_state()` are contractually required to
  /// handle it too (see the caveat on `Component::save_state()`,
  /// include/fuse/runtime/component.hpp).
  std::int64_t checkpoint_verifications = 0;
  /// Rank-agreement points this rank took part in. Zero in a serial run,
  /// and zero is correct there: with one rank there is nobody to agree with.
  std::size_t epochs_reconciled = 0;
};

/// Per-region outcome of the most recent period (spec §5.3).
struct IterationStats {
  int    iterations = 0;
  bool   converged = false;
  /// One entry per sweep: the 2-norm of the whole concatenated coupling
  /// residual `x_new - x_old`. Capacity is reserved at setup, so appending to
  /// it inside the sweep loop allocates nothing.
  std::vector<double> residual_history;
};

/// KNOWN_ISSUES.md D12's fix. One externally driven component's suspended
/// `advance()` point, reported by `Runner::resume_period()`: every
/// instruction up to (but not including) this component's own `advance()`
/// has already run, so every channel that feeds it has been exchanged and
/// its `Requires` storage is current -- unlike the deferred, end-of-period
/// scheme this replaces, which could not make that promise.
struct ExternalAdvanceRequest {
  /// The graph component name whose turn it now is -- the same string
  /// `Component::name()` returns, and the same one a caller must pass to
  /// `fuse_begin_step()` for `resume_period()` to make progress past this
  /// point. A caller that names a DIFFERENT externally driven component
  /// here is asking out of turn; the ABI layer (src/abi/participant.cpp)
  /// turns that into `E-ABI-STEP-ORDER`, naming this component as the one
  /// the stream actually expects.
  std::string component;
  /// The exact interval this component's own `advance()` must cover,
  /// including the sweep index inside an iteration region (`iteration`,
  /// StepContext's own doc comment).
  StepContext ctx;
  /// True exactly once, on the first time this component is resumed to
  /// after an iteration region's `save_state()` captured it (spec §5.3).
  /// `fuse_step_report::requires_checkpoint`'s real meaning, once this
  /// request reaches the ABI.
  bool requires_checkpoint = false;
  /// True exactly once, on the first time this component is resumed to
  /// after an iteration region's `restore_state()` rolled it back for a
  /// re-sweep. `fuse_step_report::requires_restore`'s real meaning.
  bool requires_restore = false;
  /// Unit 3 (S6 phase 3, Conformance). True when this request was
  /// recorded by `Runner::Impl::deliver_external()`
  /// (src/runtime/iteration.cpp), not by `advance()`: `component` names
  /// the component an `OpKind::ExternalDelivery` instruction's own
  /// output arc is bound for (the IMPORTER), not a component whose own
  /// `advance()` is next. `ctx`/`requires_checkpoint`/`requires_restore`
  /// above are meaningless on a request with this flag set and are left
  /// at their defaults.
  ///
  /// The lead's own ruling (2026-10-04), SUPERSEDING an earlier one the
  /// SAME day that called `component` here "purely informational" on a
  /// report: that was wrong, and produced a real contradiction (a
  /// fuse_begin_step() call that correctly consumed every pending proxy
  /// regardless of name still reached a real request naming a DIFFERENT
  /// component than the one `next_component` had just reported,
  /// because the proxy's own importer need not be the same component
  /// whose real advance comes next in the compiled stream). The
  /// corrected rule: `next_component` has exactly ONE meaning
  /// everywhere -- the component whose `fuse_begin_step()` the Runner
  /// expects next. A delivery proxy is not a component's turn, so it
  /// NEVER supplies that name directly: `fuse_end_step()`/`fuse_is_
  /// coupling_ongoing()`, finding a cached request with this flag set,
  /// report `next_component` by scanning FORWARD in the compiled
  /// stream from `delivery_instruction_index` (below) for the first
  /// real advance of an externally driven, hosted component --
  /// `Runner::Impl::next_real_advance_after()` (src/runtime/
  /// runner.cpp), the same kind of static plan scan
  /// `components_due_at_period_start()` already does.
  ///
  /// SECOND correction, the SAME day, once a scheduling-order fix
  /// (`fuse::OpKind::ExternalDelivery`'s own comment, `include/fuse/
  /// ir/op.hpp`) made a delivery instruction's own position coincide
  /// with the END of a time-window's per-sweep instruction range: a
  /// plain forward scan to the end of the compiled stream is only
  /// correct OUTSIDE every iteration region, because INSIDE one,
  /// whether execution continues past the region's own `exit` or loops
  /// back for another sweep is a convergence decision the region's own
  /// sweep loop makes AFTER this instruction runs -- not yet taken, not
  /// knowable from the compiled plan alone, and not something ANY
  /// static scan can resolve. So `next_real_advance_after()` now bounds
  /// its forward scan to the containing region's own pass `exit` and,
  /// finding nothing there, WRAPS to that pass's own `enter` and scans
  /// up to (not including) `delivery_instruction_index` -- the
  /// instruction sequence the sweep loop actually runs on another
  /// sweep. Only when the instruction lies outside every region does
  /// "found nothing" fall through to `components_due_at_period_start()`
  /// 's own answer for the next period, as before. The reported
  /// `next_component` is therefore a CONDITIONAL answer -- the
  /// component due IF the iteration continues -- never a promise of
  /// what `fuse_begin_step()` will actually do next.
  ///
  /// Because no static scan can know whether the iteration actually
  /// continues, `fuse_begin_step(C)`'s own resumption is the one place
  /// the real answer is resolved, and it may legitimately disagree with
  /// the conditional guess above: whatever `C` names, it still resumes
  /// the Runner PAST every pending proxy it encounters while priming --
  /// whatever component each one names, not only ones matching `C` --
  /// but if that resumption runs the region through to real convergence
  /// and on into the period's own end, `prime_next_request()` (src/abi/
  /// participant.cpp) treats that as a legitimate outcome, not an
  /// error: it completes this period's bookkeeping exactly as
  /// `fuse_run()` would at a period's end, starts the next one, and
  /// primes ITS OWN first request. `fuse_step_report::step_opened`
  /// (appended 2026-10-04, `entries.def`) reports which happened -- 1
  /// if a step opened on `C` as usual (whether or not a period
  /// rollover occurred first), 0 if the rolled-over period's own first
  /// request names a DIFFERENT component, in which case NO step is
  /// open on ANY component and the caller's own loop body must call
  /// `fuse_begin_step()` again with `fuse_step_report::next_component`.
  /// Only a mismatch reached WITHOUT a period rollover remains the
  /// ordinary `E-ABI-STEP-ORDER` out-of-turn refusal. Never open a step
  /// directly off a request with this flag set. The writer never needs
  /// to know which import is "due"; it writes what it produced, when
  /// it produced it -- `docs/user/c-abi.md` documents the full
  /// protocol, including `step_opened`.
  bool is_delivery_due = false;
  /// Valid only when `is_delivery_due`: this request's own `OpKind::
  /// ExternalDelivery` instruction's index into `ExecutionPlan::
  /// instructions()`, the SAME index `Runner::Impl::run_region()`
  /// passes to `deliver_external()` -- `next_real_advance_after()`'s
  /// own scan starting point.
  std::size_t delivery_instruction_index = 0;
};

/// Spec §2 layer 3: interprets the flat instruction stream. Every ArcId is
/// resolved to a raw pointer once at initialize(); the loop allocates nothing.
class Runner {
 public:
  Runner(const CouplingGraph& graph, const OperatorDAG& dag, ExecutionPlan plan,
         const MeshBackend& backend);
  ~Runner();

  Runner(Runner&&) noexcept;
  Runner& operator=(Runner&&) noexcept;

  Result<void> initialize(ComponentSet& components, FieldRegistry& registry,
                          const RunnerConfig& config = {});

  Result<void> run_period(ComponentSet& components);
  Result<void> run(ComponentSet& components, std::int64_t periods);

  /// KNOWN_ISSUES.md D12's fix: resumes the current period's instruction
  /// stream from wherever it last suspended (or begins a new period, if
  /// the previous one ran to completion), executing every instruction up
  /// to the NEXT externally driven component's own `advance()` point and
  /// returning that component's `ExternalAdvanceRequest` there -- WITHOUT
  /// calling `advance()` itself; that happens on the NEXT call to this
  /// function, which is what src/abi/participant.cpp's `fuse_end_step()`
  /// drives. A `std::nullopt` result means the period ran to completion
  /// (every component that still owed this period a tick, in-process or
  /// external, has had it) with no externally driven component left to
  /// stop at; the period's own stats (`stats()`) are updated exactly as
  /// `run_period()` updates them, at the same point in the stream.
  ///
  /// Mixing this with `run_period()`/`run()` on the SAME Runner is a
  /// caller error this does not itself detect: `run_period()` drives a
  /// period to completion in one call and refuses
  /// (`E-RT-EXTERNAL-UNSUPPORTED`) if doing so would require suspending,
  /// so the two are mutually exclusive by construction for any graph with
  /// an externally driven component, but a graph with NONE could call
  /// either; callers should pick one and keep to it, as
  /// src/abi/participant.cpp does (`fuse_run()` vs `fuse_begin_step()`/
  /// `fuse_end_step()`).
  Result<std::optional<ExternalAdvanceRequest>> resume_period(
      ComponentSet& components);

  /// Defect 3's own refinement (Conformance Unit 2, docs/dev/abi-design.md):
  /// `fuse_is_coupling_ongoing()` originally had no way to answer
  /// `next_component` AT A PERIOD'S OWN START without calling
  /// `resume_period()` -- a real side effect a previously side-effect-free
  /// entry point must never pay for just to answer a status query. But the
  /// instruction stream is static once `schedule()` has built it: the
  /// FIRST externally driven component of a period is a property of the
  /// compiled plan, not of where execution happens to be right now, and
  /// is the SAME for every period (the schedule's own common-period
  /// invariant). This returns every component the compiled stream visits
  /// at time 0 of a period, in the EXACT order it visits them -- both
  /// in-process and externally driven, since only the caller (the ABI
  /// layer, `src/abi/participant.cpp`) knows which components in THIS
  /// process are externally driven; it picks the first one that is. No
  /// `advance()` call, no coroutine, no suspension: safe to call from any
  /// state, any number of times, including before `resume_period()` has
  /// ever been called at all.
  [[nodiscard]] std::vector<ComponentId> components_due_at_period_start() const;

  /// Unit 3 (S6 phase 3, Conformance): resolves `next_component` when
  /// the cached request is a delivery proxy
  /// (`ExternalAdvanceRequest::is_delivery_due`'s own comment, above,
  /// has the full account of why the proxy's own importer name is
  /// never the right answer to report). `Runner::Impl::next_real_
  /// advance_after()` (`src/runtime/runner_impl.hpp`) has the scan's
  /// own full documentation. Returns `std::nullopt` if the scan runs
  /// off the end of this period; the caller (`src/abi/participant.cpp`)
  /// wraps to `components_due_at_period_start()`'s own answer, since
  /// the schedule repeats identically every period. Like that
  /// function, a pure query over the compiled plan: no `advance()`
  /// call, no coroutine, no suspension, safe from any state.
  [[nodiscard]] std::optional<ComponentId> next_real_advance_after(
      const ComponentSet& components, std::size_t instruction_index) const;

  [[nodiscard]] const RunStats& stats() const noexcept;
  /// One entry per iteration region of the plan, in DAG region order,
  /// describing the most recent period. Empty for a plan with no region.
  [[nodiscard]] const std::vector<IterationStats>& iteration_stats()
      const noexcept;
  /// The derived checkpoint set's explanation for every region, in DAG region
  /// order (spec §5.3: the user never wrote the set down, so the only way they
  /// can disagree with it is if it is printed). Available after initialize().
  [[nodiscard]] const std::vector<std::string>& checkpoint_explanations()
      const noexcept;
  [[nodiscard]] Result<const double*> buffer(ArcId arc) const;

  /// Unit 3 (S6 phase 3, Conformance): the C++ side of the ABI's new
  /// `view_field` entry (fuse_field_view's own `window_samples` member,
  /// include/fuse/abi/entries.def). For a field that is part of a
  /// coupling window (export or import), `window_samples` is the
  /// window's own sample capacity K and `data`/`count` describe the
  /// window path's own private buffer -- the WindowAdvance node's own
  /// output for an export, or the arc the window path reads the import
  /// from (ordinarily the window Sink's own arc; for a LowFive-bound
  /// channel, the ExternalDelivery arc instead) -- neither of which
  /// `register_field()`'s ordinary per-component storage ever aliases
  /// (`plan_buffers()`'s own `is_window_arc()` exclusion,
  /// src/schedule/buffers.cpp). `window_samples` is 0, and `data`/
  /// `count` are left null/zero, for a field with no window scope at
  /// all -- the ABI layer (`src/abi/participant.cpp`) falls back to
  /// this context's own registered storage in that case, so this
  /// accessor need only ever answer the window question. Available
  /// only after `initialize()` (the window buffers do not exist before
  /// it); `E-RT-FIELD` for an out-of-range id.
  struct FieldWindowView {
    const double* data = nullptr;
    std::uint64_t count = 0;
    std::int32_t  window_samples = 0;
  };
  [[nodiscard]] Result<FieldWindowView> field_window_view(FieldId field) const;

  /// Unit 2b (D19), step (d): whether THIS process hosts `component` --
  /// the same rule `WindowMemberConfig::hosted` (step (a)) already
  /// applies to a window member, generalised to any component. `true`
  /// before `initialize()` has run (nothing to disagree with yet) or
  /// whenever no per-rank identity is configured (`RunnerConfig::
  /// force_this_process`/`base_communicator` both unset -- tier 0, or a
  /// genuinely single-rank run): every existing caller that never names
  /// a rank keeps meaning exactly what it always meant. The ABI layer
  /// (`src/abi/participant.cpp`) reads this to narrow `fuse_run()`'s own
  /// `E-ABI-RUN-EXTERNAL` refusal to "an externally driven component IS
  /// hosted here", rather than "the GRAPH declares one anywhere at all".
  [[nodiscard]] bool hosts_component(ComponentId component) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fuse
