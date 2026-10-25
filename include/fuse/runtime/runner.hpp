#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/num/accelerator.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/model/restart.hpp"
#include "fuse/runtime/fixed_point_residual.hpp"
#include "fuse/schedule/plan.hpp"
#include "fuse/transport/transport.hpp"

namespace fuse {

struct RunnerConfig {
  std::string transport = "copy";
  bool        check_conservation = true;
  double      conservation_tolerance = 1e-12;
  bool        abort_on_violation = false;
  /// Spec Section 5.3. An iteration region that reaches `max_iterations` without
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
  /// every component in the graph -- `MPI_COMM_WORLD` under spec Section 6 mode 1,
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

  /// Overrides `build_ticks()`'s own `this_process`
  /// computation directly, bypassing `base_communicator`/`MPI_Comm_rank`
  /// entirely -- the SAME reason `force_distributed_nodal_remap` above
  /// exists: a tier-0 build (no MPI compiled in at all) can exercise the
  /// per-rank HOSTING decision (ordinary ticks' own `hosted[]`, and
  /// `WindowMemberConfig::hosted`) with a specific rank
  /// number, with no real communicator, no launcher, and no `#if
  /// FUSE_HAVE_MPI` branch ever taken. `base_communicator` need not be
  /// set at all when this is -- a test that wants both a pretend "rank 1
  /// of 3" placement AND genuine ordinary-tick filtering sets only this
  /// one field. Left unset (the default) for every production caller,
  /// which derives `this_process` from `base_communicator`.
  std::optional<int> force_this_process;

  /// `Transport`'s virtual interface (prepare/move/
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

  /// Which Kokkos execution space `Runner::Impl::execute()` dispatches
  /// its elementwise `OpKind`s (Interpolate/Remap/Sample/Accumulate/
  /// Extrapolate/Merge/Scale/Relax/Residual) on. `Host` (the default) is
  /// the plain host path -- `Device` is opt-in, and only meaningful when
  /// fuse was built against a Kokkos with a real device backend
  /// (CUDA/HIP/SYCL) compiled in; asking for it otherwise is a configuration error (see
  /// `Runner::Impl::setup()`), not a silent fall back to `Host`. Everything
  /// else `execute()` does (Transport/Epoch/Sink/Accelerate/Constrain/
  /// window refusals) is unconditionally host-only and ignores this field
  /// entirely -- see docs/dev/runner-device-dispatch.md.
  ///
  /// `Default` (the default) is resolved by `Runner::initialize()` from the
  /// process environment: `FUSE_EXECUTION_SPACE=device` selects `Device`,
  /// `host`, empty or unset selects `Host`, anything else is `E-RT-SPACE`.
  /// That is what lets a whole test binary or example run under the device
  /// interpreter unchanged (the `.device` ctest twins,
  /// docs/dev/device-resident-arcs.md step 4) while a caller that names
  /// `Host` or `Device` explicitly is never overridden.
  enum class RunnerExecutionSpace { Default, Host, Device };
  RunnerExecutionSpace execution_space = RunnerExecutionSpace::Default;
  /// `Default` resolved against the environment as described above; what
  /// `initialize()` stores. `execution_space` itself is left as given.
  [[nodiscard]] Result<RunnerExecutionSpace> resolved_execution_space() const;
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
  /// Spec Section 5.4. How many conservative remap matrices `initialize()` built for
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
  /// (spec Section 5.3: FUSE verifies that the component saved what the derived set
  /// said it must, rather than trusting `supports_rollback()`). Exposed so the
  /// verification is observable, and therefore testable, rather than asserted
  /// in a comment.
  ///
  /// "registered" is load-bearing, not decorative. Each count is a
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
  /// Instructions whose elementwise kernels ran on the device space
  /// (`RunnerConfig::execution_space == Device`). Exactly 0 under the `Host`
  /// default. A counter for the same reason `remap_weight_builds` is one:
  /// the host and device paths are meant to agree numerically, so no
  /// delivered value can tell "the device computed it" from "the host did",
  /// and a test that wants to assert the device path was actually exercised
  /// asserts on this.
  std::int64_t device_dispatches = 0;
  /// Coupler-owned arcs whose storage lives in the device memory space
  /// (docs/dev/device-resident-arcs.md): allocated once in setup() when
  /// `execution_space == Device` for every arc that only Space-generic
  /// instructions touch and no region, window or spacetime bookkeeping
  /// reads on the host. Exactly 0 under `Host`. Counts arcs, not bytes, so
  /// a test can assert the device path held data without staging.
  std::int64_t device_resident_arcs = 0;
  /// Copies across the host/device boundary in Device mode: a host-
  /// registered Provides field's device mirror refreshed before a kernel
  /// reads it (once per component step it is read after), and a Sink/Epoch/
  /// Accelerate copy whose two arcs live in different spaces. A same-space
  /// copy does not count. 0 under Host. In Device mode with every field
  /// device-registered, the only crossings left are the two around each
  /// Transport, host code until the GPU-aware transport step.
  std::int64_t device_boundary_copies = 0;
  /// Per-instruction staging copies in Device mode (`Impl::device_staging`):
  /// one for each host-side operand a kernel had to be handed through a
  /// staging lane (host arc pinned by a host-only instruction, `previous[]`,
  /// history, a host-registered field a kernel WRITES) and one for each
  /// lane written back. The number docs/dev/device-resident-arcs.md drives
  /// to zero: a plan whose instructions are all Space-generic or space-aware
  /// and whose fields are device-registered stages nothing. 0 under Host.
  std::int64_t device_staged_copies = 0;
  /// `OpKind::Transport` exchanges whose packed buffer was handed to MPI
  /// (or, for a same-process move, `Kokkos::deep_copy`'d) WITHOUT crossing
  /// to host memory: both arcs were device-resident and, for an MPI
  /// exchange, `Transport::is_gpu_aware()` was true
  /// (docs/dev/device-resident-arcs.md). 0 under Host, and 0 under
  /// Device whenever `FUSE_TRANSPORT_HOST_STAGING=1` forces the fallback
  /// below instead. Disjoint from `device_staged_copies`: the host-staged
  /// fallback (not GPU-aware, or forced) still packs/unpacks ON the device
  /// and counts here as a `device_boundary_copies` pair, never as a
  /// `device_staging`-lane staged copy -- a Transport plan's own send/recv
  /// buffers are not `device_staging`'s four lanes.
  std::int64_t device_transport_moves = 0;
};

/// Per-region outcome of the most recent period (spec Section 5.3).
struct IterationStats {
  int    iterations = 0;
  bool   converged = false;
  /// One entry per sweep: the 2-norm of the whole concatenated coupling
  /// residual `x_new - x_old`. Capacity is reserved at setup, so appending to
  /// it inside the sweep loop allocates nothing.
  std::vector<double> residual_history;
};

/// One externally driven component's suspended `advance()` point, reported
/// by `Runner::resume_period()`: every instruction up to (but not including)
/// this component's own `advance()` has already run, so every channel that
/// feeds it has been exchanged and its `Requires` storage is current.
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
  /// after an iteration region's `save_state()` captured it (spec Section 5.3).
  /// `fuse_step_report::requires_checkpoint`'s real meaning, once this
  /// request reaches the ABI.
  bool requires_checkpoint = false;
  /// True exactly once, on the first time this component is resumed to
  /// after an iteration region's `restore_state()` rolled it back for a
  /// re-sweep. `fuse_step_report::requires_restore`'s real meaning.
  bool requires_restore = false;
  /// True when this request was
  /// recorded by `Runner::Impl::deliver_external()`
  /// (src/runtime/iteration.cpp), not by `advance()`: `component` names
  /// the component an `OpKind::ExternalDelivery` instruction's own
  /// output arc is bound for (the IMPORTER), not a component whose own
  /// `advance()` is next. `ctx`/`requires_checkpoint`/`requires_restore`
  /// above are meaningless on a request with this flag set and are left
  /// at their defaults.
  ///
  /// `next_component` has exactly ONE meaning everywhere -- the component
  /// whose `fuse_begin_step()` the Runner expects next. A delivery proxy is
  /// not a component's turn, so it NEVER supplies that name directly:
  /// `fuse_end_step()`/`fuse_is_coupling_ongoing()`, finding a cached
  /// request with this flag set, report `next_component` by scanning
  /// FORWARD in the compiled stream from `delivery_instruction_index`
  /// (below) for the first real advance of an externally driven, hosted
  /// component -- `Runner::Impl::next_real_advance_after()`
  /// (src/runtime/runner.cpp), the same kind of static plan scan
  /// `components_due_at_period_start()` already does. (The proxy's own
  /// importer need not be the component whose real advance comes next in
  /// the compiled stream, so `component` on a proxy request is not
  /// that name.)
  ///
  /// A plain forward scan to the end of the compiled stream is only
  /// correct OUTSIDE every iteration region: INSIDE one, whether execution
  /// continues past the region's own `exit` or loops back for another
  /// sweep is a convergence decision the region's own sweep loop makes
  /// AFTER this instruction runs -- not knowable from the compiled plan
  /// alone. So `next_real_advance_after()` bounds its forward scan to the
  /// containing region's own pass `exit` and, finding nothing there, WRAPS
  /// to that pass's own `enter` and scans up to (not including)
  /// `delivery_instruction_index` -- the instruction sequence the sweep
  /// loop actually runs on another sweep. Only when the instruction lies
  /// outside every region does "found nothing" fall through to
  /// `components_due_at_period_start()`'s own answer for the next period.
  /// The reported `next_component` is therefore a CONDITIONAL answer -- the
  /// component due IF the iteration continues -- never a promise of what
  /// `fuse_begin_step()` will actually do next.
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
  /// (`entries.def`) reports which happened -- 1
  /// if a step opened on `C` as usual (whether or not a period
  /// rollover occurred first), 0 if the rolled-over period's own first
  /// request names a DIFFERENT component, in which case NO step is
  /// open on ANY component and the caller's own loop body must call
  /// `fuse_begin_step()` again with `fuse_step_report::next_component`.
  /// Only a mismatch reached WITHOUT a period rollover remains the
  /// ordinary `E-ABI-STEP-ORDER` out-of-turn refusal. Never open a step
  /// directly off a request with this flag set. The writer DOES need to
  /// know which export is due, once a graph has more than one
  /// independently-paced LowFive-bound channel -- `delivery_field` below
  /// is what tells it -- `docs/user/c-abi.md` documents the full protocol,
  /// including `step_opened`.
  bool is_delivery_due = false;
  /// Valid only when `is_delivery_due`: this request's own `OpKind::
  /// ExternalDelivery` instruction's index into `ExecutionPlan::
  /// instructions()`, the SAME index `Runner::Impl::run_region()`
  /// passes to `deliver_external()` -- `next_real_advance_after()`'s
  /// own scan starting point.
  std::size_t delivery_instruction_index = 0;
  /// Valid only when `is_delivery_due`: the delivering channel's own
  /// SOURCE field, as a `FieldId::value()` -- the SAME id space
  /// `fuse_field_ref::handle` uses (`impl_field_handle()`, src/abi/
  /// participant.cpp). `Runner::Impl::deliver_external()` sets this from
  /// the `OpKind::ExternalDelivery` node's own origin channel
  /// (`Node::origin`, `fuse/ir/op.hpp`) -- ONE channel, ONE field, no
  /// guessing which of a graph's several LowFive-bound channels this
  /// particular suspension is about. `fuse_step_report::delivery_field`
  /// (entries.def) is this value, verbatim, on the report that resumes
  /// past this request. -1 is never a valid value here (a delivery-due
  /// request always names a real field); it exists only as `fuse_step_
  /// report::delivery_field`'s own default when `delivery_pending` is 0.
  std::int32_t delivery_field = -1;
};

/// Spec Section 2 layer 3: interprets the flat instruction stream. Every ArcId is
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

  /// The Kokkos execution space `RunnerConfig::RunnerExecutionSpace::
  /// Device` resolves to on THIS build -- `"Cuda"` against a CUDA-enabled
  /// Kokkos, `"HIP"`/`"SYCL"` likewise. Equal to the host space's own name
  /// when no device backend was compiled in, which is exactly the build
  /// `initialize()` refuses `Device` on (`E-RT-DEVICE`), so a caller can
  /// ask before configuring rather than after failing.
  [[nodiscard]] static std::string_view device_execution_space_name() noexcept;

  /// THE PERIOD CONTRACT (stated here as the Runner's
  /// own half -- docs/dev/abi-design.md states the ABI's half of the same
  /// rule). A period is this Runner's own unit of forward progress: one
  /// call to `run_period()`, or the externally driven equivalent
  /// `resume_period()` resuming until the SAME period boundary, advances
  /// the compiled instruction stream through exactly one period and no
  /// further. What a caller is entitled to ask for is a COUNT of periods
  /// (`run()`'s own `periods` argument; an ABI caller's own loop counting
  /// `fuse_step_report::period`) -- never a promise that state has
  /// "settled" or "converged" past that count, even on a graph whose
  /// window-scoped region happens to reach a fixed point every period
  /// (e.g. a time window that converges in 12 sweeps,
  /// every period, on a graph with no declared end time at all).
  ///
  /// THE INVARIANT: **this Runner must never execute a
  /// period's real instructions beyond the last one the caller actually
  /// asked for, even when every externally driven component's own request
  /// for that period has already been resolved and cached.** Resolving
  /// period N's own first request is "looking up what that period's own
  /// start requires" and may be done eagerly with no observable effect
  /// (`fuse_is_coupling_ongoing()`'s own `next_component` does exactly
  /// this, for free, at a period's start -- its own doc comment). ACTING
  /// on a period beyond the caller's own requested count -- running a
  /// real `Transport`/`Epoch`/`Constrain` instruction, or calling a
  /// component's real `advance()` -- is not: a base-communicator
  /// collective inside that extra period has no guarantee any OTHER rank
  /// of the same component ever reaches it, because a driver with its own
  /// bound (an ABI caller's own `for (p = 0; p < periods; ++p)` loop, or
  /// `Runner::run()`'s own `for (i = 0; i < periods; ++i) run_period();`)
  /// stops exactly at that bound and never calls in again. The
  /// consequence, measured directly: two externally driven
  /// ranks (U, V) each continued into an UNREQUESTED third period's own
  /// `Constrain` reduction after a converged, no-end-time window reached
  /// its fixed point at the end of period 2, while a third rank (the
  /// in-process coupler, driven through `run(components, 2)`) had
  /// already returned -- U and V then hung forever in that `Constrain`'s
  /// `MPI_Allreduce`, waiting for a peer that would never call in again.
  ///
  /// WHAT ENFORCES THIS. `src/abi/participant.cpp`'s `prime_next_request()`
  /// is the one caller-facing place this invariant can be violated from:
  /// when the current period has just drained with no end time declared
  /// AND a base communicator is configured (`fuse_ctx::communicator_set`),
  /// it must stop -- `c.next_request.reset(); return true;` -- and leave
  /// starting the FOLLOWING period's own first request to a LATER,
  /// separate caller-initiated `fuse_begin_step()` call, rather than
  /// resuming the stream again inside the SAME call that just closed the
  /// prior period. `test/verification/abi_period_boundary_test.cpp`'s
  /// `AbiPeriodBoundary.NeverRunsAPeriodTheCallerDidNotRequestOnceState
  /// HasReachedAFixedPoint` pins exactly this, at tier 0, through the
  /// real public ABI (no MPI, `fuse::abi_test::force_communicator_set()`
  /// standing in for a live communicator the same way
  /// `brusselator_period_rollover_test.cpp` does).
  Result<void> run_period(ComponentSet& components);
  Result<void> run(ComponentSet& components, std::int64_t periods);

  /// Resumes the current period's instruction
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

  /// Answers `next_component` AT A PERIOD'S OWN START without calling
  /// `resume_period()` (docs/dev/abi-design.md): `fuse_is_coupling_ongoing()`
  /// is a side-effect-free entry point and must not pay for a real side
  /// effect just to answer a status query. The
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

  /// Resolves `next_component` when
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
  /// order (spec Section 5.3: the user never wrote the set down, so the only way they
  /// can disagree with it is if it is printed). Available after initialize().
  [[nodiscard]] const std::vector<std::string>& checkpoint_explanations()
      const noexcept;
  [[nodiscard]] Result<const double*> buffer(ArcId arc) const;

  /// TEST-ONLY (never for production use, and not part of the stable
  /// Runner API -- the same reason `fuse::abi_test::force_this_process()`
  /// (src/abi/test_support.hpp) reaches past the frozen ABI for a test
  /// that has nowhere else to stand): after a successful `initialize()`,
  /// whether the window-scoped residual block for the channel named
  /// `channel_name` (`RegionBlock::channel_name`, the SAME name
  /// `CouplingGraph::add_channel()` was given) is owned on this process
  /// (`RegionBlock::window_hosted`, src/runtime/runner_impl.hpp's own
  /// comment has the full account of what "owned" means for a window
  /// block). `std::nullopt` when no region of this plan has a block for
  /// that channel, or `initialize()` has not yet succeeded.
  ///
  /// Exists because `window_hosted` is private implementation state with
  /// no production-code reader outside `run_region()` itself, and the
  /// rule that resolves it (`WindowMemberConfig::export_consumer_hosted`)
  /// has no public-API effect reachable without a real multi-rank
  /// communicator (`run_region()`'s own `gate_by_hosted` only consults it
  /// when `RegionRuntime::participants.size() > 1`, which requires
  /// `RunnerConfig::base_communicator` -- refused outright on a build
  /// with no MPI). `test/runtime/window_hosting_test.cpp` is this seam's
  /// only caller.
  [[nodiscard]] std::optional<bool> test_only_window_block_hosted(
      const std::string& channel_name) const;

  /// The C++ side of the ABI's `view_field` entry (fuse_field_view's own
  /// `window_samples` member, include/fuse/abi/entries.def). For a field that is part of a
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

  /// Whether THIS process hosts `component` --
  /// the same rule `WindowMemberConfig::hosted` already
  /// applies to a window member, generalised to any component. `true`
  /// before `initialize()` has run (nothing to disagree with yet) or
  /// whenever no per-rank identity is configured (`RunnerConfig::
  /// force_this_process`/`base_communicator` both unset -- tier 0, or a
  /// genuinely single-rank run): every existing caller that never names
  /// a rank means a single-rank run. The ABI layer
  /// (`src/abi/participant.cpp`) reads this to narrow `fuse_run()`'s own
  /// `E-ABI-RUN-EXTERNAL` refusal to "an externally driven component IS
  /// hosted here", rather than "the GRAPH declares one anywhere at all".
  [[nodiscard]] bool hosts_component(ComponentId component) const;

  /// The Newton-Krylov seam. A `fuse::FixedPointResidual` for iteration
  /// region `region_index` (the same indexing `iteration_stats()` uses),
  /// in place of letting this `Runner` drive that region's own Picard
  /// sweep loop: the JFNK Newton solver calls `evaluate`
  /// instead of `run_period()`/`run()` ever reaching this region.
  ///
  /// `evaluate(x, f)` computes `f = x - G(x)`, `G` one Picard sweep of the
  /// region's body delivered with `x` on the interface, bracketed by
  /// every checkpointed member's own `save_state()`/`restore_state()` and
  /// by a snapshot/restore of every coupler-owned loop-carried buffer the
  /// region tracks (`IterationRegionDecl`'s own derived checkpoint set) --
  /// so a call leaves every member's state, and the region's own
  /// coupler-side buffers, exactly as it found them. The region may be
  /// re-entered afterward, by this same call or by `run_period()`'s own
  /// Picard loop, and reaches the same fixed point it would have without
  /// the evaluation, bit for bit.
  ///
  /// SCOPE: refuses `E-NEWTON-UNSUPPORTED-REGION` for a window-
  /// scoped region, a region with more than one cross-rank participant, or
  /// a region with an externally driven member -- none of those have a
  /// Picard sweep this function can run standalone without also
  /// reimplementing `run_region()`'s own window/parallel machinery, which
  /// is out of scope for the seam itself. Available only after a
  /// successful `initialize()`; refuses `E-RT-REGION` for an out-of-range
  /// `region_index`.
  [[nodiscard]] Result<FixedPointResidual> region_fixed_point_residual(
      std::size_t region_index, ComponentSet& components);

  /// TEST/DIAGNOSTIC-ONLY, the
  /// `NewtonResidualPath::Direct` sibling of `region_fixed_point_
  /// residual()` immediately above -- same shape, same save/restore
  /// contract, but `evaluate` calls `Runner::Impl::evaluate_region_
  /// residual_direct()` (src/runtime/fixed_point_residual.cpp) instead,
  /// so a caller can compare `F(x)` under both seams at the SAME `x`
  /// without running a full Newton solve. No production caller:
  /// `run_region_newton()` reaches `evaluate_region_residual_direct()` directly when
  /// `NewtonConfig::residual_path == Direct`. Refuses the same way
  /// `region_fixed_point_residual()` does, PLUS `E-RT-REGION` when the
  /// region's own `newton_config.residual_path` is not `Direct` --
  /// `RegionRuntime::newton_direct_members` is resolved only for a
  /// region actually configured for Direct (resolve_regions(), this
  /// file's own sibling), so calling this diagnostic on an
  /// Advance-configured region would otherwise silently answer with an
  /// empty evaluate that computes nothing rather than F(x). (A region
  /// whose `residual_path` IS `Direct` but is also window-scoped never
  /// reaches this function in the first place: `initialize()` itself
  /// already refused it with `E-NEWTON-DIRECT-WINDOW`, so no `Runner`
  /// exists to call this on.)
  [[nodiscard]] Result<FixedPointResidual> region_fixed_point_residual_direct(
      std::size_t region_index, ComponentSet& components);

  /// TEST/DIAGNOSTIC-ONLY (newton-krylov.md Sec. 9.7), the window-scoped
  /// sibling of `region_fixed_point_residual()` above -- see `WindowFixedPointResidual`'s own
  /// comment (fuse/runtime/fixed_point_residual.hpp) for the shape and
  /// `Runner::Impl::evaluate_window_region_residual()` (src/runtime/
  /// runner_impl.hpp) for the save/restore contract `evaluate` carries.
  /// No production caller: `run_region_newton_window()` reaches the same
  /// function directly. Refuses the same way `region_fixed_point_
  /// residual()` does for an out-of-range `region_index` or a `Runner`
  /// not yet `initialize()`d; additionally refuses `E-RT-REGION` when
  /// `region_index` does not name a WINDOW-scoped region (use `region_
  /// fixed_point_residual()` for an ordinary one).
  [[nodiscard]] Result<WindowFixedPointResidual> window_region_fixed_point_residual(
      std::size_t region_index, ComponentSet& components);

  /// TEST/DIAGNOSTIC-ONLY (newton-krylov.md Sec. 9.7) snapshot of every
  /// window member's own checkpoint state for iteration region `region_index` -- see
  /// `WindowMemberDebugSnapshot`'s own comment (fuse/runtime/fixed_
  /// point_residual.hpp) for the fields and what they let a test pin.
  /// One entry per `RegionRuntime::window_member_nodes`, in that order.
  /// Refuses the same way `window_region_fixed_point_residual()` does.
  [[nodiscard]] Result<std::vector<WindowMemberDebugSnapshot>>
  debug_window_member_snapshots(std::size_t region_index) const;

  /// TEST/DIAGNOSTIC-ONLY, the same
  /// convention as `debug_window_member_snapshots()` immediately
  /// above -- a copy of `RegionRuntime::newton_scale`, the diagonal `D`
  /// `NewtonNorm::MeasureWeighted` scales the Newton residual by (`D_i =
  /// sqrt(w_i / sum_block_w) / s_b`, built once in resolve_regions() by
  /// `build_newton_measure_weights()`), in `RegionBlock` order (`offset`
  /// gives each block's own slice). Empty for `NewtonNorm::Plain`
  /// (`run_region_newton()`/`run_region_newton_window()` read that
  /// emptiness as "do not scale" -- the identical convention those two
  /// functions themselves rely on). Lets a test compare the PRODUCTION
  /// `D` this Runner actually solves with against the formula computed
  /// independently, rather than re-deriving the formula in the test and
  /// never exercising `build_newton_measure_weights()` at all -- the gap
  /// `build-evidence/mutations/newton-model.toml`'s own "weights not
  /// applied"/"a block's scale dropped" rows need a test that actually
  /// reads this to close. Refuses the same way `region_fixed_point_
  /// residual()` does for an out-of-range `region_index` or a `Runner`
  /// not yet `initialize()`d.
  [[nodiscard]] Result<std::vector<double>> debug_newton_scale(
      std::size_t region_index) const;

  /// TEST/DIAGNOSTIC-ONLY, the same
  /// convention as `debug_newton_scale()` immediately above -- a copy of
  /// `RegionRuntime::newton_gmres_iterations` (one entry per outer
  /// Newton iteration of the MOST RECENT solve of this region, in
  /// order), the one piece of `fuse::num::NewtonResult::trace` neither
  /// `run_region_newton()` nor `run_region_newton_window()` otherwise
  /// exposes short of `FUSE_NEWTON_DEBUG=1` stderr tracing. Empty before
  /// the first Newton solve of this region. Refuses the same way
  /// `debug_newton_scale()` does for an out-of-range `region_index` or a
  /// `Runner` not yet `initialize()`d.
  [[nodiscard]] Result<std::vector<std::size_t>> debug_newton_gmres_iterations(
      std::size_t region_index) const;

  /// Restart (docs/dev/restart.md). The period
  /// counter `run_period()`/`run()` advance (`Runner::Impl::period_index`,
  /// src/runtime/runner_impl.hpp) -- a restart point records this value
  /// (docs/dev/restart.md Section 4.1's `@period_index`) so a restored
  /// run resumes counting from where the point was taken, not from zero.
  [[nodiscard]] std::int64_t current_period_index() const noexcept;

  /// The write-side counterpart: resumes the Runner's own clock
  /// (`ctx.t_begin = Rational{period_index} * plan.period() + tick.time`,
  /// `src/runtime/runner.cpp`) at `period_index`, so the NEXT `run_period()`/
  /// `run()` computes the SAME `StepContext` times an uninterrupted run
  /// would have at that period -- the one piece of state `Runner::
  /// restore_restart_fields()` does NOT touch (it restores component and
  /// accelerator state only), because the Runner's own clock is not a
  /// `fuse::Component`: `fuse::restore_restart_point()` (fuse/runtime/
  /// restart.hpp) calls this with the restart point's own recorded
  /// `@period_index` immediately after `restore_restart_fields()`
  /// succeeds, before the caller's first post-restart `run_period()`.
  /// A caller error to call this after any `run_period()`/`run()` call
  /// has already advanced the SAME Runner past period 0 for a reason
  /// other than restoring -- not checked here (this Runner has no way to
  /// tell "restoring" from "resetting"), the caller's own responsibility
  /// the same way `resume_period()`/`run_period()` mutual exclusivity is.
  void set_current_period_index(std::int64_t period_index) noexcept;

  /// Writes every restartable component's own state (Form A via
  /// `Component::write_restart()`, Form B via `Component::
  /// save_restart_blob()`, gated on `supports_restart_form_a()`/
  /// `supports_restart_form_b()` -- a component with NEITHER bit set is
  /// skipped entirely, the same default-no-op posture `save_state()`'s
  /// own default takes for a component that is never iterated) and every
  /// accelerated region's own accelerator memory (`RegionRuntime::accel`,
  /// keyed by the owning component's own global ids)
  /// into `writer`. Does NOT call `writer.finish_point()` -- the caller
  /// (`fuse::write_restart_point()`, fuse/runtime/restart.hpp) does that
  /// once, after also writing the provenance scalars this method does
  /// not touch. A caller error to call this before a successful
  /// `initialize()`, reported the same way `run_period()` reports it.
  Result<void> write_restart_fields(RestartWriter& writer,
                                    ComponentSet& components) const;

  /// The read-back counterpart of `write_restart_fields()`, called before
  /// the FIRST `run_period()`/`run()` after a restore. For each
  /// accelerated region, `allow_accelerator_reset_on_mismatch` is the
  /// explicit, off-by-default run option (docs/dev/restart.md
  /// Section 4.1): when a region's current stacked-vector layout does
  /// not match what the file recorded for it, `false` refuses by name
  /// (E-RESTART-BAD-NUMBERING); `true` calls `Accelerator::reset()`
  /// instead and sets `any_accelerator_reset` to `true` -- the caller
  /// reports this at restart, never silently.
  Result<void> restore_restart_fields(RestartReader& reader,
                                      ComponentSet& components,
                                      bool allow_accelerator_reset_on_mismatch,
                                      bool& any_accelerator_reset);

  /// The participant-side restart request (docs/dev/restart.md). An
  /// externally-driven (ABI) participant has no other
  /// way to ask that a restart point be taken, because write_restart_
  /// point() needs a RestartWriter the embedding HOST process configures
  /// (a file path, which HDF5 layout) -- something no ABI entry's fixed
  /// `(ctx, in, out)` shape can carry (entries.def's own "ONE RULE").
  /// This is therefore a FLAG, not a trigger: the participant marks its
  /// own wish, and the embedding host -- the same C++/Python driver that
  /// already calls write_restart_point() itself -- polls
  /// take_restart_point_request() after every run_period()/
  /// resume_period() call and decides whether, and how, to act on it.
  /// Idempotent: calling this more than once before the host polls
  /// leaves exactly one pending request, never a queue.
  void request_restart_point() noexcept;

  /// True if request_restart_point() was called since the last time
  /// this function returned true; CLEARS the flag on the same call that
  /// reports it (consume, not peek), so two independent callers polling
  /// cannot both believe a request is still pending after one of them
  /// has already acted on it.
  [[nodiscard]] bool take_restart_point_request() noexcept;

  /// TEST-ONLY (docs/dev/restart.md),
  /// the same `debug_` convention as `debug_newton_scale()` above --
  /// never called by production code. Replaces region `region_index`'s
  /// own `Accelerator` with `accel` in place, after `initialize()` has
  /// already built the region from the graph's own `AccelKind`. Exists
  /// for exactly one purpose: exercising `restore_restart_fields()`'s
  /// own E-RESTART-ACCEL-NOT-IMPLEMENTED refusal against an accelerator
  /// that genuinely never overrides `restart_state_implemented()`
  /// (every REAL `AccelKind` does), without reintroducing a hand-rolled
  /// test double into a graph-compiled region any other way. Refuses the same way
  /// `debug_newton_scale()` does for an out-of-range `region_index` or
  /// a `Runner` not yet `initialize()`d.
  [[nodiscard]] Result<void> debug_replace_region_accelerator(
      std::size_t region_index, std::unique_ptr<Accelerator> accel);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fuse
