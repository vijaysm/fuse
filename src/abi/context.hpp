#pragma once

/// Private to src/abi/. `fuse_ctx` (the opaque type `include/fuse/abi/types.h`
/// only forward-declares as `struct fuse_ctx;`) is defined here, not in any
/// installed header: a C caller must not be able to see, size, or allocate
/// one (the same contract SOMA's `soma_ctx_s`, upstream/soma/src/capi/soma_c.cpp,
/// states for its own context struct).
///
/// HOW begin_step()/end_step() DRIVE THE REAL RUNNER (KNOWN_ISSUES.md D13).
///
/// `fuse::Runner::resume_period()` (include/fuse/runtime/runner.hpp,
/// implemented as a C++20 coroutine over the ordinary instruction-stream
/// walk -- src/runtime/resume.hpp, src/runtime/runner.cpp,
/// src/runtime/iteration.cpp) resumes the CURRENT period's instruction
/// stream and runs it up to the next externally driven component's own
/// `advance()` point, or to the period's end, whichever comes first. Every
/// component the graph declares is wrapped, at `initialize()`, in an
/// `ExternalStepComponent` (external_step_component.hpp) whose `advance()`
/// is a no-op -- the EXTERNAL caller writes that component's storage
/// directly, through the pointer it handed to `register_field()`, between
/// its own `begin_step()`/`end_step()` calls -- and whose
/// `is_externally_driven()` override (component.hpp) is what makes
/// `resume_period()` suspend there instead of calling it.
///
/// `fuse_begin_step(name)` calls `resume_period()` (or reads the request it
/// cached from the PREVIOUS `end_step()` call, below) and checks the
/// yielded request's own component against `name`: a match returns that
/// request's `StepContext` and checkpoint/restore flags directly --
/// EXACT, because by construction every instruction before this
/// suspension point has already run, so every channel feeding this
/// component has been exchanged and its `Requires` storage holds THIS
/// period's own delivered value, not a lagged one. A caller naming a
/// DIFFERENT component than the stream's own next tick is refused
/// `E-ABI-STEP-ORDER`, naming the component the stream actually expects;
/// the request stays cached (nothing was advanced) so the correctly-named
/// `begin_step()` that follows still sees it.
///
/// `fuse_end_step(name)` calls `resume_period()` again, resuming the
/// Runner past this component's own (no-op) `advance()` to the NEXT
/// suspension point or the period's end -- caching whatever it finds for
/// the NEXT `begin_step()` to read, exactly as above. Because the two
/// calls are 1:1 with resume_period()'s own suspend points, only ONE
/// externally driven component may be open (begin_step() called, end_step()
/// not yet) at a time on a given context: `fuse_ctx::open_component` is
/// that invariant, enforced as `E-ABI-STEP-ALREADY-OPEN`. This is a real
/// tightening versus the deferred, end-of-period scheme D12 replaced --
/// that scheme let every externally driven component's begin_step()/
/// end_step() pair interleave freely, because nothing it did actually
/// advanced the Runner until the last one reported in. The replacement
/// cannot offer that: the Runner is one shared, ordered instruction
/// stream, and there is exactly one "current suspension point" to resume
/// from. A process hosting more than one externally driven component must
/// now call begin_step()/end_step() for each in the order the compiled
/// schedule visits their ticks.
///
/// A component the test harness adds directly (fuse::abi_test::inject_component,
/// test_support.hpp) is driven the opposite way: it is a real
/// `fuse::Component`, added to the same `fuse::ComponentSet`, and is driven
/// IN-PROCESS -- through `fuse_run()`, which is `FUSE_ERR_UNSUPPORTED`
/// whenever any OTHER component in the graph is still the externally-driven
/// placeholder (entries.def's own words for `run`: "FUSE_ERR_UNSUPPORTED when
/// any component is driven from outside").

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuse/abi/types.h"

#include "fuse/backend/native_backend.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"
#include "fuse/transport/connect.hpp"

#include "registry.hpp"

namespace fuse::abi_detail {

/// One externally-driven graph component's bookkeeping. Indexed by
/// `fuse::ComponentId::value()`; absent (default-constructed, `in_process`
/// left true) for a component the test harness injected instead.
struct ExternalState {
  /// False for every component built this way at all -- the default when
  /// the vector is sized to `component_count()` and nothing has been
  /// decided for this slot yet. Set at `fuse_initialize()` time.
  bool in_process = false;
  /// True strictly between a successful `begin_step()` and the matching
  /// `end_step()` on this component. `fuse_ctx::open_component` names
  /// WHICH component this is, system-wide (at most one at a time -- see
  /// this file's own header comment); `open` here is redundant with it but
  /// kept so `resolve_external()`'s existing per-component checks read the
  /// same way they always did.
  bool open = false;
  /// The request the matching `begin_step()` reported, echoed back by
  /// `end_step()` (bounds, step, iteration, checkpoint/restore flags).
  fuse::ExternalAdvanceRequest request;
};

}  // namespace fuse::abi_detail

/// The context `fuse_ctx*` denotes. Opaque to every C caller; this
/// definition is visible only inside src/abi/*.cpp (and, via
/// test_support.hpp, to test/abi/*.cpp -- a test-only seam, never installed).
struct fuse_ctx {
  enum class State { Empty, Loaded, Initialized };

  State state = State::Empty;

  fuse::CouplingGraph     graph;
  fuse::FieldRegistry     fields;
  /// `fuse::native::NativeMeshBackend`, not `fuse::SerialMeshBackend`: a
  /// generated mesh needs the per-placement decomposition NativeMeshBackend
  /// supports (`register_default_placements`, include/fuse/backend/
  /// native_backend.hpp) -- a graph with an `add_coupler()`/`remap_on()`
  /// placement, or two components on different regions of the same mesh,
  /// cannot initialize through this ABI at all on SerialMeshBackend, which
  /// has no per-placement concept. `std::optional`, not a plain member:
  /// NativeMeshBackend has no default constructor (it needs this PROCESS's
  /// own `(rank, num_ranks)`, known only once `fuse_setup_options()`'s own
  /// communicator -- or its absence -- has been read, which happens at
  /// `fuse_initialize()` time, well after `fuse_ctx` itself is
  /// constructed); `impl_initialize` (participant.cpp) constructs it in
  /// place, exactly once, before the mesh-loading loop that is its first
  /// use. Found and fixed this session (Conformance Unit 2): the hardcoded
  /// SerialMeshBackend this replaced could not express the above at all.
  /// The full ABI suite needs re-running against this swap before it can
  /// be called verified -- track that against the ctest run this change
  /// is reported with, not this comment.
  std::optional<fuse::native::NativeMeshBackend> backend;

  /// S6 phase 4, "Adapter host" track: mounts every `LaunchMode::Plugin`
  /// component's shared object at initialize() time
  /// (fuse::abi_detail::Registry::mount_dynamic). DECLARED BEFORE
  /// `components` BELOW, DELIBERATELY: members are destroyed in REVERSE
  /// declaration order, and every ProviderAdapter a dynamic mount builds
  /// keeps calling through its plugin's table -- static storage inside the
  /// still-open shared object this registry's own destructor dlclose()s.
  /// Declaring `plugin_registry` here means it is destroyed AFTER
  /// `components` (and `runner`, which also sits between them for the same
  /// reason), so every ProviderAdapter has already run its own destructor
  /// (which calls table_->destroy() one last time) before the library
  /// backing that table is ever unloaded. Declaring it after `components`
  /// instead reproduces EXACTLY the use-after-unload
  /// ConformanceTest.GraphDeclaredPluginIsAutoMountedAndRun caught the hard
  /// way (a SIGSEGV inside ComponentSet's own destructor, confirmed with
  /// fprintf checkpoints and removed once the fix below was identified).
  fuse::abi_detail::Registry plugin_registry;

  std::optional<fuse::OperatorDAG>   dag;
  std::optional<fuse::ExecutionPlan> plan;
  /// Built at initialize() and handed to fuse::schedule() -- the schedule
  /// itself is derived from it, so it must stay alive for the Runner's
  /// whole lifetime even though begin_step()/end_step() no longer read it
  /// directly (KNOWN_ISSUES.md D13: resume_period()'s own StepContext is
  /// the real bounds now, not a separately bookkept copy of them).
  std::optional<fuse::ClockSet> clocks;
  std::optional<fuse::Runner>   runner;
  fuse::ComponentSet            components;

  /// Per-ComponentId bookkeeping for begin_step()/end_step()/run(), sized to
  /// `graph.component_count()` at initialize() and never resized after.
  std::vector<fuse::abi_detail::ExternalState> ext;
  /// How many of `ext` are NOT in_process. Cached at initialize() so run()
  /// does not rescan on every call.
  std::size_t external_count = 0;
  /// KNOWN_ISSUES.md D13: the request the most recent `resume_period()`
  /// call yielded and nobody has consumed yet -- set by whichever of
  /// `fuse_begin_step()` (priming the very first one, or re-reading one a
  /// mismatched name left behind) or `fuse_end_step()` (resuming past the
  /// component it just closed) called `resume_period()` last. `std::
  /// nullopt` means either nothing has been primed yet, or the period
  /// just ran to completion with nothing further to stop at; the next
  /// `fuse_begin_step()` calls `resume_period()` itself to find out which.
  std::optional<fuse::ExternalAdvanceRequest> next_request;
  /// The ComponentId currently between a successful `begin_step()` and its
  /// matching `end_step()`, or `std::nullopt` -- see this file's own
  /// header comment on why at most one may be open at a time.
  std::optional<fuse::ComponentId> open_component;
  std::int64_t periods_completed = 0;

  /// Components the test harness supplied directly
  /// (fuse::abi_test::inject_component), keyed by the graph component name
  /// they replace. Consumed (moved out) at initialize() time; empty
  /// afterwards.
  std::unordered_map<std::string, std::unique_ptr<fuse::Component>> injected;

  /// Unit 2b (D19), step (d): `fuse::abi_test::force_this_process`'s own
  /// override, consumed (passed into `fuse::RunnerConfig::
  /// force_this_process`) at initialize() time -- the SAME test-only
  /// seam `injected` is, for the SAME reason (`inject_component()`'s own
  /// header comment): no entry in the frozen public ABI lets a C caller
  /// name a rank without a real communicator, so this is how a tier-0
  /// test exercises `Runner::hosts_component()`'s own per-rank hosting
  /// rule without one. `std::nullopt` (the default) changes nothing --
  /// every existing caller that never calls `force_this_process` keeps
  /// meaning exactly what it always meant.
  std::optional<int> test_force_this_process;

  // --- setup() state, consumed at initialize() ---
  bool               communicator_set = false;
  std::int64_t       communicator = 0;

  /// What `fuse::transport::connect()` formed during setup(), when this
  /// context's own component is launched rather than in-process. Held so
  /// that destroy() can release it: a `Connect`-mode communicator leaves
  /// the two programs connected until `fuse::transport::disconnect()` runs,
  /// and `MPI_Finalize` is collective across connected programs. Default
  /// (`Teardown::None`) for every in-process caller, where connect() is
  /// never called and there is nothing to release.
  fuse::transport::ConnectedWorld connected;
  bool               has_end_time = false;
  fuse::Rational     end_time{0};

  // --- D61 (KNOWN_ISSUES.md) test-only seam ---
  /// fuse::abi_test::force_fake_collective_peers(): when set, reconcile_
  /// collective_status() (participant.cpp) uses THIS rank's position and
  /// the peer statuses below, through the SAME pure arithmetic
  /// (fuse::abi_detail::reconcile_local_statuses, include/fuse/abi/
  /// collective_reconcile.hpp) the real MPI_Allreduce(MPI_MINLOC) glue
  /// uses, instead of calling MPI -- the only way a tier-0 test (no MPI
  /// runtime, C1) can exercise the full reconciliation path, including
  /// the message a non-originating rank gets rewritten to. std::nullopt
  /// (the default) changes nothing: every context that never calls this
  /// seam reconciles for real (or not at all, with no communicator set).
  std::optional<int>            test_fake_collective_self_rank;
  std::vector<std::int32_t>     test_fake_collective_peer_statuses;

  // --- last_error() state ---
  std::string last_error_code;
  std::string last_error_message;
  /// The fuse_status the most recent failing call on this context returned
  /// (FUSE_OK until the first failure). fuse_error::status (types.h) is
  /// documented as "the fuse_status of the failed call"; this is what
  /// impl_last_error() (participant.cpp) now reads to honour that.
  fuse_status last_error_status = FUSE_OK;
};
