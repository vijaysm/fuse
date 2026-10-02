#pragma once

/// Private to src/abi/. `fuse_ctx` (the opaque type `include/fuse/abi/types.h`
/// only forward-declares as `struct fuse_ctx;`) is defined here, not in any
/// installed header: a C caller must not be able to see, size, or allocate
/// one (the same contract SOMA's `soma_ctx_s`, upstream/soma/src/capi/soma_c.cpp,
/// states for its own context struct).
///
/// HOW begin_step()/end_step() DRIVE THE REAL RUNNER WITHOUT CHANGING IT.
///
/// `fuse::Runner` interprets a flat instruction stream per coupling period
/// and has no public hook to pause mid-period while a single component's
/// `advance()` happens outside its own call stack -- adding one would be a
/// non-trivial addition to src/runtime, and the task that froze
/// `entries.def` asked for the smallest change that honours it. This file
/// adds none at all.
///
/// Instead: every component the graph declares is wrapped, at `initialize()`,
/// in an `ExternalStepComponent` (external_step_component.hpp) whose
/// `advance()` is a no-op that trusts the storage already holds the right
/// values -- because the EXTERNAL caller already wrote them, directly into
/// the pointer it handed to `register_field()`, during the window between
/// its own `begin_step()`/`end_step()` calls, which never touch
/// `fuse::Runner` at all. `begin_step()`/`end_step()` instead read and
/// advance a SEPARATE, purely bookkeeping `fuse::ClockSet` (the same
/// mechanism SOMA's C ABI uses for its own, exchange-free, begin_step/
/// end_step pair) built from the identical graph and therefore guaranteed to
/// agree with the schedule the real `Runner` will walk. Only once every
/// externally-driven component has completed as many ticks as the plan's
/// common period requires does `end_step()` call the real
/// `Runner::run_period()`, exactly once, for that period.
///
/// WHY THIS IS EXACT, measured rather than assumed (test/abi/graph_fixture.hpp
/// records the same measurement): `run_period()`'s Remap instruction reads
/// whatever a channel's source field currently holds when the instruction
/// runs, with no buffering of its own -- `lag(k)` (fuse/model/channel.hpp)
/// changes TimeLevel typing and accumulation-window bookkeeping, not WHEN a
/// write becomes visible to a Remap. So deferring `run_period()` to the
/// moment every externally-driven component has completed its step is
/// exact PROVIDED every Requires field is read no earlier than its OWN
/// component's end_step() -- never between a component's own begin_step()
/// and end_step(), where the period's run_period() call has not
/// necessarily happened yet (it happens inside whichever externally-driven
/// component's end_step() is the LAST to complete the period, which is not
/// knowable to the caller in general). test/abi/install_consumer/main.c and
/// test/abi/participant_test.cpp's value-checking cases both read a
/// Requires field only after their own end_step() for exactly this reason,
/// which is a narrower promise than entries.def's doc string for
/// begin_step literally states ("imports are current on return"); closing
/// that gap for every possible caller ordering is an open question in the
/// task report, not attempted here.
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

#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

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
  /// `end_step()` on this component.
  bool open = false;
  /// Clock ticks this component has completed since the last time every
  /// externally-driven component reached the plan's common-period boundary
  /// together (the point at which the real `Runner::run_period()` runs).
  std::int64_t ticks_this_period = 0;
  /// `plan().period() / clock.interval()`, fixed at `initialize()`: how many
  /// of this component's own ticks make up one common period. Always an
  /// exact integer -- `fuse::ClockSet::checked_common_period()` guarantees
  /// the common period is an integer number of ticks of every interval it is
  /// the LCM of.
  std::int64_t ticks_per_period = 1;
  /// The bounds the matching `begin_step()` reported, echoed back by
  /// `end_step()`.
  fuse::Rational t_begin{0};
  fuse::Rational t_end{0};
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
  fuse::SerialMeshBackend backend;

  std::optional<fuse::OperatorDAG>   dag;
  std::optional<fuse::ExecutionPlan> plan;
  /// The reporting-only clock begin_step()/end_step() read; see this file's
  /// header comment. Built fresh at initialize() from the same graph the
  /// real fuse::Runner below schedules, so the two are guaranteed to agree.
  std::optional<fuse::ClockSet> clocks;
  std::optional<fuse::Runner>   runner;
  fuse::ComponentSet            components;

  /// Per-ComponentId bookkeeping for begin_step()/end_step()/run(), sized to
  /// `graph.component_count()` at initialize() and never resized after.
  std::vector<fuse::abi_detail::ExternalState> ext;
  /// How many of `ext` are NOT in_process. Cached at initialize() so run()
  /// and end_step() do not rescan on every call.
  std::size_t external_count = 0;
  /// How many externally-driven components have reached their per-period
  /// tick count since the last real run_period(). Reset to 0 every time it
  /// reaches external_count, which is also when run_period() actually runs.
  std::size_t ready_for_period = 0;
  std::int64_t periods_completed = 0;

  /// Components the test harness supplied directly
  /// (fuse::abi_test::inject_component), keyed by the graph component name
  /// they replace. Consumed (moved out) at initialize() time; empty
  /// afterwards.
  std::unordered_map<std::string, std::unique_ptr<fuse::Component>> injected;

  // --- setup() state, consumed at initialize() ---
  bool               communicator_set = false;
  std::int64_t       communicator = 0;
  bool               has_end_time = false;
  fuse::Rational     end_time{0};

  // --- last_error() state ---
  std::string last_error_code;
  std::string last_error_message;
};
