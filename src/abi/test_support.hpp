#pragma once

/// A TEST-ONLY seam, never installed and not part of the frozen ABI
/// (include/fuse/abi/{entries.def,types.h,participant.h}): the plain C
/// participant ABI has no entry that registers an in-process advance
/// callback (deliberately -- see docs/dev/abi-design.md and the `run` entry's
/// own doc string in entries.def), so there is no way for a C caller, or an
/// ABI-level test using only the public table, to put a component whose
/// `advance()` throws into a running graph. This header is how
/// test/abi/participant_test.cpp's exception-containment case
/// (AThrowingComponentBecomesRuntimeErrorWithTheMessage) and
/// test/abi/negotiation_test.cpp's fault-injection case construct one
/// anyway: by reaching past the C surface, in C++, exactly the way those
/// test files already reach past fuse_coupler_abi() itself to build a
/// deliberately wrong table.
///
/// Only declared here; test/abi/*.cpp link src/abi/test_support.cpp
/// directly (not through the installed library), which is why this is a
/// free function and not a table entry.

#include <cstdint>
#include <memory>
#include <string>

#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"

struct fuse_ctx;

namespace fuse::abi_test {

/// Replaces the fuse::abi_detail::ExternalStepComponent that fuse_initialize()
/// would otherwise build for the graph component named `component_name` with
/// `c`, so that component is driven IN-PROCESS (through fuse_run(), never
/// through fuse_begin_step()/fuse_end_step()) instead.
///
/// Must be called after a successful fuse_load_graph() and before
/// fuse_initialize(); `component_name` must name a component the loaded
/// graph declares. Returns false (and leaves `ctx` unchanged) when `ctx` is
/// null, the context is not in the Loaded state, `component_name` is not a
/// component of the loaded graph, or a component has already been injected
/// under that name -- in every one of those cases `c` is destroyed.
bool inject_component(fuse_ctx* ctx, const std::string& component_name,
                      std::unique_ptr<fuse::Component> c);

/// Unit 2b (D19), step (d). Sets the per-rank identity `fuse_initialize()`
/// passes into `fuse::RunnerConfig::force_this_process`, so a tier-0 test
/// (no MPI, no `fuse_setup_options()` communicator) can exercise `Runner::
/// hosts_component()`'s own per-rank hosting rule -- and, through it,
/// `fuse_run()`'s own narrowed `E-ABI-RUN-EXTERNAL` refusal -- on a single
/// process, the same reason `inject_component()` exists: no entry in the
/// frozen public ABI has a way to say this.
///
/// Must be called after a successful `fuse_load_graph()` and before
/// `fuse_initialize()`. Returns false (and leaves `ctx` unchanged) when
/// `ctx` is null or the context is not in the `Loaded` state.
bool force_this_process(fuse_ctx* ctx, int rank);

/// KNOWN_ISSUES.md D57 ("Collectives" track): the Runner's own
/// instruction/conservation-check counters (fuse::RunStats, include/fuse/
/// runtime/runner.hpp), read directly off the context's own Runner -- no
/// entry in the frozen public ABI exposes them (deliberately: a debugging
/// aid, not a coupling-physics result the ABI's own surface is scoped to),
/// so a test that needs to tell "exactly N periods' worth of Runner work
/// happened" from "N periods were REPORTED (fuse_step_report::period) but
/// N+1 periods' worth of instructions already executed to produce that
/// report" (D57's own defect: fuse_begin_step()'s own `step_opened == 0`
/// at a period boundary describes the request the CALLER should act on
/// next, never how much of a FOLLOWING period's own stream prime_next_
/// request()'s rollover already ran to find it) has to reach past every
/// ABI report, all of which describe the next pending request, not the
/// Runner's own cumulative work.
///
/// Returns a default-constructed (all-zero) fuse::RunStats when `ctx` is
/// null or not yet Initialized -- never a null-dereference crash, the same
/// defensive posture `inject_component()`/`force_this_process()` already
/// take on a misused context.
fuse::RunStats run_stats(const fuse_ctx* ctx);

/// KNOWN_ISSUES.md D57 ("Collectives" track). Sets `fuse_ctx::
/// communicator_set`/`communicator` directly -- the state `fuse_setup()`
/// (src/abi/participant.cpp) would otherwise set from a NON-ZERO
/// `fuse_setup_options::communicator`, which that same function refuses
/// outright on a build with no MPI (`E-ABI-SETUP-NOMPI`) -- so a tier-0
/// test can exercise `prime_next_request()`'s own `c.communicator_set`
/// gate (the condition that distinguishes D57's own real row, where a
/// base communicator IS configured without an end_time, from
/// `AbiParticipant.MismatchedRolloverNamesTheRealPeriodStartComponent`'s
/// own single-process, no-communicator scenario, where the PRE-EXISTING
/// eager-continuation behaviour that test pins must keep working
/// unchanged) with no MPI runtime at all. Safe to do so: every MPI call
/// site `config.base_communicator.has_value()` can reach is itself
/// compiled out under `#if FUSE_HAVE_MPI` (absent at tier 0), so setting
/// this never risks calling into an uninitialized MPI library -- a
/// Constrain instruction reached with `ci.reduce` set instead takes
/// `resolve_constrain()`'s/`execute()`'s own `#else` arm and returns a
/// clean `E-RT-CONSERVATION-REDUCE` Result::Failure (runner.cpp), never a
/// crash or a hang, which is exactly the decisive, safe signal this
/// track's own mutation proof (reverting the fix) relies on.
///
/// Callable any time after a successful `fuse_setup()`, in EITHER the
/// `Loaded` or the `Initialized` state -- unlike `force_this_process()`,
/// which `fuse_initialize()` itself consumes and so must precede. Calling
/// it AFTER `fuse_initialize()` (once the Runner already exists with
/// whatever transport that call picked) is the usual choice: it reaches
/// `prime_next_request()`'s own `c.communicator_set` read -- a plain
/// context field with no other run-time consumer -- without also
/// flipping `fuse_initialize()`'s own "mpi" vs "copy" transport choice,
/// which fails outright on a build with no "mpi" backend registered
/// (tier 0, `FUSE_HAVE_MPI` off) regardless of whether the graph's own
/// channels ever need a real Transport exchange. Returns false (and
/// leaves `ctx` unchanged) when `ctx` is null or in neither state.
bool force_communicator_set(fuse_ctx* ctx, std::int64_t value);

/// Restart track round 2, item 4's own C consumer test
/// (test/abi/conformance_test.cpp): the plain C participant ABI has no
/// entry that reads Runner::take_restart_point_request() back out --
/// that call is the EMBEDDING HOST's own job, never a plugin's (that
/// method's own doc comment, fuse/runtime/runner.hpp) -- so a test
/// proving fuse_request_restart_point() reached the real Runner needs
/// this seam to reach past the ABI and ask it directly. Returns nullptr
/// if `ctx` is null or the context has not reached Initialized (no
/// Runner exists yet).
fuse::Runner* runner_of(fuse_ctx* ctx);

}  // namespace fuse::abi_test
