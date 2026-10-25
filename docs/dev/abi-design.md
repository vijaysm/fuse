# Why the C ABI is one symbol

The usage side of this is in [](../user/c-abi.md). This page is the reasoning, kept because the
decision looks arbitrary until you have hit the alternative.

## The constraint

FUSE needs a C ABI for two opposite reasons: a **participant ABI** so a foreign code with its own
time loop can drive the coupler, and a **provider ABI** so the coupler can drive a plugin. C symbols
are unmangled and global, and the natural names collide. libfuse (Filesystem in Userspace) is
installed on essentially every Linux and HPC system and owns `fuse_*`, including `fuse_semantic_version`,
`fuse_new`, `fuse_loop`, `struct fuse_args`, `struct fuse_buf` and `struct fuse_chan` — the last
three being names a coupler would plausibly want. Any process linking both is exposed, and symbol
interposition makes the failure mode a *wrong call*, not a link error.

## Two answers we rejected

**A prefix** (`fusec_begin_step`). Correct and dull. It renames roughly fifty symbols to fifty other
symbols: the collision surface is unchanged in size, only in spelling, and every entry point carries
a wart forever.

**A macro** (`CC(fuse_)` expanding to `c_fuse_*`). Rejected on four counts. `CC` is the C compiler
variable in every Makefile and a macro in several libraries, so defining it in a public header is
hostile. Users cannot grep and IDEs cannot complete. `nm`, `gdb`, `perf` and every stack trace show a
name that appears nowhere in the source. And it still produces fifty exported symbols — it solves the
spelling, not the surface.

## What we do instead

Export **one** symbol per ABI. Everything else is a member of a versioned struct of function
pointers.

```c
const fuse_participant_abi_v1* fuse_coupler_abi(uint32_t abi_version);
```

Six consequences, in rough order of how much they matter:

1. **The collision surface is one name, and it is unmistakable.** Built with
   `-fvisibility=hidden`, the library exports that one C entry point and nothing else: the whole ABI
   is auditable with a single `nm` invocation, and there is no accidental ABI to freeze by mistake.
2. **Members get short, prefix-free names**, because the struct is the namespace.
   `fuse->begin_step(...)` reads better than either rejected spelling.
3. **Versioning is the mechanism, not a convention.** A `struct_size` header plus append-only growth
   is exactly the negotiation the upstream provider ABI already implements and mutation-tested, so
   both ABIs share one mechanism instead of having two.
4. **`dlopen` becomes one `dlsym`**, which the dynamic-provider path needs anyway.
5. **No macros in public headers.** The single-source-of-truth trick lives where macros belong: an
   *internal* entry list generates the table initialiser and the `struct_size` assertion, and a golden
   test asserts that the public header's member list matches that list exactly — the same golden-file
   discipline the IR uses. The two cannot drift, and the header a user reads is plain documented C.
6. **Testability.** A test substitutes its own table to inject a fault at one entry point, with no
   linker tricks and no rebuild.

## The costs

Stated here rather than discovered later:

- One extra line to fetch the table before first use.
- **No direct linking against individual entry points.** A build system that wants
  `-Wl,--undefined=fuse_begin_step` cannot have it. A thin optional shim library forwarding real
  symbols to the table exists for codes that need it — opt-in, and never the default.
- Function pointers do not inline across the boundary. Irrelevant here: every entry point is a setup
  or step-boundary call, never per-degree-of-freedom, and `I5` restricts dispatch only in hot kernels.
- **Fortran cannot dereference a function-pointer struct ergonomically.** It is possible
  (`type(C_FUNPTR)` plus `C_F_PROCPOINTER`) but unpleasant, and Fortran is the one audience for which
  flat symbols are genuinely better. So the Fortran shim keeps a small set of exported `bind(C)`
  symbols, generated from the same entry list. A deliberate exception with a stated reason, not a
  second ABI.

C++ callers never see any of this; `fuse::Graph` and `fuse::run` are the primary surface.

## How arguments cross the boundary: structs, and the rules for growing them

The design requirement is that new arguments never break the API. Every entry point has one shape,

```c
fuse_status name(fuse_ctx* ctx, const IN* in, OUT* out);
```

and `IN`, `OUT` are argument structs from `include/fuse/abi/types.h` whose first member is
`uint32_t struct_size`. The caller sets it to `sizeof` of the struct it was compiled against. The
rules, each of which a test enforces:

1. **Members are appended, never inserted, reordered, retyped or removed.** The order in
   `include/fuse/abi/entries.def` is the ABI. (Golden: the header is regenerated from the `.def`
   and must be byte-identical; `tools/gen-abi.py --check`.)
2. **The callee reads only the members `struct_size` covers**, and takes the documented default
   for the rest. A `struct_size` smaller than the first frozen layout, or larger than
   `FUSE_ABI_MAX_STRUCT_SIZE`, is `FUSE_ERR_VERSION`. (Mutation rows: a reader that trusts a short
   struct; a writer that sets the wrong size.)
3. **An appended member must increase `sizeof`.** This is the rule the first size table exposed:
   `fuse_version_info` has 28 bytes of members in 32 bytes of storage, so a 4-byte member appended
   there would change the layout *without* changing `struct_size`, and rule 2 would read the new
   member from an old caller's padding. So a member that would fit in trailing padding is instead
   declared together with an explicit `reserved` member that pushes `sizeof` up, or widened to
   64 bits. (Golden: `test/abi/struct_sizes.golden`, one line per struct, with
   `sizeof(void*) == 8` asserted; a change to the table is a deliberate, reviewed commit.) The
   current table, literalincluded so this page cannot drift from it:

   ```{literalinclude} ../../test/abi/struct_sizes.golden
   ```
4. **Value types never grow.** `fuse_exact_time` and `fuse_semantic_version` carry no `struct_size`; they are
   closed, and a new kind of time would be a new type.
5. **Nothing is allocated across the boundary.** Buffers (error messages, graph bytes) are the
   caller's, sized by a length query; strings are NUL-terminated and valid for the call only;
   field storage (`fuse_field_view.data`) is the caller's and must stay valid until `destroy`.
6. **Time is exact.** A step's bounds cross as `fuse_exact_time` (numerator, denominator), never as
   a double; a caller that needs a double computes it, and the agreement on where a step begins
   and ends is not lost in the rounding.
7. **Collective entries are marked in the `.def`** (`setup`, `initialize`, `begin_step`,
   `end_step`) and must be called by every process of the stated communicator in the same order
   relative to its other collectives — the same contract `MoabMeshBackend::set_communicator`
   states.

## The graph descriptor's byte order

The coupling-graph descriptor (`fuse::serialize`/`fuse::deserialize`, `include/fuse/model/
descriptor.hpp`) is also described as "the ABI narrow waist", but it is a separate encoding
from the C-ABI struct rules above and does not follow them in how it crosses the C ABI's own
narrow waist (`fuse_graph_source`/`fuse_graph_sink`, `include/fuse/abi/entries.def`) -- raw
bytes, not a `struct_size`-negotiated struct. It DOES now follow `bytes.hpp`'s own canonical
encoding, as of `kGraphVersion` 15.

**Earlier behaviour.** `detail::Writer::pod`/`detail::Reader::pod` used to write a value's native
in-memory representation via `memcpy` -- host-endian, not the little-endian rule
`include/fuse/base/bytes.hpp` states for every OTHER canonical encoding in the library (the
field-entry key, the semantic model, the operator chain). This was undocumented and untested;
every machine this library had built on was little-endian, so the gap was never visible. The
descriptor stayed host-endian and same-host-family only until a version bump routed it through
`bytes::put_integer` deliberately, at a point where the golden-file churn that bump causes was
expected. A big-endian host building and exchanging descriptors with a
little-endian peer would have silently disagreed about every multi-byte field.

**The current encoding.** `Writer::pod`/`Reader::pod` now route every POD through `bytes::
put_integer`/`bytes::get_integer` (unsigned), `bytes::put_signed`/`bytes::get_signed` (signed,
two's-complement), or `std::bit_cast` to/from an equal-width unsigned integer through the same
pair (floating-point -- only `double` appears in this file). `bytes.hpp` gained `get_integer`/
`get_signed` as part of this change: before it, it was write-only (used for canonical hashing in
`src/backend/weights_operator.cpp`, `src/model/semantics.cpp`, `src/model/transform.cpp`), with
no decode side at all, because nothing round-tripped through it until the descriptor did.
`kGraphVersion` went from 14 to 15. No field changed width or position: on a little-endian host
(every host this library has ever built on) the wire bytes for a given graph are unchanged
except for the version word itself -- the change is to the GUARANTEE (host-independence), not
the bytes a little-endian host happens to produce, which is exactly why the bug was invisible
here and would not have been on a big-endian host. `test/model/descriptor_test.cpp`'s
`V14AndV15BytesOfTheSameGraphDifferOnlyInTheVersionWord` and
`V14AndV15BytesOfTheSameGraphDecodeIdentically` verify this directly; this sandbox has no
big-endian host to build or run on, so `PodBytesNoLongerDependOnHostByteOrderViaSwapStub`
demonstrates the host-independence by construction (a byte-swapping writer stub standing in for
the superseded host) instead of by executing on one.

The ABI's own negotiation of this version is unchanged: `fuse_load_graph()`
(`src/abi/participant.cpp`) hands raw bytes straight to `fuse::deserialize()`, which refuses a
`kGraphVersion` mismatch by name (`E-DESC-VERSION`) -- `test/abi/negotiation_test.cpp`'s `LoadGraphRefusesAnOutdatedDescriptorVersionByName` exercises
this through the ABI boundary explicitly. The restart-to-disk provenance check gained its own,
independent version/hash distinction for the SAME reason -- see
`fuse/restart_io/hdf5_restart_io.hpp`'s `Provenance::descriptor_version` /
`check_restart_descriptor_hash()`.

## How the participant ABI reconciles a collective refusal

`entries.def` marks `setup`, `initialize`, `begin_step`
and `end_step` COLLECTIVE in their own doc strings (`setup` "COLLECTIVE
over the communicator"; `initialize` "COLLECTIVE"; `begin_step`/`end_step`
"COLLECTIVE over the component's processes"), and `run`'s own doc string,
while not using that literal word, states it "still executes this call's
own share of the period's collectives" -- the same reason `run`
is grouped with the other four. Each of the five would otherwise decide a
refusal from this rank's own local state only (a bad `struct_size`, the
context in the wrong state, an out-of-turn call) and return, with
nothing agreeing that decision across the rest of the component's ranks.
On a multi-rank component this is exactly the hazard `docs/dev/
invariants.md`'s `A4` names for any terminal that does IO or reduction: "a
rank-subset invocation is a refusal -- reconciled through `Epoch` so that
the refusal itself cannot hang." Without reconciliation the ABI boundary would meet every OTHER part of
that contract and none of this one.

**Mechanism.** `src/abi/participant.cpp`'s `reconcile_collective_status()`
runs after the entry's own local body has computed its verdict and before
that verdict reaches the caller: every rank of the communicator being
reconciled over contributes its own `(fuse_status, rank)` pair to one
`MPI_Allreduce` with `MPI_MINLOC` -- the one-call form of "MIN over an
encoded status" that also reports which rank the minimum came from, which
is what lets a non-originating rank's message NAME the rank that refused.
`FUSE_OK` is 0 and every `FUSE_ERR_*` code is negative (`entries.def`), so
the plain minimum already propagates any refusal over `FUSE_OK`; two
different ranks refusing with two different codes agree on the more
negative one, deterministically. The rank whose own local status equals
the agreed one keeps its own message unchanged; every other rank's
message is rewritten to name the refusing rank and the agreed code, per
this page's own standing rule that every rank must return the SAME code,
not merely a code drawn from the same enum.

**Why not the base communicator.** Reconciling over the
context's own base communicator (`c.communicator`) -- the MERGED world of
every component in a multi-executable run -- hangs. Measured on the host:
`verification.brusselator.participants.{mpmd,accept}.{3,5}` all timed
out, because U, V and the coupler each call `begin_step`/`end_step` a
DIFFERENT number of times per window, so an `MPI_Allreduce` keyed to "all
ranks of the base communicator" never finds the same NUMBER of
contributions on both sides of the reduction -- the collective itself
never completes. The general rule this measured: **a collective that
pairs calls ACROSS components can only run on a communicator whose every
member makes the SAME sequence of calls** -- the base communicator is the
wrong scope for anything keyed to one component's own call sequence,
exactly as it would be wrong to run an `MPI_Allreduce` over `MPI_COMM_
WORLD` for a quantity only some ranks compute. The reconciliation
therefore runs over a component-local communicator, precisely to keep every member of the
reduction making the identical sequence of calls.

The pure arithmetic -- given every rank's own local status, which one
wins and whose message survives -- lives in `fuse::abi_detail::
reconcile_local_statuses` (`include/fuse/abi/collective_reconcile.hpp`),
with no MPI in it, so `test/abi/collective_reconcile_test.cpp` fires every
branch (all agree, one refusal among many, two different refusals, a tie)
at tier 0. `src/abi/participant.cpp`'s own glue calls that same function
for its test-only stub-reducer path (`fuse::abi_test::
force_fake_collective_peers()`, `test/abi/participant_test.cpp`) and
calls the real `MPI_Allreduce(MPI_2INT, MPI_MINLOC, ...)` otherwise, so
the two paths cannot disagree about what "agreed" means.

**Which communicator.** `initialize`/
`begin_step`/`end_step`/`run` reconcile over `fuse_ctx::collective_comm_f`
-- a COMPONENT-LOCAL communicator, split exactly once, inside
`initialize()`'s own already-existing `if (c.communicator_set)` block, by
a direct `MPI_Comm_split(base, colour, key, &sub)` call: `colour` is this
rank's own `my_component`'s `ComponentId::value()` (a distinct integer
PER COMPONENT) when `my_component` is set, `MPI_UNDEFINED` otherwise;
`key` is this rank's own position in that component's declared `RankSet`
(`fuse::abi_detail::decide_split_colour_key`, `include/fuse/abi/
collective_reconcile.hpp` -- the pure half, tested at tier 0 with no MPI
dependency, the same split this header's other half already gets for the
reconciliation arithmetic). The split deliberately does not go through
`fuse::linalg::make_participant_set_reducer()`, for the reason given below.
`0` means "nothing to reconcile": the default; whenever `my_component`
was never named (no component to split by); and, BY CONSTRUCTION, for
every context `fuse::abi_test::force_communicator_set()` touches, since
that seam sets `communicator`/`communicator_set` directly and never
re-runs the real split path (its own doc comment: deliberately callable
AFTER a successful `initialize()`, to reach a LATER gate in isolation).
`setup()` is NOT reconciled at all -- `my_component` is
itself an ARGUMENT to `setup()`, not yet resolved when its own local
checks (`struct_size`, `reserved`, an unknown `my_component` name) can
fail, so there is no component-local communicator for `setup()` to
reconcile over yet, and reconciling it over anything ELSE (the base
communicator) is exactly the mistake described above. `setup()`'s own refusals are structural, decided from
the same bytes on every rank that is meant to agree, independent of
reconciliation.

**Why not `make_participant_set_reducer()`.** Calling it for the
colour decision with THIS rank's
OWN component's `RankSet::ranks()` as the "participant set" -- a function
written for a caller where every rank of a SHARED base communicator
passes the IDENTICAL set and asks "is my rank a member" (colour `0` if
yes, `MPI_UNDEFINED` if no). Every rank here passes a DIFFERENT
component's own set, so every rank trivially answers "yes" about its OWN
component and computes the SAME colour `0` -- landing every component of
a multi-component graph on ONE shared communicator, the identical failure
mode as reconciling over the base communicator, reached by a different
route: the rule ("a collective that pairs calls across
components can only run on a communicator whose every member makes the
same sequence of calls") is true of the SPLIT CALL that BUILDS the
communicator, not only of the reconciliation call that later runs on it.
A colour that is the same constant for every caller, regardless of which
set that caller passed, cannot separate anything -- `MPI_Comm_split`
needs a colour that VARIES with which group a rank belongs to, which is
why the split colours by the component's own identity (`ComponentId::
value()`, distinct per component by construction) rather than by a
yes/no answer that is always yes about one's own set.

**What is not reconciled, and why.** `setup()` (above), and the
other ten participant-table entries (`version`, `capabilities`, `create`,
`destroy`, `last_error`, `load_graph`, `save_graph`, `field_handle`,
`register_field`, `view_field`, `base_communicator`,
`request_restart_point`) are none of them marked COLLECTIVE in
`entries.def`, and each is a purely local operation on this rank's own
handle, this rank's own copy of the (identically loaded) graph, or this
rank's own storage -- `register_field` included: registering a field is
this rank's own storage handed to this rank's own copy of the graph, with
no cross-rank agreement meaningful for it. There is no `finalize` entry
in `entries.def` at all; `destroy` is the nearest analogue and, per its
own doc string ("a double destroy is `FUSE_ERR_INVALID`, not a crash"),
is a per-handle operation, not a collective one -- the same posture
SOMA's own `soma_destroy()` takes.

**What this does not cover.** A rank that is live but calls a DIFFERENT
entry than its peers on the same step (so `with_ctx()` itself refuses
`E-ABI-HANDLE` before any body runs on a null or already-destroyed
handle) has no communicator this reconciliation can read -- there is no
live `fuse_ctx` to ask. That is a caller confusing which handle or which
entry to call, not a refusal this entry's own body decided, and is
outside what this reconciliation covers; it is the residual exposure.

**Cost.**
`begin_step`/`end_step` are step-boundary calls (this page's own "costs"
section: "every entry point is a setup or step-boundary call, never
per-degree-of-freedom"), so one small `MPI_Allreduce` added to each is
bounded, not asymptotic -- but it is one more real collective per step,
beyond whatever `Transport`/`Epoch`/`Constrain` instructions the compiled
plan already runs for that step. This is the standing
cost. A possible cheaper mechanism would piggyback the encoded status
onto a collective the step already performs instead of a separate
`MPI_Allreduce`; it is not implemented.

## The period contract (what a window-per-period boundary must not do)

## How a plugin's mount is agreed across the base communicator

`fuse::abi_detail::Registry::mount_dynamic()` dlopen()s a `LaunchMode::Plugin` component's shared
object only on the ranks that host it, so a plugin-side refusal (`E-ABI-LOAD`,
`E-ABI-VERSION`, `E-ABI-DESCRIBE`, a device gate) is visible only to a hosting rank. This would violate the SAME standing rule the section above
states ("agree first, then every rank refuses with the same code"), for a reason that section's own
mechanism cannot reach: `reconcile_collective_status()` reconciles over `fuse_ctx::
collective_comm_f`, a communicator split by `my_component`, and `my_component` is documented
(`entries.def`) to behave as unset for `InProcess`/`Plugin` launch -- so `collective_comm_f` never
forms at all for a Plugin-only, direct-communicator graph, the exact shape this problem occurs in.

**Mechanism.** `src/abi/participant.cpp`'s `reconcile_plugin_agreement()` reconciles the plugin-mount
verdict `impl_initialize_local()`'s own component-wiring loop computes (a hosting rank's own
mount/extract/register-fields result, or a non-hosting rank's trivial success) across every rank of
the BASE communicator (`fuse_ctx::communicator`) -- not `collective_comm_f` -- with one
`MPI_Allreduce(MPI_2INT, MPI_MINLOC, ...)`, run exactly once per `fuse_initialize()` call and ONLY
when the loaded graph declares at least one `Plugin` component (a graph with none never pays this
cost at all). It reuses the identical pure arithmetic (`fuse::abi_detail::
reconcile_local_statuses`) the section above's own glue calls, so the two reconciliations -- over
two different communicators, for two different questions -- cannot disagree about what "agreed"
means.

A hosting rank's own mount/extract/register failure in the component-wiring loop's `Plugin` branch
records its verdict and `break`s the loop instead of returning early, so every rank -- hosting or
not -- reaches this one reduction point before `impl_initialize_local()` returns; an early return
would let a hosting rank that failed leave before a non-hosting rank (which has nothing to fail on
locally) learns about it.

**Why this cannot deadlock the way a base-communicator reconciliation of the step entries would.**
The rule stated above ("a collective that pairs calls ACROSS components can only run on a communicator whose every
member makes the SAME sequence of calls") is about `begin_step`/`end_step`/`run` -- entries a
multi-component graph's different components call a DIFFERENT NUMBER OF TIMES, so an
`MPI_Allreduce` keyed to "all ranks of the base communicator" never finds the same number of
contributions on every side. `fuse_initialize()` has no such cadence: it is called EXACTLY ONCE,
synchronously, by every rank of the base communicator, before any step boundary exists at all --
there is no "which call number is this" question for two ranks to disagree about. The other half of
what makes that variant unsafe -- touching a communicator handle `fuse::abi_test::
force_communicator_set()` faked, with no real MPI behind it, under a real `FUSE_HAVE_MPI` build --
is avoided the same way the pre-existing `rank`/`num_ranks` derivation at the top of
`impl_initialize_local()` already relies on: every test that fakes a communicator this way is
excluded from MPI-enabled test binaries by convention (`test/abi/conformance_test.cpp`'s own
`NonHostingRankNeverDlopensThePluginOrCallsAnyOfItsEntries` is the
precedent the plugin-agreement tests follow).

**A priori refusal for the one launch shape this cannot cover.** Agreeing a plugin-mount verdict
needs a communicator that spans every rank with an opinion on it. A multi-executable launch where
one component is `Plugin` and another is `Connect` (true separate-executable MPMD -- independent
MPI universes until `fuse::transport::connect()`'s own handshake runs) asks a process whose own
`my_component` never calls `connect()` (unset, or `Plugin`/`InProcess`/`LowFive`) to reconcile over
a communicator that was never grown to include the `Connect`-launched executable -- `communicator`
is exactly what `fuse_setup_options` handed it, unmodified. `include/fuse/abi/plugin_agreement.hpp`'s
`plugin_mount_agreement_needs_a_missing_spanning_communicator()` is a PURE function of the loaded
graph and this process's own `my_component` (no MPI call, no collective: every process decides the
SAME refusal from the SAME graph bytes, so there is nothing to reconcile and no rank left to hang
waiting for a peer) that `impl_initialize_local()` consults before any of the component-wiring loop's
work runs, refusing `E-ABI-PLUGIN-NO-SPANNING-COMM` a priori rather than letting the plugin be
dlopen()'d first. `Mpmd` is exempt (`MPI_COMM_WORLD` already spans every program before
`fuse_setup()` is called, with no `connect()` needed for that to be true); `communicator_set ==
false` is exempt (single process by construction, nothing else to span). NOT fully modelled,
recorded rather than guessed past: a graph with more than one independent `Connect` pairing, where
this process's own `my_component` connects to one pairing while the `Plugin` component is hosted in
a DIFFERENT one, is not distinguished from the safe case -- the fully general version would need
`fuse::transport::validate_launch_layout()`'s own topology check consulted before this point, which
this function deliberately does not attempt; it is an open question.

Tested at tier 0 with no MPI runtime (`test/abi/plugin_agreement_test.cpp` for the pure predicate;
`test/abi/conformance_test.cpp`'s own `PluginAgreementReconciliationNeverRunsForAGraphWithNoPlugin
Component`, `NonHostingRankAdoptsAHostingPeersPluginMountFailureThroughTheBaseCommunicatorReduction`,
`HostingRanksOwnPluginMountFailureStillReachesTheBaseCommunicatorReduction`, and
`APluginAlongsideAConnectComponentWithNoSpanningCommunicatorIsRefusedAPriori` for the real-ABI
wiring, the last two via a DEDICATED fake-peer seam, `fuse::abi_test::
force_fake_plugin_agreement_peers()`, kept separate from the one above so the two
reconciliations can be faked independently); the real multi-process row,
`parallel.provider_subset_mpi`'s own `AHostingRanksPluginMountFailureIsAgreedByEveryRankNotOnlyThe
HostingOnes`, is built under a real `FUSE_ENABLE_MPI=ON` configure but has not been run in CI.

## The period contract (what a window-per-period boundary must not do)

A caller of the participant ABI, or of `fuse::
Runner::run()` directly, is entitled to ask for a COUNT of coupling
periods -- never a promise that state has "settled" past that count, even
on a graph whose window-scoped region reaches a fixed point every period.
`fuse::Runner::run_period()`'s own doc comment (`include/fuse/runtime/
runner.hpp`) states the Runner's half of this contract in full; the
ABI's own half is `src/abi/participant.cpp`'s `prime_next_request()`
(called from `fuse_begin_step()`): when the current period has just
drained with no end time declared AND a base communicator is configured,
it must stop and leave starting the FOLLOWING period's own first request
to a LATER, separate caller-initiated call -- never resume the stream
again inside the SAME call that just closed the prior period.

**Why this is the ABI's problem, not just the Runner's.** Resolving a
period's own first request ahead of being asked for it is harmless on a
single process (there is no partner rank to disagree with). It stops
being harmless the moment a base communicator is configured: a driver
with its own bound on how many periods to run (an ABI caller's own `for
(p = 0; p < periods; ++p)` loop, or `Runner::run()`'s own `for (i = 0; i
< periods; ++i) run_period();`) stops EXACTLY at that bound and never
calls in again, while a caller with no such bound (nothing in
`fuse_setup_options` requires one) may resolve and run a further period
regardless. A real multi-rank row measured the consequence: U and V, both
driven through the ABI with no end time declared, each continued a THIRD,
unrequested period's own base-communicator `Constrain` reduction after a
converged window reached its fixed point at the end of period 2, while
the in-process coupler -- driven through `run(components, 2)`, which
executes exactly 2 periods and returns -- had already left. U and V then
hung in that `MPI_Allreduce`, each waiting for a peer that would never
call in again.

**Enforced by** `test/verification/abi_period_boundary_test.cpp`'s
`AbiPeriodBoundary.NeverRunsAPeriodTheCallerDidNotRequestOnceStateHasRea
chedAFixedPoint`, a tier-0 (no MPI) test through the real public ABI,
using `fuse::abi_test::force_communicator_set()` to put a single-process
graph into the `communicator_set == true` state the real, multi-rank row
is actually in -- the same seam `test/verification/
brusselator_period_rollover_test.cpp` (the earlier, Brusselator-
specific pin) already uses. The decisive assertion in both: at the call
where the loop's own tracked period count first reaches the requested
count, `fuse_step_report::next_component` must equal -1 -- the ONLY value
`fuse_begin_step()` reports when nothing is genuinely cached for a
further period; any other value, even with `step_opened == 0`, means a
request for an UNREQUESTED further period was already resolved.

## Fortran: `ffuse`

Fortran gets flat `bind(C)` procedures rather than the table (the exception stated under "The
costs"). The whole module is generated by `tools/gen-abi.py` from the same `entries.def`, so it
cannot disagree with the C header. The mapping rules:

| In `entries.def` | In Fortran | Note |
|---|---|---|
| argument struct `fuse_x` | `type, bind(C) :: fuse_x`, members in the same order | `C_SIZEOF` of the Fortran type must equal C `sizeof`; tested for every struct |
| `uint32_t`, `int32_t` | `integer(c_int32_t)` | unsigned fields hold values below 2^31 by construction (`FUSE_ABI_MAX_STRUCT_SIZE`); documented |
| `int64_t`, `uint64_t` | `integer(c_int64_t)` | |
| any pointer | `type(c_ptr)` | `c_loc()` of a `target` array; the caller keeps it alive as the C rule says |
| `fuse_exact_time`, `fuse_semantic_version` | `type, bind(C)` of the same name | closed value types |
| `fuse_ctx*` | `type(fuse_ctx)` wrapping one `type(c_ptr)` | opaque |
| entry `name` | module procedure `fuse_name(ctx, in, out)`; link symbol `ffuse_name` | see below |
| a type whose name equals an entry's `fuse_name` | refused by the generator | Fortran has one namespace for types and procedures; the type is renamed in the `.def` (hence `fuse_capability_set`, `fuse_setup_options`, `fuse_semantic_version`) |
| `FUSE_OK`, `FUSE_ERR_*`, `FUSE_ELEMENT_*`, ... | `integer(c_int32_t), parameter` of the same name | |

- **Link names are `ffuse_*`, never `fuse_*`.** The Fortran-visible names *are* `fuse_*`, because a
  module procedure name is not a link symbol; the `bind(C, name="ffuse_...")` symbol that the
  Fortran interface references is exported from a separate library, `libfuse-abi-fortran`, so
  `libfuse-abi` keeps exactly one exported symbol and the `ffuse_*` family cannot collide with
  libfuse's `fuse_*`.
- **`struct_size` is set by the module, not the user.** Each generated `fuse_name` procedure sets
  `in%struct_size = int(c_sizeof(in), c_int32_t)` (and the same for `out`) before the call. A
  Fortran user never sees the mechanism, and a Fortran program compiled against an older module
  keeps running against a newer library for the same reason a C one does.
- **Defaults are default initialisation.** Every argument struct's Fortran type initialises each
  member to the documented default (`0`, `c_null_ptr`, `fuse_exact_time(0, 1)`), so a user who sets
  only the members they need passes exactly what a C caller passing a zeroed struct would.
- **Strings.** Procedures that take a name accept `character(len=*)`; the generated code copies it
  into a NUL-terminated `character(kind=c_char)` array that lives for the call and passes its
  address. The error message comes back the same way: `fuse_last_error(ctx, message)` fills a
  Fortran `character(len=*)` and reports truncation through the struct's own flag.
- **Communicators** are the Fortran MPI handle, passed as `integer(c_int64_t)`; on the C side
  `MPI_Comm_f2c` recovers it, which is the convention `set_communicator` already uses everywhere in
  this library.
- **Optional behaviour is a struct member, never an optional argument.** Fortran `optional` would
  put the argument *list* in the ABI again; a member with a default costs nothing and grows by the
  rules above.
- **Status is the function result**, `integer(c_int32_t)`, compared against the `FUSE_*` parameters.

What this costs: one more generated file and one more library to install. What it buys: a
Fortran program is written against the same fifteen names and the same structs as a C program,
and the test that proves the two agree is a `C_SIZEOF` table, not a reading.

## Provider direction

The participant ABI above is the coupler being driven; the
provider ABI (`include/fuse/abi/provider.h`, user docs at [](../user/providers.md)) is the
opposite: the coupler drives a plugin, loaded at runtime from a shared object a graph names by
path (`LaunchMode::Plugin`, `include/fuse/model/launch.hpp`).

**One struct catalogue, two tables.** `entries.def` grew a second X-macro,
`FUSE_ABI_PROVIDER_ENTRY`, generating `fuse_provider_abi_v1` (`tools/gen-abi.py`'s `gen_provider`)
alongside the existing `FUSE_ABI_ENTRY` / `fuse_participant_abi_v1`. Both tables read from the
SAME `FUSE_ABI_STRUCT` list above them: `fuse_field_view` is the identical bytes on either side of
the coupler, and `fuse_step_request`/`fuse_step_report` — grown with the window fields the
provider direction needs — carry the SAME exact-time shape whichever direction is doing the
carrying. This is deliberate and is the whole point of one `entries.def`: a field or a time bound
cannot drift between the two directions because there is only one declaration of what either
looks like.

**A different opaque handle, on purpose.** The provider table's `self` parameter takes THE ONE
RULE's `ctx` slot, but is typed `fuse_provider_ctx*`, not `fuse_ctx*` — a distinct forward-declared
opaque struct, so a coupler's own context and a plugin's own instance handle cannot be passed to
the wrong table by accident, even though both are plain pointers to the compiler.

**Versioning is the SAME mechanism, applied to the table itself as well as to its structs.** A
plugin's `fuse_provider_abi(abi_version)` returns `NULL` for a version it does not speak, exactly
as `fuse_coupler_abi()` does; `fuse::abi_detail::Registry::mount_table()` additionally checks the
returned table's own `struct_size` and `abi_version` fields before trusting any member pointer in
it, refusing both failure modes with `E-ABI-VERSION` — the struct_size negotiation this file's own
"How arguments cross the boundary" section specifies for argument structs is the identical check
applied one level up, to the table that carries them.

**The adapter and the registry are NOT part of the frozen ABI.** `fuse::abi_detail::ProviderAdapter`
(`src/abi/provider_adapter.cpp`) wraps a `fuse_provider_abi_v1*` as an ordinary `fuse::Component`,
so `fuse::Runner` drives a plugin through the identical in-process path it drives any native
component through; `fuse::abi_detail::Registry` (`src/abi/registry.cpp`) mounts one from a native
`fuse::Component`, an in-memory table, or a `dlopen()`'d shared object (`mount_component`/
`mount_table`/`mount_dynamic`). Both are hidden-visibility C++ internals, never exported C symbols
— they cost this library nothing against "one symbol" (point 1 above), and
`test/abi/conformance_test.cpp` links them directly the same test-only way `test_support.hpp`
reaches `fuse_ctx`'s own private layout.

**Fortran gets the mirror of `ffuse`.** `ffuse_provider` (`tools/gen-abi.py`'s
`gen_fortran_provider_module`, written to `fortran/ffuse_provider.f90`) generates a
`fuse_provider_abi_v1` derived type with one `c_funptr` member per entry (bit-for-bit the C
layout, pinned by `test/abi/fortran/sizes_test.f90` the same way every other struct is) and
`ffuse_provider_set_<entry>()` helpers that assign a plugin author's own `bind(C)` procedures into
it via `c_funloc`. This is the INVERSE of `ffuse`'s own exception (Fortran keeps flat `bind(C)`
symbols because a function-pointer struct is unergonomic there): a provider-direction Fortran
plugin builds and exports the struct itself, since the struct's CONTENT is the plugin's own code,
not a table a Fortran caller merely dereferences.

### Device fields: a worked growth, rule 3

`fuse_field_spec` (the `describe()` OUT row, `entries.def`) is the struct the device-fields
work grew, and it is a clean worked example of rule 3 above: measured first, not assumed.
`fuse_field_spec` has `name`, `components`, `intent`, `element_type`, `struct_size` -- 28 bytes of
members in a struct whose compiled `sizeof` was already 32, so a naive 4-byte append (just
`memspace`, nothing else) would have landed entirely inside the existing trailing padding, changing
the layout an old caller's `struct_size` could not see, exactly the hazard `fuse_version_info` is
cited for above. Confirmed by a standalone compiled C `sizeof`/`offsetof` check against the
pre-change struct before writing anything: `memspace` at offset 28 if appended alone would sit in
padding bytes 28-31 of a 32-byte struct. Rule 3's own remedy applied: `memspace` (`int32_t`, the new
`FUSE_MEMSPACE_HOST`/`FUSE_MEMSPACE_DEVICE` value) is appended together with an explicit `reserved`
(`int32_t`) word, pushing `sizeof(fuse_field_spec)` from 32 to 40 -- a real, test-visible growth in
`struct_size`, re-measured the same way after the change: `memspace` at offset 28, `reserved` at
offset 32, struct size 40. `test/abi/struct_sizes.golden`'s `fuse_field_spec` line and
`test/abi/entries_golden_test.cpp`'s `FUSE_NO_TRAILING_PADDING` check were both updated to the new
last member (`reserved`); every generated file (`types.h`, `participant.h`, `provider.h`,
`ffuse.f90`, `ffuse_provider.f90`, `fuse_fortran_shim.c`) came from `tools/gen-abi.py --write`, not
a hand edit, and `tools/gen-abi.py --check` reports 0 stale.

`memspace` defaults to `FUSE_MEMSPACE_HOST` (value 0) for a `struct_size` an old caller set before
this member existed, the backward-compatible default rule 2 already requires for any new member --
an existing plugin's `describe()` that never touches `memspace` at all gets a host field, the same
behaviour it had before the member existed. The new capability bit, `FUSE_CAP_DEVICE_FIELDS`, and the two
refusal gates `fuse::abi_detail::ProviderAdapter::register_fields()` applies to a
`FUSE_MEMSPACE_DEVICE` request are documented for plugin authors in
[](../user/providers.md#device-fields); the positive device-resident path has been run on CUDA.
