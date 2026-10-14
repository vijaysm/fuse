# Rank agreement, and why it is an operator

A coupled run deadlocks in a characteristic way. One rank hits an error — a
component refuses a step, a mesh transfer finds an uncovered point, an exact
conversion overflows — and stops or takes a different branch. Its partners
are already committed to the next data exchange and block waiting for a
message that never arrives. On four ranks you notice in seconds. On four
thousand, the job sits at full allocation until the wall clock kills it, and
the log says nothing, because nothing failed on the ranks that were still
running.

The fix is structural: after every data move, every rank that took part learns
the same outcome before any of them proceeds — all succeeded, or which rank did
not and why. That is the `Epoch` operator.

## Why an operator rather than something the runtime does

Making it an instruction in the compiled plan buys three things that an
implicit runtime call does not.

**It is visible.** It appears in the plan dump, so you can see how many
agreement points a run will execute and where they are, before running it.

**It is scheduled.** It is ordered by the same rules as every other operator,
so it cannot drift relative to the move it guards.

**It is checkable.** The compiler verifies things about it that a runtime call
could only discover by hanging. See below.

## What the compiler proves

*Every data move has one.* A move with no following agreement point is refused
with `E-SCHED-EPOCH-MISSING`. Without it, a failure on one rank is discovered
by the others only when they block on the next collective, which may be many
operators later and in an unrelated part of the plan.

*It covers exactly the ranks the move covers.* Checked as
`E-SCHED-EPOCH-PARTIAL`. A point covering fewer ranks than its move leaves a
partner waiting for a report that is never sent; covering more makes ranks wait
for partners who are not coming. Both are hangs, and both are detectable here
rather than at run time.

## A gap the scan does not close

The scan in `schedule()` (`src/schedule/scheduler.cpp`) that proves every data move has a following
rank-agreement point walks the plan's instructions for `OpKind::Transport` only. A channel whose `Remap`
runs at a placement other than its source or target (`lower_channel()`'s step 5 in
`src/ir/rules_channel.cpp`, `ranks_differ && remap_moves`) also moves data across ranks, inside the
`Remap` instruction itself, with no separate `Transport` for the scan to find. Such a `Remap` carries no
`E-SCHED-EPOCH-MISSING`/`E-SCHED-EPOCH-PARTIAL` guarantee: a rank-local failure inside it (an uncovered
point, an exact-conversion overflow) is discovered by its partners only when they block on the next
collective — the hang this operator exists to prevent for the `Transport` case. Widening the scan
mechanically was not done: a `Transport` is followed by a dedicated `Epoch` unconditionally
(`lower_channel()` step 4), while a placement-moving `Remap` has no uniform post-`Remap` agreement
point to pair against, so the same pairing loop would either miss the real gap or refuse legal graphs.
Closing it needs an agreement point emitted alongside a placement-moving `Remap`, the way `Transport`'s
is — a channel-lowering change. Review 2026-10-11 IS-3, recorded rather than fixed.

## What it deliberately does not do

It is **not** a global barrier, and the compiler does not require every rank to
reach every agreement point. Two independent couplings on disjoint ranks have
two independent agreement points, and forcing all four ranks through both would
add synchronization the physics never asked for. That cost — paying for
coupling you did not ask for — is the reason people distrust couplers at scale,
and it is not necessary: the property that prevents a hang is agreement among
the participants of each move, not agreement among everyone.

## Current cost

Each agreement point carries a buffer of its own, because the plan has no
notion yet of an operator whose output aliases its input: it copies rather than
allocating nothing. That is one interface-sized buffer per data move. It is
fixed when in-place operators land with the runtime, not by removing the
agreement point.

## How the agreement is actually performed

Not with a sub-communicator. The obvious implementation splits one
communicator per exchange pair at setup and keeps it for the run, and that is
what the first version did — but a held communicator has to be released, and
`MPI_Comm_free` is **collective**. Destruction order would then become a
correctness question: if one rank destroys its `Runner` and another has not
reached that point, they deadlock in the destructor. The transport layer
avoids this deliberately (`attach_comm_graphs` frees everything before it
returns, so "nothing outlives this function"), and holding communicators for
the run would have reintroduced precisely that hazard. A host run made it
concrete by reporting unfreed communicator handles at exit.

Instead, each pair's participant list — the union of the two components' rank
sets, sorted — is computed from the graph at setup. The graph is identical on
every rank, so every participant derives the same list without communicating
to establish it. The agreement is then a binomial reduce to the first
participant and a broadcast back, point-to-point on the base communicator
under one dedicated tag, with the schedule derived only from the list length
and this rank's position in it. Nothing is allocated and nothing has to be
released.

That pattern lives in `fuse/runtime/epoch_schedule.hpp`, apart from the MPI
calls that carry it out, for a specific reason: a mistake in it is a **hang**,
not a wrong number, so it cannot be observed by a serial test and shows up
only as a stalled parallel leg. As arithmetic over participant count and
position it is checkable without MPI, and `runtime.epoch_schedule` does check
it — matched sends and receives, a hand-up chain that always terminates at
participant 0, and a message count per participant that stays logarithmic.

The tag is small and explicit (`kEpoch = 0x5C14`, next to the crystal
router's `0x5C12`). RT-7/TR-12 (review 2026-10-11): this page previously said
`0x5C13` — that is `kUnionCommCreate` (`fuse/transport/tags.hpp`'s own
history comment: `kEpoch` was first chosen as `0x5C13`, landed on the crystal
router's own collision, and was moved to `0x5C14`; `kUnionCommCreate` took
the vacated `0x5C13`), not the epoch tag this section describes. Both values
stay below `MPI_TAG_UB`'s floor either way, so the collision never reached
`E-RT-EPOCH-TAG` itself — this was doc drift, not a code defect. MPI only
guarantees tags up to `MPI_TAG_UB`, whose standard minimum
is 32767; a first version packed `'EPOC'` into an int, which is 1162891075 and
which every MPI rejects outright. Setup now reads `MPI_TAG_UB` and refuses
with `E-RT-EPOCH-TAG` if it is ever below the tag in use — a refusal before the
run is a better failure than "Invalid tag" raised inside a reduction on one
rank while its partners wait for it.

Tag collision with a data move is not possible: `MpiTransport::set_communicator`
duplicates the communicator it is handed, so a move never uses the base
communicator even when the user passes it the same one.

## What a failed rank still executes

A rank that has failed mid-step does **not** stop and does not skip
indiscriminately. It keeps executing the *collective* operators and skips the
local ones (`is_collective` in `fuse/ir/op.hpp`).

The asymmetry matters in one direction only. Skipping local arithmetic costs
nothing: the period is being abandoned, and the values would never be read.
Skipping a collective is fatal — the partners are already inside the data
move, waiting for this rank, which is the identical hang that returning at the
point of failure produces. So the move still happens; what it moves is
whatever the buffers hold, and it is never used, because the agreement point
immediately after ends the period on every participant.

`is_collective` is a function rather than a `default:` inside the execution
loop so that a test can enumerate every operator: adding one without
classifying it fails a test rather than silently inheriting an answer. An
unclassified operator is assumed collective, because that error wastes work
while the other error hangs.

A component that refuses its step carries forward the same way, both in the
period loop and inside an iteration region — where it matters most, since the
region body runs many times and a rank dropping out of the exchange
desynchronises the remaining iterations rather than one step.

### The checklist question

Review 2026-10-11 (index.md §2.1) found this class fixed per-site and
reintroduced per-subsystem seven separate times before that round, and five
more sites inside the runtime layer alone (RT-1, RT-2, RT-4) — per-site fixes
were not holding. Ask this question of **every** rank-local refusal or early
return, not only ones that look collective-adjacent:

> Does something later in this same call, on a path this refusal skips,
> perform a real collective — or can one rank reach a later collective while
> another, having taken this refusal, does not?

If the answer is "yes, a peer can reach a collective this rank now skips",
the refusal is a hang waiting on a trigger, whether or not the current test
suite exercises the trigger. Two recipes answer it, and this project has not
needed a third:

1. **Agree before the first collective.** `Allreduce(MIN)` (or an equivalent
   reduce-then-broadcast) of a local pass/fail verdict, reached by every rank
   that would otherwise decide alone, before anything collective runs — so
   either every rank enters the collectives that follow, or none does. Use
   this when the refusal is checked once, up front (a setup-time validation,
   a period-top checkpoint), rather than discovered partway through a body
   that has already reached some of its own collectives.
2. **Record and carry forward.** Record the failure (the project's own
   `step_failure`/verdict-carrying convention) instead of returning, and keep
   executing every remaining *collective* operation the rest of this call
   still owes its partners (`is_collective`, above) while skipping only the
   now-pointless local arithmetic — reaching the agreement point every
   partner is already heading for, where the failure is reported to
   everyone at once. Use this when the refusal is discovered mid-body, after
   some of this call's own collectives have already run.

Both recipes are already load-bearing elsewhere in this file: recipe 1 is
`reconcile_epoch()`'s "before" half (a rank's own verdict reaches the
reduction before the copy it guards); recipe 2 is "What a failed rank still
executes", immediately above, generalized from the period loop to any call
with its own remaining collectives.

## Partially shared placements

Two components may share ranks. The exchange between them is then described
peer by peer — `send_plan()` and `recv_plan()` — because a single list of
global ids per rank cannot say whether an entry is something this rank *holds*
or something it *wants*, and under partial overlap it is genuinely both, for
different entities. Matches whose two ends land on the same rank never travel:
they are copied locally.

The two halves are checked against each other before the run. Each rank
announces how many values it intends to send every other, and each requires
that to equal what it expects to receive. A disagreement is
`E-TRANSPORT-PEERMATCH` at setup — the failure mode it catches is a transport
given a different communicator than the rank graph was built over, where the
two rank numberings differ and every value would be delivered to the wrong
rank for the entire run.

Identical and disjoint placements keep using the matched plan, which has been
verified longer. On a pattern both forms can express, they are required to
deliver identical values, with the matched plan as the oracle.

## What is atomic, and by what mechanism

There is no transaction object in FUSE. The atomicity properties a coupler needs
are obtained by instruction ordering and by verification, and it is worth being
explicit about which mechanism provides which, because "we have transactions" is
a claim a reader cannot check:

| Property | Mechanism |
|---|---|
| A rank does not publish a value derived from a transfer its partners did not complete | `Epoch` reconciles **before** its copy — ordering, not a commit protocol |
| An iteration sweep that is discarded leaves the registered field storage as it was | `save_state()`/`restore_state()`, with the restored bytes compared against the saved bytes (`E-ITER-ROLLBACK`) |
| A component cannot merely *claim* it rolled back | the byte comparison above is a check; a component that returns success without restoring fails the run |
| Which state must be checkpointed | derived from dataflow over the graph, then verified against the components actually present — not requested from the user |

The byte comparison is the load-bearing one. A tolerance would accept a restore
that is nearly right, and a rollback that is nearly right is not a rollback: the
next sweep would start from a state no one chose.


## A failing period and the components that keep stepping

A component may refuse its step, and an iteration region may fail to converge.
Neither aborts the period: `step_period()` records the failure, keeps walking
the instruction stream so every **collective** instruction is still executed —
a rank that skipped one would leave its partners waiting inside it — and
reports the failure at the end of the period, where every rank has reached the
same conclusion.

What it does *not* do any longer is stop advancing every component. Since
2026-09-21 the rule is:

> The origin of the failure, and every component that reads — transitively —
> something the origin provides, do not advance for the rest of that period.
> Every other component does.

`Runner::Impl::downstream_of` holds that relation, one row per component: an
edge per channel from the component owning its source field to the component
owning its target, transitively closed, built once in `build_ticks()`. A
region failure is not attributable to a single component, so the origin is
every member the region re-advances.

Two reasons it is a dependence rule and not a blanket:

- A component downstream of the failure would consume a value the failed step
  never produced. That is the case the old blanket was protecting, and it is
  still refused.
- A component that reads nothing from the failure is in exactly the state it
  would have been in had the period succeeded. Skipping its step bought
  nothing and, because a period's ticks routinely all land in
  `step_period()`'s trailing loop, it meant a single refusal could stop every
  component in the period — including ones on a channel with no relation to
  the failure.

Upstream components advance. They read nothing the failed step was supposed
to produce, so the argument for blocking them does not apply.

`Runner.AFailedStepBlocksOnlyWhatReadsFromIt` pins both halves in Tier 0, and
two mutations (`a-failed-step-blocks-every-later-advance`,
`a-failed-step-blocks-nothing`) fail if either half is removed.


## A deferred verdict is one value, and clearing it loses a check

`MpiTransport::build_plan()` never returns at the point of a local failure: it
records the verdict with `fail()` and returns only at an agreement point, where
every rank is present to return the same thing. That makes the verdict a single
monotonic value per call — `fail()` keeps the first and ignores later ones.

**It must never be cleared.** Phase D (the id matching) used to clear `err` so
it could reuse it as its own loop accumulator. That was harmless when written,
because every verdict recorded before it had already been agreed and returned
at Collective 1. Collective 2b — the components agreement — then added a
verdict recorded *after* Collective 1 and read only at Collective 4, on the far
side of the clear. The consequence: that check ran, computed the correct answer
on every rank, and had it erased two phases later, so `build_plan()` returned
success on ranks that had just refused. The guard existed, was reached, was
right, and was ignored.

It survived a full review pass, a mutation entry, and five readings of the
path. What found it was printing the verdict at both ends under
`FUSE_TRANSPORT_DEBUG`: the rule reported `agreed=false` on both ranks while
the agreement two phases later reported `local=0`. Two numbers that cannot both
be true localised in one run what argument had failed to find five times.

`tools/check-invariants.py` now refuses any `err = Err::None;` in library code
(**I-VERDICT**). If a phase needs its own accumulator, it gets its own
variable.

## "Decided from local data alone" is not automatically safe

`DistSparseMatrix::from_triplets`'s own refusals (`E-LINALG-FOREIGN-ROW`,
`E-LINALG-COL-RANGE`, [](linalg-distributed.md)) return on a condition this
rank alone can see, and that is fine *there*, because nothing later in that
call issues a collective the refusal would need to be agreed before. It is
easy to read that as a general rule — "a local refusal never needs to agree
with its peers" — and it is not one.

`SchurSlabSolver::factor` ([](schur-slab-solver.md)) first checked whether
*this rank's own* row band had fewer than two rows and returned immediately
if so, before ever reaching the Schur-complement assembly's own
`Reducer::allgather`. A real two-rank run (`parallel.schur_slab_mpi.2`, host,
2026-10-10) put a one-row band on rank 0 and a fine one on rank 1: rank 0
refused and returned without entering the allgather; rank 1's own check
passed, so it called the allgather alone — and MPICH reported "Message
truncated", one rank waiting inside a collective nobody else had entered.
The fix was not to add a collective in front of the refusal (correct, more
expensive); `row_ownership` already gives every rank every participant's own
row-band bounds, so "does *any* band have fewer than two rows" is a pure
function of input every rank already holds in full — scanning all bands
costs `O(nranks)` CPU and no communication, and every rank reaches the
identical conclusion with no collective needed at all. The question to ask
of any refusal that looks locally decided is not "can I compute this from my
own data" but "does something *later in this same call*, on a path this
refusal skips, perform a real collective — and if every rank's local
decision need not agree, can one of them reach it while another does not."

## A tag that is right in file order can still be paired wrong at the call site

`MPI_Sendrecv` bundles a send and a receive in one call, and each endpoint
chooses its OWN tag for each; nothing enforces that the two calls on a given
pair of ranks agree on which tag is "my last row" and which is "my first
row" unless the author lines them up by hand. `test/parallel/
brusselator_grid_mpi_test.cpp`'s halo exchange sent its last row under tag 0
and waited for the PEER's tag 1 in the same call, then its first row under
tag 1 waiting for the peer's tag 0 — correct per-rank bookkeeping, crossed
between ranks: each rank's first `MPI_Sendrecv` then needed a message the
peer only sends from its SECOND call, which the peer reaches only after its
own first call completes. Both ranks sampled inside `MPIR_Sendrecv_impl` on
the first host run (2026-10-09), neither returning — diagnosed with macOS's
`sample` tool (see "Diagnosed with `sample`, not re-read code" below), the
same way D25's seed-chain hang was. The fix pairs the send and receive of
ONE call under the SAME tag (both ranks send their last row under tag 0 and
receive the peer's last row in the same call; then the same for the first
row under tag 1) rather than threading two different tags through two calls
by hand.

## Ownership of a window block is the consumer's, not the exporter's

A window-scoped region's per-block residual is measured through the
compiled delivery chain (Transport/Epoch/Remap/Constrain/Diagnose), which
runs on the CONSUMER's own processor set — not the exporter's. The runtime's
`window_export_meta` stored the EXPORTING member's own `hosted` flag where
the consumer's own hosting status was needed, which on a disjoint-ranks
channel inverts ownership in both directions at once: the exporting rank
wrongly claims a block it never writes, and the consuming rank — the only
one that does — wrongly disclaims it, so both sides sum a stale 0/0
residual/reference pair that reads as trivially converged. A host run of
`parallel.window_runner_mpi.{2,3}` (2026-10-08) measured exactly this:
`iterations_per_window[0] == 1` against a reference of 37, not a crash, a
silently wrong converged answer. Fixed with a field resolved from the
channel's TARGET component's own processor set
(`WindowMemberConfig::export_consumer_hosted`) rather than reusing the
exporter's own flag, then taken one level further: ownership is resolved
PER BLOCK from that block's own channel, not per member from the first
channel a fan-out member happens to feed, since a member feeding two
consumers on different processor sets would otherwise hand both blocks the
wrong one's answer. `KNOWN_ISSUES.md` D25 has the full account. The general
form: when a quantity is computed in one place (the exporter) but consumed
and measured in another (the consumer), "is this hosted here" is a question
about the CONSUMER's processor set, not the producer's, however natural it
reads to ask the producer first.

## Diagnosed with `sample`, not re-read code

A fifth re-reading of a hang rarely finds what the first four missed; a
sampling profiler attached to the stuck processes usually does in one shot.
D25's own seed-chain hang — three MPMD programs each stuck inside
`MPI_Alltoallv`, two of them (U and V) inside `Runner::initialize()`'s
seed-delivery chain and the third (the coupler) inside `run_region()`'s
first real Transport — was found this way on the host (2026-10-09): macOS's
`sample <pid>` on each of the three stuck processes named exactly which
collective each was blocked in and from where, which is what made "the
initialize-time seed chain's selection is per-rank, and after the hosting
fix it skips the non-hosting ranks entirely" discoverable at all rather than
guessed at. The same instrument found the crossed-tag `MPI_Sendrecv` hang
above. Where `FUSE_TRANSPORT_DEBUG`/`FUSE_SCHEDULE_DEBUG` ([](debugging.md))
print what the library THINKS it is doing, `sample` shows what every rank is
actually doing RIGHT NOW — the two are complementary, and a stranded
collective is squarely `sample`'s case: the library-side trace has nothing
to print because nothing failed, every rank is just waiting.


## Agreement on a schedule is not agreement on how many times to run it

Everything above proves a property of the *compiled plan*: the ranks that
take part in one agreement point are exactly the ranks the move it guards
covers, checked once at `schedule()` time (`E-SCHED-EPOCH-MISSING`,
`E-SCHED-EPOCH-PARTIAL`). It says nothing about whether two ranks agree on
how many times to execute that plan, and `KNOWN_ISSUES.md` D57 is a hang
that lives entirely in that gap.

`Constrain`'s own conservation reduction (`src/runtime/runner.cpp`,
`OpKind::Constrain`) is a real `MPI_Allreduce` over the base communicator,
not the point-to-point agreement this page otherwise describes, and its
own gate is coarser than `Epoch`'s: `ci.reduce` is
`config.base_communicator.has_value()` (`resolve_constrain()`, same file)
— true for every rank with a base communicator configured at all,
regardless of which ranks the channel's own source and target sets name.
`Epoch`'s own rank-set-scoped checks above do not apply to it, and
`schedule()` builds no equivalent proof for `Constrain` at all: there is
no `E-SCHED-CONSTRAIN-MISSING`. For an `ExternalDelivery` channel (a
LowFive-bound transfer, `FieldSpec::hdf5_path` set) this matters because
`src/ir/rules_channel.cpp` forces `ranks_differ` to `false` for exactly
that channel (`lower_channel()`, "FUSE performs no data move of its own
on this channel, so neither hop ... may fire"), which suppresses both the
`Transport` instruction and the `Epoch` point that would otherwise guard
it — but does **not** suppress `Constrain`, whose own emission is gated
on `ch.remap.flags != RemapFlags::None` alone. A conservative,
coupler-routed `ExternalDelivery` channel therefore compiles a
`Constrain` instruction with no `Epoch` anywhere near it, on every rank
of the base communicator, coupler included.

That alone is not a bug — every rank's own compiled plan is identical
(the premise this whole page relies on: "the graph is identical on every
rank, so every participant derives the same list without communicating
to establish it"), so the Nth `Constrain` instruction in the stream is
the same instruction, with the same `ci.reduce`, on every rank that
walks that stream. D57's hang is not a disagreement about the plan; it
is a disagreement about how many PERIODS of it to walk. `Runner::run()`
(the coupler's own driver, `coupler_main.c`) executes exactly the
requested period count. The ABI's own `prime_next_request()`
(`src/abi/participant.cpp`, called from `fuse_begin_step()`) used to have
no equivalent bound when the caller never declares
`fuse_setup_options::has_end_time` — true of both `u_main.c` and
`v_main.f90` — so it would silently start and run an extra period's own
first suspend point inside the SAME call that was supposed to report the
requested run as finished, and that extra period's own `Constrain`
instruction is an `MPI_Allreduce` the coupler — already past its own
exact-N-periods loop and returned from `main()` — never reaches. No
static check catches this because there is nothing to check statically:
`plan`+`graph` determine what one period executes, identically on every
rank, and say nothing about how many periods a given rank's own driver
will ask for. That is a property of the driver (`has_end_time`, and
which of `fuse_begin_step()`'s two internal paths discovers a period
boundary), not of the plan.
