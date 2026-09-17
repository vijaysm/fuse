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

The tag is small and explicit (`0x5C13`, next to the crystal router's
`0x5C12`). MPI only guarantees tags up to `MPI_TAG_UB`, whose standard minimum
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
