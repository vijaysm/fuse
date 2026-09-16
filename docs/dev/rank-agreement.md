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
