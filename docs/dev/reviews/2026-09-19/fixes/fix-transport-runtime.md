# fix-transport-runtime.md

Fixes for REVIEW.md's transport-runtime findings (TR-F1..TR-F7 / unified
R-09, R-10, R-30, R-31, R-32, R-46, R-50), from
`docs/dev/reviews/2026-09-19/transport-runtime.md`, on branch
`review-2026-09-19-fixes` starting at `ffea077`.

**Revision note.** R-10 was originally fixed with a blanket refusal
(`E-RT-REGION-MPI`) for any iteration region spanning more than one rank.
The lead rejected that as inadequate -- it removes iterated coupling across
partitioned interfaces under MPI, a load-bearing capability (Demo A,
the Gray-Scott S2 gate) -- and asked for the actual cross-rank residual
reduction. This revision implements it: `run_region()` now agrees the
residual across a region's participants on every sweep
(`reduce_region_residuals()`, `src/runtime/runner.cpp`), and
`E-RT-REGION-MPI` narrows to the one case that reduction does not cover
(a dynamic accelerator). See "Fixed: R-10" below for the full account, and
"## Commit units" for how this replaces the prior refusal-only unit.

All work stayed inside OWNED PATHS: `include/fuse/transport/**`,
`src/transport/**`, `test/transport/**`, `include/fuse/runtime/**`,
`src/runtime/**`, `test/runtime/**`, `test/parallel/**`, plus
`build-evidence/mutations/transport-runtime.toml`. No `git` state was
touched; `git diff`/`git status` only.

## Commit units

```json
[
  {
    "ids": ["R-09", "R-32"],
    "files": [
      "src/runtime/runner.cpp",
      "src/runtime/iteration.cpp"
    ],
    "message": "Carry an iteration-region failure forward instead of abandoning the period\n\nAlso reads MPI_TAG_UB from the runner's own base communicator instead of MPI_COMM_WORLD."
  },
  {
    "ids": ["R-10"],
    "files": [
      "include/fuse/transport/tags.hpp",
      "src/runtime/runner_impl.hpp",
      "src/runtime/runner.cpp",
      "src/runtime/iteration.cpp",
      "test/transport/transport_test.cpp",
      "test/runtime/runner_test.cpp"
    ],
    "message": "Agree an iteration region's residual across its participants every sweep\n\nReplaces the earlier blanket cross-rank refusal; only a dynamic accelerator is still refused."
  },
  {
    "ids": ["R-10", "R-50"],
    "files": [
      "test/parallel/runner_transport_test.cpp"
    ],
    "message": "Add MPI tests for the narrowed refusal and the reduced region's iteration count\n\nAlso switches its one hand-rolled rank skip to FUSE_REQUIRE_AT_LEAST_RANKS."
  },
  {
    "ids": ["R-30", "R-50"],
    "files": [
      "src/transport/mpi_transport.cpp",
      "test/parallel/mpi_transport_test.cpp"
    ],
    "message": "Agree on TransportSpec::components across ranks in build_plan()\n\nAlso switches this file's hand-rolled rank-count skips to the documented FUSE_REQUIRE_RANKS macros."
  },
  {
    "ids": ["R-46"],
    "files": [
      "src/transport/lowfive_transport.cpp"
    ],
    "message": "Rename the residual __soma_lowfive_probe__ dataset name\n\nCosmetic only; the name never leaves this process's own scratch HDF5 file."
  },
  {
    "ids": ["R-50"],
    "files": [
      "test/parallel/comm_graph_test.cpp"
    ],
    "message": "Switch this file's hand-rolled rank-count skips to FUSE_REQUIRE_*_RANKS\n\nThe halves()-based divisibility skips stay hand-rolled; the macros cannot express them."
  },
  {
    "ids": ["R-10", "R-30"],
    "files": [
      "build-evidence/mutations/transport-runtime.toml"
    ],
    "message": "Add mutations for the region refusal guard, the components agreement, and the residual reduction\n\nOne Tier 0 mutation verified killed here; two MPI-gated mutations for a host to assess."
  }
]
```

## Fixed

**R-09 (TR-F1, BLOCKER)** `src/runtime/runner.cpp:2112-2118`, and the
symmetric gap in `src/runtime/iteration.cpp`.

Confirmed the finding by reading `step_period()`'s two failure branches side
by side: an ordinary instruction's failure sets `step_failure` and the walk
continues (so every later `Epoch`/`Transport` a partner rank is waiting at is
still reached); a failed iteration region instead `return`ed immediately,
abandoning the rest of the period's instruction stream for every OTHER
channel too, not just the one the region belongs to. `run_region()` itself
had the same asymmetry internally: the accelerator-divergence path
`return`ed directly out of the sweep loop.

Fix, in `runner.cpp`'s `step_period()`: a failed region now sets
`step_failure` (if not already set) and lets the walk continue via
`i = r.exit + 1`, exactly like an ordinary instruction failure.

Fix, in `iteration.cpp`'s `run_region()`: the accelerator-divergence branch
now records `step_failure` and `break`s out of the sweep loop instead of
returning directly, and a check after the sweep loop returns that failure
before falling into the non-convergence tail. This closes a second,
independently-observable gap: a component's `advance()` failing on the
region's own tick-based re-advance, on the sweep that happens to be the
*last* permitted iteration, used to fall into the non-convergence tail
instead of reporting the real failure.

**Correction during this revision.** A debugging step from the original fix
session (temporarily disabling this exact post-loop check to confirm a test
caught its absence) was left disabled on disk -- `if (false &&
step_failure.has_value())` -- and was not restored before that session's
final report was written. Caught and fixed at the start of this revision
(confirmed via `grep`, then restored and rebuilt/retested before starting
the R-10 work); the Tier 0 suite was green throughout because the specific
regression test for this path (`Runner.AnAdvanceFailureInsideAnIteration
RegionSweepIsReported`) exercises the PRE-EXISTING per-sweep check earlier in
the loop, not this post-loop one -- see that test's own comment for why.

Verification: the runner.cpp/step_period() half and the accelerator-diverged
half are both fundamentally about a hang across ranks -- no MPI here, so
these are **unverified in this sandbox**; see "Needs a host run". The
tick-re-advance-failure half is single-process-observable and is covered by
`Runner.AnAdvanceFailureInsideAnIterationRegionSweepIsReported`
(test/runtime/runner_test.cpp), which passes with the fix restored.

**R-10 (TR-F2, BLOCKER)** `src/runtime/iteration.cpp:600-639` (unified id;
review's own citation range covers `resolve_regions()`/`run_region()`
broadly).

Confirmed: `run_region()`'s convergence decision was computed entirely from
this rank's own residual buffers, with no `MPI_Allreduce` or equivalent
anywhere on that path.

**The reduction (this revision).** Added `Runner::Impl::reduce_region_
residuals()` (`src/runtime/runner.cpp`), reached from `run_region()` once
per sweep whenever `RegionRuntime::participants` names more than one rank.
It packs each block's residual and reference contribution plus a status
flag into one payload, runs it through the SAME binomial reduce-then-
broadcast schedule `reconcile_epoch()` uses (`epoch_schedule.hpp`, so every
participant reaches the same collective on the same sweep), and combines
each block's contribution across ranks via `fuse::num::combine()`
(`fuse/num/residual_norm.hpp`) -- the SAME undefined-vs-zero convention
`field_norm()`'s own callers get: a rank contributing nothing to a block
(no local entries, or a locally-failed sweep) sends `owned == 0`, which
`combine()` excludes rather than reads as a converged zero. A participant's
local failure is carried through the SAME reduction and turned into
`E-RT-REGION-PEER` on every rank, mirroring `reconcile_epoch()`'s own
`E-RT-EPOCH-PEER` reasoning exactly. `R.participants` (a new field on
`RegionRuntime`, `src/runtime/runner_impl.hpp`) is derived once in
`resolve_regions()` -- the sorted, de-duplicated union of ranks of every
component the region checkpoints -- the same derivation
`resolve_epoch_participants()` uses for an `Epoch`. A new MPI tag,
`tags::kIterationRegion` (`include/fuse/transport/tags.hpp`), keeps this
collective's messages from matching an `Epoch`'s on the same communicator
(a region's body can itself contain an `Epoch`, so the two can be in flight
at once).

**The accelerator (this revision, task item 2).** Checked what
`Accelerator::update()` actually consumes (`include/fuse/num/
accelerator.hpp`): `AccelKind::None` (plain Picard, direct substitution) and
`AccelKind::Constant` (a fixed, config-supplied omega, identical on every
rank because `AccelConfig` comes from the graph) read nothing beyond the
now-correctly-reduced residual and reference values passed to
`all_satisfied()` -- both are genuinely supported by the reduction alone.
Every OTHER kind (Aitken and the IQN family) derives its relaxation
coefficient from an inner product over THIS rank's own local iterate
history, computed inside `fuse::num` (outside this fixer's owned paths),
with no hook in `Accelerator::update()`'s signature for a caller-supplied
global reduction -- two ranks with different local slices would compute
different coefficients from that inner product and take different
accelerator paths even with the residual itself correctly agreed, reopening
the same desynchronisation risk. `resolve_regions()` now refuses only this
narrowed case (`decl.acceleration.kind` outside `{None, Constant}`) with
`E-RT-REGION-MPI`, replacing the earlier blanket refusal.

**Verification.** The refusal narrowing and the guard-condition logic
(`R.participants` populated only when `config.base_communicator.has_value()`
AND more than one rank appears) are Tier-0-verifiable and covered by the
existing `Runner.StillRunsAnIterationRegionWithNoCommunicatorConfigured` test
(unchanged, still passes) plus the Tier 0 mutation described below. The
ACTUAL cross-rank reduction needs real MPI and cannot run in this sandbox;
two new Tier 1 (compile-only here) tests exist for it in
`test/parallel/runner_transport_test.cpp`:
- `RunnerTransport.RefusesADynamicAcceleratorOnACrossRankRegion` (replaces
  the prior blanket-refusal test; same disjoint-rank-set fixture,
  `AccelKind::Aitken` instead of the default `None`) -- expects
  `E-RT-REGION-MPI`.
- `RunnerTransport.RegionConvergesInTheSerialIterationCount` -- a NEW
  fixture, `LinearFixedPointJob`, described next.

**The positive test's model.** `LinearFixedPointJob` builds a genuine
two-way iterated coupling -- `TwoComponentJob`'s one-way "link" channel has
nothing to iterate on -- mirroring `test/verification/iteration_test.cpp`'s
`LinearBlock` exactly: components "A" (`out[i] = 1.0 + 0.5*in[i]`) and "B"
(`out[i] = 2.0 + 0.4*in[i]`), 4 entities, tolerance `1e-10`, `AccelKind::
None`. Both components span the WHOLE communicator (unlike
`TwoComponentJob`'s disjoint halves), so every rank owns a slice of both,
and at 3 ranks the 4 entities split 2/1/1 -- a genuinely unequal local
residual on every rank, satisfying the task's requirement without needing
asymmetric component placement. Because the map is applied entity-wise with
no coupling BETWEEN entities, the fixed point and its contraction rate do
not depend on how the entities are split across ranks; only the reduction
combining per-rank Euclidean sums back into one global norm makes that hold
under a real partition. The expected iteration count, 30, was measured
directly in this sandbox (no MPI needed) by running the EXISTING serial
test `Iteration.AitkenNeedsNoMoreIterationsThanPicard`
(test/verification/iteration_test.cpp), which already prints the Picard
count for this identical coefficient/tolerance/entity-count combination on
one process (output: `iterations: Picard 30, Aitken 12`) -- reused here as
the "serial iteration_test.cpp fixture" the task pointed at, rather than
re-derived analytically.

A Tier 0 mutation making the accel-kind refusal fire unconditionally
(`cross-rank-region-check-fires-unconditionally`,
`build-evidence/mutations/transport-runtime.toml`, carried over from the
original R-10 fix and still valid against this revision's code) was
hand-applied and confirmed to kill the guard-condition tests, then
reverted. A new MPI-gated mutation, `region-residual-skips-the-cross-rank-
reduction`, disables the reduction entirely (`if (R.participants.size() >
1)` to `if (false)`) and is recorded for a host to assess against
`RunnerTransport.RegionConvergesInTheSerialIterationCount`; see that
mutation's own `describes` field for why either an outright assertion
failure or a 120s timeout is a correct kill.

**R-30 (TR-F3, MAJOR)** `src/transport/mpi_transport.cpp:355-358` vs
`842-876`. Unchanged from the original fix session.

Confirmed by reading `MpiTransport::build_plan()` and its newer sibling
`build_plan_from_peers()` side by side: the peer-wise path's "Phase B"
numbering check catches a `TransportSpec::components` disagreement as a
side effect of its own count handshake; the older, matched `build_plan()`
path has no equivalent.

Fix: added a new collective ("Collective 2b") right after the existing
id-count `MPI_Allgather`: every rank's `spec.components` is gathered and
compared against the first such rank's value among the local src/dst sets;
a mismatch is recorded via `Err::PeerMatch` and reconciled at the function's
existing final `agree()` call.

Verification: added
`MpiTransport.BuildPlanRejectsDisagreeingComponentsAcrossRanks`
(`test/parallel/mpi_transport_test.cpp`). Built successfully under `tier1`
(compile-only). Needs a host run to actually execute.

**R-31 (TR-F4, MAJOR)** -- see "Rejected" below: already fixed in wave 0,
mis-classified in the review as a live defect. Unchanged from the original
fix session.

**R-32 (TR-F5, MAJOR)** `src/runtime/runner.cpp:1567`. Unchanged from the
original fix session.

Confirmed: `resolve_epoch_participants()`'s `MPI_TAG_UB` safety check
queried `MPI_COMM_WORLD` unconditionally, while `reconcile_epoch()` sends
and receives on `config.base_communicator`.

Fix: query `MPI_TAG_UB` against the same converted `base` communicator
`reconcile_epoch()` itself uses, with a defensive `MPI_COMM_NULL` check.

Verification: by code inspection (the fix makes the check read the SAME
communicator the protected code uses) and by every Tier 1 target that links
`fuse-coupler-runtime` building without error.

**R-46 (TR-F6, MINOR)** `src/transport/lowfive_transport.cpp:447`.
Unchanged from the original fix session.

Confirmed: `"__soma_lowfive_probe__"` is a residual upstream name in an HDF5
dataset name used only as this process's own scratch probe -- purely
cosmetic. Fix: renamed to `"__fuse_lowfive_probe__"`.

**R-50 (TR-F7, DEVIATION)** `test/parallel/*.cpp`. Unchanged from the
original fix session, plus one new site.

Decision: code, not document. Converted 28 mechanically-convertible
hand-rolled rank-count skips to `FUSE_REQUIRE_RANKS`/`FUSE_REQUIRE_AT_LEAST_
RANKS` across `comm_graph_test.cpp` (16), `mpi_transport_test.cpp` (11), and
`runner_transport_test.cpp` (1, the original test that this revision's new
tests also use `FUSE_REQUIRE_AT_LEAST_RANKS` for). Left hand-rolled, with a
comment explaining why: 7 `halves()`-based divisibility skips in
`mpi_transport_test.cpp`, and one deliberate `ASSERT_GE` in
`epoch_reconcile_test.cpp`.

## Rejected

**R-31 (TR-F4, MAJOR)**, as instructed by the task brief: mis-classified.
Grepped `include/fuse/transport/` and `src/transport/` for
`SOMA_ENABLE_LOWFIVE`: zero occurrences. Already fixed in an earlier wave;
the review's citation was a doc comment and a user-facing message string,
never a live `#ifdef`. Recorded as REJECTED-as-stated / fixed-as-stale-name
-- no code change made for this finding.

## Needs another owner

None remaining. (An earlier version of this section reported that `test/parallel/comm_graph_test.cpp` failed to compile under tier 1 on a private include; that was closed by `41535ff` in the same fix wave. Corrected 2026-09-20 per pass-2 finding N-transport-runtime-4.)

## Needs a host run

All of the following were built (Tier 1, compile-only) but could not be
executed: this sandbox has no working MPI (`MPI_Init_thread` dies
immediately).

- `parallel.runner_transport.{2,3,4,5}` --
  `RunnerTransport.RefusesADynamicAcceleratorOnACrossRankRegion` (R-10,
  replaces the prior blanket-refusal test). Expect every rank to return
  `E-RT-REGION-MPI` from `initialize()`.
- `parallel.runner_transport.{2,3,4,5}` --
  `RunnerTransport.RegionConvergesInTheSerialIterationCount` (R-10, new).
  Expect `initialize()` to succeed on every rank, `run_period()` to
  succeed, and `iteration_stats()[0].iterations == 30` on EVERY rank
  (matching `Iteration.AitkenNeedsNoMoreIterationsThanPicard`'s serial
  Picard count for the identical model). At 3+ ranks this is the case with
  genuinely unequal local residuals (4 entities do not divide evenly).
- `parallel.mpi_transport.{2,3,4,5}` --
  `MpiTransport.BuildPlanRejectsDisagreeingComponentsAcrossRanks` (R-30).
  Expect every rank to return `E-TRANSPORT-PEERMATCH`.
- The existing `parallel.epoch_reconcile.*` and `parallel.runner_transport.*`
  legs, re-run to confirm no regression from R-09's
  `step_period()`/`run_region()` change, R-10's new reduction, and R-32's
  `MPI_TAG_UB` fix.
- The MPI-gated mutations `build-plan-skips-the-components-agreement` and
  `region-residual-skips-the-cross-rank-reduction`
  (`build-evidence/mutations/transport-runtime.toml`), with
  `FUSE_MUTATE_MPI=1`.
- R-09's runner.cpp/step_period() half and the accelerator-divergence half
  of the iteration.cpp fix: fundamentally a hang-vs-no-hang distinction
  across ranks, not independently unit-testable without MPI.
- `test/parallel/comm_graph_test.cpp`'s R-50 conversion: cannot compile
  under `tier1` at all for the unrelated, pre-existing reason recorded
  under "Needs another owner" above.

## Deviation decisions

**R-50**: code, not document (detailed above under "Fixed").

## Gate results (this session, Tier 0 unless noted)

- `ctest` (Tier 0, full suite): **71/71 passed**.
- `python3 tools/check-invariants.py`: 0 violations.
- `python3 tools/check-layering.py`: 0 violations.
- `python3 tools/mutate.py --check-anchors`: **86 valid, 0 stale**.
- Tier 1 (compile-only): `fuse_runtime_runner_test`,
  `fuse_transport_transport_test`, `fuse_parallel_mpi_transport_test`,
  `fuse_parallel_runner_transport_test`, `fuse_parallel_epoch_reconcile_test`,
  `fuse_model_descriptor_mpi_test` all built cleanly.
  `fuse_parallel_comm_graph_test` did not (pre-existing, see "Needs another
  owner").
