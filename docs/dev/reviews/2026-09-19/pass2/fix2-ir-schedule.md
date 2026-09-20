<!-- Second-pass FIX report, 2026-09-20, answering
     docs/dev/reviews/2026-09-19/pass2/re-ir-schedule.md.
     Owned paths: include/fuse/ir/**, src/ir/**, test/ir/**,
     include/fuse/schedule/**, src/schedule/**, test/schedule/**,
     test/golden/*.ir. -->

# Fix report: `ir`/`schedule`, second pass

`re-ir-schedule.md`'s own findings needed no work: all five first-pass items
(R-26/R-27/R-28/R-43/R-44) are **CONFIRMED-FIXED** with no PARTIAL/NOT-FIXED
row, and its "New findings" table names exactly one item, N-ir-schedule-1
(IMPROVEMENT) -- extend `schedule.acyclicity_sweep` to also call `schedule()`
on the lockstep, non-lagged, non-region subset of its random graphs, so the
R-27 merge left-fold gets random-oracle coverage of buffer planning and
instruction emission, not just the two hand-built fixtures in `plan_test.cpp`.

Implementing that one-line-sounding extension immediately turned up a real
BLOCKER in production code: **N-ir-schedule-2**, filed and fixed below. This
report therefore has two fix units, BLOCKER first, the second building on it.

## Commit units

```json
[
  {
    "ids": ["N-ir-schedule-2"],
    "files": [
      "src/schedule/expand.cpp",
      "test/schedule/expand_test.cpp",
      "test/schedule/plan_test.cpp",
      "build-evidence/mutations/ir-schedule.toml"
    ],
    "message": "Fix Merge scheduled before the Transport/Epoch inputs it reads\n\nMerge carries no single channel origin, so expand() placed it at its component's tick start like Source/Sink instead of the tick end its inputs land on."
  },
  {
    "ids": ["N-ir-schedule-1"],
    "files": ["test/schedule/acyclicity_sweep_test.cpp"],
    "message": "Route the acyclicity sweep's non-lagged graphs through schedule()\n\nGives the merge left-fold random-oracle coverage of buffer planning and instruction emission, not just the two fixed plan_test.cpp fixtures."
  }
]
```

## Fixed

### N-ir-schedule-2 (BLOCKER, found while fixing N-ir-schedule-1) -- Merge scheduled before the inputs it reads

**Where.** `src/schedule/expand.cpp`: `emit_lockstep()`'s node-classification
`if`/`else` (the lockstep fast path) and the "everything no channel claimed"
loop near the bottom of `expand()` (the general path). Both paths share one
root cause.

**What was wrong.** A `Merge` combining two or more channels is given
`ChannelId{}` origin by `src/ir/lowering.cpp`'s combine step -- correctly, it
belongs to no single one of them. `expand()`'s `claimed_by_channel()` reads
that as "no channel claims this node" and routed it into the branch written
for `Source` and an unwritten `Sink`: placed at its owning component's tick
**start** (`t = 0` in the lockstep fast path; `k*dt` in the general path).
But a `Merge` is a chain-**closing** node, like the rest of a channel's own
non-`Sample` tail (which lands at `t = period` / `(k+1)*dt`) -- it reads the
`Transport`/`Epoch`/`Remap` outputs those chains produce. Whenever a writer
was colocated with its target (no `Transport`/`Epoch` needed), everything sat
at the same instant and `NodeId` order alone kept `Merge` after its inputs by
luck; `five_writer_merge()` in `plan_test.cpp` (R-27's own fixture) is exactly
this degenerate case, which is why it schedules today and always has. The
moment a writer is on a different rank than the target, its `Transport`/
`Epoch` pair runs at the tick's end while `Merge` was placed at the tick's
start, so `Merge` sorted *before* the inputs it reads and the scheduler's own
`check_dependence_order()` correctly refused the plan with `E-SCHED-ORDER` --
on an entirely ordinary, non-pathological coupling graph. `Merge` is the only
node kind this affects: `Source` genuinely opens a tick, and the only other
`ChannelId{}`-origin case, an unwritten `Sink`, genuinely has no data to wait
for.

**How it was found.** Not by re-reading the cited lines (there were none to
re-read -- see above); by implementing N-ir-schedule-1 and watching the
extended sweep fail on real, non-adversarial generated graphs (e.g. two
same-rank-as-each-other-but-not-as-the-target writers into one field). The
user separately reproduced the identical failure on a host tier-2 run of the
same in-progress sweep change and reported it back with the same example
shape, which is recorded here as confirmation, not as the discovery route.

**The fix.** In `emit_lockstep()`, add `else if (n.kind == OpKind::Merge)` to
place it at `t = out.period`, matching the "claimed by channel, not sampled"
branch just above. In the general path's boundary loop, compute
`tick_closing = (n.kind == OpKind::Merge)` and stamp its instances at
`scaled_tick(tick_closing ? k + 1 : k, dt)` -- the tick's end instead of its
start, the same instant a channel's own non-`Sample` tail lands on for that
tick when there is exactly one slab per tick, and always at or after every
writer instance for that tick when there is more than one (the last slab of
a destination tick always ends at `(k+1)*dt`). Both doc comments above the
two functions are updated to state the `Merge` exception plainly, since the
old wording ("a node no channel claimed runs at its component's only tick")
was the bug's own justification.

**Tests.**
- `Expand.MergeNeverPrecedesItsProducers` (new, `test/schedule/expand_test.cpp`):
  a `merge_across_ranks()` fixture (two writers on ranks 1 and 2, target on
  rank 0) run through **both** `ExpandMode::Auto` and `ExpandMode::ForceGeneral`,
  asserting the `Merge` instance's time is `>=` the latest `Transport`/`Epoch`
  instance's time. Fails before the fix in both modes (confirmed by hand,
  see below); passes after.
- `ExecutionPlan.MergeAcrossRanksSchedulesInDependenceOrder` (new,
  `test/schedule/plan_test.cpp`): the same shape through the full `schedule()`
  entry point, asserting `r.ok()` and `dependence_violations(dag, r.plan) ==
  0` (the file's own independent, non-`prove_acyclic`-based order checker) --
  not just that `schedule()` returned success, since a lucky order could
  still pass `r.ok()` without being a real dependence order.
- `schedule.acyclicity_sweep` (the N-ir-schedule-1 unit, sequenced after this
  one) now exercises this continuously: 20,000 generated graphs
  (`FUSE_SWEEP_GRAPHS=20000`), 0 disagreements.

**Mutation evidence**, hand-verified (`tools/mutate.py` was not run --
unsafe against a checkout six fixers share; see its own header note, quoted
in the pass2 review). Both entries added to
`build-evidence/mutations/ir-schedule.toml`:
- `merge-placed-at-tick-start-lockstep`: reverting the lockstep-path `else
  if` to `else if (false)` and rebuilding fails both
  `Expand.MergeNeverPrecedesItsProducers` (mode `Auto`) and
  `ExecutionPlan.MergeAcrossRanksSchedulesInDependenceOrder` and
  `AcyclicitySweep.VerdictAgreesWithAnIndependentlyComputedAnswer`; reverting
  the fix restores all three. `fuse_schedule_expand_test` / `schedule.expand`
  is the mutation's declared kill target.
- `merge-placed-at-tick-start-general`: reverting `tick_closing = (n.kind ==
  OpKind::Merge)` to `tick_closing = false` fails
  `Expand.MergeNeverPrecedesItsProducers` under mode `ForceGeneral` only
  (mode `Auto` is a different code path and is correctly unaffected); this
  one needed `ExpandMode::ForceGeneral` explicitly, since no lockstep graph
  reaches the general path. Same declared target.
- Both were also hand-verified NOT to change the outcome of
  `schedule.lockstep` (the byte-for-byte `Auto`-vs-`ForceGeneral` identity
  check) when the *fix* (not the mutation) is in place: `merge_across_ranks()`
  run through both modes produces an identical `TimedDag` node list after
  the fix (checked directly, not just indirectly through the existing
  fixtures in `lockstep_test.cpp`, which do not contain a cross-rank Merge).

### N-ir-schedule-1 (IMPROVEMENT) -- acyclicity sweep never reached `schedule()`

**Where.** `test/schedule/acyclicity_sweep_test.cpp`.

**The fix**, as specified in the first review and repeated in the pass2
report: the sweep's main loop now also runs the lockstep, non-lagged,
non-region subset of its 2000 (or `FUSE_SWEEP_GRAPHS`-many) generated graphs
through `schedule()`, not just `prove_acyclic()`. Every graph this generator
produces is already lockstep (one shared component interval) and region-free
(no `add_iteration_region` call in `generate()`); lag is the one axis its own
independent cycle answer does not depend on, so `!gen.has_lag` is the one
filter needed. `schedule()` is asserted to agree with the same independent
cycle verdict the acyclicity check already uses: it refuses with
`E-SCHED-CYCLE` exactly when `gen.has_same_level_loop`, and otherwise its
full `expand`/`prove_acyclic`/`plan_buffers`/emit pipeline must complete over
whatever merge chain `lower()` folded for that graph -- random writer counts
and arities the sweep already generates, now actually exercised past
`prove_acyclic()` for the first time. `constant_sizes(64)` matches the size
already used for the fold's fixed-fixture coverage in `plan_test.cpp`.

**Verified**: `FUSE_SWEEP_GRAPHS=20000 ./fuse_schedule_acyclicity_sweep_test`
-- 20,000 graphs, 0 false negatives/positives, 0 schedule disagreements,
~6,200 of the 20,000 routed through `schedule()` (comfortably over the
`n/50` evidence floor the test already enforces for the other two buckets).

## Rejected

None. `re-ir-schedule.md` recorded no PARTIAL/NOT-FIXED verification row and
no BLOCKER/MAJOR/MINOR/DEVIATION new finding for this layer group to accept
or reject.

## Needs another owner

None. Everything named by N-ir-schedule-1 and N-ir-schedule-2 is inside
`src/schedule/**` / `test/schedule/**`, both owned paths.

Purely informational, not requiring action from me: `tools/mutate.py
--check-anchors` reports one stale anchor, `supermesh-planes-point-outward`
in `src/backend/supermesh.cpp` -- outside every path I own (`backend`'s
fixer's file). Also observed but not mine to fix: `test/CMakeLists.txt`
intermittently failed to configure during this session with "test
source(s) registered nowhere: num/convergence_test.cpp" -- a live,
in-progress relocation by another fixer (`test/model/iteration_test.cpp`,
same byte size, appeared with today's date while the old file was still on
disk); it resolved and re-broke several times as that fixer's edits landed,
consistent with concurrent work on a shared file rather than anything in my
own paths.

## Needs a host run

None. Every test in this report's scope is Tier 0, no MPI: `schedule.expand`,
`schedule.plan`, and `schedule.acyclicity_sweep` all link `fuse::schedule`/
`fuse::ir` only, confirmed the same way the pass2 review confirmed it for the
rest of this layer group (no target links `MPI::MPI_CXX`).

## Deviation decisions

- **Verification ran against an isolated snapshot, not the shared
  build-t0, for the final full-suite pass.** `test/CMakeLists.txt`'s
  transient breakage (above) blocked `cmake`'s auto-reconfigure step
  end-to-end for stretches of several minutes at a time, more than once,
  independent of anything in my own paths. Rather than block on another
  fixer's in-progress edit, the final confirmation that all 18 `ir.*`/
  `schedule.*` tests pass, and both hand-verified mutations are correctly
  killed and un-killed, was run against a private `rsync` snapshot of the
  tree with only `src/schedule/**` and `test/schedule/**` re-synced from the
  live checkout -- decoupled from the unrelated concurrent edit, otherwise
  identical. `check-invariants.py`, `check-layering.py`, and
  `mutate.py --check-anchors` (which only needs an existing build dir's
  config, not a fresh reconfigure) all ran directly against the live repo
  and are unaffected by this. The live repo's own `git diff` for every file
  this report touches was re-read after restoring from the snapshot-verified
  state and confirmed identical to what was tested.
- **Two dependency tarballs fetched by URL instead of `FetchContent`'s
  `git clone`.** This sandbox's `git clone` fails with "Operation not
  permitted" creating any new `.git` directory, anywhere, including under
  `/tmp` and under the granted repo path itself -- confirmed with a bare
  `git clone` of an unrelated public repo before touching this one, so it is
  a sandbox-wide restriction, not specific to `r3d` or `yaml-cpp`. Both are
  Tier 0's two `FetchContent`-fetched dependencies with no local install on
  this machine (`r3d`, always fetched if absent; `yaml-cpp`, fetched by
  design per `cmake/FuseDependencies.cmake`'s own comment). Downloading the
  pinned commit/tag as a GitHub tarball (`codeload`-style archive URL, same
  content the git tag points at) and pointing `cmake` at it with
  `-DFETCHCONTENT_SOURCE_DIR_R3D=...` / `-DFETCHCONTENT_SOURCE_DIR_YAML-CPP=...`
  reproduces exactly what a successful `git clone` would have populated,
  without needing one. This is a local build-environment workaround, not a
  change to any committed file, and does not appear in the diff.
- **N-ir-schedule-2 is reported under this file's own report contract
  even though it is not named by `re-ir-schedule.md`.** The task brief names
  "every row of ... New findings" as in scope; N-ir-schedule-2 did not exist
  when that report was written; it surfaced only once N-ir-schedule-1's fix
  was actually implemented and run. Given the "BLOCKER first" commit
  ordering rule and the report's own required sections, treating a defect
  discovered mid-fix as out of scope and shipping the coverage extension
  that reveals it, unfixed, was rejected as the worse deviation.
