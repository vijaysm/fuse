# Third-pass (final) review: `include/fuse/ir`, `src/ir`, `test/ir`, `include/fuse/schedule`, `src/schedule`, `test/schedule`, `test/golden/*.ir`

Reviewed at HEAD `acf3c5b` on branch `review-2026-09-19-fixes` (the branch had advanced from the
`ced1945` this task named — see "A note on HEAD drift" below; the two commits between them touch
only `REVIEW.md`, outside this layer group, so nothing below is affected). Prior reports read in
full: first review (`bdf4346`, `docs/dev/reviews/2026-09-19/ir-schedule.md`), first fix report
(`docs/dev/reviews/2026-09-19/fixes/fix-ir-schedule.md`), second-pass review (`dbdc872`,
`docs/dev/reviews/2026-09-19/pass2/re-ir-schedule.md`), second fix report
(`docs/dev/reviews/2026-09-19/pass2/fix2-ir-schedule.md`). Binding rules re-read in full:
`docs/dev/invariants.md`, `layering.md`, `conventions.md`, `testing.md`, `rank-agreement.md`,
`porting.md`.

**Built.** `cmake --preset tier0 -S <repo> -B <workdir>/b0 && cmake --build b0 -j4` — clean
configure and build, no build directory created under the repo, nothing under the repo edited, no
state-changing git command run (`git status` in the repo is clean throughout and at the end of this
review). `ctest --test-dir b0 -R '^ir\.|^schedule\.'` → **18/18 passed**. `tools/check-invariants.py`
→ 0 violations. `tools/check-layering.py` → 0 violations. `tools/mutate.py --check-anchors
--build-dir b0` → **95/95 valid, 0 stale**.

**Mutation evidence re-executed, not just read.** Because this is the final pass, the four
mutations this layer group's evidence file (`build-evidence/mutations/ir-schedule.toml`) claims
were hand-verified were actually re-applied and re-run here, against an isolated rsync copy of the
source tree (`include/`, `src/`, `test/`, `cmake/`, `CMakeLists.txt`; excludes `.git`, `upstream/`,
and the pre-existing `build*` directories) configured with the same tier0 preset and the already-
fetched `r3d`/`yaml-cpp` sources pointed at by `FETCHCONTENT_SOURCE_DIR_*` (this sandbox's `git
clone` is refused under `/tmp` and only succeeds under the session's own workspace directory — the
same restriction recorded in this project's prior sessions). All four killed their declared test
exactly as described:

| mutation id | target test | result |
|---|---|---|
| `merge-placed-at-tick-start-lockstep` | `schedule.expand` | **KILLED** — `Expand.MergeNeverPrecedesItsProducers` fails at `mode=Auto` |
| `merge-placed-at-tick-start-general` | `schedule.expand` | **KILLED** — same test fails at `mode=ForceGeneral` only, as documented |
| `merge-not-folded-past-three-writers` | `schedule.plan` | **KILLED** — `ExecutionPlan.FiveWriterMergeFoldsToPairwiseMergesAndSchedules` reproduces the exact pre-fix `E-SCHED-ARITY` diagnostic |
| `merge-typed-at-first-writer-v2` | `schedule.acyclicity_sweep` | **KILLED** — 53/2000 disagreements against the independent cycle oracle; confirmed the mutation is a no-op against `schedule.plan`'s fixed fixtures, exactly as the mutation's own `describes` field claims |

The extended sweep was also re-run directly at `FUSE_SWEEP_GRAPHS=20000`: 20,000 graphs, 0
disagreements, 6,198 routed through `schedule()` — matching the fix report's own count.

## Verification

| unified id | verdict | evidence |
|---|---|---|
| R-26 (IR-F1, MAJOR) | CONFIRMED-FIXED | Unchanged since the second-pass review, re-confirmed here: `test/ir/golden_ir_test.cpp` exists, registered (`test/CMakeLists.txt`), and its 4 cases pass under `ctest -R ir.golden_ir`. `git log` on the three `test/golden/*.ir` fixtures shows no commit since the R-26 fix (`ff91253`) — they have not drifted. |
| R-27 (IR-F2, MAJOR) | CONFIRMED-FIXED | `src/ir/lowering.cpp`'s combine step still folds 4+ writers into a left-to-right chain of 2-input `Merge` nodes with a once-computed `newest_type`. Re-ran the fixed tests directly (`ExecutionPlan.FiveWriterMergeFoldsToPairwiseMergesAndSchedules`, `...KeepsNewestTimeLevelThroughTheFoldWhenOneIsLagged`) — pass. Re-applied `merge-not-folded-past-three-writers` myself (table above) — kills as documented. This finding's fix was touched again by the second fix pass (see N-ir-schedule-2 below): the fold itself is unchanged, but a defect in how its OUTPUT was scheduled was found and fixed one layer downstream. |
| R-28 (IR-F3, MAJOR) | CONFIRMED-FIXED | `include/fuse/schedule/buffers.hpp:85-97` (current line numbers) names `fuse::InvalidArgument` uniformly and explicitly disclaims the old three-way `std::` split; `src/schedule/buffers.cpp` is unchanged and was already correct. `Buffers.PlanBuffersThrowsFuseInvalidArgumentNotTheDocumentedStdTypes` passes under `ctest -R schedule.buffers` (18/18). |
| R-43 (IR-F4, MINOR) | CONFIRMED-FIXED | The two-pass tie-break (max-`iterate` reduction, then anchor-agreement scan restricted to the tied writers) is unchanged in current `src/ir/lowering.cpp`, refuses with `E-IR-MERGE-ANCHOR` on genuine ambiguity. Re-applied `merge-typed-at-first-writer-v2` myself (table above) — kills `schedule.acyclicity_sweep` as documented, and confirmed by hand that it is a no-op against `schedule.plan`'s fixtures (the sweep is the only oracle for this mutation, exactly as its `describes` field states). Still latent by construction: `grep -rn "Anchor::" include src` (excluding the enum and its two `WindowEnd`-only constructors) finds no call site producing a non-`WindowEnd` anchor. |
| R-44 (IR-F5, MINOR) | CONFIRMED-FIXED | `src/schedule/scheduler.cpp:599-618`'s comparison is byte-identical to `bdf4346`. `Epoch.MismatchedRankSetsAreRefused` in `test/schedule/epoch_test.cpp` splices a mismatched `Transport`/`Epoch` pair (arc `ValueType` copied by value before either `add_arc()` call, correctly avoiding the reference-invalidation hazard `dag.hpp` documents) and passes under `ctest -R schedule.epoch` (5/5). |
| N-ir-schedule-1 (IMPROVEMENT, from `re-ir-schedule.md`) | CONFIRMED-FIXED | `test/schedule/acyclicity_sweep_test.cpp` now routes every non-lagged generated graph through `schedule()` in addition to `prove_acyclic()`, asserting agreement with the same independent cycle oracle. Re-ran directly at `FUSE_SWEEP_GRAPHS=20000`: 0 disagreements, 6,198/20,000 graphs scheduled (comfortably over the `n/50` evidence floor the test enforces). |
| N-ir-schedule-2 (BLOCKER, self-reported in `fix2-ir-schedule.md`, not named by `re-ir-schedule.md`) | CONFIRMED-FIXED | This was a real, previously-undetected defect, not just a documentation claim: before this fix, **every** `Merge` node (any writer count, not only the 4+-writer fold) was scheduled at its owning component's tick *start* by `expand()`'s "no channel claimed" boundary rule — the convention meant for `Source`/an unwritten `Sink` — whenever any of its writers required a real cross-rank `Transport`/`Epoch` pair. The existing `five_writer_merge()` fixture (all writers colocated with the target) could not expose this because with no `Transport`/`Epoch` node in the graph, `Merge` and its inputs land at the same instant and `NodeId` order alone kept it after them by accident. Read `src/schedule/expand.cpp`'s current `emit_lockstep()` and general-path boundary loop directly (both quoted in "Not found" below) and confirmed the fix: `Merge` is now stamped at the tick's *end* in both paths. Re-ran `Expand.MergeNeverPrecedesItsProducers` and `ExecutionPlan.MergeAcrossRanksSchedulesInDependenceOrder` directly — pass in both `ExpandMode::Auto` and `ExpandMode::ForceGeneral`. Re-applied both of the fix's own mutations myself (table above) — both kill. This is Tier 0, no MPI: `merge_across_ranks()`'s two "different ranks" are `RankSet` values compiled and scheduled statically; no `mpiexec` involved. |

All five first-pass findings, the one improvement, and the one self-reported blocker are
**CONFIRMED-FIXED** with no PARTIAL, NOT-FIXED, or REJECTION-UPHELD verdict in this pass.

## New findings

No BLOCKER, MAJOR, or MINOR code defect found in this pass. One MINOR documentation-accuracy
finding.

| id | severity | file:line | title |
|---|---|---|---|
| N-ir-schedule-3 | MINOR | `REVIEW.md:313` | The pass-2 summary table's `ir-schedule` row omits the BLOCKER (N-ir-schedule-2) the second fix pass found and fixed, unlike the analogous late-discovered items recorded for other layers |

### N-ir-schedule-3 — MINOR — `REVIEW.md`'s per-layer summary table under-records this layer's findings

**Where:** `REVIEW.md:313` (the `ir-schedule` row of the "Totals across the six pass-2 reports"
table, §6, "Second pass (2026-09-20)").

**What is wrong, and how it was confirmed.** The row currently reads:

```
| ir-schedule | 5 (R-26, R-27, R-28, R-43, R-44) | 0 | 0 | 1 — N-ir-schedule-1 IMPROVEMENT |
```

This is the exact count from `re-ir-schedule.md` (the second-pass *review*, at `dbdc872`), but it
predates and does not reflect `pass2/fix2-ir-schedule.md` (the second *fix* pass), which
self-reported and fixed a BLOCKER, N-ir-schedule-2, discovered while implementing
N-ir-schedule-1's coverage extension (verified CONFIRMED-FIXED above). `git log --format="%h %ad
%s" -- REVIEW.md` shows the table's most recent edits (`1a87f35`, `acf3c5b`) both post-date the
commit that fixed N-ir-schedule-2 (`c92ad1c`, same day, ~13 hours earlier) — so this is not a
question of the table predating the fix; it was edited twice after the fix landed and both edits
left this row's "New findings" count and BLOCKER/MAJOR/MINOR/IMPROVEMENT breakdown unchanged. Three
other layers' rows in the *same* table already carry exactly this kind of late addition, so the
gap is inconsistent with the table's own established convention, not merely incomplete by
default: `build-tools-docs`'s row lists "`-9` BLOCKER (host-reported tier-2 configure failure,
2026-09-20; fixed)" and "`-10` MINOR (host-reported, same date; fixed)", and `backend`'s and
`transport-runtime`'s rows both carry post-review BLOCKER/MAJOR items in their own "New findings"
cells. `ir-schedule`'s row is the one row in the table that had a fix-pass-discovered BLOCKER and
did not get the same treatment.

**Why this is MINOR, not higher.** The underlying code defect (N-ir-schedule-2) is real, but it is
already fixed and independently verified in this pass (mutation-killed both ways, tests re-run
directly). No reader relying on the code, the tests, or the mutation-evidence files is misled —
only a reader relying on `REVIEW.md`'s own summary table as a complete count of this layer's
findings would undercount by one BLOCKER. Given `docs/dev/testing.md`'s five-states philosophy and
this project's own repeated insistence on records that say exactly what is true (`I10`; the
existing `acf3c5b` commit exists specifically to correct a wrong total written "from memory"), a
summary table that is off by a BLOCKER in the direction of *undercounting severity* is worth
correcting rather than leaving as a rounding error.

**The fix.** Change `REVIEW.md:313` to:

```
| ir-schedule | 5 (R-26, R-27, R-28, R-43, R-44) | 0 | 0 | 2 — N-ir-schedule-1 IMPROVEMENT, N-ir-schedule-2 BLOCKER (found during N-ir-schedule-1's own fix; fixed) |
```

matching the parenthetical style the `build-tools-docs` row already uses for its own host-reported,
already-fixed BLOCKER. This is a one-line documentation edit outside every path this layer group
owns (`REVIEW.md` is a root file, not under `include/fuse/ir`, `src/ir`, `test/ir`,
`include/fuse/schedule`, `src/schedule`, `test/schedule`, or `test/golden`), so it is reported here
rather than made here, consistent with this being a read-only review.

**Test to add:** none — this is a documentation-only correction with no executable claim to pin.

## Not found

Checked deliberately in this pass, beyond re-confirming the six verification rows above, and found
sound:

- **The `Merge`-scheduling fix's arc-lifetime safety.** `newest_type` (the single `ValueType` every
  fold-chain node carries) is captured as a `const ValueType` value, not a reference, before any of
  the fold's `add_arc()` calls (`src/ir/lowering.cpp`) — confirmed by direct reading, and consistent
  with `dag.hpp`'s documented hazard that a reference into the DAG's internal vectors does not
  survive a later mutating call on the same DAG (the same hazard the R-44 fixer's first attempt at
  `epoch_test.cpp` hit, per its own report, and that the new `Epoch.MismatchedRankSetsAreRefused`
  and `Expand.MergeNeverPrecedesItsProducers`/`ExecutionPlan.MergeAcrossRanksSchedulesInDependenceOrder`
  tests all correctly avoid).
- **Which node kinds ever carry `ChannelId{}` (no single owning channel).** `grep -n "ChannelId{}"
  src/ir/*.cpp` finds exactly three call sites: `Source` (line 107), and the two `Merge`
  `add_node()` calls (the ≤3-writer single node and the >3-writer fold, lines 238 and 252). A
  fourth case, an unwritten `Sink` for a `Bidirectional` field no channel targets, gets the same
  default-constructed `origin` by the `for` loop at line ~283 finding no match. `expand.cpp`'s fix
  distinguishes `Merge` from the other two (`Source`, unwritten `Sink`) correctly in both paths —
  confirmed there is no fourth "no channel claimed" node kind the fix's `n.kind == OpKind::Merge`
  check needed to but did not cover.
- **Whether the fix depends on lag.** `grep -n "lag\b\|offset" src/schedule/expand.cpp` finds no
  match — `expand()` does not special-case a lagged arc's *timestamp*, only its `ValueType` (used
  elsewhere, by the acyclicity prover and buffer persistence, not by `expand()`). The
  `tick_closing` classification depends only on `n.kind`, so the fix's correctness argument does not
  depend on whether a `Merge`'s inputs are lagged. No fixed or random test currently combines a
  lagged writer with a cross-rank one into the same `Merge` (`five_writer_merge()`'s lagged fixture
  keeps every writer colocated; the sweep's extension explicitly excludes lagged graphs from the
  `schedule()` check), so this combination is untested, but by the above it is not a suspected
  defect — the same "provably latent, not a live defect" status the first review gave F4, not a new
  finding.
- **The `boundary_owner()` component for a `Merge` node.** Uses the merge output arc's own
  `rank_set` (the merged field's placement), the same mechanism `Source` uses for its own output —
  confirmed by reading `boundary_owner()` directly. Multirate-safe by the same construction that
  already made `Source`/`Sink` multirate-safe: each owning component's own clock interval is looked
  up independently, not the graph-wide lockstep period.
- **`schedule.lockstep`'s byte-for-byte `Auto`-vs-`ForceGeneral` identity check.** Still passes after
  the fix; the fixer's own report additionally hand-checked that `merge_across_ranks()` produces an
  identical `TimedDag` node list under both modes, since no existing `lockstep_test.cpp` fixture
  contains a cross-rank `Merge`. Re-ran `schedule.lockstep` directly here — unaffected (9 cases, all
  pass; test count from `ctest -R schedule.lockstep --output-on-failure`).
- **Test/build registration.** `test/CMakeLists.txt`'s own `FATAL_ERROR` audit for unregistered
  `*_test.cpp` files did not fire during any of this review's three separate configures (the main
  `b0` tree and the isolated mutation-verification copy, configured twice). Separately, confirmed
  that `test/schedule/buffers_test.cpp` and `test/schedule/slab_test.cpp` — unregistered and never
  compiled at the time of the *first* review (`bdf4346`; fixed by a `build-tools-docs`-layer commit,
  `b8e3f54`, before the ir-schedule fix pass began) — are registered and passing now; that fix
  predates and is independent of the ir-schedule fix pass, and this layer group's own fix reports
  correctly do not claim credit for it.
- **`E-SCHED-ARITY`'s reachability post-fix.** Confirmed by construction, same as the second-pass
  review found: `grep -n "add_node(OpKind::" src/ir/*.cpp` shows every `Merge` emission is capped at
  3 or exactly 2 inputs, so the arity check is exercisable only through a hand-built DAG
  (`plan_test.cpp`'s `RejectsAnOperatorWithMoreInputsThanAnInstructionHolds`), which is expected and
  matches the diagnostic's own wording ("lower Merge as a left-fold of pairwise merges") now being
  accurate advice for whoever calls `schedule()` directly.
- **Mutation-anchor freshness for this layer.** `tools/mutate.py --check-anchors --build-dir b0` at
  current HEAD reports 95/95 valid, 0 stale, project-wide (up from the second-pass review's 86/86,
  reflecting other layers' later mutation additions); nothing in `ir-schedule.toml` is stale.
- **A note on HEAD drift.** This task named HEAD `ced1945`; by the time this review started reading
  files, the shared branch had advanced one commit to `acf3c5b`, a `REVIEW.md`-only edit (see
  N-ir-schedule-3's own evidence) from a concurrent session with no changes under this layer group's
  paths. All verification and mutation re-testing in this report was performed against the live
  checkout at `acf3c5b`; `git log --oneline -- <any path in this layer group>` confirms no commit
  between `ced1945` and `acf3c5b` touches this layer group, so this drift changes nothing about the
  findings above — noted for completeness, per this task's own instruction not to cut scope
  silently.

## Summary

**0 BLOCKER, 0 MAJOR, 1 MINOR (N-ir-schedule-3, documentation-only), 0 IMPROVEMENT, 0 DEVIATION**
found against `include/fuse/ir`, `include/fuse/schedule`, `src/ir`, `src/schedule`, and their tests
at HEAD `acf3c5b`. All five first-pass findings (R-26/R-27/R-28 MAJOR, R-43/R-44 MINOR), the one
second-pass improvement (N-ir-schedule-1), and the one BLOCKER the fixer found and fixed
mid-second-fix-pass (N-ir-schedule-2) are **CONFIRMED-FIXED**, each re-verified directly in this
pass by re-running the named test and, for the four mutations this layer's evidence file claims,
by re-applying every one of them myself against an isolated copy of the source tree and confirming
each kills its declared test. `tier0` builds clean with no build directory under the repository;
all 18 `ir.*`/`schedule.*` tests pass; both static gates (`check-invariants`, `check-layering`)
report zero violations; the mutation-anchor check reports 95/95 valid, 0 stale project-wide. The one
new item (N-ir-schedule-3) is a one-line inconsistency in `REVIEW.md`'s own summary table, not a
code defect, and is reported rather than fixed because `REVIEW.md` is outside every path this layer
group owns.

**Bottom line for this layer group: nothing of MAJOR or higher severity remains.** The one BLOCKER
that existed in this scope during the review sequence (N-ir-schedule-2, a real scheduling-order
defect affecting any cross-rank `Merge`, not only the 4+-writer fold) was found and fixed within the
sequence itself and is confirmed fixed here, independently, with re-executed mutation evidence.
