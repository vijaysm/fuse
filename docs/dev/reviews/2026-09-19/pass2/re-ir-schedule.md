<!-- Second-pass review, 2026-09-19/20, HEAD dbdc872, branch review-2026-09-19-fixes.
     First review at bdf4346: docs/dev/reviews/2026-09-19/ir-schedule.md.
     Fixer report:            docs/dev/reviews/2026-09-19/fixes/fix-ir-schedule.md. -->

# Second-pass review: `include/fuse/ir`, `src/ir`, `test/ir`, `include/fuse/schedule`, `src/schedule`, `test/schedule`, `test/golden`

Reviewed at HEAD `dbdc872e4c8e026e88ea65c1af91de7b7c21b0be`. Configured and built the `tier0`
preset (`cmake --preset tier0 -S <repo> -B $PWD/b0 && cmake --build b0 -j4`), then ran
`ctest --test-dir b0 -R '^ir\.|^schedule\.'`, `python3 tools/check-invariants.py`,
`python3 tools/check-layering.py`, and `python3 tools/mutate.py --check-anchors --build-dir b0`
(anchor check only — a full mutation run edits repository sources in place before restoring
them, which is not safe to run against a read-only checkout shared by other reviewers; the
fixer's own hand-applied-mutation evidence, reproduced below per finding, stands in for it).
No git command that changes repository state was run, no build directory was created under the
repo, and nothing under `/Users/mahadevan/Code/AI/coupling/fuse` was edited — `git status` is
clean at the end of this review. Confirmed no MPI dependency in this scope: every `ir.*` and
`schedule.*` target links `fuse::schedule`/`fuse::ir` only, none links `MPI::MPI_CXX`, and all 18
ran directly in this sandbox — nothing in this layer group needed a host run.

## Verification

| unified id | verdict | evidence |
|---|---|---|
| R-26 (IR-F1, MAJOR) | CONFIRMED-FIXED | `test/ir/golden_ir_test.cpp` exists, is registered in the `ir` foreach loop (`test/CMakeLists.txt:131`), and its 4 cases (`PairExplicit`, `PairAccumulateRemapRelax`, `PairIterated`, `OperatorOrderIsObservable`) pass under `ctest -R ir.golden_ir` and under a direct `gtest_filter` run. All three `test/golden/*.ir` fixtures now contain an `Epoch` node; diffing `bdf4346..dbdc872` on the three files shows the change is exactly one inserted `Epoch` line per fixture plus the consequent node/arc renumbering — no operator, order, or numeric value changed. `dump.hpp`'s locale-pinning and attribute-rendering code (`reason=...`) is unchanged and untouched by this fix, so the byte-exact comparison basis is the same as at the first review. |
| R-27 (IR-F2, MAJOR) | CONFIRMED-FIXED | `src/ir/lowering.cpp`'s combine step now folds writer counts above `kMaxMergeInputs` (3, `static_assert`-tied to `schedule`'s `Instruction::in` size via a duplicated constant, since `ir` cannot depend on `schedule`) into a left-to-right chain of 2-input `Merge` nodes, each carrying the overall-newest `ValueType` computed once before folding — read the diff and the current source directly. `ExecutionPlan.FiveWriterMergeFoldsToPairwiseMergesAndSchedules` (five real writers through `lower()`, not hand-built) and `ExecutionPlan.FiveWriterMergeKeepsNewestTimeLevelThroughTheFoldWhenOneIsLagged` both pass, confirmed by a direct `gtest_filter` run in addition to the full-suite `ctest`. The old hand-built `RejectsAnOperatorWithMoreInputsThanAnInstructionHolds` still passes and still exercises `E-SCHED-ARITY` directly (see "Not found" for what this now implies about that check's reachability). |
| R-28 (IR-F3, MAJOR) | CONFIRMED-FIXED | `include/fuse/schedule/buffers.hpp:85-97`'s docstring now names `fuse::InvalidArgument` uniformly and explicitly disclaims the old three-way `std::` split. `src/schedule/buffers.cpp`'s implementation (unchanged by this fix, correctly so — it was already right) still converts every failure through one `FUSE_THROW(InvalidArgument, ...)`. The new test `Buffers.PlanBuffersThrowsFuseInvalidArgumentNotTheDocumentedStdTypes` exercises all three previously-documented failure modes (empty `RegionSizes`, a `TimedDag` naming an unknown node, an overflowing running total) and asserts none is catchable as `std::invalid_argument`/`std::out_of_range`/`std::overflow_error`; confirmed passing directly. |
| R-43 (IR-F4, MINOR) | CONFIRMED-FIXED | The tie-break in `src/ir/lowering.cpp` is now two passes: a reduction over all writers for the greatest `iterate` (anchor-independent), then a scan restricted to writers tied at that `iterate` that flags `anchor_ambiguous` on any disagreement and refuses the combine with a new code, `E-IR-MERGE-ANCHOR`, instead of silently keeping the first-declared writer. Traced the two-pass shape by hand against the counterexample the fix report cites (writers at `(iterate=1,A), (iterate=1,B), (iterate=2,C)` in declaration order): the max-iterate pass correctly isolates `C` before any anchor comparison runs, so a single combined pass's false-ambiguity failure mode is avoided. Confirmed still latent exactly as claimed: `grep -rn "Anchor::" include src` (excluding the enum definition and its two `WindowEnd`-only named constructors) finds no call site constructing a non-`WindowEnd` anchor, so no test through `lower()`'s public surface can reach `E-IR-MERGE-ANCHOR` today; the in-code comment states this unreachability and why, which is the documented-alternative-to-a-test option the first review offered for this exact situation (its own §3, F5's fix). This is a stronger fix than the first review asked for: F4 asked only that the ambiguity be treated conservatively or documented; the fixer instead made it a refusal. |
| R-44 (IR-F5, MINOR) | CONFIRMED-FIXED | No code change, matching the first review's option (a)/(b) choice — `src/schedule/scheduler.cpp:599-618`'s comparison is byte-identical to `bdf4346`. `Epoch.MismatchedRankSetsAreRefused` in `test/schedule/epoch_test.cpp` splices a `Transport`/`Epoch` pair with a deliberately mismatched `dst_rank_set` (copying the shared arc `ValueType` by value before either `add_arc()` call, correctly avoiding the reference-invalidation hazard `dag.hpp` documents) and asserts `E-SCHED-EPOCH-PARTIAL` fires; confirmed passing directly. This is a Tier 0, no-MPI test — it constructs the DAG directly and calls `schedule()`, so the rank-agreement *check* is exercised without needing a running MPI job; no MPI dependency to flag for this finding. |

All five are also covered by the full-suite run: `ctest --test-dir b0 -R '^ir\.|^schedule\.'` →
**18/18 passed**, matching the fixer's own count. `check-invariants` and `check-layering` both
report 0 violations. `mutate.py --check-anchors` reports **86 valid, 0 stale** (the fixer's report
recorded 73/2-stale at their commit; both stale anchors they flagged as belonging to other layers
— `merge-typed-at-first-writer` and the unrelated `mesh.hpp` anchor — are gone from
`build-evidence/mutations.toml` and `mutations/ir-schedule.toml` now carries the two mutations
(`merge-typed-at-first-writer-v2`, `merge-not-folded-past-three-writers`) with hand-verified
kill evidence for R-27/R-43, consistent with a later commit by another fixer/the lead closing
out the two items the ir/schedule fixer flagged as "needs another owner"). I did not re-run the
mutation apply/build/revert cycle myself (see header); the hand-verified-kill evidence in
`build-evidence/mutations/ir-schedule.toml` and the fix report is what this verdict rests on for
the mutation-evidence half of R-27/R-43's acceptance criteria (`I9`).

## New findings

No BLOCKER, MAJOR, or MINOR findings. No DEVIATION. One IMPROVEMENT, carried forward rather than
newly discovered (see below).

| id | severity | file:line | title |
|---|---|---|---|
| N-ir-schedule-1 | IMPROVEMENT | `test/schedule/acyclicity_sweep_test.cpp:147` | The random-merge acyclicity sweep still only calls `prove_acyclic()`, never `schedule()`, so it gives the new left-fold no independent-oracle coverage of its own even though the sweep already generates random `Merge` arities |

### N-ir-schedule-1 — IMPROVEMENT — the acyclicity sweep never reaches `schedule()`, so it cannot exercise the left-fold it now indirectly creates

**Where:** `test/schedule/acyclicity_sweep_test.cpp:116` (random merge generation), `:147`
(`return prove_acyclic(g, lowered.dag, td.value());` — the sweep's only call into the scheduling
layer).

**What is wrong, and how it was confirmed.** This is not a new defect; it is the first review's
own §5 item 6 ("General mutation suggestion... independent of the findings above"), which was a
suggestion attached to no unified id and so was not in the fixer's worked list — confirmed by
reading the fix report's "Findings worked" line (`R-26, R-27, R-28, R-43, R-44` only) and its
"Rejected" section (`None`), neither of which mentions this suggestion. It remains true after the
fix: `grep -n "schedule(\|prove_acyclic" test/schedule/acyclicity_sweep_test.cpp` finds the
generator calls `g.merge(dst, static_cast<MergeOp>(rng() % 3))` with a randomly chosen writer
count, then the harness's only downstream call is `prove_acyclic()` — never `schedule()`. Before
this fix, a random sweep case with more than 3 writers into one merged field would lower to a
single N-ary `Merge` node exactly as any hand-authored graph would, so `prove_acyclic()`'s
handling of that node was fuzzed the same way either way. After this fix, the *lowered shape* for
such a case is a chain of 2-input `Merge` nodes rather than one N-ary node — a materially
different graph — but because the sweep still stops at `prove_acyclic()`, it exercises the
acyclicity prover's handling of the new chain shape without ever exercising `schedule()`'s
handling of it (buffer planning, instruction ordering, the arity check now permanently vacuous
for lowered graphs — see "Not found" below). The two targeted tests added for R-27
(`FiveWriterMergeFoldsToPairwiseMergesAndSchedules`,
`...KeepsNewestTimeLevelThroughTheFoldWhenOneIsLagged`) cover the fold's correctness with
hand-picked fixed graphs; this sweep is what would have covered it with many random ones, the
same value the base lag-cutting sweep gives the rest of the prover, and it still does not.

**Why this is IMPROVEMENT, not MINOR or MAJOR.** Nothing observed in this pass suggests the fold
is wrong for any case the random generator would produce beyond what the two new fixed tests
already cover (the fold's correctness argument — newest-type computed once, applied uniformly —
does not depend on writer count or which pairs are chosen first). This is a coverage gap in the
same family as the original suggestion, not a suspected defect, and the original review did not
raise it to finding status either.

**The fix.** As the first review already specified: add a second, narrower sweep — or extend the
existing generator's downstream call — so that at least the lockstep, non-lagged, non-region
subset of its random graphs also passes through `schedule()`, the way `plan_test.cpp`'s
`FiveWriterMerge*` tests do for their two fixed cases, giving the fold the same random-oracle
coverage the sweep already gives the acyclicity prover's lag-cutting rule.

**Test to add:** none beyond the fix itself — the sweep, once its downstream call includes
`schedule()`, is its own test.

## Not found

Checked deliberately in this pass and found sound, beyond what "Verification" above already
covers per finding:

- **The merge fold's arc-lifetime safety.** `newest_type` (used for every node in the fold chain)
  is captured as a `const ValueType` **value**, not a reference, before any of the fold's
  `add_arc()` calls — confirmed by reading `src/ir/lowering.cpp:212-213` directly, which matters
  because `dag.hpp`'s own doc comment on `arc()`/`node()` warns a reference into the DAG's
  internal vectors does not survive a later mutating call on the same DAG (the exact hazard
  `epoch_test.cpp`'s new `MismatchedRankSetsAreRefused` test's own comment names, and that the
  fixer's report says a first attempt at that test actually hit). The two-pass tie-break loop
  above it also takes only a `const TimeLevel&` reference at a time when no `add_arc`/`add_node`
  call is interleaved, so no dangling read there either.
- **Determinism of the fold across ranks.** The combine step iterates `targets` (field ids) in
  sorted order and, within a field, folds `writers` in the order `ctx.field_writers` recorded them
  — channel declaration order, unchanged by this fix — so the fold produces an identical node
  sequence, and identical labels (`"…/merge1"`, `"…/merge2"`, …), on every rank compiling the same
  graph. Every rank independently compiles this plan, so this determinism is load-bearing per
  `include/fuse/ir/lowering.hpp`'s own stated invariant; nothing in the fold introduces
  rank-dependent iteration (no pointer- or hash-order-derived sequencing anywhere in the new code).
- **Whether `E-SCHED-ARITY` remains reachable through `lower()` at all after this fix.** It does
  not, by construction: `grep -n "add_node(OpKind::" src/ir/*.cpp` shows every call site other
  than the bounded `Merge` combine (capped at 3 or exactly 2 inputs) passes 0 or 1 input, and the
  three reserved multiscale kinds are refused before scheduling per the first review's own
  "Not found" section. `E-SCHED-ARITY` is therefore now exercisable only through a hand-built DAG
  — exactly the situation `RejectsAnOperatorWithMoreInputsThanAnInstructionHolds` already covers,
  and structurally the same "unreachable via the compiler, deliberately tested via a spliced DAG"
  pattern the fix for R-44 established for `E-SCHED-EPOCH-PARTIAL`. Not a new gap: before this fix
  the check *was* reachable via `lower()` (that reachability is what F2 reported as a user-facing
  bug); the fix's job was to make it unreachable from real graphs, and it now is.
- **Interaction between the merge fold and the rank-agreement scan.** The Transport/Epoch pairing
  scan in `scheduler.cpp` (`E-SCHED-EPOCH-MISSING`/`-PARTIAL`) matches an `Epoch` to a `Transport`
  by exact output-arc identity (`candidate.in[0] != move.out`). The merge combine step operates on
  already-fully-lowered channel arcs (post-Transport/Epoch/Remap/Constrain/Diagnose/Relax), never
  on a raw `Transport` output directly, and creates no new `Transport`/`Epoch` node itself — so the
  fold cannot create a second consumer of a `Transport` output that would confuse this scan's
  first-match search. Confirmed by reading `rules_channel.cpp`'s emission order (unchanged by this
  fix) against the combine step's input list (writer arcs recorded in `field_writers`, which are
  each a completed channel's final arc).
- **The acyclicity prover's genericity over node arity.** `grep -n "inputs.size()"
  src/schedule/acyclicity.cpp` finds no arity-specific branch (the one hit is an unrelated
  cycle-length display cap) — the prover's traversal is generic over however many inputs a node
  has, so going from one N-ary `Merge` node to a chain of 2-input ones is not a shape the prover
  needed new-casing for. Consistent with `schedule.acyclicity` and `schedule.acyclicity_sweep`
  both passing unchanged.
- **Test/build registration for everything touched.** `test/ir/golden_ir_test.cpp` is registered
  (`test/CMakeLists.txt:131`), and the project-wide registration audit
  (`test/CMakeLists.txt:278-300`, a `FATAL_ERROR` on any unregistered `*_test.cpp`) did not fire
  during configure — confirming no test file anywhere in the tree, not just in this layer group,
  is currently orphaned.
- **`E-SCHED-ARITY`'s own diagnostic wording**, still "lower Merge as a left-fold of pairwise
  merges" (`scheduler.cpp:441`) — no longer misleading as F2 found it (the advice used to name a
  step that did not exist); it now correctly describes what `lower()` actually does, and the
  advice is directed at whoever calls `schedule()` on a hand-built DAG that skipped `lower()`,
  which is exactly the population left able to trigger it.

## Summary

**0 BLOCKER, 0 MAJOR, 0 MINOR, 1 IMPROVEMENT (N-ir-schedule-1), 0 DEVIATION** filed against
`include/fuse/ir`, `include/fuse/schedule`, `src/ir`, `src/schedule`, and their tests at HEAD
`dbdc872`. All five first-review findings for this layer group (R-26/R-27/R-28 MAJOR, R-43/R-44
MINOR) are **CONFIRMED-FIXED**: the golden IR-dump suite is ported, registered and passing with
regenerated fixtures whose only change is the expected `Epoch` node (R-26); the documented
left-fold of pairwise merges is now actually implemented, closing the gap between an ordinary
5+-writer `Merge` use and a misleading `E-SCHED-ARITY` failure (R-27); `plan_buffers()`'s
docstring now names the exception type the code has always thrown (R-28); the merge combine's
anchor tie-break now refuses an ambiguous case instead of silently guessing, a stronger fix than
requested (R-43); and the epoch rank-set-mismatch guard has a regression test proving it fires
(R-44). `tier0` builds clean, all 18 `ir.*`/`schedule.*` tests pass, both static gates
(`check-invariants`, `check-layering`) report zero violations, and the mutation-anchor check
reports 86/86 valid with the two stale anchors the ir/schedule fixer flagged as belonging to other
owners now resolved. The one new item (N-ir-schedule-1) is a coverage improvement already named
in the first review's own suggestions list, confirmed still open, not a defect found in this pass.
