# Fix report: `include/fuse/ir`, `include/fuse/schedule`, `src/ir`, `src/schedule`

Findings worked: R-26 (IR-F1, MAJOR), R-27 (IR-F2, MAJOR), R-28 (IR-F3, MAJOR),
R-43 (IR-F4, MINOR), R-44 (IR-F5, MINOR). All five confirmed against the cited
lines and fixed; none rejected. Order worked: BLOCKER → MAJOR → MINOR, i.e.
R-26, R-27, R-28, then R-43/R-44 (R-43's *code* fix landed inside the same
edit as R-27, since both touch the same six-line block; see "Commit units").

Built and tested with the `tier0` preset (no MPI needed for this scope) at
`$PWD/build-t0-irsched`. All three required gates are clean for my files:
`python3 tools/check-invariants.py` → 0 violations; `python3
tools/check-layering.py` → 0 violations; `python3 tools/mutate.py
--check-anchors --build-dir <build>` → 73 valid / 2 stale of 75 (both stale
anchors are outside my ownership — see "Needs another owner"). All 18
`ir.*`/`schedule.*` gtest targets pass (`ctest -R '^ir\.|^schedule\.'` → 18/18,
0 failed), including the newly-registered `ir.golden_ir`.

## Commit units

```json
[
  {
    "ids": ["R-26"],
    "files": [
      "test/ir/golden_ir_test.cpp",
      "test/CMakeLists.txt",
      "test/golden/pair_explicit.ir",
      "test/golden/pair_accumulate_remap_relax.ir",
      "test/golden/pair_iterated.ir"
    ],
    "message": "Port the golden IR-dump regression test and refresh its fixtures\n\nPorts upstream's GoldenIr suite so the checked-in .ir dumps are actually compared byte-for-byte, and regenerates all three fixtures to add the Epoch node the current lowering already emits."
  },
  {
    "ids": ["R-27", "R-43"],
    "files": [
      "src/ir/lowering.cpp",
      "test/schedule/plan_test.cpp",
      "build-evidence/mutations/ir-schedule.toml"
    ],
    "message": "Fold oversized Merge nodes into a left-fold and refuse ambiguous anchors\n\nImplements the documented left-fold of pairwise merges for 4+ writers and refuses a combine whose writers tie on iterate but disagree on anchor, instead of silently keeping whichever writer was declared first."
  },
  {
    "ids": ["R-28"],
    "files": [
      "include/fuse/schedule/buffers.hpp",
      "test/schedule/buffers_test.cpp"
    ],
    "message": "Fix plan_buffers()'s docstring to name the exception it actually throws\n\nCorrects buffers.hpp's throws-clause from three unrelated std:: exception types to the fuse::InvalidArgument every path actually throws, and pins the type with a test."
  },
  {
    "ids": ["R-44"],
    "files": ["test/schedule/epoch_test.cpp"],
    "message": "Add a regression test for the epoch rank-set-mismatch guard\n\nSplices a Transport/Epoch pair with disagreeing rank sets so E-SCHED-EPOCH-PARTIAL's check is proven reachable and correctly worded, not just read as sound."
  }
]
```

Each unit builds and passes standalone given the units before it (verified:
unit 1 touches no file unit 2 depends on and vice versa; unit 2's new tests
in `plan_test.cpp` require the `lowering.cpp` fix in the *same* unit; units 3
and 4 are independent of 1 and 2). R-27's MAJOR severity places its unit
ahead of R-28's, even though R-43 (MINOR) rides along in the same file edit —
splitting R-27 and R-43 into separate commits was not practical: the fix
restructures one six-line tie-break loop that both findings share (the
left-fold threshold check sits immediately after the anchor guard and reuses
its `newest`/`newest_type` locals).

## Fixed

**R-26 (IR-F1, MAJOR) — golden IR-dump suite never ported; fixtures stale.**
Confirmed exactly as described: `grep` for the three fixture basenames found
zero references outside `upstream/soma/` and the fixtures themselves, and all
three checked-in `.ir` files lacked an `Epoch` node although every one lowers
through the `ranks_differ` branch that emits one.
Fix: ported `upstream/soma/test/golden_ir_test.cpp` to `test/ir/golden_ir_test.cpp`
(`soma::`→`fuse::`, `SOMA_UPDATE_GOLDEN`→`FUSE_UPDATE_GOLDEN`,
`SOMA_TEST_*_DIR`→`FUSE_TEST_*_DIR`, matching the macros `test/CMakeLists.txt`
already defines), registered it in the `ir` test group, and regenerated the
three golden files with `FUSE_UPDATE_GOLDEN=1`. Reviewed the diff by hand
(reproduced below) — every changed line is a renumbering caused by the one
new `Epoch` node; no operator, order, or numeric value changed:
```
 node 1   Transport    %0           -> %1 : ...  "flux/transport" {0->1}
-node 2   Remap        %1           -> %2 : ...  "flux/remap" {remap=C}
+node 2   Epoch        %1           -> %2 : ...  "flux/epoch" {reason=after-transport}
+node 3   Remap        %2           -> %3 : ...  "flux/remap" {remap=C}
```
Verified: `ctest -R ir.golden_ir` → 4/4 pass (`PairExplicit`,
`PairAccumulateRemapRelax`, `PairIterated`, `OperatorOrderIsObservable`); ran
once before regenerating to confirm the pre-fix mismatch was exactly the
`Epoch` line, and again after to confirm a clean pass.

**R-27 (IR-F2, MAJOR) — "left-fold of pairwise merges" documented, never
implemented.** Confirmed: `src/ir/lowering.cpp`'s combine step built one
`Merge` node with the full writer list regardless of count, and
`grep -rn "left-fold\|pairwise"` found zero decomposition logic anywhere.
Fix: in the combine step, writer counts ≤ `kMaxMergeInputs` (3, duplicated
from `plan.hpp`'s `Instruction::in` size since `ir` cannot depend on
`schedule`) still produce one `Merge` node unchanged; counts above that fold
left-to-right into a chain of 2-input `Merge` nodes, each carrying the
*overall* newest contribution's `ValueType` (computed once, before folding)
rather than a per-pair recomputation — this was the specific correctness
condition the review flagged for the acyclicity prover's lag-cutting rule.
Test added: `ExecutionPlan.FiveWriterMergeFoldsToPairwiseMergesAndSchedules`
(five real writers through `lower()`, not hand-built — the gap the existing
`plan_test.cpp` arity test couldn't cover since it never calls `lower()`) and
`ExecutionPlan.FiveWriterMergeKeepsNewestTimeLevelThroughTheFoldWhenOneIsLagged`,
both in `test/schedule/plan_test.cpp`. Verified: both pass; hand-applying the
`merge-not-folded-past-three-writers` mutation (reverts the threshold check to
always take the single-node branch) fails both with the exact pre-fix
`E-SCHED-ARITY` diagnostic, confirming the tests actually exercise the fix.

**R-28 (IR-F3, MAJOR) — `buffers.hpp` documents the wrong exception types.**
Confirmed: the docstring promised `std::invalid_argument` /
`std::out_of_range` / `std::overflow_error`; the implementation converts
every failure through `FUSE_THROW(InvalidArgument, ...)`, and
`fuse::InvalidArgument` derives from `fuse::Error` → `std::runtime_error`,
none of the three. This is a documentation defect, not a code defect — the
code already follows `conventions.md`'s ban on a bare `throw std::...`.
Fix: rewrote the docstring to name `fuse::InvalidArgument` uniformly, with the
code always `E-SCHED-BUFFERS`, and to say the old three-way split was never
actually thrown. Test added: `Buffers.PlanBuffersThrowsFuseInvalidArgumentNotTheDocumentedStdTypes`
in `test/schedule/buffers_test.cpp`, covering all three previously-documented
failure modes (empty `RegionSizes`, a `TimedDag` naming an unknown node, an
overflowing running total) and asserting none is catchable as its
old-documented `std::` type. Note: the file's *other* throwing-spelling tests
already asserted `fuse::InvalidArgument` before this fix (they were correct;
only the header's prose was wrong), so this new test adds the explicit
negative check the review asked for rather than fixing an existing gap.
Verified: `ctest -R schedule.buffers` → 18/18 pass (17 pre-existing + 1 new).

**R-43 (IR-F4, MINOR) — merge's anchor tie-break silently keeps the first
writer.** Confirmed: the single-pass comparison only updated `newest` when
`b.anchor == a.anchor`, so two contributions tying on `iterate` but disagreeing
on `anchor` left `newest` at whichever was declared first, with no
diagnostic — unlike `acyclicity.cpp`/`scheduler.cpp`'s "keep the edge" for the
same ambiguity, which is safe there but not here (this would mistype the
merged output). Confirmed still latent: `grep -rn "Anchor::" include src`
(excluding the enum and its two `WindowEnd`-only named constructors) finds no
call site producing a non-`WindowEnd` anchor.
Fix: rewrote the tie-break as two passes — a reduction over ALL writers for
the greatest `iterate` (safe regardless of anchor), then a scan restricted to
just the writers tied at that `iterate` to check anchor agreement and pick the
highest offset within it. **This two-pass shape is required, not stylistic**:
a first draft used one combined pass that could flag ambiguity from an older
tie before reaching a later, strictly-newer writer that would have been the
unambiguous answer (caught by review before this report was finalized;
verified the corrected version against the same counterexample — writers with
`(iterate=1,A),(iterate=1,B),(iterate=2,C)` in declaration order now
correctly select `C`, not a false `E-IR-MERGE-ANCHOR`). On genuine ambiguity,
emits `E-IR-MERGE-ANCHOR` and refuses the combine rather than guessing.
Verified but not test-exercised: no public API constructs a non-`WindowEnd`
`TimeLevel` today (confirmed above), so — exactly as the review anticipated —
no test through `lower()`'s public surface can reach this branch; the
in-code comment states this explicitly (mirroring the project's own
"unreachable today, here's why" convention) as the documented alternative to
a test, per the review's own two options for this finding.

**R-44 (IR-F5, MINOR) — `E-SCHED-EPOCH-PARTIAL` structurally unreachable.**
Confirmed: the sole `OpKind::Epoch` emission site (`rules_channel.cpp:213-218`)
copies its rank sets verbatim from the `Transport` it guards, so the two can
never disagree along any lowering path that exists today. Not a code defect —
legitimate defense-in-depth per the review — but unexercised, so by
`testing.md`'s five states it was undertested relative to how "checked" it
reads. Fix: no code change (matches the review's own "(a) or (b)" options;
took (a)). Test added: `Epoch.MismatchedRankSetsAreRefused` in
`test/schedule/epoch_test.cpp`, splicing a `Transport`/`Epoch` pair with
deliberately disagreeing `dst_rank_set` the same way the file's existing
`ReservedMultiscaleOperatorsAreRefused` splices a reserved opcode. First
attempt held the shared arc type by reference across two `add_arc()` calls —
`dag.hpp`'s own doc comment warns a reference into the arc vector is
invalidated by a subsequent mutating call — and read `0 components` off the
dangling reference, producing an unrelated `E-SCHED-EXPAND` failure instead of
`E-SCHED-EPOCH-PARTIAL`; fixed by copying the type by value once, before
either `add_arc()` call. Verified: `ctest -R schedule.epoch` → 5/5 pass.

## Rejected

None. All five findings were confirmed against the cited lines before any
change was made.

## Needs another owner

- **`build-evidence/mutations.toml`'s `merge-typed-at-first-writer` entry is
  now stale** (`old = "        if (b.iterate > a.iterate ||"`, which the R-27/R-43
  rewrite deleted). I cannot edit that file (fixer constraint: "do not edit
  build-evidence/mutations.toml directly"). A replacement,
  `merge-typed-at-first-writer-v2`, is in my new
  `build-evidence/mutations/ir-schedule.toml` and is hand-verified to kill
  `schedule.acyclicity_sweep` (see "Fixed" R-27/R-43 and the mutation's own
  `describes` field, which also records that the more obvious replacement
  anchor — the `iterate`-reduction line — is a confirmed no-op against this
  test, so as not to be rediscovered). The lead should remove or retarget the
  stale entry in `mutations.toml` once this lands.
- **A second, unrelated stale anchor appeared during this session**:
  `mesh-accepts-a-parent-outside-the-cell-set` in
  `include/fuse/backend/native/mesh.hpp` (`tools/mutate.py --check-anchors`
  output: `STALE-ANCHOR ... old occurs 0 times`). This is outside my owned
  paths (backend layer) and outside my findings; it appeared between two of
  my own gate runs, consistent with a concurrent fixer editing that file
  while I worked. Flagging for the backend fixer/lead; I made no change
  there.

## Needs a host run

None. Everything in this scope (`ir`, `schedule`) builds and runs under the
`tier0` preset with no MPI dependency; all 18 `ir.*`/`schedule.*` targets ran
and passed directly in this sandbox.

## Deviation decisions

None. No DEVIATION-severity finding was assigned to this layer group (R-49
and R-50 belong to `num`/`transport`, outside my scope).
