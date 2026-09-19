<!-- Detailed layer review, 2026-09-19, HEAD bdf4346. Consolidated in ../../REVIEW.md; unified ids there are R-nn, this file's ids are IR-<id>. -->

# FUSE review: `include/fuse/ir`, `include/fuse/schedule`, `src/ir`, `src/schedule`

Reviewed at HEAD `bdf4346babe5f799e0cc27f61e7d689d1ab1ec0c`. Rules checked against:
`docs/dev/invariants.md`, `docs/dev/layering.md`, `docs/dev/conventions.md`,
`docs/dev/testing.md`, `docs/dev/porting.md`, `docs/dev/rank-agreement.md`, and the two
CI gates (`tools/check-layering.py`, `tools/check-invariants.py`), read in full before any
source file.

## 1. Coverage

**Read in full, line by line** (6,194 lines of docs/tools/library code, plus 4,960 lines of
test code):

| File | Lines |
|---|---|
| `include/fuse/ir/dag.hpp` | 163 |
| `include/fuse/ir/op.hpp` | 206 |
| `include/fuse/ir/lowering.hpp` | 299 |
| `include/fuse/ir/dump.hpp` | 145 |
| `include/fuse/schedule/acyclicity.hpp` | 161 |
| `include/fuse/schedule/buffers.hpp` | 109 |
| `include/fuse/schedule/clock.hpp` | 235 |
| `include/fuse/schedule/expand.hpp` | 162 |
| `include/fuse/schedule/plan.hpp` | 181 |
| `include/fuse/schedule/slab.hpp` | 55 |
| `src/ir/checkpoint.cpp` | 355 |
| `src/ir/lowering.cpp` | 229 |
| `src/ir/regions.cpp` | 62 |
| `src/ir/rules_channel.cpp` | 274 |
| `src/ir/rules_derived.cpp` | 81 |
| `src/schedule/acyclicity.cpp` | 947 |
| `src/schedule/buffers.cpp` | 156 |
| `src/schedule/expand.cpp` | 485 |
| `src/schedule/scheduler.cpp` | 626 |
| `src/schedule/slab.cpp` | 177 |

Plus all 17 test files under `test/ir/` and `test/schedule/` (4,960 lines: `ir_dag_test.cpp`,
`ir_dump_test.cpp`, `ir_op_test.cpp`, `lowering_channel_test.cpp`,
`lowering_derived_test.cpp`, `lowering_regions_test.cpp`, `lowering_test.cpp`,
`scale_test.cpp`, `acyclicity_sweep_test.cpp`, `acyclicity_test.cpp`, `buffers_test.cpp`,
`clock_test.cpp`, `epoch_test.cpp`, `expand_test.cpp`, `lockstep_test.cpp`, `plan_test.cpp`,
`slab_test.cpp`), and the three checked-in golden IR files (`test/golden/pair_explicit.ir`,
`pair_accumulate_remap_relax.ir`, `pair_iterated.ir`) plus their `test/data/*.yaml` sources.
Cross-referenced against `include/fuse/base/error.hpp`, `include/fuse/base/time_level.hpp`,
`include/fuse/model/merge.hpp`, `include/fuse/model/validate.hpp` (multi-writer check only),
and the corresponding upstream file this area was ported from
(`upstream/soma/test/golden_ir_test.cpp`) where a discrepancy needed the parent's history to
explain it.

**Compiled.** All ten `src/ir/*.cpp` and `src/schedule/*.cpp` files, individually, with
`c++ -std=c++20 -fsyntax-only -Iinclude -Ibuild-tier1/include -I/opt/homebrew/include
-I/opt/kokkos/include` (the `config.hpp` used was the one already generated under
`build-tier1/`, a pre-existing build directory — no build directory was created for this
review, and the project was not built or linked). All ten compiled clean. Re-run with `-Wall
-Wextra -Wpedantic -Wshadow -Wconversion`: the only warning was a pre-existing
sign-conversion at `include/fuse/model/connect.hpp:50`, outside this review's scope, seen
because `src/ir/*.cpp` transitively includes it. No warning inside the reviewed files.

**Not done:** no build of the project, no test execution (`ctest` was not run — this
sandbox's existing `build/` and `build-tier1/` directories are configure-only and were left
untouched), no MPI-dependent behaviour exercised. Every finding below is from static reading
plus the syntax-only compiles above; where a claim needed dynamic confirmation (e.g. "does
this test actually fail today"), it is stated as inferred from the source and marked as such.

## 2. Findings table

| id | severity | file:line | title |
|---|---|---|---|
| F1 | MAJOR | `test/golden/pair_explicit.ir`, `pair_accumulate_remap_relax.ir`, `pair_iterated.ir`; `src/ir/rules_channel.cpp:186-218` | Golden IR-dump regression suite was never ported; the three checked-in golden files are stale (missing `Epoch`) and unreferenced by any test |
| F2 | MAJOR | `src/ir/lowering.cpp:150-190`; `include/fuse/schedule/plan.hpp:19-21`; `src/schedule/scheduler.cpp:27-29,436-441` | "Merge as a left-fold of pairwise merges" is documented and referenced in an error message, but never implemented; 4+ writers into one merged field compile past `validate()`/`lower()` and then fail `schedule()` with a misattributed diagnostic |
| F3 | MAJOR | `include/fuse/schedule/buffers.hpp:85-93`; `src/schedule/buffers.cpp:147-152` | `buffers.hpp` documents `plan_buffers()`/`try_plan_buffers()` as throwing `std::invalid_argument` / `std::out_of_range` / `std::overflow_error`; the implementation throws `fuse::InvalidArgument`, which inherits none of those three, only `std::runtime_error` |
| F4 | MINOR | `src/ir/lowering.cpp:173-179` | Merge's "newest contribution" tie-break silently keeps the first writer when two contributions share `iterate` but differ in `anchor`, instead of erring towards the edge-keeping / conservative direction the rest of the file uses for the same case (currently unreachable: no lowering rule ever produces a non-`WindowEnd` anchor) |
| F5 | MINOR | `src/schedule/scheduler.cpp:599-618` | `E-SCHED-EPOCH-PARTIAL`'s rank-set-mismatch check cannot fire under any lowering path that exists today, so it is untested and its "verified" appearance (it reads like a checked invariant) overstates its actual status per the five-states in `docs/dev/testing.md` |

No BLOCKER was found in this scope: the SSA/single-producer invariant, the lag-cutting and
merge-typed-at-newest-contribution logic in the acyclicity prover (bar F4), the
epoch-participant structural check (bar F5), `try_plan_buffers`'s own arithmetic, and the
canonical-dump locale pinning all held up under reading and cross-checking against their
tests. See §4 for what was specifically checked and not found wanting.

## 3. Findings in full

### F1 — MAJOR — Golden IR-dump regression suite was never ported; the checked-in golden files are stale

**Where:** `test/golden/pair_explicit.ir`, `test/golden/pair_accumulate_remap_relax.ir`,
`test/golden/pair_iterated.ir` (all three, present, unreferenced); the missing test would
belong in `test/ir/`; the staleness traces to `src/ir/rules_channel.cpp:186-218` (`Epoch`
emission).

**What is wrong, and how it was confirmed.** `docs/dev/testing.md`'s "Golden files" section
and `include/fuse/ir/dump.hpp`'s own comment on `dump_text()` ("this text is checked into
`test/golden/` and compared byte-for-byte, so a lowering change cannot silently alter the
numerics") both assert that the IR dump has byte-exact regression coverage. Three files
matching that description exist — `test/golden/pair_explicit.ir`,
`pair_accumulate_remap_relax.ir`, `pair_iterated.ir` — along with matching YAML fixtures
under `test/data/` (`pair_explicit.yaml`, `pair_accumulate_remap_relax.yaml`,
`pair_iterated.yaml`).

Searching every `.cpp`, `.py`, and `.hpp` file in the repository (excluding `upstream/`) for
the string `pair_explicit`, `pair_accumulate_remap_relax`, or `pair_iterated` finds **zero**
references outside `upstream/soma/` and the fixture/golden files themselves. No test in
`test/ir/` or anywhere else loads these YAML files and diffs the result against these `.ir`
files. `test/ir/ir_dump_test.cpp`'s closest tests, `IrDump.IsByteStableAcrossCalls` and
`IrDump.TextIsIdenticalUnderAGroupingLocale`, only check `dump_text(dag) == dump_text(dag)`
(self-consistency) and locale-independence — neither compares against a frozen reference, so
neither would catch a lowering change that silently alters operator order or count.

The parent history explains the gap precisely: `git log --follow -- test/golden/pair_explicit.ir`
shows the file arriving in `9a66e9c` ("Import both frozen parents at fuse-import-0") and
surviving unchanged through `b113ca9` ("Port the compiler core: IR lowering and the
schedule"). The test that used to exercise it,
`upstream/soma/test/golden_ir_test.cpp` (`GoldenIr.PairExplicit`,
`GoldenIr.PairAccumulateRemapRelax`, `GoldenIr.PairIterated`,
`GoldenIr.OperatorOrderIsObservable`), was never copied into `test/ir/`. Per
`docs/dev/porting.md`'s own procedure, step 3 is "port the tests first... they are the oracle
for everything that follows" — for this specific fixture set, the data was ported (step 1) but
the oracle was dropped.

Worse, the checked-in `.ir` files are now **stale**, independent of whether the test is ever
ported back verbatim. All three fixtures place their two components on disjoint ranks
(`ranks: {first: 0, count: 1}` and `{first: 1, count: 1}`, confirmed by grepping
`test/data/pair_explicit.yaml` et al.), so under the current `lower_channel()`
(`src/ir/rules_channel.cpp:186-218`) every one of them takes the `ranks_differ` branch and
emits a `Transport` node immediately followed by an `Epoch` node
(`rules_channel.cpp:213-218`). All three checked-in golden dumps show the `Transport` node
(`{0->1}` is visible in each) but **none contains an `Epoch` node** — confirmed by reading
each file's full text (reproduced in the coverage section's search). The files therefore
predate the `Epoch`/rank-agreement operator; running `dump_text()` on these exact graphs
today would not reproduce them.

**Net effect:** the one regression mechanism this project has for "a lowering change cannot
silently alter the numerics" (per `dump.hpp`'s own words) currently provides **zero**
coverage for any lowered graph containing a cross-rank `Transport` — which is the majority of
non-trivial couplings, including every fixture that was imported specifically to exercise it.
This is exactly the "designed only, not verified" gap `docs/dev/testing.md`'s five states
exist to name: dump.hpp *documents* byte-for-byte coverage; nothing tests it.

**The fix.**
1. Port `upstream/soma/test/golden_ir_test.cpp` to `test/ir/golden_ir_test.cpp`, renaming
   `soma::` to `fuse::`, `SOMA_UPDATE_GOLDEN` to `FUSE_UPDATE_GOLDEN`, and the `SOMA_TEST_*_DIR`
   macros to their FUSE CMake equivalents (or reuse whatever macro the other `test/ir/*.cpp`
   files already use for `test/data/`).
2. Regenerate all three `.ir` files against the *current* `dump_text()` output (the update
   path the ported test itself provides), and review the diff by hand — the only expected
   change is the appearance of the `Epoch` node(s); if anything else changed, that is a second,
   independent finding.
3. Keep `GoldenIr.OperatorOrderIsObservable` — it is a cheap, valuable sanity check that two
   different lowerings of related graphs are not accidentally identical.

**Test to add:** the ported `golden_ir_test.cpp` itself is the test; no new one is needed
beyond regenerating the fixtures it depends on.

---

### F2 — MAJOR — "Merge as a left-fold of pairwise merges" is documented, never implemented

**Where:** `src/ir/lowering.cpp:150-190` (the actual combine step); documented at
`include/fuse/schedule/plan.hpp:19-21`; the failure surfaces via
`src/schedule/scheduler.cpp:432-441` (`E-SCHED-ARITY`).

**What is wrong, and how it was confirmed.** `include/fuse/schedule/plan.hpp`'s
`Instruction` doc states, as a fact about the compiler: *"Merge with more than three sources
is lowered as a left-fold of pairwise merges."* `src/schedule/scheduler.cpp:441` repeats the
same claim as advice inside the `E-SCHED-ARITY` diagnostic: *"lower Merge as a left-fold of
pairwise merges"*.

Reading `src/ir/lowering.cpp`'s multi-writer combine step (labelled "2b. Combine multiple
writers of one field") shows this never happens:

```cpp
const ArcId out = ctx.dag.add_arc(ctx.dag.arc(newest).type, qualified(g, f) + "/merge");
ctx.dag.add_node(OpKind::Merge, writers, out, OpAttrs{...}, ChannelId{}, ...);
```

`writers` is the *full* vector of every channel that targets the field, in declaration order,
with no cap and no split — one `Merge` node is created with as many inputs as there are
writers. `grep -rn "left-fold\|pairwise"` over `include/fuse`, `src`, and `test` (excluding the
two mentions above and one unrelated comment in `src/transport/crystal_router.hpp` about a
different, MPI reduction tree) returns nothing: there is no decomposition anywhere in the
tree, in `rules_channel.cpp`, `rules_derived.cpp`, or `checkpoint.cpp`.

Nothing upstream of this stops it. `include/fuse/model/validate.hpp`'s multi-writer check
(`E-MULTI-WRITER`, around line 259) only requires `g.merge_for(field) != nullptr` — it never
caps the writer count once a merge rule is declared, and `include/fuse/model/merge.hpp`'s own
comment says combining is meant to scale ("every channel that targets the field
participates"). `src/ir/lowering.cpp` builds the oversized `Merge` node without complaint.
Only at `schedule()` time does `src/schedule/scheduler.cpp`'s arity check
(`n.inputs.size() > kMaxInstructionInputs`, i.e. `> 3`) reject it, with the misleading message
quoted above — misleading because the fix it names ("lower Merge as a left-fold...") is not
something a *caller* can do; there is no user-facing API for it, and it names an
implementation step inside `lower()` that does not exist.

The one test that exercises `E-SCHED-ARITY`,
`test/schedule/plan_test.cpp:352` (`RejectsAnOperatorWithMoreInputsThanAnInstructionHolds`),
confirms the gap rather than closing it: it hand-builds a 4-input `Merge` node directly through
the low-level `OperatorDAG::add_node` API, bypassing `lower()` entirely
(`test/schedule/plan_test.cpp:360-370`). It proves the *scheduler's* gate fires on an
already-malformed node; it provides no evidence about what `lower()` itself produces for a
graph with 4+ real channels and a declared `MergeSpec`. `grep -n "merge_for\|g.merge(\|MergeOp::"`
across `test/ir/*.cpp` and `test/schedule/*.cpp` finds only two call sites
(`acyclicity_test.cpp:816`, `acyclicity_sweep_test.cpp:116`), and the sweep generator's random
merges are exercised only through `prove_acyclic()`, never through `schedule()` — so even the
one place multi-writer merges are fuzzed never reaches the arity gate.

**Concrete failure:** a domain-decomposed coupling where four or more sub-regions all
contribute to one shared boundary field with `graph.merge(field, MergeOp::Sum)` — an entirely
ordinary use of the documented merge feature — will pass `validate()`, pass `lower()`, and then
fail `schedule()` with `E-SCHED-ARITY`, a message that blames a "lowering" step that was never
written.

**The fix.** In `src/ir/lowering.cpp`'s combine step, when `writers.size() > kMaxInstructionInputs`
(reuse or duplicate the `3` from `plan.hpp`, ideally by moving the constant somewhere `ir` can
see it without depending on `schedule`), fold pairwise:

```cpp
ArcId acc = writers[0];
for (std::size_t k = 1; k < writers.size(); ++k) {
  const ArcId out = ctx.dag.add_arc(/* type of the running "newest" so far */, label);
  ctx.dag.add_node(OpKind::Merge, {acc, writers[k]}, out, attrs, ChannelId{}, label);
  acc = out;
}
```
taking care that the "newest contribution" TimeLevel selection (the fix documented at
`lowering.cpp:151-172`) is computed over the *original* `writers` list before folding, not
recomputed incorrectly pairwise (an intermediate fold's output type must carry the correct
running-newest level, not just the type of its two immediate inputs, or the acyclicity prover's
lag-cutting rule would see the wrong level partway through the chain).

**Test to add:** `test/ir/lowering_test.cpp` (or a new case in
`lowering_channel_test.cpp`) — a field with **five** declared writers and
`MergeOp::Sum`, asserting `lower()` succeeds and produces a chain of `Merge` nodes each with
exactly 2 inputs; then feed that DAG through `schedule()` (as `plan_test.cpp` does for its
other lowering-sourced cases) and assert `schedule()` also succeeds. A second case should
check that the merged output's `TimeLevel` still reflects the single newest contribution when
one of the five writers is lagged.

---

### F3 — MAJOR — `buffers.hpp` documents the wrong exception types for `plan_buffers()`

**Where:** `include/fuse/schedule/buffers.hpp:85-93` (the docstring); the actual behaviour is
`src/schedule/buffers.cpp:147-152`; the exception hierarchy is
`include/fuse/base/error.hpp:49,73,75`.

**What is wrong, and how it was confirmed.** `buffers.hpp`'s docstring, attached to the
`try_plan_buffers`/`plan_buffers` pair, states:

> Throws (this returns a plain value, not a `Result`...):
>   - `std::invalid_argument` if `sizes` is empty, if an arc's `components` is < 1, or if an
>     arc's region is invalid;
>   - `std::out_of_range` if a node's output arc id is out of range...;
>   - `std::overflow_error` if `elements * components`... does not fit `std::size_t`.

Reading `plan_buffers()`'s actual body (`src/schedule/buffers.cpp:147-152`):

```cpp
BufferPlan plan_buffers(const OperatorDAG& dag, const TimedDag& timed,
                        const RegionSizes& sizes) {
  auto r = try_plan_buffers(dag, timed, sizes);
  if (!r.ok()) {
    FUSE_THROW(InvalidArgument, "{}", r.error().message);
  }
  return std::move(r).value();
}
```

*every* failure `try_plan_buffers` detects — the empty-callback case, the bad-components case,
the invalid-region case, the out-of-range node case, and both overflow cases — is uniformly
converted to one `FUSE_THROW(InvalidArgument, ...)`, i.e. throws `fuse::InvalidArgument`
regardless of which of the three documented conditions actually fired. Reading
`include/fuse/base/error.hpp:49,73,75`:

```cpp
class InvariantViolation : public Error { ... };
class InvalidArgument    : public Error { ... };   // <- this one
...
class Error : public std::runtime_error { ... };
```

`fuse::InvalidArgument` derives only from `fuse::Error` → `std::runtime_error`. It does **not**
derive from `std::invalid_argument`, `std::out_of_range`, or `std::overflow_error` — those are
three separate, unrelated branches of the `std::exception` hierarchy (`std::logic_error` /
`std::logic_error` / `std::runtime_error` respectively in the standard library, none of which
`fuse::InvalidArgument` inherits). A direct caller of `plan_buffers()` who follows the
documented contract literally — e.g. `catch (const std::out_of_range&) { ... }` to handle a
malformed node id, per the second bullet — will not catch the exception `plan_buffers()`
actually throws. It propagates as an unhandled `fuse::InvalidArgument`, which is a
`std::runtime_error` and would only be caught by a handler for that base, `fuse::Error`, or
`...`.

This is not exercised through `schedule()`: `scheduler.cpp`'s call
(`const auto planned = try_plan_buffers(dag, td, sizes);`) correctly uses the
non-throwing spelling and converts the `Result` to a diagnostic, so the internal caller this
review's scope cares about most is unaffected. The risk is for any *other* direct caller of
`plan_buffers()` — the docstring explicitly calls it out as "the throwing spelling, for direct
callers that treat these as programming errors" — who is entitled to rely on the documented
`std::` exception types and will silently fail to.

**The fix.** This is a documentation defect, not a code defect: the code correctly follows
`docs/dev/conventions.md`'s "never a bare `throw std::...`; use `FUSE_THROW`" rule. Fix the
docstring to say what the code does: `plan_buffers()` throws `fuse::InvalidArgument` (a
`fuse::Error`, ultimately a `std::runtime_error`) uniformly, carrying the specific reason in
`.message()`/`.code()` (the code is always `E-SCHED-BUFFERS`, per `kBuffersCode` in
`buffers.cpp`). Drop the three-way `std::invalid_argument`/`std::out_of_range`/
`std::overflow_error` split entirely, or replace it with "would once have thrown X; now throws
`fuse::InvalidArgument` uniformly, code `E-SCHED-BUFFERS`" if the history is worth keeping.

**Test to add:** `test/schedule/buffers_test.cpp` already has one throwing-spelling test
in spirit (`RejectsAn...` tests all go through `try_plan_buffers`); add
`Buffers.PlanBuffersThrowsFuseInvalidArgumentNotAStdException` that calls `plan_buffers()`
directly on one of the existing failing fixtures and asserts
`EXPECT_THROW(plan_buffers(...), fuse::InvalidArgument)` while also asserting (via a
`try`/`catch (const std::out_of_range&)`) that the specific `std::` type the old docstring
promised is *not* what is thrown — pinning the discrepancy so it cannot silently reappear once
the doc is fixed to match one or the other.

---

### F4 — MINOR — Merge's newest-contribution tie-break doesn't apply the file's own conservative rule for mismatched anchors

**Where:** `src/ir/lowering.cpp:173-179`.

**What is wrong, and how it was confirmed.** The multi-writer combine step picks the "newest"
of several writer arcs to type the `Merge` node's output, with an extensive comment
(`lowering.cpp:151-172`) explaining why this must be the *latest* time level, not simply the
first writer's, citing the exact regression (`acyclicity_sweep_test.cpp`, seed-found) that
motivated it. The comparison:

```cpp
ArcId newest = writers.front();
for (const ArcId w : writers) {
  const TimeLevel& a = ctx.dag.arc(newest).type.time;
  const TimeLevel& b = ctx.dag.arc(w).type.time;
  if (b.iterate > a.iterate ||
      (b.iterate == a.iterate && b.anchor == a.anchor && b.offset > a.offset)) {
    newest = w;
  }
}
```

only updates `newest` when `b.anchor == a.anchor`. When two writers share `iterate` but differ
in `anchor`, the loop leaves `newest` at whichever writer it reached first (declaration order),
silently, with no diagnostic — even though `Anchor::WindowStart`/`Anchor::WindowEnd`/
`Anchor::Absolute` are, per `include/fuse/base/time_level.hpp`, not generally comparable
(offsets against different anchors have no shared origin). This is the same situation
`src/schedule/acyclicity.cpp:218-219` and `src/schedule/scheduler.cpp:52` both handle
explicitly with a comment: *"Offsets against different anchors are not ordered. Keep the
edge"* — i.e. treat the case as "cannot rule out a dependence" and err towards the safe
direction (which for those two call sites means *not* cutting a potential cycle-breaking edge).
`lowering.cpp`'s merge combine has no equivalent fallback: it just keeps the first writer,
which is a silent, arbitrary choice rather than an explicit "no test on omega"-style documented
default.

**Why this is MINOR, not MAJOR.** `grep -rn "Anchor::" include src` (excluding the enum
definition and its two named constructors, both of which use `Anchor::WindowEnd`) finds no
call site anywhere in `fuse::` code that ever constructs an `Anchor::WindowStart` or
`Anchor::Absolute` value. Every `TimeLevel` produced by `source_type()`, every channel-lowering
step, and every default-constructed `TimeLevel` uses the default `Anchor::WindowEnd`. This
specific tie-break can only diverge from the intended behaviour once a future lowering rule
starts producing non-default anchors (e.g. a window-start-referenced accumulation), so today it
is provably latent, not a live defect.

**The fix.** Add the same guard used elsewhere: if `b.anchor != a.anchor`, either keep
`newest` unconditionally (matching the "keep the edge" convention) *and* emit a diagnostic or
comment noting the ambiguity is unresolved, since — unlike the acyclicity prover, where
"keep the edge" only costs a possible false-positive cycle report — here silently keeping an
arbitrary writer's type mistypes the merged output for any downstream type check. The safer
concrete behaviour: refuse the merge with a new error code (e.g. `E-IR-MERGE-ANCHOR`) when two
contributions disagree on `anchor`, rather than choosing between them by declaration order.

**Test to add:** a unit test in `test/ir/lowering_test.cpp` is not currently constructible
without also hand-authoring a `TimeLevel` with a non-default anchor directly (since no public
API produces one) — which is itself worth doing once any future feature introduces a
non-`WindowEnd` anchor. Until then, note this in `KNOWN_ISSUES.md` so it is not rediscovered
independently when that feature lands.

---

### F5 — MINOR — `E-SCHED-EPOCH-PARTIAL`'s check is structurally unreachable today

**Where:** `src/schedule/scheduler.cpp:599-618`.

**What is wrong, and how it was confirmed.** `docs/dev/rank-agreement.md` describes
`E-SCHED-EPOCH-PARTIAL` as a live, meaningful check: *"It covers exactly the ranks the move
covers... A point covering fewer ranks than its move leaves a partner waiting... checked as
`E-SCHED-EPOCH-PARTIAL`."* The implementation compares the guarding `Epoch` instruction's
`attrs.src_rank_set`/`attrs.dst_rank_set` against the `Transport` instruction's own:

```cpp
if (guard->attrs.src_rank_set != move.attrs.src_rank_set ||
    guard->attrs.dst_rank_set != move.attrs.dst_rank_set) { ... E-SCHED-EPOCH-PARTIAL ... }
```

Tracing where `guard` and `move`'s attributes come from: the *only* place an `OpKind::Epoch`
node is ever created is `src/ir/rules_channel.cpp:213-218`, immediately after the paired
`Transport` node, and its attrs are copied verbatim from the same `Transport`'s attrs
(`ea.src_rank_set = ta.src_rank_set; ea.dst_rank_set = ta.dst_rank_set;`, lines 215-216). The
`Transport` node's output arc, by construction, has exactly one consumer — the `Epoch` node
created immediately after it (nothing else in `rules_channel.cpp` reads `cur` between those two
`emit()` calls) — so the `guard` the scheduler's search finds for a given `move` is always
*that* `Epoch`, with attrs that are, by construction, identical to `move`'s. There is no
lowering path today — no second call site emits `Transport` or `Epoch` — under which this
comparison can differ. `grep -rn "OpKind::Epoch" src include` confirms the single emission
site.

**Why this is MINOR, not a false claim to remove.** The check is legitimate defence-in-depth
against a *future* lowering rule (e.g. multiscale, or a Transport emitted from a different code
path) that might not preserve this pairing, and removing it would be a regression. The issue is
narrower: it is presented (by the doc, and by its presence in the code at all) as something the
compiler "checks" and, by the five-states in `docs/dev/testing.md`, an unexercised branch
counts as unverified, not verified, however sound its logic looks by inspection. No test in
`test/schedule/epoch_test.cpp` (`Epoch.TheAgreementPointCoversTheSameRanksAsItsMove`) constructs
a mismatch to force the `else` branch; it only confirms the matching case.

**The fix.** No code change is required. Either (a) add a regression test that hand-splices a
mismatched `Epoch` node the way `epoch_test.cpp`'s `ReservedMultiscaleOperatorsAreRefused` and
`plan_test.cpp`'s `RejectsAnOperatorWithMoreInputsThanAnInstructionHolds` already do for their
own unreachable-by-normal-means gates (so the diagnostic text and code path are at least proven
reachable and correctly worded), or (b) add a one-line comment at the check site noting it is
currently unreachable and why, matching the project's own convention of stating "today this
fires never" where that is true (see `checkpoint.cpp`'s rule-1 comments for the house style).

**Test to add:** `Epoch.MismatchedRankSetsAreRefused` in `test/schedule/epoch_test.cpp` —
lower a disjoint-pair graph, then before calling `schedule()`'s internals directly, splice a
second `Epoch` node (or mutate the existing one's `attrs.dst_rank_set`) between `Transport` and
its own guard the way `ReservedMultiscaleOperatorsAreRefused` splices a reserved opcode, and
assert `E-SCHED-EPOCH-PARTIAL` fires. (This needs access to the pre-`schedule()` DAG the way
that test already has it, so the shape is a direct precedent, not new machinery.)

## 4. Not found

Checked deliberately and found sound (no finding filed):

- **SSA / single-producer invariant** (`include/fuse/ir/dag.hpp:add_node`). Enforced
  unconditionally via `FUSE_THROW(InvariantViolation, ...)` on a second producer, not via
  `assert`, so it survives `-DNDEBUG` — matches `I6`. Bounds on `inputs`/`output` are `.at()`
  checked. `topological_order()`'s indegree computation correctly ignores an input arc with no
  valid producer (a `Source` node's empty `inputs`), so `is_acyclic()` is not fooled by a
  boundary node.
- **Lag-cutting in the acyclicity prover** (`is_loop_carried()`,
  `src/schedule/acyclicity.cpp:157-172`). Read against its own extensive comment and the two
  named regression tests it says pin the two halves of the rule
  (`Acyclicity.LeadingChannelIsStillRejectedAsACycle`,
  `Acyclicity.LagDoesNotDropEdgesInsideTheLaggedRegion`) — both exist in
  `test/schedule/acyclicity_test.cpp` (lines 631, 596) and exercise exactly the two failure
  modes the comment describes. Cross-checked the deliberate second copy of the lag half in
  `src/schedule/scheduler.cpp`'s `crosses_period()` against `is_loop_carried()`: the two
  functions differ by the `CarriedValues::cuts()` check, which `is_loop_carried()` has and
  `crosses_period()` does not — traced why this is *not* a divergence bug:
  `expand()`/`schedule()` instantiate an iteration region's body exactly once per period (never
  once per sweep — sweeps are a runtime re-execution of one fixed instruction range, confirmed
  by reading `src/schedule/expand.cpp` end to end and `mark_regions()` in `scheduler.cpp`), so
  the `Relax -> Sink` edge the iteration-region cut exists to break in the *cycle-detection*
  graph is, in the flat instruction stream `check_dependence_order()` walks, always genuinely
  written earlier by an ordinary same-pass producer (no exemption needed there in the first
  place). The two functions are correctly answering two different questions over two different
  graphs.
- **The merge-typed-at-newest-contribution fix itself**, for the reachable case (equal or
  differing `iterate`, same `anchor`): correct, and matches the regression
  (`acyclicity_sweep_test.cpp`) that motivated it. See F4 for the one unreachable-but-real gap
  in this same block.
- **Epoch-participant check's structural half** (`E-SCHED-EPOCH-MISSING`,
  `scheduler.cpp:583-605`): every `Transport` instruction is checked for a following `Epoch`
  reading its output arc; confirmed this is exercised by
  `Epoch.EveryDataMoveIsFollowedByAnAgreementPoint` and (negatively)
  `Epoch.NoMoveMeansNoAgreementPoint` in `test/schedule/epoch_test.cpp`. Only the *partial*-match
  half (F5) is unreachable.
- **`try_plan_buffers`'s own arithmetic** (`src/schedule/buffers.cpp`): the `checked_mul`/
  `checked_add` helpers correctly reject overflow before it can wrap (`std::size_t` overflow is
  defined-wrap and invisible to ASan/UBSan, which the file's own comment notes); the
  negative-`components`-cast-to-`size_t` hazard is explicitly guarded
  (`a.type.components < 1` before the multiply, not after).
- **Canonical-dump locale pinning** (`include/fuse/ir/dump.hpp`, and
  `ExecutionPlan::to_string()` in `src/schedule/scheduler.cpp`): every `std::ostringstream` in
  both files is immediately `imbue`d with `std::locale::classic()` before any output is
  written to it — three streams in `dump.hpp` (`format_double`, `dump_text`, `dump_graphviz`),
  one in `scheduler.cpp` (`ExecutionPlan::to_string`), all pinned, matching
  `tools/check-invariants.py`'s check (`pinned == streams`) exactly. `IrDump.TextIsIdenticalUnderAGroupingLocale`
  and `ExecutionPlan.DumpIsIndependentOfTheGlobalLocale` both exist and test it.
- **Plan interpreter contract completeness** (`include/fuse/schedule/plan.hpp`'s
  `Instruction`): checked every `OpKind` against the fields `Instruction` carries — every
  emittable kind's required attribute (`Scale`'s `scale`, `Epoch`'s `reason` +
  `src_rank_set`/`dst_rank_set`, `Relax`'s `omega`, `Extrapolate`'s `order`, `Accumulate`'s
  `accum`, `Remap`'s `remap_flags`/`method`, `Merge`'s `merge`) is present in the flat
  `OpAttrs` bag `Instruction::attrs` carries verbatim from the `Node`. The three reserved
  multiscale kinds (`Map`/`Reduce`/`Closure`) are explicitly refused by `schedule()` before an
  `Instruction` would ever need to represent them (`scheduler.cpp:397-406`), so the contract is
  not silently incomplete for them either.
- **Buffer lifetime (`is_persistent`)**: matches its documented scope exactly — it flags
  cross-*period* carriers (`Waveform` values and any negative `TimeLevel::offset`, i.e. a
  `lag()`-reached arc) and explicitly excludes cross-*sweep* (iteration-region) carriers, which
  the header says is `IterationRegion::carried`'s job, not `BufferPlan`'s. There is currently no
  buffer-reuse/aliasing pass anywhere in `try_plan_buffers` (it is a strict one-buffer-per-arc
  plan), so `is_persistent` is presently pure metadata with no consumer that could act on an
  incomplete flag — confirmed by reading the whole function; nothing branches on
  `is_persistent` to decide allocation.
- **Expand/slab arithmetic** (`src/schedule/expand.cpp`, `src/schedule/slab.cpp`): every
  multiplication and division that could overflow a 64-bit `Rational` is explicitly range-checked
  in `__int128` first (`scaled_tick`, `tick_count`, `ticks_of`, `floor_div`/`ceil_div`), consistent
  with the documented "never exceptions" contract; `TickPhase`'s causality contract
  (`BeforeTicks`/`AfterTicks`) is applied consistently between the two emission conventions
  (window-closing vs. instant chains) exactly as `expand.hpp`'s long comment specifies.

## 5. Suggested new tests / mutations

1. **F1's fix is itself the test** — port `upstream/soma/test/golden_ir_test.cpp`, regenerate
   the three `.ir` fixtures, and keep `GoldenIr.OperatorOrderIsObservable`.
2. **F2** — `Lowering.MergeOfFiveWritersFoldsToTernaryOfPairwiseMerges` (or similar): five
   channels into one field with a declared `MergeOp`, asserting `lower()` succeeds, the result
   contains a chain of `Merge` nodes each with exactly 2 inputs, and `schedule()` on the result
   also succeeds (currently would fail `E-SCHED-ARITY` before the fix, succeed after).
   A mutation to pair with it once the fix lands: revert the fold to a single N-ary `Merge`
   node and confirm the new test (not just `plan_test.cpp`'s hand-built one) catches it — this
   is the mutation `plan_test.cpp`'s existing `RejectsAnOperatorWithMoreInputsThanAnInstructionHolds`
   cannot catch, since it never calls `lower()`.
3. **F3** — `Buffers.PlanBuffersThrowsFuseInvalidArgumentNotAStdException`, pinning the actual
   exception type so a future change to either the code or the docstring can't silently
   re-diverge them.
4. **F4** — deferred until a lowering rule produces a non-`WindowEnd` anchor; note in
   `KNOWN_ISSUES.md` in the meantime so it surfaces when that feature is designed, not
   independently rediscovered.
5. **F5** — `Epoch.MismatchedRankSetsAreRefused`, hand-splicing a rank-set mismatch between a
   `Transport` and its guarding `Epoch` the same way `ReservedMultiscaleOperatorsAreRefused`
   already splices a reserved opcode.
6. **General mutation suggestion for this area, independent of the findings above**: the
   acyclicity fuzz sweep (`acyclicity_sweep_test.cpp`) explicitly excludes multirate graphs,
   iteration regions, and space-time windows from its generator (stated in its own file header).
   Those three are covered only by hand-picked named cases in `acyclicity_test.cpp`. A second,
   narrower sweep restricted to *lockstep, single-region* graphs with a randomly placed
   `.relax()` and a randomly chosen carried/not-carried channel would give the iteration-region
   cut (`CarriedValues`) the same kind of independent-oracle coverage the base sweep gives the
   lag rule, closing the one class of case the existing sweep's own docstring says it does not
   attempt.

## Summary

**0 BLOCKER, 3 MAJOR (F1, F2, F3), 2 MINOR (F4, F5), 0 IMPROVEMENT, 0 DEVIATION** filed for
`include/fuse/ir`, `include/fuse/schedule`, `src/ir`, `src/schedule`, and their tests at HEAD
`bdf4346`. The three MAJOR findings are two contract mismatches between documentation and
implementation (F2: a promised IR transformation that was never written, surfaced only by a
misleading downstream error message; F3: a wrong `throws` clause for a public function) and one
coverage gap in the project's own byte-exact regression mechanism for the IR dump (F1), where
the checked-in fixtures have quietly gone stale since the `Epoch` operator landed. No wrong
results, hangs, or memory-safety defects were found in the SSA invariant, the lag-cutting or
newest-contribution logic of the acyclicity prover, buffer arithmetic, or locale-pinned dumps;
the two MINOR findings (F4, F5) are both currently-unreachable gaps in judgement-call
conservatism that are worth a guard or a test before the feature that would make them reachable
lands, not before.
