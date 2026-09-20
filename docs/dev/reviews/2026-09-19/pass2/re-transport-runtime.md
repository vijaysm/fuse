# Second-pass review: transport and runtime (FUSE, HEAD dbdc872)

Branch `review-2026-09-19-fixes`. Scope: `include/fuse/transport/**`, `src/transport/**`,
`include/fuse/runtime/**`, `src/runtime/**`, `test/transport`, `test/runtime`, `test/parallel`.
Read against `docs/dev/invariants.md`, `layering.md`, `conventions.md`, `testing.md`,
`rank-agreement.md`, `porting.md`. First review at `bdf4346`:
`docs/dev/reviews/2026-09-19/transport-runtime.md` (7 findings, TR-F1..TR-F7, unified R-09, R-10,
R-30, R-31, R-32, R-46, R-50). Fixer report: `docs/dev/reviews/2026-09-19/fixes/fix-transport-runtime.md`.

**Method.** No file under the granted repo path was edited; no git state was changed; no build
directory was created there. `cmake --preset tier0 -S <repo> -B <workspace>/build-fuse-tier0` was
configured and built (`-j4`) and the full suite run with `ctest -j4`: **71/71 passed**, matching the
fixer's own claim. `tools/check-layering.py` and `tools/check-invariants.py` were run read-only from
the repo root: **0 violations** each, matching the fixer's claim. `tools/mutate.py --check-anchors`
against that build: **86 valid, 0 stale**, matching the fixer's claim. A `tier1` configuration
(`-DFUSE_ENABLE_MPI=ON`) was also built (compile-only) in the workspace against the same, unmodified
repo, to check every Tier-1 compile claim in the fixer's "Gate results" section directly rather than
taking it on faith. `mpiexec -n 2 <any test>` was attempted and fails exactly as documented
(`HYDU_sock_create_and_listen_portstr: failed to bind to any port`) — MPI genuinely cannot launch
here, confirming the task's stated constraint; no parallel test was executed, only compiled. To test
two specific claims empirically rather than by inspection alone, the transport/runtime/test sources
(excluding `upstream/`, `.git/`, and pre-existing `build*` directories already present under the
granted path from other reviewers) were copied into an isolated workspace tree, configured and built
as a second, independent `tier0` tree, and two one-line reverts were applied there in turn (never
under the granted path) to check whether the existing suite's outcome actually depends on the lines
the fixer's narrative says it depends on. Both experiments are reported under the relevant finding
below.

**Deviations.** `test/parallel/comm_graph_test.cpp` (1540 lines) and
`test/parallel/runner_transport_test.cpp` (933 lines) were not re-read end-to-end a second time;
new-finding effort was directed by following the evidence trails the verification pass turned up
(the mutation-evidence gap, the stale "needs another owner" claim) rather than a uniform second
line-by-line pass over every file, given the combined scope is already ~14,000 lines of source and
test. Files outside the assigned scope (`src/backend`, `src/num`, `src/model`, `src/ir`, `cmake/`,
`tools/`) were read only as far as needed to check a specific claim (e.g. `tools/check-layering.py`'s
`LEGACY_MACRO` regex, `weights_for()`'s ghost-zeroing). No MPI test was executed (see Method); every
MPI-only claim below is reported as compile-verified or code-inspected, never as run.

## Verification

| unified id | this file's id | verdict | evidence |
|---|---|---|---|
| R-09 | TR-F1 (BLOCKER) | **PARTIAL** | Code fix in `step_period()` and `run_region()` is present and structurally correct (see below), but no test — Tier 0 or otherwise — distinguishes the fixed behaviour from the pre-fix bug, and no mutation was recorded for it. See N-transport-runtime-2. |
| R-10 | TR-F2 (BLOCKER) | **CONFIRMED-FIXED** | Cross-rank residual reduction genuinely implemented (not a blanket refusal); narrowed refusal for dynamic accelerators is correct and independently Tier-0-verified in this pass. MPI-only parts compile-verified only. See below and N-transport-runtime-3. |
| R-30 | TR-F3 (MAJOR) | **CONFIRMED-FIXED** | `build_plan()`'s Collective 2b correctly agrees `components`; new test compiles cleanly under tier1. MPI execution not verifiable here (documented constraint). |
| R-31 | TR-F4 (MAJOR) | **REJECTION-UPHELD** — agree | Grepped `include/fuse/transport/`, `src/transport/`: zero `SOMA_ENABLE_LOWFIVE` occurrences. Confirmed by git ancestry that the fix (commit `4a2ad6e`) predates this fixer's own R-09/R-10/R-30 commits in the branch's linear history, i.e. it really was already fixed before this session touched the file, not a fabricated excuse. |
| R-32 | TR-F5 (MAJOR) | **CONFIRMED-FIXED** | `resolve_epoch_participants()` now converts and queries `base_communicator`, the same communicator `reconcile_epoch()` uses. Verified by reading both functions side by side; all Tier1 targets that link `fuse-coupler-runtime` build cleanly here. |
| R-46 | TR-F6 (MINOR) | **CONFIRMED-FIXED** | `src/transport/lowfive_transport.cpp:450` reads `"__fuse_lowfive_probe__"`. |
| R-50 | TR-F7 (DEVIATION) | **CONFIRMED-FIXED** (substantially) | `comm_graph_test.cpp` now has 16 `FUSE_REQUIRE_AT_LEAST_RANKS` call sites and 0 hand-rolled skips; `mpi_transport_test.cpp` and `runner_transport_test.cpp` likewise use the macros throughout. One bookkeeping inaccuracy in the fixer's own count: see N-transport-runtime-6. |

### R-09 in detail

`src/runtime/runner.cpp`'s `step_period()` now sets `step_failure` from a failed region and
continues the walk (`i = r.exit + 1; continue;`) instead of `return rr;` — read directly at
`runner.cpp:2252-2265`, matching the ordinary-instruction failure path two branches below it exactly
as `rank-agreement.md` requires. `src/runtime/iteration.cpp`'s accelerator-divergence branch
(`run_region()`, ~line 796-808) likewise now records `step_failure` and `break`s rather than
returning directly, falling through to the post-loop tail check. That tail check itself
(`if (step_failure.has_value())`, ~line 751) is exactly the guard the fix report's "Correction during
this revision" note says was found live-disabled (`if (false && ...)`) and restored; the disabled
form leaves no trace in git history (as expected — it was never committed), so this is unverifiable
by `git log`, but the code at HEAD is correct.

**What this pass adds.** The existing regression test for this area,
`Runner.AnAdvanceFailureInsideAnIterationRegionSweepIsReported`, was read against both fixed lines.
It only asserts that `run_period()` returns a non-ok `Result` with the expected error code — a
condition equally satisfied by the OLD, buggy `return rr;`/`return a.error();` and by the NEW
carry-forward code, because in both cases `run_region()` ultimately propagates the same failure to
its one caller. To check this directly rather than argue it, the fix to `step_period()`'s
region-enter branch was reverted in the isolated copy (`if (!rr.ok()) return rr;` restored) and the
full Tier 0 suite rebuilt and rerun: **all 71 tests, including all 43 in `runtime.runner`, still
pass** with the revert in place. This is reported as new finding N-transport-runtime-2, not folded
into the verdict above, because the code itself is correct — only its test/mutation coverage is
missing.

### R-10 in detail

`resolve_regions()` derives `R.participants` (the sorted, de-duplicated union of the region's
checkpointed components' ranks) only when `config.base_communicator.has_value()`
(`iteration.cpp:211`), and refuses only the one case its own reduction cannot cover — a dynamic
accelerator (`decl.acceleration.kind` outside `{None, Constant}`) — with `E-RT-REGION-MPI`
(`iteration.cpp:~236`). `run_region()`'s sweep loop reduces each block's residual/reference/status
across `R.participants` every sweep via `reduce_region_residuals()` (`runner.cpp:1772`), reusing
`epoch_schedule()`'s binomial reduce-then-broadcast and a distinct tag (`tags::kIterationRegion`,
correctly distinct from `tags::kEpoch` since a region body can itself contain an `Epoch`). The
undefined-vs-zero convention (`owned == 0` excluded rather than read as a converged zero) is
correctly threaded through `merge_region_payload()`. This is a real fix, not a renamed refusal.

The guard-condition test `Runner.StillRunsAnIterationRegionWithNoCommunicatorConfigured`
(Tier 0) passes, confirming the fix's own precondition (no communicator ⇒ unchanged, single-process
behaviour) did not regress. New MPI tests `RunnerTransport.RefusesADynamicAcceleratorOnACrossRankRegion`
and `RunnerTransport.RegionConvergesInTheSerialIterationCount` are registered and — verified directly
in this pass — build cleanly under a `tier1` (compile-only) configuration, and inherit the 120 s
`TIMEOUT` property every `fuse_add_mpi_test()` target carries (`test/CMakeLists.txt:97`), satisfying
the task's "carries a timeout" check. Neither could be executed (no MPI here).

## New findings

| id | severity | file:line | title |
|---|---|---|---|
| N-transport-runtime-1 | MAJOR | `src/runtime/runner.cpp:2106-2137`, `src/runtime/runner.cpp:~400-404` | `Constrain`'s conservation diagnostic compares purely rank-local partial integrals with no cross-rank reduction — the same rank-local-decision defect R-10 fixed elsewhere, left open |
| N-transport-runtime-2 | MAJOR | `src/runtime/runner.cpp:2252-2265`; `build-evidence/mutations/transport-runtime.toml` | R-09's fix has no recorded mutation and no test (confirmed by reverting it) distinguishes it from the pre-fix bug |
| N-transport-runtime-3 | MINOR | `build-evidence/mutations/transport-runtime.toml` (`cross-rank-region-check-fires-unconditionally`) | This mutation is mismarked `requires = "mpi"`; it is actually Tier-0-assessable and already killed by two pre-existing Tier 0 targets |
| N-transport-runtime-4 | DEVIATION | `docs/dev/reviews/2026-09-19/fixes/fix-transport-runtime.md`, "Needs another owner" | Stale: the compile blocker it describes was already fixed by this fixer's own commit in the same session |
| N-transport-runtime-5 | IMPROVEMENT | `tools/check-layering.py:91` | F4's suggested `-D<MACRO>` hardening of `LEGACY_MACRO` was not implemented; the blind spot that let `SOMA_ENABLE_LOWFIVE` through once is still live for the next such name |
| N-transport-runtime-6 | DEVIATION | `docs/dev/reviews/2026-09-19/fixes/fix-transport-runtime.md`, "Fixed: R-50" | Off-by-one in the fixer's own count: it claims 8 `halves()`-based hand-rolled skips remain in `mpi_transport_test.cpp`; there are 7 |

### N-transport-runtime-1 — MAJOR — `src/runtime/runner.cpp:2106-2137` — the conservation check has no cross-rank reduction

**What is wrong.** `OpKind::Constrain`'s runtime diagnostic computes, per period, per rank:

```cpp
const double before = num::weighted_sum(src.data, wb.data(), wb.size());
const double after  = num::weighted_sum(out, wa.data(), wa.size());
const double rel = std::abs(after - before) / scale;
if (rel > config.conservation_tolerance) { ++stats.conservation_violations; ... }
```

with no `MPI_Allreduce`/`MPI_Allgather` anywhere on this path (grepped the whole of
`src/runtime/runner.cpp` for `MPI_` near `conservation`: no hits other than the counters
themselves). The weight vectors `wb`/`wa` that make `before`/`after` are built by `weights_for()`
(`runner.cpp:~400-404`) with every ghost entity's weight zeroed (`mv->is_owned(dim, i) ? measure : 0.0`,
with the comment "Including one would double count against another rank's copy"). That is exactly
the construction that makes `before`/`after` each rank's **share** of the true global integral, not
a self-contained quantity a single rank can judge in isolation.

For a remap whose source and destination regions sit on the *same* rank set with a diagonal (rank-
local) mapping, this is harmless: each rank's local integral is conserved on its own. But that is not
the general case this layer exists for — `CommGraph`/`MpiTransport` exist precisely because a
conservative remap can and does move mass between ranks (a fine-to-coarse interpolation whose source
cell's contribution lands on a different rank's destination entity). In that configuration, a rank's
local `before` (its owned share of the pre-remap field) and local `after` (its owned share of the
post-remap field) are not expected to agree even for a *correct* remap — the mass that left is
accounted for on the receiving rank, not here. This can go wrong in both directions: a correct,
globally-conserving remap can trip a **false** `E-RT-CONSERVATION`/violation count on a rank that
sent more than it received locally, and two ranks' local errors of opposite sign can cancel in the
true global sum while neither rank's local check catches a real defect that only shows up in the
global integral.

**How this was confirmed.** Read `weights_for()`'s ghost-zeroing (`runner.cpp:~396-404`) and the
`Constrain` case (`runner.cpp:2106-2137`) together; grepped `src/runtime/runner.cpp` for `MPI_` in
the vicinity of `conservation`/`Constrain`/`weighted_sum`: none. This is the same question the first
review's "Not found" section raised and explicitly declined to settle ("this is a real open question
this review did not have time to settle and is called out provisionally here"); the fix wave's own
commit-units list does not touch this code (only `R-09`, `R-10`, `R-30`, `R-46`, `R-50` were in
scope), so it remains exactly as it was at `bdf4346`.

**What would clear this.** If every `Constrain`-guarded channel in practice only ever connects
components on identical or otherwise rank-local-only placements (so a remap never actually moves mass
across a rank boundary for a checked channel), the local check is sound and this is a documentation
gap rather than a correctness one — that placement invariant was not checked in this pass and would
need its own investigation (e.g. whether `resolve_constrain()` or an earlier lowering pass already
restricts `Remap` to same-rank-set pairs, which was not confirmed here).

**Fix.** Either (a) reduce `before` and `after` with an `MPI_Allreduce`(sum) across the ranks that
participate in the remap before computing `rel`, mirroring `reduce_region_residuals()`'s pattern, or
(b) if same-rank-set placement is already a precondition of every `Constrain` site, add a check for
that precondition and a comment citing where it is enforced, so a future reader does not have to
re-derive it from `weights_for()`'s ghost-zeroing the way this review did.

**Test to add.** A parallel test with source and destination regions on disjoint or partially
overlapping rank sets and a remap that legitimately moves mass across the boundary; assert that
`stats.worst_conservation_error` reflects the *global* relative error, not a per-rank artifact, and
that a correct cross-rank remap does not trip `conservation_violations`.

### N-transport-runtime-2 — MAJOR — R-09's fix has no mutation evidence and no distinguishing test

**What is wrong.** Per `testing.md`'s "Mutation evidence" rule, a fix or port is accepted only when
at least one **named** mutation is killed by a test; "green tests that cannot detect the defect they
were written for are decoration." The original review's own suggested mutation #5 was exactly this
line: flip `step_period()`'s region-failure handling back to `return rr;` and confirm a test changes
from hang/timeout to a clean multi-rank failure. `build-evidence/mutations/transport-runtime.toml`
records mutations for R-10 (twice) and R-30, but **none for R-09**, despite the file's own header
comment listing R-09 among the findings it covers.

**How this was confirmed.** In an isolated copy of the source tree (never under the granted path;
see Method), `step_period()`'s fix was reverted to `if (!rr.ok()) return rr;` and the full Tier 0
suite was rebuilt and rerun: **71/71 tests still pass**, including all 43 in `runtime.runner`,
specifically including `Runner.AnAdvanceFailureInsideAnIterationRegionSweepIsReported`, the test the
fix report itself points to as covering "the tick-re-advance-failure half." That test only checks
that `run_period()`'s returned `Result` is non-ok with the right error code — a condition both the
buggy and the fixed code satisfy identically, since `run_region()` propagates the same `Failure`
either way; the difference the fix makes (whether execution continues to reach a *later, unrelated*
instruction in the same period) has no side effect this test observes. No test in
`test/runtime/runner_test.cpp`'s harness (`HarnessOpts`, `build()`) constructs a scenario with an
unrelated channel scheduled after the region in the same period, so this is not merely unasserted —
it is, on the evidence gathered, unconstructable with the current single-process test harness at all;
observing the actual defect (a partner rank hanging at a later `Epoch`) needs MPI, which is
unavailable here.

**Fix.** Record a mutation entry for this line in `build-evidence/mutations/transport-runtime.toml`,
marked `requires = "mpi"` pending a host run, alongside a genuine MPI test (the original review's
suggested test #1: an iteration region plus an unrelated, later-scheduled channel on the same period,
with the region's component failing on one rank; assert every rank returns rather than hangs). Until
that lands, this fix carries the same weight as an unexercised refusal per `testing.md`'s own
"Deliberate refusal... counts as verified only if a test makes it fire" standard — code that looks
right, not code that has been shown to matter.

### N-transport-runtime-3 — MINOR — a Tier-0-assessable mutation is mismarked `requires = "mpi"`

**What is wrong.** `build-evidence/mutations/transport-runtime.toml`'s
`cross-rank-region-check-fires-unconditionally` mutation (mutating
`iteration.cpp`'s `if (config.base_communicator.has_value())` to `if (true)`) carries
`requires = "mpi"` and the comment: "The guarded block is compiled only when FUSE_HAVE_MPI, so in
tier 0 this mutant is byte-identical to the original and its survival there says nothing." Both
halves of that justification are wrong: the guarded block has no `#if FUSE_HAVE_MPI` around it at
all (grepped the whole file: zero `#if`/`#ifdef` in `iteration.cpp`), and `R.participants` is derived
from `graph.component(c).ranks` — a purely **declared** placement, independent of whether a real
communicator or MPI build is present. `test/verification/iteration_test.cpp` and
`examples/dn_heat/main.cpp` both declare an iteration region's components on different nominal
`RankSet`s while using `AccelKind::Aitken`, with no `base_communicator` configured — exactly the
configuration the mutation's own "describes" field says it would wrongly refuse.

**How this was confirmed.** In the same isolated copy used for N-transport-runtime-2, this exact
mutation (`if (true)` in place of the guard) was applied and the full Tier 0 suite rebuilt and rerun:
**2 of 71 tests fail** — `verification.iteration` and `example.dn_heat` — both with
`E-RT-REGION-MPI` fired where the unmutated build does not fire it. The mutation is therefore already
killed at Tier 0, by tests that predate this fix wave, contrary to the toml's `requires = "mpi"` tag
and its stated reasoning. The fixer's own prose elsewhere ("A Tier 0 mutation... was hand-applied and
confirmed to kill the guard-condition tests, then reverted") is consistent with what was verified
here — the inconsistency is specifically in the toml's machine-readable metadata and its comment, not
in the fixer's understanding.

**Fix.** Drop `requires = "mpi"` from this entry (or split it: keep an MPI-only variant for the
genuinely cross-rank case, and add a Tier-0 entry with `target = "fuse_verification_iteration_test"`,
`test = "verification.iteration"`) and correct the comment. As recorded, a future run of
`tools/mutate.py` would report this mutation `NOT-ASSESSED` for lack of MPI, when it is in fact
already assessed and killed — understating this port's real mutation coverage.

### N-transport-runtime-4 — DEVIATION — the fix report's "Needs another owner" section is stale

**What is wrong.** `fix-transport-runtime.md`'s "Needs another owner" section states:
"`test/parallel/comm_graph_test.cpp` still fails to compile under `tier1` for a pre-existing,
unrelated reason (`transport/crystal_router.hpp` file not found...). The lead should route this to
whoever owns `test/CMakeLists.txt`'s `fuse_add_mpi_test()` macro." This is no longer true: the same
fix session's own commit `41535ff` ("Make comm_graph_test compile and use the rank-requirement
macros") adds `target_include_directories(fuse_parallel_comm_graph_test PRIVATE
${CMAKE_SOURCE_DIR}/src)` to `test/CMakeLists.txt`, which is precisely what closes the
`crystal_router.hpp`-not-found gap the "Needs another owner" section describes as unresolved.

**How this was confirmed.** Configured and built a `tier1` (compile-only) tree against the
unmodified repo at HEAD in this pass: `fuse_parallel_comm_graph_test` **builds and links cleanly**,
with zero errors. `git show 41535ff --stat` confirms this commit touches both
`test/parallel/comm_graph_test.cpp` and `test/CMakeLists.txt` together, and its own commit message
says "It gains the private transport include path... invisible while the file compiled to nothing."
The report's commit-units JSON at the top of the file lists only `comm_graph_test.cpp` (not
`test/CMakeLists.txt`) under the R-50 unit, which is likely why "Needs another owner" was not updated
to reflect the include-path fix landing in the same session.

**Fix.** Remove or correct the "Needs another owner" section; nothing needs routing for this item.

### N-transport-runtime-5 — IMPROVEMENT — the `-D<MACRO>` blind spot in `LEGACY_MACRO` is still live

TR-F4's fix corrected the two live `SOMA_ENABLE_LOWFIVE` strings (in an earlier commit, see R-31
above) but the suggested hardening of `tools/check-layering.py`'s `LEGACY_MACRO` regex
(`r'\b(KARMA|SOMA)_[A-Z_]+\b'`, which does not match a name written as `-DSOMA_ENABLE_X=ON` because
`\b` requires a non-word character before `S` and `D` is one) was not implemented — verified by
reading `tools/check-layering.py:91` at HEAD: the regex is unchanged. This is not a live defect (both
known instances are fixed), but the mechanism that let them through once is unchanged, so a future
`-D<KARMA|SOMA>_*` string would again pass the gate silently. No fix required for this pass; flagged
for whoever next touches the gate script.

### N-transport-runtime-6 — DEVIATION — a bookkeeping inaccuracy in the fixer's own R-50 count

`fix-transport-runtime.md`'s "Fixed: R-50" section states "8 `halves()`-based divisibility skips" were
deliberately left hand-rolled in `mpi_transport_test.cpp`. Counted directly
(`awk '/halves\(/' test/parallel/mpi_transport_test.cpp`, excluding the function definition and its
one comment reference): **7** call sites (lines 290, 315, 414, 468, 716, 735, 1221). Not a functional
defect — every one of the 7 is a legitimate, documented exception the macros cannot express — just an
off-by-one in the report's own accounting.

## Not found

Checked for and did **not** find (beyond what the first review already reported):

- No regression in any of the six other Tier 1 (compile-only) targets the fixer's "Gate results"
  claims build cleanly: `fuse_runtime_runner_test`, `fuse_transport_transport_test`,
  `fuse_parallel_mpi_transport_test`, `fuse_parallel_runner_transport_test`,
  `fuse_parallel_epoch_reconcile_test` all rebuilt without error in this pass (`fuse_model_descriptor_mpi_test`
  belongs to the base-model layer's own review and was not re-checked here).
- No discrepancy in the two other Tier-0-checkable gate claims: `check-layering.py` and
  `check-invariants.py` both report 0 violations here, matching the fix report; `mutate.py
  --check-anchors` reports 86 valid / 0 stale here, matching the fix report; `ctest` reports 71/71
  passed here, matching the fix report.
- No leftover `SOMA_ENABLE_LOWFIVE` or `KARMA`/`SOMA`-prefixed identifier anywhere in
  `include/fuse/transport/` or `src/transport/` (grepped fresh, independent of the F4 fix).
- Did not find a second instance of the R-09 pattern (a `return` that bypasses `step_failure`)
  anywhere else in `runner.cpp`'s or `iteration.cpp`'s instruction-walking code by inspection; every
  other early return in both files that is reachable after a partial period either occurs before any
  collective has run (safe) or already goes through `step_failure`.
- Did not attempt to execute `parallel.mpi_transport`, `parallel.runner_transport`, or
  `parallel.epoch_reconcile` at any rank count: `mpiexec` fails immediately in this sandbox exactly as
  `testing.md` documents (`HYDU_sock_create_and_listen_portstr`), confirmed by direct attempt rather
  than assumed from the fix report's own claim.
- Did not re-derive whether `Constrain` sites are in practice always same-rank-set (which would clear
  N-transport-runtime-1 as a documentation gap rather than a live risk) — flagged as the open
  condition under that finding rather than asserted either way, consistent with how the first review
  handled unresolved questions.

## Summary

**Verification:** 5 of 7 unified findings CONFIRMED-FIXED (R-10, R-30, R-32, R-46, R-50 — the last
with one minor bookkeeping correction), 1 REJECTION-UPHELD (R-31 — genuinely already fixed by an
earlier, unrelated commit, agree with the fixer), 1 PARTIAL (R-09 — the code fix is correct by
inspection, but it has no mutation evidence and no test, Tier 0 or MPI, currently distinguishes it
from the bug it fixes; confirmed empirically by reverting it and observing zero test failures).

**New findings: 6 — 2 MAJOR, 2 DEVIATION, 1 MINOR, 1 IMPROVEMENT.** The two MAJORs are: a
conservation diagnostic (`Constrain`) that decides per rank with no cross-rank reduction, the same
class of defect R-10 fixed for iteration-region convergence, left open by this fix wave and only
provisionally flagged (not asserted) by the first review (N-1); and R-09's fix itself, which is
correctly implemented but currently unverified in a way that matters — no mutation recorded, and the
one regression test naming it cannot distinguish fixed from broken, confirmed by directly reverting
the fix and rerunning the full suite with zero failures (N-2). The MINOR and one DEVIATION are
mutation-evidence and reporting-accuracy corrections inside the fix report and its mutation TOML,
each confirmed by rebuilding and rerunning the suite with the relevant line changed (N-3, N-6); the
other DEVIATION is a stale "needs another owner" claim the fixer's own commit had already resolved,
confirmed by a clean `tier1` compile of the target in question (N-4). The IMPROVEMENT is an
unhardened gate blind spot with no live instance today (N-5).
