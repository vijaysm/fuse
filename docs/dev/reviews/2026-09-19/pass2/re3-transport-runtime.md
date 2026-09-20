# Third-pass (final) review: transport and runtime (FUSE, HEAD `ced1945`)

Branch `review-2026-09-19-fixes`. Scope: `include/fuse/transport/**`, `src/transport/**`,
`include/fuse/runtime/**`, `src/runtime/**`, `test/transport`, `test/runtime`, `test/parallel`,
`build-evidence/mutations/transport-runtime.toml`. Read against `docs/dev/invariants.md`,
`layering.md`, `conventions.md`, `testing.md`, `rank-agreement.md`, `porting.md`.

Chain: first review at `bdf4346` (`docs/dev/reviews/2026-09-19/transport-runtime.md`, 7 findings
TR-F1..TR-F7, unified R-09/R-10/R-30/R-31/R-32/R-46/R-50) → first fix
(`docs/dev/reviews/2026-09-19/fixes/fix-transport-runtime.md`) → second-pass review at `dbdc872`
(`docs/dev/reviews/2026-09-19/pass2/re-transport-runtime.md`: 5 CONFIRMED-FIXED, 1 REJECTION-UPHELD,
1 PARTIAL (R-09), 6 new findings N-transport-runtime-1..6) → second fix
(`docs/dev/reviews/2026-09-19/pass2/fix2-transport-runtime.md`, starting HEAD `1565cdf`) → this pass,
at HEAD `ced1945`.

**Method.** No file under the granted repo path was edited, no git state was changed, no build
directory was created there. `cmake --preset tier0 -S <repo> -B <workspace>/build-fuse-tier0` was
configured and built (`-j4`); `ctest -j4`: **71/71 passed**. `tools/check-layering.py` and
`tools/check-invariants.py` from the repo root: **0 violations** each. `tools/mutate.py
--check-anchors` against that build: **95 valid, 0 stale**. A `tier1` (`-DFUSE_ENABLE_MPI=ON`,
MPICH 5.0 at `/opt/mpich`) configuration was also built, compile-only, to check every Tier-1 compile
claim directly. `mpiexec -n 2 <any parallel test>` was attempted and fails exactly as documented
(`HYDU_sock_create_and_listen_portstr: failed to bind to any port`) — confirmed again in this pass,
not assumed. To test mutation claims empirically rather than by inspection alone, the whole repository
(excluding `.git` and prior reviewers' `build*` directories) was copied into an isolated workspace
tree — never under the granted path — configured as a second, independent `tier0` build, and three
one-line reverts were applied there in turn, rebuilt, and rerun (reported under the relevant finding).

## Verification

| unified id | this pass's verdict | evidence |
|---|---|---|
| R-09 (TR-F1, BLOCKER) | **CONFIRMED-FIXED** (code + Tier-0 regression test + a compiled, registered, 120s-TIMEOUT MPI mutation-kill test; MPI execution itself needs a host run) | `runner.cpp:2301-2313`, `iteration.cpp:791-813,853-862`; full-suite revert-and-rerun (71/71 still pass — the Tier-0 blind spot is real and correctly disclosed, not a residual gap) |
| N-transport-runtime-1 (MAJOR) | **CONFIRMED-FIXED** | `resolve_constrain()` (`runner.cpp:472-491`) refuses `E-RT-CONSERVATION-MPI` at setup for a multi-rank `Constrain` target; new tests pass; mutation `conservation-check-skips-the-multi-rank-refusal` hand-applied and confirmed to fail the test, then reverted |
| N-transport-runtime-2 (MAJOR) | **CONFIRMED-FIXED** | New MPI mutation `region-failure-abandons-the-rest-of-the-period` recorded, correctly `requires = "mpi"`, targets a real ctest entry (`parallel.runner_transport.2`, confirmed registered with `TIMEOUT=120`); the new MPI test compiles clean at tier1 (confirmed here); execution needs a host run |
| N-transport-runtime-3 (MINOR) | **CONFIRMED-FIXED, and completed by a later commit this report should note** | `requires = "mpi"` tag removed as claimed; but the toml's designated killing test (`runtime.runner`) did **not** actually kill this mutation until commit `1d1b62e` (dated after `fix2-transport-runtime.md` was written) added `Runner.AcceptsAnAcceleratedMultiRankRegionWhenNoCommunicatorIsConfigured`. Reproduced directly: applying the mutation to my isolated tree *without* that later commit's test left `verification.iteration`/`example.dn_heat` failing (confirming the code path really is exercised) but `runtime.runner` — the specific target the toml's `test =` field names — green; with the later commit's test included, `runtime.runner` itself now fails under the mutation. `build-evidence/runtime-mutations.md` documents exactly this history (`SURVIVED` on 2026-09-20, corrected to `KILLED` in `ced1945`) |
| N-transport-runtime-4 (DEVIATION) | **CONFIRMED-FIXED** | `fix-transport-runtime.md`'s stale "Needs another owner" section is gone (commit `1c34018`), replaced with a note that it was closed by `41535ff` in the same wave |
| N-transport-runtime-5 (IMPROVEMENT) | **NOT-FIXED, but deliberately deferred, not missed** | `tools/check-layering.py:91`'s `LEGACY_MACRO` regex is unchanged at HEAD; both fixers explicitly declined to fix this as out of scope for their pass ("No fix required for this pass; flagged for whoever next touches the gate script") — recorded here so it is not lost a second time |
| N-transport-runtime-6 (DEVIATION) | **CONFIRMED-FIXED** | `fix-transport-runtime.md` corrected `8` → `7`; recount at HEAD (`awk '/halves\(/' test/parallel/mpi_transport_test.cpp`, excluding the definition and its comment) gives exactly 7 (lines 290, 315, 414, 468, 716, 735, 1221) |
| R-31 (TR-F4, reclassified MINOR) | **REJECTION-UPHELD — agree** | Zero `SOMA_ENABLE_LOWFIVE` occurrences anywhere in `include/fuse/transport/` or `src/transport/` at HEAD, re-grepped fresh |
| R-10 (TR-F2, BLOCKER) | Not re-opened (fix2 did not touch this code; out of this pass's required scope) — spot-checked anyway, holds | `iteration.cpp:191-250` (the two-precondition guard: `base_communicator.has_value()` + dynamic-accelerator refusal `E-RT-REGION-MPI`); `iteration.cpp:695-716` and `runner.cpp:1821-1897` (`reduce_region_residuals()`, a real reduce-then-broadcast over `R.participants` on the same `epoch_schedule()` shape, distinct tag `kIterationRegion`); tier1 compile of `RunnerTransport.RegionConvergesInTheSerialIterationCount` and `RunnerTransport.RefusesADynamicAcceleratorOnACrossRankRegion` clean; guard-condition tests `StillRunsAnIterationRegionWithNoCommunicatorConfigured` and (new at HEAD) `AcceptsAnAcceleratedMultiRankRegionWhenNoCommunicatorIsConfigured` both pass in the tier0 build |
| R-30, R-32, R-46, R-50 | Not re-opened (fix2 did not touch this code; out of this pass's required scope) — spot-checked anyway, all hold | `mpi_transport.cpp:479-524` (components agreement), `runner.cpp:1679` (`base` used for `MPI_TAG_UB`), `lowfive_transport.cpp:450` (`__fuse_lowfive_probe__`), `comm_graph_test.cpp`/`mpi_transport_test.cpp`/`runner_transport_test.cpp` (16/12/4 macro call sites, 0 hand-rolled) |

### R-09 / N-transport-runtime-2 in detail

The code fix is unchanged since pass2 and still correct: `step_period()`'s region-enter branch
records `step_failure` and continues the walk (`runner.cpp:2301-2313`) instead of returning; the
accelerator-divergence branch inside `run_region()`'s sweep loop does the same (`iteration.cpp:809`)
and falls through to the post-loop tail check (`iteration.cpp:853-862`) rather than returning
directly.

The task's own suggested Tier-0 test — a region whose component refuses on the first sweep, followed
by a second channel's instruction in the same period, asserting the second channel's `advance()`
still ran and the period reports the failure — **is exactly what
`Runner.ASecondChannelStillAdvancesAfterAnUnrelatedRegionFails` (`test/runtime/runner_test.cpp:1145`)
already does, and it was tested directly in this pass: it does not distinguish the fix.** Reverting
R-09's one-line change (`runner.cpp:2312`, `if (!rr.ok() && !step_failure.has_value())
step_failure = rr.error();` → `if (!rr.ok()) return rr;`) in the isolated tree and rebuilding+rerunning
the **entire** 71-test Tier-0 suite (not just this one test, to rule out a narrower escape than the
one this test's own comment names) still gives **71/71 passed**. The reason, confirmed by reading
`step_period()`'s tick-draining loop (`runner.cpp:2287-2299`): once `step_failure` is set, a
component's own `advance()` is skipped for every tick from that point on regardless of which
mechanism sets `step_failure` first — "record it, stop advancing" is unconditional, not specific to
the region path — so a second channel's tick that lands *before* the point of failure (which, per the
scheduler's own tie-breaking, is the only shape a Tier-0-constructible test can produce for a
channel not itself inside the region — see the test's own comment, `runner_test.cpp:1133-1140`) runs
identically whether R-09's fix is present or not, and one that would land *after* it is never
observed because Tier 0 has no partner rank to hang. This test is a real regression guard for a
related, weaker, and correctly-disclosed property (an unrelated channel's earlier-scheduled tick is
not retroactively undone); it is not, and does not claim to be, a mutation kill for R-09. The genuine
kill is the new MPI test `RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi`
(`test/parallel/runner_transport_test.cpp:753`), confirmed in this pass to compile cleanly at tier1,
to be registered as `parallel.runner_transport.2` (`ctest --show-only`), and to carry the 120s
`TIMEOUT` property every `fuse_add_mpi_test()` target gets. `mpiexec` cannot launch in this sandbox
(reconfirmed above), so execution itself still needs a host run.

### N-transport-runtime-3 in detail — a fix that needed one more commit to actually land

Pass2 found the `cross-rank-region-check-fires-unconditionally` mutation mismarked `requires = "mpi"`
with a wrong justification, and fix2 corrected the tag and comment. That correction was necessary but
turned out not to be sufficient: `tools/mutate.py`'s bookkeeping (`build-evidence/runtime-mutations.md`)
records that an actual `--write-log` run on 2026-09-20, against the mutation's own designated killing
test (`runtime.runner`, i.e. `fuse_runtime_runner_test`), reported **SURVIVED** — because, at that
point, nothing in `runner_test.cpp` exercised the guard's *other* precondition (a region with a
**dynamic accelerator** *and* no communicator configured; the existing
`StillRunsAnIterationRegionWithNoCommunicatorConfigured` only covers `AccelKind::None`/`Constant`).
`verification.iteration` and `example.dn_heat` — the tests pass2 actually ran to support its
"already Tier-0-killed" claim — do exercise that combination and genuinely fail under the mutation
(reproduced directly in this pass: 6 of 32 `Iteration.*` cases and the `dn_heat` example's `aitken`
configuration both fail with `E-RT-REGION-MPI` firing where it should not), but neither is the test
named in the toml's `test =` field, so the project's own automated mutation gate did not credit the
kill. Commit `1d1b62e` — dated after `fix2-transport-runtime.md` was written, and not described in
it — adds `Runner.AcceptsAnAcceleratedMultiRankRegionWhenNoCommunicatorIsConfigured`
(`runner_test.cpp:1264-1271`) to close exactly this gap, and `ced1945` updates
`build-evidence/runtime-mutations.md` from `SURVIVED` to `KILLED` (4 killed / 6 total). Reproduced
independently in this pass: rebuilding `fuse_runtime_runner_test` with the mutation applied fails
`Runner.AcceptsAnAcceleratedMultiRankRegionWhenNoCommunicatorIsConfigured` with `E-RT-REGION-MPI`
where the unmutated build accepts it. **The code at HEAD is correct and now has a mutation-kill test
that is both real and properly wired to the tool that grades it; the fix report I was asked to verify
against is one commit behind HEAD on this specific point, which is why this is recorded as
CONFIRMED-FIXED-at-HEAD rather than simply agreeing with the report as written.**

## New findings

| id | severity | file:line | title |
|---|---|---|---|
| N-transport-runtime-7 | MINOR | `test/runtime/runner_test.cpp:1263-1266` | A designated-initializer list out of member-declaration order produces a `-Wreorder-init-list` warning on every build |
| N-transport-runtime-8 | MINOR | `build-evidence/mutations/transport-runtime.toml:1-2` | The file's own header comment still lists only R-09/R-10/R-30/R-32; it was never updated to mention the two mutations fix2 added for N-transport-runtime-1/2 |

### N-transport-runtime-7 — MINOR — stray compiler warning from the latest commit's own test

**What is wrong.** `Runner.AcceptsAnAcceleratedMultiRankRegionWhenNoCommunicatorIsConfigured`
(added by commit `1d1b62e`, the commit that actually closes N-transport-runtime-3 — see above)
initializes `HarnessOpts` as:

```cpp
auto h = build(HarnessOpts{
    .dst_rank_count = 2,
    .iteration_region = true,
    .region_convergence = one_measure(1e-6, "link"),
    .region_acceleration = fuse::AccelKind::Aitken});
```

`HarnessOpts` declares `iteration_region` (line 327), `region_convergence` (332), and
`region_acceleration` (336) all *before* `dst_rank_count` (347), so listing `.dst_rank_count` first
in the initializer is out of declaration order. This is legal C++20 (designated initializers do not
require declaration order) but ISO C++ pedantically requires the *warning*, and every build of this
target emits:

```
test/runtime/runner_test.cpp:1265:27: warning: ISO C++ requires field designators to be specified in
declaration order; field 'dst_rank_count' will be initialized after field 'iteration_region'
[-Wreorder-init-list]
```

Reproduced directly in every tier0/tier1 build performed in this pass, both against the granted repo
and the isolated copy. Harmless — the values are assigned exactly as written regardless of order —
and the project does not build with `-Werror` for the C++ compile (only the Sphinx documentation
build is warnings-as-errror per `docs/dev/testing.md`), so nothing fails. But it is new noise in
every future build log, introduced by the least-reviewed commit in the branch (dated after every
review and fix report in this chain), and a trivial one-line reorder removes it.

**Fix.** Reorder the initializer to `.iteration_region = true, .region_convergence = ...,
.region_acceleration = ..., .dst_rank_count = 2` (or any order matching `HarnessOpts`'s member
declaration order).

**Test to add.** None; this is a warning, not a behavioural defect.

### N-transport-runtime-8 — MINOR — mutation file header comment is stale

`build-evidence/mutations/transport-runtime.toml`'s header comment reads: "Mutations for
REVIEW.md's transport-runtime findings (TR-*, unified R-09, R-10, R-30, R-32)." The file now also
carries `conservation-check-skips-the-multi-rank-refusal` (N-transport-runtime-1) and
`region-failure-abandons-the-rest-of-the-period` (N-transport-runtime-2), added by the second fix
pass (commit `39c40af`); the header was not updated to mention either. Not a functional defect — each
mutation's own `describes` field is accurate and complete — but `testing.md`'s own stated rationale
for this file ("checkable by grep") is undercut slightly by a header that undercounts what the file
covers.

**Fix.** Extend the header to: "...unified R-09, R-10, R-30, R-32, and N-transport-runtime-1/2
(REVIEW.md pass2)."

## Not found

Checked for and did **not** find, beyond what the first and second-pass reviews already reported:

- No regression in any of the four parallel test targets' tier1 compile
  (`fuse_parallel_runner_transport_test`, `fuse_parallel_mpi_transport_test`,
  `fuse_parallel_comm_graph_test`, `fuse_parallel_epoch_reconcile_test`): all four rebuilt from a
  clean tier1 configure in this pass with zero errors.
- No discrepancy in any of the four Tier-0-checkable gate claims at HEAD: `ctest` 71/71,
  `check-layering.py` 0 violations, `check-invariants.py` 0 violations, `mutate.py --check-anchors`
  95 valid / 0 stale (up from 86 at the second-pass review, consistent with the mutations both fix
  passes added across every layer, not just this one).
- No case where the new `E-RT-CONSERVATION-MPI` refusal (N-transport-runtime-1's fix) fires when a
  caller has explicitly set `RunnerConfig::check_conservation = false`: `resolve_constrain()` is only
  reached from a `case OpKind::Constrain` guarded by `if (!config.check_conservation ||
  ins.n_in == 0) break;` (`runner.cpp:1198`), confirmed by direct reading — the error message's own
  suggested remedy is not a false promise.
- No second, unrelated place in `runner.cpp` or `iteration.cpp` where a `step_failure`-bypassing
  `return` of the kind R-09 fixed still exists, by the same by-hand inspection the second-pass review
  performed, repeated fresh on the current file (`step_period()`, `run_region()`, and the region-body
  instruction loop all route through `step_failure` consistently at HEAD).
- Did not find that R-50's fix is complete for all four files in its original scope:
  `test/parallel/epoch_reconcile_test.cpp` still has zero `FUSE_REQUIRE_RANKS`/
  `FUSE_REQUIRE_AT_LEAST_RANKS` call sites (a `git diff bdf4346 ced1945` on this file is empty — it
  was never touched by either fix pass). This is not a new gap the fixer missed: the corrected
  `fix-transport-runtime.md` explicitly discloses it ("one deliberate `ASSERT_GE` in
  `epoch_reconcile_test.cpp`"), and it is functionally inert — `fuse_add_mpi_test()`'s default
  `RANKS 2 3 4 5` (`test/CMakeLists.txt:96`) never registers this binary below 2 ranks, so the
  `ASSERT_GE(world_size(), 2)` this file uses instead of the macro never has an opportunity to fire
  either way. Recorded here for completeness since R-50 was outside this pass's required
  verification scope, not because it changes any verdict above.
- Did not attempt to execute `parallel.mpi_transport`, `parallel.runner_transport`,
  `parallel.comm_graph`, or `parallel.epoch_reconcile` at any rank count: `mpiexec` fails identically
  to every prior pass's own note, confirmed again by direct attempt in this pass, not assumed from
  either fix report's claim.
- Did not re-derive whether every `Constrain` site in practice sits on a single rank set (which would
  have made N-transport-runtime-1's refusal purely defensive rather than load-bearing) — the fix
  chose to refuse rather than assume this, for the reason given in its own comment
  (`runner.cpp:453-471`: reducing safely would require reclassifying `Constrain` as collective in
  `fuse::is_collective()`, outside this fixer's owned paths, and doing so without that reclassification
  reopens the exact hang class R-09 fixed one instruction over) — a sound, disclosed architectural
  choice, not something this pass needed to re-litigate.

## Summary

**Verification: 8 of 8 rows in scope hold at HEAD** — R-09 and N-transport-runtime-1/2/3/4/6 are all
**CONFIRMED-FIXED** (R-09's Tier-0 test genuinely cannot distinguish the fix, as disclosed, and its
MPI mutation-kill test is real, compiled, registered, and timeout-gated pending a host run;
N-transport-runtime-3 required tracing one commit beyond the report under review to see fully
landed, but is landed at HEAD); R-31 is **REJECTION-UPHELD**, agreed; N-transport-runtime-5 is
**NOT-FIXED but deliberately, disclosedly deferred** by both fixers, not a miss. R-30/R-32/R-46/R-50,
outside this pass's required re-verification scope because fix2 did not touch their code, were
spot-checked anyway and all hold.

**New findings: 2, both MINOR, both cosmetic.** N-transport-runtime-7 is a `-Wreorder-init-list`
compiler warning in the one test added by the branch's newest, least-reviewed commit — harmless,
one-line fix. N-transport-runtime-8 is a stale header comment on the mutation evidence file that
undercounts what it now covers.

**No BLOCKER or MAJOR severity issue remains in `include/fuse/transport/**`, `src/transport/**`,
`include/fuse/runtime/**`, `src/runtime/**`, `test/transport`, `test/runtime`, `test/parallel`, or
`build-evidence/mutations/transport-runtime.toml` at HEAD `ced1945`**, on the evidence gathered in
this pass: R-09 and the two second-pass MAJORs (N-transport-runtime-1/2) were in this pass's required
re-verification scope and are CONFIRMED-FIXED there; R-10 and R-30, the other two BLOCKER/MAJOR
findings from the first review, were outside that required scope (fix2 did not touch their code) but
were spot-checked anyway in the Verification table above and both hold. All of this rests on direct
code reading, a Tier-0 rebuild-and-rerun of the full 71-test suite, hand-applied-and-reverted
mutations in an isolated tree for every Tier-0-assessable claim, and a compile-only tier1 build for
every MPI-only claim. The two
items that remain open (N-transport-runtime-5, an unhardened gate regex with no live instance today;
N-transport-runtime-7/8, two cosmetic new findings from this pass) are all MINOR/IMPROVEMENT and none
is disputed as a correctness risk by either fixer or by this review.

**Genuine residual limitation, not a deviation from the task's own instructions:** every MPI-only
mutation and test in this layer (`region-residual-skips-the-cross-rank-reduction`,
`region-failure-abandons-the-rest-of-the-period`, `build-plan-skips-the-components-agreement`, and
the four `parallel.*` ctest binaries generally) is compile-verified and registered with a timeout in
this pass but has never actually run anywhere in this review chain, because `mpiexec` cannot launch
in this sandbox at any point in the chain (first review through this one). This is the task's own
documented, out-of-scope constraint, not a new gap.
