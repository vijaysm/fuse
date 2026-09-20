# Second-pass fix report: transport and runtime

Branch `review-2026-09-19-fixes`, HEAD `1565cdf` at start. Answers
`docs/dev/reviews/2026-09-19/pass2/re-transport-runtime.md`: R-09's PARTIAL
row, and N-transport-runtime-1 through 6.

## Commit units

```json
[
  {
    "ids": ["N-transport-runtime-1"],
    "files": ["src/runtime/runner.cpp", "test/runtime/runner_test.cpp",
              "build-evidence/mutations/transport-runtime.toml"],
    "message": "Refuse a conservation check on a multi-rank Remap target\n\nresolve_constrain() summed only this rank's owned share of the integral with no cross-rank reduction; refuse it at setup instead of reducing it, since Constrain is deliberately not collective."
  },
  {
    "ids": ["R-09", "N-transport-runtime-2"],
    "files": ["test/runtime/runner_test.cpp", "test/parallel/runner_transport_test.cpp",
              "build-evidence/mutations/transport-runtime.toml"],
    "message": "Add a Tier-0 regression and an MPI mutation-kill test for R-09\n\nThe Tier-0 test cannot distinguish the fix (recorded, not hidden); a new MPI test with a genuinely downstream independent channel does."
  },
  {
    "ids": ["N-transport-runtime-3"],
    "files": ["build-evidence/mutations/transport-runtime.toml"],
    "message": "Correct a mismarked mutation's requires tag and stale comment\n\ncross-rank-region-check-fires-unconditionally is Tier-0-killed already; drop requires = \"mpi\" and the wrong justification."
  }
]
```

## Fixed

**N-transport-runtime-1** (Constrain's rank-local conservation diagnostic).
`resolve_constrain()` (`src/runtime/runner.cpp`) now refuses at setup,
`E-RT-CONSERVATION-MPI`, when `RunnerConfig::base_communicator` is set and
the Remap's target component is declared on more than one rank.

**Why refuse instead of reduce.** `weights_for()`'s integral is built from
this rank's own owned entities with every ghost zeroed (docs/dev/rank-
agreement.md's ownership convention) -- once a channel's target spans more
than one rank, that is only this rank's share of the true global integral,
not the integral itself, so comparing two such shares in isolation can
both false-fire on a correct cross-rank remap and miss a real defect that
only shows up once every rank's share is summed. `reduce_region_residuals()`
already solves exactly this for iteration-region convergence (R-10), but
copying that pattern into Constrain needs `OpKind::Constrain` reclassified
in `fuse::is_collective()` (`include/fuse/ir/op.hpp`) -- outside this pass's
owned paths -- and skipping that reclassification is not safe to do
silently: a rank that has not yet failed elsewhere in the period would
reach a new cross-rank reduction inside Constrain expecting every
participant, while a peer that already failed earlier in the SAME period
skips Constrain entirely (it is marked non-collective on purpose, Phase 4:
"reports; it does not correct") and never sends its half. That is the exact
hang class R-09 fixed, reopened one instruction over. Refusing at setup
needs no collective at all: `graph.component(dst).ranks` is the declared
placement, identical on every rank, so every rank reaches the same verdict
independently.

Verified by `test/runtime/runner_test.cpp`:
`RefusesAConservationCheckOnAMultiRankTarget` (the refusal fires) and its
companion `StillRunsAConservationCheckWithNoCommunicatorConfigured` (the
same nominal multi-rank placement is accepted when no communicator is
configured -- the R-10-style regression guard every existing single-process
caller needs). Mutation `conservation-check-skips-the-multi-rank-refusal`
added to `build-evidence/mutations/transport-runtime.toml`; hand-applied
(`if (false)`) and confirmed to make `RefusesAConservationCheckOnAMultiRankTarget`
fail, then reverted. Full `fuse_runtime_runner_test` suite: 46/46 pass with
the fix in place.

**N-transport-runtime-3** (mismarked mutation). Removed `requires = "mpi"`
and the wrong justification comment from `cross-rank-region-check-fires-
unconditionally`; see the toml's own comment for why (`iteration.cpp` has
no `#if`/`#ifdef` around the guarded block, and the check reads a declared,
not a real, communicator).

## R-09 (PARTIAL) -- what was added, and what it does and does not prove

**Added.** `test/runtime/runner_test.cpp`:
`ASecondChannelStillAdvancesAfterAnUnrelatedRegionFails` (Tier 0). A second,
independent channel "linkB" shares producer_a/consumer_a's clock but is not
one of the iteration region's channels; producer_a refuses every advance();
the test asserts `run_period()` reports `E-TEST-ADVANCE-REFUSED` and that
consumer_b's own advance() still ran (`cons_b->steps == 1`).

**What this does NOT prove, verified rather than assumed.** Reverting
R-09's own one-line change in `step_period()` (`if (!rr.ok()) return rr;`
in place of the `step_failure = rr.error(); i = r.exit + 1; continue;` the
fix added) and rebuilding+rerunning this exact test: it still **passes**.
Confirmed by direct compile/link against both the mutated and the restored
`runner.cpp` (the shared build tree's own `cmake` reconfigure was
oscillating under six fixers editing `test/CMakeLists.txt` concurrently, so
this was done by extracting the two ninja/make compile and link commands
from `compile_commands.json` and running them directly against the
in-place source, once per code state). The reason is structural, not a
weak assertion: at Tier 0 (no MPI, no real communicator) the abandonment
R-09 fixes is unobservable -- there is no partner rank waiting on the
instructions that get skipped, so nothing this test can read differs
between the two code states. This test is a legitimate regression guard
for a related, weaker property (a later channel's own tick is not silently
undone by an earlier, unrelated region's failure), and it is recorded as
exactly that in its own comment -- not oversold as a mutation-kill.

**A genuinely Tier-0-constructible "second channel after the region" is
architecturally impossible for a channel with any collective op**, verified
by extended experimentation, not asserted: `OperatorDAG::topological_order()`
is Kahn's algorithm with ties broken by ascending `NodeId`
(`include/fuse/ir/dag.hpp`), and a region's own convergence node (`Residual`)
is created in a lowering phase that runs strictly after every channel's own
operator chain (`Remap`/`Diagnose`/`Transport`/`Epoch`) is emitted -- so any
second channel's non-boundary node always has a lower id than the region's
own exit node and sorts before it whenever both are ready at once, and
declaring the second channel's whole graph before the region (the only
ordering the scheduler's `E-SCHED-REGION` contiguity check accepts) places
that same ordering advantage on its side too. Only a channel's terminal
`Sink` -- created in a still-later lowering phase, and non-collective by
`fuse::is_collective()` -- can land after the region; that is what the
Tier-0 test above exercises, and it is exactly why it cannot distinguish
R-09's mutation.

**The real mutation-kill test is MPI-only, `test/parallel/runner_transport_test.cpp`:**
`RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi`. It
extends `TwoComponentJob` with an optional second, independent channel
("linkB") between two new components ("producer_b"/"consumer_b") declared
on the SAME `fluid_`/`solid_` disjoint rank sets the region's own channel
uses, and extends `StubComponent` with an optional `fail_after` (default
-1, every existing caller unaffected) so "fluid" can refuse its very first
`advance()`. Under a real multi-rank run this makes "linkB"'s Transport and
Epoch genuinely collective and genuinely downstream of the region -- the
shape Tier 0 cannot build. Compiled clean at Tier 1
(`mpicxx -std=gnu++20 -fsyntax-only`, MPICH 5.0 at `/opt/mpich`; a full
`cmake --build` of the target could not be completed in this pass, see
"Needs a host run"). Mutation `region-failure-abandons-the-rest-of-the-period`
added, `requires = "mpi"`, targeting this test with the exact one-line
revert; its `describes` field states plainly that the Tier-0 test's
SURVIVED verdict is expected, and that this MPI test with the 120s TIMEOUT
property is the actual kill (the same pattern the pre-existing
`region-residual-skips-the-cross-rank-reduction` mutation already uses for
R-10's own analogous hang).

## Rejected

None. Every PARTIAL/NOT-FIXED row and every new finding in scope was
addressed above or routed below.

## Needs another owner

- **N-transport-runtime-4** (`docs/dev/reviews/2026-09-19/fixes/fix-
  transport-runtime.md`'s "Needs another owner" section is stale: the
  compile blocker it names was already closed by this fix wave's own
  `41535ff`). Outside every fixer's owned paths (a docs/dev/reviews file,
  not include/src/test). Exact change: delete that section, or replace it
  with a note that nothing needs routing.
- **N-transport-runtime-6** (same file, "Fixed: R-50" section overcounts
  `halves()`-based hand-rolled skips in `mpi_transport_test.cpp` as 8; it is
  7). Same file, same reason it is out of scope here. Exact change: `8` ->
  `7` in that one sentence.
- **N-transport-runtime-5** (`tools/check-layering.py`'s `LEGACY_MACRO`
  regex still does not match a `-D<KARMA|SOMA>_*` spelling). The reviewer's
  own text says "No fix required for this pass; flagged for whoever next
  touches the gate script" -- recorded here only so it is not lost, not
  because it blocks this pass.

## Needs a host run

- `test/parallel/runner_transport_test.cpp`'s
  `RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi`
  (new). Needs `mpiexec -n 2` (or more) on a host that can actually launch
  MPI; this sandbox's `mpiexec` fails identically to the first pass's own
  note (`HYDU_sock_create_and_listen_portstr: failed to bind to any port`).
  Compiled clean at Tier 1 via direct `mpicxx -fsyntax-only`; a full `cmake
  --build`/`ctest` pass of the whole `parallel` suite (including this test
  and the pre-existing ones) still needs to run on a host that can launch
  MPI, to confirm both that this test passes with the fix in place and
  that the `region-failure-abandons-the-rest-of-the-period` mutation
  produces the predicted timeout/hang when hand-applied there.
- The pre-existing MPI legs of `fuse_parallel_runner_transport_test`,
  `fuse_parallel_mpi_transport_test`, and `fuse_parallel_comm_graph_test`
  were not re-run in this pass either (same constraint); only compile-
  checked.

## Deviation decisions

- **Manual compile/link instead of `cmake --build`/`ctest` for the Tier-0
  verification.** The shared `build-t0` tree's `cmake` reconfigure step
  kept failing with `test/num/convergence_test.cpp` registered nowhere,
  because five other fixers were concurrently editing `test/CMakeLists.txt`
  and `test/num/` in the same working tree. Waited and retried repeatedly
  (the rule of thumb in the shared REVIEW.md task: build in one's own
  working directory only); when it kept recurring, extracted the exact
  compile and link commands for `fuse_runtime_runner_test` from the tree's
  own `compile_commands.json` and ran them directly, once compiling the
  test file, once compiling `runner.cpp` under each of the two code states
  (fixed, then hand-mutated back to pre-R-09, then restored), relinking
  between each. Every command run this way was extracted from, and used
  the same flags as, the tree's own generated build files -- this is not a
  different build configuration, only a workaround for the configure step
  itself being unstable under concurrent edits elsewhere in the tree.
- **Disk space.** `/System/Volumes/Data` on this host dropped to a few
  hundred MB free partway through this pass (six fixers' build trees
  sharing one machine); two scratch snapshot build directories used while
  isolating the R-09 experiment from the same concurrent-edit problem were
  deleted once no longer needed. No files under a granted or owned path
  were removed; the deleted directories were all under `/tmp/work/`,
  outside the repository.
- **`StubComponent`'s and `TwoComponentJob`'s new optional parameters**
  (`fail_after`, `with_independent_channel`, `fail_fluid_after`) all default
  to their previous behaviour, so every existing caller in
  `test/parallel/runner_transport_test.cpp` is unaffected; not treated as a
  separate commit unit since it exists solely to support the new MPI test
  in the same unit.
