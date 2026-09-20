<!-- Second-pass review, 2026-09-19/20, HEAD dbdc872 (branch review-2026-09-19-fixes). -->
<!-- Answers the first review (docs/dev/reviews/2026-09-19/build-tools-docs.md, HEAD bdf4346) -->
<!-- and its fix pass (docs/dev/reviews/2026-09-19/fixes/fix-build-tools-docs.md). -->

# FUSE build/tools/docs — second-pass review, HEAD `dbdc872`

Layer group: `CMakeLists.txt`, `src/CMakeLists.txt`, `test/CMakeLists.txt`, `cmake/`,
`CMakePresets.json`, `tools/`, `build-evidence/`, `.github/`, `UPSTREAMS.lock`, `KNOWN_ISSUES.md`,
`REVIEW.md`, `docs/`, `examples/`, `test/install_consumer`. Read-only: nothing under
`/Users/mahadevan/Code/AI/coupling/fuse` was edited, no git command that changes state was run, and
every build directory (`b0`, `b1`, `bmut`, `scratch_repo`) was created under my own sandbox
workspace, not the repository. One command briefly wrote a stray `review_docs_build/` directory
into the granted host path (a sibling of `fuse/`, from a `sphinx-build` invocation run from the
wrong working directory); it has been removed via the trash-safe delete tool and is noted here for
the record — no file under the repository itself was touched.

Built `tier0` with `cmake --preset tier0 -S <repo> -B <workspace>/b0` and `cmake --build ./b0 -j4`;
full `ctest -j4` run there (71/71 pass). Also configured (not built) `tier1`
(`-DFUSE_ENABLE_MPI=ON`) to check the install/export path with MPI present. Tier 2 does not
configure here (MOAB was built against a superseded gcc) and MPI cannot initialize in this
sandbox — both are known, out of scope, and unchanged by this review.

## Verification

All 17 unified findings assigned to this layer (`BT-F1`..`BT-F17` / `R-05`, `R-06`, `R-07`, `R-20`
through `R-25`, `R-36` through `R-42`, `R-47`), plus two cross-layer entries recorded in `REVIEW.md`
that this layer's own scope covers (`R-51`, `R-31`).

| id | verdict | evidence |
|---|---|---|
| R-05 | CONFIRMED-FIXED | `.github/workflows/ci.yml:57` matrix is now `linalg: [eigen, builtin]`, matching `FUSE_LINALG_BACKEND`'s `STRINGS builtin eigen` in `cmake/FuseDependencies.cmake:62`. Both legs are real values; the leg itself still cannot be executed here (no runner), as before. |
| R-06 | CONFIRMED-FIXED (owned by backend layer; verified here since it gates BT-F17/R-47) | `test/backend/moab_mesh_test.cpp:1-8` now includes `fuse/base/config.hpp` before `#if FUSE_HAVE_MOAB` (was `#ifdef`, no preceding include). Confirmed by reading the file; the tier-0 build here cannot exercise the MOAB path (`FUSE_ENABLE_MOAB` is OFF at tier 0 by the ladder), so the 38 cases still need a host tier-2 run, unchanged. |
| R-07 | CONFIRMED-FIXED | `test/CMakeLists.txt:98` registers `schedule buffers THROWS` and `schedule slab`; both ran in my tier-0 `ctest`: `schedule.buffers` and `schedule.slab` pass (71/71 total). The registration-audit block (`test/CMakeLists.txt`, bottom) also now exists as a standing gate against a repeat. |
| R-20 | CONFIRMED-FIXED | `tools/sync-upstream.py:lock()` now parses `UPSTREAMS.lock` with `tomllib` instead of a line regex. Ran it directly against the real file: `soma` returns 50 superseded paths, `karma` 15 (both non-empty; the old parser returned 0 for both). |
| R-21 | CONFIRMED-FIXED, independently re-verified | The `epoch-schedule-drops-the-answer` anchor in `build-evidence/mutations.toml:447` now targets `s.result_send_count = participants - 1;` (the current binomial-tree field), not the removed `result_send_to` loop. `tools/mutate.py --check-anchors` reports 0 stale of 86. Not trusting the committed **KILLED** verdict on the word of the log, I reproduced it myself: copied the tree to an isolated scratch directory (excluding `build*`/`upstream`/`.git`), applied the exact `old`→`new` substitution from the manifest, configured+built `fuse_runtime_epoch_schedule_test` there, and ran it — `EpochSchedule.EveryParticipantEndsUpWithTheAnswer` and `EveryStatusHasExactlyOneMatchingReceive` both fail, so `ctest -R ^runtime.epoch_schedule$` returns non-zero, which is exactly `mutate.py`'s own KILLED rule. The verdict is genuine. (See N-build-tools-docs-8 for a smaller process gap this surfaced.) |
| R-22 | CONFIRMED-FIXED | `UPSTREAMS.lock`'s `[soma].superseded` array now carries `"include/soma/mesh/", "src/mesh/"` and `"include/soma/remap/", "src/remap/"` and `"test/unit/moab_mesh_test.cpp"` as real array entries (not the bare `#`-comment the first review found). Verified end-to-end: imported `sync-upstream.py`'s `classify()` and ran it against the four exact upstream paths named in the original finding (`include/soma/mesh/moab_backend.hpp`, `src/mesh/moab_backend.cpp`, `src/remap/tempest_weights.cpp`, `test/unit/moab_mesh_test.cpp`) — all four now classify `"superseded"`. |
| R-23 | CONFIRMED-FIXED | `KNOWN_ISSUES.md:165` — B1's heading is struck through, "CLOSED 2026-09-18, superseded by C1", original text preserved in a `<details>` block. `C0` (line 233) also now correctly cites `R-06` for the MOAB-test-compiles-to-nothing description and explicitly disclaims the old `R-09` mis-citation. (A *different* line of `C0`/`KNOWN_ISSUES.md` still carries stale numbers the fix pass didn't touch — see N-build-tools-docs-2 and -3.) |
| R-24 | CONFIRMED-FIXED | `docs/user/build.md:9` now reads "one of `builtin` (no dependency) or Eigen3"; the backend section (lines 25-42) documents `builtin`/`eigen` only and states plainly that a KokkosKernels backend "was designed but is not implemented" (line 42). The false "both C ABIs, the Fortran shim" claim is gone from the tier-0 row. |
| R-25 | CONFIRMED-FIXED, built and run | `src/CMakeLists.txt` carries `EXPORT_NAME` on all 9 layer targets, `install(TARGETS ... EXPORT fuse-coupler-targets)`, `install(EXPORT ...)`, `configure_package_config_file`/`write_basic_package_version_file`, backed by a new `cmake/fuse-couplerConfig.cmake.in`. Built tier 0 here, ran the full suite: `install.consumer_find_package` passes (3.3s). Also configured `tier1` (`FUSE_ENABLE_MPI=ON`) to check that `fuse-coupler-transport`'s PUBLIC `MPI::MPI_CXX` link doesn't break `install(EXPORT)` at generate time — it configures cleanly. The disclosed gap (a tier-1/2 consumer needs its own `find_dependency(MPI)`/MOAB; `fuse-couplerConfig.cmake.in` only resolves Kokkos/fmt) is real and correctly flagged by the fixer as unverifiable without a host — I could not do better than confirm it doesn't break tier 0/1 *configure*. |
| R-36 | CONFIRMED-FIXED | `src/backend/intx_weights.cpp:36` now reads `#if defined(MOAB_HAVE_MPI) && FUSE_HAVE_MPI` — no `defined()` on the FUSE macro, matching `moab_backend.cpp:24`'s pattern exactly. |
| R-37 | CONFIRMED-FIXED | `CMakeLists.txt:70-83` computes the layer list from `TARGET fuse::<name>` checks rather than a literal string. Configuring tier 0 here prints `fuse:: layers : base, model, ir, schedule, linalg, num, transport, backend, runtime` — all nine, correct. |
| R-38 | CONFIRMED-FIXED | `tools/check-layering.py`'s `check_legacy()` (line 185 area) now blanks `//` comments (`COMMENT_LINE.sub`) before applying `LEGACY_NAMESPACE`/`LEGACY_TYPE_PREFIX`/`LEGACY_MACRO`. Ran `python3 tools/check-layering.py` against the real tree: 0 violations, matching the fix report's synthetic-probe claim. |
| R-39 | CONFIRMED-FIXED | `docs/dev/layering.md`'s diagram now lists `linalg` and `field` with a `(*)` footnote, and the `interface` row is renamed `abi`. Read in full; matches `tools/check-layering.py`'s actual `LAYERS` list (11 entries, confirmed by inspection). |
| R-40 | CONFIRMED-FIXED | `CMakePresets.json:11` now reads `"binaryDir": "${sourceDir}/build-${presetName}"`. My own `tier0`/`tier1` configures land at `./b0`, `./b1` under the names I gave `-B`, consistent with the `build-<preset>` convention (I passed explicit `-B` rather than using the preset's own binaryDir, but the string itself is the fix's target and is correct). |
| R-41 | **NOT-FIXED** (expected; flagged "needs another owner" by the fixer, correctly) | `test/backend/moab_mesh_test.cpp:614`'s `#ifdef FUSE_MOAB_TEST_MESH` block is still present, and grepping the whole non-`build*`/non-`upstream` tree for `FUSE_MOAB_TEST_MESH` finds no `target_compile_definitions` anywhere. This is `test/backend/**`, owned by the backend layer, not build-tools-docs; the fixer's own report correctly declines to fix it and flags it for that owner. Still open at HEAD. |
| R-42 | CONFIRMED-FIXED | `cmake/FuseDependencies.cmake` has no `FetchContent_Declare(KokkosKernels ...)` anywhere; grepped the file and the whole tree for `kokkoskernels` (case-insensitive) — the only hits left are prose explaining why it isn't there (lines 71, 99-104) and `docs/user/build.md:41`'s equivalent note. |
| R-47 | CONFIRMED-FIXED | `tools/check-invariants.py:73`'s `scan()` area tuple is now `("include", "src", "test", "examples")`. Ran it: 0 violations at HEAD, and it is what makes R-06's regression class (and R-51's) mechanically un-repeatable — confirmed by reading the extended rule (lines 39, 86-101), which forbids `#ifdef`/`#ifndef`/`defined()` on any `FUSE_HAVE_*`/`FUSE_ENABLE_*` macro tree-wide now. |
| R-51 *(cross-layer; recorded in `REVIEW.md`, which is in my scope)* | CONFIRMED-FIXED | `test/parallel/comm_graph_test.cpp:34-42` now includes `config.hpp` before `#if FUSE_HAVE_MPI` (was `<gtest/gtest.h>` only). It is registered in `test/CMakeLists.txt:184-197` and compiles under tier 1; the file itself is owned by another layer, so I only re-confirm the macro guard and registration are correct, not the test bodies. |
| R-31 *(cross-layer; reclassification recorded in `REVIEW.md`)* | REJECTION-UPHELD, agree | Re-read both cited lines: `include/fuse/transport/transport.hpp:255` and `src/transport/lowfive_transport.cpp:66` both read `FUSE_ENABLE_LOWFIVE`, and neither is a conditional (one is a doc comment, one is an error-message string). The MAJOR→MINOR reclassification in `REVIEW.md` is correct on the evidence in front of me; I did not re-derive whether LowFive itself works (out of scope, untouched by this layer group). |

Static gates and audits re-run directly, all clean: `python3 tools/check-layering.py` → 0
violations; `python3 tools/check-invariants.py` → 0 violations; `python3 tools/verify-import.py` →
693 files, 0 problems; `python3 tools/mutate.py --check-anchors` → 86 valid, 0 stale, of 86;
`sphinx-build -W --keep-going` → build succeeded, 0 warnings.

## New findings

| id | severity | file:line | title |
|---|---|---|---|
| N-build-tools-docs-1 | MAJOR | `docs/dev/roadmap.md:13-14` | S4/S5 rows say the MOAB backend and the lazy field pipeline "remain", contradicting `KNOWN_ISSUES.md` C1, `UPSTREAMS.lock`'s S5.2 entry, and the tree itself |
| N-build-tools-docs-2 | MAJOR | `KNOWN_ISSUES.md:293-296` | Section C claims "FUSE's own suite has no MOAB-dependent test at all... there is no MOAB backend here yet", directly contradicted by C1 fifty lines above and by the now-closed B1 in the same file |
| N-build-tools-docs-3 | MINOR | `KNOWN_ISSUES.md:227,229,285-286,288-291` | C0's finding-count header and Section C's mutation-count bullets are stale relative to `REVIEW.md`'s own current header and the current evidence logs |
| N-build-tools-docs-4 | MINOR | `tools/check-invariants.py:71` | Comment cites "REVIEW.md R-09" for the MOAB-test-compiles-to-nothing defect; the correct id is R-06 (R-09 is the unrelated `step_period()` BLOCKER) |
| N-build-tools-docs-5 | MINOR | `test/CMakeLists.txt:30` | Comment cites "REVIEW.md R-10" for the buffers/slab-never-registered defect; the correct id is R-07 (R-10 is the unrelated iteration-region rank-local-residual MAJOR) |
| N-build-tools-docs-6 | MINOR | `cmake/FuseDependencies.cmake:1,77,85`, `UPSTREAMS.lock:1` | Four comments cite `docs/PLAN.md` or `docs/CHARTER.md`; neither file exists anywhere in the repository |
| N-build-tools-docs-7 | MINOR (out of my owned paths; flagged for the backend layer, same pattern as R-41) | `test/backend/moab_mesh_test.cpp:5` | Comment cites "REVIEW.md R-09" for the same MOAB defect; should be R-06 |
| N-build-tools-docs-8 | IMPROVEMENT | `build-evidence/runtime-mutations.md` | The rewritten `epoch-schedule-drops-the-answer` anchor's KILLED verdict is presented as a fresh "Run 2026-09-19" with no note on whether it came from an actual `tools/mutate.py --write-log` run or a hand-edit; the verdict is correct (I independently reproduced it — see R-21 above) but the file's own convention ("do not edit... regenerate with `--write-log`") gives a future reader no way to tell which happened here |

### N-build-tools-docs-1 — MAJOR — `docs/dev/roadmap.md:13-14` — the roadmap understates what is already built

**S5**'s row (`docs/dev/roadmap.md:14`) reads: *"in progress — `LinearOperator` and its conformance
fixture are in; transports were already done and overlapping rank sets are supported. Remaining:
the MOAB mesh backend, persisted maps, and the parallel matrix as a gate."* This is false at HEAD,
and was already false at the first review's baseline `bdf4346`: the MOAB backend landed in commit
`a2816cd` ("Port the MOAB mesh backend and the high-order entry points"), which is an ancestor of
`bdf4346` (confirmed with `git merge-base --is-ancestor a2816cd bdf4346`). `include/fuse/backend/moab_backend.hpp`
and `src/backend/moab_backend.cpp` exist and are wired into `fuse-coupler-backend`
(`src/CMakeLists.txt:225-238`); `KNOWN_ISSUES.md`'s C1 entry and `UPSTREAMS.lock`'s S5.2
supersession note both independently confirm the port landed 2026-09-18, before the review baseline.
What is actually missing is a host run of its 38-case test suite — a "designed→implemented,
untested" gap in `testing.md`'s own five-states vocabulary, not an "absent" one, and the roadmap's
"Remaining" list currently claims the stronger, wrong thing.

**S4**'s row (line 13) has a narrower version of the same defect: *"Field transactions, the lazy
field pipeline and the device leg remain."* `include/fuse/field/pipeline.hpp`, `field_ref.hpp` and
`registry.hpp` exist, and `field.pipeline` is a registered, passing test (confirmed in my tier-0
`ctest` run: `field.pipeline ... Passed`). `git log --diff-filter=A -- include/fuse/field/pipeline.hpp`
shows it was added in `add9ddc`, also an ancestor of `bdf4346` — the pipeline predates the review
baseline too. Grepping the tree for `FieldTransaction` (the concept `docs/dev/porting.md`'s merge
table names) finds nothing, so "field transactions" is still correctly listed as remaining; only
"the lazy field pipeline" clause is stale. "The device leg" is also still accurate as written: per
`docs/dev/testing.md`'s host-verified log, Kokkos here only has `OpenMP`/`Serial`/`Threads`, no CUDA
or HIP space, so no actual device coverage has run.

**Fix.** Update S5's row to say the MOAB backend has landed (code exists, wired in, built) and that
what remains is the host run of its suite plus persisted maps and the parallel matrix — matching
`KNOWN_ISSUES.md` C1's own wording. Update S4's row to drop "the lazy field pipeline" from the
remaining list, or split the sentence so "field transactions" (still genuinely absent) doesn't share
a clause with something that already exists and is tested.

### N-build-tools-docs-2 — MAJOR — `KNOWN_ISSUES.md:293-296` — Section C repeats the exact premise B1/C0 already retracted

```
- **The four `SkippedWithoutMoab` entries are the PARENT's, not FUSE's.** They
  live in `upstream/soma/test/unit/{moab_mesh,remap,remap_order,remap_runtime}_test.cpp`
  and are run by the imported-suite CI leg. FUSE's own suite has no
  MOAB-dependent test at all, which is consistent with B1: there is no MOAB
  backend here yet to test. Do not read those four skips as FUSE coverage.
```

This is the same false premise `R-23`'s fix already struck through at B1 (line 165: *"CLOSED
2026-09-18, superseded by C1... stale and false at HEAD"*) and the same citation the fixer corrected
in `C0` (line 233-234: *"REVIEW.md R-06, not R-09 as first written here"*). `test/backend/moab_mesh_test.cpp`
is FUSE's own MOAB-dependent test, registered by `test/CMakeLists.txt:157-160` whenever
`FUSE_ENABLE_MOAB` is on, 38 cases. The claim "FUSE's own suite has no MOAB-dependent test at all"
is simply false at HEAD, and "which is consistent with B1" points at an entry that is itself now
struck through for saying the same thing. The fix pass edited B1 and C0 in this exact file, in this
exact review, and did not propagate the correction three sections down to C.

**Fix.** Replace the bullet's second sentence. FUSE's own suite *does* have a MOAB-dependent test
now (`backend.moab_mesh`, 38 cases); what it lacks is a host run, per C1. The bullet's actual point —
don't read the four `upstream/soma` skips as FUSE coverage — still stands and should be kept, just
without the false premise attached to it.

### N-build-tools-docs-3 — MINOR — `KNOWN_ISSUES.md:227,229,285-286,288-291` — stale counts left behind by the fix pass

Three counts in this file no longer match the evidence that produced them, all in sections the fix
pass touched elsewhere without updating the count:

1. **C0's header and first line** (`227`: *"50 findings"*; `229`: *"Ten BLOCKER, 22 MAJOR"*) —
   `REVIEW.md`'s own header now reads *"51 findings: 11 BLOCKER, 21 MAJOR, 15 MINOR, 2 IMPROVEMENT,
   2 DEVIATION"* (verified by counting the actual table rows: 11+21+15+2+2 = 51, ids R-01..R-51 all
   present exactly once). `R-51` was added and `R-31` reclassified MAJOR→MINOR *while fixing this
   very file's B1/C0* (`REVIEW.md`'s own note, and the same commit wave); the two BLOCKER/MAJOR
   counts were never bumped to match.
2. **"Two mutations report `NOT-ASSESSED` locally"** (line 285) — counting every `NOT-ASSESSED` row
   across `build-evidence/*.md` gives **5**, not 2: 3 in `transport-mutations.md` (`peer-plan-skips-
   the-numbering-check`, `peer-plan-drops-the-local-copies`, `build-plan-skips-the-components-
   agreement`) and 2 in `runtime-mutations.md` (`cross-rank-region-check-fires-unconditionally`,
   `region-residual-skips-the-cross-rank-reduction`). The three new MPI-gated mutations were added
   by commit `93e3fe9` during this fix pass; the KNOWN_ISSUES count was not updated.
3. **"Two mutations are declared equivalent"** (line 288) — there is now only **one**:
   `build-evidence/mutations.toml`'s "backend one" (`supermesh-coverage-computed-before-the-floor`)
   had its `equivalent` declaration withdrawn on 2026-09-19 (the manifest's own comment at that
   entry: *"the declaration was withdrawn 2026-09-19... this is now an ordinary mutation that must
   be killed"*), and `build-evidence/backend-mutations.md` records it **KILLED**, not equivalent.
   The remaining equivalent mutation is `ir.scale`'s `unlowerable-step-silently-dropped`
   (`compiler-mutations.md`), a different one from the "backend" one the bullet still describes.

**Fix.** Update C0's count to 51/11/21 (matching `REVIEW.md`), and rewrite the two bullets in
Section C to name the current counts (5 NOT-ASSESSED, 1 EQUIVALENT) and, for the equivalent one, the
mutation that is actually still declared that way.

### N-build-tools-docs-4 — MINOR — `tools/check-invariants.py:71` — mis-cites R-09 for R-06

```python
    # `include` and `src` carry every rule. `test` and `examples` carry only the
    # feature-macro rules: a test may legitimately throw a std:: type in a
    # fixture, but a test that reads FUSE_HAVE_MOAB without config.hpp compiles
    # to a skip stub in every configuration and reads like a pass (REVIEW.md
    # R-09: 38 MOAB cases, never once executed). Found 2026-09-19 -- the same
```

`R-09` in `REVIEW.md` is `TR-F1`: *"A failing iteration region makes `step_period()` abandon the
rest of the period's instruction stream..."* — an unrelated `runtime`/`transport` BLOCKER. The
38-MOAB-cases-compile-to-a-skip-stub defect this comment is actually describing is `R-06` (`BT-F2`).
This is the identical mistake the fixer already found and corrected once, in `KNOWN_ISSUES.md`'s
`C0` (*"REVIEW.md R-06, not R-09 as first written here; R-09 is the unrelated `step_period()`
collective-agreement BLOCKER"*) — introduced by the same commit (`4a2ad6e`, confirmed by `git log -p
-S"R-09" -- tools/check-invariants.py`) that made that correct fix elsewhere, and not caught here.
Grepped every `R-09` in the tree outside `upstream/`/`build*`: this is one of exactly two wrong
instances (the other is N-build-tools-docs-7); every other `R-09` reference (in `runner_test.cpp`,
`runner.cpp`, `iteration.cpp`) correctly names the real `step_period()` fix.

**Fix.** Change `R-09` to `R-06` on this line.

### N-build-tools-docs-5 — MINOR — `test/CMakeLists.txt:30` — mis-cites R-10 for R-07

```cmake
# Every registration function records its source in FUSE_REGISTERED_TEST_SOURCES;
# the audit at the bottom of this file compares that against the files on disk.
# A test file that exists but is registered nowhere runs nowhere, and nothing
# else reports it: 33 cases in schedule/buffers_test.cpp and slab_test.cpp sat
# unregistered for four days (REVIEW.md R-10).
```

`R-10` is `TR-F2`: *"Iteration-region convergence is decided from a purely rank-local residual with
no cross-rank agreement..."* — a `runtime` MAJOR, unrelated to test registration. The
buffers/slab-never-registered defect this comment describes is `R-07` (`BT-F3`), confirmed by
`REVIEW.md`'s own row (`R-07 | [BT-F3] | test/CMakeLists.txt:97-101 | ... buffers_test.cpp and
slab_test.cpp (33 test cases) are never registered ...`). Introduced by commit `b8e3f54` (the wave-0
registration-audit commit, `git log -p -S"33 cases in schedule/buffers_test.cpp"`), which is itself
one of the fixes this citation is supposed to be crediting — it got its own id wrong. Grepped every
`R-10` in the tree outside `upstream/`/`build*`/the transport-runtime fixer's own report: this is the
only wrong instance; every other `R-10` (in `runner_test.cpp`, `iteration.cpp`, `runner.cpp`,
`runner_impl.hpp`, `runner_transport_test.cpp`, `transport_test.cpp`, `tags.hpp`) correctly names the
real iteration-region residual-agreement fix.

**Fix.** Change `R-10` to `R-07` on this line.

### N-build-tools-docs-6 — MINOR — dangling cross-references to `docs/PLAN.md` / `docs/CHARTER.md`

Four comments in my owned paths cite a versioned doc page that does not exist anywhere in the
repository (`find . -iname PLAN.md -o -iname CHARTER.md`, excluding `upstream/`/`build*`, returns
nothing):

- `cmake/FuseDependencies.cmake:1` — `"FUSE dependency resolution, one tier at a time (docs/PLAN.md §3.1)."`
- `cmake/FuseDependencies.cmake:77` — `"...never a silent downgrade (docs/CHARTER.md §4)."`
- `cmake/FuseDependencies.cmake:85` — `"fmt — the ONLY output formatter in the library (docs/CHARTER.md)."`
- `UPSTREAMS.lock:1` — `"Bump ONLY through docs/PLAN.md §6 (sync protocol)."`

`CHARTER.md` was retired: `plans/superseded-charter.md` exists on disk (confirmed) and its content
was redistributed — the tier-is-a-prefix rule line 77 points at is actually in
`docs/user/build.md:14` (*"A tier is a **prefix**: enabling a higher tier without its predecessors is
a configuration error..."*), and the fmt-is-the-only-formatter rule line 85 points at is in
`docs/dev/conventions.md` (*"`fmt` is the only formatter. No `iostream`..."*). `PLAN.md` never
existed under `docs/` at all: the real task-level plan is `plans/2026-09-15-fuse-plan.md`, which
`.gitignore:19` explicitly excludes from version control (*"Task-level plans stay local, as in the
parent projects."*) — confirmed with `git ls-files plans/`, which returns nothing. So even the
*correct* path is one no clone of this repository will ever have; citing it from tracked source is a
reference that can never resolve for anyone but the machine it was written on. This is the same
class of defect as `R-34` (`BM-M6`, a different layer): a doc cross-reference that points at nothing.
Two more instances of the identical pattern exist in `test/support/fuse_test_main.hpp:12,45`, outside
my owned paths (that file is shared test-harness infrastructure, not clearly any one layer's), noted
here for whichever layer's second pass covers it.

**Fix.** Point `cmake/FuseDependencies.cmake:77,85` at `docs/user/build.md` and
`docs/dev/conventions.md` respectively (the actual current homes of the content). Drop the
`docs/PLAN.md` citations at `cmake/FuseDependencies.cmake:1` and `UPSTREAMS.lock:1`, or replace them
with a citation to a tracked page — the task-level plan is deliberately untracked and cannot be a
citation target for a comment meant to survive a clone.

### N-build-tools-docs-7 — MINOR, out of my owned paths — `test/backend/moab_mesh_test.cpp:5` — same R-09/R-06 mis-citation

```cpp
// Before the #if: without this the macro is undefined, the preprocessor reads
// 0, and all 38 cases below compile to the skip stub at the bottom -- which is
// exactly what happened from the day this file was ported (REVIEW.md R-09).
#include "fuse/base/config.hpp"
```

Same defect and same origin commit (`4a2ad6e`) as N-build-tools-docs-4: should read `R-06`, not
`R-09`. `test/backend/**` is owned by the backend layer's review, not build-tools-docs (my scope
list names `test/CMakeLists.txt` and `test/install_consumer` specifically, not `test/backend/`), so I
am not fixing it, only flagging it — the same posture the fixer took for `R-41` ("needs another
owner"). Recorded here because the wrong id points at `REVIEW.md`, which is my file, and because it
is the third instance of an otherwise-rare mistake (see N-4, N-5), worth the backend reviewer's
attention as a pattern, not just a typo.

**Fix.** For the backend layer's owner: change `R-09` to `R-06` on this line.

### N-build-tools-docs-8 — IMPROVEMENT — `build-evidence/runtime-mutations.md` — a re-verified verdict presented with no provenance

The fixer's own report (`fix-build-tools-docs.md`) is explicit that the `epoch-schedule-drops-the-
answer` mutation could not be re-run here (*"needs `tools/mutate.py --area runtime --build-dir <dir>
--write-log` on a CLEAN tree... I confirmed the anchor text is valid... but could not run the full
mutation"*) and added a caveat sentence to that effect to `runtime-mutations.md`. Two later commits
(`07f16ae`, `9c990e1`) removed that caveat, changed the file's header from *"Run 2026-09-16"* to
*"Run 2026-09-19 against build `build-fuse-s1`"*, and left the row's verdict as **KILLED** — with no
statement of whether an actual `tools/mutate.py --write-log` run produced that "2026-09-19" date for
*this* row, or whether the header date was bumped for the *other* rows those commits genuinely did
change (the `R-10`-guard and coverage-audit rows in the same file) while this row's text was carried
over unchanged from the stale-anchor era. `testing.md`'s own convention for this file family is
"regenerate with `tools/mutate.py --write-log`; do not edit" — a header date bump without a
corresponding tool run is exactly the kind of hand-edit that convention exists to prevent, evidence-
trail-wise, even when (as here) the substance turns out to be correct. I did not take this on faith:
I independently reproduced the mutation in an isolated scratch copy (see R-21 in Verification above)
and confirmed **KILLED** is the right verdict. So there is no live defect — but the file itself gives
a future reader no way to distinguish "this was regenerated" from "this was retyped," which is the
one distinction the testing.md convention is written to preserve.

**Fix.** Either run `tools/mutate.py --area runtime --build-dir <dir> --write-log` for real on a
clean tree and let it regenerate this file mechanically (the intended path), or, if that is not
practical right now, restore a one-line note stating the verdict for this specific row was
independently reproduced by hand on `<date>` rather than tool-regenerated, so the provenance is
honest either way.

## Not found

- **Both static gates pass clean at HEAD.** `check-layering.py` and `check-invariants.py`: 0
  violations each, re-run directly against the working tree, not taken from a prior log.
- **`tools/mutate.py --check-anchors`: 86 valid, 0 stale, of 86.** Every anchor in
  `build-evidence/mutations.toml` and every file under `build-evidence/mutations/*.toml` matches its
  `old` string exactly once in the current source. The two anchors the fixer flagged as going stale
  *during* the fix pass (`merge-typed-at-first-writer`, `mesh-accepts-a-parent-outside-the-cell-set`)
  are both resolved at HEAD — their owning layers evidently rewrote them (`merge-typed-at-first-
  writer-v2` now exists in `compiler-mutations.md`, discussed under R-21's verification).
- **The mutation-evidence manifest and its logs still reconcile exactly.** 86 entries in the
  manifest (`mutations.toml` + `mutations/*.toml`), and summing every `*-mutations.md` file's
  "N killed / M total" footer gives 18+14+14+2+1+21+9+4+3 = 86 — no orphaned or missing log, the same
  discipline the first review confirmed at 70/70.
- **Full tier-0 suite passes: 71/71**, including `install.consumer_find_package` (R-25),
  `schedule.buffers`/`schedule.slab` (R-07), and every other test — built and run here, not asserted.
- **`tier1` configures cleanly with `FUSE_ENABLE_MPI=ON`.** Checked specifically because
  `fuse-coupler-transport` PUBLICLY links the imported `MPI::MPI_CXX` target and I wanted to know
  whether R-25's `install(EXPORT)` would choke on it the way it did on yaml-cpp/r3d before those were
  wrapped in `$<BUILD_LOCAL_INTERFACE:...>`. It does not — `find_package`-created imported targets
  don't need to be in the export set. (Building/running tier 1 is still out of scope: MPI cannot
  initialize here.)
- **`sync-upstream.py`'s TOML fix is correct end-to-end**, not just at the parser level: fed the four
  exact upstream paths named in the original F4/F6 findings through `classify()` against the real,
  current `UPSTREAMS.lock` and got `"superseded"` for all four.
- **`verify-import.py`: 693 files, 0 problems** at HEAD (manifest check; `--parents` still needs the
  parent repositories, unchanged, out of scope here as at the first review).
- **`sphinx-build -W --keep-going` succeeds with 0 warnings** across all 23 doc sources, including
  the edited `layering.md`, `build.md`, and `KNOWN_ISSUES.md`'s cross-references. (Run from a
  corrected working directory after the stray-output mistake noted above; the output was not kept.)
- **R-41 (BT-F15) is genuinely still open**, not just unverified: grepped the whole non-`build*`/
  non-`upstream` tree for `FUSE_MOAB_TEST_MESH` and found only the test file's own dead `#ifdef` and
  uses — no `target_compile_definitions` anywhere. Correctly out of this layer's fix scope (owned by
  `test/backend/**`).
- **No other `R-\d+` mis-citation found within my owned paths** beyond the three reported (N-4, N-5,
  and the four dangling-doc-page citations in N-6): grepped every `R-[0-9]+` occurrence in
  `CMakeLists.txt`, `src/CMakeLists.txt`, `test/CMakeLists.txt`, `cmake/`, `CMakePresets.json`,
  `tools/`, `build-evidence/`, `.github/`, `UPSTREAMS.lock`, `KNOWN_ISSUES.md`, `REVIEW.md`, `docs/`,
  `examples/`, `test/install_consumer/`, and checked each against the finding it names. `R-25`'s six
  occurrences, `R-51`'s one occurrence, and `R-31`'s implicit references all check out.
- **`REVIEW.md`'s own bookkeeping is internally consistent** even though `KNOWN_ISSUES.md`'s mirror
  of it is not (N-3): counted every `| R-nn |` row by severity section — 11 BLOCKER, 21 MAJOR, 15
  MINOR, 2 IMPROVEMENT, 2 DEVIATION, sums to 51, and ids R-01 through R-51 are each present exactly
  once with no gaps or duplicates.
- **`examples/CMakeLists.txt` and the two examples remain clean** — not touched by the fix pass,
  re-read, no defect (matches the first review's "Not found" for this file).
- **`cmake/fuse-couplerConfig.cmake.in` and `cmake/FuseDependencies.cmake`'s dependency-resolution
  logic (lines outside the R-42 removal) are unchanged from the first review's clean bill** — read in
  full; the OpenMP/Apple-clang workaround, the MOAB stale-toolchain guard, and `fuse_require()`'s
  fetch-vs-find logic are all as before and none is newly broken by anything else this fix pass did.

## Summary

- **17/17** unified findings in this layer confirmed fixed, except **R-41**, which is confirmed
  still open and correctly deferred to the backend layer (not a gap in this review — a scope
  boundary the original fixer respected).
- **2/2** cross-layer entries this file's scope also covers (**R-51**, **R-31**) confirmed accurate.
- **8 new findings**: 2 MAJOR (stale `roadmap.md` and `KNOWN_ISSUES.md` claims that contradict the
  tree/each other), 5 MINOR (stale counts, three `R-nn` mis-citations, four dangling doc-page
  citations), 1 IMPROVEMENT (an evidence-provenance gap on an otherwise-correct verdict).
- **0 findings rejected.** Every one of the 17 verified findings' fixes holds up against the code and
  against tests I ran myself, not just against the fixer's own account of them.
- **Deviations from a full run:** tier 2 (MOAB) and any MPI-launched test remain out of reach in this
  sandbox, unchanged from both prior reviews — every claim resting on them is marked accordingly
  above and none is reported as a firm result. Everything else in this report — the 71-test tier-0
  suite, the tier-1 configure check, the isolated mutation re-run, the doc build, and every grep-based
  cross-check — was executed directly, not inferred from the fixer's or the first reviewer's account.
