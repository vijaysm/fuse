# Second fix round — build-tools-docs, 2026-09-20

Answers `docs/dev/reviews/2026-09-19/pass2/re-build-tools-docs.md` (second-pass review, HEAD
`dbdc872`). Working tree at HEAD `1565cdf` (branch `review-2026-09-19-fixes`), five other fixers
sharing the tree concurrently, no git command that changes state run.

**Deviation from the task's full input set.** The delegation's "Desired Final Outputs" section
names five documents (`convergence-matrix.md`, `decision-register.md` Part 12,
`m0-gate-recommendation.md`, `unified-implementation-plan.md` v2, `repository-inventory.md` v2)
that belong to the earlier KARMA-absorbs-SOMA convergence track. That track was superseded
2026-09-14 by the decision to build FUSE as a new, separate repository (project memory:
`fuse-plan.md` "SUPERSEDES all earlier plan documents in this project"), and none of those five
documents exist or are referenced anywhere in the FUSE tree or in this delegation's actual "Current
Task" body. Treated as stale boilerplate carried over from an earlier phase of this project and not
produced. All work below answers the "Current Task" body only.

## Addendum — host-reported tier-2 findings, 2026-09-20

After this report was first filed, the lead relayed two problems from a real tier-2 host configure
(`-DFUSE_ENABLE_MOAB=ON -DFUSE_ENABLE_MPI=ON`, MOAB rebuilt against gcc 16), invisible at tier 0/1
where MOAB is off. Both reproduced directly in this sandbox (same filesystem, same MOAB/MPICH
installs the host used) and are fixed below as N-build-tools-docs-9 (BLOCKER) and -10 (MINOR).

**N-build-tools-docs-9 (BLOCKER).** `install(EXPORT fuse-coupler-targets ...)` failed at CMake's
Generate step: `fuse-coupler-backend` links `fuse::moab` PUBLIC, and `fuse-moab` — a plain
`add_library(... INTERFACE)` this project defines itself in `cmake/FuseDependencies.cmake`, wrapping
MOAB's old-style config which exports no imported target of its own — was in no export set. Chose
the **export-and-resolve** shape over the BUILD_INTERFACE-only alternative, because `moab_backend.hpp`
is a public header of `fuse::backend` that itself reaches MOAB types: a consumer including it needs
MOAB's include path too, not just at FUSE's own build time. Fix: `fuse-moab` is now
`add_library(... INTERFACE IMPORTED GLOBAL)` — the same exemption from the export-set check that
`Kokkos::kokkos`/`fmt::fmt`/`MPI::MPI_CXX` already get for being IMPORTED targets from real
Find/Config modules — and `cmake/fuse-couplerConfig.cmake.in` now recreates the identical wrapper for
a consumer, using the exact resolved include dirs / libraries / library dirs this build found (baked
in via `configure_package_config_file`'s `@VAR@` substitution, since MOAB's own config gives nothing
portable to `find_dependency()` against — no worse than MOAB's own non-relocatable config). Also adds
`find_dependency(MPI COMPONENTS CXX)` for a consumer, matching the same PUBLIC-link requirement.

Verified end-to-end, not just at configure: reproduced the reported failure at tier 2 in this
sandbox; fixed it; reconfigured tier 1 and tier 2 clean (isolating `-DFUSE_ENABLE_TESTS=OFF` from an
unrelated concurrent edit to `num/convergence_test.cpp` blocking the test-registration audit — not
touched by this fix, confirmed by `git diff --stat` showing my changes confined to the two `cmake/`
files); installed the tier-2 build to a prefix; wrote a standalone consumer project (`project(...
LANGUAGES CXX)`, `find_package(fuse-coupler REQUIRED)`, links `fuse::backend`, includes
`fuse/backend/moab_backend.hpp`) against that install; it configured, built, linked, and ran.

That consumer build caught a second, genuinely new problem this task did not anticipate: the first
draft of the Config addition used `find_dependency(MPI COMPONENTS C CXX)` (mirroring the main build's
own `find_package(MPI REQUIRED COMPONENTS C CXX)` in `FuseDependencies.cmake`), which failed FindMPI's
own check for the consumer project above with "MPI component 'C' was requested, but language C is not
enabled" — the main build's `project(fuse LANGUAGES C CXX ...)` enables C, but a realistic C++-only
consumer (like `test/install_consumer/CMakeLists.txt`, `LANGUAGES CXX`) does not, and FUSE's PUBLIC
interface only ever links `MPI::MPI_CXX`, never `MPI::MPI_C`. Corrected to `COMPONENTS CXX` before
finalizing; the consumer probe above is the corrected version and passes.

**N-build-tools-docs-10 (MINOR).** MOAB's `MOAB_LIBRARIES` is one old-style space-separated string
that repeats its `-Wl,-rpath,.../gfortran` and matching `-L` fragments — measured directly against
the real string this MOAB install produces: 44 `-Wl,-rpath,` tokens and 18 `-L` tokens, not the "~6
per target" the host report estimated by warning count (`ld` collapses same-path repeats into one
warning line listing the repeat count implicitly, not one line per repeat). Fixed where the stale-
directory check already tokenises the string: the same tokens are now deduplicated order-preserving
(exact-token match, so `-L` and `-Wl,-rpath,` are deduplicated independently and never against each
other) before `target_link_libraries(fuse-moab ...)`, dropping to 6 and 4 unique tokens respectively.
Deliberately left `-l` library-name tokens alone, per the report's own scoping (link order can matter
for those; ld's separate "ignoring duplicate libraries" warning for `-lblas`/`-lgfortran`/etc. is
expected to remain and did in the verification build). Verified by building `fuse_example_dn_heat` at
tier 2 before/after: the reported `duplicate -rpath` warning class is gone; the unrelated `-l`
warning is unchanged, as intended.

Both fixes are commit units of their own (below), each building cleanly on the units before them.

## Scope

16 of the 17 unified findings assigned to `build-tools-docs` in the pass2 review verified
CONFIRMED-FIXED; the other (R-41) is correctly NOT-FIXED, still open and owned by the backend
layer. The pass2 review found **no**
Verification row marked PARTIAL/NOT-FIXED that is mine to fix — the only NOT-FIXED row (R-41) is
explicitly out of scope. This round therefore fixes the pass2 review's **eight New findings**
(N-build-tools-docs-1 through -8), one of which (N-7) is outside my owned paths and is flagged for
the backend layer, and appends REVIEW.md §6 summarising all six layers' pass2 results.

## Commit units

```json
[
  {
    "ids": ["N-build-tools-docs-9", "N-build-tools-docs-10", "N-build-tools-docs-6"],
    "files": ["cmake/FuseDependencies.cmake", "cmake/fuse-couplerConfig.cmake.in"],
    "message": "Export the MOAB wrapper target and deduplicate its rpath/-L tokens\n\nAlso repoints two dangling docs/PLAN.md and docs/CHARTER.md citations in the same file (N-6)."
  },
  {
    "ids": ["N-build-tools-docs-6"],
    "files": ["UPSTREAMS.lock"],
    "message": "Repoint a dangling docs/PLAN.md citation\n\ndocs/PLAN.md does not exist in the repository; the sync-protocol reference now points at the tracked script that implements it."
  },
  {
    "ids": ["N-build-tools-docs-1"],
    "files": ["docs/dev/roadmap.md"],
    "message": "Correct roadmap.md's S4/S5 rows to match the tree\n\nThe MOAB backend and the lazy field pipeline had already landed and were still listed as remaining work."
  },
  {
    "ids": ["N-build-tools-docs-2", "N-build-tools-docs-3"],
    "files": ["KNOWN_ISSUES.md"],
    "message": "Fix stale finding counts and a retracted premise in KNOWN_ISSUES.md\n\nC0's header/count and Section C's mutation bullets had drifted from REVIEW.md and the evidence logs; one bullet still argued FUSE has no MOAB test, which B1/C0 elsewhere in the same file already retracted."
  },
  {
    "ids": ["N-build-tools-docs-4", "N-build-tools-docs-5"],
    "files": ["tools/check-invariants.py", "test/CMakeLists.txt"],
    "message": "Fix two REVIEW.md finding-id mis-citations\n\ncheck-invariants.py and test/CMakeLists.txt each cited the wrong R-nn for the defect their comment describes; corrected against REVIEW.md's actual ids."
  },
  {
    "ids": ["N-build-tools-docs-8"],
    "files": ["build-evidence/runtime-mutations.md"],
    "message": "Restore honest provenance on a re-verified mutation verdict\n\nThe epoch-schedule-drops-the-answer row's KILLED verdict was retyped with a bumped run date and no note of how it was reproduced; running tools/mutate.py beyond --check-anchors is outside this task's rules, so the row now states the verdict was reproduced by hand."
  },
  {
    "ids": ["review-pass2-summary"],
    "files": ["REVIEW.md"],
    "message": "Record the second-pass reviews in REVIEW.md\n\nAppend section 6 summarising all six layers' pass2 confirmed/partial/not-fixed counts and new findings, and mark section 4's wave items with each one's pass2 verdict."
  }
]
```

The first unit is the BLOCKER and goes first per the task's ordering rule; `cmake/FuseDependencies.cmake`
carries both the N-9/N-10 functional fix and (unavoidably, since a file can only be committed once)
the N-6 citation edits that happen to sit in the same file — the other two N-6 citations, in
`UPSTREAMS.lock`, are content-independent of that fix and ship as their own unit. Every remaining unit
only touches files inside my owned paths and builds cleanly on its own given the units before it (all
are independent comment/prose edits to disjoint files except the shared `REVIEW.md`/`KNOWN_ISSUES.md`
bookkeeping unit, which is last and depends on nothing landing first).

## Fixed

- **N-build-tools-docs-9** (BLOCKER) and **N-build-tools-docs-10** (MINOR). See the Addendum above
  for the full account: `fuse-moab` is now `IMPORTED`, exempting it from `install(EXPORT)`'s
  export-set check; `cmake/fuse-couplerConfig.cmake.in` reconstructs it (and resolves
  `find_dependency(MPI COMPONENTS CXX)`) for a consumer; `MOAB_LIBRARIES`' repeated `-Wl,-rpath,`/`-L`
  tokens are deduplicated order-preserving before linking. Verified with a real tier-2 configure,
  install, and a standalone downstream-consumer build/link/run against the installed package.
- **N-build-tools-docs-1** (MAJOR). `docs/dev/roadmap.md`: S5's row now says the MOAB backend has
  landed (code exists, wired into `fuse-coupler-backend`, built) and what remains is the host run
  of its suite, persisted maps, and the parallel matrix as a gate. S4's row drops "the lazy field
  pipeline" from the remaining-work list (it exists and is tested — `field.pipeline` passes) and
  keeps "field transactions" (genuinely absent — no `FieldTransaction` anywhere in the tree) and
  "the device leg" (accurate — Kokkos here has no CUDA/HIP space).
- **N-build-tools-docs-2** (MAJOR). `KNOWN_ISSUES.md` Section C: the bullet claiming "FUSE's own
  suite has no MOAB-dependent test at all" is replaced — `backend.moab_mesh` (38 cases) is FUSE's
  own MOAB-dependent test now; what it lacks is a host run (C1), which the bullet's actual point
  (don't read the parent's four skips as FUSE coverage) no longer needs a false premise to make.
- **N-build-tools-docs-3** (MINOR). `KNOWN_ISSUES.md` C0's header/count corrected to 51/11/21
  matching `REVIEW.md`. The NOT-ASSESSED count corrected from two to five, and the equivalent-
  mutation count from two to one, both against the actual `build-evidence/*.md` state — see the
  auditor correction below.
- **N-build-tools-docs-4 / -5** (MINOR). `tools/check-invariants.py:71` and
  `test/CMakeLists.txt:30` each cited `R-09`/`R-10` for the defect the *other* finding describes;
  both now cite the correct id (`R-06`, `R-07` respectively) against `REVIEW.md`'s actual rows.
- **N-build-tools-docs-6** (MINOR). Four dangling citations to `docs/PLAN.md`/`docs/CHARTER.md`
  (neither file exists in the repo, and the real task plan is deliberately gitignored, so no
  citation to it can resolve for a clone). `cmake/FuseDependencies.cmake:77,85` now point at
  `docs/user/build.md` and `docs/dev/conventions.md`, the actual current homes of the content
  (verified by grep — both contain the cited sentences). `cmake/FuseDependencies.cmake:1` and
  `UPSTREAMS.lock:1` drop the dangling citation (`UPSTREAMS.lock:1` now points at
  `tools/sync-upstream.py`, a tracked file that implements the protocol it names).
- **N-build-tools-docs-8** (IMPROVEMENT). `build-evidence/runtime-mutations.md`'s
  `epoch-schedule-drops-the-answer` row now states explicitly that its KILLED verdict was
  reproduced by hand 2026-09-20 in an isolated scratch copy, not by a fresh `--write-log` run —
  this task's rules permit only `tools/mutate.py --check-anchors`, so the tool-regenerated path the
  finding names as "intended" was not available; the fallback the finding itself offers was used
  instead.
- **REVIEW.md §6** appended: a per-layer table of confirmed-fixed/partial/not-fixed counts and new
  findings (ids + severities) built from all six `pass2/re-*.md` reports, plus wave-item status
  annotations in §4 citing each item's pass2 verdict.

**Post-fix, self-caught correction.** My first pass at N-build-tools-docs-3's KNOWN_ISSUES.md
rewrite attributed the two `runtime-mutations.md` NOT-ASSESSED mutations to `R-09/R-10`
"respectively" and the three `transport-mutations.md` ones to `R-01/R-02/R-30`. A fresh-context
review caught that this was fabricated: both `runtime-mutations.md` rows are tagged `R-10` in their
own description text (not a R-09/R-10 split — `R-09` is `step_period()` in `runner.cpp`, a
different function from either mutation, both of which are in `iteration.cpp`), and only one of the
three `transport-mutations.md` rows (`build-plan-skips-the-components-agreement`, tagged `R-30`)
carries an R-id at all — the other two (`peer-plan-skips-the-numbering-check`,
`peer-plan-drops-the-local-copies`) have none, and R-01/R-02 are backend/MOAB findings unrelated to
the transport layer. Corrected to state exactly what each row's own text says, with no invented
finding linkage; re-verified with `grep -n "R-[0-9]"` directly against both evidence files (commit
unit above already reflects the corrected text, not the original mistake).

## Rejected

None — no Verification row assigned to me needed rejecting; 16 of 17 were CONFIRMED-FIXED and the
remaining one (R-41) is correctly still open.

## Needs another owner

- **N-build-tools-docs-7** (MINOR). `test/backend/moab_mesh_test.cpp:5` cites `R-09` for the same
  MOAB-test-compiles-to-a-skip-stub defect (should be `R-06`) — identical mistake to N-4/N-5, same
  origin commit `4a2ad6e`. `test/backend/**` belongs to the backend layer's owned paths, not mine;
  flagged here (as the pass2 review itself did) rather than fixed.

## Needs a host run

The N-9/N-10 fix (tier-2 configure, install, and downstream-consumer build/link/run) was fully
verified directly in this sandbox against the real MOAB/MPICH installs — no host run needed for it.
The rest of this round's fixes are documentation, comments, and evidence-file text, unaffected by
tier. (Unrelated to this round: `R-01`/`R-02`/`R-41`/the MOAB test suite's execution remain host-only
per the existing REVIEW.md record; not reopened or re-verified here.)

## Deviation decisions

- **Desired-Final-Outputs mismatch**, described above: proceeded on the "Current Task" body, not
  the stale convergence-track output list.
- **N-build-tools-docs-8's "intended path" not taken.** The finding's preferred fix is a real
  `tools/mutate.py --area runtime --write-log` run; this task's rules permit only
  `--check-anchors`. Took the finding's own stated fallback (a hand-verification note) instead of
  running the full tool.
- **One MINOR (N-7) and zero MAJOR/BLOCKER left for another owner** — see above.
- **Tier-1/2 configure verification for N-9/N-10 used `-DFUSE_ENABLE_TESTS=OFF`.** A full-tests
  configure at both tiers failed on `num/convergence_test.cpp` being registered nowhere in
  `test/CMakeLists.txt` — a concurrent fixer's in-flight edit (R-49/N-num-linalg-field-1's move of
  that test from the num to the model layer), not mine: `git diff --stat` at the time showed my
  changes confined to `cmake/FuseDependencies.cmake` and `cmake/fuse-couplerConfig.cmake.in`.
  Isolated with the cache flag rather than touching `test/CMakeLists.txt` outside my own section.
- **ctest observed one failure outside my scope.** A full tier-0 `ctest -j4` run (89 test
  registrations, up from the pass2 review's 71 as other fixers landed work) showed
  `schedule.acyclicity_sweep` FAILED. `git status --porcelain` at the time showed
  `test/schedule/acyclicity_sweep_test.cpp` and other ir-schedule/base-model/num-linalg-field files
  uncommitted and mid-edit by concurrent fixers sharing this tree; none of my own changed files
  (`test/CMakeLists.txt`'s one-line diff, confirmed by `git diff --stat`) touch that test or its
  layer. Not investigated further — outside owned paths and coincident with another fixer's
  in-flight edit, not a regression introduced here.
- **Gates and build, all re-run directly, not asserted:** `check-layering.py` 0 violations,
  `check-invariants.py` 0 violations, `verify-import.py` 693 files/0 problems,
  `mutate.py --check-anchors` 89 valid/0 stale (grew from 86 as other fixers added mutations),
  `sync-upstream.py`'s `classify()` still returns 50/15 superseded paths for soma/karma,
  `sphinx-build -W --keep-going` 0 warnings (re-run after the auditor correction too), tier-0
  `cmake --build` clean, `schedule.buffers`/`schedule.slab`/`install.consumer_find_package` all
  PASS. Tier 2 (MOAB) and any MPI-launched test remain out of reach in this sandbox, unchanged from
  every prior review in this project.
