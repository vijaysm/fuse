# Fixing build-tools-docs findings (BT-\*), FUSE review 2026-09-19

Branch `review-2026-09-19-fixes`, HEAD at start `ffea077`. All work confined to my OWNED PATHS;
`git add`/`commit`/`checkout`/`stash`/`reset` never run.

Deviations from what a fully-clean-tree run would show: the tree was shared with five other
fixers editing concurrently the whole time. `python3 tools/mutate.py --check-anchors` and a full
`ctest` therefore surface a few failures that are **not mine** — see "Needs another owner" and the
notes under each unit. Everything reported as passing was re-verified after those unrelated
failures appeared, isolating my own targets/tests to be sure they were unaffected. Tier 1/2/3 (MPI,
MOAB, PETSc) cannot be exercised in this sandbox (`MPI_Init` fails here) — every claim about those
tiers is provisional / doc-only and flagged under "Needs a host run".

## Commit units

```json
[
  {
    "ids": ["R-05"],
    "files": [".github/workflows/ci.yml"],
    "message": "Fix tier0 CI's kokkoskernels linalg leg to builtin\n\nFUSE_LINALG_BACKEND only implements builtin and eigen, so the kokkoskernels leg FATAL_ERRORed on every run (R-05)."
  },
  {
    "ids": ["R-21"],
    "files": ["build-evidence/mutations.toml", "build-evidence/runtime-mutations.md"],
    "message": "Rewrite the stale epoch-schedule-drops-the-answer mutation anchor\n\nThe anchor targeted a flat result_send_to loop the binomial-tree rewrite replaced with result_send_count, so text.count(old)==0 made the KILLED verdict unverifiable (R-21)."
  },
  {
    "ids": ["R-23"],
    "files": ["KNOWN_ISSUES.md"],
    "message": "Close KNOWN_ISSUES B1 as stale, superseded by C1\n\nMOAB backend headers exist at HEAD so B1's quoted compile error cannot reproduce; also fixes C0's mis-citation of R-09 where it meant R-06 (R-23)."
  },
  {
    "ids": ["R-24"],
    "files": ["docs/user/build.md"],
    "message": "Document the builtin and eigen linalg backends, drop kokkoskernels\n\nbuild.md advertised a kokkoskernels backend that was never implemented and omitted the real builtin default, and claimed tier 0 already ships both C ABIs and a Fortran shim (R-24)."
  },
  {
    "ids": ["R-25"],
    "files": [
      "src/CMakeLists.txt",
      "cmake/fuse-couplerConfig.cmake.in",
      "test/CMakeLists.txt",
      "test/install_consumer/CMakeLists.txt",
      "test/install_consumer/main.cpp",
      "test/install_consumer/run-smoke-test.cmake"
    ],
    "message": "Add install()/export() so find_package(fuse-coupler) works\n\nEvery layer target gets EXPORT_NAME plus an install rule, a generated Config.cmake resolves Kokkos/fmt, and a standalone consumer project is configured, built and run against the install prefix as a new ctest entry (R-25)."
  },
  {
    "ids": ["R-37"],
    "files": ["CMakeLists.txt"],
    "message": "Compute the CMakeLists.txt status banner's layer list from targets\n\nThe banner's literal layer list and stage number had drifted four stages and eight layers behind the actual build (R-37)."
  },
  {
    "ids": ["R-38", "R-39"],
    "files": ["tools/check-layering.py", "docs/dev/layering.md"],
    "message": "Make check-layering's legacy checks comment-aware, fix the layering diagram\n\ncheck_legacy() flagged prose naming a legacy identifier as a violation, and layering.md's diagram omitted linalg/field and still named the renamed abi layer interface (R-38, R-39)."
  },
  {
    "ids": ["R-40"],
    "files": ["CMakePresets.json"],
    "message": "Point CMakePresets.json at the build-<preset> directory convention\n\nbinaryDir used build/<preset>, a third convention alongside the build-*/ every other doc, script and log in the tree already uses (R-40)."
  },
  {
    "ids": ["R-42"],
    "files": ["cmake/FuseDependencies.cmake"],
    "message": "Remove the dead KokkosKernels FetchContent declaration\n\nNever find_package'd or FetchContent_MakeAvailable'd anywhere -- the same abandoned kokkoskernels backend R-05 and R-24 also clean up (R-42)."
  },
  {
    "ids": ["R-31", "R-51"],
    "files": ["REVIEW.md"],
    "message": "Update REVIEW.md for the build-tools-docs fixes\n\nReclassifies R-31 MAJOR to MINOR (a stale name in prose, not a live #ifdef) and adds R-51 for comm_graph_test.cpp, then marks wave 0's four gate additions and this session's fixes DONE."
  }
]
```

## Fixed

**R-05 (BT-F1, BLOCKER)** — `ci.yml`'s `tier0` matrix `linalg: [eigen, kokkoskernels]` FATAL_ERRORed
on every `kokkoskernels` run (`FUSE_LINALG_BACKEND`'s `STRINGS` property is only `builtin|eigen`).
Changed to `[eigen, builtin]`. Verified by reading `cmake/FuseDependencies.cmake:62-76`'s guard
side by side with the new matrix value — no CI runner available here, but the mismatch was, and now
is not, purely textual.

**R-21 (BT-F5, MAJOR)** — the `epoch-schedule-drops-the-answer` mutation anchor (`old =
"    for (int k = 1; k < participants; ++k) s.result_send_to.push_back(k);"`) matched the current
`include/fuse/runtime/epoch_schedule.hpp` zero times (confirmed:
`text.count(old) == 0` before my fix). Rewrote it to target the current binomial-tree
implementation's `s.result_send_count = participants - 1;` → `= 0;` (participant 0 sends its answer
to nobody — the same defect, current code shape). Verified with the actual tool, not by eye:
`python3 tools/mutate.py --check-anchors --build-dir <build-t0>` now reports my anchor valid (only
two *other*, unrelated anchors are stale — see "Needs another owner"). Added a note to
`build-evidence/runtime-mutations.md` explaining the row's KILLED verdict is the PRIOR anchor's,
not yet re-verified against the new one — `tools/mutate.py --area runtime --write-log` needs a
clean tree to regenerate it, which I could not run (hard rule: never run `tools/mutate.py` on this
shared dirty tree beyond `--check-anchors`).

**R-23 (BT-F7, MAJOR)** — `KNOWN_ISSUES.md` B1 claimed a MOAB compile error that cannot reproduce at
HEAD (`include/fuse/backend/moab_backend.hpp` and `src/backend/moab_backend.cpp` both exist and are
wired into `fuse-coupler-backend`). Struck through the heading, marked `CLOSED 2026-09-18,
superseded by C1`, kept the original text in a `<details>` block per the file's own convention
(compare A2/A3/B1a). Also fixed a citation bug in C0 I found while doing this: C0 attributed the
MOAB-test-compiles-to-nothing description to `R-09` (`step_period()`'s collective-agreement
BLOCKER, an unrelated finding); the description is actually R-06's. Corrected the citation and
noted R-06's fix has landed (verified separately — see "Fixed elsewhere").

**R-24 (BT-F8, MAJOR)** — `docs/user/build.md` documented a `kokkoskernels` linalg backend that
`FuseDependencies.cmake` never implements, omitted the real `builtin` default, and claimed tier 0
ships "both C ABIs, the Fortran shim" (neither exists: `include/fuse/abi/` has no headers,
`FUSE_ENABLE_FORTRAN` is declared and never consumed). Rewrote the tier-0 row and the backend
section to document `builtin`/`eigen` only, and dropped the false ABI/Fortran claim rather than
retitle it "planned" — `roadmap.md` already carries that.

**R-25 (BT-F9, MAJOR)** — no `install()`/`export()` existed anywhere; `find_package(fuse-coupler)`
could not work. Added, in `src/CMakeLists.txt`: `include(GNUInstallDirs)`, `install(TARGETS
<all 9 layer targets> EXPORT fuse-coupler-targets ...)`, a header install, `install(EXPORT ...
NAMESPACE fuse::)`, and `configure_package_config_file`/`write_basic_package_version_file` against
a new `cmake/fuse-couplerConfig.cmake.in`. Two things the naive version of this fix gets wrong, both
hit and fixed here:
  1. Every layer target needed an explicit `EXPORT_NAME` (`base`, `model`, …) — without it,
     `NAMESPACE fuse::` prefixes onto the TARGET's own name (`fuse-coupler-base`), giving consumers
     `fuse::fuse-coupler-base` instead of `fuse::base`, silently breaking the alias convention
     `docs/dev/conventions.md` promises.
  2. `fuse-coupler-model`'s and `fuse-coupler-backend`'s PRIVATE links to `yaml-cpp::yaml-cpp` and
     `r3d::r3d` (both fetched, not installed, on this machine) made `install(EXPORT)` refuse with
     "requires target … that is not in any export set" — CMake requires PRIVATE dependencies of a
     STATIC library to be resolvable too. Wrapped both in `$<BUILD_LOCAL_INTERFACE:...>` so they
     stay real for this build without needing to be exportable; documented in both the source
     comment and `fuse-couplerConfig.cmake.in` that a consumer of `fuse::model`/`fuse::backend`
     must supply yaml-cpp/r3d itself. `fuse::base` (Kokkos, fmt, both PUBLIC and both real
     `find_package`-resolvable packages on this machine) has no such gap.
  Config.cmake.in also carries the same Apple/OpenMP_ROOT workaround
  `fuse_resolve_dependencies()` already has, since a consumer's `find_dependency(Kokkos)` hits the
  identical Apple-clang-ships-no-OpenMP failure the main build works around.

  **Verified end to end, locally** (macOS, cmake 4.3.2, the `_machine` preset's own prefixes):
  built all 9 layer libraries, `cmake --install` into a scratch prefix, configured+built+ran a new
  standalone `test/install_consumer/` project (links only `fuse::base`, deliberately — see the
  PRIVATE-dependency note above) against that prefix via plain `find_package(fuse-coupler CONFIG
  REQUIRED)`. Wired the same three steps into `test/CMakeLists.txt` as a new
  `install.consumer_find_package` ctest entry (own `# ---- install` section, driven by
  `test/install_consumer/run-smoke-test.cmake` since ctest's `COMMAND` runs one program and this
  check is three `cmake` invocations). **This ctest entry passed** in the full suite run (`ctest -R
  install.consumer_find_package`, label `install`, 3.05s).
  Scope note: this makes `find_package(fuse-coupler)` work at all — the R-25 finding's ask — for a
  consumer of any layer's PUBLIC interface. A consumer wanting `fuse::model`/`fuse::backend` also
  needs yaml-cpp/r3d itself (documented, not solved); a consumer wanting `fuse::transport` (MPI) or
  MOAB-enabled `fuse::backend` needs `find_dependency(MPI)`/MOAB added to `Config.cmake.in` too —
  not implemented, since I could not test tier1/2/3 here (see "Needs a host run").

**R-37 (BT-F11, MINOR)** — the top-level `CMakeLists.txt` status banner hardcoded `"stage S1"` and
`"fuse:: layers : base"`, four stages and eight layers stale. Replaced with a loop over the nine
known layer names checking `TARGET fuse::<name>` and joining the ones that exist, dropping the
unresolvable "stage" label rather than guessing at it. Verified: configure output now prints `fuse::
layers   : base, model, ir, schedule, linalg, num, transport, backend, runtime`.

**R-38 (BT-F12, MINOR)** — `check-layering.py`'s `check_legacy()` applied `LEGACY_NAMESPACE`/
`LEGACY_TYPE_PREFIX`/`LEGACY_MACRO` to raw lines, so a comment merely naming `soma::`/`KARMA_*` (a
port's rename changelog) tripped it. Added the same `COMMENT_LINE` blanking
`check-invariants.py`'s feature-macro check already uses, ahead of those three checks only (matches
the review's exact fix). Verified with a synthetic probe (not the real tree): a comment-only file
naming both legacy identifiers → 0 violations; a file with real `namespace soma`/`KARMAFoo`/
`#define KARMA_X` → 3 violations, unchanged. `check-layering.py` on the real tree stays at 0
violations.

**R-39 (BT-F13, MINOR)** — `docs/dev/layering.md`'s diagram omitted `linalg` and `field`
(real, separately-ranked layers in `tools/check-layering.py`'s `LAYERS`), and its top row was named
`interface`, matching no directory (the gate's actual name is `abi`). Added both missing rows with
a `(*)` footnote pointing at the gate script's own rank-order comment, and renamed `interface` →
`abi`.

**R-40 (BT-F14, MINOR)** — `CMakePresets.json`'s `binaryDir` used `build/<preset>`, a third
convention alongside the `build-*/` every doc/log in the tree already uses (and alongside the
`build-tier1`/`build`/`build-tier2` directories already on disk from prior sessions). Changed to
`build-${presetName}` — `cmake --preset tier1` now binds to the same `build-tier1/` the existing
on-disk build and every doc already name.

**R-42 (BT-F16, MINOR)** — `cmake/FuseDependencies.cmake` declared `FetchContent_Declare(KokkosKernels
...)` and pinned `FUSE_KOKKOSKERNELS_TAG`, neither ever consumed by `find_package`/
`FetchContent_MakeAvailable`. Removed both; left a comment pointing at R-24/R-42 for why, and what
re-adding it correctly would need.

**R-51 (added while fixing, not a numbered review finding at HEAD)** — running
`tools/mutate.py --check-anchors` surfaced that R-06's exact twin (only `<gtest/gtest.h>` preceded
`test/parallel/comm_graph_test.cpp`'s `#if FUSE_HAVE_MPI`, so its 28 CommGraph/Crystal-Router cases
compiled to nothing) was **already fixed in wave 0**, same commit (`4a2ad6e`) as the invariant-gate
extension. Recorded it in REVIEW.md as R-51, BLOCKER, DONE, so the finding count and history are
accurate.

**R-31 reclassification (BT-F4-adjacent, in REVIEW.md only)** — per the lead's instruction, re-read
`include/fuse/transport/transport.hpp:255` and `src/transport/lowfive_transport.cpp:66`: both now
read `FUSE_ENABLE_LOWFIVE`, not `SOMA_ENABLE_LOWFIVE`, and neither occurrence is a conditional — one
is a doc comment, one is an error-message string. Reclassified R-31 MAJOR→MINOR in REVIEW.md,
moved its row, corrected §2.3's prose (it had claimed a live `#ifdef`), and updated the finding-count
header (50→51: R-51 added, R-31 stays counted but moves severity buckets).

**Wave 0 items marked DONE in REVIEW.md §4** (already committed before this session; verified,
not redone): extending `check-invariants.py` to `test/`/`examples/` plus the presence-test and
legacy-macro rules (`4a2ad6e`); the `test/CMakeLists.txt` registration audit + registering
`buffers`/`slab` (`b8e3f54`); `tools/mutate.py --check-anchors` (`06e882a`); the TOML lock parser
plus S4/S5.2 superseded entries (`ffea077`).

## Fixed elsewhere (verified, not touched — outside my owned paths)

- **R-06 (BT-F2, BLOCKER)** — `test/backend/moab_mesh_test.cpp` now includes `config.hpp` before
  `#if FUSE_HAVE_MOAB` (was `#ifdef FUSE_HAVE_MOAB` with no preceding include). Fixed by another
  fixer concurrently; confirmed by reading the current file.
- **R-36 (BT-F10, MINOR)** — `src/backend/intx_weights.cpp:36` now reads
  `#if defined(MOAB_HAVE_MPI) && FUSE_HAVE_MPI` (was `defined(FUSE_HAVE_MPI)`, a presence test on a
  `#cmakedefine01` macro). Fixed by another fixer concurrently; confirmed by reading the current
  file.
- **R-07 (BT-F3, BLOCKER)**, **R-20 (BT-F4, MAJOR)**, **R-22 (BT-F6, MAJOR)**,
  **R-47 (BT-F17, IMPROVEMENT)** — all four wave-0 items above; see REVIEW.md §4 update.

## Rejected

None. Every finding in my list was confirmed either still live and fixed in this session, already
fixed in wave 0, or already fixed by another fixer's concurrent work (see above) — none were wrong
as written.

## Needs another owner

- **R-41 (BT-F15, MINOR)** — `test/backend/moab_mesh_test.cpp:614`'s `#ifdef FUSE_MOAB_TEST_MESH`
  block is still dead code: nothing in the build defines `FUSE_MOAB_TEST_MESH`. Confirmed still
  present (re-grepped the whole non-`build*` tree). Not in my owned paths (`test/backend/**`) —
  the backend fixer should either wire a `target_compile_definitions` pointing at a real MOAB
  cubed-sphere fixture, or delete the block per the review's own fallback suggestion.
- **Two mutation anchors went stale DURING this session**, discovered by
  `tools/mutate.py --check-anchors` (a tool I am allowed to run) and not part of my finding list:
  `merge-typed-at-first-writer` (`src/ir/lowering.cpp`, 0 occurrences — the ir-schedule fixer's
  live R-43 rewrite of the merge tie-break) and `mesh-accepts-a-parent-outside-the-cell-set`
  (`include/fuse/backend/native/mesh.hpp`, 0 occurrences — the backend fixer's live R-03/R-14
  device-callable-mesh rewrite). `build-evidence/mutations.toml` is nominally my owned file, but I
  did not touch either anchor: both point at code outside my owned paths that is still being
  actively rewritten by its owner, so any anchor text I wrote now could go stale again before they
  finish. Flagging for the lead / those two fixers to fix once their code settles.
- The transport/runtime/backend/model layers' own findings (R-01, R-02, R-03, R-04, R-08 through
  R-19 except where noted, R-26 through R-30, R-32 through R-35, R-43 through R-50) are outside my
  scope entirely and not touched.

## Needs a host run

- **R-06's 38 MOAB cases and R-51's 28 CommGraph cases**: both files compile again, but neither has
  actually been *executed* here — this sandbox cannot `MPI_Init` (tier 2 needs tier 1, MPI). A host
  run of `build-tier2`'s ctest is what `KNOWN_ISSUES.md` C1 already calls for; unchanged by my work
  beyond the citation fix.
- **R-21's rewritten mutation** needs `tools/mutate.py --area runtime --build-dir <dir> --write-log`
  on a CLEAN tree to produce a current KILLED/SURVIVED verdict for
  `epoch-schedule-drops-the-answer`'s new anchor. I confirmed the anchor text is valid
  (`--check-anchors`, no build) but could not run the full mutation (hard rule: never run
  `tools/mutate.py` beyond `--check-anchors` on this shared dirty tree).
- **R-25's install/export**, tier 0 only: verified on macOS/Ninja against this machine's `/opt`
  prefixes. Tier 1 (MPI)/tier 2 (MOAB) consumers are unverified — `fuse-couplerConfig.cmake.in`
  does not yet `find_dependency(MPI)`/MOAB, and I have no MPI-capable sandbox to prove or disprove
  that gap. A host run of the `install.consumer_find_package` ctest entry (already wired into
  `tier1`/`tier2` presets, since it's unconditional in `test/CMakeLists.txt`) would surface whether
  those tiers' consumers need more `find_dependency` calls.
- **R-05's CI leg fix** (`eigen`/`builtin` matrix): the YAML edit is textual and matches
  `FUSE_LINALG_BACKEND`'s actual `STRINGS` property, but no GitHub Actions runner is available here
  to execute the workflow itself.

## Deviation decisions

None of my assigned findings were DEVIATION-severity (R-49, R-50 belong to num-linalg-field and
transport-runtime respectively). The one deviation-shaped decision I made outside a numbered
finding: R-25's fix intentionally leaves yaml-cpp/r3d/MPI/MOAB dependency resolution for
`fuse::model`/`fuse::backend`/`fuse::transport` consumers unsolved rather than attempting a
COMPONENTS-keyed `Config.cmake.in` — documented in-line as the correct scope boundary for "the
missing half was the install() calls and one Config.cmake.in" (the review's own words), not a
gap I'm hiding. Recorded for the lead to decide whether it needs its own follow-up finding.
