# Lane report — build-tools (fix round 2026-10-12/13)

Lane head at handback: `acbbcc05d587` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- check-parallel-matrix.py was NOT added to the CI gates job per the task's own conditional (no exit-code/flag distinguishes its by-design gap from a broken claim); recorded in notes_for_lead instead of as a code change.
- Did not widen check-invariants.py's FEATURE_MACRO (config.hpp-dependency) regex to a bare ENABLE_* as the review's literal text suggested, because doing so flags a string-literal CMake-option mention in an unowned file (examples/brusselator/species_core.hpp) as a macro use; widened FEATURE_PRESENCE_TEST (the actual ifdef-misuse rule) instead, which covers the real bug class safely.
- No higgs build, no compiler, no MPI+MOAB ctest run performed in this lane, per the lane brief's explicit assignment of CMake-change verification (BC-4, BC-12, BC-14) to the lead's integration configure.
- BC-2 was found already fixed at the lane's starting commit, before any work in this lane; verified but not re-fixed.

## Findings

### BC-3 — fixed

Evidence: tools/mutate.py --check-anchors required --build-dir via argparse even though the --check-anchors branch never reads it; made --build-dir optional when --check-anchors is given (ap.error() otherwise). Before: `python3 tools/mutate.py --check-anchors` -> argparse error, exit 2. After: `mutation anchors: 308 valid, 0 stale, of 308`, exit 0 -- added to the CI gates job. invariants.md I9 and testing.md's CI-legs list rewritten to say exactly what is now automated (the staleness check) vs. not (running the mutations and confirming KILLED, still done on the owner's machine). Verified by reading + local gate run; no CI runner available in this sandbox to execute the workflow itself.

Record text: I9 enforcement note (invariants.md, landed): 'tools/mutate.py --check-anchors' (static, no build) now runs in the gates CI job and catches a drifted mutation anchor; running the mutations themselves and confirming KILLED is still a pre-acceptance step on the owner's machine, not a CI leg.

### check-parallel-matrix-ci-gate — deferred

Evidence: Ran `python3 tools/check-parallel-matrix.py` read-only: it reports '70 claim(s) verified, 0 broken, 1 gap(s), of 23 cells', exit 1 -- the one gap (A-NODAL-HT-PARALLEL) is the recorded, by-design red per KNOWN_ISSUES D8/D4. Read the script in full: `return 1 if (problems or gaps) else 0` -- a broken claim and a by-design gap are distinguished only in the printed MATRIX-BROKEN/MATRIX-GAP text, never in the exit code, and there is no flag to select 'fail only on broken'. Per the task's own conditional ('ONLY if ... via an exit code or flag'), did not add this script to the CI gates job: doing so today would make that leg permanently red for a reason CI cannot distinguish from an actual regression, repeating the BC-1 lesson (a permanently-red leg teaches readers to ignore it).

Record text: check-parallel-matrix.py has no exit-code/flag distinction between a by-design gap and a broken claim (verified by reading the script, 2026-10-12); adding it to CI gates is deferred until it gains one (e.g. an --allow-known-gaps flag or a separate gap-only exit code).

### BC-2 — not-a-defect

Evidence: Already fixed before this lane started: the lane's starting commit e6a92e4 is itself titled 'Re-anchor the three mutation records the gcd128 and det_dot changes moved' and its message states 'mutate.py --check-anchors reports 308/308'. Confirmed by running `python3 tools/mutate.py --check-anchors` at HEAD before any of this lane's own commits: 308 valid, 0 stale, of 308, exit 0. Not fixed by this lane; flagged because it is a precondition for BC-3's CI-gates addition.

Record text: BC-2 (3 stale mutation anchors) was already re-anchored prior to this review round, at commit e6a92e4 -- verified, not re-fixed.

### BC-9 — fixed

Evidence: check-layering.py silently `continue`d on any include/fuse/<dir> not in LAYERS. Negative self-test: created include/fuse/python/_scratch_bc9.hpp (not committed) -> `check-layering: 1 violation(s)`, exit 1, naming the unranked directory; removed it -> `check-layering: 0 violation(s)`, exit 0, matching the real tree.

### BC-11 — fixed

Evidence: The include-direction walk only ever covered include/fuse/*.h*; extended it to src/**/*.cpp against the same rank table. Negative self-test: created src/base/_scratch_bc11.cpp (not committed) with `#include "fuse/runtime/runner.hpp"` -> `check-layering: 1 violation(s)` naming 'base includes from runtime, which is above it', exit 1; removed it -> 0 violations, matching the real tree.

### BC-10 — fixed

Evidence: Widened the #ifdef/defined() misuse regex (FEATURE_PRESENCE_TEST) to FUSE_CONNECT_*_ENABLED/FUSE_CONNECT_MPMD plus a generic ENABLE_[A-Z0-9_]+, with a new exemption for the one legitimate value-testing idiom already in the tree (`#if defined(X) && X`, connect.hpp:57 and connect_test.cpp:280). Verified safe by grepping every `defined(FUSE_` / `#ifdef FUSE_` / `#ifndef FUSE_` occurrence in the tree before committing (6 hits total; 2 exempted guards, 4 unrelated macros). Added tools/src to the feature-macro scan (not as library_code, matching test/examples' treatment, since its two generators are one-shot golden/baseline writers, not code a consumer links against). Negative self-tests: a scratch tools/src/*.cpp with `#ifdef FUSE_HAVE_MOAB` and a scratch src/base/*.cpp with `#if defined(FUSE_CONNECT_ACCEPT_ENABLED)` / `#ifdef FUSE_CONNECT_MPMD` (neither committed) were each caught (2 violations each), then cleared to 0 on removal.

Record text: Deviated from the review's literal suggested fix for the OTHER regex (FEATURE_MACRO, the config.hpp-dependency check): did NOT widen it to a bare ENABLE_[A-Z0-9_]+, because doing so reads CMake option names inside error-message string literals (e.g. examples/brusselator/species_core.hpp's "...FUSE_ENABLE_MPI?") as a macro use and wrongly demands a config.hpp include. Verified by a full-tree simulation before committing; see commit message for the reasoning, preserved as a code comment in tools/check-invariants.py.

### TD-8 — fixed

Evidence: Same commit and verification as BC-10 (the two findings describe the same regex gap from the docs side and the build-tools side).

### BC-4 — fixed

Evidence: Verified by grep before removing: zero includes of fmtlog under include/ or src/; zero callers of fuse_resolve_logging(); zero mentions of fmtlog in UPSTREAMS.lock or THIRD-PARTY-NOTICES.md. Removed the FetchContent_Declare(fmtlog ...), the FUSE_FMTLOG_TAG variable, and the fuse_resolve_logging() macro. Rewrote conventions.md's 'Output and logging' section (no process-wide logger exists; the three constraints are reframed as what any future logger would have to meet) and fuse_test_main.cpp's MPI_THREAD_MULTIPLE comment (no longer attributes the request to fmtlog's nonexistent polling thread; states the real, defensive rationale -- verified no std::thread/pthread usage exists anywhere in include/src/test/support that would independently justify MULTIPLE).

### BC-6 — fixed

Evidence: I6 claimed 'Release and sanitizer legs are mandatory ... Enforced by ... the CI legs themselves'; verified only the Release leg exists (tier0's Debug/Release matrix in ci.yml) and no ASan/UBSan leg exists anywhere in the 110-line workflow. Rewrote I6 to say exactly this, matching testing.md's own 'Planned' entry for the sanitizer leg (S1 exit).

### BC-12 — fixed

Evidence: The installed libfuse-abi RPATH carried yaml-cpp and Kokkos but relied on fmt sharing yaml-cpp's directory by machine coincidence. Verified fuse_require(fmt VERSION 9) is unconditional (always resolves fmt::fmt), so added $<TARGET_FILE_DIR:fmt::fmt> to _fuse_abi_install_rpath and corrected the comment that documented the coincidence as load-bearing. Not build-verified in this lane (no compiler here); the lane brief assigns CMake-change verification to the lead's integration configure.

### BC-14 — fixed

Evidence: Moved MOAB_DIR and FUSE_MOAB_MESHFILES from the 'all' configure preset's cacheVariables into _machine's, matching _machine's own documented 'edit this ONE preset on another machine' design. Verified both are inert for tiers that don't enable MOAB: find_package(MOAB)'s consumption of MOAB_DIR sits inside `if(FUSE_ENABLE_MOAB)` in FuseDependencies.cmake, so carrying the variable unconditionally in _machine changes nothing for tier0/tier1. Added "jobs": 12 to the 'all' build preset (the only one missing it). JSON validated with json.load().

### BC-5 — fixed

Evidence: build.md's tier3 row promised an unlocked solver plugin and physics-kit component; CMakePresets.json's own tier3 description already says 'configures today; the solver plugin is not yet written' and README's Status section says S8/physics-kit has not started. Added the same disclosure to build.md's tier3 row and to the pybind11/Python orthogonal-option line (README already discloses FUSE_ENABLE_PYTHON has nothing behind it).

### BC-8 — fixed

Evidence: README said 'S0 through S6 are complete'; UPSTREAMS.lock's [soma] block has retire_at = "S6" and retired = "" with an explicit 'NOT yet superseded' comment naming the remaining paths. Rewrote the status line to separate S6's conformance case (closed 2026-10-07) from its subtree-retirement exit condition (open, per UPSTREAMS.lock).

### TD-2 — fixed

Evidence: testing.md named test/golden/waist-v1.bin as the frozen canonical descriptor and a Descriptor.MatchesTheFrozenWaistV1Bytes test that does not exist. Verified against test/model/descriptor_test.cpp and descriptor.hpp: kGraphVersion is now 13 (the review's own cited v11 was itself already stale by the time of this lane -- the tree had moved past it), the real test is Descriptor.MatchesTheFrozenWaistBytes against waist-v13.bin, and the superseded-but-kept-and-refused set is exactly {v1, v4..v12} (v2/v3 were never given their own golden file -- verified by both `ls test/golden/` and the exact array literal in Descriptor.SupersededFrozenBytesAreRefusedByThisReader). Rewrote the section and bumped the regeneration example from v2 to v14.

### TD-4 — fixed

Evidence: fsi-channel.md and examples.md marked the example plainly 'Verified' with no mention of its FUSE_ENABLE_CUSPARSE-conditional state. The review cited KNOWN D34/D35, which per the lane brief's own warning about pre-rebase numbering are now a DIFFERENT, already-fixed pair of entries (MPI_APPNUM, window-bystander deadlock) -- verified by grep against the current KNOWN_ISSUES.md. The live entries are D36 (both example.fsi_channel ctest rows DISABLED under the flag -- marginal IQN-ILS stability near the beam's first natural period) and D37 (fsi_energy_diag's one 480-period case GTEST_SKIPped -- cusolverRf is not run-to-run bitwise reproducible), confirmed against examples/CMakeLists.txt:534-538 and the skip message in fsi_energy_diag_test.cpp:950. Also corrected along the way: the fluid solver's SparseLuBackend is configuration-dependent (Eigen SparseLU vs. cusolverRf), but the beam solver (beam_solver.hpp) is always plain Eigen::SparseLU regardless of the flag -- the original draft of this fix incorrectly described both solvers as configuration-dependent and was corrected before committing.

### BM-10 — fixed

Evidence: invariants.md's E-UNITS-SCALE bullet said a scale-only mismatch 'is refused' pending S2.4. Verified against connect.hpp (ConnectPolicy::allow_automatic_unit_scale, default true) and validate.hpp (check_channel's Unit-stage diagnostics: a DimensionMismatch is E-UNITS, an inferred UnitScale step is reported as the I-UNIT-SCALE-INSERTED warning, and only an offset change or a forbidding policy reaches the default: case that emits E-UNITS-SCALE). S2.4 has landed; rewrote the bullet to describe insertion-with-warning as the default path and refusal as the exception, not the reverse.

## Notes for the lead

All 15 files touched (.github/workflows/ci.yml, CMakePresets.json, README.md, cmake/FuseDependencies.cmake, docs/dev/conventions.md, docs/dev/invariants.md, docs/dev/testing.md, docs/user/build.md, docs/user/examples.md, docs/user/fsi-channel.md, src/CMakeLists.txt, test/support/fuse_test_main.cpp, tools/check-invariants.py, tools/check-layering.py, tools/mutate.py) are inside this lane's owned paths; src/CMakeLists.txt and test/support/fuse_test_main.cpp edits are each scoped to exactly the one block the lane brief names (install-RPATH; the THREAD_MULTIPLE comment). NOT done in this lane, by the lane brief's own design: no higgs build, no MPI+MOAB ctest run, no compiler available here at all. BC-4 (CMakeLists.txt/FuseDependencies.cmake), BC-12 (RPATH) and BC-14 (CMakePresets.json) are CMake/text changes verified by reading and by grep, not by configuring -- these three need the lead's integration configure to confirm they still produce a working build, per the lane brief's explicit assignment of that verification to the lead. check-parallel-matrix.py was evaluated but NOT added to the CI gates job: it has no exit code or flag that distinguishes its one by-design gap (A-NODAL-HT-PARALLEL, KNOWN D8/D4) from an actual broken claim -- both read exit 1, differing only in the MATRIX-BROKEN vs. MATRIX-GAP line prefix. Adding it today would make the gates leg permanently red for a reason CI cannot tell apart from a regression. Recommend a future --allow-known-gaps flag (or a separate gap-only exit code) before adding it. BC-2 (3 stale mutation anchors) was already fixed before this lane started -- the lane's starting commit e6a92e4 is itself the re-anchoring commit. Flagged for the disposition table's completeness since it is a precondition of BC-3. Sphinx docs build (docs conda env, sphinx-build -W --keep-going -b html docs docs/_build/html): build succeeded with exactly one warning both before and after this lane's commits -- docs/dev/runner-device-dispatch.md is not included in any toctree. That page and docs/dev/index.md (which only cross-links it) are outside this lane's owned paths and were never touched by any commit here; the warning predates this round (introduced by commit 254f934, well before e6a92e4). Under -W (matching ci.yml's docs job exactly) this one warning fails the build, so the docs CI leg is red independent of this lane's fixes -- worth a finding for whichever lane owns docs/dev/index.md's toctree. KNOWN_ISSUES.md D-number caution confirmed in practice, not just in principle: TD-4's cited D34/D35 turned out to be a different, already-closed pair of entries by the time this lane ran (the lane brief's own warning about this proved necessary).
