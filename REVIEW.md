# Implementation review — 2026-09-19, HEAD `bdf4346` (presets at `9ea2560`)

Six independent read-only reviews, one per layer group, against the same commit with the same
rubric (docs/dev/invariants.md, layering.md, conventions.md, testing.md, rank-agreement.md, porting.md).
**51 findings: 11 BLOCKER, 21 MAJOR, 15 MINOR,
2 IMPROVEMENT, 2 DEVIATION.** (R-51 added and R-31 reclassified MAJOR→MINOR while fixing wave 0,
2026-09-19 — see their rows below for why.) The detailed reviews, with
every finding's evidence and concrete fix, are in `docs/dev/reviews/2026-09-19/`; this document is the
index, the causes, and the order to fix them in.

**What "checked" means.** Four findings marked *verified here* were re-read against source by the
author of this document, independently of the reviewer. The rest carry the reviewer's file:line
citation and were not re-derived; treat each as a claim to confirm at the moment of fixing it — the
fix step for every finding begins with reading the cited lines.

---

## 1. The findings

### BLOCKER — wrong result, hang, undefined behaviour, or a test suite that cannot run

| # | Source | Location | Finding | Checked |
|---|---|---|---|---|
| R-01 | [BK-B1](docs/dev/reviews/2026-09-19/backend.md) | `src/backend/intx_weights.cpp:420-474`, `1659-1662`, `1706-1741`, `1915-1918` | Per-rank data-dependent `resolve()`/`fit_surface()`/`orient()` failures precede the collectives `build_remap_h | reviewer |
| R-02 | [BK-B2](docs/dev/reviews/2026-09-19/backend.md) | `src/backend/moab_backend.cpp:1829-1835`, `1879-1898`, `1918`, `1927` | `plan_halo_exchange` can return per-rank on inconsistent global ids between two `MPI_Alltoall(v)` calls, hangi | reviewer |
| R-03 | [BK-B3](docs/dev/reviews/2026-09-19/backend.md) | `include/fuse/backend/native/mesh.hpp:99-103`, `144-182` | `Mesh<M,S>` has no execution/memory-space parameter; `finalize()` is host-only code that indexes the same view | reviewer |
| R-04 | [BM-B1](docs/dev/reviews/2026-09-19/base-model.md) | `include/fuse/model/descriptor.hpp:359,367,371`, `src/model/yaml_io.cpp:403,411,417` | Corrupted descriptor + `to_yaml()` = out-of-bounds array read (UB) | reviewer |
| R-05 | [BT-F1](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `.github/workflows/ci.yml:57` | `tier0` CI matrix's `kokkoskernels` leg cannot configure; `FUSE_LINALG_BACKEND` has no such value | reviewer |
| R-06 | [BT-F2](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `test/backend/moab_mesh_test.cpp:3` | The entire MOAB backend test file compiles to a single skip stub, in every configuration, because its own `#if | **verified here** |
| R-07 | [BT-F3](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `test/CMakeLists.txt:97-101` | `test/schedule/buffers_test.cpp` and `test/schedule/slab_test.cpp` (33 test cases) are never registered and ha | **verified here** |
| R-08 | [NL-B1](docs/dev/reviews/2026-09-19/num-linalg-field.md) | `include/fuse/field/pipeline.hpp:96-99` | `FieldSpan::from()` rejects the exact empty-partition `FieldRef` that `FieldRegistry::bind()` was fixed to acc | reviewer |
| R-09 | [TR-F1](docs/dev/reviews/2026-09-19/transport-runtime.md) | `src/runtime/runner.cpp:2112-2118` | A failing implicit coupling group makes `step_period()` abandon the rest of the period's instruction stream, includin | **verified here** |
| R-10 | [TR-F2](docs/dev/reviews/2026-09-19/transport-runtime.md) | `src/runtime/iteration.cpp:600-639` | Iteration-region convergence is decided from a purely rank-local residual with no cross-rank agreement; ranks  | **verified here** |
| R-51 | *(added while fixing, 2026-09-19)* | `test/parallel/comm_graph_test.cpp:1-13` | Only `<gtest/gtest.h>` preceded this file's `#if FUSE_HAVE_MPI` — the same undefined-macro bug as R-06, one directory over — so its 28 CommGraph/Crystal-Router cases compiled to nothing in every configuration. **Fixed in wave 0**, commit `4a2ad6e` (the same commit that added the `test`/`examples` scan and the presence-test rule these two findings motivated). | **verified here** |

### MAJOR — reachable in normal use, or a binding rule the gates missed

| # | Source | Location | Finding | Checked |
|---|---|---|---|---|
| R-11 | [BK-M1](docs/dev/reviews/2026-09-19/backend.md) | `docs claim vs. reality`: `include/fuse/backend/moab_backend.hpp:31,65,141,242,251`; `incl | Code comments cite `test/unit/remap_parallel_test.cpp` and four sibling files as the evidence for parallel/ord | reviewer |
| R-12 | [BK-M2](docs/dev/reviews/2026-09-19/backend.md) | `src/backend/supermesh.cpp` (`inward_plane`, in the `intersect` helper) | A degenerate clipping simplex (zero-length edge) produces an always-true half-plane instead of a zero-measure  | reviewer |
| R-13 | [BK-M3](docs/dev/reviews/2026-09-19/backend.md) | `src/backend/supermesh.cpp` (`build_supermesh_weights`) | No check that per-cell coverage stays ≤ 1 + tol, unlike the MOAB path's `over_src`/`over_tgt` audit (`intx_wei | reviewer |
| R-14 | [BK-M4](docs/dev/reviews/2026-09-19/backend.md) | `include/fuse/backend/native/mesh.hpp:189-233` vs. `238-241` | `measure()`, `centroid()`, `normal()` are not `KOKKOS_INLINE_FUNCTION` and call the host-only `FUSE_ASSERT`, c | reviewer |
| R-15 | [BM-M1](docs/dev/reviews/2026-09-19/base-model.md) | `include/fuse/base/rational.hpp:210-213` | `Rational::lcm_denominator` has the exact `int64_t` overflow its sibling function was fixed for | reviewer |
| R-16 | [BM-M2](docs/dev/reviews/2026-09-19/base-model.md) | `src/base/error.cpp:53,67,69-72,199` | `error_policy()` reads `ErrorPolicy` without the lock `set_error_policy()` writes under — data race | reviewer |
| R-17 | [BM-M3](docs/dev/reviews/2026-09-19/base-model.md) | `include/fuse/model/component.hpp:38`, `src/model/yaml_io.cpp:499-513` | `RankSet`'s default constructor silently places a component on rank 0, contradicting the class's own "empty =  | reviewer |
| R-18 | [BM-M4](docs/dev/reviews/2026-09-19/base-model.md) | `src/model/yaml_io.cpp:246,264,323,378,391` | `to_yaml()` dereferences unvalidated ids via `.at()`, throwing raw `std::out_of_range` instead of the `Result< | reviewer |
| R-19 | [BM-M5](docs/dev/reviews/2026-09-19/base-model.md) | `include/fuse/base/units.hpp` (whole file), `src/model/semantics.cpp:359`, `src/model/comp | Negative `Units::scale` is never rejected; `check_units()` will auto-insert an unrequested, sign-flipping scal | reviewer |
| R-20 | [BT-F4](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `tools/sync-upstream.py:39-46` | The lock-file parser cannot read the multi-line `superseded`/`diverged` arrays `UPSTREAMS.lock` actually uses; | reviewer |
| R-21 | [BT-F5](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `build-evidence/mutations.toml:455` | The `epoch-schedule-drops-the-answer` mutation anchor no longer exists in `epoch_schedule.hpp`; the committed  | reviewer |
| R-22 | [BT-F6](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `UPSTREAMS.lock:90-95` | The four MOAB paths superseded by the S5.2 port were recorded only as a comment, never added to `[soma].supers | reviewer |
| R-23 | [BT-F7](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `KNOWN_ISSUES.md:165-176` | B1 ("Tier 2 does not build") is stale and false at HEAD, contradicted by C1 fifty lines below it in the same f | reviewer |
| R-24 | [BT-F8](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `docs/user/build.md:9,29-37` | The build doc documents a `kokkoskernels` linear-algebra backend that does not exist, omits the real `builtin` | reviewer |
| R-25 | [BT-F9](docs/dev/reviews/2026-09-19/build-tools-docs.md) | *(absence, whole tree)* | No `install()`/`export()` anywhere; `conventions.md`'s "CMake package `fuse-coupler`" cannot currently be `fin | reviewer |
| R-26 | [IR-F1](docs/dev/reviews/2026-09-19/ir-schedule.md) | `test/golden/pair_explicit.ir`, `pair_accumulate_remap_relax.ir`, `pair_iterated.ir`; `src | Reference-output IR-dump regression suite was never ported; the three checked-in reference-output files are stale (missing `Epoch` | reviewer |
| R-27 | [IR-F2](docs/dev/reviews/2026-09-19/ir-schedule.md) | `src/ir/lowering.cpp:150-190`; `include/fuse/schedule/plan.hpp:19-21`; `src/schedule/sched | "Merge as a left-fold of pairwise merges" is documented and referenced in an error message, but never implemen | reviewer |
| R-28 | [IR-F3](docs/dev/reviews/2026-09-19/ir-schedule.md) | `include/fuse/schedule/buffers.hpp:85-93`; `src/schedule/buffers.cpp:147-152` | `buffers.hpp` documents `plan_buffers()`/`try_plan_buffers()` as throwing `std::invalid_argument` / `std::out_ | reviewer |
| R-29 | [NL-A1](docs/dev/reviews/2026-09-19/num-linalg-field.md) | `include/fuse/field/field_ref.hpp:97-98` | `Layout::well_formed()` rejects valid padded/strided layouts (off-by-one-stride-unit bound) | reviewer |
| R-30 | [TR-F3](docs/dev/reviews/2026-09-19/transport-runtime.md) | `src/transport/mpi_transport.cpp:355-358` vs `842-876` | `MpiTransport::build_plan()` has no cross-rank agreement on `TransportSpec::components`, unlike the peer-wise  | reviewer |
| R-32 | [TR-F5](docs/dev/reviews/2026-09-19/transport-runtime.md) | `src/runtime/runner.cpp:1567` | `resolve_epoch_participants()`'s `MPI_TAG_UB` safety check queries `MPI_COMM_WORLD` instead of the configured  | reviewer |

### MINOR — latent, edge case, robustness

| # | Source | Location | Finding | Checked |
|---|---|---|---|---|
| R-31 | [TR-F4](docs/dev/reviews/2026-09-19/transport-runtime.md) | `include/fuse/transport/transport.hpp:255`, `src/transport/lowfive_transport.cpp:66` | Reclassified MAJOR→MINOR while fixing (2026-09-19): confirmed by reading both cited lines that `SOMA_ENABLE_LOWFIVE` names a stale identifier only in a doc comment and an error-message string, never in a live `#ifdef`/`defined()` — LowFive was never compiled out by it. Both lines already read `FUSE_ENABLE_LOWFIVE` after wave 0's commit `4a2ad6e` (the same commit that fixed R-06's twin, R-51). **Fixed in wave 0.** | **verified here** |
| R-33 | [BK-MIN1](docs/dev/reviews/2026-09-19/backend.md) | `src/backend/spacetime.cpp` (`apply_spacetime`) | Zero-temporal-weight early return skips all spatial-operator bounds/null validation for that call, so a malfor | reviewer |
| R-34 | [BM-M6](docs/dev/reviews/2026-09-19/base-model.md) | `include/fuse/model/iteration.hpp:80,127` | Dangling doc cross-reference: "the header comment" it points to does not exist | reviewer |
| R-35 | [BM-M7](docs/dev/reviews/2026-09-19/base-model.md) | `src/base/error.cpp:167,191` | Previous `std::terminate` handler captured in `g_previous` but never chained or otherwise used | reviewer |
| R-36 | [BT-F10](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `src/backend/intx_weights.cpp:36` | `#if defined(FUSE_HAVE_MPI)` tests presence, not value, of a `#cmakedefine01` macro (latently masked by the ti | reviewer |
| R-37 | [BT-F11](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `CMakeLists.txt:65,68` | Stale status message: claims "stage S1" and lists only the `base` layer, four stages and eight layers after th | reviewer |
| R-38 | [BT-F12](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `tools/check-layering.py:76-113` (`check_legacy`) | Legacy-name checks are not comment-aware, unlike the sibling gate `check-invariants.py`; a prose comment namin | reviewer |
| R-39 | [BT-F13](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `docs/dev/layering.md` (diagram) | The canonical layer diagram omits `linalg` and `field`, which are real, separately-ranked layers in `tools/che | reviewer |
| R-40 | [BT-F14](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `CMakePresets.json:11` | `binaryDir` introduces a third build-directory convention (`build/<preset>`) alongside the `build-tier1`/`buil | reviewer |
| R-41 | [BT-F15](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `test/backend/moab_mesh_test.cpp:614` | `FUSE_MOAB_TEST_MESH` is referenced but defined nowhere in the build; that section is permanently dead code in | reviewer |
| R-42 | [BT-F16](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `cmake/FuseDependencies.cmake:99-101` | `FetchContent_Declare(KokkosKernels ...)` is dead: never `find_package`d or `FetchContent_MakeAvailable`d anyw | reviewer |
| R-43 | [IR-F4](docs/dev/reviews/2026-09-19/ir-schedule.md) | `src/ir/lowering.cpp:173-179` | Merge's "newest contribution" tie-break silently keeps the first writer when two contributions share `iterate` | reviewer |
| R-44 | [IR-F5](docs/dev/reviews/2026-09-19/ir-schedule.md) | `src/schedule/scheduler.cpp:599-618` | `E-SCHED-EPOCH-PARTIAL`'s rank-set-mismatch check cannot fire under any path through compilation into the dataflow graph that exists today, so it | reviewer |
| R-45 | [NL-A2](docs/dev/reviews/2026-09-19/num-linalg-field.md) | `src/num/accel_quasi_newton.cpp:536,632` | `linalg::least_squares_into` skips the `m >= k` precondition its sibling enforces, and the resulting `m < k` p | reviewer |
| R-46 | [TR-F6](docs/dev/reviews/2026-09-19/transport-runtime.md) | `src/transport/lowfive_transport.cpp:447` | The interception probe dataset name `"__soma_lowfive_probe__"` is a second, purely cosmetic residual upstream  | reviewer |

### IMPROVEMENT

| # | Source | Location | Finding | Checked |
|---|---|---|---|---|
| R-47 | [BT-F17](docs/dev/reviews/2026-09-19/build-tools-docs.md) | `check-invariants.py:34` (`scan`) | The `FUSE_HAVE_*`/`#ifdef`-misuse gate scans only `include/` and `src/`, never `test/` or `examples/` — the ex | reviewer |
| R-48 | [NL-I1](docs/dev/reviews/2026-09-19/num-linalg-field.md) | `include/fuse/num/waveform.hpp:478-497,536-556` | DG/Spline `evaluate`/`integrate` rebuild the shared, per-element-independent basis matrix once per field eleme | reviewer |

### DEVIATION — differs from the documented structure without a recorded decision

| # | Source | Location | Finding | Checked |
|---|---|---|---|---|
| R-49 | [NL-D1](docs/dev/reviews/2026-09-19/num-linalg-field.md) | `test/CMakeLists.txt:107-109`, `test/num/convergence_test.cpp:1-13` | `num.convergence` test exercises `fuse::model::iteration.hpp` entirely, not the `num` layer | reviewer |
| R-50 | [TR-F7](docs/dev/reviews/2026-09-19/transport-runtime.md) | `test/parallel/*.cpp` | None of the four parallel test files in scope use the documented `FUSE_REQUIRE_RANKS`/`FUSE_REQUIRE_AT_LEAST_R | reviewer |

---

## 2. What the findings have in common

Fifty findings reduce to six causes. Fixing a cause closes several findings at once, and adding the
gate that would have caught it keeps them closed.

**2.1 Tests that exist but do not run.** Three findings, and together they are the most consequential
in the set. `test/backend/moab_mesh_test.cpp` opens with `#ifdef FUSE_HAVE_MOAB` and never includes
`config.hpp`, so the macro is undefined and the whole file compiles to one skip stub in *every*
configuration — the 38 MOAB cases recorded as "unrun in the sandbox" (KNOWN_ISSUES C1) would not have
run on the host either (R-06). `test/schedule/buffers_test.cpp` and
`slab_test.cpp`, 33 cases, are not registered in `test/CMakeLists.txt` at all
(R-07). The three reference-output IR dumps were checked in but the test that compares
against them was never ported (R-26). The invariant gate that catches the first
pattern scans `include/fuse` and `src` but not `test/` or `examples/` — the same scope gap the layering
gate had until 2026-09-17, in a different tool.

**2.2 A rank fails alone and the others wait.** Five findings violate the one rule
`docs/dev/rank-agreement.md` exists to state: a rank that rejects its input must still reach every collective
its partners will reach. `step_period()` returns on a failing implicit coupling group while the comment two
lines below says the rank must keep executing collectives (R-09); iteration
convergence is decided from a rank-local residual with no reduction, so a partitioned interface exits
the group on different iterations per rank (R-10); `build_plan()` has no
cross-rank agreement on the component count (R-30); and the MOAB path has
data-dependent per-rank rejections ahead of documented collectives in both weight building and halo
planning (R-01, R-02). Every one is a hang, not an error message.

**2.3 A macro nothing defines.** The class this project has been bitten by four times is here twice
more: the MOAB test stub above (R-06) and its exact twin one directory
over, `test/parallel/comm_graph_test.cpp` (R-51) — 28 CommGraph/Crystal-Router
cases compiled to nothing for the identical reason. (An earlier draft of this section also counted
`SOMA_ENABLE_LOWFIVE` here as a live `#ifdef` compiling LowFive out of every build; re-reading both
cited lines while fixing R-31 found the stale name only in a doc comment and an error-message
string, never in a conditional — reclassified MAJOR→MINOR, see its row above.) The invariant gate
checks that a `FUSE_HAVE_*` read follows the config include; it does not check for `#ifdef` on those
macros, and it does not know that a `SOMA_`-prefixed macro is a legacy name (the layering gate does,
but only for namespaces and includes). Wave 0 closed this gap: `check-invariants.py` now also scans
`test/` and `examples/`, forbids `#ifdef`/`defined()` on any `FUSE_HAVE_*`/`FUSE_ENABLE_*`, and flags
a `SOMA_`/`KARMA_`-prefixed macro anywhere in `fuse::` code — R-06 and R-51 are both fixed by the
same commit that added it, `4a2ad6e`.

**2.4 Documentation that describes code that is not there.** `KNOWN_ISSUES` B1 still says the build
configuration with MOAB enabled does not build, fifteen lines above C1 saying it does (R-23). `build.md` documents
a `kokkoskernels` backend that was removed (R-24) and the CI matrix still has
a leg for it that cannot configure (R-05). Backend comments cite five parent
test files as evidence that were never ported (R-11). `buffers.hpp` documents exception
types the function no longer throws (R-28). "Merge as a left-fold" is documented
and referenced in an error message, and not implemented (R-27). The
`conventions.md` promise of an installable CMake package has no `install()` behind it
(R-25).

**2.5 The portability that was asked for stops at the header.** The native mesh has no execution or
memory space parameter and `finalize()` is host-only (R-03); `measure()`, `centroid()`
and `normal()` are not device-callable and use host facilities (R-14). The decomposition
tables added on 2026-09-18 *are* device-callable — but the mesh that would call them from a kernel is
not.

**2.6 Records that drifted from the tree.** A mutation anchored on text that no longer exists
(R-21); the four MOAB paths superseded on 2026-09-18 recorded only as a
comment in `UPSTREAMS.lock`, so `verify-import.py` does not see them (R-22);
`sync-upstream.py` cannot parse the lock file's multi-line arrays (R-20).

---

## 3. Deviations from the planned structure

These are not defects in code; they are places where the tree differs from what the plan and the
developer guide say it is, with no decision recorded. Each needs either the code or the document
changed, and the choice written down.

| Planned | Actual | Finding |
|---|---|---|
| Every port lands as a rename-only commit then a transform (porting.md step 2–4) | The MOAB backend, `tempest_weights.cpp` and both S5 test files landed as single commits | recorded in the plan (S5.2), not in porting.md as an allowed exception |
| Reference-output IR regression suite is the first mutation gate (S3.1) | Reference-output files present, comparing test absent | R-26 |
| Parallel tests declare rank requirements through the documented mechanism (testing.md) | Several in-scope parallel tests do not | R-50 |
| `num.convergence` tests the `fuse::num` layer | The file exercises `fuse::model` only | R-49 |
| One build-directory convention | **FIXED**: `CMakePresets.json` now uses `build-<preset>`, matching `build-*/` everywhere else | R-40 |
| `layering.md` diagram shows every layer | **FIXED**: `linalg` and `field` added, and the stale `interface` row renamed to the gate's actual `abi` | R-39 |
| Installable package `fuse-coupler` (conventions.md) | **FIXED**: `install()`/`export()`/`fuse-couplerConfig.cmake` added; `find_package(fuse-coupler)` now works and is ctest-verified (`install.consumer_find_package`) | R-25 |
| Native Kokkos mesh parameterised on execution space (S4.8 rulings) | Host-only `Mesh<M,S>` | R-03 |

---

## 4. The order to fix them in

Each wave is one or more commits; the gate additions in wave 0 are first because they turn the
rest of the waves from "fixed once" into "cannot recur". Every fix begins by reading the cited lines,
and every fix to library code carries a test that fails before it and a mutation in
`build-evidence/mutations.toml` where the gate can host one.

**Wave 0 — the gates, so the fixes stay fixed. All four DONE, 2026-09-19.**
1. **DONE** (`4a2ad6e`). `tools/check-invariants.py`: scan `test/` and `examples/` for the feature-macro
   rule; add a rule that `#ifdef`/`#ifndef`/`defined(` on any `FUSE_HAVE_*`/`FUSE_ENABLE_*` is a
   violation (they are `#cmakedefine01`, always defined); add `SOMA_`/`KARMA_` macro prefixes to the
   legacy-name rule. Fired on both R-06 and R-51 the moment it landed —
   the two live cases this exact rule was written for.
2. **DONE** (`b8e3f54`). `test/CMakeLists.txt`: a configure-time check that every `test/**/*_test.cpp`
   is registered by some `fuse_add_gtest`/`fuse_add_mpi_test` call, failing configure with the
   unregistered names; `schedule/buffers_test.cpp` and `slab_test.cpp` registered. Closes
   R-07.
3. **DONE, checker only** (`06e882a`). `tools/mutate.py --check-anchors`: verify every `old` string in
   the manifest exists exactly once in its file, without building; reads manifests from
   `build-evidence/mutations/`. Confirmed it fires on R-21's stale anchor;
   the anchor ITSELF was rewritten separately while fixing R-21 (wave 2/3, `build-tools-docs` fixer) —
   the checker catching it is not the same as it being fixed.
4. **DONE** (`ffea077`). `tools/verify-import.py`/lock reading: `UPSTREAMS.lock` parsed as TOML;
   read superseded paths from the lock file's structured arrays only; the S5.2 comment turned into
   real `[soma].superseded` array entries (closes R-22); the
   multi-line array parser in `sync-upstream.py` fixed to use `tomllib` (closes
   R-20).

**Wave 1 — BLOCKERs that must precede the first host run of the MOAB-enabled build configuration.**
5. **DONE** (backend fixer, 2026-09-19, verified here). `test/backend/moab_mesh_test.cpp`: includes
   `config.hpp` and tests `#if FUSE_HAVE_MOAB` now. The 38 cases compile again; a host run of the
   MOAB-enabled build configuration (this sandbox cannot `MPI_Init`) is still what will run them for
   the first time — see
   `KNOWN_ISSUES.md` C1 (R-06).
6. **DONE, wave 0** (`b8e3f54`; see wave 0 item 2 above). `buffers_test` and `slab_test` registered;
   both pass at the base build configuration (R-07).
7. **DONE, PARTIAL per pass2** (transport-runtime fixer). `runner.cpp` `step_period()`: a failing
   implicit coupling group records the failure and continues to the remaining collective instructions, exactly as the
   non-group path does two lines below (R-09). The code fix is correct by
   inspection (confirmed by the pass2 reviewer reverting it and seeing zero test failures), but it
   still has no mutation entry and no test that distinguishes fixed from broken — see
   `docs/dev/reviews/2026-09-19/pass2/re-transport-runtime.md` N-transport-runtime-2 (MAJOR, open).
8. **DONE, CONFIRMED-FIXED per pass2**. `iteration.cpp`: convergence measures are reduced across the
   group's processes before the decision, through the existing `residual_norm` reduction path; a
   rank owning no part of the interface contributes the undefined marker, not zero
   (R-10). Test: two ranks with unequal local residuals exit on the same
   iteration, host-verified per `re-transport-runtime.md`.
9. **DONE, CONFIRMED-FIXED per pass2**. `mpi_transport.cpp` `build_plan()`: component count agreed by
   the same handshake `build_plan_from_peers` already does
   (R-30), per `re-transport-runtime.md`.
10. **DONE, PARTIAL per pass2**. MOAB `intx_weights.cpp` and `plan_halo_exchange`: every
    data-dependent rejection ahead of a collective becomes a locally recorded failure that is reduced
    at the collective, so all ranks reject together (R-01, R-02). The
    backend pass2 review confirms R-02 CONFIRMED-FIXED but finds R-01's fix incomplete — the same
    hang-hazard class one call frame higher, in the same two functions, still open as
    N-BACKEND-1 (BLOCKER). Still needs a host run of the MOAB-enabled build configuration for either
    to execute.
11. **DONE, CONFIRMED-FIXED per pass2**. `descriptor.hpp` reader: validate every enumerator read from
    bytes against its range before the cast, rejecting with `E-DESC-ENUM`
    (R-04). Mutation: remove one range check; per `re-base-model.md`.
12. **DONE, CONFIRMED-FIXED per pass2**. `pipeline.hpp` `FieldSpan::from()`: an empty partition is a
    valid zero-extent view, not a rejection (R-08) — this is the X3 rule
    from S1 reaching the pipeline; per `re-num-linalg-field.md`.
13. **DONE**. `.github/workflows/ci.yml`: `kokkoskernels` leg replaced with `builtin`, the value
    `FUSE_LINALG_BACKEND` actually implements (R-05).
14. **DONE, PARTIAL per pass2**. Native mesh: this BLOCKER (R-03) is a
    design change, not a patch — the mesh gains an execution/memory space parameter and
    `finalize()`, `measure()`, `centroid()`, `normal()` become device-callable
    (R-14). The backend pass2 review confirms R-03 CONFIRMED-FIXED; R-14 has a residual
    defect/limitation found this pass — `measure()`/`centroid()`/`normal()` are now
    `KOKKOS_INLINE_FUNCTION`, but the `FUSE_DEBUG_ASSERT` path they call is not itself
    device-callable, so the device-callability claim only holds in `NDEBUG` builds — see
    `re-backend.md`.

**Wave 2 — MAJOR.** In the order they are listed in §1; the documentation ones
(R-23 **DONE**, R-24 **DONE**, R-11,
R-28) are one commit and go first because they are cheap and each is currently
telling a reader something false. R-31 turned out not to need a
code fix at all — see its reclassification to MINOR above; it and R-25
(**DONE**: `install()`/`export()` now exist) are the two former-MAJOR items
this wave already closes. The `error_policy()` data race (R-16) matters as
soon as two threads can throw. The `lcm_denominator` overflow (R-15) is the same
defect B1a fixed elsewhere and gets the same `try_` treatment.

**Wave 2 pass2 status, 2026-09-20 (see §6 for detail):** CONFIRMED-FIXED — R-11, R-15, R-16, R-25,
R-26, R-27, R-28, R-29, R-30, R-32, R-46, R-50 (R-50 is also DEVIATION D-2 in §3, closed the same
way). PARTIAL — R-12, R-13 (backend: correct fix, mutation entry only, no permanent regression test),
R-18, R-19 (base-model: guard/rejection covers some but not all call sites — see §6 for the two new
findings that finish the job). REJECTION-UPHELD — R-17, R-31.

**Wave 3 — MINOR, IMPROVEMENT, DEVIATION.** Group by file. For each DEVIATION in §3, decide and record:
either the code moves to match the document or the document is corrected and the decision noted in
the plan.

**Wave 3 pass2 status, 2026-09-20:** CONFIRMED-FIXED — R-33, R-34, R-35, R-36 through R-42 (except
R-41), R-43, R-44, R-45, R-47. R-41 (`test/backend/moab_mesh_test.cpp:614`'s dead
`FUSE_MOAB_TEST_MESH` guard) was deferred by the build-tools-docs fixer to the backend layer and
CONFIRMED-FIXED in the second fix round (`08848ec`: the definition is now supplied from
`test/CMakeLists.txt` when the fixture mesh is found). NOT-FIXED at pass 2 — R-49 (DEVIATION: `num.convergence` still exercises `fuse::model`, not `num`;
explicitly left for another owner). PARTIAL — R-48 (shared basis-matrix factorization is hoisted, but
the disclosed per-element scratch allocation remains). R-50's DEVIATION (parallel tests not using the
documented rank-requirement mechanism) is CONFIRMED-FIXED.

---

## 5. What the reviewers looked for and did not find

Each detailed review has a "Not found" section; read them before assuming an area is unreviewed. In
summary, the reviewers checked and found no defect in: the `try_*` exact arithmetic and its
reduction-then-fit order; the units affine rules; `Result`/`Diagnostics` accumulation; the SSA
single-producer invariant; the acyclicity prover including the newest-contribution typing; buffer
lifetime; canonical-dump locale pinning; the accelerators' rank-truncation contract; the least-squares
facade against the built-in oracle; the field pipeline's laziness and lifetime rules; MPI tags (all
from `tags.hpp`, within the bound); communicator lifetime in the transport; `CrystalRouter`; the
rename-only import integrity; the layering link graph; the compile-features audit.

Not reviewed by anyone: `examples/` beyond build correctness; the LowFive transport's body (compiled
out, see R-31); the TempestRemap path's body (no TempestRemap here).

---

## 6. Second pass (2026-09-20)

Six independent read-only re-reviews, one per layer group, against branch `review-2026-09-19-fixes`
at HEAD `dbdc872` — the same rubric, verifying every first-pass finding assigned to that layer and
looking fresh for anything the fix wave introduced or missed. Full reports:
`docs/dev/reviews/2026-09-19/pass2/re-<layer>.md`.

Totals across the six pass-2 reports, summed from their verification tables: **39 confirmed fixed, 3 rejections upheld, 8 partial, 2 not fixed** (52 outcome rows; one first-pass finding is verified in two layers), and **23 new findings** (the six pass-2 tables sum to 23; two of build-tools-docs' ten were reported from a host run of the MOAB-enabled build configuration rather than by the reviewer, and an earlier headline of 21 excluded them without saying so). Commit `1565cdf`'s body states "forty-five of fifty-one confirmed fixed"; that figure was written from memory and is wrong -- this line is the record.

| Layer | Confirmed-fixed | Partial | Not-fixed / rejection-upheld | New findings |
|---|---|---|---|---|
| backend | 4 (R-02, R-03, R-11, R-33) | 4 (R-01, R-12, R-13, R-14) | 0 | 3 — N-BACKEND-1 BLOCKER, N-BACKEND-2 MAJOR, N-BACKEND-3 MINOR/DEVIATION |
| base-model | 6 (R-04, R-15, R-16, R-34, R-35; R-17 rejection-upheld) | 2 (R-18, R-19) | 0 | 2 — N-base-model-1 MAJOR, N-base-model-2 MAJOR |
| build-tools-docs | 16 own-layer, of 17 assigned (R-05, R-06, R-07, R-20–R-25, R-36–R-40, R-42, R-47) + 2 cross-layer (R-51 confirmed, R-31 rejection-upheld) | 0 | 1 at pass 2 (R-41, deferred to backend; fixed in round 2, `08848ec`) | 10 — N-build-tools-docs-1/2 MAJOR, -3/-4/-5/-6/-7 MINOR (-7 out of scope, flagged for backend), -8 IMPROVEMENT, -9 BLOCKER (host-reported failure to configure the MOAB-enabled build configuration, 2026-09-20; fixed), -10 MINOR (host-reported, same date; fixed) |
| ir-schedule | 5 (R-26, R-27, R-28, R-43, R-44) | 0 | 0 | 2 — N-ir-schedule-1 IMPROVEMENT; N-ir-schedule-2 BLOCKER (a merge step scheduled before the synchronisation points feeding it, found by a host run of the MOAB-enabled build configuration while N-ir-schedule-1 routed the sweep through `schedule()`; fixed `c92ad1c`) |
| num-linalg-field | 3 (R-08, R-29, R-45) | 1 (R-48) | 1 (R-49, DEVIATION, explicitly left for another owner) | 1 — N-num-linalg-field-1 MINOR |
| transport-runtime | 5 (R-10, R-30, R-32, R-46, R-50) | 1 (R-09) | 1 (R-31, rejection-upheld) | 6 — N-transport-runtime-1/2 MAJOR, -3 MINOR, -4/-6 DEVIATION, -5 IMPROVEMENT |

**Reading "partial."** Every PARTIAL row is a real, working fix with a specific, evidenced gap, not
a fix that failed — e.g. backend's R-01 (a residual hang-hazard one call frame higher, now
N-BACKEND-1), base-model's R-18/R-19 (the guard covers most but not all call sites, now
N-base-model-1/2), num-linalg-field's R-48 (the disclosed scratch allocation was never claimed
fixed), transport-runtime's R-09 (correct by inspection and by revert-and-rerun, but no mutation or
test yet distinguishes it from the bug). None of the six reports rejected a finding the first fixer
had accepted, and none overturned a first-pass rejection other than confirming it (R-17, R-31 both
REJECTION-UPHELD).

**New BLOCKER.** N-BACKEND-1 is the only new BLOCKER: the same per-rank-rejection-ahead-of-a-collective
hang class as R-01, found one call frame higher in the same two MOAB functions — not yet fixed,
needs a host run of the MOAB-enabled build configuration to execute either way.

**Cross-cutting pattern.** Four of the eight new build-tools-docs findings (N-3 through N-6, and the
out-of-scope N-7) are stale cross-references — finding-id mis-citations or dangling doc-page
citations — left behind by the first fix wave. base-model's two new findings and num-linalg-field's
one are the same shape: a fix that closed the finding's evidenced instances without covering every
call site the underlying rule applies to. This second pass fixed the build-tools-docs instances of
that pattern (N-build-tools-docs-1 through -6, -8 — see the fix report,
`docs/dev/reviews/2026-09-19/fixes/fix2-build-tools-docs.md`) and left N-7 for the backend layer's
own second fix pass, since it sits in `test/backend/**`.

**Deviation from a full run.** Two of the six pass2 reviews (base-model, num-linalg-field) note their
own reduced scope: base-model did not independently re-run the static gates (relied on the fixer's
own claim, cross-checked against everything else read); num-linalg-field's one MPI-only test could
not run in the sandbox. The MOAB-enabled build configuration and any MPI-launched test remain out of reach for all six
reviews, as at the first review — every claim resting on them is marked PARTIAL or NOT-FIXED above
rather than reported as a firm result.
