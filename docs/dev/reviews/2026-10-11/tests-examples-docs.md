# Review 2026-10-11 — test infrastructure, examples, docs-vs-code consistency

Read-only review at HEAD `4a8a9d8`. Scope: `test/CMakeLists.txt`, `test/support/**`,
`test/partition/CMakeLists.txt`, examples build wiring + sources, `docs/user/**` +
`docs/dev/{testing,invariants,conventions}.md` sample-verified against code. No files modified; no
builds/tests run. Static gate `tools/check-invariants.py` run read-only: 0 violations. Findings feed
the ranked index in `index.md`; IDs here are referenced as `TD-*`.

## 1. Findings

| ID | Severity | file:line | Finding | Evidence | Why it matters | Suggested fix | KNOWN |
|---|---|---|---|---|---|---|---|
| TD-1 | **MAJOR** | `docs/user/brusselator.md:302` | "Deliberate refusal, **tested**: `E-BRUSS-MESH-INDEX`, `E-BRUSS-U-REACTION`, `E-BRUSS-DIFFUSION-DIRECT-DECOMPOSED`" — 2 of the 3 are never fired by any test. Emitters only: `examples/brusselator/brusselator_physics.hpp:356,367` (MESH-INDEX), `examples/brusselator/species_core.hpp:620` (U-REACTION). The only refusal fired in the whole brusselator verification set is `DirectSolverRefusesWhenDecomposed` (`test/verification/brusselator_species_core_decomposition_test.cpp:548-558`). No test in `test/` references or triggers the other two. | Violates testing.md's five-states ("an unexercised error path is not a feature") and invariants.md I4 ("a refusal that no test makes fire does not count as implemented"). The doc is the binding record of what is verified. | Add tests that make both refusals fire (a backend whose centroids fail the bijection recovery; a U fixed point forced past 50 solves), or re-mark the two as implemented-untested refusals | not recorded |
| TD-2 | MINOR | `docs/dev/testing.md:139-153` | "Golden files" section is stale: names `test/golden/waist-v1.bin` as "the frozen canonical descriptor" compared by "`Descriptor.MatchesTheFrozenWaistV1Bytes`" — that test does not exist. Real test is `Descriptor.MatchesTheFrozenWaistBytes` against `waist-v11.bin` (`test/model/descriptor_test.cpp:474-477`); `kGraphVersion = 11u`; v1 bytes are now *refused* (`descriptor_test.cpp:459-472`, E-DESC-VERSION). | A reader greps for the evidence and finds nothing — precisely the "never complete a name a pipe truncated" class testing.md:314-331 itself documents as review-invisible. | Update the section: canonical file is `waist-v11.bin`, test is `MatchesTheFrozenWaistBytes`, v1 kept as a refused superseded version | not recorded |
| TD-3 | MINOR | `docs/user/mms-pair.md:68-69`, `docs/user/dn-heat.md:139-140` | Docs claim `--dump` prints "without running anything, then exits 0" / "instead of running anything"; both binaries run the **full gated physics** after printing the dump. `examples/mms_pair/main.cpp:133-136` prints then continues through the whole run + gates (138-220); `examples/dn_heat/main.cpp:240-244` prints inside `run_case`, which then runs the case, and `main` runs all 8 configurations regardless. | `--dump` costs the entire run time and its exit status is the physics gates', not unconditionally 0; a user scripting dumps gets surprising behavior. | Either early-return after the dump in both mains, or fix the docs ("also runs the full gated run; exit status is the gates'") | not recorded |
| TD-4 | MINOR | `docs/user/fsi-channel.md:212-218`, `docs/user/examples.md:22` | fsi-channel page/`examples.md` mark the example "Verified" with no mention that both `example.fsi_channel` rows are **DISABLED** under `FUSE_ENABLE_CUSPARSE`, that `fsi_energy_diag`'s 480-period case skips under the flag, or that the fluid solver is cusolverRf (not "single-rank Eigen SparseLU") there. | `examples/CMakeLists.txt:487-492` (DISABLED TRUE); `test/verification/fsi_energy_diag_test.cpp:944-950` (GTEST_SKIP, D35); KNOWN_ISSUES.md D34, D35. | A user on a cusparse build sees "Not Run (Disabled)" with no explanation on the user pages; the "Eigen SparseLU" claim is configuration-dependent. | Add one sentence to the page's State section and a footnote to the examples.md table, as the brusselator row already does for D23 | **KNOWN: D34/D35** (recorded in KNOWN_ISSUES + CMake; missing only from the user-facing pages) |
| TD-5 | MINOR | `docs/user/gray-scott.md:72` | "`--ratio 2` means V steps at `dt/2`" contradicts the code and the page's own 7:13 convention. `examples/gray_scott/main.cpp:111-124`: with `--dt D --ratio m`, U's interval (the window) is `m·D` and V steps at exactly `D`. Line 73's "`--ratio 7:13` … the window is 13 parts and V is 7" uses "parts" = `D`-units. | A reader of the command two lines above (`--dt 0.5 --ratio 2`) expects V's step = 0.25; it is 0.5 (window 1.0). Same convention, two readings. | Reword: "`--ratio m` makes the window `m·dt` and V steps at `dt` (V is m× finer than the window)" | not recorded |
| TD-6 | MINOR | `src/abi/provider_adapter.cpp:93,102,198,233,263` | Provider-adapter refusal codes `E-ABI-PROVIDER-CREATE/-BIND/-SAVE/-RESTORE/-WINDOW/-ELEMENT/-GRAPH/-TABLE/-EXCEPTION` are fired by no test; only `E-ABI-PROVIDER-ADVANCE` is asserted (`test/abi/conformance_test.cpp:479`; `linear_provider.c` has only BAD_VERSION/BAD_DESCRIBE/BAD_ADVANCE variants). | I4: "a refusal test per capability **and per error code**" — these codes count as unimplemented by the project's own rule. (Same class as the ABI review's AB-9; reported from the test-coverage side.) | Add bad-create / wrong-window-echo / non-F64 plugin variants, or record the gap as D15's sibling | partially adjacent to **KNOWN: D15** |
| TD-7 | MINOR | `docs/user/layout-driver.md:73-74` | "otherwise the driver is silent on success" is false. Rank 0 always prints `shape(comp1, comp2): …` (`examples/layout_driver/main.cpp:535-538`) and the `check: N/N destination entities delivered` line under `--check` (:734-737). | Doc/behavior mismatch on the page whose subject is exactly what the driver prints. | Reword ("silent apart from the shape line — and the check line under `--check`") | not recorded |
| TD-8 | IMPROVEMENT | `tools/check-invariants.py:41` | The feature-macro rule regex is `FUSE_(HAVE_[A-Z0-9_]+\|ENABLE_EXCEPTIONS\|LINALG_EIGEN)` — an `#ifdef FUSE_ENABLE_CUSPARSE/MOAB/MPI/LOWFIVE/…` would pass the gate, though REVIEW.md:133-136 describes the rule as covering "any `FUSE_HAVE_*`/`FUSE_ENABLE_*`". Tree-wide grep today finds zero such occurrences. | Latent scope gap in the R-06/R-51-class gate; the described rule and the implemented rule differ. (Same class as BC-10; reported independently from the docs side.) | Widen the alternation to `FUSE_(HAVE\|ENABLE)_[A-Z0-9_]+` | KNOWN-class: R-47/wave-0 description |
| TD-9 | IMPROVEMENT | `docs/user/gray-scott.md:33-34` | "Newton-solved with `Eigen::SparseLU`" is now one indirection behind: `MonolithicGrayScott` holds a `fuse_examples::SparseLuBackend`. Flag-off it is bit-for-bit Eigen::SparseLU, pinned by `verification.sparse_solve_backend`; flag-on it is cusolverRf. | The doc claim is now configuration-dependent (gray-scott rows are *not* disabled under the flag, unlike fsi). | Add "(held in `SparseLuBackend` — Eigen by default, cusolverRf under `FUSE_ENABLE_CUSPARSE`)" | not recorded (recent commits 3b07d42/c266b78) |
| TD-10 | MINOR | `docs/user/examples.md:6-7` | "every one of them is run by `ctest` wherever its build configuration allows it" — `demo_b` has **no ctest row under any configuration**; even a host with LowFive-dist never runs it via ctest. | `examples/CMakeLists.txt:610-628` builds the executables only; grep for `add_test.*demo_b` across the tree: zero hits; the only driver is the human-run `examples/demo_b/run_demo.sh`. | The page's own stated principle ("an example nobody runs rots within a release") applies to demo_b even where its config is satisfied. | Register a ctest row gated on the same three conditions the build gate uses, invoking `run_demo.sh` (host-only today, but absent everywhere, which is a different state than host-verification-needed) | partially KNOWN: demo-b.md:77-96 honestly marks the demo "Implemented, untested" but does not say no ctest row exists even in principle |

## 2. Checked and found clean

- **Registration audit**: all 151 `*_test.cpp` on disk are registered or declared gated (independently re-derived; matches the configure-time audit). `test/partition/CMakeLists.txt`'s wrappers record sources correctly for the audit's glob. No test compiles to nothing.
- **Feature-macro discipline**: zero `#ifdef FUSE_HAVE_*`/`defined(FUSE_HAVE_*)` in test/ or examples/; `tools/check-invariants.py` exits 0 violations (TD-8's scope caveat aside).
- **Harness** (`test/support/fuse_test_main.cpp` read in full): MPI→Kokkos→GoogleTest init order; rank-0-only printer; `OnTestEnd` Allgather that names failing ranks; final status Allreduce; `FUSE_REQUIRE_RANKS`/`AT_LEAST` macros correct; `self_communicator_handle()` documents the MPICH f2c(0) trap.
- **`allocation_counter.hpp`**: included by exactly one TU in each of its four binaries; plain-pair-only replacement as documented.
- **testing.md's counter defect is fixed as documented**: `runner_transport_test.cpp:85-105` counts unconditionally then branches, with the history in the comment.
- **No vacuous tests**: every `SUCCEED()` found is a static_assert carrier or an explicitly documented gap; `tools/compare-runs.cmake` refuses sub-200-byte "identical" output.
- **Convergence discipline**: `mms_transfer_test.cpp` asserts observed order 1±0.1 *plus* an absolute gate *plus* a disabled-exchange control expected to fail both; `gray_scott_mms_test.cpp` gates two-sided; exact iteration-count regressions exist.
- **Golden files**: suites only read them; regeneration is a separate human tool; superseded versions are kept *and* refused.
- **Parallel rank guards**: `epoch_reconcile`'s registration without rank 1 is deliberate and documented; `native_halo_mpi` registered only at RANKS 2 3; `window_runner_mpi` uses `FUSE_REQUIRE_AT_LEAST_RANKS`.
- **THROWS labels propagate to omp legs** — the CI-incident fix is in place.
- **Examples build wiring**: `fuse_examples_common` INTERFACE with cusparse-only CUDA libs; `FUSE_KOKKOS_HAS_OPENMP` PARENT_SCOPE export (the D11 fix); 21 `layout.*` rows = docs' 21; per-target Fortran module dirs (the .mod race fix); fsi rows disabled under cusparse exactly as D34 records; D35 skip names its KNOWN id in the skip message.
- **Docs build anchors**: every literalinclude start-at/end-at string in the sampled pages resolves in the source; `CMakePresets.json` `binaryDir = build-${presetName}` matches every documented command.
- **Error-code census**: all `E-CONNECT-*`, `E-LAUNCH-*`, `E-RT-WINDOW-*`, `E-MLS-*`, `E-MAP-*` codes cited in sampled docs exist in code and are fired by tests; layout refusals fired by the 4 `layout.refuse.*` WILL_FAIL rows; `E-BRUSS-DIFFUSION-DIRECT-DECOMPOSED` fired.
- **fsi-channel.md test names** (all five sampled, per testing.md's own name-verification rule) exist exactly.
- **gray-scott doc claims**: `coupling_tol` default 1e-8, not a CLI flag; flux in checkpoint; `--check` bound `100*coupling_tol`; the two separately-recorded measurements both labeled as such.
- **brusselator `--check` thresholds** match docs exactly.
- **SparseLuBackend**: flag-off is a pure Eigen::SparseLU forwarder (bit-for-bit pinned); cusolverRf failure paths return zeros with `info()` set; half-built seeds released before re-seed.
- **conventions.md "no iostream/printf in library code"**: holds in `src/` (single grep hit is a comment).
- **demo-b**: consumer's exit code is the load-bearing assertion exactly as documented; payload files carry no FUSE references.
- **KNOWN_ISSUES spot checks**: D27 still honestly split (forward closed, reverse open); D26 still true.
- **quickstart.md**: compiles against the real API.
- Untracked working-tree file `docs/dev/handoff-kokkos-device-2026-10-11.md` noted (not reviewed, not committed).

## 3. Not reviewed

- The ~1,741 test-case bodies beyond the sampled greps (structure, headers and targeted assertions were surveyed, not line-read).
- Example physics headers not fully line-read (claim-checked via targeted reads instead): `heat_component.hpp`, `dn_component.hpp`, `beam_solver.hpp`, `stokes_solver.hpp` body (read cache/pressure-gradient comments only), `fsi_components.hpp`, `brusselator_physics.hpp`, `species_core.hpp` (the solver-cache finding NL-1 lives here and was line-read by the num/linalg review), `brusselator/species_component.hpp`, `gray_scott/species_component.hpp`, `gray_scott_physics.hpp`, `monolithic_reference.hpp` body, `brusselator_graph.hpp`, `partitioned_test_backend.hpp`.
- Docs pages outside the sample list and `docs/dev/**` beyond testing/invariants/conventions.
- Shell drivers' bodies (`run_pair.sh`, `run_participants.sh` beyond header, `run_demo.sh`), `tools/mutate.py` internals, `build-evidence/` manifests, `.github/workflows/ci.yml` (covered by `build-tools-ci.md`), `fortran/` bindings (covered by `abi-fortran.md`), all library `src/` bodies.
- Any run outcome requiring MPI/MOAB/cusparse/device. testing.md's tier-0 "179 rows" and call-site "110" figures are dated build-tree measurements — not re-derivable without configuring; a static call-site count (115 incl. partition wrappers) is consistent with post-measurement row additions, not reported as a discrepancy.

## 4. Coverage map by layer

| Area | Files / cases | State |
|---|---|---|
| base | 9 / 89 | Well covered (rational, units, error, bytes, ids, config-vs-CMake Kokkos pin) |
| model | 16 / 183 + `descriptor_mpi` at 2-5 ranks | Well covered incl. cross-rank byte agreement |
| ir | 9 / 86 + golden IR | Well covered |
| schedule | 13 / 195 + golden plan | Well covered (incl. `buffers`/`slab`, the formerly-unregistered R-07 pair) |
| num | 7 / 163 (typed per space) + omp legs | Well covered |
| runtime | 14 / 213 + `runner_device` + compile-fail pair + omp determinism legs | Well covered |
| transport | 4 / 63 | Good unit coverage; distributed LowFive path deliberately unexercised (documented); demo_b is the intended vehicle and has no row (TD-10) |
| backend | 18 / 335 | Strong: native (mesh/migration/generators/halo), MLS, nodal interp, transfer_map_io, supermesh, operator conformance; MOAB behind flag with 12 MPI rows |
| field | 1 / 13 + compile-fail/ok pair | Thin but appropriate for a header-only layer; allocation fence is the load-bearing check |
| linalg | 6 / 38 + schur/krylov/linalg MPI legs | Good; distributed containers verified in-process *and* on real MPI |
| search | 4 / 22 + distributed_mpi 1-5 | Good |
| partition | 6 / 25 + rcb_mpi 1-5 + omp legs on the atomic_add hot spot | Good |
| abi | 5 / 38 + export/symbol/size goldens + C & Fortran install consumers + provider conformance (4 kinds, 3 bad variants) + MPI agreement | Strong; provider refusal codes partially unfired (TD-6) |
| parallel | 16 / 126 at 1-5 (mostly), host-verified per testing.md records | Good; accept-mode blocked by host MPI (D23), recorded |
| verification | 23 / 152 | Strong: order gates with controls, exact iteration counts, per-example physics suites, species-core decomposition, three-executable MPMD |
| install / layouts / examples-as-tests | install rows x3, 21 `layout.*`, 13 `example.*`, graph-tool golden x2, participants x4+x2 | Good; demo_b is the one example with no row anywhere (TD-10) |
| **Thin / none** | — | Two brusselator refusals unfired (TD-1); provider-adapter refusal codes unfired (TD-6); distributed LowFive unexercised; `field` numerically thin |

Overall: the test infrastructure is unusually disciplined — the registration audit, THROWS-label
propagation, anti-vacuous devices, and order gates with negative controls all do what testing.md claims.
The findings are concentrated where the project's own rules are strictest about honesty: doc claims of
"tested" (TD-1), a stale evidence citation in the binding testing doc (TD-2), and flag-conditional states
missing from user pages (TD-4).
