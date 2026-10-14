# Disposition of the 2026-10-11 review findings

Fix round 2026-10-12/13 against review base `4a8a9d8` (work base `e6a92e4`); integrated onto `main 4635f0f (vijaysm/kokkos-device merged) — head fdb9610 + the record commit`. Scope decided by the owner on 2026-10-12: BLOCKER, MAJOR and impactful MINOR/DEVIATION findings plus the wave-0 gates and the one-line record corrections; the IMPROVEMENT tier is recorded here as open. Every library fix landed with a test that failed before it where one was constructible; the per-finding evidence (pre-fix signature, post-fix result, higgs job ids) is in the lane reports kept with this directory (`lane-reports/`). Status vocabulary: **fixed** (code or record changed, commit named), **fixed, by reading** (no runnable path on higgs — LowFive, connect-accept — fix verified by inspection and compile only), **not a defect**, **deferred** (recorded, not changed, reason given), **open** (out of this round's scope).

**Totals over the 159 catalogued findings:** 132 fixed, 20 open, 3 deferred, 2 not a defect, 2 known.

| ID | Severity | Status | Commit | Note |
|---|---|---|---|---|
| BC-1 | BLOCKER | fixed | `fdb9610` | D26 closed: 23 test and 5 example targets gated behind the eigen backend; builtin tree 264/264 on higgs; NL-5 row fires E-LINALG-SCHUR-BACKEND under builtin. |
| NM-1 | BLOCKER | fixed | `3bdbe22` |  |
| PS-1 | BLOCKER | fixed | `6b44edd` |  |
| TF-1 | BLOCKER | fixed | `c9470e9` |  |
| TR-1 | BLOCKER | fixed | `e0ccb0e` |  |
| AB-1 | MAJOR | fixed | `05c8d96` |  |
| BC-2 | MAJOR | not a defect |  | BC-2 (3 stale mutation anchors) was already re-anchored prior to this review round, at commit e6a92e4 -- verified, not re-fixed. |
| BC-3 | MAJOR | fixed | `8a5f54c` | I9 enforcement note (invariants.md, landed): 'tools/mutate.py --check-anchors' (static, no build) now runs in the gates CI job and catches a drifted mutation anchor; running the mutations themselves and confirming KILLED… |
| BM-1 | MAJOR | fixed | `8ca5af3` | Owner's strict direction: check_channel derives the declared transfer from the RemapSpec, validate() surfaces E-CONN-CONSERVATION/E-CONN-LOCATION, finish() keeps authorized Transform steps; no in-tree graph needed adapting. |
| BM-2 | MAJOR | fixed | `5eed56f` |  |
| BM-3 | MAJOR | fixed | `da212c0` |  |
| BM-4 | MAJOR | fixed | `da212c0` |  |
| IS-1 | MAJOR | fixed | `2f80d7a` |  |
| IS-2 | MAJOR | fixed | `9385e4a` |  |
| NL-1 | MAJOR | fixed | `1bc14cf` |  |
| NM-2 | MAJOR | fixed | `da6bf54` |  |
| NM-3 | MAJOR | fixed | `420ac7a` |  |
| PS-2 | MAJOR | fixed | `c78fe3c`, `3fec017` |  |
| PS-3 | MAJOR | fixed | `c78fe3c`, `3fec017` |  |
| PS-4 | MAJOR | fixed | `51b1b90` |  |
| RT-1 | MAJOR | fixed | `57f4d42` | Ordinary-region bystander now reads the agreed outcome; the peer-failure-to-bystander gap stays open for both region kinds (D42 note). |
| RT-2 | MAJOR | fixed | `57f4d42` | All five co_return sites carry forward; RT-2(c) row registered at 2 ranks because 3–5 ranks hit the open peer-failure gap (D42 note); mutation row killed. |
| TD-1 | MAJOR | fixed | `6844f46` |  |
| TF-2 | MAJOR | fixed | `82dbf9d` |  |
| TF-3 | MAJOR | fixed | `aed1bef` |  |
| TF-4 | MAJOR | fixed | `18ffaf8` |  |
| TF-5 | MAJOR | fixed | `18ffaf8` |  |
| TR-2 | MAJOR | fixed, by reading | `de212e1` | D14 addendum (review 2026-10-11 TR-2, TR-4): fuse-lowfive-vol's ConfigureOnce() now issues MPI_Comm_split/MPI_Intercomm_create on every world rank (color MPI_UNDEFINED for a rank the LowFive channel does not name) instea… |
| TR-3 | MAJOR | fixed | `8640aa0` |  |
| TR-4 | MAJOR | fixed | `de212e1` | Recorded jointly with TR-2 under the same D14 addendum; see TR-2's record_text. |
| AB-2 | MINOR | fixed | `2417b1a` |  |
| AB-3 | MINOR | fixed | `9c3fc25`, `11acb5d` |  |
| AB-4 | MINOR | fixed | `25e4917`, `f225294` |  |
| AB-5 | MINOR | fixed | `14b3d15` |  |
| AB-6 | MINOR | fixed | `11acb5d` |  |
| AB-9 | MINOR | fixed | `f225294` |  |
| AB-10 | MINOR | deferred | `4aba940` | Documented in c-abi.md; recorded as D61 — local ABI refusals are not reconciled across a component's ranks. |
| BC-4 | MINOR | fixed | `fd13e8e` |  |
| BC-5 | MINOR | fixed | `a8f2286` |  |
| BC-6 | MINOR | fixed | `ff96eb3` |  |
| BC-7 | MINOR | known |  | A1 (no MPI CI leg), owner-blocked; unchanged |
| BC-8 | MINOR | fixed | `1d1920d` |  |
| BC-9 | MINOR | fixed | `f892e8a` |  |
| BC-10 | MINOR | fixed | `f71a32e` | FEATURE_PRESENCE_TEST widened to the CONNECT macros and tools/src scanned; the config.hpp-dependency regex deliberately not widened to bare ENABLE_* (false positives on option names in strings). |
| BC-11 | MINOR | fixed | `f892e8a` |  |
| BC-12 | MINOR | fixed | `a49bc9e` |  |
| BM-6 | MINOR | fixed | `26f5296` |  |
| BM-7 | MINOR | fixed | `da212c0` |  |
| BM-8 | MINOR | fixed | `53a09fc` |  |
| BM-9 | MINOR | fixed | `3eacbba` |  |
| BM-13 | MINOR | fixed | `73464b8` |  |
| BM-14 | MINOR | fixed | `1832227` |  |
| BM-15 | MINOR | fixed | `0bf9af2` |  |
| BM-16 | MINOR | fixed | `5eed56f` |  |
| BM-17 | MINOR | fixed | `5eed56f` | D14 update (negative lag): acyclicity.cpp documents .lag(-1) as a legal lead; BM-17's scope was limited to tolerance/max_iterations per the task brief, and negative lag was deliberately left as-is. |
| BM-18 | MINOR | fixed | `da212c0` |  |
| BM-19 | MINOR | fixed | `34646c0` | D-BM-19 note: window_length() (coupling_graph.hpp) was checked against the review's claim and found clean — it performs no Rational arithmetic, only comparisons; not a defect, no change made there. |
| IS-3 | MINOR | deferred |  | Doc route: rank-agreement.md gained "A gap the scan does not close"; extending the epoch scan to placement-moving Remap needs a lowering change. |
| IS-4 | MINOR | fixed | `6e67e25` |  |
| IS-5 | MINOR | fixed | `5536890` |  |
| IS-6 | MINOR | fixed | `93cfc65` |  |
| IS-7 | MINOR | fixed | `a9fdf0f` |  |
| IS-8 | MINOR | fixed | `ddd4356` |  |
| IS-9 | MINOR | fixed | `eb276fb` |  |
| NL-2 | MINOR | fixed | `3f1c787` |  |
| NL-3 | MINOR | fixed | `7104035` |  |
| NL-4 | MINOR | fixed | `7104035` |  |
| NL-5 | MINOR | fixed | `fdb9610` | No KNOWN_ISSUES.md entry needed beyond D26's text above (NL-5 is cited there as the test that fires the backend refusal). |
| NL-6 | MINOR | fixed | `b9783d0` |  |
| NL-7 | MINOR | open |  | IMPROVEMENT tier, excluded from this round by the owner's scope decision (2026-10-12) |
| NL-8 | MINOR | open |  | IMPROVEMENT tier, excluded from this round by the owner's scope decision (2026-10-12) |
| NL-10 | MINOR | fixed | `b49f534` |  |
| NL-11 | MINOR | fixed | `df03159` |  |
| NL-14 | MINOR | fixed | `465f9f9` |  |
| NM-4 | MINOR | fixed | `420ac7a` |  |
| NM-5 | MINOR | fixed | `da6bf54` |  |
| NM-6 | MINOR | fixed | `e0ca9cb` |  |
| NM-7 | MINOR | fixed | `2f8c345` |  |
| NM-8 | MINOR | fixed | `a60b59a` |  |
| NM-9 | MINOR | open |  | NM-9 (MINOR, src/backend/native_geometry.cpp:120-128): an exactly-degenerate fan triangle (a collinear corner) is refused E-MESH-POLYGON-NOT-STAR-SHAPED because tri_norm == 0 degenerates the test to 0 <= 0 -> refuse, whe… |
| NM-10 | MINOR | fixed | `3abf95b` |  |
| NM-11 | MINOR | open |  | NM-11 (MINOR, src/backend/native_mesh.cpp:509-551): dim-0 Point-cell meshes are shadowed by vertex bookkeeping -- entity_count(0)/owned_count(0)/global_id(0)/centroid(0) answer *vertex* data rather than Point-cell data, … |
| NM-12 | MINOR | open |  | NM-12 (MINOR, src/backend/native_migration.cpp:170-171,275; native_halo.cpp:332; native_mesh.cpp:21,136-156): int overflow on large meshes -- per-rank byte counts narrowed to int in exchange_migration_buffers, box/vertex… |
| NM-13 | MINOR | fixed | `1ea7b4d` |  |
| PS-5 | MINOR | fixed | `6b44edd` |  |
| PS-7 | MINOR | fixed | `c3bfd7a` |  |
| PS-8 | MINOR | open |  | PS-8 (MINOR, latent, out of scope this round): rcb.hpp's tie-path Allgatherv counts (`static_cast<int>(local_gids.size())`) and collectives.cpp's `allgatherv_i64` displacement arithmetic are `int`, narrower than the `int… |
| PS-9 | MINOR | fixed | `3fec017` |  |
| PS-10 | MINOR | fixed | `3fec017` |  |
| RT-4 | MINOR | fixed | `57f4d42` |  |
| RT-5 | MINOR | open |  | Per task instruction, left for the lead's record_text/disposition table under the IMPROVEMENT tier. |
| RT-6 | MINOR | fixed | `b9942cb` | Open question deferred to the user/lead, not resolved by this lane (per the task's own instruction: 'do not change kFixedSweeps; the owner rules on that'): the bisection narrative in window_runner_straddle_test.cpp descr… |
| RT-7 | MINOR | fixed | `4b96f19` |  |
| RT-8 | MINOR | fixed | `3cc71fe` |  |
| RT-9 | MINOR | fixed | `2af2368` |  |
| RT-11 | MINOR | open |  | Per task instruction, no further action taken; already known per the review index. |
| TD-2 | MINOR | fixed | `abf23d2` |  |
| TD-3 | MINOR | fixed | `6a9bd72` |  |
| TD-4 | MINOR | fixed | `3491ef8` |  |
| TD-5 | MINOR | fixed | `5072174` |  |
| TD-6 | MINOR | fixed | `11acb5d` |  |
| TD-7 | MINOR | fixed | `8847fd5` |  |
| TD-10 | MINOR | fixed | `833a5dc` | The merge into main keeps main's own `example.demo_b` row (`9c5e990`, gated on `FUSE_ENABLE_CONNECT_ACCEPT`, host-verified) in place of this commit's row. |
| TF-6 | MINOR | fixed | `c9470e9` |  |
| TF-7 | MINOR | fixed | `ecf99bf` |  |
| TF-8 | MINOR | fixed | `8d440c3` |  |
| TF-9 | MINOR | open |  | TF-9 (MLS column order, src/backend/mls_weights.cpp:538-549): out of scope for the transfer lane this round. MLS emits each row's columns in k-nearest-neighbour order, violating RemapWeights' documented "col: source enti… |
| TF-10 | MINOR | fixed | `98f05ff` |  |
| TF-11 | MINOR | fixed | `c9470e9` |  |
| TF-12 | MINOR | fixed | `c9470e9` |  |
| TF-13 | MINOR | fixed | `c9470e9` |  |
| TF-14 | MINOR | open |  | TF-14 (src/backend/intx_weights.cpp:1285-1296): out of scope for the transfer lane this round. intersect_surfaces' decode accepts raw position in [0,n) as 0-based; a future/convention-different MOAB writing 1-based posit… |
| TF-15 | MINOR | fixed | `18ffaf8` |  |
| TF-16 | MINOR | fixed | `fb2e58d` |  |
| TF-17 | MINOR | open |  | TF-17 (MINOR/IMPROVEMENT, src/backend/tempest_weights.cpp:435,494,506): out of scope for the transfer lane this round (IMPROVEMENT tier). Failure and order-degradation detection keys on literal substrings "EXCEPTION" / "… |
| TR-5 | MINOR | fixed, by reading | `f647d31` | Compiled; the connect-accept path is OFF on higgs (D23 addendum). |
| TR-6 | MINOR | fixed | `2e42414` |  |
| TR-7 | MINOR | fixed | `69dd479`, `54fe459` |  |
| TR-8 | MINOR | fixed | `69dd479`, `54fe459` |  |
| TR-9 | MINOR | fixed | `54fe459`, `f647d31` |  |
| TR-10 | MINOR | fixed | `1f45389` |  |
| TR-12 | MINOR | open | `4b96f19` | out of scope this round |
| TR-13 | MINOR | fixed | `0d3c222` |  |
| TR-14 | MINOR | fixed | `8bef2ae` |  |
| TR-15 | MINOR | fixed | `8bef2ae` |  |
| TR-16 | MINOR | fixed | `fa8a66d` |  |
| TR-17 | MINOR | fixed | `1b24c04` |  |
| BM-5 | DEVIATION | deferred | `d68bdab` | Recorded as D62 — descriptor is host-endian; fix waits for a kGraphVersion bump. |
| BM-10 | DEVIATION | fixed | `d373651` |  |
| BM-11 | DEVIATION | fixed | `d171e1e` |  |
| BM-12 | DEVIATION | fixed | `0bf9af2` |  |
| IS-12 | DEVIATION | fixed | `565a48b` |  |
| NM-15 | DEVIATION | fixed | `204d970` |  |
| PS-6 | DEVIATION | fixed | `e88ca77` |  |
| RT-3 | DEVIATION | not a defect | `4cf002c` | docs/user/coupling-windows.md (commit 9188ae7) now documents: a DtPolicy::Fixed member whose interval exceeds the window length is handled by the straddle replay, not rejected, on the compiled Runner path -- diverging in… |
| TR-11 | DEVIATION | fixed | `7f1a09f` |  |
| AB-7 | IMPROVEMENT | fixed | `5e0806e` |  |
| AB-8 | IMPROVEMENT | fixed | `05c8d96` |  |
| AB-11 | IMPROVEMENT | fixed | `1643733`, `f225294` |  |
| AB-12 | IMPROVEMENT | fixed | `3348c9a` |  |
| BC-13 | IMPROVEMENT | open |  | IMPROVEMENT tier, excluded from this round by the owner's scope decision (2026-10-12) |
| BC-14 | IMPROVEMENT | fixed | `6f22d08` |  |
| BC-15 | IMPROVEMENT | open |  | IMPROVEMENT tier, excluded from this round by the owner's scope decision (2026-10-12) |
| BM-20 | IMPROVEMENT | open |  | IMPROVEMENT tier, out of scope |
| BM-21 | IMPROVEMENT | open |  | IMPROVEMENT tier, out of scope |
| BM-22 | IMPROVEMENT | fixed | `9d0fd40` |  |
| IS-10 | IMPROVEMENT | fixed | `d402129` |  |
| IS-11 | IMPROVEMENT | fixed | `e300e29` |  |
| IS-13 | IMPROVEMENT | known |  | R-43, accepted at pass 2; unchanged |
| NL-9 | IMPROVEMENT | open |  | IMPROVEMENT tier, excluded from this round by the owner's scope decision (2026-10-12) |
| NL-12 | IMPROVEMENT | fixed | `a9b2707` |  |
| NL-13 | IMPROVEMENT | open |  | IMPROVEMENT tier, excluded from this round by the owner's scope decision (2026-10-12) |
| NM-14 | IMPROVEMENT | fixed | `fb07474` |  |
| PS-11 | IMPROVEMENT | fixed | `c3bfd7a` |  |
| PS-12 | IMPROVEMENT | fixed | `a5f8779` |  |
| RT-10 | IMPROVEMENT | open |  | Per task instruction, forwarded to the lead as-is: an open aggregation question, not resolved by this lane. |
| TD-8 | IMPROVEMENT | fixed | `f71a32e` |  |
| TD-9 | IMPROVEMENT | fixed | `5072174` |  |
| TR-18 | IMPROVEMENT | open |  | IMPROVEMENT tier, out of scope |

## Verification at the integrated head

Debug, MPI+MOAB, eigen linalg, Fortran, CUDA Kokkos (`nvcc_wrapper`) on higgs, slot `fuse-lanes/slot1`,
head `fdb9610`: whole-tree build clean; 353 ctest rows (323 in the main sweep, `verification.brusselator_runner`
alone, 29 `.device` twins alone) — 351 passed. The two failures reproduce bit-for-bit at `main 4635f0f` with no
review commit applied and are recorded as D55 (`parallel.window_newton_mpi.3` hangs) and D56
(`FsiEnergyDiag.FullBudgetOnTheDominantModeCloses` tolerance); they are not regressions of this round.
`verification.fsi_transient` and `verification.brusselator_lowfive_harness` timed out in the contended sweep and
passed on a serial retry. Static gates at the head: check-invariants 0, check-layering 0, gen-abi 6/6 current,
verify-import 693/0, mutate --check-anchors 348/348 valid. The full ctest log is kept as
`review-1011-final-ctest.log` in the project's artifact store.

## Surfaced during the round, outside the catalogue

- `verification.brusselator_runner` (`PerWindowStateMatchesDriverWithStoppingRuleRemoved`) hung at the
  `e6a92e4` baseline on higgs (reproduced on an unmodified checkout by the native lane) and runs clean
  (1514 s) at the integrated head. Causality was not established; the RT-1/RT-2 carry-forward fixes are the
  candidates. Worth one bisection before anyone relies on it.
- The peer-failure-to-bystander propagation gap D42 records for window regions exists on the ordinary-region
  path as well (RT-2(c)'s test hangs at 3–5 ranks and is registered at exactly two); open for both.
- `TF-2`'s `reconcile<T>` wrapping catches per-rank *failures* in the first-order classification; a cross-rank
  *value* disagreement (a halo fringe changing a cell's class on one rank only) is a separate, unclosed hazard.
- `TR-2`'s second hazard — the LowFive rendezvous runs only on a rank whose own first HDF5 call loads the
  plugin — is recorded under D14 and not closed.
- Two defects not in the review were found and fixed alongside IS-1/IS-9: a `Merge`'s output arc placement was
  inherited from the newest writer rather than the field owner, and the first IS-2 regression test did not
  exercise the gap it was written for (redesigned before landing).
- `tools/check-parallel-matrix.py` was not added to the CI gates job: its by-design gap and a broken claim both
  exit 1 and differ only in a line prefix; a `--allow-known-gaps` flag (or a distinct exit code) is the
  prerequisite.
- `verification.fsi_transient`, `gray_scott_reference`, `brusselator_participants`, `brusselator_lowfive_harness`
  and the `*.omp_identical` rows exceed a 300–600 s cap on this shared 8-core host; a longer default timeout for
  those rows, or an uncontended host, is needed for the sweep to finish without retries.

## Decisions that still belong to the owner

- RT-6: `kFixedSweeps = 1` runs the runner-vs-driver bit-exactness comparison in the regime where agreement is
  trivial; restoring N≈28 with the measured 4.5e-14 Jacobi floor, or recording N=1 as the intended check.
- RT-10: `iteration_stats()` reports only the last window pass of a multi-pass period; document or aggregate.
- BM-17: negative `lag` is accepted everywhere as a lead; decide whether the front end should refuse it.
- IS-3: the placement-moving `Remap` has no compiler-checked agreement point (`rank-agreement.md`, new section);
  closing it is a channel-lowering change.
- The `tools/check-rank-agreement.py` static gate proposed by the index (§2.1) is not built; thirteen of the
  twenty-nine BLOCKER/MAJOR findings were this class.
- TF-9: MLS emits each row's columns in k-nearest order against `RemapWeights`' ascending-column contract.
- The IMPROVEMENT tier (allocation hoists RT-5/NL-7/NL-8/NL-9, CI cache BC-13, notices BC-15, mutators BM-20,
  precision round trip BM-21, `FieldSpan::from` const_cast NL-13) is open by scope.
