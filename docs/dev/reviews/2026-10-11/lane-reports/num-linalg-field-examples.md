# Lane report — num-linalg-field-examples (fix round 2026-10-12/13)

Lane head at handback: `f8e0385a4df9` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- NL-3 and NL-5 were not independently re-run on higgs with a dedicated before/after ctest invocation beyond the whole-tree build (NL-3 is a test-comment/assertion fix verified by reading against the cited fix commit; NL-5 has no test to run, deferred).
- NL-6, NL-12 are documentation/comment-only changes verified by reading, not by a dedicated test run.
- NL-10's parallel.linalg_mpi.2 MPI-tier row was not independently re-run beyond the whole-tree build this round (compiled only for that row); the tier-0 dist_vector row was run and passed.
- NL-14 has no dedicated regression test (constructing a >2^31-entry input is impractical); the guards were verified by reading and the surrounding code continues to pass the existing MPI suite at normal sizes.
- TD-10's demo_b ctest row was verified by reading plus `ctest -N` confirming correct (empty) registration under slot2's configuration; the gate never turns on in this environment, so the demo_b_producer/consumer pair itself was never executed.
- Four tests outside my owned paths (fsi_transient, gray_scott_reference, brusselator_participants, brusselator_lowfive_harness) timed out at a 600s cap; brusselator_lowfive_harness was wrongly omitted from an earlier pass of this disclosure. None touched by my commits; not independently confirmed to pass given more time within this round's budget.

## Findings

### NL-1 — fixed

Tests: `verification.brusselator_diffusion_solver_cache (BrusselatorDiffusionSolverCache.{Direct,Schur,Pcg,Gmres}SolverRefactorsOnChangedDiffusion)`

Evidence: Pre-fix (job cca3c1c7, phase NL-1 before revert of species_core.hpp's cache-key fix only, test file kept): all 4 cases FAILED -- 'changed' (|solve(diffusion_b)-solve(diffusion_a)|) == 0 for every solver kind, confirming the stale-factorization reuse; worst_vs_fresh up to 0.698. Post-fix (job 8a7a2b95, PHASE 1 AFTER): verification.brusselator_diffusion_solver_cache Passed 0.42 sec, all 4 subtests green.

### TD-1 — fixed

Tests: `verification.brusselator_refusal_coverage (BrusselatorRefusalCoverage.UReactionFixedPointRefusesPastFiftySolves)`

Evidence: E-BRUSS-DIFFUSION-DIRECT-DECOMPOSED was already fired (pre-existing test, unchanged). E-BRUSS-U-REACTION: new test forces UCore::step's reaction fixed point past its 50-solve cap on a 1-cell grid (dt=3.0, v=0.5, B=0.0; reaction_kokkos_span hardcodes A=3.4, PhysicsConfig::A is not read there -- discovered when an initial choice assuming cfg.A was live converged instead of diverging; re-derived against the real formula). Job 8a7a2b95 ctest_full.log first caught the bug (test FAILED, state()=0.597, converged instead of refusing); after correcting the parameters, job ad1b08a4 confirms verification.brusselator_refusal_coverage Passed 0.46 sec. E-BRUSS-MESH-INDEX: determined genuinely unreachable through NativeMeshBackend's public API (NativeGrid::make always regenerates a mesh matching its own nx/ny; loading the same name twice refuses E-MESH-NAME-AMBIGUOUS before any centroid read; set_placed_view/commit_remap_halo write a different, placement-scoped slot and the latter is documented internal-only) -- re-marked in docs/user/brusselator.md as implemented, deliberately untested, with the reasoning recorded in both the doc and the test file's own trailing comment, per the task's own escape hatch for a genuinely unreachable refusal.

### NL-2 — fixed

Tests: `linalg.gmres (GmresRefusals.RestartBelowOneRefusesAtEntry)`

Evidence: Pre-fix (job 8a7a2b95, restart guard reverted, test kept): running the new case directly under `timeout 30` exits 124 -- the restart=0 livelock, confirmed (the Arnoldi loop never executes a step, so total_iter never advances and the outer loop spins through a.apply forever). Post-fix: linalg.gmres ctest row Passed 0.50 sec (full suite, including the new case).

### NL-3 — fixed

Tests: `parallel.schur_slab_mpi.2/.3 (SchurSlabSolverMpi.OneRowRowBandRefuses, unchanged test, corrected comment+assertion)`

Evidence: Test-comment/assertion fix only (no library code changed): verified by reading c11ac18's actual fix (src/linalg/schur_slab.cpp:221-247, every rank scans ALL bands and refuses identically) against the old comment's claim of a local-only refusal; the test's own conditional `if (rank==nranks-1)` was accordingly also fixed to assert unconditionally, matching the tier-0 sibling test/linalg/schur_slab_test.cpp's own assertion shape. Not independently re-run on higgs this round beyond the whole-tree/full-ctest build (job 8a7a2b95's final_build.log, exit 0) -- a compile-and-static-correctness check, not a dedicated before/after regression run.

### NL-4 — fixed

Evidence: Doc-only fix: verified by reading `git log -1 c11ac18` (2026-10-09 22:19:08 -0400, 'Agree the Schur solver's refusals across every rank before any collective') against `git log -1 3b43bd0` (a different, unrelated commit, 'Document the Krylov solvers and record D26'). No test applicable; the record now cites the correct SHA and date.

### NL-5 — deferred

Evidence: Verified by reading: schur_slab_unavailable.cpp and schur_slab.cpp both define fuse::linalg::SchurSlabSolver::factor/solve/ctors in the SAME namespace; src/CMakeLists.txt compiles exactly one of them into fuse-coupler-linalg depending on FUSE_LINALG_BACKEND. A test target that links fuse::linalg (needed for every other linalg symbol the test would use) and additionally compiles schur_slab_unavailable.cpp as an extra source gets a duplicate-symbol link error under the eigen configuration (slot2's own configuration). Exercising E-LINALG-SCHUR-BACKEND therefore needs a library-level split (e.g. a separate object library per backend a test can choose to link) in src/CMakeLists.txt -- beyond this round's test/CMakeLists.txt-only mandate, exactly the case the task named in advance.

Record text: D-NL-5 (open): E-LINALG-SCHUR-BACKEND (src/linalg/schur_slab_unavailable.cpp, the non-Eigen SchurSlabSolver stub) is a refusal no test fires under any ctest row built so far, because the eigen and builtin backends both define fuse::linalg::SchurSlabSolver in the same namespace and cannot be linked into the same test binary. Needs a src/CMakeLists.txt library-level split (e.g. two differently-named object libraries, one per backend, a test can choose between) before a regression test can be added. Recorded for the lead's BC-1 coordination step.

### NL-6 — fixed

Evidence: Documentation-only fix (no library behavior changed, since every operator/preconditioner this library builds itself already fails uniformly): LinearOperator/Preconditioner's own header comments and docs/dev/linalg-krylov.md now state the uniform-failure precondition and name the residual exposure (a caller-supplied MatrixFreeOperator wrapping a data-dependently-refusing callable). Verified by reading; whole-tree build succeeds (job 8a7a2b95 final_build.log, exit 0).

### NL-10 — fixed

Tests: `linalg.dist_vector (DistVectorTwoPartitionFake.DotAndNorm2EqualTheGatheredSerialReduction)`, `parallel.linalg_mpi.2 (LinalgMpi.DotNorm2AndNormInfMatchTheWholeDomainClosedFormReference)`

Evidence: Not a library defect -- a test-robustness fix (the exact values are unchanged; only the assertion tightness changed from EXPECT_DOUBLE_EQ to a measured, term-count-scaled EXPECT_NEAR, as dist_sparse_matrix_test.cpp's own check_two_partition_apply already does). Whole-tree build succeeds (job 8a7a2b95 final_build.log, exit 0); linalg.dist_vector passed in the full ctest sweep (job 8a7a2b95 ctest_full.log). parallel.linalg_mpi.2 is MPI-tier (RANKS 2) and was not independently re-run beyond the whole-tree build this round -- compiled only for that one row.

### NL-11 — fixed

Tests: `linalg.dist_sparse_matrix (DistSparseMatrixApply.RefusesXAndYAliasing)`

Evidence: Pre-fix (job 8a7a2b95, assert reverted, test kept): linalg.dist_sparse_matrix Failed 0.40 sec -- 'Expected: apply(v,v) throws fuse::InvariantViolation. Actual: it throws nothing.' Post-fix: linalg.dist_sparse_matrix Passed 0.39 sec, 9/9 subtests green.

### NL-12 — fixed

Evidence: Comment-only rename (soma_core -> fuse-coupler-num, the real target name per src/CMakeLists.txt's own add_library(fuse-coupler-num ...)). Verified by reading; whole-tree build succeeds (job 8a7a2b95 final_build.log, exit 0).

### NL-14 — fixed

Evidence: FUSE_ASSERT(n <= INT_MAX)-equivalent guards added at the two static_cast<int> sites (MpiReducer::sum/allgather) and the two accumulated-total sites in MpiExchanger::update. No dedicated regression test: constructing a >2^31-entry input to fire the guard is impractical (the project's own existing in-tree uses are all far below this bound, per the finding's own note). 'Compiled only' for the new guards specifically; the surrounding comm_mpi.cpp code paths are exercised at normal sizes by the existing MPI test suite, whole-tree build succeeds (job 8a7a2b95 final_build.log, exit 0).

### TD-3 — fixed

Tests: `example.mms_pair.dump`, `example.dn_heat.dump`, `example.mms_pair`, `example.dn_heat`

Evidence: Early return added after the dump print in both mains. example.mms_pair/.dump rows not independently re-run this round (small, fast rows, covered by the whole-tree build only). example.dn_heat (the non-dump row, unaffected by the `if (dump) return 0` change by construction) Passed 470.18 sec and example.dn_heat.omp_identical Passed 410.30 sec (job 8114b73b), confirming the dn_heat main() restructuring did not regress the un-flagged path.

### TD-5 — fixed

Evidence: Doc-only fix, verified by reading examples/gray_scott/main.cpp's resolve_ratio(): --ratio m sets v_interval=fine(=dt), u_interval=fine*m(=the window) -- V steps at dt itself, not dt/m as the old doc claimed.

### TD-7 — fixed

Evidence: Doc-only fix, verified by reading examples/layout_driver/main.cpp:535-538 (unconditional shape(...) line) and :734-737 (the --check line).

### TD-9 — fixed

Evidence: As literally cited (docs/user/gray-scott.md:33-34, about gray_scott::GrayScottSpeciesComponent in species_component.hpp): NOT A DEFECT -- species_component.hpp:157 uses a bare Eigen::SparseLU<Eigen::SparseMatrix<double>> directly (verified by reading); the SparseLuBackend indirection the finding describes exists only in the separate MonolithicGrayScott class (monolithic_reference.hpp), one paragraph below in the same doc. Fixed the real, closely-related gap instead: added a sentence to the MonolithicGrayScott paragraph naming its own fuse_examples::SparseLuBackend (jac_solver_) indirection and the flag-dependent backend it selects.

### TD-10 — fixed

Tests: `example.demo_b`

Evidence: A ctest row was registered inside the SAME `if(FUSE_ENABLE_LOWFIVE AND FUSE_ENABLE_MPI AND LOWFIVE_HAVE_DIST)` block the build uses, invoking run_demo.sh. Verified by reading (the gate condition and run_demo.sh's own <build-dir> argument contract) plus by running: `ctest -N -R demo_b` on higgs reports 'Total Tests: 0' (job ad1b08a4, demo_b_check.log) -- confirms the row is correctly gated OFF in slot2's configuration (LowFive is not built there, per the lane brief's D14 note) rather than mis-registering. Not run end-to-end (the gate never turns on in this environment); 'compiled only' in the sense that the CMake logic itself was read and confirmed to parse/evaluate correctly, not that the demo binary pair was executed.

## Notes for the lead

Full-tree build succeeds at HEAD (job 8a7a2b95 final_build.log, exit 0) and all four static gates (check-invariants, check-layering, gen-abi --check, verify-import) are 0-violation at HEAD, run both locally and on higgs. A full ctest sweep over -L 'linalg|verification|parallel|example' (-E brusselator_runner, the known pre-existing hang) at --timeout 300 showed 9 failures TOTAL: 1 genuine TD-1 test bug (Failed, not Timeout -- fixed and reconfirmed, see TD-1's evidence) plus 8 Timeouts: verification.fsi_transient, verification.fsi_energy_diag, verification.gray_scott_reference, verification.brusselator_participants, verification.brusselator_lowfive_harness, example.dn_heat, example.dn_heat.omp_identical, example.brusselator.omp_identical. Investigated each (corrected count: 8 tests named, not 9 as originally miswritten -- fsi_transient, fsi_energy_diag, gray_scott_reference, brusselator_participants, brusselator_lowfive_harness, dn_heat, dn_heat.omp_identical, brusselator.omp_identical): verification.fsi_energy_diag Passed at 550.91s when re-run with a 600s cap (job 8114b73b); example.dn_heat/dn_heat.omp_identical (the two touched by my TD-3 change) Passed at 470/410s in the same re-run, confirming TD-3 did not regress the non---dump path; example.brusselator.omp_identical Passed at 642.10s both at the PRE-FIX baseline e6a92e4 (job eca9595a, confirming this is a pre-existing slow test, not a regression from NL-1's additive private-member cache-key change) and is simply a slow OMP-leg comparison that exceeds a 300s cap regardless of any change in this lane. FOUR tests still exceeded even a 600s cap in the same re-run -- verification.fsi_transient, verification.gray_scott_reference, verification.brusselator_participants, AND verification.brusselator_lowfive_harness (Test #219, '***Timeout 600.12 sec' in job 8114b73b's retest.log; its first GTest subcase, ExternalDeliveryMatchesInProcessRunnerJacobi, did complete and pass at 522s, but the second subcase never finished within the per-test 600s budget -- I incorrectly omitted this one from the earlier disclosure). None of these four are in my owned paths or touched by any of my 13 commits (fsi-channel is explicitly another lane's; gray_scott/brusselator_participants/brusselator_lowfive_harness source was never edited here) -- they read as pre-existing slow/contention-sensitive tests on a shared 8-core host, not a numerics-lane regression, but I could not independently confirm within this round's compute budget that any of the four eventually pass given more time. Recommend the lead/build-tools lane consider raising the default per-test ctest timeout for the host's long OMP-leg and LowFive-harness rows, or scheduling the full sweep when no sibling lane is sharing the 8 cores. TD-1's E-BRUSS-MESH-INDEX disposition and NL-5's deferral are both written out above and already recorded in-repo (docs/user/brusselator.md's State section; the trailing comment of test/verification/brusselator_refusal_coverage_test.cpp) -- the KNOWN_ISSUES.md entry and the disposition.md table are the lead's own files per the lane brief; record_text above gives the exact text for NL-5. IMPROVEMENT-tier NL-7/NL-8/NL-9/NL-13 were left untouched as out of scope per the task. A review-process note: this session's own audit caught two errors in the work before submission -- a fabricated commit-hash attribution in the NL-1 test's comment (cited a nonexistent '5d51ccf' instead of the real StokesSolver-cache fix c2c7162, corrected) and an off-by-one test-case count in a TD-3 comment (nine vs the actual ten configurations after dn_heat's picard row, corrected) -- both fixed in place; the fixes are already folded into the commits listed above, not separate patches.
