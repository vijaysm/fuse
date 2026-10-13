# Lane report — backend-transfer (fix round 2026-10-12/13)

Lane head at handback: `79a7959d7ce3` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- TF-5's fix was extended beyond its literal citation (MOAB-only assemble()) to also enforce RemapFlags::Conservative on the non-MOAB build_weights_via_supermesh dispatch path, since that flag was previously unchecked for ANY non-MOAB backend.
- TF-16's fix was extended beyond its literal citation (consistent_interp.hpp only) to also cover the identical defect in general_interp.hpp's 2D/3D H builder.
- TF-2's fix comment was corrected mid-round after an auditor flagged an overclaim (reconcile<T> catches per-rank failures, not cross-rank value disagreement); no deeper fix for the value-divergence hazard was attempted (out of scope / separate unclosed hazard).
- TF-8 and TF-13 are recorded as 'verified by reading'/'compiled only' rather than carrying a live test, per the task's explicit allowance for this state when a test would be disproportionately difficult to construct.
- TF-3's pre-fix failure mode is verified by reading and architectural analogy to TF-1's empirically-verified pattern, not by a clean standalone live crash/hang log (see notes_for_lead for why).
- Corrected two audit-caught report errors: static_gates.run_at now reflects an actual re-run at the true final HEAD with a clean tree, done just before this submission (not the earlier mid-stream run); and higgs_jobs' cda03cb4 entry was wrong (it ran direct mpirun --gtest_filter and captured TF-1's crash, not a ctest -R regex mismatch -- that belongs to a437fc60's prefix_ctest.log sub-step). The prior completion summary's count (13+2+3=18) was an arithmetic error; corrected breakdown sums to 17.

## Findings

### TF-1 — fixed

Tests: `parallel.nodal_remap_distributed.UnsupportedShapeSourceCellAgreesAcrossRanks (ranks 3-5)`, `parallel.nodal_remap_distributed.UnnumberedSourceVertexAgreesAcrossRanks (ranks 3-5)`

Evidence: Pre-fix: Job 387580af (higgs slot1): with the fix reverted, the fault-injecting rank throws fuse::InvariantViolation uncaught from FaultInjectingSourceMesh::global_id, i.e. aborts outside gather_global_source's own MPI_Allgatherv -- the exact per-rank-throw-before-the-collective signature TF-1 names. The script's own cleanup step could not complete past that abort on this shared host (orphaned MPI ranks outlive SIGKILL to mpirun), so no clean post-timeout exit code was captured; the crash itself is the evidence. | Post-fix: Job a437fc60 and job e939ca24 (higgs slot1): parallel.nodal_remap_distributed.1..5 all Passed (100% tests passed, 0 failed, full suite and -L mpi filtered reruns).

### TF-2 — fixed

Tests: `compiled only -- no new MPI test; see caveat`

Evidence: Pre-fix: not independently reproduced live (requires a halo-fringe-triggered per-rank classification divergence on try_boxes/intersect_boxes/fit_surface/kind-radius or orient(); not practical to construct within this round) | Post-fix: verified by reading + static gates (0 violations) + full-tree build succeeded (job e939ca24)

### TF-3 — fixed

Tests: `parallel.remap_halo_mpi.AdoptRemapHaloOneRankBadCoveringCellAgreesAcrossRanks (ranks 2-5)`

Evidence: Pre-fix: Not captured as a clean standalone log this round (the job carrying this step was cut short after reproducing TF-1's crash; see deviations). The fix follows the identical reconcile_step/reconcile_halo pattern established and empirically verified for TF-1 in the same round. | Post-fix: Job a437fc60 and job e939ca24 (higgs slot1): parallel.remap_halo_mpi.1..5 all Passed.

### TF-4 — fixed

Tests: `backend.supermesh (new cases covering y/z-aligned max-spread-axis selection and non-axis-aligned E-REMAP-NONPLANAR refusal)`

Evidence: Pre-fix: compiled only (pre-fix state reconstructable by inspection: hard-coded .x clipping mis-measures any y/z-aligned or diagonal pair) | Post-fix: Job a437fc60: backend.supermesh Passed.

### TF-5 — fixed

Tests: `backend.supermesh (new partial-overlap case via a ShiftedMeshBackend/AffineView serial fixture, exercising E-REMAP-COVERAGE on the non-MOAB build_weights_via_supermesh path)`

Evidence: Pre-fix: compiled only (reconstructable by inspection: RemapFlags::Conservative was never read on this path) | Post-fix: Job a437fc60: backend.supermesh Passed.

### TF-6 — fixed

Tests: `doc-only change, no test`

Evidence: Pre-fix: n/a (stale documentation, not a code defect) | Post-fix: verified by reading runner.cpp (~lines 690-770): build_distributed_interp_rows IS wired into Runner::Impl::resolve_remap via build_nodal_dist_remap.

### TF-7 — fixed

Tests: `doc-only change, no test`

Evidence: Pre-fix: n/a (stale documentation, not a code defect) | Post-fix: verified by reading: test/parallel/remap_halo_mpi_test.cpp's ERemapParallelRefusesAboveOneRankAndNotAtOne and BuildRemapHaloDeliversExactHaloValuesAndRoundTripsExactly tests now cover the rejection boundary and the round trip by exact identity.

### TF-8 — fixed

Tests: `none -- near-impossible to trigger MPI_Comm_size/rank failure in a test harness`

Evidence: Pre-fix: verified by reading only | Post-fix: compiled only; static gates 0 violations; full-tree build succeeded (job e939ca24)

### TF-9 — out-of-scope

Evidence: not applicable -- out of scope, recorded as text only per task brief

Record text: TF-9 (MLS column order, src/backend/mls_weights.cpp:538-549): out of scope for the transfer lane this round. MLS emits each row's columns in k-nearest-neighbour order, violating RemapWeights' documented "col: source entity, ascending within a row" contract (remap_plan.hpp:106); harmless to every current consumer but makes row-sum rounding k-order-dependent and breaks the stated COO contract. Not recorded in KNOWN_ISSUES.md by this lane; left for the lead or a future round.

### TF-10 — fixed

Tests: `backend.nodal_interp (new case covering the re-queried winning cell's corner check)`

Evidence: Pre-fix: compiled only (latent: the up-front loop already validated the same call; unreachable through any current MeshView implementation but the contract -- 'Rejects, never throws' -- is not actually enforced at that second call site) | Post-fix: Job a437fc60: backend.nodal_interp Passed.

### TF-11 — fixed

Tests: `parallel.nodal_remap_distributed (gid-range checks exercised via the new kGidTooLarge/E-RT-NODAL-DIST-GID-RANGE path; covered by the existing registered rank rows)`

Evidence: Pre-fix: verified by reading (unreachable through MOAB's int32 GLOBAL_ID; the API takes any MeshView so a non-MOAB MeshView with gid > 2^53 would silently lose precision pre-fix) | Post-fix: Job a437fc60 / e939ca24: parallel.nodal_remap_distributed.1..5 Passed.

### TF-12 — fixed

Tests: `parallel.nodal_remap_distributed (duplicate-owned-gid rejection; covered by the existing registered rank rows)`

Evidence: Pre-fix: verified by reading (apply_distributed_H silently kept the last record for a duplicated owned gid via value_of[gid]=... overwrite) | Post-fix: Job a437fc60 / e939ca24: parallel.nodal_remap_distributed.1..5 Passed.

### TF-13 — fixed

Tests: `none -- deep/rare to construct a genuine tie at different rank counts within this round`

Evidence: Pre-fix: verified by reading only | Post-fix: compiled only; static gates 0 violations; full-tree build succeeded (job e939ca24)

### TF-14 — out-of-scope

Evidence: not applicable -- out of scope, recorded as text only per task brief

Record text: TF-14 (src/backend/intx_weights.cpp:1285-1296): out of scope for the transfer lane this round. intersect_surfaces' decode accepts raw position in [0,n) as 0-based; a future/convention-different MOAB writing 1-based positions would be silently accepted off-by-one for every index except the last. Not recorded in KNOWN_ISSUES.md by this lane.

### TF-15 — fixed

Tests: `backend.supermesh (new case: every accumulated entry below kNegligible)`

Evidence: Pre-fix: compiled only (reconstructable: `double min_w = acc.empty() ? 0.0 : 1.0;` leaves min_w at its uninitialized 1.0 when acc is non-empty but every entry is dropped by the kNegligible floor) | Post-fix: Job a437fc60: backend.supermesh Passed.

### TF-16 — fixed

Tests: `backend.operator_conformance (new cases for both consistent_interp's and general_interp's H builders)`

Evidence: Pre-fix: compiled only (reconstructable: WeightsOperator::claims() read max_col_defect blindly regardless of which builder produced it) | Post-fix: Job a437fc60: backend.operator_conformance Passed.

### TF-17 — out-of-scope

Evidence: not applicable -- out of scope, recorded as text only per task brief

Record text: TF-17 (MINOR/IMPROVEMENT, src/backend/tempest_weights.cpp:435,494,506): out of scope for the transfer lane this round (IMPROVEMENT tier). Failure and order-degradation detection keys on literal substrings "EXCEPTION" / "dropping to 1st order" from TempestRemap's print stream; an upstream wording change would silently revert E-REMAP-DEGRADED to accepting a mislabelled first-order map. Not recorded in KNOWN_ISSUES.md by this lane.

## Notes for the lead

Two commit-bundling anomalies from git mechanics, both content-verified correct despite the bundling: (1) commit ba4ce542 (TF-1) also contains TF-6's header-comment fix, TF-11's gid-range guards, and TF-12's duplicate-owned-gid fix -- all land in nodal_remap_distributed.{cpp,hpp}, which I only ever staged/committed once for this file across the whole round, after all four findings' edits had already accumulated in the working tree. (2) commit 31f80258 (originally intended as TF-4 only) also contains TF-15 and TF-5, because `git commit -m ... -- <pathspec>` re-stages and commits the entire CURRENT working-tree content of the given paths, not just what was previously staged via `git apply --cached`; the commit message was amended twice to honestly document the bundling. No finding is mis-attributed in the findings table above -- each lists its real commit sha regardless of what else that commit also contains. TF-3's live pre-fix crash log could not be captured cleanly this round: the job carrying both TF-1 and TF-3 fails-before steps was cut short right after reproducing TF-1's crash, because the orphaned MPI rank processes from that crash outlived even a SIGKILL-based `timeout` wrapper on this shared, scheduler-less host, and the script never reached the TF-3 step. TF-3's fix (and its new test, which DOES pass post-fix across ranks 2-5) is architecturally identical to TF-1's already-verified reconcile_step/reconcile_halo pattern, so the disposition is 'fixed, post-fix verified live, pre-fix verified by reading + pattern analogy' rather than a live pre-fix repro. Slot1 was left in a clean, fix-restored state after this (confirmed via a separate call_command).
