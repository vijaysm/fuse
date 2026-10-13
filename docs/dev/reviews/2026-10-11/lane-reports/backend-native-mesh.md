# Lane report — backend-native-mesh (fix round 2026-10-12/13)

Lane head at handback: `6a229796794d` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- NM-13 and NM-14 ship with no executable regression test (verified by reading only) because no locale-dependent or message-text-dependent behavior is reachable from any reachable code path in this tree today -- explicitly flagged as an acceptable deviation in the task brief.
- NM-6's originally-planned end-to-end test trigger ('ambiguous combination of faces that only collide on the destination') was replaced after discovering it was geometrically impossible to construct for a planar quad; the commit was amended in place with a geometrically-safe dimension-mismatch trigger instead, keeping one commit per finding.
- NM-3/NM-4's fix design changed after the first implementation (a new GhostPattern-wrapping return type) broke the whole-tree build via an out-of-owned-path consumer (examples/brusselator/species_core.hpp); the commit was amended in place to an additive, backward-compatible API (unchanged remap_halo_pattern return type + a new remap_halo_ghost_offset accessor) instead of touching that file.
- Investigated but did not fix an unrelated test timeout (verification.brusselator_runner) discovered during whole-tree verification; confirmed via a control run against the unmodified e6a92e4 baseline that it is a pre-existing hang unrelated to any of the 15 NM-* findings or this lane's changes -- reported to the lead in notes_for_lead rather than fixed, since the test file is outside native-lane owned paths and not one of the 159 catalogued findings.
- After initial submission, amended the NM-2/NM-5 commit in place (git rebase -i) to correct a doc comment and commit message that overclaimed the scope of the pre-exchange Allreduce reconciliation; re-verified the full native ctest suite + whole-tree build on higgs post-amendment before final resubmission.
- The first post-amend verification run fetched from a stale lane.bundle instead of the regenerated one, so its pass result did not actually confirm the amended commit; corrected by regenerating lane.bundle and re-running with an explicit git rev-parse HEAD check against the expected sha, confirmed matching, before trusting the build/ctest output.

## Findings

### NM-1 — fixed

Tests: `backend.native_mesh_io / NativeMeshIo.VtkRoundTripReproducesGidsAndCoordinatesAtFullPrecision`, `backend.native_mesh_io / NativeMeshIo.WriterAssertsOnAVertexGidAtOrBeyond2Pow53`

Evidence: On higgs: reverting src/backend/native_mesh_io.cpp (and the test) to pre-fix reproduces the corruption cleanly — gid 1000003 reads back as 1000000 and the coordinate-precision assertion does not throw. After the fix (setprecision(17) + a FUSE_ASSERT on gid < 2^53), both gtest cases pass. Confirmed again at final HEAD: 100% pass, 0/21 failed across the native ctest rows.

### NM-2 — fixed

Tests: `parallel.native_halo_mpi.{2,3} / NativeHaloMpi.ADisagreeingDimensionRefusesOnEveryRankInsteadOfHanging`

Evidence: On higgs: reverting native_halo.cpp's reconciliation (plus NM-5/NM-14, same file) reproduces a clean 120s TIMEOUT on ranks 2 and 3 of the new MPI test (one rank's local dimension-mismatch refusal left peers waiting in allgather_boxes/create_from_host). After mirroring migrate()'s Allreduce(MIN) pattern both before allgather_boxes and after merge_halo_cells/create_from_host, both rows pass (~1s each). Re-confirmed at final HEAD. Commit amended in place (new sha 9dc822d) after a background review correctly flagged that the doc comment and commit message overclaimed 'before either collective' for the post-exchange merge-failure reconciliation, which actually runs after both collectives, before return, to protect the caller's next collective -- both texts now state this precisely.

### NM-3 — fixed

Tests: `backend.native_halo / (NativeHaloTest ghost-offset cases, native_halo_test.cpp:584,822)`, `parallel.native_halo_mpi.{2,3}`

Evidence: On higgs: reverting src/backend/native_backend.cpp + include/fuse/backend/native_backend.hpp produces a clean compile failure (missing member `remap_halo_ghost_offset`), the agreed pre-fix signature for this finding. Design note: the fix does NOT change remap_halo_pattern's return type (that would have broken examples/brusselator/species_core.hpp, an out-of-owned-path consumer discovered via a whole-tree build failure) — it adds a separate remap_halo_ghost_offset(source, target, dim, placement) accessor computed from the pre-replacement view's (entity_count(dim) - owned_count(dim)), leaving the GhostPattern contract backward compatible. Whole-tree build at final HEAD succeeds (100%) and verification.brusselator_weights (the actual consumer of the changed header) passes in 1.04s.

### NM-4 — fixed

Tests: `backend.native_halo (same commit/test as NM-3 — both findings are the same bookkeeping key)`

Evidence: Same commit as NM-3: the internal pattern-storage key in native_backend.cpp now includes placement, so two placements sharing (source,target,dim) no longer clobber each other's stored pattern/offset. Covered by the same native_halo_test.cpp cases and the halo_mpi rows above.

### NM-5 — fixed

Tests: `parallel.native_halo_mpi.{2,3} (same commit/tests as NM-2 — both are the same comm-validation code path)`

Evidence: Same commit and evidence as NM-2: added a comm-rank validation check alongside the existing comm-size check in build_native_remap_halo, closing the gap where backend.rank() is used for self-exclusion/wire ownership while box owners are communicator ranks.

### NM-6 — fixed

Tests: `backend.native_migration_mpi.{3,4,5} / NativeMigrationMpi.ADimensionMismatchVisibleOnlyOnTheDestinationRefusesEverywhere`

Evidence: On higgs: reverting native_migration.cpp's post-unpack reconciliation reproduces a clean 120s TIMEOUT on ranks 3/4/5. After adding the same Allreduce(MIN) pattern before resolve_vertex_ownership, all rows pass. NOTE: the originally-drafted end-to-end test for this finding (an 'ambiguous ombination of faces' trigger) was discovered to be geometrically impossible for a planar quad (for N points in convex position there are only 2N simple traversals, all forward/reverse rotations of each other) and was replaced with a geometrically-safe trigger: ranks 0/1 send a valid dim-3 Tet to rank 2, whose OWN pre-migration mesh is declared dim-2, so unpack_migrated_mesh fails E-MESH-DIM only on rank 2. Diagnosed via temporary stderr/fflush traces (discarded, not part of final history) before the test redesign; the commit was amended in place once the test design was corrected.

### NM-7 — fixed

Tests: `backend.native_generators / RefusesAHexboxSpecTooLargeForOneRankInsteadOfCrashing`, `backend.native_generators / RefusesAQuadSpecTooLargeForOneRankInsteadOfCrashing`

Evidence: On higgs: reverting native_generators.cpp reproduces the crash exactly as predicted — the hexbox case throws uncaught std::bad_alloc, the quad case throws 'cannot create std::vector larger than max_size()'. After adding checked_entity_count (kMaxGeneratedEntitiesPerRank = INT32_MAX) and changing build_grid_vertices to return Result<VertexLookup>, both tests pass with a clean E-MESH-SIZE refusal instead of a crash.

### NM-8 — fixed

Tests: `backend.native_distributed_mesh / CreateFromHostRefusesAMeshWithDimensionOutsideZeroToThree`, `backend.native_distributed_mesh / CreateFromHostRefusesAFixedShapeGivenTheWrongCornerCount`

Evidence: On higgs: reverting native_mesh.cpp's validation reproduces, respectively, a clean test failure (no E-MESH-DIM refusal) and an actual SEGFAULT (confirmed real undefined behavior, not just a theoretical OOB) for the wrong-corner-count case. After adding an E-MESH-DIM range check and an E-MESH-CONNECTIVITY corner-count check per cell, both tests pass cleanly.

### NM-9 — out-of-scope

Record text: NM-9 (MINOR, src/backend/native_geometry.cpp:120-128): an exactly-degenerate fan triangle (a collinear corner) is refused E-MESH-POLYGON-NOT-STAR-SHAPED because tri_norm == 0 degenerates the test to 0 <= 0 -> refuse, whereas the documented identity says only a sign *disagreement* should invalidate a fan triangle; flat vertices on straight polygon edges (midside nodes) are geometrically valid but currently rejected. Out of scope for the native lane per the task brief (explicitly listed as a record-only finding); not fixed in this pass. File is within native-lane owned paths (native_geometry.cpp), so this remains available for a future native-lane pass if the lead wants it picked up.

### NM-10 — fixed

Tests: `backend.native_distributed_mesh / FaceWithSameVertexSetInANonCyclicOrderIsRefused`

Evidence: On higgs: reverting native_mesh.cpp's find_or_add_face fix reproduces a clean test failure (the builder silently reuses the stored face under flip=false instead of refusing). After adding a face_mismatch_error_ member set on first ambiguous match and checked at the top of build(), the builder now returns E-MESH-POLYGON-FACE-MISMATCH and the test passes.

### NM-11 — out-of-scope

Record text: NM-11 (MINOR, src/backend/native_mesh.cpp:509-551): dim-0 Point-cell meshes are shadowed by vertex bookkeeping -- entity_count(0)/owned_count(0)/global_id(0)/centroid(0) answer *vertex* data rather than Point-cell data, and Point cells are unreachable through MeshView. The review notes no generator in the tree currently emits Point cells, so this is latent rather than reachable today. Out of scope for the native lane per the task brief (explicitly listed as a record-only finding); not fixed in this pass.

### NM-12 — out-of-scope

Record text: NM-12 (MINOR, src/backend/native_migration.cpp:170-171,275; native_halo.cpp:332; native_mesh.cpp:21,136-156): int overflow on large meshes -- per-rank byte counts narrowed to int in exchange_migration_buffers, box/vertex counts as int, builder local indices/CSR offsets as int, add_vertex returns int. The review's own cross-cutting note (index.md section 2.5) groups this with TR-14/NL-14/PS-8/TF-11 as one project-wide int-narrowing class recommending a single checked_mpi_count-style audit pass across layers, not a native-lane-local fix. Out of scope for the native lane per the task brief (explicitly listed as a record-only finding); not fixed in this pass.

### NM-13 — fixed

Evidence: No executable regression test is possible: the only numeric extraction in TokenReader's istringstreams is a single-digit dimension field, which is immune to locale-dependent formatting either way. Verified by reading: both istringstream instances in native_mesh_io.cpp now call .imbue(std::locale::classic()), matching the writer's existing locale pin and conventions.md:88-93's 'every stream it creates' rule. Static gates (check-invariants, check-layering, verify-import, gen-abi --check) and the full native ctest suite pass at final HEAD, confirming no regression.

### NM-14 — fixed

Evidence: No observable behavioral difference: the two diagnostics in native_halo.cpp were converted from std::to_string/string-concatenation to fmt::format with identical resulting message text, closing the conventions.md:52 (fmt-only) violation and the std::to_string locale-reads footgun it names. Verified by reading; static gates and the full native ctest suite pass at final HEAD.

### NM-15 — fixed

Evidence: Doc-only fix, verified by cross-reading against the primary records rather than trusting the review's own citations: KNOWN_ISSUES.md's D24 entry records the migration MPI tests' host measurement as of 2026-10-09 (not 'awaiting host run' as the stale table claimed), and native-halo.md:110 already correctly records parallel.native_halo_mpi.{2,3} passing on host 2026-10-10. native-mesh.md's 'What is verified where' table rewritten to state both facts accurately and cite D24.

## Notes for the lead

All 12 in-scope native findings (NM-1..NM-8, NM-10, NM-13..NM-15) fixed and verified on higgs with before/after evidence per finding (see each finding's `evidence` field). Final state: 10 commits on branch review-1011/native (HEAD 9b9372e5), clean tree, all static gates (check-invariants, check-layering, verify-import, gen-abi --check) at 0 violations, 100% pass on the full native ctest sweep (21/21 rows incl. omp1/omp8 and MPI ranks 2-5), and a successful whole-tree build (100%) confirming no breakage to the only discovered external consumer of the changed native-backend header, examples/brusselator/species_core.hpp (verification.brusselator_weights, its ctest row, passes in 1.04s). One item surfaced during the whole-tree verification that the lead/other lanes should be aware of: `verification.brusselator_runner` (test/verification/brusselator_runner_test.cpp, NOT an owned native-lane path) hangs indefinitely in BrusselatorRunner.PerWindowStateMatchesDriverWithStoppingRuleRemoved -- confirmed it is NOT a regression from this lane's fixes by (a) grepping the test file and finding it calls only register_default_placements on NativeMeshBackend, none of the 6 functions touched by NM-1..NM-15, and (b) a control run against the fully unmodified e6a92e4 baseline (zero native-lane changes applied) reproducing the identical hang (900s timeout, stuck at the same RUN line). This looks like a pre-existing, out-of-scope issue (not one of the 159 catalogued findings in index.md) that whichever lane/lead owns test/verification or the runtime window-coupler path should triage. NM-13 and NM-14 have no executable regression test possible (no locale-dependent or message-text-dependent behavior is reachable in-tree) -- both verified by reading per the task's explicit allowance; static gates + full suite pass confirms no regression. Out-of-scope findings NM-9, NM-11, NM-12 recorded as text only per the brief, not fixed. files_outside_owned_paths is empty: the only near-miss was examples/brusselator/species_core.hpp (a read-only consumer, not modified) discovered via the NM-3/NM-4 whole-tree build check, resolved by an additive, backward-compatible API split (remap_halo_pattern's return type is unchanged; a new remap_halo_ghost_offset accessor was added) rather than touching that file. Post-submission correction: a background review correctly flagged that native_halo.hpp's doc comment and the NM-2/NM-5 commit message both overclaimed 'before either collective' for build_native_remap_halo's post-exchange merge-failure reconciliation (it actually runs after both of this function's collectives, before return, to protect the CALLER's next collective -- not this function's own two). Fixed by rewording the header doc and amending the commit message in place via an interactive rebase (new NM-2/NM-5 commit sha 9dc822d, new branch head 6a229796794d2100eb88f6603c7b02a29a78eb90). Re-verified on higgs after the amend: full tree builds (exit 0) and all 22 relevant rows (21 native ctest rows + verification.brusselator_weights) pass. A second flagged finding, about an early-session mischaracterization of the starting native_partial.diff's content (describing its native_mesh hunks as NM-8 when they implemented NM-10), was checked and confirmed to have no effect on the delivered fix -- NM-8 and NM-10 are each implemented as separate, correctly-scoped commits matching their actual definitions, each with its own host-verified fail-before/pass-after test. Correction to the prior submission: the first post-amend verification job fetched from the stale lane.bundle (never regenerated after the rebase amend), not the amended HEAD -- flagged by review. Fixed by regenerating lane.bundle from HEAD (6a229796794d2100eb88f6603c7b02a29a78eb90), re-running the verification job with an explicit `git rev-parse HEAD` check against the expected sha before trusting any build/test output, and confirming SHA_CONFIRMED_MATCH in verify_sha.log. Whole-tree build exit 0; all 22 rows (21 native ctest rows + verification.brusselator_weights) pass at the confirmed amended head.
