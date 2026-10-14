# Lane report — partition-search (fix round 2026-10-12/13)

Lane head at handback: `bdd6dceb6b12` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- PS-9's dedicated test shares a ctest row with PS-2/PS-3's hang-class tests, so no isolated pre-fix failure signature was captured for that specific MINOR fix alone (masked by the other tests' pre-fix timeout in the same binary); verified correct by reading and by the passing post-fix/full-suite run instead.
- PS-7 and PS-11 are contract-hardening/comment corrections with no observable behavior change (ArborX already clamped k internally; the stale comment never affected generated code), so no failing-before test exists for them — reported as 'verified by reading' plus pre-existing coverage.
- PS-8 processed as out-of-scope per the task brief (record text only, no fix).

## Findings

### PS-1 — fixed

Tests: `search.SearchDistributedMpi.RefusesWhenOnlyOneRankSubmitsMixedDimensionBoxes`

Evidence: Pre-fix: ctest rows search.distributed_mpi.2..5 all ***Timeout at 120s (rank 0's mixed-dim local_boxes refused locally while peers entered ArborX::DistributedTree's collective construction -- the exact hang named). Post-fix: same rows Passed (0.4-0.8s). Job 7bd85f44-46f3-4605-b160-9333ed348894 (ctest_prefix.log / ctest_postfix.log artifacts).

### PS-5 — fixed

Tests: `search.SearchDistributedMpi.RefusesWhenRanksDisagreeAboutDimension`

Evidence: Folded into the same Allreduce reconciliation as PS-1 (same commit, same binary). Pre-fix: the binary's ctest row hung before reaching this test (PS-1's bug fires first), so no isolated pre-fix signature for THIS test alone exists -- the binary-level timeout is the honest pre-fix evidence. Post-fix: search.distributed_mpi.2..5 Passed (gtest reports nonzero on any failing case in the binary, so a row-level Pass entails this test passed). Job 7bd85f44-46f3-4605-b160-9333ed348894.

### PS-2 — fixed

Tests: `partition.RcbPartitionMpi.RefusesWhenOnlyOneRankHasAMismatchedWeightsArray`

Evidence: Pre-fix: partition.rcb_mpi.2..5 ***Timeout at 120s (rank 0's size-mismatch refusal returned while peers entered recursive_bisect's first Allreduce). Post-fix: same rows Passed. Job 7bd85f44-46f3-4605-b160-9333ed348894.

### PS-3 — fixed

Tests: `partition.RcbPartitionMpi.RefusesWhenRanksDisagreeAboutNumParts`

Evidence: Same commit/binary as PS-2; pre-fix partition.rcb_mpi.2..5 ***Timeout, post-fix Passed. Job 7bd85f44-46f3-4605-b160-9333ed348894.

### PS-4 — fixed

Tests: `partition.MeshAdapter.RefusesNoEntitiesWhenDimensionIsZero`, `partition.MeshAdapter.RefusesNoEntitiesWhenDimensionIsNegative`

Evidence: The refusal already existed in mesh_adapter.cpp (max_dim <= 0); no test in the tree made it fire, so this is a test-only change -- "the fix IS the test" per the review's own convention, no code behavior change to bracket with a failing-before state. partition.mesh_adapter Passed in the full suite (ctest_full.log, job 7bd85f44-46f3-4605-b160-9333ed348894).

### PS-6 — fixed

Tests: `partition.Repartition.RefusesNumPartsGreaterThanCommSizeAtKNullComm`, `partition.RepartitionMpi.RefusesNumPartsLessThanCommSize`

Evidence: Pre-fix: partition.repartition ***Failed (wrong code -- E-MESH-MIGRATE-RANGE instead of E-PART-REPARTITION-NUM-PARTS for num_parts > comm_size) and partition.repartition_mpi.2..5 ***Failed (num_parts < comm_size silently accepted, result.ok()==true). Post-fix: all Passed. Job 7bd85f44-46f3-4605-b160-9333ed348894.

### PS-7 — fixed

Tests: `search.BoxIndex.KNearestCapsAtIndexSize (pre-existing)`

Evidence: query_nearest now caps k explicitly (std::min(k, size())) instead of delegating to ArborX::nearest's own clamp. ArborX already clamped internally, so this is a contract-hardening change with NO observable behavior difference today -- no failing-before state exists to bracket. search.box_index Passed in the full suite (ctest_full.log).

### PS-9 — fixed

Tests: `partition.RcbPartitionMpi.DuplicateGidTiesMatchOracleAtTheSameRankCount`

Evidence: std::sort -> std::stable_sort at both tie-sort sites in rcb.hpp, mirrored identically in rcb_oracle.hpp (and its wrong "local pre-sort provably doesn't affect the decision" rationale corrected). Post-fix: PASSED at every rank count 1..5 (partition.rcb_mpi.1..5, ctest_full.log). Pre-fix: the same ctest row hung on PS-2/PS-3's bug before reaching this test's assertion, so no isolated pre-fix failure signature was captured for this specific MINOR fix -- reported as a PROVISIONAL gap, not a verified fails-before state; the fix is verified correct by reading plus the passing post-fix run.

### PS-10 — fixed

Tests: `partition.PartitionRefusals.NegativeWeightIsRefused`, `partition.RcbPartitionMpi.RefusesWhenOnlyOneRankHasANegativeWeight`

Evidence: Pre-fix: partition.refusals ***Failed (negative weight silently accepted at kNullComm) and partition.rcb_mpi.2..5 ***Timeout (folded into the same Allreduce as PS-2/PS-3 -- a one-rank-bad negative weight would otherwise strand peers). Post-fix: all Passed. Two Kokkos reducer bugs were found and fixed during this validation (see notes_for_lead) -- Kokkos::Max<int>'s IDENTITY (not 0) is what a non-combining thread/empty-range contributes, which the initial implementation misread as "negative weight found" for every legitimate all-non-negative and empty-weights call. Job 7bd85f44-46f3-4605-b160-9333ed348894.

### PS-11 — fixed

Evidence: Doc-only: box_index.cpp/.hpp's "only translation unit that names an ArborX type" corrected to "one of the two" (the other is distributed.cpp). Verified by reading; no runtime test applicable to a comment correction.

### PS-12 — fixed

Evidence: Doc-only, in the shared test/CMakeLists.txt (explicitly authorized by the lane brief): corrected the omp-legs comment's false claim of floating-point accumulation in median.hpp's histogram (Weight is int64_t; Kokkos::atomic_add on it is exact). Verified by reading; static gates (check-invariants, check-layering, gen-abi --check, verify-import) all 0 violations on the final tree.

### PS-8 — out-of-scope

Evidence: Not processed: explicitly named out-of-scope for this lane by the task brief.

Record text: PS-8 (MINOR, latent, out of scope this round): rcb.hpp's tie-path Allgatherv counts (`static_cast<int>(local_gids.size())`) and collectives.cpp's `allgatherv_i64` displacement arithmetic are `int`, narrower than the `int64` global ids RCB accepts elsewhere -- latent overflow past 2^31 on-split elements across ranks, unreachable on any realistic mesh. Part of the project-wide int-narrowing audit named in index.md section 2.5 (alongside NL-14/TR-14/NM-12); recorded here, not fixed, pending the lead's cross-layer `checked_mpi_count` pass.

## Notes for the lead

All in-scope findings (PS-1 through PS-7, PS-9 through PS-12) fixed; PS-8 recorded as out-of-scope per the task brief. Six commits on review-1011/partition-search (e6a92e4..bdd6dce), each with a test that fails (or hangs, for the rank-agreement class) before the fix and passes after -- except PS-4 (test-only, refusal pre-existed), PS-7/PS-11 (doc/contract hardening with no observable behavior change), and PS-12 (comment-only). PS-9's pre-fix signature is a provisional gap: its dedicated test shares a ctest row with PS-2/PS-3's hang, which masks an isolated pre-fix failure for the MINOR tie-sort fix alone; the fix is still verified correct by reading and by the passing post-fix/full-suite run. IMPORTANT -- self-caught regression, now fixed: my first implementation of PS-10's negative-weight check used `Kokkos::parallel_reduce(..., Kokkos::Max<int>(any_negative))` and read the combined result as 0-safe by default. Two bugs followed from the same misunderstanding of Kokkos::Max's IDENTITY value (a large negative sentinel, not 0): (a) a thread that never sees a negative weight left its local accumulator at the identity instead of explicitly combining 0, and (b) an EMPTY weights view (`extent(0) == 0`) skips the kernel entirely but the reducer still writes its identity into the output. Both misread every ordinary (non-negative or empty) call as 'found a negative weight' and refused E-PART-WEIGHT on EVERY call to rcb_partition/repartition, including the library's own pre-existing tests (bit-parity oracle suite, repartition identity test). Caught by the full-suite run on higgs before this report was written, not by static analysis or local review -- fixed by making every thread explicitly compute and max-combine its flag, and by guarding the whole reduce behind `weights.extent(0) > 0`. Final full-suite run (job 7bd85f44-46f3-4605-b160-9333ed348894) is clean: 28/28 targeted tests and 23/23 full partition+search-label tests pass; this is the evidence in the saved ctest_postfix.log/ctest_full.log artifacts, not the earlier (buggy) intermediate runs, which are not included. D14/D26 and KNOWN_ISSUES.md: not touched (outside this lane's owned paths); no partition/search finding in my list maps to either. Disposition-table text for the non-simply-fixed findings (PS-8) is in that finding's record_text above. Whole-tree build (`cmake --build build -j4`, no target) succeeded on higgs at this commit (build_full.log artifact); the full `-L 'partition|search'` ctest selection (23 tests, covering every partition/search gtest and MPI row including the pre-existing bit-parity oracle suite and the omp legs) is 100% green, confirming nothing in this lane's changes broke a neighbouring test.
