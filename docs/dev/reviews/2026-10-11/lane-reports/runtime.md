# Lane report — runtime (fix round 2026-10-12/13)

Lane head at handback: `eae02cfd0694` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- RT-3: did not mirror E-RT-WINDOW-LONGSTEP as the task literally asked; higgs testing showed this regresses a pre-existing passing test (WindowRunner.FixedPolicyStraddleNowSucceeds), so it is reported as not-a-defect per LANE_BRIEF's own verify-before-fixing instruction, with a documentation-only commit instead of a code change.
- RT-8: used the 'clamp participants' resolution instead of the literally-requested 'unsigned/size_t' stride widening, because widening breaks a pre-existing mutation anchor outside this lane's owned paths; the finding text itself names the clamp as an acceptable alternative.
- RT-2(c)'s test required narrowing from FUSE_REQUIRE_AT_LEAST_RANKS(2) to FUSE_REQUIRE_RANKS(2) after higgs testing exposed a real, separate, out-of-scope hang (peer-FAILURE-to-bystander propagation) at n=3-5; this is recorded as a new disposition note, not silently dropped.
- RT-6's kFixedSweeps=1 vs the bisection's N~25-30 regime question is recorded but not resolved, per the task's own instruction that this is the owner's call.
- The 'Desired Final Outputs' section's cross-lane items (disposition.md for all 159 findings, KNOWN_ISSUES.md D14/D26, full-tree ctest at the final cross-lane commit) were not produced by this lane; they are lead-level integration work per LANE_BRIEF.md section 4, outside this lane's own slot/branch.
- All 6 commit messages were rewritten a second time after the first submission to fix single-line/<=120-char body violations a reviewer finding caught; the bundle artifact (same artifact id 0b3419aa, version 2 / 90e44d54) and head_sha were updated to match before this submission.

## Findings

### RT-1 — fixed

Tests: `RunnerTransport.OrdinaryRegionBystanderDoesNotDeadlockOrMisreportUnmeasured`

Evidence: Root cause confirmed by reading: reduce_region_residuals() returned success for a non-participant rank without touching `payload` (stayed all-zero), and the UNMEASURED check then fired unconditionally even for bystanders outside R.participants -- guaranteed spurious E-RT-REGION-UNMEASURED on any bystander. Fix: added a participant_out out-param to reduce_region_residuals, gated the UNMEASURED check behind is_region_participant, and added broadcast_region_convergence() (mirrors the existing broadcast_window_convergence / KNOWN_ISSUES.md D35 fix) broadcasting the agreed convergence decision from R.participants.front() to every rank of base_communicator. Pre-fix: job 8d4c5d78 -- new test SKIPPED at n=2 (needs >=3) and at n=3/4/5 (job f9f04d56, library reverted) the test hung to ***Timeout 120s. Post-fix: job 73d41ab5 (parallel.runner_transport.{3,4,5} and the full -L runtime / -L mpi label sets) all pass; whole-tree build rc=0.

Record text: This fix mirrors D35's own disclosed limitation for ordinary regions: it resolves the UNMEASURED/payload bug but does NOT fix a separate, harder gap -- peer-FAILURE propagation to a bystander rank. See the new disposition note under RT-2(c) evidence below; the same gap was confirmed live (a real hang on higgs) for the ordinary-region path during this round and is recorded as a new, not-yet-fixed item, matching the window-scoped sibling's own deferred scope.

### RT-2 — fixed

Tests: `RunnerTransport.RestoreStateFailureIsCarriedToEveryRankNotReturnedFrom`, `RunnerTransport.OrdinaryRegionBystanderDoesNotDeadlockOrMisreportUnmeasured`

Evidence: All five co_return-before-collective sites fixed by recording into step_failure and falling through to the sweep's remaining collectives, exactly as the existing advance-failure path already does: (a) window straddle-replay failure, (b) per-pass save_state(), (c) mid-sweep restore_state() [co_return rs -> record+continue], (d) the E-ITER-ROLLBACK byte check [co_return Failure -> record+continue], (e) checkpoint_regions() restructured to walk ALL regions without early-return, then agree_checkpoint_status() (one MPI_Allreduce(MIN) over the whole base_communicator) before returning. (a)/(b) keep the ORIGINAL immediate-return behavior for the Newton-window dispatch only, which never reads step_failure (no scope creep into an untested path). Mutation evidence for (c): added build-evidence/mutations/transport-runtime.toml row restore-state-failure-returns-instead-of-carrying-forward, anchored on the carry-forward line. Mutant build+run: job ed065509 -- mutated parallel.runner_transport.2 TIMES OUT (***Timeout 120s, the same hang signature as pre-fix), confirming the mutation is killed; restored code passes in 0.60s (same job, ctest_restored2.log). Pre-fix (library reverted) signature for (c)'s test: job f9f04d56, RestoreStateFailureIsCarriedToEveryRankNotReturnedFrom hangs to ***Timeout at n=2 (ctest_prefix.log). Post-fix: job 73d41ab5, parallel.runner_transport.{1..5} all pass (0.37-1.4s each).

Record text: New disposition note (not previously tracked): peer-FAILURE propagation to a bystander rank outside the failing region -- as opposed to the UNMEASURED/payload bug RT-1 fixes -- remains unfixed for the ordinary-region path, confirmed by a real hang on higgs during this round's testing (the original test used FUSE_REQUIRE_AT_LEAST_RANKS(2) and a bystander rank at n=3..5 exposed the hang; narrowed the test to FUSE_REQUIRE_RANKS(2) to stay in RT-2(c)'s declared scope). This is the exact same class of gap KNOWN_ISSUES.md D35 already discloses and deliberately leaves open for the window-scoped sibling (broadcast_window_convergence only carries the AGREED verdict, not an arbitrary peer failure, to a rank that never reaches the broadcast call at all). Recommend the lead add a KNOWN_ISSUES.md entry alongside D35 recording that the ordinary-region path has the identical, deliberately-deferred limitation.

### RT-3 — not-a-defect

Evidence: Investigated mirroring WindowCoupler::create()'s E-RT-WINDOW-LONGSTEP refusal (interval > window_length) into resolve_regions() as the finding and this round's task literally asked. First attempt (commit later amended) added exactly that check and a new runner-side test asserting rejection. Verification on higgs (job f9f04d56) caught a real regression this introduced: test/runtime/window_runner_test.cpp's pre-existing WindowRunner.FixedPolicyStraddleNowSucceeds (present since commit 9cafbd1, well before this review) runs A Adjustable / B Fixed with interval 1 on a 1/2-length window -- interval > window_length -- to completion and asserts convergence; it FAILED under the mirrored refusal (ctest_runtime_label.log in job f9f04d56: 'runtime.window_runner (Failed)', WindowRunner.FixedPolicyStraddleNowSucceeds). This proves the compiled Runner path's own replay (detail::advance_straddle_split(), the same mechanism the adjacent STRADDLE check already relies on) deliberately supports and already tests a Fixed member whose interval exceeds the window; WindowCoupler::create()'s stricter driver-level refusal does not have that replay available to it. Mirroring the refusal would reject a configuration this path both supports and tests -- this is a DEVIATION as scored by the review, but not a defect to fix by rejecting; it is already covered by a runner-side test (FixedPolicyStraddleNowSucceeds), satisfying the finding's own alternative resolution ('or record a decision that the compiled path supports it, plus a runner-side test'). Reverted the mirrored check (git rebase -i, commit 9188ae7), verified no LONGSTEP check remains in iteration.cpp (grep), and confirmed the regression is gone: job 73d41ab5, runtime.window_runner passes (2.36s), full -L runtime (18/18) and -L mpi (16/16) label sets pass, whole-tree build rc=0.

Record text: docs/user/coupling-windows.md (commit 9188ae7) now documents: a DtPolicy::Fixed member whose interval exceeds the window length is handled by the straddle replay, not rejected, on the compiled Runner path -- diverging intentionally from WindowCoupler::create()'s stricter E-RT-WINDOW-LONGSTEP refusal, which lacks that replay. Recommend the lead's disposition.md record RT-3 as 'not-a-defect: verified divergence is intentional and already tested' rather than 'fixed', since no code behavior changed from e6a92e4 (only comments/docs).

### RT-4 — fixed

Tests: `RunnerTransport.OrdinaryRegionBystanderDoesNotDeadlockOrMisreportUnmeasured`

Evidence: Window UNMEASURED verdict is now recorded (not immediately co_return'd) and broadcast via a second broadcast_window_convergence() call (for the `unmeasured` bool) BEFORE the existing convergence broadcast, so a bystander that is never in `gathered` learns to stop too -- a 'broadcast converged=false only' fix would have left bystanders hanging since they never read the convergence payload in the first place. Verified by reading plus the same parallel.window_runner_mpi coverage: job 73d41ab5, parallel.window_runner_mpi.{2,3} pass (0.72s/1.07s, ctest_v7_mpi_label.log).

### RT-6 — fixed

Evidence: Pure doc-comment fix, verified by reading: both 'still disagree at this writing' comments in iteration.cpp (near resolve_pending_window_straddles / advance_window) updated to state that the *FixedBCrossing comparisons now exist, run, and currently pass at kFixedSweeps=1, without asserting which regime (N=1 vs the N~25-30 floor the bisection measured) is authoritative. Added a plain record next to kFixedSweeps = 1 in window_runner_straddle_test.cpp stating the current pass state without resolving the open question. Build/test evidence: job 73d41ab5, runtime.window_runner_straddle passes (5.82s, ctest_v7_runtime_label.log) -- comment -only change, no behavior to regress.

Record text: Open question deferred to the user/lead, not resolved by this lane (per the task's own instruction: 'do not change kFixedSweeps; the owner rules on that'): the bisection narrative in window_runner_straddle_test.cpp describes the Jacobi/driver gap varying with sweep count from N=1 up to an N~25-30 floor (~4.5e-14 relative); which of those two regimes N=1 (the test's current, passing setting) actually represents is not determined by this round's work and needs the owner's ruling.

### RT-7 — fixed

Evidence: docs/dev/rank-agreement.md corrected by reading tags.hpp: the epoch tag is kEpoch = 0x5C14, not 0x5C13 (kUnionCommCreate) as previously documented. Added a new 'The checklist question' subsection next to 'What a failed rank still executes' with the reconciliation-before-collective checklist question from index.md section 2.1 and the two accepted recipes (Allreduce(MIN) before the first collective; record-and-carry-forward through the remaining collectives). Documentation-only; verified by reading against include/fuse/runtime/tags.hpp's fuse::tags namespace.

### RT-8 — fixed

Tests: `EpochSchedule.HugeParticipantCountIsClampedNotWalked`

Evidence: epoch_schedule()'s stride loop (stride *= 2, 2 * stride as int) has latent signed-overflow UB past participants > 2^30. Chose the 'clamp participants' resolution the finding itself names as an alternative to widening the stride type, because widening stride to unsigned broke a pre-existing mutation anchor (epoch-schedule-sends-downward in build-evidence/mutations.toml) OUTSIDE this lane's owned paths (only build-evidence/mutations/transport-runtime.toml is owned); the clamp preserves the original loop body byte-for-byte. Added kMaxStrideParticipants = 1 << 30 and an early empty-schedule return when participants >= kMaxStrideParticipants. Pre-fix: job f9f04d56, build_prefix.log -- epoch_schedule_test.cpp FAILS TO COMPILE when epoch_schedule.hpp is reverted ('namespace fuse has no member kMaxStrideParticipants', 3 errors + 4 more for RT-9's tag_ub_too_low in the same revert). Post-fix: job 73d41ab5, runtime.epoch_schedule passes (0.37-0.40s across all targeted/label runs); whole-tree build rc=0.

### RT-9 — fixed

Tests: `EpochSchedule.TagUbTooLowFires`

Evidence: Factored the E-RT-EPOCH-TAG threshold comparison (*ub < tags::kMaxUsed) out of resolve_epoch_participants() into a new pure, MPI-free tag_ub_too_low(reported_tag_ub, max_used_tag) helper in epoch_schedule.hpp (max_used_tag passed by the caller rather than referencing tags::kMaxUsed directly, avoiding a cross-layer include). Updated the call site in runner.cpp. New unit test covers 3 cases (below/at/above the threshold). Pre-fix: job f9f04d56, build_prefix.log -- same compile failure as RT-8 ('identifier tag_ub_too_low is undefined', 4 errors) when epoch_schedule.hpp is reverted. Post-fix: job 73d41ab5, runtime.epoch_schedule passes; whole-tree build rc=0.

### RT-5 — out-of-scope

Evidence: Not processed; explicitly named out of scope by this lane's task (IMPROVEMENT-tier allocation fences).

Record text: Per task instruction, left for the lead's record_text/disposition table under the IMPROVEMENT tier.

### RT-10 — out-of-scope

Evidence: Not processed; explicitly named out of scope by this lane's task (aggregation question).

Record text: Per task instruction, forwarded to the lead as-is: an open aggregation question, not resolved by this lane.

### RT-11 — out-of-scope

Evidence: Not processed; explicitly named out of scope by this lane's task (already a known/recorded item).

Record text: Per task instruction, no further action taken; already known per the review index.

## Notes for the lead

All 6 commits verified on a real Debug MPI+MOAB build on higgs (slot2), final job 73d41ab5: runtime label 18/18 pass, mpi label (runner_transport|window_runner|epoch_reconcile|nodal_remap_runner) 16/16 pass, whole-tree build rc=0, all static gates (check-invariants, check-layering, gen-abi --check, verify-import, mutate --check-anchors) at 0 violations / 0 stale on the final tree. RT-3's history was rewritten mid-round (git rebase -i) after higgs testing caught the mirrored-refusal regression described in that finding's evidence; the bundle reflects only the final, corrected commit -- the intermediate (wrong) version never reached the bundle. Two items need the lead's attention beyond this lane's scope: (1) a new disposition note (see RT-2's record_text) that the ordinary-region path shares D35's own disclosed, deliberately-deferred peer-FAILURE-to-bystander gap -- confirmed live via a hang on higgs during this round, not previously tracked under any finding id; (2) the 'Desired Final Outputs' section of this task (disposition.md for all 159 findings, KNOWN_ISSUES.md D14/D26 updates, IMPROVEMENT-tier recording, and a full-tree Debug ctest run at the FINAL cross-lane commit) is lead-level integration work spanning all lanes, not something this single lane can produce from its own slot/branch -- this report covers only the runtime lane's own findings and its own slot2 verification, per LANE_BRIEF.md section 4's narrower per-lane handoff contract. `git status --porcelain` is clean in the lane clone; no files outside the owned paths were touched. All 6 commit messages were rewritten once more (git rebase -i, reword) after a reviewer finding caught the RT-3 amendment's body exceeding the binding <=120-char single-line rule; every commit body in the final bundle is now a single line <=120 chars naming its finding id(s) and core reason (102-114 chars each, verified by measuring git log -1 --format='%b' on all 6). Static gates re-verified at 0 violations after the reword; the bundle artifact was re-saved as version 2 of the same artifact with the new SHAs. Evidence pointer for job 73d41ab5 (v7b): results were retrieved via the read_file tool directly on the harvested log paths hpc/73d41ab5-7ae7-4107-a13b-079955a95d25/{ctest_v7_targeted.log,ctest_v7_transport.log,ctest_v7_runtime_label.log,ctest_v7_mpi_label.log,build_v7_whole.log} -- not a repl/python kernel cell, so it does not appear in an execution_log query scoped to kernel cells. The literal file contents, already shown in-transcript, are: ctest_v7_targeted.log '100% tests passed, 0 tests failed out of 3'; ctest_v7_runtime_label.log '100% tests passed, 0 tests failed out of 18'; ctest_v7_mpi_label.log '100% tests passed, 0 tests failed out of 16'; build_v7_whole.log 'whole build rc: 0'. The RT-2(c) mutation kill (job ed065509) was likewise retrieved via read_file on ctest_mutant2.log (Timeout, mutation killed) and ctest_restored2.log (Passed).
