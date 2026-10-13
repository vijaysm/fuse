# Lane report — ir-schedule (fix round 2026-10-12/13)

Lane head at handback: `be229821ac4a` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- IS-3: took the doc-route allowance explicitly offered by the finding instead of extending the epoch scan (risk of false positives / missed pairing); exact rank-agreement.md text supplied in record_text for the lead or runtime lane to place, since that file is outside owned paths.
- IS-6 touches include/fuse/model/coupling_graph.hpp, outside owned paths, per that finding's own instruction to coordinate if needed; listed in files_outside_owned_paths.
- No MPI-labeled ir/schedule tests exist in this tree, so the brief's 'ctest -L mpi -R <layer>' neighbour check found zero tests (confirmed, not a skipped step).
- Did not edit KNOWN_ISSUES.md/REVIEW.md/disposition.md/docs/dev/invariants.md/conventions.md/README.md directly (not in owned paths, and not named in this lane's task) -- per-finding record_text/evidence above is the lead's input for disposition.md and KNOWN_ISSUES.md.

## Findings

### IS-1 — fixed

Tests: `schedule.plan :: ExecutionPlan.MultirateMergeSchedulesInDependenceOrder`

Evidence: Before (full-library-revert to e6a92e4, job 791b95ee): E-SCHED-ORDER 'instruction 4 (Sink "dst.f_in" at t=1) reads arc %3 ("dst.f_in/merge"), which no earlier instruction has written'. After: schedule.plan 27/27 passing (jobs cb58ba70, 644fd305). Fix has two parts, both in src/ir/lowering.cpp and src/schedule/expand.cpp: (a) merge_writer_phase_at() derives a Merge's tick phase as the max of its writers' own slab phases at that instant, instead of the boundary pass's unconditional BeforeTicks; (b) a follow-on placement bug found while building the regression fixture -- the Merge's output arc inherited the 'newest' writer's own rank_set (lower_channel() step 5b only moves type.rank_set off the writer's own component when ranks actually differ), silently scheduling the Merge against that writer's own clock instead of the field owner's; now forced to g.owner_of(f) explicitly.

### IS-2 — fixed

Tests: `schedule.plan :: ExecutionPlan.WindowMemberGatesAnOrdinaryCycleOutsideTheRegion`

Evidence: Before (acyclicity.cpp reverted to e6a92e4, job 11f17eac): r.ok()==true (wrongly accepted), saw_cycle_code==false. After: PASSED (job 11f17eac, also reconfirmed in full suite jobs cb58ba70/4b43f8e1). NOTE ON TEST DESIGN: the first version of this test (a_out plus a second, independent Provides field a_out2 on the window member) passed even with the fix reverted, because a_out2 gave the window member an ordinary Source regardless of WindowAdvance -- caught via self-review before finalizing; redesigned as a 3-component cycle (a->b->c->a) that closes back onto the window member's own a_out/WindowAdvance with no alternate ordinary Source, which the before/after above measures.

### IS-3 — deferred

Evidence: Verified by reading: the rank-agreement scan in schedule() (src/schedule/scheduler.cpp) walks OpKind::Transport only; a placement-moving Remap (ranks_differ && remap_moves in lower_channel(), src/ir/rules_channel.cpp step 5) also moves data cross-rank with no accompanying Transport, and so carries no E-SCHED-EPOCH-MISSING/-PARTIAL guarantee. Took the doc route per the finding's own allowance: extending the scan risks either missing the gap (wrong pairing) or refusing legal graphs (false positives), since unlike Transport there is no uniform post-Remap agreement point to pair against today. docs/dev/rank-agreement.md is owned by the runtime lane, not touched directly; exact insertion text is in record_text for the lead/runtime lane to place.

Record text: Insert as a new subsection in docs/dev/rank-agreement.md, directly after the "## What the compiler proves" section's two bullets (*Every data move has one.* / *It covers exactly the ranks the move covers.*) and before "## What it deliberately does not do": ## A gap the scan does not close The scan in `schedule()` (src/schedule/scheduler.cpp) that proves every data move has a following rank-agreement point walks `ExecutionPlan::instructions()` for `OpKind::Transport` only. A channel whose Remap runs at a placement other than its source or target -- `lower_channel()`'s step 5 (src/ir/ rules_channel.cpp), `ranks_differ && remap_moves` -- also moves data across ranks, as part of the Remap instruction itself, with no separate Transport instruction in the plan for the scan to find. Such a Remap carries no `E-SCHED-EPOCH-MISSING` / `E-SCHED-EPOCH-PARTIAL` guarantee: a rank-local failure inside it (an uncovered point in the interpolation, an exact conversion overflow) is discovered by that Remap's partners only when they block on the next collective, which is exactly the hang this operator exists to prevent for the Transport case. Extending the scan to recognize a placement-moving Remap as a second collective kind was not attempted in this round: unlike Transport, which is followed by a dedicated Epoch unconditionally (`lower_channel()` step 4, "after-distributed- remap"/"after-transport"), there is no uniform post-Remap agreement point to pair a placement-moving Remap against today, so a mechanical widening of the same pairing loop risks either missing the real gap (if paired against the wrong candidate) or refusing legal graphs that have no Remap-following Epoch to find (false E-SCHED-EPOCH-MISSING across the existing test suite). Closing it properly needs a dedicated agreement point emitted alongside a placement- moving Remap, the same way Transport's is, which is a channel-lowering change and belongs with whichever lane next touches `lower_channel()`'s Remap step. Review 2026-10-11 IS-3, recorded rather than fixed this round.

### IS-4 — fixed

Tests: `schedule.buffers :: Buffers.OrdinaryFieldNamedWithASlashWindowSuffixIsNotMistakenForAWindowImport`

Evidence: Before (buffers.cpp+op.hpp+lowering.cpp+lowering.hpp+rules_channel.cpp reverted to d9f5f74, job ac5856f4): FAILED -- a field legally named 'q/window' was misrouted by the old '/window' label-suffix match. After: schedule.buffers 19/19 passing. Fix: new OpAttrs::window_import_sink bool flag set structurally by emit_field_sink(), read by plan_buffers() instead of the label suffix; not printed by attrs_string() so no golden dump moved.

### IS-5 — fixed

Evidence: Doc-only: plan.hpp's plan_buffers() contract parenthetical was stale relative to buffers.hpp's actual exception contract. No test applicable (text-only correction); verified by reading both files side by side after the edit. plan.hpp's own doc build is exercised indirectly by every test that includes the header; full suite green.

### IS-6 — fixed

Tests: `schedule.window_scope :: WindowScope.UnrepresentableWindowOverIntervalRatioIsRefusedNotThrown`

Evidence: Before (coupling_graph.hpp reverted to dc70dcf, job 92c5972d): FAILED -- window_sample_capacity() threw instead of returning a refusal. After: schedule.window_scope passing. Fix: routed the division through Rational::try_div, returning a SIZE_MAX sentinel on overflow, caught by the existing E-SCHED-WINDOW-CAPACITY gate instead of propagating a C++ exception out of a Result-returning function (the project's own 'Result-returning functions never throw' convention).

### IS-7 — fixed

Tests: `schedule.plan :: ExecutionPlan.ComponentInTwoWindowRegionsIsRefused`

Evidence: Before (scheduler.cpp+plan.hpp reverted to d2fe55b, job 92c5972d): FAILED -- a component/channel claimed by two window-scoped regions was silently resolved first-wins with no diagnostic. After: schedule.plan passing. Fix: new check_window_region_overlap() in scheduler.cpp, called at the start of schedule(), emitting new error code E-SCHED-WINDOW-OVERLAP (documented in plan.hpp's error-code list).

### IS-8 — fixed

Tests: `ir.ir_op :: OpKind.EveryKindHasAName`, `ir.ir_op :: OpKind.CollectiveClassificationIsDeliberate`

Evidence: Verified by reading: is_collective()/to_cstring() in op.hpp already classified all 21 OpKind enumerators correctly, including ExternalDelivery. Only the exhaustiveness TESTS were stale -- EveryKindHasAName and the is_collective count stopped at WindowAdvance, silently excluding Scale, Epoch, Map, Reduce, Closure and ExternalDelivery from coverage, and the count was an offset from WindowAdvance (+1==20) rather than counted directly, so a NEW 22nd operator would again escape undetected. This is a 'refusal no test fires' finding per the lane brief: the fix IS the test extension, no library change. Confirmed passing in full suite (ir.ir_op, job 4b43f8e1).

### IS-9 — fixed

Tests: `ir.lowering_channel :: ChannelLowering.MergedFieldAuthorizationFollowsTheActualNewestWriterNotTheFirstDeclared`, `ir.lowering_channel :: ChannelLowering.MergedFieldAuthorizedOnTheActualNewestWriterIsAccepted`

Evidence: Before (lowering.cpp+rules_channel.cpp+lowering.hpp+op.hpp reverted to 9beca78, job 92c5972d; also full-library-revert job 791b95ee): FAILED -- E-IR-TYPE false refusal, a merged field's Sink was authorized against the first-declared writer's channel instead of the actual newest contribution. After: ir.lowering_channel passing (jobs cb58ba70, 4b43f8e1, full suite). SELF-REVIEW NOTE (full debugging history in the commit's own code comments): two prior iterations of this fix regressed other tests before landing -- (1) looking up the newest writer's channel via its producer node's own Node::origin is invalid for an identity channel (bare Source arc, no chain node), fixed with a dedicated ctx.writer_arc_channel map recorded in lower_channel(); (2) retargeting Node::origin itself to the newest writer's channel (since origin also drives claimed_by_channel() in expand.cpp) misrouted the merged Sink into that channel's own slab decomposition, breaking FiveWriterMergeFoldsToPairwiseMergesAndSchedules; fixed by decoupling the two concerns with a new OpAttrs::merge_auth_channel field used only by type_check(), leaving Node::origin unchanged (always first-declared) so scheduling is untouched.

### IS-10 — fixed

Evidence: Dead-code removal, no behavior change: the member_produces vector in group_regions_within_instants() (scheduler.cpp) was computed and never read. No test applicable; verified by reading the surrounding function and by the full schedule.* suite remaining green (nothing depended on the removed vector's side effects, since it had none observable outside the function).

### IS-11 — fixed

Tests: `ir.ir_dump :: IrDump.GraphvizIsWellFormed`

Evidence: Before (dump.hpp reverted to c922dd2, job 92c5972d): FAILED -- digraph name was 'soma_ir'. After: ir.ir_dump passing. Fix: renamed to 'fuse_ir'; also corrected 3 stale comments citing soma_initialize()/src/capi/soma_c.cpp to fuse_initialize()/impl_initialize/src/abi/participant.cpp. AUDITOR-FLAGGED CORRECTION made during this session: an early draft of one corrected comment mis-attributed R-46 as having fixed the '.soma-writable-probe' dataset name at lowfive_transport.cpp:568; corrected to clarify R-46 fixed a DIFFERENT, already-renamed LowFive probe dataset, and that the :568 instance is a separate, still-open residue tracked as TR-13 (not touched by this lane).

### IS-12 — fixed

Evidence: Doc-only: fixed 3 stale sections in docs/dev/coupling-windows.md (schedule() wiring now built, mark_regions has its window multi-pass bound, CMakeLists already registers window.cpp/window_coupler.cpp), updated the refusal-code list to include GAUSSSEIDEL-NOLAG and the new IS-7 OVERLAP code, updated plan.hpp's own error-code doc list, and fixed stale scheduler.cpp/rules_channel.cpp line citations in test/schedule/epoch_test.cpp. Verified by reading against current source line numbers; no test applicable to prose.

## Notes for the lead

All 11 in-scope findings addressed (IS-1,2,4,5,6,7,8,9,10,11,12 fixed; IS-3 deferred to the doc route, exact rank-agreement.md insertion text in that finding's record_text). IS-13 left untouched (out of scope, KNOWN R-43). FILES OUTSIDE OWNED PATHS: include/fuse/model/coupling_graph.hpp was touched for IS-6 (window_sample_capacity's try_div routing) per that finding's own explicit instruction to coordinate with the base-model lane if the fix needs that file; the change is minimal (one throwing division call site replaced) and listed below. COMMIT HISTORY NOTE: two commits ended up carrying a small amount of content belonging to a different finding than their own, discovered and corrected during a self-review pass before finalizing: a stray IS-9 code fragment had leaked into the IS-4 commit during an earlier interactive-rebase split (removed, confirmed via `git show <sha>:<file> | grep` that merge_newest_origin/writer_arc_channel now appear ONLY starting at the IS-9 commit, c922dd2); and IS-11's three comment-citation fixes and IS-12's doc fixes land as their own dedicated commits as intended -- no further bundling beyond what's described per-finding above. SELF-REVIEW CAUGHT TWO ADDITIONAL DEFECTS beyond the review's own list, both now fixed as part of IS-1/IS-9's commits and documented in those findings' evidence: (a) a Merge's output arc placement (rank_set) was inherited from whichever writer was 'newest', not the field owner, which could silently schedule the Merge against the wrong clock when writer and owner ranks matched but intervals didn't (folded into IS-1); (b) the first version of the IS-2 regression test did not actually exercise the WindowAdvance component-edge gap because an incidental second ordinary Provides field on the window member supplied its own component edge regardless of the fix -- caught by re-running the before-revert and seeing it pass when it should have failed; redesigned before finalizing. HOST CONTENTION: higgs (slot1) ran concurrently with other review lanes for most of this session; a large fraction of job submissions died with an infrastructure-level exit code (8 or 124) during or around the ctest/build step with no content-related cause -- confirmed deterministic/reproducible results whenever a job DID complete. Whole-tree, no-target `cmake --build build -j4` (rebuilding essentially everything) proved too slow to fit a single job's time budget when done repeatedly in a revert/restore loop; switched to targeted `--target <test-binary>` builds for per-finding before/after cycles and reserved the untargeted whole-tree build for a single, separate final verification job. Full Debug MPI+MOAB ctest run (ir+schedule labels, 22 tests) and whole-tree build both green at HEAD on higgs slot1 (job 4b43f8e1); no MPI-labeled ir/schedule tests exist in this tree (ctest -L mpi -R 'ir|schedule' found none, job 8a1fa190). All four static gates (check-invariants, check-layering, gen-abi --check, verify-import) at 0 violations in the local clone. `git status --porcelain` clean in both the local clone and higgs slot1.
