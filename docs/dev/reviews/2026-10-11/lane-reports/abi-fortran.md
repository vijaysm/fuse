# Lane report — abi-fortran (fix round 2026-10-12/13)

Lane head at handback: `ae75851accc0` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- AB-10: documented the rank-agreement guarantee in c-abi.md as instructed, but did not add the suggested host-run 2-rank test or fix the underlying reconciliation gap in code -- the existing MPI participant fixture deliberately avoids the asymmetric call sequence this finding is about (its own header explains the hang risk of injecting it without a safety net), so I recorded the gap in KNOWN_ISSUES-ready text instead of risking an uncaught cross-rank hang in this build slot.
- AB-9's 28-row test sweep and AB-4/AB-11's own two rows all live in one new file (test/abi/refusal_sweep_test.cpp) committed together under AB-9 -- AB-4 and AB-11's library fixes are each their own commit, but their tests ride in the AB-9 commit rather than each in its own, for practical reasons (one coherent new test file). Each pairing is still verified failing-before/passing-after per finding (see evidence).
- Did not touch KNOWN_ISSUES.md, REVIEW.md, docs/dev/invariants.md, docs/dev/conventions.md, or docs/dev/abi-design.md -- not named in my task; AB-10's record_text is handed to the lead for KNOWN_ISSUES.md/disposition.md integration.

## Findings

### AB-1 — fixed

Tests: `AbiParticipant.LastErrorReportsTheFailingCallsStatus`

Evidence: Pre-fix (participant.cpp/context.hpp reverted to e6a92e4): abi.participant FAILED -- AbiParticipant.LastErrorReportsTheFailingCallsStatus (job 341ec153, ctest_prefix.log). Post-fix (restored, rebuilt): abi.participant PASSED (job 341ec153 ctest_postfix1.log; job 9bd7a68a ctest_postfix2.log and ctest_full_abi.log, 12/12 abi rows).

### AB-2 — fixed

Tests: `AbiStructSize.OldCallerFuseStepRequestIsAccepted`

Evidence: Pre-fix: abi.struct_size FAILED -- AbiStructSize.OldCallerFuseStepRequestIsAccepted (job 341ec153, ctest_prefix.log). Post-fix: abi.struct_size PASSED (job 341ec153 ctest_postfix1.log; job 9bd7a68a ctest_postfix2.log / ctest_full_abi.log).

### AB-3 — fixed

Tests: `ConformanceTest.OldPluginStepReportStructSizeIsAccepted`

Evidence: Pre-fix (provider_adapter.cpp reverted): abi.conformance FAILED -- ConformanceTest.OldPluginStepReportStructSizeIsAccepted (job 341ec153, ctest_prefix.log). Post-fix: abi.conformance PASSED, all rows including this one (job 9bd7a68a ctest_postfix2.log / ctest_full_abi.log).

### AB-4 — fixed

Tests: `AbiRefusalSweep.SetupRefusesAnMpmdComponentWithNoCommunicator`

Evidence: Pre-fix (participant.cpp reverted): abi.refusal_sweep FAILED -- AbiRefusalSweep.SetupRefusesAnMpmdComponentWithNoCommunicator (job 341ec153, ctest_prefix.log). Post-fix: abi.refusal_sweep PASSED, all 29 rows (job 9bd7a68a ctest_postfix2.log / ctest_full_abi.log). Test landed in the AB-9 sweep commit (d79fd09) alongside the library fix's own commit (a05262d).

### AB-5 — fixed

Tests: `consumer.f90's AB-5 truncation check (abi.install_consumer_fortran)`

Evidence: Pre-fix (fortran/ffuse.f90 reverted to e6a92e4): abi.install_consumer_fortran FAILED -- program output 'consumer: AB-5 regressed -- the truncated message ends in a NUL, not a real character', STOP 1, 0% tests passed (job a280a841, ctest_ab5_prefix.log). Post-fix (restored): abi.install_consumer_fortran PASSED, 100% (job a280a841 ctest_ab5_postfix.log; also job 9bd7a68a ctest_fortran_consumer.log / ctest_full_abi.log).

### AB-6 — fixed

Tests: `TD6AndAB6/ProviderAdapterRefusalTest.FiresTheExpectedCode/CapabilitiesFails`, `TD6AndAB6/ProviderAdapterRefusalTest.FiresTheExpectedCode/DescribeSuperset`

Evidence: Pre-fix (provider_adapter.cpp reverted): abi.conformance FAILED on both .../CapabilitiesFails and .../DescribeSuperset rows (job 341ec153, ctest_prefix.log). Post-fix: abi.conformance PASSED, all rows (job 9bd7a68a ctest_postfix2.log / ctest_full_abi.log). Covers both AB-6(a) capabilities()-failure refusal and AB-6(b) describe()-superset refusal.

### AB-7 — fixed

Evidence: Doc-only fix (providers.md error-code table); verified by reading against the codes provider_adapter.cpp and registry.cpp actually emit. No test applicable.

### AB-8 — fixed

Evidence: Doc-only fix (entries.def/types.h/c-abi.md prose for fuse_error::code's lifetime); verified by reading against participant.cpp's actual storage (ctx->last_error_code / thread-local, not static). Landed in the same commit as AB-1 since both correct the same fuse_error doc paragraph.

### AB-9 — fixed

Tests: `28 new AbiRefusalSweep/AbiRegisterFieldRefusal/AbiStepRefusal tests in test/abi/refusal_sweep_test.cpp (entry points: load_graph, save_graph, initialize, is_coupling_ongoing, field_handle, register_field, view_field, fuse_initialize's LowFive/in-process delivery check, begin_step, end_step, run, destroy, fuse_coupler_abi version refusal)`

Evidence: Post-fix: abi.refusal_sweep PASSED, 29/29 sub-tests (job 341ec153 ctest_postfix1.log; job 9bd7a68a ctest_postfix2.log / ctest_full_abi.log). These exercise refusal codes that were already correctly implemented but untested, per I4 'the fix IS the test' -- no library behavior changed for most of these rows (AB-4 and AB-11 rows are the exceptions, each separately confirmed failing pre-fix / passing post-fix above).

### AB-10 — deferred

Evidence: Documentation-only commit (c-abi.md, new section 'Every rank of a component must make the same calls, in the same order'). The underlying gap -- ABI-boundary refusals are decided from local state only, never reconciled across a multi-rank component's own ranks before a later collective -- is NOT fixed in code this round: doing so would mean routing every participant-ABI refusal through the same Epoch reconciliation fuse::Runner's C++ terminals use, a substantially larger change than this MINOR finding's own scope. A host-run 2-rank test was not added either: the existing MPI participant fixture (test/parallel/abi_participant_mpi_test.cpp) deliberately drives every rank symmetrically and its own header comment explains why -- injecting the asymmetry this finding is about risks a genuine, uncaught cross-rank hang in a review sandbox with no way to bound it. Recorded as open rather than silently left unaddressed.

Record text: D-new (abi): the participant ABI's per-rank refusals (E-ABI-STEP-ORDER, E-ABI-STEP-ALREADY-OPEN, version/struct-size mismatches, etc.) are decided from local state only and are never reconciled across a multi-rank component's own ranks before a later collective (resume_period()'s Transport/Epoch machinery) -- an out-of-turn or omitted call on one rank can strand its peers in that collective. Documented in docs/user/c-abi.md ('Every rank of a component must make the same calls, in the same order'); not yet covered by a host-run multi-rank test. Review 2026-10-11 AB-10.

### AB-11 — fixed

Tests: `AbiRefusalSweep.SetupRefusesANonZeroReservedField`

Evidence: Pre-fix (participant.cpp reverted): abi.refusal_sweep FAILED -- AbiRefusalSweep.SetupRefusesANonZeroReservedField (job 341ec153, ctest_prefix.log). Post-fix: abi.refusal_sweep PASSED, all 29 rows (job 9bd7a68a ctest_postfix2.log / ctest_full_abi.log). entries.def/types.h end_time doc fix verified by reading (no runtime behavior attached to that half of the commit). Test landed in the AB-9 sweep commit (d79fd09).

### AB-12 — fixed

Evidence: Readability-only dead-code removal (rolled_over was always false when OR'd); no behavior change, confirmed by reading the diff. Compiles cleanly as part of every build in this round (job 341ec153 build_full1.log; job 9bd7a68a build_postfix2.log; full-abi label 12/12 passing, job 9bd7a68a ctest_full_abi.log).

### TD-6 — fixed

Tests: `TD6AndAB6/ProviderAdapterRefusalTest.FiresTheExpectedCode/{CreateFails,BindFails,NonF64Element,WindowEchoWrong}`, `ConformanceTest.PluginSaveStateErrorIsRefused`, `ConformanceTest.PluginRestoreStateErrorIsRefused`, `ConformanceTest.ProviderAdapterRefusesWhenComponentNotInGraph`, `ConformanceTest.ProviderAdapterRefusesATableMissingARequiredEntry`, `ConformanceTest.ProviderAdapterRegisterFieldsExceptionBecomesAFailure`

Evidence: Post-fix: abi.conformance PASSED, all rows including every one of these (job 9bd7a68a ctest_postfix2.log / ctest_full_abi.log). Per I4 'the fix IS the test': these nine provider-adapter codes (E-ABI-PROVIDER-CREATE/-BIND/-WINDOW/-ELEMENT/-SAVE/-RESTORE/-GRAPH/-TABLE/-EXCEPTION) were already correctly implemented but untested anywhere in the tree; none of these rows failed in the pre-fix revert run (job 341ec153 ctest_prefix.log), confirming no library behavior changed for them -- only coverage was added.

## Notes for the lead

10 commits, e6a92e4..ae75851, covering AB-1 through AB-12 plus TD-6. Built and tested on higgs slot1 (FUSE_ENABLE_FORTRAN=ON, Debug MPI+MOAB): full abi-labeled ctest suite is 12/12 green (fortran_sizes, fortran_exports, fortran_consumer_compiles, install_consumer_fortran, entries_golden, struct_size, negotiation, participant, refusal_sweep, exported_symbols, install_consumer_c, conformance), and parallel.abi_participant_mpi (ranks 1-5) is unaffected (5/5 green) -- see the saved evidence log. Every genuine library-behavior fix (AB-1, AB-2, AB-3, AB-4, AB-5, AB-6, AB-11's validation) was confirmed failing when its own library file was reverted to e6a92e4 and passing when restored; AB-9/TD-6's many rows test already-correct, previously-uncovered refusal codes (no before/after pair applicable there, per I4's own rule). AB-7/AB-8/AB-12 are doc-only or dead-code-only changes verified by reading. AB-10 is documented but NOT fixed in code or tested under MPI -- recorded as open (record_text above) rather than silently dropped; this is the one finding in my scope still open and worth the lead's attention if the integration round wants to close it. The struct_sizes golden table did not change (no struct grew), and gen-abi.py --check / check-invariants.py / check-layering.py / verify-import.py are all 0-violation at HEAD. I did not touch KNOWN_ISSUES.md, REVIEW.md, docs/dev/invariants.md, docs/dev/conventions.md, or docs/dev/abi-design.md (my task did not name them); AB-10's suggested KNOWN_ISSUES.md entry (record_text above) and the other findings' disposition lines are for the lead's own disposition.md / KNOWN_ISSUES.md integration pass.
