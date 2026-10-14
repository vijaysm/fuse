# Lane report — base-model (fix round 2026-10-12/13)

Lane head at handback: `6270369fa969` (lane branch; commits were re-authored and rebased at integration — see disposition.md for the final shas).

## Deviations declared by the lane

- Commit granularity: BM-2+16+17 share one commit and BM-3+4+7+18 share another (interleaved edits to the same function / the task's own grouping), rather than strictly one commit per finding; every other finding has its own commit.
- BM-7's pre-fix failure signature is 'verified by reading' rather than an observed ctest FAIL: the descriptor test binary hit the unrelated BM-18 hang before reaching this test in the single combined pre-fix run.
- Did not run the sanitizer/UBSan leg, so BM-8's INT_MIN std::abs() UB claim is fixed by inspection + the E-LAG diagnostic test, not by an UBSan trap observed pre-fix.
- Did not create KNOWN_ISSUES.md/disposition.md entries or touch D14/D26 — out of this lane's owned paths per the shared brief; per-finding record_text is supplied instead for the lead's disposition table.
- BM-5 is deferred/documented, not fixed in code — flagged explicitly here and in the per-finding record (status='deferred') to avoid overclaiming; 17 of 18 in-scope findings received an actual code fix.

## Findings

### BM-2 — fixed

Tests: `YamlIo.MinimalDocumentLoadsCleanly`, `YamlIo.MissingMeshNameIsRefused`, `YamlIo.MissingRegionMeshIsRefused`, `YamlIo.MissingRegionNameIsRefused`, `YamlIo.MissingComponentNameIsRefused`, `YamlIo.MissingFieldNameIsRefused`, `YamlIo.MissingFieldRegionIsRefused`, `YamlIo.MissingCouplerNameIsRefused`, `YamlIo.MissingRanksFirstIsRefused`, `YamlIo.MissingRanksCountIsRefused`, `YamlIo.MissingChannelNameIsRefused`, `YamlIo.MissingChannelSourceIsRefused`, `YamlIo.MissingChannelTargetIsRefused`, `YamlIo.MissingIterationRegionNameIsRefused`, `YamlIo.MissingSpacetimeBasisIsRefused`, `YamlIo.WrongTypedScalarMapWhereStringExpectedIsRefused`, `YamlIo.WrongTypedScalarStringWhereNumberExpectedIsRefused`

Evidence: Pre-fix (job 6e3902da): model.yaml_io FAILED under reverted yaml_io.cpp. Post-fix (job 118358b7): model.yaml_io 100% passing (50/50 YamlIo subtests). Every required key read with node["key"].as<T>() is now guarded by require_key(); from_yaml()'s outer try/catch translates any remaining YAML::Exception (wrong-typed scalar on an optional key) into E-YAML-PARSE.

### BM-3 — fixed

Tests: `Descriptor.RejectsCorruptedChannelEnums`, `Descriptor.RejectsCorruptedFieldEnums`

Evidence: Pre-fix (job 6e3902da): both tests crashed with 'std::get: wrong index for variant' because deserialize() accepted the corrupted AccumOp/Window::Kind/TemporalBasis/Discretization/Conservation/Intent bytes and .error() was called on a successful Result. Post-fix (job 118358b7): model.descriptor 100% passing. All six enum bytes are now range-checked with E-DESC-ENUM; the silent CouplingPeriod folds in validate.hpp's channel_window() and yaml_io.cpp's emit_window() are removed (replaced by FUSE_ASSERT, unreachable through any validated path).

### BM-4 — fixed

Tests: `Descriptor.RejectsOutOfRangeIterationRegionChannelId`

Evidence: Pre-fix (job 6e3902da): r.ok() was true for a descriptor carrying an out-of-range iteration-region channel id (the std::vector::at() dereference downstream, in src/ir/regions.cpp, was not exercised by this unit test but is the documented I2 hang pattern). Post-fix: deserialize() refuses with E-DESC-CHANNEL before the id is ever stored; test passing in job 118358b7.

### BM-5 — deferred

Evidence: Verified by reading: Writer/Reader::pod() memcpy the native representation; bytes.hpp is little-endian by construction. No behavior change made (would require a version bump and golden-file churn, per the review's own fix options).

Record text: D-BM-5: the graph descriptor (include/fuse/model/descriptor.hpp, detail::Writer/Reader::pod()) is host-endian, NOT the little-endian rule fuse/base/bytes.hpp states for every other canonical encoding. Every machine this library builds on today is little-endian, so this is invisible in practice; a big-endian host or adapter would produce descriptors this project's own golden tests (and any little-endian peer) would refuse. Recorded in bytes.hpp and docs/dev/abi-design.md ("The graph descriptor's byte order"); fixing it means routing pod() through bytes::put_integer, which changes every existing golden file's bytes and therefore waits for a deliberate version bump.

### BM-6 — fixed

Tests: `Compatibility.AffineConversionRespectsTargetScale`

Evidence: Compiled and run on higgs (job 118358b7, model.compatibility passing). Constructed source {scale 1, offset 0}, target {scale 2, offset 10}: new test asserts the emitted AffineUnit step carries scale=1/2, offset=-5 (the correct (source.offset-target.offset)/target.scale), not the ported bug's un-divided -10. unit_rules()'s pre-existing AffineUnit test only ever pinned target.scale==1, where the bug is invisible.

### BM-7 — fixed

Tests: `Descriptor.RejectsNonCanonicalWireRational`

Evidence: Pre-fix (job 6e3902da, descriptor test binary hit a 300s+ timeout on the UNRELATED BM-18 count-cap test before reaching this one, so this specific test's pre-fix failure signature is 'verified by reading' the unmodified Reader::rational() rather than an observed pre-fix ctest FAIL). Post-fix (job 118358b7): passing. Checked that no golden descriptor (freeze_fixture.hpp's Rational literals: 1/4, 1/3, 1/2, 1, 1000, 3/10) is unreduced or negative-denominator, so the stricter Reader::rational() does not disturb Descriptor.MatchesTheFrozenWaistBytes, which stayed green throughout.

### BM-8 — fixed

Tests: `Validate.LagOfIntMinDoesNotTriggerUndefinedBehavior`, `Validate.LagSpanOverflowIsReportedNotThrown`

Evidence: Pre-fix (job 6e3902da): LagSpanOverflowIsReportedNotThrown THREW fuse::Refusal '[E-BASE-RATIONAL-OVERFLOW] Rational::operator*=: overflow after reduction' out of validate() (the exact escape the finding describes); LagOfIntMinDoesNotTriggerUndefinedBehavior failed its E-LAG assertion (the lag was silently mis-measured instead of refused, though UBSan was not enabled on this host so INT_MIN std::abs() UB itself was not instrumented). Post-fix (job 118358b7): model.validate 100% passing (47/47).

### BM-9 — fixed

Evidence: Doc-only; verified by reading against the pre-existing test Capabilities.ZeroDecodesToUndeclaredNotBitwise (passing throughout, job 118358b7 model.capabilities) which already pins the code the prose now matches.

### BM-11 — fixed

Evidence: Doc-only comment correction; compiled on higgs (job 118358b7, model.coupling_graph passing). No test needed — nothing behavioral changed.

### BM-12 — fixed

Tests: `YamlIo.PreviouslyUnsupportedUnitSymbolsNowParse`, `YamlIo.CompoundUnitExpressionParses`, `YamlIo.PreviouslySupportedUnitSymbolsStillParseTheSameWay`

Evidence: Post-fix (job 118358b7, model.yaml_io 100%): 'Hz' and the compound 'm/s' (neither in the old 10-entry kUnitTable) now parse via parse_unit_expression(); the 10 previously-supported symbols (tested with Pa) parse to the identical Units value, so units_node()'s emission table and every existing round-trip stay byte-identical. A follow-up bug this same test campaign caught: the E-YAML-UNITS message builder called fn["units"].as<std::string>() unconditionally, which threw when units was a MAP failing BM-15's new exponent check, turning that failure into E-YAML-PARSE; fixed in the same commit (rebased in) after job 3ef9360d caught it as the one remaining failure.

### BM-13 — fixed

Tests: `Compatibility.ConservationEnforcementRequestIsEmittedNotDropped`

Evidence: Pre-fix (job 6e3902da): model.compatibility SEGFAULTED — the test's result.chain.steps[0] indexed an EMPTY chain, because check_constraints() silently dropped the explicit ConservationEnforcement request with no step and no diagnostic (worse than the review's own description: a hard crash, not just silence). Post-fix (job 118358b7): model.compatibility passing; the step is now emitted and reaches the same E-IR-UNLOWERABLE refusal Clip already does via ir_can_emit() (src/ir/lowering.cpp, unmodified, outside my owned paths — verified by reading that it already returns false for both Clip and ConservationEnforcement).

### BM-14 — fixed

Tests: `RankSet.RefusesAnAbsurdlyLargeCount`, `RankSet.RefusesFirstPlusCountOverflow`, `RankSet.RefusalCarriesTheDocumentedCode`, `RankSet.AnOrdinaryLargeCountStillWorks`

Evidence: Pre-fix (job 6e3902da): model.component FAILED (EXPECT_THROW did not see fuse::Refusal; the unguarded reserve()/loop just ran). Post-fix (job 118358b7): model.component 100% passing. RankSet(first,count) now refuses (E-RANKSET-RANGE) a count above 2^24 or a first+count overflow before touching the vector.

### BM-15 — fixed

Tests: `YamlIo.OutOfRangeUnitExponentIsRefused`

Evidence: Post-fix (job 118358b7, model.yaml_io 100%): a structural-map exponent of 300 (out of int8_t range) is refused with E-YAML-UNITS instead of silently wrapping to 44 via static_cast<int8_t>.

### BM-16 — fixed

Tests: `YamlIo.DuplicateMeshNameIsRefused`, `YamlIo.DuplicateRegionNameIsRefused`, `YamlIo.DuplicateChannelNameIsRefused`

Evidence: Post-fix (job 118358b7, model.yaml_io 100%): a second mesh/region/channel entry with an already-used name is refused with E-YAML-DUP at parse time, instead of silently overwriting the earlier entry's name-to-id resolution.

### BM-17 — fixed

Tests: `YamlIo.NonPositiveMaxIterationsIsRefused`, `YamlIo.NonPositiveConvergenceToleranceIsRefused`

Evidence: Post-fix (job 118358b7, model.yaml_io 100%): max_iterations<1 and a non-finite/non-positive convergence tolerance are now refused with E-YAML-PARSE at load, instead of surfacing only mid-sweep via ConvergenceMeasure::satisfied()'s FUSE_THROW. Negative lag is deliberately left unchanged per the task's instruction (acyclicity.cpp documents .lag(-1) as a legal lead) — not a defect, no action taken.

Record text: D14 update (negative lag): acyclicity.cpp documents .lag(-1) as a legal lead; BM-17's scope was limited to tolerance/max_iterations per the task brief, and negative lag was deliberately left as-is.

### BM-18 — fixed

Tests: `Descriptor.RejectsImpossiblyLargeComponentRankCount`, `Descriptor.RejectsImpossiblyLargeIterationRegionChannelCount`

Evidence: Pre-fix (job 6e3902da): model.descriptor TIMED OUT at 301.62s — RejectsImpossiblyLargeComponentRankCount genuinely hung (not just slow) on a 0xFFFFFFF0 wire count, exactly the hang-class risk the finding described, confirmed empirically rather than only by static reasoning. Post-fix (job 118358b7): model.descriptor passes in 0.5s; a wire count is now checked against Reader::remaining() before reserve() for component ranks, iteration-region channel ids, and window-scope policies.

### BM-19 — fixed

Tests: `CouplingGraph.WindowSampleCapacityNeverThrowsOnAnUnrepresentableRatio`, `CouplingGraph.WindowSampleTimesNeverThrowsOnAnUnrepresentableProduct`, `Multiscale.OverlapTotalRefusesAnUnrepresentableSum`

Evidence: Pre-fix (job 6e3902da): both coupling_graph tests FAILED — WindowSampleTimesNeverThrowsOnAnUnrepresentableProduct threw fuse::Refusal '[E-BASE-RATIONAL-OVERFLOW] Rational::operator*=: overflow after reduction' out of a function with no Result in its signature, exactly the no-exceptions-leg-abort shape described. Post-fix (job 118358b7): model.coupling_graph and model.multiscale both 100% passing. window_sample_capacity()/window_sample_times() now use try_div/try_add/try_sub/try_mul throughout and fall back to the already-documented harmless minimum on overflow; OverlapSpec::total() raises the SAME E-BASE-RATIONAL-OVERFLOW explicitly via try_add rather than implicitly via operator+=. NOT A DEFECT for the window_length() function itself: read in full, it does no Rational arithmetic at all (only comparisons and a direct return), so the review's claim that it 'computes with throwing Rational operators' does not hold for the current source; the real throwing arithmetic was in window_sample_capacity()/window_sample_times(), which are fixed.

Record text: D-BM-19 note: window_length() (coupling_graph.hpp) was checked against the review's claim and found clean — it performs no Rational arithmetic, only comparisons; not a defect, no change made there.

### BM-22 — fixed

Evidence: Mechanical rename (_request/_result -> request_/result_ in compatibility.cpp's Evaluation class) verified by the whole model.compatibility suite (job 118358b7) compiling and passing unchanged.

### BM-1 — out-of-scope

Evidence: Per task: the lead runs BM-1 after the lanes land (owner chose the strict direction and examples must adapt). Not touched.

### BM-20 — out-of-scope

Evidence: IMPROVEMENT tier; per task, out of scope for this round (the lead records it).

### BM-21 — out-of-scope

Evidence: IMPROVEMENT tier; per task, out of scope for this round (the lead records it).

## Notes for the lead

window_sample_capacity (BM-19) is a shared window function per the task's coordination note; I made the BM-19 change to it regardless (try_div with a documented fallback), matching window_sample_times' own fix — please merge/reconcile with the ir-schedule lane if it also touched this function. window_length() (coupling_graph.hpp) was checked against BM-19's claim and found NOT a defect: it does no Rational arithmetic (only comparisons), so no change was made there; the real throwing arithmetic the finding described was in window_sample_capacity()/window_sample_times(), both fixed. BM-17's negative-lag question was left as-is per the task's own instruction ('document the negative-lag decision in record_text rather than changing it') — acyclicity.cpp documents .lag(-1) as a legal lead; this is not new information, just confirming no change was made. KNOWN_ISSUES.md, docs/dev/reviews/2026-10-11/disposition.md, and the D14/D26 updates named in the session's desired final outputs were NOT created by this lane — the lane brief scopes this round to library fixes, tests, and per-finding record_text for the lead's disposition table; KNOWN_ISSUES.md is explicitly off-limits to this lane ('Do NOT edit KNOWN_ISSUES.md ... unless your task names them — put the text you want there in your report instead'), which is what the record_text fields above do. Commit grouping deviates from strict one-commit-per-finding: BM-2+BM-16+BM-17 share one commit (all are insertions into the same from_yaml() restructuring, genuinely interleaved at the line level) and BM-3+BM-4+BM-7+BM-18 share another (all are deserialize()'s own waist-hardening, matching the task brief's own grouping of these four under one finding entry). Every other finding has its own commit. Full list, oldest first: 39005dde BM-14, 6b0c3c47 BM-2+16+17, da804ec6 BM-3+4+7+18, ad9b3ba3 BM-12+15, 078d9100 BM-6, d74cce0b BM-8, da96b33e BM-9, 22a3c215 BM-11, f33976f1 BM-13, 962adb51 BM-19, 847cb7b5 BM-5, 6270369f BM-22. Evidence detail: higgs jobs 6e3902da (pre-fix, revert-library evidence), 3ef9360d (post-fix + full build, pre-rebase-fix), 118358b7 (final verify after rebase fix + whole-tree build): ctest -R '^model\.' with library files reverted (git diff e6a92e4 HEAD -- <lib files> | git apply -R) showed 6/21 model tests failing/crashing/hanging; restored + rebuilt -> 21/21 (100%) model.* passing including model.descriptor_mpi.1-5; whole-tree cmake --build build -j4 exit 0, ctest -L model 21/21 passing.
