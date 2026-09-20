<!-- Third pass (final), review-2026-09-19-fixes, HEAD ced1945. Layer group: num, linalg, field. -->

# Final pass: `num`, `linalg`, `field` (FUSE, HEAD `ced1945`, branch `review-2026-09-19-fixes`)

Scope, unchanged from the first two passes: `include/fuse/num/**`, `src/num/**`, `test/num/**`,
`include/fuse/linalg/**`, `src/linalg/**`, `test/linalg/**`, `include/fuse/field/**`, `test/field/**`.
First review: `docs/dev/reviews/2026-09-19/num-linalg-field.md` (HEAD `bdf4346`). Second-pass review:
`docs/dev/reviews/2026-09-19/pass2/re-num-linalg-field.md` (HEAD `dbdc872`). Second fix report:
`docs/dev/reviews/2026-09-19/pass2/fix2-num-linalg-field.md` (started at HEAD `1565cdf`).

**A note on the filenames given in this task.** The task named the inputs
`docs/dev/reviews/2026-09-19/re-num-linalg-field.md.md`,
`docs/dev/reviews/2026-09-19/pass2/re-re-num-linalg-field.md.md` and
`docs/dev/reviews/2026-09-19/pass2/fix2-re-num-linalg-field.md.md`. None of those three paths exist in
the repository; the files that do exist, and that this report is built from, are the three named
above (`num-linalg-field.md`, `pass2/re-num-linalg-field.md`, `pass2/fix2-num-linalg-field.md`) — no
`re-` prefix, single `.md`. Confirmed by directory listing before doing anything else. This is a
filename mismatch in the task text, not a missing artifact.

**Build.** `cmake --preset tier0 -S <repo> -B $PWD/b0 -DCMAKE_PREFIX_PATH="/opt/kokkos;/opt/homebrew"`
(the two-path list is what `compute_details`/prior sessions record for this sandbox), then
`cmake --build $PWD/b0 -j4`. Configured and built clean (fmt 12.2.0, Kokkos 5.1.0, Eigen 5.0.1 via the
`eigen` linalg backend, GTest 1.17.0, 64/70 tests registered — the rest gated by tier options that are
OFF). All owned-layer and boundary-adjacent test binaries ran:

| binary | tests | result |
|---|---|---|
| `fuse_field_pipeline_test` | 13 | PASSED |
| `fuse_linalg_least_squares_test` | 6 | PASSED |
| `fuse_num_temporal_basis_test` | 22 | PASSED |
| `fuse_num_waveform_test` | 40 | PASSED |
| `fuse_num_waveform_order_test` | 12 | PASSED |
| `fuse_num_accelerator_test` | 54 | PASSED |
| `fuse_num_kernels_test` | 19 | PASSED |
| `fuse_num_residual_norm_test` | 7 | PASSED |
| `fuse_num_space_kernels_test` | 12 | PASSED |
| `fuse_model_iteration_test` | 21 | PASSED (R-49's new home; outside my owned paths but the direct check for that finding) |
| `fuse_runtime_field_ref_test` | 34 | PASSED (outside my owned paths; the direct regression check for R-08/R-29) |

185 owned-layer tests plus the two boundary-adjacent files (55 tests), 0 failures. `tools/check-layering.py` and
`tools/check-invariants.py` both report 0 violations. `tools/mutate.py --check-anchors --build-dir
$PWD/b0` reports **95 valid, 0 stale** (the pass-2 fixer's own manifest reported 91; the extra 4 are
other fixers' later anchors, landing after `fix2-num-linalg-field.md` was written, not this layer
group's business).

No file under the repository was edited, no build directory was created under it, and no `git`
command that changes state was run. My build directory, mutation-testing scratch copies and all
generated binaries live under my own workspace and `/tmp`.

**Method beyond re-running the suite.** For the two items with the weakest verification in the pass-2
review (the reviewer's own words: "I did not execute the mutation... out of scope for a read-only
pass"), I went one step further this pass: I copied only the two affected headers
(`fuse/num/temporal_basis.hpp`, `fuse/num/waveform.hpp`) into a scratch include directory under my own
workspace, applied each mutation's exact text there, and compiled and ran a small standalone program
against the mutated copy, linked against the real, unmodified `libfuse-coupler-base.a` from my tier0
build — a real compile-and-execute, not an inspection. This does not touch the repository (the mutated
copy lives entirely under my workspace) and stays inside the read-only rule.

## Verification

| id | verdict | evidence |
|----|---------|----------|
| R-48 (IMPROVEMENT) | **CONFIRMED-FIXED** | Read `include/fuse/num/waveform.hpp:477-585`: all four per-element loops (`evaluate_dg`, `evaluate_spline`, `integrate_dg`, `integrate_spline`) now hoist `coeffs`/`rhs_scratch` (resp. `moments`/`r_scratch`) outside their loops and call `solve_polynomial_fit_into`/`solve_cubic_spline_moments_into` — the disclosed per-element allocation the pass-2 review flagged as remaining is gone from all four, not just the two the new test exercises (see N-num-linalg-field-2 below for that gap). Confirmed empirically, not by inspection: copied `waveform.hpp` to a scratch include directory, reverted `evaluate_dg`'s hoisting to the pre-fix form (bare `solve_polynomial_fit` call inside the loop), compiled and ran a standalone allocation-counting program (the same technique as `test/num/waveform_test.cpp`'s `AllocationCounter`) against the mutant, linked against the real `libfuse-coupler-base.a` — result `few=10, many=806` (allocation count scales with element count), exactly matching the fix report's own reproduction of the mutant. Against the unmutated header the same program gives `few=8, many=8`. `Waveform.DgAndSplineEvaluateAllocationIsIndependentOfElementCount` (`test/num/waveform_test.cpp:613`) passes at HEAD; `fuse_num_waveform_test` 40/40. |
| R-49 (DEVIATION) | **CONFIRMED-FIXED** | `test/num/convergence_test.cpp` no longer exists on disk (`ls` fails, "No such file or directory"). `test/model/iteration_test.cpp` exists, includes only `<fuse/model/iteration.hpp>` and `<fuse/base/error.hpp>` (no `fuse/num` symbol), and is registered via `fuse_add_gtest(model iteration THROWS)` at `test/CMakeLists.txt:118`; the `num` section's `foreach(t temporal_basis waveform accelerator)` (`test/CMakeLists.txt:153`) no longer lists `convergence`. `cmake --preset tier0` configures cleanly and reports "64 registered of 70 on disk" with no `FATAL_ERROR` from the every-test-registered audit that a leftover unregistered `test/num/convergence_test.cpp` would have tripped. `fuse_model_iteration_test` builds and passes 21/21, matching the pre-move `fuse_num_convergence_test` count reported at every prior pass. The deletion and registration are two separate lead commits (`daa5fdc`, `08848ec`) built on the fixer's own `test/model/iteration_test.cpp` (`a4e7722`) — exactly the split the fixer's report asked for under "Needs another owner: delete". |
| N-num-linalg-field-1 (MINOR) | **CONFIRMED-FIXED** | `include/fuse/num/temporal_basis.hpp:294-313` (`fit_polynomial`) and `:463-474` (`cubic_spline_moments`) now check `f`'s length (and, for `fit_polynomial`, the empty-`t` case) *before* calling into `prepare_*`/`solve_*`, restoring the pre-split order. Verified against the actual pre-split source, not memory: `git show fd5bbc2:include/fuse/num/temporal_basis.hpp` confirms the original order was no-samples → length → degree for `fit_polynomial`, length → monotonicity for `cubic_spline_moments`, matching the restored order exactly. Confirmed empirically: built a standalone program against the real header calling `fit_polynomial({0,1,2}, {1,2}, degree=-1, 0.0)` (the finding's own combined-error case) — it throws "t and f have different lengths", not "negative degree". Reverting the two wrapper checks in a scratch copy (the mutation `fit-polynomial-drops-the-pre-split-length-check`) and rerunning the same program flips the result to "negative degree -1", confirming the mutation is killed by the restored order. The three new tests (`FitPolynomialReportsTheLengthMismatchBeforeTheDegreeError`, `CubicSplineMomentsReportsTheLengthMismatchBeforeTheMonotonicityError`, `FitPolynomialStillReportsNoSamplesAheadOfALengthMismatch`, `test/num/temporal_basis_test.cpp:375-424`) each construct a genuine two-error input, catch the specific exception type, and assert on message substrings in both directions (the expected message present AND the wrong one absent) — not vacuous. `fuse_num_temporal_basis_test` 22/22. |

No BLOCKER or MAJOR finding was open in this layer group going into this pass (the pass-2 review's own
summary: "0 rejections upheld or overturned... I found no basis to disagree with any of the four
accepted-and-fixed/documented dispositions"), so all three rows above are the complete verification
surface for this pass. R-08, R-29 and R-45 needed no re-verification: `git log dbdc872..ced1945 -- 
include/fuse/num include/fuse/linalg include/fuse/field src/num src/linalg test/num test/linalg
test/field test/CMakeLists.txt` shows exactly four commits since the second-pass review, none of which
touches `field_ref.hpp`, `pipeline.hpp`, `registry.hpp` or `least_squares.hpp` — the files those three
findings live in — so nothing about them could have regressed since the second-pass review confirmed
them fixed.

## New findings

| id | severity | file:line | title |
|----|----------|-----------|-------|
| N-num-linalg-field-2 | MINOR | `include/fuse/num/waveform.hpp:552-585`, `test/num/waveform_test.cpp:595-624` | R-48's pass-2 allocation fence covers `evaluate_dg`/`evaluate_spline` but not `integrate_dg`/`integrate_spline`, even though the same fix hoisted the same allocation out of all four loops — a regression that reintroduces the per-element allocation in either `integrate_*` function passes the entire suite |

### N-num-linalg-field-2 — MINOR — the R-48 pass-2 allocation fence does not cover two of the four functions it was fixed for

**Where:** `include/fuse/num/waveform.hpp:552-568` (`integrate_dg`) and `:570-585` (`integrate_spline`),
both hoisting `coeffs`/`rhs_scratch` (resp. `moments`/`r_scratch`) outside their per-element loop
exactly as `evaluate_dg`/`evaluate_spline` do. `test/num/waveform_test.cpp:595-624`
(`count_allocations_for_one_evaluate` and `Waveform.DgAndSplineEvaluateAllocationIsIndependentOfElementCount`),
the fence the fix2 report calls "the only way to prove the fix actually closes the allocation rather
than just adding an unused `_into` entry point."

**What is wrong.** The fix2 report's own text says the R-48 pass-2 fix "removes the allocation
regardless" for all four `Waveform` methods that had the per-element scratch problem — and the source
change is indeed identical across all four (confirmed by reading `waveform.hpp:477-585` in full: the
same three-line hoist-and-`_into` pattern appears in `evaluate_dg`, `evaluate_spline`, `integrate_dg`
and `integrate_spline`). But the allocation-count fence added to prove it, `count_allocations_for_one_
evaluate` (`test/num/waveform_test.cpp:595`), calls `w.evaluate(...)` only (line 607) — never
`w.integrate(...)`. No other test in `test/num/waveform_test.cpp` or `waveform_order_test.cpp` counts
allocations at all (`grep -n AllocationCounter` finds it declared once and used only inside this one
helper). So a future change that reintroduces a per-element allocation in `integrate_dg` or
`integrate_spline` specifically — say, someone "simplifies" one of them back to the bare
`solve_polynomial_fit`/`solve_cubic_spline_moments` call the way the paired mutation
(`waveform-evaluate-dg-reallocates-coefficients-per-element`) reverts `evaluate_dg` — would pass every
test in the suite.

**How I confirmed it.** Copied `waveform.hpp` into a scratch include directory and reverted
`integrate_dg`'s hoisting to the pre-fix form (the bare `num::integrate_polynomial(num::solve_
polynomial_fit(plan, ff), xa, xb, 0.0)` call inside the loop, mirroring exactly what the paired
mutation does to `evaluate_dg`). Recompiled `test/num/waveform_test.cpp` against the mutated header
(same compiler flags as the real build, taken from `compile_commands.json`) and relinked against the
real, unmodified `fuse-coupler-num`/`model`/`linalg`/`base` static libraries — a full rebuild of the
real test binary, differing only in the one hoisting revert. **Result: 40/40 tests pass**, including
`Waveform.DgAndSplineEvaluateAllocationIsIndependentOfElementCount`, because that test never calls
`integrate()`. This is a real compile-and-run, not an inference from reading the test body — the exact
defect class R-48 pass 2 was fixing is present in `integrate_dg` under this mutant and every test in
the binary is green regardless.

Severity is MINOR rather than IMPROVEMENT or MAJOR because: (a) the underlying `I5` violation is
**not currently present** — `integrate_dg`/`integrate_spline` are correctly hoisted at HEAD, this
finding is about a testing gap that would let a *future* regression back in silently, not a live
defect; (b) `Waveform` still has no caller anywhere in `src`/`include` outside its own header and
tests (same grep every pass in this series has run), so even a live regression here would be dormant,
not observable in a running coupler, today. It is not IMPROVEMENT because the fix pass explicitly
claims this class of regression is caught ("the only way to prove the fix actually closes the
allocation") and, for two of the four functions it touched, it is not — that is a correctness gap in
the pass's own verification story, not a nice-to-have.

**The fix.** Extend `count_allocations_for_one_evaluate` (or add a sibling
`count_allocations_for_one_integrate`) to call `w.integrate(Rational{0}, Rational{3}, out.data())`
instead of/in addition to `evaluate()`, and add `Waveform.DgAndSplineIntegrateAllocationIsIndependent
OfElementCount` (or fold both into one parameterized test over {evaluate, integrate} × {DG, Spline}).
Add two mutations to `build-evidence/mutations/num-linalg-field-pass2.toml` (or a pass-3 file, per this
project's one-fixer-one-file-per-round convention) mirroring `waveform-evaluate-dg-reallocates-
coefficients-per-element` for `integrate_dg` and `integrate_spline`, and confirm each is killed by the
new test the same way the existing one is documented as killed.

## Not found

Checked for, specifically, and did not find, beyond what the first and second-pass reviews already
ruled out:

- **A regression in R-08/R-29 (the `field`-layer `well_formed()`/`FieldSpan::from()` fixes) reachable
  from this pass's changes.** Nothing in this pass's four commits (`daa5fdc`, `08848ec`, `749b7b7`,
  `28f5a5e`) touches `include/fuse/field/**`, `src/field/**` or `test/field/**` at all — confirmed by
  `git show --stat` on each. `fuse_field_pipeline_test` still passes 13/13 and `fuse_runtime_field_ref_
  test`'s `Layout.*` cases still pass 3/3.
- **A regression in R-45 (the `least_squares_into` documentation fix) or anywhere in `linalg`.** Same
  check: no commit since the second-pass review touches `include/fuse/linalg/**` or `src/linalg/**`.
  `fuse_linalg_least_squares_test` passes 6/6.
- **A collective reachable from this scope, post-fix.** Re-ran `grep -rn "MPI_\|mpi\.h"` across the ten
  in-scope headers/sources plus `temporal_basis.hpp`/`waveform.hpp`'s new `_into` functions: nothing.
  `I2`/rank-agreement does not apply to this layer group, same conclusion as both prior passes.
- **Whether the byte-identical copy claim for `test/model/iteration_test.cpp` still holds at HEAD.**
  `diff test/model/iteration_test.cpp <(git show fd5bbc2 -- test/num/convergence_test.cpp 2>/dev/null)`
  is not directly meaningful since the old file predates that check by several commits; instead
  confirmed the file compiles and its 21 tests pass, and that it references no `fuse/num` symbol
  (`grep -n "fuse::num\|fuse/num"` finds nothing) — consistent with the fixer's and pass-2 reviewer's
  claim that this was always a pure test of `fuse/model/iteration.hpp`.
- **Whether the registration audit (`test/CMakeLists.txt`'s bottom-of-file every-source-registered
  check) is still enforced and passing cleanly**, since the second fix report disclosed a temporary
  local-only gate it used and removed while `test/num/convergence_test.cpp` was still on disk. Fresh
  `cmake --preset tier0` configure from a clean build directory reports "tests 64 registered of 70 on
  disk" with no `FATAL_ERROR` — the temporary gate is confirmed gone and the audit is confirmed
  currently satisfied without it.
- **Style/formatting regressions in the touched code** (trailing whitespace after `FUSE_THROW(...,
  "{}", ` before the message argument wraps). Present at eight call sites in `temporal_basis.hpp` and
  six in `waveform.hpp`, but present at every `FUSE_THROW` call site of this shape throughout both
  files, including ones untouched by any fix pass in this series — a pre-existing formatting habit of
  the codebase, not something this pass's commits introduced. Not raised as a finding.
- **A vacuous assertion in any of the six tests added across the two fix-pass rounds
  (`SolvePolynomialFitIntoAgreesWith...`, `SolveCubicSplineMomentsIntoAgreesWith...`,
  `FitPolynomialReportsTheLengthMismatchBeforeTheDegreeError`,
  `CubicSplineMomentsReportsTheLengthMismatchBeforeTheMonotonicityError`,
  `FitPolynomialStillReportsNoSamplesAheadOfALengthMismatch`,
  `DgAndSplineEvaluateAllocationIsIndependentOfElementCount`).** Read each body
  (`test/num/temporal_basis_test.cpp:315-424`, `test/num/waveform_test.cpp:595-624`): each constructs an
  input that would trip the specific defect, asserts a specific outcome (not just "no throw" or "ok"),
  and the buffer-reuse tests call the `_into` function twice through the same buffers specifically to
  catch stale-value leakage. None is vacuous.

Not attempted, and stated as a scope limit rather than a silent gap: I did not re-audit
`fuse::model::iteration.hpp`'s own internal logic (out of scope in all three passes, per the first
review's own note, and `test/model/**` is outside this layer group except for the one file R-49
concerns); I did not run `tools/mutate.py`'s full apply/build/test/revert cycle through the tool itself
against the checked-out tree (forbidden by this task's read-only rule, same constraint the pass-2
reviewer and the fix-2 pass both hit) — I instead built and ran real standalone reproductions of both
pass-2 mutations, and of the new mutation this pass's finding proposes, against scratch copies of only
the two affected headers, linked against the real, unmodified static libraries from my own tier0 build.

## Summary

**Verification:** 3 of 3 items carried into this pass are **CONFIRMED-FIXED**: R-48 (IMPROVEMENT — the
disclosed remaining per-element allocation is gone from all four `Waveform` fit/evaluate loops, not
only the two the new fence test exercises), R-49 (DEVIATION — the `num`→`model` test move is complete:
old file deleted, new file registered, registration audit passes cleanly), and N-num-linalg-field-1
(MINOR — the combined-error precedence regression from R-48's first pass is restored to the pre-split
order, independently confirmed against the actual pre-split source and by building and running the
paired mutation). 0 rejections upheld or overturned. R-08, R-29 and R-45 needed no re-verification:
no commit since the second-pass review touches the files they live in.

**New findings:** 1 (MINOR). N-num-linalg-field-2: the allocation-count fence added to prove R-48's
pass-2 fix closes the per-element allocation exercises only `evaluate_dg`/`evaluate_spline`, not
`integrate_dg`/`integrate_spline` — even though the identical fix was applied to all four functions.
Confirmed by building and running the real test binary against a scratch copy of `waveform.hpp` with
`integrate_dg`'s hoisting reverted: all 40 tests still pass. Not a live defect (the source is correctly
fixed for all four functions today, and `Waveform` has no caller yet), but a gap in the fix's own
verification story that would let a future regression in either `integrate_*` function back in
unnoticed.

**No BLOCKER or MAJOR finding remains open in `num`/`linalg`/`field` after this pass.** The one
finding this pass adds is MINOR, on dormant code, and every finding carried in from the first two
passes is independently confirmed fixed by re-compiling and re-running the affected code, not merely
by reading the diff.

**Build/test evidence for this pass:** tier0 configured and built clean from scratch (no cached state
reused across a daemon restart mid-build); 185 owned-layer tests plus 2 boundary-adjacent test files
(55 tests) all pass; `check-layering`/`check-invariants` both 0 violations; 95/95 mutation anchors
valid; three independent standalone compile-and-run reproductions performed under real compilation
against the actual library (the two pass-2 mutations plus this pass's proposed `integrate_dg`
mutation) — a step beyond re-running the existing suite, and beyond what either prior pass in this
series did for these two mutations specifically.
