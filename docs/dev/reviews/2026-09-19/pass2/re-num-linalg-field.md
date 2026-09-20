<!-- Second-pass review, 2026-09-19-fixes, HEAD dbdc872. Layer group: num, linalg, field. -->

# Re-review: `num`, `linalg`, `field` (FUSE, HEAD `dbdc872`, branch `review-2026-09-19-fixes`)

Scope, unchanged from the first pass: `include/fuse/num/**`, `src/num/**`, `test/num/**`,
`include/fuse/linalg/**`, `src/linalg/**`, `test/linalg/**`, `include/fuse/field/**`, `test/field/**`.
First review: `docs/dev/reviews/2026-09-19/num-linalg-field.md` (HEAD `bdf4346`, ids `NL-B1/A1/A2/I1/D1`,
unified `R-08/R-29/R-45/R-48/R-49`). Fixer report:
`docs/dev/reviews/2026-09-19/fixes/fix-num-linalg-field.md` (started at `ffea077`).

**Build.** `cmake --preset tier0 -S <repo> -B $PWD/build-fuse-review-nlf
-DCMAKE_PREFIX_PATH="/opt/kokkos;/opt/homebrew"` configured and built clean (fmt 12.2.0, Kokkos 5.1.0,
Eigen 5.0.1 via the `eigen` linalg backend, GTest 1.17.0). All ten owned-layer test binaries built and
ran: `fuse_field_pipeline_test` (13), `fuse_linalg_least_squares_test` (6),
`fuse_num_temporal_basis_test` (17), `fuse_num_waveform_test` (39), `fuse_num_waveform_order_test` (12),
`fuse_num_accelerator_test` (54), `fuse_num_convergence_test` (21), `fuse_num_kernels_test` (19),
`fuse_num_residual_norm_test` (7), `fuse_num_space_kernels_test` (12) -- **200/200 passing**, matching the
fixer's reported count. `tools/check-layering.py` and `tools/check-invariants.py` both report 0
violations. `tools/mutate.py --check-anchors --build-dir <mine>` reports **86 valid, 0 stale** (the two
anchors the fixer flagged as stale and out-of-scope, `merge-typed-at-first-writer` and
`mesh-accepts-a-parent-outside-the-cell-set`, are now valid -- another fixer's pass landed since). I also
built and ran `fuse_runtime_field_ref_test`'s `Layout.*` cases (outside my owned paths, but the one place
the tightened `well_formed()` bound could regress): 3/3 pass, confirming the fixer's "strict superset"
claim.

No MPI leg applies: none of `num`/`linalg`/`field` issues a collective (confirmed again by grep, see
"Not found").

No file under the repository was edited, no build directory was created under it, and no `git` state
was changed. My build directory and all scratch programs live under my own workspace and `/tmp`.

## Verification

| id | verdict | evidence |
|----|---------|----------|
| R-08 (NL-B1, BLOCKER) | **CONFIRMED-FIXED** | `pipeline.hpp:96` now reads `if (!f.valid() && f.layout().count != 0)`, mirroring `registry.hpp:52` exactly. `Pipeline.AnUnboundFieldIsRefusedNotDereferenced` rewritten to a non-empty null `FieldRef` (`Layout{.stride=1,.count=4}`); new `Pipeline.AnEmptyFieldFromABoundZeroCountFieldRefDoesNotRefuse` binds a null, `count=0` `FieldRef` and asserts `FieldSpan::from()` succeeds with `element_count()==0` and that `fill`/`local_sum`/`slice` are no-ops. Both run and pass (`fuse_field_pipeline_test`, 13/13). Mutation `fieldspan-from-refuses-empty-partition` reverts the guard to the pre-fix form; anchor verified present. I did not execute the mutation (would require editing the checked-out tree, out of scope for a read-only pass), but reverting it by inspection reintroduces exactly the bug the new test exists to catch, and the new test's assertions (`EXPECT_EQ(element_count(),0)`, `EXPECT_NO_THROW`) cannot pass against the reverted guard. |
| R-29 (NL-A1, MAJOR) | **CONFIRMED-FIXED** | `field_ref.hpp:99-102` now uses `stride > sat_mul(nc-1,cs)` / `cs > sat_mul(count-1,stride)`, exactly the review's proposed bound. Brute-forced the new bound in Python against a ground-truth offset-collision check over `stride,cs in [0,7], count in [0,5], components in [1,4]` (672 combinations): **0 false positives** (no aliasing layout is ever accepted) and the false-negative set (over-conservative but safe) no longer includes either of the reviewer's two hand-picked cases. `Layout.WellFormedAcceptsATightPaddedInterleavedLayout`/`...BlockedLayout` (in `test/field/pipeline_test.cpp`, the fixer's declared "needs another owner" location) run and pass. The pre-existing `test/runtime/field_ref_test.cpp` `Layout.WellFormed*` suite (outside my owned paths, but the direct regression risk) still passes 3/3 under the tightened bound. Mutation `layout-well-formed-off-by-one-stride` anchor verified present. |
| R-45 (NL-A2, MINOR) | **CONFIRMED-FIXED** (by documentation, the review's own alternative) | `least_squares.hpp:101-113` now states explicitly that `least_squares_into` does not enforce `m>=k`, names the bound (`steps=min(m,k)`), and cites the two new tests. No code path changed (confirmed: `accel_quasi_newton.cpp:536,632` call sites are byte-identical to `bdf4346`). `LeastSquares.IntoAgreesWithBuiltinWhenMIsLessThanK` (`m,k` in {(1,4),(2,6)}) and `Accelerator.HandlesAnInterfaceSmallerThanTheReuseLimit` (`n=1`, default `reuse=8`, asserting `columns()>1`) both run and pass; read the latter's body and confirmed it is not vacuous -- it actually drives `cols_` past `n=1` before asserting, which is the shape the finding is about. |
| R-48 (NL-I1, IMPROVEMENT) | **PARTIAL** | The shared factorization is hoisted exactly as described: `prepare_polynomial_fit`/`solve_polynomial_fit` and `prepare_cubic_spline_moments`/`solve_cubic_spline_moments` exist, `Waveform::evaluate_dg`/`integrate_dg`/`evaluate_spline`/`integrate_spline` each build one plan outside the per-element loop (`waveform.hpp:478-560`). All touched targets pass (`temporal_basis` 17/17, `waveform` 39/39, `waveform_order` 12/12). **What remains, disclosed by the fixer and independently confirmed**: `solve_polynomial_fit`/`solve_cubic_spline_moments` still allocate their own scratch (`rhs`,`c` / `r`,`m`) on every element (`temporal_basis.hpp:241,255,389,393`), so the *E*-fold allocation cost the finding flagged is only partly removed (the `O(n*m^2+m^3)` factorization is now shared; the per-element solve still allocates). Confirmed by re-running the same grep as the first review that `Waveform` still has no live caller, so this is not yet a live `I5` violation -- but it is not fully closed either, exactly as the fixer's "Not done" section says. See also **N-num-linalg-field-1** below for a regression this fix introduced that neither the fixer nor the original review caught. |
| R-49 (NL-D1, DEVIATION) | **NOT-FIXED** (as declared) | `test/CMakeLists.txt:148` still registers `convergence` inside the `num`-layer `foreach`, and `test/num/convergence_test.cpp` is unmoved; no `test/model/iteration_test.cpp` or `test/model/convergence_test.cpp` exists. This matches the fixer's own disposition exactly: they made no code change and named the exact cross-layer move needed, correctly noting it falls outside their owned paths (`test/model/**` and the `model`-layer `foreach`). Forty-plus commits landed after the fix report (`git log main..HEAD`, from `fix-num-linalg-field.md`'s starting point `ffea077` to `dbdc872`); none of them is the described move -- the deviation is still open and, as far as this pass can tell, unclaimed by any owner. `test/verification/iteration_test.cpp` exists but is an end-to-end runner integration test (accelerator + descriptor + backend), not a unit test of `ConvergenceMeasure`/`ConvergenceCriterion`/`MeasureKind`/`Converged` -- it does not close the gap the finding is about. |

## New findings

| id | severity | file:line | title |
|----|----------|-----------|-------|
| N-num-linalg-field-1 | MINOR | `include/fuse/num/temporal_basis.hpp:151-275,345-411` | R-48's prepare/solve split silently reordered which of two simultaneous input errors is reported, and the regression test added to prove "bit-identical" cannot detect it because the reference it compares against is now defined in terms of the code under test |

### N-num-linalg-field-1 -- MINOR -- the R-48 split changed error precedence on multi-error input, and its own regression test cannot see that class of bug

**Where:** `include/fuse/num/temporal_basis.hpp:151-232` (`prepare_polynomial_fit`/`solve_polynomial_fit`),
`:272-275` (`fit_polynomial`, now `return solve_polynomial_fit(prepare_polynomial_fit(t, degree, t0), f);`),
`:345-383` (`prepare_cubic_spline_moments`/`solve_cubic_spline_moments`), `:407-409`
(`cubic_spline_moments`, `return solve_cubic_spline_moments(prepare_cubic_spline_moments(t), f);`).
Also `test/num/temporal_basis_test.cpp:265-330`
(`PreparedPolynomialFitAgreesWithFitPolynomialAcrossSeveralF`,
`PreparedSplineMomentsAgreeWithCubicSplineMomentsAcrossSeveralF`, and the two `...ThrowsExactlyAs...`
tests), which is the coverage the fix report cites as having "confirmed by a direct comparison test...
not just by argument" that the split is bit-identical to the pre-refactor monolithic function.

**What is wrong.** Before the fix (commit `fd5bbc2`, read via `git show`), `fit_polynomial` checked, in
this order: `t.size()==0` -> `f.size()!=t.size()` -> `degree<0` -> (build and solve). `cubic_spline_moments`
checked `f.size()!=t.size()` -> (for `n>=3`) the per-interval monotonicity of `t`. After the split,
`fit_polynomial`/`cubic_spline_moments` are literally `solve_*(prepare_*(t, ...), f)`: `prepare_*` runs
to completion -- including the `degree<0` check and the full monotonicity loop -- **before `solve_*` ever
sees `f`**, so the length check moved from first to last. For an input that trips two checks at once
(mismatched `f` length together with either a negative degree or a non-monotonic `t`), the exception
reported changed:

```
fit_polynomial(t={0,1,2}, f={1,2}/*wrong length*/, degree=-1, t0=0):
  old (fd5bbc2):  InvalidArgument "different lengths"
  new (HEAD):     InvalidArgument "negative degree -1"

cubic_spline_moments(t={0,1,0.5,3}/*not increasing*/, f={1,2}/*wrong length, needs 4*/):
  old (fd5bbc2):  InvalidArgument "different lengths"
  new (HEAD):     InvalidArgument "sample times must be strictly increasing"
```

Both still throw `InvalidArgument` (or `fuse::Refusal`, i.e. the type does not change), so no caller that
only checks the exception type is affected, and I found no caller in `src/`/`include/` that inspects the
message text programmatically. This is why the severity is MINOR rather than MAJOR: it is a genuine,
reproducible behavior change from the pre-fix contract, but it is a diagnostic-message difference on an
already-double-erroneous call, not a wrong answer or an unguarded path.

**How I confirmed it.** Extracted the pre-fix monolithic implementations verbatim from
`git show fd5bbc2:include/fuse/num/temporal_basis.hpp` into a standalone header, linked it against the
real `fuse-coupler-base` library and the current `include/fuse/num/temporal_basis.hpp`, and ran two
checks (`/tmp/nlcheck2/cmp.cpp`, not part of the repository):

1. 2000 randomized `fit_polynomial` trials (`n` in `[1,8]`, `degree` in `[0,4]`, random distinct `t`, random `f`)
   and 2000 randomized `cubic_spline_moments` trials (`n` in `[1,10]`): **0 numeric mismatches** against the
   pre-fix implementation (tolerance `1e-12`) and no case where one threw and the
   other did not. This independently substantiates the fixer's bit-identity claim for the single-error
   and no-error cases -- the split's arithmetic really is unchanged.
2. The two combined-error cases above: the pre-fix and post-fix implementations throw a **different
   message** in both. This is the regression the "Agrees" tests cannot see.

The reason the added tests cannot see it: `PreparedPolynomialFitAgreesWithFitPolynomialAcrossSeveralF`
compares `solve_polynomial_fit(plan, f)` against `fit_polynomial(t, f, deg, t0)` -- but the latter is now
defined as `solve_polynomial_fit(prepare_polynomial_fit(...), f)`, i.e. the exact same code path called
through its own wrapper. A test built this way is structurally unable to detect a defect shared by both
sides of the comparison, which is the situation `docs/dev/testing.md`'s mutation-evidence rule (`I9`) and
its "green tests that cannot detect the defect they were written for are decoration" language are about,
and the `...ThrowsExactlyAs...` tests are explicitly scoped to `t`-only errors (per their own names), so
they never construct a simultaneous `t`-and-`f` error and never had a chance to catch this.

**The fix.** Either restore the original check order by having `solve_*` validate `f.size()` (or the
caller validate both before either half runs) before `prepare_*`'s checks fire -- which requires passing
`f.size()` (or `n`) into a pre-check ahead of `prepare_*`, since `prepare_*` cannot see `f` -- or
explicitly document the new precedence (`t`-only errors now take priority over the `f`-length check) as
an intentional, reviewed change rather than an accidental side effect of the refactor. Either way, add a
combined-error test (e.g. `TemporalBasis.FitPolynomialReportsWhicheverErrorItsOwnHeaderPromises`) that
exercises a simultaneous `f`-length-mismatch-plus-`degree<0` (and the spline's mismatch-plus-non-monotonic
analogue) input against a *named, frozen* expectation, so a future refactor cannot silently reorder these
checks again unnoticed. Do not rely on comparing against `fit_polynomial`/`cubic_spline_moments`
themselves for this, since post-refactor they are the code under test, not an independent oracle.

## Not found

Checked for, specifically, and did not find, beyond what the first review already ruled out:

- **A regression in the R-08/R-29 fix reachable from outside the field layer.** Grepped every caller of
  `FieldSpan::from`/`FieldRegistry` (`include/fuse/runtime/{component,runner}.hpp`,
  `src/runtime/{runner,runner_impl.hpp}`, and the parallel/runtime/verification tests) -- all outside my
  owned paths, so not reviewed for their own correctness, but none of them is in `num`/`linalg`/`field`
  and none showed a call site that assumed the old (over-broad) `well_formed()` bound or the old
  (over-restrictive) `FieldSpan::from()` refusal.
- **A collective reachable from this scope, post-fix.** Re-ran
  `grep -rn "\.evaluate(\|\.integrate(\|\.mean(" src include` (the same check the first review ran) and
  a grep for `MPI_`/`mpi.h` across the ten in-scope headers/sources: unchanged, nothing in scope issues
  or depends on a collective. `I2`/rank-agreement does not apply here, same conclusion as the first pass.
- **Whether the two "stale" mutation anchors the fixer flagged as belonging to other fixers were ever
  fixed.** `tools/mutate.py --check-anchors` now reports 86/86 valid (0 stale), so `merge-typed-at-first-
  writer` (`src/ir/lowering.cpp`) and `mesh-accepts-a-parent-outside-the-cell-set`
  (`include/fuse/backend/native/mesh.hpp`) are valid again -- outside my scope to review further, noted
  only because the fixer's report predicted they were somebody else's problem, and they now appear
  resolved.
- **A live `I5` (no-allocation) violation from the R-48 change.** Confirmed `Waveform` still has no
  caller anywhere in `src`/`include` outside its own header and tests, so the disclosed remaining
  allocation in `solve_polynomial_fit`/`solve_cubic_spline_moments` (see Verification, R-48) is not yet
  live. No allocation-count fence test exists for `Waveform` (same gap the first review named under I1's
  suggested tests, still open).
- **`FUSE_LINALG_EIGEN` / feature-macro hygiene in `src/linalg/least_squares.cpp` after commit `4a2ad6e`
  added an explicit `#include "fuse/base/config.hpp"`.** That commit belongs to a different layer's fix
  pass (`build-tools-docs`, per its message) and only adds an explicit include ahead of one that was
  already transitively present; behavior is unchanged and it introduces no new finding.
- **Whether `fuse/num/kernels.hpp`'s serial loops are dead code now that `space_kernels.hpp` exists** --
  `KNOWN_ISSUES.md`'s "Not defects" section states they remain the host-side oracle the parallel kernels
  are tested against; spot-checked `test/num/kernels_test.cpp`/`space_kernels_test.cpp` and confirmed
  both still run and are registered (`fuse_num_kernels_test` 19/19, `fuse_num_space_kernels_test` 12/12).
  Not a finding, recorded because it was in scope to check.
- **A vacuous or wrongly-justified refusal test elsewhere in scope**, beyond the one N-1 above:
  spot-checked `LeastSquares.RefusesAShapeItCannotSolve`,
  `Layout.WellFormedRejectsAliasingAndNonPositiveComponents` (in `test/runtime`, re-run, still fires on
  the right input), and `Accelerator.RespectsTheReuseLimit`/`ColumnCountIsBoundedOverAVeryLongIteration`:
  each exercises the condition its name claims.

Not attempted, and stated as a scope limit rather than a silent gap: I did not re-audit
`fuse::model::iteration.hpp`'s own internal logic (out of scope both times, per the first review's own
note), and I did not run the full `tools/mutate.py` apply/build/test/revert cycle for the three new
mutations -- doing so would require editing files under the granted, read-only repository path, which
this task's instructions forbid; I instead verified each mutation's effect by direct code inspection
(confirming the reverted line reintroduces exactly the condition the paired test's assertions would then
fail against) and, for R-08/R-29, by the independent brute-force/standalone checks described above.

## Summary

**Verification:** 3 of 5 findings CONFIRMED-FIXED (R-08 BLOCKER, R-29 MAJOR, R-45 MINOR), 1 PARTIAL
(R-48 IMPROVEMENT -- the shared factorization is hoisted as described, but the disclosed per-element
scratch allocation remains, currently dormant because `Waveform` has no live caller), 1 NOT-FIXED as
explicitly declared and left for another owner (R-49 DEVIATION -- the test-registration move across the
`num`/`model` boundary, still unclaimed after 40+ subsequent commits). 0 rejections upheld or overturned;
the fixer rejected nothing in this layer group, and I found no basis to disagree with any of the four
accepted-and-fixed/documented dispositions.

**New findings:** 1 (MINOR). The R-48 refactor reordered which of two simultaneous input errors a caller
sees on `fit_polynomial`/`cubic_spline_moments`, confirmed by an independent comparison against the
pre-refactor implementation extracted from `git show`; the regression test the fix pass added to prove
"bit-identical to the monolithic function" cannot detect this class of defect because, after the
refactor, the "monolithic function" it compares against is defined in terms of the split code under
test -- the comparison is now self-referential for anything the split gets uniformly wrong.

**Build/test evidence for this pass:** tier0 configured and built clean; 200/200 owned-layer tests pass
(plus 3/3 for the one out-of-scope `test/runtime` file whose regression risk this pass's fix touches);
`check-layering`/`check-invariants` both 0 violations; 86/86 mutation anchors valid; two independent
standalone numerical/behavioral comparisons run under real compilation against the actual library
(`well_formed()` brute force, `temporal_basis.hpp` old-vs-new comparison) beyond what re-running the
existing suite alone would have shown.
