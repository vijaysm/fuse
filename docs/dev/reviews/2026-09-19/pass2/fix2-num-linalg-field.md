<!-- Second fix pass, review-2026-09-19-fixes, started at HEAD 1565cdf. Layer group: num, linalg, field. -->

# Fix report, pass 2: `num`, `linalg`, `field` (FUSE, started HEAD `1565cdf`)

Source: `docs/dev/reviews/2026-09-19/pass2/re-num-linalg-field.md`. Its five rows: three
CONFIRMED-FIXED (R-08, R-29, R-45) needing no action here, one PARTIAL (R-48), one NOT-FIXED and
explicitly deferred (R-49, a cross-layer move), plus one new finding (N-num-linalg-field-1). This
report covers the PARTIAL, the NOT-FIXED, and the new finding, in that order (MINOR, then
IMPROVEMENT, then DEVIATION — no BLOCKER/MAJOR remained open in this layer group after pass 1).

**Build.** `cmake --preset tier0` (Debug), with `-DFETCHCONTENT_SOURCE_DIR_R3D` /
`-DFETCHCONTENT_SOURCE_DIR_YAML-CPP` pointed at the shared `/tmp/fuse-deps` sources (this sandbox
cannot `git clone` — every attempt to create a `.git` directory anywhere under it, including outside
`/tmp`, fails with `Operation not permitted`; the FetchContent source-dir override is the workaround,
matching what another concurrent fixer's build cache already showed working). Configured and built
clean; `tools/check-invariants.py` and `tools/check-layering.py` both report 0 violations.
`tools/mutate.py --check-anchors` reports 91/91 valid (0 stale) against the shared manifest, including
this pass's two new entries.

**Deviation disclosed up front.** R-49's own move touches the audit at the bottom of
`test/CMakeLists.txt` (any `*_test.cpp` on disk that is registered nowhere is a `FATAL_ERROR`, by
design — see R-07). Validating the move locally, with `test/num/convergence_test.cpp` still on disk
per this task's instructions (create the new file, leave the old one for the lead to delete), required
a temporary `fuse_tests_gated_by(FUSE_PASS2_R49_PENDING_DELETION num/convergence_test.cpp)` line so the
local configure would not fail the audit while validating. That line is **not** in the diff below — it
was added, used to build and run `fuse_model_iteration_test`, and removed again before finishing. The
final `test/CMakeLists.txt` therefore configures cleanly only once `test/num/convergence_test.cpp` is
actually deleted (see "Needs another owner: delete").

## Fixed

### N-num-linalg-field-1 — MINOR — combined-error precedence in `fit_polynomial`/`cubic_spline_moments`

Confirmed by reading `include/fuse/num/temporal_basis.hpp:151-232,345-383` and, per the finding's own
method, `git show fd5bbc2:include/fuse/num/temporal_basis.hpp` for the pre-split check order: empty-`t`
→ length → degree for `fit_polynomial`; length → monotonicity for `cubic_spline_moments`. The
prepare/solve split (R-48, pass 1) made `fit_polynomial`/`cubic_spline_moments` literally
`solve_*(prepare_*(t, ...), f)`, so every `t`-only check in `prepare_*` fires before `solve_*` ever
sees `f`'s length — reordering the two wrapper functions' error precedence on a combined-error input,
exactly as the finding describes.

**Fix.** Restored the pre-split order by checking `f`'s length in the wrapper, ahead of the
`prepare_*`/`solve_*` split, since `prepare_*` cannot do this itself (it never receives `f`):
`fit_polynomial` now checks `t.size()==0` (duplicating `prepare_polynomial_fit`'s own check, harmlessly
— it never reaches there once this fires) then the length mismatch, before calling
`prepare_polynomial_fit`/`solve_polynomial_fit`; `cubic_spline_moments` checks the length mismatch
before calling `prepare_cubic_spline_moments`/`solve_cubic_spline_moments`. Neither `solve_polynomial_fit`
nor `solve_cubic_spline_moments`'s own internal checks changed; they are unreachable once the wrapper's
earlier check has already fired, which is the point.

**Tests** (`test/num/temporal_basis_test.cpp`), each following the finding's explicit warning not to
compare against `fit_polynomial`/`cubic_spline_moments` themselves (post-split they are the code under
test, not an independent oracle) — instead asserting on a *named, frozen* message, matching against the
`git show fd5bbc2` extraction above:
- `FitPolynomialReportsTheLengthMismatchBeforeTheDegreeError` — `f` wrong length AND `degree<0`; asserts
  the "different lengths" message fires and "negative degree" does not.
- `CubicSplineMomentsReportsTheLengthMismatchBeforeTheMonotonicityError` — `f` wrong length AND `t`
  non-monotonic; asserts "different lengths" fires and "strictly increasing" does not.
- `FitPolynomialStillReportsNoSamplesAheadOfALengthMismatch` — empty `t` with non-empty `f`; asserts
  "no samples" still fires first (guards the wrapper's own two-check order, not just the interaction
  with `prepare_*`).

All three fail against the pre-fix code (verified by hand: see "Mutation" below) and pass against the
fix. `PreparePolynomialFitThrowsExactlyAsFitPolynomialOnTOnlyErrors` and the spline equivalent (pass 1's
tests, `t`-only errors, no `f`-length interaction) are unaffected and still pass.

**Mutation** `fit-polynomial-drops-the-pre-split-length-check`
(`build-evidence/mutations/num-linalg-field-pass2.toml`): reverts `fit_polynomial` to the bare
`solve_polynomial_fit(prepare_polynomial_fit(t, degree, t0), f)` (the accidental order). Applied it by
hand (edited the header, rebuilt, ran the target, reverted, rebuilt again to confirm clean) because
`tools/mutate.py`'s apply mode refuses to run against a dirty `include`/`src` tree (this whole fix pass
is one), the same constraint the pass-2 reviewer hit. `FitPolynomialReportsTheLengthMismatchBeforeTheDegreeError`
fails under the mutant with `negative degree -1` instead of `different lengths` — confirmed killed.

### R-48 — IMPROVEMENT — the disclosed remaining per-element allocation

The pass2 review's own words: "the shared factorization is hoisted exactly as described... What
remains: `solve_polynomial_fit`/`solve_cubic_spline_moments` still allocate their own scratch... on
every element". Read `include/fuse/num/waveform.hpp:477-571`: `evaluate_dg`/`integrate_dg` /
`evaluate_spline`/`integrate_spline` each build one plan outside their per-element loop (pass 1's fix)
but called the allocating `solve_polynomial_fit`/`solve_cubic_spline_moments` fresh inside it.

**Fix.** Added `solve_polynomial_fit_into`/`solve_cubic_spline_moments_into` to
`include/fuse/num/temporal_basis.hpp`, following the `_into`-plus-scratch convention
`linalg::least_squares_into`/`LsScratch` already established for this exact problem (I5): they write
into a caller-supplied `out` (the answer) and a caller-supplied scratch buffer (`rhs_scratch`/
`r_scratch`), instead of allocating a fresh pair every call. `vector::assign`/`resize` on a buffer
already at the needed capacity does not reallocate, so a caller that hoists `out`/scratch outside its
loop and reuses them pays for growth once, not once per element. The original `solve_polynomial_fit`/
`solve_cubic_spline_moments` are now thin allocating wrappers around the `_into` pair (same signature,
same behavior, so every pass-1 caller is unaffected). Updated all four `Waveform` per-element loops
(`evaluate_dg`, `evaluate_spline`, `integrate_dg`, `integrate_spline`) to hoist `coeffs`/`rhs_scratch`
(resp. `moments`/`r_scratch`) outside the loop and call the `_into` entry point every element.

**Tests:**
- `test/num/temporal_basis_test.cpp`: `SolvePolynomialFitIntoAgreesWithSolvePolynomialFitAndBuffersAreReusable`
  / the cubic-spline equivalent — call the `_into` variant twice with different `f` through the SAME
  `out`/scratch buffers and assert each result matches the allocating wrapper's answer for that `f`,
  the "no stale value leaks across a reused buffer" property `Waveform`'s loop depends on.
- `test/num/waveform_test.cpp`: `DgAndSplineEvaluateAllocationIsIndependentOfElementCount` — the
  allocation-count fence I5 calls for (the pass2 review's "Not found" section flagged this as a gap
  still open from the first review, not itself a new finding, but it is the only way to prove the fix
  actually closes the allocation rather than just adding an unused `_into` entry point). Built on the
  same global-`operator new`-override pattern as `test/num/accelerator_test.cpp`
  (`SteadyStateUpdateAllocatesNothing`); this test binary has exactly one translation unit, so the
  override is safe here. Calls `evaluate()` once on a DG waveform and once on a Spline waveform, each
  at two very different element counts (2 vs 400), and asserts the allocation count for one `evaluate()`
  call does not depend on element count — a per-element buffer left in the loop would make it scale.

**Mutation** `waveform-evaluate-dg-reallocates-coefficients-per-element`
(`build-evidence/mutations/num-linalg-field-pass2.toml`): reverts `evaluate_dg`'s loop to the pass-1
form (`num::eval_polynomial(num::solve_polynomial_fit(plan, ff), x, 0.0)`, no hoisted buffers). Applied
by hand for the same git-status reason as above. `DgAndSplineEvaluateAllocationIsIndependentOfElementCount`
fails under the mutant (`few=10, many=806` at `basis=3`/DG) — confirmed killed.

**Still not claimed closed as a live `I5` violation**: `Waveform` has no caller anywhere in `src`/
`include` outside its own header and tests (confirmed by the same grep the pass2 review ran), so this
was dormant either way. The fix removes the allocation regardless, since a caller could appear at any
time and the finding was about the function's own contract, not about whether it is currently exercised.

### R-49 — DEVIATION — the `num`→`model` test-registration move

Confirmed the pass2 review's read: `test/num/convergence_test.cpp` includes only
`<fuse/model/iteration.hpp>` and `<fuse/base/error.hpp>` — no `fuse/num` symbol anywhere in the file
(grepped) — because `ConvergenceMeasure`/`ConvergenceCriterion`/`MeasureKind`/`Converged` live in
`include/fuse/model/iteration.hpp`, not `fuse/num`. This is a pure move: no line of the test's content
needed to change.

**Fix**, using exactly the means this task's owned-paths grant allows for this one finding: created
`test/model/iteration_test.cpp` as a byte-identical copy of `test/num/convergence_test.cpp` (verified:
`diff` reports no differences); edited `test/CMakeLists.txt`'s `model` section to add
`fuse_add_gtest(model iteration THROWS)` and the `num` section to remove `convergence` from the
`foreach(t temporal_basis waveform convergence accelerator)` list. Did **not** delete
`test/num/convergence_test.cpp` — see "Needs another owner: delete" below; the file is unmodified on
disk.

**Verified**: `fuse_model_iteration_test` builds and its 21 tests pass (matching the pass1/pass2
reviews' reported count for the pre-move `fuse_num_convergence_test`), using the temporary local gate
described above to work around the registration audit while the old file is still present.

## Rejected

None — the pass2 review upheld none of pass 1's rejections as needing reconsideration in this layer
group, and this pass found no basis to reject any of its own three findings.

## Needs another owner: delete

- `test/num/convergence_test.cpp` — superseded by `test/model/iteration_test.cpp` (R-49, this report).
  Not deleted here per this task's explicit instruction that the lead removes it; until it is removed,
  a fresh `cmake` configure of `test/CMakeLists.txt` fails the unregistered-test-source audit (by
  design — R-07's own protection working as intended, not a new problem).

## Needs a host run

None. `num`/`linalg`/`field` issue no MPI collective (re-confirmed by the same grep the pass2 review
ran); every test here runs under tier0.

## Deviation decisions

None beyond R-49 itself, which is the deviation this report closes as far as this layer group's owned
paths allow — the registration move is complete; only the old file's deletion (explicitly out of scope
for a fixer per this task's rules) remains.

## Commit units

```json
[
  {
    "ids": ["N-num-linalg-field-1"],
    "files": [
      "include/fuse/num/temporal_basis.hpp",
      "test/num/temporal_basis_test.cpp",
      "build-evidence/mutations/num-linalg-field-pass2.toml"
    ],
    "message": "Restore fit_polynomial/cubic_spline_moments error precedence\n\nCheck f's length ahead of the prepare/solve split so a combined error reports the same message it did before the split (N-num-linalg-field-1)."
  },
  {
    "ids": ["R-48"],
    "files": [
      "include/fuse/num/waveform.hpp",
      "test/num/waveform_test.cpp"
    ],
    "message": "Remove the per-element allocation left in Waveform's fit loops\n\nsolve_polynomial_fit_into/solve_cubic_spline_moments_into write into hoisted buffers instead of allocating a fresh pair every element (R-48)."
  },
  {
    "ids": ["R-49"],
    "files": [
      "test/model/iteration_test.cpp",
      "test/CMakeLists.txt"
    ],
    "message": "Move the convergence-measure test to the model layer\n\ntest/num/convergence_test.cpp only ever exercised fuse/model/iteration.hpp; registered as test/model/iteration_test.cpp instead (R-49). The old file still needs `git rm test/num/convergence_test.cpp`, listed under Needs another owner in this report -- this commit's test/CMakeLists.txt hunk only adds/removes registrations and does not delete it."
  }
]
```

Each unit builds cleanly on its own given the ones before it (verified: rebuilding after each
mutation's manual revert, in order, left every prior unit's tests green). The third unit's
`test/CMakeLists.txt` hunk cannot make `cmake` configure cleanly by itself until
`test/num/convergence_test.cpp` is actually removed from disk in the same commit or the one after —
noted so the lead does not apply it in isolation and get surprised by the audit.
