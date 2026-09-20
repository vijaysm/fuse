# Fix report: num, linalg, field (REVIEW.md R-08, R-29, R-45, R-48, R-49)

HEAD at start: `ffea077`. Branch `review-2026-09-19-fixes`. All five findings read,
confirmed against the cited lines, and fixed or dispositioned below. No `git`
state was changed by me (no add/commit/checkout/stash/reset).

## Commit units

```json
[
  {
    "ids": ["R-08", "R-29"],
    "files": [
      "include/fuse/field/pipeline.hpp",
      "include/fuse/field/field_ref.hpp",
      "test/field/pipeline_test.cpp",
      "build-evidence/mutations/field-pipeline.toml"
    ],
    "message": "Accept an empty-partition FieldRef and tighten the padded-layout bound\n\nFieldSpan::from() now mirrors FieldRegistry::bind()'s null-for-empty allowance, and Layout::well_formed() no longer over-rejects valid padded/strided layouts by one stride unit."
  },
  {
    "ids": ["R-45"],
    "files": [
      "include/fuse/linalg/least_squares.hpp",
      "test/linalg/least_squares_test.cpp",
      "test/num/accelerator_test.cpp",
      "build-evidence/mutations/linalg-least-squares.toml"
    ],
    "message": "Document that least_squares_into supports fewer rows than columns\n\nThe accelerator's hot loop already relies on this with a small interface and the default reuse limit; this records the contract and adds the regression coverage that was missing."
  },
  {
    "ids": ["R-48"],
    "files": [
      "include/fuse/num/temporal_basis.hpp",
      "include/fuse/num/waveform.hpp",
      "test/num/temporal_basis_test.cpp"
    ],
    "message": "Hoist the shared basis factorization out of Waveform's per-element loops\n\nfit_polynomial/cubic_spline_moments split into a t-only prepare step and an f-only solve step, and DG/Spline evaluate/integrate build the shared plan once instead of once per field element."
  }
]
```

Every file I changed appears in exactly one unit above. `R-49` has no unit: I made
no code change for it (see "Needs another owner").

## Fixed

### R-08 (NL-B1, BLOCKER) — `FieldSpan::from()` refuses a legitimately-bound empty field

**Confirmed as described.** Read `pipeline.hpp:96-99` against `registry.hpp:47-56`:
`bind()` refuses null only when `count != 0` (its Task 4.1 fix, explained in the
comment there), so every `FieldRef` reachable from `registry.get()` with a null
pointer has `count == 0` by construction — yet `FieldSpan::from()`'s
`if (!f.valid())` refused all of them unconditionally.

**Fix.** `pipeline.hpp`: guard changed to `if (!f.valid() && f.layout().count != 0)`,
mirroring `bind()` exactly.

**Test.** Rewrote `Pipeline.AnUnboundFieldIsRefusedNotDereferenced` to use a
non-empty null `FieldRef` (the case that must still refuse) — the old fixture,
`fuse::Layout{}`, has `count == 0` and could not tell "unbound" apart from "bound
to nothing, on purpose". Added
`Pipeline.AnEmptyFieldFromABoundZeroCountFieldRefDoesNotRefuse`: an empty,
null-pointer `FieldRef` now produces a valid zero-extent `FieldSpan` whose
`fill`/`local_sum`/`slice` are no-ops, not throws.

**Verified.** `fuse_field_pipeline_test` (tier0 and tier1, compiles clean under
both): 13/13 pass. Also reproduced standalone against `Kokkos::Serial` under
`-fsanitize=address,undefined` (`/tmp/nlcheck/check_pipeline.cpp`, not part of
the repository): the empty-partition case succeeds with `element_count()==0`,
and a genuinely non-empty null `FieldRef` still throws `E-FIELD-NULL`. Clean
under ASan/UBSan.

**Mutation.** `build-evidence/mutations/field-pipeline.toml`,
`fieldspan-from-refuses-empty-partition`: reverts the guard to
`if (!f.valid())`; anchor verified present via
`tools/mutate.py --check-anchors`.

### R-29 (NL-A1, MAJOR) — `Layout::well_formed()` over-rejects valid padded/strided layouts

**Confirmed as described.** The disjointness bounds `stride >= sat_mul(nc, cs)` /
`cs >= sat_mul(count, stride)` are one full `cs`/`stride` more conservative than
the necessary-and-sufficient `stride > (nc-1)*cs` / `cs > (count-1)*stride`.
Reproduced the reviewer's two brute-force-checked false negatives
(`{stride=5,cs=2,count=4,components=3}` and `{stride=2,cs=5,count=3,
components=2}`) standalone: both are genuinely disjoint and the old bound
rejected both.

**Fix.** `field_ref.hpp`: tightened both bounds by one unit, exactly as the
review's suggested fix (`stride > sat_mul(nc-1, cs)`, `cs > sat_mul(count-1,
stride)`).

**Test.** Added `Layout.WellFormedAcceptsATightPaddedInterleavedLayout` and
`Layout.WellFormedAcceptsATightPaddedBlockedLayout` to
`test/field/pipeline_test.cpp` (each pinning the tight-accept case plus a
one-unit-tighter reject case). **These live in `test/field/`, not
`test/runtime/field_ref_test.cpp`** where the existing `Layout.WellFormed*`
suite lives: `test/runtime/**` is outside my owned paths for this pass, and
`Layout` is header-only, so `pipeline_test.cpp` (which already includes
`field_ref.hpp` directly) is a legitimate place to pin it without touching a
path I don't own. The existing `Layout.WellFormed*` tests in
`test/runtime/field_ref_test.cpp` are unaffected: the tightened bound accepts a
strict superset of what the old one accepted, so nothing that used to pass can
now fail — confirmed by inspection, not re-verified by me since that file isn't
mine to run against in this pass. **Needs another owner** to move (not
duplicate) these two tests into `test/runtime/field_ref_test.cpp` alongside
the rest of the `Layout.WellFormed*` suite if that is judged the more natural
long-term home; I left them in `test/field/` rather than duplicate coverage.

**Verified.** Same test run as R-08 (shared binary). Standalone: reproduced
both tight-accept cases and both one-unit-tighter reject cases under
ASan/UBSan.

**Mutation.** `build-evidence/mutations/field-pipeline.toml`,
`layout-well-formed-off-by-one-stride`: reverts to the old bounds; anchor
verified.

### R-45 (NL-A2, MINOR) — `least_squares_into`'s `m >= k` precondition is silently dropped

**Confirmed as described.** `least_squares_into` never calls `check_shape`, and
`cols_`/`blk.cols` (bounded by `cfg.reuse`, default 8) is independent of `n`
(the interface size), so `m < k` is reachable at
`accel_quasi_newton.cpp:536,632` on the very first window of a small-interface
coupling with the default reuse limit. Confirmed by inspection that the QR
loop is bounded by `steps = min(m, k)` everywhere, so no buffer overrun is
possible; confirmed the *absence* of test coverage for this shape by reading
every `n=1`/`n=2` case in `accelerator_test.cpp` and finding each resets after
two iterations, never reaching `cols_ > n`.

**Fix.** Chose the doc-comment option from the review's two alternatives
(document vs. debug-assert), since the behavior is already correct and
verified: `least_squares.hpp`'s doc on `least_squares_into` now states
explicitly that it does not enforce `m >= k`, that the shape is safe (bounded
by `steps = min(m,k)`), and names the two tests that pin it. No code branch
changed.

**Tests.** `LeastSquares.IntoAgreesWithBuiltinWhenMIsLessThanK`
(`test/linalg/least_squares_test.cpp`): calls `least_squares_into` directly at
`(m,k) = (1,4)` and `(2,6)` on a consistent system (`b` built from a known
`x_true`), asserting `rank <= m` and a round-off residual.
`Accelerator.HandlesAnInterfaceSmallerThanTheReuseLimit`
(`test/num/accelerator_test.cpp`): an `n=1` IQN-ILS accelerator with the
default `reuse=8`, run for 4 iterations, asserting `update()` succeeds, the
iterate stays finite, and `columns() > 1` (i.e. the test actually reaches the
shape it exists to cover).

**Verified.** `fuse_linalg_least_squares_test` 6/6, `fuse_num_accelerator_test`
54/54 pass. Standalone under ASan/UBSan at `(m,k) ∈ {(1,4),(2,6),(1,8)}`:
residual `1e-16`–`1e-18`, rank `<= m` in every case, matching the reviewer's
own ad hoc check (now durable).

**Mutation.** `build-evidence/mutations/linalg-least-squares.toml`,
`least-squares-into-checks-shape`: adds the `check_shape` call back in (the
change that would make the accelerator's own call sites throw); anchor
verified.

### R-48 (NL-I1, IMPROVEMENT) — DG/Spline rebuild the shared basis matrix once per field element

**Confirmed as described.** `fit_polynomial`'s normal-equations build and
Gaussian elimination, and `cubic_spline_moments`'s tridiagonal factorization,
depend only on the sample times (and, for the polynomial, `degree`/`t0`) —
never on the per-element values — yet `Waveform::evaluate_dg`/`integrate_dg`/
`evaluate_spline`/`integrate_spline` called the monolithic functions once per
element with the same `t` and a different `f` each time. Confirmed via
`grep` that `Waveform` has no live caller yet (space-time projection is
not-yet-landed), so this was dead-weight cost, not a live I5 (no-allocation)
violation today — but exactly the kind that becomes one once wired in, per the
review.

**Fix.** Split each into a `t`-only "prepare" step and an `f`-only "solve"
step:
- `temporal_basis.hpp`: `prepare_polynomial_fit(t, degree, t0)` →
  `PolynomialFitPlan` (basis values `phi`, the eliminated normal-equations
  matrix, and the recorded pivot swaps/multipliers) and
  `solve_polynomial_fit(plan, f)` replay those exactly on a fresh right-hand
  side. `prepare_cubic_spline_moments(t)` → `SplineMomentPlan` (the
  forward-eliminated tridiagonal matrix and its multipliers) and
  `solve_cubic_spline_moments(plan, f)`. `fit_polynomial`/
  `cubic_spline_moments` themselves are now one-line wrappers composing the
  two halves — same signature, same exceptions, same answer for every existing
  caller.
- `waveform.hpp`: `evaluate_dg`/`integrate_dg`/`evaluate_spline`/
  `integrate_spline` build the plan once, outside the per-element loop, and
  call `solve_*` per element instead of the monolithic function.

I verified by hand that the split is bit-identical to the monolithic
computation: the `a`-matrix build order, the elimination order (swap → compute
`fac` → eliminate), and the rhs-accumulation order are unchanged, just moved
across the prepare/solve boundary — confirmed by a direct comparison test
(below), not just by argument.

**Tests.** `TemporalBasis.PreparedPolynomialFitAgreesWithFitPolynomialAcrossSeveralF`
and `...PreparedSplineMomentsAgreeWithCubicSplineMomentsAcrossSeveralF`
(`test/num/temporal_basis_test.cpp`): one plan, two different `f` arrays,
compared bit-for-bit (`EXPECT_DOUBLE_EQ`) against calling the monolithic
function on each `f` independently. `...PreparePolynomialFitThrowsExactlyAs...`
and `...PrepareSplineMomentsThrowsExactlyAs...`: the `t`-only error paths
(empty input, negative degree, singular system, non-increasing times) still
throw the same exception types when driven through `prepare_*` directly. The
existing `Waveform.DgFitsEachElementIndependently`/`HandlesMultipleElements`
already exercise the new code path end-to-end unchanged (no edit needed) and
still pass.

**Verified.** `fuse_num_temporal_basis_test` 17/17, `fuse_num_waveform_test`
39/39, `fuse_num_waveform_order_test` 12/12 pass. Standalone, beyond the
committed tests: a 64-element DG waveform and a 32-element natural-cubic
Spline waveform, each element with an independent quadratic/transcendental
function, evaluated and (DG) integrated exactly against the closed-form
answer, plus a direct bit-identical comparison of `solve_*` against the
monolithic function across two different `f` sharing one plan — all clean
under ASan/UBSan.

**No mutation added.** This is a performance refactor with unchanged,
bit-identical semantics (proven above), not a behavioral toggle a mutation
would flip between right and wrong — there is no "old bug" to revert to.

**Not done, and out of scope for this pass.** The review's suggested
allocation-count fence test (`Accelerator`-style, analogous to
`SteadyStateUpdateAllocatesNothing`) was not added: `Waveform` still allocates
per `solve_*` call (`rhs`/`c` in the polynomial case, `r`/`m` in the spline
case) since only the shared *factorization* was hoisted, not the per-element
scratch. Per the review, this is only a live I5 concern once the space-time
projection operator calls `Waveform` repeatedly, which has not landed — I did
not add scratch-reuse plumbing (`LsScratch`-style buffers passed in) for a
path with no caller yet, to keep this fix to the shape the review asked for.
Flagging so it isn't lost: the next allocation-fence work on `Waveform` should
also hoist `rhs`/`c`/`r`/`m` into caller-supplied scratch.

## Rejected

None. All five findings in `docs/dev/reviews/2026-09-19/num-linalg-field.md`
were confirmed as described against the cited lines; none were already fixed
or wrong.

## Needs another owner

- **R-29's test placement.** The two new `Layout.WellFormed*` boundary tests
  live in `test/field/pipeline_test.cpp` rather than
  `test/runtime/field_ref_test.cpp` (where the rest of the `Layout.WellFormed*`
  suite lives), because `test/runtime/**` is outside my owned paths for this
  pass. If the `runtime`-layer owner agrees they belong alongside the existing
  suite, the exact move is: delete the two `TEST(Layout, ...)` blocks I added
  at the end of `test/field/pipeline_test.cpp` and add them to
  `test/runtime/field_ref_test.cpp` next to
  `Layout.WellFormedRejectsAliasingAndNonPositiveComponents`. No functional
  change either way; I left them in place to avoid touching a path I don't
  own.

- **R-49 (NL-D1, DEVIATION) — `num.convergence` test exercises `fuse::model`,
  not `fuse::num`.** Confirmed as described:
  `test/num/convergence_test.cpp:1-13` includes only
  `<fuse/model/iteration.hpp>` and tests exclusively
  `ConvergenceCriterion`/`ConvergenceMeasure`/`MeasureKind`/`Converged`;
  nothing from `fuse::num` appears in the file. `test/CMakeLists.txt:107-109`
  still registers it under the `num` `foreach`, so `ctest -L num` and a
  directory search both mislabel it, and `fuse::model::iteration.hpp` has no
  test filed under `test/model/` at all.
  **Deviation decision (see below): code moves to match the document** —
  relocate the file and its registration, not update `layering.md` to bless
  the current location. **I made no change for this**, because the fix
  requires touching `test/model/**` (creating
  `test/model/iteration_test.cpp` or `test/model/convergence_test.cpp`) and the
  `model`-layer `foreach` in `test/CMakeLists.txt` — neither is in my owned
  paths (`include/fuse/num/**`, `src/num/**`, `test/num/**`,
  `include/fuse/linalg/**`, `src/linalg/**`, `test/linalg/**`,
  `include/fuse/field/**`, `test/field/**`), and the CMakeLists exception I was
  given is scoped to "your layer's own `# ---- <layer>` section", not the
  `model` section. I also could not remove the `convergence` entry from my own
  `num`-layer `foreach` unilaterally: doing so without the corresponding
  `model`-layer addition landing in the same commit would deregister the only
  test `fuse::model::iteration.hpp` has, which is strictly worse than the
  current mislabeling.
  **Exact change for the `model`/`verification`-layer owner:** move
  `test/num/convergence_test.cpp` to `test/model/iteration_test.cpp` (content
  unchanged); in `test/CMakeLists.txt`, remove `convergence` from the
  `foreach(t temporal_basis waveform convergence accelerator)` loop at
  `num`'s section and add it (as `iteration`) to the `model`-layer `foreach`,
  linking `fuse::model` instead of `fuse::num`; update
  `docs/dev/layering.md`'s "A boundary the gate moved" note to also record
  that the test followed the types (I did not edit `docs/dev/layering.md`;
  also outside my owned paths).

## Needs a host run

None. All five findings and their fixes are pure host-code/library logic with
no MPI dependency (`num`/`linalg`/`field` sit below `transport` in the layer
order and issue no collectives, confirmed by the reviewer's "Not found"
section and by my own read of the changed files). Every test I added or
touched runs today, in this sandbox, under `ctest` (or standalone) at tier0;
I additionally confirmed all touched targets *compile* clean under tier1
(`cmake --build build-t1 --target fuse_field_pipeline_test
fuse_linalg_least_squares_test fuse_num_temporal_basis_test
fuse_num_waveform_test fuse_num_accelerator_test fuse_num_convergence_test`).

## Deviation decisions

- **R-49: code moves to match the document.** `docs/dev/layering.md` already
  records, as a deliberate decision, that `ConvergenceMeasure`/
  `ConvergenceCriterion`/`AccelKind`/`AccelConfig`/`Converged` moved from `num`
  to `fuse/model/iteration.hpp` during the port. The test that exercises those
  types should have moved with them; nothing argues for re-litigating the type
  location itself (the code side is already right and consistent with
  `accelerator.hpp:19-23`'s own note). The only drift is the test's
  filesystem/CMake location, so the document does not need to change — the
  test does. See "Needs another owner" above for the exact, un-made change and
  why I could not make it inside my owned paths.

## Build/test evidence

- Tools used: `cmake`/`c++`/`clang++` at `/opt/homebrew/bin` and
  `/opt/homebrew/Cellar/llvm/22.1.4/bin` (Apple clang has no `-fopenmp`, needed
  for `pipeline.hpp`'s Kokkos `OpenMP` backend, matching the original
  reviewer's approach); `/opt/kokkos`, `/opt/eigen3`, `/opt/homebrew/opt/libomp`
  on the include/lib path.
- Tier0 configure/build: `cmake --preset tier0 -S
  /Users/mahadevan/Code/AI/coupling/fuse -B $PWD/build-t0` (this workspace's
  own directory, not under the repo), then `cmake --build build-t0 -j4
  --target <owned targets>`. Hit one transient failure unrelated to my files
  (`install(EXPORT ...)` referencing a target not in any export set, in
  `src/CMakeLists.txt` — another fixer's concurrent R-25 work); retried after
  it resolved and configure succeeded cleanly.
- All ten owned-layer test binaries built and run at tier0, 200/200 tests
  passing: `fuse_field_pipeline_test` (13), `fuse_linalg_least_squares_test`
  (6), `fuse_num_temporal_basis_test` (17), `fuse_num_waveform_test` (39),
  `fuse_num_waveform_order_test` (12), `fuse_num_accelerator_test` (54),
  `fuse_num_convergence_test` (21 — untouched by me, still passes),
  `fuse_num_kernels_test` (19), `fuse_num_residual_norm_test` (7),
  `fuse_num_space_kernels_test` (12).
- Tier1 configure/build (`cmake --preset tier1 -B $PWD/build-t1`, MPI found at
  `/opt/mpich`): the same six owned targets compile clean (none are MPI tests,
  so nothing here needed a host run).
- `python3 tools/check-invariants.py` → `0 violation(s)`.
- `python3 tools/check-layering.py` → `0 violation(s)`.
- `python3 tools/mutate.py --check-anchors --build-dir build-t0` → my three new
  mutations (`fieldspan-from-refuses-empty-partition`,
  `layout-well-formed-off-by-one-stride`, `least-squares-into-checks-shape`)
  all anchor-valid. Two pre-existing stale anchors reported
  (`merge-typed-at-first-writer` in `src/ir/lowering.cpp`,
  `mesh-accepts-a-parent-outside-the-cell-set` in
  `include/fuse/backend/native/mesh.hpp`) belong to other fixers' owned paths
  (ir-schedule, backend) and are unrelated to this pass.
- Did not run `tools/mutate.py`'s full apply/build/test/revert cycle (only
  `--check-anchors`), per the hard rule that it refuses on a dirty shared tree.
  Confidence that each mutation flips its target test comes from manual
  reasoning plus the standalone ASan/UBSan reproductions described under each
  finding, not from running the mutation gate end-to-end.
- Standalone verification programs (not part of the repository, built and run
  under `-fsanitize=address,undefined`, all clean): `/tmp/nlcheck/check_pipeline.cpp`
  (R-08, R-29 against real Kokkos/`fuse::base::error` code),
  `/tmp/nlcheck/check_ls.cpp` (R-45), `/tmp/nlcheck/check_tb.cpp` (R-48 prepare/solve
  bit-identity and error paths), `/tmp/nlcheck/check_wf.cpp` (R-48 through
  `Waveform` end-to-end, 64-element DG and 32-element Spline).

## Deviations from the assigned task

- None in scope/coverage: all five findings assigned to me (R-08, R-29, R-45,
  R-48, R-49) were read, confirmed, and fixed or dispositioned above.
- Two items ended up requiring another owner despite being "my" findings
  (R-29's test-suite placement, R-49's file move) because the correct fix
  crosses my owned-paths boundary; both are recorded above with the exact
  change needed rather than left silent.
