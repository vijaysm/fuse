<!-- Detailed layer review, 2026-09-19, HEAD bdf4346. Consolidated in ../../REVIEW.md; unified ids there are R-nn, this file's ids are NL-<id>. -->

# Review: `num`, `linalg`, `field` (FUSE, HEAD bdf4346)

Scope as assigned: `include/fuse/num/**`, `src/num/**`, `include/fuse/linalg/**`,
`src/linalg/**`, `include/fuse/field/**`, and tests under `test/num`, `test/linalg`,
`test/field`. Cross-references outside this scope (`fuse/model/iteration.hpp`,
`test/runtime/field_ref_test.cpp`, CMake test registration) were read only to
confirm behavior at the scope boundary, never to review their own correctness.

## 1. Coverage

Read in full, line by line:

- `include/fuse/num/accelerator.hpp` (226), `include/fuse/num/kernels.hpp` (192),
  `include/fuse/num/residual_norm.hpp` (107), `include/fuse/num/space_kernels.hpp`
  (187), `include/fuse/num/temporal_basis.hpp` (355), `include/fuse/num/waveform.hpp`
  (569)
- `src/num/accel_quasi_newton.cpp` (815), `src/num/accel_relaxation.cpp` (536),
  `src/num/kernels_inst.cpp` (11), `src/num/residual_norm.cpp` (112)
- `include/fuse/linalg/least_squares.hpp` (109), `src/linalg/least_squares.cpp` (251)
- `include/fuse/field/extent.hpp` (58), `include/fuse/field/field_ref.hpp` (191),
  `include/fuse/field/pipeline.hpp` (235), `include/fuse/field/registry.hpp` (109)
- `docs/dev/invariants.md`, `docs/dev/layering.md`, `docs/dev/conventions.md`,
  `docs/dev/testing.md`, `docs/dev/porting.md`, `docs/dev/rank-agreement.md`
- `tools/check-layering.py`, `tools/check-invariants.py` (to know exactly what the
  automated gates do and do not scan)

Test files read for coverage cross-checking (not treated as production code, but
read to confirm which scenarios are and are not exercised): `test/num/accelerator_test.cpp`
(1954 lines, all `TEST` bodies located and the divergence/IMVJ-chain tests read in
full), `test/num/convergence_test.cpp`, `test/num/kernels_test.cpp`,
`test/num/residual_norm_test.cpp`, `test/num/space_kernels_test.cpp` (read in full),
`test/num/temporal_basis_test.cpp`, `test/num/waveform_order_test.cpp`,
`test/num/waveform_test.cpp` (TEST names enumerated, key bodies read),
`test/linalg/least_squares_test.cpp` (read in full), `test/field/pipeline_test.cpp`
(read in full), `test/field/compile_fail_dangling_span.cpp`,
`test/field/compile_ok_owning_chain.cpp` (read in full).
`test/runtime/field_ref_test.cpp` was read only for its `Layout.WellFormed*` cases
(it is outside the assigned scope; `FieldRef`/`Layout` are header-only and its test
happens to live under `test/runtime`, which is itself Finding D1 below).

**What was compiled.** Every header in scope was `-fsyntax-only`-compiled standalone
(`c++ -std=c++20`) against the repository's real generated `fuse/base/config.hpp`
(from `build-tier1/`, not written by me), `/opt/kokkos/include`, and
`/opt/homebrew/include`; `fuse/num/space_kernels.hpp` and `fuse/field/pipeline.hpp`
additionally against `/opt/homebrew/opt/libomp/include` with `-fopenmp` (Apple
clang has no `-fopenmp`, so this leg used `/opt/homebrew/Cellar/llvm/22.1.4/bin/clang++`),
and their templates were **explicitly instantiated** on `Kokkos::Serial` to force
real template-body checking, not just parse-checking. All five `.cpp` files in
scope were compiled the same way (`least_squares.cpp` additionally needed
`/opt/TPLs/eigen-3.4.0` on the include path, since the build's `config.hpp` has
`FUSE_LINALG_EIGEN=1`). All of the above compiled clean; no build was run, and
nothing under `build-tier1/` or any new build directory was created or modified
by me (I only *read* the pre-existing `build-tier1/include/fuse/base/config.hpp`
to get real macro values).

To settle two specific hypotheses that reading alone could not (Finding A1 below,
and the `least_squares_into` precondition question, Finding A2), I extracted
the exact `Layout::well_formed()` logic and the exact QR algorithm verbatim into
two standalone files and ran them under `-fsanitize=address,undefined` with
concrete inputs (`/tmp/lstest/test_layout.cpp`, `/tmp/lstest/test_ls.cpp`, not part
of the repository). Results are cited inline at each finding.

No project file was edited, no build directory was created under the repository,
and no `git` state was changed.

## 2. Findings table

| id | severity | file:line | title |
|----|----------|-----------|-------|
| B1 | BLOCKER | `include/fuse/field/pipeline.hpp:96-99` | `FieldSpan::from()` refuses the exact empty-partition `FieldRef` that `FieldRegistry::bind()` was fixed to accept |
| A1 | MAJOR | `include/fuse/field/field_ref.hpp:97-98` | `Layout::well_formed()` rejects valid padded/strided layouts (off-by-one-stride-unit bound) |
| A2 | MINOR | `src/num/accel_quasi_newton.cpp:536,632` | `linalg::least_squares_into` skips the `m >= k` precondition its sibling enforces, and the resulting `m < k` path is reachable but untested |
| I1 | IMPROVEMENT | `include/fuse/num/waveform.hpp:478-497,536-556` | DG/Spline `evaluate`/`integrate` rebuild the shared, per-element-independent basis matrix once per field element |
| D1 | DEVIATION | `test/CMakeLists.txt:107-109`, `test/num/convergence_test.cpp:1-13` | `num.convergence` test exercises `fuse::model::iteration.hpp` entirely, not the `num` layer |

## 3. Findings in full

### B1 — BLOCKER — `FieldSpan::from()` refuses a legitimately-bound empty field

**Where:** `include/fuse/field/pipeline.hpp:96-99` (the `!f.valid()` check inside
`FieldSpan::from`), against `include/fuse/field/registry.hpp:47-56` (`FieldRegistry::bind`'s
null-for-empty allowance) and pinned wrong by `test/field/pipeline_test.cpp:179-182`.

**What is wrong.** `FieldRegistry::bind()` explicitly, and at length, allows binding
a field with `data == nullptr` **when `layout().count == 0`** — the comment at
`registry.hpp:38-46` narrates this as a deliberate fix ("Task 4.1... made a
component that legitimately owns zero elements of a region impossible to bind...
on a partitioned mesh a rank owning none of a region is ordinary, not an error").
The check actually written is:

```cpp
if (!ref.valid() && ref.layout().count != 0) {   // registry.hpp:52
```

i.e. null is refused **only** when `count != 0`; a `FieldRef(nullptr, Layout{.count=0}, ...)`
binds successfully and is exactly what `registry.get(id)` will later hand back to
a caller on a rank that owns none of that field.

`FieldSpan::from()`, which is the pipeline's only sanctioned way to turn a
registered `FieldRef` into something `slice()`/`component()`/`local_sum()` can
operate on, does not carry the same allowance:

```cpp
static FieldSpan from(const FieldRef& f) {
  if (!f.valid()) {                              // pipeline.hpp:96
    FUSE_REFUSE("E-FIELD-NULL", ...);
  }
```

This checks `data != nullptr` alone, with no `count` exception. Because `bind()`
never lets a *non-empty* field through with a null pointer (that case is refused
at `registry.hpp:52-56` before it ever reaches `refs_`), **every** `FieldRef` that
can come out of `registry.get()` with `data == nullptr` is, by construction, one
with `count == 0` — the exact case `bind()` was fixed to treat as ordinary. So
`FieldSpan::from()` throws `fuse::Refusal` on 100% of the empty-partition fields
the registry is designed to hold, not on some boundary case.

**How I confirmed it.** Read both functions side by side (cited above) and traced
that `bind()`'s guard is the *only* gate a `FieldRef` passes before entering
`refs_`, so nothing with `count > 0` and `data == nullptr` can reach `from()` via
the registry — meaning `from()`'s blanket `!f.valid()` check is strictly broader
than the condition `bind()` actually refuses. Then checked the one test that
exercises this exact call, `test/field/pipeline_test.cpp:179-182`:

```cpp
TEST(Pipeline, AnUnboundFieldIsRefusedNotDereferenced) {
  const fuse::FieldRef unbound(nullptr, fuse::Layout{}, fuse::MemorySpace::Host);
  EXPECT_THROW((void)fuse::FieldSpan<>::from(unbound), fuse::Refusal);
}
```

`fuse::Layout{}` default-initializes `count = 0` (`field_ref.hpp`'s `Layout`
default member initializer). This is *not* a distinguishable "never registered"
sentinel — it is bit-for-bit the same state as a legitimately-bound empty
`FieldRef`, so the test's name ("unbound... not dereferenced") describes intent
that the code it exercises cannot actually tell apart from "bound to nothing, on
purpose." The test passes today because both functions agree with each other,
not because the behavior is right.

**Why this is a BLOCKER and not a MAJOR.** A component that legitimately owns
zero elements of a field on some rank — the registry's own justification for the
fix, and the Task 5.4 n→m redistribution scenario it names — calls
`FieldSpan::from(registry.get(id).value())` and gets an **uncaught throw** on
that rank, on every call, in the ordinary course of a partitioned parallel run.
Per `docs/dev/rank-agreement.md`, an unhandled throw on one rank while its
partners are already inside a collective is exactly the deadlock class `Epoch`
exists to prevent; a pipeline-layer refusal that a caller has no reason to expect
(the registry told them this was fine) is a live path to that deadlock, not merely
a wrong-but-caught error.

**The fix.** `FieldSpan::from()`'s guard should mirror `bind()`'s exactly:

```cpp
if (!f.valid() && f.layout().count != 0) {
  FUSE_REFUSE("E-FIELD-NULL", ...);
}
```

and when `count == 0`, build the (empty) `Kokkos::LayoutStride`/`View` the same
way it already does for a non-empty field — `l.count == 0` flows harmlessly
through `Kokkos::LayoutStride(0, l.stride, components, cs)` and produces a
zero-extent unmanaged view, which every `FieldSpan` consumer (`slice`,
`local_sum`, `fill`) already handles correctly for an empty range. Then correct
`test/field/pipeline_test.cpp:179-182` to construct a genuinely-null, non-empty
`FieldRef` (e.g. `Layout{.stride=1, .count=4}` with a null pointer, matching
`FieldRegistry.RejectsNullData`'s own fixture) so the test asserts what it claims
to.

**Test to add.** `Pipeline.AnEmptyFieldFromABoundZeroCountFieldRefDoesNotRefuse`:
bind (or directly construct) a `FieldRef(nullptr, Layout{.stride=1, .count=0,
.components=k}, ...)`, call `FieldSpan<>::from()` on it, and assert it succeeds
with `element_count() == 0`, plus that `slice`/`local_sum`/`fill` on the result
are no-ops rather than throws.

### A1 — MAJOR — `Layout::well_formed()` over-rejects valid padded/strided layouts

**Where:** `include/fuse/field/field_ref.hpp:97-98`, enforced unconditionally by
`FieldRegistry::bind()` at `registry.hpp:59-68` ("in every build configuration"
per the class doc at `field_ref.hpp:78-79`).

**What is wrong.** The disjointness check is:

```cpp
const bool interleaved = stride >= detail::sat_mul(nc, cs);            // field_ref.hpp:97
const bool blocked = stride >= 1 && cs >= detail::sat_mul(count, stride); // field_ref.hpp:98
```

The actual, necessary-and-sufficient condition for element `i`'s component block
`[i*stride, i*stride + (nc-1)*cs]` to never overlap element `i+1`'s block starting
at `(i+1)*stride` is `stride > (nc-1)*cs`, i.e. `stride >= (nc-1)*cs + 1` — **not**
`stride >= nc*cs`. Symmetrically for the blocked case the correct bound is
`cs >= (count-1)*stride + 1`, not `cs >= count*stride`. The code as written is
one full `cs` (interleaved case) or one full `stride` (blocked case) more
conservative than necessary. The two bounds only coincide with the code's
stricter version when `cs == 1` (interleaved) or `stride == 1` (blocked) — i.e.
the discrepancy is invisible for the default, tightly-packed layouts and only
bites a padded/strided layout: `component_stride > 1` (padded AoS, e.g. for SIMD
alignment) or element `stride > 1` (a blocked/SoA view into a wider, padded
array — the doc's own `blocked (x0 x1 y0 y1 z0 z1): stride = 1, component_stride
= 2` example is exactly at the boundary where the discrepancy first appears for
non-unit `cs`).

**How I confirmed it.** Extracted `well_formed()` verbatim (`sat_mul`/`sat_add`
included) into a standalone program and checked it against a brute-force,
all-pairs-offset disjointness test on two concrete, hand-picked layouts:

- `{stride=2, component_stride=5, count=3, components=2}` — offsets are
  `{0,2,4}` (component 0) and `{5,7,9}` (component 1): genuinely disjoint.
  `well_formed()` returns `false`.
- `{stride=5, component_stride=2, count=4, components=3}` — genuinely disjoint
  (checked exhaustively). `well_formed()` returns `false`.

Both are real false negatives, confirmed by brute force, not merely derived
algebraically. Then checked `test/runtime/field_ref_test.cpp:264-294`'s three
`Layout.WellFormed*` tests: every accepted case uses `component_stride ∈ {0,1,2}`
paired with `stride` chosen so the two conditions coincide (`cs=1`), or `cs=1`
combined with a wider `stride` for the "gaps are fine" case — none of the
existing tests exercises the `cs > 1` boundary for the interleaved rule or the
`stride > 1` boundary for the blocked rule, so this gap is real and unexercised,
not merely theoretical.

**Impact.** A component registering a padded/aligned interleaved or blocked
layout — a routine choice in HPC codes for SIMD/cache-line alignment — that is
at or near the actual disjointness boundary is wrongly refused by `bind()` with
`E-FIELD-LAYOUT`, even though the layout does not alias and would compute
correctly. This is a refusal, not memory corruption, but it is reachable in
ordinary use of the field layer as documented (the class's own blocked example
uses `component_stride = 2`), so it is classified MAJOR rather than MINOR.

**The fix.** Tighten both bounds by one unit:

```cpp
const bool interleaved = stride > detail::sat_mul(nc - 1, cs);
const bool blocked = stride >= 1 && cs > detail::sat_mul(count - 1, stride);
```

(using `sat_add(detail::sat_mul(nc - 1, cs), 1)` etc. if a `>` comparison against
a saturated value is thought less clear than restating it as `>=`). Re-run the
existing `Layout.WellFormed*` suite unchanged (all of it stays correct under the
tighter bound, since the tighter bound accepts a strict superset of what the
current one accepts) and add the two boundary cases below.

**Tests to add.** `Layout.WellFormedAcceptsATightPaddedInterleavedLayout` with
`{stride=5, component_stride=2, count=4, components=3}` (currently a false
negative, see above) and `Layout.WellFormedAcceptsATightPaddedBlockedLayout`
with `{stride=2, component_stride=5, count=3, components=2}` (ditto), both
asserting `well_formed() == true`; and a boundary case one unit tighter than
each (`stride` or `cs` reduced by exactly 1) asserting `well_formed() == false`,
so the fixed bound is pinned exactly, not just loosened.

### A2 — MINOR — `least_squares_into`'s `m >= k` precondition is silently dropped, and the `m < k` path it enables is real but untested

**Where:** `include/fuse/linalg/least_squares.hpp:72` (documented precondition on
the public solve) vs. `src/linalg/least_squares.cpp` (`least_squares_into`, which
implements the shared QR core but never calls `check_shape`), called with
`m = n_ < k = blk.cols` possible at `src/num/accel_quasi_newton.cpp:536` and with
`m = n < k = cols_` possible at `src/num/accel_quasi_newton.cpp:632`.

**What is wrong.** `least_squares.hpp:72` documents "Preconditions: `a.size() >= m
* k`, `b.size() >= m`, `m >= k`. Violated, they are a caller mistake and throw
rather than returning a rank" — enforced via `check_shape()`, which
`least_squares_builtin` and `least_squares_eigen` both call
(`least_squares.cpp`). `least_squares_into` — the scratch-supplied variant the
accelerator's hot loop calls at `accel_quasi_newton.cpp:536` (inside
`apply_prev_jacobian`, with `m = n_`, `k = blk.cols`) and `:632` (inside
`accelerated_step`, with `m = n_`, `k = cols_`) — never calls `check_shape` and
never checks `m >= k` itself.

`cols_`/`blk.cols` is bounded only by `max_columns_` (`cfg.reuse`, validated to
`[1, 32]`, default `8` per `include/fuse/model/iteration.hpp:302`), completely
independent of `n` (the interface vector length, which can legitimately be as
small as 1 — a single scalar exchange). So `m < k` at these two call sites is
not a hypothetical: any coupling with a small interface and the default
`reuse = 8` reaches `cols_ = 2 > n = 1` on the second call of the very first
window.

**How I confirmed it.** First read the QR loop (`least_squares.cpp`'s
`least_squares_into`) to check whether skipping the `m >= k` check leads to an
out-of-bounds read/write: `steps = min(m, k)`, and every loop bound (`for j <
steps`, pivot search over `[j, k)`, Householder rows `[j, m)`) stays within the
`a`/`b` buffers regardless of whether `m < k`, so there is no buffer overrun by
inspection. To be certain rather than confident, I extracted the exact algorithm
into `/tmp/lstest/test_ls.cpp` and ran it under `-fsanitize=address,undefined`
with `m=1, k=8` and `m=2, k=8` random inputs: both ran clean under ASan/UBSan and
returned a valid rank-truncated least-squares solution (residual `~1e-16`, as
expected for a consistent, under-determined system). So the behavior is safe,
just different from — and less validated than — the public entry point's
documented contract for the same shape.

Then checked whether this shape is exercised anywhere in
`test/num/accelerator_test.cpp`: `RespectsTheReuseLimit` uses `reuse=3, n=4`
(`m > k` throughout); `ColumnCountIsBoundedOverAVeryLongIteration` uses
`reuse=4` against a `ladder_problem()` whose `n` is not 1; every `n=1` test
(`DivergenceFallsBackToTheUnderRelaxedStepWhenItSurvives`,
`ImvjDropsARetainedJacobianThatIsWhatDiverged`, and others found by grepping for
`, 1,` / `out(1` in the file) runs exactly two iterations per window before
`reset()`, so `cols_` never exceeds `1`. No test in scope drives `cols_` or
`blk.cols` past `n_`.

**Why MINOR and not higher.** Confirmed safe by direct execution (no UB, no
wrong-shaped output, a defensible rank-truncated answer), so this is a testing
gap and a documentation/consistency gap, not a live correctness defect. Flagged
because `docs/dev/testing.md`'s five states name exactly this shape of problem
("Implemented, untested... a design document is not evidence that the contract
exists") and because the divergence between what `least_squares_into` actually
tolerates and what its own header documents as the shared contract is the kind
of drift `docs/dev/porting.md`'s "a fact... is a dated observation" rule warns
against for exactly this reason — the current safety is something I verified by
extraction and execution, not something the code states or a test protects.

**The fix.** Either state explicitly on `least_squares_into` that `m < k` is
supported and returns a rank-`m`-capped truncation (updating the doc comment at
`least_squares.hpp:99-102` to say so, since right now it only says "same code,
same arithmetic" without saying which preconditions carry over), or add a
debug-only assertion (`FUSE_DEBUG_ASSERT`, per `conventions.md`'s error taxonomy)
that documents the intent without adding overhead to the release hot path.

**Test to add.** `Accelerator.HandlesAnInterfaceSmallerThanTheReuseLimit`: a
single scalar (`n=1`) or two-component (`n=2`) `IQN_ILS`/`IQN_IMVJ` accelerator
with the default `reuse=8`, run for enough iterations within one window that
`cols_` provably exceeds `n` (e.g. 3+ iterations with genuinely distinct
residuals), asserting `update()` still succeeds and returns a finite, sane
iterate — closing the gap identified above at the `Accelerator` level rather
than only at the `linalg` level.

### I1 — IMPROVEMENT — DG/Spline `Waveform::evaluate`/`integrate` redo shared, per-call work once per field element

**Where:** `include/fuse/num/waveform.hpp:478-497` (`evaluate_dg`,
`evaluate_spline`) and `:536-556` (`integrate_dg`, `integrate_spline`).

**What is wrong.** `local_times()` is correctly hoisted out of the per-element
loop (called once, e.g. `waveform.hpp:478`), but the design-matrix-and-elimination
work inside `fit_polynomial` (`temporal_basis.hpp`) and the tridiagonal Thomas
solve inside `cubic_spline_moments` depend **only** on the shared sample times
`tt`, not on the per-element values `ff` — yet both are called fresh inside the
`for (e = 0; e < elements_; ++e)` loop, at `waveform.hpp:484`
(`num::fit_polynomial(tt, ff, deg, 0.0)`), `:495`
(`num::cubic_spline_moments(tt, ff)`), `:543`, and `:556`. For `E` field
elements and `m = deg+1 <= 5` basis functions, this repeats the `O(n*m^2 + m^3)`
normal-equations build-and-eliminate (DG) or `O(n)` tridiagonal factor (Spline)
`E` times instead of once, and each call allocates its own scratch vectors
(`a`, `rhs`, `phi`, `c` in `fit_polynomial`; the tridiagonal arrays in
`cubic_spline_moments`) rather than reusing shared storage.

**Why IMPROVEMENT and not MAJOR/BLOCKER, given I5.** `docs/dev/invariants.md`'s
`I5` ("No allocation after setup... enforced by an allocation-count fence
around the step loop") is exactly the kind of rule this pattern would violate
once `Waveform` is on a per-iteration hot path. I checked whether it already is:
`grep -rn "\.evaluate(\|\.integrate(\|\.mean(" src include` outside
`waveform.hpp`/`temporal_basis.hpp`/tests returns nothing — `Waveform` is not
yet called from anywhere in `backend`, `transport`, or `runtime` (consistent
with the project status: space-time projection is a not-yet-landed stage). So
today this is dead weight on a path nothing calls repeatedly, not a live I5
violation; it becomes one the moment the space-time projection operator is
wired in, and no test here would catch it (`waveform_test.cpp`/`waveform_order_test.cpp`
have no allocation-fence test, unlike `Accelerator.SteadyStateUpdateAllocatesNothing`
in `test/num/accelerator_test.cpp`).

**The fix.** Factor `fit_polynomial`'s normal-equations build+elimination into a
"prepare" step keyed only on `tt`/`degree`/`t0` (e.g. returning the factored
matrix and permutation), with a cheap per-element "solve for this rhs" step;
similarly split `cubic_spline_moments`'s tridiagonal factorization (the
`lo`/`di`/`up` arrays) from its per-element forward/back substitution. Both
`Waveform::evaluate_dg`/`integrate_dg` and `evaluate_spline`/`integrate_spline`
would then build the shared factorization once per call and loop only the
`O(n)` substitution per element, dropping the `E`-fold redundancy and the
per-element allocations at the same time.

**Test to add.** Once split, an allocation-count fence test analogous to
`Accelerator.SteadyStateUpdateAllocatesNothing`, exercising a multi-element (e.g.
`elements = 4096`) DG or Spline `Waveform` and asserting a bounded, `elements`-independent
allocation count for `evaluate`/`integrate` after the first call for a given
sample count — mirroring `docs/dev/testing.md`'s "measured, not asserted in a
comment" standard already applied to `Accelerator`.

### D1 — DEVIATION — `num.convergence` test is entirely `fuse::model`, not `fuse::num`

**Where:** `test/CMakeLists.txt:107-109` (registers `convergence` under the
`num` layer) and `test/num/convergence_test.cpp:1-13` (its actual includes and
subjects).

**What is wrong.** `docs/dev/layering.md`'s "A boundary the gate moved" section
narrates, as a recorded decision, that `ConvergenceMeasure`, `ConvergenceCriterion`,
`AccelKind`, `AccelConfig`, and `fuse::Converged` moved from `num` to
`fuse/model/iteration.hpp` during the compiler port, specifically because "the
layering gate rejected the port on the first run" when they lived in `num`. That
move is real and correctly reflected in `include/fuse/num/accelerator.hpp:19-23`,
which explicitly says these types are declared in `fuse/model/iteration.hpp`, not
in `num`. But `test/CMakeLists.txt:107-109`:

```cmake
foreach(t temporal_basis waveform convergence accelerator)
  fuse_add_gtest(num ${t} THROWS)
endforeach()
```

still registers `convergence` under the `num` layer (`fuse_num_convergence_test`,
CTest name `num.convergence`, label `num`), and its source,
`test/num/convergence_test.cpp:2`, includes only `<fuse/model/iteration.hpp>` and
tests exclusively `fuse::ConvergenceCriterion`, `fuse::ConvergenceMeasure`,
`fuse::MeasureKind`, and `fuse::Converged` — nothing from `fuse::num` appears
anywhere in the file. `test/model/` has no corresponding test file for
`iteration.hpp` (confirmed by listing its contents: `builder_test.cpp`,
`channel_test.cpp`, `compatibility_test.cpp`, `component_test.cpp`,
`connect_test.cpp`, `coupling_graph_test.cpp`, `descriptor*_test.cpp`,
`multiscale_test.cpp`, `region_test.cpp`, `semantics_test.cpp`,
`transform_test.cpp`, `validate_test.cpp`, `yaml_io_test.cpp` — no
`iteration_test.cpp`). So this is the *only* test for those types, and it is
filed under the wrong layer both by directory and by CTest label.

**Why this matters, concretely.** `docs/dev/testing.md`'s CI-legs section states
the project's own rule: "one leg per rung of the ladder, so no configuration can
rot unnoticed," and `ctest -L <layer>` / a per-layer coverage audit is exactly
the mechanism that rule depends on. A reviewer or tool asking "does
`fuse/model/iteration.hpp` have test coverage" via directory or label search
finds none; a reviewer asking "what does `num.convergence` cover" would
reasonably assume it is testing `num`. This is precisely the "gates that scan
some directories but not others" failure mode this review was asked to look
for, just at the test-registration layer rather than the header-scanning layer.

**The fix.** Move `test/num/convergence_test.cpp` to `test/model/iteration_test.cpp`
(or `test/model/convergence_test.cpp`) and its CMake registration from the
`num`-layer `foreach` to a `model`-layer one, linking `fuse::model` instead of
`fuse::num`. No behavioral change; this is a location/labeling fix only, and
should be recorded in `docs/dev/layering.md` alongside the existing note about
the type move, since that note is exactly what makes the current test location
look like an oversight rather than a decision.

## 4. Not found

Checked for, specifically, and did not find:

- **`FUSE_HAVE_*`/`FUSE_ENABLE_*` read without `config.hpp`, or read before it,
  or `#ifdef` on a `#cmakedefine01` macro**, anywhere in the nine in-scope
  headers/sources. `least_squares.cpp`'s `#if FUSE_LINALG_EIGEN` is preceded by
  `#include "fuse/linalg/least_squares.hpp"`, which itself includes
  `fuse/base/config.hpp` before any use — correct, and confirmed by a clean
  `-fsyntax-only` compile with the real generated `config.hpp`
  (`FUSE_LINALG_EIGEN=1` in this build). Note `tools/check-invariants.py`'s
  `FEATURE_MACRO` regex only matches `FUSE_HAVE_*` and
  `FUSE_ENABLE_EXCEPTIONS` — it does **not** match `FUSE_LINALG_EIGEN`, so this
  particular file's correctness on this point is not actually gate-enforced;
  it happens to be right by inspection and by compilation, not because the gate
  checked it. Worth a one-line addition to that regex, though I found no
  violation for it to currently catch.
- **`fmt`-only formatting / no bare `iostream` / no `printf`** in library code:
  `detail::format_accel_double` (`accel_relaxation.cpp:26-32`) uses
  `std::ostringstream` but pins `os.imbue(std::locale::classic())`, matching the
  documented `iostream` exception in `conventions.md`. No other `<sstream>`,
  `<iostream>`, or `printf`/`sprintf` family call appears in any in-scope
  library source (`kernels_test.cpp`/`space_kernels_test.cpp` use
  `std::printf` for a test-only progress line, which is test code, not library
  code, and is not subject to the rule).
- **Bare `throw std::...` or bare `assert`**: every error path in scope uses
  `FUSE_THROW`, `FUSE_REFUSE`, or returns `Result`/`Failure`; no `assert(` call
  and no `throw std::` call appears in any in-scope header or source.
- **`mpi.h`, Eigen, KokkosKernels, MOAB, PETSc, or HDF5 types in a public
  header outside their own backend directory (I7)**: `least_squares.cpp` is a
  `.cpp`, not a public header, so its guarded `#include <Eigen/Dense>` does not
  violate I7; no in-scope *header* includes any of these.
- **Upstream namespaces/prefixes (`soma::`, `KARMA*`, `karma_*`/`soma_*` file
  names, `.C` extension) (I11)**: none in scope.
- **Kokkos device-callable code calling a host-only facility** (fmtlog,
  exceptions, `std::string`) inside a `KOKKOS_LAMBDA`: every lambda in
  `space_kernels.hpp` (`accumulate`, `relax`, `gather`, `scatter`,
  `remap_apply`, `dot`, `weighted_norm_squared`) does only arithmetic and
  `View` indexing; none allocates, throws, or logs. Confirmed by actually
  instantiating every one of these templates on `Kokkos::Serial` and getting a
  clean compile (see Coverage), not just by reading them.
- **Allocation inside a Kokkos per-step kernel, or per-value virtual dispatch
  in one**: none of the seven kernels in `space_kernels.hpp` allocates inside
  its `parallel_for`/`parallel_reduce` body, and none dispatches through a
  virtual call inside the range.
- **A collective reachable by only some ranks, or a rank-agreement gap**:
  nothing in this scope issues a collective at all (`num`/`linalg`/`field` are
  explicitly below `transport` in the layer order and depend on no MPI); the
  rank-agreement invariant (`I2`) does not apply to any code read here.
- **`Result<T>`/`Failure` vs. throw-and-catch as internal control flow (the
  KNOWN_ISSUES B1a history)**: `Accelerator::update`, `least_squares`,
  `Waveform::push`/`evaluate`/`integrate`, and `field_norm`/`combine` all
  return `Result`/`Failure` for reportable failures and reserve `FUSE_THROW`
  for genuine programmer-error preconditions (documented as such in each
  case); none of the in-scope code catches its own thrown exception as a
  control-flow mechanism.
- **A test that passes vacuously, or a deliberate refusal never made to
  fire**: spot-checked `AZeroLimitDisablesEveryFilter`,
  `Qr2NeverEmptiesANonEmptyHistory`, `AnUnboundFieldIsRefusedNotDereferenced`
  (the last of which does fire, on the wrong justification — see Finding B1),
  `RefusesAShapeItCannotSolve`, `MeanRefusesWhenTheSamplesDoNotSpanTheWindow`,
  and `WellFormedRejectsAliasingAndNonPositiveComponents`: each of these
  actually exercises the refusal path (confirmed by reading the assertion
  against the exact input, not just the test name) with the single exception
  above.
- **A claim written before verification**: no in-scope doc comment makes a
  numeric or behavioral claim I could not independently confirm either by
  reading the callee or by compiling/executing a minimal reproduction; where a
  doc comment's claim did not hold under direct execution (the `jacobian_cap()`
  "min(reuse, kMaxJacobianBlocks)" formula initially looked wrong against the
  class-level doc on `kMaxJacobianBlocks` alone), the *test*
  (`Accelerator.ImvjRetainedJacobianIsBounded`, `accelerator_test.cpp:1642-1690`)
  states the `min(...)` formula explicitly and verifies it at `reuse=2` and
  `reuse=8`, which resolved it as intentional, not a defect — recorded here so
  the reasoning isn't silently discarded.

Not attempted, and stated as a scope limit rather than a silent gap: no MPI/parallel
behavior was assessed (none of the in-scope code touches MPI, consistent with the
layer table), and I did not audit `fuse::model::iteration.hpp`'s own logic
(`ConvergenceMeasure`/`ConvergenceCriterion` internals) beyond confirming Finding
D1's location mismatch — that header is outside the assigned scope.

## 5. Suggested new tests / mutations

In addition to the tests named inline under each finding above:

- **Mutation for B1**: flip `FieldSpan::from`'s guard to the corrected
  `!f.valid() && f.layout().count != 0` and confirm
  `Pipeline.AnEmptyFieldFromABoundZeroCountFieldRefDoesNotRefuse` (new) goes
  from failing to passing, and that the corrected
  `AnUnboundFieldIsRefusedNotDereferenced` (using a non-empty null `FieldRef`)
  still fires — this is the pair that proves the fix narrows the refusal rather
  than removing it.
- **Mutation for A1**: revert the tightened bound to the current
  `stride >= sat_mul(nc, cs)` / `cs >= sat_mul(count, stride)` and confirm the
  two new boundary-accepting tests (`WellFormedAcceptsATightPadded*Layout`) go
  from passing to failing — this is the named mutation `I9` requires per
  ported/new kernel, recorded under `build-evidence/` once landed.
- **`linalg::LeastSquares.IntoAgreesWithBuiltinWhenMIsLessThanK`**: call
  `least_squares_into` directly (bypassing the accelerator) with `m=1,k=4` and
  `m=2,k=6` against a random well-conditioned wide matrix, and assert the
  returned rank is `<= m` and the residual is at or near round-off — this
  turns the ad hoc ASan/UBSan check I ran outside the repository (Finding A2)
  into a durable, repository-resident regression test.
- **`Waveform.DgAndSplineGiveTheSameAnswerAfterTheFitIsSharedAcrossElements`**:
  once Finding I1's factoring lands, a regression test that the per-element
  loop's output is bit-identical before and after the refactor on a
  multi-element fixture — protects the performance fix from silently changing
  the numerics.

## Summary

**1 BLOCKER, 1 MAJOR, 1 MINOR, 1 IMPROVEMENT, 1 DEVIATION** (5 findings total).

The BLOCKER (B1) is a cross-file contract mismatch between `FieldRegistry::bind()`
and `FieldSpan::from()` over what a null-pointer, zero-count `FieldRef` means:
the registry was explicitly fixed to treat it as the ordinary case of a rank
owning nothing, but the pipeline's only entry point still refuses it
unconditionally, backed by a test that cannot distinguish the two states it
claims to distinguish — so every rank with a genuinely empty partition of a
bound field throws on first use of the pipeline. The MAJOR (A1) is a confirmed
(brute-force-checked) off-by-one-stride-unit over-restriction in
`Layout::well_formed()` that wrongly refuses valid padded/strided layouts at
`bind()` time. The MINOR (A2) is a real, currently-safe (ASan/UBSan-verified)
but wholly untested code path where the accelerator's hot-path least-squares
solve is called with fewer rows than columns, silently diverging from the
precondition its own header documents for the public entry point. The
IMPROVEMENT (I1) is a not-yet-reachable but foreseeable I5 (no-allocation)
violation in `Waveform`'s DG/Spline paths, flagged before the space-time
projection operator that will call them repeatedly is wired in. The DEVIATION
(D1) is a test misfiled under the `num` CI layer/label when its entire subject
is `fuse::model`, undermining the project's own per-layer coverage auditing.
