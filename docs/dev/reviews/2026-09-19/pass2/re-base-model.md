<!-- Second-pass review, HEAD dbdc872, branch review-2026-09-19-fixes. Verifies
     docs/dev/reviews/2026-09-19/fixes/fix-base-model.md against the code, then
     re-reviews include/fuse/base, src/base, test/base, include/fuse/model,
     src/model, test/model fresh. Read-only: nothing under the repo path was
     modified; build performed in a separate working directory. -->

# FUSE second-pass review: `base` and `model` layers

HEAD reviewed: `dbdc872`. First review: `bdf4346`
(`docs/dev/reviews/2026-09-19/base-model.md`, ids `BM-*` / unified `R-*`).
Fixer report: `docs/dev/reviews/2026-09-19/fixes/fix-base-model.md`.

**Build.** `cmake --preset tier0 -S <repo> -B $PWD/b0 && cmake --build b0 -j4`
succeeded clean. All 22 `base.*`/`model.*` ctest cases pass. `tools/mutate.py
--check-anchors` reports the 7 new base/model mutation anchors valid, and
`build-evidence/base-mutations.md` / `model-mutations.md` (run 2026-09-19)
record all 7 as **KILLED**. `tools/check-invariants.py` and
`tools/check-layering.py` were not re-run standalone (out of the time budget
for this pass); the fixer report states 0 violations from its own run and
nothing in the diffs read below touches a pattern either tool gates.
Additionally configured `tier1` (`FUSE_ENABLE_MPI=ON`, MPICH 5.0 found) and
built `fuse_model_descriptor_mpi_test` specifically to check the one
MPI-only test in this scope: it compiles cleanly. Running it aborts inside
`MPI_Init_thread` (confirmed directly: `pthread_mutex_destroy` /
`MPIR_Assert_fail` in `dbginit.c`), consistent with "MPI cannot initialize
here" — running it is a host action, as the fixer report says.

For three findings whose fix report deferred verification to a host, I went
further than re-reading the diff:

- **R-16**: compiled `error.cpp`/`error_test.cpp` standalone under
  `-fsanitize=thread` (AppleClang 21, available in this sandbox) and ran
  `Error.ErrorPolicyIsSafeUnderConcurrentReadAndWrite` under it — TSan
  reports no race. This is evidence beyond what the fix report claims (it
  only ran the test under a plain build). I additionally tried to reproduce
  the *pre-fix* shape (an unlocked reference read racing a locked
  copy-assignment) in an isolated ~40-line program to show TSan would have
  caught it; two attempts (short strings, then longer ones forcing heap
  allocation, 2000 and 200000 iterations) did not trigger a report. Recorded
  as inconclusive, not as evidence against the finding: a plain
  reference-into-shared-storage read racing a mutex-guarded write is a data
  race on the two `std::string` members by the C++ object model regardless
  of whether a short timing window happens to reproduce it, and the fix
  (return a lock-guarded copy) removes the race by construction — the two
  operations can no longer overlap.
- **R-04/R-19**: reproduced the two isolated-compile-and-run confirmations
  the first review used (`std::lcm` overflow, out-of-range enum cast into a
  raw array) are superseded by the actual fix; I did not need to re-run
  those since the ctest suite now exercises the real code paths directly
  (`Descriptor.RejectsCorruptedIterationRegionEnums`,
  `Rational.LcmDenominatorRefusesAnUnrepresentableCommonTick`,
  `Descriptor.RejectsNonPositiveUnitScale`), all passing.
- Two "Not found"/gap items from the first review (Window::Kind's unchecked
  read, and the general shape of R-18) turned out to still be live gaps
  under the fix; see New findings below, both confirmed by compiling and
  running an isolated reproduction against the built `fuse-coupler-model`/
  `fuse-coupler-base` static libraries (source in the session; not part of
  the repo).

## Verification

| id | verdict | evidence |
|----|---------|----------|
| R-04 (BM-B1, BLOCKER) | CONFIRMED-FIXED | `include/fuse/model/descriptor.hpp:359-410` now refuses `E-DESC-ENUM` for an out-of-range `MeasureKind`/`AccelKind`/`FilterKind` byte, mirroring the pre-existing `MergeOp` check; `src/model/yaml_io.cpp`'s three raw array indexes are replaced by a bounds-checked `enum_name()` helper (defense in depth). New test `Descriptor.RejectsCorruptedIterationRegionEnums` patches each of the three enum bytes in a known-good serialized buffer and asserts refusal, plus confirms the unpatched buffer still round-trips (pins the byte offsets against the deserializer, not against themselves). Passes under `ctest -R model.descriptor`. Mutations `descriptor-accepts-any-convergence-measure-kind`/`-acceleration-kind`/`-acceleration-filter` are logged KILLED in `build-evidence/model-mutations.md`. |
| R-15 (BM-M1, MAJOR) | CONFIRMED-FIXED | `include/fuse/base/rational.hpp:216-224`: `lcm_denominator()` now routes through `detail::Int128` (divide-before-multiply, `check_fits_i64`), the same route `try_gcd_rational()` already used, and `FUSE_REFUSE`s rather than admitting a wrapped value. New test `Rational.LcmDenominatorRefusesAnUnrepresentableCommonTick` (added in a follow-up commit, `07f16ae`, after the first fix commit landed without it — both are present at HEAD) uses the exact two denominators (3037000493, 3037000507) the review's reproduction used, asserts `fuse::Refusal`, and separately checks a value just below the overflow bound is exact. `base.rational` passes; mutation `rational-lcm-denominator-overflows` (reverts to the direct `std::lcm` call) is logged KILLED. |
| R-16 (BM-M2, MAJOR) | CONFIRMED-FIXED | `error_policy()` (`src/base/error.cpp:74-77`, declaration `include/fuse/base/error.hpp:119`) now takes `policy_mutex()` and returns a copy instead of a racing `const&`; every call site binds to `const auto&`/`const ErrorPolicy&`, which is well-formed and lifetime-safe against a returned-by-value temporary. New test `Error.ErrorPolicyIsSafeUnderConcurrentReadAndWrite` runs a writer thread looping `set_error_policy()` against a reader thread looping `FUSE_REFUSE`; I additionally compiled and ran it under `-fsanitize=thread` independently of the fixer's own build (see above) — no race reported. The fix report's own "Needs a host run" framing (TSan is not one of this repo's presets) is accurate as a statement about *this repo's CI*, but the check itself is runnable in this sandbox with a standalone compile, which I did. |
| R-17 (BM-M3, MAJOR) | REJECTION-UPHELD — agree | Re-derived the git history independently: `test/model/component_test.cpp`'s `ComponentDecl.DefaultsToUnitIntervalOnRankZero` (asserting `c.ranks.min_rank()==0`, `c.ranks.size()==1` for a default-constructed `ComponentDecl`) was added in the initial port commit `6632d0e`, before `RankSet` existed as a class (it was `struct RankSet { int first=0; int count=1; }` — also rank 0). `a71523e` ("Make a component's placement an explicit rank list") introduced the class-based `RankSet` with the two-argument constructor's documented `count<=0` empty-set contract, and did **not** touch that test — confirming the niladic default's rank-0 behavior was carried forward deliberately, not left over by accident. Separately, `RankSet.AnEmptyPlacementIsLegal`'s own comment ("a component may be declared and placed nowhere... not an error") describes a legitimate use the review's suggested `validate()` addition would have flagged as an error. `include/fuse/model/component.hpp:38-44` gained only an explanatory comment; I agree no code change was warranted. |
| R-18 (BM-M4, MAJOR) | PARTIAL | Four of the (at least) six unguarded `.at()`-based dereferences in `to_yaml()` are now guarded: region→mesh, channel→source/target field, merge→field, iteration-region→channel (all confirmed present in `src/model/yaml_io.cpp` and passing `model.yaml_io`'s `YamlIo.EmittingAGraphWithADanglingChannelReferenceReturnsFailure`). Two are not: a channel's `accumulate.window`/`spacetime.window` naming an out-of-range `ComponentId` (`emit_window`, `src/model/yaml_io.cpp:193`, `g.component(w.component).name`), and a field's `region` naming an out-of-range `RegionId` (`src/model/yaml_io.cpp:331`, `g.region(f.region).name`). Both still throw uncaught `std::out_of_range` out of `to_yaml()` — the exact defect R-18 was opened to fix. Confirmed by compiling and running two isolated reproductions against the built model/base libraries (see New findings N-base-model-1). |
| R-19 (BM-M5, MAJOR) | PARTIAL | The three chokepoints the fix targeted are correctly gated and tested: `semantics.cpp:359-366`'s `NegativeScale` diagnostic (distinct from the pre-existing `NonfiniteScale`/zero case), `descriptor.hpp:317-323`'s `E-DESC-UNITS-SCALE` refusal on the binary deserializer, and `compatibility.cpp:198-208`'s `UnitScaleSignNotRequested` gate ahead of the pre-existing automatic-scale branch, all with passing tests (`Semantics.StructuralValidation`, `Descriptor.RejectsNonPositiveUnitScale`, `Compatibility.UnitRules`) and 3 KILLED mutations. Not gated: `from_yaml()`'s `parse_units()` (`src/model/yaml_io.cpp:100`) parses a field's `scale` with no sign or zero check at all, so a YAML-authored graph with a negative (or zero) unit scale is accepted silently — the identical failure mode R-19 fixed for the binary descriptor, left open on the text front end. Confirmed by an isolated `from_yaml()` reproduction (see New findings N-base-model-2). |
| R-34 (BM-M6, MINOR) | CONFIRMED-FIXED | `include/fuse/model/iteration.hpp:80-88`'s doc comment no longer points at a nonexistent "header comment"; it now describes the zero-reference fallback, non-finite handling, and OR-only `AbsoluteOrRelative` semantics inline, matching `satisfied()`'s actual implementation below it. The second citation in the original finding (line ~127, "satisfied() exactly, including the zero-reference fallback") was never itself dangling — it references `satisfied()`, a real function in the same file — so leaving it untouched is correct, not an oversight. |
| R-35 (BM-M7, MINOR) | CONFIRMED-FIXED | `src/base/error.cpp:203-211`: `fuse_terminate()` now calls `g_previous()` (if non-null) after writing FUSE's own diagnostic, before the `std::abort()` fallback. New death test `ErrorDeath.TerminateHandlerChainsToAPreviouslyInstalledHandler` installs a marker handler via `std::set_terminate()` before `fuse::install_terminate_handler()` inside its own forked child and checks the death output contains both FUSE's report and the marker text. The test's implicit assumption — that `g_installed`'s `call_once` flag is still unset when this test's child forks — holds structurally: every call to `install_terminate_handler()` in this file happens inside an `EXPECT_DEATH` block (grepped: all 5 occurrences), so the parent process, which is what every child forks from, never installs it itself. |

## New findings

| id | severity | file:line | title |
|----|----------|-----------|-------|
| N-base-model-1 | MAJOR | `src/model/yaml_io.cpp:193,331`; `include/fuse/model/descriptor.hpp:114,307` | R-18's `to_yaml()` dangling-reference guard is incomplete: a channel's window-component and a field's region still throw uncaught `std::out_of_range` |
| N-base-model-2 | MAJOR | `src/model/yaml_io.cpp:100` | R-19's non-positive unit-scale refusal covers the binary descriptor but not `from_yaml()`'s text front end |

### N-base-model-1 — MAJOR — R-18's fix left two more unguarded `.at()` dereferences in `to_yaml()`

`src/model/yaml_io.cpp:193` (`emit_window`, reached from channel emission at
lines ~380 and ~401 for `accumulate.window`/`spacetime.window`) and
`src/model/yaml_io.cpp:331` (field emission).

R-18's own justification for guarding `to_yaml()` was that it "dereferences
unvalidated ids via `.at()`" reachable through raw API construction that
`validate()` — not `to_yaml()` — is meant to catch, and that nothing forces
`validate()` to run first. That reasoning applies unchanged to two sites the
fix did not touch:

```cpp
// src/model/yaml_io.cpp:193, inside emit_window()
case Window::Kind::FromComponent:
  e << YAML::Flow << YAML::BeginMap
    << YAML::Key << "kind" << YAML::Value << "from_component"
    << YAML::Key << "component" << YAML::Value
    << g.component(w.component).name          // .at()-based, unguarded
```

```cpp
// src/model/yaml_io.cpp:331, inside the component/field emission loop
<< YAML::Key << "region" << YAML::Value << g.region(f.region).name   // .at()-based, unguarded
```

Both `w.component` (a `Window`'s `ComponentId`, set via `ChannelBuilder::
accumulate()`/`spacetime()`, `include/fuse/model/coupling_graph.hpp:306-333`)
and `f.region` (a `FieldSpec`'s `RegionId`, set via `add_field()`,
`include/fuse/model/coupling_graph.hpp:123-129`) are accepted from the
caller with no bounds check at construction time — `add_field()` asserts
only that the *owning component* id is valid, never that `f.region` is;
`ChannelBuilder::accumulate`/`spacetime` do not touch `w.component` at all
beyond storing it. `validate.hpp` itself documents the window half of this
exactly: `window_component_ok()`'s doc comment states plainly that "an
out-of-range (but not sentinel) id... reachable either from direct-API
misuse or from a hand-patched descriptor, since `deserialize()` does not
itself validate embedded window component ids — must be caught here before
any `.at()`-based lookup can throw" (`include/fuse/model/validate.hpp:20-24`).
`validate()` and the defensive `channel_window()` helper both use that guard
correctly; `to_yaml()` does not use it at all, for either the window or the
region case.

**Confirmed reachable two ways**, matching R-18's own reachability
argument:

1. Raw API construction (the same technique the R-18 fix's own new test,
   `YamlIo.EmittingAGraphWithADanglingChannelReferenceReturnsFailure`, uses
   for the four sites it does cover). Compiled and ran a program against the
   built `libfuse-coupler-model.a`/`libfuse-coupler-base.a`:
   ```cpp
   g.channel(ch).accumulate(AccumOp::Mean, Window::from(ComponentId{99}));
   // ... only one component exists in g
   to_yaml(g);   // throws
   ```
   and separately
   ```cpp
   (void)g.add_field(a, FieldSpec{.name = "f", .region = RegionId{7}, ...});
   // ... no region ever added to g
   to_yaml(g);   // throws
   ```
   Both print `UNCAUGHT EXCEPTION escaped to_yaml(): vector` and exit
   non-zero — confirming an `std::out_of_range` (libc++'s `vector::at`
   message) escapes `to_yaml()`'s `Result<std::string>` contract exactly as
   M4/R-18 originally described.
2. For `f.region` specifically, also from a corrupted binary descriptor:
   `include/fuse/model/descriptor.hpp:307` reads `f.region =
   RegionId{r.pod<std::uint32_t>()}` with **no** check against `n_region`
   (the region count read at line 248) anywhere in the field-reading loop —
   contrast with `owner.value() >= n_comp`, checked two lines below at
   line ~313. This is the same ABI-narrow-waist reachability B1/R-04 was
   about, just for a different id. (`w.component` is comparably unchecked
   at `descriptor.hpp:114`, but I did not get a clean corrupted-buffer
   reproduction for it in the time available — patching the serialized
   bytes by pattern search is fragile and my attempt hit an unrelated
   truncation failure rather than proving the point either way. The
   raw-API-construction path already establishes the defect regardless.)

**No existing test catches this.** `test/model/validate_test.cpp:263,333`
covers `validate()`'s own refusal of exactly this out-of-range
`Window::from(ComponentId{99})` shape, and `test/model/yaml_io_test.cpp`'s
`pair_graph()` fixture always uses a valid, in-range window component — but
nothing calls `to_yaml()` on a graph that has an invalid window/region
reference *without* calling `validate()` first, which is the gap.

**Fix.** Add the same style of guard the R-18 fix already added for the
other four sites: before `emit_window`'s `FromComponent` branch, check
`w.component.valid() && w.component.value() < g.component_count()`
(`validate.hpp`'s own `window_component_ok()` is the right helper to reuse,
if `yaml_io.cpp` can see it, or duplicate its one-line check) and return
`Failure{"E-YAML-REF", ...}`; before the field-region line, check
`f.region.value() < g.region_count()`. Consider also adding the region
bound check to `descriptor.hpp`'s field-reading loop next to the existing
`owner.value() >= n_comp` check, for symmetry with R-19's choice to gate at
the descriptor boundary as well as at `to_yaml()`.

**Test to add.** Two cases mirroring
`YamlIo.EmittingAGraphWithADanglingChannelReferenceReturnsFailure`: one
building a graph via the raw API with `Window::from(ComponentId{99})` on a
channel's `accumulate` or `spacetime` window, one with a field whose
`.region` is out of range, each asserting `to_yaml()` returns `Failure`
with code `E-YAML-REF` rather than throwing.

### N-base-model-2 — MAJOR — R-19's non-positive scale refusal is not enforced on the YAML front end

`src/model/yaml_io.cpp:100` (`parse_units`), reached from `from_yaml()` via
the field-units branch (`src/model/yaml_io.cpp:601`).

R-19 fixed three places a non-positive `Units::scale` could enter or act on
a graph: `validate_field_semantics()`, the binary descriptor's
`deserialize()`, and `check_units()`'s automatic-scale-insertion branch. The
fix report's own reasoning for choosing `deserialize()` as a chokepoint was
that it is "the one input in this scope that crosses a real trust boundary."
That reasoning undercounts: this project's own documentation treats the
YAML text format as an equally real external boundary, not a convenience
wrapper over the binary format — `docs/dev/testing.md`'s "three-leg
round-trip" section exists specifically because "a front-end that loses a
member symmetrically passes its own round-trip" is a real failure class for
YAML, and `docs/dev/porting.md`'s naming map treats `soma_c.h`/YAML/Fortran/
Python as peer front ends to the C ABI. `parse_units()` parses `scale`
with no check at all:

```cpp
// src/model/yaml_io.cpp:100
if (n["scale"] && !parse_rational(n["scale"].as<std::string>(), u.scale)) return false;
```

`parse_rational` only checks that the string is syntactically a valid
rational (accepts `"-1000"` without complaint); nothing after this line, in
`parse_units` or in its caller, checks the sign or zero-ness of the result
before the parsed `Units` is stored into a `FieldSpec` and passed to
`add_field()`.

**Confirmed reachable.** Compiled and ran a program against the built
model/base libraries calling `from_yaml()` on:

```yaml
version: 1
meshes: [{name: m, source: "generated:box"}]
regions: [{name: r, mesh: m, entity_set: w, dimension: 2}]
components:
  - name: a
    ranks: {first: 0, count: 1}
    interval: 1
    fields: [{name: f, region: r, intent: Provides, units: {length: 1, scale: "-1000"}}]
channels: []
```

Output: `from_yaml ACCEPTED a negative unit scale silently` /
`field.units.scale = -1000` — no `Failure`, no diagnostic. The resulting
graph is exactly the shape `Semantics.NegativeScaleIsRejected` (R-19's own
new test) exists to reject, reached by a path that test does not cover.
Since `from_yaml()` never calls `validate()` or `validate_field_semantics()`
internally (confirmed by reading `from_yaml()`, `src/model/yaml_io.cpp:482`
onward, end to end — no such call), a graph loaded straight from a
YAML file with an authoring typo (a stray `-` on a scale factor) silently
carries a sign-flipping scale with no diagnostic anywhere up to the point a
caller separately remembers to call `validate()` — precisely the "no silent
fallback" (`I4`) and "the affine part... never negative" reasoning R-19's
own comments used to justify gating the binary path.

**No existing test catches this.** Grepped `test/model/yaml_io_test.cpp`
and `test/model/descriptor_test.cpp` for `scale.*-`/`NegativeScale` in a
`from_yaml`/YAML context: the only negative-scale tests added by R-19
(`Descriptor.RejectsNonPositiveUnitScale`) exercise the binary
`serialize()`/`deserialize()` pair, not `from_yaml()`.

**Fix.** Add the same non-positive check `descriptor.hpp`'s field loop now
has, in `parse_units()` (or immediately after its call in the field-parsing
branch), returning a `Failure{"E-YAML-UNITS-SCALE", ...}` — matching the
descriptor's `E-DESC-UNITS-SCALE` code family so the two front ends report
the same defect the same way.

**Test to add.** `YamlIo.RejectsNonPositiveUnitScale`, mirroring
`Descriptor.RejectsNonPositiveUnitScale`'s two cases (zero and negative)
through `from_yaml()` instead of `deserialize()`.

## Not found

Checked and confirmed still correct / not a defect:

- **The mutation and layering/invariant gates.** `tools/mutate.py
  --check-anchors` (scoped to the base/model toml) reports all anchors
  valid; `build-evidence/base-mutations.md` and `model-mutations.md` (run
  2026-09-19) record all 7 new mutations KILLED, not merely anchored.
- **`Rational::lcm_denominator`'s `Int128` route compiles and runs
  correctly** with this toolchain (AppleClang 21 / libc++): `std::gcd` on
  `__int128` operands compiled without error and `base.rational` passed,
  including the new overflow test — I checked this empirically rather than
  assuming the generic `std::gcd` template accepts a compiler-extension
  128-bit type on every implementation.
- **R-34's second citation (`iteration.hpp:~133`, "satisfied() exactly,
  including the zero-reference fallback")** is not a second instance of the
  dangling "header comment" bug the first review's B1-style two-line
  citation implied; it correctly refers to `satisfied()`, a real function a
  few lines above it in the same file. The fixer correctly left it alone.
- **The seven mutation anchors' file:line targets** all still match the
  current source (`tools/mutate.py --check-anchors`), so none of the fix's
  own follow-up commits (`93e3fe9`, `07f16ae`, etc., which touch other
  layers' mutation declarations) silently invalidated a base/model anchor.
- **`descriptor.hpp`'s other raw enum reads** (`Discretization`,
  `Conservation`, `Intent` at `descriptor.hpp:303-306`) remain unchecked by
  `deserialize()`, as the first review noted — but confirmed (again) safe,
  because every consumer (`to_cstring()` in `include/fuse/base/ids.hpp`) is
  a `switch` with a trailing `return "Unknown";`, not a raw array index.
  Not a regression and not newly in scope.
- **`error_policy()`'s new lock does not introduce a deadlock or an ABI
  break.** Every existing call site (`src/base/error.cpp:220`,
  `include/fuse/base/error.hpp:163`, both test files) binds the by-value
  return to a `const auto&`/`const ErrorPolicy&`, which is well-formed
  (reference to a temporary, lifetime-extended to the binding's scope) and
  compiles without a single call-site change, matching the fix report's own
  "source-compatible" claim.
- **The `model.descriptor_mpi` test's independence from this fix wave.**
  Grepped its source for every symbol the five base/model commits touch
  (`MeasureKind`, `AccelKind`, `FilterKind`, `RankSet`, `error_policy`,
  `Units::scale`, `lcm_denominator`) — none appear; it only calls
  `serialize()` on a fixed fixture and compares bytes across ranks. Compiles
  cleanly under `tier1` (MPI 5.0/MPICH found); running it aborts inside
  `MPI_Init_thread` in this sandbox, confirmed directly rather than assumed.

## Summary

**Verification: 6 CONFIRMED-FIXED (R-04, R-15, R-16, R-34, R-35, and the
R-17 rejection upheld), 2 PARTIAL (R-18, R-19).** Both partials are real,
verified gaps in an otherwise-correct fix: R-18's guard covers 4 of 6
unguarded dereference sites in `to_yaml()`, R-19's refusal covers 3 of 4
entry points for a non-positive unit scale. Neither partial reopens the
original finding's core claim — both original defects are fixed at the
sites the fixer addressed — but neither is fully closed either.

**Fresh review: 2 new findings, both MAJOR, 0 BLOCKER, 0 MINOR, 0
IMPROVEMENT, 0 DEVIATION.** Both are the same class of defect as the
verified-partial findings above (R-18, R-19 respectively) on paths the fix
pass did not reach — in effect, "finish the job" findings rather than new
categories of problem. Both are confirmed by compiling and running an
isolated reproduction against the actual built libraries, not by inspection
alone. No BLOCKER-class defect (raw memory-unsafety, as B1 was) was found in
this pass; both new findings throw a well-defined C++ exception rather than
exhibiting undefined behavior, which is why they are graded MAJOR rather
than BLOCKER, consistent with M4's original grading for the same escape
mechanism.

**Deviations from full coverage.** `tools/check-invariants.py` and
`tools/check-layering.py` were not independently re-run in this pass (time
budget); I relied on the fixer report's own claim of 0 violations, which is
consistent with everything else read here. The pre-fix TSan reproduction for
R-16 did not reproduce in two attempts and is reported as inconclusive, not
as counter-evidence (see Verification, R-16). One MPI-only test
(`model.descriptor_mpi`) could not be run, per the task's known constraints;
its compile, registration, rank list (2/3/4/5), and 120s timeout were all
checked directly instead.
