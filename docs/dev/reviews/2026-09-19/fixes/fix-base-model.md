# Fix report: base/model layer (docs/dev/reviews/2026-09-19/base-model.md)

Branch `review-2026-09-19-fixes`, starting HEAD `ffea077`. Owned paths:
`include/fuse/base/**`, `src/base/**`, `test/base/**`, `include/fuse/model/**`,
`src/model/**`, `test/model/**`, `test/golden/**` (descriptor golden files only).

All eight findings were read against cited lines before any edit. Seven were
confirmed and fixed (R-04, R-15, R-16, R-18, R-19, R-34, R-35); one (R-17) was
investigated and rejected on git-history evidence that contradicted the
review's premise. Verification: full `cmake --build build-t0 -j4` (tier 0,
no MPI) succeeds for the whole repository, not just the owned layers; all 22
`base.*`/`model.*` ctest cases pass; `tools/check-invariants.py` and
`tools/check-layering.py` both report 0 violations; `tools/mutate.py
--check-anchors` reports all 7 new mutation anchors valid (2 pre-existing
stale anchors belong to other fixers' files, `src/ir/lowering.cpp` and
`include/fuse/backend/native/mesh.hpp`, outside these owned paths). The one
MPI-only test in scope, `model.descriptor_mpi`, was confirmed to compile
under the `tier1` preset and is listed under "Needs a host run".

## Commit units

```json
[
  {
    "ids": ["R-04", "R-15", "R-18", "R-19"],
    "files": [
      "include/fuse/base/rational.hpp",
      "include/fuse/model/descriptor.hpp",
      "include/fuse/model/compatibility.hpp",
      "include/fuse/model/semantics.hpp",
      "src/model/compatibility.cpp",
      "src/model/semantics.cpp",
      "src/model/yaml_io.cpp",
      "test/model/descriptor_test.cpp",
      "test/model/compatibility_test.cpp",
      "test/model/semantics_test.cpp",
      "test/model/yaml_io_test.cpp",
      "build-evidence/mutations/base-model.toml"
    ],
    "message": "Refuse corrupted enum bytes, dangling ids, and bad unit scales in the model layer\n\nBounds-check descriptor enum reads and to_yaml() id dereferences, fix lcm_denominator's int64 overflow, and refuse a non-positive unit scale at both the descriptor boundary and connection compatibility."
  },
  {
    "ids": ["R-17"],
    "files": [
      "include/fuse/model/component.hpp"
    ],
    "message": "Document why RankSet's rank-0 default is not the R-17 bug it looks like\n\nGit history and a pinning test show the niladic default has always meant rank 0, unlike the two-argument constructor's explicit empty-set case R-17 conflated it with."
  },
  {
    "ids": ["R-16", "R-35"],
    "files": [
      "include/fuse/base/error.hpp",
      "src/base/error.cpp",
      "test/base/error_test.cpp"
    ],
    "message": "Lock error_policy() reads and chain to a previously installed terminate handler\n\nReturn a lock-guarded copy from error_policy() instead of a racing reference, and call the terminate handler fuse's own replaced instead of discarding it."
  },
  {
    "ids": ["R-34"],
    "files": [
      "include/fuse/model/iteration.hpp"
    ],
    "message": "Fix ConvergenceMeasure::satisfied()'s dangling doc cross-reference\n\nPoint at the inline zero-reference, non-finite, and OR-only policy it actually implements instead of a nonexistent header section."
  }
]
```

## Fixed

**R-04 (BLOCKER)** `include/fuse/model/descriptor.hpp:359,367,371`, `src/model/yaml_io.cpp` —
`deserialize()` cast raw bytes to `MeasureKind`/`AccelKind`/`FilterKind` with
no range check, matching the `MergeOp` check already present a few lines
above them (the asymmetry is the tell: one enum in this same function was
already gated). An out-of-range value round-tripped cleanly through
`CouplingGraph` and only became undefined behavior later, at `to_yaml()`'s
`kMeasureNames[...]`/`kAccelNames[...]`/`kFilterNames[...]` unchecked array
indexing. Fixed by adding the same `> static_cast<uint8_t>(<last
enumerator>)` refusal (`E-DESC-ENUM`) used for `MergeOp`, and, as defense in
depth against any other caller of the raw `add_iteration_region()` API
bypassing `deserialize()` entirely, replaced the three unchecked array
indexes in `yaml_io.cpp` with a bounds-checked `enum_name()` helper.
Verified: `Descriptor.RejectsCorruptedIterationRegionEnums` builds a
known-good serialized buffer, computes each enum byte's offset from the end
of the buffer (the iteration region is the last thing `serialize()` writes),
patches each of the three bytes out of range in turn, and checks `deserialize()`
returns `E-DESC-ENUM` for each — plus that the unpatched buffer still round-trips,
so the offsets are pinned against the deserializer rather than against
themselves. Mutations `descriptor-accepts-any-convergence-measure-kind`,
`descriptor-accepts-any-acceleration-kind`, `descriptor-accepts-any-acceleration-filter`
in `build-evidence/mutations/base-model.toml`.

**R-15 (MAJOR)** `include/fuse/base/rational.hpp:210-213` — `lcm_denominator()`
computed `std::lcm(a.den_, b.den_)` directly in `int64_t`, the exact overflow
its sibling `try_gcd_rational()`'s own denominator lcm was already fixed for
(two ~3.037e9 denominators overflow a signed 64-bit product; the wrapped
result normalizes onto the numerator, silently returning a small negative
`Rational` for what must always be a positive finest-tick). Fixed by routing
through the same `detail::Int128` divide-then-multiply-then-check path
`try_gcd_rational()` already uses, refusing via `FUSE_REFUSE` rather than
returning a wrapped value. Verified: build succeeds; existing
`base.rational` suite (including its own overflow-refusal tests for the
sibling function) passes. Mutation `rational-lcm-denominator-overflows`
reverts to the direct `std::lcm` call to confirm the suite would catch a
regression back to it.

**R-16 (MAJOR)** `src/base/error.cpp:53,67,69-72,199` — `error_policy()`
returned `const ErrorPolicy&` into `policy_storage()` with no lock, while
`set_error_policy()` writes that same storage under `policy_mutex()` —
`ErrorPolicy` carries two `std::string` members, so a concurrent read and
write is a genuine data race, not just a stale-read hazard. Fixed by taking
`policy_mutex()` inside `error_policy()` and returning a copy (source-compatible:
every existing caller already binds to a const-ref or copies, confirmed by
grepping every call site). Verified: build succeeds; new test
`Error.ErrorPolicyIsSafeUnderConcurrentReadAndWrite` runs a writer thread
looping `set_error_policy()` against a reader thread looping
`FUSE_REFUSE`/`error_policy()` for 2000 iterations. This is explicitly a
"no crash under a plain build" test, not a race detector — the race itself
needs ThreadSanitizer to observe directly, which this repository's tier0
preset does not enable; noted under "Needs a host run".

**R-18 (MAJOR)** `src/model/yaml_io.cpp:246,264,323,378,391` — `to_yaml()`
promises `Result<std::string>` but dereferenced several foreign ids
(a region's mesh, a channel's source/target field, a merge entry's field, an
iteration region's channel list) through `CouplingGraph` accessors that are
plain `std::vector::at()` calls, while the raw graph-building API
(`add_channel`, `add_iteration_region`, `merge`, `add_region`) accepts any id
with no bounds check at construction time and nothing requires `validate()`
to run first. Fixed by adding a bounds check immediately before each of the
four dereference sites, returning `Failure{"E-YAML-REF", ...}` instead of
letting `std::out_of_range` escape uncaught. Calling the full `validate()`
internally was considered and rejected as too invasive: it would refuse
graphs that are safely dereferenceable but fail an unrelated policy check,
widening `to_yaml()`'s contract well past "can safely produce YAML for this
graph." Verified: new test
`YamlIo.EmittingAGraphWithADanglingChannelReferenceReturnsFailure` builds a
graph via the raw API with an out-of-range `ChannelId` inside an iteration
region, bypassing `validate()` entirely, and checks `to_yaml()` returns
`Failure` rather than throwing.

**R-19 (MAJOR)** `include/fuse/base/units.hpp` (whole file), `src/model/semantics.cpp:359`,
`src/model/compatibility.cpp` — a negative `Units::scale` was never rejected
anywhere, and `check_units()`'s pure-scale branch would auto-insert an
unrequested `UnitScale` operator for it exactly as it would for an ordinary
positive rescale, silently flipping the sign of every value crossing the
channel. `Units` has no constructor of its own to gate at (every call site
aggregate-initializes it), so there is no single choke point at that layer;
fixed at the three places the review's own suggested fix named: (1)
`semantics.cpp`'s existing zero-scale check (`SemanticDiagnosticCode::NonfiniteScale`)
gains a sibling `NegativeScale` case rather than being widened to `<= 0`
in place, keeping "annihilator" and "sign flip" distinguishable diagnostics;
(2) `descriptor.hpp`'s `deserialize()` field loop refuses a non-positive
scale (`E-DESC-UNITS-SCALE`) right after `Reader::units()`'s one call site,
the narrowest available choke point for the one input in this scope that
crosses a real trust boundary; (3) `compatibility.cpp`'s `check_units()`
gates a negative pure-scale factor behind an explicit `UnitScale` request,
exactly like the pre-existing affine-conversion gate just above it, and a new
`ConnectionReason::UnitScaleSignNotRequested` was added to `severity_of()`'s
`Ambiguous` list (the caller can fix it by requesting the operator, the same
class as `UnitScaleNotPermitted`/`AffineConversionNotRequested`; a first
pass forgot this and the negative-scale case fell through to the default
`Unsupported` severity, caught by the new test and corrected). Verified:
`Descriptor.RejectsNonPositiveUnitScale` (zero and negative, via
`deserialize()`), `Semantics.StructuralValidation`'s extended case
(negative scale via direct `FieldSemantics` construction), and
`Compatibility.UnitRules`'s extended case (negative scale refused when
unrequested, honored when explicitly requested, verdict is `Ambiguous` not
`Unsupported`) all pass. Mutations `descriptor-accepts-non-positive-unit-scale`,
`compatibility-auto-inserts-negative-unit-scale`, `semantics-accepts-negative-unit-scale`.

**R-34 (MINOR)** `include/fuse/model/iteration.hpp:80,127` — `satisfied()`'s
doc comment pointed at "the header comment" for the zero-reference,
non-finite, and AND/OR policies; no such section exists anywhere in this
file (confirmed by grepping the exact phrases). Fixed by rewriting the
comment to describe, inline, what the function actually implements: a
zero-or-negative `reference_norm` falls back to the absolute check, a
non-finite norm reports "not converged" rather than throwing, and
`AbsoluteOrRelative` is the OR of the two measures (there is no AND case
among the current `MeasureKind` values, so "AND/OR" in the old comment
overstated what exists). Documentation-only; verified by build and the
existing `model.transform`/iteration-adjacent suites passing unchanged.

**R-35 (MINOR)** `src/base/error.cpp:167,191` — `g_previous`, the terminate
handler installed before `fuse::install_terminate_handler()` ran, was
captured and never invoked, silently discarding whatever cleanup an
embedding application or a linked third-party library (MOAB, PETSc, Kokkos)
had installed its own handler to do. Fixed by calling `g_previous()` at the
end of `fuse_terminate()`, after FUSE's own diagnostic is already on stderr,
with `std::abort()` retained as the fallback for a null `g_previous` and for
a chained handler that violates `std::terminate`'s no-return contract.
Verified: new death test
`ErrorDeath.TerminateHandlerChainsToAPreviouslyInstalledHandler` installs a
marker handler via `std::set_terminate()` before calling
`fuse::install_terminate_handler()` inside the forked child (each
`EXPECT_DEATH` child starts from the parent's still-uninstalled `call_once`
state, since every earlier death test's own child never returns to the
parent), triggers an uncaught `fuse::Error`, and checks the death output
contains both FUSE's own report and the marker text — this would not appear
without the fix. The three pre-existing terminate-handler death tests
(`TerminateHandlerReportsAnUncaughtFuseError`,
`TerminateHandlerReportsANonFuseException`, `TerminateHandlerIsIdempotent`)
still pass: their regexes match FUSE's own message, which is still written
before the chain call, regardless of what the chained default handler prints
afterward.

## Rejected

**R-17 (MAJOR)** `include/fuse/model/component.hpp:38`, `src/model/yaml_io.cpp:499-513` —
the review claimed `RankSet`'s default constructor "silently places a
component on rank 0, contradicting the class's own 'empty = unplaced'
contract." Investigated by first implementing the suggested fix (empty
default + a `validate()` refusal for an empty-ranked component connected by
a channel), which immediately broke a pre-existing, unmodified test:
`ComponentDecl.DefaultsToUnitIntervalOnRankZero`
(`test/model/component_test.cpp:22-28`), whose name and body assert exactly
the opposite as the intended contract. `git log --follow -p` on
`test/model/component_test.cpp` shows this test was added in the initial
port commit `6632d0e`, before `RankSet` existed as a class at all — at that
point `RankSet` was `struct RankSet { int first = 0; int count = 1; };`
(plain default member initializers, also rank 0). `git show a71523e`
("Make a component's placement an explicit rank list") is the commit that
introduced the class-based `RankSet` with the `(int, int)` constructor whose
doc comment says "`count <= 0` yields an empty set, which is legal: a
component may be declared and placed nowhere in a particular run" — that
commit did NOT touch `component_test.cpp`'s
`DefaultsToUnitIntervalOnRankZero` test, meaning the niladic default's
rank-0 behavior was deliberately preserved unchanged across the refactor.
The "empty = unplaced" contract the review cites is documented on the
*explicit* two-argument constructor's `count <= 0` case; it was never a
claim about the niladic default, which has meant rank 0 since before this
class existed and is pinned by a test predating the reviewed commit range.
Separately, `AnEmptyPlacementIsLegal`'s own test comment
("a component may be declared and placed nowhere in a particular run... not
an error") describes exactly the shape the review's suggested `validate()`
check (a component with empty ranks connected by a channel) would have
flagged as an error — so that half of the suggested fix would have
contradicted a second, independently documented legal use case (a
multi-executable launch where the same graph description is shared but a
given launch configuration deliberately excludes some components). No code
change made; `include/fuse/model/component.hpp` gained only an explanatory
comment recording this investigation at the point a future reader would
otherwise repeat it.

## Needs another owner

None. All seven confirmed findings were fully fixable within the owned paths.

## Needs a host run

- **R-16**'s new test, `Error.ErrorPolicyIsSafeUnderConcurrentReadAndWrite`
  (`test/base/error_test.cpp`), passes on a plain build by construction (it
  only asserts "no crash"); confirming it would have caught the pre-fix race
  needs a ThreadSanitizer build, which is not one of this repository's
  presets and was not added as part of this fix (out of scope: adding a new
  preset is a build-system change, not a base/model source change). Suggest
  running `fuse_base_error_test` under `-fsanitize=thread` on a host as a
  one-off check, or asking the lead whether a `tier0-tsan` preset belongs in
  this review pass.
- **`model.descriptor_mpi`** (`test/model/descriptor_mpi_test.cpp`) was
  confirmed to compile cleanly under the `tier1` preset (`MPI_C`/`MPI_CXX`
  5.0 found, target `fuse_model_descriptor_mpi_test` links) but cannot run
  here: `MPI_Init_thread` dies in this sandbox for every tier1/tier2 test.
  This test does not touch any of the enums, `Units::scale`, `RankSet`, or
  `error_policy()` code paths this fix pass changed (confirmed by grep), so
  it is unaffected by R-04/R-15/R-16/R-18/R-19, but it should still be run
  at `mpiexec -n {2,3,4,5}` on a host as part of the normal tier1 gate.

## Deviation decisions

None of the eight findings in this pass were classified as DEVIATION
(code-vs-document ambiguity outside the BLOCKER/MAJOR/MINOR/IMPROVEMENT
severities); all eight were either a confirmed code defect (seven, fixed) or a
reviewer misreading of an intentional, tested contract (R-17, rejected). The
one place a documentation choice was made under a DEVIATION-like judgment
call was **R-34**: rather than writing the missing "header comment" section
the old cross-reference pointed at, the fix rewrote `satisfied()`'s own doc
comment to describe its policies inline, since the policies are three short
sentences fully explained by the code three lines below the comment — adding
a separate header-level section for them would create a second place to keep
in sync with the implementation for no reader benefit. Recorded here for the
lead to fold into the plan if a header-level "convergence policy" section is
wanted for a future doc pass.
