<!-- Detailed layer review, 2026-09-19, HEAD bdf4346. Consolidated in ../../REVIEW.md; unified ids there are R-nn, this file's ids are BM-<id>. -->

# FUSE review: `base` and `model` layers

HEAD reviewed: `bdf4346`. Scope: `include/fuse/base/**`, `src/base/**`,
`include/fuse/model/**`, `src/model/**`, and their tests under `test/base`
and `test/model`. Read-only review; nothing under the repo was modified.

A note on the task framing this review arrived under: the delegating
message named a set of "Desired Final Outputs" (`convergence-matrix.md`,
`decision-register.md`, `m0-gate-recommendation.md`,
`unified-implementation-plan.md`, `repository-inventory.md`) that belong to
a different, larger merge-planning task, not to this one. The explicit,
detailed instructions in the same message name this file
(`review-base-model.md`) and a specific structure; this review follows
those instructions. No repository merge/convergence analysis is included
here — that is out of scope for a source-code review of two directories.

## 1. Coverage

Read line-by-line, in full:

- `include/fuse/base/`: `backtrace.hpp`, `bytes.hpp`, `config.hpp.in`,
  `error.hpp`, `ids.hpp`, `rational.hpp`, `result.hpp`, `time_level.hpp`,
  `units.hpp`, `version.hpp` (1,088 lines)
- `src/base/`: `backtrace.cpp`, `error.cpp`, `version.cpp` (371 lines)
- `include/fuse/model/`: `channel.hpp`, `compatibility.hpp`, `component.hpp`,
  `connect.hpp`, `coupling_graph.hpp`, `descriptor.hpp`, `field_spec.hpp`,
  `freeze_fixture.hpp`, `iteration.hpp`, `merge.hpp`, `multiscale.hpp`,
  `region.hpp`, `semantics.hpp`, `transform.hpp`, `validate.hpp`,
  `yaml_io.hpp` (2,924 lines)
- `src/model/`: `compatibility.cpp`, `multiscale.cpp`, `semantics.cpp`,
  `transform.cpp`, `yaml_io.cpp` (1,829 lines)
- Test files read in full or in the relevant sections: `rational_test.cpp`,
  `error_test.cpp`, `units_test.cpp`, `component_test.cpp`,
  `multiscale_test.cpp`, `descriptor_test.cpp`, `validate_test.cpp`; other
  test files in `test/base` and `test/model` were grepped for specific
  patterns (coverage of the findings below) rather than read end to end —
  see §4 for what those greps covered.
- Governing docs read in full before the code: `docs/dev/invariants.md`,
  `docs/dev/layering.md`, `docs/dev/conventions.md`, `docs/dev/testing.md`,
  `docs/dev/porting.md`, `docs/dev/rank-agreement.md`.

**Compiled.** No full build was attempted (the instructions permit
syntax-only single-file compiles; the project was not built or linked).
Two standalone reproductions were compiled and run outside the tree to
confirm two of the findings below at the language-semantics level:

- `std::lcm` on two `int64_t` values near `sqrt(INT64_MAX)` (mirroring
  `Rational::lcm_denominator`), compiled with `c++ -std=c++20 -O2`, run
  directly: confirmed the wrap to a negative `int64_t` (Finding M1).
- An out-of-range `enum class : uint8_t` cast used to index a 4-entry
  `const char* const[]` (mirroring the `kMeasureNames[...]` pattern in
  `yaml_io.cpp`), compiled with `c++ -std=c++20 -fsanitize=address,undefined`:
  UBSan reported `index 200 out of bounds for type 'const char *const[4]'`
  at runtime (Finding B1).

No other files were compiled; all other findings are confirmed by reading
the call sites and their callees/callers directly (per the task's method),
cross-checked against the array/table sizes and enum cardinalities involved.

**Deviation from full literal coverage:** `src/model/yaml_io.cpp` (789
lines) was read in full but not compiled — it depends on `yaml-cpp`, which
is not available in this sandbox and building it was out of scope per the
task instructions (no project build). The functional claims about it below
(§3, B1 and M4) rest on direct reading of the source plus the two isolated
reproductions above, not on running the actual `to_yaml`/`deserialize`
functions.

## 2. Findings table

| id | severity | file:line | title |
|----|----------|-----------|-------|
| B1 | BLOCKER | `include/fuse/model/descriptor.hpp:359,367,371`, `src/model/yaml_io.cpp:403,411,417` | Corrupted descriptor + `to_yaml()` = out-of-bounds array read (UB) |
| M1 | MAJOR | `include/fuse/base/rational.hpp:210-213` | `Rational::lcm_denominator` has the exact `int64_t` overflow its sibling function was fixed for |
| M2 | MAJOR | `src/base/error.cpp:53,67,69-72,199` | `error_policy()` reads `ErrorPolicy` without the lock `set_error_policy()` writes under — data race |
| M3 | MAJOR | `include/fuse/model/component.hpp:38`, `src/model/yaml_io.cpp:499-513` | `RankSet`'s default constructor silently places a component on rank 0, contradicting the class's own "empty = unplaced" contract; unchecked by `validate()` |
| M4 | MAJOR | `src/model/yaml_io.cpp:246,264,323,378,391` | `to_yaml()` dereferences unvalidated ids via `.at()`, throwing raw `std::out_of_range` instead of the `Result<std::string>` its signature promises |
| M5 | MAJOR | `include/fuse/base/units.hpp` (whole file), `src/model/semantics.cpp:359`, `src/model/compatibility.cpp` (`check_units`) | Negative `Units::scale` is never rejected; `check_units()` will auto-insert an unrequested, sign-flipping scale operator |
| M6 | MINOR | `include/fuse/model/iteration.hpp:80,127` | Dangling doc cross-reference: "the header comment" it points to does not exist |
| M7 | MINOR | `src/base/error.cpp:167,191` | Previous `std::terminate` handler captured in `g_previous` but never chained or otherwise used |

## 3. Findings in full

### B1 — BLOCKER — corrupted descriptor + `to_yaml()` produces an out-of-bounds array read

`include/fuse/model/descriptor.hpp:359,367,371`; `src/model/yaml_io.cpp:114-132,403,411,417`

`deserialize()` reads three enum fields straight off the wire with **no
range check**:

```cpp
m.kind = static_cast<MeasureKind>(r.pod<std::uint8_t>());              // :359
...
d.acceleration.kind = static_cast<AccelKind>(r.pod<std::uint8_t>());   // :367
...
d.acceleration.filter = static_cast<FilterKind>(r.pod<std::uint8_t>()); // :371
```

`MeasureKind` has 4 enumerators, `AccelKind` has 7, `FilterKind` has 4
(`include/fuse/model/iteration.hpp`). Any byte value outside those ranges
(5-255 for `MeasureKind`/`FilterKind`, 7-255 for `AccelKind`) round-trips
through `deserialize()` with **no diagnostic** — the function returns a
successful `CouplingGraph` carrying an out-of-range enum value in an
`IterationRegionDecl`.

That value is later used, unchecked, as a raw array index in
`src/model/yaml_io.cpp`'s `to_yaml()`:

```cpp
const char* const kMeasureNames[] = {"Relative", "Absolute",
                                     "AbsoluteOrRelative", "ResidualRelative"};   // 4 entries, :127-128
const char* const kAccelNames[]   = {"None", "Constant", "Aitken",
                                     "Secant", "Steffensen",
                                     "IQN_ILS", "IQN_IMVJ"};                     // 7 entries, :129-131
const char* const kFilterNames[]  = {"None", "QR1", "QR2", "QR3"};              // 4 entries, :132
...
<< kMeasureNames[static_cast<std::size_t>(m.kind)]                    // :403
...
<< kAccelNames[static_cast<std::size_t>(r.acceleration.kind)]          // :411
...
<< kFilterNames[static_cast<std::size_t>(r.acceleration.filter)]       // :417
```

Unlike `to_cstring(Discretization)`, `to_cstring(Conservation)`,
`to_cstring(Intent)`, `to_cstring(AccumOp)` and `to_cstring(TemporalBasis)`
in `include/fuse/base/ids.hpp` — every one of which is a `switch` with a
trailing `return "Unknown";` fallback, safe against an out-of-range value —
these three name tables are raw arrays indexed with no bounds check and no
fallback. `emit_window`'s `switch (w.kind)` (`yaml_io.cpp:175`) is a
`switch`, not an index, so `Window::Kind` is not exposed to this; but
`Window::Kind` is *also* deserialized unchecked (`descriptor.hpp`, the
`Reader::window()` method) and would silently no-op rather than crash if
out of range — a lesser but related gap noted for completeness.

**Confirmed** two ways: (1) direct comparison of the three array literal
sizes above against the enum declarations in `iteration.hpp`, and (2) an
isolated reproduction of the exact cast-then-index pattern, compiled with
`-fsanitize=undefined`, which reports at runtime:

```
runtime error: index 200 out of bounds for type 'const char *const[4]'
```

This is reachable through ordinary use of the public API: `deserialize()`
is the "narrow waist" the C ABI, Fortran, Python and YAML front ends all
funnel through (`descriptor.hpp`'s own header comment), so any bit flip,
truncation, or adversarial construction of a descriptor buffer that lands
one of these three byte positions outside its valid range passes
`deserialize()` silently and only manifests as UB later, at `to_yaml()` —
by which point the corruption is disconnected from its cause. Note the
asymmetry with the *merge* table a few lines above the `MeasureKind` read:
`deserialize()` explicitly checks `op > static_cast<std::uint8_t>(MergeOp::First)`
and refuses with `E-DESC-MERGE`, with a comment explaining exactly why
("refusing rather than defaulting ... would silently change the physics").
The same defensive posture was applied to one enum read from the same
untrusted buffer and not to the other three.

**Fix.** Add the same style of range check `deserialize()` already uses for
`MergeOp` to the three enum reads at `descriptor.hpp:359,367,371`
(e.g. `if (raw > static_cast<uint8_t>(FilterKind::QR3)) return Failure{"E-DESC-ACCEL", ...};`
and equivalent for `MeasureKind`/`AccelKind`), returning a `Failure` rather
than admitting the value. As defense in depth, also give
`kMeasureNames`/`kAccelNames`/`kFilterNames` bounds-checked accessors (or
convert them to `switch`-with-fallback functions matching the style already
used in `ids.hpp`) so a future caller of these tables cannot reintroduce
the same bug from a different unchecked source.

**Test to add.** A `descriptor_test.cpp` case that serializes a valid
graph, patches the byte offset of an iteration region's convergence-measure
kind (or acceleration kind/filter) to an out-of-range value, and asserts
`deserialize()` returns `Failure` — mirroring the existing
`Descriptor.InheritedVersionIsRefusedNotReinterpreted` test's technique of
patching bytes in a known-good buffer. No such test exists today for any
of these three enums (confirmed by grep: `MeasureKind`/`AccelKind`/
`FilterKind`/`out.of.range`/`corrupt`/`garbage`/`fuzz` do not appear in
`test/model/descriptor_test.cpp`). Note in passing: the *existing*
`E-DESC-MERGE` "unknown merge rule" refusal is also untested — grep finds
no `E-DESC-MERGE` in the test file — so it is "implemented, untested" by
`docs/dev/testing.md`'s own five-states rubric, not "verified"; the new
test should cover that refusal too while it is being written.

### M1 — MAJOR — `Rational::lcm_denominator` reintroduces the overflow its sibling fix removed

`include/fuse/base/rational.hpp:210-213`

```cpp
[[nodiscard]] static constexpr Rational lcm_denominator(const Rational& a,
                                                        const Rational& b) {
  return Rational{1, std::lcm(a.den_, b.den_)};
}
```

`std::lcm(a.den_, b.den_)` computes the lcm of two `int64_t` values directly
in `int64_t`. Per the standard, the result is undefined if the mathematical
lcm does not fit the common type; in practice (libstdc++/libc++) it wraps.
For two denominators near `sqrt(INT64_MAX)` (~3.037e9) that are coprime,
the true lcm (~9.22e18) is just under `INT64_MAX` but the *product* formed
internally by `std::lcm` overflows first. Compiled and run directly:

```
lcm = -9223372036709301665      // std::lcm(3037000493, 3037000507) in int64_t
```

That negative value flows into `Rational{1, <negative>}`, whose constructor
normalizes the sign onto the numerator (`reduce()`, `rational.hpp:196-207`)
and returns a small **negative** `Rational` — `lcm_denominator`, whose
entire contract is "the finest tick that divides both denominators" (a
positive quantity by construction), silently returns a negative one with no
diagnostic.

This is not a hypothetical: `gcd_rational` (a few lines below,
`rational.hpp:280-300`) computes the *identical* denominator-lcm quantity
internally, and was fixed for exactly this overflow — in `Int128`, with an
explicit `fits_i64` check and a `FUSE_REFUSE("E-BASE-RATIONAL-OVERFLOW", ...)`
on failure. `test/base/rational_test.cpp`'s
`GcdRationalThrowsOnUnrepresentableDenominatorLcm` test (lines ~176-186)
uses the *same two denominators* (`3'037'000'493`, `3'037'000'507`) to pin
that fix, and its own comment states the exact failure mode this finding
reproduces: *"a NEGATIVE rational returned as the gcd of two positive
ones, with no diagnostic. ClockSet::finest_interval() calls gcd_rational in
a fold, so that negative interval would have propagated straight into
§8.3 slab decomposition."* `lcm_denominator` computes the same arithmetic,
by the same unsafe route, and was not touched when `gcd_rational` was
fixed.

`lcm_denominator`'s own test,
`Rational.LcmDenominatorGivesFinestCommonTick`
(`test/base/rational_test.cpp:65-68`), only exercises denominators 3, 7, 4
and 2 — nowhere near the overflow domain — so it passes cleanly while the
function it tests carries the defect. This is currently latent rather than
live: a repository-wide grep finds `lcm_denominator` called only from that
one test (`include/fuse/model/coupling_graph.hpp` and callers in `schedule`/
`num` were not searched further, being out of this review's scope, but no
caller exists anywhere under `include/fuse/base` or `include/fuse/model`).
The function is public API on a base-layer numeric type documented for use
by "§8.3 slab decomposition" — the same downstream consumer named in the
`gcd_rational` comment — so it is reachable the moment that stage wires it
in, with no compiler or test warning that it needs the same care its
sibling already received.

**Fix.** Route `lcm_denominator` through the same `Int128`-safe path used
inside `try_gcd_rational` (`rational.hpp:255-269`): compute
`dg = std::gcd(ad, bd)`, `l = (ad / dg) * bd` in `detail::Int128`, and
`FUSE_REFUSE` (or, for symmetry with the rest of the class, add a
`try_lcm_denominator` returning `std::optional<Rational>`) if the result
does not fit `int64_t`.

**Test to add.** `Rational.LcmDenominatorThrowsOnUnrepresentableResult`,
mirroring `GcdRationalThrowsOnUnrepresentableDenominatorLcm` exactly (same
two denominators), asserting `fuse::Refusal` rather than a silently wrong
negative `Rational`.

### M2 — MAJOR — `error_policy()` is read without the lock its writer takes

`src/base/error.cpp:53,67,69-72,199`

```cpp
ErrorPolicy& policy_storage() { static ErrorPolicy p; return p; }        // :53
...
const ErrorPolicy& error_policy() noexcept { return policy_storage(); }  // :67 -- no lock
void set_error_policy(ErrorPolicy policy) noexcept {
  const std::lock_guard<std::mutex> lock(policy_mutex());
  policy_storage() = std::move(policy);                                  // :69-72 -- locked
}
```

`detail::raise<E>()` (`error.hpp:150-160`), which every `FUSE_THROW`/
`FUSE_REFUSE`/`FUSE_ASSERT` call goes through, calls `error_policy()`
(`error.cpp:199`) with no lock. `set_error_policy()` is documented
(`error.hpp:96-97`) as callable "programmatically" at any time, not only
during single-threaded startup. `ErrorPolicy` (`error.hpp:99-106`) holds
two `std::string` members (`terminal`, `debugger_ranks`); a thread calling
`error_policy()` while another thread is mid-way through the
copy-assignment in `set_error_policy()` can observe a `std::string` in an
inconsistent internal state — a genuine data race and undefined behavior
per the C++ memory model, not merely a stale-read hazard. `grep` across
`test/base/error_test.cpp` finds no thread-based test at all
(`thread`/`std::thread`/`concurrent` all absent), so this path has no
regression coverage in either direction.

This is exactly the class of defect the task's own conventions call out as
requiring a lock discipline check ("`concurrency-*`" failing checks in
`docs/dev/conventions.md`), and the project's stated history is that three
similar defects escaped Debug-only suites before
(`docs/dev/invariants.md`, `I6`); a lock taken on the write side only is
the textbook version of this bug and is invisible to any test that doesn't
force concurrent access.

**Fix.** Either take `policy_mutex()` in `error_policy()` too (returning a
copy, not a reference, since a `const&` cannot safely escape the lock), or
switch `policy_storage()` to an atomically-swapped
`std::shared_ptr<const ErrorPolicy>` (`std::atomic<std::shared_ptr<...>>`)
so a lock-free reader always sees a complete, consistent snapshot. The
former is the smaller change; the latter avoids taking a lock on every
throw, which matters on the hot error path this library cares about
keeping cheap.

**Test to add.** A test that spawns a thread calling `set_error_policy()`
in a loop while another thread calls `error_policy()` and throws via
`FUSE_REFUSE` in a loop, run under `-fsanitize=thread`. A plain functional
test cannot catch this class of bug; it needs TSan or an equivalent race
detector in the CI matrix (`docs/dev/testing.md`'s "Planned" CI legs list
an ASan+UBSan leg but no TSan leg).

### M3 — MAJOR — `RankSet`'s default constructor silently places a component on rank 0

`include/fuse/model/component.hpp:38-44`; `src/model/yaml_io.cpp:499-513`

```cpp
RankSet() : ranks_{0} {}                          // :38 -- NOT empty

/// The contiguous set `[first, first + count)`. `count <= 0` yields an empty
/// set, which is legal: a component may be declared and placed nowhere in a
/// particular run.
RankSet(int first, int count) {                   // :40-44
  ranks_.reserve(static_cast<std::size_t>(count > 0 ? count : 0));
  for (int i = 0; i < count; ++i) ranks_.push_back(first + i);
}
```

The class's own documentation, attached to the two-argument constructor,
states the design intent for "not placed": an empty rank set (`{0, 0}`
yields `ranks_ = {}`, confirmed by `RankSet.AnEmptyPlacementIsLegal` in
`test/model/component_test.cpp:92-96`). The niladic default constructor
does not follow that contract: `ranks_{0}` initializes a **one-element**
vector containing rank `0`, not an empty one. No test anywhere exercises
`RankSet()` (grep across the whole tree for `RankSet()` finds only its own
declaration), so this discrepancy has no regression coverage in either
direction, and nothing documents it as intentional (no mention in
`KNOWN_ISSUES.md`, `docs/dev/*.md`, or any commit-adjacent comment).

This reaches real code: `ComponentDecl::ranks` (`component.hpp:141`) has no
explicit initializer, so it default-constructs via `RankSet()`. In
`src/model/yaml_io.cpp`'s component parser:

```cpp
ComponentDecl c;                          // :499 -- c.ranks defaults to {0}
c.name = n["name"].as<std::string>();
if (n["ranks"]) {                         // :501
  ... c.ranks = RankSet{...};             // only reached if the YAML supplies "ranks"
}
```

A component whose YAML entry omits `ranks:` entirely is silently placed on
rank 0 — not left "unplaced," not refused, and not warned about. Two such
components (both omitting `ranks:`) would both land on rank 0 and compare
`same_as()` (`component.hpp:87-90`) as identically placed, which is a
meaningfully different outcome from "the user didn't say," with no
diagnostic anywhere. `include/fuse/model/validate.hpp` — the "authoring
gate" whose explicit job, per its own comments (e.g. the `f.components < 1`
check at `validate.hpp:87-93`), is to catch exactly this class of
underspecification before it reaches the scheduler — contains no mention of
`ranks`/`RankSet` at all (confirmed by grep over the whole file). This
mirrors a documented incident in the same file: the `components < 1` check
was added because nothing rejected it and the resulting corrupted quantity
propagated silently into `plan_buffers` as an unsigned-wraparound total;
the rank-default gap here is the same shape of problem (a silent,
plausible-looking default with no gate), just not yet hit in a
regression report.

**Fix.** Make `RankSet()`'s default consistent with the class's own stated
contract: `RankSet() = default;` should produce an empty `ranks_` (i.e.
change `RankSet() : ranks_{0} {}` to leave `ranks_` empty, matching
`RankSet(0, 0)`). Separately, `validate()` should refuse (or at minimum
warn on) a component whose `RankSet` is empty and which owns at least one
field with channels attached, since "declared but placed nowhere" is
exactly the kind of authoring gap `validate()` exists to report per its own
Global Constraints comment.

**Test to add.** `RankSet.DefaultConstructedIsEmpty` (asserting
`RankSet{}.empty()`), and a `Validate` test constructing a component via
`ComponentDecl{.name = "x"}` (no `.ranks`) with a field and channel attached,
asserting `validate()` reports something rather than declaring the graph
clean.

### M4 — MAJOR — `to_yaml()` throws raw `std::out_of_range` instead of returning `Failure`

`src/model/yaml_io.cpp:246` (function signature), `:264,323,378,391`
(the unguarded dereferences)

```cpp
Result<std::string> to_yaml(const CouplingGraph& g) {   // :246 -- promises Result, not throw
  ...
  << g.mesh(r.mesh).name                                 // :264
  ...
  return g.component(g.owner_of(f)).name + "." + ...      // :323
  ...
  << g.component(g.owner_of(FieldId{entry.first})).name   // :378  (merge table)
  ...
  for (const ChannelId c : r.channels) e << g.channel_at(c).name;  // :391
```

Every one of these calls a `CouplingGraph` accessor
(`mesh()`/`component()`/`channel_at()`, `coupling_graph.hpp:142-158`) that
is implemented as `container_.at(id.value())` — a plain `std::vector::at`,
which throws `std::out_of_range` (not a `fuse::Error`/`Refusal`) if the id
is out of range. `to_yaml()`'s own signature declares
`[[nodiscard]] Result<std::string>` — the type the project's own binding
rules require for a reportable failure ("`Result<T>` with `Failure`
payload for reportable failures; never throw+catch as internal control
flow inside the library") — and there is no `try`/`catch` anywhere in the
function body (confirmed by reading it end to end, `yaml_io.cpp:246-424`).
The only place `to_yaml()` itself returns a `Failure` is the emitter-error
check at the very end (`e.GetLastError()`, line ~424); nothing converts an
internal id lookup failure.

This differs materially from `serialize()`
(`include/fuse/model/descriptor.hpp:135-197`), which never dereferences a
*foreign* id (a `ChannelId` stored inside an `IterationRegionDecl`, or a
`FieldId` stored inside a merge-table entry) through an `.at()`-based
accessor — it writes the raw `uint32_t` id value and lets `deserialize()`
validate on the read side. `to_yaml()` needs *names*, not raw ids, for
human-readable YAML, so it does dereference those foreign ids, and nothing
guards the dereference.

**Reachable in normal use.** Nothing requires `validate()` to run before
`to_yaml()` — they are two independent free functions over the same graph,
and `add_iteration_region()`/`iterate()` (`coupling_graph.hpp:113-119`)
accept any `ChannelId` the caller passes with no bounds check at
construction time. A user (or a C ABI/Python/Fortran binding built on this
library) calling `g.iterate({ChannelId{99}})` on a graph with fewer than
100 channels, then `to_yaml(g)` for any reason — debugging, an error
report, a round trip before calling `validate()` — gets an uncaught
`std::out_of_range` instead of the `Failure` its signature promises. A
caller who only catches `fuse::Error` (the library's own exception
hierarchy, per `error.hpp`) will not catch this at all.

**Fix.** Either (a) wrap the body of `to_yaml()` in a
`try { ... } catch (const std::out_of_range& e) { return Failure{"E-YAML-BAD-REF", e.what()}; }`,
or, more in keeping with the rest of the file's style, (b) call `validate()`
internally at the top of `to_yaml()` and return its diagnostics as a
`Failure` before emitting anything if `has_errors()`. (b) is preferable
because it also gives a structured, per-problem diagnostic instead of a
single generic message from whichever `.at()` happened to fire first.

**Test to add.** `YamlIo.EmittingAGraphWithADanglingChannelReferenceReturnsFailure`
— build a graph via the raw API with an out-of-range `ChannelId` in an
iteration region (bypassing `validate()`), call `to_yaml()`, and assert it
returns a `Failure` rather than throwing.

### M5 — MAJOR — negative `Units::scale` is never rejected; `check_units()` will silently auto-insert a sign flip

`include/fuse/base/units.hpp` (whole file — no scale-sign check anywhere);
`src/model/semantics.cpp:359`; `src/model/compatibility.cpp` (`check_units`,
the automatic-scale-insertion branch)

`validate_field_semantics()` checks for a *zero* scale (the "annihilator"
case) but not a negative one:

```cpp
if (field.units.scale == Rational{0})                                // semantics.cpp:359
  diagnostics.push_back({SemanticDiagnosticCode::NonfiniteScale, prefix + ".units"});
```

Nothing in `include/fuse/base/units.hpp` constrains `Units::scale` to be
positive either — `Units{}`'s `scale` defaults to `Rational{1}` but any
`Rational` value, including a negative one, can be assigned to it, and
`operator*`/`operator/` on `Units` (`units.hpp:38-63`) only refuse an
*affine* (nonzero-offset) operand, never a negative scale. Physically, a
unit's scale relative to SI base units is a conversion factor and is never
negative (there is no unit whose "amount per SI base unit" is negative);
allowing `scale <= 0` other than exactly `0` is an unguarded gap, not a
deliberate design choice — no comment anywhere in `units.hpp`,
`semantics.cpp`, or `compatibility.cpp` addresses the negative case, and no
test does either (`grep` for `scale.*-1`/`Rational{-1}`/`negative` in
`test/base/units_test.cpp`, `test/model/semantics_test.cpp` and
`test/model/compatibility_test.cpp` returns nothing).

The consequence surfaces in `src/model/compatibility.cpp`'s `check_units()`,
in the pure-scale branch (offsets equal, scales differ):

```cpp
const std::optional<Rational> factor = Rational::try_div(source.scale, target.scale);
...
const bool explicit_request = evaluation.requested(OperatorKind::UnitScale);
if (!explicit_request && !request.allow_automatic_unit_scale) { ...refuse... }
OperatorStep step;
step.kind = OperatorKind::UnitScale;
step.explicitly_requested = explicit_request;
step.scale = *factor;                      // may be negative -- not checked
evaluation.emit(std::move(step));           // inserted WITHOUT an explicit request
```

If either side's declared `scale` is negative, `factor` is negative, and —
under the default policy (`ConnectPolicy::allow_automatic_unit_scale =
true`, `connect.hpp:34`) — this operator is inserted **automatically**,
with `explicitly_requested = false`. That is precisely the situation
`OperatorKind::SignFlip` exists to gate: `transform.cpp`'s
`operator_requires_explicit_request()` (line ~66) requires an explicit
request for every operator except `UnitScale`, on the stated rationale
(`connect.hpp:34-36`) that only "an exact, zero-offset unit scale" is a
"change of spelling, not of meaning" and may be inserted unasked. A
negative scale ratio is not a change of spelling; it silently negates every
value crossing the channel while reporting the connection as
`LosslessExplicit`/`LossyExplicit` with no warning distinguishing it from
an ordinary kilopascal-to-pascal rescale — the exact failure mode
`Evaluation::finish()`'s own comment (`compatibility.cpp`, "A fallback
that substitutes an operator changes the meaning of the coupled quantity
while reporting success") says must not exist.

**Fix.** Reject `scale <= 0` (not just `== 0`) wherever the zero check
exists today: promote `semantics.cpp:359`'s check to
`field.units.scale <= Rational{0}` (renaming or adding a diagnostic code
alongside `NonfiniteScale` if a distinct code is wanted for the sign case),
and add the same guard to `Units`'s own invariants (e.g. a
`FUSE_REFUSE("E-BASE-UNITS-SCALE-SIGN", ...)` in a constructor/setter, if
one exists at that layer, or at minimum in `descriptor.hpp`'s
`Reader::units()` so a corrupted descriptor cannot introduce one either).

**Test to add.** `Semantics.NegativeScaleIsRejected` (mirroring the
existing zero-scale test), and `Compatibility.NegativeScaleRatioIsRefusedNotAutoInserted`
asserting `check_connection()` reports `Ambiguous`/refuses rather than
silently emitting a `UnitScale` step when one side's scale is negative.

### M6 — MINOR — dangling documentation cross-reference in `iteration.hpp`

`include/fuse/model/iteration.hpp:80,127`

```cpp
/// See the header comment for the zero-reference, non-finite and AND/OR
/// policies.                                                    // :80
...
/// satisfied() exactly, including the zero-reference fallback.  // :127
```

No section in this file discusses "the zero-reference, non-finite and
AND/OR policies" (confirmed by grep for those exact phrases across the
file — the only two hits are the two dangling references themselves). The
behavior these comments refer to is real and correctly implemented
(`ConvergenceMeasure::satisfied()`, lines ~84-108), so this is a
documentation defect, not a logic one — but it is exactly the kind of
"claim written before verification" the project's own review history
flags: a reader following the pointer finds nothing where the comment says
something is explained.

**Fix.** Either write the referenced section (a short paragraph above
`ConvergenceMeasure` explaining the zero-reference-norm fallback to
absolute, the non-finite-means-not-converged rule, and the AND/OR
semantics of `AbsoluteOrRelative`), or change the comment to point at
`satisfied()`'s own inline comments, which already state the policy
piecemeal.

### M7 — MINOR — captured previous terminate handler is never used

`src/base/error.cpp:167,190-191`

```cpp
std::terminate_handler g_previous = nullptr;                          // :167
...
void install_terminate_handler() noexcept {
  std::call_once(g_installed, [] { g_previous = std::set_terminate(&fuse_terminate); });  // :191
}
```

`g_previous` is written once and never read anywhere in the file or the
rest of the tree (confirmed by grep). `fuse_terminate()` (lines ~171-186)
always handles the exception itself and calls `std::abort()`
unconditionally; it never chains to whatever handler was installed before
`fuse::install_terminate_handler()` ran. If an embedding application (or a
third-party library FUSE is linked into — MOAB, PETSc, Kokkos) had already
installed its own terminate handler for its own cleanup (flushing output,
calling `MPI_Abort` in a way its own runtime expects), that handler is
silently discarded rather than chained to, and the stored pointer gives a
false impression to a future maintainer that some chaining mechanism
exists.

**Fix.** Either call `g_previous()` at the end of `fuse_terminate()` for
exception types the `catch` clauses don't recognize as FUSE's own (or
always, after writing FUSE's own diagnostic, since the previous handler is
itself required by the standard to not return), or remove `g_previous`
entirely if the design intent is truly "FUSE's terminate handler is
final" — in which case say so in a comment, since the current code neither
does the former nor documents the latter.

## 4. Not found

Checked for and *not* found (i.e. these are not defects — recorded so the
reader knows what was verified rather than skipped):

- **`FUSE_HAVE_*`/`FUSE_ENABLE_*` read without `#include "fuse/base/config.hpp"`,
  or gated with `#ifdef` instead of `#if`.** Grepped every macro read in
  `include/fuse/base`, `include/fuse/model`, `src/base`, `src/model`. The
  only live macro reads are `FUSE_ENABLE_EXCEPTIONS` in `error.hpp` (with
  its own defensive `#ifndef ... #error` guard, and a correct `#if`) and a
  locally-`#define`d `FUSE_HAVE_POSIX_SPAWN_PATH` in `error.cpp` (not a
  `config.hpp` macro, so the rule doesn't apply). No `#ifdef` on a
  `#cmakedefine01` macro anywhere in scope.
- **Bare `throw`, bare `assert()`, `printf`, `std::cout`/`std::cerr`,
  unpinned `<sstream>` formatting.** Grepped across the whole scope; the
  only hits are in comments (documenting past bugs or explaining design),
  never live code. `fmt` is used consistently for formatted text; the one
  raw-`.at()`-throws-`std::out_of_range` pattern is a distinct issue,
  covered above as M4 (it is a standard-library throw reachable through
  the library's own accessor methods, not a `throw std::...` statement
  written in this code).
- **`E-ITER-NOCONVERGE`/`E-ITER-MAXITER`/`AccelConfig` finiteness
  validation being absent from `validate.hpp`.** These are real checks,
  correctly *not* present in this scope: they live in `src/ir/checkpoint.cpp`
  and `src/num/accel_relaxation.cpp` respectively, per the layering split
  documented in `docs/dev/layering.md` ("`ir` may not ask the runtime what
  a component can do" / "Iteration configuration is model, its algorithms
  are `num`"). Confirmed by grep for `E-ITER`/`E-ACCEL` across `src/` and
  `include/` — both codes exist and are exercised outside this review's
  scope, consistent with the documented layering rule rather than a gap in
  it.
- **`validate()` reporting only the first problem instead of every
  problem.** Read `validate.hpp` end to end: every check accumulates into
  one `Diagnostics` and the function never returns early except within a
  single channel's checks after an unresolvable `E-BAD-REF` (explicitly
  commented as necessary — "every later rule would dereference garbage" —
  and scoped to that one channel, not the whole graph). This matches the
  project's own Global Constraints contract.
- **`descriptor.hpp` serialize/deserialize field-order mismatches.**
  Traced every `Writer`/`Reader` call pair for meshes, regions, components
  (including the rank-sortedness check), fields, channels, iteration
  regions, and the merge table; every write has a matching read in the same
  order and type.
- **Signal-handler safety of the terminate path.** No `signal()`/
  `sigaction()`/`SIGSEGV`/`SIGABRT` handler registration exists anywhere in
  `include/fuse` or `src/base`/`src/model` (grepped); `install_terminate_handler()`
  installs a `std::terminate` handler, which runs on the normal C++ runtime
  stack, not inside a signal handler, so the classical "no malloc in a
  signal handler" constraint does not apply here as stated in the task
  brief. `fuse_terminate()` does allocate (`fmt::format`, `std::string`,
  `demangle()`'s `malloc`-backed `abi::__cxa_demangle`) while already
  handling a terminate condition; a further allocation failure (`bad_alloc`)
  inside a function marked `noexcept` would itself call `std::terminate`
  recursively. This is a real, if narrow, edge case (concurrent OOM during
  an unrelated terminate), noted here rather than raised as a numbered
  finding because it requires two independent failures to trigger and
  the task's "signal safety" framing does not describe a mechanism present
  in this codebase.
- **Multiscale/surrogate stubs "really refusing."** Verified: all five
  entry points in `multiscale.cpp` return `Failure` unconditionally except
  `substitute_surrogate` (which correctly reports contract defects first),
  and `test/model/multiscale_test.cpp` exercises every one of them and
  asserts the specific error code, satisfying `docs/dev/testing.md`'s
  "a deliberate refusal counts as verified only when a test makes it fire."
  No permissive stub found.
- **`connect.hpp`/`compatibility.cpp` stage ordering and worst-verdict
  selection.** Traced `check_connection()`'s eight-stage sequence and
  `Evaluation::refuse()`'s severity ratchet; all stages always run, and the
  verdict correctly tracks the worst outcome across every diagnostic
  raised, matching the header's documented contract.
- **`yaml_io.cpp`'s `from_yaml()` (the parse direction).** Unlike `to_yaml()`
  (M4), the parser validates every cross-reference it resolves (mesh,
  region, component, field, channel names) before use, returning
  `E-YAML-REF`/`E-YAML-PARSE`/`E-YAML-MERGE` failures rather than
  dereferencing an unresolved id. The `YAML::Load` call itself is wrapped
  in `try`/`catch` converting `YAML::Exception` to a `Failure`. No
  equivalent gap to M4 found on the read side.
- **Golden-file freeze/versioning discipline.** `freeze_fixture()` is
  shared between the test and (per its own doc comment) the regeneration
  tool; every superseded golden file (`waist-v1.bin`, `waist-v4.bin`) is
  kept and asserted to be *refused* (not reinterpreted) by the current
  reader; the current golden (`waist-v5.bin`) is compared byte-for-byte,
  not re-derived. No gap found here.

## 5. Suggested new tests / mutations

In addition to the "Test to add" items under each finding above:

1. **Mutation for `Rational::lcm_denominator` (ties to M1).** Per
   `docs/dev/invariants.md` `I9`, a public kernel needs a named, killed
   mutation. Candidate: revert the eventual `Int128` fix back to the direct
   `int64_t` `std::lcm` call and confirm
   `Rational.LcmDenominatorThrowsOnUnrepresentableResult` (new test, M1)
   kills it.
2. **Fuzz the descriptor reader.** `deserialize()` is the ABI narrow waist
   and the one place in this scope that reads directly off an untrusted
   byte buffer. A small structured fuzzer (flip one byte at a time across a
   valid `freeze_fixture()` buffer, assert `deserialize()` either accepts
   the identical graph back or returns a `Failure` — never UB, never a
   silently different graph) would have found B1 mechanically rather than
   by inspection, and is cheap to run given `deserialize()`'s total, ~200
   line body.
3. **TSan leg (ties to M2).** `docs/dev/testing.md`'s "Planned" CI list has
   an ASan+UBSan leg but no ThreadSanitizer leg. `error_policy()`/
   `set_error_policy()` is exactly the kind of defect (lock asymmetry) that
   only TSan catches reliably; worth adding before more shared mutable
   state accumulates in `base`.
4. **Property test for `Units::scale` sign (ties to M5).** Once the fix
   lands, a property-style test generating random `(source.scale,
   target.scale)` pairs with mixed signs and asserting `check_units()`
   never emits a step with `explicitly_requested == false` and a negative
   `step.scale` would give ongoing protection beyond the one fixed case.

## Summary

**1 BLOCKER, 5 MAJOR, 2 MINOR, 0 IMPROVEMENT, 0 DEVIATION.**

The blocker (B1) is a genuine memory-safety defect — an out-of-bounds array
read confirmed live under UBSan — reachable from a corrupted or
adversarially constructed descriptor byte buffer, the one input in this
scope that crosses a real trust boundary (the ABI narrow waist every
language binding funnels through). Three of the five majors (M1, M3, M5)
share a pattern: a defect that was already found and fixed once in a
sibling code path (`gcd_rational`'s Int128 fix, the `components < 1`
authoring-gate precedent, the zero-scale check) was not generalized to the
twin case that shares the same arithmetic or the same silent-default shape.
M4 is a `Result`-returning function that does not actually route its
internal failures through `Result`. M2 is the one finding of a different
kind — a lock-asymmetry data race in `error_policy()`/`set_error_policy()`,
invisible to any functional test and requiring a race detector to catch.
Both minors are documentation/dead-code hygiene, not behavioral. The `base`
and `model`
layers are otherwise carefully built to the project's own stated
invariants — `Rational`'s core arithmetic, `Units`' affine handling, the
`connect.hpp`/`compatibility.cpp` seven-stage checker, `validate()`'s
accumulate-everything discipline, the descriptor's version-gated
byte format, and the multiscale/surrogate refusal surface all held up
under line-by-line reading and, where feasible, direct execution.
