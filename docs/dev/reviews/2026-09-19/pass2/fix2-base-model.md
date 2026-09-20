# Second-pass fix report: base/model layer

Branch `review-2026-09-19-fixes`, starting HEAD `1565cdf`. Owned paths:
`include/fuse/base/**`, `src/base/**`, `test/base/**`, `include/fuse/model/**`,
`src/model/**`, `test/model/**`, `test/golden/*.bin`.

Input: `docs/dev/reviews/2026-09-19/pass2/re-base-model.md`. Its "## Verification"
table lists 6 CONFIRMED-FIXED, 1 REJECTION-UPHELD, and **2 PARTIAL (R-18,
R-19)** — nothing marked NOT-FIXED. Its "## New findings" lists
**N-base-model-1** and **N-base-model-2**, both MAJOR. All four rows (the two
PARTIALs plus the two new findings) describe the same two gaps: R-18/N-base-
model-1 is "finish to_yaml()'s dangling-reference guard", R-19/N-base-model-2
is "finish the non-positive-unit-scale refusal on the YAML front end". Fixing
each gap once closes both its PARTIAL row and its New-findings row.

Both were read against the cited lines and confirmed before any edit —
`src/model/yaml_io.cpp:193` (`emit_window`'s `FromComponent` branch) and
`:331` (field emission) were confirmed unguarded `.at()`-based dereferences
exactly as described; `:100` (`parse_units`) was confirmed to parse `scale`
with no sign or zero check, reachable through `from_yaml()`'s field-units
branch.

**Verification.** `cmake --preset tier0 -S <repo> -B build-fix2-base-model-t0
&& cmake --build build-fix2-base-model-t0 -j4` succeeds for the whole
repository (not just the owned layers — the working tree is shared with the
other five fixers running concurrently). All 22 `base.*`/`model.*` ctest
cases pass, including the 4 new gtest cases added by this pass.
`tools/check-invariants.py` and `tools/check-layering.py` both report 0
violations. `tools/mutate.py --check-anchors` reports 89/89 valid anchors,
including the 3 new ones this pass added (a full `--jobs`-driven kill run
was not attempted: the task rules restrict this pass to `--check-anchors`
only, and the tool itself additionally refuses a full run against an
uncommitted, concurrently-edited tree — "include/ or src/ has uncommitted
changes; the gate restores by copy-back" — confirming that restriction was
the right one to follow). Before trusting each new test, I disabled its
guard (temporarily, in the kernel — not committed) and reran: the field-
region test failed with the uncaught `std::out_of_range`/"vector" exception
the finding describes, confirming the test is load-bearing rather than
vacuously passing; the guard was then restored and the full 16-case
`model.yaml_io` suite re-verified green.

## Commit units

```json
[
  {
    "ids": ["R-18", "N-base-model-1"],
    "files": [
      "src/model/yaml_io.cpp",
      "test/model/yaml_io_test.cpp",
      "build-evidence/mutations/base-model.toml"
    ],
    "message": "Finish to_yaml()'s dangling-reference guard for windows and field regions\n\nemit_window() and the field-region emission site now refuse an out-of-range window component or field region (E-YAML-REF) instead of letting std::out_of_range escape, closing the two sites R-18's first-pass guard missed."
  },
  {
    "ids": ["R-19", "N-base-model-2"],
    "files": [
      "src/model/yaml_io.cpp",
      "test/model/yaml_io_test.cpp",
      "build-evidence/mutations/base-model.toml"
    ],
    "message": "Refuse a non-positive unit scale on the YAML front end too\n\nfrom_yaml()'s field-units parsing now rejects a zero or negative scale (E-YAML-UNITS-SCALE), the same rule R-19 gated at the binary descriptor boundary but left open for text-authored graphs."
  }
]
```

Both units touch `src/model/yaml_io.cpp` and are adjacent in the same
function family; they are kept as two units rather than one because they
answer two independently-readable findings and each stands on its own if the
lead wants to take one without the other. Each builds cleanly on top of the
prior one (verified: applying just the R-18/N-1 unit and rebuilding leaves
`YamlIo.RejectsNonPositiveUnitScale` failing to compile against the not-yet-
added scale check only because that test is added by the *second* unit —
there is no ordering dependency the other direction).

## Fixed

**N-base-model-1 (MAJOR, closes R-18 PARTIAL)**
`src/model/yaml_io.cpp:193` (`emit_window`'s `FromComponent` branch) and
`:331` (field emission's `region` key) each dereferenced an unvalidated id
(`w.component`, `f.region`) through an `.at()`-based `CouplingGraph`
accessor with nothing upstream guaranteeing the id is in range — the same
reachability R-18's first pass fixed for four sibling sites (region→mesh,
channel→field, merge→field, iteration-region→channel) but not these two.

Fixed by:
1. Changing `emit_window()`'s signature from `void` to
   `std::optional<Failure>`, and having it check
   `fuse::detail::window_component_ok(g, w)` — the exact helper
   `validate()` already uses for this same id (`include/fuse/model/
   validate.hpp`) — before touching `g.component(w.component)`, returning
   `Failure{"E-YAML-REF", ...}` on failure. Reusing the existing helper
   rather than duplicating the bounds check (as the review's own suggested
   fix allowed either way) means `to_yaml()` and `validate()` cannot drift
   apart on what "in range" means for a window's component. `yaml_io.cpp`
   gained `#include "fuse/model/validate.hpp"` for it; no circular include
   (`validate.hpp` only pulls in `connect.hpp`/`coupling_graph.hpp`, neither
   of which touches `yaml_io.hpp`).
2. Both call sites (`c.accumulate.window` and `c.spacetime.window`) now
   propagate the failure: `if (auto fail = emit_window(...)) return *fail;`.
3. A bounds check on `f.region` (`f.region.valid() && f.region.value() <
   g.region_count()`) immediately before the field-emission loop's
   `YAML::BeginMap`, matching the style and comment convention of the
   existing region→mesh check a few dozen lines above it.

Verified: two new tests,
`YamlIo.EmittingAGraphWithADanglingAccumulateWindowComponentReturnsFailure`
and `YamlIo.EmittingAGraphWithADanglingSpacetimeWindowComponentReturnsFailure`,
each build a two-field, one-channel graph via the raw API and then attach a
`Window::from(ComponentId{99})` to the channel's accumulate/spacetime window
respectively (no such component exists), asserting `to_yaml()` returns
`Failure` with code `E-YAML-REF`. A third,
`YamlIo.EmittingAGraphWithADanglingFieldRegionReturnsFailure`, builds a
component and a field whose `.region = RegionId{7}` with no region ever
added to the graph, asserting the same. All three pass; disabling the
field-region guard and rerunning reproduced the pre-fix
`std::out_of_range`/"vector" exception escaping the test body, confirming
each test is load-bearing. Mutations `yaml-io-emits-dangling-window-
component` and `yaml-io-emits-dangling-field-region` added to
`build-evidence/mutations/base-model.toml` (anchors valid per
`tools/mutate.py --check-anchors`; not run to a kill verdict, see
"Verification" above).

**N-base-model-2 (MAJOR, closes R-19 PARTIAL)**
`src/model/yaml_io.cpp:100` (`parse_units`, reached from `from_yaml()`'s
field-units branch) parsed a field's `scale` with no sign or zero check —
the exact defect R-19's first pass closed at the binary descriptor boundary
(`descriptor.hpp`'s `deserialize()`) and at `check_units()`/
`validate_field_semantics()`, left open on the text front end.

Fixed by adding, immediately after `parse_units()`'s existing "unknown
symbol" check at the `from_yaml()` call site, the same non-positive check
`descriptor.hpp`'s field-reading loop already has (`!(f.units.scale >
Rational{0})`), returning `Failure{"E-YAML-UNITS-SCALE", ...}` — a distinct
code from the pre-existing `E-YAML-UNITS` ("unknown unit symbol") so the two
failure modes stay distinguishable, matching the code-family convention
`E-DESC-UNITS-SCALE` already set on the descriptor side. Chose the call
site over widening `parse_units()`'s own `bool` contract, since the
call site is where the field name needed for the error message is in
scope, and the analogous descriptor-side check lives at its call site
(the field-reading loop) rather than inside a shared `Reader::units()`
helper for the same reason.

Verified: new test `YamlIo.RejectsNonPositiveUnitScale`, mirroring
`Descriptor.RejectsNonPositiveUnitScale`'s two cases (scale `"0"` and
`"-1000"`) through `from_yaml()` instead of `deserialize()`, asserts
`Failure` with code `E-YAML-UNITS-SCALE` for both. Passes. Mutation
`yaml-io-parses-non-positive-unit-scale` added to
`build-evidence/mutations/base-model.toml`.

## Rejected

None. Both PARTIAL rows and both new findings described the same two real,
confirmed gaps; nothing was contested.

## Needs another owner

None. Both fixes are entirely within `src/model/yaml_io.cpp` and
`test/model/yaml_io_test.cpp`, inside owned paths.

## Needs a host run

None for this pass's two findings — both are pure host-independent C++ unit
tests (`model.yaml_io`, tier 0, no MPI), already run and passing in this
sandbox. (The re-review's own "Needs a host run" items — `model.descriptor_mpi`
and the TSan-only R-16 check — belong to findings already closed in the
first fix pass and are outside this pass's two findings; not re-litigated
here.)

## Deviation decisions

None. Both findings were confirmed code defects with a fix directly implied
by the finding's own text and the existing sibling pattern (R-18's other
four guarded sites; R-19's descriptor-side check); no code-vs-document
ambiguity was involved.
