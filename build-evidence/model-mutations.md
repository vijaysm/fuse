# Mutation evidence — `model`

Run 2026-09-22 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `descriptor-drops-unit-scale` — the canonical bytes omit the affine unit part, so kPa hashes as Pa | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `descriptor-accepts-any-version` — the version gate waves through a descriptor from the inherited format | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `units-affine-multiply-allowed` — multiplying an affine unit silently drops the offset | `include/fuse/base/units.hpp` | **KILLED** (test fails) | `base.units` |
| `semantics-zero-scale-accepted` — a zero unit scale validates, turning every converted value into zero | `src/model/semantics.cpp` | **KILLED** (test fails) | `model.semantics` |
| `semantics-hash-ignores-order` — the semantic hash sums bytes commutatively, so a reordered record hashes alike | `src/model/semantics.cpp` | **KILLED** (test fails) | `model.semantics` |
| `validate-inserts-scale-silently` — an inferred unit scale is applied without being reported to the user | `include/fuse/model/validate.hpp` | **KILLED** (test fails) | `model.validate` |
| `connect-policy-ignored` — the site policy forbidding automatic unit scaling is ignored | `include/fuse/model/connect.hpp` | **KILLED** (test fails) | `model.validate` |
| `gll-order-collapses-to-vertex` — a high-order GLL coefficient is reported as a vertex value | `include/fuse/model/connect.hpp` | **KILLED** (test fails) | `model.connect` |
| `multiscale-overlap-permissive` — overlap accounting quietly succeeds when coverage happens to sum to one | `src/model/multiscale.cpp` | **KILLED** (test fails) | `model.multiscale` |
| `surrogate-ood-not-required` — a surrogate with no out-of-distribution detector passes review | `src/model/multiscale.cpp` | **KILLED** (test fails) | `model.multiscale` |
| `yaml-unit-fallback-unparseable` — a unit with no table symbol is emitted as a string the parser cannot read | `src/model/yaml_io.cpp` | **KILLED** (test fails) | `model.yaml_io` |
| `yaml-drops-unit-scale` — the YAML form omits a unit's exact scale, so kilopascal reads back as pascal | `src/model/yaml_io.cpp` | **KILLED** (test fails) | `model.yaml_io` |
| `descriptor-drops-the-merge-table` — the merge table is not serialized, so a graph crossing the waist loses how its writers combine | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `yaml-drops-field-declaration-order` — the text form omits the field index, so nesting fields under components renumbers every field id | `src/model/yaml_io.cpp` | **KILLED** (test fails) | `model.yaml_io` |
| `rankset-ignores-sharing` — overlaps() reports disjoint placements as sharing ranks, or the reverse | `include/fuse/model/component.hpp` | **KILLED** (test fails) | `model.component` |
| `descriptor-accepts-any-convergence-measure-kind` — deserialize() reverts to admitting any byte as a MeasureKind, the exact gap R-04 fixed (yaml_io.cpp's kMeasureNames index would go out of bounds later, disconnected from the cause) | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `descriptor-accepts-any-acceleration-kind` — deserialize() reverts to admitting any byte as an AccelKind (R-04) | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `descriptor-accepts-any-acceleration-filter` — deserialize() reverts to admitting any byte as a FilterKind (R-04) | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `descriptor-accepts-non-positive-unit-scale` — deserialize() stops rejecting a non-positive field unit scale (R-19), letting a corrupted descriptor introduce a zero (annihilator) or negative (sign-flipping) scale | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `compatibility-auto-inserts-negative-unit-scale` — check_units() stops gating a negative pure-scale conversion factor behind an explicit request (R-19), so it is auto-inserted like an ordinary positive rescale and silently flips the sign of every value crossing the channel | `src/model/compatibility.cpp` | **KILLED** (test fails) | `model.compatibility` |
| `semantics-accepts-negative-unit-scale` — validate_field_semantics() stops flagging a negative field unit scale (R-19), leaving only the zero/annihilator case checked | `src/model/semantics.cpp` | **KILLED** (test fails) | `model.semantics` |
| `yaml-io-emits-dangling-window-component` — emit_window() stops checking window_component_ok() before dereferencing a FromComponent window's component (N-base-model-1), reopening the std::out_of_range escape R-18 fixed for the other four to_yaml() sites but not this one | `src/model/yaml_io.cpp` | **KILLED** (test fails) | `model.yaml_io` |
| `yaml-io-emits-dangling-field-region` — to_yaml()'s field loop stops bounds-checking a field's RegionId before g.region(f.region).name (N-base-model-1), reopening the same std::out_of_range escape R-18 fixed for the region->mesh, channel->field, merge->field, and iteration-region->channel sites | `src/model/yaml_io.cpp` | **KILLED** (test fails) | `model.yaml_io` |
| `yaml-io-parses-non-positive-unit-scale` — from_yaml() stops refusing a non-positive parsed unit scale (N-base-model-2), reopening on the text front end the exact defect R-19 closed on the binary descriptor | `src/model/yaml_io.cpp` | **KILLED** (test fails) | `model.yaml_io` |
| `empty-placement-channel-check-never-fires` — The 2026-09-22 ruling: a channel out of or into a component placed on no rank is a modelling error, refused at validate() with E-PLACEMENT-EMPTY. Removing the check returns the diagnosis to MpiTransport::build_plan's E-TRANSPORT-RANKS two layers down -- a refusal that names a rank set when the mistake is in the graph, and only under MPI. | `include/fuse/model/validate.hpp` | **KILLED** (test fails) | `model.validate` |
| `empty-placement-check-fires-on-every-channel` — The other half of the same ruling: an empty placement stays legal for a component with NO channels, which is how a component is switched off for one run. A check that fired regardless of placement would refuse every ordinary graph, so this mutation must be killed by a test that validates a normally-placed pair. | `include/fuse/model/validate.hpp` | **KILLED** (test fails) | `model.validate` |

**26 killed / 26 total.**
