# Mutation evidence — `model`

Run 2026-09-18 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

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

**15 killed / 15 total.**
