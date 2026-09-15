# Mutation evidence — `model`

Run 2026-09-15 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `descriptor-drops-unit-scale` — the canonical bytes omit the affine unit part, so kPa hashes as Pa | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `descriptor-accepts-any-version` — the version gate waves through a descriptor from the inherited format | `include/fuse/model/descriptor.hpp` | **KILLED** (test fails) | `model.descriptor` |
| `units-affine-multiply-allowed` — multiplying an affine unit silently drops the offset | `include/fuse/base/units.hpp` | **KILLED** (test fails) | `base.units` |
| `semantics-zero-scale-accepted` — a zero unit scale validates, turning every converted value into zero | `src/model/semantics.cpp` | **KILLED** (test fails) | `model.semantics` |
| `semantics-hash-ignores-order` — the semantic hash sums bytes commutatively, so a reordered record hashes alike | `src/model/semantics.cpp` | **KILLED** (test fails) | `model.semantics` |

**5 killed / 5 total.**
