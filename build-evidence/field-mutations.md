# Mutation evidence — `field`

Run 2026-09-19 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `fieldspan-from-refuses-empty-partition` — FieldSpan::from() reverts to refusing every null FieldRef, including the legitimate count==0 partition (R-08) | `include/fuse/field/pipeline.hpp` | **KILLED** (test fails) | `field.pipeline` |
| `layout-well-formed-off-by-one-stride` — Layout::well_formed() reverts to the one-unit-too-conservative disjointness bound that rejects valid padded/strided layouts (R-29) | `include/fuse/field/field_ref.hpp` | **KILLED** (test fails) | `field.pipeline` |

**2 killed / 2 total.**
