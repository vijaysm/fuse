# Mutation evidence — `linalg`

Run against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Outcome | Killed by |
|---|---|---|---|
| `least-squares-into-checks-shape` — least_squares_into starts enforcing check_shape's m>=k precondition, which the accelerator's small-interface call sites violate by design | `src/linalg/least_squares.cpp` | **KILLED** (test fails) | `linalg.least_squares` |

**1 killed / 1 total.**
