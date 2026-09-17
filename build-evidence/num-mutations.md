# Mutation evidence — `num`

Run 2026-09-17 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `norm-ignores-cell-measure` — the weighted norm drops the measure weight, so it is the plain norm again and depends on resolution | `src/num/residual_norm.cpp` | **KILLED** (test fails) | `num.residual_norm` |
| `norm-counts-unowned-entries` — unowned entries are included, so the norm depends on how the interface was partitioned | `src/num/residual_norm.cpp` | **KILLED** (test fails) | `num.residual_norm` |
| `norm-averages-the-averages` — combine averages the per-field RMS values instead of re-averaging over the union, which is wrong unless every field carries the same total measure | `src/num/residual_norm.cpp` | **KILLED** (test fails) | `num.residual_norm` |
| `norm-adds-unlike-units` — a temperature norm is added to a flux norm without complaint | `src/num/residual_norm.cpp` | **KILLED** (test fails) | `num.residual_norm` |
| `linalg-drops-rank-truncation` — the facade reports full rank, so a filtered column comes back with a weight instead of exactly zero | `src/linalg/least_squares.cpp` | **KILLED** (test fails) | `linalg.least_squares` |

**5 killed / 5 total.**
