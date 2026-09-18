# Mutation evidence — `num`

Run 2026-09-18 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `norm-ignores-cell-measure` — the weighted norm drops the measure weight, so it is the plain norm again and depends on resolution | `src/num/residual_norm.cpp` | **KILLED** (test fails) | `num.residual_norm` |
| `norm-counts-unowned-entries` — unowned entries are included, so the norm depends on how the interface was partitioned | `src/num/residual_norm.cpp` | **KILLED** (test fails) | `num.residual_norm` |
| `norm-averages-the-averages` — combine averages the per-field RMS values instead of re-averaging over the union, which is wrong unless every field carries the same total measure | `src/num/residual_norm.cpp` | **KILLED** (test fails) | `num.residual_norm` |
| `norm-adds-unlike-units` — a temperature norm is added to a flux norm without complaint | `src/num/residual_norm.cpp` | **KILLED** (test fails) | `num.residual_norm` |
| `linalg-drops-rank-truncation` — a filtered column comes back with a small weight instead of exactly zero, so a caller cannot tell "dropped" from "barely used" | `src/linalg/least_squares.cpp` | **KILLED** (test fails) | `linalg.least_squares` |
| `gather-ignores-the-index-list` — the pack kernel copies element k instead of element index[k], so a message carries the wrong values while every size and count still agrees | `include/fuse/num/space_kernels.hpp` | **KILLED** (test fails) | `num.space_kernels` |
| `remap-apply-drops-each-row-first-entry` — the matvec starts each row one entry late, which loses mass without changing the sparsity pattern or any dimension | `include/fuse/num/space_kernels.hpp` | **KILLED** (test fails) | `num.space_kernels` |
| `relax-swaps-its-weights` — the relaxation weights are exchanged, so an under-relaxed sweep over-relaxes; convergence still happens, just to a different iterate count | `include/fuse/num/space_kernels.hpp` | **KILLED** (test fails) | `num.space_kernels` |
| `accumulate-forgets-the-first-sample` — the first substep accumulates into whatever the buffer held instead of overwriting it, so a period's accumulation inherits the previous period's residue | `include/fuse/num/space_kernels.hpp` | **KILLED** (test fails) | `num.space_kernels` |

**9 killed / 9 total.**
