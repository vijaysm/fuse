# Mutation evidence — `backend`

Run 2026-09-17 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `mesh-measure-uses-the-spatial-determinant` — the m-measure is taken from a determinant instead of the Gram determinant, so a triangle in 3D or a curve in 3D measures zero | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `mesh-accepts-a-parent-outside-the-cell-set` — a simplex may name a parent cell that does not exist, so a conservation sum silently omits it | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `mesh-returns-nan-for-a-degenerate-simplex` — a negative rounded Gram determinant reaches sqrt, so a degenerate cell yields a NaN weight that propagates through a conservation sum | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `structured-hex-decomposition-drops-a-tet` — a box reports five tetrahedra instead of six, so its simplices no longer cover it and every weight built from them is short | `src/backend/serial_backend.cpp` | **KILLED** (test fails) | `backend.serial_mesh` |
| `supermesh-planes-point-outward` — the inward test on a clipping simplex's face planes is skipped, so clipping returns the complement's measure -- a plausible number and the wrong one | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-ignores-triangle-winding` — a clockwise target triangle is passed to r2d as-is, which documents counterclockwise input | `src/backend/supermesh.cpp` | **SURVIVED** (suite still green) | `backend.supermesh` |
| `supermesh-ignores-tet-orientation` — a negatively oriented target tetrahedron is initialised as-is, so its moments come out negative | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-keeps-rounding-level-overlaps` — the negligible-overlap floor is removed, so a shared face yields a 1e-16 matrix entry and, in parallel, a real data dependency for no numerical content | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-projects-a-surface-onto-the-xy-plane` — a surface pair is clipped as its xy shadow instead of being refused, returning a plausible wrong measure | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-divides-by-the-source-measure` — the weight is normalised by the source cell's measure instead of the target's, which preserves mass but destroys constant preservation on a non-matching pair | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |

**9 killed / 10 total.**
