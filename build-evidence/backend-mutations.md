# Mutation evidence — `backend`

Run 2026-09-17 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `mesh-measure-uses-the-spatial-determinant` — the m-measure is taken from a determinant instead of the Gram determinant, so a triangle in 3D or a curve in 3D measures zero | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `mesh-accepts-a-parent-outside-the-cell-set` — a simplex may name a parent cell that does not exist, so a conservation sum silently omits it | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `mesh-returns-nan-for-a-degenerate-simplex` — a negative rounded Gram determinant reaches sqrt, so a degenerate cell yields a NaN weight that propagates through a conservation sum | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `structured-hex-decomposition-drops-a-tet` — a box reports five tetrahedra instead of six, so its simplices no longer cover it and every weight built from them is short | `src/backend/serial_backend.cpp` | **KILLED** (test fails) | `backend.serial_mesh` |

**4 killed / 4 total.**
