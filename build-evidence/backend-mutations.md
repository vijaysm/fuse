# Mutation evidence — `backend`

Run 2026-09-18 against build `build-fuse-s1`. Regenerate with `tools/mutate.py --write-log`; do not edit.

| Mutation | File | Verdict | Killed by |
|---|---|---|---|
| `mesh-measure-uses-the-spatial-determinant` — the m-measure is taken from a determinant instead of the Gram determinant, so a triangle in 3D or a curve in 3D measures zero | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `mesh-accepts-a-parent-outside-the-cell-set` — a simplex may name a parent cell that does not exist, so a conservation sum silently omits it | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `mesh-returns-nan-for-a-degenerate-simplex` — a negative rounded Gram determinant reaches sqrt, so a degenerate cell yields a NaN weight that propagates through a conservation sum | `include/fuse/backend/native/mesh.hpp` | **KILLED** (test fails) | `backend.native_mesh` |
| `structured-hex-decomposition-drops-a-tet` — a box reports five tetrahedra instead of six, so its simplices no longer cover it and every weight built from them is short | `src/backend/serial_backend.cpp` | **KILLED** (test fails) | `backend.serial_mesh` |
| `supermesh-planes-point-outward` — the inward test on a clipping simplex's face planes is skipped, so clipping returns the complement's measure -- a plausible number and the wrong one | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-ignores-triangle-winding` — a clockwise target triangle is passed to r2d as-is, which documents counterclockwise input | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-ignores-tet-orientation` — a negatively oriented target tetrahedron is initialised as-is, so its moments come out negative | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-keeps-rounding-level-overlaps` — the negligible-overlap floor is removed, so a shared face yields a 1e-16 matrix entry and, in parallel, a real data dependency for no numerical content | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-projects-a-surface-onto-the-xy-plane` — a surface pair is clipped as its xy shadow instead of being refused, returning a plausible wrong measure | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-divides-by-the-source-measure` — the weight is normalised by the source cell's measure instead of the target's, which preserves mass but destroys constant preservation on a non-matching pair | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-bounding-box-filter-drops-slivers` — the candidate filter rejects boxes that overlap by less than 1e-9, so thin slivers between coprime grids are never clipped and their mass is lost | `src/backend/supermesh.cpp` | **KILLED** (test fails) | `backend.supermesh` |
| `supermesh-coverage-computed-before-the-floor` — the audit is computed from the accumulated overlaps rather than from the entries actually kept, so mass discarded by the negligible-overlap floor is concealed | `src/backend/supermesh.cpp` | **EQUIVALENT** (Declared equivalent, with the bound that makes it so: the audit is
recomputed from the KEPT entries, and the negligible-overlap floor discards only
contributions below 1e-12 of a target cell's measure. The pre-floor and post-floor
audits therefore agree to within the floor itself, so no test can separate them
without a target cell carrying enough negligible entries for 1e-12 each to
accumulate into something visible -- which no realistic mesh pair produces. The
recomputation is kept because the audit should describe the matrix actually
returned, not the intermediate accumulation. IF THE FLOOR IS EVER RAISED this
stops being equivalent and must be re-examined.) | `backend.supermesh` |
| `composed-transpose-in-the-same-order` — (A B)^T applies A^T then B^T instead of B^T then A^T, so the composed operator's transpose is not its adjoint | `src/backend/linear_operator.cpp` | **KILLED** (test fails) | `backend.operator_conformance` |
| `composed-claims-are-inherited` — a composition claims conservation when EITHER factor does, so composing a conservative map with a non-conservative one reports conservation | `src/backend/linear_operator.cpp` | **KILLED** (test fails) | `backend.operator_conformance` |
| `compose-signature-ignores-order` — A o B and B o A hash the same, so a plan can reuse a persisted map built for the opposite order | `src/backend/linear_operator.cpp` | **KILLED** (test fails) | `backend.operator_conformance` |
| `weights-signature-covers-the-audit` — the signature hashes a measurement about the matrix rather than only what it computes, so a change in the builder's diagnostics invalidates every persisted map | `src/backend/weights_operator.cpp` | **KILLED** (test fails) | `backend.operator_conformance` |

**15 killed / 16 total.**
