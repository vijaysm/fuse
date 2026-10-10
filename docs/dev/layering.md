# Layering

Dependencies point down only. A layer may use the layers below it and must not name anything above.

```text
abi         provider ABI · participant ABI · registry/adapter host · Fortran · Python · YAML · dynamic loading
runtime     Runner · field binding · transactions · checkpoint sets · Kokkos kernels
transport   rank matching · MPI router · crystal router · external processes
partition (*) RCB partitioner (Kokkos-view core + MeshView adapter) -- ported from rcb_moab
backend     mesh access · spatial and space-time transfer · solvers · persisted maps
search  (*) ArborX-backed spatial index (box/point queries); structurally below `backend` so backend can call it
field   (*) the lazy field pipeline: FieldRef/FieldSpan, subview staging over Kokkos views
num         waveforms · temporal bases · space-time projection · accelerators · convergence metric
schedule    clocks · slabs · expansion · acyclicity proof · epochs · ExecutionPlan
ir          operator DAG (SSA) · lowering registry · transform chain · golden IR
model       semantic model · coupling graph · canonical descriptor · compatibility · inference
linalg  (*) distributed containers (DistVector/DistSparseMatrix), PCG/GMRES(m), SchurSlabSolver,
            and the dense least-squares facade (`builtin` or `eigen` behind it) -- see docs/user/build.md
base        Rational · units · TimeLevel · SemanticVersion · ids · Result/Diagnostics · byte writer
```

`(*) linalg`, `field`, `search` and `partition` are real, separately-ranked layers in
`tools/check-layering.py`'s `LAYERS` list, easy to omit from a hand-drawn version of
this diagram (REVIEW.md R-39 / BT-F13 caught `linalg`/`field` missing once already).
None of the four sits strictly "under" its neighbour the way the rest of the column
does: `linalg` is a peer utility `num` depends on (nothing depends on `num` from
here); `field` is included only by `backend` and `runtime`, not by anything below
them; `search` sits directly below `backend` *by task requirement* (`backend`'s own
nodal-remap code calls into it), which puts its own `mesh_adapter.hpp` (the one
building element bounding boxes from a `fuse::MeshView`) in tension with that
position, since `fuse::MeshView` is defined one layer ABOVE `search` — an
`#include "fuse/backend/mesh_view.hpp"` there would be exactly the upward
dependency the gate exists to catch. Resolved by making THAT adapter
(`search`'s, not `partition`'s) a structural (duck-typed) template over
`MeshView`'s shape rather than an `#include` of it — [](search.md)'s own
"Layering" section has the full resolution. `fuse::partition`'s OWN
`mesh_adapter.hpp` has no such tension: it sits ABOVE `backend` and freely
`#include`s `fuse/backend/mesh_view.hpp`, an ordinary downward dependency;
`partition` is placed there because its core
(`rcb.hpp`/`median.hpp`/`coord_transform.hpp`) depends only on `base`, but its
mesh adapter needs `backend`, and nothing above `backend` needed `partition` at
the time it was placed. The gate script's own comment
(`tools/check-layering.py:16-29`, reproduced below) is the exact-rank source of
truth; this diagram is the reader-facing summary of it, not a replacement for it.
Renamed the former `interface` row to `abi` to match the gate's actual layer name —
no directory ever matched `interface`.

`physics-kit/` is a separate optional component. It **depends on FUSE and never the reverse** — that
direction is what keeps FUSE adoptable by a code that already has its own discretisation.

## Why the bottom half is dependency-light

`base` through `num` need only a C++20 compiler, Kokkos and a linear-algebra backend. That is not
minimalism for its own sake: it is what makes the fast CI leg fast, and what lets a code with neither
MOAB nor PETSc use the coupler. MPI enters at `transport`; everything heavier enters at `backend`,
behind one directory and one build option each.

## Enforcement

```{literalinclude} ../../tools/check-layering.py
:language: python
:start-after: "# --- rules"
:end-before: "# --- end rules"
```

The script runs as a CI gate on every push. A new layer or a new permitted dependency is a one-line
change there, reviewed like any other.

## A boundary the gate moved

`model` declares WHAT was asked for; `num` computes it. The inherited graph put its iteration
region's convergence criterion and acceleration choice in the numerics layer and reached up for them,
which inverts the dependency — the layering gate rejected the port on the first run. The configuration
types (`ConvergenceMeasure`, `ConvergenceCriterion`, `AccelKind`, `AccelConfig`, and the
`fuse::Converged` named constructors) now live in `fuse/model/iteration.hpp`; the algorithms (the
`Accelerator` implementations) stay in `fuse/num`, depending on the model rather than the reverse.

## Two splits the merge forced

**`ir` may not ask the runtime what a component can do.** Upstream, the checkpoint rule both *derived*
which values an iteration region must carry and *verified* that the components present could carry
them — the second half including `runtime/component.hpp` from an `ir` source. Deriving the set is a
pure function of the coupling graph and is provable before anything runs; asking whether the
components can honour it is a property of the run. The derivation stays in `src/ir/checkpoint.cpp`;
`verify_rollback_support` and `verify_iteration_region` moved to `src/runtime/checkpoint_support.cpp`.
The layering gate is what caught it.

**Iteration configuration is model, its algorithms are `num`.** `ConvergenceCriterion` and the
accelerator settings are things a *graph* declares, so they live in `include/fuse/model/iteration.hpp`
and the compiler can read them without linking a solver. The algorithms that consume them stay above.
