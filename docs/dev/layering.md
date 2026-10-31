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
this diagram (`linalg`/`field` have been missed from it before).
None of the four sits strictly "under" its neighbour the way the rest of the column
does: `linalg` is a peer utility `num` depends on (nothing depends on `num` from
here); `field` is included only by `backend` and `runtime`, not by anything below
them; `search` sits directly below `backend` *by design* (`backend`'s own
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

Enforced by `tools/check-layering.py`'s `check_physics_kit_boundary` (added with the physics-kit skeleton):
nothing under `include/fuse/`, `src/`, `python/` or `examples/` — fuse proper — may include a
`fuse_physics/` header (physics-kit's own include root, outside `include/fuse/` entirely) or name
`fuse::physics_kit` in a `target_link_libraries()` call; `physics-kit/` itself is exempt from both,
since depending on FUSE is exactly what it is for. physics-kit is not given a rank in `LAYERS`
above: a rank taller than every fuse:: layer would still let a fuse:: layer be mis-ranked above it,
and `physics-kit/` is not under `include/fuse/` for the walk in `main()` to reach in the first
place, so the direction is checked as its own pass instead. Proven to fire against a probe header
(`#include "fuse_physics/version.hpp"` placed under `include/fuse/base/`) and a probe
`target_link_libraries(... fuse::physics_kit)` (placed under `src/`), both producing a violation
line and both removed once the proof was captured. `docs/dev/physics-kit.md` is this component's own page; naming
(`KARMA*`/`karma_`/`.C`) under `physics-kit/` is a separate gate, `tools/check-physics-kit-
naming.py`, because `tools/check-layering.py`'s own naming check (`check_legacy`) is scoped to the
four fuse-proper directories by a fixed tuple and physics-kit is deliberately not a fifth member of
it (this paragraph's own enforcement is the opposite direction from that check).

## Why the bottom half is dependency-light

`base` through `num` need only a C++20 compiler, Kokkos and a linear-algebra backend. That is not
minimalism for its own sake: it is what makes the fast CI leg fast, and what lets a code with neither
MOAB nor PETSc use the coupler. MPI enters at `transport`; everything heavier enters at `backend`,
behind one directory and one build option each.

## MPI outside `transport`

"MPI enters at `transport`," above, names where the ROUTING of field values between ranks lives; it
has never meant that no `.cpp` below `transport` may call an MPI function at all. Two translation
units do, both deliberately, both PRIVATE to the layer whose own invariant the call protects, and
neither reachable from a public header (both run through one helper,
`src/field/collective_agreement.cpp`, the translation unit of `field` that calls MPI;
`src/field/transaction.cpp` does not name it): `src/abi/participant.cpp`'s `reconcile_plugin_agreement()`
(an `MPI_Allreduce(MPI_MINLOC)` agreeing a plugin-mount verdict across the base communicator, `abi`
sitting at the TOP of the stack) and `src/field/transaction.cpp`'s `FieldTransaction::
commit_collective()` (an `MPI_Allreduce(MPI_MAXLOC)` agreeing a staged-commit verdict, `field`
sitting BELOW `transport`). Both exist for the same reason: a
collective AGREEMENT about whether an operation may proceed is not the same thing as MOVING field
values between ranks, and it belongs in the layer that owns the invariant being agreed on -- the
plugin-mount invariant is `abi`'s own, and the staged-commit invariant ("no caller-visible value
changes until every rank has agreed," `include/fuse/field/transaction.hpp`'s own file comment) is
`field`'s own. Moving that one collective up to `transport` would not remove it from the layer that
needs it; it would make `field` (or `abi`) depend upward on `transport` to get back a capability
`transport` itself has no stake in, which is the inversion `tools/check-layering.py` exists to catch
in the direction that matters -- a layer's PUBLIC header naming a layer above it. Neither of these two
`.cpp` files' own public header names `mpi.h` (`tools/check-layering.py`'s
`ALLOWED_PUBLIC_THIRD_PARTY` forbids it for every layer); each links `MPI::MPI_CXX` conditionally,
under `FUSE_ENABLE_MPI`, exactly the way every layer AT OR ABOVE `transport` already does. Not
restructured: this paragraph records the exception, it does not propose closing it.

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
