# Developer guide

FUSE unifies two verified code bases: a coupling compiler and runtime, and a coupling spine
contributing the semantic model, compatibility inference, collective reconciliation and provider ABI.
The import baselines are pinned in `UPSTREAMS.lock` at the repository root. Everything here is
organized to be read in order, foundations first.

## Foundations — read these first

[](invariants.md) is short, and every page after it assumes those rules. Then [](layering.md) for
where code goes (the thirteen layers `tools/check-layering.py` enforces, bottom to top: `base`,
`linalg`, `model`, `ir`, `schedule`, `num`, `field`, `search`, `backend`, `partition`, `transport`,
`runtime`, `abi`), [](conventions.md) for how code is written (naming, formatting, commit messages,
the project's own vocabulary), and [](testing.md) for what "done" means here — the five states
every capability is in, the test harness, and the host-verified parallel results, since this
sandbox cannot start MPI at all and every multi-rank result is a host run.

## Correctness mechanisms

[](rank-agreement.md) is how the compiler keeps a failure on one rank from becoming a hang on
every other — the `Epoch` operator, what a failed rank still executes, and the lessons three
separate host runs have added to it since (a refusal decided from local data alone can still
strand a peer in a later collective; a window block's ownership is the consumer's, not the
exporter's). [](debugging.md) covers how FUSE fails and how you find out why: the error mechanism,
the on-error hook (attach a debugger to one rank under MPI), and the opt-in tracing environment
variables (`FUSE_SCHEDULE_DEBUG`, `FUSE_TRANSPORT_DEBUG`, `FUSE_CONSERVATION_DEBUG`,
`FUSE_BRUSS_DEBUG`) that have each closed a real defect.

## Porting and the C ABI

[](porting.md) is the procedure for moving code out of `upstream/` and into `fuse::` — the naming
map and the retirement schedule. [](abi-design.md) is why the C ABI exports one symbol per
direction rather than a prefixed family, and the rules for growing an argument struct without
breaking a caller built against an older one.

## Backends and numerics

[](operators.md) is the one interface (`LinearOperator`) every spatial transfer and solver
implements. [](partitioning.md) is the Kokkos-view RCB partitioner. [](native-mesh.md) is the
native, MOAB-free distributed mesh; [](native-halo.md) is its remap halo (the off-rank source
cells a partitioned read needs adopted locally before a conservative remap can run).
[](moab-intersection.md) is the MOAB-backed counterpart: the two cell-pair intersection
drivers MOAB itself offers (an advancing front, and a KD-tree search), why the front alone
drops cells on a genuinely partitioned sphere pair, and the coverage diagnostics behind
`FUSE_REMAP_HALO_TRACE`.
[](search.md) is the ArborX-backed spatial index backends query, and why it sits below `backend`
in the layering rather than above it. [](linalg-distributed.md) is the row-distributed dense
vector and sparse matrix (`DistVector`/`DistSparseMatrix`) every native parallel solver is built
on; [](linalg-krylov.md) is the native PCG and right-preconditioned GMRES(m) built on those;
[](schur-slab-solver.md) is the exact distributed direct solver for a row-band-decomposed periodic
Laplacian, and the one-row-band refusal that taught the rank-agreement lesson above.
[](derivative-gate.md) is the adjoint-identity, Taylor-remainder and assembled-vs-matrix-free
checks every derivative-bearing operator passes before a Newton claim may cite it; and
[](newton-krylov.md) is the residual/linearisation seam, the native Jacobian-free Newton-Krylov
solver with its SNES counterparts tabulated, and the measured convergence on dn_heat and the
Brusselator's coupling window.

## Runtime features

[](coupling-windows.md) covers the multi-clock iteration-window mechanism (schedule-time opt-in
plus runtime driver). [](nodal-fields.md) covers how a field's discretization decides its buffer
extent (`ContinuousGLL` order 1 is sized by vertex count, not cell count) and what lowest-rank
vertex ownership means for a nodal field in parallel. [](coupler-placement.md) covers the
separately-placed coupler: the `ComponentKind` representation decision, the rule that lowers
`source -> C -> target`, and every `component_count()` call site checked against it.
[](runner-device-dispatch.md) covers how `Runner::Impl::execute()` gets a
real Kokkos device (Cuda) path, and the optional cuSPARSE/cuSOLVER sparse-solve backend, including why the
device-dispatch scope is limited to templating `execute()`'s elementwise `OpKind` cases.
[](device-resident-arcs.md) is the follow-on design: arcs, `previous[]`,
history and the reduction scratch resident in the execution space's memory, host memory touched only
at named boundaries, the mesh/topology builders made Space-generic, and the GPU-aware transport path
for multi-rank device runs.

## Holding an external path to the bar

[](conformance.md) defines what "conformance" means for an external code coupling through the
participant ABI, and is the record of every Runner and window-path defect found holding an external
path to that bar — the Brusselator, as a separately-launched, multi-executable participant, is the
external code that has been measured against it.
