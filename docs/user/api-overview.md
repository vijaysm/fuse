# A map of the public API

`include/fuse/` has thirteen public namespace directories. There is no single umbrella header —
`#include` the ones you actually use. This page gives one entry per directory: what it is for, two
or three of its entry points by exact name, which page in this guide covers it in depth, and its
state in the sense [](../dev/testing.md#the-five-states) defines: *verified* (a test exercises it,
on this machine or on the host this project is developed against), *implemented* (it compiles and
runs but the claim rests on fewer legs than "verified" below names), or *designed only*.

## `fuse::base` — exact time, units, ids, errors

The vocabulary every other layer depends on: `fuse::Rational` (exact rational time arithmetic,
never a `double`), `fuse::Units` (SI dimension exponents), `fuse::Id<Tag>` (strong-typed indices),
`fuse::Result<T>`/`fuse::Diagnostics` (the accumulate-every-error convention), and the
backtrace/error machinery behind `FUSE_REFUSE`. No dedicated user-guide page — it underlies every
example on every other page, and [](glossary.md)'s "Time" section defines its vocabulary.
**Verified**: ported and tested at tier 0, Debug and Release, with `FUSE_ASSERT` checked under both
`NDEBUG` settings.

## `fuse::model` — the coupling specification

`fuse::CouplingGraph`: meshes, regions, components, fields, channels, merges and iteration regions,
built by `add_component`/`add_channel`/… and refined by `fuse::ChannelBuilder`
(`g.channel(id).conservative()…`). `fuse::validate()` checks the whole graph at once into a
`Diagnostics`. This is also where the canonical descriptor lives (`fuse::serialize`/`deserialize`,
`model/descriptor.hpp`) and the YAML front end (`fuse::to_yaml`/`from_yaml`). Covered by
[](fluent-api.md) (the authoring-chain idiom), [](placement.md) (`RankSet`, couplers),
[](combining-writers.md) (merges) and [](coupling-windows.md) (`CouplingWindowScope`). **Verified**:
the descriptor is frozen (`waist-v1`) with a three-leg round-trip and cross-rank byte-agreement
test.

## `fuse::ir` — the dataflow graph

`fuse::lower(graph)` compiles a `CouplingGraph` into an `fuse::OperatorDAG` — the twelve primitives
(`Sample`, `Accumulate`, `Remap`, `Transport`, `Merge`, `Relax`, …) connected by the values they
pass, plus `Source`/`Sink` boundary nodes. `fuse::dump_text(dag)` prints it in a fixed,
golden-file-comparable text format. Covered by [](quickstart.md)'s "Seeing what the compiler did".
**Verified**: the IR is exercised by every example and test that calls `lower()`, with golden-file
dumps pinning its exact shape for representative graphs.

## `fuse::schedule` — the compiled instruction stream

`fuse::schedule(graph, dag, clocks, region_sizes)` turns the dataflow graph into a flat
`Instruction` stream for one coupling cycle, after `fuse::prove_acyclic()`'s initialisation-time
deadlock proof and `fuse::ClockSet`'s common-period arithmetic. `fuse::CouplingWindowScope`/`DtPolicy`
(`schedule/window.hpp`) are the opt-in multirate path. Covered by [](quickstart.md) and
[](coupling-windows.md). **Verified**: the acyclicity proof and clock arithmetic are tested in
serial; the window path is additionally host-verified across ranks (`parallel.window_runner_mpi.
{2,3}`, KNOWN_ISSUES.md D19).

## `fuse::runtime` — executing the schedule

`fuse::Runner` executes a compiled schedule against a `ComponentSet` and a `FieldRegistry`;
`fuse::Component` is the interface a model implements (`register_fields()`, `advance()`); rank
agreement at a synchronisation point is `Epoch` (`runtime/epoch_schedule.hpp`). `fuse::WindowCoupler`
is the standalone driver for a time-window iteration outside the compiled path. Covered by
[](quickstart.md), [](coupling-windows.md) and [](c-abi.md) (`fuse_begin_step`/`fuse_end_step`
resume the same `Runner`). **Verified in serial**, and host-verified at more than one rank,
including a genuine multi-executable run of a three-component graph at 3 and 5 ranks
(`verification.brusselator.participants.mpmd.{3,5}`, KNOWN_ISSUES.md D25).

## `fuse::num` — the numeric kernels a step uses

`fuse::Accelerator` (Picard through Aitken, Steffensen, secant, IQN-ILS/IMVJ — `num/
accelerator.hpp`), `fuse::num::field_norm(NormKind::MeasureWeightedRms, …)` (the resolution- and
partition-independent convergence measure), `fuse::Waveform` (a field's trajectory over a time
window), and the portable per-step kernels in `num/space_kernels.hpp` (`accumulate`, `gather`,
`remap_apply`, `relax`, `dot`), checked on every Kokkos execution space the build enables against
the plain-loop oracle in `num/kernels.hpp`; and `fuse::num::solve()` (`num/newton_krylov.hpp`,
`num/line_search.hpp`), the Jacobian-free Newton-Krylov solver a region selects with
`AccelKind::Newton` — finite-difference Jacobian action, right-preconditioned GMRES with
Eisenstat–Walker forcing, three line searches, named convergence and divergence codes, and a
block-Jacobi preconditioner composed from each member's `fuse::LinearizationToken`. Covered by
[](convergence.md) (the Newton section), [](coupling-windows.md), [](field-pipeline.md), and the
developer page `dev/newton-krylov.md`. **Verified** (tier 0; the window form on one rank only, D31).

## `fuse::backend` — meshes and remap

`fuse::MeshView`/`fuse::MeshBackend` is the one interface every mesh backend implements:
`fuse::SerialMeshBackend` (dependency-free, structured), `fuse::native::NativeMeshBackend` (a
distributed unstructured mesh on Kokkos views, with its own halo exchange and RCB-driven
migration), and `fuse::MoabMeshBackend`. Every remap weight builder returns the same
`fuse::RemapWeights` triplet: `fuse::build_supermesh_weights()` (the always-available exact-simplex
path), the MOAB/TempestRemap tier, `fuse::build_mls_weights()` (point clouds), and
`fuse::build_consistent_interp_weights()`/its transpose (nodal interpolation). Covered by
[](geometry.md) (the mesh backends and the remap ladder), [](transfer-maps.md) (persisting
weights), [](point-cloud-remap.md) and [](interface-transfer.md). **Mixed**: the serial backend,
native mesh and supermesh path are verified in serial; the native mesh's halo additionally has a
tier-1 MPI test registered (`parallel.native_halo_mpi.{2,3}`) that this sandbox can only
compile-check (`docs/dev/native-halo.md`: "its outcome comes from a host run"); the MOAB backend is
verified on the host at every registered rank count (12/12 rows, KNOWN_ISSUES.md C1); MLS is
serial-only by design (rejects a detectably multi-rank backend outright).

## `fuse::linalg` — distributed linear algebra

`fuse::linalg::DistVector`/`DistSparseMatrix` (row-distributed, with an owned block and a ghost
region refreshed by an injected `Reducer`/`GhostExchanger`); `fuse::linalg::pcg()`/`gmres()`
(preconditioned Krylov solvers, iterate-for-iterate against Eigen's own implementations at one
rank); `fuse::linalg::SchurSlabSolver` (an exact distributed direct solver for a row-banded
periodic Laplacian); and the derivative gate (`linalg/derivative_gate.hpp`: `adjoint_gate`,
`taylor_remainder_order`, `matrix_free_agreement`), which any operator claiming a transpose or a
Jacobian is checked against before a Newton or adjoint claim cites it. [](glossary.md)'s "Linear
algebra" section and [](build.md) (the `FUSE_LINALG_BACKEND` choice, and D26's caveat that `builtin`
does not currently configure a full build) are the pages that touch it today; the developer pages
are `dev/linalg-distributed.md`, `dev/linalg-krylov.md`, `dev/schur-slab-solver.md` and
`dev/derivative-gate.md`. There is no dedicated user-guide page yet.
**Verified** in serial against Eigen, and host-run at 2–3 ranks (`parallel.linalg_mpi`,
`parallel.krylov_mpi`, `parallel.schur_slab_mpi.{2,3}`) — `docs/dev/schur-slab-solver.md` records a
host-found and fixed collective-safety defect from that run, which is what a real multi-rank run is
for. The same solvers also back a decomposed participant's diffusion solve in the Brusselator
example's `mpmd.5` row (KNOWN_ISSUES.md D25, D28).

## `fuse::partition` — mesh repartitioning

A Kokkos port of `rcb_moab`'s recursive coordinate bisection: `fuse::partition::rcb_partition()`
over any mesh's centroids (1D, 2D planar or spherical, 3D volumetric), and
`fuse::partition::repartition()`, which composes it with `fuse::native::migrate()` to repartition
a native mesh end to end. No dedicated user-guide page; see `docs/dev/partitioning.md` for the
port's own scope and determinism argument. **Verified**: a faithful, collective-for-collective port
of the source algorithm.

## `fuse::search` — spatial queries

A facade over ArborX that no public header outside `fuse::search` itself names:
`fuse::search::BoxIndex` (a host-space BVH over axis-aligned boxes, batched nearest/intersection
queries) and `fuse::search::query_points_distributed()` (the distributed counterpart, over
`ArborX::DistributedTree`). `fuse::backend` and `fuse::partition` both consume it without depending
on ArborX directly. Used by [](point-cloud-remap.md) (MLS's own nearest-neighbour search) and the
general nodal-interpolation path [](interface-transfer.md) describes. **Verified.**

## `fuse::transport` — moving a field between processor sets

`fuse::Transport`/`fuse::CommGraph` (which physical rank holds, or wants, this data, for two
processor sets that may be disjoint, identical or partially overlapping);
`fuse::transport::connect()` (the launch/rendezvous handshake for a multi-executable coupling, both
MPMD and true separate-launch accept/connect); `fuse::LowFiveTransport` (intercepted-HDF5
transport). Covered by [](placement.md), [](multiple-executables.md) and [](lowfive.md). **Mixed**:
`CommGraph` and MPMD `connect()` are host-verified (KNOWN_ISSUES.md D21, D25); accept/connect is
implemented but blocked on this host's own MPI (D23); `LowFiveTransport`'s single-process path is
verified, its distributed (cross-process) path compiles but is unmeasured (D14).

## `fuse::field` — field storage and the lazy view pipeline

`fuse::FieldRegistry` maps a `FieldId` to the component-owned storage bound at setup;
`fuse::FieldSpan`/`fuse::OwnedField` are the lazy `slice`/`component`/`owned` view chain over an
`fuse::Extent` ([](fluent-api.md), [](field-pipeline.md)). **Verified**: an allocation-count fence
pins the laziness claim, and a compile-fail test pins the dangling-view rule.

## `fuse::abi` — the two C ABIs

Generated from one entry list, `include/fuse/abi/entries.def`: `fuse_coupler_abi()` returns the
participant table (your code drives FUSE), `fuse_provider_abi()` the provider table (FUSE drives
your plugin) — plus `ffuse`/`ffuse_provider`, the matching Fortran bindings, generated from the
same list so none of the three can disagree. Covered in full by [](c-abi.md), [](fortran.md) and
[](providers.md). **Verified**: both tables are conformance-tested against a native in-process
reference, and the Brusselator example drives a real multi-executable coupling through them
(`u_main.c`/`v_main.f90`/`coupler_main.c`).
