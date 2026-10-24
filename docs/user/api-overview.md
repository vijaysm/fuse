# A map of the public API

`include/fuse/` has fifteen public namespace directories and no single umbrella header: `#include`
the ones you use. This page helps you find the right header for a task, then gives one entry per
directory: what it is for, its main entry points by exact name, a typical call, and which page in
this guide covers it. If you are new, start with [](getting-started.md) and [](concepts.md).

## Find what you need

| I want to ... | Namespace / headers | Page |
|---|---|---|
| Describe a coupling (components, fields, channels) | `fuse::CouplingGraph`, `fuse::ChannelBuilder`: `fuse/model/coupling_graph.hpp`, `fuse/model/channel.hpp`, `fuse/model/validate.hpp` | [](graph-programming.md), [](fluent-api.md), [](quickstart.md) |
| Write the coupling as a file instead | `fuse::from_yaml()`/`to_yaml()` (`fuse/model/yaml_io.hpp`) | [](yaml.md) |
| Run it | `fuse::lower()` (`fuse/ir/lowering.hpp`), `fuse::schedule()` (`fuse/schedule/plan.hpp`), `fuse::Runner` (`fuse/runtime/runner.hpp`) | [](runtime.md), [](quickstart.md), [](drivers.md) |
| Write a component | `fuse::Component`, `fuse::ImplicitComponent` (`fuse/runtime/component.hpp`), `fuse::FieldRegistry` (`fuse/field/registry.hpp`) | [](component-interface.md) |
| Remap a field between non-matching meshes | `fuse::build_supermesh_weights()` (`fuse/backend/supermesh.hpp`), `fuse::build_mls_weights()` (`fuse/backend/mls_weights.hpp`), `fuse::build_consistent_interp_weights()` (`fuse/backend/consistent_interp.hpp`) | [](meshes.md), [](geometry.md), [](interface-transfer.md), [](point-cloud-remap.md), [](transfer-maps.md) |
| Couple an existing C or Fortran code | `fuse_coupler_abi()` (`fuse/abi/participant.h`), `fuse_provider_abi()` (`fuse/abi/provider.h`), the `ffuse` Fortran module | [](mixing-languages.md), [](c-abi.md), [](fortran.md), [](providers.md) |
| Run across ranks or as separate executables | `fuse::RankSet` and `fuse::Transport` (`fuse/model/component.hpp`, `fuse/transport/transport.hpp`), `fuse::transport::connect()` (`fuse/transport/connect.hpp`) | [](placement.md), [](multiple-executables.md), [](lowfive.md) |
| Write and restore a restart point | `fuse::write_restart_point()` / `fuse::restore_restart_point()` (`fuse/runtime/restart.hpp`), `fuse::restart_io::create_restart_point()` (`fuse/restart_io/hdf5_restart_io.hpp`) | [](restart.md) |
| Iterate a group of components to convergence | `CouplingGraph::iterate()` and `fuse::Converged` (`fuse/model/iteration.hpp`), `fuse::Accelerator` (`fuse/num/accelerator.hpp`) | [](convergence.md) |
| Use Newton instead of a fixed-point accelerator | `AccelKind::Newton` (`fuse/model/iteration.hpp`), `fuse::num::solve()` (`fuse/num/newton_krylov.hpp`), `fuse::petsc::solve()` (`fuse/petsc/solve.hpp`) | [](convergence.md), [](linear-algebra.md) |
| Use different time steps in different components | `fuse::CouplingWindowScope`, `DtPolicy` (`fuse/model/iteration.hpp`), `fuse/schedule/window.hpp` | [](coupling-windows.md) |
| Merge several writers into one field | `fuse/model/merge.hpp` | [](combining-writers.md) |
| Declare a micro-macro coupling or a surrogate | `fuse/model/multiscale.hpp` | [](multiscale.md), [](surrogates.md) |
| Control execution space, threads and environment variables | `RunnerConfig::execution_space`, `FUSE_EXECUTION_SPACE` | [](environment.md) |
| Drive a coupling from Python | `pfuse` | [](python.md) |

For deeper material see [](power-users.md); for error codes and symptoms see
[](troubleshooting.md).

## `fuse::base`: exact time, units, ids, errors

The vocabulary every other layer depends on: `fuse::Rational` (exact rational time arithmetic,
never a `double`), `fuse::Units` (SI dimension exponents, with constants in `fuse::units`),
`fuse::Id<Tag>` (strong-typed indices), `fuse::Result<T>`/`fuse::Failure`/`fuse::Diagnostics` (the
accumulate-every-error convention), and the backtrace/error machinery behind `FUSE_REFUSE`.

```cpp
const fuse::Rational dt{1, 3};                 // exactly 1/3
const fuse::Rational period = dt * fuse::Rational{3};
const fuse::Units u = fuse::units::kHeatFluxDensity;
fuse::Result<int> r = 42;
if (r.ok()) { use(r.value()); } else { report(r.error().code, r.error().message); }
```

No dedicated page; [](glossary.md)'s "Time" and "Errors" sections define its vocabulary.

Maturity: covered by the automated test suite in Debug and Release builds.

## `fuse::model`: the coupling specification

`fuse::CouplingGraph`: meshes, regions, components, fields, channels, merges and iteration regions,
built by `add_component`/`add_channel`/... and refined by `fuse::ChannelBuilder`
(`g.channel(id).conservative()...`) and `fuse::IterationBuilder` (`g.iterate({...}).until(...)`).
`fuse::validate()` checks the whole graph at once into a `Diagnostics`. The canonical descriptor
(`fuse::serialize`/`deserialize`, `model/descriptor.hpp`) and the YAML front end
(`fuse::to_yaml`/`from_yaml`, `model/yaml_io.hpp`) also live here.

```cpp
fuse::CouplingGraph g;
const auto m = g.add_mesh({.name = "m", .source = "generated:line:16"});
const auto r = g.add_region({.name = "wall", .mesh = m, .dimension = 1});
const auto a = g.add_component({.name = "a", .ranks = fuse::RankSet{0, 1}});
const auto b = g.add_component({.name = "b", .ranks = fuse::RankSet{0, 1}});
const auto out = g.add_field(a, {.name = "q", .region = r, .intent = fuse::Intent::Provides});
const auto in  = g.add_field(b, {.name = "q_in", .region = r, .intent = fuse::Intent::Requires});
g.channel(g.add_channel("flux", out, in)).conservative();
const fuse::Diagnostics d = fuse::validate(g);
fuse::Result<std::string> yaml = fuse::to_yaml(g);     // or fuse::serialize(g) for the binary descriptor
```

Covered by [](graph-programming.md) (every declaration and builder), [](fluent-api.md) (the
authoring-chain idiom), [](yaml.md) (the text form), [](placement.md) (`RankSet`, couplers),
[](combining-writers.md) (merges), [](convergence.md) (iteration regions) and
[](coupling-windows.md) (`CouplingWindowScope`). The multiscale and surrogate declarations
(`model/multiscale.hpp`) are covered by [](multiscale.md) and [](surrogates.md).

Maturity: covered by the automated test suite, including descriptor round-trip and cross-rank
byte-agreement tests.

## `fuse::ir`: the dataflow graph

`fuse::lower(graph)` compiles a `CouplingGraph` into a `fuse::OperatorDAG`: the operator
primitives (`Sample`, `Accumulate`, `Remap`, `Transport`, `Merge`, `Relax`, ...) connected by the
values they pass, plus `Source`/`Sink` boundary nodes. `fuse::dump_text(dag)` prints it in a fixed
text format; `fuse::dump_graphviz(dag)` as a DOT graph.

```cpp
const auto lowered = fuse::lower(graph);
if (!lowered.ok()) { std::cerr << lowered.diags.to_string(); return 1; }
std::cout << fuse::dump_text(lowered.dag);
for (std::uint32_t i = 0; i < lowered.dag.node_count(); ++i) {
  const fuse::Node& n = lowered.dag.node(fuse::NodeId{i});
  if (n.kind == fuse::OpKind::Remap) { /* n.label, n.inputs, n.output */ }
}
```

Covered by [](ir.md) and [](quickstart.md)'s "Seeing what the compiler did".

Maturity: exercised by every example and by golden-file tests of its dumped shape.

## `fuse::schedule`: the compiled instruction stream

`fuse::schedule(graph, dag, clocks, region_sizes)` turns the dataflow graph into a flat
`Instruction` stream for one coupling cycle, after `fuse::prove_acyclic()`'s initialisation-time
deadlock proof and `fuse::ClockSet`'s common-period arithmetic (`schedule/clock.hpp`). The opt-in
multirate path is declared with `fuse::CouplingWindowScope` and `DtPolicy` (`model/iteration.hpp`)
and expanded in `schedule/window.hpp`.

```cpp
const fuse::ClockSet clocks{graph};
auto sched = fuse::schedule(graph, lowered.dag, clocks,
                            fuse::make_region_sizes(graph, lowered.dag, backend));
if (!sched.ok()) { std::cerr << sched.diags.to_string(); return 1; }
std::cout << sched.plan.to_string();      // period, buffers, one line per instruction
const fuse::Rational period = sched.plan.period();
```

Covered by [](ir.md), [](quickstart.md) and [](coupling-windows.md).

Maturity: covered by the automated test suite, including multi-rank runs of the window path.

## `fuse::runtime`: executing the schedule

`fuse::Runner` executes a compiled schedule against a `ComponentSet` and a `FieldRegistry`;
`fuse::Component` is the interface a model implements (`register_fields()`, `advance()`,
optionally `save_state()`/`restore_state()` and the restart hooks); `fuse::ImplicitComponent` adds
`evaluate_residual()` and `linearization()` for Newton. Rank agreement at a synchronisation point
is `Epoch` (`runtime/epoch_schedule.hpp`). `fuse::WindowCoupler` is the standalone driver for a
time-window iteration outside the compiled path. Restart writing and restoring is
`runtime/restart.hpp`.

```cpp
fuse::FieldRegistry registry;
for (auto& c : components) { if (!c.register_fields(registry, g).ok()) return 1; }
fuse::Runner runner{g, lowered.dag, sched.plan, backend};
if (!runner.initialize(components, registry, fuse::RunnerConfig{}).ok()) return 1;
if (!runner.run(components, /*periods=*/10).ok()) return 1;
const fuse::RunStats& s = runner.stats();                 // periods, conservation, iterations
for (const fuse::IterationStats& it : runner.iteration_stats()) { /* it.converged, it.residual_history */ }
```

Covered by [](runtime.md), [](component-interface.md), [](drivers.md), [](quickstart.md),
[](coupling-windows.md), [](restart.md) and [](c-abi.md) (`fuse_begin_step`/`fuse_end_step`
resume the same `Runner`).

Maturity: covered by the automated test suite, including multi-rank and multi-executable runs.

## `fuse::num`: the numeric kernels a step uses

- `fuse::Accelerator`: Picard through Aitken, Steffensen, secant, IQN-ILS/IMVJ
  (`num/accelerator.hpp`), configured from the graph's `AccelConfig`.
- `fuse::num::field_norm(NormKind::MeasureWeightedRms, ...)` (`num/residual_norm.hpp`): the
  resolution- and partition-independent convergence measure.
- `fuse::Waveform` (`num/waveform.hpp`): a field's trajectory over a time window.
- Portable per-step kernels (`num/space_kernels.hpp`: `accumulate`, `gather`, `scatter`,
  `remap_apply`, `relax`, `dot`, `weighted_norm_squared`), checked on every Kokkos execution space
  the build enables against the plain-loop reference in `num/kernels.hpp`.
- `fuse::num::solve()` (`num/newton_krylov.hpp`, `num/line_search.hpp`): the Jacobian-free
  Newton-Krylov solver a region selects with `AccelKind::Newton`: finite-difference Jacobian
  action, right-preconditioned GMRES with Eisenstat-Walker forcing, three line searches, named
  convergence and divergence codes, and a block-Jacobi preconditioner composed from each member's
  `fuse::LinearizationToken`.

```cpp
fuse::num::ResidualField f{.residual = r, .measure = w, .count = n, .units = fuse::units::kKelvin};
const auto norm = fuse::num::field_norm(fuse::num::NormKind::MeasureWeightedRms, f);
if (norm.ok()) { use(norm.value().value); }        // in kelvin; .owned_count says what it covered
// Selected from the graph, not called directly:
const fuse::AccelConfig cfg{.kind = fuse::AccelKind::IQN_ILS, .filter = fuse::FilterKind::None};
```

Covered by [](convergence.md), [](coupling-windows.md), [](linear-algebra.md) and
[](field-pipeline.md); the developer page is `dev/newton-krylov.md`.

Maturity: covered by the automated test suite; the window form of Newton runs on one rank only.

## `fuse::petsc`: the PETSc substrate (optional)

Built only under `FUSE_ENABLE_PETSC`. `fuse::petsc::install()` registers PETSc's `SNESNEWTONLS` as
a Newton substrate (`fuse::petsc::solve()`, `petsc/solve.hpp`), selected per region by
`NewtonConfig::substrate = NewtonSubstrate::Petsc`; without it a region naming that substrate
refuses with `E-NEWTON-PETSC-UNAVAILABLE` at `Runner::initialize()`. `MffdToken` and
`make_assembled_token` (`petsc/token.hpp`) are `fuse::LinearizationToken`s whose Jacobian action is
`MATMFFD`/`MATAIJ` and whose preconditioner is a `PC`. No PETSc type appears in any public header.

```cpp
#include <fuse/petsc/solve.hpp>
fuse::petsc::install();                                  // once, before Runner::initialize()
fuse::NewtonConfig cfg;
cfg.substrate = fuse::NewtonSubstrate::Petsc;
g.iterate({flux, temp}).until(crit).newton(cfg);
```

See [](petsc-reaction-diffusion.md) for a worked example and [](linear-algebra.md).

Maturity: requires PETSc; see [](build.md).

## `fuse::backend`: meshes and remap

`fuse::MeshView`/`fuse::MeshBackend` is the one interface every mesh backend implements:
`fuse::SerialMeshBackend` (dependency-free, structured), `fuse::native::NativeMeshBackend` (a
distributed unstructured mesh on Kokkos views, with its own halo exchange and RCB-driven migration),
and `fuse::MoabMeshBackend` (requires MOAB). Every remap weight builder returns the same
`fuse::RemapWeights` triplet: `fuse::build_supermesh_weights()` (the always-available exact-simplex
path), the MOAB/TempestRemap tier, `fuse::build_mls_weights()` (point clouds), and
`fuse::build_consistent_interp_weights()` and its transpose (nodal interpolation).

```cpp
fuse::SerialMeshBackend backend;
for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
  if (!backend.load(g.mesh(fuse::MeshId{i})).ok()) return 1;
}
auto src = backend.view(fuse::MeshId{0});
auto tgt = backend.view(fuse::MeshId{1});
auto w = fuse::build_supermesh_weights(*src.value(), *tgt.value(), /*dim=*/1,
                                       fuse::Conservation::Extensive);
// w.value().row / .col / .val: the sparse weight matrix, plus its conservation audit
```

Covered by [](meshes.md), [](geometry.md) (the mesh backends and the remap ladder),
[](transfer-maps.md) (persisting weights), [](point-cloud-remap.md) and [](interface-transfer.md).

Maturity: the serial backend, native mesh and supermesh path are covered by serial tests; the native
mesh halo and the MOAB backend are also covered by multi-rank tests (these need MPI, and MOAB for
the latter). MLS remap is serial-only: it rejects a multi-rank backend.

## `fuse::linalg`: distributed linear algebra

- `fuse::linalg::DistVector`/`DistSparseMatrix`: row-distributed, with an owned block and a ghost
  region refreshed by an injected `Reducer`/`GhostExchanger` (`linalg/comm.hpp`;
  `make_serial_reducer()`/`make_serial_exchanger()` for one process).
- `fuse::linalg::pcg()`/`gmres()`: preconditioned Krylov solvers over any `LinearOperator` and
  `Preconditioner` (`IdentityPreconditioner`, `JacobiPreconditioner::from_matrix()`).
- `fuse::linalg::SchurSlabSolver`: an exact distributed direct solver for a row-banded periodic
  Laplacian.
- The derivative gate (`linalg/derivative_gate.hpp`: `adjoint_gate`, `taylor_remainder_order`,
  `matrix_free_agreement`), which checks that an operator claiming a transpose or a Jacobian is
  consistent before a Newton or adjoint claim relies on it.

```cpp
using namespace fuse::linalg;
auto reducer = make_serial_reducer();
auto A = DistSparseMatrix::from_triplets(row_lo, row_hi, n, my_triplets, reducer, owners);
DistVector x(row_hi - row_lo, make_serial_exchanger(), reducer), b(/* same */);
auto m = JacobiPreconditioner::from_matrix(A.value());
auto report = gmres(A.value(), m.value(), x, b, GmresOptions{.rel_tol = 1e-10, .restart = 30});
if (report.ok()) { use(report.value().iterations); }
```

Covered by [](linear-algebra.md); [](glossary.md)'s "Linear algebra" section and [](build.md) (the
`FUSE_LINALG_BACKEND` choice) touch it. Developer pages: `dev/linalg-distributed.md`,
`dev/linalg-krylov.md`, `dev/schur-slab-solver.md`, `dev/derivative-gate.md`.

Maturity: covered by serial tests against Eigen and by multi-rank tests (which need MPI).

## `fuse::partition`: mesh repartitioning

Recursive coordinate bisection: `fuse::partition::rcb_partition()` over any mesh's centroids (1D,
2D planar or spherical, 3D volumetric), and `fuse::partition::repartition()`, which composes it
with `fuse::native::migrate()` to repartition a native mesh end to end.

```cpp
fuse::partition::RepartitionOptions opts;
opts.comm = comm_f;                                   // MPI_Comm_c2f(comm); kNullComm for one process
auto moved = fuse::partition::repartition(mesh, opts); // Result<std::unique_ptr<native::NativeMeshView>>
```

Covered by [](meshes.md); developer page `dev/partitioning.md`.

Maturity: covered by the automated test suite.

## `fuse::search`: spatial queries

A facade over ArborX that no public header outside `fuse::search` itself names:
`fuse::search::BoxIndex` (a host-space BVH over axis-aligned boxes, batched nearest/intersection
queries) and `fuse::search::query_points_distributed()` (the distributed counterpart).
`fuse::backend` and `fuse::partition` consume it without depending on ArborX directly.

```cpp
auto index = fuse::search::BoxIndex::build(std::move(boxes));   // std::vector<fuse::search::Box>
auto hits = index.value().query_points(points);                  // which boxes contain each point
auto near = index.value().query_nearest(points, /*k=*/4);        // the k nearest boxes per point
```

Used by [](point-cloud-remap.md) (MLS's nearest-neighbour search) and the nodal-interpolation path
[](interface-transfer.md) describes.

Maturity: covered by the automated test suite.

## `fuse::transport`: moving a field between processor sets

`fuse::Transport`/`fuse::CommGraph` (which physical rank holds, or wants, this data, for two
processor sets that may be disjoint, identical or partially overlapping);
`fuse::transport::connect()` (the launch/rendezvous handshake for a multi-executable coupling,
both MPMD and separate-launch accept/connect); `fuse::LowFiveTransport` (intercepted-HDF5
transport). The transport is chosen by name in `RunnerConfig`:

```cpp
fuse::RunnerConfig cfg;
cfg.transport         = "mpi";                               // "copy" (default), "mpi", "lowfive"
cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
cfg.transport_setup   = [](fuse::Transport& t) -> fuse::Result<void> {
  auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
  if (mpi == nullptr) return fuse::Failure{"E-APP-TRANSPORT", "expected an MpiTransport"};
  return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
};
```

Covered by [](placement.md), [](multiple-executables.md) and [](lowfive.md).

Maturity: requires MPI for the multi-rank and multi-executable paths, and LowFive for
`LowFiveTransport`; see [](build.md). `CommGraph` and MPMD `connect()` are covered by multi-rank
tests.

## `fuse::field`: field storage and the lazy view pipeline

`fuse::FieldRegistry` maps a `FieldId` to the component-owned storage bound at setup;
`fuse::FieldSpan`/`fuse::OwnedField` are the lazy `slice`/`component`/`owned` view chain over a
`fuse::Extent`.

```cpp
auto f = fuse::FieldSpan<>::from(ref);                       // ref: the FieldRef you registered
const double s = f.owned(n_owned).component(0).local_sum();  // stages are views; local_sum reads
```

Covered by [](field-pipeline.md), [](fluent-api.md) and [](component-interface.md).

Maturity: covered by the automated test suite, including an allocation-count check of the laziness
guarantee and a compile-fail test of the dangling-view rule.

## `fuse::restart_io`: restart files (optional)

Built only under `FUSE_ENABLE_HDF5`. `fuse::restart_io::create_restart_point()` and
`open_restart_point()` (`restart_io/hdf5_restart_io.hpp`) return a `PointWriter`/`PointReader` that
plug into `fuse::write_restart_point()`/`fuse::restore_restart_point()`
(`runtime/restart.hpp`).

```cpp
fuse::restart_io::Provenance prov;
prov.descriptor_hash = fuse::restart_descriptor_hash(graph);
auto pw = fuse::restart_io::create_restart_point(
    fuse::restart_io::FileLayout::SeparateMeshAndPoints, "run.0003.h5", prov);
auto w = fuse::write_restart_point(runner, components, pw.value()->writer(),
                                   now, runner.current_period_index());
if (w.ok()) { (void)pw.value()->finish(); }           // the point exists only after finish()
```

Covered by [](restart.md).

Maturity: requires HDF5; see [](restart.md) for the build options.

## `fuse::abi`: the two C ABIs

Generated from one entry list, `include/fuse/abi/entries.def`: `fuse_coupler_abi()` returns the
participant table (your code drives FUSE), `fuse_provider_abi()` the provider table (FUSE drives
your plugin), plus `ffuse`/`ffuse_provider`, the matching Fortran bindings, generated from the same
list so none of the three can disagree.

```c
#include <fuse/abi/participant.h>
const fuse_participant_abi_v1* fuse = fuse_coupler_abi(FUSE_ABI_VERSION);
fuse_context_out co = {.struct_size = sizeof(co)};
fuse->create(NULL, NULL, &co);
fuse_graph_source src = {.struct_size = sizeof(src)};
src.yaml_path = "coupling.yaml";
fuse->load_graph(co.ctx, &src, NULL);
fuse->initialize(co.ctx, NULL, NULL);     /* then register_field, begin_step/end_step ... */
```

Covered by [](mixing-languages.md), [](c-abi.md), [](fortran.md) and [](providers.md).

Maturity: both tables are exercised by the automated tests, and the Brusselator example drives a
multi-executable coupling through them.

## `pfuse`: the Python binding

A `pybind11` module over `fuse::CouplingGraph`/`fuse::Component`/`fuse::Runner` directly, not the
ABI above: a Python physics component is an in-process peer, authored and run the same way a native
one is. It binds the graph-authoring API, the iteration/accelerator/Newton-configuration
vocabulary, `FieldRegistry` (numpy-backed, with the same buffer-contract and lifetime rules a C++
caller gets from `fuse::FieldRef`), and a `Session` wrapper over `fuse::Runner`. Covered by
[](python.md) and [](mixing-languages.md).

Maturity: built only under `FUSE_ENABLE_PYTHON`; covered by its own tests, including a Python
reproduction of the `dn_heat` example.

## See also

- [](getting-started.md), [](concepts.md) and [](cookbook.md) for introductions and recipes.
- [](environment.md) for environment variables and execution-space selection.
- [](power-users.md) for advanced configuration.
- [](troubleshooting.md) for error codes.
- [](examples.md) for the worked examples.
