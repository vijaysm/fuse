# Worked examples

The programs in `examples/` are complete couplings that drive the library the way an application
does: build a coupling specification, lower it, schedule it, run it. Use this page to find the one
closest to your problem; each has its own page with the problem, the authored graph, build and run
commands, what it prints, and the results it is checked against.

Eight examples are listed below. Each is gated on a physics or correctness property rather than on
"did not crash": it exits non-zero when its measured answer misses an analytic gate, a conservation
check never ran, or a decomposed or accelerated path disagrees with its own reference. `ctest`
runs every example whose build requirements are met. See [Building FUSE](build.md) for the build
presets and options.

| Example | Problem | Coupling exercised | Requires |
|---|---|---|---|
| [`mms_pair`](mms-pair.md) | 1D heat pair, manufactured solution | explicit (lagged) coupling, time-mean accumulation, conservation check | base build |
| [`dn_heat`](dn-heat.md) | Dirichlet-Neumann composite-wall conduction | implicit coupling group, two-field convergence criterion, Aitken / IQN-ILS acceleration, checkpoint-restore | base build; the `--newton-petsc` mode needs PETSc |
| [`petsc_reaction_diffusion`](petsc-reaction-diffusion.md) | Single-variable nonlinear reaction-diffusion, Dirichlet-Neumann split | a PETSc-object `LinearizationToken` (`Mat`/`Vec`, `MatMult`/`MatMultTranspose`, never `MATMFFD`) paired with a native-substrate twin | base build for the native twin; PETSc (`FUSE_ENABLE_PETSC`) for the `--petsc` leg |
| [`gray_scott`](gray-scott.md) | Gray-Scott reaction-diffusion, two species on one grid | implicit coupling, multirate (independent time steps), subcycling, Aitken / IQN-ILS | eigen linear-algebra backend |
| [`brusselator`](brusselator.md) | Multirate Brusselator, two independent native meshes | implicit coupling (`WindowCoupler` and `Runner`), conservative supermesh remap, grid decomposition with a pluggable diffusion solver, multi-executable (MPMD) participants | eigen backend; MPI for the multi-executable participants (the Fortran participant also needs a Fortran compiler); PETSc and LowFive optional |
| [`fsi_channel`](fsi-channel.md) | Unsteady-Stokes channel with an Euler-Bernoulli wall beam | implicit coupling group, strongly coupled FSI on a non-matching 1D interface (consistent / conservative-transpose remap), Aitken / IQN-ILS, added-mass instability | eigen backend; its rows are disabled under `FUSE_ENABLE_CUSPARSE` |
| [`layout_driver`](layout-driver.md) | none (a test driver, not a physics model) | processor-set placement (disjoint / partial / full overlap), coupler processor set, mesh migration (RCB) | MPI |
| [`demo_b`](demo-b.md) | none (a transport demonstration, not a physics model) | distributed LowFive transport: two unmodified HDF5 executables coupled by VOL interception across a real process boundary | MPI; LowFive with its distributed archive; connect/accept enabled |

## Thread-count determinism

Every physics example (`mms_pair`, `dn_heat`, `gray_scott`, `brusselator`, `fsi_channel`) has an
`example.<name>.omp_identical` `ctest` row: the same binary at one OpenMP thread and at every host
core, diffed byte for byte (`tools/compare-runs.cmake`). Each example's own components declare
`DeterminismLevel::Bitwise`, and the Runner's coupling hot loop runs on Kokkos' host execution space, so a thread-count change that perturbed a reduction
order would show up here first. Registered only when `FUSE_KOKKOS_HAS_OPENMP`; run them with
`ctest -L omp`.

## Seeing the compiled plan

`mms_pair` and `dn_heat` accept `--dump`, which prints the operator IR and the flat execution plan
without running anything:

```
./examples/fuse_example_dn_heat --dump
```

The `ctest` entries for the dump path assert on its *output* rather than the exit status, because an
empty dump still exits 0.

## Running the examples

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
ctest --test-dir build-tier0-release -L example   # the physics examples' own rows, plus dump/omp legs
```

`layout_driver` and `demo_b` need `FUSE_ENABLE_MPI` (and, for `demo_b`, LowFive with its
distributed archive), so a build without those does not register them. Rows that need PETSc carry
the additional label `petsc` (`ctest -L petsc`).

## See also

- [](getting-started.md) and [](quickstart.md) for a first coupling.
- [](cookbook.md) for short recipes.
- [](build.md) for build options.
