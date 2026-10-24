# Worked examples

The programs in `examples/` are complete couplings that drive the library the way an application
does: build a coupling specification, lower it, schedule it, run it. Use this page to find the one
closest to your problem; each has its own page with the problem, the authored graph, build and run
commands, what it prints, and the results it is checked against.

Each example is gated on a physics or correctness property rather than on "did not crash": it
exits non-zero when its measured answer misses an analytic gate, a conservation check never ran,
or a decomposed or accelerated path disagrees with its own reference. `ctest` runs every example
whose build requirements are met. See [Building FUSE](build.md) for the presets and options.

## The examples

| Example (page) | Executable | Problem | What it demonstrates | Build options |
|---|---|---|---|---|
| [`mms_pair`](mms-pair.md) | `fuse_example_mms_pair` | 1D heat pair, manufactured solution | explicit (lagged) coupling, time-mean accumulation, conservation check, dumping the IR and plan | base build |
| [`dn_heat`](dn-heat.md) | `fuse_example_dn_heat` | Dirichlet-Neumann composite-wall conduction | implicit coupling group, two-field convergence criterion, Aitken / IQN-ILS / Newton acceleration, checkpoint-restore, restart hooks | base build; `--newton-petsc` needs `FUSE_ENABLE_PETSC` |
| [`petsc_reaction_diffusion`](petsc-reaction-diffusion.md) | `fuse_example_petsc_reaction_diffusion` | nonlinear reaction-diffusion, Dirichlet-Neumann split | a PETSc-object `LinearizationToken` (`Mat`/`Vec`, `MatMult`/`MatMultTranspose`) paired with a native-substrate twin | base build for the native twin; `FUSE_ENABLE_PETSC` for `--petsc` |
| [`gray_scott`](gray-scott.md) | `fuse_example_gray_scott` | Gray-Scott reaction-diffusion, two species on one grid | implicit coupling, multirate (independent time steps), subcycling, `DtPolicy`, Aitken / IQN-ILS | `FUSE_LINALG_BACKEND=eigen` |
| [`brusselator`](brusselator.md) | `fuse_example_brusselator`; participants `brusselator_graph_tool`, `brusselator_u_main`, `brusselator_v_main`, `brusselator_coupler_main`, `brusselator_check` | multirate Brusselator, two independent native meshes | coupling windows (`WindowCoupler` and `Runner`), conservative supermesh remap, grid decomposition with a pluggable diffusion solver, Newton (native and PETSc), form A restart hooks keyed by mesh global id, multi-executable C and Fortran participants | `FUSE_LINALG_BACKEND=eigen`; `FUSE_ENABLE_MPI` for the participants (the Fortran participant also needs a Fortran compiler); `FUSE_ENABLE_PETSC` and `FUSE_ENABLE_LOWFIVE` optional |
| [`fsi_channel`](fsi-channel.md) | `fuse_example_fsi_channel` | unsteady-Stokes channel with an Euler-Bernoulli wall beam | strongly coupled FSI on a non-matching 1D interface (consistent / conservative-transpose remap), Aitken / IQN-ILS, added-mass instability | `FUSE_LINALG_BACKEND=eigen`; its physics rows are disabled under `FUSE_ENABLE_CUSPARSE` |
| [`layout_driver`](layout-driver.md) | `fuse_example_layout_driver` | none (a placement test driver) | processor-set placement (disjoint / partial / full overlap), coupler processor set, mesh migration (RCB) | `FUSE_ENABLE_MPI` |
| [`demo_b`](demo-b.md) | `demo_b_producer`, `demo_b_consumer` | none (a transport demonstration) | distributed LowFive transport: two unmodified HDF5 executables coupled by VOL interception across a process boundary | `FUSE_ENABLE_MPI`, `FUSE_ENABLE_LOWFIVE` with the distributed archive, `FUSE_ENABLE_CONNECT_ACCEPT` |

## Command-line flags

Every flag below is read by the example's own argument parser; an unknown flag is refused. Values
in brackets are the defaults where the parser sets one.

| Example | Flags |
|---|---|
| `mms_pair` | `--dump` (print the operator IR and the execution plan, run nothing) |
| `dn_heat` | `--dump`; `--newton-petsc` (run the Newton region on the PETSc substrate) |
| `petsc_reaction_diffusion` | `--petsc` (run the PETSc-token leg instead of the native twin) |
| `gray_scott` | `--n N` (grid), `--dt DT`, `--ratio R` (time-step ratio of the two species), `--policy fixed\|adjustable` [adjustable], `--accel none\|constant\|aitken\|secant\|steffensen\|iqn-ils\|iqn-imvj` [aitken], `--t-end T`, `--check`, `--output PATH`, `--F`, `--k`, `--Du`, `--Dv`, `--L` (model parameters) |
| `fsi_channel` | `--nx-fluid N`, `--ny-fluid N`, `--nx-beam N`, `--mass-ratio R`, `--accel ...` (same names as `gray_scott`) [iqn-ils], `--dt DT`, `--t-end T`, `--csv PATH`, `--check` |
| `brusselator` | `--nx-u --ny-u --nx-v --ny-v` (grids), `--dt-u --dt-v`, `--window W`, `--end T`, `--alpha A`, `--tolerance TOL`, `--max-iterations N`, `--accel none\|constant\|aitken\|secant\|steffensen\|iqn-ils\|iqn-imvj\|newton`, `--driver window\|runner` [window], `--iteration-mode jacobi\|gauss-seidel` [jacobi], `--substrate native\|petsc` [native; `--accel newton` only], `--no-block-jacobi`, `--no-forcing`, `--save-history`, `--out DIR`, `--check`, `--help` |
| `brusselator` participants | `brusselator_graph_tool`: `--launch mpmd\|accept`, `--ranks-u`, `--ranks-v`, `--ranks-coupler`, `--transport`, `--port-file`, `--out`, plus the grid/step flags above; `brusselator_u_main` / `brusselator_v_main`: `--graph FILE`, `--periods N`, `--log FILE`, `--solver`, `--transport`, `--ranks-u`, `--ranks-v`, `--ranks-coupler`; `brusselator_coupler_main`: `--graph FILE`, `--periods N`; `brusselator_check`: `--u-log`, `--v-log`, `--check` |
| `layout_driver` | `--comp1 a:b`, `--comp2 c:d`, `--coupler e:f` (inclusive world-rank ranges), `--entities N` [100], `--steps K` [1], `--check`, `--timing out.csv`, `--mesh SPEC` (`hexbox:NxMxK`, `tetbox:...`, `prismpyramidmix:...`), `--migrate` |
| `demo_b` | positional: `<port_file> <hdf5_file> <dataset_name>` for both `demo_b_producer` and `demo_b_consumer` |

An unknown `--accel` name is refused by name (`E-EXAMPLE-ACCEL`), and `fsi_channel` refuses a `--dt`
that rounds to zero on its 1e-6 time grid (`E-EXAMPLE-DT`).

## `ctest` rows

All example rows carry the label `example`; rows that need PETSc also carry `petsc`, the
thread-count rows `omp`, and the layout-driver rows `mpi;parallel;layout`.

| Row | Command | Registered when |
|---|---|---|
| `example.mms_pair`, `example.mms_pair.dump` | `fuse_example_mms_pair [--dump]` | always |
| `example.dn_heat`, `example.dn_heat.dump` | `fuse_example_dn_heat [--dump]` | always |
| `example.dn_heat.newton_petsc` | `fuse_example_dn_heat --newton-petsc` | `FUSE_ENABLE_PETSC` |
| `example.petsc_reaction_diffusion` | `fuse_example_petsc_reaction_diffusion` | always |
| `example.petsc_reaction_diffusion.petsc` | `... --petsc` | `FUSE_ENABLE_PETSC` |
| `example.gray_scott` | `fuse_example_gray_scott --n 24 --ratio 2 --t-end 4 --dt 0.1 --check` | eigen backend |
| `example.gray_scott.accel_refused` | `... --accel no-such-method` (expects `E-EXAMPLE-ACCEL`) | eigen backend |
| `example.brusselator` | `fuse_example_brusselator --end 0.5 --check` | eigen backend |
| `example.brusselator.accel_refused` | `... --accel no-such-method` | eigen backend |
| `example.brusselator.runner` | `... --driver runner --end 0.5 --check` | eigen backend |
| `example.brusselator.newton` | `... --driver runner --accel newton --end 0.1 --check` | eigen backend |
| `example.brusselator.newton_petsc` | `... --driver runner --accel newton --substrate petsc ...` | eigen backend and `FUSE_ENABLE_PETSC` |
| `verification.brusselator.participants.{mpmd,accept}.{3,5}` | `run_participants.sh` launching the graph tool, U, V and coupler executables | `FUSE_ENABLE_MPI` and the participant targets |
| `verification.brusselator.participants.lowfive.{mpmd,accept}.3` | the same over the LowFive transport | plus `FUSE_ENABLE_LOWFIVE` |
| `verification.brusselator.graph_tool.reproduces_reference*` | `brusselator_graph_tool` against the reference YAML files | always |
| `example.fsi_channel` | `fuse_example_fsi_channel --nx-fluid 16 --ny-fluid 8 --nx-beam 12 --mass-ratio 1.0 --accel iqn-ils --dt 0.01 --t-end 0.1 --check` | eigen backend; disabled under cuSPARSE |
| `example.fsi_channel.accel_refused`, `example.fsi_channel.off_grid_dt`, `example.fsi_channel.dt_refused` | the refusal and off-grid `--dt` legs | eigen backend (`off_grid_dt` disabled under cuSPARSE) |
| `example.<name>.omp_identical` for `mms_pair`, `dn_heat`, `gray_scott`, `brusselator`, `fsi_channel` | the same binary at one OpenMP thread and at every host core, diffed byte for byte (`tools/compare-runs.cmake`) | `FUSE_KOKKOS_HAS_OPENMP` |
| `example.mms_pair.device`, `example.dn_heat.device`, `example.gray_scott.device`, `example.brusselator.runner.device` | the same command line under `FUSE_EXECUTION_SPACE=device` | a Kokkos device backend |
| `layout.2.*`, `layout.3.*`, `layout.4.*` | `mpiexec -n P fuse_example_layout_driver --comp1 a:b --comp2 c:d --check` over disjoint, partially overlapping and identical placements | `FUSE_ENABLE_MPI` |
| `layout.coupler.3.disjoint`, `layout.coupler.3.equals-target`, `layout.coupler.4.overlaps-comp1` | the same with `--coupler e:f` | `FUSE_ENABLE_MPI` |
| `layout.migrate.{2,3,4}.*` | `--mesh ... --migrate` | `FUSE_ENABLE_MPI` |
| `layout.refuse.min-nonzero`, `layout.refuse.max-not-last`, `layout.refuse.reversed-range` | malformed layouts; expected to fail | `FUSE_ENABLE_MPI` |
| `example.demo_b` | `demo_b/run_demo.sh` launching producer and consumer | `FUSE_ENABLE_MPI`, `FUSE_ENABLE_LOWFIVE`, `LOWFIVE_HAVE_DIST`, `FUSE_ENABLE_CONNECT_ACCEPT` |

The `.dump` rows assert on the printed output rather than the exit status, because an empty dump
still exits 0.

## Thread-count determinism

Every physics example (`mms_pair`, `dn_heat`, `gray_scott`, `brusselator`, `fsi_channel`) has an
`example.<name>.omp_identical` row: the same binary at one OpenMP thread and at every host
core, diffed byte for byte. Each example's own components declare `DeterminismLevel::Bitwise`,
and the Runner's coupling hot loop runs on Kokkos' host execution space, so a thread-count change
that perturbed a reduction order would show up here first. Run them with `ctest -L omp`.

## Seeing the compiled plan

`mms_pair` and `dn_heat` accept `--dump`, which prints the operator IR and the flat execution plan
without running anything:

```
./examples/fuse_example_dn_heat --dump
```

## Running the examples

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
ctest --test-dir build-tier0-release -L example   # the examples' own rows, plus dump/omp legs
ctest --test-dir build-tier1 -L layout            # the layout driver, in an MPI build
```

Rows that need PETSc carry the additional label `petsc` (`ctest -L petsc`).

## See also

- [](portfolio.md) for a curated reading order, a capability matrix and a "learn X, read Y" index.
- [](getting-started.md) and [](quickstart.md) for a first coupling.
- [](cookbook.md) for short recipes.
- [](build.md) for build options and presets.
