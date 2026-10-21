# Worked examples

Seven complete couplings live in `examples/`: `mms_pair`, `dn_heat`, `gray_scott`, `brusselator`,
`fsi_channel`, `layout_driver`, and `demo_b`. They are the only code in the tree that drives the
library the way a user would — build a coupling specification, lower it, schedule it, run it —
rather than assembling a test fixture, and every one of them is run by `ctest` wherever its build
configuration allows it, because an example nobody runs rots within a release. Each is gated on its
own *physics* or *correctness property*, not on "did not crash": every one exits non-zero when its
measured answer misses an analytic gate, a conservation check never ran, or a decomposed/accelerated
path disagrees with its own reference.

This page is the index; each example has its own page with the problem, the authored graph, exact
build/run commands, what it prints, and the measured results it is checked against. See
[Building FUSE](build.md) for what tier 0/1/2 mean and the exact `cmake --preset` commands.

| Example | Problem | Coupling exercised | Tier | State |
|---|---|---|---|---|
| [`mms_pair`](mms-pair.md) | 1D heat pair, manufactured solution | explicit (lagged) coupling, time-mean accumulation, conservation check | 0 | Verified |
| [`dn_heat`](dn-heat.md) | Dirichlet-Neumann composite-wall conduction | implicit coupling group, two-field convergence criterion, Aitken / IQN-ILS acceleration, checkpoint-restore | 0 | Verified |
| [`petsc_reaction_diffusion`](petsc-reaction-diffusion.md) | Single-variable nonlinear reaction-diffusion, Dirichlet-Neumann split | a real PETSc-object `LinearizationToken` (`Mat`/`Vec`, `MatMult`/`MatMultTranspose`, never `MATMFFD`) paired with a native-substrate twin | 0 | Verified (native twin); PETSc half build-verified only in this sandbox (MPI cannot start here) |
| [`gray_scott`](gray-scott.md) | Gray-Scott reaction-diffusion, two species on one grid | implicit coupling, multirate (independent time steps), subcycling, Aitken / IQN-ILS | 0 | Verified |
| [`brusselator`](brusselator.md) | Multirate Brusselator, two independent native meshes | implicit coupling (WindowCoupler and Runner), conservative supermesh remap, grid decomposition with a pluggable diffusion solver, multi-executable (MPMD) participants | 0 (in-process); 1 (participants, MPI) | Verified (tier 0 and host MPMD); accept-mode blocked by a host MPI limitation (D23), not a defect here |
| [`fsi_channel`](fsi-channel.md) | Unsteady-Stokes channel with an Euler-Bernoulli wall beam | implicit coupling group, strongly coupled FSI on a non-matching 1D interface (consistent / conservative-transpose remap), Aitken / IQN-ILS, added-mass instability | 0 | Verified; both rows `DISABLED` under `FUSE_ENABLE_CUSPARSE` (D36), and `fsi_energy_diag`'s 480-period case `SKIP`ped under it (D37) |
| [`layout_driver`](layout-driver.md) | — (a test driver, not a physics model) | processor-set placement (disjoint / partial / full overlap), coupler processor set, mesh migration (RCB) | 1 | Verified (host-recorded); build-verified only in this sandbox (MPI cannot start here) |
| [`demo_b`](demo-b.md) | — (a transport demonstration, not a physics model) | distributed LowFive transport: two unmodified HDF5 executables coupled by VOL interception across a real process boundary | 1 + LowFive (distributed) | Implemented, untested (compiles against the host's real MOAB/LowFive/MPI; never run — MPI cannot start in this sandbox, and this exact path was never run on the host either) |

## Thread-count determinism

Every physics example (`mms_pair`, `dn_heat`, `gray_scott`, `brusselator`, `fsi_channel`) has an
`example.<name>.omp_identical` `ctest` row: the same binary at one OpenMP thread and at every host
core, diffed byte for byte (`tools/compare-runs.cmake`). These are load-bearing, not incidental —
each example's own components declare `DeterminismLevel::Bitwise`, and the Runner's coupling hot
loop has run on Kokkos' host execution space since KNOWN_ISSUES.md D11 closed, so a thread-count
change that perturbed a reduction order would show up here first. Registered only when
`FUSE_KOKKOS_HAS_OPENMP`; run them with `ctest -L omp`.

## Seeing the compiled plan

`mms_pair` and `dn_heat` accept `--dump`, which prints the operator IR and the flat execution plan
without running anything:

```
./examples/fuse_example_dn_heat --dump
```

The `ctest` entries for the dump path assert on its *output* rather than the exit status, because an
empty dump still exits 0.

## Running the whole set at tier 0

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
ctest --test-dir build-tier0-release -L example   # the 5 physics examples' own rows, plus dump/omp legs
```

Measured on this build (macOS arm64, AppleClang 21.0.0, Kokkos 5.1.0 OpenMP+Serial, Eigen 5.0.1): 13
of 13 `example.*` rows pass (65.2 s total, 39.0 s of it the 5 `omp` legs), out of 179 tier-0 `ctest`
rows registered in total. `layout_driver` and `demo_b` need `FUSE_ENABLE_MPI` (and, for `demo_b`,
LowFive with its distributed archive) and are therefore absent from a tier-0 registration, not
skipped — see their own pages for the host-recorded and build-verified status each has.
