# Example portfolio

A curated tour of the programs in `examples/`, ordered so that each one adds a single capability to
the one before it. Read it when you know *what* you want to couple but not *which* example is closest.
It is the companion to two other pages: [](examples.md) lists the same programs with their flags and
`ctest` rows, and [](cookbook.md) is organised by task rather than by example. Every program here is a
complete coupling that `ctest` runs and gates on a physics or correctness property, not on "did not
crash".

## A reading order

Each step is a working program you can build and run; later steps assume the vocabulary of the
earlier ones. Build options are summarised in [](examples.md#the-examples) and covered in full in
[](build.md).

1. **[](mms-pair.md)** — the smallest complete coupling. Two in-process components, one field,
   explicit (lagged) exchange, verified against a closed-form solution. No MPI, no MOAB, no Eigen.
   Start here, right after [](quickstart.md).
2. **[](dn-heat.md)** — turn the explicit exchange into an *implicit* one. A Dirichlet-Neumann loop
   iterated to a fixed point, with a two-field convergence criterion and a choice of accelerators
   (Aitken, IQN-ILS, Newton). Also shows checkpoint-restore and restart hooks.
3. **[](gray-scott.md)** — two components that keep *two different clocks*. Multirate coupling,
   subcycling and a `DtPolicy`, still on one shared grid. Needs the Eigen backend.
4. **[](brusselator.md)** — give each component its *own* native mesh and a conservative supermesh
   remap between them, then split the coupling across separately built C and Fortran executables over
   coupling windows. The most feature-complete example.
5. **[](fsi-channel.md)** — a strongly coupled fluid-structure interface on a *non-matching* mesh,
   where the added-mass effect makes a naive fixed-point iteration diverge and the accelerators earn
   their keep.
6. **[](petsc-reaction-diffusion.md)** — hand the Newton solve to PETSc through a `Mat`/`Vec`
   `LinearizationToken`, paired with a native-substrate twin so the two agree.
7. **[](layout-driver.md)** and **[](demo-b.md)** — the plumbing examples: processor-set placement
   and mesh migration from the command line, and two *unmodified* HDF5 programs coupled by LowFive
   interception across a process boundary.

## If you want to learn…

| You want to learn | Read |
|---|---|
| How a coupling is built, lowered, scheduled and run | [](mms-pair.md) |
| How to read the compiled IR and execution plan (`--dump`) | [](mms-pair.md), [](dn-heat.md) |
| Explicit (lagged) coupling and what the lag costs | [](mms-pair.md) |
| Implicit coupling iterated to a fixed point | [](dn-heat.md) |
| Choosing and comparing accelerators (Aitken, IQN-ILS, Newton) | [](dn-heat.md), [](fsi-channel.md) |
| Components with different time steps (multirate, subcycling) | [](gray-scott.md) |
| Coupling clocks that share no useful period (windows) | [](brusselator.md) |
| Conservative remap between non-matching meshes | [](brusselator.md), [](fsi-channel.md) |
| A strongly coupled interface between different discretisations | [](fsi-channel.md), [](interface-transfer.md) |
| Driving the Newton solve with PETSc objects | [](petsc-reaction-diffusion.md) |
| Splitting a coupling across separately built executables | [](brusselator.md), [](multiple-executables.md) |
| Mixing C, Fortran and C++ participants | [](brusselator.md), [](mixing-languages.md) |
| Processor-set placement and mesh migration | [](layout-driver.md), [](placement.md) |
| Coupling unmodified HDF5 codes | [](demo-b.md), [](lowfive.md) |
| Checkpoint-restore and restart hooks | [](dn-heat.md), [](restart.md) |

## Capability matrix

What each program exercises. A dash means the capability is out of scope for that example, not that
FUSE lacks it.

| Capability | mms_pair | dn_heat | gray_scott | brusselator | fsi_channel | petsc_rd | layout_driver | demo_b |
|---|---|---|---|---|---|---|---|---|
| Explicit (lagged) coupling | ✓ | | | | | | | |
| Implicit iteration to a fixed point | | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Accelerators (Aitken / IQN-ILS / Newton) | | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| Multirate / subcycling | ✓ | | ✓ | ✓ | | | | |
| Coupling windows | | | | ✓ | | | | |
| Conservative remap, non-matching meshes | | | | ✓ | ✓ | | | |
| Independent native meshes | | | | ✓ | | | ✓ | |
| PETSc substrate | | ✓ | | ✓ | | ✓ | | |
| Multiple executables | | | | ✓ | | | | ✓ |
| C / Fortran participants | | | | ✓ | | | | ✓ |
| MPI / rank placement | | | | ✓ | | | ✓ | ✓ |
| Mesh migration | | | | | | | ✓ | |
| LowFive transport | | | | ✓ | | | | ✓ |
| Restart to disk | | ✓ | | ✓ | | | | |
| Checked against an analytic / reference answer | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Running the portfolio

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
ctest --test-dir build-tier0-release -L example
```

The eigen-backend examples (`gray_scott`, `brusselator`, `fsi_channel`) build under the default
presets; the MPI and LowFive rows need the matching tiers. [](examples.md#ctest-rows) lists exactly
which row is registered under which build option.

## See also

- [](examples.md) — the reference table, command-line flags and `ctest` rows.
- [](cookbook.md) — the same ground organised by task.
- [](getting-started.md) and [](quickstart.md) — a first coupling from scratch.
