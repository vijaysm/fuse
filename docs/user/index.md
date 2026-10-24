# User guide

Pages are grouped by what you are trying to do. [](glossary.md) defines every term used here.

## Start here

| Page | For |
|---|---|
| [](build.md) | What to install and which options to turn on |
| [](getting-started.md) | Build, run and read a first coupling |
| [](concepts.md) | The mental model, before you write code |
| [](quickstart.md) | A complete two-component coupling, with real code |
| [](cookbook.md) | A recipe for each common task |
| [](component-interface.md) | The contract a component implements |
| [](drivers.md) | Writing your own executable: command line, YAML, configuration |
| [](troubleshooting.md) | Error codes, debugging and common questions |

## Describing a coupling

[](graph-programming.md) is the complete guide to building a coupling graph in code and
[](yaml.md) the reference for the same graph as a file. [](api-overview.md) maps the public namespaces. [](fluent-api.md) covers the chaining idiom and its
rules, [](placement.md) and [](layouts.md) where components run, [](combining-writers.md) several
sources into one field, and [](field-pipeline.md) reading field data lazily.

## Compiling and running

[](ir.md) explains the operator IR and the execution plan that FUSE compiles a graph into, and how
to read, diff and extend them. [](runtime.md) covers the Runner, its configuration, stepping and
statistics, and [](environment.md) every environment variable and runtime option.

## Meshes and transfers

[](meshes.md) covers mesh backends, generated meshes and partitioning. [](geometry.md) covers the mesh backends and the remap ladder, [](transfer-maps.md) persisting
weights, [](point-cloud-remap.md) remapping scattered points, and [](interface-transfer.md) a strongly
coupled interface between differently discretised solvers.

## Time, convergence and scales

[](convergence.md) explains how an implicit coupling decides it is done,
[](coupling-windows.md) couples components that do not share a time grid, and
[](linear-algebra.md) covers the solvers behind Newton-Krylov coupling. [](multiscale.md) covers
multiscale coupling and [](surrogates.md) learned models as components.

## Bringing your own code

[](c-abi.md) and [](fortran.md) let an existing code drive FUSE. [](providers.md) is the other
direction: FUSE driving a plugin you write. [](python.md) is the Python module and [](lowfive.md)
couples two unmodified HDF5 codes. [](multiple-executables.md) splits a coupling across separately
built executables, [](mixing-languages.md) says which combinations work together and how, and
[](restart.md) writes a run's state to disk to resume later.

## Worked examples

[](examples.md) lists the complete programs: [](mms-pair.md), [](dn-heat.md), [](gray-scott.md),
[](brusselator.md), [](fsi-channel.md), [](petsc-reaction-diffusion.md), [](layout-driver.md) and
[](demo-b.md).

## For power users

[](power-users.md) covers builds for performance, execution spaces, reproducibility, scale and
diagnostics.
