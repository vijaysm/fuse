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
| [](troubleshooting.md) | Error codes, debugging and common questions |

## Describing a coupling

[](api-overview.md) maps the public namespaces. [](fluent-api.md) covers the chaining idiom and its
rules, [](placement.md) and [](layouts.md) where components run, [](combining-writers.md) several
sources into one field, and [](field-pipeline.md) reading field data lazily.

## Meshes and transfers

[](geometry.md) covers the mesh backends and the remap ladder, [](transfer-maps.md) persisting
weights, [](point-cloud-remap.md) remapping scattered points, and [](interface-transfer.md) a strongly
coupled interface between differently discretised solvers.

## Time and convergence

[](convergence.md) explains how an implicit coupling decides it is done, and
[](coupling-windows.md) couples components that do not share a time grid.

## Bringing your own code

[](c-abi.md) and [](fortran.md) let an existing code drive FUSE. [](providers.md) is the other
direction: FUSE driving a plugin you write. [](python.md) is the Python module and [](lowfive.md)
couples two unmodified HDF5 codes. [](multiple-executables.md) splits a coupling across separately
built executables, and [](restart.md) writes a run's state to disk to resume later.

## Worked examples

[](examples.md) lists the complete programs: [](mms-pair.md), [](dn-heat.md), [](gray-scott.md),
[](brusselator.md), [](fsi-channel.md), [](petsc-reaction-diffusion.md), [](layout-driver.md) and
[](demo-b.md).

## For power users

[](power-users.md) covers builds for performance, execution spaces, reproducibility, scale and
diagnostics.
