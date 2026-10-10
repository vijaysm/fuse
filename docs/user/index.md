# User guide

[](glossary.md) defines every term used in this documentation and gives the literature term for each.

Start with [](build.md) for what you need installed — FUSE builds as a ladder of build
configurations, and the default one needs only a C++20 compiler, Kokkos and a small
linear-algebra backend. Then [](quickstart.md) for a complete two-component coupling and
[](api-overview.md) for a map of every public namespace, what each is for, and which page covers
it. [](fluent-api.md) covers the chaining idiom and its rules; [](placement.md) and [](layouts.md)
cover where a component runs and how to explore a layout from the command line; [](geometry.md)
covers the mesh backends (the native distributed mesh, and MOAB) and the remap ladder between
them, with [](transfer-maps.md) for persisting the weights a remap builds and
[](point-cloud-remap.md) for a MOAB-free remap over scattered points. [](combining-writers.md)
covers several sources writing one field, [](field-pipeline.md) the lazy chain for reading field
data, [](convergence.md) how an implicit coupling decides it is done, [](coupling-windows.md)
coupling two components that do not share a time grid, and [](interface-transfer.md) a
strongly-coupled interface between two differently discretised solvers.

[](c-abi.md) covers coupling an existing code through C, [](fortran.md) the Fortran side's own
page for the `ffuse` module's generated names and type mapping in full, and
[](multiple-executables.md) splitting a coupling across more than one separately built
executable. Those pages are about a foreign code driving FUSE; [](providers.md) is the other
direction — FUSE driving a plugin you write, in C or Fortran, named by path in the graph.
[](lowfive.md) is a fourth way to move a field: two unmodified codes exchanging it through
ordinary, intercepted HDF5 calls.
