# FUSE

**F**lexible, **U**nified, **S**calable **E**xchange — a C++20 library for coupling multiphysics
and multiscale simulations. Portable by construction, performant by measurement.

You describe a coupled simulation as a graph of components, the fields they exchange and the
transfers between them. FUSE checks the graph, compiles it into an inspectable schedule with exact
rational time and a start-up proof that it cannot deadlock, and runs it on Kokkos: on one process or
many MPI ranks, in one executable or several.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} New to FUSE
Build it, run a coupling, learn to read what it prints.

[](user/getting-started.md), then [](user/quickstart.md) and [](user/concepts.md).
:::

:::{grid-item-card} Doing a task
Find a recipe, then the page and example behind it.

[](user/cookbook.md), [](user/examples.md), [](user/troubleshooting.md).
:::

:::{grid-item-card} Running at scale
Placement, reproducibility, launch modes, diagnostics.

[](user/power-users.md), [](user/placement.md), [](user/build.md).
:::
::::

```{toctree}
:caption: User guide
:maxdepth: 2

user/index
user/getting-started
user/concepts
user/cookbook
user/power-users
user/troubleshooting
user/glossary
user/build
user/quickstart
user/api-overview
user/fluent-api
user/placement
user/layouts
user/geometry
user/transfer-maps
user/point-cloud-remap
user/combining-writers
user/field-pipeline
user/examples
user/mms-pair
user/dn-heat
user/petsc-reaction-diffusion
user/restart
user/convergence
user/coupling-windows
user/interface-transfer
user/gray-scott
user/brusselator
user/fsi-channel
user/layout-driver
user/demo-b
user/c-abi
user/fortran
user/python
user/providers
user/lowfive
user/multiple-executables
```

```{toctree}
:caption: Developer guide
:maxdepth: 2

dev/index
dev/invariants
dev/layering
dev/conventions
dev/operators
dev/testing
dev/rank-agreement
dev/debugging
dev/porting
dev/abi-design
dev/restart
dev/partitioning
dev/native-mesh
dev/native-halo
dev/moab-intersection
dev/search
dev/linalg-distributed
dev/linalg-krylov
dev/schur-slab-solver
dev/derivative-gate
dev/newton-krylov
dev/coupling-windows
dev/nodal-fields
dev/coupler-placement
dev/conformance
dev/python-surface
dev/runner-device-dispatch
dev/device-resident-arcs
dev/roadmap
dev/soma-import
```

```{toctree}
:caption: History
:maxdepth: 1

history/soma/index
```
