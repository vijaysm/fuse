# FUSE

**F**lexible, **U**nified, **S**calable **E**xchange — a C++20 library for coupling multiphysics
and multiscale simulations. Portable by construction, performant by measurement.

You describe a coupled simulation as a graph of components, the fields they exchange and the
transfers between them. FUSE checks the graph, compiles it into an inspectable schedule with exact
rational time and a start-up proof that it cannot deadlock, and runs it on Kokkos: on one process or
many MPI ranks, in one executable or several.

```{toctree}
:caption: User guide
:maxdepth: 2

user/index
user/getting-started
user/concepts
user/cookbook
user/quickstart
user/build
user/graph-programming
user/yaml
user/component-interface
user/runtime
user/ir
user/drivers
user/environment
user/power-users
user/troubleshooting
user/glossary
user/api-overview
user/fluent-api
user/placement
user/layouts
user/meshes
user/geometry
user/transfer-maps
user/point-cloud-remap
user/combining-writers
user/field-pipeline
user/examples
user/portfolio
user/mms-pair
user/dn-heat
user/petsc-reaction-diffusion
user/restart
user/convergence
user/coupling-windows
user/multiscale
user/surrogates
user/linear-algebra
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
user/mixing-languages
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
dev/physics-kit-disposition
dev/integrators-design
dev/field-semantics-matrix
dev/roadmap
dev/soma-import
```

```{toctree}
:caption: History
:maxdepth: 1

history/soma/index
```
