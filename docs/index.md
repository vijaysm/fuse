# FUSE

**F**lexible, **U**nified, **S**calable **E**xchange — a C++20 library for coupling multiphysics
and multiscale simulations. Portable by construction, performant by measurement.

A coupled simulation is authored as a graph, compiled to an inspectable operator IR, scheduled with
exact rational time and an initialisation-time deadlock proof, and executed by an allocation-free
runtime on Kokkos — over a native distributed unstructured mesh or MOAB, with collective failure
reconciliation, conservative non-matching remap, waveform and space-time coupling, quasi-Newton
acceleration, distributed Krylov and direct linear solvers, two C ABIs (participant and provider, C
and Fortran) over one canonical serialised coupling specification, and a coupling split across more
than one separately built executable.

:::{note}
FUSE is under construction. Pages marked **planned** describe an interface that is specified and
scheduled but not yet built; pages describe only what exists unless they say otherwise. The
distinction is deliberate — see [](dev/testing.md#the-five-states).
:::

```{toctree}
:caption: User guide
:maxdepth: 2

user/index
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
dev/partitioning
dev/native-mesh
dev/native-halo
dev/search
dev/linalg-distributed
dev/linalg-krylov
dev/schur-slab-solver
dev/newton-krylov
dev/coupling-windows
dev/nodal-fields
dev/coupler-placement
dev/conformance
dev/roadmap
```
