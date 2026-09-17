# FUSE

**F**lexible, **U**nified, **S**calable **E**xchange — a C++20 library for coupling multiphysics
and multiscale simulations. Portable by construction, performant by measurement.

A coupled simulation is authored as a graph, compiled to an inspectable operator IR, scheduled with
exact rational time and an initialisation-time deadlock proof, and executed by an allocation-free
runtime on Kokkos — with collective failure reconciliation, conservative non-matching remap,
waveform and space-time coupling, quasi-Newton acceleration, and two C ABIs over one canonical
descriptor.

:::{note}
FUSE is under construction. Pages marked **planned** describe an interface that is specified and
scheduled but not yet built; pages describe only what exists unless they say otherwise. The
distinction is deliberate — see [](dev/testing.md#the-five-states).
:::

```{toctree}
:caption: User guide
:maxdepth: 2

user/index
user/build
user/quickstart
user/fluent-api
user/placement
user/geometry
user/combining-writers
user/convergence
user/c-abi
```

```{toctree}
:caption: Developer guide
:maxdepth: 2

dev/index
dev/invariants
dev/layering
dev/conventions
dev/testing
dev/rank-agreement
dev/debugging
dev/porting
dev/abi-design
dev/roadmap
```
