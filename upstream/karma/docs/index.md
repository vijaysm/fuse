# KARMA Documentation

KARMA is a C++20 scientific-computing library for finite-element and mesh-free systems using PETSc, MPI, MOAB, and Eigen.

Use the scientific user guide to build KARMA and work with its current mesh, finite-element, physics, parallel, and checkpoint capabilities. Use the developer guide for architecture, correctness contracts, testing, and contribution workflow.

```{toctree}
:maxdepth: 2
:caption: Scientific User Guide

user-guide/README
user-guide/getting-started
user-guide/meshes-and-spaces
user-guide/physics-systems
user-guide/parallel-and-checkpoints
user-guide/field-transfer
user-guide/gray-scott-coupler
```

```{toctree}
:maxdepth: 2
:caption: Developer Guide

developer-guide/README
developer-guide/architecture
developer-guide/plugin-design-review
developer-guide/pyramid-quadrature-cost
developer-guide/semantic-model-soma-mapping
developer-guide/abi-soma-gap-table
developer-guide/soma-port-baseline
developer-guide/build-and-test
developer-guide/distributed-correctness
developer-guide/contributing
HOST-ACTIONS
```

```{toctree}
:maxdepth: 1
:caption: Project

roadmap
```

The historical handoff log is intentionally excluded from published navigation because it mixes superseded status entries with current evidence.
