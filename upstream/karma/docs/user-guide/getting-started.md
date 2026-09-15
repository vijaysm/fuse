# Getting Started

## Dependency Stack

KARMA requires a mutually compatible PETSc, MPI, MOAB, parallel HDF5, Eigen3, and LAPACK stack. PETSc is discovered through pkg-config and the supported ABI is double-precision real. A PETSc build configured with `--download-moab` is the simplest way to keep PETSc, MOAB, HDF5, and MPI compatible.

The verified development layout is:

```text
/opt/petsc/download-moab  PETSc, MOAB, HDF5, and MPI-facing metadata
/opt/eigen3               Eigen3 headers
```

## Configure and Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH='/opt/petsc/download-moab;/opt/eigen3'
cmake --build build -j4
```

The configure summary reports dependency locations, PETSc scalar mode, and the number of active `.C` sources. Treat an unexpected source-count reduction as a configuration error.

## Verify and Install

```bash
ctest --test-dir build --output-on-failure -j1
ctest --test-dir build -N
cmake --install build --prefix /desired/install/prefix
```

Some tests are available only when MPI and MOAB's `mbpart` utility are found. Installation places `libkarma` and public headers under `include/karma`. Because the project is in transition, include only APIs documented as active; not every historical installed header has a linked implementation.

## Runtime Lifecycle

KARMA objects wrap PETSc and MPI resources. Initialize MPI/PETSc before constructing meshes, systems, vectors, matrices, finite-element evaluators, or physics systems, and destroy those objects before `PetscFinalize()`. The programs under `test/` are the most current executable lifecycle examples.
