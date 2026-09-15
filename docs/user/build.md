# Building FUSE

FUSE builds as a **ladder**. Each tier adds one dependency family and unlocks one set of
capabilities, so you pay for what you use and the minimum build stays small enough to compile in
seconds.

| Tier | Adds | Unlocks |
|---|---|---|
| **0** (default) | C++20 compiler, Kokkos, and one of KokkosKernels or Eigen3 | The whole compiler and scheduler, a complete single-process coupled run, copy transport, serial mesh, both C ABIs, the Fortran shim, YAML input |
| **1** | MPI | Distributed transport, rendezvous matching, epochs across ranks, MPMD |
| **2** | MOAB (with TempestRemap) | Unstructured mesh backend and conservative non-matching remap |
| **3** | PETSc | The implicit/monolithic solver plugin and the optional `physics-kit` component |

Orthogonal options, each independent of the tier: LowFive/HDF5 (transport by HDF5 interception),
pybind11 (Python module), a Fortran compiler, and a GPU backend for Kokkos.

A tier is a **prefix**: enabling a higher tier without its predecessors is a configuration error,
not a silent downgrade.

```bash
cmake -B build -DFUSE_LINALG_BACKEND=eigen          # tier 0
cmake -B build -DFUSE_ENABLE_MPI=ON                 # tier 1
cmake -B build -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_MOAB=ON   # tier 2
```

## Choosing the linear-algebra backend

FUSE needs a small, fixed set of dense and sparse operations: QR with column filtering (for
quasi-Newton acceleration), small least-squares solves (temporal projection), sparse matrix–vector
products, and batched small solves. These sit behind one facade with two interchangeable backends.

`eigen`
: Header-only and the simplest build. Host execution only.

`kokkoskernels`
: Device-capable, and the right choice when you build with a GPU backend. Must match your Kokkos
  version.

Neither backend's types appear in any FUSE public header, so your code never inherits the choice.

## Dependency versions

The exact pins the build fetches when a dependency is not already installed:

```{literalinclude} ../../cmake/FuseDependencies.cmake
:language: cmake
:start-after: "# ---------------------------------------------------------------- Tier 0 (always)"
:end-before: "# fmt — the ONLY output formatter"
```

Every configured build writes `fuse-resolved-dependencies.txt` into its binary directory recording
what was actually resolved, because a build that cannot say which versions it used cannot reproduce
its own result.
