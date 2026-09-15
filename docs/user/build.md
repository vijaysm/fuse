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


## How dependencies are resolved

**What is installed is what is used.** Every dependency is resolved with `find_package` against
`CMAKE_PREFIX_PATH`, and a missing one is a configure error naming what to install. Nothing is
downloaded silently: the prefixes on an HPC machine are built for that machine, and a fetched generic
copy would quietly replace them — any timing taken afterwards would be measuring the wrong library.

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="/opt/kokkos;/opt/kokkos-kernels;/opt/homebrew"
```

`-DFUSE_ALLOW_FETCH=ON` opts into building a pinned copy of any Tier 0 dependency that is not
installed, for a machine that has none. Tiers 2 and 3 (MOAB, PETSc) are **found only, never fetched**:
they are large builds whose configuration choices — index size, debug or optimized, which MPI — must
match the rest of the stack, so a fetched default would be the wrong library in a way no version
number reveals. Extending the fetch machinery to them is a later decision, deliberately open.

Every configured build writes `fuse-resolved-dependencies.txt` recording, per dependency, whether it
was **found** (with version and config directory) or **fetched**.

### Two Kokkos installs on one machine

If a second Kokkos exists in another prefix on the search path, its headers can shadow the chosen
one's — and a header set that declares a backend the linked library does not have fails to compile,
or worse, compiles against the wrong configuration. FUSE prepends the resolved Kokkos include
directory on its base target so the chosen install wins, and
`Config.KokkosHeadersMatchTheResolvedLibrary` compares `KOKKOS_VERSION` from the headers with the
version CMake resolved, so a slip fails a test instead of producing a mystery.
