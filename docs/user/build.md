# Building FUSE

FUSE builds as a **ladder**. Each tier adds one dependency family and unlocks one set of
capabilities, so you pay for what you use and the minimum build stays small enough to compile in
seconds.

| Tier | Adds | Unlocks |
|---|---|---|
| **0** (default) | C++20 compiler, Kokkos, and one of `builtin` (no dependency) or Eigen3 | The whole compiler and scheduler, a complete single-process coupled run, copy transport, serial mesh, YAML input |
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

`builtin`
: The coupler's own column-pivoted Householder QR with rank truncation, no dependency. This is also
  the oracle the other backend is checked against, because the recorded iteration counts were
  produced with it.

`eigen`
: A second opinion from a widely used implementation. Host execution only.

Neither backend's types appear in any FUSE public header, so your code never inherits the choice. A
device-capable (KokkosKernels) backend was designed but is not implemented — `FUSE_LINALG_BACKEND`
accepts only `builtin` or `eigen` today (REVIEW.md R-24).

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
# One command per tier. The presets carry the prefixes for this machine in
# the hidden `_machine` preset -- edit that one entry on another machine.
cmake --preset tier0 && cmake --build --preset tier0 && ctest --preset tier0
cmake --preset tier1 && cmake --build --preset tier1 && ctest --preset tier1   # + MPI
cmake --preset tier2 && cmake --build --preset tier2 && ctest --preset tier2   # + MOAB
cmake --list-presets                                                          # all of them

# Or by hand, which is what the presets expand to:
cmake -S . -B build -DCMAKE_PREFIX_PATH="/opt/kokkos;/opt/kokkos-kernels;/opt/homebrew"
```

`-DFUSE_ALLOW_FETCH=ON` opts into building a pinned copy of any Tier 0 dependency that is not
installed, for a machine that has none.

**yaml-cpp is the one exception**: it is fetched at a pinned tag by default (an installed copy is
still found first). The installed-first rule exists because Kokkos, MOAB and PETSc carry machine and
stack tuning a generic build would silently replace; a YAML parser carries none of that — no MPI, no
index-size choice, no tuned kernels — so building it costs nothing and keeps the configuration
front-end available everywhere. It is linked `PRIVATE`, so it never reaches a consumer's include path. Tiers 2 and 3 (MOAB, PETSc) are **found only, never fetched**:
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

## When a dependency's own config is stale

A dependency installed from source records the link line of the toolchain it
was built with. MOAB's `MOABConfig.cmake` carries the gfortran runtime
directory of *its* gcc; upgrade gcc and that directory is gone, and the failure
would arrive as `ld: library 'emutls_w' not found` after a full build. FUSE
checks every `-L` and `-rpath` directory in that config at configure time and
refuses with the directories named. The remedy is to rebuild MOAB (and PETSc,
whose `.pc` carries the same line) against the current compiler; there is
nothing to substitute, since those libraries belong to the compiler that
produced them.
