# Building FUSE

FUSE builds as a **ladder** of build configurations, named `tier0` through `tier3` in the CMake
presets. Each build configuration adds one dependency family and unlocks one set of capabilities,
so you pay for what you use and the minimum build stays small enough to compile in seconds.

| Build configuration | Adds | Unlocks |
|---|---|---|
| **`tier0`** (default) | C++20 compiler, Kokkos, and one of `builtin` (no dependency) or Eigen3 | The whole compiler and scheduler, a complete single-process coupled run, copy transport, serial mesh, YAML input |
| **`tier1`** | MPI | Distributed transport, rendezvous matching, synchronisation points across ranks, MPMD |
| **`tier2`** | MOAB (with TempestRemap) | Unstructured mesh backend and conservative non-matching remap |
| **`tier3`** | PETSc | Configures today; the implicit/monolithic solver plugin and the optional `physics-kit` component are not yet written (S7/S8) |

Orthogonal options, each independent of the build configuration: LowFive/HDF5 (transport by HDF5
interception), pybind11 (Python module), a Fortran compiler, a GPU backend for Kokkos, and the
floating-point contraction policy ([below](#floating-point-reproducibility-vs-speed)).

A build configuration is a **prefix**: enabling a higher one without its predecessors is a
configuration error, not a silent downgrade.

```bash
cmake -B build -DFUSE_LINALG_BACKEND=eigen          # tier0: base build configuration
cmake -B build -DFUSE_ENABLE_MPI=ON                 # tier1: adds MPI
cmake -B build -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_MOAB=ON   # tier2: adds MOAB
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
accepts only `builtin` or `eigen` today (REVIEW.md R-24; the default is `eigen`).

**`builtin` now configures a full build (review-2026-10-11 BC-1 closed D26).**
`fuse::linalg` itself — including the distributed Krylov solvers
([](../dev/linalg-distributed.md), [](../dev/linalg-krylov.md)) — builds clean under `builtin`,
as it always did. The tests and examples that need Eigen's sparse and dense direct
factorisations directly (every caller of `SchurSlabSolver`, plus the FSI/Gray-Scott/Brusselator
examples and their verification rows, all `Eigen::SparseLU` or `Eigen::SimplicialLDLT` and with no
`builtin` implementation) are gated behind `FUSE_LINALG_BACKEND STREQUAL "eigen"`
(`test/CMakeLists.txt`, `examples/CMakeLists.txt`) instead of linking `Eigen3::Eigen`
unconditionally, so `-DFUSE_LINALG_BACKEND=builtin` configures and builds the rest of the suite;
a configure-time status message lists exactly which rows it skipped. `SchurSlabSolver::factor()`
itself still refuses with `E-LINALG-SCHUR-BACKEND` under `builtin` (there is no built-in direct
factorisation to substitute) — `test/linalg/schur_slab_unavailable_test.cpp` fires that refusal
in exactly this configuration (KNOWN_ISSUES.md D26).

## Floating-point reproducibility vs. speed

`a*b + c` can be rounded once, as a single fused multiply-add instruction, or twice — multiply,
round, then add. Compilers choose the once-rounded (contracted) form **by default at every
optimization level** (confirmed on this project's reference compiler, Apple clang: contraction is
the default at `-O0` just as much as at `-O2`/`-O3` — it is not an optimization). What actually
differs between a Debug and a Release build of the same reduction loop is code SHAPE, not rounding
policy as such: an optimizing build's auto-vectorizer commonly splits a runtime-sized loop into a
vectorized main body (whose multiply and add are always separate instructions, regardless of the
contraction flag) plus a short scalar remainder (whose rounding DOES follow the flag), while an
unoptimized build and a compile-time-sized reference loop are not vectorized or are fully unrolled,
respectively — so a Release build can fail a test that a Debug build of the same code passes, not
because Release "turned contraction on," but because the two build types take structurally different
paths through the same arithmetic ([](../dev/testing.md#floating-point-contraction-policy) has the
measured instruction counts). `FUSE_FP_REPRODUCIBLE` (default **ON**) resolves this with a
build-wide choice instead of leaving it to the compiler and its optimizer:

`ON` (default)
: `-ffp-contract=off` for every C, C++ and Fortran target, in every build type — including Debug,
  whose own arithmetic is contracted by default too, so this is a genuinely new numerical
  configuration there as well, not a return to Debug's historical bit pattern. The cost is twofold:
  somewhat slower per-operation arithmetic (a separately-rounded multiply-then-add instead of one
  fused instruction), and — measured, not assumed — individual operations become a correctly-rounded
  single-rounding operation's worth *less* accurate, which can, for an ill-conditioned iterative
  method, shift an iteration count by one step or push a convergence check past its tolerance (see
  [](../dev/testing.md#a-measured-cost-beyond-speed-iteration-sensitive-tests-restated-as-what-they-actually-show)
  for the two tests this was measured to affect, in both build types).

`OFF`
: The compiler's own default contraction policy, which an optimized build will generally choose for
  speed. Bit-for-bit comparisons that assumed no contraction are then checked against a derived
  forward-error bound instead (the same tests as above); every pre-existing numerical regression
  test keeps its original tolerance.

Which matters more — reproducing a reference build's exact bits, or the small per-operation accuracy
and speed difference — depends on the problem and what the result is being used for; neither choice
is correct for every use case. The `tier0-release` preset keeps the default (`ON`); a matching
`tier0-release-fast` preset sets `FUSE_FP_REPRODUCIBLE=OFF`:

```bash
cmake --preset tier0-release        # FUSE_FP_REPRODUCIBLE=ON  (default)
cmake --preset tier0-release-fast   # FUSE_FP_REPRODUCIBLE=OFF
```

A CUDA device build has the equivalent choice under a different flag (`nvcc`'s `-fmad=false` for the
reproducible policy, matching `-ffp-contract=off`) — not wired up by this option, since it governs
only the host-side C/C++/Fortran compilers; left as a note for whichever build adds device support.

## Dependency versions

The exact pins the build fetches when a dependency is not already installed:

```{literalinclude} ../../cmake/FuseDependencies.cmake
:language: cmake
:start-after: "# ---------------------------------------------------------------- Tier 0 (always)"
:end-before: "# fmt -- the ONLY output formatter"
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
# One command per build configuration. The presets carry the prefixes for this machine in
# the hidden `_machine` preset -- edit that one entry on another machine.
cmake --preset tier0 && cmake --build --preset tier0 && ctest --preset tier0
cmake --preset tier1 && cmake --build --preset tier1 && ctest --preset tier1   # + MPI
cmake --preset tier2 && cmake --build --preset tier2 && ctest --preset tier2   # + MOAB
cmake --list-presets                                                          # all of them

# Or by hand, which is what the presets expand to:
cmake -S . -B build -DCMAKE_PREFIX_PATH="/opt/kokkos;/opt/kokkos-kernels;/opt/homebrew"
```

`-DFUSE_ALLOW_FETCH=ON` opts into building a pinned copy of any base-build-configuration dependency that is not
installed, for a machine that has none.

**yaml-cpp is the one exception**: it is fetched at a pinned tag by default (an installed copy is
still found first). The installed-first rule exists because Kokkos, MOAB and PETSc carry machine and
stack tuning a generic build would silently replace; a YAML parser carries none of that — no MPI, no
index-size choice, no tuned kernels — so building it costs nothing and keeps the configuration
front-end available everywhere. It is linked `PRIVATE`, so it never reaches a consumer's include path. The `tier2`/`tier3` build configurations (MOAB, PETSc) are **found only, never fetched**:
they are large builds whose configuration choices — index size, debug or optimized, which MPI — must
match the rest of the stack, so a fetched default would be the wrong library in a way no version
number reveals. Extending the fetch machinery to them is a later decision, deliberately open.

Every configured build writes `fuse-resolved-dependencies.txt` recording, per dependency, whether it
was **found** (with version and config directory) or **fetched**.

### The accept/connect launch mode is probed, not assumed

With `FUSE_ENABLE_MPI=ON`, configuring also runs `tools/probe-mpi-dynamic.sh`: it compiles a small
program, launches it twice as two separate jobs — one accepting, one connecting — and performs the
same `MPI_Open_port`/`MPI_Comm_accept`/`MPI_Comm_connect`/`MPI_Intercomm_merge` sequence
`fuse::transport::connect()` performs for a genuinely separate-executable coupling
([](multiple-executables.md)). The verdict sets the `FUSE_ENABLE_CONNECT_ACCEPT` option, which can
be overridden directly:

```bash
cmake -S . -B build -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_CONNECT_ACCEPT=OFF
```

When the probe measures the mode broken on this MPI build (MPICH's `ch3:nemesis` device aborts
merging an intercommunicator between two same-node jobs it has no shared memory for —
KNOWN_ISSUES.md D23), the accept/connect test rows are not registered and a `Connect`-mode
component is refused at `connect()` with `E-CONNECT-ACCEPT-UNAVAILABLE` rather than aborting inside
MPI. When the probe cannot run at all (a login node where `mpiexec` needs an allocation, the
ordinary case on a cluster), the mode is left enabled and its own rows report for themselves — the
probe only refuses what it has actually measured broken. The launcher-style MPMD launch mode needs
only `MPI_APPNUM`, which every MPI provides, and is never gated by this probe.

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
rejects the configuration, naming the directories. The remedy is to rebuild MOAB (and PETSc,
whose `.pc` carries the same line) against the current compiler; there is
nothing to substitute, since those libraries belong to the compiler that
produced them.
