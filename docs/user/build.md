# Building FUSE

This page covers everything about getting FUSE built: the short path, the optional features, every
configure option, and what to do when a dependency is not where CMake expects it. If you only want a
working build, read [Quick build](#quick-build) and stop.

## Requirements

- CMake 3.24 or newer and a C++20 compiler. GCC and Clang (including Apple Clang) are used for
  development.
- A build tool (Ninja or Make) and Git.
- Python 3, for the repository's static checks and for building this documentation.

Everything else is either installed on your machine or built from a pinned version when you ask for
that ([below](#how-dependencies-are-resolved)).

(quick-build)=
## Quick build

```bash
git clone https://github.com/vijaysm/fuse.git
cd fuse

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFUSE_ALLOW_FETCH=ON
cmake --build build -j
ctest --test-dir build --output-on-failure -LE bench
```

`-DFUSE_ALLOW_FETCH=ON` builds a pinned copy of Kokkos, fmt, GoogleTest, ArborX and Eigen when they
are not installed, so a clean machine needs nothing else. The first configure downloads them from
GitHub and GitLab. Then run the smallest complete coupling:

```bash
./build/examples/fuse_example_mms_pair
./build/examples/fuse_example_mms_pair --dump
```

[](getting-started.md) continues from here.

## Build configurations

FUSE builds as a ladder of configurations. Each rung adds one dependency family and unlocks one set
of capabilities, so you pay only for what you use and the smallest build compiles quickly.

| Configuration | Add | What it unlocks |
|---|---|---|
| **Base** (default) | C++20 compiler, Kokkos, and a linear-algebra backend (`builtin` or Eigen) | The whole compiler and scheduler, a complete single-process coupled run, copy transport, serial and native meshes, YAML input |
| **+ MPI** | `-DFUSE_ENABLE_MPI=ON` | Distributed transport, multi-rank runs, rank agreement, multi-executable coupling |
| **+ MOAB** | `-DFUSE_ENABLE_MOAB=ON` (requires MPI) | The MOAB mesh backend and conservative remap with TempestRemap |
| **+ PETSc** | `-DFUSE_ENABLE_PETSC=ON` | PETSc as a Newton solver substrate; PETSc is found through `pkg-config` |

MOAB needs MPI, and asking for MOAB without it is a configure error rather than a silent downgrade.
PETSc can be added to any build.

```bash
cmake -B build -DFUSE_ALLOW_FETCH=ON                                   # base
cmake -B build -DFUSE_ENABLE_MPI=ON                                    # + MPI
cmake -B build -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_MOAB=ON              # + MOAB
```

The options below are independent of the ladder:

| Option | Default | Effect |
|---|---|---|
| `FUSE_ENABLE_LOWFIVE` | OFF | Transport by intercepted HDF5 calls ([](lowfive.md)) |
| `FUSE_ENABLE_HDF5` | OFF | Restart to disk ([](restart.md)) |
| `FUSE_ENABLE_PYTHON` | OFF | The `pfuse` Python module, needs pybind11 ([](python.md)) |
| `FUSE_ENABLE_FORTRAN` | ON | The Fortran binding `ffuse` ([](fortran.md)); turned off with a message when no Fortran compiler is found |
| `FUSE_ENABLE_TESTS` | ON | Build the test suite |
| `FUSE_ENABLE_EXCEPTIONS` | ON | OFF makes every refusal report and abort instead of throwing |
| `FUSE_ENABLE_CUSPARSE` | OFF | cuSOLVER/cuSPARSE sparse LU for the FSI and Gray-Scott examples; needs the Kokkos CUDA backend |
| `FUSE_LINALG_BACKEND` | `eigen` | `eigen` or `builtin` ([below](#choosing-the-linear-algebra-backend)) |
| `FUSE_FP_REPRODUCIBLE` | ON | `-ffp-contract=off` on every target ([below](#floating-point-reproducibility-vs-speed)) |
| `FUSE_ALLOW_FETCH` | OFF | Build a pinned copy of any base dependency that is not installed |
| `FUSE_FETCHED_KOKKOS_SHARED` | ON | Build a fetched Kokkos as shared libraries |
| `FUSE_ENABLE_CONNECT_ACCEPT` | probed | Independently launched multi-executable mode ([below](#the-acceptconnect-launch-mode-is-probed)) |
| `FUSE_MPI_DYNAMIC_PROBE_RUNS`, `FUSE_MPI_DYNAMIC_PROBE_TIMEOUT` | 6, 20 | How hard that probe tries |
| `Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_SERIAL`, ... | Kokkos' own | Execution spaces, when Kokkos is fetched; an installed Kokkos has already decided |

### Presets

`CMakePresets.json` expresses the ladder as presets: `tier0` (base), `tier0-release`,
`tier0-no-exceptions`, `tier0-release-fast`, `tier1` (+ MPI), `tier2` (+ MOAB), `tier3` (+ PETSc) and
`all`. Each of `tier1` to `tier3` includes the ones before it.

```bash
cmake --list-presets
cmake --preset tier0-release && cmake --build --preset tier0-release && ctest --preset tier0-release
cmake --preset tier1         && cmake --build --preset tier1         && ctest --preset tier1
```

All presets inherit one hidden preset, `_machine`, which holds the install prefixes
(`CMAKE_PREFIX_PATH`, `MOAB_DIR`, and so on) for one machine. Edit that one entry for yours, or leave
the presets alone and pass the same variables on the command line. Presets that need an installed
Kokkos expect it under `CMAKE_PREFIX_PATH`; they do not fetch.

Two test presets, `tier0-quick` and `all-quick`, skip the slowest tests (see
[Running the tests](#running-the-tests)).

## Choosing the linear-algebra backend

FUSE needs a small, fixed set of dense and sparse operations: QR with column filtering (for
quasi-Newton acceleration), small least-squares solves, sparse matrix-vector products and batched
small solves. They sit behind one facade with two interchangeable backends.

`builtin`
: The library's own column-pivoted Householder QR with rank truncation. No dependency. It is also the
  reference the other backend is checked against, because the recorded iteration counts were produced
  with it.

`eigen`
: Eigen 3 as a second implementation. Host execution only. This is the default.

Neither backend's types appear in any FUSE public header, so your code never inherits the choice.

`-DFUSE_LINALG_BACKEND=builtin` configures and builds the whole library. Examples and tests that use
Eigen's sparse and dense direct factorisations directly (the Schur-complement solver, and the
Gray-Scott, Brusselator and FSI examples) need `eigen` and are left out of a `builtin` build; the
configure step prints which ones it skipped. `SchurSlabSolver::factor()` refuses with
`E-LINALG-SCHUR-BACKEND` under `builtin`, because there is no built-in direct factorisation.

## Floating-point reproducibility vs. speed

`a*b + c` can be rounded once, as one fused multiply-add instruction, or twice. Compilers choose the
once-rounded form by default at every optimisation level, and an optimising build's vectoriser then
mixes fused and unfused rounding inside one loop. The same source can therefore give slightly
different bits in Debug and Release. `FUSE_FP_REPRODUCIBLE` replaces that accident with a build-wide
choice:

`ON` (default)
: `-ffp-contract=off` for every C, C++ and Fortran target in every build type. Release reproduces
  Debug's bits. The cost is slower per-operation arithmetic and, for an ill-conditioned iterative
  method, occasionally one more iteration or a convergence check that lands on the other side of its
  tolerance.

`OFF`
: The compiler's own contraction policy, normally faster. The few tests that compare bits exactly
  then check a derived forward-error bound instead.

Neither is right for every use. Choose `ON` when you must reproduce a reference result bit for bit
and `OFF` when throughput matters more.

```bash
cmake --preset tier0-release        # FUSE_FP_REPRODUCIBLE=ON
cmake --preset tier0-release-fast   # FUSE_FP_REPRODUCIBLE=OFF
```

A CUDA device build has the equivalent choice in `nvcc`'s `-fmad=false`, which this option does not
set. It governs the host compilers only.

## How dependencies are resolved

**What is installed is what is used.** Every dependency is looked up with `find_package` against
`CMAKE_PREFIX_PATH`. A missing one is a configure error that names what to install, unless you pass
`-DFUSE_ALLOW_FETCH=ON`. Nothing is downloaded silently, because the libraries on an HPC machine are
built for that machine and a generic fetched copy would quietly replace them, so any timing taken
afterwards would measure the wrong library.

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="/opt/kokkos;/opt/hdf5"     # installed prefixes
cmake -S . -B build -DFUSE_ALLOW_FETCH=ON                           # fetch what is missing
```

Dependency pins, used when something is fetched:

```{literalinclude} ../../cmake/FuseDependencies.cmake
:language: cmake
:start-after: "# ---------------------------------------------------------------- Tier 0 (always)"
:end-before: "# fmt -- the ONLY output formatter"
```

Every configure writes `fuse-resolved-dependencies.txt` into the build directory, recording for each
dependency whether it was found (with version and location) or fetched. Keep it with any result you
publish: a build that cannot say which versions it used cannot reproduce its own output.

yaml-cpp and r3d are fetched at pinned versions by default (an installed yaml-cpp is still found
first). Neither carries machine-specific tuning, and yaml-cpp is linked privately, so it never reaches
your include path. MOAB and PETSc are found only, never fetched: they are large builds whose index
size, debug or optimised mode and MPI must match the rest of your stack.

### Two Kokkos installs on one machine

If a second Kokkos lives elsewhere on the search path, its headers can shadow the one FUSE chose, and
a header set that declares a backend the linked library lacks fails to compile or, worse, compiles
against the wrong configuration. FUSE puts the chosen Kokkos include directory first, and a test
(`Config.KokkosHeadersMatchTheResolvedLibrary`) compares the header version with the one CMake
resolved, so a mix-up fails a test instead of producing a mystery.

### When a dependency's own configuration is stale

A library installed from source records the link line of the toolchain that built it. MOAB's
`MOABConfig.cmake` carries the gfortran runtime directory of *its* gcc; upgrade gcc and that
directory is gone, and the failure would otherwise appear as `ld: library 'emutls_w' not found` after
a full build. FUSE checks every `-L` and `-rpath` directory in that configuration at configure time
and rejects it, naming the missing directories. The fix is to rebuild MOAB (and PETSc, whose `.pc`
file carries the same link line) with the current compiler.

### The accept/connect launch mode is probed

A coupling split across separately started programs ([](multiple-executables.md)) can join them with
MPI's `MPI_Open_port` / `MPI_Comm_accept` / `MPI_Comm_connect`. Some MPI builds cannot do that
reliably, so with `FUSE_ENABLE_MPI=ON` the configure step runs `tools/probe-mpi-dynamic.sh`: it
compiles a small program and launches it as two separate jobs, one accepting and one connecting, for
every rank layout the accept-mode tests use, several times each. The result sets
`FUSE_ENABLE_CONNECT_ACCEPT`, which you can override:

```bash
cmake -S . -B build -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_CONNECT_ACCEPT=OFF
cmake -S . -B build -DFUSE_ENABLE_MPI=ON -DFUSE_MPI_DYNAMIC_PROBE_RUNS=3 -DFUSE_MPI_DYNAMIC_PROBE_TIMEOUT=30
```

- A layout that did not pass every run still registers its accept-mode tests, but disabled and
  labelled `accept-unreliable-on-this-mpi`. Run them anyway with
  `ctest --test-dir build -L accept-unreliable-on-this-mpi`.
- If the probe fails even at one rank per side, no accept-mode test is registered, and a component
  launched in `Connect` mode is refused with `E-CONNECT-ACCEPT-UNAVAILABLE` rather than aborting
  inside MPI. Use the MPMD launch mode, or an MPI that supports it.
- If the probe cannot run at all (typically a login node where `mpiexec` needs an allocation), the
  mode stays enabled and every test reports for itself.

The launcher-style MPMD mode needs only `MPI_APPNUM`, which every MPI provides, and is never gated by
this probe.

## Running the tests

```bash
ctest --test-dir build --output-on-failure            # everything registered
ctest --test-dir build -LE "slow|bench"               # skip the slowest tests and the benchmarks
ctest --test-dir build -L example                     # the example programs
ctest --test-dir build -R mms_pair                    # by name
```

Tests carry labels, so you can select a group:

| Label | Selects |
|---|---|
| `example` | The example programs and their `--dump` and thread-count legs |
| `omp` | Each test at one OpenMP thread and at every core, compared for identical results |
| `slow` | Tests that take more than about ten seconds; a full run still includes them |
| `mpi`, `parallel` | Multi-rank tests, run under `mpiexec` at several rank counts |
| `moab`, `lowfive` | Tests that need that dependency |
| `throws` | Tests that assert an exception is raised; excluded when `FUSE_ENABLE_EXCEPTIONS=OFF` |
| `bench` | Benchmarks |
| `device` | The device-space twin of each runner test, present only with a device Kokkos backend |
| `<layer>` | One layer's tests, for example `schedule` or `backend` |

Without exceptions, run `ctest -LE "throws|bench"`.

## Using FUSE from your own project

Install it and find the package. Every layer is an imported target under `fuse::`.

```bash
cmake --install build --prefix /path/to/prefix
```

```cmake
find_package(fuse-coupler CONFIG REQUIRED)
add_executable(my_coupling main.cpp)
target_link_libraries(my_coupling PRIVATE fuse::runtime fuse::backend)
```

Link only the layers you use: `fuse::model` to describe a coupling, `fuse::runtime` to run one,
`fuse::backend` for meshes and remap, `fuse::abi` for the C interface. The optional physics-kit
component is requested by name, `find_package(fuse-coupler CONFIG REQUIRED COMPONENTS physics_kit)`,
and linked as `fuse::physics_kit`; the request fails at configure time against an install built with
`FUSE_ENABLE_PHYSICS_KIT=OFF`. Your project must find the same
Kokkos and fmt that FUSE was built against; the package configuration does this for you when both
ship a CMake config. MPI and MOAB are required by a consumer only when FUSE was built with them, and a
consumer of a MOAB build needs MOAB at the same path.

## See also

- [](getting-started.md): from clone to a running coupling.
- [](power-users.md): choosing build options for performance and scale.
- [](troubleshooting.md): configure and run-time problems.
