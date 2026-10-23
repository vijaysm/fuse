# FUSE

[![CI](https://github.com/vijaysm/fuse/actions/workflows/ci.yml/badge.svg)](https://github.com/vijaysm/fuse/actions/workflows/ci.yml)

**F**lexible, **U**nified, **S**calable **E**xchange: a C++20 library for coupling multiphysics and
multiscale simulations.

You describe a coupled simulation as a graph of components, the fields they exchange and the
transfers between them. FUSE checks the graph, compiles it into an inspectable schedule, and runs it,
on one process or across many MPI ranks, in one executable or several.

- [CI](https://github.com/vijaysm/fuse/actions/workflows/ci.yml) · [Documentation](docs/index.md) ·
  [Contributing](CONTRIBUTING.md) · [License](LICENSE)

## Why FUSE

Coupling two solvers is easy to get almost right. The time steps do not line up, a flux is not
conserved across non-matching meshes, one rank fails while its peers wait in a collective, or a
result changes with the thread count. FUSE exists to make those problems visible, or impossible:

- **Describe once, check everything.** A coupling is a graph. `validate()` reports every defect in it
  in one pass, not the first one it trips over.
- **Exact time.** Time steps are rational numbers, so a schedule stays in step after millions of
  steps, and the scheduler can prove at start-up that the coupling cannot deadlock.
- **No surprises at run time.** The compiled plan can be printed, including every operator FUSE
  inserted that you did not write (unit conversion, time interpolation, remap).
- **Failures do not hang.** A rank that fails agrees the failure with its peers instead of leaving
  them waiting in a collective.
- **Portable and reproducible.** Kernels run on Kokkos execution spaces. Results can be made
  bit-for-bit identical across thread counts, and the build records exactly which dependency
  versions it used.
- **Bring your own physics.** Your solver stays yours: implement one small C++ interface, call FUSE
  from C or Fortran, drive it from Python, or let FUSE load your code as a plugin.

## Capabilities

**Describing a coupling**
- A fluent C++ API, a YAML front end, and a versioned binary descriptor shared by all language
  bindings.
- Components with independent time steps (multirate coupling), explicit placement on any set of MPI
  ranks, and coupler ranks that do the remapping.
- Unit checking, with automatic scale conversion between compatible units.
- Several sources writing one field (merging), lagged and extrapolated transfers.

**Moving data**
- Meshes: a serial structured mesh, a native distributed unstructured mesh with its own halo exchange
  and repartitioning, and MOAB.
- Conservative remapping between non-matching meshes (exact supermesh intersection, or MOAB with
  TempestRemap), consistent interpolation and its conservative transpose for fluid-structure
  interaction, and moving-least-squares remap for point clouds.
- Persisted transfer maps, so expensive weights are built once.
- Transports: in-process copy, MPI, and intercepted HDF5 (LowFive) for coupling unmodified codes.

**Iterating to convergence**
- Implicit coupling groups with fixed-point iteration, Aitken relaxation, interface quasi-Newton
  (IQN-ILS, IQN-IMVJ) and a Jacobian-free Newton-Krylov solver (native, with an optional PETSc
  substrate).
- Waveform iteration over coupling windows for components with incommensurate time steps, including
  subcycling.
- Resolution- and partition-independent convergence measures.
- In-memory rollback within an iteration, and restart to disk (HDF5) that can resume on a different
  number of ranks.

**Linear algebra**
- Row-distributed vectors and sparse matrices, preconditioned CG and GMRES(*m*), and an exact
  distributed direct solver for periodic banded systems. A derivative-checking toolkit (adjoint
  identity, Taylor remainder) for any operator that claims a Jacobian or a transpose.

**Language bindings and launch modes**
- C and Fortran participant interfaces (your code drives FUSE) and provider interfaces (FUSE drives
  your plugin), generated from one list so they cannot disagree.
- Python bindings (`pfuse`) over the model and runtime.
- A coupling split across several separately built executables, launched together under one
  `mpiexec` or started independently and joined over MPI.

Each page of the [documentation](docs/index.md) says what is covered by the automated tests and what
is not.

## Building

### Requirements

- CMake 3.24 or newer and a C++20 compiler (GCC and Clang are both used for development).
- A build tool such as Ninja or Make, and Git.
- Python 3 for the repository's static checks and to build the documentation.

Everything else is found on your system or, with `-DFUSE_ALLOW_FETCH=ON`, built from a pinned
version: Kokkos, fmt, GoogleTest, ArborX and, for the default linear-algebra backend, Eigen. yaml-cpp
is fetched when it is not installed. MPI, MOAB, PETSc, HDF5, a Fortran compiler and pybind11 are
optional.

### Quick start

```bash
git clone https://github.com/vijaysm/fuse.git
cd fuse

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DFUSE_ALLOW_FETCH=ON
cmake --build build -j
ctest --test-dir build --output-on-failure -LE bench
```

On a machine that already has Kokkos installed, leave `FUSE_ALLOW_FETCH` off and point CMake at it
with `-DCMAKE_PREFIX_PATH=/path/to/kokkos`. Installed copies are always preferred: a generic fetched
Kokkos should not silently replace one tuned for your machine. Every configure writes
`fuse-resolved-dependencies.txt` into the build directory, recording which dependencies were found
and which were fetched.

Run the smallest complete example:

```bash
./build/examples/fuse_example_mms_pair
./build/examples/fuse_example_mms_pair --dump    # print the compiled coupling without running it
```

### Choosing features

Features are added one dependency family at a time, so the minimal build stays small.

| Configure with | Adds |
|---|---|
| *(nothing)* | The compiler, scheduler and runtime, a complete single-process coupled run, serial and native meshes, YAML input |
| `-DFUSE_ENABLE_MPI=ON` | Distributed transport, multi-rank runs, multi-executable coupling |
| `-DFUSE_ENABLE_MOAB=ON` | MOAB mesh backend and conservative remap with TempestRemap (requires MPI) |
| `-DFUSE_ENABLE_PETSC=ON` | PETSc as a Newton solver substrate |
| `-DFUSE_ENABLE_LOWFIVE=ON` | Coupling unmodified HDF5 codes through intercepted I/O |
| `-DFUSE_ENABLE_HDF5=ON` | Restart to disk |
| `-DFUSE_ENABLE_PYTHON=ON` | The `pfuse` Python module (needs pybind11) |
| `-DFUSE_LINALG_BACKEND=builtin` | Replace Eigen with the built-in dense kernels (some examples need Eigen and are skipped) |
| `-DKokkos_ENABLE_OPENMP=ON` | Threaded kernels (when Kokkos is fetched) |

`CMakePresets.json` defines the same ladder as presets (`tier0` through `tier3`, `all`). Its hidden
`_machine` preset holds the install prefixes for one machine; edit that one entry for yours. The
[build guide](docs/user/build.md) covers every option, floating-point reproducibility, and how
dependencies are resolved.

### Documentation

The documentation is Sphinx with MyST, under [`docs/`](docs/):

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

Start with `docs/user/getting-started.md`, which takes you from a fresh clone to a running coupling.

## Planned

- `physics-kit`: an optional finite-element component library built on FUSE.
- Broader GPU support through Kokkos device backends, with performance and scaling measurements.
- A device-capable linear-algebra backend.
- Newton-Krylov iteration over coupling windows on more than one rank.
- Diagrams of a coupled setup generated from the graph and compiled plan.
- Cross-rank reductions in the field pipeline.

## Contributing and support

Bug reports and feature requests go to the
[issue tracker](https://github.com/vijaysm/fuse/issues). To contribute code or documentation, read
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD 3-Clause. © 2006–2026 Vijay Mahadevan. See [`LICENSE`](LICENSE). Third-party components are
listed in [`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md).
