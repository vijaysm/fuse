:::{note}
**Historical document.** Preserved verbatim from `upstream/soma` (parent repository `coupe`,
commit `e52b981`, imported 2026-09-14 per `UPSTREAMS.lock`). Describes the design, build and
handoff state of the retired `soma::` library as it stood at that commit -- not FUSE's current
architecture. See [](../../../dev/soma-import.md) for what from this design was ported, superseded,
or remains open.
:::

# SOMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build SOMA, a C++20 multiphysics coupling library that represents a coupled simulation as a time-flowing network graph, lowered to an operator DAG and compiled to a flat execution plan.

**Architecture:** Three representations, two passes. `CouplingGraph` (authored, serializable) → `lower()` → `OperatorDAG` (SSA, time-typed, regions) → `schedule()` → `ExecutionPlan` (flat, immutable, allocation-free). The serialized graph is the ABI narrow waist shared by C++, Fortran, Python, and YAML front-ends.

**Tech Stack:** C++20, CMake ≥3.20, GoogleTest, MPICH, MOAB 5.6, PETSc 3.25, optional TempestRemap / LowFive / pybind11 / yaml-cpp.

**Spec:** `modules/coupler/docs/specs/2026-09-02-soma-design.md` — read it before starting. The plan argues from the spec; both travel together.

## Phase files

Tasks are split per phase so each file stays reviewable. Execute in order.

| File | Phase | Milestone |
|---|---|---|
| `soma-phase-0.md` | 0 | Skeleton, CMake, CI, vocabulary types |
| `soma-phase-1.md` | 1 | `CouplingGraph`, validation, YAML round-trip |
| `soma-phase-2.md` | 2 | `OperatorDAG`, lowering registry, golden-IR tests |
| `soma-phase-3.md` | 3 | Rational clocks, slab decomposition, scheduler, acyclicity |
| `soma-phase-4.md` | 4 | Runtime + serial backend — **first coupled run** |
| `soma-phase-5.md` | 5 | MOAB backend, remap, MPI transport |
| `soma-phase-6.md` | 6 | Waveforms, space-time projection, independent intervals |
| `soma-phase-7.md` | 7 | Iteration regions, accelerators — **Demo A complete** |
| `soma-phase-8.md` | 8 | C ABI, Fortran, Python, LowFive — **Demo B complete** |

---

## Global Constraints

Every task's requirements implicitly include this section.

**Language and build**
- C++20, `CMAKE_CXX_STANDARD 20`, `CXX_EXTENSIONS OFF`.
- CMake minimum 3.20. Installed: 4.3.2.
- Compiler: Apple clang 21. Warnings `-Wall -Wextra -Wpedantic`; `-Werror` in CI only.
- Every phase ends with `ctest --test-dir build --output-on-failure` fully green.

**Naming and layout**
- Namespace `soma::`. Nested: `soma::ir`, `soma::sched`, `soma::rt`, `soma::mesh`, `soma::remap`, `soma::transport`, `soma::num`, `soma::diag`.
- Headers `include/soma/<area>/<name>.hpp`; sources `src/<area>/<name>.cpp`.
- Types `PascalCase`; functions and variables `snake_case`; constants `kPascalCase`; private members trailing `_`.
- Header guards: `#pragma once`.
- No `using namespace` at file scope in headers.

**Dependency rules**
- **No third-party type may appear in any public header.** MOAB, PETSc, MPI, HDF5 types live behind pimpl or in `src/`. Violation fails review.
- The core (tiers A–D) builds with zero optional dependencies. `cmake -B build` with no flags must configure, build, and pass tests.
- Optional features gated by `SOMA_ENABLE_{MPI,MOAB,PETSC,KOKKOS,LOWFIVE,PYTHON,FORTRAN,TESTS}`, all `OFF` by default except `TESTS`.

**Error handling**
- No exceptions across the C ABI. Internally, `soma::Result<T>` (a `std::expected`-alike) for recoverable errors; `SOMA_ASSERT` for programmer errors.
- Validation errors accumulate into a `Diagnostics` list and are reported together, never one-at-a-time-abort.

**Verified environment (2026-09-02)**
- MPICH: `/opt/mpich`
- MOAB 5.6.0 + PETSc 3.25.99: `PKG_CONFIG_PATH=/opt/petsc/download-moab/lib/pkgconfig`
- iMOAB C API present in `libMOAB.dylib`
- GoogleTest 1.17.0: `/opt/homebrew/opt/googletest`
- DIY: `/opt/diy`
- **TempestRemap: ABSENT** (`MOAB_HAVE_TEMPEST` unset, 0 Tempest symbols). Phase 5 Task 5.0 addresses this.
- **LowFive: PRESENT BUT BROKEN.** `/opt/lowfive/lib/liblowfive.dylib` fails `dlopen` — requires `libhdf5.310.dylib`, but `/opt/hdf5` ships static archives only. Phase 8 Task 8.0 addresses this.
- yaml-cpp, pybind11: absent. Phases 1 and 8 fetch them via `FetchContent`.
- Python 3.13 is system default; LowFive's extension is built for 3.11 (`/opt/homebrew/bin/python3.11`).

**Build commands**

```bash
# core only, no optional deps
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
ctest --test-dir build --output-on-failure

# with MPI + MOAB
cmake -B build-full -DCMAKE_BUILD_TYPE=Debug \
      -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_MOAB=ON \
      -DCMAKE_PREFIX_PATH=/opt/petsc/download-moab \
      -DMPI_HOME=/opt/mpich
```

**Commit discipline**
- One commit per completed task, message `feat(soma): <task summary>` or `test(soma): …`.
- Never commit a red test suite.

**Git**
- `modules/coupler/` is not yet under version control. Task 0.1 initialises a repository rooted there.

---

## Task index

Tasks are numbered `<phase>.<n>` and are globally unique. Cross-references between phase files use these numbers.

- **Phase 0** — 0.1 repo+CMake skeleton · 0.2 `Result`/`Diagnostics` · 0.3 `Rational` · 0.4 `Units` · 0.5 `TimeLevel` · 0.6 identifiers and enums
- **Phase 1** — 1.1 `Region`/`FieldSpec` · 1.2 `Component` node · 1.3 `Channel` + builder · 1.4 `CouplingGraph` · 1.5 validation · 1.6 flat descriptor · 1.7 YAML round-trip
- **Phase 2** — 2.1 `OperatorDAG` types · 2.2 op registry · 2.3 arc time-typing · 2.4 lowering driver · 2.5 channel rules · 2.6 derived ops · 2.7 iteration regions · 2.8 golden-IR harness
- **Phase 3** — 3.1 `Clock` · 3.2 slab decomposition · 3.3 time-resolved expansion · 3.4 acyclicity proof · 3.5 buffer allocation · 3.6 `ExecutionPlan` · 3.7 lockstep fast path
- **Phase 4** — 4.1 `FieldRef` registration · 4.2 serial mesh backend · 4.3 copy transport · 4.4 kernels · 4.5 plan interpreter · 4.6 `Component` API · 4.7 **first coupled run**
- **Phase 5** — 5.0 rebuild MOAB with Tempest · 5.1 MOAB backend · 5.2 `RemapPlan` + Intx weights · 5.3 Tempest high-order · 5.4 MPI transport · 5.5 conservation tests
- **Phase 6** — 6.1 `Waveform` · 6.2 temporal bases · 6.3 space-time projection · 6.4 independent-interval scheduling · 6.5 temporal order tests
- **Phase 7** — 7.1 `Residual` + convergence · 7.2 checkpoint set derivation · 7.3 accelerators · 7.4 QR filtering · 7.5 **Demo A**
- **Phase 8** — 8.0 rebuild HDF5 shared + LowFive · 8.1 C ABI · 8.2 Fortran shim · 8.3 pybind11 · 8.4 LowFive transport · 8.5 **Demo B**
