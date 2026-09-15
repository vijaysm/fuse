# SOMA

SOMA is a C++20 multiphysics coupling library. It represents a coupled
simulation as a **network graph that flows in time**: nodes are components,
edges carry the semantics of how fields are accumulated in time, merged in
space, extrapolated to resolve nonlinearity, and iterated to reduce numerical
error. Independent per-component coupling intervals are the default; lockstep
is the degenerate case.

**Status: all 8 planned phases complete, all 3 mandatory review gates closed.**
751/751 tests pass in Debug, Release, and ASan+UBSan; 771/771 under the
`lowfive` preset; 87/87 Python.

Post-completion hardening is in progress against a ranked list of 33 known
gaps — see **[`docs/HANDOFF.md`](docs/HANDOFF.md)**, which is the authoritative
status document. Tier 1 (7 real correctness/capability gaps) and Tier 2
(9 robustness items) are being worked now; Tier 3 (engineering and packaging,
including the absence of CI and of `install()` rules) follows a full
validation pass. Nothing in that list blocks the completed plan, but several
items matter before SOMA is used on a real problem — notably that the C ABI
performs no graph validation, and that conservative remap is serial only.

## Architecture

Three representations, two passes:

```
CouplingGraph  --lower()-->  OperatorDAG (SSA)  --schedule()-->  ExecutionPlan
 (authored,                  (time-typed,         (flat,
  serializable)               regions)             allocation-free)
```

The serialized `CouplingGraph` descriptor is the ABI narrow waist shared by
C++, Fortran, Python, and YAML front-ends. The C ABI (`include/soma/soma_c.h`)
is a separate, deliberately thinner, caller-driven entry point — it does not
mirror the C++ `Runner`/`ExecutionPlan` path. See its header comment ("WHAT
THIS ABI DOES NOT DO") before assuming parity between the two.

## Read these first

| Document | What it is |
|---|---|
| [`docs/specs/2026-09-02-soma-design.md`](docs/specs/2026-09-02-soma-design.md) | The vision and binding authority: architecture, data model, numerics, space-time remap, scheduling. |
| [`docs/plans/2026-09-02-soma-implementation.md`](docs/plans/2026-09-02-soma-implementation.md) | Master plan: global constraints, task index, phase files. |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Current status, what's flagged-but-unfixed, how to resume. |
| [`docs/BUILDING.md`](docs/BUILDING.md) | Build commands, all four configurations, why each exists. |
| `.superpowers/sdd/2026-09-02-soma-implementation/progress.md` | The full ledger — every task, every finding, every ruling. Git-ignored but authoritative. |

## Build and test

```bash
cmake --preset debug   && cmake --build --preset debug   -j && ctest --preset debug
cmake --preset release && cmake --build --preset release -j && ctest --preset release
cmake --preset asan    && cmake --build --preset asan    -j && ctest --preset asan
```

All three must pass — see `docs/BUILDING.md` for why Debug-only is not
sufficient (assert-only invariants vanish under `-DNDEBUG`; some UB only
manifests under the sanitizers). The core builds with **zero optional
dependencies**. Optional features (`SOMA_ENABLE_{MPI,MOAB,PETSC,KOKKOS,
LOWFIVE,PYTHON,FORTRAN}`) default OFF; enable per-feature to exercise that
integration. A fourth preset, `lowfive`, builds the HDF5-VOL transport
(Task 8.4) against `/opt/lowfive-shared` + `/opt/hdf5-shared`.

## Layout

```
include/soma/   public headers: base, graph, ir, sched, runtime, mesh,
                remap, transport, num, capi (soma_c.h)
src/            implementation, mirrored by area
fortran/        ISO_C_BINDING module over the C ABI
python/         pybind11 module with NumPy field views
test/unit/      GoogleTest suite (C++, C, Fortran, Python)
examples/       dn_heat (Demo A), lowfive_pair (Demo B), mms_pair
docs/           spec, plans, build notes, this handoff
```

## Demos

- **Demo A** (`examples/dn_heat`) — implicit Dirichlet-Neumann coupling of a
  heat-conduction pair with accelerated (IQN-ILS) iteration, measured spatial
  and temporal order ≈2. Sabotage-gated: a permanent CTest leg fails if the
  acceleration or the region logic is disabled.
- **Demo B** (`examples/lowfive_pair`) — two unmodified HDF5-writing
  executables coupled purely by intercepting their I/O through the LowFive
  HDF5-VOL transport; zero bytes touch disk. Also sabotage-gated.
