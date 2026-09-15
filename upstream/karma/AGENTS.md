# Repository Instructions for Coding Agents

## Scope and Authority

- Work only inside this repository unless the user explicitly authorizes another path.
- Inspect current headers, active CMake source lists, tests, and recent history before changing behavior.
- Treat current code and tests as authoritative. `docs/HANDOFF.md` and the migration plan contain historical entries.
- Do not claim a legacy header is supported merely because it is installed. Active production sources are explicitly listed in `src/CMakeLists.txt`.
- Keep changes minimal and do not reformat unrelated legacy code.

## Protect User Work

- The worktree may be dirty. Never revert, delete, stage, or modify unrelated user changes.
- Preserve untracked `bin/`, `project/`, `src/physics/common/impls/karma_physics_base_compute_error.cpp`, `src/physics/common/impls/karma_physics_project_solution.cpp`, and `src/utils/test_for_exception.cpp` unless explicitly requested.
- Use out-of-tree `build*` directories. Do not commit generated files, binaries, dependency caches, test outputs, or local virtual environments.
- Do not commit, push, amend, or create a pull request unless explicitly requested.

## Supported Toolchain

- CMake 3.20+, C++20, PETSc, MPI, MOAB, parallel HDF5, Eigen3, LAPACK, and pkg-config.
- The verified local prefix is `/opt/petsc/download-moab`; Eigen is under `/opt/eigen3`.
- The supported PETSc ABI is double-precision real only. Do not claim single, complex, or 64-bit-index coverage without a compatible stack and fresh verification.
- MOAB and Eigen are effectively required by the active library.
- `mbpart` is needed to register the complete partitioned integration-test matrix.

## Canonical Commands

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH='/opt/petsc/download-moab;/opt/eigen3'
cmake --build build -j4
ctest --test-dir build --output-on-failure -j1
git diff --check
```

Prefer CTest for MPI tests because it carries the configured launcher, fixtures, resource locks, rank counts, and timeouts. Use `ctest --test-dir build -N` to discover the local test count.

Build documentation with:

```bash
python3 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Source and Test Registration

- Public headers live under `include/`; implementations mirror them under `src/`.
- Add every new production `.C` file explicitly to `KARMA_SOURCES` in `src/CMakeLists.txt`. Never replace the explicit list with a glob.
- Keep the configure-time `sources built : N of M` census honest.
- Active tests are plain C++ programs under `test/`. Add an executable, link `KARMA::karma`, and register it explicitly in `test/CMakeLists.txt`.
- Give every MPI test a finite timeout. Use short timeouts for asymmetric-rank deadlock regressions.
- Use CTest fixtures for generated or partitioned inputs and `RESOURCE_LOCK karma_moab_io_files` for checkpoint/file tests.

## Distributed Correctness Contracts

- Never let one rank throw while peers proceed into MPI or PETSc collectives. Validate locally, catch failures, reconcile collectively, broadcast one deterministic rank-qualified diagnostic, and continue only after global success.
- Empty partitions and zero-owned-DoF ranks are valid collective participants.
- Use `system.dof_map().assembly_elements()` for once-only assembly; visible mesh elements include ghosts.
- Refresh PETSc ghost values before reading local forms. Scope local forms and array borrows with existing RAII guards.
- Scatter helpers stage values but do not perform collective matrix/vector assembly; callers own assembly.
- Stage caller-visible output and commit only after all validation, callbacks, scatters, and collective assembly succeed.
- Preserve conventional MOAB `GLOBAL_ID` as authoritative distributed identity. KARMA private ID tags have separate internal roles.
- Stable DoF ordering is variable-major, component-major, then basis-minor. Shared cG DoFs use canonical subentity identities; dG and SE DoFs remain element-local.

## Numerical Correctness Contracts

- Check units, signs, dimensions, ownership ranges, communicator compatibility, boundary cases, and finite values before mutating outputs.
- Validate callback outputs and finite-difference probes. Existing derivative fallback uses `cbrt(epsilon) * max(1, abs(value))`.
- Preserve scale-aware checks for lumped masses and diagonals; do not introduce arbitrary absolute thresholds.
- Preserve `KARMAFEBase` output object and buffer address stability across `reinit()` calls. Assembly code keeps references to `JxW`, `phi`, and `dphi` outside element loops.
- Neumann means prescribed outward flux with a positive load contribution. Robin uses `a*(k grad(u).n) + b*u = c`; `a == 0` is invalid.
- Numerical tests should check invariants, finiteness, solver convergence reason, independent residuals, scaling behavior, and convergence rates where applicable.
- Mutation-test critical correctness properties when feasible.

## Verification Expectations

- Bug fixes and behavior changes require regression tests.
- Distributed ownership or partitioning changes should cover serial and 2/3/4 ranks when feasible.
- Collective validation changes require an asymmetric-rank failure test with a finite timeout.
- Scientific kernels require focused tests plus relevant MMS or integration tests.
- Before reporting completion, run focused tests, the complete applicable suite, `git diff --check`, and inspect status.
- Report exact commands and outcomes. If a dependency-dependent test was not registered or run, say so.

## Documentation Maintenance

- Keep `README` concise enough for onboarding and link detail into `docs/user-guide/` and `docs/developer-guide/`.
- Author Sphinx pages in MyST Markdown. Add new pages to a `{toctree}`.
- Describe only active, tested capabilities as supported. Label proposals and excluded legacy APIs clearly.
- Update `docs/roadmap.md` when phase scope or entry/exit criteria change.
