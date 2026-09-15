# KARMA handoff

This checkout contains the reviewable native coupling foundation on branch
`codex/usable-plugin-foundation`. It is intended for in-process MPI
components that share one communicator. The original `master` worktree is
left untouched.

## Implemented contracts

- `SemanticVersion` parses and serializes strict `major.minor.patch` values.
  Version domains are independent for the semantic model, graph schema, C
  ABI, plugin configuration, transfer maps, component state, and graph
  checkpoints. Compatibility decisions carry a reason and reject unknown
  required capabilities.
- `FieldEntryKey` and `DofKey` have a dependency-free canonical byte encoding
  and rank-stable hash. The encoding is used by the compliance and compatibility
  paths without pulling PETSc, MOAB, or MPI into the identity header.
- The semantic model records units as SI dimensions, exact rational scale and
  offset, shape, location, continuity, temporal interpretation, role,
  ownership, access, and update behavior.
- The connection checker reports every incompatible property in a deterministic
  order and infers only explicitly authorized operator chains. A caller gets a
  refusal instead of an implicit conversion.
- The candidate C ABI is a separate shared library with C11 headers, hidden
  visibility, an explicit export list, version negotiation, capability
  negotiation, component lifecycle calls, field exchange, checkpoint/restore,
  exact rational advancement, and named diagnostics.
- The component registry refuses missing named implementations and ambiguous
  matches. Native reference components are driven through the same C function
  table as future plugin implementations, and the compliance harness checks
  lifecycle order, buffer ownership, determinism, and capability honesty.
- Field publication uses a transaction-local buffer and a collective
  prepare/publish/reconcile protocol. PETSc commits are limited to entries
  owned by the calling rank, so off-rank write order is never implicit.
- The existing native coupling path provides a reciprocal two-component graph,
  Lie/Strang execution, persistent compiled FEM connections, nonmatching-mesh
  transfer, rollback, safe-point restart, and mass-weighted fixed-point
  iteration with constant or safeguarded Aitken relaxation.

## Verification recorded

- `ctest -R 'karma_(semantic_version|coupling_compatibility_oracle)'` passes
  serially and with two MPI ranks (3/3 tests).
- The same focused compatibility set passes with AddressSanitizer and
  UndefinedBehaviorSanitizer enabled (3/3 tests, leak detection disabled for
  the host runtime).
- The coupling regression matrix recorded in `build-evidence` passes 36/36,
  including operator-split, fixed-point, restart, and MPI cases.
- A fresh host run of the current worktree passes the complete registered suite
  (425/425) in parallel in 187.51 seconds.
- The Sphinx documentation build completes with `-E -W --keep-going`.
- `git diff --check` is clean for the current worktree changes.
- Mutation checks for major-version inversion, minor-version inversion, and
  ignored required capability bits all fail as expected when the guarded code
  is deliberately altered.

## Boundaries and risks

The current runtime does not launch independent executables, provide a general
fan-in/fan-out graph, load shared libraries dynamically, expose Fortran or
Python adapters, execute an ArborX backend, perform conservative supermesh
remapping, provide Anderson acceleration, or write an atomic graph-wide
durable checkpoint. The PETSc-backed transient adapter is not registered
through the runtime because its zero-copy transactional binding is not yet
available.

The verified dependency stack uses PETSc double-real scalars and 32-bit
indices. Single precision and complex scalar variants are rejected by the
active ABI probe. Distorted prism and pyramid quadrature still need direct
validation, and the imported MOAB target currently contributes an `-O0` flag
after the project release flags. These are tracked as technical risks rather
than hidden assumptions.

## Reproduction

From this checkout, configure a Release build with the local PETSc, MOAB, MPI,
and Eigen installations, build, and run the focused tests before running the
full CTest suite. MPI tests need a host environment in which `mpirun`, PETSc,
MOAB, and the partitioning utility are visible in `PATH`.
