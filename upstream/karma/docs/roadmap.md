# Roadmap

KARMA is an MPI-first coupling library for scientific applications. The
current implementation is a reviewable native foundation: it composes two
in-process components, moves fields between nonmatching meshes, and exposes
the contracts needed to grow the runtime without making the existing transfer
or restart formats ambiguous.

## Current capabilities

- MOAB-backed 1D, 2D, and 3D meshes with stable distributed degree-of-freedom
  identities and partition-aware PETSc assembly.
- Native KD-tree location, stable-key finite-element evaluation, nodal
  sampling, local dG/SE/FV projection, and scalar cG global L2 projection.
- Transaction-local field updates, collective failure reconciliation, and
  macro-step rollback for the native two-component operator-split graph.
- Lie and Strang schedules, persistent compiled connections, graph
  introspection and signatures, safe-point restart, and mass-weighted
  fixed-point iteration with constant or safeguarded Aitken relaxation.
- A storage-neutral semantic model, strict version negotiation across seven
  independent domains, canonical field-entry encoding, and reason-coded
  connection compatibility decisions.
- A candidate C ABI and static component registry. Native reference components
  run through the same registry and ABI table used by future plugin
  implementations.
- Scalar transient Crank-Nicolson reaction-diffusion, Gray-Scott kinetics,
  manufactured-solution checks, and independently partitioned trajectory
  comparisons at the tested MPI rank counts.

## Capability increments

The following additions are the next engineering targets. Each must preserve
the existing contracts and add direct behavioral tests before it is presented
as supported:

1. General graph topology with fan-in/fan-out, explicit ownership, and
   communicator-safe execution for in-process components.
2. Dynamic plugin discovery and language adapters for Fortran and Python,
   including one ABI call prefix exercised by native and foreign components.
3. Canonical configuration records with deterministic serialization and
   schema negotiation.
4. A conservative volumetric remapping backend with local conservation
   diagnostics and an explicit dependency boundary.
5. Adaptive mesh epochs, transfer-map invalidation, and checkpoint metadata
   that identifies the complete geometry and topology state.
6. Partitioned transfer and graph execution that replaces root-heavy metadata
   exchanges with owner-directed routing and bounded memory.
7. Wider physics coverage, dependency variants, reproducible CI, and
   strong/weak scaling measurements on representative machines.

## Verification requirements

Every capability increment is expected to include:

- serial and MPI tests with rank-local failures reconciled before later
  collectives;
- deterministic identifiers and serialized records checked across rank
  counts and process layouts;
- analytic reproduction, conservation, residual, or convergence evidence
  appropriate to the operator;
- Release and sanitizer coverage, warning-clean documentation, and a focused
  review of ownership, lifetime, numerical stability, and memory movement.

## Explicit boundaries

The present release does not provide independently launched MPMD programs,
general graph partitioning, executable ArborX/Kokkos location, conservative
supermesh remapping, Anderson acceleration, or an atomic graph-wide durable
checkpoint. The PETSc-backed transient adapter still uses the established
native path until a zero-copy transactional field binding is available.

Legacy libMesh, PETScExt, TinyXPath, and obsolete PETSc surfaces remain
isolated or excluded from the active target. Removing them, adding adaptive
constraints, and enabling additional PETSc scalar/index variants require
concrete dependency and user requirements together with new verification.
