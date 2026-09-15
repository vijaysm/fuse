# Distributed Correctness

## Collective Failure Discipline

No rank may throw while peers continue toward a collective. A distributed operation should:

1. Validate and prepare local inputs without mutating caller-visible output.
2. Catch local exceptions.
3. Collectively determine whether any rank failed.
4. Choose the lowest failing rank and broadcast one deterministic diagnostic.
5. Proceed into later MPI/PETSc collectives only after global success.

This applies to callbacks, allocation, signatures, ghost updates, PETSc assembly, solver setup, and checkpoint publication. Empty and zero-owner ranks still participate.

## Identity and Ownership

Conventional MOAB `GLOBAL_ID` is authoritative for distributed identity. KARMA private global/local tags are internal metadata. Shared and ghost copies must agree on conventional IDs.

Stable DoF keys encode variable, component, entity, canonical slot, and identity. Owners receive contiguous ranges. cG shared entities canonicalize orientation; dG and SE remain element-local.

## Assembly

Use `dof_map().assembly_elements()` rather than all visible elements. Refresh ghost vectors before local reads and use scoped PETSc local-form/array guards. Interior faces need canonical identities so each is processed exactly once while assembling both coupling directions.

Scatter helpers do not assemble. The collective caller performs vector or matrix assembly only after every rank has staged values successfully.

## Transactional Mutation

Prepare contributions in temporary C++ storage or PETSc objects. Commit to caller output only after validation, callbacks, scatter, and assembly succeed collectively. Checkpoint restore validates complete metadata and values before changing its destination vector.

## FEM Transfer

Point location returns stable source element identities, reference coordinates, residuals, and coverage status. Remote MOAB handles must not be dereferenced on a rank that did not create them. Current distributed locator tests route through source bounding boxes and reconcile stable IDs. The registered 2/4/5-rank convergence executables use complete rank-local fixtures, so they verify replicated numerical rates rather than partitioned error integration. Independently partitioned source/target H5M fixtures separately pass exact serial-reference comparison keyed by canonical target-centroid bits at 2/4/5 ranks, while batched owner-side source packets use stable source identities.

FEM source ghost refresh is a collective setup operation. It must not occur inside rank-asymmetric point or element loops. Local-form and array borrows are scoped with RAII, and target projection solves stage RHS and output until KSP convergence and residual checks succeed.

## Test Matrix

Distributed changes should include a serial reference, 2/3/4 ranks when ownership matters, missing-ghost and malformed-identity failures, asymmetric rank-local failures with finite timeouts, empty/zero-owner ranks where feasible, and stable-key rather than local-handle comparison. FEM transfer additionally requires manufactured-solution convergence rates, coverage classification, trace-policy tests, and parallel/serial agreement.
