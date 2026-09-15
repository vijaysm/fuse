# FEM Field Transfer

KARMA provides nonmatching-mesh FEM transfer through stable field and DoF identities. The production path is native KARMA code: MOAB's `AdaptiveKDTree` supplies local candidate search, while KARMA performs inverse mapping, basis evaluation, sampling, and projection. ArborX/Kokkos is detected independently and remains optional. The MOAB Coupler library is not linked into `KARMA::karma`.

## Operators

- Nodal cG, dG, and SE targets use point sampling at target interpolation points.
- Modal dG and SE targets use target-element L2 projection.
- FV/P0 targets are represented as one-function element-local L2 projections.
- cG targets use global target mass assembly and a checked PETSc solve.

Source fields are evaluated by locating physical points, applying the source geometry inverse map, resolving stable DoF keys, and evaluating the registered `KARMABasis`.

## Coverage

Exterior points and genuine source holes are reported as `Uncovered`. They are not backend failures. Uncovered point targets require an explicit component-aware fill policy. Boundary points use deterministic lowest-stable-element-ID trace selection by default; ambiguous traces can instead be rejected.

Target quadrature with partial source coverage is explicit. The default projection path rejects partial elements rather than silently renormalizing or claiming local conservation.

## Usage

The basic flow is:

```cpp
KARMA::KARMADistributedPointLocator locator(source_mesh);
locator.setup();
KARMA::KARMAFEMFieldEvaluator source_values(source_field);

const auto sampled = KARMA::KARMAFEMTransfer::sample(
    locator, source_values, target_points, source_field.components(), &fill_policy);
```

For an element-local modal, dG, SE, or FV/P0 target, construct the target basis and call `project_element()`. For a scalar cG target variable, initialize a `KARMASystem`, create a compatible target vector, and call `project_cg()`. PETSc finite-element target commits are staged through a duplicate vector and published only after validation and assembly. MOAB tag commits and PETSc coefficient commits are staged and tested. Projection factors persist as stable-key sparse cross-operator `B` rows; scalable owner-directed loading/redistribution remains TODO. Owner-side source evaluation uses batched stable-ID request/response packets, and partitioned 2/4/5-rank source/target fixtures pass exact serial-reference comparison.

## Verification

`karma_fem_transfer_convergence_serial`, `karma_fem_transfer_convergence_mpi_2`, `karma_fem_transfer_convergence_mpi_4`, and `karma_fem_transfer_convergence_mpi_5` project the manufactured field `u=x^2+y^2` to FV/P0 targets over 1, 2, 4, and 8 cells per direction. The observed L2 rates are exactly 2 in the verified Release stack. The MPI tests use independent rank-local fixtures and deterministic global reductions; `karma_distributed_point_locator_mpi` separately tests stable-ID distributed location. Independent source/target H5M partition fixtures are compared against a serial reference by centroid coordinate identity. The comparison passes bit-for-bit at 2/4/5 ranks. Batched target-element projection performs one collective locator/evaluator phase before local projection assembly.

## Guarantees

Transfer diagnostics distinguish full-domain target integrals, covered-region integrals, and fill or extension contributions. Matching L2, sampled nonmatching projection, and Coupler normalization are not automatically conservative. Conservative supermesh remapping remains unavailable until a validated volumetric remapper is integrated.

Reusable sparse interpolation maps use stable field keys and finite sorted rows. Projection-factor persistence stores sparse cross-operator rows as stable-key `B` rows and records a target-mass signature. The target mass matrix is intentionally recomputed from the current target mesh, basis, quadrature, and geometry after signature validation. Application is: load and validate distribution, form `b = B u_s` from current stable-key source values, recompute `M_t`, solve `M_t u_t = b`, and commit transactionally. Before application, every target row must be owned by the current target rank and every source column must be available in the current source-local key set.
