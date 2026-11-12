# physics-kit

The optional finite-element component: KARMA's FEM core, physics modules (advection-reaction,
diffusion-reaction, Gray-Scott), FEM transfer and the Runge-Kutta/Rosenbrock integrator
family, ported into `fuse::` following [](integrators-design.md)'s design for the temporal
integrators. The finite-element core, DOF map, algebraic system, element kernels and the linear and
nonlinear study layers exist; the physics modules and the integrators are described by their design
page.

## What it is, and what it is not

physics-kit is a **separate optional component**, not one of the `fuse::` layers
[](layering.md) enumerates. It depends on FUSE -- the `fuse::physics_kit` target links
`fuse::runtime`, `fuse::backend`, `fuse::num` and the `fuse::linalg` facade
(`physics-kit/CMakeLists.txt`) -- and FUSE never depends back on it: nothing under `include/fuse/`,
`src/`, `python/` or `examples/` may include a `fuse_physics/` header or link `fuse::physics_kit`.
That direction is what keeps FUSE adoptable by a code that already has its own discretisation
([](layering.md)'s own wording). The namespace is `fuse::physics`; headers live under
`physics-kit/include/fuse_physics/`.

## Tier and how to enable it

physics-kit is **tier 0**: it needs nothing beyond what `fuse::runtime`, `fuse::backend`,
`fuse::num` and `fuse::linalg` already need at tier 0 (Kokkos, fmt), so `FUSE_ENABLE_PHYSICS_KIT`
(`cmake/FuseDependencies.cmake`) defaults **ON** -- unlike `FUSE_ENABLE_PYTHON`/`FUSE_ENABLE_FORTRAN`,
which default OFF because they add a real third-party dependency physics-kit does not. PETSc is
**optional**, read through the existing `FUSE_HAVE_PETSC` define (`fuse/base/config.hpp`) with no
new `find_package(PETSc)` call of its own: a physics-kit source guards PETSc-dependent code with
`#if FUSE_HAVE_PETSC`, the same pattern `examples/dn_heat/main.cpp` and
`examples/petsc_reaction_diffusion/main.cpp` already use, and `physics-kit/CMakeLists.txt` links
`fuse::petsc` only when `FUSE_ENABLE_PETSC` is on.

```
cmake -S . -B build -DFUSE_ENABLE_PHYSICS_KIT=ON    # default; needs nothing beyond tier 0
cmake -S . -B build -DFUSE_ENABLE_PHYSICS_KIT=OFF   # skip the component entirely
```

## The gates

Two static checks keep the boundary and the naming honest:

- **Layering** (`tools/check-layering.py`'s `check_physics_kit_boundary`): refuses a
  `#include ".../fuse_physics/..."` or a `fuse::physics_kit` link anywhere under `include/fuse/`,
  `src/`, `python/` or `examples/`. physics-kit/ itself is exempt -- it is the one place allowed to
  name any `fuse::` layer, by construction. The public-header third-party rule also applies to
  `physics-kit/include/fuse_physics/`: none of PETSc, MPI, MOAB, HDF5, Eigen or Kokkos-Kernels may be
  named in a header a consumer compiles, and of Kokkos only `Kokkos_Core.hpp`.
- **Naming** (`tools/check-physics-kit-naming.py`): refuses a `KARMA*` type prefix, a `karma_` file
  prefix or a `.C` extension anywhere under `physics-kit/` -- the three rules [](porting.md)'s
  naming map states for code arriving from `upstream/karma`, applied here because physics-kit is
  outside the four directories `tools/check-layering.py`'s own naming check already covers.

`tools/check-invariants.py` also scans `physics-kit/`. The test directory has its own
unregistered-test audit (`physics-kit/test/CMakeLists.txt`), because `test/CMakeLists.txt`'s audit
globs `test/*/*_test.cpp` and does not see it.

## Installation and the `physics_kit` component

`fuse_physics_kit` ships in its own export set, `fuse-physics-kit-targets`, written beside the main
`fuse-coupler-targets` in `lib/cmake/fuse-coupler/` (the main set is closed inside `src/CMakeLists.txt`
before the root adds `physics-kit/`, so it cannot be joined). The package configuration includes the
second set only when the install was configured with `FUSE_ENABLE_PHYSICS_KIT`, and exposes it as a
`find_package` component:

```cmake
find_package(fuse-coupler CONFIG REQUIRED COMPONENTS physics_kit)
target_link_libraries(app PRIVATE fuse::physics_kit)
```

Against an install built without physics-kit, `COMPONENTS physics_kit` fails at configure time through
`check_required_components`, and `OPTIONAL_COMPONENTS physics_kit` reports
`fuse-coupler_physics_kit_FOUND` false; the target is never left undefined. The headers install under
`include/fuse_physics/`. Both directions are exercised by `install.consumer_find_package`
(`test/install_consumer/`): with physics-kit on, a second consumer links the component and runs; with it
off, the consumer asks optionally and refuses if the component appears.

## The finite-element core

`fuse_physics::fe` (`physics-kit/include/fuse_physics/fe/`):

- `GeometryVariant`/`ReferenceTopology`: reference-cell shape functions, node tables and side tables
  for Edge, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Prism and Pyramid at Lagrange order 0, 1
  or 2 (the complete tensor/simplex family -- Quad9/Tet10/Hex27, Prism6/15/18, Pyramid5/13/14). The
  serendipity `Quad8` exists only as the one quadrilateral side of Prism15 and Pyramid13, never as a
  stand-alone basis family. Prism6's sides are 3 quadrilateral and 2 triangular, Pyramid5's are 4
  triangular and 1 quadrilateral. `edge_count`/`edge_vertices` expose the edge-to-corner tables used to
  place order-2 edge-midpoint nodes.
- `Quadrature`: native generation, described below.
- `Basis`/`BasisRegistry`: a type-erased Lagrange-nodal basis (Concept/Model pattern), with
  `OrientationTransform`/`apply_orientation`/`apply_orientation_inverse`
  (`EdgeOrientation`/`TriangleOrientation`/`QuadrilateralOrientation`, `local_to_canonical_vertices`)
  for DOF re-signing across a shared edge or face.
- `GeometryMap`: the isoparametric map, Jacobian, inverse map and side geometry, including the pyramid's
  taper condition (its own non-box reference cell), per-family sampling bounds in `validate()`,
  `inverse_map`'s apex fast path and the `reference_outward_normal` table.
- `FEBase`: the mesh-bound evaluator (`phi`/`dphi`/`JxW`/`xyz`/normals at a set of quadrature points).
  Face quadrature is selected per side (`quadrature_for_side()`): a quadrilateral side uses
  `quadrature_`, a triangular one `triangle_quadrature_`; both rules share a point count because they
  are built to the same degree.
- `DofMap`, `AlgebraicSystem` and the element kernels, described in the sections below.

`fuse_physics::spatial` (`physics-kit/include/fuse_physics/spatial/`):

- `Domain`: a box-shaped domain descriptor.
- `Mesh`: a native, tier-0 topology surface with variable-arity, per-element-tagged connectivity,
  ownership and global-id accessors (`is_vertex_owned`/`is_element_owned`/`owned_vertex_count`/
  `owned_element_count`/`vertex_global_id`/`element_global_id`/`vertex_owner`) and edge accessors
  (`edge_vertices`/`canonical_edge_vertex_ids`). `generate_2d_box` and a 3-D Hex8 box generator build
  structured meshes; `from_mesh_view` binds a `fuse::MeshView` (ownership, global ids and cell-to-vertex
  connectivity through `MeshView::cell_vertices` and `MeshView::owner`). physics-kit reaches MOAB only
  through `fuse::backend` ([](layering.md)). Vertices and cells are numbered in two separate id spaces,
  as `MeshView` does.
- `extract_boundary_faces` and `mesh_utils::extract_edge_at_coordinate` (the 1-D boundary of a 2-D mesh
  at a constant coordinate).

`numerics/algebra` is not adapted into a physics-kit type: every file in that directory is superseded by
`fuse::linalg`'s `DistVector`/`DistSparseMatrix`/dense least-squares facade (see `UPSTREAMS.lock`'s
`[karma]` entries).

### Thin physics-kit types

- `fuse::physics::Point`/`Vector3`/`Matrix3` (`physics-kit/include/fuse_physics/geometry.hpp`): KARMA's
  own are Eigen fixed-size types, and Eigen may never appear in a physics-kit public header
  (`tools/check-layering.py`'s `FORBIDDEN_PUBLIC`). `fuse::linalg`'s public header names nothing
  third-party at all and its facade is sized for the distributed coupled-iteration problem, not a
  per-element 1x1..3x3 array. A plain-struct `Point`/`Matrix3` with the handful of dense operations the
  geometry maps need (3x3 determinant and inverse, symmetric-eigenvalue bounds by the closed-form
  Smith/Kopp method) fills that gap.
- `fuse::physics::spatial::Mesh`: `fuse::MeshView` is remap-oriented, not FE-assembly-oriented, so the
  FE-facing topology surface is native.

### Native quadrature, not PETSc-backed

KARMA's `KARMAQuadrature` calls PETSc's `PetscDT*` functions unconditionally. physics-kit is tier 0
(PETSc optional), so `fuse_physics::fe::Quadrature` generates Gauss-Legendre/Gauss-Jacobi nodes and
weights natively via the Golub-Welsch eigenvalue construction (Golub & Welsch, 1969). The triangle and
tetrahedron use a collapsed (Duffy) conical product (Stroud, 1971; Karniadakis & Sherwin, 2005,
section 2.3) with one collapse (triangle) or two (tetrahedron). The prism is a triangle-base
Gauss-Jacobi(1) rule times an independent Gauss-Legendre line, which gives a hybrid exactness
guarantee stronger than a plain total-degree claim because the two axes are Jacobian-decoupled. The
pyramid is a single-collapse Gauss-Jacobi(alpha=2) rule: `n = (d+2)/2` points per axis gives `2n-1 >= d`,
and the collapsed monomial `x^i y^j z^k` has z-degree `i+j+k` against the `(1-z)^2` weight, so exactness
holds by construction.

The pyramid's rational collapsed-coordinate basis (`pyramid_candidates`/
`pyramid_nodal_coefficients`) uses a native Gauss-Jordan inversion with a derived condition-number
refusal threshold in place of KARMA's column-pivoted QR: the pyramid's candidate-term count equals its
node count exactly at orders 1 and 2, so the pivoted selection reduces to a plain inversion. The
distorted-pyramid accuracy has been measured for a straight-edged pyramid (volume exact to round-off;
a non-polynomial integrand such as `exp(x+y+z)` converges from 1.96 at degree 1 to 2.4e-10 at degree 10
against a degree-16 reference). A curved (order-2 geometry) pyramid is not covered.

### Orientation

At basis orders up to 2 every built-in basis places at most one DOF on any shared edge or face, so every
generated `OrientationTransform` is the identity permutation with sign +1
(`BasisTest.OrientationTransformsAreIdentityAtTheSupportedOrders` checks this directly) and the round-trip
test compares for exact equality. That is a fact about the supported orders, not about the machinery:
`local_to_canonical_vertices`' six- and eight-way permutation tables are exercised by the round-trip
path but never permute more than a one-element array. An order-3 or higher basis (two or more DOFs per
shared edge) is the first case with a non-identity permutation, and extends
`BuiltinBasis::orientation_transforms()` without changing the `Basis`/`apply_orientation` contract.

### The DOF map

`fuse_physics::fe::DofMap` (`fe/dof_map.hpp`) numbers global DoFs over a `spatial::Mesh` for one or more
`Variable`s. A DoF is keyed by an exact (never hashed) record: variable, component, entity dimension, the
sub-entity's sorted corner global vertex ids, slot. This differs from KARMA's coordinate-bit or
MOAB-identity keying (`dof_map.hpp`'s header comment has the full comparison). It numbers vertex DoFs,
edge and face DoFs (including the interior edges of 3-D cells at order 2) and cell-interior DoFs;
`DofAssociation`'s face-centre numbering is cross-checked geometrically against
`ReferenceTopology::side_topology`'s side order for Hex27, Prism18 and Pyramid14. A shared edge DoF is
numbered exactly once across the elements that share it. Variables are numbered contiguously
(variable-major `DofKey` order), which is what lets a block preconditioner take one block per variable.

A single-rank path (no communicator) and a collective path (under `FUSE_ENABLE_MPI=ON`) both exist.

**Ownership.** A shared sub-entity is owned by the minimum owner rank over its corners
(`fe::shared_sub_entity_owned_here(mesh, corners)`, a free function so it can be unit-tested against a
hand-built `Mesh` with split vertex ownership). The rule needs an owner-rank query:
`fuse::MeshView::owner(dim, i)` (a rejecting default, mirroring `cell_vertices`), implemented by
`fuse::native::NativeMeshView` (its stored `vertex_owner`/`cell_owner`), the serial backend's view
(always rank 0) and `MoabMeshView` under `FUSE_HAVE_MOAB`, which reads `ParallelComm::get_owner` from the
`ParallelComm` the backend adopted (`MoabMeshView::set_parallel_comm`, called at both construction
sites). A `MoabMeshView` without a `ParallelComm` refuses (`E-MESH-NO-OWNER`) rather than answering
rank 0: a view that cannot know the owner must not invent one for a distributed numbering to build on.

**Ghost resolution** uses a point-to-point exchange: `Reducer::alltoallv` (counts by `MPI_Alltoall`,
payload by `MPI_Alltoallv`) carries a request to the candidate owner and one answer back, so per-rank
volume is what the rank sends and owns instead of `max_count * n_ranks` records. Integer fields travel
through native `std::int64_t` transport (`Reducer::allgather` has a native `std::int64_t` overload with
a generic `std::bit_cast` default; `SerialReducer` copies and `MpiReducer` uses `MPI_INT64_T`); a
`double` field travels through the `double` channel. Padding slots are marked with a sentinel so ranks
with fewer owned DoFs than the maximum do not decode padding as a DoF. An unresolved ghost is agreed
through one `Reducer::sum` before any rank refuses (`E-PHYSICS-DOFMAP-GHOST-UNRESOLVED`; the message
carries the total and the rank's share), so every rank refuses together. The key-width refusal before
the first collective is a structural guard: interior keys carry the element id and the widest
sub-entity has four corners.

### The algebraic system

`fuse_physics::fe::AlgebraicSystem` (`fe/algebraic_system.hpp`) assembles a
`fuse::linalg::DistSparseMatrix`/`DistVector` from a caller's per-element triplet/entry callback,
looping `DofMap::owns_element_for_assembly`'s elements. Foreign-row contributions go to their owners
through the same `Reducer::alltoallv` exchange (bucketed by row owner) before
`DistSparseMatrix::from_triplets`; no second sparse-matrix type exists.

`eliminate_dirichlet_rows` performs symmetric row/column elimination with the known value moved to the
right-hand side (KARMA's own `constrain()` is a no-op). It is scoped to the single-rank/global-system
case; a distributed version needs a cross-rank right-hand-side exchange.

### Element kernels

`fe::element_mass_matrix` / `element_stiffness_matrix` (`fe/generic_operators.hpp`) are the
physics-agnostic bilinear-form kernels: mass from KARMA's `generic_mass_operator.h`, stiffness the
textbook form `integral(grad(phi_i) . grad(phi_j)) dV`.

`fe::element_transient_reaction_diffusion` / `TransientReactionDiffusionModel` compute the
Crank-Nicolson residual and analytic Jacobian. The reaction model is quadratic,
`R(u) = alpha*u^2 + beta*coupled + source`, so that `R'' != 0` and the finite-difference Jacobian check
has a truncation term to balance against round-off (an affine model collapses the check to round-off
alone). `jacobian_finite_difference_tolerance` implements the derivation: the balancing step
`h* = sqrt(2 eps |f| / |f''|)` and the bound `E(h*) = sqrt(2 eps |f| |f''|)`. The kernel returns
`residual_abs_terms`, the sum of the absolute terms of each residual row, because the residual is a
cancelling sum and Higham's bound applies to the absolute terms. Calling the kernel with
`state_old == state` removes the transient term, leaving the steady form `K u - M R(u)`. The sign
convention follows: the discrete steady equation is `-Laplacian(u) - R(u) = f`.

## Linear and nonlinear layers

**Linear layer.** KARMA's preconditioner wrappers (`karma_preconditioner.h`,
`petsc_native_preconditioner.h`, `petsc_block_preconditioner.h`) delegate entirely to a PETSc `PC`, so
there is no preconditioning math to port. The one KARMA routine with its own math,
`point_jacobi_solve` (`y <- x / diag(A)`, refusing a zero or non-finite diagonal), is
`fuse::linalg::JacobiPreconditioner`. There is no native equivalent of `ICC_PRECOND` (incomplete
Cholesky) or of a true multi-block `BJACOBI` with per-block sub-solvers. `fuse::linalg` provides
`SsorPreconditioner` (symmetric SOR on each rank's diagonal block, block-Jacobi across ranks, with
`DistSparseMatrix::local_diagonal_block()` for read access; `omega` outside `(0, 2)` is refused with
`E-LINALG-SSOR-OMEGA-RANGE`, a zero diagonal as for Jacobi) and `BlockJacobiPreconditioner`.

`physics.linear_layer` solves `M c = b` (the consistent mass matrix is the L2-projection operator)
through `pcg` + `JacobiPreconditioner`, with `b` built from a known `u_known` so the test isolates the
linear solve's algebraic accuracy from discretisation error.

**Nonlinear layer.** `physics.steady_reaction_diffusion_mms` passes `AlgebraicSystem`'s residual (the
steady form above) to `fuse::num::solve` (Newton-Krylov) as the `ResidualFn`, matrix-free; no second
Newton loop exists. The steady Jacobian `K - M diag(R'(u))` is assembled from the element operators (the
transient kernel's Jacobian is not the steady one) and wired through Jacobi and SSOR. A two-field coupled
row wires `BlockJacobiPreconditioner` with one Jacobi block per variable. Each preconditioned solve agrees
with the unpreconditioned one within the sum of the two derived algebraic bounds. The preconditioner's
Jacobian carries identity rows at Dirichlet boundary nodes, matching the operator; leaving those rows as
`K - M R'` roughly doubles the GMRES count.

The Newton-Krylov tolerances are `atol = 1e-9`, `rtol = 1e-8`, above the `O(sqrt(eps))` (about 1.5e-8)
differencing accuracy of the matrix-free Jacobian action: tighter values diverge
(`E-LINALG-NO-CONVERGENCE`) once Eisenstat-Walker forcing drops below that floor. The study is 2-D
(N = 4, 8, 16); a 3-D ladder is an open follow-up.

No test asserts a Newton, GMRES or pcg iteration count. Counts are recorded with `RecordProperty`; no
reference solver exists to compare them against.

## Boundary conditions, properties, and the diffusion-reaction physics module

**Boundary conditions.** `fuse_physics::fe::BoundaryConditionManager` (adapted from KARMA's
`karma_boundary_condition.h`/`karma_physics_bc_manager.h`) names a boundary GEOMETRICALLY: a
`BoundaryId` (`XMin`/`XMax`/`YMin`/`YMax`/`ZMin`/`ZMax`) is one of the six faces of a box
`spatial::Domain`, classified by comparing a vertex or element side's own position against the
MEASURED lower corner of the mesh's own vertex bounding box (`measured_lower_corner`, a
componentwise minimum over `mesh.vertex_positions()`) plus the domain's own extents, within a
tolerance of `2 eps max(M, L)` over the axes, `L` an axis length and `M` the larger magnitude of its
two face coordinates: with `u = eps / 2`, the generated upper face `fl(lo + fl(hi - lo))` and the
sought one `fl(lo + L)` differ by at most `u (3 M + L)` when `hi` is `lo + L` to within one rounding.
The tolerance scales with the coordinates, not only the lengths, so a box far from the origin
resolves (`ResolveOnABoxFarFromTheOriginMatchesEveryFace`: corner near (1e6, -3e5), where the two
differ by 1.2e-10). The lower corner is measured, not assumed to be the origin: `generate_box`/`generate_2d_box` both take
an explicit lower-corner point, and a mesh built away from the origin still classifies correctly.
This replaces KARMA's own `(element, side)`/node discovery through a MOAB mesh's side sets, which
physics-kit's self-generated box meshes do not carry at all. `set()` refuses an id beyond the
domain's own dimension (`E-PHYSICS-BC-UNKNOWN-BOUNDARY`), a second condition on one boundary
(`E-PHYSICS-BC-DUPLICATE`), and a non-positive Robin coefficient (`E-PHYSICS-BC-ROBIN-COEFFICIENT`
-- a Robin coefficient of exactly zero is a Neumann condition under a different name and does not
supply the coercivity a Robin-only boundary needs). `resolve()` is single-rank only, refusing
`E-PHYSICS-BC-MPI-NOT-SUPPORTED` against a multi-rank `DofMap`, and refuses
`E-PHYSICS-BC-UNMATCHED` naming the boundary if a registered condition matches no side of the
mesh at all (a caller's own typo'd or absent boundary, otherwise silently left unconstrained).
Dirichlet dof resolution goes through `DofMap::dofs_on_side`, not vertex connectivity alone, so a
higher-order basis's own edge/face dofs on a Dirichlet boundary are constrained too (`physics.
boundary_condition`'s own `ResolveWithP2BasisIncludesEdgeDofsOnTheBoundary` measures this on a P2
Quad9 mesh). All refusals are proven firing, with their own code checked, by
`physics.boundary_condition`.

**Properties.** `DiffusionReactionCoefficients` (`diffusion_reaction_properties.hpp`, adapted from
KARMA's `diffusion_reaction_assembly.h`) holds diffusion and reaction as `(position, field) ->
value` callables, with analytic derivatives for an exact (not finite-difference) Newton
linearization; `set_forcing`'s own `(position, time) -> value` callable is the sole load-term
mechanism (KARMA's own `source`/`source_derivative` coefficient members are not ported: a
position-dependent load term is already exactly what `set_forcing` carries, so a second,
unread mechanism for the same concept would only invite the two to drift apart). KARMA's own
`DiffusionReactionSystemPropertyManager` (a `map<material_id, DiffusionReactionCoefficients>`) is
NOT ported as a map: physics-kit's self-generated meshes carry no material/block id at all, so
`DiffusionReactionSystem` owns exactly ONE `DiffusionReactionCoefficients` for the whole domain --
still position-dependent, which covers a single manufactured-solution domain.

**The physics-module contract.** `DiffusionReactionSystem` is ONE CONCRETE CLASS, not an abstract
base: `fuse::num`'s `TemporalProblem`/`solve()` take a struct of `std::function` members or a plain
residual callable, neither needing a physics module to inherit from anything, so this class's public
methods (`assemble_residual`, `rhs`, `mass`, `assemble_approximate_jacobian`, `as_temporal_problem`,
`project_initial_condition`, `l2_error`) ARE the handful of KARMA `KARMAPhysicsBase` virtuals this one
physics module needs, bound by value rather than by virtual dispatch. The full virtual-by-virtual
mapping -- `form_initial_guess -> project_initial_condition`, the commented-out `compute_L2_error ->
l2_error` (implemented here, not left commented out), `find_boundary_conditions ->
BoundaryConditionManager`, and everything else DROPPED (coupled-physics registry, named-multi-system
lookup, PETSc Mat preallocation, checkpoint I/O, the nine time-stepping-framework lifecycle hooks,
material/partition numbering) with its own reason -- lives in `diffusion_reaction_system.hpp`'s own
header comment, not duplicated here.

**The assembled Jacobian.** `fuse::num::solve` differences the residual for its Jacobian action, so
no matrix is needed to solve. `assemble_approximate_jacobian` returns `K(k(x)) + M diag(R'(x)) + B`
with identity Dirichlet rows: the diffusion form with the coefficient evaluated at the state, the
derivative of the nodal reaction term (the residual adds the reaction, so the block is added), and
the boundary mass block of the Robin faces. For a diffusion coefficient that does not depend on the
field it is the Jacobian of `assemble_residual`; otherwise the derivative of the coefficient is left
out (a Picard linearisation). Its use is as the matrix a Jacobi, SSOR or block-Jacobi preconditioner
is built from. `AssembledJacobianIsTheDerivativeOfAnAffineResidual` checks it against
`F(x + v) - F(x)` for an affine residual with a Robin face and a reaction coefficient of 40, where the
identity holds without differencing error: the largest difference is 8.9e-16 against a derived
rounding bound of 3.1e-12.

**Dirichlet rows.** The steady residual has the row `x_i - g_i` at a Dirichlet dof, and the assembled
Jacobian the matching identity row. The transient forms state `u' = g'(t)` there, with `g'` read from
`BoundaryCondition::value_time_derivative`; with `u(0) = g(0)` this integrates to `u(t) = g(t)` to the
integrator's own accuracy for every tableau. A Dirichlet condition without a rate is refused in a
transient evaluation (`E-PHYSICS-DRS-DIRICHLET-RATE-MISSING`): a missing rate cannot be told apart
from data that is constant in time. `BoundaryCondition::steady_dirichlet` builds the constant case.
`DiffusionReactionSystem` is single-rank, because `BoundaryConditionManager::resolve` refuses a
multi-rank `DofMap`.

**Steady solves** (`physics.diffusion_reaction_system`, `physics.diffusion_reaction_steady_order`,
`physics.diffusion_reaction_steady_order_fine`).

- *Exact residual.* For `u = x + 2y` with a linear reaction and one Dirichlet, one Neumann and one
  Robin boundary, the residual at the nodal values is 7.2e-16 at most, against a derived rounding
  bound of 1.7e-12.
- *Order.* `-Laplacian(u) + u^2 / 2 = f`, `u = prod sin(pi x_k)`, homogeneous Dirichlet, three meshes
  each. The band is twice the change between the two observed orders plus the effect of the
  algebraic error, which is bounded from the final residual, the coercivity `d pi^2` less the
  reaction shift, and the smallest eigenvalue of the mass matrix (the derivation is the header
  comment of `physics-kit/test/diffusion_reaction_steady_order.hpp`; the bound is below 1e-10 on every level).

  | discretisation | meshes | observed orders | band |
  |---|---|---|---|
  | Q1, 2-D | 4, 8, 16 | 2.0024, 2.0005 | 0.0039 |
  | Q2, 2-D | 4, 8, 16 | 2.977, 2.995 | 0.036 |
  | Q1, 3-D | 2, 4, 8 | 2.058, 2.012 | 0.092 |
  | Q1, 3-D (`physics.diffusion_reaction_steady_order_fine`, a slow row: 46 s unoptimised) | 4, 8, 16 | 2.012, 2.003 | 0.018 |

- *Preconditioner.* `SsorFromTheAssembledJacobianReducesGmresIterations` compares the first Newton
  step alone: the same linear system, the same right-preconditioned residual, the same fixed forcing
  term of 1e-4. GMRES iterations without and with SSOR are 5 and 5 at n = 8, 9 and 8 at n = 16, 17 and
  13 at n = 32 (36 and 19 at n = 64, not run by the test). A sum over all Newton steps is not a
  comparison of preconditioners: the later systems, their forcing terms and the number of steps
  differ between the two runs.

**Transient forms** (`physics.diffusion_reaction_transient`). The semi-discrete system is
`sum_j M_ij u'_j = f_i(t, u)` on the free rows (j over all dofs) and `u'_i = g'_i(t)` on the Dirichlet
rows. Two forms are offered.

- The mass form, `as_temporal_problem()`: `rhs` and `mass`, with nothing solved inside either, for an
  integrator with mass-matrix support (`DirkIntegrator`).
- The explicit form, `as_explicit_form_temporal_problem()`: `explicit_rhs` returns `u'` by solving the
  free rows with the Dirichlet columns of M moved to the right-hand side, a symmetric positive
  definite system, to the accuracy the arithmetic allows. It is for an integrator without
  mass-matrix support (the explicit family, `RosenbrockIntegrator`).

Both mass matrices are assembled once by `initialize()`. For a solution in the finite-element space,
`u = (1 + x + y) exp(-t)`, the semi-discrete system is satisfied by `u' = -u` exactly: the mass form
leaves 6.3e-16 at most (derived bound 4.8e-13) and the explicit form returns the rate to 4.5e-13 in
the 2-norm (derived bound 2.5e-9). This checks the mass operator, including the columns that couple
a free row to Dirichlet dofs, with no discretisation error in the way.

Temporal order is measured by self-convergence on one mesh: the states after `2^k` steps to the same
final time share their spatial error, so the differences of successive states contain the temporal
error alone. The band is twice the change between the last two observed orders, and the last order
must be the closer of the two to the expected one.

| integrator | form | observed orders | band |
|---|---|---|---|
| SDIRK22 | mass | 2.067, 2.028, 2.012 | 0.030 |
| Kutta22 | explicit | 2.043, 2.021, 2.011 | 0.021 |
| ROS22 | explicit, with a supplied Jacobian action | 1.828, 1.887, 1.937 | 0.100 |

`RosenbrockIntegrator` differences `explicit_rhs` when no Jacobian action is supplied. A Rosenbrock
stage is one linear solve with no outer residual to judge it by, so a stage tolerance below the
accuracy of the differenced action is refused (`E-INTEGRATOR-STAGE-SOLVE-FAILED`, naming
`E-LINALG-ATTAINABLE-ACCURACY`) and the state is left untouched
(`Ros22RefusesAStageToleranceBelowTheDifferencedActionsAccuracy`). The order study supplies the
action exactly, from the affine structure of the problem.

**Cost of an element visit.** A Newton-Krylov solve revisits every element on every residual
evaluation. `GeometryMap` detects a constant Jacobian exactly (always for Edge2, Tri3 and Tet4; for
Quad4, Hex8 and Prism6 when the Jacobians at every vertex are bitwise identical), validates such a
cell by one evaluation in place of the recursive inspection, and returns the cached Jacobian, metric
and measure. A general cell is inspected as before. `Mesh::neighbor` is a lookup in a side index that
`add_element` maintains; a search over every side of every element made `initialize()` quadratic in
the number of elements (9.65 s for a 32 x 32 mesh in an unoptimised build, 0.15 s with the index).

## Tolerance provenance

Every tolerance in the physics-kit tests is derived or exact; none is a chosen constant adjusted after a
failure. KARMA states no derivation for its own thresholds.

| test | tolerance | derivation |
|---|---|---|
| `physics.fe_quadrature` monomial exactness (all cell families) | `Quadrature::round_off_bound(...)` | Higham's recursive-summation term with the per-term operation count, plus the Golub-Welsch eigen-solver's node and weight errors from the Jacobi sweep count the rule records; generic in `dimension()`/`n_points()`/`generation_sweeps()`. Measured largest `|error| / bound` over every monomial of every rule is 1.25e-1 (`QuadratureTest.ZZZ_DerivedBoundHasMeasuredMargin`) |
| `physics.fe_reference_topology` Kronecker, partition-of-unity and derivative-sum checks | `1e-10` to `1e-12` | measured round-off floor for the closed-form shape functions in double precision |
| `physics.fe_geometry_map` inverse-map contract | `1e-12` / `1e-10` (reference / residual) | KARMA's own literal arguments to `try_inverse_map`, carried over unchanged |
| pyramid Vandermonde-like inversion refusal | `cond > 1/sqrt(eps)` ~= 1.49e8 | Golub & Van Loan, *Matrix Computations*, 4th ed., section 2.3.3: the forward error of `Ax = b` is `O(cond * eps)`, so a solve loses about `log10(cond)` digits and fewer than about 8 remain at this threshold; the measured condition number is named in the refusal message |
| `physics.fe_basis` orientation round trip | exact equality | every transform at orders up to 2 is the identity permutation with sign +1, so the round trip reads the same value twice |
| `physics.spatial_mesh` edge extraction coordinate match | `1e-9` | an equality tolerance on a structured generator's exact grid coordinates, comfortably above round-off on O(1) coordinates |
| `physics.mass_assembly` P1-simplex and Quad4 element mass | `(n_q + 3) * eps * S`, `n_q` from `fe.n_quadrature_points()`, `S` = volume (Tet4) or area 1 (Quad4) | Higham (2002) Thm 8.1: `n_q` products of three operations each, summed; `S` bounds the sum of absolute terms since a P1 shape value lies in `[0, 1]` |
| `physics.mass_assembly` global entry sum equals domain volume | `n_elements * 8 * 8 * eps * domain_volume` | Higham summation bound over each element's 8x8 contribution, using `sum_i phi_i = 1` |
| `physics.diffusion_assembly` reference-Tet4 closed form | `8 * eps * max(1, |expected|)` | one constant-gradient dot product and one `JxW` multiply on a degree-0-exact element: a few ULPs |
| `physics.diffusion_assembly` stiffness symmetry | exact equality | `dot(a,b)` reads the same two operands in the same order for `(i,j)` and `(j,i)`; IEEE 754 gives bit-identical results |
| `physics.diffusion_assembly` row sums to zero | `8 * eps * (largest entry magnitude)` | Higham bound over 8 terms, using `grad(sum_i phi_i) = 0` |
| `physics.algebraic_system` Dirichlet-elimination symmetry | exact equality | every triplet value is an exact dyadic double, so `fl(a - b*c)` has no inexact input |
| `physics.transient_reaction_diffusion_assembly` forward-difference Jacobian | `2 sqrt(delta |f''|) + |J_ik| eps |u_k| / h*`, `delta = (3n + n_q + 4) eps S_i`, `S_i` = sum of absolute terms of `residual_i`, `f'' = 0.5 M_ik R''`, `h* = 2 sqrt(delta / |f''|)` | truncation `(h/2)|f''|`; two evaluations' round-off `2 delta / h` by Higham Thm 8.1 over the absolute terms (the residual cancels, so `|residual_i|` understates it); rounding of `u_k + h`. Measured worst `|error| / bound` over the 16 entries is 5.05e-1 |
| `physics.transient_reaction_diffusion_assembly` affine case (`alpha=0`) | `8 * eps * max(1, |expected|)` | `R'' = 0` exactly, so the Jacobian is `M/h + 0.5*K` with no reaction term; the same few-ULP floor as the diffusion closed form |
| `physics.dof_map` `EdgeVerticesMatchNodesForMidpoints*` | `1e-12` | round-off floor for an exact-rational reference-node average |
| `physics.linear_layer` projection error | `kappa(M) * rel_tol` | standard SPD residual-to-error bound; `kappa(M) = size + 1 = 5` is exact for a single P1 simplex element (`M = c(I+J)`, eigenvalues `c(size+1)` once and `c` `size-1` times, `size = dimension + 1`). The test is single-element; measured 2 pcg iterations |
| `physics.steady_reaction_diffusion_mms` algebraic error | `algebraic_l2_error_bound()` | closed-form eigenvalues of the uniform Q1 Dirichlet Laplacian, `lambda_max(M) = h^2`, the measured maximum nodal value for the reaction shift, and block-triangular handling of the identity boundary rows; the residual contract `final_residual_norm <= max(atol, rtol ||F_0||)` bounds the algebraic term. A Gershgorin bound is inconclusive because interior stiffness rows sum to zero |
| `physics.steady_reaction_diffusion_mms` convergence order | `2.0 +/- band` | Richardson leading-order prediction; the band adds the algebraic bound as `(delta_k/err_k + delta_{k+1}/err_{k+1}) / ln 2` (measured 2.8e-6 against a Richardson band of 2.2e-2); the check `|r2 - 2| < |r1 - 2|` stops a wide band from a pre-asymptotic pair passing vacuously |
| `physics_assembly_mpi_test` mass-matrix sum | Higham bound over the global term count | as `physics.mass_assembly`, over the summed-across-ranks element count |
| `physics_assembly_mpi_test` lumped row sums | `64 * eps * lumped_row` | Higham bound, 8x8 terms per row; non-negative mass entries bound the sum directly |
| `physics.boundary_condition` face classification | exact equality | the test coordinates are exactly representable; the manager's own tolerance is derived in "Boundary conditions" above |
| `physics.diffusion_reaction_system` residual of an affine solution, and `F(x + v) - F(x) - J v` | `gamma_400 * (absolute row sum)` | Higham (2002) Section 3.1: a row is a sum of at most 400 rounded terms (4 elements, 9 quadrature points, 9 basis terms, 4 boundary-face sums, up to 20 operations per term); the absolute row sums are 16/3 for the Q1 stiffness matrix, `h^2` for the mass matrix and `h` for a boundary mass block. Measured 7.2e-16 against 1.7e-12 and 8.9e-16 against 3.1e-12 |
| `physics.diffusion_reaction_steady_order` convergence order | `2 |r1 - r2|` + algebraic term | header comment of the test: coercivity `d pi^2 - 2 alpha U sqrt(kappa(M))`, `lambda_min(M) >= (c h)^d` from the element mass matrices (Wathen 1987), the nodal bound `U = 2` asserted per level |
| `physics.diffusion_reaction_transient` exact identities | `gamma_k * (absolute row sum)`; for the explicit form, divided by the smallest mass eigenvalue and with the measured residual of the inner solve added | header comment of the test |
| `physics.diffusion_reaction_transient` temporal order | `2 |p_last - p_previous|` | leading-order behaviour of an observed order; no allowance for solver tolerances, so loose tolerances can only fail the test |

## Multi-rank tests

The MPI rows need `FUSE_ENABLE_MPI=ON` and a launcher, and are registered where the target exists
(`fuse_add_mpi_test` accepts `SRC_DIR`, like `fuse_add_gtest`): `parallel.physics_mesh_mpi.{2,3}`,
`physics_assembly_mpi_test` (ranks 2 and 3, a 3-D Hex8 slab split along z; checks the global mass-matrix
entry sum against the box volume and `M * 1` against an independently summed lumped row vector),
`physics_dof_map_mpi` (ranks own 1, 2 and 3 elements, the shape that exercises padded exchanges) and
`moab_owner_mpi` (2 ranks, sphere fixture). The `moab_owner_mpi` real branch is syntax-checked only, and
`E-PHYSICS-DOFMAP-GHOST-UNRESOLVED` has no firing test because it needs a multi-rank partition that
fails by construction.

## Known limits

- The pyramid inversion refusal has no firing test: the matrix is a fixed reference-node matrix.
- Curved (order-2 geometry) pyramids have no accuracy study.
- `eliminate_dirichlet_rows` is single-rank.
- The coupled two-field row's algebraic bound sums two single-field bounds rather than using a coupled
  coercivity bound, and its load vector uses the continuum coupling term against an element average in
  the residual, so no order claim is made for it.
- KARMA's `test/assembly_test.C` (low-level PETSc scatter primitives) is not retired: every behaviour
  it checks is covered by a named `fuse::linalg`/`AlgebraicSystem` test except the vector-side
  negative/out-of-range index refusal, which has no dedicated test at this layer (the matrix-side
  equivalent is `RefusesATripletWhoseColumnIsOutOfRange`).
- A cell whose Jacobian is not constant (a distorted Quad4 or Hex8, every second-order geometry)
  is validated by the recursive inspection on every reinit; the result is not cached per element.
- `DiffusionReactionSystem` is single-rank (`E-PHYSICS-BC-MPI-NOT-SUPPORTED`).
- `RosenbrockIntegrator` on the explicit form needs a supplied Jacobian action for a stage tolerance
  below the accuracy of a differenced `explicit_rhs` (about 4e-7 relative on the test problem).
- KARMA's `diffusion_lifecycle_test.C` assertions on material-id property dispatch and on rejecting
  discontinuous or spectral bases have no counterpart: the class has one coefficient set and takes
  the continuity of the `DofMap` it is given.
- No `Component`/`ImplicitComponent` adapter couples two `DiffusionReactionSystem` instances. The
  boundary dofs of a system are a subset of its dof vector in element-traversal order, neither
  contiguous nor evenly strided, and `Layout` (`include/fuse/field/field_ref.hpp`) describes a dense,
  evenly strided run; an adapter needs a gather into a contiguous interface buffer before
  `register_fields()` and a scatter back after `advance()`.
- `DiffusionReactionCoefficients` holds `diffusion`/`reaction` (with analytic derivatives); the load
  term is `set_forcing`'s own `(position, time)` callable, with no second, unread mechanism for the
  same concept (`diffusion_reaction_properties.hpp`'s own header comment).
- Material/property dispatch is a single domain-wide `DiffusionReactionCoefficients`, not a
  material-id-keyed map: physics-kit's self-generated meshes carry no material/block id to key
  against (`diffusion_reaction_properties.hpp`'s own header comment).
- `DiffusionReactionSystem`'s nodal reaction evaluation and Dirichlet/initial-condition dof values
  use each dof's own PHYSICAL position (`dof_physical_positions`, an interpolation-point-through-
  GeometryMap lookup), valid for any basis order including a P2 basis's own edge/face dofs
  (`physics.dof_map`'s own `DofsOnSideMatchesGeometricSideMembershipForP2Quad9` and `physics.
  boundary_condition`'s own `ResolveWithP2BasisIncludesEdgeDofsOnTheBoundary` measure this).
