# physics-kit (S8)

The optional finite-element component: KARMA's FEM core, physics modules (advection-reaction,
diffusion-reaction, Gray-Scott), FEM transfer and the 22-tableau Runge-Kutta/Rosenbrock integrator
family, ported into `fuse::` per [](physics-kit-disposition.md)'s disposition and
[](integrators-design.md)'s design for the temporal integrators. [](roadmap.md)'s S8 row tracks
where the port currently stands; at the skeleton stage (this page) no physics code exists yet --
only the directory, the build option, the two gates and one placeholder test described below.

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

Two static checks keep the boundary and the naming honest, both run as part of `tools/check-
layering.py`'s and the new `tools/check-physics-kit-naming.py`'s CI job:

- **Layering** (`tools/check-layering.py`'s `check_physics_kit_boundary`): refuses a
  `#include ".../fuse_physics/..."` or a `fuse::physics_kit` link anywhere under `include/fuse/`,
  `src/`, `python/` or `examples/`. physics-kit/ itself is exempt -- it is the one place allowed to
  name any `fuse::` layer, by construction.
- **Naming** (`tools/check-physics-kit-naming.py`): refuses a `KARMA*` type prefix, a `karma_` file
  prefix or a `.C` extension anywhere under `physics-kit/` -- the three rules [](porting.md)'s
  naming map states for code arriving from `upstream/karma`, applied here because physics-kit is
  outside the four directories `tools/check-layering.py`'s own naming check already covers.

Both gates were proven to fire against a probe file (added, run, shown to produce a violation,
removed) before being accepted; the S8 skeleton track's `measured_results` has the verbatim output.

## Status

No physics code lives under `physics-kit/` yet. The skeleton (this track) is: the directory layout,
the `fuse::physics_kit` target with one placeholder translation unit, the `physics.skeleton` gtest
(asserts a version string and that `FUSE_HAVE_PETSC` reaches physics-kit code), the two gates above,
and this page. [](physics-kit-disposition.md) and [](integrators-design.md) are the read-first design
pages for what a later track ports here.

## Installation and the `physics_kit` component (S8.2, lead, 2026-10-31)

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
`include/fuse_physics/`. Both directions are measured by `install.consumer_find_package`
(`test/install_consumer/`): with physics-kit on, a second consumer links the component and runs; with it
off, the consumer asks optionally and refuses if the component appears.

`tools/check-layering.py` applies the public-header third-party rule to `physics-kit/include/fuse_physics/`
as to any `fuse::` layer: none of PETSc, MPI, MOAB, HDF5, Eigen or Kokkos-Kernels may be named in a header a
consumer compiles, and of Kokkos only `Kokkos_Core.hpp`; a probe header naming `petscsys.h` and
`Kokkos_Random.hpp` produced two violation lines and was removed. `tools/check-invariants.py` has
scanned `physics-kit/` since the skeleton.


## Wave 1: the finite-element core (S8.3, 2026-10-31)

The first physics code in `physics-kit/`: KARMA's FEM reference-cell core (`numerics/fe`) and
its generic mesh/domain surface (`numerics/spatial`), ported per
[](physics-kit-disposition.md)'s D-A ("ADAPT the KARMA FEM core to `fuse::linalg`, not a literal
port") and this wave's own read-first instructions. `numerics/algebra` is **not** adapted into a
physics-kit type at all: every file in that directory retires outright, superseded by
`fuse::linalg`'s existing `DistVector`/`DistSparseMatrix`/dense least-squares facade (see the
table below and `UPSTREAMS.lock`'s newly-added `[karma]` entries).

### What exists now

`fuse_physics::fe` (`physics-kit/include/fuse_physics/fe/`): `GeometryVariant`/`ReferenceTopology`
(reference-cell shape functions and side tables), `Quadrature` (native, not PETSc-backed --
see below), `Basis`/`BasisRegistry` (a type-erased Lagrange-nodal basis, ported from KARMA's own
Concept/Model pattern), `GeometryMap` (the isoparametric map, Jacobian, inverse map and side
geometry), and `FEBase` (the mesh-bound evaluator: `phi`/`dphi`/`JxW`/`xyz`/normals at a set of
quadrature points). Implemented cell families: Edge, Triangle, Quadrilateral, Tetrahedron,
Hexahedron, at Lagrange order 0, 1 or 2 (the complete tensor/simplex family -- Quad9/Tet10/Hex27,
not the Quad8 serendipity variant).

`fuse_physics::spatial` (`physics-kit/include/fuse_physics/spatial/`): `Domain` (a box-shaped
domain descriptor, with the I6 fix, KNOWN_ISSUES.md D99) and a **native**, serial, tier-0 `Mesh`
(structured Hex8 boxes only) plus `extract_boundary_faces`.

### Thin physics-kit types this wave introduces (D-A's "name the gap")

- `fuse::physics::Point`/`Vector3`/`Matrix3` (`physics-kit/include/fuse_physics/geometry.hpp`):
  KARMA's own `Point`/`Vector3`/`Matrix3` are Eigen fixed-size types
  (`upstream/karma/include/base/karma_geometry.h`), and Eigen may never appear in a physics-kit
  public header (`tools/check-layering.py`'s `FORBIDDEN_PUBLIC`). fuse has no generic small-vector
  type at this layer either (`fuse::linalg`'s public header names nothing third-party at all, by
  design, and its facade is sized for the distributed coupled-iteration problem, not a per-element
  1x1..3x3 array). A plain-struct `Point`/`Matrix3` with the handful of dense operations (3x3
  determinant/inverse, symmetric-eigenvalue bounds via the closed-form Smith/Kopp method) the FE
  core's geometry maps need is the physics-kit-own type this gap calls for.
- `fuse::physics::spatial::Mesh` (native, no MOAB/MPI): named as a gap in
  [](physics-kit-disposition.md) Appendix A itself ("no fuse::backend equivalent -- MeshView is
  remap-oriented, not FE-assembly-oriented"). This wave's own verification budget (tier 0, no MPI
  runs in the sandbox) cannot build or test a MOAB-backed binding even if one were written here, so
  the native, serial topology surface is what is ported and verified; the MOAB/MPI binding is an
  explicit open item (open_questions), not a silent omission.

### Native quadrature, not PETSc-backed

KARMA's `KARMAQuadrature` calls PETSc's `PetscDT*` functions unconditionally. physics-kit is tier 0
(nothing beyond Kokkos and fmt; PETSc optional, read only through `FUSE_HAVE_PETSC`), so a
quadrature generator this wave's own tier-0 tests can build and run cannot wrap PETSc at all.
`fuse_physics::fe::Quadrature` instead generates Gauss-Legendre/Gauss-Jacobi nodes and weights
natively via the Golub-Welsch eigenvalue construction (Golub & Welsch, 1969), using a collapsed
(Duffy) conical product for the triangle and tetrahedron (Stroud, 1971; Karniadakis & Sherwin,
2005, section 2.3) -- the same technique KARMA's own `create_product_quadrature()` comment already
derives for the pyramid's single collapsed axis, applied here with one collapse (triangle) or two
(tetrahedron).

### Tolerance provenance

| test | tolerance | derivation |
|---|---|---|
| `physics.fe_quadrature`'s monomial-exactness checks | `Quadrature::round_off_bound(largest_term)` = `n_points * eps * largest observed weight*abscissa-power term` | derived here: a bound on the accumulated rounding of `n_points` floating-point products-and-sums (not copied from KARMA, which states no derivation for any of its own thresholds -- [](physics-kit-disposition.md) Section 4's own "Pattern across all of them") |
| `physics.fe_reference_topology`'s Kronecker/partition-of-unity/derivative-sum checks | `1e-10` to `1e-12` | round-off floor for the analytic closed-form shape functions evaluated in double precision; not a KARMA-stated bound (this wave's own, chosen as a measured round-off floor, not widened after a failure) |
| `physics.fe_geometry_map`'s inverse-map contract (`geometry_inverse_map_contract_test.C`) | `1e-12` / `1e-10` (reference / residual) | KARMA's own test's literal arguments to `try_inverse_map`, carried over unchanged (this IS a KARMA-stated tolerance, not re-derived) |

### Deferred this wave (open_questions has the full list)

Prism and pyramid cell families (the pyramid's rational basis and collapsed quadrature are a
separate numerical undertaking; KNOWN_ISSUES.md D100 carries the distorted-pyramid accuracy risk
forward for whenever that lands); `OrientationTransform`/`apply_orientation` (basis DOF
re-signing across a shared edge/face, needed only once elements assemble into a global system --
`numerics/linear`, Wave 2); the MOAB/MPI-backed `Mesh` binding; `karma_mesh_extract_edge.C` (needs
a 2-D source mesh `spatial::Mesh` does not generate this wave); `numerics/algebra`'s full
retirement is implemented as documentation (`UPSTREAMS.lock`), not code, per D-A's own alternative
clause.

### Lead's review of wave 1 (2026-10-31)

Three corrections before landing, each measured:

- **Quadrature tolerance.** The wave's exactness test bounded only the summation (`n_points * eps *
  largest_term`) and then took `max(bound, 1e-13)`; the floor was a chosen constant 10 to 100 times
  the derived value and would have hidden a real error of that size. The bound is now derived in
  full (`Quadrature::round_off_bound`'s header comment: Higham's recursive-summation term with the
  per-term operation count, plus the Golub-Welsch eigen-solver's node and weight errors from the
  Jacobi sweep count the rule now records) and the floor is gone. Measured: the largest
  |error| / bound over every monomial of every rule is 1.25e-1, printed by
  `QuadratureTest.ZZZ_DerivedBoundHasMeasuredMargin`, so the derivation has an eightfold margin.
- **Basis-registry refusal.** The mutation row `basis-registry-unknown-key-not-refused` had been
  declared equivalent because `UnknownKeyRefuses` could not tell the early `invalid basis key`
  refusal from the factory map's `unknown basis key`; the real finding is that `valid_key()` accepts
  every Lagrange key at any order, so NO test reached the early refusal. `InvalidKeyIsRefusedBeforeTheLookup`
  (a non-Lagrange family above `maximum_basis_order`, asserting the early message) now kills it.
- **Geometry-map zero-scale refusal** stays declared equivalent with a verified reason: `scale` is the
  largest pairwise node distance, so `scale <= 0` holds exactly when every node coincides, and the
  Jacobian is then the exact zero matrix (derivatives sum to zero) which the downstream degeneracy
  check refuses. The lead checked the scale definition in `geometry_map.cpp`.

Verdicts accepted as the track declared them: algebra retires onto `fuse::linalg` (every file, per the
disposition's own rows; `dense_test.C` retires because `test/linalg/least_squares_test.cpp` names the
same behaviour, D-C satisfied); `spatial::Mesh` is native and serial (a MOAB- and MPI-bound mesh cannot
be built or run at tier 0 in the sandbox, so the gap is named, not guessed); prism and pyramid, the
2-D edge extraction and `OrientationTransform` are deferred to wave 2 with the track's reasons. D99 (the
vendored `Type != Hexagon || Type != Unstructured` tautology) was checked against the vendored lines
and is a real defect, fixed in the port as a refusal with a test and a reverting mutation row.

## Wave 1b: prism, pyramid, orientation and the MeshView binding (S8.3, 2026-10-31)

Owner instruction 2026-10-31: "Execute FE wave 1 first completely and then wave 2" -- every item
wave 1 deferred (above) closes in this wave; Wave 2 (DOF map, algebraic system, assembly,
`numerics/linear`, `numerics/nonlinear`) does not start.

### What is added

`fe::GeometryVariant`/`ReferenceTopology`: Prism6/15/18 and Pyramid5/13/14 (node tables, side
tables -- Prism6's sides are 3 quadrilateral + 2 triangular, Pyramid5's are 4 triangular + 1
quadrilateral), the serendipity `Quad8` (needed only as Prism15's/Pyramid13's one quadrilateral
side, never a stand-alone basis family), the pyramid's rational collapsed-coordinate basis
(`pyramid_candidates`/`pyramid_nodal_coefficients`, a native Gauss-Jordan matrix inversion with a
derived condition-number refusal threshold, replacing KARMA's Eigen `ColPivHouseholderQR` --
justified because the pyramid's own candidate-term count equals its node count exactly at both
order 1 and 2, verified by hand, so KARMA's column-pivoted selection reduces to pure inversion
here).

`fe::Quadrature`: Prism (triangle-base Gauss-Jacobi(1) times an independent Gauss-Legendre line --
a stronger, HYBRID exactness guarantee than a plain total-degree claim, since the two axes are
Jacobian-decoupled) and Pyramid (single-collapse Gauss-Jacobi(alpha=2), the technique KARMA's own
`create_product_quadrature()` comment already named for this axis).

`fe::Basis`: `OrientationTransform`/`apply_orientation`/`apply_orientation_inverse`
(`EdgeOrientation`/`TriangleOrientation`/`QuadrilateralOrientation`, `local_to_canonical_vertices`),
ported from `karma_basis.h`'s identical construction. At this wave's basis orders (<= 2) every
built-in basis places at most ONE dof on any shared edge or face, so every generated
`OrientationTransform` is PROVABLY the identity permutation with sign +1 -- verified directly
(`BasisTest.OrientationTransformsAreIdentityAtThisWavesOrders`), not assumed, and the round-trip
test's bound is therefore EXACT (0), not a derived epsilon.

`fe::GeometryMap`: `in_reference_cell`'s pyramid taper condition (its own non-box reference cell),
the matching per-family sampling bounds in `validate()`, `inverse_map`'s apex fast path, and the
extended `reference_outward_normal` table -- all four cite the exact KARMA source line they
reproduce (`geometry_map.cpp`'s own comments).

`fe::FEBase`: mixed-shape face quadrature (`quadrature_for_side()` -- `quadrature_` for a
quadrilateral side, `triangle_quadrature_` for a triangular one, selected per side; both rules
share a point count by construction since they are built to the same degree).

`spatial::Mesh`: bound to `fuse::MeshView` (not MOAB directly -- `docs/dev/layering.md`'s
"physics-kit reaches MOAB only through `fuse::backend`"), generalized from fixed Hex8-only
connectivity to variable-arity, per-element-tagged connectivity; `generate_2d_box` (2-D structured
Quad4, the source `karma_mesh_extract_edge.C` needs); `from_mesh_view` (reads ownership, global
ids, and cell-to-vertex connectivity through `MeshView::cell_vertices`); ownership/global-id
accessors (`is_vertex_owned`/`is_element_owned`/`owned_vertex_count`/`owned_element_count`/
`vertex_global_id`/`element_global_id`).

`spatial::mesh_utils`: `extract_edge_at_coordinate`, adapted (not a literal port -- KARMA's own
`karma_mesh_extract_edge.C` is legacy libMesh-era code in a visibly older style than the rest of
this port) from `karma_mesh_extract_edge.C`'s own operation: the 1-D boundary of a 2-D mesh at a
constant coordinate.

### Tolerance provenance (wave 1b additions; wave 1's table above is unchanged)

| test | tolerance | derivation |
|---|---|---|
| `reference_topology.cpp`'s pyramid Vandermonde-like matrix inversion refusal | `cond > 1/sqrt(eps)` ~= 1.49e8 | derived here: Golub & Van Loan, *Matrix Computations*, 4th ed., Section 2.3.3's standard rule of thumb (`Ax=b`'s relative error is `O(cond * eps)`) -- a linear solve loses ~`log10(cond)` decimal digits to rounding, so `cond >= 1/sqrt(eps)` is exactly the point at which fewer than the eps-to-sqrt(eps) crossover's ~8 significant digits remain; the measured condition number is named in the refusal message, not silently dropped |
| `physics.fe_quadrature`'s Prism/Pyramid monomial-exactness checks | `Quadrature::round_off_bound(...)`, same formula as wave 1's table | the formula is generic in `dimension()`/`n_points()`/`generation_sweeps()`, already covers the new cell families without modification -- re-verified (not re-derived): measured largest \|error\|/bound ratio is 1.25e-1, same order as wave 1's own measurement, printed by `QuadratureTest.ZZZ_DerivedBoundHasMeasuredMargin` |
| `physics.fe_basis`'s orientation round-trip (`ApplyOrientationRoundTripsExactlyAcrossOppositeLocalOrder`) | exact equality (0), not an epsilon | derived here: `OrientationTransformsAreIdentityAtThisWavesOrders` proves every transform this wave's bases generate is the identity permutation with sign +1 at this order, so `apply_orientation` followed by its inverse is the same floating-point value read twice, not an arithmetic result that could accumulate rounding |
| `physics.fe_test`'s `DistortedPyramidQuadratureAccuracy_D100` | measured, not asserted against a derived bound: `EXPECT_LT(error, 1e-10)` for the volume case is a round-off-scale sanity floor, not a derivation | KNOWN_ISSUES.md D100's own measurement write-up has the full numbers (volume: 0 to 1.8e-15 across degree 1-5; `exp(x+y+z)`: 1.96 down to 2.4e-10 across degree 1-10, converging to a degree-16 reference) |
| `physics.spatial_mesh`'s `ExtractEdgeAtCoordinateFindsTheRightLine` coordinate-match tolerance | `1e-9` | a plain equality tolerance on a structured box generator's own exact grid coordinates (no accumulated rounding to bound; chosen comfortably above double-precision round-off on O(1) coordinates, not derived from an error-propagation argument since none applies here) |

### D100 and D101

KNOWN_ISSUES.md D100 (distorted-pyramid quadrature accuracy) is now measured: the basic case
(volume, and a genuinely non-polynomial integrand's convergence rate) is closed; curved
(order-2 geometry) pyramids remain an open item for whenever that is read from an external
source. KNOWN_ISSUES.md D101 (new): the lead's review of wave 1 had worked around, but not
fixed, a bug in `valid_key()`'s `||` that accepted any Lagrange key regardless of order; fixed
this wave, with the regression test the wave-1 workaround could not have been.

### Scope note: orientation's own bound is honestly vacuous at this order

`OrientationTransformsAreIdentityAtThisWavesOrders` is a genuine proof, not a tautology dressed
up as one: it holds because this wave's own basis orders (<= 2) never place more than one dof on
a shared sub-entity, which is a fact about THIS wave's scope, not about the orientation
machinery itself (`local_to_canonical_vertices`' six- and eight-way permutation tables are ported
and exercised by the round-trip test's own code path, just never asked to permute more than a
one-element array). An order-3-or-higher basis (two or more dofs per shared edge) is the first
case that would exercise a genuinely non-identity permutation; the machinery is in place for that
wave to extend `BuiltinBasis::orientation_transforms()` without changing the surrounding
`Basis`/`apply_orientation` contract at all.

## Lead's review of wave 1b (2026-10-31)

Landed from a clean export after three changes:

- **The MPI mesh test is registered where the target exists.** The track had compiled three
  physics-kit sources into a `test/`-side executable because `fuse::physics_kit` is defined
  after `test/` is added. `fuse_add_mpi_test` gained the `SRC_DIR` option `fuse_add_gtest`
  already had, and `physics-kit/test/CMakeLists.txt` registers `parallel.physics_mesh_mpi.{2,3}`
  against the real target. The row is host-pending (the sandbox cannot launch MPI); it builds
  against MPICH with `FUSE_ENABLE_MPI=ON`.
- **physics-kit/test has its own unregistered-test audit.** `test/CMakeLists.txt`'s audit globs
  `test/*/*_test.cpp` and never sees this directory; the skeleton's comment justified that with
  "exactly one source file, in full view", which stopped being true at wave 1. The same audit
  now runs here (proved by a probe file: configure fails naming it).
- **Verified, not re-derived:** the pyramid rule's exactness is by construction (n = (d+2)/2
  points per axis gives 2n-1 >= d, and the collapsed monomial x^i y^j z^k has z-degree i+j+k
  against the (1-z)^2 weight; the test checks every monomial to degree 5 against the exact
  integrals, volume 4/3); the condition-number threshold `1/sqrt(eps)` follows from the
  O(cond * eps) forward-error bound and the 1e-10 pointwise floor the tests use.

Accepted as the track declared them: the pyramid inversion refusal has no firing test because
the matrix is a fixed reference-node matrix (named, not covered by a mutation row); D100 closed
for straight-edged pyramids only; `DofAssociation`'s face numbering for the new families is
cross-checked only for Pyramid14 (a wave-2 item); the two-space global-id convention
(vertices and cells numbered separately, as `MeshView` does) is carried into wave 2 unless the
owner prefers a tagged id type.


## Wave 2, round 1: the DOF map, algebraic system, and mass/stiffness kernels (S8.3, 2026-10-31)

Items 1-3 of the task's own eight (DOF map; algebraic system; assembly kernels and tests)
delivered as round 1. Items 4-8 (linear layer, nonlinear layer, iteration counts, the MPI
assembly test, and the full records sweep beyond what round 1 itself needs) are explicitly
DEFERRED to a later round -- the task's own permission: "If the wave is too large for one round,
deliver items 1-3 complete as round 1 and say so; do not deliver all eight half-done." Within
items 1-3 themselves, the transient reaction-diffusion kernel (`transient_reaction_diffusion_
assembly_test.C`'s own Crank-Nicolson residual/Jacobian with a derived finite-difference
tolerance) is also deferred -- the DOF map, algebraic system, and the mass/stiffness kernels
plus their tests are complete and tested; see "What is added" below.

### What is added

- `fuse_physics::fe::DofMap` (`fe/dof_map.hpp`/`.cpp`): global DoF numbering over a `spatial::
  Mesh`, one or more `Variable`s. Keys a DoF by an EXACT (never hashed) record -- variable,
  component, entity dimension, the sub-entity's own sorted corner GLOBAL vertex ids, slot --
  rather than KARMA's own coordinate-bit or MOAB-identity keying (`dof_map.hpp`'s own header
  comment has the full comparison). Numbers vertex dofs, codimension-1 face/edge dofs, and
  cell-interior dofs; REFUSES by name (`E-PHYSICS-DOFMAP-SUBENTITY-UNSUPPORTED`) a dof on a 3-D
  cell's own interior edge (order >= 2 on a Tetrahedron/Hexahedron/Prism/Pyramid), since no
  corner-vertex table for that case exists anywhere in this wave's public Mesh/ReferenceTopology
  surface -- named, tested (`RefusesSharedEdgeDofsOnAThreeDimensionalCell`), not silently
  mis-numbered. Single-rank convenience path (no communicator) and a collective path (built
  under `FUSE_ENABLE_MPI=ON`, never run in this sandbox) both exist; the collective path's ghost-
  dof resolution uses a padded-allgather broadcast (every rank's own owned-dof table sent to
  every other rank) rather than a point-to-point exchange -- a stated, non-hidden scalability
  limitation (open_questions). CLOSES the wave 1b open item: `DofAssociation`'s face-centre
  numbering is cross-checked geometrically against `ReferenceTopology::side_topology`'s own side
  order for Hex27, Prism18, and Pyramid14 (`FaceCentreAssociationIndexMatchesSideTopologyOrder*`);
  no disagreement found.
- `fuse_physics::fe::AlgebraicSystem` (`fe/algebraic_system.hpp`/`.cpp`): assembles a
  `fuse::linalg::DistSparseMatrix`/`DistVector` from a caller's per-element triplet/entry
  callback, looping `DofMap::owns_element_for_assembly`'s own elements and exchanging foreign-
  row contributions to their owners in ONE collective step (the SAME padded-allgather technique
  `DofMap`'s own ghost resolution uses, for the same stated reason) before calling
  `DistSparseMatrix::from_triplets` -- no second sparse-matrix type introduced. `eliminate_
  dirichlet_rows`: SYMMETRIC row/column elimination with the known value moved to the RHS (KARMA's
  own `constrain()`, `karma_dof_map.h` lines 143-144, is a no-op -- this is not that), scoped this
  round to the single-rank/global-system case (a distributed version needs a cross-rank RHS
  exchange not built this round, named in open_questions).
- `fuse_physics::fe::element_mass_matrix` / `element_stiffness_matrix` (`fe/generic_operators.
  hpp`/`.cpp`): the physics-agnostic bilinear-form kernels, ported from KARMA's own `include/
  physics/common/generic_mass_operator.h` (the abstraction level `test/mass_assembly_test.C`
  itself exercises, not `karma_algebraic_system.h` directly) and, for stiffness, the textbook FE
  form `integral(grad(phi_i) . grad(phi_j)) dV` -- no KARMA diffusion-operator file exists to port
  from in this export, cited and derived as such (the same posture wave 1 took for KARMA gaps).

### Tolerance provenance (wave 2 additions; wave 1/1b's tables above are unchanged)

| test | tolerance | derivation |
|---|---|---|
| `physics.mass_assembly`'s P1-simplex and Quad4 element mass matrix checks | `(n_q + 3) * eps * S`, `n_q` read from `fe.n_quadrature_points()`, `S` = volume (Tet4) or area 1 (Quad4) | Higham (2002) Thm 8.1: `n_q` products of three operations each, summed; `S` bounds the sum of the absolute terms since a P1 shape value lies in `[0, 1]`. The landed round-1 row had `16 * eps`, "a generous n = 16 for margin" -- a chosen constant, replaced by the lead 2026-11-01. The analytic comparison value itself is KARMA's own cited closed form (`mass_assembly_test.C`'s `check_simplex_mass`, and the Quad4 consistent mass table) |
| `physics.mass_assembly`'s global entry-sum-equals-domain-volume invariant | `n_elements * 8 * 8 * eps * domain_volume` | Higham summation bound over every element's own 8x8 contribution, using the partition-of-unity identity `sum_i phi_i = 1` (so `sum_{i,j} M(i,j) = integral(1) dV`) rather than an empirical constant |
| `physics.diffusion_assembly`'s reference-Tet4 closed-form check | `8 * eps * max(1, |expected|)` | a single constant-gradient dot product and one JxW multiply on a degree-0-exact (constant integrand) element -- a few ULPs, not an empirical floor |
| `physics.diffusion_assembly`'s stiffness symmetry check | exact equality (no tolerance) | `dot(a,b)` reads the SAME two operands in the SAME order regardless of which index is `i` and which is `j` -- IEEE 754 guarantees bit-identical results, so no rounding bound applies at all |
| `physics.diffusion_assembly`'s stiffness row-sum-to-zero check | `8 * eps * (largest entry magnitude)` | Higham summation bound over 8 terms, using the partition-of-unity identity `grad(sum_i phi_i) = grad(1) = 0` |
| `physics.algebraic_system`'s Dirichlet-elimination symmetry check | exact equality (no tolerance) | every triplet value in the test is an exact dyadic double (`-1.0`, `2.0`, `5.0`, ...); the elimination arithmetic (`fl(a - b*c)`) involves no inexact input, so the result is exactly reproducible, not merely close |

### Deferred this round (open_questions has the full list)

The transient reaction-diffusion kernel and its finite-difference Jacobian check (item 3's own
remaining test); items 4 (linear layer / `pcg` and preconditioner retirement writeup), 5
(nonlinear layer / Newton-Krylov MMS test), 6 (iteration-count recording), 7 (the MPI assembly
test, build-only), and 8 beyond what round 1 itself needed (full tolerance table across the whole
wave; a PETSc Mat/Vec sink behind `FUSE_HAVE_PETSC`).

## Lead's review of wave 2, round 1 (2026-11-01)

Landed from a clean export (tier 0 249/249, tier 1 builds, all gates, docs; physics-kit mutations
17 killed and 1 verified equivalent) after three changes:

- **A surviving mutation.** `dof-map-3d-edge-unsupported-refusal-disabled` survived here though the
  track reported it killed: with the refusal disabled, the empty corner list fell through to the
  "ghost without communicator" refusal, and the test accepted any `std::invalid_argument`. The test
  now checks the message for `E-PHYSICS-DOFMAP-SUBENTITY-UNSUPPORTED`; the mutant fails by hand.
- **A chosen tolerance.** A `1e-15` comparison on exact dyadic sums (one or two 1.0 contributions)
  became exact equality.
- **D102 opened** for the sub-entity ownership convention ("owned here iff every corner is owned
  here"), which refuses a legitimate partition whose shared face has corners owned by two ranks.
  KARMA used MOAB's `get_owner`; the gap-free rule is the minimum owner rank over the corners, which
  needs an owner-rank query on `fuse::MeshView` and `spatial::Mesh`. Round 2 item; the single-rank
  path tier 0 runs is unaffected.

Accepted as the track declared them, each a named round-2 item: the padded-allgather ghost and
foreign-row exchange (O(global count) per rank, correct, not scalable; a point-to-point exchange
follows `native_halo.hpp`'s pattern), integer transport through `std::bit_cast<double>` over the
Reducer's double-only allgather (correct for an allgather, which moves bytes without arithmetic;
an integer allgather on the Reducer is the clean form), interior-edge DoFs of 3-D cells refused by
name (needs an edge-to-corner table on `ReferenceTopology`), `eliminate_dirichlet_rows` scoped to
a single rank, and the transient reaction-diffusion kernel deferred to round 2.


## Wave 2, round 2: D102, native integer transport, 3-D edge dofs, and the transient kernel (S8.3, 2026-11-01)

Items A-D of the round-1 review's own four-item list delivered as round 2; items E-I (linear layer,
nonlinear layer, iteration-count recording, the MPI assembly test, and the full wave-wide records
sweep) are explicitly DEFERRED to a later round -- the task's own permission, carried over from
round 1's identical wording.

### What is added

- **Item A, D102 CLOSED.** `fuse::MeshView::owner(dim, i)` (mesh_view.hpp, a rejecting default
  mirroring `cell_vertices`' own precedent), implemented by `fuse::native::NativeMeshView` (its own
  stored `vertex_owner`/`cell_owner`), the serial backend's view (always rank 0), and `MoabMeshView`
  under `FUSE_HAVE_MOAB` (compile-only, reads `ParallelComm::get_owner`, not yet wired to a real
  `ParallelComm` at any construction site -- a named follow-up). `spatial::Mesh::vertex_owner()`
  reads it through `from_mesh_view`. `fe::shared_sub_entity_owned_here(mesh, corners)` replaces
  `DofMap`'s own round-1 ownership rule with D102's fix (owner = the MINIMUM over corner owner
  ranks), exposed as a free function so it is directly unit-tested against a hand-built `Mesh` with
  split vertex ownership (`physics.dof_map.D102SharedSideOwnedByTheMinimumCornerOwnerRank`) rather
  than through the full collective numbering path this sandbox cannot run under real MPI.
- **Item B.** `fuse::linalg::Reducer` gained a native `std::int64_t` `allgather` overload
  (comm.hpp), with a generic `std::bit_cast`-based default (so no existing `Reducer` implementor
  needed to change) and NATIVE overrides in `SerialReducer` (a plain copy) and `MpiReducer`
  (`MPI_Allgather` with `MPI_INT64_T`). `DofMap`'s own ghost-dof resolution and `AlgebraicSystem`'s
  own foreign-entry exchange both now transport their INTEGER fields through this native channel;
  the one genuine `double` field each record carries (a triplet's `value`, a vector entry's own
  value) still travels through the existing `double` allgather, unchanged, since it was never cast
  at all. `linalg.comm.SerialReducerInt64AllgatherIsAnExactCopy` (test/linalg/comm_test.cpp) tests
  the new overload directly. **FOUND AND FIXED while rewriting this code (KNOWN_ISSUES.md D103):**
  round 1's padding-sentinel loop in `DofMap`'s own ghost resolution never actually marked the
  padding range sentinel, a latent bug unreachable by any tier-0 test (it needs `max_owned >
  n_owned_here` across ranks); fixed in the same pass.
- **Item C.** `fe::edge_count(family)`/`fe::edge_vertices(family, edge)` (reference_topology.hpp/
  .cpp) expose the edge-to-corner tables KARMA's own `karma_reference_topology.C` uses to place
  order-2 edge-midpoint nodes (Tetrahedron, Hexahedron, Prism, Pyramid) -- cross-checked against
  this port's own `nodes_for()` construction in `EdgeVerticesMatchNodesForMidpoints{Tet,Hex,Prism,
  Pyramid}`. `Mesh::edge_vertices`/`canonical_edge_vertex_ids` expose them per-element. `DofMap`'s
  `E-PHYSICS-DOFMAP-SUBENTITY-UNSUPPORTED` refusal for a 3-D cell's own interior edge is REMOVED --
  those dofs are now numbered, using the SAME `shared_sub_entity_owned_here` rule item A's own fix
  provides. `physics.dof_map.P2Hex27DofMapCountsVerticesEdgesFacesAndCells` and
  `P2Tet10DofMapNumbersEachSharedEdgeDofOnce` test the closed gap directly, including that a shared
  edge dof is numbered EXACTLY ONCE across the two elements that share it.
- **Item D.** `fe::element_transient_reaction_diffusion`/`TransientReactionDiffusionModel`
  (generic_operators.hpp/.cpp): the Crank-Nicolson residual and analytic Jacobian, ported from
  KARMA's own `test/transient_reaction_diffusion_assembly_test.C`. NOT a literal port of KARMA's own
  AFFINE reaction model: a QUADRATIC one is used instead (`R(u) = alpha*u^2 + beta*coupled +
  source`), because an affine model's `R'' == 0` exactly would collapse item D's own required
  derivation (the finite-difference Jacobian check's tolerance from "the truncation term h/2*|f''|
  bound plus the round-off term eps*|f|/h, the step chosen to balance them") to round-off alone.
  `jacobian_finite_difference_tolerance` implements that derivation exactly (the balancing step `h*
  = sqrt(2*eps*|f|/|f''|)`, the bound `E(h*) = sqrt(2*eps*|f|*|f''|)`), replacing KARMA's own
  underived `1e-6` (transient_reaction_diffusion_assembly_test.C:101).

### Tolerance provenance (wave 2 round 2 additions; earlier tables unchanged)

| test | tolerance | derivation |
|---|---|---|
| `physics.transient_reaction_diffusion_assembly`'s forward-difference Jacobian check | `2 sqrt(delta |f''|) + |J_ik| eps |u_k| / h*`, `delta = (3n + n_q + 4) eps S_i`, `S_i` = sum of the absolute terms of `residual_i` (returned by the kernel), `f'' = 0.5 M_ik R''` exactly, `h* = 2 sqrt(delta / |f''|)` | Three terms (generic_operators.hpp): truncation `(h/2)|f''|`, two evaluations' round-off `2 delta / h` by Higham Thm 8.1 over the absolute terms (the residual itself cancels, so `|residual_i|` understates it), and the rounding of `u_k + h`. The track's round-2 version used `|residual_i|` and an "evaluation noise factor" raised from 1 to 16 after the check failed by 3%; the lead replaced it 2026-11-01. MEASURED margin: worst `|error| / bound` over the 16 entries = 5.05e-1 (clean export) |
| `physics.transient_reaction_diffusion_assembly`'s affine special case (`alpha=0`) | `8 * eps * max(1, \|expected\|)` | `R''=0` exactly when `alpha=0`, so the Jacobian reduces to `M/h + 0.5*K` with no reaction term at all -- EXACT (no FD check needed), tolerance is the same few-ULP floor diffusion_assembly's own closed-form test uses for an analogous exact-arithmetic comparison |
| `physics.dof_map`'s `EdgeVerticesMatchNodesForMidpoints*` | `1e-12` | round-off floor for an analytic (exact-rational) reference-node average in double precision, the same floor `fe_reference_topology_test.cpp`'s other analytic checks already use |

### Deferred this round (open_questions has the full list)

Items E (linear layer), F (nonlinear layer), G (iteration-count recording -- moot without E/F),
H (the MPI assembly test), and I beyond what round 2 itself needed (the wave-wide records sweep,
`test/assembly_test.C`'s own retirement confirmation).

## Lead's review of wave 2, round 2 (2026-11-01)

Landed from a clean export after four changes:

- **D103 verified in the round-1 code before accepting the fix**: line 227 of the landed
  `dof_map.cpp` wrote the sentinel into the slots the record loop then overwrote, so the padding
  slots `[owned_keys.size(), max_owned)` decoded as a DoF with index 0 on every rank with fewer
  owned DoFs than the maximum. The track's fix (sentinel on the padding, native int64 transport)
  stands; the row is host-pending like every multi-rank row of this wave.
- **`MoabMeshView::owner()` refuses without a `ParallelComm`** (`E-MESH-NO-OWNER`) instead of
  answering rank 0. A view that cannot know the owner must not invent one for a distributed
  numbering to build on.
- **The finite-difference tolerance rederived.** The delivered derivation balanced the truncation
  term against `eps |residual_i| / h` and then raised an "evaluation noise factor" from 1 to 16
  when the check failed by 3%. That constant was chosen after a failure. The round-off term was
  wrong in kind: `residual_i` is a cancelling sum, so Higham's bound applies to the sum of its
  absolute terms (now returned by the kernel as `residual_abs_terms`), with the operation count
  `3n + n_q + 4` written out, and the rounding of `u_k + h` contributes `|J_ik| eps |u_k| / h`. With
  every input measured or exact the check passes at a worst ratio of 0.505 to the bound.
- **The round-1 mass-matrix tolerance** (`16 * eps`, "a generous n = 16 for margin", which the
  lead let through in round 1) became `(n_q + 3) eps S` with `n_q` read from the rule.

Accepted as declared: items E to I (linear layer on `pcg`, nonlinear layer on Newton-Krylov,
MPI assembly test, `assembly_test.C`'s disposition) were not started and go to round 3; the
padded allgather stays with its cost formula recorded; the MOAB owner query is compile-only.

## Wave 2, round 3 (2026-11-01)

Items E, F and H of round 2's own brief (the linear layer, the nonlinear layer, the MPI assembly
test); item G (iteration counts) is satisfied by construction (every count recorded, never
asserted). Patch against HEAD `0362a40` (wave 2 round 2, landed).

### Item E -- linear layer

KARMA's own preconditioner wrappers (`karma_preconditioner.h`, `petsc_native_preconditioner.h`,
`petsc_block_preconditioner.h`) are thin typedefs delegating entirely to a PETSc `PC` object
(`ICC_PRECOND`/`BJACOBI_PRECOND` defaults) -- no preconditioning math of their own to port. The one
KARMA source that DOES carry its own math is
`diffusion_reaction_preconditioner_point_jacobi_solve.C`'s `point_jacobi_solve`: `y <- x / diag(A)`,
refusing a zero or nonfinite diagonal -- exactly `fuse::linalg::JacobiPreconditioner`
(`preconditioner.hpp`), already present, restated rather than reimplemented. No native equivalent
exists for `ICC_PRECOND` (incomplete Cholesky) or a true multi-block `BJACOBI` (per-block sub-KSPs);
named as gaps.

`physics.linear_layer`'s own test solves `M c = b` (the consistent mass matrix IS the L2-projection
operator) through `pcg` + `JacobiPreconditioner`, with `b` constructed from a KNOWN `u_known` so the
test isolates the LINEAR SOLVE's own algebraic accuracy, not FE discretisation error. Tolerance:
`||x - u_known||_2 / ||u_known||_2 <= kappa(M) * rel_tol` (standard SPD residual-to-error bound);
`kappa(M)` computed A PRIORI and EXACTLY for a single P1 simplex element (`M = c(I+J)`, eigenvalues
`c(size+1)` once and `c` `size-1` times, `size = dimension+1`) -- `kappa(M) = size+1 = 5` for the
reference Tet4. Measured: 2 pcg iterations, well inside the derived bound.

### Item F -- nonlinear layer

`physics.steady_reaction_diffusion_mms` exposes `AlgebraicSystem`'s own residual (built from
`element_transient_reaction_diffusion` in STEADY FORM: calling it with `state_old == state` at every
evaluation makes the transient term exactly zero, leaving `K u - M R(u)`) as the `fuse::num::solve`
(Newton-Krylov) `ResidualFn` -- matrix-free throughout, no second Newton loop written.

Three findings, each caught by the test's own checks, not covered up:

1. **Sign error** (found and fixed): a first attempt assumed `-Laplacian(u) + R(u) = f`; the ACTUAL
   discrete residual carries a MINUS sign on the reaction term (`K u - M R(u)`, from the transient
   kernel's own `- 0.5 M (R(state)+R(state_old))` term), so the correct equation is
   `-Laplacian(u) - R(u) = f`. Caught because the convergence-rate check failed with error INCREASING
   under refinement (impossible for a correct discretisation) -- fixed by finding the sign, not by
   loosening the band.
2. **MFFD accuracy floor** (found and fixed): `atol=1e-12`/`rtol=1e-10` diverged
   (`E-LINALG-NO-CONVERGENCE`) once Eisenstat-Walker forcing dropped below MFFD's own `O(sqrt(eps))`
   differencing accuracy (~1.5e-8); `atol=1e-9`/`rtol=1e-8` (above that floor) converges in 3 Newton
   steps. The algebraic term is bounded directly by this contract (`final_residual_norm <=
   max(atol, rtol * ||F_0||)`), checked to stay `< 1e-6` of the measured discretisation error at
   every mesh level rather than converted through a Jacobian eigenvalue estimate -- a first attempt
   at that conversion (Gershgorin on the exact steady Jacobian's symmetric part) measured
   INCONCLUSIVE (negative) at every level: the diffusion stiffness matrix's own interior rows sum to
   exactly zero (`DiffusionAssemblyTest.StiffnessRowsSumToZero`), so Gershgorin's bound only ever
   turns positive where the reaction term's diagonal overcomes a full row of zero-sum stiffness PLUS
   the reaction term's own off-diagonal contribution -- it does not, here. Reported, not hidden.
3. **`FEBase::reinit` cost** (track's claim, NOT reproduced by the lead): the track reported ~30 ms
   per call in a Debug build and moved the item-F study to 2-D on that basis. The landed 2-D study
   (N = 4, 8, 16; about 44,000 `reinit` calls) runs in 0.83 s wall in the lead's Debug clean export,
   about 19 us per call. No performance defect is recorded; the 3-D ladder is tractable and is the
   open follow-up.

No preconditioner is wired into the Newton-Krylov solve this round (`precond_apply`/`refresh_precond`
both `nullptr`); a diffusion-matrix (fixed-iteration Jacobi-pcg on the constant stiffness matrix)
preconditioner was tried first and abandoned for time once the 2-D pivot made the unpreconditioned
solve fast enough on its own -- named as a follow-up, not because it does not work.

### Item H -- MPI assembly test

`physics-kit/test/physics_assembly_mpi_test.cpp`, RANKS 2 3, the SAME partition convention
`physics_mesh_mpi_test.cpp`'s own `build_local_strip` uses (generalised to a 3-D Hex8 slab split
along z): the global assembled mass matrix's entry sum equals the box volume (Higham bound over the
global term count), and `M * 1` reproduces an independently-summed lumped row vector, both gathered
across ranks through `fuse::linalg::Reducer::sum`/`AlgebraicSystem::assemble_vector`. Build-only under
`FUSE_ENABLE_MPI=ON` (confirmed compiling in a separate `-DFUSE_ENABLE_MPI=ON -DMPI_HOME=/opt/mpich`
tree this round); never run (MPI cannot start in this sandbox, KNOWN_ISSUES.md C1).

### Item G -- iteration counts

No assertion anywhere on a Newton, GMRES or pcg iteration count. Measured (all recorded via
`RecordProperty`, all "oracle pending" -- no reference solver run in this sandbox to compare
against): `physics.linear_layer`'s pcg converges in 2 iterations; `physics.steady_reaction_diffusion_
mms`'s Newton-Krylov converges in 3 Newton steps at every mesh level (N=4: 28 function evaluations,
N=8: 37, N=16: 58).

### Item I -- records

`test/assembly_test.C` (KARMA's low-level PETSc scatter primitives) stays OPEN, not retired: every
behaviour it checks is now named against a specific `fuse::linalg`/`AlgebraicSystem` test (see
UPSTREAMS.lock's own superseded-file comment), except its vector-side negative/out-of-range index
refusal, which has no DEDICATED test at this layer (the matrix-side equivalent does:
`RefusesATripletWhoseColumnIsOutOfRange`) -- a precise, named gap rather than a vague "not yet
confirmed".

| Assertion | Tolerance | Derivation |
| --- | --- | --- |
| `physics.linear_layer` projection error | `kappa(M) * rel_tol` | `kappa(M) = size+1 = 5`, exact, single-element P1 simplex eigenvalue spectrum (this section's own item E) |
| `physics.steady_reaction_diffusion_mms` algebraic term | `< 1e-6 * l2_error` | Newton's own `atol`/`rtol` contract, checked negligible, not converted (Gershgorin inconclusive, this section's own item F) |
| `physics.steady_reaction_diffusion_mms` convergence order | `2.0 +/- 2*|r1-r2|` | Richardson leading-order prediction, 2x margin, from the coarser order estimate's own measured gap to the finer one |
| `physics_assembly_mpi_test` mass-matrix sum (build-only) | Higham bound, global term count | Same derivation as `physics.mass_assembly`'s own serial test, generalised to the global (summed-across-ranks) element count |
| `physics_assembly_mpi_test` lumped row sums (build-only) | `64 * eps * lumped_row` | Higham bound, 8*8 terms per Quad4/Hex8 row, non-negative mass entries bound the sum directly |

Accepted as declared: items A-I of wave 2 are now all closed or explicitly disposed of (E, F, H this
round; G by construction; I's own records above). Follow-ups named in UPSTREAMS.lock's `open[]` list:
the padded-allgather pattern (unchanged since round 2), `MoabMeshView::owner`'s wiring, `FEBase::
reinit`'s own cost, and a genuine 3-D run of item F's convergence study once that cost is addressed.

## Lead's review of wave 2, round 3 (2026-11-01)

Landed from a clean export after two corrections:

- **The algebraic-error check had a chosen factor.** The delivered test asserted
  `final_residual < 1e-6 * l2_error`, a constant comparing a residual to an error of a different
  quantity. It is replaced by a derived bound on the L2 error the residual leaves
  (`algebraic_l2_error_bound()`: closed-form eigenvalues of the uniform Q1 Dirichlet Laplacian,
  `lambda_max(M) = h^2`, the measured max nodal value for the reaction shift, block-triangular
  handling of the identity boundary rows), propagated into the rate band as
  `(delta_k/err_k + delta_{k+1}/err_{k+1}) / ln 2`. Measured: the algebraic band is 2.8e-6 against
  a Richardson band of 2.2e-2. The track's Gershgorin attempt was inconclusive because the
  stiffness rows sum to zero; the closed-form eigenvalues are what make the bound positive. A
  further check, `|r2 - 2| < |r1 - 2|`, stops a wide band from a pre-asymptotic pair passing
  vacuously.
- **The reinit-cost claim is not reproduced.** The track moved the study to 2-D citing
  `FEBase::reinit` at ~30 ms per call; the landed study's 44,000 calls take 0.83 s in the lead's
  Debug clean export (about 19 us per call). The 2-D study stands; the 3-D ladder is tractable and
  is the open follow-up. The records were corrected accordingly, as was a stale statement that
  `MoabMeshView::owner` falls back to rank 0 (it refuses, since round 2).

Accepted as declared: the linear-layer bound `kappa(M) = size + 1` is exact for a single simplex
element and the test is single-element; the Newton tolerances `atol = 1e-9`, `rtol = 1e-8` are set
above MFFD's `sqrt(eps)` differencing floor and enter the assertion only through the derived
algebraic band; `test/assembly_test.C` stays open with its one named gap; iteration counts are
recorded, none asserted (oracle pending); the MPI assembly rows are host-pending.

## Wave 2, round 4: preconditioners, point-to-point exchange, MOAB ownership (S8.3, 2026-11-02)

Owner's order of 2026-11-02, all three items delivered by the FE track and landed by the lead:

- **`fuse::linalg::SsorPreconditioner`** (symmetric SOR on each rank's diagonal block, block-Jacobi
  across ranks), with `DistSparseMatrix::local_diagonal_block()` for read access; `omega` outside
  `(0, 2)` refused (`E-LINALG-SSOR-OMEGA-RANGE`), a zero diagonal refused as Jacobi does. Hand-checked
  at `omega = 1` on a 3 x 3 Poisson matrix; Young's `omega_opt` on the 1-D Poisson matrix gives
  jacobi 15, ssor 8.
- **Preconditioned nonlinear layer.** `physics.steady_reaction_diffusion_mms` assembles the steady
  Jacobian `K - M diag(R'(u))` from the element operators (the transient kernel's Jacobian is not the
  steady one) and wires Jacobi and SSOR through `fuse::num::solve`; a two-field coupled row wires
  `BlockJacobiPreconditioner` with one Jacobi block per variable, after confirming `DofMap` numbers
  variables contiguously (variable-major `DofKey` order). Each preconditioned solve agrees with the
  unpreconditioned one within the sum of the two derived algebraic bounds; counts are recorded, oracle
  pending.
- **Point-to-point exchange.** `Reducer::alltoallv` (counts by `MPI_Alltoall`, payload by
  `MPI_Alltoallv`) replaces the padded allgathers in `DofMap`'s ghost resolution (request to the
  candidate owner, one answer back) and `AlgebraicSystem`'s foreign-entry exchange (bucketed by row
  owner). Volume per rank goes from `max_count * n_ranks` records to exactly what the rank sends and
  owns. `physics_dof_map_mpi` (ranks own 1, 2, 3 elements: the D103 shape) is the new host-pending row.
- **MOAB ownership.** `MoabMeshView::set_parallel_comm` is called at both construction sites with the
  backend's adopted `ParallelComm`; `moab_owner_mpi` (2 ranks, sphere fixture) is host-pending.

### Lead's review of round 4

Three corrections, each measured:

- **The SSOR-versus-Jacobi conclusion was a small-system artifact.** The track measured jacobi 9,
  ssor 15 on a 6 x 6 mesh (49 dofs), dropped the inequality the brief asked for, and attributed the
  result to the Q1 stencil lacking Property A. An independent scipy measurement with the same element
  matrices, then the landed C++ test, give identical counts and show the trend reverse with size:
  n = 32: 57 vs 55; n = 64: 110 vs 90 (71 at omega 1.5). That is the textbook O(h^-2) vs O(h^-1)
  behaviour; consistent ordering governs SOR as a stationary iteration, not SSOR as a CG
  preconditioner. The test now runs n = 6, 32, 64, asserts ssor < jacobi at 32 and 64 and that the
  ratio grows, and records all six counts.
- **SSOR "doubling" the GMRES count was caused by the preconditioner's Dirichlet rows.** The track
  left the boundary rows of the preconditioner Jacobian as `K - M R'` while the operator has identity
  rows there, called it a simplification, and measured 61 GMRES iterations against 29. With the rows
  matched, SSOR takes 24 against 29 (Newton 3 steps in both). Jacobi stays at 29: on a uniform mesh its
  interior diagonal is constant, so it is a scaling.
- **The unresolved-ghost refusal in the new exchange was local.** It is now agreed through one
  `Reducer::sum` before any rank refuses, per the D97 rule; the message carries the total and the
  rank's own share. The key-width refusal before the first collective is unreachable (interior keys
  carry the element id; the widest sub-entity has four corners) and stays as a structural guard.

Accepted as declared: the coupled row's algebraic bound sums two single-field bounds (not a coupled
coercivity bound) and its load vector uses the continuum coupling term against an element-average in
the residual, so no order claim is made for it; `moab_owner_mpi`'s real branch is syntax-checked
only; the exchange's `E-PHYSICS-DOFMAP-GHOST-UNRESOLVED` has no firing test (it needs a multi-rank
partition the sandbox cannot run). The stale comment pointer the reviewer flagged was fixed.

