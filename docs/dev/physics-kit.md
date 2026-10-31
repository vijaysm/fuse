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
