# physics-kit: finite-element building blocks

physics-kit is an optional library on top of FUSE for the part of a coupled simulation that is a
finite-element discretisation: reference cells, quadrature, bases, meshes of cells, global
degree-of-freedom numbering and assembly of element matrices into the distributed matrices and
vectors of [](linear-algebra.md). This page is for someone who wants to write a finite-element
component without writing the element machinery, and for anyone deciding whether the library fits
their problem.

It is a toolbox, not a solver. It provides the pieces below and no ready-made physics: there is no
heat-conduction or flow module to switch on, and it defines no `fuse::Component`. You write the
component, as in [](component-interface.md), and call physics-kit inside its `advance()`. FUSE
itself never depends on physics-kit, so a code that already has its own discretisation is unaffected.

## Build it and link it

physics-kit needs nothing beyond the base build (Kokkos and fmt), so `FUSE_ENABLE_PHYSICS_KIT`
defaults to `ON`. PETSc is not required. Its headers are under `fuse_physics/` and its code is in
the namespace `fuse::physics`.

```bash
cmake -S . -B build -DFUSE_ENABLE_PHYSICS_KIT=ON     # the default
cmake --build build -j
ctest --test-dir build -R "^physics\." --output-on-failure
```

From a project that uses an installed FUSE:

```cmake
find_package(fuse-coupler CONFIG REQUIRED COMPONENTS physics_kit)
target_link_libraries(my_fem_app PRIVATE fuse::physics_kit)
```

The request fails at configure time against an install built with `FUSE_ENABLE_PHYSICS_KIT=OFF`;
`OPTIONAL_COMPONENTS physics_kit` instead sets `fuse-coupler_physics_kit_FOUND` to false. Linking
`fuse::physics_kit` brings `fuse::runtime`, `fuse::backend`, `fuse::num` and `fuse::linalg` with it.
`fuse::physics::version()` and `fuse::physics::has_petsc()` (`fuse_physics/version.hpp`) report the
FUSE version it was built against and whether the build had PETSc.

## A worked example: Poisson on a square

The program solves `-laplace(u) = f` on the unit square with `u = 0` on the boundary and bilinear
(Q1) elements, for the exact solution `u = x (1 - x) y (1 - y)`. It builds a mesh, numbers the
degrees of freedom, assembles the stiffness matrix and the load vector from the element kernels,
solves with preconditioned conjugate gradients, and reports the largest nodal error on four meshes.
The boundary rows are replaced by the identity, which is exact here because the boundary value is
zero. It compiles and runs as shown.

```cpp
// Poisson on the unit square with Q1 elements: -laplace(u) = f, u = 0 on the boundary,
// exact u = x (1 - x) y (1 - y), so f = 2 (x (1 - x) + y (1 - y)).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/base/error.hpp"
#include "fuse/linalg/pcg.hpp"
#include "fuse/linalg/preconditioner.hpp"
#include "fuse_physics/fe/algebraic_system.hpp"
#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/fe/generic_operators.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace fe = fuse::physics::fe;
using fuse::physics::Point;
using fuse::physics::spatial::EntityHandle;
using fuse::linalg::Triplet;

double u_exact(const Point& p) { return p.x * (1.0 - p.x) * p.y * (1.0 - p.y); }
double source(const Point& p) { return 2.0 * (p.x * (1.0 - p.x) + p.y * (1.0 - p.y)); }

double solve(unsigned n, std::size_t& iterations) {
  fuse::physics::spatial::Mesh mesh(2);
  mesh.generate_2d_box(n, n, Point(0, 0), Point(1, 1));

  fe::BasisRegistry registry;
  fe::Variable u{"u", fe::FieldBasisFamily::LagrangeNodal, 1, fe::BasisContinuity::cG, 1};
  fe::DofMap dofs(mesh, {u}, registry);
  fe::AlgebraicSystem system(mesh, dofs);

  const auto basis = registry.create(
      {fe::CellFamily::Quadrilateral, fe::FieldBasisFamily::LagrangeNodal, 1, fe::BasisContinuity::cG});
  fe::FEBase element(mesh, fe::GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);

  // Vertex handles are 1-based; a Q1 dof sits on each vertex. Find the boundary dofs.
  std::vector<std::int64_t> vertex_dof(mesh.n_vertices() + 1, -1);
  std::vector<std::int64_t> idx;
  for (EntityHandle e = 1; e <= mesh.n_elements(); ++e) {
    dofs.dof_indices(e, idx);
    const auto& corners = mesh.element_connectivity(e);
    for (std::size_t k = 0; k < corners.size(); ++k) vertex_dof[corners[k]] = idx[k];
  }
  std::vector<char> boundary(static_cast<std::size_t>(dofs.n_dofs()), 0);
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    const Point& p = mesh.vertex_positions()[v];
    if (p.x == 0.0 || p.x == 1.0 || p.y == 0.0 || p.y == 1.0) boundary[vertex_dof[v + 1]] = 1;
  }

  // Stiffness matrix with the (zero) Dirichlet rows and columns replaced by the identity.
  std::vector<char> identity_done(boundary.size(), 0);
  auto K = system.assemble_matrix(
      [&](EntityHandle e, const std::vector<std::int64_t>& rows, std::vector<Triplet>& out) {
        const auto k = fe::element_stiffness_matrix(element, e);
        const std::size_t m = rows.size();
        for (std::size_t i = 0; i < m; ++i) {
          if (boundary[rows[i]]) {
            if (!identity_done[rows[i]]) {   // duplicates are summed: emit the 1 once
              identity_done[rows[i]] = 1;
              out.push_back(Triplet{rows[i], rows[i], 1.0});
            }
            continue;
          }
          for (std::size_t j = 0; j < m; ++j)
            if (!boundary[rows[j]]) out.push_back(Triplet{rows[i], rows[j], k[i * m + j]});
        }
      });
  if (!K.ok()) { std::printf("%s\n", K.error().message.c_str()); return -1.0; }

  auto b = system.new_vector();
  b.set_zero();
  auto rhs = system.assemble_vector(
      [&](EntityHandle e, const std::vector<std::int64_t>& rows,
          std::vector<std::pair<std::int64_t, double>>& out) {
        const auto f = fe::element_load_vector(element, e, source);
        for (std::size_t i = 0; i < rows.size(); ++i)
          if (!boundary[rows[i]]) out.emplace_back(rows[i], f[i]);
      },
      b);
  if (!rhs.ok()) { std::printf("%s\n", rhs.error().message.c_str()); return -1.0; }

  auto jacobi = fuse::linalg::JacobiPreconditioner::from_matrix(K.value());
  auto x = system.new_vector();
  x.set_zero();
  fuse::linalg::PcgOptions opt;
  opt.rel_tol = 1e-12;
  auto report = fuse::linalg::pcg(K.value(), jacobi.value(), x, b, opt);
  if (!report.ok() || !report.value().converged) { std::printf("pcg failed\n"); return -1.0; }
  iterations = report.value().iterations;

  double max_error = 0.0;
  for (std::size_t v = 0; v < mesh.n_vertices(); ++v) {
    const Point& p = mesh.vertex_positions()[v];
    max_error = std::fmax(max_error, std::fabs(x.data()[vertex_dof[v + 1]] - u_exact(p)));
  }
  return max_error;
}

int main(int argc, char** argv) {
  fuse::initialize();
  Kokkos::ScopeGuard kokkos(argc, argv);
  double previous = 0.0;
  for (unsigned n : {8u, 16u, 32u, 64u}) {
    std::size_t it = 0;
    const double err = solve(n, it);
    std::printf("n=%2u  max nodal error %.3e  pcg iterations %zu", n, err, it);
    if (previous > 0.0) std::printf("  ratio %.2f", previous / err);
    std::printf("\n");
    previous = err;
  }
}
```

Expected output (exact digits can vary by platform):

```text
n= 8  max nodal error 7.771e-04  pcg iterations 10
n=16  max nodal error 1.924e-04  pcg iterations 22  ratio 4.04
n=32  max nodal error 4.800e-05  pcg iterations 46  ratio 4.01
n=64  max nodal error 1.199e-05  pcg iterations 93  ratio 4.00
```

The error falls by a factor of four each time the mesh is halved, which is the second-order
convergence of Q1 elements. The pieces the program uses are described next.

## Element-level quantities

`FEBase` evaluates a basis on one cell at the quadrature points: shape functions `get_phi()`,
their physical gradients `get_dphi()`, the weights times the Jacobian determinant `get_JxW()`,
the physical points `get_xyz()` and, on a face, the outward normals `get_normals()`. Construct it with
`FEIntegrationDomain::Volume` (the default) to integrate over cells or `FEIntegrationDomain::Face` to
integrate over a side, and call `reinit(element)` or `reinit(element, side)` before reading. The
program below sums `JxW` over every cell of a `2 x 1` rectangle for its area, and over every boundary
side for its perimeter:

```cpp
#include <cstdio>
#include <vector>
#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/fe/generic_operators.hpp"
#include "fuse_physics/spatial/mesh.hpp"
#include "fuse_physics/spatial/mesh_utils.hpp"

namespace fe = fuse::physics::fe;
using fuse::physics::Point;

int main() {
  fuse::physics::spatial::Mesh mesh(2);
  mesh.generate_2d_box(4, 4, Point(0, 0), Point(2, 1));          // 2 x 1 rectangle, 16 Quad4

  fe::BasisRegistry registry;
  const auto basis = registry.create(
      {fe::CellFamily::Quadrilateral, fe::FieldBasisFamily::LagrangeNodal, 1, fe::BasisContinuity::cG});

  fe::FEBase cell(mesh, fe::GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);
  double area = 0.0;
  for (std::size_t e = 1; e <= mesh.n_elements(); ++e) {           // handles are 1-based
    cell.reinit(e);
    for (const double w : cell.get_JxW()) area += w;
  }

  fe::FEBase face(mesh, fe::GeometryVariant::Quad4, basis, 2, fe::FEIntegrationDomain::Face);
  double perimeter = 0.0;
  for (std::size_t e = 1; e <= mesh.n_elements(); ++e)
    for (unsigned side = 0; side < mesh.n_sides(e); ++side) {
      if (!mesh.is_boundary_side(e, side)) continue;
      face.reinit(e, side);
      for (const double w : face.get_JxW()) perimeter += w;
    }

  const auto boundary = fuse::physics::spatial::extract_boundary_faces(mesh);
  const auto k = fe::element_stiffness_matrix(cell, 1);
  double row_sum = 0.0;
  for (int j = 0; j < 4; ++j) row_sum += k[j];
  std::printf("area %.12g  perimeter %.12g  boundary sides %zu  stiffness row sum %.1e\n", area, perimeter, boundary.size(), row_sum);
}
```

```text
area 2  perimeter 6  boundary sides 16  stiffness row sum -2.2e-16
```

## What is in the library

| Header (`fuse_physics/`) | Provides |
|---|---|
| `geometry.hpp` | `Point`, `Vector3` (an alias of `Point`) and `Matrix3`: small plain structs with the dense operations the cell maps need. They are not a general linear-algebra library; use `fuse::linalg` for that. |
| `fe/reference_topology.hpp` | `GeometryVariant` (`Edge2`, `Edge3`, `Tri3`, `Tri6`, `Quad4`, `Quad8`, `Quad9`, `Tet4`, `Tet10`, `Hex8`, `Hex27`, `Prism6`, `Prism15`, `Prism18`, `Pyramid5`, `Pyramid13`, `Pyramid14`), `CellFamily`, and `ReferenceTopology`: reference nodes, shape functions and their derivatives, and the sides of each cell. |
| `fe/quadrature.hpp` | `Quadrature(family, degree)`: Gauss-type rules for every cell family, exact to the polynomial degree you ask for (0 to 64). The rules are generated, not looked up, and need no PETSc. |
| `fe/basis.hpp` | `Basis`, `BasisKey` and `BasisRegistry`. The registry holds Lagrange nodal bases of order 0, 1 and 2 on every cell family; `register_factory()` adds your own. `OrientationTransform` handles the re-signing of degrees of freedom shared between cells. |
| `fe/geometry_map.hpp` | `GeometryMap`: the isoparametric map from reference to physical cell, its Jacobian and metric, the inverse map (`try_inverse_map()` reports `Located`, `NotContained` or `InvalidGeometry`), and the geometry of a side. |
| `fe/fe_base.hpp` | `FEBase`, the evaluator described above. |
| `spatial/domain.hpp` | `Domain`: a box-shaped domain descriptor (`Shape::Edge`, `Square`, `Cube`; `Hexagon` and `Unstructured` are refused). |
| `spatial/mesh.hpp` | `Mesh`: vertex positions, per-cell connectivity and kind, ownership and global ids, side and edge queries, `neighbor()` and `is_boundary_side()`. Built with `generate_box()` (3-D, `Hex8`), `generate_2d_box()` (2-D, `Quad4`), `add_vertex()` / `add_element()`, or `from_mesh_view()` from any `fuse::MeshView`, so a FUSE mesh backend feeds it directly. |
| `spatial/mesh_utils.hpp` | `extract_boundary_faces()` and `extract_edge_at_coordinate()` (the 1-D boundary of a 2-D mesh on a line `x = value` or `y = value`). |
| `fe/dof_map.hpp` | `DofMap`: global numbering of the degrees of freedom of one or more `Variable`s (name, basis family, order, continuity, components) over a mesh, including the degrees of freedom on shared edges, faces and cell interiors. `dof_indices(element, indices)` gives an element's global indices. |
| `fe/algebraic_system.hpp` | `AlgebraicSystem`: `assemble_matrix()` and `assemble_vector()` take a callback per element and return a `fuse::linalg::DistSparseMatrix` or fill a `DistVector`; `eliminate_dirichlet_rows()` applies symmetric Dirichlet elimination to a triplet list. |
| `fe/generic_operators.hpp` | Element kernels: `element_mass_matrix()`, `element_stiffness_matrix()`, `element_load_vector()` for a source function, and `element_transient_reaction_diffusion()`, the Crank-Nicolson residual and analytic Jacobian of `u_t = laplace(u) + R(u)` for a quadratic reaction `R(u) = alpha u^2 + beta coupled + source`. |

## Conventions to know

- **Handles are 1-based.** `EntityHandle` is a mesh-local index starting at 1; 0 means no entity.
  Loop `for (e = 1; e <= mesh.n_elements(); ++e)`. Global ids (`vertex_global_id()`,
  `element_global_id()`) are separate, and vertices and cells are numbered in separate spaces.
- **Programmer errors throw, assembly returns.** An invalid argument (a basis the registry does not
  hold, an unsupported cell shape, an unrecognised domain shape) throws `std::invalid_argument`.
  `assemble_matrix()` and `assemble_vector()` return a `fuse::Result`, so check `.ok()`.
- **Element kernels reinitialise `FEBase` themselves.** `element_stiffness_matrix(fe, element)` calls
  `fe.reinit(element)`; build the `FEBase` once and reuse it for every element.
- **The mesh holds order-1 cells.** A `Mesh` stores `Edge2`, `Tri3`, `Quad4`, `Tet4`, `Pyramid5`,
  `Prism6` and `Hex8` cells; a higher-order basis puts its extra degrees of freedom on the edges,
  faces and interiors of those cells, and `DofMap` numbers them.
- **Callbacks are per owned element.** `assemble_matrix()` calls your function for each element this
  rank assembles, with the element's global dof indices, and you append `Triplet{row, col, value}`
  entries; duplicates are summed, so a diagonal entry that several elements would add must be emitted
  once, as in the example.

## Running on several ranks

`DofMap` and `AlgebraicSystem` take an optional Fortran communicator handle (`MPI_Comm_c2f`). Without
one they number and assemble on a single process. With one, on an MPI build (`FUSE_ENABLE_MPI=ON`),
the numbering is collective and every rank ends up with the same global numbering. A degree of
freedom on a vertex belongs to the rank that owns the vertex, one on a shared edge or face to the
lowest rank among the owners of its corners, and one inside a cell to the owner of the cell. Each
rank's mesh is expected to hold a one-layer halo of ghost vertices around the cells it owns. Ghost
indices are resolved with one point-to-point exchange, and rows that an element assembles for another
rank are sent to their owner. A rank that fails the collective check agrees the failure with its peers
(the `E-PHYSICS-DOFMAP-*` codes below) instead of leaving them waiting. Build a distributed `Mesh`
with `Mesh::from_mesh_view()` on a partitioned mesh from [](meshes.md). In an MPI build the test
suite has multi-rank tests of the mesh, the numbering and the assembled mass matrix, registered for 2
and 3 ranks; a default build exercises the single-process path.

## Limitations

- **No physics modules.** Only the generic operators above: mass, stiffness, load and the transient
  reaction-diffusion kernel. Anything else is yours to write.
- **Mesh generators are boxes.** `generate_box()` makes `Hex8` cells and `generate_2d_box()` makes
  `Quad4` cells. Other cell kinds come from `add_element()` or `from_mesh_view()`; polygonal and
  polyhedral cells are refused (`E-PHYSICS-MESH-CELL-SHAPE`).
- **Bases.** The built-in bases are Lagrange nodal of order at most 2.
- **Dirichlet elimination is single-rank.** `eliminate_dirichlet_rows()` works on a whole triplet
  list and so on one process; on several ranks, impose the boundary rows inside the element
  callback, as the example does.
- **Domain shapes.** `Domain` refuses `Shape::Hexagon` and `Shape::Unstructured`.

## Error codes

| Code | Meaning |
|---|---|
| `E-PHYSICS-MESH-CELL-SHAPE` | `Mesh::from_mesh_view()` met a cell that has no fixed-node variant (polygon, polyhedron) |
| `E-PHYSICS-DOFMAP-MPI-REQUIRED` | a communicator was given to a build without MPI |
| `E-PHYSICS-DOFMAP-GHOST-WITHOUT-COMMUNICATOR` | the mesh has ghost vertices but no communicator was given |
| `E-PHYSICS-DOFMAP-OWNERSHIP-CONFLICT` | the partition leaves a shared entity without a single owner |
| `E-PHYSICS-DOFMAP-GHOST-UNRESOLVED` | a ghost degree of freedom has no owner among the ranks |
| `E-PHYSICS-DOFMAP-DOF-NOT-VISIBLE` | an index or a `storage()` query names a degree of freedom this rank cannot see |
| `E-PHYSICS-DOFMAP-AGREEMENT-REFUSED`, `E-PHYSICS-DOFMAP-AGREEMENT-COMM-FAILURE` | the ranks disagreed, or the communication that agrees a failure failed |
| `E-PHYSICS-DOFMAP-SUBENTITY-UNSUPPORTED`, `E-PHYSICS-DOFMAP-ORIENTATION-NOT-FOUND`, `E-PHYSICS-DOFMAP-KEY-TOO-WIDE` | structural checks that a conforming mesh and a built-in basis cannot reach |

## Tests

The tests are the `physics.*` entries of `ctest` (reference topology, quadrature, basis, geometry
map, `FEBase`, faces, mesh, domain, DOF map, algebraic system, mass, diffusion and transient
reaction-diffusion assembly, the linear solve, and a manufactured-solution convergence study of a
steady reaction-diffusion problem solved with Newton-Krylov) and, in an MPI build, the
`parallel.physics_*` entries.

## See also

- [](linear-algebra.md) for `DistVector`, `DistSparseMatrix`, `pcg`, `gmres` and the preconditioners
  that solve what physics-kit assembles, and for the Newton-Krylov solver.
- [](integrators.md) for stepping a semi-discrete system in time inside a component.
- [](component-interface.md) for wrapping the result as a FUSE component.
- [](build.md) for the option and the installed component.
