// physics.linear_layer -- item E: the consistent mass
// matrix's own projection solve, through fuse::linalg::pcg +
// fuse::linalg::JacobiPreconditioner, over AlgebraicSystem's own
// DistSparseMatrix/DistVector. Disposition of KARMA's own preconditioner
// wrappers (karma_preconditioner.h, petsc_block_preconditioner.*,
// petsc_native_preconditioner.*) is in design_notes; the gist (full
// citation there): karma_preconditioner.h is a bare typedef selecting
// PetscNativePreconditioner (PETSc's own PC object, default type
// ICC_PRECOND) or, under USE_PETSC_BLOCK_EXTENSIONS,
// PetscBlockPreconditioner (PETSc's own PCBJACOBI) -- neither wrapper
// carries any preconditioning MATH of its own to port, both delegate
// entirely to a PETSc PC. The one KARMA source that DOES carry its own
// preconditioning math is diffusion_reaction_preconditioner_point_
// jacobi_solve.C's point_jacobi_solve: y <- x / diag(A), refusing a zero
// or nonfinite diagonal entry -- EXACTLY fuse::linalg::JacobiPreconditioner
// (preconditioner.hpp), named port-adapt, not port-literal only because
// the refusal code and a few details differ.
//
// PROJECTION-SOLVE TOLERANCE DERIVATION. Solving `M c = b` for a
// consistent FE mass matrix `M` IS the L2-projection step (`c`'s own
// nodal values are the projected coefficients); this test picks a KNOWN
// `u_known`, sets `b = M u_known` (so the EXACT solution is u_known with
// no discretisation question at all -- this test is purely about the
// LINEAR SOLVE's own algebraic accuracy, not about FE approximation
// error), and bounds the pcg-recovered `x`'s distance from `u_known`.
//
// Standard bound for SPD `M` (M(x* - x) = b - Mx = r, so x* - x = M^-1 r,
// and ||b||_2 = ||M x*||_2 <= lambda_max ||x*||_2):
//
//   ||x* - x||_2 / ||x*||_2 <= kappa(M) * ||r||_2 / ||b||_2
//
// where kappa(M) = lambda_max(M) / lambda_min(M) -- independent of WHICH
// preconditioner pcg used to get there (that only changes how many
// iterations it took, item G's own concern, recorded without assertion).
// pcg's own stopping test is exactly ||r||_2 <= rel_tol * ||b||_2
// (pcg.hpp's header comment), so the bound above becomes
// ||x - u_known||_2 / ||u_known||_2 <= kappa(M) * rel_tol.
//
// kappa(M), A-PRIORI (the lumped/consistent mass ratio the task names):
// on a SINGLE P1 simplex element (mass_assembly_test.cpp's own closed
// form, cited there: mass(i,i) = 2c, mass(i,j != i) = c, c = V/(size *
// (size+1)), size = dimension+1), M = c*(I + J) with J the all-ones
// matrix. J's eigenvalues are {size (once), 0 (size-1 times)} (J is
// rank 1, trace size), so M's eigenvalues are {c*(size+1) (once), c
// (size-1 times)} -- EXACTLY, not approximately, for ANY single simplex
// (the ratio c cancels, leaving a mesh- and shape-INDEPENDENT spectrum
// in units of c). kappa(M) = (size+1). For the reference Tet4 this test
// reuses from mass_assembly_test.cpp (size = 4): kappa(M) = 5 exactly.

#include "fuse_physics/fe/algebraic_system.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/pcg.hpp"
#include "fuse/linalg/preconditioner.hpp"
#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/fe/generic_operators.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::AlgebraicSystem;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::element_mass_matrix;
using fuse::physics::fe::FEBase;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::Mesh;
using fuse::linalg::JacobiPreconditioner;
using fuse::linalg::PcgOptions;
using fuse::linalg::Triplet;

TEST(LinearLayerTest, MassMatrixProjectionSolveRecoversKnownVectorWithinDerivedBound) {
  Mesh mesh(3);
  mesh.add_vertex(Point(0, 0, 0));
  mesh.add_vertex(Point(1, 0, 0));
  mesh.add_vertex(Point(0, 1, 0));
  mesh.add_vertex(Point(0, 0, 1));
  mesh.add_element(GeometryVariant::Tet4, {1, 2, 3, 4});

  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry);
  AlgebraicSystem system(mesh, dof_map);

  const auto basis = registry.create({CellFamily::Tetrahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Tet4, basis, /*quadrature_degree=*/2);

  const auto matrix = system.assemble_matrix(
      [&](fuse::physics::spatial::EntityHandle element, const std::vector<std::int64_t>& indices,
          std::vector<Triplet>& out) {
        const auto local = element_mass_matrix(fe, element);
        const std::size_t n = indices.size();
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) out.push_back(Triplet{indices[i], indices[j], local[i * n + j]});
      });
  ASSERT_TRUE(matrix.ok()) << matrix.error().message;

  const auto jacobi = JacobiPreconditioner::from_matrix(matrix.value());
  ASSERT_TRUE(jacobi.ok()) << jacobi.error().message;

  auto u_known = system.new_vector();
  const double known_values[4] = {1.0, 2.0, 3.0, 4.0};
  for (std::size_t i = 0; i < u_known.owned_size(); ++i) u_known.data()[i] = known_values[i];

  auto b = system.new_vector();
  const auto apply_result = matrix.value().apply(u_known, b);
  ASSERT_TRUE(apply_result.ok()) << apply_result.error().message;

  auto x = system.new_vector();
  x.set_zero();
  PcgOptions opt;
  opt.rel_tol = 1e-9;
  const auto report = fuse::linalg::pcg(matrix.value(), jacobi.value(), x, b, opt);
  ASSERT_TRUE(report.ok()) << report.error().message;
  EXPECT_TRUE(report.value().converged);
  // Item G: no assertion on the iteration count, recorded for the
  // provenance table only ("oracle pending" -- no reference solver run
  // in this sandbox to compare against).
  testing::Test::RecordProperty("pcg_iterations", std::to_string(report.value().iterations));
  testing::Test::RecordProperty("pcg_residual_norm", std::to_string(report.value().residual_norm));

  double diff_norm_sq = 0.0, known_norm_sq = 0.0;
  for (std::size_t i = 0; i < x.owned_size(); ++i) {
    const double d = x.data()[i] - u_known.data()[i];
    diff_norm_sq += d * d;
    known_norm_sq += u_known.data()[i] * u_known.data()[i];
  }
  const double relative_error = std::sqrt(diff_norm_sq / known_norm_sq);

  constexpr unsigned kSize = 4;  // Tet4: dimension + 1
  const double kappa_m = static_cast<double>(kSize + 1);  // = 5, exact (this file's header comment)
  const double tolerance = kappa_m * opt.rel_tol;
  EXPECT_LE(relative_error, tolerance) << "kappa(M)=" << kappa_m << " rel_tol=" << opt.rel_tol;
}

}  // namespace
