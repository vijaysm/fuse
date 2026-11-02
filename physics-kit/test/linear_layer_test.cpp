// physics.linear_layer -- item E (wave 2 round 3): the consistent mass
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

#include <iostream>
#include <stdexcept>

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
using fuse::linalg::SsorPreconditioner;
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

// physics.linear_layer.SsorBeatsJacobiOnTheAssembledQ1SystemInTheAsymptoticRegime
// -- item P1(b), wave 2 round 4, rewritten by the lead (2026-11-02).
// A = K + M (stiffness + consistent mass, generic_operators.hpp) on a
// structured Quad4 Q1 mesh of n x n cells: SPD (K is PSD with a 1-D null
// space the mass contribution removes).
//
// The round-4 track measured jacobi = 9, ssor(1.0) = 15 on a 6 x 6 mesh
// (49 dofs), dropped the inequality, and attributed the result to the
// 9-point stencil lacking Young's "Property A". The lead's independent
// measurement (scipy, the same Q1 element matrices, pcg at rel_tol 1e-10)
// shows the 6 x 6 result is a PRE-ASYMPTOTIC artifact -- at 49 dofs CG's
// finite termination on a few eigenvalue clusters dominates -- not a
// property of the stencil:
//   n =  6 (   49 dofs): jacobi   9, ssor(1.0)  15, ssor(1.5)  17
//   n = 16 (  289 dofs): jacobi  28, ssor(1.0)  31, ssor(1.5)  27
//   n = 32 ( 1089 dofs): jacobi  57, ssor(1.0)  55, ssor(1.5)  41
//   n = 64 ( 4225 dofs): jacobi 110, ssor(1.0)  90, ssor(1.5)  71
// This is the textbook behaviour (Axelsson & Barker 1984, Ch. 1 and 7):
// Jacobi leaves kappa = O(h^-2), iterations O(n); SSOR with any fixed
// omega in (0, 2) gives kappa = O(h^-1) for this class of FE matrices,
// iterations O(sqrt(n)), so the ratio jacobi/ssor GROWS with n (1.04,
// 1.39, 1.55 at omega 1.5 above; 0.90, 1.04, 1.22 at omega 1.0). Consistent ordering governs SOR as a
// stationary iteration, not SSOR as a CG preconditioner.
//
// Asserted: on the 32 x 32 and 64 x 64 meshes SSOR at omega = 1.0 takes
// strictly fewer pcg iterations than Jacobi (a property, not a count),
// and the ratio jacobi/ssor at 64 x 64 exceeds that at 32 x 32 (the
// asymptotic trend, so a one-off win cannot pass vacuously). All six
// counts are recorded, oracle pending. omega = 1.0 is the default, not
// a value chosen from the sweep; the sweep is recorded in the table.
namespace {
struct PcgCounts { std::size_t jacobi; std::size_t ssor; };
PcgCounts jacobi_and_ssor_counts(unsigned n) {
  Mesh mesh(2);
  mesh.generate_2d_box(n, n, Point(0, 0, 0), Point(1, 1, 0));
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry);
  AlgebraicSystem system(mesh, dof_map);
  const auto basis =
      registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);
  const auto matrix = system.assemble_matrix(
      [&](fuse::physics::spatial::EntityHandle element, const std::vector<std::int64_t>& indices,
          std::vector<Triplet>& out) {
        const auto mass = fuse::physics::fe::element_mass_matrix(fe, element);
        const auto stiffness = fuse::physics::fe::element_stiffness_matrix(fe, element);
        const std::size_t m = indices.size();
        for (std::size_t i = 0; i < m; ++i)
          for (std::size_t j = 0; j < m; ++j)
            out.push_back(Triplet{indices[i], indices[j], stiffness[i * m + j] + mass[i * m + j]});
      });
  if (!matrix.ok()) throw std::runtime_error(matrix.error().message);
  auto b = system.new_vector();
  for (std::size_t i = 0; i < b.owned_size(); ++i) b.data()[i] = 1.0;
  PcgOptions opt;
  opt.rel_tol = 1e-10;
  opt.max_iterations = 5000;
  const auto jacobi = JacobiPreconditioner::from_matrix(matrix.value());
  if (!jacobi.ok()) throw std::runtime_error(jacobi.error().message);
  auto x_jacobi = system.new_vector();
  const auto jacobi_report = fuse::linalg::pcg(matrix.value(), jacobi.value(), x_jacobi, b, opt);
  if (!jacobi_report.ok() || !jacobi_report.value().converged) throw std::runtime_error("jacobi pcg did not converge");
  const auto ssor = SsorPreconditioner::from_matrix(matrix.value(), /*omega=*/1.0);
  if (!ssor.ok()) throw std::runtime_error(ssor.error().message);
  auto x_ssor = system.new_vector();
  const auto ssor_report = fuse::linalg::pcg(matrix.value(), ssor.value(), x_ssor, b, opt);
  if (!ssor_report.ok() || !ssor_report.value().converged) throw std::runtime_error("ssor pcg did not converge");
  return {jacobi_report.value().iterations, ssor_report.value().iterations};
}
}  // namespace

TEST(LinearLayerTest, SsorBeatsJacobiOnTheAssembledQ1SystemInTheAsymptoticRegime) {
  const unsigned sizes[3] = {6, 32, 64};
  PcgCounts c[3];
  for (int k = 0; k < 3; ++k) {
    c[k] = jacobi_and_ssor_counts(sizes[k]);
    testing::Test::RecordProperty("n" + std::to_string(sizes[k]) + "_jacobi_iterations", std::to_string(c[k].jacobi));
    testing::Test::RecordProperty("n" + std::to_string(sizes[k]) + "_ssor_iterations", std::to_string(c[k].ssor));
    std::cout << "[ PCG      ] n=" << sizes[k] << " jacobi=" << c[k].jacobi << " ssor(1.0)=" << c[k].ssor << "\n";
  }
  // n = 6 is recorded only: pre-asymptotic (see the header comment).
  EXPECT_LT(c[1].ssor, c[1].jacobi) << "n=32";
  EXPECT_LT(c[2].ssor, c[2].jacobi) << "n=64";
  const double ratio32 = static_cast<double>(c[1].jacobi) / static_cast<double>(c[1].ssor);
  const double ratio64 = static_cast<double>(c[2].jacobi) / static_cast<double>(c[2].ssor);
  EXPECT_GT(ratio64, ratio32) << "the SSOR advantage must grow with n (O(sqrt n) vs O(n) iterations)";
}

}  // namespace
