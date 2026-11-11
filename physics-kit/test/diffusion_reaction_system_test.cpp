// physics.diffusion_reaction_system -- fuse_physics::fe::
// DiffusionReactionSystem's steady interface: the residual, the assembled
// Jacobian a preconditioner is built from, the refusals and the lifecycle.
// The convergence orders are measured in
// diffusion_reaction_steady_order_test.cpp and the transient interface in
// diffusion_reaction_transient_test.cpp.
//
// ROUNDING BOUNDS. Two tests compare quantities that are equal in exact
// arithmetic. An assembled row is a sum of at most k floating-point terms,
// so its computed value differs from the exact one by at most
// gamma_k = k eps / (1 - k eps) times the sum of the ABSOLUTE values of
// its terms (Higham, "Accuracy and Stability of Numerical Algorithms",
// 2nd ed., Section 3.1). For a bilinear element on an n x n mesh of the
// unit square, h = 1/n: a row has contributions from at most 4 elements,
// each a 9-point quadrature sum over 4 basis functions for the stiffness
// term, 4 for the nodal reaction and 1 for the load (324 terms), at most
// 4 boundary-face sums of 3 points and 3 terms (36), and each term is
// formed by at most 20 further operations (basis values, the metric
// inverse, the weight): k = 400 is used. The absolute row sums are
// 16/3 for the stiffness matrix (8/3 on the diagonal, eight neighbours of
// 1/3), h^2 for the mass matrix and h for a boundary mass block.

#include "fuse_physics/fe/diffusion_reaction_system.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/preconditioner.hpp"
#include "fuse/num/newton_krylov.hpp"
#include "fuse_physics/fe/basis.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::BoundaryCondition;
using fuse::physics::fe::BoundaryConditionKind;
using fuse::physics::fe::BoundaryId;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::DiffusionReactionCoefficients;
using fuse::physics::fe::DiffusionReactionSystem;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::Domain;
using fuse::physics::spatial::Mesh;
using fuse::physics::spatial::Shape;

constexpr double kPi = 3.14159265358979323846;
constexpr double kEps = std::numeric_limits<double>::epsilon();

double gamma_k(double k) { return k * kEps / (1.0 - k * kEps); }

/// A DiffusionReactionSystem over an n x n mesh of bilinear quadrilaterals
/// on the unit square.
struct Level {
  Mesh mesh;
  BasisRegistry registry;
  DofMap dof_map;
  DiffusionReactionSystem system;

  explicit Level(unsigned n)
      : mesh(2),
        dof_map(init_mesh(mesh, n), {Variable{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1}},
                registry),
        system(mesh, domain(), dof_map,
               registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}),
               GeometryVariant::Quad4, /*quadrature_degree=*/4) {}

  [[nodiscard]] std::size_t n_dofs() const { return static_cast<std::size_t>(dof_map.n_dofs()); }
  [[nodiscard]] fuse::linalg::DistVector vector() const {
    return fuse::linalg::DistVector(n_dofs(), fuse::linalg::make_serial_exchanger(),
                                    fuse::linalg::make_serial_reducer());
  }

  static Domain& domain() {
    static Domain d(2, Shape::Square, 1.0, 1.0, 0.0, false);
    return d;
  }

 private:
  static Mesh& init_mesh(Mesh& mesh, unsigned n) {
    mesh.generate_2d_box(n, n, Point(0, 0, 0), Point(1, 1, 0));
    return mesh;
  }
};

double max_abs(const fuse::linalg::DistVector& v) {
  double m = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) m = std::max(m, std::fabs(v.data()[i]));
  return m;
}

// physics.diffusion_reaction_system.ResidualVanishesForAnAffineSolutionWithAllThreeBoundaryKinds
// -- u = x + 2y is in the bilinear space and has zero Laplacian. With
// R(u) = u and forcing x + 2y the equation -Laplacian(u) + R(u) = f
// holds; Dirichlet data on x = 0 and y = 0, the flux du/dx = 1 on x = 1
// (Neumann) and du/dy + u = 2 + (x + 2) on y = 1 (Robin, coefficient 1)
// are those of u. The residual at the nodal values of u is zero in exact
// arithmetic on every row: a sign error in the Neumann or Robin face term
// or in the Dirichlet row leaves a residual of the size of the data.
//
// Bound (header comment): the absolute row sum is at most
//   16/3 max|u| + h^2 max|R(u)| + h^2 max|f| + h max|g_N| + h (max|g_R| + max|u|)
// with max|u| = max|R| = max|f| = 3, max|g_N| = 1, max|g_R| = 5.
TEST(DiffusionReactionSystemTest, ResidualVanishesForAnAffineSolutionWithAllThreeBoundaryKinds) {
  const unsigned n = 3;
  const double h = 1.0 / n;
  Level level(n);
  auto u_exact = [](const Point& p, double) { return p.x + 2.0 * p.y; };
  DiffusionReactionCoefficients coefficients;
  coefficients.reaction = [](const Point&, double u) { return u; };
  level.system.set_coefficients(coefficients);
  level.system.set_forcing(u_exact);
  level.system.set_boundary_condition(BoundaryId::XMin, BoundaryCondition::steady_dirichlet(u_exact));
  level.system.set_boundary_condition(BoundaryId::YMin, BoundaryCondition::steady_dirichlet(u_exact));
  level.system.set_boundary_condition(
      BoundaryId::XMax, BoundaryCondition{BoundaryConditionKind::Neumann, [](const Point&, double) { return 1.0; }, 0.0});
  level.system.set_boundary_condition(
      BoundaryId::YMax,
      BoundaryCondition{BoundaryConditionKind::Robin, [](const Point& p, double) { return 2.0 + (p.x + 2.0); }, 1.0});
  level.system.initialize();

  std::vector<double> nodal;
  level.system.project_initial_condition([&](const Point& p) { return u_exact(p, 0.0); }, nodal);
  ASSERT_EQ(nodal.size(), level.n_dofs());
  auto x = level.vector(), fx = level.vector();
  for (std::size_t i = 0; i < nodal.size(); ++i) x.data()[i] = nodal[i];
  ASSERT_TRUE(level.system.assemble_residual(x, fx).ok());

  const double row_sum = 16.0 / 3.0 * 3.0 + h * h * 3.0 + h * h * 3.0 + h * 1.0 + h * (5.0 + 3.0);
  const double bound = gamma_k(400.0) * row_sum;
  std::cout << "[ AFFINE   ] max |residual| = " << max_abs(fx) << " (bound " << bound << ")\n";
  EXPECT_LE(max_abs(fx), bound);

  // The interpolant of u is u, so the L2 error is rounding: at a
  // quadrature point the interpolated value is a sum of 4 products of
  // nonnegative basis values that sum to 1, and the point itself and the
  // evaluation of u there carry a few rounding errors each; 16 in all,
  // against max|u| = 3 on a domain of measure 1.
  const double l2 = level.system.l2_error(nodal, 0.0, u_exact);
  std::cout << "[ AFFINE   ] L2 error of the interpolant = " << l2 << " (bound " << gamma_k(16.0) * 3.0 << ")\n";
  EXPECT_LE(l2, gamma_k(16.0) * 3.0);
}

// physics.diffusion_reaction_system.AssembledJacobianIsTheDerivativeOfAnAffineResidual
// -- with a constant diffusion coefficient and R(u) = c u the residual is
// affine in the state, so F(x + v) - F(x) = J v with no differencing
// error, J the matrix assemble_approximate_jacobian returns. Dirichlet on
// x = 0, Robin on y = 1, natural conditions elsewhere; c = 40, so the
// reaction block carries most of the diagonal and a wrong sign changes
// J v by 2 c M v.
//
// Bound (header comment): each of F(x + v), F(x) and J v is a row sum
// whose absolute terms are at most
//   S(w) = (16/3 + c h^2 + c_R h) max|w| + h^2 max|f| + h max|g_R|,
// so the three together are bounded by gamma_400 (S(x + v) + S(x) + S(v)).
TEST(DiffusionReactionSystemTest, AssembledJacobianIsTheDerivativeOfAnAffineResidual) {
  const unsigned n = 4;
  const double h = 1.0 / n;
  const double c = 40.0, robin = 3.0;
  Level level(n);
  DiffusionReactionCoefficients coefficients;
  coefficients.reaction = [c](const Point&, double u) { return c * u; };
  coefficients.reaction_derivative = [c](const Point&, double) { return c; };
  level.system.set_coefficients(coefficients);
  level.system.set_forcing([](const Point& p, double) { return 1.0 + p.x; });
  level.system.set_boundary_condition(BoundaryId::XMin,
                                      BoundaryCondition::steady_dirichlet([](const Point&, double) { return 1.0; }));
  level.system.set_boundary_condition(
      BoundaryId::YMax, BoundaryCondition{BoundaryConditionKind::Robin, [](const Point&, double) { return 2.0; }, robin});
  level.system.initialize();

  auto x = level.vector(), v = level.vector(), xv = level.vector();
  for (std::size_t i = 0; i < x.size(); ++i) {
    x.data()[i] = std::sin(1.0 + 0.7 * static_cast<double>(i));
    v.data()[i] = std::cos(2.0 + 1.3 * static_cast<double>(i));
    xv.data()[i] = x.data()[i] + v.data()[i];
  }
  auto f_x = level.vector(), f_xv = level.vector();
  ASSERT_TRUE(level.system.assemble_residual(x, f_x).ok());
  ASSERT_TRUE(level.system.assemble_residual(xv, f_xv).ok());

  const auto jacobian = level.system.assemble_approximate_jacobian(x);
  ASSERT_TRUE(jacobian.ok()) << jacobian.error().message;
  fuse::linalg::DistVector operand(v), product(v);
  ASSERT_TRUE(jacobian.value().apply(operand, product).ok());

  auto row_sum = [&](double w) { return (16.0 / 3.0 + c * h * h + robin * h) * w + h * h * 2.0 + h * 2.0; };
  const double bound = gamma_k(400.0) * (row_sum(max_abs(xv)) + row_sum(max_abs(x)) + row_sum(max_abs(v)));
  double worst = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i)
    worst = std::max(worst, std::fabs((f_xv.data()[i] - f_x.data()[i]) - product.data()[i]));
  std::cout << "[ JACOBIAN ] max |F(x+v) - F(x) - J v| = " << worst << " (bound " << bound << "), max |J v| = "
            << max_abs(product) << "\n";
  EXPECT_LE(worst, bound);
}

// physics.diffusion_reaction_system.SsorFromTheAssembledJacobianReducesGmresIterations
// -- the steady problem of diffusion_reaction_steady_order_test.cpp
// (-Laplacian(u) + u^2 / 2 = f, homogeneous Dirichlet), solved by
// fuse::num::solve without a preconditioner and with the SSOR
// preconditioner built from assemble_approximate_jacobian.
//
// What is compared is the FIRST Newton step, and only that step is taken.
// Both runs start from the same state, so it solves the same linear
// system J(0) d = -F(0); GMRES is right-preconditioned, so both stop on
// the same residual; and the forcing term is fixed, so both stop at the
// same tolerance. A sum over all Newton steps is not such a comparison:
// the later systems and their forcing terms differ between the runs, and
// so can the number of steps. The forcing term is 1e-4, which is above
// the accuracy of the differenced Jacobian action, so neither run stops
// at its attainable accuracy.
//
// At the zero state J is the stiffness matrix with identity Dirichlet
// rows. Its condition number grows as h^-2, so unpreconditioned GMRES
// needs O(1/h) iterations, and SSOR reduces the condition number to
// O(1/h) (Axelsson, "Iterative Solution Methods", Section 7.2). Both
// statements are asymptotic: the right-hand side here is the load of a
// smooth function, which a few Krylov vectors resolve on a coarse mesh
// with or without a preconditioner (measured: 5 and 5 iterations at
// n = 8, 9 and 8 at n = 16, 17 and 13 at n = 32; and 36 and 19 at n = 64,
// which this test does not run).
// The assertions are therefore that the ratio of the two counts never
// decreases under refinement, that it is larger on the finest mesh than
// on the coarsest, and that SSOR needs fewer iterations on the finest.
TEST(DiffusionReactionSystemTest, SsorFromTheAssembledJacobianReducesGmresIterations) {
  const double alpha = 0.5;
  struct Counts {
    std::size_t plain = 0, ssor = 0;
  };
  const std::vector<unsigned> sizes{8, 16, 32};
  std::vector<Counts> counts;
  for (const unsigned n : sizes) {
    Level level(n);
    DiffusionReactionCoefficients coefficients;
    coefficients.reaction = [alpha](const Point&, double u) { return alpha * u * u; };
    coefficients.reaction_derivative = [alpha](const Point&, double u) { return 2.0 * alpha * u; };
    level.system.set_coefficients(coefficients);
    level.system.set_forcing([alpha](const Point& p, double) {
      const double u = std::sin(kPi * p.x) * std::sin(kPi * p.y);
      return 2.0 * kPi * kPi * u + alpha * u * u;
    });
    for (const auto id : {BoundaryId::XMin, BoundaryId::XMax, BoundaryId::YMin, BoundaryId::YMax})
      level.system.set_boundary_condition(id,
                                          BoundaryCondition::steady_dirichlet([](const Point&, double) { return 0.0; }));
    level.system.initialize();

    auto first_step_iterations = [&](bool precondition) -> std::size_t {
      auto x = level.vector();
      x.set_zero();
      fuse::num::NewtonKrylovOptions options;
      options.max_nonlinear_iterations = 1;
      options.forcing.eta_initial = options.forcing.eta_min = options.forcing.eta_max = 1e-4;
      std::optional<fuse::linalg::SsorPreconditioner> ssor;
      fuse::num::PreconditionerApplyFn apply;
      fuse::num::RefreshPreconditionerFn refresh;
      if (precondition) {
        refresh = [&](const fuse::linalg::DistVector& state, const fuse::linalg::DistVector&) -> fuse::Result<void> {
          auto jacobian = level.system.assemble_approximate_jacobian(state);
          if (!jacobian.ok()) return jacobian.error();
          auto built = fuse::linalg::SsorPreconditioner::from_matrix(jacobian.value());
          if (!built.ok()) return built.error();
          ssor = built.value();
          return fuse::Result<void>::success();
        };
        apply = [&](fuse::linalg::DistVector& r, fuse::linalg::DistVector& z) { return ssor->apply(r, z); };
      }
      const auto solved = fuse::num::solve(
          [&](fuse::linalg::DistVector& state, fuse::linalg::DistVector& f) {
            return level.system.assemble_residual(state, f);
          },
          x, apply, refresh, options);
      if (!solved.ok()) throw std::runtime_error("steady solve failed: " + solved.error().message);
      EXPECT_EQ(solved.value().trace.size(), 1u);
      EXPECT_EQ(solved.value().trace.front().eta, 1e-4);
      return solved.value().trace.front().gmres_iterations;
    };
    counts.push_back({first_step_iterations(false), first_step_iterations(true)});
    std::cout << "[ PRECOND  ] n=" << n << " first Newton step: gmres=" << counts.back().plain
              << " gmres+ssor=" << counts.back().ssor << "\n";
  }
  auto ratio = [&](std::size_t k) { return static_cast<double>(counts[k].plain) / static_cast<double>(counts[k].ssor); };
  for (std::size_t k = 0; k + 1 < sizes.size(); ++k)
    EXPECT_GE(ratio(k + 1), ratio(k)) << "n=" << sizes[k] << " to n=" << sizes[k + 1];
  EXPECT_GT(ratio(sizes.size() - 1), ratio(0));
  EXPECT_LT(counts.back().ssor, counts.back().plain) << "n=" << sizes.back();
}

template <class Call>
std::string refusal_of(Call&& call) {
  try {
    call();
  } catch (const std::invalid_argument& e) {
    return e.what();
  }
  return "accepted";
}

TEST(DiffusionReactionSystemTest, ConstructRefusesADofMapThatIsNotOneScalarField) {
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(0, 0, 0), Point(1, 1, 0));
  BasisRegistry registry;
  DofMap dof_map(mesh, {Variable{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1},
                        Variable{"v", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1}},
                 registry);
  const Domain domain(2, Shape::Square, 1.0, 1.0, 0.0, false);
  const auto basis = registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  const std::string refusal = refusal_of(
      [&] { DiffusionReactionSystem system(mesh, domain, dof_map, basis, GeometryVariant::Quad4, 2); });
  EXPECT_NE(refusal.find("E-PHYSICS-DRS-NOT-SCALAR"), std::string::npos) << refusal;
}

// Every entry point that reads state resolved by initialize() refuses
// before it has run.
TEST(DiffusionReactionSystemTest, EveryEvaluationRefusesBeforeInitialize) {
  Level level(2);
  auto x = level.vector(), fx = level.vector();
  std::vector<double> y(level.n_dofs(), 0.0), out(level.n_dofs(), 0.0), projected;
  const std::vector<std::string> refusals{
      refusal_of([&] { (void)level.system.assemble_residual(x, fx); }),
      refusal_of([&] { (void)level.system.assemble_approximate_jacobian(x); }),
      refusal_of([&] { (void)level.system.rhs(0.0, y, out); }),
      refusal_of([&] { (void)level.system.mass(y, out); }),
      refusal_of([&] { (void)level.system.explicit_rhs(0.0, y, out); }),
      refusal_of([&] { level.system.project_initial_condition([](const Point&) { return 0.0; }, projected); }),
  };
  for (std::size_t i = 0; i < refusals.size(); ++i)
    EXPECT_NE(refusals[i].find("E-PHYSICS-DRS-NOT-INITIALIZED"), std::string::npos) << "call " << i << ": " << refusals[i];
}

// physics.diffusion_reaction_system.InitializeAgainPicksUpAChangedForcingTerm
// -- at the zero state the residual of a free row is minus the load, a
// quadrature sum in which the forcing value multiplies every term. A
// forcing of 2 instead of 1 doubles every term exactly (a multiplication
// by 2 is exact in binary floating point), so the row doubles exactly.
TEST(DiffusionReactionSystemTest, InitializeAgainPicksUpAChangedForcingTerm) {
  Level level(3);
  for (const auto id : {BoundaryId::XMin, BoundaryId::XMax, BoundaryId::YMin, BoundaryId::YMax})
    level.system.set_boundary_condition(id,
                                        BoundaryCondition::steady_dirichlet([](const Point&, double) { return 0.0; }));
  level.system.set_forcing([](const Point&, double) { return 1.0; });
  level.system.initialize();

  auto x = level.vector(), before = level.vector(), after = level.vector();
  x.set_zero();
  ASSERT_TRUE(level.system.assemble_residual(x, before).ok());

  level.system.set_forcing([](const Point&, double) { return 2.0; });
  level.system.initialize();
  ASSERT_TRUE(level.system.assemble_residual(x, after).ok());

  std::size_t nonzero_rows = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    EXPECT_EQ(after.data()[i], 2.0 * before.data()[i]) << "row " << i;
    if (before.data()[i] != 0.0) ++nonzero_rows;
  }
  EXPECT_EQ(nonzero_rows, 4u) << "the 3 x 3 mesh has 4 free dofs, each with a nonzero load";
}

}  // namespace
