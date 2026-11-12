// physics.diffusion_reaction_transient -- the transient contract of
// fuse_physics::fe::DiffusionReactionSystem: its mass form (M u' = f) and
// its explicit form (u' = M^-1 f), and time integration of both through
// three fuse::num integrator families.
//
// Two problems on the unit square, Q1 elements, diffusion 1, no reaction.
//
// (A) A solution in the finite-element space, u = (1 + x + y) exp(-t). Its
//     gradient is constant in space, so K u vanishes on every free row;
//     the forcing u_t - Laplacian(u) = -u is itself in the space, so the
//     load is -M u with the degree-4 quadrature integrating the product of
//     two bilinear functions exactly. The semi-discrete system is then
//     satisfied by u' = -u EXACTLY, up to rounding: an identity that checks
//     the mass operator (including the columns that couple a free row to
//     Dirichlet dofs), the Dirichlet rows and the explicit form's inner
//     solve with no discretisation error in the way.
//
// (B) u = exp(-t) (1 + sin(pi x) sin(pi y)), with time-dependent Dirichlet
//     data exp(-t) and forcing -exp(-t) + (2 pi^2 - 1) exp(-t) sin sin.
//     Used for the temporal order of each integrator, measured by
//     self-convergence on ONE mesh: the states y_k obtained with 2^k steps
//     to the same final time share their spatial error exactly, so the
//     differences d_k = ||y_k - y_(k+1)|| contain the temporal error alone
//     and d_k / d_(k+1) -> 2^p for a method of order p.
//
// BAND. An observed order p_k = log2(d_k / d_(k+1)) differs from the
// asymptotic order p by a term proportional to a power of the step, so
// p_last - p is at most p_previous - p_last in magnitude to leading order;
// the assertion allows twice that (the convention of
// steady_reaction_diffusion_mms_test.cpp), and requires the last observed
// order to be the closer of the two to p, which is what that leading-order
// behaviour implies and what a step range outside it violates. No
// allowance is added for the
// error the nonlinear and linear solver tolerances leave: the band is the
// discretisation term alone, so a failure of an order assertion can also
// mean those tolerances are too loose for the differences being measured,
// never the reverse.

#include "fuse_physics/fe/diffusion_reaction_system.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/num/butcher_tableau.hpp"
#include "fuse/num/dirk_integrator.hpp"
#include "fuse/num/explicit_integrator.hpp"
#include "fuse/num/rosenbrock_integrator.hpp"
#include "fuse_physics/fe/basis.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BoundaryCondition;
using fuse::physics::fe::BoundaryConditionKind;
using fuse::physics::fe::BoundaryId;
using fuse::physics::fe::CellFamily;
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

struct Problem {
  std::function<double(const Point&, double)> exact;
  std::function<double(const Point&, double)> boundary;
  std::function<double(const Point&, double)> boundary_rate;
  std::function<double(const Point&, double)> forcing;
};

Problem in_space_problem() {
  Problem p;
  p.exact = [](const Point& x, double t) { return (1.0 + x.x + x.y) * std::exp(-t); };
  p.boundary = p.exact;
  p.boundary_rate = [](const Point& x, double t) { return -(1.0 + x.x + x.y) * std::exp(-t); };
  p.forcing = p.boundary_rate;  // u_t - Laplacian(u) = -u
  return p;
}

Problem smooth_problem() {
  Problem p;
  p.exact = [](const Point& x, double t) {
    return std::exp(-t) * (1.0 + std::sin(kPi * x.x) * std::sin(kPi * x.y));
  };
  p.boundary = [](const Point&, double t) { return std::exp(-t); };
  p.boundary_rate = [](const Point&, double t) { return -std::exp(-t); };
  p.forcing = [](const Point& x, double t) {
    return -std::exp(-t) + (2.0 * kPi * kPi - 1.0) * std::exp(-t) * std::sin(kPi * x.x) * std::sin(kPi * x.y);
  };
  return p;
}

/// An n x n Quad4 mesh of the unit square with `problem` bound and
/// initialized, Dirichlet on all four sides.
struct TransientLevel {
  Mesh mesh;
  fuse::physics::fe::BasisRegistry registry;
  DofMap dof_map;
  DiffusionReactionSystem system;
  Problem problem;

  TransientLevel(unsigned n, Problem p, bool with_boundary_rate = true)
      : mesh(2),
        dof_map(init_mesh(mesh, n), {Variable{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1}},
                registry),
        system(mesh, domain(), dof_map,
               registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}),
               GeometryVariant::Quad4, /*quadrature_degree=*/4),
        problem(std::move(p)) {
    system.set_forcing(problem.forcing);
    for (const auto id : {BoundaryId::XMin, BoundaryId::XMax, BoundaryId::YMin, BoundaryId::YMax}) {
      BoundaryCondition bc;
      bc.kind = BoundaryConditionKind::Dirichlet;
      bc.value = problem.boundary;
      if (with_boundary_rate) bc.value_time_derivative = problem.boundary_rate;
      system.set_boundary_condition(id, bc);
    }
    system.initialize();
  }

  [[nodiscard]] std::size_t n_dofs() const { return static_cast<std::size_t>(dof_map.n_dofs()); }

  /// Nodal values of the exact solution at time t. Both problems are
  /// exp(-t) times a function of position, and the boundary data agrees
  /// with the exact solution, so this is exp(-t) times the projection of
  /// the initial condition.
  [[nodiscard]] std::vector<double> nodal(double t) const {
    std::vector<double> y;
    system.project_initial_condition([&](const Point& x) { return problem.exact(x, 0.0); }, y);
    for (double& v : y) v *= std::exp(-t);
    return y;
  }

  /// L2 norm of the finite-element function with nodal values a - b.
  [[nodiscard]] double difference_norm(const std::vector<double>& a, const std::vector<double>& b) const {
    std::vector<double> d(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
    return system.l2_error(d, 0.0, [](const Point&, double) { return 0.0; });
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

double norm2(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s += x * x;
  return std::sqrt(s);
}
double norm_inf(const std::vector<double>& v) {
  double s = 0.0;
  for (const double x : v) s = std::max(s, std::fabs(x));
  return s;
}

// Rounding allowance for one assembled row of the free-row identity
// f_i + (M u)_i = 0 of problem (A), relative to nothing: an absolute bound.
// A free row receives contributions from 4 elements; each element
// contribution is a 9-point quadrature sum over 4 basis functions for the
// stiffness term and the load, at most 4 * 9 * (4 + 1) = 180 floating-point
// additions per row, so the accumulated relative error is at most
// gamma = 180 eps / (1 - 180 eps) (Higham, "Accuracy and Stability of
// Numerical Algorithms", 2nd ed., Section 3.1) of the sum of the ABSOLUTE
// values added. That sum is bounded by ||K_row||_1 max|u| + ||M_row||_1
// max|forcing| with the Q1 values ||K_row||_1 = 16/3 (8/3 on the diagonal
// and eight neighbours of 1/3) and ||M_row||_1 = h^2, twice (the row of the
// identity adds a second, independently rounded, M u).
double assembled_row_rounding(double h, double max_abs_u, double max_abs_forcing) {
  const double gamma = 180.0 * kEps / (1.0 - 180.0 * kEps);
  return gamma * (16.0 / 3.0 * max_abs_u + 2.0 * h * h * max_abs_forcing);
}

// physics.diffusion_reaction_transient.MassFormIsSatisfiedExactlyByASolutionInTheSpace
// -- problem (A): rhs(t, u) + mass(u) = 0 on every row. A free row that
// lost its Dirichlet columns in mass(), or a Dirichlet row that is not
// u' = g', leaves a defect of the size of the solution, not of rounding.
TEST(DiffusionReactionTransientTest, MassFormIsSatisfiedExactlyByASolutionInTheSpace) {
  const unsigned n = 8;
  const double h = 1.0 / n;
  TransientLevel level(n, in_space_problem());
  const double t = 0.3;
  const auto u = level.nodal(t);
  std::vector<double> f(u.size()), mu(u.size());
  ASSERT_TRUE(level.system.rhs(t, u, f).ok());
  ASSERT_TRUE(level.system.mass(u, mu).ok());
  const double bound = assembled_row_rounding(h, norm_inf(u), norm_inf(u));
  double worst = 0.0;
  for (std::size_t i = 0; i < u.size(); ++i) worst = std::max(worst, std::fabs(f[i] + mu[i]));
  std::cout << "[ MASS FORM ] max |rhs + M u| = " << worst << " (bound " << bound << ")\n";
  EXPECT_LE(worst, bound);
}

// physics.diffusion_reaction_transient.ExplicitFormReturnsTheExactRateOfASolutionInTheSpace
// -- problem (A): explicit_rhs(t, u) = -u. The error e of the returned
// rate satisfies M~ e = r, with M~ the symmetric positive definite matrix
// the explicit form solves and r the residual of that solve plus the
// rounding of its right-hand side, so ||e||_2 <= ||r||_2 / lambda_min(M~).
// lambda_min: the Jacobi-scaled Q1 mass matrix has its spectrum in
// [1/4, 9/4] (Wathen, "Realistic eigenvalue bounds for the Galerkin mass
// matrix", IMA J. Numer. Anal. 7, 1987), a principal submatrix of a
// symmetric positive definite matrix has a smallest eigenvalue no smaller,
// every free dof here is an interior vertex with diagonal entry 4 h^2 / 9,
// and the Dirichlet block is the identity: lambda_min >= min(1, h^2 / 9).
// ||r||_2: the solve is carried to the unit roundoff relative to its
// right-hand side b or to its attainable accuracy; the residual of the
// returned rate is MEASURED here through mass() and rhs() rather than
// assumed, and the rounding of b is sqrt(N) times the row bound above.
TEST(DiffusionReactionTransientTest, ExplicitFormReturnsTheExactRateOfASolutionInTheSpace) {
  const unsigned n = 8;
  const double h = 1.0 / n;
  TransientLevel level(n, in_space_problem());
  const double t = 0.3;
  const auto u = level.nodal(t);
  std::vector<double> rate(u.size()), f(u.size()), m_rate(u.size());
  ASSERT_TRUE(level.system.explicit_rhs(t, u, rate).ok());
  ASSERT_TRUE(level.system.rhs(t, u, f).ok());
  ASSERT_TRUE(level.system.mass(rate, m_rate).ok());
  std::vector<double> residual(u.size());
  for (std::size_t i = 0; i < u.size(); ++i) residual[i] = m_rate[i] - f[i];
  const double measured_residual = norm2(residual);
  const double rounding = std::sqrt(static_cast<double>(u.size())) * assembled_row_rounding(h, norm_inf(u), norm_inf(u));
  const double lambda_min = std::min(1.0, h * h / 9.0);
  const double bound = (measured_residual + rounding) / lambda_min;
  std::vector<double> error(u.size());
  for (std::size_t i = 0; i < u.size(); ++i) error[i] = rate[i] + u[i];
  std::cout << "[ EXPLICIT FORM ] ||rate + u||_2 = " << norm2(error) << " (bound " << bound
            << "; residual of the inner solve " << measured_residual << ", relative " << measured_residual / norm2(f)
            << ")\n";
  EXPECT_LE(norm2(error), bound);
  // The inner solve itself: no worse than its attainable accuracy. The
  // residual is at most the rounding of forming it, sqrt(N) rows each
  // accumulating at most 9 products of size ||M_row||_1 max|rate|.
  const double product_rounding =
      std::sqrt(static_cast<double>(u.size())) * (9.0 * kEps / (1.0 - 9.0 * kEps)) * h * h * norm_inf(rate);
  EXPECT_LE(measured_residual, kEps * norm2(f) + rounding + product_rounding);
}

// physics.diffusion_reaction_transient.DirichletConditionWithoutARateRefuses
// -- a transient right-hand side with a Dirichlet condition that carries
// no value_time_derivative is refused by code, in both forms; the
// steady_dirichlet helper supplies the zero rate.
TEST(DiffusionReactionTransientTest, DirichletConditionWithoutARateRefuses) {
  TransientLevel level(4, in_space_problem(), /*with_boundary_rate=*/false);
  const auto u = level.nodal(0.0);
  std::vector<double> out(u.size());
  const auto r = level.system.rhs(0.0, u, out);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-PHYSICS-DRS-DIRICHLET-RATE-MISSING");
  const auto e = level.system.explicit_rhs(0.0, u, out);
  ASSERT_FALSE(e.ok());
  EXPECT_EQ(e.error().code, "E-PHYSICS-DRS-DIRICHLET-RATE-MISSING");

  const auto steady = BoundaryCondition::steady_dirichlet([](const Point&, double) { return 2.5; });
  ASSERT_TRUE(static_cast<bool>(steady.value_time_derivative));
  EXPECT_EQ(steady.value_time_derivative(Point(0.3, 0.4, 0.0), 7.0), 0.0);
  EXPECT_EQ(steady.value(Point(0.3, 0.4, 0.0), 7.0), 2.5);
}

template <class Integrator>
std::vector<double> advance(TransientLevel& level, fuse::num::TemporalProblem<double>& problem, Integrator& integrator,
                            double t_final, int n_steps) {
  std::vector<double> y;
  level.system.project_initial_condition([&](const Point& x) { return level.problem.exact(x, 0.0); }, y);
  const double h = t_final / n_steps;
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    auto r = integrator.step(problem, t, h, std::span<double>(y));
    if (!r.ok()) throw std::runtime_error("step failed: " + r.error().code + ": " + r.error().message);
    t += h;
  }
  return y;
}

struct SelfConvergence {
  std::vector<double> d;      ///< d_k = ||y_k - y_(k+1)||
  std::vector<double> order;  ///< p_k = log2(d_k / d_(k+1))
  double band = 0.0;          ///< 2 |p_last - p_previous|
};

/// `run(steps)` returns the state after `steps` steps to the fixed final
/// time; level k uses first_steps * 2^k steps.
template <class Run>
SelfConvergence self_convergence(TransientLevel& level, const char* name, int first_steps, int levels, Run&& run) {
  SelfConvergence s;
  std::vector<std::vector<double>> states;
  for (int k = 0; k < levels; ++k) states.push_back(run(first_steps << k));
  for (int k = 0; k + 1 < levels; ++k) s.d.push_back(level.difference_norm(states[k], states[k + 1]));
  for (std::size_t k = 0; k + 1 < s.d.size(); ++k) s.order.push_back(std::log2(s.d[k] / s.d[k + 1]));
  const std::size_t last = s.order.size() - 1;
  s.band = 2.0 * std::fabs(s.order[last] - s.order[last - 1]);
  std::cout << "[ " << name << " ] d:";
  for (const double v : s.d) std::cout << " " << v;
  std::cout << " | orders:";
  for (const double v : s.order) std::cout << " " << v;
  std::cout << " | band " << s.band << "\n";
  return s;
}

void expect_order(const SelfConvergence& s, double p) {
  const std::size_t last = s.order.size() - 1;
  EXPECT_LT(std::fabs(s.order[last] - p), std::fabs(s.order[last - 1] - p))
      << "the observed orders are not approaching " << p << ": the step range is not asymptotic";
  EXPECT_NEAR(s.order[last], p, s.band);
}

// The stage systems of the mass form are linear here and are differenced
// by fuse::num::solve, which takes the step of a linear solve that stopped
// at the accuracy of the differenced operator and judges it by the stage
// residual (fuse/num/newton_krylov.hpp).
fuse::num::NewtonKrylovOptions dirk_options() {
  fuse::num::NewtonKrylovOptions opt;
  opt.rtol = 1e-10;
  return opt;
}

// physics.diffusion_reaction_transient.Sdirk22IsSecondOrderOnTheMassForm
// -- problem (B) through DirkIntegrator on the mass form: the first
// exercise of DirkIntegrator's implicit stages with a mass operator that
// is not the identity. The final time is 0.05, where steps of 0.025 and
// smaller are in the asymptotic range (measured to 256 steps: 2.067,
// 2.028, 2.012, 2.006, 2.003, 2.001). With a final time of 0.4 the coarse
// steps are large against the fastest mode of the 4 x 4 mesh and the
// observed orders are not yet monotone (3.23, 1.957, 1.958 for 2 to 32
// steps, then 1.975, 1.986, 1.993 to 256 steps), so that range does not
// satisfy the premise of the band.
TEST(DiffusionReactionTransientTest, Sdirk22IsSecondOrderOnTheMassForm) {
  TransientLevel level(4, smooth_problem());
  auto problem = level.system.as_temporal_problem();
  const auto s = self_convergence(level, "SDIRK22 mass form", 2, 5, [&](int steps) {
    fuse::num::DirkIntegrator<double> integrator(fuse::num::sdirk22_tableau(), dirk_options());
    return advance(level, problem, integrator, 0.05, steps);
  });
  expect_order(s, 2.0);
}

// physics.diffusion_reaction_transient.Kutta22IsSecondOrderOnTheExplicitForm
// -- problem (B) through ExplicitIntegrator on the explicit form. The
// explicit family refuses a problem with a mass operator, so this is the
// path by which it integrates a consistent-mass discretisation.
TEST(DiffusionReactionTransientTest, Kutta22IsSecondOrderOnTheExplicitForm) {
  TransientLevel level(4, smooth_problem());
  auto problem = level.system.as_explicit_form_temporal_problem();
  const auto s = self_convergence(level, "Kutta22 explicit form", 16, 5, [&](int steps) {
    fuse::num::ExplicitIntegrator<double> integrator(fuse::num::kutta22_tableau());
    return advance(level, problem, integrator, 0.05, steps);
  });
  expect_order(s, 2.0);

  auto mass_form = level.system.as_temporal_problem();
  fuse::num::ExplicitIntegrator<double> integrator(fuse::num::kutta22_tableau());
  std::vector<double> y = level.nodal(0.0);
  const auto refused = integrator.step(mass_form, 0.0, 1e-3, std::span<double>(y));
  ASSERT_FALSE(refused.ok()) << "the explicit family must refuse the mass form";
}

// The Jacobian action of the explicit form of problem (B), without a
// differencing step. With no reaction the explicit form is affine in the
// state, u' = A u + c(t), so J v = explicit_rhs(t, v) - explicit_rhs(t, 0)
// exactly; the only error is the rounding of that subtraction.
void set_affine_jacobian_action(const TransientLevel& level, fuse::num::TemporalProblem<double>& problem) {
  const DiffusionReactionSystem* system = &level.system;
  const std::size_t n = level.n_dofs();
  problem.jacobian_action = [system, n](double t, std::span<const double>, std::span<const double> v,
                                        std::span<double> out) -> fuse::Result<void> {
    std::vector<double> zero(n, 0.0), at_zero(n);
    if (auto r = system->explicit_rhs(t, v, out); !r.ok()) return r.error();
    if (auto r = system->explicit_rhs(t, zero, std::span<double>(at_zero)); !r.ok()) return r.error();
    for (std::size_t i = 0; i < n; ++i) out[i] -= at_zero[i];
    return fuse::Result<void>::success();
  };
}

// physics.diffusion_reaction_transient.Ros22IsSecondOrderOnTheExplicitForm
// -- problem (B) through RosenbrockIntegrator on the explicit form, with
// the affine Jacobian action above. A Rosenbrock stage is one linear solve
// with no outer iteration to correct it, so its tolerance enters the state
// directly.
TEST(DiffusionReactionTransientTest, Ros22IsSecondOrderOnTheExplicitForm) {
  TransientLevel level(4, smooth_problem());
  auto problem = level.system.as_explicit_form_temporal_problem();
  set_affine_jacobian_action(level, problem);
  const auto s = self_convergence(level, "ROS22 explicit form", 2, 5, [&](int steps) {
    fuse::num::RosenbrockIntegrator<double> integrator(fuse::num::ros22_tableau(), /*step_rtol=*/1e-10,
                                                        /*stage_tolerance_factor=*/1.0);
    return advance(level, problem, integrator, 0.4, steps);
  });
  expect_order(s, 2.0);
}

// physics.diffusion_reaction_transient.Ros22RefusesAStageToleranceBelowTheDifferencedActionsAccuracy
// -- without a supplied Jacobian action RosenbrockIntegrator differences
// explicit_rhs. The differenced action is accurate to a relative level of
// order sqrt(unit roundoff) times the ratio of the right-hand side to its
// directional derivative (measured on this problem: the stage residual
// stops near 4e-7 relative), so a stage tolerance of 1e-10 cannot be met.
// A Rosenbrock stage has no outer residual by which an inexact solve could
// be judged, so the integrator refuses the step and names the cause; it
// does not return a state computed from a stage of unknown accuracy.
TEST(DiffusionReactionTransientTest, Ros22RefusesAStageToleranceBelowTheDifferencedActionsAccuracy) {
  TransientLevel level(8, smooth_problem());
  auto problem = level.system.as_explicit_form_temporal_problem();
  fuse::num::RosenbrockIntegrator<double> integrator(fuse::num::ros22_tableau(), /*step_rtol=*/1e-10,
                                                      /*stage_tolerance_factor=*/1.0);
  std::vector<double> y = level.nodal(0.0);
  const std::vector<double> before = y;
  const auto r = integrator.step(problem, 0.0, 0.2, std::span<double>(y));
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-INTEGRATOR-STAGE-SOLVE-FAILED");
  EXPECT_NE(r.error().message.find("E-LINALG-ATTAINABLE-ACCURACY"), std::string::npos) << r.error().message;
  EXPECT_EQ(y, before) << "a refused step must leave the state untouched";
}

}  // namespace
