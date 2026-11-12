// num.newton_attainable_accuracy -- what fuse::num::solve does when a
// forcing term asks the linear solve for more than the differenced
// Jacobian action can deliver (newton_krylov.hpp, "ATTAINABLE ACCURACY OF
// THE LINEAR SOLVE").
//
// What each test pins:
//   ConvergesWhenTheForcingTermIsBelowTheAccuracyOfTheDifferencedAction --
//     a linear residual with a constant forcing term of 1e-14. The linear
//     solve stops at its attainable accuracy, the step is taken, and the
//     iteration converges on the nonlinear residual; the trace says which
//     steps stopped that way. The control with a forcing term of 1e-4
//     records no such step.
//   ALinearSolveWithoutProgressIsNotAStep -- a residual whose directional
//     derivative at the starting point is not linear in the direction, so
//     the recurrence of the first GMRES cycle reports convergence while
//     the recomputed residual has doubled. No step exists; the result is
//     DivergedLinearSolve with x untouched, not ConvergedStep on a zero
//     step.
//   AStepThatDoesNotReduceTheResidualEndsTheIteration -- arctan residuals
//     started where a full Newton step overshoots, with a full-step line
//     search and a forcing term of 1e-14. The step from the linear solve at
//     its attainable accuracy increases ||F||, and the iteration ends
//     there.
//   AShortStepThatDoesNotReduceTheResidualIsNotConvergence -- the same
//     residuals with one more unknown, of value 1e12 and already at its
//     root, so that the same overshooting step is short against ||x||. It
//     is still reported as DivergedLinearSolve, not as convergence by step
//     length.

#include "fuse/num/newton_krylov.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"

namespace {

using fuse::Result;
using fuse::linalg::DistVector;
using fuse::num::EisenstatWalkerChoice;
using fuse::num::LineSearchKind;
using fuse::num::NewtonKrylovOptions;
using fuse::num::NewtonResult;
using fuse::num::NewtonStatus;
using fuse::num::ResidualFn;

DistVector vector_of(std::size_t n) {
  DistVector v(n, fuse::linalg::make_serial_exchanger(), fuse::linalg::make_serial_reducer());
  v.set_zero();
  return v;
}

constexpr std::size_t kN = 40;

/// y = A x for A = tridiag(-1, 4, -1): symmetric, eigenvalues
/// 4 - 2 cos(k pi / (n + 1)) in (2, 6), so ||A^-1||_2 < 1/2.
void apply_tridiagonal(const double* x, double* y) {
  for (std::size_t i = 0; i < kN; ++i) {
    double s = 4.0 * x[i];
    if (i > 0) s -= x[i - 1];
    if (i + 1 < kN) s -= x[i + 1];
    y[i] = s;
  }
}

struct LinearProblem {
  std::vector<double> reference = std::vector<double>(kN);
  std::vector<double> rhs = std::vector<double>(kN);
  LinearProblem() {
    for (std::size_t i = 0; i < kN; ++i) reference[i] = std::sin(0.3 * static_cast<double>(i + 1));
    apply_tridiagonal(reference.data(), rhs.data());
  }
  [[nodiscard]] ResidualFn residual() const {
    return [this](DistVector& x, DistVector& fx) -> Result<void> {
      apply_tridiagonal(x.data(), fx.data());
      for (std::size_t i = 0; i < kN; ++i) fx.data()[i] -= rhs[i];
      return Result<void>::success();
    };
  }
};

NewtonKrylovOptions constant_forcing(double eta) {
  NewtonKrylovOptions opt;
  opt.forcing.choice = EisenstatWalkerChoice::Constant;
  opt.forcing.eta_constant = eta;
  opt.rtol = 1e-12;
  return opt;
}

void print_trace(const char* name, const NewtonResult& r) {
  std::cout << "[ NEWTON   ] " << name << ": " << fuse::num::to_cstring(r.status) << " after " << r.iterations
            << " step(s), ||F0||=" << r.initial_residual_norm << " ||F||=" << r.final_residual_norm << "\n";
  for (const auto& rec : r.trace)
    std::cout << "[ NEWTON   ]   k=" << rec.iteration << " |F|=" << rec.norm_f << " eta=" << rec.eta
              << " gmres=" << rec.gmres_iterations << " linear residual=" << rec.gmres_residual_norm
              << " at attainable accuracy=" << rec.linear_solve_at_attainable_accuracy << "\n";
}

TEST(NewtonAttainableAccuracy, ConvergesWhenTheForcingTermIsBelowTheAccuracyOfTheDifferencedAction) {
  const LinearProblem problem;
  DistVector x = vector_of(kN);
  const auto solved = fuse::num::solve(problem.residual(), x, nullptr, nullptr, constant_forcing(1e-14));
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  const NewtonResult& r = solved.value();
  print_trace("forcing term 1e-14", r);

  EXPECT_TRUE(fuse::num::is_converged(r.status)) << fuse::num::to_cstring(r.status);
  EXPECT_LE(r.final_residual_norm, 1e-12 * r.initial_residual_norm);
  ASSERT_FALSE(r.trace.empty());
  // The first step is the one this test is about: its forcing term cannot
  // be met, the linear residual it reports is nevertheless below ||F||
  // (an inexact Newton step), and above the forcing term.
  const auto& first = r.trace.front();
  EXPECT_TRUE(first.linear_solve_at_attainable_accuracy);
  EXPECT_EQ(first.eta, 1e-14);
  EXPECT_LT(first.gmres_residual_norm, first.norm_f);
  EXPECT_GT(first.gmres_residual_norm, first.eta * first.norm_f);

  // The solution, against the vector the right-hand side was built from:
  // A (x - reference) = F(x) - F(reference), and ||A^-1||_2 < 1/2.
  DistVector reference = vector_of(kN), f_reference = vector_of(kN);
  for (std::size_t i = 0; i < kN; ++i) reference.data()[i] = problem.reference[i];
  ASSERT_TRUE(problem.residual()(reference, f_reference).ok());
  double error_sq = 0.0;
  for (std::size_t i = 0; i < kN; ++i) {
    const double e = x.data()[i] - problem.reference[i];
    error_sq += e * e;
  }
  EXPECT_LE(std::sqrt(error_sq), 0.5 * (r.final_residual_norm + f_reference.norm2()));

  // Control: a forcing term the differenced action can meet.
  DistVector x_control = vector_of(kN);
  const auto control = fuse::num::solve(problem.residual(), x_control, nullptr, nullptr, constant_forcing(1e-4));
  ASSERT_TRUE(control.ok()) << control.error().message;
  print_trace("forcing term 1e-4", control.value());
  EXPECT_TRUE(fuse::num::is_converged(control.value().status));
  for (const auto& rec : control.value().trace)
    EXPECT_FALSE(rec.linear_solve_at_attainable_accuracy) << "step " << rec.iteration;
}

TEST(NewtonAttainableAccuracy, ALinearSolveWithoutProgressIsNotAStep) {
  // F_i(x) = -|x_i| - c_i with c_i > 0 has no root. At x = 0 its
  // directional derivative along v is -|v|: along the first Krylov vector
  // c / ||c|| it acts as minus the identity, so the first GMRES step
  // proposes d = -c with a recurrence residual at rounding level, while
  // the residual recomputed along d is c + |d| = 2 c.
  const std::vector<double> c = {0.5, 1.0, 1.5, 2.0};
  const ResidualFn residual = [&c](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < c.size(); ++i) fx.data()[i] = -std::fabs(x.data()[i]) - c[i];
    return Result<void>::success();
  };
  DistVector x = vector_of(c.size());
  const auto solved = fuse::num::solve(residual, x, nullptr, nullptr, NewtonKrylovOptions{});
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  const NewtonResult& r = solved.value();
  print_trace("no linear progress", r);

  EXPECT_EQ(r.status, NewtonStatus::DivergedLinearSolve) << fuse::num::to_cstring(r.status);
  EXPECT_EQ(r.iterations, 0);
  ASSERT_EQ(r.trace.size(), 1u);
  EXPECT_TRUE(r.trace.front().linear_solve_at_attainable_accuracy);
  EXPECT_EQ(r.final_residual_norm, r.initial_residual_norm);
  for (std::size_t i = 0; i < c.size(); ++i) EXPECT_EQ(x.data()[i], 0.0) << "component " << i;
}

TEST(NewtonAttainableAccuracy, AStepThatDoesNotReduceTheResidualEndsTheIteration) {
  // F_i(x) = arctan(x_i). From |x_i| > 1.3917 a full Newton step
  // x - (1 + x^2) arctan(x) lands farther from the root than it started
  // (the classical divergence of Newton's method for arctan), so ||F||
  // grows.
  const std::vector<double> start = {3.0, 3.5, 4.0, 4.5, 5.0, 5.5};
  const ResidualFn residual = [](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < fx.size(); ++i) fx.data()[i] = std::atan(x.data()[i]);
    return Result<void>::success();
  };
  DistVector x = vector_of(start.size());
  for (std::size_t i = 0; i < start.size(); ++i) x.data()[i] = start[i];
  NewtonKrylovOptions opt = constant_forcing(1e-14);
  opt.line_search.kind = LineSearchKind::Basic;
  const auto solved = fuse::num::solve(residual, x, nullptr, nullptr, opt);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  const NewtonResult& r = solved.value();
  print_trace("overshooting step", r);

  EXPECT_EQ(r.status, NewtonStatus::DivergedLinearSolve) << fuse::num::to_cstring(r.status);
  EXPECT_EQ(r.iterations, 1);
  ASSERT_EQ(r.trace.size(), 1u);
  EXPECT_TRUE(r.trace.front().linear_solve_at_attainable_accuracy);
  EXPECT_LT(r.trace.front().gmres_residual_norm, r.trace.front().norm_f);
  EXPECT_GE(r.final_residual_norm, r.initial_residual_norm);
  // x is at the step that was taken.
  for (std::size_t i = 0; i < start.size(); ++i) EXPECT_LT(x.data()[i], -start[i]) << "component " << i;
}

TEST(NewtonAttainableAccuracy, AShortStepThatDoesNotReduceTheResidualIsNotConvergence) {
  // The last unknown has the residual x - 1e12 and starts at its root, so
  // ||x|| is 1e12 and the step-length test passes for any step shorter than
  // stol * ||x|| = 1e4; the overshooting arctan step is of order 100.
  const std::vector<double> start = {3.0, 3.5, 4.0, 4.5, 5.0, 5.5};
  const double large = 1e12;
  const std::size_t n = start.size();
  const ResidualFn residual = [n, large](DistVector& x, DistVector& fx) -> Result<void> {
    for (std::size_t i = 0; i < n; ++i) fx.data()[i] = std::atan(x.data()[i]);
    fx.data()[n] = x.data()[n] - large;
    return Result<void>::success();
  };
  DistVector x = vector_of(n + 1);
  for (std::size_t i = 0; i < n; ++i) x.data()[i] = start[i];
  x.data()[n] = large;
  NewtonKrylovOptions opt = constant_forcing(1e-14);
  opt.line_search.kind = LineSearchKind::Basic;
  const auto solved = fuse::num::solve(residual, x, nullptr, nullptr, opt);
  ASSERT_TRUE(solved.ok()) << solved.error().message;
  const NewtonResult& r = solved.value();
  print_trace("short overshooting step", r);

  ASSERT_EQ(r.trace.size(), 1u);
  ASSERT_TRUE(r.trace.front().linear_solve_at_attainable_accuracy);
  ASSERT_LT(r.trace.front().gmres_residual_norm, r.trace.front().norm_f);
  ASSERT_GE(r.final_residual_norm, r.initial_residual_norm);
  double step_sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) step_sq += (x.data()[i] - start[i]) * (x.data()[i] - start[i]);
  step_sq += (x.data()[n] - large) * (x.data()[n] - large);
  ASSERT_LE(std::sqrt(step_sq), opt.stol * large) << "the premise: the step passes the step-length test";
  EXPECT_EQ(r.status, NewtonStatus::DivergedLinearSolve) << fuse::num::to_cstring(r.status);
}

}  // namespace
