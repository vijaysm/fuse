// linalg.gmres_attainable_accuracy -- fuse::linalg::gmres's stopping rule
// for an operator that is applied inexactly (gmres.hpp, "ATTAINABLE
// ACCURACY").
//
// The inexact operator here is the nonsymmetric tridiagonal matrix
// A = tridiag(-1/2, 3, -1) with every entry of the product rounded to
// single precision before it is returned: a deterministic, nonlinear
// perturbation of relative size eps_float / 2 per entry, the same kind of
// error a finite-differenced Jacobian action or a reduced-precision kernel
// makes. Because `b - A x` is itself evaluated through that operator, its
// norm cannot be driven below a level of order eps_float * ||b||, whatever
// the Krylov method does, so a tolerance of 1e-12 is unreachable. The same
// matrix applied in double precision is the control.
//
// What each test pins:
//   ReportsAttainableAccuracyAsAFailureByDefault -- the default return is
//     Failure{E-LINALG-ATTAINABLE-ACCURACY}, never a success a caller could
//     mistake for convergence, and it comes long before max_iterations.
//   ReturnsTheCycleStartIterateWhenAsked -- with
//     return_at_attainable_accuracy the report says converged == false,
//     at_attainable_accuracy == true; its residual_norm is EXACTLY the
//     residual of the returned x recomputed through the same operator
//     (the same floating-point operations in the same order); that
//     residual is above the tolerance and below the initial residual, so
//     the solve did restart and make true progress before stopping. No
//     Arnoldi step follows an estimate that met the tolerance while the
//     recomputed residual did not: the cycle ends there.
//   ExactOperatorStillConverges / ReachableToleranceStillConverges -- no
//     false positive: the double-precision operator converges at 1e-12,
//     and the inexact one converges at a tolerance above its error level.
//   RestartStagnationIsNotAttainableAccuracy -- GMRES(1) on a plane
//     rotation makes no progress at all (the classical restart stagnation,
//     Saad and Schultz 1986, Section 3), but its recurrence never reports
//     convergence either: that still ends in E-LINALG-NO-CONVERGENCE.

#include "fuse/linalg/gmres.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/operator.hpp"
#include "fuse/linalg/preconditioner.hpp"

namespace {

using fuse::linalg::DistVector;
using fuse::linalg::GmresOptions;
using fuse::linalg::IdentityPreconditioner;
using fuse::linalg::MatrixFreeOperator;

constexpr std::size_t kN = 40;

DistVector make_vector() {
  return DistVector(kN, fuse::linalg::make_serial_exchanger(), fuse::linalg::make_serial_reducer());
}

// y <- tridiag(-1/2, 3, -1) x, optionally rounded entrywise to float.
MatrixFreeOperator tridiagonal_operator(bool round_to_float) {
  return MatrixFreeOperator([round_to_float](DistVector& x, DistVector& y) -> fuse::Result<void> {
    for (std::size_t i = 0; i < kN; ++i) {
      double v = 3.0 * x.data()[i];
      if (i > 0) v -= 0.5 * x.data()[i - 1];
      if (i + 1 < kN) v -= x.data()[i + 1];
      y.data()[i] = round_to_float ? static_cast<double>(static_cast<float>(v)) : v;
    }
    return fuse::Result<void>::success();
  });
}

DistVector right_hand_side() {
  DistVector b = make_vector();
  for (std::size_t i = 0; i < kN; ++i) b.data()[i] = 1.0 + std::sin(0.7 * static_cast<double>(i));
  return b;
}

double true_residual_norm(const MatrixFreeOperator& a, DistVector& x, const DistVector& b) {
  DistVector ax = make_vector();
  EXPECT_TRUE(a.apply(x, ax).ok());
  DistVector r(b);
  r.copy_from(b);
  r.axpy(-1.0, ax);
  return r.norm2();
}

TEST(GmresAttainableAccuracy, ReportsAttainableAccuracyAsAFailureByDefault) {
  const auto a = tridiagonal_operator(/*round_to_float=*/true);
  const DistVector b = right_hand_side();
  DistVector x = make_vector();
  x.set_zero();
  GmresOptions opt;
  opt.rel_tol = 1e-12;
  opt.restart = 20;
  opt.max_iterations = 5000;
  std::size_t last_iteration = 0;
  opt.monitor = [&last_iteration](std::size_t it, double) { last_iteration = it; };
  const auto result = fuse::linalg::gmres(a, IdentityPreconditioner{}, x, b, opt);
  ASSERT_FALSE(result.ok()) << "an unreachable tolerance must not be reported as a success";
  EXPECT_EQ(result.error().code, "E-LINALG-ATTAINABLE-ACCURACY") << result.error().message;
  EXPECT_LT(last_iteration, opt.max_iterations)
      << "the condition must be detected, not found by exhausting the iteration budget";
}

TEST(GmresAttainableAccuracy, ReturnsTheCycleStartIterateWhenAsked) {
  const auto a = tridiagonal_operator(/*round_to_float=*/true);
  const DistVector b = right_hand_side();
  const double b_norm = b.norm2();
  // Two cycle lengths. With restart = 20 the first cycle ends before its
  // estimate reaches the tolerance. With restart = kN (full GMRES on this
  // system) the estimate DOES reach it inside the first cycle while the
  // recomputed residual is still above the tolerance; that cycle reduced
  // the true residual from ||b||, so it must be restarted from, and only a
  // later cycle that makes no true progress is the attainable accuracy.
  for (const std::size_t restart : {std::size_t{20}, kN}) {
    DistVector x = make_vector();
    x.set_zero();
    GmresOptions opt;
    opt.rel_tol = 1e-12;
    opt.restart = restart;
    opt.max_iterations = 5000;
    opt.return_at_attainable_accuracy = true;
    // Once the rotation estimate has met the tolerance and the recomputed
    // residual has not, the cycle ends: the next monitor report must carry
    // the SAME iteration count (the restart's fresh residual, or the final
    // report), never a further Arnoldi step judged by that estimate.
    std::size_t steps_after_a_refuted_estimate = 0;
    bool previous_met_tolerance = false;
    std::size_t previous_iteration = 0;
    opt.monitor = [&](std::size_t it, double r) {
      if (previous_met_tolerance && it != previous_iteration) ++steps_after_a_refuted_estimate;
      previous_met_tolerance = r <= opt.rel_tol * b_norm;
      previous_iteration = it;
    };
    const auto result = fuse::linalg::gmres(a, IdentityPreconditioner{}, x, b, opt);
    ASSERT_TRUE(result.ok()) << result.error().message;
    const auto& report = result.value();
    EXPECT_EQ(steps_after_a_refuted_estimate, 0u) << "restart=" << restart;
    EXPECT_FALSE(report.converged) << "restart=" << restart;
    EXPECT_TRUE(report.at_attainable_accuracy) << "restart=" << restart;
    EXPECT_LT(report.iterations, opt.max_iterations) << "restart=" << restart;
    // Exact: the same operations in the same order as the solver's own
    // restart-boundary residual of this x.
    EXPECT_EQ(report.residual_norm, true_residual_norm(a, x, b)) << "restart=" << restart;
    EXPECT_GT(report.residual_norm, opt.rel_tol * b_norm) << "otherwise it would have converged";
    EXPECT_LT(report.residual_norm, b_norm)
        << "restart=" << restart
        << ": the returned iterate must be better than the initial guess -- a cycle that made "
           "true progress is restarted from, not reported as the attainable accuracy";
    std::cout << "[ GMRES    ] attainable accuracy: restart=" << restart << " iterations=" << report.iterations
              << " relative residual=" << report.residual_norm / b_norm << "\n";
  }
}

TEST(GmresAttainableAccuracy, ExactOperatorStillConverges) {
  const auto a = tridiagonal_operator(/*round_to_float=*/false);
  const DistVector b = right_hand_side();
  DistVector x = make_vector();
  x.set_zero();
  GmresOptions opt;
  opt.rel_tol = 1e-12;
  opt.restart = 20;
  opt.max_iterations = 5000;
  opt.return_at_attainable_accuracy = true;
  const auto result = fuse::linalg::gmres(a, IdentityPreconditioner{}, x, b, opt);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_TRUE(result.value().converged);
  EXPECT_FALSE(result.value().at_attainable_accuracy);
  EXPECT_LE(result.value().residual_norm, opt.rel_tol * b.norm2());
}

TEST(GmresAttainableAccuracy, ReachableToleranceStillConverges) {
  const auto a = tridiagonal_operator(/*round_to_float=*/true);
  const DistVector b = right_hand_side();
  DistVector x = make_vector();
  x.set_zero();
  GmresOptions opt;
  // Entrywise rounding to float perturbs A x by at most eps_float / 2 of
  // each entry, so ||fl(A x) - A x|| <= (eps_float / 2) ||A x||; with
  // ||A x|| of the order of ||b|| a tolerance four orders above eps_float
  // is far above the operator's error.
  opt.rel_tol = 1e4 * static_cast<double>(std::numeric_limits<float>::epsilon());
  opt.restart = 20;
  opt.max_iterations = 5000;
  const auto result = fuse::linalg::gmres(a, IdentityPreconditioner{}, x, b, opt);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_TRUE(result.value().converged);
  EXPECT_FALSE(result.value().at_attainable_accuracy);
}

TEST(GmresAttainableAccuracy, RestartStagnationIsNotAttainableAccuracy) {
  // A plane rotation by 90 degrees: A r is orthogonal to r, so the single
  // Arnoldi step of GMRES(1) cannot reduce the residual, cycle after cycle.
  MatrixFreeOperator rotation([](DistVector& x, DistVector& y) -> fuse::Result<void> {
    y.data()[0] = -x.data()[1];
    y.data()[1] = x.data()[0];
    return fuse::Result<void>::success();
  });
  DistVector b(2, fuse::linalg::make_serial_exchanger(), fuse::linalg::make_serial_reducer());
  b.data()[0] = 1.0;
  b.data()[1] = 0.0;
  DistVector x(b);
  x.set_zero();
  GmresOptions opt;
  opt.rel_tol = 1e-10;
  opt.restart = 1;
  opt.max_iterations = 25;
  const auto result = fuse::linalg::gmres(rotation, IdentityPreconditioner{}, x, b, opt);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, "E-LINALG-NO-CONVERGENCE") << result.error().message;
}

}  // namespace
