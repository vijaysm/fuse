// Dense output.
//
// continuous_extension.hpp's own top comment states the deviation: this ships the
// cubic Hermite interpolant, not DOPRI5's own method-specific quartic (a reliable
// source for that polynomial's exact coefficients was not found). This
// file MEASURES the cubic Hermite's actual interior order rather than asserting a
// label -- the project's own standing rule, "measured, not asserted".

#include "fuse/num/continuous_extension.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/num/explicit_integrator.hpp"

namespace fuse::num {
namespace {

// Real Dahlquist y' = lambda*y: exact solution and its own derivative are both
// closed-form, so both step endpoints AND the interior reference used for the order
// fit are exact, isolating the interpolant's own error from any integration error.
TemporalProblem<double> dahlquist_real(double lambda) {
  TemporalProblem<double> p;
  p.dimension = 1;
  p.rhs = [lambda](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = lambda * y[0];
    return Result<void>::success();
  };
  return p;
}

// One accepted DOPRI45 step of size h from (t0, y0), returning (y1, f0, f1) -- f0/f1
// are the SAME rhs evaluations continuous_extension.hpp's own top comment describes a
// caller as already holding (f0 is the tableau's own first-stage evaluation, f1 the
// FSAL last stage DOPRI45 already carries per the corrected is_FSAL); this
// test does not reach into ExplicitIntegrator's own internals for them, it simply
// evaluates the SAME rhs function at the two endpoints, which is externally
// observable and all continuous_extension.hpp's own interface requires.
struct SteppedInterval {
  double y0 = 0.0, f0 = 0.0, y1 = 0.0, f1 = 0.0;
};

SteppedInterval take_one_dopri45_step(double lambda, double t0, double y0, double h) {
  TemporalProblem<double> problem = dahlquist_real(lambda);
  ExplicitIntegrator<double> integ(dopri45_tableau());  // fixed-step: h is this test's own control variable
  std::vector<double> state{y0};
  auto rec = integ.step(problem, t0, h, std::span<double>(state));
  EXPECT_TRUE(rec.ok());
  EXPECT_TRUE(rec.value().accepted);
  SteppedInterval out;
  out.y0 = y0;
  out.y1 = state[0];
  std::vector<double> f0v(1), f1v(1);
  problem.rhs(t0, std::span<const double>(&out.y0, 1), std::span<double>(f0v));
  problem.rhs(t0 + h, std::span<const double>(&out.y1, 1), std::span<double>(f1v));
  out.f0 = f0v[0];
  out.f1 = f1v[0];
  return out;
}

// Same log-log least-squares slope fit used throughout the integrator order tests
// (explicit_integrator_test.cpp, dirk_integrator_test.cpp, rosenbrock_integrator_test.cpp).
struct OrderFit {
  double slope = 0.0;
  double slope_stderr = 0.0;
  int n_points = 0;
};

OrderFit fit_order(const std::vector<double>& hs, const std::vector<double>& errs) {
  OrderFit fit;
  std::vector<double> xs, ys;
  for (std::size_t i = 0; i < hs.size(); ++i) {
    if (errs[i] > 0.0) {
      xs.push_back(std::log(hs[i]));
      ys.push_back(std::log(errs[i]));
    }
  }
  const int n = static_cast<int>(xs.size());
  fit.n_points = n;
  if (n < 2) return fit;
  double mean_x = 0.0, mean_y = 0.0;
  for (int i = 0; i < n; ++i) { mean_x += xs[i]; mean_y += ys[i]; }
  mean_x /= n; mean_y /= n;
  double sxx = 0.0, sxy = 0.0;
  for (int i = 0; i < n; ++i) { sxx += (xs[i]-mean_x)*(xs[i]-mean_x); sxy += (xs[i]-mean_x)*(ys[i]-mean_y); }
  fit.slope = sxy / sxx;
  double sum_sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double pred = mean_y + fit.slope * (xs[i] - mean_x);
    sum_sq += (ys[i] - pred) * (ys[i] - pred);
  }
  const double dof = static_cast<double>(n - 2);
  const double resid_var = dof > 0.0 ? sum_sq / dof : 0.0;
  fit.slope_stderr = std::sqrt(resid_var / sxx);
  return fit;
}

// kOrderToleranceMultiplier=8.0: same multiplier used throughout the order tests (the
// regression's own standard error scaled up to absorb the fact that only a decade of
// h values is sampled, not an asymptotic family) -- not a number tuned to this test.
constexpr double kOrderToleranceMultiplier = 8.0;

TEST(ContinuousExtension, CubicHermiteInteriorErrorMeasuredOrder) {
  const double lambda = -1.3;
  const double t0 = 0.0;
  const double y0 = 2.0;
  const double theta = 0.5;  // interior sample point, fixed across the h sweep

  std::vector<double> hs, errs;
  for (int k = 2; k <= 9; ++k) {
    const double h = std::pow(2.0, -static_cast<double>(k));  // decade-ish sweep in log2(h)
    SteppedInterval seg = take_one_dopri45_step(lambda, t0, y0, h);
    CubicHermiteExtension<double> hermite(h, {seg.y0}, {seg.f0}, {seg.y1}, {seg.f1});
    const double y_interp = hermite.evaluate(theta)[0];
    const double y_exact = y0 * std::exp(lambda * theta * h);
    hs.push_back(h);
    errs.push_back(std::abs(y_interp - y_exact));
  }
  OrderFit fit = fit_order(hs, errs);
  std::printf("CubicHermiteInteriorErrorMeasuredOrder: fitted slope=%.4f stderr=%.4f n=%d\n",
              fit.slope, fit.slope_stderr, fit.n_points);
  // The cubic Hermite interpolant (value+derivative match at both ends) has a textbook
  // LOCAL truncation error of O(h^4) when the data it interpolates (y0,f0,y1,f1) is
  // itself exact to the order DOPRI45 delivers here (5th order, far above 4th) --
  // measuring, not assuming, whether that textbook local rate is what this
  // concrete evaluate() achieves.
  const double tol = kOrderToleranceMultiplier * fit.slope_stderr;
  EXPECT_NEAR(fit.slope, 4.0, tol)
      << "measured order " << fit.slope << " +/- " << fit.slope_stderr
      << " (tol " << tol << ") vs the cubic Hermite interpolant's own textbook local order 4";
}

TEST(ContinuousExtension, EndpointsAreExactToRoundoff) {
  const double lambda = -0.7;
  const double h = 0.1;
  SteppedInterval seg = take_one_dopri45_step(lambda, 0.0, 1.0, h);
  CubicHermiteExtension<double> hermite(h, {seg.y0}, {seg.f0}, {seg.y1}, {seg.f1});
  EXPECT_NEAR(hermite.evaluate(0.0)[0], seg.y0, 1e-14);
  EXPECT_NEAR(hermite.evaluate(1.0)[0], seg.y1, 1e-14);
}

}  // namespace
}  // namespace fuse::num
