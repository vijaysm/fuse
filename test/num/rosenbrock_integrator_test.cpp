// GRK4A's embedded "d prime" row could be re-derived from Kaps & Rentrop 1979
// (Numer. Math. 33, 55-68). The paper itself IS locatable (numerik.mi.fu-berlin.de's
// own scanned reproduction), but its numeric table was not reliably extractable to
// full precision (a scanned 1979 paper, not a machine-readable table), so
// embedded_order stays nullopt for GRK4A, recorded here rather than silently left
// unexplained.

#include "fuse/num/rosenbrock_integrator.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <string>

#include <gtest/gtest.h>

namespace fuse::num {
namespace {

TemporalProblem<double> dahlquist_problem(double a, double b) {
  TemporalProblem<double> p;
  p.dimension = 2;
  p.rhs = [a, b](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = a * y[0] - b * y[1];
    out[1] = b * y[0] + a * y[1];
    return Result<void>::success();
  };
  p.jacobian_action = [a, b](double, std::span<const double>, std::span<const double> v,
                            std::span<double> out) -> Result<void> {
    out[0] = a * v[0] - b * v[1];
    out[1] = b * v[0] + a * v[1];
    return Result<void>::success();
  };
  return p;
}

std::array<double, 2> dahlquist_exact(double a, double b, double t) {
  const double mag = std::exp(a * t);
  return {mag * std::cos(b * t), mag * std::sin(b * t)};
}

TemporalProblem<double> logistic_problem() {
  TemporalProblem<double> p;
  p.dimension = 1;
  p.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = y[0] * (1.0 - y[0]);
    return Result<void>::success();
  };
  return p;
}

double logistic_exact(double y0, double t) {
  const double k = (1.0 - y0) / y0;
  return 1.0 / (1.0 + k * std::exp(-t));
}

struct OrderFit { double slope = 0.0; double slope_stderr = 0.0; };

OrderFit fit_order(const std::vector<double>& h_values, const std::vector<double>& errors) {
  const int n = static_cast<int>(h_values.size());
  std::vector<double> lx(n), ly(n);
  for (int i = 0; i < n; ++i) { lx[i] = std::log(h_values[i]); ly[i] = std::log(errors[i]); }
  double mean_x = 0.0, mean_y = 0.0;
  for (int i = 0; i < n; ++i) { mean_x += lx[i]; mean_y += ly[i]; }
  mean_x /= n; mean_y /= n;
  double sxx = 0.0, sxy = 0.0;
  for (int i = 0; i < n; ++i) { sxx += (lx[i]-mean_x)*(lx[i]-mean_x); sxy += (lx[i]-mean_x)*(ly[i]-mean_y); }
  const double slope = sxy / sxx;
  const double intercept = mean_y - slope * mean_x;
  double ssr = 0.0;
  for (int i = 0; i < n; ++i) { const double resid = ly[i]-(slope*lx[i]+intercept); ssr += resid*resid; }
  const double dof = static_cast<double>(n - 2);
  const double mse = (dof > 0.0) ? ssr/dof : 0.0;
  return {slope, std::sqrt(mse/sxx)};
}

std::vector<double> decade_of_h(double h_max, double h_min_ratio, int n_points) {
  std::vector<double> hs(n_points);
  const double log_ratio = std::log10(h_min_ratio);
  for (int i = 0; i < n_points; ++i) hs[i] = h_max * std::pow(10.0, log_ratio * static_cast<double>(i) / (n_points - 1));
  return hs;
}

constexpr double kOrderToleranceMultiplier = 8.0;

void expect_order(const OrderFit& fit, double expected_order, const char* method) {
  const double tol = kOrderToleranceMultiplier * fit.slope_stderr;
  EXPECT_NEAR(fit.slope, expected_order, tol)
      << method << ": fitted order " << fit.slope << " +/- " << fit.slope_stderr
      << " (tolerance " << tol << ")";
}

template <class MakeIntegrator, class Exact>
std::pair<double, double> fixed_step_final_error(MakeIntegrator make_integrator, TemporalProblem<double>& problem,
                              std::vector<double> y0, double h, double t_end, Exact exact) {
  auto integrator = make_integrator();
  std::vector<double> y = y0;
  const auto n_steps = static_cast<long>(std::lround(t_end / h));
  const double actual_h = t_end / static_cast<double>(n_steps);
  double t = 0.0;
  for (long i = 0; i < n_steps; ++i) {
    auto r = integrator.step(problem, t, actual_h, y);
    if (!r.ok()) { ADD_FAILURE() << "unexpected refusal: " << r.error().code << ": " << r.error().message; return {-1.0, actual_h}; }
    t = r.value().t_new;
  }
  const auto ex = exact(t);
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) { const double d = y[i]-ex[i]; sum_sq += d*d; }
  return {std::sqrt(sum_sq), actual_h};
}

template <class MakeIntegrator>
std::pair<double, double> fixed_step_final_error_1d(MakeIntegrator make_integrator,
                                                    TemporalProblem<double>& problem, double y0,
                                                    double h, double t_end, double exact_t_end) {
  auto integrator = make_integrator();
  std::vector<double> y{y0};
  const auto n_steps = static_cast<long>(std::lround(t_end / h));
  const double actual_h = t_end / static_cast<double>(n_steps);
  double t = 0.0;
  for (long i = 0; i < n_steps; ++i) {
    auto r = integrator.step(problem, t, actual_h, y);
    if (!r.ok()) { ADD_FAILURE() << "unexpected refusal: " << r.error().code << ": " << r.error().message; return {-1.0, actual_h}; }
    t = r.value().t_new;
  }
  return {std::fabs(y[0] - exact_t_end), actual_h};
}

}  // namespace

TEST(RosenbrockIntegrator, RefusesWhenRhsIsUnset) {
  RosenbrockIntegrator<double> integ(ros22_tableau(), 1e-6);
  TemporalProblem<double> problem;
  std::vector<double> y{1.0, 0.0};
  auto r = integ.step(problem, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kNoRhsCode);
}

TEST(RosenbrockIntegrator, RefusesAWrongFamilyTableau) {
  RosenbrockIntegrator<double> integ(sdirk22_tableau(), 1e-6);
  auto problem = dahlquist_problem(-1.0, 0.0);
  std::vector<double> y{1.0, 0.0};
  auto r = integ.step(problem, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kWrongFamilyCode);
}

// The stage solve's own failure is a NAMED refusal, not a silent fold of a
// half-converged k_i into the step. The operator (I - h gamma J) for the
// Dahlquist rotation (a = -1, b = 2) has the complex-conjugate eigenvalues
// 1 - h gamma (a +- i b); no real vector is an eigenvector, so one Arnoldi
// step cannot reduce the residual below the stage tolerance
// (rtol * factor = 1e-6 * 1e-2 = 1e-8): the one-step residual ratio of a real
// linear polynomial on that pair is of order sin(arg lambda), about 0.3 here,
// MEASURED by the solver's own report (it refuses with E-LINALG-NO-CONVERGENCE).
// With the default budget the same step succeeds, which pins that the budget
// and not the problem is what fires the refusal.
TEST(RosenbrockIntegrator, StageSolveFailureIsRefusedNotFoldedIntoTheStep) {
  auto problem = dahlquist_problem(-1.0, 2.0);
  std::vector<double> y{1.0, 0.0};
  {
    RosenbrockIntegrator<double> ok(ros3p_tableau(), /*step_rtol=*/1e-6);
    std::vector<double> y_ok = y;
    ASSERT_TRUE(ok.step(problem, 0.0, 0.5, y_ok).ok());
  }
  RosenbrockIntegrator<double> integ(ros3p_tableau(), /*step_rtol=*/1e-6);
  integ.set_stage_gmres_max_iterations(1);
  const auto r = integ.step(problem, 0.0, 0.5, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kStageSolveFailedCode);
  EXPECT_NE(r.error().message.find("E-LINALG-NO-CONVERGENCE"), std::string::npos) << r.error().message;
  EXPECT_NE(r.error().message.find("stage 0"), std::string::npos) << r.error().message;
  // Nothing was folded into the state.
  EXPECT_EQ(y[0], 1.0);
  EXPECT_EQ(y[1], 0.0);
}

TEST(RosenbrockIntegrator, RefusesADimensionMismatch) {
  RosenbrockIntegrator<double> integ(ros22_tableau(), 1e-6);
  auto problem = dahlquist_problem(-1.0, 0.0);
  problem.dimension = 7;
  std::vector<double> y{1.0, 0.0};
  auto r = integ.step(problem, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kDimensionMismatchCode);
}

TEST(RosenbrockIntegrator, JacobianActionIsActuallyUsedWhenSupplied) {
  // A problem whose own `jacobian_action` is called counts its own invocations; if
  // RosenbrockIntegrator silently fell back to MFFD despite one being supplied, this
  // counter would stay zero.
  int jac_calls = 0;
  auto problem = dahlquist_problem(-1.0, 2.0);
  auto base_jac = problem.jacobian_action;
  problem.jacobian_action = [&, base_jac](double t, std::span<const double> y, std::span<const double> v,
                                          std::span<double> out) -> Result<void> {
    ++jac_calls;
    return base_jac(t, y, v, out);
  };
  RosenbrockIntegrator<double> integ(ros22_tableau(), 1e-6);
  std::vector<double> y{1.0, 0.0};
  auto r = integ.step(problem, 0.0, 0.1, y);
  ASSERT_TRUE(r.ok()) << r.error().code;
  EXPECT_GT(jac_calls, 0);
}

struct DahlquistCase { const char* name; ButcherTableau tableau; int order; double h_max; int n_points; };

TEST(RosenbrockIntegrator, SixRosenbrockMethodsConvergeAtTheirDeclaredOrderOnDahlquist) {
  const std::vector<DahlquistCase> cases = {
      {"ROS22", ros22_tableau(), 2, 0.05, 6},
      {"GRK4A", grk4a_tableau(), 4, 0.2, 6},
      {"GRK4T", grk4t_tableau(), 4, 0.2, 6},
      {"GRK4Sn", grk4sn_tableau(), 4, 0.2, 6},
      {"ROS3P", ros3p_tableau(), 3, 0.1, 6},
      {"RODASP", rodasp_tableau(), 4, 0.12, 6},
  };
  auto problem = dahlquist_problem(-2.0, 3.0);
  for (const auto& c : cases) {
    const auto hs = decade_of_h(c.h_max, 1e-2, c.n_points);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error(
          [&] { return RosenbrockIntegrator<double>(c.tableau, 1e-10); },
          problem, {1.0, 0.0}, h, 1.0,
          [](double t) { auto e = dahlquist_exact(-2.0, 3.0, t); return std::array<double,2>{e[0], e[1]}; });
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    printf("[Rosenbrock/Dahlquist] %-8s errors: ", c.name);
    for (double e : errors) printf("%.3e ", e);
    printf("\n");
    auto fit = fit_order(actual_hs, errors);
    printf("[Rosenbrock/Dahlquist] %-8s fitted order = %.5f +/- %.5f (expected %d)\n",
           c.name, fit.slope, fit.slope_stderr, c.order);
    expect_order(fit, static_cast<double>(c.order), c.name);
  }
}

TEST(RosenbrockIntegrator, ConvergesAtDeclaredOrderOnTheLogisticEquation) {
  const std::vector<DahlquistCase> cases = {
      {"ROS22", ros22_tableau(), 2, 0.1, 6},
      {"GRK4A", grk4a_tableau(), 4, 0.3, 6},
      {"RODASP", rodasp_tableau(), 4, 0.3, 6},
  };
  for (const auto& c : cases) {
    auto problem = logistic_problem();
    const auto hs = decade_of_h(c.h_max, 1e-2, c.n_points);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error_1d(
          [&] { return RosenbrockIntegrator<double>(c.tableau, 1e-10); },
          problem, 0.5, h, 2.0, logistic_exact(0.5, 2.0));
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    auto fit = fit_order(actual_hs, errors);
    printf("[Rosenbrock/logistic] %-8s fitted order = %.5f +/- %.5f (expected %d)\n",
           c.name, fit.slope, fit.slope_stderr, c.order);
    expect_order(fit, static_cast<double>(c.order), c.name);
  }
}

TEST(RosenbrockIntegrator, InexactStageSolveOrderDegradationMeasuredAtTwoRatios) {
  // Section 6.2's own degradation: a LOOSER stage GMRES tolerance leaks linear-solve
  // error into the step, corrupting the method's own classical order once that error
  // is no longer small relative to the smallest h's own truncation error. MEASURED
  // directly: `step_rtol * factor` with the design's starting factor (1e-2) and a
  // tighter one (1e-4) both solve the n=2 Dahlquist system to round-off regardless
  // (GMRES is EXACT within 2 Arnoldi steps for a 2-dimensional system, so neither
  // factor binds) -- a genuinely LOOSE tolerance (`rel_tol ~= 0.3`, far above the
  // design's starting factor) is the one that visibly degrades the fitted order
  // below 3 here, which is the comparison this test actually needs to show the EFFECT
  // exists, not merely that two already-tight settings agree.
  auto problem = dahlquist_problem(-2.0, 3.0);
  const auto tableau = ros3p_tableau();  // order 3, a representative Rosenbrock case
  struct Setting { const char* label; double step_rtol; double factor; };
  const std::vector<Setting> settings = {
      {"tight (default, 1e-2)", 1e-10, 1e-2},
      {"tighter (1e-4)", 1e-10, 1e-4},
      {"loose (rel_tol~0.3)", 1.0, 0.3},
  };
  std::vector<double> fitted;
  for (const auto& s : settings) {
    const auto hs = decade_of_h(0.1, 1e-2, 6);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error(
          [&] { return RosenbrockIntegrator<double>(tableau, s.step_rtol, s.factor); },
          problem, {1.0, 0.0}, h, 1.0,
          [](double t) { auto e = dahlquist_exact(-2.0, 3.0, t); return std::array<double,2>{e[0], e[1]}; });
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    auto fit = fit_order(actual_hs, errors);
    printf("[Rosenbrock/ROS3P %-24s] fitted order = %.5f +/- %.5f (expected 3)\n",
           s.label, fit.slope, fit.slope_stderr);
    fitted.push_back(fit.slope);
  }
  // The two TIGHT settings both show the full declared order (within the usual band).
  EXPECT_NEAR(fitted[0], 3.0, 0.3);
  EXPECT_NEAR(fitted[1], 3.0, 0.3);
  // The LOOSE setting measurably degrades the order -- the effect this test exists to
  // show, not just that tightening never hurts.
  EXPECT_LT(fitted[2], 2.5) << "a rel_tol~0.3 stage solve should visibly degrade ROS3P's "
                               "own order 3 below what the two tight settings show";
}

}  // namespace fuse::num
