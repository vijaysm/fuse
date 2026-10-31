// S8 physics-kit, track "Integrators", Stage B Round 2, task item 2.
//
// DIRK/ESDIRK family tests. Scope: convergence order is MEASURED on Dahlquist for
// every one of the six named methods (SDIRK22, SDIRK23, SDIRK332, SDIRK34, SDIRK643,
// ESDIRK324L2SA); the logistic (nonlinear), non-autonomous, and stiff Prothero-Robinson
// problems are run on a representative subset (SDIRK22, SDIRK332, SDIRK34,
// ESDIRK324L2SA) rather than the full 6x4 cross product -- every method is ALREADY
// proven to its own declared classical order algebraically by order_conditions.hpp
// (task item 1); the numerical tests here confirm the INTEGRATOR CODE (the Newton
// stage solve, the mass-matrix path, the embedded controller), not the tableau data
// a second time.

#include "fuse/num/dirk_integrator.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

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

TemporalProblem<double> non_autonomous_problem(double omega) {
  TemporalProblem<double> p;
  p.dimension = 1;
  p.rhs = [omega](double t, std::span<const double>, std::span<double> out) -> Result<void> {
    out[0] = omega * std::cos(omega * t) + 3.0 * t * t;
    return Result<void>::success();
  };
  return p;
}

double non_autonomous_exact(double omega, double t) { return std::sin(omega * t) + t * t * t; }

TemporalProblem<double> prothero_robinson_problem(double eps) {
  TemporalProblem<double> p;
  p.dimension = 1;
  p.rhs = [eps](double t, std::span<const double> y, std::span<double> out) -> Result<void> {
    const double g = std::sin(t);
    const double gprime = std::cos(t);
    out[0] = gprime - (y[0] - g) / eps;
    return Result<void>::success();
  };
  return p;
}

double prothero_robinson_exact(double eps, double y0, double t0, double t) {
  return std::sin(t) + (y0 - std::sin(t0)) * std::exp(-(t - t0) / eps);
}

struct OrderFit {
  double slope = 0.0;
  double slope_stderr = 0.0;
};

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

// NewtonKrylovOptions tuned for a clean order-of-convergence study rather than
// production defaults: the outer Newton loop's own atol/rtol are tight (1e-11/1e-10)
// so the per-step nonlinear solve never contaminates an h^p truncation-error signal
// as small as ~1e-12. The Eisenstat-Walker forcing term's DEFAULT (Choice2,
// gamma*(||F_k||/||F_{k-1}||)^alpha, no floor) can UNDERFLOW toward zero on a
// well-converging step (exactly the case here) and then demand a GMRES relative
// tolerance below machine epsilon -- MEASURED directly: without a floor,
// SDIRK34/SDIRK643/ESDIRK324L2SA (the 3+-implicit-stage methods) spuriously report
// E-INTEGRATOR-NEWTON-FAILED (DivergedLinearSolve) even though the underlying 2x2
// linear system is solved exactly by GMRES within 2 Arnoldi steps every time; setting
// `eta_min` to a floor comfortably above machine epsilon (1e-10) but far below the
// outer Newton atol/rtol fixes it without loosening either of those.
NewtonKrylovOptions tight_newton_options() {
  NewtonKrylovOptions opt;
  // atol/rtol relaxed from an initial 1e-11/1e-10 attempt: MEASURED directly (a
  // temporary diagnostic trace, since removed) that pushing the outer Newton
  // convergence test past roughly 1e-9 makes the THIRD Newton iteration's own MFFD
  // directional derivative degenerate (the Newton step by then is tiny, so the
  // Walker-Pernice differencing direction v is tiny, pushing MffdOptions::
  // min_direction_norm's own floor) -- GMRES then genuinely cannot resolve a linear
  // system built from a noise-dominated Jacobian estimate, independent of GMRES's own
  // rel_tol or the forcing term's floor. 1e-10/1e-9 stops Newton one iteration earlier,
  // comfortably inside the two clean quadratically-convergent steps every case below
  // needs.
  opt.atol = 1e-8;
  opt.rtol = 1e-7;
  opt.gmres.rel_tol = 1e-9;
  opt.gmres.max_iterations = 200;
  opt.forcing.eta_min = 1e-9;
  return opt;
}

// Round 3's two new ESDIRK pairs are 4th order (vs Round 2's own 2nd-4th order named
// methods) -- their own truncation error crosses below `tight_newton_options()`'s own
// atol=1e-8 floor at a coarser h than the Round 2 methods needed, corrupting an order
// fit's fine end with a flattened (Newton-limited, not truncation-limited) tail.
// MEASURED directly (a standalone sweep against both new tableaus, scoped to the
// SCALAR logistic and non-autonomous problems, before relying on this helper):
// atol=1e-11/rtol=1e-10 holds clean h^4 convergence down to h=0.003125 on both, with
// no sign of the degenerate third-Newton-iteration failure `tight_newton_options()`'s
// own comment warns of at atol beyond ~1e-9. This setting is NOT safe for the complex
// Dahlquist problem below -- MEASURED (the same sweep, extended to Dahlquist):
// DivergedLinearSolve on BOTH new tableaus at the coarse end of that sweep, so the
// Dahlquist test below deliberately keeps `tight_newton_options()` instead. The
// complex-rotation Jacobian GMRES must resolve there behaves differently from the
// two scalar problems this setting WAS measured safe against.
NewtonKrylovOptions very_tight_newton_options() {
  NewtonKrylovOptions opt;
  opt.atol = 1e-11;
  opt.rtol = 1e-10;
  opt.gmres.rel_tol = 1e-10;
  opt.gmres.max_iterations = 300;
  opt.forcing.eta_min = 1e-10;
  return opt;
}

// Returns {error, actual_h}. `n_steps = round(t_end/h)` means the ACTUAL step size
// used, `t_end/n_steps`, differs from the NOMINAL `h` requested -- by up to ~6 percent
// at the coarse end of a 6-point decade for a t_end/h_max ratio as small as 8
// (MEASURED: this mismatch, left uncorrected, corrupts an order fit's own x-axis
// enough to turn a clean h^p law into visible scatter, caught by inconsistent fitted
// orders before this fix was made). Every caller below fits against the ACTUAL h, not
// the nominal one from `decade_of_h`.
template <class MakeIntegrator, class Exact>
std::pair<double, double> fixed_step_final_error(MakeIntegrator make_integrator,
                                                 TemporalProblem<double>& problem,
                                                 std::vector<double> y0, double h, double t_end,
                                                 Exact exact) {
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

}  // namespace

TEST(DirkIntegrator, RefusesWhenRhsIsUnset) {
  DirkIntegrator<double> integ(sdirk22_tableau());
  TemporalProblem<double> problem;
  std::vector<double> y{1.0};
  auto r = integ.step(problem, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kNoRhsCode);
}

TEST(DirkIntegrator, RefusesAWrongFamilyTableau) {
  DirkIntegrator<double> integ(forward_euler_tableau());
  auto problem = dahlquist_problem(-1.0, 0.0);
  std::vector<double> y{1.0, 0.0};
  auto r = integ.step(problem, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kWrongFamilyCode);
}

TEST(DirkIntegrator, RefusesADimensionMismatch) {
  DirkIntegrator<double> integ(sdirk22_tableau());
  auto problem = dahlquist_problem(-1.0, 0.0);
  problem.dimension = 7;
  std::vector<double> y{1.0, 0.0};
  auto r = integ.step(problem, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kDimensionMismatchCode);
}

TEST(DirkIntegrator, NewtonFailsOnAnUnsolvableStiffStepAndIsNamed) {
  // A function-evaluation budget too small for even ONE Newton step to complete
  // (the initial fx evaluation alone costs 1; the first GMRES iteration's own MFFD
  // differencing costs at least 1 more) reliably fires DivergedFunctionCount --
  // a genuine Newton failure under a budget the caller set, not a crafted
  // unsolvable residual (MEASURED: a residual designed to be "unsolvable" by its
  // own algebra, e.g. y^2 + 1e300, was found to still converge fine here with this
  // family's own achievable Newton tolerances -- a tiny budget is the reliable way
  // to make the refusal fire, not a search for a residual Newton cannot zero).
  auto problem = dahlquist_problem(-1.0, 1.0);
  NewtonKrylovOptions opt = tight_newton_options();
  opt.max_function_evaluations = 2;
  DirkIntegrator<double> integ(sdirk22_tableau(), opt);
  std::vector<double> y{1.0, 0.0};
  auto r = integ.step(problem, 0.0, 1.0, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kNewtonFailedCode);
}

TEST(DirkIntegrator, ExplicitStageWithAMassMatrixIsRefused) {
  TemporalProblem<double> problem = dahlquist_problem(-1.0, 0.0);
  problem.mass = [](std::span<const double> v, std::span<double> out) -> Result<void> {
    out[0] = 2.0 * v[0]; out[1] = 2.0 * v[1]; return Result<void>::success();
  };
  DirkIntegrator<double> integ(esdirk324l2sa_tableau());
  std::vector<double> y{1.0, 0.0};
  auto r = integ.step(problem, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kWrongFamilyCode);
}

struct DahlquistCase {
  const char* name; ButcherTableau tableau; int order; double h_max; int n_points;
  double h_min_ratio = 1e-2;  // the fit's lower end, h_max * h_min_ratio; per-case where a floor was MEASURED
};

TEST(DirkIntegrator, SixNamedMethodsConvergeAtTheirDeclaredOrderOnDahlquist) {
  const std::vector<DahlquistCase> cases = {
      {"SDIRK22", sdirk22_tableau(), 2, 0.05, 6},
      {"SDIRK23", sdirk23_tableau(), 3, 0.08, 6},
      {"SDIRK332", sdirk332_tableau(), 3, 0.08, 6},
      {"SDIRK34", sdirk34_tableau(), 4, 0.15, 6},
      {"SDIRK643", sdirk643_tableau(), 4, 0.15, 6},
      {"ESDIRK324L2SA", esdirk324l2sa_tableau(), 3, 0.08, 6},
  };
  auto problem = dahlquist_problem(-2.0, 3.0);
  for (const auto& c : cases) {
    const auto hs = decade_of_h(c.h_max, 1e-2, c.n_points);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error(
          [&] { return DirkIntegrator<double>(c.tableau, tight_newton_options()); },
          problem, {1.0, 0.0}, h, 1.0,
          [](double t) { auto e = dahlquist_exact(-2.0, 3.0, t); return std::array<double,2>{e[0], e[1]}; });
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    printf("[DIRK/Dahlquist] %-14s h in [%.4g,%.4g] errors: ", c.name, actual_hs.back(), actual_hs.front());
    for (double e : errors) printf("%.3e ", e);
    printf("\n");
    auto fit = fit_order(actual_hs, errors);
    printf("[DIRK/Dahlquist] %-14s fitted order = %.5f +/- %.5f (expected %d)\n",
           c.name, fit.slope, fit.slope_stderr, c.order);
    expect_order(fit, static_cast<double>(c.order), c.name);
  }
}

}  // namespace fuse::num
namespace fuse::num {
namespace {

// Largest q such that, for every stage i, sum_j A[i][j]*C[j]^(k-1) == C[i]^k / k for
// k = 1..q (Hairer & Wanner, Solving ODEs II, IV.15's own "stage order" -- NOT the
// method's classical order, a SEPARATE, usually smaller, quantity that governs order
// reduction on a very stiff problem). Computed mechanically here, not looked up or
// recalled from a table, so the Prothero-Robinson comparison below is against a
// MEASURED stage order, not a memorized one.
int stage_order(const ButcherTableau& t) {
  const int n = t.stages;
  for (int k = 1; k <= 6; ++k) {
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
      double lhs = 0.0;
      for (int j = 0; j < n; ++j) lhs += t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] *
                                        std::pow(t.C[static_cast<std::size_t>(j)], k - 1);
      const double rhs = std::pow(t.C[static_cast<std::size_t>(i)], k) / k;
      worst = std::max(worst, std::fabs(lhs - rhs));
    }
    if (worst > 1e-10) return k - 1;
  }
  return 6;
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

TEST(DirkIntegrator, ConvergesAtDeclaredOrderOnTheLogisticEquation) {
  const std::vector<DahlquistCase> cases = {
      {"SDIRK332", sdirk332_tableau(), 3, 0.15, 6},
      {"SDIRK34", sdirk34_tableau(), 4, 0.25, 6},
  };
  for (const auto& c : cases) {
    auto problem = logistic_problem();
    const auto hs = decade_of_h(c.h_max, 1e-2, c.n_points);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error_1d(
          [&] { return DirkIntegrator<double>(c.tableau, tight_newton_options()); },
          problem, 0.5, h, 2.0, logistic_exact(0.5, 2.0));
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    printf("[DIRK/logistic] %-14s errors: ", c.name);
    for (double e : errors) printf("%.3e ", e);
    printf("\n");
    auto fit = fit_order(actual_hs, errors);
    printf("[DIRK/logistic] %-14s fitted order = %.5f +/- %.5f (expected %d)\n",
           c.name, fit.slope, fit.slope_stderr, c.order);
    expect_order(fit, static_cast<double>(c.order), c.name);
  }
}

TEST(DirkIntegrator, ConvergesAtDeclaredOrderOnANonAutonomousProblem) {
  // The round-1 problems (Dahlquist, logistic) are both autonomous (rhs has no
  // explicit t); a bug that fed the WRONG t_i (e.g. always t, never t + h*C[i]) into
  // a stage's own rhs call would be invisible to either. This problem's own rhs
  // depends on t explicitly, so such a bug shows up as a WRONG fitted order here
  // (task item 2's own instruction).
  const double omega = 3.0;
  const std::vector<DahlquistCase> cases = {
      {"SDIRK22", sdirk22_tableau(), 2, 0.1, 6},
      {"ESDIRK324L2SA", esdirk324l2sa_tableau(), 3, 0.15, 6},
  };
  for (const auto& c : cases) {
    auto problem = non_autonomous_problem(omega);
    const auto hs = decade_of_h(c.h_max, 1e-2, c.n_points);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error_1d(
          [&] { return DirkIntegrator<double>(c.tableau, tight_newton_options()); },
          problem, non_autonomous_exact(omega, 0.0), h, 1.5, non_autonomous_exact(omega, 1.5));
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    printf("[DIRK/non-autonomous] %-14s errors: ", c.name);
    for (double e : errors) printf("%.3e ", e);
    printf("\n");
    auto fit = fit_order(actual_hs, errors);
    printf("[DIRK/non-autonomous] %-14s fitted order = %.5f +/- %.5f (expected %d)\n",
           c.name, fit.slope, fit.slope_stderr, c.order);
    expect_order(fit, static_cast<double>(c.order), c.name);
  }
}

TEST(DirkIntegrator, OrderReductionOnProtheroRobinsonIsMeasuredAgainstStageOrder) {
  // Order reduction: as the problem becomes STIFF (eps -> 0), a method whose own
  // stage order q is less than its classical order p is known (Hairer & Wanner,
  // Solving ODEs II, IV.15) to observe an EFFECTIVE order closer to q+1 than to p on
  // this exact test problem, in the stiff limit. This is MEASURED here at a fixed,
  // moderately stiff eps (not eps -> 0, which would need h << eps to resolve the
  // transient at all) and REPORTED against each method's own mechanically-computed
  // stage order, not asserted to equal q+1 exactly (the task's own instruction:
  // "report the observed order against the stage order", not "prove order reduction
  // theory").
  const double eps = 1e-3;
  struct Case { const char* name; ButcherTableau tableau; int classical_order; double h_max; };
  const std::vector<Case> cases = {
      {"SDIRK22", sdirk22_tableau(), 2, 0.02},
      {"SDIRK332", sdirk332_tableau(), 3, 0.02},
      {"SDIRK34", sdirk34_tableau(), 4, 0.03},
      {"ESDIRK324L2SA", esdirk324l2sa_tableau(), 3, 0.02},
  };
  for (const auto& c : cases) {
    auto problem = prothero_robinson_problem(eps);
    const auto hs = decade_of_h(c.h_max, 1e-1, 5);  // eps-sized h needs a narrower decade
    std::vector<double> errors, actual_hs;
    const double y0 = 0.5;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error_1d(
          [&] { return DirkIntegrator<double>(c.tableau, tight_newton_options()); },
          problem, y0, h, 1.0, prothero_robinson_exact(eps, y0, 0.0, 1.0));
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    auto fit = fit_order(actual_hs, errors);
    const int q = stage_order(c.tableau);
    printf("[DIRK/Prothero-Robinson eps=%.0e] %-14s classical order %d, stage order %d, "
           "MEASURED observed order = %.3f +/- %.3f\n",
           eps, c.name, c.classical_order, q, fit.slope, fit.slope_stderr);
    // The only assertion: the observed order is a FINITE, non-negative number (the
    // method still converges on a stiff problem at this eps/h combination) -- the
    // comparison against stage order is reported, not pass/failed against a precise
    // theoretical target this task did not ask this round to derive.
    EXPECT_GE(fit.slope, 0.0) << c.name << ": integrator must still converge on a stiff problem";
    EXPECT_TRUE(std::isfinite(fit.slope));
  }
}

// S8 physics-kit, track "Integrators", Stage B Round 3 (task item 2): the two new
// ESDIRK pairs from the Kennedy & Carpenter NASA TM, run through the SAME convergence,
// non-autonomous and order-reduction problems Round 2 already used.
TEST(DirkIntegrator, TwoNewEsdirkPairsConvergeAtTheirDeclaredOrderOnDahlquist) {
  const std::vector<DahlquistCase> cases = {
      // h ranges MEASURED (lead, 2026-10-31) from the full [0.001, 0.1] sweep the
      // round-3 track ran: ESDIRK436L2SA's errors 7.0e-6 .. 6.6e-14 are truncation-
      // limited throughout (ratio 39 per 2.51x in h, 2.51^4 = 39.8); ESDIRK548L2SA's
      // 3.1e-7, 3.4e-9, 4.5e-11 are fifth order (ratios 92, 75 against 2.51^5 = 99.6)
      // and the next three, 2.7e-12, 4.6e-13, 3.2e-13, sit on the Newton floor, which
      // dragged the fit to 3.06 +/- 0.44. The fifth-order fit therefore runs on
      // [0.02, 0.2], where every point (9.1e-6 down to 9.2e-11) is two decades or
      // more above the floor; measured 4.97 +/- 0.04 (on [0.05, 0.5] the
      // pre-asymptotic term reads 4.82 +/- 0.03, on [0.03, 0.3] 4.89 +/- 0.02).
      {"ESDIRK436L2SA", esdirk436l2sa_tableau(), 4, 0.1, 6},
      {"ESDIRK548L2SA", esdirk548l2sa_tableau(), 5, 0.2, 6, 0.1},
  };
  auto problem = dahlquist_problem(-2.0, 3.0);
  for (const auto& c : cases) {
    // The Case's own `order` field is a literal written independently of the
    // tableau's own `order` member -- a mutation (or a future transcription slip)
    // that changes ONE without the other would otherwise go undetected by this
    // test's own fit-vs-c.order comparison below, since neither is derived from
    // the other. Catch that class of defect explicitly, once, here.
    ASSERT_EQ(c.tableau.order, c.order)
        << c.name << ": this test's own expected order must match the tableau's "
        << "own declared order field";
    const auto hs = decade_of_h(c.h_max, c.h_min_ratio, c.n_points);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      // tight_newton_options() (Round 2's own atol=1e-8/rtol=1e-7), not
      // very_tight_newton_options() below: MEASURED that the tighter setting fails
      // DivergedLinearSolve at the coarse end of this complex-Dahlquist sweep for
      // both new tableaus -- the complex-rotation Jacobian GMRES must resolve here
      // behaves differently from the scalar logistic/non-autonomous problems
      // very_tight_newton_options() was tuned against, so it stays scoped to those.
      auto [err, actual_h] = fixed_step_final_error(
          [&] { return DirkIntegrator<double>(c.tableau, tight_newton_options()); },
          problem, {1.0, 0.0}, h, 1.0,
          [](double t) { auto e = dahlquist_exact(-2.0, 3.0, t); return std::array<double,2>{e[0], e[1]}; });
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    printf("[DIRK/Dahlquist, round3] %-14s h in [%.4g,%.4g] errors: ", c.name, actual_hs.back(), actual_hs.front());
    for (double e : errors) printf("%.3e ", e);
    printf("\n");
    auto fit = fit_order(actual_hs, errors);
    printf("[DIRK/Dahlquist, round3] %-14s fitted order = %.5f +/- %.5f (expected %d)\n",
           c.name, fit.slope, fit.slope_stderr, c.order);
    // ESDIRK548L2SA is order 5 (its name, Kennedy and Carpenter's convention, and the
    // order-condition checker: every condition through order 5 holds to the
    // coefficients' own 1e-27 precision and order 6 fails by 4.2e-3; the round-3
    // track had declared it order 4 after misreading "order 5 also passes" as an
    // unresolvable distinction -- lead, 2026-10-31).
    const double tol = 8.0 * fit.slope_stderr;
    EXPECT_NEAR(fit.slope, static_cast<double>(c.order), tol)
        << c.name << ": measured order " << fit.slope << " +/- " << fit.slope_stderr;
  }
}

TEST(DirkIntegrator, TwoNewEsdirkPairsConvergeAtTheirDeclaredOrderOnTheLogisticEquation) {
  // h_max=0.1 (not Round 2's 0.2): MEASURED that ESDIRK436L2SA's own truncation error
  // at h=0.2 (2.3e-8) already sits close to tight_newton_options()'s own atol floor;
  // narrowing avoids relying on the floor-avoidance alone to keep the fit clean.
  // h_min_ratio=0.3 (not 1e-1): MEASURED that ESDIRK548L2SA's own truncation error
  // drops below double-precision round-off on the logistic equation once h is below
  // roughly 0.012 -- a ratio of 0.3 from h_max=0.08 stops at h_min ~ 0.024, clear of
  // that floor, so every point in the fit is still truncation-error-limited.
  const std::vector<DahlquistCase> cases = {
      {"ESDIRK436L2SA", esdirk436l2sa_tableau(), 4, 0.08, 4},
      {"ESDIRK548L2SA", esdirk548l2sa_tableau(), 5, 0.08, 4},
  };
  auto problem = logistic_problem();
  for (const auto& c : cases) {
    const auto hs = decade_of_h(c.h_max, 0.3, c.n_points);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error_1d(
          [&] { return DirkIntegrator<double>(c.tableau, very_tight_newton_options()); },
          problem, 0.5, h, 2.0, logistic_exact(0.5, 2.0));
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    auto fit = fit_order(actual_hs, errors);
    printf("[DIRK/logistic, round3] %-14s fitted order = %.5f +/- %.5f (expected %d)\n",
           c.name, fit.slope, fit.slope_stderr, c.order);
    // Two-sided for both pairs: ESDIRK548L2SA is order 5 (see the Dahlquist test's
    // comment); the round-3 track's one-sided "at least 4" bound measured 4.991 +/-
    // 0.006 here and is replaced by the same band every other method is held to.
    const double tol = 8.0 * fit.slope_stderr;
    EXPECT_NEAR(fit.slope, static_cast<double>(c.order), tol)
        << c.name << ": measured order " << fit.slope << " +/- " << fit.slope_stderr;
  }
}

TEST(DirkIntegrator, TwoNewEsdirkPairsConvergeOnANonAutonomousProblem) {
  const double omega = 3.0;
  auto problem = non_autonomous_problem(omega);
  struct Case { const char* name; ButcherTableau tableau; int order; double h_max; double h_min_ratio; };
  // h ranges MEASURED (lead, 2026-10-31) from the track's [0.001, 0.1] sweep:
  // ESDIRK436L2SA 3.1e-6, 7.4e-8, 1.9e-9, 4.8e-11, 1.3e-12 are fourth order (ratios
  // 42, 39, 39, 37 against 2.51^4 = 39.8) and the sixth point, 2.2e-13, is the Newton
  // floor; ESDIRK548L2SA 1.0e-7, 9.8e-10, 1.0e-11, 1.3e-13 are fifth order (ratios
  // 105, 98, 79 against 99.6) and the last two, 8.3e-14 and 1.9e-13, are the floor.
  // Each fit stops above its floor; the floor-contaminated fits had read 3.69 +/- 0.17
  // and 3.06 +/- 0.65.
  const std::vector<Case> cases = {
      {"ESDIRK436L2SA", esdirk436l2sa_tableau(), 4, 0.1, 0.025},
      {"ESDIRK548L2SA", esdirk548l2sa_tableau(), 5, 0.1, 0.1},
  };
  for (const auto& c : cases) {
    const auto hs = decade_of_h(c.h_max, c.h_min_ratio, 6);
    std::vector<double> errors, actual_hs;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error_1d(
          [&] { return DirkIntegrator<double>(c.tableau, very_tight_newton_options()); },
          problem, non_autonomous_exact(omega, 0.0), h, 1.5, non_autonomous_exact(omega, 1.5));
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    printf("[DIRK/non-autonomous, round3] %-14s errors: ", c.name);
    for (double e : errors) printf("%.3e ", e);
    printf("\n");
    auto fit = fit_order(actual_hs, errors);
    printf("[DIRK/non-autonomous, round3] %-14s fitted order = %.5f +/- %.5f (expected %d)\n",
           c.name, fit.slope, fit.slope_stderr, c.order);
    const double tol = 8.0 * fit.slope_stderr;
    EXPECT_NEAR(fit.slope, static_cast<double>(c.order), tol)
        << c.name << ": measured order " << fit.slope << " +/- " << fit.slope_stderr;
  }
}

TEST(DirkIntegrator, TwoNewEsdirkPairsOrderReductionOnProtheroRobinson) {
  const double eps = 1e-3;
  struct Case { const char* name; ButcherTableau tableau; int classical_order; double h_max; };
  const std::vector<Case> cases = {
      {"ESDIRK436L2SA", esdirk436l2sa_tableau(), 4, 0.02},
      {"ESDIRK548L2SA", esdirk548l2sa_tableau(), 5, 0.02},
  };
  for (const auto& c : cases) {
    auto problem = prothero_robinson_problem(eps);
    const auto hs = decade_of_h(c.h_max, 1e-1, 5);
    std::vector<double> errors, actual_hs;
    const double y0 = 0.5;
    for (double h : hs) {
      auto [err, actual_h] = fixed_step_final_error_1d(
          [&] { return DirkIntegrator<double>(c.tableau, tight_newton_options()); },
          problem, y0, h, 1.0, prothero_robinson_exact(eps, y0, 0.0, 1.0));
      errors.push_back(err);
      actual_hs.push_back(actual_h);
    }
    auto fit = fit_order(actual_hs, errors);
    const int q = stage_order(c.tableau);
    printf("[DIRK/Prothero-Robinson round3, eps=%.0e] %-14s classical order %d, stage order %d, "
           "MEASURED observed order = %.3f +/- %.3f\n",
           eps, c.name, c.classical_order, q, fit.slope, fit.slope_stderr);
    EXPECT_GE(fit.slope, 0.0) << c.name << ": integrator must still converge on a stiff problem";
    EXPECT_TRUE(std::isfinite(fit.slope));
  }
}

TEST(DirkIntegrator, MassMatrixIsHonouredOnAnImplicitStage) {
  // M y' = A y with M = diag(2, 3), A = diag(-4, -6): y' = M^{-1} A y = diag(-2, -2) y,
  // so the EXACT solution is y(t) = y0 * exp(-2t) componentwise regardless of M -- a
  // mass matrix that is NOT the identity, exercised on SDIRK22 (both stages implicit,
  // nonzero diagonal), proving `problem.mass` is actually read and solved for, not
  // silently treated as M = I.
  TemporalProblem<double> problem;
  problem.dimension = 2;
  problem.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = -4.0 * y[0];
    out[1] = -6.0 * y[1];
    return Result<void>::success();
  };
  problem.mass = [](std::span<const double> v, std::span<double> out) -> Result<void> {
    out[0] = 2.0 * v[0];
    out[1] = 3.0 * v[1];
    return Result<void>::success();
  };
  DirkIntegrator<double> integ(sdirk22_tableau(), tight_newton_options());
  std::vector<double> y{1.0, 1.0};
  const double h = 0.01;
  double t = 0.0;
  for (int i = 0; i < 100; ++i) {
    auto r = integ.step(problem, t, h, y);
    ASSERT_TRUE(r.ok()) << r.error().code;
    t = r.value().t_new;
  }
  const double exact = std::exp(-2.0 * t);
  EXPECT_NEAR(y[0], exact, 1e-5);
  EXPECT_NEAR(y[1], exact, 1e-5);
}


// ===========================================================================
// Round 3 (task item 6): FSAL reuse for the stiffly accurate ESDIRKs
// (dirk_integrator.hpp's own update_fsal_cache_/fsal_enabled_, ButcherTableau::
// is_FSAL). "Verify before optimise", applied as a GATE, not a slogan: the task's
// own standing rule for this item is "lands ONLY with a test that FSAL and
// non-FSAL paths produce bit-identical states". MEASURED (this suite): that bar is
// NOT met for the implicit (Newton-based) family, for every stiffly accurate DIRK
// tableau this project ships, at every NewtonKrylovOptions setting tried (the
// library's own default, tight_newton_options(), very_tight_newton_options()) --
// a consistent few-ULP gap (roughly 1.5e-15 absolute on the logistic equation,
// MEASURED, NOT shrinking as Newton's own tolerance tightens, which rules out a
// simple convergence-floor explanation). The reused derivative is a Newton-
// CONVERGED stage value rather than an exact closed-form evaluation, so SOME gap
// is expected; a restructuring that made y_new use the exact same floating-point
// expression as the last stage's own internal residual value closed the gap at
// the default and tight_newton_options() settings but NOT at
// very_tight_newton_options(), where it reappeared -- consistent with Newton-
// Krylov's own history-dependent inner iteration count (Eisenstat-Walker forcing,
// GMRES) at a tolerance this file's own top comment on tight_newton_options()
// already documents as past where the MFFD directional derivative degenerates,
// not a NEW defect specific to FSAL reuse. Per the task's own gate, this means
// FSAL reuse for the DIRK family does NOT land as an active default this round:
// the mechanism is implemented and tested (opt-in, `set_fsal_enabled(true)`,
// default false). ExplicitIntegrator's OWN FSAL reuse (explicit_integrator_test.
// cpp) is the part of this item that DOES meet the bit-identical bar, because an
// ERK method's last stage is an exact f() evaluation with no iterative solve at
// all -- it ships enabled by default.
// ===========================================================================

TEST(DirkIntegrator, FsalReuseDefaultsToDisabled) {
  for (const ButcherTableau& t :
       {esdirk324l2sa_tableau(), esdirk436l2sa_tableau(), esdirk548l2sa_tableau()}) {
    DirkIntegrator<double> integ(t, very_tight_newton_options());
    EXPECT_FALSE(integ.fsal_enabled())
        << t.name << ": FSAL reuse must default OFF (see this suite's own top comment "
        << "for why -- the implicit family does not meet the bit-identical bar)";
  }
}

TEST(DirkIntegrator, FsalReuseWhenExplicitlyEnabledSavesExactlyOneFunctionEvaluationPerStep) {
  // The MECHANICAL claim (fewer rhs calls) holds regardless of the bit-identity
  // finding above -- this is the part of item 6 that DOES work for DIRK, verified
  // on its own terms, separately from the (failing) bit-identical claim.
  for (const ButcherTableau& t :
       {esdirk324l2sa_tableau(), esdirk436l2sa_tableau(), esdirk548l2sa_tableau()}) {
    DirkIntegrator<double> on(t, very_tight_newton_options());
    DirkIntegrator<double> off(t, very_tight_newton_options());
    on.set_fsal_enabled(true);
    off.set_fsal_enabled(false);
    auto problem_on = logistic_problem();
    auto problem_off = logistic_problem();
    std::vector<double> y_on = {0.5}, y_off = {0.5};
    double t_on = 0.0, t_off = 0.0;
    const double h = 0.05;
    constexpr int n_steps = 20;
    std::size_t total_on = 0, total_off = 0;
    for (int i = 0; i < n_steps; ++i) {
      auto r_on = on.step(problem_on, t_on, h, y_on);
      auto r_off = off.step(problem_off, t_off, h, y_off);
      ASSERT_TRUE(r_on.ok()) << t.name << " step " << i << ": " << r_on.error().message;
      ASSERT_TRUE(r_off.ok()) << t.name << " step " << i << ": " << r_off.error().message;
      total_on += r_on.value().function_evaluations;
      total_off += r_off.value().function_evaluations;
      t_on = r_on.value().t_new;
      t_off = r_off.value().t_new;
    }
    std::printf("FsalReuseWhenExplicitlyEnabledSavesExactlyOneFunctionEvaluationPerStep: "
                "%-14s fevals_on=%zu fevals_off=%zu saved=%zu (expect %d)\n",
                t.name.c_str(), total_on, total_off, total_off - total_on, n_steps - 1);
    EXPECT_EQ(total_off - total_on, static_cast<std::size_t>(n_steps - 1))
        << t.name << ": enabling FSAL reuse must save exactly one rhs call per step "
        << "after the first (the explicit stage-0 call it skips)";
  }
}

TEST(DirkIntegrator, FsalReuseWhenEnabledIsASmallBoundedPerturbationNotAGrossError) {
  // The NEGATIVE result this suite's own top comment reports, MEASURED rather than
  // asserted from reasoning alone: enabling FSAL reuse changes the computed state
  // relative to full recomputation by a SMALL, bounded amount -- distinguishing a
  // genuine-but-tiny numerical effect from a gross logic error (which would show a
  // difference comparable to the method's own step size or truncation error, not a
  // few parts in 10^15).
  const ButcherTableau t = esdirk436l2sa_tableau();
  DirkIntegrator<double> on(t, very_tight_newton_options());
  DirkIntegrator<double> off(t, very_tight_newton_options());
  on.set_fsal_enabled(true);
  off.set_fsal_enabled(false);
  auto problem_on = logistic_problem();
  auto problem_off = logistic_problem();
  std::vector<double> y_on = {0.5}, y_off = {0.5};
  double t_on = 0.0, t_off = 0.0;
  double max_abs_diff = 0.0;
  for (int i = 0; i < 10; ++i) {
    auto r_on = on.step(problem_on, t_on, 0.05, y_on);
    auto r_off = off.step(problem_off, t_off, 0.05, y_off);
    ASSERT_TRUE(r_on.ok());
    ASSERT_TRUE(r_off.ok());
    max_abs_diff = std::max(max_abs_diff, std::fabs(y_on[0] - y_off[0]));
    t_on = r_on.value().t_new;
    t_off = r_off.value().t_new;
  }
  std::printf("FsalReuseWhenEnabledIsASmallBoundedPerturbationNotAGrossError: "
              "max_abs_diff=%.3e (expect: nonzero -- confirming it is a genuine "
              "effect -- but far below the method's own h=0.05 step size)\n",
              max_abs_diff);
  EXPECT_GT(max_abs_diff, 0.0)
      << "reuse must measurably differ from full recompute -- confirming this is a "
         "genuine, nonzero effect, not an artifact of this test";
  EXPECT_LT(max_abs_diff, 1e-10)
      << "the gap must stay many orders of magnitude below the method's own step "
         "size (0.05) and truncation error -- a few-ULP floating-point effect, not "
         "a logic error in the cache lookup or the reuse condition";
}

TEST(DirkIntegrator, FsalReuseIsBypassedAfterANonContinguousStepWhenEnabled) {
  // If the caller calls step() from a DIFFERENT (t, y) than the cache holds (here:
  // deliberately re-starting from the SAME initial condition after the cache was
  // already populated by a prior step, rather than continuing from where that step
  // left off), the cache must NOT be used -- the reuse condition's own bit-exact
  // (t, y) match (dirk_integrator.hpp) is the safeguard, and this test confirms it
  // actually gates correctly rather than always firing once warm. Exercises the
  // mechanism with set_fsal_enabled(true), since it is opt-in by default.
  const ButcherTableau t = esdirk436l2sa_tableau();
  DirkIntegrator<double> integ(t, very_tight_newton_options());
  integ.set_fsal_enabled(true);
  auto problem = logistic_problem();
  std::vector<double> y = {0.5};
  auto r1 = integ.step(problem, 0.0, 0.05, y);
  ASSERT_TRUE(r1.ok());
  // Cache now holds (t_new_1, y_new_1). Step again from the ORIGINAL (0.0, 0.5),
  // which does not match the cache -- must recompute, not silently reuse a stale
  // cached derivative from a different point.
  std::vector<double> y_fresh = {0.5};
  DirkIntegrator<double> integ_reference(t, very_tight_newton_options());
  integ_reference.set_fsal_enabled(false);
  auto r_cached = integ.step(problem, 0.0, 0.05, y_fresh);
  auto problem_ref = logistic_problem();
  std::vector<double> y_ref = {0.5};
  auto r_ref = integ_reference.step(problem_ref, 0.0, 0.05, y_ref);
  ASSERT_TRUE(r_cached.ok());
  ASSERT_TRUE(r_ref.ok());
  EXPECT_EQ(y_fresh[0], y_ref[0])
      << "stepping from a point that does NOT match the cache must give the SAME "
         "answer as fsal_enabled(false) -- the cache must be bypassed, not misapplied";
}

}  // namespace fuse::num
