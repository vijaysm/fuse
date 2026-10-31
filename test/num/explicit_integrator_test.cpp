// S8 physics-kit, track "Integrators", Stage B Round 1 (task item 5).
//
// Every tolerance below is DERIVED and printed next to the assertion that uses it
// (the project's own standing rule): the convergence-order bound comes from the
// least-squares fit's own standard error on the slope, not a number chosen after a
// first run came out close enough.

#include "fuse/num/explicit_integrator.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

namespace fuse::num {
namespace {

using fuse::Rational;

// ===========================================================================
// A complex Dahlquist equation y' = lambda*y, lambda = a + bi, carried as a REAL
// 2-vector (Re, Im): Re' = a*Re - b*Im, Im' = b*Re + a*Im. Standard practice for
// testing an RK method against a non-real eigenvalue without the integrator itself
// needing to know about std::complex (explicit_integrator.hpp's own error-norm scaling
// is real-valued, matching KARMA's own PETSc-real convention).
// ===========================================================================
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

// ===========================================================================
// Order measurement: fixed-step integration at a decade of h values, final error
// against the closed-form solution, log-log least-squares slope. The tolerance on the
// fitted slope is the regression's OWN standard error, scaled by a fixed multiplier
// (documented at the call site) -- derived from the fit, not chosen after seeing the
// result.
// ===========================================================================
struct OrderFit {
  double slope = 0.0;
  double slope_stderr = 0.0;
  int n_points = 0;
};

OrderFit fit_order(const std::vector<double>& h_values, const std::vector<double>& errors) {
  const int n = static_cast<int>(h_values.size());
  std::vector<double> lx(n), ly(n);
  for (int i = 0; i < n; ++i) { lx[i] = std::log(h_values[i]); ly[i] = std::log(errors[i]); }
  double mean_x = 0.0, mean_y = 0.0;
  for (int i = 0; i < n; ++i) { mean_x += lx[i]; mean_y += ly[i]; }
  mean_x /= n; mean_y /= n;
  double sxx = 0.0, sxy = 0.0;
  for (int i = 0; i < n; ++i) { sxx += (lx[i] - mean_x) * (lx[i] - mean_x); sxy += (lx[i] - mean_x) * (ly[i] - mean_y); }
  const double slope = sxy / sxx;
  const double intercept = mean_y - slope * mean_x;
  double ssr = 0.0;
  for (int i = 0; i < n; ++i) {
    const double resid = ly[i] - (slope * lx[i] + intercept);
    ssr += resid * resid;
  }
  const double dof = static_cast<double>(n - 2);
  const double mse = (dof > 0.0) ? ssr / dof : 0.0;
  const double slope_stderr = std::sqrt(mse / sxx);
  return {slope, slope_stderr, n};
}

// Runs `make_integrator()` (a fresh integrator per h, since ExplicitIntegrator carries
// no cross-call state for a fixed-step method) over N = round(t_end/h) fixed steps and
// returns the Euclidean final error against `exact(t_end)`.
template <class MakeIntegrator, class Exact>
double fixed_step_final_error(MakeIntegrator make_integrator, TemporalProblem<double>& problem,
                              std::vector<double> y0, double h, double t_end, Exact exact) {
  auto integrator = make_integrator();
  std::vector<double> y = y0;
  const auto n_steps = static_cast<long>(std::lround(t_end / h));
  double t = 0.0;
  for (long i = 0; i < n_steps; ++i) {
    auto r = integrator.step(problem, t, h, y);
    if (!r.ok()) { ADD_FAILURE() << "unexpected refusal: " << r.error().code; return -1.0; }
    t = r.value().t_new;
  }
  const auto ex = exact(t);
  double sum_sq = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) { const double d = y[i] - ex[i]; sum_sq += d * d; }
  return std::sqrt(sum_sq);
}

// A fixed multiplier on the fit's own standard error, not tuned per method: n=6 points
// (one decade, 5 intervals) over a method whose own leading error constant is smooth in
// h gives a standard error on the order of 1-3% of the true order for these four
// methods (MEASURED once with this assertion removed, printed to confirm before being
// relied on); 8 standard errors is therefore a wide two-sided margin, not a value
// chosen to make a borderline case pass.
constexpr double kOrderToleranceMultiplier = 8.0;

void expect_order(const OrderFit& fit, double expected_order, const char* method) {
  const double tol = kOrderToleranceMultiplier * fit.slope_stderr;
  EXPECT_NEAR(fit.slope, expected_order, tol)
      << method << ": fitted order " << fit.slope << " +/- " << fit.slope_stderr
      << " (tolerance " << tol << " = " << kOrderToleranceMultiplier << " stderr)";
}

std::vector<double> decade_of_h(double h_max, int n_points) {
  std::vector<double> hs(n_points);
  for (int i = 0; i < n_points; ++i) {
    hs[i] = h_max * std::pow(10.0, -static_cast<double>(i) / (n_points - 1));
  }
  return hs;
}

TEST(ExplicitIntegrator, ForwardEulerIsFirstOrderOnDahlquist) {
  auto problem = dahlquist_problem(-1.0, 5.0);
  // h_max derived to stay in the asymptotic regime (comment at kOrderToleranceMultiplier
  // explains the general bound; this value was chosen by requiring the fitted-order
  // bias stay under ~80% of that bound, MEASURED by sweeping h_max before relying on
  // the result -- a choice of WHICH h VALUES enter the fit, not a widened acceptance
  // band).
  const auto hs = decade_of_h(0.002, 6);
  std::vector<double> errors;
  for (double h : hs) {
    errors.push_back(fixed_step_final_error(
        [] { return ExplicitIntegrator<double>(forward_euler_tableau()); },
        problem, {1.0, 0.0}, h, 1.0,
        [](double t) { auto e = dahlquist_exact(-1.0, 5.0, t); return std::array<double,2>{e[0], e[1]}; }));
  }
  expect_order(fit_order(hs, errors), 1.0, "Forward Euler");
}

TEST(ExplicitIntegrator, Kutta22IsSecondOrderOnDahlquist) {
  auto problem = dahlquist_problem(-1.0, 5.0);
  const auto hs = decade_of_h(0.006, 6);
  std::vector<double> errors;
  for (double h : hs) {
    errors.push_back(fixed_step_final_error(
        [] { return ExplicitIntegrator<double>(kutta22_tableau()); },
        problem, {1.0, 0.0}, h, 1.0,
        [](double t) { auto e = dahlquist_exact(-1.0, 5.0, t); return std::array<double,2>{e[0], e[1]}; }));
  }
  expect_order(fit_order(hs, errors), 2.0, "Kutta22");
}

TEST(ExplicitIntegrator, Kutta44IsFourthOrderOnDahlquist) {
  auto problem = dahlquist_problem(-1.0, 5.0);
  const auto hs = decade_of_h(0.02, 6);
  std::vector<double> errors;
  for (double h : hs) {
    errors.push_back(fixed_step_final_error(
        [] { return ExplicitIntegrator<double>(kutta44_tableau()); },
        problem, {1.0, 0.0}, h, 1.0,
        [](double t) { auto e = dahlquist_exact(-1.0, 5.0, t); return std::array<double,2>{e[0], e[1]}; }));
  }
  expect_order(fit_order(hs, errors), 4.0, "Kutta44");
}

// DOPRI45 used FIXED-step here (no controller) to measure its own PRIMARY order (5),
// the same way the other three are measured -- its adaptive behaviour is a separate
// set of tests below.
TEST(ExplicitIntegrator, Dopri45IsFifthOrderOnDahlquistAtFixedStep) {
  auto problem = dahlquist_problem(-1.0, 5.0);
  const auto hs = decade_of_h(0.05, 6);
  std::vector<double> errors;
  for (double h : hs) {
    errors.push_back(fixed_step_final_error(
        [] { return ExplicitIntegrator<double>(dopri45_tableau()); },
        problem, {1.0, 0.0}, h, 1.0,
        [](double t) { auto e = dahlquist_exact(-1.0, 5.0, t); return std::array<double,2>{e[0], e[1]}; }));
  }
  expect_order(fit_order(hs, errors), 5.0, "DOPRI45 (fixed step)");
}

// The nonlinear check (task item 5): Dahlquist alone cannot distinguish a correctly
// adapted nonlinear stage loop from one that only happens to work because lambda is a
// scalar constant -- the logistic equation's own right-hand side depends on y itself.
TEST(ExplicitIntegrator, Kutta44IsFourthOrderOnTheLogisticEquation) {
  auto problem = logistic_problem();
  const auto hs = decade_of_h(0.05, 6);
  std::vector<double> errors;
  for (double h : hs) {
    errors.push_back(fixed_step_final_error(
        [] { return ExplicitIntegrator<double>(kutta44_tableau()); },
        problem, {0.5}, h, 2.0,
        [](double t) { return std::array<double,2>{logistic_exact(0.5, t), 0.0}; }));
  }
  // errors above compares a 1-vector padded with a 0; exact()[1] and y[1] both absent
  // from the 1-dimensional state, so fixed_step_final_error's own 2-term sum reduces
  // to the single real component -- fine for a magnitude comparison.
  expect_order(fit_order(hs, errors), 4.0, "Kutta44 (logistic)");
}

// ===========================================================================
// Mutation-sensitivity companion to the order tests above (build-evidence/mutations/
// integrators.toml's own "coefficient perturbation" row is proved against THIS test):
// perturbing one Butcher coefficient must make the fitted order collapse well outside
// the tolerance band `expect_order` uses, confirming the convergence test actually
// notices a wrong tableau rather than passing regardless.
// ===========================================================================
TEST(ExplicitIntegrator, APerturbedKutta44CoefficientFailsTheOrderFourBand) {
  ButcherTableau mutant = kutta44_tableau();
  mutant.B[0] += 1.0e-2;  // a coefficient perturbation far above round-off
  auto problem = dahlquist_problem(-1.0, 5.0);
  const auto hs = decade_of_h(0.1, 6);
  std::vector<double> errors;
  for (double h : hs) {
    errors.push_back(fixed_step_final_error(
        [&mutant] { return ExplicitIntegrator<double>(mutant); },
        problem, {1.0, 0.0}, h, 1.0,
        [](double t) { auto e = dahlquist_exact(-1.0, 5.0, t); return std::array<double,2>{e[0], e[1]}; }));
  }
  const OrderFit fit = fit_order(hs, errors);
  const double tol = kOrderToleranceMultiplier * fit.slope_stderr;
  EXPECT_GT(std::fabs(fit.slope - 4.0), tol)
      << "a mutated B[0] should collapse the observed order (a weight error is a "
         "first-order inconsistency, Hairer Thm II.1.5), got " << fit.slope
         << " +/- " << fit.slope_stderr;
}

// ===========================================================================
// DOPRI45 adaptive: reaches the window end EXACTLY (Rational equality, via
// SubStepClock -- the owner's own decision), with step rejections exercised under a
// tight tolerance.
// ===========================================================================
TEST(ExplicitIntegrator, Dopri45AdaptiveLandsExactlyOnTheWindowEndViaSubStepClock) {
  ControllerOptions opt;
  opt.kind = ControllerKind::PI;
  opt.order = 5; opt.embedded_order = 4; opt.safety = 0.9; opt.beta = 0.08;
  opt.fac_min = 0.2; opt.fac_max = 4.0; opt.h_min = 1e-10;
  ExplicitIntegrator<double> integrator(dopri45_tableau(), EmbeddedController(opt), 1e-8, 1e-10);

  auto problem = dahlquist_problem(-1.0, 5.0);
  std::vector<double> y{1.0, 0.0};

  SubStepClock clk(Rational(0), Rational(1));
  double h = 0.05;
  int accepted = 0, rejected = 0;
  while (!clk.at_window_end()) {
    const double t = clk.now().to_double();
    // round_to_quantum BEFORE stepping (not after): the integrator must advance the
    // ODE by the SAME step the clock will advance by, including the clip to the
    // window end -- stepping by the controller's raw (unclipped) proposal and only
    // clipping the CLOCK afterwards would silently integrate past the window while
    // the clock itself stopped exactly at it, a mismatch between what was computed
    // and what was recorded.
    const std::int64_t n = clk.round_to_quantum(h);
    const double h_clock = (clk.quantum() * Rational(n)).to_double();
    auto r = integrator.step(problem, t, h_clock, y);
    ASSERT_TRUE(r.ok()) << "unexpected refusal: " << r.error().code;
    if (r.value().accepted) {
      clk.advance(n);
      ++accepted;
    } else {
      ++rejected;
    }
    h = r.value().h_next;
  }
  EXPECT_EQ(clk.now(), Rational(1)) << "SubStepClock must land EXACTLY (Rational "
                                       "equality) on the window end";
  EXPECT_TRUE(clk.at_window_end());
  EXPECT_GT(accepted, 0);

  const auto exact = dahlquist_exact(-1.0, 5.0, 1.0);
  const double err = std::hypot(y[0] - exact[0], y[1] - exact[1]);
  EXPECT_LT(err, 1e-6) << "DOPRI45 at rtol=1e-8 should resolve the window to well under 1e-6";
}

// Step rejections, made to fire (the report contract's own standard): starting with an
// h far too large for the requested tolerance forces at least one rejection before the
// controller settles.
TEST(ExplicitIntegrator, Dopri45RejectsAnOversizedInitialStepUnderATightTolerance) {
  ControllerOptions opt;
  opt.kind = ControllerKind::PI;
  opt.order = 5; opt.embedded_order = 4; opt.safety = 0.9; opt.beta = 0.08;
  opt.fac_min = 0.2; opt.fac_max = 4.0; opt.h_min = 1e-12;
  ExplicitIntegrator<double> integrator(dopri45_tableau(), EmbeddedController(opt), 1e-12, 1e-14);

  auto problem = dahlquist_problem(-2.0, 20.0);  // fast oscillation: a large h is a bad step
  std::vector<double> y{1.0, 0.0};
  double t = 0.0, h = 2.0;  // deliberately oversized relative to the dynamics
  int rejected = 0;
  for (int i = 0; i < 50 && t < 1.0; ++i) {
    auto r = integrator.step(problem, t, h, y);
    ASSERT_TRUE(r.ok()) << r.error().code;
    if (r.value().accepted) { t = r.value().t_new; } else { ++rejected; }
    h = r.value().h_next;
  }
  EXPECT_GT(rejected, 0) << "an oversized initial step under a tight tolerance must be "
                            "rejected at least once";
}

// ===========================================================================
// Byte-identical agreement against an independent 40-line reference RK loop, on the
// SAME fixed step sequence (task item 5) -- the reference uses the SAME tableau
// (butcher_tableau.hpp), never a second, separately-typed copy of the coefficients, so
// this test checks ExplicitIntegrator::step()'s own ARITHMETIC, not whether two
// independently-transcribed coefficient sets happen to agree.
// ===========================================================================
std::vector<double> reference_rk_step(const ButcherTableau& t, std::vector<double> y, double time,
                                      double h, const std::function<void(double, const std::vector<double>&, std::vector<double>&)>& f) {
  const int s = t.stages;
  const auto n = y.size();
  std::vector<std::vector<double>> k(static_cast<std::size_t>(s), std::vector<double>(n));
  for (int i = 0; i < s; ++i) {
    std::vector<double> ys = y;
    for (int j = 0; j < i; ++j) {
      for (std::size_t m = 0; m < n; ++m) ys[m] += h * t.A[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * k[static_cast<std::size_t>(j)][m];
    }
    f(time + h * t.C[static_cast<std::size_t>(i)], ys, k[static_cast<std::size_t>(i)]);
  }
  std::vector<double> y_new = y;
  for (int i = 0; i < s; ++i) {
    for (std::size_t m = 0; m < n; ++m) y_new[m] += h * t.B[static_cast<std::size_t>(i)] * k[static_cast<std::size_t>(i)][m];
  }
  return y_new;
}

void expect_bit_identical_to_reference(const ButcherTableau& tableau, const char* name) {
  auto problem = dahlquist_problem(-1.0, 5.0);
  auto f = [](double, const std::vector<double>& y, std::vector<double>& out) {
    out[0] = -1.0 * y[0] - 5.0 * y[1];
    out[1] = 5.0 * y[0] + -1.0 * y[1];
  };
  ExplicitIntegrator<double> integrator(tableau);
  std::vector<double> y_integrator{1.0, 0.0};
  std::vector<double> y_reference{1.0, 0.0};
  double t = 0.0;
  const double h = 0.013;
  for (int step = 0; step < 37; ++step) {
    auto r = integrator.step(problem, t, h, y_integrator);
    ASSERT_TRUE(r.ok()) << name << ": " << r.error().code;
    t = r.value().t_new;
    y_reference = reference_rk_step(tableau, y_reference, static_cast<double>(step) * h, h, f);
    EXPECT_EQ(y_integrator[0], y_reference[0]) << name << " step " << step << " component 0";
    EXPECT_EQ(y_integrator[1], y_reference[1]) << name << " step " << step << " component 1";
  }
}

TEST(ExplicitIntegrator, ForwardEulerIsBitIdenticalToTheReferenceLoop) {
  expect_bit_identical_to_reference(forward_euler_tableau(), "FE");
}
TEST(ExplicitIntegrator, Kutta22IsBitIdenticalToTheReferenceLoop) {
  expect_bit_identical_to_reference(kutta22_tableau(), "Kutta22");
}
TEST(ExplicitIntegrator, Kutta44IsBitIdenticalToTheReferenceLoop) {
  expect_bit_identical_to_reference(kutta44_tableau(), "Kutta44");
}
TEST(ExplicitIntegrator, Dopri45FixedStepIsBitIdenticalToTheReferenceLoop) {
  expect_bit_identical_to_reference(dopri45_tableau(), "DOPRI45");
}

// ===========================================================================
// Refusals named in integrator.hpp/explicit_integrator.hpp, made to fire.
// ===========================================================================
TEST(ExplicitIntegrator, RefusesAProblemWithNoRhs) {
  ExplicitIntegrator<double> integrator(forward_euler_tableau());
  TemporalProblem<double> empty;
  std::vector<double> y{1.0};
  auto r = integrator.step(empty, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kNoRhsCode);
}

TEST(ExplicitIntegrator, RefusesAMassMatrixProblem) {
  ExplicitIntegrator<double> integrator(forward_euler_tableau());
  TemporalProblem<double> p = dahlquist_problem(-1.0, 0.0);
  p.mass = [](std::span<const double>, std::span<double>) { return Result<void>::success(); };
  std::vector<double> y{1.0, 0.0};
  auto r = integrator.step(p, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kMassUnsupportedCode);
}

TEST(ExplicitIntegrator, RefusesADimensionMismatch) {
  ExplicitIntegrator<double> integrator(forward_euler_tableau());
  TemporalProblem<double> p = dahlquist_problem(-1.0, 0.0);
  p.dimension = 2;
  std::vector<double> y{1.0};  // wrong size
  auto r = integrator.step(p, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kDimensionMismatchCode);
}

TEST(ExplicitIntegrator, RefusesAnEmbeddedControllerOverANonEmbeddedTableau) {
  ControllerOptions opt;
  opt.order = 2; opt.embedded_order = 1;
  // forward_euler_tableau() has no embedded estimate (Round 1 misuse case).
  ExplicitIntegrator<double> integrator(forward_euler_tableau(), EmbeddedController(opt), 1e-6, 1e-8);
  auto p = dahlquist_problem(-1.0, 0.0);
  std::vector<double> y{1.0, 0.0};
  auto r = integrator.step(p, 0.0, 0.1, y);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, kWrongFamilyCode);
}


}  // namespace
}  // namespace fuse::num

// S8 physics-kit, Stage B Round 2, task item 4 (SSP methods): SSPRK33/SSPRK104 run
// through ExplicitIntegrator unchanged (both are plain Erk-family tableaus); their
// SSP property is verified here by a monotonicity test, not re-proving the order
// (order_conditions_test.cpp's own RoundTwoTableaus suite already does that
// algebraically) and not the SSP coefficient itself (ssp_coefficient.hpp's own
// computation, cross-checked in butcher_tableau_test.cpp... actually verified via the
// factory functions' own use of `ssp_coefficient_from_shu_osher`, reproducing the
// published 1 and 6 -- see butcher_tableau.hpp's own ssprk33_tableau()/
// ssprk104_tableau() doc comments).
namespace fuse::num {
namespace {

// A periodic first-order upwind advection discretisation, y_i' = (y_{i-1} - y_i)/dx
// (Gottlieb, Shu & Tadmor's own standard SSP test problem, SIAM Review 43 (2001),
// Section 2's own illustration): forward Euler on this LINEAR system is a CONVEX
// COMBINATION of y_i and y_{i-1} whenever 0 <= h/dx <= 1 (h_FE = dx exactly), which is
// what makes it total-variation-diminishing (TVD) there -- the textbook h_FE this
// SSP coefficient's own "h <= C * h_FE" bound refers to.
TemporalProblem<double> periodic_upwind_advection(int n, double dx) {
  TemporalProblem<double> p;
  p.dimension = static_cast<std::size_t>(n);
  p.rhs = [n, dx](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    for (int i = 0; i < n; ++i) {
      const int im1 = (i == 0) ? n - 1 : i - 1;
      out[static_cast<std::size_t>(i)] = (y[static_cast<std::size_t>(im1)] - y[static_cast<std::size_t>(i)]) / dx;
    }
    return Result<void>::success();
  };
  return p;
}

double total_variation_periodic(const std::vector<double>& y) {
  double tv = 0.0;
  const std::size_t n = y.size();
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t im1 = (i == 0) ? n - 1 : i - 1;
    tv += std::fabs(y[i] - y[im1]);
  }
  return tv;
}

std::vector<double> step_function_initial_data(int n) {
  std::vector<double> y(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n / 2; ++i) y[static_cast<std::size_t>(i)] = 1.0;
  return y;
}

}  // namespace

// The task's own BOTH-DIRECTIONS instruction: h <= C * h_FE preserves the initial
// total variation (no new oscillation can appear from a TVD scheme); h > C * h_FE is
// NOT guaranteed to, and MEASURED here to actually violate it for a step comfortably
// past the method's own SSP coefficient -- both halves run for every SSP tableau this
// round adds, over several steps (one step alone can be deceptively forgiving).
TEST(ExplicitIntegrator, SspMethodsPreserveMonotonicityWithinTheirCoefficientAndViolateItBeyond) {
  struct Case { const char* name; ButcherTableau tableau; };
  const std::vector<Case> cases = {
      {"SSPRK33", ssprk33_tableau()},
      {"SSPRK104", ssprk104_tableau()},
  };
  constexpr int n = 20;
  constexpr double dx = 1.0;
  for (const auto& c : cases) {
    ASSERT_TRUE(c.tableau.ssp_coefficient.has_value());
    const double C = *c.tableau.ssp_coefficient;

    // Within the coefficient: h = 0.9 * C * h_FE, 10 steps.
    {
      auto problem = periodic_upwind_advection(n, dx);
      auto y0 = step_function_initial_data(n);
      const double tv0 = total_variation_periodic(y0);
      ExplicitIntegrator<double> integ(c.tableau);
      std::vector<double> y = y0;
      const double h = 0.9 * C * dx;
      double t = 0.0;
      double worst_tv = tv0;
      for (int step = 0; step < 10; ++step) {
        auto r = integ.step(problem, t, h, y);
        ASSERT_TRUE(r.ok()) << c.name << ": " << r.error().code;
        t = r.value().t_new;
        worst_tv = std::max(worst_tv, total_variation_periodic(y));
      }
      EXPECT_LE(worst_tv, tv0 + 1e-9)
          << c.name << ": h = 0.9*C*h_FE (C=" << C << ") must not increase total variation, "
          << "initial TV = " << tv0 << ", worst observed = " << worst_tv;
      std::printf("[SSP within] %-9s C=%.4g h=%.4g tv0=%.6f worst=%.6f\n", c.name, C, h, tv0, worst_tv);
    }

    // Beyond the coefficient: h = 1.5 * C * h_FE -- MEASURED to actually violate
    // monotonicity (not merely "not guaranteed to preserve it"), over enough steps
    // that a one-step fluke cannot hide the effect.
    {
      auto problem = periodic_upwind_advection(n, dx);
      auto y0 = step_function_initial_data(n);
      const double tv0 = total_variation_periodic(y0);
      ExplicitIntegrator<double> integ(c.tableau);
      std::vector<double> y = y0;
      const double h = 1.5 * C * dx;
      double t = 0.0;
      double worst_tv = tv0;
      for (int step = 0; step < 10; ++step) {
        auto r = integ.step(problem, t, h, y);
        ASSERT_TRUE(r.ok()) << c.name << ": " << r.error().code;
        t = r.value().t_new;
        worst_tv = std::max(worst_tv, total_variation_periodic(y));
      }
      EXPECT_GT(worst_tv, tv0 * 1.01)
          << c.name << ": h = 1.5*C*h_FE should measurably violate the TV bound "
          << "(initial TV = " << tv0 << ", worst observed = " << worst_tv << ") -- "
          << "if this fails, the chosen step/problem does not exercise the SSP limit "
          << "the way intended and needs a different multiplier, not a removed assertion";
      std::printf("[SSP beyond] %-9s C=%.4g h=%.4g tv0=%.6f worst=%.6f\n", c.name, C, h, tv0, worst_tv);
    }
  }
}

// SSPRK104's own Butcher form was derived by hand (Shu-Osher substitution,
// butcher_tableau.hpp's own ssprk104_tableau() doc comment), not copied from a
// published Butcher table -- proved here against the LITERAL recursion (Ketcheson
// 2008's own Pseudocode 3) run step by step on the same fixed step sequence.
// EXPECT_NEAR at 1e-13, not EXPECT_EQ: the two computations accumulate the SAME sums
// in a DIFFERENT order (the recursion's own chained intermediate "u" values vs the
// derived A/B/C matrix's row-by-row stage loop), so floating-point non-associativity
// can differ in the last bit or two even though both are mathematically identical --
// unlike round 1's own byte-identical reference checks, which ran the IDENTICAL
// stage-by-stage Butcher loop both times.
TEST(ExplicitIntegrator, Ssprk104MatchesTheShuOsherRecursion) {
  constexpr double h = 0.1;
  const int n_steps = 5;
  auto rhs = [](double, const std::vector<double>& y) {
    std::vector<double> out(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) out[i] = -2.0 * y[i] + 0.3 * y[i] * y[i];
    return out;
  };
  std::vector<double> y_recursion{1.0};
  const double h6 = h / 6.0;
  for (int step = 0; step < n_steps; ++step) {
    std::vector<double> u0 = y_recursion;
    std::vector<double> u = u0;
    std::vector<std::vector<double>> stages;  // u1..u5 (0-indexed 0..4)
    stages.push_back(u);  // u1 = u0
    for (int i = 0; i < 4; ++i) {
      auto g = rhs(0.0, stages.back());
      std::vector<double> next(u0.size());
      for (std::size_t m = 0; m < u0.size(); ++m) next[m] = stages.back()[m] + h6 * g[m];
      stages.push_back(next);
    }
    // stages[4] = u5
    auto g5 = rhs(0.0, stages[4]);
    std::vector<double> u6(u0.size());
    for (std::size_t m = 0; m < u0.size(); ++m) {
      u6[m] = 0.6 * u0[m] + 0.4 * (stages[4][m] + h6 * g5[m]);
    }
    stages.push_back(u6);  // stages[5] = u6
    for (int i = 0; i < 4; ++i) {  // u7, u8, u9, u10 -- FOUR stages, not three
      auto g = rhs(0.0, stages.back());
      std::vector<double> next(u0.size());
      for (std::size_t m = 0; m < u0.size(); ++m) next[m] = stages.back()[m] + h6 * g[m];
      stages.push_back(next);
    }
    // stages[9] = u10
    auto g10 = rhs(0.0, stages[9]);
    std::vector<double> u_plus(u0.size());
    for (std::size_t m = 0; m < u0.size(); ++m) {
      u_plus[m] = (1.0 / 25.0) * u0[m] + (9.0 / 25.0) * (stages[4][m] + h6 * g5[m]) +
                 (3.0 / 5.0) * (stages[9][m] + h6 * g10[m]);
    }
    y_recursion = u_plus;
  }

  TemporalProblem<double> problem;
  problem.dimension = 1;
  problem.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = -2.0 * y[0] + 0.3 * y[0] * y[0];
    return Result<void>::success();
  };
  ExplicitIntegrator<double> integ(ssprk104_tableau());
  std::vector<double> y_tableau{1.0};
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step) {
    auto r = integ.step(problem, t, h, y_tableau);
    ASSERT_TRUE(r.ok()) << r.error().code;
    t = r.value().t_new;
  }
  ASSERT_EQ(y_recursion.size(), y_tableau.size());
  EXPECT_NEAR(y_recursion[0], y_tableau[0], 1e-13)
      << "SSPRK104's Butcher-form step must match the literal Shu-Osher recursion to "
         "round-off, over " << n_steps << " steps on a nonlinear test problem";
}

// ===========================================================================
// Round 3 (task item 4): the SSPRK(10,4) embedded pair (Conde, Fekete and Shadid
// 2018, their own recommended ~b_3 row, butcher_tableau.hpp's own ssprk104_tableau()
// top comment) -- "the monotonicity test of round 2 extended to the pair": adaptive
// stepping (EmbeddedController active, not the fixed-step monotonicity test above)
// must still leave every ACCEPTED step's own total variation non-increasing, using
// the SAME periodic upwind advection problem, confirming the embedded pair's own
// error estimate does not silently let a non-monotone step through.
// ===========================================================================
TEST(ExplicitIntegrator, Ssprk104EmbeddedPairKeepsAdaptiveStepsMonotone) {
  const ButcherTableau t = ssprk104_tableau();
  ASSERT_TRUE(t.has_embedded_estimate())
      << "ssprk104_tableau() must carry the Conde/Fekete/Shadid embedded pair for this test";
  constexpr int n = 20;
  constexpr double dx = 1.0;
  auto problem = periodic_upwind_advection(n, dx);
  auto y0 = step_function_initial_data(n);
  const double tv0 = total_variation_periodic(y0);

  ControllerOptions opt;
  opt.kind = ControllerKind::I;
  opt.order = t.order;
  opt.embedded_order = *t.embedded_order;
  ExplicitIntegrator<double> integ(t, EmbeddedController(opt), 1e-4, 1e-6);
  std::vector<double> y = y0;
  double worst_tv = tv0;
  double t_now = 0.0;
  // Start comfortably INSIDE the SSP coefficient's own step bound (C=6) so the first
  // few steps are accepted and the controller has real error data to adapt from.
  double h = 0.9 * (*t.ssp_coefficient) * dx;
  int accepted = 0, rejected = 0;
  for (int iter = 0; iter < 30 && accepted < 10; ++iter) {
    auto r = integ.step(problem, t_now, h, y);
    ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
    if (r.value().accepted) {
      ++accepted;
      t_now = r.value().t_new;
      worst_tv = std::max(worst_tv, total_variation_periodic(y));
    } else {
      ++rejected;
    }
    h = r.value().h_next;
  }
  std::printf("Ssprk104EmbeddedPairKeepsAdaptiveStepsMonotone: accepted=%d rejected=%d "
              "tv0=%.6f worst=%.6f\n", accepted, rejected, tv0, worst_tv);
  EXPECT_GT(accepted, 0) << "the controller must accept at least one step for this to test anything";
  EXPECT_LE(worst_tv, tv0 + 1e-9)
      << "an accepted adaptive step, using the embedded pair's own error estimate, "
      << "must not increase total variation: tv0=" << tv0 << " worst=" << worst_tv;
}


// ===========================================================================
// Round 3 (task item 5): DynamicalTimeScaleController (docs/dev/integrators-design.md
// Section 2, mechanism 1; dynamical_time_scale_controller.hpp). A problem with a
// KNOWN, constant time scale (linear decay y' = -lambda*y, scale = 1/|lambda|) must
// have its chosen step PINNED to that scale -- an exact (not statistical) check,
// since the formula is a closed-form clamp, not a measured convergence rate.
// ===========================================================================

namespace {

TemporalProblem<double> linear_decay_problem(double lambda) {
  TemporalProblem<double> problem;
  problem.dimension = 1;
  problem.rhs = [lambda](double, std::span<const double> y,
                         std::span<double> out) -> Result<void> {
    out[0] = -lambda * y[0];
    return Result<void>::success();
  };
  problem.dynamical_time_scales = [lambda](double, std::span<const double>,
                                           std::vector<double>& out) -> Result<void> {
    out.clear();
    out.push_back(1.0 / lambda);
    return Result<void>::success();
  };
  return problem;
}

}  // namespace

TEST(ExplicitIntegrator, DynamicalControllerAlonePinsStepToTheProblemsOwnTimeScale) {
  // No EmbeddedController: a KARMA-mechanism-1-only path (Section 2 item 1's own
  // "it needs no embedded tableau at all (it runs even for plain FE/BE)"). FE has
  // no embedded estimate at all, which is exactly the point of this test.
  ButcherTableau fe;
  fe.name = "FE";
  fe.family = ButcherFamily::Erk;
  fe.order = 1;
  fe.A = {{0.0}};
  fe.B = {1.0};
  fe.C = {0.0};

  const double lambda = 4.0;
  const DynamicalControllerOptions opt{/*safety=*/0.8, /*fac_min=*/0.2, /*fac_max=*/4.0};
  ExplicitIntegrator<double> integ(fe);
  integ.set_dynamical_controller(DynamicalTimeScaleController(opt));

  auto problem = linear_decay_problem(lambda);
  std::vector<double> state = {1.0};
  const double h = 0.1;
  auto rec = integ.step(problem, 0.0, h, std::span<double>(state));
  ASSERT_TRUE(rec.ok()) << rec.error().code << ": " << rec.error().message;

  // h_cfl = safety / lambda = 0.8 * 0.25 = 0.2, within [fac_min*h, fac_max*h] =
  // [0.02, 0.4], so unclamped: the PINNED value this test exists to check.
  const double expected = opt.safety / lambda;
  std::printf("DynamicalControllerAlonePinsStepToTheProblemsOwnTimeScale: h_next=%.10f "
              "expected=%.10f\n", static_cast<double>(rec.value().h_next), expected);
  EXPECT_TRUE(rec.value().accepted)
      << "mechanism 1 has no accept/reject notion of its own: every step is accepted";
  EXPECT_NEAR(static_cast<double>(rec.value().h_next), expected, 1e-13)
      << "exact closed-form clamp result, not a measured rate: round-off tolerance only";
}

TEST(ExplicitIntegrator, DynamicalControllerUsesTheMostRestrictiveOfSeveralScales) {
  // A problem reporting TWO time scales (two decay modes, lambda1 != lambda2): the
  // controller must bound h by the SMALLER scale (the faster/more restrictive mode),
  // not the larger one -- a single-scale problem cannot distinguish a min-vs-max bug
  // in the selection itself (either comparator picks the only element), so this test
  // exists specifically to catch that class of defect.
  ButcherTableau fe;
  fe.name = "FE";
  fe.family = ButcherFamily::Erk;
  fe.order = 1;
  fe.A = {{0.0}};
  fe.B = {1.0};
  fe.C = {0.0};
  const double lambda_fast = 10.0, lambda_slow = 1.0;  // scales 0.1 and 1.0
  const DynamicalControllerOptions opt{0.5, 0.01, 100.0};  // wide band: result is h_cfl, unclamped
  ExplicitIntegrator<double> integ(fe);
  integ.set_dynamical_controller(DynamicalTimeScaleController(opt));
  TemporalProblem<double> problem;
  problem.dimension = 1;
  problem.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = -y[0];
    return Result<void>::success();
  };
  problem.dynamical_time_scales = [lambda_fast, lambda_slow](
      double, std::span<const double>, std::vector<double>& out) -> Result<void> {
    out.clear();
    out.push_back(1.0 / lambda_slow);  // 1.0, listed FIRST (not the answer)
    out.push_back(1.0 / lambda_fast);  // 0.1, the more restrictive scale
    return Result<void>::success();
  };
  std::vector<double> state = {1.0};
  auto rec = integ.step(problem, 0.0, 0.1, std::span<double>(state));
  ASSERT_TRUE(rec.ok()) << rec.error().code << ": " << rec.error().message;
  const double expected = opt.safety * (1.0 / lambda_fast);
  std::printf("DynamicalControllerUsesTheMostRestrictiveOfSeveralScales: h_next=%.10f "
              "expected=%.10f (bound by the SMALLER of the two scales)\n",
              static_cast<double>(rec.value().h_next), expected);
  EXPECT_NEAR(static_cast<double>(rec.value().h_next), expected, 1e-13);
}

TEST(ExplicitIntegrator, DynamicalControllerClampsToTheFacMinFacMaxBandAroundCurrentH) {
  // A time scale far SMALLER than fac_min*h would allow must be clamped UP to
  // fac_min*h, not followed past the band -- the same clamp EmbeddedController's
  // own proposal is held to (embedded_controller.hpp).
  ButcherTableau fe;
  fe.name = "FE";
  fe.family = ButcherFamily::Erk;
  fe.order = 1;
  fe.A = {{0.0}};
  fe.B = {1.0};
  fe.C = {0.0};
  const double lambda = 1.0e6;  // scale = 1e-6, far below fac_min*h = 0.02
  const DynamicalControllerOptions opt{0.8, 0.2, 4.0};
  ExplicitIntegrator<double> integ(fe);
  integ.set_dynamical_controller(DynamicalTimeScaleController(opt));
  auto problem = linear_decay_problem(lambda);
  std::vector<double> state = {1.0};
  const double h = 0.1;
  auto rec = integ.step(problem, 0.0, h, std::span<double>(state));
  ASSERT_TRUE(rec.ok()) << rec.error().code << ": " << rec.error().message;
  const double expected = opt.fac_min * h;
  std::printf("DynamicalControllerClampsToTheFacMinFacMaxBandAroundCurrentH: h_next=%.10f "
              "expected=%.10f (fac_min*h floor)\n",
              static_cast<double>(rec.value().h_next), expected);
  EXPECT_NEAR(static_cast<double>(rec.value().h_next), expected, 1e-13);
}

TEST(ExplicitIntegrator, DynamicalControllerRefusesWhenTheProblemHasNoCallback) {
  ButcherTableau fe;
  fe.name = "FE";
  fe.family = ButcherFamily::Erk;
  fe.order = 1;
  fe.A = {{0.0}};
  fe.B = {1.0};
  fe.C = {0.0};
  ExplicitIntegrator<double> integ(fe);
  integ.set_dynamical_controller(DynamicalTimeScaleController({}));
  TemporalProblem<double> problem;  // dynamical_time_scales left unset
  problem.dimension = 1;
  problem.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = -y[0];
    return Result<void>::success();
  };
  std::vector<double> state = {1.0};
  auto rec = integ.step(problem, 0.0, 0.1, std::span<double>(state));
  ASSERT_FALSE(rec.ok())
      << "a DynamicalTimeScaleController attached to an integrator whose problem "
         "supplies no dynamical_time_scales callback is a misconfiguration, not a "
         "silently-ignored option";
  EXPECT_EQ(rec.error().code, kNoDynamicalScalesCode);
}

TEST(ExplicitIntegrator, DynamicalControllerOwnRefusalPropagatesThroughStep) {
  // Every reported scale non-positive: DynamicalTimeScaleController's OWN refusal
  // (dynamical_time_scale_controller.hpp), not the integrator's misconfiguration one
  // -- the two refusal codes this round introduces must stay distinguishable.
  ButcherTableau fe;
  fe.name = "FE";
  fe.family = ButcherFamily::Erk;
  fe.order = 1;
  fe.A = {{0.0}};
  fe.B = {1.0};
  fe.C = {0.0};
  ExplicitIntegrator<double> integ(fe);
  integ.set_dynamical_controller(DynamicalTimeScaleController({}));
  TemporalProblem<double> problem;
  problem.dimension = 1;
  problem.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = -y[0];
    return Result<void>::success();
  };
  problem.dynamical_time_scales = [](double, std::span<const double>,
                                     std::vector<double>& out) -> Result<void> {
    out.clear();
    out.push_back(0.0);
    out.push_back(-1.0);
    return Result<void>::success();
  };
  std::vector<double> state = {1.0};
  auto rec = integ.step(problem, 0.0, 0.1, std::span<double>(state));
  ASSERT_FALSE(rec.ok());
  EXPECT_NE(rec.error().code, kNoDynamicalScalesCode)
      << "this must be the CONTROLLER's own refusal (every scale non-positive), "
         "distinct from the integrator's misconfiguration refusal above";
  std::printf("DynamicalControllerOwnRefusalPropagatesThroughStep: code=%s\n",
              rec.error().code.c_str());
}

TEST(ExplicitIntegrator, EmbeddedAndDynamicalControllersCombineByTakingTheMinimum) {
  // Round 3's own combination rule (open_questions): with both attached, the
  // accept/reject call stays the embedded controller's, but h_next is the MINIMUM
  // of the two proposals. Constructed so the two proposals are clearly different
  // (confirmed against each controller's own solo result) so the test cannot pass
  // by coincidence.
  const ButcherTableau t = dopri45_tableau();
  ControllerOptions eopt;
  eopt.kind = ControllerKind::PI;
  eopt.order = t.order;
  eopt.embedded_order = *t.embedded_order;
  const double lambda = 4.0;
  const double h0 = 0.01;

  // Solo embedded-controller proposal (tiny error on this easy problem -> grows h).
  {
    ExplicitIntegrator<double> solo(t, EmbeddedController(eopt), 1e-6, 1e-9);
    auto problem = linear_decay_problem(lambda);
    std::vector<double> state = {1.0};
    auto rec = solo.step(problem, 0.0, h0, std::span<double>(state));
    ASSERT_TRUE(rec.ok());
    ASSERT_GT(static_cast<double>(rec.value().h_next), h0)
        << "the embedded controller's own solo proposal must be GROWING h on this "
           "easy problem, or this test does not actually exercise the minimum";
  }

  const DynamicalControllerOptions dopt{/*safety=*/0.001, 0.2, 4.0};  // deliberately tiny
  ExplicitIntegrator<double> combined(t, EmbeddedController(eopt), 1e-6, 1e-9);
  combined.set_dynamical_controller(DynamicalTimeScaleController(dopt));
  auto problem = linear_decay_problem(lambda);
  std::vector<double> state = {1.0};
  auto rec = combined.step(problem, 0.0, h0, std::span<double>(state));
  ASSERT_TRUE(rec.ok()) << rec.error().code << ": " << rec.error().message;

  // Dynamical solo proposal, clamped: h_cfl = 0.001*0.25 = 0.00025, floor =
  // fac_min*h0 = 0.002, so the clamped dynamical proposal is 0.002.
  const double dynamical_floor = dopt.fac_min * h0;
  std::printf("EmbeddedAndDynamicalControllersCombineByTakingTheMinimum: "
              "combined_h_next=%.8f dynamical_floor=%.8f\n",
              static_cast<double>(rec.value().h_next), dynamical_floor);
  EXPECT_TRUE(rec.value().accepted)
      << "accept/reject stays the embedded controller's; this problem's error is tiny";
  EXPECT_NEAR(static_cast<double>(rec.value().h_next), dynamical_floor, 1e-13)
      << "the combined h_next must equal the smaller (dynamical) proposal, not the "
         "larger (embedded) one measured above";
}


// ===========================================================================
// Round 3 (task item 6): FSAL reuse for DOPRI45 (explicit_integrator.hpp's own
// fsal_cached_*/fsal_enabled_, ButcherTableau::is_FSAL). Unlike DirkIntegrator's
// own implicit-family attempt (dirk_integrator_test.cpp's own FSAL suite, where
// bit-identity was NOT achievable), an ERK method's last stage is an EXACT,
// non-iterative f() evaluation: DOPRI45's own B[6] == 0.0 and A[6] == B
// (butcher_tableau.hpp) mean the last stage's own y_stage accumulation and y_new's
// own b-weighted sum are the IDENTICAL floating-point expression, not merely an
// algebraically equivalent one -- so they round identically BY CONSTRUCTION, with
// no Newton solve anywhere to introduce a convergence-tolerance gap. This IS the
// part of item 6 that meets the task's own "lands ONLY with a test that FSAL and
// non-FSAL paths produce bit-identical states" bar, and ships enabled by default.
// ===========================================================================

namespace {

TemporalProblem<double> fsal_logistic_problem() {
  TemporalProblem<double> p;
  p.dimension = 1;
  p.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = y[0] * (1.0 - y[0]);
    return Result<void>::success();
  };
  return p;
}

}  // namespace

TEST(ExplicitIntegrator, Dopri45FsalReuseProducesBitIdenticalStatesFixedStep) {
  const ButcherTableau t = dopri45_tableau();
  ASSERT_TRUE(t.is_FSAL) << "DOPRI45 must be declared FSAL for this test to exercise anything";
  ExplicitIntegrator<double> fsal_on(t);
  ExplicitIntegrator<double> fsal_off(t);
  fsal_off.set_fsal_enabled(false);
  ASSERT_TRUE(fsal_on.fsal_enabled());
  ASSERT_FALSE(fsal_off.fsal_enabled());

  auto problem_on = fsal_logistic_problem();
  auto problem_off = fsal_logistic_problem();
  std::vector<double> y_on = {0.5}, y_off = {0.5};
  double t_on = 0.0, t_off = 0.0;
  const double h = 0.05;
  constexpr int n_steps = 60;
  std::size_t total_on = 0, total_off = 0;
  for (int i = 0; i < n_steps; ++i) {
    auto r_on = fsal_on.step(problem_on, t_on, h, y_on);
    auto r_off = fsal_off.step(problem_off, t_off, h, y_off);
    ASSERT_TRUE(r_on.ok()) << r_on.error().message;
    ASSERT_TRUE(r_off.ok()) << r_off.error().message;
    ASSERT_EQ(y_on[0], y_off[0])
        << "step " << i << ": FSAL reuse produced a DIFFERENT float64 value than "
        << "recomputing stage 0 from scratch -- these must be bit-identical";
    total_on += r_on.value().function_evaluations;
    total_off += r_off.value().function_evaluations;
    t_on = r_on.value().t_new;
    t_off = r_off.value().t_new;
  }
  std::printf("Dopri45FsalReuseProducesBitIdenticalStatesFixedStep: fevals_on=%zu "
              "fevals_off=%zu saved=%zu (expect %d)\n",
              total_on, total_off, total_off - total_on, n_steps - 1);
  EXPECT_EQ(total_off - total_on, static_cast<std::size_t>(n_steps - 1))
      << "FSAL reuse must save exactly one function evaluation per step after the first";
}

TEST(ExplicitIntegrator, Dopri45FsalReuseProducesBitIdenticalStatesUnderAdaptiveStepping) {
  const ButcherTableau t = dopri45_tableau();
  ControllerOptions opt;
  opt.kind = ControllerKind::PI;
  opt.order = t.order;
  opt.embedded_order = *t.embedded_order;
  ExplicitIntegrator<double> fsal_on(t, EmbeddedController(opt), 1e-6, 1e-9);
  ExplicitIntegrator<double> fsal_off(t, EmbeddedController(opt), 1e-6, 1e-9);
  fsal_off.set_fsal_enabled(false);

  auto problem_on = fsal_logistic_problem();
  auto problem_off = fsal_logistic_problem();
  std::vector<double> y_on = {0.5}, y_off = {0.5};
  double t_on = 0.0, t_off = 0.0, h_on = 0.05, h_off = 0.05;
  int accepted = 0, rejected = 0;
  std::size_t total_on = 0, total_off = 0;
  for (int i = 0; i < 200 && accepted < 40; ++i) {
    auto r_on = fsal_on.step(problem_on, t_on, h_on, y_on);
    auto r_off = fsal_off.step(problem_off, t_off, h_off, y_off);
    ASSERT_TRUE(r_on.ok()) << r_on.error().message;
    ASSERT_TRUE(r_off.ok()) << r_off.error().message;
    ASSERT_EQ(r_on.value().accepted, r_off.value().accepted)
        << "step " << i << ": enabling FSAL reuse must not change the accept/reject "
           "DECISION (only which rhs calls are skipped on an accepted step)";
    if (r_on.value().accepted) {
      ++accepted;
      ASSERT_EQ(y_on[0], y_off[0])
          << "accepted step " << i << ": FSAL reuse must still be bit-identical "
             "under adaptive stepping, including after a rejected-then-retried step";
    } else {
      ++rejected;
    }
    total_on += r_on.value().function_evaluations;
    total_off += r_off.value().function_evaluations;
    t_on = r_on.value().t_new;
    t_off = r_off.value().t_new;
    h_on = static_cast<double>(r_on.value().h_next);
    h_off = static_cast<double>(r_off.value().h_next);
  }
  std::printf("Dopri45FsalReuseProducesBitIdenticalStatesUnderAdaptiveStepping: "
              "accepted=%d rejected=%d fevals_on=%zu fevals_off=%zu saved=%zu\n",
              accepted, rejected, total_on, total_off, total_off - total_on);
  EXPECT_GT(accepted, 0) << "the controller must accept at least one step for this to test anything";
  // The cache updates ONLY on acceptance, but a REJECTED step retries from the
  // SAME (t, y) (state is left untouched, integrator.hpp's own contract) -- so a
  // retry still matches whatever the last ACCEPTED step cached, and gets the same
  // reuse benefit. Every `step()` CALL except the very first (which has no cache
  // yet) therefore saves one evaluation, regardless of accept/reject.
  const int total_calls = accepted + rejected;
  EXPECT_EQ(total_off - total_on, static_cast<std::size_t>(total_calls - 1))
      << "FSAL reuse must save exactly one rhs call on every step() CALL after the "
         "first (accepted=" << accepted << " rejected=" << rejected << ")";
}

TEST(ExplicitIntegrator, Dopri45FsalCacheGuardsAgainstADifferentlySizedState) {
  // If a caller reuses the SAME integrator instance across two UNRELATED problems
  // of different dimension (TemporalProblem::dimension left at its default 0 means
  // no mismatch refusal fires), the cache's own size guard
  // (fsal_cached_y_->size() == n, explicit_integrator.hpp) must stop a reuse
  // attempt BEFORE the element-by-element comparison loop would read past the
  // cached vector's own end -- this test exercises exactly that transition.
  const ButcherTableau t = dopri45_tableau();
  ExplicitIntegrator<double> integ(t);
  TemporalProblem<double> problem1;
  problem1.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = y[0] * (1.0 - y[0]);
    return Result<void>::success();
  };
  std::vector<double> y1 = {0.5};
  auto r1 = integ.step(problem1, 0.0, 0.05, y1);
  ASSERT_TRUE(r1.ok());
  const double t_after_1 = static_cast<double>(r1.value().t_new);

  // A SECOND, unrelated 2-dimensional problem, stepped from the SAME t the cache
  // holds (deliberately, to try to trigger a reuse attempt) but a state of a
  // DIFFERENT size than the cache's own cached vector.
  TemporalProblem<double> problem2;
  problem2.rhs = [](double, std::span<const double> y, std::span<double> out) -> Result<void> {
    out[0] = -y[0];
    out[1] = -2.0 * y[1];
    return Result<void>::success();
  };
  std::vector<double> y2 = {1.0, 1.0};
  auto r2 = integ.step(problem2, t_after_1, 0.05, y2);
  ASSERT_TRUE(r2.ok()) << r2.error().message;

  ExplicitIntegrator<double> reference(t);
  reference.set_fsal_enabled(false);
  TemporalProblem<double> problem2_ref;
  problem2_ref.rhs = problem2.rhs;
  std::vector<double> y2_ref = {1.0, 1.0};
  auto r2_ref = reference.step(problem2_ref, t_after_1, 0.05, y2_ref);
  ASSERT_TRUE(r2_ref.ok());
  EXPECT_EQ(y2[0], y2_ref[0])
      << "a 2-dimensional step must not be corrupted by a 1-dimensional cache";
  EXPECT_EQ(y2[1], y2_ref[1])
      << "a 2-dimensional step must not be corrupted by a 1-dimensional cache";

  // The direction that only the SIZE guard can catch (lead, 2026-10-31): the cache
  // now holds problem2's 2-vector state at t2; a 1-dimensional problem stepped from
  // that same t2 with a state EQUAL to the cache's first entry makes the
  // element-by-element loop (which runs over the NEW state's n = 1 entries) report
  // a match, so without the size guard the cached 2-vector derivative of problem2
  // (-y) would be reused as stage 0 of problem1 (y (1 - y)), and the result would
  // differ from a fresh integrator's. The mutation row
  // integrators-dopri45-fsal-cache-match-check-broken had been declared equivalent
  // because the 1 -> 2 transition above cannot see the guard; this 2 -> 1 transition can.
  const double t2 = static_cast<double>(r2.value().t_new);
  std::vector<double> y3 = {y2[0]};
  auto r3 = integ.step(problem1, t2, 0.05, y3);
  ASSERT_TRUE(r3.ok()) << r3.error().message;
  ExplicitIntegrator<double> reference3(t);
  reference3.set_fsal_enabled(false);
  std::vector<double> y3_ref = {y2[0]};
  auto r3_ref = reference3.step(problem1, t2, 0.05, y3_ref);
  ASSERT_TRUE(r3_ref.ok());
  EXPECT_EQ(y3[0], y3_ref[0])
      << "a 1-dimensional step must not reuse a 2-dimensional cache whose first entry matches";
}

TEST(ExplicitIntegrator, Dopri45FsalReuseIsBypassedAfterANonContinguousStep) {
  const ButcherTableau t = dopri45_tableau();
  ExplicitIntegrator<double> integ(t);
  auto problem = fsal_logistic_problem();
  std::vector<double> y = {0.5};
  auto r1 = integ.step(problem, 0.0, 0.05, y);
  ASSERT_TRUE(r1.ok());
  const double t_after_1 = static_cast<double>(r1.value().t_new);
  // Cache now holds (t_after_1, y after step 1). Step again at the SAME t
  // (t_after_1) but a DIFFERENT y (deliberately perturbed, not continuing from
  // where step 1 left off) -- the cache's own t-match alone must NOT be enough to
  // trigger reuse; a t-only check (this test's own reason for existing) would
  // incorrectly reuse a derivative evaluated at the WRONG y.
  std::vector<double> y_perturbed = {y[0] + 0.1};
  ExplicitIntegrator<double> integ_reference(t);
  integ_reference.set_fsal_enabled(false);
  auto r_cached = integ.step(problem, t_after_1, 0.05, y_perturbed);
  auto problem_ref = fsal_logistic_problem();
  std::vector<double> y_ref = {y[0] + 0.1};
  auto r_ref = integ_reference.step(problem_ref, t_after_1, 0.05, y_ref);
  ASSERT_TRUE(r_cached.ok());
  ASSERT_TRUE(r_ref.ok());
  EXPECT_EQ(y_perturbed[0], y_ref[0])
      << "stepping from a point whose t matches the cache but whose y does NOT "
         "must give the SAME answer as fsal_enabled(false) -- the cache's own y "
         "check, not just its t check, must gate reuse";
}

}  // namespace fuse::num
