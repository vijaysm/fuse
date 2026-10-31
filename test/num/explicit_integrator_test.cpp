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

}  // namespace fuse::num
