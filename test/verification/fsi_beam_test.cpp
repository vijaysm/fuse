// Verifies `fsi::BeamSolver` (examples/fsi_channel/beam_solver.hpp) against
// two closed-form results for a clamped-clamped Euler-Bernoulli beam:
//   * the static deflection under a uniform and under a linear (affine)
//     distributed load, EI y'''' = q0 + q1 x, y(0)=y'(0)=y(L)=y'(L)=0,
//     derived by direct integration (docs/user/fsi-channel.md carries the
//     same derivation);
//   * the first natural frequency, from the classical clamped-clamped
//     characteristic root beta_1 L = 4.730040744862704.
// Both are checked for CONVERGENCE at the FD scheme's order, not just a
// loose tolerance (docs/dev/testing.md: an order gate is what actually
// distinguishes a correct discretisation from an accidentally-close wrong
// one).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "beam_solver.hpp"

namespace {

double observed_order(const std::vector<double>& h,
                      const std::vector<double>& err) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = static_cast<double>(h.size());
  for (std::size_t k = 0; k < h.size(); ++k) {
    const double x = std::log(h[k]), y = std::log(err[k]);
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

double max_error_against_affine(std::size_t n, double L, double EI,
                                double q0, double q1) {
  const fsi::BeamSolver beam(fsi::BeamParams{.n = n, .L = L, .EI = EI,
                                             .rho_s = 1.0, .h_s = 1.0});
  std::vector<double> q(n + 1);
  for (std::size_t i = 0; i <= n; ++i) q[i] = q0 + q1 * beam.node_x(i);
  const std::vector<double> w = beam.solve_static(q);
  double worst = 0.0;
  for (std::size_t i = 0; i <= n; ++i) {
    const double want =
        fsi::clamped_beam_affine_deflection(beam.node_x(i), L, EI, q0, q1);
    worst = std::max(worst, std::abs(w[i] - want));
  }
  return worst;
}

}  // namespace

TEST(FsiBeam, StaticDeflectionUnderUniformLoadConvergesAtSecondOrder) {
  constexpr double L = 1.7, EI = 0.9, q0 = 3.0;
  std::vector<double> hs, errs;
  for (const std::size_t n : {8u, 16u, 32u, 64u}) {
    errs.push_back(max_error_against_affine(n, L, EI, q0, 0.0));
    hs.push_back(L / static_cast<double>(n));
  }
  const double order = observed_order(hs, errs);
  EXPECT_GT(order, 1.8) << "observed order " << order << ", finest error "
                        << errs.back();
  EXPECT_LT(order, 2.3) << "suspiciously high order " << order;
  EXPECT_LT(errs.back(), 2.0e-4) << "finest-grid error too large: "
                                 << errs.back();
}

TEST(FsiBeam, StaticDeflectionUnderLinearLoadConvergesAtSecondOrder) {
  constexpr double L = 1.3, EI = 0.6, q0 = 1.5, q1 = -4.0;
  std::vector<double> hs, errs;
  for (const std::size_t n : {8u, 16u, 32u, 64u}) {
    errs.push_back(max_error_against_affine(n, L, EI, q0, q1));
    hs.push_back(L / static_cast<double>(n));
  }
  const double order = observed_order(hs, errs);
  EXPECT_GT(order, 1.8) << "observed order " << order << ", finest error "
                        << errs.back();
  EXPECT_LT(order, 2.3) << "suspiciously high order " << order;
}

TEST(FsiBeam, UniformLoadMaxDeflectionMatchesTheTextbookFormula) {
  // The well-known q0 L^4 / (384 EI) at midspan is the q1=0 special case of
  // clamped_beam_affine_deflection -- checked directly here as an
  // independent sanity anchor on the closed form itself.
  constexpr double L = 2.0, EI = 1.0, q0 = 5.0;
  const double want = q0 * std::pow(L, 4) / (384.0 * EI);
  const double got = fsi::clamped_beam_affine_deflection(L / 2.0, L, EI, q0, 0.0);
  EXPECT_NEAR(got, want, 1e-12 * want);
}

// `BeamSolver::step_midpoint_load` replaces
// `step()`'s "force at t_{n+1}" convention with a load already representing
// t_{n+1/2} (matching Crank-Nicolson's own trapezoidal terms on the fluid
// side). Substituting the midpoint value for the trapezoidal average of end-
// point values is an O(dt^2) change for smooth forcing, so it should NOT
// cost the scheme's temporal order -- checked directly here, on the BEAM
// ALONE (no fluid, no coupling), against a self-convergence construction
// identical in spirit to fsi_transient_test.cpp's own (no closed-form
// forced-transient solution is needed: e_k = |w(dt_k)-w(dt_k/2)| ~ C dt_k^p).
double run_midpoint_forced_midspan(std::size_t n, double L, double EI,
                                   double rho_s, double h_s, double Q0,
                                   double Omega, double dt,
                                   std::size_t n_steps) {
  fsi::BeamSolver beam(
      fsi::BeamParams{.n = n, .L = L, .EI = EI, .rho_s = rho_s, .h_s = h_s});
  std::vector<double> q_mid(n + 1);
  double t = 0.0;
  for (std::size_t s = 0; s < n_steps; ++s) {
    const double load = Q0 * std::sin(Omega * (t + 0.5 * dt));
    std::fill(q_mid.begin(), q_mid.end(), load);
    beam.step_midpoint_load(dt, q_mid);
    t += dt;
  }
  return beam.displacement()[beam.n_nodes() / 2];
}

TEST(FsiBeam, MidpointLoadStepConvergesAtSecondOrderInDt) {
  constexpr double L = 1.0, EI = 2.0, rho_s = 1.0, h_s = 0.5;
  constexpr std::size_t n = 24;  // fine enough that FD spatial error is
                                 // negligible next to the dt error measured
  const double omega1 = fsi::clamped_beam_first_frequency(L, EI, rho_s, h_s);
  const double T1 = 2.0 * M_PI / omega1;
  // Forcing well off resonance (0.37*omega1, an arbitrary irrational-ish
  // fraction), asymptotic regime (finest dt/T1 ~ 1/2560), fixed final time
  // a quarter of the natural period -- same reasoning as
  // fsi_transient_test.cpp's own choice.
  const double Omega = 0.37 * omega1;
  const double t_final = T1 / 4.0;

  std::vector<double> dts, w_at_dt;
  for (const std::size_t n_steps : {40u, 80u, 160u, 320u, 640u}) {
    const double dt = t_final / static_cast<double>(n_steps);
    const double w = run_midpoint_forced_midspan(n, L, EI, rho_s, h_s, 1.0,
                                                 Omega, dt, n_steps);
    dts.push_back(dt);
    w_at_dt.push_back(w);
  }
  std::vector<double> diffs;
  for (std::size_t k = 0; k + 1 < w_at_dt.size(); ++k) {
    diffs.push_back(std::abs(w_at_dt[k] - w_at_dt[k + 1]));
  }
  ASSERT_GE(diffs.size(), 2u);
  const double order =
      observed_order(std::vector<double>(dts.begin(), dts.end() - 1), diffs);
  std::printf("MidpointLoadStep self-convergence: order=%.4f, diffs=[%.3e, "
             "%.3e, %.3e, %.3e]\n",
             order, diffs[0], diffs[1], diffs[2], diffs[3]);
  EXPECT_GT(order, 1.8) << "observed order " << order
                        << " -- midpoint substitution cost temporal order";
  EXPECT_LT(order, 2.3) << "suspiciously high order " << order;
}

TEST(FsiBeam, FirstNaturalFrequencyConvergesAtSecondOrder) {
  constexpr double L = 1.0, EI = 2.0, rho_s = 3.0, h_s = 0.5;
  const double want = fsi::clamped_beam_first_frequency(L, EI, rho_s, h_s);
  ASSERT_GT(want, 0.0);
  std::vector<double> hs, errs;
  for (const std::size_t n : {8u, 16u, 32u, 64u}) {
    const fsi::BeamSolver beam(
        fsi::BeamParams{.n = n, .L = L, .EI = EI, .rho_s = rho_s, .h_s = h_s});
    const std::vector<double> freqs = beam.natural_frequencies(1);
    ASSERT_EQ(freqs.size(), 1u);
    errs.push_back(std::abs(freqs[0] - want));
    hs.push_back(L / static_cast<double>(n));
  }
  const double order = observed_order(hs, errs);
  EXPECT_GT(order, 1.7) << "observed order " << order << ", finest error "
                        << errs.back();
  EXPECT_LT(errs.back(), 1.5e-3 * want) << "finest-grid relative error too "
                                       "large: " << errs.back() / want;
}
