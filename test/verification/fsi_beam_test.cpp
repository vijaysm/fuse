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

#include <cmath>
#include <cstddef>
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
