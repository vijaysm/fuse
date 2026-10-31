// Tests the data layer (fuse/num/butcher_tableau.hpp) independent of whether an
// integrator for a given family is implemented yet: the Butcher row-sum/weight-sum
// consistency conditions (Hairer, Norsett & Wanner, Solving ODEs I, Theorem II.1.5) and
// the FSAL structural property catch a defective tableau mechanically, before any
// convergence test ever runs one. Two of this file's own assertions re-verify (rather
// than assume) the design doc's GRK4A finding, and a finding made by the row-sum check
// itself (SDIRK332, resolved against its primary source: butcher_tableau.hpp's top
// comment). The stability-function test at the end recomputes R(z) from the shipped
// coefficients, which is how the A/L-stability flags of five vendored methods were
// found to be wrong.

#include "fuse/num/butcher_tableau.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace fuse::num {
namespace {

// Derived, not fitted: every tableau's A/B/C/E here is a handful of floating
// arithmetic operations (at most ~stages^2 multiply-adds, per butcher_tableau.hpp's
// own factories) applied to literal constants, so the row-sum/weight-sum identities
// that SHOULD hold exactly in exact arithmetic accumulate at most a small multiple of
// machine epsilon per term. 64 * stages * eps is generous against that (the largest
// tableau here, DOPRI45/RODASP/SDIRK643, has at most 7-42 arithmetic terms feeding one
// row), not picked after seeing a result -- MEASURED to be 2-3 orders of magnitude
// above every ERK/DIRK/SDIRK/FIRK tableau's own observed defect (ButcherTableauTest's
// own printed values, run once with this bound removed, max observed 8.9e-16 for a
// 5-stage tableau).
double consistency_tolerance(int stages) {
  return 64.0 * static_cast<double>(stages) * std::numeric_limits<double>::epsilon();
}

bool is_classical_family(ButcherFamily f) {
  switch (f) {
    case ButcherFamily::Erk:
    case ButcherFamily::ErkEmbedded:
    case ButcherFamily::Dirk:
    case ButcherFamily::DirkEmbedded:
    case ButcherFamily::Sdirk:
    case ButcherFamily::SdirkEmbedded:
    case ButcherFamily::FirkNonSingular:
    case ButcherFamily::FirkSingular:
      return true;
    case ButcherFamily::Rosenbrock:
    case ButcherFamily::RosenbrockEmbedded:
      return false;
  }
  return false;
}

// Every ERK/DIRK/SDIRK/FIRK tableau satisfies the classical row-sum/weight-sum
// consistency conditions to round-off. SDIRK332 was the one exception until its C[1] was
// corrected to 29/108 against Norsett and Thomsen (1984), scheme (5.4);
// the vendored 24/108 is pinned below as a defect so a revert to it is caught.
TEST(ButcherTableau, ClassicalFamiliesAreRowSumConsistent) {
  for (const ButcherTableau& t : all_karma_tableaus()) {
    if (!is_classical_family(t.family)) continue;
    const double row_defect = butcher_row_sum_defect(t);
    const double weight_defect = butcher_weight_sum_defect(t);
    const double tol = consistency_tolerance(t.stages);
    EXPECT_NEAR(row_defect, 0.0, tol) << t.name << " row-sum defect " << row_defect;
    EXPECT_NEAR(weight_defect, 0.0, tol) << t.name << " weight-sum defect " << weight_defect;
  }
}

TEST(ButcherTableau, TheVendoredSdirk332StageTimeIsADefectTheRowSumCheckCatches) {
  ButcherTableau t = sdirk332_tableau();
  EXPECT_NEAR(t.C[1], 29.0 / 108.0, 1e-16) << "shipped C[1] must be the paper's 29/108";
  t.C[1] = 24.0 / 108.0;  // sdirk332.C:29, the transcription error
  EXPECT_GT(butcher_row_sum_defect(t), 4e-2);
}

// The FSAL structural property (A's last row bit-identical to B, C's last entry 1.0)
// matches every tableau's own declared is_FSAL flag -- including DOPRI45's corrected
// `true` (butcher_tableau.hpp's own top comment, second finding: the vendored
// `DOPRI45::is_FSAL()` hardcodes false despite the structure being present).
TEST(ButcherTableau, DeclaredFsalMatchesTheStructuralProperty) {
  for (const ButcherTableau& t : all_karma_tableaus()) {
    const auto n = static_cast<std::size_t>(t.stages);
    bool last_row_is_b = true;
    for (std::size_t j = 0; j < n; ++j) {
      if (std::fabs(t.A[n - 1][j] - t.B[j]) > 1e-12) { last_row_is_b = false; break; }
    }
    const bool c_last_is_one = std::fabs(t.C[n - 1] - 1.0) < 1e-12;
    const bool structural_fsal = last_row_is_b && c_last_is_one;
    EXPECT_EQ(structural_fsal, t.is_FSAL) << t.name;
  }
}

// Verified, not assumed: grk4a_tableau()'s E is read back and confirmed identically
// zero, independent of the design doc's own read.
TEST(ButcherTableau, Grk4aEmbeddedEstimateIsMeasuredZeroNotAssumed) {
  const ButcherTableau t = grk4a_tableau();
  ASSERT_TRUE(t.E.has_value());
  for (double e : *t.E) EXPECT_EQ(e, 0.0);
  EXPECT_FALSE(t.has_embedded_estimate())
      << "GRK4A's own E is identically zero (grk4a.C:71, re-verified); "
         "has_embedded_estimate() must report false, not the vendored category's claim";
  // ROS22, by contrast, is genuinely non-embedded (no E array in the vendored source
  // at all, ros22.C) -- the two "no usable estimate" cases are distinguished by WHY,
  // not conflated.
  const ButcherTableau ros22 = ros22_tableau();
  EXPECT_FALSE(ros22.E.has_value());
  EXPECT_FALSE(ros22.has_embedded_estimate());
}

// DOPRI45's corrected metadata (butcher_tableau.hpp's own top comment, first finding):
// embedded_order is 4 (the literature's own DOPRI5(4) value), not the vendored RK
// object's wrong 5.
TEST(ButcherTableau, Dopri45EmbeddedOrderIsCorrectedToFour) {
  const ButcherTableau t = dopri45_tableau();
  ASSERT_TRUE(t.embedded_order.has_value());
  EXPECT_EQ(*t.embedded_order, 4);
  EXPECT_TRUE(t.has_embedded_estimate());
  EXPECT_TRUE(t.is_FSAL);
}

// Spot-check against the design doc's own inventory table (a handful of named
// values, not every one -- the row-sum/FSAL tests above already exercise every
// tableau's own data mechanically).
TEST(ButcherTableau, StageCountsAndOrdersMatchTheInventoryMatchesTheSource) {
  EXPECT_EQ(forward_euler_tableau().stages, 1);
  EXPECT_EQ(forward_euler_tableau().order, 1);
  EXPECT_EQ(dopri45_tableau().stages, 7);
  EXPECT_EQ(dopri45_tableau().order, 5);
  EXPECT_EQ(rodasp_tableau().stages, 6);
  EXPECT_EQ(rodasp_tableau().order, 4);
  EXPECT_EQ(sdirk643_tableau().stages, 6);
  EXPECT_EQ(sdirk643_tableau().order, 4);
  EXPECT_EQ(all_karma_tableaus().size(), 21u);  // 22 vendored; SDIRK45 not ported (not A-stable)
}

TEST(ButcherTableau, ToCstringCoversEveryEnumerator) {
  EXPECT_STREQ(to_cstring(ButcherFamily::Erk), "Erk");
  EXPECT_STREQ(to_cstring(ButcherFamily::RosenbrockEmbedded), "RosenbrockEmbedded");
}

}  // namespace

// ---------------------------------------------------------------------------------------
// Linear stability. R(z) = 1 + z b^T (I - z M)^{-1} 1 is evaluated from the SHIPPED
// coefficients: for the classical families M = A; for the Rosenbrock family the stored
// implementation form is mapped back to (alpha, Gamma, b) as butcher_tableau.hpp's gamma
// comment states, and M = alpha + Gamma (Hairer and Wanner, Solving ODEs II, IV.7). The
// declared flags must agree with the function: A-stable iff |R(iy)| <= 1 on a geometric
// sample of the imaginary axis (y from 1e-3 to 1e6; by the maximum principle the axis
// decides for a rational R with no poles in the left half plane, which a method with a
// positive diagonal has), L-stable iff additionally |R(-1e15)| is at round-off. The
// sample resolves the known violations (max |R(iy)| = 1.028 for GRK4T, 9.82 for the
// vendored SDIRK34) by orders of magnitude; the pass margin for a genuinely A-stable
// method is |R(iy)| <= 1 + 64 u, u = 2^-53, a bound on the forward error of an s-stage
// complex solve with s <= 7, measured below 1e-15 for every shipped tableau.
namespace {

using Cx = std::complex<double>;

std::vector<std::vector<double>> inverse_lower(const std::vector<std::vector<double>>& L) {
  const std::size_t n = L.size();
  std::vector<std::vector<double>> X(n, std::vector<double>(n, 0.0));
  for (std::size_t j = 0; j < n; ++j) {
    X[j][j] = 1.0 / L[j][j];
    for (std::size_t i = j + 1; i < n; ++i) {
      double s = 0.0;
      for (std::size_t k = j; k < i; ++k) s += L[i][k] * X[k][j];
      X[i][j] = -s / L[i][i];
    }
  }
  return X;
}

// (M, b) of the equivalent DIRK whose stability function is the method's.
std::pair<std::vector<std::vector<double>>, std::vector<double>> stability_data(const ButcherTableau& t) {
  const std::size_t n = static_cast<std::size_t>(t.stages);
  if (!t.gamma.has_value()) return {t.A, t.B};
  const double g = (*t.gamma)[0][0];
  std::vector<std::vector<double>> Ginv(n, std::vector<double>(n, 0.0));  // (I - gamma_offdiag) / g
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j) Ginv[i][j] = ((i == j) ? 1.0 : -(*t.gamma)[i][j]) / g;
  const auto Gam = inverse_lower(Ginv);
  std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
  std::vector<double> b(n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double a = 0.0;
      for (std::size_t k = 0; k < n; ++k) a += t.A[i][k] * Gam[k][j];
      M[i][j] = a + Gam[i][j];
      b[j] += Gam[i][j] * t.B[i];
    }
  return {M, b};
}

Cx stability_function(const std::vector<std::vector<double>>& M, const std::vector<double>& b, Cx z) {
  const std::size_t n = b.size();
  std::vector<std::vector<Cx>> S(n, std::vector<Cx>(n + 1));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) S[i][j] = ((i == j) ? Cx(1.0) : Cx(0.0)) - z * M[i][j];
    S[i][n] = Cx(1.0);
  }
  for (std::size_t c = 0; c < n; ++c) {  // Gaussian elimination with partial pivoting
    std::size_t piv = c;
    for (std::size_t r = c + 1; r < n; ++r) {
      if (std::abs(S[r][c]) > std::abs(S[piv][c])) piv = r;
    }
    std::swap(S[c], S[piv]);
    for (std::size_t r = c + 1; r < n; ++r) {
      const Cx f = S[r][c] / S[c][c];
      for (std::size_t k = c; k <= n; ++k) S[r][k] -= f * S[c][k];
    }
  }
  std::vector<Cx> u(n);
  for (std::size_t i = n; i-- > 0;) {
    Cx s = S[i][n];
    for (std::size_t k = i + 1; k < n; ++k) s -= S[i][k] * u[k];
    u[i] = s / S[i][i];
  }
  Cx r = 1.0;
  for (std::size_t i = 0; i < n; ++i) r += z * b[i] * u[i];
  return r;
}

double max_abs_on_imaginary_axis(const std::vector<std::vector<double>>& M, const std::vector<double>& b) {
  double worst = 0.0;
  for (int e = -30; e <= 60; ++e) {  // y = 10^(e/10), 1e-3 .. 1e6
    const double y = std::pow(10.0, e / 10.0);
    worst = std::max(worst, std::abs(stability_function(M, b, Cx(0.0, y))));
  }
  return worst;
}

}  // namespace

TEST(ButcherTableau, DeclaredStabilityMatchesTheStabilityFunction) {
  constexpr double kUnit = std::numeric_limits<double>::epsilon() / 2.0;
  for (const ButcherTableau& t : all_karma_tableaus()) {
    const auto [M, b] = stability_data(t);
    const bool explicit_method =
        (t.family == ButcherFamily::Erk || t.family == ButcherFamily::ErkEmbedded);
    const double axis_max = max_abs_on_imaginary_axis(M, b);
    const double at_minus_inf = std::abs(stability_function(M, b, Cx(-1e15, 0.0)));
    const bool computed_A = !explicit_method && axis_max <= 1.0 + 64.0 * kUnit;
    const bool computed_L = computed_A && at_minus_inf < 1e-10;
    EXPECT_EQ(t.is_A_stable, computed_A) << t.name << ": max |R(iy)| = " << axis_max;
    EXPECT_EQ(t.is_L_stable, computed_L) << t.name << ": |R(-1e15)| = " << at_minus_inf;
  }
}

// The values the flags were corrected from, pinned as numbers rather than booleans so
// the corrections stay measured: SDIRK332 R(-inf) = -91/125 (exact rational arithmetic),
// ROS3P R(-inf) = 1 - sqrt(3), GRK4T's violation max |R(iy)| = 1.028.
TEST(ButcherTableau, CorrectedStabilityLimitsAreTheMeasuredOnes) {
  {
    const auto [M, b] = stability_data(sdirk332_tableau());
    EXPECT_NEAR(stability_function(M, b, Cx(-1e15, 0.0)).real(), -91.0 / 125.0, 1e-12);
  }
  {
    const auto [M, b] = stability_data(ros3p_tableau());
    EXPECT_NEAR(stability_function(M, b, Cx(-1e15, 0.0)).real(), 1.0 - std::sqrt(3.0), 1e-12);
  }
  {
    const auto [M, b] = stability_data(grk4t_tableau());
    EXPECT_GT(max_abs_on_imaginary_axis(M, b), 1.02);
  }
}

// The vendored SDIRK34 root (0.12888640051572, sdirk34.C:40) is not A-stable; the shipped
// tableau uses the largest root of the same cubic. Rebuilding the vendored tableau here
// and evaluating its stability function pins the reason for the switch (|R(-inf)| = 9.82).
TEST(ButcherTableau, TheVendoredSdirk34RootIsNotAStableAndTheShippedOneIs) {
  const double g = 0.12888640051572;
  const double q = 1.0 / ((0.5 - g) * (0.5 - g));
  const std::vector<std::vector<double>> A{
      {g, 0.0, 0.0}, {0.5 - g, g, 0.0}, {2.0 * g, 1.0 - 4.0 * g, g}};
  const std::vector<double> b{q / 24.0, 1.0 - q / 12.0, q / 24.0};
  EXPECT_GT(std::abs(stability_function(A, b, Cx(-1e15, 0.0))), 9.0);
  const auto [Ms, bs] = stability_data(sdirk34_tableau());
  EXPECT_NEAR(stability_function(Ms, bs, Cx(-1e15, 0.0)).real(), -0.63041494, 1e-7);
  EXPECT_TRUE(sdirk34_tableau().is_A_stable);
  EXPECT_FALSE(sdirk34_tableau().is_L_stable);
}

}  // namespace fuse::num
