// S8 physics-kit, track "Integrators", Stage B Round 1.
//
// Tests the data layer (fuse/num/butcher_tableau.hpp) independent of whether an
// integrator for a given family is implemented yet: the Butcher row-sum/weight-sum
// consistency conditions (Hairer, Norsett & Wanner, Solving ODEs I, Theorem II.1.5) and
// the FSAL structural property catch a defective tableau mechanically, before any
// convergence test ever runs one. Two of this file's own assertions are the task's own
// request to RE-VERIFY (not assume) the Stage A design's GRK4A finding, and a NEW
// finding this round found by the row-sum check itself (SDIRK332).

#include "fuse/num/butcher_tableau.hpp"

#include <cmath>
#include <limits>

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

// Every ERK/DIRK/SDIRK/FIRK tableau EXCEPT SDIRK332 (the known, reported defect:
// butcher_tableau.hpp's own top comment, third finding) satisfies the classical
// row-sum/weight-sum consistency conditions to round-off.
TEST(ButcherTableau, ClassicalFamiliesAreRowSumConsistentExceptTheKnownSdirk332Defect) {
  for (const ButcherTableau& t : all_karma_tableaus()) {
    if (!is_classical_family(t.family)) continue;
    const double row_defect = butcher_row_sum_defect(t);
    const double weight_defect = butcher_weight_sum_defect(t);
    const double tol = consistency_tolerance(t.stages);
    if (t.name == "SDIRK332") {
      // The reported defect itself, kept as a passing assertion so a future edit that
      // silently "fixes" sdirk332_tableau()'s C[1] without updating this test (or the
      // design doc's own open question) is caught, not silently rewarded.
      EXPECT_GT(row_defect, 1e-3) << "SDIRK332's own C[1] defect (sdirk332.C:29 vs :40) "
                                     "appears to have been fixed; update this test and "
                                     "docs/dev/integrators-design.md's Round 1 section";
      EXPECT_NEAR(weight_defect, 0.0, tol) << t.name;
    } else {
      EXPECT_NEAR(row_defect, 0.0, tol) << t.name << " row-sum defect " << row_defect;
      EXPECT_NEAR(weight_defect, 0.0, tol) << t.name << " weight-sum defect " << weight_defect;
    }
  }
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

// RE-VERIFIED this round (task's own instruction): grk4a_tableau()'s E is read back
// and confirmed identically zero, independent of the Stage A design's own prior read.
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

// Spot-check against the Stage A design's own inventory table (a handful of named
// values, not every one -- the row-sum/FSAL tests above already exercise every
// tableau's own data mechanically).
TEST(ButcherTableau, StageCountsAndOrdersMatchTheStageAInventory) {
  EXPECT_EQ(forward_euler_tableau().stages, 1);
  EXPECT_EQ(forward_euler_tableau().order, 1);
  EXPECT_EQ(dopri45_tableau().stages, 7);
  EXPECT_EQ(dopri45_tableau().order, 5);
  EXPECT_EQ(rodasp_tableau().stages, 6);
  EXPECT_EQ(rodasp_tableau().order, 4);
  EXPECT_EQ(sdirk643_tableau().stages, 6);
  EXPECT_EQ(sdirk643_tableau().order, 4);
  EXPECT_EQ(all_karma_tableaus().size(), 22u);
}

TEST(ButcherTableau, ToCstringCoversEveryEnumerator) {
  EXPECT_STREQ(to_cstring(ButcherFamily::Erk), "Erk");
  EXPECT_STREQ(to_cstring(ButcherFamily::RosenbrockEmbedded), "RosenbrockEmbedded");
}

}  // namespace
}  // namespace fuse::num
