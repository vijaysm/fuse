// AgreementBounds track: a tier-0 unit test of examples/brusselator/
// agreement_bound.hpp's own `agreement_bound()` (the SHARP, round-3
// formula) and `legacy_agreement_bound()` (round 2's own formula, kept
// only for the informational `[agreement-bound legacy]` print), with
// hand-built numbers -- proving the calibrated formula's own arithmetic
// (the error term's `(1 + residual_ratio)` scaling) without needing the
// native-only calibration (test/petsc/brusselator_newton_test.cpp's own
// `MeasuredErrorAgainstATightReference`, which this track verified
// separately) or a PETSc run this sandbox cannot execute (KNOWN_ISSUES.md
// D32's own "cannot complete a bare MPI_Init()" note). No FUSE library
// dependency: the header under test is a pair of pure free functions
// over doubles.

#include <gtest/gtest.h>

#include "agreement_bound.hpp"

TEST(BrusselatorAgreementBound, MatchesHandComputedValueForRepresentativeInputs) {
  // e_n=1e-6, f_final_native=1e-7, f_final_petsc=2e-7:
  //   residual_ratio = 2e-7/1e-7 = 2
  //   bound = 1e-6*(1+2) = 3e-6
  const double bound = brusselator_bound::agreement_bound(1e-6, 1e-7, 2e-7);
  EXPECT_NEAR(bound, 3e-6, 1e-18);
}

TEST(BrusselatorAgreementBound, EqualResidualsGiveExactlyTwiceEn) {
  // residual_ratio == 1 when the two solvers' own final residuals are
  // equal -> bound = e_n * 2, exactly, independent of the residuals' own
  // common magnitude.
  const double bound = brusselator_bound::agreement_bound(5e-6, 1e-5, 1e-5);
  EXPECT_DOUBLE_EQ(bound, 1e-5);
}

TEST(BrusselatorAgreementBound, ZeroErrorGivesAZeroBound) {
  // With e_n == 0 the bound is exactly zero (regardless of the residual
  // ratio) -- round 3 carries NO separate inter-Krylov or safety term,
  // unlike round 2's own formula (see LegacyAgreementBound tests below).
  EXPECT_DOUBLE_EQ(brusselator_bound::agreement_bound(0.0, 1.0, 1.0), 0.0);
}

// build-evidence/mutations/agreement-bounds.toml's own
// `brusselator-agreement-bound-drops-calibrated-error` row removes
// `agreement_bound()`'s dependence on `e_n` (replaces the bound with
// 0.0) -- this is the test that row names: a pair of calibrations whose
// own `e_n` differs by enough to change the bound by more than
// round-off must report a DIFFERENT bound, never the same one.
TEST(BrusselatorAgreementBound, TheBoundGrowsWithTheCalibratedError) {
  const double bound_small = brusselator_bound::agreement_bound(1e-10, 1.0, 1.0);
  const double bound_large = brusselator_bound::agreement_bound(1e-3, 1.0, 1.0);
  EXPECT_GT(bound_large, bound_small);
  // The difference is driven by e_n alone (residual_ratio == 1 for
  // both, so bound = e_n * 2) -- hand-computed, not re-derived from the
  // function under test.
  EXPECT_NEAR(bound_large - bound_small, 2.0 * (1e-3 - 1e-10), 1e-12);
}

TEST(BrusselatorAgreementBound, LargerPetscResidualGrowsTheBoundLinearly) {
  // Fixing e_n and f_final_native, the bound is exactly affine in
  // f_final_petsc (bound = e_n + e_n/f_final_native * f_final_petsc) --
  // a second, independent way of pinning the `(1 + residual_ratio)`
  // scaling, over the ratio term rather than over e_n itself.
  const double bound_ratio_1 = brusselator_bound::agreement_bound(2e-5, 1e-6, 1e-6);
  const double bound_ratio_10 = brusselator_bound::agreement_bound(2e-5, 1e-6, 1e-5);
  EXPECT_NEAR(bound_ratio_1, 4e-5, 1e-18);   // 2e-5 * (1+1)
  EXPECT_NEAR(bound_ratio_10, 2.2e-4, 1e-18);  // 2e-5 * (1+10)
}

// LEGACY (round 2's own formula) -- kept here only so its own arithmetic
// stays pinned for the informational `[agreement-bound legacy]` print;
// NEVER the asserted bound (see agreement_bound.hpp's own "ROUND 3"
// section for why).
TEST(BrusselatorLegacyAgreementBound, MatchesHandComputedValueForRepresentativeInputs) {
  // e_n=1e-6, f_final_native=1e-7, f_final_petsc=2e-7, native_norm=10:
  //   residual_ratio = 2e-7/1e-7 = 2
  //   error_term = 1e-6*(1+2) = 3e-6
  //   gmres_term = 3.9e-8*10 = 3.9e-7
  //   bound = 3e-6 + 3.9e-7 = 3.39e-6
  const double bound = brusselator_bound::legacy_agreement_bound(1e-6, 1e-7, 2e-7, 10.0);
  EXPECT_NEAR(bound, 3.39e-6, 1e-18);
}

TEST(BrusselatorLegacyAgreementBound, ZeroErrorLeavesOnlyTheMeasuredGmresTerm) {
  const double bound = brusselator_bound::legacy_agreement_bound(0.0, 1.0, 1.0, 50.0);
  EXPECT_DOUBLE_EQ(bound, 3.9e-8 * 50.0);
}
