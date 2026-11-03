// A tier-0 unit test of examples/dn_heat/
// agreement_bound.hpp's own `agreement_bound()` (the SHARP per-period-sum
// formula) and `legacy_agreement_bound()` (the earlier formula, kept only
// for the informational `[agreement-bound legacy]` print), with hand-built
// numbers -- proving the derivation, without needing a full PETSc run. The
// bound sums the per-period terminal residuals over the whole run, so it is
// not a single-period-only bound applied across a 20-period run, and it
// carries no ad-hoc 20x safety margin or stopping-criterion mismatch. No
// FUSE library
// dependency: the header under test is a pair of pure free functions,
// one over two vectors of doubles and one over five doubles.

#include <gtest/gtest.h>

#include "agreement_bound.hpp"

TEST(DnHeatAgreementBound, JacobianInverseInfNormIsExactlyFourThirds) {
  // dn_component.hpp's own kStripRatio minimum is 0.50; the header's own
  // derivation gives ||J^-1||_inf = 2/(1+0.50) = 4/3 exactly -- hand-
  // computed here, not re-deriving the same arithmetic the header itself
  // performs.
  EXPECT_DOUBLE_EQ(dn_heat::kJacobianInverseInfNorm, 4.0 / 3.0);
}

TEST(DnHeatAgreementBound, MatchesHandComputedValueForRepresentativeInputs) {
  // Three periods, each solver's own terminal residual:
  //   native: {1e-10, 2e-10, 3e-10} (sum 6e-10)
  //   petsc:  {4e-10}               (sum 4e-10)
  //   bound = (4/3) * (6e-10 + 4e-10) = (4/3) * 1e-9 = 1.333...e-9
  const double bound = dn_heat::agreement_bound({1e-10, 2e-10, 3e-10}, {4e-10});
  EXPECT_NEAR(bound, (4.0 / 3.0) * 1e-9, 1e-23);
}

TEST(DnHeatAgreementBound, EmptyVectorsGiveAZeroBound) {
  // An empty vector contributes nothing to the sum (a solver that ran
  // zero periods disagreed with nothing) -- the header's own comment on
  // `agreement_bound()` states this choice (zero, not a refusal)
  // explicitly, since an all-periods run always passes a non-empty
  // vector and this is the only input this function cannot get from a
  // real run, only from a test.
  EXPECT_DOUBLE_EQ(dn_heat::agreement_bound({}, {}), 0.0);
  EXPECT_DOUBLE_EQ(dn_heat::agreement_bound({}, {1e-10}),
                    (4.0 / 3.0) * 1e-10);
}

TEST(DnHeatAgreementBound, ZeroBothTerminalResidualsGivesAZeroBound) {
  // With every period's own terminal residual exactly zero the bound is
  // exactly zero -- the sharp bound carries NO separate inter-Krylov or
  // safety term, unlike the legacy formula (see LegacyAgreementBound tests
  // below).
  EXPECT_DOUBLE_EQ(dn_heat::agreement_bound({0.0, 0.0}, {0.0}), 0.0);
}

TEST(DnHeatAgreementBound, SumIsOrderIndependent) {
  // Summing is commutative: permuting a vector's own entries must not
  // change the bound (to full double precision here, since these
  // particular values associate without rounding either way).
  const double forward = dn_heat::agreement_bound({1e-10, 2e-10, 3e-10}, {4e-10, 5e-10});
  const double reversed = dn_heat::agreement_bound({3e-10, 1e-10, 2e-10}, {5e-10, 4e-10});
  EXPECT_DOUBLE_EQ(forward, reversed);
}

// build-evidence/mutations/agreement-bounds.toml's own
// `dn-heat-agreement-bound-sums-only-the-last-period` row replaces
// `agreement_bound()`'s own per-vector summation with "take only the
// last element of each vector". A bound that only
// ever looks at the final period says nothing about what periods
// 1..N-1 may have contributed. A pair of per-period vectors whose
// EARLIER entries are much larger than their LAST entries must report a
// bound reflecting the SUM, not just the last (smaller) entry.
TEST(DnHeatAgreementBound, TheBoundSumsEveryPeriodNotJustTheLast) {
  // native's own first two periods dwarf its last; a last-element-only
  // formula would report (4/3)*(1e-8 + 1e-8) = 2.6667e-8, far below the
  // correct sum-based bound.
  const double bound = dn_heat::agreement_bound({1.0, 1.0, 1e-8}, {1e-8});
  EXPECT_NEAR(bound, (4.0 / 3.0) * (2.0 + 2e-8), 1e-12);
  EXPECT_GT(bound, 1.0);  // a last-element-only mutant would be of order 1e-8
}

TEST(DnHeatAgreementBound, TheBoundGrowsWithEitherSolversTerminalResidual) {
  const double bound_small = dn_heat::agreement_bound({1.0}, {1.0});
  const double bound_large = dn_heat::agreement_bound({1.0e6}, {1.0e6});
  EXPECT_GT(bound_large, bound_small);
  // The difference is driven entirely by the terminal-residual sum --
  // hand-computed, not re-derived from the function under test.
  EXPECT_NEAR(bound_large - bound_small,
              (4.0 / 3.0) * 2.0 * (1.0e6 - 1.0), 1e-6);
}

TEST(DnHeatAgreementBound, AsymmetricTerminalResidualsSumLinearly) {
  // agreement_bound() is linear in the combined sum of its two vectors'
  // own entries -- a second, independent way of pinning the formula's
  // own additive structure, varying one solver's own vector while
  // holding the other fixed, and using vectors of DIFFERENT lengths
  // (the header's own comment states this is supported).
  EXPECT_NEAR(dn_heat::agreement_bound({2e-10}, {1e-10}), (4.0 / 3.0) * 3e-10, 1e-24);
  EXPECT_NEAR(dn_heat::agreement_bound({2e-10}, {1e-10, 4e-10}), (4.0 / 3.0) * 7e-10, 1e-24);
}

// LEGACY (the earlier formula) -- kept here only so its own arithmetic
// stays pinned for the informational `[agreement-bound legacy]` print;
// NEVER the asserted bound (see agreement_bound.hpp's own comments for
// why).
TEST(DnHeatLegacyAgreementBound, MatchesHandComputedValueForRepresentativeInputs) {
  // rtol = 1e-8 for both solvers, ||F_0|| = 100 (native) and 80 (petsc),
  // native solution norm = 50:
  //   rtol_term = (4/3) * (1e-8*100 + 1e-8*80) = (4/3) * 1.8e-6 = 2.4e-6
  //   gmres_term = 3.9e-8 * 50 = 1.95e-6
  //   bound = 2.4e-6 + 1.95e-6 = 4.35e-6
  const double bound = dn_heat::legacy_agreement_bound(1e-8, 100.0, 1e-8, 80.0, 50.0);
  EXPECT_NEAR(bound, 4.35e-6, 1e-18);
}

TEST(DnHeatLegacyAgreementBound, ZeroInitialResidualLeavesOnlyTheMeasuredGmresTerm) {
  const double bound = dn_heat::legacy_agreement_bound(1e-8, 0.0, 1e-8, 0.0, 50.0);
  EXPECT_DOUBLE_EQ(bound, 3.9e-8 * 50.0);
}
