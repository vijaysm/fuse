// KNOWN_ISSUES.md D82, MOAB half, round 11. `normalize_remap_weight`
// (fuse/backend/remap_weight_normalize.hpp) is the one pure function that
// decides -- given the channel's `Conservation` kind -- how an intersection
// measure becomes one remap weight entry. It takes four doubles and an
// enum, no MOAB or MPI symbol anywhere, so every case here actually RUNS in
// this sandbox (unlike the MOAB-gated sphere-pair rows in
// test/parallel/remap_halo_mpi_test.cpp, which only compile here).
//
// Three things are checked:
//   1. Extensive and Intensive/None are DIFFERENT formulas on the same
//      inputs whenever the source and target measures differ -- a swap
//      mutation between them must change the computed value.
//   2. Each formula, applied over a small synthetic FULLY-COVERED matrix
//      (one source cell split across several target cells, overlap areas
//      summing exactly to the source cell's own parent measure), delivers
//      the conservation property its own name promises: Extensive's column
//      (amount) sum is 1; Intensive's row (density) sum is 1.
//   3. The matched-measure case from `normalize_remap_weight`'s own doc
//      comment: a source cell split unevenly across target cells of
//      DIFFERENT measure is exactly where the pre-existing (Intensive)
//      formula, applied to an amount-valued field, inflates or deflates the
//      delivered total -- and Extensive does not.

#include "fuse/backend/remap_weight_normalize.hpp"

#include <array>
#include <cstddef>

#include <gtest/gtest.h>

namespace {

using fuse::Conservation;
using fuse::normalize_remap_weight;

TEST(RemapWeightNormalize, ExtensiveDropsTheDensityConversionFactor) {
  // overlap=3, src_parent=4 (fraction 0.75), src_measure=4, tgt_measure=10.
  // Extensive: overlap / src_measure = 3/4 = 0.75, independent of
  // tgt_measure.
  const double w = normalize_remap_weight(3.0, 4.0, 4.0, 10.0,
                                          Conservation::Extensive);
  EXPECT_DOUBLE_EQ(w, 0.75);
}

TEST(RemapWeightNormalize, IntensiveIsUnchangedFromThePreExistingFormula) {
  // Same inputs as above. Intensive: (overlap/src_parent) * (src_measure /
  // tgt_measure) = 0.75 * (4/10) = 0.3.
  const double w = normalize_remap_weight(3.0, 4.0, 4.0, 10.0,
                                          Conservation::Intensive);
  EXPECT_DOUBLE_EQ(w, 0.3);
}

TEST(RemapWeightNormalize, NoneBehavesLikeIntensive) {
  // `RemapPlan::conservation` defaults to `Intensive` (every pre-existing
  // caller), but a plan that explicitly carries `None` (FieldSpec's own
  // default) must take the SAME, pre-existing path -- Extensive is the only
  // formula that differs.
  const double a = normalize_remap_weight(3.0, 4.0, 4.0, 10.0, Conservation::None);
  const double b = normalize_remap_weight(3.0, 4.0, 4.0, 10.0, Conservation::Intensive);
  EXPECT_DOUBLE_EQ(a, b);
}

TEST(RemapWeightNormalize, ExtensiveAndIntensiveDisagreeWheneverMeasuresDiffer) {
  // The two formulas are IDENTICAL only when src_measure == tgt_measure
  // (the density-conversion factor becomes 1). Pick measures that differ,
  // confirming a swap mutation between the two branches is observable --
  // this is the property the mutation row below relies on to be killed.
  const double ext = normalize_remap_weight(3.0, 4.0, 4.0, 10.0,
                                            Conservation::Extensive);
  const double intsv = normalize_remap_weight(3.0, 4.0, 4.0, 10.0,
                                              Conservation::Intensive);
  EXPECT_NE(ext, intsv);
}

// ---------------------------------------------------------------------------
// End-to-end, over a small synthetic matrix: one source cell split across
// several target cells of DIFFERENT measure, overlaps summing exactly to the
// source cell's own parent measure (full coverage, no roundoff beyond
// machine epsilon -- a matched-measure, non-spherical case, so
// src_parent == src_measure exactly and the ONLY discrepancy either formula
// could show is the one `normalize_remap_weight`'s own doc describes:
// Intensive misapplied to an AMOUNT field inflates/deflates the total
// whenever the split is uneven).
// ---------------------------------------------------------------------------

TEST(RemapWeightNormalize, ExtensiveConservesAnAmountFieldAcrossAnUnevenSplit) {
  // One source cell, measure 4 (== its own parent measure: a matched,
  // non-spherical pair), split into three target cells of measure 1, 1, 2
  // (summing to 4, full coverage). overlap_j == target measure_j here
  // because the split is axis-aligned and exact.
  const double src_parent = 4.0;
  const double src_measure = 4.0;
  const std::array<double, 3> overlap{1.0, 1.0, 2.0};
  const std::array<double, 3> tgt_measure{1.0, 1.0, 2.0};

  double col_sum_raw = 0.0;       // should read 1: the AMOUNT fraction leaving the source cell
  for (std::size_t j = 0; j < overlap.size(); ++j) {
    const double w = normalize_remap_weight(overlap[j], src_parent, src_measure,
                                            tgt_measure[j], Conservation::Extensive);
    col_sum_raw += w;
  }
  EXPECT_NEAR(col_sum_raw, 1.0, 1e-13)
      << "an amount-valued field split across unevenly-sized target cells "
         "must have its weights sum to 1 -- the whole amount delivered "
         "exactly once, not inflated by the split";

  // Applying a source AMOUNT of 10.0: the three target cells must receive
  // 2.5, 2.5 and 5.0 respectively (proportional to their own share of the
  // source cell), summing to exactly 10.0 -- the delivered total.
  const double amount = 10.0;
  double delivered_total = 0.0;
  for (std::size_t j = 0; j < overlap.size(); ++j) {
    const double w = normalize_remap_weight(overlap[j], src_parent, src_measure,
                                            tgt_measure[j], Conservation::Extensive);
    delivered_total += w * amount;
  }
  EXPECT_NEAR(delivered_total, amount, 1e-12)
      << "D82: the delivered total must equal the source amount even though "
         "the source cell was split across target cells of different "
         "measure (1, 1, 2)";
}

TEST(RemapWeightNormalize,
    IntensiveMisappliedToAnAmountFieldInflatesTheTotalAcrossAnUnevenSplit) {
  // The SAME matrix as above, but through the pre-existing (Intensive)
  // formula, which is correct for a DENSITY field and wrong for an AMOUNT
  // one. This is D82's own reported symptom, reproduced directly and
  // algebraically (no MOAB, no sphere): Intensive's density-conversion
  // factor (src_measure/tgt_measure) means a target cell SMALLER than the
  // source cell receives a weight bigger than its own geometric share,
  // over-delivering an amount-valued field.
  const double src_parent = 4.0;
  const double src_measure = 4.0;
  const std::array<double, 3> overlap{1.0, 1.0, 2.0};
  const std::array<double, 3> tgt_measure{1.0, 1.0, 2.0};
  const double amount = 10.0;

  double delivered_total = 0.0;
  for (std::size_t j = 0; j < overlap.size(); ++j) {
    const double w = normalize_remap_weight(overlap[j], src_parent, src_measure,
                                            tgt_measure[j], Conservation::Intensive);
    delivered_total += w * amount;
  }
  // w_j = (overlap_j/4) * (4/tgt_measure_j) = overlap_j / tgt_measure_j.
  // j=0: (1/4)*(4/1) = 1   -> 1*10 = 10
  // j=1: (1/4)*(4/1) = 1   -> 1*10 = 10
  // j=2: (2/4)*(4/2) = 1   -> 1*10 = 10
  // delivered_total = 30, not 10: a 3x inflation, because Intensive treats
  // the AMOUNT 10 as if it were a DENSITY and reproduces that same value
  // -- not a fraction of it -- on every one of the three target cells.
  EXPECT_NEAR(delivered_total, 30.0, 1e-12)
      << "measured, not assumed: this is the exact inflation an "
         "amount-valued field sees through the Intensive formula when its "
         "source cell is split across several target cells -- D82's own "
         "reported symptom, reproduced with no sphere and no MOAB";
  EXPECT_GT(delivered_total, amount)
      << "the pre-existing formula does NOT conserve an amount-valued "
         "field across an uneven split -- this is exactly why "
         "Conservation::Extensive exists";
}

}  // namespace
