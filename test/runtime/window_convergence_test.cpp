// Unit 2b (D19), step (c) -- AGREED CONVERGENCE for a window-scoped
// iteration region, checked without MPI.
//
// A mistake in `decide_window_convergence()` is a WRONG NUMBER, falsifiable
// right here with hand-built `gathered` data -- exactly the reason it is
// kept apart from the MPI exchange that gathers it (`epoch_schedule_test.
// cpp`'s own top comment, for the same reason, about a schedule instead of
// a decision).

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <fuse/runtime/window_convergence.hpp>

namespace {

using fuse::ConvergenceCriterion;
using fuse::ConvergenceMeasure;
using fuse::MeasureKind;
using fuse::WindowBlockSample;
using fuse::WindowConvergenceResult;
using fuse::decide_window_convergence;

ConvergenceCriterion two_absolute_measures(double tol = 1e-9) {
  ConvergenceCriterion c;
  c.add(ConvergenceMeasure{MeasureKind::Absolute, tol, "a"});
  c.add(ConvergenceMeasure{MeasureKind::Absolute, tol, "b"});
  return c;
}

TEST(WindowConvergence, SingleRankMatchesLocalComputationBitForBit) {
  // The single-process case (RunnerTransport.NonHostedMember... and every
  // OTHER window test in this suite today): `gathered.size() == 1`, every
  // block owned by that one entry -- the pre-this-fix local decision,
  // reproduced exactly, not approximated.
  const ConvergenceCriterion measures = two_absolute_measures(1e-9);
  const std::vector<std::size_t> measure_block{0, 1};
  std::vector<double> block_first{0.0, 0.0};
  std::vector<double> residual_norms, reference_norms;

  const std::vector<std::vector<WindowBlockSample>> gathered{
      {WindowBlockSample{1e-12 * 1e-12, 4.0, true},
       WindowBlockSample{9.0, 9.0, true}}};

  const WindowConvergenceResult wc = decide_window_convergence(
      gathered, measures, measure_block, block_first, /*first_sweep=*/true,
      residual_norms, reference_norms);

  EXPECT_EQ(wc.unmeasured_block, -1);
  EXPECT_DOUBLE_EQ(wc.block_residual[0], 1e-12);
  EXPECT_DOUBLE_EQ(wc.block_residual[1], 3.0);
  EXPECT_DOUBLE_EQ(wc.block_value[0], 2.0);
  EXPECT_DOUBLE_EQ(wc.block_value[1], 3.0);
  // Block 1's own residual (3.0) is nowhere near 1e-9: not converged.
  EXPECT_FALSE(wc.converged);
}

TEST(WindowConvergence, EachBlockIsSummedOnlyFromItsOwningRanks) {
  // Two "ranks": rank 0 owns block 0 only (small residual, converged on
  // its own); rank 1 owns block 1 only (also small). Neither rank's own
  // NON-owned entry is a legitimate zero -- set to NaN here specifically
  // to prove the implementation EXCLUDES it rather than summing it (a
  // `0.0 * NaN` or `NaN + anything` would poison the block it touches;
  // if this test passes, the exclusion happens before any arithmetic
  // touches the non-owned entry at all).
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const ConvergenceCriterion measures = two_absolute_measures(1e-9);
  const std::vector<std::size_t> measure_block{0, 1};
  std::vector<double> block_first{0.0, 0.0};
  std::vector<double> residual_norms, reference_norms;

  const std::vector<std::vector<WindowBlockSample>> gathered{
      /*rank 0*/ {WindowBlockSample{1e-20, 1.0, true},
                  WindowBlockSample{nan, nan, false}},
      /*rank 1*/ {WindowBlockSample{nan, nan, false},
                  WindowBlockSample{1e-20, 1.0, true}},
  };

  const WindowConvergenceResult wc = decide_window_convergence(
      gathered, measures, measure_block, block_first, /*first_sweep=*/true,
      residual_norms, reference_norms);

  ASSERT_EQ(wc.unmeasured_block, -1);
  EXPECT_TRUE(std::isfinite(wc.block_residual[0]));
  EXPECT_TRUE(std::isfinite(wc.block_residual[1]));
  EXPECT_LT(wc.block_residual[0], 1e-9);
  EXPECT_LT(wc.block_residual[1], 1e-9);
  EXPECT_TRUE(wc.converged);
}

TEST(WindowConvergence, NoOwnerIsReportedNotSilentlyTreatedAsConverged) {
  // Neither rank owns block 1 at all -- a setup defect (mirrors
  // reduce_region_residuals()'s own "no participant owns this block"
  // check), never a converged zero.
  const ConvergenceCriterion measures = two_absolute_measures(1e-9);
  const std::vector<std::size_t> measure_block{0, 1};
  std::vector<double> block_first{0.0, 0.0};
  std::vector<double> residual_norms, reference_norms;

  const std::vector<std::vector<WindowBlockSample>> gathered{
      {WindowBlockSample{0.0, 1.0, true}, WindowBlockSample{0.0, 0.0, false}},
      {WindowBlockSample{0.0, 1.0, true}, WindowBlockSample{0.0, 0.0, false}},
  };

  const WindowConvergenceResult wc = decide_window_convergence(
      gathered, measures, measure_block, block_first, /*first_sweep=*/true,
      residual_norms, reference_norms);

  EXPECT_EQ(wc.unmeasured_block, 1);
  EXPECT_FALSE(wc.converged);
}

TEST(WindowConvergence, SumIsFixedOrderAcrossDifferentParticipantCounts) {
  // The SAME total contribution to one block (residual^2 = 5.0, split
  // three different ways: all on one rank, split 2 ways, split 3 ways)
  // must agree bit-for-bit, since the sum walks ranks in ascending index
  // order regardless of how many there are -- the property `rank_
  // agreement.hpp`'s own "a rule you can only exercise under mpiexec is a
  // rule you cannot debug" exists to make checkable without MPI.
  ConvergenceCriterion measures;
  measures.add(ConvergenceMeasure{MeasureKind::Absolute, 1e-9, "a"});
  const std::vector<std::size_t> measure_block{0};
  std::vector<double> block_first{0.0};
  std::vector<double> rn, fn;

  const WindowConvergenceResult one = decide_window_convergence(
      {{WindowBlockSample{5.0, 1.0, true}}}, measures, measure_block,
      block_first, true, rn, fn);

  block_first = {0.0};
  const WindowConvergenceResult two = decide_window_convergence(
      {{WindowBlockSample{2.0, 0.4, true}},
       {WindowBlockSample{3.0, 0.6, true}}},
      measures, measure_block, block_first, true, rn, fn);

  block_first = {0.0};
  const WindowConvergenceResult three = decide_window_convergence(
      {{WindowBlockSample{1.0, 0.2, true}},
       {WindowBlockSample{1.5, 0.3, true}},
       {WindowBlockSample{2.5, 0.5, true}}},
      measures, measure_block, block_first, true, rn, fn);

  EXPECT_DOUBLE_EQ(one.block_residual[0], two.block_residual[0]);
  EXPECT_DOUBLE_EQ(one.block_residual[0], three.block_residual[0]);
  EXPECT_DOUBLE_EQ(one.block_value[0], two.block_value[0]);
  EXPECT_DOUBLE_EQ(one.block_value[0], three.block_value[0]);
}

TEST(WindowConvergence, BlockFirstIsCapturedFromTheAgreedResidualNotALocalOne) {
  // ResidualRelative divides by block_first -- the FIRST sweep's own
  // AGREED residual, not any one rank's own local slice. Split across
  // two ranks so a bug that captured block_first from only ONE rank's
  // own contribution (rather than the agreed sum) would be visible.
  ConvergenceCriterion measures;
  measures.add(ConvergenceMeasure{MeasureKind::ResidualRelative, 1e-9, "a"});
  const std::vector<std::size_t> measure_block{0};
  std::vector<double> block_first{0.0};
  std::vector<double> rn, fn;

  // Sweep 0: agreed residual^2 = 3.0^2 + 4.0^2 = 25 -> residual = 5.
  const WindowConvergenceResult first = decide_window_convergence(
      {{WindowBlockSample{9.0, 1.0, true}},
       {WindowBlockSample{16.0, 1.0, true}}},
      measures, measure_block, block_first, /*first_sweep=*/true, rn, fn);
  ASSERT_EQ(first.unmeasured_block, -1);
  EXPECT_DOUBLE_EQ(block_first[0], 5.0);
  EXPECT_DOUBLE_EQ(fn[0], 5.0);  // ResidualRelative's own reference at k==0

  // Sweep 1: residual has dropped to 5e-10 total -- block_first must
  // still read 5.0 (sweep 0's own agreed value), not be re-captured.
  const WindowConvergenceResult second = decide_window_convergence(
      {{WindowBlockSample{1e-19, 1.0, true}},
       {WindowBlockSample{1.5e-19, 1.0, true}}},
      measures, measure_block, block_first, /*first_sweep=*/false, rn, fn);
  ASSERT_EQ(second.unmeasured_block, -1);
  EXPECT_DOUBLE_EQ(block_first[0], 5.0);
  EXPECT_DOUBLE_EQ(fn[0], 5.0);
  EXPECT_TRUE(second.converged);
}

TEST(WindowConvergence, MutationDecidingFromOneRanksOwnPairsOnlyDisagrees) {
  // The lead's own suggested mutation for this step: "a rank deciding
  // from its local pairs only" instead of the agreed sum. Demonstrated
  // here as a direct comparison rather than a code mutation, since
  // `decide_window_convergence()` is new code with no PRIOR behaviour to
  // revert to (unlike window-runner-hosting-ignored/-priming-through-
  // transfer-disabled, which mutate an EXISTING line) -- feeding the
  // SAME gathered data but only rank 0's own slice (`{gathered[0]}`,
  // what a rank "deciding from its own local pairs only" would compute)
  // reaches a DIFFERENT, wrong verdict than feeding the full gathered
  // array: rank 0 alone sees its own small residual and calls block 0
  // converged; the agreed sum (both ranks' own contributions) is well
  // above tolerance and correctly refuses.
  ConvergenceCriterion measures;
  measures.add(ConvergenceMeasure{MeasureKind::Absolute, 1e-6, "a"});
  const std::vector<std::size_t> measure_block{0};
  std::vector<double> block_first{0.0};
  std::vector<double> rn, fn;

  const std::vector<std::vector<WindowBlockSample>> gathered{
      {WindowBlockSample{1e-14, 1.0, true}},   // rank 0: tiny, "converged"
      {WindowBlockSample{4.0, 1.0, true}},     // rank 1: large, NOT converged
  };

  const WindowConvergenceResult agreed = decide_window_convergence(
      gathered, measures, measure_block, block_first, true, rn, fn);
  EXPECT_FALSE(agreed.converged)
      << "the AGREED sum includes rank 1's own large contribution";

  block_first = {0.0};
  const std::vector<std::vector<WindowBlockSample>> rank0_only{gathered[0]};
  const WindowConvergenceResult local_only = decide_window_convergence(
      rank0_only, measures, measure_block, block_first, true, rn, fn);
  EXPECT_TRUE(local_only.converged)
      << "a rank deciding from its own local pairs only reaches the WRONG "
         "verdict -- exactly the defect this step closes";
}

}  // namespace
