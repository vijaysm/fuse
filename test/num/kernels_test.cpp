#include <gtest/gtest.h>
#include <fuse/num/kernels.hpp>

#include <cmath>
#include <limits>
#include <vector>

using fuse::AccumOp;
namespace num = fuse::num;

TEST(Kernels, Copy) {
  const std::vector<double> a{1, 2, 3};
  std::vector<double> b(3, 0.0);
  num::copy(a.data(), b.data(), 3);
  EXPECT_EQ(a, b);
}

TEST(Kernels, AccumulateSum) {
  std::vector<double> acc(2, 0.0);
  const std::vector<double> s1{1.0, 2.0};
  const std::vector<double> s2{3.0, 4.0};
  num::accumulate_into(AccumOp::Sum, s1.data(), acc.data(), 2, 0);
  num::accumulate_into(AccumOp::Sum, s2.data(), acc.data(), 2, 1);
  num::accumulate_finalize(AccumOp::Sum, acc.data(), 2, 2);
  EXPECT_DOUBLE_EQ(acc[0], 4.0);
  EXPECT_DOUBLE_EQ(acc[1], 6.0);
}

TEST(Kernels, AccumulateMeanDividesBySampleCount) {
  std::vector<double> acc(2, 0.0);
  const std::vector<double> s1{1.0, 2.0};
  const std::vector<double> s2{3.0, 6.0};
  num::accumulate_into(AccumOp::Mean, s1.data(), acc.data(), 2, 0);
  num::accumulate_into(AccumOp::Mean, s2.data(), acc.data(), 2, 1);
  num::accumulate_finalize(AccumOp::Mean, acc.data(), 2, 2);
  EXPECT_DOUBLE_EQ(acc[0], 2.0);
  EXPECT_DOUBLE_EQ(acc[1], 4.0);
}

TEST(Kernels, AccumulateMinMax) {
  std::vector<double> lo(1, 0.0), hi(1, 0.0);
  const double a = 5.0, b = -2.0, c = 9.0;
  num::accumulate_into(AccumOp::Min, &a, lo.data(), 1, 0);
  num::accumulate_into(AccumOp::Min, &b, lo.data(), 1, 1);
  num::accumulate_into(AccumOp::Min, &c, lo.data(), 1, 2);
  EXPECT_DOUBLE_EQ(lo[0], -2.0);

  num::accumulate_into(AccumOp::Max, &a, hi.data(), 1, 0);
  num::accumulate_into(AccumOp::Max, &b, hi.data(), 1, 1);
  num::accumulate_into(AccumOp::Max, &c, hi.data(), 1, 2);
  EXPECT_DOUBLE_EQ(hi[0], 9.0);
}

TEST(Kernels, AccumulateIntegralScalesByDt) {
  std::vector<double> acc(1, 0.0);
  const double s = 2.0;
  num::accumulate_into(AccumOp::Integral, &s, acc.data(), 1, 0);
  num::accumulate_into(AccumOp::Integral, &s, acc.data(), 1, 1);
  num::accumulate_finalize_integral(acc.data(), 1, 0.5);
  EXPECT_DOUBLE_EQ(acc[0], 2.0);   // (2 + 2) * 0.5
}

TEST(Kernels, AccumulateInstantTakesTheLatest) {
  std::vector<double> acc(1, 0.0);
  const double a = 1.0, b = 9.0;
  num::accumulate_into(AccumOp::Instant, &a, acc.data(), 1, 0);
  num::accumulate_into(AccumOp::Instant, &b, acc.data(), 1, 1);
  num::accumulate_finalize(AccumOp::Instant, acc.data(), 1, 2);
  EXPECT_DOUBLE_EQ(acc[0], 9.0);
}

TEST(Kernels, RelaxBlends) {
  const std::vector<double> prev{0.0, 10.0};
  const std::vector<double> cur{10.0, 0.0};
  std::vector<double> out(2, 0.0);
  num::relax(0.25, prev.data(), cur.data(), out.data(), 2);
  EXPECT_DOUBLE_EQ(out[0], 2.5);
  EXPECT_DOUBLE_EQ(out[1], 7.5);
}

TEST(Kernels, RelaxOmegaOneIsTheCurrentValue) {
  const std::vector<double> prev{1.0}, cur{5.0};
  std::vector<double> out(1, 0.0);
  num::relax(1.0, prev.data(), cur.data(), out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 5.0);
}

TEST(Kernels, ExtrapolateOrderZeroCopies) {
  const std::vector<double> hist{1.0, 2.0, 3.0};   // newest last
  std::vector<double> out(1, 0.0);
  num::extrapolate(0, &hist[2], 1, out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 3.0);
}

TEST(Kernels, ExtrapolateOrderOneIsLinear) {
  // y_{k-1}=2, y_k=3 -> predict 4
  const std::vector<double> hist{1.0, 2.0, 3.0};
  std::vector<double> out(1, 0.0);
  num::extrapolate(1, &hist[2], 1, out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 4.0);
}

TEST(Kernels, ExtrapolateOrderTwoIsExactOnAQuadratic) {
  // y = k^2 at k = 0,1,2 -> 0,1,4; predict k=3 -> 9
  const std::vector<double> hist{0.0, 1.0, 4.0};
  std::vector<double> out(1, 0.0);
  num::extrapolate(2, &hist[2], 1, out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 9.0);
}

// Multi-element samples: `history` points at the newest sample and older
// samples lie `stride` elements earlier. Element i of the previous sample is
// (history - stride)[i], which must be formed as pointer arithmetic on the
// caller's buffer rather than as history[i - stride].
TEST(Kernels, ExtrapolateHandlesMultiElementSamplesWithStride) {
  // Two components; comp0 = k^2, comp1 = k^2 + 10, at k = 0, 1, 2.
  const std::vector<double> hist{0.0, 10.0, 1.0, 11.0, 4.0, 14.0};
  const std::size_t stride = 2;
  const double* newest = hist.data() + 4;

  std::vector<double> out(2, 0.0);
  num::extrapolate(0, newest, stride, out.data(), 2);
  EXPECT_DOUBLE_EQ(out[0], 4.0);
  EXPECT_DOUBLE_EQ(out[1], 14.0);

  num::extrapolate(1, newest, stride, out.data(), 2);
  EXPECT_DOUBLE_EQ(out[0], 7.0);    // 2*4 - 1
  EXPECT_DOUBLE_EQ(out[1], 17.0);   // 2*14 - 11

  num::extrapolate(2, newest, stride, out.data(), 2);
  EXPECT_DOUBLE_EQ(out[0], 9.0);    // 3*4 - 3*1 + 0
  EXPECT_DOUBLE_EQ(out[1], 19.0);   // 3*14 - 3*11 + 10
}

// A negative order must not reach back into history at all; it degrades to
// order 0 rather than reading history[-stride].
TEST(Kernels, ExtrapolateNegativeOrderDegradesToCopy) {
  const std::vector<double> hist{7.0};
  std::vector<double> out(1, 0.0);
  num::extrapolate(-1, hist.data(), 1, out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 7.0);
}

TEST(Kernels, SumAndWeightedSum) {
  const std::vector<double> v{1.0, 2.0, 3.0};
  const std::vector<double> w{0.5, 0.5, 1.0};
  EXPECT_DOUBLE_EQ(num::sum(v.data(), 3), 6.0);
  EXPECT_DOUBLE_EQ(num::weighted_sum(v.data(), w.data(), 3), 4.5);
}

// Standing rule 3: the reductions accumulate strictly in ascending index
// order. This input is order-sensitive -- left to right it cancels to 0.0,
// while any reassociation (summing the two large terms first) gives 1.0.
// The test pins the accumulation order documented on the declarations.
TEST(Kernels, SumAccumulatesInAscendingIndexOrder) {
  const std::vector<double> v{1e16, 1.0, -1e16};
  EXPECT_DOUBLE_EQ(num::sum(v.data(), 3), 0.0);

  const std::vector<double> ones(3, 1.0);
  EXPECT_DOUBLE_EQ(num::weighted_sum(v.data(), ones.data(), 3), 0.0);
}

TEST(Kernels, Norms) {
  const std::vector<double> v{3.0, -4.0};
  EXPECT_DOUBLE_EQ(num::norm_l2(v.data(), 2), 5.0);
  EXPECT_DOUBLE_EQ(num::norm_linf(v.data(), 2), 4.0);

  const std::vector<double> a{1.0, 1.0}, b{1.0, 2.0};
  EXPECT_DOUBLE_EQ(num::norm_l2_diff(a.data(), b.data(), 2), 1.0);
}

TEST(Kernels, DetectsNonFinite) {
  std::vector<double> v{1.0, 2.0};
  EXPECT_FALSE(num::has_nonfinite(v.data(), 2));
  v[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(num::has_nonfinite(v.data(), 2));
  v[1] = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(num::has_nonfinite(v.data(), 2));
}

TEST(Kernels, WorkForFloatToo) {
  const std::vector<float> v{3.0F, -4.0F};
  EXPECT_FLOAT_EQ(num::norm_l2(v.data(), 2), 5.0F);
}

// Empty ranges are well defined: the reductions return their identity and no
// element is read.
TEST(Kernels, EmptyRangesAreIdentities) {
  const double* none = nullptr;
  EXPECT_DOUBLE_EQ(num::sum(none, 0), 0.0);
  EXPECT_DOUBLE_EQ(num::norm_l2(none, 0), 0.0);
  EXPECT_DOUBLE_EQ(num::norm_linf(none, 0), 0.0);
  EXPECT_FALSE(num::has_nonfinite(none, 0));
}
