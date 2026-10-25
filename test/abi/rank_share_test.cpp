// Tier-0, MPI-free tests of the pure
// arithmetic src/abi/provider_adapter.cpp's register_fields() uses to
// size a multi-rank component's own registration per-position -- see
// include/fuse/abi/rank_share.hpp's own header comment for why this
// duplicates fuse::native::rank_slice's own formula rather than calling
// it.

#include "fuse/abi/rank_share.hpp"

#include <gtest/gtest.h>

using fuse::abi_detail::even_rank_share;

// THE EXACT MEASURED CASE (provider_subset_mpi_test.cpp's own 3-rank
// graph): one cell, two positions -- the
// first position gets it, the second gets none.
TEST(EvenRankShare, OneCellOverTwoPositionsGivesItAllToTheFirst) {
  const auto [lo0, hi0] = even_rank_share(1, 0, 2);
  EXPECT_EQ(hi0 - lo0, 1u);
  const auto [lo1, hi1] = even_rank_share(1, 1, 2);
  EXPECT_EQ(hi1 - lo1, 0u);
}

TEST(EvenRankShare, EvenlyDivisibleTotalSplitsExactly) {
  const auto [lo0, hi0] = even_rank_share(4, 0, 2);
  const auto [lo1, hi1] = even_rank_share(4, 1, 2);
  EXPECT_EQ(hi0 - lo0, 2u);
  EXPECT_EQ(hi1 - lo1, 2u);
  EXPECT_EQ(hi0, lo1) << "the two slices must be contiguous and non-overlapping";
}

// 7 over 3: the first (7 % 3 = 1) position gets one extra.
TEST(EvenRankShare, RemainderGoesToTheEarliestPositionsOnly) {
  const auto [lo0, hi0] = even_rank_share(7, 0, 3);
  const auto [lo1, hi1] = even_rank_share(7, 1, 3);
  const auto [lo2, hi2] = even_rank_share(7, 2, 3);
  EXPECT_EQ(hi0 - lo0, 3u);
  EXPECT_EQ(hi1 - lo1, 2u);
  EXPECT_EQ(hi2 - lo2, 2u);
  EXPECT_EQ(hi0, lo1);
  EXPECT_EQ(hi1, lo2);
  EXPECT_EQ(hi2, 7u);
}

TEST(EvenRankShare, GroupSizeOfOneOwnsTheWholeRange) {
  const auto [lo, hi] = even_rank_share(5, 0, 1);
  EXPECT_EQ(lo, 0u);
  EXPECT_EQ(hi, 5u);
}

TEST(EvenRankShare, GroupSizeOfZeroDegeneratesToTheWholeRangeLikeRankSliceDoes) {
  const auto [lo, hi] = even_rank_share(5, 0, 0);
  EXPECT_EQ(lo, 0u);
  EXPECT_EQ(hi, 5u);
}

TEST(EvenRankShare, EmptyTotalGivesEveryPositionNothing) {
  const auto [lo0, hi0] = even_rank_share(0, 0, 3);
  const auto [lo1, hi1] = even_rank_share(0, 2, 3);
  EXPECT_EQ(hi0 - lo0, 0u);
  EXPECT_EQ(hi1 - lo1, 0u);
}
