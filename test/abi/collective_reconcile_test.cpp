// Tier-0, MPI-free tests of the pure arithmetic
// src/abi/participant.cpp's real MPI_Allreduce(..., MPI_2INT, MPI_MINLOC,
// ...) glue, and its own test-only stub reducer (test_support.hpp's
// force_fake_collective_peers(), exercised end to end in
// participant_test.cpp), both reduce to -- see collective_reconcile.hpp's
// own header comment for why the split exists.

#include "fuse/abi/collective_reconcile.hpp"

#include <gtest/gtest.h>

#include "fuse/abi/types.h"  // FUSE_OK, FUSE_ERR_*

using fuse::abi_detail::decide_split_colour_key;
using fuse::abi_detail::reconcile_local_statuses;

TEST(ReconcileLocalStatuses, EveryRankOkAgreesOk) {
  const auto r = reconcile_local_statuses({FUSE_OK, FUSE_OK, FUSE_OK});
  EXPECT_EQ(r.status, FUSE_OK);
  EXPECT_EQ(r.origin_rank, 0);
}

TEST(ReconcileLocalStatuses, OneRefusalAmongManyWins) {
  const auto r = reconcile_local_statuses({FUSE_OK, FUSE_ERR_STATE, FUSE_OK});
  EXPECT_EQ(r.status, FUSE_ERR_STATE);
  EXPECT_EQ(r.origin_rank, 1);
}

TEST(ReconcileLocalStatuses, TheRefusingRankNeedNotBeRankZero) {
  const auto r = reconcile_local_statuses({FUSE_OK, FUSE_OK, FUSE_ERR_INVALID, FUSE_OK});
  EXPECT_EQ(r.status, FUSE_ERR_INVALID);
  EXPECT_EQ(r.origin_rank, 2);
}

// FUSE_ERR_STATE is -3, FUSE_ERR_INVALID is -1 (entries.def); the more
// negative code wins deterministically regardless of rank order, which is
// what makes "every rank refuses with the same code" possible without a
// second round of communication to break the tie.
TEST(ReconcileLocalStatuses, TwoDifferentRefusalsPickTheMoreNegativeCodeDeterministically) {
  const auto r_forward = reconcile_local_statuses({FUSE_ERR_INVALID, FUSE_ERR_STATE});
  EXPECT_EQ(r_forward.status, FUSE_ERR_STATE);
  EXPECT_EQ(r_forward.origin_rank, 1);

  const auto r_reversed = reconcile_local_statuses({FUSE_ERR_STATE, FUSE_ERR_INVALID});
  EXPECT_EQ(r_reversed.status, FUSE_ERR_STATE);
  EXPECT_EQ(r_reversed.origin_rank, 0);
}

TEST(ReconcileLocalStatuses, TiesBreakByTheLowestRank) {
  const auto r = reconcile_local_statuses({FUSE_OK, FUSE_ERR_INVALID, FUSE_ERR_INVALID});
  EXPECT_EQ(r.status, FUSE_ERR_INVALID);
  EXPECT_EQ(r.origin_rank, 1);
}

TEST(ReconcileLocalStatuses, SingleRankAlwaysOriginatesItsOwnStatus) {
  const auto ok = reconcile_local_statuses({FUSE_OK});
  EXPECT_EQ(ok.status, FUSE_OK);
  EXPECT_EQ(ok.origin_rank, 0);
  const auto err = reconcile_local_statuses({FUSE_ERR_RUNTIME});
  EXPECT_EQ(err.status, FUSE_ERR_RUNTIME);
  EXPECT_EQ(err.origin_rank, 0);
}

TEST(ReconcileLocalStatuses, EmptyInputIsDefensiveOk) {
  const auto r = reconcile_local_statuses({});
  EXPECT_EQ(r.status, FUSE_OK);
  EXPECT_EQ(r.origin_rank, 0);
}

// The pure half of the per-component split decision
// src/abi/participant.cpp's real MPI_Comm_split call feeds from -- see
// collective_reconcile.hpp's own doc comment for why a single
// make_participant_set_reducer call would compute the SAME colour for every
// component (which this function's own existence avoids).

TEST(DecideSplitColourKey, NoComponentReturnsNullopt) {
  EXPECT_FALSE(decide_split_colour_key(std::nullopt, 0).has_value());
}

TEST(DecideSplitColourKey, NegativePositionReturnsNulloptEvenWithAComponent) {
  EXPECT_FALSE(decide_split_colour_key(std::uint32_t{7}, -1).has_value());
}

TEST(DecideSplitColourKey, ColourIsExactlyTheComponentValue) {
  const auto ck = decide_split_colour_key(std::uint32_t{3}, 0);
  ASSERT_TRUE(ck.has_value());
  EXPECT_EQ(ck->colour, 3);
}

TEST(DecideSplitColourKey, KeyIsExactlyThePositionUnchanged) {
  const auto ck = decide_split_colour_key(std::uint32_t{3}, 5);
  ASSERT_TRUE(ck.has_value());
  EXPECT_EQ(ck->key, 5);
}

// THE PROPERTY ROUND 3's BUG DID NOT HAVE: two DIFFERENT components must
// never produce the SAME colour -- make_participant_set_reducer's own
// boolean "am I in my own set" always answered yes, uniformly 0, for
// every caller regardless of which component it represented.
TEST(DecideSplitColourKey, TwoDifferentComponentsNeverShareAColour) {
  const auto u = decide_split_colour_key(std::uint32_t{0}, 0);
  const auto v = decide_split_colour_key(std::uint32_t{1}, 0);
  ASSERT_TRUE(u.has_value());
  ASSERT_TRUE(v.has_value());
  EXPECT_NE(u->colour, v->colour);
}

// Two ranks of the SAME component get the SAME colour, distinguished only
// by key -- MPI_Comm_split groups by colour, so this is what keeps them on
// one communicator together.
TEST(DecideSplitColourKey, SameComponentDifferentRanksShareColourDistinctKey) {
  const auto rank_a = decide_split_colour_key(std::uint32_t{2}, 0);
  const auto rank_b = decide_split_colour_key(std::uint32_t{2}, 1);
  ASSERT_TRUE(rank_a.has_value());
  ASSERT_TRUE(rank_b.has_value());
  EXPECT_EQ(rank_a->colour, rank_b->colour);
  EXPECT_NE(rank_a->key, rank_b->key);
}
