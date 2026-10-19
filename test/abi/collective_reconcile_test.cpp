// D61 (KNOWN_ISSUES.md). Tier-0, MPI-free tests of the pure arithmetic
// src/abi/participant.cpp's real MPI_Allreduce(..., MPI_2INT, MPI_MINLOC,
// ...) glue, and its own test-only stub reducer (test_support.hpp's
// force_fake_collective_peers(), exercised end to end in
// participant_test.cpp), both reduce to -- see collective_reconcile.hpp's
// own header comment for why the split exists.

#include "fuse/abi/collective_reconcile.hpp"

#include <gtest/gtest.h>

#include "fuse/abi/types.h"  // FUSE_OK, FUSE_ERR_*

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
