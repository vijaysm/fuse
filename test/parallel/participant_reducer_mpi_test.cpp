// fuse::linalg::make_participant_set_reducer()
// (src/linalg/comm_mpi.cpp), over a REAL MPI_COMM_WORLD. Mirrors test/
// parallel/linalg_mpi_test.cpp's own structure and its own closed-form
// reference convention (every rank reconstructs the identical answer a
// single rank computing the whole problem serially would get, with no
// extra collective needed to check it).
//
// MPI CANNOT START IN THIS SANDBOX (same constraint linalg_mpi_test.cpp
// names): this file is compiled -- registered with fuse_add_mpi_test,
// RANKS 2 3 -- but never run here; outcomes are a host action.
//
// THE PROPERTY THIS FILE IS FOR, that linalg_mpi_test.cpp's own
// `make_mpi_reducer()` tests cannot exercise at all: a reduction over a
// PROPER SUBSET of the base communicator's own ranks, with a rank
// OUTSIDE that subset getting no Reducer object to call at all ("a rank
// outside the participant set must not enter a reduction",
// comm.hpp's own header comment on this factory).

#include "fuse/linalg/comm.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

namespace {

using fuse::linalg::ParticipantSetReducerResult;

/// The closed-form reference at GLOBAL index `i`, identical convention to
/// linalg_mpi_test.cpp's own `g()`.
double g(int i) { return std::sin(1.7 * static_cast<double>(i) + 0.4); }

// Every rank of a 3-rank world EXCEPT rank 1 participates -- a
// non-contiguous, non-prefix subset, deliberately: a bug that only shows
// up when the participant set is not "ranks 0..k-1" (e.g. an off-by-one
// in `MPI_UNDEFINED` colouring, or the split's own rank order silently
// reverting to base rank number instead of position in `participants`)
// cannot hide behind a subset that happens to look like the whole
// communicator's own prefix.
TEST(ParticipantSetReducerMpi, NonContiguousSubsetSumMatchesTheClosedFormReference) {
  FUSE_REQUIRE_RANKS(3);
  const int me = fuse::testing::rank();
  const std::vector<int> participants = {0, 2};
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const auto built = fuse::linalg::make_participant_set_reducer(comm_f, participants);
  ASSERT_TRUE(built.ok()) << built.error().message;
  const ParticipantSetReducerResult& res = built.value();

  const bool participating = (me == 0 || me == 2);
  if (!participating) {
    EXPECT_EQ(res.reducer, nullptr)
        << "rank " << me << " is not in {0, 2} and must get no Reducer at all";
    return;
  }
  ASSERT_NE(res.reducer, nullptr);

  // rank()/size() read as POSITION IN participants, not the base rank
  // number -- participants[0] == 0 is sub-rank 0, participants[1] == 2
  // is sub-rank 1.
  EXPECT_EQ(res.reducer->size(), 2);
  EXPECT_EQ(res.reducer->rank(), me == 0 ? 0 : 1);

  const double local = g(me);
  const double expect_sum = g(0) + g(2);
  EXPECT_DOUBLE_EQ(res.reducer->sum(local), expect_sum);

  const double expect_max = std::max(std::fabs(g(0)), std::fabs(g(2)));
  EXPECT_DOUBLE_EQ(res.reducer->max(std::fabs(local)), expect_max);

  std::vector<double> gathered(2, 0.0);
  res.reducer->allgather(&local, 1, gathered.data());
  EXPECT_DOUBLE_EQ(gathered[0], g(0));
  EXPECT_DOUBLE_EQ(gathered[1], g(2));
}

// The vectorised sum(double*, n) overload, over the SAME non-contiguous
// subset, at whatever world size is running (participants = every even
// rank -- exercises the 2-rank leg, where "every even rank" is just {0},
// a one-participant split, as well as the 3-rank leg's {0, 2}).
TEST(ParticipantSetReducerMpi, VectorisedSumMatchesTheClosedFormReferenceOverEvenRanks) {
  const int world = fuse::testing::size();
  const int me = fuse::testing::rank();
  std::vector<int> participants;
  for (int r = 0; r < world; r += 2) participants.push_back(r);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const auto built = fuse::linalg::make_participant_set_reducer(comm_f, participants);
  ASSERT_TRUE(built.ok()) << built.error().message;
  const ParticipantSetReducerResult& res = built.value();

  const bool participating =
      std::find(participants.begin(), participants.end(), me) != participants.end();
  if (!participating) {
    EXPECT_EQ(res.reducer, nullptr);
    return;
  }
  ASSERT_NE(res.reducer, nullptr);
  EXPECT_EQ(res.reducer->size(), static_cast<int>(participants.size()));

  constexpr int kWidth = 3;
  std::vector<double> local(kWidth);
  for (int k = 0; k < kWidth; ++k) local[k] = g(me * 100 + k);
  std::vector<double> expect(kWidth, 0.0);
  for (const int p : participants) {
    for (int k = 0; k < kWidth; ++k) expect[k] += g(p * 100 + k);
  }

  res.reducer->sum(local.data(), local.size());
  for (int k = 0; k < kWidth; ++k) {
    EXPECT_DOUBLE_EQ(local[k], expect[k]) << "component " << k;
  }
}

// Every rank participates (participants == every base rank, in order):
// the split's own rank/size must then agree exactly with the base
// communicator's own rank/size -- the "whole communicator" degenerate
// case of the same factory, proving it is not merely correct on a
// proper subset by accident of this file's other two cases.
TEST(ParticipantSetReducerMpi, EveryRankParticipatingMatchesTheBaseCommunicator) {
  const int world = fuse::testing::size();
  const int me = fuse::testing::rank();
  std::vector<int> participants(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r) participants[static_cast<std::size_t>(r)] = r;
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const auto built = fuse::linalg::make_participant_set_reducer(comm_f, participants);
  ASSERT_TRUE(built.ok()) << built.error().message;
  ASSERT_NE(built.value().reducer, nullptr);
  EXPECT_EQ(built.value().reducer->size(), world);
  EXPECT_EQ(built.value().reducer->rank(), me);

  const double expect_sum = [world] {
    double s = 0.0;
    for (int r = 0; r < world; ++r) s += g(r);
    return s;
  }();
  EXPECT_DOUBLE_EQ(built.value().reducer->sum(g(me)), expect_sum);
}

}  // namespace
