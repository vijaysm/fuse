// The rank-agreement exchange pattern, checked without MPI.
//
// A mismatch in this schedule is a HANG: one rank posts a receive nobody sends
// to, or sends where nobody is listening. That is invisible to a serial test
// and expensive to observe in a parallel one -- it shows up as a stalled leg
// on someone else's machine. As arithmetic over (participants, index) it is
// falsifiable here, which is the whole reason it was separated from the MPI
// calls that carry it out.

#include <gtest/gtest.h>

#include <map>
#include <type_traits>
#include <set>
#include <utility>
#include <vector>

#include <fuse/runtime/epoch_schedule.hpp>

namespace {

using fuse::epoch_schedule;

/// Every (sender, receiver) pair either side of the exchange believes in.
struct Pairs {
  std::multiset<std::pair<int, int>> sends;     // as the sender sees them
  std::multiset<std::pair<int, int>> receives;  // as the receiver sees them
};

Pairs collect(int n) {
  Pairs p;
  for (int i = 0; i < n; ++i) {
    const auto s = epoch_schedule(n, i);
    for (int r = 0; r < s.reduce_recv_count; ++r) {
      p.receives.emplace(s.reduce_recv_from[static_cast<std::size_t>(r)], i);
    }
    if (s.reduce_send_to >= 0) p.sends.emplace(i, s.reduce_send_to);
    if (s.result_recv_from >= 0) p.receives.emplace(s.result_recv_from, i);
    // Participant 0 sends the answer to 1..n-1; the schedule holds the count
    // rather than the list, because that list is the one part whose size grows
    // with the job and materialising it would allocate.
    for (int k = 1; k <= s.result_send_count; ++k) p.sends.emplace(i, k);
  }
  return p;
}

}  // namespace

TEST(EpochSchedule, EverySendHasExactlyOneMatchingReceive) {
  for (int n = 2; n <= 16; ++n) {
    const Pairs p = collect(n);
    EXPECT_EQ(p.sends, p.receives)
        << "participants=" << n
        << ": a send has no matching receive, or the other way round, which is "
           "a hang rather than a wrong answer";
  }
}

TEST(EpochSchedule, EveryParticipantEndsUpWithTheAnswer) {
  for (int n = 2; n <= 16; ++n) {
    for (int i = 0; i < n; ++i) {
      const auto s = epoch_schedule(n, i);
      if (i == 0) {
        EXPECT_EQ(s.reduce_send_to, -1) << "n=" << n << " participant 0 sent its result away";
        EXPECT_EQ(s.result_recv_from, -1);
        EXPECT_EQ(s.result_send_count, n - 1)
            << "n=" << n << ": participant 0 did not tell everyone";
      } else {
        EXPECT_GE(s.reduce_send_to, 0) << "n=" << n << " participant " << i
                                       << " never handed its status up";
        EXPECT_EQ(s.result_recv_from, 0);
        EXPECT_EQ(s.result_send_count, 0);
      }
    }
  }
}

TEST(EpochSchedule, EveryStatusReachesParticipantZero) {
  // Following reduce_send_to from any participant must arrive at 0 in a finite
  // number of hops. A cycle here would deadlock; a chain that stops short
  // would silently lose one rank's status, so a failed rank would go unnoticed
  // -- which is worse than a hang, because the run would continue.
  for (int n = 2; n <= 16; ++n) {
    for (int i = 1; i < n; ++i) {
      int at = i;
      int hops = 0;
      std::set<int> seen{at};
      while (at != 0) {
        const auto s = epoch_schedule(n, at);
        ASSERT_GE(s.reduce_send_to, 0) << "n=" << n << " chain from " << i
                                       << " dead-ends at " << at;
        at = s.reduce_send_to;
        ASSERT_TRUE(seen.insert(at).second) << "n=" << n << " cycle via " << at;
        ASSERT_LT(++hops, n) << "n=" << n << " chain from " << i << " does not terminate";
      }
    }
  }
}

TEST(EpochSchedule, CostsLogarithmicallyManyMessages) {
  // Not a performance claim -- a structural one. A linear reduce would also
  // pass the tests above while making an agreement point cost O(ranks)
  // messages, which is the objection to using a global barrier in the first
  // place.
  for (int n = 2; n <= 16; ++n) {
    for (int i = 0; i < n; ++i) {
      const auto s = epoch_schedule(n, i);
      if (i == 0) continue;  // participant 0 broadcasts, so it is O(n) by design
      EXPECT_LE(s.reduce_recv_count, 4) << "n=" << n << " participant " << i;
    }
  }
}

TEST(EpochSchedule, ANonParticipantDoesNothing) {
  // A rank outside the exchange must not join the agreement: joining makes the
  // others wait for a status it has no business reporting.
  for (const auto [n, i] : std::vector<std::pair<int, int>>{{4, 4}, {4, -1}, {1, 0}, {0, 0}}) {
    const auto s = epoch_schedule(n, i);
    EXPECT_EQ(s.reduce_recv_count, 0) << "n=" << n << " index=" << i;
    EXPECT_EQ(s.reduce_send_to, -1);
    EXPECT_EQ(s.result_recv_from, -1);
    EXPECT_EQ(s.result_send_count, 0);
  }
}

TEST(EpochSchedule, IsAllocationFree) {
  // The runtime's standing property is that a running plan touches no
  // allocator, and this is built once per agreement point per step. A first
  // version returned std::vectors and cost eight allocations per period on a
  // two-component job -- caught only by a parallel test on the host, so the
  // property is pinned here as well, where it is cheap to check.
  static_assert(std::is_trivially_copyable_v<fuse::EpochSchedule>,
                "EpochSchedule must hold no owning member");
  static_assert(sizeof(fuse::EpochSchedule) <= 160,
                "EpochSchedule grew; it is built per agreement point per step");
  SUCCEED();
}
