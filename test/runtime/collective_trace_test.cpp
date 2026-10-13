// KNOWN_ISSUES.md D57 ("Collectives" track). Pins the exact text
// FUSE_COLLECTIVE_TRACE=1 prints for one collective instruction's own
// execution (fuse::format_collective_trace_line(), include/fuse/runtime/
// collective_trace.hpp). No coroutine, no communicator, no graph: this is
// the pure formatter the Runner calls from run_region() (src/runtime/
// iteration.cpp), kept in its own translation unit for exactly this
// reason -- it is tier-0 testable with no MPI at all.
#include "fuse/runtime/collective_trace.hpp"

#include <gtest/gtest.h>

namespace {

TEST(CollectiveTrace, FormatsEveryFieldInTheDocumentedOrder) {
  const std::string line = fuse::format_collective_trace_line(
      /*rank=*/1, /*region=*/"u2v_window", /*period=*/3, /*sweep=*/5,
      /*instruction_id=*/42, /*kind=*/"Constrain", /*communicator=*/"base",
      /*ordinal=*/7);
  EXPECT_EQ(line,
            "FUSE_COLLECTIVE_TRACE rank=1 region=u2v_window period=3 sweep=5 "
            "instr=42 kind=Constrain comm=base ordinal=7");
}

// The one case a host diff actually keys on (KNOWN_ISSUES.md D57's own
// diagnosis): a rank with no base communicator configured prints rank=-1
// rather than some default-constructed MPI_COMM_WORLD rank, so a trace
// line can never be silently misread as "rank 0 agrees".
TEST(CollectiveTrace, NoCommunicatorPrintsRankMinusOne) {
  const std::string line = fuse::format_collective_trace_line(
      /*rank=*/-1, /*region=*/"solo", /*period=*/0, /*sweep=*/0,
      /*instruction_id=*/0, /*kind=*/"Constrain", /*communicator=*/"base",
      /*ordinal=*/1);
  EXPECT_NE(line.find("rank=-1"), std::string::npos);
}

// Two ranks diverging on their own Nth collective is exactly the signature
// this trace exists to surface -- two lines with the same ordinal but a
// different kind/region must not format as equal strings.
TEST(CollectiveTrace, SameOrdinalDifferentKindProducesDifferentLines) {
  const std::string rank_a = fuse::format_collective_trace_line(
      0, "u2v_window", 2, 0, 10, "Constrain", "base", 4);
  const std::string rank_b = fuse::format_collective_trace_line(
      1, "u2v_window", 3, 0, 0, "Constrain", "base", 4);
  EXPECT_NE(rank_a, rank_b);
}

}  // namespace
