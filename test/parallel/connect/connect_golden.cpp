// Tier-0, MPI-free golden generator for test/parallel/connect's
// identity field transfer. Run BY A HUMAN (same convention as
// tools/src/write_golden_descriptor.cpp), not by a test:
//
//   fuse_parallel_connect_golden --na 2 --nb 3 > test/parallel/connect/golden-2+3.txt
//
// WHY THIS NEEDS NO MPI AND NO IN-PROCESS RUNNER RUN AT ALL. "a"'s
// ScalarComponent (connect_common.hpp) writes, for every entity it owns,
// that entity's own global id as its value -- so after an identity
// transfer, whatever value "b" holds at a given local index is exactly
// that index's global id. Which global ids land at which (rank, local
// index) of "b" is `partition_block()`, a pure function of na/nb with no
// MPI in it at all (connect_common.hpp's own comment on why it is a free
// function). This tool enumerates every rank of "b" (merged ranks
// [na, na+nb)) and prints each one's partition_block() result in the
// EXACT format run_participant() prints on a real host run, so `diff` is
// the whole comparison.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "connect_partition.hpp"

int main(int argc, char** argv) {
  int na = 0, nb = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
    if (arg == "--na") na = std::atoi(next().c_str());
    else if (arg == "--nb") nb = std::atoi(next().c_str());
  }
  if (na <= 0 || nb <= 0) {
    std::fprintf(stderr, "usage: fuse_parallel_connect_golden --na N --nb N\n");
    return 2;
  }
  const fuse::RankSet b_ranks{na, nb};
  for (const int rank : b_ranks) {
    const auto gids = fuse_connect_test::partition_block(
        b_ranks, rank, static_cast<std::size_t>(na));
    for (std::size_t i = 0; i < gids.size(); ++i) {
      std::printf("rank=%d i=%zu value=%a\n", rank, i, static_cast<double>(gids[i]));
    }
  }
  return 0;
}
