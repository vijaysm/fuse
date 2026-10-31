// Tier-0, MPI-free golden generator for test/parallel/connect's
// identity field transfer. Run BY A HUMAN (same convention as
// tools/src/write_golden_descriptor.cpp), not by a test:
//
//   fuse_parallel_connect_golden --na 2 --nb 3 > test/parallel/connect/golden-2+3.txt
//   fuse_parallel_connect_golden --chain 3 > test/parallel/connect/golden-chain-3.txt
//
// WHY THIS NEEDS NO MPI AND NO IN-PROCESS RUNNER RUN AT ALL. "a"'s
// ScalarComponent (connect_common.hpp) writes, for every entity it owns,
// that entity's own global id as its value -- so after an identity
// transfer, whatever value "b" holds at a given local index is exactly
// that index's global id. Which global ids land at which (rank, local
// index) of "b" is `partition_block()`, a pure function of na/nb with no
// MPI in it at all (connect_common.hpp's own comment on why it is a free
// function). The `--na`/`--nb` mode enumerates every rank of "b" (merged
// ranks [na, na+nb)) and prints each one's partition_block() result in
// the EXACT format run_participant() prints on a real host run, so
// `diff` is the whole comparison.
//
// `--chain <count>`: the N-way chain's own golden, which `--na`/`--nb`
// cannot produce -- the chain's producer "a" and the pair test's
// ScalarComponent do NOT write the same thing, so golden-1+1.txt cannot
// be reused for every chain length. The chain's own kEntities is 4
// (connect_chain_common.hpp's own constant), not the pair test's `na`,
// and the chain's LAST component sits at merged rank `count - 1`
// (component index order is preserved by the merge: a 3-component
// chain's last participant reports rank=2, not the pair golden's
// hardcoded rank=1), not always rank 1.
// `--chain` reproduces BOTH: "a" is a ONE-rank component owning the
// WHOLE `kEntities`-sized mesh (partition_block() with a single-rank
// RankSet is the identity split, so this is `partition_block({0,1}, 0,
// kEntities)`), and every later link is a single-rank IDENTITY
// passthrough (connect_chain_common.hpp's own ChainLinkComponent), so
// the values that reach the last component are EXACTLY "a"'s own,
// unpermuted, regardless of how many links separate them.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "connect_partition.hpp"

namespace {

constexpr int kChainCount = 2;  // run_pair.sh / connect_common.hpp
// connect_chain_common.hpp's own kEntities constant, duplicated here
// (not included: connect_chain_common.hpp pulls in <mpi.h>, fuse/
// runtime/runner.hpp and fuse/transport/connect.hpp, none of which this
// tier-0, MPI-free tool may depend on). Kept in sync by inspection, not
// by a shared header, because factoring it out would mean giving this
// tiny constant its own header for one integer -- if
// connect_chain_common.hpp's own kEntities changes, this one must change
// with it, or this generator silently produces a golden for the WRONG
// entity count.
constexpr std::size_t kChainEntities = 4;

void PrintPairGolden(int na, int nb) {
  const fuse::RankSet b_ranks{na, nb};
  for (const int rank : b_ranks) {
    const auto gids = fuse_connect_test::partition_block(
        b_ranks, rank, static_cast<std::size_t>(na));
    for (std::size_t i = 0; i < gids.size(); ++i) {
      std::printf("rank=%d i=%zu value=%a\n", rank, i, static_cast<double>(gids[i]));
    }
  }
}

void PrintChainGolden(int count) {
  // "a" is always component 0, a one-rank placement {0, 1}: with a
  // single-rank RankSet, partition_block() is the identity over the
  // whole kChainEntities range (own file header, and connect_chain_
  // common.hpp's own header comment on why "partition_block() at every
  // hop is the identity"). The LAST component sits at merged rank
  // `count - 1` -- component order is preserved end to end by the merge
  // (this file's own header comment).
  const fuse::RankSet a_ranks{0, 1};
  const auto gids = fuse_connect_test::partition_block(a_ranks, /*me=*/0, kChainEntities);
  const int last_rank = count - 1;
  for (std::size_t i = 0; i < gids.size(); ++i) {
    std::printf("rank=%d i=%zu value=%a\n", last_rank, i, static_cast<double>(gids[i]));
  }
}

}  // namespace

int main(int argc, char** argv) {
  int na = 0, nb = 0, chain = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string{}; };
    if (arg == "--na") na = std::atoi(next().c_str());
    else if (arg == "--nb") nb = std::atoi(next().c_str());
    else if (arg == "--chain") chain = std::atoi(next().c_str());
  }
  if (chain > 0) {
    if (chain < kChainCount) {
      std::fprintf(stderr, "usage: fuse_parallel_connect_golden --chain N (N >= 2)\n");
      return 2;
    }
    PrintChainGolden(chain);
    return 0;
  }
  if (na <= 0 || nb <= 0) {
    std::fprintf(stderr,
                 "usage: fuse_parallel_connect_golden --na N --nb N | --chain N\n");
    return 2;
  }
  PrintPairGolden(na, nb);
  return 0;
}
