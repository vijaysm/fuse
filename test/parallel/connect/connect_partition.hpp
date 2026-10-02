#pragma once

// Pure, MPI-free: the population-balanced partition connect_common.hpp's
// ConnectTestBackend uses to decompose the producer's entities over a
// component's ranks, factored out into its own header (no <mpi.h>, no
// fuse/runtime/runner.hpp) so connect_golden.cpp -- a tier-0 tool run by a
// human, never under mpiexec -- can compute the EXACT SAME partition a real
// host run will produce without linking anything MPI-shaped at all.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "fuse/model/component.hpp"

namespace fuse_connect_test {

/// Verbatim from test/support/partitioned_test_backend.hpp's own block():
/// a population-balanced, position-in-set (not offset-from-first-rank)
/// contiguous slice of the first `n` global ids, stored in reverse local
/// order for the same CommGraph-matching reason that file documents.
[[nodiscard]] inline std::vector<std::size_t> partition_block(fuse::RankSet rs,
                                                              int me,
                                                              std::size_t n) {
  int pos = 0;
  for (int r = rs.min_rank(); r < me; ++r) pos += rs.contains(r) ? 1 : 0;
  const std::size_t base = n / static_cast<std::size_t>(static_cast<int>(rs.size()));
  const std::size_t rem  = n % static_cast<std::size_t>(static_cast<int>(rs.size()));
  const std::size_t begin =
      static_cast<std::size_t>(pos) * base + std::min<std::size_t>(pos, rem);
  const std::size_t count = base + (static_cast<std::size_t>(pos) < rem ? 1 : 0);
  std::vector<std::size_t> gids(count);
  for (std::size_t k = 0; k < count; ++k) gids[k] = begin + (count - 1 - k);
  return gids;
}

}  // namespace fuse_connect_test
