#pragma once

#include <cstddef>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/search/types.hpp"

/// The distributed variant of `BoxIndex`'s point query
/// (`ArborX::DistributedTree`), isolated in its own translation unit
/// (src/search/distributed.cpp) so `mpi.h` never reaches a public header --
/// the same isolation `fuse::partition::detail::collectives` and
/// `fuse::backend`'s `native_migration.cpp` use.
namespace fuse::search {

/// One rank's containing box for one query point. `rank` is the MPI rank
/// that OWNS `index`; `index` is a position in THAT rank's own `local_boxes`
/// array (the same numbering `query_points_distributed`'s caller on `rank`
/// passed in) -- never a global id, exactly like `BoxIndex`'s own indices
/// are never global.
struct PointHit {
  int rank = 0;
  std::size_t index = 0;
};

/// CSR-style batched result: query `q`'s hits are
/// `hits[offsets[q] .. offsets[q + 1])`, sorted ascending by `(rank, index)`
/// -- the ordering the task's determinism requirement names explicitly, so
/// downstream code never depends on which rank's ArborX traversal happened
/// to answer first.
struct DistributedQueryResult {
  std::vector<std::size_t> offsets;  ///< size == number of queries + 1
  std::vector<PointHit> hits;        ///< size == offsets.back()
};

/// For every point in `points` (this rank's own query points), every box --
/// on ANY rank in `comm` -- whose `local_boxes` entry contains it. Same
/// containment rule as `BoxIndex::query_points`: `local_boxes` are used
/// EXACTLY as given, so a caller wanting the shared-face tolerance pads them
/// first (`geometry.hpp::padded`, or `element_boxes` which already does).
///
/// COLLECTIVE over `comm` when `comm != kNullComm`: every rank must call
/// this, even one with zero local boxes or zero local query points
/// (`fuse::partition`'s "Global collective discipline",
/// docs/dev/partitioning.md) -- skipping the call on such a rank because it
/// locally has nothing to contribute is exactly the bug that discipline
/// exists to prevent.
///
/// `comm == kNullComm` (one participant) needs no MPI at all: it is answered
/// locally, and every hit carries `rank == 0`, matching what a single-rank
/// caller of the distributed API should see.
///
/// Errors, never throws:
///   * `E-SEARCH-MIXED-DIM` -- this rank's `local_boxes` fail the same
///     per-box dimension check `BoxIndex::build` applies (checked locally
///     before any collective, so a badly-formed local input never reaches
///     one).
///   * `E-SEARCH-NO-MPI` -- `comm != kNullComm` on a build with no MPI
///     (`FUSE_HAVE_MPI == 0`); mirrors `fuse::partition`'s `E-PART-NO-MPI`.
[[nodiscard]] Result<DistributedQueryResult> query_points_distributed(
    CommHandle comm, std::vector<Box> local_boxes, const std::vector<Point>& points);

}  // namespace fuse::search
