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
/// `space` follows `BoxIndex::build`'s own convention
/// (`resolve_execution_space()`, types.hpp): `Host` (the default) builds and
/// queries `ArborX::DistributedTree` on `Kokkos::DefaultHostExecutionSpace`,
/// byte-identical to this function's behaviour before the parameter existed;
/// `Device` (explicit, or `FUSE_EXECUTION_SPACE=device`) builds and queries
/// on `Kokkos::DefaultExecutionSpace` and its `memory_space`, copying
/// `local_boxes`/`points` to the device and the result back exactly the way
/// `BoxIndex::build(std::vector<Box>, ExecutionSpace)` does for the local
/// tree. Ignored at `comm == kNullComm` only in the sense that there is no
/// MPI traffic to place on a space either way; the local `BoxIndex` it
/// delegates to still honours it.
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
///   * `E-SEARCH-MIXED-DIM` -- either this rank's own `local_boxes` fail
///     the same per-box dimension check `BoxIndex::build` applies, or this
///     rank's `local_boxes` agree with themselves but disagree in
///     dimension with another rank's `local_boxes`. Both are rank-local
///     computations, but NEITHER is returned on its own: each is reduced
///     with an `MPI_Allreduce` across the whole of `comm` (`MIN` of a
///     locally-computed validity flag for the first, `MAX`/`MIN` of the
///     agreed dimension for the second) BEFORE `ArborX::DistributedTree`
///     is constructed -- the same agree-before-collective recipe
///     `fuse::native::migrate` uses for its own per-rank rejections, so
///     every rank in `comm` returns the same verdict and none is left
///     waiting inside a collective a peer already refused to enter. A rank
///     with zero local boxes casts no vote in the cross-rank comparison
///     (it has nothing to disagree about), matching `BoxIndex`'s own
///     empty-index convention (`dimension() == 0`).
///   * `E-SEARCH-NO-MPI` -- `comm != kNullComm` on a build with no MPI
///     (`FUSE_HAVE_MPI == 0`); mirrors `fuse::partition`'s `E-PART-NO-MPI`.
///   * `E-SEARCH-SPACE` -- `space == ExecutionSpace::Default` and
///     `FUSE_EXECUTION_SPACE` names neither `host` nor `device`.
[[nodiscard]] Result<DistributedQueryResult> query_points_distributed(
    CommHandle comm, std::vector<Box> local_boxes, const std::vector<Point>& points,
    ExecutionSpace space = ExecutionSpace::Default);

}  // namespace fuse::search
