// src/search/distributed.cpp -- the only translation unit in fuse::search
// that names mpi.h or ArborX::DistributedTree (distributed.hpp's header
// comment: the same isolation fuse::partition::detail::collectives and
// fuse::backend's native_migration.cpp use for their own MPI symbols).

#include "fuse/search/distributed.hpp"

#include <algorithm>

#include <fmt/format.h>

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"
#include "fuse/search/box_index.hpp"
#include "fuse/search/geometry.hpp"

#if FUSE_HAVE_MPI
#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <mpi.h>
#endif

namespace fuse::search {

namespace {

/// Shared with BoxIndex::build: every rank validates its OWN local_boxes
/// before any collective touches them, so a badly-formed local input on one
/// rank never reaches MPI at all.
Result<int> classify_local(const std::vector<Box>& boxes) {
  if (boxes.empty()) return 0;
  const int dim0 = classify_dim(boxes[0]);
  for (std::size_t i = 1; i < boxes.size(); ++i) {
    const int di = classify_dim(boxes[i]);
    if (di != dim0) {
      return Failure{"E-SEARCH-MIXED-DIM",
                     fmt::format("local box 0 classifies as {}D but local box {} classifies as "
                                 "{}D; a spatial index cannot mix dimensionalities in one build",
                                 dim0, i, di)};
    }
  }
  return dim0;
}

}  // namespace

#if FUSE_HAVE_MPI

namespace {

using Coordinate = double;
using ArborBox = ArborX::Box<3, Coordinate>;
using ArborPoint = ArborX::Point<3, Coordinate>;
using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using Value = ArborX::PairValueIndex<ArborBox, int>;

/// Runs on the rank that OWNS the matched box (ArborX::DistributedTree
/// invokes the callback locally, where the data lives, then ships the
/// result back to the querying rank) -- exactly the
/// examples/distributed_tree/distributed_knn.cpp pattern this mirrors, with
/// intersects predicates in place of that example's nearest ones.
struct TagWithRank {
  int rank;
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const&, Value const& value,
                                  OutputFunctor const& out) const {
    out(PointHit{rank, static_cast<std::size_t>(value.index)});
  }
};

}  // namespace

Result<DistributedQueryResult> query_points_distributed(CommHandle comm,
                                                        std::vector<Box> local_boxes,
                                                        const std::vector<Point>& points) {
  auto dim = classify_local(local_boxes);
  if (!dim.ok()) return dim.error();

  DistributedQueryResult empty_result;
  empty_result.offsets.assign(points.size() + 1, 0);

  if (comm == kNullComm) {
    // One participant: answered locally, no MPI symbol touched, every hit
    // tagged rank 0 -- what a single-rank caller of the distributed API
    // should see (fuse::partition's own kNullComm convention).
    auto local = BoxIndex::build(std::move(local_boxes));
    if (!local.ok()) return local.error();
    auto local_result = local.value().query_points(points);
    if (!local_result.ok()) return local_result.error();
    DistributedQueryResult out;
    out.offsets = local_result.value().offsets;
    out.hits.reserve(local_result.value().indices.size());
    for (std::size_t idx : local_result.value().indices) out.hits.push_back(PointHit{0, idx});
    return out;
  }

  ExecutionSpace space{};
  MPI_Comm mpi_comm = MPI_Comm_f2c(comm);
  int my_rank = 0;
  MPI_Comm_rank(mpi_comm, &my_rank);

  Kokkos::View<ArborBox*, MemorySpace> arbor_boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::dist_boxes"),
      local_boxes.size());
  for (std::size_t i = 0; i < local_boxes.size(); ++i) {
    const Box& b = local_boxes[i];
    arbor_boxes(i) = ArborBox{ArborPoint{b.lo.x, b.lo.y, b.lo.z}, ArborPoint{b.hi.x, b.hi.y, b.hi.z}};
  }

  ArborX::DistributedTree<MemorySpace, Value> tree(
      mpi_comm, space, ArborX::Experimental::attach_indices<int>(arbor_boxes));

  Kokkos::View<decltype(ArborX::intersects(ArborPoint{}))*, MemorySpace> preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::dist_preds"), points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Point& p = points[i];
    preds(i) = ArborX::intersects(ArborPoint{p.x, p.y, p.z});
  }

  Kokkos::View<PointHit*, MemorySpace> values("fuse::search::dist_values", 0);
  Kokkos::View<int*, MemorySpace> offsets("fuse::search::dist_offsets", 0);
  tree.query(space, preds, TagWithRank{my_rank}, values, offsets);

  DistributedQueryResult out;
  out.offsets.assign(offsets.data(), offsets.data() + offsets.extent(0));
  out.hits.assign(values.data(), values.data() + values.extent(0));
  for (std::size_t q = 0; q + 1 < out.offsets.size(); ++q) {
    std::sort(out.hits.begin() + static_cast<std::ptrdiff_t>(out.offsets[q]),
              out.hits.begin() + static_cast<std::ptrdiff_t>(out.offsets[q + 1]),
              [](const PointHit& a, const PointHit& b) {
                return a.rank != b.rank ? a.rank < b.rank : a.index < b.index;
              });
  }
  return out;
}

#else  // !FUSE_HAVE_MPI

Result<DistributedQueryResult> query_points_distributed(CommHandle comm,
                                                        std::vector<Box> local_boxes,
                                                        const std::vector<Point>& points) {
  auto dim = classify_local(local_boxes);
  if (!dim.ok()) return dim.error();

  if (comm != kNullComm) {
    return Failure{"E-SEARCH-NO-MPI",
                   "a non-null communicator was supplied but this build has no MPI "
                   "(FUSE_HAVE_MPI=0); rebuild with -DFUSE_ENABLE_MPI=ON"};
  }

  auto local = BoxIndex::build(std::move(local_boxes));
  if (!local.ok()) return local.error();
  auto local_result = local.value().query_points(points);
  if (!local_result.ok()) return local_result.error();
  DistributedQueryResult out;
  out.offsets = local_result.value().offsets;
  out.hits.reserve(local_result.value().indices.size());
  for (std::size_t idx : local_result.value().indices) out.hits.push_back(PointHit{0, idx});
  return out;
}

#endif  // FUSE_HAVE_MPI

}  // namespace fuse::search
