// src/search/distributed.cpp -- the only translation unit in fuse::search
// that names mpi.h or ArborX::DistributedTree (distributed.hpp's header
// comment: the same isolation fuse::partition::detail::collectives and
// fuse::backend's native_migration.cpp use for their own MPI symbols).

#include "fuse/search/distributed.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

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
using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using DeviceExecutionSpace = Kokkos::DefaultExecutionSpace;
using Value = ArborX::PairValueIndex<ArborBox, int>;

/// True only when this build's default execution space is a real device --
/// see box_index.cpp's identical constant.
inline constexpr bool kHasDeviceBackend =
    !std::is_same_v<DeviceExecutionSpace, HostExecutionSpace>;

/// Runs on the rank that OWNS the matched box (ArborX::DistributedTree
/// invokes the callback locally, where the data lives, then ships the
/// result back to the querying rank) -- exactly the
/// examples/distributed_tree/distributed_knn.cpp pattern this mirrors, with
/// intersects predicates in place of that example's nearest ones. `Value`
/// names no space, so the same callback serves a host or a device tree.
struct TagWithRank {
  int rank;
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const&, Value const& value,
                                  OutputFunctor const& out) const {
    out(PointHit{rank, static_cast<std::size_t>(value.index)});
  }
};

/// The whole distributed query, templated on the execution space it runs
/// on: `HostExecutionSpace` is byte-identical to this function's behaviour
/// before `fuse::search::ExecutionSpace` existed; `DeviceExecutionSpace`
/// stages `local_boxes`/`points` to `Space::memory_space` first (the same
/// "existing std::vector overload, copying to/from the device inside"
/// contract `BoxIndex::build` honours) and copies the result back to host
/// memory to assemble the `std::vector`-based `DistributedQueryResult`.
template <class Space>
DistributedQueryResult run_distributed_query(MPI_Comm mpi_comm, int my_rank,
                                             const std::vector<Box>& local_boxes,
                                             const std::vector<Point>& points) {
  using MemorySpace = typename Space::memory_space;
  using Tree = ArborX::DistributedTree<MemorySpace, Value>;

  Space space{};

  Kokkos::View<ArborBox*, Kokkos::HostSpace> host_boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::dist_boxes_host"),
      local_boxes.size());
  for (std::size_t i = 0; i < local_boxes.size(); ++i) {
    const Box& b = local_boxes[i];
    host_boxes(i) = ArborBox{ArborPoint{b.lo.x, b.lo.y, b.lo.z}, ArborPoint{b.hi.x, b.hi.y, b.hi.z}};
  }
  Kokkos::View<ArborBox*, MemorySpace> arbor_boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::dist_boxes"),
      local_boxes.size());
  Kokkos::deep_copy(arbor_boxes, host_boxes);

  Kokkos::View<decltype(ArborX::intersects(ArborPoint{}))*, Kokkos::HostSpace> host_preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::dist_preds_host"),
      points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const Point& p = points[i];
    host_preds(i) = ArborX::intersects(ArborPoint{p.x, p.y, p.z});
  }
  Kokkos::View<decltype(ArborX::intersects(ArborPoint{}))*, MemorySpace> preds(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::dist_preds"), points.size());
  Kokkos::deep_copy(preds, host_preds);

  Tree tree(mpi_comm, space, ArborX::Experimental::attach_indices<int>(arbor_boxes));

  Kokkos::View<PointHit*, MemorySpace> values("fuse::search::dist_values", 0);
  Kokkos::View<int*, MemorySpace> offsets("fuse::search::dist_offsets", 0);
  tree.query(space, preds, TagWithRank{my_rank}, values, offsets);

  Kokkos::View<PointHit*, Kokkos::HostSpace> host_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::dist_values_host"),
      values.extent(0));
  Kokkos::View<int*, Kokkos::HostSpace> host_offsets(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "fuse::search::dist_offsets_host"),
      offsets.extent(0));
  Kokkos::deep_copy(host_values, values);
  Kokkos::deep_copy(host_offsets, offsets);

  DistributedQueryResult out;
  out.offsets.assign(host_offsets.data(), host_offsets.data() + host_offsets.extent(0));
  out.hits.assign(host_values.data(), host_values.data() + host_values.extent(0));
  for (std::size_t q = 0; q + 1 < out.offsets.size(); ++q) {
    std::sort(out.hits.begin() + static_cast<std::ptrdiff_t>(out.offsets[q]),
              out.hits.begin() + static_cast<std::ptrdiff_t>(out.offsets[q + 1]),
              [](const PointHit& a, const PointHit& b) {
                return a.rank != b.rank ? a.rank < b.rank : a.index < b.index;
              });
  }
  return out;
}

}  // namespace

Result<DistributedQueryResult> query_points_distributed(CommHandle comm,
                                                        std::vector<Box> local_boxes,
                                                        const std::vector<Point>& points,
                                                        ExecutionSpace space) {
  const Result<ExecutionSpace> resolved = resolve_execution_space(space);
  if (!resolved.ok()) return resolved.error();
  const bool use_device = resolved.value() == ExecutionSpace::Device && kHasDeviceBackend;

  DistributedQueryResult empty_result;
  empty_result.offsets.assign(points.size() + 1, 0);

  if (comm == kNullComm) {
    // One participant: no collective is possible or needed, so the
    // rank-local classification is both necessary and sufficient -- see
    // classify_local's own comment. Answered locally, no MPI symbol
    // touched, every hit tagged rank 0 -- what a single-rank caller of the
    // distributed API should see (fuse::partition's own kNullComm
    // convention). The local BoxIndex honours the same resolved execution
    // space.
    auto dim = classify_local(local_boxes);
    if (!dim.ok()) return dim.error();
    auto local = BoxIndex::build(std::move(local_boxes), resolved.value());
    if (!local.ok()) return local.error();
    auto local_result = local.value().query_points(points);
    if (!local_result.ok()) return local_result.error();
    DistributedQueryResult out;
    out.offsets = local_result.value().offsets;
    out.hits.reserve(local_result.value().indices.size());
    for (std::size_t idx : local_result.value().indices) out.hits.push_back(PointHit{0, idx});
    return out;
  }

  MPI_Comm mpi_comm = MPI_Comm_f2c(comm);
  int my_rank = 0;
  MPI_Comm_rank(mpi_comm, &my_rank);

  // Reconcile the rank-local dimension check across the WHOLE
  // communicator BEFORE any rank reaches a collective -- an
  // `MPI_Allreduce(MIN)` of a locally-computed validity flag, mirroring
  // `fuse::native::migrate`'s agree-before-exchange recipe (see
  // distributed.hpp's header comment and docs/dev/rank-agreement.md). A
  // rank whose own local_boxes fail classify_local must never refuse alone
  // while its peers proceed into DistributedTree construction, which
  // `run_distributed_query` (above) performs unconditionally.
  auto local_dim = classify_local(local_boxes);
  const int locally_ok = local_dim.ok() ? 1 : 0;
  int globally_ok = 0;
  MPI_Allreduce(&locally_ok, &globally_ok, 1, MPI_INT, MPI_MIN, mpi_comm);
  if (globally_ok == 0) {
    if (!local_dim.ok()) return local_dim.error();
    return Failure{"E-SEARCH-MIXED-DIM",
                   fmt::format("rank {} is proceeding, but at least one other rank in this "
                               "communicator refused its own local_boxes' dimension check; "
                               "every rank refuses together so none hangs waiting on the "
                               "DistributedTree build",
                               my_rank)};
  }

  // Every rank's own boxes agree with themselves; agree them
  // with each other next. A rank with zero local_boxes classifies as 0 (no
  // opinion, matching BoxIndex's own empty-index convention) and must not
  // be read as "this rank votes for 0D" -- it is excluded from the
  // MIN by substituting the largest representable value for it.
  const std::int64_t dim_value = static_cast<std::int64_t>(local_dim.value());
  std::int64_t global_max_dim = 0;
  MPI_Allreduce(&dim_value, &global_max_dim, 1, MPI_INT64_T, MPI_MAX, mpi_comm);
  const std::int64_t dim_or_no_vote =
      (dim_value == 0) ? std::numeric_limits<std::int64_t>::max() : dim_value;
  std::int64_t global_min_voted_dim = 0;
  MPI_Allreduce(&dim_or_no_vote, &global_min_voted_dim, 1, MPI_INT64_T, MPI_MIN, mpi_comm);
  if (global_max_dim > 0 && global_min_voted_dim != global_max_dim) {
    return Failure{"E-SEARCH-MIXED-DIM",
                   fmt::format("ranks disagree about local_boxes' dimension (ranges from {} to "
                               "{} across ranks that submitted any boxes); a distributed "
                               "spatial index cannot mix dimensionalities across ranks",
                               global_min_voted_dim, global_max_dim)};
  }

  if (use_device) {
    return run_distributed_query<DeviceExecutionSpace>(mpi_comm, my_rank, local_boxes, points);
  }
  return run_distributed_query<HostExecutionSpace>(mpi_comm, my_rank, local_boxes, points);
}

#else  // !FUSE_HAVE_MPI

Result<DistributedQueryResult> query_points_distributed(CommHandle comm,
                                                        std::vector<Box> local_boxes,
                                                        const std::vector<Point>& points,
                                                        ExecutionSpace space) {
  auto dim = classify_local(local_boxes);
  if (!dim.ok()) return dim.error();

  if (comm != kNullComm) {
    return Failure{"E-SEARCH-NO-MPI",
                   "a non-null communicator was supplied but this build has no MPI "
                   "(FUSE_HAVE_MPI=0); rebuild with -DFUSE_ENABLE_MPI=ON"};
  }

  auto local = BoxIndex::build(std::move(local_boxes), space);
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
