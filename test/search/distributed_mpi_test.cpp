// test/search/distributed_mpi_test.cpp -- fuse::search's distributed point
// query under a REAL mpiexec.
//
// NOT YET RUN: this sandbox cannot initialize MPI (docs/dev/testing.md),
// exactly the posture test/partition/rcb_mpi_test.cpp documents for itself.
// This file is registered only in a Tier 1 build (test/CMakeLists.txt gates
// it on FUSE_ENABLE_MPI) and compiled here; its verdict comes from a host
// run at 1 to 5 ranks.
//
// THE ESCAPE HATCH THIS USES (rcb_mpi_test.cpp's own phrase): each rank owns
// a fixed, disjoint slice of one global box set (contiguous blocks by
// global index, the simplest partition that still puts different boxes on
// different ranks) and queries the SAME global set of points every rank
// also holds. The distributed result, gathered back into one flat list per
// point, must equal the SERIAL result (comm == kNullComm) of building one
// `BoxIndex` over the WHOLE gathered box set and querying it directly --
// except that a distributed hit's `index` is LOCAL to its owning rank
// (distributed.hpp's own doc), so the comparison maps each distributed hit
// to the GLOBAL box id its (rank, local index) pair names before comparing.

#include "fuse/search/distributed.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "fuse/search/box_index.hpp"
#include "fuse_test_main.hpp"

namespace fuse::search {
namespace {

// A fixed GLOBAL box set and query point set, identical content on every
// rank regardless of `size` -- only the OWNERSHIP split (below) depends on
// `size`. Deliberately not huge: this is a correctness check, not a
// scaling benchmark.
std::vector<Box> global_boxes() { return testing::random_boxes(/*dim=*/3, 40); }
std::vector<Point> global_points() { return testing::random_points(/*dim=*/3, 25); }

// Rank `r` of `size` owns global boxes
// `[r * n / size, (r + 1) * n / size)` -- contiguous blocks, the same
// scheme rcb_oracle.hpp's MPI comparison emulates for partition.
std::vector<std::size_t> owned_range(std::size_t n, int rank, int size) {
  const std::size_t lo = (static_cast<std::size_t>(rank) * n) / static_cast<std::size_t>(size);
  const std::size_t hi =
      (static_cast<std::size_t>(rank + 1) * n) / static_cast<std::size_t>(size);
  std::vector<std::size_t> ids(hi - lo);
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = lo + i;
  return ids;
}

TEST(SearchDistributedMpi, MatchesSerialOnTheGatheredInput) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto boxes = global_boxes();
  const auto points = global_points();
  const auto my_global_ids = owned_range(boxes.size(), rank, size);

  std::vector<Box> local_boxes;
  local_boxes.reserve(my_global_ids.size());
  for (std::size_t gid : my_global_ids) local_boxes.push_back(boxes[gid]);

  const auto dist =
      query_points_distributed(MPI_Comm_c2f(MPI_COMM_WORLD), local_boxes, points);
  ASSERT_TRUE(dist.ok()) << dist.error().message;
  ASSERT_EQ(dist.value().offsets.size(), points.size() + 1);

  // The reference: one serial index over every box, queried once, on every
  // rank (cheap enough at this size, and it means no rank depends on
  // gathering the others' partial answers just to build its own
  // expectation).
  auto serial = BoxIndex::build(boxes);
  ASSERT_TRUE(serial.ok()) << serial.error().message;
  const auto serial_qr = serial.value().query_points(points);
  ASSERT_TRUE(serial_qr.ok());

  // Rank r's local box index k corresponds to global id owned_range(...)[k]
  // -- computed once per rank so the distributed hit (rank, local index) can
  // be translated back to a global id for comparison.
  std::vector<std::vector<std::size_t>> global_ids_by_rank(static_cast<std::size_t>(size));
  for (int r = 0; r < size; ++r) global_ids_by_rank[static_cast<std::size_t>(r)] =
      owned_range(boxes.size(), r, size);

  for (std::size_t q = 0; q < points.size(); ++q) {
    std::vector<std::size_t> got_global;
    for (std::size_t j = dist.value().offsets[q]; j < dist.value().offsets[q + 1]; ++j) {
      const PointHit& hit = dist.value().hits[j];
      ASSERT_GE(hit.rank, 0);
      ASSERT_LT(static_cast<std::size_t>(hit.rank), global_ids_by_rank.size());
      const auto& ids = global_ids_by_rank[static_cast<std::size_t>(hit.rank)];
      ASSERT_LT(hit.index, ids.size());
      got_global.push_back(ids[hit.index]);
    }
    std::sort(got_global.begin(), got_global.end());

    std::vector<std::size_t> expected(
        serial_qr.value().indices.begin() +
            static_cast<std::ptrdiff_t>(serial_qr.value().offsets[q]),
        serial_qr.value().indices.begin() +
            static_cast<std::ptrdiff_t>(serial_qr.value().offsets[q + 1]));

    EXPECT_EQ(got_global, expected) << "query point " << q << " at rank " << rank;
  }
}

// Determinism: hits within one query, sorted by
// (rank, index) -- checked directly, independent of what the values are.
TEST(SearchDistributedMpi, HitsWithinOneQuerySortedByRankThenIndex) {
  const auto boxes = global_boxes();
  const auto points = global_points();
  const auto dist =
      query_points_distributed(MPI_Comm_c2f(MPI_COMM_WORLD), boxes, points);
  ASSERT_TRUE(dist.ok()) << dist.error().message;
  for (std::size_t q = 0; q < points.size(); ++q) {
    for (std::size_t j = dist.value().offsets[q]; j + 1 < dist.value().offsets[q + 1]; ++j) {
      const PointHit& a = dist.value().hits[j];
      const PointHit& b = dist.value().hits[j + 1];
      const bool ordered = (a.rank < b.rank) || (a.rank == b.rank && a.index < b.index);
      EXPECT_TRUE(ordered) << "query " << q << " hit " << j;
    }
  }
}

// Rank 0 alone submits a mixed-dimension local_boxes array; every
// other rank's own local_boxes are perfectly well-formed. Rank 0 must not
// refuse locally and return while its peers enter
// ArborX::DistributedTree's collective construction -- a hang that
// ctest's TIMEOUT (not a passing/failing assertion) would be the only thing
// to report. This test asserts the refusal itself, which is
// only observable once every rank returns it.
TEST(SearchDistributedMpi, RefusesWhenOnlyOneRankSubmitsMixedDimensionBoxes) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Box> local_boxes;
  if (rank == 0) {
    local_boxes = {
        Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 0.0}},  // 2D
        Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}},  // 3D
    };
  } else {
    local_boxes = testing::random_boxes(/*dim=*/3, 5);
  }
  const std::vector<Point> points = {Point{0.5, 0.5, 0.5}};

  const auto r = query_points_distributed(MPI_Comm_c2f(MPI_COMM_WORLD), local_boxes, points);
  ASSERT_FALSE(r.ok()) << "every rank must refuse together, including ranks whose own "
                          "local_boxes were fine, rank "
                       << rank;
  EXPECT_EQ(r.error().code, "E-SEARCH-MIXED-DIM");
}

// No single rank's own local_boxes are internally mixed-dimension,
// but rank 0's are uniformly 2D while every other rank's are uniformly
// 3D -- a cross-rank disagreement classify_local alone cannot see.
// Accepting this silently would produce hits the caller
// has no consistent way to interpret.
TEST(SearchDistributedMpi, RefusesWhenRanksDisagreeAboutDimension) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<Box> local_boxes;
  if (rank == 0) {
    local_boxes = {Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 0.0}}};  // 2D
  } else {
    local_boxes = {Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}}};  // 3D
  }
  const std::vector<Point> points = {Point{0.5, 0.5, 0.0}};

  const auto r = query_points_distributed(MPI_Comm_c2f(MPI_COMM_WORLD), local_boxes, points);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SEARCH-MIXED-DIM");
}

}  // namespace
}  // namespace fuse::search
