// test/search/box_index_test.cpp — BoxIndex against a brute-force reference
// in 1D, 2D and 3D; shared faces/vertices; k-nearest; the empty index.

#include "fuse/search/box_index.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "fixtures.hpp"
#include "fuse/search/geometry.hpp"
#include "fuse_test_main.hpp"

namespace fuse::search {
namespace {

using testing::brute_force_contains;
using testing::is_valid_knn;
using testing::random_boxes;
using testing::random_points;

// ---------------------------------------------------------------- build / dimension / size

TEST(BoxIndex, EmptyIndexIsValidNotARefusal) {
  auto idx = BoxIndex::build({});
  ASSERT_TRUE(idx.ok()) << idx.error().message;
  EXPECT_TRUE(idx.value().empty());
  EXPECT_EQ(idx.value().size(), 0u);
  EXPECT_EQ(idx.value().dimension(), 0);

  // Querying an empty index is not a refusal either -- every query is
  // vacuously empty.
  auto pts = random_points(3, 5);
  auto qr = idx.value().query_points(pts);
  ASSERT_TRUE(qr.ok());
  EXPECT_EQ(qr.value().offsets.size(), pts.size() + 1);
  EXPECT_TRUE(qr.value().indices.empty());
  for (std::size_t off : qr.value().offsets) EXPECT_EQ(off, 0u);

  auto kr = idx.value().query_nearest(pts, 3);
  ASSERT_TRUE(kr.ok());
  EXPECT_TRUE(kr.value().indices.empty());
}

TEST(BoxIndex, DimensionAndSizeReportWhatWasBuilt) {
  for (int dim : {1, 2, 3}) {
    auto boxes = random_boxes(dim, 17);
    auto idx = BoxIndex::build(boxes);
    ASSERT_TRUE(idx.ok()) << idx.error().message;
    EXPECT_EQ(idx.value().dimension(), dim);
    EXPECT_EQ(idx.value().size(), boxes.size());
    EXPECT_FALSE(idx.value().empty());
  }
}

// ---------------------------------------------------------------- point queries vs brute force

TEST(BoxIndex, PointQueriesMatchBruteForceIn1D2D3D) {
  for (int dim : {1, 2, 3}) {
    SCOPED_TRACE(::testing::Message() << "dim=" << dim);
    auto boxes = random_boxes(dim, 200);
    auto pts = random_points(dim, 300);
    auto idx = BoxIndex::build(boxes);
    ASSERT_TRUE(idx.ok()) << idx.error().message;

    auto qr = idx.value().query_points(pts);
    ASSERT_TRUE(qr.ok());
    ASSERT_EQ(qr.value().offsets.size(), pts.size() + 1);

    std::size_t total_hits = 0;
    for (std::size_t q = 0; q < pts.size(); ++q) {
      const auto expected = brute_force_contains(boxes, pts[q]);
      std::vector<std::size_t> got(
          qr.value().indices.begin() + static_cast<std::ptrdiff_t>(qr.value().offsets[q]),
          qr.value().indices.begin() + static_cast<std::ptrdiff_t>(qr.value().offsets[q + 1]));
      EXPECT_EQ(got, expected) << "query point " << q;
      total_hits += expected.size();
    }
    // A sanity floor that the random fixture actually exercises some hits
    // (otherwise an empty-vs-empty comparison above would pass vacuously).
    EXPECT_GT(total_hits, 0u);
  }
}

TEST(BoxIndex, SharedFaceInTwoDimensionsFindsBothCells) {
  // Two unit squares sharing the edge x == 1, padded so a point exactly on
  // that edge is found in both -- geometry.hpp::padded is the tolerance
  // rule under test, BoxIndex itself adds none of its own.
  Tolerance tol;
  std::vector<Box> boxes = {
      padded(Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 0.0}}, tol),
      padded(Box{Point{1.0, 0.0, 0.0}, Point{2.0, 1.0, 0.0}}, tol),
  };
  auto idx = BoxIndex::build(boxes);
  ASSERT_TRUE(idx.ok()) << idx.error().message;

  auto qr = idx.value().query_points({Point{1.0, 0.5, 0.0}});
  ASSERT_TRUE(qr.ok());
  ASSERT_EQ(qr.value().offsets.size(), 2u);
  std::vector<std::size_t> hits(qr.value().indices.begin(), qr.value().indices.end());
  std::sort(hits.begin(), hits.end());
  EXPECT_EQ(hits, (std::vector<std::size_t>{0, 1}));
}

TEST(BoxIndex, SharedVertexInThreeDimensionsFindsAllFourCells) {
  // Four unit cubes meeting at the vertex (1, 1, 1), padded the same way.
  Tolerance tol;
  std::vector<Box> boxes;
  for (double dx : {0.0, 1.0}) {
    for (double dy : {0.0, 1.0}) {
      boxes.push_back(padded(
          Box{Point{dx, dy, 0.0}, Point{dx + 1.0, dy + 1.0, 2.0}}, tol));
    }
  }
  auto idx = BoxIndex::build(boxes);
  ASSERT_TRUE(idx.ok()) << idx.error().message;

  auto qr = idx.value().query_points({Point{1.0, 1.0, 1.0}});
  ASSERT_TRUE(qr.ok());
  ASSERT_EQ(qr.value().offsets.size(), 2u);
  std::vector<std::size_t> hits(qr.value().indices.begin(), qr.value().indices.end());
  std::sort(hits.begin(), hits.end());
  EXPECT_EQ(hits, (std::vector<std::size_t>{0, 1, 2, 3}));
}

TEST(BoxIndex, UnpaddedSharedFaceFindsOnlyOneCellByFloatingPointLuckOrBoth) {
  // Without padding, a point exactly on a shared face relies on `<=`/`>=`
  // (geometry.hpp::contains is closed on both ends), so it is legitimately
  // found in BOTH exact boxes too when the coordinate is exactly
  // representable -- this pins that `contains` is closed, not open, on its
  // boundary, which is what makes `padded` an ENLARGEMENT rather than the
  // only way to see a boundary point at all.
  std::vector<Box> boxes = {
      Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 0.0}},
      Box{Point{1.0, 0.0, 0.0}, Point{2.0, 1.0, 0.0}},
  };
  auto idx = BoxIndex::build(boxes);
  ASSERT_TRUE(idx.ok());
  auto qr = idx.value().query_points({Point{1.0, 0.5, 0.0}});
  ASSERT_TRUE(qr.ok());
  std::vector<std::size_t> hits(qr.value().indices.begin(), qr.value().indices.end());
  std::sort(hits.begin(), hits.end());
  EXPECT_EQ(hits, (std::vector<std::size_t>{0, 1}));
}

TEST(BoxIndex, PointJustOutsideTheExactBoxIsFoundOnlyThroughPadding) {
  // Unlike the two shared-face tests above, this point does NOT land on an
  // exactly-representable shared coordinate: it sits strictly outside the
  // ORIGINAL [0, 1] box by half the default tolerance's padding, so finding
  // it genuinely exercises `padded()`'s growth rather than `contains()`'s
  // closed boundary happening to catch it anyway (build-evidence/mutations/
  // search.toml's "tolerance dropped" row targets exactly this).
  const Box raw{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 0.0}};
  const Tolerance tol;
  const Box grown = padded(raw, tol);
  const double half_pad = 0.5 * (grown.hi.x - raw.hi.x);
  ASSERT_GT(half_pad, 0.0);

  auto idx = BoxIndex::build({grown});
  ASSERT_TRUE(idx.ok()) << idx.error().message;

  auto qr = idx.value().query_points({Point{1.0 + half_pad, 0.5, 0.0}});
  ASSERT_TRUE(qr.ok());
  ASSERT_EQ(qr.value().offsets.size(), 2u);
  EXPECT_EQ(qr.value().offsets[1] - qr.value().offsets[0], 1u)
      << "a point just outside the exact box, but inside its padded one, must be found";
}

// ---------------------------------------------------------------- k-nearest vs brute force

TEST(BoxIndex, KNearestMatchesBruteForceIn1D2D3D) {
  for (int dim : {1, 2, 3}) {
    SCOPED_TRACE(::testing::Message() << "dim=" << dim);
    auto boxes = random_boxes(dim, 150);
    auto pts = random_points(dim, 40);
    auto idx = BoxIndex::build(boxes);
    ASSERT_TRUE(idx.ok()) << idx.error().message;

    constexpr int k = 5;
    auto kr = idx.value().query_nearest(pts, k);
    ASSERT_TRUE(kr.ok()) << kr.error().message;
    ASSERT_EQ(kr.value().offsets.size(), pts.size() + 1);

    for (std::size_t q = 0; q < pts.size(); ++q) {
      std::vector<std::size_t> got(
          kr.value().indices.begin() + static_cast<std::ptrdiff_t>(kr.value().offsets[q]),
          kr.value().indices.begin() + static_cast<std::ptrdiff_t>(kr.value().offsets[q + 1]));
      EXPECT_TRUE(is_valid_knn(boxes, pts[q], k, got)) << "query point " << q;
      EXPECT_EQ(got.size(), static_cast<std::size_t>(k));
    }
  }
}

TEST(BoxIndex, KNearestCapsAtIndexSize) {
  auto boxes = random_boxes(2, 3);
  auto idx = BoxIndex::build(boxes);
  ASSERT_TRUE(idx.ok());
  auto kr = idx.value().query_nearest({Point{5.0, 5.0, 0.0}}, 10);  // k > size()
  ASSERT_TRUE(kr.ok());
  ASSERT_EQ(kr.value().offsets.size(), 2u);
  EXPECT_EQ(kr.value().offsets[1] - kr.value().offsets[0], boxes.size());
}

// ---------------------------------------------------------------- determinism

TEST(BoxIndex, ResultsAreSortedAscendingByIndex) {
  auto boxes = random_boxes(3, 100);
  auto pts = random_points(3, 50);
  auto idx = BoxIndex::build(boxes);
  ASSERT_TRUE(idx.ok());

  auto qr = idx.value().query_points(pts);
  ASSERT_TRUE(qr.ok());
  for (std::size_t q = 0; q < pts.size(); ++q) {
    for (std::size_t j = qr.value().offsets[q]; j + 1 < qr.value().offsets[q + 1]; ++j) {
      EXPECT_LT(qr.value().indices[j], qr.value().indices[j + 1]);
    }
  }
}

// ---------------------------------------------------------------- execution space
// docs/dev/search.md's "Execution space" section: BoxIndex::build/query on
// Kokkos::DefaultExecutionSpace when fuse::search::ExecutionSpace resolves
// to Device, Host (the pre-existing behaviour) otherwise.

TEST(BoxIndex, HostIndexReportsHostSpace) {
  auto idx = BoxIndex::build(random_boxes(2, 10), ExecutionSpace::Host);
  ASSERT_TRUE(idx.ok()) << idx.error().message;
  EXPECT_FALSE(idx.value().executes_on_device());
  EXPECT_STREQ(idx.value().space_name(), Kokkos::DefaultHostExecutionSpace::name());
}

TEST(BoxIndex, DeviceRequestExecutesOnDeviceWhenABackendExists) {
  // On a build with no CUDA/HIP/SYCL backend, Kokkos::DefaultExecutionSpace
  // IS Kokkos::DefaultHostExecutionSpace, so `Device` degrades to `Host`
  // (box_index.cpp's `kHasDeviceBackend`) rather than failing -- both
  // branches below are therefore correct, not alternatives picked by luck.
  auto idx = BoxIndex::build(random_boxes(2, 10), ExecutionSpace::Device);
  ASSERT_TRUE(idx.ok()) << idx.error().message;
  if constexpr (!std::is_same_v<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>) {
    EXPECT_TRUE(idx.value().executes_on_device());
    EXPECT_STREQ(idx.value().space_name(), Kokkos::DefaultExecutionSpace::name());
  } else {
    EXPECT_FALSE(idx.value().executes_on_device());
  }
}

TEST(BoxIndex, DeviceBuildMatchesHostBuildForPointAndNearestQueries) {
  auto boxes = random_boxes(3, 200);
  auto pts = random_points(3, 100);

  auto host_idx = BoxIndex::build(boxes, ExecutionSpace::Host);
  ASSERT_TRUE(host_idx.ok()) << host_idx.error().message;
  auto device_idx = BoxIndex::build(boxes, ExecutionSpace::Device);
  ASSERT_TRUE(device_idx.ok()) << device_idx.error().message;

  auto host_qr = host_idx.value().query_points(pts);
  auto device_qr = device_idx.value().query_points(pts);
  ASSERT_TRUE(host_qr.ok());
  ASSERT_TRUE(device_qr.ok());
  EXPECT_EQ(host_qr.value().offsets, device_qr.value().offsets);
  EXPECT_EQ(host_qr.value().indices, device_qr.value().indices);

  constexpr int k = 5;
  auto host_kr = host_idx.value().query_nearest(pts, k);
  auto device_kr = device_idx.value().query_nearest(pts, k);
  ASSERT_TRUE(host_kr.ok());
  ASSERT_TRUE(device_kr.ok());
  EXPECT_EQ(host_kr.value().offsets, device_kr.value().offsets);
  EXPECT_EQ(host_kr.value().indices, device_kr.value().indices);
}

TEST(BoxIndex, EnvironmentVariableSelectsDeviceLikeRunnerConfig) {
  // FUSE_EXECUTION_SPACE mirrors RunnerConfig::resolved_execution_space()'s
  // own convention (runner.hpp). Set/unset directly here (rather than only
  // through a fuse_add_device_twin ctest row, test/CMakeLists.txt) so the
  // reading side is proven within this one binary too.
  ::setenv("FUSE_EXECUTION_SPACE", "device", 1);
  auto idx = BoxIndex::build(random_boxes(1, 5));
  ::unsetenv("FUSE_EXECUTION_SPACE");
  ASSERT_TRUE(idx.ok()) << idx.error().message;
  constexpr bool kHasDeviceBackend =
      !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(idx.value().executes_on_device(), kHasDeviceBackend);
}

TEST(BoxIndex, ViewBasedBuildAndQueryMatchVectorBasedOnTheSameIndex) {
  using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
  auto boxes = random_boxes(2, 50);
  auto pts = random_points(2, 30);

  Kokkos::View<Box*, MemSpace> boxes_view("test_boxes_device", boxes.size());
  auto boxes_mirror = Kokkos::create_mirror_view(boxes_view);
  for (std::size_t i = 0; i < boxes.size(); ++i) boxes_mirror(i) = boxes[i];
  Kokkos::deep_copy(boxes_view, boxes_mirror);

  auto idx = BoxIndex::build_device_resident(Kokkos::View<const Box*, MemSpace>(boxes_view));
  ASSERT_TRUE(idx.ok()) << idx.error().message;
  constexpr bool kHasDeviceBackend =
      !std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
  EXPECT_EQ(idx.value().executes_on_device(), kHasDeviceBackend);

  Kokkos::View<Point*, MemSpace> pts_view("test_pts_device", pts.size());
  auto pts_mirror = Kokkos::create_mirror_view(pts_view);
  for (std::size_t i = 0; i < pts.size(); ++i) pts_mirror(i) = pts[i];
  Kokkos::deep_copy(pts_view, pts_mirror);

  auto qrv = idx.value().query_points_device_resident(Kokkos::View<const Point*, MemSpace>(pts_view));
  ASSERT_TRUE(qrv.ok()) << qrv.error().message;
  auto offsets_mirror = Kokkos::create_mirror_view(qrv.value().offsets);
  Kokkos::deep_copy(offsets_mirror, qrv.value().offsets);
  auto indices_mirror = Kokkos::create_mirror_view(qrv.value().indices);
  Kokkos::deep_copy(indices_mirror, qrv.value().indices);

  auto vector_qr = idx.value().query_points(pts);
  ASSERT_TRUE(vector_qr.ok());
  ASSERT_EQ(offsets_mirror.extent(0), vector_qr.value().offsets.size());
  for (std::size_t i = 0; i < offsets_mirror.extent(0); ++i) {
    EXPECT_EQ(offsets_mirror(i), vector_qr.value().offsets[i]);
  }
  ASSERT_EQ(indices_mirror.extent(0), vector_qr.value().indices.size());
  for (std::size_t i = 0; i < indices_mirror.extent(0); ++i) {
    EXPECT_EQ(indices_mirror(i), vector_qr.value().indices[i]);
  }
}

}  // namespace
}  // namespace fuse::search
