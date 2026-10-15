// test/partition/rcb_partition_test.cpp -- typed over EnabledExecutionSpaces
// (fuse_test_main.hpp). Verifies fuse::partition::rcb_partition against the
// serial oracle (rcb_oracle.hpp) at the base build configuration (comm == kNullComm, i.e. the
// library's single-process path -- what THIS sandbox can run; P > 1
// requires a real MPI launcher, which cannot start here (see rcb_mpi_test.cpp
// and docs/dev/partitioning.md "What is verified where").

#include "fuse/partition/rcb.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "fixtures.hpp"
#include "fuse_test_main.hpp"
#include "rcb_oracle.hpp"

namespace fuse::partition {
namespace {

using fuse::partition::testing::OracleWeight;
using fuse::partition::testing::owner_by_contiguous_blocks;
using fuse::partition::testing::recursive_bisect_oracle;

template <class Space>
class RcbPartition : public ::testing::Test {
 protected:
  using MemSpace = typename Space::memory_space;

  static Kokkos::View<const double* [3], MemSpace> upload_centroids(
      const std::vector<fuse::partition::testing::OraclePoint3>& c) {
    Kokkos::View<double* [3], MemSpace> dev("centroids", c.size());
    auto host = Kokkos::create_mirror_view(dev);
    for (std::size_t i = 0; i < c.size(); ++i) {
      host(i, 0) = c[i][0];
      host(i, 1) = c[i][1];
      host(i, 2) = c[i][2];
    }
    Kokkos::deep_copy(dev, host);
    return dev;
  }
  static Kokkos::View<const Weight*, MemSpace> upload_weights(
      const std::vector<OracleWeight>& w) {
    Kokkos::View<Weight*, MemSpace> dev("weights", w.size());
    auto host = Kokkos::create_mirror_view(dev);
    for (std::size_t i = 0; i < w.size(); ++i) host(i) = w[i];
    Kokkos::deep_copy(dev, host);
    return dev;
  }
  static Kokkos::View<const std::int64_t*, MemSpace> upload_gids(
      const std::vector<std::int64_t>& g) {
    Kokkos::View<std::int64_t*, MemSpace> dev("gids", g.size());
    auto host = Kokkos::create_mirror_view(dev);
    for (std::size_t i = 0; i < g.size(); ++i) host(i) = g[i];
    Kokkos::deep_copy(dev, host);
    return dev;
  }
  static std::vector<int> download_part_id(const RcbResult<Space>& r) {
    auto host = Kokkos::create_mirror_view(r.part_id);
    Kokkos::deep_copy(host, r.part_id);
    std::vector<int> v(r.part_id.extent(0));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = host(i);
    return v;
  }
};

}  // namespace

FUSE_TYPED_TEST_ON_SPACES(RcbPartition);

// ------------------------------------------------------------ oracle parity

TYPED_TEST(RcbPartition, LineUniformWeightsMatchesOracle) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::line(41);
  const auto w = fuse::partition::testing::fixtures::uniform_weights(ps.centroids.size());
  const auto owner = owner_by_contiguous_blocks(ps.centroids.size(), 1);

  auto dc = F::upload_centroids(ps.centroids);
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  for (int num_parts : {1, 2, 5, 7}) {
    RcbOptions opts;
    opts.num_parts = num_parts;
    const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
    const auto got = F::download_part_id(result);
    const auto want = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, num_parts);
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i], want[i]) << "num_parts=" << num_parts << " element " << i;
    }
  }
}

TYPED_TEST(RcbPartition, PlanarMixedSkewedWeightsMatchesOracle) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::planar_mixed(6);  // 36 points
  const auto w = fuse::partition::testing::fixtures::skewed_weights(ps.centroids.size());
  const auto owner = owner_by_contiguous_blocks(ps.centroids.size(), 1);

  auto dc = F::upload_centroids(ps.centroids);
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  for (int num_parts : {1, 3, 4, 9}) {
    RcbOptions opts;
    opts.num_parts = num_parts;
    const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
    const auto got = F::download_part_id(result);
    const auto want = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, num_parts);
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i], want[i]) << "num_parts=" << num_parts << " element " << i;
    }
  }
}

TYPED_TEST(RcbPartition, HexBoxUniformWeightsMatchesOracle) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::hex_box(5);  // 125 points
  const auto w = fuse::partition::testing::fixtures::uniform_weights(ps.centroids.size());
  const auto owner = owner_by_contiguous_blocks(ps.centroids.size(), 1);

  auto dc = F::upload_centroids(ps.centroids);
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  for (int num_parts : {1, 4, 8, 11}) {
    RcbOptions opts;
    opts.num_parts = num_parts;
    const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
    const auto got = F::download_part_id(result);
    const auto want = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, num_parts);
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i], want[i]) << "num_parts=" << num_parts << " element " << i;
    }
  }
}

TYPED_TEST(RcbPartition, UnstructuredCloudTetPrismPyramidPolyhedralMixedMatchesOracle) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::unstructured_cloud(200);
  const auto w = fuse::partition::testing::fixtures::skewed_weights(ps.centroids.size());
  const auto owner = owner_by_contiguous_blocks(ps.centroids.size(), 1);

  auto dc = F::upload_centroids(ps.centroids);
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  for (int num_parts : {1, 6, 13}) {
    RcbOptions opts;
    opts.num_parts = num_parts;
    const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
    const auto got = F::download_part_id(result);
    const auto want = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, num_parts);
    ASSERT_EQ(got.size(), want.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i], want[i]) << "num_parts=" << num_parts << " element " << i;
    }
  }
}

// The gnomonic path needs the SAME projection applied to the oracle's input
// as a pre-step (rcb_oracle.hpp knows nothing about coordinate transforms --
// it ports only the recursion). This local copy of the point-projection
// formula is intentionally independent of coord_transform.hpp's
// KOKKOS_INLINE_FUNCTION (a host-only float loop calling it would not
// exercise the device kernel), so it must agree with
// fuse::partition::detail::gnomonic_project_point's arithmetic exactly --
// CoordTransform.cpp's face selection and (xi, eta) ratios, copied here as
// plain host code.
fuse::partition::testing::OraclePoint3 gnomonic_project_host(
    const fuse::partition::testing::OraclePoint3& p, double face_separation) {
  double dx = p[0], dy = p[1], dz = p[2];
  const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (r <= 0.0) return {0.0, 0.0, 0.0};
  dx /= r;
  dy /= r;
  dz /= r;
  const double ax = std::fabs(dx), ay = std::fabs(dy), az = std::fabs(dz);
  int face;
  double xi, eta;
  if (ax >= ay && ax >= az) {
    face = (dx >= 0.0) ? 0 : 1;
    xi = dy / ax;
    eta = dz / ax;
  } else if (ay >= az) {
    face = (dy >= 0.0) ? 2 : 3;
    xi = dx / ay;
    eta = dz / ay;
  } else {
    face = (dz >= 0.0) ? 4 : 5;
    xi = dx / az;
    eta = dy / az;
  }
  return {xi, eta, static_cast<double>(face) * face_separation};
}

TYPED_TEST(RcbPartition, CubedSphereGnomonicMatchesOracleOnPreProjectedInput) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::cubed_sphere(4);  // 6*16 = 96 points
  const auto w = fuse::partition::testing::fixtures::uniform_weights(ps.centroids.size());
  const auto owner = owner_by_contiguous_blocks(ps.centroids.size(), 1);

  GnomonicOptions gopts;
  std::vector<fuse::partition::testing::OraclePoint3> projected(ps.centroids.size());
  for (std::size_t i = 0; i < ps.centroids.size(); ++i) {
    projected[i] = gnomonic_project_host(ps.centroids[i], gopts.face_separation);
  }

  auto dc = F::upload_centroids(ps.centroids);  // RAW sphere points to the library
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  RcbOptions opts;
  opts.num_parts = 6;
  opts.coord_transform = CoordTransform::Gnomonic;
  opts.gnomonic = gopts;
  const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
  const auto got = F::download_part_id(result);
  // Oracle sees the PRE-PROJECTED points directly (no transform option --
  // it never had one).
  const auto want = recursive_bisect_oracle(projected, w, ps.gids, owner, 6);
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_EQ(got[i], want[i]) << i;
}

// -------------------------------------------------------- degenerate cases

TYPED_TEST(RcbPartition, AllPointsIdenticalHandledWithoutCrashing) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::all_identical(24);
  const auto w = fuse::partition::testing::fixtures::uniform_weights(ps.centroids.size());

  auto dc = F::upload_centroids(ps.centroids);
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  RcbOptions opts;
  opts.num_parts = 4;
  const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
  const auto got = F::download_part_id(result);
  std::vector<int> counts(4, 0);
  for (int p : got) {
    ASSERT_GE(p, 0);
    ASSERT_LT(p, 4);
    ++counts[static_cast<std::size_t>(p)];
  }
  for (int c : counts) EXPECT_EQ(c, 6);  // 24 identical points / 4 parts, uniform weight
}

TYPED_TEST(RcbPartition, ZeroTotalWeightAssignsEveryElement) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::hex_box(3);
  const auto w = fuse::partition::testing::fixtures::zero_weights(ps.centroids.size());

  auto dc = F::upload_centroids(ps.centroids);
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  RcbOptions opts;
  opts.num_parts = 4;
  const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
  const auto got = F::download_part_id(result);
  EXPECT_EQ(got.size(), ps.centroids.size());
  for (int p : got) {
    EXPECT_GE(p, 0);
    EXPECT_LT(p, 4);
  }
}

TYPED_TEST(RcbPartition, SinglePartAssignsEveryElementToZero) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::planar_mixed(4);
  const auto w = fuse::partition::testing::fixtures::skewed_weights(ps.centroids.size());

  auto dc = F::upload_centroids(ps.centroids);
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  RcbOptions opts;
  opts.num_parts = 1;
  const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
  const auto got = F::download_part_id(result);
  for (int p : got) EXPECT_EQ(p, 0);
  EXPECT_EQ(result.num_parts_used, 1);
}

TYPED_TEST(RcbPartition, EmptyInputProducesEmptyResultRatherThanCrashing) {
  using F = RcbPartition<TypeParam>;
  std::vector<fuse::partition::testing::OraclePoint3> empty_c;
  std::vector<OracleWeight> empty_w;
  std::vector<std::int64_t> empty_g;

  auto dc = F::upload_centroids(empty_c);
  auto dw = F::upload_weights(empty_w);
  auto dg = F::upload_gids(empty_g);

  RcbOptions opts;
  opts.num_parts = 4;
  const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
  EXPECT_EQ(result.part_id.extent(0), 0u);
}

// ------------------------------------------------------------------ imbalance

TYPED_TEST(RcbPartition, UniformWeightHexBoxImbalanceWithinBound) {
  using F = RcbPartition<TypeParam>;
  const auto ps = fuse::partition::testing::fixtures::hex_box(8);  // 512 points
  const auto w = fuse::partition::testing::fixtures::uniform_weights(ps.centroids.size());

  auto dc = F::upload_centroids(ps.centroids);
  auto dw = F::upload_weights(w);
  auto dg = F::upload_gids(ps.gids);

  RcbOptions opts;
  opts.num_parts = 8;
  const auto result = rcb_partition<TypeParam>(opts, dc, dw, dg);
  const auto got = F::download_part_id(result);

  std::vector<long long> part_weight(8, 0);
  for (std::size_t i = 0; i < got.size(); ++i) part_weight[static_cast<std::size_t>(got[i])] += w[i];
  const double ideal = static_cast<double>(ps.centroids.size()) / 8.0;
  for (long long pw : part_weight) {
    // Exact, not approximate. The histogram median only has to land the split
    // inside the right final bucket; every point inside that bucket is then a
    // tie, and ties are handed left in ascending global id until the left
    // weight equals the target exactly. With 512 uniform weights and 8 parts
    // every part is therefore exactly 64 (rcb_moab's own sweep measured an
    // excess imbalance of 1.0000 on every evenly dividing fixture). A 15%
    // slack, which this test once had, would pass an RCB whose tie
    // distribution had stopped working.
    EXPECT_EQ(static_cast<double>(pw), ideal) << pw;
  }
}

}  // namespace fuse::partition
