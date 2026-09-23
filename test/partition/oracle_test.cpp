// test/partition/oracle_test.cpp — self-consistency of the serial RCB
// oracle (test/partition/rcb_oracle.hpp) at emulated P = 1..5. No Kokkos,
// no MPI: this is the property rcb_moab calls RcbBfBAcrossRanks, checked
// on the oracle alone, runnable in this sandbox.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "fixtures.hpp"
#include "rcb_oracle.hpp"

namespace fuse::partition::testing {
namespace {

using fixtures::PointSet;

std::vector<int> owners_for(std::size_t n, int P) { return owner_by_contiguous_blocks(n, P); }

// -------------------------------------------------------- comm-size independence

class OracleCommSizeIndependence : public ::testing::TestWithParam<int> {};

TEST_P(OracleCommSizeIndependence, LineUniformWeights) {
  const PointSet ps = fixtures::line(37);
  const std::vector<OracleWeight> w = fixtures::uniform_weights(ps.centroids.size());
  const int P = GetParam();
  for (int num_parts : {1, P, 2 * P, P + 3}) {
    if (num_parts <= 0) continue;
    const auto owner_p1 = owners_for(ps.centroids.size(), 1);
    const auto owner_pn = owners_for(ps.centroids.size(), P);
    const auto a1 = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner_p1, num_parts);
    const auto an = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner_pn, num_parts);
    EXPECT_EQ(a1, an) << "P=1 vs P=" << P << " disagree at num_parts=" << num_parts;
  }
}

TEST_P(OracleCommSizeIndependence, HexBoxSkewedWeights) {
  const PointSet ps = fixtures::hex_box(6);  // 216 points
  const std::vector<OracleWeight> w = fixtures::skewed_weights(ps.centroids.size());
  const int P = GetParam();
  for (int num_parts : {1, P, 2 * P, P + 3}) {
    if (num_parts <= 0) continue;
    const auto owner_p1 = owners_for(ps.centroids.size(), 1);
    const auto owner_pn = owners_for(ps.centroids.size(), P);
    const auto a1 = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner_p1, num_parts);
    const auto an = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner_pn, num_parts);
    EXPECT_EQ(a1, an) << "P=1 vs P=" << P << " disagree at num_parts=" << num_parts;
  }
}

INSTANTIATE_TEST_SUITE_P(EmulatedP, OracleCommSizeIndependence, ::testing::Values(1, 2, 3, 4, 5));

// -------------------------------------------------------------- distribution shapes

TEST(OracleDistributionShapes, AllIdenticalPointsDistributesByGid) {
  const PointSet ps = fixtures::all_identical(20);
  const std::vector<OracleWeight> w = fixtures::uniform_weights(ps.centroids.size());
  const auto owner = owners_for(ps.centroids.size(), 4);
  const auto assignment = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, 4);
  // Every point ties (same coordinate); with uniform weight the algorithm
  // must still produce a balanced split by GID, exercising the "no axis
  // has positive extent -> depth % 3 fallback, then GID-ordered tie
  // distribution" path rather than crashing or leaving assignments unset.
  std::vector<int> counts(4, 0);
  for (int p : assignment) {
    ASSERT_GE(p, 0);
    ASSERT_LT(p, 4);
    ++counts[static_cast<std::size_t>(p)];
  }
  for (int c : counts) EXPECT_EQ(c, 5);  // 20 identical points, 4 parts, weight 1 each
}

TEST(OracleDistributionShapes, ZeroTotalWeightStillAssignsEveryElement) {
  const PointSet ps = fixtures::hex_box(4);
  const std::vector<OracleWeight> w = fixtures::zero_weights(ps.centroids.size());
  const auto owner = owners_for(ps.centroids.size(), 3);
  const auto assignment = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, 5);
  EXPECT_EQ(assignment.size(), ps.centroids.size());
  for (int p : assignment) {
    EXPECT_GE(p, 0);
    EXPECT_LT(p, 5);
  }
}

TEST(OracleDistributionShapes, EmptyRankContributesNothingButIsIncluded) {
  // Owner assignment where emulated rank 2 (of 4) owns zero elements --
  // rcb_moab's "empty ranks" case. The recursion must still reach every
  // reduction (never skip one because a local slice is empty) and produce
  // a valid full assignment.
  const PointSet ps = fixtures::line(30);
  const std::vector<OracleWeight> w = fixtures::uniform_weights(ps.centroids.size());
  std::vector<int> owner(ps.centroids.size());
  for (std::size_t i = 0; i < owner.size(); ++i) {
    const int r = static_cast<int>(i % 3);  // never produces rank 3
    owner[i] = (r == 2) ? 3 : r;            // ranks {0,1,3} used, rank 2 empty, P=4
  }
  const auto assignment = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, 4);
  EXPECT_EQ(assignment.size(), ps.centroids.size());
  for (int p : assignment) {
    EXPECT_GE(p, 0);
    EXPECT_LT(p, 4);
  }
}

TEST(OracleDistributionShapes, SinglePartAssignsEveryElementToZero) {
  const PointSet ps = fixtures::planar_mixed(5);
  const std::vector<OracleWeight> w = fixtures::skewed_weights(ps.centroids.size());
  const auto owner = owners_for(ps.centroids.size(), 3);
  const auto assignment = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, 1);
  for (int p : assignment) EXPECT_EQ(p, 0);
}

// ------------------------------------------------------------------ imbalance

TEST(OracleImbalance, UniformWeightHexBoxWithinBound) {
  const PointSet ps = fixtures::hex_box(8);  // 512 points, uniform weight
  const std::vector<OracleWeight> w = fixtures::uniform_weights(ps.centroids.size());
  const int num_parts = 8;
  const auto owner = owners_for(ps.centroids.size(), 1);
  const auto assignment = recursive_bisect_oracle(ps.centroids, w, ps.gids, owner, num_parts);

  std::vector<OracleWeight> part_weight(static_cast<std::size_t>(num_parts), 0);
  OracleWeight total = 0;
  for (std::size_t i = 0; i < assignment.size(); ++i) {
    part_weight[static_cast<std::size_t>(assignment[i])] += w[i];
    total += w[i];
  }
  const double ideal = static_cast<double>(total) / static_cast<double>(num_parts);
  for (OracleWeight pw : part_weight) {
    // Exact, not approximate. The histogram median only has to land the split
    // inside the right final bucket; every point inside that bucket is then a
    // tie, and ties are handed left in ascending global id until the left
    // weight equals the target exactly. With 512 uniform weights and 8 parts
    // every part is therefore exactly 64 (rcb_moab's own sweep measured an
    // excess imbalance of 1.0000 on every evenly dividing fixture). A 15%
    // slack, which this test once had, would pass an RCB whose tie
    // distribution had stopped working.
    EXPECT_EQ(static_cast<double>(pw), ideal)
        << "part weight " << pw << " vs ideal " << ideal;
  }
}

}  // namespace
}  // namespace fuse::partition::testing
