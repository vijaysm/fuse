// Repartition migration, emulated in one process: pack() and unpack() are
// called directly (no exchange() -- see native_migration.hpp's own comment
// on why this needs no MPI), splitting a mesh into P pieces and recombining
// it under a different assignment, exactly as the partition track tests
// RCB against an emulated-rank oracle.

#include "fuse/backend/native_migration.hpp"

#include "fuse/backend/native_backend.hpp"
#include "fuse/partition/rcb.hpp"

#include <numeric>
#include <set>

#include <gtest/gtest.h>

namespace {
using namespace fuse::native;
}  // namespace

TEST(NativeMigration, EmulatedTwoRankMigrationConservesGlobalCellCountAndMeasure) {
  auto arrays1 = generate_native_mesh("hexbox:4x4x4", 0, 1);
  ASSERT_TRUE(arrays1.ok());
  auto view1 = NativeMeshView::create_from_host(arrays1.value(), 0);
  ASSERT_TRUE(view1.ok());
  const std::size_t ncells = view1.value()->cell_count();

  // Migrate everything from "rank 0" onto a new 2-way split: even cell
  // index -> rank 0, odd -> rank 1.
  std::vector<int> dest(ncells);
  for (std::size_t c = 0; c < ncells; ++c) dest[c] = static_cast<int>(c % 2);

  std::vector<std::unique_ptr<NativeMeshView>> sources;
  sources.push_back(std::move(view1.value()));

  auto packed = pack_for_migration(*sources[0], dest, 2);
  ASSERT_TRUE(packed.ok()) << packed.error().message;

  std::vector<std::unique_ptr<NativeMeshView>> result;
  std::set<std::int64_t> seen_gids;
  double total = 0.0;
  for (int r = 0; r < 2; ++r) {
    std::vector<std::vector<std::byte>> received{packed.value()[static_cast<std::size_t>(r)]};
    auto arrays = unpack_migrated_mesh(received, 3, r);
    ASSERT_TRUE(arrays.ok()) << arrays.error().message;
    auto view = NativeMeshView::create_from_host(arrays.value(), r);
    ASSERT_TRUE(view.ok()) << view.error().message;
    for (std::size_t c = 0; c < view.value()->cell_count(); ++c) {
      const auto gid = static_cast<std::int64_t>(view.value()->global_id(3, c));
      EXPECT_TRUE(seen_gids.insert(gid).second) << "cell " << gid << " duplicated across ranks";
      total += view.value()->measure(3, c);
      EXPECT_TRUE(view.value()->is_owned(3, c)) << "a migrated cell is always owned by its "
                                                   "destination";
    }
    result.push_back(std::move(view.value()));
  }
  EXPECT_EQ(seen_gids.size(), ncells);
  EXPECT_NEAR(total, 1.0, 1e-11);
}

TEST(NativeMigration, EmulatedMigrationAtPFromOneToFivePreservesMeasureAndUniqueIds) {
  for (int p : {1, 2, 3, 4, 5}) {
    auto arrays1 = generate_native_mesh("hexbox:5x3x2", 0, 1);
    ASSERT_TRUE(arrays1.ok());
    auto view1 = NativeMeshView::create_from_host(arrays1.value(), 0);
    ASSERT_TRUE(view1.ok());
    const std::size_t ncells = view1.value()->cell_count();

    std::vector<int> dest(ncells);
    for (std::size_t c = 0; c < ncells; ++c) dest[c] = static_cast<int>(c % static_cast<std::size_t>(p));

    auto packed = pack_for_migration(*view1.value(), dest, p);
    ASSERT_TRUE(packed.ok()) << "p=" << p << ": " << packed.error().message;

    std::set<std::int64_t> seen;
    double total = 0.0;
    for (int r = 0; r < p; ++r) {
      std::vector<std::vector<std::byte>> received{packed.value()[static_cast<std::size_t>(r)]};
      auto arrays = unpack_migrated_mesh(received, 3, r);
      ASSERT_TRUE(arrays.ok()) << "p=" << p << ": " << arrays.error().message;
      auto view = NativeMeshView::create_from_host(arrays.value(), r);
      ASSERT_TRUE(view.ok()) << "p=" << p << ": " << view.error().message;
      for (std::size_t c = 0; c < view.value()->cell_count(); ++c) {
        EXPECT_TRUE(seen.insert(static_cast<std::int64_t>(view.value()->global_id(3, c))).second)
            << "p=" << p;
        total += view.value()->measure(3, c);
      }
    }
    EXPECT_EQ(seen.size(), ncells) << "p=" << p;
    EXPECT_NEAR(total, 1.0, 1e-10) << "p=" << p;
  }
}

TEST(NativeMigration, EmulatedRoundTripThroughTwoDifferentPartitionsReproducesTheMesh) {
  auto arrays0 = generate_native_mesh("hexbox:3x3x3", 0, 1);
  ASSERT_TRUE(arrays0.ok());
  auto view0 = NativeMeshView::create_from_host(arrays0.value(), 0);
  ASSERT_TRUE(view0.ok());
  const std::size_t ncells = view0.value()->cell_count();
  double original_total = 0.0;
  std::set<std::int64_t> original_gids;
  for (std::size_t c = 0; c < ncells; ++c) {
    original_total += view0.value()->measure(3, c);
    original_gids.insert(static_cast<std::int64_t>(view0.value()->global_id(3, c)));
  }

  // Round 1: split onto 3 ranks by (cell_gid % 3).
  std::vector<int> dest1(ncells);
  for (std::size_t c = 0; c < ncells; ++c) dest1[c] = static_cast<int>(c % 3);
  auto packed1 = pack_for_migration(*view0.value(), dest1, 3);
  ASSERT_TRUE(packed1.ok());

  std::vector<std::unique_ptr<NativeMeshView>> round1;
  for (int r = 0; r < 3; ++r) {
    std::vector<std::vector<std::byte>> received{packed1.value()[static_cast<std::size_t>(r)]};
    auto arrays = unpack_migrated_mesh(received, 3, r);
    ASSERT_TRUE(arrays.ok());
    auto view = NativeMeshView::create_from_host(arrays.value(), r);
    ASSERT_TRUE(view.ok());
    round1.push_back(std::move(view.value()));
  }

  // Round 2: from round1's 3-way split, migrate everything onto a SINGLE
  // rank (a different partition than the original: a 3->1 gather).
  std::vector<std::vector<std::byte>> per_dest_round2(1);
  for (auto& piece : round1) {
    std::vector<int> dest2(piece->owned_count(3), 0);
    auto packed2 = pack_for_migration(*piece, dest2, 1);
    ASSERT_TRUE(packed2.ok());
    auto& bytes = packed2.value()[0];
    per_dest_round2[0].insert(per_dest_round2[0].end(), bytes.begin(), bytes.end());
  }
  auto final_arrays = unpack_migrated_mesh(per_dest_round2, 3, 0);
  ASSERT_TRUE(final_arrays.ok()) << final_arrays.error().message;
  auto final_view = NativeMeshView::create_from_host(final_arrays.value(), 0);
  ASSERT_TRUE(final_view.ok()) << final_view.error().message;

  EXPECT_EQ(final_view.value()->cell_count(), ncells);
  double final_total = 0.0;
  std::set<std::int64_t> final_gids;
  for (std::size_t c = 0; c < final_view.value()->cell_count(); ++c) {
    final_total += final_view.value()->measure(3, c);
    final_gids.insert(static_cast<std::int64_t>(final_view.value()->global_id(3, c)));
  }
  EXPECT_NEAR(final_total, original_total, 1e-10);
  EXPECT_EQ(final_gids, original_gids)
      << "a round trip through two different partitions must reproduce the same set of "
        "global cell ids (up to local ordering)";
}

// ---- rejections -------------------------------------------------------------

TEST(NativeMigration, RefusesAnIncompleteAssignment) {
  auto arrays = generate_native_mesh("hexbox:2x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  std::vector<int> dest(view.value()->cell_count() - 1, 0);  // one short
  auto packed = pack_for_migration(*view.value(), dest, 1);
  ASSERT_FALSE(packed.ok());
  EXPECT_EQ(packed.error().code, "E-MESH-MIGRATE-INCOMPLETE");
}

TEST(NativeMigration, RefusesADestinationOutsideTheCommunicator) {
  auto arrays = generate_native_mesh("hexbox:2x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  std::vector<int> dest(view.value()->cell_count(), 5);  // no rank 5 in a 2-rank comm
  auto packed = pack_for_migration(*view.value(), dest, 2);
  ASSERT_FALSE(packed.ok());
  EXPECT_EQ(packed.error().code, "E-MESH-MIGRATE-RANGE");
}

TEST(NativeMigration, MigrateAtKNullCommIsTheIdentityUpToLocalOrdering) {
  auto arrays = generate_native_mesh("hexbox:2x2x3", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  std::vector<int> dest(view.value()->cell_count(), 0);  // everyone stays on rank 0
  auto migrated = migrate(*view.value(), dest, kNullComm);
  ASSERT_TRUE(migrated.ok()) << migrated.error().message;
  EXPECT_EQ(migrated.value()->cell_count(), view.value()->cell_count());
  EXPECT_EQ(migrated.value()->vertex_count(), view.value()->vertex_count())
      << "adjacent cells share vertices; unpack must dedup by global id rather than minting a "
        "new vertex per incoming closure entry, or this count inflates";
  double total = 0.0;
  for (std::size_t c = 0; c < migrated.value()->cell_count(); ++c) {
    total += migrated.value()->measure(3, c);
    EXPECT_TRUE(migrated.value()->is_owned(3, c));
  }
  EXPECT_NEAR(total, 1.0, 1e-11);
  for (std::size_t v = 0; v < migrated.value()->vertex_count(); ++v) {
    EXPECT_TRUE(migrated.value()->is_owned(0, v)) << "kNullComm has one rank, which owns "
                                                     "everything";
  }
}

TEST(NativeMigration, MigrateRefusesAnIncompleteAssignmentAtKNullComm) {
  auto arrays = generate_native_mesh("hexbox:2x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  std::vector<int> dest(view.value()->cell_count() - 1, 0);
  auto migrated = migrate(*view.value(), dest, kNullComm);
  ASSERT_FALSE(migrated.ok());
  EXPECT_EQ(migrated.error().code, "E-MESH-MIGRATE-INCOMPLETE");
}
