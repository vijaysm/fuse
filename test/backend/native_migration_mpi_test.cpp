// test/backend/native_migration_mpi_test.cpp — real migration under a REAL
// mpiexec, RCB-driven.
//
// NOT YET RUN: this sandbox cannot initialize MPI (mpiexec cannot bind a
// port, and a singleton MPI_Init aborts here). Registered only in a Tier 1
// build (test/CMakeLists.txt gates it on FUSE_ENABLE_MPI); running it is a
// host action -- see docs/dev/native-mesh.md "What is verified where".
//
// WHAT THIS CHECKS THAT THE EMULATED TIER-0 TESTS CANNOT: each rank here
// builds its OWN slice via generate_native_mesh(spec, rank, size) -- a
// REAL distributed mesh, one process per rank, not pack()/unpack() called
// directly in one process -- then fuse::partition::repartition (RCB +
// migrate) runs the actual collective sequence
// (centroids_of's dimension-agreement Allreduce, rcb_partition's own
// collectives, migrate's Allreduce refusal-agreement, the real
// MPI_Alltoallv exchange, and resolve_vertex_ownership's Allgatherv).

#include "fuse/partition/repartition.hpp"

#include "fuse/backend/native_backend.hpp"

#include <mpi.h>

#include <cstdint>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"

namespace {

using namespace fuse::native;

fuse::partition::CommHandle world_handle() { return MPI_Comm_c2f(MPI_COMM_WORLD); }

}  // namespace

TEST(NativeMigrationMpi, RepartitionConservesGlobalCellCountAndMeasureAndUniqueIds) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  FUSE_REQUIRE_AT_LEAST_RANKS(2);

  auto arrays = generate_native_mesh("hexbox:6x5x4", rank, size);  // 120 cells, fixed globally
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), rank);
  ASSERT_TRUE(view.ok()) << view.error().message;

  const int local_cells = static_cast<int>(view.value()->owned_count(3));
  int global_cells = 0;
  MPI_Allreduce(&local_cells, &global_cells, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  ASSERT_EQ(global_cells, 120);

  double local_measure = 0.0;
  for (std::size_t c = 0; c < view.value()->cell_count(); ++c) {
    if (view.value()->is_owned(3, c)) local_measure += view.value()->measure(3, c);
  }
  double global_measure_before = 0.0;
  MPI_Allreduce(&local_measure, &global_measure_before, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_NEAR(global_measure_before, 1.0, 1e-9);

  fuse::partition::RepartitionOptions opts;
  opts.comm = world_handle();
  auto result = fuse::partition::repartition(*view.value(), opts);
  ASSERT_TRUE(result.ok()) << result.error().message;

  const int migrated_local_cells = static_cast<int>(result.value()->owned_count(3));
  int migrated_global_cells = 0;
  MPI_Allreduce(&migrated_local_cells, &migrated_global_cells, 1, MPI_INT, MPI_SUM,
               MPI_COMM_WORLD);
  EXPECT_EQ(migrated_global_cells, 120) << "the global cell count must be conserved by "
                                           "migration";

  double migrated_local_measure = 0.0;
  for (std::size_t c = 0; c < result.value()->cell_count(); ++c) {
    EXPECT_TRUE(result.value()->is_owned(3, c)) << "a migrated cell is always owned by its "
                                                   "destination";
    migrated_local_measure += result.value()->measure(3, c);
  }
  double global_measure_after = 0.0;
  MPI_Allreduce(&migrated_local_measure, &global_measure_after, 1, MPI_DOUBLE, MPI_SUM,
               MPI_COMM_WORLD);
  EXPECT_NEAR(global_measure_after, global_measure_before, 1e-9)
      << "total measure must be preserved by migration";

  // Global cell ids are unique: gather every rank's local gid COUNT, then
  // every rank's gids, and check for duplicates in the concatenated set.
  std::vector<std::int64_t> local_gids;
  for (std::size_t c = 0; c < result.value()->cell_count(); ++c) {
    local_gids.push_back(static_cast<std::int64_t>(result.value()->global_id(3, c)));
  }
  const int local_count = static_cast<int>(local_gids.size());
  std::vector<int> counts(static_cast<std::size_t>(size));
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> displs(static_cast<std::size_t>(size));
  int total = 0;
  for (int r = 0; r < size; ++r) { displs[static_cast<std::size_t>(r)] = total; total += counts[static_cast<std::size_t>(r)]; }
  std::vector<std::int64_t> all_gids(static_cast<std::size_t>(total));
  MPI_Allgatherv(local_gids.data(), local_count, MPI_INT64_T, all_gids.data(), counts.data(),
                displs.data(), MPI_INT64_T, MPI_COMM_WORLD);
  std::set<std::int64_t> unique_gids(all_gids.begin(), all_gids.end());
  EXPECT_EQ(unique_gids.size(), all_gids.size()) << "global cell ids must be unique after "
                                                     "migration";
  EXPECT_EQ(unique_gids.size(), 120u);
}

TEST(NativeMigrationMpi, RefusesAnAssignmentThatIsIncompleteOnOnlyOneRank) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  FUSE_REQUIRE_AT_LEAST_RANKS(2);

  auto arrays = generate_native_mesh("hexbox:4x4x4", rank, size);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), rank);
  ASSERT_TRUE(view.ok());

  std::vector<int> dest(view.value()->owned_count(3), 0);
  if (rank == 0) dest.pop_back();  // only rank 0's assignment is short

  auto result = migrate(*view.value(), dest, world_handle());
  ASSERT_FALSE(result.ok()) << "every rank must refuse together, including ranks whose own "
                               "assignment was fine";
}

TEST(NativeMigrationMpi, RoundTripThroughTwoDifferentPartitionsReproducesGlobalMeasure) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  FUSE_REQUIRE_AT_LEAST_RANKS(2);

  auto arrays = generate_native_mesh("hexbox:5x5x2", rank, size);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), rank);
  ASSERT_TRUE(view.ok());

  // Round 1: RCB onto the current communicator.
  fuse::partition::RepartitionOptions opts;
  opts.comm = world_handle();
  auto round1 = fuse::partition::repartition(*view.value(), opts);
  ASSERT_TRUE(round1.ok()) << round1.error().message;

  // Round 2: migrate everything to rank 0 (a different partition: an
  // all-to-one gather).
  std::vector<int> dest2(round1.value()->owned_count(3), 0);
  auto final_mesh = migrate(*round1.value(), dest2, world_handle());
  ASSERT_TRUE(final_mesh.ok()) << final_mesh.error().message;

  double local_final_measure = 0.0;
  for (std::size_t c = 0; c < final_mesh.value()->cell_count(); ++c) {
    local_final_measure += final_mesh.value()->measure(3, c);
  }
  double global_final_measure = 0.0;
  MPI_Allreduce(&local_final_measure, &global_final_measure, 1, MPI_DOUBLE, MPI_SUM,
               MPI_COMM_WORLD);
  if (rank == 0) {
    EXPECT_NEAR(global_final_measure, 1.0, 1e-9);
    EXPECT_EQ(final_mesh.value()->cell_count(), 50u);  // 5*5*2
  }
}
