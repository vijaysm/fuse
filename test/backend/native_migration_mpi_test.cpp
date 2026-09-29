// test/backend/native_migration_mpi_test.cpp — real migration under a REAL
// mpiexec, RCB-driven.
//
// NOT YET RUN: this sandbox cannot initialize MPI (mpiexec cannot bind a
// port, and a singleton MPI_Init aborts here). Registered only in the MPI-enabled build configuration
// build (test/CMakeLists.txt gates it on FUSE_ENABLE_MPI); running it is a
// host action -- see docs/dev/native-mesh.md "What is verified where".
//
// WHAT THIS CHECKS THAT THE EMULATED BASE-BUILD-CONFIGURATION TESTS CANNOT: each rank here
// builds its OWN slice via generate_native_mesh(spec, rank, size) -- a
// REAL distributed mesh, one process per rank, not pack()/unpack() called
// directly in one process -- then fuse::partition::repartition (RCB +
// migrate) runs the actual collective sequence
// (centroids_of's dimension-agreement Allreduce, rcb_partition's own
// collectives, migrate's Allreduce rejection-agreement, the real
// MPI_Alltoallv exchange, and resolve_vertex_ownership's Allgatherv).

#include "fuse/partition/repartition.hpp"

#include "fuse/backend/native_backend.hpp"

#include <mpi.h>

#include <cstdint>
#include <set>
#include <unordered_map>
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
  ASSERT_FALSE(result.ok()) << "every rank must reject together, including ranks whose own "
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

// D3 (KNOWN_ISSUES) / `vertex-ownership-rule-picks-max-not-min-rank`
// (build-evidence/mutations/native-mesh.toml): docs/dev/native-mesh.md's own
// words for `resolve_vertex_ownership` -- "a vertex's new owner is the
// MINIMUM rank, over the whole communicator, among ranks that now hold a
// copy of it" -- and `src/backend/native_migration.cpp`'s own comparison,
// `if (it == min_rank_of.end() || r < it->second) min_rank_of[gid] = r;`,
// which the mutation flips to `r >`. `RepartitionConservesGlobalCellCount-
// AndMeasureAndUniqueIds` (above) does not catch this: conservation and
// uniqueness of CELL ids never read `vertex_owner`/`vertex_is_owned` at
// all, so the mutant is invisible to every check that test makes.
//
// This test recomputes the minimum-holder-rank rule INDEPENDENTLY, from the
// same shape of Allgatherv `resolve_vertex_ownership` itself performs (every
// rank's own vertex gid list, concatenated), and checks it against what
// `vertex_owner_of()` actually reports on every rank holding a copy of each
// shared vertex -- both that every holder AGREES on one owner, and that the
// agreed owner is the minimum rank in that vertex's holder set. Under the
// mutant, any vertex shared by more than one rank (guaranteed at an interior
// RCB partition boundary) is agreed to the MAXIMUM holder instead, and this
// test's own (unmutated, independently-recomputed) minimum disagrees with
// it outright.
TEST(NativeMigrationMpi, VertexOwnerIsTheMinimumRankAmongHoldersAfterMigration) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  FUSE_REQUIRE_AT_LEAST_RANKS(2);

  auto arrays = generate_native_mesh("hexbox:6x5x4", rank, size);
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), rank);
  ASSERT_TRUE(view.ok()) << view.error().message;

  fuse::partition::RepartitionOptions opts;
  opts.comm = world_handle();
  auto result = fuse::partition::repartition(*view.value(), opts);
  ASSERT_TRUE(result.ok()) << result.error().message;
  const NativeMeshView& mesh = *result.value();

  const std::size_t nv = mesh.vertex_count();
  std::vector<std::int64_t> local_gids(nv);
  std::vector<int>          local_owners(nv);
  for (std::size_t v = 0; v < nv; ++v) {
    local_gids[v]   = mesh.vertex_gid_of(v);
    local_owners[v] = mesh.vertex_owner_of(v);
  }

  // Same Allgather-counts-then-Allgatherv-payload shape
  // resolve_vertex_ownership itself uses, run TWICE: once for the gid each
  // rank holds a copy of (reconstructs every vertex's holder set), once for
  // what that rank's own copy says the owner is (checks agreement).
  const int local_count = static_cast<int>(nv);
  std::vector<int> counts(static_cast<std::size_t>(size));
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> displs(static_cast<std::size_t>(size));
  int total = 0;
  for (int r = 0; r < size; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }
  std::vector<std::int64_t> gathered_gids(static_cast<std::size_t>(total));
  MPI_Allgatherv(local_gids.data(), local_count, MPI_INT64_T, gathered_gids.data(),
                counts.data(), displs.data(), MPI_INT64_T, MPI_COMM_WORLD);
  std::vector<int> gathered_owners(static_cast<std::size_t>(total));
  MPI_Allgatherv(local_owners.data(), local_count, MPI_INT, gathered_owners.data(),
                counts.data(), displs.data(), MPI_INT, MPI_COMM_WORLD);

  // Independently recomputed minimum holder rank per gid (this loop is
  // this TEST's own code, never mutated), and the owner every holder of
  // that gid actually reported.
  std::unordered_map<std::int64_t, int> min_rank_of;
  std::unordered_map<std::int64_t, int> agreed_owner_of;
  bool every_holder_agrees = true;
  for (int r = 0; r < size; ++r) {
    const int start = displs[static_cast<std::size_t>(r)];
    const int count = counts[static_cast<std::size_t>(r)];
    for (int k = 0; k < count; ++k) {
      const std::size_t       idx    = static_cast<std::size_t>(start + k);
      const std::int64_t      gid    = gathered_gids[idx];
      const int               owner  = gathered_owners[idx];
      auto min_it = min_rank_of.find(gid);
      if (min_it == min_rank_of.end() || r < min_it->second) min_rank_of[gid] = r;
      auto agree_it = agreed_owner_of.find(gid);
      if (agree_it == agreed_owner_of.end()) {
        agreed_owner_of[gid] = owner;
      } else if (agree_it->second != owner) {
        every_holder_agrees = false;
      }
    }
  }
  EXPECT_TRUE(every_holder_agrees)
      << "every rank holding a copy of a vertex must agree on its owner "
         "(resolve_vertex_ownership's whole point)";

  std::size_t shared_vertices_checked = 0;
  for (std::size_t v = 0; v < nv; ++v) {
    const std::int64_t gid = local_gids[v];
    ASSERT_TRUE(min_rank_of.count(gid) != 0)
        << "gid " << gid << " is this rank's own vertex; it must appear in "
           "the gathered holder set this rank itself contributed to";
    EXPECT_EQ(local_owners[v], min_rank_of.at(gid))
        << "vertex gid " << gid << ": rank " << rank << "'s own copy reports "
           "owner " << local_owners[v] << ", but the minimum rank among "
           "every holder of this gid is " << min_rank_of.at(gid);
    ++shared_vertices_checked;
  }
  EXPECT_GT(shared_vertices_checked, 0u);
}
