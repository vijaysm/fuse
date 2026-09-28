// test/parallel/moab_placed_view_mpi_test.cpp -- KNOWN_ISSUES D5, the MOAB
// half: one moab::Core + moab::ParallelComm per declared placement, run
// under a REAL communicator of more than one rank.
//
// test/backend/moab_mesh_test.cpp covers everything Mode A (no
// communicator adopted) can prove without MPI at all -- every refusal this
// item adds, and that a one-rank placement equal to the whole (size-1)
// communicator matches the default view exactly. What that file cannot
// reach is the real split-and-read machinery: a genuine
// `moab::ParallelComm`, on its own `moab::Core`, split from the base
// communicator over exactly one placement's ranks, reading a REAL
// partitioned MOAB mesh. That needs >1 rank, hence this file.
//
// SCOPE, stated rather than assumed: this proves the VIEW layer only --
// that `placed_view` answers a genuine per-placement partition of a real
// mesh, entity for entity, and an empty view to every rank outside it.
// Per-placement REMAP WEIGHTS on MOAB (RemapPlan::placement resolving
// through this same `placed_view`) are not built here: assembling a
// conservative weight matrix from two independently-partitioned MOAB
// meshes is its own undertaking, out of reach in this round, and faking
// the check by asserting on the view layer alone and calling it a weights
// test would be worse than not claiming it. KNOWN_ISSUES.md records this
// as still open.
//
// NOT YET RUN: this sandbox cannot initialise MPI at all -- not even at
// one rank (docs/dev/testing.md; KNOWN_ISSUES.md C1) -- and MOAB in a
// build with FUSE_ENABLE_MPI links against an MPI-enabled MOAB whose
// moab::Core constructor itself calls MPI_Init_thread, so even a
// single-process run of ANY MOAB-linked binary aborts before main() runs a
// single test -- confirmed empirically against this tree's PRE-EXISTING
// fuse_backend_moab_mesh_test binary (it crashes identically on tests this
// change did not touch), not assumed from this file's own header comment.
// Compiled here; the verdict comes from a host run, like every other
// tier-2 MPI test in this tree.

#include <cstdint>
#include <set>
#include <vector>

#include <mpi.h>

#include <gtest/gtest.h>

#include "fuse/backend/moab_backend.hpp"
#include "fuse/model/coupling_graph.hpp"

#include "fuse_test_main.hpp"

namespace {

int world_handle() { return static_cast<int>(MPI_Comm_c2f(MPI_COMM_WORLD)); }

/// Sums `local` over every rank of `comm` -- every partition-shape
/// assertion below reduces through this, so it is the SAME communicator a
/// component's own ranks form, not a proxy over `MPI_COMM_WORLD`.
long long allreduce_sum(MPI_Comm comm, long long local) {
  long long total = 0;
  EXPECT_EQ(MPI_Allreduce(&local, &total, 1, MPI_LONG_LONG, MPI_SUM, comm),
            MPI_SUCCESS);
  return total;
}

}  // namespace

#ifdef FUSE_SPHERE_ATM

namespace {

/// A component's own communicator, reconstructed from its `RankSet` the
/// SAME way `MoabMeshBackend::set_placements` builds it -- `MPI_Comm_split`
/// with `key = rs.index_of(rank)` -- so this test's own cross-checks below
/// run over exactly the group `placed_view` answers for, not an
/// independently-invented proxy for it.
MPI_Comm component_comm(const fuse::RankSet& rs, int rank) {
  const bool mine = rs.contains(rank);
  MPI_Comm sub = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, mine ? 0 : MPI_UNDEFINED,
                 mine ? rs.index_of(rank) : 0, &sub);
  return sub;
}

/// The shape both `TEST`s below share: declare `placements`, load the real
/// sphere mesh into every one of them, and check -- on EVERY placement --
/// that the entities its OWN ranks own tile the mesh's global entity count
/// exactly (no cell counted twice, none missing: `MoabMeshView::global_id`
/// is unique per mesh_view.hpp's own contract, so a duplicate-free local
/// set on every rank plus a sum that matches the global count together
/// rule out both a double-owned entity and a dropped one), and that a rank
/// outside the placement holds none of it.
void check_placements_partition_the_mesh(std::vector<fuse::RankSet> placements) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  fuse::MoabMeshBackend backend;
  ASSERT_TRUE(backend.set_communicator(world_handle()).ok());
  ASSERT_TRUE(backend.set_placements(placements).ok());
  const auto loaded = backend.load({.name = "atm", .source = FUSE_SPHERE_ATM});
  ASSERT_TRUE(loaded.ok()) << loaded.error().code << ": " << loaded.error().message;

  // The GLOBAL entity count, independent of any placement: the default
  // decomposition's own owned entities, summed over the whole job.
  const auto default_view = backend.view(fuse::MeshId{0});
  ASSERT_TRUE(default_view.ok());
  const int dim = default_view.value()->dimension();
  const long long global_count = allreduce_sum(
      MPI_COMM_WORLD, static_cast<long long>(default_view.value()->owned_count(dim)));
  ASSERT_GT(global_count, 0);

  for (std::size_t p = 0; p < placements.size(); ++p) {
    const fuse::ComponentId cid{static_cast<std::uint32_t>(p)};
    const auto placed = backend.placed_view(fuse::MeshId{0}, cid);
    ASSERT_TRUE(placed.ok()) << placed.error().code << ": " << placed.error().message;
    const bool mine = placements[p].contains(rank);

    if (!mine) {
      // Outside the placement: every dimension answers 0, NEVER the
      // default decomposition's own count -- the silent-wrong-answer this
      // item exists to remove.
      for (int d = 0; d <= 3; ++d) {
        EXPECT_EQ(placed.value()->entity_count(d), 0u)
            << "rank " << rank << " placement " << p << " dim " << d;
      }
      continue;
    }

    MPI_Comm comp_comm = component_comm(placements[p], rank);
    ASSERT_NE(comp_comm, MPI_COMM_NULL);

    // 1. This placement's OWN ranks tile the mesh's global entity count
    //    exactly -- the whole point of D5's MOAB half: each component
    //    holds the WHOLE region over its own ranks, not a slice shared
    //    with every other component.
    const long long placement_total = allreduce_sum(
        comp_comm, static_cast<long long>(placed.value()->owned_count(dim)));
    EXPECT_EQ(placement_total, global_count)
        << "placement " << p << " does not tile the mesh over its own ranks";

    // 2. No entity this rank owns is counted twice locally, and every
    //    owned global id is genuinely unique on this rank.
    std::vector<std::size_t> owned_gids;
    for (std::size_t i = 0; i < placed.value()->entity_count(dim); ++i) {
      if (placed.value()->is_owned(dim, i)) {
        owned_gids.push_back(placed.value()->global_id(dim, i));
      }
    }
    const std::set<std::size_t> unique_gids(owned_gids.begin(), owned_gids.end());
    EXPECT_EQ(unique_gids.size(), owned_gids.size())
        << "rank " << rank << " placement " << p
        << " owns a duplicate global id locally";

    MPI_Comm_free(&comp_comm);
  }
}

}  // namespace

TEST(MoabPlacedViewMpi, TwoDisjointPlacementsEachTileTheWholeMesh) {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  const int mid = size / 2;
  check_placements_partition_the_mesh(
      {fuse::RankSet(0, mid), fuse::RankSet(mid, size - mid)});
}

TEST(MoabPlacedViewMpi, APartiallyOverlappingPairEachStillTilesTheWholeMesh) {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
  // A on [0, size-1), B on [1, size): each covers size-1 of the size ranks,
  // sharing every rank except the two endpoints -- the partially-
  // overlapping shape docs/user/placement.md itself uses as its example.
  check_placements_partition_the_mesh(
      {fuse::RankSet(0, size - 1), fuse::RankSet(1, size - 1)});
}

TEST(MoabPlacedViewMpi, APlacementNamingNoRankGetsAnEmptyViewOnEveryRank) {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  fuse::MoabMeshBackend backend;
  ASSERT_TRUE(backend.set_communicator(world_handle()).ok());
  // Component 0 covers everyone; component 1 (a switched-off coupler, in
  // docs/user/placement.md's vocabulary) covers no rank at all.
  ASSERT_TRUE(backend
                  .set_placements({fuse::RankSet(0, size), fuse::RankSet(0, 0)})
                  .ok());
  const auto loaded = backend.load({.name = "atm", .source = FUSE_SPHERE_ATM});
  ASSERT_TRUE(loaded.ok()) << loaded.error().code << ": " << loaded.error().message;
  const auto placed = backend.placed_view(fuse::MeshId{0}, fuse::ComponentId{1});
  ASSERT_TRUE(placed.ok()) << placed.error().code << ": " << placed.error().message;
  for (int d = 0; d <= 3; ++d) {
    EXPECT_EQ(placed.value()->entity_count(d), 0u) << "dim " << d;
  }
}

#else  // !FUSE_SPHERE_ATM

TEST(MoabPlacedViewMpi, SkippedWithoutTheSphereMeshFixture) {
  GTEST_SKIP() << "wholeATM_T.h5m not found; pass -DFUSE_MOAB_MESHFILES=<moab-source>/"
                  "MeshFiles/unittest to compile the real partition checks";
}

#endif  // FUSE_SPHERE_ATM
