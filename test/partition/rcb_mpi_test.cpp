// test/partition/rcb_mpi_test.cpp -- fuse::partition under a REAL mpiexec.
//
// NOT YET RUN: this sandbox cannot initialize MPI (mpiexec cannot bind a
// port, and a singleton MPI_Init aborts here), so nothing in this file has
// a result from this environment. It is registered only in an MPI-enabled build
// (test/CMakeLists.txt gates it on FUSE_ENABLE_MPI); running it is a host
// action -- see docs/dev/partitioning.md "What is verified where".
//
// WHAT THIS CHECKS THAT rcb_partition_test.cpp CANNOT: rcb_partition_test.cpp
// only exercises the base build configuration (comm == kNullComm) path -- one process,
// no MPI symbol touched. The collective sequence in detail::recursive_bisect
// (the axis-extent Allreduce, the median's histogram Allreduce loop, the
// tie-weight Allreduce, the tie Allgatherv) is untested by that alone. Here,
// each rank builds Kokkos views over only ITS OWN SLICE of a fixed global
// point set (owner_by_contiguous_blocks, exactly rcb_oracle.hpp's emulation
// scheme, now for real ranks instead of emulated ones) and the result is
// compared against `recursive_bisect_oracle` run ONCE on the full gathered
// input (P = 1) -- the escape hatch docs/dev/partitioning.md names: "compare against the oracle recomputed on the
// gathered input" is exactly rcb_moab's RCBBfBAcrossRanks property, checked
// the one way this sandbox cannot check it itself.

#include "fuse/partition/mesh_adapter.hpp"
#include "fuse/partition/rcb.hpp"

#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fixtures.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/error.hpp"
#include "fuse_test_main.hpp"
#include "rcb_oracle.hpp"

namespace fuse::partition {
namespace {

using Space = Kokkos::DefaultHostExecutionSpace;

TEST(RcbPartitionMpi, MatchesOracleRecomputedOnTheGatheredInput) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto ps = fuse::partition::testing::fixtures::hex_box(6);  // 216 points, fixed globally
  const auto w = fuse::partition::testing::fixtures::skewed_weights(ps.centroids.size());
  const auto owner =
      fuse::partition::testing::owner_by_contiguous_blocks(ps.centroids.size(), size);

  std::vector<std::size_t> local_indices;
  for (std::size_t i = 0; i < ps.centroids.size(); ++i) {
    if (owner[i] == rank) local_indices.push_back(i);
  }
  const std::size_t n_local = local_indices.size();

  Kokkos::View<double* [3], Space::memory_space> c("c", n_local);
  Kokkos::View<Weight*, Space::memory_space> wv("w", n_local);
  Kokkos::View<std::int64_t*, Space::memory_space> g("g", n_local);
  for (std::size_t k = 0; k < n_local; ++k) {
    const std::size_t i = local_indices[k];
    c(k, 0) = ps.centroids[i][0];
    c(k, 1) = ps.centroids[i][1];
    c(k, 2) = ps.centroids[i][2];
    wv(k) = w[i];
    g(k) = ps.gids[i];
  }

  for (int num_parts : {1, size, 2 * size, size + 3}) {
    RcbOptions opts;
    opts.num_parts = num_parts;
    opts.comm = MPI_Comm_c2f(MPI_COMM_WORLD);
    const auto result = rcb_partition<Space>(opts, c, wv, g);
    ASSERT_EQ(result.part_id.extent(0), n_local);

    // Gather (gid, part_id) from every rank so each rank can independently
    // reassemble the full global assignment and check it -- no funnel
    // through rank 0, so a rank-0-only bug cannot hide from the others.
    std::vector<std::int64_t> local_gids(n_local);
    std::vector<std::int64_t> local_parts(n_local);
    for (std::size_t k = 0; k < n_local; ++k) {
      local_gids[k] = g(k);
      local_parts[k] = result.part_id(k);
    }
    int local_count = static_cast<int>(n_local);
    std::vector<int> counts(static_cast<std::size_t>(size), 0);
    MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    int total = 0;
    for (int r = 0; r < size; ++r) {
      displs[static_cast<std::size_t>(r)] = total;
      total += counts[static_cast<std::size_t>(r)];
    }
    std::vector<std::int64_t> all_gids(static_cast<std::size_t>(total));
    std::vector<std::int64_t> all_parts(static_cast<std::size_t>(total));
    MPI_Allgatherv(local_gids.data(), local_count, MPI_INT64_T, all_gids.data(), counts.data(),
                  displs.data(), MPI_INT64_T, MPI_COMM_WORLD);
    MPI_Allgatherv(local_parts.data(), local_count, MPI_INT64_T, all_parts.data(), counts.data(),
                  displs.data(), MPI_INT64_T, MPI_COMM_WORLD);

    std::vector<int> got_by_gid(ps.centroids.size(), -1);
    for (int k = 0; k < total; ++k) {
      got_by_gid[static_cast<std::size_t>(all_gids[static_cast<std::size_t>(k)])] =
          static_cast<int>(all_parts[static_cast<std::size_t>(k)]);
    }

    const auto owner_p1 =
        fuse::partition::testing::owner_by_contiguous_blocks(ps.centroids.size(), 1);
    const auto want = fuse::partition::testing::recursive_bisect_oracle(ps.centroids, w, ps.gids,
                                                                        owner_p1, num_parts);
    for (std::size_t i = 0; i < want.size(); ++i) {
      EXPECT_EQ(got_by_gid[i], want[i]) << "num_parts=" << num_parts << " gid=" << i;
    }
  }
}

// Rank 0 alone passes a too-short weights array; every other rank's
// own arrays are well-formed. Rank 0 must not throw
// E-PART-SIZE-MISMATCH directly and return while its peers enter
// recursive_bisect's first Allreduce -- that would be a hang, not a refusal.
// This test only observes a result at all once every rank refuses together.
TEST(RcbPartitionMpi, RefusesWhenOnlyOneRankHasAMismatchedWeightsArray) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::size_t n = 10;
  Kokkos::View<double* [3], Space::memory_space> c("c", n);
  Kokkos::deep_copy(c, 0.0);
  const std::size_t n_weights = (rank == 0) ? n - 1 : n;  // rank 0 alone is short
  Kokkos::View<Weight*, Space::memory_space> w("w", n_weights);
  Kokkos::deep_copy(w, Weight{1});
  Kokkos::View<std::int64_t*, Space::memory_space> g("g", n);
  Kokkos::deep_copy(g, std::int64_t{0});

  RcbOptions opts;
  opts.num_parts = 2;
  opts.comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  EXPECT_THROW((void)(rcb_partition<Space>(opts, c, w, g)), fuse::Refusal)
      << "every rank must refuse together, including ranks whose own arrays were fine, rank "
      << rank;
}

// Every rank's own RcbOptions are individually valid, but rank 0
// asks for a different num_parts than everyone else -- the recursion's
// own shape, which must be reduced across ranks. Otherwise this either
// hangs (a differing number of histogram Allreduce levels) or silently
// produces two different, each internally-consistent, partitions.
TEST(RcbPartitionMpi, RefusesWhenRanksDisagreeAboutNumParts) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t n = 10;
  Kokkos::View<double* [3], Space::memory_space> c("c", n);
  Kokkos::deep_copy(c, 0.0);
  Kokkos::View<Weight*, Space::memory_space> w("w", n);
  Kokkos::deep_copy(w, Weight{1});
  Kokkos::View<std::int64_t*, Space::memory_space> g("g", n);
  Kokkos::deep_copy(g, std::int64_t{0});

  RcbOptions opts;
  opts.num_parts = (rank == 0) ? 1 : size;  // both individually valid, but they disagree
  opts.comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  EXPECT_THROW((void)(rcb_partition<Space>(opts, c, w, g)), fuse::Refusal) << "rank " << rank;
}

// A fully degenerate, fully-duplicated-gid fixture -- every centroid
// at the origin (so every element becomes a tie at every level) and every
// gid equal to 0 (so the ENTIRE tie order is decided by the secondary key
// alone). `rcb_partition` does not require unique gids
// (docs/dev/partitioning.md "Ties"), and this is the case where an
// unstable sort's tie resolution is observable in the final assignment,
// not just an implementation detail. `std::stable_sort` (both in
// rcb.hpp and mirrored in rcb_oracle.hpp) is what makes this a GENUINE
// parity check rather than two independently-unstable sorts that might
// happen to agree: the library (real ranks) and the oracle (the SAME P,
// emulated) are expected to assign every element identically because both
// resolve the all-ties case by the one well-defined rule the doc
// names -- gid, then gathered position -- not by whatever an unstable
// sort's internal algorithm does on each side.
TEST(RcbPartitionMpi, DuplicateGidTiesMatchOracleAtTheSameRankCount) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const std::size_t n = 2 * static_cast<std::size_t>(size);
  const auto owner = fuse::partition::testing::owner_by_contiguous_blocks(n, size);

  std::vector<std::size_t> local_indices;
  for (std::size_t i = 0; i < n; ++i) {
    if (owner[i] == rank) local_indices.push_back(i);
  }
  const std::size_t n_local = local_indices.size();

  Kokkos::View<double* [3], Space::memory_space> c("c", n_local);
  Kokkos::View<Weight*, Space::memory_space> wv("w", n_local);
  Kokkos::View<std::int64_t*, Space::memory_space> g("g", n_local);
  Kokkos::deep_copy(c, 0.0);   // every centroid at the origin -- fully degenerate
  Kokkos::deep_copy(wv, Weight{1});
  Kokkos::deep_copy(g, std::int64_t{0});  // every gid identical -- fully duplicated

  RcbOptions opts;
  opts.num_parts = 2;
  opts.comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  const auto result = rcb_partition<Space>(opts, c, wv, g);
  ASSERT_EQ(result.part_id.extent(0), n_local);

  std::vector<std::int64_t> local_parts(n_local);
  for (std::size_t k = 0; k < n_local; ++k) local_parts[k] = result.part_id(k);
  int local_count = static_cast<int>(n_local);
  std::vector<int> counts(static_cast<std::size_t>(size), 0);
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> displs(static_cast<std::size_t>(size), 0);
  int total = 0;
  for (int r = 0; r < size; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }
  std::vector<std::int64_t> all_parts(static_cast<std::size_t>(total));
  MPI_Allgatherv(local_parts.data(), local_count, MPI_INT64_T, all_parts.data(), counts.data(),
                displs.data(), MPI_INT64_T, MPI_COMM_WORLD);

  std::vector<fuse::partition::testing::OraclePoint3> oc(n, {0.0, 0.0, 0.0});
  std::vector<fuse::partition::testing::OracleWeight> ow(n, 1);
  std::vector<std::int64_t> og(n, 0);
  const auto want =
      fuse::partition::testing::recursive_bisect_oracle(oc, ow, og, owner, opts.num_parts);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(all_parts[i], want[i]) << "global index " << i;
  }
}

// Rank 0 alone holds one negative weight; every other rank's own
// weights are non-negative. Folded into the SAME agreement Allreduce as
// the size and num_parts checks above, so a rank whose own weights are fine must not proceed while
// rank 0 refuses alone.
TEST(RcbPartitionMpi, RefusesWhenOnlyOneRankHasANegativeWeight) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const std::size_t n = 10;
  Kokkos::View<double* [3], Space::memory_space> c("c", n);
  Kokkos::deep_copy(c, 0.0);
  Kokkos::View<Weight*, Space::memory_space> w("w", n);
  Kokkos::deep_copy(w, Weight{1});
  if (rank == 0) {
    auto w_host = Kokkos::create_mirror_view(w);
    Kokkos::deep_copy(w_host, w);
    w_host(3) = Weight{-1};
    Kokkos::deep_copy(w, w_host);
  }
  Kokkos::View<std::int64_t*, Space::memory_space> g("g", n);
  Kokkos::deep_copy(g, std::int64_t{0});

  RcbOptions opts;
  opts.num_parts = 2;
  opts.comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  EXPECT_THROW((void)(rcb_partition<Space>(opts, c, w, g)), fuse::Refusal)
      << "every rank must refuse together, including ranks whose own weights were fine, rank "
      << rank;
}

TEST(MeshAdapterMpi, RefusesWhenRanksDisagreeAboutMeshDimension) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  fuse::SerialMeshBackend backend;
  // Rank 0 loads a 1D mesh; every other rank loads a 3D box -- a genuine
  // cross-rank disagreement about MeshView::dimension().
  const std::string source = (rank == 0) ? "generated:line:5" : "generated:box:2x2x2";
  ASSERT_TRUE(backend.load(fuse::MeshDecl{.name = "m", .source = source}).ok());
  const auto view_r = backend.view(fuse::MeshId{0});
  ASSERT_TRUE(view_r.ok());

  const auto r = centroids_of(*view_r.value(), MPI_Comm_c2f(MPI_COMM_WORLD));
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-PART-MIXED-DIM");
}

}  // namespace
}  // namespace fuse::partition
