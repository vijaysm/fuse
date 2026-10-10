// Grid track, tier 1: test/verification/brusselator_grid_decomposition_test.cpp's
// requirements 2-4 (halo exchange, explicit-quantity agreement, placement
// count) repeated on 2 REAL MPI ranks instead of an in-process two-
// partition simulation. Per this track's standing instructions, MPI
// cannot start in this sandbox at all (KNOWN_ISSUES.md C1) -- this file is
// build-verified only here; the host runs it as
// parallel.brusselator_grid_mpi.2. The MPI reducer/exchanger below are
// test-local, hand-written for exactly 2 ranks (not a general
// fuse::linalg implementation -- that is a different track's work); no
// example header includes mpi.h, so the tier-0 build this test does not
// participate in stays MPI-free.

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <fuse/backend/native_backend.hpp>
#include <fuse/model/coupling_graph.hpp>

#include "brusselator_physics.hpp"

namespace {

using brusselator::GridGhostPattern;
using brusselator::GridHaloExchanger;
using brusselator::GridReducer;
using brusselator::NativeGrid;
using brusselator::SlabField;
using brusselator::Vector;

/// `GridReducer` over a REAL communicator -- `fuse::linalg::Reducer`'s
/// planned shape, backed by `MPI_Allreduce` rather than the tier-0 tests'
/// serial/paired stand-ins.
class MpiReducer final : public GridReducer {
 public:
  explicit MpiReducer(MPI_Comm comm) : comm_(comm) {
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);
  }
  [[nodiscard]] double sum(double local) const override {
    double total = 0.0;
    MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return total;
  }
  [[nodiscard]] double max(double local) const override {
    double total = 0.0;
    MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_MAX, comm_);
    return total;
  }
  [[nodiscard]] int rank() const override { return rank_; }
  [[nodiscard]] int size() const override { return size_; }

 private:
  MPI_Comm comm_;
  int rank_ = 0, size_ = 1;
};

/// `GridHaloExchanger` for EXACTLY 2 ranks of a periodic row decomposition:
/// each rank's only neighbour (both above and below) is the other rank, so
/// the exchange is one pair of `MPI_Sendrecv` calls -- this rank's first
/// and last owned row traded for the other's. Specific to the 2-rank case
/// this test declares (`FUSE_REQUIRE_RANKS(2)`), not a general N-rank
/// implementation -- that belongs beside `fuse::linalg::make_mpi_exchanger`.
class TwoRankMpiHaloExchanger final : public GridHaloExchanger {
 public:
  TwoRankMpiHaloExchanger(MPI_Comm comm, int nx) : comm_(comm), nx_(nx) {}

  [[nodiscard]] fuse::Result<void> update(const double* owned, std::size_t owned_n,
                                          double* ghosts, std::size_t ghost_n) const override {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &size);
    if (size != 2) {
      return fuse::Failure{"E-TEST-HALO-RANKS", "TwoRankMpiHaloExchanger: needs exactly 2 ranks"};
    }
    if (ghost_n != static_cast<std::size_t>(2 * nx_) || owned_n < static_cast<std::size_t>(nx_)) {
      return fuse::Failure{"E-TEST-HALO-SIZE", "TwoRankMpiHaloExchanger: size mismatch"};
    }
    const int other = 1 - rank;
    std::vector<double> send_last(owned + (owned_n - static_cast<std::size_t>(nx_)), owned + owned_n);
    std::vector<double> send_first(owned, owned + nx_);
    std::vector<double> recv_first(static_cast<std::size_t>(nx_)), recv_last(static_cast<std::size_t>(nx_));
    // This rank's LAST row becomes the other's "below" ghost; this rank's
    // FIRST row becomes the other's "above" ghost -- tag 0/1 names which.
    // The send and the receive of ONE call carry the same tag: both ranks
    // send their last row (tag 0) and receive the peer's last row in the
    // same call, then the same for the first row (tag 1). The first host
    // run (2026-10-10) hung here with the receives crossed (call 1 sending
    // tag 0 and waiting for tag 1): each rank's first MPI_Sendrecv then
    // needed the message the peer only sends from its SECOND call, which
    // the peer reaches only after its first completes -- both ranks sampled
    // inside MPIR_Sendrecv_impl, neither returning.
    MPI_Sendrecv(send_last.data(), nx_, MPI_DOUBLE, other, 0, recv_last.data(), nx_, MPI_DOUBLE,
                other, 0, comm_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_first.data(), nx_, MPI_DOUBLE, other, 1, recv_first.data(), nx_, MPI_DOUBLE,
                other, 1, comm_, MPI_STATUS_IGNORE);
    // ghosts[0..nx) = below row = other's last row (received as recv_last);
    // ghosts[nx..2nx) = above row = other's first row (recv_first) -- the
    // SAME order NativeGrid::ghost_pattern documents.
    for (int i = 0; i < nx_; ++i) ghosts[static_cast<std::size_t>(i)] = recv_last[static_cast<std::size_t>(i)];
    for (int i = 0; i < nx_; ++i) {
      ghosts[static_cast<std::size_t>(nx_ + i)] = recv_first[static_cast<std::size_t>(i)];
    }
    return fuse::Result<void>::success();
  }

 private:
  MPI_Comm comm_;
  int nx_;
};

}  // namespace

// ---- requirement 4: placement owned count equals NativeGrid's own -------
TEST(BrusselatorGridMpi, OwnedCountMatchesThePlacedViewOnTwoRealRanks) {
  FUSE_REQUIRE_RANKS(2);
  const int rank = fuse::testing::rank();
  constexpr int kNx = 16, kNy = 16;
  const std::string spec = "generated:quad:16x16";

  fuse::CouplingGraph g;
  g.add_mesh(fuse::MeshDecl{"m", spec});
  const auto comp = g.add_component(
      fuse::ComponentDecl{.name = "A", .ranks = fuse::RankSet(std::vector<int>{0, 1})});

  fuse::native::NativeMeshBackend placement_backend(rank, 2);
  ASSERT_TRUE(placement_backend.load(fuse::MeshDecl{"m", spec}, g).ok());
  const auto mesh_id = placement_backend.mesh_id("m");
  ASSERT_TRUE(mesh_id.ok());
  const auto placed = placement_backend.placed_view(mesh_id.value(), comp);
  ASSERT_TRUE(placed.ok()) << placed.error().message;

  fuse::native::NativeMeshBackend grid_backend(rank, 2);
  auto grid = NativeGrid::make(grid_backend, "g", kNx, kNy);
  ASSERT_TRUE(grid.ok()) << grid.error().code << ": " << grid.error().message;

  EXPECT_EQ(placed.value()->owned_count(2), static_cast<std::size_t>(grid.value().owned_size()));
}

// ---- requirements 1-3: decomposition, real ghost exchange, explicit -----
// ---- quantities, on 2 real ranks -----------------------------------------
TEST(BrusselatorGridMpi, HaloExchangeAndExplicitQuantitiesAgreeOnTwoRealRanks) {
  FUSE_REQUIRE_RANKS(2);
  constexpr int kNx = 8, kNy = 8;
  const int rank = fuse::testing::rank();
  const MPI_Comm comm = MPI_COMM_WORLD;

  fuse::native::NativeMeshBackend backend(rank, 2);
  auto g = NativeGrid::make(backend, "g", kNx, kNy);
  ASSERT_TRUE(g.ok()) << g.error().message;
  const auto& grid = g.value();
  ASSERT_EQ(grid.owned_rows(), 4);

  // Seed with the GLOBAL canonical index, exactly as the tier-0 two-
  // partition test does, so a ghost's value after a REAL MPI exchange is
  // checkable against the row it claims to hold.
  SlabField u(static_cast<std::size_t>(grid.owned_size()), 2 * kNx);
  for (int lj = 0; lj < grid.owned_rows(); ++lj) {
    const int j = grid.j_lo() + lj;
    for (int i = 0; i < kNx; ++i) u.owned()[grid.local_index(i, j)] = static_cast<double>(grid.index(i, j));
  }

  const TwoRankMpiHaloExchanger exchanger(comm, kNx);
  ASSERT_TRUE(u.update_ghosts(exchanger).ok());

  const int below_row = ((grid.j_lo() - 1) % kNy + kNy) % kNy;
  const int above_row = grid.j_hi() % kNy;
  for (int i = 0; i < kNx; ++i) {
    EXPECT_DOUBLE_EQ(u.ghosts()[i], static_cast<double>(grid.index(i, below_row)));
    EXPECT_DOUBLE_EQ(u.ghosts()[kNx + i], static_cast<double>(grid.index(i, above_row)));
  }

  // Explicit quantities: reaction is pointwise (no ghost read needed), and
  // the global mean is an owned partial plus a REAL MPI_Allreduce.
  const Vector mask_owned = brusselator::forcing_mask_owned(kNx, kNy, grid.j_lo(), grid.owned_rows());
  SlabField mask_slab(static_cast<std::size_t>(grid.owned_size()), 0);
  mask_slab.assign_owned(mask_owned);
  const MpiReducer reducer(comm);
  const double mean = mask_slab.owned_mean(reducer, static_cast<std::size_t>(kNx * kNy));

  const Vector mask_whole = brusselator::forcing_mask(kNx, kNy);
  EXPECT_NEAR(mean, mask_whole.mean(), 1e-9);
}
