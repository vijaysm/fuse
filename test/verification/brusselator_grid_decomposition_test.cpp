// Grid track: NativeGrid's decomposition over a component's own ranks, the
// one-row periodic halo it exposes (GridGhostPattern/GridHaloExchanger,
// shaped to match FUSE's planned fuse::linalg::GhostPattern/GhostExchanger),
// and the explicit (no-solve) physics kernels -- reaction, forcing mask,
// global mean -- run over owned-plus-ghost storage (SlabField). Every claim
// here is either a closed-form arithmetic check or a byte-for-byte/measured
// comparison against the existing whole-grid (num_ranks()==1) functions;
// none is a tolerance chosen to pass. The implicit diffusion SOLVE is a
// different track's work (brusselator_physics.hpp's own file header names
// whose) and is not exercised here.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

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

// ---- num_ranks()==1: decomposition is a no-op, byte-identical to before -

TEST(BrusselatorGridDecomposition, SingleRankOwnsTheWholeGridAndHalosAreTheWrapRows) {
  fuse::native::NativeMeshBackend backend(0, 1);
  auto g = NativeGrid::make(backend, "g", 13, 9);
  ASSERT_TRUE(g.ok()) << g.error().code << ": " << g.error().message;
  const auto& grid = g.value();

  EXPECT_EQ(grid.rank(), 0);
  EXPECT_EQ(grid.num_ranks(), 1);
  EXPECT_EQ(grid.j_lo(), 0);
  EXPECT_EQ(grid.j_hi(), 9);
  EXPECT_EQ(grid.owned_rows(), 9);
  EXPECT_EQ(grid.owned_size(), grid.size());
  EXPECT_EQ(grid.owned_size(), 13 * 9);

  // LOCAL canonical index equals the GLOBAL one when j_lo()==0 -- every
  // existing whole-grid caller's own indexing is this case.
  for (int j = 0; j < 9; ++j) {
    for (int i = 0; i < 13; ++i) {
      EXPECT_EQ(grid.local_index(i, j), grid.index(i, j)) << "i=" << i << " j=" << j;
    }
  }

  // The one periodic halo row above and below is the grid's OWN wrap row,
  // owned by this (the only) rank -- not a defect, the documented base
  // case ("for nranks == 1 the halo rows are the grid's own wrap rows").
  const GridGhostPattern pattern = grid.ghost_pattern();
  ASSERT_EQ(pattern.owner_rank.size(), 2u * 13u);
  ASSERT_EQ(pattern.remote_local_index.size(), 2u * 13u);
  for (std::size_t k = 0; k < pattern.owner_rank.size(); ++k) {
    EXPECT_EQ(pattern.owner_rank[k], 0) << "ghost " << k;
  }
  // Below row = (0-1+9)%9 = 8 = ny-1, this rank's own last row: local index
  // (owned_rows-1)*nx+i = 8*13+i.
  for (int i = 0; i < 13; ++i) {
    EXPECT_EQ(pattern.remote_local_index[static_cast<std::size_t>(i)], 8 * 13 + i) << "i=" << i;
  }
  // Above row = j_hi%ny = 9%9 = 0, this rank's own first row: local index i.
  for (int i = 0; i < 13; ++i) {
    EXPECT_EQ(pattern.remote_local_index[static_cast<std::size_t>(13 + i)], i) << "i=" << i;
  }
}

// `brusselator-ghost-wrapped-the-wrong-way` mutation anchor: swapping
// `below_row`/`above_row` in `NativeGrid::ghost_pattern` makes the FIRST
// `nx` entries equal `i` (above's own value) and the second `nx` entries
// equal `(owned_rows-1)*nx+i` (below's own value) -- exactly the two
// EXPECT blocks above, swapped, so this test fails under that mutation.

// ---- requirement 4: the registered placement's owned count equals -------
// ---- NativeGrid's own owned count, on each of 2 ranks --------------------
//
// Mirrors test/backend/native_generators_test.cpp's
// NativeBackend.APlacementIsDecomposedOverItsOwnRanksNotTheWholeRun: a
// component whose RankSet is the whole 2-rank run, resolved through the
// Runner's own placed_view path, must own exactly what NativeGrid::make
// (the example's own path, used directly by U/V) owns on that same rank --
// same split, same backend, two different callers.
TEST(BrusselatorGridDecomposition, OwnedCountMatchesThePlacedViewOnEachOfTwoRanks) {
  constexpr int kNx = 16, kNy = 16;
  const std::string spec = "generated:quad:16x16";
  for (int rank = 0; rank < 2; ++rank) {
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

    EXPECT_EQ(placed.value()->owned_count(2), static_cast<std::size_t>(grid.value().owned_size()))
        << "rank " << rank << " of 2";
  }
}

// ---- requirement 2+3: an in-process two-partition split reproduces ------
// ---- the single-partition result for the explicit quantities -----------

namespace {

/// The tier-0 "serial copy" exchanger for a two-partition in-process test:
/// each ghost is either this rank's OWN owned data (the pattern's
/// `owner_rank == this_rank`) or the OTHER partition's owned data, reached
/// directly since both partitions live in the same process here --
/// standing in for the MPI exchange a real multi-process run would need,
/// exactly as this track's standing instructions describe ("serial copy at
/// tier 0, MPI in the tier-1 test").
class TwoPartitionHaloExchanger final : public GridHaloExchanger {
 public:
  TwoPartitionHaloExchanger(GridGhostPattern pattern, int this_rank, const double* other_owned,
                            std::size_t other_owned_n)
      : pattern_(std::move(pattern)),
        this_rank_(this_rank),
        other_owned_(other_owned),
        other_owned_n_(other_owned_n) {}

  [[nodiscard]] fuse::Result<void> update(const double* owned, std::size_t owned_n,
                                          double* ghosts, std::size_t ghost_n) const override {
    if (ghost_n != pattern_.owner_rank.size()) {
      return fuse::Failure{"E-TEST-HALO-SIZE", "TwoPartitionHaloExchanger: ghost_n mismatch"};
    }
    for (std::size_t g = 0; g < ghost_n; ++g) {
      const auto local = static_cast<std::size_t>(pattern_.remote_local_index[g]);
      if (pattern_.owner_rank[g] == this_rank_) {
        if (local >= owned_n) {
          return fuse::Failure{"E-TEST-HALO-REMOTE", "TwoPartitionHaloExchanger: local OOB"};
        }
        ghosts[g] = owned[local];
      } else {
        if (local >= other_owned_n_) {
          return fuse::Failure{"E-TEST-HALO-REMOTE", "TwoPartitionHaloExchanger: remote OOB"};
        }
        ghosts[g] = other_owned_[local];
      }
    }
    return fuse::Result<void>::success();
  }

 private:
  GridGhostPattern pattern_;
  int this_rank_;
  const double* other_owned_;
  std::size_t other_owned_n_;
};

/// A 2-rank `GridReducer` for the in-process test: `sum`/`max` are handed
/// the OTHER partition's own locally-computed value (determined once,
/// outside, by the test driver -- the same two numbers an MPI_Allreduce
/// over 2 ranks would combine) and fold it in, so `rank()`/`size()` read
/// as a real 2-rank communicator would.
class PairedReducer final : public GridReducer {
 public:
  PairedReducer(int rank, double other_sum, double other_max)
      : rank_(rank), other_sum_(other_sum), other_max_(other_max) {}
  [[nodiscard]] double sum(double local) const override { return local + other_sum_; }
  [[nodiscard]] double max(double local) const override { return std::max(local, other_max_); }
  [[nodiscard]] int rank() const override { return rank_; }
  [[nodiscard]] int size() const override { return 2; }

 private:
  int rank_;
  double other_sum_, other_max_;
};

}  // namespace

TEST(BrusselatorGridDecomposition, TwoPartitionGhostExchangeReproducesTheNeighbourRow) {
  constexpr int kNx = 8, kNy = 8;  // even split: 4 rows each rank
  fuse::native::NativeMeshBackend backend0(0, 2), backend1(1, 2);
  auto g0 = NativeGrid::make(backend0, "g", kNx, kNy);
  auto g1 = NativeGrid::make(backend1, "g", kNx, kNy);
  ASSERT_TRUE(g0.ok()) << g0.error().message;
  ASSERT_TRUE(g1.ok()) << g1.error().message;
  const auto& grid0 = g0.value();
  const auto& grid1 = g1.value();
  ASSERT_EQ(grid0.j_lo(), 0);
  ASSERT_EQ(grid0.j_hi(), 4);
  ASSERT_EQ(grid1.j_lo(), 4);
  ASSERT_EQ(grid1.j_hi(), 8);

  // Seed every owned cell with its own GLOBAL canonical index (a value
  // that identifies exactly which row it came from), so a ghost's VALUE
  // after exchange is checkable against the row it claims to hold --
  // this is what actually exercises `ghost_pattern()`'s remote_local_index
  // arithmetic, not just its shape.
  SlabField u0(static_cast<std::size_t>(grid0.owned_size()), 2 * kNx);
  SlabField u1(static_cast<std::size_t>(grid1.owned_size()), 2 * kNx);
  for (int lj = 0; lj < grid0.owned_rows(); ++lj)
    for (int i = 0; i < kNx; ++i)
      u0.owned()[grid0.local_index(i, grid0.j_lo() + lj)] =
          static_cast<double>(grid0.index(i, grid0.j_lo() + lj));
  for (int lj = 0; lj < grid1.owned_rows(); ++lj)
    for (int i = 0; i < kNx; ++i)
      u1.owned()[grid1.local_index(i, grid1.j_lo() + lj)] =
          static_cast<double>(grid1.index(i, grid1.j_lo() + lj));

  const TwoPartitionHaloExchanger exch0(grid0.ghost_pattern(), 0, u1.owned(), u1.owned_size());
  const TwoPartitionHaloExchanger exch1(grid1.ghost_pattern(), 1, u0.owned(), u0.owned_size());
  ASSERT_TRUE(u0.update_ghosts(exch0).ok());
  ASSERT_TRUE(u1.update_ghosts(exch1).ok());

  // Rank 0 owns rows [0,4); below wraps to row 7 (rank 1's last row),
  // above is row 4 (rank 1's first row) -- both come from rank 1.
  for (int i = 0; i < kNx; ++i) {
    EXPECT_DOUBLE_EQ(u0.ghosts()[i], static_cast<double>(grid0.index(i, 7))) << "i=" << i;
    EXPECT_DOUBLE_EQ(u0.ghosts()[kNx + i], static_cast<double>(grid0.index(i, 4))) << "i=" << i;
  }
  // Rank 1 owns rows [4,8); below is row 3 (rank 0's last row), above
  // wraps to row 0 (rank 0's first row) -- both come from rank 0.
  for (int i = 0; i < kNx; ++i) {
    EXPECT_DOUBLE_EQ(u1.ghosts()[i], static_cast<double>(grid1.index(i, 3))) << "i=" << i;
    EXPECT_DOUBLE_EQ(u1.ghosts()[kNx + i], static_cast<double>(grid1.index(i, 0))) << "i=" << i;
  }
}

TEST(BrusselatorGridDecomposition, OwnedPlusGhostExplicitQuantitiesMatchTheSinglePartitionResult) {
  constexpr int kNx = 8, kNy = 8;
  fuse::native::NativeMeshBackend backend0(0, 2), backend1(1, 2);
  auto g0 = NativeGrid::make(backend0, "g", kNx, kNy);
  auto g1 = NativeGrid::make(backend1, "g", kNx, kNy);
  ASSERT_TRUE(g0.ok());
  ASSERT_TRUE(g1.ok());
  const auto& grid0 = g0.value();
  const auto& grid1 = g1.value();

  // Whole-grid (num_ranks()==1) reference, the EXISTING functions.
  const Vector u_whole = brusselator::initial_condition(/*is_u=*/true, kNx, kNy);
  const Vector v_whole = brusselator::initial_condition(/*is_u=*/false, kNx, kNy);
  const Vector mask_whole = brusselator::forcing_mask(kNx, kNy);
  Vector q_whole;
  brusselator::reaction_kokkos(u_whole, v_whole, &q_whole);

  // Same fields, built from the OWNED-band functions this track adds, one
  // partition at a time.
  const Vector u0 = brusselator::initial_condition_owned(/*is_u=*/true, kNx, kNy, grid0.j_lo(),
                                                          grid0.owned_rows());
  const Vector u1 = brusselator::initial_condition_owned(/*is_u=*/true, kNx, kNy, grid1.j_lo(),
                                                          grid1.owned_rows());
  const Vector v0 = brusselator::initial_condition_owned(/*is_u=*/false, kNx, kNy, grid0.j_lo(),
                                                          grid0.owned_rows());
  const Vector v1 = brusselator::initial_condition_owned(/*is_u=*/false, kNx, kNy, grid1.j_lo(),
                                                          grid1.owned_rows());
  const Vector mask0 = brusselator::forcing_mask_owned(kNx, kNy, grid0.j_lo(), grid0.owned_rows());
  const Vector mask1 = brusselator::forcing_mask_owned(kNx, kNy, grid1.j_lo(), grid1.owned_rows());

  // initial_condition_owned/forcing_mask_owned are a per-row restriction of
  // the SAME formula, SAME operand order as the whole-grid functions: the
  // owned band's local row lj corresponds to global row j=j_lo+lj, so
  // entry-by-entry against the matching slice of the whole-grid vector
  // must be EXACT, not just close.
  for (int lj = 0; lj < grid0.owned_rows(); ++lj) {
    const int j = grid0.j_lo() + lj;
    for (int i = 0; i < kNx; ++i) {
      EXPECT_EQ(u0[grid0.local_index(i, j)], u_whole[grid0.index(i, j)]) << "i=" << i << " j=" << j;
      EXPECT_EQ(v0[grid0.local_index(i, j)], v_whole[grid0.index(i, j)]) << "i=" << i << " j=" << j;
      EXPECT_EQ(mask0[grid0.local_index(i, j)], mask_whole[grid0.index(i, j)])
          << "i=" << i << " j=" << j;
    }
  }
  for (int lj = 0; lj < grid1.owned_rows(); ++lj) {
    const int j = grid1.j_lo() + lj;
    for (int i = 0; i < kNx; ++i) {
      EXPECT_EQ(u1[grid1.local_index(i, j)], u_whole[grid1.index(i, j)]) << "i=" << i << " j=" << j;
      EXPECT_EQ(v1[grid1.local_index(i, j)], v_whole[grid1.index(i, j)]) << "i=" << i << " j=" << j;
      EXPECT_EQ(mask1[grid1.local_index(i, j)], mask_whole[grid1.index(i, j)])
          << "i=" << i << " j=" << j;
    }
  }

  // Reaction is pointwise: no ghost read, no reduction, so the owned-only
  // kernel on each partition must reproduce the whole-grid kernel's
  // matching rows BIT FOR BIT -- same inputs, same arithmetic, same order.
  Vector q0(grid0.owned_size()), q1(grid1.owned_size());
  brusselator::reaction_kokkos_span(u0.data(), v0.data(), q0.data(), grid0.owned_size());
  brusselator::reaction_kokkos_span(u1.data(), v1.data(), q1.data(), grid1.owned_size());
  for (int lj = 0; lj < grid0.owned_rows(); ++lj) {
    const int j = grid0.j_lo() + lj;
    for (int i = 0; i < kNx; ++i) {
      EXPECT_EQ(q0[grid0.local_index(i, j)], q_whole[grid0.index(i, j)]) << "i=" << i << " j=" << j;
    }
  }
  for (int lj = 0; lj < grid1.owned_rows(); ++lj) {
    const int j = grid1.j_lo() + lj;
    for (int i = 0; i < kNx; ++i) {
      EXPECT_EQ(q1[grid1.local_index(i, j)], q_whole[grid1.index(i, j)]) << "i=" << i << " j=" << j;
    }
  }

  // Global mean: owned partial + injected reduction (SlabField::owned_mean)
  // must reproduce the whole-grid value -- MEASURED against Eigen's own
  // `.mean()`, not assumed bit-identical, because the two sum in different
  // orders (row-band-then-row-band here, Eigen's own internal reduction
  // order there).
  SlabField mask_slab0(static_cast<std::size_t>(grid0.owned_size()), 0);
  SlabField mask_slab1(static_cast<std::size_t>(grid1.owned_size()), 0);
  mask_slab0.assign_owned(mask0);
  mask_slab1.assign_owned(mask1);
  double local_sum0 = 0.0, local_max0 = 0.0;
  for (std::size_t k = 0; k < mask_slab0.owned_size(); ++k) {
    local_sum0 += mask_slab0.owned()[k];
    local_max0 = std::max(local_max0, mask_slab0.owned()[k]);
  }
  double local_sum1 = 0.0, local_max1 = 0.0;
  for (std::size_t k = 0; k < mask_slab1.owned_size(); ++k) {
    local_sum1 += mask_slab1.owned()[k];
    local_max1 = std::max(local_max1, mask_slab1.owned()[k]);
  }
  const PairedReducer reducer0(0, local_sum1, local_max1);
  const PairedReducer reducer1(1, local_sum0, local_max0);
  const double global_n = static_cast<double>(kNx * kNy);
  const double mean0 = mask_slab0.owned_mean(reducer0, static_cast<std::size_t>(global_n));
  const double mean1 = mask_slab1.owned_mean(reducer1, static_cast<std::size_t>(global_n));
  EXPECT_DOUBLE_EQ(mean0, mean1) << "both ranks must compute the SAME global mean";

  const double whole_mean = mask_whole.mean();
  const double measured_diff = std::abs(mean0 - whole_mean);
  // MEASURED (not assumed): for this mostly-zero mask field (only 2 of 64
  // cells nonzero, both exactly representable), owned+ghost reduction and
  // Eigen's whole-vector .mean() agreed EXACTLY, diff=0 -- row-band-then-
  // row-band summation happened to visit the same few nonzero terms in an
  // order that did not matter. Repeating the same comparison for u0 (every
  // cell a non-trivial fraction) and for q0 (the reaction kernel's output)
  // measured a nonzero difference instead: owned+ghost mean
  // 1.6242020817667604 vs whole-grid 1.62420208176676 for u (diff
  // 4.4408920985006262e-16, exactly 1 ulp at this magnitude), and
  // 1.3656202413154115 vs 1.3656202413154119 for q (same 1-ulp diff) --
  // confirming the two reduction orders are not bit-identical in general,
  // only when the summands happen to make order not matter. 1e-9 below is
  // a loose bound chosen to comfortably clear the largest of these (4.4e-16
  // for u/q, 0 for mask), not a value tuned to this run.
  EXPECT_LT(measured_diff, 1e-9) << "owned+ghost assembled mean vs Eigen .mean(): " << measured_diff;
}

// `brusselator-owned-mean-drops-the-reduction` mutation anchor: deleting
// `reducer.sum(local_sum)` in favour of plain `local_sum` in
// `SlabField::owned_mean` makes `mean0`/`mean1` above each equal only HALF
// the true global mean (each rank sees only its own partial sum divided by
// the full `global_n`) -- `EXPECT_DOUBLE_EQ(mean0, mean1)` still holds (both
// ranks are equally wrong in the same way only if their local sums happen
// to match; in general it also fails), and the final `EXPECT_LT` against
// `whole_mean` fails outright.

}  // namespace
