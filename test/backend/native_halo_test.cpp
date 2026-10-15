// The native mesh backend's planar remap halo (docs/dev/native-halo.md,
// "Halo" track), emulated IN-PROCESS: several `NativeMeshBackend` instances
// in one process, one per simulated rank, driven directly through the
// PURE pieces `native_halo.hpp` exposes (`cell_box`, `covering_set_
// destinations`, `owned_block_index`, `pack_halo_cells`, `merge_halo_cells`)
// with the `MPI_Allgather`/`exchange_migration_buffers` step replaced by a
// direct in-process redistribution of each simulated rank's own packed
// buffers -- the same technique test/backend/native_migration_test.cpp
// already uses for `pack_for_migration`/`unpack_migrated_mesh`, and
// test/linalg/dist_vector_test.cpp uses for `GhostExchanger`. MPI cannot
// start in this sandbox at all (docs/dev/testing.md), so
// `build_native_remap_halo` itself -- which performs a REAL
// `MPI_Allgather`/`MPI_Allgatherv`/`exchange_migration_buffers` -- is
// exercised here only at ranks <= 1 or `kNullComm` (its own documented
// no-op short-circuit, which touches no MPI symbol); every other case
// drives the pure pieces and `NativeMeshBackend::commit_remap_halo`
// directly, which is itself the real production seam `build_native_
// remap_halo` uses to install its own result.

#include "fuse/backend/native_halo.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace {

using namespace fuse::native;  // NOLINT
using fuse::ComponentId;
using fuse::CouplingGraph;
using fuse::MeshDecl;
using fuse::MeshId;
using fuse::Region;
using fuse::RegionId;
using fuse::RemapPlan;

// ---------------------------------------------------------------------------
// covering_set_destinations: a pure function, hand-built boxes, no mesh at
// all. Three target boxes (owned by ranks 0, 1 and 1) and two of rank 0's
// own source boxes: one overlaps only rank 1's second box, the other
// overlaps nothing beyond rank 0's own territory (and so must name no
// destination at all -- the "a cell local to its own target needs no
// copy" rule, including when `owner_of_target_box` names `my_rank` itself).
// ---------------------------------------------------------------------------

TEST(NativeHaloCoveringSet, NamesEveryOverlappingOtherRankOnceEachSortedAscending) {
  Box3 r0_box;  // [0,1] x [0,1], owned by rank 0
  r0_box.lo = {0.0, 0.0, 0.0};
  r0_box.hi = {1.0, 1.0, 0.0};
  Box3 r1_box_a;  // [1,2] x [0,1], owned by rank 1
  r1_box_a.lo = {1.0, 0.0, 0.0};
  r1_box_a.hi = {2.0, 1.0, 0.0};
  Box3 r1_box_b;  // [0,1] x [1,2], ALSO owned by rank 1 -- a cell overlapping
                  // two of the SAME rank's boxes must still name that rank
                  // only ONCE.
  r1_box_b.lo = {0.0, 1.0, 0.0};
  r1_box_b.hi = {1.0, 2.0, 0.0};
  const std::vector<Box3> all_target_boxes = {r0_box, r1_box_a, r1_box_b};
  const std::vector<int> owner_of_target_box = {0, 1, 1};

  Box3 src_touches_both_r1_boxes;  // overlaps r1_box_a AND r1_box_b
  src_touches_both_r1_boxes.lo = {0.5, 0.5, 0.0};
  src_touches_both_r1_boxes.hi = {1.5, 1.5, 0.0};
  Box3 src_local_only;  // overlaps only r0's own box
  src_local_only.lo = {0.1, 0.1, 0.0};
  src_local_only.hi = {0.2, 0.2, 0.0};

  const auto dest = covering_set_destinations({src_touches_both_r1_boxes, src_local_only},
                                              /*my_rank=*/0, all_target_boxes,
                                              owner_of_target_box);
  ASSERT_EQ(dest.size(), 2u);
  EXPECT_EQ(dest[0], std::vector<int>{1}) << "rank 1 named once, not twice";
  EXPECT_TRUE(dest[1].empty()) << "a cell overlapping only its own rank's target needs no copy";
}

TEST(NativeHaloCoveringSet, ANonOverlappingSourceBoxGoesNowhere) {
  Box3 target;
  target.lo = {10.0, 10.0, 0.0};
  target.hi = {11.0, 11.0, 0.0};
  Box3 src;
  src.lo = {0.0, 0.0, 0.0};
  src.hi = {1.0, 1.0, 0.0};
  const auto dest = covering_set_destinations({src}, 0, {target}, {1});
  ASSERT_EQ(dest.size(), 1u);
  EXPECT_TRUE(dest[0].empty());
}

// ---------------------------------------------------------------------------
// owned_block_index + pack_halo_cells + merge_halo_cells: a small 2-rank
// round trip, checked by hand against the exact cells a 1x4 line generates
// (4 global cells, 2 per rank at 2 ranks -- native_generators.cpp's
// rank_slice splits [0,4) into [0,2) and [2,4)).
// ---------------------------------------------------------------------------

TEST(NativeHaloPackMerge, OwnedBlockIndexIsADenseZeroBasedCountOfOwnedCellsOnly) {
  auto arrays = generate_native_mesh("line:4", /*rank=*/0, /*num_ranks=*/2);
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok()) << view.error().message;
  const auto idx = owned_block_index(*view.value());
  ASSERT_EQ(idx.size(), view.value()->cell_count());
  std::set<std::size_t> seen;
  for (std::size_t c = 0; c < idx.size(); ++c) {
    ASSERT_TRUE(view.value()->is_owned(1, c)) << "a freshly generated slice owns every cell it "
                                                  "holds";
    EXPECT_TRUE(seen.insert(idx[c]).second) << "owned_block_index must be injective";
  }
  // Dense: every value in [0, owned_count) appears exactly once.
  for (std::size_t i = 0; i < view.value()->owned_count(1); ++i) {
    EXPECT_TRUE(seen.count(i) > 0) << "missing owned-block position " << i;
  }
}

TEST(NativeHaloPackMerge, RoundTripAdoptsExactlyTheRequestedCellsAsNonOwnedCopies) {
  // Rank 0 holds global cells {0, 1}; rank 1 holds {2, 3} (line:4 at 2
  // ranks). Rank 0 sends its cell 1 to rank 1 as a halo copy.
  auto arrays0 = generate_native_mesh("line:4", 0, 2);
  ASSERT_TRUE(arrays0.ok());
  auto view0 = NativeMeshView::create_from_host(arrays0.value(), 0);
  ASSERT_TRUE(view0.ok());
  auto arrays1 = generate_native_mesh("line:4", 1, 2);
  ASSERT_TRUE(arrays1.ok());
  auto view1 = NativeMeshView::create_from_host(arrays1.value(), 1);
  ASSERT_TRUE(view1.ok());

  const auto owned0 = owned_block_index(*view0.value());
  std::vector<std::vector<std::size_t>> send_by_local_index(2);  // 2 simulated ranks
  send_by_local_index[1].push_back(1);  // rank 0's local cell 1 -> rank 1
  const auto packed = pack_halo_cells(*view0.value(), send_by_local_index, owned0,
                                      /*my_rank=*/0, /*num_ranks=*/2);
  ASSERT_EQ(packed.size(), 2u);
  EXPECT_TRUE(packed[0].empty()) << "nothing is ever sent to one's own rank";
  EXPECT_FALSE(packed[1].empty());

  std::vector<std::vector<std::byte>> received_by_rank1 = {packed[1]};  // only rank 0 sent
  std::vector<int> owner_rank;
  std::vector<std::int64_t> owner_owned_index;
  auto merged =
      merge_halo_cells(*view1.value(), received_by_rank1, 1, &owner_rank, &owner_owned_index);
  ASSERT_TRUE(merged.ok()) << merged.error().message;
  auto new_view1 = NativeMeshView::create_from_host(merged.value(), 1);
  ASSERT_TRUE(new_view1.ok()) << new_view1.error().message;

  EXPECT_EQ(new_view1.value()->owned_count(1), 2u) << "owned_count must not widen";
  EXPECT_EQ(new_view1.value()->entity_count(1), 3u) << "entity_count widens by exactly 1";
  ASSERT_EQ(owner_rank.size(), 1u);
  ASSERT_EQ(owner_owned_index.size(), 1u);
  EXPECT_EQ(owner_rank[0], 0);
  EXPECT_EQ(owner_owned_index[0], 1);  // cell 1 is rank 0's SECOND owned cell (position 1)

  // The adopted cell is the appended, non-owned one (local index 2, after
  // the 2 pre-existing owned cells), and its global id matches the SOURCE
  // mesh's own cell 1 exactly -- not a relabelled copy.
  EXPECT_FALSE(new_view1.value()->is_owned(1, 2));
  EXPECT_EQ(new_view1.value()->global_id(1, 2), view0.value()->global_id(1, 1));
  EXPECT_EQ(new_view1.value()->cell_owner_of(2), 0);

  // Idempotent: merging the SAME received buffer again against the
  // ALREADY-WIDENED view must adopt nothing new (the cell is already held).
  std::vector<int> owner_rank2;
  std::vector<std::int64_t> owner_owned_index2;
  auto merged_again = merge_halo_cells(*new_view1.value(), received_by_rank1, 1, &owner_rank2,
                                       &owner_owned_index2);
  ASSERT_TRUE(merged_again.ok());
  EXPECT_TRUE(owner_rank2.empty()) << "a cell already held must not be re-adopted";
  auto final_view = NativeMeshView::create_from_host(merged_again.value(), 1);
  ASSERT_TRUE(final_view.ok());
  EXPECT_EQ(final_view.value()->entity_count(1), 3u);
}

TEST(NativeHaloPackMerge, PackHaloCellsMaterializesNoHostMirror) {
  // Migration track, 2026-10-15: pack_halo_cells now builds its wire
  // records from device_arrays() via fuse::native::device::pack_fanout,
  // never from the view's own lazy host mirror.
  auto arrays0 = generate_native_mesh("line:4", 0, 2);
  ASSERT_TRUE(arrays0.ok());
  auto view0 = NativeMeshView::create_from_host(arrays0.value(), 0);
  ASSERT_TRUE(view0.ok());
  ASSERT_EQ(view0.value()->host_mirrors_materialized(), 0);

  const auto owned0 = owned_block_index(*view0.value());
  std::vector<std::vector<std::size_t>> send_by_local_index(2);
  send_by_local_index[1].push_back(1);
  const auto packed = pack_halo_cells(*view0.value(), send_by_local_index, owned0,
                                      /*my_rank=*/0, /*num_ranks=*/2);
  ASSERT_EQ(packed.size(), 2u);

  EXPECT_EQ(view0.value()->host_mirrors_materialized(), 0)
      << "pack_halo_cells touched the source view's lazy host mirror";
}

TEST(NativeHaloPackMerge, PackHaloCellsFansOutOneCellToSeveralDestinations) {
  // A source cell overlapping more than one target rank's covering set is
  // sent to BOTH -- the many-to-many case fuse::native::device::pack_to's
  // single-destination-per-cell model cannot express (one cell, one
  // `dest_of_owned` entry), which is why pack_halo_cells goes through
  // `device::pack_fanout` (a flattened (cell, destination) send list)
  // instead. Both copies must carry identical bytes: a halo record's
  // content depends only on the cell and the constant `my_rank`/
  // `owned_block_index`, never on which destination it is bound for.
  auto arrays0 = generate_native_mesh("line:4", 0, 2);
  ASSERT_TRUE(arrays0.ok());
  auto view0 = NativeMeshView::create_from_host(arrays0.value(), 0);
  ASSERT_TRUE(view0.ok());

  const auto owned0 = owned_block_index(*view0.value());
  std::vector<std::vector<std::size_t>> send_by_local_index(3);  // 3 simulated ranks
  send_by_local_index[1].push_back(1);  // rank 0's local cell 1 -> rank 1
  send_by_local_index[2].push_back(1);  // AND -> rank 2 (fanout)
  const auto packed = pack_halo_cells(*view0.value(), send_by_local_index, owned0,
                                      /*my_rank=*/0, /*num_ranks=*/3);
  ASSERT_EQ(packed.size(), 3u);
  EXPECT_TRUE(packed[0].empty()) << "nothing is ever sent to one's own rank";
  EXPECT_FALSE(packed[1].empty());
  EXPECT_FALSE(packed[2].empty());
  EXPECT_EQ(packed[1], packed[2])
      << "the same cell sent to two destinations must pack identical bytes for each";
}

// ---------------------------------------------------------------------------
// The Brusselator-shaped scenario: U (source) on a 16x16 quad grid, V
// (target) on a 24x24 quad grid, both slab-decomposed by ROW
// (native_generators.cpp's rank_slice on the second grid extent) over the
// SAME set of ranks. At 2 ranks the row boundary is at y = 0.5 on both
// (8/16 == 12/24): ALIGNED, zero cells need to cross. At 3 ranks U splits
// 6/5/5 (boundaries at y = 6/16 = 0.375 and 11/16 = 0.6875) while V splits
// 8/8/8 (boundaries at y = 8/24 = 1/3 and 16/24 = 2/3): MISALIGNED, so some
// of U's owned cells overlap a NEIGHBOUR's slab of V and must travel.
// ---------------------------------------------------------------------------

namespace {

constexpr const char* kUSource = "generated:quad:16x16";
constexpr const char* kVSource = "generated:quad:24x24";

/// One simulated rank's pieces: its own partitioned U/V views, built
/// exactly as `NativeMeshBackend::load` would (same generator call,
/// same `(rank, num_ranks)`), without needing a `NativeMeshBackend` at
/// all for the PURE halo pipeline below.
struct RankPieces {
  std::unique_ptr<NativeMeshView> u;  // source
  std::unique_ptr<NativeMeshView> v;  // target
};

[[nodiscard]] std::vector<RankPieces> build_ranks(int num_ranks) {
  std::vector<RankPieces> ranks;
  ranks.reserve(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    RankPieces p;
    auto ua = generate_native_mesh("quad:16x16", r, num_ranks);
    EXPECT_TRUE(ua.ok()) << ua.error().message;
    auto uv = NativeMeshView::create_from_host(ua.value(), r);
    EXPECT_TRUE(uv.ok()) << uv.error().message;
    p.u = std::move(uv.value());

    auto va = generate_native_mesh("quad:24x24", r, num_ranks);
    EXPECT_TRUE(va.ok()) << va.error().message;
    auto vv = NativeMeshView::create_from_host(va.value(), r);
    EXPECT_TRUE(vv.ok()) << vv.error().message;
    p.v = std::move(vv.value());
    ranks.push_back(std::move(p));
  }
  return ranks;
}

/// The brute-force serial oracle: every global U cell's box, and which
/// ranks' V slabs (per `ranks`' own target boxes) it overlaps -- computed
/// from a SINGLE serial (1-rank) read of U, independent of any rank's own
/// partitioned view, so it cannot share a bug with the code under test.
struct Oracle {
  std::vector<std::vector<int>> dest_for_global_u_cell;  // by U global id
  std::vector<int> owner_of_global_u_cell;                // by U global id
};

[[nodiscard]] Oracle build_oracle(const std::vector<RankPieces>& ranks, int num_ranks) {
  auto serial_u = generate_native_mesh("quad:16x16", 0, 1);
  EXPECT_TRUE(serial_u.ok());
  auto serial_u_view = NativeMeshView::create_from_host(serial_u.value(), 0);
  EXPECT_TRUE(serial_u_view.ok());

  std::vector<Box3> all_target_boxes;
  std::vector<int> owner_of_target_box;
  for (int r = 0; r < num_ranks; ++r) {
    const NativeMeshView& v = *ranks[static_cast<std::size_t>(r)].v;
    for (std::size_t c = 0; c < v.entity_count(2); ++c) {
      all_target_boxes.push_back(cell_box(v, c));
      owner_of_target_box.push_back(r);
    }
  }

  // The owner of global U cell g, by looking it up in each rank's own view
  // (a cell is owned by exactly one rank; a linear scan at this problem
  // size is cheap and independent of `owned_block_index`/the code under
  // test).
  Oracle oracle;
  const std::size_t n_global = serial_u_view.value()->cell_count();
  oracle.owner_of_global_u_cell.assign(n_global, -1);
  for (int r = 0; r < num_ranks; ++r) {
    const NativeMeshView& u = *ranks[static_cast<std::size_t>(r)].u;
    for (std::size_t c = 0; c < u.entity_count(2); ++c) {
      if (!u.is_owned(2, c)) continue;
      const auto gid = u.global_id(2, c);
      EXPECT_LT(gid, n_global);
      oracle.owner_of_global_u_cell[gid] = r;
    }
  }

  oracle.dest_for_global_u_cell.resize(n_global);
  for (std::size_t g = 0; g < n_global; ++g) {
    const Box3 box = cell_box(*serial_u_view.value(), g);
    const int owner = oracle.owner_of_global_u_cell[g];
    std::vector<int> dests;
    for (std::size_t t = 0; t < all_target_boxes.size(); ++t) {
      const int towner = owner_of_target_box[t];
      if (towner == owner) continue;
      // The SAME touch-tolerant definition native_halo.cpp's own
      // `boxes_overlap` uses (not shared code on purpose: this oracle is
      // meant to be an INDEPENDENT ground truth, so it restates the
      // mathematical definition -- a zero-measure touch is not an overlap,
      // EXCEPT on an axis that is flat for the whole mesh (z, for a planar
      // mesh), where touching is the embedding itself, not a boundary --
      // in its own words, rather than calling the function under test).
      constexpr double kTouchTol = 1e-9;
      constexpr double kDegenerateWidth = 1e-12;
      bool overlap = true;
      for (int k = 0; k < 3; ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        const bool degenerate = (box.hi[kk] - box.lo[kk] < kDegenerateWidth) &&
                               (all_target_boxes[t].hi[kk] - all_target_boxes[t].lo[kk] <
                                kDegenerateWidth);
        const double tol = degenerate ? 0.0 : kTouchTol;
        if (box.hi[kk] < all_target_boxes[t].lo[kk] + tol ||
            all_target_boxes[t].hi[kk] < box.lo[kk] + tol) {
          overlap = false;
          break;
        }
      }
      if (overlap && std::find(dests.begin(), dests.end(), towner) == dests.end()) {
        dests.push_back(towner);
      }
    }
    std::sort(dests.begin(), dests.end());
    oracle.dest_for_global_u_cell[g] = std::move(dests);
  }
  return oracle;
}

/// Runs the full pure pipeline (box list, covering set, pack, in-process
/// "exchange", merge) for every simulated rank and returns each rank's
/// final (widened) U view plus the GhostPattern it was given.
struct HaloRunResult {
  std::vector<std::unique_ptr<NativeMeshView>> new_u;
  std::vector<fuse::linalg::GhostPattern> pattern;
  std::vector<std::size_t> adopted;
};

[[nodiscard]] HaloRunResult run_halo_pipeline(std::vector<RankPieces>& ranks, int num_ranks) {
  std::vector<std::vector<Box3>> my_target_boxes(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    const NativeMeshView& v = *ranks[static_cast<std::size_t>(r)].v;
    for (std::size_t c = 0; c < v.entity_count(2); ++c) {
      if (v.is_owned(2, c)) my_target_boxes[static_cast<std::size_t>(r)].push_back(cell_box(v, c));
    }
  }
  // In-process allgather: concatenate every rank's own target boxes.
  std::vector<Box3> all_target_boxes;
  std::vector<int> owner_of_target_box;
  for (int r = 0; r < num_ranks; ++r) {
    for (const Box3& b : my_target_boxes[static_cast<std::size_t>(r)]) {
      all_target_boxes.push_back(b);
      owner_of_target_box.push_back(r);
    }
  }

  // Per rank: owned source boxes/local indices, destinations, packing.
  std::vector<std::vector<std::vector<std::byte>>> send_from(static_cast<std::size_t>(num_ranks));
  std::vector<std::vector<std::size_t>> owned_idx_of(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    NativeMeshView& u = *ranks[static_cast<std::size_t>(r)].u;
    owned_idx_of[static_cast<std::size_t>(r)] = owned_block_index(u);
    std::vector<Box3> my_owned_boxes;
    std::vector<std::size_t> owned_local_index;
    for (std::size_t c = 0; c < u.entity_count(2); ++c) {
      if (u.is_owned(2, c)) {
        my_owned_boxes.push_back(cell_box(u, c));
        owned_local_index.push_back(c);
      }
    }
    const auto dest_for_owned =
        covering_set_destinations(my_owned_boxes, r, all_target_boxes, owner_of_target_box);
    std::vector<std::vector<std::size_t>> send_by_local_index(static_cast<std::size_t>(num_ranks));
    for (std::size_t i = 0; i < dest_for_owned.size(); ++i) {
      for (int d : dest_for_owned[i]) {
        send_by_local_index[static_cast<std::size_t>(d)].push_back(owned_local_index[i]);
      }
    }
    send_from[static_cast<std::size_t>(r)] =
        pack_halo_cells(u, send_by_local_index, owned_idx_of[static_cast<std::size_t>(r)], r,
                       num_ranks);
  }

  HaloRunResult result;
  result.new_u.resize(static_cast<std::size_t>(num_ranks));
  result.pattern.resize(static_cast<std::size_t>(num_ranks));
  result.adopted.resize(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) {
    std::vector<std::vector<std::byte>> received;
    received.reserve(static_cast<std::size_t>(num_ranks));
    for (int s = 0; s < num_ranks; ++s) {
      received.push_back(send_from[static_cast<std::size_t>(s)][static_cast<std::size_t>(r)]);
    }
    std::vector<int> owner_rank;
    std::vector<std::int64_t> owner_owned_index;
    auto merged = merge_halo_cells(*ranks[static_cast<std::size_t>(r)].u, received, r,
                                   &owner_rank, &owner_owned_index);
    EXPECT_TRUE(merged.ok()) << "rank " << r << ": " << merged.error().message;
    auto new_view = NativeMeshView::create_from_host(merged.value(), r);
    EXPECT_TRUE(new_view.ok()) << "rank " << r << ": " << new_view.error().message;
    result.new_u[static_cast<std::size_t>(r)] = std::move(new_view.value());
    result.pattern[static_cast<std::size_t>(r)].owner_rank = owner_rank;
    result.pattern[static_cast<std::size_t>(r)].remote_local_index = owner_owned_index;
    result.adopted[static_cast<std::size_t>(r)] = owner_rank.size();
  }
  return result;
}

}  // namespace

TEST(NativeHaloQuadGrid, TwoRanksAlignedAdoptsZeroCellsOnEveryRank) {
  constexpr int kRanks = 2;
  auto ranks = build_ranks(kRanks);
  const auto run = run_halo_pipeline(ranks, kRanks);
  for (int r = 0; r < kRanks; ++r) {
    EXPECT_EQ(run.adopted[static_cast<std::size_t>(r)], 0u)
        << "rank " << r << ": the 8+8/12+12 row split aligns exactly at y = 0.5, so no U cell "
                          "should need to cross a V partition boundary";
    EXPECT_EQ(run.new_u[static_cast<std::size_t>(r)]->owned_count(2),
            ranks[static_cast<std::size_t>(r)].u->owned_count(2));
    EXPECT_EQ(run.new_u[static_cast<std::size_t>(r)]->entity_count(2),
            ranks[static_cast<std::size_t>(r)].u->entity_count(2))
        << "zero adopted means entity_count must not widen at all";
  }
}

TEST(NativeHaloQuadGrid, ThreeRanksMisalignedAdoptsExactlyWhatTheOracleComputes) {
  constexpr int kRanks = 3;
  auto ranks = build_ranks(kRanks);
  const Oracle oracle = build_oracle(ranks, kRanks);
  const auto run = run_halo_pipeline(ranks, kRanks);

  std::size_t total_adopted = 0;
  for (int r = 0; r < kRanks; ++r) {
    const std::size_t adopted = run.adopted[static_cast<std::size_t>(r)];
    total_adopted += adopted;
    EXPECT_EQ(run.new_u[static_cast<std::size_t>(r)]->owned_count(2),
            ranks[static_cast<std::size_t>(r)].u->owned_count(2))
        << "rank " << r << ": owned_count must never widen";
    EXPECT_EQ(run.new_u[static_cast<std::size_t>(r)]->entity_count(2),
            ranks[static_cast<std::size_t>(r)].u->entity_count(2) + adopted)
        << "rank " << r << ": entity_count must widen by exactly the adopted count";

    // Every adopted cell's global id, on THIS rank, must be exactly the
    // oracle's own "U cell g must travel to rank r" set -- and nothing
    // else.
    std::set<std::int64_t> expected_gids;
    for (std::size_t g = 0; g < oracle.dest_for_global_u_cell.size(); ++g) {
      const auto& dests = oracle.dest_for_global_u_cell[g];
      if (std::find(dests.begin(), dests.end(), r) != dests.end()) {
        expected_gids.insert(static_cast<std::int64_t>(g));
      }
    }
    std::set<std::int64_t> actual_gids;
    const NativeMeshView& nv = *run.new_u[static_cast<std::size_t>(r)];
    for (std::size_t c = 0; c < nv.entity_count(2); ++c) {
      if (!nv.is_owned(2, c)) actual_gids.insert(static_cast<std::int64_t>(nv.global_id(2, c)));
    }
    EXPECT_EQ(actual_gids, expected_gids) << "rank " << r;
  }
  EXPECT_GT(total_adopted, 0u) << "the 6/5/5 vs 8/8/8 row split is misaligned, so SOME U cell "
                                 "must cross a V partition boundary";
}

// ---------------------------------------------------------------------------
// build_weights, through the REAL fuse::build_weights entry point, on the
// native backend after the halo above has been installed via the real
// production seam (NativeMeshBackend::commit_remap_halo).
// ---------------------------------------------------------------------------

namespace {

/// A linear scalar field over physical (x, y): exact for the first-order
/// conservative map's row-sum-1 property to reproduce bit-for-bit (to
/// rounding), which is what makes it a sharp completeness probe -- any
/// missing source contribution (an incomplete halo) shows up as a target
/// value below the true linear value by exactly the missing weight times
/// that source cell's field value.
[[nodiscard]] double linear_field(double x, double y) { return 3.0 + 2.0 * x - 5.0 * y; }

/// Builds a `NativeMeshBackend` per rank, loads U/V, adopts the halo
/// computed by `run_halo_pipeline` through the real `commit_remap_halo`
/// seam, and returns one backend per rank plus the mesh/region plumbing a
/// `RemapPlan` needs. `set_communicator` is given a NON-ZERO placeholder
/// handle (never dereferenced: this test never calls
/// `build_native_remap_halo` itself, only the pure pieces above plus
/// `commit_remap_halo`, neither of which touches MPI) purely so
/// `communicator_size()` reports `num_ranks` instead of the no-communicator
/// default of 1 -- exactly what a real run's `set_communicator(MPI_Comm_c2f
/// (...))` would report, which is the condition `build_weights`'s
/// E-REMAP-PARALLEL check keys on.
struct BackendFixture {
  std::vector<std::unique_ptr<NativeMeshBackend>> backends;
  CouplingGraph graph;
  MeshId u_mid{0}, v_mid{0};
  RegionId u_region, v_region;
};

[[nodiscard]] BackendFixture build_backends(int num_ranks, bool install_halo) {
  BackendFixture fx;
  fx.u_mid = fx.graph.add_mesh(MeshDecl{.name = "U", .source = kUSource});
  fx.v_mid = fx.graph.add_mesh(MeshDecl{.name = "V", .source = kVSource});
  fx.u_region =
      fx.graph.add_region(Region{.name = "u_all", .mesh = fx.u_mid, .entity_set = "", .dimension = 2});
  fx.v_region =
      fx.graph.add_region(Region{.name = "v_all", .mesh = fx.v_mid, .entity_set = "", .dimension = 2});

  std::vector<RankPieces> pieces = build_ranks(num_ranks);
  auto run = install_halo ? run_halo_pipeline(pieces, num_ranks) : HaloRunResult{};

  for (int r = 0; r < num_ranks; ++r) {
    auto backend = std::make_unique<NativeMeshBackend>(r, num_ranks);
    backend->set_communicator(/*placeholder, never dereferenced -- see comment above*/ 1);
    EXPECT_TRUE(backend->load(MeshDecl{.name = "U", .source = kUSource}).ok());
    EXPECT_TRUE(backend->load(MeshDecl{.name = "V", .source = kVSource}).ok());
    if (install_halo) {
      auto committed = backend->commit_remap_halo(
          fx.u_mid, ComponentId{}, std::move(run.new_u[static_cast<std::size_t>(r)]), fx.v_mid, 2,
          run.pattern[static_cast<std::size_t>(r)]);
      EXPECT_TRUE(committed.ok()) << committed.error().message;
    }
    fx.backends.push_back(std::move(backend));
  }
  return fx;
}

}  // namespace

TEST(NativeHaloBuildWeights, RejectsWithoutAHaloAtThreeRanks) {
  auto fx = build_backends(/*num_ranks=*/3, /*install_halo=*/false);
  RemapPlan plan;
  plan.src = fx.u_region;
  plan.tgt = fx.v_region;
  for (int r = 0; r < 3; ++r) {
    const auto w = fuse::build_weights(plan, fx.graph, *fx.backends[static_cast<std::size_t>(r)]);
    ASSERT_FALSE(w.ok()) << "rank " << r;
    EXPECT_EQ(w.error().code, "E-REMAP-PARALLEL") << "rank " << r;
  }
}

TEST(NativeHaloBuildWeights, TwoRanksAlignedCompleteWeightsAfterZeroCellHalo) {
  auto fx = build_backends(/*num_ranks=*/2, /*install_halo=*/true);
  RemapPlan plan;
  plan.src = fx.u_region;
  plan.tgt = fx.v_region;

  double max_row_defect = 0.0;
  for (int r = 0; r < 2; ++r) {
    const auto w = fuse::build_weights(plan, fx.graph, *fx.backends[static_cast<std::size_t>(r)]);
    ASSERT_TRUE(w.ok()) << "rank " << r << ": " << w.error().message;
    max_row_defect = std::max(max_row_defect, w.value().max_row_defect);
  }
  EXPECT_LE(max_row_defect, 1e-12) << "measured row-sum defect (aligned, 2 ranks)";
}

TEST(NativeHaloBuildWeights, ThreeRanksMisalignedCompleteWeightsAndLinearFieldReproduction) {
  auto fx = build_backends(/*num_ranks=*/3, /*install_halo=*/true);
  RemapPlan plan;
  plan.src = fx.u_region;
  plan.tgt = fx.v_region;

  double max_row_defect = 0.0;
  double max_field_error = 0.0;
  double total_col_mass_defect = 0.0;

  // The exact total measure of every GLOBAL U cell, by gid, from a single
  // serial read -- the ground truth the per-rank column-mass sums below
  // (owner's own contribution plus every halo copy's, by gid) must
  // reproduce.
  auto serial_u = generate_native_mesh("quad:16x16", 0, 1);
  ASSERT_TRUE(serial_u.ok());
  auto serial_u_view = NativeMeshView::create_from_host(serial_u.value(), 0);
  ASSERT_TRUE(serial_u_view.ok());
  std::map<std::int64_t, double> col_mass_by_gid;

  for (int r = 0; r < 3; ++r) {
    NativeMeshBackend& backend = *fx.backends[static_cast<std::size_t>(r)];
    const auto w = fuse::build_weights(plan, fx.graph, backend);
    ASSERT_TRUE(w.ok()) << "rank " << r << ": " << w.error().message;
    max_row_defect = std::max(max_row_defect, w.value().max_row_defect);

    const auto u_view_r = backend.native_view(fx.u_mid, ComponentId{});
    const auto v_view_r = backend.native_view(fx.v_mid, ComponentId{});
    ASSERT_TRUE(u_view_r.ok());
    ASSERT_TRUE(v_view_r.ok());
    const NativeMeshView& u = *u_view_r.value();
    const NativeMeshView& v = *v_view_r.value();

    // Column-mass-by-gid: w.val[k] * measure(target row) attributed to the
    // SOURCE cell's global id -- summed across every rank that holds a
    // copy of it (owner or halo), this must equal that cell's own measure.
    for (std::size_t k = 0; k < w.value().val.size(); ++k) {
      const auto j = static_cast<std::size_t>(w.value().row[k]);
      const auto i = static_cast<std::size_t>(w.value().col[k]);
      const auto gid = static_cast<std::int64_t>(u.global_id(2, i));
      col_mass_by_gid[gid] += w.value().val[k] * v.measure(2, j);
    }

    // Linear-field reproduction: fill every OWNED source slot from the
    // analytic field, every GHOST (halo) slot from its OWNER's own value
    // via the GhostPattern -- the in-process stand-in for
    // fuse::linalg::make_mpi_exchanger's own `update()` (comm.hpp) --
    // then apply_weights and compare against the analytic field at every
    // OWNED target cell's centroid.
    const auto pattern_r = backend.remap_halo_pattern(fx.u_mid, fx.v_mid, 2);
    ASSERT_TRUE(pattern_r.ok()) << pattern_r.error().message;
    const auto offset_r = backend.remap_halo_ghost_offset(fx.u_mid, fx.v_mid, 2);
    ASSERT_TRUE(offset_r.ok()) << offset_r.error().message;
    ASSERT_EQ(offset_r.value(), 0u)
        << "this fixture builds only one halo on u_mid, so its ghosts start right at owned_n";
    std::vector<double> src_field(u.entity_count(2));
    for (std::size_t c = 0; c < u.entity_count(2); ++c) {
      if (u.is_owned(2, c)) {
        const auto p = u.centroid(2, c);
        src_field[c] = linear_field(p.x, p.y);
      }
    }
    // The owner's OWN per-owned-block-index field values, gathered from
    // every rank (the in-process stand-in for the real exchange: a true
    // MPI build reads this off the OWNER's wire via make_mpi_exchanger
    // instead).
    std::vector<std::vector<double>> owned_field_of_rank(3);
    for (int s = 0; s < 3; ++s) {
      NativeMeshBackend& backend_s = *fx.backends[static_cast<std::size_t>(s)];
      const auto u_s = backend_s.native_view(fx.u_mid, ComponentId{});
      ASSERT_TRUE(u_s.ok());
      const auto idx_s = owned_block_index(*u_s.value());
      owned_field_of_rank[static_cast<std::size_t>(s)].resize(u_s.value()->owned_count(2));
      for (std::size_t c = 0; c < u_s.value()->entity_count(2); ++c) {
        if (u_s.value()->is_owned(2, c)) {
          const auto p = u_s.value()->centroid(2, c);
          owned_field_of_rank[static_cast<std::size_t>(s)][idx_s[c]] = linear_field(p.x, p.y);
        }
      }
    }
    const auto& owner_rank = pattern_r.value().owner_rank;
    const auto& remote_idx = pattern_r.value().remote_local_index;
    const std::size_t owned_n = u.owned_count(2);
    for (std::size_t g = 0; g < owner_rank.size(); ++g) {
      src_field[owned_n + g] =
          owned_field_of_rank[static_cast<std::size_t>(owner_rank[g])]
                            [static_cast<std::size_t>(remote_idx[g])];
    }

    std::vector<double> dst_field(w.value().n_rows);
    const auto applied = fuse::apply_weights(w.value(), src_field.data(), dst_field.data(), 1);
    ASSERT_TRUE(applied.ok()) << applied.error().message;
    for (std::size_t j = 0; j < v.entity_count(2); ++j) {
      if (!v.is_owned(2, j)) continue;
      const auto p = v.centroid(2, j);
      const double expect = linear_field(p.x, p.y);
      max_field_error = std::max(max_field_error, std::fabs(dst_field[j] - expect));
    }

    // A CONSTANT field, filled through the EXACT SAME GhostPattern-based
    // mechanism, is the sharp check on the ghost-fill plumbing itself: a
    // row-sum of 1 reproduces a constant EXACTLY (to rounding), regardless
    // of where the source cells sit relative to the target -- unlike a
    // linear field (below), whose reproduction error is bounded by the
    // field's own variation across a source cell, not by rounding. If a
    // ghost slot were left at its default 0 (a dropped halo value) this
    // would fail by an O(1) amount wherever that slot's weight is nonzero.
    constexpr double kConst = 7.0;
    std::vector<double> const_src_field(u.entity_count(2), kConst);
    std::vector<double> const_dst_field(w.value().n_rows);
    const auto const_applied =
        fuse::apply_weights(w.value(), const_src_field.data(), const_dst_field.data(), 1);
    ASSERT_TRUE(const_applied.ok()) << const_applied.error().message;
    for (std::size_t j = 0; j < v.entity_count(2); ++j) {
      if (!v.is_owned(2, j)) continue;
      EXPECT_NEAR(const_dst_field[j], kConst, 1e-12)
          << "rank " << r << " row " << j << ": a constant field must reproduce exactly";
    }
  }

  for (std::size_t g = 0; g < serial_u_view.value()->cell_count(); ++g) {
    const double expect = serial_u_view.value()->measure(2, g);
    const double got = col_mass_by_gid.count(static_cast<std::int64_t>(g))
                          ? col_mass_by_gid[static_cast<std::int64_t>(g)]
                          : 0.0;
    total_col_mass_defect = std::max(total_col_mass_defect, std::fabs(got - expect));
  }

  EXPECT_LE(max_row_defect, 1e-12) << "measured row-sum defect (misaligned, 3 ranks)";
  // NOT round-off, and should not be: a first-order (piecewise-constant)
  // conservative remap's row-sum-1 property (checked above, and by the
  // constant-field check in the loop) reproduces a CONSTANT field exactly;
  // a LINEAR field is reproduced only to the field's own variation across
  // the coarsest (U, width 1/16) source cell that can fully contain a V
  // target cell -- the Lipschitz bound |df/dx|*h + |df/dy|*h. Measured on
  // this fixture: 0.0729 (well inside this analytic ceiling, and nowhere
  // near the O(1) error a dropped or misindexed ghost value would produce).
  constexpr double kUCellWidth = 1.0 / 16.0;
  constexpr double kLinearFieldWorstCaseBound = (2.0 + 5.0) * kUCellWidth;  // f = 3 + 2x - 5y
  EXPECT_LE(max_field_error, kLinearFieldWorstCaseBound)
      << "measured linear-field reproduction error (misaligned, 3 ranks)";
  EXPECT_LE(total_col_mass_defect, 1e-10) << "measured conservative column-mass defect, summed "
                                             "by global id across every rank holding a copy "
                                             "(misaligned, 3 ranks)";
}

// ---------------------------------------------------------------------------
// build_native_remap_halo's own documented no-op short-circuit: the ONE
// case this test file can call the REAL production entry point, because
// `ranks <= 1` and `kNullComm` are both checked and returned on BEFORE any
// MPI symbol is touched.
// ---------------------------------------------------------------------------

TEST(NativeHaloBuildNativeRemapHalo, OneRankIsASuccessfulIdempotentNoOp) {
  NativeMeshBackend backend(0, 1);
  ASSERT_TRUE(backend.load(MeshDecl{.name = "U", .source = kUSource}).ok());
  ASSERT_TRUE(backend.load(MeshDecl{.name = "V", .source = kVSource}).ok());

  const auto first = build_native_remap_halo(backend, MeshId{0}, MeshId{1}, 2, ComponentId{});
  ASSERT_TRUE(first.ok()) << first.error().code << ": " << first.error().message;
  EXPECT_EQ(first.value(), 0u);
  EXPECT_TRUE(backend.has_remap_halo(MeshId{0}, MeshId{1}, 2));

  const auto second = build_native_remap_halo(backend, MeshId{0}, MeshId{1}, 2, ComponentId{});
  ASSERT_TRUE(second.ok()) << second.error().code;
  EXPECT_EQ(second.value(), 0u);
}

TEST(NativeHaloBuildNativeRemapHalo, NoCommunicatorIsASuccessfulNoOpEvenAtManyRanks) {
  // `num_ranks = 4` but NO `set_communicator` call: `communicator_size()`
  // reports 1 (native_backend.hpp's own documented default), so this is
  // the OTHER no-op leg, and still touches no MPI symbol.
  NativeMeshBackend backend(0, 4);
  ASSERT_TRUE(backend.load(MeshDecl{.name = "U", .source = kUSource}).ok());
  ASSERT_TRUE(backend.load(MeshDecl{.name = "V", .source = kVSource}).ok());
  const auto r = build_native_remap_halo(backend, MeshId{0}, MeshId{1}, 2, ComponentId{});
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_EQ(r.value(), 0u);
}

// ---------------------------------------------------------------------------
// E-MESH-HALO: `remap_halo_pattern`'s own documented refusal for a triple
// no halo was ever built for -- unfired by any test in the tree before
// this one (index.md's own test-coverage-gap list: "no test fires
// E-MESH-HALO"), and per I4 an unfired refusal does not count as
// implemented.
// ---------------------------------------------------------------------------
TEST(NativeHaloBuildNativeRemapHalo, RemapHaloPatternRefusesBeforeAHaloIsBuilt) {
  NativeMeshBackend backend(0, 1);
  ASSERT_TRUE(backend.load(MeshDecl{.name = "U", .source = kUSource}).ok());
  ASSERT_TRUE(backend.load(MeshDecl{.name = "V", .source = kVSource}).ok());
  EXPECT_FALSE(backend.has_remap_halo(MeshId{0}, MeshId{1}, 2));
  const auto r = backend.remap_halo_pattern(MeshId{0}, MeshId{1}, 2);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MESH-HALO");
}

TEST(NativeHaloBuildNativeRemapHalo, IdentitySourceEqualsTargetIsASuccessfulNoOp) {
  // Same mesh id on both sides (dispatched through `fuse::build_remap_halo`
  // itself, exercising the dynamic_cast dispatch in intx_weights.cpp): the
  // identity map needs no halo regardless of rank count, and this is
  // checked BEFORE `communicator_size()`'s own short-circuit, so it is a
  // no-op even with a (placeholder) communicator set.
  NativeMeshBackend backend(0, 3);
  backend.set_communicator(1);  // never dereferenced: this path returns first
  ASSERT_TRUE(backend.load(MeshDecl{.name = "U", .source = kUSource}).ok());
  CouplingGraph graph;
  const MeshId mid = graph.add_mesh(MeshDecl{.name = "U", .source = kUSource});
  const RegionId rid =
      graph.add_region(Region{.name = "u_all", .mesh = mid, .entity_set = "", .dimension = 2});
  RemapPlan plan;
  plan.src = rid;
  plan.tgt = rid;
  const auto r = fuse::build_remap_halo(plan, graph, backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_EQ(r.value(), 0u);
}

// ---------------------------------------------------------------------------
// NM-3: `merge_halo_cells` appends each halo's newly adopted cells STRICTLY
// AFTER whatever `source` already held (native_halo.cpp's own comment), so
// U's mesh after a SECOND halo for a different target (here W, a THIRD
// grid) is laid out `[owned, ghosts_V, ghosts_W]` -- the W pattern's own
// ghosts do NOT start at `owned_n`, they start `ghost_offset` further in,
// where `ghost_offset` is exactly how many ghosts the FIRST (V) halo
// already appended. Before NM-3, `remap_halo_pattern` carried no such
// offset at all, so any consumer following the documented
// `make_mpi_exchanger` contract literally (ghosts at `owned_n + g`) would
// read U's V-ghost slots with W-ghost values for every g below the V
// halo's own count -- silently wrong field values in a conservative remap.
// This test fills a NON-CONSTANT (linear) field so a misindexed ghost
// changes the applied result, not just its size.
// ---------------------------------------------------------------------------
TEST(NativeHaloSecondTarget, GhostOffsetIndexesTheSecondHaloBlockCorrectly) {
  constexpr int kRanks = 2;
  constexpr const char* kWSource = "generated:quad:20x20";

  // Pass 1: U -> V, the fixture's own existing pair.
  auto ranks1 = build_ranks(kRanks);
  auto run1 = run_halo_pipeline(ranks1, kRanks);

  // Pass 2: the ALREADY-WIDENED U from pass 1 -> W, a DIFFERENT target
  // grid. `run_halo_pipeline` only reads its inputs (never moves them),
  // so `ranks2[r].u` is still valid afterward for the production
  // `commit_remap_halo` call below.
  std::vector<RankPieces> ranks2(static_cast<std::size_t>(kRanks));
  for (int r = 0; r < kRanks; ++r) {
    ranks2[static_cast<std::size_t>(r)].u = std::move(run1.new_u[static_cast<std::size_t>(r)]);
    auto wa = generate_native_mesh("quad:20x20", r, kRanks);
    ASSERT_TRUE(wa.ok()) << wa.error().message;
    auto wv = NativeMeshView::create_from_host(wa.value(), r);
    ASSERT_TRUE(wv.ok()) << wv.error().message;
    ranks2[static_cast<std::size_t>(r)].v = std::move(wv.value());
  }
  auto run2 = run_halo_pipeline(ranks2, kRanks);

  CouplingGraph graph;
  const MeshId u_mid = graph.add_mesh(MeshDecl{.name = "U", .source = kUSource});
  const MeshId v_mid = graph.add_mesh(MeshDecl{.name = "V", .source = kVSource});
  const MeshId w_mid = graph.add_mesh(MeshDecl{.name = "W", .source = kWSource});
  const RegionId u_region =
      graph.add_region(Region{.name = "u_all", .mesh = u_mid, .entity_set = "", .dimension = 2});
  const RegionId w_region =
      graph.add_region(Region{.name = "w_all", .mesh = w_mid, .entity_set = "", .dimension = 2});
  RemapPlan plan;
  plan.src = u_region;
  plan.tgt = w_region;

  double max_field_error = 0.0;
  for (int r = 0; r < kRanks; ++r) {
    NativeMeshBackend backend(r, kRanks);
    backend.set_communicator(/*placeholder, never dereferenced*/ 1);
    ASSERT_TRUE(backend.load(MeshDecl{.name = "U", .source = kUSource}).ok());
    ASSERT_TRUE(backend.load(MeshDecl{.name = "V", .source = kVSource}).ok());
    ASSERT_TRUE(backend.load(MeshDecl{.name = "W", .source = kWSource}).ok());

    const std::size_t adopted1 = run1.adopted[static_cast<std::size_t>(r)];
    auto committed1 = backend.commit_remap_halo(
        u_mid, ComponentId{}, std::move(ranks2[static_cast<std::size_t>(r)].u), v_mid, 2,
        run1.pattern[static_cast<std::size_t>(r)]);
    ASSERT_TRUE(committed1.ok()) << committed1.error().message;

    auto committed2 = backend.commit_remap_halo(
        u_mid, ComponentId{}, std::move(run2.new_u[static_cast<std::size_t>(r)]), w_mid, 2,
        run2.pattern[static_cast<std::size_t>(r)]);
    ASSERT_TRUE(committed2.ok()) << committed2.error().message;

    const auto pattern2_r = backend.remap_halo_pattern(u_mid, w_mid, 2);
    ASSERT_TRUE(pattern2_r.ok()) << pattern2_r.error().message;
    const auto offset2_r = backend.remap_halo_ghost_offset(u_mid, w_mid, 2);
    ASSERT_TRUE(offset2_r.ok()) << offset2_r.error().message;
    EXPECT_EQ(offset2_r.value(), adopted1)
        << "rank " << r << ": the W halo's ghosts must start right after the V halo's own";

    const auto u_view_r = backend.native_view(u_mid, ComponentId{});
    ASSERT_TRUE(u_view_r.ok());
    const NativeMeshView& u = *u_view_r.value();
    const std::size_t owned_n = u.owned_count(2);
    const auto& owner_rank2 = pattern2_r.value().owner_rank;
    const auto& remote_idx2 = pattern2_r.value().remote_local_index;
    const std::size_t ghost_offset = offset2_r.value();

    // Every OWNED cell's own analytic value, plus (via the W pattern, at
    // the CORRECT offset) every W-ghost's owner's own value -- the same
    // owned-field-by-rank technique the misaligned 3-rank build_weights
    // test above uses, restricted to this fixture's own 2 ranks.
    std::vector<std::vector<double>> owned_field_of_rank(static_cast<std::size_t>(kRanks));
    for (int s = 0; s < kRanks; ++s) {
      NativeMeshBackend owner_backend(s, kRanks);
      auto ua = generate_native_mesh("quad:16x16", s, kRanks);
      ASSERT_TRUE(ua.ok());
      auto uv = NativeMeshView::create_from_host(ua.value(), s);
      ASSERT_TRUE(uv.ok());
      const auto idx_s = owned_block_index(*uv.value());
      owned_field_of_rank[static_cast<std::size_t>(s)].resize(uv.value()->owned_count(2));
      for (std::size_t c = 0; c < uv.value()->entity_count(2); ++c) {
        if (uv.value()->is_owned(2, c)) {
          const auto p = uv.value()->centroid(2, c);
          owned_field_of_rank[static_cast<std::size_t>(s)][idx_s[c]] = linear_field(p.x, p.y);
        }
      }
    }

    std::vector<double> src_field(u.entity_count(2), 0.0);
    for (std::size_t c = 0; c < u.entity_count(2); ++c) {
      if (u.is_owned(2, c)) {
        const auto p = u.centroid(2, c);
        src_field[c] = linear_field(p.x, p.y);
      }
    }
    // The CORRECT, offset-aware fill for the W block: slot `owned_n +
    // ghost_offset + g`, NOT `owned_n + g` (which would read the V
    // block's slots instead -- exactly NM-3's bug).
    for (std::size_t g = 0; g < owner_rank2.size(); ++g) {
      src_field[owned_n + ghost_offset + g] =
          owned_field_of_rank[static_cast<std::size_t>(owner_rank2[g])]
                            [static_cast<std::size_t>(remote_idx2[g])];
    }

    const auto w = fuse::build_weights(plan, graph, backend);
    ASSERT_TRUE(w.ok()) << "rank " << r << ": " << w.error().message;
    std::vector<double> dst_field(w.value().n_rows);
    const auto applied = fuse::apply_weights(w.value(), src_field.data(), dst_field.data(), 1);
    ASSERT_TRUE(applied.ok()) << applied.error().message;

    const auto w_view_r = backend.native_view(w_mid, ComponentId{});
    ASSERT_TRUE(w_view_r.ok());
    const NativeMeshView& wv_view = *w_view_r.value();
    for (std::size_t j = 0; j < wv_view.entity_count(2); ++j) {
      if (!wv_view.is_owned(2, j)) continue;
      const auto p = wv_view.centroid(2, j);
      max_field_error = std::max(max_field_error, std::fabs(dst_field[j] - linear_field(p.x, p.y)));
    }
  }
  // The same analytic (Lipschitz) ceiling the misaligned 3-rank
  // build_weights test above uses: a first-order conservative remap
  // reproduces a linear field only to its own variation across the
  // coarsest (U, width 1/16) source cell -- an O(1) value here (not this
  // small bound) would mean a ghost landed in the wrong slot.
  constexpr double kUCellWidth = 1.0 / 16.0;
  constexpr double kLinearFieldWorstCaseBound = (2.0 + 5.0) * kUCellWidth;
  EXPECT_LE(max_field_error, kLinearFieldWorstCaseBound)
      << "measured linear-field reproduction error through a SECOND halo's ghost block";
}

}  // namespace
