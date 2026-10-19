// test/parallel/moab_placed_weights_mpi_test.cpp -- KNOWN_ISSUES D5, the
// last remaining half: per-placement remap WEIGHTS across two
// INDEPENDENTLY decomposed MOAB processor sets, closed by HALO BRIDGING
// (the lead's round-2 ruling, consistent with the owner's 2026-10-18 D73
// decision to build remap halos automatically): `RemapPlan::placement`
// stays the TARGET's own placement; the source side is brought onto the
// target's ranks by a remap halo built from the SOURCE's own placement
// (`fuse::build_remap_halo`'s new 4-argument overload, remap_plan.hpp),
// so a subsequent `build_weights(plan, graph, backend)` resolves BOTH
// sides through `plan.placement` alone, exactly as the single-placement
// case already does.
//
// GENERATED MESHES, not the real sphere fixture -- see this file's own
// previous revision's note (test/parallel/moab_placed_view_mpi_test.cpp's
// header: a generated mesh under a communicator does not actually
// PARTITION) for why that limitation does not matter for what THIS file
// measures: whether the halo-bridging overload can resolve a source and a
// target through two DIFFERENT, rank-disjoint placements at all.
//
// MEASURED FINDING, round 2 (read AND built, not yet host-run -- see this
// file's own "NOT YET RUN" paragraph): `MoabMeshBackend::set_placements`
// gives every declared processor set its OWN `moab::Core` (moab_backend.cpp's
// `PlacementContext::core`), and MOAB's own covering-set construction
// (`Intx2MeshOnSphere`/`FindMaxEdges`/`construct_covering_set`,
// src/backend/intx_weights.cpp) needs BOTH meshes in the SAME
// `moab::Interface` -- there is no MOAB entry point that intersects a file
// set living in one Core against a file set living in another. Bridging two
// DIFFERENT per-placement Cores therefore needs a migrate-by-global-id
// exchange (the MOAB analogue of `fuse::native::build_native_remap_halo`'s
// own box-overlap/pack/merge architecture, native_halo.cpp) that is NOT YET
// BUILT. `fuse::build_remap_halo`'s new 4-argument overload
// (`include/fuse/backend/remap_plan.hpp`, `src/backend/intx_weights.cpp`)
// therefore does exactly three things, all measured here: (1) when the
// calling rank is structurally outside the TARGET's own placement, a
// successful no-op (checked via `MoabMeshBackend::placement_participates`,
// not a local size); (2) when the source and target placements happen to
// resolve to the SAME `moab::Core` (the degenerate case: no placements
// declared, or an identical processor set given for both), it delegates to
// the EXISTING, already-tested single-placement `build_remap_halo`, and
// this row checks that delegation reproduces the single-placement answer
// exactly; (3) when the two Cores genuinely DIFFER (a true cross-placement
// pair), an explicit, named refusal (`E-MESH-HALO-CROSS-PLACEMENT-
// UNIMPLEMENTED`) rather than an incomplete or silently wrong halo.
// KNOWN_ISSUES.md D5 records this as the measured, built-but-incomplete
// state; closing it needs the migrate-by-global-id mechanism above, which
// this track's own report flags as not completed in the time available.
//
// NOT YET RUN: this sandbox cannot initialise MPI at all (docs/dev/
// testing.md; KNOWN_ISSUES.md C1), and a build with FUSE_ENABLE_MOAB AND
// FUSE_ENABLE_MPI links against an MPI-enabled MOAB whose moab::Core
// constructor itself calls MPI_Init_thread, so even a single-process run
// of this binary aborts before main() runs a single test -- the identical
// constraint test/parallel/moab_placed_view_mpi_test.cpp's own header
// states. Compiled here; the outcome comes from a host run.

#include <cstdio>
#include <vector>

#include <mpi.h>

#include <gtest/gtest.h>

#include "fuse/backend/moab_backend.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/model/coupling_graph.hpp"

#include "fuse_test_main.hpp"

namespace {

int world_handle() { return static_cast<int>(MPI_Comm_c2f(MPI_COMM_WORLD)); }

/// Splits `size` ranks into two DISJOINT, non-empty rank sets: A gets
/// `[0, size/2)`, B gets `[size/2, size)` -- at `size == 2`, A = {0},
/// B = {1}; at `size == 3`, A = {0}, B = {1, 2}. Neither set is ever empty
/// at the registered rank counts (2, 3).
struct DisjointRankSets {
  fuse::RankSet a;
  fuse::RankSet b;

  explicit DisjointRankSets(int size) {
    const int split = size / 2;
    std::vector<int> ranks_a, ranks_b;
    for (int r = 0; r < split; ++r) ranks_a.push_back(r);
    for (int r = split; r < size; ++r) ranks_b.push_back(r);
    a = fuse::RankSet(ranks_a);
    b = fuse::RankSet(ranks_b);
  }
};

/// Box geometry (try_boxes, intx_weights.cpp), NOT the real sphere fixture.
/// Index 0 -> A (the SOURCE's own placement), index 1 -> B (the TARGET's).
struct DisjointPlacementFixture {
  fuse::MoabMeshBackend backend;
  fuse::CouplingGraph graph;
  fuse::MeshId smid, tmid;
  fuse::RemapPlan plan;  // plan.placement = B, the target's own

  explicit DisjointPlacementFixture(int size) {
    const DisjointRankSets sets(size);
    EXPECT_TRUE(backend.set_communicator(world_handle()).ok());
    EXPECT_TRUE(backend.set_placements({sets.a, sets.b}).ok());

    const char* src = "generated:box:4x4x4";
    const char* tgt = "generated:box:2x2x2";
    EXPECT_TRUE(backend.load({.name = "s", .source = src}).ok());
    EXPECT_TRUE(backend.load({.name = "t", .source = tgt}).ok());
    smid = graph.add_mesh({.name = "s", .source = src});
    tmid = graph.add_mesh({.name = "t", .source = tgt});
    const fuse::RegionId srid =
        graph.add_region({.name = "s_all", .mesh = smid, .dimension = 3});
    const fuse::RegionId trid =
        graph.add_region({.name = "t_all", .mesh = tmid, .dimension = 3});
    plan.src = srid;
    plan.tgt = trid;
    plan.placement = fuse::ComponentId{1};  // B, the target's own placement
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// (1) A rank structurally outside the TARGET's own placement (B): a
// successful no-op, not a refusal -- `placement_participates` derives this
// from the DECLARED processor set, never from a local entity count.
// ---------------------------------------------------------------------------
TEST(MoabPlacedWeightsMpi, ARankOutsideTheTargetPlacementIsASuccessfulNoOp) {
  int size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ASSERT_GE(size, 2) << "this row is only meaningful with more than one rank";
  DisjointPlacementFixture f(size);
  const DisjointRankSets sets(size);

  const auto r = fuse::build_remap_halo(f.plan, f.graph, f.backend, fuse::ComponentId{0});
  // Printed before any assertion (host instruction, round 3 FAIL 2): a
  // non-uniform per-rank failure here previously surfaced only as rank 0's
  // own "failed on rank(s) ..." line (fuse_test_main.cpp's MpiFailureReducer),
  // with the failing rank's own code/message absent from the captured log
  // because -outfile-pattern only kept rank 0's stdout.
  std::fprintf(stderr, "rank %d (in B=%d): build_remap_halo(source_placement=A) ok=%d%s%s\n",
              rank, sets.b.contains(rank) ? 1 : 0, r.ok() ? 1 : 0,
              r.ok() ? "" : " code=", r.ok() ? "" : r.error().code.c_str());
  if (!sets.b.contains(rank)) {
    // This row's own title and header comment: a rank OUTSIDE the target's
    // placement B must see a successful no-op. That is all this row checks.
    ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
    EXPECT_EQ(r.value(), 0u) << "rank " << rank << ": outside the target's own placement, "
                                                   "nothing to adopt";
    return;
  }
  // A rank INSIDE B, bridging a genuinely DISJOINT source placement A: the
  // design this file's own header documents refuses this with the named
  // cross-placement gap (checked explicitly by
  // DisjointSourceAndTargetPlacementsRefuseWithTheNamedCrossPlacementGap
  // below) -- it is not a no-op, so nothing further is asserted about it
  // here. Measured 2026-10-19 (round 3 FAIL 2): a prior revision of this
  // row asserted `r.ok()` for EVERY rank, including ranks inside B, which
  // is the refusal case by design -- that assertion was simply wrong for
  // this fixture, not a sign of a defect in the overload under test.
}

// ---------------------------------------------------------------------------
// (2) The degenerate same-Core case: source placement A and target
// placement B are both declared, but this row asks for the SAME placement
// on both sides (a RemapPlan whose own source_placement == plan.placement),
// which must delegate to, and reproduce exactly, the EXISTING
// single-placement build_remap_halo -- the D5 item already closed
// 2026-09-25, checked here as a regression guard on the new overload's
// own "same Core" branch.
// ---------------------------------------------------------------------------
TEST(MoabPlacedWeightsMpi, SamePlacementOnBothSidesDelegatesToTheSinglePlacementPath) {
  int size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ASSERT_GE(size, 2) << "this row is only meaningful with more than one rank";
  DisjointPlacementFixture f(size);
  f.plan.placement = fuse::ComponentId{0};  // both sides through A this time
  const DisjointRankSets sets(size);

  const auto via_overload = fuse::build_remap_halo(f.plan, f.graph, f.backend, fuse::ComponentId{0});
  std::fprintf(stderr, "rank %d (in A=%d): via_overload ok=%d%s%s\n", rank,
              sets.a.contains(rank) ? 1 : 0, via_overload.ok() ? 1 : 0,
              via_overload.ok() ? "" : " code=",
              via_overload.ok() ? "" : via_overload.error().code.c_str());
  if (!sets.a.contains(rank)) {
    // The overload's own structural no-op (placement_participates) must
    // still succeed off A -- checked here, but NOT compared against
    // via_single below: the pre-existing single-placement path was never
    // designed to tolerate a rank outside plan.placement (its own
    // resolve() call defaults allow_empty=false and refuses E-REMAP-REGION
    // there by design, unchanged by this track) -- measured 2026-10-19
    // (round 3 FAIL 2): a prior revision of this row called via_single
    // unconditionally and asserted it `ok()` on every rank, which fails
    // off A for exactly this reason, not because of a defect in the new
    // overload the row's own title names.
    ASSERT_TRUE(via_overload.ok()) << via_overload.error().code << ": "
                                   << via_overload.error().message;
    EXPECT_EQ(via_overload.value(), 0u) << "rank " << rank << ": outside A, nothing to adopt";
    return;
  }
  ASSERT_TRUE(via_overload.ok()) << via_overload.error().code << ": "
                                 << via_overload.error().message;
  const auto via_single = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  std::fprintf(stderr, "rank %d: via_single ok=%d%s%s\n", rank, via_single.ok() ? 1 : 0,
              via_single.ok() ? "" : " code=",
              via_single.ok() ? "" : via_single.error().code.c_str());
  ASSERT_TRUE(via_single.ok()) << via_single.error().code << ": " << via_single.error().message;
  EXPECT_EQ(via_overload.value(), via_single.value())
      << "the same-Core delegation must reproduce the single-placement answer exactly";
}

// ---------------------------------------------------------------------------
// (3) The genuinely cross-placement case this item measures: source on A,
// target on B, A and B DISJOINT -- an explicit, specific refusal, not a
// hang, not an incomplete halo. This is the row KNOWN_ISSUES.md D5 records
// as the measured, built-but-incomplete state.
// ---------------------------------------------------------------------------
TEST(MoabPlacedWeightsMpi, DisjointSourceAndTargetPlacementsRefuseWithTheNamedCrossPlacementGap) {
  int size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ASSERT_GE(size, 2) << "this row is only meaningful with more than one rank";
  DisjointPlacementFixture f(size);
  const DisjointRankSets sets(size);

  const auto r = fuse::build_remap_halo(f.plan, f.graph, f.backend, fuse::ComponentId{0});
  if (sets.b.contains(rank)) {
    ASSERT_FALSE(r.ok()) << "rank " << rank
                         << ": a target-placement rank asking to bridge a genuinely "
                            "disjoint source placement must refuse, not silently produce "
                            "an incomplete halo";
    EXPECT_EQ(r.error().code, "E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED") << "rank " << rank;
  } else {
    ASSERT_TRUE(r.ok()) << "rank " << rank << ": outside the target placement, this call is a "
                                              "no-op -- " << r.error().code << ": "
                       << r.error().message;
  }
}
