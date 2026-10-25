// test/parallel/moab_placed_weights_mpi_test.cpp -- per-placement remap
// WEIGHTS across two INDEPENDENTLY decomposed MOAB processor sets, by HALO
// BRIDGING: `RemapPlan::placement` stays the TARGET's own placement; the
// source side is brought onto the target's ranks by a remap halo built from
// the SOURCE's own placement (`fuse::build_remap_halo`'s 4-argument
// overload, remap_plan.hpp), so a subsequent `build_weights(plan, graph,
// backend)` resolves BOTH sides through `plan.placement` alone, exactly as
// the single-placement case already does.
//
// THE REAL, NON-MATCHING SPHERE PAIR (FUSE_SPHERE_ATM/FUSE_SPHERE_OCN,
// test/CMakeLists.txt), NOT a generated box mesh. MOAB's own covering-set
// construction (Intx2MeshOnSphere/FindMaxEdges/construct_covering_set,
// src/backend/intx_weights.cpp) is declared on dimension-2 SURFACES ONLY
// (the SAME restriction the single-placement `build_remap_halo` already
// states and tests, remap_halo_test.cpp/remap_halo_mpi_test.cpp), so a
// dimension-3 fixture hits E-MESH-HALO-GEOMETRY the moment ANY path through
// this file's own overload reaches a REAL covering-set attempt (the
// same-Core delegation, test (2) below), before the cross-placement logic
// under test is ever exercised. The production couplings this exists for
// (Brusselator, dn_heat) are both 2-D surfaces. `BoxPairMpi`
// (remap_halo_mpi_test.cpp) deliberately uses a non-spherical box pair to
// test E-MESH-HALO-GEOMETRY itself, which makes it the wrong fixture for a
// test that needs a REAL halo to actually build. This file therefore uses
// the SAME non-matching ATM/OCN sphere pair `RemapHaloMpiSphere`'s own
// tests use under a real communicator (parallel.remap_halo_mpi.{1..5}),
// gated identically (`#ifdef FUSE_SPHERE_ATM`, which CMake defines only
// when BOTH wholeATM_T.h5m and recMeshOcn.h5m were found), with the same
// `#else` GTEST_SKIP fallback moab_placed_view_mpi_test.cpp uses when they
// are not.
//
// WHAT THE 4-ARGUMENT OVERLOAD DOES. `MoabMeshBackend::set_placements`
// gives every declared processor set its OWN `moab::Core`
// (moab_backend.cpp's `PlacementContext::core`), and MOAB's own
// covering-set construction needs BOTH meshes in the SAME
// `moab::Interface` -- there is no MOAB entry point that intersects a
// file set living in one Core against a file set living in another.
// Bridging two DIFFERENT per-placement Cores would therefore need a
// migrate-by-global-id exchange (the MOAB analogue of
// `fuse::native::build_native_remap_halo`'s own box-overlap/pack/merge
// architecture, native_halo.cpp), which is not implemented.
// `fuse::build_remap_halo`'s 4-argument overload
// (`include/fuse/backend/remap_plan.hpp`, `src/backend/intx_weights.cpp`)
// therefore does exactly three things: (1) when the calling rank is
// structurally outside the TARGET's own placement, a successful no-op
// (checked via `MoabMeshBackend::placement_participates`, not a local
// size); (2) when the source and target placements happen to resolve to
// the SAME `moab::Core` (the degenerate case: no placements declared, or
// an identical processor set given for both), it delegates to the existing
// single-placement `build_remap_halo`, and this row checks that delegation
// reproduces the single-placement answer exactly, against a REAL halo
// build between two genuinely dimension-2 meshes; (3) when the two Cores
// genuinely DIFFER (a true cross-placement pair), an explicit, named
// refusal (`E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED`) rather than an
// incomplete or silently wrong halo.
//
// On an in-B rank, E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED is the
// expected answer for the genuinely disjoint pair: test (3) below is
// EXACTLY the row that asserts this refusal on every in-B rank, by design.
// Test (2)'s `ASSERT_TRUE(via_overload.ok())` concerns only its own
// same-Core case.
//
// Needs a working MPI launcher (docs/dev/testing.md). A build with
// FUSE_ENABLE_MOAB AND FUSE_ENABLE_MPI links against an MPI-enabled MOAB
// whose moab::Core constructor itself calls MPI_Init_thread, so even a
// single-process run of this binary aborts before main() runs a single
// test unless MPI can start -- the identical constraint
// test/parallel/moab_placed_view_mpi_test.cpp's own header states.

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
/// B = {1}; at `size == 3`, A = {0}, B = {1, 2}; at `size == 4`, A =
/// {0, 1}, B = {2, 3}. Neither set is ever empty at the registered rank
/// counts (2, 3, 4). `size == 4` is the FIRST registered row where a
/// placement's own sub-communicator (A, here) has MORE than one rank -- at
/// 2 and 3 ranks A is always exactly {0}, a single-rank ParallelComm that
/// takes a DIFFERENT code path (the degenerate "serial/one-rank/identity"
/// shortcut, intx_weights.cpp) than a genuine multi-rank covering-set
/// construction ever reaches, so nothing at 2 or 3 ranks measures the
/// configuration this exists for: a real coupling places SEVERAL ranks per
/// component.
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

}  // namespace

#ifdef FUSE_SPHERE_ATM

namespace {

/// The REAL, non-matching sphere pair `RemapHaloMpiSphere`'s own tests
/// already build a halo from successfully (remap_halo_mpi_test.cpp) --
/// atmosphere as the SOURCE, ocean as the TARGET, dimension 2 on both
/// sides. Index 0 -> A (the SOURCE's own placement), index 1 -> B (the
/// TARGET's).
struct DisjointPlacementFixture {
  fuse::MoabMeshBackend backend;
  fuse::CouplingGraph graph;
  fuse::MeshId smid, tmid;
  fuse::RemapPlan plan;  // plan.placement = B, the target's own

  explicit DisjointPlacementFixture(int size) {
    const DisjointRankSets sets(size);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    EXPECT_TRUE(backend.set_communicator(world_handle()).ok());
    EXPECT_TRUE(backend.set_placements({sets.a, sets.b}).ok());

    const char* src = FUSE_SPHERE_ATM;
    const char* tgt = FUSE_SPHERE_OCN;
    EXPECT_TRUE(backend.load({.name = "s", .source = src}).ok());
    EXPECT_TRUE(backend.load({.name = "t", .source = tgt}).ok());
    smid = graph.add_mesh({.name = "s", .source = src});
    tmid = graph.add_mesh({.name = "t", .source = tgt});

    // Rescaling is the DRIVER's explicit responsibility, never the
    // library's. `set_placements` gives placements A and B their own,
    // independently-read Core copies
    // of BOTH meshes, so each copy that this file's own three tests
    // resolve a halo through needs its OWN call: the source ("s", ATM)
    // through A -- every test below resolves `source_placement` as A --
    // and the target ("t", OCN) through BOTH A (test 2's own same-Core
    // case) and B (test 1/3's own target placement). Each call is
    // COLLECTIVE over that placement's own split communicator, so only
    // the ranks that participate in a given placement make it.
    if (sets.a.contains(rank)) {
      const auto pa = backend.project_onto_unit_sphere(smid, fuse::ComponentId{0});
      EXPECT_TRUE(pa.ok()) << pa.error().code << ": " << pa.error().message;
      const auto pt_a = backend.project_onto_unit_sphere(tmid, fuse::ComponentId{0});
      EXPECT_TRUE(pt_a.ok()) << pt_a.error().code << ": " << pt_a.error().message;
    }
    if (sets.b.contains(rank)) {
      const auto pt_b = backend.project_onto_unit_sphere(tmid, fuse::ComponentId{1});
      EXPECT_TRUE(pt_b.ok()) << pt_b.error().code << ": " << pt_b.error().message;
    }

    const fuse::RegionId srid =
        graph.add_region({.name = "s_all", .mesh = smid, .dimension = 2});
    const fuse::RegionId trid =
        graph.add_region({.name = "t_all", .mesh = tmid, .dimension = 2});
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
  // Printed before any assertion: a non-uniform per-rank failure otherwise
  // surfaces only as rank 0's own "failed on rank(s) ..." line
  // (fuse_test_main.cpp's MpiFailureReducer), with the failing rank's own
  // code/message absent from the captured log because -outfile-pattern only
  // keeps rank 0's stdout.
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
  // here.
}

// ---------------------------------------------------------------------------
// (2) The degenerate same-Core case: source placement A and target
// placement B are both declared, but this row asks for the SAME placement
// on both sides (a RemapPlan whose own source_placement == plan.placement),
// which must delegate to, and reproduce exactly, the EXISTING
// single-placement build_remap_halo, checked here as a regression guard on
// the new overload's own "same Core" branch.
//
// It asserts a SUCCESSFUL delegated halo build on the ADOPTED CELL COUNT.
// The fixture's two meshes have different radii (wholeATM_T.h5m = 1.0,
// recMeshOcn.h5m = 6371229.0, Earth radius in metres), so the fixture's
// own constructor calls the `project_onto_unit_sphere(MeshId, ComponentId)`
// overload on placement A's own copy of BOTH meshes before either test in
// this file runs a geometry-sensitive call.
//
// At 2 and 3 ranks, A is always exactly ONE rank, which takes
// `intx_weights.cpp`'s degenerate "serial/one-rank/identity" shortcut
// (`active_comm_size(a.pcomm) <= 1`) and never reaches MOAB's real,
// cross-rank covering-set construction at all (`cells = nullptr, count =
// 0`). At 4 ranks, A = {0, 1} -- a GENUINE two-rank ParallelComm -- so
// THIS row is the first one that actually exercises that construction
// over a placement's own, non-degenerate sub-communicator, with real,
// non-zero adopted cells expected on EVERY rank of A. The path through
// `MoabMeshBackend::adopt_remap_halo` (and its own
// `regather_with_halo`/`plan_halo_exchange`) for this configuration may
// still refuse at 4 ranks for a structural reason other than the
// single-rank guard above.
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
    // there by design).
    ASSERT_TRUE(via_overload.ok()) << via_overload.error().code << ": "
                                   << via_overload.error().message;
    EXPECT_EQ(via_overload.value(), 0u) << "rank " << rank << ": outside A, nothing to adopt";
    return;
  }
  ASSERT_TRUE(via_overload.ok()) << via_overload.error().code << ": "
                                 << via_overload.error().message;
  // A real halo, not just a successful delegation -- at 4 ranks
  // (A's own two members), the rescaled ATM/OCN pair genuinely overlaps
  // across the partition boundary WITHIN A, so each of A's own ranks must
  // adopt at least one covering cell. At 2 and 3 ranks A is a single rank
  // holding BOTH meshes' entire content already, so a zero adopted count
  // there would be equally legitimate (there is no partition boundary
  // within A to cross) -- this assertion is deliberately scoped to the
  // row where it is actually meaningful.
  if (size >= 4) {
    EXPECT_GT(via_overload.value(), 0u)
        << "rank " << rank << ": a 2-rank placement's own partition boundary should need a "
                              "real halo, not an empty one";
  }
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
// hang, not an incomplete halo.
// ---------------------------------------------------------------------------
TEST(MoabPlacedWeightsMpi, DisjointSourceAndTargetPlacementsRefuseWithTheNamedCrossPlacementGap) {
  int size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ASSERT_GE(size, 2) << "this row is only meaningful with more than one rank";
  DisjointPlacementFixture f(size);
  const DisjointRankSets sets(size);

  const auto r = fuse::build_remap_halo(f.plan, f.graph, f.backend, fuse::ComponentId{0});
  std::fprintf(stderr, "rank %d (in B=%d): build_remap_halo(source_placement=A) ok=%d%s%s\n",
              rank, sets.b.contains(rank) ? 1 : 0, r.ok() ? 1 : 0,
              r.ok() ? "" : " code=", r.ok() ? "" : r.error().code.c_str());
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

// ---------------------------------------------------------------------------
// (4) The placement selection: rescaling the backend's DEFAULT Core (the MeshId-only
// overload) must NOT
// make a DECLARED placement's own, independently-read Core copy of the
// SAME mesh pass -- `set_placements` gives each one a genuinely separate
// read, and `project_onto_unit_sphere(id, placement)`/`sphere_radius(id,
// placement)` must resolve THAT placement's own Core
// (`native_mesh(id, placement)`'s own target), never silently fall back to
// the default one. A mutation that swapped the placement overload's own
// resolution for the default-Core one would make this row's own measured
// radius agree with the rescale it never should have seen.
// ---------------------------------------------------------------------------
TEST(MoabPlacedWeightsMpi, RescalingTheDefaultCoreDoesNotAffectAPlacementsOwnCopy) {
  int size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ASSERT_GE(size, 2) << "this row is only meaningful with more than one rank";
  const DisjointRankSets sets(size);

  fuse::MoabMeshBackend backend;
  ASSERT_TRUE(backend.set_communicator(world_handle()).ok());
  ASSERT_TRUE(backend.set_placements({sets.a, sets.b}).ok());
  ASSERT_TRUE(backend.load({.name = "t", .source = FUSE_SPHERE_OCN}).ok());
  const fuse::MeshId tmid = backend.mesh_id("t").value();

  // Rescale the DEFAULT Core's own copy ONLY -- every rank participates in
  // the base communicator this overload resolves through, regardless of
  // placement membership.
  const auto rescaled_default = backend.project_onto_unit_sphere(tmid);
  ASSERT_TRUE(rescaled_default.ok()) << rescaled_default.error().code << ": "
                                     << rescaled_default.error().message;
  EXPECT_FALSE(rescaled_default.value().already_unit)
      << "recMeshOcn.h5m is at Earth radius, not the unit sphere, before any rescale";

  // Placement A's OWN copy must be UNTOUCHED by the default-Core rescale
  // above -- still measuring the file's own, un-rescaled Earth radius.
  if (sets.a.contains(rank)) {
    const auto ra = backend.sphere_radius(tmid, fuse::ComponentId{0});
    ASSERT_TRUE(ra.ok()) << ra.error().code << ": " << ra.error().message;
    EXPECT_NEAR(ra.value().mean_radius, 6371229.0, 1.0)
        << "rank " << rank
        << ": rescaling the DEFAULT Core must not touch placement A's own, "
           "independently-read copy -- if this now measures ~1.0, the "
           "placement overload resolved the wrong Core";
  }
  // Symmetrically for B.
  if (sets.b.contains(rank)) {
    const auto rb = backend.sphere_radius(tmid, fuse::ComponentId{1});
    ASSERT_TRUE(rb.ok()) << rb.error().code << ": " << rb.error().message;
    EXPECT_NEAR(rb.value().mean_radius, 6371229.0, 1.0)
        << "rank " << rank
        << ": rescaling the DEFAULT Core must not touch placement B's own, "
           "independently-read copy either";
  }
}

// ---------------------------------------------------------------------------
// (5) The "do not alias" requirement (adoption is placement-aware): the
// default Core's own recorded halo for a (source, target, dim) triple and a
// DECLARED placement's own recorded halo for the IDENTICAL triple must be
// two independent cache entries -- `halo_pairs`/`halos` are keyed by
// `(mesh, mesh, dim, placement)`, not `(mesh, mesh, dim)` alone. A mutation
// that drops `placement` from either key would make querying ONE answer
// from having only ever built the OTHER -- checked directly here via
// `adopt_remap_halo`/`has_remap_halo`, without needing a real covering-set
// build (a `count == 0` call is a legitimate "built, nothing adopted"
// answer either way, and is all the cache-key independence question
// needs).
// ---------------------------------------------------------------------------
TEST(MoabPlacedWeightsMpi, DefaultCoreAndAPlacementsHaloForTheSamePairDoNotAlias) {
  int size = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ASSERT_GE(size, 2) << "this row is only meaningful with more than one rank";
  const DisjointRankSets sets(size);

  fuse::MoabMeshBackend backend;
  ASSERT_TRUE(backend.set_communicator(world_handle()).ok());
  ASSERT_TRUE(backend.set_placements({sets.a, sets.b}).ok());
  ASSERT_TRUE(backend.load({.name = "s", .source = FUSE_SPHERE_ATM}).ok());
  ASSERT_TRUE(backend.load({.name = "t", .source = FUSE_SPHERE_OCN}).ok());
  const fuse::MeshId smid = backend.mesh_id("s").value();
  const fuse::MeshId tmid = backend.mesh_id("t").value();

  // Build ONLY the default Core's own halo (no placement argument --
  // ComponentId{}'s own default).
  const auto default_built =
      backend.adopt_remap_halo(smid, tmid, /*dim=*/2, nullptr, 0);
  ASSERT_TRUE(default_built.ok()) << default_built.error().code << ": "
                                 << default_built.error().message;
  EXPECT_TRUE(backend.has_remap_halo(smid, tmid, 2))
      << "the default Core's own halo must be recorded";

  // Placement A's own halo for the IDENTICAL (source, target, dim) must
  // NOT already read as built -- it has never itself been adopted, only
  // the DEFAULT Core's copy has. A mutation dropping `placement` from
  // `halo_pairs`'s own key would make this EXPECT_FALSE read true instead
  // (the default Core's entry would satisfy both queries).
  if (sets.a.contains(rank)) {
    EXPECT_FALSE(backend.has_remap_halo(smid, tmid, 2, fuse::ComponentId{0}))
        << "rank " << rank
        << ": placement A's own halo must be independent of the default "
           "Core's -- building the default Core's must not make A's own "
           "query read as already built too";
  }
  // Symmetrically for B.
  if (sets.b.contains(rank)) {
    EXPECT_FALSE(backend.has_remap_halo(smid, tmid, 2, fuse::ComponentId{1}))
        << "rank " << rank << ": placement B's own halo must be independent too";
  }

  // Now build placement A's OWN halo too, and confirm the default Core's
  // own answer is UNCHANGED -- independence in both directions, not just
  // one.
  if (sets.a.contains(rank)) {
    const auto a_built = backend.adopt_remap_halo(smid, tmid, /*dim=*/2,
                                                  nullptr, 0, fuse::ComponentId{0});
    ASSERT_TRUE(a_built.ok()) << a_built.error().code << ": " << a_built.error().message;
    EXPECT_TRUE(backend.has_remap_halo(smid, tmid, 2, fuse::ComponentId{0}))
        << "rank " << rank << ": placement A's own halo, once built, must read as built";
  }
  EXPECT_TRUE(backend.has_remap_halo(smid, tmid, 2))
      << "rank " << rank
      << ": the default Core's own halo must still read as built after A's own was "
         "adopted separately -- the two entries must not have collided";
}

#else  // !FUSE_SPHERE_ATM

TEST(MoabPlacedWeightsMpi, SkippedWithoutTheSphereMeshFixture) {
  GTEST_SKIP() << "wholeATM_T.h5m / recMeshOcn.h5m not found; pass -DFUSE_MOAB_MESHFILES=<moab-"
                  "source>/MeshFiles/unittest to compile the halo-bridging checks";
}

#endif  // FUSE_SPHERE_ATM
