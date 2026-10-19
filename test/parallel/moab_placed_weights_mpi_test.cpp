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
// THE REAL, NON-MATCHING SPHERE PAIR (round 4 fix; FUSE_SPHERE_ATM/
// FUSE_SPHERE_OCN, test/CMakeLists.txt), NOT a generated box mesh. Round
// 2/3 used `generated:box:4x4x4`/`generated:box:2x2x2` (dimension 3) on
// the stated reasoning that a generated mesh under a communicator does
// not actually partition anyway (moab_placed_view_mpi_test.cpp's own
// header), so a structural-resolution check did not need real geometry.
// MEASURED WRONG, round 4 (host run at HEAD 512ab69): MOAB's own
// covering-set construction (Intx2MeshOnSphere/FindMaxEdges/
// construct_covering_set, src/backend/intx_weights.cpp) is declared on
// dimension-2 SURFACES ONLY (the SAME restriction the single-placement
// `build_remap_halo` already states and tests, remap_halo_test.cpp/
// remap_halo_mpi_test.cpp) -- a dimension-3 fixture hits
// E-MESH-HALO-GEOMETRY the moment ANY path through this file's own
// overload reaches a REAL covering-set attempt (the same-Core delegation,
// test (2) below), before the cross-placement logic under test is ever
// exercised. READ AGAINST THE BRIEF (round 4 question 1): the production
// couplings this item exists for (Brusselator, dn_heat) are both 2-D
// surfaces; nothing in D5's own history or the owner's D73 decision asks
// for a VOLUMETRIC covering set. The dimension-3 fixture was therefore an
// accident of copying `BoxPairMpi` (remap_halo_mpi_test.cpp), which
// deliberately uses a non-spherical box pair to test E-MESH-HALO-GEOMETRY
// itself -- the wrong fixture to reuse for a test that needs a REAL halo
// to actually build. Fixed by switching to the SAME non-matching
// ATM/OCN sphere pair `RemapHaloMpiSphere`'s own tests already use
// successfully under a real communicator (parallel.remap_halo_mpi.{1..5}
// PASS at this HEAD) -- gated identically (`#ifdef FUSE_SPHERE_ATM`,
// which CMake defines only when BOTH wholeATM_T.h5m and recMeshOcn.h5m
// were found), with the same `#else` GTEST_SKIP fallback
// moab_placed_view_mpi_test.cpp uses when they are not.
//
// MEASURED FINDING, round 2 (read AND built, not yet host-run at the
// time): `MoabMeshBackend::set_placements` gives every declared processor
// set its OWN `moab::Core` (moab_backend.cpp's `PlacementContext::core`),
// and MOAB's own covering-set construction needs BOTH meshes in the SAME
// `moab::Interface` -- there is no MOAB entry point that intersects a
// file set living in one Core against a file set living in another.
// Bridging two DIFFERENT per-placement Cores therefore needs a
// migrate-by-global-id exchange (the MOAB analogue of
// `fuse::native::build_native_remap_halo`'s own box-overlap/pack/merge
// architecture, native_halo.cpp) that is NOT YET BUILT.
// `fuse::build_remap_halo`'s 4-argument overload
// (`include/fuse/backend/remap_plan.hpp`, `src/backend/intx_weights.cpp`)
// therefore does exactly three things, all measured here: (1) when the
// calling rank is structurally outside the TARGET's own placement, a
// successful no-op (checked via `MoabMeshBackend::placement_participates`,
// not a local size); (2) when the source and target placements happen to
// resolve to the SAME `moab::Core` (the degenerate case: no placements
// declared, or an identical processor set given for both), it delegates to
// the EXISTING, already-tested single-placement `build_remap_halo`, and
// this row checks that delegation reproduces the single-placement answer
// exactly -- now against a REAL halo build between two genuinely
// dimension-2 meshes, not a dimension check refusing before the
// delegation is ever meaningfully exercised; (3) when the two Cores
// genuinely DIFFER (a true cross-placement pair), an explicit, named
// refusal (`E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED`) rather than an
// incomplete or silently wrong halo. KNOWN_ISSUES.md D5 records this as
// the measured, built-but-incomplete state; closing it needs the
// migrate-by-global-id mechanism above, which this track's own report
// flags as not completed in the time available.
//
// ROUND 4, QUESTION 2 (read, not re-designed): is
// E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED on an in-B rank the refusal
// round 2 added for the case round 3's test (3) below EXPECTS, or the
// overload refusing where it should have adopted? Read against this
// file's own three rows: test (3) is EXACTLY the row that asserts this
// refusal on every in-B rank, by design, for the genuinely disjoint pair
// -- so seeing it there is correct, not a defect. The round-3 host
// failure was test (2)'s `ASSERT_TRUE(via_overload.ok())`, which failed
// on an in-A rank for an UNRELATED reason (E-MESH-HALO-GEOMETRY from the
// wrong, dimension-3 fixture, via the delegated single-placement path),
// not because the overload answered wrongly for test (2)'s own same-Core
// case. No refusal-path logic in `build_remap_halo`'s 4-argument overload
// changes this round; only the fixture's mesh pair does.
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
/// B = {1}; at `size == 3`, A = {0}, B = {1, 2}; at `size == 4`, A =
/// {0, 1}, B = {2, 3}. Neither set is ever empty at the registered rank
/// counts (2, 3, 4). KNOWN_ISSUES D81 round 9: `size == 4` is the FIRST
/// registered row where a placement's own sub-communicator (A, here) has
/// MORE than one rank -- at 2 and 3 ranks A is always exactly {0}, a
/// single-rank ParallelComm that round 8 measured takes a DIFFERENT code
/// path (the degenerate "serial/one-rank/identity" shortcut,
/// intx_weights.cpp) than a genuine multi-rank covering-set construction
/// ever reaches, so nothing at 2 or 3 ranks measures the configuration D5
/// and D81 actually exist for: a real coupling places SEVERAL ranks per
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

    // KNOWN_ISSUES D81, owner ruling 2026-10-19: rescaling is the DRIVER's
    // explicit responsibility, never the library's. `set_placements`
    // gives placements A and B their own, independently-read Core copies
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
//
// ROUND 6 (KNOWN_ISSUES D81, owner ruling 2026-10-19): a SUCCESSFUL
// delegated halo build, asserted on the ADOPTED CELL COUNT -- this is
// what round 5's own outcome-identity assertion (both paths refuse
// identically) stood in for while the fixture's own radius mismatch
// (wholeATM_T.h5m = 1.0, recMeshOcn.h5m = 6371229.0, Earth radius in
// metres) had no placement-aware rescale available. The fixture's own
// constructor now calls the new `project_onto_unit_sphere(MeshId,
// ComponentId)` overload on placement A's own copy of BOTH meshes before
// either test in this file runs a geometry-sensitive call, so this row
// goes back to its original intent.
//
// ROUND 9 (KNOWN_ISSUES D81, owner note 2026-10-19): at 2 and 3 ranks, A
// is always exactly ONE rank -- round 8's own fix makes that case take
// `intx_weights.cpp`'s degenerate "serial/one-rank/identity" shortcut
// (`active_comm_size(a.pcomm) <= 1`), which never reaches MOAB's real,
// cross-rank covering-set construction at all (`cells = nullptr, count =
// 0`). At 4 ranks, A = {0, 1} -- a GENUINE two-rank ParallelComm -- so
// THIS row is the first one that actually exercises that construction
// over a placement's own, non-degenerate sub-communicator, with real,
// non-zero adopted cells expected on EVERY rank of A. See KNOWN_ISSUES
// D83 for what reading `MoabMeshBackend::adopt_remap_halo` (and its own
// `regather_with_halo`/`plan_halo_exchange`) found about this
// configuration before any host run: a structural reason this row may
// still refuse at 4 ranks that is NOT the round-7/8 guard, and is NOT
// fixed this round.
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
  // ROUND 9: a real halo, not just a successful delegation -- at 4 ranks
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
// (4) KNOWN_ISSUES D81's own mutation target "the placement selection":
// rescaling the backend's DEFAULT Core (the MeshId-only overload) must NOT
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

#else  // !FUSE_SPHERE_ATM

TEST(MoabPlacedWeightsMpi, SkippedWithoutTheSphereMeshFixture) {
  GTEST_SKIP() << "wholeATM_T.h5m / recMeshOcn.h5m not found; pass -DFUSE_MOAB_MESHFILES=<moab-"
                  "source>/MeshFiles/unittest to compile the halo-bridging checks";
}

#endif  // FUSE_SPHERE_ATM
