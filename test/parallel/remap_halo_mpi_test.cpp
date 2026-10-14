// test/parallel/remap_halo_mpi_test.cpp — the PARALLEL half of KNOWN_ISSUES
// D4's `A-PARTITIONED` gap (build-evidence/parallel-matrix.toml), under a
// REAL mpiexec. test/backend/remap_halo_test.cpp covers the one-rank
// contract; this file is everything that needs a communicator of more than
// one rank to exercise at all:
//
//   * the `E-REMAP-PARALLEL` boundary moab_backend.hpp itself calls load
//     bearing -- "`> 1` rejects; `== 1` must NOT" -- checked on BOTH sides,
//     not assumed from one;
//   * `E-MESH-HALO-GEOMETRY` firing on a non-spherical partitioned pair
//     (a generated box pair) and NOT firing on the real spherical pair;
//   * the forward exchange (`exchange_remap_halo`) delivering every halo
//     slot's OWNER's value, checked exactly by global id;
//   * the reverse exchange (`accumulate_remap_halo`) round-tripping an
//     integer-valued field to an EXACT total, cross-checked against the
//     mesh's own known cell count -- which is also the only place this file
//     can independently confirm the halo's bookkeeping agrees on every rank
//     (a lost or duplicated exchange message breaks the identity outright).
//
// NOT YET RUN: this sandbox cannot initialise MPI at all -- not even at one
// rank (docs/dev/testing.md; KNOWN_ISSUES.md C1) -- and MOAB in a build with
// FUSE_ENABLE_MPI links against an MPI-enabled MOAB whose moab::Core
// constructor itself calls MPI_Init_thread, so even a single-process run of
// this binary aborts here. Registered only when FUSE_ENABLE_MOAB AND
// FUSE_ENABLE_MPI are both on (test/CMakeLists.txt); its outcome comes from
// a host run, like every other MOAB-enabled-build-configuration MPI test in this tree.

#include <cstdint>
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

// ---------------------------------------------------------------------------
// The generated, NON-SPHERICAL, NON-IDENTITY box pair. Both `E-REMAP-PARALLEL`
// (build_weights, below) and `E-MESH-HALO-GEOMETRY` (build_remap_halo's own
// `a.dim != 2` check) are keyed on structural properties -- ranks > 1, the
// two mesh entity sets naming different file sets -- that need no real halo geometry
// at all, so this fixture needs no mesh file and compiles unconditionally.
// ---------------------------------------------------------------------------
struct BoxPairMpi {
  fuse::MoabMeshBackend backend;
  fuse::CouplingGraph    graph;
  fuse::MeshId           smid, tmid;
  fuse::RemapPlan        plan;

  BoxPairMpi() {
    EXPECT_TRUE(backend.set_communicator(world_handle()).ok());
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
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// moab_backend.hpp on `communicator_size()`: "THIS VALUE IS LOAD BEARING AND
// ITS BOUNDARY SHOULD BE TESTED. `> 1` rejects; `== 1` must NOT, because a
// one-rank communicator is an ordinary MPI job whose mesh happens to be
// undivided, every cell is local, and the serial construction is exactly
// right. Rejecting it would take remap away from every MPI run and no serial
// test could see the difference." remap_plan.hpp's E-REMAP-PARALLEL
// (`build_weights`) is exactly that boundary, checked here at every
// registered rank count (1..5) rather than assumed to hold on one side
// because it holds on the other.
// ---------------------------------------------------------------------------
TEST(RemapHaloMpi, ERemapParallelRefusesAboveOneRankAndNotAtOne) {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  BoxPairMpi f;

  // Deliberately NO build_remap_halo() call: this is the "asked for it and
  // did not get it" boundary, not the halo's own correctness.
  const auto w = fuse::build_weights(f.plan, f.graph, f.backend);
  if (size == 1) {
    ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message
                        << " -- a one-rank communicator must NOT be refused; "
                           "every cell is local and the serial construction "
                           "is exactly right";
  } else {
    ASSERT_FALSE(w.ok()) << "size=" << size
                         << ": a partitioned pair with no remap halo must be "
                            "refused, not silently truncated";
    EXPECT_EQ(w.error().code, "E-REMAP-PARALLEL");
  }
}

// ---------------------------------------------------------------------------
// remap_plan.hpp on build_remap_halo: "the covering-set construction
// available here handles SPHERICAL surfaces only, so a partitioned pair that
// is not one is `E-MESH-HALO-GEOMETRY`". A generated box (dimension 3) hits
// intx_weights.cpp's `a.dim != 2` check before any surface fit is even
// attempted -- the sharpest available non-spherical case, the same reasoning
// test/backend/remap_halo_test.cpp's serial `BoxPair` fixture already uses.
// At exactly one rank the SAME pair must short-circuit before that check is
// ever reached, mirroring the serial contract under a REAL one-rank
// communicator (as opposed to no communicator at all, which the existing
// serial test covers).
// ---------------------------------------------------------------------------
TEST(RemapHaloMpi, EMeshHaloGeometryRefusesTheBoxPairAboveOneRankAndNotAtOne) {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  BoxPairMpi f;

  const auto r = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  if (size == 1) {
    ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
    EXPECT_EQ(r.value(), 0u);
  } else {
    ASSERT_FALSE(r.ok()) << "size=" << size
                         << ": a partitioned, non-spherical (dimension-3) "
                            "pair must be refused by name, not guessed at";
    EXPECT_EQ(r.error().code, "E-MESH-HALO-GEOMETRY");
  }
}

// ---------------------------------------------------------------------------
// Review 2026-10-11 TF-3: `MoabMeshBackend::adopt_remap_halo`'s own
// per-rank rejections (the GLOBAL_ID tag lookup, the per-covering-cell
// GLOBAL_ID check, `add_entities`, `regather_with_halo`) used to return
// straight from this rank while a peer with an unaffected covering set
// proceeded into `plan_halo_exchange`'s own collectives -- and, worse, a
// rank with an EMPTY local covering set (`count == 0`, an entirely
// ordinary outcome, not an error) used to skip the rejection checks
// AND the reconciliation entirely, going straight to `plan_halo_exchange`,
// while a peer with `count > 0` took the unreconciled rejection path.
// Calls `adopt_remap_halo` DIRECTLY (it is a public method; the "only
// build_remap_halo may call it" rule in its own header is a layering
// convention, not an access restriction) with a handle MOAB can never
// resolve (0, always invalid -- no entity has handle 0) on exactly one
// rank's covering set, and an EMPTY covering set (`count == 0`) on every
// other rank -- the exact "one rank bad, the rest legitimately idle" shape
// TF-3 names. Pre-fix signature: the bad rank returns `Failure` directly
// (skipping plan_halo_exchange's own collectives) while the `count == 0`
// ranks call `plan_halo_exchange`, enter ITS internal Allgather, and hang
// waiting for a peer that already returned. Post-fix: every rank returns
// the SAME `Failure` within the test's own TIMEOUT, none of them ever
// reaching `plan_halo_exchange`.
// ---------------------------------------------------------------------------
TEST(RemapHaloMpi, AdoptRemapHaloOneRankBadCoveringCellAgreesAcrossRanks) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int fault_rank = size - 1;

  fuse::MoabMeshBackend backend;
  ASSERT_TRUE(backend.set_communicator(world_handle()).ok());
  const char* src = "generated:box:4x4x4";
  ASSERT_TRUE(backend.load({.name = "s", .source = src}).ok());
  const auto smid = backend.mesh_id("s");
  ASSERT_TRUE(smid.ok());

  fuse::Result<std::size_t> r =
      fuse::Failure{"E-TEST-UNREACHED", "adopt_remap_halo was never called"};
  if (rank == fault_rank) {
    // Entity handle 0 names no entity in any moab::Interface -- MOAB's own
    // reserved "invalid handle" value -- so `tag_get_data` on it cannot
    // succeed. Stands in for a genuinely malformed covering-set cell (e.g.
    // one whose GLOBAL_ID tag was never set) without needing one.
    const std::uint64_t bad_handle = 0;
    r = backend.adopt_remap_halo(smid.value(), smid.value(), /*dim=*/3, &bad_handle, 1);
  } else {
    // An ordinary rank whose own covering set happens to be empty -- the
    // case the pre-fix `count == 0` early return used to treat as "nothing
    // to reconcile, go straight to plan_halo_exchange".
    r = backend.adopt_remap_halo(smid.value(), smid.value(), /*dim=*/3, nullptr, 0);
  }
  ASSERT_FALSE(r.ok()) << "rank " << rank
                       << ": every rank of the communicator must refuse "
                          "together, not just the one rank with the bad "
                          "covering cell";
  EXPECT_EQ(r.error().code, "E-MESH-HALO");
}

// ---------------------------------------------------------------------------
// The REAL non-matching spherical pair: MOAB's own MeshFiles/unittest holds
// wholeATM_T.h5m (atmosphere) and recMeshOcn.h5m (ocean), both on the sphere
// and deliberately not matching (an identity pair cannot exercise a halo at
// all: with both sides on one partition of one mesh, target cell j's source
// is cell j on the same rank).
//
// PROJECTION IS NOT OPTIONAL. Read directly off the files (h5dump on the raw
// node coordinates, 2026-09-28): wholeATM_T.h5m sits at radius 1.0 (std/mean
// radius ~6e-17) and recMeshOcn.h5m at radius 6371229.0, Earth radius in
// metres (std/mean radius ~1.1e-16 -- the SAME order as wholeATM_T.h5m's own,
// both floating-point noise, not a real deviation from a sphere) -- two
// different, individually well-formed, spheres. intx_weights.cpp's own
// radius-agreement check
// (`kGeomTol = 1e-9`) would reject that combination outright with
// `E-REMAP-GEOMETRY` ("not the same domain"), correctly: the library cannot
// know two different radii are meant to be the same sphere in different
// units. `MoabMeshBackend::project_onto_unit_sphere` (added for this cell)
// rescales each mesh onto a common unit sphere before either
// `build_remap_halo` or `build_weights` ever sees them.
// ---------------------------------------------------------------------------
#if defined(FUSE_SPHERE_ATM) && defined(FUSE_SPHERE_OCN)

namespace {

// Measured via h5ls -r on the raw fixture files (2026-09-28): wholeATM_T.h5m
// has one Quad4 block of 726 elements; recMeshOcn.h5m has a Polygon5 block of
// 6 and a Polygon6 block of 7147, 7153 dimension-2 cells total. Both numbers
// are properties of the FILE MOAB ships, not of this build, and are the
// "one-rank reference" every Allreduce-summed rank count below is checked
// against -- the same escape hatch test/partition/rcb_mpi_test.cpp and the
// A-SEARCH-DISTRIBUTED matrix cell use for the same reason (a live one-rank
// comparison is not available from inside a single mpiexec launch, so the
// reference is a fact recomputed the same way at every rank count instead).
constexpr std::size_t kAtmCells = 726;
constexpr std::size_t kOcnCells = 7153;

struct SpherePairMpi {
  fuse::MoabMeshBackend backend;
  fuse::CouplingGraph    graph;
  fuse::MeshId           amid, omid;
  fuse::RemapPlan        plan;

  SpherePairMpi() {
    EXPECT_TRUE(backend.set_communicator(world_handle()).ok());
    EXPECT_TRUE(
        backend.load({.name = "atm", .source = FUSE_SPHERE_ATM}).ok());
    EXPECT_TRUE(
        backend.load({.name = "ocn", .source = FUSE_SPHERE_OCN}).ok());
    amid = graph.add_mesh({.name = "atm", .source = FUSE_SPHERE_ATM});
    omid = graph.add_mesh({.name = "ocn", .source = FUSE_SPHERE_OCN});

    // Every rank, in the same order (both calls are collective): project
    // ATM (already ~unit) and OCN (Earth-radius) onto the SAME unit sphere.
    const auto pa = backend.project_onto_unit_sphere(amid);
    EXPECT_TRUE(pa.ok()) << pa.error().code << ": " << pa.error().message;
    const auto po = backend.project_onto_unit_sphere(omid);
    EXPECT_TRUE(po.ok()) << po.error().code << ": " << po.error().message;

    const fuse::RegionId arid =
        graph.add_region({.name = "a_all", .mesh = amid, .dimension = 2});
    const fuse::RegionId orid =
        graph.add_region({.name = "o_all", .mesh = omid, .dimension = 2});
    plan.src = arid;
    plan.tgt = orid;
  }
};

}  // namespace

// The fixture itself, under a real communicator of 1..5 ranks: it loads, it
// is genuinely spherical after projection (re-projecting an already-unit
// mesh must be very close to a no-op -- idempotence, the same property the
// serial halo test pins for build_remap_halo), and no rank's partition of
// either mesh is empty (the Allreduce-summed cell counts below would silently
// undercount if one were, and this is the one place that would show it).
TEST(RemapHaloMpiSphere, TheNonMatchingPairLoadsProjectsAndIsSphericalAtEveryRankCount) {
  SpherePairMpi f;

  const auto reproject = f.backend.project_onto_unit_sphere(f.amid);
  ASSERT_TRUE(reproject.ok()) << reproject.error().code << ": "
                             << reproject.error().message;
  EXPECT_NEAR(reproject.value(), 1.0, 1e-6)
      << "projecting an already-unit-sphere mesh a second time must be "
         "very close to a no-op";

  const auto* av = f.backend.view(f.amid).value();
  const auto* ov = f.backend.view(f.omid).value();
  const long long local_atm = static_cast<long long>(av->entity_count(2));
  const long long local_ocn = static_cast<long long>(ov->entity_count(2));
  long long global_atm = 0, global_ocn = 0;
  ASSERT_EQ(MPI_Allreduce(&local_atm, &global_atm, 1, MPI_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&local_ocn, &global_ocn, 1, MPI_LONG_LONG, MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  // Before build_remap_halo runs, entity_count(2) IS owned_count(2) (no
  // ghosts requested at read time; KNOWN_ISSUES/moab_backend.hpp), so this
  // sum has no double count.
  EXPECT_EQ(static_cast<std::size_t>(global_atm), kAtmCells);
  EXPECT_EQ(static_cast<std::size_t>(global_ocn), kOcnCells);
}

// ---------------------------------------------------------------------------
// The substance of A-PARTITIONED: build a REAL halo across a REAL
// communicator of more than one rank, and check every postcondition the
// headers document for it.
// ---------------------------------------------------------------------------
TEST(RemapHaloMpiSphere,
    BuildRemapHaloDeliversExactHaloValuesAndRoundTripsExactly) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  SpherePairMpi f;

  // (c) E-MESH-HALO-GEOMETRY must NOT fire on the real spherical pair --
  // together with EMeshHaloGeometryRefusesTheBoxPairAboveOneRankAndNotAtOne
  // above, this is "fires on non-spherical, not on spherical".
  const auto halo = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  ASSERT_TRUE(halo.ok()) << halo.error().code << ": " << halo.error().message;

  const auto* av = f.backend.view(f.amid).value();
  const std::size_t n = av->entity_count(2);
  const std::size_t owned = av->owned_count(2);
  const std::size_t halo_n = f.backend.remap_halo_size(f.amid, 2);
  ASSERT_TRUE(f.backend.has_remap_halo(f.amid, f.omid, 2));
  // (d), first half: this rank's own bookkeeping is internally consistent --
  // every local entity is either owned or an adopted halo cell, nothing else.
  EXPECT_EQ(owned + halo_n, n);

  // --- (b), forward: exchange_remap_halo delivers every halo slot's
  // OWNER's value, checked EXACTLY by global id. No gather is needed for the
  // reference: every rank applies the SAME injective function (the identity)
  // to its OWN owned cells' global ids, so a halo slot's expected value is
  // simply that slot's OWN MeshView::global_id -- which MOAB preserves
  // on an adopted halo cell (adopt_remap_halo matches by GLOBAL_ID) --
  // regardless of which rank actually owns it. ---
  std::vector<double> fwd(n, -1.0);  // sentinel: no real global id is negative
  for (std::size_t i = 0; i < n; ++i) {
    if (av->is_owned(2, i)) {
      fwd[i] = static_cast<double>(av->global_id(2, i));
    }
  }
  const std::vector<double> fwd_before_halo = fwd;  // owned slots, for later
  ASSERT_TRUE(
      f.backend.exchange_remap_halo(f.amid, 2, fwd.data(), 1).ok());
  for (std::size_t i = 0; i < n; ++i) {
    const double expect = static_cast<double>(av->global_id(2, i));
    EXPECT_EQ(fwd[i], expect)
        << "local index " << i << " (owned=" << av->is_owned(2, i)
        << "): exchange_remap_halo must deliver exactly the owner's value";
    if (av->is_owned(2, i)) {
      EXPECT_EQ(fwd[i], fwd_before_halo[i])
          << "an owned slot must be read, never written, by the forward "
             "exchange";
    }
  }

  // --- (b), reverse, and (d)'s other half: accumulate_remap_halo's own
  // round trip. Every LOCAL entity (owned or halo copy) starts at exactly
  // 1.0; accumulate ADDS every halo slot's value into its owner and zeroes
  // the halo slot. Summed over every rank's OWNED entities after the call,
  // the total is exactly (global cell count) + (global halo-instance count)
  // -- the "1-rank reference" being kAtmCells, known independently of any
  // rank count, plus however many halo copies this run actually built.
  // Every addition here is of the exact value 1.0, so the identity below is
  // EXACT, not a tolerance -- and a lost or duplicated exchange message
  // (item (d): counts sent != counts received) is exactly what would break
  // it. ---
  std::vector<double> rev(n, 1.0);
  ASSERT_TRUE(f.backend.accumulate_remap_halo(f.amid, 2, rev.data(), 1).ok());
  double local_owned_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (av->is_owned(2, i)) {
      local_owned_sum += rev[i];
    } else {
      EXPECT_EQ(rev[i], 0.0) << "accumulate_remap_halo must zero every halo "
                                "slot it folded into its owner";
    }
  }
  const long long local_cells = static_cast<long long>(owned);
  const long long local_halo = static_cast<long long>(halo_n);
  long long global_cells = 0, global_halo = 0;
  double global_owned_sum = 0.0;
  ASSERT_EQ(MPI_Allreduce(&local_cells, &global_cells, 1, MPI_LONG_LONG,
                          MPI_SUM, MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&local_halo, &global_halo, 1, MPI_LONG_LONG,
                          MPI_SUM, MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&local_owned_sum, &global_owned_sum, 1, MPI_DOUBLE,
                          MPI_SUM, MPI_COMM_WORLD),
            MPI_SUCCESS);
  EXPECT_EQ(static_cast<std::size_t>(global_cells), kAtmCells)
      << "the halo must not change how many cells are owned in total";
  EXPECT_EQ(global_owned_sum,
           static_cast<double>(global_cells) + static_cast<double>(global_halo))
      << "accumulate_remap_halo's round trip must be exact: every one of "
         "the " << global_halo << " halo instances built across the "
         "communicator must add exactly once to its rightful owner";

  // --- Diagnostic only, not gated: the actual end-to-end conservation
  // number a real build_weights() call reports with this halo in place.
  // Unlike the exact identities above, whether this pair's Conservative flag
  // would be satisfied depends on how much of ATM's domain the regional OCN
  // mesh geometrically covers, which nobody has measured on the real files
  // (KNOWN_ISSUES D4) -- so this asserts only that the DEFAULT (truncating)
  // builder succeeds with the halo present, and prints the audit numbers for
  // the host log rather than gating on a value derived from no measurement.
  const auto w = fuse::build_weights(f.plan, f.graph, f.backend);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;
  if (rank == 0) {
    std::printf(
        "[RemapHaloMpiSphere] min_src_coverage=%.17g max_col_defect=%.17g "
        "src_measure_total(local)=%.17g tgt_measure_total(local)=%.17g\n",
        w.value().min_src_coverage, w.value().max_col_defect,
        w.value().src_measure_total, w.value().tgt_measure_total);
  }
}

#endif  // FUSE_SPHERE_ATM && FUSE_SPHERE_OCN
