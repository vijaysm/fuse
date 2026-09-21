// The remap halo's SERIAL contract — the part of KNOWN_ISSUES D4's
// `A-PARTITIONED` gap that can be tested without a launcher.
//
// `build_remap_halo`, `exchange_remap_halo` and `accumulate_remap_halo` had 34
// references across `src/` and `include/` and NOT ONE test, which is how the
// S5.5 matrix gate found them. Their parallel behaviour needs `mpiexec` and a
// partitioned spherical pair and is still uncovered; what is covered here is
// the contract their own headers state for one rank, and that contract is
// load-bearing in a specific way:
//
//   "Idempotent per (source mesh, target mesh, dimension). Serial, or a
//    communicator of one rank, is a successful no-op, so a caller may make
//    the call unconditionally."
//
// Every unconditional caller depends on that. If a serial call returned an
// error -- or, worse, tried the covering-set construction and returned
// `E-MESH-HALO-GEOMETRY` because the pair is not spherical -- then a serial
// run of any coupled model would fail at setup. Nothing checked it.

#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/moab_backend.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace {

/// Two DELIBERATELY NON-SPHERICAL meshes. The covering-set construction
/// handles spherical surfaces only and reports `E-MESH-HALO-GEOMETRY`
/// otherwise, so a pair of boxes is the sharpest available test that the
/// serial path short-circuits BEFORE any geometry: a no-op cannot fail on a
/// geometry it never looks at.
struct BoxPair {
  fuse::MoabMeshBackend backend;
  fuse::CouplingGraph   graph;
  fuse::MeshId          smid, tmid;
  fuse::RemapPlan       plan;

  BoxPair() {
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

TEST(RemapHalo, ASerialCallIsASuccessfulNoOpAndIsIdempotent) {
  BoxPair f;  // no communicator set: one rank by construction

  const auto first = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  ASSERT_TRUE(first.ok())
      << first.error().code << ": " << first.error().message
      << " -- a serial call must succeed, because every caller is documented "
         "as free to make it unconditionally";
  EXPECT_EQ(first.value(), 0u) << "one rank owns everything, so there is "
                                  "nothing to halo";

  // Idempotence is part of the contract, not an accident of caching: a
  // Runner that re-initialises calls this again on the same triple.
  const auto second = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  ASSERT_TRUE(second.ok()) << second.error().code;
  EXPECT_EQ(second.value(), 0u);
}

TEST(RemapHalo, ANonSphericalPairIsNotRefusedInSerial) {
  BoxPair f;
  const auto r = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  // The distinction this pins: the geometry limit belongs to the covering-set
  // construction, which a one-rank call must never reach. A serial run of a
  // planar or volumetric model would otherwise fail at setup with
  // E-MESH-HALO-GEOMETRY on a pair it can remap perfectly well. Success is
  // the whole assertion -- reading `error()` off a successful Result to name
  // the code we did NOT get is not a stronger check, it is a second bug.
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

TEST(RemapHalo, TheBookkeepingAgreesWithWhatWasBuilt) {
  BoxPair f;
  ASSERT_TRUE(fuse::build_remap_halo(f.plan, f.graph, f.backend).ok());
  EXPECT_EQ(f.backend.remap_halo_size(f.smid, 3), 0u);
  // An unknown id and a wrong dimension must answer rather than throw -- both
  // accessors are noexcept and are called from paths that cannot handle an
  // exception.
  EXPECT_FALSE(f.backend.has_remap_halo(fuse::MeshId{99}, f.tmid, 3));
  EXPECT_EQ(f.backend.remap_halo_size(fuse::MeshId{99}, 3), 0u);
  EXPECT_EQ(f.backend.remap_halo_size(f.smid, 1), 0u);
}

TEST(RemapHalo, WeightsBuildInSerialWithoutTheParallelRefusal) {
  BoxPair f;
  ASSERT_TRUE(fuse::build_remap_halo(f.plan, f.graph, f.backend).ok());

  // The other half of the boundary moab_backend.hpp calls load bearing: a
  // communicator of ONE must not produce E-REMAP-PARALLEL. (`> 1` without a
  // halo must; that half needs mpiexec and is the uncovered part of D4's
  // A-PARTITIONED.)
  const auto w = fuse::build_weights(f.plan, f.graph, f.backend);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;
}

TEST(RemapHalo, TheExchangeChecksItsArgumentsBeforeAnythingElse) {
  BoxPair f;
  ASSERT_TRUE(fuse::build_remap_halo(f.plan, f.graph, f.backend).ok());

  std::vector<double> buf(64, 0.0);
  // A null buffer and a non-positive component count are refused by both
  // directions. These are the only refusals of the exchange that do not need
  // a peer, and an unchecked null here would be a segfault inside MPI.
  for (const bool accumulate : {false, true}) {
    const auto null_buf =
        accumulate ? f.backend.accumulate_remap_halo(f.smid, 3, nullptr, 1)
                   : f.backend.exchange_remap_halo(f.smid, 3, nullptr, 1);
    ASSERT_FALSE(null_buf.ok()) << "accumulate=" << accumulate;
    EXPECT_EQ(null_buf.error().code, "E-MESH-ARG");

    const auto zero_comp =
        accumulate ? f.backend.accumulate_remap_halo(f.smid, 3, buf.data(), 0)
                   : f.backend.exchange_remap_halo(f.smid, 3, buf.data(), 0);
    ASSERT_FALSE(zero_comp.ok()) << "accumulate=" << accumulate;
    EXPECT_EQ(zero_comp.error().code, "E-MESH-ARG");
  }
}

TEST(RemapHalo, AnExchangeWithNoHaloLeavesTheBufferAlone) {
  BoxPair f;
  ASSERT_TRUE(fuse::build_remap_halo(f.plan, f.graph, f.backend).ok());

  const std::size_t n = f.backend.view(f.smid).value()->entity_count(3);
  std::vector<double> buf(n, 7.25);
  const std::vector<double> before = buf;

  // With no halo slots there is nothing to send or receive, so both
  // directions must succeed and touch nothing. `accumulate` additionally
  // zeroes halo slots -- of which there are none, so an implementation that
  // zeroed OWNED slots instead would show up right here.
  ASSERT_TRUE(f.backend.exchange_remap_halo(f.smid, 3, buf.data(), 1).ok());
  EXPECT_EQ(buf, before);
  ASSERT_TRUE(f.backend.accumulate_remap_halo(f.smid, 3, buf.data(), 1).ok());
  EXPECT_EQ(buf, before) << "accumulate zeroed owned values; only halo slots "
                            "may be zeroed";
}

}  // namespace
