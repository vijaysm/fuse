// test/parallel/moab_owner_mpi_test.cpp -- the
// set_parallel_comm wiring at MoabMeshView's TWO construction call
// sites (moab_backend.cpp's base load() path and load_into_placements(),
// this file's own header comment states which), exercised here through
// the BASE (non-placement) load path -- owner() must agree, across ranks,
// about who owns a shared entity: a rank's own ghost copy and the
// claimed owner's own account of the SAME entity must name the IDENTICAL
// global id.
//
// RANKS 2, registered in test/CMakeLists.txt immediately after the
// moab_placed_view_mpi block, with the same fixture gating:
// chosen deliberately, not merely the minimum, because with exactly two
// ranks "the owning rank" a ghost names is unambiguously the OTHER rank,
// letting the cross-check use a single MPI_Sendrecv rather than an
// Allgather keyed by an unknown number of participants.
//
// FIXTURE. The SAME real, partitioned sphere mesh (FUSE_SPHERE_ATM) the
// existing moab_placed_view_mpi_test.cpp/moab_placed_weights_mpi_test.cpp
// family already uses for a genuine point-to-point ghost check (chosen
// here, not a "generated:box:..." source, because a synthetically
// generated mesh is built identically and redundantly on every rank with
// no partition/sharing information for ParallelComm to resolve at all --
// a ghost entity does not exist on it to read owner() from in the first
// place). Gated identically (#ifdef FUSE_SPHERE_ATM): SKIPPED, not
// silently passed, when the fixture is not configured (this sandbox's
// own MOAB install -- configure's own "spherecube.h5m not found" notice).
//
// NOT YET RUN: this sandbox cannot initialise MPI at all
// (moab_placed_view_mpi_test.cpp's own header comment states the same
// fact for this MOAB-linked family specifically). Compiled here; the
// outcome comes from a host run.

#include <cstdint>
#include <string>

#include <mpi.h>

#include <gtest/gtest.h>

#include "fuse/backend/moab_backend.hpp"
#include "fuse/field/collective_agreement.hpp"

#include "fuse_test_main.hpp"

namespace {

int world_handle() { return static_cast<int>(MPI_Comm_c2f(MPI_COMM_WORLD)); }

}  // namespace

#ifdef FUSE_SPHERE_ATM

// MoabOwnerMpi.GhostEntityOwnerAgreesWithTheOwningRanksOwnReport -- this
// file's own header comment above states the full cross-check. Either
// rank may have no ghost at all among the top-dimension entities it
// holds (a legitimate outcome of however the sphere mesh happens to be
// partitioned at RANKS 2, not assumed to always occur) -- the check is
// then vacuously satisfied on that side, not skipped: the ABSENCE of a
// ghost is itself a fact `fuse::agree_outcome` still reports identically
// on both ranks.
TEST(MoabOwnerMpi, GhostEntityOwnerAgreesWithTheOwningRanksOwnReport) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  fuse::MoabMeshBackend backend;
  ASSERT_TRUE(backend.set_communicator(world_handle()).ok());
  const auto loaded = backend.load({.name = "atm", .source = FUSE_SPHERE_ATM});
  ASSERT_TRUE(loaded.ok()) << loaded.error().code << ": " << loaded.error().message;

  const auto view = backend.view(fuse::MeshId{0});
  ASSERT_TRUE(view.ok());
  const int dim = view.value()->dimension();

  // This rank's own first GHOST entity (owner() != this rank), if any --
  // set_parallel_comm is exactly what makes owner() answer at
  // all here rather than refusing E-MESH-NO-OWNER.
  long long my_ghost_gid = -1;  // sentinel: no ghost found on this rank
  int my_ghost_owner = -1;
  for (std::size_t i = 0; i < view.value()->entity_count(dim); ++i) {
    if (!view.value()->is_owned(dim, i)) {
      const auto owner_result = view.value()->owner(dim, i);
      ASSERT_TRUE(owner_result.ok()) << owner_result.error().message;
      my_ghost_gid = static_cast<long long>(view.value()->global_id(dim, i));
      my_ghost_owner = owner_result.value();
      break;
    }
  }

  // Exchange with the OTHER rank (RANKS 2: "the owning rank" a ghost
  // names, if it names either of us at all, is unambiguously the peer).
  long long send_buf[2] = {my_ghost_gid, static_cast<long long>(my_ghost_owner)};
  long long recv_buf[2] = {-1, -1};
  MPI_Sendrecv(send_buf, 2, MPI_LONG_LONG, 1 - rank, 0, recv_buf, 2, MPI_LONG_LONG, 1 - rank, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
  const long long peer_ghost_gid = recv_buf[0];
  const int peer_ghost_owner = static_cast<int>(recv_buf[1]);

  const fuse::Result<void> local_check = [&]() -> fuse::Result<void> {
    if (peer_ghost_gid < 0) return fuse::Result<void>::success();  // peer had no ghost to report
    if (peer_ghost_owner != rank) return fuse::Result<void>::success();  // peer's ghost names someone else
    bool found = false;
    for (std::size_t i = 0; i < view.value()->entity_count(dim) && !found; ++i) {
      if (view.value()->is_owned(dim, i) && static_cast<long long>(view.value()->global_id(dim, i)) == peer_ghost_gid)
        found = true;
    }
    if (!found) {
      return fuse::Failure{"E-TEST-MOAB-OWNER-MISMATCH",
                            "rank " + std::to_string(rank) + ": the peer's ghost (global id " +
                                std::to_string(peer_ghost_gid) +
                                ") names me as owner, but I report no owned entity with that global id"};
    }
    return fuse::Result<void>::success();
  }();

  // Collective agreement: a disagreement is reported identically on every rank, naming
  // the origin, rather than only on the rank that happened to see it.
  const auto agreed = fuse::agree_outcome(world_handle(), local_check, "E-TEST-MOAB-OWNER-AGREEMENT",
                                           "E-TEST-MOAB-OWNER-COMM-FAILURE", "moab_owner_mpi_test ghost check");
  EXPECT_TRUE(agreed.ok()) << "rank " << rank << ": " << (agreed.ok() ? "" : agreed.error().message);
}

#else  // !FUSE_SPHERE_ATM

TEST(MoabOwnerMpi, SkippedWithoutTheSphereMeshFixture) {
  GTEST_SKIP() << "wholeATM_T.h5m not found; pass -DFUSE_MOAB_MESHFILES=<moab-source>/"
                  "MeshFiles/unittest to compile the real owner-agreement check";
}

#endif  // FUSE_SPHERE_ATM
