// physics-kit/test/physics_dof_map_mpi_test.cpp -:
// fuse_physics::fe::DofMap's own ghost resolution (pass 3 of its
// constructor, dof_map.cpp) over a REAL multi-rank partition with
// UNEQUAL owned-dof counts per rank -- the exact shape the issue tracker
// names (the round-1 padded-allgather version only mispacked its
// sentinel markers when max_owned > n_owned_here on some rank, which
// needs ranks to disagree on how many dofs they own; a single-rank or
// equal-split partition can never exercise it). Registered with RANKS
// 2 3 (physics-kit/test/CMakeLists.txt), build-only under
// FUSE_ENABLE_MPI=ON -- MPI cannot start in this sandbox
//, matching every other *_mpi_test.cpp in this
// tree; host_pending names the exact mpirun commands.
//
// PARTITION. A 1-D line of Edge2 elements, split so RANK r OWNS
// (r + 1) ELEMENTS -- 1, 2, 3, ... -- strictly increasing, so no two
// ranks in a RANKS 2 3 run ever own the same count and `max_owned`
// (the largest owned-dof count across ranks) always exceeds every
// other rank's own count except the last rank's. Rank r holds vertex
// layers [offset_r, offset_r + (r+1)] (global ids), where offset_r is
// the running total of elements owned by ranks before it; the TOP
// layer is a ghost (owned by rank r+1) unless r is the last rank, the
// SAME "every rank owns its own low end" rule build_local_strip uses
// in physics_mesh_mpi_test.cpp, generalised from exactly 1 element per
// rank to a per-rank-varying count.
//
// ASSERTIONS (both never executed, build-verified only, per this
// file's own header comment above):
//  1. GhostDofResolvesToOwningRanksFirstOwnedIndex -- a non-last rank's
//     one ghost dof (its top vertex, owned by rank+1) must resolve, via
//     DofMap's new point-to-point pass 3 (dof_map.cpp, item X), to
//     EXACTLY `row_lo_by_rank()[rank + 1]`: the next rank's own first
//     owned global index, since that ghost vertex is local index 0
//     within the next rank's own owned block by this partition's own
//     construction (its v_lo). A self-contained, locally-checkable
//     identity -- no cross-rank gather needed to state it, since
//     `row_lo_by_rank()` (DofMap's pass 2, unchanged this round) is
//     already the same contiguous-offset table every rank holds in
//     full.
//  2. GhostCountMatchesExactlyOneNonLastRank -- `ghost_indices().size()`
//     is exactly 1 for every rank except the last (0 for it) -- guards
//     against a padding-shaped regression (own bogus all-zero
//     ghost insertion) reappearing, though the new point-to-point
//     exchange has no padding concept left to leak one from.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>

#include "fuse/backend/native_mesh.hpp"
#include "fuse/field/collective_agreement.hpp"
#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/spatial/mesh.hpp"

#include "fuse_test_main.hpp"

namespace {

using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::Mesh;

/// Rank r owns (r + 1) elements -- strictly increasing across ranks,
/// so this file's own header comment's shape (max_owned >
/// n_owned_here on every rank but the last) always holds for RANKS >= 2.
unsigned n_elements_owned(int rank) { return static_cast<unsigned>(rank) + 1; }

/// The running total of elements owned by every rank BEFORE `rank` --
/// this rank's own global vertex/element numbering offset.
std::int64_t global_offset(int rank) {
  std::int64_t total = 0;
  for (int r = 0; r < rank; ++r) total += static_cast<std::int64_t>(n_elements_owned(r));
  return total;
}

Mesh build_local_strip_unequal(int rank, int world_size) {
  fuse::native::NativeMeshBuilder builder(1);
  const bool last_rank = (rank == world_size - 1);
  const unsigned n_r = n_elements_owned(rank);
  const std::int64_t v_lo_global = global_offset(rank);

  // n_r + 1 local vertices: v_lo_global .. v_lo_global + n_r. The top
  // one (index n_r) is a ghost, owned by rank + 1, unless last_rank.
  std::vector<int> v(n_r + 1);
  for (unsigned k = 0; k <= n_r; ++k) {
    const std::int64_t gid = v_lo_global + static_cast<std::int64_t>(k);
    const bool owned_here = (k < n_r) || last_rank;
    const int owner_rank = owned_here ? rank : rank + 1;
    v[k] = builder.add_vertex(gid, static_cast<double>(gid), 0, 0, owner_rank, owned_here);
  }
  for (unsigned k = 0; k < n_r; ++k) {
    const std::int64_t element_gid = v_lo_global + static_cast<std::int64_t>(k);
    builder.add_cell(fuse::native::CellType::Segment, {v[k], v[k + 1]}, element_gid, rank, /*is_owned=*/true);
  }

  const auto built = builder.build();
  if (!built.ok()) throw std::runtime_error("NativeMeshBuilder::build failed: " + built.error().message);
  const auto view = fuse::native::NativeMeshView::create(built.value(), rank);
  if (!view.ok()) throw std::runtime_error("NativeMeshView::create failed: " + view.error().message);
  return Mesh::from_mesh_view(*view.value());
}

TEST(PhysicsDofMapMpi, GhostDofResolvesToOwningRanksFirstOwnedIndex) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const Mesh mesh = build_local_strip_unequal(rank, world_size);
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry, comm_f);

  const bool last_rank = (rank == world_size - 1);
  const auto& row_lo = dof_map.row_lo_by_rank();

  const fuse::Result<void> local_check = [&]() -> fuse::Result<void> {
    if (last_rank) {
      if (!dof_map.ghost_indices().empty())
        return fuse::Failure{"E-TEST-DOF-MAP-UNEXPECTED-GHOST",
                              "last rank " + std::to_string(rank) + " owns every vertex it holds; expected no ghosts"};
      return fuse::Result<void>::success();
    }
    if (dof_map.ghost_indices().size() != 1)
      return fuse::Failure{"E-TEST-DOF-MAP-GHOST-COUNT",
                            "rank " + std::to_string(rank) + " expected exactly one ghost dof, saw " +
                                std::to_string(dof_map.ghost_indices().size())};
    const auto resolved = dof_map.ghost_indices().front();
    const auto expected = row_lo.at(static_cast<std::size_t>(rank + 1));  // next rank's own first owned dof
    if (resolved != expected)
      return fuse::Failure{"E-TEST-DOF-MAP-GHOST-MISMATCH",
                            "rank " + std::to_string(rank) + "'s ghost resolved to " + std::to_string(resolved) +
                                ", expected rank " + std::to_string(rank + 1) + "'s first owned dof " +
                                std::to_string(expected)};
    return fuse::Result<void>::success();
  }();

  // Collective agreement (the issue tracker
  //): a disagreement is reported identically on every rank, naming
  // the origin, rather than only on the rank that happened to see it --
  // the same device physics_mesh_mpi_test.cpp's own ghost check uses.
  const auto agreed = fuse::agree_outcome(comm_f, local_check, "E-TEST-DOF-MAP-GHOST-AGREEMENT",
                                           "E-TEST-DOF-MAP-GHOST-COMM-FAILURE", "physics_dof_map_mpi_test ghost check");
  EXPECT_TRUE(agreed.ok()) << "rank " << rank << ": " << (agreed.ok() ? "" : agreed.error().message);
}

TEST(PhysicsDofMapMpi, GhostCountMatchesExactlyOneNonLastRank) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const Mesh mesh = build_local_strip_unequal(rank, world_size);
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry, comm_f);

  const bool last_rank = (rank == world_size - 1);
  const std::size_t expected_ghosts = last_rank ? 0u : 1u;
  EXPECT_EQ(dof_map.ghost_indices().size(), expected_ghosts) << "rank " << rank;
  // own regression shape: the old padded-allgather version could
  // insert a bogus all-zero-key ghost when a padding slot's sentinel
  // was not marked (unreachable on the single-rank path that round's
  // own tier-0 testing covered); the new point-to-point exchange has
  // no padding slot to leak one from, so n_dofs() seen locally (owned
  // + ghost) must match n_local_dofs() + expected_ghosts exactly, not
  // more.
  EXPECT_EQ(dof_map.local_to_global().size(),
           static_cast<std::size_t>(dof_map.n_local_dofs()) + expected_ghosts)
      << "rank " << rank;
}

}  // namespace
