// physics-kit/test/physics_mesh_mpi_test.cpp -- fuse_physics::spatial::Mesh
// bound to a REAL multi-rank fuse::backend::NativeMeshView over
// MPI_COMM_WORLD, the collective twin of physics-kit/test/spatial_mesh_
// test.cpp's own FromMeshViewReadsOwnershipAndGlobalIds (a single-process
// stand-in for the same ownership/global-id contract). Owner instruction
// 2026-10-31, item 2: "Using with just MPI_COMM_SELF is dumb. This has to
// run in parallel to split up the compute work" -- registered with RANKS
// 2 3 through fuse_add_mpi_test(... SRC_DIR .) from physics-kit/test/
// CMakeLists.txt, where the fuse::physics_kit target exists (the lead moved
// it here from test/parallel/ on landing; the track had compiled physics-kit
// sources into a fuse-side executable because the target is defined after
// test/ is added).
//
// MPI CANNOT START IN THIS SANDBOX (KNOWN_ISSUES.md C1): this file is
// compiled here, never run; host_pending names the exact `mpirun`
// commands the lead runs on their own host.
//
// REGISTRATION. physics-kit/test/CMakeLists.txt's own fuse_add_gtest calls
// run AFTER add_subdirectory(test) at the root (root CMakeLists.txt's own
// ordering, test/CMakeLists.txt's header comment), so fuse::physics_kit
// does not exist as a CMake target while THIS directory's listfile
// executes -- there is no ordering fix that does not reorder the whole
// tree. physics-kit/test/CMakeLists.txt's own manual block for this one
// target compiles physics-kit's needed .cpp sources directly into this
// executable instead of linking fuse::physics_kit, the same device
// test/partition/'s own fuse_partition_add_gtest wrapper already uses for
// an analogous cross-directory-scope problem.
//
// SCOPE. Mesh/MeshView plumbing only (spatial::Mesh::from_mesh_view,
// ownership, global ids) -- no fe:: basis/quadrature/geometry_map
// involved, so only reference_topology.cpp (for GeometryVariant's own
// free functions Mesh's header uses), mesh.cpp and mesh_utils.cpp are
// compiled in (physics-kit/test/CMakeLists.txt's own source list for this
// target).

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fuse/backend/native_mesh.hpp"
#include "fuse_physics/spatial/mesh.hpp"

#include "fuse/field/collective_agreement.hpp"

#include "fuse_test_main.hpp"

namespace {

using fuse::physics::spatial::EntityHandle;
using fuse::physics::spatial::Mesh;

// A global line of (world_size + 1) vertices, 0..world_size, and
// world_size Edge2 elements, element r connecting vertex r to vertex
// r+1. Rank r owns vertex r and element r; it also holds vertex r+1 --
// owned if r is the last rank (there is no rank r+1 to own it instead),
// a GHOST otherwise, matching the owner's own rule: ownership is a
// per-rank fact about each entity, read through is_owned(), never
// assumed from a contiguous prefix (mesh_view.hpp's own documented
// constraint, carried through spatial::Mesh::from_mesh_view unchanged).
Mesh build_local_strip(int rank, int world_size) {
  fuse::native::NativeMeshBuilder builder(1);
  const bool last_rank = (rank == world_size - 1);
  const int v_lo = builder.add_vertex(rank, static_cast<double>(rank), 0, 0, /*owner=*/rank, /*is_owned=*/true);
  const int v_hi = builder.add_vertex(rank + 1, static_cast<double>(rank + 1), 0, 0,
                                      /*owner=*/last_rank ? rank : rank + 1, /*is_owned=*/last_rank);
  builder.add_cell(fuse::native::CellType::Segment, {v_lo, v_hi}, rank, /*owner=*/rank, /*is_owned=*/true);

  const auto built = builder.build();
  if (!built.ok()) throw std::runtime_error("NativeMeshBuilder::build failed: " + built.error().message);
  const auto view = fuse::native::NativeMeshView::create(built.value(), rank);
  if (!view.ok()) throw std::runtime_error("NativeMeshView::create failed: " + view.error().message);
  return Mesh::from_mesh_view(*view.value());
}

// physics.parallel.physics_mesh_mpi.OwnedCountsSumToTheGlobalTotal --
// every vertex and every element is owned by EXACTLY one rank (the
// strip construction's own invariant); MPI_Allreduce over owned_vertex_
// count()/owned_element_count() must recover the known global totals
// (world_size + 1 vertices, world_size elements), the same "no
// double-count, no gap" property a real partitioned mesh's conservation
// sums depend on.
TEST(PhysicsMeshMpi, OwnedCountsSumToTheGlobalTotal) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  const Mesh mesh = build_local_strip(rank, world_size);
  const std::int64_t local_vertices = static_cast<std::int64_t>(mesh.owned_vertex_count());
  const std::int64_t local_elements = static_cast<std::int64_t>(mesh.owned_element_count());
  std::int64_t global_vertices = 0, global_elements = 0;
  MPI_Allreduce(&local_vertices, &global_vertices, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_elements, &global_elements, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  EXPECT_EQ(global_vertices, static_cast<std::int64_t>(world_size) + 1) << "rank " << rank;
  EXPECT_EQ(global_elements, static_cast<std::int64_t>(world_size)) << "rank " << rank;
}

// physics.parallel.physics_mesh_mpi.GhostVertexGlobalIdAgreesWithItsOwner
// -- rank r's ghost copy of vertex r+1 (held only so its own element r
// has both endpoints available) must report the SAME global id that
// rank r+1 reports for its own OWNED copy of that vertex -- read through
// Mesh::vertex_global_id, not the raw NativeMeshBuilder gid argument
// each rank passed in independently, so this genuinely exercises the
// from_mesh_view plumbing's own id translation rather than only
// confirming two literal constants were typed the same way.
TEST(PhysicsMeshMpi, GhostVertexGlobalIdAgreesWithItsOwner) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const Mesh mesh = build_local_strip(rank, world_size);

  // Rank r's ghost is its local vertex 2 (v_hi, global r+1); its owner,
  // rank r+1, holds that SAME vertex as its local vertex 1 (v_lo). So the
  // gathered quantity is every rank's v_lo global id and ownership, and
  // rank r compares its v_hi against rank r+1's v_lo. The round-1b version
  // gathered v_hi and so compared global r+1 with global r+2; it failed on
  // its first host run (KNOWN_ISSUES.md D105, 2026-11-02) -- a test
  // defect, not a mesh one.
  const std::int64_t my_hi_gid = static_cast<std::int64_t>(mesh.vertex_global_id(2));
  const std::int64_t my_lo_gid = static_cast<std::int64_t>(mesh.vertex_global_id(1));
  const std::uint8_t i_own_lo = mesh.is_vertex_owned(1) ? 1 : 0;
  std::vector<std::int64_t> all_lo_gid(static_cast<std::size_t>(world_size));
  std::vector<std::uint8_t> all_owns_lo(static_cast<std::size_t>(world_size));
  MPI_Allgather(&my_lo_gid, 1, MPI_INT64_T, all_lo_gid.data(), 1, MPI_INT64_T, MPI_COMM_WORLD);
  MPI_Allgather(&i_own_lo, 1, MPI_UNSIGNED_CHAR, all_owns_lo.data(), 1, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);

  const bool last_rank = (rank == world_size - 1);
  const fuse::Result<void> local_check = [&]() -> fuse::Result<void> {
    if (last_rank) return fuse::Result<void>::success();  // no ghost to check; this rank owns its own v_hi
    const int owner = rank + 1;
    if (all_lo_gid[static_cast<std::size_t>(owner)] != my_hi_gid)
      return fuse::Failure{"E-TEST-GHOST-GID-MISMATCH",
                            "rank " + std::to_string(rank) + "'s ghost global id disagreed with rank " +
                                std::to_string(owner) + "'s own owned copy"};
    if (all_owns_lo[static_cast<std::size_t>(owner)] != 1)
      return fuse::Failure{"E-TEST-GHOST-GID-MISMATCH",
                            "rank " + std::to_string(owner) + " did not report owning the vertex rank " +
                                std::to_string(rank) + " holds as a ghost"};
    return fuse::Result<void>::success();
  }();

  // Collective agreement (KNOWN_ISSUES.md D97, owner instruction
  // 2026-10-31): a disagreement is reported identically on every rank,
  // naming the origin, rather than only on the rank that happened to see
  // it -- the same device field_transaction_mpi_test.cpp already uses.
  const auto agreed = fuse::agree_outcome(MPI_Comm_c2f(MPI_COMM_WORLD), local_check, "E-TEST-GHOST-GID-AGREEMENT",
                                           "E-TEST-GHOST-GID-COMM-FAILURE", "physics_mesh_mpi_test ghost check");
  EXPECT_TRUE(agreed.ok()) << "rank " << rank << ": " << (agreed.ok() ? "" : agreed.error().message);
}

}  // namespace
