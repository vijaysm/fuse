// test/parallel/native_halo_mpi_test.cpp -- the native mesh backend's
// planar remap halo ("Halo" track, docs/dev/native-halo.md) under a REAL
// mpiexec. test/backend/native_halo_test.cpp covers the pure-function
// pieces and the in-process-emulated 16x16/24x24 quad-grid scenario at 2
// and 3 simulated ranks; this file is the same scenario at 2 and 3 REAL
// ranks, exercising the one thing the in-process emulation cannot: the
// REAL `MPI_Allgather`/`MPI_Allgatherv`/`MPI_Alltoallv` collectives inside
// `fuse::native::build_native_remap_halo` itself, and the REAL
// `fuse::linalg::make_mpi_exchanger` filling the adopted cells' field
// values (as opposed to the tier-0 file's in-process stand-in that reads
// a peer's array directly).
//
// NOT YET RUN: this sandbox cannot initialize MPI at all, not even at one
// rank (docs/dev/testing.md; KNOWN_ISSUES.md C1) -- the same reason
// test/backend/native_migration_mpi_test.cpp and
// test/parallel/remap_halo_mpi_test.cpp are "built, not run" here.
// Registered only when FUSE_ENABLE_MPI is on (test/CMakeLists.txt); its
// outcome comes from a host run.

#include "fuse/backend/native_halo.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/model/coupling_graph.hpp"

#include "fuse_test_main.hpp"

namespace {

using namespace fuse::native;  // NOLINT
using fuse::ComponentId;
using fuse::CouplingGraph;
using fuse::MeshDecl;
using fuse::MeshId;
using fuse::Region;
using fuse::RegionId;
using fuse::RemapPlan;

constexpr const char* kUSource = "generated:quad:16x16";
constexpr const char* kVSource = "generated:quad:24x24";

CommHandle world_handle() { return static_cast<CommHandle>(MPI_Comm_c2f(MPI_COMM_WORLD)); }

[[nodiscard]] double linear_field(double x, double y) { return 3.0 + 2.0 * x - 5.0 * y; }

struct Fixture {
  NativeMeshBackend backend;
  CouplingGraph graph;
  MeshId u_mid{0}, v_mid{0};
  RegionId u_region, v_region;
  RemapPlan plan;

  explicit Fixture(int rank, int size) : backend(rank, size) {
    backend.set_communicator(world_handle());
    EXPECT_TRUE(backend.load(MeshDecl{.name = "U", .source = kUSource}).ok());
    EXPECT_TRUE(backend.load(MeshDecl{.name = "V", .source = kVSource}).ok());
    u_mid = graph.add_mesh(MeshDecl{.name = "U", .source = kUSource});
    v_mid = graph.add_mesh(MeshDecl{.name = "V", .source = kVSource});
    u_region = graph.add_region(
        Region{.name = "u_all", .mesh = u_mid, .entity_set = "", .dimension = 2});
    v_region = graph.add_region(
        Region{.name = "v_all", .mesh = v_mid, .entity_set = "", .dimension = 2});
    plan.src = u_region;
    plan.tgt = v_region;
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// moab_backend.hpp's own boundary, restated for the native backend:
// `communicator_size() > 1` rejects when no halo was built; `== 1` must
// NOT. Checked at every registered rank count (RANKS 2 3 in
// test/CMakeLists.txt) rather than assumed to hold from the tier-0 file's
// own coverage of it with a placeholder (never-dereferenced) communicator
// handle.
// ---------------------------------------------------------------------------
TEST(NativeHaloMpi, ERemapParallelRefusesAboveOneRankWithNoHalo) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Fixture f(rank, size);

  // Deliberately no build_remap_halo() call.
  const auto w = fuse::build_weights(f.plan, f.graph, f.backend);
  if (size == 1) {
    ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;
  } else {
    ASSERT_FALSE(w.ok()) << "size=" << size;
    EXPECT_EQ(w.error().code, "E-REMAP-PARALLEL");
  }
}

// ---------------------------------------------------------------------------
// build_remap_halo -> build_weights -> make_mpi_exchanger -> apply_weights,
// the complete sequence remap_plan.hpp's own header documents, under a
// REAL communicator. At size == 2 the 8+8/12+12 row split aligns exactly
// (see the tier-0 file's own fixture comment) and zero cells are adopted;
// at every other registered size (3) it is misaligned and some are.
// ---------------------------------------------------------------------------
TEST(NativeHaloMpi, BuildRemapHaloThenBuildWeightsThenExchangeReproducesALinearFieldToOrderH) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Fixture f(rank, size);

  const auto adopted = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  ASSERT_TRUE(adopted.ok()) << adopted.error().code << ": " << adopted.error().message;
  if (size == 2) {
    EXPECT_EQ(adopted.value(), 0u) << "rank " << rank
                                   << ": the 8+8/12+12 row split aligns exactly at y = 0.5";
  }

  const auto w = fuse::build_weights(f.plan, f.graph, f.backend);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;

  double local_row_defect = w.value().max_row_defect;
  double global_row_defect = 0.0;
  MPI_Allreduce(&local_row_defect, &global_row_defect, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  EXPECT_LE(global_row_defect, 1e-12) << "row-sum defect, reduced over every rank";

  const auto u_view_r = f.backend.native_view(f.u_mid, ComponentId{});
  const auto v_view_r = f.backend.native_view(f.v_mid, ComponentId{});
  ASSERT_TRUE(u_view_r.ok());
  ASSERT_TRUE(v_view_r.ok());
  const NativeMeshView& u = *u_view_r.value();
  const NativeMeshView& v = *v_view_r.value();

  // make_mpi_exchanger needs a GhostPattern and the OWNED count, exactly
  // as fuse::linalg::DistVector's own constructor contract states
  // (comm.hpp) -- native_backend.hpp's own remap_halo_pattern() returns
  // precisely that pattern, unchanged (NM-3: the ghost offset a SECOND
  // halo on the same source would need rides a separate accessor,
  // remap_halo_ghost_offset, precisely so this type stays source-
  // compatible with examples/brusselator's own consumer; 0 here since
  // this fixture builds only ONE halo on `u_mid`).
  const auto pattern_r = f.backend.remap_halo_pattern(f.u_mid, f.v_mid, 2);
  ASSERT_TRUE(pattern_r.ok()) << pattern_r.error().message;
  const auto offset_r = f.backend.remap_halo_ghost_offset(f.u_mid, f.v_mid, 2);
  ASSERT_TRUE(offset_r.ok()) << offset_r.error().message;
  ASSERT_EQ(offset_r.value(), 0u);
  const std::size_t owned_n = u.owned_count(2);
  auto exchanger = fuse::linalg::make_mpi_exchanger(world_handle(), pattern_r.value(), owned_n);
  ASSERT_NE(exchanger, nullptr);
  EXPECT_EQ(exchanger->ghost_size(), pattern_r.value().owner_rank.size());

  // owned_block_index (native_halo.hpp) gives each owned cell's OWN
  // position among the owned block -- the same numbering
  // `make_mpi_exchanger`'s pattern addresses via `remote_local_index`.
  std::vector<double> owned_field(owned_n);
  const auto owned_idx = owned_block_index(u);
  for (std::size_t c = 0; c < u.entity_count(2); ++c) {
    if (!u.is_owned(2, c)) continue;
    const auto p = u.centroid(2, c);
    owned_field[owned_idx[c]] = linear_field(p.x, p.y);
  }
  std::vector<double> ghost_field(exchanger->ghost_size());
  const auto updated =
      exchanger->update(owned_field.data(), owned_n, ghost_field.data(), ghost_field.size());
  ASSERT_TRUE(updated.ok()) << updated.error().message;

  std::vector<double> src_field(u.entity_count(2));
  for (std::size_t c = 0; c < u.entity_count(2); ++c) {
    if (u.is_owned(2, c)) {
      src_field[c] = owned_field[owned_idx[c]];
    } else {
      src_field[c] = ghost_field[c - owned_n];
    }
  }
  std::vector<double> dst_field(w.value().n_rows);
  const auto applied = fuse::apply_weights(w.value(), src_field.data(), dst_field.data(), 1);
  ASSERT_TRUE(applied.ok()) << applied.error().message;

  // The analytic O(h) ceiling from the tier-0 file's own fixture (a
  // first-order conservative remap reproduces a CONSTANT field exactly --
  // checked separately below -- but a LINEAR one only to the field's own
  // variation across the coarsest, U, source cell, width 1/16).
  constexpr double kUCellWidth = 1.0 / 16.0;
  constexpr double kBound = (2.0 + 5.0) * kUCellWidth;
  double max_err = 0.0;
  for (std::size_t j = 0; j < v.entity_count(2); ++j) {
    if (!v.is_owned(2, j)) continue;
    const auto p = v.centroid(2, j);
    max_err = std::max(max_err, std::fabs(dst_field[j] - linear_field(p.x, p.y)));
  }
  EXPECT_LE(max_err, kBound) << "rank " << rank;

  // The sharp check on the exchange plumbing itself: a CONSTANT field
  // must reproduce exactly (to rounding), regardless of where its ghost
  // values came from.
  constexpr double kConst = 7.0;
  std::vector<double> const_owned(owned_n, kConst);
  std::vector<double> const_ghost(exchanger->ghost_size());
  ASSERT_TRUE(
      exchanger->update(const_owned.data(), owned_n, const_ghost.data(), const_ghost.size())
          .ok());
  std::vector<double> const_src(u.entity_count(2));
  for (std::size_t c = 0; c < u.entity_count(2); ++c) {
    const_src[c] = u.is_owned(2, c) ? kConst : const_ghost[c - owned_n];
  }
  std::vector<double> const_dst(w.value().n_rows);
  ASSERT_TRUE(fuse::apply_weights(w.value(), const_src.data(), const_dst.data(), 1).ok());
  for (std::size_t j = 0; j < v.entity_count(2); ++j) {
    if (!v.is_owned(2, j)) continue;
    EXPECT_NEAR(const_dst[j], kConst, 1e-12) << "rank " << rank << " row " << j;
  }
}

// ---------------------------------------------------------------------------
// Idempotence under a real communicator: a second build_remap_halo() call
// for the SAME (source, target, dimension) must adopt nothing further.
// ---------------------------------------------------------------------------
TEST(NativeHaloMpi, BuildRemapHaloIsIdempotentUnderARealCommunicator) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  Fixture f(rank, size);

  const auto first = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  ASSERT_TRUE(first.ok()) << first.error().code << ": " << first.error().message;
  const auto second = fuse::build_remap_halo(f.plan, f.graph, f.backend);
  ASSERT_TRUE(second.ok()) << second.error().code << ": " << second.error().message;
  EXPECT_EQ(second.value(), 0u) << "rank " << rank
                                << ": every cell the pair needs was already adopted by the "
                                   "first call";
}

// ---------------------------------------------------------------------------
// NM-2/NM-5: `build_native_remap_halo`'s pre-exchange rejections (resolving
// either view, the dimension check) are rank-LOCAL -- before this fix,
// returned directly with no reconciliation, so a rank that fails locally
// while its peers do not would return while those peers block forever on
// `allgather_boxes`'s collective. Only RANK 0 is given the wrong
// dimension here (every other rank asks for the dimension U/V actually
// are), a data-dependent mismatch no earlier check sees coming -- exactly
// the shape `migrate()`'s own `E-MESH-MIGRATE-PEER-REFUSED` test exercises
// for the sibling migration path. Pre-fix: rank 0 returns immediately
// while ranks 1.. hang in the box allgather, so this ROW (registered with
// a `--timeout`, docs/dev/testing.md) reports Timeout rather than a test
// failure. Post-fix: every rank returns within the timeout, rank 0 with
// its own E-REMAP-DIM and every other rank with E-MESH-HALO-PEER-REFUSED.
// ---------------------------------------------------------------------------
TEST(NativeHaloMpi, ADisagreeingDimensionRefusesOnEveryRankInsteadOfHanging) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  ASSERT_GE(size, 2) << "this row is only meaningful with more than one rank";
  Fixture f(rank, size);

  const int dim = (rank == 0) ? 1 : 2;
  const auto r = build_native_remap_halo(f.backend, f.u_mid, f.v_mid, dim, ComponentId{});
  ASSERT_FALSE(r.ok()) << "rank " << rank << " unexpectedly succeeded";
  if (rank == 0) {
    EXPECT_EQ(r.error().code, "E-REMAP-DIM") << "rank " << rank;
  } else {
    EXPECT_EQ(r.error().code, "E-MESH-HALO-PEER-REFUSED") << "rank " << rank;
  }
}
