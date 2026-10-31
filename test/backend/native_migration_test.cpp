// Repartition migration, emulated in one process: pack() and unpack() are
// called directly (no exchange() -- see native_migration.hpp's own comment
// on why this needs no MPI), splitting a mesh into P pieces and recombining
// it under a different assignment, exactly as the partition tests exercise
// RCB against an emulated-rank oracle.

#include "fuse/backend/native_migration.hpp"

#include "fuse/backend/native_backend.hpp"
#include "fuse/partition/rcb.hpp"

#include <numeric>
#include <set>

#include <gtest/gtest.h>

namespace {
using namespace fuse::native;
}  // namespace

TEST(NativeMigration, EmulatedTwoRankMigrationConservesGlobalCellCountAndMeasure) {
  auto arrays1 = generate_native_mesh("hexbox:4x4x4", 0, 1);
  ASSERT_TRUE(arrays1.ok());
  auto view1 = NativeMeshView::create_from_host(arrays1.value(), 0);
  ASSERT_TRUE(view1.ok());
  const std::size_t ncells = view1.value()->cell_count();

  // Migrate everything from "rank 0" onto a new 2-way split: even cell
  // index -> rank 0, odd -> rank 1.
  std::vector<int> dest(ncells);
  for (std::size_t c = 0; c < ncells; ++c) dest[c] = static_cast<int>(c % 2);

  std::vector<std::unique_ptr<NativeMeshView>> sources;
  sources.push_back(std::move(view1.value()));

  auto packed = pack_for_migration(*sources[0], dest, 2);
  ASSERT_TRUE(packed.ok()) << packed.error().message;

  std::vector<std::unique_ptr<NativeMeshView>> result;
  std::set<std::int64_t> seen_gids;
  double total = 0.0;
  for (int r = 0; r < 2; ++r) {
    std::vector<std::vector<std::byte>> received{packed.value()[static_cast<std::size_t>(r)]};
    auto arrays = unpack_migrated_mesh(received, 3, r);
    ASSERT_TRUE(arrays.ok()) << arrays.error().message;
    auto view = NativeMeshView::create_from_host(arrays.value(), r);
    ASSERT_TRUE(view.ok()) << view.error().message;
    for (std::size_t c = 0; c < view.value()->cell_count(); ++c) {
      const auto gid = static_cast<std::int64_t>(view.value()->global_id(3, c));
      EXPECT_TRUE(seen_gids.insert(gid).second) << "cell " << gid << " duplicated across ranks";
      total += view.value()->measure(3, c);
      EXPECT_TRUE(view.value()->is_owned(3, c)) << "a migrated cell is always owned by its "
                                                   "destination";
    }
    result.push_back(std::move(view.value()));
  }
  EXPECT_EQ(seen_gids.size(), ncells);
  EXPECT_NEAR(total, 1.0, 1e-11);
}

TEST(NativeMigration, EmulatedMigrationAtPFromOneToFivePreservesMeasureAndUniqueIds) {
  for (int p : {1, 2, 3, 4, 5}) {
    auto arrays1 = generate_native_mesh("hexbox:5x3x2", 0, 1);
    ASSERT_TRUE(arrays1.ok());
    auto view1 = NativeMeshView::create_from_host(arrays1.value(), 0);
    ASSERT_TRUE(view1.ok());
    const std::size_t ncells = view1.value()->cell_count();

    std::vector<int> dest(ncells);
    for (std::size_t c = 0; c < ncells; ++c) dest[c] = static_cast<int>(c % static_cast<std::size_t>(p));

    auto packed = pack_for_migration(*view1.value(), dest, p);
    ASSERT_TRUE(packed.ok()) << "p=" << p << ": " << packed.error().message;

    std::set<std::int64_t> seen;
    double total = 0.0;
    for (int r = 0; r < p; ++r) {
      std::vector<std::vector<std::byte>> received{packed.value()[static_cast<std::size_t>(r)]};
      auto arrays = unpack_migrated_mesh(received, 3, r);
      ASSERT_TRUE(arrays.ok()) << "p=" << p << ": " << arrays.error().message;
      auto view = NativeMeshView::create_from_host(arrays.value(), r);
      ASSERT_TRUE(view.ok()) << "p=" << p << ": " << view.error().message;
      for (std::size_t c = 0; c < view.value()->cell_count(); ++c) {
        EXPECT_TRUE(seen.insert(static_cast<std::int64_t>(view.value()->global_id(3, c))).second)
            << "p=" << p;
        total += view.value()->measure(3, c);
      }
    }
    EXPECT_EQ(seen.size(), ncells) << "p=" << p;
    EXPECT_NEAR(total, 1.0, 1e-10) << "p=" << p;
  }
}

TEST(NativeMigration, EmulatedRoundTripThroughTwoDifferentPartitionsReproducesTheMesh) {
  auto arrays0 = generate_native_mesh("hexbox:3x3x3", 0, 1);
  ASSERT_TRUE(arrays0.ok());
  auto view0 = NativeMeshView::create_from_host(arrays0.value(), 0);
  ASSERT_TRUE(view0.ok());
  const std::size_t ncells = view0.value()->cell_count();
  double original_total = 0.0;
  std::set<std::int64_t> original_gids;
  for (std::size_t c = 0; c < ncells; ++c) {
    original_total += view0.value()->measure(3, c);
    original_gids.insert(static_cast<std::int64_t>(view0.value()->global_id(3, c)));
  }

  // First pass: split onto 3 ranks by (cell_gid % 3).
  std::vector<int> dest1(ncells);
  for (std::size_t c = 0; c < ncells; ++c) dest1[c] = static_cast<int>(c % 3);
  auto packed1 = pack_for_migration(*view0.value(), dest1, 3);
  ASSERT_TRUE(packed1.ok());

  std::vector<std::unique_ptr<NativeMeshView>> round1;
  for (int r = 0; r < 3; ++r) {
    std::vector<std::vector<std::byte>> received{packed1.value()[static_cast<std::size_t>(r)]};
    auto arrays = unpack_migrated_mesh(received, 3, r);
    ASSERT_TRUE(arrays.ok());
    auto view = NativeMeshView::create_from_host(arrays.value(), r);
    ASSERT_TRUE(view.ok());
    round1.push_back(std::move(view.value()));
  }

  // Second pass: from round1's 3-way split, migrate everything onto a SINGLE
  // rank (a different partition than the original: a 3->1 gather).
  //
  // Each sender's packed buffer is its own entry in `per_dest_round2`, not
  // raw-byte-concatenated into one: the wire-format-v2 header
  // (mesh_pack_device.hpp's own "WIRE FORMAT v2" comment) makes every `pack_for_migration` buffer self-describing
  // (record count + offsets from ITS OWN start), so concatenating two
  // senders' buffers byte-for-byte -- valid under the old header-less
  // format, which this test originally relied on -- hides every record
  // past the first sender's own declared extent from the parser, the same
  // way every OTHER multi-sender case in this suite already keeps one
  // buffer per sender.
  std::vector<std::vector<std::byte>> per_dest_round2;
  for (auto& piece : round1) {
    std::vector<int> dest2(piece->owned_count(3), 0);
    auto packed2 = pack_for_migration(*piece, dest2, 1);
    ASSERT_TRUE(packed2.ok());
    per_dest_round2.push_back(std::move(packed2.value()[0]));
  }
  auto final_arrays = unpack_migrated_mesh(per_dest_round2, 3, 0);
  ASSERT_TRUE(final_arrays.ok()) << final_arrays.error().message;
  auto final_view = NativeMeshView::create_from_host(final_arrays.value(), 0);
  ASSERT_TRUE(final_view.ok()) << final_view.error().message;

  EXPECT_EQ(final_view.value()->cell_count(), ncells);
  double final_total = 0.0;
  std::set<std::int64_t> final_gids;
  for (std::size_t c = 0; c < final_view.value()->cell_count(); ++c) {
    final_total += final_view.value()->measure(3, c);
    final_gids.insert(static_cast<std::int64_t>(final_view.value()->global_id(3, c)));
  }
  EXPECT_NEAR(final_total, original_total, 1e-10);
  EXPECT_EQ(final_gids, original_gids)
      << "a round trip through two different partitions must reproduce the same set of "
        "global cell ids (up to local ordering)";
}

TEST(NativeMigration, PackForMigrationMaterializesNoHostMirror) {
  // pack_for_migration builds its wire
  // records from device_arrays() via fuse::native::device::pack_to, never
  // from the view's own lazy host mirror (cell_vertices/vertex_x/
  // cell_faces) -- host_mirrors_materialized() must stay 0 across the
  // whole call.
  auto arrays = generate_native_mesh("hexbox:3x3x3", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  ASSERT_EQ(view.value()->host_mirrors_materialized(), 0);

  std::vector<int> dest(view.value()->cell_count(), 0);
  auto packed = pack_for_migration(*view.value(), dest, 1);
  ASSERT_TRUE(packed.ok()) << packed.error().message;

  EXPECT_EQ(view.value()->host_mirrors_materialized(), 0)
      << "pack_for_migration touched the source view's lazy host mirror";
}

// ---- rejections -------------------------------------------------------------

TEST(NativeMigration, RefusesAnIncompleteAssignment) {
  auto arrays = generate_native_mesh("hexbox:2x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  std::vector<int> dest(view.value()->cell_count() - 1, 0);  // one short
  auto packed = pack_for_migration(*view.value(), dest, 1);
  ASSERT_FALSE(packed.ok());
  EXPECT_EQ(packed.error().code, "E-MESH-MIGRATE-INCOMPLETE");
}

TEST(NativeMigration, RefusesADestinationOutsideTheCommunicator) {
  auto arrays = generate_native_mesh("hexbox:2x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  std::vector<int> dest(view.value()->cell_count(), 5);  // no rank 5 in a 2-rank comm
  auto packed = pack_for_migration(*view.value(), dest, 2);
  ASSERT_FALSE(packed.ok());
  EXPECT_EQ(packed.error().code, "E-MESH-MIGRATE-RANGE");
}

TEST(NativeMigration, MigrateAtKNullCommIsTheIdentityUpToLocalOrdering) {
  auto arrays = generate_native_mesh("hexbox:2x2x3", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  std::vector<int> dest(view.value()->cell_count(), 0);  // everyone stays on rank 0
  auto migrated = migrate(*view.value(), dest, kNullComm);
  ASSERT_TRUE(migrated.ok()) << migrated.error().message;
  EXPECT_EQ(migrated.value()->cell_count(), view.value()->cell_count());
  EXPECT_EQ(migrated.value()->vertex_count(), view.value()->vertex_count())
      << "adjacent cells share vertices; unpack must dedup by global id rather than minting a "
        "new vertex per incoming closure entry, or this count inflates";
  double total = 0.0;
  for (std::size_t c = 0; c < migrated.value()->cell_count(); ++c) {
    total += migrated.value()->measure(3, c);
    EXPECT_TRUE(migrated.value()->is_owned(3, c));
  }
  EXPECT_NEAR(total, 1.0, 1e-11);
  for (std::size_t v = 0; v < migrated.value()->vertex_count(); ++v) {
    EXPECT_TRUE(migrated.value()->is_owned(0, v)) << "kNullComm has one rank, which owns "
                                                     "everything";
  }
}

// Device-resident merge:
// `migrate` rebuilds the RESULT view through `device::pack_to` ->
// `exchange_migration_buffers<Space>` -> `device::unpack_migrated_mesh_
// device<Space>` -> `resolve_vertex_ownership_device<Space>` ->
// `NativeMeshView::create<Space>` end to end -- checked here with EAGER
// accessors only (global_id/is_owned/entity_count/host_mirrors_
// materialized never call ensure_host_mirror(); see NativeMeshView's own
// header comment), so this test does not itself materialize what it is
// trying to prove stayed unmaterialized.
TEST(NativeMigration, FullMigrateRoundTripMaterializesNoHostMirrorOnEitherView) {
  auto arrays = generate_native_mesh("hexbox:2x2x3", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  ASSERT_EQ(view.value()->host_mirrors_materialized(), 0);

  std::vector<int> dest(view.value()->cell_count(), 0);  // everyone stays on rank 0
  auto migrated = migrate(*view.value(), dest, kNullComm);
  ASSERT_TRUE(migrated.ok()) << migrated.error().message;

  EXPECT_EQ(view.value()->host_mirrors_materialized(), 0)
      << "migrate touched the SOURCE view's lazy host mirror";
  EXPECT_EQ(migrated.value()->host_mirrors_materialized(), 0)
      << "migrate's device-resident unpack/merge touched the RESULT view's lazy host mirror "
        "before any caller asked it to";

  EXPECT_EQ(migrated.value()->cell_count(), view.value()->cell_count());
  EXPECT_EQ(migrated.value()->vertex_count(), view.value()->vertex_count());
  for (std::size_t c = 0; c < migrated.value()->cell_count(); ++c) {
    EXPECT_TRUE(migrated.value()->is_owned(3, c));
  }
}

TEST(NativeMigration, MigrateRefusesAnIncompleteAssignmentAtKNullComm) {
  auto arrays = generate_native_mesh("hexbox:2x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  std::vector<int> dest(view.value()->cell_count() - 1, 0);
  auto migrated = migrate(*view.value(), dest, kNullComm);
  ASSERT_FALSE(migrated.ok());
  EXPECT_EQ(migrated.error().code, "E-MESH-MIGRATE-INCOMPLETE");
}

// Device-resident merge, POLYHEDRON LIMITATION: no OTHER test in this suite's migration/halo/
// repartition coverage exercises `CellType::Polyhedron` at all (every
// generator here is hexbox/line/quad -- fixed-shape Hex/Segment/Quad),
// so this is the first evidence the device-resident unpack's Polyhedron
// path (mesh_merge_device.hpp's own doc comment) produces a geometrically
// correct cell at all, under the declared limitation it also documents:
// a migrated Polyhedron's faces are rebuilt as NEW, PRIVATE entries (an
// identity `cell_face_indices`, never deduplicated against another cell's
// face) rather than `NativeMeshBuilder::add_polyhedron_cell`'s own cross-
// cell face-sharing dedup -- invisible here because a single migrated
// TET has no neighbour to share a face with either way; this checks
// conservation (cell/vertex count, measure, ownership), not face count.
TEST(NativeMigration, PolyhedronCellSurvivesADeviceResidentMigrationRoundTrip) {
  NativeMeshBuilder b(3);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 0.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 0.0, 1.0, 0, true);
  // The same outward-CCW face table cell_topology.hpp's `fixed_face_
  // corners(CellType::Tet, ...)` uses, spelled out explicitly as a
  // Polyhedron's own face-to-vertex CSR instead of relying on Tet's
  // fixed-shape path -- exercising the SAME wire records/device unpack a
  // true general polyhedron (a cut cell, an agglomerated volume) would.
  const std::vector<std::vector<int>> faces = {
      {v1, v2, v3}, {v0, v3, v2}, {v0, v1, v3}, {v0, v2, v1}};
  b.add_polyhedron_cell(faces, /*gid=*/100, /*owner=*/0, /*is_owned=*/true);
  auto arrays = b.build();
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok()) << view.error().message;
  ASSERT_EQ(view.value()->cell_count(), 1u);
  const double original_measure = view.value()->measure(3, 0);
  EXPECT_NEAR(original_measure, 1.0 / 6.0, 1e-12) << "unit right tet, volume 1/6";

  std::vector<int> dest(view.value()->cell_count(), 0);
  auto migrated = migrate(*view.value(), dest, kNullComm);
  ASSERT_TRUE(migrated.ok()) << migrated.error().message;

  EXPECT_EQ(migrated.value()->cell_count(), 1u);
  EXPECT_EQ(migrated.value()->vertex_count(), 4u);
  ASSERT_TRUE(migrated.value()->is_owned(3, 0));
  EXPECT_EQ(static_cast<std::int64_t>(migrated.value()->global_id(3, 0)), 100);
  EXPECT_NEAR(migrated.value()->measure(3, 0), original_measure, 1e-12)
      << "a device-resident Polyhedron migration must reproduce the same cell volume";
  for (std::size_t v = 0; v < migrated.value()->vertex_count(); ++v) {
    EXPECT_TRUE(migrated.value()->is_owned(0, v));
  }
}
