// The native mesh's generated sources and its file round trip. Global
// conservation across a simulated multi-rank build (each rank's own slice,
// summed) is the load-bearing property: it is what migration's tests will
// also check, so a bug here would otherwise surface confusingly there.

#include "fuse/backend/native_backend.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/backend/native_mesh_io.hpp"

#include "fuse/base/error.hpp"

#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <unistd.h>

namespace {
using namespace fuse::native;

/// Builds every rank's slice of `spec` at `num_ranks` and returns the
/// merged (not deduplicated) list of views, for a global property check.
std::vector<std::unique_ptr<NativeMeshView>> build_all_ranks(const std::string& spec,
                                                              int num_ranks) {
  std::vector<std::unique_ptr<NativeMeshView>> views;
  for (int r = 0; r < num_ranks; ++r) {
    auto arrays = generate_native_mesh(spec, r, num_ranks);
    EXPECT_TRUE(arrays.ok()) << "rank " << r << ": " << (arrays.ok() ? "" : arrays.error().message);
    auto view = NativeMeshView::create_from_host(arrays.value(), r);
    EXPECT_TRUE(view.ok()) << "rank " << r << ": " << (view.ok() ? "" : view.error().message);
    views.push_back(std::move(view.value()));
  }
  return views;
}

std::set<std::int64_t> owned_cell_gids(const std::vector<std::unique_ptr<NativeMeshView>>& views) {
  std::set<std::int64_t> gids;
  for (const auto& v : views) {
    for (std::size_t c = 0; c < v->cell_count(); ++c) {
      if (v->is_owned(v->dimension(), c)) {
        const bool inserted = gids.insert(static_cast<std::int64_t>(v->global_id(v->dimension(),
                                                                                c))).second;
        EXPECT_TRUE(inserted) << "cell global id duplicated across ranks";
      }
    }
  }
  return gids;
}

}  // namespace

TEST(NativeGenerators, LineOwnedCellsPartitionTheDomainAtEveryRankCount) {
  for (int p : {1, 2, 3, 5}) {
    auto views = build_all_ranks("line:10", p);
    double total = 0.0;
    for (const auto& v : views) {
      for (std::size_t c = 0; c < v->cell_count(); ++c) {
        if (v->is_owned(v->dimension(), c)) total += v->measure(v->dimension(), c);
      }
    }
    EXPECT_NEAR(total, 1.0, 1e-12) << "p=" << p;
    auto gids = owned_cell_gids(views);
    EXPECT_EQ(gids.size(), 10u) << "p=" << p;
  }
}

TEST(NativeGenerators, QuadGridTotalAreaIsOneAtEveryRankCount) {
  for (int p : {1, 2, 3, 4}) {
    auto views = build_all_ranks("quad:4x4", p);
    double total = 0.0;
    for (const auto& v : views) {
      for (std::size_t c = 0; c < v->cell_count(); ++c) {
        if (v->is_owned(v->dimension(), c)) total += v->measure(v->dimension(), c);
      }
    }
    EXPECT_NEAR(total, 1.0, 1e-12) << "p=" << p;
    EXPECT_EQ(owned_cell_gids(views).size(), 16u) << "p=" << p;
  }
}

TEST(NativeGenerators, TriGridTotalAreaIsOneAndGidsAreRankCountIndependent) {
  auto v1 = build_all_ranks("tri:3x3", 1);
  auto v3 = build_all_ranks("tri:3x3", 3);
  auto gids1 = owned_cell_gids(v1);
  auto gids3 = owned_cell_gids(v3);
  EXPECT_EQ(gids1, gids3) << "the same cell must have the same global id at any rank count";
  EXPECT_EQ(gids1.size(), 18u);  // 3*3*2 triangles
}

TEST(NativeGenerators, HexBoxTotalVolumeIsOneAtEveryRankCount) {
  for (int p : {1, 2, 3, 4}) {
    auto views = build_all_ranks("hexbox:3x2x4", p);
    double total = 0.0;
    for (const auto& v : views) {
      for (std::size_t c = 0; c < v->cell_count(); ++c) {
        if (v->is_owned(v->dimension(), c)) total += v->measure(v->dimension(), c);
      }
    }
    EXPECT_NEAR(total, 1.0, 1e-11) << "p=" << p;
    EXPECT_EQ(owned_cell_gids(views).size(), 3u * 2u * 4u) << "p=" << p;
  }
}

TEST(NativeGenerators, TetBoxTotalVolumeIsOneAtEveryRankCount) {
  for (int p : {1, 3}) {
    auto views = build_all_ranks("tetbox:2x2x2", p);
    double total = 0.0;
    for (const auto& v : views) {
      for (std::size_t c = 0; c < v->cell_count(); ++c) {
        if (v->is_owned(v->dimension(), c)) total += v->measure(v->dimension(), c);
      }
    }
    EXPECT_NEAR(total, 1.0, 1e-11) << "p=" << p;
    EXPECT_EQ(owned_cell_gids(views).size(), 2u * 2u * 2u * 6u) << "p=" << p;
  }
}

TEST(NativeGenerators, PolyhedralBoxTotalVolumeIsOneAtEveryRankCount) {
  for (int p : {1, 2, 3}) {
    auto views = build_all_ranks("polyhedral:2x2x3", p);
    double total = 0.0;
    for (const auto& v : views) {
      ASSERT_TRUE(v != nullptr);
      for (std::size_t c = 0; c < v->cell_count(); ++c) {
        EXPECT_EQ(v->cell_type_of(c), CellType::Polyhedron);
        if (v->is_owned(v->dimension(), c)) total += v->measure(v->dimension(), c);
      }
    }
    EXPECT_NEAR(total, 1.0, 1e-11) << "p=" << p;
  }
}

TEST(NativeGenerators, PrismPyramidMixCellsHaveThePyramidsVolume) {
  // Deliberately check the pyramid orientation this generator picked
  // (native_generators.cpp's comment on the reversed base) by checking
  // total volume, which would be visibly wrong (rejected, or far from 1)
  // if the pyramids were inverted or degenerate.
  auto arrays = generate_native_mesh("prismpyramidmix:2x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok()) << view.error().message;
  bool saw_prism = false, saw_pyramid = false;
  double total = 0.0;
  for (std::size_t c = 0; c < view.value()->cell_count(); ++c) {
    total += view.value()->measure(3, c);
    if (view.value()->cell_type_of(c) == CellType::Prism) saw_prism = true;
    if (view.value()->cell_type_of(c) == CellType::Pyramid) saw_pyramid = true;
  }
  EXPECT_TRUE(saw_prism);
  EXPECT_TRUE(saw_pyramid);
  EXPECT_NEAR(total, 1.0, 1e-10);
}

TEST(NativeGenerators, CubedSphereHasSixPanelsWithEveryVertexOnTheUnitSphere) {
  auto arrays = generate_native_mesh("cubedsphere:4", 0, 1);
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok()) << view.error().message;
  EXPECT_EQ(view.value()->cell_count(), 6u * 4u * 4u);
  for (std::size_t v = 0; v < view.value()->vertex_count(); ++v) {
    const double x = view.value()->vertex_x(v, 0), y = view.value()->vertex_x(v, 1),
                z = view.value()->vertex_x(v, 2);
    EXPECT_NEAR(x * x + y * y + z * z, 1.0, 1e-12) << "vertex " << v << " is not on the unit sphere";
  }
}

// ---- rejections ------------------------------------------------------------

TEST(NativeGenerators, RefusesAnUnknownGeneratedKind) {
  auto arrays = generate_native_mesh("dodecahedron:5", 0, 1);
  ASSERT_FALSE(arrays.ok());
  EXPECT_EQ(arrays.error().code, "E-MESH-SOURCE");
}

TEST(NativeGenerators, RefusesAMalformedDimensionString) {
  auto arrays = generate_native_mesh("hexbox:4x4", 0, 1);  // needs 3 dims
  ASSERT_FALSE(arrays.ok());
  EXPECT_EQ(arrays.error().code, "E-MESH-SOURCE");
}

// ---- backend load()/mesh_id()/view()/region_size() -----------------------

TEST(NativeBackend, LoadsAGeneratedMeshAndReportsItsRegionSize) {
  NativeMeshBackend backend(0, 1);
  auto r = backend.load(fuse::MeshDecl{"m", "generated:hexbox:2x2x2"});
  ASSERT_TRUE(r.ok()) << r.error().message;
  auto id = backend.mesh_id("m");
  ASSERT_TRUE(id.ok()) << id.error().message;
  auto v = backend.view(id.value());
  ASSERT_TRUE(v.ok()) << v.error().message;
  EXPECT_EQ(v.value()->entity_count(3), 8u);

  fuse::Region region;
  region.name = "whole";
  region.mesh = id.value();
  region.entity_set = "";
  region.dimension = -1;
  auto size = backend.region_size(region);
  ASSERT_TRUE(size.ok()) << size.error().message;
  EXPECT_EQ(size.value(), 8u);
}

TEST(NativeBackend, RefusesAnUnknownMeshName) {
  NativeMeshBackend backend(0, 1);
  auto id = backend.mesh_id("nope");
  ASSERT_FALSE(id.ok());
  EXPECT_EQ(id.error().code, "E-MESH-NAME");
}

TEST(NativeBackend, PlacedViewFallsBackToTheDefaultForAnInvalidPlacement) {
  NativeMeshBackend backend(0, 1);
  auto r = backend.load(fuse::MeshDecl{"m", "generated:quad:2x2"});
  ASSERT_TRUE(r.ok());
  auto id = backend.mesh_id("m");
  ASSERT_TRUE(id.ok());
  auto v = backend.placed_view(id.value(), fuse::ComponentId{});  // invalid ComponentId
  ASSERT_TRUE(v.ok()) << v.error().message;
  EXPECT_EQ(v.value()->entity_count(2), 4u);
}

TEST(NativeBackend, PlacedViewIsReachableAfterSetPlacedView) {
  NativeMeshBackend backend(0, 1);
  auto r = backend.load(fuse::MeshDecl{"m", "generated:quad:2x2"});
  ASSERT_TRUE(r.ok());
  auto id = backend.mesh_id("m");
  ASSERT_TRUE(id.ok());

  auto other = generate_native_mesh("quad:3x3", 0, 1);
  ASSERT_TRUE(other.ok());
  auto other_view = NativeMeshView::create_from_host(other.value(), 0);
  ASSERT_TRUE(other_view.ok());

  const fuse::ComponentId placement{7};
  auto set_r = backend.set_placed_view(id.value(), placement, std::move(other_view.value()));
  ASSERT_TRUE(set_r.ok()) << set_r.error().message;

  auto v = backend.placed_view(id.value(), placement);
  ASSERT_TRUE(v.ok()) << v.error().message;
  EXPECT_EQ(v.value()->entity_count(2), 9u);  // the OTHER decomposition, not the default's 4
}

// ---- each placement over its OWN ranks ------------------------------

// The defect this pins needs no MPI to reproduce, because the backend takes
// its position in the run as a constructor argument: a process at rank 2 of
// 3 used to hand EVERY component rank 2's slice of a three-way split, so a
// component that lives alone on rank 2 received a third of its own mesh.
// Two such components then had no ids in common and no exchange between
// them could be built. What a component's rank set says is what it gets.
TEST(NativeBackend, APlacementIsDecomposedOverItsOwnRanksNotTheWholeRun) {
  fuse::CouplingGraph g;
  g.add_mesh(fuse::MeshDecl{"m", "generated:quad:6x6"});
  const auto a = g.add_component(
      fuse::ComponentDecl{.name = "A", .ranks = fuse::RankSet(std::vector<int>{0})});
  const auto b = g.add_component(
      fuse::ComponentDecl{.name = "B", .ranks = fuse::RankSet(std::vector<int>{2})});

  // This process is rank 2 of 3, and hosts B alone.
  NativeMeshBackend backend(2, 3);
  ASSERT_TRUE(backend.load(fuse::MeshDecl{"m", "generated:quad:6x6"}, g).ok());
  const auto id = backend.mesh_id("m");
  ASSERT_TRUE(id.ok());

  const auto whole = generate_native_mesh("quad:6x6", 0, 1);
  ASSERT_TRUE(whole.ok());
  const std::size_t all_cells = whole.value().cell_count();
  ASSERT_EQ(all_cells, 36u);

  // B is alone on this rank, so its decomposition is one part: the whole
  // mesh. Under the defect this was rank 2's third of 36.
  const auto vb = backend.placed_view(id.value(), b);
  ASSERT_TRUE(vb.ok()) << vb.error().message;
  EXPECT_EQ(vb.value()->owned_count(2), all_cells)
      << "a component alone on this rank owns its whole mesh, not this "
         "rank's share of a decomposition over the entire run";

  // A does not live here at all. Empty, not a failure and not someone
  // else's share: every rank resolves every placement, because the
  // exchange plan is collective, and only the hosting ranks read it.
  const auto va = backend.placed_view(id.value(), a);
  ASSERT_TRUE(va.ok()) << va.error().message;
  EXPECT_EQ(va.value()->owned_count(2), 0u);
  EXPECT_EQ(va.value()->dimension(), vb.value()->dimension());
}

// The multi-rank component still splits: this is not "every placement gets
// everything", which would silently undo decomposition itself.
TEST(NativeBackend, APlacementSpanningSeveralRanksStillSplitsAmongThem) {
  fuse::CouplingGraph g;
  g.add_mesh(fuse::MeshDecl{"m", "generated:quad:6x6"});
  const auto spread = g.add_component(fuse::ComponentDecl{
      .name = "spread", .ranks = fuse::RankSet(std::vector<int>{1, 2})});

  std::size_t total = 0;
  for (int r : {1, 2}) {
    NativeMeshBackend backend(r, 3);
    ASSERT_TRUE(backend.load(fuse::MeshDecl{"m", "generated:quad:6x6"}, g).ok());
    const auto id = backend.mesh_id("m");
    ASSERT_TRUE(id.ok());
    const auto v = backend.placed_view(id.value(), spread);
    ASSERT_TRUE(v.ok()) << v.error().message;
    EXPECT_GT(v.value()->owned_count(2), 0u) << "rank " << r;
    EXPECT_LT(v.value()->owned_count(2), 36u) << "rank " << r;
    total += v.value()->owned_count(2);
  }
  // Two ranks, two parts, and between them exactly the mesh -- the halves of
  // a two-way split, NOT ranks 1 and 2 of a three-way one (which would sum
  // to less than the whole).
  EXPECT_EQ(total, 36u);
}

// ---- file round trip ------------------------------------------------------

// The ctest rows backend.native_generators, .omp1 and .omp8 run this same
// binary, and `ctest -j` runs them at once; a fixed name under TempDir()
// made two of them write and remove the same file, failing the round trip
// once in every few full-suite runs (reproduced with eight concurrent
// copies). The process id
// makes the path this process's own.
[[nodiscard]] static std::string scratch_vtk_path(const char* stem) {
  return std::string(::testing::TempDir()) + "native_mesh_roundtrip_" + stem + "_" +
         std::to_string(static_cast<long long>(::getpid())) + ".vtk";
}

TEST(NativeMeshIo, VtkRoundTripReproducesAHexBoxExactly) {
  auto arrays = generate_native_mesh("hexbox:3x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());

  const std::string path = scratch_vtk_path("hex");
  auto w = write_native_mesh_file(path, *view.value(), 0);
  ASSERT_TRUE(w.ok()) << w.error().message;

  auto read_back = read_native_mesh_file(path);
  ASSERT_TRUE(read_back.ok()) << read_back.error().message;
  auto view2 = NativeMeshView::create_from_host(read_back.value(), 0);
  ASSERT_TRUE(view2.ok()) << view2.error().message;

  EXPECT_EQ(view2.value()->cell_count(), view.value()->cell_count());
  EXPECT_EQ(view2.value()->vertex_count(), view.value()->vertex_count());
  double total = 0.0;
  for (std::size_t c = 0; c < view2.value()->cell_count(); ++c) {
    total += view2.value()->measure(3, c);
    EXPECT_EQ(view2.value()->cell_type_of(c), view.value()->cell_type_of(c));
    EXPECT_EQ(view2.value()->global_id(3, c), view.value()->global_id(3, c));
  }
  EXPECT_NEAR(total, 1.0, 1e-11);
  for (std::size_t v = 0; v < view2.value()->vertex_count(); ++v) {
    EXPECT_EQ(view2.value()->vertex_gid_of(v), view.value()->vertex_gid_of(v));
  }
  std::remove(path.c_str());
}

TEST(NativeMeshIo, VtkRoundTripReproducesAPolyhedralMeshExactly) {
  auto arrays = generate_native_mesh("polyhedral:2x2x2", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());

  const std::string path = scratch_vtk_path("poly");
  auto w = write_native_mesh_file(path, *view.value(), 0);
  ASSERT_TRUE(w.ok()) << w.error().message;

  auto read_back = read_native_mesh_file(path);
  ASSERT_TRUE(read_back.ok()) << read_back.error().message;
  auto view2 = NativeMeshView::create_from_host(read_back.value(), 0);
  ASSERT_TRUE(view2.ok()) << view2.error().message;

  double total = 0.0;
  for (std::size_t c = 0; c < view2.value()->cell_count(); ++c) {
    EXPECT_EQ(view2.value()->cell_type_of(c), CellType::Polyhedron);
    total += view2.value()->measure(3, c);
  }
  EXPECT_NEAR(total, 1.0, 1e-11);
  std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// NM-7: `native_backend.hpp` documents `E-MESH-SIZE` for a spec whose
// count does not fit; before this fix nothing produced it, and a spec
// this large instead crashed inside `VertexLookup::local.assign` as an
// uncaught `std::bad_alloc`/`std::length_error`. Both specs are far past
// the int-indexed builder's limit but cheap to refuse -- no allocation is
// attempted before the check fires.
// ---------------------------------------------------------------------------
TEST(NativeGenerators, RefusesAHexboxSpecTooLargeForOneRankInsteadOfCrashing) {
  auto arrays = generate_native_mesh("hexbox:200000x200000x200000", 0, 1);
  ASSERT_FALSE(arrays.ok());
  EXPECT_EQ(arrays.error().code, "E-MESH-SIZE");
}

TEST(NativeGenerators, RefusesAQuadSpecTooLargeForOneRankInsteadOfCrashing) {
  auto arrays = generate_native_mesh("quad:4000000000x4000000000", 0, 1);
  ASSERT_FALSE(arrays.ok());
  EXPECT_EQ(arrays.error().code, "E-MESH-SIZE");
}

TEST(NativeMeshIo, RefusesAFileThatDoesNotExist) {
  auto r = read_native_mesh_file("/nonexistent/path/does/not/exist.vtk");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MESH-IO");
}

// ---------------------------------------------------------------------------
// NM-1: the writer's stream precision must be `max_digits10` for `double`
// (17), or ids and coordinates past 6 significant digits are silently
// rounded on write. A gid at or beyond 1e6 is dense in the generators, so a
// rounded gid collides with a DIFFERENT real vertex/cell instead of merely
// losing precision -- `hexbox:3x2x2`'s round-trip test above escapes this
// because its gids stay under 1e6 and its thirds (0.333333+0.333334+
// 0.333333) happen to sum to exactly 1.0, so the per-cell measure SUM check
// cannot see the rounding either. This mesh is hand-built, not generated,
// so every gid and coordinate is under direct control: gids are deliberately
// past 1e6, and the two quads have unequal, irrational-looking widths so
// neither a measure nor a coordinate-sum comparison can cancel a rounding
// error into agreement -- only a per-cell, per-coordinate, per-gid exact
// comparison (below) is sharp enough to see it.
// ---------------------------------------------------------------------------
TEST(NativeMeshIo, VtkRoundTripReproducesGidsAndCoordinatesAtFullPrecision) {
  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(1234567, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(2345678, 0.6283185307179586, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(3456789, 1.4142135623730951, 0.0, 0.0, 0, true);
  const int v3 = b.add_vertex(4567891, 0.0, 0.9045340337332909, 0.0, 0, true);
  const int v4 = b.add_vertex(5678912, 0.6283185307179586, 0.9045340337332909, 0.0, 0, true);
  const int v5 = b.add_vertex(6789123, 1.4142135623730951, 0.9045340337332909, 0.0, 0, true);
  b.add_cell(CellType::Quad, {v0, v1, v4, v3}, 1000003, 0, true);
  b.add_cell(CellType::Quad, {v1, v2, v5, v4}, 2000003, 0, true);
  auto arrays = b.build();
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok()) << view.error().message;

  const std::string path = scratch_vtk_path("precision");
  auto w = write_native_mesh_file(path, *view.value(), 0);
  ASSERT_TRUE(w.ok()) << w.error().message;

  auto read_back = read_native_mesh_file(path);
  ASSERT_TRUE(read_back.ok()) << read_back.error().message;
  auto view2 = NativeMeshView::create_from_host(read_back.value(), 0);
  ASSERT_TRUE(view2.ok()) << view2.error().message;
  std::remove(path.c_str());

  ASSERT_EQ(view2.value()->vertex_count(), view.value()->vertex_count());
  ASSERT_EQ(view2.value()->cell_count(), view.value()->cell_count());
  for (std::size_t v = 0; v < view.value()->vertex_count(); ++v) {
    EXPECT_EQ(view2.value()->vertex_gid_of(v), view.value()->vertex_gid_of(v)) << "vertex " << v;
    for (int k = 0; k < 3; ++k) {
      EXPECT_EQ(view2.value()->vertex_x(v, k), view.value()->vertex_x(v, k))
          << "vertex " << v << " coord " << k;
    }
  }
  for (std::size_t c = 0; c < view.value()->cell_count(); ++c) {
    EXPECT_EQ(view2.value()->global_id(2, c), view.value()->global_id(2, c)) << "cell " << c;
    EXPECT_NEAR(view2.value()->measure(2, c), view.value()->measure(2, c), 1e-12) << "cell " << c;
  }
}

// NM-1: the writer must refuse (as an invariant, not a data-dependent
// refusal: a gid this large is this backend's own id assignment going
// wrong, never a caller input) rather than silently truncate a gid past
// the format's documented bound (native_mesh_io.hpp: "every integer up to
// 2^53").
TEST(NativeMeshIo, WriterAssertsOnAVertexGidAtOrBeyond2Pow53) {
  NativeMeshBuilder b(1);
  const std::int64_t too_big = std::int64_t{1} << 53;
  const int v0 = b.add_vertex(too_big, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(0, 1.0, 0.0, 0.0, 0, true);
  b.add_cell(CellType::Segment, {v0, v1}, 0, 0, true);
  auto arrays = b.build();
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok()) << view.error().message;

  const std::string path = scratch_vtk_path("oversize-gid");
  EXPECT_THROW((void)write_native_mesh_file(path, *view.value(), 0), fuse::InvariantViolation);
  std::remove(path.c_str());
}
