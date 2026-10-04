// The native mesh's generated sources and its file round trip. Global
// conservation across a simulated multi-rank build (each rank's own slice,
// summed) is the load-bearing property: it is what migration's tests will
// also check, so a bug here would otherwise surface confusingly there.

#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/native_mesh_io.hpp"

#include <cstdio>
#include <set>

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

// ---- file round trip ------------------------------------------------------

// The ctest rows backend.native_generators, .omp1 and .omp8 run this same
// binary, and `ctest -j` runs them at once; a fixed name under TempDir()
// made two of them write and remove the same file, failing the round trip
// once in every few full-suite runs (first seen 2026-09-30, reproduced
// 2026-10-04 with eight concurrent copies on a clean HEAD). The process id
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

TEST(NativeMeshIo, RefusesAFileThatDoesNotExist) {
  auto r = read_native_mesh_file("/nonexistent/path/does/not/exist.vtk");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MESH-IO");
}
