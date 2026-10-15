// test/search/mesh_adapter_test.cpp -- element_boxes against
// fuse::SerialMeshBackend's generated line (1D) and box (3D) meshes, proving
// the duck-typed MeshLike template really does instantiate against the
// concrete fuse::MeshView with no cast and no include of mesh_view.hpp from
// fuse::search itself (mesh_adapter.hpp's header comment). The 2D case and
// the two refusals are exercised in refusals_test.cpp's FakeMesh instead,
// which needs no fuse::backend link.

#include "fuse/search/mesh_adapter.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "fuse/backend/native_mesh.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/search/box_index.hpp"
#include "fuse_test_main.hpp"

namespace fuse::search {
namespace {

TEST(MeshAdapter, GeneratedLineYieldsOneBoxPerCellAtDimension1) {
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(fuse::MeshDecl{.name = "l", .source = "generated:line:10"}).ok());
  const auto view_r = backend.view(fuse::MeshId{0});
  ASSERT_TRUE(view_r.ok());
  const fuse::MeshView& view = *view_r.value();

  std::vector<std::size_t> owned;
  const auto r = element_boxes(view, owned);
  ASSERT_TRUE(r.ok()) << r.error().message;
  const std::vector<Box>& boxes = r.value();
  ASSERT_EQ(boxes.size(), 10u);
  ASSERT_EQ(owned.size(), 10u);

  // 10 cells tiling [0, 1]: box k's [lo.x, hi.x] should bracket
  // [k/10, (k+1)/10], grown by the default tolerance -- and be exactly flat
  // (lo == hi) in y and z, since a line mesh's cells are.
  for (std::size_t k = 0; k < boxes.size(); ++k) {
    const std::size_t entity = owned[k];
    EXPECT_LE(boxes[k].lo.x, static_cast<double>(entity) / 10.0);
    EXPECT_GE(boxes[k].hi.x, static_cast<double>(entity + 1) / 10.0);
    EXPECT_EQ(boxes[k].lo.y, boxes[k].hi.y);
    EXPECT_EQ(boxes[k].lo.z, boxes[k].hi.z);
  }

  // Feeds a real BoxIndex build directly -- the composed pipeline
  // element_boxes -> BoxIndex::build the next-phase (nodal remap) track will
  // actually call.
  auto idx = BoxIndex::build(boxes);
  ASSERT_TRUE(idx.ok()) << idx.error().message;
  EXPECT_EQ(idx.value().dimension(), 1);
  EXPECT_EQ(idx.value().size(), 10u);
}

TEST(MeshAdapter, GeneratedBoxUsesTheVolumeDimensionNotTheBoundarySkin) {
  // generated:box exposes owned entities at BOTH dim 3 (cells) and dim 2
  // (the boundary skin) simultaneously (mesh_view.hpp's contract,
  // partition's own mesh_adapter_test.cpp pins the same property);
  // element_boxes must read dim from mesh.dimension() (== 3), never scan
  // for "any populated dimension".
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(fuse::MeshDecl{.name = "m", .source = "generated:box:3x2x2"}).ok());
  const auto view_r = backend.view(fuse::MeshId{0});
  ASSERT_TRUE(view_r.ok());
  const fuse::MeshView& view = *view_r.value();
  ASSERT_EQ(view.dimension(), 3);

  std::vector<std::size_t> owned;
  const auto r = element_boxes(view, owned);
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_EQ(r.value().size(), 12u);  // 3*2*2 cells
  EXPECT_EQ(owned.size(), 12u);

  auto idx = BoxIndex::build(r.value());
  ASSERT_TRUE(idx.ok()) << idx.error().message;
  EXPECT_EQ(idx.value().dimension(), 3);

  // A point at the centre of cell (0,0,0) (unit cube, 3x2x2 cells: that
  // cell spans [0, 1/3] x [0, 1/2] x [0, 1/2]) must be found.
  auto qr = idx.value().query_points({Point{1.0 / 6.0, 0.25, 0.25}});
  ASSERT_TRUE(qr.ok());
  ASSERT_EQ(qr.value().offsets.size(), 2u);
  EXPECT_GT(qr.value().offsets[1] - qr.value().offsets[0], 0u);
}

// ---------------------------------------------------------------------------
// element_boxes_device_resident (docs/dev/native-mesh.md, "Execution
// space"): a native mesh's device-resident simplex CSR
// (fuse::native::NativeMeshView::device_simplex_offsets()/
// device_simplex_corners()) builds the SAME boxes element_boxes would, with
// no host round trip -- this file links fuse::backend (test/CMakeLists.txt),
// so it is the one place in fuse::search's own test suite that can name a
// concrete mesh backend to prove it.
// ---------------------------------------------------------------------------

TEST(MeshAdapter, DeviceElementBoxesMatchHostElementBoxesForANativeMesh) {
  using fuse::native::CellType;
  using fuse::native::NativeMeshBuilder;
  using fuse::native::NativeMeshView;

  NativeMeshBuilder b(2);
  const int v0 = b.add_vertex(0, 0.0, 0.0, 0.0, 0, true);
  const int v1 = b.add_vertex(1, 1.0, 0.0, 0.0, 0, true);
  const int v2 = b.add_vertex(2, 1.0, 1.0, 0.0, 0, true);
  const int v3 = b.add_vertex(3, 0.0, 1.0, 0.0, 0, true);
  b.add_cell(CellType::Quad, {v0, v1, v2, v3}, 0, 0, true);
  const int v4 = b.add_vertex(4, 2.0, 0.0, 0.0, 0, true);
  const int v5 = b.add_vertex(5, 3.0, 0.0, 0.0, 0, true);
  const int v6 = b.add_vertex(6, 3.0, 1.0, 0.0, 0, true);
  const int v7 = b.add_vertex(7, 2.5, 1.0, 0.0, 0, true);
  const int v8 = b.add_vertex(8, 2.5, 2.0, 0.0, 0, true);
  const int v9 = b.add_vertex(9, 2.0, 2.0, 0.0, 0, true);
  b.add_cell(CellType::Polygon, {v4, v5, v6, v7, v8, v9}, 1, 0, true);
  auto arrays_r = b.build();
  ASSERT_TRUE(arrays_r.ok());

  // Two independently constructed views over the SAME arrays: one serves
  // the host reference, the other is checked to never touch its own host
  // mirror while the device path runs.
  auto host_view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(host_view_r.ok()) << host_view_r.error().message;
  auto device_view_r = NativeMeshView::create_from_host(arrays_r.value(), 0);
  ASSERT_TRUE(device_view_r.ok()) << device_view_r.error().message;
  const auto& dv = *device_view_r.value();

  std::vector<std::size_t> owned_ids;
  const auto host_boxes_r = element_boxes(*host_view_r.value(), owned_ids);
  ASSERT_TRUE(host_boxes_r.ok()) << host_boxes_r.error().message;

  ASSERT_EQ(dv.host_mirrors_materialized(), 0)
      << "the device view must not have been touched by the host reference computation above";
  const auto device_boxes_r = element_boxes_device_resident(
      dv.device_simplex_offsets(), dv.device_simplex_corners(), dv.dimension() + 1);
  ASSERT_TRUE(device_boxes_r.ok()) << device_boxes_r.error().message;
  EXPECT_EQ(dv.host_mirrors_materialized(), 0)
      << "building element boxes from the device CSR must not materialize a host mirror";

  auto mirror = Kokkos::create_mirror_view(device_boxes_r.value());
  Kokkos::deep_copy(mirror, device_boxes_r.value());
  ASSERT_EQ(mirror.extent(0), host_boxes_r.value().size());
  for (std::size_t k = 0; k < mirror.extent(0); ++k) {
    SCOPED_TRACE(::testing::Message() << "box " << k);
    EXPECT_NEAR(mirror(k).lo.x, host_boxes_r.value()[k].lo.x, 1e-13);
    EXPECT_NEAR(mirror(k).lo.y, host_boxes_r.value()[k].lo.y, 1e-13);
    EXPECT_NEAR(mirror(k).lo.z, host_boxes_r.value()[k].lo.z, 1e-13);
    EXPECT_NEAR(mirror(k).hi.x, host_boxes_r.value()[k].hi.x, 1e-13);
    EXPECT_NEAR(mirror(k).hi.y, host_boxes_r.value()[k].hi.y, 1e-13);
    EXPECT_NEAR(mirror(k).hi.z, host_boxes_r.value()[k].hi.z, 1e-13);
  }

  // Feeds BoxIndex::build_device_resident directly -- index construction
  // for this native mesh never touches the host.
  Kokkos::View<const Box*, Kokkos::DefaultExecutionSpace::memory_space> const_boxes =
      device_boxes_r.value();
  auto idx = BoxIndex::build_device_resident(const_boxes);
  ASSERT_TRUE(idx.ok()) << idx.error().message;
  EXPECT_EQ(idx.value().dimension(), 2);
  EXPECT_EQ(idx.value().size(), 2u);
  EXPECT_EQ(dv.host_mirrors_materialized(), 0)
      << "BoxIndex::build_device_resident must not have touched the device view's host mirror";
}

}  // namespace
}  // namespace fuse::search
