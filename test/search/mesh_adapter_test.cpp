// test/search/mesh_adapter_test.cpp — element_boxes against
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

}  // namespace
}  // namespace fuse::search
