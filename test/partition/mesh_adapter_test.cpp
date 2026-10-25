// test/partition/mesh_adapter_test.cpp -- centroids_of against
// SerialMeshBackend's generated line and box meshes (a fixture source
// independent of the native mesh). The
// E-PART-MIXED-DIM rejection itself is a cross-rank property (see
// mesh_adapter.hpp's header comment) and is exercised only under a real
// communicator -- see rcb_mpi_test.cpp; comm == kNullComm agrees with
// itself trivially and cannot reach that branch, so it is not repeated
// here.

#include "fuse/partition/mesh_adapter.hpp"

#include <cstddef>

#include <gtest/gtest.h>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse_test_main.hpp"

namespace fuse::partition {
namespace {

// A minimal MeshView stub whose dimension() answers <= 0 -- the one
// documented, Tier-0-reachable refusal (mesh_adapter.hpp's own doc
// comment, mesh_adapter.cpp).
// Every other pure virtual is unreachable once dimension() <= 0 refuses
// before a single entity is read, so they return harmless placeholders.
class DimensionStubMesh : public fuse::MeshView {
 public:
  explicit DimensionStubMesh(int dim) : dim_(dim) {}
  [[nodiscard]] int dimension() const override { return dim_; }
  [[nodiscard]] std::size_t entity_count(int) const override { return 0; }
  [[nodiscard]] std::size_t owned_count(int) const override { return 0; }
  [[nodiscard]] bool is_owned(int, std::size_t) const override { return false; }
  [[nodiscard]] fuse::Point centroid(int, std::size_t) const override { return fuse::Point{}; }
  [[nodiscard]] double measure(int, std::size_t) const override { return 0.0; }
  [[nodiscard]] std::size_t global_id(int, std::size_t) const override { return 0; }
  [[nodiscard]] std::size_t simplex_count(int, std::size_t) const override { return 0; }
  void simplex_points(int, std::size_t, std::size_t, fuse::Point*) const override {}

 private:
  int dim_;
};

TEST(MeshAdapter, RefusesNoEntitiesWhenDimensionIsZero) {
  DimensionStubMesh mesh(0);
  auto r = centroids_of(mesh);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-PART-NO-ENTITIES");
}

TEST(MeshAdapter, RefusesNoEntitiesWhenDimensionIsNegative) {
  DimensionStubMesh mesh(-1);
  auto r = centroids_of(mesh);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-PART-NO-ENTITIES");
}

TEST(MeshAdapter, GeneratedLineYieldsOneCentroidPerCellAtDimension1) {
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(fuse::MeshDecl{.name = "l", .source = "generated:line:10"}).ok());
  const auto view_r = backend.view(fuse::MeshId{0});
  ASSERT_TRUE(view_r.ok());
  const fuse::MeshView& view = *view_r.value();

  const auto r = centroids_of(view);
  ASSERT_TRUE(r.ok()) << r.error().message;
  const CentroidData& cd = r.value();
  EXPECT_EQ(cd.dimension_used, 1);
  EXPECT_EQ(cd.centroids.extent(0), 10u);
  EXPECT_EQ(cd.global_ids.extent(0), 10u);

  // Cell centroids of a 10-cell unit line are at (k+0.5)/10, y=z=0, in some
  // order -- SerialMeshBackend's serial semantics make local index == global
  // id (its own doc comment), so this is directly checkable.
  std::vector<bool> seen(10, false);
  for (std::size_t k = 0; k < cd.centroids.extent(0); ++k) {
    const auto gid = cd.global_ids(k);
    ASSERT_GE(gid, 0);
    ASSERT_LT(gid, 10);
    EXPECT_FALSE(seen[static_cast<std::size_t>(gid)]) << "duplicate gid " << gid;
    seen[static_cast<std::size_t>(gid)] = true;
    EXPECT_NEAR(cd.centroids(k, 0), (static_cast<double>(gid) + 0.5) / 10.0, 1e-12);
    EXPECT_NEAR(cd.centroids(k, 1), 0.0, 1e-12);
    EXPECT_NEAR(cd.centroids(k, 2), 0.0, 1e-12);
  }
  for (bool s : seen) EXPECT_TRUE(s);
}

TEST(MeshAdapter, GeneratedBoxUsesTheVolumeDimensionNotTheBoundarySkin) {
  // A box mesh exposes owned entities at BOTH dim 3 (cells) and dim 2 (the
  // boundary skin) simultaneously -- serial_backend.hpp's own doc. This is
  // exactly the case centroids_of must NOT misread as "mixed dimension":
  // MeshView::dimension() == 3 here, and the skin is a legitimate lower-
  // dimension entity set coexisting with it (mesh_view.hpp's contract).
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(fuse::MeshDecl{.name = "m", .source = "generated:box:3x2x2"}).ok());
  const auto view_r = backend.view(fuse::MeshId{0});
  ASSERT_TRUE(view_r.ok());
  const fuse::MeshView& view = *view_r.value();

  ASSERT_EQ(view.dimension(), 3);
  ASSERT_GT(view.owned_count(2), 0u);  // the skin really is present and owned
  ASSERT_GT(view.owned_count(3), 0u);

  const auto r = centroids_of(view);
  ASSERT_TRUE(r.ok()) << r.error().message;
  const CentroidData& cd = r.value();
  EXPECT_EQ(cd.dimension_used, 3);
  EXPECT_EQ(cd.centroids.extent(0), 3u * 2u * 2u);  // 12 cells, not the skin's face count
}

TEST(MeshAdapter, EveryCentroidLiesInsideTheUnitCube) {
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(fuse::MeshDecl{.name = "m", .source = "generated:box:4x3x2"}).ok());
  const auto view_r = backend.view(fuse::MeshId{0});
  ASSERT_TRUE(view_r.ok());
  const auto r = centroids_of(*view_r.value());
  ASSERT_TRUE(r.ok()) << r.error().message;
  const CentroidData& cd = r.value();
  for (std::size_t k = 0; k < cd.centroids.extent(0); ++k) {
    for (int a = 0; a < 3; ++a) {
      EXPECT_GE(cd.centroids(k, a), 0.0);
      EXPECT_LE(cd.centroids(k, a), 1.0);
    }
  }
}

}  // namespace
}  // namespace fuse::partition
