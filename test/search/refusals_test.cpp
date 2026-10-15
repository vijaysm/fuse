// test/search/refusals_test.cpp -- deliberate refusals firing
// (docs/dev/testing.md: "counts as verified only when a test makes it
// fire"). fuse::search returns Result<T>/Failure at its boundary (like
// fuse::backend's MeshBackend), not FUSE_REFUSE/exceptions, so every case
// here asserts `!result.ok()` and the exact code -- no THROWS label.

#include "fuse/search/box_index.hpp"
#include "fuse/search/distributed.hpp"
#include "fuse/search/mesh_adapter.hpp"

#include <gtest/gtest.h>

#include <vector>

#include "fuse/base/config.hpp"
#include "fuse_test_main.hpp"

namespace fuse::search {
namespace {

TEST(SearchRefusals, MixedDimensionBoxesAreRefused) {
  // Box 0 is 2D (nonzero y-extent, zero z-extent); box 1 is 3D.
  std::vector<Box> boxes = {
      Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 0.0}},
      Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}},
  };
  auto idx = BoxIndex::build(boxes);
  ASSERT_FALSE(idx.ok());
  EXPECT_EQ(idx.error().code, "E-SEARCH-MIXED-DIM");
}

TEST(SearchRefusals, MixedDimensionOneVsThreeIsRefused) {
  std::vector<Box> boxes = {
      Box{Point{0.0, 0.0, 0.0}, Point{1.0, 0.0, 0.0}},  // 1D
      Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}},  // 3D
  };
  auto idx = BoxIndex::build(boxes);
  ASSERT_FALSE(idx.ok());
  EXPECT_EQ(idx.error().code, "E-SEARCH-MIXED-DIM");
}

TEST(SearchRefusals, NonPositiveKIsRefused) {
  std::vector<Box> boxes = {Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}}};
  auto idx = BoxIndex::build(boxes);
  ASSERT_TRUE(idx.ok());

  auto zero = idx.value().query_nearest({Point{0.5, 0.5, 0.5}}, 0);
  ASSERT_FALSE(zero.ok());
  EXPECT_EQ(zero.error().code, "E-SEARCH-K");

  auto negative = idx.value().query_nearest({Point{0.5, 0.5, 0.5}}, -3);
  ASSERT_FALSE(negative.ok());
  EXPECT_EQ(negative.error().code, "E-SEARCH-K");
}

TEST(SearchRefusals, DistributedMixedDimensionIsRefusedBeforeAnyCollective) {
  // Checked locally, so this fires the same way with comm == kNullComm --
  // no MPI is needed to exercise it, only the caller's own bad input.
  std::vector<Box> boxes = {
      Box{Point{0.0, 0.0, 0.0}, Point{1.0, 0.0, 0.0}},  // 1D
      Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 0.0}},  // 2D
  };
  auto r = query_points_distributed(kNullComm, boxes, {Point{0.5, 0.0, 0.0}});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SEARCH-MIXED-DIM");
}

#if !FUSE_HAVE_MPI
// This build has no MPI at all (Tier 0) -- a non-null communicator handle
// can only be a caller error, refused before anything tries to resolve it.
// In an MPI (Tier 1) build this branch does not compile in at all; the
// corresponding real-MPI behaviour is exercised on the host by
// distributed_mpi_test.cpp.
TEST(SearchRefusals, NonNullCommWithNoMpiBuildIsRefused) {
  std::vector<Box> boxes = {Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 1.0}}};
  auto r = query_points_distributed(/*comm=*/42, boxes, {Point{0.5, 0.5, 0.5}});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SEARCH-NO-MPI");
}
#endif

// ---------------------------------------------------------------- mesh_adapter

/// A minimal hand-rolled stand-in for `fuse::MeshView`'s SHAPE, used only to
/// exercise `element_boxes`'s own two refusals without needing to link
/// fuse::backend for THIS file (mesh_adapter_test.cpp does that, against the
/// real thing). Deliberately not `fuse::MeshView`-derived: proving the
/// concept is duck-typed, not just satisfied by inheritance.
struct FakeMesh {
  int dim = 2;
  std::size_t simplex_count_value = 1;  // 0 to trigger E-SEARCH-NO-GEOMETRY

  [[nodiscard]] int dimension() const { return dim; }
  [[nodiscard]] std::size_t entity_count(int d) const { return d == dim ? 1u : 0u; }
  [[nodiscard]] bool is_owned(int, std::size_t) const { return true; }
  [[nodiscard]] std::size_t simplex_count(int, std::size_t) const { return simplex_count_value; }
  [[nodiscard]] Point centroid(int, std::size_t) const { return Point{0.5, 0.5, 0.0}; }
  void simplex_points(int, std::size_t, std::size_t, Point* out) const {
    out[0] = Point{0.0, 0.0, 0.0};
    out[1] = Point{1.0, 0.0, 0.0};
    out[2] = Point{0.0, 1.0, 0.0};
  }
};

TEST(SearchRefusals, MeshAdapterRefusesInvalidDimension) {
  FakeMesh mesh;
  mesh.dim = 4;
  std::vector<std::size_t> owned;
  auto r = element_boxes(mesh, owned);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SEARCH-DIM");
}

TEST(SearchRefusals, MeshAdapterRefusesEntityWithNoGeometry) {
  FakeMesh mesh;
  mesh.simplex_count_value = 0;
  std::vector<std::size_t> owned;
  auto r = element_boxes(mesh, owned);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SEARCH-NO-GEOMETRY");
}

TEST(SearchRefusals, MeshAdapterAcceptsAWellFormedFakeMesh) {
  FakeMesh mesh;
  std::vector<std::size_t> owned;
  auto r = element_boxes(mesh, owned);
  ASSERT_TRUE(r.ok()) << r.error().message;
  ASSERT_EQ(r.value().size(), 1u);
  ASSERT_EQ(owned.size(), 1u);
  EXPECT_EQ(owned[0], 0u);
}

}  // namespace
}  // namespace fuse::search
