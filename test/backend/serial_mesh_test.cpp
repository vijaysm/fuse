#include <gtest/gtest.h>
#include <fuse/model/validate.hpp>
#include <fuse/base/error.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/schedule/buffers.hpp>
#include <fuse/schedule/expand.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using fuse::CouplingGraph;
using fuse::MeshDecl;
using fuse::MeshId;
using fuse::MeshView;
using fuse::Region;
using fuse::RegionId;
using fuse::SerialMeshBackend;

namespace {

SerialMeshBackend loaded(const std::string& source) {
  SerialMeshBackend b;
  const auto r = b.load(MeshDecl{.name = "m", .source = source});
  EXPECT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  return b;
}

std::size_t size_of(const SerialMeshBackend& b, const std::string& set,
                    int dim) {
  const auto n = b.region_size(Region{
      .name = "r", .mesh = MeshId{0}, .entity_set = set, .dimension = dim});
  EXPECT_TRUE(n.ok()) << n.error().code << ": " << n.error().message;
  return n.ok() ? n.value() : 0;
}

}  // namespace

// ---------------------------------------------------------------------------
// Loading and counts
// ---------------------------------------------------------------------------

TEST(SerialMesh, LoadsAGeneratedLine) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "l", .source = "generated:line:10"}).ok());
  const auto v = b.view(MeshId{0});
  ASSERT_TRUE(v.ok()) << v.error().message;
  EXPECT_EQ(v.value()->dimension(), 1);
  EXPECT_EQ(v.value()->entity_count(1), 10u);   // 10 cells
  EXPECT_EQ(v.value()->entity_count(0), 11u);   // 11 vertices
}

TEST(SerialMesh, LoadsAGeneratedBox) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:4x4x4"}).ok());
  const auto v = b.view(MeshId{0});
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value()->dimension(), 3);
  EXPECT_EQ(v.value()->entity_count(3), 64u);
  EXPECT_EQ(v.value()->entity_count(0), 125u);  // 5^3 vertices
  // Dimension 2 is the boundary skin, 2*(16+16+16); interior faces are not
  // enumerated.
  EXPECT_EQ(v.value()->entity_count(2), 96u);
}

TEST(SerialMesh, MeshIdsFollowSuccessfulLoadOrder) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "a", .source = "generated:line:3"}).ok());
  // A failed load must not consume an id, or every later MeshId shifts.
  ASSERT_FALSE(b.load(MeshDecl{.name = "bad", .source = "nonsense"}).ok());
  ASSERT_TRUE(b.load(MeshDecl{.name = "c", .source = "generated:box:2x2x2"}).ok());
  ASSERT_TRUE(b.view(MeshId{0}).ok());
  ASSERT_TRUE(b.view(MeshId{1}).ok());
  EXPECT_EQ(b.view(MeshId{0}).value()->entity_count(1), 3u);
  EXPECT_EQ(b.view(MeshId{1}).value()->entity_count(3), 8u);
  EXPECT_FALSE(b.view(MeshId{2}).ok());
  EXPECT_EQ(b.view(MeshId{2}).error().code, "E-MESH-ID");
  EXPECT_FALSE(b.view(MeshId{}).ok());  // default-constructed = invalid
}

TEST(SerialMesh, EntityCountIsATotalQueryAndNeverThrows) {
  const SerialMeshBackend b = loaded("generated:line:4");
  const auto* v = b.view(MeshId{0}).value();
  EXPECT_EQ(v->entity_count(2), 0u);   // a line has no faces
  EXPECT_EQ(v->entity_count(3), 0u);
  EXPECT_EQ(v->entity_count(-1), 0u);  // must not index anything with this
  EXPECT_EQ(v->entity_count(99), 0u);
  EXPECT_EQ(v->owned_count(-1), 0u);
}

TEST(SerialMesh, BackendIsNamedForDiagnostics) {
  EXPECT_EQ(SerialMeshBackend{}.name(), "serial");
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

TEST(SerialMesh, CellCentroidsAreInsideTheUnitBox) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:2x2x2"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  for (std::size_t i = 0; i < v->entity_count(3); ++i) {
    const auto c = v->centroid(3, i);
    EXPECT_GT(c.x, 0.0); EXPECT_LT(c.x, 1.0);
    EXPECT_GT(c.y, 0.0); EXPECT_LT(c.y, 1.0);
    EXPECT_GT(c.z, 0.0); EXPECT_LT(c.z, 1.0);
  }
}

TEST(SerialMesh, VertexCentroidsSpanTheUnitBox) {
  const SerialMeshBackend b = loaded("generated:box:2x2x2");
  const auto* v = b.view(MeshId{0}).value();
  ASSERT_EQ(v->entity_count(0), 27u);
  const auto lo = v->centroid(0, 0);
  EXPECT_DOUBLE_EQ(lo.x, 0.0);
  EXPECT_DOUBLE_EQ(lo.y, 0.0);
  EXPECT_DOUBLE_EQ(lo.z, 0.0);
  const auto hi = v->centroid(0, 26);
  EXPECT_DOUBLE_EQ(hi.x, 1.0);
  EXPECT_DOUBLE_EQ(hi.y, 1.0);
  EXPECT_DOUBLE_EQ(hi.z, 1.0);
  // x-fastest ordering: index 1 steps one cell in x only.
  const auto next = v->centroid(0, 1);
  EXPECT_DOUBLE_EQ(next.x, 0.5);
  EXPECT_DOUBLE_EQ(next.y, 0.0);
  EXPECT_DOUBLE_EQ(next.z, 0.0);
}

TEST(SerialMesh, LineVertexCentroidsAreTheGridPoints) {
  const SerialMeshBackend b = loaded("generated:line:4");
  const auto* v = b.view(MeshId{0}).value();
  EXPECT_DOUBLE_EQ(v->centroid(0, 0).x, 0.0);
  EXPECT_DOUBLE_EQ(v->centroid(0, 4).x, 1.0);
  EXPECT_DOUBLE_EQ(v->centroid(1, 0).x, 0.125);  // first cell centre
}

TEST(SerialMesh, BoundaryFaceCentroidsLieOnTheBoundary) {
  const SerialMeshBackend b = loaded("generated:box:5x4x2");
  const auto* v = b.view(MeshId{0}).value();
  ASSERT_EQ(v->entity_count(2), 76u);
  for (std::size_t i = 0; i < v->entity_count(2); ++i) {
    const auto c = v->centroid(2, i);
    const double xyz[3] = {c.x, c.y, c.z};
    int on_boundary = 0;
    for (const double t : xyz) {
      ASSERT_GE(t, 0.0);
      ASSERT_LE(t, 1.0);
      if (t == 0.0 || t == 1.0) ++on_boundary;
    }
    // Exactly one coordinate is pinned to a face plane; the other two are
    // strictly interior cell centres. Anything else means the face blocks and
    // the in-face indexing disagree.
    EXPECT_EQ(on_boundary, 1) << "face " << i << " at (" << c.x << "," << c.y
                              << "," << c.z << ")";
  }
}

// ---------------------------------------------------------------------------
// measure -- the definition conservation checks depend on
// ---------------------------------------------------------------------------

// Conservation checks integrate against these, so they must sum to the domain.
TEST(SerialMesh, CellMeasuresSumToTheDomainVolume) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:5x4x2"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  double total = 0.0;
  for (std::size_t i = 0; i < v->entity_count(3); ++i) total += v->measure(3, i);
  EXPECT_NEAR(total, 1.0, 1e-14);
}

TEST(SerialMesh, LineMeasuresSumToUnitLength) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "l", .source = "generated:line:7"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  double total = 0.0;
  for (std::size_t i = 0; i < v->entity_count(1); ++i) total += v->measure(1, i);
  EXPECT_NEAR(total, 1.0, 1e-14);
}

// A 5x4x2 unit box has spacings h = (1/5, 1/4, 1/2) = (0.2, 0.25, 0.5), each
// exactly representable in binary floating point, so every measure below is a
// hand-computed exact value rather than a tolerance.
TEST(SerialMesh, CellMeasureIsTheHandComputedCellVolume) {
  const SerialMeshBackend b = loaded("generated:box:5x4x2");
  const auto* v = b.view(MeshId{0}).value();
  ASSERT_EQ(v->entity_count(3), 40u);
  // 0.2 * 0.25 * 0.5 = 0.025, and 40 * 0.025 = 1.
  for (std::size_t i = 0; i < v->entity_count(3); ++i) {
    EXPECT_DOUBLE_EQ(v->measure(3, i), 0.025) << "cell " << i;
  }
}

// The sharp one: on a non-cubic box the three face orientations have three
// DIFFERENT areas, so measure(2, i) must actually depend on i. A backend that
// returns one constant passes every cube test and still gets the boundary
// integral wrong.
TEST(SerialMesh, BoundaryFaceMeasuresAreHandComputedAndNonUniform) {
  const SerialMeshBackend b = loaded("generated:box:5x4x2");
  const auto* v = b.view(MeshId{0}).value();

  // Face blocks, in order: x-min, x-max, y-min, y-max, z-min, z-max.
  //   x-normal: 4*2 = 8 faces of area 0.25 * 0.5  = 0.125
  //   y-normal: 5*2 = 10 faces of area 0.2 * 0.5  = 0.1
  //   z-normal: 5*4 = 20 faces of area 0.2 * 0.25 = 0.05
  struct Block { std::size_t count; double area; };
  const Block blocks[6] = {{8, 0.125}, {8, 0.125}, {10, 0.1},
                           {10, 0.1},  {20, 0.05}, {20, 0.05}};

  std::size_t i = 0;
  double total = 0.0;
  for (const auto& blk : blocks) {
    for (std::size_t k = 0; k < blk.count; ++k, ++i) {
      ASSERT_LT(i, v->entity_count(2));
      EXPECT_DOUBLE_EQ(v->measure(2, i), blk.area) << "face " << i;
      total += v->measure(2, i);
    }
  }
  EXPECT_EQ(i, v->entity_count(2));  // the blocks tile the whole skin
  EXPECT_EQ(v->entity_count(2), 76u);

  // 2*(8*0.125 + 10*0.1 + 20*0.05) = 2*(1 + 1 + 1) = 6, the surface area of
  // the unit cube.
  EXPECT_NEAR(total, 6.0, 1e-14);

  // And three distinct values really do occur, so `i` is load bearing.
  EXPECT_NE(v->measure(2, 0), v->measure(2, 16));
  EXPECT_NE(v->measure(2, 16), v->measure(2, 36));
}

TEST(SerialMesh, MeasureIsUnsignedAndPositive) {
  const SerialMeshBackend b = loaded("generated:box:3x5x7");
  const auto* v = b.view(MeshId{0}).value();
  for (std::size_t i = 0; i < v->entity_count(3); ++i) {
    EXPECT_GT(v->measure(3, i), 0.0);
  }
  for (std::size_t i = 0; i < v->entity_count(2); ++i) {
    EXPECT_GT(v->measure(2, i), 0.0);
  }
}

// A zero measure would silently drop vertices from a conservation sum, so the
// backend refuses rather than inventing one.
TEST(SerialMesh, VertexMeasureIsRefusedRatherThanZero) {
  const SerialMeshBackend b = loaded("generated:box:2x2x2");
  const auto* v = b.view(MeshId{0}).value();
  EXPECT_THROW((void)v->measure(0, 0), fuse::InvalidArgument);
  const SerialMeshBackend l = loaded("generated:line:3");
  EXPECT_THROW((void)l.view(MeshId{0}).value()->measure(0, 0),
               fuse::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Bounds: an unchecked index is a wrong answer, not a crash
// ---------------------------------------------------------------------------

TEST(SerialMesh, OutOfRangeEntityIndexThrows) {
  const SerialMeshBackend b = loaded("generated:box:2x2x2");
  const MeshView* v = b.view(MeshId{0}).value();
  ASSERT_EQ(v->entity_count(3), 8u);
  // Without the check, centroid(3, 999999) returns a point 125000 units away
  // from a unit box, with no memory error for a sanitizer to see.
  EXPECT_THROW((void)v->centroid(3, 8), fuse::InvalidArgument);
  EXPECT_THROW((void)v->centroid(3, 999999), fuse::InvalidArgument);
  EXPECT_THROW((void)v->measure(3, 8), fuse::InvalidArgument);
  EXPECT_THROW((void)v->global_id(3, 8), fuse::InvalidArgument);
  EXPECT_THROW((void)v->is_owned(3, 8), fuse::InvalidArgument);
  EXPECT_THROW((void)v->centroid(2, v->entity_count(2)), fuse::InvalidArgument);
  EXPECT_THROW((void)v->measure(2, v->entity_count(2)), fuse::InvalidArgument);
  EXPECT_THROW((void)v->centroid(0, v->entity_count(0)), fuse::InvalidArgument);
  // The last valid index must still work.
  EXPECT_NO_THROW((void)v->centroid(3, 7));
  EXPECT_NO_THROW((void)v->measure(2, v->entity_count(2) - 1));
}

TEST(SerialMesh, UnrepresentedDimensionThrows) {
  const SerialMeshBackend b = loaded("generated:line:4");
  const MeshView* v = b.view(MeshId{0}).value();
  EXPECT_THROW((void)v->centroid(2, 0), fuse::InvalidArgument);
  EXPECT_THROW((void)v->measure(3, 0), fuse::InvalidArgument);
  EXPECT_THROW((void)v->global_id(-1, 0), fuse::InvalidArgument);
  EXPECT_THROW((void)v->is_owned(7, 0), fuse::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Ownership and ids
// ---------------------------------------------------------------------------

TEST(SerialMesh, GlobalIdsAreDenseAndUnique) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "l", .source = "generated:line:5"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  for (std::size_t i = 0; i < v->entity_count(1); ++i) {
    EXPECT_EQ(v->global_id(1, i), i);
  }
}

// Serial has no ghosts. `owned_count` and `is_owned` still have to agree with
// each other, because the distributed backend's conservation sums will use
// `is_owned` and nothing else.
TEST(SerialMesh, EverythingIsOwnedInSerial) {
  const SerialMeshBackend b = loaded("generated:box:3x2x2");
  const auto* v = b.view(MeshId{0}).value();
  for (const int dim : {0, 2, 3}) {
    EXPECT_EQ(v->owned_count(dim), v->entity_count(dim)) << "dim " << dim;
    std::size_t owned = 0;
    for (std::size_t i = 0; i < v->entity_count(dim); ++i) {
      if (v->is_owned(dim, i)) ++owned;
    }
    EXPECT_EQ(owned, v->owned_count(dim)) << "dim " << dim;
  }
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

TEST(SerialMesh, RegionSizeOfWholeMesh) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:3x3x3"}).ok());
  const auto n = b.region_size(Region{.name = "all", .mesh = MeshId{0},
                                      .entity_set = "", .dimension = 3});
  ASSERT_TRUE(n.ok()) << n.error().message;
  EXPECT_EQ(n.value(), 27u);
}

TEST(SerialMesh, RegionDimensionMinusOneMeansTheMeshDimension) {
  const SerialMeshBackend b = loaded("generated:box:3x3x3");
  EXPECT_EQ(size_of(b, "", -1), 27u);
  EXPECT_EQ(size_of(b, "all", -1), 27u);
  EXPECT_EQ(size_of(b, "", 0), 64u);   // 4^3 vertices
  EXPECT_EQ(size_of(b, "", 2), 54u);   // 2*(9+9+9) boundary faces
  const SerialMeshBackend l = loaded("generated:line:6");
  EXPECT_EQ(size_of(l, "", -1), 6u);
}

TEST(SerialMesh, RegionSizeOfABoundaryFaceSet) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:4x4x4"}).ok());
  const auto n = b.region_size(Region{.name = "w", .mesh = MeshId{0},
                                      .entity_set = "wall", .dimension = 2});
  ASSERT_TRUE(n.ok()) << n.error().message;
  // "wall" is the whole boundary of the box, 2*(16+16+16), not one 4x4 face.
  EXPECT_EQ(n.value(), 96u);
}

// A cube cannot tell the six named faces apart -- they all have 16 entities --
// so the orientation of each named set is only pinned on a non-cubic box.
TEST(SerialMesh, NamedBoundarySetsPickTheRightFaceOnANonCubicBox) {
  const SerialMeshBackend b = loaded("generated:box:5x4x2");  // n = (5, 4, 2)
  EXPECT_EQ(size_of(b, "left", 2), 8u);     // x-normal: 4*2
  EXPECT_EQ(size_of(b, "right", 2), 8u);
  EXPECT_EQ(size_of(b, "front", 2), 10u);   // y-normal: 5*2
  EXPECT_EQ(size_of(b, "back", 2), 10u);
  EXPECT_EQ(size_of(b, "bottom", 2), 20u);  // z-normal: 5*4
  EXPECT_EQ(size_of(b, "top", 2), 20u);
  // The six named faces partition "wall".
  EXPECT_EQ(size_of(b, "wall", 2), 2u * (8u + 10u + 20u));
  EXPECT_EQ(size_of(b, "wall", 2), size_of(b, "", 2));
}

TEST(SerialMesh, UnknownEntitySetIsAnError) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:2x2x2"}).ok());
  const auto n = b.region_size(Region{.name = "x", .mesh = MeshId{0},
                                      .entity_set = "nonexistent",
                                      .dimension = 2});
  ASSERT_FALSE(n.ok());
  EXPECT_EQ(n.error().code, "E-MESH-SET");
}

TEST(SerialMesh, BoundarySetsNeedABox) {
  const SerialMeshBackend l = loaded("generated:line:5");
  const auto n = l.region_size(Region{
      .name = "w", .mesh = MeshId{0}, .entity_set = "wall", .dimension = 2});
  ASSERT_FALSE(n.ok());
  EXPECT_EQ(n.error().code, "E-MESH-SET");
}

// A boundary set holds faces. Answering with a face count for a region whose
// author wrote dimension 3 would size a volumetric buffer from a surface.
TEST(SerialMesh, BoundarySetRejectsANonFaceDimension) {
  const SerialMeshBackend b = loaded("generated:box:4x4x4");
  const auto n = b.region_size(Region{
      .name = "w", .mesh = MeshId{0}, .entity_set = "wall", .dimension = 3});
  ASSERT_FALSE(n.ok());
  EXPECT_EQ(n.error().code, "E-MESH-DIM");
  // -1 defers to the set, which is two-dimensional.
  EXPECT_EQ(size_of(b, "wall", -1), 96u);
}

// Silently returning 0 here would produce an empty buffer for a real region.
TEST(SerialMesh, RegionDimensionTheMeshDoesNotRepresentIsAnError) {
  const SerialMeshBackend l = loaded("generated:line:10");
  for (const int dim : {2, 3, 4}) {
    const auto n = l.region_size(Region{
        .name = "r", .mesh = MeshId{0}, .entity_set = "", .dimension = dim});
    ASSERT_FALSE(n.ok()) << "dimension " << dim;
    EXPECT_EQ(n.error().code, "E-MESH-DIM");
  }
  const auto neg = l.region_size(Region{
      .name = "r", .mesh = MeshId{0}, .entity_set = "", .dimension = -7});
  ASSERT_FALSE(neg.ok());
  EXPECT_EQ(neg.error().code, "E-MESH-DIM");
}

TEST(SerialMesh, RegionOnAnUnknownMeshIsAnError) {
  const SerialMeshBackend b = loaded("generated:box:2x2x2");
  const auto n = b.region_size(Region{
      .name = "r", .mesh = MeshId{4}, .entity_set = "", .dimension = 3});
  ASSERT_FALSE(n.ok());
  EXPECT_EQ(n.error().code, "E-MESH-ID");
  const auto bad = b.region_size(
      Region{.name = "r", .mesh = MeshId{}, .entity_set = "", .dimension = 3});
  ASSERT_FALSE(bad.ok());
  EXPECT_EQ(bad.error().code, "E-MESH-ID");
}

// ---------------------------------------------------------------------------
// Source parsing
// ---------------------------------------------------------------------------

TEST(SerialMesh, MalformedSourceIsAnError) {
  SerialMeshBackend b;
  EXPECT_FALSE(b.load(MeshDecl{.name = "m", .source = "garbage"}).ok());
  EXPECT_FALSE(b.load(MeshDecl{.name = "m", .source = "generated:box:x"}).ok());
}

TEST(SerialMesh, SourceParsingRejectsEverythingItDoesNotMean) {
  SerialMeshBackend b;
  const char* bad[] = {
      "",                          // an empty source is not "generated:"
      "generated",                 // no kind
      "generated:",                // no kind, no dims
      "generated:line",            // no colon after the kind
      "generated:line:",           // no count
      "generated:line:0",          // zero cells is not a mesh
      "generated:line:-3",         // negative
      "generated:line:+3",         // strtol would have taken this
      "generated:line: 3",         // strtol would have taken this too
      "generated:line:3 ",
      "generated:line:3.5",
      "generated:line:0x10",
      "generated:line:99999999999999999999999999",  // beyond size_t
      "generated:blob:3",          // unknown kind
      "generated:box:",
      "generated:box:2",           // one dimension
      "generated:box:2x2",         // two
      "generated:box:2x2x2x2",     // four; the old loop dropped the extra
      "generated:box:2xx2x2",      // empty token
      "generated:box:2x0x2",       // zero extent
      "generated:box:2x2x2x",      // trailing separator
      "/path/to/mesh.h5m",         // a file: not this backend's job
  };
  for (const char* s : bad) {
    const auto r = b.load(MeshDecl{.name = "m", .source = s});
    EXPECT_FALSE(r.ok()) << "should have been rejected: '" << s << "'";
  }
  // Nothing above was allowed to leave a mesh behind.
  EXPECT_FALSE(b.view(MeshId{0}).ok());
}

// std::size_t is unsigned, so an overflowing product wraps to a small,
// plausible count and neither ASan nor UBSan says a word.
TEST(SerialMesh, AbsurdCountsAreRejectedRatherThanWrapping) {
  SerialMeshBackend b;
  const char* huge[] = {
      "generated:box:3000000000x3000000000x3000000000",
      "generated:box:18446744073709551615x18446744073709551615x2",
      "generated:box:18446744073709551615x1x1",  // vertices would overflow
      "generated:line:18446744073709551615",     // vertices = n + 1
  };
  for (const char* s : huge) {
    const auto r = b.load(MeshDecl{.name = "m", .source = s});
    ASSERT_FALSE(r.ok()) << "should have been rejected: '" << s << "'";
    EXPECT_EQ(r.error().code, "E-MESH-SIZE") << s;
  }
  // A large but representable grid is still fine.
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:1000x1000x1000"}).ok());
  EXPECT_EQ(b.view(MeshId{0}).value()->entity_count(3), 1000000000u);
}

// ---------------------------------------------------------------------------
// Moved-from state
// ---------------------------------------------------------------------------

TEST(SerialMesh, MovedFromBackendIsDiagnosedNotDereferenced) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:2x2x2"}).ok());
  SerialMeshBackend other = std::move(b);
  ASSERT_TRUE(other.view(MeshId{0}).ok());  // the mesh moved with it

  const auto l = b.load(MeshDecl{.name = "z", .source = "generated:line:3"});  // NOLINT
  ASSERT_FALSE(l.ok());
  EXPECT_EQ(l.error().code, "E-MESH-MOVED");
  const auto v = b.view(MeshId{0});  // NOLINT
  ASSERT_FALSE(v.ok());
  EXPECT_EQ(v.error().code, "E-MESH-MOVED");
  const auto n = b.region_size(  // NOLINT
      Region{.name = "r", .mesh = MeshId{0}, .entity_set = "", .dimension = 3});
  ASSERT_FALSE(n.ok());
  EXPECT_EQ(n.error().code, "E-MESH-MOVED");

  SerialMeshBackend assigned;
  assigned = std::move(other);
  EXPECT_TRUE(assigned.view(MeshId{0}).ok());
}

// ---------------------------------------------------------------------------
// make_region_sizes: the seam into Phase 3's buffer planner
// ---------------------------------------------------------------------------

namespace {

/// Two components on two differently sized meshes, so a buffer plan that used
/// one region's size for the other would show up.
CouplingGraph two_mesh_graph(const std::string& src_a, const std::string& src_b,
                             const std::string& set_a = "",
                             int dim_a = -1) {
  CouplingGraph g;
  const MeshId ma = g.add_mesh(MeshDecl{.name = "ma", .source = src_a});
  const MeshId mb = g.add_mesh(MeshDecl{.name = "mb", .source = src_b});
  const RegionId ra = g.add_region(
      Region{.name = "ra", .mesh = ma, .entity_set = set_a, .dimension = dim_a});
  const RegionId rb =
      g.add_region(Region{.name = "rb", .mesh = mb, .dimension = -1});
  const auto ca = g.add_component(fuse::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1},
      .interval = fuse::Rational{1}});
  const auto cb = g.add_component(fuse::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1},
      .interval = fuse::Rational{1}});
  const auto fa = g.add_field(ca, fuse::FieldSpec{
      .name = "f", .region = ra, .units = fuse::units::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto fb = g.add_field(cb, fuse::FieldSpec{
      .name = "f_in", .region = rb, .units = fuse::units::kKelvin,
      .intent = fuse::Intent::Requires});
  g.add_channel("c", fa, fb);
  return g;
}

/// Load every mesh a graph declares, in declaration order, so backend MeshIds
/// line up with the graph's.
void load_graph_meshes(SerialMeshBackend& b, const CouplingGraph& g) {
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto r = b.load(g.mesh(MeshId{i}));
    ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  }
}

}  // namespace

TEST(SerialMesh, MakeRegionSizesReportsTheBackendsCounts) {
  const CouplingGraph g =
      two_mesh_graph("generated:box:3x3x3", "generated:line:11");
  SerialMeshBackend b;
  ASSERT_NO_FATAL_FAILURE(load_graph_meshes(b, g));
  const fuse::RegionSizes sizes = make_region_sizes(g, b);
  EXPECT_EQ(sizes(RegionId{0}), 27u);
  EXPECT_EQ(sizes(RegionId{1}), 11u);
}

TEST(SerialMesh, MakeRegionSizesHandlesABoundaryRegion) {
  const CouplingGraph g = two_mesh_graph("generated:box:5x4x2",
                                         "generated:line:3", "left", 2);
  SerialMeshBackend b;
  ASSERT_NO_FATAL_FAILURE(load_graph_meshes(b, g));
  const fuse::RegionSizes sizes = make_region_sizes(g, b);
  EXPECT_EQ(sizes(RegionId{0}), 8u);
}

// Returning 0 for a region the backend cannot size would allocate an empty
// buffer and let the run produce silently empty exchanges. RegionSizes has no
// error channel, so the only honest option is to throw.
TEST(SerialMesh, MakeRegionSizesThrowsRatherThanReturningZero) {
  const CouplingGraph g = two_mesh_graph("generated:box:2x2x2",
                                         "generated:line:3", "nonexistent", 2);
  SerialMeshBackend b;
  ASSERT_NO_FATAL_FAILURE(load_graph_meshes(b, g));
  const fuse::RegionSizes sizes = make_region_sizes(g, b);
  EXPECT_THROW((void)sizes(RegionId{0}), std::runtime_error);
  EXPECT_EQ(sizes(RegionId{1}), 3u);  // the healthy region still answers
}

TEST(SerialMesh, MakeRegionSizesThrowsOnAnOutOfRangeRegionId) {
  const CouplingGraph g =
      two_mesh_graph("generated:box:2x2x2", "generated:line:3");
  SerialMeshBackend b;
  ASSERT_NO_FATAL_FAILURE(load_graph_meshes(b, g));
  const fuse::RegionSizes sizes = make_region_sizes(g, b);
  EXPECT_THROW((void)sizes(RegionId{}), fuse::InvalidArgument);       // invalid
  EXPECT_THROW((void)sizes(RegionId{7}), fuse::InvalidArgument);      // past the end
}

// End to end: a real mesh actually sizes a real buffer plan. This is the whole
// point of the adapter, and it is the one test that proves `RegionSizes` as
// declared in schedule/buffers.hpp is satisfied by what this file produces.
TEST(SerialMesh, BufferPlanIsSizedByTheRealMesh) {
  const CouplingGraph g =
      two_mesh_graph("generated:box:3x3x3", "generated:line:11");
  ASSERT_FALSE(fuse::validate(g).has_errors()) << fuse::validate(g).to_string();

  SerialMeshBackend b;
  ASSERT_NO_FATAL_FAILURE(load_graph_meshes(b, g));

  auto lowered = fuse::lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::OperatorDAG& dag = lowered.dag;
  const fuse::ClockSet cs{g};
  auto timed = fuse::expand(g, dag, cs);
  ASSERT_TRUE(timed.ok()) << timed.error().message;

  const fuse::BufferPlan plan =
      plan_buffers(dag, timed.value(), make_region_sizes(g, b));
  ASSERT_EQ(plan.buffers.size(), dag.arc_count());
  ASSERT_GT(plan.buffers.size(), 0u);
  for (const auto& decl : plan.buffers) {
    ASSERT_TRUE(decl.region.valid());
    const std::size_t expect = decl.region == RegionId{0} ? 27u : 11u;
    EXPECT_EQ(decl.elements, expect) << decl.label;
  }
  // Both meshes are represented, so the two sizes really were distinguished.
  bool saw_27 = false;
  bool saw_11 = false;
  for (const auto& decl : plan.buffers) {
    saw_27 = saw_27 || decl.elements == 27u;
    saw_11 = saw_11 || decl.elements == 11u;
  }
  EXPECT_TRUE(saw_27);
  EXPECT_TRUE(saw_11);
}

// ---------------------------------------------------------------------------
// mesh_id: the name-to-id lookup Task 4.2 flagged as missing
// ---------------------------------------------------------------------------

TEST(SerialMesh, MeshIdResolvesTheDeclaredName) {
  SerialMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "ocean", .source = "generated:box:2x2x2"}).ok());
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "atmos", .source = "generated:line:7"}).ok());
  const auto ocean = b.mesh_id("ocean");
  const auto atmos = b.mesh_id("atmos");
  ASSERT_TRUE(ocean.ok()) << ocean.error().message;
  ASSERT_TRUE(atmos.ok()) << atmos.error().message;
  EXPECT_EQ(ocean.value(), MeshId{0});
  EXPECT_EQ(atmos.value(), MeshId{1});
}

TEST(SerialMesh, MeshIdRejectsUnknownEmptyAndAmbiguousNames) {
  SerialMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "twin", .source = "generated:line:2"}).ok());
  EXPECT_EQ(b.mesh_id("nope").error().code, "E-MESH-NAME");
  // The empty name matches nothing, even though nothing forbids an unnamed
  // mesh: "the mesh with no name" is not something a caller can mean.
  EXPECT_EQ(b.mesh_id("").error().code, "E-MESH-NAME");
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "twin", .source = "generated:line:3"}).ok());
  EXPECT_EQ(b.mesh_id("twin").error().code, "E-MESH-NAME-AMBIGUOUS");
}

TEST(SerialMesh, MeshIdSurvivesAFailedLoad) {
  SerialMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "first", .source = "generated:line:2"}).ok());
  ASSERT_FALSE(b.load(MeshDecl{.name = "bad", .source = "nonsense"}).ok());
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "second", .source = "generated:line:5"}).ok());
  EXPECT_EQ(b.mesh_id("first").value(), MeshId{0});
  EXPECT_EQ(b.mesh_id("second").value(), MeshId{1});
  EXPECT_EQ(b.mesh_id("bad").error().code, "E-MESH-NAME");
}

TEST(SerialMesh, MeshIdOnAMovedFromBackendReports) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:line:2"}).ok());
  SerialMeshBackend other = std::move(b);
  // NOLINTNEXTLINE(bugprone-use-after-move): that is what is under test.
  EXPECT_EQ(b.mesh_id("m").error().code, "E-MESH-MOVED");
  EXPECT_EQ(other.mesh_id("m").value(), MeshId{0});
}

// THE BUG THIS CLOSES. make_region_sizes used to pass the GRAPH's MeshId
// straight to the backend. Load the graph's meshes in any other order -- MOAB
// reading a partitioned set of files, a driver loading on demand -- and every
// region was sized against the wrong mesh: a plausible number, no diagnostic.
// Resolving by name makes the answer independent of load order.
TEST(SerialMesh, MakeRegionSizesIsIndependentOfBackendLoadOrder) {
  const CouplingGraph g =
      two_mesh_graph("generated:box:3x3x3", "generated:line:11");
  ASSERT_EQ(g.mesh_count(), 2u);

  SerialMeshBackend forwards;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    ASSERT_TRUE(forwards.load(g.mesh(MeshId{i})).ok());
  }
  SerialMeshBackend backwards;
  ASSERT_TRUE(backwards.load(g.mesh(MeshId{1})).ok());
  ASSERT_TRUE(backwards.load(g.mesh(MeshId{0})).ok());

  const fuse::RegionSizes a = make_region_sizes(g, forwards);
  const fuse::RegionSizes b = make_region_sizes(g, backwards);
  EXPECT_EQ(a(RegionId{0}), 27u);
  EXPECT_EQ(a(RegionId{1}), 11u);
  EXPECT_EQ(b(RegionId{0}), 27u);
  EXPECT_EQ(b(RegionId{1}), 11u);
}

// A backend that loaded extra meshes, or fewer, still answers correctly for
// the ones it has and refuses loudly for the ones it does not.
TEST(SerialMesh, MakeRegionSizesThrowsWhenTheBackendNeverLoadedTheMesh) {
  const CouplingGraph g =
      two_mesh_graph("generated:box:3x3x3", "generated:line:11");
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "unrelated",
                              .source = "generated:box:9x9x9"}).ok());
  ASSERT_TRUE(b.load(g.mesh(MeshId{1})).ok());  // "mb"
  const fuse::RegionSizes sizes = make_region_sizes(g, b);
  EXPECT_EQ(sizes(RegionId{1}), 11u);
  // Region 0 lives on "ma", which was never loaded. Positionally it would
  // have resolved to the 9x9x9 box and returned 729.
  EXPECT_THROW((void)sizes(RegionId{0}), std::runtime_error);
}
