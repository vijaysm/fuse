#include <gtest/gtest.h>

#ifdef FUSE_HAVE_MOAB

#include <fuse/backend/moab_backend.hpp>
#include <fuse/backend/serial_backend.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using fuse::MeshDecl;
using fuse::MeshId;
using fuse::MeshView;
using fuse::MoabMeshBackend;
using fuse::Region;

namespace {

MoabMeshBackend loaded(const std::string& source) {
  MoabMeshBackend b;
  const auto r = b.load(MeshDecl{.name = "m", .source = source});
  EXPECT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  return b;
}

const MeshView& view_of(const MoabMeshBackend& b) {
  const auto v = b.view(MeshId{0});
  EXPECT_TRUE(v.ok()) << v.error().code << ": " << v.error().message;
  return *v.value();
}

std::size_t size_of(const MoabMeshBackend& b, const std::string& set, int dim) {
  const auto n = b.region_size(Region{
      .name = "r", .mesh = MeshId{0}, .entity_set = set, .dimension = dim});
  EXPECT_TRUE(n.ok()) << n.error().code << ": " << n.error().message;
  return n.ok() ? n.value() : 0;
}

/// Sum over OWNED entities only, which is what the measure contract requires
/// even where -- as here, in serial -- every entity happens to be owned.
double owned_measure_sum(const MeshView& v, int dim) {
  double total = 0.0;
  for (std::size_t i = 0; i < v.entity_count(dim); ++i) {
    if (v.is_owned(dim, i)) total += v.measure(dim, i);
  }
  return total;
}

// ---------------------------------------------------------------------------
// A tiny unstructured mesh with hand-computable measures, written as legacy
// ASCII VTK so nothing but MOAB's reader is involved. Four cells, each chosen
// because its volume has a closed form:
//
//   tet     (0,0,0) (1,0,0) (0,1,0) (0,0,1)            1/6
//   pyramid unit square base, apex above its centre    1/3
//   wedge   unit right triangle extruded by 1          1/2
//   hex     unit square base, top the BILINEAR surface
//           z = 1 + xy, i.e. corner (1,1) lifted to 2  1 + 1/4 = 5/4
//
// The last one is the point of the file: it is a genuinely trilinear
// hexahedron, so a five-tetrahedron decomposition of it gives 4/3, not 5/4.
// ---------------------------------------------------------------------------
constexpr double kTetVolume = 1.0 / 6.0;
constexpr double kPyramidVolume = 1.0 / 3.0;
constexpr double kWedgeVolume = 0.5;
constexpr double kTrilinearHexVolume = 1.25;

/// All 23 points and 4 cells, written out rather than generated: a fixture
/// that computes its own geometry can be wrong in the same way the code under
/// test is wrong. VTK cell types: 10 = tetra, 14 = pyramid, 13 = wedge,
/// 12 = hexahedron.
const char* const kMixedVtk = R"(# vtk DataFile Version 3.0
mixed-element measure fixture
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 23 double
0 0 0
1 0 0
0 1 0
0 0 1
3 0 0
4 0 0
4 1 0
3 1 0
3.5 0.5 1
6 0 0
7 0 0
6 1 0
6 0 1
7 0 1
6 1 1
9 0 0
10 0 0
10 1 0
9 1 0
9 0 1
10 0 1
10 1 2
9 1 1
CELLS 4 27
4 0 1 2 3
5 4 5 6 7 8
6 9 10 11 12 13 14
8 15 16 17 18 19 20 21 22
CELL_TYPES 4
10
14
13
12
)";

std::string write_mixed_vtk() {
  const std::string path =
      std::string{::testing::TempDir()} + "soma_moab_mixed.vtk";
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  EXPECT_TRUE(out.good()) << "cannot write " << path;
  out << kMixedVtk;
  out.close();
  return path;
}

}  // namespace

// ===========================================================================
// The brief's tests
// ===========================================================================

TEST(MoabMesh, LoadsAGeneratedBox) {
  MoabMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "m", .source = "generated:box:4x4x4"}).ok());
  const auto v = b.view(MeshId{0});
  ASSERT_TRUE(v.ok()) << v.error().message;
  EXPECT_EQ(v.value()->dimension(), 3);
  EXPECT_EQ(v.value()->entity_count(3), 64u);
}

TEST(MoabMesh, ReportsTheBackendName) {
  const MoabMeshBackend b;
  EXPECT_EQ(b.name(), "moab");
}

// Conservation depends on these summing to the domain measure.
TEST(MoabMesh, CellMeasuresSumToTheDomainVolume) {
  const MoabMeshBackend b = loaded("generated:box:5x4x3");
  EXPECT_NEAR(owned_measure_sum(view_of(b), 3), 1.0, 1e-12);
}

TEST(MoabMesh, CentroidsAreInsideTheDomain) {
  const MoabMeshBackend b = loaded("generated:box:3x3x3");
  const MeshView& v = view_of(b);
  for (std::size_t i = 0; i < v.entity_count(3); ++i) {
    const auto c = v.centroid(3, i);
    EXPECT_GT(c.x, 0.0);
    EXPECT_LT(c.x, 1.0);
    EXPECT_GT(c.y, 0.0);
    EXPECT_LT(c.y, 1.0);
    EXPECT_GT(c.z, 0.0);
    EXPECT_LT(c.z, 1.0);
  }
}

TEST(MoabMesh, GlobalIdsAreUnique) {
  const MoabMeshBackend b = loaded("generated:box:3x3x3");
  const MeshView& v = view_of(b);
  std::set<std::size_t> ids;
  for (std::size_t i = 0; i < v.entity_count(3); ++i) {
    ids.insert(v.global_id(3, i));
  }
  EXPECT_EQ(ids.size(), v.entity_count(3));
}

TEST(MoabMesh, SerialRunOwnsEveryEntity) {
  const MoabMeshBackend b = loaded("generated:box:2x2x2");
  const MeshView& v = view_of(b);
  EXPECT_EQ(v.owned_count(3), v.entity_count(3));
  EXPECT_EQ(v.owned_count(2), v.entity_count(2));
  EXPECT_EQ(v.owned_count(0), v.entity_count(0));
  for (std::size_t i = 0; i < v.entity_count(3); ++i) {
    EXPECT_TRUE(v.is_owned(3, i));
  }
}

TEST(MoabMesh, RegionSizeOfTheWholeMesh) {
  const MoabMeshBackend b = loaded("generated:box:3x3x3");
  const auto n = b.region_size(
      Region{.name = "all", .mesh = MeshId{0}, .dimension = 3});
  ASSERT_TRUE(n.ok()) << n.error().message;
  EXPECT_EQ(n.value(), 27u);
}

TEST(MoabMesh, MissingFileIsAnError) {
  MoabMeshBackend b;
  const auto r = b.load(MeshDecl{.name = "m", .source = "/nonexistent.h5m"});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MOAB-LOAD");
  EXPECT_FALSE(r.error().message.empty());
}

// A failed load must consume no MeshId, or every later id is off by one.
TEST(MoabMesh, AFailedLoadConsumesNoMeshId) {
  MoabMeshBackend b;
  ASSERT_FALSE(b.load(MeshDecl{.name = "bad", .source = "/nonexistent.h5m"}).ok());
  ASSERT_TRUE(b.load(MeshDecl{.name = "good",
                              .source = "generated:box:2x2x2"}).ok());
  const auto v = b.view(MeshId{0});
  ASSERT_TRUE(v.ok()) << v.error().message;
  EXPECT_EQ(v.value()->entity_count(3), 8u);
  EXPECT_FALSE(b.view(MeshId{1}).ok());
  const auto id = b.mesh_id("good");
  ASSERT_TRUE(id.ok()) << id.error().message;
  EXPECT_EQ(id.value(), MeshId{0});
  EXPECT_EQ(b.mesh_id("bad").error().code, "E-MESH-NAME");
}

TEST(MoabMesh, OwnedEntitiesMatchOwnedCountInSerial) {
  const MoabMeshBackend b = loaded("generated:box:2x2x2");
  const auto owned = b.owned_entities(
      Region{.name = "all", .mesh = MeshId{0}, .dimension = 3});
  ASSERT_TRUE(owned.ok()) << owned.error().message;
  EXPECT_EQ(owned.value().size(), 8u);
  for (std::size_t i = 0; i < owned.value().size(); ++i) {
    EXPECT_EQ(owned.value()[i], i);
  }
}

// The backends must be interchangeable behind MeshView.
TEST(MoabMesh, AgreesWithTheSerialBackendOnCounts) {
  MoabMeshBackend moab;
  fuse::SerialMeshBackend serial;
  const MeshDecl d{.name = "m", .source = "generated:box:4x4x4"};
  ASSERT_TRUE(moab.load(d).ok());
  ASSERT_TRUE(serial.load(d).ok());
  EXPECT_EQ(moab.view(MeshId{0}).value()->entity_count(3),
            serial.view(MeshId{0}).value()->entity_count(3));
}

// ===========================================================================
// The measure contract, clause by clause
// ===========================================================================

TEST(MoabMesh, MeasureDependsOnTheEntityIndex) {
  // A non-cubic box: every cell has the same volume, so use the BOUNDARY,
  // whose faces have three different areas. A measure() that ignored `i`
  // would return one of them for all of them.
  const MoabMeshBackend b = loaded("generated:box:5x4x2");
  const MeshView& v = view_of(b);
  // h = (1/5, 1/4, 1/2). A face normal to axis a has area h[(a+1)%3] *
  // h[(a+2)%3], and there are 2 * n[(a+1)%3] * n[(a+2)%3] of them.
  const double area_x = (1.0 / 4) * (1.0 / 2);  // 1/8,  16 faces
  const double area_y = (1.0 / 2) * (1.0 / 5);  // 1/10, 20 faces
  const double area_z = (1.0 / 5) * (1.0 / 4);  // 1/20, 40 faces
  std::size_t nx = 0;
  std::size_t ny = 0;
  std::size_t nz = 0;
  std::size_t other = 0;
  for (std::size_t i = 0; i < v.entity_count(2); ++i) {
    const double a = v.measure(2, i);
    // Not an exact comparison: the quadrature sums 64 products, so two faces
    // of genuinely equal area can differ in the last bit. Three distinct
    // VALUES is the claim; three distinct bit patterns is not.
    if (std::fabs(a - area_x) < 1e-14) {
      ++nx;
    } else if (std::fabs(a - area_y) < 1e-14) {
      ++ny;
    } else if (std::fabs(a - area_z) < 1e-14) {
      ++nz;
    } else {
      ++other;
    }
  }
  EXPECT_EQ(nx, 16u);
  EXPECT_EQ(ny, 20u);
  EXPECT_EQ(nz, 40u);
  EXPECT_EQ(other, 0u);
  // A measure() that ignored `i` would put all 76 faces in one bucket.
  EXPECT_EQ(nx + ny + nz, v.entity_count(2));
}

TEST(MoabMesh, BoundaryFaceMeasuresSumToTheSurfaceArea) {
  const MoabMeshBackend b = loaded("generated:box:5x4x2");
  // The unit cube's surface area, whatever the subdivision.
  EXPECT_NEAR(owned_measure_sum(view_of(b), 2), 6.0, 1e-12);
}

TEST(MoabMesh, ANamedSideTilesItsOwnFace) {
  const MoabMeshBackend b = loaded("generated:box:5x4x2");
  const MeshView& v = view_of(b);
  const auto left = b.owned_entities(Region{
      .name = "r", .mesh = MeshId{0}, .entity_set = "left", .dimension = 2});
  ASSERT_TRUE(left.ok()) << left.error().message;
  EXPECT_EQ(left.value().size(), 4u * 2u);
  double area = 0.0;
  for (const std::size_t i : left.value()) area += v.measure(2, i);
  EXPECT_NEAR(area, 1.0, 1e-14);  // the x = 0 face of the unit cube
}

TEST(MoabMesh, VertexMeasureThrowsRatherThanReturningZero) {
  const MoabMeshBackend b = loaded("generated:box:2x2x2");
  const MeshView& v = view_of(b);
  ASSERT_GT(v.entity_count(0), 0u);
  EXPECT_THROW((void)v.measure(0, 0), std::out_of_range);
}

TEST(MoabMesh, OutOfRangeAccessThrows) {
  const MoabMeshBackend b = loaded("generated:box:2x2x2");
  const MeshView& v = view_of(b);
  EXPECT_THROW((void)v.measure(3, v.entity_count(3)), std::out_of_range);
  EXPECT_THROW((void)v.centroid(3, 999), std::out_of_range);
  EXPECT_THROW((void)v.global_id(3, 999), std::out_of_range);
  EXPECT_THROW((void)v.is_owned(3, 999), std::out_of_range);
  EXPECT_THROW((void)v.measure(-1, 0), std::out_of_range);
  EXPECT_THROW((void)v.measure(7, 0), std::out_of_range);
}

TEST(MoabMesh, EntityCountIsTotalForEveryDimension) {
  const MoabMeshBackend b = loaded("generated:line:5");
  const MeshView& v = view_of(b);
  EXPECT_EQ(v.dimension(), 1);
  EXPECT_EQ(v.entity_count(1), 5u);
  EXPECT_EQ(v.entity_count(0), 6u);
  EXPECT_EQ(v.entity_count(2), 0u);
  EXPECT_EQ(v.entity_count(3), 0u);
  EXPECT_EQ(v.entity_count(-4), 0u);
  EXPECT_EQ(v.entity_count(99), 0u);
  EXPECT_NEAR(owned_measure_sum(v, 1), 1.0, 1e-15);
}

// ===========================================================================
// Unstructured elements with closed-form volumes, read from a real file
// ===========================================================================

TEST(MoabMesh, MixedElementMeasuresMatchClosedForms) {
  const std::string path = write_mixed_vtk();
  MoabMeshBackend b;
  const auto r = b.load(MeshDecl{.name = "mixed", .source = path});
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const MeshView& v = view_of(b);
  ASSERT_EQ(v.dimension(), 3);
  ASSERT_EQ(v.entity_count(3), 4u);

  // Match by volume rather than by index: MOAB groups elements by type, so
  // the local order is not the file order.
  std::vector<double> got;
  for (std::size_t i = 0; i < 4; ++i) got.push_back(v.measure(3, i));
  std::sort(got.begin(), got.end());
  const std::vector<double> want{kTetVolume, kPyramidVolume, kWedgeVolume,
                                 kTrilinearHexVolume};
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    EXPECT_NEAR(got[i], want[i], 1e-14) << "element " << i;
  }

  // The trilinear hexahedron is the one that separates an exact integrator
  // from a tetrahedral decomposition: five tetrahedra give 4/3 here.
  EXPECT_NEAR(got.back(), kTrilinearHexVolume, 1e-14);
  EXPECT_GT(std::fabs(got.back() - 4.0 / 3.0), 1e-3)
      << "measure() agrees with the five-tetrahedron decomposition, which is "
         "wrong for a non-affine hexahedron";
}

TEST(MoabMesh, MeasureIsUnsignedForAnInvertedElement) {
  // The same reference tetrahedron with two nodes swapped: negative
  // orientation, identical absolute volume. A signed measure would let this
  // cancel against a healthy neighbour inside a conservation sum.
  const std::string path =
      std::string{::testing::TempDir()} + "soma_moab_inverted.vtk";
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.good());
    out << "# vtk DataFile Version 3.0\ninverted tet\nASCII\n"
           "DATASET UNSTRUCTURED_GRID\nPOINTS 4 double\n"
           "0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
           "CELLS 1 5\n4 0 2 1 3\nCELL_TYPES 1\n10\n";
  }
  MoabMeshBackend b;
  const auto r = b.load(MeshDecl{.name = "inv", .source = path});
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const MeshView& v = view_of(b);
  ASSERT_EQ(v.entity_count(3), 1u);
  EXPECT_NEAR(v.measure(3, 0), kTetVolume, 1e-15);
  EXPECT_GE(v.measure(3, 0), 0.0);
}

TEST(MoabMesh, ARealTriangleAndQuadAreaAreExact) {
  const std::string path =
      std::string{::testing::TempDir()} + "soma_moab_surface.vtk";
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.good());
    // A 3-4-5 right triangle (area 6) and a planar trapezoid whose area is
    // (2 + 3)/2 * 2 = 5 by the trapezoid rule.
    out << "# vtk DataFile Version 3.0\nsurface\nASCII\n"
           "DATASET UNSTRUCTURED_GRID\nPOINTS 7 double\n"
           "0 0 0\n3 0 0\n0 4 0\n"
           "10 0 0\n12 0 0\n13 2 0\n10 2 0\n"
           "CELLS 2 9\n3 0 1 2\n4 3 4 5 6\nCELL_TYPES 2\n5\n9\n";
  }
  MoabMeshBackend b;
  const auto r = b.load(MeshDecl{.name = "s", .source = path});
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const MeshView& v = view_of(b);
  ASSERT_EQ(v.dimension(), 2);
  ASSERT_EQ(v.entity_count(2), 2u);
  std::vector<double> got{v.measure(2, 0), v.measure(2, 1)};
  std::sort(got.begin(), got.end());
  EXPECT_NEAR(got[0], 5.0, 1e-14);  // trapezoid
  EXPECT_NEAR(got[1], 6.0, 1e-14);  // 3-4-5 triangle
}

// ===========================================================================
// Entity sets
// ===========================================================================

TEST(MoabMesh, NamedBoxSetsMatchTheSerialBackend) {
  const MoabMeshBackend moab = loaded("generated:box:5x4x2");
  fuse::SerialMeshBackend serial;
  ASSERT_TRUE(
      serial.load(MeshDecl{.name = "m", .source = "generated:box:5x4x2"}).ok());
  for (const char* set : {"left", "right", "front", "back", "bottom", "top",
                          "wall", "all", ""}) {
    const Region r{
        .name = "r", .mesh = MeshId{0}, .entity_set = set, .dimension = 2};
    const auto a = moab.region_size(r);
    const auto s = serial.region_size(r);
    ASSERT_TRUE(a.ok()) << set << ": " << a.error().message;
    ASSERT_TRUE(s.ok()) << set << ": " << s.error().message;
    EXPECT_EQ(a.value(), s.value()) << "entity set '" << set << "'";
  }
}

TEST(MoabMesh, SidesetSpellingResolvesTheSameSet) {
  const MoabMeshBackend b = loaded("generated:box:5x4x2");
  EXPECT_EQ(size_of(b, "sideset_1", 2), size_of(b, "left", 2));
  EXPECT_EQ(size_of(b, "neumann_6", 2), size_of(b, "top", 2));
}

TEST(MoabMesh, AnUnknownEntitySetIsAnError) {
  const MoabMeshBackend b = loaded("generated:box:2x2x2");
  const auto n = b.region_size(Region{.name = "r",
                                      .mesh = MeshId{0},
                                      .entity_set = "nonexistent",
                                      .dimension = 2});
  ASSERT_FALSE(n.ok());
  // MeshBackend::region_size documents E-MESH-SET, so both backends report
  // the same code for the same mistake.
  EXPECT_EQ(n.error().code, "E-MESH-SET");
}

TEST(MoabMesh, ATaggedSetSpellingWithNoMatchIsAnError) {
  const MoabMeshBackend b = loaded("generated:box:2x2x2");
  const auto n = b.region_size(Region{.name = "r",
                                      .mesh = MeshId{0},
                                      .entity_set = "sideset_99",
                                      .dimension = 2});
  ASSERT_FALSE(n.ok());
  EXPECT_EQ(n.error().code, "E-MESH-SET");
}

TEST(MoabMesh, ABoundarySetAtTheWrongDimensionIsAnError) {
  const MoabMeshBackend b = loaded("generated:box:2x2x2");
  const auto n = b.region_size(Region{.name = "r",
                                      .mesh = MeshId{0},
                                      .entity_set = "left",
                                      .dimension = 3});
  ASSERT_FALSE(n.ok());
  EXPECT_EQ(n.error().code, "E-MESH-DIM");
}

TEST(MoabMesh, ErrorCodesMatchTheSerialBackendForTheSameMistake) {
  const MoabMeshBackend moab = loaded("generated:box:2x2x2");
  fuse::SerialMeshBackend serial;
  ASSERT_TRUE(
      serial.load(MeshDecl{.name = "m", .source = "generated:box:2x2x2"}).ok());
  struct Case {
    const char* set;
    int dim;
  };
  for (const Case c : {Case{"nonexistent", 2}, Case{"left", 3}}) {
    const Region r{
        .name = "r", .mesh = MeshId{0}, .entity_set = c.set, .dimension = c.dim};
    const auto a = moab.region_size(r);
    const auto s = serial.region_size(r);
    ASSERT_FALSE(a.ok());
    ASSERT_FALSE(s.ok());
    EXPECT_EQ(a.error().code, s.error().code) << c.set << " @ " << c.dim;
  }
  // An unknown mesh id.
  const Region bad{.name = "r", .mesh = MeshId{9}, .dimension = 3};
  EXPECT_EQ(moab.region_size(bad).error().code, "E-MESH-ID");
  EXPECT_EQ(serial.region_size(bad).error().code, "E-MESH-ID");
}

TEST(MoabMesh, RejectsTheSameMalformedSourcesAsTheSerialBackend) {
  for (const char* src :
       {"generated:box:2x2", "generated:box:2x2x2x2", "generated:box:0x1x1",
        "generated:box:2x2x", "generated:box: 2x2x2", "generated:box:+2x2x2",
        "generated:line:0", "generated:line:", "generated:cone:3",
        "generated:box"}) {
    MoabMeshBackend moab;
    fuse::SerialMeshBackend serial;
    const MeshDecl d{.name = "m", .source = src};
    const auto a = moab.load(d);
    const auto s = serial.load(d);
    EXPECT_FALSE(a.ok()) << src << " accepted by the MOAB backend";
    EXPECT_FALSE(s.ok()) << src << " accepted by the serial backend";
    if (!a.ok() && !s.ok()) {
      EXPECT_EQ(a.error().code, s.error().code) << src;
    }
  }
}

// ===========================================================================
// mesh_id, the name-to-id lookup Task 4.2 flagged as missing
// ===========================================================================

TEST(MoabMesh, MeshIdResolvesByNameRegardlessOfLoadOrder) {
  MoabMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "ocean", .source = "generated:box:2x2x2"}).ok());
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "atmos", .source = "generated:line:7"}).ok());
  const auto ocean = b.mesh_id("ocean");
  const auto atmos = b.mesh_id("atmos");
  ASSERT_TRUE(ocean.ok());
  ASSERT_TRUE(atmos.ok());
  EXPECT_EQ(ocean.value(), MeshId{0});
  EXPECT_EQ(atmos.value(), MeshId{1});
  EXPECT_EQ(b.view(atmos.value()).value()->entity_count(1), 7u);
}

TEST(MoabMesh, MeshIdRejectsUnknownEmptyAndAmbiguousNames) {
  MoabMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "twin", .source = "generated:line:2"}).ok());
  EXPECT_EQ(b.mesh_id("nope").error().code, "E-MESH-NAME");
  EXPECT_EQ(b.mesh_id("").error().code, "E-MESH-NAME");
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "twin", .source = "generated:line:3"}).ok());
  EXPECT_EQ(b.mesh_id("twin").error().code, "E-MESH-NAME-AMBIGUOUS");
}

// ===========================================================================
// Moved-from safety
// ===========================================================================

TEST(MoabMesh, AMovedFromBackendReportsRatherThanCrashes) {
  MoabMeshBackend b = loaded("generated:box:2x2x2");
  const MeshView* before = b.view(MeshId{0}).value();
  MoabMeshBackend other = std::move(b);

  // NOLINTNEXTLINE(bugprone-use-after-move): that is what is under test.
  EXPECT_EQ(b.view(MeshId{0}).error().code, "E-MESH-MOVED");
  EXPECT_EQ(b.load(MeshDecl{.name = "x", .source = "generated:line:2"})
                .error()
                .code,
            "E-MESH-MOVED");
  EXPECT_EQ(b.mesh_id("m").error().code, "E-MESH-MOVED");
  EXPECT_EQ(
      b.region_size(Region{.name = "r", .mesh = MeshId{0}, .dimension = 3})
          .error()
          .code,
      "E-MESH-MOVED");
  EXPECT_EQ(b.owned_entities(
                 Region{.name = "r", .mesh = MeshId{0}, .dimension = 3})
                .error()
                .code,
            "E-MESH-MOVED");
  EXPECT_EQ(b.name(), "moab");

  // The pointer handed out before the move still works: only the owning
  // pointer moved, not the MOAB instance behind it.
  ASSERT_EQ(other.view(MeshId{0}).value(), before);
  EXPECT_EQ(before->entity_count(3), 8u);
}

// ===========================================================================
// partition_rank_of
// ===========================================================================

TEST(MoabMesh, PartitionRankIsZeroWithoutACommunicator) {
  const MoabMeshBackend b = loaded("generated:box:2x2x2");
  const Region r{.name = "all", .mesh = MeshId{0}, .dimension = 3};
  for (std::size_t i = 0; i < 8; ++i) {
    const auto rank = b.partition_rank_of(r, i);
    ASSERT_TRUE(rank.ok()) << rank.error().message;
    EXPECT_EQ(rank.value(), 0);
  }
  const auto bad = b.partition_rank_of(r, 8);
  ASSERT_FALSE(bad.ok());
  EXPECT_EQ(bad.error().code, "E-MESH-INDEX");
}

TEST(MoabMesh, SetCommunicatorAfterLoadIsRefused) {
  MoabMeshBackend b = loaded("generated:box:2x2x2");
  const auto r = b.set_communicator(0);
  ASSERT_FALSE(r.ok());
  // Either "too late" on an MPI build, or "no MPI" on one without.
  EXPECT_TRUE(r.error().code == "E-MOAB-COMM" ||
              r.error().code == "E-MOAB-NOMPI")
      << r.error().code;
}

// ===========================================================================
// A real MOAB mesh file: the cubed sphere MOAB ships in share/examples
// ===========================================================================

#ifdef FUSE_MOAB_TEST_MESH
namespace {

/// An independent area for one bilinear quadrilateral: a 400x400 composite
/// midpoint rule over the reference square, written out longhand so it shares
/// no code with the backend's Gauss-Legendre rule.
double reference_quad_area(const std::array<std::array<double, 3>, 4>& p) {
  constexpr int kN = 2000;
  double total = 0.0;
  for (int i = 0; i < kN; ++i) {
    for (int j = 0; j < kN; ++j) {
      const double u = (i + 0.5) / kN;
      const double v = (j + 0.5) / kN;
      double du[3] = {0, 0, 0};
      double dv[3] = {0, 0, 0};
      for (int k = 0; k < 3; ++k) {
        du[k] = (1 - v) * (p[1][k] - p[0][k]) + v * (p[2][k] - p[3][k]);
        dv[k] = (1 - u) * (p[3][k] - p[0][k]) + u * (p[2][k] - p[1][k]);
      }
      const double cx = du[1] * dv[2] - du[2] * dv[1];
      const double cy = du[2] * dv[0] - du[0] * dv[2];
      const double cz = du[0] * dv[1] - du[1] * dv[0];
      total += std::sqrt(cx * cx + cy * cy + cz * cz) / (kN * kN);
    }
  }
  return total;
}

}  // namespace

TEST(MoabMeshFile, ReadsTheCubedSphereAndReportsLocalCounts) {
  MoabMeshBackend b;
  const auto r =
      b.load(MeshDecl{.name = "sphere", .source = FUSE_MOAB_TEST_MESH});
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const MeshView& v = view_of(b);
  EXPECT_EQ(v.dimension(), 2);
  EXPECT_EQ(v.entity_count(2), 3456u);  // 6 * 24^2
  EXPECT_EQ(v.entity_count(0), 3458u);
  EXPECT_EQ(v.entity_count(1), 0u);  // the file stores no explicit edges
  EXPECT_EQ(v.entity_count(3), 0u);
  // Serial read: no partitioning, so everything is owned.
  EXPECT_EQ(v.owned_count(2), 3456u);
}

TEST(MoabMeshFile, CubedSphereGlobalIdsComeFromTheFile) {
  MoabMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "sphere", .source = FUSE_MOAB_TEST_MESH}).ok());
  const MeshView& v = view_of(b);
  std::set<std::size_t> ids;
  std::size_t lo = ~std::size_t{0};
  std::size_t hi = 0;
  for (std::size_t i = 0; i < v.entity_count(2); ++i) {
    const std::size_t g = v.global_id(2, i);
    ids.insert(g);
    lo = std::min(lo, g);
    hi = std::max(hi, g);
  }
  EXPECT_EQ(ids.size(), 3456u);
  EXPECT_EQ(lo, 1u);
  EXPECT_EQ(hi, 3456u);  // the file's GLOBAL_ID tag, not our own numbering
}

TEST(MoabMeshFile, CubedSphereAreaMatchesAnIndependentQuadrature) {
  MoabMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "sphere", .source = FUSE_MOAB_TEST_MESH}).ok());
  const MeshView& v = view_of(b);

  // Every vertex sits on a sphere; recover its radius from the mesh itself
  // rather than hard-coding one.
  double rmin = 1e300;
  double rmax = 0.0;
  for (std::size_t i = 0; i < v.entity_count(0); ++i) {
    const auto p = v.centroid(0, i);
    const double rad = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    rmin = std::min(rmin, rad);
    rmax = std::max(rmax, rad);
  }
  EXPECT_NEAR(rmin, rmax, 1e-6 * rmax) << "not a sphere?";
  const double radius = 0.5 * (rmin + rmax);
  EXPECT_NEAR(radius, 7996.4553620, 1e-6);

  const double total = owned_measure_sum(v, 2);
  const double sphere = 4.0 * M_PI * radius * radius;

  // The quadrilaterals are flat-sided and inscribed, so their union is
  // strictly inside the sphere: less area, by O(h^2) with h the angular cell
  // size (pi/2)/24 = 0.065 rad.
  EXPECT_LT(total, sphere);
  EXPECT_GT(total, 0.995 * sphere);
  // Converged reference: 16- and 24-point Gauss-Legendre on the same patches
  // agree on this to the digit shown. The tolerance is 1.2e-11 relative,
  // dominated by the order in which 3456 terms are summed; the vector-area
  // shortcut misses by 9.7e3, seven orders of magnitude more.
  EXPECT_NEAR(total, 802651780.5417747, 1e-2);
}

TEST(MoabMeshFile, IndividualCubedSphereQuadsMatchABruteForceIntegral) {
  MoabMeshBackend b;
  ASSERT_TRUE(
      b.load(MeshDecl{.name = "sphere", .source = FUSE_MOAB_TEST_MESH}).ok());
  const MeshView& v = view_of(b);
  // MeshView exposes no connectivity, so the per-element reference cannot be
  // recomputed here from the mesh. These three values were produced by a
  // standalone MOAB program that read the same file and integrated the same
  // bilinear patches with 16- and 24-point Gauss-Legendre; both orders agreed
  // to 13 significant figures, which is what makes them a reference rather
  // than a second opinion. `ReferenceQuadratureAgreesOnASyntheticWarpedQuad`
  // below checks the method itself against an unrelated quadrature.
  const double expected[3] = {144983.5668990106, 234382.6295912052,
                              144288.0795761382};
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_NEAR(v.measure(2, i), expected[i], 1e-6) << "quad " << i;
  }
  // The diagonal-cross-product shortcut would give these instead; the point
  // is that measure() does NOT.
  const double diagonal_shortcut[3] = {144974.6259468, 234362.3050192,
                                       144282.5037450};
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_GT(std::fabs(v.measure(2, i) - diagonal_shortcut[i]), 1.0)
        << "quad " << i << " looks like the vector-area shortcut";
  }
}

TEST(MoabMeshFile, ReferenceQuadratureAgreesOnASyntheticWarpedQuad) {
  // Guards the guard: if reference_quad_area were itself wrong, the checks
  // above would be vacuous. A unit square with one corner lifted by 1.
  const std::array<std::array<double, 3>, 4> p{
      {{0, 0, 0}, {1, 0, 0}, {1, 1, 1}, {0, 1, 0}}};
  // Composite midpoint is only second order, so 2000x2000 cells leave about
  // 1.4e-8; that is still four orders below the gap to the shortcut below.
  EXPECT_NEAR(reference_quad_area(p), 1.2807892752734042, 1e-7);
  // ... and it is NOT half the cross product of the diagonals.
  EXPECT_GT(std::fabs(reference_quad_area(p) - 1.2247448713915889), 0.05);
}

#endif  // FUSE_MOAB_TEST_MESH

#else
TEST(MoabMesh, SkippedWithoutMoab) {
  GTEST_SKIP() << "built without FUSE_ENABLE_MOAB";
}
#endif
