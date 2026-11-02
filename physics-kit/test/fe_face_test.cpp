// physics.fe_face_test -- ported from KARMA's fe_face_test.C (face-
// integration portions). WAVE 1b (owner instruction 2026-10-31, item 1)
// ports the mixed triangle/quad face case too, now that Prism and
// Pyramid are in scope: EveryFaceAreaOfPrismAndPyramidMatchesItsOwnShape
// exercises fe_base.cpp's own quadrature_for_side() dispatch directly --
// every quad side goes through `quadrature_`, every triangular side
// through `triangle_quadrature_`, on the SAME FEBase instance.

#include "fuse_physics/fe/fe_base.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::dot;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::FEBase;
using fuse::physics::fe::FEIntegrationDomain;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::spatial::Mesh;

Mesh single_unit_cube() {
  Mesh mesh(3);
  mesh.generate_box(1, 1, 1, Point(0, 0, 0), Point(1, 1, 1));
  return mesh;
}

// physics.fe_face_test.EveryFaceAreaOfUnitCubeIsOne -- each of the unit
// cube's six faces has area 1 (sum of JxW over the face's quadrature
// points), the face-integration analogue of fe_test's volume check.
TEST(FEFaceTest, EveryFaceAreaOfUnitCubeIsOne) {
  const Mesh mesh = single_unit_cube();
  const BasisRegistry registry;
  FEBase fe(mesh, GeometryVariant::Hex8,
            registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2,
            FEIntegrationDomain::Face);
  for (unsigned side = 0; side < 6; ++side) {
    fe.reinit(1, side);
    ASSERT_TRUE(fe.is_valid());
    double area = 0.0;
    for (const double jxw : fe.get_JxW()) area += jxw;
    EXPECT_NEAR(area, 1.0, 1.0e-10) << "side " << side;
  }
}

// physics.fe_face_test.NormalsPointAwayFromCentroid -- every face normal
// of the unit cube points away from the element centroid (0.5,0.5,0.5),
// the orientation convention side_from_parent's canonical-co-normal check
// (geometry_map.cpp) exists to guarantee.
TEST(FEFaceTest, NormalsPointAwayFromCentroid) {
  const Mesh mesh = single_unit_cube();
  const BasisRegistry registry;
  FEBase fe(mesh, GeometryVariant::Hex8,
            registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2,
            FEIntegrationDomain::Face);
  const Point centroid = mesh.centroid(1);
  for (unsigned side = 0; side < 6; ++side) {
    fe.reinit(1, side);
    for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
      const Point outward = fe.get_xyz()[q] - centroid;
      EXPECT_GT(dot(fe.get_normals()[q], outward), 0.0) << "side " << side << " point " << q;
    }
  }
}

// physics.fe_face_test.FaceReinitOnVolumeFiniteElementRefuses -- calling
// the (element, side) overload on an FEBase built with
// FEIntegrationDomain::Volume must refuse (fe_base.cpp's own domain check).
TEST(FEFaceTest, FaceReinitOnVolumeFiniteElementRefuses) {
  const Mesh mesh = single_unit_cube();
  const BasisRegistry registry;
  FEBase fe(mesh, GeometryVariant::Hex8,
            registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2,
            FEIntegrationDomain::Volume);
  EXPECT_THROW(fe.reinit(1, 0), std::invalid_argument);
}

// physics.fe_face_test.EveryFaceAreaOfPrismAndPyramidMatchesItsOwnShape
// -- the reference Prism6 and Pyramid5 (identity-mapped), each face's
// area computed by hand from its own corner coordinates (reference_
// topology.cpp's side_table: Prism6 sides 0-2 are the three
// quadrilateral side faces, 3-4 the two triangular caps; Pyramid5 sides
// 0-3 are the four triangular sides, 4 the quadrilateral base).
TEST(FEFaceTest, EveryFaceAreaOfPrismAndPyramidMatchesItsOwnShape) {
  const BasisRegistry registry;
  {
    Mesh mesh(3);
    const std::array<Point, 6> corners{Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0),
                                        Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1)};
    std::vector<fuse::physics::spatial::EntityHandle> vertices;
    for (const auto& corner : corners) vertices.push_back(mesh.add_vertex(corner));
    mesh.add_element(GeometryVariant::Prism6, vertices);
    FEBase fe(mesh, GeometryVariant::Prism6,
              registry.create({CellFamily::Prism, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2,
              FEIntegrationDomain::Face);
    const std::array<double, 5> expected_area{1.0, std::sqrt(2.0), 1.0, 0.5, 0.5};
    for (unsigned side = 0; side < 5; ++side) {
      fe.reinit(1, side);
      ASSERT_TRUE(fe.is_valid()) << "prism side " << side;
      double area = 0.0;
      for (const double jxw : fe.get_JxW()) area += jxw;
      EXPECT_NEAR(area, expected_area[side], 1.0e-10) << "prism side " << side;
    }
  }
  {
    Mesh mesh(3);
    const std::array<Point, 5> corners{Point(-1, -1, 0), Point(1, -1, 0), Point(1, 1, 0), Point(-1, 1, 0),
                                        Point(0, 0, 1)};
    std::vector<fuse::physics::spatial::EntityHandle> vertices;
    for (const auto& corner : corners) vertices.push_back(mesh.add_vertex(corner));
    mesh.add_element(GeometryVariant::Pyramid5, vertices);
    FEBase fe(mesh, GeometryVariant::Pyramid5,
              registry.create({CellFamily::Pyramid, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2,
              FEIntegrationDomain::Face);
    const std::array<double, 5> expected_area{std::sqrt(2.0), std::sqrt(2.0), std::sqrt(2.0), std::sqrt(2.0), 4.0};
    for (unsigned side = 0; side < 5; ++side) {
      fe.reinit(1, side);
      ASSERT_TRUE(fe.is_valid()) << "pyramid side " << side;
      double area = 0.0;
      for (const double jxw : fe.get_JxW()) area += jxw;
      EXPECT_NEAR(area, expected_area[side], 1.0e-9) << "pyramid side " << side;
    }
  }
}

// physics.fe_face_test.PrismAndPyramidNormalsPointAwayFromCentroid --
// the same orientation convention NormalsPointAwayFromCentroid checks
// on a Hex8, checked here on a mixed-face Prism6 -- the quad and
// triangular sides both go through side_from_parent's own canonical-
// co-normal check (geometry_map.cpp), regardless of which quadrature
// rule supplied the side's own points.
TEST(FEFaceTest, PrismAndPyramidNormalsPointAwayFromCentroid) {
  const BasisRegistry registry;
  Mesh mesh(3);
  const std::array<Point, 6> corners{Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0),
                                      Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1)};
  std::vector<fuse::physics::spatial::EntityHandle> vertices;
  for (const auto& corner : corners) vertices.push_back(mesh.add_vertex(corner));
  mesh.add_element(GeometryVariant::Prism6, vertices);
  FEBase fe(mesh, GeometryVariant::Prism6,
            registry.create({CellFamily::Prism, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2,
            FEIntegrationDomain::Face);
  const Point centroid = mesh.centroid(1);
  for (unsigned side = 0; side < 5; ++side) {
    fe.reinit(1, side);
    for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
      const Point outward = fe.get_xyz()[q] - centroid;
      EXPECT_GT(dot(fe.get_normals()[q], outward), 0.0) << "side " << side << " point " << q;
    }
  }
}

}  // namespace
