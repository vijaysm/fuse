// physics.fe_face_test -- ported from KARMA's fe_face_test.C (face-
// integration portions; the mixed triangle/quad face case, which only
// arises for prism/pyramid parents, is not ported -- this wave's only
// face-capable parent is Hex8, whose faces are uniformly quadrilateral).

#include "fuse_physics/fe/fe_base.hpp"

#include <cmath>

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

}  // namespace
