// physics.fe_test -- ported from KARMA's fe_test.C (the volume-
// integration portions; item 6's pyramid-gradients section is not ported,
// no pyramid variant exists this wave). Exercises FEBase against a real
// mesh element (fuse_physics::spatial::Mesh's native Hex8 generate_box),
// the one place this wave's FE core, basis and geometry-map ports all run
// together.

#include "fuse_physics/fe/fe_base.hpp"

#include <cmath>

#include <gtest/gtest.h>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::FEBase;
using fuse::physics::fe::FEIntegrationDomain;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::spatial::Mesh;

Mesh unit_cube_mesh(unsigned n) {
  Mesh mesh(3);
  mesh.generate_box(n, n, n, Point(0, 0, 0), Point(1, 1, 1));
  return mesh;
}

// physics.fe_test.PhiSumsToOneAtEveryQuadraturePoint -- the partition-of-
// unity property (reference_topology_test's own check) must survive
// FEBase's own evaluate_reference_points plumbing unchanged, since phi is
// copied straight from the reference basis with no transform applied.
TEST(FEBaseTest, PhiSumsToOneAtEveryQuadraturePoint) {
  const Mesh mesh = unit_cube_mesh(2);
  const BasisRegistry registry;
  FEBase fe(mesh, GeometryVariant::Hex8,
            registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2);
  fe.reinit(1);
  ASSERT_TRUE(fe.is_valid());
  for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
    double sum = 0.0;
    for (const auto& row : fe.get_phi()) sum += row[q];
    EXPECT_NEAR(sum, 1.0, 1.0e-12);
  }
}

// physics.fe_test.SumOfJxWEqualsElementVolume -- integrating the constant
// function 1 over the element (sum of JxW) recovers the element's own
// volume, for a non-cubic (anisotropically scaled) box element.
TEST(FEBaseTest, SumOfJxWEqualsElementVolume) {
  Mesh mesh(3);
  mesh.generate_box(2, 3, 1, Point(0, 0, 0), Point(2.0, 3.0, 0.5));
  const BasisRegistry registry;
  FEBase fe(mesh, GeometryVariant::Hex8,
            registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG}), 2);
  double total = 0.0;
  for (std::size_t element = 1; element <= mesh.n_elements(); ++element) {
    fe.reinit(element);
    for (const double jxw : fe.get_JxW()) total += jxw;
  }
  const double expected_volume = 2.0 * 3.0 * 0.5;
  EXPECT_NEAR(total, expected_volume, 1.0e-10);
}

// physics.fe_test.GradientsSumToZero -- the mapped physical gradients
// (get_dphi) still sum to zero at every quadrature point: the chain rule
// (gradient_transform = J * metric^-1) is linear, so the reference
// derivative-sums-to-zero identity (fe_reference_topology_test.cpp)
// survives the physical-space transform unchanged.
TEST(FEBaseTest, GradientsSumToZero) {
  const Mesh mesh = unit_cube_mesh(1);
  const BasisRegistry registry;
  FEBase fe(mesh, GeometryVariant::Hex8,
            registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2);
  fe.reinit(1);
  for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
    Point sum = Point::zero();
    for (const auto& row : fe.get_dphi()) sum += row[q];
    EXPECT_NEAR(sum.x, 0.0, 1.0e-9);
    EXPECT_NEAR(sum.y, 0.0, 1.0e-9);
    EXPECT_NEAR(sum.z, 0.0, 1.0e-9);
  }
}

// physics.fe_test.MismatchedBasisCellRefuses -- constructing an FEBase
// with a basis whose BasisKey.cell disagrees with the geometry variant's
// own cell family must refuse (fe_base.cpp's own constructor check).
TEST(FEBaseTest, MismatchedBasisCellRefuses) {
  const Mesh mesh = unit_cube_mesh(1);
  const BasisRegistry registry;
  EXPECT_THROW(FEBase(mesh, GeometryVariant::Hex8,
                       registry.create({CellFamily::Tetrahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}),
                       2),
               std::invalid_argument);
}

}  // namespace
