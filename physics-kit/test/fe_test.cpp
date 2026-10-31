// physics.fe_test -- ported from KARMA's fe_test.C (the volume-
// integration portions). Exercises FEBase against a real mesh element
// (fuse_physics::spatial::Mesh's native Hex8 generate_box), the one
// place this wave's FE core, basis and geometry-map ports all run
// together.
//
// WAVE 1b (owner instruction 2026-10-31, item 1): KARMA's own item 6
// pyramid-gradients section, not ported in wave 1 (no pyramid variant
// existed), is ported now -- PhiSumsToOne and GradientsSumToZero both
// get a Prism and a Pyramid case, built directly via Mesh::add_vertex/
// add_element (generate_box is still Hex8-only) rather than through a
// box generator neither cell has one for.

#include "fuse_physics/fe/fe_base.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::cross;
using fuse::physics::dot;
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

/// One Prism6 element, physical nodes equal to the reference Prism6's
/// own corners (geometry_map_test.cpp's own identity-map construction) --
/// nonzero, nondegenerate volume, with no box generator needed.
Mesh single_prism_mesh() {
  Mesh mesh(3);
  const std::array<Point, 6> corners{Point(0, 0, 0), Point(1, 0, 0), Point(0, 1, 0),
                                      Point(0, 0, 1), Point(1, 0, 1), Point(0, 1, 1)};
  std::vector<fuse::physics::spatial::EntityHandle> vertices;
  for (const auto& corner : corners) vertices.push_back(mesh.add_vertex(corner));
  mesh.add_element(GeometryVariant::Prism6, vertices);
  return mesh;
}

/// One Pyramid5 element, physical nodes equal to the reference Pyramid5's
/// own corners.
Mesh single_pyramid_mesh() {
  Mesh mesh(3);
  const std::array<Point, 5> corners{Point(-1, -1, 0), Point(1, -1, 0), Point(1, 1, 0), Point(-1, 1, 0),
                                      Point(0, 0, 1)};
  std::vector<fuse::physics::spatial::EntityHandle> vertices;
  for (const auto& corner : corners) vertices.push_back(mesh.add_vertex(corner));
  mesh.add_element(GeometryVariant::Pyramid5, vertices);
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

// physics.fe_test.PhiSumsToOneAndGradientsSumToZeroOnPrismAndPyramid --
// KARMA's own fe_test.C item 6 (pyramid gradients), ported now that
// Prism and Pyramid are both in scope (owner instruction 2026-10-31,
// item 1): the same two identities PhiSumsToOneAtEveryQuadraturePoint
// and GradientsSumToZero above check on a Hex8, checked here on a
// single Prism6 and a single Pyramid5 element.
TEST(FEBaseTest, PhiSumsToOneAndGradientsSumToZeroOnPrismAndPyramid) {
  const BasisRegistry registry;
  for (const auto& [mesh, variant, cell] :
       {std::tuple{single_prism_mesh(), GeometryVariant::Prism6, CellFamily::Prism},
        std::tuple{single_pyramid_mesh(), GeometryVariant::Pyramid5, CellFamily::Pyramid}}) {
    FEBase fe(mesh, variant, registry.create({cell, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG}), 2);
    fe.reinit(1);
    ASSERT_TRUE(fe.is_valid()) << "cell " << static_cast<int>(cell);
    for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) {
      double phi_sum = 0.0;
      Point gradient_sum = Point::zero();
      for (std::size_t i = 0; i < fe.get_phi().size(); ++i) {
        phi_sum += fe.get_phi()[i][q];
        gradient_sum += fe.get_dphi()[i][q];
      }
      EXPECT_NEAR(phi_sum, 1.0, 1.0e-10) << "cell " << static_cast<int>(cell) << " q " << q;
      EXPECT_NEAR(gradient_sum.x, 0.0, 1.0e-8) << "cell " << static_cast<int>(cell) << " q " << q;
      EXPECT_NEAR(gradient_sum.y, 0.0, 1.0e-8) << "cell " << static_cast<int>(cell) << " q " << q;
      EXPECT_NEAR(gradient_sum.z, 0.0, 1.0e-8) << "cell " << static_cast<int>(cell) << " q " << q;
    }
  }
}

// physics.fe_test.SumOfJxWEqualsElementVolumeOnPrismAndPyramid -- the
// same volume-recovery identity as SumOfJxWEqualsElementVolume above,
// on the identity-mapped Prism6 (true volume 1/2, triangle base x unit
// height) and Pyramid5 (true volume 4/3, a 2x2 base x unit height
// pyramid: (1/3) x base_area x height = (1/3) x 4 x 1) built above.
TEST(FEBaseTest, SumOfJxWEqualsElementVolumeOnPrismAndPyramid) {
  const BasisRegistry registry;
  {
    const Mesh mesh = single_prism_mesh();
    FEBase fe(mesh, GeometryVariant::Prism6,
              registry.create({CellFamily::Prism, FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG}), 2);
    fe.reinit(1);
    double total = 0.0;
    for (const double jxw : fe.get_JxW()) total += jxw;
    EXPECT_NEAR(total, 0.5, 1.0e-10);
  }
  {
    const Mesh mesh = single_pyramid_mesh();
    FEBase fe(mesh, GeometryVariant::Pyramid5,
              registry.create({CellFamily::Pyramid, FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG}), 2);
    fe.reinit(1);
    double total = 0.0;
    for (const double jxw : fe.get_JxW()) total += jxw;
    EXPECT_NEAR(total, 4.0 / 3.0, 1.0e-9);
  }
}

// physics.fe_test.DistortedPyramidQuadratureAccuracy_D100 -- KNOWN_ISSUES.md
// D100 (KARMA's own KNOWN_ISSUES.md item 1, carried forward since wave 1
// as an unverified risk because no pyramid variant existed to test it):
// a DISTORTED (non-reference-shaped) pyramid's isoparametric map is a
// genuinely RATIONAL function of the reference coordinate (not a
// polynomial, even for a straight-edged pyramid -- the collapsed-
// coordinate basis itself is rational), so the quadrature rule's own
// polynomial exactness (QuadratureTest.PyramidExactToDeclaredDegree)
// does not immediately carry over to the Jacobian-weighted integrand on
// a distorted cell; KARMA's own KNOWN_ISSUES.md item 1 states this is
// "a question of relative accuracy, not of exactness." MEASURED here:
// the quadrature-integrated volume of a markedly distorted (planar
// base, but irregular quadrilateral, apex off-centre and not above the
// base in any symmetric sense) pyramid, at increasing declared degree,
// against the volume computed independently of this FE code entirely
// (the standard two-tetrahedron decomposition of a straight-edged
// pyramid, exact in closed form for ANY planar-base pyramid regardless
// of distortion -- this is genuinely independent ground truth, not a
// value produced by GeometryMap/Quadrature). g_pyramid_distortion_errors
// is read back by the ZZZ test below to put the actual numbers in the
// submission's measured_results and KNOWN_ISSUES.md D100's update.
std::vector<double> g_pyramid_distortion_errors;
std::vector<double> g_pyramid_smooth_errors;

TEST(FEBaseTest, DistortedPyramidQuadratureAccuracy_D100) {
  // Planar base (z=0, so the solid pyramid's volume is unambiguous --
  // verified by hand: splitting the quad base along EITHER diagonal into
  // two tetrahedra gives the same total, 2.52, to the last digit), apex
  // well off-centre and off-axis.
  const std::array<Point, 5> corners{Point(-1, -1, 0), Point(1.3, -0.9, 0), Point(1.1, 1.4, 0), Point(-0.8, 1.2, 0),
                                      Point(0.5, 0.4, 1.6)};
  const auto tet_volume = [](const Point& a, const Point& b, const Point& c, const Point& d) {
    return std::abs(dot(a - d, cross(b - d, c - d))) / 6.0;
  };
  const double analytic_volume = tet_volume(corners[0], corners[1], corners[2], corners[4]) +
                                  tet_volume(corners[0], corners[2], corners[3], corners[4]);
  ASSERT_NEAR(analytic_volume, 2.52, 1.0e-12);  // the hand computation this test's comment states

  Mesh mesh(3);
  std::vector<fuse::physics::spatial::EntityHandle> vertices;
  for (const auto& corner : corners) vertices.push_back(mesh.add_vertex(corner));
  mesh.add_element(GeometryVariant::Pyramid5, vertices);
  const BasisRegistry registry;

  for (const int degree : {1, 2, 3, 4, 5}) {
    FEBase fe(mesh, GeometryVariant::Pyramid5,
              registry.create({CellFamily::Pyramid, FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG}),
              degree);
    fe.reinit(1);
    double quadrature_volume = 0.0;
    for (const double jxw : fe.get_JxW()) quadrature_volume += jxw;
    const double error = std::abs(quadrature_volume - analytic_volume);
    g_pyramid_distortion_errors.push_back(error);
  }
  // The risk D100 names is a persistent, non-shrinking bias; demonstrate
  // there is none here (error shrinks, and is already tiny well before
  // the quadrature degree this wave's own basis orders need, degree 2).
  for (std::size_t i = 1; i < g_pyramid_distortion_errors.size(); ++i)
    EXPECT_LE(g_pyramid_distortion_errors[i], g_pyramid_distortion_errors[i - 1] + 1.0e-13)
        << "degree " << (i + 1) << " was less accurate than degree " << i;
  EXPECT_LT(g_pyramid_distortion_errors.back(), 1.0e-10);

  // KARMA's own D100 wording ("a question of relative accuracy, not of
  // exactness") is squarely about a NON-polynomial integrand, which the
  // constant-function volume check above cannot exercise (a constant is
  // degree 0, trivially exact at any degree even through a rational
  // Jacobian, as the measurement above's own round-off-level errors
  // show). A smooth but genuinely non-polynomial integrand -- here
  // exp(x+y+z), evaluated at the SAME quadrature points this FEBase
  // already reports via get_xyz() -- against a reference value from an
  // independent, much higher-degree rule, isolates the truncation
  // behaviour the constant-function check cannot.
  const auto smooth_integrand = [](const Point& p) { return std::exp(p.x + p.y + p.z); };
  std::vector<double> smooth_values;
  for (const int degree : {1, 2, 3, 4, 5, 10, 16}) {
    FEBase fe(mesh, GeometryVariant::Pyramid5,
              registry.create({CellFamily::Pyramid, FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG}),
              degree);
    fe.reinit(1);
    double integral = 0.0;
    for (std::size_t q = 0; q < fe.n_quadrature_points(); ++q) integral += fe.get_JxW()[q] * smooth_integrand(fe.get_xyz()[q]);
    smooth_values.push_back(integral);
  }
  const double reference_value = smooth_values.back();  // degree 16, the most accurate available
  for (std::size_t i = 0; i < smooth_values.size(); ++i)
    g_pyramid_smooth_errors.push_back(std::abs(smooth_values[i] - reference_value));
  // Converges towards the high-degree reference (not a persistent
  // plateaued bias): each of the first few degrees is at least as
  // accurate as the one before it.
  for (std::size_t i = 1; i + 1 < g_pyramid_smooth_errors.size(); ++i)
    EXPECT_LE(g_pyramid_smooth_errors[i], g_pyramid_smooth_errors[i - 1] + 1.0e-12)
        << "smooth integrand degree index " << i << " regressed";
}

// physics.fe_test.ZZZ_D100MeasurementIsPrinted -- runs last (gtest's
// definition-order rule, same device fe_quadrature_test.cpp's own ZZZ
// test uses), so the five degree -> error pairs DistortedPyramidQuadrature
// Accuracy_D100 measured are a plain, greppable line in the test log for
// KNOWN_ISSUES.md D100's update and the submission's measured_results,
// not a number this file's own comments would otherwise only assert
// silently passed.
TEST(FEBaseTest, ZZZ_D100MeasurementIsPrinted) {
  ASSERT_EQ(g_pyramid_distortion_errors.size(), 5u);
  for (std::size_t i = 0; i < g_pyramid_distortion_errors.size(); ++i)
    std::printf("D100 distorted pyramid: degree %zu volume error = %.3e\n", i + 1, g_pyramid_distortion_errors[i]);
  const std::array<int, 7> smooth_degrees{1, 2, 3, 4, 5, 10, 16};
  ASSERT_EQ(g_pyramid_smooth_errors.size(), smooth_degrees.size());
  for (std::size_t i = 0; i < g_pyramid_smooth_errors.size(); ++i)
    std::printf("D100 distorted pyramid: degree %d exp(x+y+z) error vs degree-16 reference = %.3e\n",
                smooth_degrees[i], g_pyramid_smooth_errors[i]);
}

}  // namespace
