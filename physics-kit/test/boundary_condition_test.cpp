// physics.boundary_condition -- fuse_physics::fe::BoundaryConditionManager,
// adapted from KARMA's karma_boundary_condition.h/karma_physics_bc_manager.h.
// See boundary_condition.hpp's own header comment for how physics-kit's
// geometric boundary identity and single-rank scope diverge from KARMA's
// own MOAB side-set lookup.

#include "fuse_physics/fe/boundary_condition.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/spatial/domain.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::Basis;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisKey;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::BoundaryCondition;
using fuse::physics::fe::BoundaryConditionKind;
using fuse::physics::fe::BoundaryConditionManager;
using fuse::physics::fe::BoundaryId;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::classify_position;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::Domain;
using fuse::physics::spatial::Mesh;
using fuse::physics::spatial::Shape;

Variable scalar_p1(std::string name) {
  return Variable{std::move(name), FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
}

Variable scalar_p2(std::string name) {
  return Variable{std::move(name), FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1};
}

// physics.boundary_condition.ClassifyPositionFindsEveryFaceAPointLiesOn --
// an interior point classifies to no face; a face point to exactly one;
// a corner to the two (2-D) or three (3-D) faces it lies on at once.
// The box's own lower corner (lo) is passed explicitly and need not be
// the origin (ClassifyPositionUsesTheGivenLowerCornerNotTheOrigin below
// exercises a non-zero one).
TEST(BoundaryConditionTest, ClassifyPositionFindsEveryFaceAPointLiesOn) {
  const Point lo(0.0, 0.0, 0.0);
  const double tol = 0.0;  // every coordinate below is exactly representable
  EXPECT_TRUE(classify_position(Point(1.0, 1.5, 0.0), lo, 2.0, 3.0, 0.0, 2, tol).empty());
  EXPECT_EQ(classify_position(Point(0.0, 1.5, 0.0), lo, 2.0, 3.0, 0.0, 2, tol), (std::vector<BoundaryId>{BoundaryId::XMin}));
  EXPECT_EQ(classify_position(Point(2.0, 1.5, 0.0), lo, 2.0, 3.0, 0.0, 2, tol), (std::vector<BoundaryId>{BoundaryId::XMax}));
  EXPECT_EQ(classify_position(Point(1.0, 0.0, 0.0), lo, 2.0, 3.0, 0.0, 2, tol), (std::vector<BoundaryId>{BoundaryId::YMin}));
  const auto corner = classify_position(Point(0.0, 0.0, 0.0), lo, 2.0, 3.0, 0.0, 2, tol);
  EXPECT_EQ(corner.size(), 2u);
  EXPECT_NE(std::find(corner.begin(), corner.end(), BoundaryId::XMin), corner.end());
  EXPECT_NE(std::find(corner.begin(), corner.end(), BoundaryId::YMin), corner.end());
  // dimension == 2 means ZMin/ZMax never match, even at z == 0 -- there
  // is no third axis to classify against.
  EXPECT_TRUE(classify_position(Point(0.0, 0.0, 0.0), lo, 2.0, 3.0, 0.0, 2, tol).size() == 2u);
}

// physics.boundary_condition.ClassifyPositionUsesTheGivenLowerCornerNotTheOrigin
// -- a box whose own lower corner sits at (5, -3, 0), not the origin:
// the SAME point classifies differently depending on which lower corner
// is given, proving the lower corner is read, not assumed to be zero.
TEST(BoundaryConditionTest, ClassifyPositionUsesTheGivenLowerCornerNotTheOrigin) {
  // Box: x in [5, 7], y in [5, 7] -- every face sits away from x=0
  // and y=0, so the ORIGIN (0, 0, 0) is unambiguously outside every
  // face of this box (distance 5 to each of the four face planes),
  // with no coordinate of the origin coinciding with any face by
  // construction. A classifier that (incorrectly) compared against
  // the literal 0.0 instead of the measured lower corner would wrongly
  // report the origin as matching XMin/YMin here.
  const Point lo(5.0, 5.0, 0.0);
  const double tol = 0.0;  // every coordinate below is exactly representable
  EXPECT_TRUE(classify_position(Point(0.0, 0.0, 0.0), lo, 2.0, 2.0, 0.0, 2, tol).empty());
  EXPECT_TRUE(classify_position(Point(6.0, 6.0, 0.0), lo, 2.0, 2.0, 0.0, 2, tol).empty());
  EXPECT_EQ(classify_position(Point(5.0, 6.0, 0.0), lo, 2.0, 2.0, 0.0, 2, tol), (std::vector<BoundaryId>{BoundaryId::XMin}));
  EXPECT_EQ(classify_position(Point(7.0, 6.0, 0.0), lo, 2.0, 2.0, 0.0, 2, tol), (std::vector<BoundaryId>{BoundaryId::XMax}));
  EXPECT_EQ(classify_position(Point(6.0, 5.0, 0.0), lo, 2.0, 2.0, 0.0, 2, tol), (std::vector<BoundaryId>{BoundaryId::YMin}));
  EXPECT_EQ(classify_position(Point(6.0, 7.0, 0.0), lo, 2.0, 2.0, 0.0, 2, tol), (std::vector<BoundaryId>{BoundaryId::YMax}));
}

// physics.boundary_condition.ResolveSeparatesEveryBoundaryTypeOnTheUnitSquare
// -- a 2x2 Quad4 mesh on [0,2]x[0,2] with XMin Dirichlet, XMax Neumann,
// YMin Robin, YMax Dirichlet resolves to the geometrically correct
// dirichlet dof count, neumann face count and robin face count: exactly
// the boundary vertices/sides of the box, none missed or duplicated.
TEST(BoundaryConditionTest, ResolveSeparatesEveryBoundaryTypeOnTheUnitSquare) {
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(0, 0, 0), Point(2, 2, 0));
  BasisRegistry registry;
  DofMap dof_map(mesh, {scalar_p1("u")}, registry);
  const Basis basis = registry.create(BasisKey{CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1,
                                                BasisContinuity::cG});
  const Domain domain(2, Shape::Square, 2.0, 2.0, 0.0, false);

  BoundaryConditionManager manager(mesh, domain);
  manager.set(BoundaryId::XMin, BoundaryCondition{BoundaryConditionKind::Dirichlet,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.set(BoundaryId::XMax, BoundaryCondition{BoundaryConditionKind::Neumann,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.set(BoundaryId::YMin, BoundaryCondition{BoundaryConditionKind::Robin,
                                                   [](const Point&, double) { return 0.0; }, 1.0});
  manager.set(BoundaryId::YMax, BoundaryCondition{BoundaryConditionKind::Dirichlet,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.resolve(dof_map, basis);

  // 3 dofs sit exactly on the XMin edge (n=2 -> 3 vertices per edge) and
  // 3 on YMax, sharing ONE corner, (0,2) -- 3 + 3 - 1 = 5 distinct
  // Dirichlet dofs.
  EXPECT_EQ(manager.dirichlet_dofs().size(), 5u);
  // 2 boundary sides sit on XMax (n=2 -> 2 elements along that edge) and
  // 2 on YMin.
  EXPECT_EQ(manager.neumann_faces().size(), 2u);
  EXPECT_EQ(manager.robin_faces().size(), 2u);
}

// physics.boundary_condition.ResolveOnAShiftedBoxUsesTheMeasuredLowerCorner
// -- the SAME mesh and boundary registration as the test above, generated
// from the lower corner (5, -3, 0) instead of the origin: the counts must
// be identical, since resolution reads the measured lower corner, not an
// assumed one.
TEST(BoundaryConditionTest, ResolveOnAShiftedBoxUsesTheMeasuredLowerCorner) {
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(5, -3, 0), Point(7, -1, 0));
  BasisRegistry registry;
  DofMap dof_map(mesh, {scalar_p1("u")}, registry);
  const Basis basis = registry.create(BasisKey{CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1,
                                                BasisContinuity::cG});
  const Domain domain(2, Shape::Square, 2.0, 2.0, 0.0, false);

  BoundaryConditionManager manager(mesh, domain);
  manager.set(BoundaryId::XMin, BoundaryCondition{BoundaryConditionKind::Dirichlet,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.set(BoundaryId::XMax, BoundaryCondition{BoundaryConditionKind::Neumann,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.set(BoundaryId::YMin, BoundaryCondition{BoundaryConditionKind::Robin,
                                                   [](const Point&, double) { return 0.0; }, 1.0});
  manager.set(BoundaryId::YMax, BoundaryCondition{BoundaryConditionKind::Dirichlet,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.resolve(dof_map, basis);

  EXPECT_EQ(manager.dirichlet_dofs().size(), 5u);
  EXPECT_EQ(manager.neumann_faces().size(), 2u);
  EXPECT_EQ(manager.robin_faces().size(), 2u);
  for (const auto& d : manager.dirichlet_dofs())
    // Integer coordinates: the generated positions are exact.
    EXPECT_TRUE(d.position.x == 5.0 || d.position.x == 7.0 || d.position.y == -1.0)
        << "dirichlet dof position (" << d.position.x << "," << d.position.y << ") not on the shifted box boundary";
}

// physics.boundary_condition.ResolveOnABoxFarFromTheOriginMatchesEveryFace
// -- a 7 x 5 mesh of the box [1000000.3, 1000002.6] x [-300000.7,
// -299999.0], of size 2.3 x 1.7. The generated coordinate of the face
// x = max and the lower corner plus 2.3 are both rounded at the magnitude
// 1e6 and differ by 1.2e-10 (asserted below as the premise), far above
// the unit roundoff times the box size, so a face tolerance scaled by the
// lengths alone does not find that face. Every face must resolve, with
// the counts of the grid: 6 + 6 vertices on the two Dirichlet faces
// x = min and x = max, and 7 sides on each of y = min (Neumann) and
// y = max (Robin).
TEST(BoundaryConditionTest, ResolveOnABoxFarFromTheOriginMatchesEveryFace) {
  const Point lo(1000000.3, -300000.7, 0.0);
  const Point hi(1000002.6, -299999.0, 0.0);
  Mesh mesh(2);
  mesh.generate_2d_box(7, 5, lo, hi);
  BasisRegistry registry;
  DofMap dof_map(mesh, {scalar_p1("u")}, registry);
  const Basis basis = registry.create(BasisKey{CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1,
                                                BasisContinuity::cG});
  const Domain domain(2, Shape::Square, 2.3, 1.7, 0.0, false);
  // The premise. The left operand is the coordinate generate_2d_box gives
  // the face x = max; the right one is where the manager looks for it.
  const volatile double generated = lo.x + (hi.x - lo.x);
  const volatile double sought = lo.x + 2.3;
  ASSERT_GT(std::fabs(generated - sought), 2.0 * std::numeric_limits<double>::epsilon() * 2.3);

  BoundaryConditionManager manager(mesh, domain);
  manager.set(BoundaryId::XMin, BoundaryCondition{BoundaryConditionKind::Dirichlet,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.set(BoundaryId::XMax, BoundaryCondition{BoundaryConditionKind::Dirichlet,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.set(BoundaryId::YMin, BoundaryCondition{BoundaryConditionKind::Neumann,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  manager.set(BoundaryId::YMax, BoundaryCondition{BoundaryConditionKind::Robin,
                                                   [](const Point&, double) { return 0.0; }, 1.0});
  manager.resolve(dof_map, basis);
  EXPECT_EQ(manager.dirichlet_dofs().size(), 12u);
  EXPECT_EQ(manager.neumann_faces().size(), 7u);
  EXPECT_EQ(manager.robin_faces().size(), 7u);
}

// physics.boundary_condition.ResolveWithP2BasisIncludesEdgeDofsOnTheBoundary
// -- a 2x2 Quad9 (P2) mesh, Dirichlet on every side: the Dirichlet dof
// count must include the EDGE MIDPOINT dofs on the boundary, not only
// the corner (vertex) dofs -- the bug this test guards against left
// edge dofs on a Dirichlet boundary unconstrained.
//
// A 2x2 structured Quad4 mesh (n=2 elements per axis) has a (n+1) x
// (n+1) = 3x3 = 9 vertex grid; exactly 1 of the 9 is strictly interior
// (the centre), so 8 are on the boundary. The SAME grid has
// n*(n+1) + (n+1)*n = 2*3 + 3*2 = 12 edges (P2's own edge-midpoint
// dof, one per edge); each of the 4 box sides is spanned by n = 2
// element-edge segments, so 4*2 = 8 of the 12 edges are boundary
// edges (the remaining 12 - 8 = 4 are interior, matching the 2
// interior horizontal and 2 interior vertical edges a 3x3 grid has).
// Total Dirichlet dofs with all four sides registered = 8 boundary
// vertices + 8 boundary edge-midpoints = 16.
TEST(BoundaryConditionTest, ResolveWithP2BasisIncludesEdgeDofsOnTheBoundary) {
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(0, 0, 0), Point(2, 2, 0));
  BasisRegistry registry;
  DofMap dof_map(mesh, {scalar_p2("u")}, registry);
  const Basis basis = registry.create(BasisKey{CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 2,
                                                BasisContinuity::cG});
  const Domain domain(2, Shape::Square, 2.0, 2.0, 0.0, false);

  BoundaryConditionManager manager(mesh, domain);
  for (const auto id : {BoundaryId::XMin, BoundaryId::XMax, BoundaryId::YMin, BoundaryId::YMax})
    manager.set(id, BoundaryCondition{BoundaryConditionKind::Dirichlet, [](const Point&, double) { return 0.0; }, 0.0});
  manager.resolve(dof_map, basis);

  EXPECT_EQ(manager.dirichlet_dofs().size(), 16u);
  // Every resolved dof sits on the box boundary: an interior dof resolved
  // by mistake is caught here even if the count is right. The vertex
  // coordinates are 0, 1 and 2 and an edge midpoint is the mean of two of
  // them, so every position is exact.
  for (const auto& d : manager.dirichlet_dofs()) {
    const bool on_boundary = d.position.x == 0.0 || d.position.x == 2.0 || d.position.y == 0.0 || d.position.y == 2.0;
    EXPECT_TRUE(on_boundary) << "dof at (" << d.position.x << "," << d.position.y << ") is not on the boundary";
  }
}

// physics.boundary_condition.UnmatchedConditionRefusesByName -- a
// condition registered on a boundary id that geometrically never occurs
// (a 1-D-only mesh extent given a 3-D id, or here, a domain dimension
// wider than the mesh's own measured extent reaches) is refused rather
// than silently resolving to zero Dirichlet dofs and zero faces with no
// diagnostic.
TEST(BoundaryConditionTest, UnmatchedConditionRefusesByName) {
  // A 2x2 mesh on [0,1]x[0,1] paired with a Domain claiming length_y =
  // 5.0: YMax (y == 5) never occurs on this mesh, so it must refuse.
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(0, 0, 0), Point(1, 1, 0));
  BasisRegistry registry;
  DofMap dof_map(mesh, {scalar_p1("u")}, registry);
  const Basis basis = registry.create(BasisKey{CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1,
                                                BasisContinuity::cG});
  const Domain domain(2, Shape::Square, 1.0, 5.0, 0.0, false);

  BoundaryConditionManager manager(mesh, domain);
  manager.set(BoundaryId::YMax, BoundaryCondition{BoundaryConditionKind::Dirichlet,
                                                   [](const Point&, double) { return 0.0; }, 0.0});
  try {
    manager.resolve(dof_map, basis);
    FAIL() << "a condition on a boundary that never geometrically occurs must be refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("E-PHYSICS-BC-UNMATCHED"), std::string::npos)
        << "refused for another reason: " << e.what();
  }
}

// physics.boundary_condition.UnknownBoundaryIdRefusesByName -- registering
// a ZMin/ZMax condition on a 2-D domain is refused, not silently ignored
// (silently ignoring it would leave that face's whole boundary
// unconstrained with no diagnostic at all).
TEST(BoundaryConditionTest, UnknownBoundaryIdRefusesByName) {
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(0, 0, 0), Point(1, 1, 0));
  const Domain domain(2, Shape::Square, 1.0, 1.0, 0.0, false);
  BoundaryConditionManager manager(mesh, domain);
  try {
    manager.set(BoundaryId::ZMin,
                BoundaryCondition{BoundaryConditionKind::Dirichlet, [](const Point&, double) { return 0.0; }, 0.0});
    FAIL() << "a ZMin condition on a 2-D domain must be refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("E-PHYSICS-BC-UNKNOWN-BOUNDARY"), std::string::npos)
        << "refused for another reason: " << e.what();
  }
}

// physics.boundary_condition.DuplicateConditionOnOneBoundaryRefusesByName
TEST(BoundaryConditionTest, DuplicateConditionOnOneBoundaryRefusesByName) {
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(0, 0, 0), Point(1, 1, 0));
  const Domain domain(2, Shape::Square, 1.0, 1.0, 0.0, false);
  BoundaryConditionManager manager(mesh, domain);
  manager.set(BoundaryId::XMin,
              BoundaryCondition{BoundaryConditionKind::Dirichlet, [](const Point&, double) { return 0.0; }, 0.0});
  try {
    manager.set(BoundaryId::XMin,
                BoundaryCondition{BoundaryConditionKind::Neumann, [](const Point&, double) { return 0.0; }, 0.0});
    FAIL() << "a second condition on XMin must be refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("E-PHYSICS-BC-DUPLICATE"), std::string::npos)
        << "refused for another reason: " << e.what();
  }
}

// physics.boundary_condition.NonPositiveRobinCoefficientRefusesByName --
// coefficient == 0 (a Neumann condition in disguise) and coefficient < 0
// (which would make the boundary term indefinite, not merely weaker)
// both refuse.
TEST(BoundaryConditionTest, NonPositiveRobinCoefficientRefusesByName) {
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(0, 0, 0), Point(1, 1, 0));
  const Domain domain(2, Shape::Square, 1.0, 1.0, 0.0, false);
  for (const double bad_coefficient : {0.0, -1.0}) {
    BoundaryConditionManager manager(mesh, domain);
    try {
      manager.set(BoundaryId::XMin, BoundaryCondition{BoundaryConditionKind::Robin,
                                                       [](const Point&, double) { return 0.0; }, bad_coefficient});
      FAIL() << "a non-positive Robin coefficient (" << bad_coefficient << ") must be refused";
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find("E-PHYSICS-BC-ROBIN-COEFFICIENT"), std::string::npos)
          << "refused for another reason: " << e.what();
    }
  }
}

// physics.boundary_condition.NoConditionsRegisteredRefusesByName -- calling
// resolve() with nothing ever set() is refused rather than silently
// returning three empty lists (which a caller could easily mistake for
// "this problem has no boundary conditions", a modelling error the
// manager can and should catch instead of propagating silently).
TEST(BoundaryConditionTest, NoConditionsRegisteredRefusesByName) {
  Mesh mesh(2);
  mesh.generate_2d_box(2, 2, Point(0, 0, 0), Point(1, 1, 0));
  BasisRegistry registry;
  DofMap dof_map(mesh, {scalar_p1("u")}, registry);
  const Basis basis = registry.create(BasisKey{CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1,
                                                BasisContinuity::cG});
  const Domain domain(2, Shape::Square, 1.0, 1.0, 0.0, false);
  BoundaryConditionManager manager(mesh, domain);
  try {
    manager.resolve(dof_map, basis);
    FAIL() << "resolve() with no registered condition must be refused";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("E-PHYSICS-BC-NO-CONDITIONS"), std::string::npos)
        << "refused for another reason: " << e.what();
  }
}

}  // namespace
