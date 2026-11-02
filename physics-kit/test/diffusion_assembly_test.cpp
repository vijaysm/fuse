// physics.diffusion_assembly -- adapted from the core bilinear-form
// content of KARMA's test/diffusion_assembly_test.C. That file's own
// elaborate BC-manager machinery (Dirichlet/Neumann/Robin condition
// discovery, replicated/asymmetric boundary conditions -- a full
// physics/diffusion-reaction/ example application, not numerics/'s own
// core scope) is OUT OF SCOPE; this
// file ports only the mathematical stiffness-operator content: the
// standard FE bilinear form integral(grad(phi_i) . grad(phi_j)) dV
// (generic_operators.hpp's own element_stiffness_matrix) and its
// algebraic properties.

#include "fuse_physics/fe/generic_operators.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::element_stiffness_matrix;
using fuse::physics::fe::FEBase;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::spatial::Mesh;

// physics.diffusion_assembly.ReferenceTet4StiffnessMatchesClosedForm --
// on the reference Tet4 (vertices at the origin and the three unit axis
// points, volume = 1/6), the four P1 shape-function GRADIENTS are the
// constant vectors grad(N0) = (-1,-1,-1), grad(N1) = (1,0,0),
// grad(N2) = (0,1,0), grad(N3) = (0,0,1) -- a textbook closed form
// (no KARMA source has a diffusion-operator
// equivalent to port from, generic_operators.hpp's own header comment).
// stiffness(i,j) = volume * dot(grad_i, grad_j), computed here by hand
// and compared to element_stiffness_matrix's own output. Quadrature
// degree 0 is already EXACT (a constant integrand on an affine
// element), so the only floating-point error possible is a single
// multiply-accumulate per entry -- the tolerance is a FEW ULPS, derived
// (not empirical) from that fact.
TEST(DiffusionAssemblyTest, ReferenceTet4StiffnessMatchesClosedForm) {
  Mesh mesh(3);
  mesh.add_vertex(Point(0, 0, 0));
  mesh.add_vertex(Point(1, 0, 0));
  mesh.add_vertex(Point(0, 1, 0));
  mesh.add_vertex(Point(0, 0, 1));
  mesh.add_element(GeometryVariant::Tet4, {1, 2, 3, 4});

  BasisRegistry registry;
  const auto basis = registry.create({CellFamily::Tetrahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Tet4, basis, /*quadrature_degree=*/0);
  const auto stiffness = element_stiffness_matrix(fe, 1);
  ASSERT_EQ(stiffness.size(), 16u);

  const double volume = 1.0 / 6.0;
  const std::array<std::array<double, 3>, 4> gradients{{{-1, -1, -1}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  const double eps = std::numeric_limits<double>::epsilon();
  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      double dot_ij = 0.0;
      for (unsigned c = 0; c < 3; ++c) dot_ij += gradients[i][c] * gradients[j][c];
      const double expected = volume * dot_ij;
      const double tolerance = 8.0 * eps * std::max(1.0, std::abs(expected));
      EXPECT_NEAR(stiffness.at(i * 4 + j), expected, tolerance) << "i=" << i << " j=" << j;
    }
  }
}

// physics.diffusion_assembly.StiffnessMatrixIsExactlySymmetric -- dot()
// is commutative bit-for-bit under IEEE 754 (multiplication and
// addition of the SAME two operands in the same order produce the SAME
// bits, dphi[i][q] . dphi[j][q] == dphi[j][q] . dphi[i][q] by
// construction: swapping i and j only swaps which stored gradient is
// read first into an otherwise identical a*x+b*y+c*z expression), so no
// rounding tolerance is needed here at all -- EXACT equality, on a
// Hex8 element (a shape whose gradients are NOT constant, unlike the
// tet case above, so this check is the one that would catch an
// accidental i/j asymmetry the closed-form check above could not).
TEST(DiffusionAssemblyTest, StiffnessMatrixIsExactlySymmetric) {
  Mesh mesh(3);
  mesh.generate_box(1, 1, 1, Point(0, 0, 0), Point(1, 1, 1));
  BasisRegistry registry;
  const auto basis = registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Hex8, basis, /*quadrature_degree=*/2);
  const auto stiffness = element_stiffness_matrix(fe, 1);
  ASSERT_EQ(stiffness.size(), 64u);
  for (unsigned i = 0; i < 8; ++i)
    for (unsigned j = 0; j < 8; ++j) EXPECT_EQ(stiffness.at(i * 8 + j), stiffness.at(j * 8 + i));
}

// physics.diffusion_assembly.StiffnessRowsSumToZero -- a cG Lagrange
// basis is a partition of unity (sum_i phi_i = 1 everywhere), so
// grad(sum_i phi_i) = 0, and sum_j stiffness(i,j) = integral(grad(phi_i)
// . grad(sum_j phi_j)) dV = integral(grad(phi_i) . 0) dV = 0 for every
// row i. Tolerance: a Higham summation bound over n_basis terms, each
// bounded by the entry magnitude the closed-form test above already
// established is O(1) for this element.
TEST(DiffusionAssemblyTest, StiffnessRowsSumToZero) {
  Mesh mesh(3);
  mesh.generate_box(1, 1, 1, Point(0, 0, 0), Point(1, 1, 1));
  BasisRegistry registry;
  const auto basis = registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Hex8, basis, /*quadrature_degree=*/2);
  const auto stiffness = element_stiffness_matrix(fe, 1);

  const double eps = std::numeric_limits<double>::epsilon();
  double max_abs_entry = 0.0;
  for (const double v : stiffness) max_abs_entry = std::max(max_abs_entry, std::abs(v));
  const double tolerance = 8.0 * eps * max_abs_entry;  // Higham (2002) Thm 8.1, n=8 terms
  for (unsigned i = 0; i < 8; ++i) {
    double row_sum = 0.0;
    for (unsigned j = 0; j < 8; ++j) row_sum += stiffness.at(i * 8 + j);
    EXPECT_NEAR(row_sum, 0.0, tolerance) << "row " << i;
  }
}

}  // namespace
