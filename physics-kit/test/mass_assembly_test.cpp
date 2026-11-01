// physics.mass_assembly -- adapted from KARMA's test/mass_assembly_test.C.
// See generic_operators.hpp's own header comment: element_mass_matrix is
// the port of KARMA's include/physics/common/generic_mass_operator.h's
// own element_mass_matrix, the right abstraction level
// test/mass_assembly_test.C itself exercises (not karma_algebraic_
// system.h directly). The golden-reference-file / serial-vs-partitioned
// comparison infrastructure KARMA's own test file also carries
// (MassReference, write_partitioned_mass_reference) is MPI/partition
// test infrastructure, out of scope here -- physics_assembly_mpi_test.cpp
// (item 7, build-only in this sandbox) is where that comparison belongs.

#include "fuse_physics/fe/generic_operators.hpp"

#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "fuse_physics/fe/algebraic_system.hpp"
#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::AlgebraicSystem;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::element_mass_matrix;
using fuse::physics::fe::FEBase;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::Mesh;
using fuse::linalg::Triplet;

// physics.mass_assembly.P1SimplexElementMassMatrixMatchesAnalyticFormula
// -- KARMA's own test/mass_assembly_test.C::check_simplex_mass formula,
// cited (not re-derived): for a P1 Lagrange basis on a `dimension`-
// simplex of measure `V`, with `size = dimension + 1` vertices,
// mass(i, j) = V / (size * (size + 1)) * (i == j ? 2 : 1). Checked on
// the reference Tet4 (vertices at the origin and the three unit axis
// points, volume = 1/6, size = 4): mass(i,i) = (1/6)/20*2 = 1/60,
// mass(i,j (i!=j)) = (1/6)/20 = 1/120. Quadrature degree 2 is EXACT for
// phi_i*phi_j (each a degree-1 polynomial, product degree 2) on this
// AFFINE element (constant Jacobian) -- design_notes states this
// derivation; the tolerance below is a Higham-style floating-point
// summation bound for a sum of `n_quadrature_points` terms, each itself
// one single-precision-safe double product, not an independent
// empirical constant.
TEST(MassAssemblyTest, P1SimplexElementMassMatrixMatchesAnalyticFormula) {
  Mesh mesh(3);
  mesh.add_vertex(Point(0, 0, 0));
  mesh.add_vertex(Point(1, 0, 0));
  mesh.add_vertex(Point(0, 1, 0));
  mesh.add_vertex(Point(0, 0, 1));
  mesh.add_element(GeometryVariant::Tet4, {1, 2, 3, 4});

  BasisRegistry registry;
  const auto basis = registry.create({CellFamily::Tetrahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Tet4, basis, /*quadrature_degree=*/2);
  const auto mass = element_mass_matrix(fe, 1);
  ASSERT_EQ(mass.size(), 16u);

  const double volume = 1.0 / 6.0;
  constexpr unsigned size = 4;  // dimension + 1
  const double diagonal = volume / (size * (size + 1)) * 2.0;
  const double off_diagonal = volume / (size * (size + 1)) * 1.0;
  // Higham (2002) Thm 8.1-style bound for summing n products of two
  // doubles each exactly representable to working precision here
  // (quadrature weights and shape values at a degree-2-exact rule):
  // |computed - exact| <= n * eps * |exact|, eps = 2^-52. n here is the
  // number of quadrature points a degree-2 Tet4 rule uses (<= 8,
  // quadrature.cpp's own table) -- a generous n=16 is used for margin.
  const double eps = std::numeric_limits<double>::epsilon();
  const double tolerance = 16.0 * eps * diagonal;
  for (unsigned i = 0; i < 4; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      const double expected = i == j ? diagonal : off_diagonal;
      EXPECT_NEAR(mass.at(i * 4 + j), expected, tolerance) << "i=" << i << " j=" << j;
    }
  }
}

// physics.mass_assembly.Quad4UnitBoxElementMassMatrixMatchesKnownFormula
// -- the SAME Quad4 consistent mass matrix
// test/transient_reaction_diffusion_assembly_test.C's own KARMA source
// hand-codes (mass_data[4][4] = {{4,2,1,2},{2,4,2,1},{1,2,4,2},
// {2,1,2,4}}/36 for a unit box) -- cited there, re-derived here as a
// cross-check from mass_assembly_test.C's own abstraction instead of
// copied verbatim, confirming both KARMA sources agree. Quadrature
// degree 2 is EXACT for a bilinear phi_i*phi_j (degree <= 2 in each
// reference coordinate) on an AFFINE (here: AXIS-ALIGNED RECTANGLE)
// element.
TEST(MassAssemblyTest, Quad4UnitBoxElementMassMatrixMatchesKnownFormula) {
  Mesh mesh(2);
  mesh.generate_2d_box(1, 1, Point(0, 0, 0), Point(1, 1, 0));
  BasisRegistry registry;
  const auto basis =
      registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);
  const auto mass = element_mass_matrix(fe, 1);
  ASSERT_EQ(mass.size(), 16u);

  const double unit = 1.0 / 36.0;
  const double expected_table[4][4] = {
      {4 * unit, 2 * unit, 1 * unit, 2 * unit},
      {2 * unit, 4 * unit, 2 * unit, 1 * unit},
      {1 * unit, 2 * unit, 4 * unit, 2 * unit},
      {2 * unit, 1 * unit, 2 * unit, 4 * unit},
  };
  const double eps = std::numeric_limits<double>::epsilon();
  const double tolerance = 16.0 * eps * (4 * unit);
  for (unsigned i = 0; i < 4; ++i)
    for (unsigned j = 0; j < 4; ++j) EXPECT_NEAR(mass.at(i * 4 + j), expected_table[i][j], tolerance);
}

// physics.mass_assembly.GlobalAssembledMassMatrixEntrySumEqualsDomainVolume
// -- the SAME invariant item 7's MPI test checks (build-only there),
// exercised here SERIALLY: sum_i sum_j M(i,j) = integral((sum_i
// phi_i)^2) dV = integral(1) dV = domain volume, since a cG Lagrange
// basis is a partition of unity. Tolerance: a Higham summation bound
// over n_elements * n_basis^2 terms, each itself bounded by the element
// test's own per-entry tolerance above.
TEST(MassAssemblyTest, GlobalAssembledMassMatrixEntrySumEqualsDomainVolume) {
  Mesh mesh(3);
  mesh.generate_box(2, 2, 2, Point(0, 0, 0), Point(2, 1, 1));
  BasisRegistry registry;
  Variable u{"u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1};
  DofMap dof_map(mesh, {u}, registry);
  AlgebraicSystem system(mesh, dof_map);

  const auto basis = registry.create({CellFamily::Hexahedron, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Hex8, basis, /*quadrature_degree=*/2);

  const auto matrix = system.assemble_matrix(
      [&](fuse::physics::spatial::EntityHandle element, const std::vector<std::int64_t>& indices,
          std::vector<Triplet>& out) {
        const auto local = element_mass_matrix(fe, element);
        const std::size_t n = indices.size();
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) out.push_back(Triplet{indices[i], indices[j], local[i * n + j]});
      });
  ASSERT_TRUE(matrix.ok()) << matrix.error().message;

  // Sum every entry by applying the matrix to the all-ones vector and
  // reducing the result -- avoids needing a private entry iterator on
  // DistSparseMatrix (apply() + DistVector::dot() are both already
  // part of its tested public surface).
  auto ones = system.new_vector();
  for (std::size_t i = 0; i < ones.owned_size(); ++i) ones.data()[i] = 1.0;
  auto result = system.new_vector();
  const auto apply_result = matrix.value().apply(ones, result);
  ASSERT_TRUE(apply_result.ok()) << apply_result.error().message;
  double total = 0.0;
  for (std::size_t i = 0; i < result.owned_size(); ++i) total += result.data()[i];

  const double domain_volume = 2.0 * 1.0 * 1.0;
  const std::size_t n_terms = mesh.n_elements() * 8 * 8;
  const double eps = std::numeric_limits<double>::epsilon();
  const double tolerance = static_cast<double>(n_terms) * eps * domain_volume;
  EXPECT_NEAR(total, domain_volume, tolerance);
}

}  // namespace
