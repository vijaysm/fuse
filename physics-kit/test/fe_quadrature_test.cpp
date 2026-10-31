// physics.fe_quadrature -- ported from the quadrature-exactness portions of
// KARMA's fe_test.C/fe_face_test.C (item 6's pyramid-quadrature section is
// explicitly NOT ported -- no pyramid variant exists this wave, see
// quadrature.hpp's header comment). Verifies exact integration of
// monomials up to the rule's declared degree, with a round-off TOLERANCE
// DERIVED from the rule's own data (Quadrature::round_off_bound), per the
// task's standing rule that a tolerance must be derived, not chosen after
// a failure.
//
// REFERENCE VALUES. The analytic integral of a monomial over each
// reference cell is a closed form, not re-derived numerically:
//   edge [0,1]:        integral x^a dx = 1/(a+1)
//   quad/hex [0,1]^d:  product of the 1-D edge formula over each axis
//   triangle (Dirichlet/Beta-function integral over the unit 2-simplex):
//       integral x^a y^b dA = a! b! / (a+b+2)!
//   tetrahedron (same identity over the unit 3-simplex):
//       integral x^a y^b z^c dV = a! b! c! / (a+b+c+3)!

#include "fuse_physics/fe/quadrature.hpp"

#include <cmath>
#include <vector>

#include <cstdio>

#include <gtest/gtest.h>

namespace {

using fuse::physics::Point;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::Quadrature;

double factorial(int n) {
  double result = 1.0;
  for (int i = 2; i <= n; ++i) result *= i;
  return result;
}

double monomial(const Point& point, unsigned a, unsigned b, unsigned c) {
  return std::pow(point.x, static_cast<int>(a)) * std::pow(point.y, static_cast<int>(b)) *
         std::pow(point.z, static_cast<int>(c));
}

double analytic_integral(CellFamily family, unsigned a, unsigned b, unsigned c) {
  switch (family) {
    case CellFamily::Edge:
      return 1.0 / (a + 1);
    case CellFamily::Quadrilateral:
      return 1.0 / (a + 1) / (b + 1);
    case CellFamily::Hexahedron:
      return 1.0 / (a + 1) / (b + 1) / (c + 1);
    case CellFamily::Triangle:
      return factorial(static_cast<int>(a)) * factorial(static_cast<int>(b)) /
             factorial(static_cast<int>(a + b + 2));
    case CellFamily::Tetrahedron:
      return factorial(static_cast<int>(a)) * factorial(static_cast<int>(b)) * factorial(static_cast<int>(c)) /
             factorial(static_cast<int>(a + b + c + 3));
  }
  return 0.0;
}

/// For a tensor-product cell (edge/quad/hex), exactness is required
/// independently on each axis up to `degree` (total monomial degree can
/// reach dim*degree and still integrate exactly, since each 1-D factor is
/// separately exact to `degree`). For a simplex cell (triangle/tet), only
/// the TOTAL degree a+b+c <= degree is guaranteed exact -- the collapsed-
/// coordinate construction (quadrature.cpp) is built for that, not for
/// per-axis degree independently.
bool is_tensor_product(CellFamily family) {
  return family == CellFamily::Edge || family == CellFamily::Quadrilateral || family == CellFamily::Hexahedron;
}

// The largest |error| / bound ratio seen across every monomial of every
// rule in this file; printed by the last test so the derivation's slack is
// a measured number (Quadrature::round_off_bound's header comment).
double g_max_error_ratio = 0.0;

void check_one_monomial(const Quadrature& rule, CellFamily family, int degree, unsigned a, unsigned b, unsigned c) {
  double numeric = 0.0;
  double sum_abs_terms = 0.0;
  for (std::size_t q = 0; q < rule.n_points(); ++q) {
    const double term = rule.weights()[q] * monomial(rule.points()[q], a, b, c);
    numeric += term;
    sum_abs_terms += std::abs(term);
  }
  const double expected = analytic_integral(family, a, b, c);
  const double measure = analytic_integral(family, 0, 0, 0);
  const double tolerance = rule.round_off_bound(sum_abs_terms, a + b + c, measure);
  if (tolerance > 0.0) g_max_error_ratio = std::max(g_max_error_ratio, std::abs(numeric - expected) / tolerance);
  EXPECT_NEAR(numeric, expected, tolerance)
      << "family " << static_cast<int>(family) << " degree " << degree << " monomial x^" << a << " y^" << b << " z^"
      << c << " (bound " << tolerance << ", sweeps " << rule.generation_sweeps() << ")";
}

void check_exact_to_degree(CellFamily family, int degree) {
  const Quadrature rule(family, degree);
  const unsigned dim = rule.dimension();
  const unsigned d = static_cast<unsigned>(degree);
  if (is_tensor_product(family)) {
    for (unsigned a = 0; a <= d; ++a)
      for (unsigned b = 0; b <= (dim >= 2 ? d : 0); ++b)
        for (unsigned c = 0; c <= (dim >= 3 ? d : 0); ++c) check_one_monomial(rule, family, degree, a, b, c);
    return;
  }
  for (unsigned a = 0; a <= d; ++a)
    for (unsigned b = 0; a + b <= d; ++b) {
      if (dim == 2) {
        check_one_monomial(rule, family, degree, a, b, 0);
        continue;
      }
      for (unsigned c = 0; a + b + c <= d; ++c) check_one_monomial(rule, family, degree, a, b, c);
    }
}

TEST(QuadratureTest, EdgeExactToDeclaredDegree) {
  for (int degree = 0; degree <= 7; ++degree) check_exact_to_degree(CellFamily::Edge, degree);
}
TEST(QuadratureTest, QuadrilateralExactToDeclaredDegree) {
  for (int degree = 0; degree <= 6; ++degree) check_exact_to_degree(CellFamily::Quadrilateral, degree);
}
TEST(QuadratureTest, HexahedronExactToDeclaredDegree) {
  for (int degree = 0; degree <= 5; ++degree) check_exact_to_degree(CellFamily::Hexahedron, degree);
}
TEST(QuadratureTest, TriangleExactToDeclaredDegree) {
  for (int degree = 0; degree <= 6; ++degree) check_exact_to_degree(CellFamily::Triangle, degree);
}
TEST(QuadratureTest, TetrahedronExactToDeclaredDegree) {
  for (int degree = 0; degree <= 5; ++degree) check_exact_to_degree(CellFamily::Tetrahedron, degree);
}

// physics.fe_quadrature.GaussLegendreMatchesClassicalRecurrence -- a
// smoke check that the degree-0 edge rule (one point) is the midpoint
// rule, the simplest member of the Gauss-Legendre family, exactly.
TEST(QuadratureTest, DegreeZeroEdgeRuleIsMidpoint) {
  const Quadrature rule(CellFamily::Edge, 0);
  ASSERT_EQ(rule.n_points(), 1u);
  EXPECT_NEAR(rule.points()[0].x, 0.5, 1.0e-14);
  EXPECT_NEAR(rule.weights()[0], 1.0, 1.0e-14);
}

// physics.fe_quadrature.RefusesUnsupportedFamily -- pyramid/prism are not
// implemented this wave (quadrature.hpp's own scope note); there is no
// CellFamily enumerator for them (reference_topology.hpp only names the
// five implemented families), so the refusal surface is the out-of-range
// degree check instead -- verified here that it actually fires.
// Runs last (gtest runs a file's tests in definition order): the slack of the
// derived bound as a measured number. A ratio above 1 anywhere would already
// have failed the test that saw it; a ratio near 1 would mean the derivation
// has no margin and must be re-examined before any rule is added.
TEST(QuadratureTest, ZZZ_DerivedBoundHasMeasuredMargin) {
  ASSERT_GT(g_max_error_ratio, 0.0);
  EXPECT_LT(g_max_error_ratio, 1.0);
  std::printf("quadrature: largest |error| / derived bound over all monomials = %.3e\n", g_max_error_ratio);
}

TEST(QuadratureTest, RefusesOutOfRangeDegree) {
  EXPECT_THROW(Quadrature(CellFamily::Edge, -1), std::invalid_argument);
  EXPECT_THROW(Quadrature(CellFamily::Edge, 65), std::invalid_argument);
}

}  // namespace
