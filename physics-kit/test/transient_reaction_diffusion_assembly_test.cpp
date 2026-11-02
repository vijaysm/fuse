// physics.transient_reaction_diffusion_assembly -- adapted from KARMA's
// test/transient_reaction_diffusion_assembly_test.C. See generic_
// operators.hpp's own header comment for why this port's reaction
// model is quadratic, not KARMA's own affine one (item D's own
// instruction: the finite-difference Jacobian check's tolerance must
// be derived from the reaction term's SECOND-derivative bound, which
// is identically zero for an affine model).

#include "fuse_physics/fe/generic_operators.hpp"

#include <cmath>
#include <limits>

#include <cstdio>
#include <algorithm>
#include <gtest/gtest.h>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace {

using fuse::physics::Point;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::element_transient_reaction_diffusion;
using fuse::physics::fe::FEBase;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::jacobian_finite_difference_tolerance;
using fuse::physics::fe::reaction;
using fuse::physics::fe::TransientReactionDiffusionModel;
using fuse::physics::spatial::Mesh;

// physics.transient_reaction_diffusion_assembly.
// AnalyticJacobianMatchesForwardDifferenceWithinDerivedTolerance -- a
// unit Quad4 element (the SAME element KARMA's own test uses), a
// quadratic reaction model (alpha != 0, so R'' = 2*alpha != 0 -- the
// non-affine case item D's own instruction requires to exercise a
// genuine truncation-term derivation), Crank-Nicolson residual and
// Jacobian at an arbitrary (non-uniform, so the nodal reaction terms
// differ node to node) state and old state. Every Jacobian entry
// (i, k) is checked against a FORWARD difference of residual_i with
// respect to state_k, at the step jacobian_finite_difference_
// tolerance's own DERIVATION picks (generic_operators.hpp's own header
// comment: the step balancing the h/2*|f''| truncation term against
// the eps*|f|/h round-off term) -- not KARMA's own underived 1e-6
// (transient_reaction_diffusion_assembly_test.C:101), replaced here by
// a derived bound as item D's own instruction requires.
TEST(TransientReactionDiffusionAssemblyTest, AnalyticJacobianMatchesForwardDifferenceWithinDerivedTolerance) {
  Mesh mesh(2);
  mesh.generate_2d_box(1, 1, Point(0, 0, 0), Point(1, 1, 0));
  BasisRegistry registry;
  const auto basis =
      registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);

  const TransientReactionDiffusionModel model{/*alpha=*/0.7, /*beta=*/0.3, /*source=*/0.1, /*coupled=*/0.4};
  const double h = 0.05;
  const std::vector<double> state{0.10, 0.35, -0.20, 0.50};
  const std::vector<double> state_old{0.05, 0.30, -0.25, 0.40};

  const auto result = element_transient_reaction_diffusion(fe, 1, model, h, state, state_old);
  ASSERT_EQ(result.residual.size(), 4u);
  ASSERT_EQ(result.jacobian.size(), 16u);

  // Every input of the derived bound is measured or exact (generic_operators.hpp):
  // |M_ik| from an independent element_mass_matrix call (not from the Jacobian
  // being checked), the row's absolute-term sum from the kernel, n = 4 dofs,
  // n_q = 4 points (degree 2 on a Quad4), state_k and J_ik for the step-rounding
  // term. The worst |error| / tolerance ratio over the 16 entries is printed as
  // the measured margin (lead, 2026-11-01; the round-2 version carried an
  // adjustable factor raised from 1 to 16 after a failure, which this replaces).
  FEBase fe2(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);
  const auto mass_independent = fuse::physics::fe::element_mass_matrix(fe2, 1);
  ASSERT_EQ(fe.n_quadrature_points(), 4u);
  double worst_ratio = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t k = 0; k < 4; ++k) {
      const auto check = jacobian_finite_difference_tolerance(
          model, mass_independent.at(i * 4 + k), result.residual_abs_terms[i], /*n_dofs=*/4,
          /*n_quadrature_points=*/fe.n_quadrature_points(), state[k], result.jacobian[i * 4 + k]);
      std::vector<double> perturbed_state = state;
      perturbed_state[k] += check.step;
      const auto perturbed = element_transient_reaction_diffusion(fe, 1, model, h, perturbed_state, state_old);
      const double forward_difference = (perturbed.residual[i] - result.residual[i]) / check.step;
      const double error = std::abs(result.jacobian[i * 4 + k] - forward_difference);
      worst_ratio = std::max(worst_ratio, error / check.tolerance);
      EXPECT_LE(error, check.tolerance)
          << "entry (" << i << "," << k << "), step=" << check.step << " error=" << error
          << " tolerance=" << check.tolerance;
    }
  }
  std::printf("[FD Jacobian, derived bound] worst |error|/tolerance over 16 entries = %.3e\n", worst_ratio);
}

// physics.transient_reaction_diffusion_assembly.
// AffineSpecialCaseReproducesKarmaOwnClosedForm -- alpha == 0 (no
// quadratic term) recovers KARMA's own affine model exactly; the
// closed form round-1's own analysis derived (jacobian = (1/h -
// alpha'/2)*mass, alpha' here the AFFINE coefficient, which this
// model represents as `beta*coupled + source` held fixed and a
// *linear* term folded into `alpha` would need -- here alpha=0 means
// R(u) = beta*coupled + source, a CONSTANT, so R'(u) = 0 and the
// Jacobian reduces to EXACTLY (1/h)*M + 0.5*K, independent of state
// entirely) -- checked to EXACT equality (no FD needed: R'' = 0 here
// too, so this case does not exercise item D's own derivation at all,
// which is why the test above uses alpha != 0 instead).
TEST(TransientReactionDiffusionAssemblyTest, AffineSpecialCaseReproducesKarmaOwnClosedForm) {
  Mesh mesh(2);
  mesh.generate_2d_box(1, 1, Point(0, 0, 0), Point(1, 1, 0));
  BasisRegistry registry;
  const auto basis =
      registry.create({CellFamily::Quadrilateral, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG});
  FEBase fe(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);

  const TransientReactionDiffusionModel model{/*alpha=*/0.0, /*beta=*/0.3, /*source=*/0.1, /*coupled=*/0.4};
  const double h = 0.1;
  const std::vector<double> state{0.1, 0.2, 0.3, 0.4};
  const std::vector<double> state_old{0.0, 0.1, 0.2, 0.3};

  const auto result = element_transient_reaction_diffusion(fe, 1, model, h, state, state_old);
  FEBase fe2(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);
  const auto mass = fuse::physics::fe::element_mass_matrix(fe2, 1);
  FEBase fe3(mesh, GeometryVariant::Quad4, basis, /*quadrature_degree=*/2);
  const auto stiffness = fuse::physics::fe::element_stiffness_matrix(fe3, 1);

  const double eps = std::numeric_limits<double>::epsilon();
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t k = 0; k < 4; ++k) {
      const double expected = mass.at(i * 4 + k) / h + 0.5 * stiffness.at(i * 4 + k);
      EXPECT_NEAR(result.jacobian.at(i * 4 + k), expected, 8.0 * eps * std::max(1.0, std::abs(expected)));
    }
  }
}

}  // namespace
