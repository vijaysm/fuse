// physics.diffusion_reaction_steady_order -- convergence order of the steady
// nonlinear diffusion-reaction problem for bilinear and biquadratic
// quadrilaterals and trilinear hexahedra. The problem, the band and the
// bound on the algebraic error are derived in
// diffusion_reaction_steady_order.hpp.

#include "diffusion_reaction_steady_order.hpp"

namespace {

using steady_order_study::expect_order;

TEST(DiffusionReactionSteadyOrderTest, BilinearQuadrilateralsAreSecondOrder) {
  expect_order("Q1, 2-D", 2, 1, {4, 8, 16});
}

TEST(DiffusionReactionSteadyOrderTest, BiquadraticQuadrilateralsAreThirdOrder) {
  expect_order("Q2, 2-D", 2, 2, {4, 8, 16});
}

TEST(DiffusionReactionSteadyOrderTest, TrilinearHexahedraAreSecondOrder) {
  expect_order("Q1, 3-D", 3, 1, {2, 4, 8});
}

}  // namespace
