// physics.diffusion_reaction_steady_order_fine -- the trilinear-hexahedron
// order study of physics.diffusion_reaction_steady_order on meshes of 4, 8
// and 16 cells per side. One level finer, the two observed orders are
// closer to 2 and to each other, so the band (twice their difference) is
// about five times narrower: 2.012 and 2.003 with a band of 0.018, against
// 2.058 and 2.012 with a band of 0.092 on 2, 4 and 8. The 16^3 level makes
// this a slow row (cmake/FuseSlowTests.cmake); the derivations are in
// diffusion_reaction_steady_order.hpp.

#include "diffusion_reaction_steady_order.hpp"

namespace {

TEST(DiffusionReactionSteadyOrderFineTest, TrilinearHexahedraAreSecondOrderOnTheFinerLadder) {
  steady_order_study::expect_order("Q1, 3-D, meshes 4 8 16", 3, 1, {4, 8, 16});
}

}  // namespace
