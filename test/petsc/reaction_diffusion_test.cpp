// S7.4 (owner decision 2026-10-21, Option C). The structural analogue of
// test/runtime/petsc_reaction_diffusion_component_test.cpp for
// NeumannHalfPetsc (examples/petsc_reaction_diffusion/
// reaction_diffusion_component_petsc.hpp): gated FUSE_ENABLE_PETSC,
// BUILT and ctest-registered, never RUN in this sandbox
// (NeumannHalfPetsc's own constructor calls
// fuse::petsc::ensure_initialized(), which SEGFAULTS on MPI_Init() here --
// KNOWN_ISSUES.md D32, the same already-documented, already-measured
// barrier every other petsc.* row carries). A host run is what measures
// whether these cases pass; see docs/user/petsc-reaction-diffusion.md.

#include <gtest/gtest.h>

#include "reaction_diffusion_component.hpp"
#include "reaction_diffusion_component_petsc.hpp"

using petsc_reaction_diffusion::NeumannHalfNative;
using petsc_reaction_diffusion::NeumannHalfPetsc;
using petsc_reaction_diffusion::PhysicsParams;
using petsc_reaction_diffusion::right_newton_solve;
using petsc_reaction_diffusion::right_newton_solve_petsc;

TEST(PetscReactionDiffusion, PetscInnerSolveAgreesWithNativeAtOnePoint) {
  // Both substrates solve the IDENTICAL discrete system (reaction_
  // diffusion_physics.hpp's own right_residual_jacobian) by a DIRECT
  // linear solve (Thomas algorithm vs KSP+PCLU) -- round-off-scale
  // agreement expected, the same derivation main.cpp's own
  // kSubstrateAgreementBound uses.
  PhysicsParams p;
  const int m = 4;
  const double dx = p.L_right / (m + 1);
  const double dt = 0.01, t_new = 0.01, q_I = 0.35;
  std::vector<double> u_old(static_cast<std::size_t>(m));
  for (int i = 0; i < m; ++i) {
    u_old[static_cast<std::size_t>(i)] =
        petsc_reaction_diffusion::manufactured_u(p.L_left + (i + 1) * dx, 0.0, p);
  }

  const auto native = right_newton_solve(u_old, q_I, dt, dx, t_new, p);
  const auto petsc_result = right_newton_solve_petsc(u_old, q_I, dt, dx, t_new, p);
  ASSERT_TRUE(petsc_result.ok()) << petsc_result.error().message;

  constexpr double kBound = 100.0 * 1e-8;  // main.cpp's own kSubstrateAgreementBound
  for (int i = 0; i < m; ++i) {
    EXPECT_NEAR(native.u[static_cast<std::size_t>(i)],
               petsc_result.value().u[static_cast<std::size_t>(i)], kBound);
  }
  EXPECT_NEAR(native.u_interface, petsc_result.value().u_interface, kBound);
  EXPECT_NEAR(native.dG_dq_I, petsc_result.value().dG_dq_I, kBound);
}

TEST(PetscReactionDiffusion, ComponentConstructsAndSeedsProfile) {
  PhysicsParams p;
  NeumannHalfPetsc comp("neumann_petsc", /*m=*/4, p);
  comp.seed_initial_condition();
  EXPECT_EQ(comp.profile().size(), 4u);
  EXPECT_EQ(comp.name(), "neumann_petsc");
}
