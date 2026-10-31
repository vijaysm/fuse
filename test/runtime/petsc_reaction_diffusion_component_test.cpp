// Direct, component-level
// tests of DirichletHalf/NeumannHalfNative's own
// fuse::ImplicitComponent/LinearizationToken wiring (reaction_diffusion_
// component.hpp) -- NOT through a full CouplingGraph/Runner (that is
// examples/petsc_reaction_diffusion/main.cpp's own job, exercised as
// example.petsc_reaction_diffusion); this file isolates the component's
// own contract: evaluate_residual's purity, linearization()'s token
// shape and signature, and save_state()/restore_state()'s round trip.
// Tier-0, native only (NeumannHalfPetsc's own structural analogue is
// test/petsc/reaction_diffusion_test.cpp, gated FUSE_ENABLE_PETSC,
// build-only in this sandbox).

#include <gtest/gtest.h>

#include <memory>

#include "reaction_diffusion_component.hpp"

using petsc_reaction_diffusion::DirichletHalf;
using petsc_reaction_diffusion::NeumannHalfNative;
using petsc_reaction_diffusion::PhysicsParams;

namespace {
// dt = 1/100 = 0.01, matching main.cpp's own kPeriodDen -- Rational, not
// double, per StepContext's own contract (component.hpp).
fuse::StepContext make_ctx() {
  fuse::StepContext ctx;
  ctx.t_begin = fuse::Rational{0};
  ctx.t_end = fuse::Rational{1, 100};
  ctx.step = 0;
  ctx.iteration = 0;
  return ctx;
}
}  // namespace

TEST(PetscReactionDiffusionComponent, DirichletHalfEvaluateResidualIsPure) {
  // component.hpp's own contract: evaluate_residual must not mutate
  // anything save_state() would capture -- a save/eval/save bracket must
  // produce two identical checkpoints (test/runtime/implicit_component_
  // test.cpp's own convention for DirichletSlab/NeumannSlab, applied here).
  PhysicsParams p;
  DirichletHalf comp("dirichlet", /*m=*/4, p);
  comp.seed_initial_condition();
  comp.save_state();
  const std::vector<double> before = comp.profile();

  std::array<double, 1> in{1.1};
  std::array<double, 1> out{0.0};
  const fuse::StepContext ctx = make_ctx();
  const auto r = comp.evaluate_residual(ctx, in, out);
  ASSERT_TRUE(r.ok()) << r.error().message;

  comp.save_state();
  const std::vector<double> after = comp.profile();
  EXPECT_EQ(before, after) << "evaluate_residual mutated the checkpointed profile";
}

TEST(PetscReactionDiffusionComponent, NeumannHalfNativeEvaluateResidualIsPure) {
  PhysicsParams p;
  NeumannHalfNative comp("neumann", /*m=*/4, p);
  comp.seed_initial_condition();
  comp.save_state();
  const std::vector<double> before = comp.profile();

  std::array<double, 1> in{0.2};
  std::array<double, 1> out{0.0};
  const fuse::StepContext ctx = make_ctx();
  const auto r = comp.evaluate_residual(ctx, in, out);
  ASSERT_TRUE(r.ok()) << r.error().message;

  comp.save_state();
  const std::vector<double> after = comp.profile();
  EXPECT_EQ(before, after) << "evaluate_residual mutated the checkpointed profile";
}

TEST(PetscReactionDiffusionComponent, DirichletHalfTokenApplyJacobianMatchesSensitivity) {
  PhysicsParams p;
  DirichletHalf comp("dirichlet", /*m=*/4, p);
  comp.seed_initial_condition();
  comp.save_state();

  std::array<double, 1> in{1.1};
  std::array<double, 1> out{0.0};
  const fuse::StepContext ctx = make_ctx();
  ASSERT_TRUE(comp.evaluate_residual(ctx, in, out).ok());

  fuse::LinearizationToken* tok = comp.linearization();
  ASSERT_NE(tok, nullptr);
  EXPECT_EQ(tok->rows(), 1u);
  EXPECT_EQ(tok->cols(), 1u);

  std::array<double, 1> x{1.0};
  std::array<double, 1> y{0.0};
  ASSERT_TRUE(tok->apply_jacobian(x, y).ok());
  std::array<double, 1> yt{0.0};
  ASSERT_TRUE(tok->apply_jacobian_transpose(x, yt).ok());
  EXPECT_DOUBLE_EQ(y[0], yt[0]) << "a 1x1 operator's transpose must equal itself";

  // A finite-difference cross-check against evaluate_residual itself --
  // the same check the Python/C++ prototypes ran (design_notes.md
  // Section 5).
  const double h = 1e-6;
  std::array<double, 1> in2{in[0] + h};
  std::array<double, 1> out2{0.0};
  ASSERT_TRUE(comp.evaluate_residual(ctx, in2, out2).ok());
  const double fd_slope = (out2[0] - out[0]) / h;
  EXPECT_NEAR(y[0], fd_slope, 1e-4 * std::max(1.0, std::fabs(fd_slope)));
}

TEST(PetscReactionDiffusionComponent, SignatureChangesAcrossEvaluations) {
  // component.hpp's own contract: signature() must change "whenever the
  // linearisation point moves (a new evaluate_residual() call)".
  PhysicsParams p;
  DirichletHalf comp("dirichlet", /*m=*/4, p);
  comp.seed_initial_condition();
  comp.save_state();

  std::array<double, 1> in{1.1}, out{0.0};
  const fuse::StepContext ctx = make_ctx();
  ASSERT_TRUE(comp.evaluate_residual(ctx, in, out).ok());
  const std::uint64_t sig1 = comp.linearization()->signature();

  std::array<double, 1> in2{1.3}, out2{0.0};
  ASSERT_TRUE(comp.evaluate_residual(ctx, in2, out2).ok());
  const std::uint64_t sig2 = comp.linearization()->signature();

  EXPECT_NE(sig1, sig2);
}

TEST(PetscReactionDiffusionComponent, SaveRestoreRoundTripIsExact) {
  PhysicsParams p;
  NeumannHalfNative comp("neumann", /*m=*/4, p);
  comp.seed_initial_condition();
  comp.save_state();
  const std::vector<double> checkpoint = comp.profile();

  // Mutate via advance(), then restore -- must come back bit-identical.
  std::array<double, 1> q{0.3};
  // advance() reads the registered Requires field directly; register_fields()
  // is not exercised here (no graph), so set the field storage this
  // component's own register_fields() would have bound -- the public
  // accessor does not expose a setter, so this test goes through
  // evaluate_residual's own side-effect-free path instead, and verifies
  // restore_state() undoes a REAL advance()-driven mutation by calling
  // advance() through the one path available without a graph: directly,
  // relying on q_in_'s default-constructed 0.0 (register_fields() is
  // what would normally rebind it; absent that, advance() still runs
  // the SAME solve at q_in_ == 0.0, which is sufficient to prove the
  // round trip -- the VALUE advanced to is not this test's concern).
  (void)q;
  const fuse::StepContext ctx = make_ctx();
  ASSERT_TRUE(comp.advance(ctx).ok());
  EXPECT_NE(comp.profile(), checkpoint) << "advance() should have changed the profile";

  ASSERT_TRUE(comp.restore_state().ok());
  EXPECT_EQ(comp.profile(), checkpoint) << "restore_state() must undo advance()'s mutation exactly";
}
