// The derivative gate
// (docs/dev/derivative-gate.md, include/fuse/linalg/derivative_gate.hpp)
// applied to reaction_diffusion_physics.hpp's own inner Jacobian --
// shared by BOTH substrates (the native Thomas-algorithm solve and the
// PETSc Vec/Mat/KSP solve, reaction_diffusion_component_petsc.hpp, assemble
// and solve the IDENTICAL stencil this file tests). Tier-0, no PETSc, no
// MPI: the PETSc substrate is a real, assembled Mat/Vec wired through
// MatMult/MatMultTranspose (component_petsc.hpp's own header comment),
// never MATMFFD, but the inner Jacobian's OWN correctness -- the thing
// this gate measures -- is a property of the shared stencil formulas,
// provable here without touching a PETSc type at all.
//
// THE FAILURE THIS FILE'S OWN CONTROL CASE REPRODUCES.
// `UncorrectedGhostJacobianMeasuresOrderOneNotTwo` uses the Jacobian a
// naive `right_residual_jacobian` would have: row 0's diagonal built from the
// generic interior-row formula (`2*D/dx^2`) without correcting for the
// ghost value's own `d(left_contrib)/du[0] = 1` dependence (which should
// reduce it to `D/dx^2`). That Jacobian is still SYMMETRIC (a diagonal
// perturbation cannot break symmetry), so the adjoint-identity check
// below cannot catch it -- only the Taylor-remainder order can, which is
// exactly why this file carries both checks rather than relying on
// either alone.

#include <gtest/gtest.h>

#include <functional>
#include <random>
#include <vector>

#include "fuse/linalg/derivative_gate.hpp"
#include "reaction_diffusion_physics.hpp"

namespace {

using petsc_reaction_diffusion::PhysicsParams;
using petsc_reaction_diffusion::ResidualJacobian;
using petsc_reaction_diffusion::right_residual_jacobian;
using petsc_reaction_diffusion::TridiagSystem;

// Dense tridiagonal matrix-vector product -- the EXACT operator
// `right_newton_solve`'s own Newton iteration and
// `right_newton_solve_petsc`'s own `MatMult` apply (component_petsc.hpp)
// both act through (the latter via a real, assembled PETSc `Mat`
// carrying the IDENTICAL three diagonals).
std::vector<double> tridiag_matvec(const TridiagSystem& J, const std::vector<double>& v) {
  const std::size_t m = J.diag.size();
  std::vector<double> out(m);
  for (std::size_t i = 0; i < m; ++i) {
    double s = J.diag[i] * v[i];
    if (i > 0) s += J.sub[i - 1] * v[i - 1];
    if (i + 1 < m) s += J.sup[i] * v[i + 1];
    out[i] = s;
  }
  return out;
}

// A representative, fixed evaluation point: m=6 interior unknowns, the
// physical defaults, a realistic dt/dx/t, and a solution profile offset
// from zero (so the cubic reaction term's own derivative, 3*k*u^2, is
// not trivially zero everywhere -- u=0 would make this gate vacuously
// agree with ANY Jacobian that gets the diffusion part right).
struct Fixture {
  PhysicsParams p;
  std::size_t m = 6;
  double dx = 0.2;
  double dt = 0.01;
  double t_new = 0.07;
  double q_I = 0.35;
  std::vector<double> u_old;
  std::vector<double> u_eval;

  Fixture() {
    u_old.resize(m);
    u_eval.resize(m);
    std::mt19937 rng(7u);
    std::uniform_real_distribution<double> du(0.2, 1.8);
    for (std::size_t i = 0; i < m; ++i) {
      u_old[i] = du(rng);
      u_eval[i] = u_old[i] + 0.1 * du(rng);
    }
  }
};

}  // namespace

TEST(ReactionDiffusionDerivativeGate, AdjointIdentityHoldsForTheInnerJacobian) {
  // The Right-half stencil is symmetric by construction (diffusion's own
  // centered stencil is symmetric; the reaction term and the row-0 ghost
  // correction only ever touch the diagonal) -- apply_transpose == apply.
  Fixture fx;
  const ResidualJacobian rj =
      right_residual_jacobian(fx.u_eval, fx.u_old, fx.q_I, fx.dt, fx.dx, fx.t_new, fx.p);
  const std::size_t m = fx.m;

  fuse::linalg::HostApplyFn apply = [&](std::span<const double> x, std::span<double> y) {
    const std::vector<double> xv(x.begin(), x.end());
    const std::vector<double> yv = tridiag_matvec(rj.J, xv);
    std::copy(yv.begin(), yv.end(), y.begin());
    return fuse::Result<void>::success();
  };

  const auto norm_est = fuse::linalg::estimate_operator_norm(m, m, apply, apply);
  ASSERT_TRUE(norm_est.ok()) << norm_est.error().message;

  std::mt19937 rng(11u);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> x(m), y(m);
  for (double& e : x) e = u(rng);
  for (double& e : y) e = u(rng);

  const auto gate = fuse::linalg::adjoint_gate(m, m, apply, apply, x, y, norm_est.value(),
                                               fuse::linalg::kAdjointGateConstant,
                                               "reaction_diffusion.right_inner_jacobian");
  ASSERT_TRUE(gate.ok()) << gate.error().message;
  EXPECT_TRUE(gate.value().passed)
      << "defect " << gate.value().defect << " bound " << gate.value().bound;
}

TEST(ReactionDiffusionDerivativeGate, RightInnerJacobianMeasuresOrderTwo) {
  Fixture fx;
  const auto F = [fx](const std::vector<double>& u) {
    return right_residual_jacobian(u, fx.u_old, fx.q_I, fx.dt, fx.dx, fx.t_new, fx.p).R;
  };
  const auto J = [fx](const std::vector<double>& u, const std::vector<double>& v) {
    const ResidualJacobian rj =
        right_residual_jacobian(u, fx.u_old, fx.q_I, fx.dt, fx.dx, fx.t_new, fx.p);
    return tridiag_matvec(rj.J, v);
  };

  std::mt19937 rng(13u);
  std::uniform_real_distribution<double> dv(-1.0, 1.0);
  std::vector<double> v(fx.m);
  for (double& e : v) e = dv(rng);

  const auto r = fuse::linalg::taylor_remainder_order(F, J, fx.u_eval, v, /*h0=*/1e-1,
                                                       /*ndecades=*/9);
  ASSERT_FALSE(r.at_floor_everywhere) << "the correct Jacobian never reached a decreasing trend";
  EXPECT_GT(r.order, 1.8) << "observed order " << r.order << ", floor " << r.floor;
  EXPECT_LT(r.order, 2.3) << "observed order " << r.order << " suspiciously high";
}

TEST(ReactionDiffusionDerivativeGate, WrongReactionJacobianMeasuresOrderOneNotTwo) {
  // The test's own test (derivative-gate.md's own convention): the
  // reaction term's derivative deliberately wrong (2*k*u^2 instead of
  // 3*k*u^2), diffusion part left correct.
  Fixture fx;
  const auto F = [fx](const std::vector<double>& u) {
    return right_residual_jacobian(u, fx.u_old, fx.q_I, fx.dt, fx.dx, fx.t_new, fx.p).R;
  };
  const auto J_wrong = [fx](const std::vector<double>& u, const std::vector<double>& v) {
    ResidualJacobian rj =
        right_residual_jacobian(u, fx.u_old, fx.q_I, fx.dt, fx.dx, fx.t_new, fx.p);
    for (std::size_t i = 0; i < fx.m; ++i) {
      const double correct_reaction = 3.0 * fx.p.k * u[i] * u[i];
      const double wrong_reaction = 2.0 * fx.p.k * u[i] * u[i];
      rj.J.diag[i] += (wrong_reaction - correct_reaction);
    }
    return tridiag_matvec(rj.J, v);
  };

  std::mt19937 rng(13u);
  std::uniform_real_distribution<double> dv(-1.0, 1.0);
  std::vector<double> v(fx.m);
  for (double& e : v) e = dv(rng);

  const auto r =
      fuse::linalg::taylor_remainder_order(F, J_wrong, fx.u_eval, v, /*h0=*/1e-1, /*ndecades=*/9);
  ASSERT_FALSE(r.at_floor_everywhere);
  EXPECT_GT(r.order, 0.7) << "observed order " << r.order;
  EXPECT_LT(r.order, 1.3) << "observed order " << r.order
                         << " -- a wrong Jacobian measured as order 2 would mean this check "
                            "cannot tell a correct linearisation from an incorrect one";
}

TEST(ReactionDiffusionDerivativeGate, UncorrectedGhostJacobianMeasuresOrderOneNotTwo) {
  // A naive right_residual_jacobian would build row 0's diagonal from the
  // generic interior-row formula (2*D/dx^2), never correcting for the
  // ghost value's own d(left_contrib)/du[0] = 1 dependence (which must
  // reduce it to D/dx^2). Reproduced here, permanently, as a
  // regression control: the uncorrected Jacobian is proven, not
  // asserted, to measure order ~1, not 2.
  Fixture fx;
  const auto F = [fx](const std::vector<double>& u) {
    return right_residual_jacobian(u, fx.u_old, fx.q_I, fx.dt, fx.dx, fx.t_new, fx.p).R;
  };
  const auto J_uncorrected = [fx](const std::vector<double>& u, const std::vector<double>& v) {
    ResidualJacobian rj =
        right_residual_jacobian(u, fx.u_old, fx.q_I, fx.dt, fx.dx, fx.t_new, fx.p);
    rj.J.diag[0] += fx.p.D / (fx.dx * fx.dx);  // undo the correction: back to 2D/dx^2
    return tridiag_matvec(rj.J, v);
  };

  std::mt19937 rng(13u);
  std::uniform_real_distribution<double> dv(-1.0, 1.0);
  std::vector<double> v(fx.m);
  for (double& e : v) e = dv(rng);

  const auto r = fuse::linalg::taylor_remainder_order(F, J_uncorrected, fx.u_eval, v,
                                                       /*h0=*/1e-1, /*ndecades=*/9);
  ASSERT_FALSE(r.at_floor_everywhere);
  EXPECT_GT(r.order, 0.7) << "observed order " << r.order;
  EXPECT_LT(r.order, 1.3)
      << "observed order " << r.order
      << " -- the uncorrected ghost Jacobian (the naive formula) measured as "
         "order 2 would mean this gate cannot tell it from the corrected formula";
}
