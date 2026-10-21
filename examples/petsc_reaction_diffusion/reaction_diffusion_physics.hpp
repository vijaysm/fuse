#pragma once

/// S7.4 (retitled; see docs/dev/roadmap.md and docs/dev/newton-krylov.md
/// Sec. 14), owner decision 2026-10-21, Option C: the smallest real port of
/// an `ImplicitComponent` onto a PETSc-object-backed `LinearizationToken`
/// (a real `Mat`/`Vec`, `apply_jacobian`/`apply_jacobian_transpose` as
/// `MatMult`/`MatMultTranspose`, never `MATMFFD`), paired with an existing
/// native-substrate implementation of the SAME physics in one
/// Newton-coupled region -- the dn_heat pattern (DirichletSlab/NeumannSlab,
/// examples/dn_heat/dn_component.hpp), generalized from a quasi-steady,
/// closed-form, linear slab to a genuinely time-dependent, discretized,
/// NONLINEAR reaction-diffusion half-domain.
///
/// THIS FILE is the physics shared by every substrate (native Thomas-
/// algorithm solve, PETSc Vec/Mat/KSP solve) and by the tier-0 derivative-
/// gate test (test/linalg/reaction_diffusion_derivative_gate_test.cpp):
/// plain `double`/`std::vector`, no `fuse::` type, no PETSc type. Every
/// function here is a PURE function of its arguments.
///
/// THE PROBLEM. A single scalar field u(x, t) obeys the nonlinear
/// reaction-diffusion equation
///
///   u_t = D u_xx - k u^3 + f(x, t),           x in [0, Ltot], t >= 0
///
/// on a 1-D domain split at x = L_left into two abutting half-domains
/// (Dirichlet-Neumann, exactly dn_heat's own split):
///
///   Left  (native substrate):  x in [0, L_left]
///   Right (petsc substrate):   x in [L_left, L_left + L_right]
///
/// Both outer ends are homogeneous Dirichlet (u = 0). Each half is
/// discretized by `m` interior, backward-Euler, centered-difference grid
/// points (dx_half = L_half / (m + 1)) -- "no new mesh type, no new DOF
/// map" (owner's own Section 4 wording): this is a plain array with two
/// boundary conventions, the same level of abstraction
/// examples/dn_heat/dn_component.hpp's own registered arrays already use,
/// not `fuse::native::NativeMeshBackend`/`NativeGrid`
/// (examples/brusselator/brusselator_physics.hpp), which is multi-rank,
/// MOAB-mesh-adjacent machinery built for a different (2-D, decomposed,
/// remapped) problem and would be a disproportionately large dependency
/// for a single-variable, single-rank demonstration. See this track's own
/// design_notes.md Section 3 for the full reasoning and the alternatives
/// (literal reuse of DirichletSlab/NeumannSlab or
/// UComponent/VComponent) this rejected, and why.
///
/// THE SPLIT, generalizing dn_heat's own flux/temperature exchange:
///
///   Left  (`left_newton_solve`) RECEIVES the interface value u_I
///         (Dirichlet BC at its right end), solves its own m-point
///         nonlinear backward-Euler system, and PROVIDES the flux
///         q_I = -D (u_I - u_m) / dx_L it took to hold that boundary value
///         (a one-sided two-point estimate of -D u_x at the interface).
///
///   Right (`right_newton_solve`) RECEIVES that flux q_I (Neumann BC at
///         its left end, via a one-sided ghost value u_g = u_1 +
///         q_I dx_R / D consistent with the SAME two-point flux formula),
///         solves its own m-point system, and PROVIDES u_g itself back --
///         the interface value ITS OWN solution implies -- as the next
///         Dirichlet datum for Left.
///
/// Composing the two gives the SAME kind of fixed-point map dn_heat's own
/// header documents (u_I <- G_right(G_left(u_I))), with a measured
/// Picard contraction factor whose magnitude EXCEEDS 1 at this track's own
/// default parameters (measured ~1.40, see design_notes.md Section 5) --
/// i.e., naive Picard diverges here, by construction, demonstrating
/// exactly the case S7.2's native Newton-Krylov coupler (and this
/// component's own analytic `LinearizationToken`) exists for: Newton's
/// local quadratic convergence does not depend on the Picard contraction
/// rate, only on linearizing about a nearby point (the previous time
/// level's own converged interface value, reused as the initial guess).
///
/// THE OUTER SENSITIVITY (`dG/du_I` for Left, `dG/dq_I` for Right) --
/// what `ImplicitComponent::linearization()` returns, wrapped in a real
/// PETSc Mat for the `petsc` substrate -- is obtained by IMPLICIT
/// DIFFERENTIATION of the converged inner Newton solve: at the solved
/// point, the inner Jacobian `J` (already formed and factored for the
/// last Newton step) solves ONE more linear system `J v = b` for a
/// SINGLE right-hand side `b` (the partial derivative of the inner
/// residual with respect to the received interface datum, nonzero at
/// exactly the one row the datum enters), and `dG/d(datum) = c^T v`
/// (`c` being "how the provided quantity reads off the solved profile").
/// This is exact (not a finite-difference approximation) and reuses the
/// SAME linear algebra the last Newton step already paid for.
///
/// MEASURED (this track, this session, a throwaway Python prototype of
/// this exact discretization and sign convention, NOT this C++ file --
/// see design_notes.md Section 5): the analytic `dG/du_I`/`dG/dq_I`
/// formulas below agree with a central finite difference of the composed
/// solve to 9+ significant digits once the ghost point's OWN dependence
/// on the unknown it is attached to (`d(left_contrib)/du_1 = 1` for
/// Right's row 0) is correctly folded into that row's diagonal -- a bug
/// the prototype caught BEFORE this file was written (the first draft
/// used the generic interior-row diagonal unconditionally, which left
/// Newton converging only linearly, not quadratically, and threw the
/// implicit-differentiation sensitivity off by about 0.4%; both symptoms
/// are recorded in design_notes.md Section 5 as the reason this
/// correction exists).

#include <cmath>
#include <cstddef>
#include <vector>

namespace petsc_reaction_diffusion {

/// Shared physical and discretization parameters. Defaults chosen so the
/// manufactured solution below (a) satisfies BOTH outer homogeneous
/// Dirichlet ends exactly (sin(pi) = 0) and (b) has a NONZERO value and
/// NONZERO flux at the interface x = L_left (avoiding the degenerate
/// choice L_left = Ltot/2, where the interface sits exactly at the
/// manufactured sine's own extremum and the exchanged flux is
/// identically zero for all time -- caught the same way as the Jacobian
/// bug above, by evaluating the closed-form derivative at x = L_left
/// before committing to these numbers).
struct PhysicsParams {
  double D = 0.7;          ///< diffusivity
  double k = 0.3;           ///< cubic reaction (absorption) coefficient
  double u0 = 1.3;          ///< manufactured-solution amplitude
  double tau = 2.0;         ///< manufactured-solution decay time
  double L_left = 1.0;      ///< Left half-domain length
  double L_right = 1.5;     ///< Right half-domain length
  [[nodiscard]] double Ltot() const noexcept { return L_left + L_right; }
};

/// The manufactured solution u(x,t) = u0 sin(pi x / Ltot) exp(-t/tau).
/// EXACTLY satisfies u_t = D u_xx - k u^3 + f with `manufactured_source`
/// below, by direct substitution (verified symbolically, this track's own
/// design_notes.md Section 5) -- u(0,t) = u(Ltot,t) = 0 for every t, so it
/// also satisfies both outer boundary conditions exactly, for any D, k,
/// u0, tau.
[[nodiscard]] inline double manufactured_u(double x, double t, const PhysicsParams& p) {
  return p.u0 * std::sin(M_PI * x / p.Ltot()) * std::exp(-t / p.tau);
}

/// du/dx of `manufactured_u`, used only to confirm (design_notes.md,
/// never at run time by a component) that the interface flux is
/// nonzero at this track's own default parameters.
[[nodiscard]] inline double manufactured_u_x(double x, double t, const PhysicsParams& p) {
  const double Ltot = p.Ltot();
  return p.u0 * (M_PI / Ltot) * std::cos(M_PI * x / Ltot) * std::exp(-t / p.tau);
}

/// f(x,t) = u_t - D u_xx + k u^3 evaluated on `manufactured_u`, in closed
/// form (sympy-derived, design_notes.md Section 5): the forcing term that
/// makes `manufactured_u` an EXACT solution of this file's own PDE.
[[nodiscard]] inline double manufactured_source(double x, double t, const PhysicsParams& p) {
  const double Ltot = p.Ltot();
  const double s = std::sin(M_PI * x / Ltot);
  const double e1 = std::exp(-t / p.tau);
  return (M_PI * M_PI * p.D * p.u0 * e1 * s) / (Ltot * Ltot)
         + p.k * p.u0 * p.u0 * p.u0 * std::exp(-3.0 * t / p.tau) * s * s * s
         - p.u0 * e1 * s / p.tau;
}

/// A tridiagonal linear system `sub`/`diag`/`sup` (sub.size() ==
/// sup.size() == diag.size() - 1, `sub[i]` below `diag[i+1]`, `sup[i]`
/// above `diag[i]`), solved by the Thomas algorithm -- the native
/// substrate's own linear solve; the PETSc substrate assembles the
/// IDENTICAL three diagonals into a real `Mat` and solves through `KSP`
/// instead (reaction_diffusion_component.hpp).
struct TridiagSystem {
  std::vector<double> sub;    ///< sub[i] multiplies unknown i (i = 1..m-1)
  std::vector<double> diag;   ///< diag[i] multiplies unknown i (i = 0..m-1)
  std::vector<double> sup;    ///< sup[i] multiplies unknown i+1 (i = 0..m-2)
};

/// Solves `J x = rhs` for a `TridiagSystem` J, by the Thomas algorithm
/// (forward elimination, back substitution) -- O(m), exact up to
/// round-off for a diagonally dominant system, which every Jacobian this
/// file builds is (the reaction term only ever ADDS to the diagonal).
[[nodiscard]] inline std::vector<double> thomas_solve(const TridiagSystem& J,
                                                       const std::vector<double>& rhs) {
  const std::size_t m = J.diag.size();
  std::vector<double> c(J.sup), d(rhs), b(J.diag);
  for (std::size_t i = 1; i < m; ++i) {
    const double w = J.sub[i - 1] / b[i - 1];
    b[i] -= w * c[i - 1];
    d[i] -= w * d[i - 1];
  }
  std::vector<double> x(m, 0.0);
  x[m - 1] = d[m - 1] / b[m - 1];
  for (std::size_t ii = m - 1; ii-- > 0;) {
    x[ii] = (d[ii] - c[ii] * x[ii + 1]) / b[ii];
  }
  return x;
}

/// Left half's own backward-Euler residual AND Jacobian at `u` (m
/// interior unknowns at x_i = i * dx, i = 1..m, dx = L_left/(m+1)),
/// given the Dirichlet interface datum `u_I` and the previous time
/// level's own converged profile `u_old`. `t_new` is the time level this
/// residual is evaluated at (ctx.t_end, for a component).
struct ResidualJacobian {
  std::vector<double> R;
  TridiagSystem J;
};

[[nodiscard]] inline ResidualJacobian left_residual_jacobian(
    const std::vector<double>& u, const std::vector<double>& u_old, double u_I, double dt,
    double dx, double t_new, const PhysicsParams& p) {
  const std::size_t m = u.size();
  ResidualJacobian out;
  out.R.resize(m);
  out.J.diag.resize(m);
  out.J.sub.resize(m - 1);
  out.J.sup.resize(m - 1);
  for (std::size_t i = 0; i < m; ++i) {
    const double x = static_cast<double>(i + 1) * dx;
    const double left_u = (i > 0) ? u[i - 1] : 0.0;             // outer Dirichlet BC = 0
    const double right_u = (i + 1 < m) ? u[i + 1] : u_I;        // interface Dirichlet BC
    out.R[i] = (u[i] - u_old[i]) / dt - p.D * (right_u - 2.0 * u[i] + left_u) / (dx * dx) +
               p.k * u[i] * u[i] * u[i] - manufactured_source(x, t_new, p);
    out.J.diag[i] = 1.0 / dt + 2.0 * p.D / (dx * dx) + 3.0 * p.k * u[i] * u[i];
    if (i > 0) out.J.sub[i - 1] = -p.D / (dx * dx);
    if (i + 1 < m) out.J.sup[i] = -p.D / (dx * dx);
  }
  return out;
}

/// Right half's own backward-Euler residual AND Jacobian at `u` (m
/// interior unknowns at x_i = L_left + i*dx, i = 1..m), given the
/// Neumann interface datum `q_I` (a flux) through the one-sided ghost
/// `u_g = u[0] + q_I*dx/D` -- see this file's own top comment for the
/// derivation. ROW 0's diagonal carries the ghost's own `d(u_g)/du[0] =
/// 1` correction (`-D/dx^2`, folded into the usual `+2D/dx^2` baseline
/// to leave `+D/dx^2`): THE CORRECTION THIS FILE'S OWN TOP COMMENT NAMES
/// AS A BUG CAUGHT BY THE PROTOTYPE -- omitting it still converges (the
/// residual is unaffected, only the Jacobian is), but only linearly, and
/// throws the implicit-differentiation sensitivity below off by about
/// 0.4% at this track's own defaults (measured, design_notes.md Section 5).
[[nodiscard]] inline ResidualJacobian right_residual_jacobian(
    const std::vector<double>& u, const std::vector<double>& u_old, double q_I, double dt,
    double dx, double t_new, const PhysicsParams& p) {
  const std::size_t m = u.size();
  ResidualJacobian out;
  out.R.resize(m);
  out.J.diag.resize(m);
  out.J.sub.resize(m - 1);
  out.J.sup.resize(m - 1);
  for (std::size_t i = 0; i < m; ++i) {
    const double x = p.L_left + static_cast<double>(i + 1) * dx;
    const double left_contrib = (i > 0) ? u[i - 1] : (u[0] + q_I * dx / p.D);  // ghost at i=0
    const double right_u = (i + 1 < m) ? u[i + 1] : 0.0;  // outer Dirichlet BC = 0
    out.R[i] = (u[i] - u_old[i]) / dt - p.D * (right_u - 2.0 * u[i] + left_contrib) / (dx * dx) +
               p.k * u[i] * u[i] * u[i] - manufactured_source(x, t_new, p);
    double diag_i = 1.0 / dt + 2.0 * p.D / (dx * dx) + 3.0 * p.k * u[i] * u[i];
    if (i == 0) diag_i -= p.D / (dx * dx);  // ghost's own d(left_contrib)/du[0] = 1 correction
    out.J.diag[i] = diag_i;
    if (i > 0) out.J.sub[i - 1] = -p.D / (dx * dx);
    if (i + 1 < m) out.J.sup[i] = -p.D / (dx * dx);
  }
  return out;
}

/// One Left-half Newton solve: `u_old` (m,), interface datum `u_I`,
/// returns the converged profile, the number of iterations taken (for
/// the quadratic-rate acceptance criterion), the provided flux q_I, and
/// the analytic sensitivity dG/du_I (by implicit differentiation of the
/// converged system -- one more `thomas_solve` with the SAME `J`).
/// `max_residual_history`, when non-null, receives every sweep's own
/// residual norm (quadratic-rate measurement; empty/null for a caller
/// that only wants the converged answer).
struct LeftSolveResult {
  std::vector<double> u;
  double q_I = 0.0;
  double dG_du_I = 0.0;
  int iterations = 0;
};

[[nodiscard]] inline LeftSolveResult left_newton_solve(
    const std::vector<double>& u_old, double u_I, double dt, double dx, double t_new,
    const PhysicsParams& p, double tol = 1e-13, int max_iterations = 40,
    std::vector<double>* residual_history = nullptr) {
  std::vector<double> u = u_old;
  const std::size_t m = u.size();
  int it = 0;
  for (; it < max_iterations; ++it) {
    const ResidualJacobian rj = left_residual_jacobian(u, u_old, u_I, dt, dx, t_new, p);
    double nrm2 = 0.0;
    for (double r : rj.R) nrm2 += r * r;
    const double nrm = std::sqrt(nrm2);
    if (residual_history != nullptr) residual_history->push_back(nrm);
    if (nrm < tol) break;
    std::vector<double> neg_r(m);
    for (std::size_t i = 0; i < m; ++i) neg_r[i] = -rj.R[i];
    const std::vector<double> du = thomas_solve(rj.J, neg_r);
    for (std::size_t i = 0; i < m; ++i) u[i] += du[i];
  }
  const ResidualJacobian rj = left_residual_jacobian(u, u_old, u_I, dt, dx, t_new, p);
  std::vector<double> b(m, 0.0);
  b[m - 1] = p.D / (dx * dx);
  const std::vector<double> sens = thomas_solve(rj.J, b);  // d(u)/d(u_I)

  LeftSolveResult out;
  out.u = u;
  out.iterations = it + 1;
  out.q_I = -p.D * (u_I - u[m - 1]) / dx;
  out.dG_du_I = -p.D / dx * (1.0 - sens[m - 1]);
  return out;
}

struct RightSolveResult {
  std::vector<double> u;
  double u_interface = 0.0;  ///< the interface value this half's solution implies
  double dG_dq_I = 0.0;
  int iterations = 0;
};

[[nodiscard]] inline RightSolveResult right_newton_solve(
    const std::vector<double>& u_old, double q_I, double dt, double dx, double t_new,
    const PhysicsParams& p, double tol = 1e-13, int max_iterations = 40,
    std::vector<double>* residual_history = nullptr) {
  std::vector<double> u = u_old;
  const std::size_t m = u.size();
  int it = 0;
  for (; it < max_iterations; ++it) {
    const ResidualJacobian rj = right_residual_jacobian(u, u_old, q_I, dt, dx, t_new, p);
    double nrm2 = 0.0;
    for (double r : rj.R) nrm2 += r * r;
    const double nrm = std::sqrt(nrm2);
    if (residual_history != nullptr) residual_history->push_back(nrm);
    if (nrm < tol) break;
    std::vector<double> neg_r(m);
    for (std::size_t i = 0; i < m; ++i) neg_r[i] = -rj.R[i];
    const std::vector<double> du = thomas_solve(rj.J, neg_r);
    for (std::size_t i = 0; i < m; ++i) u[i] += du[i];
  }
  const ResidualJacobian rj = right_residual_jacobian(u, u_old, q_I, dt, dx, t_new, p);
  std::vector<double> b(m, 0.0);
  b[0] = 1.0 / dx;
  const std::vector<double> sens = thomas_solve(rj.J, b);  // d(u)/d(q_I)

  RightSolveResult out;
  out.u = u;
  out.iterations = it + 1;
  out.u_interface = u[0] + q_I * dx / p.D;
  out.dG_dq_I = sens[0] + dx / p.D;
  return out;
}

}  // namespace petsc_reaction_diffusion
