// KNOWN_ISSUES.md D9 diagnosis: "Find where the coupled FSI gains energy".
//
// The lead's method (docs/user/fsi-channel.md's "Known open issues", this
// task's own instructions): make the question linear-algebraic. Both halves
// of examples/fsi_channel are LINEAR per step (Stokes and Euler-Bernoulli are
// both linear PDEs, discretised linearly; every boundary function here is a
// FIXED function of (position, time) only, never of the solution) and the
// coupling (consistent H / conservative H^T D) is a fixed linear remap too.
// So one converged coupled step is an EXACT affine map
//
//   x_{n+1} = M x_n + b
//
// over the full carried state -- and, since every boundary datum here is
// time-INDEPENDENT (the inflow profile is steady, the no-slip walls are
// steady 0), M and b are the SAME every coupling cycle: this is genuinely a fixed
// linear recurrence, not merely "linearised about a waveform point". Its
// spectral radius rho(M) is therefore the exact per-coupling-cycle growth factor
// this whole test suite has been chasing: rho(M) > 1 confirms a genuine
// SCHEME instability, independent of the coupling iteration (which the
// existing tests already established converges tightly every window), and
// the dominant eigenvector says WHICH carried-state entries carry it.
//
// "The full carried state" (this task's own phrasing) is exactly what each
// component's own save_state()/restore_state() pair checkpoints:
//   - BeamComponent: BeamSolver::State { w, wdot, wddot } (interior nodes
//     only -- the clamped ends are permanently 0, never perturbed);
//   - FluidComponent: FlowState { u, v, p }, AND `velocity_in_t0_` --
//     save_state()'s own body treats the wall-velocity-at-this-coupling-cycle's-t0
//     snapshot as part of what it captures (see fsi_components.hpp), so it
//     is "anything else the component checkpoints" this task's instructions
//     call out, not an incidental detail. `FluidComponent::
//     set_velocity_in_for_test()` (added this round) is the seam that
//     makes it independently perturbable -- normally only the "wall_
//     velocity" field transfer delivery (H applied to the beam's CURRENT
//     velocity) ever writes it, which is unreachable from outside the
//     implicit coupling group machinery for an arbitrary, non-waveform state.
//
// M is built by finite-differencing this map at x=0 (so b = M*0+b is the
// zero-state run's own output, and every other run's column is
// (run(eps*e_i) - b)/eps) -- exact up to the coupling iteration's own
// tolerance (driven far below the eps*1e-3 perturbation scale by a tight
// `tol`) and ordinary floating-point round-off, since the map is genuinely
// linear, not merely well-approximated as such.
//
// OUTCOME (KNOWN_ISSUES.md D9 has the full numbers): rho(M) = 1.108145 on
// the production grid, rho^2 matching the measured 1.22796 per-step
// energy ratio to ~4 figures; on the smaller grid the tests below
// actually run (for runtime), rho(M) = 1.115-1.116 and rho^2 agrees only
// to ~2 figures -- same qualitative mechanism, weaker quantitative match
// at coarser resolution. Either way this confirms the growth is a
// genuine scheme instability. The dominant eigenvector is ~85-96% pressure, ~4-15% the
// beam's own first bending mode; its velocity field is divergence-free
// everywhere except the outflow column, where the pressure-outlet
// condition's own documented trade-off leaves a local mass source. A
// global mass-conservation correction (StokesSolver::step) was added and
// verified harmless, but does NOT resolve D9 (rho and the long-run energy
// ratio both move by under 1%; see DISABLED_LongRunEnergyGrowth... below).
// Direct ablation shows the instability is carried almost entirely by
// wall_traction()'s pressure term. D9 stays OPEN.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fuse/base/config.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include "beam_solver.hpp"
#include "fsi_components.hpp"
#include "stokes_solver.hpp"

namespace {

constexpr fuse::Units kMps{.length = 1, .time = -1};

[[nodiscard]] double spectral_radius(const Eigen::MatrixXd& M) {
  Eigen::EigenSolver<Eigen::MatrixXd> es(M, /*computeEigenvectors=*/false);
  double r = 0.0;
  for (Eigen::Index i = 0; i < es.eigenvalues().size(); ++i) {
    r = std::max(r, std::abs(es.eigenvalues()[i]));
  }
  return r;
}

/// Returns (spectral radius, index of the dominant eigenvalue, the
/// eigenvalues themselves) -- callers that need the eigenvector look it up
/// at the returned index.
struct EigResult {
  double rho;
  Eigen::Index dominant;
  Eigen::EigenSolver<Eigen::MatrixXd> solver;
};

[[nodiscard]] EigResult eig_analysis(const Eigen::MatrixXd& M) {
  EigResult r{0.0, 0, Eigen::EigenSolver<Eigen::MatrixXd>(M, true)};
  for (Eigen::Index i = 0; i < r.solver.eigenvalues().size(); ++i) {
    const double mag = std::abs(r.solver.eigenvalues()[i]);
    if (mag > r.rho) {
      r.rho = mag;
      r.dominant = i;
    }
  }
  return r;
}

/// Generic column-by-column affine-map builder: `run(x)` must return the
/// map's output for state `x`, and `run(0)` is used as `b`. `eps` is a
/// single, moderate finite-difference step -- exact for a genuinely affine
/// map (confirmed below by cross-checking two different `eps` values agree
/// to near round-off), so nothing here hunts for a "good" step size the way
/// a nonlinear Jacobian estimate would have to.
template <class F>
Eigen::MatrixXd build_affine_map(std::size_t D, double eps, F&& run,
                                 Eigen::VectorXd* b_out = nullptr) {
  const Eigen::VectorXd x0 = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(D));
  const Eigen::VectorXd b = run(x0);
  if (b_out) *b_out = b;
  Eigen::MatrixXd M(static_cast<Eigen::Index>(D), static_cast<Eigen::Index>(D));
  for (std::size_t j = 0; j < D; ++j) {
    Eigen::VectorXd xp = x0;
    xp[static_cast<Eigen::Index>(j)] += eps;
    M.col(static_cast<Eigen::Index>(j)) = (run(xp) - b) / eps;
  }
  return M;
}

// ---------------------------------------------------------------------------
// Beam alone: q=0 (no fluid feedback), interior (w, wdot, wddot) only.
// ---------------------------------------------------------------------------
Eigen::MatrixXd beam_alone_map(const fsi::BeamParams& p, double dt, double eps) {
  const std::size_t m = p.n - 1;
  const std::size_t D = 3 * m;
  auto run = [&](const Eigen::VectorXd& x) {
    fsi::BeamSolver bs(p);
    std::vector<double> w(p.n + 1, 0.0), wd(p.n + 1, 0.0), wdd(p.n + 1, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
      w[i + 1] = x[static_cast<Eigen::Index>(i)];
      wd[i + 1] = x[static_cast<Eigen::Index>(m + i)];
      wdd[i + 1] = x[static_cast<Eigen::Index>(2 * m + i)];
    }
    bs.set_state(w, wd, wdd);
    bs.step_midpoint_load(dt, std::vector<double>(p.n + 1, 0.0));
    Eigen::VectorXd y(static_cast<Eigen::Index>(D));
    const auto& W = bs.displacement();
    const auto& WD = bs.velocity();
    const auto& WDD = bs.acceleration();
    for (std::size_t i = 0; i < m; ++i) {
      y[static_cast<Eigen::Index>(i)] = W[i + 1];
      y[static_cast<Eigen::Index>(m + i)] = WD[i + 1];
      y[static_cast<Eigen::Index>(2 * m + i)] = WDD[i + 1];
    }
    return y;
  };
  return build_affine_map(D, eps, run);
}

// ---------------------------------------------------------------------------
// Fluid alone: wall velocity held at a FIXED (state-independent) value at
// both CN time levels -- decoupled from any beam, exactly the "fluid under
// a prescribed wall velocity" configuration KNOWN_ISSUES.md/fsi-channel.md
// already call bounded. The fixed value itself does not affect M (only b),
// so 0 is as good as any other choice for isolating the linear operator.
// ---------------------------------------------------------------------------
Eigen::MatrixXd fluid_alone_map(const fsi::ChannelParams& p, double dt, double eps) {
  const std::size_t Nu = (p.nx + 1) * p.ny, Nv = p.nx * (p.ny + 1), Np = p.nx * p.ny;
  const std::size_t D = Nu + Nv + Np;
  const std::vector<double> wall(p.nx, 0.0);
  const auto inflow = [](double) { return 0.0; };
  auto run = [&](const Eigen::VectorXd& x) {
    fsi::StokesSolver fs(p);
    fsi::FlowState st;
    st.u.assign(Nu, 0.0);
    st.v.assign(Nv, 0.0);
    st.p.assign(Np, 0.0);
    for (std::size_t i = 0; i < Nu; ++i) st.u[i] = x[static_cast<Eigen::Index>(i)];
    for (std::size_t i = 0; i < Nv; ++i) st.v[i] = x[static_cast<Eigen::Index>(Nu + i)];
    for (std::size_t i = 0; i < Np; ++i) st.p[i] = x[static_cast<Eigen::Index>(Nu + Nv + i)];
    fs.set_state(st);
    fs.step(dt, 0.0, dt, wall, wall, inflow);
    Eigen::VectorXd y(static_cast<Eigen::Index>(D));
    for (std::size_t i = 0; i < Nu; ++i) y[static_cast<Eigen::Index>(i)] = fs.u()[i];
    for (std::size_t i = 0; i < Nv; ++i) y[static_cast<Eigen::Index>(Nu + i)] = fs.v()[i];
    for (std::size_t i = 0; i < Np; ++i) y[static_cast<Eigen::Index>(Nu + Nv + i)] = fs.p()[i];
    return y;
  };
  return build_affine_map(D, eps, run);
}

// ---------------------------------------------------------------------------
// D9 round 3: the lead's elimination argument, measured directly.
//
// WHAT WAS DERIVED AND VALIDATED (not merely assumed -- see this file's own
// git history for the derivation and the numerical cross-check that caught
// two sign/attribution errors in an earlier draft of this comment).
// Dotting every INTERIOR momentum row (u at 1<=i<=nx-1, v at 1<=j<=ny-1 --
// the DOFs with a genuine governing equation) with its own CN-averaged
// velocity and summing gives, EXACTLY (validated to round-off, this test's
// own `closure_residual`):
//
//   (E1_interior-E0_interior)/dt
//     = mu*sum(vel_avg . Lap_avg)          [interior_viscous_power]
//     + sum_cells(p1 . div_avg)             [pressure_div_power]
//     - boundary_bt_term
//
// where boundary_bt_term = sum_i v_wall_avg(i)*p_topcell(i)*dx
//                        + sum_j u_outflow_avg(j)*p_prevcol(j)*dy
// is the ADJOINT (B^T) contribution of the two boundary velocities that
// have NO momentum row of their own (v_wall, u_outflow) but DO appear as
// neighbours in the continuity operator B -- a term the naive identity
// "sum_interior(vel_avg.Gp) = -sum_cells(p.div_avg)" (true only when EVERY
// velocity DOF referenced by B also has its own G row) misses entirely.
//
// mu*sum(vel_avg.Lap_avg) further splits CLEANLY (via the standard
// discrete Green's identity for a 3-point Laplacian -- an algebraic fact,
// proved directly by hand for the boundary case and cross-checked
// numerically here) into bulk dissipation (always <=0) plus ONE-SIDED
// viscous boundary flux terms at v_wall and u_outflow, each with
// coefficient mu (NOT 2*mu) -- `W_wall_viscous_natural` /
// `B_out_viscous_natural` below.
//
// `sum_cells(p1.div_avg)`, however, does NOT further localise to the two
// boundaries: it depends on div_avg at EVERY cell, which in THIS
// diagnostic's own rest-initial-condition setup is nonzero even for
// interior cells (v0's own rest state does not satisfy continuity at
// t0 -- only the SOLVED t1 state does, and div_avg mixes both time
// levels). This term is real, but it is a property of the GLOBAL
// continuity constraint's own Lagrange multiplier, not something that
// splits into "hypothesis A's own contribution" vs "hypothesis B's own
// contribution" the way the viscous term does. Consequence: there is no
// clean R_A + R_B = 0 split to report for the PRESSURE field transfer the way
// there is for viscosity -- see this task's own submission for how this
// was resolved (direct measurement of rho(M) under each candidate
// traction/outflow formula, rather than insisting on a closed-form local
// split that this discretization's own pressure term does not support).
/// Runs ONE cn_step from rest (u0=v0=0 except v0's own WALL slot, which
/// must carry v_wall_old -- cn_step's explicit-side Laplacian reads it as
/// row j=ny-1's north neighbour at t0), zero inflow, no-slip bottom, a
/// NONZERO and CN-time-level-DISTINCT top wall velocity, and the CURRENT
/// (zero-curvature ghost + p=0) outflow treatment -- then computes every
/// term of the energy identity above directly from the solved (u1,v1,p1),
/// by MECHANICAL summation mirroring cn_step's own stencils exactly (no
/// hand-derived closed form for the interior sum -- see `closure_residual`
/// for why that is the trustworthy way to do this).
struct FullIdentityResult {
  double E0 = 0.0, E1 = 0.0;
  double interior_viscous_power = 0.0;  // mu*sum(vel_avg . Lap_avg), ALL DOFs
  double pressure_div_power = 0.0;      // sum_cells(p1 . div_avg)
  double boundary_bt_term = 0.0;  // the missing B^T piece -- see file header
  double ke_change_v_wall = 0.0, ke_change_u_outflow = 0.0;
  double closure_residual = 0.0;  // VALIDATION: should be ~0 (round-off)
  double W_wall_viscous_natural = 0.0;   // mu, one-sided, no pressure
  double B_out_viscous_natural = 0.0;    // mu, one-sided, no pressure
  double W_wall_current = 0.0, W_wall_conjugate = 0.0;  // WITH pressure
  double B_out_natural_form = 0.0, B_out_donothing_form = 0.0;  // WITH pressure
};

/// Runs ONE cn_step from rest (u0=v0=0), zero inflow, no-slip bottom, a
/// NONZERO and CN-time-level-DISTINCT top wall velocity, and the CURRENT
/// (zero-curvature ghost + p=0) outflow treatment -- then computes EVERY
/// term of the discrete energy identity by DIRECT, MECHANICAL summation
/// (mirroring cn_step's own Laplacian/gradient stencils exactly, including
/// every quadratic-ghost and outflow-ghost special case), not via any
/// hand-derived closed form. The one algebraic fact this relies on (see
/// this function's own printed `closure_residual`, which MUST be ~0 to
/// round-off if the mechanical sum below is coded correctly): dotting
/// every INTERIOR momentum row with its own CN-averaged velocity and
/// summing is EXACTLY (E1_interior-E0_interior)/dt, because that is
/// nothing more than "each row's own equation holds" -- true by
/// construction of the linear solve. `closure_residual` verifies THIS
/// specific implementation reproduces cn_step's own stencils correctly,
/// not any hypothesis about A or B.
FullIdentityResult measure_full_energy_identity(const fsi::ChannelParams& c,
                                                double dt) {
  using fsi::detail::idx_p;
  using fsi::detail::idx_u;
  using fsi::detail::idx_v;
  const std::size_t nx = c.nx, ny = c.ny;
  const std::size_t Nu = (nx + 1) * ny, Nv = nx * (ny + 1);
  const std::vector<double> u0(Nu, 0.0);
  std::vector<double> v0(Nv, 0.0);
  std::vector<double> v_wall_old(nx), v_wall_new(nx);
  for (std::size_t i = 0; i < nx; ++i) {
    v_wall_old[i] = 0.05 * (1.0 + 0.1 * static_cast<double>(i));
    v_wall_new[i] = 0.08 * (1.0 + 0.1 * static_cast<double>(i));
  }
  // v0's WALL slot (j=ny, one of v's own ny+1 rows per column) is a real
  // array entry, not a separate quantity -- it must carry the t0 wall
  // velocity, not 0, or cn_step's own EXPLICIT-side Laplacian (which reads
  // v0 at j=ny as row j=ny-1's north neighbour) silently uses the wrong
  // value. Every OTHER v0 entry is genuinely 0 (rest, away from the wall).
  for (std::size_t i = 0; i < nx; ++i) {
    v0[idx_v(c, i, ny)] = v_wall_old[i];
  }
  const double dx = c.L / static_cast<double>(nx);
  const double dy = c.H / static_cast<double>(ny);
  const double t0 = 0.0, t1 = dt, t_mid = 0.5 * dt;
  const double mu = c.mu;

  fsi::ChannelBCs bcs;
  bcs.bottom.u = [](double, double) { return 0.0; };
  bcs.bottom.v = [](double, double) { return 0.0; };
  bcs.top.u = [](double, double) { return 0.0; };
  bcs.top.v = [&](double x, double t) {
    const long long ii = std::lround(x / dx - 0.5);
    const std::size_t i = static_cast<std::size_t>(
        std::clamp<long long>(ii, 0, static_cast<long long>(nx) - 1));
    return (t < t_mid) ? v_wall_old[i] : v_wall_new[i];
  };
  bcs.left.u = [](double, double) { return 0.0; };  // zero inflow
  bcs.left.v = [](double, double) { return 0.0; };
  bcs.right.outflow = true;

  const fsi::PressurePin pin{0, ny - 1, 0.0};
  const fsi::FlowState next =
      fsi::cn_step(c, bcs, fsi::SourceTerms{}, dt, t0, t1, u0, v0, pin,
                  nullptr, nullptr);
  const std::vector<double>& u1 = next.u;
  const std::vector<double>& v1 = next.v;
  const std::vector<double>& p1 = next.p;

  auto uavg = [&](std::size_t i, std::size_t j) {
    return 0.5 * (u1[idx_u(c, i, j)] + u0[idx_u(c, i, j)]);
  };
  auto vavg = [&](std::size_t i, std::size_t j) {
    return 0.5 * (v1[idx_v(c, i, j)] + v0[idx_v(c, i, j)]);
  };
  // v at j=ny (top) is the WALL, not a stored DOF in v1/v0 in the usual
  // sense -- idx_v(c,i,ny) IS a valid index (v has ny+1 rows), and cn_step
  // writes the wall's own Dirichlet value there via the trivial identity
  // row, so v1[idx_v(c,i,ny)] == v_wall_new[i] exactly; v0's own wall slot
  // was set to v_wall_old[i] explicitly above for the same reason
  // (cn_step's EXPLICIT-side Laplacian reads it as row j=ny-1's north
  // neighbour at t0) -- v_wall_old[i]/v_wall_new[i] and v0/v1's own wall
  // slot are therefore the SAME numbers; this helper is just a clear name.
  auto vwall_avg = [&](std::size_t i) {
    return 0.5 * (v_wall_new[i] + v_wall_old[i]);
  };

  double sum0 = 0.0, sum1 = 0.0;
  for (const double u : u0) sum0 += u * u;
  for (const double v : v0) sum0 += v * v;
  for (const double u : u1) sum1 += u * u;
  for (const double v : v1) sum1 += v * v;
  const double cell = dx * dy;

  FullIdentityResult r;
  r.E0 = 0.5 * c.rho * cell * sum0;
  r.E1 = 0.5 * c.rho * cell * sum1;

  double kev = 0.0;
  for (std::size_t i = 0; i < nx; ++i) {
    kev += v_wall_new[i] * v_wall_new[i] - v_wall_old[i] * v_wall_old[i];
  }
  r.ke_change_v_wall = 0.5 * c.rho * cell * kev;

  double keu = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    const double uu = u1[idx_u(c, nx, j)];
    keu += uu * uu;
  }
  r.ke_change_u_outflow = 0.5 * c.rho * cell * keu;

  // ---- mu*sum(vel_avg . Lap_avg) over EVERY interior momentum DOF -------
  // u: i=1..nx-1, j=0..ny-1 (y uses the quadratic ghost at j=0/ny-1, D=0
  // both walls; x is always the standard 3-point stencil since i=1..nx-1
  // are never adjacent to the OUTFLOW ghost -- that only affects v).
  double visc_power = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 1; i < nx; ++i) {
      auto lap_x = [&](const auto& field, std::size_t jj) {
        return (field(i + 1, jj) - 2.0 * field(i, jj) + field(i - 1, jj)) /
               (dx * dx);
      };
      double lap1, lap0;
      if (j == 0) {
        const double c1 = u1[idx_u(c, i, 0)], c2 = u1[idx_u(c, i, 1)];
        const double lap_y1 = (8.0 * 0.0 - 12.0 * c1 + 4.0 * c2) / (3.0 * dy * dy);
        const double c1_0 = u0[idx_u(c, i, 0)], c2_0 = u0[idx_u(c, i, 1)];
        const double lap_y0 = (8.0 * 0.0 - 12.0 * c1_0 + 4.0 * c2_0) / (3.0 * dy * dy);
        lap1 = lap_x([&](std::size_t ii, std::size_t jj) { return u1[idx_u(c, ii, jj)]; }, 0) + lap_y1;
        lap0 = lap_x([&](std::size_t ii, std::size_t jj) { return u0[idx_u(c, ii, jj)]; }, 0) + lap_y0;
      } else if (j == ny - 1) {
        const double c1 = u1[idx_u(c, i, ny - 1)], c2 = u1[idx_u(c, i, ny - 2)];
        const double lap_y1 = (8.0 * 0.0 - 12.0 * c1 + 4.0 * c2) / (3.0 * dy * dy);
        const double c1_0 = u0[idx_u(c, i, ny - 1)], c2_0 = u0[idx_u(c, i, ny - 2)];
        const double lap_y0 = (8.0 * 0.0 - 12.0 * c1_0 + 4.0 * c2_0) / (3.0 * dy * dy);
        lap1 = lap_x([&](std::size_t ii, std::size_t jj) { return u1[idx_u(c, ii, jj)]; }, ny - 1) + lap_y1;
        lap0 = lap_x([&](std::size_t ii, std::size_t jj) { return u0[idx_u(c, ii, jj)]; }, ny - 1) + lap_y0;
      } else {
        const double lap_y1 = (u1[idx_u(c, i, j + 1)] - 2.0 * u1[idx_u(c, i, j)] +
                               u1[idx_u(c, i, j - 1)]) / (dy * dy);
        const double lap_y0 = (u0[idx_u(c, i, j + 1)] - 2.0 * u0[idx_u(c, i, j)] +
                               u0[idx_u(c, i, j - 1)]) / (dy * dy);
        lap1 = lap_x([&](std::size_t ii, std::size_t jj) { return u1[idx_u(c, ii, jj)]; }, j) + lap_y1;
        lap0 = lap_x([&](std::size_t ii, std::size_t jj) { return u0[idx_u(c, ii, jj)]; }, j) + lap_y0;
      }
      const double lap_avg = 0.5 * (lap1 + lap0);
      visc_power += mu * uavg(i, j) * lap_avg * cell;
    }
  }
  // v: i=0..nx-1, j=1..ny-1 (x uses the quadratic ghost at i=0/nx-1, D=0
  // both side walls -- since bcs.right.outflow=true, i=nx-1 instead uses
  // the ghost-free one-sided stencil; y is always standard 3-point).
  for (std::size_t j = 1; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const bool west_outflow = false;  // left is Dirichlet here
      const bool east_outflow = (i == nx - 1);  // bcs.right.outflow
      double lap_x1, lap_x0;
      if (east_outflow) {
        lap_x1 = (v1[idx_v(c, i, j)] - 2.0 * v1[idx_v(c, i - 1, j)] +
                 v1[idx_v(c, i - 2, j)]) / (dx * dx);
        lap_x0 = (v0[idx_v(c, i, j)] - 2.0 * v0[idx_v(c, i - 1, j)] +
                 v0[idx_v(c, i - 2, j)]) / (dx * dx);
      } else if (i == 0) {
        const double c1 = v1[idx_v(c, 0, j)], c2 = v1[idx_v(c, 1, j)];
        lap_x1 = (8.0 * 0.0 - 12.0 * c1 + 4.0 * c2) / (3.0 * dx * dx);
        const double c1_0 = v0[idx_v(c, 0, j)], c2_0 = v0[idx_v(c, 1, j)];
        lap_x0 = (8.0 * 0.0 - 12.0 * c1_0 + 4.0 * c2_0) / (3.0 * dx * dx);
      } else {
        lap_x1 = (v1[idx_v(c, i + 1, j)] - 2.0 * v1[idx_v(c, i, j)] +
                 v1[idx_v(c, i - 1, j)]) / (dx * dx);
        lap_x0 = (v0[idx_v(c, i + 1, j)] - 2.0 * v0[idx_v(c, i, j)] +
                 v0[idx_v(c, i - 1, j)]) / (dx * dx);
      }
      (void)west_outflow;
      // y: v_wall (j==ny neighbour for j=ny-1) uses v_wall_new/old, NOT
      // v1/v0's own array slot (this function's v0/v1 do not carry the
      // wall's t0/t1 values in the j=ny slot -- see vwall_avg's comment).
      double v_np1_1, v_np1_0, v_nm1_1, v_nm1_0;
      v_nm1_1 = v1[idx_v(c, i, j - 1)];
      v_nm1_0 = v0[idx_v(c, i, j - 1)];
      if (j == ny - 1) {
        v_np1_1 = v_wall_new[i];
        v_np1_0 = v_wall_old[i];
      } else {
        v_np1_1 = v1[idx_v(c, i, j + 1)];
        v_np1_0 = v0[idx_v(c, i, j + 1)];
      }
      const double c1v = v1[idx_v(c, i, j)], c0v = v0[idx_v(c, i, j)];
      const double lap_y1 = (v_np1_1 - 2.0 * c1v + v_nm1_1) / (dy * dy);
      const double lap_y0 = (v_np1_0 - 2.0 * c0v + v_nm1_0) / (dy * dy);
      const double lap1 = lap_x1 + lap_y1;
      const double lap0 = lap_x0 + lap_y0;
      const double lap_avg = 0.5 * (lap1 + lap0);
      visc_power += mu * vavg(i, j) * lap_avg * cell;
    }
  }
  r.interior_viscous_power = visc_power;

  // ---- sum_cells(p1 . div_avg) ------------------------------------------
  double press_power = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const double u_e = uavg(i + 1, j), u_w = uavg(i, j);
      const double v_n = (j == ny - 1) ? vwall_avg(i) : vavg(i, j + 1);
      const double v_s = vavg(i, j);
      const double div_avg = (u_e - u_w) / dx + (v_n - v_s) / dy;
      press_power += p1[idx_p(c, i, j)] * div_avg * cell;
    }
  }
  r.pressure_div_power = press_power;

  // ---- boundary_bt_term: the B^T contribution of v_wall/u_outflow --------
  // (see file header -- these two boundary velocities are referenced by
  // the continuity operator B but have no momentum row of their own, so
  // "sum_interior(vel_avg.Gp) = -sum_cells(p.div_avg)" alone misses their
  // contribution; this term is exactly what closes the identity below).
  double boundary_bt_term = 0.0;
  for (std::size_t i = 0; i < nx; ++i) {
    boundary_bt_term += vwall_avg(i) * p1[idx_p(c, i, ny - 1)] / dy * cell;
  }
  for (std::size_t j = 0; j < ny; ++j) {
    boundary_bt_term += uavg(nx, j) * p1[idx_p(c, nx - 1, j)] / dx * cell;
  }
  r.boundary_bt_term = boundary_bt_term;

  // VALIDATION: this MUST be ~0 (round-off) if the mechanical sum above
  // correctly mirrors cn_step's own stencils. (E1_int-E0_int)/dt =
  // interior_viscous_power + pressure_div_power - boundary_bt_term.
  const double e_interior_change =
      (r.E1 - r.E0) - r.ke_change_v_wall - r.ke_change_u_outflow;
  r.closure_residual =
      e_interior_change -
      dt * (r.interior_viscous_power + r.pressure_div_power - boundary_bt_term);

  // ---- W_wall / B_out candidates -----------------------------------------
  double w_visc_nat = 0.0, w_current = 0.0, w_conjugate = 0.0;
  for (std::size_t i = 0; i < nx; ++i) {
    const double vw_avg = vwall_avg(i);
    const double v_interior_avg = vavg(i, ny - 1);
    const double dvdy_natural_mid = (vw_avg - v_interior_avg) / dy;
    w_visc_nat += dx * mu * dvdy_natural_mid * vw_avg;

    const double p_last = p1[idx_p(c, i, ny - 1)];
    const double p_prev = ny >= 2 ? p1[idx_p(c, i, ny - 2)] : p_last;
    const double p_wall_current = 1.5 * p_last - 0.5 * p_prev;
    const double v_m1_1 = v1[idx_v(c, i, ny - 1)], v_m1_0 = v0[idx_v(c, i, ny - 1)];
    const double v_m2_1 = ny >= 2 ? v1[idx_v(c, i, ny - 2)] : v_m1_1;
    const double v_m2_0 = ny >= 2 ? v0[idx_v(c, i, ny - 2)] : v_m1_0;
    const auto dvdy_2nd = [&](double vwall, double vm1, double vm2) {
      return (3.0 * vwall - 4.0 * vm1 + vm2) / (2.0 * dy);
    };
    const double dvdy_current_mid =
        0.5 * (dvdy_2nd(v_wall_old[i], v_m1_0, v_m2_0) +
              dvdy_2nd(v_wall_new[i], v_m1_1, v_m2_1));
    w_current += dx * (-p_wall_current + 2.0 * mu * dvdy_current_mid) * vw_avg;
    w_conjugate += dx * (-p_last + mu * dvdy_natural_mid) * vw_avg;
  }
  r.W_wall_viscous_natural = dt * w_visc_nat;
  r.W_wall_current = dt * w_current;
  r.W_wall_conjugate = dt * w_conjugate;

  double b_visc_nat = 0.0, b_natural_form = 0.0, b_donothing = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    const double u_out_avg = uavg(nx, j);
    const double u_prev_avg = uavg(nx - 1, j);
    const double dudx_mid = (u_out_avg - u_prev_avg) / dx;
    b_visc_nat += dy * mu * dudx_mid * u_out_avg;
    const double p_prev = p1[idx_p(c, nx - 1, j)];
    b_natural_form += dy * (-p_prev + mu * dudx_mid) * u_out_avg;
    b_donothing += dy * (-p_prev + 2.0 * mu * dudx_mid) * u_out_avg;
  }
  r.B_out_viscous_natural = dt * b_visc_nat;
  r.B_out_natural_form = dt * b_natural_form;
  r.B_out_donothing_form = dt * b_donothing;
  return r;
}

// ---------------------------------------------------------------------------
// The real coupled system, through the real FUSE graph -- same field transfer
// declarations as examples/fsi_channel/main.cpp and the other fsi_*_test.cpp
// files (each of which already duplicates this small graph-builder locally).
// ---------------------------------------------------------------------------
struct CoupledCase {
  std::size_t nx_beam, nx_fluid, ny_fluid;
  double L = 1.0, H = 0.2, mu = 0.01, rho_f = 1.0;
  double EI = 2.0, rho_s = 1.0, h_s = 0.5;
  double dt;
  double tol = 1e-13;
  int max_iterations = 800;
  // Peak inflow velocity. Defaults to 1.0 (every EXISTING use of this
  // struct/`run_one_coupled_period` relies on that default and is
  // unaffected). Round 4's own diagnostic sets this to 0.0: since the
  // beam has no external forcing of its own, zero inflow is the ONLY
  // way to make the coupled step's additive/inhomogeneous term `b`
  // (`run_one_coupled_period`'s output on the zero state) vanish
  // EXACTLY, so a single real production run started from a
  // (homogeneous) eigenvector `x` of the linear map `M` reproduces
  // `M`'s own `lambda*x` directly -- letting the energy budget below be
  // read off an ACTUAL run through the real Runner/IQN-ILS path, not a
  // finite-difference construction.
  double Umax = 1.0;
};

fuse::Rational rationalize_dt(double dt) {
  const auto num = static_cast<std::int64_t>(std::llround(dt * 1.0e8));
  return fuse::Rational{num, 100'000'000};
}

fuse::CouplingGraph build_fsi_graph(const CoupledCase& c) {
  fuse::CouplingGraph g;
  const auto m_beam = g.add_mesh(fuse::MeshDecl{
      .name = "beam_iface", .source = "generated:line:" + std::to_string(c.nx_beam)});
  const auto m_fluid = g.add_mesh(fuse::MeshDecl{
      .name = "fluid_iface",
      .source = "generated:line:" + std::to_string(c.nx_fluid)});
  const auto r_beam =
      g.add_region(fuse::Region{.name = "r_beam", .mesh = m_beam, .dimension = 1});
  const auto r_fluid = g.add_region(
      fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});

  const fuse::Rational period = rationalize_dt(c.dt);
  const auto beam = g.add_component(fuse::ComponentDecl{
      .name = "beam", .ranks = {0, 1}, .interval = period});
  const auto fluid = g.add_component(fuse::ComponentDecl{
      .name = "fluid", .ranks = {0, 1}, .interval = period});

  const auto vel_out = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_velocity", .region = r_beam,
      .disc = fuse::Discretization::ContinuousGLL, .order = 1,
      .units = kMps, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto vel_in = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_velocity_in", .region = r_fluid,
      .units = kMps, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});
  const auto vch = g.add_channel("wall_velocity", vel_out, vel_in);
  g.channel(vch).remap_method("consistent");

  const auto force_out = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_force", .region = r_fluid, .units = fuse::units::kNewton,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto load_in = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_load", .region = r_beam,
      .disc = fuse::Discretization::ContinuousGLL, .order = 1,
      .units = fuse::units::kNewton, .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto fch = g.add_channel("wall_force", force_out, load_in);
  g.channel(fch).remap_method("consistent-transpose").conservative();
  g.channel(fch).lag(1).allow_time_level_mismatch(
      "resolved by the beam-fluid implicit coupling group");

  fuse::ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(c.tol, "beam_velocity"));
  crit.add(fuse::Converged::Relative(c.tol, "fluid_force"));
  fuse::AccelConfig accel{.kind = fuse::AccelKind::IQN_ILS,
                          .initial_relaxation = 0.5, .reuse = 8,
                          .filter = fuse::FilterKind::None};
  g.iterate({vch, fch})
      .name("fsi_beam_fluid")
      .until(std::move(crit))
      .max_iterations(c.max_iterations)
      .accelerate(accel);
  return g;
}

/// Dimensions of the coupled carried state, and the offsets of each block
/// within the flat vector `run_one_coupled_period` reads/writes -- named
/// once here so the diagnostic below can attribute an eigenvector's mass to
/// a physical block by name instead of a bare index range.
struct CoupledDims {
  std::size_t m_beam, Nu, Nv, Np, nx_fluid;
  std::size_t off_w, off_wd, off_wdd, off_u, off_v, off_p, off_vin;
  std::size_t D;
};

CoupledDims coupled_dims(const CoupledCase& c) {
  CoupledDims d{};
  d.m_beam = c.nx_beam - 1;
  d.Nu = (c.nx_fluid + 1) * c.ny_fluid;
  d.Nv = c.nx_fluid * (c.ny_fluid + 1);
  d.Np = c.nx_fluid * c.ny_fluid;
  d.nx_fluid = c.nx_fluid;
  d.off_w = 0;
  d.off_wd = d.off_w + d.m_beam;
  d.off_wdd = d.off_wd + d.m_beam;
  d.off_u = d.off_wdd + d.m_beam;
  d.off_v = d.off_u + d.Nu;
  d.off_p = d.off_v + d.Nv;
  d.off_vin = d.off_p + d.Np;
  d.D = d.off_vin + d.nx_fluid;
  return d;
}

/// Builds a fresh graph/runner/components, seeds the FULL carried state
/// (beam interior w/wdot/wddot, fluid u/v/p, and fluid's own checkpointed
/// velocity_in_t0_ -- see the file header for why that last one is part of
/// the state) from `x`, runs exactly ONE coupling cycle, and returns the resulting
/// carried state in the same layout.
Eigen::VectorXd run_one_coupled_period(const CoupledCase& c, const CoupledDims& d,
                                       const Eigen::VectorXd& x) {
  // Stage-by-stage tracing used to isolate this round's use-after-free
  // (fixed: FluidComponent::set_velocity_in_for_test, fsi_components.hpp)
  // and a checkpoint-consistency failure at an over-tight tol (fixed: this
  // function's own `c.tol`/`c.max_iterations`, plumbed from CoupledCase
  // rather than hardcoded). Left as a compile-time-eliminated switch,
  // cheap to flip back on for the next person who needs it.
  constexpr bool kTrace = false;
  if (kTrace) { std::printf("  [trace] build_fsi_graph\n"); std::fflush(stdout); }
  fuse::CouplingGraph graph = build_fsi_graph(c);
  fuse::SerialMeshBackend backend;
  if (kTrace) { std::printf("  [trace] mesh load\n"); std::fflush(stdout); }
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) throw std::runtime_error("mesh load: " + r.error().message);
  }
  if (kTrace) { std::printf("  [trace] lower\n"); std::fflush(stdout); }
  const auto lowered = lower(graph);
  if (!lowered.ok()) throw std::runtime_error("lower: " + lowered.diags.to_string());
  if (kTrace) { std::printf("  [trace] schedule\n"); std::fflush(stdout); }
  const fuse::ClockSet clocks{graph};
  auto sched = fuse::schedule(graph, lowered.dag, clocks,
                              fuse::make_region_sizes(graph, backend));
  if (!sched.ok()) throw std::runtime_error("schedule: " + sched.diags.to_string());
  if (kTrace) { std::printf("  [trace] scheduled ok\n"); std::fflush(stdout); }

  const double H = c.H, Umax = c.Umax;
  const auto inflow = [Umax, H](double y) { return 4.0 * Umax * y * (H - y) / (H * H); };

  fuse::ComponentSet components;
  auto beam = std::make_unique<fsi::BeamComponent>(
      "beam", fsi::BeamParams{.n = c.nx_beam, .L = c.L, .EI = c.EI,
                              .rho_s = c.rho_s, .h_s = c.h_s});
  auto fluid = std::make_unique<fsi::FluidComponent>(
      "fluid",
      fsi::ChannelParams{.nx = c.nx_fluid, .ny = c.ny_fluid, .L = c.L,
                        .H = c.H, .mu = c.mu, .rho = c.rho_f},
      inflow);
  fsi::BeamComponent* beam_ptr = beam.get();
  fsi::FluidComponent* fluid_ptr = fluid.get();
  components.add(std::move(beam));
  components.add(std::move(fluid));

  if (kTrace) { std::printf("  [trace] register_fields\n"); std::fflush(stdout); }
  fuse::FieldRegistry registry;
  for (auto& comp : components) {
    const auto r = comp.register_fields(registry, graph);
    if (!r.ok()) throw std::runtime_error("register_fields: " + r.error().message);
  }
  fuse::RunnerConfig cfg;
  cfg.abort_on_nonconvergence = true;
  if (kTrace) { std::printf("  [trace] Runner ctor\n"); std::fflush(stdout); }
  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (kTrace) { std::printf("  [trace] runner.initialize\n"); std::fflush(stdout); }
  if (const auto r = runner.initialize(components, registry, cfg); !r.ok()) {
    throw std::runtime_error("initialize: " + r.error().message);
  }
  if (kTrace) { std::printf("  [trace] initialized ok\n"); std::fflush(stdout); }

  // Seed the full carried state from x.
  std::vector<double> w(c.nx_beam + 1, 0.0), wd(c.nx_beam + 1, 0.0),
      wdd(c.nx_beam + 1, 0.0);
  for (std::size_t i = 0; i < d.m_beam; ++i) {
    w[i + 1] = x[static_cast<Eigen::Index>(d.off_w + i)];
    wd[i + 1] = x[static_cast<Eigen::Index>(d.off_wd + i)];
    wdd[i + 1] = x[static_cast<Eigen::Index>(d.off_wdd + i)];
  }
  beam_ptr->solver().set_state(w, wd, wdd);
  beam_ptr->refresh_velocity_from_current_state();

  fsi::FlowState fs;
  fs.u.assign(d.Nu, 0.0);
  fs.v.assign(d.Nv, 0.0);
  fs.p.assign(d.Np, 0.0);
  for (std::size_t i = 0; i < d.Nu; ++i) fs.u[i] = x[static_cast<Eigen::Index>(d.off_u + i)];
  for (std::size_t i = 0; i < d.Nv; ++i) fs.v[i] = x[static_cast<Eigen::Index>(d.off_v + i)];
  for (std::size_t i = 0; i < d.Np; ++i) fs.p[i] = x[static_cast<Eigen::Index>(d.off_p + i)];
  fluid_ptr->solver().set_state(fs);
  fluid_ptr->refresh_force_from_current_state();

  std::vector<double> vin(d.nx_fluid, 0.0);
  for (std::size_t i = 0; i < d.nx_fluid; ++i) {
    vin[i] = x[static_cast<Eigen::Index>(d.off_vin + i)];
  }
  fluid_ptr->set_velocity_in_for_test(vin);
  if (kTrace) { std::printf("  [trace] state seeded, calling run_period\n"); std::fflush(stdout); }

  const auto r = runner.run_period(components);
  if (!r.ok()) throw std::runtime_error("run_period: " + r.error().message);
  if (kTrace) { std::printf("  [trace] run_period ok\n"); std::fflush(stdout); }

  Eigen::VectorXd y(static_cast<Eigen::Index>(d.D));
  const auto& W = beam_ptr->solver().displacement();
  const auto& WD = beam_ptr->solver().velocity();
  const auto& WDD = beam_ptr->solver().acceleration();
  for (std::size_t i = 0; i < d.m_beam; ++i) {
    y[static_cast<Eigen::Index>(d.off_w + i)] = W[i + 1];
    y[static_cast<Eigen::Index>(d.off_wd + i)] = WD[i + 1];
    y[static_cast<Eigen::Index>(d.off_wdd + i)] = WDD[i + 1];
  }
  const auto& U = fluid_ptr->solver().u();
  const auto& V = fluid_ptr->solver().v();
  const auto& P = fluid_ptr->solver().p();
  for (std::size_t i = 0; i < d.Nu; ++i) y[static_cast<Eigen::Index>(d.off_u + i)] = U[i];
  for (std::size_t i = 0; i < d.Nv; ++i) y[static_cast<Eigen::Index>(d.off_v + i)] = V[i];
  for (std::size_t i = 0; i < d.Np; ++i) y[static_cast<Eigen::Index>(d.off_p + i)] = P[i];
  const auto& Vin = fluid_ptr->velocity_in();
  for (std::size_t i = 0; i < d.nx_fluid; ++i) {
    y[static_cast<Eigen::Index>(d.off_vin + i)] = Vin[i];
  }
  return y;
}

Eigen::MatrixXd coupled_map(const CoupledCase& c, double eps) {
  const CoupledDims d = coupled_dims(c);
  return build_affine_map(
      d.D, eps, [&](const Eigen::VectorXd& x) { return run_one_coupled_period(c, d, x); });
}

/// Sums |eigenvector component|^2 within each named block, normalised to
/// fractions of the total -- attributes a mode to physical blocks without
/// requiring the reader to interpret 35 raw complex numbers.
void print_eigenvector_block_breakdown(const CoupledDims& d,
                                       const Eigen::VectorXcd& v) {
  struct Block {
    const char* name;
    std::size_t off, n;
  };
  const Block blocks[] = {
      {"beam w", d.off_w, d.m_beam},       {"beam wdot", d.off_wd, d.m_beam},
      {"beam wddot", d.off_wdd, d.m_beam}, {"fluid u", d.off_u, d.Nu},
      {"fluid v", d.off_v, d.Nv},          {"fluid p", d.off_p, d.Np},
      {"velocity_in_t0", d.off_vin, d.nx_fluid},
  };
  double total = 0.0;
  for (Eigen::Index i = 0; i < v.size(); ++i) total += std::norm(v[i]);
  for (const Block& b : blocks) {
    double s = 0.0;
    for (std::size_t i = 0; i < b.n; ++i) {
      s += std::norm(v[static_cast<Eigen::Index>(b.off + i)]);
    }
    std::printf("    %-16s %.4f\n", b.name, total > 0.0 ? s / total : 0.0);
  }
}

}  // namespace

TEST(FsiEnergyDiag, BeamAloneStepMapSpectralRadiusIsAtMostOne) {
  // Same material parameters as the D9 measurement (mass_ratio 2.5).
  constexpr double L = 1.0, EI = 2.0, rho_s = 1.0, h_s = 0.5;
  const double omega1 = fsi::clamped_beam_first_frequency(L, EI, rho_s, h_s);
  const double T1 = 2.0 * M_PI / omega1;
  const double dt = T1 / 40.0;
  fsi::BeamParams p{.n = 8, .L = L, .EI = EI, .rho_s = rho_s, .h_s = h_s};

  const Eigen::MatrixXd M1 = beam_alone_map(p, dt, 1.0e-3);
  const Eigen::MatrixXd M2 = beam_alone_map(p, dt, 1.0e-5);
  const double linearity_gap = (M1 - M2).cwiseAbs().maxCoeff();
  std::printf("beam-alone map: %lldx%lld, linearity check |M(eps1)-M(eps2)| = %.3e\n",
             static_cast<long long>(M1.rows()), static_cast<long long>(M1.cols()),
             linearity_gap);
  EXPECT_LT(linearity_gap, 1e-8) << "map is not exactly affine at round-off -- "
                                  "epsilon-dependence would invalidate the "
                                  "eigenvalue reading";

  const double rho = spectral_radius(M1);
  std::printf("beam-alone step map spectral radius: %.10f (undamped -> expect "
             "<= 1 + round-off)\n",
             rho);
  EXPECT_LE(rho, 1.0 + 1e-6) << "beam alone is unconditionally stable (Newmark "
                             "average-acceleration); rho=" << rho
                             << " would contradict the 80-period Emax/E0=1.0000 "
                                "measurement in docs/user/fsi-channel.md";
}

TEST(FsiEnergyDiag, FluidAloneStepMapSpectralRadiusIsAtMostOne) {
  constexpr double L = 1.0, EI = 2.0, rho_s = 1.0, h_s = 0.5;
  const double omega1 = fsi::clamped_beam_first_frequency(L, EI, rho_s, h_s);
  const double dt = (2.0 * M_PI / omega1) / 40.0;
  fsi::ChannelParams p{.nx = 10, .ny = 6, .L = L, .H = 0.2, .mu = 0.01, .rho = 1.0};

  const Eigen::MatrixXd M = fluid_alone_map(p, dt, 1.0e-3);
  const double rho = spectral_radius(M);
  std::printf("fluid-alone (fixed wall velocity) step map spectral radius: "
             "%.10f, dim %lld\n",
             rho, static_cast<long long>(M.rows()));
  EXPECT_LE(rho, 1.0 + 1e-6) << "Crank-Nicolson is unconditionally stable for "
                             "this linear Stokes problem; rho=" << rho
                             << " would contradict the fluid-alone boundedness "
                                "already measured";

  if (std::getenv("FSI_DIAG_DUMP_DIR") != nullptr) {
    const EigResult eig = eig_analysis(M);
    const std::size_t Nu = (p.nx + 1) * p.ny, Nv = p.nx * (p.ny + 1);
    const auto& vec = eig.solver.eigenvectors().col(eig.dominant);
    double mu = 0, mv = 0, mp = 0, tot = 0;
    for (Eigen::Index i = 0; i < vec.size(); ++i) tot += std::norm(vec[i]);
    for (std::size_t i = 0; i < Nu; ++i) mu += std::norm(vec[static_cast<Eigen::Index>(i)]);
    for (std::size_t i = 0; i < Nv; ++i) mv += std::norm(vec[static_cast<Eigen::Index>(Nu + i)]);
    for (std::size_t i = 0; i < static_cast<std::size_t>(vec.size()) - Nu - Nv; ++i) {
      mp += std::norm(vec[static_cast<Eigen::Index>(Nu + Nv + i)]);
    }
    std::printf("fluid-alone dominant |lambda|=%.10f block mass: u=%.4f v=%.4f "
               "p=%.4f\n",
               eig.rho, mu / tot, mv / tot, mp / tot);
  }
}

// D9 round 3: the lead's elimination argument, measured directly (not
// inferred from rho(M)). See measure_energy_identity()'s own header
// comment for the derivation of why R_A and R_B are each independently
// computable without needing a separate bulk-dissipation formula.
TEST(FsiEnergyDiag, EnergyIdentityResidualRevealsWhichHypothesisHolds) {
  fsi::ChannelParams p{.nx = 10, .ny = 6, .L = 1.0, .H = 0.2, .mu = 0.01,
                      .rho = 1.0};
  const double dt = 0.01;
  const FullIdentityResult r = measure_full_energy_identity(p, dt);
  std::printf(
      "E0=%.6e E1=%.6e (E1-E0)=%.6e\n"
      "  interior_viscous_power*dt=%.6e  pressure_div_power*dt=%.6e  "
      "boundary_bt_term*dt=%.6e\n"
      "  ke_change(v_wall)=%.6e  ke_change(u_outflow)=%.6e\n"
      "  CLOSURE RESIDUAL (validation, should be ~0 round-off): %.6e\n"
      "  hypothesis A (top wall), viscous channel only (cleanly local):\n"
      "    W_wall(viscous-only natural, mu, no pressure)=%.6e\n"
      "  hypothesis B (outflow), viscous channel only (cleanly local):\n"
      "    B_out(viscous-only natural, mu, no pressure)=%.6e\n"
      "  reference only -- NOT a clean local split (pressure_div_power is a\n"
      "  GLOBAL quantity, see file header; these combine the local viscous\n"
      "  term with a boundary-attributed pressure piece for comparison\n"
      "  against the CURRENT delivered traction, nothing more):\n"
      "    W_wall(current delivered: extrapolated p, 2mu dv/dy)=%.6e\n"
      "    W_wall(conjugate: adjacent-cell p, mu dv/dy)=%.6e\n"
      "    B_out(natural form WITH pressure, mu)=%.6e\n"
      "    B_out(do-nothing form WITH pressure, 2mu)=%.6e\n",
      r.E0, r.E1, r.E1 - r.E0, dt * r.interior_viscous_power,
      dt * r.pressure_div_power, dt * r.boundary_bt_term, r.ke_change_v_wall,
      r.ke_change_u_outflow, r.closure_residual, r.W_wall_viscous_natural,
      r.B_out_viscous_natural, r.W_wall_current, r.W_wall_conjugate,
      r.B_out_natural_form, r.B_out_donothing_form);
  // Reporting only -- see this task's own submission for what these
  // numbers mean for D9. `closure_residual` is the one number that MUST
  // be near round-off for the rest of this measurement to be trusted (it
  // validates that the direct, mechanical Laplacian/divergence sums above
  // correctly mirror cn_step's own stencils, INCLUDING the boundary_bt_term
  // correction derived and cross-checked this round) -- if it is NOT
  // small, the remaining numbers are not meaningful evidence either way.
  EXPECT_LT(std::abs(r.closure_residual), 1e-9 * std::max(r.E1, 1e-12))
      << "closure_residual should be round-off relative to E1; if this "
         "fails, the mechanical sum no longer mirrors cn_step's own "
         "stencils and the reported numbers below are not trustworthy";
}

// The main diagnostic. Reports the coupled scheme's own step-map spectral
// radius on a small grid (kept small so a dense eigensolve over ~35 states,
// run ~2*D+1 times through the full FUSE graph, stays well inside this
// task's 30s-per-test budget) at the D9 mass ratio and dt, and breaks the
// dominant eigenvector down by physical block. NOT gated on rho<=1: this
// test's job is to MEASURE and REPORT, matching the "Find where" title --
// the fix itself is verified afterwards by FsiTransient's own re-enabled
// bound and by a direct rho(M) check post-fix (see below).
// DISABLED, not loosened: KNOWN_ISSUES.md D9 remains open. Direct, non-
// linearised cross-check of the round's affine-map finding: reproduces the
// D9 long-run measurement (mass_ratio 2.5, dt=T1/40, 480 steps = 12 beam
// periods) and reports the SAME per-step total-energy ratio, averaged over
// periods 3-12 exactly as the original measurement was windowed, with this
// round's outflow mass-conservation correction applied (StokesSolver::step,
// see its own comment). Total energy = beam KE + beam strain energy + fluid
// KE (StokesSolver::kinetic_energy(), added this round).
//
// Measured (this round): geometric-mean per-step ratio 1.226473 over
// periods 3-12 -- statistically the SAME as D9's original, unfixed 1.22796
// (a <0.2% difference, consistent with the affine-map spectral radius
// moving by well under 1% for the same correction). The outflow mass fix
// does NOT resolve D9; see KNOWN_ISSUES.md D9 for what was ruled out and
// what the affine-map eigenvector analysis actually points at. Left
// DISABLED with these numbers, not loosened, exactly as this task's
// standing instruction requires. Re-enable (tightening EXPECT_LE back to a
// bound near 1) once a genuine fix is found -- the bound below documents
// the target, not today's result.
TEST(FsiEnergyDiag, LongRunEnergyIsBounded) {
#if FUSE_HAVE_CUSPARSE
  // KNOWN_ISSUES.md D37: with the cusolverRf fluid solve this 480-period run
  // is not reproducible run to run (traces fork at ~1e-12 after ~70 periods)
  // and 11 of 15 measured runs failed to converge at period 140. Flag off,
  // the same binary is bitwise reproducible and passes. Skipped, not
  // loosened; the other five cases in this binary still run under the flag.
  GTEST_SKIP() << "D37: nondeterministic under FUSE_ENABLE_CUSPARSE (cusolverRf solve)";
#endif
  constexpr double L = 1.0, H = 0.2, mu = 0.01, rho_f = 1.0;
  constexpr double EI = 2.0, rho_s = 1.0, h_s = 0.5;
  constexpr std::size_t nx_beam = 8, nx_fluid = 10, ny_fluid = 6;
  const double omega1 = fsi::clamped_beam_first_frequency(L, EI, rho_s, h_s);
  const double T1 = 2.0 * M_PI / omega1;
  const double dt = T1 / 40.0;
  constexpr std::size_t n_steps = 480;  // 12 beam periods, D9's own window

  CoupledCase c;
  c.nx_beam = nx_beam;
  c.nx_fluid = nx_fluid;
  c.ny_fluid = ny_fluid;
  c.L = L;
  c.H = H;
  c.mu = mu;
  c.rho_f = rho_f;
  c.EI = EI;
  c.rho_s = rho_s;
  c.h_s = h_s;
  c.dt = dt;
  c.tol = 1e-9;
  c.max_iterations = 300;

  fuse::CouplingGraph graph = build_fsi_graph(c);
  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    ASSERT_TRUE(backend.load(graph.mesh(fuse::MeshId{i})).ok());
  }
  const auto lowered = lower(graph);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet clocks{graph};
  auto sched = fuse::schedule(graph, lowered.dag, clocks,
                              fuse::make_region_sizes(graph, backend));
  ASSERT_TRUE(sched.ok());

  const auto inflow = [H](double y) { return 4.0 * 1.0 * y * (H - y) / (H * H); };
  fuse::ComponentSet components;
  auto beam = std::make_unique<fsi::BeamComponent>(
      "beam", fsi::BeamParams{.n = nx_beam, .L = L, .EI = EI, .rho_s = rho_s,
                              .h_s = h_s});
  auto fluid = std::make_unique<fsi::FluidComponent>(
      "fluid",
      fsi::ChannelParams{.nx = nx_fluid, .ny = ny_fluid, .L = L, .H = H,
                        .mu = mu, .rho = rho_f},
      inflow);
  fsi::BeamComponent* beam_ptr = beam.get();
  fsi::FluidComponent* fluid_ptr = fluid.get();
  components.add(std::move(beam));
  components.add(std::move(fluid));

  fuse::FieldRegistry registry;
  for (auto& comp : components) {
    ASSERT_TRUE(comp.register_fields(registry, graph).ok());
  }
  fuse::RunnerConfig cfg;
  cfg.abort_on_nonconvergence = true;
  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  ASSERT_TRUE(runner.initialize(components, registry, cfg).ok());

  const auto total_energy = [&]() {
    return beam_ptr->solver().kinetic_energy() + beam_ptr->solver().strain_energy() +
           fluid_ptr->solver().kinetic_energy();
  };

  std::vector<double> energy(n_steps + 1);
  energy[0] = total_energy();
  for (std::size_t s = 0; s < n_steps; ++s) {
    const auto r = runner.run_period(components);
    ASSERT_TRUE(r.ok()) << "step " << s << ": " << r.error().message;
    energy[s + 1] = total_energy();
  }

  // Same window as the original measurement: periods 3-12 (steps 120-480
  // at 40 steps/period), geometric-mean per-step ratio over that window.
  const std::size_t lo = 120, hi = n_steps;
  double log_ratio_sum = 0.0;
  std::size_t n_ratios = 0;
  for (std::size_t s = lo; s < hi; ++s) {
    if (energy[s] > 0.0 && energy[s + 1] > 0.0) {
      log_ratio_sum += std::log(energy[s + 1] / energy[s]);
      ++n_ratios;
    }
  }
  const double mean_ratio = std::exp(log_ratio_sum / static_cast<double>(n_ratios));
  std::printf("energy[0]=%.6e energy[120]=%.6e energy[480]=%.6e\n", energy[0],
             energy[120], energy[480]);
  std::printf("geometric-mean per-step energy ratio over periods 3-12: %.6f "
             "(D9's unfixed measurement: 1.22796)\n",
             mean_ratio);
  EXPECT_LE(mean_ratio, 1.01)
      << "energy still grows by " << mean_ratio << " per step -- D9 not fixed";
}

TEST(FsiEnergyDiag, CoupledStepMapRevealsTheGrowthMechanism) {
  constexpr double L = 1.0, EI = 2.0, rho_s = 1.0, h_s = 0.5;
  const double omega1 = fsi::clamped_beam_first_frequency(L, EI, rho_s, h_s);
  const double T1 = 2.0 * M_PI / omega1;

  CoupledCase c;
  c.nx_beam = 6;
  c.nx_fluid = 5;
  c.ny_fluid = 3;
  c.L = L;
  c.H = 0.2;
  c.mu = 0.01;
  c.rho_f = 1.0;
  c.EI = EI;
  c.rho_s = rho_s;
  c.h_s = h_s;
  c.dt = T1 / 40.0;
  c.tol = 1e-9;
  c.max_iterations = 500;

  const CoupledDims d = coupled_dims(c);
  std::printf("coupled state dim: %zu (beam interior %zu x3, fluid u=%zu v=%zu "
             "p=%zu, velocity_in_t0=%zu)\n",
             d.D, d.m_beam, d.Nu, d.Nv, d.Np, d.nx_fluid);
  std::fflush(stdout);

  // Sanity check the plumbing on the UNPERTURBED (zero) state before ever
  // building a finite-difference column: isolates a graph/registration bug
  // from a perturbation/indexing bug.
  try {
    const Eigen::VectorXd x0 = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(d.D));
    const Eigen::VectorXd b = run_one_coupled_period(c, d, x0);
    std::printf("baseline (zero-state) run ok, |b| = %.6e\n", b.norm());
    std::fflush(stdout);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "baseline run threw: %s\n", e.what());
    std::fflush(stderr);
    throw;
  }

  const Eigen::MatrixXd M1 = coupled_map(c, 1.0e-3);
  std::printf("M1 built ok\n");
  std::fflush(stdout);
  const Eigen::MatrixXd M2 = coupled_map(c, 1.0e-4);
  std::printf("M2 built ok\n");
  std::fflush(stdout);
  const double linearity_gap = (M1 - M2).cwiseAbs().maxCoeff();
  std::printf("coupled map linearity check |M(eps1)-M(eps2)| = %.3e (tol driving "
             "sweeps is %.1e)\n",
             linearity_gap, c.tol);

  const EigResult eig = eig_analysis(M1);
  std::printf("coupled step map spectral radius: %.6f  (per-period ENERGY "
             "ratio this would predict, rho^2: %.6f)\n",
             eig.rho, eig.rho * eig.rho);
  std::printf("dominant eigenvalue: (%.6f, %.6fi), |.| = %.6f\n",
             eig.solver.eigenvalues()[eig.dominant].real(),
             eig.solver.eigenvalues()[eig.dominant].imag(), eig.rho);
  std::printf("  dominant eigenvector mass by block (fraction of total |v|^2):\n");
  print_eigenvector_block_breakdown(d, eig.solver.eigenvectors().col(eig.dominant));

  // Cross-check against a real multi-period run on this SAME small grid: if
  // rho(M) truly is the per-period growth factor, N periods from a generic
  // (here: the eigenvector-aligned) perturbation should grow like rho^N.
  // KNOWN_ISSUES.md D9, fixed 2026-09-25: with the wall traction carrying
  // the force ON the wall (it carried the force on the fluid, a negative
  // added mass), the coupled step map is neutrally stable -- rho = 1.000000
  // measured, the |lambda| = 1 mode being Newmark's acceleration passenger
  // (wddot_{n+1} = 2(v_{n+1}-v_n)/dt - wddot_n, which no other state reads).
  // The bound is the finite-difference construction's own accuracy; the
  // defect it guards against measured 1.115466 on this very grid.
  EXPECT_LE(eig.rho, 1.0 + 1e-6)
      << "the coupled step map amplifies: rho = " << eig.rho
      << " (D9's defect gave 1.115466 here)";

  if (const char* dump_dir = std::getenv("FSI_DIAG_DUMP_DIR")) {
    const std::string base = dump_dir;
    {
      std::FILE* f = std::fopen((base + "/M.csv").c_str(), "w");
      for (Eigen::Index i = 0; i < M1.rows(); ++i) {
        for (Eigen::Index j = 0; j < M1.cols(); ++j) {
          std::fprintf(f, "%.17g%s", M1(i, j), j + 1 < M1.cols() ? "," : "\n");
        }
      }
      std::fclose(f);
    }
    {
      std::FILE* f = std::fopen((base + "/eigvec.csv").c_str(), "w");
      const auto& vec = eig.solver.eigenvectors().col(eig.dominant);
      for (Eigen::Index i = 0; i < vec.size(); ++i) {
        std::fprintf(f, "%.17g,%.17g\n", vec[i].real(), vec[i].imag());
      }
      std::fclose(f);
    }
    {
      std::FILE* f = std::fopen((base + "/dims.txt").c_str(), "w");
      std::fprintf(f,
                   "m_beam=%zu\nNu=%zu\nNv=%zu\nNp=%zu\nnx_fluid=%zu\n"
                   "ny_fluid=%zu\noff_w=%zu\noff_wd=%zu\noff_wdd=%zu\n"
                   "off_u=%zu\noff_v=%zu\noff_p=%zu\noff_vin=%zu\nD=%zu\n"
                   "rho=%.17g\n",
                   d.m_beam, d.Nu, d.Nv, d.Np, d.nx_fluid, c.ny_fluid, d.off_w,
                   d.off_wd, d.off_wdd, d.off_u, d.off_v, d.off_p, d.off_vin,
                   d.D, eig.rho);
      std::fclose(f);
    }
    std::printf("dumped M, eigvec, dims to %s\n", base.c_str());
  }
}

// ============================================================================
// ROUND 4: full discrete energy budget evaluated ON THE UNSTABLE MODE ITSELF.
// ============================================================================
// Round 3 validated a fluid-alone energy identity from a hand-picked (rest,
// synthetic wall velocity) initial condition. This round evaluates the SAME
// kind of identity -- extended to the FULL coupled system (fluid + beam,
// via the beam's own exact Newmark energy identity) and to the mass
// correction that sits outside `cn_step`'s own saddle point -- but on the
// coupled map `M`'s OWN dominant eigenvector, run through the ACTUAL
// production Runner/IQN-ILS path for one coupling cycle. Zero inflow (`Umax=0`)
// makes the coupled step's additive term `b` vanish exactly (the beam has
// no forcing of its own), so a state seeded to an eigenvector `x` of the
// (purely linear, since `b=0`) map should come back as `lambda*x` to
// solver tolerance -- letting every term below be read off one concrete,
// reproducible run rather than a linearization.

/// Generalises `measure_full_energy_identity` (which hard-codes a rest
/// initial fluid state and a synthetic top-wall velocity, both convenient
/// for round 3's specific elimination argument but not applicable to an
/// arbitrary carried state) to ARBITRARY t0 fluid data and ARBITRARY wall
/// velocities. Same mechanical-sum method and stencils (mirrors `cn_step`
/// exactly, not any closed-form shortcut); the one substantive fix
/// relative to the round-3 version is `ke_change_u_outflow`, which must
/// become u1^2-u0^2 once u0 is no longer assumed zero. Always run with
/// zero inflow (`bcs.left.u=0`) matching round 4's `Umax=0` diagnostic --
/// NOT a general-Umax fluid identity. Returns the RAW (uncorrected)
/// `cn_step` solution alongside the budget so the caller can isolate any
/// post-solve correction (StokesSolver's own outflow mass-conservation
/// shift) separately -- that correction lives OUTSIDE this function by
/// construction, since it is applied AFTER `cn_step` returns.
struct GeneralIdentityResult {
  FullIdentityResult id;
  std::vector<double> u_raw, v_raw, p_raw;  // cn_step's own, uncorrected
};

GeneralIdentityResult measure_full_energy_identity_general(
    const fsi::ChannelParams& c, double dt, const std::vector<double>& u0,
    const std::vector<double>& v0_in, const std::vector<double>& v_wall_old,
    const std::vector<double>& v_wall_new) {
  using fsi::detail::idx_p;
  using fsi::detail::idx_u;
  using fsi::detail::idx_v;
  const std::size_t nx = c.nx, ny = c.ny;
  std::vector<double> v0 = v0_in;
  // v0's own wall slot (j=ny) must carry the t0 wall velocity -- cn_step's
  // explicit-side Laplacian reads it as row j=ny-1's north neighbour at
  // t0 (round 3's bug #1). The caller's v0 is the carried-state array,
  // which (like `run_one_coupled_period`'s own v0 seeding) does NOT
  // necessarily have this slot populated with the correct t0 value, so
  // set it explicitly here, exactly as round 3's version did.
  for (std::size_t i = 0; i < nx; ++i) v0[idx_v(c, i, ny)] = v_wall_old[i];
  const double dx = c.L / static_cast<double>(nx);
  const double dy = c.H / static_cast<double>(ny);
  const double t0 = 0.0, t1 = dt, t_mid = 0.5 * dt;
  const double mu = c.mu;

  fsi::ChannelBCs bcs;
  bcs.bottom.u = [](double, double) { return 0.0; };
  bcs.bottom.v = [](double, double) { return 0.0; };
  bcs.top.u = [](double, double) { return 0.0; };
  bcs.top.v = [&](double x, double t) {
    const long long ii = std::lround(x / dx - 0.5);
    const std::size_t i = static_cast<std::size_t>(
        std::clamp<long long>(ii, 0, static_cast<long long>(nx) - 1));
    return (t < t_mid) ? v_wall_old[i] : v_wall_new[i];
  };
  bcs.left.u = [](double, double) { return 0.0; };  // Umax=0, round 4's own run
  bcs.left.v = [](double, double) { return 0.0; };
  bcs.right.outflow = true;

  const fsi::PressurePin pin{0, ny - 1, 0.0};
  const fsi::FlowState next =
      fsi::cn_step(c, bcs, fsi::SourceTerms{}, dt, t0, t1, u0, v0, pin,
                  nullptr, nullptr);
  const std::vector<double>& u1 = next.u;
  const std::vector<double>& v1 = next.v;
  const std::vector<double>& p1 = next.p;

  auto uavg = [&](std::size_t i, std::size_t j) {
    return 0.5 * (u1[idx_u(c, i, j)] + u0[idx_u(c, i, j)]);
  };
  auto vavg = [&](std::size_t i, std::size_t j) {
    return 0.5 * (v1[idx_v(c, i, j)] + v0[idx_v(c, i, j)]);
  };
  auto vwall_avg = [&](std::size_t i) {
    return 0.5 * (v_wall_new[i] + v_wall_old[i]);
  };

  double sum0 = 0.0, sum1 = 0.0;
  for (const double u : u0) sum0 += u * u;
  for (const double v : v0) sum0 += v * v;
  for (const double u : u1) sum1 += u * u;
  for (const double v : v1) sum1 += v * v;
  const double cell = dx * dy;

  FullIdentityResult r;
  r.E0 = 0.5 * c.rho * cell * sum0;
  r.E1 = 0.5 * c.rho * cell * sum1;

  double kev = 0.0;
  for (std::size_t i = 0; i < nx; ++i) {
    kev += v_wall_new[i] * v_wall_new[i] - v_wall_old[i] * v_wall_old[i];
  }
  r.ke_change_v_wall = 0.5 * c.rho * cell * kev;

  // FIX relative to round 3's version: u0 is NOT assumed zero here, so
  // this must be the CHANGE u1^2-u0^2, not u1^2 alone.
  double keu0 = 0.0, keu1 = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    const double u0o = u0[idx_u(c, nx, j)], u1o = u1[idx_u(c, nx, j)];
    keu0 += u0o * u0o;
    keu1 += u1o * u1o;
  }
  r.ke_change_u_outflow = 0.5 * c.rho * cell * (keu1 - keu0);

  double visc_power = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 1; i < nx; ++i) {
      auto lap_x = [&](const auto& field, std::size_t jj) {
        return (field(i + 1, jj) - 2.0 * field(i, jj) + field(i - 1, jj)) /
               (dx * dx);
      };
      double lap1, lap0;
      if (j == 0) {
        const double c1 = u1[idx_u(c, i, 0)], c2 = u1[idx_u(c, i, 1)];
        const double lap_y1 = (8.0 * 0.0 - 12.0 * c1 + 4.0 * c2) / (3.0 * dy * dy);
        const double c1_0 = u0[idx_u(c, i, 0)], c2_0 = u0[idx_u(c, i, 1)];
        const double lap_y0 = (8.0 * 0.0 - 12.0 * c1_0 + 4.0 * c2_0) / (3.0 * dy * dy);
        lap1 = lap_x([&](std::size_t ii, std::size_t jj) { return u1[idx_u(c, ii, jj)]; }, 0) + lap_y1;
        lap0 = lap_x([&](std::size_t ii, std::size_t jj) { return u0[idx_u(c, ii, jj)]; }, 0) + lap_y0;
      } else if (j == ny - 1) {
        const double c1 = u1[idx_u(c, i, ny - 1)], c2 = u1[idx_u(c, i, ny - 2)];
        const double lap_y1 = (8.0 * 0.0 - 12.0 * c1 + 4.0 * c2) / (3.0 * dy * dy);
        const double c1_0 = u0[idx_u(c, i, ny - 1)], c2_0 = u0[idx_u(c, i, ny - 2)];
        const double lap_y0 = (8.0 * 0.0 - 12.0 * c1_0 + 4.0 * c2_0) / (3.0 * dy * dy);
        lap1 = lap_x([&](std::size_t ii, std::size_t jj) { return u1[idx_u(c, ii, jj)]; }, ny - 1) + lap_y1;
        lap0 = lap_x([&](std::size_t ii, std::size_t jj) { return u0[idx_u(c, ii, jj)]; }, ny - 1) + lap_y0;
      } else {
        const double lap_y1 = (u1[idx_u(c, i, j + 1)] - 2.0 * u1[idx_u(c, i, j)] +
                               u1[idx_u(c, i, j - 1)]) / (dy * dy);
        const double lap_y0 = (u0[idx_u(c, i, j + 1)] - 2.0 * u0[idx_u(c, i, j)] +
                               u0[idx_u(c, i, j - 1)]) / (dy * dy);
        lap1 = lap_x([&](std::size_t ii, std::size_t jj) { return u1[idx_u(c, ii, jj)]; }, j) + lap_y1;
        lap0 = lap_x([&](std::size_t ii, std::size_t jj) { return u0[idx_u(c, ii, jj)]; }, j) + lap_y0;
      }
      const double lap_avg = 0.5 * (lap1 + lap0);
      visc_power += mu * uavg(i, j) * lap_avg * cell;
    }
  }
  for (std::size_t j = 1; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const bool east_outflow = (i == nx - 1);
      double lap_x1, lap_x0;
      if (east_outflow) {
        lap_x1 = (v1[idx_v(c, i, j)] - 2.0 * v1[idx_v(c, i - 1, j)] +
                 v1[idx_v(c, i - 2, j)]) / (dx * dx);
        lap_x0 = (v0[idx_v(c, i, j)] - 2.0 * v0[idx_v(c, i - 1, j)] +
                 v0[idx_v(c, i - 2, j)]) / (dx * dx);
      } else if (i == 0) {
        const double c1 = v1[idx_v(c, 0, j)], c2 = v1[idx_v(c, 1, j)];
        lap_x1 = (8.0 * 0.0 - 12.0 * c1 + 4.0 * c2) / (3.0 * dx * dx);
        const double c1_0 = v0[idx_v(c, 0, j)], c2_0 = v0[idx_v(c, 1, j)];
        lap_x0 = (8.0 * 0.0 - 12.0 * c1_0 + 4.0 * c2_0) / (3.0 * dx * dx);
      } else {
        lap_x1 = (v1[idx_v(c, i + 1, j)] - 2.0 * v1[idx_v(c, i, j)] +
                 v1[idx_v(c, i - 1, j)]) / (dx * dx);
        lap_x0 = (v0[idx_v(c, i + 1, j)] - 2.0 * v0[idx_v(c, i, j)] +
                 v0[idx_v(c, i - 1, j)]) / (dx * dx);
      }
      double v_np1_1, v_np1_0, v_nm1_1, v_nm1_0;
      v_nm1_1 = v1[idx_v(c, i, j - 1)];
      v_nm1_0 = v0[idx_v(c, i, j - 1)];
      if (j == ny - 1) {
        v_np1_1 = v_wall_new[i];
        v_np1_0 = v_wall_old[i];
      } else {
        v_np1_1 = v1[idx_v(c, i, j + 1)];
        v_np1_0 = v0[idx_v(c, i, j + 1)];
      }
      const double c1v = v1[idx_v(c, i, j)], c0v = v0[idx_v(c, i, j)];
      const double lap_y1 = (v_np1_1 - 2.0 * c1v + v_nm1_1) / (dy * dy);
      const double lap_y0 = (v_np1_0 - 2.0 * c0v + v_nm1_0) / (dy * dy);
      const double lap1 = lap_x1 + lap_y1;
      const double lap0 = lap_x0 + lap_y0;
      const double lap_avg = 0.5 * (lap1 + lap0);
      visc_power += mu * vavg(i, j) * lap_avg * cell;
    }
  }
  r.interior_viscous_power = visc_power;

  double press_power = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const double u_e = uavg(i + 1, j), u_w = uavg(i, j);
      const double v_n = (j == ny - 1) ? vwall_avg(i) : vavg(i, j + 1);
      const double v_s = vavg(i, j);
      const double div_avg = (u_e - u_w) / dx + (v_n - v_s) / dy;
      press_power += p1[idx_p(c, i, j)] * div_avg * cell;
    }
  }
  r.pressure_div_power = press_power;

  double boundary_bt_term = 0.0;
  for (std::size_t i = 0; i < nx; ++i) {
    boundary_bt_term += vwall_avg(i) * p1[idx_p(c, i, ny - 1)] / dy * cell;
  }
  for (std::size_t j = 0; j < ny; ++j) {
    boundary_bt_term += uavg(nx, j) * p1[idx_p(c, nx - 1, j)] / dx * cell;
  }
  r.boundary_bt_term = boundary_bt_term;

  // Same closure identity as `measure_full_energy_identity` above, just
  // generalised to an arbitrary t0 state -- rewritten via `r.boundary_bt_term`
  // (just assigned) instead of re-reading the local `boundary_bt_term`,
  // so this text stays distinct from that function's own mutation-anchor
  // line (`build-evidence/mutations/fsi.toml`'s `fsi-energy-diag-
  // boundary-bt-term-dropped` anchors THAT exact text; duplicating it
  // here made the anchor ambiguous -- `tools/mutate.py --check-anchors`
  // caught it).
  const double e_interior_change_general =
      (r.E1 - r.E0) - r.ke_change_v_wall - r.ke_change_u_outflow;
  r.closure_residual = e_interior_change_general -
                       dt * (r.interior_viscous_power + r.pressure_div_power -
                             r.boundary_bt_term);

  double w_visc_nat = 0.0, w_current = 0.0, w_conjugate = 0.0;
  for (std::size_t i = 0; i < nx; ++i) {
    const double vw_avg = vwall_avg(i);
    const double v_interior_avg = vavg(i, ny - 1);
    const double dvdy_natural_mid = (vw_avg - v_interior_avg) / dy;
    w_visc_nat += dx * mu * dvdy_natural_mid * vw_avg;

    const double p_last = p1[idx_p(c, i, ny - 1)];
    const double p_prev = ny >= 2 ? p1[idx_p(c, i, ny - 2)] : p_last;
    const double p_wall_current = 1.5 * p_last - 0.5 * p_prev;
    const double v_m1_1 = v1[idx_v(c, i, ny - 1)], v_m1_0 = v0[idx_v(c, i, ny - 1)];
    const double v_m2_1 = ny >= 2 ? v1[idx_v(c, i, ny - 2)] : v_m1_1;
    const double v_m2_0 = ny >= 2 ? v0[idx_v(c, i, ny - 2)] : v_m1_0;
    const auto dvdy_2nd = [&](double vwall, double vm1, double vm2) {
      return (3.0 * vwall - 4.0 * vm1 + vm2) / (2.0 * dy);
    };
    const double dvdy_current_mid =
        0.5 * (dvdy_2nd(v_wall_old[i], v_m1_0, v_m2_0) +
              dvdy_2nd(v_wall_new[i], v_m1_1, v_m2_1));
    w_current += dx * (-p_wall_current + 2.0 * mu * dvdy_current_mid) * vw_avg;
    w_conjugate += dx * (-p_last + mu * dvdy_natural_mid) * vw_avg;
  }
  r.W_wall_viscous_natural = dt * w_visc_nat;
  r.W_wall_current = dt * w_current;
  r.W_wall_conjugate = dt * w_conjugate;

  double b_visc_nat = 0.0, b_natural_form = 0.0, b_donothing = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    const double u_out_avg = uavg(nx, j);
    const double u_prev_avg = uavg(nx - 1, j);
    const double dudx_mid = (u_out_avg - u_prev_avg) / dx;
    b_visc_nat += dy * mu * dudx_mid * u_out_avg;
    const double p_prev = p1[idx_p(c, nx - 1, j)];
    b_natural_form += dy * (-p_prev + mu * dudx_mid) * u_out_avg;
    b_donothing += dy * (-p_prev + 2.0 * mu * dudx_mid) * u_out_avg;
  }
  r.B_out_viscous_natural = dt * b_visc_nat;
  r.B_out_natural_form = dt * b_natural_form;
  r.B_out_donothing_form = dt * b_donothing;
  return {r, u1, v1, p1};
}

/// Everything `run_one_coupled_period` returns, PLUS the intermediate
/// component-level state (beam energies, delivered load, fluid t0/t1
/// state and wall velocities) round 4's budget needs -- captured from
/// the SAME real Runner/IQN-ILS run, not a second/replayed one.
struct CoupledPeriodDiag {
  Eigen::VectorXd y_final;
  double beam_KE0 = 0, beam_PE0 = 0, fluid_KE0 = 0;
  double beam_KE1 = 0, beam_PE1 = 0, fluid_KE1 = 0;
  std::vector<double> u0, v0, p0, u1, v1, p1;
  std::vector<double> vwall0, vwall1;      // size nx_fluid
  std::vector<double> wd0, wd1;            // size m_beam (interior nodes)
  std::vector<double> load_in_interior;    // size m_beam, beam's OWN nodes
};

CoupledPeriodDiag run_one_coupled_period_with_budget(const CoupledCase& c,
                                                     const CoupledDims& d,
                                                     const Eigen::VectorXd& x) {
  fuse::CouplingGraph graph = build_fsi_graph(c);
  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) throw std::runtime_error("mesh load: " + r.error().message);
  }
  const auto lowered = lower(graph);
  if (!lowered.ok()) throw std::runtime_error("lower: " + lowered.diags.to_string());
  const fuse::ClockSet clocks{graph};
  auto sched = fuse::schedule(graph, lowered.dag, clocks,
                              fuse::make_region_sizes(graph, backend));
  if (!sched.ok()) throw std::runtime_error("schedule: " + sched.diags.to_string());

  const double H = c.H, Umax = c.Umax;
  const auto inflow = [Umax, H](double y) { return 4.0 * Umax * y * (H - y) / (H * H); };
  fuse::ComponentSet components;
  auto beam = std::make_unique<fsi::BeamComponent>(
      "beam", fsi::BeamParams{.n = c.nx_beam, .L = c.L, .EI = c.EI,
                              .rho_s = c.rho_s, .h_s = c.h_s});
  auto fluid = std::make_unique<fsi::FluidComponent>(
      "fluid",
      fsi::ChannelParams{.nx = c.nx_fluid, .ny = c.ny_fluid, .L = c.L,
                        .H = c.H, .mu = c.mu, .rho = c.rho_f},
      inflow);
  fsi::BeamComponent* beam_ptr = beam.get();
  fsi::FluidComponent* fluid_ptr = fluid.get();
  components.add(std::move(beam));
  components.add(std::move(fluid));

  fuse::FieldRegistry registry;
  for (auto& comp : components) {
    const auto r = comp.register_fields(registry, graph);
    if (!r.ok()) throw std::runtime_error("register_fields: " + r.error().message);
  }
  fuse::RunnerConfig cfg;
  cfg.abort_on_nonconvergence = true;
  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry, cfg); !r.ok()) {
    throw std::runtime_error("initialize: " + r.error().message);
  }

  std::vector<double> w(c.nx_beam + 1, 0.0), wd(c.nx_beam + 1, 0.0),
      wdd(c.nx_beam + 1, 0.0);
  for (std::size_t i = 0; i < d.m_beam; ++i) {
    w[i + 1] = x[static_cast<Eigen::Index>(d.off_w + i)];
    wd[i + 1] = x[static_cast<Eigen::Index>(d.off_wd + i)];
    wdd[i + 1] = x[static_cast<Eigen::Index>(d.off_wdd + i)];
  }
  beam_ptr->solver().set_state(w, wd, wdd);
  beam_ptr->refresh_velocity_from_current_state();

  fsi::FlowState fs;
  fs.u.assign(d.Nu, 0.0);
  fs.v.assign(d.Nv, 0.0);
  fs.p.assign(d.Np, 0.0);
  for (std::size_t i = 0; i < d.Nu; ++i) fs.u[i] = x[static_cast<Eigen::Index>(d.off_u + i)];
  for (std::size_t i = 0; i < d.Nv; ++i) fs.v[i] = x[static_cast<Eigen::Index>(d.off_v + i)];
  for (std::size_t i = 0; i < d.Np; ++i) fs.p[i] = x[static_cast<Eigen::Index>(d.off_p + i)];
  fluid_ptr->solver().set_state(fs);
  fluid_ptr->refresh_force_from_current_state();

  std::vector<double> vin(d.nx_fluid, 0.0);
  for (std::size_t i = 0; i < d.nx_fluid; ++i) {
    vin[i] = x[static_cast<Eigen::Index>(d.off_vin + i)];
  }
  fluid_ptr->set_velocity_in_for_test(vin);

  CoupledPeriodDiag diag;
  diag.beam_KE0 = beam_ptr->solver().kinetic_energy();
  diag.beam_PE0 = beam_ptr->solver().strain_energy();
  diag.fluid_KE0 = fluid_ptr->solver().kinetic_energy();
  diag.u0 = fs.u;
  diag.v0 = fs.v;
  diag.p0 = fs.p;
  diag.vwall0 = vin;
  diag.wd0.assign(d.m_beam, 0.0);
  for (std::size_t i = 0; i < d.m_beam; ++i) diag.wd0[i] = wd[i + 1];

  const auto r = runner.run_period(components);
  if (!r.ok()) throw std::runtime_error("run_period: " + r.error().message);

  diag.beam_KE1 = beam_ptr->solver().kinetic_energy();
  diag.beam_PE1 = beam_ptr->solver().strain_energy();
  diag.fluid_KE1 = fluid_ptr->solver().kinetic_energy();
  diag.u1 = fluid_ptr->solver().u();
  diag.v1 = fluid_ptr->solver().v();
  diag.p1 = fluid_ptr->solver().p();
  diag.vwall1 = fluid_ptr->velocity_in();
  const std::vector<double>& WD1 = beam_ptr->solver().velocity();
  diag.wd1.assign(d.m_beam, 0.0);
  for (std::size_t i = 0; i < d.m_beam; ++i) diag.wd1[i] = WD1[i + 1];
  const std::vector<double>& LOAD = beam_ptr->load_in();
  diag.load_in_interior.assign(d.m_beam, 0.0);
  for (std::size_t i = 0; i < d.m_beam; ++i) diag.load_in_interior[i] = LOAD[i + 1];

  Eigen::VectorXd y(static_cast<Eigen::Index>(d.D));
  const auto& W = beam_ptr->solver().displacement();
  const auto& WDD = beam_ptr->solver().acceleration();
  for (std::size_t i = 0; i < d.m_beam; ++i) {
    y[static_cast<Eigen::Index>(d.off_w + i)] = W[i + 1];
    y[static_cast<Eigen::Index>(d.off_wd + i)] = WD1[i + 1];
    y[static_cast<Eigen::Index>(d.off_wdd + i)] = WDD[i + 1];
  }
  for (std::size_t i = 0; i < d.Nu; ++i) y[static_cast<Eigen::Index>(d.off_u + i)] = diag.u1[i];
  for (std::size_t i = 0; i < d.Nv; ++i) y[static_cast<Eigen::Index>(d.off_v + i)] = diag.v1[i];
  for (std::size_t i = 0; i < d.Np; ++i) y[static_cast<Eigen::Index>(d.off_p + i)] = diag.p1[i];
  for (std::size_t i = 0; i < d.nx_fluid; ++i) {
    y[static_cast<Eigen::Index>(d.off_vin + i)] = diag.vwall1[i];
  }
  diag.y_final = y;
  return diag;
}

TEST(FsiEnergyDiag, FullBudgetOnTheDominantModeCloses) {
  constexpr double L = 1.0, EI = 2.0, rho_s = 1.0, h_s = 0.5;
  const double omega1 = fsi::clamped_beam_first_frequency(L, EI, rho_s, h_s);
  const double T1 = 2.0 * M_PI / omega1;

  // Same small grid as CoupledStepMapRevealsTheGrowthMechanism, so this
  // test's rho reproduces that test's already-validated 1.115466 exactly
  // (a regression check on this round's own CoupledCase::Umax addition:
  // Umax defaults to 1.0, so M is unaffected).
  CoupledCase c;
  c.nx_beam = 6;
  c.nx_fluid = 5;
  c.ny_fluid = 3;
  c.L = L;
  c.H = 0.2;
  c.mu = 0.01;
  c.rho_f = 1.0;
  c.EI = EI;
  c.rho_s = rho_s;
  c.h_s = h_s;
  c.dt = T1 / 40.0;
  c.tol = 1e-12;
  c.max_iterations = 800;

  const CoupledDims d = coupled_dims(c);
  // eps=1e-4 measured (not assumed) to be closer to the finite-difference
  // sweet spot than 1e-5 for THIS c.tol: tightening eps to 1e-5 made the
  // eigenvector residual WORSE (1.09e-6 vs 8.8e-8), the classic FD
  // trade-off between O(eps) truncation and O(c.tol/eps) cancellation --
  // evidence this construction is cancellation-limited by c.tol here, not
  // truncation-limited, which is exactly why the tolerance below is tied
  // to the MEASURED eigenvector residual rather than a fixed number.
  const Eigen::MatrixXd M1 = coupled_map(c, 1.0e-4);
  const EigResult eig = eig_analysis(M1);
  const double lambda_im = eig.solver.eigenvalues()[eig.dominant].imag();
  ASSERT_LT(std::abs(lambda_im), 1e-6)
      << "dominant eigenvalue has a nonzero imaginary part -- the "
         "real-eigenvector, single-run protocol below does not apply";
  const double lambda = eig.solver.eigenvalues()[eig.dominant].real();
  std::printf("dominant eigenvalue lambda = %.9f (rho check: %.9f)\n", lambda,
             eig.rho);
  // Written for D9's real, POSITIVE growing eigenvalue (1.115466). With the
  // traction sign fixed the dominant eigenvalue is -1 (Newmark's acceleration
  // passenger), still real, so the single-run protocol below applies with
  // |lambda| = rho; every identity it checks is quadratic in the state and
  // indifferent to the sign.
  ASSERT_NEAR(std::abs(lambda), eig.rho, 1e-9);

  // The eigenvalue -1 is repeated (one acceleration passenger per beam
  // degree of freedom), so the eigen-decomposition is free to return any
  // basis of that eigenspace, real or complex: the host run happens to get
  // a real vector, the same row under FUSE_EXECUTION_SPACE=device (where
  // the Relax/Remap kernels round differently at the 1e-14 level and the
  // finite-difference columns move with them) got one with |imag| up to
  // 0.23. For a real eigenvalue of a real matrix the real part of any
  // eigenvector is itself an eigenvector, so x is that real part; what the
  // protocol needs is only that it is not negligible, and `eig_quality`
  // below checks that it IS an eigenvector of the actual coupled run.
  Eigen::VectorXd x(static_cast<Eigen::Index>(d.D));
  double max_im = 0.0, vnorm2 = 0.0;
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    const std::complex<double> v = eig.solver.eigenvectors().col(eig.dominant)[i];
    x[i] = v.real();
    max_im = std::max(max_im, std::abs(v.imag()));
    vnorm2 += std::norm(v);
  }
  std::printf("eigenvector max |imag part| = %.3e (a real basis vector of the "
              "repeated eigenvalue is taken as its real part)\n", max_im);
  ASSERT_GT(x.norm(), 0.1 * std::sqrt(vnorm2))
      << "the eigenvector's real part is negligible; pick another basis vector";

  // Round 4's own homogeneous run: Umax=0 so b=0 exactly (only inflow
  // forces this system; the beam has none of its own), letting a SINGLE
  // real production coupling cycle on x reproduce lambda*x directly.
  CoupledCase c0 = c;
  c0.Umax = 0.0;
  const CoupledPeriodDiag diag = run_one_coupled_period_with_budget(c0, d, x);

  const double resid = (diag.y_final - lambda * x).norm();
  const double xnorm = x.norm();
  const double eig_quality = resid / xnorm;
  std::printf("|y_final - lambda*x| / |x| = %.3e (validates x is a genuine "
             "eigenvector under the ACTUAL nonlinear IQN-ILS run, not just "
             "the finite-difference-linearized M)\n",
             eig_quality);
  ASSERT_LT(eig_quality, 1e-6);
  // `eig_quality` becomes the basis for the two downstream closure
  // tolerances below (dE_beam identity, mass-correction isolation): `x`
  // itself carries this much relative error (from `coupled_map`'s
  // finite-difference construction of M, cancellation-limited by
  // `c.tol`, not eps -- measured directly: tightening eps 1e-4->1e-5
  // made eig_quality WORSE, 8.8e-8->1.09e-6, the classic finite-
  // difference cancellation trade-off), so no downstream identity built
  // on `x` can close tighter than a small multiple of THIS number.

  // ---- Total energy bookkeeping ------------------------------------------
  const double E0 = diag.beam_KE0 + diag.beam_PE0 + diag.fluid_KE0;
  const double E1 = diag.beam_KE1 + diag.beam_PE1 + diag.fluid_KE1;
  const double dE_beam = (diag.beam_KE1 + diag.beam_PE1) - (diag.beam_KE0 + diag.beam_PE0);
  const double dE_fluid = diag.fluid_KE1 - diag.fluid_KE0;
  const double predicted_E1 = lambda * lambda * E0;
  std::printf("\n==== ENERGY BOOKKEEPING ====\n");
  std::printf("E0 = %.9e  (beam KE0=%.6e PE0=%.6e, fluid KE0=%.6e)\n", E0,
             diag.beam_KE0, diag.beam_PE0, diag.fluid_KE0);
  std::printf("E1 = %.9e  (beam KE1=%.6e PE1=%.6e, fluid KE1=%.6e)\n", E1,
             diag.beam_KE1, diag.beam_PE1, diag.fluid_KE1);
  std::printf("lambda^2 * E0 = %.9e  (E1 should match this)\n", predicted_E1);
  std::printf("relative mismatch (E1 vs lambda^2*E0) = %.3e\n",
             std::abs(E1 - predicted_E1) / E0);
  std::printf("dE_total = E1-E0 = %.9e ; (lambda^2-1)*E0 = %.9e\n", E1 - E0,
             (lambda * lambda - 1.0) * E0);
  std::printf("dE_beam = %.9e ; dE_fluid = %.9e\n", dE_beam, dE_fluid);
  // With D9 fixed the dominant mode is Newmark's acceleration passenger and
  // carries NO energy: measured E0 = 6.3e-21 at |x| = 1 against
  // eig_quality = 7.7e-11, i.e. E0 is eig_quality^2 -- the energy of the
  // eigenvector's own error, nothing more. Every identity below divides by
  // E0, so on this mode they would compare noise with noise. The assertion
  // that fits the fixed scheme is that the dominant mode IS energetically
  // empty; the budget stays armed and runs in full whenever a mode with
  // energy of its own is dominant again, which is what a returning growth
  // mode would be (D9's carried E0 at the 1e-9 level for the same |x|).
  //
  // The floor is a MAX of two scales, not eig_quality^2 alone. Measured
  // 2026-10-11 with the examples' SparseLuBackend on cusolverRf
  // (FUSE_ENABLE_CUSPARSE=ON, mixed-precision refinement): the same run gave
  // eig_quality = 1.07e-11 -- seven times better than Eigen's 7.7e-11,
  // because that backend's forward error is ~1e-17 against Eigen's ~1e-14 --
  // while E0 stayed at 1.4e-20, the finite-difference construction's own
  // noise energy (set by `c.tol` and the FD step, not by the solve). A floor
  // of eig_quality^2 alone then fell BELOW E0 (1.1e-20 < 1.4e-20) and the
  // budget ran on noise, failing at 3e-4 -- the exact case this guard exists
  // to skip. 1e-16*|x|^2 is seven orders below the energy a real growth
  // mode carried (D9: E0 ~ 1e-9 at the same |x|; the old eig_quality^2 floor
  // sat eleven orders below it) and four orders above the noise, so it
  // cannot hide a returning mode.
  const double energy_floor =
      std::max(100.0 * eig_quality * eig_quality, 1e-16) * xnorm * xnorm;
  if (E0 < energy_floor) {
    std::printf("dominant mode carries no energy (E0 = %.3e < floor %.3e): the "
                "Newmark acceleration passenger; budget not evaluated on noise\n",
                E0, energy_floor);
    EXPECT_LE(eig.rho, 1.0 + 1e-6)
        << "an energetically empty dominant mode with rho > 1 would be a new defect";
    return;
  }
  ASSERT_LT(std::abs(E1 - predicted_E1) / E0, 1e-6)
      << "the run is not behaving as a clean eigenmode -- budget below "
         "would not be trustworthy";

  // ---- Beam-side exact Newmark energy identity ---------------------------
  // Derivation: step_midpoint_load solves M*wddot_avg + K*w_avg = q_mid
  // (mconst*wddot_avg_i + (K*w_avg)_i = q_mid_i per interior node i,
  // mconst=rho_s*h_s, q_mid_i = load_in_i/tributary_length(i) =
  // load_in_i/dx for every INTERIOR node). Dot with wdot_avg_i and sum
  // over i with weight dx (BeamSolver::kinetic_energy/strain_energy's OWN
  // quadrature convention): sum_i dx*mconst*wddot_avg_i*wdot_avg_i =
  // d(KE)/dt exactly (mconst*wddot*wdot = d(0.5*mconst*wdot^2)/dt
  // pointwise); sum_i dx*(K*w_avg)_i*wdot_avg_i = w_avg^T K wdot_avg * dx
  // = d(0.5*dx*w^T K w)/dt via K's symmetry -- but `strain_energy()`
  // itself carries NO dx weight (0.5*w^T*K*w, not 0.5*dx*w^T*K*w), so the
  // dx that appears on the MASS side must NOT appear on the stiffness
  // side for the two to combine into d(KE+PE)/dt. Substituting
  // q_mid_i=load_in_i/dx converts the mass-side dx*q_mid_i back to
  // load_in_i directly (the dx cancels), which is exactly what makes the
  // two sides consistent: dt*sum_i(load_in_i*wdot_avg_i) = dE_beam,
  // load_in_i being the NODAL point force (Newtons), needing no spatial
  // weight of its own. Verified numerically below, not assumed.
  // MEASURED CORRECTION to the derivation above: dotting the recurrence
  // with wdot_avg and summing with a plain (unweighted) sum, as the
  // comment above derives, gives dKE_alt+dPE = dt*qmid.wdot_avg where
  // KE_alt=0.5*mconst*sum(wdot^2) is the UNWEIGHTED counterpart of
  // kinetic_energy() -- i.e. kinetic_energy()'s dx_ weight and
  // strain_energy()'s absence of one are NOT a consistent pair (M in the
  // actual recurrence has no dx either, per this file's own header
  // comment ("M is the constant diagonal rho_s*h_s with no dx factor"),
  // so kinetic_energy()'s dx_ multiplier does NOT match the dynamics'
  // own mass convention the way strain_energy() deliberately mirrors
  // K_'s). Tested directly below: the naive dE_beam=dt*sum(load_in*
  // wdot_avg) identity FAILS (verified to fail first, not assumed to
  // hold); the identity that actually closes is
  // dE_beam = beam_wall_power + (1-dx_beam)*dPE, dx_beam=L/nx_beam --
  // i.e. `strain_energy()` is missing a dx_beam factor to be the SAME
  // physical quadrature kinetic_energy() already uses (0.5*w^T*K*w
  // equals PE_true/dx_beam, not PE_true itself, by the discrete
  // integration-by-parts identity for the pointwise 4th-difference K_).
  // This is a REAL, independent defect in beam_solver.hpp -- but it
  // does NOT move rho(M) (the state-space growth rate rho=1.115466 is
  // computed straight from the carried state, never through kinetic_
  // energy()/strain_energy()), so it cannot be the (or the whole)
  // mechanism sustaining D9; verified with numbers below, not assumed.
  double beam_wall_power = 0.0;
  for (std::size_t i = 0; i < d.m_beam; ++i) {
    const double wdot_avg = 0.5 * (diag.wd0[i] + diag.wd1[i]);
    beam_wall_power += diag.load_in_interior[i] * wdot_avg;
  }
  beam_wall_power *= 1.0;  // dt applied explicitly below, kept as a plain sum here
  const double dx_beam = c.L / static_cast<double>(c.nx_beam);
  const double dPE = diag.beam_PE1 - diag.beam_PE0;
  const double dE_beam_naive = c.dt * beam_wall_power;
  const double dE_beam_predicted = dE_beam_naive + (1.0 - dx_beam) * dPE;
  std::printf("\n==== BEAM-SIDE IDENTITY ====\n");
  std::printf("dx_beam = L/nx_beam = %.9f\n", dx_beam);
  std::printf("beam-side wall power (dt*sum(load_in*wdot_avg))  = %.9e\n",
             dE_beam_naive);
  std::printf("dPE (strain_energy() change, AS CODED, no dx)   = %.9e\n", dPE);
  std::printf("naive identity (dE_beam == wall power) mismatch  = %.3e "
             "(EXPECTED to be large -- this identity is WRONG, see comment "
             "above; kept here as evidence it was tested and failed, not "
             "assumed)\n",
             std::abs(dE_beam - dE_beam_naive));
  std::printf("corrected identity: wall_power + (1-dx_beam)*dPE = %.9e\n",
             dE_beam_predicted);
  std::printf("dE_beam (actual, KE+PE change) = %.9e ; mismatch = %.3e\n",
             dE_beam, std::abs(dE_beam - dE_beam_predicted));
  ASSERT_GT(std::abs(dE_beam - dE_beam_naive), 1e-8)
      << "the NAIVE identity unexpectedly closed -- the dx_beam mismatch "
         "this test documents may not exist on this grid (dx_beam=1 "
         "coincidence?); re-examine before trusting the 'corrected' "
         "identity below";
  // Tolerance DERIVED from eig_quality (measured above), not chosen: `x`
  // is an eigenvector only to relative accuracy eig_quality, so no
  // identity evaluated on it (energies computed from x's own w/wdot
  // blocks) can close tighter than a modest multiple of eig_quality*E0.
  // Measured (this run): mismatch/(eig_quality*E0) = 0.225 -- a safety
  // factor of 5 gives >>20x headroom over the actual mismatch without
  // being loosened past what the eigenvector's own quality can support.
  // The floor, not a chosen slack. `x` is a finite-difference eigenvector
  // accurate to `eig_quality` (measured 8.8e-8) in relative terms, so a
  // quadratic identity evaluated on it cannot close below
  // eig_quality*|E0| however right the algebra is. Factor 1.0: the measured
  // mismatch is 2.118e-12 against a tolerance of 9.4e-12, i.e. 22% of it, so
  // this assertion is load-bearing.
  //
  // Stated plainly because it matters: this is LOOSER than the 1e-9*|E0|+1e-12
  // bound this test carried before the identity was evaluated on an
  // eigenvector at all. That bound was 1.1e-12, below the 2.1e-12 the
  // eigenvector's own accuracy permits, so it was unachievable for a reason
  // now understood rather than merely inconvenient. It is TIGHTER than the
  // 1e-7*|E0| and 5*eig_quality*|E0| values used while that was being worked
  // out.
  const double beam_identity_tol = 1.0 * eig_quality * std::abs(E0);
  std::printf("beam identity tolerance = eig_quality*|E0| = %.3e "
             "(measured mismatch/that ratio = %.3f)\n",
             beam_identity_tol,
             std::abs(dE_beam - dE_beam_predicted) / beam_identity_tol);
  ASSERT_NEAR(dE_beam, dE_beam_predicted, beam_identity_tol)
      << "beam's own exact Newmark energy identity (CORRECTED for the "
         "kinetic_energy()/strain_energy() quadrature mismatch) does not "
         "close -- nothing past this point can be trusted until it does";

  // ---- Fluid-side: isolate the post-solve mass correction ----------------
  const fsi::ChannelParams fc{.nx = c.nx_fluid, .ny = c.ny_fluid, .L = c.L,
                              .H = c.H, .mu = c.mu, .rho = c.rho_f};
  const GeneralIdentityResult gen = measure_full_energy_identity_general(
      fc, c.dt, diag.u0, diag.v0, diag.vwall0, diag.vwall1);
  const std::vector<double>& u_raw = gen.u_raw;
  using fsi::detail::idx_u;
  double max_abs_diff_non_outflow = 0.0, max_abs_diff_outflow = 0.0;
  double correction_sum = 0.0;
  for (std::size_t j = 0; j < c.ny_fluid; ++j) {
    for (std::size_t i = 0; i <= c.nx_fluid; ++i) {
      const double diff = diag.u1[idx_u(fc, i, j)] - u_raw[idx_u(fc, i, j)];
      if (i == c.nx_fluid) {
        max_abs_diff_outflow = std::max(max_abs_diff_outflow, std::abs(diff));
        correction_sum += diff;
      } else {
        max_abs_diff_non_outflow = std::max(max_abs_diff_non_outflow, std::abs(diff));
      }
    }
  }
  const double correction = correction_sum / static_cast<double>(c.ny_fluid);
  double max_correction_deviation = 0.0;
  for (std::size_t j = 0; j < c.ny_fluid; ++j) {
    const double diff = diag.u1[idx_u(fc, c.nx_fluid, j)] - u_raw[idx_u(fc, c.nx_fluid, j)];
    max_correction_deviation = std::max(max_correction_deviation, std::abs(diff - correction));
  }
  // Tolerance DERIVED, not chosen: `u_raw` is an INDEPENDENT `cn_step`
  // solve fed `diag.u0`/`diag.v0`/`diag.vwall0`/`diag.vwall1`, all
  // extracted from the SAME production run that produced `diag.u1` --
  // any mismatch away from the outflow column comes from the SAME
  // eig_quality-level relative error in those inputs (see the beam
  // identity's own comment above), scaled by `u_raw`'s own magnitude,
  // not from a second, independent error source.
  double u_raw_scale = 0.0;
  for (const double v : u_raw) u_raw_scale = std::max(u_raw_scale, std::abs(v));
  // Away from the outflow column the two solves differ only by the solver's
  // own convergence, which is why this is keyed to eig_quality rather than to
  // round-off. Factor 0.5, not 20: the measured difference is 1.115e-10
  // against 1.4e-9, and at the factor of 20 it had been 0.2% of the tolerance,
  // which asserts nothing. This is also tighter than the plain 1e-8 bound it
  // originally replaced.
  const double mass_corr_tol = 0.5 * eig_quality * u_raw_scale;
  std::printf("\n==== MASS-CORRECTION ISOLATION ====\n");
  std::printf("max|u_raw| (velocity scale for the tolerance below) = %.3e\n",
             u_raw_scale);
  std::printf("mass-correction isolation tolerance = 0.5*eig_quality*max|u_raw| "
             "= %.3e\n",
             mass_corr_tol);
  std::printf("max|u1-u_raw| away from outflow column = %.3e (must be ~0: "
             "correction touches ONLY the outflow column; ratio to "
             "tolerance = %.3f)\n",
             max_abs_diff_non_outflow, max_abs_diff_non_outflow / mass_corr_tol);
  std::printf("outflow correction shift = %.9e (uniform across j to within "
             "%.3e, ratio to tolerance = %.3f)\n",
             correction, max_correction_deviation,
             max_correction_deviation / mass_corr_tol);
  ASSERT_LT(max_abs_diff_non_outflow, mass_corr_tol);
  ASSERT_LT(max_correction_deviation, mass_corr_tol);

  ASSERT_LT(std::abs(gen.id.closure_residual), 1e-9 * std::abs(gen.id.E1))
      << "the RAW (uncorrected) cn_step solution's own energy identity "
         "does not close -- the mechanical sum above does not correctly "
         "mirror cn_step's stencils on this arbitrary (non-rest) state";

  double correction_energy = 0.0;
  for (std::size_t j = 0; j < c.ny_fluid; ++j) {
    const double u1o = diag.u1[idx_u(fc, c.nx_fluid, j)];
    const double uro = u_raw[idx_u(fc, c.nx_fluid, j)];
    correction_energy += u1o * u1o - uro * uro;
  }
  const double cellf = (c.L / c.nx_fluid) * (c.H / c.ny_fluid);
  correction_energy *= 0.5 * c.rho_f * cellf;

  const double dt = c.dt;
  const double bulk_dissipation =
      gen.id.W_wall_viscous_natural + gen.id.B_out_viscous_natural -
      dt * gen.id.interior_viscous_power;
  std::printf("\n==== FLUID-SIDE BUDGET (evaluated on the RAW, uncorrected "
             "cn_step solution; correction handled as a separate term) "
             "====\n");
  std::printf("interior_viscous_power*dt         = %+.9e\n", dt * gen.id.interior_viscous_power);
  std::printf("pressure_div_power*dt (GLOBAL, per round 3, not locally "
             "attributable) = %+.9e\n",
             dt * gen.id.pressure_div_power);
  std::printf("boundary_bt_term*dt                = %+.9e\n", dt * gen.id.boundary_bt_term);
  std::printf("  -> bulk viscous dissipation (W_wall_nat+B_out_nat-interior) "
             "= %+.9e (should be POSITIVE: it removes energy)\n",
             bulk_dissipation);
  std::printf("ke_change_v_wall (top wall KE, RAW soln)   = %+.9e\n", gen.id.ke_change_v_wall);
  std::printf("ke_change_u_outflow (outflow KE, RAW soln) = %+.9e\n", gen.id.ke_change_u_outflow);
  std::printf("correction_energy (mass correction's OWN "
             "KE contribution, added AFTER the saddle-point solve) = %+.9e\n",
             correction_energy);
  std::printf("W_wall_viscous_natural*1 (already *dt)     = %+.9e\n", gen.id.W_wall_viscous_natural);
  std::printf("B_out_viscous_natural*1 (already *dt)      = %+.9e\n", gen.id.B_out_viscous_natural);
  std::printf("W_wall_current (production traction form, WITH pressure, "
             "already *dt) = %+.9e\n",
             gen.id.W_wall_current);
  std::printf("B_out_natural_form (already *dt)           = %+.9e\n", gen.id.B_out_natural_form);
  std::printf("B_out_donothing_form (already *dt)         = %+.9e\n", gen.id.B_out_donothing_form);
  std::printf("gen.id.closure_residual (validation, should be ~0) = %.3e\n",
             gen.id.closure_residual);

  const double dE_fluid_predicted =
      dt * (gen.id.interior_viscous_power + gen.id.pressure_div_power -
           gen.id.boundary_bt_term) +
      gen.id.ke_change_v_wall + gen.id.ke_change_u_outflow + correction_energy;
  std::printf("\ndE_fluid (actual, KE change)                  = %+.9e\n", dE_fluid);
  std::printf("dE_fluid (predicted: RAW identity + correction_energy) = %+.9e\n",
             dE_fluid_predicted);
  std::printf("mismatch = %.3e\n", std::abs(dE_fluid - dE_fluid_predicted));
  ASSERT_NEAR(dE_fluid, dE_fluid_predicted, 1e-9 * std::abs(E0) + 1e-12)
      << "fluid-side identity (RAW cn_step budget + isolated mass "
         "correction) does not close against the ACTUAL (corrected) fluid "
         "KE change";

  // ---- Fluid-side vs beam-side delivered power (H/H^T adjointness) -------
  // The beam receives `beam_wall_power` on its OWN nx_beam-resolution
  // grid; the fluid's traction (using the ACTUAL production formula,
  // W_wall_current) is computed on the FLUID's own, generally DIFFERENT,
  // nx_fluid-resolution grid. If `H` (velocity, fluid->beam) and `H^T D`
  // (force, beam->fluid, "conservative") are not exact adjoints under
  // their respective quadratures -- entirely possible whenever
  // nx_beam != nx_fluid, independent of any of the three named
  // candidates -- energy can leak or be created at the interface itself.
  std::printf("\n==== INTERFACE ADJOINTNESS CHECK (H vs H^T D) ====\n");
  std::printf("power beam RECEIVES (dt*sum(load_in*wdot_avg), beam's own "
             "grid, nx_beam=%zu) = %+.9e\n",
             c.nx_beam, dE_beam_naive);
  std::printf("power fluid DELIVERS (W_wall_current, production traction, "
             "fluid's own grid, nx_fluid=%zu) = %+.9e\n",
             c.nx_fluid, gen.id.W_wall_current);
  std::printf("mismatch (beam received - fluid delivered) = %+.9e (nonzero "
             "here is a genuine, DIFFERENT candidate mechanism from the "
             "three named ones -- a non-adjoint interface remap -- since "
             "nx_beam=%zu != nx_fluid=%zu)\n",
             dE_beam_naive - gen.id.W_wall_current, c.nx_beam, c.nx_fluid);

  // ---- div(u_{n+1}) cell-by-cell, and which rows are NOT continuity ------
  using fsi::detail::idx_v;
  const double dxf = c.L / static_cast<double>(c.nx_fluid);
  const double dyf = c.H / static_cast<double>(c.ny_fluid);
  std::printf("\n==== div(u_1) CELL-BY-CELL (actual, corrected solution) "
             "====\n");
  double max_div_interior = 0.0, max_div_outflow_col = 0.0;
  for (std::size_t j = 0; j < c.ny_fluid; ++j) {
    std::string row = "  j=" + std::to_string(j) + ": ";
    for (std::size_t i = 0; i < c.nx_fluid; ++i) {
      const double u_e = diag.u1[idx_u(fc, i + 1, j)], u_w = diag.u1[idx_u(fc, i, j)];
      const double v_n = (j == c.ny_fluid - 1) ? diag.vwall1[i] : diag.v1[idx_v(fc, i, j + 1)];
      const double v_s = diag.v1[idx_v(fc, i, j)];
      const double div = (u_e - u_w) / dxf + (v_n - v_s) / dyf;
      row += " " + std::to_string(div);
      if (i == c.nx_fluid - 1) {
        max_div_outflow_col = std::max(max_div_outflow_col, std::abs(div));
      } else {
        max_div_interior = std::max(max_div_interior, std::abs(div));
      }
    }
    std::printf("%s\n", row.c_str());
  }
  std::printf("max|div| away from outflow column = %.3e\n", max_div_interior);
  std::printf("max|div| AT outflow column         = %.3e\n", max_div_outflow_col);
  std::printf("non-continuity rows in this saddle-point system: the outflow "
             "column's p=0 override (i=nx_fluid-1, all %zu rows of j) -- "
             "the pin row is INACTIVE whenever bcs.right.outflow=true (see "
             "stokes_solver.hpp's own `if (!bcs.right.outflow && ...)`), "
             "which is unconditionally the case in this example.\n",
             c.ny_fluid);

  SUCCEED() << "diagnostic test: see stdout for the full term-by-term "
               "energy budget on the coupled system's own unstable mode";
}
