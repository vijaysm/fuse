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
#include <limits>
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
  auto sched = fuse::schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, backend));
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
  // KNOWN_ISSUES.md D44: with the cusolverRf fluid solve this 480-period run
  // is not reproducible run to run (traces fork at ~1e-12 after ~70 periods)
  // and 11 of 15 measured runs failed to converge at period 140. Flag off,
  // the same binary is bitwise reproducible and passes. Skipped, not
  // loosened; the other five cases in this binary still run under the flag.
  GTEST_SKIP() << "D44: nondeterministic under FUSE_ENABLE_CUSPARSE (cusolverRf solve)";
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
  auto sched = fuse::schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, backend));
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
    const std::vector<double>& v_wall_new,
    // DIAGNOSTIC ONLY (track FsiMassCorr): if non-null, filled with the
    // assembled saddle-point matrix cn_step() solved, for a condition-number
    // measurement (the mass-correction isolation tolerance below is derived
    // from it). Null by default; no existing call site is affected.
    Eigen::SparseMatrix<double>* debug_matrix_out = nullptr) {
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
                  debug_matrix_out, nullptr);
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

CoupledPeriodDiag run_one_coupled_period_with_budget(
    const CoupledCase& c, const CoupledDims& d, const Eigen::VectorXd& x,
    // DIAGNOSTIC ONLY (track FsiMassCorr). When non-null, filled with the
    // wall velocity actually fed into the LAST fluid advance() call of
    // this period -- see FluidComponent::debug_last_velocity_in_used's own
    // comment (examples/fsi_channel/fsi_components.hpp). Null by default,
    // so every existing call site is unaffected.
    std::vector<double>* out_last_velocity_in_used = nullptr,
    std::vector<double>* out_last_u0_used = nullptr,
    std::vector<double>* out_last_v0_used = nullptr) {
  fuse::CouplingGraph graph = build_fsi_graph(c);
  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) throw std::runtime_error("mesh load: " + r.error().message);
  }
  const auto lowered = lower(graph);
  if (!lowered.ok()) throw std::runtime_error("lower: " + lowered.diags.to_string());
  const fuse::ClockSet clocks{graph};
  auto sched = fuse::schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, backend));
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
  if (out_last_velocity_in_used != nullptr) {
    out_last_velocity_in_used->assign(c.nx_fluid, 0.0);
    fluid_ptr->debug_last_velocity_in_used = out_last_velocity_in_used;
  }
  if (out_last_u0_used != nullptr) {
    fluid_ptr->solver().debug_last_u0_used = out_last_u0_used;
  }
  if (out_last_v0_used != nullptr) {
    fluid_ptr->solver().debug_last_v0_used = out_last_v0_used;
  }
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

/// Cheap, Runner-free energy functional: Q(x) = beam_KE(wd-block) +
/// beam_PE(w-block) + fluid_KE(u,v-block) of a raw carried-state vector,
/// evaluated by constructing fresh (unregistered) BeamComponent/
/// FluidComponent objects and calling set_state() directly -- the SAME
/// formulas CoupledPeriodDiag::beam_KE0/beam_PE0/fluid_KE0 (and the *1
/// counterparts) use inside run_one_coupled_period_with_budget, confirmed
/// identical to machine precision (MEASURED 2026-10-12, FpPolicy track,
/// round 4: Q(x) computed this way vs. diag.beam_KE0+beam_PE0+fluid_KE0
/// from a REAL production run on the same x, both before and after the
/// period -- relative difference 0.000e+00 on both the chosen mode's real
/// and imaginary parts). No dependence on wdd/p/vin, confirmed by
/// inspection of kinetic_energy()/strain_energy(): nothing else enters.
/// Used ONLY to rank the full D-dimensional eigenspectrum by energy content
/// cheaply: D calls to this are instant, where D calls to the real
/// run_one_coupled_period_with_budget (a full nonlinear IQN-ILS coupled
/// solve each) would cost this test's entire ~2-minute budget D times over.
double fsi_energy_of_state(const CoupledCase& c, const CoupledDims& d,
                           const Eigen::VectorXd& x) {
  fsi::BeamComponent beam("beam", fsi::BeamParams{.n = c.nx_beam, .L = c.L, .EI = c.EI,
                                                  .rho_s = c.rho_s, .h_s = c.h_s});
  std::vector<double> w(c.nx_beam + 1, 0.0), wd(c.nx_beam + 1, 0.0), wdd(c.nx_beam + 1, 0.0);
  for (std::size_t i = 0; i < d.m_beam; ++i) {
    w[i + 1] = x[static_cast<Eigen::Index>(d.off_w + i)];
    wd[i + 1] = x[static_cast<Eigen::Index>(d.off_wd + i)];
    wdd[i + 1] = x[static_cast<Eigen::Index>(d.off_wdd + i)];
  }
  beam.solver().set_state(w, wd, wdd);

  const double H = c.H, Umax = c.Umax;
  const auto inflow = [Umax, H](double y) { return 4.0 * Umax * y * (H - y) / (H * H); };
  fsi::FluidComponent fluid("fluid",
      fsi::ChannelParams{.nx = c.nx_fluid, .ny = c.ny_fluid, .L = c.L, .H = c.H,
                        .mu = c.mu, .rho = c.rho_f}, inflow);
  fsi::FlowState fs;
  fs.u.assign(d.Nu, 0.0);
  fs.v.assign(d.Nv, 0.0);
  fs.p.assign(d.Np, 0.0);
  for (std::size_t i = 0; i < d.Nu; ++i) fs.u[i] = x[static_cast<Eigen::Index>(d.off_u + i)];
  for (std::size_t i = 0; i < d.Nv; ++i) fs.v[i] = x[static_cast<Eigen::Index>(d.off_v + i)];
  for (std::size_t i = 0; i < d.Np; ++i) fs.p[i] = x[static_cast<Eigen::Index>(d.off_p + i)];
  fluid.solver().set_state(fs);

  return beam.solver().kinetic_energy() + beam.solver().strain_energy() +
        fluid.solver().kinetic_energy();
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
  // truncation-limited, which is exactly why the tolerances below are tied
  // to MEASURED eigenvector residuals rather than a fixed number.
  const Eigen::MatrixXd M1 = coupled_map(c, 1.0e-4);
  const EigResult eig = eig_analysis(M1);

  // ---- The passenger mode: a retained sanity check, no closure ----------
  // MEASURED 2026-10-12/13 (FpPolicy track, rounds 3-4): the dominant
  // eigenvalue (real, lambda=-1) is Newmark's acceleration passenger and
  // carries no energy of its own -- E0 = 6.755e-22 (default, contracted
  // arithmetic) / 3.043e-20 (FUSE_FP_REPRODUCIBLE=ON) against a noise
  // floor of 100*eig_quality^2*|x|^2 = 1.010e-19 / 5.646e-20 respectively
  // (round 3's measurement, on the single real eigenvector this mode
  // produces) -- E0 is within 1-2 orders of magnitude of its OWN noise
  // floor in both policies, the signature of a mode whose "energy" is
  // just the eigenvector's own finite-difference error, not a physical
  // quantity. No energy-budget closure is evaluated on it any more (that
  // used to compare noise with noise, see KNOWN_ISSUES.md D30's FpPolicy
  // closing note); what is kept is the cheap, policy-independent growth-
  // rate check this mode still usefully performs.
  const double lambda_dominant_im = eig.solver.eigenvalues()[eig.dominant].imag();
  ASSERT_LT(std::abs(lambda_dominant_im), 1e-6)
      << "dominant eigenvalue has a nonzero imaginary part";
  const double lambda_dominant = eig.solver.eigenvalues()[eig.dominant].real();
  std::printf("dominant eigenvalue (passenger mode) lambda = %.9f (rho check: %.9f)\n",
             lambda_dominant, eig.rho);
  ASSERT_NEAR(std::abs(lambda_dominant), eig.rho, 1e-9);
  EXPECT_LE(eig.rho, 1.0 + 1e-6)
      << "the dominant mode's own growth rate exceeded 1 -- a new defect, "
         "independent of whether this mode carries energy of its own";

  // ---- Choose a mode that carries REAL energy: the beam's own -----------
  // oscillatory mode, by a stated criterion -------------------------------
  // The passenger mode above is energetically empty by construction (it is
  // Newmark's own acceleration bookkeeping, not a physical state), so no
  // energy-budget assertion can be evaluated on it without comparing noise
  // with noise. KNOWN_ISSUES.md D30's FpPolicy closing note (user decision
  // 2026-10-12): evaluate the budget instead on a mode that carries real
  // energy -- the beam's own bending oscillation, which appears in this
  // one-coupling-step map as a COMPLEX-CONJUGATE eigenvalue pair near
  // exp(+-i*omega1*dt) (the phase a free oscillator at the beam's own
  // fundamental frequency would advance in one coupling step).
  //
  // CRITERION (applied in code, not by eyeballing): among eigenvalues with
  // a genuinely nonzero imaginary part (|Im(lambda)| > 1e-9) AND a
  // non-negligible share of the full spectrum's total two-part energy
  // (Q(Re v)+Q(Im v) > 1e-6 of the sum over all eigenvectors -- excluding
  // numerical-noise pairs that could coincidentally have a nearby phase),
  // pick the one whose argument is closest to omega1*dt.
  //
  // MEASURED 2026-10-12: omega1*dt = 9.0000 deg exactly (2*pi/40, dt being
  // 1/40 of the beam's own closed-form period). The discrete 5-interior-
  // node beam ALONE (no fluid) has its own fundamental frequency at 8.09
  // deg/step (natural_frequencies(1), vs. the continuum closed-form 9.00
  // deg/step -- an ~10% discretization gap expected for this coarse a
  // grid). The criterion above selects a complex pair at 4.3931 deg/step,
  // |lambda|=0.9951 -- roughly HALF the beam-alone estimate, consistent
  // with fluid ADDED MASS lowering the coupled system's effective
  // frequency (expected: the fluid's density is the same order as the
  // beam's own areal density here, so a large added-mass shift is
  // physically unsurprising, not a selection error). Its own energy
  // breakdown (checked directly, not assumed) is dominated by beam PE/KE
  // and fluid KE, NOT by its raw state mass (which is mostly pressure, an
  // energetically-inert degree of freedom in this bookkeeping) -- exactly
  // why the selection screens by ENERGY fraction, not mass fraction.
  const double expected_phase = omega1 * c.dt;
  const Eigen::Index n = static_cast<Eigen::Index>(d.D);
  const auto& evals = eig.solver.eigenvalues();
  const auto& evecs = eig.solver.eigenvectors();
  std::vector<double> q_energy(static_cast<std::size_t>(n));
  double q_total = 0.0;
  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::VectorXd re_i(n), im_i(n);
    for (Eigen::Index k = 0; k < n; ++k) {
      re_i[k] = evecs.col(i)[k].real();
      im_i[k] = evecs.col(i)[k].imag();
    }
    const double qi = fsi_energy_of_state(c, d, re_i) + fsi_energy_of_state(c, d, im_i);
    q_energy[static_cast<std::size_t>(i)] = qi;
    q_total += qi;
  }
  std::printf("\n==== FULL SPECTRUM (printed once; %lld eigenvalues) ====\n",
             static_cast<long long>(n));
  std::printf("%4s %14s %14s %10s %14s\n", "idx", "|lambda|", "arg(deg)", "complex?",
             "energy_frac");
  for (Eigen::Index i = 0; i < n; ++i) {
    std::printf("%4lld %14.9f %14.4f %10s %14.6e\n", static_cast<long long>(i),
               std::abs(evals[i]), std::arg(evals[i]) * 180.0 / M_PI,
               std::abs(evals[i].imag()) > 1e-9 ? "yes" : "no",
               q_energy[static_cast<std::size_t>(i)] / q_total);
  }
  Eigen::Index chosen = -1;
  double best_dist = 1e300;
  for (Eigen::Index i = 0; i < n; ++i) {
    if (std::abs(evals[i].imag()) <= 1e-9) continue;
    if (q_energy[static_cast<std::size_t>(i)] / q_total <= 1e-6) continue;
    const double dist = std::abs(std::abs(std::arg(evals[i])) - expected_phase);
    if (dist < best_dist) { best_dist = dist; chosen = i; }
  }
  ASSERT_GE(chosen, 0) << "no complex eigenvalue pair with non-negligible energy "
                          "fraction was found -- the selection criterion needs "
                          "revisiting, not a fallback to the old passenger mode";
  const std::complex<double> lambda_osc = evals[chosen];
  std::printf("SELECTED idx=%lld |lambda|=%.9f arg(deg)=%.4f energy_frac=%.6e "
             "dist_to_expected_phase=%.6e rad\n",
             static_cast<long long>(chosen), std::abs(lambda_osc),
             std::arg(lambda_osc) * 180.0 / M_PI,
             q_energy[static_cast<std::size_t>(chosen)] / q_total, best_dist);

  const double a_osc = lambda_osc.real(), b_osc = lambda_osc.imag();
  const double lambda_osc_mag2 = a_osc * a_osc + b_osc * b_osc;

  // `x` (the real part) and `x_im` (the imaginary part) of the chosen
  // eigenvector. `x` deliberately reuses the name the beam-side/fluid-side
  // identity code below already expects (see that code's own comments,
  // unchanged from before this round): it is fed exactly ONE real state
  // vector and one real production run, same as it always was -- that
  // state vector is now x_re of the chosen oscillatory mode rather than
  // the old passenger mode's own (real) dominant eigenvector, but nothing
  // in the identity derivations below assumed anything about WHICH real
  // state vector `x` was pending this substitution.
  Eigen::VectorXd x(n), x_im(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    x[i] = evecs.col(chosen)[i].real();
    x_im[i] = evecs.col(chosen)[i].imag();
  }

  // Round 4's own homogeneous run: Umax=0 so b=0 exactly (only inflow
  // forces this system; the beam has none of its own), letting TWO real
  // production coupling cycles (one from x, one from x_im) stand in for
  // the complex eigenvector's own one-step map.
  CoupledCase c0 = c;
  c0.Umax = 0.0;
  // DIAGNOSTIC (track FsiMassCorr): capture the wall velocity actually fed
  // into the LAST fluid advance() call of this period, as distinct from
  // whatever fluid_ptr->velocity_in() reads back once run_period() has
  // returned (diag.vwall1, below). See FluidComponent::
  // debug_last_velocity_in_used's own comment.
  std::vector<double> vwall1_actually_used, u0_actually_used, v0_actually_used;
  const CoupledPeriodDiag diag = run_one_coupled_period_with_budget(
      c0, d, x, &vwall1_actually_used, &u0_actually_used, &v0_actually_used);
  const CoupledPeriodDiag diag_im = run_one_coupled_period_with_budget(c0, d, x_im);

  // For a real matrix M with M v = lambda v (v = x + i*x_im, lambda =
  // a_osc + i*b_osc): M x = a_osc*x - b_osc*x_im and M x_im = b_osc*x +
  // a_osc*x_im (equate real/imaginary parts of M v = lambda v). These,
  // not a_osc*x or lambda*x individually, are the correct one-step
  // predictions to measure each part's eigenvector residual against.
  const Eigen::VectorXd pred_re = a_osc * x - b_osc * x_im;
  const Eigen::VectorXd pred_im = b_osc * x + a_osc * x_im;
  const double resid = (diag.y_final - pred_re).norm();
  const double xnorm = x.norm();
  const double eig_quality = resid / xnorm;
  const double resid_im = (diag_im.y_final - pred_im).norm();
  const double xnorm_im = x_im.norm();
  const double eig_quality_im = resid_im / xnorm_im;
  std::printf("|y-pred| / |x| : re-part=%.3e im-part=%.3e (validates x/x_im as the "
             "real/imaginary parts of a genuine eigenvector under the ACTUAL "
             "nonlinear IQN-ILS run, not just the finite-difference-linearized M)\n",
             eig_quality, eig_quality_im);
  ASSERT_LT(eig_quality, 1e-6);
  ASSERT_LT(eig_quality_im, 1e-6);
  // `eig_quality` (re-part) becomes the basis for the two downstream
  // closure tolerances below (dE_beam identity, mass-correction
  // isolation), exactly as before this round: `x` itself carries this much
  // relative error, so no downstream identity built on `x` alone (the
  // beam-side/fluid-side sections, which only ever look at ONE real
  // production run) can close tighter than a small multiple of THIS
  // number. The TOP-LEVEL closure check further below uses BOTH parts'
  // eig_quality together, since it combines both production runs.

  // ---- Total energy bookkeeping (re-part, feeds the retained beam-side/--
  // fluid-side identity checks below UNCHANGED; the combined re+im total
  // further below is what the redesigned closure assertion actually uses)
  const double E0 = diag.beam_KE0 + diag.beam_PE0 + diag.fluid_KE0;
  const double E1 = diag.beam_KE1 + diag.beam_PE1 + diag.fluid_KE1;
  const double dE_beam = (diag.beam_KE1 + diag.beam_PE1) - (diag.beam_KE0 + diag.beam_PE0);
  const double dE_fluid = diag.fluid_KE1 - diag.fluid_KE0;
  const double E0_im = diag_im.beam_KE0 + diag_im.beam_PE0 + diag_im.fluid_KE0;
  const double E1_im = diag_im.beam_KE1 + diag_im.beam_PE1 + diag_im.fluid_KE1;
  std::printf("\n==== ENERGY BOOKKEEPING (re-part) ====\n");
  std::printf("E0 = %.9e  (beam KE0=%.6e PE0=%.6e, fluid KE0=%.6e)\n", E0,
             diag.beam_KE0, diag.beam_PE0, diag.fluid_KE0);
  std::printf("E1 = %.9e  (beam KE1=%.6e PE1=%.6e, fluid KE1=%.6e)\n", E1,
             diag.beam_KE1, diag.beam_PE1, diag.fluid_KE1);
  std::printf("dE_beam = %.9e ; dE_fluid = %.9e\n", dE_beam, dE_fluid);
  std::printf("\n==== ENERGY BOOKKEEPING (im-part) ====\n");
  std::printf("E0_im = %.9e  E1_im = %.9e\n", E0_im, E1_im);

  // ---- The redesigned closure: Re(v)/Im(v) combined, a quantity that --
  // is INVARIANT to the eigenvector's own arbitrary phase -----------------
  // For ANY quadratic form Q(z) = z^T A z (A symmetric; here A is the
  // beam_KE+beam_PE+fluid_KE functional's own, PSD, matrix -- a sum of
  // physical kinetic/potential energies, never negative) and a REAL
  // matrix M with M v = lambda v, decomposing v = x + i*x_im gives
  // EXACTLY (algebra in the derivation this round's report carries,
  // verified numerically on this M1/mode to a relative difference of
  // 7.3e-17, i.e. machine precision):
  //   Q(M x) + Q(M x_im) = |lambda|^2 * (Q(x) + Q(x_im))
  // with the cross term between the real and imaginary parts cancelling
  // identically in the sum -- which is also why Q(x)+Q(x_im) (not either
  // part alone) is the quantity to use: it does not depend on the
  // arbitrary unit-phase convention Eigen's eigensolver picks for v
  // (MEASURED: this sum agreed to 7 significant figures between the two
  // contraction policies -- 3.871039808e-04 vs 3.871040e-04 -- even though
  // the INDIVIDUAL re/im energies differed between policies, confirming
  // this phase-independence directly, not just by derivation).
  const double E0_total = E0 + E0_im;
  const double E1_total = E1 + E1_im;
  const double predicted_E1_total = lambda_osc_mag2 * E0_total;
  const double mismatch_total = std::abs(E1_total - predicted_E1_total);
  std::printf("\n==== COMBINED CLOSURE (re+im, phase-invariant) ====\n");
  std::printf("E0_total = Q(x)+Q(x_im) = %.9e\n", E0_total);
  std::printf("E1_total = Q(y_re)+Q(y_im) = %.9e\n", E1_total);
  std::printf("|lambda|^2 * E0_total = %.9e\n", predicted_E1_total);
  std::printf("mismatch = %.3e (relative: %.3e)\n", mismatch_total,
             mismatch_total / E0_total);

  // ---- Tolerance, DERIVED not chosen -------------------------------------
  // With r = y - (the exact linear eigen-prediction) for each part,
  // |r| <= eig_quality * |x| (measured the same way as above) for each
  // part separately. Cauchy-Schwarz on the energy inner product
  // <u,w>_A = u^T A w (A positive semi-definite, as it is here) gives, for
  // the SAME M/lambda/Q combination as the identity above:
  //   |Q(y) - |lambda|^2 Q(x)| <= 2|lambda| sqrt(Q(x)) sqrt(Q(r)) + Q(r)
  // (full derivation in this round's own report; the cross terms between
  // the "re" and "im" contributions cancel in the SUM the same way the
  // exact identity's do, which is why this bound, too, is stated for the
  // COMBINED Q(x)=Q(x_re)+Q(x_im), Q(r)=Q(r_re)+Q(r_im), not per-part).
  // Q(r) <= q_max * |r|^2, with q_max the largest generalised ratio
  // Q(z)/|z|^2 this test can bound. The energy matrix A itself is not
  // directly accessible here (it is assembled inside BeamSolver/
  // StokesSolver, not exposed), so q_max is a MEASURED PROXY, stated as
  // such rather than passed off as a certified operator-norm bound:
  // q_max = max(Q(x)/|x|^2, Q(x_im)/|x_im|^2), the largest energy density
  // actually observed on the two vectors this computation already has in
  // hand.
  const double eig_quality_combined = std::max(eig_quality, eig_quality_im);
  const double xnorm2_total = xnorm * xnorm + xnorm_im * xnorm_im;
  const double q_max = std::max(E0 / (xnorm * xnorm), E0_im / (xnorm_im * xnorm_im));
  const double Qr_bound = q_max * eig_quality_combined * eig_quality_combined * xnorm2_total;
  const double closure_tol =
      2.0 * std::sqrt(lambda_osc_mag2) * std::sqrt(E0_total) * std::sqrt(Qr_bound) + Qr_bound;
  std::printf("eig_quality_combined=%.3e q_max(measured proxy)=%.3e Q(r) bound=%.3e\n",
             eig_quality_combined, q_max, Qr_bound);
  std::printf("DERIVED TOLERANCE = 2|lambda|*sqrt(E0_total)*sqrt(Q(r)) + Q(r) = %.3e\n",
             closure_tol);
  std::printf("mismatch / tolerance = %.3e (must be < 1)\n", mismatch_total / closure_tol);
  ASSERT_LT(mismatch_total, closure_tol)
      << "the energy budget does not close on the beam's own oscillatory "
         "mode to within the tolerance its own eigenvector accuracy "
         "permits -- this is the actual finding this test now reports, "
         "not a pre-existing property to work around";

  // ---- Non-vacuity: this mode must actually carry energy -----------------
  // The point of this round: fail LOUDLY, not pass silently, if this test
  // ever lands on an energetically-empty mode (like the old passenger
  // mode) instead of a real one. MEASURED 2026-10-12/13: the passenger
  // mode's own E0/(eig_quality^2*|x|^2) ratio was at most ~54 across both
  // contraction policies and 48 perturbed runs (round 3); THIS mode
  // measures ~1.0e12 to 4.5e12 (round 4, both policies) -- 10+ orders of
  // magnitude higher. The factor below (1e6) sits comfortably between the
  // two: 4-5 orders of magnitude above the passenger mode's worst measured
  // value, and 6+ orders of magnitude below this mode's own measured
  // value, so ordinary numerical noise in the eigendecomposition or the
  // production run cannot plausibly trip this check by accident in either
  // direction.
  const double noise_floor_total = eig_quality_combined * eig_quality_combined * xnorm2_total;
  const double energy_to_noise_ratio = E0_total / noise_floor_total;
  constexpr double kNonVacuityFactor = 1.0e6;
  std::printf("NON-VACUITY: E0_total/(eig_quality_combined^2*|x|^2) = %.3e "
             "(must be > %.1e)\n",
             energy_to_noise_ratio, kNonVacuityFactor);
  ASSERT_GT(energy_to_noise_ratio, kNonVacuityFactor)
      << "the chosen mode does not carry enough energy above its own "
         "eigenvector-accuracy noise floor to trust the closure check "
         "above -- this test has landed on an effectively empty mode and "
         "needs a different selection, not a passing result";


  // ---- From here on: UNCHANGED from before this round, except that
  // `x`/`diag`/`eig_quality`/`E0` now refer to the chosen oscillatory
  // mode's REAL PART specifically (one genuine real production run),
  // not the old passenger mode's own dominant eigenvector -- nothing in
  // the beam-side/fluid-side derivations below assumed anything about
  // WHICH real state vector and production run they were given, only
  // that there IS one. The imaginary-part run (`diag_im`) is not used
  // again past this point; it already did its job in the combined
  // closure check above.
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
  // CORRECTED 2026-10-12 (FpPolicy track, final round, KNOWN_ISSUES.md D34):
  // build_fsi_graph() above does not hand the coupling graph c.dt itself --
  // it quantizes it first, through this very file's rationalize_dt()
  // (8-decimal-place rational; defined near build_fsi_graph(), ~60 lines
  // above), and the production run this identity is evaluated against
  // physically advanced on THAT quantized dt, not on the literal c.dt.
  // Using the literal value here compared the identity's right-hand side
  // against a dt the run never actually took. Measured (this file's own
  // investigation, not assumed): mismatch = |c.dt - dt_delivered| *
  // |beam_wall_power|, to 7-8 significant figures in both FP policies, and
  // substituting dt_delivered for a FRESH, isolated beam step reproduces
  // the production run's own w1/wd1 to exactly 0.0 in both policies.
  // Ruled out directly, not assumed: this is NOT FUSE's core scheduler --
  // `fuse::StepContext::dt()` and the coupling graph's own component
  // interval already carry the quantized Rational; nothing downstream of
  // rationalize_dt() ever sees the literal c.dt again. The defect was
  // entirely in this test's own bookkeeping of which dt its two
  // independently-computed sides should agree on.
  const double dt_delivered = rationalize_dt(c.dt).to_double();
  const double dE_beam_naive = dt_delivered * beam_wall_power;
  const double dE_beam_predicted = dE_beam_naive + (1.0 - dx_beam) * dPE;
  std::printf("\n==== BEAM-SIDE IDENTITY ====\n");
  std::printf("dx_beam = L/nx_beam = %.9f\n", dx_beam);
  std::printf("c.dt (literal) = %.17e ; dt_delivered (rationalize_dt) = "
             "%.17e ; diff = %.6e\n",
             c.dt, dt_delivered, c.dt - dt_delivered);
  std::printf("beam-side wall power (dt_delivered*sum(load_in*wdot_avg)) = %.9e\n",
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
  // ---- Tolerance: Higham recursive-summation/inner-product bound --------
  // HISTORY (condensed; superseded below): this tolerance used to be
  // `1.0 * eig_quality * std::abs(E0)`, on the theory that `x` is an
  // eigenvector only to relative accuracy eig_quality so no identity
  // evaluated on it can close tighter. RE-EXAMINED 2026-10-12 (round 4):
  // on the beam's oscillatory mode that factor narrowly FAILED under the
  // default (contracted) policy (mismatch/tolerance=1.029); two
  // independent sweeps (CoupledCase::tol over 6 decades; x's direction
  // perturbed to move eig_quality by 4919x) both showed the mismatch does
  // NOT track eig_quality -- dE_beam/dE_beam_predicted are read off ONE
  // production run's own data with no dependence on x being an
  // eigenvector of anything, so eig_quality was never the right scale for
  // this identity's own evaluation error. No derivation was found that
  // round; the factor was left at 1.0 (expected to fail OFF) rather than
  // be widened without one.
  //
  // CORRECTED 2026-10-12 (final round, KNOWN_ISSUES.md D34): with the
  // dt_delivered fix above, the mismatch itself dropped to round-off
  // (measured ~1e-16-1e-17 relative to E0, both policies) -- the ~1e-12
  // floor this history describes was entirely the dt-literal/dt-delivered
  // bookkeeping defect, not a genuine eig_quality or conditioning limit.
  // The tolerance below replaces eig_quality with the actual floating-
  // point evaluation-error bound for the identity's own arithmetic, via
  // Higham's recursive-summation/inner-product result ("Accuracy and
  // Stability of Numerical Algorithms", 2nd ed., Sec. 3.1: for an n-term
  // sum/inner product evaluated in the usual left-to-right order,
  // |fl(sum x_i y_i) - sum x_i y_i| <= gamma_n * sum|x_i y_i|,
  // gamma_n = n*u/(1-n*u), u = machine epsilon), plus the beam's Keff2
  // solve's own backward-stable (direct sparse LU) forward error,
  // bounded by kappa_2(Keff2)*u.
  //
  // The identity combines three terms: dt_delivered*beam_wall_power
  // (beam_wall_power itself an m_beam=5-term dot product -- gamma_m
  // applies directly), and (1-dx_beam)*dPE, dPE=PE1-PE0 with
  // PE=0.5*w^T*K*w an nnz(K)-term quadratic form. K is the beam's
  // pentadiagonal stiffness operator under mirror-reflection boundary
  // conditions; this test has no access to the solver's private K_, so
  // it rebuilds the SAME public, documented construction (EI/dx_beam^4
  // times the 5-point stencil {1,-4,6,-4,1}) independently below --
  // confirmed elsewhere (investigation scratch, not shipped) to match
  // BeamSolver::K_ to 0.0 in the matrix norm for this exact
  // (nx_beam, L, EI). The quadratic form's own sum of absolute products
  // cancels substantially against its final (small) value -- measured
  // ~44x for this mode in both policies -- so the Higham bound must
  // scale with that uncancelled sum, not with |dPE| itself; using |dPE|
  // there would understate the true rounding floor by that same factor.
  // kappa_2(Keff2) is likewise computed directly below (JacobiSVD on the
  // same replica), not cited as an external constant.
  Eigen::MatrixXd K_beam_replica =
      Eigen::MatrixXd::Zero(static_cast<int>(d.m_beam), static_cast<int>(d.m_beam));
  {
    const double factor = c.EI / (dx_beam * dx_beam * dx_beam * dx_beam);
    const double stencil[5] = {1.0, -4.0, 6.0, -4.0, 1.0};
    const auto n_ll = static_cast<long long>(c.nx_beam);
    const auto reflect = [n_ll](long long j) -> long long {
      if (j == -1) return 1;
      if (j == n_ll + 1) return n_ll - 1;
      return j;
    };
    for (std::size_t idx = 0; idx < d.m_beam; ++idx) {
      const long long node = static_cast<long long>(idx) + 1;
      for (int off = -2; off <= 2; ++off) {
        const long long j = node + off;
        if (j < -1 || j > n_ll + 1) continue;
        const long long j_ref = reflect(j);
        if (j_ref == 0 || j_ref == n_ll) continue;
        const auto col = static_cast<std::size_t>(j_ref - 1);
        K_beam_replica(static_cast<int>(idx), static_cast<int>(col)) +=
            factor * stencil[off + 2];
      }
    }
  }
  std::size_t nnz_K_beam = 0;
  for (int i = 0; i < K_beam_replica.rows(); ++i)
    for (int j = 0; j < K_beam_replica.cols(); ++j)
      if (K_beam_replica(i, j) != 0.0) ++nnz_K_beam;
  const double mconst_beam = c.rho_s * c.h_s;
  const Eigen::MatrixXd Keff2_beam_replica =
      (dt_delivered / 4.0) * K_beam_replica +
      (mconst_beam / dt_delivered) *
          Eigen::MatrixXd::Identity(K_beam_replica.rows(), K_beam_replica.cols());
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd_beam(Keff2_beam_replica);
  const auto& sv_beam = svd_beam.singularValues();
  const double kappa_beam = sv_beam(0) / sv_beam(sv_beam.size() - 1);

  Eigen::VectorXd w1_beam(static_cast<Eigen::Index>(d.m_beam)),
      w0_beam(static_cast<Eigen::Index>(d.m_beam));
  for (std::size_t i = 0; i < d.m_beam; ++i) {
    w1_beam[static_cast<Eigen::Index>(i)] =
        diag.y_final[static_cast<Eigen::Index>(d.off_w + i)];
    w0_beam[static_cast<Eigen::Index>(i)] =
        x[static_cast<Eigen::Index>(d.off_w + i)];
  }
  double sum_abs_load_wdot = 0.0, sum_wdot1_sq = 0.0, sum_wdot0_sq = 0.0;
  for (std::size_t i = 0; i < d.m_beam; ++i) {
    const double wdot_avg = 0.5 * (diag.wd0[i] + diag.wd1[i]);
    sum_abs_load_wdot += std::abs(diag.load_in_interior[i] * wdot_avg);
    sum_wdot1_sq += diag.wd1[i] * diag.wd1[i];
    sum_wdot0_sq += diag.wd0[i] * diag.wd0[i];
  }
  double sum_abs_K_w1_w1 = 0.0, sum_abs_K_w0_w0 = 0.0;
  for (int i = 0; i < K_beam_replica.rows(); ++i) {
    for (int j = 0; j < K_beam_replica.cols(); ++j) {
      if (K_beam_replica(i, j) == 0.0) continue;
      sum_abs_K_w1_w1 += std::abs(K_beam_replica(i, j) * w1_beam[i] * w1_beam[j]);
      sum_abs_K_w0_w0 += std::abs(K_beam_replica(i, j) * w0_beam[i] * w0_beam[j]);
    }
  }
  const double u_mach = std::numeric_limits<double>::epsilon();
  const auto gamma_n = [u_mach](double n_terms) {
    return n_terms * u_mach / (1.0 - n_terms * u_mach);
  };
  const double gamma_m_beam = gamma_n(static_cast<double>(d.m_beam));
  const double gamma_PE_beam = gamma_n(static_cast<double>(nnz_K_beam + d.m_beam));
  // Absolute error bounds: gamma_n*(sum of absolute products) for the
  // evaluation error, plus 2*kappa*u (quadratic sensitivity) for
  // solve-derived quantities (w1/wd1), 0 for w0/wd0 (the exact
  // eigenvector input, not a solve output).
  const double delta_P_beam = gamma_m_beam * sum_abs_load_wdot +
                              kappa_beam * u_mach * std::abs(beam_wall_power);
  const double delta_KE1_beam = gamma_m_beam * sum_wdot1_sq * 0.5 * mconst_beam +
                                2.0 * kappa_beam * u_mach * diag.beam_KE1;
  const double delta_KE0_beam = gamma_m_beam * sum_wdot0_sq * 0.5 * mconst_beam;
  const double delta_PE1_beam = gamma_PE_beam * sum_abs_K_w1_w1 * 0.5 +
                                2.0 * kappa_beam * u_mach * diag.beam_PE1;
  const double delta_PE0_beam = gamma_PE_beam * sum_abs_K_w0_w0 * 0.5;
  const double delta_dE_beam =
      delta_KE1_beam + delta_KE0_beam + delta_PE1_beam + delta_PE0_beam;
  const double delta_dt_P_beam = dt_delivered * delta_P_beam;
  const double delta_dPE_term_beam =
      (1.0 - dx_beam) * (delta_PE1_beam + delta_PE0_beam);
  // Convert each rigorous absolute bound into the form
  // tol = c*u*(|dE_beam|+|dt*P|+|(1-dx_beam)*dPE|) by finding, for each of
  // the three terms, the multiplier c_i = delta_i/(u*|term_i|); c=max(c_i)
  // then guarantees c*u*(sum of magnitudes) >= sum of the three rigorous
  // bounds, for any nonnegative weights. c is DERIVED and PRINTED below,
  // never fitted.
  const double c_dE_beam = delta_dE_beam / (u_mach * std::abs(dE_beam));
  const double c_dt_P_beam = delta_dt_P_beam / (u_mach * std::abs(dE_beam_naive));
  const double c_dPE_beam =
      delta_dPE_term_beam / (u_mach * std::abs((1.0 - dx_beam) * dPE));
  const double c_beam = std::max({c_dE_beam, c_dt_P_beam, c_dPE_beam});
  std::printf("\n==== BEAM IDENTITY TOLERANCE (Higham, derived) ====\n");
  std::printf("nnz(K)=%zu kappa_2(Keff2)=%.6f (both measured directly, not cited)\n",
             nnz_K_beam, kappa_beam);
  std::printf("sum|K_ij*w1_i*w1_j|=%.6e vs 2*|PE1|=%.6e (cancellation ratio=%.3f)\n",
             sum_abs_K_w1_w1, 2.0 * diag.beam_PE1,
             sum_abs_K_w1_w1 / (2.0 * diag.beam_PE1));
  std::printf("c_dE=%.6e c_dt_P=%.6e c_dPE=%.6e c(overall,max,NOT tuned)=%.6e\n",
             c_dE_beam, c_dt_P_beam, c_dPE_beam, c_beam);
  const double beam_identity_tol =
      c_beam * u_mach *
      (std::abs(dE_beam) + std::abs(dE_beam_naive) + std::abs((1.0 - dx_beam) * dPE));
  std::printf("beam identity tolerance = c*u*(|dE_beam|+|dt*P|+|(1-dx_beam)*dPE|) "
             "= %.3e (measured mismatch/tolerance = %.3e, i.e. tolerance is "
             "%.1fx the observed mismatch)\n",
             beam_identity_tol,
             std::abs(dE_beam - dE_beam_predicted) / beam_identity_tol,
             beam_identity_tol / std::abs(dE_beam - dE_beam_predicted));
  ASSERT_NEAR(dE_beam, dE_beam_predicted, beam_identity_tol)
      << "beam's own exact Newmark energy identity (CORRECTED for the "
         "kinetic_energy()/strain_energy() quadrature mismatch, and for "
         "the dt_delivered/literal-c.dt bookkeeping above) does not "
         "close -- nothing past this point can be trusted until it does";

  // ---- Fluid-side: isolate the post-solve mass correction ----------------
  //
  // MECHANISM, measured directly (track FsiMassCorr, not assumed): this
  // identity used to key its tolerance to `eig_quality` (the eigensolver's
  // own residual), on the theory that `u_raw` -- an INDEPENDENT `cn_step`
  // solve fed the SAME `diag.u0`/`diag.v0`/`diag.vwall0`/`diag.vwall1`
  // extracted from the production run -- differs from `diag.u1` only by
  // the eigenvector's own inaccuracy. Three alternative mechanisms were
  // tested directly, in order, each by a measurement that isolates it:
  //
  //  (1) hypothesis (c), a test sampling artifact ("u1 and u_raw sampled
  //      at different states"): `diag.vwall1` is `fluid_ptr->velocity_in()`
  //      read back AFTER `Runner::run_period()` returns -- i.e. the
  //      region's own `G(y)` (src/runtime/iteration.cpp's own header
  //      comment: "the Sink instructions ... deliver G(y)") -- which need
  //      not equal `y`, the value the LAST `advance()` actually solved
  //      from, by more than the region's own convergence tolerance.
  //      `FluidComponent::debug_last_velocity_in_used` (examples/
  //      fsi_channel/fsi_components.hpp) captures `y` directly: measured
  //      |y - G(y)| = 3.90e-18 (round-off) on this case, and feeding `y`
  //      instead of `diag.vwall1` reproduces the IDENTICAL 1.081e-10
  //      non-outflow mismatch to 7 significant figures. REJECTED.
  //  (2) the cross-call cached saddle-point factorisation (`StokesSolver::
  //      lu_cache_`): `step()` reuses it; this reconstruction's own
  //      `cn_step` call always refactorises. Forcing `step()` to always
  //      refactorise too (temporarily, in isolation) reproduced the
  //      IDENTICAL 1.081237e-10 to every printed digit. REJECTED.
  //  (3) `diag.u0`/`diag.v0` not being bit-identical to the (u_, v_) the
  //      last `advance()` actually restored and solved from --
  //      `StokesSolver::debug_last_u0_used`/`debug_last_v0_used` measure
  //      this directly: both gaps are EXACTLY 0.0. REJECTED.
  //
  // What remained, and confirmed the mechanism: this call passed the
  // LITERAL `c.dt` as `cn_step`'s own `dt` argument, while every
  // PRODUCTION `cn_step` call in this run executed on `dt_delivered =
  // rationalize_dt(c.dt).to_double()` -- the SAME test-owned
  // rationalize_dt()/dt_delivered split D34 (above) already found and
  // fixed for the beam-side identity, reintroduced here for the fluid
  // side. `cn_step`'s own matrix carries `dt` through the time-derivative
  // (O(1/dt)) coefficient, so a relative dt error of
  // |c.dt - dt_delivered| / dt_delivered = 4.596e-7 (measured above, beam
  // section) perturbs EVERY column of the solve, not just the outflow
  // one the correction touches -- exactly an across-the-board leak, not a
  // defect confined to one column. Feeding `dt_delivered` instead of the
  // literal `c.dt` collapsed the non-outflow mismatch from 1.081e-10 to
  // 4.443e-15 (round-off, ~4 orders of magnitude below `u_raw_scale`
  // times machine epsilon) -- CONFIRMED. Fixed below by using
  // `dt_delivered` directly; hypotheses (a) [this one] vs. (b) [a genuine
  // defect in the correction's own column-restriction] are thereby
  // distinguished: (b) is REJECTED too, since the correction's own code
  // (examples/fsi_channel/stokes_solver.hpp, the outflow-mass-conservation
  // block) writes to exactly one `idx_u(p_, p_.nx, j)` per row and no
  // other index, and the non-outflow mismatch vanishes once the INPUT
  // bookkeeping (not the correction) is fixed.
  const fsi::ChannelParams fc{.nx = c.nx_fluid, .ny = c.ny_fluid, .L = c.L,
                              .H = c.H, .mu = c.mu, .rho = c.rho_f};
  // Evidence for the mechanism story above (hypotheses (c) and the
  // checkpoint/restore contract), measured rather than assumed.
  double vwall1_gap = 0.0, u0_gap = 0.0, v0_gap = 0.0;
  for (std::size_t i = 0; i < vwall1_actually_used.size(); ++i) {
    vwall1_gap = std::max(vwall1_gap,
                          std::abs(vwall1_actually_used[i] - diag.vwall1[i]));
  }
  for (std::size_t i = 0; i < u0_actually_used.size(); ++i) {
    u0_gap = std::max(u0_gap, std::abs(u0_actually_used[i] - diag.u0[i]));
  }
  for (std::size_t i = 0; i < v0_actually_used.size(); ++i) {
    v0_gap = std::max(v0_gap, std::abs(v0_actually_used[i] - diag.v0[i]));
  }
  std::printf("\n==== MASS-CORRECTION MECHANISM EVIDENCE (measured, see comment "
             "above) ====\n");
  std::printf("max|y - G(y)| on fluid_velocity_in (hypothesis (c) check) = %.3e\n",
             vwall1_gap);
  std::printf("max|u0_actually_used - diag.u0|, |v0_actually_used - diag.v0| "
             "(checkpoint/restore contract check) = %.3e, %.3e\n", u0_gap, v0_gap);

  Eigen::SparseMatrix<double> A_fluid_debug;
  const GeneralIdentityResult gen = measure_full_energy_identity_general(
      fc, dt_delivered, diag.u0, diag.v0, diag.vwall0, diag.vwall1,
      &A_fluid_debug);
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
  double u_raw_scale = 0.0;
  for (const double v : u_raw) u_raw_scale = std::max(u_raw_scale, std::abs(v));
  // Tolerance DERIVED from the solve's own backward stability, not from
  // eig_quality (eig_quality is the EIGENSOLVER's residual; the
  // eigenvector x and the fluid saddle-point solve are unrelated linear
  // algebra, and tying one's tolerance to the other's residual was the
  // original defect this round replaced). With the dt bookkeeping fixed
  // above, `diag.u1` (away from the outflow column) and `u_raw` are two
  // INDEPENDENT solves of the IDENTICAL assembled system `A x = rhs`
  // (production's cached factorisation vs. this call's fresh one) --
  // each backward-stable per Higham (Accuracy and Stability of Numerical
  // Algorithms, 2nd ed., sec. 9.3-9.5), with relative error bounded by
  // this project's own established per-solve constant `2*kappa_2(A)*u`
  // (the SAME form already used for the beam's own direct solve, above).
  // Two independent solves of one system can therefore differ by up to
  // the SUM of their two bounds, `4*kappa_2(A)*u`, scaled by the
  // solution's own magnitude. `kappa_2(A)` is measured directly (dense
  // JacobiSVD on the assembled saddle-point matrix cn_step solved -- A is
  // small enough, N=53 on this grid, for an exact dense SVD).
  const Eigen::MatrixXd A_fluid_dense = Eigen::MatrixXd(A_fluid_debug);
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd_fluid(A_fluid_dense);
  const Eigen::VectorXd sv_fluid = svd_fluid.singularValues();
  const double kappa_fluid = sv_fluid(0) / sv_fluid(sv_fluid.size() - 1);
  const double mass_corr_tol = 4.0 * kappa_fluid * u_mach * u_raw_scale;
  std::printf("\n==== MASS-CORRECTION ISOLATION TOLERANCE (Higham, derived) ====\n");
  std::printf("N(A_fluid)=%d kappa_2(A_fluid)=%.6e (measured directly, not cited)\n",
             static_cast<int>(A_fluid_dense.rows()), kappa_fluid);
  std::printf("max|u_raw| (velocity scale) = %.3e\n", u_raw_scale);
  std::printf("mass-correction isolation tolerance = 4*kappa_2(A_fluid)*u*max|u_raw| "
             "= %.3e\n", mass_corr_tol);
  std::printf("max|u1-u_raw| away from outflow column = %.3e (must be ~0: "
             "correction touches ONLY the outflow column; ratio to "
             "tolerance = %.3f)\n",
             max_abs_diff_non_outflow, max_abs_diff_non_outflow / mass_corr_tol);
  std::printf("outflow correction shift = %.9e (uniform across j to within "
             "%.3e, ratio to tolerance = %.3f)\n",
             correction, max_correction_deviation,
             max_correction_deviation / mass_corr_tol);
  ASSERT_LT(max_abs_diff_non_outflow, mass_corr_tol)
      << "the mass correction touches a column other than the outflow one "
         "(or the dt_delivered/literal-c.dt bookkeeping above has regressed)";
  ASSERT_LT(max_correction_deviation, mass_corr_tol)
      << "the outflow correction is not uniform across j (or the "
         "dt_delivered/literal-c.dt bookkeeping above has regressed)";

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

  // KNOWN_ISSUES.md D34 item (A), FIXED (track FsiAdjointness, 2026-10-22):
  // this used to read `const double dt = c.dt;` here -- the LITERAL dt,
  // not `dt_delivered` -- while every gen.id.* quantity below was computed
  // by `measure_full_energy_identity_general`'s own cn_step-style
  // reconstruction called with `dt_delivered` a few lines above (D80's own
  // fix). Same mechanism D34 already found and fixed for the beam-side
  // identity, reintroduced here for this fluid-side one. Measured
  // directly (not assumed): using the literal dt here contributes
  // |c.dt-dt_delivered|*|interior_viscous_power+pressure_div_power-
  // boundary_bt_term| = 6.361e-13 to this identity's mismatch under the
  // default (ON) policy -- 99% of the identity's OWN measured mismatch
  // (6.444e-13) and 59% of the existing `1e-9*|E0|+1e-12` tolerance
  // budget (1.077e-12); under FP_REPRODUCIBLE=OFF it is 1.6e-14, 100% of
  // that run's own mismatch (1.6e-14) but only 1.2% of its (looser,
  // bigger-E0) tolerance. The tolerance was never breached -- this is
  // NOT a currently-failing test -- but it was absorbing a real,
  // non-round-off bookkeeping error rather than the round-off the
  // ASSERT_NEAR's comment implies. Fixed below by using `dt_delivered`;
  // collapses the mismatch to 8.26e-15 (ON) / 4.51e-17 (OFF), both >=2
  // orders of magnitude tighter against the SAME, UNCHANGED tolerance.
  const double dt_literal_for_record = c.dt;  // kept ONLY for the recorded (not asserted) comparison below
  const double dt = dt_delivered;
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
  // Recorded, not asserted: what this identity's mismatch would have been
  // with the OLD literal-c.dt bookkeeping (see comment above) -- measured,
  // not assumed, kept here as evidence the defect was real and that the
  // fix tightens rather than masks the comparison.
  {
    const double dE_fluid_predicted_literal =
        dt_literal_for_record *
            (gen.id.interior_viscous_power + gen.id.pressure_div_power -
             gen.id.boundary_bt_term) +
        gen.id.ke_change_v_wall + gen.id.ke_change_u_outflow + correction_energy;
    std::printf("[recorded, not asserted: with the OLD literal-c.dt "
               "bookkeeping this identity's mismatch would have been "
               "%.3e (%.1f%% of the %.3e tolerance budget); fixed, it is "
               "%.3e]\n",
               std::abs(dE_fluid - dE_fluid_predicted_literal),
               100.0 * std::abs(dE_fluid - dE_fluid_predicted_literal) /
                   (1e-9 * std::abs(E0) + 1e-12),
               1e-9 * std::abs(E0) + 1e-12,
               std::abs(dE_fluid - dE_fluid_predicted));
  }

  // ---- Fluid-side vs beam-side delivered power (H/H^T adjointness) -------
  // KNOWN_ISSUES.md D34 item (B), INVESTIGATED (track FsiAdjointness,
  // 2026-10-22): this used to compare `dE_beam_naive - gen.id.W_wall_current`
  // and label a nonzero result "a non-adjoint interface remap". Measured
  // directly, three hypotheses, all tested rather than assumed:
  //
  //  (h1) non-adjoint H/H^T D under nx_beam != nx_fluid: FALSIFIED. A
  //       standalone program built H (fuse::build_consistent_interp_weights)
  //       and the production H^T D matrix (build_consistent_weights_for_plan's
  //       own "consistent-transpose" formula, reproduced by hand) for this
  //       exact pair (nx_beam=6 elements/7 nodes, nx_fluid=5 cells/6
  //       vertices) and for 8 further (matched and mismatched) pairs up to
  //       nx=10, with fixed deterministic test vectors. The documented
  //       identity (docs/user/interface-transfer.md: "(H v)^T (D rho) ==
  //       v^T (H^T D rho)... bit for bit") held to round-off (relative
  //       defect <= 1.75e-16) in EVERY case, matched or mismatched. A
  //       NAIVE comparison that omits the D weighting shows an O(1)
  //       (not O(1e-6)) defect, confirming the weighting -- not the mesh
  //       mismatch -- is what a broken comparison would actually look like,
  //       and that is not this.
  //  (h2) dt_delivered/literal-c.dt bookkeeping (same class as item (A)):
  //       FALSIFIED. Both `dE_beam_naive` and `gen.id.W_wall_current`
  //       already use `dt_delivered` consistently (the former explicitly,
  //       the latter because `measure_full_energy_identity_general` is
  //       called with `dt_delivered`, D80's fix). Moving `c.dt` exactly
  //       onto the rationalize_dt() 1e-8 grid (so diff=0 identically)
  //       left this section's mismatch bit-for-bit unchanged.
  //  (h3) CONFIRMED: this is how the test formed the comparison, not a
  //       defect in the interface operators. `W_wall_current` is this
  //       function's own one-sided wall TRACTION dotted with the wall
  //       velocity (file header above: "mu*sum(vel_avg.Lap_avg)" splits
  //       into bulk dissipation (<=0) plus `W_wall_viscous_natural`/
  //       `B_out_viscous_natural`, which must be POSITIVE contributions to
  //       (E1_interior-E0_interior)/dt for `bulk_dissipation` above to
  //       come out positive, as its own "should be POSITIVE: it removes
  //       energy" comment requires) -- i.e. POSITIVE W_wall_current means
  //       power FLOWING INTO the fluid across that boundary (the standard
  //       continuum-mechanics convention, traction dotted with velocity,
  //       outward normal). `dE_beam_naive` is power FLOWING INTO the beam.
  //       At a kinematically-matched, non-dissipative interface these are
  //       NEGATIVES of each other, not equal: power leaving the beam's own
  //       bookkeeping must arrive as power ENTERING the fluid's wall term
  //       with a MINUS sign already baked into W_wall_current's own sign
  //       convention, so the correct identity is `dE_beam_naive +
  //       gen.id.W_wall_current == 0`, not their difference. Measured: the
  //       SUM is round-off scale in both FP policies (below), while the
  //       (mislabelled) difference is O(1e-6)/O(1e-9) purely because it
  //       adds two numbers that should have been subtracted. The printf
  //       label "power fluid DELIVERS" for W_wall_current was itself the
  //       error -- it describes what the quantity morally means, not the
  //       sign the code actually computes it with.
  //
  // NOT asserted (deliberately): a first attempt bounded `interface_sum`
  // below with a Higham recursive-summation tolerance in the same style as
  // the beam identity tolerance above (gamma_n(nx_fluid)*sum-of-abs-products
  // plus kappa_fluid*u*|W_wall_current|). Measured once with temporary
  // instrumentation that has since been removed (so this figure is a
  // record, not reproducible from this file): the per-term sums show almost
  // no internal cancellation here (sum-of-|terms| is 1.00-2.12x the signed
  // result in both policies, nothing like the ~44x the beam's own dPE
  // needed), so that bound came out to 4.0e-17 (ON) /
  // 3.7e-19 (OFF) -- SMALLER than the actual measured |interface_sum|
  // (5.99e-17 ON / 1.80e-16 OFF), i.e. the bound would have FAILED even
  // though the underlying physics is fine. The real error source is most
  // likely the fluid solve's own forward error (the SAME two-independent-
  // solves mechanism already bounded elsewhere in this file as
  // `4*kappa_2(A_fluid)*u*max|u_raw|`, D80) propagating through the wall
  // traction formula's own sensitivity to v1/p1 -- not through
  // W_wall_current's already-cancelled-small net value, the same lesson
  // this file's own beam-identity-tolerance history already drew for
  // dPE. Deriving that sensitivity rigorously (a Lipschitz constant for
  // the traction formula, not simply reusing an existing kappa) is a
  // real piece of work, not a one-line fix, and this track's standing
  // rule is that a tolerance is never chosen after seeing a measurement
  // fail it. Left as a RECORDED diagnostic (printed, not asserted) --
  // see open_questions for the follow-up.
  const double interface_sum = dE_beam_naive + gen.id.W_wall_current;
  std::printf("\n==== INTERFACE ADJOINTNESS CHECK (H vs H^T D) ====\n");
  std::printf("power beam RECEIVES (dt_delivered*sum(load_in*wdot_avg), beam's "
             "own grid, nx_beam=%zu) = %+.9e\n",
             c.nx_beam, dE_beam_naive);
  std::printf("power INTO FLUID at the wall (W_wall_current, production "
             "traction form, fluid's own grid, nx_fluid=%zu; NEGATIVE of "
             "what the fluid delivers to the beam, see derivation above) = "
             "%+.9e\n",
             c.nx_fluid, gen.id.W_wall_current);
  std::printf("CORRECTED interface power balance (beam receives + fluid's "
             "own into-fluid-sign W_wall_current) = %+.3e (round-off scale; "
             "h1 [non-adjoint remap] and h2 [dt bookkeeping] FALSIFIED, h3 "
             "[sign convention in the comparison] CONFIRMED -- see "
             "derivation above; NOT asserted, see comment above for why)\n",
             interface_sum);
  std::printf("[recorded, not asserted: the OLD (mislabelled) difference "
             "beam_received-fluid_delivered = %+.9e -- this is the number "
             "KNOWN_ISSUES.md D34 originally reported nonzero; it is "
             "EXPECTED to be nonzero -- see derivation above, it is not "
             "the quantity that should vanish]\n",
             dE_beam_naive - gen.id.W_wall_current);

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
               "energy budget on the beam's own oscillatory mode";
}
