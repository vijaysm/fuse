// Method-of-manufactured-solutions verification of `fsi::cn_step`
// (examples/fsi_channel/stokes_solver.hpp): the unsteady-Stokes Crank-
// Nicolson saddle-point solve the FSI channel example uses for its fluid
// side. Exercises the SAME assembler the example's StokesSolver::step()
// calls, with a divergence-free streamfunction-derived manufactured
// solution and Dirichlet boundaries on all four sides -- so the outflow/
// zero-gradient branch (a deliberate, first-order modelling simplification;
// see the header) is never touched here, and this test measures the
// scheme's genuine order.
//
// MANUFACTURED SOLUTION, derived via a streamfunction (automatically
// divergence-free for any smooth psi):
//   psi(x,y,t) = sin(kx x) sin(ky y) cos(w t),   kx = pi/L, ky = pi/H
//   u =  psi_y = (pi/H) sin(kx x) cos(ky y) cos(w t)
//   v = -psi_x = -(pi/L) cos(kx x) sin(ky y) cos(w t)
//   p = P0 cos(kx x) cos(ky y) cos(w t)
// with the body force f = rho u_t + grad p - mu Lap(u) computed by hand and
// cross-checked against an independent sympy derivation; see docs/user/fsi-channel.md for the closed forms.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "stokes_solver.hpp"

namespace {

struct Manufactured {
  double L, H, mu, rho, P0, Tm;
  [[nodiscard]] double kx() const { return M_PI / L; }
  [[nodiscard]] double ky() const { return M_PI / H; }
  [[nodiscard]] double w() const { return 2.0 * M_PI / Tm; }

  [[nodiscard]] double u(double x, double y, double t) const {
    return (M_PI / H) * std::sin(kx() * x) * std::cos(ky() * y) * std::cos(w() * t);
  }
  [[nodiscard]] double v(double x, double y, double t) const {
    return -(M_PI / L) * std::cos(kx() * x) * std::sin(ky() * y) * std::cos(w() * t);
  }
  [[nodiscard]] double p(double x, double y, double t) const {
    return P0 * std::cos(kx() * x) * std::cos(ky() * y) * std::cos(w() * t);
  }
  [[nodiscard]] double fu(double x, double y, double t) const {
    const double s = std::sin(kx() * x) * std::cos(ky() * y);
    return s * (-rho * (M_PI / H) * w() * std::sin(w() * t) +
               (M_PI / H) * mu * (kx() * kx() + ky() * ky()) * std::cos(w() * t) -
               P0 * kx() * std::cos(w() * t));
  }
  [[nodiscard]] double fv(double x, double y, double t) const {
    const double s = std::cos(kx() * x) * std::sin(ky() * y);
    return s * (rho * (M_PI / L) * w() * std::sin(w() * t) -
               P0 * ky() * std::cos(w() * t) -
               mu * (kx() * kx() + ky() * ky()) * (M_PI / L) * std::cos(w() * t));
  }
};

fsi::ChannelBCs mms_bcs(const Manufactured& m) {
  fsi::ChannelBCs bcs;
  bcs.left.u = [&m](double y, double t) { return m.u(0.0, y, t); };
  bcs.left.v = [&m](double y, double t) { return m.v(0.0, y, t); };
  bcs.right.u = [&m](double y, double t) { return m.u(m.L, y, t); };
  bcs.right.v = [&m](double y, double t) { return m.v(m.L, y, t); };
  bcs.bottom.u = [&m](double x, double t) { return m.u(x, 0.0, t); };
  bcs.bottom.v = [&m](double x, double t) { return m.v(x, 0.0, t); };
  bcs.top.u = [&m](double x, double t) { return m.u(x, m.H, t); };
  bcs.top.v = [&m](double x, double t) { return m.v(x, m.H, t); };
  return bcs;
}

/// Runs `n_steps` of size `dt` from t=0, initial condition the manufactured
/// solution at t=0, and returns the L2 error of u and v (over their own
/// staggered points) against the manufactured solution at the final time.
struct Errors {
  double u_l2, v_l2;
};

Errors run_and_measure(std::size_t nx, std::size_t ny, double dt,
                       std::size_t n_steps, const Manufactured& m) {
  fsi::ChannelParams cp{.nx = nx, .ny = ny, .L = m.L, .H = m.H, .mu = m.mu,
                       .rho = m.rho};
  fsi::SourceTerms src;
  src.fu = [&m](double x, double y, double t) { return m.fu(x, y, t); };
  src.fv = [&m](double x, double y, double t) { return m.fv(x, y, t); };
  const fsi::ChannelBCs bcs = mms_bcs(m);
  const double dx = m.L / static_cast<double>(nx);
  const double dy = m.H / static_cast<double>(ny);

  std::vector<double> u0((nx + 1) * ny), v0(nx * (ny + 1));
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i <= nx; ++i) {
      u0[j * (nx + 1) + i] = m.u(static_cast<double>(i) * dx,
                                 (static_cast<double>(j) + 0.5) * dy, 0.0);
    }
  }
  for (std::size_t j = 0; j <= ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      v0[j * nx + i] = m.v((static_cast<double>(i) + 0.5) * dx,
                           static_cast<double>(j) * dy, 0.0);
    }
  }

  fsi::PressurePin pin{0, ny - 1, m.p(dx / 2.0, m.H - dy / 2.0, dt)};
  double t0 = 0.0;
  fsi::FlowState state{u0, v0, {}};
  for (std::size_t s = 0; s < n_steps; ++s) {
    const double t1 = t0 + dt;
    pin.value = m.p((static_cast<double>(pin.i) + 0.5) * dx,
                    (static_cast<double>(pin.j) + 0.5) * dy, t1);
    state = fsi::cn_step(cp, bcs, src, dt, t0, t1, state.u, state.v, pin);
    t0 = t1;
  }

  double su = 0.0, sv = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i <= nx; ++i) {
      const double want = m.u(static_cast<double>(i) * dx,
                              (static_cast<double>(j) + 0.5) * dy, t0);
      const double e = state.u[j * (nx + 1) + i] - want;
      su += e * e * dx * dy;
    }
  }
  for (std::size_t j = 0; j <= ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const double want = m.v((static_cast<double>(i) + 0.5) * dx,
                              static_cast<double>(j) * dy, t0);
      const double e = state.v[j * nx + i] - want;
      sv += e * e * dx * dy;
    }
  }
  return Errors{std::sqrt(su), std::sqrt(sv)};
}

double observed_order(const std::vector<double>& h,
                      const std::vector<double>& err) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = static_cast<double>(h.size());
  for (std::size_t k = 0; k < h.size(); ++k) {
    const double x = std::log(h[k]), y = std::log(err[k]);
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

}  // namespace

TEST(FsiStokesMms, SpatialOrderIsTwo) {
  const Manufactured m{.L = 1.3, .H = 0.7, .mu = 0.9, .rho = 1.1, .P0 = 0.6,
                       .Tm = 2.1};
  // dt fixed and tiny: its O(dt^2) contribution (bounded by the temporal-
  // order test above, which measures it directly at a comparable dt) stays
  // far under the coarsest spatial error tested, so refining the grid alone
  // isolates the spatial order. Grids kept small (<=24) to run under CTest
  // in Debug in a fraction of a second; SpatialOrderIsTwoOnTheSlowerGrids
  // below re-measures the same order at the original {8,16,32,48} range.
  const double dt = 2.0e-4;
  const std::size_t n_steps = 10;  // T_final = 2e-3, well inside one period
  std::vector<double> hs, errs;
  for (const std::size_t n : {6u, 9u, 15u, 24u}) {
    const Errors e = run_and_measure(n, n, dt, n_steps, m);
    hs.push_back(m.L / static_cast<double>(n));
    errs.push_back(e.u_l2 + e.v_l2);
  }
  const double order = observed_order(hs, errs);
  std::printf("SpatialOrderIsTwo (n={6,9,15,24}): order=%.6f, errors=[%.6e, %.6e, %.6e, %.6e]\n",
             order, errs[0], errs[1], errs[2], errs[3]);
  // The REQUIRED order is 2 (this is a second-order scheme in the interior).
  // The mirror-ghost Dirichlet wall treatment for a tangential no-slip
  // velocity must not be LINEAR (ghost = 2D - c), which is exact only for an
  // AFFINE profile through the wall and costs roughly half an order globally
  // (order ~1.436-1.688 depending on grid range, below the required >1.8).
  // It is a QUADRATIC ghost through the
  // wall value D and the first two interior nodes c1, c2:
  // ghost = (8 D - 6 c1 + c2) / 3, exact for a QUADRATIC profile through
  // the wall -- see docs/user/fsi-channel.md's derivation and
  // stokes_solver.hpp's own comments at the u- and v-momentum boundary
  // rows. Measured after the fix: order back above 2 on this fast grid
  // range (see the printed line above for the exact number).
  EXPECT_GT(order, 1.8) << "observed spatial order " << order
                        << ", finest error " << errs.back();
  EXPECT_LT(order, 2.3) << "suspiciously high spatial order " << order;
}

TEST(FsiStokesMms, SpatialOrderIsTwoOnTheSlowerGrids) {
  // Re-confirms SpatialOrderIsTwo's measurement at the original grid range
  // this file used before it was shrunk for CTest speed, to rule out the
  // faster {6,9,15,24} grids hiding a resolution-dependent effect.
  const Manufactured m{.L = 1.3, .H = 0.7, .mu = 0.9, .rho = 1.1, .P0 = 0.6,
                       .Tm = 2.1};
  const double dt = 2.0e-4;
  const std::size_t n_steps = 10;
  std::vector<double> hs, errs;
  for (const std::size_t n : {8u, 16u, 32u, 48u}) {
    const Errors e = run_and_measure(n, n, dt, n_steps, m);
    hs.push_back(m.L / static_cast<double>(n));
    errs.push_back(e.u_l2 + e.v_l2);
  }
  const double order = observed_order(hs, errs);
  std::printf("SpatialOrderIsTwoOnTheSlowerGrids (n={8,16,32,48}): order=%.6f, "
             "errors=[%.6e, %.6e, %.6e, %.6e]\n",
             order, errs[0], errs[1], errs[2], errs[3]);
  EXPECT_GT(order, 1.8) << "observed spatial order " << order
                        << ", finest error " << errs.back();
  EXPECT_LT(order, 2.3) << "suspiciously high spatial order " << order;
}

// ---- pure-temporal accuracy: a manufactured solution the MAC scheme -------
// ---- reproduces EXACTLY in space, at ANY resolution -----------------------
//
// u(x,y,t) = sin(w t) * y,  v = 0,  p = 0.
//
// div(u,v) = du/dx + dv/dy = 0 + 0 = 0 identically (u does not depend on x;
// v is identically 0) -- exact, discretely, at any nx,ny.
// Lap(u) = d^2u/dy^2 = 0 (u is LINEAR in y): the QUADRATIC mirror-ghost
// Dirichlet construction this scheme uses at a wall (ghost =
// (8D - 6 c1 + c2)/3, see stokes_solver.hpp's own header comment) is
// exact for a quadratic profile through the boundary by construction, and
// a linear profile is a quadratic with a zero leading coefficient, so the
// discrete Laplacian is EXACTLY zero even in the boundary-adjacent row,
// not just in the interior -- verified directly below
// (SpatialExactnessHoldsAtAnyGrid) before relying on it.
// With Lap(u)=Lap(v)=0 and dp/dx=dp/dy=0 everywhere, the discrete momentum
// system degenerates to the SCALAR ODE rho*du/dt = fu, integrated by
// Crank-Nicolson's own trapezoidal rule -- so ANY residual error is PURE
// CN time-truncation, not spatial error, at any grid, and the ODE's
// solution sin(w t) is genuinely nonlinear in time (a linear-in-t case
// would make trapezoidal-rule exact and hide the O(dt^2) term entirely).
struct TemporalOnly {
  double H, rho, mu, omega;
  [[nodiscard]] double u(double /*x*/, double y, double t) const {
    return std::sin(omega * t) * y;
  }
  [[nodiscard]] double v(double, double, double) const { return 0.0; }
  [[nodiscard]] double fu(double /*x*/, double y, double t) const {
    return rho * omega * std::cos(omega * t) * y;
  }
  [[nodiscard]] double fv(double, double, double) const { return 0.0; }
};

fsi::ChannelBCs temporal_bcs(const TemporalOnly& m) {
  fsi::ChannelBCs bcs;
  bcs.left.u = [&m](double y, double t) { return m.u(0.0, y, t); };
  bcs.left.v = [&m](double, double) { return 0.0; };
  bcs.right.u = [&m](double y, double t) { return m.u(0.0, y, t); };
  bcs.right.v = [&m](double, double) { return 0.0; };
  bcs.bottom.u = [](double, double) { return 0.0; };
  bcs.bottom.v = [](double, double) { return 0.0; };
  bcs.top.u = [&m](double x, double t) { return m.u(x, m.H, t); };
  bcs.top.v = [](double, double) { return 0.0; };
  return bcs;
}

Errors run_temporal_only(std::size_t nx, std::size_t ny, double dt,
                         std::size_t n_steps, const TemporalOnly& m) {
  fsi::ChannelParams cp{.nx = nx, .ny = ny, .L = 1.0, .H = m.H, .mu = m.mu,
                       .rho = m.rho};
  fsi::SourceTerms src;
  src.fu = [&m](double x, double y, double t) { return m.fu(x, y, t); };
  src.fv = [&m](double x, double y, double t) { return m.fv(x, y, t); };
  const fsi::ChannelBCs bcs = temporal_bcs(m);
  const double dx = cp.L / static_cast<double>(nx);
  const double dy = m.H / static_cast<double>(ny);

  std::vector<double> u0((nx + 1) * ny), v0(nx * (ny + 1), 0.0);
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i <= nx; ++i) {
      u0[j * (nx + 1) + i] = m.u(static_cast<double>(i) * dx,
                                 (static_cast<double>(j) + 0.5) * dy, 0.0);
    }
  }
  fsi::PressurePin pin{0, ny - 1, 0.0};
  double t0 = 0.0;
  std::vector<double> u = u0, v = v0;
  for (std::size_t s = 0; s < n_steps; ++s) {
    const double t1 = t0 + dt;
    const fsi::FlowState next = fsi::cn_step(cp, bcs, src, dt, t0, t1, u, v, pin);
    u = next.u;
    v = next.v;
    t0 = t1;
  }
  double su = 0.0;
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i <= nx; ++i) {
      const double want = m.u(static_cast<double>(i) * dx,
                              (static_cast<double>(j) + 0.5) * dy, t0);
      const double e = u[j * (nx + 1) + i] - want;
      su += e * e * dx * dy;
    }
  }
  return Errors{std::sqrt(su), 0.0};
}

TEST(FsiStokesMms, SpatialExactnessHoldsAtAnyGridForTheTemporalFixture) {
  // Evidence for the claim above: at ONE dt, two different grids give the
  // SAME error to round-off, because there IS no spatial error to refine
  // away -- confirming run_temporal_only isolates pure time error before
  // the order test below relies on that.
  const TemporalOnly m{.H = 0.8, .rho = 1.3, .mu = 0.7, .omega = 12.0};
  const double dt = 1.0e-3;
  const Errors e1 = run_temporal_only(4, 3, dt, 5, m);
  const Errors e2 = run_temporal_only(11, 9, dt, 5, m);
  // Not the convergence-order gate -- a floating-point sanity check that
  // two DIFFERENTLY-SIZED SparseLU factorizations of an exactly-reproduced
  // (to the scheme's own arithmetic) solution agree.
  //
  // Measured e1=6.5891e-08, e2=6.4599e-08, so the errors
  // THEMSELVES are about 6.5x the 1.0e-8 threshold below (not ">50x
  // below"). What the
  // EXPECT_LT actually compares is |e1-e2| = 1.2926e-09, which IS about
  // 50x SMALLER than e1/e2 themselves (confirming "near-round-off
  // RELATIVE TO THE ERRORS", the claim this test is actually evidence
  // for), but the 1.0e-8 THRESHOLD sits only ~7.7x above that difference
  // -- a real but modest margin. 1.0e-8 is kept as-is: SparseLU on two
  // differently-sized systems is not expected to agree closer than this
  // (the 7.7x margin already absorbs that), and tightening it further
  // risks flaking on a different machine's BLAS/LAPACK without buying
  // additional confidence that the fixture is exact -- the exactness
  // claim itself rests on the ANALYTIC argument in this test's own
  // header comment (a quadratic velocity is Laplacian-exact on this
  // stencil), not on how tight this floating-point check is.
  EXPECT_LT(std::abs(e1.u_l2 - e2.u_l2), 1.0e-8)
      << "e1=" << e1.u_l2 << " e2=" << e2.u_l2
      << " -- these should agree to near-round-off (several orders below "
         "the errors themselves) if the fixture is truly spatially exact";
}

TEST(FsiStokesMms, TemporalOrderIsTwo) {
  const TemporalOnly m{.H = 0.8, .rho = 1.3, .mu = 0.7, .omega = 12.0};
  // Small, fixed grid throughout -- spatially exact (see the test above),
  // so it contributes nothing to refine away, and stays fast under CTest.
  constexpr std::size_t nx = 6, ny = 5;
  const double t_final = 0.05;  // well inside one period (2*pi/omega ~ 0.52)
  std::vector<double> hs, errs;
  for (const std::size_t n_steps : {4u, 8u, 16u, 32u}) {
    const double dt = t_final / static_cast<double>(n_steps);
    const Errors e = run_temporal_only(nx, ny, dt, n_steps, m);
    hs.push_back(dt);
    errs.push_back(e.u_l2);
  }
  const double order = observed_order(hs, errs);
  EXPECT_GT(order, 1.8) << "observed temporal order " << order
                        << ", finest error " << errs.back();
  EXPECT_LT(order, 2.3) << "suspiciously high temporal order " << order;
}

// The dt-keyed LU cache (`fsi::CnStepLuCache`, stokes_solver.hpp) is
// sound only if dt is the ONLY per-call argument that changes A. `p0` is not
// rhs-only: the pressure-gradient coefficient is 0.5/dx when p0 is supplied
// (trapezoidal) and 1/dx when it is not (fully implicit), so A itself differs.
// A cache seeded by an implicit-only call must not be reused for a
// trapezoidal call at the same dt; the cached result has to equal the
// always-fresh one bit for bit (flag off, the backend IS Eigen::SparseLU).
TEST(FsiStokesLuCache, DoesNotReuseAnImplicitFactorisationForATrapezoidalStep) {
  const fsi::ChannelParams cp{.nx = 10, .ny = 6, .L = 1.0, .H = 0.2, .mu = 0.01, .rho = 1.0};
  fsi::ChannelBCs bcs;
  bcs.left.u = [](double y, double) { return 4.0 * y * (0.2 - y) / (0.2 * 0.2); };
  bcs.right.outflow = true;
  const fsi::SourceTerms src;
  const std::vector<double> u0((cp.nx + 1) * cp.ny, 0.0), v0(cp.nx * (cp.ny + 1), 0.0);
  const fsi::PressurePin pin{cp.nx - 1, 0, 0.0};
  const double dt = 0.01;

  fsi::CnStepLuCache cache;
  // Seed the cache with the fully implicit matrix (p0 == nullptr).
  const fsi::FlowState implicit =
      fsi::cn_step(cp, bcs, src, dt, 0.0, dt, u0, v0, pin, nullptr, nullptr, &cache);
  ASSERT_TRUE(cache.valid);
  // A non-trivial p0 (the implicit step's own pressure), same dt.
  const std::vector<double> p0 = implicit.p;
  ASSERT_GT(std::abs(*std::max_element(p0.begin(), p0.end())), 0.0);

  const fsi::FlowState fresh =
      fsi::cn_step(cp, bcs, src, dt, 0.0, dt, u0, v0, pin, nullptr, &p0, nullptr);
  const fsi::FlowState cached =
      fsi::cn_step(cp, bcs, src, dt, 0.0, dt, u0, v0, pin, nullptr, &p0, &cache);

  ASSERT_EQ(cached.u.size(), fresh.u.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < fresh.u.size(); ++i) worst = std::max(worst, std::abs(cached.u[i] - fresh.u[i]));
  for (std::size_t i = 0; i < fresh.v.size(); ++i) worst = std::max(worst, std::abs(cached.v[i] - fresh.v[i]));
  for (std::size_t i = 0; i < fresh.p.size(); ++i) worst = std::max(worst, std::abs(cached.p[i] - fresh.p[i]));
  EXPECT_EQ(worst, 0.0) << "cached trapezoidal step differs from the fresh one by " << worst
                        << ": the cache reused a factorisation of the wrong matrix";
}
