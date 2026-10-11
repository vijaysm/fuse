#pragma once

/// Unsteady, incompressible Stokes flow in a 2D channel [0,L] x [0,H],
///
///   rho u_t = -grad p + mu Lap(u),   div u = 0,
///
/// on a structured, staggered MAC grid (Nx cells in x, Ny cells in y):
/// u lives on the Nx+1 vertical cell faces per row ((Nx+1) x Ny values),
/// v lives on the Ny+1 horizontal cell faces per column (Nx x (Ny+1)
/// values), p lives at the Nx x Ny cell centres. This is the classical
/// Harlow-Welch staggering, chosen so the discrete divergence and gradient
/// are exact adjoints and the saddle-point system below has no spurious
/// pressure modes on a structured grid.
///
/// TIME: Crank-Nicolson (trapezoidal), i.e. the viscous term is averaged
/// between t0 and t1 and every boundary value is imposed strongly AT ITS
/// OWN time level (t1 for the implicit array, t0 for the frozen array) --
/// second order in time, unconditionally stable for this linear problem.
///
/// SPACE: every interior momentum row is the standard 5-point Laplacian.
/// A momentum unknown exactly ON a Dirichlet boundary (v at the bottom and
/// top walls, u/v at an inflow/Dirichlet-both boundary) gets a trivial
/// identity row. A momentum unknown one HALF-cell in from a wall (u's
/// tangential value near the top and bottom walls, v's near the left/right
/// boundaries) needs a wall value that sits between two grid points, which
/// is supplied by the standard mirror-ghost construction: with D the wall's
/// Dirichlet value and c the nearest real node, the ghost obeys
/// (ghost + c)/2 = D, i.e. ghost = 2D - c -- linear reconstruction through D
/// at the exact wall location. Substituting the ghost into the 5-point
/// stencil folds it into that row's own diagonal (see `docs/user/fsi-
/// channel.md`'s worked derivation) rather than adding a new unknown.
/// A ZERO-CURVATURE (`outflow = true`) side instead extrapolates the ghost
/// linearly through the two nearest real neighbours (the discrete
/// d^2V/dn^2 = 0 condition), with no wall value at all -- a plain
/// zero-GRADIENT ghost (an earlier, wrong version of this file) instead
/// degenerates the adjacent cell's own continuity equation into a hard
/// constraint incompatible with a nonzero transpiration velocity; see
/// the v-momentum assembly's own comment for the derivation.
///
/// PRESSURE GAUGE. Every boundary here is a velocity condition (Dirichlet
/// or zero-gradient), so pressure appears only through its OWN gradient and
/// the linear system is singular by exactly one degree of freedom (a
/// constant shift). One continuity row is replaced by a trivial identity
/// `p(pin) = pin_value` to fix the gauge, exactly as the task's MODEL
/// section specifies.
///
/// THE SADDLE-POINT SYSTEM `[A B^T; B 0] x = rhs` (A the momentum block, B
/// the discrete divergence) is assembled fresh every `cn_step()` call -- the
/// RHS always needs it, since it is built in the same triplet loops as A --
/// but the `Eigen::SparseLU` FACTORISATION of A is now cached and reused
/// across `StokesSolver::step()` calls whenever `dt` is unchanged (see
/// `CnStepLuCache`, `StokesSolver::lu_cache_`, below). This used to be an
/// unconditional "assemble and refactorise every step, never cache" policy,
/// justified by this project's standing rule ("robustness and accuracy
/// beats efficiency always") on the grounds that a fresh factorisation can
/// never go stale relative to a boundary condition that changes shape or a
/// solver re-tared with different parameters mid-run. That risk does not
/// actually apply to `StokesSolver`: it has NO public setter for `p_` (grid,
/// mu, rho) or `pin_` -- both are fixed for the object's entire lifetime by
/// the constructor -- and `step()` rebuilds `bcs` with a hardcoded,
/// unchanging shape (no-slip top/bottom, Dirichlet/outflow left/right) on
/// every call. `dt` is therefore the ONLY input that can make A differ from
/// one `step()` call to the next on a given instance, and the cache is keyed
/// on exactly that value: the factorisation is recomputed whenever `dt`
/// changes and reused, bit-for-bit, only when it doesn't -- A's own numeric
/// values are then bit-identical to the matrix the cached factors came from,
/// so this changes wall-clock only, never the computed field. `cn_step()`
/// itself, called directly (not through `StokesSolver`) by
/// `fsi_stokes_mms_test.cpp` across a sweep of DIFFERENT grids/mu/rho/bcs,
/// passes no cache (`lu_cache = nullptr`, the default) and keeps the
/// original always-fresh-factorisation behaviour exactly, since nothing
/// about ITS inputs is guaranteed fixed across calls. This mirrors
/// `BeamSolver::rebuild_effective_midpoint()`'s own
/// `keff2_lu_`/`cached_dt2_`/`keff2_valid_` pattern in this same file
/// (`beam_solver.hpp`) -- same justification, same guard shape, applied here
/// to the fluid side's saddle-point solve. The other per-call arguments
/// (`top_wall_v_old`/`top_wall_v_new`/`inflow_u`, `t0`/`t1`, `u0`/`v0`) all
/// flow only into `rhs` -- every `bcs.*` closure that reads them lands in an
/// identity row (`trip.emplace_back(row, row, 1.0)`) contributing a fixed
/// `1.0` to A and the time-varying value to `rhs` alone -- so none of them
/// can invalidate a `dt`-keyed cache.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Sparse>

namespace fsi {

/// One side of the channel. `outflow` selects the zero-gradient (Neumann)
/// condition; otherwise both `u` and `v` are evaluated as Dirichlet values,
/// each a function of (coordinate along the boundary, time) -- the
/// coordinate is y for Left/Right, x for Bottom/Top; the caller (the
/// assembler below) always passes the coordinate of the exact grid point
/// the value is needed at, never a cell index.
struct WallBC {
  bool outflow = false;
  std::function<double(double, double)> u = [](double, double) { return 0.0; };
  std::function<double(double, double)> v = [](double, double) { return 0.0; };
};

struct ChannelBCs {
  WallBC left, right, bottom, top;
};

/// Optional body force for the method-of-manufactured-solutions test. Zero
/// by default, i.e. an unforced physical run.
struct SourceTerms {
  std::function<double(double, double, double)> fu =
      [](double, double, double) { return 0.0; };
  std::function<double(double, double, double)> fv =
      [](double, double, double) { return 0.0; };
};

struct ChannelParams {
  std::size_t nx = 24, ny = 12;
  double      L = 1.0, H = 1.0;
  double      mu = 1.0, rho = 1.0;
};

/// Where the pressure gauge is pinned, and to what value.
struct PressurePin {
  std::size_t i = 0, j = 0;
  double      value = 0.0;
};

struct FlowState {
  std::vector<double> u, v, p;
  // KNOWN_ISSUES.md D9 (this round): the wall traction delivered to the
  // beam, at t_{n+1/2} -- the SAME time level Crank-Nicolson's own
  // trapezoidal terms live at (see StokesSolver::step()'s own comment for
  // the full derivation). It is PART OF FlowState, not a separate member,
  // specifically so it rides through save()/restore()/set_state() and the
  // implicit coupling group's checkpoint verification correctly: a solver-
  // private member `wall_traction()` depends on, that `save()`/`restore()`
  // does not carry, breaks that check the moment it changes between a
  // save and its matching restore -- an earlier round's own experiment
  // with a similarly-unsaved member hit exactly this failure mode.
  std::vector<double> traction_mid;
};

namespace detail {

[[nodiscard]] inline std::size_t idx_u(const ChannelParams& c, std::size_t i,
                                      std::size_t j) {
  return j * (c.nx + 1) + i;
}
[[nodiscard]] inline std::size_t idx_v(const ChannelParams& c, std::size_t i,
                                      std::size_t j) {
  return j * c.nx + i;
}
[[nodiscard]] inline std::size_t idx_p(const ChannelParams& c, std::size_t i,
                                      std::size_t j) {
  return j * c.nx + i;
}

}  // namespace detail

/// Optional cross-call cache for `cn_step`'s saddle-point factorisation.
/// When non-null, `cn_step` reuses the cached `Eigen::SparseLU` numeric
/// factorisation whenever `dt` matches the value it was built from, instead
/// of refactorising an assembled-but-numerically-identical A. Safe ONLY if
/// every OTHER input that feeds the matrix (grid, mu, rho, pin, bcs shape)
/// also stays fixed across every call sharing one cache -- see `cn_step`'s
/// own header comment above for why that holds for `StokesSolver::step()`
/// specifically. A caller that cannot make that guarantee (e.g.
/// `fsi_stokes_mms_test.cpp`, which varies grid/mu/rho/bcs across calls)
/// passes `nullptr` (the default) and gets the original
/// always-fresh-factorisation behaviour, unchanged.
struct CnStepLuCache {
  Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
  double cached_dt = std::numeric_limits<double>::quiet_NaN();
  bool   valid = false;
};

/// One Crank-Nicolson step of the saddle-point system above, from
/// `(u0, v0)` at time `t0` to the returned state at `t1 = t0 + dt`.
/// `bcs` is evaluated at `t1` for the implicit (unknown) array's boundary
/// terms and at `t0` for the terms that come from the known array `(u0,
/// v0)` -- both arrays are boundary-inclusive, i.e. `u0`/`v0` already carry
/// whatever value satisfied the boundary condition at `t0`. `src`, if
/// non-zero, is averaged between `t0` and `t1` (the same trapezoidal rule
/// as the viscous term) so the forcing does not degrade the scheme's second
/// order in time.
[[nodiscard]] inline FlowState cn_step(const ChannelParams& c,
                                      const ChannelBCs& bcs,
                                      const SourceTerms& src, double dt,
                                      double t0, double t1,
                                      const std::vector<double>& u0,
                                      const std::vector<double>& v0,
                                      const PressurePin& pin,
                                      // Optional: if non-null, the assembled
                                      // saddle-point matrix is copied out for
                                      // offline diagnosis (condition number,
                                      // null-space inspection) -- not used by
                                      // any production caller.
                                      Eigen::SparseMatrix<double>* debug_matrix_out = nullptr,
                                      // Optional: if non-null, the pressure
                                      // GRADIENT in the momentum equation is
                                      // TRAPEZOIDAL (0.5*Gp1 + 0.5*Gp0, this
                                      // array supplying Gp0) instead of fully
                                      // implicit (Gp1 alone). Root-caused and
                                      // added this session: a fully-implicit
                                      // pressure gradient, paired with an
                                      // otherwise-trapezoidal (viscous-term-
                                      // averaged) momentum balance, makes the
                                      // solved pressure best approximate the
                                      // CONTINUOUS p at t_{n+1/2}, not t_{n+1}
                                      // -- confirmed directly against a
                                      // manufactured solution (order ~0 vs
                                      // t1/t0, smallest and most consistent
                                      // residual vs t_mid). Delivering THAT
                                      // pressure to a partitioned coupling
                                      // partner UNDER THE LABEL "the load at
                                      // t_{n+1}" is exactly the classic
                                      // partitioned-FSI time-level mismatch;
                                      // see docs/user/fsi-channel.md. nullptr
                                      // (the default) preserves the ORIGINAL,
                                      // fully-implicit behaviour exactly, so
                                      // every existing direct cn_step() caller
                                      // (the MMS tests) is unaffected; only
                                      // StokesSolver::step() passes one, so
                                      // FluidComponent's own p1 is genuinely
                                      // t1-accurate.
                                      const std::vector<double>* p0 = nullptr,
                                      // Optional: see CnStepLuCache's own
                                      // comment. nullptr (the default)
                                      // preserves the original
                                      // always-fresh-factorisation behaviour
                                      // exactly.
                                      CnStepLuCache* lu_cache = nullptr) {
  using detail::idx_p;
  using detail::idx_u;
  using detail::idx_v;
  const std::size_t nx = c.nx, ny = c.ny;
  const double dx = c.L / static_cast<double>(nx);
  const double dy = c.H / static_cast<double>(ny);
  const double rho = c.rho, mu = c.mu;

  const std::size_t Nu = (nx + 1) * ny;
  const std::size_t Nv = nx * (ny + 1);
  const std::size_t Np = nx * ny;
  const std::size_t N = Nu + Nv + Np;
  const std::size_t ov = Nu, op = Nu + Nv;

  if (u0.size() != Nu || v0.size() != Nv) {
    throw std::invalid_argument("cn_step: u0/v0 size mismatch with (nx,ny)");
  }

  std::vector<Eigen::Triplet<double>> trip;
  trip.reserve(N * 6);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(N));

  const double idx2 = 1.0 / (dx * dx), idy2 = 1.0 / (dy * dy);
  const double halfmu = 0.5 * mu;

  // ---- u-momentum: i = 0..nx, j = 0..ny-1 ----------------------------------
  for (std::size_t j = 0; j < ny; ++j) {
    const double y_half = (static_cast<double>(j) + 0.5) * dy;
    for (std::size_t i = 0; i <= nx; ++i) {
      const std::size_t row = idx_u(c, i, j);
      if (i == 0) {
        if (bcs.left.outflow) {
          // Zero-curvature (linear-extrapolation) outflow, NOT zero-
          // gradient: u(0,j) = 2 u(1,j) - u(2,j). A hard zero-gradient
          // identity (u(0,j)=u(1,j)) would force the SAME row's own
          // continuity equation to demand a zero y-derivative of v there
          // too -- see the v-momentum block's comment for why that
          // contradicts a nonzero transpiration velocity. Requires nx>=2,
          // always true for a channel with more than one wall-face column.
          trip.emplace_back(static_cast<int>(row), static_cast<int>(row), 1.0);
          trip.emplace_back(static_cast<int>(row),
                            static_cast<int>(idx_u(c, 1, j)), -2.0);
          trip.emplace_back(static_cast<int>(row),
                            static_cast<int>(idx_u(c, 2, j)), 1.0);
          rhs[static_cast<Eigen::Index>(row)] = 0.0;
        } else {
          trip.emplace_back(static_cast<int>(row), static_cast<int>(row), 1.0);
          rhs[static_cast<Eigen::Index>(row)] = bcs.left.u(y_half, t1);
        }
        continue;
      }
      if (i == nx) {
        if (bcs.right.outflow) {
          trip.emplace_back(static_cast<int>(row), static_cast<int>(row), 1.0);
          trip.emplace_back(static_cast<int>(row),
                            static_cast<int>(idx_u(c, nx - 1, j)), -2.0);
          trip.emplace_back(static_cast<int>(row),
                            static_cast<int>(idx_u(c, nx - 2, j)), 1.0);
          rhs[static_cast<Eigen::Index>(row)] = 0.0;
        } else {
          trip.emplace_back(static_cast<int>(row), static_cast<int>(row), 1.0);
          rhs[static_cast<Eigen::Index>(row)] = bcs.right.u(y_half, t1);
        }
        continue;
      }
      const double x_i = static_cast<double>(i) * dx;
      if (ny < 2) {
        throw std::invalid_argument(
            "cn_step: ny>=2 required for the quadratic wall ghost's second "
            "interior node");
      }

      // Implicit side (unknown array), evaluated at t1.
      double diag = rho / dt + halfmu * 2.0 * idx2;
      trip.emplace_back(static_cast<int>(row), static_cast<int>(idx_u(c, i + 1, j)),
                        -halfmu * idx2);
      trip.emplace_back(static_cast<int>(row), static_cast<int>(idx_u(c, i - 1, j)),
                        -halfmu * idx2);
      double rhs_const_implicit = 0.0;

      // y-direction: no-slip tangential Dirichlet walls (bottom, top) use a
      // QUADRATIC ghost through the wall value D and the first two interior
      // nodes c1 (this row's own value, one half-cell from the wall) and c2
      // (the next node inward): ghost = (8 D - 6 c1 + c2) / 3, the value at
      // y = -dy/2 (bottom) or y = H + dy/2 (top) of the UNIQUE quadratic
      // through (0,D), (dy/2,c1), (3dy/2,c2) -- exact for a quadratic
      // profile, unlike the LINEAR mirror ghost = 2D - c1 this file used
      // before (exact only for an affine profile, hence the O(h) local
      // truncation at these rows that cost the scheme roughly half an order
      // globally; see docs/user/fsi-channel.md and
      // FsiStokesMms.SpatialOrderIsTwo). Substituting into the 3-point
      // y-Laplacian (c2 - 2 c1 + ghost)/dy^2 gives
      //   Lap_y = (8 D - 12 c1 + 4 c2) / (3 dy^2),
      // so the row's own diagonal gets +4*halfmu*idy2 (not the interior
      // +2*halfmu*idy2), the neighbour c2 gets -(4/3)*halfmu*idy2 (not the
      // interior -halfmu*idy2), and D's constant term moves to the RHS as
      // +(8/3)*halfmu*idy2*D.
      if (j == 0) {
        diag += 4.0 * halfmu * idy2;
        trip.emplace_back(static_cast<int>(row),
                          static_cast<int>(idx_u(c, i, j + 1)),
                          -(4.0 / 3.0) * halfmu * idy2);
        rhs_const_implicit += (8.0 / 3.0) * halfmu * idy2 * bcs.bottom.u(x_i, t1);
      } else if (j == ny - 1) {
        diag += 4.0 * halfmu * idy2;
        trip.emplace_back(static_cast<int>(row),
                          static_cast<int>(idx_u(c, i, j - 1)),
                          -(4.0 / 3.0) * halfmu * idy2);
        rhs_const_implicit += (8.0 / 3.0) * halfmu * idy2 * bcs.top.u(x_i, t1);
      } else {
        diag += 2.0 * halfmu * idy2;
        trip.emplace_back(static_cast<int>(row),
                          static_cast<int>(idx_u(c, i, j - 1)), -halfmu * idy2);
        trip.emplace_back(static_cast<int>(row),
                          static_cast<int>(idx_u(c, i, j + 1)), -halfmu * idy2);
      }
      trip.emplace_back(static_cast<int>(row), static_cast<int>(row), diag);
      // Pressure gradient: TRAPEZOIDAL (half implicit p1, half explicit p0)
      // when p0 is supplied, fully implicit (p1 alone) otherwise -- see
      // this function's own header comment for why.
      const double p_coeff = p0 != nullptr ? 0.5 / dx : 1.0 / dx;
      trip.emplace_back(static_cast<int>(row), static_cast<int>(op + idx_p(c, i, j)),
                        p_coeff);
      trip.emplace_back(static_cast<int>(row),
                        static_cast<int>(op + idx_p(c, i - 1, j)), -p_coeff);
      if (p0 != nullptr) {
        rhs_const_implicit -= 0.5 * ((*p0)[idx_p(c, i, j)] - (*p0)[idx_p(c, i - 1, j)]) / dx;
      }

      // Explicit side (known array u0), evaluated at t0 -- a plain number.
      // Same quadratic-ghost Laplacian as the implicit side above, applied
      // to the KNOWN t0 values instead of assembling matrix rows.
      const double c0 = u0[idx_u(c, i, j)];
      const double e0 = u0[idx_u(c, i + 1, j)];
      const double w0 = u0[idx_u(c, i - 1, j)];
      double lap_y0;
      if (j == 0) {
        const double c2 = u0[idx_u(c, i, j + 1)];
        const double D = bcs.bottom.u(x_i, t0);
        lap_y0 = (8.0 * D - 12.0 * c0 + 4.0 * c2) * idy2 / 3.0;
      } else if (j == ny - 1) {
        const double c2 = u0[idx_u(c, i, j - 1)];
        const double D = bcs.top.u(x_i, t0);
        lap_y0 = (8.0 * D - 12.0 * c0 + 4.0 * c2) * idy2 / 3.0;
      } else {
        const double s0 = u0[idx_u(c, i, j - 1)];
        const double n0 = u0[idx_u(c, i, j + 1)];
        lap_y0 = (n0 - 2.0 * c0 + s0) * idy2;
      }
      const double lap_u0 = (e0 - 2.0 * c0 + w0) * idx2 + lap_y0;

      const double f_avg =
          0.5 * (src.fu(x_i, y_half, t1) + src.fu(x_i, y_half, t0));

      // rhs_const_implicit already carries the sign that moves the ghost's
      // constant term from the implicit Laplacian (LHS) to the RHS -- see
      // docs/user/fsi-channel.md's worked derivation for why this is a
      // PLUS, not the more intuitive-looking minus.
      rhs[static_cast<Eigen::Index>(row)] =
          rho / dt * c0 + halfmu * lap_u0 + rhs_const_implicit + f_avg;
    }
  }

  // ---- v-momentum: i = 0..nx-1, j = 0..ny ----------------------------------
  for (std::size_t i = 0; i < nx; ++i) {
    const double x_half = (static_cast<double>(i) + 0.5) * dx;
    for (std::size_t j = 0; j <= ny; ++j) {
      const std::size_t row = ov + idx_v(c, i, j);
      if (j == 0) {
        trip.emplace_back(static_cast<int>(row), static_cast<int>(row), 1.0);
        rhs[static_cast<Eigen::Index>(row)] = bcs.bottom.v(x_half, t1);
        continue;
      }
      if (j == ny) {
        trip.emplace_back(static_cast<int>(row), static_cast<int>(row), 1.0);
        rhs[static_cast<Eigen::Index>(row)] = bcs.top.v(x_half, t1);
        continue;
      }
      const double y_j = static_cast<double>(j) * dy;

      // y-direction: always the plain interior 3-point stencil (both
      // neighbours are real -- j==0/j==ny are handled by the identity rows
      // above, so 0<j<ny here).
      double diag = rho / dt + halfmu * 2.0 * idy2;
      trip.emplace_back(static_cast<int>(row), static_cast<int>(ov + idx_v(c, i, j + 1)),
                        -halfmu * idy2);
      trip.emplace_back(static_cast<int>(row), static_cast<int>(ov + idx_v(c, i, j - 1)),
                        -halfmu * idy2);
      double rhs_const_implicit = 0.0;

      // x-direction. A ZERO-CURVATURE (linear-extrapolation) outflow ghost
      // on EITHER side -- v(boundary+1) = 2*v(boundary) - v(boundary-1) --
      // makes the x-Laplacian's east/west pair cancel EXACTLY:
      //   (ghost - 2c + real)/dx^2 = ((2c-real) - 2c + real)/dx^2 = 0,
      // identically, for ANY c and real. So a row adjacent to an outflow
      // side on x carries NO x-direction contribution at all (not even from
      // its own real neighbour on the other side, because that neighbour IS
      // the "real" value the cancellation above already consumed) -- see
      // docs/user/fsi-channel.md's derivation of why the SIMPLER zero-
      // gradient ghost this file used to use is wrong: it degenerates the
      // adjacent cell's OWN continuity equation into a hard v-is-constant
      // constraint, contradicting any nonzero transpiration velocity.
      const bool west_outflow = (i == 0) && bcs.left.outflow;
      const bool east_outflow = (i == nx - 1) && bcs.right.outflow;
      if (west_outflow) {
        // GHOST-FREE one-sided second difference over the three nearest
        // REAL interior points (i, i+1, i+2), d^2v/dx^2 ~ (v(i)-2v(i+1)+
        // v(i+2))/dx^2 -- first-order, but introduces no invented ghost
        // value at all. Requires nx>=3. A CENTRED ghost formula here (this
        // file's two earlier attempts: zero-gradient, then zero-curvature)
        // couples this row's x-block either into a hard, load-incompatible
        // constraint on the SAME column's own continuity equation, or --
        // less obviously, but confirmed by direct perturbation testing
        // (build-evidence/mutations/fsi.toml) -- into a genuine near-null
        // direction spanning the two outflow-adjacent columns together.
        // The one-sided real-point stencil has neither failure mode: it
        // only ever references real, independently-governed unknowns.
        diag += -halfmu * idx2;
        trip.emplace_back(static_cast<int>(row),
                          static_cast<int>(ov + idx_v(c, i + 1, j)), 2.0 * halfmu * idx2);
        trip.emplace_back(static_cast<int>(row),
                          static_cast<int>(ov + idx_v(c, i + 2, j)), -halfmu * idx2);
      } else if (east_outflow) {
        diag += -halfmu * idx2;
        trip.emplace_back(static_cast<int>(row),
                          static_cast<int>(ov + idx_v(c, i - 1, j)), 2.0 * halfmu * idx2);
        trip.emplace_back(static_cast<int>(row),
                          static_cast<int>(ov + idx_v(c, i - 2, j)), -halfmu * idx2);
      } else {
        // x-direction, non-outflow (Dirichlet tangential v): QUADRATIC
        // ghost through the wall value D and the first two interior nodes
        // c1 (this row's own value), c2 (the next node inward) -- same
        // construction and derivation as u-momentum's bottom/top walls
        // above. Requires nx>=2.
        if (nx < 2) {
          throw std::invalid_argument(
              "cn_step: nx>=2 required for the quadratic wall ghost's "
              "second interior node");
        }
        if (i == 0) {
          diag += 4.0 * halfmu * idx2;
          trip.emplace_back(static_cast<int>(row),
                            static_cast<int>(ov + idx_v(c, i + 1, j)),
                            -(4.0 / 3.0) * halfmu * idx2);
          rhs_const_implicit += (8.0 / 3.0) * halfmu * idx2 * bcs.left.v(y_j, t1);
        } else if (i == nx - 1) {
          diag += 4.0 * halfmu * idx2;
          trip.emplace_back(static_cast<int>(row),
                            static_cast<int>(ov + idx_v(c, i - 1, j)),
                            -(4.0 / 3.0) * halfmu * idx2);
          rhs_const_implicit += (8.0 / 3.0) * halfmu * idx2 * bcs.right.v(y_j, t1);
        } else {
          diag += halfmu * 2.0 * idx2;
          trip.emplace_back(static_cast<int>(row),
                            static_cast<int>(ov + idx_v(c, i - 1, j)), -halfmu * idx2);
          trip.emplace_back(static_cast<int>(row),
                            static_cast<int>(ov + idx_v(c, i + 1, j)), -halfmu * idx2);
        }
      }
      trip.emplace_back(static_cast<int>(row), static_cast<int>(row), diag);
      const double p_coeff_v = p0 != nullptr ? 0.5 / dy : 1.0 / dy;
      trip.emplace_back(static_cast<int>(row), static_cast<int>(op + idx_p(c, i, j)),
                        p_coeff_v);
      trip.emplace_back(static_cast<int>(row),
                        static_cast<int>(op + idx_p(c, i, j - 1)), -p_coeff_v);
      if (p0 != nullptr) {
        rhs_const_implicit -= 0.5 * ((*p0)[idx_p(c, i, j)] - (*p0)[idx_p(c, i, j - 1)]) / dy;
      }

      const double c0 = v0[idx_v(c, i, j)];
      const double n0 = v0[idx_v(c, i, j + 1)];
      const double s0 = v0[idx_v(c, i, j - 1)];
      double lap_x0 = 0.0;
      if (west_outflow) {
        lap_x0 = (c0 - 2.0 * v0[idx_v(c, i + 1, j)] + v0[idx_v(c, i + 2, j)]) * idx2;
      } else if (east_outflow) {
        lap_x0 = (c0 - 2.0 * v0[idx_v(c, i - 1, j)] + v0[idx_v(c, i - 2, j)]) * idx2;
      } else if (i == 0) {
        const double c2 = v0[idx_v(c, i + 1, j)];
        const double D = bcs.left.v(y_j, t0);
        lap_x0 = (8.0 * D - 12.0 * c0 + 4.0 * c2) * idx2 / 3.0;
      } else if (i == nx - 1) {
        const double c2 = v0[idx_v(c, i - 1, j)];
        const double D = bcs.right.v(y_j, t0);
        lap_x0 = (8.0 * D - 12.0 * c0 + 4.0 * c2) * idx2 / 3.0;
      } else {
        const double w0 = v0[idx_v(c, i - 1, j)];
        const double e0 = v0[idx_v(c, i + 1, j)];
        lap_x0 = (e0 - 2.0 * c0 + w0) * idx2;
      }
      const double lap_v0 = lap_x0 + (n0 - 2.0 * c0 + s0) * idy2;

      const double f_avg =
          0.5 * (src.fv(x_half, y_j, t1) + src.fv(x_half, y_j, t0));

      rhs[static_cast<Eigen::Index>(row)] =
          rho / dt * c0 + halfmu * lap_v0 + rhs_const_implicit + f_avg;
    }
  }

  // ---- continuity: i = 0..nx-1, j = 0..ny-1 --------------------------------
  //
  // PRESSURE GAUGE. A single interior pin removes only the pressure
  // field's additive constant -- the "d.c. offset" freedom every all-
  // velocity-boundary Stokes problem has. With an OUTFLOW boundary
  // (velocity left floating there), the discrete operator carries a
  // SECOND, genuine null direction: a uniform shift added to u for every
  // column except the fixed-Dirichlet inflow, paired with a pressure ramp
  // whose slope exactly cancels it in every interior momentum row (the
  // outflow's own zero-curvature/zero-gradient row is, by construction,
  // satisfied by ANY affine-in-i perturbation, so it offers no resistance
  // -- confirmed directly: `JacobiSVD` on this system's small-grid
  // assembly gives a singular value at 1e-17, i.e. machine-zero, and its
  // right singular vector has exactly this shape). One point-pin cannot
  // remove BOTH the offset and the slope of an affine mode; the standard,
  // properly-posed fix is a PRESSURE-OUTLET condition: p is Dirichlet
  // (here, 0) along the WHOLE outflow column, not just at one interior
  // cell -- fixing p at every row of that column pins both the offset and
  // the slope at once, and is the conventional outflow treatment (velocity
  // floats, pressure is referenced) rather than an ad hoc extra pin.
  // `pin` becomes irrelevant whenever `bcs.right.outflow` is set; it still
  // serves its original single-point role for a fully-Dirichlet channel
  // (the MMS verification configuration), which carries no outflow and
  // therefore no second null direction.
  for (std::size_t j = 0; j < ny; ++j) {
    for (std::size_t i = 0; i < nx; ++i) {
      const std::size_t row = op + idx_p(c, i, j);
      if (bcs.right.outflow && i == nx - 1) {
        trip.emplace_back(static_cast<int>(row), static_cast<int>(row), 1.0);
        rhs[static_cast<Eigen::Index>(row)] = 0.0;
        continue;
      }
      if (!bcs.right.outflow && i == pin.i && j == pin.j) {
        trip.emplace_back(static_cast<int>(row), static_cast<int>(row), 1.0);
        rhs[static_cast<Eigen::Index>(row)] = pin.value;
        continue;
      }
      trip.emplace_back(static_cast<int>(row),
                        static_cast<int>(idx_u(c, i + 1, j)), 1.0 / dx);
      trip.emplace_back(static_cast<int>(row), static_cast<int>(idx_u(c, i, j)),
                        -1.0 / dx);
      trip.emplace_back(static_cast<int>(row),
                        static_cast<int>(ov + idx_v(c, i, j + 1)), 1.0 / dy);
      trip.emplace_back(static_cast<int>(row), static_cast<int>(ov + idx_v(c, i, j)),
                        -1.0 / dy);
      rhs[static_cast<Eigen::Index>(row)] = 0.0;
    }
  }

  Eigen::SparseMatrix<double> A(static_cast<int>(N), static_cast<int>(N));
  A.setFromTriplets(trip.begin(), trip.end());
  A.makeCompressed();
  if (debug_matrix_out) *debug_matrix_out = A;

  // See CnStepLuCache's own comment: lu_cache == nullptr (every direct
  // cn_step() caller except StokesSolver::step()) takes need_factor's
  // unconditional-true branch below, identical to the original
  // unconditional `lu.compute(A)` every call.
  Eigen::SparseLU<Eigen::SparseMatrix<double>> local_lu;
  Eigen::SparseLU<Eigen::SparseMatrix<double>>* lu = &local_lu;
  bool need_factor = true;
  if (lu_cache != nullptr) {
    lu = &lu_cache->lu;
    need_factor = !lu_cache->valid || lu_cache->cached_dt != dt;
  }
  if (need_factor) {
    lu->compute(A);
    if (lu->info() != Eigen::Success) {
      throw std::runtime_error("cn_step: SparseLU factorization failed");
    }
    if (lu_cache != nullptr) {
      lu_cache->cached_dt = dt;
      lu_cache->valid = true;
    }
  }
  const Eigen::VectorXd sol = lu->solve(rhs);
  if (lu->info() != Eigen::Success) {
    throw std::runtime_error("cn_step: SparseLU solve failed");
  }

  FlowState out;
  out.u.resize(Nu);
  out.v.resize(Nv);
  out.p.resize(Np);
  for (std::size_t k = 0; k < Nu; ++k) out.u[k] = sol[static_cast<Eigen::Index>(k)];
  for (std::size_t k = 0; k < Nv; ++k) out.v[k] = sol[static_cast<Eigen::Index>(ov + k)];
  for (std::size_t k = 0; k < Np; ++k) out.p[k] = sol[static_cast<Eigen::Index>(op + k)];
  return out;
}

/// The channel this example runs: Poiseuille inflow, no-slip bottom,
/// zero-curvature outflow, and a top wall whose normal velocity is the
/// beam's -- the transpiration condition, set fresh every call to `step()`.
class StokesSolver {
 public:
  explicit StokesSolver(ChannelParams p)
      : p_(p),
        pin_{0, p.ny - 1, 0.0} {
    const std::size_t Nu = (p_.nx + 1) * p_.ny;
    const std::size_t Nv = p_.nx * (p_.ny + 1);
    const std::size_t Np = p_.nx * p_.ny;
    u_.assign(Nu, 0.0);
    v_.assign(Nv, 0.0);
    p_field_.assign(Np, 0.0);
    traction_mid_.assign(p_.nx, 0.0);
  }

  [[nodiscard]] const ChannelParams& params() const { return p_; }
  [[nodiscard]] double dx() const { return p_.L / static_cast<double>(p_.nx); }
  [[nodiscard]] double dy() const { return p_.H / static_cast<double>(p_.ny); }

  /// x-coordinate of wall-face column `i` (i = 0..nx-1): the fluid wall's
  /// own FV cell centroids, the SAME convention FUSE's "generated:line:n"
  /// mesh uses for cell centres ((i+0.5)/n on [0,1]) once scaled by L.
  [[nodiscard]] double wall_x(std::size_t i) const {
    return (static_cast<double>(i) + 0.5) * dx();
  }

  /// One physical CN step: `top_wall_v_new` is the beam's normal velocity
  /// already interpolated (through H) onto this solver's own nx wall-face
  /// x-positions, AT THE NEW TIME t1 = t0 + dt. `top_wall_v_old` is the
  /// SAME quantity at t0 (the coupling cycle's start -- CONSTANT across every
  /// sweep of an implicit coupling group, unlike `top_wall_v_new`, which changes
  /// sweep to sweep as the beam's own guess converges).
  ///
  /// TIME-LEVEL CONSISTENCY (found and fixed by isolating a many-period
  /// energy-growth defect in the coupled FSI channel -- see
  /// docs/user/fsi-channel.md's "Partitioned time-level consistency"
  /// section and test/verification/fsi_transient_test.cpp's energy
  /// diagnostics): an EARLIER version of this method took only ONE
  /// velocity array and applied it identically at both cn_step's EXPLICIT
  /// (t0) and IMPLICIT (t1) boundary evaluations -- i.e. treated the wall
  /// velocity as jumping instantaneously to its new value at the START of
  /// every step, a backward-Euler-consistent boundary treatment glued
  /// onto an otherwise second-order-in-time (trapezoidal) interior
  /// scheme. That mismatch is exactly the kind of partitioned-coupling
  /// defect that injects spurious energy over many periods even though
  /// EACH side (beam alone, fluid alone) is unconditionally stable in
  /// isolation. Both time levels are now threaded through explicitly so
  /// the wall's CONTRIBUTION to CN's trapezoidal-averaged viscous term
  /// matches the SAME v_old/v_new convention the interior stencil uses.
  void step(double dt, double t0, double t1,
            const std::vector<double>& top_wall_v_old,
            const std::vector<double>& top_wall_v_new,
            const std::function<double(double)>& inflow_u) {
    if (top_wall_v_old.size() != p_.nx || top_wall_v_new.size() != p_.nx) {
      throw std::invalid_argument("StokesSolver::step: top_wall_v_old/new "
                                   "size must equal nx");
    }
    ChannelBCs bcs;
    bcs.bottom.u = [](double, double) { return 0.0; };
    bcs.bottom.v = [](double, double) { return 0.0; };
    bcs.top.u = [](double, double) { return 0.0; };  // no tangential slip
    const double t_mid = 0.5 * (t0 + t1);
    bcs.top.v = [this, &top_wall_v_old, &top_wall_v_new, t_mid](double x, double t) {
      // Invert wall_x(i) = (i + 0.5) * dx exactly (up to rounding): every
      // call site passes x = x_half for some real column, never an
      // arbitrary point, so a simple round-and-clamp is exact.
      const long long ii = std::lround(x / dx() - 0.5);
      const std::size_t i = static_cast<std::size_t>(
          std::clamp<long long>(ii, 0, static_cast<long long>(p_.nx) - 1));
      // cn_step calls a boundary functor with EXACTLY t0 (explicit side)
      // or EXACTLY t1 (implicit side); comparing against the midpoint
      // distinguishes the two exactly, with no floating-point ambiguity
      // for any dt > 0.
      return (t < t_mid) ? top_wall_v_old[i] : top_wall_v_new[i];
    };
    bcs.left.u = [&inflow_u](double y, double) { return inflow_u(y); };
    bcs.left.v = [](double, double) { return 0.0; };
    bcs.right.outflow = true;

    // KNOWN_ISSUES.md D9, root cause and fix (this round). `cn_step`'s
    // trapezoidal VISCOUS term already averages t0 and t1; incompressibility
    // (div u = 0, enforced at t1 by the continuity rows) determines only
    // that ONE combination of pressure and velocity that makes the t1
    // momentum balance hold -- which, for a fully-implicit pressure
    // gradient (p0 = nullptr, this round's fix), the manufactured-solution
    // fixture already showed (docs/user/fsi-channel.md, "Partitioned
    // time-level consistency: the force side") converges to the ANALYTIC
    // p(t_{n+1/2}), not p(t1): the solved `p_field_` genuinely IS the
    // trapezoidal scheme's own Lagrange multiplier at the step's MIDPOINT,
    // not a t1 quantity mislabelled or a t0 quantity carried as extra state.
    //
    // An earlier round's fix instead made the pressure GRADIENT trapezoidal
    // (half implicit p1, half explicit p0, with p0 = the PREVIOUS step's
    // p_field_ carried forward as literal state) specifically to make the
    // solved value t1-accurate. That traded one defect for another: p0 as
    // carried state is a FREE parasitic component -- incompressibility only
    // ever pins the average p_mid = (p0+p1)/2, so p1 = 2*p_mid - p0 has a
    // genuine |lambda|=1 mode in p0 alone (confirmed directly, this round:
    // FsiEnergyDiag.FluidAloneStepMapSpectralRadiusIsAtMostOne measures
    // rho=1.0000000000 for a mode that is ~100% pressure with u=v=0
    // identically -- the fluid alone never feels it, since only p_mid
    // enters its own momentum balance). Reverting to a fully-implicit
    // pressure gradient removes that parasitic state entirely: there is
    // no p0 to carry, only the genuinely mid-step p_field_ this call
    // produces directly -- confirmed directly by the fluid-alone spectral
    // radius moving off exactly 1 (to 0.9946) once p0 is gone.
    //
    // IMPORTANT, measured directly (this round, NOT assumed): removing
    // this parasitic mode does NOT fix D9's headline coupled-system
    // growth. The wall traction delivered to the beam used to carry p1
    // (the "t1-accurate" value, including the parasitic component), and
    // the natural hypothesis was that THIS closed the feedback loop
    // driving the coupled step map's rho~1.11 (KNOWN_ISSUES.md D9's own
    // numbers). Measured before/after this exact change:
    // FsiEnergyDiag.CoupledStepMapRevealsTheGrowthMechanism's own rho
    // moves from 1.115467 to 1.115466 -- unchanged to 4 significant
    // figures -- and an ablation that zeroes the pressure term in the
    // delivered traction gives the identical rho=1.000512 before and
    // after this change. The parasitic p0 mode is real and is correctly
    // removed here, but it is demonstrably NOT the (or not the only)
    // mechanism sustaining D9's coupled energy growth; that mechanism is
    // still open (KNOWN_ISSUES.md D9's "round 2" section).
    FlowState next =
        cn_step(p_, bcs, SourceTerms{}, dt, t0, t1, u_, v_, pin_, nullptr, nullptr,
               &lu_cache_);

    // GLOBAL MASS CONSERVATION AT THE OUTFLOW -- kept, independent of the
    // pressure-labelling fix above. This is a correct statement of
    // discrete continuity at the outlet REGARDLESS of what time level the
    // pressure represents: `cn_step`'s pressure-outlet condition (this
    // file's own header, "PRESSURE GAUGE") REPLACES the outflow column's
    // own continuity equation with `p=0` at every row -- a deliberate,
    // documented trade-off to kill the discrete operator's second null
    // direction (a uniform u-shift paired with a linear pressure ramp;
    // docs/user/fsi-channel.md's "Outflow" section). Its documented cost
    // ("the exact discrete Poiseuille state is only an APPROXIMATE ...
    // fixed point") is that with that column's continuity unenforced, the
    // OUTPUT velocity field is free to violate global mass balance by
    // whatever the solve happens to produce there. Global continuity for
    // an incompressible channel with a moving (transpiring) top wall
    // reads Q_out(t) = Q_in(t) + Q_wall(t), Q_wall(t) = integral of the
    // wall's own normal velocity over the channel length -- exactly the
    // "transpiration volume source" this task named. `cn_step`'s outflow
    // treatment leaves `Q_out` floating (a zero-curvature extrapolation)
    // with nothing tying it to `Q_in + Q_wall`; this correction ties it,
    // by a single uniform shift of the outflow column's own `u` (the ONE
    // quantity the pressure-outlet condition left unconstrained there)
    // chosen so the corrected `Q_out` matches exactly. A uniform shift
    // changes nothing else -- not any OTHER column's `u`, nor `v`, nor
    // `p` -- so it cannot reopen the null direction the pressure-outlet
    // fix removed (that mode needs the WHOLE field perturbed together).
    // At the exact discrete Poiseuille state with a resting beam
    // (Q_wall=0), `Q_in` and the uncorrected `Q_out` already agree to
    // round-off (confirmed by
    // `FsiCoupled.FluidReconstructedPoiseuilleIsAFixedPointOfCnStep`, which
    // calls `cn_step` directly and is therefore untouched by this fix --
    // it lives entirely here, never in `cn_step` itself, so the MMS
    // spatial/temporal-order suite is unaffected too), so the correction
    // is a genuine no-op there and a real, physically-motivated one only
    // when the wall actually moves. An earlier round measured this
    // correction alone changes the coupled step map's spectral radius by
    // well under 1% -- consistent with it correcting a real but
    // SEPARATE local-continuity defect, not the parasitic-pressure-state
    // mechanism the fix above addresses.
    {
      using detail::idx_u;
      const double dyc = dy();
      double q_in = 0.0, q_out = 0.0;
      for (std::size_t j = 0; j < p_.ny; ++j) {
        q_in += next.u[idx_u(p_, 0, j)] * dyc;
        q_out += next.u[idx_u(p_, p_.nx, j)] * dyc;
      }
      double q_wall = 0.0;
      const double dxc = dx();
      for (std::size_t i = 0; i < p_.nx; ++i) q_wall += top_wall_v_new[i] * dxc;
      const double correction = (q_in + q_wall - q_out) / p_.H;
      for (std::size_t j = 0; j < p_.ny; ++j) {
        next.u[idx_u(p_, p_.nx, j)] += correction;
      }
    }

    // Mid-step traction, computed and cached HERE (not derived on demand
    // by wall_traction()): the pressure half is already mid-step (see
    // this method's own comment above -- `next.p` IS p_mid, no averaging
    // needed), but the VISCOUS half of the traction, `2*mu*dv/dy`, still
    // needs the SAME t0/t1 trapezoidal averaging the momentum equation's
    // own viscous term uses -- `dv/dy` from a one-sided difference is a
    // SPATIAL construct evaluated at a single time level, so computing it
    // once from `next.v` alone would stay t1-only regardless of how the
    // pressure is handled. `v_` here is still the t0 array (assigned to
    // `v_` only below); `next.v` is t1.
    std::vector<double> traction(p_.nx, 0.0);
    {
      using detail::idx_p;
      using detail::idx_v;
      const double h = dy();
      for (std::size_t i = 0; i < p_.nx; ++i) {
        const double p_last = next.p[idx_p(p_, i, p_.ny - 1)];
        const double p_prev = p_.ny >= 2 ? next.p[idx_p(p_, i, p_.ny - 2)] : p_last;
        const double p_wall_mid = 1.5 * p_last - 0.5 * p_prev;

        const auto dvdy_at = [&](const std::vector<double>& v) {
          const double v_wall = v[idx_v(p_, i, p_.ny)];
          const double v_m1 = v[idx_v(p_, i, p_.ny - 1)];
          const double v_m2 = p_.ny >= 2 ? v[idx_v(p_, i, p_.ny - 2)] : v_m1;
          return (3.0 * v_wall - 4.0 * v_m1 + v_m2) / (2.0 * h);
        };
        const double dvdy_mid = 0.5 * (dvdy_at(v_) + dvdy_at(next.v));
        // The force the FLUID exerts ON THE WALL, per unit area, in the
        // fluid's +y (the beam's +w, since the transpiration condition sets
        // v_wall = +wdot). Cauchy's sigma.n with the fluid's outward normal
        // n = +e_y is the traction the wall exerts on the fluid; the force
        // on the wall is its negative: +p - 2 mu dv/dy. A pressurised
        // channel pushes its wall OUTWARD. This carried the opposite sign
        // (-p + 2 mu dv/dy, i.e. the force on the fluid) until 2026-09-25,
        // which handed the beam a NEGATIVE added mass: KNOWN_ISSUES.md D9's
        // real eigenvalue 1.115466 with a pressure-and-first-bending-mode
        // eigenvector, insensitive to every boundary detail four rounds of
        // energy budgets varied, because every check compared the delivered
        // vector with itself. With the sign right the coupled step map's
        // spectral radius is 1.000000 (the Newmark acceleration passenger at
        // -1), measured in fsi_energy_diag_test.cpp before this line changed.
        traction[i] = p_wall_mid - 2.0 * p_.mu * dvdy_mid;
      }
    }

    u_ = next.u;
    v_ = next.v;
    p_field_ = next.p;
    traction_mid_ = std::move(traction);
  }

  /// Mid-step wall traction, `-p_{n+1/2} + 2 mu (dv/dy)_{n+1/2}`, one value
  /// per wall-face column (size nx) -- see `step()`'s own comment for the
  /// KNOWN_ISSUES.md D9 derivation. Cached in `step()` (needs the t0 `v_`
  /// array, no longer available once `step()` returns), not recomputed
  /// here; a plain accessor.
  [[nodiscard]] const std::vector<double>& wall_traction() const {
    return traction_mid_;
  }

  /// Recomputes and caches `traction_mid_` from the CURRENT `p_field_`/`v_`
  /// alone, evaluating the mid-step formula with the SAME array used for
  /// both the "t0" and "t1" sample (so the viscous half's averaging is a
  /// no-op: `0.5*(dvdy(v_)+dvdy(v_)) = dvdy(v_)`). For a caller that just
  /// bypassed `step()` via a direct `set_state(...)` (a test-only pattern
  /// -- see `FluidComponent::refresh_force_from_current_state()`'s own
  /// comment): `step()` is the only place `traction_mid_` is normally
  /// updated, and it needs the t0 `v_` array `set_state()` has no access
  /// to. Exact for the one case this is actually used for, a genuine
  /// STEADY state (t0=t1 trivially, so evaluating "as if" they were the
  /// same array is not an approximation there at all).
  void recompute_traction_from_current_state() {
    using detail::idx_p;
    using detail::idx_v;
    const double h = dy();
    std::vector<double> traction(p_.nx, 0.0);
    for (std::size_t i = 0; i < p_.nx; ++i) {
      const double p_last = p_field_[idx_p(p_, i, p_.ny - 1)];
      const double p_prev = p_.ny >= 2 ? p_field_[idx_p(p_, i, p_.ny - 2)] : p_last;
      const double p_wall = 1.5 * p_last - 0.5 * p_prev;
      const double v_wall = v_[idx_v(p_, i, p_.ny)];
      const double v_m1 = v_[idx_v(p_, i, p_.ny - 1)];
      const double v_m2 = p_.ny >= 2 ? v_[idx_v(p_, i, p_.ny - 2)] : v_m1;
      const double dvdy = (3.0 * v_wall - 4.0 * v_m1 + v_m2) / (2.0 * h);
      traction[i] = p_wall - 2.0 * p_.mu * dvdy;  // force ON the wall; see step()
    }
    traction_mid_ = std::move(traction);
  }

  /// Discrete kinetic energy [J, per unit out-of-plane depth], consistent
  /// with this scheme's own mass convention: `cn_step`'s momentum rows
  /// carry `rho/dt`, not `rho*dx*dy/dt` (a pointwise, collocation
  /// convention, exactly as `BeamSolver::kinetic_energy()`'s own header
  /// explains for the beam's `wdot`). The matching quadrature is therefore
  /// a uniform `dx*dy` cell weight at every `u`/`v` degree of freedom,
  /// boundary ones included (they carry real values -- Dirichlet or
  /// floating -- not placeholders). Used by the transient-energy
  /// diagnostics in test/verification/fsi_transient_test.cpp and
  /// fsi_energy_diag_test.cpp; not on any solve path.
  [[nodiscard]] double kinetic_energy() const {
    const double cell = dx() * dy();
    double sum = 0.0;
    for (const double uv : u_) sum += uv * uv;
    for (const double vv : v_) sum += vv * vv;
    return 0.5 * p_.rho * cell * sum;
  }

  [[nodiscard]] FlowState save() const {
    return FlowState{u_, v_, p_field_, traction_mid_};
  }
  void restore(const FlowState& s) {
    u_ = s.u;
    v_ = s.v;
    p_field_ = s.p;
    traction_mid_ =
        s.traction_mid.empty() ? std::vector<double>(p_.nx, 0.0) : s.traction_mid;
  }
  void set_state(FlowState s) {
    u_ = std::move(s.u);
    v_ = std::move(s.v);
    p_field_ = std::move(s.p);
    traction_mid_ = s.traction_mid.empty() ? std::vector<double>(p_.nx, 0.0)
                                           : std::move(s.traction_mid);
  }

  [[nodiscard]] const std::vector<double>& u() const { return u_; }
  [[nodiscard]] const std::vector<double>& v() const { return v_; }
  [[nodiscard]] const std::vector<double>& p() const { return p_field_; }

 private:
  ChannelParams p_;
  PressurePin   pin_;
  std::vector<double> u_, v_, p_field_;
  std::vector<double> traction_mid_;  // see wall_traction()'s own comment

  // Cross-step cache for cn_step's saddle-point factorisation -- see
  // CnStepLuCache's own comment for exactly what this is and why it's safe
  // here: StokesSolver has no setters for p_ or pin_ (grid, mu, rho, pin are
  // fixed for the object's entire lifetime), and step() rebuilds `bcs` with
  // a hardcoded shape every call, so `dt` is the ONLY input to cn_step's
  // matrix that can vary call to call on a single instance.
  CnStepLuCache lu_cache_;
};

}  // namespace fsi
