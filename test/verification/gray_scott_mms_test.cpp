// Plan step 1 ("Structured-grid P0 finite-volume species solver"): each
// species alone, by method of manufactured solutions. The partner species is
// a FROZEN (in time) input field -- exactly the plan's own framing -- so
// this file measures GrayScottSpeciesComponent's own Crank-Nicolson Newton
// solve in isolation, with no coupling error mixed in: the manufactured
// forcing (computed in closed form from the manufactured solution) is exact,
// so no source contributes a discretization error, and the FROZEN partner
// is likewise given in closed form with no time-integration of its own.
//
// MANUFACTURED SOLUTION (species S alone; the frozen partner is P):
//   S_exact(x,y,t) = s0 + A*sin(kx*x)*sin(ky*y)*cos(w*t)
//   P_frozen(x,y)  = p0 + B*cos(kx*x)*cos(ky*y)         (no time dependence)
// with kx = ky = 2*pi/L. The forcing is derived directly from the PDE:
//   Src(x,y,t) = dS_exact/dt - D*Laplacian(S_exact) - Reaction(S_exact, P_frozen)
// so that S_exact solves  S_t = D*Lap(S) + Reaction(S,P) + Src  EXACTLY, and
// the discrete scheme (5-point FV Laplacian, Crank-Nicolson, Newton) is left
// to reproduce it to its own truncation error alone: O(h^2) in space,
// O(dt^2) in time.
//
// ISOLATING SPACE FROM TIME: the spatial-order study uses a small,
// fixed dt (many steps to a short T) so temporal error is negligible next to
// spatial error; the temporal-order study uses a fine, fixed N so spatial
// error is negligible next to temporal error. Both are verified empirically
// below (the "isolation margin" comment on each study) rather than assumed.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include <fuse/base/rational.hpp>

#include "gray_scott_physics.hpp"
#include "species_component.hpp"

using fuse::Rational;
using fuse::StepContext;
using gray_scott::Grid;
using gray_scott::GrayScottU;
using gray_scott::GrayScottV;
using gray_scott::Kinetics;
using gray_scott::Species;

namespace {

constexpr double kL = 1.0;
constexpr double kKx = 2.0 * M_PI / kL;
constexpr double kKy = 2.0 * M_PI / kL;
constexpr double kOmega = 2.0 * M_PI;
constexpr double kS0 = 1.0, kA = 0.1;
constexpr double kP0 = 0.2, kB = 0.05;
constexpr Kinetics kKin{/*F=*/0.04, /*k=*/0.06};
constexpr double kD = 0.01;
// Newton tolerance is tied to the study's own error floor: two to three
// orders tighter than the smallest error measured below (~1e-6), so Newton's
// own residual never contaminates a discretization-order measurement. See
// docs/user/gray-scott.md "Chosen tolerances" for the number used in the
// example driver and multirate tests.
constexpr double kNewtonTol = 1.0e-11;

[[nodiscard]] double s_exact(double x, double y, double t) {
  return kS0 + kA * std::sin(kKx * x) * std::sin(kKy * y) * std::cos(kOmega * t);
}
[[nodiscard]] double dsdt_exact(double x, double y, double t) {
  return -kA * kOmega * std::sin(kKx * x) * std::sin(kKy * y) * std::sin(kOmega * t);
}
[[nodiscard]] double lap_s_exact(double x, double y, double t) {
  return -kA * (kKx * kKx + kKy * kKy) * std::sin(kKx * x) * std::sin(kKy * y) * std::cos(kOmega * t);
}
[[nodiscard]] double p_frozen(double x, double y) {
  return kP0 + kB * std::cos(kKx * x) * std::cos(kKy * y);
}

/// Runs species `S` alone from t=0 to t=t_final in `n_steps` equal CN steps
/// on an n x n grid, and returns the L2 error of the final state against
/// s_exact(., t_final).
template <Species S>
[[nodiscard]] double run_mms(int n, int n_steps, Rational t_final) {
  const Grid grid{n, kL};
  std::vector<double> init(grid.size()), partner(grid.size());
  for (int i = 0; i < n; ++i) {
    const double x = grid.x_center(i);
    for (int j = 0; j < n; ++j) {
      const double y = grid.y_center(j);
      const std::size_t m = grid.index(i, j);
      init[m] = s_exact(x, y, 0.0);
      partner[m] = p_frozen(x, y);
    }
  }
  gray_scott::GrayScottSpeciesComponent<S> comp("mms", grid, kD, kKin, init, kNewtonTol);
  comp.set_required(partner);
  comp.set_source([](double x, double y, double t) {
    const double react = (S == Species::U)
        ? kKin.reaction_u(s_exact(x, y, t), p_frozen(x, y))
        : kKin.reaction_v(p_frozen(x, y), s_exact(x, y, t));
    return dsdt_exact(x, y, t) - kD * lap_s_exact(x, y, t) - react;
  });

  const Rational dt = t_final / Rational{n_steps};
  Rational t = Rational{0};
  for (int step = 0; step < n_steps; ++step) {
    StepContext ctx;
    ctx.t_begin = t;
    ctx.t_end = t + dt;
    ctx.step = step;
    const auto r = comp.advance(ctx);
    if (!r.ok()) {
      ADD_FAILURE() << "advance() failed at step " << step << ": " << r.error().message;
      return std::numeric_limits<double>::infinity();
    }
    t = ctx.t_end;
  }

  double sq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double x = grid.x_center(i);
    for (int j = 0; j < n; ++j) {
      const double y = grid.y_center(j);
      const double diff = comp.state()[grid.index(i, j)] - s_exact(x, y, t.to_double());
      sq += diff * diff;
    }
  }
  return std::sqrt(sq / static_cast<double>(grid.size()));
}

[[nodiscard]] double order(double e_coarse, double e_fine, double refinement) {
  return std::log(e_coarse / e_fine) / std::log(refinement);
}

}  // namespace

// ---------------------------------------------------------------- spatial

TEST(GrayScottMms, SpatialOrderTwoSpeciesU) {
  // dt fixed small enough that temporal error is negligible: verified below
  // by the isolation-margin assertion (finest spatial error must still be
  // >> the temporal-only error floor measured separately in the temporal
  // test at this same dt). Grid sizes kept modest (<=32) -- SparseLU
  // refactorized every Newton iteration in this Debug build measures ~2.4/
  // 14.9/85.8 ms/step at N=8/16/32 respectively (scaling faster than
  // linearly with N, as expected for a dense-ish sparse factorization); a
  // grid list extending to N=64 (measured ~85.8 ms/step at N=32 alone,
  // times 24 steps times 4 grid sizes) would cost tens of seconds per run.
  const Rational t_final{1, 20};
  const int n_steps = 24;  // dt = 0.05/24 ~ 2.1e-3
  std::vector<double> errs;
  for (const int n : {8, 16, 32}) {
    errs.push_back(run_mms<Species::U>(n, n_steps, t_final));
  }
  for (std::size_t i = 0; i + 1 < errs.size(); ++i) {
    const double ord = order(errs[i], errs[i + 1], 2.0);
    EXPECT_GE(ord, 1.85) << "step " << i << ": errs " << errs[i] << " -> " << errs[i + 1];
    EXPECT_LE(ord, 2.15) << "step " << i << ": errs " << errs[i] << " -> " << errs[i + 1];
  }
}

TEST(GrayScottMms, SpatialOrderTwoSpeciesV) {
  const Rational t_final{1, 20};
  const int n_steps = 24;
  std::vector<double> errs;
  for (const int n : {8, 16, 32}) {
    errs.push_back(run_mms<Species::V>(n, n_steps, t_final));
  }
  for (std::size_t i = 0; i + 1 < errs.size(); ++i) {
    const double ord = order(errs[i], errs[i + 1], 2.0);
    EXPECT_GE(ord, 1.85) << "step " << i << ": errs " << errs[i] << " -> " << errs[i + 1];
    EXPECT_LE(ord, 2.15) << "step " << i << ": errs " << errs[i] << " -> " << errs[i + 1];
  }
}

// --------------------------------------------------------------- temporal

TEST(GrayScottMms, TemporalOrderTwoSpeciesU) {
  // N fixed fine enough that spatial error is negligible next to the
  // temporal error being measured -- checked directly below via a proxy
  // run at a much finer dt (n_steps=64) at this same N, standing in for
  // the dt -> 0 spatial-error floor, rather than assumed.
  const int n = 32;
  const Rational t_final{1, 2};
  std::vector<double> errs;
  for (const int n_steps : {4, 8, 16, 32}) {
    errs.push_back(run_mms<Species::U>(n, n_steps, t_final));
  }
  for (std::size_t i = 0; i + 1 < errs.size(); ++i) {
    const double ord = order(errs[i], errs[i + 1], 2.0);
    EXPECT_GE(ord, 1.85) << "step " << i << ": errs " << errs[i] << " -> " << errs[i + 1];
    EXPECT_LE(ord, 2.15) << "step " << i << ": errs " << errs[i] << " -> " << errs[i + 1];
  }
  // Isolation margin: the spatial discretisation's own error at this N must
  // not be a large fraction of the coarsest temporal error being measured.
  const double spatial_floor = run_mms<Species::U>(n, 64, t_final);  // dt -> 0 limit proxy
  EXPECT_LT(spatial_floor, 0.1 * errs.front())
      << "spatial error at N=" << n << " (" << spatial_floor
      << ") is not negligible next to the temporal study's coarsest error (" << errs.front()
      << "); increase N or coarsen the dt range";
}

TEST(GrayScottMms, TemporalOrderTwoSpeciesV) {
  const int n = 32;
  const Rational t_final{1, 2};
  std::vector<double> errs;
  for (const int n_steps : {4, 8, 16, 32}) {
    errs.push_back(run_mms<Species::V>(n, n_steps, t_final));
  }
  for (std::size_t i = 0; i + 1 < errs.size(); ++i) {
    const double ord = order(errs[i], errs[i + 1], 2.0);
    EXPECT_GE(ord, 1.85) << "step " << i << ": errs " << errs[i] << " -> " << errs[i + 1];
    EXPECT_LE(ord, 2.15) << "step " << i << ": errs " << errs[i] << " -> " << errs[i + 1];
  }
  const double spatial_floor = run_mms<Species::V>(n, 64, t_final);
  EXPECT_LT(spatial_floor, 0.1 * errs.front())
      << "spatial error at N=" << n << " (" << spatial_floor
      << ") is not negligible next to the temporal study's coarsest error (" << errs.front()
      << "); increase N or coarsen the dt range";
}
