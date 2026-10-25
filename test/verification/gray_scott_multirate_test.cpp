// Plan step 2 ("Multirate implicit coupling of U and V"): U and V, on the
// SAME structured grid, coupled through fuse::WindowCoupler at independent
// time steps -- both ways, implicitly (iterated to convergence within each
// window), covering:
//   (a) commensurate ratios dt_v = dt_u/m, m = 1, 2, 4;
//   (b) the incommensurate 7:13 ratio, window defined by the 13-part member;
//   (c) both DtPolicy values for the member that can actually straddle.
//
// U IS ALWAYS THE WINDOW-DEFINER (its own interval equals cfg.window_length),
// exactly the role "A" plays in test/runtime/window_coupler_test.cpp's
// run_two_member: a window-definer's own step never exceeds the window and
// is therefore structurally incapable of straddling, whatever DtPolicy it is
// given. V is the (possibly) finer member; only V's DtPolicy is varied here,
// which is what "the member that straddles" means for every configuration
// below -- U never straddles, so testing U at both policies would not
// exercise anything the straddle logic touches.
//
// THE ENVELOPE (E-RT-WINDOW-STRADDLE / -LONGSTEP).
// WindowCoupler::create rejects a Fixed member whose interval exceeds the
// window (E-RT-WINDOW-LONGSTEP) and rejects two members that could both
// straddle the same window end (E-RT-WINDOW-STRADDLE). Every configuration
// below keeps U as the window itself (so U's "interval exceeds the window"
// is never true) and gives V an interval that is at most the window's
// (never true for V either) with only V ever Fixed -- i.e. every case here
// sits inside the envelope by construction. `RespectsTheStraddleEnvelope`
// at the bottom confirms a configuration that leaves it (both members
// Fixed on a window neither divides) is rejected with the documented code,
// so "respects the envelope" is demonstrated, not merely asserted.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <fuse/runtime/window_coupler.hpp>

#include "gray_scott_physics.hpp"
#include "species_component.hpp"

using fuse::AccelConfig;
using fuse::AccelKind;
using fuse::ConvergenceMeasure;
using fuse::DtPolicy;
using fuse::MeasureKind;
using fuse::Rational;
using fuse::WindowChannel;
using fuse::WindowCoupler;
using fuse::WindowCouplerConfig;
using fuse::WindowMember;
using gray_scott::Grid;
using gray_scott::GrayScottU;
using gray_scott::GrayScottV;
using gray_scott::Kinetics;

namespace {

constexpr int      kN = 12;
constexpr double   kL = 1.0;
constexpr double   kDu = 0.01, kDv = 0.005;
constexpr Kinetics kKin{/*F=*/0.04, /*k=*/0.06};
constexpr double   kNewtonTol = 1.0e-11;
constexpr double   kCouplingTol = 1.0e-9;

[[nodiscard]] std::vector<double> initial_u(const Grid& g) {
  std::vector<double> u(g.size());
  for (int i = 0; i < g.n; ++i) {
    const double x = g.x_center(i);
    for (int j = 0; j < g.n; ++j) {
      const double y = g.y_center(j);
      u[g.index(i, j)] = 1.0 + 0.1 * std::sin(2.0 * M_PI * x) * std::cos(2.0 * M_PI * y);
    }
  }
  return u;
}
[[nodiscard]] std::vector<double> initial_v(const Grid& g) {
  std::vector<double> v(g.size());
  for (int i = 0; i < g.n; ++i) {
    const double x = g.x_center(i);
    for (int j = 0; j < g.n; ++j) {
      const double y = g.y_center(j);
      v[g.index(i, j)] = 0.1 + 0.05 * std::cos(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
    }
  }
  return v;
}

struct MultirateResult {
  bool                 ok = false;
  std::string          error;
  std::vector<double>  u, v;
  std::vector<int>     iterations;
};

/// U is the window-definer (interval == window_length, so it never
/// straddles regardless of `pol_u`); V has `interval_v` <= window_length.
[[nodiscard]] MultirateResult run_multirate(Rational window, Rational interval_v,
                                            int n_windows, DtPolicy pol_u, DtPolicy pol_v,
                                            AccelKind accel = AccelKind::Aitken) {
  MultirateResult out;
  const Grid grid{kN, kL};
  auto comp_u = std::make_unique<GrayScottU>("U", grid, kDu, kKin, initial_u(grid), kNewtonTol);
  auto comp_v = std::make_unique<GrayScottV>("V", grid, kDv, kKin, initial_v(grid), kNewtonTol);

  WindowMember mu;
  mu.component = comp_u.get();
  mu.name = "U";
  mu.interval = window;
  mu.policy = pol_u;
  mu.provided = comp_u->provided_ptr();
  mu.required = comp_u->required_ptr();
  mu.length = grid.size();

  WindowMember mv;
  mv.component = comp_v.get();
  mv.name = "V";
  mv.interval = interval_v;
  mv.policy = pol_v;
  mv.provided = comp_v->provided_ptr();
  mv.required = comp_v->required_ptr();
  mv.length = grid.size();

  WindowCouplerConfig cfg;
  cfg.window_length = window;
  cfg.max_iterations = 200;
  cfg.acceleration = AccelConfig{.kind = accel};
  // A convergence measure names
  // a field TRANSFER (a WindowChannel, by its own `name`), not the whole
  // coupled system -- see gray_scott_reference_test.cpp's identical
  // comment. Split into the two real transfers, same kind/tolerance as
  // the aggregate measure this replaces; not byte-identical to before by
  // the user's own choice.
  ConvergenceMeasure meas_u;
  meas_u.kind = MeasureKind::Relative;
  meas_u.tolerance = kCouplingTol;
  meas_u.field = "u_out";
  cfg.convergence.add(meas_u);
  ConvergenceMeasure meas_v;
  meas_v.kind = MeasureKind::Relative;
  meas_v.tolerance = kCouplingTol;
  meas_v.field = "v_out";
  cfg.convergence.add(meas_v);
  cfg.waveform_capacity = 512;

  auto created = WindowCoupler::create(
      {mu, mv},
      {{.provider = 0, .consumer = 1, .name = "u_out"},
       {.provider = 1, .consumer = 0, .name = "v_out"}},
      cfg);
  if (!created.ok()) {
    out.error = created.error().code + ": " + created.error().message;
    return out;
  }
  auto coupler = std::move(created.value());
  for (int w = 0; w < n_windows; ++w) {
    const auto r = coupler->run_window();
    if (!r.ok()) {
      out.error = r.error().code + ": " + r.error().message;
      return out;
    }
    out.iterations.push_back(coupler->stats().iterations);
  }
  out.u = comp_u->state();
  out.v = comp_v->state();
  out.ok = true;
  return out;
}

[[nodiscard]] bool all_finite_and_bounded(const std::vector<double>& x, double lo, double hi) {
  for (double v : x) {
    if (!std::isfinite(v) || v < lo || v > hi) return false;
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------- (a)

TEST(GrayScottMultirate, CommensurateRatiosConverge) {
  const Rational window{1, 20};
  for (const int m : {1, 2, 4}) {
    const MultirateResult r =
        run_multirate(window, window / Rational{m}, 5, DtPolicy::Adjustable, DtPolicy::Adjustable);
    ASSERT_TRUE(r.ok) << "m=" << m << ": " << r.error;
    EXPECT_TRUE(all_finite_and_bounded(r.u, -0.5, 2.0)) << "m=" << m;
    EXPECT_TRUE(all_finite_and_bounded(r.v, -0.5, 2.0)) << "m=" << m;
    for (int it : r.iterations) EXPECT_GT(it, 0) << "m=" << m;
  }
}

TEST(GrayScottMultirate, CommensurateRatiosFixedVDivideNeverStraddles) {
  // V's interval divides the window exactly at m=2,4, so DtPolicy::Fixed
  // never triggers a straddle for V either (the window coupler's own
  // "AFixedStepThatDividesTheWindowNeverStraddlesAndIsAccepted" case) --
  // exercised here with the real Gray-Scott components, not the linear ODE.
  const Rational window{1, 20};
  for (const int m : {2, 4}) {
    const MultirateResult r =
        run_multirate(window, window / Rational{m}, 5, DtPolicy::Adjustable, DtPolicy::Fixed);
    ASSERT_TRUE(r.ok) << "m=" << m << ": " << r.error;
  }
}

// ---------------------------------------------------------------- (b)/(c)

TEST(GrayScottMultirate, IncommensurateSevenThirteenBothVPolicies) {
  // Window = 13 parts (U, the window-definer); V = 7 parts, which does not
  // divide 13, so V is the one member that can straddle. Both of its
  // DtPolicy values are exercised, per the task's item (c).
  const Rational unit{1, 2000};  // 13*unit = 0.0065, small enough to keep Newton in its basin
  const Rational window = unit * Rational{13};
  const Rational v_interval = unit * Rational{7};
  for (const DtPolicy pol_v : {DtPolicy::Fixed, DtPolicy::Adjustable}) {
    const MultirateResult r = run_multirate(window, v_interval, 6, DtPolicy::Adjustable, pol_v);
    ASSERT_TRUE(r.ok) << (pol_v == DtPolicy::Fixed ? "Fixed" : "Adjustable") << ": " << r.error;
    EXPECT_TRUE(all_finite_and_bounded(r.u, -0.5, 2.0));
    EXPECT_TRUE(all_finite_and_bounded(r.v, -0.5, 2.0));
    for (int it : r.iterations) EXPECT_GT(it, 0);
  }
}

// ---------------------------------------------------------------- (f)

TEST(GrayScottMultirate, IterationCountsAcrossAccelerators) {
  const Rational unit{1, 2000};
  const Rational window = unit * Rational{13};
  const Rational v_interval = unit * Rational{7};
  int total_none = 0, total_aitken = 0, total_iqn = 0;
  for (const auto [accel, total] :
       std::initializer_list<std::pair<AccelKind, int*>>{
           {AccelKind::None, &total_none},
           {AccelKind::Aitken, &total_aitken},
           {AccelKind::IQN_ILS, &total_iqn}}) {
    const MultirateResult r =
        run_multirate(window, v_interval, 4, DtPolicy::Fixed, DtPolicy::Adjustable, accel);
    ASSERT_TRUE(r.ok) << r.error;
    for (int it : r.iterations) *total += it;
  }
  // MEASURED: 9/13/12 sweeps total (None/Aitken/IQN-ILS) over 4 windows.
  // (It was 14/17/16 while WindowCoupler wrote its sweep vector
  // transposed, then 9/13/12 once that was fixed. An intermediate Unit
  // C3 attempt re-seeded window N>0's own first sweep from V's real
  // previous-window trajectory directly, shifted one window back,
  // instead of the driver's documented extrapolate_boundary_value()
  // predictor -- measured to raise this pin to 12/16/15, i.e. MORE
  // sweeps, which is the tell: that seed was persistence, a
  // zeroth-order predictor in time, not the second-order-or-better
  // extrapolation the class is documented to use, and a strictly more
  // accurate initial guess cannot need more iterations. Reverted; this
  // pin is back to 9/13/12. Splitting the single aggregate convergence
  // measure into the two real per-channel transfers (u_out/v_out, Unit
  // C3's per-field-transfer stopping rule) measured NO change to this
  // pin -- on this problem the two channels' own residuals track each
  // other closely enough that testing them separately rather than
  // combined never changes which sweep first satisfies tolerance.)
  // None is NOT the worst here -- on this weakly-coupled, smooth
  // problem (Gray-Scott's u*v^2 coupling term is a small perturbation on
  // the diffusion-dominated Newton system at this dt/diffusion scale)
  // plain Picard already converges in very few sweeps per window, and
  // Aitken's dynamic relaxation occasionally overshoots and costs an extra
  // sweep rather than saving one -- the opposite of the dn_heat example's
  // stiffer Dirichlet-Neumann problem, where acceleration wins decisively
  // (examples/dn_heat/main.cpp). Regression-tested exactly
  // (docs/dev/testing.md: "a silent change in count is how an acceleration
  // bug presents"), with no ordering assumption baked in -- the numbers
  // are reported as measured, not forced to match an expectation.
  EXPECT_EQ(total_none, 9);
  EXPECT_EQ(total_aitken, 13);
  EXPECT_EQ(total_iqn, 12);
}

// ---------------------------------------------------------------- envelope

TEST(GrayScottMultirate, RespectsTheStraddleEnvelope) {
  // Both members Fixed, both intervals fail to divide the window: outside
  // the straddle envelope, rejected at create() with E-RT-WINDOW-STRADDLE --
  // demonstrating (not merely asserting) that every configuration ABOVE,
  // which never does this, sits inside it.
  const Grid grid{kN, kL};
  auto comp_u = std::make_unique<GrayScottU>("U", grid, kDu, kKin, initial_u(grid), kNewtonTol);
  auto comp_v = std::make_unique<GrayScottV>("V", grid, kDv, kKin, initial_v(grid), kNewtonTol);
  const Rational unit{1, 2000};
  WindowMember mu;
  mu.component = comp_u.get(); mu.name = "U"; mu.interval = unit * Rational{5};
  mu.policy = DtPolicy::Fixed; mu.provided = comp_u->provided_ptr();
  mu.required = comp_u->required_ptr(); mu.length = grid.size();
  WindowMember mv;
  mv.component = comp_v.get(); mv.name = "V"; mv.interval = unit * Rational{7};
  mv.policy = DtPolicy::Fixed; mv.provided = comp_v->provided_ptr();
  mv.required = comp_v->required_ptr(); mv.length = grid.size();
  WindowCouplerConfig cfg;
  cfg.window_length = unit * Rational{12};  // neither 5 nor 7 divides 12
  const auto r = WindowCoupler::create({mu, mv}, {{0, 1}, {1, 0}}, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-STRADDLE") << r.error().message;
}
