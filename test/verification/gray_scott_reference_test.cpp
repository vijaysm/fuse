// Plan step 3 ("Verify against a monolithic reference"): a fully coupled,
// single-rate Crank-Nicolson + Newton solve of BOTH species together
// (monolithic_reference.hpp) is the ground truth the multirate windowed
// coupling (species_component.hpp through fuse::WindowCoupler) is checked
// against.
//
// FOUR SEPARATE THINGS, FOUR SECTIONS BELOW, because they are different
// claims with different evidence:
//
//   (1) At m=1 the windowed result matches the monolithic one to the
//       coupling ITERATION tolerance -- but only at a small enough dt. It is
//       not an unconditional property of this model, and this file reports why with a measured
//       number rather than silently loosening a bound (see
//       WindowedMatchesMonolithicAtASufficientlySmallDt). Reaction terms in
//       Gray-Scott are QUADRATIC in the partner (u*v^2): the partitioned
//       scheme freezes the partner at its CN-window MEAN and uses that
//       SAME frozen value at both the CN step's old and new reaction
//       evaluations, while the monolithic scheme uses the partner's actual
//       old/new values separately. u*v^2's nonlinearity in v means those
//       two are not the same expression, so the mismatch does not vanish
//       at any FIXED dt > 0 -- it is a real, order-2-in-dt discrepancy,
//       which WindowedVsMonolithicMismatchConvergesAtOrderTwo measures
//       directly and asserts is in [1.9, 2.1] (the bound used for
//       every order claim in this file).
//   (2) For other ratios (commensurate m=2 and incommensurate 7:13), the
//       windowed scheme's error against a FINE (well-resolved) monolithic
//       reference -- not the coarse, same-dt one used in (1) -- falls at
//       order 2 in dt as the windowed scheme's own dt is refined, holding
//       the ratio fixed (Order2StateErrorCommensurate,
//       Order2StateErrorSevenThirteen).
//   (3) The mass-exchange mismatch: U's own equation removes
//       -u*v^2 using U's frozen mean of v; V's own equation gains
//       +u*v^2 using V's frozen mean of u. Because those means differ
//       (each is the OTHER species' own waveform, not a shared value),
//       "what U removes" and "what V gains" need not cancel exactly.
//       MassExchangeMismatchConvergesAtOrderTwo measures the SUM (which is
//       zero iff they cancel) using the SAME refinement runs as (2), and
//       reports whether it vanishes at the coupling tolerance or converges
//       at order 2 -- this measures which.
//   (4) A long run at Pearson's published self-replicating-spot parameters
//       stays in [0,1] and develops a non-uniform pattern from a
//       near-uniform start (LongPatternFormingRunStaysBoundedAndFormsPattern).
//
// PEARSON PARAMETERS. Pearson (1993) ran the Gray-Scott model on a
// 2.5 x 2.5 periodic square with Du=2e-5, Dv=1e-5, varying F and k; the
// "lambda" class (self-replicating spots) is F=0.0392, k=0.0649
// (mrob.com/pub/comp/xmorphia/pearson-classes.html, a standard secondary
// source for Pearson's own table). Used verbatim in section (4) and as the
// example driver's defaults (examples/gray_scott/main.cpp).
//
// ASYMPTOTIC REGIME (task requirement: "choose parameters so that the
// reaction rates times dt are well below 1"). Sections (1)-(3) use
// F=0.04, k=0.06 (order-1 with Pearson's own values, chosen round for
// readability). dt (or, for the multi-subcycling step members in sections (2)/
// (3), the coarsest member's own window) ranges from 0.0025 (section (1)'s
// finest refinement level) up to 0.25 (section (2)'s commensurate m=2,
// the coarsest level tested anywhere in this file) -- NOT uniformly below
// 0.05 as an earlier draft of this comment claimed. F*dt is therefore
// at most 0.04*0.25 = 0.01 (not <= 0.002), still well below 1, which is
// the actual requirement being satisfied; the tighter 0.05/0.002 figures
// were never true of every refinement level and have been corrected here
// rather than the parameters changed to match them.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include <fuse/runtime/window_coupler.hpp>

#include "gray_scott_physics.hpp"
#include "monolithic_reference.hpp"
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
using gray_scott::MonolithicGrayScott;

namespace {

constexpr int      kN = 8;
constexpr double   kL = 1.0;
constexpr double   kDu = 0.01, kDv = 0.005;
constexpr Kinetics  kKin{/*F=*/0.04, /*k=*/0.06};
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

struct WindowedRun {
  bool                ok = false;
  std::string         error;
  std::vector<double> u, v;
  double              flux_mismatch = 0.0;  ///< comp_u.coupling_flux() + comp_v.coupling_flux()
};

/// U is the window-definer (interval == window); V's interval <= window.
[[nodiscard]] WindowedRun run_windowed(const Grid& grid, Rational window, Rational interval_v,
                                       int n_windows, DtPolicy pol_v) {
  WindowedRun out;
  auto comp_u = std::make_unique<GrayScottU>("U", grid, kDu, kKin, initial_u(grid), kNewtonTol);
  auto comp_v = std::make_unique<GrayScottV>("V", grid, kDv, kKin, initial_v(grid), kNewtonTol);

  WindowMember mu;
  mu.component = comp_u.get(); mu.name = "U"; mu.interval = window; mu.policy = DtPolicy::Adjustable;
  mu.provided = comp_u->provided_ptr(); mu.required = comp_u->required_ptr(); mu.length = grid.size();
  WindowMember mv;
  mv.component = comp_v.get(); mv.name = "V"; mv.interval = interval_v; mv.policy = pol_v;
  mv.provided = comp_v->provided_ptr(); mv.required = comp_v->required_ptr(); mv.length = grid.size();

  WindowCouplerConfig cfg;
  cfg.window_length = window;
  cfg.max_iterations = 300;
  cfg.acceleration = AccelConfig{.kind = AccelKind::Aitken};
  // A convergence measure names
  // a field TRANSFER (a WindowChannel, by its own `name`), not the whole
  // coupled system -- there is no single channel a combined "gray_scott"
  // measure over BOTH directions could name. Split into the two real
  // transfers, same kind/tolerance as the aggregate measure this
  // replaces; both must be satisfied (ConvergenceCriterion::
  // all_satisfied() is a conjunction), which is stricter in general than
  // one combined relative norm over both directions at once. Not
  // byte-identical to before by the user's own choice -- see this file's
  // before/after diagnostics.
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
  cfg.waveform_capacity = 1024;

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
  }
  out.u = comp_u->state();
  out.v = comp_v->state();
  out.flux_mismatch = comp_u->coupling_flux() + comp_v->coupling_flux();
  out.ok = true;
  return out;
}

[[nodiscard]] double state_error(const std::vector<double>& u, const std::vector<double>& v,
                                 const std::vector<double>& u_ref, const std::vector<double>& v_ref) {
  double sq = 0.0;
  for (std::size_t m = 0; m < u.size(); ++m) {
    const double du = u[m] - u_ref[m], dv = v[m] - v_ref[m];
    sq += du * du + dv * dv;
  }
  return std::sqrt(sq / static_cast<double>(u.size()));
}

[[nodiscard]] double order(double e_coarse, double e_fine, double refinement) {
  return std::log(e_coarse / e_fine) / std::log(refinement);
}

void expect_order_two(const std::vector<double>& errs, const char* label) {
  ASSERT_GE(errs.size(), 2u);
  for (std::size_t i = 0; i + 1 < errs.size(); ++i) {
    const double ord = order(errs[i], errs[i + 1], 2.0);
    EXPECT_GE(ord, 1.9) << label << " step " << i << ": " << errs[i] << " -> " << errs[i + 1];
    EXPECT_LE(ord, 2.1) << label << " step " << i << ": " << errs[i] << " -> " << errs[i + 1];
  }
}

}  // namespace

// ---------------------------------------------------------------- (1)

TEST(GrayScottReference, WindowedMatchesMonolithicAtASufficientlySmallDt) {
  const Grid grid{kN, kL};
  const Rational dt{1, 200};  // F*dt = 0.04/200 = 2e-4, deep asymptotic
  const int n_windows = 5;
  const WindowedRun r = run_windowed(grid, dt, dt, n_windows, DtPolicy::Adjustable);
  ASSERT_TRUE(r.ok) << r.error;

  MonolithicGrayScott mono(grid, kDu, kDv, kKin, initial_u(grid), initial_v(grid), kNewtonTol);
  Rational t{0};
  for (int w = 0; w < n_windows; ++w) {
    ASSERT_TRUE(mono.step(dt.to_double(), t.to_double()));
    t = t + dt;
  }
  const double diff = state_error(r.u, r.v, mono.u(), mono.v());
  // Tight, iteration-tolerance-level bound --
  // achievable here because dt is small enough for the O(dt^2) mismatch
  // characterized in the next test to already sit below it.
  EXPECT_LE(diff, 100.0 * kCouplingTol) << "diff=" << diff;
}

TEST(GrayScottReference, WindowedVsMonolithicMismatchConvergesAtOrderTwo) {
  // Same comparison as above (m=1, windowed vs monolithic at the SAME dt),
  // refined at FIXED TOTAL TIME T (n_windows scales with 1/dt so T stays
  // constant across the refinement ladder). Fixed T matters here: an
  // earlier draft of this test fixed n_windows instead and let T shrink as
  // dt shrank, which measures a LOCAL (per-step) O(dt^3) truncation term,
  // not the GLOBAL (accumulated-over-T) O(dt^2) one wanted here --
  // it produced a spurious order ~3. With T fixed, the order below is a
  // clean, repeatable 2.0.
  const Grid grid{kN, kL};
  const double T = 0.1;
  std::vector<double> diffs;
  for (const int denom : {50, 100, 200, 400}) {
    const Rational dt{1, denom};
    const int n_windows = static_cast<int>(std::llround(T * denom));
    const WindowedRun r = run_windowed(grid, dt, dt, n_windows, DtPolicy::Adjustable);
    ASSERT_TRUE(r.ok) << r.error;
    MonolithicGrayScott mono(grid, kDu, kDv, kKin, initial_u(grid), initial_v(grid), kNewtonTol);
    Rational t{0};
    for (int w = 0; w < n_windows; ++w) {
      ASSERT_TRUE(mono.step(dt.to_double(), t.to_double()));
      t = t + dt;
    }
    diffs.push_back(state_error(r.u, r.v, mono.u(), mono.v()));
  }
  expect_order_two(diffs, "m=1 windowed-vs-monolithic mismatch");
}

// ---------------------------------------------------------------- (2)/(3)

// Multirate order: the state error against a fine monolithic reference, and
// the mismatch between what U removes and what V gains through u v^2, both at
// order 2 in dt at a fixed ratio. These were disabled while WindowCoupler
// wrote its (element, subcycling step) sweep vector transposed -- invisible for a
// scalar member or one subcycling step per window, and the reason U was once fed a
// "mean" of V outside V's own range. The mass measure only means something
// because each species checkpoints its accumulated flux with its state;
// outside the checkpoint it summed every sweep and replay.
TEST(GrayScottReference, Order2StateErrorAndMassExchangeCommensurate) {
  const Grid grid{kN, kL};
  const double T = 0.5;
  // Fine truth: dt 16x finer than the finest level tested below (m=16).
  const Rational dt_truth = Rational{static_cast<std::int64_t>(T * 100000), 100000} / Rational{16 * 16};
  MonolithicGrayScott truth(grid, kDu, kDv, kKin, initial_u(grid), initial_v(grid), kNewtonTol);
  {
    Rational t{0};
    const int n = 16 * 16;
    for (int s = 0; s < n; ++s) {
      ASSERT_TRUE(truth.step(dt_truth.to_double(), t.to_double()));
      t = t + dt_truth;
    }
  }

  std::vector<double> errs, flux_mismatches;
  for (const int m : {2, 4, 8, 16}) {
    const Rational window = Rational{static_cast<std::int64_t>(T * 100000), 100000} / Rational{m};
    const Rational v_interval = window / Rational{2};  // ratio fixed at 2 across the refinement
    const WindowedRun r = run_windowed(grid, window, v_interval, m, DtPolicy::Adjustable);
    ASSERT_TRUE(r.ok) << "m=" << m << ": " << r.error;
    errs.push_back(state_error(r.u, r.v, truth.u(), truth.v()));
    flux_mismatches.push_back(std::abs(r.flux_mismatch));
  }
  expect_order_two(errs, "commensurate state error vs fine reference");
  expect_order_two(flux_mismatches, "commensurate mass-exchange mismatch");
  // "State which is true": the mismatch does NOT vanish at the coupling
  // tolerance -- it is well above kCouplingTol at every refinement level
  // measured, and converges at order 2 instead (asserted above).
  for (double fm : flux_mismatches) EXPECT_GT(fm, kCouplingTol);
}

// The same at the incommensurate 7:13 ratio, where V straddles window ends.
TEST(GrayScottReference, Order2StateErrorAndMassExchangeSevenThirteen) {
  // Phase-aligned refinement, mirroring
  // test/runtime/window_coupler_test.cpp's asymptotic_refinements(): window
  // = 13 parts, V = 7 parts, d = 35*m, n_windows = 7*m holds
  // T = 7m*13/(35m) = 91/35 = 2.6 fixed across every m, and 7:13's gcd = 1
  // means the straddle's phase within its own subcycling step repeats every 7
  // windows -- n_windows = 7*m is always a whole multiple of that cycle.
  const Grid grid{kN, kL};
  const double T = 2.6;

  // Fine truth at the finest level's own scale, 16x further refined.
  const int m_finest = 8;
  const Rational dt_truth = Rational{7, 35 * m_finest * 16};
  MonolithicGrayScott truth(grid, kDu, kDv, kKin, initial_u(grid), initial_v(grid), kNewtonTol);
  {
    Rational t{0};
    const int n = static_cast<int>(std::llround(T / dt_truth.to_double()));
    for (int s = 0; s < n; ++s) {
      ASSERT_TRUE(truth.step(dt_truth.to_double(), t.to_double()));
      t = t + dt_truth;
    }
  }

  std::vector<double> errs, flux_mismatches;
  for (const int m : {2, 4, 8}) {
    const std::int64_t d = 35 * m;
    const Rational window{13, d};
    const Rational v_interval{7, d};
    const int n_windows = 7 * m;
    const WindowedRun r = run_windowed(grid, window, v_interval, n_windows, DtPolicy::Fixed);
    ASSERT_TRUE(r.ok) << "m=" << m << ": " << r.error;
    errs.push_back(state_error(r.u, r.v, truth.u(), truth.v()));
    flux_mismatches.push_back(std::abs(r.flux_mismatch));
  }
  expect_order_two(errs, "7:13 state error vs fine reference");
  expect_order_two(flux_mismatches, "7:13 mass-exchange mismatch");
  for (double fm : flux_mismatches) EXPECT_GT(fm, kCouplingTol);
}

// ---------------------------------------------------------------- (4)

TEST(GrayScottReference, LongPatternFormingRunStaysBoundedAndFormsPattern) {
  // Pearson (1993) "lambda" class: self-replicating spots.
  const Grid grid{48, 2.5};
  const Kinetics pearson{/*F=*/0.0392, /*k=*/0.0649};
  const double Du = 2.0e-5, Dv = 1.0e-5;

  std::vector<double> u0(grid.size(), 1.0), v0(grid.size(), 0.0);
  const int lo = grid.n / 2 - 2, hi = grid.n / 2 + 2;
  for (int i = lo; i < hi; ++i) {
    for (int j = lo; j < hi; ++j) {
      u0[grid.index(i, j)] = 0.5;
      v0[grid.index(i, j)] = 0.25;
    }
  }
  // Near-uniform: the seeded patch is 4x4 cells out of 48x48 (< 0.7% of the
  // domain); everywhere else starts exactly at the trivial state (u=1,v=0).
  ASSERT_LT(static_cast<double>((hi - lo) * (hi - lo)) / static_cast<double>(grid.size()), 0.01);

  MonolithicGrayScott mono(grid, Du, Dv, pearson, u0, v0, 1.0e-10);
  const double dt = 2.0;
  double t = 0.0;
  // MEASURED (re-verified at this N=48/domain-2.5/seed-4x4 configuration,
  // at -O2 for the timing only -- physics identical to this -O0 test
  // binary): variance(v) is 1.8760e-04 at step 40 (t=82), dips to a local
  // minimum of 1.3787e-04 at step 50 (t=102) -- still above 1e-4 -- then
  // rises to a sustained plateau of 2.4719e-04 from step ~150 (t~302)
  // onward, unchanged through step 219 (t=440) (a single stable spot at
  // this domain size/resolution, not sustained splitting -- a legitimate
  // "pattern forms and persists" outcome, not the runaway splitting
  // cascade a larger domain would show). Smaller proxy grids (N=16, N=24)
  // were tried first to shorten this test's runtime and REJECTED: at
  // both, the same seed instead decays back toward the trivial uniform
  // state (variance falling to ~1e-43 by t=500), never sustaining a
  // pattern -- this reaction-diffusion model's spot-sustaining behaviour
  // is resolution-sensitive, so N=48 is kept and only the run length is
  // cut. Run length below (60 steps, t=120) is the
  // smallest that stays safely past that first crossing -- this
  // MonolithicGrayScott::step() Newton-SparseLU solve is markedly slower
  // in this project's Debug (-O0) test build than at -O2 (measured ~2 s/
  // step here vs ~25 ms/step at -O2), so 220 steps (the original,
  // fully-plateaued run) cost ~8 minutes; 60 is the practical CTest-scale
  // compromise. See docs/user/gray-scott.md "Long pattern-forming run"
  // for the full measured trace and why this resolution/domain size was
  // chosen over the literature's finer ones.
  constexpr double kVarianceThreshold = 1.0e-4;
  bool threshold_reached = false;
  for (int step = 0; step < 60; ++step) {
    ASSERT_TRUE(mono.step(dt, t)) << "Newton failed to converge at step " << step;
    t += dt;
    for (double x : mono.u()) ASSERT_GE(x, -1.0e-9) << "u negative at step " << step;
    for (double x : mono.u()) ASSERT_LE(x, 1.0 + 1.0e-9) << "u above 1 at step " << step;
    for (double x : mono.v()) ASSERT_GE(x, -1.0e-9) << "v negative at step " << step;
    for (double x : mono.v()) ASSERT_LE(x, 1.0 + 1.0e-9) << "v above 1 at step " << step;
  }
  double mean_v = 0.0;
  for (double x : mono.v()) mean_v += x;
  mean_v /= static_cast<double>(grid.size());
  double var_v = 0.0;
  for (double x : mono.v()) var_v += (x - mean_v) * (x - mean_v);
  var_v /= static_cast<double>(grid.size());
  threshold_reached = var_v > kVarianceThreshold;
  EXPECT_TRUE(threshold_reached) << "variance(v)=" << var_v << " did not exceed " << kVarianceThreshold;
}
