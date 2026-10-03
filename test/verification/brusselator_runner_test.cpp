// Time windows track (Conformance, Unit 1): the Brusselator's own graph
// (brusselator_graph.hpp), compiled through fuse::Runner, reproduces BIT
// FOR BIT (EXPECT_EQ, no tolerance) the existing fuse::WindowCoupler-
// driven path (examples/brusselator/main.cpp's `--driver window`,
// unchanged) -- the same oracle relationship test/runtime/window_runner_
// multirate_test.cpp pins for its own linear-ODE pair, now for a
// genuinely spatial, non-matching-mesh, supermesh-remapped pair of
// components with a real implicit (Picard) reaction fixed point inside
// every advance().
//
// STATUS (Unit 1 exit, see the lead's own ledger): building this test
// found and FIXED FOUR real `fuse::Runner` defects, every one
// regression-tested against the full existing window_runner_* suite
// (unaffected) --
//   (a) `Runner::Impl::advance_window()`'s own "first sweep of a new
//       window instance" detection compared `Instruction::dst_step`
//       (period-RELATIVE) directly, with no period counter -- every
//       window after the first silently did not run at all. Fixed by
//       commit 7939643 (`window_absolute_instance()`).
//   (b) `ClockSet`'s own constructor fed a window-scoped member's own
//       fine subcycling interval into `checked_common_period()`'s LCM
//       instead of its region's window length, reintroducing a disguised
//       least-common-multiple window (E-SCHED-PERIOD on the example's own
//       --dt-u 0.002/--dt-v 0.003/--window 0.02, since 0.02 does not
//       divide lcm(0.002,0.003)=0.006). Fixed by commit c083ea6.
//   (c) `advance_window()`'s own window-start "Sample 0" re-stamp read
//       `cfg.provided[e]` LIVE at the top of every sweep instead of a
//       value captured once, before any sweep in the pass could mutate
//       it: `Component::restore_state()` is only contractually required
//       to restore what `save_state()` captured (the component's own
//       hidden state), never to resynchronize a `Provides` field that is
//       itself a function of that state -- so a rejected sweep's own
//       last `advance()` output silently became "the window-start value"
//       from the SECOND sweep of every window on. This is the EXACT
//       defect `WindowCoupler::sweep_window()`'s own `start_value[i][e]`
//       (window_coupler.cpp's own comment, "captured ONCE, right after
//       save_state(), before any advance() this window can mutate
//       `provided`") was already fixed to remove on the driver side --
//       found here (not guessed) by an element-by-element, sweep-by-
//       sweep hexfloat trace of the example's own first window against
//       the driver on the example's own real parameters: sweeps 0 and 1
//       of window 0 matched bit for bit; sweep 2's own import samples
//       diverged completely, traced to the exporting member's own
//       "Sample 0" silently holding sweep 1's own last output instead of
//       the true window-start checkpoint. `Runner::Impl::window_start_
//       value` (src/runtime/runner_impl.hpp/iteration.cpp) is the fix:
//       cached once on `first_sweep_of_pass`, mirroring `window_pass_
//       start_clock`'s own existing pattern for the identical class of
//       bug in the clock instead of the exported value. Before this fix,
//       `--driver runner --end 0.1` on the example's own real parameters
//       measured max_window_balance_error=1.686e-03 -- to four digits,
//       the EXACT historical pre-fix conservation defect examples/
//       brusselator's own 2026-09-26 history (docs/user/brusselator.md,
//       KNOWN_ISSUES) already named and fixed once, on the driver side,
//       for the identical reason. After the fix: 1.703e-13 (round-off),
//       bit-for-bit identical to the driver's own figure.
//   (d) `run_region()`'s own per-block, named-measure residual norm
//       (`MeasureKind::Relative`, the only kind the Brusselator's own
//       graph uses) summed the window-scoped value arc's FULL
//       `capacity`-wide buffer -- window-START sample included -- while
//       `WindowCoupler::sweep_window()`'s own named-measure residual
//       (its own comment: "each measure is tested against the residual
//       of the field transfer it names, as the Runner does") sums
//       EXACTLY the consumer's own substep count, never +1 for the
//       window-start sample. That sample never changes within one
//       window instance (fix (c)'s own `window_start_value`), so it
//       never moved the RESIDUAL, but inflated the Runner's own
//       reference norm by a constant every sweep, loosening a Relative
//       tolerance just enough to converge one sweep earlier than the
//       driver on the example's own window 20 -- the two sides'
//       hexfloat `ref` values disagreeing from the third significant
//       digit (`0x1.6be7e489b2a8cp+5` vs `0x1.5b510b3c3ad89p+5` at
//       sweep k=1), confirmed NOT a repeat of (c) by the fixed-
//       iterations bisection below (`PerWindowStateMatchesDriverWith
//       StoppingRuleRemoved`): every one of the 150 windows' own
//       committed state matched bit for bit with the stopping rule
//       removed, so only the DECISION, never the state, disagreed.
//       `RegionBlock::window_export_capacity`/`window_export_valid_
//       substeps` (src/runtime/runner_impl.hpp) and the stripped-
//       gather norm in `run_region()` (src/runtime/iteration.cpp) are
//       the fix. After it, `JacobiMatchesDriver`/`GaussSeidelMatches
//       Driver` below pass EXPECT_EQ bit for bit over the full
//       150-window span, both waveform iterations.
//
// Grid: small (mutually non-nested, not a divisor/multiple of one
// another, so the supermesh remap genuinely overlaps cells) -- this
// file's job is the structural claim at a size that runs in seconds, not
// the physics scale the example's own `example.brusselator`/`example.
// brusselator.runner` ctest rows already cover at the production grid
// (16x16/24x24). Forcing is turned off here (`PhysicsConfig::forcing =
// false`) so the comparison is not entangled with the forcing-switch
// boundary-insertion logic `examples/brusselator/main.cpp`'s own
// `window_boundaries()` handles. dt_u/dt_v/window are the example's own
// real values (0.002/0.003/0.02, not an LCM-equal diagnostic substitute
// -- defect (b) above is what made the real values schedule at all), and
// kNumWindows spans the example's own `--check` acceptance span (--end
// 3 / window 0.02 = 150 windows).

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/runtime/window_coupler.hpp>

#include "brusselator_graph.hpp"
#include "brusselator_physics.hpp"
#include "species_component.hpp"

using fuse::AccelConfig;
using fuse::AccelKind;
using fuse::ComponentSet;
using fuse::DtPolicy;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Result;
using fuse::WaveformIteration;
using fuse::WindowChannel;
using fuse::WindowCoupler;
using fuse::WindowCouplerConfig;
using fuse::WindowMember;

namespace {

// Small, mutually non-nested quad meshes -- not a divisor/multiple of one
// another, so the supermesh remap genuinely overlaps cells rather than
// degenerating to a permutation.
constexpr int kNxU = 5, kNyU = 5, kNxV = 7, kNyV = 7;
// The example's own real values (examples/brusselator/main.cpp's own
// defaults): dt_u does not divide dt_v's own lcm relationship specially,
// and the window (0.02) divides neither member's own fine interval's lcm
// (0.006) evenly -- defect (b) in this file's own header is exactly what
// makes this schedule at all; ClockSet now gives a window-scoped member's
// clock its region's window length instead of its own fine interval, so
// the compiled period is 0.02, not 0.006.
constexpr double kDtU = 0.002, kDtV = 0.003, kWindow = 0.02;
// The example's own --check span: --end 3 / window 0.02 = 150 windows.
constexpr int kNumWindows = 150;
constexpr double kTol = 1e-10;
constexpr int kMaxIterations = 100;

struct RunResult {
  bool ok = false;
  std::string error;
  brusselator::Vector u_final, v_final;
  std::vector<int> iterations_per_window;
  std::vector<double> u0_per_window;  // diagnostic: element 0 of U after each window
};

/// Builds fresh U/V components and the weights they capture pointers to --
/// everything a run (either path) owns, so two runs in the same test never
/// alias.
struct Rig {
  fuse::native::NativeMeshBackend backend;
  brusselator::NativeGrid u_grid;
  brusselator::NativeGrid v_grid;
  fuse::RemapWeights w_vu, w_uv;  // V->U, U->V
  std::unique_ptr<brusselator::VComponent> comp_v;
  std::unique_ptr<brusselator::UComponent> comp_u;

  Rig(fuse::native::NativeMeshBackend backend_in, brusselator::NativeGrid u,
      brusselator::NativeGrid v)
      : backend(std::move(backend_in)), u_grid(std::move(u)), v_grid(std::move(v)) {}
};

[[nodiscard]] Result<std::unique_ptr<Rig>> make_rig() {
  fuse::native::NativeMeshBackend backend{0, 1};
  auto u_grid_r = brusselator::NativeGrid::make(backend, "u", kNxU, kNyU);
  if (!u_grid_r.ok()) return u_grid_r.error();
  auto v_grid_r = brusselator::NativeGrid::make(backend, "v", kNxV, kNyV);
  if (!v_grid_r.ok()) return v_grid_r.error();

  auto rig = std::make_unique<Rig>(std::move(backend), std::move(u_grid_r.value()),
                                   std::move(v_grid_r.value()));

  fuse::CouplingGraph scratch_graph;
  const fuse::MeshId u_mesh = scratch_graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" + std::to_string(kNyU)});
  const fuse::MeshId v_mesh = scratch_graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(kNxV) + "x" + std::to_string(kNyV)});
  const fuse::RegionId u_region =
      scratch_graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  const fuse::RegionId v_region =
      scratch_graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  fuse::RemapPlan plan_vu;
  plan_vu.src = v_region;
  plan_vu.tgt = u_region;
  auto w_vu = fuse::build_weights(plan_vu, scratch_graph, rig->backend);
  if (!w_vu.ok()) return w_vu.error();
  fuse::RemapPlan plan_uv;
  plan_uv.src = u_region;
  plan_uv.tgt = v_region;
  auto w_uv = fuse::build_weights(plan_uv, scratch_graph, rig->backend);
  if (!w_uv.ok()) return w_uv.error();
  rig->w_vu = std::move(w_vu.value());
  rig->w_uv = std::move(w_uv.value());

  brusselator::PhysicsConfig cfg;
  cfg.forcing = false;  // see file header
  const int n_u = rig->u_grid.size();
  rig->comp_v = std::make_unique<brusselator::VComponent>(
      "V", rig->u_grid, rig->v_grid, cfg, &rig->w_uv, &rig->w_vu);
  brusselator::Vector v0_on_u =
      Eigen::Map<brusselator::Vector>(rig->comp_v->provided_ptr(), n_u);
  rig->comp_u = std::make_unique<brusselator::UComponent>("U", rig->u_grid, cfg, v0_on_u);
  return rig;
}

// Diagnostic knob (Unit 1 debugging, temporary): when > 0, both paths use
// an unreachable tolerance and exactly this many sweeps per window --
// window_runner_straddle_test.cpp's own technique for telling "the two
// stopping rules disagree" apart from "the underlying physics disagree".
int g_fixed_iterations = 0;

[[nodiscard]] RunResult run_via_driver(WaveformIteration mode) {
  RunResult out;
  auto rig_r = make_rig();
  if (!rig_r.ok()) { out.error = rig_r.error().message; return out; }
  auto rig = std::move(rig_r.value());
  const int n_u = rig->u_grid.size();

  WindowMember mu;
  mu.component = rig->comp_u.get();
  mu.name = "U";
  mu.interval = Rational{std::llround(kDtU * 1000000), 1000000};
  mu.policy = DtPolicy::Adjustable;
  mu.provided = rig->comp_u->provided_ptr();
  mu.required = rig->comp_u->required_ptr();
  mu.length = static_cast<std::size_t>(n_u);

  WindowMember mv;
  mv.component = rig->comp_v.get();
  mv.name = "V";
  mv.interval = Rational{std::llround(kDtV * 1000000), 1000000};
  mv.policy = DtPolicy::Adjustable;
  mv.provided = rig->comp_v->provided_ptr();
  mv.required = rig->comp_v->required_ptr();
  mv.length = static_cast<std::size_t>(n_u);

  const bool gauss_seidel = (mode == WaveformIteration::GaussSeidel);
  const std::vector<WindowChannel> channels = {
      {.provider = 0, .consumer = 1, .lagged = false, .name = "u_out"},
      {.provider = 1, .consumer = 0, .lagged = gauss_seidel, .name = "v_out"}};

  const double tol = g_fixed_iterations > 0 ? 1e-300 : kTol;
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{std::llround(kWindow * 1000000), 1000000};
  cfg.max_iterations = g_fixed_iterations > 0 ? g_fixed_iterations : kMaxIterations;
  cfg.acceleration = AccelConfig{.kind = AccelKind::None};
  cfg.iteration = mode;
  cfg.order = 2;
  cfg.waveform_capacity = 64;
  cfg.abort_on_nonconvergence = (g_fixed_iterations == 0);
  fuse::ConvergenceMeasure mu_measure;
  mu_measure.kind = fuse::MeasureKind::Relative;
  mu_measure.tolerance = tol;
  mu_measure.field = "u_out";
  cfg.convergence.add(mu_measure);
  fuse::ConvergenceMeasure mv_measure;
  mv_measure.kind = fuse::MeasureKind::Relative;
  mv_measure.tolerance = tol;
  mv_measure.field = "v_out";
  cfg.convergence.add(mv_measure);

  auto created = WindowCoupler::create({mu, mv}, channels, cfg);
  if (!created.ok()) { out.error = created.error().message; return out; }
  auto coupler = std::move(created.value());

  for (int w = 0; w < kNumWindows; ++w) {
    const auto r = coupler->run_window();
    if (!r.ok()) { out.error = r.error().message; return out; }
    out.iterations_per_window.push_back(coupler->stats().iterations);
    out.u0_per_window.push_back(rig->comp_u->state()[0]);
  }
  out.ok = true;
  out.u_final = rig->comp_u->state();
  out.v_final = rig->comp_v->state();
  return out;
}

[[nodiscard]] RunResult run_via_runner(WaveformIteration mode) {
  RunResult out;
  auto rig_r = make_rig();
  if (!rig_r.ok()) { out.error = rig_r.error().message; return out; }
  auto rig = std::move(rig_r.value());

  fuse::CouplingGraph graph;
  const fuse::MeshId u_mesh = graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" + std::to_string(kNyU)});
  const fuse::MeshId v_mesh = graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(kNxV) + "x" + std::to_string(kNyV)});
  const fuse::RegionId u_region = graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  (void)graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  const Rational dt_u{std::llround(kDtU * 1000000), 1000000};
  const Rational dt_v{std::llround(kDtV * 1000000), 1000000};
  const Rational window{std::llround(kWindow * 1000000), 1000000};
  const double tol = g_fixed_iterations > 0 ? 1e-300 : kTol;
  const int max_iters = g_fixed_iterations > 0 ? g_fixed_iterations : kMaxIterations;
  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      graph, u_region, dt_u, dt_v, window, tol, max_iters, AccelKind::None, mode);
  (void)handles;
  if (const auto r = rig->backend.register_default_placements(graph, u_mesh); !r.ok()) {
    out.error = r.error().message;
    return out;
  }

  ComponentSet components;
  components.add(std::move(rig->comp_u));
  components.add(std::move(rig->comp_v));

  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, graph);
    if (!r.ok()) { out.error = r.error().message; return out; }
  }

  const auto lowered = lower(graph);
  if (!lowered.ok()) { out.error = lowered.diags.to_string(); return out; }
  const fuse::ClockSet clocks{graph};
  auto sched = schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, rig->backend));
  if (!sched.ok()) { out.error = sched.diags.to_string(); return out; }

  fuse::RunnerConfig rcfg;
  rcfg.abort_on_nonconvergence = (g_fixed_iterations == 0);
  fuse::Runner runner{graph, lowered.dag, sched.plan, rig->backend};
  const auto init = runner.initialize(components, registry, rcfg);
  if (!init.ok()) { out.error = init.error().message; return out; }

  brusselator::UComponent* pu = nullptr;
  brusselator::VComponent* pv = nullptr;
  for (auto& c : components) {
    if (c.name() == "U") pu = dynamic_cast<brusselator::UComponent*>(&c);
    if (c.name() == "V") pv = dynamic_cast<brusselator::VComponent*>(&c);
  }
  if (pu == nullptr || pv == nullptr) { out.error = "component lookup failed"; return out; }

  for (int w = 0; w < kNumWindows; ++w) {
    const auto r = runner.run_period(components);
    if (!r.ok()) { out.error = r.error().message; return out; }
    if (runner.iteration_stats().empty()) { out.error = "no iteration stats"; return out; }
    out.iterations_per_window.push_back(runner.iteration_stats().front().iterations);
    out.u0_per_window.push_back(pu->state()[0]);
  }
  out.ok = true;
  out.u_final = pu->state();
  out.v_final = pv->state();
  return out;
}

void expect_vectors_equal(const brusselator::Vector& a, const brusselator::Vector& b,
                          const char* label) {
  ASSERT_EQ(a.size(), b.size()) << label;
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i], b[i]) << label << " differs at element " << i;
  }
}

}  // namespace

// Historical (kept DISABLED, not deleted): this file's header, defect (c),
// has the full account -- this is the "pin the sweep count, remove the
// stopping rule from the comparison" technique (window_runner_straddle_
// test.cpp's own convention) that FIRST isolated defect (c)'s own fixed-
// point disagreement from a stopping-rule difference. CURRENT STATUS
// (measured, not claimed): with defect (c) fixed, this exact diagnostic
// (g_fixed_iterations=30, one window) passes -- but JacobiMatchesDriver/
// GaussSeidelMatchesDriver below, run over the example's own real
// --end-3 span (150 windows) with the REAL stopping rule, still FAIL,
// by a MUCH smaller and qualitatively DIFFERENT margin than defect (c)
// ever showed: bit-for-bit identical through window 19, then a one-ULP-
// level difference in U/V's own committed state from window 20 on,
// accompanied by a one-sweep iteration-count disagreement at window 20
// itself (Jacobi: 7 vs 9; GaussSeidel: 5 vs 6). Re-running the SAME
// 150-window comparison with g_fixed_iterations pinned (bypassing the
// stopping rule entirely) found ZERO divergence across all 150 windows
// -- so this is a convergence-DECISION sensitivity (the Runner's and
// driver's own residual-norm computations, while using the SAME
// deterministic_norm_squared[_diff] formula by the user's own 2026-09-29
// ruling, apparently sum this field's own multiple elements/substeps in
// a different order, producing a last-ULP-different norm that crosses
// the 1e-10 threshold on a different sweep for a multi-element,
// multi-substep field), not a repeat of defect (c)'s own state
// corruption. Left DISABLED, not yet fixed; see this file's own report
// to the lead for the measured first-differing window and value.
TEST(BrusselatorRunner, DISABLED_DiagnosticFixedIterations) {
  g_fixed_iterations = 30;
  const RunResult via_runner = run_via_runner(WaveformIteration::Jacobi);
  const RunResult via_driver = run_via_driver(WaveformIteration::Jacobi);
  g_fixed_iterations = 0;
  ASSERT_TRUE(via_runner.ok) << via_runner.error;
  ASSERT_TRUE(via_driver.ok) << via_driver.error;
  expect_vectors_equal(via_runner.u_final, via_driver.u_final, "U state (fixed-iterations)");
  expect_vectors_equal(via_runner.v_final, via_driver.v_final, "V state (fixed-iterations)");
}

// Isolated defect (d) above (JacobiMatchesDriver/GaussSeidelMatchesDriver's
// own one-sweep convergence-decision divergence from window 20 on, before
// that fix) from defect (c): the SAME 150-window comparison, with the
// stopping rule taken out of the picture entirely (window_runner_
// straddle_test.cpp's own "unreachable tolerance, fixed sweep count"
// technique -- this file's own DISABLED_DiagnosticFixedIterations used it
// for defect (c)). PASSES: every one of the 150 windows' own committed
// U[0] is bit-for-bit identical between the Runner and the driver when
// neither side's own convergence check is ever consulted -- this is what
// proved defect (d) a convergence-DECISION sensitivity (a residual/
// reference-norm difference occasionally crossing the 1e-10 threshold on
// a different sweep), not a second instance of defect (c)'s own state
// corruption, which this exact technique already proved present at EVERY
// window once forced sweeps were used. Kept, post-fix, as a standing
// regression distinct from JacobiMatchesDriver below: it isolates STATE
// from the DECISION, so a future regression in either shows up as a
// failure in the test that actually isolates it.
TEST(BrusselatorRunner, PerWindowStateMatchesDriverWithStoppingRuleRemoved) {
  g_fixed_iterations = 15;
  const RunResult via_runner = run_via_runner(WaveformIteration::Jacobi);
  const RunResult via_driver = run_via_driver(WaveformIteration::Jacobi);
  g_fixed_iterations = 0;
  ASSERT_TRUE(via_runner.ok) << via_runner.error;
  ASSERT_TRUE(via_driver.ok) << via_driver.error;
  ASSERT_EQ(via_runner.u0_per_window.size(), via_driver.u0_per_window.size());
  for (std::size_t w = 0; w < via_runner.u0_per_window.size(); ++w) {
    EXPECT_EQ(via_runner.u0_per_window[w], via_driver.u0_per_window[w]) << "window " << w;
  }
}

// Unit 1's own acceptance bar: the Runner, driven by the real stopping
// rule (not the fixed-iterations bisection above), bit for bit against
// the driver over the full 150-window span -- committed state AND
// iteration count both. Depends on all four defects in this file's own
// header being fixed; before defect (d)'s own fix this failed first at
// window 20 (a one-sweep iteration-count disagreement), after matching
// bit for bit through window 19.
TEST(BrusselatorRunner, JacobiMatchesDriver) {
  const RunResult via_runner = run_via_runner(WaveformIteration::Jacobi);
  ASSERT_TRUE(via_runner.ok) << via_runner.error;
  const RunResult via_driver = run_via_driver(WaveformIteration::Jacobi);
  ASSERT_TRUE(via_driver.ok) << via_driver.error;

  expect_vectors_equal(via_runner.u_final, via_driver.u_final, "U state (Jacobi)");
  expect_vectors_equal(via_runner.v_final, via_driver.v_final, "V state (Jacobi)");
  ASSERT_EQ(via_runner.iterations_per_window.size(), via_driver.iterations_per_window.size());
  for (std::size_t w = 0; w < via_runner.iterations_per_window.size(); ++w) {
    EXPECT_EQ(via_runner.iterations_per_window[w], via_driver.iterations_per_window[w])
        << "window " << w;
  }
}

TEST(BrusselatorRunner, GaussSeidelMatchesDriver) {
  const RunResult via_runner = run_via_runner(WaveformIteration::GaussSeidel);
  ASSERT_TRUE(via_runner.ok) << via_runner.error;
  const RunResult via_driver = run_via_driver(WaveformIteration::GaussSeidel);
  ASSERT_TRUE(via_driver.ok) << via_driver.error;

  expect_vectors_equal(via_runner.u_final, via_driver.u_final, "U state (GaussSeidel)");
  expect_vectors_equal(via_runner.v_final, via_driver.v_final, "V state (GaussSeidel)");
  ASSERT_EQ(via_runner.iterations_per_window.size(), via_driver.iterations_per_window.size());
  for (std::size_t w = 0; w < via_runner.iterations_per_window.size(); ++w) {
    EXPECT_EQ(via_runner.iterations_per_window[w], via_driver.iterations_per_window[w])
        << "window " << w;
  }
}

// GaussSeidel converges in no more sweeps than Jacobi on this problem (the
// same property window_coupler_test.cpp/window_runner_multirate_test.cpp
// pin for their own problems) -- confirms this test actually exercises
// two distinct algorithms, not one mode silently falling back to the
// other.
TEST(BrusselatorRunner, GaussSeidelConvergesInNoMoreIterationsThanJacobi) {
  const RunResult jacobi = run_via_runner(WaveformIteration::Jacobi);
  ASSERT_TRUE(jacobi.ok) << jacobi.error;
  const RunResult gs = run_via_runner(WaveformIteration::GaussSeidel);
  ASSERT_TRUE(gs.ok) << gs.error;
  ASSERT_EQ(jacobi.iterations_per_window.size(), gs.iterations_per_window.size());
  for (std::size_t w = 0; w < jacobi.iterations_per_window.size(); ++w) {
    EXPECT_LE(gs.iterations_per_window[w], jacobi.iterations_per_window[w]) << "window " << w;
  }
}
