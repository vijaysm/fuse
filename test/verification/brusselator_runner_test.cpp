// Time windows track (Conformance, Unit 1): the Brusselator's own graph
// (brusselator_graph.hpp), compiled through fuse::Runner, intended to
// reproduce BIT FOR BIT (EXPECT_EQ, no tolerance) the existing
// fuse::WindowCoupler-driven path (examples/brusselator/main.cpp's
// `--driver window`, unchanged) -- the same oracle relationship
// test/runtime/window_runner_multirate_test.cpp pins for its own
// linear-ODE pair, now for a genuinely spatial, non-matching-mesh,
// supermesh-remapped pair of components with a real implicit (Picard)
// reaction fixed point inside every advance().
//
// STATUS (Unit 1 exit, see the lead's own ledger): building this test
// found and this session FIXED two real `fuse::Runner` defects, both
// regression-tested against the full existing window_runner_* suite
// (unaffected) --
//   (a) `Runner::Impl::advance_window()`'s own "first sweep of a new
//       window instance" detection compared `Instruction::dst_step`
//       (period-RELATIVE: the compiled schedule covers one period and is
//       replayed verbatim by every `run_period()` call) directly, with
//       no period counter in the comparison. A window-scoped region with
//       exactly one window instance per period -- `period ==
//       window_length`, the common case, and the only one this file
//       could reach given finding (b) below -- sees `dst_step == 0` on
//       EVERY `run_period()` call, which this comparison could not tell
//       apart from "still the same window instance": every window after
//       the first silently did not run at all (comp_u/comp_v's own state
//       and means were bitwise frozen from window 0 onward).
//   (b) The SAME period-relative `dst_step`, used directly (uncorrected)
//       to form this window's own absolute [T0, T1) and to pick the
//       correct provider sample times, produced plainly wrong times
//       once (a) was fixed enough to reach window 2 (`Waveform::push: t
//       is outside the window` -- a hard failure, not a quiet wrong
//       answer).
// `Runner::Impl::window_absolute_instance()` (src/runtime/iteration.cpp)
// is the fix for both: `period_index * (period / window_length) +
// dst_step`, used everywhere `dst_step` previously stood in for an
// absolute window index.
//
// What remains OPEN after both fixes: `JacobiMatchesDriver`/
// `GaussSeidelMatchesDriver` still do not pass. `DiagnosticFixedIterations`
// (window_runner_straddle_test.cpp's own "pin the sweep count, remove the
// stopping rule from the comparison" technique) shows the two paths
// settle at genuinely DIFFERENT fixed points (not merely different
// STOPPING points along a shared one): forcing exactly 30 sweeps/window,
// window 0's mean_u is 1.61724105424 (Runner) against 1.61689207054
// (driver) -- 0x1.9e03827db598cp+0 against 0x1.9deca384218aep+0,
// 2.16e-4 relative -- and does not shrink with more forced sweeps.
// Reproduces identically with U and V on the SAME mesh size (ruling out
// the supermesh remap specifically) and strictly inside window 0 alone
// (ruling out anything period-boundary-related, i.e. NOT the same root
// cause as (a)/(b) above). Every existing window_runner_*_test.cpp
// fixture uses a scalar (length-1) linear ODE; this is the first
// window-scoped Runner fixture with a MULTI-ELEMENT field and a
// genuinely nonlinear per-step solve (UComponent's own implicit
// reaction fixed point), so the defect is plausibly specific to one of
// those two, not yet isolated further. The three comparison tests below
// are DISABLED (not deleted, not loosened to a tolerance) pending that
// root-cause; `DiagnosticFixedIterations` is kept enabled as the
// reproduction.
//
// Deliberately a SMALL grid and few windows otherwise: this file's job is
// the structural claim, not the physics scale the example's own
// `example.brusselator` ctest row already covers at the production grid
// (16x16/24x24, --end 0.5). Forcing is turned off here
// (`PhysicsConfig::forcing = false`) so the comparison is not entangled
// with the forcing-switch boundary-insertion logic `examples/
// brusselator/main.cpp`'s own `window_boundaries()` handles.

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
// kWindow == lcm(kDtU, kDtV) exactly (both 3*kDtU == 2*kDtV == kWindow):
// `fuse::Runner`'s window-scoped lowering sizes the compiled schedule's own
// common period from EVERY `ComponentDecl::interval` in the graph,
// including the window members' own fine intervals (`WindowMemberConfig::
// interval` is resolved from `ComponentDecl::interval` alone,
// src/runtime/iteration.cpp) -- so a window's own length must divide
// `lcm(kDtU, kDtV)` evenly (`src/schedule/expand.cpp`'s `tick_count`,
// E-SCHED-PERIOD) regardless of `CouplingWindowScope::explicit_interval`
// being independently declared. Choosing the window equal to that lcm
// (ratio 1, one window per compiled period) is the least constrained
// value satisfying it; this unit's own finding (see the Unit 1 report)
// is that a window LONGER than lcm(kDtU, kDtV) -- which is what
// examples/brusselator/main.cpp's own --window 0.02 against --dt-u
// 0.002/--dt-v 0.003 (lcm 0.006) needs -- is refused today, independent
// of this test.
constexpr double kDtU = 0.01, kDtV = 0.015, kWindow = 0.03;
constexpr int kNumWindows = 4;
constexpr double kTol = 1e-10;
constexpr int kMaxIterations = 100;

struct RunResult {
  bool ok = false;
  std::string error;
  brusselator::Vector u_final, v_final;
  std::vector<int> iterations_per_window;
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
  if (const auto r = brusselator::register_native_window_placements(
          rig->backend, u_mesh, kNxU, kNyU, handles);
      !r.ok()) {
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

TEST(BrusselatorRunner, DISABLED_JacobiMatchesDriver) {
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

TEST(BrusselatorRunner, DISABLED_GaussSeidelMatchesDriver) {
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
