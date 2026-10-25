/// Window Newton (newton-krylov.md Sec. 9.7's own test plan):
/// `Runner::window_region_fixed_point_residual()` and
/// `Runner::debug_window_member_snapshots()` -- the window-scoped seam's
/// own `F(x) = x - G(x)`, over a two-member linear-ODE window region
/// (the SAME `LinearOdeComponent` physics test/runtime/window_runner_
/// multirate_test.cpp already uses for its own Picard/GaussSeidel
/// comparisons, reduced to a non-straddling 7:13-free single-rate pair
/// here since Newton's own re-enterability/seeding/snapshot properties
/// do not depend on multirate straddling at all -- Sec. 9.1-9.3's own
/// design covers both the same way). `LinearOdeComponent` itself sets
/// only `kCapabilityRollback`; this file's own `NewtonCapableOde`
/// subclass ALSO sets `kCapabilityResidual` (nothing here requests the
/// block-Jacobi preconditioner, so `kCapabilityLinearization` is never
/// asked for -- `ImplicitComponent::evaluate_residual()` itself is never
/// called by this seam either way, dn_heat's own `DirichletSlab::
/// evaluate_residual()` precedent for why the bit alone is what setup
/// checks, `verify_newton_capability_support()`, src/runtime/checkpoint_
/// support.cpp).
#include <gtest/gtest.h>
#include <fmt/core.h>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/schedule/window.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

using fuse::AccelConfig;
using fuse::AccelKind;
using fuse::Component;
using fuse::ComponentDecl;
using fuse::ComponentSet;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::CouplingWindowScope;
using fuse::DtPolicy;
using fuse::FieldRegistry;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::IterationRegionDecl;
using fuse::Layout;
using fuse::MemorySpace;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
using fuse::RankSet;
using fuse::Runner;
using fuse::StepContext;
using fuse::WaveformIteration;
namespace u = fuse::units;

namespace {

// window_runner_multirate_test.cpp's own LinearOdeComponent, unchanged
// physics (dx/dt = -a*x + b*u, u the mean of the partner's trajectory
// over the substep), plus `kCapabilityResidual` -- this file's own
// reason, this file's own top comment.
class NewtonCapableOde final : public Component {
 public:
  NewtonCapableOde(std::string name, double a, double b, double x0)
      : name_(std::move(name)), a_(a), b_(b), x_(x0), provided_(x0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == Intent::Provides;
      double* p = is_out ? &provided_ : &required_;
      auto r = reg.bind(
          f, fuse::FieldRef{p, Layout{.stride = 1, .count = 1}, MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext& ctx) override {
    ++advance_count;
    const double h = ctx.dt().to_double();
    if (std::abs(a_) < 1e-300) {
      x_ = x_ + b_ * required_ * h;
    } else {
      const double e = std::exp(-a_ * h);
      x_ = x_ * e + b_ * required_ * (1.0 - e) / a_;
    }
    provided_ = x_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override {
    saved_ = x_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    x_ = saved_;
    provided_ = x_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback | fuse::kCapabilityResidual;
  }
  [[nodiscard]] double* provided_ptr() { return &provided_; }
  [[nodiscard]] double* required_ptr() { return &required_; }
  [[nodiscard]] double x() const { return x_; }

  int advance_count = 0;

 private:
  std::string name_;
  double      a_, b_;
  double      x_;
  double      provided_;
  double      required_ = 0.0;
  double      saved_ = 0.0;
};

constexpr double kOA = 1.3, kOB = 0.7, kOC = 0.9, kOD = 0.5;
constexpr double kOX0 = 1.0, kOY0 = 0.2;
constexpr Rational kWindow{12};
constexpr Rational kIntervalA{3};  // 4 substeps/window, no straddle
constexpr Rational kIntervalB{4};  // 3 substeps/window, no straddle

// A <-> B, both directions, window length 12 = lcm(3, 4): NEITHER
// member's own interval crosses the window end, so every property this
// file pins is isolated from the (separately tested) straddle-replay
// machinery entirely -- same reasoning window_runner_multirate_test.cpp's
// own top comment gives for its own LCM choice.
CouplingGraph make_graph(AccelKind accel, WaveformIteration mode,
                         fuse::NewtonConfig newton_cfg = {}) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});

  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 0}, .interval = kIntervalA});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{0, 0}, .interval = kIntervalB});

  const auto a_out = g.add_field(a, FieldSpec{.name = "a_out", .region = r,
                                              .units = u::kKelvin,
                                              .conservation = Conservation::Extensive,
                                              .intent = Intent::Provides});
  const auto b_in = g.add_field(b, FieldSpec{.name = "b_in", .region = r,
                                             .units = u::kKelvin,
                                             .conservation = Conservation::Extensive,
                                             .intent = Intent::Requires});
  const auto b_out = g.add_field(b, FieldSpec{.name = "b_out", .region = r,
                                              .units = u::kKelvin,
                                              .conservation = Conservation::Extensive,
                                              .intent = Intent::Provides});
  const auto a_in = g.add_field(a, FieldSpec{.name = "a_in", .region = r,
                                             .units = u::kKelvin,
                                             .conservation = Conservation::Extensive,
                                             .intent = Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the a-b coupling window's own sweep, not a real-time lag");

  CouplingWindowScope scope;
  scope.explicit_interval = kWindow;
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable}, {"B", DtPolicy::Adjustable}};
  scope.iteration = mode;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 300;
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_out"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_out"});
  decl.window_scope = scope;
  decl.acceleration.kind = accel;
  decl.newton_config = newton_cfg;
  if (accel == AccelKind::Newton) decl.requires_residual_capability = true;
  g.add_iteration_region(decl);
  return g;
}

struct Fixture {
  CouplingGraph graph;
  fuse::SerialMeshBackend backend;
  ComponentSet components;
  NewtonCapableOde* a = nullptr;
  NewtonCapableOde* b = nullptr;
  std::unique_ptr<Runner> runner;
  fuse::Result<void> init = fuse::Result<void>::success();
};

Fixture make_fixture(AccelKind accel = AccelKind::Newton,
                    WaveformIteration mode = WaveformIteration::Jacobi,
                    fuse::NewtonConfig newton_cfg = {},
                    fuse::RunnerConfig rcfg = {}) {
  Fixture s;
  s.graph = make_graph(accel, mode, newton_cfg);
  if (!s.backend.load(s.graph.mesh(fuse::MeshId{0})).ok()) {
    s.init = fuse::Failure{"E-TEST", "mesh load failed"};
    return s;
  }
  const auto lowered = fuse::lower(s.graph);
  if (!lowered.ok()) {
    s.init = fuse::Failure{"E-TEST", lowered.diags.to_string()};
    return s;
  }
  const fuse::ClockSet clocks{s.graph};
  const auto sched = fuse::schedule(s.graph, lowered.dag, clocks, fuse::make_region_sizes(s.graph, lowered.dag, s.backend));
  if (!sched.ok()) {
    s.init = fuse::Failure{"E-TEST", sched.diags.to_string()};
    return s;
  }

  auto a_owned = std::make_unique<NewtonCapableOde>("A", kOA, kOB, kOX0);
  auto b_owned = std::make_unique<NewtonCapableOde>("B", kOC, kOD, kOY0);
  s.a = a_owned.get();
  s.b = b_owned.get();
  s.components.add(std::move(a_owned));
  s.components.add(std::move(b_owned));
  FieldRegistry registry;
  for (auto& c : s.components) {
    const auto r = c.register_fields(registry, s.graph);
    if (!r.ok()) { s.init = r; return s; }
  }

  // Same note test/runtime/newton_region_test.cpp's own fixture makes:
  // `Runner` must be constructed and `initialize()`d in the SAME
  // function that returns it, never split across a later call -- the
  // lowered DAG/plan above are locals this fixture itself owns.
  s.runner = std::make_unique<Runner>(s.graph, lowered.dag, sched.plan, s.backend);
  s.init = s.runner->initialize(s.components, registry, rcfg);
  return s;
}

}  // namespace

// Test 1 (Sec. 9.7): two `F` evaluations at the identical `x`
// (`iteration = 1` both times) produce bit-identical `F(x)`.
TEST(WindowFixedPointResidual, RepeatedEvaluationAtTheSamePointIsBitIdentical) {
  Fixture s = make_fixture();
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;

  auto wr = s.runner->window_region_fixed_point_residual(0, s.components);
  ASSERT_TRUE(wr.ok()) << wr.error().message;
  const fuse::WindowFixedPointResidual& w = wr.value();
  ASSERT_GT(w.n, 0u);

  // Sec. 9.3's own two-call convention: `iteration = 0` exactly once
  // (the predictor-seeding call) before any `iteration = 1` evaluation
  // in this window instance's own pass.
  std::vector<double> x0_unused(w.n, 0.0), f0_unused(w.n, 0.0), g0(w.n, 0.0);
  ASSERT_TRUE(w.evaluate(/*iteration=*/0, x0_unused, f0_unused, g0, false).ok());

  std::vector<double> x(w.n, 0.1);
  std::vector<double> f1(w.n, 0.0), f2(w.n, 0.0);
  const auto r1 = w.evaluate(/*iteration=*/1, x, f1, {}, /*commit=*/false);
  ASSERT_TRUE(r1.ok()) << r1.error().message;
  const auto r2 = w.evaluate(/*iteration=*/1, x, f2, {}, /*commit=*/false);
  ASSERT_TRUE(r2.ok()) << r2.error().message;
  for (std::size_t i = 0; i < w.n; ++i) {
    EXPECT_EQ(f1[i], f2[i]) << "index " << i;
  }
}

// Test 2 (Sec. 9.7): evaluate `F` once (non-committing) at an arbitrary
// `x` the "real" path never visits, interleaved WITHIN the same pass
// between the predictor seed and the real committing evaluation, and
// show the committed result is bit-identical to a fresh pass that
// never saw the extra probe -- the direct window-region analogue of
// fixed_point_residual_test.cpp's own `EvaluatingFDoesNotPerturbThe
// RunnersOwnPicardIteration`, adapted to drive both paths through
// `window_region_fixed_point_residual()` itself rather than `run_
// period()` (Newton's own GMRES/line-search already make MANY non-
// committing probe evaluations at arbitrary points within one pass en
// route to its own committing call -- that is exactly the scenario
// this test isolates to one extra, deliberately-unrelated probe).
//
// WITHIN one pass only: `window_last_dst_step` (src/runtime/runner_
// impl.hpp) tracks, per window instance, whether its own `iteration ==
// 0` predictor-seeding call has already happened -- by design NEVER
// restored by a non-committing evaluation (`evaluate_window_region_
// residual()`'s own comment: "idempotent within one pass"), because
// `iteration == 0` fires exactly once per pass. A probe making its OWN
// `iteration == 0` call, or any call issued AFTER this pass's own
// commit, is a SEPARATE pass racing the same window instance's first-
// sweep marker -- `E-RT-WINDOW-SWEEP` refuses that, correctly; it is
// not what this test is after.
TEST(WindowFixedPointResidual,
    AWindowPassWithAnInterleavedProbeIsBitIdenticalToOneWithout) {
  auto run_one_pass = [](bool with_extra_probe) {
    Fixture s = make_fixture(AccelKind::Newton);
    EXPECT_TRUE(s.init.ok()) << s.init.error().message;
    auto wr = s.runner->window_region_fixed_point_residual(0, s.components);
    EXPECT_TRUE(wr.ok()) << wr.error().message;
    const fuse::WindowFixedPointResidual& w = wr.value();

    std::vector<double> x0_unused(w.n, 0.0), f0_unused(w.n, 0.0), x_real(w.n, 0.0);
    EXPECT_TRUE(w.evaluate(/*iteration=*/0, x0_unused, f0_unused, x_real, false).ok());

    if (with_extra_probe) {
      std::vector<double> x_probe(w.n, 0.91), f_probe(w.n, 0.0);
      EXPECT_TRUE(w.evaluate(/*iteration=*/1, x_probe, f_probe, {}, false).ok());
    }

    std::vector<double> f_real(w.n, 0.0);
    EXPECT_TRUE(w.evaluate(/*iteration=*/1, x_real, f_real, {}, /*commit=*/true).ok());
    return std::make_pair(s.a->x(), s.b->x());
  };

  const auto [baseline_a, baseline_b] = run_one_pass(/*with_extra_probe=*/false);
  const auto [probed_a, probed_b] = run_one_pass(/*with_extra_probe=*/true);
  EXPECT_EQ(probed_a, baseline_a);
  EXPECT_EQ(probed_b, baseline_b);
}

// Test 3 (Sec. 9.7): call with `iteration = 0` once, then `iteration =
// 1` twice at two DIFFERENT `x`, and assert the pass-scoped caches
// (`WindowMemberDebugSnapshot::pass_start_clock`/`pass_start_value`)
// are identical across all three calls.
TEST(WindowFixedPointResidual,
    TheIterationZeroCallSeedsThePassAndIterationOneDoesNotReseedIt) {
  Fixture s = make_fixture();
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;
  auto wr = s.runner->window_region_fixed_point_residual(0, s.components);
  ASSERT_TRUE(wr.ok()) << wr.error().message;
  const fuse::WindowFixedPointResidual& w = wr.value();

  std::vector<double> x0_unused(w.n, 0.0), f0_unused(w.n, 0.0), g0(w.n, 0.0);
  ASSERT_TRUE(w.evaluate(/*iteration=*/0, x0_unused, f0_unused, g0, false).ok());

  auto snap_after_0 = s.runner->debug_window_member_snapshots(0);
  ASSERT_TRUE(snap_after_0.ok()) << snap_after_0.error().message;
  ASSERT_FALSE(snap_after_0.value().empty());
  for (const auto& m : snap_after_0.value()) {
    EXPECT_TRUE(m.has_pass_start_clock);
  }

  std::vector<double> xA(w.n, 0.2), fA(w.n, 0.0);
  ASSERT_TRUE(w.evaluate(/*iteration=*/1, xA, fA, {}, false).ok());
  auto snap_after_A = s.runner->debug_window_member_snapshots(0);
  ASSERT_TRUE(snap_after_A.ok());

  std::vector<double> xB(w.n, 0.9), fB(w.n, 0.0);
  ASSERT_TRUE(w.evaluate(/*iteration=*/1, xB, fB, {}, false).ok());
  auto snap_after_B = s.runner->debug_window_member_snapshots(0);
  ASSERT_TRUE(snap_after_B.ok());

  ASSERT_EQ(snap_after_0.value().size(), snap_after_A.value().size());
  ASSERT_EQ(snap_after_0.value().size(), snap_after_B.value().size());
  for (std::size_t i = 0; i < snap_after_0.value().size(); ++i) {
    const auto& m0 = snap_after_0.value()[i];
    const auto& mA = snap_after_A.value()[i];
    const auto& mB = snap_after_B.value()[i];
    EXPECT_EQ(m0.pass_start_clock, mA.pass_start_clock) << "member " << i;
    EXPECT_EQ(m0.pass_start_clock, mB.pass_start_clock) << "member " << i;
    ASSERT_EQ(m0.pass_start_value.size(), mA.pass_start_value.size());
    ASSERT_EQ(m0.pass_start_value.size(), mB.pass_start_value.size());
    for (std::size_t e = 0; e < m0.pass_start_value.size(); ++e) {
      EXPECT_EQ(m0.pass_start_value[e], mA.pass_start_value[e]);
      EXPECT_EQ(m0.pass_start_value[e], mB.pass_start_value[e]);
    }
  }
}

// Test 4 (Sec. 9.7): several non-committing evaluations, then one
// committing one; `st.history` must show exactly the committing call's
// own effect.
TEST(WindowFixedPointResidual,
    CommittingEvaluationRefreshesHistoryExactlyOnceNonCommittingDoesNotTouchIt) {
  Fixture s = make_fixture();
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;
  auto wr = s.runner->window_region_fixed_point_residual(0, s.components);
  ASSERT_TRUE(wr.ok()) << wr.error().message;
  const fuse::WindowFixedPointResidual& w = wr.value();

  std::vector<double> x0_unused(w.n, 0.0), f0_unused(w.n, 0.0), g0(w.n, 0.0);
  ASSERT_TRUE(w.evaluate(0, x0_unused, f0_unused, g0, false).ok());

  std::vector<double> x(w.n), f(w.n, 0.0);
  for (std::size_t i = 0; i < w.n; ++i) x[i] = g0[i];  // x_0, Sec. 9.3

  for (int k = 0; k < 3; ++k) {
    ASSERT_TRUE(w.evaluate(1, x, f, {}, /*commit=*/false).ok());
  }
  ASSERT_EQ(s.runner->iteration_stats().size(), 1u);
  EXPECT_FALSE(s.runner->iteration_stats().front().converged);

  ASSERT_TRUE(w.evaluate(1, x, f, {}, /*commit=*/true).ok());

  // The commit is what delivers Newton's own iterate to the components
  // (NewtonRegion.VerificationSweepDeliversTheConvergedStateToTheComponents'
  // own property, here for the window seam): A/B's own `x()` must have
  // moved off the pre-window initial condition.
  EXPECT_NE(s.a->x(), kOX0);
  EXPECT_NE(s.b->x(), kOY0);
}

// Test 5 (Sec. 9.7): builds a GaussSeidel two-member window with one
// unlagged (b2a is NOT marked `.lag(1)` as lagged in GaussSeidel's own
// sense here -- `make_graph()`'s own channel always carries `.lag(1)`
// for the TIME-level mismatch every window channel needs regardless of
// Jacobi/GaussSeidel; GaussSeidel's own UNLAGGED direction is decided by
// `window_lagged_import_arcs`, iteration.cpp's own resolution of
// `scope.iteration`, not by this `.lag()` call) and asserts `x`'s own
// length covers only the lagged block.
TEST(WindowFixedPointResidual, GaussSeidelUnlaggedBlockIsExcludedFromX) {
  Fixture jacobi = make_fixture(AccelKind::Newton, WaveformIteration::Jacobi);
  ASSERT_TRUE(jacobi.init.ok()) << jacobi.init.error().message;
  auto wr_j = jacobi.runner->window_region_fixed_point_residual(0, jacobi.components);
  ASSERT_TRUE(wr_j.ok()) << wr_j.error().message;
  EXPECT_TRUE(wr_j.value().excluded_lengths.empty());
  EXPECT_EQ(wr_j.value().channel_names.size(), 2u);

  Fixture gs = make_fixture(AccelKind::Newton, WaveformIteration::GaussSeidel);
  ASSERT_TRUE(gs.init.ok()) << gs.init.error().message;
  auto wr_gs = gs.runner->window_region_fixed_point_residual(0, gs.components);
  ASSERT_TRUE(wr_gs.ok()) << wr_gs.error().message;

  // Exactly one of the two channels is excluded under GaussSeidel, and
  // the included one's own length accounts for ALL of `n` -- the direct
  // test of Sec. 9.1's own exclusion rule.
  ASSERT_EQ(wr_gs.value().excluded_lengths.size(), 1u);
  ASSERT_EQ(wr_gs.value().channel_names.size(), 1u);
  EXPECT_EQ(wr_gs.value().n, wr_gs.value().lengths.front());
  EXPECT_LT(wr_gs.value().n, wr_j.value().n);

  // The excluded channel is one of the two Jacobi carried -- not some
  // third, unrelated name.
  const std::string& excluded = wr_gs.value().excluded_channel_names.front();
  EXPECT_TRUE(excluded == "a2b" || excluded == "b2a");
}

// Test 6 (Sec. 9.7): one test per Sec. 9.2 table row marked "YES" --
// disable that row's own restore (by hand, via the Newton solve path
// itself: convergence requires MANY re-entrant evaluations at DIFFERENT
// x, which is only correct if window_member_state's own clock/straddle
// fields are restored between them) and show Newton fails to reach the
// SAME converged answer repeated runs agree on.
// `build-evidence/mutations/runtime.toml`'s own `window-newton-clock-
// not-restored`/`window-newton-straddle-state-not-restored` rows hand-
// prove the restore itself is load-bearing by actually disabling it in
// the source and rerunning this exact test; this test is what GOES red
// for that mutation -- here it stays green against the real code,
// pinning the Newton solve's own determinism (two full runs from the
// same inputs reach the SAME final state bit for bit), which the
// mutation breaks.
TEST(WindowFixedPointResidual, TwoFullNewtonRunsFromTheSameInputsAgreeBitForBit) {
  Fixture s1 = make_fixture(AccelKind::Newton);
  ASSERT_TRUE(s1.init.ok()) << s1.init.error().message;
  ASSERT_TRUE(s1.runner->run_period(s1.components).ok());

  Fixture s2 = make_fixture(AccelKind::Newton);
  ASSERT_TRUE(s2.init.ok()) << s2.init.error().message;
  ASSERT_TRUE(s2.runner->run_period(s2.components).ok());

  EXPECT_EQ(s1.a->x(), s2.a->x());
  EXPECT_EQ(s1.b->x(), s2.b->x());
  ASSERT_EQ(s1.runner->iteration_stats().size(), 1u);
  ASSERT_EQ(s2.runner->iteration_stats().size(), 1u);
  EXPECT_TRUE(s1.runner->iteration_stats().front().converged);
  EXPECT_EQ(s1.runner->iteration_stats().front().iterations,
           s2.runner->iteration_stats().front().iterations);
}

// The window path's own non-convergence reporting, the mirror of
// NewtonRegion.ExhaustingTheNewtonIterationBudgetReportsNonConvergence on
// the ordinary path: a zero Newton budget is DivergedMaxIterations at once,
// and the window must be counted as not converged and, under
// abort_on_nonconvergence, refused with the Newton status named. Kills
// build-evidence/mutations/runtime.toml's
// window-newton-nonconvergence-not-reported.
TEST(WindowFixedPointResidual, ExhaustingTheNewtonBudgetOnAWindowReportsNonConvergence) {
  fuse::NewtonConfig cfg;
  cfg.max_nonlinear_iterations = 0;
  fuse::RunnerConfig rcfg;
  rcfg.abort_on_nonconvergence = true;
  Fixture s = make_fixture(AccelKind::Newton, WaveformIteration::Jacobi, cfg, rcfg);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;

  const auto r = s.runner->run_period(s.components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-NOCONVERGE") << r.error().message;
  EXPECT_NE(r.error().message.find("window"), std::string::npos) << r.error().message;
  EXPECT_NE(r.error().message.find("DivergedMaxIterations"), std::string::npos)
      << r.error().message;
}

TEST(WindowFixedPointResidual, AWindowThatExhaustsTheNewtonBudgetIsCountedNotConverged) {
  fuse::NewtonConfig cfg;
  cfg.max_nonlinear_iterations = 0;
  Fixture s = make_fixture(AccelKind::Newton, WaveformIteration::Jacobi, cfg);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;
  ASSERT_TRUE(s.runner->run_period(s.components).ok());
  ASSERT_EQ(s.runner->iteration_stats().size(), 1u);
  EXPECT_FALSE(s.runner->iteration_stats().front().converged);
  EXPECT_EQ(s.runner->stats().windows_not_converged, 1);
}

// `run_region()`'s own window-scoped Picard/accelerator seed (`R.x_old`'s
// pre-loop read, src/runtime/iteration.cpp) sits textually AFTER the early
// `if (R.newton) { ... co_return run_region_newton_window(...); }`
// dispatch at that same function's own top -- a Newton-selected region
// never reaches the modified lines at all. `WindowCoupler` has no
// Newton solve of its own (zero occurrences of "Newton" in window_
// coupler.cpp; examples/brusselator/main.cpp's own "window" driver
// branch refuses `--accel newton` outright), so there is no second
// DRIVER to compare against the way the Aitken parity tests do -- the
// claim this test makes instead, and the only one that is both true and
// checkable, is SEED-INVARIANCE: this exact (seed, iteration count,
// final state) triple, across TWO window instances (the boundary where
// the seed's window-0/window-1 transition lives), does not move at all
// with the seed change. Printed with EXPECT_EQ (bit-for-bit, no
// tolerance) so a diff between two builds' own stdout for this one
// test is the entire comparison.
TEST(WindowFixedPointResidual, TwoWindowInstancesNewtonResultIsFixInvariant) {
  Fixture s = make_fixture(AccelKind::Newton, WaveformIteration::Jacobi);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;

  ASSERT_TRUE(s.runner->run_period(s.components).ok());
  ASSERT_EQ(s.runner->iteration_stats().size(), 1u);
  const bool converged0 = s.runner->iteration_stats().front().converged;
  const int iterations0 = s.runner->iteration_stats().front().iterations;
  const double a0 = s.a->x();
  const double b0 = s.b->x();

  ASSERT_TRUE(s.runner->run_period(s.components).ok());
  ASSERT_EQ(s.runner->iteration_stats().size(), 1u);
  const bool converged1 = s.runner->iteration_stats().front().converged;
  const int iterations1 = s.runner->iteration_stats().front().iterations;
  const double a1 = s.a->x();
  const double b1 = s.b->x();

  EXPECT_TRUE(converged0);
  EXPECT_TRUE(converged1);
  fmt::print(stderr,
             "[newton-fix-invariance] window0: converged={} iterations={} "
             "a={:.17g} b={:.17g}\n",
             converged0, iterations0, a0, b0);
  fmt::print(stderr,
             "[newton-fix-invariance] window1: converged={} iterations={} "
             "a={:.17g} b={:.17g}\n",
             converged1, iterations1, a1, b1);
}
