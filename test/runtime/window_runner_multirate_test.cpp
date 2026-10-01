// Time windows track (C3): NOT the acceptance matrix -- the user rejected
// LCM coupling windows outright, in favour of the temporal interpolant
// this whole track implements, so `explicit_interval = lcm(m, n)` (below)
// is a design this codebase does not offer as the real multirate case.
// For a 7:13 pair, the window is one member's own step (13 or 7) and the
// OTHER member's steps cross the window end -- that is what the straddle
// replay (`fuse::detail::advance_straddle_split()`, not yet wired into
// advance_window()) exists for; the real acceptance matrix belongs in a
// separate file, once that replay lands.
//
// This file exists for a narrower, still useful reason: it is what
// diagnosed, and now regression-guards, two bugs the cold-start fix and
// the window_component scheduling fix (both items in the lead's own
// five-item list) needed something to verify against -- an LCM window
// happens to make BOTH members' own intervals divide the window evenly
// (91 = lcm(7,13): 91/13 = 7, 91/7 = 13), so it schedules and runs with
// NO straddle involved, isolating the cold-start/query-range bugs from
// the (separate, still unimplemented) straddle-replay work entirely.
// Once the real window_component-based 13:7-with-a-crossing-member
// matrix exists and passes, this file's own job is done; until then it
// stays, explicitly labelled as the LCM form item 4's own allowance
// permits as ONE extra test ("An LCM-window case may remain as ONE extra
// test, labelled as such") but excludes from the acceptance set.

#include <gtest/gtest.h>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/runtime/window_coupler.hpp>
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
using fuse::FieldId;
using fuse::FieldRef;
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
using fuse::Result;
using fuse::Runner;
using fuse::StepContext;
using fuse::WaveformIteration;
using fuse::WindowChannel;
using fuse::WindowCoupler;
using fuse::WindowCouplerConfig;
using fuse::WindowMember;
namespace u = fuse::units;

namespace {

// dx/dt = -a*x + b*u, u the MEAN of the partner's trajectory over the
// substep -- window_coupler_test.cpp's own LinearOdeComponent, unchanged,
// rebound through FieldRegistry (Runner components resolve storage that
// way; WindowCoupler binds WindowMember::provided/required directly) so
// the SAME closed-form-exact-given-a-mean physics runs on both sides of
// every comparison in this file.
class LinearOdeComponent final : public Component {
 public:
  LinearOdeComponent(std::string name, double a, double b, double x0)
      : name_(std::move(name)), a_(a), b_(b), x_(x0), provided_(x0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  Result<void> register_fields(FieldRegistry& reg,
                                const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == Intent::Provides;
      double* p = is_out ? &provided_ : &required_;
      auto r = reg.bind(
          f, FieldRef{p, Layout{.stride = 1, .count = 1}, MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return Result<void>::success();
  }

  Result<void> advance(const StepContext& ctx) override {
    ++advance_count;
    const double h = ctx.dt().to_double();
    if (std::abs(a_) < 1e-300) {
      x_ = x_ + b_ * required_ * h;
    } else {
      const double e = std::exp(-a_ * h);
      x_ = x_ * e + b_ * required_ * (1.0 - e) / a_;
    }
    provided_ = x_;
    return Result<void>::success();
  }
  Result<void> save_state() override {
    saved_ = x_;
    return Result<void>::success();
  }
  Result<void> restore_state() override {
    x_ = saved_;
    provided_ = x_;
    return Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
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

constexpr double kAA = 1.3, kAB = 0.7, kAC = 0.9, kAD = 0.5;
constexpr double kAX0 = 1.0, kAY0 = 0.2;

// A <-> B, both directions through the FULL compiled graph
// (lower()+schedule()+Runner), window length 91 = lcm(13, 7), A's own
// interval 13, B's own interval 7. `swap_declaration_order`: whether B's
// ComponentDecl/fields are added to the graph BEFORE A's -- Gauss-
// Seidel's own sweep order must come from topological_sweep_order()
// (the unlagged channel's own direction), never from declaration order
// (window_coupler.cpp's own rule, pinned for WindowCoupler directly by
// window_coupler_test.cpp's GaussSeidelResultIsIndependentOfDeclaration
// Order; this is the SAME property, through the Runner).
CouplingGraph make_multirate_graph(WaveformIteration mode, DtPolicy pol_a,
                                   DtPolicy pol_b, AccelKind accel,
                                   bool swap_declaration_order = false) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});

  fuse::ComponentId a, b;
  if (swap_declaration_order) {
    b = g.add_component(ComponentDecl{
        .name = "B", .ranks = RankSet{0, 0}, .interval = Rational{7}});
    a = g.add_component(ComponentDecl{
        .name = "A", .ranks = RankSet{0, 0}, .interval = Rational{13}});
  } else {
    a = g.add_component(ComponentDecl{
        .name = "A", .ranks = RankSet{0, 0}, .interval = Rational{13}});
    b = g.add_component(ComponentDecl{
        .name = "B", .ranks = RankSet{0, 0}, .interval = Rational{7}});
  }

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
  scope.explicit_interval = Rational{91};
  scope.order = 2;
  scope.policies = {{"A", pol_a}, {"B", pol_b}};
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
  g.add_iteration_region(decl);
  return g;
}

struct RunnerResult {
  bool   ok = false;
  bool   converged = false;
  int    iterations = 0;
  double xA = 0.0, xB = 0.0;
  int    advance_count_a = 0, advance_count_b = 0;
  std::string error;
};

RunnerResult run_via_runner(WaveformIteration mode, DtPolicy pol_a,
                            DtPolicy pol_b, AccelKind accel,
                            bool swap_declaration_order = false) {
  RunnerResult out;
  const CouplingGraph g = make_multirate_graph(mode, pol_a, pol_b, accel,
                                               swap_declaration_order);

  const auto lowered = lower(g);
  if (!lowered.ok()) { out.error = lowered.diags.to_string(); return out; }

  fuse::SerialMeshBackend backend;
  if (!backend.load(g.mesh(fuse::MeshId{0})).ok()) {
    out.error = "mesh load failed";
    return out;
  }
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  if (!sched.ok()) { out.error = sched.diags.to_string(); return out; }

  auto a_owned = std::make_unique<LinearOdeComponent>("A", kAA, kAB, kAX0);
  auto b_owned = std::make_unique<LinearOdeComponent>("B", kAC, kAD, kAY0);
  LinearOdeComponent* a = a_owned.get();
  LinearOdeComponent* b = b_owned.get();
  ComponentSet components;
  components.add(std::move(a_owned));
  components.add(std::move(b_owned));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    if (!r.ok()) { out.error = r.error().message; return out; }
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  if (!init.ok()) { out.error = init.error().message; return out; }

  const auto ran = runner.run_period(components);
  if (!ran.ok()) { out.error = ran.error().message; return out; }

  if (runner.iteration_stats().empty()) {
    out.error = "no iteration stats recorded";
    return out;
  }
  const fuse::IterationStats& st = runner.iteration_stats().front();
  out.ok = true;
  out.converged = st.converged;
  out.iterations = st.iterations;
  out.xA = a->x();
  out.xB = b->x();
  out.advance_count_a = a->advance_count;
  out.advance_count_b = b->advance_count;
  return out;
}

// The SAME physics/interval/policy/accel/order run directly through the
// standalone WindowCoupler, one window of length 91 -- the reference this
// file's Runner-side result must reproduce.
RunnerResult run_via_driver(WaveformIteration mode, DtPolicy pol_a,
                            DtPolicy pol_b, AccelKind accel,
                            bool swap_declaration_order = false) {
  RunnerResult out;
  auto comp_a = std::make_unique<LinearOdeComponent>("A", kAA, kAB, kAX0);
  auto comp_b = std::make_unique<LinearOdeComponent>("B", kAC, kAD, kAY0);

  WindowMember mA;
  mA.component = comp_a.get();
  mA.name = "A";
  mA.interval = Rational{13};
  mA.policy = pol_a;
  mA.provided = comp_a->provided_ptr();
  mA.required = comp_a->required_ptr();
  mA.length = 1;

  WindowMember mB;
  mB.component = comp_b.get();
  mB.name = "B";
  mB.interval = Rational{7};
  mB.policy = pol_b;
  mB.provided = comp_b->provided_ptr();
  mB.required = comp_b->required_ptr();
  mB.length = 1;

  std::vector<WindowMember> members;
  std::size_t a_idx, b_idx;
  if (swap_declaration_order) {
    members.push_back(std::move(mB));
    members.push_back(std::move(mA));
    b_idx = 0; a_idx = 1;
  } else {
    members.push_back(std::move(mA));
    members.push_back(std::move(mB));
    a_idx = 0; b_idx = 1;
  }

  const std::vector<WindowChannel> channels = {
      {.provider = a_idx, .consumer = b_idx, .lagged = false, .name = "a_out"},
      {.provider = b_idx, .consumer = a_idx, .lagged = true, .name = "b_out"}};

  WindowCouplerConfig cfg;
  cfg.window_length = Rational{91};
  cfg.max_iterations = 300;
  cfg.acceleration = AccelConfig{.kind = accel};
  cfg.order = 2;
  cfg.waveform_capacity = 4096;
  cfg.iteration = mode;
  // Must match the Runner side's own decl.convergence exactly (both
  // Absolute, 1e-12): left empty, WindowCoupler::run_window() falls back
  // to `resid_norm <= 1e-10 * max(1.0, ref_norm)` (window_coupler.cpp) --
  // a DIFFERENT threshold that stops at a different sweep, which looked
  // like a Runner-vs-driver algorithmic discrepancy the first time this
  // file was written but was actually just an unmatched test fixture.
  cfg.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_out"});
  cfg.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_out"});

  Result<std::unique_ptr<WindowCoupler>> created =
      WindowCoupler::create(members, channels, cfg);
  if (!created.ok()) { out.error = created.error().message; return out; }
  std::unique_ptr<WindowCoupler> coupler = std::move(created.value());

  const Result<void> r = coupler->run_window();
  if (!r.ok()) { out.error = r.error().message; return out; }

  out.ok = true;
  out.converged = true;  // run_window() itself fails (E-RT-WINDOW-NOCONVERGE) otherwise
  out.iterations = coupler->stats().iterations;
  out.xA = comp_a->x();
  out.xB = comp_b->x();
  out.advance_count_a = comp_a->advance_count;
  out.advance_count_b = comp_b->advance_count;
  return out;
}

}  // namespace

// Identity-weight (matching-mesh: both members bind on the same 1-element
// region, so the channel's own remap is the identity) case, Adjustable/
// Adjustable, AccelKind::None -- the baseline every other case in this
// file is a variation of.
TEST(WindowRunnerMultirate, SevenThirteenAdjustableAdjustableJacobiMatchesDriver) {
  const RunnerResult via_runner = run_via_runner(
      WaveformIteration::Jacobi, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None);
  ASSERT_TRUE(via_runner.ok) << via_runner.error;
  EXPECT_TRUE(via_runner.converged);

  const RunnerResult via_driver = run_via_driver(
      WaveformIteration::Jacobi, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None);
  ASSERT_TRUE(via_driver.ok) << via_driver.error;

  EXPECT_EQ(via_runner.xA, via_driver.xA);
  EXPECT_EQ(via_runner.xB, via_driver.xB);
  EXPECT_EQ(via_runner.iterations, via_driver.iterations);
  EXPECT_EQ(via_runner.advance_count_a, via_driver.advance_count_a);
  EXPECT_EQ(via_runner.advance_count_b, via_driver.advance_count_b);
  // Every substep of every sweep actually ran (plan_substeps() gives A 7
  // substeps, B 13, per window/sweep): a per-iterate count, not just "ran".
  EXPECT_EQ(via_runner.advance_count_a, 7 * via_runner.iterations);
  EXPECT_EQ(via_runner.advance_count_b, 13 * via_runner.iterations);
}

// BOTH straddle policies (item 4's own wording): Fixed/Fixed here never
// actually straddles -- 91 is a multiple of both 13 and 7, exactly the
// "a Fixed step that divides the window never straddles" case
// window_coupler_test.cpp's own AFixedStepThatDividesTheWindowNever
// StraddlesAndIsAccepted pins for the driver directly. Confirms a
// straddle-FREE Fixed member matches the driver with no straddle
// machinery (resolve_pending_window_straddles(), iteration.cpp)
// involved at all -- it is specifically the crossing, not the policy
// label, that would engage it (test/runtime/window_runner_straddle_
// test.cpp's own tests are where it actually fires).
TEST(WindowRunnerMultirate, SevenThirteenFixedFixedJacobiMatchesDriver) {
  const RunnerResult via_runner = run_via_runner(
      WaveformIteration::Jacobi, DtPolicy::Fixed, DtPolicy::Fixed,
      AccelKind::None);
  ASSERT_TRUE(via_runner.ok) << via_runner.error;
  EXPECT_TRUE(via_runner.converged);

  const RunnerResult via_driver = run_via_driver(
      WaveformIteration::Jacobi, DtPolicy::Fixed, DtPolicy::Fixed,
      AccelKind::None);
  ASSERT_TRUE(via_driver.ok) << via_driver.error;

  EXPECT_EQ(via_runner.xA, via_driver.xA);
  EXPECT_EQ(via_runner.xB, via_driver.xB);
  EXPECT_EQ(via_runner.iterations, via_driver.iterations);
  EXPECT_EQ(via_runner.advance_count_a, via_driver.advance_count_a);
  EXPECT_EQ(via_runner.advance_count_b, via_driver.advance_count_b);
}

// Acceleration None (above) AND the default accelerator (AccelKind::
// Constant) -- item 4's own wording -- Jacobi only: GaussSeidel rejects
// any accelerator but None (E-RT-WINDOW-ACCEL) until R.blocks excludes
// the unlagged transfer's own residual, so this is the one mode that CAN
// exercise a real accelerator today.
TEST(WindowRunnerMultirate, SevenThirteenJacobiWithDefaultAcceleratorMatchesDriver) {
  const RunnerResult via_runner = run_via_runner(
      WaveformIteration::Jacobi, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::Constant);
  ASSERT_TRUE(via_runner.ok) << via_runner.error;
  EXPECT_TRUE(via_runner.converged);

  const RunnerResult via_driver = run_via_driver(
      WaveformIteration::Jacobi, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::Constant);
  ASSERT_TRUE(via_driver.ok) << via_driver.error;

  EXPECT_EQ(via_runner.xA, via_driver.xA);
  EXPECT_EQ(via_runner.xB, via_driver.xB);
  EXPECT_EQ(via_runner.iterations, via_driver.iterations);
  EXPECT_EQ(via_runner.advance_count_a, via_driver.advance_count_a);
  EXPECT_EQ(via_runner.advance_count_b, via_driver.advance_count_b);
  // Constant relaxation (omega=0.5, AccelConfig's own default) changes the
  // PATH, not the fixed point: confirm it actually took a DIFFERENT
  // number of sweeps than the unaccelerated case above, i.e. this test is
  // exercising the accelerator, not silently ignoring `accel`.
  const RunnerResult unaccelerated = run_via_runner(
      WaveformIteration::Jacobi, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None);
  ASSERT_TRUE(unaccelerated.ok) << unaccelerated.error;
  EXPECT_NE(via_runner.iterations, unaccelerated.iterations);
}

// GaussSeidel, AccelKind::None (the only accelerator it may use today --
// E-RT-WINDOW-ACCEL rejects the rest), EXPECT_EQ against the driver --
// item 4's "in BOTH modes".
TEST(WindowRunnerMultirate, SevenThirteenGaussSeidelMatchesDriver) {
  const RunnerResult via_runner = run_via_runner(
      WaveformIteration::GaussSeidel, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None);
  ASSERT_TRUE(via_runner.ok) << via_runner.error;
  EXPECT_TRUE(via_runner.converged);

  const RunnerResult via_driver = run_via_driver(
      WaveformIteration::GaussSeidel, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None);
  ASSERT_TRUE(via_driver.ok) << via_driver.error;

  EXPECT_EQ(via_runner.xA, via_driver.xA);
  EXPECT_EQ(via_runner.xB, via_driver.xB);
  EXPECT_EQ(via_runner.iterations, via_driver.iterations);
  EXPECT_EQ(via_runner.advance_count_a, via_driver.advance_count_a);
  EXPECT_EQ(via_runner.advance_count_b, via_driver.advance_count_b);
  // GaussSeidel's own defining property against Jacobi on this same
  // problem (window_coupler_test.cpp's GaussSeidelConvergesInFewer
  // IterationsOnThisProblem): fewer sweeps to converge, through the
  // Runner too.
  const RunnerResult jacobi = run_via_runner(
      WaveformIteration::Jacobi, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None);
  ASSERT_TRUE(jacobi.ok) << jacobi.error;
  EXPECT_LT(via_runner.iterations, jacobi.iterations);
}

// GaussSeidel declared in NON-default (non-topological) member order --
// item 4's own wording. WindowCoupler::create() (and, through the Runner,
// the compiled schedule's own topological_sweep_order()) must derive the
// sweep order from the unlagged channel's own direction, never from
// declaration order (C2's own GaussSeidelResultIsIndependentOfDeclaration
// Order, for WindowCoupler directly) -- this pins the SAME property for
// the compiled graph: `swap_declaration_order=true` declares B's
// ComponentDecl before A's, and the CouplingGraph's own field/channel
// wiring (find_component() by name, not by declaration index) must still
// place A's unlagged advance before B's.
TEST(WindowRunnerMultirate, GaussSeidelResultIsIndependentOfDeclarationOrderThroughRunner) {
  const RunnerResult a_first = run_via_runner(
      WaveformIteration::GaussSeidel, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None, /*swap_declaration_order=*/false);
  ASSERT_TRUE(a_first.ok) << a_first.error;

  const RunnerResult b_first = run_via_runner(
      WaveformIteration::GaussSeidel, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None, /*swap_declaration_order=*/true);
  ASSERT_TRUE(b_first.ok) << b_first.error;

  EXPECT_EQ(a_first.xA, b_first.xA);
  EXPECT_EQ(a_first.xB, b_first.xB);
  EXPECT_EQ(a_first.iterations, b_first.iterations);
  EXPECT_EQ(a_first.advance_count_a, b_first.advance_count_a);
  EXPECT_EQ(a_first.advance_count_b, b_first.advance_count_b);

  // And the declaration-order-swapped driver run agrees with both, so the
  // "order doesn't matter" property and the "Runner matches driver"
  // property are checked together, not independently.
  const RunnerResult driver_b_first = run_via_driver(
      WaveformIteration::GaussSeidel, DtPolicy::Adjustable, DtPolicy::Adjustable,
      AccelKind::None, /*swap_declaration_order=*/true);
  ASSERT_TRUE(driver_b_first.ok) << driver_b_first.error;
  EXPECT_EQ(b_first.xA, driver_b_first.xA);
  EXPECT_EQ(b_first.xB, driver_b_first.xB);
}