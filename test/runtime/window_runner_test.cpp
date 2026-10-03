// Time windows track (C3): the FIRST end-to-end proof that a window-scoped
// iteration region actually EXECUTES through the compiled dataflow graph +
// schedule() + Runner -- not just compiles (test/schedule/
// window_region_golden_test.cpp) or runs through the standalone
// WindowCoupler driver directly (test/runtime/window_coupler_test.cpp).
//
// Deliberately the SIMPLEST possible window graph: one element, one mesh
// shared by both members (so Remap is the identity -- nothing here
// exercises a real spatial transfer yet), interval_A = interval_B =
// window_length = 1 (no multirate subcycling within the window itself --
// exactly one substep per member per window). This is the smoke test for
// the wiring itself (does a WindowAdvance instruction actually run, does
// the per-window checkpoint/rollback/residual/converge machinery
// generically reused from an ordinary iteration region behave, is the
// K-wide field-width plumbing self-consistent end to end); a genuinely
// multirate (7:13-style) comparison against the direct WindowCoupler
// driver is the next file's job.

#include <gtest/gtest.h>
#include <fuse/backend/serial_backend.hpp>

#include <cmath>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/schedule/window.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

using fuse::ComponentDecl;
using fuse::ComponentSet;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::CouplingWindowScope;
using fuse::DtPolicy;
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
using fuse::Runner;
using fuse::WaveformIteration;
namespace u = fuse::units;

namespace {

// out' = a + k * mean(required-over-the-substep) -- a linear map, so the
// converged fixed point of the two-member coupling is exactly solvable by
// hand (the SAME closed form test/verification/iteration_test.cpp's
// LinearBlock supports for an ordinary, non-window region).
class LinearBlock final : public fuse::Component {
 public:
  LinearBlock(std::string name, double a, double k)
      : name_(std::move(name)), a_(a), k_(k) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == Intent::Provides;
      double* p = is_out ? &out_ : &in_;
      auto r = reg.bind(
          f, FieldRef{p, Layout{.stride = 1, .count = 1}, MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& /*ctx*/) override {
    ++advance_count;
    out_ = a_ + k_ * in_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override {
    saved_out_ = out_;
    saved_in_ = in_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    out_ = saved_out_;
    in_ = saved_in_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }
  [[nodiscard]] double out() const { return out_; }
  [[nodiscard]] double in() const { return in_; }

  int advance_count = 0;

 private:
  std::string name_;
  double      a_, k_;
  double      out_ = 0.0, in_ = 0.0, saved_out_ = 0.0, saved_in_ = 0.0;
};

CouplingGraph make_single_window_graph(
    WaveformIteration mode, fuse::AccelKind accel = fuse::AccelKind::None) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});

  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 0}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{0, 0}, .interval = Rational{1}});

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
  scope.window_component = "A";
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable}, {"B", DtPolicy::Adjustable}};
  scope.iteration = mode;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 50;
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_out"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_out"});
  decl.window_scope = scope;
  decl.acceleration.kind = accel;
  g.add_iteration_region(decl);
  return g;
}

// Same two-member shape and SAME two component intervals (both 1, the
// same clock the scheduler already accepts in
// make_single_window_graph()) -- ONLY the window's own length changes,
// via `explicit_interval` rather than `window_component`, to 1/2. A's
// own single substep is truncated onto the window end (DtPolicy::
// Adjustable, no straddle); B, DtPolicy::Fixed with the SAME interval
// (1) against a window half as long, takes its one substep in full,
// landing at 1 -- past the window end (1/2) -- a "straddle"
// (plan_substeps()'s own contract, WindowSubstep::straddle). Jacobi and
// AccelKind::None so the ONLY thing this graph can fail on is the
// straddle rejection itself, not the (separately tested) GaussSeidel-
// accel one.
CouplingGraph make_straddle_window_graph() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});

  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 0}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{0, 0}, .interval = Rational{1}});

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
  scope.explicit_interval = Rational{1, 2};
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable}, {"B", DtPolicy::Fixed}};
  scope.iteration = WaveformIteration::Jacobi;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 50;
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_out"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_out"});
  decl.window_scope = scope;
  decl.acceleration.kind = fuse::AccelKind::None;
  g.add_iteration_region(decl);
  return g;
}

// B's own `b_out` -- the straddling-provider channel's own source, fed
// to A's `a_in` -- lives on a SECOND mesh/region with a different cell
// count (2, against `r`'s own 1), so the channel needs a genuine
// (non-identity) Remap: `resolve_remap()`'s own "proved-coincident"
// check (fuse/runtime/runner_impl.hpp's `Resolved::remap` comment)
// cannot pass for two regions whose own entity counts differ. Otherwise
// identical to `make_straddle_window_graph()` -- same window shape
// (explicit_interval = 1/2, B interval 1, Fixed, straddles every
// window), same Jacobi/no-accelerator choice, so this is the decisive
// test for E-RT-WINDOW-STRADDLE-REPUBLISH alone, not conflated with
// the other rejections make_straddle_window_graph()'s own tests check.
// The SAME straddling shape as make_straddle_window_graph(), identity
// remap (one shared mesh/region, so the OTHER half of E-RT-WINDOW-
// STRADDLE-REPUBLISH's own OR condition is isolated from the remap
// one above), but A and B are declared on DIFFERENT, non-overlapping
// ranks (`RankSet::same_as()` false) -- the "nominal placement, one
// real process" pattern test/runtime/runner_test.cpp's own class
// comment on `dst_rank_count` documents: no `RunnerConfig::
// base_communicator` is configured anywhere in this file, so this is
// a legitimate single-process test of a graph-level placement
// mismatch, not an MPI test.
CouplingGraph make_straddle_window_graph_with_rank_mismatch() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});

  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{1, 1}, .interval = Rational{1}});

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
  scope.explicit_interval = Rational{1, 2};
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable}, {"B", DtPolicy::Fixed}};
  scope.iteration = WaveformIteration::Jacobi;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 50;
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_out"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_out"});
  decl.window_scope = scope;
  decl.acceleration.kind = fuse::AccelKind::None;
  g.add_iteration_region(decl);
  return g;
}

CouplingGraph make_straddle_window_graph_with_remap() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});
  const auto m2 = g.add_mesh(MeshDecl{.name = "m2", .source = "generated:line:2"});
  const auto r2 = g.add_region(Region{.name = "r2", .mesh = m2, .dimension = 1});

  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 0}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{0, 0}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, FieldSpec{.name = "a_out", .region = r,
                                              .units = u::kKelvin,
                                              .conservation = Conservation::Extensive,
                                              .intent = Intent::Provides});
  const auto b_in = g.add_field(b, FieldSpec{.name = "b_in", .region = r,
                                             .units = u::kKelvin,
                                             .conservation = Conservation::Extensive,
                                             .intent = Intent::Requires});
  const auto b_out = g.add_field(b, FieldSpec{.name = "b_out", .region = r2,
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
  scope.explicit_interval = Rational{1, 2};
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable}, {"B", DtPolicy::Fixed}};
  scope.iteration = WaveformIteration::Jacobi;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 50;
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_out"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_out"});
  decl.window_scope = scope;
  decl.acceleration.kind = fuse::AccelKind::None;
  g.add_iteration_region(decl);
  return g;
}

// B's own field sizes differ (`b_in`, 1 element, on `r`; `b_out`, 2
// elements, on `r2`) -- `LinearBlock` above always binds `count = 1`
// for both its own fields, so it cannot stand in for B here. The only
// thing this component's own `advance()` needs to do is give `b_out`
// SOME value derived from `b_in` every step; the Remap this test means
// to exercise sits entirely in the CHANNEL, not in this component's own
// physics, which nothing here checks for a particular closed form
// (unlike `LinearBlock`'s own callers) -- the test only needs
// `runner.initialize()` to be refused before any of this ever runs.
class BWithVectorOut final : public fuse::Component {
 public:
  BWithVectorOut(std::string name, double a, double k)
      : name_(std::move(name)), a_(a), k_(k) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      if (g.field(f).intent == Intent::Provides) {
        auto r = reg.bind(f, FieldRef{out_.data(),
                                      Layout{.stride = 1, .count = out_.size()},
                                      MemorySpace::Host});
        if (!r.ok()) return r;
      } else {
        auto r = reg.bind(
            f, FieldRef{&in_, Layout{.stride = 1, .count = 1}, MemorySpace::Host});
        if (!r.ok()) return r;
      }
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& /*ctx*/) override {
    ++advance_count;
    const double v = a_ + k_ * in_;
    for (double& e : out_) e = v;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override {
    saved_out_ = out_;
    saved_in_ = in_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    out_ = saved_out_;
    in_ = saved_in_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

  int advance_count = 0;

 private:
  std::string name_;
  double      a_, k_;
  std::array<double, 2> out_{0.0, 0.0}, saved_out_{0.0, 0.0};
  double      in_ = 0.0, saved_in_ = 0.0;
};

}  // namespace

TEST(WindowRunner, SingleWindowRegionExecutesAndConverges) {
  const CouplingGraph g = make_single_window_graph(WaveformIteration::GaussSeidel);

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  auto a_owned = std::make_unique<LinearBlock>("A", 1.0, 0.5);
  auto b_owned = std::make_unique<LinearBlock>("B", 2.0, 0.4);
  LinearBlock* a = a_owned.get();
  LinearBlock* b = b_owned.get();
  ComponentSet components;
  components.add(std::move(a_owned));
  components.add(std::move(b_owned));
  FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_TRUE(init.ok()) << init.error().message;

  const auto ran = runner.run_period(components);
  ASSERT_TRUE(ran.ok()) << ran.error().message;

  ASSERT_EQ(runner.iteration_stats().size(), 1u);
  const fuse::IterationStats& st = runner.iteration_stats().front();
  EXPECT_TRUE(st.converged) << "did not converge in " << st.iterations
                            << " sweeps";

  // Closed form, accounting for the WINDOW mechanism's own defining
  // property (not an ordinary point-value exchange): with window_length
  // == interval for both members, each gets exactly ONE substep spanning
  // the WHOLE window, so what it reads is the MEAN of its partner's
  // Linear-basis trajectory over [window_start, window_end) -- the
  // average of the window-START value (0, the initial condition; this is
  // the very first and only window) and the window-END value (the
  // partner's own converged `out()`), i.e. exactly half of the partner's
  // converged output:
  //   B_in = 0.5*A_out,  A_in = 0.5*B_out  (the lag on b2a vanishes at the
  //   fixed point: sweep k and k-1 agree once converged).
  //   A_out = 1 + 0.5*A_in = 1 + 0.25*B_out
  //   B_out = 2 + 0.4*B_in = 2 + 0.2*A_out
  // => A_out = 1 + 0.25*(2 + 0.2*A_out) = 1.5 + 0.05*A_out
  // => A_out = 1.5/0.95 = 30/19; B_out = 2 + 0.2*(30/19) = 44/19.
  EXPECT_NEAR(a->out(), 30.0 / 19.0, 1e-9);
  EXPECT_NEAR(b->out(), 44.0 / 19.0, 1e-9);
  EXPECT_GT(a->advance_count, 0);
  EXPECT_GT(b->advance_count, 0);
}

// The unlagged transfer's own residual block is not yet excluded from
// R.accel's accelerated vector (WindowCoupler's own accel_members/
// accel_member_offset/accel_total restriction, not yet mirrored here --
// iteration.cpp's own comment ahead of E-RT-WINDOW-ACCEL). Until that
// restriction is wired in, a GaussSeidel window naming any accelerator
// but AccelKind::None must be REJECTED at Runner setup, not silently
// accelerated wrong.
TEST(WindowRunner, GaussSeidelWindowRejectsAcceleratedRegion) {
  const CouplingGraph g = make_single_window_graph(
      WaveformIteration::GaussSeidel, fuse::AccelKind::Constant);

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  auto a_owned = std::make_unique<LinearBlock>("A", 1.0, 0.5);
  auto b_owned = std::make_unique<LinearBlock>("B", 2.0, 0.4);
  ComponentSet components;
  components.add(std::move(a_owned));
  components.add(std::move(b_owned));
  FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_FALSE(init.ok())
      << "a GaussSeidel window with a real accelerator should be refused, "
         "not silently accelerated wrong";
  EXPECT_EQ(init.error().code, "E-RT-WINDOW-ACCEL") << init.error().message;
}

// A `DtPolicy::Fixed` member (here, B: interval 1 against a window of
// 1/2) whose own last substep crosses the window end. B's own native
// step (interval 1) straddles this window's end (0.5): under
// DtPolicy::Fixed, B does not shrink to fit, so B's one substep this
// window crosses T1. The self-straddle half of the replay (B's OWN
// substep, planned from its OWN carried-forward pass_start_clock) was
// already wired in and correct on its own; B is also A's own import
// provider here (b_out -> a_in, lagged), and the CONSUMER side now
// reconstructs "what grid did B use" from the graph alone
// (fuse::window_sample_times()/window_sample_shift(), fuse/model/
// coupling_graph.hpp) instead of reading B's own carried-forward clock
// -- this configuration SUCCEEDS now (E-RT-WINDOW-STRADDLE-REPLAY
// removed). test/runtime/window_runner_straddle_test.cpp's own
// matches-driver tests are the decisive, bit-for-bit check against
// WindowCoupler directly for a genuinely multirate (13:7) pair; this
// smoke test only confirms the simplest straddling shape runs to
// convergence through the compiled graph at all.
TEST(WindowRunner, FixedPolicyStraddleNowSucceeds) {
  const CouplingGraph g = make_straddle_window_graph();

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  auto a_owned = std::make_unique<LinearBlock>("A", 1.0, 0.5);
  auto b_owned = std::make_unique<LinearBlock>("B", 2.0, 0.4);
  ComponentSet components;
  components.add(std::move(a_owned));
  components.add(std::move(b_owned));
  FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_TRUE(init.ok()) << init.error().message;

  const auto ran = runner.run_period(components);
  ASSERT_TRUE(ran.ok()) << ran.error().message;
  ASSERT_FALSE(runner.iteration_stats().empty());
  EXPECT_TRUE(runner.iteration_stats().front().converged);
}

// The SAME straddling-provider shape as FixedPolicyStraddleNowSucceeds
// above, but the channel feeding A from B's own straddling output now
// crosses a genuine (non-identity) Remap -- B's `b_out` on a 2-cell
// region, A's `a_in` on a 1-cell one. resolve_pending_window_straddles()
// publishes a straddling provider's real post-replay value straight
// into a consumer's own `window_import` arc, bypassing whatever
// Transport/Remap the channel's own compiled chain would otherwise run
// -- exact only for a same-rank, same-region (identity) transfer. This
// configuration is neither, so `Runner::initialize()` must refuse it
// with a named error rather than silently deliver the provider's
// unremapped value to A.
TEST(WindowRunner, FixedPolicyStraddleWithGenuineRemapIsRejected) {
  const CouplingGraph g = make_straddle_window_graph_with_remap();

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{1})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  auto a_owned = std::make_unique<LinearBlock>("A", 1.0, 0.5);
  auto b_owned = std::make_unique<BWithVectorOut>("B", 2.0, 0.4);
  ComponentSet components;
  components.add(std::move(a_owned));
  components.add(std::move(b_owned));
  FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_FALSE(init.ok())
      << "a straddling Fixed provider feeding a consumer through a "
         "genuine Remap should be refused at setup, not silently "
         "delivered unremapped";
  EXPECT_EQ(init.error().code, "E-RT-WINDOW-STRADDLE-REPUBLISH")
      << init.error().message;
}

// The OTHER half of E-RT-WINDOW-STRADDLE-REPUBLISH's own OR condition:
// same mesh/region (identity remap), but A and B are declared on
// DIFFERENT, non-overlapping ranks. No `RunnerConfig::
// base_communicator` is configured (this process is "rank 0" nominally
// either way), so this exercises the graph-level `RankSet::same_as()`
// check itself, not an actual cross-process transfer.
TEST(WindowRunner, FixedPolicyStraddleWithRankMismatchIsRejected) {
  const CouplingGraph g = make_straddle_window_graph_with_rank_mismatch();

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  auto a_owned = std::make_unique<LinearBlock>("A", 1.0, 0.5);
  auto b_owned = std::make_unique<LinearBlock>("B", 2.0, 0.4);
  ComponentSet components;
  components.add(std::move(a_owned));
  components.add(std::move(b_owned));
  FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_FALSE(init.ok())
      << "a straddling Fixed provider and its own consumer declared on "
         "different ranks should be refused at setup, not silently "
         "delivered as if they were co-located";
  EXPECT_EQ(init.error().code, "E-RT-WINDOW-STRADDLE-REPUBLISH")
      << init.error().message;
}

// Time windows track (Conformance, Unit 2 start): does a window-scoped
// channel's own Remap run correctly when ROUTED to a third placement
// (fuse::ComponentKind::Coupler / RemapSpec::remap_on), the shape Unit 2
// needs for U/V/Coupler on three separate rank sets -- never exercised
// together anywhere in the existing suite BEFORE this test (grepped: no
// runtime test combined window_scope with remap_on/add_coupler).
// make_single_window_graph()'s own identity-remap closed form (30/19,
// 44/19) is the oracle: if the coupler-routed remap reproduces the
// identity transfer correctly (forced via an explicit Conservative flag,
// layout_driver/main.cpp's own technique, since both ends already share
// one mesh/region), the converged values must be unchanged from
// SingleWindowRegionExecutesAndConverges above. PASSES: the mechanism
// was not missing, only untested -- docs/dev/coupler-placement.md's own
// enumerated call sites (R.participants, resolve_remap(), etc.) are
// already window-agnostic, exactly as they are launch-mode-agnostic.
TEST(WindowRunner, WindowChannelRemapRoutedToCouplerMatchesIdentityClosedForm) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});

  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 0}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{0, 0}, .interval = Rational{1}});

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

  // layout_driver/main.cpp's own technique: same mesh/region on both ends,
  // so nothing would otherwise emit a Remap at all -- Conservative asks
  // for one unconditionally, forcing the coupler-routed path regardless.
  const auto coupler = g.add_coupler(
      fuse::CouplerDecl{.name = "coupler", .ranks = RankSet{0, 0}});
  g.channel(fwd).remap(fuse::RemapFlags::Conservative).remap_on(coupler);

  CouplingWindowScope scope;
  scope.window_component = "A";
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable}, {"B", DtPolicy::Adjustable}};
  scope.iteration = WaveformIteration::GaussSeidel;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 50;
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_out"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_out"});
  decl.window_scope = scope;
  decl.acceleration.kind = fuse::AccelKind::None;
  g.add_iteration_region(decl);

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  auto a_owned = std::make_unique<LinearBlock>("A", 1.0, 0.5);
  auto b_owned = std::make_unique<LinearBlock>("B", 2.0, 0.4);
  LinearBlock* lb_a = a_owned.get();
  LinearBlock* lb_b = b_owned.get();
  ComponentSet components;
  components.add(std::move(a_owned));
  components.add(std::move(b_owned));
  FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_TRUE(init.ok()) << init.error().message;

  const auto ran = runner.run_period(components);
  ASSERT_TRUE(ran.ok()) << ran.error().message;

  ASSERT_EQ(runner.iteration_stats().size(), 1u);
  const fuse::IterationStats& st = runner.iteration_stats().front();
  EXPECT_TRUE(st.converged) << "did not converge in " << st.iterations
                            << " sweeps";
  EXPECT_NEAR(lb_a->out(), 30.0 / 19.0, 1e-9);
  EXPECT_NEAR(lb_b->out(), 44.0 / 19.0, 1e-9);
}
