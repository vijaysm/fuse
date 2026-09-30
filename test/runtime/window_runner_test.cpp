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

// A `DtPolicy::Fixed` member (here, B: interval 2/3 against a window of
// 1) whose own last substep crosses the window end must be REJECTED at
// run time, not served the old clamped-mean approximation: the correct
// treatment (fuse::detail::advance_straddle_split(), mirroring
// WindowCoupler::resolve_pending_straddles()) is not yet wired into
// advance_window() (iteration.cpp's own SCOPED LIMITATION comment).
// B's own native step (interval 1) straddles this window's end (0.5):
// under DtPolicy::Fixed, B does not shrink to fit, so B's one substep this
// window crosses T1. The self-straddle half of the replay (B's OWN
// substep, planned from its OWN carried-forward pass_start_clock) is now
// wired in and correct on its own -- but B is also A's own import
// provider here (b_out -> a_in, lagged), and the CONSUMER side (A's own
// reconstruction of "what grid did B use", via plan_substeps(import_t0,
// ...)) has no way to see B's own carried-forward clock offset once B
// has straddled in an earlier window -- discovered and diagnosed this
// session, iteration.cpp's own comment at the reinstated rejection.
// Still refuses (E-RT-WINDOW-STRADDLE-REPLAY) until that gap is closed.
TEST(WindowRunner, FixedPolicyStraddleIsRejected) {
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
  ASSERT_FALSE(ran.ok())
      << "B's own step straddles the window end and B is A's own import "
         "provider; the consumer side of the replay is not safe yet -- "
         "see this test's own comment -- so this must still refuse";
  EXPECT_EQ(ran.error().code, "E-RT-WINDOW-STRADDLE-REPLAY")
      << ran.error().message;
}
