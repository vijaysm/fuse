#include <gtest/gtest.h>
#include <fuse/model/descriptor.hpp>
#include <fuse/model/yaml_io.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <vector>

using fuse::AccelConfig;
using fuse::AccelKind;
using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
namespace C = fuse::Converged;
namespace u = fuse::units;

// ---------------------------------------------------------------------------
// Allocation counting. Spec Section 2 layer 3 requires the interpreted loop to
// allocate nothing, and Task 4.6 proved that for a plan with no implicit coupling group by
// replacing the global operator new. A sweep loop that allocates per iteration
// would slip past that test entirely, so the same instrument is used here.
// ---------------------------------------------------------------------------
namespace {
std::size_t g_allocations = 0;
bool        g_counting = false;

class AllocationCounter {
 public:
  AllocationCounter() { g_allocations = 0; g_counting = true; }
  ~AllocationCounter() { g_counting = false; }
  AllocationCounter(const AllocationCounter&) = delete;
  AllocationCounter& operator=(const AllocationCounter&) = delete;
  [[nodiscard]] std::size_t count() const { return g_allocations; }
};
}  // namespace

void* operator new(std::size_t n) {
  // #ifndef/#ifdef __CUDA_ARCH__ around the counting AND the failure path: a
  // CUDA-enabled Kokkos dual-compiles every global operator new/delete
  // override for device too (nothing here is reachable from a kernel; nvcc
  // does this unconditionally), and the device side can neither support
  // exceptions nor see g_counting/g_allocations, host globals.
#ifndef __CUDA_ARCH__
  if (g_counting) ++g_allocations;
#endif
  void* p = std::malloc(n == 0 ? 1 : n);
#ifdef __CUDA_ARCH__
  if (p == nullptr) __trap();  // the device pass cannot throw (and never runs this)
#else
  if (p == nullptr) throw std::bad_alloc();  // the standard contract, on the host
#endif
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
#ifndef __CUDA_ARCH__
  if (g_counting) ++g_allocations;
#endif
  return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
  return ::operator new(n, t);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}

namespace {

/// x <- a + k*y, a linear coupled pair whose fixed point is known exactly.
///
/// `history_` is state that `advance()` MUTATES rather than overwrites, and it
/// exists to make a missing rollback visible. Every field the component
/// provides is written from scratch on each advance, so a checkpoint bug
/// cannot show up in `out_` at all -- the test would pass against a runner
/// that never restored anything. `history_` accumulates, so a sweep that
/// starts from state the previous sweep left behind reads a different value,
/// and CheckpointsAreRestoredBetweenSweeps asserts on it.
class LinearBlock final : public fuse::Component {
 public:
  LinearBlock(std::string name, double a, double k, std::size_t n)
      : name_(std::move(name)), a_(a), k_(k), out_(n, 0.0), in_(n, 0.0) {
    // Reserved, and never grown past: this component is advanced inside the
    // allocation-counted loop, so a push_back that reallocates would be
    // counted against the runner.
    history_at_entry.reserve(kTrace);
    iteration_at_entry.reserve(kTrace);
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == fuse::Intent::Provides;
      auto& buf = is_out ? out_ : in_;
      auto r = reg.bind(f, fuse::FieldRef{buf.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = buf.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    ++advance_count;
    if (history_at_entry.size() < kTrace) {
      history_at_entry.push_back(history_);
      iteration_at_entry.push_back(ctx.iteration);
    }
    history_ += 1.0;
    // A pure function of (restored state, inputs, ctx). The drift keeps the
    // fixed point moving from coupling cycle to coupling cycle, so a later coupling cycle cannot start
    // already converged and leave the sweep loop untested. It is zero in
    // coupling cycle 0, where every value in this file is asserted.
    const double a = a_ + 0.25 * ctx.t_begin.to_double();
    for (std::size_t i = 0; i < out_.size(); ++i) {
      out_[i] = a + k_ * in_[i];
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_out_ = out_;
    saved_in_ = in_;
    saved_history_ = history_;
    ++save_count;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    // Element-wise, per Component::register_fields()'s lifetime rule: the
    // registered pointers must not move.
    for (std::size_t i = 0; i < out_.size(); ++i) out_[i] = saved_out_[i];
    for (std::size_t i = 0; i < in_.size(); ++i) in_[i] = saved_in_[i];
    history_ = saved_history_;
    ++restore_count;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }

  [[nodiscard]] const std::vector<double>& out() const { return out_; }
  [[nodiscard]] const std::vector<double>& in() const { return in_; }
  [[nodiscard]] double history() const { return history_; }

  static constexpr std::size_t kTrace = 4096;
  int advance_count = 0;
  int save_count = 0;
  int restore_count = 0;
  std::vector<double> history_at_entry;
  std::vector<int>    iteration_at_entry;

 private:
  std::string         name_;
  double              a_, k_;
  std::vector<double> out_, in_, saved_out_, saved_in_;
  double              history_ = 0.0;
  double              saved_history_ = 0.0;
};

/// Claims rollback support and does not restore the field storage it
/// registered. Spec Section 5.3 says the runtime verifies that the component saved
/// what the derived set said it must; this is the component that proves the
/// verification is a check and not a comment.
class LyingBlock final : public fuse::Component {
 public:
  LyingBlock(std::string name, double a, double k, std::size_t n)
      : name_(std::move(name)), a_(a), k_(k), out_(n, 0.0), in_(n, 0.0) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == fuse::Intent::Provides;
      auto& buf = is_out ? out_ : in_;
      auto r = reg.bind(f, fuse::FieldRef{buf.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = buf.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    for (std::size_t i = 0; i < out_.size(); ++i) out_[i] = a_ + k_ * in_[i];
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override {
    return fuse::Result<void>::success();  // saves nothing
  }
  fuse::Result<void> restore_state() override {
    return fuse::Result<void>::success();  // restores nothing, reports success
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }

 private:
  std::string         name_;
  double              a_, k_;
  std::vector<double> out_, in_;
};

/// A component that cannot roll back, for the negative test.
class NoRollback final : public fuse::Component {
 public:
  explicit NoRollback(std::size_t n) : data_(n, 0.0), in_(n, 0.0) {}
  [[nodiscard]] std::string name() const override { return "B"; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("B");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == fuse::Intent::Provides;
      auto& buf = is_out ? data_ : in_;
      auto r = reg.bind(f, fuse::FieldRef{buf.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = buf.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> data_, in_;
};

enum class BKind { Rollback, NoRollback, Lying };

constexpr std::size_t kCells = 4;

/// A(a_in) = 1 + 0.5*a_in, B(b_in) = 2 + 0.4*b_in, coupled both ways.
/// The unique fixed point is a_out = 2.5, b_out = 3.
CouplingGraph make_graph(AccelConfig accel, int max_iters,
                         fuse::ConvergenceCriterion crit,
                         double channel_relax = 0.0) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m",
                                           .source = "generated:line:4"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m,
                                            .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m,
                                            .dimension = 1});
  const auto ca = g.add_component(fuse::ComponentDecl{
      .name = "A", .ranks = {0, 1}, .interval = Rational{1}});
  const auto cb = g.add_component(fuse::ComponentDecl{
      .name = "B", .ranks = {1, 1}, .interval = Rational{1}});

  // Grouped by owner. `to_yaml` nests a field inside its component, so an
  // interleaved declaration order does not survive the YAML round trip as
  // identical FieldIds -- a pre-existing property of the schema, not of this
  // task, but one the structural comparison below would trip over.
  const auto a_out = g.add_field(ca, fuse::FieldSpec{
      .name = "a_out", .region = r1, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(ca, fuse::FieldSpec{
      .name = "a_in", .region = r1, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_in = g.add_field(cb, fuse::FieldSpec{
      .name = "b_in", .region = r2, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(cb, fuse::FieldSpec{
      .name = "b_out", .region = r2, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated");
  if (channel_relax > 0.0) {
    // A graph-level Relax inside the implicit coupling group is what makes
    // IterationRegion::carried non-empty, so it is the only way to exercise
    // the loop-carried-buffer path at all.
    g.channel(fwd).relax(channel_relax);
    g.channel(back).relax(channel_relax);
  }

  g.iterate({fwd, back})
      .until(std::move(crit))
      .max_iterations(max_iters)
      .accelerate(accel);
  return g;
}

/// BOTH field transfers, deliberately.
///
/// The obvious single-measure spelling -- `Converged::Relative(tol, "a_out")`
/// -- measures only the a2b residual, and this pair has two. Measured, before
/// the runtime learned to reject it: IQN-ILS zeroed the a2b residual EXACTLY
/// on sweep 3 while the b2a residual still stood at 0.8, and the implicit coupling group
/// reported convergence at a_out = 2.25 against a true fixed point of 2.5.
/// resolve_regions() rejects a partially measured implicit coupling group with
/// E-ITER-MEASURE (see RejectsARegionWithAnUnmeasuredChannel).
fuse::ConvergenceCriterion both_channels(double tol) {
  fuse::ConvergenceCriterion c;
  c.add(C::Relative(tol, "a_out"));
  c.add(C::Relative(tol, "b_out"));
  return c;
}

CouplingGraph make_graph(AccelConfig accel, int max_iters,
                         double tol = 1e-10) {
  return make_graph(accel, max_iters, both_channels(tol));
}

struct Fixture {
  CouplingGraph           graph;
  fuse::SerialMeshBackend backend;
  fuse::OperatorDAG       dag;
  fuse::ExecutionPlan     plan;
  ComponentSet            components;
  FieldRegistry           registry;
  std::unique_ptr<Runner> runner;
  LinearBlock*            a = nullptr;
  LinearBlock*            b = nullptr;
};

std::unique_ptr<Fixture> build_from(CouplingGraph graph,
                                    BKind bkind = BKind::Rollback) {
  auto f = std::make_unique<Fixture>();
  f->graph = std::move(graph);
  CouplingGraph& g = f->graph;

  EXPECT_TRUE(f->backend.load(g.mesh(fuse::MeshId{0})).ok());
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  f->dag = lowered.dag;

  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, f->dag, clocks, fuse::make_region_sizes(g, f->dag, f->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  f->plan = sched.plan;

  auto a = std::make_unique<LinearBlock>("A", 1.0, 0.5, kCells);
  f->a = a.get();
  f->components.add(std::move(a));
  switch (bkind) {
    case BKind::Rollback: {
      auto b = std::make_unique<LinearBlock>("B", 2.0, 0.4, kCells);
      f->b = b.get();
      f->components.add(std::move(b));
      break;
    }
    case BKind::NoRollback:
      f->components.add(std::make_unique<NoRollback>(kCells));
      break;
    case BKind::Lying:
      f->components.add(std::make_unique<LyingBlock>("B", 2.0, 0.4, kCells));
      break;
  }
  return f;
}

std::unique_ptr<Fixture> build(AccelConfig accel, int max_iters,
                               BKind bkind = BKind::Rollback,
                               double tol = 1e-10) {
  return build_from(make_graph(accel, max_iters, tol), bkind);
}

/// Registers, initializes and runs one coupling cycle. The Runner is owned by the
/// fixture, so `f.runner` stays valid for the assertions and nothing leaks
/// (the asan leg would see it if it did).
fuse::Result<void> run(Fixture& f, RunnerConfig cfg = {}) {
  f.runner = std::make_unique<Runner>(f.graph, f.dag, f.plan, f.backend);
  for (auto& c : f.components) {
    const auto r = c.register_fields(f.registry, f.graph);
    if (!r.ok()) return r;
  }
  const auto init = f.runner->initialize(f.components, f.registry, cfg);
  if (!init.ok()) return init;
  return f.runner->run_period(f.components);
}

}  // namespace

// ---------------------------------------------------------------------------
// The iteration itself.
// ---------------------------------------------------------------------------

TEST(Iteration, PicardConvergesOnALinearPair) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());
  EXPECT_GT(f->runner->stats().total_iterations, 1);
  EXPECT_EQ(f->runner->stats().windows_not_converged, 0);
  ASSERT_EQ(f->runner->iteration_stats().size(), 1u);
  EXPECT_TRUE(f->runner->iteration_stats()[0].converged);

  // Fixed point of a_out = 1 + 0.5*(2 + 0.4*a_out) is 2 / (1 - 0.2) = 2.5.
  for (const double v : f->a->out()) EXPECT_NEAR(v, 2.5, 1e-8);
  for (const double v : f->b->out()) EXPECT_NEAR(v, 3.0, 1e-8);
}

TEST(Iteration, ComponentsAdvanceOncePerIteration) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());
  EXPECT_EQ(f->a->advance_count, f->runner->stats().total_iterations);
  EXPECT_EQ(f->b->advance_count, f->runner->stats().total_iterations);
}

TEST(Iteration, TheSweepIndexReachesTheComponent) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());
  // StepContext::iteration is 0-based and increases by one per sweep, which is
  // the only thing a component can key an iteration-dependent tolerance off.
  ASSERT_EQ(static_cast<int>(f->a->iteration_at_entry.size()),
            f->a->advance_count);
  for (std::size_t i = 0; i < f->a->iteration_at_entry.size(); ++i) {
    EXPECT_EQ(f->a->iteration_at_entry[i], static_cast<int>(i));
  }
}

// ---------------------------------------------------------------------------
// Checkpoint and restore. Measured, not inferred from termination.
// ---------------------------------------------------------------------------

TEST(Iteration, CheckpointsAreSavedAndRestored) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());
  EXPECT_GT(f->a->save_count, 0) << "the derived set must include component A";
  EXPECT_GT(f->a->restore_count, 0) << "a rejected iterate must roll back";
  EXPECT_EQ(f->a->restore_count, f->runner->stats().total_iterations - 1)
      << "every sweep but the last is preceded by a rollback";
}

// A test that CAN fail where one with from-scratch Provides fields cannot:
// there every Provides field is written from scratch by advance(), so a
// runner that never restores anything still converges to 2.5. Here
// `history_` accumulates instead, so the state each sweep STARTS from is
// observable.
TEST(Iteration, CheckpointsAreRestoredBetweenSweeps) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());
  ASSERT_GT(f->a->advance_count, 3);

  std::printf("history at the start of each sweep of A:");
  for (const double h : f->a->history_at_entry) std::printf(" %g", h);
  std::printf("\n");

  for (std::size_t i = 0; i < f->a->history_at_entry.size(); ++i) {
    EXPECT_DOUBLE_EQ(f->a->history_at_entry[i], 0.0)
        << "sweep " << i
        << " started from state a previous sweep mutated: the checkpoint was "
           "not restored, so the iteration is converging to the fixed point of "
           "a different operator";
  }
  // ... and it really is mutated, so the assertion above has something to see.
  EXPECT_DOUBLE_EQ(f->a->history(), 1.0);
}

// Spec Section 5.3: the runtime verifies the component can do what the derived set
// requires of it.
TEST(Iteration, RejectsAComponentThatCannotRollBack) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 10, BKind::NoRollback);
  const auto r = run(*f);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ITER-NOROLLBACK");
}

// Spec Section 5.3's distinguishing claim over preCICE: FUSE does not merely ask the
// component to promise, it checks that the state came back.
TEST(Iteration, RejectsAComponentWhoseRestoreDoesNothing) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 10, BKind::Lying);
  const auto r = run(*f);
  ASSERT_FALSE(r.ok())
      << "supports_rollback() == true and restore_state() == success is a "
         "claim, not a rollback";
  EXPECT_EQ(r.error().code, "E-ITER-ROLLBACK");
  std::printf("rollback verification failure:\n  %s\n", r.error().message.c_str());
}

TEST(Iteration, CountsItsRollbackVerifications) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());
  // Two components, each with one Provides field, verified after every
  // rollback. An unchecked invariant is not enforced; this is the check being
  // observable.
  EXPECT_EQ(f->runner->stats().checkpoint_verifications,
            2 * (f->runner->stats().total_iterations - 1));
}

TEST(Iteration, PrintsTheDerivedCheckpointSet) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());
  ASSERT_EQ(f->runner->checkpoint_explanations().size(), 1u);
  const std::string& why = f->runner->checkpoint_explanations()[0];
  std::printf("%s\n", why.c_str());
  EXPECT_NE(why.find("\n    A"), std::string::npos);
  EXPECT_NE(why.find("\n    B"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Acceleration.
// ---------------------------------------------------------------------------

TEST(Iteration, AitkenNeedsNoMoreIterationsThanPicard) {
  auto picard = build(AccelConfig{.kind = AccelKind::None}, 200);
  auto aitken = build(AccelConfig{.kind = AccelKind::Aitken,
                                  .initial_relaxation = 0.5}, 200);
  ASSERT_TRUE(run(*picard).ok());
  ASSERT_TRUE(run(*aitken).ok());
  std::printf("iterations: Picard %lld, Aitken %lld\n",
              static_cast<long long>(picard->runner->stats().total_iterations),
              static_cast<long long>(aitken->runner->stats().total_iterations));
  EXPECT_LE(aitken->runner->stats().total_iterations,
            picard->runner->stats().total_iterations);
}

// IQN-ILS, deliberately not IQN-IMVJ. On a nonlinear global contraction IMVJ
// measured 1.3-3.6x slower than IQN-ILS with erratic per-window iteration
// counts (the signature of a stale Jacobian, which it has no restart mode to
// refresh), and on a multi-root map it landed in a different valid fixed point
// in 5 windows of 8. It is not the default and is not used to demonstrate
// convergence quality anywhere in this suite.
TEST(Iteration, IqnIlsConvergesInFewIterations) {
  auto f = build(AccelConfig{.kind = AccelKind::IQN_ILS,
                             .initial_relaxation = 0.5,
                             .reuse = 8,
                             .filter = fuse::FilterKind::QR2,
                             .limit = 1e-2}, 100);
  ASSERT_TRUE(run(*f).ok());
  std::printf("IQN-ILS iterations: %lld\n",
              static_cast<long long>(f->runner->stats().total_iterations));
  EXPECT_LT(f->runner->stats().total_iterations, 12);
  for (const double v : f->a->out()) EXPECT_NEAR(v, 2.5, 1e-8);
}

// IQN-IMVJ, measured against the answer rather than against its own
// bookkeeping. Until this test existed IMVJ ran in NO convergence-quality test
// anywhere in the suite -- the unit tests count columns and retained blocks,
// the comment above says in as many words that IMVJ "is not used to
// demonstrate convergence quality anywhere in this suite", and that is how the
// retained Jacobian came to survive a divergence unnoticed.
//
// Two claims, neither of them a pinned iteration count (quasi-Newton
// counts are not stable across build types, and this problem is only linear
// per window):
//
//   * IMVJ finds the same fixed point as unaccelerated Picard, coupling cycle after
//     coupling cycle, to 1e-8 -- a fast wrong answer is not an answer;
//   * what it retains across the window boundary earns its keep. The coupled
//     operator here is linear with a constant Jacobian and only the constant
//     term drifts with time, so the inverse Jacobian retained from the first
//     window is exact for every later one and later windows cost strictly
//     less than IQN-ILS's, which start from nothing every time.
//
// The comparison is against IQN-ILS, NOT against IMVJ's own first window:
// MEASURED, every window after the first is cheaper than the first whatever
// the accelerator retains, simply because the components start it from the
// previous window's converged state. A "later windows are cheaper" assertion
// passes against an IMVJ with retention disabled (6 4 4 4 4 4 against a true
// 6 2 2 2 2 2) and would measure nothing.
//
// No count is pinned; quasi-Newton counts are not stable across build
// types, and while this problem is linear per window the point here is a
// comparison, not a number.
TEST(Iteration, ImvjConvergesToTheSameAnswerAndItsRetainedJacobianPays) {
  constexpr int kPeriods = 6;
  auto picard = build(AccelConfig{.kind = AccelKind::None}, 400);
  auto ils    = build(AccelConfig{.kind = AccelKind::IQN_ILS,
                                  .initial_relaxation = 0.5,
                                  .reuse              = 8}, 100);
  auto imvj   = build(AccelConfig{.kind = AccelKind::IQN_IMVJ,
                                  .initial_relaxation = 0.5,
                                  .reuse              = 8}, 100);
  ASSERT_TRUE(run(*picard).ok());
  ASSERT_TRUE(run(*ils).ok());
  ASSERT_TRUE(run(*imvj).ok());

  std::vector<int> imvj_windows;
  std::vector<int> ils_windows;
  for (int p = 0; p < kPeriods; ++p) {
    if (p > 0) {
      ASSERT_TRUE(picard->runner->run_period(picard->components).ok());
      ASSERT_TRUE(ils->runner->run_period(ils->components).ok());
      ASSERT_TRUE(imvj->runner->run_period(imvj->components).ok());
    }
    ASSERT_EQ(imvj->runner->iteration_stats().size(), 1u);
    EXPECT_TRUE(imvj->runner->iteration_stats()[0].converged) << "window " << p;
    imvj_windows.push_back(imvj->runner->iteration_stats()[0].iterations);
    ils_windows.push_back(ils->runner->iteration_stats()[0].iterations);
    for (std::size_t i = 0; i < picard->a->out().size(); ++i) {
      EXPECT_NEAR(imvj->a->out()[i], picard->a->out()[i], 1e-8)
          << "window " << p << ", a_out[" << i << "]";
      EXPECT_NEAR(imvj->b->out()[i], picard->b->out()[i], 1e-8)
          << "window " << p << ", b_out[" << i << "]";
    }
  }

  int imvj_later = 0;
  int ils_later  = 0;
  for (std::size_t w = 1; w < imvj_windows.size(); ++w) {
    imvj_later += imvj_windows[w];
    ils_later  += ils_windows[w];
    EXPECT_LE(imvj_windows[w], ils_windows[w]) << "window " << w;
  }
  std::printf("sweeps per window   IMVJ:");
  for (const int c : imvj_windows) std::printf(" %d", c);
  std::printf("   IQN-ILS:");
  for (const int c : ils_windows) std::printf(" %d", c);
  std::printf("\n  %lld IMVJ sweeps in %d windows against IQN-ILS's %lld and "
              "Picard's %lld; a_out=%.15g (Picard %.15g)\n",
              static_cast<long long>(imvj->runner->stats().total_iterations),
              kPeriods,
              static_cast<long long>(ils->runner->stats().total_iterations),
              static_cast<long long>(picard->runner->stats().total_iterations),
              imvj->a->out()[0], picard->a->out()[0]);

  EXPECT_EQ(imvj->runner->stats().windows_not_converged, 0);
  EXPECT_LT(imvj->runner->stats().total_iterations,
            picard->runner->stats().total_iterations / 3)
      << "a quasi-Newton method that does not beat Picard by a wide margin on "
         "a linear problem is not working";
  EXPECT_LT(imvj_later, ils_later)
      << "after the first window IMVJ cost as much as IQN-ILS, so nothing is "
         "being carried across the window boundary -- the retained inverse "
         "Jacobian is exact for this problem and must show up as fewer sweeps";
}

TEST(Iteration, ResidualHistoryDecreasesMonotonicallyForPicard) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());
  const auto& h = f->runner->iteration_stats()[0].residual_history;
  ASSERT_GT(h.size(), 2u);
  EXPECT_EQ(static_cast<int>(h.size()),
            f->runner->iteration_stats()[0].iterations);
  std::printf("Picard residual history:");
  for (const double r : h) std::printf(" %.3e", r);
  std::printf("\n");
  // A contraction with factor sqrt(0.2) must reduce the residual every sweep.
  for (std::size_t i = 1; i < h.size(); ++i) {
    EXPECT_LT(h[i], h[i - 1]) << "at sweep " << i;
  }
}

// ---------------------------------------------------------------------------
// A converged answer is a BETTER answer, with numbers.
// ---------------------------------------------------------------------------

TEST(Iteration, ConvergedBeatsExplicitOnTheSameProblem) {
  // One sweep is exactly the explicit scheme: advance once, exchange once. The
  // runner rejects to call that converged (max_iterations = 1 with a criterion
  // it cannot meet), so the comparison is against the state that run leaves.
  auto explicit_run = build(AccelConfig{.kind = AccelKind::None}, 1);
  RunnerConfig lenient;
  lenient.abort_on_nonconvergence = false;
  ASSERT_TRUE(run(*explicit_run, lenient).ok());
  ASSERT_EQ(explicit_run->runner->stats().windows_not_converged, 1);

  auto implicit_run = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*implicit_run).ok());
  ASSERT_EQ(implicit_run->runner->stats().windows_not_converged, 0);

  const double exact_a = 2.5;
  const double exact_b = 3.0;
  const double e_explicit =
      std::max(std::abs(explicit_run->a->out()[0] - exact_a),
               std::abs(explicit_run->b->out()[0] - exact_b));
  const double e_implicit =
      std::max(std::abs(implicit_run->a->out()[0] - exact_a),
               std::abs(implicit_run->b->out()[0] - exact_b));
  std::printf(
      "one explicit sweep: a_out=%.15g b_out=%.15g  |error|=%.3e\n"
      "%lld iterated sweeps: a_out=%.15g b_out=%.15g  |error|=%.3e\n"
      "error reduced by a factor of %.3e\n",
      explicit_run->a->out()[0], explicit_run->b->out()[0], e_explicit,
      static_cast<long long>(implicit_run->runner->stats().total_iterations),
      implicit_run->a->out()[0], implicit_run->b->out()[0], e_implicit,
      e_explicit / e_implicit);
  EXPECT_GT(e_explicit, 1e-2) << "the explicit answer really is wrong";
  EXPECT_LT(e_implicit, 1e-9) << "the iterated answer really is right";
  EXPECT_LT(e_implicit * 1e6, e_explicit);
}

// ---------------------------------------------------------------------------
// Non-convergence.
// ---------------------------------------------------------------------------

TEST(Iteration, ReportsNonConvergenceRatherThanLooping) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 2);  // too few
  RunnerConfig cfg;
  cfg.abort_on_nonconvergence = false;
  ASSERT_TRUE(run(*f, cfg).ok());
  EXPECT_EQ(f->runner->stats().windows_not_converged, 1);
  EXPECT_EQ(f->runner->stats().total_iterations, 2);
  EXPECT_FALSE(f->runner->iteration_stats()[0].converged);
  EXPECT_EQ(f->runner->iteration_stats()[0].iterations, 2);
}

TEST(Iteration, AbortsOnNonConvergenceWhenConfigured) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 2);
  RunnerConfig cfg;
  cfg.abort_on_nonconvergence = true;
  const auto r = run(*f, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-NOCONVERGE");
  std::printf("non-convergence report:\n  %s\n", r.error().message.c_str());
}

// The state after a non-converged window is defined: the last iterate, and the
// last iterate is what the components hold. Not a rollback to the checkpoint,
// and not silence.
TEST(Iteration, NonConvergenceLeavesTheLastIterate) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 2);
  RunnerConfig cfg;
  cfg.abort_on_nonconvergence = false;
  ASSERT_TRUE(run(*f, cfg).ok());
  // Sweep 1 fed y = 0 -> a_out = 1. Sweep 2 fed y = (a_out, b_out) = (1, 2)
  // -> a_out = 1 + 0.5*2 = 2, b_out = 2 + 0.4*1 = 2.4. Those are the values
  // left behind, and they are neither the checkpoint (0) nor the answer (2.5).
  EXPECT_DOUBLE_EQ(f->a->out()[0], 2.0);
  EXPECT_DOUBLE_EQ(f->b->out()[0], 2.4);
  EXPECT_DOUBLE_EQ(f->a->history(), 1.0) << "state is not double-advanced";
}

// An empty criterion is what a graph arrives with if any boundary it crossed
// dropped the field -- and it reports satisfied, so the implicit coupling group would stop after
// one sweep and call itself converged.
TEST(Iteration, RejectsAnEmptyConvergenceCriterion) {
  auto f = build_from(make_graph(AccelConfig{.kind = AccelKind::None}, 10,
                                 fuse::ConvergenceCriterion{}));
  const auto r = run(*f);
  ASSERT_FALSE(r.ok())
      << "an empty ConvergenceCriterion is satisfied by definition, so the "
         "region would run one sweep and report convergence";
  EXPECT_EQ(r.error().code, "E-ITER-NOCONVERGE");
  std::printf("empty criterion:\n%s\n", r.error().message.c_str());
}

TEST(Iteration, RejectsMaxIterationsBelowOne) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 0);
  const auto r = run(*f);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ITER-MAXITER");
}

// A measure naming a field no residual of the implicit coupling group carries is never tested
// against anything. Silently ignoring it would leave the implicit coupling group stopping on
// whatever measures did match, or running to max_iterations with no measure at
// all -- both indistinguishable from working.
TEST(Iteration, RejectsAConvergenceMeasureThatNamesNothing) {
  fuse::ConvergenceCriterion c;
  c.add(C::Relative(1e-8, "no_such_field"));
  auto f = build_from(
      make_graph(AccelConfig{.kind = AccelKind::None}, 10, std::move(c)));
  const auto r = run(*f);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ITER-MEASURE");
  std::printf("unmatched measure:\n%s\n", r.error().message.c_str());
}

// AND across measures (convergence.hpp): tightening ONE conjunct must cost
// sweeps, or that conjunct is not being tested.
TEST(Iteration, EveryConjunctIsActuallyTested) {
  fuse::ConvergenceCriterion tight;
  tight.add(C::Relative(1e-10, "a_out"));
  tight.add(C::Relative(1e-13, "b_out"));
  auto f = build_from(
      make_graph(AccelConfig{.kind = AccelKind::None}, 200, std::move(tight)));
  ASSERT_TRUE(run(*f).ok());
  EXPECT_EQ(f->runner->stats().windows_not_converged, 0);

  auto loose = build(AccelConfig{.kind = AccelKind::None}, 200);
  ASSERT_TRUE(run(*loose).ok());
  std::printf("1e-10 AND 1e-13: %lld sweeps; 1e-10 AND 1e-10: %lld\n",
              static_cast<long long>(f->runner->stats().total_iterations),
              static_cast<long long>(loose->runner->stats().total_iterations));
  EXPECT_GT(f->runner->stats().total_iterations,
            loose->runner->stats().total_iterations)
      << "the tighter conjunct must cost extra sweeps, or it is not being "
         "tested at all";
}

// A graph-level Relax inside the implicit coupling group: the only construct that makes
// IterationRegion::carried non-empty today. It must still converge to the same
// fixed point, at the same tolerance.
//
// This is the case that settles what `carried` means. Restoring the Relax
// shadow to its coupling-cycle-start value x0 on every sweep -- the literal reading of
// src/ir/checkpoint.cpp -- makes the composed map's fixed point satisfy
// x* = w*chain(x*) + (1-w)*x0, i.e. NOT 2.5. The sweep loop therefore keeps
// the shadow in step with the delivered iterate instead of resetting it, and
// this test is what tells the two apart.
TEST(Iteration, ARelaxedChannelInsideTheRegionStillFindsTheFixedPoint) {
  auto f = build_from(make_graph(AccelConfig{.kind = AccelKind::None}, 300,
                                 both_channels(1e-10), /*channel_relax=*/0.7));
  ASSERT_TRUE(run(*f).ok());
  EXPECT_EQ(f->runner->stats().windows_not_converged, 0);
  std::printf("relaxed (omega=0.7) Picard: %lld sweeps, a_out=%.15g\n",
              static_cast<long long>(f->runner->stats().total_iterations),
              f->a->out()[0]);
  for (const double v : f->a->out()) EXPECT_NEAR(v, 2.5, 1e-8);
  for (const double v : f->b->out()) EXPECT_NEAR(v, 3.0, 1e-8);
}

// A space-time field transfer inside an implicit coupling group cannot be
// built: EVERY construction attempted here is rejected at
// `Runner::initialize()`, before a single sweep runs, so there is no
// idempotency question to answer against the current runtime -- the
// combination is refused, not merely untested.
//
// TWO CONSTRUCTIONS ARE TRIED, both real graphs put through the full
// lower -> schedule -> initialize pipeline, not a reading of the code:
//
//   1. `.spacetime(...)` directly on `fwd` (A -> B, both already in the
//      implicit coupling group, mirroring the pattern `ARelaxedChannelInsideTheRegionStill-
//      FindsTheFixedPoint` uses for Relax). `schedule()` accepts it. Runner
//      set-up (`resolve_regions()`, src/runtime/iteration.cpp) does not: the
//      space-time field transfer's `Sample` node runs "after-ticks" at t=0 -- the
//      exact instant A itself ticks, because a single-source-step history
//      (the only kind `E-SCHED-REGION`'s contiguity rule admits inside an
//      implicit coupling group) has nothing to wait for
//      before sampling. That makes `Sample` the implicit coupling group's chronologically
//      first body-member instruction, which pins `enter_time` to 0, and A's
//      own tick -- which the implicit coupling group must re-run every sweep for the pair to
//      iterate at all -- is then "at or after the region begins," rejected
//      with E-RT-REGION.
//
//   2. Moving the space-time field transfer's SOURCE outside the A<->B cycle, to a
//      third, non-cyclic component `Ext` feeding a fourth, `Snk`, via
//      `ext2snk`, with `ext2snk` in the SAME `iterate({fwd, back, ext2snk})`
//      call (there is no API to extend an already-declared implicit coupling group). This
//      does not sidestep the problem: `derive_checkpoint_set()`
//      (src/ir/checkpoint.cpp) pulls in the OWNING component of every
//      body-member field transfer, not only the ones forming the cyclic dependency,
//      so `Ext` and `Snk` are checkpointed too. `Ext`'s own tick is then
//      "mine" for the SAME guard, and since `Ext`'s Sample is (by the same
//      after-ticks reasoning as construction 1) the implicit coupling group's chronologically
//      first instruction, `enter_time` is pinned to 0 again -- and this time
//      it is A's tick that collides, exactly as in construction 1, just one
//      hop removed. Every topology tried funnels into the same collision.
//
// REACHABILITY OF THE GUARD. `resolve_regions()`'s tick-time check
// (src/runtime/iteration.cpp, a few lines above where `R.tick_indices`
// is built) guards a coincidence that does not require an implicit
// coupling group whose field transfers straddle two time grids (which
// E-SCHED-REGION rejects first). A space-time field transfer's `Sample`
// node straddles exactly the pair of instants -- "after this tick" and "as
// the implicit coupling group's own first instruction" -- so the guard is
// reachable; this is shown here concretely, the same way
// `check_region_isolation()`'s reachability is checked below rather than
// assumed.
//
// No idempotency bug is found, and none can be looked for: the runtime
// rejects the combination safely and with a clear diagnostic (E-RT-REGION)
// rather than running it incorrectly, so nothing here is unsafe. Since
// nothing exercises the combination, the correct pin is the negative
// regression below (this combination stays rejected, on purpose, until a
// future change deliberately relaxes the tick-time guard or gives
// space-time field transfers a different scheduling treatment inside
// implicit coupling groups).
TEST(Iteration, ASpacetimeChannelInAnIterationRegionIsRefusedNotSilentlyWrong) {
  CouplingGraph g = make_graph(AccelConfig{.kind = AccelKind::None}, 300,
                               both_channels(1e-10));
  // `fwd` is FieldId-addressed the same way
  // `RejectsAnExplicitChannelHangingOffAnIteratedComponent` reaches into an
  // already-built graph: field transfer 0 is "a2b" (A.a_out -> B.b_in), declared
  // first by `make_graph()` and already a member of its implicit coupling group.
  g.channel(fuse::ChannelId{0})
      .spacetime(fuse::TemporalBasis::Constant, 0, fuse::Window::coupling_period());

  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  // Unlike the explicit-field-transfer-hanging-off-a-component case, this construction
  // is NOT caught by `schedule()`: a space-time field transfer with a single source
  // and target step per period satisfies E-SCHED-REGION's contiguity rule
  // exactly as an ordinary instant field transfer would. The rejection is the
  // Runner's, at initialize() time.
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  components.add(std::make_unique<LinearBlock>("A", 1.0, 0.5, kCells));
  components.add(std::make_unique<LinearBlock>("B", 2.0, 0.4, kCells));
  FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_FALSE(init.ok())
      << "a space-time field transfer inside an implicit coupling group now runs -- "
         "its idempotency needs actually testing, not just documenting as "
         "rejected";
  EXPECT_EQ(init.error().code, "E-RT-REGION");
  EXPECT_NE(init.error().message.find("at or after the region begins"),
           std::string::npos)
      << init.error().message;
  std::printf("space-time-in-region refusal:\n  %s\n",
             init.error().message.c_str());
}

// The finding that changed the fixture. An implicit coupling group whose criterion covers only
// some of its field transfers can stop at a point that is not the coupled fixed
// point, and an accelerator makes that the normal case rather than the corner
// case: measured here, IQN-ILS zeroed the measured residual exactly while the
// unmeasured one stood at 0.8, reporting convergence at a_out = 2.25 against
// a true 2.5.
TEST(Iteration, RejectsARegionWithAnUnmeasuredChannel) {
  fuse::ConvergenceCriterion partial;
  partial.add(C::Relative(1e-10, "a_out"));  // says nothing about b2a
  auto f = build_from(make_graph(AccelConfig{.kind = AccelKind::IQN_ILS}, 20,
                                 std::move(partial)));
  const auto r = run(*f);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ITER-MEASURE");
  std::printf("unmeasured channel:\n%s\n", r.error().message.c_str());
}

// An explicit field transfer that hangs off a component the implicit coupling group re-advances would
// run once, outside the loop, on the pre-iteration value, and the converged
// iterate would never reach its consumer. Rejected -- and rejected BEFORE the
// runner sees it, by schedule()'s contiguity rule, because the extra field transfer's
// instructions interleave with the implicit coupling group body. The Runner's own isolation
// check (E-RT-REGION, check_region_isolation) is the backstop for a plan that
// did not come from schedule(); measured here, every graph this test could
// construct is stopped one layer earlier, which is why this asserts on
// E-SCHED-REGION and not on the runner's code.
TEST(Iteration, RejectsAnExplicitChannelHangingOffAnIteratedComponent) {
  CouplingGraph g = make_graph(AccelConfig{.kind = AccelKind::None}, 10);
  const auto m = fuse::MeshId{0};
  const auto r3 =
      g.add_region(fuse::Region{.name = "r3", .mesh = m, .dimension = 1});
  const auto cc = g.add_component(fuse::ComponentDecl{
      .name = "C", .ranks = {2, 1}, .interval = Rational{1}});
  const auto c_in = g.add_field(cc, fuse::FieldSpec{
      .name = "c_in", .region = r3, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  // A.a_out is FieldId 0 and is re-advanced by the implicit coupling group on every sweep.
  g.add_channel("a2c", fuse::FieldId{0}, c_in);

  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(m)).ok());
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  const auto sched = schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  ASSERT_FALSE(sched.ok());
  EXPECT_NE(sched.diags.to_string().find("E-SCHED-REGION"), std::string::npos)
      << sched.diags.to_string();
  std::printf("isolation:\n  %s\n", sched.diags.to_string().c_str());
}

// `check_region_isolation()` (src/runtime/iteration.cpp) is
// documented as "A BACKSTOP, not the first line of defence... catches...
// any plan that did not come from schedule()." The test above measures that
// every graph an AUTHOR can build is stopped one layer earlier, by
// E-SCHED-REGION; it does not, and cannot, measure the backstop itself,
// since it never reaches the Runner. This test does, by taking the "any plan
// that did not come from schedule()" clause literally: it schedules a real
// graph to get a genuinely valid `ExecutionPlan`, then hands `Runner` a
// byte-for-byte copy with exactly one `Instruction::region_exit` marker moved
// one slot earlier -- still a well-formed plan (same instructions, same
// buffers, same edges; only which one is tagged "last instruction of the
// implicit coupling group" changes) -- and lets `Runner::initialize()` see it. `Runner`'s
// constructor takes an `ExecutionPlan` by value with no provenance check
// (include/fuse/runtime/runner.hpp), and `Instruction`/`ExecutionPlan` are
// both public, mutable types (include/fuse/schedule/plan.hpp) built exactly
// for this: a test seam, not a backdoor.
//
// Reaching this through the real schedule() -> Runner path is not
// achievable: every construction that puts an
// out-of-implicit coupling group instruction downstream of a re-advanced component's own
// output also makes that instruction's node a member of the implicit coupling group body by
// construction (`build_iteration_regions()`, src/ir/regions.cpp, assigns
// every node whose origin is a member field transfer to the body), which
// `E-SCHED-REGION`'s contiguity rule then rejects first -- the same result
// `RejectsAnExplicitChannelHangingOffAnIteratedComponent` above measures for
// its own construction. There is no public way to get a body node excluded
// from `[region_enter, region_exit]` except by handing `Runner` a plan
// `schedule()` did not produce, which is exactly this test's premise, not a
// workaround of it.
//
// WHY MOVING `region_exit` REACHES IT, CONCRETELY. `make_graph()`'s plan
// (dumped by `PrintsTheDerivedCheckpointSet`) marks instruction 9 ("Residual"
// on the b2a leg, reading edge %7 = "b2a/diagnose") as `region_exit`, the
// implicit coupling group's last re-executed instruction. Moving that marker to instruction 8
// (a2b's own Residual) excludes instruction 9 from every sweep after the
// first -- it becomes an ordinary once-per-coupling-cycle instruction sitting just
// past the loop, still reading %7. %7 is downstream of B's own `Source`
// output (B is checkpointed, re-advanced every sweep), so it is exactly the
// "tainted" case the isolation check exists to name: a value the loop keeps
// changing, read by an instruction the loop does not re-run. Concretely,
// "b2a/residual" would report the FIRST sweep's residual forever after,
// while the implicit coupling group kept converging around it -- a diagnostic that silently
// stops meaning anything, with no crash and no wrong physics (the fixed
// point itself does not depend on this edge), which is precisely the
// low-drama, easy-to-miss failure mode the backstop's own comment describes.
TEST(Iteration, CheckRegionIsolationCatchesAHandBuiltPlanScheduleWouldRefuse) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);

  std::vector<fuse::Instruction> mutated = f->plan.instructions();
  int old_exit = -1, new_exit = -1;
  for (std::size_t i = 0; i < mutated.size(); ++i) {
    if (mutated[i].region_exit == 0) {
      old_exit = static_cast<int>(i);
      mutated[i].region_exit = -1;
    }
  }
  ASSERT_GE(old_exit, 1) << "no region_exit marker found in the plan; "
                            "make_graph()'s shape changed under this test";
  new_exit = old_exit - 1;
  ASSERT_EQ(mutated[static_cast<std::size_t>(new_exit)].region_exit, -1)
      << "instruction " << new_exit
      << " was already inside/at a region boundary; pick a different "
         "instruction to move the marker to";
  mutated[static_cast<std::size_t>(new_exit)].region_exit = 0;

  fuse::ExecutionPlan shrunk{std::move(mutated), f->plan.buffers(),
                            f->plan.period(), f->plan.lockstep(),
                            f->plan.region_count()};

  ComponentSet components;
  components.add(std::make_unique<LinearBlock>("A", 1.0, 0.5, kCells));
  components.add(std::make_unique<LinearBlock>("B", 2.0, 0.4, kCells));
  FieldRegistry registry;
  for (auto& c : components) {
    ASSERT_TRUE(c.register_fields(registry, f->graph).ok());
  }

  Runner runner{f->graph, f->dag, shrunk, f->backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_FALSE(init.ok())
      << "check_region_isolation() did not fire against a plan built to "
         "trip it -- either the backstop or this test's construction of the "
         "'plan schedule() would reject' case is broken";
  EXPECT_EQ(init.error().code, "E-RT-REGION");
  EXPECT_NE(init.error().message.find("lies outside iteration region"),
           std::string::npos)
      << init.error().message;
  std::printf("check_region_isolation backstop:\n  %s\n",
             init.error().message.c_str());
}

// ---------------------------------------------------------------------------
// Nothing in the sweep loop allocates.
// ---------------------------------------------------------------------------

// Under FUSE_EXECUTION_SPACE=device (the `.device` twin of this row) the
// sweep loop's device copies and kernel launches go through Kokkos::fence()
// and Kokkos::parallel_for() with string labels, each a std::string Kokkos
// builds per call; those allocations are Kokkos's, not the sweep loop's, so
// the two allocation rows below measure nothing about FUSE there and skip.
bool runner_executes_on_the_device() {
  const auto space = RunnerConfig{}.resolved_execution_space();
  return space.ok() &&
         space.value() == RunnerConfig::RunnerExecutionSpace::Device;
}

TEST(Iteration, TheSweepLoopAllocatesNothing) {
  if (runner_executes_on_the_device()) {
    GTEST_SKIP() << "Kokkos fence/launch labels allocate under the device interpreter";
  }
  // The components drift with time, so every coupling cycle has a new fixed point and
  // runs a full set of sweeps -- accelerate, roll back, verify, re-deliver,
  // re-advance -- rather than starting already converged and leaving the loop
  // untested.
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  ASSERT_TRUE(run(*f).ok());  // period 0 warms every buffer
  const std::int64_t before = f->runner->stats().total_iterations;

  std::size_t allocations = 0;
  {
    AllocationCounter counter;
    for (int p = 0; p < 8; ++p) {
      ASSERT_TRUE(f->runner->run_period(f->components).ok());
    }
    allocations = counter.count();
  }
  const std::int64_t swept = f->runner->stats().total_iterations - before;
  std::printf("Picard: %lld sweeps over 8 periods, %zu allocations\n",
              static_cast<long long>(swept), allocations);
  EXPECT_GT(swept, 8) << "at least one coupling cycle must have taken several sweeps, "
                         "or the loop was never exercised";
  EXPECT_EQ(allocations, 0u);
  EXPECT_EQ(f->runner->stats().windows_not_converged, 0);
}

// The accelerator is inside the sweep loop, so the rule above is its rule too.
//
// AitkenAccel::update() must not build a
// `std::vector<double> r(n)` on every call (src/num/accel_relaxation.cpp), nor may
// the quasi-Newton members allocate their whole least-squares workspace per
// call: that would be exactly 56 allocations for the 56 accelerator updates
// below. Every one of those buffers is a member of the accelerator, and this
// asserts the zero rather than a bound -- a `<= updates` bound is satisfied by one
// allocation per update, which is the defect, and by none, which is the fix,
// so it cannot tell them apart.
//
// The first coupling cycle, run above, is warm-up: a buffer has to be allocated once
// before it can be reused, and "steady state" is the claim.
TEST(Iteration, TheAcceleratorAllocatesNothingPerUpdate) {
  if (runner_executes_on_the_device()) {
    GTEST_SKIP() << "Kokkos fence/launch labels allocate under the device interpreter";
  }
  auto f = build(AccelConfig{.kind = AccelKind::Aitken}, 100);
  ASSERT_TRUE(run(*f).ok());
  const std::int64_t before = f->runner->stats().total_iterations;

  std::size_t allocations = 0;
  {
    AllocationCounter counter;
    for (int p = 0; p < 8; ++p) {
      ASSERT_TRUE(f->runner->run_period(f->components).ok());
    }
    allocations = counter.count();
  }
  const std::int64_t swept = f->runner->stats().total_iterations - before;
  const std::int64_t updates = swept - 8;  // one fewer update than sweeps
  std::printf("Aitken: %lld sweeps over 8 periods, %lld accelerator updates, "
              "%zu allocations\n",
              static_cast<long long>(swept), static_cast<long long>(updates),
              allocations);
  ASSERT_GT(updates, 0) << "no accelerator update ran, so nothing was measured";
  EXPECT_EQ(allocations, 0u)
      << "an accelerated region must allocate nothing per sweep, exactly like "
         "the unaccelerated one above";
}

// ---------------------------------------------------------------------------
// The criterion has to survive the serialised coupling specification (spec Section 3.2).
// ---------------------------------------------------------------------------

TEST(Iteration, ConvergenceAndAccelerationCrossTheDescriptor) {
  CouplingGraph g = make_graph(AccelConfig{.kind = AccelKind::IQN_ILS,
                                           .initial_relaxation = 0.25,
                                           .omega = 0.75,
                                           .reuse = 6,
                                           .filter = fuse::FilterKind::QR3,
                                           .limit = 5e-3,
                                           .scale_fields = false},
                               17, 1e-9);
  const auto bytes = fuse::serialize(g);
  const auto back = fuse::deserialize(bytes);
  ASSERT_TRUE(back.ok()) << back.error().message;

  const auto& d = back.value().iteration_region(fuse::IterationRegionIdx{0});
  EXPECT_EQ(d.max_iterations, 17);
  ASSERT_EQ(d.convergence.size(), 2u);
  EXPECT_EQ(d.convergence.measures()[0].kind, fuse::MeasureKind::Relative);
  EXPECT_DOUBLE_EQ(d.convergence.measures()[0].tolerance, 1e-9);
  EXPECT_EQ(d.convergence.measures()[0].field, "a_out");
  EXPECT_EQ(d.convergence.measures()[1].field, "b_out");
  EXPECT_EQ(d.acceleration.kind, AccelKind::IQN_ILS);
  EXPECT_DOUBLE_EQ(d.acceleration.initial_relaxation, 0.25);
  EXPECT_DOUBLE_EQ(d.acceleration.omega, 0.75);
  EXPECT_EQ(d.acceleration.reuse, 6);
  EXPECT_EQ(d.acceleration.filter, fuse::FilterKind::QR3);
  EXPECT_DOUBLE_EQ(d.acceleration.limit, 5e-3);
  EXPECT_FALSE(d.acceleration.scale_fields);

  EXPECT_TRUE(structurally_equal(g, back.value()));
}

// The descriptor must be able to TELL two graphs apart that differ only in the
// criterion. If serialize() ignored the field, these bytes would be identical
// and structurally_equal() would call an unconverging graph equal to a
// converging one -- which is exactly how a criterion goes missing at the
// serialised coupling specification without anyone noticing.
TEST(Iteration, TheDescriptorDistinguishesGraphsByTheirCriterion) {
  const CouplingGraph with =
      make_graph(AccelConfig{.kind = AccelKind::None}, 30, 1e-10);
  const CouplingGraph tighter =
      make_graph(AccelConfig{.kind = AccelKind::None}, 30, 1e-12);
  const CouplingGraph bare = make_graph(AccelConfig{.kind = AccelKind::None},
                                        30, fuse::ConvergenceCriterion{});
  EXPECT_NE(fuse::serialize(with), fuse::serialize(tighter));
  EXPECT_NE(fuse::serialize(with), fuse::serialize(bare));
  EXPECT_FALSE(structurally_equal(with, bare));

  const CouplingGraph fast = make_graph(
      AccelConfig{.kind = AccelKind::IQN_ILS}, 30, 1e-10);
  EXPECT_NE(fuse::serialize(with), fuse::serialize(fast));
}

TEST(Iteration, ConvergenceAndAccelerationCrossTheYaml) {
  CouplingGraph g = make_graph(AccelConfig{.kind = AccelKind::Aitken,
                                           .initial_relaxation = 0.125,
                                           .reuse = 4,
                                           .filter = fuse::FilterKind::QR1,
                                           .limit = 0.25},
                               23, 1e-7);
  const auto text = fuse::to_yaml(g);
  ASSERT_TRUE(text.ok()) << text.error().message;
  std::printf("%s\n", text.value().c_str());
  const auto back = fuse::from_yaml(text.value());
  ASSERT_TRUE(back.ok()) << back.error().message;

  const auto& d = back.value().iteration_region(fuse::IterationRegionIdx{0});
  ASSERT_EQ(d.convergence.size(), 2u);
  EXPECT_EQ(d.convergence.measures()[0].field, "a_out");
  EXPECT_EQ(d.convergence.measures()[1].field, "b_out");
  EXPECT_DOUBLE_EQ(d.convergence.measures()[0].tolerance, 1e-7);
  EXPECT_EQ(d.acceleration.kind, AccelKind::Aitken);
  EXPECT_EQ(d.acceleration.filter, fuse::FilterKind::QR1);
  EXPECT_TRUE(structurally_equal(g, back.value()))
      << "the YAML round trip must preserve every byte the descriptor carries, "
         "or a graph authored in YAML iterates with a different stopping rule "
         "than the one written down";
}

// Every enumerator, not just the one the fixture happens to use: the YAML
// tables are index-keyed, so a reordered enum reads one kind back as another.
TEST(Iteration, EveryMeasureAndAcceleratorKindRoundTripsThroughYaml) {
  const fuse::MeasureKind kinds[] = {
      fuse::MeasureKind::Relative, fuse::MeasureKind::Absolute,
      fuse::MeasureKind::AbsoluteOrRelative,
      fuse::MeasureKind::ResidualRelative};
  const AccelKind accels[] = {AccelKind::None,       AccelKind::Constant,
                              AccelKind::Aitken,     AccelKind::Secant,
                              AccelKind::Steffensen, AccelKind::IQN_ILS,
                              AccelKind::IQN_IMVJ};
  const fuse::FilterKind filters[] = {
      fuse::FilterKind::None, fuse::FilterKind::QR1, fuse::FilterKind::QR2,
      fuse::FilterKind::QR3};

  for (const fuse::MeasureKind mk : kinds) {
    for (std::size_t i = 0; i < std::size(accels); ++i) {
      const fuse::FilterKind fk = filters[i % std::size(filters)];
      fuse::ConvergenceCriterion c;
      c.add(fuse::ConvergenceMeasure{mk, 1e-5, "a_out"});
      const CouplingGraph h = make_graph(
          AccelConfig{.kind = accels[i], .filter = fk}, 9, std::move(c));

      const auto text = fuse::to_yaml(h);
      ASSERT_TRUE(text.ok());
      const auto back = fuse::from_yaml(text.value());
      ASSERT_TRUE(back.ok()) << back.error().message;
      const auto& d = back.value().iteration_region(fuse::IterationRegionIdx{0});
      EXPECT_EQ(d.convergence.measures()[0].kind, mk)
          << to_cstring(mk) << " came back as "
          << to_cstring(d.convergence.measures()[0].kind);
      EXPECT_EQ(d.acceleration.kind, accels[i]);
      EXPECT_EQ(d.acceleration.filter, fk);
      EXPECT_TRUE(structurally_equal(h, back.value()))
          << to_cstring(accels[i]) << " / " << to_cstring(fk);
    }
  }
}
