// The window-Newton seam's own multi-rank SHAPE,
// pinned the SAME way test/runtime/window_hosting_test.cpp already pins
// Picard's identical `WindowMemberConfig::export_consumer_hosted` rule:
// via `RunnerConfig::force_this_process`, which gives a tier-0 test a
// specific rank's own `this_process` with NO communicator at all, so
// `initialize()` reaches every hosting/exclusion computation
// `resolve_regions()` makes for a window-Newton region -- including the
// `b.window_hosted`-gated `newton_offset` exclusion -- without ever exercising
// the real cross-rank reduction downstream. This file's own tests stop at
// `initialize()`/`window_region_fixed_ point_residual()`'s own STATIC shape,
// exactly as window_hosting_test. cpp's header comment explains for Picard:
// the real agreement behind `ParticipantSetReducer` has no tier-0 oracle, and
// its own proof is the host-only `parallel.window_newton_mpi.{2,3}` row
// (test/parallel/window_newton_mpi_test.cpp).
//
// MEASURED while writing this file (not assumed): `RegionRuntime::
// participants` (iteration.cpp's own construction) is populated with
// more than the empty set ONLY when `RunnerConfig::base_communicator`
// has a value -- under `force_this_process` ALONE (no communicator,
// this file's own shape) it stays EMPTY regardless of how many ranks
// the graph's own RankSets name, which is exactly why the `newton_
// offset` exclusion below is gated on `RegionBlock::window_hosted`
// DIRECTLY, never on `R.participants.size()` (`resolve_regions()`'s
// own comment on that exclusion has the full account) -- the first
// version of this file gated on participant count too, by analogy with
// `run_region()`'s own Picard-path `gate_by_hosted`, and every
// EXPECT_FALSE/EXPECT_EQ below failed against the real binary until
// that analogy was dropped.
//
// THE FIXTURE: window_hosting_test.cpp's own `two_member_window_graph_
// disjoint_ranks()`/`StubBlock`/`initialize_forced()`, reused verbatim
// (not shared via a header -- the same "file-local helpers are cheaper
// than a new shared header two tests would need to agree to depend on"
// reasoning resolve_regions()'s own comment gives, which window_hosting_
// test.cpp already cites for the identical duplication), with `decl.
// acceleration.kind` switched from `AccelKind::None` to `AccelKind::
// Newton` and a default `NewtonConfig` (no block-Jacobi preconditioner
// requested, so `StubBlock`'s own `kCapabilityRollback`-only capability
// set needs nothing more -- `verify_newton_capability_support()`'s own
// gate is a no-op when neither `requires_residual_capability` nor
// `requires_linearization_capability` is set, and a window-scoped
// Newton region's own `NewtonResidualPath::Advance` -- the only path a
// window region may select, `E-NEWTON-DIRECT-WINDOW` refuses `Direct` --
// never calls `ImplicitComponent::evaluate_residual()` at all, driving
// `advance_window()` directly instead).

#include <gtest/gtest.h>
#include <fuse/backend/serial_backend.hpp>

#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/schedule/window.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>

using fuse::AccelKind;
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
using fuse::NewtonConfig;
using fuse::Rational;
using fuse::Region;
using fuse::RankSet;
using fuse::RunnerConfig;
using fuse::WaveformIteration;
namespace u = fuse::units;

namespace {

// window_hosting_test.cpp's own StubBlock, duplicated: advance()/save_
// state()/restore_state() are never exercised by anything this file
// does either (it stops at initialize(), never run()).
class StubBlock final : public fuse::Component {
 public:
  explicit StubBlock(std::string name) : name_(std::move(name)) {}
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
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

 private:
  std::string name_;
  double out_ = 0.0;
  double in_ = 0.0;
};

// window_hosting_test.cpp's own two_member_window_graph_disjoint_ranks(),
// with the accelerator switched to Newton. A on rank 0, B on rank 1.
CouplingGraph two_member_window_newton_graph_disjoint_ranks() {
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
  scope.window_component = "A";
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable}, {"B", DtPolicy::Adjustable}};
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
  // The one line that differs from window_hosting_
  // test.cpp's own fixture. Default NewtonConfig -- no block-Jacobi
  // preconditioner, no capability requirement StubBlock cannot meet.
  decl.acceleration.kind = AccelKind::Newton;
  decl.newton_config = NewtonConfig{};
  g.add_iteration_region(decl);
  return g;
}

void initialize_forced(const CouplingGraph& g, std::optional<int> rank,
                       ComponentSet& components, FieldRegistry& registry,
                       fuse::Runner& runner) {
  components.add(std::make_unique<StubBlock>("A"));
  components.add(std::make_unique<StubBlock>("B"));
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  fuse::RunnerConfig cfg;
  cfg.force_this_process = rank;
  const fuse::Result<void> r = runner.initialize(components, registry, cfg);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

}  // namespace

// THE RULE THIS FILE PINS: on A's own rank (0), the window-Newton
// interface vector excludes the a2b block (A's export, consumed by B on
// rank 1 -- not hosted here) and includes b2a (B's export, consumed by
// A, hosted right here) -- `RegionBlock::newton_offset`'s own NEW
// `window_hosted` gate (resolve_regions(), src/runtime/iteration.cpp),
// read back through `Runner::window_region_fixed_point_residual()`'s own
// `channel_names`/`excluded_channel_names`.
TEST(WindowNewtonHosting, LocalNewtonVectorKeepsOnlyTheBlockThisRankHostsOnAsRank) {
  const CouplingGraph g = two_member_window_newton_graph_disjoint_ranks();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  FieldRegistry registry;
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_NO_FATAL_FAILURE(
      initialize_forced(g, /*rank=*/0, components, registry, runner));

  const fuse::Result<fuse::WindowFixedPointResidual> wfpr =
      runner.window_region_fixed_point_residual(0, components);
  ASSERT_TRUE(wfpr.ok()) << wfpr.error().code << ": " << wfpr.error().message;

  const auto& included = wfpr.value().channel_names;
  const auto& excluded = wfpr.value().excluded_channel_names;
  ASSERT_EQ(included.size(), 1u);
  ASSERT_EQ(excluded.size(), 1u);
  EXPECT_EQ(included.front(), "b2a")
      << "b2a's own consumer is A, hosted right here on rank 0";
  EXPECT_EQ(excluded.front(), "a2b")
      << "a2b's own consumer is B, on rank 1 -- not hosted on A's own rank 0";
  EXPECT_EQ(wfpr.value().n, wfpr.value().lengths.front())
      << "R.newton_n must equal exactly this rank's own hosted block length";
  EXPECT_LT(wfpr.value().n, wfpr.value().n + wfpr.value().excluded_lengths.front())
      << "the excluded block's own length must be strictly positive, or "
         "this test could pass with nothing actually excluded";
}

// The mirror on B's own rank (1): a2b included, b2a excluded.
TEST(WindowNewtonHosting, LocalNewtonVectorKeepsOnlyTheBlockThisRankHostsOnBsRank) {
  const CouplingGraph g = two_member_window_newton_graph_disjoint_ranks();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  FieldRegistry registry;
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_NO_FATAL_FAILURE(
      initialize_forced(g, /*rank=*/1, components, registry, runner));

  const fuse::Result<fuse::WindowFixedPointResidual> wfpr =
      runner.window_region_fixed_point_residual(0, components);
  ASSERT_TRUE(wfpr.ok()) << wfpr.error().code << ": " << wfpr.error().message;

  const auto& included = wfpr.value().channel_names;
  const auto& excluded = wfpr.value().excluded_channel_names;
  ASSERT_EQ(included.size(), 1u);
  ASSERT_EQ(excluded.size(), 1u);
  EXPECT_EQ(included.front(), "a2b")
      << "a2b's own consumer is B, hosted right here on rank 1";
  EXPECT_EQ(excluded.front(), "b2a")
      << "b2a's own consumer is A, on rank 0 -- not hosted on B's own rank 1";
  EXPECT_EQ(wfpr.value().n, wfpr.value().lengths.front());
}

// `RegionRuntime::participants` is populated ONLY
// when `RunnerConfig::base_communicator` has a value (iteration.cpp's
// own comment on that construction) -- with NEITHER a communicator NOR
// `force_this_process` configured, a window-Newton region whose graph
// still names two different nominal ranks is run "purely local," on
// the one process that exists, exactly as the Picard path
// is: `R.participants` stays empty, `R.newton_reducer` is the
// serial reducer, and -- because `window_hosted` defaults to `true`
// whenever `this_process` has no value at all -- EVERY block stays
// included, nothing is excluded. This is the SHAPE the
// E-RT-REGION-MPI exemption (iteration.cpp, `AccelKind::Newton` alongside
// `None`/`Constant`) and `check_region_supports_newton()`
// (fixed_point_residual.cpp) must both let through
// without error -- proven here by `initialize()` succeeding and by
// `n` covering BOTH blocks, not a narrowed subset.
TEST(WindowNewtonHosting, NoCommunicatorAndNoForcedRankRunsPurelyLocalWithNothingExcluded) {
  const CouplingGraph g = two_member_window_newton_graph_disjoint_ranks();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  FieldRegistry registry;
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_NO_FATAL_FAILURE(
      initialize_forced(g, /*rank=*/std::nullopt, components, registry, runner));

  const fuse::Result<fuse::WindowFixedPointResidual> wfpr =
      runner.window_region_fixed_point_residual(0, components);
  ASSERT_TRUE(wfpr.ok()) << wfpr.error().code << ": " << wfpr.error().message;

  EXPECT_EQ(wfpr.value().channel_names.size(), 2u)
      << "with no this_process at all, every block defaults to hosted";
  EXPECT_TRUE(wfpr.value().excluded_channel_names.empty());
}
