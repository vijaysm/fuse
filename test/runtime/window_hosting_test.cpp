// Unit 2b (D19) correction, measured against a real host MPI run of
// `parallel.window_runner_mpi.{2,3}` (2026-10-08): the unit test of the
// rule `WindowMemberConfig::export_consumer_hosted` replaces.
//
// This drives the public `Runner` API exactly as window_runner_test.cpp
// does, plus one TEST-ONLY accessor (`Runner::test_only_window_block_
// hosted()`, include/fuse/runtime/runner.hpp's own comment has the full
// account of why it exists) that reads back `RegionBlock::window_hosted`
// for a named channel -- state with no OTHER production-code reader outside
// `run_region()` itself.
//
// This rule has NO tier-0 oracle through `Runner::run()`'s own observable
// behaviour: `RegionBlock::window_hosted` is only ever CONSULTED
// (`run_region()`'s own `gate_by_hosted`) when `RegionRuntime::participants.
// size() > 1`, which requires `RunnerConfig::base_communicator` to be set --
// and a build with no MPI compiled in refuses that outright
// (`E-RT-TRANSPORT-NOMPI`, `Runner::Impl::attach_comm_graphs()`).
// `RunnerConfig::force_this_process` (used throughout this file) gives a
// tier-0 test a specific rank's own `this_process` with NO communicator at
// all, so `initialize()` reaches every hosting computation `resolve_regions
// ()` makes -- including the one this file checks -- without ever
// exercising the gate that consumes it downstream. That gate, and the real
// cross-rank agreement behind it, has no tier-0 oracle; its own proof is the
// host-only `parallel.window_runner_mpi.{2,3}` this defect was found in.
//
// THE MEASUREMENT THAT MOTIVATES THIS FILE (the lead's host run,
// 2026-10-08, `build-all`, MPI on): `parallel.window_runner_mpi.{2,3}` FAIL.
// Window 0, rank 0: `got.iterations_per_window[0] == 1` against a reference
// of 37 (windows 1..4: reference 35, 25, 15, 3 vs `got` 1 every time);
// `got.x_b_per_window[0] == 0.2`, B's own unmodified initial value, since no
// real exchange ever converged. Traced to `window_export_meta` (src/runtime/
// iteration.cpp) storing the EXPORTING member's own `WindowMemberConfig::
// hosted` where `run_region()`'s convergence gate needs the CONSUMING
// member's: `test/golden/window_two_member.plan` shows the a2b block's
// value arc is reached from A's export only through the compiled delivery
// chain that executes on B's own processor set, so on A's rank the a2b
// block was wrongly marked owned (reading A's own `hosted`, which is true
// on A's own rank) even though A's rank never writes that block's buffer at
// all -- and the mirror for b2a on B's rank. `WindowMemberConfig::
// export_consumer_hosted`'s own comment (runner_impl.hpp) has the complete
// account, including how the resulting 0/0 residual/reference pair reads as
// trivially converged.

#include <gtest/gtest.h>
#include <fuse/backend/serial_backend.hpp>

#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/schedule/window.hpp>

#include <memory>
#include <optional>
#include <string>

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
using fuse::RunnerConfig;
using fuse::WaveformIteration;
namespace u = fuse::units;

namespace {

// A trivial Component: register_fields() binds storage, advance() is never
// called by anything this file does (it stops at `Runner::initialize()`,
// never `run()`), so its body only needs to exist to satisfy the interface.
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

// Two window members A, B on DISJOINT single-rank processor sets -- A on
// rank 0, B on rank 1 -- the same shape window_runner_test.cpp's own
// `make_single_window_graph_with_disjoint_ranks()` uses for the hosting
// (step (a)) test, duplicated here rather than shared across the two
// translation units for the same reason resolve_regions()'s own comment
// gives for duplicating member_exported_field()/member_imported_field():
// file-local helpers are cheaper than a new shared header two tests would
// need to agree to depend on.
CouplingGraph two_member_window_graph_disjoint_ranks() {
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
  decl.acceleration.kind = fuse::AccelKind::None;
  g.add_iteration_region(decl);
  return g;
}

// Same two-channel shape, but A and B are placed on the SAME single rank
// (0) -- the "every existing window test in this suite" shape, where the
// hosting question this file is about has only one possible answer
// (everyone is hosted everywhere) and must stay that way.
CouplingGraph two_member_window_graph_shared_rank() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});

  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{0, 1}, .interval = Rational{1}});

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
  decl.acceleration.kind = fuse::AccelKind::None;
  g.add_iteration_region(decl);
  return g;
}

// FAN-OUT: one exporter, two consumers on DIFFERENT processor sets. A's
// own `a_out` feeds both B (rank 1) and C (rank 2), so the two blocks that
// carry it are measured on two different processor sets and cannot share
// one answer. A per-MEMBER hosting flag -- which is what
// `WindowMemberConfig::export_consumer_hosted` is, resolved from the FIRST
// channel in `decl.channels` whose source is this member's export -- gives
// both blocks A's single first-match answer, and is therefore wrong for one
// of them on every rank. Resolving per BLOCK from the block's own channel
// target (resolve_regions(), the `b.window_export_capacity > 0` branch) is
// what this case pins.
CouplingGraph fan_out_window_graph() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});
  const auto a = g.add_component(ComponentDecl{
      .name = "A", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "B", .ranks = RankSet{1, 1}, .interval = Rational{1}});
  const auto c = g.add_component(ComponentDecl{
      .name = "C", .ranks = RankSet{2, 1}, .interval = Rational{1}});
  const auto a_out = g.add_field(a, FieldSpec{.name = "a_out", .region = r,
                                              .units = u::kKelvin,
                                              .conservation = Conservation::Extensive,
                                              .intent = Intent::Provides});
  const auto b_in = g.add_field(b, FieldSpec{.name = "b_in", .region = r,
                                             .units = u::kKelvin,
                                             .conservation = Conservation::Extensive,
                                             .intent = Intent::Requires});
  const auto c_in = g.add_field(c, FieldSpec{.name = "c_in", .region = r,
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
  const auto to_b = g.add_channel("a2b", a_out, b_in);
  const auto to_c = g.add_channel("a2c", a_out, c_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the a-b coupling window's own sweep, not a real-time lag");
  CouplingWindowScope scope;
  scope.window_component = "A";
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Adjustable},
                    {"B", DtPolicy::Adjustable},
                    {"C", DtPolicy::Adjustable}};
  scope.iteration = WaveformIteration::Jacobi;
  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {to_b, to_c, back};
  decl.max_iterations = 50;
  // Each measure names the CONSUMER's own field, not the shared `a_out`:
  // a measure that matches more than one residual of the region is refused
  // (`E-ITER-MEASURE`, measured while writing this fixture), which is
  // exactly what naming the fanned-out source would do here.
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_in"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "c_in"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_in"});
  decl.window_scope = scope;
  decl.acceleration.kind = fuse::AccelKind::None;
  g.add_iteration_region(decl);
  return g;
}

/// Binds `A`/`B`'s own storage into `registry` and calls `runner.
/// initialize()` with `RunnerConfig::force_this_process` set to `rank` --
/// never `run()`, which this file has no need of and which a StubBlock
/// cannot usefully drive anyway. `runner` is already constructed (from the
/// same graph/dag/plan/backend the caller built) by the time this is
/// called; ownership of `components`/`registry` stays with the caller,
/// since `Runner::test_only_window_block_hosted()` is read off `runner`
/// after this returns and `runner` must outlive that call.
void initialize_forced(const CouplingGraph& g, int rank,
                       ComponentSet& components, FieldRegistry& registry,
                       fuse::Runner& runner) {
  auto a_owned = std::make_unique<StubBlock>("A");
  auto b_owned = std::make_unique<StubBlock>("B");
  components.add(std::move(a_owned));
  components.add(std::move(b_owned));
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  fuse::RunnerConfig cfg;
  cfg.force_this_process = rank;
  const fuse::Result<void> r = runner.initialize(components, registry, cfg);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

/// `initialize_forced` for the fan-out graph, which has a third component.
void initialize_forced_abc(const CouplingGraph& g, int rank,
                           ComponentSet& components, FieldRegistry& registry,
                           fuse::Runner& runner) {
  for (const char* name : {"A", "B", "C"}) {
    components.add(std::make_unique<StubBlock>(name));
  }
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

// THE RULE (must fail before the fix, when `window_export_meta` stored the
// EXPORTING member's own `hosted` instead of the CONSUMER's): on A's own
// rank (0), the a2b block (A's export, measured through the chain that
// executes on B's own processor set) is NOT owned here, and the b2a block
// (B's export, measured through the chain that executes on A's own
// processor set) IS.
TEST(WindowHosting, BlockOwnershipFollowsTheConsumersProcessorSetOnAsRank) {
  const CouplingGraph g = two_member_window_graph_disjoint_ranks();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  FieldRegistry registry;
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_NO_FATAL_FAILURE(
      initialize_forced(g, /*rank=*/0, components, registry, runner));

  const std::optional<bool> a2b = runner.test_only_window_block_hosted("a2b");
  const std::optional<bool> b2a = runner.test_only_window_block_hosted("b2a");
  ASSERT_TRUE(a2b.has_value());
  ASSERT_TRUE(b2a.has_value());

  EXPECT_FALSE(*a2b)
      << "a2b's own consumer is B, on rank 1 -- not hosted on A's own rank 0";
  EXPECT_TRUE(*b2a)
      << "b2a's own consumer is A, hosted right here on rank 0";
}

// The mirror on B's own rank (1): b2a is NOT owned (its consumer, A, is on
// rank 0), a2b IS (its consumer, B, is hosted right here).
TEST(WindowHosting, BlockOwnershipFollowsTheConsumersProcessorSetOnBsRank) {
  const CouplingGraph g = two_member_window_graph_disjoint_ranks();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  FieldRegistry registry;
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_NO_FATAL_FAILURE(
      initialize_forced(g, /*rank=*/1, components, registry, runner));

  const std::optional<bool> a2b = runner.test_only_window_block_hosted("a2b");
  const std::optional<bool> b2a = runner.test_only_window_block_hosted("b2a");
  ASSERT_TRUE(a2b.has_value());
  ASSERT_TRUE(b2a.has_value());

  EXPECT_TRUE(*a2b)
      << "a2b's own consumer is B, hosted right here on rank 1";
  EXPECT_FALSE(*b2a)
      << "b2a's own consumer is A, on rank 0 -- not hosted on B's own rank 1";
}

// The single-rank shape every OTHER window test in this suite uses: both
// members on rank 0, `force_this_process = 0`. Every block must stay
// owned -- this is the behaviour 168 tier-0 tests already pin indirectly
// by never failing, and this fix must not disturb it.
TEST(WindowHosting, EveryBlockIsOwnedWhenBothMembersShareOneRank) {
  const CouplingGraph g = two_member_window_graph_shared_rank();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  FieldRegistry registry;
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_NO_FATAL_FAILURE(
      initialize_forced(g, /*rank=*/0, components, registry, runner));

  const std::optional<bool> a2b = runner.test_only_window_block_hosted("a2b");
  const std::optional<bool> b2a = runner.test_only_window_block_hosted("b2a");
  ASSERT_TRUE(a2b.has_value());
  ASSERT_TRUE(b2a.has_value());

  EXPECT_TRUE(*a2b) << "a2b must be owned when every member shares one rank";
  EXPECT_TRUE(*b2a) << "b2a must be owned when every member shares one rank";
}

// One exporter, two consumers, three processor sets: on B's own rank the
// a2b block is owned and the a2c block is NOT, even though both carry the
// SAME export arc of the same member. Fails if a window block's ownership
// is resolved per member rather than per channel.
TEST(WindowHosting, FanOutBlocksFollowTheirOwnConsumerNotTheFirstChannel) {
  const CouplingGraph g = fan_out_window_graph();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{0})).ok());
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  FieldRegistry registry;
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_NO_FATAL_FAILURE(
      initialize_forced_abc(g, /*rank=*/1, components, registry, runner));

  const std::optional<bool> a2b = runner.test_only_window_block_hosted("a2b");
  const std::optional<bool> a2c = runner.test_only_window_block_hosted("a2c");
  ASSERT_TRUE(a2b.has_value());
  ASSERT_TRUE(a2c.has_value());
  EXPECT_TRUE(*a2b) << "a2b's own consumer is B, hosted right here on rank 1";
  EXPECT_FALSE(*a2c)
      << "a2c carries the same export arc as a2b, but ITS consumer is C on "
         "rank 2 -- this rank never writes that block and must not claim it";
}
