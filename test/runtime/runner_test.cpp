// Explicit, not transitive: the feature macros are read below (docs/dev/invariants.md).
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../support/allocation_counter.hpp"

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::SerialMeshBackend;
using fuse::StepContext;
using fuse::Transport;
using fuse::TransportRegistry;
using fuse::TransportSpec;
namespace u = fuse::units;

namespace {

/// A one-measure ConvergenceCriterion. `fuse::Converged::Relative(...)`
/// returns a bare `ConvergenceMeasure`, which `ConvergenceCriterion` does
/// not implicitly construct from (test/runtime/checkpoint_test.cpp's
/// `one_measure()` is the same pattern for the same reason).
fuse::ConvergenceCriterion one_measure(double tol, std::string field) {
  fuse::ConvergenceCriterion c;
  c.add(fuse::Converged::Relative(tol, std::move(field)));
  return c;
}

/// How the producer allocates, fills and registers its array. The defaults
/// reproduce the plain case; the other fields exist so the tests can register
/// storage that does *not* match the plan and check the Runner refuses it
/// rather than overreading it.
struct ProducerOpts {
  std::size_t cells = 8;
  double      value = 2.0;
  bool        ramp = false;      // data[i] = value*step + i, not a constant
  std::size_t bind_count = 0;    // 0 -> cells
  std::size_t bind_stride = 1;   // > 1 registers a strided view
  bool        poison = false;    // write NaN instead of a number
  /// An iteration region requires every component it
  /// checkpoints to support rollback (E-ITER-NOROLLBACK); default false
  /// matches every existing caller of this harness, none of which puts this
  /// component in a region.
  bool        rollback = false;
  /// Refuse every advance() instead of running it. Used to
  /// put a real Failure on the re-advance path inside an iteration region's
  /// sweep loop.
  bool        fail_advance = false;
  /// The harness's single producer
  /// is always named "producer"; a test that needs a SECOND, independent
  /// producer/channel in the same graph sets this to a distinct name.
  std::string name = "producer";
};

/// Producer writes a value; consumer just holds what it receives.
class Producer final : public fuse::Component {
 public:
  explicit Producer(ProducerOpts o)
      : opts_(o),
        // Only as much storage as it registers: a short registration must be a
        // genuinely short array, or the overread it invites is in bounds by
        // accident and ASan sees nothing.
        data_((o.bind_count == 0 ? o.cells : o.bind_count) * o.bind_stride,
              o.value) {}

  [[nodiscard]] std::string name() const override { return opts_.name; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(opts_.name);
    const std::size_t count =
        opts_.bind_count == 0 ? opts_.cells : opts_.bind_count;
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(
          f, fuse::FieldRef{data_.data(),
                            fuse::Layout{.stride = opts_.bind_stride,
                                         .count = count},
                            fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext&) override {
    // Fails starting from the SECOND call, so the first (establishing
    // y_0, before any region sweep) succeeds and only a RE-advance fails.
    if (opts_.fail_advance && steps > 0) {
      return fuse::Failure{"E-TEST-ADVANCE-REFUSED",
                          "producer refuses to advance (test-injected)"};
    }
    ++steps;
    for (std::size_t i = 0; i < data_.size(); ++i) {
      if (opts_.poison) {
        data_[i] = std::numeric_limits<double>::quiet_NaN();
      } else if (opts_.ramp) {
        data_[i] =
            opts_.value * static_cast<double>(steps) + static_cast<double>(i);
      } else {
        data_[i] = opts_.value * static_cast<double>(steps);
      }
    }
    return fuse::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }
  void poison() { opts_.poison = true; }

  // Opt-in rollback, exercised only by a harness that puts
  // this component in an iteration region. `saved_` is a byte-for-byte copy
  // because rank-agreement.md's E-ITER-ROLLBACK check compares exactly that.
  fuse::Result<void> save_state() override {
    if (opts_.rollback) saved_ = data_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    if (opts_.rollback) data_ = saved_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return opts_.rollback ? fuse::kCapabilityRollback : 0;
  }

  int steps = 0;

 private:
  ProducerOpts        opts_;
  std::vector<double> data_;
  std::vector<double> saved_;
};

class Consumer final : public fuse::Component {
 public:
  /// `rollback`: see Producer::rollback_ -- same opt-in,
  /// same reason, default false to match every existing caller.
  /// `name`: a second, independent
  /// consumer in the same graph needs a name distinct from "consumer".
  explicit Consumer(std::size_t n, bool rollback = false,
                    std::string name = "consumer")
      : data_(n, 0.0), rollback_(rollback), name_(std::move(name)) {
    seen_.reserve(64);
    seen_at_.reserve(64);
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext& ctx) override {
    ++steps;
    // seen_ is reserved in the constructor: a push_back that reallocated would
    // show up in AllocatesNothingWhileRunning as the Runner's fault.
    seen_.push_back(data_.empty() ? 0.0 : data_[0]);
    seen_at_.push_back(ctx.t_begin);
    return fuse::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }
  /// data_[0] as it looked at the start of each advance(): what the component
  /// actually integrated against.
  [[nodiscard]] const std::vector<double>& seen() const { return seen_; }
  /// ... and the absolute time that advance began at, so a test can ask
  /// whether what was integrated against was causally available by then.
  [[nodiscard]] const std::vector<Rational>& seen_at() const {
    return seen_at_;
  }

  fuse::Result<void> save_state() override {
    if (rollback_) saved_ = data_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    if (rollback_) data_ = saved_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return rollback_ ? fuse::kCapabilityRollback : 0; }

  int steps = 0;

 private:
  std::vector<double>   data_;
  std::vector<double>   seen_;
  std::vector<Rational> seen_at_;
  bool                  rollback_;
  std::vector<double>   saved_;
  std::string           name_;
};

/// A component that participates in nothing: it declares one `Provides` field
/// no channel reads, and exists only to put a THIRD interval into the clock
/// set. That matters more than it sounds -- the common period is the lcm over
/// ALL clocks, so a bystander can make the period a proper multiple of
/// `lcm(src_dt, dst_dt)` and put source and target boundaries on the same
/// instant strictly inside a period, which two components alone cannot do.
class Bystander final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "bystander"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("bystander");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> data_ = std::vector<double>(8, 0.0);
};

/// A component the graph knows nothing about.
class Stowaway final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "stowaway"; }
  fuse::Result<void> register_fields(FieldRegistry&,
                                     const CouplingGraph&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }
};

/// The RunnerConfig::transport_setup hook, exercised without dragging
/// in MPI or LowFive. A local-transport CopyTransport lookalike, registered
/// under its own name so it never shadows "copy" in the process-wide
/// TransportRegistry.
class ProbeTransport final : public Transport {
 public:
  fuse::Result<void> prepare(const TransportSpec&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> move(const double* src, double* dst,
                          std::size_t n) override {
    std::copy(src, src + n, dst);
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::string name() const override { return "probe"; }
  [[nodiscard]] bool is_local() const override { return true; }
};

struct Harness {
  CouplingGraph       graph;
  SerialMeshBackend   backend;
  fuse::OperatorDAG   dag;
  fuse::ExecutionPlan plan;
  ComponentSet        components;
  FieldRegistry       registry;

  [[nodiscard]] Producer* producer() {
    return dynamic_cast<Producer*>(components.find("producer"));
  }
  [[nodiscard]] Consumer* consumer() {
    return dynamic_cast<Consumer*>(components.find("consumer"));
  }
  /// The output arc of the node lowering gave this label.
  [[nodiscard]] fuse::ArcId arc_of(std::string_view label) const {
    for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
      const fuse::Node& n = dag.node(fuse::NodeId{i});
      if (n.label == label) return n.output;
    }
    return fuse::ArcId{};
  }
};

struct HarnessOpts {
  fuse::AccumOp accum = fuse::AccumOp::Instant;
  Rational      src_dt{1};
  Rational      dst_dt{1};
  /// Entities in each region; also the length of each component's array.
  std::size_t   src_cells = 8;
  std::size_t   dst_cells = 8;
  /// Empty means "generated:line:<cells>" at dimension 1.
  std::string   src_mesh;
  std::string   dst_mesh;
  int           src_dim = 1;
  int           dst_dim = 1;
  fuse::Conservation conservation = fuse::Conservation::None;
  ProducerOpts  producer{};
  bool          iteration_region = false;
  /// Empty (the default) reproduces the existing behaviour: an iteration
  /// region with no convergence measure, which `initialize()` refuses with
  /// E-ITER-NOCONVERGE. A caller that wants a region that actually runs sets
  /// this to a real criterion.
  fuse::ConvergenceCriterion region_convergence;
  /// The region's accelerator. `Constant` (the AccelConfig default) is what
  /// every existing test declares; `Aitken` and the IQN family are the kinds
  /// resolve_regions() refuses on a multi-rank region under MPI.
  fuse::AccelKind region_acceleration = fuse::AccelKind::Constant;
  /// Non-zero adds a `Bystander` on this interval, which only widens the
  /// common period. See the class comment for why a test would want that.
  Rational      bystander_dt{0};
  /// How many ranks the consumer is declared on. 1 (the default) reproduces
  /// every existing caller. A caller that wants the channel's target on more
  /// than one DECLARED rank, with no `RunnerConfig::base_communicator` set, is
  /// the same Tier-0-legitimate "nominal placement, one real process" case
  /// resolve_regions() already relies on
  /// (StillRunsAnIterationRegionWithNoCommunicatorConfigured).
  int           dst_rank_count = 1;
  /// When set, the channel's Remap is routed
  /// through a `ComponentKind::Coupler` placed on a rank neither the
  /// producer nor the consumer occupies, via `remap_on`. No
  /// `RunnerConfig::base_communicator` is configured anywhere in this file
  /// (the class comment on `dst_rank_count` above explains why that is
  /// legitimate here), so the coupler's nominal placement never becomes a
  /// second real process -- what this actually exercises is that a
  /// `ComponentSet` with NO Component supplied for the coupler still
  /// initializes and runs, and that the two extra Transport/Epoch hops
  /// deliver the identical value a direct producer->consumer channel would.
  bool          with_coupler = false;
};

std::unique_ptr<Harness> build(HarnessOpts o) {
  auto h = std::make_unique<Harness>();
  auto& g = h->graph;

  const std::string src_source =
      o.src_mesh.empty() ? "generated:line:" + std::to_string(o.src_cells)
                         : o.src_mesh;
  const std::string dst_source =
      o.dst_mesh.empty() ? "generated:line:" + std::to_string(o.dst_cells)
                         : o.dst_mesh;

  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = src_source});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = dst_source});
  const auto r1 = g.add_region(
      fuse::Region{.name = "r1", .mesh = m1, .dimension = o.src_dim});
  const auto r2 = g.add_region(
      fuse::Region{.name = "r2", .mesh = m2, .dimension = o.dst_dim});
  const auto p = g.add_component(fuse::ComponentDecl{
      .name = "producer", .ranks = fuse::RankSet{0, 1},
      .interval = o.src_dt});
  const auto c = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = fuse::RankSet{1, o.dst_rank_count},
      .interval = o.dst_dt});
  const auto s = g.add_field(
      p, fuse::FieldSpec{.name = "out", .region = r1, .units = u::kKelvin,
                         .conservation = o.conservation,
                         .intent = fuse::Intent::Provides});
  const auto t = g.add_field(
      c, fuse::FieldSpec{.name = "in", .region = r2, .units = u::kKelvin,
                         .conservation = o.conservation,
                         .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", s, t);
  if (o.with_coupler) {
    const auto coupler = g.add_coupler(
        {.name = "coupler", .ranks = fuse::RankSet{5, 1}});
    g.channel(ch).remap_on(coupler);
  }
  if (o.accum != fuse::AccumOp::Instant) g.channel(ch).accumulate(o.accum);
  const bool with_bystander = o.bystander_dt > Rational{0};
  if (with_bystander) {
    const auto m3 =
        g.add_mesh(fuse::MeshDecl{.name = "m3", .source = dst_source});
    const auto r3 = g.add_region(
        fuse::Region{.name = "r3", .mesh = m3, .dimension = o.dst_dim});
    const auto bc = g.add_component(fuse::ComponentDecl{
        .name = "bystander", .ranks = fuse::RankSet{2, 1},
        .interval = o.bystander_dt});
    g.add_field(bc, fuse::FieldSpec{.name = "idle", .region = r3,
                                    .units = u::kKelvin,
                                    .intent = fuse::Intent::Provides});
    EXPECT_TRUE(h->backend.load(g.mesh(m3)).ok());
  }
  if (o.iteration_region) {
    g.add_iteration_region(fuse::IterationRegionDecl{
        .name = "outer", .channels = {ch}, .max_iterations = 12,
        .convergence = o.region_convergence,
        .acceleration = fuse::AccelConfig{.kind = o.region_acceleration}});
  }

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const fuse::ClockSet clocks{g};
  auto sched =
      schedule(g, h->dag, clocks, fuse::make_region_sizes(g, h->dag, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  ProducerOpts po = o.producer;
  po.cells = o.src_cells;
  // A region with a real (non-empty) convergence
  // criterion re-advances its components every sweep, which requires
  // rollback (E-ITER-NOROLLBACK). A region left at the default empty
  // criterion -- the existing RefusesAnIterationRegionWithNoConvergenceMeasure
  // test -- is refused before rollback is ever checked, so leaving rollback
  // off there reproduces that test's exact prior behaviour.
  const bool needs_rollback =
      o.iteration_region && !o.region_convergence.measures().empty();
  po.rollback = needs_rollback;
  h->components.add(std::make_unique<Producer>(po));
  h->components.add(std::make_unique<Consumer>(o.dst_cells, needs_rollback));
  if (with_bystander) h->components.add(std::make_unique<Bystander>());
  return h;
}

/// Matching 8-cell meshes so the identity Remap is meaningful.
std::unique_ptr<Harness> make_harness(fuse::AccumOp accum, Rational src_dt,
                                      Rational dst_dt) {
  return build(HarnessOpts{.accum = accum, .src_dt = src_dt, .dst_dt = dst_dt});
}

/// Bind every component's storage; the Runner refuses a plan otherwise.
void bind_all(Harness& h) {
  for (auto& c : h.components) {
    ASSERT_TRUE(c.register_fields(h.registry, h.graph).ok());
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Core runner tests.
// ---------------------------------------------------------------------------

TEST(Runner, InitializesAndAllocatesBuffers) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  EXPECT_EQ(runner.stats().periods, 0);
}

TEST(Runner, RejectsUnboundFields) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  // Deliberately skip register_fields.
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-UNBOUND");
}

// The first coupled exchange.
TEST(Runner, TransfersAValueFromProducerToConsumer) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());

  const Consumer* consumer = h->consumer();
  ASSERT_NE(consumer, nullptr);
  for (const double v : consumer->data()) {
    EXPECT_DOUBLE_EQ(v, 2.0) << "producer wrote 2.0 on its first step";
  }
}

// ---- coupler placement (design C) --------------------------------------

// A coupler declared on the channel's remap_on, placed on a rank neither
// the producer nor the consumer occupies. No Component is supplied for it
// (build() never adds one when with_coupler is set) -- build_ticks() must
// not require one -- and the delivered value must be bit-for-bit identical
// to the direct producer->consumer channel: source -> C -> target, with
// every hop an ordinary local copy since no RunnerConfig::base_communicator
// is configured anywhere in this harness.
TEST(Runner, ACouplerNeedsNoComponentAndDeliversTheIdenticalValue) {
  auto plain = build(HarnessOpts{});
  Runner plain_runner{plain->graph, plain->dag, plain->plan, plain->backend};
  bind_all(*plain);
  ASSERT_TRUE(plain_runner.initialize(plain->components, plain->registry).ok());
  ASSERT_TRUE(plain_runner.run_period(plain->components).ok());
  ASSERT_TRUE(plain_runner.run_period(plain->components).ok());

  auto coupled = build(HarnessOpts{.with_coupler = true});
  Runner coupled_runner{coupled->graph, coupled->dag, coupled->plan, coupled->backend};
  bind_all(*coupled);
  ASSERT_TRUE(coupled_runner.initialize(coupled->components, coupled->registry).ok())
      << "a Coupler-kind component must not need a matching Component";
  ASSERT_TRUE(coupled_runner.run_period(coupled->components).ok());
  ASSERT_TRUE(coupled_runner.run_period(coupled->components).ok());

  const Consumer* plain_consumer = plain->consumer();
  const Consumer* coupled_consumer = coupled->consumer();
  ASSERT_NE(plain_consumer, nullptr);
  ASSERT_NE(coupled_consumer, nullptr);
  ASSERT_EQ(plain_consumer->data().size(), coupled_consumer->data().size());
  for (std::size_t i = 0; i < plain_consumer->data().size(); ++i) {
    EXPECT_EQ(plain_consumer->data()[i], coupled_consumer->data()[i])
        << "entity " << i << ": routing through a coupler must not change "
                              "the delivered value, assert exactly (no tolerance)";
  }
}

// The coupler's own placement gets no Tick at all: it is never due, so a
// Component supplied under its name (if one somehow were) would never see
// advance() called. Checked indirectly here via runner.stats(): the period
// count advances normally, which it would not if build_ticks() had tried
// and failed to schedule a tick for a component with no interval-driven
// physics.
TEST(Runner, ACouplerContributesNoTicks) {
  auto h = build(HarnessOpts{.with_coupler = true});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_EQ(runner.stats().periods, 1);
}

TEST(Runner, AdvancesEveryComponentOncePerPeriodInLockstep) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 3).ok());

  EXPECT_EQ(h->producer()->steps, 3);
  EXPECT_EQ(h->consumer()->steps, 3);
  EXPECT_EQ(runner.stats().periods, 3);
}

TEST(Runner, SubcyclingAdvancesTheFastComponentMoreOften) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1, 4}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 1).ok());

  EXPECT_EQ(h->producer()->steps, 4);
  EXPECT_EQ(h->consumer()->steps, 1);
}

TEST(Runner, MeanAccumulationAveragesOverTheWindow) {
  // Producer writes 2*step: over 4 substeps that is 2,4,6,8 -> mean 5.
  auto h = make_harness(fuse::AccumOp::Mean, Rational{1, 4}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());

  const Consumer* consumer = h->consumer();
  ASSERT_NE(consumer, nullptr);
  EXPECT_NEAR(consumer->data()[0], 5.0, 1e-12);
}

TEST(Runner, CountsExecutedInstructions) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_EQ(runner.stats().instructions_executed,
            static_cast<std::int64_t>(h->plan.instructions().size()));
}

TEST(Runner, DetectsNonFiniteValues) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  RunnerConfig cfg;
  cfg.abort_on_violation = true;
  ASSERT_TRUE(runner.initialize(h->components, h->registry, cfg).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_EQ(runner.stats().conservation_violations, 0);
}

// ---------------------------------------------------------------------------
// The milestone, with values that vary per element.
//
// The test above writes one constant over the whole mesh, so an interpreter
// that broadcast element 0, reversed the array, or copied only the first cell
// would pass it. This one writes a distinct value per cell.
// ---------------------------------------------------------------------------

TEST(Runner, TransfersAPerElementProfileNotJustAConstant) {
  auto h = build(HarnessOpts{.producer = ProducerOpts{.ramp = true}});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());

  // Step 1 of the ramp is 2*1 + i.
  const std::vector<double>& got = h->consumer()->data();
  ASSERT_EQ(got.size(), 8u);
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_DOUBLE_EQ(got[i], 2.0 + static_cast<double>(i))
        << "cell " << i << " must carry its own value, not its neighbour's";
  }
  // The producer's own array is untouched by the exchange.
  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(h->producer()->data()[i], 2.0 + static_cast<double>(i));
  }
}

TEST(Runner, ASecondPeriodDeliversTheSecondStepsValues) {
  auto h = build(HarnessOpts{.producer = ProducerOpts{.ramp = true}});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 2).ok());

  const std::vector<double>& got = h->consumer()->data();
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_DOUBLE_EQ(got[i], 4.0 + static_cast<double>(i));
  }
  // The exchange is explicit and one period lagged: the consumer's advance in
  // period p integrates against what period p-1 delivered. Pinning this makes
  // an accidental change of the tick convention visible.
  ASSERT_EQ(h->consumer()->seen().size(), 2u);
  EXPECT_DOUBLE_EQ(h->consumer()->seen()[0], 0.0);
  EXPECT_DOUBLE_EQ(h->consumer()->seen()[1], 2.0);
}

// ---------------------------------------------------------------------------
// Spec Section 4.2: who owns which buffer.
// ---------------------------------------------------------------------------

TEST(Runner, ComponentOwnedArcsAliasComponentStorageAndAreNotAllocated) {
  auto h = build(HarnessOpts{});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());

  const fuse::ArcId src = h->arc_of("producer.out");
  const fuse::ArcId dst = h->arc_of("consumer.in");
  ASSERT_TRUE(src.valid());
  ASSERT_TRUE(dst.valid());

  const auto src_buf = runner.buffer(src);
  const auto dst_buf = runner.buffer(dst);
  ASSERT_TRUE(src_buf.ok());
  ASSERT_TRUE(dst_buf.ok());

  // The plan says these two are component-owned...
  const auto& decls = h->plan.buffers().buffers;
  EXPECT_TRUE(decls.at(src.value()).is_component_owned);
  EXPECT_TRUE(decls.at(dst.value()).is_component_owned);
  // ...so the Runner must hold the component's own address, not a shadow copy.
  // A shadow copy is silent: the component never sees the coupled values.
  EXPECT_EQ(src_buf.value(), h->producer()->data().data());
  EXPECT_EQ(dst_buf.value(), h->consumer()->data().data());
}

TEST(Runner, CouplerOwnedArcsAreAllocatedAndAliasNoComponent) {
  auto h = build(HarnessOpts{});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());

  const double* producer_base = h->producer()->data().data();
  const double* consumer_base = h->consumer()->data().data();

  std::size_t coupler_arcs = 0;
  std::vector<const double*> seen;
  for (const fuse::BufferDecl& b : h->plan.buffers().buffers) {
    const auto buf = runner.buffer(b.arc);
    ASSERT_TRUE(buf.ok());
    ASSERT_NE(buf.value(), nullptr) << "arc %" << b.arc.value();
    if (!b.is_component_owned) {
      ++coupler_arcs;
      // Writing through an intermediate must never reach a component array.
      EXPECT_NE(buf.value(), producer_base);
      EXPECT_NE(buf.value(), consumer_base);
    }
    // Distinct arcs never share storage. This is what makes `is_persistent`
    // safe with no extra machinery: nothing is recycled inside a period.
    for (const double* other : seen) EXPECT_NE(buf.value(), other);
    seen.push_back(buf.value());
  }
  // Four, not three: transport, the rank-agreement point that follows it,
  // remap, and diagnose staging. The agreement point costs a buffer because
  // the plan has no notion of an operator whose output aliases its input, so
  // it copies -- a known cost, documented in docs/dev/rank-agreement.md, and
  // the fix is in-place operators, not removing the agreement.
  EXPECT_EQ(coupler_arcs, 4u)
      << "transport, its agreement point, remap and diagnose staging";
  EXPECT_EQ(h->plan.buffers().total_coupler_elements, 32u);
}

TEST(Runner, RefusesAFieldRegisteredShorterThanItsRegion) {
  // 8-cell region, 4 registered values. Every kernel reads the region's
  // length, so accepting this is a heap overread with no bounds check left.
  auto h = build(HarnessOpts{.producer = ProducerOpts{.bind_count = 4}});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-SIZE");
}

TEST(Runner, RefusesAFieldRegisteredWithANonContiguousLayout) {
  // The interpreter walks every buffer as a flat double[]; a stride of 2 would
  // be read as if it were contiguous, silently mixing in the gaps.
  auto h = build(HarnessOpts{.producer = ProducerOpts{.bind_stride = 2}});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-LAYOUT");
}

// ---------------------------------------------------------------------------
// Spec Section 2 layer 3: no allocation in the hot loop.
// ---------------------------------------------------------------------------

TEST(Runner, AllocatesNothingWhileRunning) {
  // Subcycled, accumulating and conservation-checked: Sample, Accumulate,
  // Transport, Remap, Constrain, Diagnose and Sink all run.
  auto h = build(HarnessOpts{.accum = fuse::AccumOp::Mean,
                             .src_dt = Rational{1, 4},
                             .conservation = fuse::Conservation::Extensive});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());

  // One warm-up period outside the counter, so something allocated lazily
  // exactly once would still be caught by the counted periods below.
  ASSERT_TRUE(runner.run_period(h->components).ok());

  std::size_t allocations = 0;
  bool        ok = false;
  {
    const AllocationCounter counter;
    ok = runner.run(h->components, 8).ok();
    allocations = counter.count();
  }
  ASSERT_TRUE(ok);
  EXPECT_EQ(allocations, 0u)
      << "run() allocated " << allocations
      << " times; spec Section 2 layer 3 requires the interpreted loop to allocate "
         "nothing";
}

TEST(Runner, AllocationCounterActuallyCounts) {
  // A check that cannot fail proves nothing; show the counter is live.
  std::size_t allocations = 0;
  {
    const AllocationCounter counter;
    std::vector<double> v(1024, 1.0);
    allocations = counter.count();
    EXPECT_DOUBLE_EQ(v[0], 1.0);
  }
  EXPECT_GE(allocations, 1u);
}

// ---------------------------------------------------------------------------
// Persistence across periods.
// ---------------------------------------------------------------------------

TEST(Runner, PersistentAccumulatorIsNotContaminatedAcrossPeriods) {
  // Substeps 1..4 give 2,4,6,8 (mean 5); 5..8 give 10,12,14,16 (mean 13);
  // 9..12 give 18,20,22,24 (mean 21). A sample carried over, or a sample
  // counter that was not reset, shows up here and nowhere else.
  auto h = make_harness(fuse::AccumOp::Mean, Rational{1, 4}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());

  const fuse::ArcId sample = h->arc_of("link/sample");
  ASSERT_TRUE(sample.valid());
  EXPECT_TRUE(h->plan.buffers().buffers.at(sample.value()).is_persistent)
      << "the waveform accumulator must be marked persistent";

  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_NEAR(h->consumer()->data()[0], 5.0, 1e-12);

  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_NEAR(h->consumer()->data()[0], 13.0, 1e-12);

  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_NEAR(h->consumer()->data()[0], 21.0, 1e-12);
}

// ---------------------------------------------------------------------------
// The causality contract (see TickPhase, include/fuse/schedule/expand.hpp).
// ---------------------------------------------------------------------------

/// THE CONTRACT, MEASURED AT RUN TIME RATHER THAN ARGUED FROM THE SCHEDULE.
///
/// No target step may integrate against source state that was not finished by
/// the time that step began. On an Instant channel the delivered number IS the
/// source's state at the end of one source step, so provenance is exact: the
/// ramp makes `data[0] == 2 * step` with `step` 1-based, so a consumed value
/// `v` was produced over `[(v/2 - 1) * src_dt, (v/2) * src_dt)` and the
/// contract is `(v/2) * src_dt <= t_begin`.
///
/// Marking every chain instance `BeforeTicks` would make a delivery jump the
/// tick boundary even when its own source step was still running -- on 1
/// against 1/2 the target's second sub-step would consume source state from
/// `[0, 1)` at `t = 1/2`, its own future, and buy nothing for it, because the
/// slab that follows carries the identical `src_step`. The phase is therefore
/// conditional on the contract.
///
/// SCOPE, stated because the excluded family is a real defect and not a
/// convenience. Withholding a target-limited delivery from the tick at its own
/// stamp buys exactly ONE target step of protection: the delivery is still
/// written into the target's array, and the NEXT target tick, at
/// `(k+2)*dst_dt`, will read it. That is enough iff a source step spans no
/// more than two target steps, `src_dt <= 2*dst_dt`. Beyond that the phase
/// cannot help -- the value would have to not be written at all -- and those
/// schedules are still acausal. See
/// `ASourceStepSpanningMoreThanTwoTargetStepsIsStillDeliveredEarly`.
TEST(Runner, NoTargetStepIntegratesAgainstUnfinishedSourceState) {
  struct Case {
    const char* name;
    Rational    src_dt;
    Rational    dst_dt;
  };
  // Refining, coarsening, non-nesting both ways, and lockstep -- every one
  // with src_dt <= 2*dst_dt. Three of these (1/2 -> 1/3, 1/3 -> 1/2,
  // 1/3 -> 1/4) would be acausal under a blanket `BeforeTicks` marking, so
  // this is not merely a restoration of a simpler behaviour.
  const Case cases[] = {
      {"1 -> 1/2", Rational{1}, Rational{1, 2}},
      {"1/2 -> 1/3", Rational{1, 2}, Rational{1, 3}},
      {"1/3 -> 1/2", Rational{1, 3}, Rational{1, 2}},
      {"1/3 -> 1/4", Rational{1, 3}, Rational{1, 4}},
      {"2/3 -> 1/2", Rational{2, 3}, Rational{1, 2}},
      {"1/2 -> 1", Rational{1, 2}, Rational{1}},
      {"1/4 -> 1", Rational{1, 4}, Rational{1}},
      {"1 -> 1", Rational{1}, Rational{1}},
  };
  for (const Case& c : cases) {
    auto h = build(HarnessOpts{.src_dt = c.src_dt,
                               .dst_dt = c.dst_dt,
                               .producer = ProducerOpts{.ramp = true}});
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    bind_all(*h);
    ASSERT_TRUE(runner.initialize(h->components, h->registry).ok()) << c.name;
    ASSERT_TRUE(runner.run(h->components, 3).ok()) << c.name;

    const Consumer& rx = *h->consumer();
    ASSERT_EQ(rx.seen().size(), rx.seen_at().size());
    bool any_real = false;
    for (std::size_t k = 0; k < rx.seen().size(); ++k) {
      const double v = rx.seen()[k];
      if (v == 0.0) continue;  // nothing delivered yet; no provenance to check
      any_real = true;
      const auto step = static_cast<std::int64_t>(v / 2.0 + 0.5);
      ASSERT_DOUBLE_EQ(v, 2.0 * static_cast<double>(step))
          << c.name << ": consumed a value no single source step produced";
      const Rational finished = Rational{step} * c.src_dt;
      EXPECT_LE(finished, rx.seen_at()[k])
          << c.name << ": the target step beginning at t="
          << rx.seen_at()[k].to_string() << " integrated against source step "
          << step << ", which was still running until t="
          << finished.to_string();
    }
    EXPECT_TRUE(any_real) << c.name << ": the exchange delivered nothing, so "
                                       "the check above proved nothing";
  }
}

/// THE RESIDUAL, PINNED RATHER THAN HIDDEN. The phase rule does not fix this
/// case; it is a known limitation of the schedule.
///
/// A source step that spans more than two target steps still reaches a target
/// step that began before it finished. With `src_dt = 1` and `dst_dt = 1/3`
/// the chain writes at 1/3, 2/3 and 1, all carrying `src_step = 0`, whose
/// state is not complete until t=1. `TickPhase` keeps the write at 1/3 away
/// from the tick at 1/3, but the value sits in the target's array and the tick
/// at 2/3 reads it -- two target steps too early.
///
/// The phase cannot fix this, because the problem is the write, not its order:
/// an Instant chain emits one instance per SLAB, and a slab decomposition
/// gives a long source step several instances that all carry the same,
/// unfinished, state. The fix is to emit an Instant chain once per SOURCE
/// step, at that step's end, which changes instance counts, the buffer plan
/// and every plan dump. That is a redesign of slab-based emission, out of
/// scope here and recorded as R34 in docs/HANDOFF.md.
///
/// This test exists so the residual cannot be mistaken for absent, and so that
/// whoever does R34 has the failing shape already written down. Flip the
/// expectations when it lands.
TEST(Runner, ASourceStepSpanningMoreThanTwoTargetStepsIsStillDeliveredEarly) {
  auto h = build(HarnessOpts{.src_dt = Rational{1},
                             .dst_dt = Rational{1, 3},
                             .producer = ProducerOpts{.ramp = true}});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 2).ok());

  const Consumer& rx = *h->consumer();
  ASSERT_EQ(rx.seen().size(), 6u);
  // t = 0, 1/3 see nothing; t = 2/3 sees source step 1, produced over [0, 1)
  // and therefore NOT finished -- that is the defect. t = 1 and beyond are
  // causal.
  EXPECT_DOUBLE_EQ(rx.seen()[0], 0.0);
  EXPECT_DOUBLE_EQ(rx.seen()[1], 0.0);
  EXPECT_EQ(rx.seen_at()[2], Rational(2, 3));
  EXPECT_DOUBLE_EQ(rx.seen()[2], 2.0)
      << "the early delivery is gone -- if R34 landed, update this test and "
         "add 1 -> 1/3 to NoTargetStepIntegratesAgainstUnfinishedSourceState";
  EXPECT_DOUBLE_EQ(rx.seen()[3], 2.0);  // t = 1, now causal
}

/// The other half of the contract: waiting for causality must not cost a
/// delivery. Every source step the producer completes inside the run has to
/// reach the consumer, or "causal" would be satisfiable by never exchanging.
///
/// 1 against 1/2 is F1's schedule: the consumer takes two steps per period and
/// the producer one, so the two sub-steps SHARE a value -- the chain instance
/// at t=1/2 and the one at t=1 both carry `src_step = 0`. That is why the
/// early delivery bought nothing there, and it is what distinguishes this case
/// from a space-time or accumulating channel, where the per-target-step values
/// genuinely differ.
TEST(Runner, EveryCompletedSourceStepStillReachesTheTarget) {
  auto h = build(HarnessOpts{.src_dt = Rational{1},
                             .dst_dt = Rational{1, 2},
                             .producer = ProducerOpts{.ramp = true}});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 3).ok());

  const Consumer& rx = *h->consumer();
  ASSERT_EQ(rx.seen().size(), 6u);
  // The lagged explicit exchange, restored and now pinned: period p's source
  // value reaches both of the consumer's steps in period p+1 and neither of
  // its steps in period p.
  const std::vector<double> want = {0.0, 0.0, 2.0, 2.0, 4.0, 4.0};
  for (std::size_t k = 0; k < want.size(); ++k) {
    EXPECT_DOUBLE_EQ(rx.seen()[k], want[k])
        << "consumer advance " << k << " at t="
        << rx.seen_at()[k].to_string();
  }
  // Source steps 1 and 2 (of 3) were both delivered; only the last is still
  // in flight when the run ends.
  EXPECT_EQ(h->producer()->steps, 3);
}

/// F2: a `Sample` and the reduction that closes a target step CAN be stamped
/// at the same instant, and the phase is what orders them.
///
/// This needs three clocks. The common period is the lcm over ALL of them, so
/// with only a source and a target `lcm(src_dt, dst_dt)` IS the period and a
/// shared instant could only fall at the period end, where the target has no
/// tick. A bystander on 1/3 makes the period 1 while the exchange still runs
/// on 1/2, so `t = 1/2` carries both `Sample(src_step 1)` and the reduction
/// closing target step 0 -- strictly inside the period, with a target tick on
/// it.
///
/// The reduction is `BeforeTicks` and the `Sample` `AfterTicks`, so the window
/// closes over source step 0 alone, exactly as `Instruction::src_step` says.
/// If the `Sample` ran first -- lower topological rank, and a bare timestamp
/// could not separate them -- it would leak source step 1 into a window that
/// had already ended, delivering 0,0,6,6 where the source produced 2,4,6,8.
TEST(Runner, ASampleSharingAnInstantWithAReductionDoesNotLeakIntoItsWindow) {
  auto h = build(HarnessOpts{.accum = fuse::AccumOp::Mean,
                             .src_dt = Rational{1, 2},
                             .dst_dt = Rational{1, 2},
                             .producer = ProducerOpts{.ramp = true},
                             .bystander_dt = Rational{1, 3}});
  ASSERT_EQ(h->plan.period(), Rational{1})
      << "the bystander did not widen the common period, so this test is not "
         "exercising a shared instant at all";

  // The shared instant, asserted rather than assumed.
  bool sample_at_half = false;
  bool close_at_half = false;
  for (const auto& ins : h->plan.instructions()) {
    if (ins.time != Rational(1, 2)) continue;
    if (ins.kind == fuse::OpKind::Sample) sample_at_half = true;
    if (ins.kind == fuse::OpKind::Accumulate) close_at_half = true;
  }
  ASSERT_TRUE(sample_at_half && close_at_half)
      << "t=1/2 does not carry both a Sample and a reduction";

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 2).ok());

  // Producer ramp over four advances: 2, 4, 6, 8. One source step per target
  // step, so each window is a single sample and the mean is that sample.
  const Consumer& rx = *h->consumer();
  ASSERT_EQ(rx.seen().size(), 4u);
  EXPECT_DOUBLE_EQ(rx.seen()[0], 0.0);
  EXPECT_DOUBLE_EQ(rx.seen()[1], 2.0);
  EXPECT_DOUBLE_EQ(rx.seen()[2], 4.0);
  EXPECT_DOUBLE_EQ(rx.seen()[3], 6.0);
}

/// The phase rule on an ACCUMULATING channel: two target steps per period, and each
/// window's reduction must reach the target step that begins where it closes.
///
/// 1/3 against 1/2 gives a period of 1, three source substeps and two target
/// steps. Target step 0 closes at `t = 1/2` over source steps 0..1 and step 1
/// closes at `t = 1` over step 2 -- an uneven split, because 1/3 does not
/// divide 1/2, and that is the documented window rule ("every source step that
/// BEGINS inside the target step"), not an artefact.
///
/// On THIS schedule the `TickPhase` rule affects only WHO SEES the reduction,
/// not its composition: 1/3 and 1/2 share no instant below their lcm of 1, and
/// 1 is the period here, so no `Sample` is ever stamped where a reduction is.
/// That is a property of this pair of intervals and NOT a general one -- with a
/// third clock the period can be a proper multiple of `lcm(src_dt, dst_dt)` and
/// the two do collide, which is what
/// `ASampleSharingAnInstantWithAReductionDoesNotLeakIntoItsWindow` covers.
/// What is general is the argument in `expand()`: a reduction closing target
/// step k reads source steps j with `j*src_dt` STRICTLY less than
/// `(k+1)*dst_dt`, so it never depends on a `Sample` stamped at that instant,
/// and ordering the `Sample` after it is always safe.
///
/// What changed here is the target's own tick at `t = 1/2`, which used to run
/// first: before the fix the consumer saw 0, 0, 6, 6 instead of 0, 3, 6, 9.
TEST(Runner, EachAccumulationWindowReachesTheTargetStepItClosesInto) {
  // Producer ramp: source substeps 1.. give 2, 4, 6, 8, 10, 12.
  auto h = build(HarnessOpts{.accum = fuse::AccumOp::Mean,
                             .src_dt = Rational{1, 3},
                             .dst_dt = Rational{1, 2},
                             .producer = ProducerOpts{.ramp = true}});
  ASSERT_EQ(h->plan.period(), Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 2).ok());

  const std::vector<double>& seen = h->consumer()->seen();
  ASSERT_EQ(seen.size(), 4u);
  EXPECT_DOUBLE_EQ(seen[0], 0.0);  // t=0: nothing delivered yet
  EXPECT_DOUBLE_EQ(seen[1], 3.0);  // t=1/2: mean of 2 and 4, closed there
  EXPECT_DOUBLE_EQ(seen[2], 6.0);  // t=1:   mean of 6 alone, closed at t=1
  EXPECT_DOUBLE_EQ(seen[3], 9.0);  // t=3/2: mean of 8 and 10
}

TEST(Runner, MaxAccumulationTakesTheLargestSampleOfTheWindow) {
  auto h = make_harness(fuse::AccumOp::Max, Rational{1, 4}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_DOUBLE_EQ(h->consumer()->data()[0], 8.0);
}

TEST(Runner, SumAccumulationAddsEverySampleInTheWindow) {
  auto h = make_harness(fuse::AccumOp::Sum, Rational{1, 4}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_DOUBLE_EQ(h->consumer()->data()[0], 2.0 + 4.0 + 6.0 + 8.0);
}

// ---------------------------------------------------------------------------
// Failure paths.
// ---------------------------------------------------------------------------

TEST(Runner, ReportsANonFiniteValueProducedByAComponent) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());

  h->producer()->poison();
  const auto r = runner.run_period(h->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-NONFINITE");
}

// Iteration regions are implemented (src/runtime/iteration.cpp), not
// blanket-refused. The harness declares its region with no `.until(...)`, so
// what initialize() must refuse is narrow and sharp: an EMPTY
// ConvergenceCriterion is satisfied by definition (true is the identity of
// AND), so the region would stop after one sweep and report that it converged
// -- an explicit scheme wearing an implicit scheme's report. This is the test
// that proves verify_iteration_region() is actually reached from initialize();
// delete the criterion check and it fails.
TEST(Runner, RefusesAnIterationRegionWithNoConvergenceMeasure) {
  auto h = build(HarnessOpts{.iteration_region = true});
  ASSERT_GT(h->plan.region_count(), 0u)
      << "the graph was meant to produce an iteration region";
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok())
      << "running an iteration region's body once and reporting success turns "
         "an implicit scheme into an explicit one with no diagnostic";
  EXPECT_EQ(r.error().code, "E-ITER-NOCONVERGE");
}

// ---------------------------------------------------------------------------
// The iteration region's convergence decision has no
// cross-rank agreement, so a region whose checkpointed components sit on
// different ranks is refused rather than silently risking a hang or a wrong
// answer under a real multi-rank run (docs/dev/invariants.md I4). This test
// is fully verifiable without MPI: `config.base_communicator` is what the
// check reads, and setting it needs no working MPI, only the Fortran handle
// slot RunnerConfig already carries for exactly this purpose.
// The refusal itself (RunnerConfig::base_communicator set, and the region's
// checkpointed components declared on different ranks) is NOT tested here:
// Runner::Impl::attach_comm_graphs() refuses ANY base_communicator in a
// build without FUSE_HAVE_MPI, before resolve_regions() -- and this check --
// is ever reached (E-RT-TRANSPORT-NOMPI). Reaching this refusal needs a
// Tier 1 (FUSE_HAVE_MPI) build; see
// test/parallel/runner_transport_test.cpp's
// RunnerTransport.RefusesACrossRankIterationRegionUnderMpi (needs a host
// run).

// Regression guard. A component's advance() failing
// inside an iteration region's body is carried out correctly by
// the per-sweep check at the top of run_region()'s k-loop
// (`if (step_failure.has_value()) return f;`, right after the body's own
// instructions execute) -- this test pins that behaviour so it cannot
// regress. It does NOT exercise the accelerator-divergence `break` (same
// function), which skips the REST of the current sweep and relies on the
// NEXT k-loop iteration to reach that same check; that iteration does not
// exist when the divergence falls on the last permitted sweep, which the
// post-loop `step_failure` check (and the matching handling in
// step_period(), runner.cpp) covers. Deterministically forcing a built-in
// accelerator to diverge on a specific, low-iteration-count sweep is not
// constructed here; that path is not covered by a test.
TEST(Runner, AnAdvanceFailureInsideAnIterationRegionSweepIsReported) {
  ProducerOpts po;
  po.fail_advance = true;
  auto h = build(HarnessOpts{
      .producer = po,
      .iteration_region = true,
      .region_convergence = one_measure(1e-6, "link")});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());

  const auto r = runner.run_period(h->components);
  ASSERT_FALSE(r.ok())
      << "the producer refuses every advance() after the first; a failure "
         "inside the region's sweep loop must surface as that failure, not "
         "as quiet success or a stale non-convergence report";
  EXPECT_EQ(r.error().code, "E-TEST-ADVANCE-REFUSED") << r.error().message;
}

// The test above pins run_region()'s per-sweep check; it has nothing
// downstream of the region in the same period, so it cannot tell
// step_period() carrying a region failure forward
// (`i = r.exit + 1; continue;`) apart from a plain `return rr;`.
// Reverting exactly that one-line change shows zero failures in the test
// above: with a SECOND, independent channel present, the reverted line
// returns before step_period() ever reaches that channel's own
// instructions, abandoning them -- silent here (Tier 0 has no MPI, so
// nothing hangs), but the abandonment itself is real and is what causes
// the partner-rank hang under MPI (test/parallel/runner_transport_test.cpp,
// RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi, which
// needs a host run).
//
// linkB is declared on the SAME clock as linkA (both dt=1) but is NOT one of
// the region's channels, so its Remap/Diagnose land BEFORE the region in the
// scheduled stream and only its Sink (consumer_b.in) lands after (Kahn's
// algorithm ties broken by ascending NodeId, include/fuse/ir/dag.hpp: the
// region's own Residual is created in a later lowering phase than every
// channel's operator chain, so it always sorts after linkB's Remap/Diagnose
// -- declaring linkB's whole graph before linkA/the region keeps ITS chain
// from interleaving INSIDE the region's own run, which the scheduler refuses
// outright, E-SCHED-REGION, if the declaration order is reversed). Sink is
// non-collective (fuse::is_collective(), op.hpp) and is skipped once
// step_failure is set whether or not that carry-forward is present -- so what
// this test can and does pin is that consumer_b's OWN tick (Consumer::advance(),
// scheduled by absolute time and already satisfied before the region's
// failure exists) is not retroactively undone by a later, unrelated
// region's failure, and that the period still reports that failure.
// The dependence rule, both halves, with the failure FIRST so
// that every other advance in the period is governed by it.
//
// No iteration region here on purpose. In
// ASecondChannelStillAdvancesAfterAnUnrelatedRegionFails the region body
// re-advances its own members, so that fixture cannot show what the rule does
// to a component merely downstream of a failure.
//
// Declaration order puts the A pair first, so the refusing producer's tick
// comes before the other three: consumer_a is downstream of it over linkA and
// must NOT step, while producer_b and consumer_b read nothing it provides and
// must.
TEST(Runner, AFailedStepBlocksOnlyWhatReadsFromIt) {
  CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:8"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:8"});
  const auto m3 = g.add_mesh(fuse::MeshDecl{.name = "m3", .source = "generated:line:8"});
  const auto m4 = g.add_mesh(fuse::MeshDecl{.name = "m4", .source = "generated:line:8"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto r3 = g.add_region(fuse::Region{.name = "r3", .mesh = m3, .dimension = 1});
  const auto r4 = g.add_region(fuse::Region{.name = "r4", .mesh = m4, .dimension = 1});

  const auto pa = g.add_component(fuse::ComponentDecl{
      .name = "producer_a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto ca = g.add_component(fuse::ComponentDecl{
      .name = "consumer_a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto pb = g.add_component(fuse::ComponentDecl{
      .name = "producer_b", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto cb = g.add_component(fuse::ComponentDecl{
      .name = "consumer_b", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});

  const auto sa = g.add_field(pa, fuse::FieldSpec{
      .name = "out", .region = r1, .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto ta = g.add_field(ca, fuse::FieldSpec{
      .name = "in", .region = r2, .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto sb = g.add_field(pb, fuse::FieldSpec{
      .name = "out", .region = r3, .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto tb = g.add_field(cb, fuse::FieldSpec{
      .name = "in", .region = r4, .units = u::kKelvin, .intent = fuse::Intent::Requires});

  g.add_channel("linkA", sa, ta);
  g.add_channel("linkB", sb, tb);

  SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(m1)).ok());
  ASSERT_TRUE(backend.load(g.mesh(m2)).ok());
  ASSERT_TRUE(backend.load(g.mesh(m3)).ok());
  ASSERT_TRUE(backend.load(g.mesh(m4)).ok());

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::OperatorDAG dag = lowered.dag;
  const fuse::ClockSet clocks{g};
  const auto sched = schedule(g, dag, clocks, fuse::make_region_sizes(g, dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();
  fuse::ExecutionPlan plan = sched.plan;

  ProducerOpts po_a;
  po_a.name         = "producer_a";
  po_a.fail_advance = true;
  ProducerOpts po_b;
  po_b.name = "producer_b";

  ComponentSet components;
  components.add(std::make_unique<Producer>(po_a));
  components.add(std::make_unique<Consumer>(8, /*rollback=*/false, "consumer_a"));
  components.add(std::make_unique<Producer>(po_b));
  components.add(std::make_unique<Consumer>(8, /*rollback=*/false, "consumer_b"));

  FieldRegistry registry;
  for (auto& c : components) {
    ASSERT_TRUE(c.register_fields(registry, g).ok());
  }

  Runner runner{g, dag, plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  auto* cons_a = dynamic_cast<Consumer*>(components.find("consumer_a"));
  auto* cons_b = dynamic_cast<Consumer*>(components.find("consumer_b"));
  ASSERT_NE(cons_a, nullptr);
  ASSERT_NE(cons_b, nullptr);

  // The stub refuses from its SECOND advance() on, so the first period is the
  // ordinary one and the second is the one under test.
  const auto first = runner.run_period(components);
  ASSERT_TRUE(first.ok()) << first.error().code << ": " << first.error().message;
  ASSERT_EQ(cons_a->steps, 1);
  ASSERT_EQ(cons_b->steps, 1);

  const auto r = runner.run_period(components);
  ASSERT_FALSE(r.ok()) << "producer_a refuses its second step";
  EXPECT_EQ(r.error().code, "E-TEST-ADVANCE-REFUSED") << r.error().message;

  EXPECT_EQ(cons_a->steps, 1)
      << "consumer_a reads producer_a's output over linkA, so it must NOT "
         "have taken a second step: it would consume a value producer_a "
         "never produced";
  EXPECT_EQ(cons_b->steps, 2)
      << "consumer_b reads nothing producer_a provides, so its step is as "
         "valid as in a period that succeeded; blocking it is what made one "
         "component's refusal stop every other component in the period";
}

TEST(Runner, ASecondChannelStillAdvancesAfterAnUnrelatedRegionFails) {
  CouplingGraph g;
  const auto m3 = g.add_mesh(fuse::MeshDecl{.name = "m3", .source = "generated:line:8"});
  const auto m4 = g.add_mesh(fuse::MeshDecl{.name = "m4", .source = "generated:line:8"});
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:8"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:8"});
  const auto r3 = g.add_region(fuse::Region{.name = "r3", .mesh = m3, .dimension = 1});
  const auto r4 = g.add_region(fuse::Region{.name = "r4", .mesh = m4, .dimension = 1});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});

  const auto pb = g.add_component(fuse::ComponentDecl{
      .name = "producer_b", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto cb = g.add_component(fuse::ComponentDecl{
      .name = "consumer_b", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto pa = g.add_component(fuse::ComponentDecl{
      .name = "producer_a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto ca = g.add_component(fuse::ComponentDecl{
      .name = "consumer_a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});

  const auto sb = g.add_field(pb, fuse::FieldSpec{
      .name = "out", .region = r3, .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto tb = g.add_field(cb, fuse::FieldSpec{
      .name = "in", .region = r4, .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto sa = g.add_field(pa, fuse::FieldSpec{
      .name = "out", .region = r1, .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto ta = g.add_field(ca, fuse::FieldSpec{
      .name = "in", .region = r2, .units = u::kKelvin, .intent = fuse::Intent::Requires});

  g.add_channel("linkB", sb, tb);
  const auto chA = g.add_channel("linkA", sa, ta);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "outer", .channels = {chA}, .max_iterations = 12,
      .convergence = one_measure(1e-6, "linkA")});

  SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(m1)).ok());
  ASSERT_TRUE(backend.load(g.mesh(m2)).ok());
  ASSERT_TRUE(backend.load(g.mesh(m3)).ok());
  ASSERT_TRUE(backend.load(g.mesh(m4)).ok());

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  fuse::OperatorDAG dag = lowered.dag;
  const fuse::ClockSet clocks{g};
  const auto sched = schedule(g, dag, clocks, fuse::make_region_sizes(g, dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();
  fuse::ExecutionPlan plan = sched.plan;

  ProducerOpts po_a;
  po_a.name = "producer_a";
  po_a.fail_advance = true;
  po_a.rollback = true;
  ProducerOpts po_b;
  po_b.name = "producer_b";

  ComponentSet components;
  components.add(std::make_unique<Producer>(po_a));
  components.add(std::make_unique<Consumer>(8, /*rollback=*/true, "consumer_a"));
  components.add(std::make_unique<Producer>(po_b));
  components.add(std::make_unique<Consumer>(8, /*rollback=*/false, "consumer_b"));

  FieldRegistry registry;
  for (auto& c : components) {
    ASSERT_TRUE(c.register_fields(registry, g).ok());
  }

  Runner runner{g, dag, plan, backend};
  const auto init = runner.initialize(components, registry);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  auto* cons_b = dynamic_cast<Consumer*>(components.find("consumer_b"));
  ASSERT_NE(cons_b, nullptr);

  const auto r = runner.run_period(components);
  ASSERT_FALSE(r.ok())
      << "producer_a's region still failed; a second, independent channel "
         "completing must not turn that into a reported success";
  EXPECT_EQ(r.error().code, "E-TEST-ADVANCE-REFUSED") << r.error().message;
  EXPECT_EQ(cons_b->steps, 1)
      << "linkB has nothing to do with region 'outer'; its consumer's own "
         "advance() must not be skipped or rolled back just because an "
         "unrelated region failed later in the same period";

}

// The companion regression: every existing single-process caller -- no
// RunnerConfig::base_communicator set at all -- must keep working exactly as
// before. The guard reads the same declared placement as the test above;
// only the presence of a communicator distinguishes "unverified multi-rank"
// from "there is no second rank to desynchronise from".
TEST(Runner, StillRunsAnIterationRegionWithNoCommunicatorConfigured) {
  auto h = build(HarnessOpts{
      .iteration_region = true,
      .region_convergence = one_measure(1e-6, "link")});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  // initialize() only: whether THIS particular producer/consumer pair's
  // values actually converge under a sweep loop is a different question
  // (Producer::advance() counts every re-advance, so its output keeps
  // moving) from the one this test asks, which is whether resolve_regions()
  // still accepts a declaratively cross-rank region when no communicator is
  // configured -- exactly the guard's own condition.
  const auto init = runner.initialize(h->components, h->registry);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;
}

// The refusal's OTHER precondition. The guard refuses a multi-rank region
// only when a communicator is configured AND the accelerator's coefficient
// would be computed from rank-local history. The test above covers the first
// half (no communicator: accepted). This one covers the second: a region whose
// components are declared on two ranks, driven by Aitken, with no communicator
// configured, must ALSO be accepted -- because with no communicator there is
// no second rank for the coefficients to disagree with. A guard that fired on
// the declared placement alone would refuse every serial run of an accelerated
// multi-rank graph, which is how single-process development of a parallel
// coupling is done. Without this case the mutation that removes the
// communicator test would survive.
TEST(Runner, AcceptsAnAcceleratedMultiRankRegionWhenNoCommunicatorIsConfigured) {
  auto h = build(HarnessOpts{
      .iteration_region = true,
      .region_convergence = one_measure(1e-6, "link"),
      .region_acceleration = fuse::AccelKind::Aitken,
      .dst_rank_count = 2});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto init = runner.initialize(h->components, h->registry);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;
}

// ---------------------------------------------------------------------------
// Spec Section 5.4: which Remaps the runtime will and will not execute.
//
// A `Remap` is now either real weights from `build_weights` or an identity
// that was PROVED correct, and nothing else. The three tests below pin the
// three outcomes on the dependency-free backend, where no supermesh can be
// built at all: refuse, refuse, and take the proved identity.
// ---------------------------------------------------------------------------

TEST(Runner, RemapsBetweenDifferentlySizedRegions) {
  // This test asserted a REFUSAL until the dependency-free supermesh landed:
  // the dependency-free backend could build no weights for 8 cells onto 4, so
  // initialize() failed with the builder's own no-weights code. It now
  // succeeds, because that is the capability that was added -- lowest-order
  // conservative remap on a non-matching pair, with no optional library.
  //
  // The conservation properties of the resulting map are asserted where the
  // weights are inspectable, in test/backend/supermesh_test.cpp; what belongs
  // here is that the runtime accepts the plan and resolves it.
  auto h = build(HarnessOpts{.src_cells = 8, .dst_cells = 4});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

// THE TEST THE OLD SIZE GUARD COULD NOT HAVE PASSED. The box's 24 boundary
// faces and the line's 24 cells agree on a count and on nothing else: they are
// different dimensions, different places, and different measures (total area 6
// against total length 1). Accepting and element-wise copying that pair is
// the whole hazard -- equal counts are not a correspondence. It is refused,
// and the message has to say why, because
// "sizes differ" was never the reason and a user told that would go and resize
// something.
TEST(Runner, RefusesEqualCountRegionsThatAreNotProvablyCoincident) {
  auto h = build(HarnessOpts{.src_cells = 24,
                             .dst_cells = 24,
                             .src_mesh = "generated:box:2x2x2",
                             .dst_mesh = "generated:line:24",
                             .src_dim = 2,
                             .dst_dim = 1,
                             .conservation = fuse::Conservation::Extensive});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok())
      << "24 boundary faces and 24 line cells have the same count and no "
         "correspondence; copying between them is silently wrong";
  // The refusal survived the supermesh builder, but it is now that builder's
  // code rather than the identity-prover's: a conservative remap needs cells
  // of the SAME dimension on both sides, and 2 against 1 is not a hard case
  // to support, it is a pair with no common measure to conserve.
  EXPECT_EQ(r.error().code, "E-REMAP-DIM") << r.error().message;
  EXPECT_NE(r.error().message.find("dimension"), std::string::npos)
      << r.error().message;
}

// A Remap on a component's OWN decomposition is accepted: RemapPlan
// carries the placement and every builder resolves its sides through
// placed_view (rather than each mesh's default view), so the graph
// initialises. That the weights really follow the placed view --
// and not merely that the refusal is gone -- is asserted where the matrix can
// be read: Supermesh.WeightsAreBuiltOnThePlacementsOwnDecomposition, with a
// view whose numbering is reversed.
//
// The backend below forwards everything to the harness's, except that
// `placed_view` answers from a SECOND backend holding the same meshes: a
// valid decomposition, just not the default object.
namespace {

class OtherDecompositionBackend final : public fuse::MeshBackend {
 public:
  OtherDecompositionBackend(const fuse::MeshBackend& inner,
                            const fuse::CouplingGraph& g)
      : inner_(inner) {
    for (std::uint32_t m = 0; m < g.mesh_count(); ++m) {
      EXPECT_TRUE(other_.load(g.mesh(fuse::MeshId{m})).ok());
    }
  }
  fuse::Result<void> load(const fuse::MeshDecl&) override {
    return fuse::Failure{"E-TEST", "not loadable"};
  }
  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(
      std::string_view name) const override {
    return inner_.mesh_id(name);
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(
      fuse::MeshId id) const override {
    return inner_.view(id);
  }
  [[nodiscard]] fuse::Result<std::size_t> region_size(
      const fuse::Region& r) const override {
    return inner_.region_size(r);
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(
      fuse::MeshId id, fuse::ComponentId placement) const override {
    if (!placement.valid()) return inner_.view(id);
    return other_.view(id);
  }
  [[nodiscard]] std::string name() const override { return "other-decomposition"; }

 private:
  const fuse::MeshBackend& inner_;
  SerialMeshBackend        other_;
};

}  // namespace

TEST(Runner, AcceptsARemapOnANonDefaultPlacementDecomposition) {
  auto h = build(HarnessOpts{.src_cells = 8, .dst_cells = 4});
  const OtherDecompositionBackend backend{h->backend, h->graph};
  Runner runner{h->graph, h->dag, h->plan, backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message
                      << " -- weights are built per placement";
  const auto ran = runner.run_period(h->components);
  ASSERT_TRUE(ran.ok()) << ran.error().code << ": " << ran.error().message;
}

// The sharper version of the same hazard, and the one that isolates the
// GEOMETRIC half of the correspondence proof. A 2x2x2 box and an 8x1x1 box
// both tile the unit cube with eight cells; they agree on the count, on the
// topological dimension, on every individual cell measure (0.125), and on the
// total. Nothing short of comparing cell `i` on one side with cell `i` on the
// other can tell them apart -- and an identity copy between them scrambles the
// field. Delete the centroid comparison in regions_coincide() and this is the
// test that fails; the box-versus-line pair above is caught earlier, by the
// dimension check, so it does not cover this.
TEST(Runner, RemapsEqualCountEqualMeasureRegionsInADifferentArrangement) {
  auto h = build(HarnessOpts{.src_cells = 8,
                             .dst_cells = 8,
                             .src_mesh = "generated:box:2x2x2",
                             .dst_mesh = "generated:box:8x1x1",
                             .src_dim = 3,
                             .dst_dim = 3});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  // Also a refusal until the supermesh landed, and for a good reason then: a
  // copy between these two would have been silently wrong, and nothing short
  // of a per-cell comparison could see it. The supermesh does not need to see
  // it -- it computes the actual overlaps, so the different arrangement is
  // handled rather than detected. That the resulting map is NOT the identity
  // is asserted in test/backend/supermesh_test.cpp, where the weights can be
  // read.
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

// The other side of the same coin: two separately loaded, geometrically
// identical meshes still take the cheap path, and -- measured, not asserted by
// inspection -- build no weights at all. A supermesh is the most expensive
// thing this library does; paying for one to rediscover the identity would be
// a real regression, and `remap_weight_builds` is how a test can see it.
TEST(Runner, ProvedCoincidentRegionsTakeTheIdentityAndBuildNoWeights) {
  auto h = build(HarnessOpts{.producer = ProducerOpts{.ramp = true}});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  EXPECT_EQ(runner.stats().remap_weight_builds, 0);
  EXPECT_EQ(runner.stats().remap_weight_nonzeros, 0);
  ASSERT_TRUE(runner.run_period(h->components).ok());
  const std::vector<double>& got = h->consumer()->data();
  ASSERT_EQ(got.size(), 8u);
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_DOUBLE_EQ(got[i], 2.0 + static_cast<double>(i));
  }
}

TEST(Runner, RefusesToRunBeforeInitialize) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = runner.run_period(h->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-INIT");
}

TEST(Runner, RefusesAnUnknownTransportBackend) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  RunnerConfig cfg;
  cfg.transport = "carrier-pigeon";
  const auto r = runner.initialize(h->components, h->registry, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-BACKEND");
}

// ---------------------------------------------------------------------------
// RunnerConfig::transport_setup, the hook that closes the gap
// between the Runner and a transport's one-time, backend-specific setup
// (LowFiveTransport::set_location+build_plan, MpiTransport::set_communicator+
// build_plan) -- neither of which Transport's virtual interface can express.
// ---------------------------------------------------------------------------

TEST(Runner, DefaultRunnerConfigLeavesTransportSetupEmpty) {
  // A guard against a future default-member-initializer mistake: if this ever
  // stopped being empty, every existing CopyTransport caller that never heard
  // of this hook would suddenly run an unexpected callback.
  const RunnerConfig cfg;
  EXPECT_FALSE(static_cast<bool>(cfg.transport_setup));
}

TEST(Runner, TransportSetupRunsExactlyOnceBeforeInstructionsResolveAndDataStillMoves) {
  TransportRegistry::instance().register_backend(
      "probe", [] { return std::make_unique<ProbeTransport>(); });

  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  bind_all(*h);
  Runner runner{h->graph, h->dag, h->plan, h->backend};

  int          calls = 0;
  RunnerConfig cfg;
  cfg.transport = "probe";
  cfg.transport_setup = [&](Transport& t) -> fuse::Result<void> {
    ++calls;
    // The reference is the SAME object the registry just constructed for
    // this Runner, not some other default-constructed instance.
    EXPECT_EQ(t.name(), "probe");
    return fuse::Result<void>::success();
  };

  ASSERT_TRUE(runner.initialize(h->components, h->registry, cfg).ok());
  EXPECT_EQ(calls, 1);

  // The hook ran before resolve_instructions() and did not stand in the way
  // of it: the run still moves real data end to end, exactly as CopyTransport
  // does with no hook at all.
  ASSERT_TRUE(runner.run_period(h->components).ok());
  const Consumer* consumer = h->consumer();
  ASSERT_NE(consumer, nullptr);
  for (const double v : consumer->data()) {
    EXPECT_DOUBLE_EQ(v, 2.0);
  }
}

TEST(Runner, TransportSetupFailureFailsInitializeWithoutResolvingInstructions) {
  TransportRegistry::instance().register_backend(
      "probe", [] { return std::make_unique<ProbeTransport>(); });

  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  bind_all(*h);
  Runner runner{h->graph, h->dag, h->plan, h->backend};

  RunnerConfig cfg;
  cfg.transport = "probe";
  cfg.transport_setup = [](Transport&) -> fuse::Result<void> {
    return fuse::Failure{"E-DEMO-PROBE", "backend-specific setup refused"};
  };

  const auto init = runner.initialize(h->components, h->registry, cfg);
  ASSERT_FALSE(init.ok());
  EXPECT_EQ(init.error().code, "E-DEMO-PROBE");

  // initialize() never reached resolve_instructions()/resolve_regions(), so
  // the Runner is left exactly as uninitialized as if initialize() had never
  // been called.
  const auto run = runner.run_period(h->components);
  ASSERT_FALSE(run.ok());
  EXPECT_EQ(run.error().code, "E-RT-INIT");
}

TEST(Runner, RefusesAComponentTheGraphDoesNotDeclare) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  bind_all(*h);
  h->components.add(std::make_unique<Stowaway>());
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok()) << "a component nothing schedules is never advanced";
  EXPECT_EQ(r.error().code, "E-RT-COMPONENT");
}

TEST(Runner, RefusesAGraphComponentWithNoImplementation) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  bind_all(*h);
  ComponentSet only_one;
  only_one.add(std::make_unique<Producer>(ProducerOpts{}));
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  const auto r = runner.initialize(only_one, h->registry);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-COMPONENT");
}

TEST(Runner, RejectsAnInvalidArcInBufferQuery) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  const auto r = runner.buffer(fuse::ArcId{});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-ARC");
}

// ---------------------------------------------------------------------------
// Conservation. Lowering inserts `Constrain` after a Remap when the field
// declares a Conservation flag (src/ir/rules_derived.cpp).
// ---------------------------------------------------------------------------

TEST(Runner, ConservativeIdentityRemapBetweenCoincidentRegionsReportsNoError) {
  auto h = build(HarnessOpts{.conservation = fuse::Conservation::Extensive});
  ASSERT_TRUE(h->arc_of("link/constrain").valid())
      << "the graph was meant to emit a Constrain operator";
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 2).ok());

  EXPECT_EQ(runner.stats().conservation_checks, 2);
  EXPECT_EQ(runner.stats().conservation_violations, 0);
  EXPECT_LT(runner.stats().worst_conservation_error, 1e-15);
}

// A NOTE ON HOW THE CONSERVATION CHECK IS TESTED.
//
// The tests that show the conservation check has teeth --
// that it integrates against each region's own measures, and that a violation
// can abort the run -- cannot drive it with a box-wall-to-line identity copy
// whose integral jumps by a factor of six. That pair never reaches the
// interpreter at all: it is refused at setup, by
// RefusesEqualCountRegionsThatAreNotProvablyCoincident above, precisely
// because copying across it was wrong. Producing a genuinely non-conservative
// transfer now takes a genuinely non-conservative REMAP -- a truncating map
// over meshes that do not cover the same domain -- and that needs a supermesh.
// Both tests therefore live in test/unit/remap_runtime_test.cpp, on the MOAB
// leg, where the numbers are the real ones. Nothing here is left unpinned; it
// is pinned somewhere a real remap exists.

TEST(Runner, ConservationCheckCanBeTurnedOff) {
  auto h = build(HarnessOpts{.conservation = fuse::Conservation::Extensive});
  ASSERT_TRUE(h->arc_of("link/constrain").valid());
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  RunnerConfig cfg;
  cfg.check_conservation = false;
  ASSERT_TRUE(runner.initialize(h->components, h->registry, cfg).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  // The Constrain node still runs and still copies its input through; what the
  // switch removes is the integral comparison, so the count -- not the error
  // magnitude, which is zero either way -- is the observable.
  EXPECT_EQ(runner.stats().conservation_checks, 0);
  EXPECT_EQ(runner.stats().conservation_violations, 0);
}

// ---------------------------------------------------------------------------
// resolve_constrain() accepts a conservation check whose Remap target
// spans more than one rank: the integral it builds is this rank's owned
// share, Constrain is collective (fuse::is_collective), and the shares
// are summed over the base communicator, so the multi-rank target is
// ACCEPTED at setup (reducing inside a non-collective operator would hang
// a rank that had already failed).
//
// What Tier 0 can pin: that setup with a 2-rank target does not fail on
// E-RT-CONSERVATION-MPI. It cannot run the reduction: with
// no MPI in this build, ANY base_communicator is refused one guard later
// (attach_comm_graphs' "#if !FUSE_HAVE_MPI"), so this test configures none
// and checks that the graph initialises and its Constrain runs. The reduction
// itself is exercised by parallel.nodal_remap_runner at 2-5 ranks, whose
// wall_force channel is conservative onto a multi-rank target and whose OWN
// MPI_Allreduce of the same integrals is what the Runner's verdict is checked
// against.
TEST(Runner, AcceptsAConservationCheckOnAMultiRankTarget) {
  auto h = build(HarnessOpts{.conservation = fuse::Conservation::Extensive,
                             .dst_rank_count = 2});
  ASSERT_TRUE(h->arc_of("link/constrain").valid());
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message
                      << " -- a multi-rank conservation target is accepted; "
                         "its integrals are reduced";
  const auto ran = runner.run_period(h->components);
  ASSERT_TRUE(ran.ok()) << ran.error().code << ": " << ran.error().message;
  EXPECT_GE(runner.stats().conservation_checks, 1u)
      << "the Constrain ran rather than being skipped";
}

// The companion regression, mirroring
// StillRunsAnIterationRegionWithNoCommunicatorConfigured: the identical
// nominal placement (consumer on 2 declared ranks) must still be accepted
// when no communicator is configured -- there is no second rank to
// desynchronise from, and every single-process caller that happens to
// declare a multi-rank placement (the common case in this file's OTHER
// tests, which never sets base_communicator at all) must keep working.
TEST(Runner, StillRunsAConservationCheckWithNoCommunicatorConfigured) {
  auto h = build(HarnessOpts{.conservation = fuse::Conservation::Extensive,
                             .dst_rank_count = 2});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

TEST(Runner, RejectsANegativePeriodCount) {
  auto h = make_harness(fuse::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  const auto r = runner.run(h->components, -3);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-PERIODS");
  EXPECT_EQ(runner.stats().periods, 0);
}
