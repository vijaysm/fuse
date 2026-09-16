#include <gtest/gtest.h>
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

  [[nodiscard]] std::string name() const override { return "producer"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
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

  int steps = 0;

 private:
  ProducerOpts        opts_;
  std::vector<double> data_;
};

class Consumer final : public fuse::Component {
 public:
  explicit Consumer(std::size_t n) : data_(n, 0.0) {
    seen_.reserve(64);
    seen_at_.reserve(64);
  }

  [[nodiscard]] std::string name() const override { return "consumer"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
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
  int steps = 0;

 private:
  std::vector<double>   data_;
  std::vector<double>   seen_;
  std::vector<Rational> seen_at_;
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

/// What `build_weights` says when this file's mesh backend cannot give it a
/// supermesh. Every Harness here uses `SerialMeshBackend`, which has no
/// connectivity or coordinates to intersect, so the two answers are: with MOAB
/// compiled in, the builder gets as far as looking at the backend and rejects
/// it; without MOAB it never gets that far. Both are the builder's own code,
/// deliberately not folded into one runtime code -- they tell the user two
/// different things to do.
constexpr const char* kNoWeightsHere =
#if FUSE_HAVE_MOAB
    "E-REMAP-BACKEND";
#else
    "E-REMAP-NOMOAB";
#endif

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

/// Task 8.5's RunnerConfig::transport_setup hook, exercised without dragging
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
  /// Non-zero adds a `Bystander` on this interval, which only widens the
  /// common period. See the class comment for why a test would want that.
  Rational      bystander_dt{0};
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
      .name = "producer", .ranks = {.first = 0, .count = 1},
      .interval = o.src_dt});
  const auto c = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = {.first = 1, .count = 1},
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
  if (o.accum != fuse::AccumOp::Instant) g.channel(ch).accumulate(o.accum);
  const bool with_bystander = o.bystander_dt > Rational{0};
  if (with_bystander) {
    const auto m3 =
        g.add_mesh(fuse::MeshDecl{.name = "m3", .source = dst_source});
    const auto r3 = g.add_region(
        fuse::Region{.name = "r3", .mesh = m3, .dimension = o.dst_dim});
    const auto bc = g.add_component(fuse::ComponentDecl{
        .name = "bystander", .ranks = {.first = 2, .count = 1},
        .interval = o.bystander_dt});
    g.add_field(bc, fuse::FieldSpec{.name = "idle", .region = r3,
                                    .units = u::kKelvin,
                                    .intent = fuse::Intent::Provides});
    EXPECT_TRUE(h->backend.load(g.mesh(m3)).ok());
  }
  if (o.iteration_region) {
    g.add_iteration_region(fuse::IterationRegionDecl{
        .name = "outer", .channels = {ch}, .max_iterations = 12});
  }

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const fuse::ClockSet clocks{g};
  auto sched =
      schedule(g, h->dag, clocks, fuse::make_region_sizes(g, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  ProducerOpts po = o.producer;
  po.cells = o.src_cells;
  h->components.add(std::make_unique<Producer>(po));
  h->components.add(std::make_unique<Consumer>(o.dst_cells));
  if (with_bystander) h->components.add(std::make_unique<Bystander>());
  return h;
}

/// Matching 8-cell meshes so Phase 4's identity Remap is meaningful.
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
// The brief's tests.
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
// Spec §4.2: who owns which buffer.
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
// Spec §2 layer 3: no allocation in the hot loop.
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
      << " times; spec §2 layer 3 requires the interpreted loop to allocate "
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
/// This is the test R2's first fix needed and did not have. Marking every
/// chain instance `BeforeTicks` made a delivery jump the tick boundary even
/// when its own source step was still running -- on 1 against 1/2 the target's
/// second sub-step consumed source state from `[0, 1)` at `t = 1/2`, its own
/// future, and bought nothing for it, because the slab that follows carries
/// the identical `src_step`. The phase is now conditional on the contract.
///
/// SCOPE, stated because the excluded family is a real defect and not a
/// convenience. Withholding a target-limited delivery from the tick at its own
/// stamp buys exactly ONE target step of protection: the delivery is still
/// written into the target's array, and the NEXT target tick, at
/// `(k+2)*dst_dt`, will read it. That is enough iff a source step spans no
/// more than two target steps, `src_dt <= 2*dst_dt`. Beyond that the phase
/// cannot help -- the value would have to not be written at all -- and those
/// schedules are still acausal, exactly as they were before R2. See
/// `ASourceStepSpanningMoreThanTwoTargetStepsIsStillDeliveredEarly`.
TEST(Runner, NoTargetStepIntegratesAgainstUnfinishedSourceState) {
  struct Case {
    const char* name;
    Rational    src_dt;
    Rational    dst_dt;
  };
  // Refining, coarsening, non-nesting both ways, and lockstep -- every one
  // with src_dt <= 2*dst_dt. Three of these (1/2 -> 1/3, 1/3 -> 1/2,
  // 1/3 -> 1/4) were acausal before R2 as well, so this is not merely a
  // restoration of the old behaviour.
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

/// THE RESIDUAL, PINNED RATHER THAN HIDDEN. Not introduced by R2 and not
/// fixed by it: measured identical on 9902f82, the commit before any of this
/// work.
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
/// Before R2's fix the `Sample` ran first -- lower topological rank, and a
/// bare timestamp could not separate them -- and leaked source step 1 into a
/// window that had already ended, delivering 0,0,6,6 where the source produced
/// 2,4,6,8.
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

/// R2 on an ACCUMULATING channel: two target steps per period, and each
/// window's reduction must reach the target step that begins where it closes.
///
/// 1/3 against 1/2 gives a period of 1, three source substeps and two target
/// steps. Target step 0 closes at `t = 1/2` over source steps 0..1 and step 1
/// closes at `t = 1` over step 2 -- an uneven split, because 1/3 does not
/// divide 1/2, and that is the documented window rule ("every source step that
/// BEGINS inside the target step"), not an artefact.
///
/// On THIS schedule R2's `TickPhase` fix changed only WHO SEES the reduction,
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

// Phase 7 Task 7.5 replaced the blanket E-RT-REGION refusal of any plan with
// an iteration region by a real implementation (src/runtime/iteration.cpp).
// The harness declares its region with no `.until(...)`, so what initialize()
// must now refuse is narrower and sharper: an EMPTY ConvergenceCriterion is
// satisfied by definition (true is the identity of AND), so the region would
// stop after one sweep and report that it converged -- an explicit scheme
// wearing an implicit scheme's report. This is the test that proves
// verify_iteration_region() is actually reached from initialize(); delete the
// criterion check and it fails.
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
// Spec §5.4, Task 5.5: which Remaps the runtime will and will not execute.
//
// A `Remap` is now either real weights from `build_weights` or an identity
// that was PROVED correct, and nothing else. The three tests below pin the
// three outcomes on the dependency-free backend, where no supermesh can be
// built at all: refuse, refuse, and take the proved identity.
// ---------------------------------------------------------------------------

TEST(Runner, RefusesARemapBetweenDifferentlySizedRegionsItCannotBuildWeightsFor) {
  auto h = build(HarnessOpts{.src_cells = 8, .dst_cells = 4});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok());
  // The builder's own code, not a runtime code that hides it: what the user
  // has to fix is that no supermesh can be built here -- because this build
  // has no MOAB, or because this is the dependency-free backend -- and NOT the
  // size difference, which is merely why an identity was not an option.
  EXPECT_EQ(r.error().code, kNoWeightsHere);
  EXPECT_NE(r.error().message.find("not provably the same cells"),
            std::string::npos)
      << r.error().message;
}

// THE TEST THE OLD SIZE GUARD COULD NOT HAVE PASSED. The box's 24 boundary
// faces and the line's 24 cells agree on a count and on nothing else: they are
// different dimensions, different places, and different measures (total area 6
// against total length 1). Until Task 5.5 that pair was accepted and
// element-wise copied, which is the whole hazard -- equal counts are not a
// correspondence. Now it is refused, and the message has to say why, because
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
  EXPECT_EQ(r.error().code, kNoWeightsHere);
  EXPECT_NE(r.error().message.find("different dimension"), std::string::npos)
      << r.error().message;
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
TEST(Runner, RefusesEqualCountEqualMeasureRegionsInADifferentArrangement) {
  auto h = build(HarnessOpts{.src_cells = 8,
                             .dst_cells = 8,
                             .src_mesh = "generated:box:2x2x2",
                             .dst_mesh = "generated:box:8x1x1",
                             .src_dim = 3,
                             .dst_dim = 3});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  bind_all(*h);
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok())
      << "eight cells of 0.125 each on both sides, arranged differently: only "
         "a per-cell comparison can see that a copy would be wrong";
  EXPECT_EQ(r.error().code, kNoWeightsHere);
  EXPECT_NE(r.error().message.find("does not name the same cell"),
            std::string::npos)
      << r.error().message;
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
// Task 8.5: RunnerConfig::transport_setup, the hook that closes the gap
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

// A NOTE ON WHAT MOVED, AND WHY.
//
// Until Task 5.5 the two tests that showed the conservation check has teeth --
// that it integrates against each region's own measures, and that a violation
// can abort the run -- drove it with a box-wall-to-line identity copy whose
// integral jumped by a factor of six. That pair no longer reaches the
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
