// KNOWN_ISSUES.md A5: `Runner::Impl::execute()` dispatching its elementwise
// `OpKind`s on `RunnerConfig::RunnerExecutionSpace::Device`, run through a
// real CouplingGraph -> lower -> schedule -> Runner pipeline (never by
// calling a kernel directly -- test/num/space_kernels_test.cpp already does
// that for every enabled space). docs/dev/runner-device-dispatch.md has the
// scope: only the Space-generic cases (Sample/Accumulate/Scale/Remap/Relax/
// Residual/Merge/Extrapolate/Interpolate-identity) move; Transport/Epoch/
// Sink/Constrain/Diagnose and Remap's distributed-nodal branch
// (apply_nodal_dist_remap(), an MPI exchange) stay host-only
// unconditionally. Neither scenario here builds a distributed-nodal Remap
// (it needs a multi-rank placement), so that fallback is documented, not
// exercised.
//
// WHAT IS ASSERTED, AND HOW STRONGLY. `RunStats::device_dispatches` is the
// evidence the device path ran at all -- the delivered values are meant to
// agree with the host's, so they cannot be. Agreement is BITWISE for the
// copy/accumulate/scale-shaped arcs (each is one rounding per element, in
// the same order, on both spaces) and to a few ulps for the relax and
// weighted-remap arcs: nvcc contracts `a*b + c` into a single FMA inside
// `RelaxFunctor`/`RemapMultiFunctor` (space_kernels.hpp), one rounding where
// the host does two -- the same gap D38 closed for the deterministic dot
// products with `unfused_multiply_add`, deliberately NOT applied to those two
// functors here because build-evidence/mutations.toml anchors their exact
// source lines ("relax-swaps-its-weights", "remap-apply-drops-each-row-first-
// entry").
//
// On a build with no device backend at all the Device cases assert the
// `E-RT-DEVICE` refusal instead of skipping, so the test never reads as a
// pass for a configuration it did not exercise.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::OpKind;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::SerialMeshBackend;
using fuse::StepContext;
using Space = RunnerConfig::RunnerExecutionSpace;
namespace u = fuse::units;

namespace {

constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
    true;
#else
    false;
#endif

/// Writes `value * step + 0.125 * i` on every advance: varies per cell AND per
/// step, so a weighted remap moves something non-constant and a wrong
/// accumulation count or a stale previous iterate shows up in the numbers.
class Producer final : public fuse::Component {
 public:
  Producer(std::size_t cells, double value) : data_(cells, 0.0), value_(value) {}
  [[nodiscard]] std::string name() const override { return "producer"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
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
    ++steps_;
    for (std::size_t i = 0; i < data_.size(); ++i) {
      data_[i] = value_ * static_cast<double>(steps_) + 0.125 * static_cast<double>(i);
    }
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> data_;
  double              value_;
  int                 steps_ = 0;
};

class Consumer final : public fuse::Component {
 public:
  explicit Consumer(std::size_t cells) : data_(cells, 0.0) {}
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

  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::vector<double> data_;
};

// The same two components with their fields in the default execution
// space's memory (CudaSpace on a CUDA build), registered as
// fuse::MemorySpace::Device: docs/dev/device-resident-arcs.md step 2. The
// producer computes on the host and copies in once per advance(), which is
// what a host-side physics code handing device memory to the coupler does;
// the consumer's values() copies out once, for the assertions only.
using DeviceMem = Kokkos::DefaultExecutionSpace::memory_space;
using DeviceVec = Kokkos::View<double*, DeviceMem>;

class DeviceProducer final : public fuse::Component {
 public:
  DeviceProducer(std::size_t cells, double value)
      : data_("producer.out", cells), host_(cells, 0.0), value_(value) {}
  [[nodiscard]] std::string name() const override { return "producer"; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = data_.extent(0)},
                                          fuse::MemorySpace::Device});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    ++steps_;
    for (std::size_t i = 0; i < host_.size(); ++i) {
      host_[i] = value_ * static_cast<double>(steps_) + 0.125 * static_cast<double>(i);
    }
    Kokkos::deep_copy(data_, Kokkos::View<const double*, Kokkos::HostSpace,
                                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                                 host_.data(), host_.size()));
    return fuse::Result<void>::success();
  }
 private:
  DeviceVec           data_;
  std::vector<double> host_;
  double              value_;
  int                 steps_ = 0;
};

class DeviceConsumer final : public fuse::Component {
 public:
  explicit DeviceConsumer(std::size_t cells) : data_("consumer.in", cells) {}
  [[nodiscard]] std::string name() const override { return "consumer"; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = data_.extent(0)},
                                          fuse::MemorySpace::Device});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::vector<double> values() const {
    std::vector<double> out(data_.extent(0), 0.0);
    Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(out.data(), out.size()),
                      data_);
    return out;
  }
 private:
  DeviceVec data_;
};

struct Scenario {
  bool          device_fields = false;  // register the fields in device memory
  fuse::AccumOp accum = fuse::AccumOp::Instant;
  Rational      src_dt{1};
  Rational      dst_dt{1};
  std::size_t   src_cells = 8;
  std::size_t   dst_cells = 8;
  /// 0 leaves the channel without a Relax node.
  double        relax_omega = 0.0;
};

/// Sample(Mean) four times per period, Accumulate (copy + 1/4 scale) once,
/// identity Remap between two 8-cell lines: every kernel on the path is one
/// rounding per element.
Scenario mean_over_four_substeps() {
  return Scenario{.accum = fuse::AccumOp::Mean, .src_dt = Rational{1, 4},
                  .dst_dt = Rational{1}};
}

/// A conservative weighted Remap (8 cells -> 6 cells, so the regions cannot
/// coincide) followed by Relax(0.5): the two FMA-shaped kernels.
Scenario weighted_remap_then_relax() {
  return Scenario{.src_cells = 8, .dst_cells = 6, .relax_omega = 0.5};
}

struct Harness {
  CouplingGraph       graph;
  SerialMeshBackend   backend;
  fuse::OperatorDAG   dag;
  fuse::ExecutionPlan plan;
  ComponentSet        components;
  FieldRegistry       registry;

  [[nodiscard]] std::vector<double> consumer_values() const {
    const fuse::Component* c = components.find("consumer");
    if (const auto* h = dynamic_cast<const Consumer*>(c)) return h->data();
    if (const auto* d = dynamic_cast<const DeviceConsumer*>(c)) return d->values();
    return {};
  }
  [[nodiscard]] int count(OpKind kind) const {
    int n = 0;
    for (const fuse::Instruction& ins : plan.instructions()) n += ins.kind == kind;
    return n;
  }
  [[nodiscard]] bool has(OpKind kind) const { return count(kind) > 0; }
};

std::unique_ptr<Harness> build(const Scenario& s) {
  auto h = std::make_unique<Harness>();
  auto& g = h->graph;
  const auto m1 = g.add_mesh(fuse::MeshDecl{
      .name = "m1", .source = "generated:line:" + std::to_string(s.src_cells)});
  const auto m2 = g.add_mesh(fuse::MeshDecl{
      .name = "m2", .source = "generated:line:" + std::to_string(s.dst_cells)});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{
      .name = "producer", .ranks = fuse::RankSet{0, 1}, .interval = s.src_dt});
  const auto c = g.add_component(fuse::ComponentDecl{
      .name = "consumer", .ranks = fuse::RankSet{1, 1}, .interval = s.dst_dt});
  const auto src = g.add_field(
      p, fuse::FieldSpec{.name = "out", .region = r1, .units = u::kKelvin,
                         .intent = fuse::Intent::Provides});
  const auto tgt = g.add_field(
      c, fuse::FieldSpec{.name = "in", .region = r2, .units = u::kKelvin,
                         .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", src, tgt);
  if (s.accum != fuse::AccumOp::Instant) g.channel(ch).accumulate(s.accum);
  if (s.relax_omega != 0.0) g.channel(ch).relax(s.relax_omega);

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, h->dag, clocks, fuse::make_region_sizes(g, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  if (s.device_fields) {
    h->components.add(std::make_unique<DeviceProducer>(s.src_cells, 2.0));
    h->components.add(std::make_unique<DeviceConsumer>(s.dst_cells));
  } else {
    h->components.add(std::make_unique<Producer>(s.src_cells, 2.0));
    h->components.add(std::make_unique<Consumer>(s.dst_cells));
  }
  return h;
}

struct Outcome {
  fuse::Result<void>  init = fuse::Result<void>::success();
  std::vector<double> consumer;
  fuse::RunStats      stats;
  bool                has_relax = false;
  bool                has_sample = false;
  bool                has_accumulate = false;
  bool                has_remap = false;
  int                 transports = 0;  // Transport instructions per period
};

Outcome run(const Scenario& s, Space space, int periods) {
  auto h = build(s);
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& comp : h->components) {
    EXPECT_TRUE(comp.register_fields(h->registry, h->graph).ok());
  }
  RunnerConfig cfg;
  cfg.execution_space = space;
  Outcome o;
  o.has_relax = h->has(OpKind::Relax);
  o.has_sample = h->has(OpKind::Sample);
  o.has_accumulate = h->has(OpKind::Accumulate);
  o.has_remap = h->has(OpKind::Remap);
  o.transports = h->count(OpKind::Transport);
  o.init = runner.initialize(h->components, h->registry, cfg);
  if (!o.init.ok()) return o;
  for (int k = 0; k < periods; ++k) {
    const auto r = runner.run_period(h->components);
    EXPECT_TRUE(r.ok()) << r.error().message;
  }
  o.consumer = h->consumer_values();
  o.stats = runner.stats();
  return o;
}

}  // namespace

TEST(RunnerDevice, HostIsTheDefaultAndDispatchesNothingToTheDevice) {
  EXPECT_EQ(RunnerConfig{}.execution_space, Space::Host);
  const Outcome host = run(mean_over_four_substeps(), Space::Host, 2);
  ASSERT_TRUE(host.init.ok()) << host.init.error().message;
  EXPECT_EQ(host.stats.device_dispatches, 0);
  EXPECT_EQ(host.stats.periods, 2);
}

TEST(RunnerDevice, DeviceNamesTheRealDeviceSpaceOfThisBuild) {
  const std::string_view name = Runner::device_execution_space_name();
#if defined(KOKKOS_ENABLE_CUDA)
  EXPECT_EQ(name, "Cuda");
#elif defined(KOKKOS_ENABLE_HIP)
  EXPECT_EQ(name, "HIP");
#elif defined(KOKKOS_ENABLE_SYCL)
  EXPECT_EQ(name, "SYCL");
#else
  EXPECT_EQ(name, std::string_view{Kokkos::DefaultHostExecutionSpace::name()})
      << "with no device backend compiled in, Device must resolve to the host "
         "space -- and initialize() must refuse it, see the next case";
#endif
}

TEST(RunnerDevice, DeviceIsRefusedByNameWhenNoDeviceBackendIsCompiledIn) {
  if constexpr (kHasDeviceBackend) {
    GTEST_SKIP() << "this build has a device backend ("
                 << Runner::device_execution_space_name()
                 << "); the refusal is unreachable here by construction";
  }
  const Outcome dev = run(mean_over_four_substeps(), Space::Device, 1);
  ASSERT_FALSE(dev.init.ok());
  EXPECT_EQ(dev.init.error().code, "E-RT-DEVICE");
}

TEST(RunnerDevice, DeviceModeRunsTheElementwiseKernelsOnTheDevice) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";
  const Outcome dev = run(mean_over_four_substeps(), Space::Device, 2);
  ASSERT_TRUE(dev.init.ok()) << dev.init.error().message;
  // The scenario must actually contain the kernels this file claims to
  // exercise; a lowering change that dropped one would otherwise turn this
  // into a test of nothing.
  EXPECT_TRUE(dev.has_sample);
  EXPECT_TRUE(dev.has_accumulate);
  EXPECT_TRUE(dev.has_remap);
  EXPECT_GT(dev.stats.device_dispatches, 0)
      << "execution_space = Device was configured, but no instruction's "
         "kernel ran on " << Runner::device_execution_space_name();
  // docs/dev/device-resident-arcs.md step 1: the intermediates between the
  // Space-generic instructions live in device memory; only the component-
  // owned endpoints and the arcs a host-only case touches stay on the host.
  // The RED for this assertion is the staging-only tree, where every arc is
  // host memory and the counter stays 0.
  EXPECT_GT(dev.stats.device_resident_arcs, 0)
      << "Device mode placed no arc in device memory";
  EXPECT_EQ(dev.stats.periods, 2);
}

TEST(RunnerDevice, DeviceAndHostAgreeBitwiseOnAccumulateScaleAndCopy) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";
  const Outcome host = run(mean_over_four_substeps(), Space::Host, 3);
  const Outcome dev = run(mean_over_four_substeps(), Space::Device, 3);
  ASSERT_TRUE(host.init.ok()) << host.init.error().message;
  ASSERT_TRUE(dev.init.ok()) << dev.init.error().message;
  ASSERT_EQ(host.consumer.size(), dev.consumer.size());
  ASSERT_FALSE(host.consumer.empty());
  for (std::size_t i = 0; i < host.consumer.size(); ++i) {
    EXPECT_EQ(host.consumer[i], dev.consumer[i])
        << "cell " << i << ": copy/accumulate/scale are one rounding per "
                           "element in the same order on both spaces, so "
                           "assert exactly (no tolerance)";
  }
  // Not the initial zeros: the producer wrote a ramp, the mean is non-trivial.
  EXPECT_NE(host.consumer.front(), 0.0);
  EXPECT_EQ(host.stats.device_dispatches, 0);
  EXPECT_EQ(host.stats.device_resident_arcs, 0);
  EXPECT_GT(dev.stats.device_dispatches, 0);
  EXPECT_GT(dev.stats.device_resident_arcs, 0);
}

TEST(RunnerDevice, DeviceAndHostAgreeToRoundingOnRelaxAndWeightedRemap) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";
  const Outcome host = run(weighted_remap_then_relax(), Space::Host, 3);
  const Outcome dev = run(weighted_remap_then_relax(), Space::Device, 3);
  ASSERT_TRUE(host.init.ok()) << host.init.error().message;
  ASSERT_TRUE(dev.init.ok()) << dev.init.error().message;
  EXPECT_TRUE(dev.has_relax);
  EXPECT_TRUE(dev.has_remap);
  // A REAL matrix, not the proved-coincident identity: 8 cells onto 6.
  EXPECT_GT(host.stats.remap_weight_builds, 0);
  EXPECT_GT(dev.stats.remap_weight_builds, 0);
  EXPECT_GT(dev.stats.device_dispatches, 0);
  ASSERT_EQ(host.consumer.size(), dev.consumer.size());
  ASSERT_FALSE(host.consumer.empty());
  for (std::size_t i = 0; i < host.consumer.size(); ++i) {
    const double a = host.consumer[i];
    const double b = dev.consumer[i];
    // A handful of ulps: one FMA contraction per relax element and per
    // remap row (see the file comment), nothing larger.
    EXPECT_NEAR(a, b, 1e-14 * std::max(1.0, std::abs(a)))
        << "cell " << i;
  }
  EXPECT_NE(host.consumer.front(), 0.0);
}

// docs/dev/device-resident-arcs.md step 2: fields registered in device
// memory are aliased by the Runner directly, so a chain whose endpoints are
// both device fields is mirrored nowhere and its Sink copies nothing across
// the boundary. The one host stop left is the channel's Transport (a local
// move on one rank, still host code until the GPU-aware transport step):
// the Epoch before it copies onto the host and the Epoch after it copies
// back, two crossings per Transport per period and no other.
TEST(RunnerDevice, DeviceRegisteredFieldsCrossOnlyAtTheTransportAndAgreeWithHost) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";
  Scenario s = mean_over_four_substeps();
  const Outcome host = run(s, Space::Host, 3);
  s.device_fields = true;
  const Outcome dev = run(s, Space::Device, 3);
  ASSERT_TRUE(host.init.ok()) << host.init.error().message;
  ASSERT_TRUE(dev.init.ok()) << dev.init.error().message;
  ASSERT_EQ(host.consumer.size(), dev.consumer.size());
  ASSERT_FALSE(host.consumer.empty());
  for (std::size_t i = 0; i < host.consumer.size(); ++i) {
    EXPECT_EQ(host.consumer[i], dev.consumer[i]) << "cell " << i;
  }
  EXPECT_GT(dev.stats.device_dispatches, 0);
  EXPECT_GT(dev.stats.device_resident_arcs, 0);
  // RED on the step-1 tree: Sink copied device arc -> device field through
  // host memcpy (a crash or garbage), and device fields were refused.
  ASSERT_GT(dev.transports, 0);
  EXPECT_EQ(dev.stats.device_boundary_copies, 2 * dev.transports * 3)
      << "a device-registered chain crossed the boundary somewhere other "
         "than around its Transport";
}

// The host-registered components in Device mode: the producer's field is
// mirrored once per producer step (the Sample after each tick refreshes
// it; Impl::component_epoch) and the Sink crosses back once per consumer
// step, so the boundary copies are counted per component step, not per
// instruction.
TEST(RunnerDevice, HostRegisteredFieldsCrossTheBoundaryOncePerStepEachWay) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";
  const Outcome dev = run(mean_over_four_substeps(), Space::Device, 2);
  ASSERT_TRUE(dev.init.ok()) << dev.init.error().message;
  // Two periods: the 1/4-step producer advances 8 times (one mirror refresh
  // each), the consumer twice (one Sink copy each), plus the two crossings
  // around each period's Transport (see the device-registered test above);
  // a per-instruction staging design would cross at least once per kernel
  // launch -- 8 Samples, 2 Accumulates and 2 Epochs more.
  const std::int64_t producer_steps = 2 * 4, consumer_steps = 2;
  ASSERT_GT(dev.transports, 0);
  EXPECT_EQ(dev.stats.device_boundary_copies,
            producer_steps + consumer_steps + 2 * dev.transports * 2);
}

// A device-registered field under the Host default is a configuration
// error, not a silent host dereference of device memory.
TEST(RunnerDevice, DeviceRegisteredFieldIsRefusedUnderHostExecution) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";
  Scenario s = mean_over_four_substeps();
  s.device_fields = true;
  const Outcome host = run(s, Space::Host, 1);
  ASSERT_FALSE(host.init.ok());
  EXPECT_EQ(host.init.error().code, "E-RT-SPACE");
}
