// KNOWN_ISSUES.md A5: `Runner::Impl::execute()` dispatching its elementwise
// `OpKind`s on `RunnerConfig::RunnerExecutionSpace::Device`, run through a
// real CouplingGraph -> lower -> schedule -> Runner pipeline (never by
// calling a kernel directly -- test/num/space_kernels_test.cpp already does
// that for every enabled space). docs/dev/runner-device-dispatch.md has the
// scope: only the Space-generic cases (Sample/Accumulate/Scale/Remap/Relax/
// Residual/Merge/Extrapolate/Interpolate-identity) move; Epoch/Sink/
// Constrain/Diagnose and Remap's distributed-nodal branch
// (apply_nodal_dist_remap(), an MPI exchange) stay host-only
// unconditionally. `Transport` moved out of that group in the transport
// track (docs/dev/device-resident-arcs.md Phase 3): its own case packs and
// unpacks on the device when both its arcs are device-resident, handing
// MPI a device pointer when `Transport::is_gpu_aware()` and staging
// through host memory when not -- see
// RunnerDevice.DeviceRegisteredFieldsCrossOnlyAtHostOnlyInstructionsAndAgreeWithHost
// below, whose own comment has the updated crossing count. The scenarios
// here use the no-graph (single-process, no CommGraph) Transport path,
// never the MPI one; `test/parallel/runner_transport_test.cpp`'s `.device`
// twins (`fuse_add_mpi_test(... DEVICE_TWIN)`) exercise the MPI path's
// device/host and GPU-aware/not branches instead. Neither scenario here
// builds a distributed-nodal Remap
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
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/iteration.hpp>
#include <fuse/runtime/runner.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
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
  int                 transports = 0;   // Transport instructions per period
  int                 diagnoses = 0;    // Diagnose instructions per period
  int                 accumulates = 0;  // Accumulate instructions per period
  int                 remaps = 0;       // Remap instructions per period
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
  o.diagnoses = h->count(OpKind::Diagnose);
  o.accumulates = h->count(OpKind::Accumulate);
  o.remaps = h->count(OpKind::Remap);
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
  EXPECT_EQ(RunnerConfig{}.execution_space, Space::Default);
  ::unsetenv("FUSE_EXECUTION_SPACE");
  const auto resolved = RunnerConfig{}.resolved_execution_space();
  ASSERT_TRUE(resolved.ok());
  EXPECT_EQ(resolved.value(), Space::Host);
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

// docs/dev/device-resident-arcs.md step 2 and, as of the transport track,
// Phase 3: fields registered in device memory are aliased by the Runner
// directly, so a chain whose endpoints are both device fields is mirrored
// nowhere. The one host stop left in this plan is its single remaining
// unconditionally host-only instruction -- the Diagnose after the Remap (a
// host reduction) -- which pins the arc it reads and the arc it writes, so
// a space-aware copy next to it (the Sink) crosses once. The channel's
// Transport is no longer a host stop: it is a same-process, no-graph move
// (this scenario builds no CommGraph), both its arcs are coupler-owned and
// therefore device-resident like everything around them, so its case takes
// the `Kokkos::deep_copy` branch and counts in `device_transport_moves`,
// not `device_boundary_copies` -- and the Epoch next to it, now reading a
// device arc on both sides, is a same-space copy that crosses nothing
// either (RunStats::device_boundary_copies' own comment: "a same-space
// copy does not count"). Measured on this plan (FUSE_DEVICE_DEBUG=1):
// Sample -> Accumulate -> Transport -> Epoch -> Remap -> Diagnose -> Sink;
// only the Sink crosses, host -> device: one per period, down from two
// before this track (the former Epoch crossing is gone).
TEST(RunnerDevice, DeviceRegisteredFieldsCrossOnlyAtHostOnlyInstructionsAndAgreeWithHost) {
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
  // GREEN since the transport track: Transport's own no-graph arcs are now
  // device-resident and its move is a Kokkos::deep_copy, counted here, not
  // staged through host memory and not refused (RED before this track:
  // place_arcs_on_device() pinned Transport's arcs to host unconditionally,
  // which this scenario's device-registered chain would have crossed at
  // twice over, not once).
  ASSERT_GT(dev.transports, 0);
  EXPECT_EQ(dev.stats.device_transport_moves, dev.transports * 3);
  // Diagnose is the only instruction left that pins both the arc it reads
  // and the arc it writes; the Sink beside it (Diagnose's output feeding a
  // device-registered consumer field) is this plan's one remaining
  // crossing.
  const int host_only = dev.diagnoses;
  ASSERT_GT(host_only, 0);
  EXPECT_GT(dev.stats.device_boundary_copies, 0);
  EXPECT_LE(dev.stats.device_boundary_copies, host_only * 3)
      << "a device-registered chain crossed the boundary somewhere other "
         "than next to a host-only instruction";
  // The same count every period: nothing crosses at setup or only once.
  const Outcome one = run(s, Space::Device, 1);
  ASSERT_TRUE(one.init.ok()) << one.init.error().message;
  EXPECT_EQ(dev.stats.device_boundary_copies, 3 * one.stats.device_boundary_copies);
  // The one kernel whose output a host-only instruction reads (the Remap
  // before the Diagnose) writes through a staging lane: one copy in, one
  // back, per instruction. Accumulate's output no longer stages -- Transport
  // reads it directly now that Transport's own arcs are device-resident, so
  // Accumulate's output was never pinned by place_arcs_on_device() in the
  // first place. No field and no other arc is staged.
  ASSERT_GT(dev.remaps, 0);
  EXPECT_EQ(dev.stats.device_staged_copies, 2 * dev.remaps * 3)
      << "something other than a host-only instruction's input was staged";
  EXPECT_EQ(host.stats.device_staged_copies, 0);
}

// The host-registered components in Device mode: the producer's field is
// mirrored once per producer step (the Sample after each tick refreshes
// it; Impl::component_epoch), never per instruction. The chain between the
// two fields is the same plan as the device-registered run above, so the
// difference between the two runs' crossing counts is exactly the mirror
// refreshes (one per producer step) minus the Sinks that no longer cross
// (the consumer's field is host memory here, like the Diagnose output the
// Sink reads): producer_steps - consumer_steps.
TEST(RunnerDevice, HostRegisteredFieldsAreMirroredOncePerProducerStep) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";
  Scenario s = mean_over_four_substeps();
  const Outcome host_fields = run(s, Space::Device, 2);
  s.device_fields = true;
  const Outcome device_fields = run(s, Space::Device, 2);
  ASSERT_TRUE(host_fields.init.ok()) << host_fields.init.error().message;
  ASSERT_TRUE(device_fields.init.ok()) << device_fields.init.error().message;
  const std::int64_t producer_steps = 2 * 4, consumer_steps = 2;
  EXPECT_EQ(host_fields.stats.device_boundary_copies - device_fields.stats.device_boundary_copies,
            producer_steps - consumer_steps)
      << "the producer's field was mirrored other than once per producer step";
  // A per-instruction staging design would also stage the 8 Sample reads;
  // the mirror replaces them, so the staged copies are the same two
  // host-only inputs as in the device-registered run.
  EXPECT_EQ(host_fields.stats.device_staged_copies, device_fields.stats.device_staged_copies);
}

// docs/dev/device-resident-arcs.md step 4: a RunnerConfig that leaves
// `execution_space` at `Default` takes the process environment's choice,
// which is how the `.device` ctest twins run every verification binary
// under the device interpreter without touching its source.
namespace {
struct ScopedEnv {
  const char* key;
  explicit ScopedEnv(const char* k, const char* value) : key(k) {
    ::setenv(key, value, 1);
  }
  ~ScopedEnv() { ::unsetenv(key); }
};
}  // namespace

TEST(RunnerDevice, DefaultFollowsFuseExecutionSpaceInTheEnvironment) {
  ::unsetenv("FUSE_EXECUTION_SPACE");
  const Outcome unset = run(mean_over_four_substeps(), Space::Default, 1);
  ASSERT_TRUE(unset.init.ok()) << unset.init.error().message;
  EXPECT_EQ(unset.stats.device_dispatches, 0);
  {
    const ScopedEnv env{"FUSE_EXECUTION_SPACE", "host"};
    const Outcome host = run(mean_over_four_substeps(), Space::Default, 1);
    ASSERT_TRUE(host.init.ok()) << host.init.error().message;
    EXPECT_EQ(host.stats.device_dispatches, 0);
  }
  {
    const ScopedEnv env{"FUSE_EXECUTION_SPACE", "Device"};
    const Outcome dev = run(mean_over_four_substeps(), Space::Default, 1);
    if constexpr (kHasDeviceBackend) {
      ASSERT_TRUE(dev.init.ok()) << dev.init.error().message;
      EXPECT_GT(dev.stats.device_dispatches, 0);
      EXPECT_GT(dev.stats.device_resident_arcs, 0);
    } else {
      ASSERT_FALSE(dev.init.ok());
      EXPECT_EQ(dev.init.error().code, "E-RT-DEVICE");
    }
  }
  {
    // An explicit choice is never overridden by the environment.
    const ScopedEnv env{"FUSE_EXECUTION_SPACE", "device"};
    const Outcome host = run(mean_over_four_substeps(), Space::Host, 1);
    ASSERT_TRUE(host.init.ok()) << host.init.error().message;
    EXPECT_EQ(host.stats.device_dispatches, 0);
  }
  {
    const ScopedEnv env{"FUSE_EXECUTION_SPACE", "gpu"};
    const Outcome bad = run(mean_over_four_substeps(), Space::Default, 1);
    ASSERT_FALSE(bad.init.ok());
    EXPECT_EQ(bad.init.error().code, "E-RT-SPACE");
  }
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

// ---------------------------------------------------------------------------
// docs/dev/device-resident-arcs.md step 3: an ordinary (non-window) iteration
// region. Its block arcs are read and written by iteration.cpp's host-side
// bookkeeping (accelerator gather/scatter, checkpoint snapshots, the verify
// compare) only through the space-aware slot helpers, so in Device mode they
// live on the device and the Picard sweep still converges to the same values
// in the same number of sweeps as the host run.
// ---------------------------------------------------------------------------
namespace {

// out = a + k * in on each cell, with rollback support: the fixed point of
// the two-way coupling is reached by Picard iteration with |k_a k_b| < 1.
class LinearBlock final : public fuse::Component {
 public:
  LinearBlock(std::string name, double a, double k, std::size_t n)
      : name_(std::move(name)), a_(a), k_(k), out_(n, 0.0), in_(n, 0.0) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == fuse::Intent::Provides;
      auto& buf = is_out ? out_ : in_;
      auto r = reg.bind(f, fuse::FieldRef{buf.data(),
                                          fuse::Layout{.stride = 1, .count = buf.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext& ctx) override {
    const double a = a_ + 0.25 * ctx.t_begin.to_double();
    for (std::size_t i = 0; i < out_.size(); ++i) {
      out_[i] = a + k_ * in_[i] + 0.01 * static_cast<double>(i);
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { saved_out_ = out_; saved_in_ = in_; return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { out_ = saved_out_; in_ = saved_in_; return fuse::Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(true, fuse::DeterminismLevel::Bitwise, false);
  }
  [[nodiscard]] const std::vector<double>& in() const { return in_; }
 private:
  std::string name_;
  double a_, k_;
  std::vector<double> out_, in_, saved_out_, saved_in_;
};

struct RegionOutcome {
  fuse::Result<void>  init = fuse::Result<void>::success();
  std::vector<double> a_in, b_in;
  fuse::RunStats      stats;
  int                 iterations = -1;
  bool                converged = false;
};

RegionOutcome run_region(Space space, int periods) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m", .source = "generated:line:6"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m, .dimension = 1});
  const auto ca = g.add_component(fuse::ComponentDecl{.name = "A", .ranks = {0, 1}, .interval = Rational{1}});
  const auto cb = g.add_component(fuse::ComponentDecl{.name = "B", .ranks = {1, 1}, .interval = Rational{1}});
  const auto a_out = g.add_field(ca, fuse::FieldSpec{.name = "a_out", .region = r1, .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto a_in  = g.add_field(ca, fuse::FieldSpec{.name = "a_in",  .region = r1, .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto b_in  = g.add_field(cb, fuse::FieldSpec{.name = "b_in",  .region = r2, .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(cb, fuse::FieldSpec{.name = "b_out", .region = r2, .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto fwd  = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated");
  g.channel(fwd).relax(0.5);
  g.channel(back).relax(0.5);
  // Both transfers measured (iteration_test.cpp's both_channels(): a single
  // measure on a two-channel implicit coupling group is E-ITER-MEASURE).
  fuse::ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(1e-10, "a_out"));
  crit.add(fuse::Converged::Relative(1e-10, "b_out"));
  g.iterate({fwd, back}).until(std::move(crit)).max_iterations(100);

  SerialMeshBackend backend;
  EXPECT_TRUE(backend.load(g.mesh(m)).ok());
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  // |k_a k_b| = 0.12 under relax 0.5: the Picard error contracts by about
  // 0.56 per sweep, 1e-10 in some 40 sweeps.
  components.add(std::make_unique<LinearBlock>("A", 1.0, 0.3, 6));
  components.add(std::make_unique<LinearBlock>("B", -2.0, 0.4, 6));
  FieldRegistry registry;
  for (auto& comp : components) EXPECT_TRUE(comp.register_fields(registry, g).ok());

  Runner runner{g, lowered.dag, sched.plan, backend};
  RunnerConfig cfg;
  cfg.execution_space = space;
  RegionOutcome o;
  o.init = runner.initialize(components, registry, cfg);
  if (!o.init.ok()) return o;
  for (int k = 0; k < periods; ++k) {
    const auto r = runner.run_period(components);
    EXPECT_TRUE(r.ok()) << r.error().message;
  }
  o.a_in = dynamic_cast<const LinearBlock*>(components.find("A"))->in();
  o.b_in = dynamic_cast<const LinearBlock*>(components.find("B"))->in();
  o.stats = runner.stats();
  if (!runner.iteration_stats().empty()) {
    o.iterations = runner.iteration_stats()[0].iterations;
    o.converged = runner.iteration_stats()[0].converged;
  }
  return o;
}

}  // namespace

TEST(RunnerDevice, OrdinaryIterationRegionConvergesOnTheDeviceLikeTheHost) {
  if constexpr (!kHasDeviceBackend) GTEST_SKIP() << "no device backend";
  const RegionOutcome host = run_region(Space::Host, 2);
  const RegionOutcome dev = run_region(Space::Device, 2);
  ASSERT_TRUE(host.init.ok()) << host.init.error().message;
  ASSERT_TRUE(dev.init.ok()) << dev.init.error().message;
  ASSERT_TRUE(host.converged) << "host run: " << host.iterations << " sweeps";
  EXPECT_TRUE(dev.converged) << "device run: " << dev.iterations << " sweeps";
  EXPECT_EQ(host.iterations, dev.iterations);
  ASSERT_EQ(host.a_in.size(), dev.a_in.size());
  ASSERT_EQ(host.b_in.size(), dev.b_in.size());
  // Relax runs on the device (1e-14 relative, D38's accepted contraction
  // gap); everything else in the loop is a copy, so the converged state
  // agrees to that precision, not just to the convergence tolerance.
  for (std::size_t i = 0; i < host.a_in.size(); ++i) {
    EXPECT_NEAR(host.a_in[i], dev.a_in[i], 1e-13 * std::max(1.0, std::abs(host.a_in[i]))) << "A cell " << i;
    EXPECT_NEAR(host.b_in[i], dev.b_in[i], 1e-13 * std::max(1.0, std::abs(host.b_in[i]))) << "B cell " << i;
  }
  EXPECT_EQ(host.stats.device_resident_arcs, 0);
  // RED on the step-2 tree: the region's block arcs were pinned to the host,
  // so no arc of this plan was device-resident.
  EXPECT_GT(dev.stats.device_resident_arcs, 0)
      << "the ordinary region's block arcs stayed on the host";
  EXPECT_GT(dev.stats.device_dispatches, 0);
}
