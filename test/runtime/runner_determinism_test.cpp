// Matrix cell A-SPACES-TRANSPORT.
//
// A Runner-level, integration proof that the coupling hot loop's move onto
// Kokkos' default host execution space (fuse/num/space_kernels.hpp) is
// thread-count-independent, not just at the kernel level
// (test/num/space_kernels_test.cpp) but through a real iteration region:
// Relax + Residual (elementwise) feed the SAME block-value/block-residual
// reduction run_region() (src/runtime/iteration.cpp) uses to decide
// convergence.
//
// WHY A NEW FIXTURE RATHER THAN REUSING runner_test.cpp's Harness. That
// harness's Producer/Consumer pair is anonymous-namespace-local to
// runner_test.cpp and not something a second translation unit can include;
// duplicating the small slice this test needs (rather than factoring the
// whole harness out, which is not this closing's job) is the same call
// test/runtime/checkpoint_test.cpp already made for its own one_measure()
// helper.
//
// WHY 2048 CELLS, NOT 8. Every EXISTING iteration-region fixture in this
// tree (runner_test.cpp, checkpoint_test.cpp, verification/iteration_test.cpp)
// uses a field short enough to fit in ONE fixed chunk
// (space_kernels.hpp's kDeterministicChunk == 256) -- exactly where the
// deterministic reduction is bit-identical to the single-accumulator sum it
// replaced, by that header's own argument, and therefore where a broken
// chunk boundary or a broken combine step could hide. 2048 cells spans 8
// chunks, so a wrong chunk boundary, a wrong combine order, or a
// reassociation that leaked thread-count dependence would have somewhere to
// show up.
//
// WHAT "THREAD-COUNT INDEPENDENT" MEANS HERE. Kokkos reads OMP_NUM_THREADS
// once, at Kokkos::initialize(); this binary cannot vary it mid-process.
// The proof is therefore across PROCESSES: fuse_add_omp_legs (test/
// CMakeLists.txt) runs this same binary once per thread count from 1 to the
// host's core count, and every run asserts the SAME hardcoded literal via
// ASSERT_DOUBLE_EQ (bit-for-bit, no tolerance) -- exactly the
// compare-runs.cmake convention examples/*.omp_identical already uses,
// spelled as a gtest assertion instead of a byte-diff of stdout because the
// value under test is a field inside the process, not stdout. The literals
// below are MEASURED (this test's own first passing run, at whatever thread
// count the sandbox gave it), not derived analytically; the property this
// test pins is that every OTHER thread count reproduces them exactly.

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include <memory>
#include <string>
#include <vector>

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::SerialMeshBackend;
using fuse::StepContext;
namespace u = fuse::units;

namespace {

constexpr std::size_t kCells = 2048;  // > kDeterministicChunk (256): see above

/// Writes a non-trivial, non-uniform field every advance -- a constant input
/// would let a wrong index range or a dropped chunk pass unnoticed.
class RampProducer final : public fuse::Component {
 public:
  RampProducer() : data_(kCells, 0.0) {}
  [[nodiscard]] std::string name() const override { return "producer"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = kCells},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext&) override {
    ++steps_;
    for (std::size_t i = 0; i < kCells; ++i) {
      data_[i] = std::sin(0.013 * static_cast<double>(i)) +
                0.001 * static_cast<double>(steps_) *
                    static_cast<double>(i % 37);
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_ = data_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    data_ = saved_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

 private:
  std::vector<double> data_;
  std::vector<double> saved_;
  int                  steps_ = 0;
};

class SinkConsumer final : public fuse::Component {
 public:
  SinkConsumer() : data_(kCells, 0.0) {}
  [[nodiscard]] std::string name() const override { return "consumer"; }

  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1,
                                                       .count = kCells},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override {
    saved_ = data_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    data_ = saved_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }
  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::vector<double> data_;
  std::vector<double> saved_;
};

}  // namespace

// A Picard-relaxed (Relax, omega < 1) iteration region on a 2048-element
// channel: every sweep runs Transport + Relax + Residual, and run_region()'s
// block_value/block_residual reduction (iteration.cpp) is exactly the
// deterministic_norm_squared/deterministic_norm_squared_diff path this
// closing added. `max_iterations` is generous; the test does not require
// convergence, only that whichever iterate the run stops at (converged or
// exhausted) is thread-count-identical -- convergence itself is already
// covered by test/verification/iteration_test.cpp's
// ResidualHistoryDecreasesMonotonicallyForPicard and is not re-proven here.
TEST(RunnerDeterminism, LargeIterationRegionIsBitIdenticalAcrossThreadCounts) {
  CouplingGraph g;
  const std::string mesh_source = "generated:line:" + std::to_string(kCells);
  const auto m1 =
      g.add_mesh(fuse::MeshDecl{.name = "m1", .source = mesh_source});
  const auto m2 =
      g.add_mesh(fuse::MeshDecl{.name = "m2", .source = mesh_source});
  const auto r1 =
      g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 =
      g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(
      fuse::ComponentDecl{.name = "producer", .ranks = fuse::RankSet{0, 1},
                          .interval = Rational{1}});
  const auto c = g.add_component(
      fuse::ComponentDecl{.name = "consumer", .ranks = fuse::RankSet{0, 1},
                          .interval = Rational{1}});
  const auto s = g.add_field(
      p, fuse::FieldSpec{.name = "out", .region = r1, .units = u::kKelvin,
                         .intent = fuse::Intent::Provides});
  const auto t = g.add_field(
      c, fuse::FieldSpec{.name = "in", .region = r2, .units = u::kKelvin,
                         .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", s, t);

  fuse::ConvergenceCriterion conv;
  conv.add(fuse::Converged::Relative(1e-9, "link"));
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "region", .channels = {ch}, .max_iterations = 20,
      .convergence = conv,
      .acceleration = fuse::AccelConfig{.kind = fuse::AccelKind::Constant,
                                        .initial_relaxation = 0.6}});

  SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(m1)).ok());
  ASSERT_TRUE(backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  const auto sched =
      schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  ComponentSet components;
  components.add(std::make_unique<RampProducer>());
  components.add(std::make_unique<SinkConsumer>());
  FieldRegistry registry;
  for (auto& comp : components) {
    ASSERT_TRUE(comp.register_fields(registry, g).ok());
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_TRUE(runner.initialize(components, registry).ok());
  ASSERT_TRUE(runner.run_period(components).ok());

  const auto* consumer = dynamic_cast<SinkConsumer*>(components.find("consumer"));
  ASSERT_NE(consumer, nullptr);
  const std::vector<double>& out = consumer->data();
  ASSERT_EQ(out.size(), kCells);

  // MEASURED on this test's own first passing run (Kokkos::DefaultHostExecution
  // Space at whatever thread count the sandbox gave it): the property this
  // pins is that every OTHER registered omp leg (OMP_NUM_THREADS 1..N,
  // fuse_add_omp_legs below) reproduces every one of these exactly, not that
  // these particular sixteen digits are independently meaningful.
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_DOUBLE_EQ(out[1], 0.032999633836427428);
  EXPECT_DOUBLE_EQ(out[512], 0.98423821113845833);
  EXPECT_DOUBLE_EQ(out[1023], 1.148826678362836);
  EXPECT_DOUBLE_EQ(out[1024], 1.1784343299144513);
  EXPECT_DOUBLE_EQ(out[2000], 0.80255845047960273);
  EXPECT_DOUBLE_EQ(out[kCells - 1], 1.2357214548870061);

  double checksum = 0.0;
  for (std::size_t i = 0; i < kCells; ++i) checksum += out[i] * out[i];
  EXPECT_DOUBLE_EQ(checksum, 1426.2068496377644);

  const auto& region_stats = runner.iteration_stats();
  ASSERT_FALSE(region_stats.empty());
  EXPECT_EQ(region_stats[0].iterations, 20);
  EXPECT_DOUBLE_EQ(region_stats[0].residual_history.back(),
                  1.8888945275990152);
}
