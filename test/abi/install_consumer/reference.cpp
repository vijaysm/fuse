// ABI-C-CONSUMER's in-process half (build-evidence/abi-matrix.toml): the
// SAME two-component, lag(1) graph main.c drives through fuse_coupler_abi(),
// here driven directly through fuse::Runner -- no ABI, no C at all. Built
// as part of the normal tier-0 test suite (it is plain fuse:: C++, like
// every other test here), unlike main.c, which only exists inside the
// standalone sub-build run-smoke-test.cmake configures against the install
// prefix. Prints the identical format main.c does, so the two can be
// string-compared byte for byte.
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "fuse/backend/serial_backend.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

#include "../graph_fixture.hpp"

namespace {

class Upstream final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "upstream"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component("upstream");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const auto r = reg.bind(f, fuse::FieldRef{storage_.data(),
                                                 fuse::Layout{.stride = 1, .count = 4},
                                                 fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    for (double& v : storage_) v = 10.0 + static_cast<double>(period_);
    ++period_;
    return fuse::Result<void>::success();
  }

 private:
  std::vector<double> storage_ = std::vector<double>(4, 0.0);
  int period_ = 0;
};

class Downstream final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "downstream"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component("downstream");
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const auto r = reg.bind(f, fuse::FieldRef{storage_.data(),
                                                 fuse::Layout{.stride = 1, .count = 4},
                                                 fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] const double* storage() const { return storage_.data(); }

 private:
  std::vector<double> storage_ = std::vector<double>(4, 0.0);
};

}  // namespace

int main() {
  // This translation unit has no fuse_test_main.cpp (it is a standalone
  // executable, not a gtest binary) and no fuse_create() (it drives
  // fuse::Runner directly, not through the ABI, whose own fuse_create()
  // would otherwise do this) -- so nothing else in this process ever calls
  // Kokkos::initialize().
  Kokkos::ScopeGuard kokkos_guard;
  fuse::CouplingGraph graph = fuse::abi_test::build_two_component_lag1_graph();
  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    if (!backend.load(graph.mesh(fuse::MeshId{i})).ok()) {
      std::fprintf(stderr, "backend.load failed\n");
      return 1;
    }
  }
  const fuse::LoweringResult lowered = fuse::lower(graph);
  if (!lowered.ok()) {
    std::fprintf(stderr, "lower failed: %s\n", lowered.diags.to_string().c_str());
    return 1;
  }
  const fuse::ClockSet clocks{graph};
  const fuse::ScheduleResult sched =
      fuse::schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, backend));
  if (!sched.ok()) {
    std::fprintf(stderr, "schedule failed: %s\n", sched.diags.to_string().c_str());
    return 1;
  }

  fuse::FieldRegistry registry;
  fuse::ComponentSet components;
  auto* down = new Downstream();
  components.add(std::make_unique<Upstream>());
  components.add(std::unique_ptr<fuse::Component>(down));
  for (auto& c : components) {
    if (!c.register_fields(registry, graph).ok()) {
      std::fprintf(stderr, "register_fields failed\n");
      return 1;
    }
  }

  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (!runner.initialize(components, registry).ok()) {
    std::fprintf(stderr, "Runner::initialize failed\n");
    return 1;
  }

  for (int period = 0; period < 2; ++period) {
    if (!runner.run_period(components).ok()) {
      std::fprintf(stderr, "run_period failed\n");
      return 1;
    }
    for (int i = 0; i < 4; ++i) {
      std::printf("period %d downstream.in[%d] = %a\n", period, i, down->storage()[i]);
    }
  }
  return 0;
}
