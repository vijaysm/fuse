// Scaled-bench track, D69 fix follow-up. KNOWN_ISSUES.md D69: a general
// (non-nodal) weighted Remap between two components on DISJOINT MPI rank
// sets, backed by `fuse::native::NativeMeshBackend`, used to build a
// SILENTLY EMPTY conservative matrix (`RunStats::remap_weight_nonzeros ==
// 0`) at every rank count above 1 -- no automatic remap-halo exchange, and
// no refusal either. This file is that fix's RED/GREEN row: two
// differently-resolved native quad meshes (2x2 and 3x3 cells), provider and
// consumer placed on disjoint rank sets, routed through the general
// weighted builder (no `.remap_method()` override). `Runner::initialize()`
// must now REFUSE by name (`E-RT-REMAP-NO-HALO`) rather than silently
// proceed to a zero-nonzero matrix -- the chosen fix (`src/runtime/
// runner.cpp`'s own comment on the D69 block explains why the automatic-
// halo alternative was not taken in this round).
//
// test/parallel/nodal_remap_runner.* and its serial twin exercise the
// NODAL path this same disjoint-placement shape also supports (an
// orthogonal, unaffected code path -- KNOWN_ISSUES.md D69's own header
// explains why the two differ); this file is the general-weighted path's
// own disjoint-placement row, which had none before this fix.

#include <fuse/ir/lowering.hpp>
#include <fuse/backend/native_backend.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/transport/transport.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::StepContext;

namespace {

int world_size() {
  int n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}
int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

/// A single real (`double`-backed) field, host-resident -- all that is
/// needed to get `Runner::initialize()` far enough to reach `resolve_remap`
/// for the single channel this file builds. No data is ever meaningfully
/// exchanged: `initialize()` is expected to refuse before any period runs.
class Side final : public fuse::Component {
 public:
  explicit Side(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    for (const fuse::FieldId f : g.component(*g.find_component(name_)).fields) {
      auto& buf = bufs[g.field(f).name];
      buf.assign(local_count(g, f), 0.0);
      const auto r = reg.bind(
          f, fuse::FieldRef{buf.data(), fuse::Layout{.stride = 1, .count = buf.size()},
                            fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override { return fuse::Result<void>::success(); }
  std::function<std::size_t(const CouplingGraph&, fuse::FieldId)> local_count;
  std::map<std::string, std::vector<double>> bufs;

 private:
  std::string name_;
};

std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST-TRANSPORT", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

/// Disjoint, matching `scaled_device_benchmark.cpp`'s own `placement_for`:
/// provider owns the first half of the ranks, consumer the rest. Both
/// `fuse_add_mpi_test`'s `RANKS 2 3` split this evenly (one rank each at
/// world 2) and unevenly (one rank, two ranks, at world 3) -- either way,
/// genuinely disjoint, which is D69's own repro condition.
struct Placement {
  fuse::RankSet provider;
  fuse::RankSet consumer;
};
[[nodiscard]] Placement placement_for(int world) {
  const int half = world / 2;
  return Placement{fuse::RankSet{0, half}, fuse::RankSet{half, world - half}};
}

struct Harness {
  CouplingGraph                   graph;
  fuse::native::NativeMeshBackend backend;
  fuse::OperatorDAG               dag;
  fuse::ExecutionPlan             plan;
  ComponentSet                    components;
  FieldRegistry                   registry;

  explicit Harness(int rank, int world) : backend(rank, world) {}
};

/// Two differently-resolved native quad meshes (2x2 and 3x3 cells, both
/// plain FV at dimension 2) on disjoint rank sets, through a GENERAL
/// (non-nodal) weighted channel -- no `.remap_method()` override, so
/// `Runner::Impl::resolve_remap()` takes the branch this file's own D69
/// fix touches, never the nodal-distributed one.
[[nodiscard]] std::unique_ptr<Harness> build() {
  const int rank = world_rank();
  const int world = world_size();
  const Placement pl = placement_for(world);
  auto h = std::make_unique<Harness>(rank, world);
  CouplingGraph& g = h->graph;

  const auto mesh_provider =
      g.add_mesh(fuse::MeshDecl{.name = "provider_mesh", .source = "generated:quad:2x2"});
  const auto mesh_consumer =
      g.add_mesh(fuse::MeshDecl{.name = "consumer_mesh", .source = "generated:quad:3x3"});
  const auto r_provider =
      g.add_region(fuse::Region{.name = "r_provider", .mesh = mesh_provider, .dimension = 2});
  const auto r_consumer =
      g.add_region(fuse::Region{.name = "r_consumer", .mesh = mesh_consumer, .dimension = 2});

  const auto provider_id = g.add_component(
      fuse::ComponentDecl{.name = "provider", .ranks = pl.provider, .interval = Rational{1}});
  const auto consumer_id = g.add_component(
      fuse::ComponentDecl{.name = "consumer", .ranks = pl.consumer, .interval = Rational{1}});

  const auto out = g.add_field(
      provider_id, fuse::FieldSpec{.name = "out", .region = r_provider,
                                  .units = fuse::units::kKelvin, .intent = fuse::Intent::Provides});
  const auto in = g.add_field(
      consumer_id, fuse::FieldSpec{.name = "in", .region = r_consumer,
                                  .units = fuse::units::kKelvin, .intent = fuse::Intent::Requires});
  g.add_channel("link", out, in);  // no remap_method(): the general weighted path

  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    EXPECT_TRUE(h->backend.load(g.mesh(fuse::MeshId{i}), g).ok());
  }

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;
  const fuse::ClockSet clocks{g};
  const auto sched = schedule(g, h->dag, clocks, fuse::make_region_sizes(g, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  // Real local counts, read off the backend's own placed region size --
  // `Side::register_fields` only needs a non-throwing buffer size, since
  // this test never reaches a period (initialize() is expected to refuse
  // first); a 0-sized bind for a rank hosting neither component is legal.
  // Captures the backend's own STABLE heap address (h.get(), via the
  // Harness's unique_ptr) -- NOT `&h`, which names the local variable this
  // function is about to move out of on return, and would dangle the
  // instant it does.
  fuse::native::NativeMeshBackend* const backend_ptr = &h->backend;
  auto provider_side = std::make_unique<Side>("provider");
  provider_side->local_count = [backend_ptr, provider_id](const CouplingGraph& gg,
                                                           fuse::FieldId f) {
    const auto sz = backend_ptr->placed_region_size(gg.region(gg.field(f).region), provider_id);
    return sz.ok() ? sz.value() : std::size_t{0};
  };
  auto consumer_side = std::make_unique<Side>("consumer");
  consumer_side->local_count = [backend_ptr, consumer_id](const CouplingGraph& gg,
                                                           fuse::FieldId f) {
    const auto sz = backend_ptr->placed_region_size(gg.region(gg.field(f).region), consumer_id);
    return sz.ok() ? sz.value() : std::size_t{0};
  };
  h->components.add(std::move(provider_side));
  h->components.add(std::move(consumer_side));
  return h;
}

}  // namespace

TEST(NativeRemapHalo, DisjointGeneralWeightedRemapRefusesRatherThanBuildingAnEmptyMatrix) {
  // D69's own repro shape, at whatever rank count this ctest row runs
  // (`fuse_add_mpi_test`'s own RANKS 2 3): provider and consumer never
  // share a rank, so every rank's local-only intersection search would
  // otherwise see only one side's geometry. Before this fix,
  // `initialize()` returned success with `RunStats::remap_weight_nonzeros
  // == 0`; now it must refuse by name instead.
  ASSERT_GE(world_size(), 2) << "this row is registered RANKS 2 3; at 1 rank "
                                "provider and consumer coincide and nothing is disjoint";
  auto h = build();

  // The CALLER's responsibility, not Runner::initialize()'s own (every
  // other Harness/run() helper in this tree does the same, immediately
  // before constructing Runner -- scaled_device_benchmark.cpp's own
  // run_one(), runner_device_test.cpp's run()).
  for (auto& comp : h->components) {
    const auto r = comp.register_fields(h->registry, h->graph);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();

  const auto init = runner.initialize(h->components, h->registry, cfg);
  ASSERT_FALSE(init.ok()) << "expected E-RT-REMAP-NO-HALO; initialize() succeeded instead "
                             "(the D69 fix regressed, or an automatic halo now makes this "
                             "channel genuinely buildable -- in either case, update this test)";
  EXPECT_EQ(init.error().code, "E-RT-REMAP-NO-HALO");
}
