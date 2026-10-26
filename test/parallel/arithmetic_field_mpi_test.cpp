// A field of double-double values between two components on MPI ranks, through a
// real distributed remap (plans/2026-10-22-universal-number-types.md, stage U4):
// a general weighted conservative channel between two differently resolved native
// quad meshes, as in native_remap_halo_runner_test.cpp, with the components on
// disjoint ranks (provider on the first half, consumer on the rest). Overlapping
// placements are not covered: the native backend refuses to build the weights for
// them. What this adds to the single-rank tests: the limbs of a dd value cross
// ranks through the MPI transport and the halo, the remap runs in dd on the ranks
// that own target cells, and the conservation check sums its integrals over the
// ranks in dd (src/runtime/arithmetic_reduce.hpp).
//
// The configuration is run twice, once with the field as dd and once as binary64
// (the control), and the dd run is held to the control: the delivered values agree
// to a few binary64 ulps (the dd row sum is one rounding of the exact one, the
// binary64 one a rounding per term), and the conservation check reaches the same
// verdict, with the same error, on every rank. The check's absolute verdict is not
// asserted, because for a given placement it is whatever the binary64 check says.

#include <fuse/ir/lowering.hpp>
#include <fuse/backend/native_backend.hpp>
#include <fuse/num/dd.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/transport/transport.hpp>

#include <algorithm>
#include <cmath>
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
using fuse::num::dd;

namespace {

int world_size() { int n = 0; MPI_Comm_size(MPI_COMM_WORLD, &n); return n; }
int world_rank() { int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }

enum class Kind { Dd, Binary64 };

/// One field, host-resident, as dd values or as binary64: `raw` holds the limbs (two
/// per cell for dd) and starts as `fill` in every cell for the provider, zero for the
/// consumer.
class Side final : public fuse::Component {
 public:
  Side(std::string name, Kind kind) : name_(std::move(name)), kind_(kind) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    for (const fuse::FieldId f : g.component(*g.find_component(name_)).fields) {
      const std::size_t n = local_count(g, f);
      auto& buf = bufs[g.field(f).name];
      fuse::Result<void> r = fuse::Result<void>::success();
      if (kind_ == Kind::Dd) {
        buf.assign(2 * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
          buf[2 * i] = fill.hi;
          buf[2 * i + 1] = fill.lo;
        }
        r = reg.bind(f, fuse::BasicFieldRef<dd>(reinterpret_cast<dd*>(buf.data()), fuse::Layout{1, 1, n, 1},
                                                fuse::MemorySpace::Host));
      } else {
        buf.assign(n, fill.hi);
        r = reg.bind(f, fuse::FieldRef(buf.data(), fuse::Layout{1, 0, n, 1}, fuse::MemorySpace::Host));
      }
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override { return fuse::Result<void>::success(); }
  std::function<std::size_t(const CouplingGraph&, fuse::FieldId)> local_count;
  std::map<std::string, std::vector<double>> bufs;
  dd fill{};

 private:
  std::string name_;
  Kind kind_;
};

std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST-TRANSPORT", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

struct Harness {
  CouplingGraph graph;
  fuse::native::NativeMeshBackend backend;
  fuse::OperatorDAG dag;
  fuse::ExecutionPlan plan;
  ComponentSet components;
  FieldRegistry registry;
  explicit Harness(int rank, int world) : backend(rank, world) {}
};

std::unique_ptr<Harness> build(Kind kind, fuse::Conservation conservation, dd fill) {
  const int rank = world_rank(), world = world_size();
  // disjoint, as in native_remap_halo_runner_test.cpp: provider on the first half of the
  // ranks, consumer on the rest (one rank each at 2, one and two at 3)
  const int half = world / 2;
  const fuse::RankSet provider_ranks{0, half};
  const fuse::RankSet consumer_ranks{half, world - half};
  auto h = std::make_unique<Harness>(rank, world);
  CouplingGraph& g = h->graph;
  const auto mesh_provider = g.add_mesh(fuse::MeshDecl{.name = "provider_mesh", .source = "generated:quad:2x2"});
  const auto mesh_consumer = g.add_mesh(fuse::MeshDecl{.name = "consumer_mesh", .source = "generated:quad:3x3"});
  const auto r_provider = g.add_region(fuse::Region{.name = "r_provider", .mesh = mesh_provider, .dimension = 2});
  const auto r_consumer = g.add_region(fuse::Region{.name = "r_consumer", .mesh = mesh_consumer, .dimension = 2});
  const auto provider_id = g.add_component(fuse::ComponentDecl{.name = "provider", .ranks = provider_ranks, .interval = Rational{1}});
  const auto consumer_id = g.add_component(fuse::ComponentDecl{.name = "consumer", .ranks = consumer_ranks, .interval = Rational{1}});
  // the graph counts binary64 values per cell: a dd value is two
  const int comps = kind == Kind::Dd ? 2 : 1;
  const auto out = g.add_field(provider_id, fuse::FieldSpec{.name = "out", .region = r_provider, .components = comps,
                                                            .units = fuse::units::kKelvin, .conservation = conservation,
                                                            .intent = fuse::Intent::Provides});
  const auto in = g.add_field(consumer_id, fuse::FieldSpec{.name = "in", .region = r_consumer, .components = comps,
                                                           .units = fuse::units::kKelvin, .conservation = conservation,
                                                           .intent = fuse::Intent::Requires});
  const auto link = g.add_channel("link", out, in);
  g.channel(link).remap(fuse::RemapFlags::Conservative);
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    EXPECT_TRUE(h->backend.load(g.mesh(fuse::MeshId{i}), g).ok());
  }
  h->backend.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, h->dag, clocks, fuse::make_region_sizes(g, h->dag, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  fuse::native::NativeMeshBackend* const backend_ptr = &h->backend;
  auto provider_side = std::make_unique<Side>("provider", kind);
  provider_side->fill = fill;
  provider_side->local_count = [backend_ptr, provider_id](const CouplingGraph& gg, fuse::FieldId f) {
    const auto sz = backend_ptr->placed_region_size(gg.region(gg.field(f).region), provider_id);
    return sz.ok() ? sz.value() : std::size_t{0};
  };
  auto consumer_side = std::make_unique<Side>("consumer", kind);
  consumer_side->local_count = [backend_ptr, consumer_id](const CouplingGraph& gg, fuse::FieldId f) {
    const auto sz = backend_ptr->placed_region_size(gg.region(gg.field(f).region), consumer_id);
    return sz.ok() ? sz.value() : std::size_t{0};
  };
  h->components.add(std::move(provider_side));
  h->components.add(std::move(consumer_side));
  return h;
}

struct Outcome {
  std::vector<double> delivered;  // this rank's consumer cells, one value each (dd: the nearest binary64 value)
  std::vector<double> delivered_lo;  // dd only: the lo limbs
  fuse::RunStats stats;
  std::size_t arithmetic_fields = 0;
};

Outcome run(Kind kind, fuse::Conservation conservation, dd fill) {
  Outcome out;
  auto h = build(kind, conservation, fill);
  for (auto& comp : h->components) {
    const auto r = comp.register_fields(h->registry, h->graph);
    EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
  }
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  cfg.check_conservation = true;
  const auto init = runner.initialize(h->components, h->registry, cfg);
  EXPECT_TRUE(init.ok()) << (init.ok() ? std::string{} : init.error().message);
  if (!init.ok()) return out;
  const auto rp = runner.run_period(h->components);
  EXPECT_TRUE(rp.ok()) << (rp.ok() ? std::string{} : rp.error().message);
  out.stats = runner.stats();
  out.arithmetic_fields = runner.stats().arithmetic_fields.size();
  Side* const consumer = dynamic_cast<Side*>(h->components.find("consumer"));
  if (consumer != nullptr) {
    const std::vector<double>& raw = consumer->bufs["in"];
    if (kind == Kind::Dd) {
      for (std::size_t i = 0; i + 1 < raw.size(); i += 2) {
        out.delivered.push_back(raw[i] + raw[i + 1]);
        out.delivered_lo.push_back(raw[i + 1]);
      }
    } else {
      out.delivered = raw;
    }
  }
  return out;
}

void expect_dd_agrees_with_the_control() {
  ASSERT_GE(world_size(), 2) << "registered for 2 and 3 ranks";
  // 7.1 is not exactly representable and the lo limb holds more of the value than
  // binary64 can, so a transfer that drops limbs shows in the lo limbs
  const dd fill = dd::from_sum(7.1, 0x1p-57 * 1.3);
  // Intensive: a uniform field must come out uniform, whatever the overlaps
  const Outcome d = run(Kind::Dd, fuse::Conservation::Intensive, fill);
  const Outcome b = run(Kind::Binary64, fuse::Conservation::Intensive, fill);
  EXPECT_EQ(d.arithmetic_fields, 2u);
  EXPECT_EQ(b.arithmetic_fields, 0u);

  // the same cells, the same transfer: dd and binary64 differ by binary64 round-off
  ASSERT_EQ(d.delivered.size(), b.delivered.size());
  std::size_t nonzero = 0;
  for (std::size_t i = 0; i < d.delivered.size(); ++i) {
    if (b.delivered[i] == 0.0 && d.delivered[i] == 0.0) continue;  // a ghost or unowned entry
    ++nonzero;
    EXPECT_NEAR(d.delivered[i], b.delivered[i], 16 * 0x1p-53 * std::fabs(b.delivered[i])) << "cell " << i;
    EXPECT_NEAR(d.delivered[i], fill.hi, 1e-13 * fill.hi) << "cell " << i;
  }
  // and the dd run kept more than the leading limb: some delivered cell has a lo limb
  bool any_lo = false;
  for (const double lo : d.delivered_lo) any_lo = any_lo || lo != 0.0;
  if (nonzero > 0) EXPECT_TRUE(any_lo);

  // the conservation check: the same number of checks, the same verdict, and the same
  // error to the accuracy of the control's own sums, on every rank
  EXPECT_EQ(d.stats.conservation_checks, b.stats.conservation_checks);
  EXPECT_EQ(d.stats.conservation_violations, b.stats.conservation_violations);
  const double scale = std::max(1.0, std::fabs(b.stats.worst_conservation_error));
  EXPECT_NEAR(d.stats.worst_conservation_error, b.stats.worst_conservation_error, 1e-12 * scale);
  double lo = 0.0, hi = 0.0;
  const double mine = d.stats.worst_conservation_error;
  ASSERT_EQ(MPI_Allreduce(&mine, &lo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD), MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&mine, &hi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD), MPI_SUCCESS);
  EXPECT_EQ(lo, hi) << "the ranks disagree on the conservation error: the reduction is not one sum";
}

}  // namespace

TEST(ArithmeticFieldMpi, ADoubleDoubleFieldOnDisjointRanksAgreesWithTheBinary64Control) {
  expect_dd_agrees_with_the_control();
}
