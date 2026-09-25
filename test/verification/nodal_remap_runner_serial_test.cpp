// The one-rank case of test/parallel/nodal_remap_runner_test.cpp, without MPI.
//
// That file is registered only under FUSE_ENABLE_MPI, and its `.1` case failed
// on the user's host with no message this sandbox could obtain: MPI cannot
// initialise here at all, not even at one process. But a single participant
// needs no MPI to be meaningful -- every placement below is the whole mesh,
// the distributed path is not reached, and MultiMeshGeometricBackend now
// compiles without MPI (test/support/partitioned_test_backend.hpp). So the
// same graph, the same backend and the same channels run here, where they can
// actually be debugged.
//
// It is NOT a duplicate of interface_transfer_test.cpp's own consistent pair:
// that one drives a generated SerialMeshBackend mesh, and the difference
// between the two fixtures is exactly what had to be bisected.

#include <fuse/backend/consistent_interp.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "../support/partitioned_test_backend.hpp"
#include "fuse_test_main.hpp"

namespace u = fuse::units;
using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::StepContext;

namespace {

inline constexpr fuse::Units kMetersPerSecond{.length = 1, .time = -1};

class Side final : public fuse::Component {
 public:
  Side(std::string name, std::function<void(Side&)> fill)
      : name_(std::move(name)), fill_(std::move(fill)) {}
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
  fuse::Result<void> advance(const StepContext&) override {
    fill_(*this);
    return fuse::Result<void>::success();
  }
  std::function<std::size_t(const CouplingGraph&, fuse::FieldId)> local_count;
  std::map<std::string, std::vector<double>> bufs;

 private:
  std::string name_;
  std::function<void(Side&)> fill_;
};

/// The single, unpartitioned reference the MPI fixture compares against,
/// reproduced here: H built nodal-side = beam, eval-side = fluid cells, then
/// its transpose scaled by the fluid cell measure (an FV source is a density,
/// not yet a load).
std::vector<double> compute_reference(std::size_t n_beam, std::size_t n_fluid) {
  GeometricLineMesh ref_beam(n_beam, /*num_ranks_in_set=*/1, /*my_rank_in_set=*/0, 0.0, 1.0);
  GeometricLineMesh ref_fluid(n_fluid, 1, 0, 0.0, 1.0);
  const auto h = fuse::build_consistent_interp_weights(ref_beam, ref_fluid, /*tgt_dim=*/1);
  if (!h.ok()) throw std::runtime_error("reference H: " + h.error().message);
  const fuse::RemapWeights& hw = h.value();
  fuse::RemapWeights ht;
  ht.n_rows = hw.n_cols;
  ht.n_cols = hw.n_rows;
  for (std::size_t k = 0; k < hw.val.size(); ++k) {
    const auto fluid_cell = static_cast<std::size_t>(hw.row[k]);
    ht.row.push_back(hw.col[k]);
    ht.col.push_back(hw.row[k]);
    ht.val.push_back(hw.val[k] * ref_fluid.measure(1, fluid_cell));
  }
  const std::vector<double> density(n_fluid, 1.0);
  std::vector<double> load(n_beam + 1, 0.0);
  const auto applied = fuse::apply_weights(ht, density.data(), load.data(), 1);
  if (!applied.ok()) throw std::runtime_error("reference H^T D: " + applied.error().message);
  return load;
}

struct Outcome {
  std::vector<double> fluid_in, fluid_force, beam_load, beam_velocity;
};

/// The same graph the MPI fixture builds, at one participant.
Outcome run_pair(std::size_t n_beam, std::size_t n_fluid, double vel_a, double vel_b) {
  CouplingGraph g;
  const auto m_beam = g.add_mesh(fuse::MeshDecl{.name = "beam_mesh", .source = "n/a"});
  const auto m_fluid = g.add_mesh(fuse::MeshDecl{.name = "fluid_mesh", .source = "n/a"});
  const auto r_beam = g.add_region(fuse::Region{.name = "r_beam", .mesh = m_beam, .dimension = 1});
  const auto r_fluid = g.add_region(fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});
  const fuse::RankSet one{0, 1};
  const auto beam = g.add_component(
      fuse::ComponentDecl{.name = "beam", .ranks = one, .interval = Rational{1}});
  const auto fluid = g.add_component(
      fuse::ComponentDecl{.name = "fluid", .ranks = one, .interval = Rational{1}});
  const auto vel_out = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_velocity", .region = r_beam, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = kMetersPerSecond, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto vel_in = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_velocity_in", .region = r_fluid, .units = kMetersPerSecond,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Requires});
  const auto vch = g.add_channel("wall_velocity", vel_out, vel_in);
  g.channel(vch).remap_method("consistent");
  const auto force_out = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_force", .region = r_fluid, .units = u::kNewton,
      .conservation = fuse::Conservation::Extensive, .intent = fuse::Intent::Provides});
  const auto force_in = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_load", .region = r_beam, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = u::kNewton, .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto fch = g.add_channel("wall_force", force_out, force_in);
  g.channel(fch).remap_method("consistent-transpose").conservative().lag(1)
      .allow_time_level_mismatch("capability test: only transfer accuracy is under test");

  MultiMeshGeometricBackend backend;
  backend.add_mesh_spec("beam_mesh", {n_beam, one, 0.0, 1.0});
  backend.add_mesh_spec("fluid_mesh", {n_fluid, one, 0.0, 1.0});
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto r = backend.load(g.mesh(fuse::MeshId{i}));
    if (!r.ok()) throw std::runtime_error("load: " + r.error().message);
  }

  const auto lowered = fuse::lower(g);
  if (!lowered.ok()) throw std::runtime_error("lower: " + lowered.diags.to_string());
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  if (!sched.ok()) throw std::runtime_error("schedule: " + sched.diags.to_string());

  const auto local_count = [&](const CouplingGraph& gg, fuse::FieldId fid) -> std::size_t {
    const auto& fs = gg.field(fid);
    const auto mid = backend.mesh_id(gg.mesh(gg.region(fs.region).mesh).name);
    if (!mid.ok()) throw std::runtime_error("local_count: " + mid.error().message);
    const auto view = backend.view(mid.value());
    if (!view.ok()) throw std::runtime_error("local_count: " + view.error().message);
    const int dim = fs.disc == fuse::Discretization::ContinuousGLL ? 0 : 1;
    return view.value()->entity_count(dim);
  };

  auto b = std::make_unique<Side>("beam", [&](Side& me) {
    auto& v = me.bufs["beam_velocity"];
    const auto mid = backend.mesh_id("beam_mesh").value();
    const auto* view = backend.view(mid).value();
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = vel_a + vel_b * view->centroid(0, i).x;
  });
  auto f = std::make_unique<Side>("fluid", [&](Side& me) {
    for (double& x : me.bufs["fluid_force"]) x = 1.0;
  });
  b->local_count = local_count;
  f->local_count = local_count;
  Side* bp = b.get();
  Side* fp = f.get();
  ComponentSet components;
  components.add(std::move(b));
  components.add(std::move(f));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    if (!r.ok()) throw std::runtime_error("register_fields: " + r.error().message);
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  RunnerConfig cfg;  // no base_communicator: one participant, no MPI in this build
  {
    const auto r = runner.initialize(components, registry, cfg);
    if (!r.ok()) throw std::runtime_error("initialize: " + r.error().code + ": " + r.error().message);
  }
  for (int period = 0; period < 2; ++period) {
    const auto r = runner.run_period(components);
    if (!r.ok()) throw std::runtime_error("run_period: " + r.error().code + ": " + r.error().message);
  }
  return Outcome{fp->bufs["fluid_velocity_in"], fp->bufs["fluid_force"],
                 bp->bufs["beam_load"], bp->bufs["beam_velocity"]};
}

}  // namespace

TEST(NodalRemapRunnerSerial, TheOneRankCaseOfTheMpiFixtureRunsAndIsExact) {
  // Exactly the three parameter sets the MPI cases use, so a failure here is
  // that case's failure and not a different problem wearing its name.
  struct Case { std::size_t n_beam, n_fluid; double a, b; const char* name; };
  for (const Case c : {Case{7, 11, 0.3, 2.0, "Disjoint/Identical (7, 11)"},
                       Case{8, 6, -0.5, 1.5, "PartiallyOverlapping (8, 6)"}}) {
    Outcome o;
    try {
      o = run_pair(c.n_beam, c.n_fluid, c.a, c.b);
    } catch (const std::exception& e) {
      ADD_FAILURE() << c.name << ": " << e.what();
      continue;
    }
    ASSERT_EQ(o.fluid_in.size(), c.n_fluid) << c.name;
    ASSERT_EQ(o.beam_load.size(), c.n_beam + 1) << c.name;
    const double h_fluid = 1.0 / static_cast<double>(c.n_fluid);
    for (std::size_t j = 0; j < o.fluid_in.size(); ++j) {
      const double x = (static_cast<double>(j) + 0.5) * h_fluid;
      EXPECT_NEAR(o.fluid_in[j], c.a + c.b * x, 1e-12)
          << c.name << ": H is not exact at fluid cell " << j;
    }
    const std::vector<double> ref = compute_reference(c.n_beam, c.n_fluid);
    for (std::size_t i = 0; i < o.beam_load.size(); ++i) {
      EXPECT_NEAR(o.beam_load[i], ref.at(i), 1e-12 * std::abs(ref.at(i)) + 1e-13)
          << c.name << ": beam vertex " << i << " differs from the serial reference";
    }
    double sent = 0.0, received = 0.0, p_fluid = 0.0, p_beam = 0.0;
    for (std::size_t j = 0; j < o.fluid_force.size(); ++j) {
      sent += o.fluid_force[j] * h_fluid;
      p_fluid += o.fluid_in[j] * o.fluid_force[j] * h_fluid;
    }
    for (std::size_t i = 0; i < o.beam_load.size(); ++i) {
      received += o.beam_load[i];
      p_beam += o.beam_velocity[i] * o.beam_load[i];
    }
    EXPECT_NEAR(received, sent, 1e-13 * std::abs(sent)) << c.name << ": H^T D lost force";
    EXPECT_NEAR(p_beam, p_fluid, 1e-13 * (std::abs(p_fluid) + std::abs(p_beam)))
        << c.name << ": interface power does not balance";
  }
}
