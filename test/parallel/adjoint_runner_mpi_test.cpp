// Reverse mode through the Runner when the components are on different ranks
// (plans/2026-10-22-universal-number-types.md, stage U6; docs/dev/adjoint-schedule.md).
//
// The coupling of adjoint_runner_test.cpp, with the producer and the consumer on
// disjoint ranks (the Transport between them is an MPI_Alltoallv, `MpiTransport::move`)
// or on partially overlapping ranks (a rank that holds both, so that some entries
// are matched on the rank and never travel). The components are linear
// (out <- a out + c in + drive), so the composed map G, from the initial states and the
// drives of every rank to the final states of every rank, is one matrix, and the
// derivative gate's identity
//
//   |<G x, w> - <x, G^T w>| <= c * eps * ||G||_est * ||x|| * ||w||
//
// holds for the whole distributed vectors. The gate is the library's own
// (fuse/linalg/derivative_gate.hpp) and takes whole vectors, so each rank's part is
// gathered onto every rank after each application; every rank then makes the
// same sequence of calls, which keeps the collective runs in step.
//
// Also here: the adjoint of the exchange by exact value (a seed on the consumer's
// delivered field reaches the producer entity with the same global id, with no
// rounding), the identity's control (a component whose adjoint leaves out the factor
// of its state fails the same gate), and the failure of one rank in the middle of a sweep
// (every rank that took part in the exchange comes out of the sweep with an error and
// none is left waiting).

#include <fuse/ir/lowering.hpp>
#include <fuse/linalg/derivative_gate.hpp>
#include <fuse/model/component.hpp>
#include <fuse/runtime/component.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/transport/transport.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"
#include "../support/partitioned_test_backend.hpp"

using fuse::AccumOp;
using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::StepContext;

namespace {

int world_size() { int n = 1; MPI_Comm_size(MPI_COMM_WORLD, &n); return n; }
int world_rank() { int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }

enum class Shape {
  Disjoint,  // the producer on the first half of the ranks, the consumer on the rest
  Partial,   // the producer on ranks 0..W-2, the consumer on 1..W-1: ranks 1..W-2 hold both
  Nested,    // the producer on ranks 0..W-2, the consumer on every rank: where the two blocks
             // of a rank share an entity the match is made on the rank and never travels
};

const char* shape_name(Shape s) {
  return s == Shape::Disjoint ? "disjoint" : s == Shape::Partial ? "partial" : "nested";
}

enum class Fault {
  None,
  SkipRecurrence,  // the adjoint of a step forgets the factor of the state: a wrong adjoint
  FailAdjoint,     // the consumer's adjoint refuses its step
};

struct Spec {
  Shape shape = Shape::Disjoint;
  bool loop = false;                // a lagged channel from the consumer back to the producer
  AccumOp accum = AccumOp::Instant;
  Rational fluid_dt{1};
  int fluid_steps = 1;              // steps of the producer in one period, by construction
  double a_f = 0.0, c_f = 0.0, a_s = 0.0, c_s = 0.0;
  Fault fluid_fault = Fault::None;
  Fault solid_fault = Fault::None;
  bool region = false;  // the channel across ranks is iterated to a convergence measure
};

fuse::RankSet fluid_ranks(Shape s) {
  const int w = world_size();
  return s == Shape::Disjoint ? fuse::RankSet{0, w / 2} : fuse::RankSet{0, w - 1};
}
fuse::RankSet solid_ranks(Shape s) {
  const int w = world_size();
  switch (s) {
    case Shape::Disjoint: return fuse::RankSet{w / 2, w - w / 2};
    case Shape::Partial: return fuse::RankSet{1, w - 1};
    case Shape::Nested: break;
  }
  return fuse::RankSet{0, w};
}

/// out <- a out + c in + drive(step), cell by cell, on the cells of this rank's block. Its adjoint
/// follows the conventions of `fuse::AdjointComponent`.
class MpiLin final : public fuse::Component, public fuse::AdjointComponent {
 public:
  MpiLin(std::string name, std::size_t cells, double a, double c, Fault fault)
      : name_(std::move(name)), cells_(cells), a_(a), c_(c), fault_(fault) {
    out_.assign(cells, 0.0);
    in_.assign(cells, 0.0);
    out_bar_.assign(cells, 0.0);
    in_bar_.assign(cells, 0.0);
  }
  [[nodiscard]] std::string name() const override { return name_; }
  [[nodiscard]] fuse::AdjointComponent* adjoint() override { return this; }
  [[nodiscard]] std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }

  fuse::Result<void> save_state() override {
    saved_out_ = out_;
    saved_in_ = in_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    out_ = saved_out_;
    in_ = saved_in_;
    return fuse::Result<void>::success();
  }

  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      (provides ? has_out_ : has_in_) = true;
      auto r = bind(reg, f, provides ? out_ : in_);
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> register_adjoint_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      auto r = bind(reg, f, provides ? out_bar_ : in_bar_);
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext& ctx) override {
    const auto d = drive.find(ctx.step);
    for (std::size_t k = 0; k < cells_; ++k) {
      out_[k] = a_ * out_[k] + c_ * in_[k] + (d != drive.end() ? d->second[k] : 0.0);
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance_adjoint(const StepContext& ctx) override {
    if (fault_ == Fault::FailAdjoint) {
      return fuse::Failure{"E-TEST-ADJOINT-REFUSED", "'" + name_ + "' refuses the adjoint of its step"};
    }
    std::vector<double>& db = drive_bar[ctx.step];
    db.assign(cells_, 0.0);
    for (std::size_t k = 0; k < cells_; ++k) {
      db[k] = out_bar_[k];             // d out / d drive = 1
      in_bar_[k] += c_ * out_bar_[k];  // a Requires field: add
      if (fault_ != Fault::SkipRecurrence) out_bar_[k] *= a_;  // a Provides field: overwrite
    }
    return fuse::Result<void>::success();
  }

  [[nodiscard]] bool has_out() const { return has_out_; }
  [[nodiscard]] bool has_in() const { return has_in_; }
  [[nodiscard]] std::size_t cells() const { return cells_; }

  std::vector<double> out_, in_, out_bar_, in_bar_;
  std::map<std::int64_t, std::vector<double>> drive;      // forward input at each step
  std::map<std::int64_t, std::vector<double>> drive_bar;  // its adjoint

 private:
  fuse::Result<void> bind(FieldRegistry& reg, fuse::FieldId f, std::vector<double>& v) const {
    return reg.bind(f, fuse::FieldRef(v.data(), fuse::Layout{1, 1, cells_, 1}, fuse::MemorySpace::Host));
  }

  std::string name_;
  std::size_t cells_;
  double a_, c_;
  Fault fault_;
  bool has_out_ = false, has_in_ = false;
  std::vector<double> saved_out_, saved_in_;
};

std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

constexpr std::size_t kEntities = 12;
constexpr int kDim = 2;

struct Net {
  fuse::RankSet fluid_set, solid_set;
  PartitionedTestBackend backend;
  CouplingGraph graph;
  fuse::OperatorDAG dag;
  fuse::ExecutionPlan plan;
  ComponentSet components;
  FieldRegistry registry, adjoint_registry;
  std::unique_ptr<Runner> runner;
  MpiLin* fluid = nullptr;
  MpiLin* solid = nullptr;
  fuse::RegionId region;
  fuse::MeshId mesh;

  explicit Net(const Spec& s)
      : fluid_set(fluid_ranks(s.shape)), solid_set(solid_ranks(s.shape)),
        backend(fluid_set, solid_set, kEntities, kDim) {
    auto& g = graph;
    mesh = g.add_mesh(fuse::MeshDecl{.name = "shared", .source = "test"});
    region = g.add_region(fuse::Region{.name = "r", .mesh = mesh, .entity_set = "", .dimension = kDim});
    const auto pf = g.add_component(fuse::ComponentDecl{.name = "fluid", .ranks = fluid_set, .interval = s.fluid_dt});
    const auto sc = g.add_component(fuse::ComponentDecl{.name = "solid", .ranks = solid_set, .interval = Rational{1}});
    const auto out = g.add_field(pf, fuse::FieldSpec{.name = "out", .region = region,
                                                     .units = fuse::units::kKelvin,
                                                     .intent = fuse::Intent::Provides});
    const auto in = g.add_field(sc, fuse::FieldSpec{.name = "in", .region = region,
                                                    .units = fuse::units::kKelvin,
                                                    .intent = fuse::Intent::Requires});
    const auto link = g.add_channel("link", out, in);
    if (s.accum != AccumOp::Instant) g.channel(link).accumulate(s.accum);
    if (s.loop) {
      const auto sout = g.add_field(sc, fuse::FieldSpec{.name = "out", .region = region,
                                                        .units = fuse::units::kKelvin,
                                                        .intent = fuse::Intent::Provides});
      const auto fin = g.add_field(pf, fuse::FieldSpec{.name = "in", .region = region,
                                                       .units = fuse::units::kKelvin,
                                                       .intent = fuse::Intent::Requires});
      const auto back = g.add_channel("back", sout, fin);
      g.channel(back).lag(1).allow_time_level_mismatch("the loop is closed by one step of lag");
    }
    if (s.region) {
      fuse::ConvergenceCriterion crit;
      crit.add(fuse::Converged::Relative(1e-6, "link"));
      g.add_iteration_region(fuse::IterationRegionDecl{
          .name = "outer", .channels = {link}, .max_iterations = 4, .convergence = crit});
    }
    // each component's own decomposition, laid out over its own ranks
    std::vector<fuse::RankSet> placements;
    for (std::uint32_t c = 0; c < g.component_count(); ++c) placements.push_back(g.component(fuse::ComponentId{c}).ranks);
    backend.set_placements(std::move(placements));
    EXPECT_TRUE(backend.load(g.mesh(mesh)).ok());
    const auto lowered = lower(g);
    EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
    dag = lowered.dag;
    const fuse::ClockSet clocks{g};
    auto sched = schedule(g, dag, clocks, fuse::make_region_sizes(g, dag, backend));
    EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
    plan = sched.plan;

    const auto size_of = [&](std::uint32_t cid) {
      const auto n = backend.placed_region_size(g.region(region), fuse::ComponentId{cid});
      EXPECT_TRUE(n.ok());
      return n.ok() ? n.value() : std::size_t{0};
    };
    auto lf = std::make_unique<MpiLin>("fluid", size_of(0), s.a_f, s.c_f, s.fluid_fault);
    auto ls = std::make_unique<MpiLin>("solid", size_of(1), s.a_s, s.c_s, s.solid_fault);
    fluid = lf.get();
    solid = ls.get();
    components.add(std::move(lf));
    components.add(std::move(ls));
    for (auto& c : components) {
      const auto r = c.register_fields(registry, graph);
      EXPECT_TRUE(r.ok()) << (r.ok() ? std::string{} : r.error().message);
    }
    runner = std::make_unique<Runner>(graph, dag, plan, backend);
  }

  /// COLLECTIVE.
  fuse::Result<void> init_forward() {
    fuse::RunnerConfig cfg;
    cfg.transport = "mpi";
    cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
    cfg.transport_setup = mpi_comm_setup();
    return runner->initialize(components, registry, cfg);
  }
  fuse::Result<void> init_adjoint() {
    for (auto& c : components) {
      auto r = c.adjoint()->register_adjoint_fields(adjoint_registry, graph);
      if (!r.ok()) return r;
    }
    return runner->initialize_adjoint(components, adjoint_registry);
  }
  std::vector<MpiLin*> members() { return {fluid, solid}; }
};

/// Sizes of this rank's part of the input x (the initial state of each field a component provides,
/// then the drive of the fluid at every step) and of the output y (what each component holds
/// after the run: its provided and its delivered fields).
struct Dims {
  std::size_t nx = 0, ny = 0;
};
Dims local_dims(const Spec& s, int periods) {
  Net n(s);
  Dims d;
  for (MpiLin* c : n.members()) {
    if (c->has_out()) d.nx += c->cells();
    d.ny += (c->has_out() ? c->cells() : 0) + (c->has_in() ? c->cells() : 0);
  }
  d.nx += n.fluid->cells() * static_cast<std::size_t>(periods * s.fluid_steps);
  return d;
}

/// y = G x on this rank's parts. COLLECTIVE.
fuse::Result<std::vector<double>> forward_map(const Spec& s, int periods, std::span<const double> x) {
  Net n(s);
  if (auto r = n.init_forward(); !r.ok()) return r.error();
  std::size_t at = 0;
  for (MpiLin* c : n.members()) {
    if (!c->has_out()) continue;
    for (double& v : c->out_) v = x[at++];
  }
  for (int step = 0; step < periods * s.fluid_steps; ++step) {
    std::vector<double>& d = n.fluid->drive[step];
    d.assign(n.fluid->cells(), 0.0);
    for (double& v : d) v = x[at++];
  }
  if (auto r = n.runner->run(n.components, periods); !r.ok()) return r.error();
  std::vector<double> y;
  for (MpiLin* c : n.members()) {
    if (c->has_out()) y.insert(y.end(), c->out_.begin(), c->out_.end());
    if (c->has_in()) y.insert(y.end(), c->in_.begin(), c->in_.end());
  }
  return y;
}

/// The reverse sweep seeded with `w` (laid out like y), over a forward run from zero. Returns
/// the net so that the caller can read the adjoint variables. COLLECTIVE.
fuse::Result<std::unique_ptr<Net>> reverse_run(const Spec& s, int periods, std::span<const double> w) {
  auto n = std::make_unique<Net>(s);
  if (auto r = n->init_forward(); !r.ok()) return r.error();
  if (auto r = n->runner->run(n->components, periods); !r.ok()) return r.error();
  if (auto r = n->init_adjoint(); !r.ok()) return r.error();
  std::size_t at = 0;
  for (MpiLin* c : n->members()) {
    if (c->has_out()) for (double& v : c->out_bar_) v = w[at++];
    if (c->has_in()) for (double& v : c->in_bar_) v = w[at++];
  }
  for (int i = 0; i < periods; ++i) {
    if (auto r = n->runner->run_period_adjoint(n->components); !r.ok()) return r.error();
  }
  return std::move(n);  // not `return n;`: gcc 13 will not convert an lvalue unique_ptr into the Result
}

/// x_bar = G^T w on this rank's parts, laid out like x. COLLECTIVE.
fuse::Result<std::vector<double>> adjoint_map(const Spec& s, int periods, std::span<const double> w) {
  auto run = reverse_run(s, periods, w);
  if (!run.ok()) return run.error();
  Net& n = *run.value();
  std::vector<double> xbar;
  for (MpiLin* c : n.members()) {
    if (c->has_out()) xbar.insert(xbar.end(), c->out_bar_.begin(), c->out_bar_.end());
  }
  for (int step = 0; step < periods * s.fluid_steps; ++step) {
    const auto it = n.fluid->drive_bar.find(step);
    if (it == n.fluid->drive_bar.end()) {
      if (n.fluid->cells() == 0) continue;  // a rank that holds none of the fluid steps nothing
      return fuse::Failure{"E-TEST", "no adjoint for step " + std::to_string(step)};
    }
    xbar.insert(xbar.end(), it->second.begin(), it->second.end());
  }
  return xbar;
}

std::vector<double> random_vector(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& e : v) e = u(rng);
  return v;
}

/// How the whole vectors are laid out: this rank's part after the parts of the lower ranks.
struct Gathered {
  std::vector<int> nx, ny, ox, oy;
  std::size_t NX = 0, NY = 0;
};
Gathered gather_layout(const Dims& d) {
  const int w = world_size();
  Gathered g;
  g.nx.assign(static_cast<std::size_t>(w), 0);
  g.ny.assign(static_cast<std::size_t>(w), 0);
  const int mx = static_cast<int>(d.nx), my = static_cast<int>(d.ny);
  MPI_Allgather(&mx, 1, MPI_INT, g.nx.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&my, 1, MPI_INT, g.ny.data(), 1, MPI_INT, MPI_COMM_WORLD);
  g.ox.assign(static_cast<std::size_t>(w), 0);
  g.oy.assign(static_cast<std::size_t>(w), 0);
  for (int r = 0; r < w; ++r) {
    const auto k = static_cast<std::size_t>(r);
    if (r > 0) {
      g.ox[k] = g.ox[k - 1] + g.nx[k - 1];
      g.oy[k] = g.oy[k - 1] + g.ny[k - 1];
    }
    g.NX += static_cast<std::size_t>(g.nx[k]);
    g.NY += static_cast<std::size_t>(g.ny[k]);
  }
  return g;
}

struct GateRun {
  double worst_ratio = 0.0;
  double norm = 0.0;
  bool all_passed = true;
  std::string error;
};

/// The derivative gate's adjoint identity for the distributed coupling of `s`. COLLECTIVE; every
/// rank returns the same result, because it is computed from the whole vectors.
GateRun gate(const Spec& s, int periods, int draws = 3) {
  GateRun out;
  const Dims d = local_dims(s, periods);
  const Gathered g = gather_layout(d);
  const auto me = static_cast<std::size_t>(world_rank());
  const fuse::linalg::HostApplyFn fwd = [&](std::span<const double> X, std::span<double> Y) -> fuse::Result<void> {
    const std::vector<double> x(X.begin() + g.ox[me], X.begin() + g.ox[me] + g.nx[me]);
    auto r = forward_map(s, periods, x);
    if (!r.ok()) return r.error();
    MPI_Allgatherv(r.value().data(), g.ny[me], MPI_DOUBLE, Y.data(), g.ny.data(), g.oy.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return fuse::Result<void>::success();
  };
  const fuse::linalg::HostApplyFn adj = [&](std::span<const double> W, std::span<double> XB) -> fuse::Result<void> {
    const std::vector<double> w(W.begin() + g.oy[me], W.begin() + g.oy[me] + g.ny[me]);
    auto r = adjoint_map(s, periods, w);
    if (!r.ok()) return r.error();
    MPI_Allgatherv(r.value().data(), g.nx[me], MPI_DOUBLE, XB.data(), g.nx.data(), g.ox.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return fuse::Result<void>::success();
  };
  const auto norm = fuse::linalg::estimate_operator_norm(g.NY, g.NX, fwd, adj);
  if (!norm.ok()) {
    out.error = norm.error().code + ": " + norm.error().message;
    out.all_passed = false;
    return out;
  }
  out.norm = norm.value();
  const double eps = std::numeric_limits<double>::epsilon();
  for (int k = 0; k < draws; ++k) {
    const std::vector<double> x = random_vector(g.NX, 100 + static_cast<std::uint64_t>(k));
    const std::vector<double> y = random_vector(g.NY, 200 + static_cast<std::uint64_t>(k));
    const auto r = fuse::linalg::adjoint_gate(g.NY, g.NX, fwd, adj, x, y, norm.value());
    if (!r.ok()) {
      out.error = r.error().code + ": " + r.error().message;
      out.all_passed = false;
      return out;
    }
    out.all_passed = out.all_passed && r.value().passed;
    out.worst_ratio = std::max(out.worst_ratio, r.value().defect / (eps * r.value().norm_est *
                                                                  r.value().x_norm * r.value().y_norm));
  }
  return out;
}

struct NamedCase {
  std::string name;
  Spec spec;
  int periods;
};

std::vector<NamedCase> identity_cases() {
  std::vector<NamedCase> c;
  for (const Shape shape : {Shape::Disjoint, Shape::Partial, Shape::Nested}) {
    const std::string tag = shape_name(shape);
    {
      Spec s;  // one period, an instantaneous channel across ranks
      s.shape = shape;
      s.a_f = 0.9;
      s.c_f = 0.0;
      s.a_s = 1.0;
      s.c_s = 0.4;
      c.push_back({tag + "_pipeline", s, 1});
    }
    {
      Spec s;  // state in both components, a loop closed by a lagged channel across ranks
      s.shape = shape;
      s.loop = true;
      s.a_f = 0.9;
      s.c_f = 0.4;
      s.a_s = 0.8;
      s.c_s = 0.3;
      c.push_back({tag + "_loop_4_periods", s, 4});
    }
    {
      Spec s;  // the loop with a mean over four steps of the faster producer
      s.shape = shape;
      s.loop = true;
      s.accum = AccumOp::Mean;
      s.fluid_dt = Rational{1, 4};
      s.fluid_steps = 4;
      s.a_f = 0.95;
      s.c_f = 0.2;
      s.a_s = 0.7;
      s.c_s = 0.5;
      c.push_back({tag + "_loop_mean_3_periods", s, 3});
    }
  }
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// The adjoint identity, within the gate's bound
// ---------------------------------------------------------------------------

TEST(RuntimeAdjointMpi, AdjointIdentityHoldsAcrossRanks) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  double worst = 0.0;
  for (const NamedCase& c : identity_cases()) {
    const GateRun r = gate(c.spec, c.periods);
    EXPECT_TRUE(r.error.empty()) << c.name << ": " << r.error;
    EXPECT_TRUE(r.all_passed) << c.name << ": worst ratio " << r.worst_ratio << " against c = "
                              << fuse::linalg::kAdjointGateConstant;
    EXPECT_GT(r.norm, 0.0) << c.name << ": the map is zero, so the identity says nothing";
    worst = std::max(worst, r.worst_ratio);
    if (world_rank() == 0) {
      std::fprintf(stderr, "[adjoint-mpi] %-30s ranks = %d ||G||_est = %-8.4g worst ratio = %.3f\n",
                   c.name.c_str(), world_size(), r.norm, r.worst_ratio);
    }
  }
  if (world_rank() == 0) {
    std::fprintf(stderr, "[adjoint-mpi] max observed adjoint ratio = %.3f (bound constant %.0f)\n", worst,
                 fuse::linalg::kAdjointGateConstant);
  }
  EXPECT_LT(worst, fuse::linalg::kAdjointGateConstant);
}

// The control: a producer whose adjoint leaves out the factor of its state fails the same gate by a
// wide margin. Without it the identity above could pass for any adjoint.
TEST(RuntimeAdjointMpi, AWrongComponentAdjointIsDetectedAcrossRanks) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  Spec s;
  s.loop = true;
  s.a_f = 0.9;
  s.c_f = 0.4;
  s.a_s = 0.8;
  s.c_s = 0.3;
  s.fluid_fault = Fault::SkipRecurrence;
  const GateRun r = gate(s, 3);
  EXPECT_TRUE(r.error.empty()) << r.error;
  EXPECT_FALSE(r.all_passed);
  EXPECT_GT(r.worst_ratio, 100.0 * fuse::linalg::kAdjointGateConstant) << "worst ratio " << r.worst_ratio;
}

// ---------------------------------------------------------------------------
// The adjoint of the exchange by exact value
// ---------------------------------------------------------------------------

// A seed on the consumer's delivered field is one value per entity. The producer is driven at
// its step with value d, so the adjoint of the drive is the adjoint of the producer's output
// there, which only the exchange can have brought from the consumer: entity by entity, the value
// with the same global id, to the last bit (a permutation adds nothing).
TEST(RuntimeAdjointMpi, TheExchangeReturnsEachAdjointToTheEntityItCameFrom) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  for (const Shape shape : {Shape::Disjoint, Shape::Partial, Shape::Nested}) {
    Spec s;
    s.shape = shape;
    s.a_f = 0.5;
    s.c_f = 0.0;
    s.a_s = 0.5;
    s.c_s = 0.3;
    // the seed: the global id of each consumer entity, on the delivered field only
    auto n = std::make_unique<Net>(s);
    ASSERT_TRUE(n->init_forward().ok());
    ASSERT_TRUE(n->runner->run(n->components, 1).ok());
    ASSERT_TRUE(n->init_adjoint().ok());
    const auto solid_view = n->backend.placed_view(n->graph.region(n->region).mesh, fuse::ComponentId{1});
    const auto fluid_view = n->backend.placed_view(n->graph.region(n->region).mesh, fuse::ComponentId{0});
    ASSERT_TRUE(solid_view.ok() && fluid_view.ok());
    for (std::size_t i = 0; i < n->solid->cells(); ++i) {
      n->solid->in_bar_[i] = static_cast<double>(solid_view.value()->global_id(kDim, i));
    }
    ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
    const auto it = n->fluid->drive_bar.find(0);
    if (n->fluid->cells() > 0) {
      ASSERT_NE(it, n->fluid->drive_bar.end());
      for (std::size_t i = 0; i < n->fluid->cells(); ++i) {
        EXPECT_EQ(it->second[i], static_cast<double>(fluid_view.value()->global_id(kDim, i)))
            << shape_name(shape) << ", rank " << world_rank() << ", producer cell " << i;
      }
    }
    // and the delivered field's adjoint has been consumed
    for (std::size_t i = 0; i < n->solid->cells(); ++i) {
      EXPECT_EQ(n->solid->in_bar_[i], 0.0) << "rank " << world_rank() << ", consumer cell " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// A failure on one rank in the middle of a sweep
// ---------------------------------------------------------------------------

// The consumer refuses the adjoint of its step. The ranks that hold the consumer return that error
// at the end of the sweep; the ranks that only exchange with them are told at the agreement point
// and return an error too, and none is left waiting in the exchange (the ctest timeout is the
// witness: a rank that returned early would leave the others blocked in MPI_Alltoallv).
TEST(RuntimeAdjointMpi, AFailureOnOneRankEndsTheSweepOnEveryRankThatExchangesWithIt) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  Spec s;
  s.loop = true;  // the lagged channel's exchange comes after the consumer's step in the reverse order
  s.a_f = 0.5;
  s.c_f = 0.2;
  s.a_s = 0.5;
  s.c_s = 0.3;
  s.solid_fault = Fault::FailAdjoint;
  auto n = std::make_unique<Net>(s);
  ASSERT_TRUE(n->init_forward().ok());
  ASSERT_TRUE(n->runner->run(n->components, 1).ok());
  ASSERT_TRUE(n->init_adjoint().ok());
  const auto r = n->runner->run_period_adjoint(n->components);
  ASSERT_FALSE(r.ok()) << "rank " << world_rank() << " returned from the sweep with no error";
  const bool holds_solid = n->solid_set.contains(world_rank());
  if (holds_solid) {
    EXPECT_EQ(r.error().code, "E-TEST-ADJOINT-REFUSED") << "rank " << world_rank() << ": " << r.error().message;
  } else {
    EXPECT_EQ(r.error().code, "E-RT-EPOCH-PEER") << "rank " << world_rank() << ": " << r.error().message;
  }
  // the failed period did not happen: asking again reverses it again (and fails again, the same way)
  const auto again = n->runner->run_period_adjoint(n->components);
  EXPECT_FALSE(again.ok());
}

// An iteration region whose components are on different ranks agrees its convergence over the ranks,
// and its adjoint iteration would have to as well: refused, on every rank, by name.
TEST(RuntimeAdjointMpi, AnIterationRegionAcrossRanksIsRefused) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  Spec s;
  s.region = true;
  s.a_f = 0.5;
  s.a_s = 0.5;
  auto n = std::make_unique<Net>(s);
  ASSERT_TRUE(n->init_forward().ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  const auto r = n->init_adjoint();
  ASSERT_FALSE(r.ok()) << "rank " << world_rank();
  EXPECT_EQ(r.error().code, "E-ADJ-REGION") << r.error().message;
  EXPECT_NE(r.error().message.find("spans several ranks"), std::string::npos) << r.error().message;
}

// ---------------------------------------------------------------------------
// The transport alone
// ---------------------------------------------------------------------------

namespace {

constexpr std::size_t kGlobal = 12;
constexpr int kComponents = 2;

/// An entity sent to every destination rank: rank 0 holds the ids 0..11, and every other rank wants
/// all of them, in reverse order, so that a source entry is read by several ranks and the
/// destination order is not the source order.
fuse::TransportSpec fan_out_spec(int rank) {
  const int w = world_size();
  fuse::TransportSpec s;
  s.src_rank_set = 0;
  s.dst_rank_set = 1;
  s.src_ranks = {0};
  for (int r = 1; r < w; ++r) s.dst_ranks.push_back(r);
  s.components = kComponents;
  if (rank == 0) {
    for (std::size_t g = 0; g < kGlobal; ++g) s.global_ids.push_back(g);
  } else {
    for (std::size_t g = kGlobal; g-- > 0;) s.global_ids.push_back(g);
  }
  s.elements = s.global_ids.size();
  return s;
}

/// The value carried by component c of global id g: an integer, so that every sum below is exact.
double value_of(std::size_t g, int c) { return static_cast<double>(g) * 10.0 + 1.0 + static_cast<double>(c); }

}  // namespace

// A source entry that `move()` sends to several ranks receives the SUM of their adjoints. The
// weight of rank r is r, so the sum over the W-1 destination ranks is (W-1) W / 2 times the value;
// and `<move(x), y> = <x, move_adjoint(y)>` over the whole communicator, exactly (integers).
TEST(RuntimeAdjointMpi, TheAdjointOfAnExchangeSumsTheAdjointOfEveryCopy) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  const int rank = world_rank(), w = world_size();
  const fuse::TransportSpec spec = fan_out_spec(rank);
  const std::size_t total = spec.elements * static_cast<std::size_t>(kComponents);

  fuse::MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto plan = t.build_plan(spec);
  ASSERT_TRUE(plan.ok()) << plan.error().code << ": " << plan.error().message;
  ASSERT_TRUE(t.has_adjoint());

  std::vector<double> x(total, 0.0), y(total, -1.0);  // x on the source, y the copy it is moved to
  std::vector<double> ybar(total, 0.0), xbar(total, -7.0);
  for (std::size_t i = 0; i < spec.elements; ++i) {
    for (int c = 0; c < kComponents; ++c) {
      const std::size_t k = i * kComponents + static_cast<std::size_t>(c);
      if (rank == 0) x[k] = value_of(spec.global_ids[i], c);
      if (rank != 0) ybar[k] = static_cast<double>(rank) * value_of(spec.global_ids[i], c);
    }
  }
  ASSERT_TRUE(t.prepare(spec).ok());
  ASSERT_TRUE(t.move(x.data(), y.data(), total).ok());
  ASSERT_TRUE(t.move_adjoint(ybar.data(), xbar.data(), total).ok());

  const double weight_sum = static_cast<double>(w - 1) * static_cast<double>(w) / 2.0;
  for (std::size_t i = 0; i < spec.elements; ++i) {
    for (int c = 0; c < kComponents; ++c) {
      const std::size_t k = i * kComponents + static_cast<std::size_t>(c);
      const double want = rank == 0 ? weight_sum * value_of(spec.global_ids[i], c) : 0.0;
      EXPECT_EQ(xbar[k], want) << "rank " << rank << " entity " << spec.global_ids[i] << " component " << c;
    }
  }
  double local[2] = {0.0, 0.0}, global[2] = {0.0, 0.0};
  for (std::size_t k = 0; k < total; ++k) {
    if (rank != 0) local[0] += y[k] * ybar[k];  // <move(x), ybar>
    local[1] += x[k] * xbar[k];                // <x, move_adjoint(ybar)>
  }
  ASSERT_EQ(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), MPI_SUCCESS);
  EXPECT_EQ(global[0], global[1]);
  EXPECT_GT(global[0], 0.0);
}

TEST(RuntimeAdjointMpi, MoveAdjointRefusesWhatMoveRefuses) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  double a[4] = {0, 0, 0, 0}, b[4] = {0, 0, 0, 0};
  {
    fuse::MpiTransport unplanned;
    const auto r = unplanned.move_adjoint(a, b, 4);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-TRANSPORT-UNPLANNED");
  }
  const int rank = world_rank();
  const fuse::TransportSpec spec = fan_out_spec(rank);
  const std::size_t total = spec.elements * static_cast<std::size_t>(kComponents);
  fuse::MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  ASSERT_TRUE(t.build_plan(spec).ok());
  std::vector<double> buf(total, 0.0);
  const auto wrong = t.move_adjoint(buf.data(), buf.data(), total + 1);
  ASSERT_FALSE(wrong.ok());
  EXPECT_EQ(wrong.error().code, "E-TRANSPORT-SIZE");
  const auto null = t.move_adjoint(nullptr, buf.data(), total);
  ASSERT_FALSE(null.ok());
  EXPECT_EQ(null.error().code, "E-TRANSPORT-NULL");
}

namespace {
/// A backend that implements nothing optional, to see what the base class says.
class MinimalTransport final : public fuse::Transport {
 public:
  fuse::Result<void> prepare(const fuse::TransportSpec&) override { return fuse::Result<void>::success(); }
  fuse::Result<void> move(const double*, double*, std::size_t) override { return fuse::Result<void>::success(); }
  [[nodiscard]] std::string name() const override { return "minimal"; }
  [[nodiscard]] bool is_local() const override { return true; }
};
}  // namespace

TEST(RuntimeAdjointMpi, TheCopyTransportReversesToTheCopyBackwardsAndABackendWithoutAnAdjointSaysSo) {
  fuse::CopyTransport copy;
  fuse::TransportSpec spec;
  spec.elements = 4;
  spec.components = 2;
  ASSERT_TRUE(copy.prepare(spec).ok());
  EXPECT_TRUE(copy.has_adjoint());
  const std::vector<double> dst_bar{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<double> src_bar(8, -1.0);
  ASSERT_TRUE(copy.move_adjoint(dst_bar.data(), src_bar.data(), 8).ok());
  EXPECT_EQ(src_bar, dst_bar);

  MinimalTransport minimal;
  EXPECT_FALSE(minimal.has_adjoint());
  const auto r = minimal.move_adjoint(dst_bar.data(), src_bar.data(), 8);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-NOADJOINT");
}
