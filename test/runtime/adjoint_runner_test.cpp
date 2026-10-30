// Reverse mode through the Runner (plans/2026-10-22-universal-number-types.md,
// stage U6; docs/dev/adjoint-schedule.md).
//
// The composed coupling is a linear map G from the inputs of a run to its final
// state: the initial state of each component and the values each producer is
// driven with at each step, to the state each component holds after N periods.
// The components here are linear (out <- a out + c in + drive), so G is the same
// for every run and the adjoint identity <G x, y> = <x, G^T y> can be checked
// with the derivative gate's own bound (fuse/linalg/derivative_gate.hpp):
//
//   |<G x, y> - <x, G^T y>| <= c * eps * ||G||_est * ||x|| * ||y||.
//
// G x is the Runner's forward run. G^T y is the Runner's reverse sweep: the
// adjoint variables of the final state are the seed, `run_period_adjoint()` walks
// each period backwards, and the answer is read from the components' adjoint
// arrays and from the adjoint of each drive. Nothing in the test knows the
// remap weights; the identity holds for whatever matrices the Runner applied.
//
// Also here: the adjoint of each coupling rule by exact value (a mean spreads 1/n
// to each sample, an integral dt, a sum copies, a first feeds only the first,
// a unit conversion multiplies, a remap is the transpose of the matrix the
// forward run applied), the order of the reverse sweep against the forward one,
// a control showing that the gate detects a wrong adjoint, and every refusal.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/linalg/derivative_gate.hpp>
#include <fuse/num/arithmetic_kernels.hpp>
#include <fuse/runtime/runner.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

using fuse::AccumOp;
using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::MergeOp;
using fuse::Rational;
using fuse::Runner;
using fuse::SerialMeshBackend;
using fuse::StepContext;
namespace u = fuse::units;

namespace {

std::uint64_t bits(double x) {
  std::uint64_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

/// What a deliberately broken component does, to reach each refusal.
enum class Fault {
  None,
  SkipRecurrence,   // the adjoint forgets the factor `a` of the state: a wrong adjoint
  NoAdjoint,        // Component::adjoint() is null
  UnboundAdjoint,   // register_adjoint_fields() binds nothing
  ShortAdjoint,     // an adjoint array of the wrong size
  AliasedAdjoint,   // the adjoint array of a Provides field is the field's own array
  DoubleDouble,     // the fields are double-double values (two limbs per cell)
};

struct Call {
  std::string who;
  bool reverse = false;
  Rational t_begin{0}, t_end{0};
  std::int64_t step = 0;
};
using CallLog = std::vector<Call>;

/// out <- a * out + c * in + drive(step), cell by cell and component by
/// component. The field named "out" is the one it provides, "in" the one it
/// requires; either may be absent. Its adjoint follows the conventions of
/// `fuse::AdjointComponent`.
class Lin final : public fuse::Component, public fuse::AdjointComponent {
 public:
  Lin(std::string name, std::size_t cells, int comps, double a, double c, CallLog* log, Fault fault)
      : name_(std::move(name)), cells_(cells), comps_(comps), a_(a), c_(c), log_(log), fault_(fault) {
    const std::size_t n = size();
    out_.assign(n, 0.0);
    in_.assign(n, 0.0);
    out_bar_.assign(n, 0.0);
    in_bar_.assign(n, 0.0);
  }
  [[nodiscard]] std::string name() const override { return name_; }
  [[nodiscard]] fuse::AdjointComponent* adjoint() override {
    return fault_ == Fault::NoAdjoint ? nullptr : this;
  }
  [[nodiscard]] std::size_t size() const { return cells_ * static_cast<std::size_t>(comps_); }
  // An iteration region needs the component to roll back; the region is refused later.
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
      auto r = bind(reg, f, provides ? out_ : in_, cells_);
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> register_adjoint_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    if (fault_ == Fault::UnboundAdjoint) return fuse::Result<void>::success();
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      std::vector<double>& target = provides ? out_bar_ : in_bar_;
      auto r = fault_ == Fault::ShortAdjoint
                   ? bind(reg, f, target, cells_ - 1)
                   : fault_ == Fault::AliasedAdjoint && provides ? bind(reg, f, out_, cells_)
                                                                 : bind(reg, f, target, cells_);
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const StepContext& ctx) override {
    if (log_ != nullptr) log_->push_back({name_, false, ctx.t_begin, ctx.t_end, ctx.step});
    const auto d = drive.find(ctx.step);
    for (std::size_t k = 0; k < out_.size(); ++k) {
      out_[k] = a_ * out_[k] + c_ * in_[k] + (d != drive.end() ? d->second[k] : 0.0);
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance_adjoint(const StepContext& ctx) override {
    if (log_ != nullptr) log_->push_back({name_, true, ctx.t_begin, ctx.t_end, ctx.step});
    std::vector<double>& db = drive_bar[ctx.step];
    db.assign(out_.size(), 0.0);
    for (std::size_t k = 0; k < out_.size(); ++k) {
      db[k] = out_bar_[k];              // d out / d drive = 1
      in_bar_[k] += c_ * out_bar_[k];   // a Requires field: add
      if (fault_ != Fault::SkipRecurrence) out_bar_[k] *= a_;  // a Provides field: overwrite
    }
    return fuse::Result<void>::success();
  }

  bool has_out() const { return has_out_; }
  bool has_in() const { return has_in_; }

  std::vector<double> out_, in_, out_bar_, in_bar_;
  std::map<std::int64_t, std::vector<double>> drive;      // forward input at each step
  std::map<std::int64_t, std::vector<double>> drive_bar;  // its adjoint

 private:
  fuse::Result<void> bind(FieldRegistry& reg, fuse::FieldId f, std::vector<double>& v,
                          std::size_t count) const {
    if (fault_ == Fault::DoubleDouble) {
      return reg.bind(f, fuse::BasicFieldRef<fuse::num::dd>(reinterpret_cast<fuse::num::dd*>(v.data()),
                                                            fuse::Layout{1, 1, count, 1},
                                                            fuse::MemorySpace::Host));
    }
    return reg.bind(f, fuse::FieldRef(v.data(),
                                      fuse::Layout{static_cast<std::size_t>(comps_), 1, count, comps_},
                                      fuse::MemorySpace::Host));
  }

  std::string name_;
  std::size_t cells_;
  int comps_;
  double a_, c_;
  CallLog* log_;
  Fault fault_;
  bool has_out_ = false, has_in_ = false;
  std::vector<double> saved_out_, saved_in_;
};

enum class Topology {
  Pipeline,  // p -> q
  Loop,      // p -> q and q -> p, the second lagged by one step
};

struct Spec {
  Topology topology = Topology::Pipeline;
  std::size_t p_cells = 8, q_cells = 5;
  int comps = 1;
  AccumOp accum = AccumOp::Instant;
  Rational p_dt{1}, q_dt{1};
  fuse::Units p_units = u::kKelvin, q_units = u::kKelvin;
  int p_steps_per_period = 1;  // the number of steps of p in one period, by construction
  bool second = false;         // a second producer into q's field, merged with the first
  MergeOp merge = MergeOp::Sum;
  bool region = false;
  bool spacetime = false;    // a linear space-time projection on the channel p -> q
  double relax = 0.0;        // an under-relaxation factor on the channel p -> q, 0 for none
  bool same_region = false;  // q's fields are on p's region (no remap); q_cells is then p_cells
  int extrapolate = -1;      // the order of an Extrapolate on the channel p -> q, -1 for none
  fuse::Conservation conservation = fuse::Conservation::None;
  double a_p = 0.0, c_p = 0.0, a_q = 0.0, c_q = 0.0;
  Fault p_fault = Fault::None, q_fault = Fault::None;
};

struct Net {
  CouplingGraph graph;
  SerialMeshBackend backend;
  fuse::OperatorDAG dag;
  fuse::ExecutionPlan plan;
  ComponentSet components;
  FieldRegistry registry, adjoint_registry;
  CallLog log;
  Lin* p = nullptr;
  Lin* p2 = nullptr;
  Lin* q = nullptr;
  std::unique_ptr<Runner> runner;
};

std::unique_ptr<Net> make_net(const Spec& s) {
  auto n = std::make_unique<Net>();
  auto& g = n->graph;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:" + std::to_string(s.p_cells)});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:" + std::to_string(s.q_cells)});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{.name = "p", .ranks = fuse::RankSet{0, 1}, .interval = s.p_dt});
  const auto q = g.add_component(fuse::ComponentDecl{.name = "q", .ranks = fuse::RankSet{1, 1}, .interval = s.q_dt});
  const auto out = g.add_field(p, fuse::FieldSpec{.name = "out", .region = r1, .components = s.comps,
                                                  .units = s.p_units, .conservation = s.conservation,
                                                  .intent = fuse::Intent::Provides});
  const auto rq = s.same_region ? r1 : r2;
  const auto in = g.add_field(q, fuse::FieldSpec{.name = "in", .region = rq, .components = s.comps,
                                                 .units = s.q_units, .conservation = s.conservation,
                                                 .intent = fuse::Intent::Requires});
  const auto link = g.add_channel("link", out, in);
  if (s.accum != AccumOp::Instant) g.channel(link).accumulate(s.accum);
  if (s.extrapolate >= 0) g.channel(link).extrapolate(s.extrapolate);
  if (s.spacetime) g.channel(link).spacetime(fuse::TemporalBasis::Linear, 1);
  if (s.relax != 0.0) g.channel(link).relax(s.relax);
  if (s.second) {
    const auto p2 = g.add_component(fuse::ComponentDecl{.name = "p2", .ranks = fuse::RankSet{0, 1}, .interval = s.p_dt});
    const auto out2 = g.add_field(p2, fuse::FieldSpec{.name = "out", .region = r1, .components = s.comps,
                                                      .units = s.p_units, .intent = fuse::Intent::Provides});
    g.add_channel("link2", out2, in);
    g.merge(in, s.merge);
  }
  if (s.topology == Topology::Loop) {
    const auto qout = g.add_field(q, fuse::FieldSpec{.name = "out", .region = rq, .components = s.comps,
                                                     .units = s.q_units, .intent = fuse::Intent::Provides});
    const auto pin = g.add_field(p, fuse::FieldSpec{.name = "in", .region = r1, .components = s.comps,
                                                    .units = s.p_units, .intent = fuse::Intent::Requires});
    const auto back = g.add_channel("back", qout, pin);
    g.channel(back).lag(1).allow_time_level_mismatch("the loop is closed by one step of lag");
  }
  if (s.region) {
    fuse::ConvergenceCriterion crit;
    crit.add(fuse::Converged::Relative(1e-6, "link"));
    g.add_iteration_region(fuse::IterationRegionDecl{
        .name = "outer", .channels = {link}, .max_iterations = 12, .convergence = crit});
  }
  EXPECT_TRUE(n->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(n->backend.load(g.mesh(m2)).ok());
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  n->dag = lowered.dag;
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, n->dag, clocks, fuse::make_region_sizes(g, n->dag, n->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  n->plan = sched.plan;

  auto lp = std::make_unique<Lin>("p", s.p_cells, s.comps, s.a_p, s.c_p, &n->log, s.p_fault);
  auto lq = std::make_unique<Lin>("q", s.same_region ? s.p_cells : s.q_cells, s.comps, s.a_q, s.c_q, &n->log, s.q_fault);
  n->p = lp.get();
  n->q = lq.get();
  n->components.add(std::move(lp));
  if (s.second) {
    auto lp2 = std::make_unique<Lin>("p2", s.p_cells, s.comps, 0.0, 0.0, &n->log, s.p_fault);
    n->p2 = lp2.get();
    n->components.add(std::move(lp2));
  }
  n->components.add(std::move(lq));
  n->runner = std::make_unique<Runner>(n->graph, n->dag, n->plan, n->backend);
  return n;
}

fuse::Result<void> init_forward(Net& n, const fuse::RunnerConfig& cfg = {}) {
  for (auto& c : n.components) {
    auto r = c.register_fields(n.registry, n.graph);
    if (!r.ok()) return r;
  }
  return n.runner->initialize(n.components, n.registry, cfg);
}

fuse::Result<void> init_adjoint(Net& n) {
  for (auto& c : n.components) {
    if (c.adjoint() == nullptr) continue;  // the Runner refuses it by name
    auto r = c.adjoint()->register_adjoint_fields(n.adjoint_registry, n.graph);
    if (!r.ok()) return r;
  }
  return n.runner->initialize_adjoint(n.components, n.adjoint_registry);
}

/// The components that make up the state, in a fixed order.
std::vector<Lin*> members(Net& n) {
  std::vector<Lin*> m{n.p};
  if (n.p2 != nullptr) m.push_back(n.p2);
  m.push_back(n.q);
  return m;
}
std::vector<Lin*> producers(Net& n) {
  std::vector<Lin*> m{n.p};
  if (n.p2 != nullptr) m.push_back(n.p2);
  return m;
}

/// Sizes of the input vector x (initial state of every component that provides a
/// field, then the drive of every producer at every step) and of the output
/// vector y (the state after the run: the provided field and the delivered field
/// of every component that has them).
struct Dims {
  std::size_t nx = 0, ny = 0;
};
Dims dims_of(const Spec& s, int periods) {
  Spec probe = s;
  auto n = make_net(probe);
  EXPECT_TRUE(init_forward(*n).ok());
  Dims d;
  for (Lin* c : members(*n)) {
    if (c->has_out()) d.nx += c->size();
    d.ny += (c->has_out() ? c->size() : 0) + (c->has_in() ? c->size() : 0);
  }
  for (Lin* c : producers(*n)) d.nx += c->size() * static_cast<std::size_t>(periods * s.p_steps_per_period);
  return d;
}

/// y = G x: the forward run.
fuse::Result<std::vector<double>> forward_map(const Spec& s, int periods, std::span<const double> x) {
  auto n = make_net(s);
  if (auto r = init_forward(*n); !r.ok()) return r.error();
  std::size_t at = 0;
  for (Lin* c : members(*n)) {
    if (!c->has_out()) continue;
    for (double& v : c->out_) v = x[at++];
  }
  for (Lin* c : producers(*n)) {
    for (int step = 0; step < periods * s.p_steps_per_period; ++step) {
      std::vector<double>& d = c->drive[step];
      d.assign(c->size(), 0.0);
      for (double& v : d) v = x[at++];
    }
  }
  if (auto r = n->runner->run(n->components, periods); !r.ok()) return r.error();
  std::vector<double> y;
  for (Lin* c : members(*n)) {
    if (c->has_out()) y.insert(y.end(), c->out_.begin(), c->out_.end());
    if (c->has_in()) y.insert(y.end(), c->in_.begin(), c->in_.end());
  }
  return y;
}

/// The reverse sweep seeded with `w` (laid out like y), over a forward run from
/// zero. Returns the net so the caller can read the adjoint variables.
fuse::Result<std::unique_ptr<Net>> reverse_run(const Spec& s, int periods, std::span<const double> w) {
  auto n = make_net(s);
  if (auto r = init_forward(*n); !r.ok()) return r.error();
  if (auto r = n->runner->run(n->components, periods); !r.ok()) return r.error();
  if (auto r = init_adjoint(*n); !r.ok()) return r.error();
  std::size_t at = 0;
  for (Lin* c : members(*n)) {
    if (c->has_out()) for (double& v : c->out_bar_) v = w[at++];
    if (c->has_in()) for (double& v : c->in_bar_) v = w[at++];
  }
  for (int i = 0; i < periods; ++i) {
    if (auto r = n->runner->run_period_adjoint(n->components); !r.ok()) return r.error();
  }
  return std::move(n);  // not `return n;`: gcc 13 will not convert an lvalue unique_ptr into the Result
}

/// x_bar = G^T w, laid out like x.
fuse::Result<std::vector<double>> adjoint_map(const Spec& s, int periods, std::span<const double> w) {
  auto run = reverse_run(s, periods, w);
  if (!run.ok()) return run.error();
  Net& n = *run.value();
  std::vector<double> xbar;
  for (Lin* c : members(n)) {
    if (c->has_out()) xbar.insert(xbar.end(), c->out_bar_.begin(), c->out_bar_.end());
  }
  for (Lin* c : producers(n)) {
    for (int step = 0; step < periods * s.p_steps_per_period; ++step) {
      const auto it = c->drive_bar.find(step);
      if (it == c->drive_bar.end()) return fuse::Failure{"E-TEST", "no adjoint for step " + std::to_string(step)};
      xbar.insert(xbar.end(), it->second.begin(), it->second.end());
    }
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

struct GateRun {
  double worst_ratio = 0.0;  // defect / (eps * norm * |x| * |y|), the largest over the draws
  double norm = 0.0;
  bool all_passed = true;
  std::string error;
};

/// The derivative gate's adjoint identity for the composed coupling of `s`.
GateRun gate(const Spec& s, int periods, int draws = 3) {
  GateRun out;
  const Dims d = dims_of(s, periods);
  const fuse::linalg::HostApplyFn fwd = [&](std::span<const double> x, std::span<double> y) -> fuse::Result<void> {
    auto r = forward_map(s, periods, x);
    if (!r.ok()) return r.error();
    std::copy(r.value().begin(), r.value().end(), y.begin());
    return fuse::Result<void>::success();
  };
  const fuse::linalg::HostApplyFn adj = [&](std::span<const double> w, std::span<double> xbar) -> fuse::Result<void> {
    auto r = adjoint_map(s, periods, w);
    if (!r.ok()) return r.error();
    std::copy(r.value().begin(), r.value().end(), xbar.begin());
    return fuse::Result<void>::success();
  };
  const auto norm = fuse::linalg::estimate_operator_norm(d.ny, d.nx, fwd, adj);
  if (!norm.ok()) {
    out.error = norm.error().code + ": " + norm.error().message;
    out.all_passed = false;
    return out;
  }
  out.norm = norm.value();
  const double eps = std::numeric_limits<double>::epsilon();
  for (int k = 0; k < draws; ++k) {
    const std::vector<double> x = random_vector(d.nx, 100 + static_cast<std::uint64_t>(k));
    const std::vector<double> y = random_vector(d.ny, 200 + static_cast<std::uint64_t>(k));
    const auto r = fuse::linalg::adjoint_gate(d.ny, d.nx, fwd, adj, x, y, norm.value());
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
  const char* name;
  Spec spec;
  int periods;
};

std::vector<NamedCase> identity_cases() {
  std::vector<NamedCase> c;
  {
    Spec s;  // one period, an instantaneous channel through a real 8 -> 5 remap
    s.a_q = 1.0;  // q keeps what it was delivered
    c.push_back({"instant_remap_8_to_5", s, 1});
  }
  {
    Spec s;  // the remap the other way, 5 -> 8
    s.p_cells = 5;
    s.q_cells = 8;
    c.push_back({"instant_remap_5_to_8", s, 1});
  }
  {
    Spec s;  // components with three values per cell
    s.comps = 3;
    c.push_back({"three_components_per_cell", s, 1});
  }
  {
    Spec s;  // a mean over four steps of the producer, two periods
    s.accum = AccumOp::Mean;
    s.p_dt = Rational{1, 4};
    s.p_steps_per_period = 4;
    c.push_back({"mean_of_four_steps", s, 2});
  }
  {
    Spec s;  // a sum over four steps
    s.accum = AccumOp::Sum;
    s.p_dt = Rational{1, 4};
    s.p_steps_per_period = 4;
    c.push_back({"sum_of_four_steps", s, 2});
  }
  {
    Spec s;  // an integral, for a consumer slower than the period of the producer's samples
    s.accum = AccumOp::Integral;
    s.p_dt = Rational{1, 4};
    s.p_steps_per_period = 4;
    s.q_units = u::kKelvin * u::kSecond;
    c.push_back({"integral_of_four_steps", s, 2});
  }
  {
    Spec s;  // a faster producer read instantaneously: the value it holds when the chain runs
    s.p_dt = Rational{1, 4};
    s.p_steps_per_period = 4;
    c.push_back({"instant_of_four_steps", s, 2});
  }
  {
    Spec s;  // a conservative remap, with the conservation check the Runner makes after it
    s.conservation = fuse::Conservation::Extensive;
    c.push_back({"conservative_remap", s, 2});
  }
  {
    Spec s;  // a loop on one region: the lagged channel has no remap, so the lag is an Interpolate
    s.topology = Topology::Loop;
    s.same_region = true;
    s.a_p = 0.9;
    s.c_p = 0.4;
    s.a_q = 0.8;
    s.c_q = 0.3;
    c.push_back({"loop_one_region_lag_only", s, 4});
  }
  {
    Spec s;  // an extrapolation of order 0 on the channel
    s.extrapolate = 0;
    c.push_back({"extrapolate_order_0", s, 1});
  }
  {
    Spec s;  // a unit conversion (a multiplication by 1000) ahead of the remap
    s.p_units = u::kPascal;
    s.p_units.scale = Rational{1000};
    s.q_units = u::kPascal;
    c.push_back({"unit_conversion", s, 1});
  }
  {
    Spec s;  // two producers merged by a sum
    s.second = true;
    s.merge = MergeOp::Sum;
    c.push_back({"merge_sum", s, 1});
  }
  {
    Spec s;
    s.second = true;
    s.merge = MergeOp::Mean;
    c.push_back({"merge_mean", s, 1});
  }
  {
    Spec s;
    s.second = true;
    s.merge = MergeOp::First;
    c.push_back({"merge_first", s, 1});
  }
  {
    Spec s;  // a loop closed by a lagged channel, with dynamics in both components, five periods
    s.topology = Topology::Loop;
    s.a_p = 0.9;
    s.c_p = 0.4;
    s.a_q = 0.8;
    s.c_q = 0.3;
    c.push_back({"loop_lagged_5_periods", s, 5});
  }
  {
    Spec s;  // the loop with a mean over four steps of the faster component
    s.topology = Topology::Loop;
    s.accum = AccumOp::Mean;
    s.p_dt = Rational{1, 4};
    s.p_steps_per_period = 4;
    s.a_p = 0.95;
    s.c_p = 0.2;
    s.a_q = 0.7;
    s.c_q = 0.5;
    c.push_back({"loop_mean_multirate_3_periods", s, 3});
  }
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// The adjoint identity for the composed coupling, within the gate's bound
// ---------------------------------------------------------------------------

TEST(RuntimeAdjoint, AdjointIdentityHoldsForTheComposedCoupling) {
  double worst = 0.0;
  for (const NamedCase& c : identity_cases()) {
    const GateRun r = gate(c.spec, c.periods);
    EXPECT_TRUE(r.error.empty()) << c.name << ": " << r.error;
    EXPECT_TRUE(r.all_passed) << c.name << ": worst ratio " << r.worst_ratio << " against c = "
                              << fuse::linalg::kAdjointGateConstant;
    EXPECT_GT(r.norm, 0.0) << c.name << ": the map is zero, so the identity says nothing";
    worst = std::max(worst, r.worst_ratio);
    std::fprintf(stderr, "[adjoint] %-34s ||G||_est = %-10.4g worst ratio = %.3f\n", c.name, r.norm,
                 r.worst_ratio);
  }
  std::fprintf(stderr, "[adjoint] max observed adjoint ratio = %.3f (bound constant %.0f)\n", worst,
               fuse::linalg::kAdjointGateConstant);
  EXPECT_LT(worst, fuse::linalg::kAdjointGateConstant);
}

// The cases above exercise these instruction kinds between them; the list is printed so that
// what the identity covers is on record, and the kinds the adjoint rules are written for and
// that no case reaches are named.
TEST(RuntimeAdjoint, TheCasesReachTheInstructionKindsTheDocumentationClaims) {
  std::map<std::string, int> seen;
  for (const NamedCase& c : identity_cases()) {
    auto n = make_net(c.spec);
    for (const fuse::Instruction& ins : n->plan.instructions()) ++seen[fuse::to_cstring(ins.kind)];
  }
  std::string list;
  for (const auto& [k, v] : seen) list += " " + k + "=" + std::to_string(v);
  std::fprintf(stderr, "[adjoint] instruction kinds over the identity cases:%s\n", list.c_str());
  for (const char* must : {"Source", "Sink", "Remap", "Sample", "Accumulate", "Merge", "Scale",
                           "Extrapolate", "Interpolate", "Transport", "Epoch", "Constrain", "Diagnose"}) {
    EXPECT_GT(seen[must], 0) << must << " is in no identity case";
  }
}

// The control: a component whose adjoint leaves out the factor of its state
// produces an adjoint the same gate rejects, by a wide margin. Without it the
// check above could pass for any adjoint.
TEST(RuntimeAdjoint, AWrongComponentAdjointIsDetected) {
  Spec s;
  s.topology = Topology::Loop;
  s.a_p = 0.9;
  s.c_p = 0.4;
  s.a_q = 0.8;
  s.c_q = 0.3;
  s.q_fault = Fault::SkipRecurrence;
  const GateRun r = gate(s, 5);
  EXPECT_TRUE(r.error.empty()) << r.error;
  EXPECT_FALSE(r.all_passed);
  EXPECT_GT(r.worst_ratio, 100.0 * fuse::linalg::kAdjointGateConstant)
      << "a wrong adjoint should be far outside the bound";
}

// ---------------------------------------------------------------------------
// The adjoint of each coupling rule, by exact value
// ---------------------------------------------------------------------------

namespace {

/// The cotangent the producers receive at each step of a one-period run seeded
/// with `w` on the delivered field of `q` (and zero elsewhere), for a channel
/// whose remap is the identity (equal cell counts).
struct Routed {
  std::vector<std::vector<double>> p;   // per step of p
  std::vector<std::vector<double>> p2;  // per step of p2 (second producer)
  std::vector<double> w;
};

Routed route(Spec s, int periods = 1) {
  s.p_cells = s.q_cells = 8;
  const std::size_t n = static_cast<std::size_t>(s.comps) * 8;
  Dims d = dims_of(s, periods);
  std::vector<double> w(d.ny, 0.0);
  // y holds, for every member in order, its provided field (if any) then its
  // delivered field (if any): the delivered field of q is the last n entries.
  const std::vector<double> seed = random_vector(n, 7);
  std::copy(seed.begin(), seed.end(), w.end() - static_cast<std::ptrdiff_t>(n));
  auto run = reverse_run(s, periods, w);
  EXPECT_TRUE(run.ok()) << (run.ok() ? "" : run.error().message);
  Routed out;
  out.w = seed;
  if (!run.ok()) return out;
  for (int step = 0; step < periods * s.p_steps_per_period; ++step) {
    out.p.push_back(run.value()->p->drive_bar[step]);
    if (run.value()->p2 != nullptr) out.p2.push_back(run.value()->p2->drive_bar[step]);
  }
  return out;
}

void expect_scaled(const std::vector<double>& got, const std::vector<double>& w, double f,
                   const std::string& what) {
  ASSERT_EQ(got.size(), w.size()) << what;
  for (std::size_t k = 0; k < w.size(); ++k) {
    const double expect = f == 0.0 ? 0.0 : f * w[k];  // not -0.0 for a negative w
    EXPECT_EQ(bits(got[k]), bits(expect)) << what << " entry " << k;
  }
}

}  // namespace

TEST(RuntimeAdjoint, AMeanGivesEachSampleAnEqualShare) {
  Spec s;
  s.accum = AccumOp::Mean;
  s.p_dt = Rational{1, 4};
  s.p_steps_per_period = 4;
  const Routed r = route(s);
  ASSERT_EQ(r.p.size(), 4u);
  for (std::size_t j = 0; j < 4; ++j) expect_scaled(r.p[j], r.w, 0.25, "step " + std::to_string(j));
}

TEST(RuntimeAdjoint, ASumGivesEachSampleTheWholeAdjoint) {
  Spec s;
  s.accum = AccumOp::Sum;
  s.p_dt = Rational{1, 4};
  s.p_steps_per_period = 4;
  const Routed r = route(s);
  ASSERT_EQ(r.p.size(), 4u);
  for (std::size_t j = 0; j < 4; ++j) expect_scaled(r.p[j], r.w, 1.0, "step " + std::to_string(j));
}

TEST(RuntimeAdjoint, AnIntegralGivesEachSampleTheIntervalBetweenSamples) {
  // A consumer step of two periods of the producer's sampling: eight samples at
  // dt = 1/4. A mean would give 1/8 to each; the integral gives dt = 1/4.
  Spec s;
  s.accum = AccumOp::Integral;
  s.p_dt = Rational{1, 4};
  s.q_dt = Rational{2};
  s.q_units = u::kKelvin * u::kSecond;
  s.p_steps_per_period = 8;
  const Routed r = route(s);
  ASSERT_EQ(r.p.size(), 8u);
  for (std::size_t j = 0; j < 8; ++j) expect_scaled(r.p[j], r.w, 0.25, "step " + std::to_string(j));
}

TEST(RuntimeAdjoint, TheMeanOfEightSamplesIsAnEighthNotTheIntervalBetweenSamples) {
  Spec s;
  s.accum = AccumOp::Mean;
  s.p_dt = Rational{1, 4};
  s.q_dt = Rational{2};
  s.p_steps_per_period = 8;
  const Routed r = route(s);
  ASSERT_EQ(r.p.size(), 8u);
  for (std::size_t j = 0; j < 8; ++j) expect_scaled(r.p[j], r.w, 0.125, "step " + std::to_string(j));
}

TEST(RuntimeAdjoint, AUnitConversionMultipliesTheAdjointByTheSameFactor) {
  Spec s;
  s.p_units = u::kPascal;
  s.p_units.scale = Rational{1000};
  s.q_units = u::kPascal;
  const Routed r = route(s);
  ASSERT_EQ(r.p.size(), 1u);
  expect_scaled(r.p[0], r.w, 1000.0, "scaled");
}

TEST(RuntimeAdjoint, AMergeSumCopiesTheAdjointToEveryProducer) {
  Spec s;
  s.second = true;
  s.merge = MergeOp::Sum;
  const Routed r = route(s);
  ASSERT_EQ(r.p.size(), 1u);
  ASSERT_EQ(r.p2.size(), 1u);
  expect_scaled(r.p[0], r.w, 1.0, "first");
  expect_scaled(r.p2[0], r.w, 1.0, "second");
}

TEST(RuntimeAdjoint, AMergeMeanGivesEachProducerAHalf) {
  Spec s;
  s.second = true;
  s.merge = MergeOp::Mean;
  const Routed r = route(s);
  expect_scaled(r.p[0], r.w, 0.5, "first");
  expect_scaled(r.p2[0], r.w, 0.5, "second");
}

TEST(RuntimeAdjoint, AMergeFirstFeedsOnlyTheFirstProducer) {
  Spec s;
  s.second = true;
  s.merge = MergeOp::First;
  const Routed r = route(s);
  expect_scaled(r.p[0], r.w, 1.0, "first");
  expect_scaled(r.p2[0], r.w, 0.0, "second");
}

TEST(RuntimeAdjoint, ARemapIsReversedByTheTransposeOfTheMatrixTheForwardRunApplied) {
  // The matrix, one column at a time, from the forward run (a unit vector in).
  Spec s;  // 8 -> 5 cells
  const Dims d = dims_of(s, 1);
  // x is the initial state of p (a_p = 0 makes it irrelevant), then the drive of p at its one step.
  ASSERT_EQ(d.nx, 2 * s.p_cells);
  const std::size_t drive_at = s.p_cells;
  std::vector<std::vector<double>> column;
  for (std::size_t j = 0; j < s.p_cells; ++j) {
    std::vector<double> x(d.nx, 0.0);
    x[drive_at + j] = 1.0;
    const auto y = forward_map(s, 1, x);
    ASSERT_TRUE(y.ok());
    column.push_back(std::vector<double>(y.value().end() - static_cast<std::ptrdiff_t>(s.q_cells), y.value().end()));
  }
  const std::vector<double> w = random_vector(s.q_cells, 11);
  std::vector<double> seed(d.ny, 0.0);
  std::copy(w.begin(), w.end(), seed.end() - static_cast<std::ptrdiff_t>(s.q_cells));
  const auto xbar = adjoint_map(s, 1, seed);
  ASSERT_TRUE(xbar.ok()) << xbar.error().message;
  ASSERT_EQ(xbar.value().size(), d.nx);
  for (std::size_t j = 0; j < s.p_cells; ++j) {
    double expect = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < s.q_cells; ++i) {
      expect += column[j][i] * w[i];
      scale += std::fabs(column[j][i] * w[i]);
    }
    EXPECT_NEAR(xbar.value()[drive_at + j], expect, 8.0 * std::numeric_limits<double>::epsilon() * scale)
        << "source cell " << j;
  }
  // and the matrix is not a trivial one
  double total = 0.0;
  for (const auto& col : column) for (double v : col) total += std::fabs(v);
  EXPECT_GT(total, 0.0);
}

// ---------------------------------------------------------------------------
// The order of the reverse sweep
// ---------------------------------------------------------------------------

TEST(RuntimeAdjoint, TheReverseSweepVisitsEveryStepInTheReverseOfTheForwardOrder) {
  Spec s;
  s.topology = Topology::Loop;
  s.accum = AccumOp::Mean;
  s.p_dt = Rational{1, 4};
  s.p_steps_per_period = 4;
  s.a_p = 0.9;
  s.c_p = 0.4;
  s.a_q = 0.8;
  s.c_q = 0.3;
  const Dims d = dims_of(s, 3);
  const auto run = reverse_run(s, 3, std::vector<double>(d.ny, 1.0));
  ASSERT_TRUE(run.ok()) << run.error().message;
  const CallLog& log = run.value()->log;
  CallLog fwd, rev;
  for (const Call& c : log) (c.reverse ? rev : fwd).push_back(c);
  ASSERT_EQ(fwd.size(), rev.size());
  ASSERT_EQ(fwd.size(), 3u * (4u + 1u));  // three periods of four steps of p and one of q
  for (std::size_t i = 0; i < fwd.size(); ++i) {
    const Call& f = fwd[fwd.size() - 1 - i];
    const Call& r = rev[i];
    EXPECT_EQ(f.who, r.who) << "reverse call " << i;
    EXPECT_EQ(f.step, r.step) << "reverse call " << i;
    EXPECT_TRUE(f.t_begin == r.t_begin && f.t_end == r.t_end) << "reverse call " << i;
  }
}

// ---------------------------------------------------------------------------
// Refusals, by name
// ---------------------------------------------------------------------------

namespace {

/// Builds the net, runs one forward period, and asks for the adjoint.
std::string refusal_code(const Spec& s) {
  auto n = make_net(s);
  if (auto r = init_forward(*n); !r.ok()) return "forward:" + r.error().code;
  if (auto r = n->runner->run_period(n->components); !r.ok()) return "forward-run:" + r.error().code;
  const auto r = init_adjoint(*n);
  return r.ok() ? std::string{} : r.error().code;
}

}  // namespace

TEST(RuntimeAdjoint, AcceptsAnOrdinaryRun) { EXPECT_EQ(refusal_code(Spec{}), ""); }

TEST(RuntimeAdjoint, RefusesAMinimumOrMaximumAccumulation) {
  for (const AccumOp op : {AccumOp::Min, AccumOp::Max}) {
    Spec s;
    s.accum = op;
    s.p_dt = Rational{1, 4};
    s.p_steps_per_period = 4;
    EXPECT_EQ(refusal_code(s), "E-ADJ-ACCUM");
  }
}

TEST(RuntimeAdjoint, RefusesAnExtrapolationOfOrderAboveZero) {
  Spec s;
  s.extrapolate = 1;
  EXPECT_EQ(refusal_code(s), "E-ADJ-OP");
}

TEST(RuntimeAdjoint, RefusesATimeHistory) {
  Spec s;
  s.spacetime = true;
  s.p_dt = Rational{1, 4};
  s.p_steps_per_period = 4;
  EXPECT_EQ(refusal_code(s), "E-ADJ-SPACETIME");
}

TEST(RuntimeAdjoint, RefusesAnInstructionWithNoAdjoint) {
  Spec s;
  s.relax = 0.5;  // a Relax: nonlinear in the sense that it keeps the previous iterate
  EXPECT_EQ(refusal_code(s), "E-ADJ-OP");
}

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
TEST(RuntimeAdjoint, RefusesDeviceExecution) {
  auto n = make_net(Spec{});
  fuse::RunnerConfig cfg;
  cfg.execution_space = fuse::RunnerConfig::RunnerExecutionSpace::Device;
  ASSERT_TRUE(init_forward(*n, cfg).ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  const auto r = init_adjoint(*n);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-SPACE");
  // by the execution space itself, not by the first arc that happens to be on the device
  EXPECT_NE(r.error().message.find("host execution space"), std::string::npos) << r.error().message;
}
#endif

TEST(RuntimeAdjoint, RefusesAnIterationRegion) {
  Spec s;
  s.region = true;
  EXPECT_EQ(refusal_code(s), "E-ADJ-REGION");
}

TEST(RuntimeAdjoint, RefusesAComponentWithoutAnAdjoint) {
  Spec s;
  s.q_fault = Fault::NoAdjoint;
  EXPECT_EQ(refusal_code(s), "E-ADJ-COMPONENT");
}

TEST(RuntimeAdjoint, RefusesAFieldWithoutAnAdjointArray) {
  Spec s;
  s.q_fault = Fault::UnboundAdjoint;
  EXPECT_EQ(refusal_code(s), "E-ADJ-UNBOUND");
}

TEST(RuntimeAdjoint, RefusesAnAdjointArrayOfTheWrongSize) {
  Spec s;
  s.q_fault = Fault::ShortAdjoint;
  EXPECT_EQ(refusal_code(s), "E-ADJ-LAYOUT");
}

TEST(RuntimeAdjoint, RefusesAnAdjointArrayThatIsTheFieldItself) {
  Spec s;
  s.p_fault = Fault::AliasedAdjoint;
  EXPECT_EQ(refusal_code(s), "E-ADJ-ALIAS");
}

TEST(RuntimeAdjoint, RefusesFieldsOfAnArithmeticValueType) {
  Spec s;
  s.comps = 2;
  s.p_fault = s.q_fault = Fault::DoubleDouble;
  EXPECT_EQ(refusal_code(s), "E-ADJ-FIELD-TYPE");
}

TEST(RuntimeAdjoint, RefusesTheReverseSweepBeforeInitializeAdjoint) {
  auto n = make_net(Spec{});
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  const auto r = n->runner->run_period_adjoint(n->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-INIT");
}

TEST(RuntimeAdjoint, RefusesInitializeAdjointBeforeInitialize) {
  auto n = make_net(Spec{});
  const auto r = n->runner->initialize_adjoint(n->components, n->adjoint_registry);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-INIT");
}

TEST(RuntimeAdjoint, RefusesToReverseAPeriodThatHasNotRun) {
  auto n = make_net(Spec{});
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(init_adjoint(*n).ok());
  const auto r = n->runner->run_period_adjoint(n->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-ORDER");
  EXPECT_NE(r.error().message.find("no forward period"), std::string::npos) << r.error().message;
}

TEST(RuntimeAdjoint, RefusesToReverseMorePeriodsThanTheForwardRunCompleted) {
  auto n = make_net(Spec{});
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(n->runner->run(n->components, 2).ok());
  ASSERT_TRUE(init_adjoint(*n).ok());
  ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
  ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
  const auto r = n->runner->run_period_adjoint(n->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-ORDER");
}

TEST(RuntimeAdjoint, RefusesAForwardPeriodInTheMiddleOfASweep) {
  auto n = make_net(Spec{});
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(n->runner->run(n->components, 2).ok());
  ASSERT_TRUE(init_adjoint(*n).ok());
  ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  const auto r = n->runner->run_period_adjoint(n->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-ORDER");
}

TEST(RuntimeAdjoint, AFreshSweepStartsAfterTheForwardRunContinues) {
  // Reverse all periods, run forward again, and reverse the longer run.
  auto n = make_net(Spec{});
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(init_adjoint(*n).ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
  ASSERT_TRUE(n->runner->run(n->components, 2).ok());
  ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
  ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
  ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
  EXPECT_FALSE(n->runner->run_period_adjoint(n->components).ok());
}

