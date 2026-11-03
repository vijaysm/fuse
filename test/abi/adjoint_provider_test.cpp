// The adjoint of a provider plugin. A plugin that sets
// FUSE_CAP_ADJOINT and fills bind_adjoint and advance_adjoint is a fuse::AdjointComponent behind
// the same ProviderAdapter that hosts it forward, so the Runner's reverse sweep drives it like
// any native component.
//
// The plugin here is an in-memory table of the same linear step the Runner's own adjoint test uses,
//
//   out <- a out + c in,
//
// so the composed coupling is a linear map G from the initial state of each component to the state
// after N periods, and the adjoint identity <G x, y> = <x, G^T y> is checked within the derivative
// gate's own bound (fuse/linalg/derivative_gate.hpp). G x is the forward run; G^T y is the reverse
// sweep, which calls the plugin's advance_adjoint() for every step in the reverse of the order
// advance() ran. A plugin whose adjoint drops the factor `a` must fail the same gate. The rest is the
// ABI itself: what bind_adjoint() is handed, the window advance_adjoint() is asked to reverse, the
// refusal of a plugin or a table that does not offer the entries, and the failures of the entries.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "fuse/abi/provider.h"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/base/units.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/linalg/derivative_gate.hpp"
#include "fuse/runtime/runner.hpp"

#include "provider_adapter.hpp"

namespace {

namespace u = fuse::units;

// ---- the plugin: one instance per component, `who` 0 for p and 1 for q ----------------------

enum class Fault {
  None,
  AdjointDropsTheFactor,      // advance_adjoint() forgets the factor a of the state
  AdvanceAdjointFails,        // advance_adjoint() returns FUSE_ERR_RUNTIME
  BindAdjointFails,           // bind_adjoint() returns FUSE_ERR_INVALID
  AdjointEchoesAnotherWindow, // advance_adjoint() reports a window other than the one it was asked for
  NoCapabilityBit,            // the entries are filled, FUSE_CAP_ADJOINT is not set
  NoEntries,                  // FUSE_CAP_ADJOINT is set, the two entries are null
};

struct Config {
  double a[2] = {0.0, 0.0};
  double c[2] = {0.0, 0.0};
  bool loop = false;  // q returns its out to p: p then also has a field `in`
  int comps = 1;
  Fault fault = Fault::None;
  int element_type = FUSE_ELEMENT_F64;
};
Config g_cfg;
int g_created = 0;

struct Event {
  std::string kind;  // "advance", "advance_adjoint", "bind_adjoint"
  int who = 0;
  std::int64_t step = 0;
  std::int64_t t_begin_num = 0, t_end_num = 0;
};
std::vector<Event> g_log;

struct View {
  std::int32_t handle = 0, element_type = 0, components = 0, memspace = 0, ownership = 0;
  std::uint64_t count = 0, owned_count = 0;
  std::int64_t stride = 0;
  void* data = nullptr;
};
std::vector<View> g_bound, g_bound_adjoint;

struct Inst {
  int who = 0;
  int binds = 0, adjoint_binds = 0;
  double *out = nullptr, *in = nullptr, *out_bar = nullptr, *in_bar = nullptr;
  std::size_t n = 0;
};

// bind() and bind_adjoint() come in the order the graph declares the fields: p out, [in]; q in, out.
bool is_out(const Inst* s, int index) { return s->who == 0 ? index == 0 : index == 1; }

fuse_status t_create(fuse_provider_ctx*, const fuse_null*, fuse_provider_instance* out) {
  Inst* s = new Inst;
  s->who = g_created++;
  out->struct_size = sizeof(fuse_provider_instance);
  out->self = reinterpret_cast<fuse_provider_ctx*>(s);
  return FUSE_OK;
}
fuse_status t_destroy(fuse_provider_ctx* self, const fuse_null*, fuse_null*) {
  delete reinterpret_cast<Inst*>(self);
  return FUSE_OK;
}
fuse_status t_describe(fuse_provider_ctx* self, const fuse_null*, fuse_field_table* out) {
  const Inst* s = reinterpret_cast<const Inst*>(self);
  const bool has_in = s->who == 1 || g_cfg.loop;
  out->struct_size = sizeof(fuse_field_table);
  out->written = has_in ? 2 : 1;
  if (out->buffer != nullptr && out->capacity >= out->written) {
    // the rows follow the graph's order: p out, in; q in, out
    std::size_t row = 0;
    const auto put = [&](const char* name, int intent) {
      fuse_field_spec& r = out->buffer[row++];
      r.struct_size = sizeof(fuse_field_spec);
      r.name = name;
      r.components = g_cfg.comps;
      r.intent = intent;
      r.element_type = g_cfg.element_type;
      r.memspace = FUSE_MEMSPACE_HOST;
      r.gradient_size = 0;
    };
    if (s->who == 0) {
      put("out", 0);
      if (has_in) put("in", 1);
    } else {
      put("in", 1);
      put("out", 0);
    }
  }
  return FUSE_OK;
}
void keep(const fuse_field_view* in, std::vector<View>& to) {
  View v;
  v.handle = in->handle;
  v.element_type = in->element_type;
  v.components = in->components;
  v.memspace = in->memspace;
  v.ownership = in->ownership;
  v.count = in->count;
  v.owned_count = in->owned_count;
  v.stride = in->stride;
  v.data = in->data;
  to.push_back(v);
}
fuse_status t_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null*) {
  Inst* s = reinterpret_cast<Inst*>(self);
  keep(in, g_bound);
  double* p = static_cast<double*>(in->data);
  (is_out(s, s->binds) ? s->out : s->in) = p;
  s->n = static_cast<std::size_t>(in->count) * static_cast<std::size_t>(in->components);
  ++s->binds;
  return FUSE_OK;
}
fuse_status t_bind_adjoint(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null*) {
  Inst* s = reinterpret_cast<Inst*>(self);
  g_log.push_back({"bind_adjoint", s->who, 0, 0, 0});
  if (g_cfg.fault == Fault::BindAdjointFails) return FUSE_ERR_INVALID;
  keep(in, g_bound_adjoint);
  double* p = static_cast<double*>(in->data);
  (is_out(s, s->adjoint_binds) ? s->out_bar : s->in_bar) = p;
  ++s->adjoint_binds;
  return FUSE_OK;
}
void echo(const fuse_step_request* in, fuse_step_report* out, bool other_window) {
  out->struct_size = sizeof(fuse_step_report);
  out->step = in->step;
  out->iteration = in->iteration;
  out->t_begin = in->t_begin;
  out->t_end = in->t_end;
  if (other_window) out->t_end.numerator += in->t_end.denominator;
}
fuse_status t_advance(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  Inst* s = reinterpret_cast<Inst*>(self);
  g_log.push_back({"advance", s->who, in->step, in->t_begin.numerator, in->t_end.numerator});
  const double a = g_cfg.a[s->who], c = g_cfg.c[s->who];
  for (std::size_t i = 0; i < s->n; ++i) s->out[i] = a * s->out[i] + (s->in != nullptr ? c * s->in[i] : 0.0);
  echo(in, out, false);
  return FUSE_OK;
}
fuse_status t_advance_adjoint(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  Inst* s = reinterpret_cast<Inst*>(self);
  g_log.push_back({"advance_adjoint", s->who, in->step, in->t_begin.numerator, in->t_end.numerator});
  if (g_cfg.fault == Fault::AdvanceAdjointFails) return FUSE_ERR_RUNTIME;
  const double a = g_cfg.a[s->who], c = g_cfg.c[s->who];
  for (std::size_t i = 0; i < s->n; ++i) {
    if (s->in_bar != nullptr) s->in_bar[i] += c * s->out_bar[i];
    if (g_cfg.fault != Fault::AdjointDropsTheFactor) s->out_bar[i] *= a;
  }
  echo(in, out, g_cfg.fault == Fault::AdjointEchoesAnotherWindow);
  return FUSE_OK;
}
fuse_status t_capabilities(fuse_provider_ctx*, const fuse_null*, fuse_capability_set* out) {
  out->struct_size = sizeof(fuse_capability_set);
  out->flags = g_cfg.fault == Fault::NoCapabilityBit ? 0 : FUSE_CAP_ADJOINT;
  return FUSE_OK;
}

fuse_provider_abi_v1 make_table(bool with_entries) {
  fuse_provider_abi_v1 t{};
  t.struct_size = sizeof(fuse_provider_abi_v1);
  t.abi_version = FUSE_ABI_VERSION;
  t.create = t_create;
  t.destroy = t_destroy;
  t.describe = t_describe;
  t.bind = t_bind;
  t.advance = t_advance;
  t.capabilities = t_capabilities;
  if (with_entries) {
    t.bind_adjoint = t_bind_adjoint;
    t.advance_adjoint = t_advance_adjoint;
  }
  return t;
}
const fuse_provider_abi_v1 kTable = make_table(true);
const fuse_provider_abi_v1 kTableWithoutEntries = make_table(false);

// ---- the coupling: p -> q through a remap, and q -> p one step late when `loop` -----------------

struct Net {
  fuse::CouplingGraph graph;
  fuse::SerialMeshBackend backend;
  fuse::OperatorDAG dag;
  fuse::ExecutionPlan plan;
  fuse::ComponentSet components;
  fuse::FieldRegistry registry, adjoint_registry;
  std::unique_ptr<fuse::Runner> runner;
  fuse::abi_detail::ProviderAdapter* adapter[2] = {nullptr, nullptr};
};

std::unique_ptr<Net> make_net(const fuse_provider_abi_v1* table = &kTable) {
  g_created = 0;
  g_log.clear();
  g_bound.clear();
  g_bound_adjoint.clear();
  auto n = std::make_unique<Net>();
  auto& g = n->graph;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:8"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:5"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{.name = "p", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto q = g.add_component(fuse::ComponentDecl{.name = "q", .ranks = fuse::RankSet{1, 1}, .interval = fuse::Rational{1}});
  const auto out = g.add_field(p, fuse::FieldSpec{.name = "out", .region = r1, .components = g_cfg.comps,
                                                  .units = u::kKelvin, .intent = fuse::Intent::Provides});
  if (g_cfg.loop) {
    g.add_field(p, fuse::FieldSpec{.name = "in", .region = r1, .components = g_cfg.comps,
                                   .units = u::kKelvin, .intent = fuse::Intent::Requires});
  }
  const auto in = g.add_field(q, fuse::FieldSpec{.name = "in", .region = r2, .components = g_cfg.comps,
                                                 .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto qout = g.add_field(q, fuse::FieldSpec{.name = "out", .region = r2, .components = g_cfg.comps,
                                                   .units = u::kKelvin, .intent = fuse::Intent::Provides});
  g.add_channel("link", out, in);
  if (g_cfg.loop) {
    const auto pin = g.find_field(p, "in").value();
    const auto back = g.add_channel("back", qout, pin);
    g.channel(back).lag(1).allow_time_level_mismatch("the loop is closed by one step of lag");
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
  auto ap = std::make_unique<fuse::abi_detail::ProviderAdapter>("p", table);
  auto aq = std::make_unique<fuse::abi_detail::ProviderAdapter>("q", table);
  n->adapter[0] = ap.get();
  n->adapter[1] = aq.get();
  n->components.add(std::move(ap));
  n->components.add(std::move(aq));
  n->runner = std::make_unique<fuse::Runner>(n->graph, n->dag, n->plan, n->backend);
  return n;
}

fuse::Result<void> init_forward(Net& n) {
  for (auto& c : n.components) {
    auto r = c.register_fields(n.registry, n.graph);
    if (!r.ok()) return r;
  }
  return n.runner->initialize(n.components, n.registry, fuse::RunnerConfig{});
}

fuse::Result<void> init_adjoint(Net& n) {
  for (auto& c : n.components) {
    if (c.adjoint() == nullptr) continue;  // the Runner refuses it by name
    auto r = c.adjoint()->register_adjoint_fields(n.adjoint_registry, n.graph);
    if (!r.ok()) return r;
  }
  return n.runner->initialize_adjoint(n.components, n.adjoint_registry);
}

fuse::FieldId field(Net& n, const char* comp, const char* name) {
  return n.graph.find_field(n.graph.find_component(comp).value(), name).value();
}
std::span<double> values(fuse::FieldRegistry& r, fuse::FieldId f) {
  auto ref = r.get_scalars(f);
  EXPECT_TRUE(ref.ok());
  return {ref.value().data(), ref.value().layout().count * static_cast<std::size_t>(ref.value().layout().components)};
}

// The vector layouts. x: the initial `out` of p, then of q. y: the final out of p, [in of p], in of q, out of q.
struct Dims {
  std::size_t nx = 0, ny = 0;
};
Dims dims() {
  const std::size_t comps = static_cast<std::size_t>(g_cfg.comps);
  Dims d;
  d.nx = (8 + 5) * comps;
  d.ny = d.nx + 5 * comps + (g_cfg.loop ? 8 * comps : 0);
  return d;
}
std::vector<std::pair<const char*, const char*>> y_order() {
  std::vector<std::pair<const char*, const char*>> v{{"p", "out"}};
  if (g_cfg.loop) v.push_back({"p", "in"});
  v.push_back({"q", "in"});
  v.push_back({"q", "out"});
  return v;
}

fuse::Result<std::vector<double>> forward_map(int periods, std::span<const double> x) {
  auto n = make_net();
  if (auto r = init_forward(*n); !r.ok()) return r.error();
  std::size_t at = 0;
  for (const char* who : {"p", "q"}) {
    for (double& v : values(n->registry, field(*n, who, "out"))) v = x[at++];
  }
  if (auto r = n->runner->run(n->components, periods); !r.ok()) return r.error();
  std::vector<double> y;
  for (const auto& [who, name] : y_order()) {
    const auto v = values(n->registry, field(*n, who, name));
    y.insert(y.end(), v.begin(), v.end());
  }
  return y;
}

fuse::Result<std::vector<double>> adjoint_map(int periods, std::span<const double> w) {
  auto n = make_net();
  if (auto r = init_forward(*n); !r.ok()) return r.error();
  if (auto r = n->runner->run(n->components, periods); !r.ok()) return r.error();
  if (auto r = init_adjoint(*n); !r.ok()) return r.error();
  std::size_t at = 0;
  for (const auto& [who, name] : y_order()) {
    for (double& v : values(n->adjoint_registry, field(*n, who, name))) v = w[at++];
  }
  for (int i = 0; i < periods; ++i) {
    if (auto r = n->runner->run_period_adjoint(n->components); !r.ok()) return r.error();
  }
  std::vector<double> xbar;
  for (const char* who : {"p", "q"}) {
    const auto v = values(n->adjoint_registry, field(*n, who, "out"));
    xbar.insert(xbar.end(), v.begin(), v.end());
  }
  return xbar;
}

std::vector<double> random_vector(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> d(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& e : v) e = d(rng);
  return v;
}

struct GateRun {
  double worst_ratio = 0.0;
  bool all_passed = true;
  std::string error;
};

GateRun gate(int periods, int draws = 3) {
  GateRun out;
  const Dims d = dims();
  const fuse::linalg::HostApplyFn fwd = [&](std::span<const double> x, std::span<double> y) -> fuse::Result<void> {
    auto r = forward_map(periods, x);
    if (!r.ok()) return r.error();
    std::copy(r.value().begin(), r.value().end(), y.begin());
    return fuse::Result<void>::success();
  };
  const fuse::linalg::HostApplyFn adj = [&](std::span<const double> w, std::span<double> xbar) -> fuse::Result<void> {
    auto r = adjoint_map(periods, w);
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
  const double eps = std::numeric_limits<double>::epsilon();
  for (int k = 0; k < draws; ++k) {
    const std::vector<double> x = random_vector(d.nx, 100 + static_cast<std::uint64_t>(k));
    const std::vector<double> y = random_vector(d.ny, 200 + static_cast<std::uint64_t>(k));
    const auto r = fuse::linalg::adjoint_gate(d.ny, d.nx, fwd, adj, x, y, norm.value(),
                                               fuse::linalg::kAdjointGateConstant);
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

void configure(bool loop, int comps, Fault fault = Fault::None) {
  g_cfg = Config{};
  g_cfg.loop = loop;
  g_cfg.comps = comps;
  g_cfg.fault = fault;
  g_cfg.a[0] = 0.9;
  g_cfg.c[0] = 0.4;
  g_cfg.a[1] = 0.8;
  g_cfg.c[1] = 0.3;
}

std::string refusal_code() {
  auto n = make_net(g_cfg.fault == Fault::NoEntries ? &kTableWithoutEntries : &kTable);
  if (auto r = init_forward(*n); !r.ok()) return "forward:" + r.error().code;
  if (auto r = n->runner->run_period(n->components); !r.ok()) return "forward-run:" + r.error().code;
  const auto r = init_adjoint(*n);
  return r.ok() ? std::string{} : r.error().code;
}

}  // namespace

// ---------------------------------------------------------------------------

TEST(AbiAdjointProvider, AdjointIdentityHoldsForPluginsThatImplementTheEntries) {
  struct C {
    const char* name;
    bool loop;
    int comps;
    int periods;
  };
  const C cases[] = {{"pipeline", false, 1, 2}, {"pipeline_three_components", false, 3, 2},
                     {"loop_lagged", true, 1, 4}, {"loop_two_components", true, 2, 3}};
  double worst = 0.0;
  for (const C& c : cases) {
    configure(c.loop, c.comps);
    const GateRun r = gate(c.periods);
    EXPECT_TRUE(r.all_passed) << c.name << ": " << r.error;
    worst = std::max(worst, r.worst_ratio);
  }
  std::printf("[abi adjoint provider] max observed adjoint ratio = %.3f (bound constant %.0f)\n", worst,
              fuse::linalg::kAdjointGateConstant);
}

TEST(AbiAdjointProvider, APluginWhoseAdjointDropsTheFactorOfTheStateIsDetected) {
  configure(true, 1, Fault::AdjointDropsTheFactor);
  const GateRun r = gate(3);
  EXPECT_FALSE(r.all_passed) << "the gate accepted an adjoint that is not the transpose";
}

TEST(AbiAdjointProvider, BindAdjointIsHandedZeroedStorageLikeBindsOwnButNotTheSameBytes) {
  configure(true, 2);
  auto n = make_net();
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  ASSERT_TRUE(init_adjoint(*n).ok());
  // p: out, in; q: in, out -- four fields, bound once each and bound for the adjoint once each
  ASSERT_EQ(g_bound.size(), 4u);
  ASSERT_EQ(g_bound_adjoint.size(), 4u);
  for (std::size_t i = 0; i < g_bound.size(); ++i) {
    const View& f = g_bound[i];
    const View& b = g_bound_adjoint[i];
    EXPECT_EQ(b.handle, f.handle) << i;
    EXPECT_EQ(b.element_type, f.element_type) << i;
    EXPECT_EQ(b.components, f.components) << i;
    EXPECT_EQ(b.memspace, f.memspace) << i;
    EXPECT_EQ(b.ownership, f.ownership) << i;
    EXPECT_EQ(b.count, f.count) << i;
    EXPECT_EQ(b.owned_count, f.owned_count) << i;
    EXPECT_EQ(b.stride, f.stride) << i;
    EXPECT_NE(b.data, f.data) << i;
    ASSERT_NE(b.data, nullptr) << i;
    const double* p = static_cast<const double*>(b.data);
    for (std::uint64_t k = 0; k < b.count * static_cast<std::uint64_t>(b.components); ++k) {
      EXPECT_EQ(p[k], 0.0) << "field " << i << " entry " << k;
    }
  }
  // and the Runner's own adjoint registry holds the very bytes the plugin was handed
  EXPECT_EQ(values(n->adjoint_registry, field(*n, "p", "out")).data(), g_bound_adjoint[0].data);
  EXPECT_EQ(values(n->adjoint_registry, field(*n, "q", "out")).data(), g_bound_adjoint[3].data);
}

TEST(AbiAdjointProvider, EachStepIsReversedOnceWithTheWindowItWasAdvancedOverInTheReverseOrder) {
  configure(true, 1);
  auto n = make_net();
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(n->runner->run(n->components, 3).ok());
  ASSERT_TRUE(init_adjoint(*n).ok());
  for (int i = 0; i < 3; ++i) ASSERT_TRUE(n->runner->run_period_adjoint(n->components).ok());
  std::vector<Event> forward, reverse;
  for (const Event& e : g_log) {
    if (e.kind == "advance") forward.push_back(e);
    if (e.kind == "advance_adjoint") reverse.push_back(e);
  }
  ASSERT_EQ(forward.size(), 6u);  // p and q, three periods
  ASSERT_EQ(reverse.size(), forward.size());
  for (std::size_t i = 0; i < forward.size(); ++i) {
    const Event& f = forward[forward.size() - 1 - i];
    const Event& r = reverse[i];
    EXPECT_EQ(r.who, f.who) << i;
    EXPECT_EQ(r.step, f.step) << i;
    EXPECT_EQ(r.t_begin_num, f.t_begin_num) << i;
    EXPECT_EQ(r.t_end_num, f.t_end_num) << i;
  }
}

TEST(AbiAdjointProvider, RefusesAPluginThatDoesNotSetTheCapabilityBit) {
  configure(false, 1, Fault::NoCapabilityBit);
  EXPECT_EQ(refusal_code(), "E-ADJ-COMPONENT");
}

TEST(AbiAdjointProvider, RefusesAPluginThatSetsTheCapabilityBitAndLeavesTheEntriesNull) {
  configure(false, 1, Fault::NoEntries);
  EXPECT_EQ(refusal_code(), "E-ADJ-COMPONENT");
}

TEST(AbiAdjointProvider, ReadsOnlyTheEntriesTheTablesStructSizeCovers) {
  // A plugin built against the header before the two entries existed: the table is filled as a
  // longer one would be (the bytes past its struct_size hold the entries), but struct_size says
  // they are not there.
  configure(false, 1);
  fuse_provider_abi_v1 old = kTable;
  old.struct_size = static_cast<std::uint32_t>(offsetof(fuse_provider_abi_v1, bind_adjoint));
  ASSERT_NE(old.advance_adjoint, nullptr);
  auto n = make_net(&old);
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  const auto r = init_adjoint(*n);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-COMPONENT");
  for (const Event& e : g_log) EXPECT_NE(e.kind, "bind_adjoint");
}

TEST(AbiAdjointProvider, ABindAdjointThatFailsIsRefusedByName) {
  configure(false, 1, Fault::BindAdjointFails);
  EXPECT_EQ(refusal_code(), "E-ABI-PROVIDER-BIND-ADJOINT");
}

TEST(AbiAdjointProvider, AnAdvanceAdjointThatFailsIsReportedByTheSweep) {
  configure(false, 1, Fault::AdvanceAdjointFails);
  auto n = make_net();
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  ASSERT_TRUE(init_adjoint(*n).ok());
  const auto r = n->runner->run_period_adjoint(n->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ABI-PROVIDER-ADVANCE-ADJOINT");
}

TEST(AbiAdjointProvider, AnAdvanceAdjointThatReportsAnotherWindowIsRefused) {
  configure(false, 1, Fault::AdjointEchoesAnotherWindow);
  auto n = make_net();
  ASSERT_TRUE(init_forward(*n).ok());
  ASSERT_TRUE(n->runner->run_period(n->components).ok());
  ASSERT_TRUE(init_adjoint(*n).ok());
  const auto r = n->runner->run_period_adjoint(n->components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ABI-PROVIDER-WINDOW");
}

TEST(AbiAdjointProvider, AFieldOfAStorageTypeHasNoAdjointArray) {
  // binary32 fields are accepted forward and refused for the reverse sweep, by name, by the adapter
  configure(false, 1);
  g_cfg.element_type = FUSE_ELEMENT_F32;
  auto n = make_net();
  ASSERT_TRUE(init_forward(*n).ok());
  fuse::FieldRegistry adjoint;
  const auto r = n->adapter[0]->register_adjoint_fields(adjoint, n->graph);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-FIELD-TYPE");
}

TEST(AbiAdjointProvider, ARankThatDoesNotHostThePluginOffersTheAdjointWithEmptyViews) {
  // the shell of a component this process does not host asks the plugin nothing: it cannot know whether
  // the plugin offers the adjoint, so it offers it, and registers the empty views the forward registration did
  configure(true, 2);
  auto n = make_net();
  fuse::abi_detail::ProviderAdapter shell("p");
  fuse::FieldRegistry forward, adjoint;
  ASSERT_TRUE(shell.register_fields(forward, n->graph).ok());
  ASSERT_NE(shell.adjoint(), nullptr);
  ASSERT_TRUE(shell.adjoint()->register_adjoint_fields(adjoint, n->graph).ok());
  for (const char* name : {"out", "in"}) {
    const auto ref = adjoint.get_scalars(field(*n, "p", name));
    ASSERT_TRUE(ref.ok()) << name;
    EXPECT_EQ(ref.value().layout().count, 0u) << name;
    EXPECT_EQ(ref.value().layout().components, 2) << name;
  }
  EXPECT_TRUE(g_bound_adjoint.empty()) << "the plugin was asked nothing";
}

TEST(AbiAdjointProvider, AdjointFieldsCannotBeRegisteredBeforeTheFieldsThemselves) {
  configure(false, 1);
  auto n = make_net();
  fuse::FieldRegistry adjoint;
  const auto r = n->adapter[0]->register_adjoint_fields(adjoint, n->graph);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ADJ-INIT");
}
