// The reverse sweep through the participant ABI (plans/2026-10-22-universal-number-types.md, stage
// U6, the C and Fortran entry points; docs/user/c-abi.md, "Reverse mode"). Everything here goes
// through the C table fuse_coupler_abi() returns, as a C or Fortran driver would: two components
// the caller drives itself with begin_step()/end_step(), each the linear step
//
//   out <- a out + c in,
//
// so the composed coupling is a linear map G from the initial state of each component to the state
// after N periods, and the adjoint identity <G x, y> = <x, G^T y> is checked within the derivative
// gate's own bound (fuse/linalg/derivative_gate.hpp). G x is the forward loop. G^T y is the reverse
// loop: the caller registers the adjoint array of each field, seeds those of the final state, and
// reverses each step between begin_adjoint_step() and end_adjoint_step(), newest first. A reverse
// step that drops the factor `a` must fail the same gate; the reports must carry the windows the
// forward steps had; and every misuse of the entries is refused by name.

#include "fuse/base/config.hpp"

#include <gtest/gtest.h>

#include "fuse_test_main.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/base/units.hpp"
#include "fuse/linalg/derivative_gate.hpp"
#include "fuse/model/channel.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/iteration.hpp"

namespace {

namespace u = fuse::units;

const fuse_participant_abi_v1* Table() { return fuse_coupler_abi(FUSE_ABI_VERSION); }

struct Cfg {
  bool loop = false;       // q returns its out to p one step late
  bool subcycle = false;   // p steps four times per period, its out averaged for q (no loop with it: the lag of the
                           // return channel is counted in coupling windows, and E-LAG refuses one of a whole period)
  int comps = 1;
  int periods = 1;
  double a[2] = {0.0, 0.0};
  double c[2] = {0.0, 0.0};
  bool drop_factor = false;  // the reverse step forgets the factor a of the state
  bool region = false;       // the channel p -> q is iterated to a convergence measure
};

constexpr std::size_t kCells[2] = {8, 5};
constexpr const char* kName[2] = {"p", "q"};

fuse::CouplingGraph build_graph(const Cfg& cfg) {
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:8"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:5"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{
      .name = "p", .ranks = fuse::RankSet{0, 1},
      .interval = cfg.subcycle ? fuse::Rational{1, 4} : fuse::Rational{1}});
  const auto q = g.add_component(
      fuse::ComponentDecl{.name = "q", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto out = g.add_field(p, fuse::FieldSpec{.name = "out", .region = r1, .components = cfg.comps,
                                                  .units = u::kKelvin, .intent = fuse::Intent::Provides});
  fuse::FieldId pin{};
  if (cfg.loop) {
    pin = g.add_field(p, fuse::FieldSpec{.name = "in", .region = r1, .components = cfg.comps,
                                         .units = u::kKelvin, .intent = fuse::Intent::Requires});
  }
  const auto in = g.add_field(q, fuse::FieldSpec{.name = "in", .region = r2, .components = cfg.comps,
                                                 .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto qout = g.add_field(q, fuse::FieldSpec{.name = "out", .region = r2, .components = cfg.comps,
                                                   .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto link = g.add_channel("link", out, in);
  if (cfg.subcycle) g.channel(link).accumulate(fuse::AccumOp::Mean);
  if (cfg.region) {
    fuse::ConvergenceCriterion crit;
    crit.add(fuse::Converged::Relative(1e-6, "link"));
    g.add_iteration_region(fuse::IterationRegionDecl{.name = "outer", .channels = {link}, .max_iterations = 12,
                                                     .convergence = crit,
                                                     .acceleration = fuse::AccelConfig{.kind = fuse::AccelKind::None}});
  }
  if (cfg.loop) {
    const auto back = g.add_channel("back", qout, pin);
    g.channel(back).lag(1).allow_time_level_mismatch("the loop is closed by one step of lag");
  }
  return g;
}

/// One run through the table: the context, the arrays the caller owns (forward and adjoint), and
/// what the forward loop saw, for the reverse loop to compare against.
struct Driver {
  const fuse_participant_abi_v1* t = Table();
  fuse_ctx* ctx = nullptr;
  Cfg cfg;
  bool has_in[2] = {false, false};
  std::vector<double> out[2], in[2], out_bar[2], in_bar[2];
  std::int32_t handle[2][2] = {{-1, -1}, {-1, -1}};  // [component][0 out, 1 in]
  struct Step {
    int component;
    std::int64_t step;
    fuse_exact_time t_begin, t_end;
  };
  std::vector<std::vector<Step>> forward;  // per period, in the order the caller was asked for them

  explicit Driver(const Cfg& c) : cfg(c) {
    has_in[0] = c.loop;
    has_in[1] = true;
    for (int k = 0; k < 2; ++k) {
      const std::size_t n = kCells[k] * static_cast<std::size_t>(c.comps);
      out[k].assign(n, 0.0);
      in[k].assign(n, 0.0);
      out_bar[k].assign(n, 0.0);
      in_bar[k].assign(n, 0.0);
    }
  }
  ~Driver() {
    if (ctx != nullptr) (void)t->destroy(ctx, nullptr, nullptr);
  }

  fuse_field_view view(std::int32_t h, std::vector<double>& v, int k) {
    fuse_field_view f{.struct_size = sizeof(fuse_field_view)};
    f.handle = h;
    f.element_type = FUSE_ELEMENT_F64;
    f.memspace = FUSE_MEMSPACE_HOST;
    f.ownership = FUSE_OWNERSHIP_OWNED;
    f.components = cfg.comps;
    f.count = kCells[k];
    f.owned_count = kCells[k];
    f.stride = cfg.comps;
    f.component_stride = 1;
    f.data = v.data();
    return f;
  }

  /// create, load, register the arrays of the run, initialize; false and a message on any failure
  bool start(std::string* why) {
    const std::vector<std::byte> bytes = fuse::serialize(build_graph(cfg));
    fuse_context_out co{.struct_size = sizeof(co)};
    if (t->create(nullptr, nullptr, &co) != FUSE_OK) return (*why = "create", false);
    ctx = co.ctx;
    fuse_graph_source src{.struct_size = sizeof(src)};
    src.bytes = bytes.data();
    src.byte_count = bytes.size();
    if (t->load_graph(ctx, &src, nullptr) != FUSE_OK) return (*why = "load_graph: " + last_error(), false);
    fuse_setup_options opts{.struct_size = sizeof(opts)};
    opts.has_end_time = 1;
    opts.end_time = fuse_exact_time{cfg.periods, 1};
    if (t->setup(ctx, &opts, nullptr) != FUSE_OK) return (*why = "setup: " + last_error(), false);
    for (int k = 0; k < 2; ++k) {
      for (int f = 0; f < 2; ++f) {
        if (f == 1 && !has_in[k]) continue;
        fuse_field_name name{.struct_size = sizeof(name)};
        name.component = kName[k];
        name.field = f == 0 ? "out" : "in";
        fuse_field_ref ref{.struct_size = sizeof(ref)};
        if (t->field_handle(ctx, &name, &ref) != FUSE_OK) return (*why = "field_handle: " + last_error(), false);
        handle[k][f] = ref.handle;
        fuse_field_view v = view(ref.handle, f == 0 ? out[k] : in[k], k);
        if (t->register_field(ctx, &v, nullptr) != FUSE_OK) return (*why = "register_field: " + last_error(), false);
      }
    }
    if (t->initialize(ctx, nullptr, nullptr) != FUSE_OK) return (*why = "initialize: " + last_error(), false);
    return true;
  }

  fuse_status register_adjoint(int k, int f) {
    fuse_field_view v = view(handle[k][f], f == 0 ? out_bar[k] : in_bar[k], k);
    return t->register_adjoint_field(ctx, &v, nullptr);
  }
  void register_all_adjoints() {
    for (int k = 0; k < 2; ++k) {
      for (int f = 0; f < 2; ++f) {
        if (f == 1 && !has_in[k]) continue;
        ASSERT_EQ(register_adjoint(k, f), FUSE_OK);
      }
    }
  }

  /// the forward loop: the caller steps whichever component the coupler expects next
  bool run_forward(std::string* why) {
    for (int period = 0; period < cfg.periods; ++period) {
      forward.emplace_back();
      while (true) {
        fuse_ongoing now{.struct_size = sizeof(now)};
        if (t->is_coupling_ongoing(ctx, nullptr, &now) != FUSE_OK) return (*why = "is_coupling_ongoing", false);
        if (now.ongoing == 0 || now.period != period) break;
        const int k = now.next_component;
        if (k < 0 || k > 1) return (*why = "next_component", false);
        fuse_step_request req{.struct_size = sizeof(req)};
        req.component = kName[k];
        fuse_step_report rep{.struct_size = sizeof(rep)};
        if (t->begin_step(ctx, &req, &rep) != FUSE_OK) return (*why = last_error(), false);
        forward.back().push_back({k, rep.step, rep.t_begin, rep.t_end});
        const std::size_t n = out[k].size();
        for (std::size_t i = 0; i < n; ++i) {
          out[k][i] = cfg.a[k] * out[k][i] + (has_in[k] ? cfg.c[k] * in[k][i] : 0.0);
        }
        if (t->end_step(ctx, &req, &rep) != FUSE_OK) return (*why = last_error(), false);
      }
    }
    return true;
  }

  std::string last_error() {
    fuse_error e{.struct_size = sizeof(e)};
    char buf[512] = {0};
    e.buffer = buf;
    e.capacity = sizeof(buf);
    (void)t->last_error(ctx, nullptr, &e);
    return std::string(e.code != nullptr ? e.code : "") + ": " + buf;
  }
  std::string last_code() {
    fuse_error e{.struct_size = sizeof(e)};
    (void)t->last_error(ctx, nullptr, &e);
    return e.code != nullptr ? e.code : "";
  }

  /// the reverse loop, newest period first and each period's steps in the reverse of the forward
  /// order; returns the reports' mismatches against the forward steps
  bool run_reverse(std::string* why, int* mismatches) {
    *mismatches = 0;
    for (int period = cfg.periods - 1; period >= 0; --period) {
      const std::vector<Step>& steps = forward[static_cast<std::size_t>(period)];
      for (std::size_t j = steps.size(); j-- > 0;) {
        const Step& fs = steps[j];
        fuse_step_request req{.struct_size = sizeof(req)};
        req.component = kName[fs.component];
        fuse_step_report begin{.struct_size = sizeof(begin)};
        if (t->begin_adjoint_step(ctx, &req, &begin) != FUSE_OK) return (*why = last_error(), false);
        if (begin.step != fs.step || begin.t_begin.numerator != fs.t_begin.numerator ||
            begin.t_begin.denominator != fs.t_begin.denominator || begin.t_end.numerator != fs.t_end.numerator ||
            begin.t_end.denominator != fs.t_end.denominator || begin.period != period) {
          ++*mismatches;
        }
        const int k = fs.component;
        const std::size_t n = out[k].size();
        for (std::size_t i = 0; i < n; ++i) {
          if (has_in[k]) in_bar[k][i] += cfg.c[k] * out_bar[k][i];
          if (!cfg.drop_factor) out_bar[k][i] *= cfg.a[k];
        }
        fuse_step_report end{.struct_size = sizeof(end)};
        if (t->end_adjoint_step(ctx, &req, &end) != FUSE_OK) return (*why = last_error(), false);
        const std::int32_t expected_next = j == 0 ? -1 : steps[j - 1].component;
        if (end.next_component != expected_next || end.step != fs.step) ++*mismatches;
      }
    }
    return true;
  }
};

// The vector layouts. x: the initial `out` of p, then of q. y: the final out of p, [in of p], in of q,
// out of q (the same order in both directions).
struct Dims {
  std::size_t nx = 0, ny = 0;
};
Dims dims(const Cfg& c) {
  const std::size_t comps = static_cast<std::size_t>(c.comps);
  Dims d;
  d.nx = (8 + 5) * comps;
  d.ny = d.nx + 5 * comps + (c.loop ? 8 * comps : 0);
  return d;
}

std::vector<std::vector<double>*> y_arrays(Driver& r, bool bars) {
  std::vector<std::vector<double>*> v;
  v.push_back(bars ? &r.out_bar[0] : &r.out[0]);
  if (r.cfg.loop) v.push_back(bars ? &r.in_bar[0] : &r.in[0]);
  v.push_back(bars ? &r.in_bar[1] : &r.in[1]);
  v.push_back(bars ? &r.out_bar[1] : &r.out[1]);
  return v;
}

fuse::Result<std::vector<double>> forward_map(const Cfg& cfg, std::span<const double> x) {
  Driver r(cfg);
  std::string why;
  if (!r.start(&why)) return fuse::Failure{"E-TEST", why};
  std::size_t at = 0;
  for (int k = 0; k < 2; ++k) {
    for (double& v : r.out[k]) v = x[at++];
  }
  if (!r.run_forward(&why)) return fuse::Failure{"E-TEST", why};
  std::vector<double> y;
  for (const auto* a : y_arrays(r, false)) y.insert(y.end(), a->begin(), a->end());
  return y;
}

fuse::Result<std::vector<double>> adjoint_map(const Cfg& cfg, std::span<const double> w, int* mismatches = nullptr) {
  Driver r(cfg);
  std::string why;
  if (!r.start(&why) || !r.run_forward(&why)) return fuse::Failure{"E-TEST", why};
  r.register_all_adjoints();
  if (r.t->initialize_adjoint(r.ctx, nullptr, nullptr) != FUSE_OK) {
    return fuse::Failure{"E-TEST", "initialize_adjoint: " + r.last_error()};
  }
  std::size_t at = 0;
  for (auto* a : y_arrays(r, true)) {
    for (double& v : *a) v = w[at++];
  }
  int bad = 0;
  if (!r.run_reverse(&why, &bad)) return fuse::Failure{"E-TEST", why};
  if (mismatches != nullptr) *mismatches = bad;
  std::vector<double> xbar;
  for (int k = 0; k < 2; ++k) xbar.insert(xbar.end(), r.out_bar[k].begin(), r.out_bar[k].end());
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

GateRun gate(const Cfg& cfg, int draws = 3) {
  GateRun out;
  const Dims d = dims(cfg);
  const fuse::linalg::HostApplyFn fwd = [&](std::span<const double> x, std::span<double> y) -> fuse::Result<void> {
    auto r = forward_map(cfg, x);
    if (!r.ok()) return r.error();
    std::copy(r.value().begin(), r.value().end(), y.begin());
    return fuse::Result<void>::success();
  };
  const fuse::linalg::HostApplyFn adj = [&](std::span<const double> w, std::span<double> xbar) -> fuse::Result<void> {
    auto r = adjoint_map(cfg, w);
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

Cfg make(bool loop, int comps, int periods, bool subcycle = false) {
  Cfg c;
  c.loop = loop;
  c.comps = comps;
  c.periods = periods;
  c.subcycle = subcycle;
  c.a[0] = 0.9;
  c.c[0] = 0.4;
  c.a[1] = 0.8;
  c.c[1] = 0.3;
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------

TEST(AbiAdjointParticipant, AdjointIdentityHoldsForStepsTheCallerReversesItself) {
  struct C {
    const char* name;
    Cfg cfg;
  };
  const C cases[] = {{"pipeline", make(false, 1, 2)},
                     {"pipeline_three_components", make(false, 3, 2)},
                     {"loop_lagged", make(true, 1, 4)},
                     {"loop_two_components", make(true, 2, 3)},
                     {"pipeline_subcycled_mean", make(false, 1, 3, true)}};
  double worst = 0.0;
  for (const C& c : cases) {
    const GateRun r = gate(c.cfg);
    EXPECT_TRUE(r.all_passed) << c.name << ": " << r.error;
    worst = std::max(worst, r.worst_ratio);
  }
  std::printf("[abi adjoint participant] max observed adjoint ratio = %.3f (bound constant %.0f)\n", worst,
              fuse::linalg::kAdjointGateConstant);
}

TEST(AbiAdjointParticipant, ACallerWhoseReverseStepDropsTheFactorOfTheStateIsDetected) {
  Cfg c = make(true, 1, 3);
  c.drop_factor = true;
  EXPECT_FALSE(gate(c).all_passed) << "the gate accepted an adjoint that is not the transpose";
}

TEST(AbiAdjointParticipant, TheReverseStepsReportTheWindowsTheForwardStepsHadInTheReverseOrder) {
  // p steps four times per period: the reports of the reverse steps must be those of the forward
  // steps, window by exact window, and next_component must walk the period backwards
  const Cfg c = make(false, 1, 3, true);
  const Dims d = dims(c);
  int mismatches = -1;
  const auto r = adjoint_map(c, random_vector(d.ny, 7), &mismatches);
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_EQ(mismatches, 0);
  Driver forward(c);
  std::string why;
  ASSERT_TRUE(forward.start(&why)) << why;
  ASSERT_TRUE(forward.run_forward(&why)) << why;
  ASSERT_EQ(forward.forward.size(), 3u);
  for (const auto& period : forward.forward) EXPECT_EQ(period.size(), 5u);  // four steps of p and one of q
}

TEST(AbiAdjointParticipant, TheViewOfAnAdjointArrayIsTheArrayTheCallerRegistered) {
  Driver r(make(false, 2, 1));
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  ASSERT_TRUE(r.run_forward(&why)) << why;
  r.register_all_adjoints();
  ASSERT_EQ(r.t->initialize_adjoint(r.ctx, nullptr, nullptr), FUSE_OK);
  fuse_field_ref ref{.struct_size = sizeof(ref)};
  ref.handle = r.handle[1][0];
  fuse_field_view v{.struct_size = sizeof(v)};
  ASSERT_EQ(r.t->view_adjoint_field(r.ctx, &ref, &v), FUSE_OK);
  EXPECT_EQ(v.data, static_cast<void*>(r.out_bar[1].data()));
  EXPECT_EQ(v.element_type, FUSE_ELEMENT_F64);
  EXPECT_EQ(v.memspace, FUSE_MEMSPACE_HOST);
  EXPECT_EQ(v.count, 5u);
  EXPECT_EQ(v.components, 2);
  EXPECT_EQ(v.stride, 2);
}

TEST(AbiAdjointParticipant, CapabilitiesReportTheReverseMode) {
  fuse_capability_set caps{.struct_size = sizeof(caps)};
  ASSERT_EQ(Table()->capabilities(nullptr, nullptr, &caps), FUSE_OK);
  EXPECT_NE(caps.flags & FUSE_CAP_ADJOINT, 0u);
}

// ---- refusals ---------------------------------------------------------------------------------

TEST(AbiAdjointParticipant, RunAdjointIsRefusedWhileAStepIsDrivenFromOutside) {
  Driver r(make(false, 1, 1));
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  ASSERT_TRUE(r.run_forward(&why)) << why;
  r.register_all_adjoints();
  ASSERT_EQ(r.t->initialize_adjoint(r.ctx, nullptr, nullptr), FUSE_OK);
  fuse_run_request req{.struct_size = sizeof(req)};
  req.periods = 1;
  EXPECT_EQ(r.t->run_adjoint(r.ctx, &req, nullptr), FUSE_ERR_UNSUPPORTED);
  EXPECT_EQ(r.last_code(), "E-ABI-RUN-EXTERNAL");
}

TEST(AbiAdjointParticipant, EntriesAreRefusedBeforeTheirPrerequisites) {
  Driver r(make(false, 1, 1));
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  ASSERT_TRUE(r.run_forward(&why)) << why;
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "q";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_NE(r.t->begin_adjoint_step(r.ctx, &req, &rep), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ABI-ADJOINT-STATE");
  fuse_field_ref ref{.struct_size = sizeof(ref)};
  ref.handle = r.handle[1][0];
  fuse_field_view v{.struct_size = sizeof(v)};
  EXPECT_NE(r.t->view_adjoint_field(r.ctx, &ref, &v), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ABI-ADJOINT-STATE");
  fuse_run_request run{.struct_size = sizeof(run)};
  run.periods = 1;
  EXPECT_NE(r.t->run_adjoint(r.ctx, &run, nullptr), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ABI-ADJOINT-STATE");
}

TEST(AbiAdjointParticipant, InitializeAdjointNamesAFieldWithoutAnAdjointArray) {
  Driver r(make(false, 1, 1));
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  ASSERT_TRUE(r.run_forward(&why)) << why;
  // every field but q.in
  ASSERT_EQ(r.register_adjoint(0, 0), FUSE_OK);
  ASSERT_EQ(r.register_adjoint(1, 0), FUSE_OK);
  EXPECT_NE(r.t->initialize_adjoint(r.ctx, nullptr, nullptr), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ADJ-UNBOUND");
}

TEST(AbiAdjointParticipant, InitializeAdjointRefusesAComponentTheCallerGaveNoAdjointArrays) {
  Driver r(make(false, 1, 1));
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  ASSERT_TRUE(r.run_forward(&why)) << why;
  // not one adjoint array registered at all
  EXPECT_NE(r.t->initialize_adjoint(r.ctx, nullptr, nullptr), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ADJ-COMPONENT");
}

TEST(AbiAdjointParticipant, RegisterAdjointFieldRefusesWhatItCannotUse) {
  Driver r(make(false, 1, 1));
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  {  // an element type other than binary64
    fuse_field_view v = r.view(r.handle[0][0], r.out_bar[0], 0);
    v.element_type = FUSE_ELEMENT_F32;
    EXPECT_NE(r.t->register_adjoint_field(r.ctx, &v, nullptr), FUSE_OK);
    EXPECT_EQ(r.last_code(), "E-ABI-ADJOINT-REGISTER-ELEMENT");
  }
  {  // device memory
    fuse_field_view v = r.view(r.handle[0][0], r.out_bar[0], 0);
    v.memspace = FUSE_MEMSPACE_DEVICE;
    EXPECT_NE(r.t->register_adjoint_field(r.ctx, &v, nullptr), FUSE_OK);
    EXPECT_EQ(r.last_code(), "E-ABI-ADJOINT-REGISTER-MEMSPACE");
  }
  {  // a handle that is no field
    fuse_field_view v = r.view(9999, r.out_bar[0], 0);
    EXPECT_NE(r.t->register_adjoint_field(r.ctx, &v, nullptr), FUSE_OK);
    EXPECT_EQ(r.last_code(), "E-ABI-ADJOINT-REGISTER-HANDLE");
  }
  {  // another layout than the field's own
    fuse_field_view v = r.view(r.handle[0][0], r.out_bar[0], 0);
    v.count = 7;
    v.owned_count = 7;
    EXPECT_NE(r.t->register_adjoint_field(r.ctx, &v, nullptr), FUSE_OK);
    EXPECT_EQ(r.last_code(), "E-ABI-ADJOINT-REGISTER-LAYOUT");
  }
  {  // twice
    ASSERT_EQ(r.register_adjoint(0, 0), FUSE_OK);
    EXPECT_NE(r.register_adjoint(0, 0), FUSE_OK);
    EXPECT_EQ(r.last_code(), "E-ABI-ADJOINT-REGISTER-DUPLICATE");
  }
}

TEST(AbiAdjointParticipant, ReverseStepsAreRefusedOutOfTurnTwiceOpenedOrNotOpened) {
  Driver r(make(false, 1, 1));
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  ASSERT_TRUE(r.run_forward(&why)) << why;
  r.register_all_adjoints();
  ASSERT_EQ(r.t->initialize_adjoint(r.ctx, nullptr, nullptr), FUSE_OK);
  fuse_step_request p{.struct_size = sizeof(p)}, q{.struct_size = sizeof(q)};
  p.component = "p";
  q.component = "q";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  // the newest step is q's: p is out of turn, and nothing is lost by asking
  EXPECT_NE(r.t->begin_adjoint_step(r.ctx, &p, &rep), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ABI-STEP-ORDER");
  EXPECT_NE(r.t->end_adjoint_step(r.ctx, &q, &rep), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ABI-STEP-NOT-OPEN");
  ASSERT_EQ(r.t->begin_adjoint_step(r.ctx, &q, &rep), FUSE_OK);
  EXPECT_NE(r.t->begin_adjoint_step(r.ctx, &q, &rep), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ABI-STEP-ALREADY-OPEN");
  EXPECT_NE(r.t->end_adjoint_step(r.ctx, &p, &rep), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ABI-STEP-NOT-OPEN");
  ASSERT_EQ(r.t->end_adjoint_step(r.ctx, &q, &rep), FUSE_OK);
  EXPECT_EQ(rep.next_component, 0);  // p, the component declared first
}

TEST(AbiAdjointParticipant, AStepTheCallerDrivesInsideAnIterationRegionIsRefused) {
  // the adjoint iteration of a region repeats its steps once per pass and cannot ask the caller for each
  Cfg c = make(false, 1, 1);
  c.region = true;
  Driver r(c);
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  r.register_all_adjoints();
  EXPECT_NE(r.t->initialize_adjoint(r.ctx, nullptr, nullptr), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ADJ-REGION");
}

TEST(AbiAdjointParticipant, AllPeriodsReversedIsAnError) {
  Driver r(make(false, 1, 1));
  std::string why;
  ASSERT_TRUE(r.start(&why)) << why;
  ASSERT_TRUE(r.run_forward(&why)) << why;
  r.register_all_adjoints();
  ASSERT_EQ(r.t->initialize_adjoint(r.ctx, nullptr, nullptr), FUSE_OK);
  int bad = 0;
  ASSERT_TRUE(r.run_reverse(&why, &bad)) << why;
  fuse_step_request q{.struct_size = sizeof(q)};
  q.component = "q";
  fuse_step_report rep{.struct_size = sizeof(rep)};
  EXPECT_NE(r.t->begin_adjoint_step(r.ctx, &q, &rep), FUSE_OK);
  EXPECT_EQ(r.last_code(), "E-ADJ-ORDER");
}
