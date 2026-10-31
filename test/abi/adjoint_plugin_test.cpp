// A provider plugin with an adjoint, end to end through the C table (plans/2026-10-22-universal-
// number-types.md, stage U6, the C and Fortran entry points; docs/user/providers.md, "Adjoint").
// Both components of the coupling are C plugins (test/abi/plugins/adjoint_provider.c, built once
// for each): the graph names their shared objects, fuse_initialize() mounts them, and fuse_run()
// steps them. The caller never touches a component. It sets the initial state and reads the final
// state through fuse_view_field(), runs the coupling, initializes the reverse sweep, seeds the
// adjoint of the final state through fuse_view_adjoint_field(), reverses with fuse_run_adjoint(),
// and reads the adjoint of the initial state the same way; the derivative gate's identity
// <G x, y> = <x, G^T y> holds within its bound (fuse/linalg/derivative_gate.hpp).

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
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/base/units.hpp"
#include "fuse/linalg/derivative_gate.hpp"
#include "fuse/model/channel.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/launch.hpp"

namespace {

namespace u = fuse::units;

const fuse_participant_abi_v1* Table() { return fuse_coupler_abi(FUSE_ABI_VERSION); }

constexpr const char* kName[2] = {"p", "q"};

fuse::CouplingGraph build_graph() {
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "m1", .source = "generated:line:8"});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "m2", .source = "generated:line:5"});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{
      .name = "p", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin, .plugin_path = FUSE_TEST_ADJOINT_PLUGIN_P}});
  const auto q = g.add_component(fuse::ComponentDecl{
      .name = "q", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin, .plugin_path = FUSE_TEST_ADJOINT_PLUGIN_Q}});
  const auto out = g.add_field(p, fuse::FieldSpec{.name = "out", .region = r1, .components = 1,
                                                  .units = u::kKelvin, .intent = fuse::Intent::Provides});
  const auto pin = g.add_field(p, fuse::FieldSpec{.name = "in", .region = r1, .components = 1,
                                                  .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto in = g.add_field(q, fuse::FieldSpec{.name = "in", .region = r2, .components = 1,
                                                 .units = u::kKelvin, .intent = fuse::Intent::Requires});
  const auto qout = g.add_field(q, fuse::FieldSpec{.name = "out", .region = r2, .components = 1,
                                                   .units = u::kKelvin, .intent = fuse::Intent::Provides});
  g.add_channel("link", out, in);
  const auto back = g.add_channel("back", qout, pin);
  g.channel(back).lag(1).allow_time_level_mismatch("the loop is closed by one step of lag");
  return g;
}

struct Driver {
  const fuse_participant_abi_v1* t = Table();
  fuse_ctx* ctx = nullptr;
  int periods = 1;
  std::int32_t handle[2][2] = {{-1, -1}, {-1, -1}};  // [component][0 out, 1 in]

  explicit Driver(int n) : periods(n) {}
  ~Driver() {
    if (ctx != nullptr) (void)t->destroy(ctx, nullptr, nullptr);
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
  bool start(std::string* why) {
    const std::vector<std::byte> bytes = fuse::serialize(build_graph());
    fuse_context_out co{.struct_size = sizeof(co)};
    if (t->create(nullptr, nullptr, &co) != FUSE_OK) return (*why = "create", false);
    ctx = co.ctx;
    fuse_graph_source src{.struct_size = sizeof(src)};
    src.bytes = bytes.data();
    src.byte_count = bytes.size();
    if (t->load_graph(ctx, &src, nullptr) != FUSE_OK) return (*why = "load_graph: " + last_error(), false);
    for (int k = 0; k < 2; ++k) {
      for (int f = 0; f < 2; ++f) {
        fuse_field_name name{.struct_size = sizeof(name)};
        name.component = kName[k];
        name.field = f == 0 ? "out" : "in";
        fuse_field_ref ref{.struct_size = sizeof(ref)};
        if (t->field_handle(ctx, &name, &ref) != FUSE_OK) return (*why = "field_handle", false);
        handle[k][f] = ref.handle;
      }
    }
    if (t->initialize(ctx, nullptr, nullptr) != FUSE_OK) return (*why = "initialize: " + last_error(), false);
    return true;
  }
  bool forward(std::string* why) {
    fuse_run_request run{.struct_size = sizeof(run)};
    run.periods = periods;
    if (t->run(ctx, &run, nullptr) != FUSE_OK) return (*why = "run: " + last_error(), false);
    return true;
  }
  /// the array FUSE owns for the field of component k (f 0 out, 1 in), forward or adjoint
  std::span<double> array(int k, int f, bool adjoint) {
    fuse_field_ref ref{.struct_size = sizeof(ref)};
    ref.handle = handle[k][f];
    fuse_field_view v{.struct_size = sizeof(v)};
    const fuse_status rc = adjoint ? t->view_adjoint_field(ctx, &ref, &v) : t->view_field(ctx, &ref, &v);
    if (rc != FUSE_OK) return {};
    return {static_cast<double*>(v.data), static_cast<std::size_t>(v.count) * static_cast<std::size_t>(v.components)};
  }
  /// register nothing (the plugins' adjoint arrays are FUSE's own) and initialize the reverse sweep
  bool start_adjoint(std::string* why) {
    if (t->initialize_adjoint(ctx, nullptr, nullptr) != FUSE_OK) return (*why = "initialize_adjoint: " + last_error(), false);
    return true;
  }
};

// x: the initial `out` of p, then of q. y: the final out, in of p, in, out of q.
constexpr std::size_t kNx = 13, kNy = 13 + 8 + 5;
constexpr int kOrder[4][2] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};  // (component, field) of y, in order

fuse::Result<std::vector<double>> forward_map(int periods, std::span<const double> x) {
  Driver d(periods);
  std::string why;
  if (!d.start(&why)) return fuse::Failure{"E-TEST", why};
  std::size_t at = 0;
  for (int k = 0; k < 2; ++k) {
    for (double& v : d.array(k, 0, false)) v = x[at++];
  }
  if (!d.forward(&why)) return fuse::Failure{"E-TEST", why};
  std::vector<double> y;
  for (const auto& kf : kOrder) {
    const auto a = d.array(kf[0], kf[1], false);
    y.insert(y.end(), a.begin(), a.end());
  }
  return y;
}

fuse::Result<std::vector<double>> adjoint_map(int periods, std::span<const double> w) {
  Driver d(periods);
  std::string why;
  if (!d.start(&why) || !d.forward(&why) || !d.start_adjoint(&why)) return fuse::Failure{"E-TEST", why};
  std::size_t at = 0;
  for (const auto& kf : kOrder) {
    for (double& v : d.array(kf[0], kf[1], true)) v = w[at++];
  }
  fuse_run_request run{.struct_size = sizeof(run)};
  run.periods = -1;  // every period not yet reversed
  if (d.t->run_adjoint(d.ctx, &run, nullptr) != FUSE_OK) return fuse::Failure{"E-TEST", "run_adjoint: " + d.last_error()};
  std::vector<double> xbar;
  for (int k = 0; k < 2; ++k) {
    const auto a = d.array(k, 0, true);
    xbar.insert(xbar.end(), a.begin(), a.end());
  }
  return xbar;
}

std::vector<double> random_vector(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> v(n);
  for (double& e : v) e = dist(rng);
  return v;
}

}  // namespace

TEST(AbiAdjointPlugin, AdjointIdentityHoldsForCPluginsReversedThroughTheTable) {
  for (const int periods : {1, 4}) {
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
    const auto norm = fuse::linalg::estimate_operator_norm(kNy, kNx, fwd, adj);
    ASSERT_TRUE(norm.ok()) << norm.error().code << ": " << norm.error().message;
    const double eps = std::numeric_limits<double>::epsilon();
    double worst = 0.0;
    for (int k = 0; k < 3; ++k) {
      const auto x = random_vector(kNx, 100 + static_cast<std::uint64_t>(k));
      const auto y = random_vector(kNy, 200 + static_cast<std::uint64_t>(k));
      const auto r = fuse::linalg::adjoint_gate(kNy, kNx, fwd, adj, x, y, norm.value(),
                                                 fuse::linalg::kAdjointGateConstant);
      ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
      EXPECT_TRUE(r.value().passed) << periods << " periods, draw " << k;
      worst = std::max(worst, r.value().defect / (eps * r.value().norm_est * r.value().x_norm * r.value().y_norm));
    }
    std::printf("[abi adjoint plugin] %d periods: max observed adjoint ratio = %.3f (bound constant %.0f)\n", periods,
                worst, fuse::linalg::kAdjointGateConstant);
  }
}

TEST(AbiAdjointPlugin, TheAdjointArraysOfAPluginAreFusesOwnAndCannotBeRegistered) {
  Driver d(1);
  std::string why;
  ASSERT_TRUE(d.start(&why)) << why;
  std::vector<double> mine(8, 0.0);
  fuse_field_view v{.struct_size = sizeof(v)};
  v.handle = d.handle[0][0];
  v.element_type = FUSE_ELEMENT_F64;
  v.memspace = FUSE_MEMSPACE_HOST;
  v.ownership = FUSE_OWNERSHIP_OWNED;
  v.components = 1;
  v.count = 8;
  v.owned_count = 8;
  v.stride = 1;
  v.component_stride = 1;
  v.data = mine.data();
  EXPECT_NE(d.t->register_adjoint_field(d.ctx, &v, nullptr), FUSE_OK);
  EXPECT_EQ(d.last_code(), "E-ABI-ADJOINT-REGISTER-OWNER");
}

TEST(AbiAdjointPlugin, AdjointArraysAreZeroUntilTheCallerSeedsThem) {
  Driver d(2);
  std::string why;
  ASSERT_TRUE(d.start(&why) && d.forward(&why) && d.start_adjoint(&why)) << why;
  for (int k = 0; k < 2; ++k) {
    for (int f = 0; f < 2; ++f) {
      const auto a = d.array(k, f, true);
      ASSERT_FALSE(a.empty()) << k << f;
      for (const double v : a) EXPECT_EQ(v, 0.0);
    }
  }
}

TEST(AbiAdjointPlugin, EveryPeriodCanBeReversedOnceAndNoMore) {
  Driver d(2);
  std::string why;
  ASSERT_TRUE(d.start(&why) && d.forward(&why) && d.start_adjoint(&why)) << why;
  fuse_run_request run{.struct_size = sizeof(run)};
  run.periods = 1;
  EXPECT_EQ(d.t->run_adjoint(d.ctx, &run, nullptr), FUSE_OK);
  EXPECT_EQ(d.t->run_adjoint(d.ctx, &run, nullptr), FUSE_OK);
  EXPECT_NE(d.t->run_adjoint(d.ctx, &run, nullptr), FUSE_OK);
  EXPECT_EQ(d.last_code(), "E-ADJ-ORDER");
}

TEST(AbiAdjointPlugin, InitializeAdjointIsRefusedTwiceAndAdjointViewsBeforeIt) {
  Driver d(1);
  std::string why;
  ASSERT_TRUE(d.start(&why) && d.forward(&why)) << why;
  EXPECT_TRUE(d.array(0, 0, true).empty());
  EXPECT_EQ(d.last_code(), "E-ABI-ADJOINT-STATE");
  ASSERT_TRUE(d.start_adjoint(&why)) << why;
  EXPECT_NE(d.t->initialize_adjoint(d.ctx, nullptr, nullptr), FUSE_OK);
  EXPECT_EQ(d.last_code(), "E-ABI-ADJOINT-STATE");
}
