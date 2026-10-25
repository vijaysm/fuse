// The tier-0 test for brusselator_check_core's own assembler. Drives a short real run
// through the SAME participant ABI + shim + u_loop_step/v_loop_step
// test/verification/brusselator_participants_test.cpp uses, logging
// each window's mean_u/external_total/exchange_out (U) and
// mean_v/exchange_in (V) to two temp files in brusselator_check_core.
// hpp's own log format, then calls run_brusselator_check() directly on
// them (no subprocess -- the SAME split shim.h/loop_step.h already
// follow between physics/ABI logic and whatever drives it).
//
// The mutation "restore dt*q_prev in the shim" is proven separately, by
// hand, against species_core.hpp's own UCore::step() (the shared physics
// this shim wraps) -- build-evidence/mutations/brusselator.toml has the
// account; this file's own test is the harness that mutation is proven
// against (PASSES on the real code, FAILS once that one line is reverted).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

#include "brusselator_check_core.hpp"
#include "brusselator_graph.hpp"
#include "shim.h"
#include "u_loop.h"
#include "v_loop.h"
#include "species_component.hpp"

namespace {

using fuse::Rational;
using fuse::RankSet;
using fuse::WaveformIteration;

constexpr int kNxU = 2, kNyU = 2, kNxV = 2, kNyV = 2;
const Rational kDtU{1, 200}, kDtV{1, 200}, kWindow{1, 200};  // degenerate: one substep per window
constexpr double kTolerance = 1e-10;
constexpr int kMaxIterations = 60;
constexpr int kPeriods = 2;

const fuse_participant_abi_v1* Table() {
  static const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  return t;
}

struct WrittenLogs {
  std::string u_path, v_path;
};

/// Runs the graph through the participant ABI for `kPeriods` periods
/// (one substep per window, the degenerate case -- matches
/// brusselator_participants_test.cpp's own first test), writing both
/// logs as it goes. Returns the two temp file paths.
WrittenLogs run_and_log(const ::testing::TestInfo* info) {
  const brusselator::PhysicsConfig cfg;
  fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/1);
  auto u_grid_r = brusselator::NativeGrid::make(backend, "u", kNxU, kNyU);
  auto v_grid_r = brusselator::NativeGrid::make(backend, "v", kNxV, kNyV);
  EXPECT_TRUE(u_grid_r.ok());
  EXPECT_TRUE(v_grid_r.ok());

  fuse::CouplingGraph graph;
  const fuse::MeshId u_mesh = graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" + std::to_string(kNyU)});
  const fuse::MeshId v_mesh = graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(kNxV) + "x" + std::to_string(kNyV)});
  const fuse::RegionId u_region = graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  const fuse::RegionId v_region = graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      graph, u_region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      fuse::AccelKind::None, WaveformIteration::Jacobi, RankSet{0, 1});
  (void)handles;

  fuse::RemapPlan plan_vu;
  plan_vu.src = v_region;
  plan_vu.tgt = u_region;
  auto w_vu = fuse::build_weights(plan_vu, graph, backend);
  EXPECT_TRUE(w_vu.ok());
  fuse::RemapPlan plan_uv;
  plan_uv.src = u_region;
  plan_uv.tgt = v_region;
  auto w_uv = fuse::build_weights(plan_uv, graph, backend);
  EXPECT_TRUE(w_uv.ok());

  const auto bytes = fuse::serialize(graph);
  const auto* t = Table();
  EXPECT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  EXPECT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  EXPECT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  const int nu = kNxU * kNyU, nv = kNxV * kNyV;
  std::vector<double> u_out(nu, 0.0), u_in(nu, 0.0), v_out(nu, 0.0), v_in(nu, 0.0);

  brusselator::VComponent v_seed("V", u_grid_r.value(), v_grid_r.value(), cfg, &w_uv.value(), &w_vu.value());
  std::vector<double> v0_on_u(static_cast<std::size_t>(nu), 0.0);
  for (int i = 0; i < nu; ++i) v0_on_u[static_cast<std::size_t>(i)] = v_seed.provided_ptr()[i];
  brusselator::UComponent u_seed(
      "U", u_grid_r.value(), cfg, Eigen::Map<const brusselator::Vector>(v0_on_u.data(), nu));
  for (int i = 0; i < nu; ++i) {
    u_out[static_cast<std::size_t>(i)] = u_seed.provided_ptr()[i];
    v_out[static_cast<std::size_t>(i)] = v_seed.provided_ptr()[i];
  }

  auto register_one = [&](const char* component, const char* field, double* data, int n) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = component;
    fname.field = field;
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    EXPECT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
    fuse_field_view fview{.struct_size = sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = static_cast<uint64_t>(n);
    fview.owned_count = static_cast<uint64_t>(n);
    fview.stride = 1;
    fview.data = data;
    EXPECT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  };
  register_one("U", "u_out", u_out.data(), nu);
  register_one("U", "u_in", u_in.data(), nu);
  register_one("V", "v_out", v_out.data(), nu);
  register_one("V", "v_in", v_in.data(), nu);

  EXPECT_EQ(t->initialize(ctx, nullptr, nullptr), FUSE_OK);

  bruss_u_handle* uh = bruss_u_create(kNxU, kNyU, cfg.diffusion, cfg.A, cfg.B,
                                      cfg.forcing ? 1 : 0, v0_on_u.data(),
                                      /*rank=*/0, /*num_ranks=*/1, /*comm_f=*/0, "auto");
  bruss_v_handle* vh = bruss_v_create(kNxU, kNyU, kNxV, kNyV, cfg.diffusion,
                                      /*rank=*/0, /*num_ranks=*/1, /*comm_f=*/0, "auto");
  EXPECT_NE(uh, nullptr);
  EXPECT_NE(vh, nullptr);

  const std::string base = std::string(::testing::TempDir()) +
      (info ? info->name() : "brusselator_check");
  WrittenLogs logs{base + "_u.log", base + "_v.log"};
  std::ofstream u_log(logs.u_path, std::ios::trunc);
  std::ofstream v_log(logs.v_path, std::ios::trunc);

  // window=0 baseline row -- before any window has advanced.
  char buf[256];
  std::snprintf(buf, sizeof(buf), "window=0 mean_u=%a external_total=%a exchange_out=%a\n",
               u_seed.state().mean(), bruss_u_external_total(uh), bruss_u_exchange_out(uh));
  u_log << buf;
  std::snprintf(buf, sizeof(buf), "window=0 mean_v=%a exchange_in=%a\n",
               v_seed.state().mean(), bruss_v_exchange_in(vh));
  v_log << buf;

  fuse_step_request u_req{.struct_size = sizeof(u_req)};
  u_req.component = "U";
  fuse_step_request v_req{.struct_size = sizeof(v_req)};
  v_req.component = "V";
  const auto u_cid = graph.find_component("U");
  const auto v_cid = graph.find_component("V");
  EXPECT_TRUE(u_cid.has_value());
  EXPECT_TRUE(v_cid.has_value());
  const std::int32_t u_idx = static_cast<std::int32_t>(u_cid->value());
  const std::int32_t v_idx = static_cast<std::int32_t>(v_cid->value());

  for (int p = 0; p < kPeriods; ++p) {
    // Driven by next_component until the period closes (end_step's own
    // next_component == -1), exactly test/verification/brusselator_
    // participants_test.cpp's own ExternallyDrivenMultiSubstepWindow
    // DrivenByNextComponent does -- this test has no reference Runner's
    // own iteration count to drive a fixed number of sweeps by, so it
    // cannot assume one sweep converges the window the way a single-
    // sweep-per-window test could.
    fuse_ongoing period_start{.struct_size = sizeof(period_start)};
    EXPECT_EQ(t->is_coupling_ongoing(ctx, nullptr, &period_start), FUSE_OK);
    std::int32_t next = period_start.next_component;
    for (;;) {
      const bool is_u = (next == u_idx);
      EXPECT_TRUE(is_u || next == v_idx) << "period " << p << ": next_component=" << next;
      fuse_step_request& req = is_u ? u_req : v_req;
      fuse_step_report rep{.struct_size = sizeof(rep)};
      fuse_step_report begin_rep{.struct_size = sizeof(begin_rep)};
      const fuse_status rc = is_u
          ? u_loop_step(ctx, uh, &req, u_in.data(), u_out.data(), &rep, &begin_rep)
          : bruss_v_loop_step(ctx, vh, &req, v_in.data(), v_out.data(), &rep, &begin_rep);
      EXPECT_EQ(rc, FUSE_OK) << "period " << p << " (" << (is_u ? "U" : "V") << ")";
      if (rc != FUSE_OK || rep.next_component == -1) break;  // period complete (or a failure already reported)
      next = rep.next_component;
    }

    std::vector<double> u_state(static_cast<std::size_t>(nu));
    bruss_u_state(uh, u_state.data());
    double mean_u = 0.0;
    for (double x : u_state) mean_u += x;
    mean_u /= static_cast<double>(nu);

    std::vector<double> v_state(static_cast<std::size_t>(nv));
    bruss_v_state(vh, v_state.data());
    // v_seed's own grid is kNxV*kNyV; mean_v here must be V's own native
    // mean (matches main.cpp's own `pv->state().mean()`, V's native
    // state, not the u-grid-shaped exchange field).
    double mean_v = 0.0;
    for (double x : v_state) mean_v += x;
    mean_v /= static_cast<double>(nv);

    std::snprintf(buf, sizeof(buf), "window=%d mean_u=%a external_total=%a exchange_out=%a\n",
                 p + 1, mean_u, bruss_u_external_total(uh), bruss_u_exchange_out(uh));
    u_log << buf;
    std::snprintf(buf, sizeof(buf), "window=%d mean_v=%a exchange_in=%a\n",
                 p + 1, mean_v, bruss_v_exchange_in(vh));
    v_log << buf;
  }
  u_log.close();
  v_log.close();

  bruss_u_destroy(uh);
  bruss_v_destroy(vh);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
  return logs;
}

}  // namespace

TEST(BrusselatorCheck, PassesOnARealShimDrivenRun) {
  const WrittenLogs logs = run_and_log(::testing::UnitTest::GetInstance()->current_test_info());
  const brusselator_check::Verdict v =
      brusselator_check::run_brusselator_check(logs.u_path, logs.v_path);
  ASSERT_TRUE(v.ok) << v.error;
  EXPECT_TRUE(v.failed_thresholds.empty())
      << [&] { std::string s; for (auto& f : v.failed_thresholds) s += f + "; "; return s; }();
  EXPECT_EQ(v.windows, kPeriods);
  EXPECT_LT(v.max_window_balance_error, 1e-11);
  EXPECT_LT(v.max_window_exchange_error, 1e-9);
}

TEST(BrusselatorCheck, StructurallyMismatchedLogsAreRejected) {
  const std::string base = std::string(::testing::TempDir()) + "mismatched";
  const std::string u_path = base + "_u.log", v_path = base + "_v.log";
  {
    std::ofstream u(u_path, std::ios::trunc);
    u << "window=0 mean_u=0x1p+0 external_total=0x0p+0 exchange_out=0x0p+0\n";
    u << "window=1 mean_u=0x1p+0 external_total=0x0p+0 exchange_out=0x0p+0\n";
  }
  {
    std::ofstream v(v_path, std::ios::trunc);
    v << "window=0 mean_v=0x1p+0 exchange_in=0x0p+0\n";
    // V's log is one row short -- a structural mismatch, not a physics failure.
  }
  const brusselator_check::Verdict v = brusselator_check::run_brusselator_check(u_path, v_path);
  EXPECT_FALSE(v.ok);
  EXPECT_FALSE(v.error.empty());
}
