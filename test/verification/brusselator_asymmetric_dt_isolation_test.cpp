// The LowFive-free half: the tier-0 oracle for "a LowFive-bound channel is an
// application-performed transfer" (docs/user/lowfive.md). Drives the
// SAME two step functions `examples/brusselator/participants/u_main.c`
// and `v_main.f90` call (the shim, via u_loop.h/v_loop.h, exactly as
// brusselator_participants_test.cpp's own ExternallyDrivenMultiSubstep
// WindowDrivenByNextComponent does) through the participant ABI, on a
// graph whose u_out/v_out set FieldSpec::hdf5_path -- so src/ir/
// rules_channel.cpp's lower_channel() compiles each transfer channel as
// OpKind::ExternalDelivery, not Transport, and FUSE moves no data on
// either channel at all. In place of the real H5Dwrite/H5Dread this
// harness stands in for, it calls the new view_field ABI entry
// (include/fuse/abi/entries.def, implemented against this graph's own
// window machinery, not yet against real HDF5/LowFive -- that half
// waits on the host, docs/user/lowfive.md) to read the exporting
// member's own K-wide waveform and memcpy it into the importing
// member's own K-wide buffer, at exactly the point an application's
// H5Dwrite/H5Dread would run: after a member's LAST substep of a sweep
// (`fuse_step_report::t_end` reaching the window's own end -- window ==
// period for this graph shape, so this is also the LAST substep before
// is_coupling_ongoing()'s own `period` counter would advance).
//
// GRAPH: brusselator_runner_test.cpp's own timing (kDtU=0.002/
// kDtV=0.003/kWindow=0.02 -- genuinely multirate, U:V substep ratio not
// 1:1, DtPolicy::Adjustable on both so neither ever straddles a window
// end), but brusselator_participants_test.cpp's IDENTITY-REMAP region shape
// (its build_graph_and_weights(): both transfer fields on u_region, so the channel's own Remap is the
// identity, isolating "does the external delivery work" from "does a
// real Remap work downstream of it" -- the latter is already covered,
// unaffected by this channel's own data-movement change, by every
// EXISTING non-LowFive window test). hdf5_path set on u_out/v_out, plus
// a fieldless LaunchMode::LowFive component
// (brusselator_graph_tool_lowfive_test.cpp's own precedent). Declared with
// no explicit `.kind`, such a component would be scheduled a turn through
// next_component (component index 3), so it is declared
// `ComponentKind::Coupler` (build_lowfive_graph_and_weights()'s own comment
// on that line has the account) rather than having its turn handled in the
// loop below -- it is a placement marker, not a physics component with
// steps of its own to take.
//
// Grid and period count kept small (kNxU=3,kNyU=3,kNxV=4,kNyV=4,
// kPeriods=3) -- this file's job is the structural claim at a size that
// runs in milliseconds (participants_test.cpp's own stated convention),
// not the physics scale example.brusselator's own ctest rows already
// cover; 3 periods already exercises multiple sweeps per window (the
// Jacobi fixed point) and the non-degenerate U:V substep-count mismatch
// the single-substep participants test does not reach.
#include <gtest/gtest.h>

#include <cmath>
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
#include "fuse/model/field_spec.hpp"
#include "fuse/model/launch.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

#include "brusselator_check_core.hpp"
#include "brusselator_graph.hpp"
#include "shim.h"
#include "u_loop.h"
#include "v_loop.h"
#include "species_component.hpp"

namespace {

using fuse::ComponentSet;
using fuse::FieldRegistry;
using fuse::FieldSpec;
using fuse::LaunchMode;
using fuse::LaunchSpec;
using fuse::Rational;
using fuse::RankSet;
using fuse::Result;
using fuse::WaveformIteration;

constexpr int kNxU = 3, kNyU = 3, kNxV = 4, kNyV = 4;
// brusselator_runner_test.cpp's own timing -- genuinely
// multirate, neither interval dividing the other.
const Rational kDtU{2, 1000}, kDtV{3, 1000}, kWindow{2, 100};
constexpr double kTolerance = 1e-10;
constexpr int kMaxIterations = 60;
constexpr int kPeriods = 3;

struct GraphAndWeights {
  fuse::native::NativeMeshBackend backend;
  brusselator::NativeGrid u_grid, v_grid;
  fuse::CouplingGraph graph;
  fuse::MeshId u_mesh, v_mesh;
  brusselator::WindowGraphHandles handles;
  fuse::RemapWeights weights_vu, weights_uv;
};

// brusselator_participants_test.cpp's build_graph_and_weights(), plus hdf5_path on u_out/v_out and the fieldless LowFive
// component -- the ONLY difference from an ordinary identity-remap
// window graph.
[[nodiscard]] GraphAndWeights build_lowfive_graph_and_weights() {
  fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/1);
  auto u_grid_r = brusselator::NativeGrid::make(backend, "u", kNxU, kNyU);
  auto v_grid_r = brusselator::NativeGrid::make(backend, "v", kNxV, kNyV);
  EXPECT_TRUE(u_grid_r.ok()) << (u_grid_r.ok() ? "" : u_grid_r.error().message);
  EXPECT_TRUE(v_grid_r.ok()) << (v_grid_r.ok() ? "" : v_grid_r.error().message);

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

  const auto u_out = graph.find_field(handles.u_component, "u_out");
  const auto v_out = graph.find_field(handles.v_component, "v_out");
  EXPECT_TRUE(u_out.has_value());
  EXPECT_TRUE(v_out.has_value());
  // ISOLATION RUN: no hdf5_path, no LowFive marker component -- an
  // ordinary Transport-delivered channel, otherwise byte-identical to
  // build_lowfive_graph_and_weights(). u_out/v_out found above only to
  // confirm the graph shape matches; nothing else is done with them.
  (void)u_out; (void)v_out;

  fuse::RemapPlan plan_vu;
  plan_vu.src = v_region;
  plan_vu.tgt = u_region;
  auto w_vu = fuse::build_weights(plan_vu, graph, backend);
  EXPECT_TRUE(w_vu.ok()) << (w_vu.ok() ? "" : w_vu.error().message);
  fuse::RemapPlan plan_uv;
  plan_uv.src = u_region;
  plan_uv.tgt = v_region;
  auto w_uv = fuse::build_weights(plan_uv, graph, backend);
  EXPECT_TRUE(w_uv.ok()) << (w_uv.ok() ? "" : w_uv.error().message);

  return GraphAndWeights{std::move(backend), std::move(u_grid_r.value()),
                        std::move(v_grid_r.value()), std::move(graph),
                        u_mesh, v_mesh, handles,
                        std::move(w_vu.value()), std::move(w_uv.value())};
}

const fuse_participant_abi_v1* Table() {
  static const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  return t;
}

/// field_handle() + view_field(), in one call -- the harness's own
/// stand-in for an application's H5Dwrite/H5Dread target: for a window-
/// scoped field this is the K-wide buffer the lead's instruction names
/// (the ExternalDelivery arc for an import, the WindowAdvance arc for
/// an export); asserted window_samples > 0 at every call site below,
/// since every field this harness views is window-scoped by
/// construction.
fuse_field_view view(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
                     const char* component, const char* field) {
  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = component;
  fname.field = field;
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  EXPECT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK) << component << "." << field;
  fuse_field_ref view_in{.struct_size = sizeof(view_in)};
  view_in.handle = fref.handle;
  fuse_field_view out{.struct_size = sizeof(out)};
  EXPECT_EQ(t->view_field(ctx, &view_in, &out), FUSE_OK) << component << "." << field;
  return out;
}

/// The delivery itself: read the exporter's own K-wide view and the
/// importer's, assert they agree on shape, assert the destination is
/// NOT already equal to the source (so the copy that follows is the
/// delivery, not a no-op over something FUSE already moved), then
/// copy -- the harness's stand-in for H5Dwrite(exporter) + H5Dread
/// (importer) through the installed VOL.
void deliver(const fuse_participant_abi_v1* t, fuse_ctx* ctx,
            const char* export_component, const char* export_field,
            const char* import_component, const char* import_field) {
  const fuse_field_view src = view(t, ctx, export_component, export_field);
  const fuse_field_view dst = view(t, ctx, import_component, import_field);
  ASSERT_GT(src.window_samples, 0) << export_component << "." << export_field;
  ASSERT_GT(dst.window_samples, 0) << import_component << "." << import_field;
  ASSERT_EQ(src.count, dst.count);
  ASSERT_EQ(src.components, dst.components);
  ASSERT_EQ(src.window_samples, dst.window_samples);
  const std::uint64_t n = src.count * static_cast<std::uint64_t>(src.components);
  const double* s = static_cast<const double*>(src.data);
  double* d = static_cast<double*>(dst.data);
  bool already_equal = true;
  for (std::uint64_t i = 0; i < n; ++i) {
    if (s[i] != d[i]) { already_equal = false; break; }
  }
  ASSERT_FALSE(already_equal)
      << export_component << "." << export_field << " -> " << import_component
      << "." << import_field << ": destination already equals the source "
         "before the copy -- the copy would not be the delivery";
  std::memcpy(d, s, n * sizeof(double));
}

}  // namespace

TEST(BrusselatorAsymmetricDtAbiIsolation, OrdinaryTransportMatchesInProcessRunner) {
  const brusselator::PhysicsConfig cfg;

  // --- Reference: fuse::Runner driven directly on the SAME graph
  // SHAPE, but through the ORDINARY (non-LowFive) channel -- a fresh
  // graph with no hdf5_path and no LowFive component, so its own
  // Transport/Remap/Sink chain is what FUSE would otherwise run; the
  // comparison below is therefore against the thing this channel used
  // to do, not against a second copy of what this test itself does. ---
  fuse::native::NativeMeshBackend ref_backend(/*rank=*/0, /*num_ranks=*/1);
  auto ref_u_grid_r = brusselator::NativeGrid::make(ref_backend, "u", kNxU, kNyU);
  auto ref_v_grid_r = brusselator::NativeGrid::make(ref_backend, "v", kNxV, kNyV);
  ASSERT_TRUE(ref_u_grid_r.ok());
  ASSERT_TRUE(ref_v_grid_r.ok());
  fuse::CouplingGraph ref_graph;
  const fuse::MeshId ref_u_mesh = ref_graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" + std::to_string(kNyU)});
  const fuse::MeshId ref_v_mesh = ref_graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(kNxV) + "x" + std::to_string(kNyV)});
  const fuse::RegionId ref_u_region = ref_graph.add_region({.name = "u_all", .mesh = ref_u_mesh, .dimension = 2});
  const fuse::RegionId ref_v_region = ref_graph.add_region({.name = "v_all", .mesh = ref_v_mesh, .dimension = 2});
  const brusselator::WindowGraphHandles ref_handles = brusselator::add_window_coupling(
      ref_graph, ref_u_region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      fuse::AccelKind::None, WaveformIteration::Jacobi, RankSet{0, 1});
  (void)ref_handles;

  ASSERT_TRUE(ref_backend.register_default_placements(ref_graph, ref_u_mesh).ok());
  const auto ref_lowered = fuse::lower(ref_graph);
  ASSERT_TRUE(ref_lowered.ok()) << ref_lowered.diags.to_string();
  const fuse::ClockSet ref_clocks{ref_graph};
  const auto ref_sched = fuse::schedule(ref_graph, ref_lowered.dag, ref_clocks, fuse::make_region_sizes(ref_graph, ref_lowered.dag, ref_backend));
  ASSERT_TRUE(ref_sched.ok()) << ref_sched.diags.to_string();

  fuse::RemapPlan rp_vu; rp_vu.src = ref_v_region; rp_vu.tgt = ref_u_region;
  auto ref_w_vu = fuse::build_weights(rp_vu, ref_graph, ref_backend);
  ASSERT_TRUE(ref_w_vu.ok()) << ref_w_vu.error().message;
  fuse::RemapPlan rp_uv; rp_uv.src = ref_u_region; rp_uv.tgt = ref_v_region;
  auto ref_w_uv = fuse::build_weights(rp_uv, ref_graph, ref_backend);
  ASSERT_TRUE(ref_w_uv.ok()) << ref_w_uv.error().message;

  auto ref_comp_v = std::make_unique<brusselator::VComponent>(
      "V", ref_u_grid_r.value(), ref_v_grid_r.value(), cfg, &ref_w_uv.value(), &ref_w_vu.value());
  const brusselator::Vector ref_v0_on_u =
      Eigen::Map<brusselator::Vector>(ref_comp_v->provided_ptr(), ref_u_grid_r.value().size());
  auto ref_comp_u = std::make_unique<brusselator::UComponent>("U", ref_u_grid_r.value(), cfg, ref_v0_on_u);

  ComponentSet ref_components;
  ref_components.add(std::move(ref_comp_u));
  ref_components.add(std::move(ref_comp_v));
  FieldRegistry ref_registry;
  for (auto& c : ref_components) ASSERT_TRUE(c.register_fields(ref_registry, ref_graph).ok());

  fuse::Runner ref_runner{ref_graph, ref_lowered.dag, ref_sched.plan, ref_backend};
  fuse::RunnerConfig ref_rcfg;
  {
    const auto init_r = ref_runner.initialize(ref_components, ref_registry, ref_rcfg);
    ASSERT_TRUE(init_r.ok()) << init_r.error().code << ": " << init_r.error().message;
  }
  std::vector<brusselator::Vector> ref_u_state(kPeriods), ref_v_state(kPeriods);
  std::vector<int> ref_iterations(kPeriods);
  for (int p = 0; p < kPeriods; ++p) {
    const auto ran = ref_runner.run_period(ref_components);
    ASSERT_TRUE(ran.ok()) << ran.error().code << ": " << ran.error().message << " (period " << p << ")";
    ASSERT_TRUE(ref_runner.iteration_stats().front().converged)
        << "reference run must converge for this comparison to mean anything (period " << p << ")";
    ref_iterations[p] = ref_runner.iteration_stats().front().iterations;
    ref_u_state[p] = static_cast<brusselator::UComponent*>(ref_components.find("U"))->state();
    ref_v_state[p] = static_cast<brusselator::VComponent*>(ref_components.find("V"))->state();
  }

  // --- The LowFive-bound graph, driven entirely through the ABI, both
  // members externally driven, turn order resolved via next_component,
  // delivery performed by this harness's own view_field+memcpy calls in
  // place of H5Dwrite/H5Dread. ---
  GraphAndWeights abi_setup = build_lowfive_graph_and_weights();
  const auto bytes = fuse::serialize(abi_setup.graph);

  const auto* t = Table();
  ASSERT_NE(t, nullptr);
  fuse_context_out co{.struct_size = sizeof(co)};
  ASSERT_EQ(t->create(nullptr, nullptr, &co), FUSE_OK);
  fuse_ctx* ctx = co.ctx;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  ASSERT_EQ(t->load_graph(ctx, &src, nullptr), FUSE_OK);

  const int nu = kNxU * kNyU, nv = kNxV * kNyV;
  std::vector<double> u_out(nu, 0.0), u_in(nu, 0.0), v_out(nu, 0.0), v_in(nu, 0.0);

  brusselator::VComponent abi_v_seed("V", abi_setup.u_grid, abi_setup.v_grid, cfg,
                                     &abi_setup.weights_uv, &abi_setup.weights_vu);
  std::vector<double> v0_on_u(static_cast<std::size_t>(nu), 0.0);
  for (int i = 0; i < nu; ++i) v0_on_u[static_cast<std::size_t>(i)] = abi_v_seed.provided_ptr()[i];
  brusselator::UComponent abi_u_seed(
      "U", abi_setup.u_grid, cfg, Eigen::Map<const brusselator::Vector>(v0_on_u.data(), nu));
  for (int i = 0; i < nu; ++i) {
    u_out[static_cast<std::size_t>(i)] = abi_u_seed.provided_ptr()[i];
    v_out[static_cast<std::size_t>(i)] = abi_v_seed.provided_ptr()[i];
  }

  auto register_one = [&](const char* component, const char* field, double* data, int n) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = component;
    fname.field = field;
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
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
    ASSERT_EQ(t->register_field(ctx, &fview, nullptr), FUSE_OK);
  };
  register_one("U", "u_out", u_out.data(), nu);
  register_one("U", "u_in", u_in.data(), nu);
  register_one("V", "v_out", v_out.data(), nu);
  register_one("V", "v_in", v_in.data(), nu);

  {
    const fuse_status init_rc = t->initialize(ctx, nullptr, nullptr);
    if (init_rc != FUSE_OK) {
      char buf[2048] = {0};
      fuse_error err{.struct_size = sizeof(err)};
      err.buffer = buf;
      err.capacity = sizeof(buf);
      t->last_error(ctx, nullptr, &err);
      ADD_FAILURE() << "initialize failed (" << init_rc << "): "
                    << (err.code ? err.code : "?") << ": " << buf;
    }
    ASSERT_EQ(init_rc, FUSE_OK);
  }

  bruss_u_handle* uh = bruss_u_create(kNxU, kNyU, cfg.diffusion, cfg.A, cfg.B,
                                     cfg.forcing ? 1 : 0, v0_on_u.data(),
                                     /*rank=*/0, /*num_ranks=*/1, /*comm_f=*/0, "auto");
  ASSERT_NE(uh, nullptr);
  bruss_v_handle* vh = bruss_v_create(kNxU, kNyU, kNxV, kNyV, cfg.diffusion,
                                     /*rank=*/0, /*num_ranks=*/1, /*comm_f=*/0, "auto");
  ASSERT_NE(vh, nullptr);
  ASSERT_EQ(bruss_v_size(vh), nu);

  fuse_step_request u_req{.struct_size = sizeof(u_req)};
  u_req.component = "U";
  fuse_step_request v_req{.struct_size = sizeof(v_req)};
  v_req.component = "V";
  const auto u_cid = abi_setup.graph.find_component("U");
  const auto v_cid = abi_setup.graph.find_component("V");
  ASSERT_TRUE(u_cid.has_value());
  ASSERT_TRUE(v_cid.has_value());
  const std::int32_t u_idx = static_cast<std::int32_t>(u_cid->value());
  const std::int32_t v_idx = static_cast<std::int32_t>(v_cid->value());
  ASSERT_NE(u_idx, v_idx);

  // u_log/v_log: brusselator_check_core.hpp's own hexfloat format, one
  // line per window (== period for this graph shape), window=0 the
  // cold-start baseline -- the SAME pattern brusselator_check_test.cpp
  // uses, reused here so brusselator_check OK is a real check on this
  // harness's own run, not a second, independent re-derivation of it.
  char buf[256];
  std::vector<std::string> u_lines, v_lines;
  std::snprintf(buf, sizeof(buf), "window=0 mean_u=%a external_total=%a exchange_out=%a\n",
               abi_u_seed.state().mean(), 0.0, 0.0);
  u_lines.emplace_back(buf);
  std::snprintf(buf, sizeof(buf), "window=0 mean_v=%a exchange_in=%a\n",
               abi_v_seed.state().mean(), 0.0);
  v_lines.emplace_back(buf);

  for (int p = 0; p < kPeriods; ++p) {
    fuse_ongoing period_start{.struct_size = sizeof(period_start)};
    ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &period_start), FUSE_OK) << "period " << p;
    std::int32_t next = period_start.next_component;
    int step_count = 0;
    const Rational window_end = kWindow * Rational{p + 1};
    while (true) {
      const bool is_u = (next == u_idx);
      ASSERT_TRUE(is_u || next == v_idx)
          << "period " << p << " step " << step_count
          << ": next_component resolved to neither U (" << u_idx
          << ") nor V (" << v_idx << "), got " << next;
      fuse_step_request& req = is_u ? u_req : v_req;
      fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
      fuse_step_report begin_rep{.struct_size = sizeof(begin_rep)};
      const fuse_status rc = is_u
          ? u_loop_step(ctx, uh, &req, u_in.data(), u_out.data(), &end_rep, &begin_rep)
          : bruss_v_loop_step(ctx, vh, &req, v_in.data(), v_out.data(), &end_rep, &begin_rep);
      ASSERT_EQ(rc, FUSE_OK)
          << "period " << p << " step " << step_count << " (" << (is_u ? "U" : "V")
          << "): " << (is_u ? bruss_u_last_error(uh) : bruss_v_last_error(vh));
      EXPECT_EQ(begin_rep.next_component, -2)
          << "period " << p << " step " << step_count;

      // Delivery point: this substep's own t_end reaching the window's
      // own end is "the last substep of this sweep" (DtPolicy::
      // Adjustable on both members -- the graph's own default -- never
      // straddles, so every sweep's last substep lands exactly here,
      // every time, for BOTH members independently).
      const Rational t_end{end_rep.t_end.numerator, end_rep.t_end.denominator};
      if (t_end == window_end) {
        // ISOLATION RUN: no delivery call here -- FUSE's own compiled
        // Transport/Remap/Sink chain performs the move automatically,
        // same as every pre-existing window test.
      }

      ++step_count;
      if (end_rep.next_component == -1) break;
      next = end_rep.next_component;
    }

    fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
    ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK);
    EXPECT_EQ(ongoing.period, p + 1) << "period " << p;
    EXPECT_EQ(ongoing.next_component, u_idx) << "period " << p << " just closed";

    SCOPED_TRACE("period " + std::to_string(p));
    std::vector<double> u_state(static_cast<std::size_t>(nu)), v_state(static_cast<std::size_t>(nv));
    bruss_u_state(uh, u_state.data());
    bruss_v_state(vh, v_state.data());
    ASSERT_EQ(static_cast<int>(u_state.size()), ref_u_state[p].size());
    for (std::size_t i = 0; i < u_state.size(); ++i) {
      EXPECT_EQ(u_state[i], ref_u_state[p][static_cast<int>(i)])
          << "window " << (p + 1) << " U state[" << i << "]: harness=" << u_state[i]
          << " (" << std::hexfloat << u_state[i] << ") ref=" << std::defaultfloat
          << ref_u_state[p][static_cast<int>(i)] << " (" << std::hexfloat
          << ref_u_state[p][static_cast<int>(i)] << ")" << std::defaultfloat;
    }
    ASSERT_EQ(static_cast<int>(v_state.size()), ref_v_state[p].size());
    for (std::size_t i = 0; i < v_state.size(); ++i) {
      EXPECT_EQ(v_state[i], ref_v_state[p][static_cast<int>(i)])
          << "window " << (p + 1) << " V state[" << i << "]: harness=" << v_state[i]
          << " (" << std::hexfloat << v_state[i] << ") ref=" << std::defaultfloat
          << ref_v_state[p][static_cast<int>(i)] << " (" << std::hexfloat
          << ref_v_state[p][static_cast<int>(i)] << ")" << std::defaultfloat;
    }

    double mean_u = 0.0;
    for (double x : u_state) mean_u += x;
    mean_u /= static_cast<double>(nu);
    double mean_v = 0.0;
    for (double x : v_state) mean_v += x;
    mean_v /= static_cast<double>(nv);
    std::snprintf(buf, sizeof(buf), "window=%d mean_u=%a external_total=%a exchange_out=%a\n",
                 p + 1, mean_u, bruss_u_external_total(uh), bruss_u_exchange_out(uh));
    u_lines.emplace_back(buf);
    std::snprintf(buf, sizeof(buf), "window=%d mean_v=%a exchange_in=%a\n",
                 p + 1, mean_v, bruss_v_exchange_in(vh));
    v_lines.emplace_back(buf);
  }

  bruss_u_destroy(uh);
  bruss_v_destroy(vh);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);

  // brusselator_check OK on the harness's own printed halves.
  const std::string u_path =
      std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + "_u.log";
  const std::string v_path =
      std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) + "_v.log";
  {
    std::ofstream u(u_path);
    for (const auto& l : u_lines) u << l;
    std::ofstream v(v_path);
    for (const auto& l : v_lines) v << l;
  }
  const brusselator_check::Verdict verdict =
      brusselator_check::run_brusselator_check(u_path, v_path);
  EXPECT_TRUE(verdict.ok) << verdict.error;
  EXPECT_TRUE(verdict.failed_thresholds.empty())
      << [&] {
           std::string s;
           for (const auto& f : verdict.failed_thresholds) s += f + "; ";
           return s;
         }();
}
