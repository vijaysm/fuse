// The tier-0 oracle for "a LowFive-bound channel is an
// application-performed transfer" (docs/user/lowfive.md). Drives the
// SAME two step functions `examples/brusselator/participants/u_main.c`
// and `v_main.f90` call (the shim, via u_loop.h/v_loop.h, exactly as
// brusselator_participants_test.cpp's own ExternallyDrivenMultiSubstep
// WindowDrivenByNextComponent does) through the participant ABI, on a
// graph whose u_out/v_out set FieldSpec::hdf5_path -- so src/ir/
// rules_channel.cpp's lower_channel() compiles each transfer channel as
// OpKind::ExternalDelivery, not Transport, and FUSE moves no data on
// either channel at all. In place of the real H5Dwrite/H5Dread this
// harness stands in for, it calls the view_field ABI entry
// (include/fuse/abi/entries.def, implemented against this graph's own
// window machinery, not against real HDF5/LowFive -- that half
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
// end), but brusselator_participants_test.cpp's IDENTITY-REMAP region
// shape (its build_graph_and_weights(): both transfer fields on
// u_region, so the channel's own Remap is the identity, isolating "does
// the external delivery work" from "does a real Remap work downstream of
// it" -- the latter is already covered, unaffected by this channel's own
// data-movement change, by every non-LowFive window test). hdf5_path set
// on u_out/v_out, plus a fieldless LaunchMode::LowFive component
// (brusselator_graph_tool_lowfive_test.cpp's own precedent). Declared
// with no explicit `.kind`, such a component would be scheduled a turn
// through next_component (component index 3), so it is declared
// `ComponentKind::Coupler` (build_lowfive_graph_and_weights()'s own
// comment on that line has the account) rather than having its turn
// handled in the loop below -- it is a placement marker, not a physics
// component with steps of its own to take.
//
// The REAL Brusselator example's own grid and period count (kNxU=16,
// kNyU=16,kNxV=24,kNyV=24, kPeriods=150, --end 3 at window=1/50): a
// small placeholder (kNxU=3,kNyU=3,kNxV=4,kNyV=4, kPeriods=3) already
// exercises multiple sweeps per window (the Jacobi fixed point) and the
// non-degenerate U:V substep-count mismatch the single-substep
// participants test does not reach; this file confirms the SAME claim
// holds at the example's own real physics scale, for both waveform
// iterations (Jacobi and GaussSeidel, below).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
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

// The REAL Brusselator example's own grid and period count
// (examples/brusselator/main.cpp's own Args defaults, mirroring
// brusselator_graph_tool.cpp's and brusselator_participants_test.cpp's
// own) -- --end 3 at window=1/50 (0.02) is kPeriods=150 windows.
constexpr int kNxU = 16, kNyU = 16, kNxV = 24, kNyV = 24;
// brusselator_runner_test.cpp's own timing -- genuinely multirate,
// neither interval dividing the other.
const Rational kDtU{2, 1000}, kDtV{3, 1000}, kWindow{2, 100};
constexpr double kTolerance = 1e-11;
constexpr int kMaxIterations = 60;
constexpr int kPeriods = 150;

struct GraphAndWeights {
  fuse::native::NativeMeshBackend backend;
  brusselator::NativeGrid u_grid, v_grid;
  fuse::CouplingGraph graph;
  fuse::MeshId u_mesh, v_mesh;
  brusselator::WindowGraphHandles handles;
  fuse::RemapWeights weights_vu, weights_uv;
};

// brusselator_participants_test.cpp's build_graph_and_weights(), plus
// hdf5_path on u_out/v_out and the fieldless LowFive
// component -- the ONLY difference from an ordinary identity-remap
// window graph.
[[nodiscard]] GraphAndWeights build_lowfive_graph_and_weights(
    WaveformIteration wi = WaveformIteration::Jacobi) {
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
      fuse::AccelKind::None, wi, RankSet{0, 1});

  const auto u_out = graph.find_field(handles.u_component, "u_out");
  const auto v_out = graph.find_field(handles.v_component, "v_out");
  EXPECT_TRUE(u_out.has_value());
  EXPECT_TRUE(v_out.has_value());
  const_cast<FieldSpec&>(graph.field(*u_out)).hdf5_path = "brusselator_u.h5/grid/u_out";
  const_cast<FieldSpec&>(graph.field(*v_out)).hdf5_path = "brusselator_v.h5/grid/v_out";
  // Fieldless marker component (docs/user/lowfive.md, launch.hpp).
  // MEASURED: with no explicit `.kind` -- the
  // ComponentDecl default, ComponentKind::Physics -- fuse_initialize()'s
  // own wiring loop (src/abi/participant.cpp) does NOT exempt it the
  // way it exempts a real ComponentKind::Coupler entry
  // (`if (decl.kind == ComponentKind::Coupler) continue;`), so it
  // becomes a THIRD externally-driven component and next_component
  // resolves to it (component index 3) at period start -- this test's
  // own loop then has no step call for it and fails. `ComponentKind::
  // Coupler` is the right declaration for a fieldless placement marker
  // regardless of WHICH kind of placement it marks (ordinary coupler-
  // placement routing, or this one), and gets it the same exemption.
  // Left here as a finding for the graph-authoring contract: brusselator_
  // graph_tool.cpp's and brusselator_graph_tool_lowfive_test.cpp's own
  // "lowfive" component has the
  // IDENTICAL omission, unobserved there only because neither ever
  // drives it through a live Runner/ABI the way this test does.
  (void)graph.add_component({.name = "lowfive", .ranks = RankSet{0, 1},
                             .interval = Rational{1},
                             .kind = fuse::ComponentKind::Coupler,
                             .launch = LaunchSpec{.mode = LaunchMode::LowFive}});

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
/// scoped field this is the K-wide buffer
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
  // Diagnostic: for window 1 sweep 0,
  // element 0, all K samples, in hexfloat -- the first two delivery
  // calls in the whole run (u2v then v2u, both at window 1's own sweep
  // 0, since delivery only happens at a window boundary and this is
  // the first one either member reaches).
  static int call_count = 0;
  if (call_count < 2) {
    const double* s = static_cast<const double*>(src.data);
    const double* d = static_cast<const double*>(dst.data);
    fprintf(stderr, "deliver() call %d (%s.%s -> %s.%s), element 0, K=%d:\n",
           call_count, export_component, export_field, import_component, import_field,
           src.window_samples);
    for (int k = 0; k < src.window_samples; ++k) {
      fprintf(stderr, "  k=%d src=%a dst(before copy)=%a\n", k, s[k], d[k]);
    }
  }
  ++call_count;
  ASSERT_GT(src.window_samples, 0) << export_component << "." << export_field;
  ASSERT_GT(dst.window_samples, 0) << import_component << "." << import_field;
  ASSERT_EQ(src.count, dst.count);
  ASSERT_EQ(src.components, dst.components);
  ASSERT_EQ(src.window_samples, dst.window_samples);
  const std::uint64_t n = src.count * static_cast<std::uint64_t>(src.components);
  const double* s = static_cast<const double*>(src.data);
  double* d = static_cast<double*>(dst.data);
  // This function does not assert that the destination was NEVER already
  // equal to the source (a check for a genuine double-delivery): a single
  // field value can legitimately land on its own prior delivery once, late
  // in a long run (measured: window 149, the last, on its own final (5th)
  // sweep's b2a delivery, not during a monotonic approach to a fixed sweep
  // count -- periods 135-147 held steady at 4 sweeps; 148-149 rose to 5).
  // That is not distinguishable from a double-delivery by inspecting one
  // copy alone -- the real test is the state/sweep-count comparison below,
  // which checks every one of the kPeriods windows bit-for-bit.
  std::memcpy(d, s, n * sizeof(double));
}

}  // namespace

// Parameterized on waveform iteration (Jacobi and GaussSeidel, both
// exercised by the two TEST() wrappers below) so this claim is checked
// both ways at the real example's own physics scale.
void RunExternalDeliveryMatchesInProcessRunner(WaveformIteration wi) {
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
      fuse::AccelKind::None, wi, RankSet{0, 1});
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
  GraphAndWeights abi_setup = build_lowfive_graph_and_weights(wi);
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

  // This run's own requested period count is the fact that bounds it,
  // the SAME fact prime_next_request()'s own rollover
  // (src/abi/participant.cpp) checks before starting a period beyond it,
  // and the SAME fact fuse_is_coupling_ongoing() reads to answer
  // `ongoing` -- without declaring it here, there is nothing for either to
  // check against, and this graph's own window-per-period shape (every
  // sweep's own trailing ExternalDelivery) would let the Runner roll
  // forward forever once state reaches a fixed point.
  {
    fuse_setup_options opts{.struct_size = sizeof(opts)};
    opts.has_end_time = 1;
    const Rational end_time = kWindow * Rational{kPeriods};
    opts.end_time = fuse_exact_time{end_time.num(), end_time.den()};
    ASSERT_EQ(t->setup(ctx, &opts, nullptr), FUSE_OK);
  }

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

  std::int32_t u_out_handle = -1, v_out_handle = -1;
  auto register_one = [&](const char* component, const char* field, double* data, int n) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = component;
    fname.field = field;
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    ASSERT_EQ(t->field_handle(ctx, &fname, &fref), FUSE_OK);
    if (std::string(field) == "u_out") u_out_handle = fref.handle;
    if (std::string(field) == "v_out") v_out_handle = fref.handle;
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

  // Cold start: period 0's
  // very first ExternalDelivery suspension is hit before any step at
  // all -- resolve_regions()'s own "ordinary prime" pass (src/runtime/
  // iteration.cpp, the chain-execute loop keyed off each window member's
  // own import_channel) runs once, at fuse_initialize(), entirely
  // OUTSIDE any coroutine, and for an ExternalDelivery arc its own
  // execute() case is a no-op -- so unlike an ordinary Transport-delivered
  // channel, which gets ITS first value from that pass (the provider's own
  // registered initial state, moved through a REAL Transport/Remap/Sink),
  // a LowFive-bound import gets NOTHING from it. The suspension itself
  // only happens once the Runner's own coroutine is entered for the first
  // time, which happens on this test's own FIRST fuse_begin_step() call,
  // at the bottom of the loop below -- too late for the application to
  // know ahead of time which component to call first without hitting
  // E-ABI-STEP-ORDER on BOTH ExternalDelivery arcs in some order first.
  // So: deliver each exporter's own INITIAL (pre-any-step) u_out/v_out --
  // constant over the window by construction, nothing has happened yet to
  // vary it -- into the other's import, unconditionally, before the very
  // first fuse_begin_step() of the run. This is the ONE place this harness
  // does something a real H5Dwrite/H5Dread application would also have to
  // do differently from the Transport path: a provider's registered
  // initial value lives in THIS process's own FieldRegistry, which does not
  // exist on another process, so nothing but the application itself can
  // supply period 0's first waveform.
  deliver(t, ctx, "U", "u_out", "V", "v_in");
  deliver(t, ctx, "V", "v_out", "U", "u_in");

  // fuse_step_report::step_opened's own comment (entries.def): on THIS
  // graph, a channel's own ExternalDelivery instruction now stands right
  // after its exporting member's own WindowAdvance (the scheduling-order
  // fix, OpKind::ExternalDelivery's own comment, include/fuse/ir/op.hpp)
  // -- so the LAST instruction of every sweep is always a pending
  // delivery proxy, never a real advance. That means fuse_end_step()'s
  // OWN resume_period() call, used only to PREFETCH, never actually
  // reaches "the period ran to completion" for this graph: it always
  // finds that trailing proxy first and caches it, so
  // fuse_step_report::next_component == -1 structurally never fires
  // here. The ONLY place a period boundary is ever discovered is
  // fuse_begin_step()'s OWN consuming loop, resuming PAST that cached
  // proxy and finding the period's real end -- reported as
  // step_opened == 0, not -1. `pending_next_component` carries that
  // discovery across the period boundary: set when a rollover is seen,
  // read at the top of the NEXT `p` iteration in place of a freshly
  // asked fuse_is_coupling_ongoing()'s own answer (which would recompute
  // the identical value from the SAME already-rolled-over cached
  // request, `resolve_cached_next_component()`'s own doc, src/abi/
  // participant.cpp -- reusing the already-known answer here just
  // avoids an identical redundant lookup, not a second source of truth).
  // Drive by fuse_is_coupling_ongoing()
  // alone -- the shape the ordinary participants harness and the two mains
  // already have (test/verification/brusselator_participants_test.cpp's
  // own WindowDrivenByNextComponent-style loop; examples/brusselator/
  // participants/{u_main.c,v_main.f90}), not a harness-owned window/period
  // counter: counting periods itself would let this harness run the Runner
  // off its own requested end (the region-aware scan and the
  // end_time-bounded rollover, src/runtime/runner.cpp and
  // src/abi/participant.cpp, prevent that).
  // fuse_is_coupling_ongoing()'s own `ongoing` flag and `period` field are
  // the Runner's OWN bookkeeping, not a second, divergeable copy of it --
  // `period` is read fresh every iteration, even mid-period
  // (resolve_cached_next_component()'s own doc, src/abi/participant.cpp,
  // confirms this is a cache read, no resumption). fuse_step_report::period
  // (entries.def, the SAME counter fuse_ongoing::period reports) is the one
  // place a MATCHED rollover (step_opened=1, but periods_completed still
  // incremented mid-call because the newly rolled-into period's own first
  // request happened to name the SAME component this call asked for) is
  // otherwise invisible: without reading it, this loop would validate
  // windows 0..kPeriods-1 correctly but silently skip the one-window-late
  // bookkeeping a matched rollover causes.
  int window_idx = 0;
  std::int32_t last_period = 0;
  std::int32_t max_iteration_seen = -1;
  int step_count = 0;
  // The sequence of fired fields,
  // one char per delivery_pending==1 report ('u' for u_out, 'v' for v_out)
  // -- checked below for alternation: the harness asserts the fired field
  // alternates u_out/v_out in the order the stream dictates.
  std::string delivery_field_sequence;
  // `fuse_is_coupling_ongoing()`'s own `next_component` is a free,
  // side-effect-free read -- safe for `ongoing`/`period` every
  // iteration, but NOT for addressing: is_delivery_due's own report
  // (src/abi/participant.cpp) deliberately leaves `next_request` reset
  // (not re-primed) so the CALLER's own action on that report (the
  // H5Dwrite/H5Dread the protocol requires) happens before
  // the coroutine resumes any further -- a free read sandwiched in
  // between would resume it early, observing stale, not-yet-delivered
  // data (measured as a window falsely "converging" after one sweep).
  // So: trust the LAST report's own next_component (begin_rep's on an
  // is_delivery_due/mismatch stop, end_rep's after a real step) --
  // -1 means "not yet primed", the ONE case a fresh read is safe
  // (nothing has run yet to go stale).
  std::int32_t next_hint = -1;

  // Reads u_state/v_state and runs the window-boundary comparison for the
  // period this harness has JUST finished -- `new_period` is whatever
  // counter first reports the transition.
  //
  // MEASURED (UCore::step()'s own per-call trace, both UCore instances
  // sharing the SAME call sequence through period 0): call=119
  // (t_end=0x1.47ae147ae147bp-6 = 0.02, the window-0/window-1 boundary
  // exactly) produces state_new0=0x1.a812fd6999c04p+0 (1.6565397627653065)
  // in BOTH the reference's own UCore and the harness's own -- EXACTLY
  // this test's own `ref_u_state[0][0]`. call=120 (t_end~=0.0205, window
  // 1's own FIRST substep) produces state_new0=0x1.a8681633ac936p+0
  // (1.6578382373960046) in BOTH instances too. The harness must read
  // state after call=119, as the REFERENCE's own run_period() does (it
  // returns at call=119 and never runs call=120 for period 0 at all);
  // reading after call=120 would be a read-too-late bug in this harness's
  // own loop, not a numerics bug.
  //
  // Also measured directly around this exact transition:
  // `ongoing.period`, read at the top of the loop BEFORE the step call that
  // performs call=120, still reads 0 (`last_period`) --
  // `is_coupling_ongoing()`'s own cached read does not advance until AFTER
  // the call whose own begin_step() runs prime_next_request()'s rollover,
  // by which point that SAME call's own u_loop_step()/bruss_v_loop_step()
  // wrapper has ALREADY run the new period's first physics substep
  // unconditionally. The `ongoing.period` check below therefore never
  // fires BEFORE call=120 on this graph -- there is no point outside the
  // Runner from which this harness can observe the transition before that
  // extra substep runs. It is kept (harmless, and a real defense for a
  // graph shape where the cached read DOES update in time), but
  // `begin_rep.period`, checked after, does all the real work here;
  // `handle_window_boundary()` is shared between the two checks.
  const auto handle_window_boundary = [&](std::int32_t new_period) {
    ASSERT_LT(window_idx, kPeriods)
        << "more windows completed than requested (period " << new_period << ")";
    // Diagnostic: sweep count
    // actually taken on the ABI-driven side (highest fuse_step_report::
    // iteration seen this window, 0-indexed) vs. the in-process
    // reference's own converged sweep count -- a one-sweep difference
    // in the STOPPING DECISION would explain a tiny value difference by
    // itself, without any arithmetic defect.
    fprintf(stderr, "period %d: harness max sweep index=%d, reference iterations=%d (0-indexed max=%d)\n",
           window_idx, max_iteration_seen, ref_iterations[window_idx], ref_iterations[window_idx] - 1);

    SCOPED_TRACE("period " + std::to_string(window_idx));
    std::vector<double> u_state(static_cast<std::size_t>(nu)), v_state(static_cast<std::size_t>(nv));
    bruss_u_state(uh, u_state.data());
    bruss_v_state(vh, v_state.data());
    ASSERT_EQ(static_cast<int>(u_state.size()), ref_u_state[window_idx].size());
    for (std::size_t i = 0; i < u_state.size(); ++i) {
      EXPECT_EQ(u_state[i], ref_u_state[window_idx][static_cast<int>(i)])
          << "window " << (window_idx + 1) << " U state[" << i << "]: harness=" << u_state[i]
          << " (" << std::hexfloat << u_state[i] << ") ref=" << std::defaultfloat
          << ref_u_state[window_idx][static_cast<int>(i)] << " (" << std::hexfloat
          << ref_u_state[window_idx][static_cast<int>(i)] << ")" << std::defaultfloat;
    }
    ASSERT_EQ(static_cast<int>(v_state.size()), ref_v_state[window_idx].size());
    for (std::size_t i = 0; i < v_state.size(); ++i) {
      EXPECT_EQ(v_state[i], ref_v_state[window_idx][static_cast<int>(i)])
          << "window " << (window_idx + 1) << " V state[" << i << "]: harness=" << v_state[i]
          << " (" << std::hexfloat << v_state[i] << ") ref=" << std::defaultfloat
          << ref_v_state[window_idx][static_cast<int>(i)] << " (" << std::hexfloat
          << ref_v_state[window_idx][static_cast<int>(i)] << ")" << std::defaultfloat;
    }

    double mean_u = 0.0;
    for (double x : u_state) mean_u += x;
    mean_u /= static_cast<double>(nu);
    double mean_v = 0.0;
    for (double x : v_state) mean_v += x;
    mean_v /= static_cast<double>(nv);
    std::snprintf(buf, sizeof(buf), "window=%d mean_u=%a external_total=%a exchange_out=%a\n",
                 window_idx + 1, mean_u, bruss_u_external_total(uh), bruss_u_exchange_out(uh));
    u_lines.emplace_back(buf);
    std::snprintf(buf, sizeof(buf), "window=%d mean_v=%a exchange_in=%a\n",
                 window_idx + 1, mean_v, bruss_v_exchange_in(vh));
    v_lines.emplace_back(buf);

    ++window_idx;
    last_period = new_period;
    max_iteration_seen = -1;
  };

  while (true) {
    fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
    ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK)
        << "step " << step_count;
    if (!ongoing.ongoing) break;

    // Checked BEFORE this iteration's own step call -- see
    // `handle_window_boundary`'s own comment for why this is the one
    // point that can read a just-finished period's own true final state,
    // in the ordinary (unmatched-rollover) case.
    if (ongoing.period != last_period) {
      handle_window_boundary(ongoing.period);
    }

    if (next_hint == -1) next_hint = ongoing.next_component;
    const std::int32_t next = next_hint;
    const bool is_u = (next == u_idx);
    ASSERT_TRUE(is_u || next == v_idx)
        << "step " << step_count << ": next_component resolved to neither U ("
        << u_idx << ") nor V (" << v_idx << "), got " << next;
    fuse_step_request& req = is_u ? u_req : v_req;
    fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
    fuse_step_report begin_rep{.struct_size = sizeof(begin_rep)};
    const fuse_status rc = is_u
        ? u_loop_step(ctx, uh, &req, u_in.data(), u_out.data(), &end_rep, &begin_rep)
        : bruss_v_loop_step(ctx, vh, &req, v_in.data(), v_out.data(), &end_rep, &begin_rep);
    ASSERT_EQ(rc, FUSE_OK)
        << "step " << step_count << " (" << (is_u ? "U" : "V") << "): "
        << (is_u ? bruss_u_last_error(uh) : bruss_v_last_error(vh));
    // fuse_step_report::period's own comment (entries.def): set on EVERY
    // report, whether or not a step opened -- the one signal that catches
    // a MATCHED rollover (step_opened=1, but the request that opened came
    // from a NEW period started mid-call) -- the one case the `ongoing.
    // period` check above, which read BEFORE this call, could not have
    // seen yet. Exactly one period can complete per fuse_begin_step()
    // call (prime_next_request()'s own consuming loop only ever reaches a
    // rolled-over request through a REAL, non-delivery-due suspend, which
    // ends that loop immediately -- src/abi/participant.cpp), so this is
    // `if`, not a draining `while`; guarded by `last_period` so the
    // ordinary case above is never double-handled here.
    if (begin_rep.period != last_period) {
      handle_window_boundary(begin_rep.period);
    }

    // fuse_step_report::step_opened's own comment (entries.def): the one
    // place this loop body must look at the report. On 0, no step opened
    // (u_loop_step()/bruss_v_loop_step() skipped the physics and end_step()
    // entirely, `end_rep` is not meaningful) -- either a rollover that
    // landed on a DIFFERENT component (handled already above, via
    // `period`) or this run's own requested end (ongoing.ongoing will be 0
    // at the top of the next iteration).
    //
    // docs/user/c-abi.md's own `delivery_pending`/`delivery_field`
    // protocol: on a report naming field F, write F if producing it or
    // read F if consuming it, then begin_step() again -- never elsewhere,
    // and never both directions unconditionally (with TWO
    // independently-paced LowFive-bound channels, u_out's and v_out's own
    // deliveries do not happen together, and delivering the wrong one
    // early reads a not-yet-updated, stale value). This IS that report
    // when it fires (step_opened=0 is the ONLY place delivery_pending can
    // ever be 1, entries.def's own comment on the field).
    if (!begin_rep.step_opened) {
      if (begin_rep.delivery_pending) {
        // iteration.cpp's own deliver_external() suspends for BOTH the
        // exporter and the importer of a channel, each its own report
        // naming the SAME source field -- the real-process case needs both (one process
        // writes, the other reads), but this harness drives both
        // components through the SAME loop, so the exporter's own
        // notification and the importer's own notification both land
        // here. Acting on both would copy twice (the second copy
        // finding the destination already equal to the source, the
        // double-delivery this test's own `already_equal` check is
        // for) -- so, matching u_main.c/v_main.f90's own split (one
        // participant only ever sees ITS OWN produce-or-consume role
        // for a given field), act only when THIS report's own address
        // is the CONSUMER. `in->component` (and therefore `is_u`,
        // which reflects who the test asked to STEP, not who the
        // cached delivery proxy names) is never checked by the ABI's
        // own is_delivery_due branch -- the report is whatever the
        // Runner's instruction stream has cached next, regardless of
        // `req.component`. `begin_rep.next_component` IS that address
        // (resolve_cached_next_component() reports the delivery's own
        // req.component directly), so it -- not `is_u` -- decides the
        // action: the producer's own notification is a no-op here
        // (nothing to stage -- this harness's deliver() stands in for
        // both H5Dwrite and H5Dread together, performed once, at the
        // consumer's own turn).
        const bool addressed_to_u = (begin_rep.next_component == u_idx);
        if (begin_rep.delivery_field == u_out_handle) {
          if (!addressed_to_u) {
            deliver(t, ctx, "U", "u_out", "V", "v_in");
            delivery_field_sequence.push_back('u');
          }
        } else if (begin_rep.delivery_field == v_out_handle) {
          if (addressed_to_u) {
            deliver(t, ctx, "V", "v_out", "U", "u_in");
            delivery_field_sequence.push_back('v');
          }
        } else {
          FAIL() << "step " << step_count << ": delivery_field="
                << begin_rep.delivery_field << " names neither u_out ("
                << u_out_handle << ") nor v_out (" << v_out_handle << ")";
        }
      }
      next_hint = begin_rep.next_component;
      ++step_count;
      continue;
    }
    EXPECT_EQ(begin_rep.next_component, -2) << "step " << step_count;

    max_iteration_seen = std::max(max_iteration_seen, end_rep.iteration);
    next_hint = end_rep.next_component;
    ++step_count;
  }
  ASSERT_EQ(window_idx, kPeriods) << "not all requested windows were validated";

  // The fired field alternates u_out/v_out
  // in the order the stream dictates -- both channels' own deliveries
  // appear (never one silently starved), and the sequence is not
  // segregated into one block of each (both a u->v and a v->u
  // transition occur, collapsing consecutive repeats of the SAME field
  // -- expected in pairs, exporter then importer, both hosted here).
  ASSERT_FALSE(delivery_field_sequence.empty())
      << "no delivery_pending==1 report was ever seen";
  std::string collapsed;
  for (char c : delivery_field_sequence) {
    if (collapsed.empty() || collapsed.back() != c) collapsed.push_back(c);
  }
  EXPECT_NE(collapsed.find('u'), std::string::npos)
      << "u_out's own delivery was never reported: " << delivery_field_sequence;
  EXPECT_NE(collapsed.find('v'), std::string::npos)
      << "v_out's own delivery was never reported: " << delivery_field_sequence;
  EXPECT_GT(collapsed.size(), 2u)
      << "fields did not alternate (one block of each only): " << collapsed;

  bruss_u_destroy(uh);
  bruss_v_destroy(vh);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);

  // brusselator_check OK on the harness's own printed halves.
  // Under TempDir(), not the working directory: run from a shell this
  // binary otherwise leaves two logs wherever the caller stands (it left
  // them at the repository root once).
  const std::string stem = std::string(::testing::TempDir()) +
      ::testing::UnitTest::GetInstance()->current_test_info()->name();
  const std::string u_path = stem + "_u.log";
  const std::string v_path = stem + "_v.log";
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

TEST(BrusselatorLowFiveHarness, ExternalDeliveryMatchesInProcessRunnerJacobi) {
  RunExternalDeliveryMatchesInProcessRunner(WaveformIteration::Jacobi);
}

TEST(BrusselatorLowFiveHarness, ExternalDeliveryMatchesInProcessRunnerGaussSeidel) {
  RunExternalDeliveryMatchesInProcessRunner(WaveformIteration::GaussSeidel);
}

namespace {

/// Builds the SAME LowFive graph `build_lowfive_graph_and_weights()` builds,
/// parameterized on waveform iteration so the structural test below can
/// check both (that helper hardcodes Jacobi, matching the main test above).
[[nodiscard]] fuse::ExecutionPlan build_lowfive_plan(WaveformIteration wi) {
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
  (void)v_region;
  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      graph, u_region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations, fuse::AccelKind::None, wi,
      RankSet{0, 1});
  const auto u_out = graph.find_field(handles.u_component, "u_out");
  const auto v_out = graph.find_field(handles.v_component, "v_out");
  EXPECT_TRUE(u_out.has_value());
  EXPECT_TRUE(v_out.has_value());
  const_cast<FieldSpec&>(graph.field(*u_out)).hdf5_path = "brusselator_u.h5/grid/u_out";
  const_cast<FieldSpec&>(graph.field(*v_out)).hdf5_path = "brusselator_v.h5/grid/v_out";
  (void)graph.add_component({.name = "lowfive", .ranks = RankSet{0, 1}, .interval = Rational{1},
                              .kind = fuse::ComponentKind::Coupler,
                              .launch = LaunchSpec{.mode = LaunchMode::LowFive}});
  EXPECT_TRUE(backend.register_default_placements(graph, u_mesh).ok());
  const auto lowered = fuse::lower(graph);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{graph};
  auto sched = fuse::schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  return std::move(sched.plan);
}

/// The SAME graph shape `build_lowfive_plan()` builds, minus the
/// `hdf5_path`/fieldless-coupler additions -- an ordinary (Transport-
/// routed, here same-rank so Transport itself collapses to nothing)
/// channel over the identical two meshes, timing, and waveform
/// iteration, for the positional comparison below.
[[nodiscard]] fuse::ExecutionPlan build_ordinary_plan(WaveformIteration wi) {
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
  (void)v_region;
  (void)brusselator::add_window_coupling(graph, u_region, kDtU, kDtV, kWindow, kTolerance,
                                          kMaxIterations, fuse::AccelKind::None, wi, RankSet{0, 1});
  EXPECT_TRUE(backend.register_default_placements(graph, u_mesh).ok());
  const auto lowered = fuse::lower(graph);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{graph};
  auto sched = fuse::schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  return std::move(sched.plan);
}

/// Just the `OpKind` sequence, for a structural comparison that does not
/// also pin arc numbering (which the byte-exact golden IR already covers).
[[nodiscard]] std::vector<fuse::OpKind> kinds_of(const fuse::ExecutionPlan& plan) {
  std::vector<fuse::OpKind> out;
  out.reserve(plan.instructions().size());
  for (const auto& ins : plan.instructions()) out.push_back(ins.kind);
  return out;
}

/// Walks a single channel's own DATA-FLOW chain forward from `start`
/// (matched by `out`/`in[0]` arc identity, the same edge the scheduler
/// itself orders by), collecting each instruction's own `kind` until no
/// instruction reads `out` as its own `in[0]` (the chain's own end,
/// normally a Sink). This is the "positional, kind-for-kind" comparison
/// this test's own name promises: it checks what a chain IS, in the
/// order its OWN data actually flows, not where the scheduler happened
/// to place it in the global instruction stream -- the Sink/WindowAdvance
/// GROUPING across channels is a scheduler tie-break this test's own
/// header comment already disclaims as not a correctness question (and
/// genuinely differs between Jacobi and GaussSeidel for the SAME LowFive
/// graph, without either chain's own internal shape changing at all).
[[nodiscard]] std::vector<fuse::OpKind> chain_from(const fuse::ExecutionPlan& plan,
                                                    std::size_t start_index) {
  std::vector<fuse::OpKind> out;
  const auto& stream = plan.instructions();
  std::size_t i = start_index;
  std::unordered_set<std::size_t> visited;
  while (true) {
    out.push_back(stream[i].kind);
    if (!visited.insert(i).second) break;  // defensive: no cycle expected
    const fuse::ArcId produced = stream[i].out;
    std::optional<std::size_t> next;
    for (std::size_t j = 0; j < stream.size(); ++j) {
      if (j == i) continue;
      if (stream[j].n_in > 0 && stream[j].in[0].value() == produced.value()) {
        next = j;
        break;
      }
    }
    if (!next.has_value()) break;
    i = *next;
  }
  return out;
}

/// Every chain in `plan` that starts with an instruction of `start_kind`,
/// each followed to its own end via `chain_from()`.
[[nodiscard]] std::vector<std::vector<fuse::OpKind>> chains_starting_with(
    const fuse::ExecutionPlan& plan, fuse::OpKind start_kind) {
  std::vector<std::vector<fuse::OpKind>> out;
  const auto& stream = plan.instructions();
  for (std::size_t i = 0; i < stream.size(); ++i) {
    if (stream[i].kind == start_kind) out.push_back(chain_from(plan, i));
  }
  return out;
}

}  // namespace

/// A LowFive-bound channel must still get Constrain/Diagnose --
/// docs/user/lowfive.md promises "Interpolate/Remap/Constrain if the two sides' meshes
/// differ... is what FUSE does for a LowFive-bound channel instead".
/// `insert_derived_ops()`'s `channel_chain()` walks forward from the
/// channel's `chain_entry_arc` (LoweringContext, rules_channel.cpp,
/// rules_derived.cpp); walking from the channel's ORIGINAL `source_arc`
/// would be unreachable from a zero-input `ExternalDelivery` root, so
/// `has_remap` would come back false and both ops would go unemitted for
/// every LowFive-bound channel, silently.
///
/// Measured: the ordinary (Transport) graph's own stream for
/// this identity-remap shape is, per channel, contiguous:
/// WindowAdvance->Remap->Constrain->Diagnose->Sink (x2 channels), then
/// Residual x2 -- 12 instructions, Transport itself absent (same-rank
/// channel, collapses to nothing). The LowFive graph's stream has each
/// channel's own ExternalDelivery->Remap->Constrain->Diagnose chain
/// contiguous (x2; Transport, here nothing, is replaced by one
/// ExternalDelivery), but BOTH channels' Sink
/// grouped together, then BOTH WindowAdvance together, then Residual x2 --
/// 14 instructions. The Sink/WindowAdvance grouping differs from the
/// ordinary graph's own interleaving (a scheduler tie-break over nodes with
/// no edges forcing a specific slot -- WindowAdvance is a dead end in both
/// graphs' DAGs once it no longer feeds Remap directly); it is not a
/// correctness question for this test, which is only about which ops
/// appear. Identical for both waveform iterations (checked below) -- this
/// shape has no sweep-order dependence.
TEST(BrusselatorLowFiveHarness, ChannelGetsConstrainAndDiagnoseLikeAnOrdinaryChannel) {
  for (const WaveformIteration wi : {WaveformIteration::Jacobi, WaveformIteration::GaussSeidel}) {
    SCOPED_TRACE(wi == WaveformIteration::Jacobi ? "Jacobi" : "GaussSeidel");
    const fuse::ExecutionPlan lowfive_plan = build_lowfive_plan(wi);
    const fuse::ExecutionPlan ord_plan = build_ordinary_plan(wi);

    // The positional, kind-for-kind comparison this test's own name
    // promises: walk each channel's own DATA-FLOW chain forward from its
    // own WindowAdvance (the exporting member's own tick) via arc
    // identity (`chain_from()`'s own comment has the full account), NOT
    // via the global instruction stream's own index. That global index --
    // which Sink/WindowAdvance group the scheduler happens to place first
    // -- genuinely differs between Jacobi and GaussSeidel for the SAME
    // LowFive graph, even though neither chain's own internal shape
    // changes at all; a single hardcoded `expected` vector for both would
    // pin that scheduler tie-break as if it were a correctness
    // invariant.
    const std::vector<std::vector<fuse::OpKind>> lowfive_chains =
        chains_starting_with(lowfive_plan, fuse::OpKind::WindowAdvance);
    const std::vector<std::vector<fuse::OpKind>> ord_chains =
        chains_starting_with(ord_plan, fuse::OpKind::WindowAdvance);
    ASSERT_EQ(lowfive_chains.size(), 2u) << "one chain per channel direction";
    ASSERT_EQ(ord_chains.size(), 2u) << "one chain per channel direction";
    for (std::size_t c = 0; c < 2; ++c) {
      SCOPED_TRACE("channel " + std::to_string(c));
      // "ExternalDelivery standing where Transport stands" (this test's
      // own header comment): a LowFive-bound channel's own chain is the
      // ordinary channel's own chain with exactly one ExternalDelivery
      // inserted right after WindowAdvance -- precisely the slot a
      // cross-rank Transport instruction would occupy, collapsed to
      // nothing here only because this graph is same-rank.
      std::vector<fuse::OpKind> lowfive_stripped;
      for (const fuse::OpKind k : lowfive_chains[c]) {
        if (k != fuse::OpKind::ExternalDelivery) lowfive_stripped.push_back(k);
      }
      EXPECT_EQ(lowfive_stripped, ord_chains[c]);
      ASSERT_GE(lowfive_chains[c].size(), 2u);
      EXPECT_EQ(lowfive_chains[c][0], fuse::OpKind::WindowAdvance);
      EXPECT_EQ(lowfive_chains[c][1], fuse::OpKind::ExternalDelivery)
          << "ExternalDelivery must stand immediately after WindowAdvance, "
             "where Transport would";
      EXPECT_EQ(ord_chains[c].front(), fuse::OpKind::WindowAdvance);
      EXPECT_EQ(ord_chains[c].back(), fuse::OpKind::Sink);
      // Finding C's own claim: a LowFive-bound channel still gets
      // Constrain/Diagnose, exactly like the ordinary channel's own
      // chain -- checked here, within the chain, not just by a global
      // count (which the loop below also does, redundantly on purpose).
      for (const fuse::OpKind must_have :
           {fuse::OpKind::Remap, fuse::OpKind::Constrain, fuse::OpKind::Diagnose, fuse::OpKind::Sink}) {
        EXPECT_NE(std::find(lowfive_chains[c].begin(), lowfive_chains[c].end(), must_have),
                  lowfive_chains[c].end())
            << "kind " << fuse::to_cstring(must_have) << " missing from the LowFive chain";
      }
    }

    // Global counts: redundant with the per-chain checks above for
    // anything a chain covers, but also the one place Residual (which
    // never sits inside either chain -- it reads BOTH channels' own
    // post-convergence values, not a single producer's output) and the
    // absence of Transport (same-rank channel, collapses to nothing in
    // EITHER graph) get checked.
    const std::vector<fuse::OpKind> lowfive_kinds = kinds_of(lowfive_plan);
    const std::vector<fuse::OpKind> ord_kinds = kinds_of(ord_plan);
    EXPECT_EQ(std::count(lowfive_kinds.begin(), lowfive_kinds.end(), fuse::OpKind::Transport), 0);
    EXPECT_EQ(std::count(ord_kinds.begin(), ord_kinds.end(), fuse::OpKind::Transport), 0);
    EXPECT_EQ(std::count(lowfive_kinds.begin(), lowfive_kinds.end(), fuse::OpKind::ExternalDelivery), 2);
    for (const fuse::OpKind k : {fuse::OpKind::Remap, fuse::OpKind::Constrain, fuse::OpKind::Diagnose,
                                  fuse::OpKind::Sink, fuse::OpKind::WindowAdvance, fuse::OpKind::Residual}) {
      EXPECT_EQ(std::count(lowfive_kinds.begin(), lowfive_kinds.end(), k),
                std::count(ord_kinds.begin(), ord_kinds.end(), k))
          << "kind " << fuse::to_cstring(k);
    }
  }
}




