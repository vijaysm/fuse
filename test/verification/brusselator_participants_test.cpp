// Time windows track (Conformance, Unit 2): the tier-0 harness the lead
// specified -- drives the SAME two step functions `examples/brusselator/
// participants/u_main.c` and `v_main.f90` call (the extern "C" shim,
// shim.h/shim.cpp) through the participant ABI (fuse_begin_step()/
// fuse_end_step()), in ONE process, with EXPECT_EQ against Unit 1's own
// in-process fuse::Runner run.
//
// THE GRAPH IS UNIT 1's GRAPH, plus only add_coupler()/remap_on() (the
// lead's own ruling): brusselator_graph.hpp's `add_window_coupling`,
// called with `coupler_ranks` set. Both transfer fields stay on
// `u_region` (species_component.hpp's own file header explains why), so
// the coupler-routed Remap this compiles is the IDENTITY.
//
// This is the FIRST test anywhere in the suite to drive TWO externally-
// driven window members (U and V both suspended at their own
// WindowAdvance, stepped by this test's own loop via the SAME shim
// functions u_main.c/v_main.f90 call) through the SAME coupler-routed,
// EXPLICIT-interval window region together.
// AbiParticipant.ExternallyDrivenWindowMemberMatchesInProcessRunner
// (test/abi/participant_test.cpp) proved the single-member case.
//
// History (left here because it explains why this file still uses the
// degenerate window == dt_u == dt_v case, kDtU=kDtV=kWindow=1/500,
// rather than Unit 1's own dt_u != dt_v != window shape): this test was
// originally DISABLED on two findings, both reported to the lead rather
// than guessed past. Finding A -- is_coupling_ongoing()'s own period/now
// never advanced, and U's own state staircased through several window
// instances before settling -- was a real library defect, not a test
// bug: ClockSet's own aggregates (common_period/finest_interval/
// checked_common_period/is_lockstep/earliest_next/due_at) folded a
// coupler's default interval (Rational{1}, since it declares none of
// its own) into the lcm/gcd like any physics component's real interval,
// silently inflating this graph's own compiled period to lcm(1/50, 1) =
// 1 -- 50 window instances per period instead of one.
// include/fuse/schedule/clock.hpp's own ClockSet constructor comment has
// the full account; fixed there (Conformance Unit 2, Defect 1), covered
// by test/schedule/clock_test.cpp's ClockSet.ACouplersDefaultInterval
// NeverEntersTheLcmOrGcd and test/schedule/window_region_golden_test.cpp's
// WindowRegionGolden.ACouplerDoesNotEnterTheWindowPeriod. RE-MEASURED
// after that fix: this test now PASSES outright with no further change
// needed here -- is_coupling_ongoing()'s period/now track correctly once
// the period itself is right, so there is no second defect in
// impl_is_coupling_ongoing.
//
// Finding B (window spans more than one of a member's own substeps) was
// a SEPARATE defect from Finding A: strict U-then-V alternation once
// per Picard sweep gets E-ABI-STEP-ORDER (component asked out of turn)
// the moment a window spans more than one of a member's own substeps,
// because there is then no single fixed alternation that is right for
// every sweep. FIXED as Defect 3 by growing the ABI: `next_component`
// appended to fuse_step_report/fuse_ongoing (entries.def's own comment
// on that field has the full account), resolved for free inside
// fuse_end_step()'s own pre-existing resume_period() call -- no new
// cost, no retry loop. ExternallyDrivenMultiSubstepWindowDrivenByNext
// Component below is the non-degenerate case this mechanism was built
// for: window == 4*dt_u == 4*dt_v, so each Picard sweep takes four U
// ticks and four V ticks, in whatever order the Runner's own schedule
// actually interleaves them -- driven with ZERO guessing past the
// very first call of each period (declaration order breaks that one
// tie, same convention test/abi/participant_test.cpp's own
// StepBoundsAreTheSchedulesRationals relies on).
#include <gtest/gtest.h>

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
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

#include "brusselator_graph.hpp"
#include "shim.h"
#include "species_component.hpp"

namespace {

using fuse::ComponentSet;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::RankSet;
using fuse::Result;
using fuse::WaveformIteration;

// Mutually non-nested grids (brusselator_graph.hpp's own convention) --
// this file's job is the structural claim (does the ABI-driven run
// match Unit 1's own in-process Runner, bit for bit, on the IDENTICAL
// graph) at a size that runs in milliseconds, not the physics scale
// examples/brusselator/main.cpp's own ctest rows already cover.
constexpr int kNxU = 3, kNyU = 3, kNxV = 4, kNyV = 4;

// dt_u == dt_v == window: the degenerate, one-substep-per-window case --
// this file's own header, Finding A, has the full account of why even
// this simplest case is not yet understood for two simultaneous
// externally-driven window members.
const Rational kDtU{1, 500}, kDtV{1, 500}, kWindow{1, 500};
constexpr double kTolerance = 1e-10;
constexpr int kMaxIterations = 60;
constexpr int kPeriods = 2;

/// Everything `main.cpp` builds once, shared by the reference Runner and
/// (for the ABI-driven side) `bruss_v_create`'s own internal copy --
/// backend, NativeGrids, the graph with U/V/coupler, and the two
/// `RemapWeights` `VComponent`/`VParticipant` both need. Built fresh for
/// EACH side (the ABI's own `fuse_ctx` builds its own internal backend
/// from the serialized graph; `bruss_v_create` builds its own again,
/// species_core.hpp's own `VParticipant::make` comment explains why) so
/// neither side's backend/weights are shared with the other -- only the
/// SERIALIZED GRAPH BYTES are, through `fuse::serialize`.
struct GraphAndWeights {
  fuse::native::NativeMeshBackend backend;
  brusselator::NativeGrid u_grid, v_grid;
  fuse::CouplingGraph graph;
  fuse::MeshId u_mesh, v_mesh;
  brusselator::WindowGraphHandles handles;
  fuse::RemapWeights weights_vu, weights_uv;  // V->U, U->V (main.cpp's own names)
};

[[nodiscard]] GraphAndWeights build_graph_and_weights(
    Rational dt_u = kDtU, Rational dt_v = kDtV, Rational window = kWindow) {
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
      graph, u_region, dt_u, dt_v, window, kTolerance, kMaxIterations,
      fuse::AccelKind::None, WaveformIteration::Jacobi, RankSet{0, 1});

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

}  // namespace

TEST(BrusselatorParticipants, ExternallyDrivenBothMembersMatchInProcessRunner) {
  const brusselator::PhysicsConfig cfg;  // defaults: diffusion=10, A=3.4, B=1, forcing=true

  // --- Reference: fuse::Runner driven directly, Unit 1's own graph
  // (plus the coupler), Unit 1's own UComponent/VComponent. ---
  GraphAndWeights ref = build_graph_and_weights();
  // ONE call, on u_mesh only -- brusselator_runner_test.cpp's own
  // make_rig()+run_via_runner() precedent (this session's earlier item 2
  // work): every component's placement on a tier-0 backend is the same
  // deterministic (rank_, num_ranks_) view regardless of which mesh asks,
  // so registering against u_mesh alone already covers every component
  // this test's graph has.
  ASSERT_TRUE(ref.backend.register_default_placements(ref.graph, ref.u_mesh).ok());

  const auto ref_lowered = fuse::lower(ref.graph);
  ASSERT_TRUE(ref_lowered.ok()) << ref_lowered.diags.to_string();
  const fuse::ClockSet ref_clocks{ref.graph};
  const auto ref_sched = fuse::schedule(ref.graph, ref_lowered.dag, ref_clocks,
                                        fuse::make_region_sizes(ref.graph, ref.backend));
  ASSERT_TRUE(ref_sched.ok()) << ref_sched.diags.to_string();

  // V built first, U seeded from V's own initial `provided` value remapped
  // onto the U grid (not zero) -- make_rig()'s own precedent, so the
  // reference's first reaction sub-step sees the same v-on-u value the
  // ABI-driven side's bruss_u_create() below is seeded with too.
  auto ref_comp_v = std::make_unique<brusselator::VComponent>(
      "V", ref.u_grid, ref.v_grid, cfg, &ref.weights_uv, &ref.weights_vu);
  const brusselator::Vector ref_v0_on_u =
      Eigen::Map<brusselator::Vector>(ref_comp_v->provided_ptr(), ref.u_grid.size());
  auto ref_comp_u = std::make_unique<brusselator::UComponent>("U", ref.u_grid, cfg, ref_v0_on_u);

  ComponentSet ref_components;
  ref_components.add(std::move(ref_comp_u));
  ref_components.add(std::move(ref_comp_v));
  FieldRegistry ref_registry;
  for (auto& c : ref_components) ASSERT_TRUE(c.register_fields(ref_registry, ref.graph).ok());

  fuse::Runner ref_runner{ref.graph, ref_lowered.dag, ref_sched.plan, ref.backend};
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

  // --- Same graph, driven entirely through the ABI: BOTH U and V
  // externally driven, stepped by the SAME shim functions u_main.c/
  // v_main.f90 call -- the Unit 2 shape in miniature, in one process.
  GraphAndWeights abi_setup = build_graph_and_weights();
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

  // u_out/v_out must hold their REAL initial (cold-start) values BEFORE
  // t->initialize() runs, NOT zero: initialize()'s own cold-start
  // priming (iteration.cpp, the per-window-member "before window 0 ever
  // runs" seeding) reads each provider's CURRENTLY REGISTERED value
  // through `registry.get()` -- for the reference side this is already
  // right because UComponent's/VComponent's own constructors compute a
  // real initial `provided_` before register_fields() ever binds it. An
  // externally-driven participant has no such constructor running
  // before initialize(), so THIS caller must supply the same value by
  // hand -- throwaway UComponent/VComponent instances, exactly the
  // physics bruss_u_create()/bruss_v_create() below will reconstruct
  // internally for the real run, read only for their constructors' own
  // initial provided_ptr(). VERIFIED: leaving u_out/v_out at the zero-
  // initialized default (an earlier version of this setup did) produced
  // a much larger, unconditional divergence from the very first sweep;
  // this fix removed that specific divergence (confirmed by comparing
  // traces with and without it).
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
  // v_out/v_in live on u_region too (Unit 1's own graph shape) -- length
  // nu, NOT nv.
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

  // The SAME extern "C" shim both standalone mains call -- stands in for
  // u_main.c's/v_main.f90's own time loop here, in process. bruss_u_create/
  // bruss_v_create recompute the SAME initial state abi_u_seed/abi_v_seed
  // already did above (deterministically, from the same inputs) -- the
  // seed objects were only ever needed for their constructors' own
  // initial provided_ptr(), already copied into u_out/v_out/v0_on_u.
  bruss_u_handle* uh = bruss_u_create(kNxU, kNyU, cfg.diffusion, cfg.A, cfg.B,
                                     cfg.forcing ? 1 : 0, v0_on_u.data());
  ASSERT_NE(uh, nullptr);
  bruss_v_handle* vh = bruss_v_create(kNxU, kNyU, kNxV, kNyV, cfg.diffusion);
  ASSERT_NE(vh, nullptr);
  ASSERT_EQ(bruss_v_size(vh), nu);

  fuse_step_request u_req{.struct_size = sizeof(u_req)};
  u_req.component = "U";
  fuse_step_request v_req{.struct_size = sizeof(v_req)};
  v_req.component = "V";

  // ImplicitCoupling's own established call pattern (test/abi/
  // participant_test.cpp): U then V, once per sweep, for
  // ref_iterations[p] sweeps -- exactly one substep per window (this
  // file's own header explains why: window == dt_u == dt_v here, the
  // degenerate case Finding B's own E-ABI-STEP-ORDER does not reach).
  for (int p = 0; p < kPeriods; ++p) {
    for (int sweep = 0; sweep < ref_iterations[p]; ++sweep) {
      fuse_step_report u_rep{.struct_size = sizeof(u_rep)};
      ASSERT_EQ(t->begin_step(ctx, &u_req, &u_rep), FUSE_OK) << "period " << p << " sweep " << sweep;
      // requires_checkpoint/requires_restore (KNOWN_ISSUES.md D12): an
      // externally-driven window member's own integrator state is
      // opaque to the Runner -- it cannot roll U/V back to the
      // window-start checkpoint itself the way it does for an
      // in-process Component's own save_state()/restore_state(), so it
      // tells THIS caller when to do it instead.
      if (u_rep.requires_checkpoint) bruss_u_save_state(uh);
      if (u_rep.requires_restore) bruss_u_restore_state(uh);
      // Exact rationals straight through to the shim -- not two doubles:
      // shim.h's own comment on bruss_u_step() has the full account of
      // why `t_end - t_begin` as a double subtraction is not always
      // bit-identical to the exact rational difference.
      ASSERT_EQ(bruss_u_step(uh, u_rep.t_begin.numerator, u_rep.t_begin.denominator,
                             u_rep.t_end.numerator, u_rep.t_end.denominator,
                             u_in.data(), u_out.data()), 0)
          << bruss_u_last_error(uh);
      ASSERT_EQ(t->end_step(ctx, &u_req, &u_rep), FUSE_OK) << "period " << p << " sweep " << sweep;

      fuse_step_report v_rep{.struct_size = sizeof(v_rep)};
      ASSERT_EQ(t->begin_step(ctx, &v_req, &v_rep), FUSE_OK) << "period " << p << " sweep " << sweep;
      if (v_rep.requires_checkpoint) bruss_v_save_state(vh);
      if (v_rep.requires_restore) bruss_v_restore_state(vh);
      // Exact rationals straight through, exactly like bruss_u_step above
      // -- shim.h's own comment on bruss_v_step has the full account.
      ASSERT_EQ(bruss_v_step(vh, v_rep.t_begin.numerator, v_rep.t_begin.denominator,
                             v_rep.t_end.numerator, v_rep.t_end.denominator,
                             v_in.data(), v_out.data()), 0)
          << bruss_v_last_error(vh);
      ASSERT_EQ(t->end_step(ctx, &v_req, &v_rep), FUSE_OK) << "period " << p << " sweep " << sweep;
    }
    fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
    ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK);
    EXPECT_EQ(ongoing.period, p + 1) << "period " << p;

    SCOPED_TRACE("period " + std::to_string(p));
    std::vector<double> u_state(static_cast<std::size_t>(nu)), v_state(static_cast<std::size_t>(nv));
    bruss_u_state(uh, u_state.data());
    bruss_v_state(vh, v_state.data());
    ASSERT_EQ(static_cast<int>(u_state.size()), ref_u_state[p].size());
    for (std::size_t i = 0; i < u_state.size(); ++i) {
      EXPECT_EQ(u_state[i], ref_u_state[p][static_cast<int>(i)]) << "U state[" << i << "]";
    }
    ASSERT_EQ(static_cast<int>(v_state.size()), ref_v_state[p].size());
    for (std::size_t i = 0; i < v_state.size(); ++i) {
      EXPECT_EQ(v_state[i], ref_v_state[p][static_cast<int>(i)]) << "V state[" << i << "]";
    }
  }

  bruss_u_destroy(uh);
  bruss_v_destroy(vh);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}

// BrusselatorParticipants.ExternallyDrivenMultiSubstepWindowDrivenByNextComponent
// (Defect 3). window == 4*dt_u == 4*dt_v: Finding B's own non-degenerate
// shape, where a single fixed U-then-V alternation is wrong (each Picard
// sweep now takes four ticks of EACH member, interleaved by the Runner's
// own schedule, not by this test's guess). Driven purely by
// fuse_step_report::next_component/fuse_ongoing::next_component after the
// one declaration-order tie at the very start of each period -- no
// E-ABI-STEP-ORDER retry anywhere in this loop.
TEST(BrusselatorParticipants, ExternallyDrivenMultiSubstepWindowDrivenByNextComponent) {
  const brusselator::PhysicsConfig cfg;
  const Rational kFineDtU{1, 2000}, kFineDtV{1, 2000}, kFineWindow{1, 500};  // window = 4*dt_u = 4*dt_v
  constexpr int kFinePeriods = 2;

  // --- Reference: fuse::Runner driven directly. ---
  GraphAndWeights ref = build_graph_and_weights(kFineDtU, kFineDtV, kFineWindow);
  ASSERT_TRUE(ref.backend.register_default_placements(ref.graph, ref.u_mesh).ok());
  const auto ref_lowered = fuse::lower(ref.graph);
  ASSERT_TRUE(ref_lowered.ok()) << ref_lowered.diags.to_string();
  const fuse::ClockSet ref_clocks{ref.graph};
  const auto ref_sched = fuse::schedule(ref.graph, ref_lowered.dag, ref_clocks,
                                        fuse::make_region_sizes(ref.graph, ref.backend));
  ASSERT_TRUE(ref_sched.ok()) << ref_sched.diags.to_string();

  auto ref_comp_v = std::make_unique<brusselator::VComponent>(
      "V", ref.u_grid, ref.v_grid, cfg, &ref.weights_uv, &ref.weights_vu);
  const brusselator::Vector ref_v0_on_u =
      Eigen::Map<brusselator::Vector>(ref_comp_v->provided_ptr(), ref.u_grid.size());
  auto ref_comp_u = std::make_unique<brusselator::UComponent>("U", ref.u_grid, cfg, ref_v0_on_u);

  ComponentSet ref_components;
  ref_components.add(std::move(ref_comp_u));
  ref_components.add(std::move(ref_comp_v));
  FieldRegistry ref_registry;
  for (auto& c : ref_components) ASSERT_TRUE(c.register_fields(ref_registry, ref.graph).ok());

  fuse::Runner ref_runner{ref.graph, ref_lowered.dag, ref_sched.plan, ref.backend};
  fuse::RunnerConfig ref_rcfg;
  {
    const auto init_r = ref_runner.initialize(ref_components, ref_registry, ref_rcfg);
    ASSERT_TRUE(init_r.ok()) << init_r.error().code << ": " << init_r.error().message;
  }
  std::vector<brusselator::Vector> ref_u_state(kFinePeriods), ref_v_state(kFinePeriods);
  for (int p = 0; p < kFinePeriods; ++p) {
    const auto ran = ref_runner.run_period(ref_components);
    ASSERT_TRUE(ran.ok()) << ran.error().code << ": " << ran.error().message << " (period " << p << ")";
    ASSERT_TRUE(ref_runner.iteration_stats().front().converged)
        << "reference run must converge for this comparison to mean anything (period " << p << ")";
    ref_u_state[p] = static_cast<brusselator::UComponent*>(ref_components.find("U"))->state();
    ref_v_state[p] = static_cast<brusselator::VComponent*>(ref_components.find("V"))->state();
  }

  // --- Same graph, driven entirely through the ABI, turn order resolved
  // via next_component. ---
  GraphAndWeights abi_setup = build_graph_and_weights(kFineDtU, kFineDtV, kFineWindow);
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
                                     cfg.forcing ? 1 : 0, v0_on_u.data());
  ASSERT_NE(uh, nullptr);
  bruss_v_handle* vh = bruss_v_create(kNxU, kNyU, kNxV, kNyV, cfg.diffusion);
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

  for (int p = 0; p < kFinePeriods; ++p) {
    // The one tie this loop ever guesses: both members' very first tick
    // of a period starts at the same instant, and nothing is cached yet
    // for fuse_is_coupling_ongoing() to report for free (it stays
    // side-effect-free by design -- see its own comment in
    // src/abi/participant.cpp). U was declared first, so U opens, same
    // convention test/abi/participant_test.cpp's own
    // StepBoundsAreTheSchedulesRationals relies on.
    std::int32_t next = u_idx;
    int step_count = 0;
    while (true) {
      const bool is_u = (next == u_idx);
      ASSERT_TRUE(is_u || next == v_idx)
          << "period " << p << " step " << step_count
          << ": next_component resolved to neither U (" << u_idx
          << ") nor V (" << v_idx << "), got " << next;
      fuse_step_request& req = is_u ? u_req : v_req;
      fuse_step_report rep{.struct_size = sizeof(rep)};
      ASSERT_EQ(t->begin_step(ctx, &req, &rep), FUSE_OK)
          << "period " << p << " step " << step_count << " (" << (is_u ? "U" : "V") << ")";
      EXPECT_EQ(rep.next_component, -2)
          << "begin_step() cannot resolve next_component synchronously (its own comment "
             "in entries.def explains why) -- period " << p << " step " << step_count;
      if (is_u) {
        if (rep.requires_checkpoint) bruss_u_save_state(uh);
        if (rep.requires_restore) bruss_u_restore_state(uh);
        ASSERT_EQ(bruss_u_step(uh, rep.t_begin.numerator, rep.t_begin.denominator,
                               rep.t_end.numerator, rep.t_end.denominator,
                               u_in.data(), u_out.data()), 0)
            << bruss_u_last_error(uh);
      } else {
        if (rep.requires_checkpoint) bruss_v_save_state(vh);
        if (rep.requires_restore) bruss_v_restore_state(vh);
        ASSERT_EQ(bruss_v_step(vh, rep.t_begin.numerator, rep.t_begin.denominator,
                               rep.t_end.numerator, rep.t_end.denominator,
                               v_in.data(), v_out.data()), 0)
            << bruss_v_last_error(vh);
      }
      fuse_step_report end_rep{.struct_size = sizeof(end_rep)};
      ASSERT_EQ(t->end_step(ctx, &req, &end_rep), FUSE_OK)
          << "period " << p << " step " << step_count << " (" << (is_u ? "U" : "V") << ")";
      ++step_count;
      if (end_rep.next_component == -1) break;  // period complete
      next = end_rep.next_component;
    }

    fuse_ongoing ongoing{.struct_size = sizeof(ongoing)};
    ASSERT_EQ(t->is_coupling_ongoing(ctx, nullptr, &ongoing), FUSE_OK);
    EXPECT_EQ(ongoing.period, p + 1) << "period " << p;
    // is_coupling_ongoing() never calls resume_period() itself (see its
    // own comment in src/abi/participant.cpp), and the LAST end_step()
    // above reset c.next_request on closing the period -- so this read
    // is -2 ("undetermined without stepping"), not -1: with nothing
    // cached, "the period just closed" and "the next period has not
    // been primed yet" are the same state from here, and only -1 means
    // specifically "this graph has nothing externally driven at all".
    EXPECT_EQ(ongoing.next_component, -2) << "period " << p << " just closed";

    SCOPED_TRACE("period " + std::to_string(p));
    std::vector<double> u_state(static_cast<std::size_t>(nu)), v_state(static_cast<std::size_t>(nv));
    bruss_u_state(uh, u_state.data());
    bruss_v_state(vh, v_state.data());
    ASSERT_EQ(static_cast<int>(u_state.size()), ref_u_state[p].size());
    for (std::size_t i = 0; i < u_state.size(); ++i) {
      EXPECT_EQ(u_state[i], ref_u_state[p][static_cast<int>(i)]) << "U state[" << i << "]";
    }
    ASSERT_EQ(static_cast<int>(v_state.size()), ref_v_state[p].size());
    for (std::size_t i = 0; i < v_state.size(); ++i) {
      EXPECT_EQ(v_state[i], ref_v_state[p][static_cast<int>(i)]) << "V state[" << i << "]";
    }
  }

  bruss_u_destroy(uh);
  bruss_v_destroy(vh);
  EXPECT_EQ(t->destroy(ctx, nullptr, nullptr), FUSE_OK);
}
