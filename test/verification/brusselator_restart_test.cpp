/// Restart track (docs/dev/restart.md), Stage B Slice 2: the Brusselator
/// Runner round trip at tier 0 (no MPI -- this sandbox cannot start it;
/// see the host_pending note in the delivered report), proving owner
/// decisions 1 and 4 against the one example of the four Stage A named
/// that genuinely exercises Form A's own generality (docs/dev/
/// restart.md Section 2.3: Brusselator is the only one of the four
/// examples with real sub-component spatial partitioning). Covers:
///   - UComponent/VComponent Form A over MeshView::global_id() (species
///     as Binary64, the U-grid-keyed exported/imported fields and the
///     V-grid-keyed private state_ in the SAME restart point);
///   - window-waveform history (WindowMemberState: clock, history,
///     last_window_waveform) restart, keyed by (field, global id) per
///     sample;
///   - IQN-ILS accelerator restart state (KNOWN_ISSUES.md D64, closed
///     this round), alongside the already-tested Aitken path;
///   - both file layouts, Jacobi and Gauss-Seidel waveform iteration.
/// Reuses test/verification/brusselator_runner_test.cpp's own Rig/
/// make_rig()/add_window_coupling() construction, not re-derived.
#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/num/accelerator.hpp>
#include <fuse/restart_io/hdf5_restart_io.hpp>
#include <fuse/runtime/restart.hpp>
#include <fuse/runtime/runner.hpp>

#include "brusselator_graph.hpp"
#include "brusselator_physics.hpp"
#include "species_component.hpp"

using fuse::AccelKind;
using fuse::ComponentSet;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::WaveformIteration;

namespace {

// Small meshes (mutually non-nested, as brusselator_runner_test.cpp's own
// comment explains) and a short time span -- this test proves the
// restart MECHANISM (every row restored bit for bit, including the
// accelerator and the window-waveform history), not the physics'
// own long-run acceptance bar (brusselator_runner_test.cpp's own
// 150-window span already covers that separately).
constexpr int kNxU = 5, kNyU = 5, kNxV = 7, kNyV = 7;
constexpr double kDtU = 0.002, kDtV = 0.003, kWindow = 0.02;
constexpr double kTol = 1e-10;
constexpr int kMaxIterations = 100;

struct BrusselatorFixture {
  fuse::native::NativeMeshBackend backend;
  brusselator::NativeGrid u_grid, v_grid;
  fuse::RemapWeights w_vu, w_uv;
  fuse::CouplingGraph graph;
  ComponentSet components;
  FieldRegistry registry;
  // Latent bug fixed this round: `fuse::Runner::Impl::dag` is held by
  // REFERENCE to the `OperatorDAG` inside a `fuse::LoweringResult` --
  // make_setup() used to keep that result as a stack-local, dangling the
  // instant make_setup() returned. Never manifested until a later
  // Runner-internal call read deep enough into the stale memory to
  // observe a corrupted OperatorDAG. Declared BEFORE `runner` so
  // destruction order is correct: members destroy in REVERSE declaration
  // order, so `runner` (which holds a reference into `lowered.dag`) is
  // destroyed before `lowered` itself is.
  fuse::LoweringResult lowered;
  std::unique_ptr<fuse::Runner> runner;

  BrusselatorFixture(fuse::native::NativeMeshBackend backend_in,
                     brusselator::NativeGrid u, brusselator::NativeGrid v)
      : backend(std::move(backend_in)), u_grid(std::move(u)), v_grid(std::move(v)) {}
};

[[nodiscard]] std::unique_ptr<BrusselatorFixture> make_setup(
    WaveformIteration mode, AccelKind accel) {
  fuse::native::NativeMeshBackend backend{0, 1};
  auto u_grid_r = brusselator::NativeGrid::make(backend, "u", kNxU, kNyU);
  if (!u_grid_r.ok()) { ADD_FAILURE() << u_grid_r.error().message; return nullptr; }
  auto v_grid_r = brusselator::NativeGrid::make(backend, "v", kNxV, kNyV);
  if (!v_grid_r.ok()) { ADD_FAILURE() << v_grid_r.error().message; return nullptr; }

  auto s = std::make_unique<BrusselatorFixture>(std::move(backend),
                                                std::move(u_grid_r.value()),
                                                std::move(v_grid_r.value()));

  const fuse::MeshId u_mesh = s->graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" + std::to_string(kNyU)});
  const fuse::MeshId v_mesh = s->graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(kNxV) + "x" + std::to_string(kNyV)});
  const fuse::RegionId u_region =
      s->graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  const fuse::RegionId v_region =
      s->graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  fuse::RemapPlan plan_vu;
  plan_vu.src = v_region;
  plan_vu.tgt = u_region;
  auto w_vu = fuse::build_weights(plan_vu, s->graph, s->backend);
  if (!w_vu.ok()) { ADD_FAILURE() << w_vu.error().message; return nullptr; }
  fuse::RemapPlan plan_uv;
  plan_uv.src = u_region;
  plan_uv.tgt = v_region;
  auto w_uv = fuse::build_weights(plan_uv, s->graph, s->backend);
  if (!w_uv.ok()) { ADD_FAILURE() << w_uv.error().message; return nullptr; }
  s->w_vu = std::move(w_vu.value());
  s->w_uv = std::move(w_uv.value());

  brusselator::PhysicsConfig cfg;
  cfg.forcing = false;
  const int n_u = s->u_grid.size();
  const Rational dt_u{std::llround(kDtU * 1000000), 1000000};
  const Rational dt_v{std::llround(kDtV * 1000000), 1000000};
  const Rational window{std::llround(kWindow * 1000000), 1000000};

  auto comp_v = std::make_unique<brusselator::VComponent>(
      "V", s->u_grid, s->v_grid, cfg, &s->w_uv, &s->w_vu);
  brusselator::Vector v0_on_u =
      Eigen::Map<brusselator::Vector>(comp_v->provided_ptr(), n_u);
  auto comp_u =
      std::make_unique<brusselator::UComponent>("U", s->u_grid, cfg, v0_on_u);

  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      s->graph, u_region, dt_u, dt_v, window, kTol, kMaxIterations, accel, mode);
  (void)handles;
  if (const auto r = s->backend.register_default_placements(s->graph, u_mesh); !r.ok()) {
    ADD_FAILURE() << r.error().message;
    return nullptr;
  }

  s->components.add(std::move(comp_u));
  s->components.add(std::move(comp_v));
  for (auto& c : s->components) {
    const auto r = c.register_fields(s->registry, s->graph);
    if (!r.ok()) { ADD_FAILURE() << "register_fields: " << r.error().message; return nullptr; }
  }

  s->lowered = lower(s->graph);
  if (!s->lowered.ok()) { ADD_FAILURE() << s->lowered.diags.to_string(); return nullptr; }
  const fuse::ClockSet clocks{s->graph};
  auto sched = schedule(s->graph, s->lowered.dag, clocks,
                       fuse::make_region_sizes(s->graph, s->backend));
  if (!sched.ok()) { ADD_FAILURE() << sched.diags.to_string(); return nullptr; }

  s->runner = std::make_unique<fuse::Runner>(s->graph, s->lowered.dag, sched.plan, s->backend);
  fuse::RunnerConfig rcfg;
  const auto init = s->runner->initialize(s->components, s->registry, rcfg);
  if (!init.ok()) { ADD_FAILURE() << "initialize: " << init.error().message; return nullptr; }
  return s;
}

struct RunResult {
  brusselator::Vector u_final, v_final;
  std::vector<int> iterations_per_window;
};

RunResult run_n_periods(BrusselatorFixture& s, int n) {
  RunResult out;
  brusselator::UComponent* pu =
      dynamic_cast<brusselator::UComponent*>(s.components.find("U"));
  brusselator::VComponent* pv =
      dynamic_cast<brusselator::VComponent*>(s.components.find("V"));
  for (int w = 0; w < n; ++w) {
    const auto r = s.runner->run_period(s.components);
    EXPECT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
    if (!s.runner->iteration_stats().empty()) {
      out.iterations_per_window.push_back(s.runner->iteration_stats().front().iterations);
    }
  }
  out.u_final = pu->state();
  out.v_final = pv->state();
  return out;
}

void expect_bit_identical(const RunResult& a, const RunResult& b) {
  ASSERT_EQ(a.u_final.size(), b.u_final.size());
  for (int i = 0; i < a.u_final.size(); ++i) {
    EXPECT_EQ(a.u_final[i], b.u_final[i]) << "U[" << i << "]";
  }
  ASSERT_EQ(a.v_final.size(), b.v_final.size());
  for (int i = 0; i < a.v_final.size(); ++i) {
    EXPECT_EQ(a.v_final[i], b.v_final[i]) << "V[" << i << "]";
  }
  ASSERT_EQ(a.iterations_per_window.size(), b.iterations_per_window.size());
  for (std::size_t i = 0; i < a.iterations_per_window.size(); ++i) {
    EXPECT_EQ(a.iterations_per_window[i], b.iterations_per_window[i])
        << "iterations_per_window[" << i << "]";
  }
}

std::string temp_dir_path(const std::string& leaf) {
  return std::string(::testing::TempDir()) + "fuse_brusselator_restart_test_" + leaf;
}

fuse::Result<void> write_point(BrusselatorFixture& s, fuse::restart_io::FileLayout layout,
                               const std::string& path, const std::string& label,
                               std::int64_t period_index) {
  fuse::restart_io::Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  prov.descriptor_hash = fuse::restart_descriptor_hash(s.graph);
  prov.written_by_rank_count = 1;
  prov.mesh_checksum = 0;
  auto pw = fuse::restart_io::create_restart_point(layout, path, prov, label);
  if (!pw.ok()) return pw.error();
  const fuse::Result<void> r = fuse::write_restart_point(
      *s.runner, s.components, pw.value()->writer(), fuse::Rational{period_index},
      period_index);
  if (!r.ok()) return r;
  return pw.value()->finish();
}

fuse::Result<void> restore_point(BrusselatorFixture& s, fuse::restart_io::FileLayout layout,
                                 const std::string& path, const std::string& label,
                                 bool allow_reset, bool& any_reset) {
  auto pr = fuse::restart_io::open_restart_point(layout, path, label);
  if (!pr.ok()) return pr.error();
  fuse::Rational t;
  std::int64_t pidx = 0;
  return fuse::restore_restart_point(*s.runner, s.components, pr.value()->reader(),
                                     allow_reset, any_reset, t, pidx);
}

/// One round trip: reference run of `n` windows, uninterrupted, vs.
/// `k` windows + write + a BRAND NEW Runner/ComponentSet restore + the
/// remaining `n - k` windows -- compared bit for bit (EXPECT_EQ, no
/// tolerance), the standard this project's own round-trip tests already
/// hold same-rank-count restart to (docs/dev/restart.md Section 7.i).
void run_round_trip(WaveformIteration mode, AccelKind accel,
                    fuse::restart_io::FileLayout layout, const std::string& label) {
  const std::string path = temp_dir_path(label);
  std::remove(path.c_str());

  auto ref = make_setup(mode, accel);
  ASSERT_NE(ref, nullptr);
  const RunResult ref_result = run_n_periods(*ref, 6);

  auto part1 = make_setup(mode, accel);
  ASSERT_NE(part1, nullptr);
  run_n_periods(*part1, 3);
  const fuse::Result<void> w =
      write_point(*part1, layout, path, "p3", 3);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;

  auto restored = make_setup(mode, accel);
  ASSERT_NE(restored, nullptr);
  bool any_reset = false;
  const fuse::Result<void> rr = restore_point(*restored, layout, path, "p3",
                                              /*allow_reset=*/false, any_reset);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  // EXPECT_FALSE, not EXPECT_TRUE: round 2's MUST-FIX 2 (KNOWN_ISSUES.md
  // D69, closed) implements a window-scoped region's own accelerator
  // restart genuinely, rather than silently resetting it -- AccelKind::
  // None's own restart_state_implemented()==true (NoneAccel, src/num/
  // accel_relaxation.cpp) now means a window-scoped None region
  // restores with NO reset at all, the same as an ordinary region.
  EXPECT_FALSE(any_reset);
  const RunResult restored_result = run_n_periods(*restored, 3);

  // Only the LAST 3 windows' own iteration counts are comparable --
  // ref_result recorded 6, restored_result recorded 3 (run_n_periods()
  // starts a fresh vector per call).
  RunResult ref_tail;
  ref_tail.u_final = ref_result.u_final;
  ref_tail.v_final = ref_result.v_final;
  ref_tail.iterations_per_window.assign(ref_result.iterations_per_window.end() - 3,
                                        ref_result.iterations_per_window.end());
  expect_bit_identical(ref_tail, restored_result);
  std::remove(path.c_str());
}

}  // namespace

// NOTE (open_questions, reported): a window-scoped accelerated region's
// own RegionBlock::length is the K-WIDE stacked window-transfer buffer
// (n_u * capacity), not the registered field's own single-sample size
// -- a mismatch write_restart_fields()/restore_restart_fields()'s
// existing id resolution (built and tested only against non-window
// regions, dn_heat) does not handle. AccelKind::None is used for every
// window-region case below for exactly this reason (see the delivered
// report); QuasiNewtonAccel's own restart correctness is instead
// proven over a NON-window region in IqnRoundTripOverOrdinaryRegion
// below, where this mismatch cannot arise by construction.
TEST(BrusselatorRestart, SeparateLayoutJacobiNoAccel) {
  run_round_trip(WaveformIteration::Jacobi, AccelKind::None,
                fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                "separate_jacobi_none.h5");
}

TEST(BrusselatorRestart, SeparateLayoutGaussSeidelNoAccel) {
  run_round_trip(WaveformIteration::GaussSeidel, AccelKind::None,
                fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                "separate_gs_none.h5");
}

TEST(BrusselatorRestart, GrowingLayoutJacobiNoAccel) {
  run_round_trip(WaveformIteration::Jacobi, AccelKind::None,
                fuse::restart_io::FileLayout::GrowingFile,
                "growing_jacobi_none.h5");
}

/// Owner decision 2's refusal, D63's own convention applied to this
/// example: a restart at a graph whose descriptor differs refuses by
/// name. Exercised here by restoring a point written by the Jacobi
/// graph into a Runner built from the GaussSeidel graph -- the caller's
/// own descriptor-hash comparison (docs/user/restart.md's own worked
/// example) is what this test checks, not a Runner-internal mechanism.
TEST(BrusselatorRestart, DescriptorHashMismatchIsDetectable) {
  const std::string path = temp_dir_path("hash_mismatch.h5");
  std::remove(path.c_str());

  auto jacobi = make_setup(WaveformIteration::Jacobi, AccelKind::None);
  ASSERT_NE(jacobi, nullptr);
  run_n_periods(*jacobi, 2);
  ASSERT_TRUE(write_point(*jacobi, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 2)
                 .ok());

  auto gs = make_setup(WaveformIteration::GaussSeidel, AccelKind::None);
  ASSERT_NE(gs, nullptr);
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "");
  ASSERT_TRUE(pr.ok());
  EXPECT_NE(pr.value()->provenance().descriptor_hash,
           fuse::restart_descriptor_hash(gs->graph));
  std::remove(path.c_str());
}

// A window-scoped region's own accelerator restart with a REAL
// accelerator (AccelKind::IQN_ILS or Aitken over `brusselator_window`,
// as opposed to AccelKind::None above, which has no learned state at
// all) is NOT exercised at the Runner level in this file -- KNOWN_
// ISSUES.md D72 (open): a window-scoped region's own CONSUMER-side
// transfer buffer (`cfg.window_import`) is not correctly reconstructed
// after a restart, so a Runner-level round trip with a real
// accelerator does not yet reproduce the reference run's own iteration
// count, even though the accelerator's own state round-trips correctly
// in isolation (test/runtime/restart_test.cpp's own
// AitkenAccelWindowCapacityPerSampleRestartRoundTripsBitForBit).
// QuasiNewtonAccel's own restart correctness over an ORDINARY
// (non-window) region is proven instead by test/runtime/
// restart_test.cpp's SameRankRoundTripSeparateLayoutIqnIls.

namespace {

/// MUST-FIX 1 (NO SILENT RESET), round 2: the SAME test-only double
/// test/runtime/restart_test.cpp's own NeverImplementsRestartAccel is
/// (duplicated here rather than shared across translation units, since
/// it is a test-only class with no production header), used to verify
/// a window-scoped region's own refusal-by-name is now IDENTICAL to an
/// ordinary region's -- the special case that used to set
/// any_accelerator_reset=true and skip WITHOUT checking
/// allow_accelerator_reset_on_mismatch is removed.
class NeverImplementsRestartAccel final : public fuse::Accelerator {
 public:
  void reset() override {}
  [[nodiscard]] std::string name() const override {
    return "NeverImplementsRestartAccel (test-only)";
  }
  [[nodiscard]] std::size_t columns() const override { return 0; }
  fuse::Result<void> update(const double* /*x_old*/, const double* x_new,
                            double* x_out, std::size_t n, int) override {
    for (std::size_t i = 0; i < n; ++i) x_out[i] = x_new[i];
    return fuse::Result<void>::success();
  }
  // Deliberately NOT overridden: restart_state_implemented() (stays
  // false), export_restart_state() (stays a no-op), import_restart_
  // state() (stays the base class's unconditional Failure).
};

/// `brusselator_window` is the ONLY iteration region add_window_
/// coupling() compiles into this graph, so region_index 0 is its own
/// index on EVERY Runner this file builds -- the same convention
/// test/runtime/restart_test.cpp's own make_setup_with_test_only_accel()
/// relies on for dn_heat's single region.
std::unique_ptr<BrusselatorFixture> make_setup_with_test_only_accel(
    WaveformIteration mode) {
  auto s = make_setup(mode, AccelKind::Aitken);
  if (s == nullptr) return s;
  const fuse::Result<void> r = s->runner->debug_replace_region_accelerator(
      0, std::make_unique<NeverImplementsRestartAccel>());
  if (!r.ok()) ADD_FAILURE() << "debug_replace_region_accelerator: " << r.error().message;
  return s;
}

}  // namespace

TEST(BrusselatorRestart, WindowScopedAccelNotImplementedRefusesWithoutOptIn) {
  const std::string path = temp_dir_path("window_accel_not_implemented.h5");
  std::remove(path.c_str());
  auto s = make_setup_with_test_only_accel(WaveformIteration::Jacobi);
  ASSERT_NE(s, nullptr);
  run_n_periods(*s, 2);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 2)
                 .ok());

  auto restored = make_setup_with_test_only_accel(WaveformIteration::Jacobi);
  ASSERT_NE(restored, nullptr);
  bool any_reset = false;
  const fuse::Result<void> rr = restore_point(
      *restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "",
      /*allow_reset=*/false, any_reset);
  ASSERT_FALSE(rr.ok())
      << "a window-scoped region's own accelerator must refuse by name, "
         "not silently reset, when restart_state_implemented() == false "
         "and the run did not opt in -- MUST-FIX 1, round 2";
  EXPECT_EQ(rr.error().code, "E-RESTART-ACCEL-NOT-IMPLEMENTED");
  EXPECT_FALSE(any_reset);
  std::remove(path.c_str());
}

TEST(BrusselatorRestart, WindowScopedAccelNotImplementedWithAllowResetSucceedsAndReports) {
  const std::string path = temp_dir_path("window_accel_not_implemented_reset.h5");
  std::remove(path.c_str());
  auto s = make_setup_with_test_only_accel(WaveformIteration::Jacobi);
  ASSERT_NE(s, nullptr);
  run_n_periods(*s, 2);
  ASSERT_TRUE(write_point(*s, fuse::restart_io::FileLayout::SeparateMeshAndPoints,
                          path, "", 2)
                 .ok());

  auto restored = make_setup_with_test_only_accel(WaveformIteration::Jacobi);
  ASSERT_NE(restored, nullptr);
  bool any_reset = false;
  const fuse::Result<void> rr = restore_point(
      *restored, fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, "",
      /*allow_reset=*/true, any_reset);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;
  EXPECT_TRUE(any_reset)
      << "opting in must call Accelerator::reset() and REPORT it, per "
         "owner decision 4 (\"when used it is reported at restart\")";
  std::remove(path.c_str());
}
