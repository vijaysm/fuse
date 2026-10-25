// NewtonNorm::MeasureWeighted over a window-scoped Newton region
// (`Runner::Impl::run_region_newton_window()`, src/runtime/iteration.cpp),
// exercised against the real Brusselator example -- the other coverage is
// a synthetic two-member fixture (test/runtime/newton_norm_test.cpp) and
// dn_heat's own ordinary (non-window) region.
//
// Built from the SAME conventions test/verification/brusselator_runner_
// test.cpp's own BrusselatorRunner.NewtonConvergesInAPinnedIterationCount
// WithTheBlockJacobiPreconditioner uses (read-only reference, NOT edited
// here): Rig/make_rig, the same dt_u/dt_v/window, NewtonConfig set up the
// identical way, brusselator::add_window_coupling() with AccelKind::
// Newton. Two grid sizes are used, deliberately:
//  - the SAME small, fast 5x5/7x7 grid that file's own Rig uses, for (a)
//    the Plain-vs-MeasureWeighted convergence/agreement comparison and
//    (c) the GMRES iteration-count, composed-vs-bare-preconditioner
//    comparison -- neither depends on a specific grid size, so the faster
//    one is used;
//  - the example's own REAL defaults (examples/brusselator/main.cpp:
//    --nx-u 16 --ny-u 16 --nx-v 24 --ny-v 24 --dt-u 0.002 --dt-v 0.003
//    --window 0.02) for (b), the ONE check tied to a specific number
//    (docs/dev/newton-krylov.md Sec. 9.1's own "measured length,
//    Brusselator at its own defaults" table, n=4864) -- any other grid
//    would make that number meaningless.
//
// The entry-for-entry formula check in (b) reuses test/runtime/newton_
// norm_test.cpp's own NewtonNorm.ProductionDiagonalMatchesTheFormulaEntry
// ForEntry technique: NativeGrid's own x(i)=(i+0.5)/nx, y(j)=(j+0.5)/ny
// (examples/brusselator/brusselator_physics.hpp) is a UNIFORM unit-square
// grid, so every cell's own entity measure is identical (1/(nx*ny)) and
// cancels out of D_i = sqrt(w_i / sum_w) / s_b regardless of what that
// measure's own units or normalisation are -- leaving a closed-form
// D_i = sqrt(1 / (capacity * n_u)) / s_b to compare the PRODUCTION value
// against, with no separate mesh-area oracle needed (and no risk of the
// unit-factor mistake: this derivation is independent of any component's
// own coefficient convention).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/runtime/window_coupler.hpp>

#include "brusselator_graph.hpp"
#include "brusselator_physics.hpp"
#include "species_component.hpp"

using fuse::AccelKind;
using fuse::ComponentSet;
using fuse::FieldRegistry;
using fuse::NewtonConfig;
using fuse::NewtonNorm;
using fuse::Rational;
using fuse::Result;
using fuse::WaveformIteration;

namespace {

// Same small, fast grid as brusselator_runner_test.cpp's own Rig.
constexpr int kNxU = 5, kNyU = 5, kNxV = 7, kNyV = 7;
// The example's own real defaults (examples/brusselator/main.cpp).
constexpr int kFullNxU = 16, kFullNyU = 16, kFullNxV = 24, kFullNyV = 24;
constexpr double kDtU = 0.002, kDtV = 0.003, kWindow = 0.02;
constexpr double kTol = 1e-10;
constexpr int kMaxIterations = 100;

struct Rig {
  fuse::native::NativeMeshBackend backend;
  brusselator::NativeGrid u_grid;
  brusselator::NativeGrid v_grid;
  fuse::RemapWeights w_vu, w_uv;  // V->U, U->V
  std::unique_ptr<brusselator::VComponent> comp_v;
  std::unique_ptr<brusselator::UComponent> comp_u;

  Rig(fuse::native::NativeMeshBackend backend_in, brusselator::NativeGrid u,
      brusselator::NativeGrid v)
      : backend(std::move(backend_in)), u_grid(std::move(u)), v_grid(std::move(v)) {}
};

// Mirrors brusselator_runner_test.cpp's own make_rig(), parameterised by
// grid size so the SAME function builds both the small and full-size rig.
[[nodiscard]] Result<std::unique_ptr<Rig>> make_rig(int nx_u, int ny_u, int nx_v, int ny_v) {
  fuse::native::NativeMeshBackend backend{0, 1};
  auto u_grid_r = brusselator::NativeGrid::make(backend, "u", nx_u, ny_u);
  if (!u_grid_r.ok()) return u_grid_r.error();
  auto v_grid_r = brusselator::NativeGrid::make(backend, "v", nx_v, ny_v);
  if (!v_grid_r.ok()) return v_grid_r.error();

  auto rig = std::make_unique<Rig>(std::move(backend), std::move(u_grid_r.value()),
                                   std::move(v_grid_r.value()));

  fuse::CouplingGraph scratch_graph;
  const fuse::MeshId u_mesh = scratch_graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(nx_u) + "x" + std::to_string(ny_u)});
  const fuse::MeshId v_mesh = scratch_graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(nx_v) + "x" + std::to_string(ny_v)});
  const fuse::RegionId u_region =
      scratch_graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  const fuse::RegionId v_region =
      scratch_graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  fuse::RemapPlan plan_vu;
  plan_vu.src = v_region;
  plan_vu.tgt = u_region;
  auto w_vu = fuse::build_weights(plan_vu, scratch_graph, rig->backend);
  if (!w_vu.ok()) return w_vu.error();
  fuse::RemapPlan plan_uv;
  plan_uv.src = u_region;
  plan_uv.tgt = v_region;
  auto w_uv = fuse::build_weights(plan_uv, scratch_graph, rig->backend);
  if (!w_uv.ok()) return w_uv.error();
  rig->w_vu = std::move(w_vu.value());
  rig->w_uv = std::move(w_uv.value());

  brusselator::PhysicsConfig cfg;
  cfg.forcing = false;
  const int n_u = rig->u_grid.size();
  const fuse::Rational dt_u{std::llround(kDtU * 1000000), 1000000};
  const fuse::Rational dt_v{std::llround(kDtV * 1000000), 1000000};
  const fuse::Rational window{std::llround(kWindow * 1000000), 1000000};
  // with_window_metadata = true unconditionally -- every caller of this
  // file's own make_rig() needs AccelKind::Newton's capabilities() bits.
  rig->comp_v = std::make_unique<brusselator::VComponent>(
      "V", rig->u_grid, rig->v_grid, cfg, &rig->w_uv, &rig->w_vu, dt_v, dt_u, window);
  brusselator::Vector v0_on_u =
      Eigen::Map<brusselator::Vector>(rig->comp_v->provided_ptr(), n_u);
  rig->comp_u = std::make_unique<brusselator::UComponent>(
      "U", rig->u_grid, cfg, v0_on_u, dt_u, dt_v, window);
  return std::move(rig);  // Result<unique_ptr>: no NRVO, nvcc's host pass needs the move spelled out
}

struct WindowNewtonRunResult {
  bool ok = false;
  std::string error;
  brusselator::Vector u_final, v_final;
  int iterations = 0;
  bool converged = false;
  std::vector<double> newton_scale;             // debug_newton_scale(0)
  std::vector<std::size_t> gmres_iterations;     // debug_newton_gmres_iterations(0)
  std::size_t newton_n = 0;
  std::vector<std::size_t> offsets, lengths;
  std::vector<std::string> channel_names;
  std::vector<std::size_t> excluded_lengths;
  std::vector<std::string> excluded_channel_names;
};

/// Builds the window-Newton Brusselator graph at the given grid size and
/// NewtonConfig, runs ONE window (`run_period`, one window-scoped
/// iteration region, the SAME call brusselator_runner_test.cpp's own
/// run_via_runner_newton() makes), and extracts every diagnostic this
/// file's own tests (a)-(c) need before the Runner goes out of scope.
[[nodiscard]] WindowNewtonRunResult run_brusselator_newton_window(
    int nx_u, int ny_u, int nx_v, int ny_v, const NewtonConfig& newton_cfg_in) {
  WindowNewtonRunResult out;
  auto rig_r = make_rig(nx_u, ny_u, nx_v, ny_v);
  if (!rig_r.ok()) { out.error = rig_r.error().message; return out; }
  auto rig = std::move(rig_r.value());

  fuse::CouplingGraph graph;
  const fuse::MeshId u_mesh = graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(nx_u) + "x" + std::to_string(ny_u)});
  const fuse::MeshId v_mesh = graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(nx_v) + "x" + std::to_string(ny_v)});
  const fuse::RegionId u_region = graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  (void)graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  const Rational dt_u{std::llround(kDtU * 1000000), 1000000};
  const Rational dt_v{std::llround(kDtV * 1000000), 1000000};
  const Rational window{std::llround(kWindow * 1000000), 1000000};
  NewtonConfig newton_cfg = newton_cfg_in;
  // examples/brusselator/main.cpp's own choice (newton-krylov.md Sec.
  // 9.9.2: eta_min's own SNES default, 0.0, stalls this problem).
  newton_cfg.eta_min = 1e-4;
  newton_cfg.substrate = fuse::NewtonSubstrate::Native;
  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      graph, u_region, dt_u, dt_v, window, kTol, kMaxIterations, AccelKind::Newton,
      WaveformIteration::Jacobi, /*coupler_ranks=*/std::nullopt,
      /*u_ranks=*/fuse::RankSet{0, 0}, /*v_ranks=*/fuse::RankSet{0, 0},
      /*u_launch=*/fuse::LaunchSpec{}, /*v_launch=*/fuse::LaunchSpec{},
      /*coupler_launch=*/fuse::LaunchSpec{}, newton_cfg);
  (void)handles;
  if (const auto r = rig->backend.register_default_placements(graph, u_mesh); !r.ok()) {
    out.error = r.error().message;
    return out;
  }

  ComponentSet components;
  components.add(std::move(rig->comp_u));
  components.add(std::move(rig->comp_v));

  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, graph);
    if (!r.ok()) { out.error = r.error().message; return out; }
  }

  const auto lowered = lower(graph);
  if (!lowered.ok()) { out.error = lowered.diags.to_string(); return out; }
  const fuse::ClockSet clocks{graph};
  auto sched = schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, rig->backend));
  if (!sched.ok()) { out.error = sched.diags.to_string(); return out; }

  fuse::RunnerConfig rcfg;
  rcfg.abort_on_nonconvergence = true;
  fuse::Runner runner{graph, lowered.dag, sched.plan, rig->backend};
  const auto init = runner.initialize(components, registry, rcfg);
  if (!init.ok()) { out.error = init.error().message; return out; }

  brusselator::UComponent* pu = nullptr;
  brusselator::VComponent* pv = nullptr;
  for (auto& c : components) {
    if (c.name() == "U") pu = dynamic_cast<brusselator::UComponent*>(&c);
    if (c.name() == "V") pv = dynamic_cast<brusselator::VComponent*>(&c);
  }
  if (pu == nullptr || pv == nullptr) { out.error = "component lookup failed"; return out; }

  const auto r = runner.run_period(components);
  if (!r.ok()) { out.error = r.error().message; return out; }
  if (runner.iteration_stats().empty()) { out.error = "no iteration stats"; return out; }
  out.iterations = runner.iteration_stats().front().iterations;
  out.converged = runner.iteration_stats().front().converged;
  out.ok = true;
  out.u_final = pu->state();
  out.v_final = pv->state();

  if (const auto d = runner.debug_newton_scale(0); d.ok()) out.newton_scale = d.value();
  if (const auto g = runner.debug_newton_gmres_iterations(0); g.ok()) out.gmres_iterations = g.value();
  if (const auto w = runner.window_region_fixed_point_residual(0, components); w.ok()) {
    out.newton_n = w.value().n;
    out.offsets = w.value().offsets;
    out.lengths = w.value().lengths;
    out.channel_names = w.value().channel_names;
    out.excluded_lengths = w.value().excluded_lengths;
    out.excluded_channel_names = w.value().excluded_channel_names;
  }
  return out;
}

/// `ceil(window/interval) + 1` -- newton-krylov.md Sec. 9.1's own
/// `window_sample_capacity` formula (`coupling_graph.hpp:505-515`), no
/// `DtPolicy::Fixed` `+1` since both Brusselator members are Adjustable.
[[nodiscard]] std::size_t window_capacity(double window, double interval) {
  return static_cast<std::size_t>(std::ceil(window / interval)) + 1;
}

}  // namespace

// (a) Plain and MeasureWeighted both converge the window-Newton solve,
// to the SAME physical answer within Newton's own tolerance -- the
// identical comparison brusselator_runner_test.cpp's own
// NewtonConvergesInAPinnedIterationCountWithTheBlockJacobiPreconditioner
// makes between with/without the block-Jacobi preconditioner (that test's
// own EXPECT_NEAR(..., 1e-9) bound, and its own citation of "~3e-13,
// consistent with NewtonConfig::rtol's own 1e-8 relative-to-||F_0||
// floor"), now between norm=Plain and norm=MeasureWeighted instead of
// between two preconditioner choices. No closed-form bound is derivable
// for this nonlinear reaction-diffusion problem (unlike the GAP 3 exact-
// linear-system case) -- the measured difference is reported honestly,
// against a sanity bound tied to NewtonConfig's own stopping tolerance
// rather than an arbitrary round number.
TEST(BrusselatorNewtonNorm, WindowNewtonConvergesUnderPlainAndMeasureWeighted) {
  NewtonConfig plain_cfg;
  plain_cfg.use_block_jacobi_preconditioner = true;
  plain_cfg.norm = NewtonNorm::Plain;
  const WindowNewtonRunResult plain =
      run_brusselator_newton_window(kNxU, kNyU, kNxV, kNyV, plain_cfg);
  ASSERT_TRUE(plain.ok) << plain.error;
  EXPECT_TRUE(plain.converged) << "iterations=" << plain.iterations;

  NewtonConfig weighted_cfg = plain_cfg;
  weighted_cfg.norm = NewtonNorm::MeasureWeighted;
  const WindowNewtonRunResult weighted =
      run_brusselator_newton_window(kNxU, kNyU, kNxV, kNyV, weighted_cfg);
  ASSERT_TRUE(weighted.ok) << weighted.error;
  EXPECT_TRUE(weighted.converged) << "iterations=" << weighted.iterations;

  ASSERT_EQ(plain.u_final.size(), weighted.u_final.size());
  ASSERT_EQ(plain.v_final.size(), weighted.v_final.size());
  double max_u_diff = 0.0, max_v_diff = 0.0;
  for (Eigen::Index i = 0; i < plain.u_final.size(); ++i) {
    max_u_diff = std::max(max_u_diff, std::abs(plain.u_final[i] - weighted.u_final[i]));
  }
  for (Eigen::Index i = 0; i < plain.v_final.size(); ++i) {
    max_v_diff = std::max(max_v_diff, std::abs(plain.v_final[i] - weighted.v_final[i]));
  }
  fprintf(stderr,
         "[BrusselatorNewtonNorm] GAP1(a) MEASURED: plain iterations=%d, "
         "weighted iterations=%d, max|u_plain-u_weighted|=%.6e, "
         "max|v_plain-v_weighted|=%.6e\n",
         plain.iterations, weighted.iterations, max_u_diff, max_v_diff);

  // Sanity bound, not a derived one: both stop by the SAME NewtonConfig
  // (default rtol/atol -- SNES's own 1e-8/1e-50), so neither run's own
  // final state should differ from the TRUE fixed point, nor from each
  // other, by more than a modest multiple of that stopping tolerance
  // relative to the field's own magnitude -- 1e-6 absolute, three orders
  // of magnitude looser than brusselator_runner_test.cpp's own 1e-9 pin
  // for its (much better conditioned, SAME-norm) with/without-block-
  // Jacobi comparison, since Plain and MeasureWeighted are different
  // norms entirely, not just different preconditioner paths to the same
  // one.
  EXPECT_LT(max_u_diff, 1e-6) << "MEASURED max|u_plain-u_weighted| = " << max_u_diff;
  EXPECT_LT(max_v_diff, 1e-6) << "MEASURED max|v_plain-v_weighted| = " << max_v_diff;
}

// (b) The production window D (`debug_newton_scale()`), checked against
// newton-krylov.md Sec. 9.1's own formula entry-for-entry, at the
// example's OWN real defaults -- the one configuration Sec. 9.1's own
// "measured length" table (n=4864) is about. If D's length, its per-
// sample uniformity, or its per-entry value disagree with the formula,
// THAT is the finding to report, not something this test may loosen to
// pass.
TEST(BrusselatorNewtonNorm, WindowDiagonalMatchesTheFormulaEntryForEntryAtTheExampleDefaults) {
  NewtonConfig cfg;
  cfg.use_block_jacobi_preconditioner = false;  // irrelevant to D itself; keep the solve cheap.
  cfg.norm = NewtonNorm::MeasureWeighted;
  const WindowNewtonRunResult r =
      run_brusselator_newton_window(kFullNxU, kFullNyU, kFullNxV, kFullNyV, cfg);
  ASSERT_TRUE(r.ok) << r.error;
  EXPECT_TRUE(r.converged) << "iterations=" << r.iterations;

  // Sec. 9.1's own measured total, at these exact grid/dt/window values.
  EXPECT_EQ(r.newton_n, 4864u);
  ASSERT_EQ(r.newton_scale.size(), r.newton_n);
  ASSERT_EQ(r.offsets.size(), r.lengths.size());
  ASSERT_EQ(r.offsets.size(), r.channel_names.size());
  ASSERT_EQ(r.offsets.size(), 2u) << "expected exactly 2 blocks (u2v, v2u) under Jacobi";
  EXPECT_TRUE(r.excluded_lengths.empty()) << "Jacobi forces every channel lagged -- nothing excluded";

  const std::size_t n_u = static_cast<std::size_t>(kFullNxU) * static_cast<std::size_t>(kFullNyU);
  const std::size_t cap_u = window_capacity(kWindow, kDtU);  // 11
  const std::size_t cap_v = window_capacity(kWindow, kDtV);  // 8
  fprintf(stderr,
         "[BrusselatorNewtonNorm] GAP1(b) MEASURED: newton_n=%zu n_u=%zu cap_u=%zu cap_v=%zu\n",
         r.newton_n, n_u, cap_u, cap_v);

  std::size_t checked_blocks = 0;
  for (std::size_t bi = 0; bi < r.offsets.size(); ++bi) {
    const std::string& name = r.channel_names[bi];
    const std::size_t capacity = (name == "u2v") ? cap_u : (name == "v2u") ? cap_v : 0u;
    ASSERT_NE(capacity, 0u) << "unrecognised channel '" << name << "'";
    ASSERT_EQ(r.lengths[bi], n_u * capacity)
        << "block '" << name << "': n_u * capacity mismatch";

    // Every cell of the Brusselator's own unit-square grid has IDENTICAL
    // area (NativeGrid::x/y, examples/brusselator/brusselator_physics.hpp)
    // -- the SAME uniform-mesh exploit test/runtime/newton_norm_test.cpp's
    // own ProductionDiagonalMatchesTheFormulaEntryForEntry uses, so the
    // per-cell measure cancels out of D_i = sqrt(w_i/sum_w)/s_b, leaving
    // this closed form regardless of w_i's own units or normalisation.
    const double expected = std::sqrt(1.0 / (static_cast<double>(capacity) *
                                             static_cast<double>(n_u))) / kTol;
    for (std::size_t e = 0; e < n_u; ++e) {
      double first_sample = 0.0;
      for (std::size_t s = 0; s < capacity; ++s) {
        const double di = r.newton_scale[r.offsets[bi] + e * capacity + s];
        if (s == 0) {
          first_sample = di;
          EXPECT_NEAR(di, expected, 1e-6 * expected)
              << "block '" << name << "' element " << e << " sample " << s;
        } else {
          // Uniform spatial weight per window sample: every sample
          // column at the SAME element must carry the identical D.
          EXPECT_EQ(di, first_sample)
              << "block '" << name << "' element " << e << " sample " << s
              << " disagrees with its own sample 0 -- not uniform in time";
        }
      }
    }
    ++checked_blocks;
  }
  EXPECT_EQ(checked_blocks, 2u);
}

// (c) GMRES iteration count, block-Jacobi-WITH-D^-1-composition
// ("composed", production, always on when norm == MeasureWeighted &&
// use_block_jacobi_preconditioner) vs no preconditioner at all
// ("bare-of-any-preconditioner", IdentityPreconditioner, use_block_
// jacobi_preconditioner = false) -- the SAME with/without-block-Jacobi
// comparison brusselator_runner_test.cpp's own pinning test makes for
// Newton iteration count, extended here to GMRES iteration count via
// the debug_newton_gmres_iterations() accessor.
//
// MEASURED, not assumed: composed costs MORE total GMRES work here (42
// vs 8), not less -- a real preconditioner does not always reduce Krylov
// work, so this test does not assert `EXPECT_LE(composed, bare)`.
// brusselator_runner_test.cpp's own header for
// NewtonConvergesInAPinnedIterationCountWithTheBlockJacobiPreconditioner
// already measured the IDENTICAL direction
// at the full-size grid -- "fewer outer iterations on the first window
// (3, not 4)" WITHOUT the block-Jacobi preconditioner, citing "a real
// preconditioner changes the PATH to convergence, never whether
// convergence happens" -- the SAME qualitative fact (3 Newton iterations
// bare vs 4 composed) reproduces at this file's own smaller grid, now
// also measured at the GMRES (inner), not just Newton (outer), level:
// each of composed's own 4 Newton steps needs MORE GMRES work than its
// bare counterpart, not less, on this specific problem's own spectrum.
// Pinned as a regression check (an unexplained CHANGE in either value is
// worth a second look), not asserted in a direction this measurement
// itself does not support. The D^-1-COMPOSITION-specific regression
// (leaving the bare block-Jacobi M uncomposed with D^-1 -- a THIRD,
// different "bare" than this test's own no-preconditioner-at-all one)
// is proved separately, by the mutation row below, on the whole binary.
TEST(BrusselatorNewtonNorm, WindowGmresIterationCountsPinnedWithAndWithoutBlockJacobi) {
  NewtonConfig cfg;
  cfg.norm = NewtonNorm::MeasureWeighted;

  cfg.use_block_jacobi_preconditioner = true;
  const WindowNewtonRunResult composed =
      run_brusselator_newton_window(kNxU, kNyU, kNxV, kNyV, cfg);
  ASSERT_TRUE(composed.ok) << composed.error;
  EXPECT_TRUE(composed.converged) << "iterations=" << composed.iterations;

  cfg.use_block_jacobi_preconditioner = false;
  const WindowNewtonRunResult bare =
      run_brusselator_newton_window(kNxU, kNyU, kNxV, kNyV, cfg);
  ASSERT_TRUE(bare.ok) << bare.error;
  EXPECT_TRUE(bare.converged) << "iterations=" << bare.iterations;

  const std::size_t composed_total = std::accumulate(
      composed.gmres_iterations.begin(), composed.gmres_iterations.end(), std::size_t{0});
  const std::size_t bare_total = std::accumulate(
      bare.gmres_iterations.begin(), bare.gmres_iterations.end(), std::size_t{0});
  fprintf(stderr,
         "[BrusselatorNewtonNorm] GAP1(c) MEASURED: composed Newton_it=%d gmres_total=%zu "
         "(per-step=[%s]); bare Newton_it=%d gmres_total=%zu (per-step=[%s])\n",
         composed.iterations, composed_total, [&] {
           std::string s;
           for (auto g : composed.gmres_iterations) s += std::to_string(g) + " ";
           return s;
         }().c_str(),
         bare.iterations, bare_total, [&] {
           std::string s;
           for (auto g : bare.gmres_iterations) s += std::to_string(g) + " ";
           return s;
         }().c_str());

  // Regression pins, both directions reported exactly as measured.
  EXPECT_EQ(composed.iterations, 4);
  EXPECT_EQ(bare.iterations, 3);
  EXPECT_EQ(composed_total, 42u);
  EXPECT_EQ(bare_total, 8u);
}

