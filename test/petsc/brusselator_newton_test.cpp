// The Brusselator's own window-Newton region
// (examples/brusselator, docs/dev/newton-krylov.md Sec. 9), solved by the
// PETSc substrate (fuse::petsc::solve(), Sec. 10/11) instead of the native
// one -- NewtonConfig::substrate is the only difference between this file
// and test/verification/brusselator_runner_test.cpp's own
// NewtonConvergesInAPinnedIterationCountWithTheBlockJacobiPreconditioner,
// which this file's own Rig/run helper mirrors deliberately (the project's
// own "near-duplicate rather than shared" convention, examples/dn_heat/
// main.cpp's run_case(NewtonConfig, ...) overload comment: each one's own
// line stays provably untouched by the other).
//
// WHY A SEPARATE BINARY, not one more TEST() in brusselator_runner_test.cpp.
// fuse::petsc::ensure_initialized() reaches PETSc's own PetscInitialize(),
// which calls MPI_Init() when nothing has -- and where MPI cannot start
// that SEGFAULTS the whole process, not merely fails one gtest case.
// Putting this case inside brusselator_runner_test.cpp would take down
// EVERY other case in that binary (JacobiMatchesDriver,
// NewtonConvergesInAPinnedIterationCountWithTheBlockJacobiPreconditioner,
// etc. -- otherwise-passing, PETSc-independent coverage) via one shared-
// process crash. Every existing petsc.* test (test/petsc/{init,token,
// derivative_gate,snes}_test.cpp) is its own dedicated binary for
// precisely this reason (token.hpp's own comment on why init_test.cpp in
// particular is never shared); this file follows the SAME convention.
// `fuse_add_gtest(petsc brusselator_newton ...)` (test/CMakeLists.txt)
// registers it as its own process, `petsc.brusselator_newton`.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/petsc/solve.hpp>
#include <fuse/runtime/runner.hpp>

#include "agreement_bound.hpp"
#include "brusselator_graph.hpp"
#include "brusselator_physics.hpp"
#include "species_component.hpp"

using fuse::AccelKind;
using fuse::ComponentSet;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Result;

namespace {

// Identical fixture shape to test/verification/brusselator_runner_test.cpp's
// own kNxU/kNyU/kNxV/kNyV/kDtU/kDtV/kWindow -- the SAME problem, so a
// native-vs-PETSc comparison is apples to apples.
constexpr int kNxU = 5, kNyU = 5, kNxV = 7, kNyV = 7;
constexpr double kDtU = 0.002, kDtV = 0.003, kWindow = 0.02;
constexpr double kTol = 1e-10;
constexpr int kMaxIterations = 100;

struct RunResult {
  bool ok = false;
  std::string error;
  brusselator::Vector u_final, v_final;
  int iterations = 0;
  // The window-Newton region's own residual
  // history (IterationStats::residual_history, fuse/runtime/runner.hpp):
  // one entry per Newton step's own pre-step ||F||, in order, followed by
  // the solve's own final ||F|| -- run_region_newton_window() (src/
  // runtime/iteration.cpp) pushes `rec.norm_f` for every trace entry
  // before appending `final_residual_norm`. `residual_history.back()` is
  // therefore the solve's own converged ||F|| (`f_final` in
  // `brusselator_bound::agreement_bound()`'s own comment,
  // examples/brusselator/agreement_bound.hpp).
  std::vector<double> residual_history;
};

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

// WITH window metadata unconditionally -- this file's own ONLY use is the
// Newton path, which needs `capabilities()` to report `kCapabilityResidual`/
// `kCapabilityLinearization` (brusselator_runner_test.cpp's own
// `make_rig(true)` precedent).
[[nodiscard]] Result<std::unique_ptr<Rig>> make_rig() {
  fuse::native::NativeMeshBackend backend{0, 1};
  auto u_grid_r = brusselator::NativeGrid::make(backend, "u", kNxU, kNyU);
  if (!u_grid_r.ok()) return u_grid_r.error();
  auto v_grid_r = brusselator::NativeGrid::make(backend, "v", kNxV, kNyV);
  if (!v_grid_r.ok()) return v_grid_r.error();

  auto rig = std::make_unique<Rig>(std::move(backend), std::move(u_grid_r.value()),
                                   std::move(v_grid_r.value()));

  fuse::CouplingGraph scratch_graph;
  const fuse::MeshId u_mesh = scratch_graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" + std::to_string(kNyU)});
  const fuse::MeshId v_mesh = scratch_graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(kNxV) + "x" + std::to_string(kNyV)});
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
  rig->comp_v = std::make_unique<brusselator::VComponent>(
      "V", rig->u_grid, rig->v_grid, cfg, &rig->w_uv, &rig->w_vu, dt_v, dt_u, window);
  brusselator::Vector v0_on_u =
      Eigen::Map<brusselator::Vector>(rig->comp_v->provided_ptr(), n_u);
  rig->comp_u = std::make_unique<brusselator::UComponent>(
      "U", rig->u_grid, cfg, v0_on_u, dt_u, dt_v, window);
  return rig;
}

// `max_nonlinear_iterations_cap`/`abort_on_nonconvergence`/`rtol_override`:
// Defaults keep the plain behaviour: no cap, i.e. `NewtonConfig`'s own
// default of 50 steps; abort on non-convergence; `NewtonConfig`'s own
// default `rtol` of 1e-8. `rtol_override`, when set, is this function's own way
// of running the IDENTICAL window at a tighter or looser relative
// convergence test than `ConvergesAgreeingWithNative`'s own native/PETSc
// solves use -- `measure_native_reference_error()` below is the only
// caller that sets it, to obtain a tight reference solution for
// calibrating the agreement bound's own error term against.
[[nodiscard]] RunResult run_one_window(
    bool use_block_jacobi_preconditioner, fuse::NewtonSubstrate substrate,
    std::optional<int> max_nonlinear_iterations_cap = std::nullopt,
    bool abort_on_nonconvergence = true,
    std::optional<double> rtol_override = std::nullopt) {
  RunResult out;
  auto rig_r = make_rig();
  if (!rig_r.ok()) { out.error = rig_r.error().message; return out; }
  auto rig = std::move(rig_r.value());

  fuse::CouplingGraph graph;
  const fuse::MeshId u_mesh = graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" + std::to_string(kNyU)});
  const fuse::MeshId v_mesh = graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(kNxV) + "x" + std::to_string(kNyV)});
  const fuse::RegionId u_region = graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  (void)graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  const Rational dt_u{std::llround(kDtU * 1000000), 1000000};
  const Rational dt_v{std::llround(kDtV * 1000000), 1000000};
  const Rational window{std::llround(kWindow * 1000000), 1000000};
  fuse::NewtonConfig newton_cfg;
  newton_cfg.use_block_jacobi_preconditioner = use_block_jacobi_preconditioner;
  newton_cfg.eta_min = 1e-4;  // examples/brusselator/main.cpp's own choice
  newton_cfg.substrate = substrate;
  if (max_nonlinear_iterations_cap.has_value()) {
    newton_cfg.max_nonlinear_iterations = *max_nonlinear_iterations_cap;
  }
  if (rtol_override.has_value()) {
    newton_cfg.rtol = *rtol_override;
  }
  const brusselator::WindowGraphHandles handles = brusselator::add_window_coupling(
      graph, u_region, dt_u, dt_v, window, kTol, kMaxIterations, AccelKind::Newton,
      fuse::WaveformIteration::Jacobi, /*coupler_ranks=*/std::nullopt,
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
  rcfg.abort_on_nonconvergence = abort_on_nonconvergence;
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
  out.residual_history = runner.iteration_stats().front().residual_history;
  out.ok = true;
  out.u_final = pu->state();
  out.v_final = pv->state();
  return out;
}

// The error-term calibration
// `brusselator_bound::agreement_bound()` needs (examples/brusselator/
// agreement_bound.hpp carries the full derivation): the native solver's
// own measured error at the CHECK's own `rtol`, against a tight
// reference solve of the IDENTICAL window -- same grid, same
// preconditioner, same everything except `rtol`. Runs entirely on the
// NATIVE substrate, never touching PETSc, so it is always safe to call
// where `MPI_Init()` cannot complete.
struct ReferenceCalibration {
  bool ok = false;
  std::string error;
  /// max(e_n_u, e_n_v) -- the single scalar `agreement_bound()` takes,
  /// since the real check applies ONE bound to both fields.
  double e_n = 0.0;
  double e_n_u = 0.0;
  double e_n_v = 0.0;
  /// residual_history.back() of the CHECK-tolerance solve -- `f_final_
  /// native` in `agreement_bound()`'s own signature.
  double f_final_native = 0.0;
  double reference_final_residual = 0.0;
  int check_iterations = 0;
  int reference_iterations = 0;
  /// ||reference_1 - reference_2||_inf, over both fields -- the
  /// required cross-check: the two tight references must agree to
  /// far less than `e_n`, or this measurement is not trustworthy.
  double reference_cross_check_diff = 0.0;
  bool reference_stable = false;
};

[[nodiscard]] ReferenceCalibration measure_native_reference_error(
    bool use_block_jacobi_preconditioner, double reference_rtol_1,
    double reference_rtol_2) {
  ReferenceCalibration out;

  // The CHECK's own tolerance: `NewtonConfig`'s own default (1e-8),
  // unoverridden -- the IDENTICAL call `ConvergesAgreeingWithNative`'s
  // own native run makes, by construction (no separate constant to
  // drift out of sync with the real check).
  const RunResult check =
      run_one_window(use_block_jacobi_preconditioner, fuse::NewtonSubstrate::Native);
  if (!check.ok) {
    out.error = "check-tolerance solve: " + check.error;
    return out;
  }

  const RunResult ref1 = run_one_window(
      use_block_jacobi_preconditioner, fuse::NewtonSubstrate::Native,
      /*max_nonlinear_iterations_cap=*/std::nullopt, /*abort_on_nonconvergence=*/true,
      reference_rtol_1);
  if (!ref1.ok) {
    out.error = "reference solve (rtol=" + std::to_string(reference_rtol_1) + "): " + ref1.error;
    return out;
  }

  const RunResult ref2 = run_one_window(
      use_block_jacobi_preconditioner, fuse::NewtonSubstrate::Native,
      /*max_nonlinear_iterations_cap=*/std::nullopt, /*abort_on_nonconvergence=*/true,
      reference_rtol_2);
  if (!ref2.ok) {
    out.error = "reference solve (rtol=" + std::to_string(reference_rtol_2) + "): " + ref2.error;
    return out;
  }

  const auto inf_norm_diff = [](const brusselator::Vector& a, const brusselator::Vector& b) {
    return (a - b).template lpNorm<Eigen::Infinity>();
  };

  out.reference_cross_check_diff =
      std::max(inf_norm_diff(ref1.u_final, ref2.u_final), inf_norm_diff(ref1.v_final, ref2.v_final));
  out.e_n_u = inf_norm_diff(check.u_final, ref1.u_final);
  out.e_n_v = inf_norm_diff(check.v_final, ref1.v_final);
  out.e_n = std::max(out.e_n_u, out.e_n_v);

  // Stability check: the two references
  // must agree to far less than the error about to be measured. A
  // stated threshold -- 1% of e_n -- never loosened to make this
  // pass. If it fails, `ok` stays false and the caller must not build
  // on this measurement.
  out.reference_stable = out.reference_cross_check_diff <= 0.01 * out.e_n;
  if (!out.reference_stable) {
    out.error = "reference solves at rtol=" + std::to_string(reference_rtol_1) + " and rtol=" +
        std::to_string(reference_rtol_2) + " disagree by " +
        std::to_string(out.reference_cross_check_diff) +
        ", not far below the measured e_n=" + std::to_string(out.e_n) +
        " -- the reference has not stabilised; do not calibrate from it";
    return out;
  }

  out.f_final_native = check.residual_history.empty() ? 0.0 : check.residual_history.back();
  out.reference_final_residual = ref1.residual_history.empty() ? 0.0 : ref1.residual_history.back();
  out.check_iterations = check.iterations;
  out.reference_iterations = ref1.iterations;
  out.ok = true;
  return out;
}

// Reference values. `kReferenceRtol1`/`kReferenceRtol2`
// are deliberately far tighter than the check's own `rtol` (NewtonConfig's
// default, 1e-8) -- five and six orders of magnitude tighter. Whether they
// are tight enough is never assumed: `MeasuredErrorAgainstATightReference`
// below RE-DERIVES, every time it runs, whether the two references agree to
// far less than the measured error (`ReferenceCalibration::reference_
// stable`) -- these two numbers are a starting point for that check, not a
// claim about the finite-difference noise floor made in advance.
constexpr double kReferenceRtol1 = 1e-13;
constexpr double kReferenceRtol2 = 1e-14;

}  // namespace

// UNLIKE `ConvergesAgreeingWithNative` below, this
// case never calls `fuse::petsc::install()` or selects
// `NewtonSubstrate::Petsc` -- it exercises ONLY the native substrate, so
// it is safe to run where `MPI_Init()` cannot complete, directly with
// `--gtest_filter=BrusselatorNewtonPetsc.MeasuredErrorAgainstATightReference`
// (the ctest row `petsc.brusselator_newton` runs the WHOLE binary,
// including `ConvergesAgreeingWithNative`, so that row still crashes
// where MPI cannot start).
//
// Reports the SAME calibration `ConvergesAgreeingWithNative` below
// consumes, as its own measurement rather than a value buried inside
// that test's own bound -- `examples/brusselator/agreement_bound.hpp`
// has the full derivation of what `e_n` means and the scaling assumption
// built on top of it.
TEST(BrusselatorNewtonPetsc, MeasuredErrorAgainstATightReference) {
  const ReferenceCalibration calib = measure_native_reference_error(
      /*use_block_jacobi_preconditioner=*/true, kReferenceRtol1, kReferenceRtol2);
  ASSERT_TRUE(calib.ok) << calib.error;

  std::fprintf(stderr,
               "[agreement-bound] check_iters=%d reference_iters=%d "
               "f_final_native=%.6e reference_final_residual=%.6e "
               "reference_cross_check_diff=%.6e e_n_u=%.6e e_n_v=%.6e e_n=%.6e\n",
               calib.check_iterations, calib.reference_iterations, calib.f_final_native,
               calib.reference_final_residual, calib.reference_cross_check_diff, calib.e_n_u,
               calib.e_n_v, calib.e_n);
  EXPECT_GT(calib.e_n, 0.0);
  EXPECT_TRUE(calib.reference_stable);
}

// Needs a host where a bare MPI_Init() completes; it is reached by
// fuse::petsc::ensure_initialized() the first time this test calls
// fuse::petsc::install()/the registry resolves NewtonSubstrate::Petsc.
// Built and registered everywhere FUSE_ENABLE_PETSC is on. Does NOT assert the
// two substrates take the same outer-iteration count (SNES's own GMRES/
// line-search implementation is not required to follow the identical
// path, docs/dev/newton-krylov.md Section 10.3's own measured step-1
// difference) -- only that PETSc converges, and that its final state
// agrees with the native one to the CALIBRATED bound
// (`brusselator_bound::agreement_bound()`, examples/brusselator/
// agreement_bound.hpp): `e_n` MEASURED fresh every time this test runs
// (`measure_native_reference_error()` above, never a hardcoded number
// from a prior session) times a stated scaling assumption (the error of
// a Newton iterate scales with its own terminal residual, both solvers
// near the SAME root under the SAME operator). The asserted bound has no
// separate inter-Krylov term: such a term would measure a SYNTHETIC
// problem's own first-Newton-step solver disagreement, not a bound on two
// CONVERGED Brusselator solutions; `examples/brusselator/agreement_bound.hpp`
// has the full diagnosis. The legacy formula is computed and printed,
// labelled `[agreement-bound legacy]`, purely informationally -- never
// asserted against.
TEST(BrusselatorNewtonPetsc, ConvergesAgreeingWithNative) {
  fuse::petsc::install();

  const RunResult native =
      run_one_window(/*use_block_jacobi_preconditioner=*/true, fuse::NewtonSubstrate::Native);
  ASSERT_TRUE(native.ok) << native.error;

  const RunResult petsc =
      run_one_window(/*use_block_jacobi_preconditioner=*/true, fuse::NewtonSubstrate::Petsc);
  ASSERT_TRUE(petsc.ok) << petsc.error;

  const ReferenceCalibration calib = measure_native_reference_error(
      /*use_block_jacobi_preconditioner=*/true, kReferenceRtol1, kReferenceRtol2);
  ASSERT_TRUE(calib.ok) << calib.error;

  const double f_final_petsc = petsc.residual_history.empty() ? 0.0 : petsc.residual_history.back();
  const double bound = brusselator_bound::agreement_bound(
      calib.e_n, calib.f_final_native, f_final_petsc);
  std::fprintf(stderr,
               "[agreement-bound] e_n=%.6e f_final_native=%.6e f_final_petsc=%.6e "
               "residual_ratio=%.6e bound=%.6e (sharp)\n",
               calib.e_n, calib.f_final_native, f_final_petsc,
               f_final_petsc / calib.f_final_native, bound);

  // INFORMATIONAL ONLY, never asserted: the legacy formula,
  // printed alongside the sharp bound so a host run can report BOTH
  // observed/sharp and observed/legacy -- agreement_bound.hpp's own
  // "LEGACY" section.
  const double legacy_bound = brusselator_bound::legacy_agreement_bound(
      calib.e_n, calib.f_final_native, f_final_petsc, native.u_final.norm());
  std::fprintf(stderr,
               "[agreement-bound legacy] e_n=%.6e f_final_native=%.6e "
               "f_final_petsc=%.6e native_norm=%.6e legacy_bound=%.6e (sharp bound "
               "%.6e; the observed disagreement is the next line)\n",
               calib.e_n, calib.f_final_native, f_final_petsc, native.u_final.norm(),
               legacy_bound, bound);

  // The largest element-wise disagreement over both species, printed before the
  // assertions below so a host run records observed/sharp. The assertions
  // themselves are per element, and a pass says only that this maximum is at
  // most `bound`; without this line a passing run reports no observed value.
  double max_disagreement = 0.0;
  for (Eigen::Index i = 0; i < native.u_final.size(); ++i) {
    max_disagreement = std::max(max_disagreement, std::abs(native.u_final[i] - petsc.u_final[i]));
  }
  for (Eigen::Index i = 0; i < native.v_final.size(); ++i) {
    max_disagreement = std::max(max_disagreement, std::abs(native.v_final[i] - petsc.v_final[i]));
  }
  std::fprintf(stderr,
               "[agreement-bound] observed max |native - petsc| over u_final and v_final = "
               "%.6e  sharp_bound=%.6e  ratio=%.6e  legacy_ratio=%.6e\n",
               max_disagreement, bound, bound > 0.0 ? max_disagreement / bound : 0.0,
               legacy_bound > 0.0 ? max_disagreement / legacy_bound : 0.0);
  for (Eigen::Index i = 0; i < native.u_final.size(); ++i) {
    EXPECT_NEAR(native.u_final[i], petsc.u_final[i], bound) << "u_final[" << i << "]";
  }
  for (Eigen::Index i = 0; i < native.v_final.size(); ++i) {
    EXPECT_NEAR(native.v_final[i], petsc.v_final[i], bound) << "v_final[" << i << "]";
  }
}
