// S7.2b (PetscSelect track): the Brusselator's own window-Newton region
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
// which calls MPI_Init() when nothing has -- and in THIS sandbox (KNOWN_
// ISSUES.md D32) that SEGFAULTS the whole process, not merely fails one
// gtest case. Measured directly: an earlier draft of this track put this
// exact case inside brusselator_runner_test.cpp, and running it there
// took down EVERY other case in that binary (JacobiMatchesDriver,
// NewtonConvergesInAPinnedIterationCountWithTheBlockJacobiPreconditioner,
// etc. -- otherwise-passing, PETSc-independent coverage) via one shared-
// process crash. Every existing petsc.* test (test/petsc/{init,token,
// derivative_gate,snes}_test.cpp) is ALREADY its own dedicated binary for
// precisely this reason (token.hpp's own comment on why init_test.cpp in
// particular is never shared); this file follows the SAME convention
// rather than reproducing the collateral-damage defect the first draft
// had. `fuse_add_gtest(petsc brusselator_newton ...)` (test/CMakeLists.txt)
// registers it as its own process, `petsc.brusselator_newton`.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/petsc/solve.hpp>
#include <fuse/runtime/runner.hpp>

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

[[nodiscard]] RunResult run_one_window(bool use_block_jacobi_preconditioner,
                                       fuse::NewtonSubstrate substrate) {
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
  auto sched = schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, rig->backend));
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
  out.ok = true;
  out.u_final = pu->state();
  out.v_final = pv->state();
  return out;
}

}  // namespace

// HOST-ONLY: this sandbox cannot complete a bare MPI_Init() (KNOWN_
// ISSUES.md D32), reached by fuse::petsc::ensure_initialized() the first
// time this test calls fuse::petsc::install()/the registry resolves
// NewtonSubstrate::Petsc. Built and registered everywhere FUSE_ENABLE_
// PETSC is on; not expected to pass IN THIS SANDBOX. Does NOT assert the
// two substrates take the same outer-iteration count (SNES's own GMRES/
// line-search implementation is not required to follow the identical
// path, docs/dev/newton-krylov.md §10.3's own measured step-1
// difference) -- only that PETSc converges, and that its final state
// agrees with the native one to a bound derived from kTol and the
// §10.3-measured 3.9e-8 relative inter-solver difference (the SAME
// formula examples/dn_heat/main.cpp's own --newton-petsc path uses).
TEST(BrusselatorNewtonPetsc, ConvergesAgreeingWithNative) {
  fuse::petsc::install();

  const RunResult native =
      run_one_window(/*use_block_jacobi_preconditioner=*/true, fuse::NewtonSubstrate::Native);
  ASSERT_TRUE(native.ok) << native.error;

  const RunResult petsc =
      run_one_window(/*use_block_jacobi_preconditioner=*/true, fuse::NewtonSubstrate::Petsc);
  ASSERT_TRUE(petsc.ok) << petsc.error;

  const double rtol_term = 20.0 * 1.5 * 2.0 * kTol * native.u_final.norm();
  const double gmres_term = 3.9e-8 * native.u_final.norm();
  const double bound = rtol_term + gmres_term;
  for (Eigen::Index i = 0; i < native.u_final.size(); ++i) {
    EXPECT_NEAR(native.u_final[i], petsc.u_final[i], bound) << "u_final[" << i << "]";
  }
  for (Eigen::Index i = 0; i < native.v_final.size(); ++i) {
    EXPECT_NEAR(native.v_final[i], petsc.v_final[i], bound) << "v_final[" << i << "]";
  }
}
