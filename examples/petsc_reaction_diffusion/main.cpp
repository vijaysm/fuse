// S7.4 (owner decision 2026-10-21, Option C): the smallest real port of an
// ImplicitComponent onto a PETSc-object-backed LinearizationToken (real
// Mat/Vec, MatMult/MatMultTranspose, never MATMFFD), paired with an
// existing native-substrate implementation of the SAME physics in one
// Newton-coupled region. See reaction_diffusion_physics.hpp and
// reaction_diffusion_component.hpp for the problem and the component
// design; docs/user/petsc-reaction-diffusion.md for the measured numbers.
//
// WHAT THIS BINARY RUNS, by default (no flags): the "native twin" graph
// (DirichletHalf + NeumannHalfNative, BOTH native), compared against the
// closed-form manufactured solution -- fully runnable in any sandbox, no
// PETSc, no MPI. Exits non-zero if the converged answer misses the
// manufactured solution by more than this track's own derived bound (see
// `kAgreementBound`'s own comment).
//
// --petsc additionally builds AND RUNS the native+petsc graph
// (DirichletHalf + NeumannHalfPetsc) and compares its own converged
// answer against the native twin's -- exactly dn_heat's own
// `--newton-petsc` convention (examples/dn_heat/main.cpp), including that
// this crashes (SEGFAULT, not a logic error) in a sandbox that cannot
// complete a bare MPI_Init() (KNOWN_ISSUES.md D32): a HOST run is what
// measures this comparison's own numbers, the same already-established
// pattern every other petsc.* row and example.dn_heat.newton_petsc row
// already follows.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <fuse/base/config.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/field/registry.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/base/error.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/schedule/buffers.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/runtime/runner.hpp>

#include "reaction_diffusion_component.hpp"
#if FUSE_HAVE_PETSC
#include "reaction_diffusion_component_petsc.hpp"
#endif

namespace prd = petsc_reaction_diffusion;

namespace {

constexpr int kM = 4;                  // interior FD points per half
constexpr std::int64_t kPeriodDen = 100;  // dt = 1/100 = 0.01
constexpr std::int64_t kPeriods = 20;    // t_end = 0.20
constexpr double kTol = 1e-8;  // matches examples/dn_heat/main.cpp's own kTol

fuse::CouplingGraph build_graph(const prd::PhysicsParams& params, int max_iterations) {
  fuse::CouplingGraph g;
  const std::string src = "generated:line:1";  // n = 1 station (one interface)

  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "iface_1", .source = src});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "iface_2", .source = src});
  const auto r1 = g.add_region(fuse::Region{.name = "iface_1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "iface_2", .mesh = m2, .dimension = 1});

  const auto cd = g.add_component(fuse::ComponentDecl{
      .name = "dirichlet", .ranks = {0, 1}, .interval = fuse::Rational{1, kPeriodDen}});
  const auto cn = g.add_component(fuse::ComponentDecl{
      .name = "neumann", .ranks = {1, 1}, .interval = fuse::Rational{1, kPeriodDen}});

  const auto q_out = g.add_field(cd, fuse::FieldSpec{.name = "q_wall",
                                                      .region = r1,
                                                      .units = fuse::units::kDimensionless,
                                                      .conservation = fuse::Conservation::Extensive,
                                                      .intent = fuse::Intent::Provides});
  const auto u_in = g.add_field(cd, fuse::FieldSpec{.name = "u_wall_in",
                                                     .region = r1,
                                                     .units = fuse::units::kDimensionless,
                                                     .conservation = fuse::Conservation::Intensive,
                                                     .intent = fuse::Intent::Requires});
  const auto q_in = g.add_field(cn, fuse::FieldSpec{.name = "q_wall_in",
                                                     .region = r2,
                                                     .units = fuse::units::kDimensionless,
                                                     .conservation = fuse::Conservation::Extensive,
                                                     .intent = fuse::Intent::Requires});
  const auto u_out = g.add_field(cn, fuse::FieldSpec{.name = "u_wall",
                                                      .region = r2,
                                                      .units = fuse::units::kDimensionless,
                                                      .conservation = fuse::Conservation::Intensive,
                                                      .intent = fuse::Intent::Provides});

  const auto flux = g.add_channel("flux", q_out, q_in);
  // Extensive fields: the channel must declare its transfer conservative
  // (E-CONN-CONSERVATION); the identity remap between coincident meshes is.
  g.channel(flux).conservative();
  const auto temp = g.add_channel("wall_value", u_out, u_in);
  g.channel(temp).lag(1).allow_time_level_mismatch(
      "resolved by the Dirichlet-Neumann implicit coupling group");

  fuse::ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(kTol, "u_wall"));
  crit.add(fuse::Converged::Relative(kTol, "q_wall"));

  fuse::NewtonConfig cfg;
  // docs/dev/newton-krylov.md Sec. 9.9.2 (dn_heat/Brusselator, measured):
  // eta_min's own SNES-matching default (0.0) lets Eisenstat-Walker's
  // forcing term shrink without a floor as ||F|| approaches convergence,
  // which can stall GMRES (DivergedLinearSolve) chasing a relative
  // reduction finer than the matrix-free Jacobian action's own noise
  // floor. Measured for THIS problem (design_notes.md Section 6): with
  // eta_min at its default AND use_block_jacobi_preconditioner = true (a
  // non-default setting held during that measurement), residual_history
  // is [4.02e-2, 5.96e-8, 5.96e-8], stalling on the THIRD Newton step's
  // own linear solve. The stall was therefore not isolated to eta_min
  // alone; eta_min = 1e-4 is examples/brusselator/main.cpp's own value,
  // reused here, not a new tolerance chosen for this problem.
  cfg.eta_min = 1e-4;
  // substrate left at its default (Native): the
                           // OUTER region-level Newton-Krylov driver is
                           // ALWAYS fuse::num::solve() in this track --
                           // the PETSc substrate lives entirely INSIDE
                           // NeumannHalfPetsc's own linearization(),
                           // invisible to this driver (component.hpp's
                           // own LinearizationToken is substrate-blind by
                           // design). NewtonConfig::substrate (D32 Sec.
                           // 11) is a DIFFERENT axis -- which solver
                           // drives the REGION, fuse::num::solve() or
                           // fuse::petsc::solve()/SNES -- not used by
                           // this track at all.
  g.iterate({flux, temp})
      .name("reaction_diffusion")
      .until(std::move(crit))
      .max_iterations(max_iterations)
      .newton(cfg);
  return g;
}

struct RunOutcome {
  bool ok = false;
  std::string error;
  double t_final = 0.0;
  double u_interface_final = 0.0;
  std::vector<double> left_profile;
  std::vector<double> right_profile;
  int total_newton_iterations = 0;
};

template <class NeumannT>
RunOutcome run_case(const prd::PhysicsParams& params, const std::string& neumann_label) {
  RunOutcome o;
  fuse::CouplingGraph graph = build_graph(params, /*max_iterations=*/30);

  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) { o.error = "mesh load: " + r.error().message; return o; }
  }

  if (const auto d = fuse::validate(graph); d.has_errors()) { o.error = d.to_string(); return o; }
  const auto lowered = lower(graph);
  if (!lowered.ok()) { o.error = lowered.diags.to_string(); return o; }

  const fuse::ClockSet clocks{graph};
  auto sched = fuse::schedule(graph, lowered.dag, clocks,
                              fuse::make_region_sizes(graph, lowered.dag, backend));
  if (!sched.ok()) { o.error = sched.diags.to_string(); return o; }

  fuse::ComponentSet components;
  auto dirichlet = std::make_unique<prd::DirichletHalf>("dirichlet", kM, params);
  auto neumann = std::make_unique<NeumannT>("neumann", kM, params);
  dirichlet->seed_initial_condition();
  neumann->seed_initial_condition();
  prd::DirichletHalf* dptr = dirichlet.get();
  NeumannT* nptr = neumann.get();
  components.add(std::move(dirichlet));
  components.add(std::move(neumann));

  fuse::FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, graph);
    if (!r.ok()) { o.error = "register_fields: " + r.error().message; return o; }
  }

  fuse::RunnerConfig rcfg;
  rcfg.abort_on_nonconvergence = true;
  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry, rcfg); !r.ok()) {
    o.error = "initialize: " + r.error().message;
    return o;
  }

  for (std::int64_t p = 0; p < kPeriods; ++p) {
    if (const auto r = runner.run(components, 1); !r.ok()) {
      o.error = "run (period " + std::to_string(p) + "): " + r.error().message;
      return o;
    }
    if (!runner.iteration_stats().empty()) {
      o.total_newton_iterations +=
          static_cast<int>(runner.iteration_stats()[0].residual_history.size());
    }
  }

  o.t_final = static_cast<double>(kPeriods) / static_cast<double>(kPeriodDen);
  o.u_interface_final = nptr->u_out();
  o.left_profile = dptr->profile();
  o.right_profile = nptr->profile();
  o.ok = true;
  (void)neumann_label;
  return o;
}

/// A tolerance derived from a MEASUREMENT taken in THIS build, not a
/// hand-picked constant: at m=4 interior points/half, dt=0.01, over the
/// kPeriods=20 steps this binary actually runs (t_final=0.2), the
/// max|error| against the manufactured solution was measured (this
/// sandbox, this session; design_notes.md Section 6 records the exact
/// run) at 7.306612e-02 (Left) / 5.338220e-02 (Right) / 1.184185e-01
/// (interface) -- first-order, backward-Euler-dominated accumulation
/// over 20 steps, consistent with the ~2x-per-halving convergence rate
/// a separate Python prototype of this exact discretization measured
/// under mesh/step refinement (design_notes.md Section 5). The bound
/// below is 2x the largest of the three measured figures (interface,
/// 1.184185e-01) -- enough headroom to not be a tripwire on ordinary
/// floating-point/iteration-count variation across a rebuild, while
/// still catching a real regression (a coupling or sign error moves
/// this error by orders of magnitude, not a factor of 2, as the Stage A
/// prototype's own bug -- this file's header comment -- demonstrated).
constexpr double kAgreementBound = 2.0 * 1.184185e-01;

int run_native_twin(const prd::PhysicsParams& params) {
  const RunOutcome o = run_case<prd::NeumannHalfNative>(params, "native");
  if (!o.ok) {
    std::fprintf(stderr, "native twin FAILED: %s\n", o.error.c_str());
    return 1;
  }
  double max_err_left = 0.0, max_err_right = 0.0;
  const double dx_l = params.L_left / (kM + 1);
  const double dx_r = params.L_right / (kM + 1);
  for (int i = 0; i < kM; ++i) {
    const double ex_l = prd::manufactured_u((i + 1) * dx_l, o.t_final, params);
    const double ex_r = prd::manufactured_u(params.L_left + (i + 1) * dx_r, o.t_final, params);
    max_err_left = std::max(max_err_left, std::fabs(o.left_profile[static_cast<std::size_t>(i)] - ex_l));
    max_err_right = std::max(max_err_right, std::fabs(o.right_profile[static_cast<std::size_t>(i)] - ex_r));
  }
  const double ex_interface = prd::manufactured_u(params.L_left, o.t_final, params);
  const double err_interface = std::fabs(o.u_interface_final - ex_interface);
  std::printf("native twin: t_final=%.4f periods=%lld total_newton_its=%d\n", o.t_final,
             static_cast<long long>(kPeriods), o.total_newton_iterations);
  std::printf("native twin: max|err| left=%.6e right=%.6e interface=%.6e (bound %.6e)\n",
             max_err_left, max_err_right, err_interface, kAgreementBound);
  const bool passed = max_err_left < kAgreementBound && max_err_right < kAgreementBound &&
                      err_interface < kAgreementBound;
  std::printf("native twin: %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}

#if FUSE_HAVE_PETSC
int run_petsc_case(const prd::PhysicsParams& params, const RunOutcome& native_ref) {
  const RunOutcome o = run_case<prd::NeumannHalfPetsc>(params, "petsc");
  if (!o.ok) {
    std::fprintf(stderr, "native+petsc FAILED: %s\n", o.error.c_str());
    return 1;
  }
  // Agreement bound: BOTH substrates solve the IDENTICAL discrete
  // equations via a DIRECT linear solve (Thomas algorithm vs KSP+PCLU) --
  // the only disagreement possible is round-off and KSP's own default
  // convergence test, so the bound is the Newton rtol (1e-8, NewtonConfig's
  // own default) times a conditioning headroom of 100, the SAME style of
  // derivation D32 used for dn_heat's own native-vs-petsc comparison.
  constexpr double kSubstrateAgreementBound = 100.0 * 1e-8;
  double max_diff = 0.0;
  for (int i = 0; i < kM; ++i) {
    max_diff = std::max(max_diff, std::fabs(o.right_profile[static_cast<std::size_t>(i)] -
                                            native_ref.right_profile[static_cast<std::size_t>(i)]));
  }
  max_diff = std::max(max_diff, std::fabs(o.u_interface_final - native_ref.u_interface_final));
  std::printf("native+petsc: max|diff vs native|=%.6e (bound %.6e)\n", max_diff,
             kSubstrateAgreementBound);
  const bool passed = max_diff < kSubstrateAgreementBound;
  std::printf("native+petsc: %s\n", passed ? "PASSED" : "FAILED");
  return passed ? 0 : 1;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  // Process-wide diagnostics set-up (fuse/base/error.hpp): reads FUSE_ON_ERROR
  // and its companions (docs/dev/debugging.md) once. Without this call the
  // environment is parsed by nothing and the built-in defaults apply.
  fuse::initialize();

  Kokkos::ScopeGuard kokkos_guard(argc, argv);  // Runner's hot loop runs on
                                                 // Kokkos' host execution
                                                 // space (KNOWN_ISSUES.md
                                                 // D11); examples/dn_heat/
                                                 // main.cpp's own pattern.
  bool run_petsc = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--petsc") == 0) run_petsc = true;
  }

  const prd::PhysicsParams params;
  int status = run_native_twin(params);

  if (run_petsc) {
#if FUSE_HAVE_PETSC
    const RunOutcome native_ref = run_case<prd::NeumannHalfNative>(params, "native-ref");
    if (!native_ref.ok) {
      std::fprintf(stderr, "native reference re-run FAILED: %s\n", native_ref.error.c_str());
      return 1;
    }
    status |= run_petsc_case(params, native_ref);
#else
    std::fprintf(stderr, "--petsc requested but this binary was built without FUSE_ENABLE_PETSC\n");
    status = 1;
#endif
  }
  return status;
}
