// Multirate 2D Brusselator: U and V on independent, non-nested native
// meshes ("generated:quad:NxM"), coupled at independent time steps through
// fuse::WindowCoupler, exchanging through FUSE's conservative supermesh
// remap weights (fuse::build_weights / fuse::apply_weights). Ported from
// the user's independently-written Toy (/Users/mahadevan/Code/AI/Toy) --
// see brusselator_physics.hpp and species_component.hpp for what was
// preserved exactly and what necessarily differs, and
// docs/user/brusselator.md for the full account and the measured numbers.
//
// --check runs the budget/exchange/positivity/convergence gate described
// in docs/user/brusselator.md and this file's run_check(); the deep
// verification (weights-vs-Toy-oracle, invariants, cross-code agreement,
// convergence order) lives in test/verification/brusselator_*_test.cpp.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/base/config.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/base/error.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/runtime/window_coupler.hpp>

// S7.2b (PetscSelect track): see examples/dn_heat/main.cpp's own identical
// comment on this pair -- fuse/petsc/solve.hpp names no PETSc type and
// compiles either way, but fuse::petsc::install() is defined only in
// fuse-coupler-petsc (FUSE_ENABLE_PETSC), so both the include and every
// call are guarded by the same macro.
#if FUSE_HAVE_PETSC
#include <fuse/petsc/solve.hpp>
#endif

#include "accel_names.hpp"
#include "brusselator_graph.hpp"
#include "brusselator_physics.hpp"
#include "species_component.hpp"

using fuse::AccelConfig;
using fuse::AccelKind;
using fuse::DtPolicy;
using fuse::Rational;
using fuse::WindowChannel;
using fuse::WindowCoupler;
using fuse::WindowCouplerConfig;
using fuse::WindowMember;

namespace {

struct Args {
  int nx_u = 16, ny_u = 16, nx_v = 24, ny_v = 24;
  double dt_u = 0.002, dt_v = 0.003, window = 0.02, end = 11.5, alpha = 10.0;
  double tolerance = 1e-11;
  int max_iterations = 60;
  bool forcing = true;
  bool save_history = false;
  bool check = false;
  std::string accel = "none";
  std::string out = "results";
  /// "window" (default, unchanged): `fuse::WindowCoupler`, driven directly
  /// as this example always has. "runner": the SAME graph compiled
  /// through `fuse::Runner`, the Time windows track's compiled path
  /// (docs/user/coupling-windows.md "Through schedule() and the
  /// Runner") -- see main()'s own comment at the branch.
  std::string driver = "window";
  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.6): "jacobi" (default,
  /// unchanged -- every window channel lagged) or "gauss-seidel" (the
  /// v2u channel lagged, u2v resolved within the sweep it is fed from --
  /// `brusselator_graph.hpp::add_window_coupling()`'s own `iteration_
  /// mode` parameter, never exposed on this CLI before this track).
  /// Read by BOTH drivers: `--driver window` builds its own
  /// `WindowCouplerConfig::iteration` from it directly; `--driver
  /// runner` threads it into the graph.
  std::string iteration_mode = "jacobi";
  /// S7.2 WindowNewton (newton-krylov.md Sec. 9.5): whether `--accel
  /// newton` additionally asks for the per-species block-Jacobi
  /// preconditioner (`NewtonConfig::use_block_jacobi_preconditioner`).
  /// Defaulting to ON (not NewtonConfig's own OFF default) is this
  /// example's own choice, not the library's: demonstrating the
  /// physics-based preconditioner IS this track's own point, so a
  /// reader who types `--accel newton` and nothing else sees it;
  /// `--no-block-jacobi` reproduces the plain, unpreconditioned solve
  /// for comparison.
  bool block_jacobi = true;
  /// S7.2b (PetscSelect track): "native" (default, unchanged -- every
  /// existing invocation of this binary keeps running the library's own
  /// JFNK coupler) or "petsc" (`NewtonConfig::substrate`, read only
  /// when `--accel newton` is also selected -- the auto-trigger that
  /// sets `requires_residual_capability` does not depend on this field
  /// at all, so an invalid combination such as `--substrate petsc
  /// --accel aitken` simply leaves the substrate choice unread, the
  /// same "declared but unused until Newton is selected" convention
  /// `NewtonConfig` itself already documents).
  std::string substrate = "native";
};

[[nodiscard]] Args parse_args(int argc, char** argv) {
  Args a;
  const auto next = [&](int& i) -> std::string {
    return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--nx-u") a.nx_u = std::atoi(next(i).c_str());
    else if (arg == "--ny-u") a.ny_u = std::atoi(next(i).c_str());
    else if (arg == "--nx-v") a.nx_v = std::atoi(next(i).c_str());
    else if (arg == "--ny-v") a.ny_v = std::atoi(next(i).c_str());
    else if (arg == "--dt-u") a.dt_u = std::atof(next(i).c_str());
    else if (arg == "--dt-v") a.dt_v = std::atof(next(i).c_str());
    else if (arg == "--window") a.window = std::atof(next(i).c_str());
    else if (arg == "--end") a.end = std::atof(next(i).c_str());
    else if (arg == "--alpha") a.alpha = std::atof(next(i).c_str());
    else if (arg == "--tolerance") a.tolerance = std::atof(next(i).c_str());
    else if (arg == "--max-iterations") a.max_iterations = std::atoi(next(i).c_str());
    else if (arg == "--no-forcing") a.forcing = false;
    else if (arg == "--save-history") a.save_history = true;
    else if (arg == "--check") a.check = true;
    else if (arg == "--accel") a.accel = next(i);
    else if (arg == "--out") a.out = next(i);
    else if (arg == "--driver") a.driver = next(i);
    else if (arg == "--iteration-mode") a.iteration_mode = next(i);
    else if (arg == "--no-block-jacobi") a.block_jacobi = false;
    else if (arg == "--substrate") a.substrate = next(i);
    else if (arg == "--help") {
      std::cout << "Couple independent U/V native-mesh solvers with FUSE's conservative "
                   "supermesh remap and fuse::WindowCoupler.\n"
                   "--nx-u 16 --ny-u 16 --nx-v 24 --ny-v 24\n"
                   "--dt-u 0.002 --dt-v 0.003 --window 0.02 --end 11.5 --alpha 10\n"
                   "--tolerance 1e-11 --max-iterations 60 --out results\n"
                   "--accel none|constant|aitken|secant|steffensen|iqn-ils|iqn-imvj|newton "
                   "(default none; an unknown name is refused)\n"
                   "--no-forcing --save-history --check\n"
                   "--driver window|runner (default window; runner requires every "
                   "window the same length)\n"
                   "--iteration-mode jacobi|gauss-seidel (default jacobi)\n"
                   "--no-block-jacobi (--accel newton's own default is ON)\n"
                   "--substrate native|petsc (--accel newton only; default native; "
                   "petsc needs a build configured with -DFUSE_ENABLE_PETSC=ON)\n";
      std::exit(0);
    } else {
      std::fprintf(stderr, "FAIL: unknown argument '%s'\n", arg.c_str());
      std::exit(1);
    }
  }
  return a;
}

/// `--accel`'s names: the shared table (examples/common/accel_names.hpp)
/// plus `newton`. A name in neither is `std::nullopt`; main() refuses it
/// before any mesh is built (E-EXAMPLE-ACCEL). Until 2026-10-20 an
/// unrecognised name silently ran with no acceleration.
[[nodiscard]] std::optional<AccelKind> try_parse_accel(const std::string& s) {
  // S7.2 WindowNewton (newton-krylov.md Sec. 7): `AccelKind::Newton`,
  // never routed through `make_accelerator()` -- `resolve_regions()`'s
  // own `if (decl.acceleration.kind == AccelKind::Newton)` branch takes
  // the Newton path before `R.accel` is ever touched (iteration.cpp).
  if (s == "newton") return AccelKind::Newton;
  return fuse_examples::accel_kind_from_name(s);
}

/// The validated form: every call site runs after main()'s check, so the
/// fallback below is unreachable and is a refusal rather than a default.
[[nodiscard]] AccelKind parse_accel(const std::string& s) {
  if (const auto k = try_parse_accel(s)) return *k;
  std::fprintf(stderr, "FAIL: E-EXAMPLE-ACCEL: unknown --accel '%s'\n", s.c_str());
  std::exit(1);
}

/// S7.2 WindowNewton (newton-krylov.md Sec. 9.6): the `--iteration-mode`
/// flag's own parser -- `fuse::WaveformIteration::Jacobi` is the default
/// for any value other than exactly "gauss-seidel", matching every other
/// flag on this CLI's own "unrecognised value falls back to the default"
/// convention (`parse_accel`'s own identical shape, above).
[[nodiscard]] fuse::WaveformIteration parse_iteration_mode(const std::string& s) {
  if (s == "gauss-seidel" || s == "gaussseidel" || s == "gs") {
    return fuse::WaveformIteration::GaussSeidel;
  }
  return fuse::WaveformIteration::Jacobi;
}

/// S7.2b (PetscSelect track): the `--substrate` flag's own parser, same
/// "unrecognised value falls back to the default" convention as
/// `parse_accel()`/`parse_iteration_mode()` above -- anything other
/// than exactly "petsc" is `NewtonSubstrate::Native`.
[[nodiscard]] fuse::NewtonSubstrate parse_substrate(const std::string& s) {
  if (s == "petsc") return fuse::NewtonSubstrate::Petsc;
  return fuse::NewtonSubstrate::Native;
}

/// Decimal -> exact Rational at a fixed denominator, exactly the
/// examples/gray_scott/main.cpp convention (its own comment explains why
/// every FUSE time value must be exact): every CLI time is converted once,
/// here, and no downstream code sees a double time.
[[nodiscard]] Rational to_rational(double v) {
  constexpr std::int64_t kDen = 1000000;
  return Rational{static_cast<std::int64_t>(std::llround(v * kDen)), kDen};
}

/// Window boundaries over [0, end]: 0, window, 2*window, ... with the last
/// one before `end` possibly shortened to land on `end` exactly, AND with
/// t=1.1 inserted as an extra boundary if it is not already one -- the
/// Toy's own move (src/brusselator.cpp `simulate`'s `windows.insert`), so
/// the forcing switch is a window boundary here too. Unlike the Toy, this
/// needs no epsilon snapping: Rational equality is exact, so a boundary
/// that already lands on 1.1 (e.g. the default --window 0.02, for which
/// 55*0.02 == 1.1 exactly) is detected exactly, not to a tolerance.
[[nodiscard]] std::vector<Rational> window_boundaries(Rational window, Rational end) {
  std::vector<Rational> b{Rational{0}};
  while (b.back() < end) {
    Rational next = b.back() + window;
    if (next > end) next = end;
    b.push_back(next);
  }
  const Rational switch_t{11, 10};
  if (switch_t > Rational{0} && switch_t < end &&
      std::find(b.begin(), b.end(), switch_t) == b.end()) {
    b.insert(std::upper_bound(b.begin(), b.end(), switch_t), switch_t);
  }
  return b;
}

void write_field_csv(const std::string& path, const brusselator::NativeGrid& grid,
                     const brusselator::Vector& field, double time) {
  std::ofstream out(path);
  out << std::setprecision(17) << "time,x,y,value\n";
  for (int j = 0; j < grid.ny(); ++j) {
    for (int i = 0; i < grid.nx(); ++i) {
      out << time << ',' << grid.x(i) << ',' << grid.y(j) << ',' << field[grid.index(i, j)] << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse_args(argc, argv);
  if (!try_parse_accel(args.accel)) {
    std::fprintf(stderr,
                 "FAIL: E-EXAMPLE-ACCEL: unknown --accel '%s'; use one of %.*s|newton\n",
                 args.accel.c_str(), static_cast<int>(fuse_examples::kAccelNames.size()),
                 fuse_examples::kAccelNames.data());
    return 1;
  }
  if (args.driver != "window" && args.driver != "runner") {
    std::fprintf(stderr, "FAIL: --driver must be 'window' or 'runner', got '%s'\n",
                 args.driver.c_str());
    return 1;
  }
  // Moved up from just above the window loop (unchanged computation): the
  // `--driver runner` branch needs `window` to declare the coupling
  // graph's own `CouplingWindowScope::explicit_interval` before any grid
  // or weight setup, not only to walk `boundaries` afterwards.
  const Rational window = to_rational(args.window);
  const Rational end = to_rational(args.end);

  // Process-wide diagnostics set-up (fuse/base/error.hpp): reads FUSE_ON_ERROR
  // and its companions (docs/dev/debugging.md) once. Without this call the
  // environment is parsed by nothing and the built-in defaults apply.
  fuse::initialize();

  // Both the native mesh backend's own Kokkos::View allocations
  // (MeshArrays<Kokkos::HostSpace>) and this example's own reaction kernel
  // (brusselator_physics.hpp's `reaction_kokkos`) need Kokkos initialised
  // first. KNOWN_ISSUES.md D11 (closing): fuse_example_gray_scott/mms_pair/
  // dn_heat/fsi_channel now ALSO need this -- the Runner's own coupling hot
  // loop moved onto Kokkos' default host execution space
  // (fuse/num/space_kernels.hpp), so every one of them gained its own
  // `Kokkos::ScopeGuard` at the same place this one has always had it.
  // Same pattern as test/support/fuse_test_main.cpp's.
  Kokkos::ScopeGuard kokkos_guard(argc, argv);

  // S7.2b (PetscSelect track): registers NewtonSubstrate::Petsc against
  // fuse::petsc::solve() -- a no-op for every `--substrate native`
  // invocation (every existing ctest row), and what makes `--substrate
  // petsc --accel newton` resolve instead of refusing with
  // E-NEWTON-PETSC-UNAVAILABLE. Guarded by FUSE_HAVE_PETSC the same way
  // dn_heat/main.cpp's own identical call is -- the symbol does not
  // exist to link against otherwise.
#if FUSE_HAVE_PETSC
  fuse::petsc::install();
#endif

  fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/1);
  auto u_grid_r = brusselator::NativeGrid::make(backend, "u", args.nx_u, args.ny_u);
  if (!u_grid_r.ok()) {
    std::fprintf(stderr, "FAIL: U grid: %s: %s\n", u_grid_r.error().code.c_str(),
                 u_grid_r.error().message.c_str());
    return 1;
  }
  auto v_grid_r = brusselator::NativeGrid::make(backend, "v", args.nx_v, args.ny_v);
  if (!v_grid_r.ok()) {
    std::fprintf(stderr, "FAIL: V grid: %s: %s\n", v_grid_r.error().code.c_str(),
                 v_grid_r.error().message.c_str());
    return 1;
  }
  brusselator::NativeGrid u_grid = std::move(u_grid_r.value());
  brusselator::NativeGrid v_grid = std::move(v_grid_r.value());
  const int n_u = u_grid.size(), n_v = v_grid.size();

  // The graph's own MeshDecl.source is not consulted by build_weights --
  // it resolves the region's mesh NAME through `backend` directly (see
  // supermesh_side() in src/backend/intx_weights.cpp), and `backend`
  // already has "u"/"v" loaded from NativeGrid::make above. Recorded
  // here anyway so the graph is self-describing.
  fuse::CouplingGraph graph;
  const fuse::MeshId u_mesh = graph.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(args.nx_u) + "x" + std::to_string(args.ny_u)});
  const fuse::MeshId v_mesh = graph.add_mesh(
      {.name = "v", .source = "generated:quad:" + std::to_string(args.nx_v) + "x" + std::to_string(args.ny_v)});
  const fuse::RegionId u_region = graph.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  const fuse::RegionId v_region = graph.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  // U, V, their two window-transfer fields (u_out/v_out, matching the
  // --driver window path's own WindowChannel names below exactly) and the
  // coupling-window iteration region (brusselator_graph.hpp) -- declared
  // on the graph UNCONDITIONALLY, so the graph is self-describing either
  // way, but only actually compiled and run when --driver runner is
  // selected (see main()'s own branch, below the window loop setup). The
  // --driver window path never consults these declarations: WindowCoupler
  // is handed `mu`/`mv`/`channels` directly and never looks at `graph`.
  // S7.2 WindowNewton (newton-krylov.md Sec. 9.6): `newton_cfg.use_block_
  // jacobi_preconditioner` is read by `add_window_coupling()` only when
  // `parse_accel(args.accel) == AccelKind::Newton` (that function's own
  // auto-trigger) -- harmlessly ignored otherwise, so passing it
  // unconditionally changes nothing for every other `--accel` value.
  fuse::NewtonConfig newton_cfg;
  newton_cfg.use_block_jacobi_preconditioner = args.block_jacobi;
  // `eta_min`'s own default (0.0, SNES's own) lets Eisenstat-Walker Choice2
  // tighten the forcing term without a floor -- fine for a single-evaluation
  // `G`, but this region's own `F` composes MFFD over a WHOLE window's worth
  // of substeps (11 U-steps, 8 V-steps at the defaults), and the forcing
  // term's own quadratic tightening (`forcing_alpha = 2.0`) reaches a
  // demanded GMRES relative tolerance below MFFD's finite-difference noise
  // floor in as few as 3 Newton steps at these defaults (observed, FUSE_
  // NEWTON_DEBUG trace, with `eta_min` still at its 0.0 default: eta
  // reaching ~6.6e-9 for the 4th step while ||F|| was already at ~9e-8
  // relative to ||F_0|| -- GMRES's own convergence test
  // (`E-LINALG-NO-CONVERGENCE`) then fails to reach that eta -- the trace
  // records 0 completed outer GMRES iterations against a demanded
  // tolerance the finite-difference Jacobian action cannot supply at that
  // scale, despite 1208 MFFD evaluations spent trying -- reported as
  // `DivergedLinearSolve` even though the nonlinear iterate itself was
  // already converged for every practical purpose). A floor here is
  // exactly the caller-set option `NewtonConfig::eta_min`'s own doc
  // comment (`fuse/model/iteration.hpp`) names for this case. `1e-4`
  // (two orders of magnitude above the observed stagnation point) was
  // chosen over a tighter floor such as `1e-6` after measuring both:
  // `1e-6` also converges every window (575/575 windows, `--end 11.5`,
  // confirmed without divergence over the first 222 before being cut
  // short for runtime) but spends 70-110 GMRES iterations on the last
  // Newton step of every one of those 222 windows chasing a linear-solve accuracy the
  // outer nonlinear test does not need; `1e-4` converges the same windows
  // in the same 3-4 Newton steps at a fraction of the GMRES cost (as low
  // as 2-6 iterations per step once the trajectory settles into its
  // oscillatory regime, `--end 1.0` measurement, 50/50 windows).
  newton_cfg.eta_min = 1e-4;
  // S7.2b (PetscSelect track): read only when --accel newton is also
  // selected (NewtonConfig's own "declared but unused until Newton is
  // selected" convention) -- a region naming NewtonSubstrate::Petsc with
  // nothing registered for it (no FUSE_ENABLE_PETSC, or this process
  // never called fuse::petsc::install()) refuses at Runner::initialize()
  // with E-NEWTON-PETSC-UNAVAILABLE, caught by the `!r.ok()` checks every
  // driver branch below already has for every OTHER initialize() failure.
  newton_cfg.substrate = parse_substrate(args.substrate);
  [[maybe_unused]] const brusselator::WindowGraphHandles window_graph =
      brusselator::add_window_coupling(graph, u_region, to_rational(args.dt_u),
                                       to_rational(args.dt_v), window, args.tolerance,
                                       args.max_iterations, parse_accel(args.accel),
                                       parse_iteration_mode(args.iteration_mode),
                                       /*coupler_ranks=*/std::nullopt,
                                       /*u_ranks=*/fuse::RankSet{0, 1},
                                       /*v_ranks=*/fuse::RankSet{0, 1},
                                       /*u_launch=*/fuse::LaunchSpec{},
                                       /*v_launch=*/fuse::LaunchSpec{},
                                       /*coupler_launch=*/fuse::LaunchSpec{}, newton_cfg);

  fuse::RemapPlan plan_vu;  // V (source) -> U (target): row=U native idx, col=V native idx
  plan_vu.src = v_region;
  plan_vu.tgt = u_region;
  auto w_vu = fuse::build_weights(plan_vu, graph, backend);
  if (!w_vu.ok()) {
    std::fprintf(stderr, "FAIL: build_weights(V->U): %s: %s\n", w_vu.error().code.c_str(),
                 w_vu.error().message.c_str());
    return 1;
  }
  fuse::RemapPlan plan_uv;  // U (source) -> V (target): row=V native idx, col=U native idx
  plan_uv.src = u_region;
  plan_uv.tgt = v_region;
  auto w_uv = fuse::build_weights(plan_uv, graph, backend);
  if (!w_uv.ok()) {
    std::fprintf(stderr, "FAIL: build_weights(U->V): %s: %s\n", w_uv.error().code.c_str(),
                 w_uv.error().message.c_str());
    return 1;
  }
  if (w_vu.value().max_row_defect > 1e-10 || std::abs(w_vu.value().src_measure_total - 1.0) > 1e-10 ||
      w_uv.value().max_row_defect > 1e-10) {
    std::fprintf(stderr,
                 "FAIL: E-BRUSS-REMAP-NOT-CONSERVATIVE: max_row_defect V->U=%.3e U->V=%.3e "
                 "(both meshes tile the same unit square, so both must hold to ~1e-12)\n",
                 w_vu.value().max_row_defect, w_uv.value().max_row_defect);
    return 1;
  }

  brusselator::PhysicsConfig phys_cfg;
  phys_cfg.diffusion = args.alpha;
  phys_cfg.forcing = args.forcing;

  // S7.2 WindowNewton (newton-krylov.md Sec. 9.5): the window metadata
  // `UComponent`/`VComponent`'s own `evaluate_residual()`/`linearization()`
  // need -- harmless to pass unconditionally (every pre-existing `--accel`
  // value never calls either; `capabilities()` gates on it being set, not
  // on which `--accel` was chosen).
  auto comp_v = std::make_unique<brusselator::VComponent>(
      "V", u_grid, v_grid, phys_cfg, &w_uv.value(), &w_vu.value(), to_rational(args.dt_v),
      to_rational(args.dt_u), window);
  brusselator::Vector v0_on_u = Eigen::Map<brusselator::Vector>(comp_v->provided_ptr(), n_u);
  auto comp_u = std::make_unique<brusselator::UComponent>(
      "U", u_grid, phys_cfg, v0_on_u, to_rational(args.dt_u), to_rational(args.dt_v), window);
  // Raw observers, valid regardless of which driver below ends up OWNING
  // the pointees: `--driver window` never moves `comp_u`/`comp_v` out of
  // this scope; `--driver runner` moves them into a `fuse::ComponentSet`,
  // which outlives every later read through these pointers (`components`
  // is declared in that branch and not destroyed until main() returns).
  brusselator::UComponent* const pu = comp_u.get();
  brusselator::VComponent* const pv = comp_v.get();

  WindowMember mu;
  mu.component = comp_u.get();
  mu.name = "U";
  mu.interval = to_rational(args.dt_u);
  mu.policy = DtPolicy::Adjustable;
  mu.provided = comp_u->provided_ptr();
  mu.required = comp_u->required_ptr();
  mu.length = static_cast<std::size_t>(n_u);

  WindowMember mv;
  mv.component = comp_v.get();
  mv.name = "V";
  mv.interval = to_rational(args.dt_v);
  mv.policy = DtPolicy::Adjustable;
  mv.provided = comp_v->provided_ptr();
  mv.required = comp_v->required_ptr();
  mv.length = static_cast<std::size_t>(n_u);  // both channels live on the U grid; see NUMERICS.

  const auto boundaries = window_boundaries(window, end);

  double max_window_balance_error = 0.0, max_window_exchange_error = 0.0;
  int max_iterations_used = 0;
  long long total_windows = 0;
  const double initial_mass = pu->state().mean() + pv->state().mean();
  double mean_u = pu->state().mean(), mean_v = pv->state().mean();

  std::ofstream window_log;
  if (args.save_history) {
    std::filesystem::create_directories(args.out);
    window_log.open(std::filesystem::path(args.out) / "windows.csv");
    window_log << std::setprecision(17) << "t0,t1,mean_u,mean_v,iterations\n";
  }

  // Declared here, OUTSIDE the --driver runner branch below, even though
  // only that branch populates it: `pu`/`pv` keep pointing at
  // `comp_u`/`comp_v` after that branch moves them into `components`
  // (ComponentSet owns them from that point on), and the post-loop
  // diagnostics/output code below -- shared by both drivers -- reads
  // `pu->state()`/`pv->state()` after the branch's own closing brace. A
  // `components` scoped to the branch itself would destroy the owner
  // there and leave `pu`/`pv` dangling for every line after it.
  fuse::ComponentSet components;

  if (args.driver == "window") {
    std::size_t i = 0;
    while (i + 1 < boundaries.size()) {
      const Rational len = boundaries[i + 1] - boundaries[i];
      std::size_t j = i;
      while (j + 1 < boundaries.size() && boundaries[j + 1] - boundaries[j] == len) ++j;
      const int repeats = static_cast<int>(j - i);

      WindowCouplerConfig cfg;
      cfg.window_length = len;
      cfg.max_iterations = args.max_iterations;
      cfg.acceleration = AccelConfig{.kind = parse_accel(args.accel)};
      // S7.2 WindowNewton (newton-krylov.md Sec. 9.6): --accel newton is
      // Runner-only (make_accelerator() has no case for it, and
      // WindowCoupler never gained a Newton solve of its own) -- a
      // caller asking for both refuses cleanly, with E-ACCEL-KIND, from
      // inside WindowCoupler::create() below; nothing added here makes
      // that combination succeed. (An unknown --accel name never gets this
      // far: main() refuses it with E-EXAMPLE-ACCEL.)
      const fuse::WaveformIteration iteration_mode = parse_iteration_mode(args.iteration_mode);
      cfg.iteration = iteration_mode;
      // The user's decision (relayed 2026-09-29): a convergence measure
      // names a field TRANSFER (a WindowChannel, by its own `name`), not
      // the whole coupled system. Split into the two real transfers, same
      // kind/tolerance as the single aggregate measure this replaces; both
      // must be satisfied. Not byte-identical to before by the user's own
      // choice -- see this program's before/after diagnostics.
      fuse::ConvergenceMeasure measure_u;
      measure_u.kind = fuse::MeasureKind::Relative;
      measure_u.tolerance = args.tolerance;
      measure_u.field = "u_out";
      cfg.convergence.add(measure_u);
      fuse::ConvergenceMeasure measure_v;
      measure_v.kind = fuse::MeasureKind::Relative;
      measure_v.tolerance = args.tolerance;
      measure_v.field = "v_out";
      cfg.convergence.add(measure_v);
      const double min_dt = std::min(args.dt_u, args.dt_v);
      cfg.waveform_capacity =
          static_cast<std::size_t>(std::ceil(len.to_double() / std::max(min_dt, 1e-9))) + 8;

      // S7.2 WindowNewton (newton-krylov.md Sec. 9.6): under
      // GaussSeidel, "v_out" (V -> U) is the lagged direction --
      // `brusselator_graph.hpp::add_window_coupling()`'s own identical
      // choice for `ch_v2u` under the same mode, reproduced here so the
      // two drivers agree on which direction is lagged. Silently
      // ignored under Jacobi (`WindowChannel::lagged`'s own docstring).
      auto created = WindowCoupler::create(
          {mu, mv},
          {{.provider = 0, .consumer = 1, .name = "u_out"},
           {.provider = 1, .consumer = 0,
            .lagged = iteration_mode == fuse::WaveformIteration::GaussSeidel,
            .name = "v_out"}},
          cfg);
      if (!created.ok()) {
        std::fprintf(stderr, "FAIL: WindowCoupler::create at t=%.6f: %s\n",
                     boundaries[i].to_double(), created.error().message.c_str());
        return 1;
      }
      auto coupler = std::move(created.value());

      for (int r = 0; r < repeats; ++r) {
        const double t0 = boundaries[i + static_cast<std::size_t>(r)].to_double();
        const double eu0 = pu->external_total(), xo0 = pu->exchange_out();
        const double xi0 = pv->exchange_in();
        const double mu0 = mean_u, mv0 = mean_v;

        const auto r_win = coupler->run_window();
        if (!r_win.ok()) {
          std::fprintf(stderr, "FAIL: run_window() at t=%.6f: %s\n", t0, r_win.error().message.c_str());
          return 1;
        }
        max_iterations_used = std::max(max_iterations_used, coupler->stats().iterations);
        ++total_windows;

        mean_u = pu->state().mean();
        mean_v = pv->state().mean();
        const double d_external = pu->external_total() - eu0;
        const double d_out = pu->exchange_out() - xo0;
        const double d_in = pv->exchange_in() - xi0;
        const double balance_err = std::abs((mean_u + mean_v - mu0 - mv0) - d_external);
        const double exchange_err = std::abs(d_out - d_in);
        max_window_balance_error = std::max(max_window_balance_error, balance_err);
        max_window_exchange_error = std::max(max_window_exchange_error, exchange_err);

        if (pu->state().minCoeff() < -1e-12 || pv->state().minCoeff() < -1e-12) {
          std::fprintf(stderr, "FAIL: negative concentration at t=%.6f (min U=%.3e min V=%.3e)\n",
                       boundaries[i + static_cast<std::size_t>(r) + 1].to_double(),
                       pu->state().minCoeff(), pv->state().minCoeff());
          return 1;
        }
        if (window_log.is_open()) {
          window_log << t0 << ',' << boundaries[i + static_cast<std::size_t>(r) + 1].to_double() << ','
                     << mean_u << ',' << mean_v << ',' << coupler->stats().iterations << '\n';
        }
      }
      i = j;
    }
  } else {
    // --driver runner: the SAME graph (`window_graph`, built above),
    // compiled through `fuse::Runner` instead of driven directly through
    // `fuse::WindowCoupler` -- docs/user/coupling-windows.md "Through
    // schedule() and the Runner". Requires every window the SAME length
    // (`window`): `CouplingWindowScope::explicit_interval` is baked into
    // ONE compiled schedule here, unlike the --driver window branch
    // above, which reconstructs a new `WindowCoupler` (and so may change
    // `cfg.window_length`) at every group of equal-length windows. A
    // --window/--end combination whose forcing-switch insertion or final
    // truncation would need a different length is refused below rather
    // than silently run against the wrong one.
    for (std::size_t k = 0; k + 1 < boundaries.size(); ++k) {
      if (boundaries[k + 1] - boundaries[k] != window) {
        std::fprintf(
            stderr,
            "FAIL: E-BRUSS-RUNNER-NONUNIFORM-WINDOWS: window %zu is [%s, %s), length %s, "
            "expected %s -- the Runner driver requires every window the same length "
            "(use --driver window for a --window/--end combination that needs a "
            "shortened or inserted boundary)\n",
            k, boundaries[k].to_string().c_str(), boundaries[k + 1].to_string().c_str(),
            (boundaries[k + 1] - boundaries[k]).to_string().c_str(), window.to_string().c_str());
        return 1;
      }
    }

    components.add(std::move(comp_u));
    components.add(std::move(comp_v));

    fuse::FieldRegistry registry;
    for (auto& c : components) {
      const auto r = c.register_fields(registry, graph);
      if (!r.ok()) {
        std::fprintf(stderr, "FAIL: register_fields('%s'): %s: %s\n", c.name().c_str(),
                     r.error().code.c_str(), r.error().message.c_str());
        return 1;
      }
    }

    // U and V both need their own placed view of u_mesh (both transfer
    // fields live there -- brusselator_graph.hpp's own file header),
    // registered before schedule()/Runner::initialize() run. Discovered
    // automatically from the graph now (NativeMeshBackend::
    // register_default_placements, include/fuse/backend/native_backend.hpp) --
    // this example used to build that by hand, once per generated mesh;
    // the logic now lives in the backend, not here.
    if (const auto r = backend.register_default_placements(graph, u_mesh); !r.ok()) {
      std::fprintf(stderr, "FAIL: register_default_placements: %s: %s\n",
                   r.error().code.c_str(), r.error().message.c_str());
      return 1;
    }

    if (const auto d = fuse::validate(graph); d.has_errors()) {
      std::fprintf(stderr, "FAIL: validate(graph): %s\n", d.to_string().c_str());
      return 1;
    }
    const auto lowered = lower(graph);
    if (!lowered.ok()) {
      std::fprintf(stderr, "FAIL: lower(graph): %s\n", lowered.diags.to_string().c_str());
      return 1;
    }
    const fuse::ClockSet clocks{graph};
    auto sched = schedule(graph, lowered.dag, clocks,
                         fuse::make_region_sizes(graph, lowered.dag, backend));
    if (!sched.ok()) {
      std::fprintf(stderr, "FAIL: schedule(graph): %s\n", sched.diags.to_string().c_str());
      return 1;
    }

    fuse::RunnerConfig rcfg;
    // Matches WindowCouplerConfig's own default (the --driver window
    // branch never overrides it): a window that exhausts max_iterations
    // without converging fails the run rather than silently continuing.
    rcfg.abort_on_nonconvergence = true;
    fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
    if (const auto r = runner.initialize(components, registry, rcfg); !r.ok()) {
      std::fprintf(stderr, "FAIL: Runner::initialize: %s\n", r.error().message.c_str());
      return 1;
    }

    for (std::size_t k = 0; k + 1 < boundaries.size(); ++k) {
      const double t0 = boundaries[k].to_double();
      const double eu0 = pu->external_total(), xo0 = pu->exchange_out();
      const double xi0 = pv->exchange_in();
      const double mu0 = mean_u, mv0 = mean_v;

      const auto r_win = runner.run_period(components);
      if (!r_win.ok()) {
        std::fprintf(stderr, "FAIL: run_period() at t=%.6f: %s\n", t0, r_win.error().message.c_str());
        return 1;
      }
      if (runner.iteration_stats().empty()) {
        std::fprintf(stderr, "FAIL: run_period() at t=%.6f recorded no iteration stats\n", t0);
        return 1;
      }
      const int iterations = runner.iteration_stats().front().iterations;
      max_iterations_used = std::max(max_iterations_used, iterations);
      ++total_windows;

      mean_u = pu->state().mean();
      mean_v = pv->state().mean();
      const double d_external = pu->external_total() - eu0;
      const double d_out = pu->exchange_out() - xo0;
      const double d_in = pv->exchange_in() - xi0;
      const double balance_err = std::abs((mean_u + mean_v - mu0 - mv0) - d_external);
      const double exchange_err = std::abs(d_out - d_in);
      max_window_balance_error = std::max(max_window_balance_error, balance_err);
      max_window_exchange_error = std::max(max_window_exchange_error, exchange_err);

      if (pu->state().minCoeff() < -1e-12 || pv->state().minCoeff() < -1e-12) {
        std::fprintf(stderr, "FAIL: negative concentration at t=%.6f (min U=%.3e min V=%.3e)\n",
                     boundaries[k + 1].to_double(), pu->state().minCoeff(), pv->state().minCoeff());
        return 1;
      }
      if (window_log.is_open()) {
        window_log << t0 << ',' << boundaries[k + 1].to_double() << ',' << mean_u << ','
                   << mean_v << ',' << iterations << '\n';
      }
    }
  }

  const double global_balance_error =
      std::abs(mean_u + mean_v - initial_mass - pu->external_total());

  std::filesystem::create_directories(args.out);
  write_field_csv((std::filesystem::path(args.out) / "u.csv").string(), u_grid, pu->state(),
                  args.end);
  write_field_csv((std::filesystem::path(args.out) / "v.csv").string(), v_grid, pv->state(),
                  args.end);

  std::ofstream diag(std::filesystem::path(args.out) / "diagnostics.txt");
  diag << std::setprecision(17) << "nx_u=" << args.nx_u << " ny_u=" << args.ny_u
       << " nx_v=" << args.nx_v << " ny_v=" << args.ny_v << '\n'
       << "dt_u=" << args.dt_u << " dt_v=" << args.dt_v << " window=" << args.window
       << " end=" << args.end << '\n'
       << "A=3.4 B=1 alpha=" << phys_cfg.diffusion << " forcing=" << phys_cfg.forcing << '\n'
       << "tolerance=" << args.tolerance << " max_iterations=" << args.max_iterations << '\n'
       << "windows=" << total_windows << '\n'
       << "max_window_balance_error=" << max_window_balance_error << '\n'
       << "global_balance_error=" << global_balance_error << '\n'
       << "max_window_exchange_error=" << max_window_exchange_error << '\n'
       << "max_iterations_used=" << max_iterations_used << '\n'
       << "weights_vu_max_row_defect=" << w_vu.value().max_row_defect << '\n'
       << "weights_uv_max_row_defect=" << w_uv.value().max_row_defect << '\n'
       << "final_mean_u=" << mean_u << " final_mean_v=" << mean_v << '\n';
  diag.close();

  std::printf(
      "Brusselator windowed coupling: U %dx%d V %dx%d dt_u=%g dt_v=%g window=%g end=%g "
      "windows=%lld max_iterations=%d\n"
      "  final_mean_u=%.12g final_mean_v=%.12g\n"
      "  max_window_balance_error=%.3e global_balance_error=%.3e max_window_exchange_error=%.3e\n"
      "  Results: %s\n",
      args.nx_u, args.ny_u, args.nx_v, args.ny_v, args.dt_u, args.dt_v, args.window, args.end,
      total_windows, max_iterations_used, mean_u, mean_v, max_window_balance_error,
      global_balance_error, max_window_exchange_error, args.out.c_str());

  if (args.check) {
    // Thresholds below are DERIVED FROM MEASUREMENT, not chosen to pass:
    //
    // max_window_exchange_error is a SPATIAL statement (mean(q) as
    // reported by U vs. mean(P_UV q) as received by V) and holds to
    // round-off (measured 7.9e-14 at the defaults, 4.2e-15 at 4x
    // refinement) -- see species_component.hpp's UComponent::advance
    // comment for why the bookkeeping has to match WindowCoupler's own
    // trapezoidal reconstruction for this to be true.
    //
    // max_window_balance_error/global_balance_error ARE round-off, as in the
    // Toy. They were not in the first version of this port (1.686e-3 per
    // window, first order under refinement): U consumed dt*q_prev while
    // WindowCoupler delivered V the trapezoid 0.5*dt*(q_prev + q_new), so the
    // two species exchanged different amounts of the same reaction. U now
    // consumes exactly what it provides (UComponent::advance's implicit
    // reaction fixed point). MEASURED after that change, --end 0.1, joint
    // 1x/2x/4x refinement: max_window_balance_error = 1.703e-13, 8.970e-14,
    // 8.802e-15; global 5.300e-13, 2.112e-13, 7.964e-14; at --end 0.5 (this
    // test's arguments) 1.703e-13 / 2.407e-12; the Toy's own run at 11.5 is
    // 1.239e-13 / 1.599e-11. Thresholds: ~60x over the per-window floor and
    // ~40x over the long-run global one, both still round-off scale; the
    // first-order defect they replace measured four orders of magnitude
    // above them, so a regression to it fails immediately.
    bool ok = true;
    if (max_window_balance_error > 1e-11) {
      std::fprintf(stderr, "FAIL(--check): max_window_balance_error %.3e > 1e-11\n",
                   max_window_balance_error);
      ok = false;
    }
    if (std::abs(global_balance_error) > 1e-9) {
      std::fprintf(stderr, "FAIL(--check): global_balance_error %.3e > 1e-9\n", global_balance_error);
      ok = false;
    }
    if (max_window_exchange_error > 1e-9) {
      std::fprintf(stderr, "FAIL(--check): max_window_exchange_error %.3e > 1e-9\n",
                   max_window_exchange_error);
      ok = false;
    }
    if (pu->state().minCoeff() < -1e-12 || pv->state().minCoeff() < -1e-12) {
      std::fprintf(stderr, "FAIL(--check): final state has a negative concentration\n");
      ok = false;
    }
    if (!ok) return 1;
    std::printf("  --check: OK\n");
  }

  return 0;
}
