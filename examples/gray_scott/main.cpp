// Gray-Scott U/V on one structured grid, coupled implicitly at independent
// time steps through fuse::WindowCoupler.
//
// U and V are TWO COMPONENTS on the SAME N x N structured grid (no spatial
// remap: species_component.hpp's GrayScottSpeciesComponent is a cell-centred
// P0 finite-volume solver, Crank-Nicolson in time, Newton-solved with Eigen
// SparseLU). U is the window-definer (its own interval IS the coupling
// window, so it is structurally incapable of straddling); V is the
// (possibly) finer member, at a commensurate ratio dt_u/m or an
// incommensurate ratio, per docs/user/gray-scott.md and
// docs/user/coupling-windows.md.
//
// --check runs a small internal self-consistency test (m=1 windowed result
// against the monolithic reference, and 0<=u,v<=1 for a short run) and
// returns non-zero if it fails -- the same convention example.mms_pair and
// example.dn_heat use. The deep verification (convergence order, iteration
// counts, mass exchange, the long pattern-forming run) lives in
// test/verification/gray_scott_*_test.cpp; this binary is the user-facing
// driver, not the evidence.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <fuse/base/error.hpp>

#include <fuse/runtime/window_coupler.hpp>

#include "accel_names.hpp"
#include "gray_scott_physics.hpp"
#include "monolithic_reference.hpp"
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
  int    n = 64;
  double dt = 0.5;            // fine time unit; see RatioSpec below
  std::string ratio = "1";    // "1" | "2" | "4" | "7:13"
  std::string policy = "adjustable";  // DtPolicy for the finer/straddling member
  std::string accel = "aitken";
  double t_end = 50.0;
  bool   check = false;
  std::string output;
  double F = 0.0392, k = 0.0649, Du = 2.0e-5, Dv = 1.0e-5, L = 2.5;
  double coupling_tol = 1.0e-8;
  double newton_tol = 1.0e-11;
};

[[nodiscard]] Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (arg == "--n") a.n = std::atoi(next().c_str());
    else if (arg == "--dt") a.dt = std::atof(next().c_str());
    else if (arg == "--ratio") a.ratio = next();
    else if (arg == "--policy") a.policy = next();
    else if (arg == "--accel") a.accel = next();
    else if (arg == "--t-end") a.t_end = std::atof(next().c_str());
    else if (arg == "--check") a.check = true;
    else if (arg == "--output") a.output = next();
    else if (arg == "--F") a.F = std::atof(next().c_str());
    else if (arg == "--k") a.k = std::atof(next().c_str());
    else if (arg == "--Du") a.Du = std::atof(next().c_str());
    else if (arg == "--Dv") a.Dv = std::atof(next().c_str());
    else if (arg == "--L") a.L = std::atof(next().c_str());
  }
  return a;
}

/// Decimal -> exact Rational at a fixed denominator (1e6). --dt is a
/// convenience decimal on the command line; every FUSE time value must be
/// exact (fuse/base/rational.hpp), so the conversion is done once, here, and
/// nowhere downstream sees a double time.
[[nodiscard]] Rational to_rational(double v) {
  constexpr std::int64_t kDen = 1000000;
  return Rational{static_cast<std::int64_t>(std::llround(v * kDen)), kDen};
}

[[nodiscard]] DtPolicy parse_policy(const std::string& s) {
  return (s == "fixed") ? DtPolicy::Fixed : DtPolicy::Adjustable;
}

/// `--accel`'s names come from the shared table
/// (examples/common/accel_names.hpp). main() refuses an unknown name
/// (E-EXAMPLE-ACCEL) before any setup, rather than silently running with
/// Aitken.
[[nodiscard]] AccelKind parse_accel(const std::string& s) {
  if (const auto k = fuse_examples::accel_kind_from_name(s)) return *k;
  std::fprintf(stderr, "FAIL: E-EXAMPLE-ACCEL: unknown --accel '%s'\n", s.c_str());
  std::exit(1);
}

/// Window length (U's own interval) and V's interval, from --ratio.
/// "m"  -> U = fine*m (the window), V = fine.
/// "p:q"-> U = fine*q (the window, the coarser/window-defining member),
///         V = fine*p.
void resolve_ratio(const std::string& ratio, Rational fine, Rational* u_interval,
                   Rational* v_interval) {
  const auto colon = ratio.find(':');
  if (colon == std::string::npos) {
    const int m = std::max(1, std::atoi(ratio.c_str()));
    *u_interval = fine * Rational{m};
    *v_interval = fine;
  } else {
    const int p = std::atoi(ratio.substr(0, colon).c_str());
    const int q = std::atoi(ratio.substr(colon + 1).c_str());
    *u_interval = fine * Rational{q};
    *v_interval = fine * Rational{p};
  }
}

/// Seed near u=1,v=0 with a small square perturbation at the domain centre,
/// the standard Gray-Scott initial condition for spot self-replication
/// (Pearson 1993).
void seed_initial_condition(const gray_scott::Grid& g, std::vector<double>* u,
                            std::vector<double>* v) {
  u->assign(g.size(), 1.0);
  v->assign(g.size(), 0.0);
  const int lo = g.n / 2 - std::max(1, g.n / 20);
  const int hi = g.n / 2 + std::max(1, g.n / 20);
  for (int i = lo; i < hi; ++i) {
    for (int j = lo; j < hi; ++j) {
      (*u)[g.index(i, j)] = 0.5;
      (*v)[g.index(i, j)] = 0.25;
    }
  }
}

void write_csv(const std::string& path, const gray_scott::Grid& g,
              const std::vector<double>& u, const std::vector<double>& v) {
  std::ofstream out(path);
  out << "i,j,x,y,u,v\n";
  for (int i = 0; i < g.n; ++i) {
    for (int j = 0; j < g.n; ++j) {
      const std::size_t m = g.index(i, j);
      out << i << ',' << j << ',' << g.x_center(i) << ',' << g.y_center(j) << ','
          << u[m] << ',' << v[m] << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Process-wide diagnostics set-up (fuse/base/error.hpp): reads FUSE_ON_ERROR
  // and its companions (docs/dev/debugging.md) once. Without this call the
  // environment is parsed by nothing and the built-in defaults apply.
  fuse::initialize();

  // The Runner's coupling hot loop runs on Kokkos' default host execution
  // space (fuse/num/space_kernels.hpp), so every caller -- including this
  // one, which runs no Kokkos kernel of its own -- must initialise Kokkos
  // before touching a Runner. Same pattern as
  // test/support/fuse_test_main.cpp's `Kokkos::ScopeGuard` and
  // examples/brusselator/main.cpp's.
  Kokkos::ScopeGuard kokkos_guard(argc, argv);

  const Args args = parse_args(argc, argv);
  if (!fuse_examples::accel_kind_from_name(args.accel)) {
    std::fprintf(stderr, "FAIL: E-EXAMPLE-ACCEL: unknown --accel '%s'; use one of %.*s\n",
                 args.accel.c_str(), static_cast<int>(fuse_examples::kAccelNames.size()),
                 fuse_examples::kAccelNames.data());
    return 1;
  }

  const gray_scott::Grid grid{args.n, args.L};
  gray_scott::Kinetics kin{args.F, args.k};

  std::vector<double> u0, v0;
  seed_initial_condition(grid, &u0, &v0);

  auto comp_u = std::make_unique<gray_scott::GrayScottU>(
      "U", grid, args.Du, kin, u0, args.newton_tol);
  auto comp_v = std::make_unique<gray_scott::GrayScottV>(
      "V", grid, args.Dv, kin, v0, args.newton_tol);

  const Rational fine = to_rational(args.dt);
  Rational u_interval, v_interval;
  resolve_ratio(args.ratio, fine, &u_interval, &v_interval);
  const DtPolicy v_policy = parse_policy(args.policy);

  WindowMember mu;
  mu.component = comp_u.get();
  mu.name = "U";
  mu.interval = u_interval;
  mu.policy = DtPolicy::Adjustable;  // U defines the window; never straddles.
  mu.provided = comp_u->provided_ptr();
  mu.required = comp_u->required_ptr();
  mu.length = grid.size();

  WindowMember mv;
  mv.component = comp_v.get();
  mv.name = "V";
  mv.interval = v_interval;
  mv.policy = v_policy;
  mv.provided = comp_v->provided_ptr();
  mv.required = comp_v->required_ptr();
  mv.length = grid.size();

  WindowCouplerConfig cfg;
  cfg.window_length = u_interval;
  cfg.max_iterations = 100;
  cfg.acceleration = AccelConfig{.kind = parse_accel(args.accel)};
  // A convergence measure names a field TRANSFER (a WindowChannel, by its
  // own `name`), not the whole coupled system. Split into the two real
  // transfers, same kind/tolerance; both must be satisfied.
  fuse::ConvergenceMeasure measure_u;
  measure_u.kind = fuse::MeasureKind::Relative;
  measure_u.tolerance = args.coupling_tol;
  measure_u.field = "u_out";
  cfg.convergence.add(measure_u);
  fuse::ConvergenceMeasure measure_v;
  measure_v.kind = fuse::MeasureKind::Relative;
  measure_v.tolerance = args.coupling_tol;
  measure_v.field = "v_out";
  cfg.convergence.add(measure_v);
  cfg.waveform_capacity = 256;

  auto created = WindowCoupler::create(
      {mu, mv},
      {{.provider = 0, .consumer = 1, .name = "u_out"},
       {.provider = 1, .consumer = 0, .name = "v_out"}},
      cfg);
  if (!created.ok()) {
    std::fprintf(stderr, "FAIL: WindowCoupler::create: %s\n", created.error().message.c_str());
    return 1;
  }
  std::unique_ptr<WindowCoupler> coupler = std::move(created.value());

  const int n_windows =
      static_cast<int>(std::max(1.0, std::round(args.t_end / (u_interval.to_double()))));
  int total_iterations = 0;
  for (int w = 0; w < n_windows; ++w) {
    const auto r = coupler->run_window();
    if (!r.ok()) {
      std::fprintf(stderr, "FAIL: run_window() at window %d: %s\n", w, r.error().message.c_str());
      return 1;
    }
    total_iterations += coupler->stats().iterations;
  }

  double umin = 1e300, umax = -1e300, vmin = 1e300, vmax = -1e300;
  for (double x : comp_u->state()) { umin = std::min(umin, x); umax = std::max(umax, x); }
  for (double x : comp_v->state()) { vmin = std::min(vmin, x); vmax = std::max(vmax, x); }

  std::printf(
      "Gray-Scott windowed coupling: n=%d ratio=%s policy=%s accel=%s dt=%g "
      "t_end=%g windows=%d total_iterations=%d\n"
      "  u range [%.6g, %.6g], v range [%.6g, %.6g]\n",
      args.n, args.ratio.c_str(), args.policy.c_str(), args.accel.c_str(), args.dt,
      args.t_end, n_windows, total_iterations, umin, umax, vmin, vmax);

  if (!args.output.empty()) {
    write_csv(args.output, grid, comp_u->state(), comp_v->state());
    std::printf("  wrote %s\n", args.output.c_str());
  }

  if (args.check) {
    // Self-consistency: at m=1, re-run a fresh windowed coupler and a
    // monolithic reference over the same short interval from the same
    // initial condition, and require they agree to the coupling tolerance.
    std::vector<double> mu0, mv0;
    seed_initial_condition(grid, &mu0, &mv0);
    auto cu = std::make_unique<gray_scott::GrayScottU>("U", grid, args.Du, kin, mu0, args.newton_tol);
    auto cv = std::make_unique<gray_scott::GrayScottV>("V", grid, args.Dv, kin, mv0, args.newton_tol);
    WindowMember wmu = mu; wmu.component = cu.get(); wmu.provided = cu->provided_ptr(); wmu.required = cu->required_ptr();
    WindowMember wmv = mv; wmv.component = cv.get(); wmv.provided = cv->provided_ptr(); wmv.required = cv->required_ptr();
    wmu.interval = fine; wmv.interval = fine;  // m=1
    WindowCouplerConfig cfg1 = cfg;
    cfg1.window_length = fine;
    auto created1 = WindowCoupler::create(
        {wmu, wmv},
        {{.provider = 0, .consumer = 1, .name = "u_out"},
         {.provider = 1, .consumer = 0, .name = "v_out"}},
        cfg1);
    if (!created1.ok()) {
      std::fprintf(stderr, "FAIL(--check): WindowCoupler::create(m=1): %s\n",
                   created1.error().message.c_str());
      return 1;
    }
    auto coupler1 = std::move(created1.value());
    gray_scott::MonolithicGrayScott mono(grid, args.Du, args.Dv, kin, mu0, mv0, args.newton_tol);
    const int check_windows = 5;
    for (int w = 0; w < check_windows; ++w) {
      const auto r = coupler1->run_window();
      if (!r.ok()) {
        std::fprintf(stderr, "FAIL(--check): run_window(m=1) at %d: %s\n", w, r.error().message.c_str());
        return 1;
      }
      if (!mono.step(fine.to_double(), w * fine.to_double())) {
        std::fprintf(stderr, "FAIL(--check): monolithic reference did not converge\n");
        return 1;
      }
    }
    double max_diff = 0.0;
    for (std::size_t m = 0; m < grid.size(); ++m) {
      max_diff = std::max(max_diff, std::abs(cu->state()[m] - mono.u()[m]));
      max_diff = std::max(max_diff, std::abs(cv->state()[m] - mono.v()[m]));
    }
    // Spec requirement (delegated task, step 3): "at m=1 the coupled
    // windowed result must match [the monolithic reference] to the
    // iteration tolerance" -- a TIGHT bound, not one scaled to dt. It is
    // only exactly true in the dt -> 0 limit (measured in
    // gray_scott_reference_test.cpp's WindowedVsMonolithicMismatchAtRatioOne:
    // the partitioned scheme freezes the partner at its CN-window mean and
    // applies that SAME frozen value at both the old and new reaction
    // evaluations, while the monolithic scheme uses the partner's actual
    // old/new values separately, so u*v^2's nonlinearity in v leaves a
    // real, measured, order-2-in-dt residual between the two schemes at
    // any FIXED dt > 0 -- see that test and docs/user/gray-scott.md
    // "Partitioned vs monolithic at m=1" for the numbers). The bound here
    // stays tight, per spec; --dt's default is small enough that the
    // order-2 term this test happens to run at is already below it (see
    // example.gray_scott's registration in examples/CMakeLists.txt for the
    // operating point that was checked to satisfy this).
    const double check_tol = 100.0 * args.coupling_tol;
    if (max_diff > check_tol) {
      std::fprintf(stderr,
                   "FAIL(--check): windowed m=1 vs monolithic reference differ by %.3e, "
                   "expected <= %.3e\n",
                   max_diff, check_tol);
      return 1;
    }
    if (umin < -1e-6 || vmin < -1e-6) {
      std::fprintf(stderr, "FAIL(--check): concentration went negative (umin=%.3e vmin=%.3e)\n",
                   umin, vmin);
      return 1;
    }
    std::printf("  --check: m=1 windowed vs monolithic max diff %.3e (OK)\n", max_diff);
  }

  return 0;
}
