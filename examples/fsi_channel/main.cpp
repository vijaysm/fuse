// 2D Stokes channel with an Euler-Bernoulli wall beam, strongly coupled
// through FUSE's implicit coupling group on a non-matching 1D interface (consistent
// H, conservative H^T). docs/user/fsi-channel.md carries the full model,
// discretisation and coupling write-up; this binary is the runnable driver
// and the CLI --check gate on top of it.
//
// SMALL-DISPLACEMENT ASSUMPTION, STATED HERE AND EVERYWHERE ELSE THIS MODEL
// IS DISCUSSED: the fluid domain is FIXED. The beam's motion enters the
// fluid solve only through the top wall's normal-velocity (transpiration)
// boundary condition, never through a moving mesh.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <fuse/ir/lowering.hpp>
#include <fuse/backend/consistent_interp.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include "accel_names.hpp"
#include "beam_solver.hpp"
#include "fsi_components.hpp"
#include "stokes_solver.hpp"

namespace {

struct Args {
  std::size_t nx_fluid = 24, ny_fluid = 12, nx_beam = 16;
  double      mass_ratio = 1.0;
  std::string accel = "iqn-ils";
  double      dt = 0.01;
  double      t_end = 1.0;
  bool        check = false;
  std::string csv_path = "fsi_channel.csv";
};

[[nodiscard]] Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string(flag) + ": missing value");
      }
      return argv[++i];
    };
    if (arg == "--nx-fluid") a.nx_fluid = std::stoul(next("--nx-fluid"));
    else if (arg == "--ny-fluid") a.ny_fluid = std::stoul(next("--ny-fluid"));
    else if (arg == "--nx-beam") a.nx_beam = std::stoul(next("--nx-beam"));
    else if (arg == "--mass-ratio") a.mass_ratio = std::stod(next("--mass-ratio"));
    else if (arg == "--accel") a.accel = next("--accel");
    else if (arg == "--dt") a.dt = std::stod(next("--dt"));
    else if (arg == "--t-end") a.t_end = std::stod(next("--t-end"));
    else if (arg == "--csv") a.csv_path = next("--csv");
    else if (arg == "--check") a.check = true;
    else throw std::invalid_argument("unknown argument: " + arg);
  }
  return a;
}

// --- fixed physical parameters (see docs/user/fsi-channel.md) --------------
constexpr double kL = 1.0;       // channel length [m]
constexpr double kH = 0.2;       // channel height [m]
constexpr double kMu = 0.01;     // fluid dynamic viscosity [Pa s]
constexpr double kRhoF = 1.0;    // fluid density [kg/m^3]
constexpr double kUMax = 1.0;    // inflow peak (centreline) velocity [m/s]
constexpr double kEI = 2.0;      // beam flexural rigidity [N m^2] -- sized so
                                  // the channel's own traction scale (~mu
                                  // U_max / H) gives a mid-span deflection
                                  // several orders below H, genuinely
                                  // satisfying the small-displacement
                                  // assumption (docs/user/fsi-channel.md)
constexpr double kRhoS = 1.0;    // beam material density [kg/m^3] (nominal)

/// Names come from the shared table (examples/common/accel_names.hpp); the
/// settings stay this example's. `none`, `aitken` and `iqn-ils` keep exactly
/// the settings they had. `constant`, `secant` and `steffensen` use the same
/// initial relaxation as `aitken`, and `iqn-imvj` the same history and filter
/// as `iqn-ils`; none was tuned. Measured 2026-10-20 at the registered command
/// line: iqn-imvj and iqn-ils converge, constant, steffensen and secant do not
/// (docs/user/fsi-channel.md).
fuse::AccelConfig accel_config(const std::string& name) {
  const auto kind = fuse_examples::accel_kind_from_name(name);
  if (!kind) {
    throw std::invalid_argument("E-EXAMPLE-ACCEL: unknown --accel '" + name +
                                "'; use one of " + std::string(fuse_examples::kAccelNames));
  }
  switch (*kind) {
    case fuse::AccelKind::None:
      return fuse::AccelConfig{.kind = fuse::AccelKind::None};
    case fuse::AccelKind::IQN_ILS:
    case fuse::AccelKind::IQN_IMVJ:
      return fuse::AccelConfig{.kind = *kind,
                               .initial_relaxation = 0.5,
                               .reuse = 8,
                               .filter = fuse::FilterKind::None};
    default:
      return fuse::AccelConfig{.kind = *kind, .initial_relaxation = 0.5};
  }
}

fuse::Rational rationalize_dt(double dt) {
  const auto num = static_cast<std::int64_t>(std::llround(dt * 1.0e6));
  return fuse::Rational{num, 1'000'000};
}

fuse::CouplingGraph build_graph(std::size_t nx_beam, std::size_t nx_fluid,
                               fuse::AccelConfig accel, double dt,
                               int max_iterations) {
  fuse::CouplingGraph g;
  const auto m_beam = g.add_mesh(fuse::MeshDecl{
      .name = "beam_iface", .source = "generated:line:" + std::to_string(nx_beam)});
  const auto m_fluid = g.add_mesh(fuse::MeshDecl{
      .name = "fluid_iface",
      .source = "generated:line:" + std::to_string(nx_fluid)});
  const auto r_beam =
      g.add_region(fuse::Region{.name = "r_beam", .mesh = m_beam, .dimension = 1});
  const auto r_fluid = g.add_region(
      fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});

  const fuse::Rational period = rationalize_dt(dt);
  const auto beam = g.add_component(fuse::ComponentDecl{
      .name = "beam", .ranks = {0, 1}, .interval = period});
  const auto fluid = g.add_component(fuse::ComponentDecl{
      .name = "fluid", .ranks = {0, 1}, .interval = period});

  constexpr fuse::Units kMps{.length = 1, .time = -1};

  const auto vel_out = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_velocity", .region = r_beam,
      .disc = fuse::Discretization::ContinuousGLL, .order = 1,
      .units = kMps, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto vel_in = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_velocity_in", .region = r_fluid,
      .units = kMps, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});
  const auto vch = g.add_channel("wall_velocity", vel_out, vel_in);
  g.channel(vch).remap_method("consistent");

  const auto force_out = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_force", .region = r_fluid, .units = fuse::units::kNewton,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto load_in = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_load", .region = r_beam,
      .disc = fuse::Discretization::ContinuousGLL, .order = 1,
      .units = fuse::units::kNewton, .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto fch = g.add_channel("wall_force", force_out, load_in);
  g.channel(fch).remap_method("consistent-transpose").conservative();
  // The velocity and force field transfers close a cycle within the coupling cycle, so
  // one of them must carry a lag for the graph to be schedulable at all
  // (E-SCHED-CYCLE otherwise) -- exactly examples/dn_heat/main.cpp's
  // pattern. Inside the iterate() implicit coupling group below, that lag is precisely
  // what the sweep resolves.
  g.channel(fch).lag(1).allow_time_level_mismatch(
      "resolved by the beam-fluid implicit coupling group");

  fuse::ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(1e-7, "beam_velocity"));
  crit.add(fuse::Converged::Relative(1e-7, "fluid_force"));
  g.iterate({vch, fch})
      .name("fsi_beam_fluid")
      .until(std::move(crit))
      .max_iterations(max_iterations)
      .accelerate(accel);
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  // KNOWN_ISSUES.md D11 (closing): the Runner's coupling hot loop now runs
  // on Kokkos' default host execution space (fuse/num/space_kernels.hpp),
  // so every caller -- including this one, which ran no Kokkos kernel at
  // all before this closing -- must initialise Kokkos before touching a
  // Runner. Same pattern as test/support/fuse_test_main.cpp's
  // `Kokkos::ScopeGuard` and examples/brusselator/main.cpp's.
  Kokkos::ScopeGuard kokkos_guard(argc, argv);

  Args args;
  try {
    args = parse(argc, argv);
    (void)accel_config(args.accel);  // refuse an unknown name before any output
  } catch (const std::exception& e) {
    std::fprintf(stderr, "argument error: %s\n", e.what());
    return 2;
  }

  const double h_s = args.mass_ratio * kRhoF * kH / kRhoS;
  const double omega1 =
      fsi::clamped_beam_first_frequency(kL, kEI, kRhoS, h_s);
  std::printf(
      "FSI channel: fluid %zux%zu, beam %zu elements, mass ratio %.4g "
      "(rho_s h_s = %.4g), accel=%s, dt=%.4g, t_end=%.4g\n"
      "  beam first natural frequency: %.4g rad/s (period %.4g s)\n",
      args.nx_fluid, args.ny_fluid, args.nx_beam, args.mass_ratio, h_s,
      args.accel.c_str(), args.dt, args.t_end, omega1, 2.0 * M_PI / omega1);

  const int max_iterations = args.accel == "none" ? 50 : 200;
  fuse::CouplingGraph graph = build_graph(args.nx_beam, args.nx_fluid,
                                          accel_config(args.accel), args.dt,
                                          max_iterations);

  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) {
      std::fprintf(stderr, "mesh load: %s\n", r.error().message.c_str());
      return 1;
    }
  }
  const auto lowered = lower(graph);
  if (!lowered.ok()) {
    std::fprintf(stderr, "lower: %s\n", lowered.diags.to_string().c_str());
    return 1;
  }
  const fuse::ClockSet clocks{graph};
  auto sched = fuse::schedule(graph, lowered.dag, clocks,
                              fuse::make_region_sizes(graph, lowered.dag, backend));
  if (!sched.ok()) {
    std::fprintf(stderr, "schedule: %s\n", sched.diags.to_string().c_str());
    return 1;
  }

  const auto inflow = [](double y) {
    return 4.0 * kUMax * y * (kH - y) / (kH * kH);
  };

  fuse::ComponentSet components;
  auto beam = std::make_unique<fsi::BeamComponent>(
      "beam", fsi::BeamParams{.n = args.nx_beam, .L = kL, .EI = kEI,
                              .rho_s = kRhoS, .h_s = h_s});
  auto fluid = std::make_unique<fsi::FluidComponent>(
      "fluid",
      fsi::ChannelParams{.nx = args.nx_fluid, .ny = args.ny_fluid, .L = kL,
                        .H = kH, .mu = kMu, .rho = kRhoF},
      inflow);
  fsi::BeamComponent* beam_ptr = beam.get();
  fsi::FluidComponent* fluid_ptr = fluid.get();
  components.add(std::move(beam));
  components.add(std::move(fluid));

  fuse::FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, graph);
    if (!r.ok()) {
      std::fprintf(stderr, "register_fields: %s\n", r.error().message.c_str());
      return 1;
    }
  }

  fuse::RunnerConfig cfg;
  cfg.abort_on_nonconvergence = false;
  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry, cfg); !r.ok()) {
    std::fprintf(stderr, "initialize: %s\n", r.error().message.c_str());
    return 1;
  }

  const std::size_t n_steps =
      static_cast<std::size_t>(std::llround(args.t_end / args.dt));
  std::ofstream csv(args.csv_path);
  csv << "step,time,sweeps,converged,beam_mid_deflection,beam_max_deflection\n";

  std::int64_t not_converged_total = 0;
  for (std::size_t step = 0; step < n_steps; ++step) {
    if (const auto r = runner.run_period(components); !r.ok()) {
      std::fprintf(stderr, "run_period: %s\n", r.error().message.c_str());
      return 1;
    }
    const auto& istats = runner.iteration_stats();
    const int sweeps = istats.empty() ? 0 : istats.front().iterations;
    const bool converged = istats.empty() ? true : istats.front().converged;
    if (!converged) ++not_converged_total;

    const std::vector<double>& w = beam_ptr->solver().displacement();
    const double mid = w[w.size() / 2];
    double wmax = 0.0;
    for (const double v : w) wmax = std::max(wmax, std::abs(v));

    const double t = static_cast<double>(step + 1) * args.dt;
    csv << step << ',' << t << ',' << sweeps << ',' << (converged ? 1 : 0)
        << ',' << mid << ',' << wmax << '\n';
  }
  csv.close();

  const std::vector<double>& w_final = beam_ptr->solver().displacement();
  std::printf(
      "ran %zu steps; total sweeps %lld, windows not converged %lld\n"
      "final beam mid-span deflection: %.6e m\n"
      "wrote %s\n",
      n_steps, static_cast<long long>(runner.stats().total_iterations),
      static_cast<long long>(not_converged_total),
      w_final[w_final.size() / 2], args.csv_path.c_str());
  (void)fluid_ptr;

  if (args.check) {
    if (args.accel != "none" && not_converged_total != 0) {
      std::fprintf(stderr,
                   "FAIL: %lld window(s) did not converge with accel=%s\n",
                   static_cast<long long>(not_converged_total),
                   args.accel.c_str());
      return 1;
    }
    const std::vector<double>& w = beam_ptr->solver().displacement();
    bool any_finite_nonzero = false;
    for (const double v : w) {
      if (!std::isfinite(v)) {
        std::fprintf(stderr, "FAIL: beam displacement is not finite\n");
        return 1;
      }
      if (v != 0.0) any_finite_nonzero = true;
    }
    if (!any_finite_nonzero) {
      std::fprintf(stderr, "FAIL: beam never moved -- the coupling did "
                           "nothing measurable\n");
      return 1;
    }
    std::printf("all checks passed\n");
  }
  return 0;
}
