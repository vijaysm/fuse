// Demo A of spec §11 at the Phase 7 milestone: IMPLICIT coupling.
//
// A classical Dirichlet-Neumann pair of heat conduction domains, iterated to
// its fixed point inside a `g.iterate({...})` implicit coupling group with a derived checkpoint
// set, a convergence criterion covering both field transfers, and quasi-Newton
// acceleration. The physics, the analytic solution and the reason the error is
// attributable to the coupler are all in dn_component.hpp.
//
// WHAT THIS BINARY IS FOR. An example that merely runs proves nothing. Task
// 4.7's demo printed an authoritative-looking "L2 error 1.30e-06" that was
// completely insensitive to whether the coupling happened at all, because the
// component it measured had been accidentally uncoupled. This one therefore
// runs the SAME graph in several configurations and gates its exit status on
// the relationships between their errors:
//
//   explicit   one sweep per coupling cycle -- the classical staggered scheme
//   picard     the same implicit coupling group, iterated to convergence, no acceleration
//   aitken     ... with Aitken relaxation
//   iqn-ils    ... with IQN-ILS and no column filter
//   iqn+qr2    ... with IQN-ILS and AccelConfig's default QR2 filter at its
//                  default limit of 1e-2, which is measurably the wrong
//                  setting for this problem (see the note by iqn_qr2_cfg)
//   severed    picard, with both participants' incoming fields bound to decoy
//              arrays: the exchange happens and lands nowhere
//   *-long     explicit and IQN-ILS over 10x as many coupling cycles, so the explicit
//              comparison is also made in the asymptotic regime and not only
//              while the initial transient is still the largest term
//   newton     the SAME affine graph (dn::DirichletSlab's beta = 0.0 default,
//              so its own conductivity is identical to every row above), with
//              the iteration region selecting S7.2's native Jacobian-free
//              Newton-Krylov solver (`.newton()`) instead of `.accelerate()`.
//              Enters the SAME correctness gates (agree with picard, converged,
//              reproducible) as the quasi-Newton rows; "sweeps" means something
//              different for it (one per coupling period, not one per Newton
//              step -- see the dedicated NOTE at its own print_row call) and is
//              deliberately left out of the "took fewer sweeps" ladder gate 6
//              compares the Picard-family rows on.
//   newton-nl  dn::DirichletSlab::beta = 0.1 (a real nonlinearity in the wall's
//              own conductivity, docs/dev/newton-krylov.md §8.3), still solved
//              by Newton: informational, like iqn+qr2, reporting the FITTED
//              convergence order from the final period's own residual history
//              rather than entering the correctness ladder (it is not the same
//              physical problem the other rows solve, so it has no fixed point
//              to agree with them about).
//
// The gates: the implicit runs must agree with each other and with the
// analytic solution; the implicit answer must beat the explicit one by orders
// of magnitude, both cold and asymptotically; and it must beat the SEVERED one
// by orders of magnitude, which is the measurement proving the headline number
// moves when the coupling breaks. IQN-ILS must cost far fewer sweeps than
// Picard.
//
// WHAT WAS MEASURED WHILE BUILDING THIS, so the claims above are numbers and
// not opinions (32 stations, 20 coupling cycles, tolerance 1e-8, debug build):
//
//   * deleting either convergence measure -- i.e. writing the single-measure
//     form spec §5.3's example suggests -- is rejected at initialize() with
//     E-ITER-MEASURE naming the unmeasured field transfer. Both directions were
//     tried. Nothing silently converges to the wrong fixed point here, but
//     only because the runtime rejects to start.
//   * severing the exchange moves the headline number by seven orders of
//     magnitude, and every partial break is caught too:
//         connected                        L2 err u = 7.9e-07 K
//         neumann's flux input severed     L2 err u = 3.03e+01 K
//         dirichlet's temperature severed  L2 err u = 2.41e+02 K
//         both severed                     L2 err u = 3.03e+01 K
//     all three severed runs CONVERGE, in two or three sweeps per coupling cycle,
//     with a monotone residual history and no diagnostic of any kind.
//
// IQN-ILS, not IQN-IMVJ, deliberately. Measured elsewhere in this project on a
// nonlinear contraction, IMVJ was 1.3-3.6x slower with erratic per-window
// counts (40/62/54/45/36/119/52/74/39 against a flat 15 for ILS), landed in a
// different valid fixed point in 5 windows of 8, and has no restart mode to
// refresh its retained Jacobian. It is not used to demonstrate convergence
// quality anywhere here.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include "dn_component.hpp"

namespace {

/// 32 interface stations, 8 per material strip.
constexpr std::size_t  kStations = 32;
/// 20 coupling cycles of H = 1/20, so t runs over one full cycle of T1.
constexpr std::int64_t kPeriods = 20;
constexpr std::int64_t kPeriodDen = 20;
/// The asymptotic pair. An explicit run's error contracts by only r = 0.96 per
/// coupling cycle, so after 20 coupling cycles it is still dominated by its initial transient
/// (0.96^20 = 0.44). After 200 it is not (0.96^200 = 3.3e-4).
constexpr std::int64_t kLongPeriods = 200;
/// Relative, on both field transfers. See build_graph().
constexpr double       kTol = 1.0e-8;
/// Generous: Picard on this problem needs several hundred sweeps per coupling cycle,
/// and the point of quoting that number is that it is large.
constexpr int          kMaxIterations = 1500;

// --- the gates -------------------------------------------------------------
/// The converged interface temperature must sit within this of the analytic
/// two-layer-wall solution [K]. The participants have no discretisation error,
/// so this is the coupler's error and nothing else's; it is a decade above
/// what kTol * ||u|| ~ 3e-6 K permits.
constexpr double kImplicitGate = 3.0e-5;
/// ...and the accelerators must land on the same fixed point [K].
constexpr double kAgreementGate = 1.0e-4;
/// An explicit run must be visibly worse, and a severed one much worse [K].
constexpr double kExplicitFloor = 0.5;
constexpr double kSeveredFloor = 5.0;
/// The margin by which implicit must beat both.
constexpr double kEvidenceRatio = 1.0e4;
/// IQN-ILS must cost at most this fraction of Picard's sweeps.
constexpr double kAccelRatio = 0.1;

fuse::CouplingGraph build_graph(fuse::AccelConfig accel, int max_iterations) {
  fuse::CouplingGraph g;
  const std::string src = "generated:line:" + std::to_string(kStations);

  // Two coincident meshes: each participant owns its own discretisation of the
  // shared interface, which is the honest coupling setup. They are coincident,
  // so Remap is the identity -- the only configuration in which the serial
  // backend's Remap is the correct operator.
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "iface_1", .source = src});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "iface_2", .source = src});
  const auto r1 = g.add_region(
      fuse::Region{.name = "iface_1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(
      fuse::Region{.name = "iface_2", .mesh = m2, .dimension = 1});

  const auto cd = g.add_component(fuse::ComponentDecl{
      .name = "dirichlet", .ranks = {0, 1},
      .interval = fuse::Rational{1, kPeriodDen}});
  const auto cn = g.add_component(fuse::ComponentDecl{
      .name = "neumann", .ranks = {1, 1},
      .interval = fuse::Rational{1, kPeriodDen}});

  // Grouped by owner: to_yaml() nests a field inside its component, so an
  // interleaved declaration order does not survive a round trip as identical
  // FieldIds.
  const auto q_out = g.add_field(cd, fuse::FieldSpec{
      .name = "q_wall", .region = r1, .units = fuse::units::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto u_in = g.add_field(cd, fuse::FieldSpec{
      .name = "u_wall_in", .region = r1, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});
  const auto q_in = g.add_field(cn, fuse::FieldSpec{
      .name = "q_wall_in", .region = r2,
      .units = fuse::units::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto u_out = g.add_field(cn, fuse::FieldSpec{
      .name = "u_wall", .region = r2, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});

  const auto flux = g.add_channel("flux", q_out, q_in);
  const auto temp = g.add_channel("wall_temperature", u_out, u_in);
  // The two field transfers close a cycle, so one of them must carry a lag for the
  // graph to be schedulable at all. Inside an implicit coupling group that lag is
  // what the sweep resolves; that is precisely the difference between implicit
  // and staggered coupling.
  g.channel(temp).lag(1).allow_time_level_mismatch(
      "resolved by the Dirichlet-Neumann implicit coupling group");

  // THE CRITERION COVERS BOTH FIELD TRANSFERS, and this is not a style choice.
  // `Converged::Relative(tol, "u_wall")` alone -- the obvious reading of spec
  // §5.3's single-measure example -- would leave the flux residual untested. A
  // quasi-Newton step is a least-squares fit over the CONCATENATED residual
  // and is perfectly capable of zeroing the measured block at the expense of
  // the unmeasured one: measured on the sibling two-field-transfer test, IQN-ILS
  // drove the measured residual to exactly zero on sweep 3 while the other
  // still stood at 0.8, and the implicit coupling group reported convergence 10% away from the
  // true fixed point with a monotone residual history the whole way. The
  // runtime now rejects a partially measured implicit coupling group with E-ITER-MEASURE, so
  // deleting either line below turns this demo into that error rather than
  // into a wrong answer -- but the error is the backstop, not the design.
  //
  // Relative, not Absolute, because the two field transfers differ by more than an
  // order of magnitude in scale (about 300 K against 20 W/m^2), so one
  // absolute tolerance cannot mean the same thing for both.
  fuse::ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(kTol, "u_wall"));
  crit.add(fuse::Converged::Relative(kTol, "q_wall"));

  g.iterate({flux, temp})
      .name("dirichlet_neumann")
      .until(std::move(crit))
      .max_iterations(max_iterations)
      .accelerate(accel);
  return g;
}

/// S7.2's own overload: the IDENTICAL mesh/region/component/field/channel/
/// convergence-criterion setup above, differing only in the iteration
/// region's own final call -- `.newton(cfg)` rather than `.accelerate(accel)`.
/// A second function, not a `bool`/`std::variant` parameter threaded through
/// the one above, so the picard/aitken/iqn-ils/iqn+qr2/severed rows this file
/// already gates on keep calling the UNCHANGED original and stay provably
/// untouched by this addition (cmp against a HEAD build of this same binary
/// is how that claim is checked, not just asserted).
fuse::CouplingGraph build_graph(fuse::NewtonConfig cfg, int max_iterations) {
  fuse::CouplingGraph g;
  const std::string src = "generated:line:" + std::to_string(kStations);

  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "iface_1", .source = src});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "iface_2", .source = src});
  const auto r1 = g.add_region(
      fuse::Region{.name = "iface_1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(
      fuse::Region{.name = "iface_2", .mesh = m2, .dimension = 1});

  const auto cd = g.add_component(fuse::ComponentDecl{
      .name = "dirichlet", .ranks = {0, 1},
      .interval = fuse::Rational{1, kPeriodDen}});
  const auto cn = g.add_component(fuse::ComponentDecl{
      .name = "neumann", .ranks = {1, 1},
      .interval = fuse::Rational{1, kPeriodDen}});

  const auto q_out = g.add_field(cd, fuse::FieldSpec{
      .name = "q_wall", .region = r1, .units = fuse::units::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto u_in = g.add_field(cd, fuse::FieldSpec{
      .name = "u_wall_in", .region = r1, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});
  const auto q_in = g.add_field(cn, fuse::FieldSpec{
      .name = "q_wall_in", .region = r2,
      .units = fuse::units::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto u_out = g.add_field(cn, fuse::FieldSpec{
      .name = "u_wall", .region = r2, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});

  const auto flux = g.add_channel("flux", q_out, q_in);
  const auto temp = g.add_channel("wall_temperature", u_out, u_in);
  g.channel(temp).lag(1).allow_time_level_mismatch(
      "resolved by the Dirichlet-Neumann implicit coupling group");

  fuse::ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(kTol, "u_wall"));
  crit.add(fuse::Converged::Relative(kTol, "q_wall"));

  g.iterate({flux, temp})
      .name("dirichlet_neumann")
      .until(std::move(crit))
      .max_iterations(max_iterations)
      .newton(cfg);
  return g;
}

struct Outcome {
  bool                ok = false;
  std::string         error;
  std::int64_t        periods = 0;
  std::int64_t        sweeps = 0;
  std::int64_t        not_converged = 0;
  std::int64_t        checkpoint_verifications = 0;
  std::int64_t        conservation_checks = 0;
  std::int64_t        conservation_violations = 0;
  double              worst_conservation = 0.0;
  double              u_l2 = 0.0;
  double              u_max = 0.0;
  double              q_l2 = 0.0;
  double              q_max = 0.0;
  double              u_norm = 0.0;
  double              t_final = 0.0;
  double              d_history = 0.0;
  double              n_history = 0.0;
  std::vector<double> u_final;
  std::vector<double> q_final;
  std::vector<double> last_residuals;
  std::string         checkpoints;
  bool                addresses_fixed = false;
};

/// Build, run and measure one configuration. Everything is local, so no two
/// cases can share a Runner, a registry or a component.
Outcome run_case(fuse::AccelConfig accel, int max_iterations,
                 std::int64_t periods, dn::Wiring wiring,
                 bool abort_on_nonconvergence, bool dump = false) {
  Outcome o;
  o.periods = periods;
  fuse::CouplingGraph graph = build_graph(accel, max_iterations);

  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) { o.error = "mesh load: " + r.error().message; return o; }
  }

  const auto lowered = lower(graph);
  if (!lowered.ok()) { o.error = lowered.diags.to_string(); return o; }

  const fuse::ClockSet clocks{graph};
  auto sched = schedule(graph, lowered.dag, clocks,
                        fuse::make_region_sizes(graph, backend));
  if (!sched.ok()) { o.error = sched.diags.to_string(); return o; }

  if (dump) {
    std::printf("=== IR ===\n%s\n=== PLAN ===\n%s\n",
                dump_text(lowered.dag).c_str(),
                sched.plan.to_string().c_str());
  }

  fuse::ComponentSet components;
  auto dirichlet =
      std::make_unique<dn::DirichletSlab>("dirichlet", kStations, wiring);
  auto neumann =
      std::make_unique<dn::NeumannSlab>("neumann", kStations, wiring);
  dn::DirichletSlab* d = dirichlet.get();
  dn::NeumannSlab*   n = neumann.get();
  components.add(std::move(dirichlet));
  components.add(std::move(neumann));

  // Component::register_fields()'s lifetime rule: the registered address must
  // not move for the whole run. Recorded before and compared after, because
  // both idioms that break it -- a swap() of the state and scratch arrays, and
  // a copy-assignment in restore_state() -- move it silently, and the coupler
  // then keeps reading whatever now lives at the old address.
  const double* d_addr = d->registered_provides();
  const double* n_addr = n->registered_provides();

  fuse::FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, graph);
    if (!r.ok()) { o.error = "register_fields: " + r.error().message; return o; }
  }

  fuse::RunnerConfig cfg;
  cfg.abort_on_nonconvergence = abort_on_nonconvergence;
  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry, cfg); !r.ok()) {
    o.error = "initialize: " + r.error().message;
    return o;
  }
  if (!runner.checkpoint_explanations().empty()) {
    o.checkpoints = runner.checkpoint_explanations().front();
  }
  if (const auto r = runner.run(components, periods); !r.ok()) {
    o.error = "run: " + r.error().message;
    return o;
  }

  o.addresses_fixed = (d_addr == d->registered_provides()) &&
                      (n_addr == n->registered_provides());
  o.sweeps = runner.stats().total_iterations;
  o.not_converged = runner.stats().windows_not_converged;
  o.checkpoint_verifications = runner.stats().checkpoint_verifications;
  o.conservation_checks = runner.stats().conservation_checks;
  o.conservation_violations = runner.stats().conservation_violations;
  o.worst_conservation = runner.stats().worst_conservation_error;
  if (!runner.iteration_stats().empty()) {
    o.last_residuals = runner.iteration_stats().front().residual_history;
  }

  const double t = n->time();
  o.t_final = t;
  o.u_final = n->u();
  o.q_final = d->q();
  const auto ue = [&](std::size_t i) { return dn::exact_u(i, kStations, t); };
  const auto qe = [&](std::size_t i) { return dn::exact_q(i, kStations, t); };
  o.u_l2 = dn::l2_error(o.u_final, ue);
  o.u_max = dn::max_error(o.u_final, ue);
  o.q_l2 = dn::l2_error(o.q_final, qe);
  o.q_max = dn::max_error(o.q_final, qe);
  o.u_norm = dn::l2_norm(o.u_final);
  o.d_history = d->history();
  o.n_history = n->history();
  o.ok = true;
  return o;
}

/// S7.2's own overload. A near-duplicate of the body above rather than a
/// shared helper the two funnel through, for the same reason the
/// `build_graph` overload above is a second function and not a branch in the
/// first: the original's every line stays provably untouched.
///
/// `dirichlet_beta`/`neumann_initial_u` both default to the values that make
/// `dn::DirichletSlab`/`dn::NeumannSlab` behave exactly as the picard/aitken/
/// iqn-ils rows' own components do (beta = 0.0 is bit-identical to the affine
/// map; initial_u = NaN means "the component's own today's T_outer_2(0.0)",
/// its own existing default) -- the `newton` row below passes neither, so it
/// solves the IDENTICAL physical problem the other rows do. `newton-nl`
/// passes beta = 0.1 for a real nonlinearity.
Outcome run_case(fuse::NewtonConfig cfg, int max_iterations,
                 std::int64_t periods, dn::Wiring wiring,
                 bool abort_on_nonconvergence, double dirichlet_beta = 0.0,
                 double neumann_initial_u =
                     std::numeric_limits<double>::quiet_NaN(),
                 bool dump = false) {
  Outcome o;
  o.periods = periods;
  fuse::CouplingGraph graph = build_graph(cfg, max_iterations);

  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) { o.error = "mesh load: " + r.error().message; return o; }
  }

  const auto lowered = lower(graph);
  if (!lowered.ok()) { o.error = lowered.diags.to_string(); return o; }

  const fuse::ClockSet clocks{graph};
  auto sched = schedule(graph, lowered.dag, clocks,
                        fuse::make_region_sizes(graph, backend));
  if (!sched.ok()) { o.error = sched.diags.to_string(); return o; }

  if (dump) {
    std::printf("=== IR ===\n%s\n=== PLAN ===\n%s\n",
                dump_text(lowered.dag).c_str(),
                sched.plan.to_string().c_str());
  }

  fuse::ComponentSet components;
  auto dirichlet = std::make_unique<dn::DirichletSlab>(
      "dirichlet", kStations, wiring, dirichlet_beta);
  auto neumann = std::make_unique<dn::NeumannSlab>(
      "neumann", kStations, wiring, neumann_initial_u);
  dn::DirichletSlab* d = dirichlet.get();
  dn::NeumannSlab*   n = neumann.get();
  components.add(std::move(dirichlet));
  components.add(std::move(neumann));

  const double* d_addr = d->registered_provides();
  const double* n_addr = n->registered_provides();

  fuse::FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, graph);
    if (!r.ok()) { o.error = "register_fields: " + r.error().message; return o; }
  }

  fuse::RunnerConfig cfg2;
  cfg2.abort_on_nonconvergence = abort_on_nonconvergence;
  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry, cfg2); !r.ok()) {
    o.error = "initialize: " + r.error().message;
    return o;
  }
  if (!runner.checkpoint_explanations().empty()) {
    o.checkpoints = runner.checkpoint_explanations().front();
  }
  if (const auto r = runner.run(components, periods); !r.ok()) {
    o.error = "run: " + r.error().message;
    return o;
  }

  o.addresses_fixed = (d_addr == d->registered_provides()) &&
                      (n_addr == n->registered_provides());
  o.sweeps = runner.stats().total_iterations;
  o.not_converged = runner.stats().windows_not_converged;
  o.checkpoint_verifications = runner.stats().checkpoint_verifications;
  o.conservation_checks = runner.stats().conservation_checks;
  o.conservation_violations = runner.stats().conservation_violations;
  o.worst_conservation = runner.stats().worst_conservation_error;
  if (!runner.iteration_stats().empty()) {
    o.last_residuals = runner.iteration_stats().front().residual_history;
  }

  const double t = n->time();
  o.t_final = t;
  o.u_final = n->u();
  o.q_final = d->q();
  const auto ue = [&](std::size_t i) { return dn::exact_u(i, kStations, t); };
  const auto qe = [&](std::size_t i) { return dn::exact_q(i, kStations, t); };
  o.u_l2 = dn::l2_error(o.u_final, ue);
  o.u_max = dn::max_error(o.u_final, ue);
  o.q_l2 = dn::l2_error(o.q_final, qe);
  o.q_max = dn::max_error(o.q_final, qe);
  o.u_norm = dn::l2_norm(o.u_final);
  o.d_history = d->history();
  o.n_history = n->history();
  o.ok = true;
  return o;
}

/// The orders S7.2's own `NewtonRegion.QuadraticRateMeasuredOnTheNonlinear
/// Problem` (test/runtime/newton_region_test.cpp) fits, reproduced here so
/// the claim is a number THIS binary also measures and gates on, not only
/// the unit test: q_k over every consecutive residual triple still above a
/// round-off floor relative to the period's own ||F_0||.
std::vector<double> fitted_orders(const std::vector<double>& h) {
  std::vector<double> orders;
  if (h.size() < 4) return orders;
  const double floor = h.front() * 1.0e-10;
  for (std::size_t k = 1; k + 1 < h.size(); ++k) {
    if (h[k - 1] <= floor || h[k] <= floor || h[k + 1] <= floor) continue;
    if (h[k] >= h[k - 1] || h[k + 1] >= h[k]) continue;
    orders.push_back(std::log(h[k + 1] / h[k]) / std::log(h[k] / h[k - 1]));
  }
  return orders;
}

void print_row(const char* label, const Outcome& o) {
  std::printf("  %-9s %6lld %8lld %6lld  %10.3e %10.3e  %10.3e %10.3e\n", label,
              static_cast<long long>(o.periods),
              static_cast<long long>(o.sweeps),
              static_cast<long long>(o.not_converged), o.u_l2, o.u_max, o.q_l2,
              o.q_max);
}

void print_residuals(const char* label, const std::vector<double>& h) {
  std::printf("  %-9s %4zu sweeps:", label, h.size());
  const std::size_t head = h.size() < 6 ? h.size() : 6;
  for (std::size_t i = 0; i < head; ++i) std::printf(" %.2e", h[i]);
  if (h.size() > head + 1) std::printf(" ...");
  if (h.size() > head) std::printf(" %.2e", h.back());
  std::printf("\n");
}

int g_failures = 0;

void check(bool cond, const std::string& what) {
  if (cond) return;
  std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  ++g_failures;
}

/// Largest per-station disagreement between two runs' interface temperatures.
[[nodiscard]] double disagreement(const Outcome& a, const Outcome& b) {
  double            m = 0.0;
  const std::size_t n = std::min(a.u_final.size(), b.u_final.size());
  for (std::size_t i = 0; i < n; ++i) {
    m = std::max(m, std::abs(a.u_final[i] - b.u_final[i]));
  }
  return m;
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

  bool dump = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump") == 0) dump = true;
  }

  const fuse::AccelConfig picard_cfg{.kind = fuse::AccelKind::None};
  const fuse::AccelConfig aitken_cfg{.kind = fuse::AccelKind::Aitken,
                                     .initial_relaxation = 0.5};
  // No column filter, deliberately, and measured. AccelConfig defaults to
  // FilterKind::QR2 with limit = 1e-2, which is preCICE's default; on THIS
  // problem, whose secant columns are well conditioned, that default throws
  // away information the least-squares fit could still have used. The iqn+qr2
  // row of the table below is the same method at the same tolerance with the
  // default filter: roughly 30x the sweeps, and some windows that do not
  // converge at all. accelerator.hpp says as much in prose ("Filtering is not
  // free on well-conditioned problems"); the table is the number.
  const fuse::AccelConfig iqn_cfg{.kind = fuse::AccelKind::IQN_ILS,
                                  .initial_relaxation = 0.5,
                                  .reuse = 8,
                                  .filter = fuse::FilterKind::None};
  const fuse::AccelConfig iqn_qr2_cfg{.kind = fuse::AccelKind::IQN_ILS,
                                      .initial_relaxation = 0.5,
                                      .reuse = 8,
                                      .filter = fuse::FilterKind::QR2,
                                      .limit = 1.0e-2};

  // One sweep per coupling cycle is exactly the classical explicit staggered scheme.
  // The runner rejects to call it converged, which is why it runs lenient.
  const Outcome expl =
      run_case(picard_cfg, 1, kPeriods, dn::Wiring::Connected, false);
  // --dump prints the iterated plan, not the explicit one: the implicit coupling group line
  // carries max_iter, so dumping the one-sweep configuration would show
  // max_iter=1 and read like an explicit scheme.
  const Outcome picard = run_case(picard_cfg, kMaxIterations, kPeriods,
                                  dn::Wiring::Connected, true, dump);
  const Outcome aitken = run_case(aitken_cfg, kMaxIterations, kPeriods,
                                  dn::Wiring::Connected, true);
  const Outcome iqn =
      run_case(iqn_cfg, kMaxIterations, kPeriods, dn::Wiring::Connected, true);
  const Outcome iqn_qr2 = run_case(iqn_qr2_cfg, kMaxIterations, kPeriods,
                                   dn::Wiring::Connected, false);
  // The exchange runs; it lands in a decoy. See dn::Wiring.
  const Outcome severed = run_case(picard_cfg, kMaxIterations, kPeriods,
                                   dn::Wiring::Severed, false);
  // The asymptotic explicit-vs-implicit pair, past the initial transient.
  const Outcome expl_long =
      run_case(picard_cfg, 1, kLongPeriods, dn::Wiring::Connected, false);
  const Outcome iqn_long = run_case(iqn_cfg, kMaxIterations, kLongPeriods,
                                    dn::Wiring::Connected, true);
  // Bit-for-bit reproducibility of the accelerated configuration, which is the
  // one with the most floating-point machinery behind it.
  const Outcome repeat =
      run_case(iqn_cfg, kMaxIterations, kPeriods, dn::Wiring::Connected, true);

  // S7.2: the SAME affine graph (beta = 0.0, the default, bit-identical to
  // every row above -- Dn.NonlinearAtBetaZeroIsBitIdenticalToTheAffineMap,
  // test/runtime/newton_region_test.cpp), solved by the native Jacobian-free
  // Newton-Krylov region instead of a quasi-Newton accelerator wrapped around
  // the Picard sweep.
  const fuse::NewtonConfig newton_cfg{};
  const Outcome newton =
      run_case(newton_cfg, kMaxIterations, kPeriods, dn::Wiring::Connected,
               true);
  // Bit-for-bit reproducibility of the Newton configuration too -- the same
  // non-triviality guard gate 7 below applies to iqn/repeat.
  const Outcome newton_repeat =
      run_case(newton_cfg, kMaxIterations, kPeriods, dn::Wiring::Connected,
               true);
  // Informational, like iqn+qr2: a REAL nonlinearity in the wall's own
  // conductivity (beta = 0.1, docs/dev/newton-krylov.md §8.3), rtol loosened
  // to 1e-4 for the SAME reason the dedicated unit test loosens it -- reaching
  // 1e-8 from here needs a near-machine-precision last linear solve this
  // unpreconditioned, matrix-free iteration cannot complete within a sane
  // GMRES budget, and the quadratic order is already visible two orders of
  // magnitude earlier. Not in the `all[]` run-completed list below for the
  // same reason iqn_qr2 is not: a different physical problem than the other
  // rows solve, with its own informational gate (order ~= 2.0) below instead.
  fuse::NewtonConfig newton_nl_cfg{};
  newton_nl_cfg.rtol = 1.0e-4;
  const Outcome newton_nl =
      run_case(newton_nl_cfg, kMaxIterations, kPeriods, dn::Wiring::Connected,
               false, /*dirichlet_beta=*/0.1);

  // Every configuration a gate below reads must have run. iqn+qr2 and
  // newton-nl are deliberately NOT in this list: they are informational rows
  // (see their own construction above), run with abort_on_nonconvergence off
  // by design, and a failure to converge is a documented runtime outcome
  // rather than a regression. Their failure is printed, not fatal.
  const Outcome* all[] = {&expl,      &picard,   &aitken,  &iqn,     &severed,
                          &expl_long, &iqn_long, &repeat,  &newton,  &newton_repeat};
  const char*    names[] = {"explicit",  "picard",   "aitken",     "iqn-ils",
                            "severed",   "expl-long", "iqn-long",  "repeat",
                            "newton",    "newton-repeat"};
  constexpr std::size_t kCases = sizeof(all) / sizeof(all[0]);
  for (std::size_t i = 0; i < kCases; ++i) {
    if (!all[i]->ok) {
      std::fprintf(stderr, "FAIL: configuration '%s' did not run: %s\n",
                   names[i], all[i]->error.c_str());
      return 1;
    }
  }
  if (!iqn_qr2.ok) {
    std::printf("NOTE: the informational iqn+qr2 row did not complete: %s\n",
                iqn_qr2.error.c_str());
  }
  if (!newton_nl.ok) {
    std::printf("NOTE: the informational newton-nl row did not complete: %s\n",
                newton_nl.error.c_str());
  }

  std::printf(
      "Demo A -- implicit Dirichlet-Neumann coupling of two heat conduction "
      "domains\n"
      "  interface stations : %zu, in %zu conductivity strips, "
      "r = k1*L2/(k2*L1) = 0.50 0.75 0.90 0.96\n"
      "  coupling period    : H = 1/%lld, quasi-steady. Both participants "
      "evaluate their\n"
      "                       exact steady response in closed form, so they "
      "carry NO\n"
      "                       discretisation error and the whole error below "
      "is the\n"
      "                       coupler's, measured against the analytic "
      "two-layer wall\n"
      "                         u* = (T2 + r T1)/(1 + r),   "
      "q* = (T1 - T2)/(L1/k1 + L2/k2)\n"
      "  convergence        : Relative(%g) on BOTH channels, u_wall AND "
      "q_wall\n\n",
      kStations, dn::kStrips, static_cast<long long>(kPeriodDen), kTol);

  std::printf("%s\n", picard.checkpoints.c_str());

  std::printf(
      "             periods   sweeps  !conv    L2 err u/K  max err u/K    "
      "L2 err q     max q\n");
  print_row("explicit", expl);
  print_row("picard", picard);
  print_row("aitken", aitken);
  print_row("iqn-ils", iqn);
  print_row("iqn+qr2", iqn_qr2);
  print_row("severed", severed);
  print_row("expl-long", expl_long);
  print_row("iqn-long", iqn_long);
  print_row("newton", newton);
  std::printf(
      "  NOTE: 'newton' row's own sweeps column (%lld) counts one per "
      "coupling period, not one per Newton step -- a Newton-selected region "
      "bypasses the Picard sweep loop entirely (docs/dev/newton-krylov.md "
      "§8.1) and so does not accumulate `total_iterations` the way the rows "
      "above do. The final period alone took %zu Newton step(s) (its own "
      "residual history below).\n",
      static_cast<long long>(newton.sweeps), newton.last_residuals.size());
  std::printf("\n");

  std::printf("residual history of the final period (2-norm of the "
              "concatenated residual)\n");
  print_residuals("picard", picard.last_residuals);
  print_residuals("aitken", aitken.last_residuals);
  print_residuals("iqn-ils", iqn.last_residuals);
  print_residuals("severed", severed.last_residuals);
  print_residuals("newton", newton.last_residuals);
  std::printf("\n");

  // The nonlinear, informational row: dn_heat's own quadratic-rate claim,
  // measured by THIS binary rather than only test/runtime/newton_region_
  // test.cpp's NewtonRegion.QuadraticRateMeasuredOnTheNonlinearProblem.
  if (newton_nl.ok) {
    print_residuals("newton-nl", newton_nl.last_residuals);
    const std::vector<double> orders = fitted_orders(newton_nl.last_residuals);
    std::printf("  newton-nl fitted order(s) from consecutive residual "
                "triples above the round-off floor:");
    for (const double q : orders) std::printf(" %.2f", q);
    std::printf("%s\n", orders.empty() ? " (none -- too few samples "
                                          "above the floor)"
                                        : "");
  }
  std::printf("\n");

  std::printf(
      "implicit vs explicit, cold      : %.3e K -> %.3e K, factor %.2e\n"
      "implicit vs explicit, asymptotic: %.3e K -> %.3e K, factor %.2e\n"
      "implicit vs SEVERED exchange    : %.3e K -> %.3e K, factor %.2e\n"
      "sweeps: Picard %lld | Aitken %lld (%.1fx) | IQN-ILS %lld (%.1fx) | "
      "IQN-ILS+QR2(1e-2) %lld (%.1fx)\n"
      "rollback verifications %lld; advances surviving rollback %g of %lld "
      "sweeps over %lld periods\n"
      "conservation checks %lld, violations %lld, worst %.3e\n"
      "accelerators agree to: Picard-Aitken %.3e K, Picard-IQN %.3e K\n",
      expl.u_l2, picard.u_l2, expl.u_l2 / picard.u_l2, expl_long.u_l2,
      iqn_long.u_l2, expl_long.u_l2 / iqn_long.u_l2, severed.u_l2, picard.u_l2,
      severed.u_l2 / picard.u_l2, static_cast<long long>(picard.sweeps),
      static_cast<long long>(aitken.sweeps),
      static_cast<double>(picard.sweeps) / static_cast<double>(aitken.sweeps),
      static_cast<long long>(iqn.sweeps),
      static_cast<double>(picard.sweeps) / static_cast<double>(iqn.sweeps),
      static_cast<long long>(iqn_qr2.sweeps),
      static_cast<double>(picard.sweeps) / static_cast<double>(iqn_qr2.sweeps),
      static_cast<long long>(picard.checkpoint_verifications), picard.d_history,
      static_cast<long long>(picard.sweeps), static_cast<long long>(kPeriods),
      static_cast<long long>(picard.conservation_checks),
      static_cast<long long>(picard.conservation_violations),
      picard.worst_conservation, disagreement(picard, aitken),
      disagreement(picard, iqn));
  std::printf("\n");

  // -------------------------------------------------------------------------
  // The gates. Exit status is on the physics, not on "did not crash".
  // -------------------------------------------------------------------------

  // 1. Every configuration that claims to converge did, to the analytic
  //    answer, without moving a registered address or losing a rollback.
  const Outcome* converging[] = {&picard, &aitken, &iqn, &iqn_long};
  const char*    conv_names[] = {"picard", "aitken", "iqn-ils", "iqn-long"};
  for (std::size_t i = 0; i < 4; ++i) {
    const Outcome&    o = *converging[i];
    const std::string nm = conv_names[i];
    check(o.not_converged == 0,
          nm + ": " + std::to_string(o.not_converged) +
              " window(s) hit max_iterations without converging");
    check(o.u_l2 < kImplicitGate,
          nm + ": interface temperature L2 error " + std::to_string(o.u_l2) +
              " K >= " + std::to_string(kImplicitGate));
    check(o.q_l2 < kImplicitGate,
          nm + ": interface flux L2 error " + std::to_string(o.q_l2) +
              " W/m^2 >= " + std::to_string(kImplicitGate));
    check(o.addresses_fixed,
          nm + ": a registered field address moved during the run");
    // Exactly one advance per coupling cycle survives; every other sweep was rolled
    // back. q_ and u_ are written from scratch on every sweep, so a runner
    // that never restored anything would still reach the same fixed point --
    // dn::*Slab::history_ is the only state in which a missing rollback is
    // visible at all, and the runtime's byte comparison of registered storage
    // cannot see it either, because it is private to the component.
    check(o.d_history == static_cast<double>(o.periods) &&
              o.n_history == static_cast<double>(o.periods),
          nm + ": private component state accumulated to " +
              std::to_string(o.d_history) + "/" + std::to_string(o.n_history) +
              " over " + std::to_string(o.sweeps) +
              " sweeps; with a working rollback it must be exactly " +
              std::to_string(o.periods) +
              ", one surviving advance per coupling period");
    check(o.checkpoint_verifications > 0,
          nm + ": the runtime never verified a rollback, so the checkpoint "
               "claim is untested");
    check(o.sweeps > o.periods,
          nm + ": " + std::to_string(o.sweeps) + " sweeps over " +
              std::to_string(o.periods) +
              " periods is not an iteration, it is an explicit scheme");
  }

  // 1b. The Newton-selected region, gated separately from gate 1 above: its
  //     own "sweeps" (runner.stats().total_iterations) counts one per
  //     coupling period, not one per Newton step (docs/dev/newton-krylov.md
  //     §8.1), so the `sweeps > periods` check above would read it as "not
  //     an iteration" -- true of the COUNTER, false of what actually ran.
  //     `checkpoint_verifications` is also not asserted here: it counts the
  //     OUTER checkpoint/rollback byte-verification a REJECTED Picard sweep
  //     triggers, and a Newton-selected region has no rejected-sweep
  //     rollback to verify (its own commit is the only non-restored write,
  //     docs/dev/newton-krylov.md §8.1) -- 0 here is the honest answer, not
  //     a missing check.
  check(newton.not_converged == 0,
        "newton: " + std::to_string(newton.not_converged) +
            " window(s) hit max_nonlinear_iterations without converging");
  check(newton.u_l2 < kImplicitGate,
        "newton: interface temperature L2 error " +
            std::to_string(newton.u_l2) + " K >= " +
            std::to_string(kImplicitGate));
  check(newton.q_l2 < kImplicitGate,
        "newton: interface flux L2 error " + std::to_string(newton.q_l2) +
            " W/m^2 >= " + std::to_string(kImplicitGate));
  check(newton.addresses_fixed,
        "newton: a registered field address moved during the run");
  // docs/dev/newton-krylov.md §8.3b: this track's own fix for a genuine,
  // numerically-harmless bookkeeping defect found while measuring §8.3 --
  // run_region_newton() originally left the generic per-period tick every
  // region (Newton or Picard) is already advanced by before its own body
  // runs un-rolled-back, so this read 2x the period count. Regression-
  // tested directly in test/runtime/newton_region_test.cpp's own
  // NewtonRegion.HistoryAccumulatesExactlyOncePerPeriodNotTwice; asserted
  // here too because this binary is where the ORIGINAL defect was found.
  check(newton.d_history == static_cast<double>(newton.periods) &&
            newton.n_history == static_cast<double>(newton.periods),
        "newton: private component state accumulated to " +
            std::to_string(newton.d_history) + "/" +
            std::to_string(newton.n_history) + " over " +
            std::to_string(newton.periods) +
            " periods; with a working rollback it must be exactly " +
            std::to_string(newton.periods) +
            ", one surviving advance per coupling period");

  // 2. The answer must be an answer. Every ratio gate below would pass
  //    trivially on a field of zeros.
  check(picard.u_norm > 100.0,
        "the converged interface temperature has norm " +
            std::to_string(picard.u_norm) +
            " K, which is not a physical wall temperature");

  // 3. The accelerators must find the SAME fixed point. If they do not, at
  //    most one of them is solving the coupling problem -- which is exactly
  //    what IQN-IMVJ was observed doing on a multi-root map.
  check(disagreement(picard, aitken) < kAgreementGate,
        "Picard and Aitken disagree by " +
            std::to_string(disagreement(picard, aitken)) + " K");
  check(disagreement(picard, iqn) < kAgreementGate,
        "Picard and IQN-ILS disagree by " +
            std::to_string(disagreement(picard, iqn)) + " K");
  // S7.2's own native solver is held to the SAME agreement gate: a
  // different fixed point here would mean Newton and the quasi-Newton
  // accelerators are not solving the same coupling problem.
  check(disagreement(picard, newton) < kAgreementGate,
        "Picard and Newton disagree by " +
            std::to_string(disagreement(picard, newton)) + " K");

  // 4. IMPLICIT BEATS EXPLICIT, cold and asymptotically. Without this the demo
  //    performs implicit coupling without demonstrating that it bought
  //    anything.
  check(expl.not_converged == kPeriods,
        "the explicit configuration reported " +
            std::to_string(expl.not_converged) +
            " unconverged windows; one sweep per period must be reported "
            "unconverged in all " + std::to_string(kPeriods));
  check(expl.u_l2 > kExplicitFloor,
        "the explicit answer's L2 error is only " + std::to_string(expl.u_l2) +
            " K, so this problem does not distinguish the two schemes");
  check(picard.u_l2 * kEvidenceRatio < expl.u_l2,
        "the implicit error " + std::to_string(picard.u_l2) +
            " K is not decisively below the explicit error " +
            std::to_string(expl.u_l2) + " K");
  check(expl_long.u_l2 > kExplicitFloor,
        "past its initial transient the explicit scheme's L2 error is only " +
            std::to_string(expl_long.u_l2) +
            " K, so the cold comparison above was measuring the initial "
            "condition rather than the splitting error");
  check(iqn_long.u_l2 * kEvidenceRatio < expl_long.u_l2,
        "asymptotically, the implicit error " + std::to_string(iqn_long.u_l2) +
            " K is not decisively below the explicit error " +
            std::to_string(expl_long.u_l2) + " K");

  // 5. THE SABOTAGE GATE. A severed exchange still converges -- in two sweeps,
  //    monotonically, because a constant map has a trivial fixed point -- and
  //    nothing else in the run reports anything wrong. Only this comparison
  //    catches it, and it is the reason the headline number can be believed to
  //    be about the coupling at all.
  check(severed.u_l2 > kSeveredFloor,
        "severing both incoming fields moved the headline error only to " +
            std::to_string(severed.u_l2) +
            " K, so that error does not measure the coupling");
  check(picard.u_l2 * kEvidenceRatio < severed.u_l2,
        "the coupled error " + std::to_string(picard.u_l2) +
            " K is not decisively below the severed error " +
            std::to_string(severed.u_l2) +
            " K; the gate would pass with the exchange broken");
  check(severed.not_converged == 0,
        "the severed run failed to converge, which would make the gate above "
        "a convergence test rather than a physics test");

  // 6. Acceleration must actually accelerate, in the order the ladder claims.
  check(aitken.sweeps < picard.sweeps,
        "Aitken took " + std::to_string(aitken.sweeps) +
            " sweeps against Picard's " + std::to_string(picard.sweeps));
  check(iqn.sweeps < aitken.sweeps,
        "IQN-ILS took " + std::to_string(iqn.sweeps) +
            " sweeps against Aitken's " + std::to_string(aitken.sweeps));
  check(static_cast<double>(iqn.sweeps) <
            kAccelRatio * static_cast<double>(picard.sweeps),
        "IQN-ILS took " + std::to_string(iqn.sweeps) +
            " sweeps against Picard's " + std::to_string(picard.sweeps) +
            "; a quasi-Newton method must do far better than that");
  check(picard.sweeps > 10 * kPeriods,
        "Picard converged in " + std::to_string(picard.sweeps) +
            " sweeps over " + std::to_string(kPeriods) +
            " periods, so the acceleration comparison has no room in it");

  // 7. Reproducibility, with the non-triviality guard: two runs of all zeros
  //    are also bit-identical.
  const bool identical =
      repeat.u_final.size() == iqn.u_final.size() &&
      repeat.q_final.size() == iqn.q_final.size() &&
      std::memcmp(repeat.u_final.data(), iqn.u_final.data(),
                  iqn.u_final.size() * sizeof(double)) == 0 &&
      std::memcmp(repeat.q_final.data(), iqn.q_final.data(),
                  iqn.q_final.size() * sizeof(double)) == 0;
  check(identical, "two runs of the same configuration were not bit-identical");
  check(repeat.sweeps == iqn.sweeps,
        "two runs of the same configuration took different sweep counts");
  std::printf("reproducibility: %s over %zu values, on a non-trivial state "
              "(||u||_2 = %.6f K, min %.3f K, max %.3f K)\n",
              identical ? "bit-identical" : "DIFFERENT", iqn.u_final.size(),
              iqn.u_norm,
              *std::min_element(iqn.u_final.begin(), iqn.u_final.end()),
              *std::max_element(iqn.u_final.begin(), iqn.u_final.end()));

  // 7b. The same non-triviality guard, for Newton: GMRES/MFFD/line-search
  //     all carry floating-point machinery of their own, none of it seeded
  //     from anything but the components' own deterministic advance().
  const bool newton_identical =
      newton_repeat.u_final.size() == newton.u_final.size() &&
      newton_repeat.q_final.size() == newton.q_final.size() &&
      std::memcmp(newton_repeat.u_final.data(), newton.u_final.data(),
                  newton.u_final.size() * sizeof(double)) == 0 &&
      std::memcmp(newton_repeat.q_final.data(), newton.q_final.data(),
                  newton.q_final.size() * sizeof(double)) == 0;
  check(newton_identical,
        "two runs of the same Newton configuration were not bit-identical");
  check(newton_repeat.sweeps == newton.sweeps,
        "two runs of the same Newton configuration took different sweep "
        "counts");

  // 8. The derived checkpoint set was printed, and it names both participants.
  //    The user never wrote the set down, so a set they cannot read is a set
  //    they cannot disagree with.
  check(picard.checkpoints.find("dirichlet") != std::string::npos &&
            picard.checkpoints.find("neumann") != std::string::npos,
        "the derived checkpoint set does not name both participants");

  // 9. Conservation ran. Extensive on the flux field transfer is what makes compilation into the dataflow graph
  //    emit Constrain; with no check having run, "0 violations" says nothing.
  check(picard.conservation_checks > 0,
        "no conservation check ran, so the reported violation count is "
        "vacuous");
  check(picard.conservation_violations == 0,
        std::to_string(picard.conservation_violations) +
            " conservation violation(s) on coincident interfaces");

  if (g_failures != 0) {
    std::fprintf(stderr, "\n%d gate(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nall gates passed\n");
  return 0;
}
