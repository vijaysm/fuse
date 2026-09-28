// Demo A of spec §11 at the Phase 7 milestone: IMPLICIT coupling.
//
// A classical Dirichlet-Neumann pair of heat conduction domains, iterated to
// its fixed point inside a `g.iterate({...})` region with a derived checkpoint
// set, a convergence criterion covering both channels, and quasi-Newton
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
//   explicit   one sweep per period -- the classical staggered scheme
//   picard     the same region, iterated to convergence, no acceleration
//   aitken     ... with Aitken relaxation
//   iqn-ils    ... with IQN-ILS and no column filter
//   iqn+qr2    ... with IQN-ILS and AccelConfig's default QR2 filter at its
//                  default limit of 1e-2, which is measurably the wrong
//                  setting for this problem (see the note by iqn_qr2_cfg)
//   severed    picard, with both participants' incoming fields bound to decoy
//              arrays: the exchange happens and lands nowhere
//   *-long     explicit and IQN-ILS over 10x as many periods, so the explicit
//              comparison is also made in the asymptotic regime and not only
//              while the initial transient is still the largest term
//
// The gates: the implicit runs must agree with each other and with the
// analytic solution; the implicit answer must beat the explicit one by orders
// of magnitude, both cold and asymptotically; and it must beat the SEVERED one
// by orders of magnitude, which is the measurement proving the headline number
// moves when the coupling breaks. IQN-ILS must cost far fewer sweeps than
// Picard.
//
// WHAT WAS MEASURED WHILE BUILDING THIS, so the claims above are numbers and
// not opinions (32 stations, 20 periods, tolerance 1e-8, debug build):
//
//   * deleting either convergence measure -- i.e. writing the single-measure
//     form spec §5.3's example suggests -- is refused at initialize() with
//     E-ITER-MEASURE naming the unmeasured channel. Both directions were
//     tried. Nothing silently converges to the wrong fixed point here, but
//     only because the runtime refuses to start.
//   * severing the exchange moves the headline number by seven orders of
//     magnitude, and every partial break is caught too:
//         connected                        L2 err u = 7.9e-07 K
//         neumann's flux input severed     L2 err u = 3.03e+01 K
//         dirichlet's temperature severed  L2 err u = 2.41e+02 K
//         both severed                     L2 err u = 3.03e+01 K
//     all three severed runs CONVERGE, in two or three sweeps per period,
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
/// 20 coupling periods of H = 1/20, so t runs over one full cycle of T1.
constexpr std::int64_t kPeriods = 20;
constexpr std::int64_t kPeriodDen = 20;
/// The asymptotic pair. An explicit run's error contracts by only r = 0.96 per
/// period, so after 20 periods it is still dominated by its initial transient
/// (0.96^20 = 0.44). After 200 it is not (0.96^200 = 3.3e-4).
constexpr std::int64_t kLongPeriods = 200;
/// Relative, on both channels. See build_graph().
constexpr double       kTol = 1.0e-8;
/// Generous: Picard on this problem needs several hundred sweeps per period,
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
  // The two channels close a cycle, so one of them must carry a lag for the
  // graph to be schedulable at all. Inside an iteration region that lag is
  // what the sweep resolves; that is precisely the difference between implicit
  // and staggered coupling.
  g.channel(temp).lag(1).allow_time_level_mismatch(
      "resolved by the Dirichlet-Neumann iteration region");

  // THE CRITERION COVERS BOTH CHANNELS, and this is not a style choice.
  // `Converged::Relative(tol, "u_wall")` alone -- the obvious reading of spec
  // §5.3's single-measure example -- would leave the flux residual untested. A
  // quasi-Newton step is a least-squares fit over the CONCATENATED residual
  // and is perfectly capable of zeroing the measured block at the expense of
  // the unmeasured one: measured on the sibling two-channel test, IQN-ILS
  // drove the measured residual to exactly zero on sweep 3 while the other
  // still stood at 0.8, and the region reported convergence 10% away from the
  // true fixed point with a monotone residual history the whole way. The
  // runtime now refuses a partially measured region with E-ITER-MEASURE, so
  // deleting either line below turns this demo into that error rather than
  // into a wrong answer -- but the error is the backstop, not the design.
  //
  // Relative, not Absolute, because the two channels differ by more than an
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

  // One sweep per period is exactly the classical explicit staggered scheme.
  // The runner refuses to call it converged, which is why it runs lenient.
  const Outcome expl =
      run_case(picard_cfg, 1, kPeriods, dn::Wiring::Connected, false);
  // --dump prints the iterated plan, not the explicit one: the region line
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

  // Every configuration a gate below reads must have run. iqn+qr2 is
  // deliberately NOT in this list: it is an informational row measuring what
  // AccelConfig's default filter costs here, it runs with
  // abort_on_nonconvergence off by design, and E-ACCEL-DIVERGED from a
  // deliberately badly filtered least-squares solve is a documented runtime
  // outcome rather than a regression. Its failure is printed, not fatal.
  const Outcome* all[] = {&expl,      &picard,   &aitken, &iqn,
                          &severed,   &expl_long, &iqn_long, &repeat};
  const char*    names[] = {"explicit",  "picard",   "aitken",   "iqn-ils",
                            "severed",   "expl-long", "iqn-long", "repeat"};
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
  std::printf("\n");

  std::printf("residual history of the final period (2-norm of the "
              "concatenated residual)\n");
  print_residuals("picard", picard.last_residuals);
  print_residuals("aitken", aitken.last_residuals);
  print_residuals("iqn-ils", iqn.last_residuals);
  print_residuals("severed", severed.last_residuals);
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
    // Exactly one advance per period survives; every other sweep was rolled
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

  // 8. The derived checkpoint set was printed, and it names both participants.
  //    The user never wrote the set down, so a set they cannot read is a set
  //    they cannot disagree with.
  check(picard.checkpoints.find("dirichlet") != std::string::npos &&
            picard.checkpoints.find("neumann") != std::string::npos,
        "the derived checkpoint set does not name both participants");

  // 9. Conservation ran. Extensive on the flux channel is what makes lowering
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
