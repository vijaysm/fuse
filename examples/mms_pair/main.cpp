// Demo A of spec Section 11: two in-process C++ components
// exchanging a field through graph -> lower -> schedule -> run, with an
// analytic answer to check the result against.
//
//   left   1D heat equation, explicit Euler, T_t = alpha T_xx on [0,1],
//          T(0,t) = T(1,t) = 0, T(x,0) = sin(pi x).  Provides "T".
//   right  dR/dt = kappa (T - R), integrated in closed form.  Requires "T_bc".
//
// The coupling cycle is 1/1000 and `left` subcycles four times inside it, so
// the field transfer's Mean reduction actually reduces something. The exchange is
// explicit and lagged by one coupling cycle (see Runner::Impl::step_period), which is
// the dominant error in `right`'s answer and the thing this example measures.
//
// Both meshes are coincident 64-cell lines: the Remap is an identity
// copy, so that is the only configuration in which it is the correct operator.
//
// Exit status is 0 only if the run succeeded, no conservation violation was
// reported, and both components met their error gates -- ctest runs this
// binary, so a silent regression here fails the suite.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

#include <Kokkos_Core.hpp>

#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include "heat_component.hpp"

namespace {

constexpr std::size_t  kCells   = 64;
constexpr double       kAlpha   = 0.01;
constexpr double       kKappa   = 2.0;
constexpr std::int64_t kPeriods = 100;
/// Coupling cycle H = 1/1000; `left` takes four steps of H/4 inside it.
constexpr std::int64_t kPeriodDen  = 1000;
constexpr std::int64_t kSubsteps   = 4;

/// `left` is exact to ~1e-6 at t = 0.1 (its spatial error is O(dx^2) with
/// dx = 1/64 and its time error O(dt) with dt = 1/4000), so 1e-4 catches a
/// broken solver without pretending the discretisation is not there.
constexpr double kHeatGate = 1.0e-4;
/// `right`'s error is the coupler's: ~H * dR/dt at t = 0.1, about 1.2e-3.
constexpr double kRelaxGate = 3.0e-3;
/// ...and it must be a small fraction of what a run with no exchange gives.
constexpr double kCouplingEvidenceRatio = 0.05;

fuse::CouplingGraph build_graph(std::size_t cells) {
  fuse::CouplingGraph g;
  const auto m_left = g.add_mesh(fuse::MeshDecl{
      .name = "left_mesh",
      .source = "generated:line:" + std::to_string(cells)});
  const auto m_right = g.add_mesh(fuse::MeshDecl{
      .name = "right_mesh",
      .source = "generated:line:" + std::to_string(cells)});
  const auto r_left = g.add_region(fuse::Region{
      .name = "left", .mesh = m_left, .dimension = 1});
  const auto r_right = g.add_region(fuse::Region{
      .name = "right", .mesh = m_right, .dimension = 1});

  const auto left = g.add_component(fuse::ComponentDecl{
      .name = "left", .ranks = {0, 1},
      .interval = fuse::Rational{1, kPeriodDen * kSubsteps}});
  const auto right = g.add_component(fuse::ComponentDecl{
      .name = "right", .ranks = {1, 1},
      .interval = fuse::Rational{1, kPeriodDen}});

  // Conservation::Extensive is what makes compilation into the dataflow graph emit the Constrain
  // operator; without it the printed check count is zero and says nothing.
  const auto t_out = g.add_field(left, fuse::FieldSpec{
      .name = "T", .region = r_left, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto t_in = g.add_field(right, fuse::FieldSpec{
      .name = "T_bc", .region = r_right, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});

  const auto ch = g.add_channel("T_link", t_out, t_in);
  g.channel(ch).accumulate(fuse::AccumOp::Mean);
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  // The Runner's coupling hot loop runs on Kokkos' default host execution
  // space (fuse/num/space_kernels.hpp), so every caller -- including this
  // one, which runs no Kokkos kernel of its own -- must initialise Kokkos
  // before touching a Runner. Same pattern as
  // test/support/fuse_test_main.cpp's `Kokkos::ScopeGuard` and
  // examples/brusselator/main.cpp's.
  Kokkos::ScopeGuard kokkos_guard(argc, argv);

  bool dump = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump") == 0) dump = true;
  }

  fuse::CouplingGraph graph = build_graph(kCells);

  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) {
      std::cerr << "mesh load failed: " << r.error().message << "\n";
      return 1;
    }
  }

  const auto lowered = lower(graph);
  if (!lowered.ok()) {
    std::cerr << lowered.diags.to_string();
    return 1;
  }

  const fuse::ClockSet clocks{graph};
  auto sched = schedule(graph, lowered.dag, clocks,
                        fuse::make_region_sizes(graph, lowered.dag, backend));
  if (!sched.ok()) {
    std::cerr << sched.diags.to_string();
    return 1;
  }

  if (dump) {
    std::cout << "=== IR ===\n" << dump_text(lowered.dag)
              << "\n=== PLAN ===\n" << sched.plan.to_string() << "\n";
    // docs/user/mms-pair.md says --dump prints "without running anything,
    // then exits 0": return here rather than falling through into the
    // full gated physics run below, which would cost the entire run and
    // return the gates' own exit status rather than an unconditional 0.
    return 0;
  }

  fuse::ComponentSet components;
  auto left = std::make_unique<mms::HeatComponent>("left", kCells, kAlpha);
  auto right = std::make_unique<mms::RelaxComponent>(
      "right", kCells, kKappa, mms::kPi * mms::kPi * kAlpha);
  mms::HeatComponent*  left_ptr = left.get();
  mms::RelaxComponent* right_ptr = right.get();
  components.add(std::move(left));
  components.add(std::move(right));

  fuse::FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, graph);
    if (!r.ok()) {
      std::cerr << "register_fields failed: " << r.error().message << "\n";
      return 1;
    }
  }

  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry); !r.ok()) {
    std::cerr << "initialize failed: " << r.error().message << "\n";
    return 1;
  }
  if (const auto r = runner.run(components, kPeriods); !r.ok()) {
    std::cerr << "run failed: " << r.error().message << "\n";
    return 1;
  }

  const double heat_err = left_ptr->l2_error();
  const double relax_err = right_ptr->l2_error();
  const double uncoupled = right_ptr->uncoupled_error();

  std::cout << "periods                   : " << runner.stats().periods << "\n"
            << "instructions              : "
            << runner.stats().instructions_executed << "\n"
            << "conservation checks       : "
            << runner.stats().conservation_checks << "\n"
            << "violations                : "
            << runner.stats().conservation_violations << "\n"
            << "worst conservation error  : "
            << runner.stats().worst_conservation_error << "\n"
            << "t                         : " << left_ptr->time() << "\n"
            << "L2 error left  (own disc.): " << heat_err << "\n"
            << "L2 error right (coupled)  : " << relax_err << "\n"
            << "L2 error right (no exch.) : " << uncoupled << "\n"
            << "coupled / uncoupled       : " << relax_err / uncoupled << "\n";

  int status = 0;
  if (runner.stats().conservation_checks == 0) {
    std::cerr << "FAIL: no conservation check ran, so the reported violation "
                 "count is vacuous\n";
    status = 1;
  }
  if (runner.stats().conservation_violations != 0) {
    std::cerr << "FAIL: " << runner.stats().conservation_violations
              << " conservation violation(s) on coincident regions\n";
    status = 1;
  }
  if (!(heat_err < kHeatGate)) {
    std::cerr << "FAIL: left L2 error " << heat_err << " >= " << kHeatGate
              << "\n";
    status = 1;
  }
  if (!(heat_err > 0.0)) {
    std::cerr << "FAIL: left L2 error is exactly zero, so advance() never "
                 "ran\n";
    status = 1;
  }
  if (!(relax_err < kRelaxGate)) {
    std::cerr << "FAIL: right L2 error " << relax_err << " >= " << kRelaxGate
              << "\n";
    status = 1;
  }
  // The gate above is only evidence about the coupling if a run WITHOUT the
  // exchange would miss it. With no exchange R stays zero and the error is
  // ||R_exact||, printed above.
  if (!(relax_err < kCouplingEvidenceRatio * uncoupled)) {
    std::cerr << "FAIL: the coupled error " << relax_err
              << " is not decisively below the no-exchange error " << uncoupled
              << "; the error gate would pass with the coupling switched off\n";
    status = 1;
  }
  return status;
}
