#pragma once

#include <vector>

/// The canonical descriptor fixture: every optional member at a NON-DEFAULT
/// value. Shared by the golden-bytes test and the tool that regenerates the
/// golden file, so the two cannot describe different graphs -- which is the one
/// way a golden test can pass while the frozen bytes mean nothing.
///
/// A round trip over defaults proves nothing about a member: zero bytes written
/// and zero bytes read agree.

#include "fuse/model/coupling_graph.hpp"

namespace fuse::testing {

inline CouplingGraph freeze_fixture() {
  using namespace fuse;
  CouplingGraph g;
  const auto mesh = g.add_mesh({.name = "ocean.h5m", .source = "generated:cube(8)"});
  const auto region = g.add_region(
      {.name = "interface", .mesh = mesh, .entity_set = "top", .dimension = 2});

  const auto fluid = g.add_component(
      // Strided, not a block. A component's ranks became an explicit list at
      // descriptor v5, and a frozen file built from two contiguous placements
      // would not exercise the change it exists to pin. Still disjoint from
      // "solid": shared placements are a separate step.
      {.name = "fluid", .ranks = RankSet{std::vector<int>{2, 4, 6}},
       .interval = Rational{1, 4},
       // LaunchMode::Mpmd at a non-default appnum: descriptor v9's reason
       // for the version bump (include/fuse/model/launch.hpp).
       .launch = LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1}});
  const auto solid = g.add_component(
      {.name = "solid", .ranks = {7, 2}, .interval = Rational{1, 3},
       // The other non-default launch mode, with every LaunchSpec member at
       // a non-default value: a round trip over defaults proves nothing
       // about a member (this file's own opening comment).
       .launch = LaunchSpec{.mode = LaunchMode::Connect,
                            .appnum = -1,
                            .is_acceptor = true,
                            .port_file = "/tmp/fuse-connect.port",
                            .service_name = "fuse-solid-fluid"}});
  // A plugin-launched component: `LaunchMode::Plugin` is the one non-default
  // enumerator neither `fluid` nor `solid` could reach (descriptor v10's
  // reason for the bump, S6 phase 4). Fieldless -- `plugin_path` is the one
  // member this component exists to pin at a non-default value. It sits
  // before the coupler only because the v10 bytes were frozen that way;
  // since couplers carry `index` in the YAML, the order no longer matters
  // for the round trip (YamlIo.ACouplerAuthoredBetweenPhysicsComponentsKeepsItsIndex).
  (void)g.add_component(
      {.name = "gas", .ranks = {3, 5}, .interval = Rational{1, 2},
       .launch = LaunchSpec{.mode = LaunchMode::Plugin,
                            .plugin_path = "/opt/fuse-plugins/ideal-gas.so"}});
  // A coupler: `ComponentKind::Coupler` is the one non-default enumerator a
  // freeze fixture built only from `add_component()` calls could never
  // reach, and `add_coupler()` is the API that reaches it. Placed disjoint
  // from `fluid`, `solid` and `gas` so the frozen bytes also exercise a real
  // fourth placement, not a same_as-elided one.
  const auto coupler = g.add_coupler({.name = "midpoint", .ranks = {9, 1}});

  Units kilopascal{.length = -1, .mass = 1, .time = -2};
  kilopascal.scale = Rational{1000};

  const auto q_out = g.add_field(fluid, {.name = "heat_flux",
                                         .region = region,
                                         .disc = Discretization::DiscontinuousGLL,
                                         .order = 4,
                                         .components = 3,
                                         .units = kilopascal,
                                         .conservation = Conservation::Extensive,
                                         .intent = Intent::Provides});
  const auto q_in = g.add_field(solid, {.name = "heat_flux",
                                        .region = region,
                                        .disc = Discretization::FV,
                                        .components = 3,
                                        .units = kilopascal,
                                        .conservation = Conservation::Extensive,
                                        .intent = Intent::Requires});
  // A second writer and the rule for combining the two, so the merge table is
  // non-empty in the frozen bytes. A descriptor field that is only ever
  // serialized at its default is not covered by a round trip that passes.
  const auto q_out2 = g.add_field(fluid, {.name = "heat_flux_radiative",
                                          .region = region,
                                          .disc = Discretization::FV,
                                          .components = 3,
                                          .units = kilopascal,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Provides});
  (void)g.add_channel("flux_radiative", q_out2, q_in);
  g.merge(q_in, MergeOp::Sum);

  const auto ch = g.add_channel("flux", q_out, q_in);
  g.channel(ch)
      .accumulate(AccumOp::Integral)
      .spacetime(TemporalBasis::Linear, 1)
      .remap_method("conservative-intersection")
      .remap_on(coupler)
      .relax(0.7)
      .extrapolate(2)
      .lag(1);

  // An iteration region, at non-default values throughout -- this fixture
  // had none before the multirate-window-on-the-Runner track, so
  // IterationRegionDecl's own members (max_iterations, convergence,
  // acceleration) were never exercised by the golden bytes either; adding
  // one region closes that gap as well as pinning the new `window_scope`
  // member (fuse::CouplingWindowScope, v7's reason for the descriptor
  // version bump).
  g.iterate({ch})
      .name("outer")
      .max_iterations(12)
      .until(Converged::Relative(1e-7, "heat_flux"))
      .accelerate(AccelConfig{.kind = AccelKind::Aitken,
                             .initial_relaxation = 0.2,
                             .omega = 0.6,
                             .reuse = 6,
                             .filter = FilterKind::QR1,
                             .limit = 5e-3,
                             .scale_fields = false});
  {
    CouplingWindowScope scope;
    scope.explicit_interval = Rational{3, 10};
    scope.order = 3;
    scope.policies.push_back({.component = "fluid", .policy = DtPolicy::Fixed});
    scope.policies.push_back(
        {.component = "solid", .policy = DtPolicy::Adjustable});
    // Non-default like the rest of this scope (v8's reason for the
    // descriptor version bump): the default is Jacobi.
    scope.iteration = WaveformIteration::GaussSeidel;
    const_cast<IterationRegionDecl&>(g.iteration_region(IterationRegionIdx{0}))
        .window_scope = std::move(scope);
  }
  return g;
}
}  // namespace fuse::testing
