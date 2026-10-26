#pragma once

/// The Brusselator's U/V coupling, expressed as a `fuse::CouplingGraph`
/// coupling window (`fuse::CouplingWindowScope`,
/// docs/user/coupling-windows.md), shared between `examples/brusselator/
/// main.cpp`'s `--driver runner` and `test/verification/
/// brusselator_runner_test.cpp` so the two declare the identical graph.
///
/// WHY BOTH TRANSFER FIELDS LIVE ON `u_region`. `species_component.hpp`'s
/// file header explains the design this mirrors: `UComponent::provided_`
/// (`q`, field `u_out`) is already on the U grid, and
/// `VComponent::provided_` (`P_VU(state_)`, field `v_out`) is V's OWN new
/// state already remapped onto the U grid by `VComponent::advance()`
/// itself, using the weights `main.cpp` builds with `fuse::build_weights`/
/// `fuse::apply_weights` -- never through a graph-level `Remap`. Both of
/// V's window-transfer fields therefore have length `n_u`, not `n_v`
/// (`WindowMember::length = N_u` for both members, in the `--driver
/// window` path this must reproduce bit for bit), and both channels below
/// connect two same-region, same-length fields: the Runner's own channel
/// is an IDENTITY transfer (window_runner_multirate_test.cpp's own
/// "matching-mesh" comment), so no `fuse::Remap` intervenes between the
/// window-scope's own per-member `WindowAdvance` node and the registered
/// storage `register_fields()` binds -- exactly reproducing what the
/// `--driver window` path already does by direct pointer access.
///
/// NAMING. `u_out`/`v_out` match `examples/brusselator/main.cpp`'s own
/// `WindowChannel::name`/`ConvergenceMeasure::field` for the `--driver
/// window` path exactly, so a convergence tolerance printed or logged
/// means the same field transfer under either driver. `u_in`/`v_in` (the
/// `Intent::Requires` half of each component) are new names -- the
/// `--driver window` path never binds them through the graph at all -- 
/// chosen to match `window_runner_multirate_test.cpp`'s own
/// `"<component>_in"` convention (`a_in`, `b_in`).
///
/// WHY NO `.lag()`/`.allow_time_level_mismatch()` ON EITHER CHANNEL. These
/// two channels close a cycle (U feeds V, V feeds U) with neither
/// direction holding a real-time lag, but `src/ir/lowering.cpp`'s
/// `type_check()` resolves the apparent cycle itself for a window-scoped
/// region: a `WaveformIteration::Jacobi` window's own auto-lag
/// (`effective_channel_lag()`, `src/ir/rules_channel.cpp`) authorises the
/// time-level mismatch the compiler itself introduces, with no
/// `.allow_time_level_mismatch()` call needed from this builder (that
/// comment names the exact mechanism). `WaveformIteration::GaussSeidel`
/// (tested here too, for the acceptance criterion's "both waveform
/// iterations") additionally needs the directed graph of UNLAGGED
/// channels to be acyclic (`E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG`,
/// `fuse/schedule/window.hpp`) -- `ch_v2u` below is lagged for exactly
/// that mode, harmlessly ignored under Jacobi
/// (`WindowChannel::lagged`'s own docstring: "silently unused under the
/// default Jacobi").

#include <optional>
#include <string>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/native_mesh.hpp>
#include <fuse/base/units.hpp>
#include <fuse/model/coupling_graph.hpp>

namespace brusselator {

/// The two component ids `add_window_coupling` creates, so a caller can
/// attach its own `Component` instances to them via
/// `ComponentSet`/`register_fields()`. `coupler` is set only when
/// `add_window_coupling`'s own `coupler_ranks` argument was given --
/// `ComponentId{}` (invalid) otherwise.
struct WindowGraphHandles {
  fuse::ComponentId u_component;
  fuse::ComponentId v_component;
  fuse::ChannelId   u_to_v;
  fuse::ChannelId   v_to_u;
  fuse::ComponentId coupler{};
};

/// Adds U, V, their four window-transfer fields and the window-scoped
/// iteration region to `g`. `u_region` must already be the U grid's
/// region (`main.cpp` builds it, alongside `v_region`, before calling
/// this); V's two fields are declared on it too -- see the file header.
/// `iteration_mode == WaveformIteration::GaussSeidel` marks `v_to_u`
/// lagged so `E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG` does not fire; under
/// `Jacobi` this has no effect.
///
/// `coupler_ranks`: Conformance Unit 2's own addition -- when set, adds
/// a THIRD `ComponentKind::Coupler` placement and routes BOTH channels'
/// `.remap(RemapFlags::Conservative).remap_on(coupler)` through it,
/// exactly `examples/layout_driver/main.cpp`'s own `--coupler` flag does
/// for its own single channel. Both channels already connect two fields
/// on the SAME region (`u_region` -- this file's own "WHY BOTH TRANSFER
/// FIELDS LIVE ON u_region" header comment), so the Remap this compiles
/// is the IDENTITY: U as a separately-built participant, V as another,
/// and the coupler as a third, reproduce Unit 1's own `--driver window`/
/// `runner` values bit for bit, which is what makes the executables this
/// graph shape feeds a genuine conformance case rather than a different
/// graph that merely resembles one. `std::nullopt` (the default)
/// reproduces every existing caller's own graph exactly -- no coupler
/// component, no Remap, untouched.
///
/// `u_ranks`/`v_ranks`/`u_launch`/`v_launch`/`coupler_launch`: also
/// Conformance Unit 2's own addition, for `examples/brusselator/
/// participants/brusselator_graph_tool` -- the in-process callers above
/// (the tier-0 harness, `main.cpp`) never pass them, so their default
/// (`RankSet{0, 1}`, `LaunchSpec{}` i.e. `LaunchMode::InProcess`)
/// reproduces exactly what this function declared before these
/// parameters existed. `coupler_launch` rides on the `CouplerDecl`
/// itself: a coupler launched as its own program must take part in
/// `connect()`'s own collective like any other launched entry, which is
/// why that field exists at all (`component.hpp`'s own comment on it).
/// `newton_config`: S7.2 WindowNewton's own addition (newton-krylov.md
/// Sec. 9, trailing and defaulted -- every existing caller passing
/// `accel != AccelKind::Newton` is unaffected). Read only when `accel ==
/// AccelKind::Newton`, in which case `decl.requires_residual_capability`
/// is set unconditionally and `decl.requires_linearization_capability`
/// is set when `newton_config.use_block_jacobi_preconditioner` is --
/// the SAME auto-trigger `fuse::IterationBuilder::accelerate()`/`::
/// newton()` apply for a graph built through that fluent API
/// (newton-krylov.md Sec. 7), reproduced BY HAND here because this
/// function builds `decl` directly rather than through `IterationBuilder`
/// (its own call below, `g.add_iteration_region(decl)`, never passes
/// through that builder either).
[[nodiscard]] inline WindowGraphHandles add_window_coupling(
    fuse::CouplingGraph& g, fuse::RegionId u_region, fuse::Rational dt_u,
    fuse::Rational dt_v, fuse::Rational window, double tolerance,
    int max_iterations, fuse::AccelKind accel,
    fuse::WaveformIteration iteration_mode = fuse::WaveformIteration::Jacobi,
    std::optional<fuse::RankSet> coupler_ranks = std::nullopt,
    fuse::RankSet u_ranks = fuse::RankSet{0, 1},
    fuse::RankSet v_ranks = fuse::RankSet{0, 1},
    fuse::LaunchSpec u_launch = {}, fuse::LaunchSpec v_launch = {},
    fuse::LaunchSpec coupler_launch = {},
    // `newton_config` is LAST, not alongside `iteration_mode` above,
    // DELIBERATELY: every pre-existing call site that passes one or more
    // of the RankSet/LaunchSpec parameters above POSITIONALLY (`test/
    // verification/brusselator_participants_test.cpp`, `examples/
    // brusselator/participants/brusselator_graph_tool.cpp`) would bind
    // the wrong argument to the wrong parameter, or fail to compile
    // outright, if this were inserted any earlier in the list --
    // appending it after every existing parameter, all of which already
    // have defaults, is the only placement that changes nothing for any
    // existing caller.
    fuse::NewtonConfig newton_config = fuse::NewtonConfig{}) {
  using fuse::ComponentDecl;
  using fuse::ComponentKind;
  using fuse::Conservation;
  using fuse::ConvergenceMeasure;
  using fuse::CouplingWindowScope;
  using fuse::DtPolicy;
  using fuse::FieldSpec;
  using fuse::Intent;
  using fuse::IterationRegionDecl;
  using fuse::MeasureKind;
  using fuse::RankSet;
  using fuse::RemapFlags;
  using fuse::WindowMemberPolicy;

  const fuse::ComponentId u = g.add_component(
      ComponentDecl{.name = "U", .ranks = u_ranks, .interval = dt_u, .launch = u_launch});
  const fuse::ComponentId v = g.add_component(
      ComponentDecl{.name = "V", .ranks = v_ranks, .interval = dt_v, .launch = v_launch});

  const fuse::FieldId u_out = g.add_field(
      u, FieldSpec{.name = "u_out",
                   .region = u_region,
                   .units = fuse::units::kDimensionless,
                   .conservation = Conservation::Extensive,
                   .intent = Intent::Provides});
  const fuse::FieldId u_in = g.add_field(
      u, FieldSpec{.name = "u_in",
                   .region = u_region,
                   .units = fuse::units::kDimensionless,
                   .conservation = Conservation::Intensive,
                   .intent = Intent::Requires});
  const fuse::FieldId v_out = g.add_field(
      v, FieldSpec{.name = "v_out",
                   .region = u_region,
                   .units = fuse::units::kDimensionless,
                   .conservation = Conservation::Intensive,
                   .intent = Intent::Provides});
  const fuse::FieldId v_in = g.add_field(
      v, FieldSpec{.name = "v_in",
                   .region = u_region,
                   .units = fuse::units::kDimensionless,
                   .conservation = Conservation::Extensive,
                   .intent = Intent::Requires});

  const fuse::ChannelId ch_u2v = g.add_channel("u2v", u_out, v_in);
  const fuse::ChannelId ch_v2u = g.add_channel("v2u", v_out, u_in);
  // `v_in` is Conservation::Extensive, and validate() requires a channel into
  // an Extensive field to declare that its transfer conserves it
  // (E-CONN-CONSERVATION, KNOWN_ISSUES.md D94). Both fields live on
  // `u_region`, so the transfer is the identity, which does; the coupler
  // branch below adds the same flag together with its placement, so the
  // graph it writes (the reference_*.yaml files) is unchanged by this line.
  g.channel(ch_u2v).conservative();
  if (iteration_mode == fuse::WaveformIteration::GaussSeidel) {
    g.channel(ch_v2u).lag(1).allow_time_level_mismatch(
        "resolved by the U-V coupling window's own sweep, not a real-time lag");
  }

  fuse::ComponentId coupler{};
  if (coupler_ranks.has_value()) {
    coupler = g.add_coupler(fuse::CouplerDecl{.name = "coupler",
                                        .ranks = *coupler_ranks,
                                        .launch = coupler_launch});
    g.channel(ch_u2v).remap(RemapFlags::Conservative).remap_on(coupler);
    g.channel(ch_v2u).remap(RemapFlags::Conservative).remap_on(coupler);
  }

  IterationRegionDecl decl;
  decl.name = "brusselator_window";
  decl.channels = {ch_u2v, ch_v2u};
  decl.max_iterations = max_iterations;
  decl.convergence.add(ConvergenceMeasure{MeasureKind::Relative, tolerance, "u_out"});
  decl.convergence.add(ConvergenceMeasure{MeasureKind::Relative, tolerance, "v_out"});
  decl.acceleration.kind = accel;
  // S7.2 WindowNewton: the SAME auto-trigger `fuse::IterationBuilder::
  // accelerate()`/`::newton()` apply (newton-krylov.md Sec. 7), by hand,
  // since this function never goes through that builder.
  if (accel == fuse::AccelKind::Newton) {
    decl.requires_residual_capability = true;
    decl.newton_config = newton_config;
    if (newton_config.use_block_jacobi_preconditioner) {
      decl.requires_linearization_capability = true;
    }
  }

  CouplingWindowScope scope;
  scope.explicit_interval = window;
  scope.order = 2;
  scope.policies = {WindowMemberPolicy{"U", DtPolicy::Adjustable},
                    WindowMemberPolicy{"V", DtPolicy::Adjustable}};
  scope.iteration = iteration_mode;
  decl.window_scope = scope;
  g.add_iteration_region(decl);

  return WindowGraphHandles{u, v, ch_u2v, ch_v2u, coupler};
}

}  // namespace brusselator
