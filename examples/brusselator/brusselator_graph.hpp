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

#include <string>

#include <fuse/backend/native_backend.hpp>
#include <fuse/backend/native_mesh.hpp>
#include <fuse/base/units.hpp>
#include <fuse/model/coupling_graph.hpp>

namespace brusselator {

/// The two component ids `add_window_coupling` creates, so a caller can
/// attach its own `Component` instances to them via
/// `ComponentSet`/`register_fields()`.
struct WindowGraphHandles {
  fuse::ComponentId u_component;
  fuse::ComponentId v_component;
  fuse::ChannelId   u_to_v;
  fuse::ChannelId   v_to_u;
};

/// Adds U, V, their four window-transfer fields and the window-scoped
/// iteration region to `g`. `u_region` must already be the U grid's
/// region (`main.cpp` builds it, alongside `v_region`, before calling
/// this); V's two fields are declared on it too -- see the file header.
/// `iteration_mode == WaveformIteration::GaussSeidel` marks `v_to_u`
/// lagged so `E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG` does not fire; under
/// `Jacobi` this has no effect.
[[nodiscard]] inline WindowGraphHandles add_window_coupling(
    fuse::CouplingGraph& g, fuse::RegionId u_region, fuse::Rational dt_u,
    fuse::Rational dt_v, fuse::Rational window, double tolerance,
    int max_iterations, fuse::AccelKind accel,
    fuse::WaveformIteration iteration_mode = fuse::WaveformIteration::Jacobi) {
  using fuse::ComponentDecl;
  using fuse::Conservation;
  using fuse::ConvergenceMeasure;
  using fuse::CouplingWindowScope;
  using fuse::DtPolicy;
  using fuse::FieldSpec;
  using fuse::Intent;
  using fuse::IterationRegionDecl;
  using fuse::MeasureKind;
  using fuse::RankSet;
  using fuse::WindowMemberPolicy;

  const fuse::ComponentId u = g.add_component(
      ComponentDecl{.name = "U", .ranks = RankSet{0, 1}, .interval = dt_u});
  const fuse::ComponentId v = g.add_component(
      ComponentDecl{.name = "V", .ranks = RankSet{0, 1}, .interval = dt_v});

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
  if (iteration_mode == fuse::WaveformIteration::GaussSeidel) {
    g.channel(ch_v2u).lag(1).allow_time_level_mismatch(
        "resolved by the U-V coupling window's own sweep, not a real-time lag");
  }

  IterationRegionDecl decl;
  decl.name = "brusselator_window";
  decl.channels = {ch_u2v, ch_v2u};
  decl.max_iterations = max_iterations;
  decl.convergence.add(ConvergenceMeasure{MeasureKind::Relative, tolerance, "u_out"});
  decl.convergence.add(ConvergenceMeasure{MeasureKind::Relative, tolerance, "v_out"});
  decl.acceleration.kind = accel;

  CouplingWindowScope scope;
  scope.explicit_interval = window;
  scope.order = 2;
  scope.policies = {WindowMemberPolicy{"U", DtPolicy::Adjustable},
                    WindowMemberPolicy{"V", DtPolicy::Adjustable}};
  scope.iteration = iteration_mode;
  decl.window_scope = scope;
  g.add_iteration_region(decl);

  return WindowGraphHandles{u, v, ch_u2v, ch_v2u};
}

/// `fuse::native::NativeMeshBackend` keeps a SEPARATE decomposition per
/// (mesh, placement) pair (`set_placed_view`/`placed_view`,
/// include/fuse/backend/native_backend.hpp): the one `NativeGrid::make`
/// loads (examples/brusselator/brusselator_physics.hpp) is reachable only
/// through the UNPLACED `view()`/`region_size()` overloads. Both
/// `fuse::make_region_sizes` (schedule-time buffer sizing,
/// src/backend/serial_backend.cpp) and `fuse::Runner` itself (every
/// `backend.placed_view(mesh, component)` call in src/runtime/runner.cpp)
/// resolve a FIELD's mesh through its OWNING COMPONENT's placement, never
/// the unplaced default -- so U and V both need their OWN placed view of
/// `u_mesh` registered before `schedule()`/`Runner::initialize()` run, even
/// though this example is single-rank and every placement is identical to
/// the unplaced one (no real migration anywhere in this graph). Call this
/// once, right after `add_window_coupling`, with the SAME `u_mesh`/
/// `nx_u`/`ny_u` `NativeGrid::make("u", ...)` loaded `backend` with.
[[nodiscard]] inline fuse::Result<void> register_native_window_placements(
    fuse::native::NativeMeshBackend& backend, fuse::MeshId u_mesh, int nx_u, int ny_u,
    const WindowGraphHandles& handles) {
  const std::string spec = "quad:" + std::to_string(nx_u) + "x" + std::to_string(ny_u);
  for (const fuse::ComponentId cid : {handles.u_component, handles.v_component}) {
    auto arrays = fuse::native::generate_native_mesh(spec, backend.rank(), backend.num_ranks());
    if (!arrays.ok()) return arrays.error();
    auto view = fuse::native::NativeMeshView::create_from_host(arrays.value(), backend.rank());
    if (!view.ok()) return view.error();
    auto set = backend.set_placed_view(u_mesh, cid, std::move(view.value()));
    if (!set.ok()) return set.error();
  }
  return fuse::Result<void>::success();
}

}  // namespace brusselator
