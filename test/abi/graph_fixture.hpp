#pragma once

/// Test-only fixture shared by test/abi/*.cpp (not installed, not part of
/// the ABI). Builds the small graph most of the ABI's own tests drive
/// through fuse_begin_step()/fuse_end_step(): two single-rank, lockstep
/// (interval 1) components, "upstream" -Provides-> "downstream" over one
/// scalar field on matching single-dimension meshes.
///
/// NOT lagged, on purpose, after measuring what `lag(k)` actually changes
/// (see context.hpp's header comment): outside an iteration region a
/// one-way channel's Remap reads whatever the source's registered storage
/// holds AT THE MOMENT run_period() reaches it -- `lag` affects only
/// TimeLevel typing (who may read which instant) and accumulation-window
/// bookkeeping, not WHEN the value crosses. So an un-lagged channel already
/// delivers a period's export to that SAME period's import once both
/// components have taken their step, provided the import is read no
/// earlier than the exporting component's own end_step() -- which is
/// exactly the ordering fuse_begin_step()/fuse_end_step()'s deferred
/// exchange guarantees, and exactly why this fixture needs no lag and no
/// `allow_time_level_mismatch` override to stay exact.
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "fuse/base/units.hpp"
#include "fuse/model/channel.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/iteration.hpp"
#include "fuse/model/launch.hpp"

namespace fuse::abi_test {

[[nodiscard]] inline fuse::CouplingGraph build_two_component_lag1_graph(
    fuse::Rational upstream_interval = fuse::Rational{1},
    int cells = 4,
    std::optional<fuse::AccumOp> accum = std::nullopt) {
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(
      fuse::MeshDecl{.name = "m1", .source = "generated:line:" + std::to_string(cells)});
  const auto m2 = g.add_mesh(
      fuse::MeshDecl{.name = "m2", .source = "generated:line:" + std::to_string(cells)});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto up = g.add_component(fuse::ComponentDecl{
      .name = "upstream", .ranks = fuse::RankSet{0, 1}, .interval = upstream_interval});
  const auto down = g.add_component(fuse::ComponentDecl{
      .name = "downstream", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto out_f = g.add_field(
      up, fuse::FieldSpec{.name = "out", .region = r1, .components = 1,
                          .units = fuse::units::kKelvin,
                          .conservation = fuse::Conservation::None,
                          .intent = fuse::Intent::Provides});
  const auto in_f = g.add_field(
      down, fuse::FieldSpec{.name = "in", .region = r2, .components = 1,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("link", out_f, in_f);
  g.channel(ch).remap(fuse::RemapFlags::Conservative);
  if (accum.has_value()) g.channel(ch).accumulate(*accum);
  return g;
}

[[nodiscard]] inline std::vector<std::byte> build_two_component_lag1_bytes(
    fuse::Rational upstream_interval = fuse::Rational{1}, int cells = 4) {
  return fuse::serialize(build_two_component_lag1_graph(upstream_interval, cells));
}

/// "upstream" subcycles 4x per common period (`upstream_interval = 1/4`)
/// and exports under `AccumOp::Mean` -- "downstream" must see the MEAN of
/// upstream's four substep values, not the last one, which is exactly what
/// a deferred-exchange scheme could never produce (everything the schedule
/// places WITHIN a period would be lost). Same two-component, one-channel
/// shape as build_two_component_lag1_graph() otherwise.
[[nodiscard]] inline fuse::CouplingGraph build_subcycled_mean_graph(
    fuse::Rational upstream_interval = fuse::Rational{1, 4}, int cells = 4) {
  return build_two_component_lag1_graph(upstream_interval, cells, fuse::AccumOp::Mean);
}

[[nodiscard]] inline std::vector<std::byte> build_subcycled_mean_bytes(
    fuse::Rational upstream_interval = fuse::Rational{1, 4}, int cells = 4) {
  return fuse::serialize(build_subcycled_mean_graph(upstream_interval, cells));
}

/// Two single-rank components, "alpha" and "beta", two-way coupled INSIDE an
/// iteration region -- the shape test/runtime/checkpoint_test.cpp's own
/// `iterated_two_way()` fixture builds, reproduced here so an ABI test can
/// drive it entirely through fuse_begin_step()/fuse_end_step() rather than
/// fuse_run(). The return channel is relaxed (a genuine loop-carried value) so
/// the sweep count is not trivially one.
[[nodiscard]] inline fuse::CouplingGraph build_implicit_coupling_graph(
    int cells = 4, int max_iterations = 30) {
  fuse::CouplingGraph g;
  const auto ma = g.add_mesh(
      fuse::MeshDecl{.name = "ma", .source = "generated:line:" + std::to_string(cells)});
  const auto mb = g.add_mesh(
      fuse::MeshDecl{.name = "mb", .source = "generated:line:" + std::to_string(cells)});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma, .dimension = 1});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb, .dimension = 1});
  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "alpha", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "beta", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});

  const auto a_out = g.add_field(
      a, fuse::FieldSpec{.name = "a_out", .region = ra, .components = 1,
                        .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(
      b, fuse::FieldSpec{.name = "b_in", .region = rb, .components = 1,
                        .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(
      b, fuse::FieldSpec{.name = "b_out", .region = rb, .components = 1,
                        .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(
      a, fuse::FieldSpec{.name = "a_in", .region = ra, .components = 1,
                        .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated scheme");
  g.channel(back).relax(0.7);

  fuse::ConvergenceCriterion convergence;
  convergence.add(fuse::Converged::Relative(1e-9, "a_out"));
  convergence.add(fuse::Converged::Relative(1e-9, "a_in"));
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "fsi", .channels = {fwd, back}, .max_iterations = max_iterations,
      .convergence = convergence});
  return g;
}

[[nodiscard]] inline std::vector<std::byte> build_implicit_coupling_bytes(
    int cells = 4, int max_iterations = 30) {
  return fuse::serialize(build_implicit_coupling_graph(cells, max_iterations));
}

/// build_implicit_coupling_graph()'s own shape above, UNCHANGED except for
/// "beta"'s own launch mode -- LaunchMode::Plugin naming `beta_plugin_path`
/// instead of the default (InProcess, test-injected) -- so "beta" is
/// auto-mounted at fuse_initialize() the same way ConformanceTest.
/// GraphDeclaredPluginIsAutoMountedAndRun's own "upstream" already is,
/// while "alpha" stays native and test-injected exactly as before. "beta"
/// keeps RankSet{0, 1} and interval 1 -- a Plugin component needs no rank
/// block of its own (fuse::transport::validate_launch_layout() treats it
/// like InProcess, docs/dev/abi-design.md's own "Provider direction"
/// section), so this field is unused by the mounting mechanism but kept
/// for ComponentDecl's own required shape.
[[nodiscard]] inline fuse::CouplingGraph build_implicit_coupling_graph_with_plugin_beta(
    const char* beta_plugin_path, int cells = 4, int max_iterations = 30) {
  fuse::CouplingGraph g;
  const auto ma = g.add_mesh(
      fuse::MeshDecl{.name = "ma", .source = "generated:line:" + std::to_string(cells)});
  const auto mb = g.add_mesh(
      fuse::MeshDecl{.name = "mb", .source = "generated:line:" + std::to_string(cells)});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma, .dimension = 1});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb, .dimension = 1});
  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "alpha", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "beta", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin,
                                 .plugin_path = beta_plugin_path}});

  const auto a_out = g.add_field(
      a, fuse::FieldSpec{.name = "a_out", .region = ra, .components = 1,
                        .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(
      b, fuse::FieldSpec{.name = "b_in", .region = rb, .components = 1,
                        .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(
      b, fuse::FieldSpec{.name = "b_out", .region = rb, .components = 1,
                        .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(
      a, fuse::FieldSpec{.name = "a_in", .region = ra, .components = 1,
                        .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated scheme");
  g.channel(back).relax(0.7);

  fuse::ConvergenceCriterion convergence;
  convergence.add(fuse::Converged::Relative(1e-9, "a_out"));
  convergence.add(fuse::Converged::Relative(1e-9, "a_in"));
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "fsi", .channels = {fwd, back}, .max_iterations = max_iterations,
      .convergence = convergence});
  return g;
}

/// The graph that puts an externally-driven component
/// (fuse_begin_step()/fuse_end_step()) inside a WINDOW-scoped iteration
/// region, not merely an ordinary one -- build_implicit_coupling_graph()'s own
/// shape above, with `interval == window_length` for both members (the same
/// "deliberately simplest" choice window_runner_test.cpp's own header
/// explains: no multirate subcycling within the window itself, so there is
/// exactly one advance_window() substep per member per window, isolating "does
/// the suspend/resume protocol work at all for a WindowAdvance instruction"
/// from the K-wide waveform/subcycling machinery test/runtime/
/// window_runner_multirate_test.cpp already covers in-process). Both
/// "alpha" and "beta" are window members (CouplingWindowScope.policies),
/// matching the shape of U and V both externally driven through the SAME
/// window, in miniature.
[[nodiscard]] inline fuse::CouplingGraph build_implicit_window_coupling_graph(
    int cells = 4, int max_iterations = 30,
    fuse::DtPolicy pol_a = fuse::DtPolicy::Adjustable,
    fuse::DtPolicy pol_b = fuse::DtPolicy::Adjustable,
    // `decl.channels`' own order, not declaration order, is what
    // `members_in_first_seen_order()`
    // (src/ir/window_lowering.cpp, mirroring window_region_members()) walks
    // to decide which member's WindowAdvance node is created FIRST, which
    // in turn is what makes it appear first in the compiled instruction
    // stream (no other tie-break applies -- a WindowAdvance node has no
    // formal DAG inputs at all, op.hpp's own comment on
    // OpAttrs::window_import explains why). `false` (the default)
    // reproduces every existing caller's own graph exactly: channels =
    // {fwd, back}, alpha (declared first) first in plan order too. `true`
    // swaps the list to {back, fwd} -- beta (declared SECOND) is listed
    // first via `back`'s own source, so beta's WindowAdvance is created
    // first despite alpha being ComponentId 0: a graph where "first
    // declared" and "first in plan order" genuinely disagree, needed to
    // prove fuse_is_coupling_ongoing() reads the PLAN, not declaration
    // order (AbiParticipant.IsCouplingOngoingReportsThePlanFirstComponent,
    // NotTheFirstDeclaredOne).
    bool reorder_channels_so_beta_is_first_in_plan_order = false) {
  fuse::CouplingGraph g;
  const auto ma = g.add_mesh(
      fuse::MeshDecl{.name = "ma", .source = "generated:line:" + std::to_string(cells)});
  const auto mb = g.add_mesh(
      fuse::MeshDecl{.name = "mb", .source = "generated:line:" + std::to_string(cells)});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma, .dimension = 1});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb, .dimension = 1});
  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "alpha", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "beta", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});

  const auto a_out = g.add_field(
      a, fuse::FieldSpec{.name = "a_out", .region = ra, .components = 1,
                        .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(
      b, fuse::FieldSpec{.name = "b_in", .region = rb, .components = 1,
                        .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(
      b, fuse::FieldSpec{.name = "b_out", .region = rb, .components = 1,
                        .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(
      a, fuse::FieldSpec{.name = "a_in", .region = ra, .components = 1,
                        .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the window's own sweep, not a real-time lag");
  g.channel(back).relax(0.7);

  fuse::CouplingWindowScope scope;
  scope.explicit_interval = fuse::Rational{1};
  scope.order = 2;
  scope.policies = {{"alpha", pol_a}, {"beta", pol_b}};
  scope.iteration = fuse::WaveformIteration::Jacobi;

  fuse::ConvergenceCriterion convergence;
  convergence.add(fuse::Converged::Relative(1e-9, "a_out"));
  convergence.add(fuse::Converged::Relative(1e-9, "a_in"));
  const std::vector<fuse::ChannelId> channel_order =
      reorder_channels_so_beta_is_first_in_plan_order
          ? std::vector<fuse::ChannelId>{back, fwd}
          : std::vector<fuse::ChannelId>{fwd, back};
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "window", .channels = channel_order, .max_iterations = max_iterations,
      .convergence = convergence, .window_scope = scope});
  return g;
}

[[nodiscard]] inline std::vector<std::byte> build_implicit_window_coupling_bytes(
    int cells = 4, int max_iterations = 30,
    fuse::DtPolicy pol_a = fuse::DtPolicy::Adjustable,
    fuse::DtPolicy pol_b = fuse::DtPolicy::Adjustable,
    bool reorder_channels_so_beta_is_first_in_plan_order = false) {
  return fuse::serialize(build_implicit_window_coupling_graph(
      cells, max_iterations, pol_a, pol_b,
      reorder_channels_so_beta_is_first_in_plan_order));
}

/// "mixed" is a LaunchMode::Plugin component (`mixed_plugin_path`,
/// test/abi/plugins/mixed_memspace_provider.c) with TWO Provides fields
/// on ONE instance -- "host_out" (FUSE_MEMSPACE_HOST) and "device_out"
/// (FUSE_MEMSPACE_DEVICE) -- each fed through its own one-way channel to
/// a native "sink" component's own two Requires fields, "host_in" and
/// "device_in". No iteration region, no lag: a one-way channel already
/// delivers a period's export to that SAME period's import once both
/// components have taken their step (this file's own header comment, the
/// SAME reasoning build_two_component_lag1_graph's own sibling fixtures
/// rely on), which is all this fixture needs to prove the mixed
/// host/device registration survives an ordinary channel transfer, not
/// only a direct FieldRegistry read.
[[nodiscard]] inline fuse::CouplingGraph build_mixed_memspace_graph(
    const char* mixed_plugin_path, int cells = 1) {
  fuse::CouplingGraph g;
  const auto m1 = g.add_mesh(
      fuse::MeshDecl{.name = "m1", .source = "generated:line:" + std::to_string(cells)});
  const auto m2 = g.add_mesh(
      fuse::MeshDecl{.name = "m2", .source = "generated:line:" + std::to_string(cells)});
  const auto r1 = g.add_region(fuse::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(fuse::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto mixed = g.add_component(fuse::ComponentDecl{
      .name = "mixed", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1},
      .launch = fuse::LaunchSpec{.mode = fuse::LaunchMode::Plugin,
                                 .plugin_path = mixed_plugin_path}});
  const auto sink = g.add_component(fuse::ComponentDecl{
      .name = "sink", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto host_out = g.add_field(
      mixed, fuse::FieldSpec{.name = "host_out", .region = r1, .components = 1,
                             .units = fuse::units::kKelvin,
                             .conservation = fuse::Conservation::None,
                             .intent = fuse::Intent::Provides});
  const auto device_out = g.add_field(
      mixed, fuse::FieldSpec{.name = "device_out", .region = r1, .components = 1,
                             .units = fuse::units::kKelvin,
                             .conservation = fuse::Conservation::None,
                             .intent = fuse::Intent::Provides});
  const auto host_in = g.add_field(
      sink, fuse::FieldSpec{.name = "host_in", .region = r2, .components = 1,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});
  const auto device_in = g.add_field(
      sink, fuse::FieldSpec{.name = "device_in", .region = r2, .components = 1,
                            .units = fuse::units::kKelvin,
                            .conservation = fuse::Conservation::None,
                            .intent = fuse::Intent::Requires});
  const auto host_link = g.add_channel("host_link", host_out, host_in);
  g.channel(host_link).remap(fuse::RemapFlags::Conservative);
  const auto device_link = g.add_channel("device_link", device_out, device_in);
  g.channel(device_link).remap(fuse::RemapFlags::Conservative);
  return g;
}

}  // namespace fuse::abi_test
