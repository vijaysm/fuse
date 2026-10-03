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

/// KNOWN_ISSUES.md D12, test (2): "upstream" subcycles 4x per common
/// period (`upstream_interval = 1/4`) and exports under `AccumOp::Mean` --
/// "downstream" must see the MEAN of upstream's four substep values, not
/// the last one, which is exactly what a deferred-exchange scheme (the
/// fix this file's own test exercises) could never produce (KNOWN_ISSUES.md
/// D12's own finding: "everything the schedule places WITHIN a period is
/// lost"). Same two-component, one-channel shape as
/// build_two_component_lag1_graph() otherwise.
[[nodiscard]] inline fuse::CouplingGraph build_subcycled_mean_graph(
    fuse::Rational upstream_interval = fuse::Rational{1, 4}, int cells = 4) {
  return build_two_component_lag1_graph(upstream_interval, cells, fuse::AccumOp::Mean);
}

[[nodiscard]] inline std::vector<std::byte> build_subcycled_mean_bytes(
    fuse::Rational upstream_interval = fuse::Rational{1, 4}, int cells = 4) {
  return fuse::serialize(build_subcycled_mean_graph(upstream_interval, cells));
}

/// KNOWN_ISSUES.md D12, test (3): two single-rank components, "alpha" and
/// "beta", two-way coupled INSIDE an iteration region -- the shape
/// test/runtime/checkpoint_test.cpp's own `iterated_two_way()` fixture
/// builds, reproduced here so an ABI test can drive it entirely through
/// fuse_begin_step()/fuse_end_step() rather than fuse_run(). The return
/// channel is relaxed (a genuine loop-carried value) so the sweep count is
/// not trivially one.
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

/// Time windows track (Conformance, Unit 2): the FIRST graph to put an
/// externally-driven component (fuse_begin_step()/fuse_end_step()) inside
/// a WINDOW-scoped iteration region, not merely an ordinary one --
/// build_implicit_coupling_graph()'s own shape above, with `interval ==
/// window_length` for both members (the same "deliberately simplest"
/// choice window_runner_test.cpp's own header explains: no multirate
/// subcycling within the window itself, so there is exactly one
/// advance_window() substep per member per window, isolating "does the
/// suspend/resume protocol work at all for a WindowAdvance instruction"
/// from the K-wide waveform/subcycling machinery test/runtime/
/// window_runner_multirate_test.cpp already covers in-process). Both
/// "alpha" and "beta" are window members (CouplingWindowScope.policies),
/// matching the eventual Unit 2 shape (U and V both externally driven,
/// through the SAME window) in miniature.
[[nodiscard]] inline fuse::CouplingGraph build_implicit_window_coupling_graph(
    int cells = 4, int max_iterations = 30,
    fuse::DtPolicy pol_a = fuse::DtPolicy::Adjustable,
    fuse::DtPolicy pol_b = fuse::DtPolicy::Adjustable) {
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
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "window", .channels = {fwd, back}, .max_iterations = max_iterations,
      .convergence = convergence, .window_scope = scope});
  return g;
}

[[nodiscard]] inline std::vector<std::byte> build_implicit_window_coupling_bytes(
    int cells = 4, int max_iterations = 30,
    fuse::DtPolicy pol_a = fuse::DtPolicy::Adjustable,
    fuse::DtPolicy pol_b = fuse::DtPolicy::Adjustable) {
  return fuse::serialize(
      build_implicit_window_coupling_graph(cells, max_iterations, pol_a, pol_b));
}

}  // namespace fuse::abi_test
