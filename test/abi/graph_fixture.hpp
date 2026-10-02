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
#include <string>
#include <vector>

#include "fuse/base/units.hpp"
#include "fuse/model/channel.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"

namespace fuse::abi_test {

[[nodiscard]] inline fuse::CouplingGraph build_two_component_lag1_graph(
    fuse::Rational upstream_interval = fuse::Rational{1},
    int cells = 4) {
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
  return g;
}

[[nodiscard]] inline std::vector<std::byte> build_two_component_lag1_bytes(
    fuse::Rational upstream_interval = fuse::Rational{1}, int cells = 4) {
  return fuse::serialize(build_two_component_lag1_graph(upstream_interval, cells));
}

}  // namespace fuse::abi_test
