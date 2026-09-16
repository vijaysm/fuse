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
       .interval = Rational{1, 4}});
  const auto solid = g.add_component(
      {.name = "solid", .ranks = {7, 2}, .interval = Rational{1, 3}});

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
      .relax(0.7)
      .extrapolate(2)
      .lag(1);
  return g;
}
}  // namespace fuse::testing
