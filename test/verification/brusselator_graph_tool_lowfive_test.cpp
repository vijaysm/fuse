// brusselator_graph_tool.cpp's own
// `--transport lowfive` wiring is exercised end to end by the committed
// reference YAML + verify-graph-tool-reference.cmake (examples/
// CMakeLists.txt: verification.brusselator.graph_tool.reproduces_
// reference_lowfive), which proves the TOOL's text output is stable --
// it does not prove that text is actually a VALID graph. This file
// reproduces the same wiring (add_window_coupling(), then hdf5_path on
// u_out/v_out plus a LowFive-launched component) directly against
// fuse::validate(), so a change that keeps the committed YAML's bytes
// stable but breaks what it describes still fails a test.

#include <gtest/gtest.h>

#include "fuse/ir/lowering.hpp"
#include "fuse/ir/op.hpp"
#include "fuse/model/validate.hpp"
#include "fuse/model/yaml_io.hpp"

#include "brusselator_graph.hpp"

using fuse::AccelKind;
using fuse::CouplingGraph;
using fuse::Diagnostics;
using fuse::FieldSpec;
using fuse::LaunchMode;
using fuse::LaunchSpec;
using fuse::MeshDecl;
using fuse::OpKind;
using fuse::Rational;
using fuse::RankSet;
using fuse::Region;
using fuse::WaveformIteration;

namespace {

// Mirrors brusselator_graph_tool.cpp's own --ranks-u 1 --ranks-v 1
// --ranks-coupler 1 --launch mpmd --transport lowfive invocation --
// the exact arguments examples/brusselator/participants/
// reference_mpmd_3_lowfive.yaml was generated from.
CouplingGraph build_lowfive_graph() {
  CouplingGraph g;
  const auto u_mesh = g.add_mesh({.name = "u", .source = "generated:quad:3x3"});
  const auto v_mesh = g.add_mesh({.name = "v", .source = "generated:quad:4x4"});
  const auto u_region = g.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  (void)g.add_region({.name = "v_all", .mesh = v_mesh, .dimension = 2});

  const RankSet u_ranks{0, 1}, v_ranks{1, 1}, coupler_ranks{2, 1};
  const LaunchSpec u_launch{.mode = LaunchMode::Mpmd, .appnum = 0};
  const LaunchSpec v_launch{.mode = LaunchMode::Mpmd, .appnum = 1};
  const LaunchSpec coupler_launch{.mode = LaunchMode::Mpmd, .appnum = 2};

  const auto handles = brusselator::add_window_coupling(
      g, u_region, Rational{1, 500}, Rational{1, 500}, Rational{1, 500}, 1e-10,
      60, AccelKind::None, WaveformIteration::Jacobi, coupler_ranks, u_ranks,
      v_ranks, u_launch, v_launch, coupler_launch);

  const auto u_out = g.find_field(handles.u_component, "u_out");
  const auto v_out = g.find_field(handles.v_component, "v_out");
  EXPECT_TRUE(u_out.has_value());
  EXPECT_TRUE(v_out.has_value());
  const_cast<FieldSpec&>(g.field(*u_out)).hdf5_path = "brusselator_u.h5/grid/u_out";
  const_cast<FieldSpec&>(g.field(*v_out)).hdf5_path = "brusselator_v.h5/grid/v_out";
  // `.kind = ComponentKind::Coupler`: mirrors brusselator_graph_tool.cpp's
  // own fix (that file's own comment on this identical declaration has
  // the full account -- a fieldless marker with no explicit `.kind`
  // defaults to ComponentKind::Physics, becoming a THIRD externally
  // driven component fuse_initialize() never exempts). Neither test in
  // this file drives the generated graph through a live Runner/ABI, so
  // this mirror's own omission would have gone unnoticed exactly as the
  // real tool's own did -- fixed here anyway so this file's own claim to
  // mirror the real tool's invocation stays true.
  (void)g.add_component({.name = "lowfive", .ranks = RankSet{0, 3},
                         .interval = Rational{1},
                         .kind = fuse::ComponentKind::Coupler,
                         .launch = LaunchSpec{.mode = LaunchMode::LowFive}});
  return g;
}

}  // namespace

TEST(BrusselatorGraphToolLowFive, TheGeneratedGraphValidatesClean) {
  const CouplingGraph g = build_lowfive_graph();
  const Diagnostics d = validate(g);
  EXPECT_FALSE(d.has_errors()) << d.to_string();
}

TEST(BrusselatorGraphToolLowFive, OmittingTheLowFiveComponentIsRefused) {
  // The negative control: without the "lowfive"-launched component, the
  // SAME two fields' hdf5_path must be refused -- proving the positive
  // result above is not simply validate() never looking at hdf5_path.
  CouplingGraph g;
  const auto u_mesh = g.add_mesh({.name = "u", .source = "generated:quad:3x3"});
  (void)g.add_mesh({.name = "v", .source = "generated:quad:4x4"});
  const auto u_region = g.add_region({.name = "u_all", .mesh = u_mesh, .dimension = 2});
  const auto handles = brusselator::add_window_coupling(
      g, u_region, Rational{1, 500}, Rational{1, 500}, Rational{1, 500}, 1e-10,
      60, AccelKind::None);
  const auto u_out = g.find_field(handles.u_component, "u_out");
  ASSERT_TRUE(u_out.has_value());
  const_cast<FieldSpec&>(g.field(*u_out)).hdf5_path = "brusselator_u.h5/grid/u_out";
  const Diagnostics d = validate(g);
  EXPECT_TRUE([&] {
    for (const auto& item : d.items())
      if (item.code == "E-LOWFIVE-NO-COMPONENT") return true;
    return false;
  }()) << d.to_string();
}

// IR/runtime half: the real tool-built
// graph -- window-scoped (add_window_coupling()), u_mesh/v_mesh genuinely
// different (3x3 vs 4x4, so Remap is actually needed), u_ranks/v_ranks
// genuinely different ({0,1} vs {1,1}) -- lowers with no OpKind::Transport
// and no OpKind::Epoch anywhere in the u2v/v2u channels' own chain, and an
// OpKind::ExternalDelivery node stands in their place. This is the same
// mechanism LoweringChannel.LowFiveBoundChannelEmitsExternalDeliveryNot
// Transport (test/ir/lowering_channel_test.cpp) proves on a synthetic
// single-clock pair; this test is what proves the window-scoped and
// genuinely-cross-rank case (the Brusselator's own LowFive graph) behaves
// identically, not merely a graph built to make the mechanism look clean.
TEST(BrusselatorGraphToolLowFive, LowersWithExternalDeliveryAndNoTransport) {
  const CouplingGraph g = build_lowfive_graph();
  const auto r = fuse::lower(g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  int external_delivery_count = 0, transport_count = 0, epoch_count = 0;
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(fuse::NodeId{i});
    if (n.kind == OpKind::ExternalDelivery) {
      ++external_delivery_count;
      // ONE input, an ORDERING edge onto the source field's own
      // producer -- here, a window-scoped export, so WindowAdvance
      // (OpKind::ExternalDelivery's own comment, op.hpp; the scheduling-
      // order fix this comment documents, MEASURED a later session,
      // needs this edge to place the node after the exporting member's
      // own last substep this sweep) -- never a data dependency
      // (execute()'s own ExternalDelivery case, src/runtime/runner.cpp,
      // does not read it).
      ASSERT_EQ(n.inputs.size(), 1u);
      EXPECT_EQ(r.dag.node(r.dag.arc(n.inputs.front()).producer).kind,
                OpKind::WindowAdvance);
    } else if (n.kind == OpKind::Transport) {
      ++transport_count;
    } else if (n.kind == OpKind::Epoch) {
      ++epoch_count;
    }
  }
  // Two LowFive-bound channels (u2v carries u_out, v2u carries v_out) ->
  // two ExternalDelivery nodes, one per channel.
  EXPECT_EQ(external_delivery_count, 2);
  EXPECT_EQ(transport_count, 0);
  EXPECT_EQ(epoch_count, 0);
}
