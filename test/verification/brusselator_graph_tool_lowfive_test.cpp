// S6 phase 3 ("LowFive" track), Unit 3. brusselator_graph_tool.cpp's own
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
  (void)g.add_component({.name = "lowfive", .ranks = RankSet{0, 3},
                         .interval = Rational{1},
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
