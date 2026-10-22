// KNOWN_ISSUES.md D15 ("Registry agreement" track, owner decision
// 2026-10-22). Tier-0, MPI-free tests of the pure graph-shape predicate
// src/abi/participant.cpp's impl_initialize_local() consults a priori,
// before any Plugin-mount work runs -- include/fuse/abi/plugin_agreement.
// hpp's own header comment has the full account of why this check cannot
// be collective and exactly which launch shape it refuses.

#include "fuse/abi/plugin_agreement.hpp"

#include <gtest/gtest.h>

#include "fuse/base/units.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/launch.hpp"

using fuse::abi_detail::graph_has_launch_mode;
using fuse::abi_detail::plugin_mount_agreement_needs_a_missing_spanning_communicator;

namespace {

/// Builds a graph with one component per entry of `modes` (named "c0",
/// "c1", ...), each with one field, so a test can declare exactly the
/// LaunchMode mix it needs without a channel or a second region --
/// nothing this predicate reads depends on either.
fuse::CouplingGraph build_graph_with_modes(const std::vector<fuse::LaunchMode>& modes) {
  fuse::CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m, .dimension = 1});
  for (std::size_t i = 0; i < modes.size(); ++i) {
    fuse::LaunchSpec spec;
    spec.mode = modes[i];
    if (modes[i] == fuse::LaunchMode::Plugin) {
      spec.plugin_path = "irrelevant-to-this-predicate.so";
    }
    if (modes[i] == fuse::LaunchMode::Connect) {
      spec.port_file = "irrelevant-to-this-predicate.port";
    }
    if (modes[i] == fuse::LaunchMode::Mpmd) {
      spec.appnum = static_cast<int>(i);
    }
    const auto c = g.add_component(fuse::ComponentDecl{
        .name = "c" + std::to_string(i), .ranks = fuse::RankSet{0, 1},
        .interval = fuse::Rational{1}, .launch = spec});
    (void)g.add_field(c, fuse::FieldSpec{.name = "f", .region = r, .components = 1,
                                         .units = fuse::units::kKelvin,
                                         .conservation = fuse::Conservation::None,
                                         .intent = fuse::Intent::Provides});
  }
  return g;
}

}  // namespace

TEST(GraphHasLaunchMode, FindsAModeDeclaredByAnyComponent) {
  const fuse::CouplingGraph g =
      build_graph_with_modes({fuse::LaunchMode::InProcess, fuse::LaunchMode::Plugin});
  EXPECT_TRUE(graph_has_launch_mode(g, fuse::LaunchMode::Plugin));
  EXPECT_TRUE(graph_has_launch_mode(g, fuse::LaunchMode::InProcess));
  EXPECT_FALSE(graph_has_launch_mode(g, fuse::LaunchMode::Connect));
  EXPECT_FALSE(graph_has_launch_mode(g, fuse::LaunchMode::Mpmd));
}

TEST(GraphHasLaunchMode, EmptyGraphHasNoLaunchModeAtAll) {
  const fuse::CouplingGraph g = build_graph_with_modes({});
  EXPECT_FALSE(graph_has_launch_mode(g, fuse::LaunchMode::Plugin));
  EXPECT_FALSE(graph_has_launch_mode(g, fuse::LaunchMode::Connect));
}

// The refused shape: Plugin + Connect in the same graph, and this
// process's own my_component does not itself call connect() (here,
// std::nullopt -- my_component was never named).
TEST(PluginMountAgreementNeedsAMissingSpanningCommunicator,
    RefusesPluginPlusConnectWhenMyComponentIsUnset) {
  const fuse::CouplingGraph g =
      build_graph_with_modes({fuse::LaunchMode::Plugin, fuse::LaunchMode::Connect});
  EXPECT_TRUE(plugin_mount_agreement_needs_a_missing_spanning_communicator(
      g, std::nullopt, /*communicator_set=*/true));
}

// Same graph, but my_component names a Plugin or InProcess component --
// entries.def's own comment says BOTH behave exactly as "unset" for
// connect() purposes, so this must refuse identically to the
// std::nullopt case above.
TEST(PluginMountAgreementNeedsAMissingSpanningCommunicator,
    RefusesPluginPlusConnectWhenMyComponentIsPluginOrInProcess) {
  const fuse::CouplingGraph g =
      build_graph_with_modes({fuse::LaunchMode::Plugin, fuse::LaunchMode::Connect});
  EXPECT_TRUE(plugin_mount_agreement_needs_a_missing_spanning_communicator(
      g, fuse::LaunchMode::Plugin, /*communicator_set=*/true));
  EXPECT_TRUE(plugin_mount_agreement_needs_a_missing_spanning_communicator(
      g, fuse::LaunchMode::InProcess, /*communicator_set=*/true));
}

// This process's OWN my_component IS the Connect component: connect()
// DOES run for this process, and its own result, if it succeeds, is
// assumed (this header's own "WHAT THIS DOES NOT COVER" paragraph) to be
// the spanning communicator -- no refusal.
TEST(PluginMountAgreementNeedsAMissingSpanningCommunicator,
    DoesNotRefuseWhenMyComponentIsTheConnectHalfItself) {
  const fuse::CouplingGraph g =
      build_graph_with_modes({fuse::LaunchMode::Plugin, fuse::LaunchMode::Connect});
  EXPECT_FALSE(plugin_mount_agreement_needs_a_missing_spanning_communicator(
      g, fuse::LaunchMode::Connect, /*communicator_set=*/true));
}

// Mpmd, not Connect: MPI_COMM_WORLD already spans every program before
// fuse_setup() runs (launch.hpp's own doc comment on LaunchMode::Mpmd),
// so a Plugin component alongside an Mpmd one is safe with no connect()
// call at all.
TEST(PluginMountAgreementNeedsAMissingSpanningCommunicator,
    DoesNotRefusePluginPlusMpmdEvenWhenMyComponentIsUnset) {
  const fuse::CouplingGraph g =
      build_graph_with_modes({fuse::LaunchMode::Plugin, fuse::LaunchMode::Mpmd});
  EXPECT_FALSE(plugin_mount_agreement_needs_a_missing_spanning_communicator(
      g, std::nullopt, /*communicator_set=*/true));
}

// Plugin alone (today's working, already-fixed shape, "Registry hosting"
// track) -- never refused, regardless of my_component.
TEST(PluginMountAgreementNeedsAMissingSpanningCommunicator,
    DoesNotRefusePluginAlone) {
  const fuse::CouplingGraph g = build_graph_with_modes({fuse::LaunchMode::Plugin});
  EXPECT_FALSE(plugin_mount_agreement_needs_a_missing_spanning_communicator(
      g, std::nullopt, /*communicator_set=*/true));
}

// No Plugin component at all -- nothing for a plugin-mount agreement to
// ever be about, even with a Connect component present.
TEST(PluginMountAgreementNeedsAMissingSpanningCommunicator,
    DoesNotRefuseConnectAloneWithNoPluginComponent) {
  const fuse::CouplingGraph g = build_graph_with_modes({fuse::LaunchMode::Connect});
  EXPECT_FALSE(plugin_mount_agreement_needs_a_missing_spanning_communicator(
      g, std::nullopt, /*communicator_set=*/true));
}

// communicator_set == false: single process by construction (context.hpp's
// own comment on `backend`) -- there is no OTHER executable to fail to
// span, so even the otherwise-refused Plugin+Connect shape must not
// refuse here. Exercises this function's own top-level gate directly,
// the one impl_initialize_local() relies on to avoid rejecting a
// single-process exercise of a graph that merely DECLARES a Connect
// component without ever running it that way.
TEST(PluginMountAgreementNeedsAMissingSpanningCommunicator,
    DoesNotRefuseWhenNoCommunicatorIsConfiguredAtAll) {
  const fuse::CouplingGraph g =
      build_graph_with_modes({fuse::LaunchMode::Plugin, fuse::LaunchMode::Connect});
  EXPECT_FALSE(plugin_mount_agreement_needs_a_missing_spanning_communicator(
      g, std::nullopt, /*communicator_set=*/false));
}
