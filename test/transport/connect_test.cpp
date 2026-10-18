// Tier-0 unit tests for include/fuse/transport/connect.hpp: everything that
// does not need a second process (KNOWN_ISSUES.md C1 -- this sandbox cannot
// initialise MPI at all, even at one rank). `connect()` itself is exercised
// only by test/parallel/connect's host-run executables.

#include "fuse/transport/connect.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include <filesystem>
#include <unistd.h>

#include "fuse/model/coupling_graph.hpp"

using fuse::ComponentId;
using fuse::CouplingGraph;
using fuse::LaunchMode;
using fuse::LaunchSpec;
using fuse::RankSet;
using fuse::transport::check_mpmd_layout;
using fuse::transport::validate_launch_layout;
using fuse::transport::wait_for_port_file;

namespace {

CouplingGraph two_component_graph(LaunchSpec a_launch, LaunchSpec b_launch,
                                  RankSet a_ranks, RankSet b_ranks) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = a_ranks, .launch = std::move(a_launch)});
  g.add_component({.name = "b", .ranks = b_ranks, .launch = std::move(b_launch)});
  return g;
}

// Scratch files for wait_for_port_file() go under the system temp
// directory, not test/data (FUSE_TEST_DATA_DIR): that directory holds
// checked-in fixtures a test reads, never a file a test run writes, and
// this process-unique subdirectory keeps a parallel ctest run from racing
// another run's identically-named scratch file.
std::string scratch_path(const std::string& name) {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("fuse-connect-test-" + std::to_string(static_cast<long>(::getpid())));
  std::filesystem::create_directories(dir);
  return (dir / name).string();
}

}  // namespace

// ---- validate_launch_layout ------------------------------------------------

TEST(ValidateLaunchLayout, RefusesAGraphWithNoLaunchedComponent) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = {0, 4}});
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
}

TEST(ValidateLaunchLayout, AcceptsConsecutiveBlocksInComponentOrder) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0},
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1}, RankSet{0, 2},
      RankSet{2, 3});
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

TEST(ValidateLaunchLayout, RefusesANonContiguousComponentRankSet) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0},
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1},
      RankSet{std::vector<int>{0, 2}},  // strided, not contiguous
      RankSet{1, 1});
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
}

// The mutation `merge order swapped` (build-evidence/mutations/
// transport-connect.toml) targets exactly this case: if connect_pairwise()
// stopped fixing MPI_Intercomm_merge's high/low by component ORDER and used
// acceptor/connector role instead, two graphs that declare the SAME two
// components in the opposite order from how they are launched would merge
// to a communicator numbered backwards relative to what the graph
// declares -- and this is the test that would have to catch it, because it
// is the one place the declared layout is checked before any MPI call.
TEST(ValidateLaunchLayout, RefusesRanksNotStartingWhereThePreviousComponentEnded) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Connect, .is_acceptor = true, .port_file = "/tmp/x"},
      LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"},
      RankSet{0, 2},
      RankSet{5, 3});  // should start at 2, not 5: a gap
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
}

TEST(ValidateLaunchLayout, RefusesOverlappingRankSets) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0},
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1}, RankSet{0, 3},
      RankSet{2, 3});  // overlaps [0,3) at rank 2
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
}

TEST(ValidateLaunchLayout, InProcessComponentsAreIgnored) {
  CouplingGraph g;
  // A launched component must still start the block at rank 0 (nothing
  // precedes it among LAUNCHED components), regardless of where an
  // InProcess sibling declares its own ranks to be: an InProcess
  // component's `ranks` describes a placement in whatever communicator its
  // own code already runs on, which connect() never touches, so it imposes
  // no requirement on the launched components' numbering -- including, as
  // here, one that happens to overlap rank 0.
  g.add_component({.name = "mpmd-a",
                   .ranks = {0, 2},
                   .launch = LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0}});
  g.add_component({.name = "serial", .ranks = {0, 1}});  // InProcess default
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// S6 phase 4, "Adapter host" track: a Plugin component shares the host
// process exactly like an InProcess one (launch.hpp's own doc on
// LaunchMode::Plugin), so it imposes no requirement on the launched
// components' numbering either.
// KNOWN_ISSUES.md D15, round 2 (item 3, "a plugin on a rank subset"): the
// existing PluginComponentsAreIgnored case below only ever gives Plugin a
// RankSet spanning the WHOLE run; this one gives it a STRICT SUBSET of a
// larger world and confirms validate_launch_layout() needs no special
// case for that -- the existing rule ("Plugin needs no rank block of its
// own") already covers a subset, since it imposes no RankSet constraint
// on a Plugin component at all, whatever that RankSet happens to be.
TEST(ValidateLaunchLayout, AcceptsAPluginComponentOnAStrictRankSubset) {
  CouplingGraph g;
  g.add_component({.name = "mpmd_host", .ranks = {0, 3},
                   .launch = {.mode = LaunchMode::Mpmd, .appnum = 0}});
  // Plugin, placed on ranks {1} of this 3-rank world -- a subset, not
  // the whole run and not merely "ranks after the launched block".
  g.add_component({.name = "plugin_on_subset", .ranks = {1, 1},
                   .launch = {.mode = LaunchMode::Plugin}});

  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << r.error().message;
}

TEST(ValidateLaunchLayout, PluginComponentsAreIgnored) {
  CouplingGraph g;
  g.add_component({.name = "mpmd-a",
                   .ranks = {0, 2},
                   .launch = LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0}});
  g.add_component({.name = "ideal-gas",
                   .ranks = {0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Plugin,
                                        .plugin_path = "/opt/fuse-plugins/ideal-gas.so"}});
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// ---- validate_launch_layout: D36's gap rule --------------------------
//
// Accept mode's fix (KNOWN_ISSUES.md D36): a component's declared RankSet
// must be exactly the ranks that host it, so U's own block no longer
// spans the coupler's bundled ranks too -- it stops one rank short of
// V's own block, with the coupler's own in-process RankSet filling the
// gap between them. These pin validate_launch_layout()'s own half of
// that fix: the new shape validates, each of the two new refusals fires
// on exactly the row that should trip it, and the OLD shape (U's own
// declared range already including the coupler's) still validates
// unchanged -- nothing that passed before this fix stops passing.

// The pre-D36 shape: U declares the FULL local executable world (its own
// physics rank plus the coupler's bundled one), the coupler a trailing,
// overlapping subset of it, no gap between U's and V's own blocks at
// all. Never reaches check_gap_filled_by_in_process_components() --
// this is the row that row's own doc comment calls out as continuing to
// validate unchanged.
TEST(ValidateLaunchLayout, ThePreD36OverlappingDeclarationStillValidates) {
  CouplingGraph g;
  g.add_component({.name = "U",
                   .ranks = RankSet{0, 2},  // U's own physics rank 0 + coupler's bundled rank 1
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true,
                                        .port_file = "/tmp/x"}});
  g.add_component({.name = "coupler", .ranks = RankSet{1, 1}});  // InProcess default
  g.add_component({.name = "V",
                   .ranks = RankSet{2, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// The D36 shape itself: U declares only its own physics rank, the
// coupler's own in-process RankSet exactly fills the gap between U's
// block and V's, V starts right after the gap.
TEST(ValidateLaunchLayout, AGapExactlyCoveredByAnInProcessComponentValidates) {
  CouplingGraph g;
  g.add_component({.name = "U",
                   .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true,
                                        .port_file = "/tmp/x"}});
  g.add_component({.name = "coupler", .ranks = RankSet{1, 1}});  // fills the gap [1, 2)
  g.add_component({.name = "V",
                   .ranks = RankSet{2, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// A Plugin-mode component fills a gap exactly the same way an InProcess
// one does (both share the host process's own communicator, launch.hpp's
// own doc on LaunchMode::Plugin) -- the generalisation
// PluginComponentsAreIgnored already pins for the no-gap case.
TEST(ValidateLaunchLayout, AGapExactlyCoveredByAPluginComponentValidates) {
  CouplingGraph g;
  g.add_component({.name = "U",
                   .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true,
                                        .port_file = "/tmp/x"}});
  g.add_component({.name = "ideal-gas",
                   .ranks = RankSet{1, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Plugin,
                                        .plugin_path = "/opt/fuse-plugins/ideal-gas.so"}});
  g.add_component({.name = "V",
                   .ranks = RankSet{2, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// A wider gap, split exactly between two in-process components -- the
// rule is "covered", not "covered by exactly one component".
TEST(ValidateLaunchLayout, AGapCoveredByTwoInProcessComponentsTogetherValidates) {
  CouplingGraph g;
  g.add_component({.name = "U",
                   .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true,
                                        .port_file = "/tmp/x"}});
  g.add_component({.name = "coupler-a", .ranks = RankSet{1, 1}});  // gap rank 1
  g.add_component({.name = "coupler-b", .ranks = RankSet{2, 1}});  // gap rank 2
  g.add_component({.name = "V",
                   .ranks = RankSet{3, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// Refusal 1: a gap rank no component -- launched or in-process -- hosts
// at all. Mirrors RefusesRanksNotStartingWhereThePreviousComponentEnded's
// own two-component gap, but names the new UNCOVERED code path rather
// than the old unconditional one (this graph has no in-process component
// whatsoever, so check_gap_filled_by_in_process_components() refuses with
// an empty `fillers` list).
TEST(ValidateLaunchLayout, RefusesAGapNoComponentHosts) {
  CouplingGraph g;
  g.add_component({.name = "U",
                   .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true,
                                        .port_file = "/tmp/x"}});
  g.add_component({.name = "V",
                   .ranks = RankSet{2, 1},  // rank 1 is uncovered: no coupler declares it
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
  EXPECT_NE(r.error().message.find("rank(s) 1"), std::string::npos) << r.error().message;
}

// Refusal 1b: the gap is PARTLY covered -- one of two gap ranks has no
// host. Distinguishes "covered exactly" from "covered by at least one
// component somewhere in the gap".
TEST(ValidateLaunchLayout, RefusesAPartlyCoveredGap) {
  CouplingGraph g;
  g.add_component({.name = "U",
                   .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true,
                                        .port_file = "/tmp/x"}});
  g.add_component({.name = "coupler", .ranks = RankSet{1, 1}});  // covers rank 1 only
  g.add_component({.name = "V",
                   .ranks = RankSet{3, 1},  // gap is [1, 3): rank 2 is uncovered
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
  EXPECT_NE(r.error().message.find("rank(s) 2"), std::string::npos) << r.error().message;
}

// Refusal 2: an in-process component straddles the boundary between a
// launched block and the gap after it -- part of its own RankSet sits on
// U's own declared rank, part in the gap. check_gap_filled_by_in_process_
// components() cannot then say which side of the boundary the
// component's ranks actually belong to, so this is refused rather than
// treated as "partial coverage, the rest is someone else's problem".
TEST(ValidateLaunchLayout, RefusesAnInProcessComponentStraddlingIntoTheGap) {
  CouplingGraph g;
  g.add_component({.name = "U",
                   .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true,
                                        .port_file = "/tmp/x"}});
  // Straddles: rank 0 is inside U's own block, rank 1 is in the gap [1, 2).
  g.add_component({.name = "straddler", .ranks = RankSet{std::vector<int>{0, 1}}});
  g.add_component({.name = "V",
                   .ranks = RankSet{2, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
  EXPECT_NE(r.error().message.find("straddle"), std::string::npos) << r.error().message;
}

// Refusal 2b: the mirror straddle, into V's own block on the far side of
// the gap rather than back into U's.
TEST(ValidateLaunchLayout, RefusesAnInProcessComponentStraddlingOutOfTheGap) {
  CouplingGraph g;
  g.add_component({.name = "U",
                   .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true,
                                        .port_file = "/tmp/x"}});
  // Straddles: rank 1 is in the gap [1, 2), rank 2 is inside V's own block.
  g.add_component({.name = "straddler", .ranks = RankSet{std::vector<int>{1, 2}}});
  g.add_component({.name = "V",
                   .ranks = RankSet{2, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
  EXPECT_NE(r.error().message.find("straddle"), std::string::npos) << r.error().message;
}

// ---- validate_launch_layout: the LowFive marker (D36's second defect) -----
//
// Measured on the owner's host, 2026-10-12 (a LowFive-enabled tree):
// verification.brusselator.participants.lowfive.mpmd.3 refused E-CONNECT-SPEC
// at fuse_setup() -- "component 'lowfive' starts at rank 0, but the launched
// components before it in the graph's own order occupy ranks [0, 3)" -- on a
// graph that validate() itself accepts cleanly
// (BrusselatorGraphToolLowFive.TheGeneratedGraphValidatesClean). The skip
// list validate_launch_layout() uses for components needing no rank block of
// their own (InProcess, Plugin) omitted LaunchMode::LowFive, even though
// connect()'s own `case LaunchMode::LowFive` (src/transport/connect.cpp) and
// launch.hpp's own doc on that value both say a LowFive-launched component's
// ranks "come from whatever OTHER launch story this run uses" -- the
// identical behaviour InProcess/Plugin already get here. These pin the fix
// against the REAL shape that graph came from (mirrored here, not built
// through brusselator_graph_tool.cpp: that tool links fuse::model only, and
// test/verification/brusselator_graph_tool_lowfive_test.cpp's own
// build_lowfive_graph() already is this mirror for the mpmd case -- these
// cases reuse its exact component/rank shape rather than a third copy, and
// add the accept-mode one that tool's own --transport lowfive --launch
// accept combination would produce once this track's own D36 fix applies).

TEST(ValidateLaunchLayout, ALowFiveMarkerAlongsideConsecutiveMpmdBlocksValidates) {
  CouplingGraph g;
  g.add_component({.name = "U", .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0}});
  g.add_component({.name = "V", .ranks = RankSet{1, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1}});
  g.add_component({.name = "coupler", .ranks = RankSet{2, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 2}});
  // The fieldless lowfive marker: ranks span the WHOLE run, unrelated to the
  // three launched blocks' own consecutive partition of it.
  g.add_component({.name = "lowfive", .ranks = RankSet{0, 3},
                   .launch = LaunchSpec{.mode = LaunchMode::LowFive}});
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

TEST(ValidateLaunchLayout, ALowFiveMarkerAlongsideTheAcceptLayoutsGapValidates) {
  // The D36 accept-mode shape (U={0,ranks_u}, coupler fills the gap, V
  // after it) plus the SAME whole-run lowfive marker. The marker's own
  // ranks touch the gap AND both launched blocks on either side of it --
  // confirming it is not treated as a gap-filling candidate at all (which
  // would refuse this as a STRADDLE), the way an actual in-process filler
  // confined to the gap itself is.
  CouplingGraph g;
  g.add_component({.name = "U", .ranks = RankSet{0, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect,
                                        .is_acceptor = true, .port_file = "/tmp/x"}});
  g.add_component({.name = "coupler", .ranks = RankSet{1, 1}});  // fills the gap [1, 2)
  g.add_component({.name = "V", .ranks = RankSet{2, 1},
                   .launch = LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}});
  g.add_component({.name = "lowfive", .ranks = RankSet{0, 3},
                   .launch = LaunchSpec{.mode = LaunchMode::LowFive}});
  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

TEST(ValidateLaunchLayout, ALowFiveMarkerWithNoLaunchedComponentStillRefusesNoLaunch) {
  // A lowfive component alone, with no Mpmd/Connect component at all, must
  // still trip the EXISTING "nothing for connect() to do" refusal -- the
  // fix does not make a lowfive-only graph look like it has a launched
  // component when it has none.
  CouplingGraph g;
  g.add_component({.name = "lowfive", .ranks = RankSet{0, 3},
                   .launch = LaunchSpec{.mode = LaunchMode::LowFive}});
  const auto r = validate_launch_layout(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
  EXPECT_NE(r.error().message.find("no component in the graph declares an mpmd or connect launch"),
           std::string::npos)
      << r.error().message;
}

// ---- check_mpmd_layout ------------------------------------------------

TEST(CheckMpmdLayout, AcceptsAMatchingLayout) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0},
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1}, RankSet{0, 2},
      RankSet{2, 3});
  const std::vector<int> appnum_by_rank = {0, 0, 1, 1, 1};
  const auto r = check_mpmd_layout(g, appnum_by_rank);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}

// The mutation `appnum check removed` targets exactly this: an appnum no
// component declares must be refused, not silently ignored (which would
// mean a rank that got misclassified by the launcher's own argument order
// runs as a stray, uncoupled participant instead of failing the job).
//
// The stray appnum (99) is reported by a rank OUTSIDE both declared
// RankSets, on its own fifth rank -- not by stealing a rank from "a" or
// "b"'s own count, which would ALSO fire
// RefusesWhenTheDeclaredRankSetDisagreesWithTheActualOne and mask whether
// THIS check (the undeclared-appnum lookup) ran at all. Found by the
// mutation itself: the first version of this test shifted a rank out of
// "b"'s reporting set, and `check_mpmd_layout(g, appnum_by_rank)` still
// refused under the mutated (checked-in-removed) code, via the unrelated
// mismatch check, with the SAME error code -- a false pass.
TEST(CheckMpmdLayout, RefusesAnAppnumNoComponentDeclares) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0},
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1}, RankSet{0, 2},
      RankSet{2, 2});
  const std::vector<int> appnum_by_rank = {0, 0, 1, 1, 99};  // appnum 99: undeclared, its own rank
  const auto r = check_mpmd_layout(g, appnum_by_rank);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-APPNUM");
}

TEST(CheckMpmdLayout, RefusesWhenTheDeclaredRankSetDisagreesWithTheActualOne) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0},
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1}, RankSet{0, 2},
      RankSet{2, 3});
  // Ranks reporting appnum 0 are {0, 2}, not the declared {0, 1}.
  const std::vector<int> appnum_by_rank = {0, 1, 0, 1, 1};
  const auto r = check_mpmd_layout(g, appnum_by_rank);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-APPNUM");
}

TEST(CheckMpmdLayout, RefusesADeclaredAppnumNoRankReports) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0},
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 1}, RankSet{0, 5},
      RankSet{5, 0});
  const std::vector<int> appnum_by_rank = {0, 0, 0, 0, 0};  // nobody reports appnum 1
  const auto r = check_mpmd_layout(g, appnum_by_rank);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-APPNUM");
}

TEST(CheckMpmdLayout, RefusesTwoComponentsClaimingTheSameAppnum) {
  const auto g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0},
      LaunchSpec{.mode = LaunchMode::Mpmd, .appnum = 0}, RankSet{0, 2},
      RankSet{2, 3});
  const std::vector<int> appnum_by_rank = {0, 0, 0, 0, 0};
  const auto r = check_mpmd_layout(g, appnum_by_rank);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
}

// ---- wait_for_port_file ------------------------------------------------

TEST(WaitForPortFile, ReadsAnAlreadyPublishedPort) {
  const std::string path = scratch_path("connect-port-already-there.txt");
  {
    std::ofstream out(path, std::ios::trunc);
    out << "tag#0:1234:abc\n";
  }
  const auto r = wait_for_port_file(path, /*timeout_s=*/1.0, /*poll_interval_s=*/0.01);
  ASSERT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
  EXPECT_EQ(r.value(), "tag#0:1234:abc");
}

// The mutation `timeout check removed` targets exactly this: a partner that
// never launches must become a named refusal in bounded time, never a hang
// -- see connect.hpp's own module comment on the residual blocking window
// this does NOT cover (MPI_Comm_connect itself), which is distinct from
// the one this test covers (the file never appearing at all).
TEST(WaitForPortFile, RefusesWithoutSleepingWhenTimeoutIsZero) {
  const std::string path = scratch_path("connect-port-never-appears.txt");
  const auto r = wait_for_port_file(path, /*timeout_s=*/0.0, /*poll_interval_s=*/5.0);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-TIMEOUT");
}

TEST(WaitForPortFile, RefusesWhenTheFileExistsButIsEmpty) {
  const std::string path = scratch_path("connect-port-empty.txt");
  { std::ofstream out(path, std::ios::trunc); }
  const auto r = wait_for_port_file(path, /*timeout_s=*/0.0, /*poll_interval_s=*/0.01);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-TIMEOUT");
}

// ---- the two-connect-mode-component rule (D18, review 2026-10-11 TR-17) --
//
// D18's own limit on connect_pairwise(): exactly two `Connect`-mode
// components. Before this, the rule lived inside connect_pairwise() itself,
// behind `#if FUSE_HAVE_MPI` and `MPI_Initialized()` -- a real, reachable
// refusal with no test anywhere in the tree, because its only caller needed
// a live MPI world to reach at all (test/transport/transport.md's own
// finding). Pulled out to connect.hpp as a pure function so it fires here.

// KNOWN_ISSUES.md D18: connect_mode_components() used to refuse anything
// other than exactly two Connect-mode components (RefusesAThirdConnect
// ModeComponent's own old name and assertion, before this round); the
// N-way chain generalisation (connect_chain_plan(), immediately below)
// means three -- or any count >= 2 -- is now accepted here, in the
// graph's own declaration order. The chain's OWN per-step rules (each
// joining component beyond the first must declare is_acceptor and a
// port_file) are connect_chain_plan()'s job, tested separately below;
// this function's only remaining limit is "at least two".
TEST(ConnectModeComponents, AcceptsThreeConnectModeComponentsInGraphOrder) {
  CouplingGraph g;
  const LaunchSpec connect_a{
      .mode = LaunchMode::Connect, .is_acceptor = true, .port_file = "/tmp/x"};
  const LaunchSpec connect_b{.mode = LaunchMode::Connect, .port_file = "/tmp/x"};
  const LaunchSpec connect_c{
      .mode = LaunchMode::Connect, .is_acceptor = true, .port_file = "/tmp/y"};
  g.add_component({.name = "a", .ranks = {0, 1}, .launch = connect_a});
  g.add_component({.name = "b", .ranks = {1, 1}, .launch = connect_b});
  g.add_component({.name = "c", .ranks = {2, 1}, .launch = connect_c});

  const auto r = fuse::transport::connect_mode_components(g);
  ASSERT_TRUE(r.ok()) << r.error().message;
  ASSERT_EQ(r.value().size(), 3u);
  EXPECT_EQ(r.value()[0], ComponentId{0});
  EXPECT_EQ(r.value()[1], ComponentId{1});
  EXPECT_EQ(r.value()[2], ComponentId{2});
}

TEST(ConnectModeComponents, RefusesJustOneConnectModeComponent) {
  CouplingGraph g;
  g.add_component({.name = "a",
                   .ranks = {0, 1},
                   .launch = {.mode = LaunchMode::Connect,
                              .is_acceptor = true,
                              .port_file = "/tmp/x"}});
  g.add_component({.name = "serial", .ranks = {1, 1}});  // InProcess default

  const auto r = fuse::transport::connect_mode_components(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
}

TEST(ConnectModeComponents, AcceptsExactlyTwoInGraphOrder) {
  const CouplingGraph g = two_component_graph(
      LaunchSpec{.mode = LaunchMode::Connect, .is_acceptor = true, .port_file = "/tmp/x"},
      LaunchSpec{.mode = LaunchMode::Connect, .port_file = "/tmp/x"}, {0, 1}, {1, 1});

  const auto r = fuse::transport::connect_mode_components(g);
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_EQ(r.value().size(), 2u);
  EXPECT_EQ(r.value()[0], ComponentId{0});
  EXPECT_EQ(r.value()[1], ComponentId{1});
}

// ---- connect_chain_plan() (KNOWN_ISSUES.md D18's N-way generalisation) ---

// Three components: step 1 (a/b) is today's two-component rule,
// unchanged; step 2 (the chain so far, joined by c) requires c itself
// to declare is_acceptor and a port_file -- "each component publishes
// its port name once", connect.hpp's own rule.
TEST(ConnectChainPlan, ThreeComponentsProduceTwoSteps) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = {0, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/ab"}});
  g.add_component({.name = "b", .ranks = {1, 1},
                   .launch = {.mode = LaunchMode::Connect, .port_file = "/tmp/ab"}});
  g.add_component({.name = "c", .ranks = {2, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/c"}});

  const auto r = fuse::transport::connect_chain_plan(g);
  ASSERT_TRUE(r.ok()) << r.error().message;
  ASSERT_EQ(r.value().size(), 2u);
  EXPECT_EQ(r.value()[0].joining, ComponentId{1});
  EXPECT_EQ(r.value()[0].step_index, 1u);
  EXPECT_EQ(r.value()[1].joining, ComponentId{2});
  EXPECT_EQ(r.value()[1].step_index, 2u);
}

// Four components: three steps, each later one (b joining at step 1 is
// exempt -- the unchanged two-component rule; c at step 2 and d at step
// 3 both need their own is_acceptor/port_file).
TEST(ConnectChainPlan, FourComponentsProduceThreeSteps) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = {0, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/ab"}});
  g.add_component({.name = "b", .ranks = {1, 1},
                   .launch = {.mode = LaunchMode::Connect, .port_file = "/tmp/ab"}});
  g.add_component({.name = "c", .ranks = {2, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/c"}});
  g.add_component({.name = "d", .ranks = {3, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/d"}});

  const auto r = fuse::transport::connect_chain_plan(g);
  ASSERT_TRUE(r.ok()) << r.error().message;
  ASSERT_EQ(r.value().size(), 3u);
  EXPECT_EQ(r.value()[0].joining, ComponentId{1});
  EXPECT_EQ(r.value()[1].joining, ComponentId{2});
  EXPECT_EQ(r.value()[1].step_index, 2u);
  EXPECT_EQ(r.value()[2].joining, ComponentId{3});
  EXPECT_EQ(r.value()[2].step_index, 3u);
}

TEST(ConnectChainPlan, RefusesAStepTwoJoinerThatDoesNotDeclareIsAcceptor) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = {0, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/ab"}});
  g.add_component({.name = "b", .ranks = {1, 1},
                   .launch = {.mode = LaunchMode::Connect, .port_file = "/tmp/ab"}});
  g.add_component({.name = "c", .ranks = {2, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = false,
                              .port_file = "/tmp/c"}});

  const auto r = fuse::transport::connect_chain_plan(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
  EXPECT_NE(r.error().message.find("c"), std::string::npos) << r.error().message;
  EXPECT_NE(r.error().message.find("is_acceptor"), std::string::npos) << r.error().message;
}

TEST(ConnectChainPlan, RefusesAStepTwoJoinerWithNoPortFile) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = {0, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/ab"}});
  g.add_component({.name = "b", .ranks = {1, 1},
                   .launch = {.mode = LaunchMode::Connect, .port_file = "/tmp/ab"}});
  g.add_component({.name = "c", .ranks = {2, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true}});

  const auto r = fuse::transport::connect_chain_plan(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
  EXPECT_NE(r.error().message.find("port_file"), std::string::npos) << r.error().message;
}

TEST(ConnectChainPlan, RefusesWithFewerThanTwoConnectModeComponents) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = {0, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/x"}});
  g.add_component({.name = "serial", .ranks = {1, 1}});  // InProcess default

  const auto r = fuse::transport::connect_chain_plan(g);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-SPEC");
}

// validate_launch_layout() is unaffected by the chain generalisation: a
// 3- or 4-component Connect chain is still one consecutive block per
// component in graph order (D36's gap rule, tested above with mpmd/
// in-process blocks, applies identically regardless of how many
// Connect-mode blocks there are).
TEST(ValidateLaunchLayout, AcceptsThreeConsecutiveConnectBlocksInGraphOrder) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = {0, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/ab"}});
  g.add_component({.name = "b", .ranks = {1, 1},
                   .launch = {.mode = LaunchMode::Connect, .port_file = "/tmp/ab"}});
  g.add_component({.name = "c", .ranks = {2, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/c"}});

  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << r.error().message;
}

TEST(ValidateLaunchLayout, AcceptsFourConsecutiveConnectBlocksWithAGapFilledByInProcess) {
  CouplingGraph g;
  g.add_component({.name = "a", .ranks = {0, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/ab"}});
  g.add_component({.name = "b", .ranks = {1, 1},
                   .launch = {.mode = LaunchMode::Connect, .port_file = "/tmp/ab"}});
  // A gap at rank 2, covered by an in-process component (D36's own
  // rule), before the third and fourth Connect-mode blocks.
  g.add_component({.name = "filler", .ranks = {2, 1}});  // InProcess default
  g.add_component({.name = "c", .ranks = {3, 1},
                   .launch = {.mode = LaunchMode::Connect, .is_acceptor = true,
                              .port_file = "/tmp/c"}});
  g.add_component({.name = "d", .ranks = {4, 1},
                   .launch = {.mode = LaunchMode::Connect, .port_file = "/tmp/c"}});

  const auto r = validate_launch_layout(g);
  EXPECT_TRUE(r.ok()) << r.error().message;
}

// ---- the accept/connect capability gate (KNOWN_ISSUES D23) ---------------

// The verdict itself is settled at configure time, so the refusal is reached
// in a normal build only on a machine whose MPI cannot carry the mode. Taking
// the decision as an argument is what lets it be exercised anywhere: the gate
// that nobody can make fire is the gate nobody has read.
TEST(ConnectAcceptCapability, RefusesAConnectComponentWhenTheModeIsUnavailable) {
  const auto r = fuse::transport::check_connect_accept_available(
      /*enabled=*/false, "ocean");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-CONNECT-ACCEPT-UNAVAILABLE");
  // The message must name the component, so a reader knows WHICH declaration
  // to change, and the way out, so the refusal is actionable rather than a
  // dead end: MPMD needs nothing this MPI lacks.
  EXPECT_NE(r.error().message.find("ocean"), std::string::npos)
      << r.error().message;
  EXPECT_NE(r.error().message.find("MPMD"), std::string::npos)
      << r.error().message;
  EXPECT_NE(r.error().message.find("D23"), std::string::npos)
      << r.error().message;
}

TEST(ConnectAcceptCapability, SaysNothingWhenTheModeIsAvailable) {
  EXPECT_TRUE(
      fuse::transport::check_connect_accept_available(true, "ocean").ok());
}

// Whatever this build decided, the constant and the build setting agree --
// a consumer compiled against a different value than the library would take
// the opposite branch at the call site from the one the library serves.
TEST(ConnectAcceptCapability, TheConstantMatchesTheBuildSetting) {
#if defined(FUSE_CONNECT_ACCEPT_ENABLED) && FUSE_CONNECT_ACCEPT_ENABLED
  EXPECT_TRUE(fuse::transport::kConnectAcceptEnabled);
#else
  EXPECT_FALSE(fuse::transport::kConnectAcceptEnabled);
#endif
}
