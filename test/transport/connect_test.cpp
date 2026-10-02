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
