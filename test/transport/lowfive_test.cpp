// S6 phase 3, "LowFive" track. Everything in this file compiles and runs in
// EVERY configuration, including tier 0 (no LowFive, no MPI, no MOAB) --
// see the comment at the top of src/transport/lowfive_transport.cpp for why
// that file is structured the same way. The LowFiveFieldBindingTable tests
// need no HDF5 at all (fuse/transport/lowfive.hpp is header-only, pure
// string parsing). The LowFiveTransport tests assert the REFUSAL path in a
// build without LowFive (`E-TRANSPORT-NOLOWFIVE` etc.) and, only where
// `FUSE_HAVE_LOWFIVE`/`FUSE_HAVE_MPI` actually let the connector attach, the
// setup-level checks that do not require MPI to be running (the distributed
// `set_intercomm` path needs a live MPI_Init -- see DistInterceptionIsNever
// ExercisedHere below for what that means for this file's own coverage).

#include "fuse/base/config.hpp"  // FUSE_HAVE_LOWFIVE, FUSE_HAVE_MPI -- see note below
#include <fuse/transport/lowfive.hpp>
#include <fuse/transport/transport.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <dirent.h>
#include <unistd.h>

using fuse::LowFiveFieldBinding;
using fuse::LowFiveFieldBindingTable;
using fuse::LowFiveTransport;
using fuse::split_hdf5_path;
using fuse::TransportSpec;

// ---------------------------------------------------------------------------
// split_hdf5_path / LowFiveFieldBindingTable -- no LowFive, no HDF5, no MPI.
// ---------------------------------------------------------------------------

TEST(SplitHdf5Path, SplitsFileAndDataset) {
  const auto r = split_hdf5_path("producer_out.h5/grid/temperature");
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_EQ(r.value().first, "producer_out.h5");
  EXPECT_EQ(r.value().second, "grid/temperature");
}

TEST(SplitHdf5Path, WildcardFileIsJustAnOrdinaryFileComponent) {
  const auto r = split_hdf5_path("*.h5/particles");
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_EQ(r.value().first, "*.h5");
  EXPECT_EQ(r.value().second, "particles");
}

TEST(SplitHdf5Path, RefusesNoSeparator) {
  const auto r = split_hdf5_path("not_an_hdf5_path");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-PORT");
}

TEST(SplitHdf5Path, RefusesTwoSeparators) {
  const auto r = split_hdf5_path("a.h5/b.h5/c");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-PORT");
}

TEST(SplitHdf5Path, RefusesEmptyFileComponent) {
  const auto r = split_hdf5_path(".h5/dataset");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-PORT");
}

TEST(SplitHdf5Path, RefusesEmptyDatasetComponent) {
  const auto r = split_hdf5_path("file.h5/");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-PORT");
}

TEST(LowFiveFieldBindingTable, AddsAndFindsByFieldAndPath) {
  LowFiveFieldBindingTable table;
  const auto r = table.add(LowFiveFieldBinding{
      .field_name = "heat_flux", .hdf5_path = "producer_out.h5/grid/q"});
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_EQ(table.size(), 1u);

  const auto* by_field = table.find_by_field("heat_flux");
  ASSERT_NE(by_field, nullptr);
  EXPECT_EQ(by_field->hdf5_path, "producer_out.h5/grid/q");

  const auto* by_path = table.find_by_path("producer_out.h5/grid/q");
  ASSERT_NE(by_path, nullptr);
  EXPECT_EQ(by_path->field_name, "heat_flux");

  EXPECT_EQ(table.find_by_field("nope"), nullptr);
  EXPECT_EQ(table.find_by_path("nope.h5/x"), nullptr);
}

TEST(LowFiveFieldBindingTable, RefusesAMalformedPath) {
  LowFiveFieldBindingTable table;
  const auto r = table.add(
      LowFiveFieldBinding{.field_name = "q", .hdf5_path = "no_separator"});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-PORT");
  EXPECT_TRUE(table.empty());
}

TEST(LowFiveFieldBindingTable, RefusesRebindingAFieldName) {
  LowFiveFieldBindingTable table;
  ASSERT_TRUE(table.add(LowFiveFieldBinding{.field_name = "q",
                                            .hdf5_path = "a.h5/x"})
                  .ok());
  const auto r = table.add(
      LowFiveFieldBinding{.field_name = "q", .hdf5_path = "b.h5/y"});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-DUPFIELD");
  EXPECT_EQ(table.size(), 1u);
}

TEST(LowFiveFieldBindingTable, RefusesTwoFieldsOnOneExactPath) {
  LowFiveFieldBindingTable table;
  ASSERT_TRUE(table.add(LowFiveFieldBinding{.field_name = "q",
                                            .hdf5_path = "a.h5/x"})
                  .ok());
  const auto r = table.add(
      LowFiveFieldBinding{.field_name = "r", .hdf5_path = "a.h5/x"});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-DUPPATH");
}

TEST(LowFiveFieldBindingTable, AllowsTwoWildcardFilesOnDifferentDatasets) {
  // "*" file components are not path-collision-checked against each other
  // here -- see the comment on LowFiveFieldBindingTable::add(). Two
  // DIFFERENT dataset paths under a wildcard file is the ordinary case.
  LowFiveFieldBindingTable table;
  ASSERT_TRUE(table.add(LowFiveFieldBinding{.field_name = "q",
                                            .hdf5_path = "*.h5/grid/q"})
                  .ok());
  const auto r = table.add(
      LowFiveFieldBinding{.field_name = "r", .hdf5_path = "*.h5/grid/r"});
  EXPECT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_EQ(table.size(), 2u);
}

// ---------------------------------------------------------------------------
// plan_lowfive_rendezvous -- Unit 3b's own rank-set arithmetic, pure and
// MPI-free (the same "split for testability" reasoning connect.hpp's own
// module comment gives), so it runs in the base build configuration this
// sandbox can always start rather than only inside an MPI job it cannot.
// ---------------------------------------------------------------------------

TEST(PlanLowFiveRendezvous, WriterSideReadsItsOwnSplitAndRemoteLeader) {
  const fuse::RankSet writer{0, 2};   // {0, 1}
  const fuse::RankSet reader{2, 3};   // {2, 3, 4}
  const auto p = fuse::plan_lowfive_rendezvous(writer, reader, /*this_rank=*/1);
  ASSERT_TRUE(p.ok()) << p.error().code << ": " << p.error().message;
  EXPECT_TRUE(p.value().is_writer);
  EXPECT_EQ(p.value().split_color, 0);
  EXPECT_EQ(p.value().split_key, 1);     // rank 1 is index 1 within {0,1}
  EXPECT_EQ(p.value().local_leader, 0);
  EXPECT_EQ(p.value().remote_leader, 2); // reader's own min_rank()
}

TEST(PlanLowFiveRendezvous, ReaderSideReadsItsOwnSplitAndRemoteLeader) {
  const fuse::RankSet writer{0, 2};   // {0, 1}
  const fuse::RankSet reader{2, 3};   // {2, 3, 4}
  const auto p = fuse::plan_lowfive_rendezvous(writer, reader, /*this_rank=*/4);
  ASSERT_TRUE(p.ok()) << p.error().code << ": " << p.error().message;
  EXPECT_FALSE(p.value().is_writer);
  EXPECT_EQ(p.value().split_color, 1);
  EXPECT_EQ(p.value().split_key, 2);     // rank 4 is index 2 within {2,3,4}
  EXPECT_EQ(p.value().local_leader, 0);
  EXPECT_EQ(p.value().remote_leader, 0); // writer's own min_rank()
}

// A non-contiguous RankSet (the graph's own declared order, not a launcher
// assumption of contiguity) still gives a stable split_key via index_of,
// not the raw rank number -- the one place this arithmetic is genuinely
// non-trivial rather than reading a field straight off RankSet.
TEST(PlanLowFiveRendezvous, NonContiguousRankSetsUseIndexOfNotTheRawRank) {
  const fuse::RankSet writer{std::vector<int>{5, 1, 3}};   // sorted: {1, 3, 5}
  const fuse::RankSet reader{std::vector<int>{8, 2}};      // sorted: {2, 8}
  const auto p5 = fuse::plan_lowfive_rendezvous(writer, reader, 5);
  ASSERT_TRUE(p5.ok());
  EXPECT_EQ(p5.value().split_key, 2);   // 5 is index 2 within {1,3,5}
  EXPECT_EQ(p5.value().remote_leader, 2);
  const auto p8 = fuse::plan_lowfive_rendezvous(writer, reader, 8);
  ASSERT_TRUE(p8.ok());
  EXPECT_FALSE(p8.value().is_writer);
  EXPECT_EQ(p8.value().split_key, 1);   // 8 is index 1 within {2,8}
  EXPECT_EQ(p8.value().remote_leader, 1);
}

TEST(PlanLowFiveRendezvous, OverlappingRankSetsAreRefused) {
  const fuse::RankSet writer{0, 3};  // {0,1,2}
  const fuse::RankSet reader{2, 3};  // {2,3,4}
  const auto p = fuse::plan_lowfive_rendezvous(writer, reader, 2);
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-LOWFIVE-RENDEZVOUS-OVERLAP");
}

TEST(PlanLowFiveRendezvous, ARankInNeitherSetIsRefused) {
  const fuse::RankSet writer{0, 2};  // {0,1}
  const fuse::RankSet reader{2, 2};  // {2,3}
  const auto p = fuse::plan_lowfive_rendezvous(writer, reader, 9);
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-LOWFIVE-RENDEZVOUS-UNPLACED");
}

// ---------------------------------------------------------------------------
// LowFiveTransport -- the refusal path in a build without LowFive, and the
// setup-level checks that need no running MPI job in one with it.
// ---------------------------------------------------------------------------

TEST(LowFiveTransport, IsAvailableMatchesTheBuildConfiguration) {
#if FUSE_HAVE_LOWFIVE
  EXPECT_TRUE(LowFiveTransport::is_available());
#else
  EXPECT_FALSE(LowFiveTransport::is_available());
#endif
}

TEST(LowFiveTransport, NameAndLocality) {
  const LowFiveTransport t;
  EXPECT_EQ(t.name(), "lowfive");
  // False in every configuration: even the single-process interception path
  // sends data through the HDF5 stack rather than keeping it local, and a
  // build without LowFive has nothing AT ALL to call `is_local()` true of.
  EXPECT_FALSE(t.is_local());
}

#if !FUSE_HAVE_LOWFIVE

TEST(LowFiveTransport, EveryEntryPointRefusesWithoutLowFive) {
  LowFiveTransport t;
  EXPECT_EQ(t.set_location("a.h5", "d").error().code, "E-TRANSPORT-NOLOWFIVE");
  EXPECT_EQ(t.build_plan(TransportSpec{.elements = 1}).error().code,
            "E-TRANSPORT-NOLOWFIVE");
  EXPECT_EQ(t.set_intercomm(0, 0).error().code, "E-TRANSPORT-NOLOWFIVE");
  EXPECT_EQ(t.serve_as_producer().error().code, "E-TRANSPORT-NOLOWFIVE");
  EXPECT_EQ(t.receive_as_consumer().error().code, "E-TRANSPORT-NOLOWFIVE");
  double v = 1.0;
  EXPECT_EQ(t.move(&v, &v, 1).error().code, "E-TRANSPORT-NOLOWFIVE");
  EXPECT_LT(t.access_plist_handle(), 0);
  EXPECT_EQ(t.intercepted_writes(), 0u);
}

#else  // FUSE_HAVE_LOWFIVE

TEST(LowFiveTransport, SetLocationRequiresBothNames) {
  LowFiveTransport t;
  EXPECT_FALSE(t.set_location("", "dataset").ok());
  EXPECT_FALSE(t.set_location("file.h5", "").ok());
}

// Demo B launches its producer and consumer on one path in one directory.
// set_location() proves the directory writable by creating and removing a
// probe file; with a fixed probe name the second process's O_EXCL open failed
// while the first one's probe existed (measured on the owner's machine,
// 2026-10-13, E-TRANSPORT-LF-LOCATION). A file at that old fixed name stands
// in for the peer's live probe: the claim must still succeed, and no probe
// may be left behind.
TEST(LowFiveTransport, SetLocationIsNotBlockedByAPeersProbe) {
  char dir_template[] = "/tmp/fuse-lf-probe-XXXXXX";
  ASSERT_NE(::mkdtemp(dir_template), nullptr);
  const std::string dir = dir_template;
  const std::string file = dir + "/shared.h5";
  const std::string peer_probe = file + ".soma-writable-probe";
  {
    std::FILE* f = std::fopen(peer_probe.c_str(), "w");
    ASSERT_NE(f, nullptr);
    std::fclose(f);
  }
  LowFiveTransport t;
  const auto r = t.set_location(file, "/grid/values");
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().code + ": " + r.error().message);
  std::vector<std::string> left;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (const dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if (n != "." && n != ".." && n != "shared.h5.soma-writable-probe") left.push_back(n);
    }
    ::closedir(d);
  }
  EXPECT_TRUE(left.empty()) << "left in " << dir << ": " << (left.empty() ? "" : left.front());
  std::remove(peer_probe.c_str());
  ::rmdir(dir.c_str());
}

TEST(LowFiveTransport, BuildPlanBeforeSetLocationIsRefused) {
  LowFiveTransport t;
  const auto r = t.build_plan(TransportSpec{.elements = 1});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-LF-LOCATION");
}

TEST(LowFiveTransport, MoveBeforeBuildPlanIsRefused) {
  LowFiveTransport t;
  double v = 1.0;
  EXPECT_EQ(t.move(&v, &v, 1).error().code, "E-TRANSPORT-UNPLANNED");
}

#if FUSE_HAVE_MPI
#include <mpi.h>
TEST(LowFiveTransport, SetIntercommRefusesANullHandle) {
  // A handle of 0 is not a communicator. This must be refused BEFORE any
  // MPI query on it: the host run of 2026-10-04 aborted inside
  // MPI_Comm_test_inter (MPICH's default handler is fatal) because the
  // check came after the query. Without MPI_Init the uninitialised-MPI
  // refusal fires first, which is also acceptable here.
  LowFiveTransport t;
  const auto r = t.set_intercomm(0, 0);
  ASSERT_FALSE(r.ok());
  EXPECT_TRUE(r.error().code == "E-TRANSPORT-MPIINIT" ||
              r.error().code == "E-TRANSPORT-COMM" ||
              r.error().code == "E-TRANSPORT-NOMPI" ||
              r.error().code == "E-TRANSPORT-NOLOWFIVE")
      << r.error().code << ": " << r.error().message;
}

TEST(LowFiveTransport, SetIntercommRefusesAnOrdinaryIntracommunicator) {
  // MPI_COMM_WORLD, as its Fortran handle, is a real intracommunicator --
  // exactly the mistake set_intercomm() must catch (transport.hpp: "form
  // one with MPI_Comm_connect/accept ... not with MPI_Comm_dup/
  // MPI_Comm_split"). The check needs a live MPI; when the binary runs
  // without MPI_Init the test skips rather than pin the wrong refusal.
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) GTEST_SKIP() << "needs MPI_Init to hold a real intracommunicator";
  LowFiveTransport t;
  const int world = static_cast<int>(MPI_Comm_c2f(MPI_COMM_WORLD));
  const auto r = t.set_intercomm(world, world);
  ASSERT_FALSE(r.ok());
  EXPECT_TRUE(r.error().code == "E-TRANSPORT-COMM" ||
              r.error().code == "E-TRANSPORT-NOMPI" ||
              r.error().code == "E-TRANSPORT-NOLOWFIVE")
      << r.error().code << ": " << r.error().message;
}
#else
TEST(LowFiveTransport, SetIntercommNeedsMpi) {
  LowFiveTransport t;
  EXPECT_EQ(t.set_intercomm(0, 0).error().code, "E-TRANSPORT-NOMPI");
}
#endif  // FUSE_HAVE_MPI

// The distributed round trip (two processes joined by a real
// intercommunicator) is not exercised in this single-process binary; it is
// the ctest row example.demo_b (examples/CMakeLists.txt), which runs
// examples/demo_b/run_demo.sh. A placeholder SUCCEED() test that stood here
// was removed: it could not fail.

#endif  // FUSE_HAVE_LOWFIVE
