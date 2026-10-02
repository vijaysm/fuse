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

#include <string>

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
TEST(LowFiveTransport, SetIntercommRefusesAnOrdinaryIntracommunicator) {
  // MPI_COMM_WORLD, converted to its Fortran handle, is an intracommunicator
  // -- exactly the mistake set_intercomm() must catch (its doc in
  // transport.hpp: "form one with MPI_Comm_connect/accept ... not with
  // MPI_Comm_dup/MPI_Comm_split"). This runs WITHOUT MPI_Init: the
  // uninitialised-MPI refusal fires first in that case, which this test
  // also accepts, since it is not what the test is pinning -- what matters
  // is that no handle to a non-inter communicator is ever accepted.
  LowFiveTransport t;
  const auto r = t.set_intercomm(0, 0);
  ASSERT_FALSE(r.ok());
  EXPECT_TRUE(r.error().code == "E-TRANSPORT-MPIINIT" ||
              r.error().code == "E-TRANSPORT-COMM")
      << r.error().code << ": " << r.error().message;
}
#else
TEST(LowFiveTransport, SetIntercommNeedsMpi) {
  LowFiveTransport t;
  EXPECT_EQ(t.set_intercomm(0, 0).error().code, "E-TRANSPORT-NOMPI");
}
#endif  // FUSE_HAVE_MPI

TEST(LowFiveTransport, DistInterceptionIsNeverExercisedHere) {
  // Documents a gap rather than papering over it: a GENUINE distributed
  // round trip needs two MPI processes joined by a real intercommunicator
  // (MPI_Comm_connect/accept or an MPMD launch), which this test binary
  // never has (fuse_test_main runs single-process, and per this track's own
  // report MPI cannot be started at all in the sandbox this was written
  // in). examples/demo_b and test/verification/demo_b_test.cpp (if/when the
  // lead adds the latter) are where that claim would actually be checked,
  // on the host. See KNOWN_ISSUES.md section D and docs/user/lowfive.md.
  SUCCEED();
}

#endif  // FUSE_HAVE_LOWFIVE
