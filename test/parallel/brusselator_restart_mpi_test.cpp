// The collective multi-task WRITE (fuse::restart_io::
// create_restart_point_collective(), src/restart_io/hdf5_restart_io_mpi.cpp)
// and the rank-count-independent READ (fuse::restart_io::
// open_restart_point(), UNCHANGED since it already resolves every row by
// id, never by position -- Restart.GetFieldResolvesRowsByIdNotByFilePosition,
// test/runtime/restart_test.cpp), over a
// Brusselator-SHAPED field (a representative U-grid-sized array, real
// global ids, deterministic values every rank can verify independently
// without needing the writer's own in-memory state).
//
// This file is registered with fuse_add_mpi_test, RANKS 2 3, and needs a
// working MPI launcher (docs/dev/testing.md).
//
// Declared scope reduction: this file exercises the restart_io layer
// DIRECTLY (fuse::restart_io::create_restart_point_collective()/
// open_restart_point()), not a full Runner-driven Brusselator round trip
// over real MPI ranks -- building a genuinely rank-partitioned
// Brusselator Runner (real Transport, real register_default_placements()
// per-rank partitioning) is separate engineering; the Runner-level orchestration
// (fuse::write_restart_point()/restore_restart_point(), fuse/runtime/
// restart.hpp) is parameterized ONLY by which RestartWriter/RestartReader
// implementation it is handed, so this file's own collective-write
// correctness composes with the already-proven tier-0 Runner-level round
// trip (test/verification/brusselator_restart_test.cpp) rather than
// needing to re-prove it.
#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "fuse/restart_io/hdf5_restart_io.hpp"
#include "fuse/restart_io/hdf5_restart_io_mpi.hpp"

namespace {

using fuse::restart_io::create_restart_point_collective;
using fuse::restart_io::FileLayout;
using fuse::restart_io::open_restart_point;
using fuse::restart_io::Provenance;

constexpr const char kFieldName[] = "U/u_state";
constexpr std::int64_t kGlobalN = 25;  // brusselator_restart_test.cpp's own kNxU*kNyU (5x5)

/// Every rank's own value for global id `id` -- a pure function of the id
/// alone, so ANY rank, at ANY rank count, can independently recompute
/// "what should be at this id" without needing the writer's own values in
/// memory. Mirrors the tier-0 fixtures' preference for a
/// deterministic, checkable pattern over a hardcoded table.
[[nodiscard]] double expected_value(std::int64_t id) {
  return 100.0 + static_cast<double>(id) * 0.5;
}

/// Rank r of `size` owns [r*chunk, (r+1)*chunk), the last rank taking the
/// remainder -- the SAME convention test/parallel/restart_collective_mpi_
/// test.cpp's own my_disjoint_range() uses, generalised here to an
/// explicit (rank, size) pair rather than always reading the real
/// communicator's own, so a sub-communicator's own partition can be
/// computed the identical way.
[[nodiscard]] std::pair<std::int64_t, std::int64_t> disjoint_range(
    int rank, int size, std::int64_t global_n) {
  const std::int64_t chunk = global_n / size;
  const std::int64_t lo = static_cast<std::int64_t>(rank) * chunk;
  const std::int64_t hi = (rank == size - 1) ? global_n : lo + chunk;
  return {lo, hi};
}

fuse::Result<void> write_collective(MPI_Comm comm, const std::string& path,
                                    std::int64_t global_n) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  Provenance prov;
  prov.format_version = fuse::restart_io::kFormatVersion;
  prov.fuse_version = "test";
  prov.mesh_checksum = 0;
  const int comm_f = MPI_Comm_c2f(comm);
  auto pw = create_restart_point_collective(FileLayout::SeparateMeshAndPoints,
                                            path, prov, comm_f);
  if (!pw.ok()) return pw.error();
  int size = 0;
  MPI_Comm_size(comm, &size);
  const auto [lo, hi] = disjoint_range(rank, size, global_n);
  std::vector<std::int64_t> ids;
  std::vector<double> values;
  for (std::int64_t id = lo; id < hi; ++id) {
    ids.push_back(id);
    values.push_back(expected_value(id));
  }
  const fuse::Result<void> wf = pw.value()->writer().put_field<double>(
      kFieldName, ids, std::span<const double>(values), 1, global_n);
  if (!wf.ok()) return wf;
  return pw.value()->finish();
}

/// Every rank independently reads its OWN current partition's ids back
/// and checks them against expected_value() -- the SAME
/// open_restart_point()/get_field() path a single-rank reader already
/// uses, UNCHANGED, run from multiple real ranks at once. No
/// communicator is needed for this half at all: each rank's own read is
/// a fully independent, ordinary file open.
fuse::Result<void> read_and_verify(const std::string& path, int rank, int size,
                                   std::int64_t global_n) {
  auto pr = open_restart_point(FileLayout::SeparateMeshAndPoints, path);
  if (!pr.ok()) return pr.error();
  const auto [lo, hi] = disjoint_range(rank, size, global_n);
  std::vector<std::int64_t> ids;
  for (std::int64_t id = lo; id < hi; ++id) ids.push_back(id);
  std::vector<double> values(ids.size());
  const auto rr = pr.value()->reader().get_field<double>(
      kFieldName, ids, std::span<double>(values), 1);
  if (!rr.ok()) return rr.error();
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (values[i] != expected_value(ids[i])) {
      return fuse::Failure{"E-TEST-MISMATCH",
                           "read_and_verify: id " + std::to_string(ids[i]) +
                               " got " + std::to_string(values[i]) +
                               " expected " +
                               std::to_string(expected_value(ids[i]))};
    }
  }
  return fuse::Result<void>::success();
}

std::string temp_dir_path(const std::string& leaf) {
  return std::string(::testing::TempDir()) +
        "fuse_brusselator_restart_mpi_test_" + leaf;
}

/// (i) Write at N real ranks, restore at the SAME N: every rank's own
/// restored rows equal, BY GLOBAL ID, exactly what it wrote -- the
/// strongest exact comparison restart.md Section 7.ii allows, since
/// writer and reader agree on the decomposition by construction.
TEST(BrusselatorRestartMpi, WriteAtNRanksRestoreAtSameNByGlobalId) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  const std::string path = temp_dir_path("same_n.h5");
  if (fuse::testing::rank() == 0) std::remove(path.c_str());
  MPI_Barrier(MPI_COMM_WORLD);
  const fuse::Result<void> w = write_collective(MPI_COMM_WORLD, path, kGlobalN);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;
  MPI_Barrier(MPI_COMM_WORLD);
  const fuse::Result<void> r =
      read_and_verify(path, fuse::testing::rank(), fuse::testing::size(), kGlobalN);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  MPI_Barrier(MPI_COMM_WORLD);
  if (fuse::testing::rank() == 0) std::remove(path.c_str());
}

/// (ii), 1 -> current size: rank 0 ALONE writes serially (the EXISTING,
/// unchanged create_restart_point() -- "write at 1 rank" IS the
/// already-proven tier-0 path, exercised here only for the file every
/// OTHER rank then reads). Every rank (including rank 0) then
/// independently restores its OWN current-size partition and checks it
/// BY GLOBAL ID against the SAME expected_value() the writer used --
/// proving the read side is genuinely rank-count-independent of the
/// write side, not merely unchanged in isolation.
TEST(BrusselatorRestartMpi, WriteAtOneRankRestoreAtCurrentSizeByGlobalId) {
  const std::string path = temp_dir_path("one_to_n.h5");
  if (fuse::testing::rank() == 0) {
    std::remove(path.c_str());
    Provenance prov;
    prov.format_version = fuse::restart_io::kFormatVersion;
    prov.fuse_version = "test";
    prov.mesh_checksum = 0;
    auto pw = fuse::restart_io::create_restart_point(
        FileLayout::SeparateMeshAndPoints, path, prov);
    ASSERT_TRUE(pw.ok()) << pw.error().message;
    std::vector<std::int64_t> ids(static_cast<std::size_t>(kGlobalN));
    std::vector<double> values(static_cast<std::size_t>(kGlobalN));
    for (std::int64_t id = 0; id < kGlobalN; ++id) {
      ids[static_cast<std::size_t>(id)] = id;
      values[static_cast<std::size_t>(id)] = expected_value(id);
    }
    const fuse::Result<void> wf = pw.value()->writer().put_field<double>(
        kFieldName, ids, std::span<const double>(values), 1, kGlobalN);
    ASSERT_TRUE(wf.ok()) << wf.error().message;
    ASSERT_TRUE(pw.value()->finish().ok());
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const fuse::Result<void> r =
      read_and_verify(path, fuse::testing::rank(), fuse::testing::size(), kGlobalN);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  MPI_Barrier(MPI_COMM_WORLD);
  if (fuse::testing::rank() == 0) std::remove(path.c_str());
}

/// (ii), fewer ranks -> current size (covers 2 -> 3 at RANKS 3): a
/// sub-communicator of exactly 2 ranks (0 and 1; rank 2, if present,
/// does not enter the collective write at all) writes COLLECTIVELY,
/// then every rank of the FULL communicator restores its OWN
/// current-size partition. Requires at least 3 ranks so the sub-comm
/// (2) is genuinely FEWER than the full one (3) -- at exactly 2 ranks
/// this would be the identical case WriteAtNRanksRestoreAtSameNByGlobalId
/// already covers, so it skips rather than silently repeating that
/// test under a different name.
TEST(BrusselatorRestartMpi, WriteAtFewerRanksRestoreAtMoreByGlobalId) {
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
  const int rank = fuse::testing::rank();
  const int color = (rank < 2) ? 0 : MPI_UNDEFINED;
  MPI_Comm sub_comm = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub_comm);
  const std::string path = temp_dir_path("fewer_to_more.h5");
  if (rank == 0) std::remove(path.c_str());
  MPI_Barrier(MPI_COMM_WORLD);
  if (sub_comm != MPI_COMM_NULL) {
    const fuse::Result<void> w = write_collective(sub_comm, path, kGlobalN);
    ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;
    MPI_Comm_free(&sub_comm);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  const fuse::Result<void> r =
      read_and_verify(path, rank, fuse::testing::size(), kGlobalN);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::remove(path.c_str());
}

/// (ii), MORE ranks -> FEWER (covers 3 -> 2): genuinely needs a smaller
/// real communicator than the one that wrote the file, which no single
/// mpiexec invocation can form (a sub-communicator is always <= the
/// invocation's own size, never >). Writes to a path named by
/// FUSE_BRUSSELATOR_RESTART_MPI_FIXED_PATH when that variable is set
/// (the two-invocation sequence: run this
/// binary at 3 ranks first with the variable set, then at 2 ranks with
/// the SAME variable set, in that order) -- skips, naming why, when it
/// is not, rather than silently asserting nothing.
TEST(BrusselatorRestartMpi, WriteAtMoreRanksRestoreAtFewerByGlobalId) {
  const char* fixed = std::getenv("FUSE_BRUSSELATOR_RESTART_MPI_FIXED_PATH");
  if (fixed == nullptr) {
    GTEST_SKIP() << "needs FUSE_BRUSSELATOR_RESTART_MPI_FIXED_PATH set and "
                    "this binary run at a LARGER rank count first, then a "
                    "SMALLER one against the same path";
  }
  const std::string path(fixed);
  const int rank = fuse::testing::rank();
  const int size = fuse::testing::size();
  // The FIRST (larger-rank-count) invocation writes; the SECOND
  // (smaller) one only reads. Both invocations run this same test, so
  // the write must be idempotent-safe: only perform it when this
  // invocation's own size is the one the host sequence names as the
  // WRITER (odd-one-out convention: the host sets a SECOND variable to
  // say which role this invocation plays, rather than guessing from
  // size() alone, since "3 then 2" and "2 then 3" are both meaningful
  // sequences and only the host's own script knows which it is running).
  const char* role = std::getenv("FUSE_BRUSSELATOR_RESTART_MPI_ROLE");
  ASSERT_NE(role, nullptr) << "FUSE_BRUSSELATOR_RESTART_MPI_ROLE must be "
                              "'write' or 'read' alongside the fixed path";
  if (std::string(role) == "write") {
    if (rank == 0) std::remove(path.c_str());
    MPI_Barrier(MPI_COMM_WORLD);
    const fuse::Result<void> w = write_collective(MPI_COMM_WORLD, path, kGlobalN);
    ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;
    return;
  }
  ASSERT_EQ(std::string(role), "read");
  const fuse::Result<void> r = read_and_verify(path, rank, size, kGlobalN);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

}  // namespace
