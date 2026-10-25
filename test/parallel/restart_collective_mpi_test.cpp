// The cross-rank duplicate-id and coverage check
// (fuse::restart_io::collective_duplicate_and_coverage_check(),
// src/restart_io/collective_check_mpi.cpp), over a REAL MPI_COMM_WORLD.
//
// Registered with fuse_add_mpi_test, RANKS 2 3; needs a working MPI
// launcher (docs/dev/testing.md).
//
// Every test below makes every rank compute the SAME expected verdict
// from a pattern that scales with fuse::testing::size(), rather than
// hardcoding a 2-rank or 3-rank shape, so the identical file is a real
// test of BOTH registered rank counts, not two different scenarios that
// happen to share a filename.
#include "fuse/restart_io/collective_check_mpi.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

namespace {

using fuse::restart_io::collective_duplicate_and_coverage_check;

int comm_f() { return MPI_Comm_c2f(MPI_COMM_WORLD); }

/// Every rank owns a disjoint slice of [0, global_n): rank r owns
/// [r*chunk, (r+1)*chunk), except the LAST rank, which also takes
/// whatever remainder does not divide evenly -- the same "last rank
/// takes the remainder" convention linalg_mpi_test.cpp's own
/// my_row_range() helper uses, generalised from 2 ranks to size().
std::pair<std::int64_t, std::int64_t> my_disjoint_range(std::int64_t global_n) {
  const int r = fuse::testing::rank();
  const int n = fuse::testing::size();
  const std::int64_t chunk = global_n / n;
  const std::int64_t lo = static_cast<std::int64_t>(r) * chunk;
  const std::int64_t hi = (r == n - 1) ? global_n : lo + chunk;
  return {lo, hi};
}

TEST(RestartCollectiveMpi, DisjointIdsSameFieldAcrossRanksSucceeds) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  constexpr std::int64_t global_n = 12;
  const auto [lo, hi] = my_disjoint_range(global_n);
  std::vector<std::pair<std::string, std::int64_t>> local_ids;
  for (std::int64_t i = lo; i < hi; ++i) {
    local_ids.emplace_back("brusselator/u", i);
  }
  std::unordered_map<std::string, std::int64_t> declared{
      {"brusselator/u", global_n}};
  const fuse::Result<void> r =
      collective_duplicate_and_coverage_check(comm_f(), local_ids, declared);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

TEST(RestartCollectiveMpi, IdenticalLocalIdsDifferentFieldNamesAcrossRanksSucceeds) {
  // The cross-rank analog of the single-rank EXACT-KEY check: every
  // rank reports the IDENTICAL local ids {0,1,2,3}, but under a field
  // name that is UNIQUE to that rank -- dirichlet/neumann's own shape,
  // generalised to size() components instead of 2. A hash-combined key
  // would risk a collision here if any two
  // ranks' field-name hashes collided in their own high bits; this
  // exact-key design cannot collide at all, because two different field
  // names are never written under the same key by construction.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  const std::string field_name =
      "component_" + std::to_string(fuse::testing::rank()) + "/field";
  std::vector<std::pair<std::string, std::int64_t>> local_ids = {
      {field_name, 0}, {field_name, 1}, {field_name, 2}, {field_name, 3}};
  std::unordered_map<std::string, std::int64_t> declared{{field_name, 4}};
  const fuse::Result<void> r =
      collective_duplicate_and_coverage_check(comm_f(), local_ids, declared);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

TEST(RestartCollectiveMpi, DuplicateIdAcrossRanksForSameFieldRefuses) {
  // Every rank reports the SAME (name, id) pair for the SAME field --
  // the genuinely cross-rank duplicate, which a single-rank run cannot
  // even express.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  std::vector<std::pair<std::string, std::int64_t>> local_ids = {
      {"brusselator/u", 5}};
  std::unordered_map<std::string, std::int64_t> declared{{"brusselator/u", 1}};
  const fuse::Result<void> r =
      collective_duplicate_and_coverage_check(comm_f(), local_ids, declared);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RESTART-DUPLICATE-ID");
}

TEST(RestartCollectiveMpi, DeclaredSizeDisagreementAcrossRanksRefuses) {
  // Rank 0 declares a different global size than every other rank for
  // the SAME field -- the cross-rank half of E-RESTART-GLOBAL-SIZE-
  // MISMATCH, which a single-rank run can only check within one rank's own
  // calls, never across two different callers on two different ranks.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  const std::int64_t declared_value = (fuse::testing::rank() == 0) ? 10 : 11;
  std::vector<std::pair<std::string, std::int64_t>> local_ids = {
      {"x", static_cast<std::int64_t>(fuse::testing::rank())}};
  std::unordered_map<std::string, std::int64_t> declared{{"x", declared_value}};
  const fuse::Result<void> r =
      collective_duplicate_and_coverage_check(comm_f(), local_ids, declared);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RESTART-GLOBAL-SIZE-MISMATCH");
}

TEST(RestartCollectiveMpi, UndercoverageAcrossRanksRefuses) {
  // Every rank agrees on a declared size the combined collective never
  // reaches: each rank reports only ONE id (itself), but every rank
  // declares a global size of 1000 -- the sum across the whole
  // collective (size(), at most a handful of ranks here) never gets
  // close.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  std::vector<std::pair<std::string, std::int64_t>> local_ids = {
      {"x", static_cast<std::int64_t>(fuse::testing::rank())}};
  std::unordered_map<std::string, std::int64_t> declared{{"x", 1000}};
  const fuse::Result<void> r =
      collective_duplicate_and_coverage_check(comm_f(), local_ids, declared);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RESTART-BAD-NUMBERING");
}

}  // namespace
