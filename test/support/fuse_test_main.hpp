#pragma once
/// FUSE test harness: one GoogleTest main for every tier, MPI- and device-aware.
///
/// The harness exists because the two obvious ways of testing a distributed coupler
/// are both wrong:
///
///   * Plain GoogleTest under `mpiexec` lets rank 3 fail an assertion and return
///     while ranks 0-2 proceed into the next collective. The suite then HANGS, and a
///     hang is not a test result. This harness reduces the failure status across the
///     communicator at the end of every test, so a failure anywhere fails everywhere
///     — the same discipline the library itself applies with `Epoch`
///     (docs/CHARTER.md §2).
///   * Every rank printing its own progress makes the log unreadable and interleaved.
///     Only rank 0 keeps the default printer; other ranks stay silent unless they are
///     the rank that failed, and then they print only their own failures.
///
/// Device coverage is by execution space, not by `#ifdef`: a test written once with
/// `FUSE_TYPED_TEST_ON_SPACES` runs on every space the build enabled, so the device
/// leg tests the same assertions as the host leg rather than a parallel copy of them.

#include <gtest/gtest.h>

#include <fuse/base/config.hpp>   // FUSE_HAVE_MPI, FUSE_HAVE_KOKKOS

#include <Kokkos_Core.hpp>

#if FUSE_HAVE_MPI
#  include <mpi.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace fuse::testing {

/// Rank and size, valid in every tier (1/0 when built without MPI).
int rank();
int size();

/// Declare a test's rank requirement. A test that needs exactly N ranks must say so;
/// running it at the wrong size is a configuration error, reported once by rank 0,
/// not a silent pass. Registration is per ctest entry, so the matrix in
/// docs/PLAN.md §S5.5 is checkable by grep.
bool requires_ranks(int exact);
bool requires_at_least_ranks(int minimum);

/// Skip with a reason that names the missing capability, never a bare GTEST_SKIP.
#define FUSE_REQUIRE_RANKS(n)                                                            \
  do {                                                                                   \
    if (!::fuse::testing::requires_ranks(n)) {                                            \
      GTEST_SKIP() << "needs exactly " << (n) << " ranks, have " << ::fuse::testing::size(); \
    }                                                                                     \
  } while (0)

#define FUSE_REQUIRE_AT_LEAST_RANKS(n)                                                   \
  do {                                                                                   \
    if (!::fuse::testing::requires_at_least_ranks(n)) {                                   \
      GTEST_SKIP() << "needs >= " << (n) << " ranks, have " << ::fuse::testing::size();   \
    }                                                                                     \
  } while (0)

/// The spaces this build enabled, in a fixed order, for typed tests.
using EnabledExecutionSpaces = ::testing::Types<
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::Cuda,
#elif defined(KOKKOS_ENABLE_HIP)
    Kokkos::HIP,
#elif defined(KOKKOS_ENABLE_SYCL)
    Kokkos::Experimental::SYCL,
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    Kokkos::OpenMP,
#endif
// Threads was missing until 2026-09-17, and the omission mattered: the
// machine-tuned Kokkos this project resolves against reports
// `Kokkos_DEVICES = THREADS;SERIAL` with NO OpenMP, so every typed test
// instantiated exactly ONCE, on Serial. "Runs on every enabled space" was
// true and vacuous. OpenMP and Threads are both host-parallel and a build
// normally enables one, so listing both costs nothing where only one exists.
#if defined(KOKKOS_ENABLE_THREADS)
    Kokkos::Threads,
#endif
    Kokkos::Serial>;

/// One definition, every enabled space. Device coverage is then a build option,
/// not a second test file that can drift from the first.
#define FUSE_TYPED_TEST_ON_SPACES(fixture)                                               \
  TYPED_TEST_SUITE(fixture, ::fuse::testing::EnabledExecutionSpaces)

/// Reduces the per-test failure status across the communicator in OnTestEnd, so no
/// rank can leave a test believing it succeeded while a peer failed. Installed by
/// fuse_test_main; nothing else should install it.
class MpiFailureReducer : public ::testing::EmptyTestEventListener {
 public:
  void OnTestEnd(const ::testing::TestInfo& info) override;
};

}  // namespace fuse::testing
