// fuse::petsc::ensure_initialized()/finalize_if_owned() -- the ONE PETSc
// ownership branch this sandbox can reach (fuse/petsc/token.hpp's own
// header comment on why only one): nothing in this tier-0 process (no
// FUSE_ENABLE_MPI harness) has called MPI_Init() before this runs.
//
// THIS IS ITS OWN, SEPARATE TEST BINARY (test/CMakeLists.txt's own
// `fuse_add_gtest(petsc init ...)` row -- a distinct ctest target, hence a
// distinct OS process, from petsc/token_test.cpp, petsc/
// derivative_gate_test.cpp and petsc/snes_test.cpp). That separation is
// load-bearing, not incidental: MPI_Finalize() is irreversible within a
// process (the MPI standard forbids MPI_Init() after it), so
// finalize_if_owned()'s own real effect can be exercised here, in a
// process that does nothing PETSc-related afterward, without stranding
// any OTHER PETSc test that would otherwise share the process. Every
// other petsc/*_test.cpp calls ensure_initialized() (idempotently, safe
// from any process) and NEVER calls finalize_if_owned() at all, relying
// on process exit -- this file is the only place the full
// init-then-finalize cycle actually runs.

#include "fuse/petsc/token.hpp"

#include <mpi.h>

#include <gtest/gtest.h>

using fuse::petsc::comm_self_f;
using fuse::petsc::ensure_initialized;
using fuse::petsc::finalize_if_owned;
using fuse::petsc::owns_petsc_initialization;

TEST(PetscInit, OwnsAndReleasesMpiWhenNothingElseStartedIt) {
  int mpi_before = 0;
  ASSERT_EQ(MPI_Initialized(&mpi_before), MPI_SUCCESS);
  if (mpi_before != 0) {
    // An MPI build: the test harness (test/support/fuse_test_main.cpp)
    // has already called MPI_Init, so this case's premise -- nothing has
    // -- is false here, and the branch it pins is the one the sibling
    // case below covers. Measured on the host 2026-10-11 in the
    // FUSE_ENABLE_MPI=ON tree: MPI_Initialized reported 1 at this line.
    GTEST_SKIP() << "MPI already initialised by the harness; see "
                    "InitialisesOnTheCallersCommunicatorWhenMpiIsAlreadyUpAndRefusesAfterFinalize";
  }
  ASSERT_EQ(mpi_before, 0) << "this test's own premise -- a tier-0 process where nothing has "
                              "called MPI_Init() yet -- does not hold; it must run in its own "
                              "process (test/CMakeLists.txt's own separate ctest row for this "
                              "file)";

  ASSERT_FALSE(owns_petsc_initialization())
      << "owns_petsc_initialization() must read false before ensure_initialized() ever ran";

  fuse::Result<void> r = ensure_initialized(comm_self_f());
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;

  EXPECT_TRUE(owns_petsc_initialization())
      << "ensure_initialized() was the first call in this process; it must record ownership";

  int mpi_after_init = 0;
  ASSERT_EQ(MPI_Initialized(&mpi_after_init), MPI_SUCCESS);
  EXPECT_NE(mpi_after_init, 0)
      << "MEASURED: PetscInitialize() must have called MPI_Init() itself, since nothing else "
         "in this process did";

  // Idempotence: a second call, from "elsewhere" in the same process, must
  // be a safe no-op that does not re-claim ownership or re-initialise.
  fuse::Result<void> r2 = ensure_initialized(comm_self_f());
  ASSERT_TRUE(r2.ok()) << r2.error().code << ": " << r2.error().message;
  EXPECT_TRUE(owns_petsc_initialization());

  finalize_if_owned();
  EXPECT_FALSE(owns_petsc_initialization())
      << "finalize_if_owned() must clear the ownership flag once it actually finalises";

  int mpi_after_finalize = 0;
  ASSERT_EQ(MPI_Finalized(&mpi_after_finalize), MPI_SUCCESS);
  EXPECT_NE(mpi_after_finalize, 0)
      << "MEASURED: PetscFinalize() must have called MPI_Finalize() itself, matching PETSc's "
         "own internal tracking of having called MPI_Init() earlier in this same process";

  // A second finalize_if_owned() (the flag is already clear) must be a
  // harmless no-op, never a second MPI_Finalize() call (illegal under the
  // MPI standard).
  finalize_if_owned();
}

// The other branch of ensure_initialized(): MPI is already up (an MPI
// build's harness, or a participant program), so PETSc must initialise on
// the caller's communicator WITHOUT touching MPI's lifetime -- in
// particular finalize_if_owned() finalises PETSc but leaves MPI running,
// since PetscFinalize only calls MPI_Finalize when PetscInitialize was
// the one that called MPI_Init (PETSc's own rule, relied on here).
TEST(PetscInit, InitialisesOnTheCallersCommunicatorWhenMpiIsAlreadyUpAndRefusesAfterFinalize) {
  int mpi_before = 0;
  ASSERT_EQ(MPI_Initialized(&mpi_before), MPI_SUCCESS);
  if (mpi_before == 0) {
    GTEST_SKIP() << "no MPI in this process; the branch is covered by "
                    "OwnsAndReleasesMpiWhenNothingElseStartedIt";
  }
  int mpi_finalized_before = 0;
  ASSERT_EQ(MPI_Finalized(&mpi_finalized_before), MPI_SUCCESS);
  if (mpi_finalized_before != 0) {
    // The tier-0 binary reaches here AFTER the previous case released the
    // MPI this library started: MPI_Initialized still reads 1 (the
    // standard keeps it so after MPI_Finalize), and the first draft of
    // this case walked into a finalised MPICH and aborted (host,
    // 2026-10-11). The property here is the refusal, by name.
    fuse::Result<void> refused = ensure_initialized(comm_self_f());
    ASSERT_FALSE(refused.ok());
    EXPECT_EQ(refused.error().code, "E-PETSC-MPI-FINALIZED") << refused.error().message;
    return;
  }
  fuse::Result<void> r = ensure_initialized(comm_self_f());
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  EXPECT_TRUE(owns_petsc_initialization());

  finalize_if_owned();
  EXPECT_FALSE(owns_petsc_initialization());
  int mpi_finalized = 1;
  ASSERT_EQ(MPI_Finalized(&mpi_finalized), MPI_SUCCESS);
  EXPECT_EQ(mpi_finalized, 0) << "finalize_if_owned() must never finalise an MPI it did not start";
  int mpi_still_up = 0;
  ASSERT_EQ(MPI_Initialized(&mpi_still_up), MPI_SUCCESS);
  EXPECT_NE(mpi_still_up, 0);
}

