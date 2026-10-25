// src/petsc/init.cpp -- PetscInitialize/PetscFinalize ownership
// (fuse/petsc/token.hpp's own header comment has the full contract).
//
// In a tier-0 process (FUSE_ENABLE_MPI=OFF, so nothing in the harness has
// called MPI_Init()), PetscInitialize() succeeds and itself calls
// MPI_Init() -- see test/petsc/init_test.cpp for the exact
// MPI_Initialized()/MPI_Finalized() transitions this pins.

#include "fuse/petsc/token.hpp"

#include <petsc.h>

namespace fuse::petsc {

namespace {
bool g_owns_petsc = false;
}  // namespace

Result<void> ensure_initialized(int comm_f) {
  PetscBool already = PETSC_FALSE;
  PetscErrorCode ierr = PetscInitialized(&already);
  if (ierr) {
    return Failure{"E-PETSC-CALL",
                   "PetscInitialized failed with code " + std::to_string(ierr)};
  }
  if (already) {
    return Result<void>::success();
  }

  // Two cases, decided by MPI_Initialized, never assumed:
  //  - MPI is up (a tier-1 harness, or a participant program): the caller's
  //    communicator becomes PETSC_COMM_WORLD, and PetscInitialize sees MPI
  //    running and leaves it alone.
  //  - MPI is NOT up (a tier-0 test binary): the Fortran handle cannot be
  //    converted (MPI_Comm_f2c before MPI_Init is undefined) and setting
  //    PETSC_COMM_WORLD would make PetscInitialize SKIP its own MPI_Init
  //    and fail with PETSC_ERR_SUP (PetscInitialize returns 56 for exactly
  //    this reason). So PETSC_COMM_WORLD is left at its default and
  //    PetscInitialize calls MPI_Init itself; PetscFinalize then owns the
  //    matching MPI_Finalize, which is PETSc's own documented rule.
  int mpi_up = 0;
  if (MPI_Initialized(&mpi_up) != MPI_SUCCESS) {
    return Failure{"E-PETSC-CALL", "MPI_Initialized failed"};
  }
  // MPI_Initialized stays true after MPI_Finalize (MPI standard, 8.7), so
  // it alone cannot tell "up" from "over". A process whose MPI has been
  // finalised -- here, after finalize_if_owned() released an MPI this
  // library started -- cannot have PETSc again: PetscInitialize would call
  // into a finalised MPICH, which aborts ("Attempting to use an MPI routine
  // ... after finalizing MPICH", as the tier-0 init test's second case
  // shows). Refuse by name instead.
  int mpi_done = 0;
  if (MPI_Finalized(&mpi_done) != MPI_SUCCESS) {
    return Failure{"E-PETSC-CALL", "MPI_Finalized failed"};
  }
  if (mpi_done) {
    return Failure{"E-PETSC-MPI-FINALIZED",
                   "MPI has already been finalised in this process; PETSc cannot be "
                   "initialised again after finalize_if_owned() released the MPI it started"};
  }
  if (mpi_up) {
    PETSC_COMM_WORLD = MPI_Comm_f2c(static_cast<MPI_Fint>(comm_f));
  }
  int argc = 0;
  char** argv = nullptr;
  ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
  if (ierr) {
    return Failure{"E-PETSC-INIT",
                   "PetscInitialize failed with code " + std::to_string(ierr)};
  }
  g_owns_petsc = true;
  return Result<void>::success();
}

void finalize_if_owned() {
  if (!g_owns_petsc) {
    return;
  }
  // PetscFinalize() itself calls MPI_Finalize() only if PetscInitialize()
  // is the one that called MPI_Init() -- PETSc's own internal tracking
  // (fuse/petsc/token.hpp's own comment on ensure_initialized() has the
  // reasoning this file relies on rather than re-derives).
  PetscFinalize();
  g_owns_petsc = false;
}

bool owns_petsc_initialization() { return g_owns_petsc; }

int comm_self_f() { return MPI_Comm_c2f(MPI_COMM_SELF); }

}  // namespace fuse::petsc
