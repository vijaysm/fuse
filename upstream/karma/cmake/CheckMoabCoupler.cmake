include(CheckCXXSourceCompiles)

function(karma_check_moab_coupler)
  set(KARMA_HAVE_MOAB_COUPLER OFF PARENT_SCOPE)
  if(NOT KARMA_ENABLE_MOAB_COUPLER OR NOT KARMA_HAVE_MOAB)
    return()
  endif()

  find_path(KARMA_MOAB_COUPLER_INCLUDE_DIR
    NAMES mbcoupler/Coupler.hpp
    HINTS ${MOAB_PREFIX} ${PETSC_PREFIX}
    PATH_SUFFIXES include)
  find_library(KARMA_MOAB_COUPLER_LIBRARY
    NAMES mbcoupler
    HINTS ${MOAB_PREFIX} ${PETSC_PREFIX}
    PATH_SUFFIXES lib)
  if(NOT KARMA_MOAB_COUPLER_INCLUDE_DIR OR NOT KARMA_MOAB_COUPLER_LIBRARY)
    message(STATUS "MOAB Coupler: header or library not found")
    return()
  endif()

  set(CMAKE_REQUIRED_INCLUDES ${KARMA_MOAB_COUPLER_INCLUDE_DIR} ${MOAB_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${KARMA_MOAB_COUPLER_LIBRARY} PkgConfig::MOAB MPI::MPI_CXX)
  unset(KARMA_MOAB_COUPLER_WORKS CACHE)
  check_cxx_source_compiles("#include <mpi.h>
#include <mbcoupler/Coupler.hpp>
#include <moab/Core.hpp>
#include <moab/ParallelComm.hpp>
#include <moab/Range.hpp>
int main() {
  moab::Core core;
  moab::Range elements;
  moab::ParallelComm pcomm(&core, MPI_COMM_WORLD);
  moab::Coupler coupler(&core, &pcomm, elements, 0, false);
  return 0;
}"
    KARMA_MOAB_COUPLER_WORKS)
  unset(CMAKE_REQUIRED_INCLUDES)
  unset(CMAKE_REQUIRED_LIBRARIES)

  if(KARMA_MOAB_COUPLER_WORKS)
    set(KARMA_HAVE_MOAB_COUPLER ON PARENT_SCOPE)
    set(KARMA_MOAB_COUPLER_LIBRARY ${KARMA_MOAB_COUPLER_LIBRARY} PARENT_SCOPE)
    set(KARMA_MOAB_COUPLER_INCLUDE_DIR ${KARMA_MOAB_COUPLER_INCLUDE_DIR} PARENT_SCOPE)
  else()
    message(STATUS "MOAB Coupler: compile/link probe failed")
  endif()
endfunction()
