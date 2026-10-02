# Optional Fortran language enablement for the participant binding ffuse
# (docs/dev/abi-design.md, "Fortran: ffuse"; docs/user/fortran.md).
#
# FUSE_ENABLE_FORTRAN is declared by the top-level CMakeLists.txt, ON by
# default, BEFORE this file is included: a CMake cache variable takes its
# initial value from whichever option() call reaches it first in a fresh
# configure, and this module must run after that call to have anything to
# read. This file does not redeclare the option; it only acts on it.
#
# A compiler is probed with check_language() rather than assumed: a machine
# with no Fortran compiler at all must still configure cleanly with the
# default ON, producing every OTHER target (the C participant ABI, the rest
# of the library and its tests) with FUSE_ENABLE_FORTRAN silently false
# rather than a configure-time failure over an optional binding. An explicit
# -DFUSE_ENABLE_FORTRAN=OFF short-circuits the probe entirely.
if(FUSE_ENABLE_FORTRAN)
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/fortran" "${CMAKE_CURRENT_BINARY_DIR}/fortran")
  else()
    message(STATUS "fuse: no Fortran compiler found; FUSE_ENABLE_FORTRAN was ON but ffuse "
                   "(fortran/, test/abi/fortran/) will not be built")
    set(FUSE_ENABLE_FORTRAN OFF CACHE BOOL
        "Build the Fortran participant binding ffuse (needs a Fortran compiler)" FORCE)
  endif()
endif()
