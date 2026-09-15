# Discovers MOAB and detects which optional capabilities it was actually built
# with, then defines the imported target soma::moab.
#
# Sets, in the scope of whatever include()s this file:
#   SOMA_HAVE_MOAB     - MOAB was found
#   SOMA_HAVE_TEMPEST  - that MOAB links TempestRemap (high-order/monotone remap)
#   SOMA_HAVE_IMOAB    - that MOAB installs the iMOAB C interface
#
# This file is meant to be include()d, not find_package()d. include() does not
# open a new variable scope, so the plain set() calls below land in the
# includer's scope. Using set(... PARENT_SCOPE) here would be wrong: from the
# top-level CMakeLists.txt there is no parent scope, so every SOMA_HAVE_* would
# silently vanish and the caller's "not found" branch would always fire.

find_package(PkgConfig REQUIRED)

# GLOBAL so soma::moab is usable from any subdirectory, not just this one.
pkg_check_modules(MOAB IMPORTED_TARGET GLOBAL MOAB)

if(NOT MOAB_FOUND)
  set(SOMA_HAVE_MOAB OFF)
  set(SOMA_HAVE_TEMPEST OFF)
  set(SOMA_HAVE_IMOAB OFF)
  message(STATUS "MOAB: not found; SOMA_ENABLE_MOAB features are unavailable")
  return()
endif()

set(SOMA_HAVE_MOAB ON)

# MOABConfig.h records the capabilities the library was actually built with, so
# it is the only trustworthy source: two MOAB installs of the same version can
# differ entirely in which optional backends are compiled in. Search only the
# include directories pkg-config reported (NO_DEFAULT_PATH) — this machine has
# several MOAB prefixes, and picking up a different one's MOABConfig.h would
# report capabilities the linked library does not have.
find_file(SOMA_MOAB_CONFIG_HEADER
          NAMES moab/MOABConfig.h
          PATHS ${MOAB_INCLUDE_DIRS}
          NO_DEFAULT_PATH)
if(NOT SOMA_MOAB_CONFIG_HEADER)
  message(FATAL_ERROR
    "MOAB: pkg-config found MOAB ${MOAB_VERSION} but moab/MOABConfig.h is not "
    "under its reported include dirs (${MOAB_INCLUDE_DIRS}). The install is "
    "incomplete; capability detection cannot proceed.")
endif()
file(READ "${SOMA_MOAB_CONFIG_HEADER}" _soma_moab_config)

# The autotools build emits `#define MOAB_HAVE_TEMPESTREMAP 1` and the CMake
# build emits `#define MOAB_HAVE_TEMPESTREMAP ON`; a MOAB without it emits
# nothing at all (not an #undef). Anchor at line start so a commented-out or
# /* #undef ... */ form cannot read as a positive.
#
# NOTE THE SPELLING: MOAB_HAVE_TEMPESTREMAP, not MOAB_HAVE_TEMPEST. The shorter
# name does not exist in any MOAB header and #ifdef-ing on it compiles the
# TempestRemap path out of every build in silence.
#
# This grep is a PRECONDITION, not the answer. `nm -gU libMOAB.dylib | grep -ci
# tempest` is worse than useless here -- it returns 1 on an install with no
# TempestRemap at all -- and the header can advertise a library that is not
# linkable from this prefix (TempestRemap is a separate .a/.dylib and its own
# headers need netcdf.h on the include path). The real answer is below, after
# the imported target exists: a compile AND link of the two entry points
# src/remap/tempest_weights.cpp actually calls.
if(_soma_moab_config MATCHES "(^|\n)[ \t]*#[ \t]*define[ \t]+MOAB_HAVE_TEMPESTREMAP[ \t]")
  set(_soma_moab_config_claims_tempest ON)
else()
  set(_soma_moab_config_claims_tempest OFF)
endif()

if(NOT _soma_moab_config MATCHES "(^|\n)[ \t]*#[ \t]*define[ \t]+MOAB_HAVE_MPI[ \t]")
  message(WARNING "MOAB: built without MPI; parallel coupling is unavailable")
endif()

find_file(SOMA_IMOAB_HEADER
          NAMES moab/iMOAB.h
          PATHS ${MOAB_INCLUDE_DIRS}
          NO_DEFAULT_PATH)
if(SOMA_IMOAB_HEADER)
  set(SOMA_HAVE_IMOAB ON)
else()
  set(SOMA_HAVE_IMOAB OFF)
endif()

# MOAB.pc ships the flags MOAB itself was compiled with in Cflags: on this
# machine that includes `-O2 -DNDEBUG`. pkg_check_modules copies those onto
# PkgConfig::MOAB's INTERFACE_COMPILE_OPTIONS, which would then apply -DNDEBUG
# to SOMA's own sources in a Debug build and compile out every assert() —
# precisely the Debug/Release divergence docs/BUILDING.md exists to prevent.
# Consumers need MOAB's headers and libraries, not its build flags, so drop the
# compile options and keep the include dirs and link interface.
set_target_properties(PkgConfig::MOAB PROPERTIES INTERFACE_COMPILE_OPTIONS "")

add_library(soma::moab ALIAS PkgConfig::MOAB)

# ---------------------------------------------------------------------------
# TempestRemap: compile AND link the entry points, do not take anyone's word
# ---------------------------------------------------------------------------
#
# The two functions below are exactly the ones src/remap/tempest_weights.cpp
# calls. A prefix that has the headers but not the library, or the library but
# not netcdf.h on the include path, fails here at configure time with a legible
# message instead of at link time in one translation unit.
#
# Deliberately NOT moab/Remapping/TempestRemapper.hpp: that header pulls in
# moab/ParallelComm.hpp on an MPI-enabled MOAB and so needs mpi.h, which MOAB's
# pkg-config Cflags do not provide. TempestRemap's own API needs no MPI.
set(SOMA_HAVE_TEMPEST OFF)
if(_soma_moab_config_claims_tempest)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LIBRARIES PkgConfig::MOAB)
  set(CMAKE_REQUIRED_QUIET ON)
  check_cxx_source_compiles("
#include <GridElements.h>
#include <OfflineMap.h>
#include <TempestRemapAPI.h>
int main() {
  Mesh a, b, o;
  OfflineMap m;
  GenerateOfflineMapAlgorithmOptions opts;
  int rc = GenerateOverlapWithMeshes(a, b, o, \"\", \"Netcdf4\", \"exact\",
                                     false, false, true, false);
  rc += GenerateOfflineMapWithMeshes(a, b, o, \"fv\", \"fv\", opts, m);
  return rc;
}
" SOMA_TEMPEST_LINKS)
  unset(CMAKE_REQUIRED_LIBRARIES)
  unset(CMAKE_REQUIRED_QUIET)
  if(SOMA_TEMPEST_LINKS)
    set(SOMA_HAVE_TEMPEST ON)
  else()
    message(WARNING
      "MOAB: MOABConfig.h defines MOAB_HAVE_TEMPESTREMAP, but a program that "
      "calls GenerateOverlapWithMeshes/GenerateOfflineMapWithMeshes does not "
      "compile and link against this prefix. High-order and monotone remap "
      "will report E-REMAP-NOTEMPEST. See "
      "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeConfigureLog.yaml for the "
      "compiler output.")
  endif()
endif()

if(NOT SOMA_HAVE_TEMPEST)
  message(STATUS
    "MOAB: no usable TempestRemap. Conservative first-order FV<->FV remap via "
    "Intx2Mesh is available; high-order and monotone methods are not. "
    "See docs/BUILDING.md to rebuild with --download-tempestremap.")
endif()
