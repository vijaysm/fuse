# FUSE — top-level superbuild.
#
# At S0 this builds the two imported parent trees and nothing else: there is no
# fuse/ source yet, and a target that compiles nothing is worse than no target.
# Each stage from S1 adds its own layer directory here.
#
# The imported trees keep their OWN build systems and are configured as
# subprojects, unmodified. That is invariant I8: an import that had to be
# patched to build is not an import, and its suite would no longer be evidence
# about the code we took.

cmake_minimum_required(VERSION 3.24)
project(fuse LANGUAGES C CXX VERSION 0.1.0)

# fuse-abi-shared (src/CMakeLists.txt) links fuse::runtime and, through it,
# every other fuse-coupler-* static archive in this tree -- the only one of
# those archives built -fPIC on its own is fuse-abi-objects, an OBJECT
# library whose own POSITION_INDEPENDENT_CODE was set because its objects
# were always known to end up in a .so. Without this, ld only discovers the
# gap at the final link: a non-PIC archive compiled this way can still
# define an ordinary global, but a `thread_local` one (first hit: Kokkos'
# SharedAllocationRecord::t_tracking_enabled, pulled in through
# runner.cpp.o) gets the initial-exec TLS model, and `R_X86_64_TPOFF32
# against hidden symbol ... cannot be used when making a shared object`
# fails the link. Global, not per-target, because the whole static closure
# feeds that one shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(FUSE_BUILD_UPSTREAM_SOMA  "Build the imported coupler tree under upstream/soma"  ON)
option(FUSE_BUILD_UPSTREAM_KARMA "Build the imported spine tree under upstream/karma"   OFF)

# Declared here, BEFORE include(FuseDependencies) below, so this option()
# call is the one that seeds the FUSE_ENABLE_FORTRAN cache entry: a cache
# variable's initial value comes from whichever option() call reaches it
# first in a fresh configure, and cmake/FuseDependencies.cmake also declares
# FUSE_ENABLE_FORTRAN (unused there, default OFF, predating the Fortran
# binding below) -- without this ordering that later call would win and
# ffuse would never default on. cmake/FuseFortran.cmake turns this back OFF,
# with a message rather than a configure error, when no Fortran compiler is
# found.
option(FUSE_ENABLE_FORTRAN "Build the Fortran participant binding ffuse (needs a Fortran compiler)" ON)
include(FuseFortran)

include(CTest)

# The imported coupler tree fetches yaml-cpp 0.8.0, whose CMakeLists declares
# `cmake_minimum_required(VERSION 3.4)`. CMake 4.x removed compatibility with
# anything below 3.5, so configuring fails unless a floor is supplied. We set it
# here rather than patching the import (invariant I8). A yaml-cpp already
# installed on the system is found first and avoids the fetch entirely.
if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM)
  set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING
      "Floor for imported third-party CMake projects that predate CMake 3.5")
endif()

# --- imported parents ------------------------------------------------------
# Namespaced build directories and an EXCLUDE_FROM_ALL-free add_subdirectory so
# their own ctest registrations land in this project's test list: the S0 exit
# condition is those suites passing *here*, not passing somewhere else.
if(FUSE_BUILD_UPSTREAM_SOMA)
  add_subdirectory(upstream/soma upstream-soma)
endif()

if(FUSE_BUILD_UPSTREAM_KARMA)
  # Needs tier 2+: MOAB and PETSc. Off by default so the fast path stays fast.
  add_subdirectory(upstream/karma upstream-karma)
endif()

# --- the library ------------------------------------------------------------
include(FuseDependencies)
fuse_resolve_dependencies()
add_subdirectory(src)
if(FUSE_ENABLE_TESTS)
  add_subdirectory(test)

  # Regenerates a frozen golden file. A tool, not a test: golden data is never
  # rewritten by the suite that checks it (tools/src/write_golden_descriptor.cpp).
  add_executable(fuse_write_golden_descriptor tools/src/write_golden_descriptor.cpp)
  target_link_libraries(fuse_write_golden_descriptor PRIVATE fuse::model)
endif()

# AFTER enable_testing() and after test/, because the examples register ctest
# entries of their own: added earlier, no CTestTestfile.cmake is generated for
# that directory and the examples would build but never run. The parent
# recorded that trap and it applies here unchanged.
add_subdirectory(examples)

# The banner used to name the stage and the layer list as literal strings and
# drifted for four stages and eight layers (REVIEW.md R-37): read the layer
# list back from src/CMakeLists.txt's own targets instead of restating it, so
# it cannot drift again. `fuse::` is the alias namespace every layer target
# registers itself under (src/CMakeLists.txt); this reads them back rather
# than hand-listing the layers a second time here.
#
# It used to loop over a hand-written candidate list, which is exactly the
# restatement this comment forbids: `partition` was added as a layer and the
# summary silently left it out. So the list now comes from the targets
# src/CMakeLists.txt actually defined, in definition order (which is layer
# order), keeping each `fuse-coupler-<layer>` library that has its `fuse::`
# alias.
set(_fuse_layers "")
get_property(_fuse_src_targets DIRECTORY "${PROJECT_SOURCE_DIR}/src"
             PROPERTY BUILDSYSTEM_TARGETS)
foreach(_fuse_target IN LISTS _fuse_src_targets)
  # Two separate ifs, not one: `${CMAKE_MATCH_1}` in an if() is expanded
  # BEFORE that if's MATCHES runs, so a combined condition tests the PREVIOUS
  # iteration's layer -- which dropped `base`, the first, from the summary.
  if(_fuse_target MATCHES "^fuse-coupler-(.+)$")
    set(_fuse_layer "${CMAKE_MATCH_1}")
    if(TARGET fuse::${_fuse_layer})
      list(APPEND _fuse_layers ${_fuse_layer})
    endif()
  endif()
endforeach()
list(JOIN _fuse_layers ", " _fuse_layers_text)

message(STATUS "")
message(STATUS "fuse ${PROJECT_VERSION}")
message(STATUS "  upstream/soma   : ${FUSE_BUILD_UPSTREAM_SOMA}")
message(STATUS "  upstream/karma  : ${FUSE_BUILD_UPSTREAM_KARMA}")
message(STATUS "  fuse:: layers   : ${_fuse_layers_text}")
message(STATUS "  linalg backend  : ${FUSE_LINALG_BACKEND}")
message(STATUS "")
