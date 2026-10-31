# FUSE -- top-level superbuild.
#
# This builds the imported parent trees and the fuse:: layers; each layer adds
# its own directory here.
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

# Floating-point contraction policy: included here, before FuseFortran,
# for the identical ordering reason the comment above gives for
# FUSE_ENABLE_FORTRAN -- FuseFortran.cmake calls
# fuse_apply_fp_contract_policy_fortran() (cmake/FuseFpPolicy.cmake) right
# after it enables the Fortran language, which needs the
# FUSE_FP_REPRODUCIBLE option and both functions already defined.
include(FuseFpPolicy)
include(FuseSlowTests)
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
# their own ctest registrations land in this project's test list: the condition
# is those suites passing *here*, not passing somewhere else.
# upstream/soma is no longer built here: UPSTREAMS.lock's [soma].retired field
# records the detachment. The subtree itself stays present and unmodified (I8)
# until it is removed -- see docs/dev/soma-import.md for what from it fuse:: still needs.

if(FUSE_BUILD_UPSTREAM_KARMA)
  # Needs tier 2+: MOAB and PETSc. Off by default so the fast path stays fast.
  add_subdirectory(upstream/karma upstream-karma)
endif()

# --- the library ------------------------------------------------------------
include(FuseDependencies)
fuse_resolve_dependencies()

# Floating-point contraction policy: AFTER fuse_resolve_dependencies(),
# once every FetchContent_MakeAvailable call it makes (r3d, yaml-cpp on a
# machine with no installed copy) has already run, so the contraction
# flag reaches fuse's own src/test/examples/tools targets below without
# reaching into vendored third-party sources (cmake/FuseFpPolicy.cmake's
# own comment explains why that exclusion is deliberate).
fuse_apply_fp_contract_policy_c_cxx()
add_subdirectory(src)
if(FUSE_ENABLE_TESTS)
  add_subdirectory(test)

  # Regenerates a frozen golden file. A tool, not a test: golden data is never
  # rewritten by the suite that checks it (tools/src/write_golden_descriptor.cpp).
  add_executable(fuse_write_golden_descriptor tools/src/write_golden_descriptor.cpp)
  target_link_libraries(fuse_write_golden_descriptor PRIVATE fuse::model)
endif()

# physics-kit (docs/dev/physics-kit.md), the optional finite-element
# component -- a separate component, not a fuse:: layer (docs/dev/
# layering.md), so it is added here rather than from inside src/
# CMakeLists.txt's layer list. AFTER the FUSE_ENABLE_TESTS block just
# above, and for the SAME reason the comment below gives for examples/:
# physics-kit/test/ registers its own gtest through test/CMakeLists.txt's
# fuse_add_gtest (test/CMakeLists.txt's own function, called from outside
# its directory tree -- see physics-kit/test/CMakeLists.txt's comment),
# which must already be defined, and its add_test() needs the
# CTestTestfile.cmake machinery include(CTest) above already started.
if(FUSE_ENABLE_PHYSICS_KIT)
  add_subdirectory(physics-kit)
endif()

# AFTER enable_testing() and after test/, because the examples register ctest
# entries of their own: added earlier, no CTestTestfile.cmake is generated for
# that directory and the examples would build but never run. The parent
# recorded that trap and it applies here unchanged.
add_subdirectory(examples)

# pfuse (docs/dev/python-surface.md). FUSE_ENABLE_PYTHON is declared OFF by default in
# cmake/FuseDependencies.cmake; AFTER test/ and examples/ for the same ctest-registration
# reason as the comment above.
if(FUSE_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

# The banner used to name the stage and the layer list as literal strings and
# drifted as layers were added: read the layer
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
message(STATUS "  upstream/karma  : ${FUSE_BUILD_UPSTREAM_KARMA}")
message(STATUS "  fuse:: layers   : ${_fuse_layers_text}")
message(STATUS "  linalg backend  : ${FUSE_LINALG_BACKEND}")
message(STATUS "")
