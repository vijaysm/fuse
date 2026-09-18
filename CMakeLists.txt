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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(FUSE_BUILD_UPSTREAM_SOMA  "Build the imported coupler tree under upstream/soma"  ON)
option(FUSE_BUILD_UPSTREAM_KARMA "Build the imported spine tree under upstream/karma"   OFF)

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

message(STATUS "")
message(STATUS "fuse ${PROJECT_VERSION} — stage S1")
message(STATUS "  upstream/soma   : ${FUSE_BUILD_UPSTREAM_SOMA}")
message(STATUS "  upstream/karma  : ${FUSE_BUILD_UPSTREAM_KARMA}")
message(STATUS "  fuse:: layers   : base")
message(STATUS "  linalg backend  : ${FUSE_LINALG_BACKEND}")
message(STATUS "")
