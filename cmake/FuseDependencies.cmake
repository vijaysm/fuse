# FUSE dependency resolution, one tier at a time (docs/PLAN.md §3.1).
#
# Policy, decided 2026-09-15: USE WHAT IS INSTALLED. Every dependency is resolved
# by find_package against the prefixes in CMAKE_PREFIX_PATH, and a missing one is a
# hard error naming what to install -- never a silent download. The machine's
# prefixes are tuned for it (Kokkos at /opt/kokkos is built for this CPU); a fetched
# generic build would quietly replace that and any timing taken from it would be a
# measurement of the wrong library.
#
# FUSE_ALLOW_FETCH=ON opts into building a pinned copy of whatever is missing, for a
# machine that has no such prefix. Pins are exact so that leg stays reproducible; a
# pin bump is its own commit. `fuse-resolved-dependencies.txt` in the build tree
# records, per dependency, whether it was FOUND or FETCHED.
#
# Tiers 2 and 3 (MOAB, PETSc) are found only, never fetched: they are large builds
# with their own configuration choices, and a fetched default would differ from the
# installed one in ways that matter (index size, debug/optimized, which MPI).
# Extending the fetch machinery to them is a later decision, deliberately open.

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
option(FUSE_ALLOW_FETCH "Build a pinned copy of any Tier 0 dependency that is not installed" OFF)

# ---------------------------------------------------------------- Tier 0 (always)
set(FUSE_FMT_TAG        "11.0.2")
set(FUSE_FMTLOG_TAG     "v2.2.1")
set(FUSE_YAMLCPP_TAG     "0.8.0")
set(FUSE_GTEST_TAG       "v1.15.2")   # git tag; the find_package version below is the same number without the v
set(FUSE_GTEST_VERSION   "1.15.2")
set(FUSE_KOKKOS_TAG     "4.4.01")
set(FUSE_KOKKOSKERNELS_TAG "4.4.01")   # must match Kokkos
set(FUSE_EIGEN_TAG      "3.4.0")

# ------------------------------------------------- options
# Declared BEFORE any FetchContent_Declare that consults them: on a fresh cache
# an option read ahead of its declaration is empty, so a declaration guarded by
# it is silently skipped and resolution later fails with 'no content details'.
option(FUSE_ENABLE_MPI     "Tier 1: MPI transport, CommGraph, epochs across ranks" OFF)
option(FUSE_ENABLE_MOAB    "Tier 2: MOAB mesh backend, Intx2Mesh, TempestRemap"    OFF)
option(FUSE_ENABLE_PETSC   "Tier 3: PETSc solver plugin and physics-kit"           OFF)
option(FUSE_ENABLE_LOWFIVE "HDF5-VOL interception transport"                        OFF)
option(FUSE_ENABLE_PYTHON  "pybind11 module"                                        OFF)
option(FUSE_ENABLE_FORTRAN "Fortran participant shim"                               OFF)
option(FUSE_ENABLE_TESTS   "Build the test suite"                                   ON)
# OFF turns every FUSE_THROW/ASSERT/REFUSE into report-then-abort after the same
# on-error policy has run; the types and call sites are unconditional either way.
option(FUSE_ENABLE_EXCEPTIONS "Throw C++ exceptions on errors (OFF = report and abort)" ON)

# The linalg facade has two interchangeable backends; the hand-written QR inherited
# from a parent stays as the test oracle, not as a third backend.
set(FUSE_LINALG_BACKEND "eigen" CACHE STRING "kokkoskernels | eigen")
set_property(CACHE FUSE_LINALG_BACKEND PROPERTY STRINGS kokkoskernels eigen)

# A tier is a prefix: enabling a higher one without its predecessors is a
# configuration error, never a silent downgrade (docs/CHARTER.md §4).
if(FUSE_ENABLE_MOAB AND NOT FUSE_ENABLE_MPI)
  message(FATAL_ERROR "FUSE_ENABLE_MOAB (tier 2) requires FUSE_ENABLE_MPI (tier 1)")
endif()
if(FUSE_ENABLE_PETSC AND NOT FUSE_ENABLE_MOAB)
  message(FATAL_ERROR "FUSE_ENABLE_PETSC (tier 3) requires FUSE_ENABLE_MOAB (tier 2)")
endif()

# fmt — the ONLY output formatter in the library (docs/CHARTER.md). No iostreams in
# library code, no printf in host code; fmt::format_to with a caller-provided buffer
# on any path that must not allocate.
# ------------------------------------------------- declarations for the fetch path
# Consulted only when FUSE_ALLOW_FETCH is ON and find_package came up empty.
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG ${FUSE_FMT_TAG} GIT_SHALLOW ON)
FetchContent_Declare(fmtlog
  GIT_REPOSITORY https://github.com/MengRao/fmtlog.git GIT_TAG ${FUSE_FMTLOG_TAG}
  GIT_SUBMODULES "" GIT_SHALLOW ON)   # its submodules are a vendored fmt and two benchmark loggers
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG ${FUSE_GTEST_TAG} GIT_SHALLOW ON)
FetchContent_Declare(Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git GIT_TAG ${FUSE_KOKKOS_TAG} GIT_SHALLOW ON)
FetchContent_Declare(KokkosKernels
  GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git GIT_TAG ${FUSE_KOKKOSKERNELS_TAG} GIT_SHALLOW ON)
# yaml-cpp is the ONE dependency fetched by default rather than required from a
# prefix. The installed-first rule exists because Kokkos, MOAB and PETSc carry
# machine and stack tuning that a generic build would silently replace; a YAML
# parser carries none of that -- it has no MPI, no index-size choice, no tuned
# kernels -- so building a pinned copy costs nothing and makes the front-end
# available on a machine that has no packaged one. An installed yaml-cpp is still
# found first.
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git GIT_TAG ${FUSE_YAMLCPP_TAG}
  GIT_SHALLOW ON
  FIND_PACKAGE_ARGS ${FUSE_YAMLCPP_TAG} NAMES yaml-cpp)

FetchContent_Declare(Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG ${FUSE_EIGEN_TAG} GIT_SHALLOW ON)

# ------------------------------------------------- resolution
# The per-dependency record is APPENDED TO A FILE as each one resolves, rather
# than accumulated in a variable: a cache STRING cannot hold newlines (CMake
# truncates it with a warning), and a plain list set inside a function does not
# survive the function's scope. Both were tried; the file is what works.
set(FUSE_RESOLUTION_FILE "${CMAKE_BINARY_DIR}/fuse-resolved-dependencies.txt")
file(WRITE "${FUSE_RESOLUTION_FILE}" "")

# fuse_require(<package> [VERSION v] [NAMES ...] [FETCH_AS <content-name>] [HINT <text>])
#
# find_package first. If it is missing: fetch it when FUSE_ALLOW_FETCH is ON,
# otherwise fail with the hint, because guessing is how a build silently stops
# measuring the library the machine was tuned with.
function(fuse_require package)
  cmake_parse_arguments(ARG "" "VERSION;FETCH_AS;HINT" "NAMES" ${ARGN})
  set(_names ${ARG_NAMES})
  if(NOT _names)
    set(_names ${package})
  endif()
  find_package(${package} ${ARG_VERSION} QUIET CONFIG NAMES ${_names})
  # find_package inside a function sets its result variables in the FUNCTION scope.
  # The imported targets it creates are directory-scoped and so survive, but
  # <pkg>_VERSION and <pkg>_DIR do not -- which silently emptied the Kokkos version
  # in the generated config.hpp and made the header/library test compare against "".
  set(${package}_FOUND   "${${package}_FOUND}"   PARENT_SCOPE)
  set(${package}_VERSION "${${package}_VERSION}" PARENT_SCOPE)
  set(${package}_DIR     "${${package}_DIR}"     PARENT_SCOPE)
  if(${package}_FOUND)
    file(APPEND "${FUSE_RESOLUTION_FILE}"
         "${package}=found ${${package}_VERSION} ${${package}_DIR}\n")
    message(STATUS "fuse: ${package} ${${package}_VERSION} found at ${${package}_DIR}")
    return()
  endif()
  if(NOT FUSE_ALLOW_FETCH)
    message(FATAL_ERROR
      "fuse: ${package} not found.\n"
      "  ${ARG_HINT}\n"
      "  Point CMAKE_PREFIX_PATH at its prefix, or configure with -DFUSE_ALLOW_FETCH=ON to "
      "build a pinned copy (which will NOT be tuned for this machine).")
  endif()
  set(_content ${package})
  if(ARG_FETCH_AS)
    set(_content ${ARG_FETCH_AS})
  endif()
  message(STATUS "fuse: ${package} not installed; building the pinned copy (FUSE_ALLOW_FETCH=ON)")
  FetchContent_MakeAvailable(${_content})
  file(APPEND "${FUSE_RESOLUTION_FILE}" "${package}=fetched (pinned tag, not tuned for this machine)\n")
endfunction()

# Strip include directories that do not exist from an imported target.
#
# /opt/kokkos's KokkosTargets.cmake hard-codes the SDK path it was built against
# (MacOSX15.4.sdk); Xcode has since moved on and that directory is gone, which CMake
# reports as a hard error on any target that links Kokkos. The prefix is otherwise
# correct and is the machine-tuned build we want, so the stale entry is dropped here
# rather than by editing someone else's install. Only NON-EXISTENT paths are removed,
# so this cannot mask a real include directory.
function(fuse_drop_missing_include_dirs)
  foreach(target ${ARGN})
    if(NOT TARGET ${target})
      continue()
    endif()
    get_target_property(_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT _dirs)
      continue()
    endif()
    set(_kept "")
    foreach(dir ${_dirs})
      if(EXISTS "${dir}" OR dir MATCHES "^\\$<")
        list(APPEND _kept "${dir}")
      else()
        message(STATUS "fuse: dropping stale include dir from ${target}: ${dir}")
      endif()
    endforeach()
    set_target_properties(${target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_kept}")
  endforeach()
endfunction()

# Resolved separately from the tier ladder: it is the one always-fetchable
# dependency, and only the front-end links it (PRIVATE, so it never reaches a
# consumer's include path).
macro(fuse_resolve_yaml)
  find_package(yaml-cpp ${FUSE_YAMLCPP_TAG} QUIET CONFIG)
  if(yaml-cpp_FOUND)
    file(APPEND "${FUSE_RESOLUTION_FILE}" "yaml-cpp=found ${yaml-cpp_VERSION} ${yaml-cpp_DIR}\n")
    message(STATUS "fuse: yaml-cpp ${yaml-cpp_VERSION} found at ${yaml-cpp_DIR}")
  else()
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml-cpp)
    file(APPEND "${FUSE_RESOLUTION_FILE}" "yaml-cpp=fetched ${FUSE_YAMLCPP_TAG}\n")
    message(STATUS "fuse: yaml-cpp ${FUSE_YAMLCPP_TAG} built from source (pinned)")
  endif()
endmacro()

macro(fuse_resolve_dependencies)
  fuse_require(fmt HINT "fmt is the only output formatter (docs/dev/conventions.md).")
  fuse_require(Kokkos HINT "Kokkos is required at every tier; /opt/kokkos on this machine.")
  # Sanitized before anything links it. The offender on this machine is
  # Kokkos::LIBDL, whose only usage requirement is the SDK include path Kokkos was
  # configured against; the rest are listed so a different install's stale entry is
  # caught the same way.
  fuse_resolve_yaml()

  fuse_drop_missing_include_dirs(
    Kokkos::kokkos Kokkos::kokkoscore Kokkos::kokkoscontainers Kokkos::kokkossimd
    Kokkos::LIBDL Kokkos::THREADS Kokkos::SERIAL Kokkos::OPENMP
    Kokkos::HWLOC Kokkos::LIBRT Kokkos::LIBQUADMATH Kokkos::CUDA)

  # This machine has a second Kokkos in /opt/homebrew/include, and any target that
  # puts a homebrew prefix on the command line ahead of Kokkos's own include dir
  # picks up ITS Kokkos_Core.hpp -- which declares an OpenMP backend the chosen
  # install does not have. Relying on link order to prevent that is fragile, so the
  # chosen include dir is captured here and prepended (BEFORE) on the base target;
  # Config.KokkosHeadersMatchTheResolvedLibrary is the backstop if it ever slips.
  get_target_property(FUSE_KOKKOS_INCLUDE_DIRS Kokkos::kokkoscore INTERFACE_INCLUDE_DIRECTORIES)
  set(FUSE_KOKKOS_INCLUDE_DIRS "${FUSE_KOKKOS_INCLUDE_DIRS}" CACHE INTERNAL "chosen Kokkos include dirs")

  if(FUSE_LINALG_BACKEND STREQUAL "kokkoskernels")
    fuse_require(KokkosKernels HINT "Set FUSE_LINALG_BACKEND=eigen to use Eigen instead.")
    fuse_drop_missing_include_dirs(Kokkos::kokkoskernels)
  elseif(FUSE_LINALG_BACKEND STREQUAL "eigen")
    # No version constraint: Eigen3Config is SameMajorVersion, so asking for 3.4
    # rejects an installed 5.x outright. The facade uses dense QR, small
    # least-squares and sparse products, whose API is identical across both.
    fuse_require(Eigen3 NAMES Eigen3 HINT "Set FUSE_LINALG_BACKEND=kokkoskernels to use KokkosKernels instead.")
  else()
    message(FATAL_ERROR "FUSE_LINALG_BACKEND must be kokkoskernels or eigen, got '${FUSE_LINALG_BACKEND}'")
  endif()

  if(FUSE_ENABLE_TESTS)
    fuse_require(GTest VERSION ${FUSE_GTEST_VERSION} NAMES GTest FETCH_AS googletest
                 HINT "GoogleTest is the test harness; -DFUSE_ENABLE_TESTS=OFF skips it.")
  endif()

  if(FUSE_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX)
  endif()
  if(FUSE_ENABLE_MOAB)
    find_package(MOAB REQUIRED)     # found, never fetched; TempestRemap arrives through MOAB's configuration
  endif()
  if(FUSE_ENABLE_PETSC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)   # found, never fetched
  endif()
  if(FUSE_ENABLE_LOWFIVE)
    find_package(HDF5 REQUIRED COMPONENTS C)   # must be a SHARED build; LowFive dlopens the VOL
    find_package(LowFive REQUIRED)
  endif()

  # A build that cannot say which versions it used cannot reproduce its own result.
  file(APPEND "${FUSE_RESOLUTION_FILE}"
       "linalg_backend=${FUSE_LINALG_BACKEND}\n"
       "allow_fetch=${FUSE_ALLOW_FETCH}\n"
       "mpi=${FUSE_ENABLE_MPI}\nmoab=${FUSE_ENABLE_MOAB}\npetsc=${FUSE_ENABLE_PETSC}\n"
       "lowfive=${FUSE_ENABLE_LOWFIVE}\nexceptions=${FUSE_ENABLE_EXCEPTIONS}\n")
endmacro()

macro(fuse_resolve_logging)
  fuse_require(fmtlog FETCH_AS fmtlog
               HINT "fmtlog is the logger (docs/dev/conventions.md); header-only over our fmt.")
  if(NOT TARGET fuse::fmtlog)
    add_library(fuse-fmtlog INTERFACE)
    add_library(fuse::fmtlog ALIAS fuse-fmtlog)
    target_include_directories(fuse-fmtlog INTERFACE ${fmtlog_SOURCE_DIR} ${FMTLOG_INCLUDE_DIR})
    target_compile_definitions(fuse-fmtlog INTERFACE FMTLOG_HEADER_ONLY)
    target_link_libraries(fuse-fmtlog INTERFACE fmt::fmt)
  endif()
endmacro()
