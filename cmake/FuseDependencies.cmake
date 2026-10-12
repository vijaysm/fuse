# FUSE dependency resolution, one tier at a time.
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

# ---------------------------------------------------------------- LowFive +
# fmt -- see KNOWN_ISSUES.md D14's own Unit 3b account for the full
# measured record of how this was actually diagnosed (three successive,
# each-time-incomplete theories, this comment included).
#
# THE ROOT CAUSE, found last: `fuse_require(fmt ...)` (below, where fmt
# itself is resolved) had no minimum version, so with /opt/lowfive FIRST
# on CMAKE_PREFIX_PATH, find_package(fmt) happily accepted the FIRST
# config it found -- and /opt/lowfive ships its OWN bundled fmt 8.1.1
# WITH a discoverable fmt-config.cmake, whose own fmt/format-inl.h fails
# this host's C++20 consteval checking outright. `fuse_require(fmt
# VERSION 9 ...)` is the actual fix: it rejects 8.1.1 and finds
# homebrew's working 12.2.0 instead, in EVERY CMAKE_PREFIX_PATH ordering
# -- fmt::fmt itself now always resolves correctly, project-wide, and no
# per-target include-directory juggling is needed at all.
#
# The `target_include_directories(... BEFORE ...)` machinery just below
# and the `fuse_link_lowfive_with_fmt_fix` function after it were built
# and measured against two EARLIER, incomplete theories (first: "two fmt
# versions disagree in one translation unit, order the include dirs";
# then: "fmt 8.1.1 itself cannot compile here at all, force fmt 12
# everywhere a LowFive header appears") -- both chased a symptom
# (`fmt::fmt`'s own `INTERFACE_INCLUDE_DIRECTORIES` property was ITSELF
# `/opt/lowfive/include` the whole time, which is why prepending
# "fmt::fmt's own include directory" changed nothing: that was never
# homebrew's fmt to begin with). With the version floor above fixing
# `fmt::fmt` at its source, everything below this point is REDUNDANT --
# kept, not load-bearing, because removing six call sites cleanly in the
# time remaining risked a worse mistake than one harmless extra
# target_include_directories call per target. Correctness here rests on
# the VERSION 9 floor alone; a future cleanup pass can remove the rest.
#
# [Superseded reasoning, kept for the record rather than deleted mid-
# investigation: an EARLIER version of this fix relied on target_link_
# libraries' own call ORDER (fuse::model before fuse::lowfive), which
# changes include-directory search order only by accident and broke the
# moment a different, still-supported CMAKE_PREFIX_PATH ordering was
# tried -- `BEFORE` is independent of both CMAKE_PREFIX_PATH's own
# ordering and of target_link_libraries' own call order.]
#
# This function is now a thin, idempotent convenience wrapper -- the fix
# itself is already guaranteed by the INTERFACE target above the moment
# any target links fuse::lowfive at all. Kept (and still called at
# several sites below) only so a future reader lands on the full account
# via whichever call site they happen to be reading, rather than having
# to already know to look at the INTERFACE target's own definition.
function(fuse_link_lowfive_with_fmt_fix target)
  if(NOT TARGET fuse::lowfive)
    message(FATAL_ERROR "fuse_link_lowfive_with_fmt_fix(${target}): fuse::lowfive does not "
                        "exist -- guard the call site with if(TARGET fuse::lowfive) first.")
  endif()
  target_link_libraries(${target} PRIVATE fuse::lowfive)
  if(TARGET fmt::fmt)
    target_include_directories(${target} BEFORE PRIVATE
      $<TARGET_PROPERTY:fmt::fmt,INTERFACE_INCLUDE_DIRECTORIES>)
  endif()
endfunction()

# ---------------------------------------------------------------- Tier 0 (always)
set(FUSE_FMT_TAG        "11.0.2")
set(FUSE_FMTLOG_TAG     "v2.2.1")
# r3d: exact clipping of simplices and analytic moments (Powell & Abel 2015,
# LA-UR-15-26964). It is the supermesh kernel for the ALWAYS-AVAILABLE
# conservative path, so like yaml-cpp it is fetched by default rather than
# required from a prefix: it carries no MPI, no index-size choice and no tuned
# kernels, so a pinned build replaces nothing machine-specific. An installed
# copy is still found first. Its licence is a LANL public notice that must be
# reproduced -- see THIRD-PARTY-NOTICES.md.
set(FUSE_R3D_TAG "58dfbfb2fd2a89e36e3c7ceb5ca6aef3f9c4c4e6" CACHE STRING "r3d commit")
set(FUSE_YAMLCPP_TAG     "0.8.0")
set(FUSE_GTEST_TAG       "v1.15.2")   # git tag; the find_package version below is the same number without the v
set(FUSE_GTEST_VERSION   "1.15.2")
set(FUSE_KOKKOS_TAG     "4.4.01")
set(FUSE_EIGEN_TAG      "3.4.0")
# ArborX: the scalable BVH tree and point-cloud interpolants fuse::search
# wraps (docs/dev/search.md) -- required at every tier, like Kokkos and fmt,
# once Kokkos is (Kokkos is required unconditionally, so ArborX riding on it
# costs nothing new). Unlike yaml-cpp/r3d just above, ArborX is NOT one of the
# two always-fetched exceptions: it follows the ordinary "found first, ON
# ALLOW_FETCH build the pinned copy, OFF with none installed is a configure
# error" policy every other required Tier 0 dependency does, via its own
# fuse_resolve_arborx() below rather than the generic fuse_require() (it needs
# CACHE variables set BEFORE the fetch, the same reason r3d/yaml-cpp have
# their own macros instead of using fuse_require()).
set(FUSE_ARBORX_VERSION "2.1")
# Pinned by COMMIT, not just the tag, so a tag that moves upstream cannot
# silently change what a fetch build compiles. This is the exact commit v2.1
# resolves to; verified by the lead, and proven here against Kokkos 5.1.0
# (UPSTREAMS.lock's [arborx] entry, docs/dev/search.md).
set(FUSE_ARBORX_COMMIT  "8f89be60bc3797670b9be8843da7dcdbc5f2333f")

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

# FUSE_FP_REPRODUCIBLE (default ON; KNOWN_ISSUES.md D30) lives NEXT TO this
# block in spirit but is declared in cmake/FuseFpPolicy.cmake and
# include()'d from CMakeLists.txt BEFORE this file, for the same ordering
# reason FUSE_ENABLE_FORTRAN is declared early there: FuseFpPolicy.cmake's
# Fortran half must run before FuseFortran.cmake's add_subdirectory(fortran),
# which itself runs before this file is included. Not redeclared here --
# see cmake/FuseFpPolicy.cmake for the option, its two apply functions, and
# the full rationale.

# The linalg facade's dense least-squares backend. `builtin` is the coupler's own
# column-pivoted Householder QR with rank truncation and needs nothing; `eigen`
# is a second opinion from a widely used implementation. `builtin` is also the
# oracle the other is checked against, because the recorded iteration counts
# were produced with it.
set(FUSE_LINALG_BACKEND "eigen" CACHE STRING "builtin | eigen")
set_property(CACHE FUSE_LINALG_BACKEND PROPERTY STRINGS builtin eigen)
# A value with no implementation is a configuration error, not a silent
# downgrade: the backends differ in the last bits, an accelerated iteration's
# path depends on those bits, and a run whose reported backend is not the one
# that ran cannot be reproduced from its report.
if(NOT FUSE_LINALG_BACKEND MATCHES "^(builtin|eigen)$")
  message(FATAL_ERROR
    "FUSE_LINALG_BACKEND='${FUSE_LINALG_BACKEND}' has no implementation. "
    "Valid values: builtin (the coupler's own column-pivoted Householder QR, "
    "no dependency) or eigen. A KokkosKernels backend is not available: it has "
    "no column-pivoted QR with rank truncation in the form the quasi-Newton "
    "solve needs.")
endif()

# A tier is a prefix: enabling a higher one without its predecessors is a
# configuration error, never a silent downgrade (docs/user/build.md).
if(FUSE_ENABLE_MOAB AND NOT FUSE_ENABLE_MPI)
  message(FATAL_ERROR "FUSE_ENABLE_MOAB (tier 2) requires FUSE_ENABLE_MPI (tier 1)")
endif()
# PetscToken track (S7.2b), 2026-10-11: FUSE_ENABLE_PETSC no longer requires
# FUSE_ENABLE_MOAB. The tier-3 "PETSc AND MOAB" bundle this gate used to
# enforce was built for a PETSc+MOAB PHYSICS KIT (a mesh-aware solver
# integration, KNOWN_ISSUES.md D32's own "a later rebuild" framing) -- the
# PETSc SOLVER PLUGIN behind S7's Newton seam (fuse-coupler-petsc,
# src/petsc/) needs neither MOAB nor a mesh at all: it is a
# fuse::LinearizationToken (MATMFFD) and a SNES driver over
# Runner::region_fixed_point_residual()'s plain double arrays, found via
# pkg-config alone (the `if(FUSE_ENABLE_PETSC)` block below, pkg_check_
# modules(PETSc)). A tier-0 build (FUSE_ENABLE_MPI=OFF, hence FUSE_ENABLE_
# MOAB=OFF too, since MOAB itself requires MPI two lines above) can
# therefore carry PETSc alone. OBSERVED from /opt/petsc/download-moab's
# own PETSc.pc: its Libs.private names -lmpi from /opt/mpich (PETSc
# itself was configured with MPI), so a tier-0 fuse-coupler-petsc target
# still links libmpi transitively through PkgConfig::PETSc even with
# FUSE_HAVE_MPI=0. Whether PetscInitialize() then calls MPI_Init() itself
# in that case is TO BE MEASURED by test/petsc/init_test.cpp -- see that
# file and this track's own report for the transition actually observed.
# A future MOAB-aware PETSc integration, if one is built, re-adds its OWN
# FUSE_ENABLE_MOAB requirement at that point rather than this
# unconditional one.

# fmt — the ONLY output formatter in the library (docs/dev/conventions.md). No iostreams in
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
# A KokkosKernels linear-algebra backend was designed and pinned here, then the
# project went with `builtin` instead (REVIEW.md R-24/R-42); this declaration
# was never resolved (no find_package/FetchContent_MakeAvailable anywhere) and
# has been removed rather than left dead. Re-add it, wired into
# fuse_resolve_dependencies() and src/CMakeLists.txt's linalg target, if a
# KokkosKernels backend is implemented.
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

FetchContent_Declare(r3d
  GIT_REPOSITORY https://github.com/devonmpowell/r3d.git GIT_TAG ${FUSE_R3D_TAG}
  FIND_PACKAGE_ARGS NAMES r3d)

FetchContent_Declare(Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG ${FUSE_EIGEN_TAG} GIT_SHALLOW ON)

# ArborX. GIT_TAG is the pinned COMMIT (see FUSE_ARBORX_COMMIT's comment
# above), not the tag string -- FIND_PACKAGE_ARGS still names the version so
# an installed ArborX is matched by its real version number. GIT_SHALLOW is
# NOT used here: a shallow clone of a specific commit that is not the tip of
# any branch fails on some git-over-https configurations, and ArborX is
# header-only, so the extra history costs a few seconds once, not a repeated
# build cost.
FetchContent_Declare(ArborX
  GIT_REPOSITORY https://github.com/arborx/arborx.git GIT_TAG ${FUSE_ARBORX_COMMIT})
# Deliberately NO FIND_PACKAGE_ARGS here, unlike the entries above.
# fuse_resolve_arborx() already does its own find_package(ArborX) and only
# reaches FetchContent_MakeAvailable when that failed, so the redirection
# would merely repeat it -- and repeating it is harmful: an installed
# ArborX's ArborXSettings.cmake does a plain `set(ARBORX_ENABLE_MPI ON)`,
# which survives even when the find as a whole fails, and MakeAvailable's
# internal retry re-leaked it AFTER the macro had reset it. The fetched
# ArborX then ran `find_package(MPI REQUIRED)` and a Tier-0 configure died
# on a machine with no discoverable MPI.

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
  # Eigen (and, generically, any future fetched dependency using the same
  # convention) gates its OWN bundled test suite on the standard BUILD_TESTING
  # option, which this project's own enable_testing()/include(CTest) has
  # already turned ON -- add_subdirectory() (what FetchContent_MakeAvailable
  # does) reads that same cache variable, so Eigen registered ~830 of its own
  # "Not Run: Unsupported" CTest entries into THIS project's test suite, and a
  # bare `ctest` failed on them even though every fuse:: test passed (found
  # running the tier0-eigen legs for the first time, CI run 36124149902,
  # 2026-09-25). Saved and restored, not just forced OFF, because fuse's own
  # tests need it back ON afterward.
  set(_fuse_saved_build_testing "${BUILD_TESTING}")
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(${_content})
  set(BUILD_TESTING "${_fuse_saved_build_testing}" CACHE BOOL "" FORCE)
  unset(_fuse_saved_build_testing)
  # Propagate the fetched dependency's version exactly as the find_package
  # branch does. Without it the version is the empty string on the fetch path
  # and Config.KokkosHeadersMatchTheResolvedLibrary fails parsing "" rather
  # than comparing anything -- found 2026-09-17 running the OpenMP CI leg by
  # hand, the second defect that leg would have caught and never did.
  #
  # The variable has to be read out of the SUBPROJECT's directory scope:
  # FetchContent_MakeAvailable() is an add_subdirectory(), so the `set()` the
  # dependency's own CMakeLists performs is invisible here and is not cached.
  # Reading it this way reports the version that was actually BUILT rather than
  # the tag we asked for, which are the same thing until a tag is moved.
  string(TOLOWER "${_content}" _content_lc)
  if(DEFINED ${_content_lc}_SOURCE_DIR)
    get_directory_property(_fetched_version
      DIRECTORY "${${_content_lc}_SOURCE_DIR}" DEFINITION ${package}_VERSION)
    if(_fetched_version)
      set(${package}_VERSION "${_fetched_version}" PARENT_SCOPE)
    endif()
  endif()
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
    # An ALIAS cannot carry properties, and set_target_properties on one is a
    # hard CMake error. Both kinds appear here depending on how the dependency
    # arrived: an INSTALLED Kokkos exports IMPORTED targets (settable), while a
    # Kokkos built from source exposes ALIASES onto its own real targets. The
    # stale-include-dir problem this function exists for is specific to the
    # installed case -- an exported config naming an SDK path that no longer
    # exists -- so skipping aliases loses nothing: a target built in this very
    # build tree cannot have a stale include directory.
    #
    # Found 2026-09-17 by running the OpenMP CI leg's recipe by hand. The leg
    # fetches Kokkos, so it would have failed at CONFIGURE, and it has never
    # run: the defect was invisible to every build done here, all of which used
    # the installed Kokkos.
    get_target_property(_alias ${target} ALIASED_TARGET)
    if(_alias)
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
# The supermesh kernel. Resolved with the core rather than in the tier ladder:
# the lowest-order conservative path must work in the DEFAULT build, which is
# the whole point of not depending on TempestRemap for it.
macro(fuse_resolve_r3d)
  find_package(r3d QUIET CONFIG)
  if(r3d_FOUND)
    file(APPEND "${FUSE_RESOLUTION_FILE}" "r3d=found ${r3d_DIR}\n")
    message(STATUS "fuse: r3d found at ${r3d_DIR}")
  else()
    # 512 is r3d's own default and is the bound on vertices in a clipped
    # polyhedron. Clipping one tetrahedron by another cannot approach it; the
    # value is pinned here so the limit is a recorded choice rather than
    # whatever a future upstream default becomes.
    set(R3D_MAX_VERTS 512 CACHE STRING "" FORCE)
    set(ENABLE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(r3d)
    file(APPEND "${FUSE_RESOLUTION_FILE}" "r3d=fetched ${FUSE_R3D_TAG}\n")
    message(STATUS "fuse: r3d built from source (pinned ${FUSE_R3D_TAG})")
  endif()
endmacro()

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

# ArborX. Found first, otherwise always fetched (never gated on
# FUSE_ALLOW_FETCH), and written by
# hand rather than calling fuse_require() itself, because the fetch path needs
# CACHE variables set BEFORE FetchContent_MakeAvailable(ArborX) runs --
# ARBORX_ENABLE_MPI in particular must track FUSE_ENABLE_MPI, which
# fuse_require()'s generic fetch branch has no hook for (the same reason
# fuse_resolve_r3d()/fuse_resolve_yaml() above are hand-written too).
macro(fuse_resolve_arborx)
  find_package(ArborX ${FUSE_ARBORX_VERSION} QUIET CONFIG)
  if(ArborX_FOUND)
    file(APPEND "${FUSE_RESOLUTION_FILE}" "ArborX=found ${ArborX_VERSION} ${ArborX_DIR}\n")
    message(STATUS "fuse: ArborX ${ArborX_VERSION} found at ${ArborX_DIR}")
  else()
    # Always fetched when absent, like r3d and yaml-cpp above, and unlike the
    # FUSE_ALLOW_FETCH-gated Tier 0 dependencies: the user asked for ArborX to
    # be fetched dynamically, and a default configure must not fail for want
    # of it. An installed ArborX >= 2.1 on CMAKE_PREFIX_PATH is still preferred.
    # Built against the SAME Kokkos this build already resolved above:
    # ArborX's own CMakeLists calls find_package(Kokkos 4.5 REQUIRED CONFIG)
    # itself, which is a no-op re-use of the existing imported targets once
    # Kokkos::kokkoscore already exists in this configure -- it never fetches
    # or builds a second Kokkos. (/opt/kokkos 5.1.0's version file is
    # AnyNewer, so ArborX's ">= 4.5" request is satisfied by it; proven to
    # actually COMPILE against 5.1.0, not just configure, per
    # docs/dev/search.md.)
    # The failed find_package above is not side-effect free. An INSTALLED
    # ArborX's ArborXSettings.cmake does a plain `set(ARBORX_ENABLE_MPI ON)`,
    # which survives in this scope even when the find as a whole fails (its
    # own find_dependency(MPI) not resolving is the usual way that happens).
    # A normal variable shadows the cache entry, so FORCE-ing the cache alone
    # left the fetched ArborX seeing ON: its CMakeLists then ran
    # `find_package(MPI REQUIRED)` and a Tier-0 configure died on a machine
    # with no discoverable MPI. Clear the normal variable, then set both.
    unset(ARBORX_ENABLE_MPI)
    set(ARBORX_ENABLE_MPI ${FUSE_ENABLE_MPI} CACHE BOOL "" FORCE)
    set(ARBORX_ENABLE_MPI ${FUSE_ENABLE_MPI})
    set(ARBORX_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(ARBORX_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(ARBORX_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
    message(STATUS "fuse: no usable installed ArborX (>= ${FUSE_ARBORX_VERSION}); "
                   "building the pinned copy (commit ${FUSE_ARBORX_COMMIT}). If one IS "
                   "installed, its own config did not resolve -- an MPI-enabled ArborX "
                   "needs MPI discoverable even for a Tier-0 build")
    FetchContent_MakeAvailable(ArborX)
    file(APPEND "${FUSE_RESOLUTION_FILE}"
         "ArborX=fetched (pinned ${FUSE_ARBORX_COMMIT}, not tuned for this machine)\n")
  endif()
endmacro()

macro(fuse_resolve_dependencies)
  # VERSION 9: the actual root cause of this session's own LowFive/fmt
  # saga (KNOWN_ISSUES.md D14's Unit 3b account), found only after the
  # earlier "two fmt versions in one translation unit" diagnosis turned
  # out to be incomplete. fuse_require's own find_package(fmt ...) call
  # had NO version floor, so with /opt/lowfive FIRST on CMAKE_PREFIX_PATH
  # it happily accepted the FIRST fmt config it found -- and /opt/lowfive
  # ships its OWN bundled fmt 8.1.1 WITH a discoverable fmt-config.cmake
  # (/opt/lowfive/lib/cmake/fmt/fmt-config.cmake), whose own fmt/
  # format-inl.h fails this host's C++20 consteval checking outright,
  # in ANY translation unit that ends up using it, with or without any
  # FUSE code involved (measured: a standalone translation unit naming
  # only <lowfive/vol-metadata.hpp>, no FUSE header at all, failed
  # identically). A floor of 9 is comfortably above 8.1.1 and
  # comfortably below every fmt this project has ever targeted (FETCH
  # pins 11.0.2; homebrew on this host carries 12.2.0) -- the goal is
  # excluding the one known-bad install, not pinning a ceiling.
  fuse_require(fmt VERSION 9 HINT "fmt is the only output formatter (docs/dev/conventions.md).")

  # Apple's clang ships no OpenMP runtime, so CMake's FindOpenMP fails with no
  # hints even when libomp IS installed. A Kokkos built with OpenMP does
  # find_dependency(OpenMP) from its own config file, so a perfectly good
  # install then fails to resolve with "Could NOT find OpenMP_CXX (missing:
  # OpenMP_CXX_FLAGS OpenMP_CXX_LIB_NAMES)" -- which reads as though Kokkos
  # were broken rather than as a missing hint.
  #
  # Hit on 2026-09-17 when the installed Kokkos moved to 5.1.0 with
  # OPENMP;SERIAL. Supplying the conventional prefix here rather than expecting
  # it on every command line: the caller should not have to know that this
  # platform's compiler needs a third-party runtime to honour a standard
  # pragma. An explicit -DOpenMP_ROOT always wins.
  if(APPLE AND NOT DEFINED OpenMP_ROOT AND NOT DEFINED CACHE{OpenMP_ROOT})
    foreach(_omp_prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
      if(EXISTS "${_omp_prefix}/include/omp.h")
        set(OpenMP_ROOT "${_omp_prefix}")
        # The LIBRARY has to be pinned too, not just the prefix. FindOpenMP
        # validates clang's flags (`-Xclang -fopenmp`) and then searches for a
        # runtime independently, and on this machine it selected
        # /usr/local/lib/libgomp.dylib -- GNU's runtime, which exports
        # `GOMP_*` and NOT the `__kmpc_*` entry points clang's code generation
        # and Kokkos' own objects call. The build then failed at link with
        # "Undefined symbols ... ___kmpc_fork_call, referenced from
        # Kokkos::OpenMP::impl_initialize", which reads as a broken Kokkos
        # rather than as two OpenMP runtimes on one machine and the wrong one
        # chosen. Both are arm64 and both export `GOMP_parallel`, so neither
        # architecture nor a missing file is the discriminator: only the ABI
        # is, and it must match the compiler that generated the code.
        if(EXISTS "${_omp_prefix}/lib/libomp.dylib"
           AND NOT DEFINED CACHE{OpenMP_libomp_LIBRARY})
          set(OpenMP_libomp_LIBRARY "${_omp_prefix}/lib/libomp.dylib"
              CACHE FILEPATH "OpenMP runtime matching the compiler's ABI")
        endif()
        message(STATUS "fuse: OpenMP runtime at ${_omp_prefix} "
                       "(Apple clang provides none; override with -DOpenMP_ROOT)")
        break()
      endif()
    endforeach()
  endif()

  fuse_require(Kokkos HINT "Kokkos is required at every tier; /opt/kokkos on this machine. If it was built with OpenMP and this fails on OpenMP_CXX, install libomp (brew install libomp) or pass -DOpenMP_ROOT.")
  # Sanitized before anything links it. The offender on this machine is
  # Kokkos::LIBDL, whose only usage requirement is the SDK include path Kokkos was
  # configured against; the rest are listed so a different install's stale entry is
  # caught the same way.
  fuse_resolve_r3d()
  fuse_resolve_yaml()
  fuse_resolve_arborx()

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

  if(FUSE_LINALG_BACKEND STREQUAL "eigen")
    # No version constraint: Eigen3Config is SameMajorVersion, so asking for 3.4
    # rejects an installed 5.x outright. The facade uses dense QR, small
    # least-squares and sparse products, whose API is identical across both.
    fuse_require(Eigen3 NAMES Eigen3 HINT "Set FUSE_LINALG_BACKEND=builtin to use the built-in factorisation instead.")
  endif()
  # `builtin` resolves nothing: that is the point of having it.

  if(FUSE_ENABLE_TESTS)
    fuse_require(GTest VERSION ${FUSE_GTEST_VERSION} NAMES GTest FETCH_AS googletest
                 HINT "GoogleTest is the test harness; -DFUSE_ENABLE_TESTS=OFF skips it.")
  endif()

  if(FUSE_ENABLE_MPI)
    if(FUSE_ENABLE_FORTRAN)
      # Conformance Unit 2's own v_main.f90/v_loop.f90 (examples/
      # brusselator/participants/) call MPI_Init/MPI_Comm_rank/MPI_Comm_
      # size/MPI_Finalize/MPI_Abort directly through the standard Fortran
      # `mpi` module, needing MPI::MPI_Fortran -- requested only when
      # FUSE_ENABLE_FORTRAN already enabled the Fortran language (include
      # (FuseFortran) above runs before this file, cmake/FuseFortran.cmake's
      # own header comment), so this request is never made on a machine
      # with no Fortran compiler (FUSE_ENABLE_FORTRAN would already have
      # been forced back OFF there).
      find_package(MPI REQUIRED COMPONENTS C CXX Fortran)
    else()
      find_package(MPI REQUIRED COMPONENTS C CXX)
    endif()

    # ---- can this MPI connect two separately launched programs?
    # The two multi-executable launch modes do not need the same things from
    # an MPI. MPMD asks only for MPI_APPNUM, which every launcher provides.
    # Accept/connect asks for dynamic process management AND for it to work
    # between two jobs that share a node, which is where MPICH's ch3:nemesis
    # device fails (KNOWN_ISSUES.md D23). So the capability is probed, not
    # assumed, and the two modes are enabled separately.
    #
    # Unlike the two-HDF5-image probe above, a failure here is NOT fatal: it
    # disables one launch mode and leaves everything else -- including MPMD,
    # which is the mode the examples use -- working.
    #
    # THREE verdicts, not two, and the third is why this is written out
    # rather than inlined: "could not test" must behave differently from
    # "measured broken". A login node where `mpiexec` cannot start a job
    # without an allocation is the ordinary case of the former, and
    # disabling a working capability there would be a worse error than
    # leaving a broken one enabled, which merely fails loudly at the rows
    # that use it. So untested leaves the mode ON and says so.
    set(_fuse_dyn_script "${PROJECT_SOURCE_DIR}/tools/probe-mpi-dynamic.sh")
    set(_fuse_dyn_default ON)
    set(_fuse_dyn_verdict "assumed (probe not run)")
    if(NOT DEFINED FUSE_MPI_DYNAMIC_PROBED AND EXISTS "${_fuse_dyn_script}")
      message(STATUS "fuse: probing whether this MPI can connect two same-node jobs ...")
      execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
          "FUSE_PROBE_MPICC=${MPI_C_COMPILER}"
          "FUSE_PROBE_MPIEXEC=${MPIEXEC_EXECUTABLE}"
          "FUSE_PROBE_BUILD_DIR=${CMAKE_BINARY_DIR}/fuse-probe-mpi-dynamic"
          "${_fuse_dyn_script}"
        RESULT_VARIABLE _fuse_dyn_result
        OUTPUT_VARIABLE _fuse_dyn_output
        ERROR_VARIABLE  _fuse_dyn_error)
      if(_fuse_dyn_result EQUAL 0)
        set(_fuse_dyn_default ON)
        set(_fuse_dyn_verdict "measured working")
        message(STATUS "fuse: accept/connect works on this MPI")
      elseif(_fuse_dyn_result EQUAL 1)
        set(_fuse_dyn_default OFF)
        set(_fuse_dyn_verdict "measured broken")
        message(STATUS
          "fuse: this MPI cannot connect two same-node jobs, so the "
          "accept/connect launch mode is DISABLED (KNOWN_ISSUES.md D23). "
          "MPMD is unaffected. Probe output:\n${_fuse_dyn_output}${_fuse_dyn_error}")
      else()
        set(_fuse_dyn_default ON)
        set(_fuse_dyn_verdict "not tested")
        message(STATUS
          "fuse: could not test accept/connect here (the probe could not "
          "build or launch), so the mode is left enabled and its rows will "
          "report for themselves. Reason:\n${_fuse_dyn_error}")
      endif()
      set(FUSE_MPI_DYNAMIC_PROBED "${_fuse_dyn_verdict}" CACHE INTERNAL
          "Verdict of tools/probe-mpi-dynamic.sh")
    elseif(DEFINED FUSE_MPI_DYNAMIC_PROBED)
      set(_fuse_dyn_verdict "${FUSE_MPI_DYNAMIC_PROBED}")
      if(_fuse_dyn_verdict STREQUAL "measured broken")
        set(_fuse_dyn_default OFF)
      endif()
    endif()
    option(FUSE_ENABLE_CONNECT_ACCEPT
           "Multi-executable: the accept/connect launch mode (needs an MPI whose dynamic process management works between same-node jobs)"
           ${_fuse_dyn_default})
    message(STATUS "fuse: accept/connect launch mode: "
                   "${FUSE_ENABLE_CONNECT_ACCEPT} (${_fuse_dyn_verdict})")
  endif()
  if(FUSE_ENABLE_MOAB)
    # MOAB ships an OLD-STYLE config: it sets MOAB_INCLUDE_DIRS /
    # MOAB_LIBRARIES and exports no imported target (checked against
    # MOABConfig.cmake, which contains zero add_library calls). A bare
    # find_package(MOAB REQUIRED) therefore SUCCEEDS and attaches nothing --
    # configure passes, FUSE_HAVE_MOAB becomes 1, and the first MOAB include
    # fails to compile. That is an option claiming a capability the build
    # cannot deliver, so the variables are wrapped in a target here and the
    # result is recorded like every other dependency.
    find_package(MOAB REQUIRED)
    # MOAB's config bakes in the link line of the toolchain it was BUILT with,
    # including the gfortran runtime directory of that gcc. When that gcc is
    # upgraded the directory disappears and the config is stale: configure
    # still succeeds, and the failure arrives as `ld: library 'emutls_w' not
    # found` at the end of a full build, which reads like a FUSE bug. Found
    # 2026-09-19 after a Homebrew gcc 15 -> 16 upgrade. Checked here so the
    # message names the actual remedy.
    set(_fuse_moab_stale "")
    # MOAB_LIBRARIES is ONE space-separated string (an old-style config), not a
    # CMake list, so it is tokenised on whitespace first.
    string(REGEX REPLACE "[ \t]+" ";" _fuse_moab_tokens "${MOAB_LIBRARIES}")
    foreach(_tok IN LISTS _fuse_moab_tokens)
      string(REGEX MATCH "^(-L|-Wl,-rpath,)(.+)$" _m "${_tok}")
      if(_m AND NOT IS_DIRECTORY "${CMAKE_MATCH_2}")
        list(APPEND _fuse_moab_stale "${CMAKE_MATCH_2}")
      endif()
    endforeach()
    if(_fuse_moab_stale)
      list(REMOVE_DUPLICATES _fuse_moab_stale)
      list(JOIN _fuse_moab_stale "\n    " _fuse_moab_stale_text)
      message(FATAL_ERROR
        "fuse: the MOAB found at ${MOAB_DIR} was built against a toolchain that "
        "is no longer installed. Its MOABConfig.cmake links against directories "
        "that do not exist:\n    ${_fuse_moab_stale_text}\n"
        "  Linking would fail late with `ld: library ... not found`. Remedy: "
        "rebuild MOAB (and PETSc, whose .pc carries the same line) against the "
        "current compiler, or point CMAKE_PREFIX_PATH at a MOAB that matches it. "
        "There is nothing FUSE can substitute: the libraries in that directory "
        "belong to the compiler that produced them.")
    endif()
    # MOAB_LIBRARIES repeats the gfortran -Wl,-rpath/-L entries up to three
    # times (its old-style config concatenates several link-line fragments
    # that each carry the same toolchain directory), which otherwise reaches
    # the linker verbatim and prints one `ld: warning: duplicate -rpath ...
    # ignored` per repeat, per target -- harmless, but noisy on every tier-2
    # link (REVIEW.md pass2 N-build-tools-docs-10). Reuse the same
    # whitespace-tokenised form the stale-directory check above already
    # built; only -L/-Wl,-rpath tokens are deduplicated, order-preserving --
    # a repeated plain -lFoo is left alone since library link order can
    # matter and de-duplicating it is not what this defect is about.
    #
    # `-framework X` is ONE link item spelt as two words. Tokenising on
    # whitespace splits it, and CMake then renders the orphaned name as a plain
    # library: `ld: framework '-lAccelerate' not found`. A MOAB built with
    # TempestRemap on macOS carries BLAS as "-framework Accelerate", so
    # /opt/moab-tempest failed to link every tier-2 target. The pair is folded
    # back into the single token `-Wl,-framework,X`, which survives as a list
    # element and reaches the linker unchanged.
    set(_fuse_moab_libs_dedup "")
    set(_fuse_moab_seen_dirs "")
    set(_fuse_moab_pending_framework FALSE)
    foreach(_tok IN LISTS _fuse_moab_tokens)
      if(_tok STREQUAL "")
        continue()
      endif()
      if(_fuse_moab_pending_framework)
        list(APPEND _fuse_moab_libs_dedup "-Wl,-framework,${_tok}")
        set(_fuse_moab_pending_framework FALSE)
        continue()
      endif()
      if(_tok STREQUAL "-framework")
        set(_fuse_moab_pending_framework TRUE)
        continue()
      endif()
      string(REGEX MATCH "^(-L|-Wl,-rpath,)(.+)$" _m "${_tok}")
      if(_m)
        list(FIND _fuse_moab_seen_dirs "${_tok}" _fuse_moab_seen_idx)
        if(_fuse_moab_seen_idx EQUAL -1)
          list(APPEND _fuse_moab_seen_dirs "${_tok}")
          list(APPEND _fuse_moab_libs_dedup "${_tok}")
        endif()
      else()
        list(APPEND _fuse_moab_libs_dedup "${_tok}")
      endif()
    endforeach()
    # A MOAB's config does not always export its dependencies' headers.
    # /opt/moab/cmake-all absorbs libTempestRemap into libMOAB (the symbols
    # are exported, so linking is fine) but never sets MOAB_PACKAGE_INCLUDES,
    # so MOAB_INCLUDE_DIRS names only MOAB's own include directory:
    # tempest_weights.cpp fails on <Announce.h>, then TempestRemap's own
    # headers fail on <netcdf.h>. The config DOES record, per dependency it
    # was built with, MOAB_USE_<PKG> and <PKG>_DIR, so each such dependency's
    # <PKG>_DIR/include is added when it exists and is not already listed.
    # A MOAB whose config does export them (/opt/moab-tempest) is unchanged.
    foreach(_pkg NETCDF PNETCDF HDF5 TEMPESTREMAP EIGEN3 ZOLTAN)
      set(_pkg_inc "${${_pkg}_DIR}/include")
      if(MOAB_USE_${_pkg} AND ${_pkg}_DIR AND IS_DIRECTORY "${_pkg_inc}")
        list(FIND MOAB_INCLUDE_DIRS "${_pkg_inc}" _pkg_idx)
        if(_pkg_idx EQUAL -1)
          list(APPEND MOAB_INCLUDE_DIRS "${_pkg_inc}")
          message(STATUS "fuse: ${_pkg} headers from ${_pkg_inc} "
                         "(this MOAB's config does not export them)")
        endif()
      endif()
    endforeach()
    if(MOAB_USE_TEMPESTREMAP)
      set(_fuse_tempest_reachable FALSE)
      foreach(_d IN LISTS MOAB_INCLUDE_DIRS)
        if(EXISTS "${_d}/Announce.h")
          set(_fuse_tempest_reachable TRUE)
        endif()
      endforeach()
      if(NOT _fuse_tempest_reachable)
        if(TEMPESTREMAP_DIR AND EXISTS "${TEMPESTREMAP_DIR}/include/Announce.h")
          list(APPEND MOAB_INCLUDE_DIRS "${TEMPESTREMAP_DIR}/include")
          message(STATUS "fuse: TempestRemap headers from ${TEMPESTREMAP_DIR}/include "
                         "(MOAB's config does not export them)")
        else()
          message(FATAL_ERROR
            "fuse: the MOAB at ${MOAB_DIR} was built with TempestRemap, but its "
            "headers (Announce.h) are on none of MOAB_INCLUDE_DIRS and "
            "TEMPESTREMAP_DIR='${TEMPESTREMAP_DIR}' has no include/Announce.h. "
            "Point TEMPESTREMAP_DIR at the TempestRemap install MOAB was built against.")
        endif()
      endif()
    endif()
    if(NOT TARGET fuse::moab)
      # An IMPORTED target is exempt from install(EXPORT)'s "every PUBLIC
      # link dependency must be in some export set" check -- the same
      # exemption Kokkos::kokkos/fmt::fmt/MPI::MPI_CXX already rely on,
      # because those are IMPORTED targets created by find_package's own
      # Find/Config modules. fuse-moab is not: MOAB's shipped MOABConfig.cmake
      # creates no imported target at all (see the comment above), so this
      # project's own wrapper is what needs the IMPORTED marker instead.
      # Without it, `fuse-coupler-backend`'s PUBLIC link to `fuse::moab`
      # fails install(EXPORT) at CMake's Generate step with "requires target
      # fuse-moab that is not in any export set" -- invisible at tier 0/1
      # (MOAB is off there), a BLOCKER the first time tier 2 configures
      # (REVIEW.md pass2 N-build-tools-docs-9). Being IMPORTED does not by
      # itself make a downstream consumer's find_package(fuse-coupler) able
      # to reconstruct this target -- fuse-couplerConfig.cmake.in does that
      # explicitly, baking in the same resolved paths, since MOAB's own
      # config gives it nothing portable to find_dependency() against.
      add_library(fuse-moab INTERFACE IMPORTED GLOBAL)
      add_library(fuse::moab ALIAS fuse-moab)
      target_include_directories(fuse-moab SYSTEM INTERFACE ${MOAB_INCLUDE_DIRS})
      target_link_libraries(fuse-moab INTERFACE ${_fuse_moab_libs_dedup})
      target_link_directories(fuse-moab INTERFACE ${MOAB_LIBRARY_DIRS})
      # MOAB's public headers include <moab_mpi.h>, which includes <mpi.h>
      # unconditionally for a parallel MOAB build -- so anything consuming a
      # MOAB header needs MPI's include path even if it never calls MPI
      # itself. Without this the very first MOAB include fails with
      # "'mpi.h' file not found", which reads like a missing MPI install
      # rather than a missing interface property.
      if(FUSE_ENABLE_MPI AND TARGET MPI::MPI_CXX)
        target_link_libraries(fuse-moab INTERFACE MPI::MPI_CXX)
      endif()
    endif()
    # Baked in verbatim (via configure_package_config_file's @VAR@
    # substitution) into fuse-couplerConfig.cmake.in so a downstream
    # find_package(fuse-coupler) consumer can recreate this exact target --
    # see that file for why these are absolute, machine-specific paths
    # rather than a portable find_dependency() call.
    set(FUSE_MOAB_INCLUDE_DIRS "${MOAB_INCLUDE_DIRS}")
    set(FUSE_MOAB_LIBRARIES_RESOLVED "${_fuse_moab_libs_dedup}")
    set(FUSE_MOAB_LIBRARY_DIRS "${MOAB_LIBRARY_DIRS}")
    # Whether high-order (spherical, TempestRemap-backed) remap is available is
    # a property of WHICH MOAB was found, not of this project: a machine can
    # have several MOAB builds with different capabilities. Recorded rather
    # than assumed, so a skipped high-order test can be explained.
    if(MOAB_USE_TEMPESTREMAP)
      set(FUSE_MOAB_HAS_TEMPESTREMAP ON)
    else()
      set(FUSE_MOAB_HAS_TEMPESTREMAP OFF)
      message(STATUS "fuse: MOAB at ${MOAB_INCLUDE_DIRS} has no TempestRemap; "
                     "high-order spherical remap will be unavailable")
    endif()
    file(APPEND "${FUSE_RESOLUTION_FILE}"
         "MOAB=found ${MOAB_VERSION} ${MOAB_INCLUDE_DIRS}\n"
         "MOAB_tempestremap=${FUSE_MOAB_HAS_TEMPESTREMAP}\n")
    # config.hpp's FUSE_HAVE_TEMPEST comes from here, and nowhere else.
    set(FUSE_HAVE_TEMPEST ${FUSE_MOAB_HAS_TEMPESTREMAP})
  endif()
  if(FUSE_ENABLE_PETSC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)   # found, never fetched
    # PkgConfig::PETSc's own Cflags (PETSc.pc) is `-I${includedir}` only --
    # PETSc's OWN include dir, not MPI's. <petsc.h> unconditionally
    # includes <mpi.h> (PETSc links MPI regardless of FUSE_ENABLE_MPI,
    # measured from /opt/petsc/download-moab's own PETSc.pc Libs.private
    # naming -lmpi), so a tier-0 build (FUSE_ENABLE_MPI=OFF, MPI::MPI_C
    # not yet found by the FUSE_ENABLE_MPI branch above) still needs an
    # MPI include path for fuse-coupler-petsc (src/petsc/*.cpp, the only
    # translation units naming <petsc.h>/<mpi.h> in that library). Found
    # here, independently of FUSE_ENABLE_MPI, guarded so a build that
    # already found MPI (FUSE_ENABLE_MPI=ON) does not re-find it.
    if(NOT TARGET MPI::MPI_C)
      find_package(MPI REQUIRED COMPONENTS C)
    endif()
  endif()
  if(FUSE_ENABLE_LOWFIVE)
    # ------------------------------------------------------------ LowFive
    # Task 8.4 / S6 LowFive track. must be a SHARED HDF5 build; LowFive
    # dlopens the VOL, and a static HDF5 cannot back a runtime-loaded plugin.
    find_package(HDF5 REQUIRED COMPONENTS C)
    find_package(LowFive REQUIRED)   # cmake/FindLowFive.cmake: no vendor config ships one

    if(TARGET HDF5::HDF5)
      set(_fuse_lowfive_hdf5_target HDF5::HDF5)
    else()
      # Older CMake's bundled FindHDF5.cmake sets only the legacy variables;
      # wrap them the same defensive way MOAB's old-style config is wrapped
      # a few dozen lines above, rather than assume a target exists.
      if(NOT TARGET fuse-hdf5-raw)
        add_library(fuse-hdf5-raw INTERFACE IMPORTED GLOBAL)
        target_include_directories(fuse-hdf5-raw SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(fuse-hdf5-raw INTERFACE ${HDF5_C_LIBRARIES} ${HDF5_LIBRARIES})
      endif()
      set(_fuse_lowfive_hdf5_target fuse-hdf5-raw)
    endif()

    if(NOT TARGET fuse::lowfive)
      add_library(fuse-lowfive INTERFACE IMPORTED GLOBAL)
      add_library(fuse::lowfive ALIAS fuse-lowfive)
      target_link_libraries(fuse-lowfive INTERFACE fuse::lowfive-raw ${_fuse_lowfive_hdf5_target})
      if(FUSE_ENABLE_MPI AND TARGET MPI::MPI_CXX)
        # LowFive's own headers are built against MPI (vol-metadata.hpp
        # forward-declares diy::mpi types); see cmake/FindLowFive.cmake.
        target_link_libraries(fuse-lowfive INTERFACE MPI::MPI_CXX)
      endif()
      # /opt/lowfive's own include directory carries a bundled fmt
      # 8.1.1 whose own fmt/format-inl.h fails this host's C++20
      # consteval checking OUTRIGHT -- measured this session in a
      # translation unit that names ONLY <lowfive/vol-metadata.hpp>,
      # no FUSE header at all (src/transport/lowfive_plugin_vol.cpp):
      # this is not "two fmt versions disagree," it is "fmt 8.1.1 does
      # not compile on this host under any circumstance." Every
      # consumer of fuse::lowfive therefore gets fmt::fmt's own
      # (homebrew, 12.2.0, working) include directory prepended ahead
      # of whatever fuse::lowfive-raw itself contributes, via `BEFORE`
      # on this INTERFACE property -- independent of CMAKE_PREFIX_PATH's
      # own ordering and of target_link_libraries' own call order alike,
      # unlike an earlier version of this fix that relied on link order
      # and broke under a different, still-supported CMAKE_PREFIX_PATH.
      # Fixed here, once, rather than per-consumer (fuse_link_lowfive_
      # with_fmt_fix, just above this macro, still exists and is still
      # called at several sites below -- now a harmless, idempotent
      # belt-and-suspenders restatement of what this interface target
      # already guarantees, not load-bearing on its own, kept so a
      # future reader at any ONE of those call sites still finds the
      # full account without having to already know to look here).
      if(TARGET fmt::fmt)
        target_include_directories(fuse-lowfive BEFORE INTERFACE
          $<TARGET_PROPERTY:fmt::fmt,INTERFACE_INCLUDE_DIRECTORIES>)
      endif()
    endif()

    # ---- the two-HDF5-image hazard: DYNAMIC acceptance test, at configure
    # time, not a warning. KNOWN_ISSUES.md section D, docs/user/lowfive.md,
    # tools/probe-hdf5-images.{c,sh}. The hazard needs BOTH libraries linked
    # into the same binary, so it only applies when MOAB is also enabled;
    # with FUSE_ENABLE_MOAB=OFF there is nothing for LowFive's HDF5 to
    # collide with and the probe would prove nothing.
    if(FUSE_ENABLE_MOAB)
      get_filename_component(_fuse_lowfive_root "${LOWFIVE_INCLUDE_DIR}" DIRECTORY)
      get_filename_component(_fuse_moab_root "${MOAB_LIBRARY_DIRS}" DIRECTORY)
      list(GET HDF5_INCLUDE_DIRS 0 _fuse_hdf5_include0)
      get_filename_component(_fuse_hdf5_root "${_fuse_hdf5_include0}" DIRECTORY)
      set(_fuse_probe_script "${PROJECT_SOURCE_DIR}/tools/probe-hdf5-images.sh")
      if(NOT EXISTS "${_fuse_probe_script}")
        message(FATAL_ERROR
          "fuse: ${_fuse_probe_script} is missing; cannot verify LowFive and "
          "MOAB can safely share a process (KNOWN_ISSUES.md section D). This "
          "check is not optional -- see docs/user/lowfive.md.")
      endif()
      message(STATUS "fuse: probing for the two-HDF5-image hazard "
                     "(lowfive=${_fuse_lowfive_root} moab=${_fuse_moab_root} "
                     "hdf5=${_fuse_hdf5_root}) ...")
      # Pass what THIS configure already resolved -- MOAB's own compiler and
      # its package include dirs (MOAB_CXX, MOAB_PACKAGE_INCLUDES, both set
      # by MOABConfig.cmake above) and the fmt include FindLowFive.cmake
      # settled on -- rather than let the script fall back to its own /opt
      # defaults, which are this lab's but not necessarily the next host's.
      string(REPLACE ";" " " _fuse_moab_extra_includes_sp "${MOAB_PACKAGE_INCLUDES}")
      execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
          "FUSE_PROBE_CXX=${MOAB_CXX}"
          "FUSE_PROBE_MOAB_EXTRA_INCLUDES=${_fuse_moab_extra_includes_sp}"
          "FUSE_PROBE_FMT_INCLUDE=${LOWFIVE_FMT_INCLUDE_DIR}"
          "FUSE_PROBE_BUILD_DIR=${CMAKE_BINARY_DIR}/fuse-probe-hdf5-images"
          "${_fuse_probe_script}" "${_fuse_lowfive_root}" "${_fuse_moab_root}" "${_fuse_hdf5_root}"
        RESULT_VARIABLE _fuse_probe_result
        OUTPUT_VARIABLE _fuse_probe_output
        ERROR_VARIABLE  _fuse_probe_error)
      message(STATUS "${_fuse_probe_output}")
      if(_fuse_probe_result EQUAL 1)
        message(FATAL_ERROR
          "fuse: REFUSING this configuration. tools/probe-hdf5-images.sh "
          "measured more than one libhdf5 image loading when this LowFive "
          "(${_fuse_lowfive_root}) and this MOAB (${_fuse_moab_root}) are "
          "linked into one process -- LowFive's VOL interception would "
          "silently not apply to MOAB's own HDF5 traffic (or vice versa), "
          "with no link error and no runtime error. Full probe output:\n"
          "${_fuse_probe_output}\n${_fuse_probe_error}\n"
          "Rebuild LowFive against the SAME HDF5 prefix MOAB uses "
          "(see the probe's own output above, and docs/user/lowfive.md) and "
          "point -DCMAKE_PREFIX_PATH at that new prefix instead.")
      elseif(NOT _fuse_probe_result EQUAL 0)
        message(FATAL_ERROR
          "fuse: tools/probe-hdf5-images.sh could not determine whether the "
          "two-HDF5-image hazard applies here (exit ${_fuse_probe_result}, "
          "unrelated to the hazard itself -- see its output above and "
          "docs/user/lowfive.md). Configuring FUSE_ENABLE_LOWFIVE with MOAB "
          "also enabled without this proof is exactly the silent failure "
          "mode this check exists to prevent, so this is refused rather "
          "than downgraded to a warning.")
      endif()
    else()
      message(STATUS "fuse: FUSE_ENABLE_MOAB is OFF; the two-HDF5-image "
                     "hazard needs both libraries linked to occur, so the "
                     "probe is skipped (nothing to test).")
    endif()
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
