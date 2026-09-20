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
if(FUSE_ENABLE_PETSC AND NOT FUSE_ENABLE_MOAB)
  message(FATAL_ERROR "FUSE_ENABLE_PETSC (tier 3) requires FUSE_ENABLE_MOAB (tier 2)")
endif()

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

macro(fuse_resolve_dependencies)
  fuse_require(fmt HINT "fmt is the only output formatter (docs/dev/conventions.md).")

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
    find_package(MPI REQUIRED COMPONENTS C CXX)
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
    set(_fuse_moab_libs_dedup "")
    set(_fuse_moab_seen_dirs "")
    foreach(_tok IN LISTS _fuse_moab_tokens)
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
