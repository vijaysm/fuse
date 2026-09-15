# FUSE dependency resolution, one tier at a time (docs/PLAN.md §3.1).
#
# Policy: prefer a system/spack package; fall back to a PINNED FetchContent tag.
# Pins are exact because an unpinned dependency makes a green CI leg unreproducible,
# and because a previous incident in an upstream parent had find_package silently
# resolve a dependency out of an unrelated build tree. Bump a pin in its own commit.

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# ---------------------------------------------------------------- Tier 0 (always)
set(FUSE_FMT_TAG        "11.0.2")
set(FUSE_FMTLOG_TAG     "v2.2.1")
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
FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG ${FUSE_FMT_TAG}
  FIND_PACKAGE_ARGS ${FUSE_FMT_TAG} CONFIG)

# fmtlog — multi-threaded, low-latency logger over fmt. Per-thread SPSC queues and a
# single polling thread, so a Kokkos host-parallel region can log without a mutex.
# Constraints that are part of the contract, not incidental:
#   * NOT callable from device code. Device diagnostics use Kokkos::printf only.
#   * Requires fmtlog::poll() or startPollingThread(); fuse::initialize() starts it and
#     fuse::finalize() drains it, so no line is lost on a clean shutdown.
#   * First log from a thread allocates that thread's buffer: fuse::initialize()
#     pre-touches one buffer per host thread so no allocation happens after setup
#     (docs/CHARTER.md §5).
#   * Every line carries the MPI rank; cross-rank ordering is by timestamp, and a log
#     line is never evidence of a collective outcome (that is Epoch's job).
FetchContent_Declare(fmtlog
  GIT_REPOSITORY https://github.com/MengRao/fmtlog.git GIT_TAG ${FUSE_FMTLOG_TAG}
  GIT_SUBMODULES ""          # its submodules are a vendored fmt and two benchmark loggers; we use our own fmt
  GIT_SHALLOW ON
  FIND_PACKAGE_ARGS CONFIG)

FetchContent_Declare(Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git GIT_TAG ${FUSE_KOKKOS_TAG}
  FIND_PACKAGE_ARGS ${FUSE_KOKKOS_TAG} CONFIG)

# Linear-algebra backend: exactly one, chosen here, hidden behind fuse::linalg.
# Neither backend's types may appear in a public header (docs/CHARTER.md §7).
set(FUSE_LINALG_BACKEND "eigen" CACHE STRING "kokkoskernels | eigen")
set_property(CACHE FUSE_LINALG_BACKEND PROPERTY STRINGS kokkoskernels eigen)

if(FUSE_LINALG_BACKEND STREQUAL "kokkoskernels")
  FetchContent_Declare(KokkosKernels
    GIT_REPOSITORY https://github.com/kokkos/kokkos-kernels.git GIT_TAG ${FUSE_KOKKOSKERNELS_TAG}
    FIND_PACKAGE_ARGS ${FUSE_KOKKOSKERNELS_TAG} CONFIG)
elseif(FUSE_LINALG_BACKEND STREQUAL "eigen")
  # Eigen's version file is SameMajorVersion, so asking for 3.4.0 rejects an
  # installed 5.0 outright and forces a fetch. The facade uses only dense QR,
  # small least-squares and sparse products, whose API is identical across
  # 3.4 and 5.x, so both majors are accepted from the system; the fetch
  # fallback pins 3.4.0 because that is the version the oracle tests were
  # written against.
  FetchContent_Declare(Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git GIT_TAG ${FUSE_EIGEN_TAG}
    GIT_SHALLOW ON
    FIND_PACKAGE_ARGS NAMES Eigen3 CONFIG)
else()
  message(FATAL_ERROR "FUSE_LINALG_BACKEND must be kokkoskernels or eigen, got '${FUSE_LINALG_BACKEND}'")
endif()

if(FUSE_ENABLE_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG ${FUSE_GTEST_TAG}
    FIND_PACKAGE_ARGS ${FUSE_GTEST_VERSION} NAMES GTest CONFIG)
endif()

macro(fuse_resolve_dependencies)
  set(_fuse_deps fmt Kokkos)
  if(FUSE_LINALG_BACKEND STREQUAL "kokkoskernels")
    list(APPEND _fuse_deps KokkosKernels)
  else()
    list(APPEND _fuse_deps Eigen3)
  endif()
  if(FUSE_ENABLE_TESTS)
    list(APPEND _fuse_deps googletest)
  endif()
  FetchContent_MakeAvailable(${_fuse_deps})
  # find_package sets Kokkos_VERSION; a FetchContent build does not, and its
  # project() scope is not ours. Fall back to the tag so config.hpp can record
  # what was built and the header/library consistency test has a value.
  if(NOT Kokkos_VERSION)
    set(Kokkos_VERSION "${FUSE_KOKKOS_TAG}")
  endif()
  message(STATUS "fuse: Kokkos ${Kokkos_VERSION} (${Kokkos_DIR})")

  if(FUSE_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX)
  endif()
  if(FUSE_ENABLE_MOAB)
    find_package(MOAB REQUIRED)     # TempestRemap arrives through MOAB's own configuration
  endif()
  if(FUSE_ENABLE_PETSC)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET PETSc)
  endif()
  if(FUSE_ENABLE_LOWFIVE)
    find_package(HDF5 REQUIRED COMPONENTS C)   # must be a SHARED build; LowFive dlopens the VOL
    find_package(LowFive REQUIRED)
  endif()

  # Record what was actually resolved. A CI leg that cannot say which versions it
  # used cannot reproduce its own result.
  file(WRITE "${CMAKE_BINARY_DIR}/fuse-resolved-dependencies.txt"
       "linalg=${FUSE_LINALG_BACKEND}\nfmt=${FUSE_FMT_TAG}\nfmtlog=${FUSE_FMTLOG_TAG}\n"
       "kokkos=${FUSE_KOKKOS_TAG}\nmpi=${FUSE_ENABLE_MPI}\nmoab=${FUSE_ENABLE_MOAB}\n"
       "petsc=${FUSE_ENABLE_PETSC}\nlowfive=${FUSE_ENABLE_LOWFIVE}\n")
endmacro()

# The logger is resolved by the first layer that logs (runtime, S4), not at
# Tier 0's entry point: base/model/ir/schedule never log, and a dependency no
# target links is a fetch for nothing. Header-only on top of OUR fmt, so there
# is exactly one fmt in the process.
macro(fuse_resolve_logging)
  FetchContent_MakeAvailable(fmtlog)
  if(NOT TARGET fuse::fmtlog)
    add_library(fuse-fmtlog INTERFACE)
    add_library(fuse::fmtlog ALIAS fuse-fmtlog)
    target_include_directories(fuse-fmtlog INTERFACE ${fmtlog_SOURCE_DIR})
    target_compile_definitions(fuse-fmtlog INTERFACE FMTLOG_HEADER_ONLY)
    target_link_libraries(fuse-fmtlog INTERFACE fmt::fmt)
  endif()
endmacro()
