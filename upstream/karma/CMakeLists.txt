## KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
##
## Minimal build. Dependencies are expected to live under a single PETSc
## prefix built with --download-moab, so PETSc, MOAB, HDF5 and MPI are all
## discovered from one place:
##
##   cmake -B build -DCMAKE_PREFIX_PATH=/opt/petsc/download-moab
##   cmake --build build
##   ctest --test-dir build

cmake_minimum_required(VERSION 3.20)

# C is enabled for the stable ABI: its headers are compiled as C11 in the
# conformance tests, and the install-tree consumer is a pure-C project. MPI's C
# component is deliberately NOT required - the ABI carries a framework-owned
# communicator token and no MPI type, so a C consumer needs no MPI at all.
project(KARMA VERSION 0.0.5 LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(KARMA_ENABLE_SHARED     "Build KARMA as a shared library"       ON)
option(KARMA_ENABLE_EXCEPTIONS "Throw C++ exceptions on errors"        ON)
option(KARMA_ENABLE_TRACEFILES "Write stack traces on errors"          ON)
option(KARMA_ENABLE_LIBMESH    "Build the libMesh-dependent sources"   OFF)
option(KARMA_ENABLE_TESTS      "Build the unit tests"                  ON)
option(KARMA_ENABLE_EXAMPLES  "Build the optional examples"            OFF)
option(KARMA_ENABLE_MOAB_COUPLER "Probe and test MOAB Coupler availability" ON)
option(KARMA_ENABLE_ARBORX "Enable the ArborX/Kokkos locator backend" OFF)
set(KARMA_ARBORX_PREFIX "" CACHE PATH "ArborX installation prefix")
set(KARMA_KOKKOS_PREFIX "" CACHE PATH "Kokkos installation prefix")

## ---------------------------------------------------------------------------
## Dependencies
## ---------------------------------------------------------------------------

find_package(PkgConfig REQUIRED)
pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
set(KARMA_HAVE_PETSC ON)
include(PetscScalarProbe)
unset(KARMA_PETSC_CONFIG_HEADER CACHE)
find_file(KARMA_PETSC_CONFIG_HEADER petscconf.h
  PATHS ${PETSC_INCLUDE_DIRS}
  NO_DEFAULT_PATH REQUIRED)
karma_parse_petsc_scalar_config("${KARMA_PETSC_CONFIG_HEADER}" KARMA_PETSC_SCALAR_MODE)
karma_validate_petsc_scalar_mode("${KARMA_PETSC_SCALAR_MODE}" _karma_scalar_error)
if(_karma_scalar_error)
  message(FATAL_ERROR "Configured PETSc is unsupported: ${_karma_scalar_error}")
endif()
set(KARMA_PETSC_CANDIDATE_PKGCONFIG_DIRS "" CACHE STRING
    "Additional PETSc pkg-config directories to inventory without affecting dependency discovery")
karma_collect_petsc_candidate_dirs(KARMA_PETSC_CANDIDATE_DIRS
  EXPLICIT_DIRS ${KARMA_PETSC_CANDIDATE_PKGCONFIG_DIRS})
karma_probe_petsc_candidates(KARMA_PETSC_CANDIDATE_REPORTS
  ${KARMA_PETSC_CANDIDATE_DIRS})
karma_summarize_petsc_variants(KARMA_PETSC_VARIANT_SUMMARIES
  "${KARMA_PETSC_SCALAR_MODE}" ${KARMA_PETSC_CANDIDATE_REPORTS})
set(KARMA_PETSC_VARIANT_PKGCONFIG_DIR "" CACHE PATH
    "Discovered compatible PETSc pkg-config directory for the opt-in variant test target")
if(KARMA_PETSC_VARIANT_PKGCONFIG_DIR)
  cmake_path(ABSOLUTE_PATH KARMA_PETSC_VARIANT_PKGCONFIG_DIR NORMALIZE
             OUTPUT_VARIABLE _karma_variant_pc_dir)
  if(NOT _karma_variant_pc_dir IN_LIST KARMA_PETSC_CANDIDATE_DIRS)
    message(FATAL_ERROR
      "KARMA_PETSC_VARIANT_PKGCONFIG_DIR must be one of the discovered candidates: ${KARMA_PETSC_CANDIDATE_DIRS}")
  endif()
  karma_probe_petsc_candidates(_karma_variant_report "${_karma_variant_pc_dir}")
  karma_validate_petsc_candidate_report("${_karma_variant_report}" _karma_variant_error)
  if(_karma_variant_error)
    message(FATAL_ERROR
      "KARMA_PETSC_VARIANT_PKGCONFIG_DIR is not testable: ${_karma_variant_error}. Candidate: ${_karma_variant_report}")
  endif()
  _karma_read_pc_variable("${_karma_variant_pc_dir}/PETSc.pc" prefix _karma_variant_prefix)
  set(_karma_variant_pkg_config_dirs "${_karma_variant_pc_dir}")
  if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
    cmake_path(CONVERT "$ENV{PKG_CONFIG_PATH}" TO_CMAKE_PATH_LIST
               _karma_inherited_pkg_config_dirs NORMALIZE)
    list(APPEND _karma_variant_pkg_config_dirs ${_karma_inherited_pkg_config_dirs})
  endif()
  cmake_path(CONVERT "${_karma_variant_pkg_config_dirs}"
             TO_NATIVE_PATH_LIST _karma_variant_pkg_config_path)
  add_custom_target(karma_test_petsc_variant
    COMMAND ${CMAKE_COMMAND} -E env
            "PKG_CONFIG_PATH=${_karma_variant_pkg_config_path}"
            ${CMAKE_COMMAND} -S "${CMAKE_SOURCE_DIR}"
            -B "${CMAKE_BINARY_DIR}/petsc-variant"
            -DKARMA_PETSC_VARIANT_PKGCONFIG_DIR=
            -DKARMA_ENABLE_TESTS=ON
            "-DCMAKE_PREFIX_PATH=${_karma_variant_prefix}"
    COMMAND ${CMAKE_COMMAND} -E env
            "PKG_CONFIG_PATH=${_karma_variant_pkg_config_path}"
            ${CMAKE_COMMAND} --build "${CMAKE_BINARY_DIR}/petsc-variant"
    COMMAND ${CMAKE_COMMAND} -E env
            "PKG_CONFIG_PATH=${_karma_variant_pkg_config_path}"
            ${CMAKE_CTEST_COMMAND} --test-dir "${CMAKE_BINARY_DIR}/petsc-variant"
            --output-on-failure
    USES_TERMINAL
    COMMENT "Configure, build, and test KARMA with ${_karma_variant_pc_dir}")
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
set(KARMA_HAVE_MPI ON)

# utils/lapack_utils.C calls dgels/dgelss/dgelsy directly.
find_package(LAPACK REQUIRED)

# MOAB ships a pkg-config file alongside PETSc when PETSc is configured with
# --download-moab, so the same prefix satisfies both.
pkg_check_modules(MOAB IMPORTED_TARGET MOAB)
if(MOAB_FOUND)
	set(KARMA_HAVE_MOAB ON)

	# MOAB.pc records the prefix MOAB was *built* in, which for a
	# --download-moab PETSc is a scratch directory inside the PETSc source
	# tree.  Prefer the copy installed next to PETSc so that the link does
	# not break when that scratch tree is cleaned.
	set(_moab_installed "${PETSC_PREFIX}/lib/libMOAB${CMAKE_SHARED_LIBRARY_SUFFIX}")
	if(EXISTS "${_moab_installed}" AND NOT MOAB_PREFIX STREQUAL PETSC_PREFIX)
		message(STATUS "MOAB: preferring ${PETSC_PREFIX} over build tree ${MOAB_PREFIX}")
		# pkg-config resolves -lMOAB to an absolute path in the build tree, so
		# the interface library has to be replaced outright, not just steered
		# with extra link directories.
		set_target_properties(PkgConfig::MOAB PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "${PETSC_PREFIX}/include"
			INTERFACE_LINK_LIBRARIES      "${_moab_installed}")
		set(MOAB_PREFIX "${PETSC_PREFIX}")
	endif()
endif()

find_path(EIGEN_INCLUDE_DIR
	NAMES Eigen/Dense
	HINTS ${PETSC_INCLUDE_DIRS} /opt/eigen3
	PATH_SUFFIXES eigen3)
if(EIGEN_INCLUDE_DIR)
	set(KARMA_HAVE_EIGEN ON)
endif()

if(KARMA_ENABLE_LIBMESH)
	find_package(LibMesh REQUIRED)
	set(KARMA_HAVE_LIBMESH ON)
endif()

## ---------------------------------------------------------------------------
## Platform feature checks
## ---------------------------------------------------------------------------

include(CheckIncludeFileCXX)
include(CheckCXXSourceCompiles)
include(CheckMoabCoupler)
karma_check_moab_coupler()
include(CheckArborX)
karma_check_arborx()

foreach(hdr csignal locale sstream)
	string(TOUPPER "${hdr}" _up)
	check_include_file_cxx("${hdr}" KARMA_HAVE_${_up})
endforeach()
# <sstream> is spelled KARMA_HAVE_SSTREAM by the sources.
set(KARMA_HAVE_SSTREAM ${KARMA_HAVE_SSTREAM})

foreach(hdr dlfcn.h getopt.h inttypes.h stdint.h stdlib.h string.h unistd.h
            sys/stat.h sys/types.h)
	string(TOUPPER "${hdr}" _up)
	string(REGEX REPLACE "[./]" "_" _up "${_up}")
	check_include_file_cxx("${hdr}" KARMA_HAVE_${_up})
endforeach()

check_cxx_source_compiles("
	#include <execinfo.h>
	int main() { void *a[4]; backtrace(a, 4); return 0; }"
	KARMA_HAVE_GLIBC_BACKTRACE)

check_cxx_source_compiles("
	#include <cxxabi.h>
	int main() { int s; abi::__cxa_demangle(\"x\", 0, 0, &s); return 0; }"
	KARMA_HAVE_GCC_ABI_DEMANGLE)

configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake/karma_config.h.in"
	"${CMAKE_CURRENT_BINARY_DIR}/include/karma_config.h"
	@ONLY)

## ---------------------------------------------------------------------------
## Library
## ---------------------------------------------------------------------------

# Registration policy for every test in test/ and examples/ - private
# working directories and honest PROCESSORS counts. Included here rather
# than in either subdirectory because CMake functions are global and the
# override must exist before the first add_test call in any of them.
include("${CMAKE_SOURCE_DIR}/cmake/karma_test_helpers.cmake")

add_subdirectory(src)

if(KARMA_ENABLE_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()

if(KARMA_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

## ---------------------------------------------------------------------------
## Summary
## ---------------------------------------------------------------------------

message(STATUS "")
message(STATUS "KARMA ${PROJECT_VERSION} configuration")
message(STATUS "  build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  PETSc         : ${PETSC_PREFIX} (${PETSC_VERSION}, ${KARMA_PETSC_SCALAR_MODE}, configured build)")
if(KARMA_PETSC_CANDIDATE_REPORTS)
  foreach(report IN LISTS KARMA_PETSC_CANDIDATE_REPORTS)
    message(STATUS "  PETSc probe   : ${report}")
  endforeach()
else()
  message(STATUS "  PETSc probes  : unavailable in standard or explicit search inputs")
endif()
foreach(summary IN LISTS KARMA_PETSC_VARIANT_SUMMARIES)
  message(STATUS "  PETSc variant : ${summary}")
endforeach()
message(STATUS "  MPI           : ${MPI_CXX_COMPILER}")
if(KARMA_HAVE_MOAB)
	message(STATUS "  MOAB          : ${MOAB_PREFIX} (${MOAB_VERSION})")
else()
	message(STATUS "  MOAB          : not found")
endif()
if(KARMA_HAVE_MOAB_COUPLER)
  message(STATUS "  MOAB Coupler  : ${KARMA_MOAB_COUPLER_LIBRARY}")
else()
  message(STATUS "  MOAB Coupler  : unavailable")
endif()
if(KARMA_HAVE_EIGEN)
	message(STATUS "  Eigen         : ${EIGEN_INCLUDE_DIR}")
else()
	message(STATUS "  Eigen         : not found")
endif()
message(STATUS "  libMesh       : ${KARMA_ENABLE_LIBMESH}")
message(STATUS "  sources built : ${KARMA_SOURCE_COUNT} of ${KARMA_SOURCE_TOTAL}")
message(STATUS "")
