cmake_minimum_required(VERSION 3.20)
# C is enabled for one reason: the C ABI's test translation unit
# (test/unit/capi_test.c) is compiled by a C compiler. That is the only thing
# that actually proves include/soma/soma_c.h is C-compatible -- compiling it
# as C++ proves nothing, since every C++-ism it must not contain compiles
# there.
project(SOMA VERSION 0.1.0 LANGUAGES CXX C)

# FetchContent's FIND_PACKAGE_ARGS lets an already-installed copy of a
# dependency satisfy the request instead of a network fetch. Without these
# two settings, find_package() also consults the CMake user/system package
# registry (~/.cmake/packages/<Pkg>/...), which can silently point at a
# build tree from an unrelated project on this machine (observed: an
# unrelated project's yaml-cpp build directory was resolved here, with no
# warning that it, rather than the pinned FetchContent version, was linked).
# Disabling the registry, combined with a version constraint on each
# FIND_PACKAGE_ARGS call below, keeps "reuse a correct installed copy"
# working while making a version-mismatched or registry-injected copy fall
# back to the pinned fetch instead of being silently accepted.
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SomaOptions)

# Optional dependencies. Everything below is OFF by default: the core build
# resolves zero external packages, so `cmake --preset debug` works on a bare
# machine. SOMA_HAVE_* are capability facts discovered here; the rest of the
# tree keys off those rather than probing MOAB directly.
if(SOMA_ENABLE_MOAB)
  include(FindSomaMOAB)
  if(NOT SOMA_HAVE_MOAB)
    message(FATAL_ERROR
      "SOMA_ENABLE_MOAB=ON but MOAB was not found. Set CMAKE_PREFIX_PATH to a "
      "MOAB prefix, e.g. -DCMAKE_PREFIX_PATH=/opt/moab-tempest")
  endif()
endif()

if(SOMA_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(SOMA_HAVE_MPI ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# LowFive is opt-in and OFF by default, so the dependency-free core build
# never pays for this probe. When it is requested but unusable, configuration
# continues with SOMA_HAVE_LOWFIVE OFF rather than failing: only Demo B needs
# it, and the find module explains what is wrong.
set(SOMA_HAVE_LOWFIVE OFF)
if(SOMA_ENABLE_LOWFIVE)
  find_package(SomaLowFive)
endif()

# A Python extension module is a shared object, and every static library that
# ends up inside one must be position-independent. On ELF targets linking
# non-PIC soma_core (or the yaml-cpp it pulls in) into _soma.so fails outright
# with "relocation R_X86_64_32S against ... can not be used when making a
# shared object". Set globally rather than per-target because the property has
# to reach yaml-cpp and any other FetchContent dependency too, and scoped
# inside this `if` so the default build -- where SOMA_ENABLE_PYTHON is OFF --
# produces byte-identical objects to before.
if(SOMA_ENABLE_PYTHON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# Tier A/B/C are header-only, so the base target is INTERFACE.
# Phase 4 adds a compiled `soma_core` target alongside it.
add_library(soma INTERFACE)
target_include_directories(soma INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(soma INTERFACE cxx_std_20)

add_subdirectory(src)

# The Fortran shim, spec 6 mode 2. OFF by default, so a machine with no
# Fortran compiler configures exactly as before. Added before test/ because
# the Fortran test executable needs both the language enabled and the
# soma_fortran target to exist.
#
# enable_language() is here rather than in fortran/CMakeLists.txt: called from
# a subdirectory it sets CMAKE_Fortran_COMPILE_OBJECT and friends in that
# directory's scope only, and the Fortran test executable lives in the sibling
# test/ directory, where the generate step then fails with "required internal
# CMake variable not set: CMAKE_Fortran_COMPILE_OBJECT" (observed with CMake
# 4.3.2). At top level the variables are visible to every subdirectory.
if(SOMA_ENABLE_FORTRAN)
  enable_language(Fortran)

  # Sanitizer runtimes do not mix. When the Fortran and C/C++ compilers come
  # from different vendors -- the case here is Apple clang plus a Homebrew
  # gfortran -- CMake's Fortran ABI probe runs with CMAKE_EXE_LINKER_FLAGS
  # applied, so a `-fsanitize=...` in that variable makes GCC's libasan and
  # libubsan part of CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES. They are then
  # linked alongside clang's libclang_rt.asan into the same executable, and
  # the program dies before main() with "Interceptors are not working. This
  # may be because AddressSanitizer is loaded too late" (observed with the
  # asan preset plus -DSOMA_ENABLE_FORTRAN=ON).
  #
  # The Fortran translation units are not instrumented -- nothing adds
  # -fsanitize to CMAKE_Fortran_FLAGS -- so they have no use for GCC's
  # runtime. Dropping it leaves exactly one sanitizer runtime, the C++ one,
  # which is the one doing the work. A same-vendor toolchain is left alone.
  if(NOT CMAKE_Fortran_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID)
    list(REMOVE_ITEM CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES
         asan ubsan tsan lsan msan)
  endif()

  add_subdirectory(fortran)
endif()

if(SOMA_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# After the tests block, not after add_subdirectory(src): examples/ registers
# its demo with add_test(), and enable_testing() must have run in this
# directory first or the CTestTestfile.cmake for examples/ is never generated
# and the example silently stops being run.
add_subdirectory(examples)

# The pybind11 front-end, spec 6 mode 1 for in-process Python surrogates. OFF
# by default, so a machine without Python development headers configures
# exactly as before and the core build still resolves zero external packages.
#
# Added AFTER the test/ block on purpose: python/CMakeLists.txt registers a
# ctest entry, and enable_testing() is called there. Adding it earlier would
# leave the Python test silently unregistered on a tests-enabled build, which
# is the failure mode where a binding ships without ever being run.
if(SOMA_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

soma_print_summary()
