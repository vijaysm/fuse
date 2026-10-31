# physics-kit/test/ -- the skeleton's own test registration.
#
# Calls test/CMakeLists.txt's own fuse_add_gtest directly, with its new
# SRC_DIR argument (see that function's header comment for why one is
# needed here and not in test/partition/, the only other caller outside
# test/<layer>/): fuse_add_gtest is defined while test/CMakeLists.txt
# executes, and is visible here because CMake functions, unlike variables,
# are not scoped to the directory that defines them -- only ORDER matters,
# and root CMakeLists.txt's own add_subdirectory(physics-kit) runs after
# its FUSE_ENABLE_TESTS block's add_subdirectory(test) for exactly this
# reason.
#
# NO_LAYER_LIB + explicit LIBS: the ctest name the task names is
# "physics.skeleton" (fuse::physics is the component's NAMESPACE), but the
# CMake target alias is "fuse::physics_kit" (the task's own name for it) --
# the two are deliberately different, so fuse_add_gtest's default
# "fuse::${layer}" link (which would look for a nonexistent "fuse::physics"
# target) is turned off and the real target named explicitly.
#
# These tests are NOT swept by test/CMakeLists.txt's own end-of-file audit
# (file(GLOB_RECURSE ... ${CMAKE_CURRENT_SOURCE_DIR}/*/*_test.cpp) globs
# relative to test/ and never descends into physics-kit/), so this file
# runs the same audit over its own directory at its end (added 2026-10-31,
# when the directory grew from one test source to ten).
fuse_add_gtest(physics skeleton SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)

# S8.3 wave 1 (numerics/fe, numerics/spatial): the five test rows named in
# docs/dev/physics-kit-disposition.md's porting scope, plus the spatial
# layer's own domain/mesh tests. Same NO_LAYER_LIB + explicit LIBS pattern
# as physics.skeleton above, for the same reason (fuse::physics_kit is the
# CMake target; fuse::physics is only the C++ namespace).
fuse_add_gtest(physics fe_reference_topology SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe_quadrature SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe_basis SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe_geometry_map SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics fe_face SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics spatial_mesh SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)
fuse_add_gtest(physics spatial_domain SRC_DIR . NO_LAYER_LIB LIBS fuse::physics_kit)

# The collective mesh contract over a real communicator (S8.3 wave 1b): owned
# counts sum to the global total and a ghost vertex's global id agrees with its
# owner's; refusals through fuse::agree_outcome (D97). Registered HERE, where
# fuse::physics_kit exists, rather than compiling physics-kit sources into a
# fuse-side test executable; host-only (the sandbox cannot launch MPI).
fuse_tests_gated_by(FUSE_ENABLE_MPI physics_mesh_mpi_test.cpp)
if(FUSE_ENABLE_MPI)
  fuse_add_mpi_test(parallel physics_mesh_mpi SRC_DIR . RANKS 2 3 LIBS fuse::physics_kit fuse::field)
endif()

# The same audit test/CMakeLists.txt runs over test/<layer>/: a *_test.cpp in
# this directory that is neither registered above nor declared gated would
# run nowhere, silently. fuse_add_gtest / fuse_add_mpi_test record the
# relative source path ("./<name>_test.cpp" with SRC_DIR .), and
# fuse_tests_gated_by records what it is given, so both spellings are
# accepted here.
file(GLOB _fuse_physics_test_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/*_test.cpp)
get_property(_fuse_physics_registered GLOBAL PROPERTY FUSE_REGISTERED_TEST_SOURCES)
get_property(_fuse_physics_gated GLOBAL PROPERTY FUSE_GATED_TEST_SOURCES)
set(_fuse_physics_unregistered "")
foreach(f IN LISTS _fuse_physics_test_files)
  if(NOT f IN_LIST _fuse_physics_registered AND NOT "./${f}" IN_LIST _fuse_physics_registered
     AND NOT f IN_LIST _fuse_physics_gated AND NOT "./${f}" IN_LIST _fuse_physics_gated)
    list(APPEND _fuse_physics_unregistered ${f})
  endif()
endforeach()
if(_fuse_physics_unregistered)
  list(JOIN _fuse_physics_unregistered "\n    " _fuse_physics_unregistered_text)
  message(FATAL_ERROR "fuse physics-kit: test source(s) registered nowhere, so they would run nowhere:\n"
                      "    ${_fuse_physics_unregistered_text}\n"
                      "  Register each with fuse_add_gtest / fuse_add_mpi_test (SRC_DIR .), or declare it "
                      "gated with fuse_tests_gated_by(<option> <file>).")
endif()
list(LENGTH _fuse_physics_test_files _n_physics_files)
message(STATUS "fuse physics-kit: ${_n_physics_files} test source(s) on disk, every one registered or gated")

