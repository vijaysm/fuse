# Test registration. Every GoogleTest binary links the harness in test/support,
# which reduces failure status across the communicator (docs/dev/testing.md).
# Tests that still carry their upstream custom main() -- the rename-only state
# of a port -- are registered as plain executables until their conversion
# commit lands. There is no third kind.

add_library(fuse_test_main STATIC support/fuse_test_main.cpp)
target_include_directories(fuse_test_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/support)
target_link_libraries(fuse_test_main PUBLIC GTest::gtest fuse::base)
if(FUSE_ENABLE_MPI)
  target_link_libraries(fuse_test_main PUBLIC MPI::MPI_CXX)
endif()

# fuse_add_gtest(<layer> <name> [LIBS ...]) -> test/<layer>/<name>_test.cpp
function(fuse_add_gtest layer name)
  cmake_parse_arguments(ARG "" "" "LIBS" ${ARGN})
  set(target fuse_${layer}_${name}_test)
  add_executable(${target} ${layer}/${name}_test.cpp)
  target_link_libraries(${target} PRIVATE fuse_test_main fuse::${layer} ${ARG_LIBS})
  add_test(NAME ${layer}.${name} COMMAND ${target})
  set_tests_properties(${layer}.${name} PROPERTIES LABELS "${layer};gtest")
endfunction()

# fuse_add_upstream_main_test(<layer> <name>) -> same path, own main(), no harness.
function(fuse_add_upstream_main_test layer name)
  set(target fuse_${layer}_${name}_test)
  add_executable(${target} ${layer}/${name}_test.cpp)
  target_link_libraries(${target} PRIVATE fuse::${layer})
  add_test(NAME ${layer}.${name} COMMAND ${target})
  set_tests_properties(${layer}.${name} PROPERTIES LABELS "${layer};upstream-main")
endfunction()

# ---------------------------------------------------------------- base
foreach(t config rational time_level result ids units error version bytes)
  fuse_add_gtest(base ${t})
endforeach()
