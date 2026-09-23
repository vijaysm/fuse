# The two examples named in S4's exit criteria, ported from the coupler parent.
#
# They are EXAMPLES, and they are also the only code in the tree that drives
# the library the way a user would -- building a graph with the fluent API,
# lowering it, and running it -- rather than assembling a fixture. That makes
# them worth running rather than merely compiling.
#
# add_test() here works only because the root CMakeLists.txt adds examples/
# AFTER enable_testing(); added before it, no CTestTestfile.cmake is generated
# for this directory and the examples build but are never run. The parent
# recorded that trap and it applies unchanged.

add_executable(fuse_example_mms_pair mms_pair/main.cpp)
target_link_libraries(fuse_example_mms_pair PRIVATE fuse::runtime fuse::backend)
target_include_directories(fuse_example_mms_pair PRIVATE mms_pair)

add_executable(fuse_example_dn_heat dn_heat/main.cpp)
target_link_libraries(fuse_example_dn_heat PRIVATE fuse::runtime fuse::backend)
target_include_directories(fuse_example_dn_heat PRIVATE dn_heat)

if(FUSE_ENABLE_TESTS)
  # Gated on the PHYSICS, not on "did not crash". mms_pair exits non-zero when
  # the coupled answer misses its analytic gate, when no conservation check
  # ran, or when the coupled error is not decisively better than a run with no
  # exchange. dn_heat exits non-zero when the converged answer misses the
  # analytic two-layer-wall solution, when the implicit run is not decisively
  # better than an explicit one or than a run whose exchange has been severed,
  # when the accelerators disagree about the fixed point, when IQN-ILS fails to
  # beat Picard, when a rollback was never verified, or when two runs of the
  # same configuration differ.
  add_test(NAME example.mms_pair COMMAND fuse_example_mms_pair)
  add_test(NAME example.dn_heat  COMMAND fuse_example_dn_heat)

  # The --dump path has its own failure mode: an empty dump still exits 0, so
  # these assert on the output rather than the status.
  add_test(NAME example.mms_pair.dump COMMAND fuse_example_mms_pair --dump)
  add_test(NAME example.dn_heat.dump  COMMAND fuse_example_dn_heat --dump)
  set_tests_properties(example.mms_pair.dump example.dn_heat.dump PROPERTIES
    PASS_REGULAR_EXPRESSION "=== IR ===.*=== PLAN ===")
  set_tests_properties(example.mms_pair example.dn_heat
    example.mms_pair.dump example.dn_heat.dump PROPERTIES LABELS "example")
endif()

# ---------------------------------------------------------------------------
# PE-layout driver (plan-track-d). MPI-only: it links fuse::transport for
# MpiTransport and needs mpiexec to mean anything, so the whole block is
# gated on FUSE_ENABLE_MPI, exactly as test/CMakeLists.txt gates its own
# parallel/ targets.
# ---------------------------------------------------------------------------
if(FUSE_ENABLE_MPI)
  add_executable(fuse_example_layout_driver layout_driver/main.cpp)
  target_link_libraries(fuse_example_layout_driver PRIVATE
    fuse::runtime fuse::backend fuse::transport MPI::MPI_CXX)
  # PartitionedTestBackend (test/support/partitioned_test_backend.hpp) is the
  # only MeshBackend in the tree that gives each graph component its own
  # decomposition of a shared mesh -- see that header's own comment. Sharing
  # it here, rather than a second copy, is why this include directory reaches
  # into test/.
  target_include_directories(fuse_example_layout_driver PRIVATE
    ${CMAKE_SOURCE_DIR}/test/support)

  # fuse_add_mpi_test (test/CMakeLists.txt) always builds a NEW target from
  # test/<layer>/<name>_test.cpp -- one gtest binary per registration. This is
  # ONE binary, driven by different CLI arguments at each rank count and
  # placement, so the mpiexec line is written out directly instead.
  function(fuse_add_layout_test name ranks)
    cmake_parse_arguments(ARG "WILL_FAIL" "" "ARGS" ${ARGN})
    add_test(NAME ${name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                     $<TARGET_FILE:fuse_example_layout_driver> ${ARG_ARGS})
    set_tests_properties(${name} PROPERTIES TIMEOUT 120 LABELS "mpi;parallel;layout")
    if(ARG_WILL_FAIL)
      set_tests_properties(${name} PROPERTIES WILL_FAIL TRUE)
    endif()
  endfunction()

  # Identical, disjoint and partially-overlapping placements at 2, 3 and 4
  # ranks -- partial overlap needs >= 3 ranks (2 ranks can only be identical
  # or disjoint), and at 3 and 4 ranks one arrangement of each non-identical
  # shape puts comp2 on the LOWER range, not comp1.
  fuse_add_layout_test(layout.2.c1-0-0.c2-1-1 2 ARGS --comp1 0:0 --comp2 1:1 --check)
  fuse_add_layout_test(layout.2.c1-1-1.c2-0-0 2 ARGS --comp1 1:1 --comp2 0:0 --check)
  fuse_add_layout_test(layout.2.c1-0-1.c2-0-1 2 ARGS --comp1 0:1 --comp2 0:1 --check)

  fuse_add_layout_test(layout.3.c1-0-0.c2-1-2 3 ARGS --comp1 0:0 --comp2 1:2 --check)
  fuse_add_layout_test(layout.3.c1-0-1.c2-1-2 3 ARGS --comp1 0:1 --comp2 1:2 --check)
  fuse_add_layout_test(layout.3.c1-1-2.c2-0-1 3 ARGS --comp1 1:2 --comp2 0:1 --check)
  fuse_add_layout_test(layout.3.c1-0-2.c2-0-2 3 ARGS --comp1 0:2 --comp2 0:2 --check)

  fuse_add_layout_test(layout.4.c1-0-1.c2-2-3 4 ARGS --comp1 0:1 --comp2 2:3 --check)
  fuse_add_layout_test(layout.4.c1-0-2.c2-1-3 4 ARGS --comp1 0:2 --comp2 1:3 --check)
  fuse_add_layout_test(layout.4.c1-1-3.c2-0-2 4 ARGS --comp1 1:3 --comp2 0:2 --check)
  fuse_add_layout_test(layout.4.c1-0-3.c2-0-3 4 ARGS --comp1 0:3 --comp2 0:3 --check)

  # Deliberate refusals (docs/dev/testing.md: counts as coverage only when a
  # test makes the refusal fire). Run at 2 ranks -- the rank count is
  # incidental to what each of these is testing.
  fuse_add_layout_test(layout.refuse.min-nonzero 2 WILL_FAIL
    ARGS --comp1 1:1 --comp2 1:1)
  fuse_add_layout_test(layout.refuse.max-not-last 2 WILL_FAIL
    ARGS --comp1 0:0 --comp2 0:0)
  fuse_add_layout_test(layout.refuse.reversed-range 2 WILL_FAIL
    ARGS --comp1 1:0 --comp2 0:1)
  fuse_add_layout_test(layout.refuse.coupler 2 WILL_FAIL
    ARGS --comp1 0:1 --comp2 0:1 --coupler 0:1)
endif()
