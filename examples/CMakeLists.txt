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
# ---------------------------------------------------------------------------
# Gray-Scott (Time windows track): U and V on one structured grid, coupled
# implicitly at independent time steps through fuse::WindowCoupler. Needs
# Eigen3::Eigen directly (its Newton solves use Eigen::SparseLU) -- available
# because FUSE_LINALG_BACKEND=eigen is the default (cmake/FuseDependencies.cmake).
# ---------------------------------------------------------------------------
add_executable(fuse_example_gray_scott
  gray_scott/main.cpp)
target_link_libraries(fuse_example_gray_scott PRIVATE fuse::runtime fuse::backend Eigen3::Eigen)
target_include_directories(fuse_example_gray_scott PRIVATE gray_scott)

if(FUSE_ENABLE_TESTS)
  # Gated on the physics, same convention as mms_pair/dn_heat above:
  # --check exits non-zero if the m=1 windowed result misses the monolithic
  # reference (to a TIGHT, iteration-tolerance-level bound -- see the
  # comment at that check in main.cpp) or a concentration goes negative.
  # --dt 0.1 is not incidental: the m=1 windowed/monolithic mismatch is a
  # genuine order-2-in-dt effect (gray_scott_reference_test.cpp measures it
  # exactly), not merely the iteration residual, so it is checked at a dt
  # small enough for that order-2 term to already be below the tight bound,
  # rather than by loosening the bound. The --check block's own m=1
  # sub-run is internally fixed at 5 windows of --dt regardless of --ratio/
  # --t-end (main.cpp), so --dt is what this bound actually depends on;
  # measured AT THIS EXACT COMMAND LINE: diff 3.022e-08, bound 1.0e-6.
  add_test(NAME example.gray_scott COMMAND fuse_example_gray_scott --n 24 --ratio 2 --t-end 4 --dt 0.1 --check)
  set_tests_properties(example.gray_scott PROPERTIES LABELS "example")
endif()

if(FUSE_ENABLE_MPI)
  add_executable(fuse_example_layout_driver layout_driver/main.cpp)
  target_link_libraries(fuse_example_layout_driver PRIVATE
    fuse::runtime fuse::backend fuse::partition fuse::transport MPI::MPI_CXX)
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

  # --mesh/--migrate: comp2's mesh is comp1's, migrated with RCB onto
  # comp2's placement (Args::mesh_spec's own comment on what --check means
  # in this mode: cell count and measure conservation, not the field
  # channel). One box generator per rank count, on the same
  # disjoint/overlapping/identical shapes as the field-transfer rows above.
  fuse_add_layout_test(layout.migrate.2.c1-0-0.c2-1-1 2
    ARGS --comp1 0:0 --comp2 1:1 --mesh hexbox:4x4x4 --migrate --check)
  fuse_add_layout_test(layout.migrate.3.c1-0-1.c2-1-2 3
    ARGS --comp1 0:1 --comp2 1:2 --mesh hexbox:4x4x4 --migrate --check)
  fuse_add_layout_test(layout.migrate.4.c1-0-2.c2-1-3 4
    ARGS --comp1 0:2 --comp2 1:3 --mesh hexbox:4x4x4 --migrate --check)

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
