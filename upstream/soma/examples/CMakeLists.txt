# Demo A of spec 11 at the Phase 4 milestone.
add_executable(mms_pair mms_pair/main.cpp)
target_link_libraries(mms_pair PRIVATE soma soma_core)
target_include_directories(mms_pair PRIVATE mms_pair)
target_compile_options(mms_pair PRIVATE -Wall -Wextra -Wpedantic)

# An example nobody runs rots within a phase, so ctest runs it. The binary
# returns non-zero when the coupled answer misses its analytic gate, when no
# conservation check ran, or when the coupled error is not decisively better
# than a run with no exchange -- see examples/mms_pair/main.cpp.
if(SOMA_ENABLE_TESTS)
  add_test(NAME mms_pair_example COMMAND mms_pair)
  # The --dump path is a separate deliverable of the phase exit criteria and
  # has its own failure modes (an empty dump still exits 0 above).
  add_test(NAME mms_pair_example_dump COMMAND mms_pair --dump)
  set_tests_properties(mms_pair_example_dump PROPERTIES
    PASS_REGULAR_EXPRESSION "=== IR ===.*=== PLAN ===")
endif()

# Demo A of spec 11 at the Phase 7 milestone: the same demo made IMPLICIT.
# A Dirichlet-Neumann pair iterated inside a g.iterate({...}) region with a
# derived checkpoint set, a criterion covering both channels, and quasi-Newton
# acceleration.
add_executable(dn_heat dn_heat/main.cpp)
target_link_libraries(dn_heat PRIVATE soma soma_core)
target_include_directories(dn_heat PRIVATE dn_heat)
target_compile_options(dn_heat PRIVATE -Wall -Wextra -Wpedantic)

# Gated on the physics, not on "did not crash": the binary exits non-zero when
# the converged answer misses the analytic two-layer-wall solution, when the
# implicit run is not decisively better than an explicit one, when it is not
# decisively better than a run whose exchange has been severed (the
# measurement that proves the headline number moves when the coupling breaks),
# when the accelerators disagree about the fixed point, when IQN-ILS fails to
# beat Picard, when a rollback was never verified, or when two runs of the same
# configuration differ. See examples/dn_heat/main.cpp.
#
# add_test() here works only because the root CMakeLists.txt adds examples/
# AFTER enable_testing(); added before it, no CTestTestfile.cmake is generated
# for this directory and the example builds but is never run.
if(SOMA_ENABLE_TESTS)
  add_test(NAME dn_heat_example COMMAND dn_heat)
  # The dump path has its own failure modes: an empty dump still exits 0.
  add_test(NAME dn_heat_example_dump COMMAND dn_heat --dump)
  set_tests_properties(dn_heat_example_dump PROPERTIES
    PASS_REGULAR_EXPRESSION "=== IR ===.*=== PLAN ===")
endif()

# Demo B of spec 11, Task 8.5: the mms_pair physics of Task 4.7, unmodified,
# coupled through the LowFive HDF5-VOL transport instead of CopyTransport --
# real interception, in memory, zero bytes on disk. Gated on SOMA_HAVE_LOWFIVE
# like test/CMakeLists.txt gates lowfive_transport_test.cpp's HDF5 link: the
# three default legs (LowFive off) must not build or register this target at
# all, following the same conditional-registration idiom test/CMakeLists.txt
# uses for SOMA_ENABLE_MOAB.
#
# examples/lowfive_pair/main.cpp includes <hdf5.h> directly for the same
# reason test/unit/lowfive_transport_test.cpp does: proving "no file on disk"
# requires a CONTROL that is an "unmodified HDF5 component" speaking raw
# HDF5 with H5P_DEFAULT on the same path. src/CMakeLists.txt's dependency note
# ("the only translation unit in the project that may include <hdf5.h>")
# scopes to soma_core's own sources, not to consumer test/example code -- the
# existing test file is the precedent.
if(SOMA_HAVE_LOWFIVE)
  add_executable(lowfive_pair lowfive_pair/main.cpp)
  target_link_libraries(lowfive_pair PRIVATE soma soma_core soma::lowfive)
  target_include_directories(lowfive_pair PRIVATE lowfive_pair
    ${CMAKE_CURRENT_SOURCE_DIR}/mms_pair)
  target_compile_options(lowfive_pair PRIVATE -Wall -Wextra -Wpedantic)

  # Gated on the transport, not on "did not crash": the binary exits non-zero
  # when the coupled physics misses mms_pair's proven analytic gate, when no
  # file appears in the control leg (which would make every absence claim
  # below it vacuous), when a real file is found after the coupled run, when
  # LowFive's own after_dataset_write callback never fired, or when any of the
  # three sabotage legs fails to reproduce its expected failure. See
  # examples/lowfive_pair/main.cpp.
  if(SOMA_ENABLE_TESTS)
    add_test(NAME lowfive_pair_example COMMAND lowfive_pair)
  endif()
endif()
