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
