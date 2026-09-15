add_executable(karma_gray_scott_operator_split gray_scott_operator_split.C)
target_link_libraries(karma_gray_scott_operator_split PRIVATE KARMA::karma)

if(KARMA_ENABLE_TESTS)
  set(_karma_gray_scott_smoke_output
      "${CMAKE_CURRENT_BINARY_DIR}/gray-scott-smoke-output")
  add_test(NAME karma_gray_scott_example_smoke_serial
    COMMAND $<TARGET_FILE:karma_gray_scott_operator_split>
            -gs_u_resolution 3,3 -gs_v_resolution 4,3
            -gs_dt 0.002 -gs_final_time 0.004
            -gs_checkpoint_cadence 1
            -gs_output_dir "${_karma_gray_scott_smoke_output}/serial")
  set_tests_properties(karma_gray_scott_example_smoke_serial PROPERTIES
    FIXTURES_SETUP karma_gray_scott_example_output
    RESOURCE_LOCK karma_moab_io_files TIMEOUT 60)

  add_test(NAME karma_gray_scott_example_bad_dt
    COMMAND $<TARGET_FILE:karma_gray_scott_operator_split> -gs_dt -0.01)
  set_tests_properties(karma_gray_scott_example_bad_dt PROPERTIES
    WILL_FAIL TRUE TIMEOUT 20)

  add_test(NAME karma_gray_scott_example_bad_scheme
    COMMAND $<TARGET_FILE:karma_gray_scott_operator_split> -gs_scheme invalid)
  set_tests_properties(karma_gray_scott_example_bad_scheme PROPERTIES
    WILL_FAIL TRUE TIMEOUT 20)

  if(MPIEXEC_EXECUTABLE)
    add_test(NAME karma_gray_scott_example_smoke_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_gray_scott_operator_split>
              -gs_u_resolution 3,3 -gs_v_resolution 4,3
              -gs_dt 0.002 -gs_final_time 0.004
              -gs_checkpoint_cadence 1
              -gs_output_dir "${_karma_gray_scott_smoke_output}/mpi")
    set_tests_properties(karma_gray_scott_example_smoke_mpi PROPERTIES
      RESOURCE_LOCK karma_moab_io_files TIMEOUT 60)

    add_test(NAME karma_gray_scott_example_asymmetric_invalid_option_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
              $<TARGET_FILE:karma_gray_scott_operator_split>
              -gs_checkpoint_cadence 0
              : ${MPIEXEC_NUMPROC_FLAG} 1
              $<TARGET_FILE:karma_gray_scott_operator_split>
              -gs_scheme invalid -gs_checkpoint_cadence 0)
    set_tests_properties(karma_gray_scott_example_asymmetric_invalid_option_mpi PROPERTIES
      WILL_FAIL TRUE TIMEOUT 20)

    add_test(NAME karma_gray_scott_example_asymmetric_valid_options_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
              $<TARGET_FILE:karma_gray_scott_operator_split>
              -gs_dt 0.002 -gs_final_time 0.004 -gs_checkpoint_cadence 0
              : ${MPIEXEC_NUMPROC_FLAG} 1
              $<TARGET_FILE:karma_gray_scott_operator_split>
              -gs_dt 0.001 -gs_final_time 0.004 -gs_checkpoint_cadence 0)
    set_tests_properties(karma_gray_scott_example_asymmetric_valid_options_mpi PROPERTIES
      WILL_FAIL TRUE TIMEOUT 20)

    add_test(NAME karma_gray_scott_example_rank_local_mesh_failure_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
              -wdir "${_karma_gray_scott_smoke_output}/serial"
              $<TARGET_FILE:karma_gray_scott_operator_split>
              -gs_u_mesh step-1-U.h5m -gs_checkpoint_cadence 0
              : ${MPIEXEC_NUMPROC_FLAG} 1
              -wdir "${CMAKE_CURRENT_BINARY_DIR}"
              $<TARGET_FILE:karma_gray_scott_operator_split>
              -gs_u_mesh step-1-U.h5m -gs_checkpoint_cadence 0)
    set_tests_properties(karma_gray_scott_example_rank_local_mesh_failure_mpi PROPERTIES
      FIXTURES_REQUIRED karma_gray_scott_example_output
      WILL_FAIL TRUE TIMEOUT 20)
  endif()
endif()
