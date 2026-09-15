## Tests for the KARMA core library.
##
## The historical unit tests under test/unit/ are built on cpptest and on
## KARMAPackage, which lives in the PETScExt-dependent sources; they are not
## wired up here yet. smoke_test exercises what the library currently
## exports, and is the check that the build actually links and runs.

add_executable(karma_smoke_test smoke_test.C)
target_link_libraries(karma_smoke_test PRIVATE KARMA::karma)

add_test(NAME karma_smoke COMMAND karma_smoke_test)

add_executable(karma_semantic_version_test semantic_version_test.C)
target_link_libraries(karma_semantic_version_test PRIVATE KARMA::karma)
add_test(NAME karma_semantic_version COMMAND karma_semantic_version_test)
set_tests_properties(karma_semantic_version PROPERTIES TIMEOUT 20)

add_executable(karma_field_entry_encoding_test field_entry_encoding_test.C)
target_link_libraries(karma_field_entry_encoding_test PRIVATE KARMA::karma)
add_test(NAME karma_field_entry_encoding COMMAND karma_field_entry_encoding_test)
set_tests_properties(karma_field_entry_encoding PROPERTIES TIMEOUT 20)

add_executable(karma_field_entry_encoding_mpi_test field_entry_encoding_mpi_test.C)
target_link_libraries(karma_field_entry_encoding_mpi_test PRIVATE KARMA::karma)
if(MPIEXEC_EXECUTABLE)
  foreach(ranks 2 3 4 5)
    add_test(NAME karma_field_entry_encoding_reassociation_${ranks}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:karma_field_entry_encoding_mpi_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(karma_field_entry_encoding_reassociation_${ranks}_mpi PROPERTIES TIMEOUT 20)
  endforeach()
endif()

add_executable(karma_coupling_semantics_test coupling_semantics_test.C)
target_link_libraries(karma_coupling_semantics_test PRIVATE KARMA::karma)
add_test(NAME karma_coupling_semantics COMMAND karma_coupling_semantics_test)
set_tests_properties(karma_coupling_semantics PROPERTIES TIMEOUT 20)

# The stable C ABI. `karma_abi_layout_c_test` is compiled as C11 against the
# ABI headers alone: it would fail to build if a header leaked a C++ construct,
# and it links only karma_abi, never libkarma.
add_executable(karma_abi_layout_c_test abi_c11_conformance_test.c)
target_link_libraries(karma_abi_layout_c_test PRIVATE KARMA::karma_abi)
set_target_properties(karma_abi_layout_c_test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
add_test(NAME karma_abi_layout_c COMMAND karma_abi_layout_c_test)
set_tests_properties(karma_abi_layout_c PROPERTIES TIMEOUT 20)

# The ABI install check: install into a scratch prefix, then build and run a
# standalone pure-C project against it with one include path.
add_test(NAME karma_abi_install_consumer
  COMMAND ${CMAKE_COMMAND}
          -DKARMA_BUILD_DIR=${PROJECT_BINARY_DIR}
          -DKARMA_SOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DKARMA_SCRATCH=${PROJECT_BINARY_DIR}/abi-install-check
          -P ${PROJECT_SOURCE_DIR}/cmake/abi_install_consumer_check.cmake)
set_tests_properties(karma_abi_install_consumer PROPERTIES TIMEOUT 600)

add_executable(karma_language_adapter_schema_test language_adapter_schema_test.C)
target_link_libraries(karma_language_adapter_schema_test PRIVATE KARMA::karma_abi)
add_test(NAME karma_language_adapter_schema COMMAND karma_language_adapter_schema_test)
set_tests_properties(karma_language_adapter_schema PROPERTIES TIMEOUT 20)

# Asserts the exported set equals the documented export list. Nothing else
# would catch an accidentally exported C++ symbol.
add_test(NAME karma_abi_exports
  COMMAND ${CMAKE_COMMAND}
          -DKARMA_ABI_LIBRARY=$<TARGET_FILE:karma_abi>
          -DKARMA_ABI_EXPORT_LIST=${PROJECT_SOURCE_DIR}/src/abi/karma_exports.sym
          -P ${PROJECT_SOURCE_DIR}/cmake/abi_exports_check.cmake)
set_tests_properties(karma_abi_exports PROPERTIES TIMEOUT 60)

add_executable(karma_abi_negotiation_test abi_negotiation_test.C)
target_link_libraries(karma_abi_negotiation_test PRIVATE KARMA::karma_abi)
add_test(NAME karma_abi_negotiation COMMAND karma_abi_negotiation_test)
set_tests_properties(karma_abi_negotiation PROPERTIES TIMEOUT 20)

add_executable(karma_abi_layout_test abi_cxx_layout_test.C)
target_link_libraries(karma_abi_layout_test PRIVATE KARMA::karma_abi)
add_test(NAME karma_abi_layout COMMAND karma_abi_layout_test)
set_tests_properties(karma_abi_layout PROPERTIES TIMEOUT 20)

add_executable(karma_component_compliance_test component_compliance_test.C)
target_link_libraries(karma_component_compliance_test PRIVATE KARMA::karma KARMA::karma_abi)
add_test(NAME karma_component_compliance COMMAND karma_component_compliance_test)
set_tests_properties(karma_component_compliance PROPERTIES TIMEOUT 20)

add_executable(karma_byte_writer_test byte_writer_test.C)
target_link_libraries(karma_byte_writer_test PRIVATE KARMA::karma)
add_test(NAME karma_byte_writer COMMAND karma_byte_writer_test)
set_tests_properties(karma_byte_writer PROPERTIES TIMEOUT 20)

add_executable(karma_static_registry_test static_registry_test.C)
target_link_libraries(karma_static_registry_test PRIVATE KARMA::karma KARMA::karma_abi)
add_test(NAME karma_static_registry COMMAND karma_static_registry_test)
set_tests_properties(karma_static_registry PROPERTIES TIMEOUT 20)

add_executable(karma_coupling_compatibility_test coupling_compatibility_test.C)
target_link_libraries(karma_coupling_compatibility_test PRIVATE KARMA::karma)
add_test(NAME karma_coupling_compatibility COMMAND karma_coupling_compatibility_test)
set_tests_properties(karma_coupling_compatibility PROPERTIES TIMEOUT 20)

add_executable(karma_transform_inference_test transform_inference_test.C)
target_link_libraries(karma_transform_inference_test PRIVATE KARMA::karma)
add_test(NAME karma_transform_inference COMMAND karma_transform_inference_test)
set_tests_properties(karma_transform_inference PROPERTIES TIMEOUT 20)

add_executable(karma_collective_outcome_mpi_test collective_outcome_mpi_test.C)
target_link_libraries(karma_collective_outcome_mpi_test PRIVATE KARMA::karma)
if(MPIEXEC_EXECUTABLE)
  foreach(ranks 2 3 4)
    add_test(NAME karma_collective_outcome_reconcile_${ranks}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
              $<TARGET_FILE:karma_collective_outcome_mpi_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(karma_collective_outcome_reconcile_${ranks}_mpi PROPERTIES TIMEOUT 20)
  endforeach()
endif()

add_executable(karma_collective_outcome_test collective_outcome_test.C)
target_link_libraries(karma_collective_outcome_test PRIVATE KARMA::karma)
add_test(NAME karma_collective_outcome COMMAND karma_collective_outcome_test)
set_tests_properties(karma_collective_outcome PROPERTIES TIMEOUT 20)

add_executable(karma_field_transaction_test field_transaction_test.C)
target_link_libraries(karma_field_transaction_test PRIVATE KARMA::karma)
add_test(NAME karma_field_transaction COMMAND karma_field_transaction_test)
set_tests_properties(karma_field_transaction PROPERTIES TIMEOUT 20)

add_executable(karma_coupling_compatibility_oracle_test coupling_compatibility_oracle_test.C)
target_link_libraries(karma_coupling_compatibility_oracle_test PRIVATE KARMA::karma)
add_test(NAME karma_coupling_compatibility_oracle
  COMMAND karma_coupling_compatibility_oracle_test "${CMAKE_CURRENT_BINARY_DIR}/compatibility-serial")
set_tests_properties(karma_coupling_compatibility_oracle PROPERTIES TIMEOUT 60
  RESOURCE_LOCK karma_moab_io_files)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_coupling_compatibility_oracle_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:karma_coupling_compatibility_oracle_test> ${MPIEXEC_POSTFLAGS}
      "${CMAKE_CURRENT_BINARY_DIR}/compatibility-mpi")
  set_tests_properties(karma_coupling_compatibility_oracle_mpi PROPERTIES TIMEOUT 20
    RESOURCE_LOCK karma_moab_io_files)
endif()

add_executable(karma_operator_split_graph_test operator_split_graph_test.C)
target_link_libraries(karma_operator_split_graph_test PRIVATE KARMA::karma)
add_test(NAME karma_operator_split_graph COMMAND karma_operator_split_graph_test)

add_executable(karma_operator_split_introspection_test operator_split_introspection_test.C)
target_link_libraries(karma_operator_split_introspection_test PRIVATE KARMA::karma)
add_test(NAME karma_operator_split_introspection COMMAND karma_operator_split_introspection_test)

add_executable(karma_fixed_point_coupling_test fixed_point_coupling_test.C)
target_link_libraries(karma_fixed_point_coupling_test PRIVATE KARMA::karma)
add_test(NAME karma_fixed_point_coupling COMMAND karma_fixed_point_coupling_test)
set_tests_properties(karma_fixed_point_coupling PROPERTIES TIMEOUT 30)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_fixed_point_coupling_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_fixed_point_coupling_test>)
  set_tests_properties(karma_fixed_point_coupling_mpi PROPERTIES TIMEOUT 30)
  add_test(NAME karma_fixed_point_asymmetric_failure_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_fixed_point_coupling_test> --asymmetric-failure)
  set_tests_properties(karma_fixed_point_asymmetric_failure_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_fixed_point_asymmetric_metadata_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_fixed_point_coupling_test> --asymmetric-metadata)
  set_tests_properties(karma_fixed_point_asymmetric_metadata_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_fixed_point_asymmetric_options_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_fixed_point_coupling_test> --asymmetric-options)
  set_tests_properties(karma_fixed_point_asymmetric_options_mpi PROPERTIES TIMEOUT 20)
endif()

add_executable(karma_operator_split_scheduler_test operator_split_scheduler_test.C)
target_link_libraries(karma_operator_split_scheduler_test PRIVATE KARMA::karma)
add_test(NAME karma_operator_split_scheduler COMMAND karma_operator_split_scheduler_test)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_operator_split_scheduler_asymmetric_failure_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_operator_split_scheduler_test> --asymmetric-advance-failure)
  set_tests_properties(karma_operator_split_scheduler_asymmetric_failure_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_operator_split_scheduler_asymmetric_interval_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_operator_split_scheduler_test> --asymmetric-interval)
  set_tests_properties(karma_operator_split_scheduler_asymmetric_interval_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_operator_split_scheduler_asymmetric_graph_metadata_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_operator_split_scheduler_test> --asymmetric-graph-metadata)
  set_tests_properties(karma_operator_split_scheduler_asymmetric_graph_metadata_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_operator_split_scheduler_asymmetric_valid_graph_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_operator_split_scheduler_test> --asymmetric-valid-graph)
  set_tests_properties(karma_operator_split_scheduler_asymmetric_valid_graph_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_operator_split_scheduler_asymmetric_runtime_metadata_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_operator_split_scheduler_test> --asymmetric-runtime-metadata)
  set_tests_properties(karma_operator_split_scheduler_asymmetric_runtime_metadata_mpi PROPERTIES TIMEOUT 20)
endif()

add_executable(karma_operator_split_connection_test operator_split_connection_test.C)
target_link_libraries(karma_operator_split_connection_test PRIVATE KARMA::karma)
add_test(NAME karma_operator_split_connection COMMAND karma_operator_split_connection_test)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_operator_split_connection_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_operator_split_connection_test>)
  set_tests_properties(karma_operator_split_connection_mpi PROPERTIES TIMEOUT 60)
endif()

add_executable(karma_compiled_connection_test compiled_connection_test.C)
target_link_libraries(karma_compiled_connection_test PRIVATE KARMA::karma)
add_test(NAME karma_compiled_connection COMMAND karma_compiled_connection_test)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_compiled_connection_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_compiled_connection_test>)
  set_tests_properties(karma_compiled_connection_mpi PROPERTIES TIMEOUT 60)
endif()

add_executable(karma_field_transfer_contract_test field_transfer_contract_test.C)
set_property(TARGET karma_field_transfer_contract_test APPEND PROPERTY COMPILE_OPTIONS "-Werror")
target_link_libraries(karma_field_transfer_contract_test PRIVATE KARMA::karma)
add_test(NAME karma_field_transfer_contract COMMAND karma_field_transfer_contract_test)

add_executable(karma_transfer_map_test transfer_map_test.C)
target_link_libraries(karma_transfer_map_test PRIVATE KARMA::karma)
add_test(NAME karma_transfer_map COMMAND karma_transfer_map_test)

add_executable(karma_transfer_map_apply_test transfer_map_apply_test.C)
target_link_libraries(karma_transfer_map_apply_test PRIVATE KARMA::karma)
add_test(NAME karma_transfer_map_apply COMMAND karma_transfer_map_apply_test)

add_executable(karma_transfer_map_io_test transfer_map_io_test.C)
target_link_libraries(karma_transfer_map_io_test PRIVATE KARMA::karma)
add_test(NAME karma_transfer_map_io
  COMMAND $<TARGET_FILE:karma_transfer_map_io_test> "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_map_test.bin")
set_tests_properties(karma_transfer_map_io PROPERTIES RESOURCE_LOCK karma_transfer_map_files)

add_executable(karma_transfer_map_parallel_io_test transfer_map_parallel_io_test.C)
target_link_libraries(karma_transfer_map_parallel_io_test PRIVATE KARMA::karma)
if(MPIEXEC_EXECUTABLE)
  foreach(ranks 2 4 5)
    add_test(NAME karma_transfer_map_io_mpi_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_transfer_map_parallel_io_test>
              "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_map_parallel_${ranks}.bin")
    set_tests_properties(karma_transfer_map_io_mpi_${ranks} PROPERTIES
      RESOURCE_LOCK karma_transfer_map_files TIMEOUT 30)
  endforeach()
endif()

add_executable(karma_conservative_transfer_capability_test conservative_transfer_capability_test.C)
target_link_libraries(karma_conservative_transfer_capability_test PRIVATE KARMA::karma)
add_test(NAME karma_conservative_transfer_capability COMMAND karma_conservative_transfer_capability_test)

add_executable(karma_point_locator_contract_test point_locator_contract_test.C)
target_link_libraries(karma_point_locator_contract_test PRIVATE KARMA::karma)
add_test(NAME karma_point_locator_contract COMMAND karma_point_locator_contract_test)

add_executable(karma_geometry_inverse_map_contract_test geometry_inverse_map_contract_test.C)
target_link_libraries(karma_geometry_inverse_map_contract_test PRIVATE KARMA::karma)
add_test(NAME karma_geometry_inverse_map_contract COMMAND karma_geometry_inverse_map_contract_test)

add_executable(karma_moab_point_locator_test moab_point_locator_test.C)
target_link_libraries(karma_moab_point_locator_test PRIVATE KARMA::karma)
add_test(NAME karma_moab_point_locator COMMAND karma_moab_point_locator_test)

add_executable(karma_distributed_point_locator_test distributed_point_locator_test.C)
target_link_libraries(karma_distributed_point_locator_test PRIVATE KARMA::karma)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_distributed_point_locator_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_distributed_point_locator_test>)
  set_tests_properties(karma_distributed_point_locator_mpi PROPERTIES TIMEOUT 20)
endif()

add_executable(karma_fem_field_evaluator_test fem_field_evaluator_test.C)
target_link_libraries(karma_fem_field_evaluator_test PRIVATE KARMA::karma)
add_test(NAME karma_fem_field_evaluator COMMAND karma_fem_field_evaluator_test)

add_executable(karma_fem_transfer_sampling_test fem_transfer_sampling_test.C)
target_link_libraries(karma_fem_transfer_sampling_test PRIVATE KARMA::karma)
add_test(NAME karma_fem_transfer_sampling COMMAND karma_fem_transfer_sampling_test)

add_executable(karma_fem_transfer_projection_test fem_transfer_projection_test.C)

add_executable(karma_fem_transfer_serial_reference_test fem_transfer_serial_reference_test.C)
target_link_libraries(karma_fem_transfer_serial_reference_test PRIVATE KARMA::karma)

add_executable(karma_fem_transfer_partitioned_reference_test fem_transfer_partitioned_reference_test.C)
target_link_libraries(karma_fem_transfer_partitioned_reference_test PRIVATE KARMA::karma)
target_link_libraries(karma_fem_transfer_serial_reference_test PRIVATE KARMA::karma)
if(MPIEXEC_EXECUTABLE)
  foreach(ranks 2 4 5)
    add_test(NAME karma_fem_transfer_serial_reference_mpi_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_fem_transfer_serial_reference_test>)
    set_tests_properties(karma_fem_transfer_serial_reference_mpi_${ranks} PROPERTIES TIMEOUT 120)
  endforeach()
endif()
target_link_libraries(karma_fem_transfer_projection_test PRIVATE KARMA::karma)
add_test(NAME karma_fem_transfer_projection COMMAND karma_fem_transfer_projection_test)

add_executable(karma_transfer_diagnostics_test transfer_diagnostics_test.C)
target_link_libraries(karma_transfer_diagnostics_test PRIVATE KARMA::karma)
add_test(NAME karma_transfer_diagnostics COMMAND karma_transfer_diagnostics_test)

add_executable(karma_cg_transfer_accuracy_test cg_transfer_accuracy_test.C)
target_link_libraries(karma_cg_transfer_accuracy_test PRIVATE KARMA::karma)
add_test(NAME karma_cg_transfer_accuracy_serial COMMAND karma_cg_transfer_accuracy_test)
set_tests_properties(karma_cg_transfer_accuracy_serial PROPERTIES TIMEOUT 120)

add_executable(karma_native_quasi_interpolation_test native_quasi_interpolation_test.C)
target_link_libraries(karma_native_quasi_interpolation_test PRIVATE KARMA::karma)
add_test(NAME karma_native_quasi_interpolation COMMAND karma_native_quasi_interpolation_test)

add_executable(karma_field_commit_test field_commit_test.C)
target_link_libraries(karma_field_commit_test PRIVATE KARMA::karma)
add_test(NAME karma_field_commit COMMAND karma_field_commit_test)

add_executable(karma_moab_field_commit_test moab_field_commit_test.C)
target_link_libraries(karma_moab_field_commit_test PRIVATE KARMA::karma)
add_test(NAME karma_moab_field_commit COMMAND karma_moab_field_commit_test)

add_executable(karma_fem_transfer_convergence_test fem_transfer_convergence_test.C)
target_link_libraries(karma_fem_transfer_convergence_test PRIVATE KARMA::karma)
add_test(NAME karma_fem_transfer_convergence_serial
  COMMAND $<TARGET_FILE:karma_fem_transfer_convergence_test>)
if(MPIEXEC_EXECUTABLE)
  foreach(ranks 2 4 5)
    add_test(NAME karma_fem_transfer_convergence_mpi_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_fem_transfer_convergence_test>)
    set_tests_properties(karma_fem_transfer_convergence_mpi_${ranks} PROPERTIES TIMEOUT 120)
  endforeach()
endif()
add_test(NAME karma_petsc_scalar_probe
  COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/test_petsc_scalar_probe.cmake)
add_test(NAME karma_petsc_configured_scalar
  COMMAND ${CMAKE_COMMAND}
          -DPETSC_CONFIG_HEADER=${KARMA_PETSC_CONFIG_HEADER}
          -DEXPECTED_MODE=${KARMA_PETSC_SCALAR_MODE}
          -P ${PROJECT_SOURCE_DIR}/cmake/check_petsc_scalar.cmake)

if(KARMA_HAVE_MOAB_COUPLER)
  add_executable(karma_moab_coupler_probe moab_coupler_probe.C)
  target_link_libraries(karma_moab_coupler_probe PRIVATE KARMA::karma ${KARMA_MOAB_COUPLER_LIBRARY})
  add_test(NAME karma_moab_coupler_probe COMMAND karma_moab_coupler_probe)
  if(MPIEXEC_EXECUTABLE)
    add_test(NAME karma_moab_coupler_probe_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_moab_coupler_probe>)
    set_tests_properties(karma_moab_coupler_probe_mpi PROPERTIES TIMEOUT 20)
  endif()
endif()

# Eigen-backed geometry and element-local dense algebra. Header-only, so this
# needs no MPI launch.
add_executable(karma_dense_test dense_test.C)
target_link_libraries(karma_dense_test PRIVATE KARMA::karma)

add_test(NAME karma_dense COMMAND karma_dense_test)

# Reference topology and linear/quadratic geometry mappings.
#
# Registered one section per independent part rather than as a single test, so
# CTest can schedule them concurrently. The section names are the executable's
# own selectors; an unrecognized one makes the binary exit non-zero rather than
# pass having run nothing, so a typo here fails loudly instead of silently
# dropping coverage. Running the executable with no argument still runs
# everything, which is what to do when reproducing a failure by hand.
add_executable(karma_geometry_test geometry_test.C)
target_link_libraries(karma_geometry_test PRIVATE KARMA::karma)
foreach(section reference-variants affine-and-curved adversarial-tet10 mesh-loading)
  # Test names use underscores to match every other test here; the selector
  # keeps the hyphens it has on the command line.
  string(REPLACE "-" "_" section_name "${section}")
  add_test(NAME karma_geometry_${section_name} COMMAND karma_geometry_test --${section})
endforeach()

# Geometry-independent scalar reference bases and their instance registry.
add_executable(karma_basis_test basis_test.C)
target_link_libraries(karma_basis_test PRIVATE KARMA::karma)
add_test(NAME karma_basis COMMAND karma_basis_test)

# Geometry-independent field bases composed with mesh-backed geometry maps.
#
# These two were the longest tests in the suite (~152s and ~194s), long enough
# that they set the floor on a parallel run's wall time no matter how many
# cores were available. Each is now registered one test per independent
# section; see the comment on karma_geometry above for how the selectors
# behave.
add_executable(karma_fe_test fe_test.C)
target_link_libraries(karma_fe_test PRIVATE KARMA::karma)
set(karma_fe_sections quadrature quadrature-exactness pyramid-gradients allocation embedded-1d degenerate basis
    variants-line-surface variants-tet-hex variants-prism-pyramid hex27 volumes)
foreach(section ${karma_fe_sections})
  string(REPLACE "-" "_" section_name "${section}")
  add_test(NAME karma_fe_${section_name} COMMAND karma_fe_test --${section})
endforeach()

add_executable(karma_fe_face_test fe_face_test.C)
target_link_libraries(karma_fe_face_test PRIVATE KARMA::karma)
# Split into the sections that do work under MPI and those guarded by
# `size == 1`. Only the first group is worth registering as an MPI test: the
# rest would initialize PETSc, run no assertion, and report a pass.
set(karma_fe_face_parallel_sections reinit-paths box-mesh face-integration)
set(karma_fe_face_serial_sections
    parent-variants-line-surface parent-variants-tet-hex parent-variants-prism-pyramid
    nonconforming tet-prism edge-cases)
foreach(section ${karma_fe_face_parallel_sections} ${karma_fe_face_serial_sections})
  string(REPLACE "-" "_" section_name "${section}")
  add_test(NAME karma_fe_face_${section_name} COMMAND karma_fe_face_test --${section})
endforeach()
add_test(NAME karma_fe_face_shared_hex COMMAND karma_fe_face_test --shared-hex)
# Replicated-SPMD numerical checks only; distributed neighbor traversal remains deferred.
if(MPIEXEC_EXECUTABLE)
  # This fixture is constructed by the test itself and must not depend on mbpart.
  add_test(NAME karma_mesh_collective_nonmanifold_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_mesh_test> --collective-nonmanifold)
  set_tests_properties(karma_mesh_collective_nonmanifold_2 PROPERTIES TIMEOUT 20)
  add_test(NAME karma_mesh_empty_partition_global_id_repair_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_mesh_test> --empty-partition-global-id-repair)
  set_tests_properties(karma_mesh_empty_partition_global_id_repair_2 PROPERTIES TIMEOUT 20)
  add_test(NAME karma_mesh_reserved_global_id_validation_tag_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_mesh_test> --reserved-global-id-validation-tag)
  set_tests_properties(karma_mesh_reserved_global_id_validation_tag_2 PROPERTIES TIMEOUT 20)
  add_test(NAME karma_advection_partitioned_nonmanifold_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_advection_flux_test> --partitioned-advection-nonmanifold)
  set_tests_properties(karma_advection_partitioned_nonmanifold_2 PROPERTIES TIMEOUT 20)
  add_test(NAME karma_boundary_empty_partition_rank_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_boundary_condition_test> --empty-partition-boundaries)
  set_tests_properties(karma_boundary_empty_partition_rank_mpi PROPERTIES TIMEOUT 20)

  # Sectioned for the same reason as the serial legs above: the single
  # karma_fe_replicated_spmd_mpi test ran ~154s and every other rank-2 test
  # waited behind it.
  foreach(section ${karma_fe_sections})
    string(REPLACE "-" "_" section_name "${section}")
    add_test(NAME karma_fe_${section_name}_replicated_spmd_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_fe_test> --${section})
  endforeach()
  # Only the sections that actually execute at size > 1; the `size == 1`
  # sections are covered by their serial registrations above.
  foreach(section ${karma_fe_face_parallel_sections})
    string(REPLACE "-" "_" section_name "${section}")
    add_test(NAME karma_fe_face_${section_name}_replicated_spmd_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_fe_face_test> --${section})
  endforeach()
  add_test(NAME karma_fe_face_shared_hex_replicated_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_fe_face_test> --shared-hex)
  set_tests_properties(karma_fe_face_shared_hex_replicated_mpi PROPERTIES TIMEOUT 20)
endif()

# The MOAB-backed mesh.
add_executable(karma_mesh_test mesh_test.C)
target_link_libraries(karma_mesh_test PRIVATE KARMA::karma)

add_test(NAME karma_mesh COMMAND karma_mesh_test)
if(MPIEXEC_EXECUTABLE)
	add_test(NAME karma_mesh_mpi
		COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
		        $<TARGET_FILE:karma_mesh_test>)
endif()

add_executable(karma_moab_io_test moab_io_test.C)
target_link_libraries(karma_moab_io_test PRIVATE KARMA::karma)
add_test(NAME karma_moab_io COMMAND karma_moab_io_test)
add_test(NAME karma_moab_io_repeated_checkpoint_threshold
  COMMAND karma_moab_io_test --repeated-checkpoint-threshold)
set_tests_properties(karma_moab_io PROPERTIES RESOURCE_LOCK karma_moab_io_files)
set_tests_properties(karma_moab_io_repeated_checkpoint_threshold PROPERTIES
  RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_moab_io_repeated_checkpoint_threshold_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_moab_io_test> --repeated-checkpoint-threshold)
  set_tests_properties(karma_moab_io_repeated_checkpoint_threshold_mpi PROPERTIES
    RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
  add_test(NAME karma_moab_io_replicated_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_moab_io_test>)
  set_tests_properties(karma_moab_io_replicated_mpi PROPERTIES
    RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
  foreach(ranks 2 3 4)
    add_test(NAME karma_moab_io_owner_only_gather_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_moab_io_test> --owner-only-checkpoint-gather)
    set_tests_properties(karma_moab_io_owner_only_gather_${ranks} PROPERTIES TIMEOUT 120)
  endforeach()
  add_test(NAME karma_moab_io_asymmetric_filename_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_moab_io_test> --asymmetric-checkpoint-filename)
  set_tests_properties(karma_moab_io_asymmetric_filename_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_moab_io_asymmetric_system_name_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_moab_io_test> --asymmetric-checkpoint-system-name)
  set_tests_properties(karma_moab_io_asymmetric_system_name_mpi PROPERTIES
    RESOURCE_LOCK karma_moab_io_files TIMEOUT 20)
  add_test(NAME karma_moab_io_invalid_temporary_path_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_moab_io_test> --invalid-checkpoint-temporary-path)
  set_tests_properties(karma_moab_io_invalid_temporary_path_mpi PROPERTIES
    RESOURCE_LOCK karma_moab_io_files TIMEOUT 30)
  add_test(NAME karma_moab_io_asymmetric_read_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_moab_io_test> --asymmetric-read-filename)
  set_tests_properties(karma_moab_io_asymmetric_read_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_moab_io_replicated_parallel_write_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_moab_io_test> --replicated-parallel-write)
  set_tests_properties(karma_moab_io_replicated_parallel_write_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_moab_io_asymmetric_global_ids_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_moab_io_test> --asymmetric-replicated-global-ids)
  set_tests_properties(karma_moab_io_asymmetric_global_ids_mpi PROPERTIES TIMEOUT 20)
  foreach(failure size layout state nonfinite)
    add_test(NAME karma_moab_io_asymmetric_restore_vector_${failure}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_moab_io_test> --asymmetric-restore-vector-${failure})
    set_tests_properties(karma_moab_io_asymmetric_restore_vector_${failure}_mpi PROPERTIES
      RESOURCE_LOCK karma_moab_io_files TIMEOUT 20)
  endforeach()
endif()

find_program(KARMA_MBPART_EXECUTABLE NAMES mbpart HINTS /opt/moab/cmake-all/bin)
if(MPIEXEC_EXECUTABLE AND KARMA_MBPART_EXECUTABLE)
  set(_karma_partition_fixture "${CMAKE_CURRENT_BINARY_DIR}/karma_partition_fixture.h5m")
  set(_karma_face_fixture "${CMAKE_CURRENT_BINARY_DIR}/karma_face_fixture.h5m")
  set(_karma_transfer_source_fixture "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_source_fixture.h5m")
  set(_karma_transfer_target_fixture "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_target_fixture.h5m")
  set(_karma_transfer_reference "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_reference.bin")
  set(_karma_partitioned_face_fixture "${CMAKE_CURRENT_BINARY_DIR}/karma_partitioned_face_fixture_2.h5m")
  add_test(NAME karma_fem_transfer_write_source_fixture
    COMMAND $<TARGET_FILE:karma_moab_io_test> --write-partition-fixture "${_karma_transfer_source_fixture}")
  set_tests_properties(karma_fem_transfer_write_source_fixture PROPERTIES
    FIXTURES_SETUP karma_fem_transfer_raw_source TIMEOUT 60)
  add_test(NAME karma_fem_transfer_write_target_fixture
    COMMAND $<TARGET_FILE:karma_moab_io_test> --write-partition-fixture "${_karma_transfer_target_fixture}")
  set_tests_properties(karma_fem_transfer_write_target_fixture PROPERTIES
    FIXTURES_SETUP karma_fem_transfer_raw_target TIMEOUT 60)
  add_test(NAME karma_fem_transfer_write_serial_reference
    COMMAND $<TARGET_FILE:karma_fem_transfer_partitioned_reference_test> --write-reference
            "${_karma_transfer_source_fixture}" "${_karma_transfer_target_fixture}" "${_karma_transfer_reference}")
  set_tests_properties(karma_fem_transfer_write_serial_reference PROPERTIES
    FIXTURES_REQUIRED "karma_fem_transfer_raw_source;karma_fem_transfer_raw_target"
    FIXTURES_SETUP karma_fem_transfer_reference TIMEOUT 120)

  add_test(NAME karma_moab_io_write_partition_fixture
    COMMAND karma_moab_io_test --write-partition-fixture "${_karma_partition_fixture}")
  set_tests_properties(karma_moab_io_write_partition_fixture PROPERTIES
    FIXTURES_SETUP karma_moab_raw_fixture TIMEOUT 60)
  foreach(ranks 2 3 4)
    add_test(NAME karma_moab_io_restore_serial_checkpoint_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_moab_io_test> --verify-partition-file "${_karma_partition_fixture}")
    set_tests_properties(karma_moab_io_restore_serial_checkpoint_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_raw_fixture RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
  endforeach()
  foreach(ranks 2 3 4 5)
    set(_karma_transfer_source_partitioned "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_source_${ranks}.h5m")
    set(_karma_transfer_target_partitioned "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_target_${ranks}.h5m")
    add_test(NAME karma_fem_transfer_partition_source_${ranks}
      COMMAND "${KARMA_MBPART_EXECUTABLE}" -z RCB -j ${ranks}
              "${_karma_transfer_source_fixture}" "${_karma_transfer_source_partitioned}")
    set_tests_properties(karma_fem_transfer_partition_source_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_fem_transfer_raw_source
      FIXTURES_SETUP karma_fem_transfer_partitioned_source_${ranks} TIMEOUT 60)
    add_test(NAME karma_fem_transfer_partition_target_${ranks}
      COMMAND "${KARMA_MBPART_EXECUTABLE}" -z RCB -j ${ranks}
              "${_karma_transfer_target_fixture}" "${_karma_transfer_target_partitioned}")
    set_tests_properties(karma_fem_transfer_partition_target_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_fem_transfer_raw_target
      FIXTURES_SETUP karma_fem_transfer_partitioned_target_${ranks} TIMEOUT 60)
    add_test(NAME karma_fem_transfer_verify_partitioned_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_fem_transfer_partitioned_reference_test> --verify-partitioned
              "${_karma_transfer_source_partitioned}" "${_karma_transfer_target_partitioned}" "${_karma_transfer_reference}")
    set_tests_properties(karma_fem_transfer_verify_partitioned_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_fem_transfer_reference;karma_fem_transfer_partitioned_source_${ranks};karma_fem_transfer_partitioned_target_${ranks}"
      TIMEOUT 120)
    if(ranks LESS_EQUAL 4)
      add_test(NAME karma_cg_transfer_accuracy_partitioned_${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:karma_cg_transfer_accuracy_test> --partitioned
                "${_karma_transfer_source_partitioned}" "${_karma_transfer_target_partitioned}")
      set_tests_properties(karma_cg_transfer_accuracy_partitioned_${ranks} PROPERTIES
        FIXTURES_REQUIRED "karma_fem_transfer_partitioned_source_${ranks};karma_fem_transfer_partitioned_target_${ranks}"
        TIMEOUT 120)
    endif()
  endforeach()
  add_test(NAME karma_fem_transfer_project_cg_asymmetric_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_fem_transfer_partitioned_reference_test>
            --verify-project-cg-asymmetric
            "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_source_2.h5m")
  set_tests_properties(karma_fem_transfer_project_cg_asymmetric_mpi PROPERTIES
    FIXTURES_REQUIRED karma_fem_transfer_partitioned_source_2
    TIMEOUT 20)
  add_test(NAME karma_fem_transfer_project_cg_asymmetric_uncovered_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_fem_transfer_partitioned_reference_test>
            --verify-project-cg-asymmetric-uncovered
            "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_source_2.h5m"
            "${CMAKE_CURRENT_BINARY_DIR}/karma_transfer_target_2.h5m")
  set_tests_properties(karma_fem_transfer_project_cg_asymmetric_uncovered_mpi PROPERTIES
    FIXTURES_REQUIRED "karma_fem_transfer_partitioned_source_2;karma_fem_transfer_partitioned_target_2"
    TIMEOUT 20)

  # independently generated and RCB-partitioned nonmatching Gray-Scott meshes.
  set(_karma_gray_scott_u_raw "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_u_raw.h5m")
  set(_karma_gray_scott_v_raw "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_v_raw.h5m")
  set(_karma_gray_scott_reference "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_partitioned_reference.bin")
  add_test(NAME karma_gray_scott_write_u_fixture
    COMMAND $<TARGET_FILE:karma_gray_scott_partitioned_test> --write-fixture "${_karma_gray_scott_u_raw}" 5 4)
  set_tests_properties(karma_gray_scott_write_u_fixture PROPERTIES
    FIXTURES_SETUP karma_gray_scott_u_raw_fixture RESOURCE_LOCK karma_moab_io_files TIMEOUT 60)
  add_test(NAME karma_gray_scott_write_v_fixture
    COMMAND $<TARGET_FILE:karma_gray_scott_partitioned_test> --write-fixture "${_karma_gray_scott_v_raw}" 7 3)
  set_tests_properties(karma_gray_scott_write_v_fixture PROPERTIES
    FIXTURES_SETUP karma_gray_scott_v_raw_fixture RESOURCE_LOCK karma_moab_io_files TIMEOUT 60)
  add_test(NAME karma_gray_scott_write_partitioned_reference
    COMMAND $<TARGET_FILE:karma_gray_scott_partitioned_test> --write-reference
            "${_karma_gray_scott_u_raw}" "${_karma_gray_scott_v_raw}" "${_karma_gray_scott_reference}")
  set_tests_properties(karma_gray_scott_write_partitioned_reference PROPERTIES
    FIXTURES_REQUIRED "karma_gray_scott_u_raw_fixture;karma_gray_scott_v_raw_fixture"
    FIXTURES_SETUP karma_gray_scott_partitioned_reference RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
  foreach(ranks 2 3 4)
    set(_karma_gray_scott_u_partitioned "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_u_${ranks}.h5m")
    set(_karma_gray_scott_v_partitioned "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_v_${ranks}.h5m")
    add_test(NAME karma_gray_scott_partition_u_${ranks}
      COMMAND "${KARMA_MBPART_EXECUTABLE}" -z RCB -j ${ranks}
              "${_karma_gray_scott_u_raw}" "${_karma_gray_scott_u_partitioned}")
    set_tests_properties(karma_gray_scott_partition_u_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_gray_scott_u_raw_fixture
      FIXTURES_SETUP karma_gray_scott_u_partitioned_${ranks} RESOURCE_LOCK karma_moab_io_files TIMEOUT 60)
    add_test(NAME karma_gray_scott_partition_v_${ranks}
      COMMAND "${KARMA_MBPART_EXECUTABLE}" -z RCB -j ${ranks}
              "${_karma_gray_scott_v_raw}" "${_karma_gray_scott_v_partitioned}")
    set_tests_properties(karma_gray_scott_partition_v_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_gray_scott_v_raw_fixture
      FIXTURES_SETUP karma_gray_scott_v_partitioned_${ranks} RESOURCE_LOCK karma_moab_io_files TIMEOUT 60)
    add_test(NAME karma_gray_scott_partitioned_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_gray_scott_partitioned_test> --verify-partitioned
              "${_karma_gray_scott_u_partitioned}" "${_karma_gray_scott_v_partitioned}" "${_karma_gray_scott_reference}")
    set_tests_properties(karma_gray_scott_partitioned_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_gray_scott_partitioned_reference;karma_gray_scott_u_partitioned_${ranks};karma_gray_scott_v_partitioned_${ranks}"
      RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
  endforeach()
  add_test(NAME karma_gray_scott_partitioned_remote_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_partitioned_test> --verify-remote
            "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_u_2.h5m" "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_v_2.h5m")
  set_tests_properties(karma_gray_scott_partitioned_remote_2 PROPERTIES
    FIXTURES_REQUIRED "karma_gray_scott_u_partitioned_2;karma_gray_scott_v_partitioned_2"
    RESOURCE_LOCK karma_moab_io_files TIMEOUT 20)
  # The adapter injects the failure on rank 1, verifies collective rollback, then retries from that checkpoint.
  # Restore-failure injection is intentionally excluded: the adapter's public restore contract has no test seam
  # for a one-shot asymmetric restore failure without changing production code.
  # Commit injection is safely recoverable through the existing adapter seam. Exchange/advance injection
  # would throw before their internal PETSc/MPI collectives and deadlock peers, so they are not faked here.
  add_test(NAME karma_gray_scott_partitioned_retry_commit_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_partitioned_test> --retry-commit
            "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_u_2.h5m" "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_v_2.h5m")
  set_tests_properties(karma_gray_scott_partitioned_retry_commit_2 PROPERTIES
    FIXTURES_REQUIRED "karma_gray_scott_u_partitioned_2;karma_gray_scott_v_partitioned_2"
    RESOURCE_LOCK karma_moab_io_files TIMEOUT 20)

  add_test(NAME karma_system_write_face_fixture
    COMMAND karma_system_test --write-face-fixture "${_karma_face_fixture}")
  set_tests_properties(karma_system_write_face_fixture PROPERTIES
    FIXTURES_SETUP karma_face_raw_fixture TIMEOUT 60)
  add_test(NAME karma_system_partition_face_fixture
    COMMAND "${KARMA_MBPART_EXECUTABLE}" -z RCB -j 2
            "${_karma_face_fixture}" "${_karma_partitioned_face_fixture}")
  set_tests_properties(karma_system_partition_face_fixture PROPERTIES
    FIXTURES_REQUIRED karma_face_raw_fixture
    FIXTURES_SETUP karma_face_partitioned_fixture TIMEOUT 60)
  add_test(NAME karma_system_partitioned_face_dofs_2
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_system_test> --partitioned-face-fixture
            "${_karma_partitioned_face_fixture}")
  set_tests_properties(karma_system_partitioned_face_dofs_2 PROPERTIES
    FIXTURES_REQUIRED karma_face_partitioned_fixture TIMEOUT 120)

  foreach(ranks 2 3 4)
    set(_karma_partitioned_fixture "${CMAKE_CURRENT_BINARY_DIR}/karma_partitioned_fixture_${ranks}.h5m")
    set(_karma_dof_reference "${CMAKE_CURRENT_BINARY_DIR}/karma_dof_reference_${ranks}.txt")
    set(_karma_mass_reference "${CMAKE_CURRENT_BINARY_DIR}/karma_mass_reference_${ranks}.txt")
    set(_karma_advection_reference "${CMAKE_CURRENT_BINARY_DIR}/karma_advection_reference_${ranks}.txt")
    set(_karma_advection_p1_reference "${CMAKE_CURRENT_BINARY_DIR}/karma_advection_p1_reference_${ranks}.txt")
    set(_karma_natural_boundary_reference "${CMAKE_CURRENT_BINARY_DIR}/karma_natural_boundary_reference_${ranks}.txt")
    set(_karma_sparsity_reference "${CMAKE_CURRENT_BINARY_DIR}/karma_sparsity_reference_${ranks}.txt")
    set(_karma_parallel_checkpoint "${CMAKE_CURRENT_BINARY_DIR}/karma_parallel_checkpoint_${ranks}.h5m")
    add_test(NAME karma_moab_io_partition_fixture_${ranks}
      COMMAND "${KARMA_MBPART_EXECUTABLE}" -z RCB -j ${ranks}
              "${_karma_partition_fixture}" "${_karma_partitioned_fixture}")
    set_tests_properties(karma_moab_io_partition_fixture_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_raw_fixture
      FIXTURES_SETUP karma_moab_partitioned_fixture_${ranks}
      TIMEOUT 60)
    add_test(NAME karma_system_write_dof_reference_${ranks}
      COMMAND karma_system_test --write-dof-reference
              "${_karma_partitioned_fixture}" "${_karma_dof_reference}")
    set_tests_properties(karma_system_write_dof_reference_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks}
      FIXTURES_SETUP karma_dof_reference_${ranks} TIMEOUT 60)
    add_test(NAME karma_system_partitioned_dofs_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_system_test> --partitioned-dof-fixture
              "${_karma_partitioned_fixture}" "${_karma_dof_reference}")
    set_tests_properties(karma_system_partitioned_dofs_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_dof_reference_${ranks}"
      TIMEOUT 120)
    add_test(NAME karma_mass_write_reference_${ranks}
      COMMAND $<TARGET_FILE:karma_mass_assembly_test> --write-partitioned-mass-reference
              "${_karma_partition_fixture}" "${_karma_mass_reference}")
    set_tests_properties(karma_mass_write_reference_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_raw_fixture
      FIXTURES_SETUP karma_mass_reference_${ranks} TIMEOUT 60)
    add_test(NAME karma_mass_partitioned_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_mass_assembly_test> --verify-partitioned-mass
              "${_karma_partitioned_fixture}" "${_karma_mass_reference}")
    set_tests_properties(karma_mass_partitioned_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_mass_reference_${ranks}"
      TIMEOUT 120)
    add_test(NAME karma_advection_write_reference_${ranks}
      COMMAND $<TARGET_FILE:karma_advection_flux_test> --write-partitioned-advection-reference
              "${_karma_partition_fixture}" "${_karma_advection_reference}")
    set_tests_properties(karma_advection_write_reference_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_raw_fixture
      FIXTURES_SETUP karma_advection_reference_${ranks} TIMEOUT 60)
    add_test(NAME karma_advection_partitioned_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_advection_flux_test> --verify-partitioned-advection
              "${_karma_partitioned_fixture}" "${_karma_advection_reference}")
    set_tests_properties(karma_advection_partitioned_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_advection_reference_${ranks}"
      TIMEOUT 120)
    add_test(NAME karma_advection_partitioned_missing_ghost_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_advection_flux_test> --partitioned-advection-without-ghosts
              "${_karma_partitioned_fixture}")
    if(ranks EQUAL 2)
      add_test(NAME karma_advection_partitioned_distinct_handles_${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:karma_advection_flux_test> --partitioned-advection-distinct-handles
                "${_karma_partitioned_fixture}")
      set_tests_properties(karma_advection_partitioned_distinct_handles_${ranks} PROPERTIES
        FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 20)
    endif()
    set_tests_properties(karma_advection_partitioned_missing_ghost_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 20)
    add_test(NAME karma_advection_write_p1_reference_${ranks}
      COMMAND $<TARGET_FILE:karma_advection_flux_test> --write-partitioned-advection-p1-reference
              "${_karma_partition_fixture}" "${_karma_advection_p1_reference}")
    set_tests_properties(karma_advection_write_p1_reference_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_raw_fixture
      FIXTURES_SETUP karma_advection_p1_reference_${ranks} TIMEOUT 60)
    add_test(NAME karma_advection_partitioned_p1_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_advection_flux_test> --verify-partitioned-advection-p1
              "${_karma_partitioned_fixture}" "${_karma_advection_p1_reference}")
    set_tests_properties(karma_advection_partitioned_p1_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_advection_p1_reference_${ranks}"
      TIMEOUT 120)
    if(ranks EQUAL 2)
      add_test(NAME karma_advection_partitioned_dg_face_mutation_${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:karma_advection_dg_face_mutation_test> --verify-partitioned-advection-mutation
                "${_karma_partitioned_fixture}" "${_karma_advection_reference}")
      set_tests_properties(karma_advection_partitioned_dg_face_mutation_${ranks} PROPERTIES
        FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_advection_reference_${ranks}"
        TIMEOUT 120)
      add_test(NAME karma_mass_partitioned_cg_coupling_mutation_${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:karma_mass_assembly_test> --verify-partitioned-mass-mutation
                "${_karma_partitioned_fixture}" "${_karma_mass_reference}")
      set_tests_properties(karma_mass_partitioned_cg_coupling_mutation_${ranks} PROPERTIES
        FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_mass_reference_${ranks}"
        TIMEOUT 120)
    endif()
    add_test(NAME karma_mesh_partitioned_neighbors_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_mesh_test> --partitioned-neighbors
              "${_karma_partitioned_fixture}")
    set_tests_properties(karma_mesh_partitioned_neighbors_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 60)
    add_test(NAME karma_mesh_partitioned_missing_ghost_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_mesh_test> --partitioned-neighbors-without-ghosts
              "${_karma_partitioned_fixture}")
    set_tests_properties(karma_mesh_partitioned_missing_ghost_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 20)
    add_test(NAME karma_boundary_partitioned_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_boundary_condition_test> --partitioned-boundaries
              "${_karma_partitioned_fixture}")
    set_tests_properties(karma_boundary_partitioned_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 60)
    add_test(NAME karma_boundary_partitioned_conflict_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_boundary_condition_test> --partitioned-conflicting-dirichlet
              "${_karma_partitioned_fixture}")
    set_tests_properties(karma_boundary_partitioned_conflict_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 20)
    add_test(NAME karma_boundary_partitioned_time_dirichlet_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_boundary_condition_test> --partitioned-time-dirichlet
              "${_karma_partitioned_fixture}")
    set_tests_properties(karma_boundary_partitioned_time_dirichlet_${ranks} PROPERTIES
     FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 60)
    add_test(NAME karma_diffusion_partitioned_natural_boundary_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_diffusion_assembly_test> --partitioned-natural-boundary
              "${_karma_partitioned_fixture}" "${_karma_natural_boundary_reference}")
    set_tests_properties(karma_diffusion_partitioned_natural_boundary_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_natural_boundary_reference_${ranks}" TIMEOUT 60)
    add_test(NAME karma_diffusion_write_natural_boundary_reference_${ranks}
      COMMAND $<TARGET_FILE:karma_diffusion_assembly_test> --write-partitioned-natural-reference
              "${_karma_partition_fixture}" "${_karma_natural_boundary_reference}")
    set_tests_properties(karma_diffusion_write_natural_boundary_reference_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_raw_fixture
      FIXTURES_SETUP karma_natural_boundary_reference_${ranks} TIMEOUT 60)
    add_test(NAME karma_diffusion_partitioned_natural_boundary_mutation_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_diffusion_assembly_test> --partitioned-natural-boundary-mutation
              "${_karma_partitioned_fixture}" "${_karma_natural_boundary_reference}")
    set_tests_properties(karma_diffusion_partitioned_natural_boundary_mutation_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_natural_boundary_reference_${ranks}"
      TIMEOUT 60)
    if(ranks EQUAL 2)
      add_test(NAME karma_boundary_partitioned_callback_failure_${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:karma_boundary_condition_test> --partitioned-callback-failure
                "${_karma_partitioned_fixture}")
      set_tests_properties(karma_boundary_partitioned_callback_failure_${ranks} PROPERTIES
        FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 20)
      add_test(NAME karma_boundary_partitioned_asymmetric_continuity_${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:karma_boundary_condition_test> --partitioned-asymmetric-continuity
                "${_karma_partitioned_fixture}")
      set_tests_properties(karma_boundary_partitioned_asymmetric_continuity_${ranks} PROPERTIES
        FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 20)
    endif()
    add_test(NAME karma_system_write_sparsity_reference_${ranks}
      COMMAND karma_system_test --write-sparsity-reference
              "${_karma_partition_fixture}" "${_karma_sparsity_reference}")
    set_tests_properties(karma_system_write_sparsity_reference_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_raw_fixture
      FIXTURES_SETUP karma_sparsity_reference_${ranks} TIMEOUT 60)
    add_test(NAME karma_system_partitioned_sparsity_no_halo_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_system_test> --partitioned-sparsity-no-halo
              "${_karma_partitioned_fixture}" "${_karma_sparsity_reference}")
    set_tests_properties(karma_system_partitioned_sparsity_no_halo_${ranks} PROPERTIES
      FIXTURES_REQUIRED "karma_moab_partitioned_fixture_${ranks};karma_sparsity_reference_${ranks}" TIMEOUT 120)
    if(ranks EQUAL 2)
      add_test(NAME karma_system_asymmetric_missing_subentity_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:karma_system_test> --asymmetric-missing-subentity
                "${_karma_partitioned_fixture}")
      set_tests_properties(karma_system_asymmetric_missing_subentity_mpi PROPERTIES
        FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks} TIMEOUT 20)
    endif()
    add_test(NAME karma_moab_io_partitioned_mpi_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_moab_io_test> --parallel-partition-roundtrip
              "${_karma_partitioned_fixture}" "${_karma_parallel_checkpoint}")
    set_tests_properties(karma_moab_io_partitioned_mpi_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_partitioned_fixture_${ranks}
      FIXTURES_SETUP karma_moab_parallel_fixture_${ranks}
      TIMEOUT 120)
    add_test(NAME karma_moab_io_verify_parallel_checkpoint_${ranks}
      COMMAND karma_moab_io_test --verify-partition-file "${_karma_parallel_checkpoint}")
    set_tests_properties(karma_moab_io_verify_parallel_checkpoint_${ranks} PROPERTIES
      FIXTURES_REQUIRED karma_moab_parallel_fixture_${ranks}
      TIMEOUT 60)
    if(ranks EQUAL 2)
      add_test(NAME karma_moab_io_restore_2_checkpoint_3
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
                $<TARGET_FILE:karma_moab_io_test> --verify-partition-file "${_karma_parallel_checkpoint}")
      set_tests_properties(karma_moab_io_restore_2_checkpoint_3 PROPERTIES
        FIXTURES_REQUIRED karma_moab_parallel_fixture_2 RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
    endif()
  endforeach()
endif()

# KARMASystem and the DoF map.
add_executable(karma_system_test system_test.C)
target_link_libraries(karma_system_test PRIVATE KARMA::karma)

add_test(NAME karma_system COMMAND karma_system_test)

add_executable(karma_system_owner_test system_owner_test.C)
target_link_libraries(karma_system_owner_test PRIVATE KARMA::karma)
add_test(NAME karma_system_owner COMMAND karma_system_owner_test)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_system_heterogeneous_partition_cells_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_system_test> --heterogeneous-partition-cells)
  set_tests_properties(karma_system_heterogeneous_partition_cells_mpi PROPERTIES TIMEOUT 20)
  add_test(NAME karma_system_empty_partition_rank_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_system_test> --empty-partition-rank)
  set_tests_properties(karma_system_empty_partition_rank_mpi PROPERTIES TIMEOUT 20)
  foreach(ranks 2 3 4)
    add_test(NAME karma_system_mpi_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_system_test>)
  endforeach()
endif()

add_executable(karma_assembly_test assembly_test.C)
target_link_libraries(karma_assembly_test PRIVATE KARMA::karma)
add_test(NAME karma_assembly COMMAND karma_assembly_test)

add_executable(karma_mass_assembly_test mass_assembly_test.C)
target_link_libraries(karma_mass_assembly_test PRIVATE KARMA::karma)
add_test(NAME karma_mass_assembly COMMAND karma_mass_assembly_test)

add_executable(karma_diffusion_assembly_test diffusion_assembly_test.C)
target_link_libraries(karma_diffusion_assembly_test PRIVATE KARMA::karma)
add_test(NAME karma_diffusion_assembly COMMAND karma_diffusion_assembly_test)

add_executable(karma_diffusion_mms_test diffusion_mms_test.C)
target_link_libraries(karma_diffusion_mms_test PRIVATE KARMA::karma)
add_test(NAME karma_diffusion_mms COMMAND karma_diffusion_mms_test)
set_tests_properties(karma_diffusion_mms PROPERTIES LABELS serial)

add_executable(karma_diffusion_lifecycle_test diffusion_lifecycle_test.C)
target_link_libraries(karma_diffusion_lifecycle_test PRIVATE KARMA::karma)
add_test(NAME karma_diffusion_lifecycle COMMAND karma_diffusion_lifecycle_test)

add_executable(karma_gray_scott_model_test gray_scott_model_test.C)
target_link_libraries(karma_gray_scott_model_test PRIVATE KARMA::karma)
add_test(NAME karma_gray_scott_model COMMAND karma_gray_scott_model_test)

add_executable(karma_gray_scott_operator_split_test gray_scott_operator_split_test.C)
target_link_libraries(karma_gray_scott_operator_split_test PRIVATE KARMA::karma)

# The source is part of the active test target and is kept beside its harness.
add_executable(karma_gray_scott_partitioned_test gray_scott_partitioned_test.C)
target_link_libraries(karma_gray_scott_partitioned_test PRIVATE KARMA::karma)

add_executable(karma_gray_scott_restart_test gray_scott_restart_test.C)
target_link_libraries(karma_gray_scott_restart_test PRIVATE KARMA::karma)
foreach(mode roundtrip missing-v metadata incompatible wrong-mesh apply-failure)
  add_test(NAME karma_gray_scott_restart_${mode}_serial
    COMMAND $<TARGET_FILE:karma_gray_scott_restart_test> ${mode}
            "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_restart_${mode}_serial")
  set_tests_properties(karma_gray_scott_restart_${mode}_serial PROPERTIES
    RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
  if(MPIEXEC_EXECUTABLE)
    add_test(NAME karma_gray_scott_restart_${mode}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_gray_scott_restart_test> ${mode}
              "${CMAKE_CURRENT_BINARY_DIR}/karma_gray_scott_restart_${mode}_mpi")
    set_tests_properties(karma_gray_scott_restart_${mode}_mpi PROPERTIES
      RESOURCE_LOCK karma_moab_io_files TIMEOUT 120)
  endif()
endforeach()

add_executable(karma_gray_scott_spatial_mms_test gray_scott_spatial_mms_test.C)
target_link_libraries(karma_gray_scott_spatial_mms_test PRIVATE KARMA::karma)
# Was a single ~195s test - the longest in the suite. Its two checks are
# independent (each builds its own meshes and systems), so they are registered
# separately; --all still runs both for a by-hand reproduction.
add_test(NAME karma_gray_scott_strong_form
  COMMAND karma_gray_scott_spatial_mms_test --strong-form)
set_tests_properties(karma_gray_scott_strong_form PROPERTIES LABELS serial TIMEOUT 300)
# The refinement study is irreducible - a convergence rate needs every mesh
# level in one process - so it stays whole and remains the longest test here.
# The mutation checks it used to carry are independent and run at a single
# medium resolution, so they are their own test now.
add_test(NAME karma_gray_scott_spatial_convergence
  COMMAND karma_gray_scott_spatial_mms_test --spatial-convergence)
set_tests_properties(karma_gray_scott_spatial_convergence PROPERTIES LABELS serial TIMEOUT 600)
add_test(NAME karma_gray_scott_mutation_detection
  COMMAND karma_gray_scott_spatial_mms_test --mutation-detection)
set_tests_properties(karma_gray_scott_mutation_detection PROPERTIES LABELS serial TIMEOUT 300)
add_test(NAME karma_gray_scott_fixed_point_mms
  COMMAND karma_gray_scott_spatial_mms_test --fixed-point)
set_tests_properties(karma_gray_scott_fixed_point_mms PROPERTIES LABELS serial TIMEOUT 120)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_gray_scott_fixed_point_mms_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_spatial_mms_test> --fixed-point)
  set_tests_properties(karma_gray_scott_fixed_point_mms_mpi PROPERTIES TIMEOUT 120)
endif()
add_test(NAME karma_gray_scott_operator_split_one_step
  COMMAND karma_gray_scott_operator_split_test one-step)
add_test(NAME karma_gray_scott_temporal_lie
  COMMAND karma_gray_scott_operator_split_test lie)
add_test(NAME karma_gray_scott_temporal_strang
  COMMAND karma_gray_scott_operator_split_test strang)
add_test(NAME karma_gray_scott_rollback_exchange
  COMMAND karma_gray_scott_operator_split_test rollback-exchange)
add_test(NAME karma_gray_scott_rollback_advance
  COMMAND karma_gray_scott_operator_split_test rollback-advance)
add_test(NAME karma_gray_scott_rollback_commit
  COMMAND karma_gray_scott_operator_split_test rollback-commit)
add_test(NAME karma_gray_scott_initial_condition_transactional
  COMMAND karma_gray_scott_operator_split_test initial-condition-serial)
set_tests_properties(karma_gray_scott_initial_condition_transactional PROPERTIES TIMEOUT 20)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_gray_scott_fixed_point_null_mass_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_operator_split_test> fixed-point-null-mass)
  add_test(NAME karma_gray_scott_fixed_point_output_failure_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_operator_split_test> fixed-point-output-failure)
  add_test(NAME karma_gray_scott_fixed_point_runtime_output_failure_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_operator_split_test> fixed-point-runtime-output-failure)
  add_test(NAME karma_gray_scott_initial_condition_callback_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_operator_split_test> initial-condition-callback-mpi)
  add_test(NAME karma_gray_scott_initial_condition_nonfinite_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_operator_split_test> initial-condition-nonfinite-mpi)
  add_test(NAME karma_gray_scott_initial_condition_empty_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_gray_scott_operator_split_test> initial-condition-empty-mpi)
  set_tests_properties(
    karma_gray_scott_fixed_point_null_mass_mpi
    karma_gray_scott_fixed_point_output_failure_mpi
    karma_gray_scott_fixed_point_runtime_output_failure_mpi
    karma_gray_scott_initial_condition_callback_mpi
    karma_gray_scott_initial_condition_nonfinite_mpi
    karma_gray_scott_initial_condition_empty_mpi
    PROPERTIES TIMEOUT 20)
endif()
set_tests_properties(karma_gray_scott_operator_split_one_step karma_gray_scott_rollback_exchange
  karma_gray_scott_rollback_advance karma_gray_scott_rollback_commit PROPERTIES TIMEOUT 120)
set_tests_properties(karma_gray_scott_temporal_lie karma_gray_scott_temporal_strang PROPERTIES TIMEOUT 30)

add_executable(karma_transient_reaction_diffusion_assembly_test transient_reaction_diffusion_assembly_test.C)
target_link_libraries(karma_transient_reaction_diffusion_assembly_test PRIVATE KARMA::karma)
add_test(NAME karma_transient_reaction_diffusion_assembly COMMAND karma_transient_reaction_diffusion_assembly_test)

add_executable(karma_transient_reaction_diffusion_component_test transient_reaction_diffusion_component_test.C)
target_link_libraries(karma_transient_reaction_diffusion_component_test PRIVATE KARMA::karma)
add_test(NAME karma_transient_reaction_diffusion_component COMMAND karma_transient_reaction_diffusion_component_test)

add_executable(karma_transient_reaction_diffusion_mms_test transient_reaction_diffusion_mms_test.C)
target_link_libraries(karma_transient_reaction_diffusion_mms_test PRIVATE KARMA::karma)
add_test(NAME karma_transient_reaction_diffusion_mms COMMAND karma_transient_reaction_diffusion_mms_test)
set_tests_properties(karma_transient_reaction_diffusion_mms PROPERTIES LABELS serial TIMEOUT 120)

if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_transient_reaction_diffusion_component_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_transient_reaction_diffusion_component_test>)
  set_tests_properties(karma_transient_reaction_diffusion_component_mpi PROPERTIES TIMEOUT 30)
endif()

add_executable(karma_advection_flux_test advection_flux_test.C)
target_link_libraries(karma_advection_flux_test PRIVATE KARMA::karma)
add_test(NAME karma_advection_flux COMMAND karma_advection_flux_test)
add_test(NAME karma_advection_flux_mixed_tet_prism COMMAND karma_advection_flux_test --mixed-tet-prism)

add_library(karma_advection_dg_face_mutation OBJECT
  ../src/physics/advection-reaction/advection_reaction_system_residual.C)
target_compile_definitions(karma_advection_dg_face_mutation PRIVATE
  KARMA_TEST_DUPLICATE_INTERIOR_FACE_TRAVERSAL)
target_link_libraries(karma_advection_dg_face_mutation PRIVATE KARMA::karma)
add_executable(karma_advection_dg_face_mutation_test
  advection_flux_test.C
  $<TARGET_OBJECTS:karma_advection_dg_face_mutation>)
target_compile_definitions(karma_advection_dg_face_mutation_test PRIVATE KARMA_DG_FACE_MUTATION_TEST)
target_link_libraries(karma_advection_dg_face_mutation_test PRIVATE KARMA::karma)

add_executable(karma_advection_mms_test advection_mms_test.C)
target_link_libraries(karma_advection_mms_test PRIVATE KARMA::karma)
add_test(NAME karma_advection_mms COMMAND karma_advection_mms_test)
set_tests_properties(karma_advection_mms PROPERTIES LABELS serial)

add_executable(karma_algebraic_system_test algebraic_system_test.C)
target_link_libraries(karma_algebraic_system_test PRIVATE KARMA::karma)
add_test(NAME karma_algebraic_system COMMAND karma_algebraic_system_test)

add_executable(karma_ode_system_test ode_system_test.C)
target_link_libraries(karma_ode_system_test PRIVATE KARMA::karma)
add_test(NAME karma_ode_system COMMAND karma_ode_system_test)
if(MPIEXEC_EXECUTABLE)
  add_test(NAME karma_advection_flux_replicated_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_advection_flux_test>)
  set_tests_properties(karma_advection_flux_replicated_mpi PROPERTIES TIMEOUT 30)
  add_test(NAME karma_advection_asymmetric_local_element_map_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_advection_flux_test> --asymmetric-local-element-map-failure)
  set_tests_properties(karma_advection_asymmetric_local_element_map_mpi PROPERTIES TIMEOUT 20)
  foreach(ranks 2 3 4)
    foreach(failure config model connectivity)
      add_test(NAME karma_advection_asymmetric_${failure}_mpi_${ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:karma_advection_flux_test> --asymmetric-init-${failure})
      set_tests_properties(karma_advection_asymmetric_${failure}_mpi_${ranks} PROPERTIES TIMEOUT 20)
    endforeach()
  endforeach()
  foreach(ranks 2 3 4)
    add_test(NAME karma_algebraic_system_mpi_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_algebraic_system_test>)
  endforeach()
  foreach(failure empty name components)
    add_test(NAME karma_algebraic_asymmetric_${failure}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_algebraic_system_test> --asymmetric-${failure})
    set_tests_properties(karma_algebraic_asymmetric_${failure}_mpi PROPERTIES TIMEOUT 20)
  endforeach()
  foreach(ranks 2 3 4)
    add_test(NAME karma_ode_system_mpi_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_ode_system_test>)
    set_tests_properties(karma_ode_system_mpi_${ranks} PROPERTIES TIMEOUT 30)
  endforeach()
  foreach(failure empty name model)
    add_test(NAME karma_ode_asymmetric_init_${failure}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_ode_system_test> --asymmetric-init-${failure})
    set_tests_properties(karma_ode_asymmetric_init_${failure}_mpi PROPERTIES TIMEOUT 20)
  endforeach()
  foreach(callback rhs jacobian mass)
    foreach(failure throw nonfinite)
      set(suffix "")
      if(failure STREQUAL "nonfinite")
        set(suffix "-nonfinite")
      endif()
      add_test(NAME karma_ode_asymmetric_${callback}_${failure}_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:karma_ode_system_test> --asymmetric-${callback}${suffix})
      set_tests_properties(karma_ode_asymmetric_${callback}_${failure}_mpi PROPERTIES TIMEOUT 20)
    endforeach()
  endforeach()
  foreach(scalar mass-time residual-time jacobian-time shift initial-time final-time dt)
    foreach(failure different nonfinite)
      set(suffix "")
      if(failure STREQUAL "nonfinite")
        set(suffix "-nonfinite")
      endif()
      add_test(NAME karma_ode_asymmetric_scalar_${scalar}_${failure}_mpi
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:karma_ode_system_test> --asymmetric-scalar-${scalar}${suffix})
      set_tests_properties(karma_ode_asymmetric_scalar_${scalar}_${failure}_mpi PROPERTIES TIMEOUT 20)
    endforeach()
  endforeach()
  foreach(failure solution-null residual-null solution-size solution-comm residual-size residual-comm jacobian-size jacobian-comm jacobian-presence)
    add_test(NAME karma_advection_asymmetric_object_${failure}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_advection_flux_test> --asymmetric-object-${failure})
    set_tests_properties(karma_advection_asymmetric_object_${failure}_mpi PROPERTIES TIMEOUT 20)
  endforeach()
endif()

add_executable(karma_boundary_condition_test boundary_condition_test.C)
target_link_libraries(karma_boundary_condition_test PRIVATE KARMA::karma)
add_test(NAME karma_boundary_condition COMMAND karma_boundary_condition_test)

add_executable(karma_legacy_bc_api_probe legacy_bc_api_probe.C)
target_link_libraries(karma_legacy_bc_api_probe PRIVATE KARMA::karma)
if(MPIEXEC_EXECUTABLE)
  foreach(ranks 2 3 4)
    add_test(NAME karma_assembly_mpi_${ranks}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
              $<TARGET_FILE:karma_assembly_test>)
  endforeach()
  add_test(NAME karma_mass_assembly_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_mass_assembly_test>)
  foreach(failure matrix-null matrix-size matrix-comm matrix-mapping input-null input-size input-comm input-mapping output-null output-size output-comm output-mapping rhs-null rhs-size rhs-comm rhs-mapping solution-null solution-size solution-comm solution-mapping dirichlet-count dirichlet-list quadrature dirichlet-pointer dirichlet-range)
    add_test(NAME karma_mass_asymmetric_${failure}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_mass_assembly_test> --asymmetric-${failure})
    set_tests_properties(karma_mass_asymmetric_${failure}_mpi PROPERTIES TIMEOUT 20)
  endforeach()
  add_test(NAME karma_diffusion_assembly_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_diffusion_assembly_test>)
  set_tests_properties(karma_diffusion_assembly_mpi PROPERTIES TIMEOUT 30)
  add_test(NAME karma_diffusion_replicated_natural_boundary_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_diffusion_assembly_test> --replicated-natural-boundary)
  set_tests_properties(karma_diffusion_replicated_natural_boundary_mpi PROPERTIES TIMEOUT 30)
  add_test(NAME karma_diffusion_replicated_dirichlet_boundary_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_diffusion_assembly_test> --replicated-dirichlet-boundary)
  set_tests_properties(karma_diffusion_replicated_dirichlet_boundary_mpi PROPERTIES TIMEOUT 30)
  foreach(failure solution-null solution-size solution-comm solution-mapping residual-null residual-size residual-comm residual-mapping jacobian-size jacobian-comm jacobian-mapping jacobian-presence boundary-presence boundary-dirichlet boundary-natural boundary-missing-id boundary-duplicate-id)
    add_test(NAME karma_diffusion_asymmetric_${failure}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_diffusion_assembly_test> --asymmetric-${failure})
    set_tests_properties(karma_diffusion_asymmetric_${failure}_mpi PROPERTIES TIMEOUT 20)
  endforeach()
  foreach(failure name quadrature continuity registry config material)
    add_test(NAME karma_diffusion_lifecycle_asymmetric_${failure}_mpi
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:karma_diffusion_lifecycle_test> --asymmetric-init-${failure})
    set_tests_properties(karma_diffusion_lifecycle_asymmetric_${failure}_mpi PROPERTIES TIMEOUT 20)
  endforeach()
  add_test(NAME karma_boundary_condition_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:karma_boundary_condition_test>)
endif()

get_property(karma_registered_tests DIRECTORY PROPERTY TESTS)
foreach(test_name IN LISTS karma_registered_tests)
  if(test_name MATCHES "^karma_moab_io($|_)")
    set_tests_properties(${test_name} PROPERTIES RESOURCE_LOCK karma_moab_io_files)
  endif()
endforeach()

# The same binary under MPI, to confirm the parallel link is sound.
if(MPIEXEC_EXECUTABLE)
	add_test(NAME karma_smoke_mpi
		COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
		        $<TARGET_FILE:karma_smoke_test>)

  get_property(karma_registered_tests DIRECTORY PROPERTY TESTS)
  foreach(test_name IN LISTS karma_registered_tests)
    if(test_name MATCHES "_mpi($|_)")
      get_test_property(${test_name} TIMEOUT test_timeout)
      if(test_timeout STREQUAL "NOTFOUND")
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 120)
      endif()
    endif()
  endforeach()

  # Each section is a fraction of the ~154s the single karma_fe MPI test took,
  # but the timeout stays generous: these now run alongside other tests rather
  # than alone, so wall-clock per test depends on machine load.
  foreach(section ${karma_fe_sections})
    string(REPLACE "-" "_" section_name "${section}")
    set_tests_properties(karma_fe_${section_name}_replicated_spmd_mpi PROPERTIES TIMEOUT 600)
  endforeach()
  foreach(section ${karma_fe_face_parallel_sections})
    string(REPLACE "-" "_" section_name "${section}")
    set_tests_properties(karma_fe_face_${section_name}_replicated_spmd_mpi PROPERTIES TIMEOUT 600)
  endforeach()
  set_tests_properties(karma_boundary_empty_partition_rank_mpi
    karma_fe_face_shared_hex_replicated_mpi PROPERTIES TIMEOUT 20)
  foreach(failure size layout state nonfinite)
    set_tests_properties(karma_moab_io_asymmetric_restore_vector_${failure}_mpi PROPERTIES TIMEOUT 20)
  endforeach()

  function(karma_assert_test_timeout test_name expected_timeout)
    get_test_property(${test_name} TIMEOUT actual_timeout)
    if(NOT actual_timeout STREQUAL "${expected_timeout}")
      message(FATAL_ERROR
        "${test_name} has TIMEOUT ${actual_timeout}; expected ${expected_timeout}")
    endif()
  endfunction()
  foreach(test_name IN LISTS karma_registered_tests)
    if(test_name MATCHES "^karma_moab_io($|_)")
      get_test_property(${test_name} RESOURCE_LOCK resource_lock)
      if(NOT resource_lock STREQUAL "karma_moab_io_files")
        message(FATAL_ERROR
          "${test_name} has RESOURCE_LOCK ${resource_lock}; expected karma_moab_io_files")
      endif()
    endif()
  endforeach()
  karma_assert_test_timeout(karma_ode_asymmetric_rhs_throw_mpi 20)
  karma_assert_test_timeout(karma_diffusion_lifecycle_asymmetric_name_mpi 20)
  karma_assert_test_timeout(karma_advection_asymmetric_object_solution-null_mpi 20)
  karma_assert_test_timeout(karma_moab_io_invalid_temporary_path_mpi 30)
  if(KARMA_MBPART_EXECUTABLE)
    karma_assert_test_timeout(karma_mass_partitioned_2 120)
  endif()
  karma_assert_test_timeout(karma_fe_quadrature_replicated_spmd_mpi 600)
endif()
