## The KARMA core library.
##
## Sources are listed explicitly rather than globbed, so that what is and is
## not currently built stays visible in review. The excluded sets are recorded
## at the bottom of this file.

set(KARMA_SOURCES
  coupling/karma_field_entry_encoding.C
  coupling/karma_collective_outcome.C
  coupling/karma_compatibility.C
  coupling/karma_field_transaction.C
  coupling/karma_field_view.C
  coupling/karma_semantics.C
  coupling/karma_registry.C
  coupling/karma_component_runtime.C
  coupling/karma_petsc_field_storage.C
  coupling/karma_reference_components.C
  coupling/karma_semantics_adapter.C
  coupling/karma_transform.C
  coupling/karma_semantic_version.C
  coupling/karma_component.C
  coupling/karma_field.C
  coupling/karma_field_transfer.C
  coupling/karma_transfer_map.C
  coupling/karma_field_coverage.C
  coupling/karma_conservative_transfer.C
  coupling/karma_moab_point_locator.C
  coupling/karma_distributed_point_locator.C
  coupling/karma_fem_field_evaluator.C
  coupling/karma_fem_transfer.C
  coupling/karma_compiled_fem_connection.C
  coupling/karma_transfer_diagnostics.C
  coupling/karma_quasi_interpolation.C
  coupling/karma_transfer_map_io.C
  coupling/karma_operator_split_graph.C
  coupling/karma_operator_split_checkpoint.C
  coupling/karma_fixed_point.C
  coupling/karma_transient_reaction_diffusion_adapter.C
  base/ikarma.C
	base/karma.C
	base/karma_log.C
	base/karma_print_trace.C

	enums/bc_type.C
	enums/karma_linear_operations.C
	enums/karma_nonlinear_operations.C
	enums/partitioner_type.C
	enums/physics_properties.C
	enums/runge_kutta_schemes.C

  io/karma_moab_io.C

	numerics/icollection.C
  numerics/karma_algebraic_system.C
	numerics/karma_dof_map.C
	numerics/karma_system.C
	numerics/user_sparsity.C
  numerics/fe/karma_basis.C
  numerics/fe/karma_fe_base.C
  numerics/fe/karma_geometry_map.C
  numerics/fe/karma_reference_topology.C
	numerics/algebra/linear_algebra_factory.C
  numerics/algebra/petsc_native_scatter.C
	numerics/algebra/petsc_block_matrix.C
	numerics/algebra/petsc_block_vector.C
	numerics/linear/preconditioner/petsc_block_preconditioner.C
	numerics/spatial/karma_domain.C
	numerics/spatial/karma_mesh.C

	numerics/temporal/backward_euler.C
	numerics/temporal/crank_nicolson.C
	numerics/temporal/dopri45.C
	numerics/temporal/forward_euler.C
	numerics/temporal/grk4a.C
	numerics/temporal/grk4sn.C
	numerics/temporal/grk4t.C
	numerics/temporal/karma_ts_base.C
	numerics/temporal/kutta22.C
	numerics/temporal/kutta44.C
	numerics/temporal/rk.C
	numerics/temporal/rodasp.C
	numerics/temporal/ros22.C
	numerics/temporal/ros3p.C
	numerics/temporal/rosenbrock.C
	numerics/temporal/sdirk22.C
	numerics/temporal/sdirk23.C
	numerics/temporal/sdirk33.C
	numerics/temporal/sdirk332.C
	numerics/temporal/sdirk34.C
	numerics/temporal/sdirk44.C
	numerics/temporal/sdirk45.C
	numerics/temporal/sdirk543.C
	numerics/temporal/sdirk643.C
	numerics/temporal/trbe.C

  # Transitional legacy seam: karma_physics_base.C remains intact for excluded
  # ISystem consumers while the active physics stack remains dependency-light.
	physics/common/karma_boundary_condition.C
  physics/common/karma_physics_bc_manager.C
  physics/common/impls/generic_mass_matrix.C
  physics/common/impls/generic_mass_matrix_multiply.C
  physics/common/impls/generic_mass_matrix_solve.C
  physics/diffusion-reaction/diffusion_reaction_mass_operator.C
  physics/diffusion-reaction/diffusion_reaction_preconditioner_operator.C
  physics/diffusion-reaction/diffusion_reaction_preconditioner_point_jacobi_solve.C
  physics/diffusion-reaction/diffusion_reaction_system.C
  physics/diffusion-reaction/diffusion_reaction_system_jacobian_matrix.C
  physics/diffusion-reaction/diffusion_reaction_system_operator.C
  physics/diffusion-reaction/diffusion_reaction_system_property_manager.C
  physics/diffusion-reaction/diffusion_reaction_system_residual.C
  physics/diffusion-reaction/gray_scott_model.C
  physics/diffusion-reaction/transient_reaction_diffusion_assembly.C
  physics/diffusion-reaction/transient_reaction_diffusion_component.C
  physics/advection-reaction/advection_reaction_mass_operator.C
  physics/advection-reaction/advection_reaction_preconditioner_operator.C
  physics/advection-reaction/advection_reaction_system.C
  physics/advection-reaction/advection_reaction_system_jacobian_matrix.C
  physics/advection-reaction/advection_reaction_system_operator.C
  physics/advection-reaction/advection_reaction_system_property_manager.C
  physics/advection-reaction/advection_reaction_system_residual.C
  physics/ode-system/ode_system.C
  physics/ode-system/ode_system_operator.C
  physics/ode-system/ode_system_property_manager.C
  physics/ode-system/ode_system_mass_operator.C

	utils/lapack_utils.C
	utils/static_vector.C
	utils/typename_traits.C
)

if(KARMA_ENABLE_SHARED)
	add_library(karma SHARED ${KARMA_SOURCES})
else()
	add_library(karma STATIC ${KARMA_SOURCES})
endif()
add_library(KARMA::karma ALIAS karma)

# New coupling code is warning-clean by policy. Legacy migrated sources remain
# outside this target-level Werror boundary until their warnings are addressed.
set(_karma_coupling_sources
  coupling/karma_field_entry_encoding.C
  coupling/karma_collective_outcome.C
  coupling/karma_compatibility.C
  coupling/karma_field_transaction.C
  coupling/karma_field_view.C
  coupling/karma_semantics.C
  coupling/karma_registry.C
  coupling/karma_component_runtime.C
  coupling/karma_petsc_field_storage.C
  coupling/karma_reference_components.C
  coupling/karma_semantics_adapter.C
  coupling/karma_transform.C
  coupling/karma_semantic_version.C
  coupling/karma_component.C
  coupling/karma_field.C
  coupling/karma_field_coverage.C
  coupling/karma_field_transfer.C
  coupling/karma_fem_field_evaluator.C
  coupling/karma_fem_transfer.C
  coupling/karma_compiled_fem_connection.C
  coupling/karma_moab_point_locator.C
  coupling/karma_distributed_point_locator.C
  coupling/karma_quasi_interpolation.C
  coupling/karma_transfer_diagnostics.C
  coupling/karma_transfer_map.C
  coupling/karma_transfer_map_io.C
  coupling/karma_operator_split_graph.C
  coupling/karma_operator_split_checkpoint.C
  coupling/karma_transient_reaction_diffusion_adapter.C
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set_source_files_properties(${_karma_coupling_sources} PROPERTIES COMPILE_OPTIONS "-Werror")
endif()

# The headers are spread over a directory tree and included by bare filename,
# so every directory under include/ has to be on the search path.
file(GLOB_RECURSE _karma_include_dirs
	LIST_DIRECTORIES true
	"${PROJECT_SOURCE_DIR}/include/*")
set(KARMA_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
foreach(_d ${_karma_include_dirs})
	if(IS_DIRECTORY "${_d}")
		list(APPEND KARMA_INCLUDE_DIRS "${_d}")
	endif()
endforeach()

# The generated karma_config.h must come first: a stale copy generated by the
# old waf build is still committed under include/base/, and would otherwise
# shadow it (declaring KARMA_HAVE_LIBMESH and paths from another machine).
target_include_directories(karma BEFORE PUBLIC
	"${PROJECT_BINARY_DIR}/include")
target_include_directories(karma PUBLIC
	${KARMA_INCLUDE_DIRS})

target_link_libraries(karma PUBLIC
	PkgConfig::PETSC
	MPI::MPI_CXX
	LAPACK::LAPACK)

if(KARMA_HAVE_MOAB)
	target_link_libraries(karma PUBLIC PkgConfig::MOAB)
endif()

# Native KARMA transfer does not link or call MOAB Coupler. The optional
# Coupler probe remains available for capability comparison and future adapter work.

if(KARMA_HAVE_EIGEN)
	target_include_directories(karma PUBLIC ${EIGEN_INCLUDE_DIR})
endif()

set_target_properties(karma PROPERTIES
	VERSION   ${PROJECT_VERSION}
	SOVERSION ${PROJECT_VERSION_MAJOR}
	OUTPUT_NAME karma)

## ---------------------------------------------------------------------------
## Stable C ABI, version 1 (candidate)
## ---------------------------------------------------------------------------
##
## A separate shared object on purpose. The ABI is the only stable surface, so
## it is the only target built with hidden visibility and an explicit export
## list; libkarma keeps its C++ symbols visible because 70 test executables and
## the examples link against them. Building both from one target would either
## hide those symbols or export the ABI's implementation details.
##
## libkarma is linked PRIVATE so a C consumer does not inherit PETSc, MOAB, or
## MPI include paths and link flags.

add_library(karma_abi SHARED
	abi/karma_abi.C
	abi/karma_component_abi.C)
add_library(KARMA::karma_abi ALIAS karma_abi)

set_source_files_properties(abi/karma_abi.C abi/karma_component_abi.C
	PROPERTIES COMPILE_OPTIONS "-Werror")

target_include_directories(karma_abi PUBLIC
	"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/abi>"
	"$<INSTALL_INTERFACE:include/karma/abi>")
target_link_libraries(karma_abi PRIVATE karma)

set_target_properties(karma_abi PROPERTIES
	CXX_VISIBILITY_PRESET hidden
	VISIBILITY_INLINES_HIDDEN ON
	VERSION   ${PROJECT_VERSION}
	SOVERSION ${PROJECT_VERSION_MAJOR}
	OUTPUT_NAME karma_abi
	# So the installed ABI library finds libkarma beside it without the
	# consumer setting a library search path by hand.
	INSTALL_RPATH "$<IF:$<BOOL:${APPLE}>,@loader_path,$ORIGIN>")

if(APPLE)
	target_link_options(karma_abi PRIVATE
		"-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/abi/karma_exports.sym")
else()
	target_link_options(karma_abi PRIVATE
		"-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/abi/karma_exports.map")
endif()

install(TARGETS karma_abi
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)

install(TARGETS karma
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
	DESTINATION include/karma
	FILES_MATCHING PATTERN "*.h")

## ---------------------------------------------------------------------------
## Not yet built
## ---------------------------------------------------------------------------
##
## Reported in the configure summary so the gap is never silent.
##
## The configure summary computes the current active and discovered .C source counts.
## Untracked .cpp files remain outside this census.
## The prior per-dependency counts became historical as sources migrated into this list;
## remaining exclusions still span libMesh, PETScExt, TinyXPath, PETSc API
## drift, and pre-existing defects and require a fresh file-level inventory.

list(LENGTH KARMA_SOURCES _n)
file(GLOB_RECURSE _all_sources "${PROJECT_SOURCE_DIR}/src/*.C")
list(LENGTH _all_sources _total)
set(KARMA_SOURCE_COUNT ${_n} PARENT_SCOPE)
set(KARMA_SOURCE_TOTAL ${_total} PARENT_SCOPE)
