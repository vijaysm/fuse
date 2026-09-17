# fuse:: library targets, one per layer (docs/dev/layering.md). Each layer links
# only the layers below it; the layering gate checks the includes, this file
# checks the link graph -- both must agree.

# ---------------------------------------------------------------- base
# Exact time, units, ids, versions, diagnostics, canonical bytes. Almost
# header-only; version.cpp is the one translation unit. Depends on nothing
# above Tier 0: Kokkos is linked for the execution-space vocabulary later
# layers need in their public headers, fmt for Diagnostics formatting.
set(FUSE_HAVE_MPI    ${FUSE_ENABLE_MPI})
set(FUSE_HAVE_KOKKOS ON)
set(FUSE_HAVE_MOAB   ${FUSE_ENABLE_MOAB})
set(FUSE_HAVE_LOWFIVE ${FUSE_ENABLE_LOWFIVE})
if(FUSE_LINALG_BACKEND STREQUAL "eigen")
  set(FUSE_LINALG_EIGEN 1)
else()
  set(FUSE_LINALG_EIGEN 0)
endif()
set(FUSE_HAVE_PETSC  ${FUSE_ENABLE_PETSC})
configure_file(${PROJECT_SOURCE_DIR}/include/fuse/base/config.hpp.in
               ${PROJECT_BINARY_DIR}/include/fuse/base/config.hpp @ONLY)

add_library(fuse-coupler-base
  base/backtrace.cpp
  base/error.cpp
  base/version.cpp
)
add_library(fuse::base ALIAS fuse-coupler-base)
target_include_directories(fuse-coupler-base PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(fuse-coupler-base PUBLIC cxx_std_20)
# Exceptions on by default; OFF turns every FUSE_THROW/ASSERT/REFUSE into
# report-then-abort after the same on-error policy has run. The types and
# call sites are unconditional either way (docs/dev/conventions.md).
target_link_libraries(fuse-coupler-base PUBLIC ${CMAKE_DL_LIBS})
target_link_libraries(fuse-coupler-base PUBLIC Kokkos::kokkos fmt::fmt)
# Prepended so the resolved Kokkos's headers precede every other prefix on the
# command line (cmake/FuseDependencies.cmake explains the shadowing hazard).
target_include_directories(fuse-coupler-base BEFORE PUBLIC ${FUSE_KOKKOS_INCLUDE_DIRS})
set_target_properties(fuse-coupler-base PROPERTIES
  CXX_VISIBILITY_PRESET hidden      # the C ABI exports one symbol per ABI; nothing else leaks
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- model
# The semantic model, the N-ary coupling graph, its validation and the canonical
# descriptor -- the narrow waist. Header-heavy by design: everything the waist
# declares must be inspectable by a front-end without linking the runtime.
add_library(fuse-coupler-model
  model/semantics.cpp
  model/transform.cpp
  model/compatibility.cpp
  model/multiscale.cpp
  model/yaml_io.cpp
)
add_library(fuse::model ALIAS fuse-coupler-model)
target_link_libraries(fuse-coupler-model PUBLIC fuse::base)
# PRIVATE: the YAML parser is an implementation detail of the front-end and
# must never reach a consumer's include path or link line.
target_link_libraries(fuse-coupler-model PRIVATE yaml-cpp::yaml-cpp)
set_target_properties(fuse-coupler-model PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- ir
# Graph -> SSA operator DAG. Depends on the model and nothing above it: the
# lowering rules are a pure function of the graph, which is what makes the
# compiled IR inspectable and golden-comparable without a runtime.
add_library(fuse-coupler-ir
  ir/lowering.cpp
  ir/regions.cpp
  ir/rules_channel.cpp
  ir/rules_derived.cpp
  ir/checkpoint.cpp
)
add_library(fuse::ir ALIAS fuse-coupler-ir)
target_link_libraries(fuse-coupler-ir PUBLIC fuse::model)
set_target_properties(fuse-coupler-ir PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- schedule
# Operator DAG -> flat execution plan: exact clocks, the init-time acyclicity
# proof, window expansion, buffer assignment.
add_library(fuse-coupler-schedule
  schedule/acyclicity.cpp
  schedule/buffers.cpp
  schedule/expand.cpp
  schedule/scheduler.cpp
  schedule/slab.cpp
)
add_library(fuse::schedule ALIAS fuse-coupler-schedule)
target_link_libraries(fuse-coupler-schedule PUBLIC fuse::ir)
set_target_properties(fuse-coupler-schedule PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- linalg
# One dense least-squares solve behind one declaration, so the acceleration
# code does not know which factorisation is present. Sits beside `num` rather
# than under it: `num` calls it, nothing calls `num` from here.
add_library(fuse-coupler-linalg
  linalg/least_squares.cpp
)
add_library(fuse::linalg ALIAS fuse-coupler-linalg)
target_link_libraries(fuse-coupler-linalg PUBLIC fuse::base)
# PRIVATE: which factorisation is inside is not part of the interface, and
# Eigen must not land on a consumer's include path.
if(FUSE_LINALG_BACKEND STREQUAL "eigen")
  target_link_libraries(fuse-coupler-linalg PRIVATE Eigen3::Eigen)
endif()
set_target_properties(fuse-coupler-linalg PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- num
# Numerics: waveforms, temporal bases, convergence measures, accelerators, and
# the elementwise kernels. Depends on the model (for the iteration
# configuration types) and on nothing above it, so a numerics test needs no
# runtime and no mesh.
add_library(fuse-coupler-num
  num/accel_quasi_newton.cpp
  num/accel_relaxation.cpp
  num/kernels_inst.cpp
  num/residual_norm.cpp
)
add_library(fuse::num ALIAS fuse-coupler-num)
target_link_libraries(fuse-coupler-num PUBLIC fuse::model)
target_link_libraries(fuse-coupler-num PRIVATE fuse::linalg)
set_target_properties(fuse-coupler-num PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- transport
# Moving field values between rank sets. The copy path is always built; the MPI
# router and the rank-matching graph are compiled only in an MPI build, because
# they are the parts that name MPI types.
add_library(fuse-coupler-transport
  transport/copy_transport.cpp
  transport/lowfive_transport.cpp
)
if(FUSE_ENABLE_MPI)
  target_sources(fuse-coupler-transport PRIVATE
    transport/comm_graph.cpp
    transport/crystal_router.cpp
    transport/mpi_transport.cpp
  )
endif()
add_library(fuse::transport ALIAS fuse-coupler-transport)
target_link_libraries(fuse-coupler-transport PUBLIC fuse::schedule)
if(FUSE_ENABLE_MPI)
  target_link_libraries(fuse-coupler-transport PUBLIC MPI::MPI_CXX)
endif()
set_target_properties(fuse-coupler-transport PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- backend
# Mesh access and spatial/space-time transfer behind one interface. Only the
# serial mesh and the space-time projection are here; the mesh database and
# weight generators that need external libraries arrive at S5.
add_library(fuse-coupler-backend
  backend/serial_backend.cpp
  backend/supermesh.cpp
  backend/spacetime.cpp
  # Supermesh intersection: the real implementation needs MOAB, and the
  # file's own no-MOAB path refuses with E-REMAP-NOMOAB rather than
  # leaving the symbol undefined. S5 turns the option on.
  backend/intx_weights.cpp
)
add_library(fuse::backend ALIAS fuse-coupler-backend)
target_link_libraries(fuse-coupler-backend PUBLIC fuse::schedule)
# PRIVATE: r3d is the supermesh kernel, used inside backend/supermesh.cpp only.
# No public header mentions it, so a consumer never needs it on an include path
# and never links it -- which is what keeps the geometric kernel swappable.
target_link_libraries(fuse-coupler-backend PRIVATE r3d::r3d)
if(FUSE_ENABLE_MOAB)
  # PUBLIC, not PRIVATE: intx_weights.cpp is compiled into this library, but
  # the mesh-backend headers a consumer includes also reach MOAB types, so a
  # consumer that cannot see MOAB's headers cannot use this library at all.
  target_link_libraries(fuse-coupler-backend PUBLIC fuse::moab)
endif()
set_target_properties(fuse-coupler-backend PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- runtime
# Executing a plan: field binding, both entry modes, iteration regions.
# The top of the library -- it links everything below and nothing links it.
add_library(fuse-coupler-runtime
  runtime/runner.cpp
  runtime/iteration.cpp
  runtime/checkpoint_support.cpp
)
add_library(fuse::runtime ALIAS fuse-coupler-runtime)
target_link_libraries(fuse-coupler-runtime PUBLIC
  fuse::schedule fuse::num fuse::transport fuse::backend)
set_target_properties(fuse-coupler-runtime PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
