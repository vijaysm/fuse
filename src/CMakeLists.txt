# fuse:: library targets, one per layer (docs/dev/layering.md). Each layer links
# only the layers below it; the layering gate checks the includes, this file
# checks the link graph -- both must agree.

# ---------------------------------------------------------------- base
# Exact time, units, ids, versions, diagnostics, canonical bytes. Almost
# header-only; version.cpp is the one translation unit. Depends on nothing
# above Tier 0: Kokkos is linked for the execution-space vocabulary later
# layers need in their public headers, fmt for Diagnostics formatting.
set(FUSE_HAVE_MPI    ${FUSE_ENABLE_MPI})
include(${PROJECT_SOURCE_DIR}/cmake/FuseMpiGpuAware.cmake)
if(FUSE_ENABLE_MPI)
  fuse_probe_mpi_gpu_aware(FUSE_MPI_GPU_AWARE)
else()
  set(FUSE_MPI_GPU_AWARE 0)
endif()
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

# C++20 is required on EVERY layer below, not inherited from one of them: the
# requirement was originally stated once (PUBLIC, on base) and propagated only
# while every layer linked base. A layer that stopped would fall back to the
# compiler's default, and the symptom would be a template error inside a header
# rather than a message naming the standard.
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
  # `install(EXPORT ... NAMESPACE fuse::)` (below) prefixes fuse:: onto the
  # TARGET's own name, not onto the fuse::base ALIAS above -- an installed
  # consumer would otherwise see fuse::fuse-coupler-base instead of fuse::base
  # and every other layer's alias, silently diverging from the in-tree name
  # docs/dev/conventions.md promises (REVIEW.md R-25/BT-F9). EXPORT_NAME makes
  # the two agree; every layer below sets the same property.
  EXPORT_NAME base
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
target_compile_features(fuse-coupler-model PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-model PUBLIC fuse::base)
# PRIVATE: the YAML parser is an implementation detail of the front-end and
# must never reach a consumer's include path or link line.
# $<BUILD_LOCAL_INTERFACE:...>, not a bare PRIVATE reference: yaml-cpp is
# fetched on any machine without an installed copy (FuseDependencies.cmake),
# which makes it an in-tree build target with no package of its own to
# install(EXPORT) alongside -- CMake refuses to export fuse-coupler-model
# otherwise ("requires target yaml-cpp that is not in any export set",
# REVIEW.md R-25/BT-F9). This tells CMake the dependency is real for OUR
# build only and never becomes part of what an installed consumer must
# supply through this package; a consumer linking fuse::model with a
# fetched yaml-cpp must find/link yaml-cpp itself (documented in
# cmake/fuse-couplerConfig.cmake.in).
target_link_libraries(fuse-coupler-model PRIVATE $<BUILD_LOCAL_INTERFACE:yaml-cpp::yaml-cpp>)
set_target_properties(fuse-coupler-model PROPERTIES
  EXPORT_NAME model
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
  ir/window_lowering.cpp
  ir/rules_channel.cpp
  ir/rules_derived.cpp
  ir/checkpoint.cpp
)
add_library(fuse::ir ALIAS fuse-coupler-ir)
target_compile_features(fuse-coupler-ir PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-ir PUBLIC fuse::model)
set_target_properties(fuse-coupler-ir PROPERTIES
  EXPORT_NAME ir
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
  schedule/window.cpp
)
add_library(fuse::schedule ALIAS fuse-coupler-schedule)
target_compile_features(fuse-coupler-schedule PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-schedule PUBLIC fuse::ir)
set_target_properties(fuse-coupler-schedule PROPERTIES
  EXPORT_NAME schedule
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- linalg
# One dense least-squares solve behind one declaration, so the acceleration
# code does not know which factorisation is present. Sits beside `num` rather
# than under it: `num` calls it, nothing calls `num` from here.
#
# "Containers" track: the row-distributed DistVector/DistSparseMatrix pair
# (comm_serial.cpp is the always-built, one-participant Reducer/
# GhostExchanger; comm_mpi.cpp is the MPI implementation of the same two
# facades, added below only under FUSE_ENABLE_MPI, the same conditional
# fuse::transport uses for its own MPI-only translation units).
add_library(fuse-coupler-linalg
  linalg/least_squares.cpp
  linalg/comm_serial.cpp
  linalg/dist_vector.cpp
  linalg/dist_sparse_matrix.cpp
)
if(FUSE_ENABLE_MPI)
  target_sources(fuse-coupler-linalg PRIVATE linalg/comm_mpi.cpp)
endif()
add_library(fuse::linalg ALIAS fuse-coupler-linalg)
target_compile_features(fuse-coupler-linalg PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-linalg PUBLIC fuse::base)
# PRIVATE: which factorisation is inside is not part of the interface, and
# Eigen must not land on a consumer's include path.
if(FUSE_LINALG_BACKEND STREQUAL "eigen")
  target_link_libraries(fuse-coupler-linalg PRIVATE Eigen3::Eigen)
endif()
# PRIVATE: comm_mpi.cpp is the only translation unit in this library that
# names an MPI type (comm.hpp's own header comment); no public header here
# includes mpi.h, so a consumer of fuse::linalg never needs this link.
if(FUSE_ENABLE_MPI)
  target_link_libraries(fuse-coupler-linalg PRIVATE MPI::MPI_CXX)
endif()
set_target_properties(fuse-coupler-linalg PROPERTIES
  EXPORT_NAME linalg
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
target_compile_features(fuse-coupler-num PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-num PUBLIC fuse::model)
target_link_libraries(fuse-coupler-num PRIVATE fuse::linalg)
set_target_properties(fuse-coupler-num PROPERTIES
  EXPORT_NAME num
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

# The accept/connect launch mode, as cmake/FuseDependencies.cmake's probe left
# it. PUBLIC because connect.hpp's `kConnectAcceptEnabled` reads it, and a
# consumer that saw a different value than the library was built with would
# take the opposite branch at the call site from the one the library serves.
target_compile_definitions(fuse-coupler-transport PUBLIC
  FUSE_CONNECT_ACCEPT_ENABLED=$<BOOL:${FUSE_ENABLE_CONNECT_ACCEPT}>)
target_compile_features(fuse-coupler-transport PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-transport PUBLIC fuse::schedule)
if(FUSE_ENABLE_MPI)
  target_link_libraries(fuse-coupler-transport PUBLIC MPI::MPI_CXX)
endif()
set_target_properties(fuse-coupler-transport PROPERTIES
  EXPORT_NAME transport
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---- S6 phase 3, "MPI connect" track: begin -------------------------------
# transport/connect.cpp: fuse::transport::connect() (inter-executable rank
# rendezvous). Added unconditionally, not inside the FUSE_ENABLE_MPI block
# above, for the same reason backend/nodal_remap_distributed.cpp is (see
# that file's own comment in this script): its MPI-calling body is entirely
# inside `#if FUSE_HAVE_MPI`, so it compiles to an empty translation unit
# without MPI, and its MPI-free helpers (validate_launch_layout,
# check_mpmd_layout, wait_for_port_file) are header-only inline functions in
# connect.hpp, needing no library source at all to be tier-0 testable.
target_sources(fuse-coupler-transport PRIVATE transport/connect.cpp)
# ---- S6 phase 3, "MPI connect" track: end ---------------------------------

# ---- S6 phase 3, "LowFive" track: begin -----------------------------------
# transport/lowfive_transport.cpp is already in the add_library() sources
# list above (every configuration, per that file's own top-of-file comment);
# what was missing is the actual link: without this, the #include <hdf5.h>
# and #include <lowfive/...> inside its FUSE_HAVE_LOWFIVE-guarded body have
# no include path and no library to resolve against the moment anyone
# actually turns FUSE_ENABLE_LOWFIVE on (cmake/FuseDependencies.cmake's
# fuse::lowfive target exists; nothing attached it to a consumer before this).
if(FUSE_ENABLE_LOWFIVE)
  # fuse_link_lowfive_with_fmt_fix (cmake/FuseDependencies.cmake, defined
  # near the top of this file) links fuse::lowfive AND applies the
  # order-independent fmt fix every file in THIS target needs -- not
  # just lowfive_transport.cpp, which is the only one that actually
  # names a <lowfive/...> header, but comm_graph.cpp/connect.cpp/copy_
  # transport.cpp/crystal_router.cpp/mpi_transport.cpp too, since CMake
  # applies a target's own include directories to every one of its
  # sources, and every one of them transitively includes fuse/base/
  # result.hpp (via fuse/transport/transport.hpp) and needs fmt 12.
  # Measured, this session, from a FRESH configure with /opt/lowfive;
  # /opt/hdf5 FIRST on CMAKE_PREFIX_PATH: EVERY source in this target
  # failed with an identical fmt/format-inl.h consteval error, including
  # lowfive_transport.cpp itself -- contradicting this track's own
  # earlier assumption ("lowfive_transport.cpp compiles because it
  # includes no FUSE fmt-using header": false, it includes
  # transport.hpp, which includes result.hpp directly).
  fuse_link_lowfive_with_fmt_fix(fuse-coupler-transport)
  # LOWFIVE_HAVE_DIST (cmake/FindLowFive.cmake): whether a liblowfive-dist
  # archive was found alongside the main library. lowfive_transport.cpp's
  # distributed-mode code (set_intercomm/serve_as_producer/receive_as_consumer)
  # needs BOTH this and FUSE_HAVE_MPI; without this define it compiles out
  # even in an MPI build, rather than fail the final link the way
  # fuse_transport_lowfive_test did against this lab's own LowFive installs
  # before this check existed (neither ships DistMetadataVOL in the main
  # library -- see cmake/FindLowFive.cmake's own comment).
  if(LOWFIVE_HAVE_DIST)
    target_compile_definitions(fuse-coupler-transport PRIVATE FUSE_LOWFIVE_DIST_LIB_FOUND=1)
  endif()

  # ---- S6 phase 3 (Conformance, "LowFive" track), Unit 3b: the loadable
  # HDF5 VOL plugin -----------------------------------------------------
  # TWO OBJECT libraries with DISJOINT dependencies, not one translation
  # unit linking both fuse::model and fuse::lowfive -- the lead's own
  # ruling after a from-clean-configure build failure a single-TU version
  # (and, separately, a single-link-order "fix" to it) both missed:
  # lowfive_plugin_internal.hpp's own header comment has the full account
  # of why ANY translation unit seeing both a FUSE header that needs fmt
  # 12 (fuse/base/result.hpp) and a LowFive header (whose own include
  # directory carries a bundled, incompatible fmt 8.1.1) fails to
  # compile, REGARDLESS of target_link_libraries order -- the only
  # configure-order-independent fix is to never let one translation unit
  # see both at all. OBJECT, not merely "two .cpp files in one target":
  # a target's own include directories apply to EVERY one of its
  # sources (the exact mechanism that made fuse-coupler-transport's
  # OTHER, non-LowFive files fail too, just above), so the two
  # halves need their OWN, separate targets with their own, separate
  # dependencies -- then combined into the one SHARED library HDF5's
  # loader actually dlopen()s.
  add_library(fuse-lowfive-vol-graph OBJECT transport/lowfive_plugin_graph.cpp)
  target_link_libraries(fuse-lowfive-vol-graph PRIVATE fuse::model fuse::base)
  # NEVER fuse::lowfive here -- see the block comment above. MPI::MPI_CXX
  # only for <mpi.h>'s own include path (this file's #if FUSE_HAVE_MPI
  # body, world-rank/world-size queries only) -- not fuse::lowfive's own
  # MPI dependency, which this file never touches.
  if(FUSE_ENABLE_MPI AND TARGET MPI::MPI_CXX)
    target_link_libraries(fuse-lowfive-vol-graph PRIVATE MPI::MPI_CXX)
  endif()

  add_library(fuse-lowfive-vol-impl OBJECT transport/lowfive_plugin_vol.cpp)
  target_link_libraries(fuse-lowfive-vol-impl PRIVATE fuse::lowfive fuse::base)
  # fuse::base ONLY for fuse/base/config.hpp's own FUSE_HAVE_* macros
  # (zero includes of its own, by construction -- config.hpp.in's own
  # generated content is #define lines only); NEVER fuse::model.

  if(LOWFIVE_HAVE_DIST)
    target_compile_definitions(fuse-lowfive-vol-graph PRIVATE FUSE_LOWFIVE_DIST_LIB_FOUND=1)
    target_compile_definitions(fuse-lowfive-vol-impl PRIVATE FUSE_LOWFIVE_DIST_LIB_FOUND=1)
  endif()

  # SHARED, not the default STATIC every other fuse-coupler-* target
  # uses: HDF5's HDF5_PLUGIN_PATH loader dlopen()s this file directly --
  # a static archive is not a thing HDF5's loader can load at all, the
  # same must-be-shared constraint cmake/FuseDependencies.cmake's own
  # comment records for HDF5 itself ("LowFive dlopens the VOL, and a
  # static HDF5 cannot back a runtime-loaded plugin"). Linking the two
  # OBJECT libraries (not $<TARGET_OBJECTS:...>, CMake's own older,
  # manual spelling of the same thing) also re-propagates each one's own
  # PRIVATE link requirements (fuse::lowfive's libraries, fuse::model's)
  # onto the final link step, which actually producing a loadable .so
  # needs even though neither half's OWN compile needed the other's.
  add_library(fuse-lowfive-vol SHARED)
  target_link_libraries(fuse-lowfive-vol PRIVATE fuse-lowfive-vol-graph fuse-lowfive-vol-impl)
  set_target_properties(fuse-lowfive-vol PROPERTIES
    # HDF5's own plugin-naming convention (H5PL_PATH's own scan):
    # a loadable VOL's filename is otherwise unconstrained -- this
    # name is this plugin's own choice, documented here and in the
    # ctest ENVIRONMENT that points HDF5_PLUGIN_PATH at its directory
    # (examples/CMakeLists.txt's own lowfive ctest rows).
    OUTPUT_NAME fuse_lowfive_vol
  )
  # Deliberately NOT CXX_VISIBILITY_PRESET hidden, unlike every other
  # fuse-coupler-* target in this file: the two extern "C" H5PLget_
  # plugin_* symbols (defined in lowfive_plugin_vol.cpp) ARE this
  # plugin's entire public API (HDF5's own loader contract fixes their
  # names and requires default visibility to dlsym() them).
endif()
# ---- S6 phase 3, "LowFive" track: end -------------------------------------

# ---------------------------------------------------------------- search
# fuse::search: the ArborX-backed spatial index (BVH point/box queries,
# k-nearest, and the distributed variant) fuse::backend can build spatial
# queries on (docs/dev/search.md). Sits BELOW backend in
# docs/dev/layering.md's ordering -- the reverse of fuse::partition's own
# mesh_adapter.hpp, deliberately, so that fuse::backend CAN depend on this
# layer -- and depends on fuse::base alone: no ArborX type escapes a public
# header (box_index.hpp is pimpl'd; distributed.hpp never names mpi.h), so
# nothing above this layer has to see ArborX either.
add_library(fuse-coupler-search
  search/box_index.cpp
  search/distributed.cpp
)
add_library(fuse::search ALIAS fuse-coupler-search)
target_compile_features(fuse-coupler-search PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-search PUBLIC fuse::base)
# PRIVATE: ArborX is an implementation detail of this layer only, exactly the
# way r3d is private to fuse-coupler-backend below. $<BUILD_LOCAL_INTERFACE:...>
# for the same reason fuse-coupler-model's yaml-cpp link and
# fuse-coupler-backend's r3d link use it: a FETCHED ArborX has no installable
# package of its own to sit alongside in this library's install(EXPORT) set,
# whether or not THIS configure happened to find one instead.
target_link_libraries(fuse-coupler-search PRIVATE $<BUILD_LOCAL_INTERFACE:ArborX::ArborX>)
if(FUSE_ENABLE_MPI)
  # src/search/distributed.cpp is this layer's own single MPI translation
  # unit (distributed.hpp's header comment) -- the same isolation
  # fuse::partition's collectives.cpp and fuse::backend's
  # native_migration.cpp use one layer up.
  target_link_libraries(fuse-coupler-search PRIVATE MPI::MPI_CXX)
endif()
set_target_properties(fuse-coupler-search PROPERTIES
  EXPORT_NAME search
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
  backend/linear_operator.cpp
  backend/transfer_map_io.cpp
  backend/weights_operator.cpp
  backend/spacetime.cpp
  backend/native_geometry.cpp
  backend/native_mesh.cpp
  backend/native_generators.cpp
  backend/native_backend.cpp
  backend/native_mesh_io.cpp
  backend/native_migration.cpp
  # Supermesh intersection: the real implementation needs MOAB, and the
  # file's own no-MOAB path refuses with E-REMAP-NOMOAB rather than
  # leaving the symbol undefined. S5 turns the option on.
  backend/intx_weights.cpp
  backend/tempest_weights.cpp
  # MOAB-free moving-least-squares remap weights for point clouds and for
  # mesh fields sampled at vertices/centroids (the "Point clouds" track).
  # Dependency-free like intx_weights.cpp's own no-MOAB path: needs only
  # fuse::search (below) for k-nearest and fuse::linalg (below) for the
  # local polynomial solve, neither of which needs MOAB.
  backend/mls_weights.cpp
  # Nodal-remap track, step 2: distributed rows and the exact transpose
  # (KNOWN_ISSUES D8). Its own single MPI translation unit, isolated the same
  # way native_migration.cpp is above -- the whole file body is inside
  # #if FUSE_HAVE_MPI, so it compiles to an empty translation unit without
  # MPI, exactly like transport/comm_graph.cpp.
  backend/nodal_remap_distributed.cpp
)
add_library(fuse::backend ALIAS fuse-coupler-backend)
target_compile_features(fuse-coupler-backend PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-backend PUBLIC fuse::schedule)
# fuse::search sits BELOW fuse::backend (docs/dev/layering.md), added here by
# the nodal-remap track: general_interp.hpp (header-only, included from
# consistent_interp.hpp / intx_weights.cpp) calls fuse::search::element_boxes
# (template, inline) and fuse::search::BoxIndex (pimpl'd, non-inline methods
# defined in libfuse-coupler-search) for the broad-phase point-location query
# a general (non-1D) nodal basis needs. PUBLIC because BoxIndex's symbols must
# resolve in any executable that links fuse::backend.
target_link_libraries(fuse-coupler-backend PUBLIC fuse::search)
# PRIVATE: fuse::linalg backs backend/mls_weights.cpp's local polynomial
# solve (the moving-least-squares moment-matrix system) the same way
# fuse-coupler-num links it -- no public mls_weights.hpp type names
# fuse::linalg, so a consumer of fuse::backend never needs to see it.
target_link_libraries(fuse-coupler-backend PRIVATE fuse::linalg)
# PRIVATE: r3d is the supermesh kernel, used inside backend/supermesh.cpp only.
# No public header mentions it, so a consumer never needs it on an include path
# and never links it -- which is what keeps the geometric kernel swappable.
# Same reasoning as fuse-coupler-model's yaml-cpp link above: r3d has no
# installed-package path at all today (cmake/FuseDependencies.cmake fetches
# it unconditionally), so it cannot be exported alongside fuse-coupler-backend.
target_link_libraries(fuse-coupler-backend PRIVATE $<BUILD_LOCAL_INTERFACE:r3d::r3d>)
# native_migration.cpp is this layer's own single MPI translation unit (the
# exchange step only -- docs/dev/layering.md), the same isolation
# fuse::partition's collectives.cpp uses one layer up.
if(FUSE_ENABLE_MPI)
  target_link_libraries(fuse-coupler-backend PRIVATE MPI::MPI_CXX)
endif()
if(FUSE_ENABLE_MOAB)
  # Unlike intx_weights.cpp, which carries its own no-MOAB path, this file IS
  # the MOAB backend: it includes moab/ headers unconditionally, so it can only
  # be compiled when they exist.
  target_sources(fuse-coupler-backend PRIVATE backend/moab_backend.cpp)

  # PUBLIC, not PRIVATE: intx_weights.cpp is compiled into this library, but
  # the mesh-backend headers a consumer includes also reach MOAB types, so a
  # consumer that cannot see MOAB's headers cannot use this library at all.
  target_link_libraries(fuse-coupler-backend PUBLIC fuse::moab)
endif()
set_target_properties(fuse-coupler-backend PROPERTIES
  EXPORT_NAME backend
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- partition
# fuse::partition: the Kokkos-view RCB mesh partitioner, ported from the
# user's rcb_moab. The core (rcb.hpp/median.hpp/coord_transform.hpp) is
# header-only and templated on an execution space; the one translation unit
# here (collectives.cpp) is where every MPI symbol in this layer lives, so
# the public headers stay free of mpi.h (docs/dev/layering.md). The other
# (mesh_adapter.cpp) is the only file in this layer that needs fuse::backend,
# for the MeshView it adapts into (centroids, global_ids).
add_library(fuse-coupler-partition
  partition/collectives.cpp
  partition/mesh_adapter.cpp
)
add_library(fuse::partition ALIAS fuse-coupler-partition)
target_compile_features(fuse-coupler-partition PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-partition PUBLIC fuse::base fuse::backend)
if(FUSE_ENABLE_MPI)
  target_link_libraries(fuse-coupler-partition PRIVATE MPI::MPI_CXX)
endif()
set_target_properties(fuse-coupler-partition PROPERTIES
  EXPORT_NAME partition
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
  runtime/window_coupler.cpp
)
add_library(fuse::runtime ALIAS fuse-coupler-runtime)
target_compile_features(fuse-coupler-runtime PUBLIC cxx_std_20)
target_link_libraries(fuse-coupler-runtime PUBLIC
  fuse::schedule fuse::num fuse::transport fuse::backend)
set_target_properties(fuse-coupler-runtime PROPERTIES
  EXPORT_NAME runtime
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- abi
# The participant C ABI (docs/dev/abi-design.md, docs/user/c-abi.md):
# fuse_coupler_abi() and the fuse_participant_abi_v1 table behind it. Links
# fuse::runtime (Runner, ComponentSet) and nothing above it -- the ABI is a
# thin, exception-containing C wrapper over the same public C++ surface
# examples/ uses, not a new layer of the coupler itself.
#
# Built TWICE from one compilation (an OBJECT library compiled once, linked
# into both variants) because the task that specified this library asked for
# both a shared and a static consumer: `fuse::abi` (shared) is what
# `dlopen()` and the install consumer use; `fuse::abi-static` (the archive)
# is for a caller that links everything into one binary and wants no extra
# .so to ship. -fvisibility=hidden is set on the OBJECT library, so it
# applies identically to both: the static archive's object files carry
# hidden symbols the same way the shared library's dynamic symbol table
# does, which is what keeps `fuse_coupler_abi` the one name either variant
# exposes (abi-design.md point 1).
# abi/provider_adapter.cpp and abi/registry.cpp (S6 phase 4, "Adapter host"
# track): the provider direction's own C++ infrastructure -- wrapping a
# fuse_provider_abi_v1 table as a fuse::Component, and mounting one from a
# native Component, an in-memory table, or a dlopen'd shared object.
# Neither exports a NEW C symbol (both are internal to this OBJECT library
# and hidden the same way participant.cpp's own helpers are; the one C
# symbol a PLUGIN exports, fuse_provider_abi, is never defined here -- it is
# the plugin's own), so they cost this library nothing against "one symbol"
# (abi-design.md point 1) while giving test/abi/conformance_test.cpp a
# single mechanism for all four kinds it drives.
add_library(fuse-abi-objects OBJECT
  abi/participant.cpp
  abi/provider_adapter.cpp
  abi/registry.cpp
)
target_compile_features(fuse-abi-objects PUBLIC cxx_std_20)
target_link_libraries(fuse-abi-objects PUBLIC fuse::runtime)
target_compile_definitions(fuse-abi-objects PRIVATE
  FUSE_PROJECT_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  FUSE_PROJECT_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  FUSE_PROJECT_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)
set_target_properties(fuse-abi-objects PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  POSITION_INDEPENDENT_CODE ON
)

#
# fuse-abi-objects is linked PRIVATE and only via $<BUILD_LOCAL_INTERFACE:...>:
# its object files are baked into each variant at link time, but it is a
# build-time device with no install-time existence of its own (same reason
# fuse-coupler-model's yaml-cpp link and fuse-coupler-search's ArborX link
# above use that generator expression) -- an exported target may not
# PUBLIC-link a target that is not itself part of the same export set, and
# fuse-abi-objects, being an OBJECT library with no public header of its
# own, has nothing an installed consumer needs from it directly.
add_library(fuse-abi-shared SHARED)
target_link_libraries(fuse-abi-shared
  PRIVATE $<BUILD_LOCAL_INTERFACE:fuse-abi-objects> ${CMAKE_DL_LIBS}
  PUBLIC  fuse::runtime
)
add_library(fuse::abi ALIAS fuse-abi-shared)
set_target_properties(fuse-abi-shared PROPERTIES
  EXPORT_NAME abi
  OUTPUT_NAME fuse-abi
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  # fuse::model (pulled in PRIVATE, via fuse-abi-objects, for YAML load/save)
  # dynamically depends on libyaml-cpp at RUNTIME whenever the yaml-cpp this
  # build resolved is itself a shared library -- true of a found system
  # package, never true of r3d (always built from source as a static
  # archive, the other PRIVATE $<BUILD_LOCAL_INTERFACE:...> dependency this
  # library pulls in the same way). Every OTHER target in this tree is
  # static, so this is the first one a dynamic loader ever has to resolve at
  # all; an installed consumer has no other way to find it, since yaml-cpp
  # is deliberately not part of the fuse-coupler install package (this
  # file's own comment on fuse-coupler-model's identical link line explains
  # why). Pointing INSTALL_RPATH at wherever THIS configure resolved it is
  # the same answer a consumer's own CMAKE_PREFIX_PATH would need to give
  # find_package(fuse-coupler) in the first place.
  INSTALL_RPATH "$<TARGET_FILE_DIR:yaml-cpp::yaml-cpp>"
)
# -fvisibility=hidden alone governs only the object files THIS library
# compiles; every statically-linked fuse:: layer below it carries its own
# dependencies' default-visibility symbols (Kokkos, fmt, yaml-cpp, ...),
# which ld would otherwise re-export wholesale. An explicit allowlist at the
# link step is what makes the one-symbol claim actually true of the built
# .so/.dylib (see src/abi/exported_symbols.{map,sym}'s own header comments;
# ported from upstream/karma/src/CMakeLists.txt's identical two-line branch
# for libkarma_abi).
if(APPLE)
  target_link_options(fuse-abi-shared PRIVATE
    "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/abi/exported_symbols.sym")
  set_property(TARGET fuse-abi-shared APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/abi/exported_symbols.sym")
else()
  target_link_options(fuse-abi-shared PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/abi/exported_symbols.map")
  set_property(TARGET fuse-abi-shared APPEND PROPERTY LINK_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/abi/exported_symbols.map")
endif()

add_library(fuse-abi-static STATIC)
target_link_libraries(fuse-abi-static
  PRIVATE $<BUILD_LOCAL_INTERFACE:fuse-abi-objects> ${CMAKE_DL_LIBS}
  PUBLIC  fuse::runtime
)
add_library(fuse::abi-static ALIAS fuse-abi-static)
set_target_properties(fuse-abi-static PROPERTIES
  EXPORT_NAME abi-static
  OUTPUT_NAME fuse-abi
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# ---------------------------------------------------------------- install
# find_package()-able install of the CMake package docs/dev/conventions.md
# already names in prose ("CMake package `fuse-coupler`"): every target above
# already carries the BUILD_INTERFACE/INSTALL_INTERFACE split on its include
# directories, so the missing half was only the install() calls themselves
# and one Config.cmake.in (REVIEW.md R-25/BT-F9).
include(GNUInstallDirs)

set(FUSE_INSTALL_TARGETS
  fuse-coupler-base fuse-coupler-model fuse-coupler-ir fuse-coupler-schedule
  fuse-coupler-linalg fuse-coupler-num fuse-coupler-transport fuse-coupler-search
  fuse-coupler-backend fuse-coupler-partition fuse-coupler-runtime
)

install(TARGETS ${FUSE_INSTALL_TARGETS}
  EXPORT fuse-coupler-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# fuse-abi-shared/fuse-abi-static (this file's own "---- abi" section,
# above): a SEPARATE install(TARGETS) call into the SAME export set, which
# CMake allows as long as every contributing call runs before the single
# install(EXPORT fuse-coupler-targets ...) below -- so ABI-C-CONSUMER's
# find_package(fuse-coupler) + `target_link_libraries(... fuse::abi)`
# (test/abi/install_consumer/CMakeLists.txt) resolves the same way every
# other fuse:: alias here does.
install(TARGETS fuse-abi-shared fuse-abi-static
  EXPORT fuse-coupler-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/fuse
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)
# The one generated header, from the BUILD tree (configure_file'd above), not
# the source tree -- a consumer needs the config it was actually built with.
install(FILES ${PROJECT_BINARY_DIR}/include/fuse/base/config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/fuse/base
)

install(EXPORT fuse-coupler-targets
  FILE fuse-coupler-targets.cmake
  NAMESPACE fuse::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fuse-coupler
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/fuse-couplerConfig.cmake.in
  ${PROJECT_BINARY_DIR}/fuse-couplerConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fuse-coupler
)
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/fuse-couplerConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${PROJECT_BINARY_DIR}/fuse-couplerConfig.cmake
  ${PROJECT_BINARY_DIR}/fuse-couplerConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fuse-coupler
)
