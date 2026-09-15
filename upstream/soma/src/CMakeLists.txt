include(FetchContent)
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        0.8.0
  FIND_PACKAGE_ARGS 0.8.0 NAMES yaml-cpp)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

add_library(soma_core
  capi/soma_c.cpp
  graph/yaml_io.cpp
  ir/lowering.cpp
  ir/rules_channel.cpp
  ir/rules_derived.cpp
  ir/regions.cpp
  ir/checkpoint.cpp
  mesh/serial_backend.cpp
  num/accel_quasi_newton.cpp
  num/accel_relaxation.cpp
  num/kernels_inst.cpp
  schedule/buffers.cpp
  schedule/acyclicity.cpp
  remap/intx_weights.cpp
  # Compiled unconditionally, TempestRemap or not: without it the file still
  # defines every function high_order_backend.hpp declares, and they return
  # E-REMAP-NOTEMPEST. Task 5.2 hit the other arrangement -- an always-visible
  # declaration whose definition was behind an #ifdef -- and it fails at link
  # time in the configuration nobody built.
  remap/tempest_weights.cpp
  remap/spacetime.cpp
  runtime/iteration.cpp
  runtime/runner.cpp
  schedule/expand.cpp
  schedule/scheduler.cpp
  schedule/slab.cpp
  transport/copy_transport.cpp
  # Compiled in EVERY configuration, not guarded like mpi_transport.cpp below.
  # LowFiveTransport is declared unconditionally in the public header, so all
  # of its members must link everywhere; without LowFive the body returns
  # E-TRANSPORT-NOLOWFIVE.
  transport/lowfive_transport.cpp)

target_link_libraries(soma_core
  PUBLIC  soma
  PRIVATE yaml-cpp::yaml-cpp)   # PRIVATE: never leaks into consumers

target_compile_options(soma_core PRIVATE -Wall -Wextra -Wpedantic)

# PRIVATE, and deliberately so: it makes `#include "transport/crystal_router.
# hpp"` resolve inside the library without putting src/ -- or the mpi.h that
# header pulls in -- on any consumer's include path. The parallel test that
# exercises the router directly adds this same directory itself.
target_include_directories(soma_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# soma_version() reports this. PRIVATE: it is an implementation detail of
# src/capi/soma_c.cpp, not something consumers should key off.
target_compile_definitions(soma_core PRIVATE
  SOMA_VERSION_STRING="${PROJECT_VERSION}")

# The C ABI reports a communicator as unsupported rather than silently
# ignoring it on a build with no MPI transport, so it needs to know.
#
# PUBLIC, not PRIVATE: the MOAB backend's parallel test has to know whether
# MoabMeshBackend::set_communicator can succeed, and a test that guesses would
# either skip silently on a build that supports MPI or fail on one that does
# not. It is a capability macro, not a type, so it does not put mpi.h on a
# consumer's include path.
if(SOMA_HAVE_MPI)
  target_compile_definitions(soma_core PUBLIC SOMA_HAVE_MPI=1)
  # The MPI transport, spec 4.1's n->m redistribution. Like the MOAB backend
  # it is the only translation unit that names an MPI type; the communicator
  # crosses the public boundary as MPI_Comm_c2f(comm).
  target_sources(soma_core PRIVATE transport/mpi_transport.cpp)
  # R5's missing upstream half: the rendezvous that decides which rank holds
  # or wants each entity, and the portable Crystal Router underneath it.
  # crystal_router.hpp lives in src/ rather than include/ because it names
  # MPI_Comm in its interface and has exactly one caller inside the library;
  # see the note at the top of that file.
  target_sources(soma_core PRIVATE transport/comm_graph.cpp
                                   transport/crystal_router.cpp)
  # PRIVATE: mesh/moab_backend.cpp calls MPI_Comm_f2c, but no SOMA header
  # names an MPI type, so mpi.h stays off a consumer's include path. On a
  # static library CMake still propagates the LINK requirement.
  target_link_libraries(soma_core PRIVATE MPI::MPI_CXX)
endif()

# The LowFive HDF5-VOL transport, spec 5 mechanism 4. Unlike MOAB and MPI
# there is no target_sources() here: transport/lowfive_transport.cpp is in the
# unconditional source list above and only its body is #ifdef'd, because
# LowFiveTransport is declared unconditionally in the public header.
#
# soma::lowfive is PRIVATE -- src/transport/lowfive_transport.cpp is the only
# translation unit in the project that may include <hdf5.h> or any
# lowfive/*.hpp, and the public header names no HDF5 type (the FAPL crosses
# the boundary as a `long long`, the same way MpiTransport takes a Fortran
# communicator handle).
#
# PUBLIC on the definition, though: test/unit/lowfive_transport_test.cpp keys
# its #ifdef off it, and a test that guessed would either skip silently on a
# build that has LowFive or fail on one that does not.
if(SOMA_HAVE_LOWFIVE)
  target_compile_definitions(soma_core PUBLIC SOMA_HAVE_LOWFIVE=1)
  target_link_libraries(soma_core PRIVATE soma::lowfive)
endif()

# The MOAB backend. src/mesh/moab_backend.cpp is the ONLY translation unit in
# the project that may include moab/Core.hpp (Global Constraints dependency
# rule), so soma::moab is linked PRIVATE: nothing downstream sees MOAB.
if(SOMA_ENABLE_MOAB)
  target_sources(soma_core PRIVATE mesh/moab_backend.cpp)
  target_link_libraries(soma_core PRIVATE soma::moab)
  # PUBLIC because test/unit/moab_mesh_test.cpp keys its #ifdef off it. A
  # capability macro is not a MOAB type, so this does not violate the
  # dependency rule.
  target_compile_definitions(soma_core PUBLIC SOMA_HAVE_MOAB)
  if(SOMA_HAVE_TEMPEST)
    target_compile_definitions(soma_core PUBLIC SOMA_HAVE_TEMPEST)
  endif()
endif()
