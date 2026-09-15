include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
  FIND_PACKAGE_ARGS 1.17.0 NAMES GTest)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(soma_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE soma soma_core GTest::gtest_main)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_definitions(${name} PRIVATE
    SOMA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    SOMA_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
  # DISCOVERY_TIMEOUT defaults to 5 seconds. Discovery runs the freshly linked
  # binary with --gtest_list_tests at BUILD time, so on a loaded machine (a
  # parallel build, several configurations at once, a busy CI runner) the
  # process is descheduled long enough to blow the default and the *build*
  # fails -- with a message about a test executable, which reads like a broken
  # test rather than a starved one. Observed here: 30 such failures across
  # three legs with zero compiler diagnostics. 60 seconds is still far below
  # any real hang, so a genuinely wedged binary is still caught.
  gtest_discover_tests(${name} DISCOVERY_TIMEOUT 60)
endfunction()

soma_add_test(smoke_test unit/smoke_test.cpp)
soma_add_test(result_test unit/result_test.cpp)
soma_add_test(rational_test unit/rational_test.cpp)
soma_add_test(units_test unit/units_test.cpp)
soma_add_test(time_level_test unit/time_level_test.cpp)
soma_add_test(ids_test unit/ids_test.cpp)
soma_add_test(region_test unit/region_test.cpp)
soma_add_test(component_test unit/component_test.cpp)
soma_add_test(channel_test unit/channel_test.cpp)
soma_add_test(coupling_graph_test unit/coupling_graph_test.cpp)
soma_add_test(validate_test unit/validate_test.cpp)
soma_add_test(descriptor_test unit/descriptor_test.cpp)
soma_add_test(yaml_io_test unit/yaml_io_test.cpp)
soma_add_test(ir_op_test unit/ir_op_test.cpp)
soma_add_test(ir_dag_test unit/ir_dag_test.cpp)
soma_add_test(ir_dump_test unit/ir_dump_test.cpp)
soma_add_test(lowering_test unit/lowering_test.cpp)
soma_add_test(lowering_channel_test unit/lowering_channel_test.cpp)
soma_add_test(lowering_derived_test unit/lowering_derived_test.cpp)
soma_add_test(lowering_regions_test unit/lowering_regions_test.cpp)
soma_add_test(checkpoint_test unit/checkpoint_test.cpp)
soma_add_test(field_ref_test unit/field_ref_test.cpp)
soma_add_test(clock_test unit/clock_test.cpp)
soma_add_test(slab_test unit/slab_test.cpp)
soma_add_test(expand_test unit/expand_test.cpp)
soma_add_test(buffers_test unit/buffers_test.cpp)
soma_add_test(acyclicity_test unit/acyclicity_test.cpp)
soma_add_test(plan_test unit/plan_test.cpp)
soma_add_test(lockstep_test unit/lockstep_test.cpp)
soma_add_test(transport_test unit/transport_test.cpp)

# Compiled in EVERY configuration, like the transport itself: with LowFive off
# it checks that every entry point still links and reports
# E-TRANSPORT-NOLOWFIVE, which is the failure Task 5.2 shipped. With LowFive
# on it needs <hdf5.h> of its own -- the "unmodified HDF5 component" and the
# native-VOL control that makes "no file on disk" mean anything are both raw
# HDF5 by construction. soma::lowfive carries the HDF5 include directories
# (and, because this HDF5 is parallel and <H5public.h> includes <mpi.h>
# unconditionally, MPI's too).
soma_add_test(lowfive_transport_test unit/lowfive_transport_test.cpp)
if(SOMA_HAVE_LOWFIVE)
  target_link_libraries(lowfive_transport_test PRIVATE soma::lowfive)
endif()
soma_add_test(kernels_test unit/kernels_test.cpp)
soma_add_test(convergence_test unit/convergence_test.cpp)
soma_add_test(accelerator_test unit/accelerator_test.cpp)
soma_add_test(temporal_basis_test unit/temporal_basis_test.cpp)
soma_add_test(waveform_test unit/waveform_test.cpp)
soma_add_test(waveform_order_test unit/waveform_order_test.cpp)
soma_add_test(component_iface_test unit/component_iface_test.cpp)
soma_add_test(serial_mesh_test unit/serial_mesh_test.cpp)
soma_add_test(runner_test unit/runner_test.cpp)
soma_add_test(moab_mesh_test unit/moab_mesh_test.cpp)
soma_add_test(remap_test unit/remap_test.cpp)
# Task 5.3's order-of-accuracy study. Separate from remap_test because it builds
# twenty weight matrices over a refinement sequence and is measured in seconds,
# not milliseconds -- and because an order claim deserves its own failing test
# rather than one assertion buried among forty.
soma_add_test(remap_order_test unit/remap_order_test.cpp)
soma_add_test(remap_runtime_test unit/remap_runtime_test.cpp)
soma_add_test(spacetime_test unit/spacetime_test.cpp)
soma_add_test(spacetime_order_test unit/spacetime_order_test.cpp)
soma_add_test(spacetime_runtime_test unit/spacetime_runtime_test.cpp)
soma_add_test(spacetime_conservation_test unit/spacetime_conservation_test.cpp)
soma_add_test(golden_ir_test golden_ir_test.cpp)

# Spec 10 level 4: the manufactured-solution study.
soma_add_test(mms_transfer_test verification/mms_transfer_test.cpp)

# A real MOAB mesh, not a generated box. MOAB installs a 3456-element cubed
# sphere under share/examples; it is unstructured, its quadrilaterals are
# non-planar, and it carries eight PARALLEL_PARTITION sets, so it exercises
# everything a generated box cannot. The path is discovered from the MOAB
# prefix pkg-config reported rather than copied into the repository, and the
# tests that need it skip when it is absent.
if(SOMA_ENABLE_MOAB AND MOAB_INCLUDE_DIRS)
  list(GET MOAB_INCLUDE_DIRS 0 _soma_moab_inc)
  get_filename_component(_soma_moab_prefix "${_soma_moab_inc}" DIRECTORY)
  find_file(SOMA_MOAB_TEST_MESH_FILE
            NAMES spherecube.h5m
            PATHS "${_soma_moab_prefix}/share/examples/fortran"
            NO_DEFAULT_PATH)
  if(SOMA_MOAB_TEST_MESH_FILE)
    target_compile_definitions(moab_mesh_test PRIVATE
      SOMA_MOAB_TEST_MESH="${SOMA_MOAB_TEST_MESH_FILE}")
    # remap_test needs it too: the cubed sphere is the only real, non-planar,
    # non-matching pair the remap weights can be checked against.
    target_compile_definitions(remap_test PRIVATE
      SOMA_MOAB_TEST_MESH="${SOMA_MOAB_TEST_MESH_FILE}")
    # Task 6.4 composes T with the weights build_weights actually produces on
    # that pair, so the space-time conservation study can state the real
    # cubed-sphere numbers rather than an emulation of them.
    target_compile_definitions(spacetime_conservation_test PRIVATE
      SOMA_MOAB_TEST_MESH="${SOMA_MOAB_TEST_MESH_FILE}")
    message(STATUS "MOAB: real-mesh tests will use ${SOMA_MOAB_TEST_MESH_FILE}")
  else()
    message(STATUS
      "MOAB: spherecube.h5m not found under ${_soma_moab_prefix}/share; the "
      "real-mesh tests are skipped and only generated meshes are covered")
  endif()
endif()

# The distributed test. Deliberately not a soma_add_test(): it calls MPI_Init,
# it is launched by mpiexec, and a rank that fails must abort the job rather
# than let the others block in a collective, none of which GoogleTest does.
# It is the only test that can show `entity_count` is a LOCAL count -- every
# ownership assertion is vacuous in serial.
if(SOMA_ENABLE_MOAB AND SOMA_HAVE_MPI AND SOMA_MOAB_TEST_MESH_FILE)
  add_executable(moab_parallel_test unit/moab_parallel_test.cpp)
  target_link_libraries(moab_parallel_test PRIVATE soma soma_core MPI::MPI_CXX)
  target_compile_options(moab_parallel_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME moab_parallel_test_np2
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:moab_parallel_test>
                   ${MPIEXEC_POSTFLAGS} "${SOMA_MOAB_TEST_MESH_FILE}")
  # A one-rank run of the same binary: it must agree with the serial backend,
  # which is what proves the partitioned path degenerates correctly.
  add_test(NAME moab_parallel_test_np1
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:moab_parallel_test>
                   ${MPIEXEC_POSTFLAGS} "${SOMA_MOAB_TEST_MESH_FILE}")
  set_tests_properties(moab_parallel_test_np1 moab_parallel_test_np2
                       PROPERTIES TIMEOUT 300)

  # R3's guard, exercised. Same reasons for not being a soma_add_test(); see
  # the file's own header for what each rank count establishes. Three rank
  # counts on purpose: 1 is the POSITIVE control (a real communicator of size
  # one must still build weights, against the constants the serial test
  # asserts for the same pair), 2 and 4 are the refusal. Without the np1 leg a
  # guard that refused on
  # "a communicator was set" rather than on "the communicator has more than
  # one rank" would pass, and every MPI job would silently lose remap.
  add_executable(remap_parallel_test unit/remap_parallel_test.cpp)
  target_link_libraries(remap_parallel_test PRIVATE soma soma_core
                        MPI::MPI_CXX)
  target_compile_options(remap_parallel_test PRIVATE -Wall -Wextra -Wpedantic)
  foreach(_np 1 2 4)
    add_test(NAME remap_parallel_test_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:remap_parallel_test>
                     ${MPIEXEC_POSTFLAGS} "${SOMA_MOAB_TEST_MESH_FILE}")
    # Measured at 1.0 s for the np1 leg (a 3456 x 3456 supermesh) and 0.1 s for
    # the others. A generous ceiling so a loaded machine does not produce a
    # spurious failure, but low enough that a deadlocked collective -- the
    # failure mode this whole area is about -- is reported rather than waited
    # on forever.
    set_tests_properties(remap_parallel_test_np${_np} PROPERTIES TIMEOUT 600)
  endforeach()

  # ---- R3's other half: a halo, and conservation across a partition ------
  #
  # Needs a SECOND partitioned mesh, partitioned DIFFERENTLY from
  # spherecube.h5m. Nothing ships one, and a pair of identically-partitioned
  # meshes cannot test a halo (a partition-local intersection is accidentally
  # correct on one). The fixture below writes it. The exact parameters are
  # load bearing -- docs/HANDOFF.md quotes measurements taken with them -- so
  # they live here, in one place, rather than being passed around.
  #
  # make_partitioned_sphere is a TEST TOOL that includes MOAB directly; see
  # its header for why that does not breach the two-translation-unit rule.
  add_executable(make_partitioned_sphere tools/make_partitioned_sphere.cpp)
  target_link_libraries(make_partitioned_sphere PRIVATE soma::moab)
  target_compile_options(make_partitioned_sphere PRIVATE -Wall -Wextra)
  set(SOMA_PARTITIONED_PARTNER
      "${CMAKE_CURRENT_BINARY_DIR}/cs16_block4.h5m")
  add_test(NAME make_partitioned_partner
           COMMAND make_partitioned_sphere 16 7996.4553620 4
                   "${SOMA_PARTITIONED_PARTNER}" block)
  set_tests_properties(make_partitioned_partner PROPERTIES
                       FIXTURES_SETUP soma_partitioned_partner TIMEOUT 300)

  add_executable(remap_halo_parallel_test unit/remap_halo_parallel_test.cpp)
  target_link_libraries(remap_halo_parallel_test PRIVATE soma soma_core
                        MPI::MPI_CXX)
  target_compile_options(remap_halo_parallel_test PRIVATE
                         -Wall -Wextra -Wpedantic)
  foreach(_np 1 2 4)
    add_test(NAME remap_halo_parallel_test_np${_np}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
                     ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:remap_halo_parallel_test>
                     ${MPIEXEC_POSTFLAGS} "${SOMA_MOAB_TEST_MESH_FILE}"
                     "${SOMA_PARTITIONED_PARTNER}")
    set_tests_properties(remap_halo_parallel_test_np${_np} PROPERTIES
                         FIXTURES_REQUIRED soma_partitioned_partner
                         TIMEOUT 600)
  endforeach()
endif()

# The MPI transport's distributed test. Like moab_parallel_test it is not a
# soma_add_test(): it calls MPI_Init, it is launched by mpiexec, and it links
# GTest::gtest rather than GTest::gtest_main because it supplies its own main.
# It is also not gtest_discover_tests()'d -- that would run the binary once per
# TEST on a single rank, and a single-rank "MPI" test exchanges nothing.
#
# Two rank counts on purpose. -n 4 is where the asymmetric 3->1 and 1->3
# patterns, the idle-rank case and the zero-element-rank case live; -n 2 shows
# the same code path is not tuned to four ranks.
if(SOMA_HAVE_MPI)
  add_executable(mpi_transport_test parallel/mpi_transport_test.cpp)
  target_link_libraries(mpi_transport_test PRIVATE soma soma_core
                        GTest::gtest MPI::MPI_CXX)
  target_compile_options(mpi_transport_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME mpi_transport_4rank
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpi_transport_test>
                   ${MPIEXEC_POSTFLAGS})
  add_test(NAME mpi_transport_2rank
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpi_transport_test>
                   ${MPIEXEC_POSTFLAGS})
  # A deadlock must show up as a failed test, not as a wedged CI job.
  set_tests_properties(mpi_transport_4rank mpi_transport_2rank
                       PROPERTIES TIMEOUT 120)

  # CommGraph and the Crystal Router (R5). Same three reasons as above for not
  # being a soma_add_test().
  #
  # THREE rank counts, and the odd one is not decoration. The router bisects
  # its rank range each round and pairs `lo+i` with `mid+1+i`; when a range has
  # an ODD number of ranks exactly one rank is left without a partner and sends
  # to `mid+1` instead, which is a code path -n 2 and -n 4 never reach at all.
  # -n 3 also makes the two components unequal in size in every shape.
  add_executable(comm_graph_test parallel/comm_graph_test.cpp)
  target_link_libraries(comm_graph_test PRIVATE soma soma_core
                        GTest::gtest MPI::MPI_CXX)
  # src/ so the test can reach the private crystal_router.hpp directly. It
  # already includes mpi.h itself, which is the only reason that header is not
  # in include/.
  target_include_directories(comm_graph_test PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_compile_options(comm_graph_test PRIVATE -Wall -Wextra -Wpedantic)
  foreach(_np 2 3 4)
    add_test(NAME comm_graph_${_np}rank
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:comm_graph_test>
                     ${MPIEXEC_POSTFLAGS})
    set_tests_properties(comm_graph_${_np}rank PROPERTIES TIMEOUT 180)
  endforeach()

  # Runner-level transport wiring. Same three reasons as comm_graph_test above
  # for not being a soma_add_test(). Task 6 widens the rank counts.
  add_executable(runner_transport_test parallel/runner_transport_test.cpp)
  target_link_libraries(runner_transport_test PRIVATE soma soma_core
                        GTest::gtest MPI::MPI_CXX)
  target_compile_options(runner_transport_test PRIVATE -Wall -Wextra -Wpedantic)
  # Four rank counts. 2 puts one rank in each component; 3 makes the two
  # components unequal in size AND leaves the crystal router with an odd range;
  # 4 is the even, balanced case; 5 is odd and unbalanced at once.
  # comm_graph_test documents why the odd counts are not decoration.
  foreach(_np 2 3 4 5)
    add_test(NAME runner_transport_${_np}rank
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_np}
                     ${MPIEXEC_PREFLAGS} $<TARGET_FILE:runner_transport_test>
                     ${MPIEXEC_POSTFLAGS})
    set_tests_properties(runner_transport_${_np}rank PROPERTIES TIMEOUT 180)
  endforeach()
endif()

# The C ABI's distributed test. Same three reasons as mpi_transport_test above
# -- MPI_Init, mpiexec, MPI_Abort on failure -- plus capi_test's own reason for
# not being a soma_add_test(): it is a C translation unit and must be compiled
# by a C compiler.
#
# Two rank counts, and the -n 1 entry is load-bearing rather than symmetric. It
# is the positive control for soma_initialize()'s parallel-safety gate, which
# refuses a rank-local convergence early exit above one rank: without an -n 1
# run, a gate that refused unconditionally on any MPI build would look correct.
if(SOMA_HAVE_MPI)
  add_executable(capi_parallel_test parallel/capi_parallel_test.c)
  set_target_properties(capi_parallel_test PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX)
  # MPI::MPI_CXX, not MPI::MPI_C: the top-level find_package(MPI) requests only
  # the CXX component, and asking for another here would make the whole build
  # depend on a component nothing else needs. What this target actually wants
  # from it -- the MPI include directory and the MPI libraries -- is not
  # language-specific, and the link is done by the C++ driver anyway (see
  # LINKER_LANGUAGE above, needed because soma_core is C++).
  target_link_libraries(capi_parallel_test PRIVATE soma_core MPI::MPI_CXX)
  target_compile_options(capi_parallel_test PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME capi_parallel_2rank
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:capi_parallel_test>
                   ${MPIEXEC_POSTFLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/data")
  add_test(NAME capi_parallel_1rank
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
                   ${MPIEXEC_PREFLAGS} $<TARGET_FILE:capi_parallel_test>
                   ${MPIEXEC_POSTFLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/data")
  set_tests_properties(capi_parallel_1rank capi_parallel_2rank
                       PROPERTIES TIMEOUT 120)
endif()

# The C ABI test is deliberately not a soma_add_test(): it is a C translation
# unit, so it cannot link GoogleTest and must not be compiled as C++.
# Compiling it with a C compiler is the only thing that actually proves
# include/soma/soma_c.h is C-compatible -- as C++ it would compile even if it
# were full of references, default arguments, and `bool`.
add_executable(capi_test unit/capi_test.c)
set_target_properties(capi_test PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  # soma_core is C++ and pulls in libstdc++/libc++; link with the C++ driver.
  LINKER_LANGUAGE CXX)
target_link_libraries(capi_test PRIVATE soma_core)
target_compile_options(capi_test PRIVATE -Wall -Wextra -Wpedantic)
target_compile_definitions(capi_test PRIVATE
  SOMA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
add_test(NAME capi_test COMMAND capi_test)

# The Fortran shim's test. Like capi_test above it is not a soma_add_test():
# it is a Fortran translation unit, it cannot link GoogleTest, and compiling
# it with an actual Fortran compiler is the only thing that proves
# fortran/soma_mod.f90 is a usable module rather than a plausible-looking
# file. Guarded on SOMA_ENABLE_FORTRAN, which also guards the
# enable_language(Fortran) that makes add_executable() below legal.
if(SOMA_ENABLE_FORTRAN)
  add_executable(fortran_capi_test fortran/capi_test.f90)
  target_link_libraries(fortran_capi_test PRIVATE soma_fortran)
  # The data directory arrives as argv(1), not as a -D preprocessor macro.
  # Three reasons, all of which bit: the flag that turns the preprocessor on
  # is compiler-specific (-cpp for gfortran and flang, -fpp for ifx/ifort,
  # -fpp for NAG), so hard-coding one makes the test GNU-only; a macro that
  # expands to a quoted string has to survive CMake's, the shell's, and the
  # preprocessor's quoting rules; and Fortran's `//` concatenation operator is
  # a comment introducer to a non-traditional cpp, which silently truncates
  # the line that builds the path. get_command_argument has none of those
  # problems and is plain Fortran 2003.
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    # -Wno-compare-reals is deliberate. The window bounds this test checks
    # (0, 1/2, 1) are exactly representable, and SOMA derives them from exact
    # rational arithmetic, so an approximate comparison here would accept a
    # schedule that had started to drift -- which is the one thing spec 4.3
    # says must never happen. test/unit/capi_test.c compares them the same
    # way.
    target_compile_options(fortran_capi_test PRIVATE
      -Wall -Wextra -Wno-compare-reals -std=f2018 -pedantic)
  endif()
  add_test(NAME fortran_capi_test
           COMMAND fortran_capi_test "${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()
soma_add_test(iteration_test verification/iteration_test.cpp)
