# test/partition/CMakeLists.txt -- fuse::partition's own test registration.
#
# NOT fuse_add_gtest/fuse_add_mpi_test (test/CMakeLists.txt's own functions):
# those resolve "${layer}/${name}_test.cpp" relative to the CALLER's
# CMAKE_CURRENT_SOURCE_DIR, which is test/partition/ once add_subdirectory
# has descended into it -- calling fuse_add_gtest(partition foo) from here
# would look for test/partition/partition/foo_test.cpp. The thin wrappers
# below do the same registration (executable, links, fuse_record_test_source,
# add_test, labels) with paths resolved correctly for this directory, and
# record sources under "partition/<name>_test.cpp" -- the path
# test/CMakeLists.txt's audit (its glob is RELATIVE to the top-level test/,
# unaffected by add_subdirectory) expects to see.

function(fuse_partition_add_gtest name)
  cmake_parse_arguments(ARG "THROWS" "" "LIBS" ${ARGN})
  set(target fuse_partition_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  fuse_record_test_source(partition/${name}_test.cpp)
  target_link_libraries(${target} PRIVATE fuse::partition fuse_test_main ${ARG_LIBS})
  add_test(NAME partition.${name} COMMAND ${target})
  set(labels "partition;gtest")
  # test/CMakeLists.txt's OpenMP thread-count leg: the partitioner is the one
  # library area whose accumulation order is the threads' (median.hpp). The omp
  # legs run the same executable as the base test, so THROWS must propagate to
  # them too, or `ctest -LE throws` (tier0-no-exceptions) will not exclude them.
  if(ARG_THROWS)
    fuse_add_omp_legs(partition.${name} ${target} THROWS)
    list(APPEND labels throws)
  else()
    fuse_add_omp_legs(partition.${name} ${target})
  endif()
  set_tests_properties(partition.${name} PROPERTIES LABELS "${labels}")
endfunction()

# fuse_partition_add_mpi_test(<name> [LIBS ...]) -- same shape as
# fuse_add_mpi_test, called only when FUSE_ENABLE_MPI is ON (see below); the
# gated-source registration for the OFF case is unconditional, exactly
# test/CMakeLists.txt's own model/descriptor_mpi_test.cpp pattern.
function(fuse_partition_add_mpi_test name)
  cmake_parse_arguments(ARG "" "" "LIBS" ${ARGN})
  set(target fuse_partition_${name}_test)
  add_executable(${target} ${name}_test.cpp)
  fuse_record_test_source(partition/${name}_test.cpp)
  target_link_libraries(${target} PRIVATE fuse::partition fuse_test_main ${ARG_LIBS})
  foreach(n 1 2 3 4 5)
    add_test(NAME partition.${name}.${n}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${n} $<TARGET_FILE:${target}>)
    # Same per-rank OMP_NUM_THREADS cap and OMPI_MCA alltoallv-algorithm
    # workaround as test/CMakeLists.txt's own fuse_add_mpi_test -- see that
    # function's identical ENVIRONMENT line for why both are needed on this
    # host's Open MPI (4.1.5).
    math(EXPR _fuse_partition_mpi_threads "${_fuse_host_cores} / ${n}")
    if(_fuse_partition_mpi_threads LESS 1)
      set(_fuse_partition_mpi_threads 1)
    endif()
    set_tests_properties(partition.${name}.${n} PROPERTIES TIMEOUT 120
                         LABELS "partition;mpi;parallel"
                         ENVIRONMENT "OMP_NUM_THREADS=${_fuse_partition_mpi_threads};OMP_PROC_BIND=false;OMPI_MCA_coll_tuned_use_dynamic_rules=1;OMPI_MCA_coll_tuned_alltoallv_algorithm=2;OMPI_MCA_rmaps_base_oversubscribe=1;PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe")
  endforeach()
endfunction()

fuse_partition_add_gtest(oracle)
fuse_partition_add_gtest(rcb_partition)
fuse_partition_add_gtest(mesh_adapter LIBS fuse::backend)
fuse_partition_add_gtest(repartition LIBS fuse::backend)
fuse_partition_add_gtest(refusals THROWS)

fuse_tests_gated_by(FUSE_ENABLE_MPI partition/rcb_mpi_test.cpp)
if(FUSE_ENABLE_MPI)
  fuse_partition_add_mpi_test(rcb_mpi LIBS fuse::backend MPI::MPI_CXX)
endif()

# repartition's num_parts <
# comm_size case needs a real communicator with more than one rank to
# reproduce (kNullComm's comm_size is always 1) -- see
# repartition_mpi_test.cpp's header comment.
fuse_tests_gated_by(FUSE_ENABLE_MPI partition/repartition_mpi_test.cpp)
if(FUSE_ENABLE_MPI)
  fuse_partition_add_mpi_test(repartition_mpi LIBS fuse::backend MPI::MPI_CXX)
endif()
