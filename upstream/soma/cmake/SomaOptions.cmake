option(SOMA_ENABLE_TESTS    "Build unit tests"                ON)
option(SOMA_ENABLE_MPI      "Enable MPI transport"            OFF)
option(SOMA_ENABLE_MOAB     "Enable MOAB mesh backend"        OFF)
option(SOMA_ENABLE_PETSC    "Enable PETSc solvers"            OFF)
option(SOMA_ENABLE_KOKKOS   "Enable Kokkos execution spaces"  OFF)
option(SOMA_ENABLE_LOWFIVE  "Enable LowFive transport"        OFF)
option(SOMA_ENABLE_PYTHON   "Build Python bindings"           OFF)
option(SOMA_ENABLE_FORTRAN  "Build Fortran shim"              OFF)

function(soma_print_summary)
  message(STATUS "SOMA configuration:")
  foreach(opt TESTS MPI MOAB PETSC KOKKOS LOWFIVE PYTHON FORTRAN)
    message(STATUS "  SOMA_ENABLE_${opt}: ${SOMA_ENABLE_${opt}}")
  endforeach()
  # SOMA_ENABLE_* is what was asked for; SOMA_HAVE_* is what was found. The two
  # genuinely differ, and both known cases are capability-not-version problems:
  # a MOAB present but built without TempestRemap, and a LowFive present but
  # unusable because it is not shared or its libhdf5 no longer resolves. Print
  # the discovered capabilities separately, and only those actually probed.
  foreach(cap MOAB TEMPEST IMOAB MPI LOWFIVE)
    if(DEFINED SOMA_HAVE_${cap})
      message(STATUS "  SOMA_HAVE_${cap}: ${SOMA_HAVE_${cap}}")
    endif()
  endforeach()
endfunction()
