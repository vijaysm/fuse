# Driver for the "abi.install_consumer_fortran" ctest entry
# (ABI-FORTRAN-CONSUMER, build-evidence/abi-matrix.toml). Modelled on
# test/install_consumer/run-smoke-test.cmake: ctest's add_test COMMAND runs
# one program; this `cmake -P` script is that program, since the check is
# itself several separate cmake invocations (install, configure a second
# project, build it, run it), not one executable.
#
# Required -D arguments: FUSE_BUILD_DIR, FUSE_INSTALL_PREFIX, CONSUMER_SRC_DIR,
# CONSUMER_BUILD_DIR, CMAKE_COMMAND, CMAKE_GENERATOR, CMAKE_PREFIX_PATH.
# Optional: REFERENCE_OUTPUT -- a file holding the C consumer's own stdout
# (test/abi/install_consumer/, the other S6-phase-2 track) to byte-diff
# against. At the time this file was written that sibling did not exist yet
# (neither did libfuse-abi, which this Fortran binding's only undefined
# symbol needs at final link -- fortran/CMakeLists.txt's own header), so
# this script runs the comparison ONLY when a REFERENCE_OUTPUT is actually
# supplied, and says so plainly rather than inventing a golden of its own.
foreach(_fuse_required FUSE_BUILD_DIR FUSE_INSTALL_PREFIX CONSUMER_SRC_DIR
                        CONSUMER_BUILD_DIR CMAKE_COMMAND CMAKE_GENERATOR)
  if(NOT DEFINED ${_fuse_required})
    message(FATAL_ERROR "run-smoke-test.cmake: -D${_fuse_required}=... is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${FUSE_INSTALL_PREFIX}" "${CONSUMER_BUILD_DIR}")

message(STATUS "abi.install_consumer_fortran: cmake --install ${FUSE_BUILD_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${FUSE_BUILD_DIR}" --prefix "${FUSE_INSTALL_PREFIX}"
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "cmake --install failed (${_fuse_rc})")
endif()

# The install prefix itself goes first on the consumer's search path: a
# dependency this build FETCHED (Kokkos, fmt, ... on a FUSE_ALLOW_FETCH=ON
# configure, every CI leg) has no prefix of its own to forward, but
# `cmake --install` above put its package config into this same prefix, so
# the consumer finds that copy -- the one fuse-coupler was built against.
# An installed dependency still resolves through the <pkg>_DIR forwarded in
# FUSE_CONSUMER_DEP_ARGS, which find_package consults before any prefix.
set(_fuse_consumer_prefix_path "${FUSE_INSTALL_PREFIX}" ${CMAKE_PREFIX_PATH})

message(STATUS "abi.install_consumer_fortran: configuring the consumer against the install prefix")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${CONSUMER_SRC_DIR}" -B "${CONSUMER_BUILD_DIR}"
          -G "${CMAKE_GENERATOR}"
          "-Dfuse-coupler_DIR=${FUSE_INSTALL_PREFIX}/lib/cmake/fuse-coupler"
          "-DCMAKE_PREFIX_PATH=${_fuse_consumer_prefix_path}"
          # <pkg>_DIR for every package this build's configure found, so the
          # consumer resolves the same ones (test/CMakeLists.txt, optional).
          ${FUSE_CONSUMER_DEP_ARGS}
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "configuring the Fortran install consumer failed (${_fuse_rc}) -- "
                      "find_package(fuse-coupler)'s fuse::abi-fortran import does not work")
endif()

message(STATUS "abi.install_consumer_fortran: building the consumer")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${CONSUMER_BUILD_DIR}"
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "building the Fortran install consumer failed (${_fuse_rc})")
endif()

message(STATUS "abi.install_consumer_fortran: running the consumer")
# WORKING_DIRECTORY matters here: consumer.f90 opens "two_component.yaml" by
# its bare relative name (install_consumer/CMakeLists.txt's POST_BUILD step
# copies the fixture next to the built executable, in CONSUMER_BUILD_DIR),
# and execute_process() otherwise defaults to THIS script's own invoking
# directory -- confirmed by running this exact command without it: "cannot
# open 'two_component.yaml'", status FUSE_ERR_INVALID, found looking in the
# wrong directory entirely.
execute_process(
  COMMAND "${CONSUMER_BUILD_DIR}/consumer_fortran"
  WORKING_DIRECTORY "${CONSUMER_BUILD_DIR}"
  OUTPUT_VARIABLE _fuse_actual_output
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "the installed-and-linked Fortran consumer exited ${_fuse_rc}:\n${_fuse_actual_output}")
endif()
message(STATUS "abi.install_consumer_fortran: consumer output:\n${_fuse_actual_output}")

if(DEFINED REFERENCE_OUTPUT)
  if(NOT EXISTS "${REFERENCE_OUTPUT}")
    message(FATAL_ERROR "REFERENCE_OUTPUT was given but does not exist: ${REFERENCE_OUTPUT}")
  endif()
  file(READ "${REFERENCE_OUTPUT}" _fuse_expected_output)
  string(STRIP "${_fuse_actual_output}" _fuse_actual_stripped)
  string(STRIP "${_fuse_expected_output}" _fuse_expected_stripped)
  if(NOT _fuse_actual_stripped STREQUAL _fuse_expected_stripped)
    message(FATAL_ERROR "abi.install_consumer_fortran: output differs from "
                        "${REFERENCE_OUTPUT}:\n--- expected ---\n${_fuse_expected_stripped}\n"
                        "--- actual ---\n${_fuse_actual_stripped}")
  endif()
  message(STATUS "abi.install_consumer_fortran: matches ${REFERENCE_OUTPUT} byte-for-byte")
else()
  message(STATUS "abi.install_consumer_fortran: no REFERENCE_OUTPUT given (the C consumer's "
                 "own golden, test/abi/install_consumer/, did not exist when this was written) "
                 "-- passing on exit code alone; wire REFERENCE_OUTPUT at integration")
endif()
