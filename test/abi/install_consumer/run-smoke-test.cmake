# Driver for the "abi.install_consumer_c" ctest entry (ABI-C-CONSUMER).
# Modelled directly on test/install_consumer/run-smoke-test.cmake (the
# fuse::base install check this one is the ABI's sibling of), with one
# extra step: after the pure-C consumer runs, the ALREADY-BUILT in-process
# C++ reference (reference.cpp, built as part of the normal tier-0 suite)
# runs too, and the row's claim -- "its result differs from the in-process
# C++ run... beyond 0" -- is checked as an exact string comparison of the
# two captured outputs.
#
# Required -D arguments: FUSE_BUILD_DIR, FUSE_INSTALL_PREFIX, CONSUMER_SRC_DIR,
# CONSUMER_BUILD_DIR, CMAKE_COMMAND, CMAKE_GENERATOR, CMAKE_PREFIX_PATH,
# FUSE_ABI_REFERENCE_EXE.
foreach(_fuse_required FUSE_BUILD_DIR FUSE_INSTALL_PREFIX CONSUMER_SRC_DIR
                        CONSUMER_BUILD_DIR CMAKE_COMMAND CMAKE_GENERATOR
                        FUSE_ABI_REFERENCE_EXE)
  if(NOT DEFINED ${_fuse_required})
    message(FATAL_ERROR "run-smoke-test.cmake: -D${_fuse_required}=... is required")
  endif()
endforeach()
if(NOT EXISTS "${FUSE_ABI_REFERENCE_EXE}")
  message(FATAL_ERROR "run-smoke-test.cmake: reference executable not found: "
                      "${FUSE_ABI_REFERENCE_EXE}")
endif()

file(REMOVE_RECURSE "${FUSE_INSTALL_PREFIX}" "${CONSUMER_BUILD_DIR}")

message(STATUS "abi.install_consumer_c: cmake --install ${FUSE_BUILD_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${FUSE_BUILD_DIR}" --prefix "${FUSE_INSTALL_PREFIX}"
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "cmake --install failed (${_fuse_rc})")
endif()

message(STATUS "abi.install_consumer_c: configuring the C consumer against the install prefix")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${CONSUMER_SRC_DIR}" -B "${CONSUMER_BUILD_DIR}"
          -G "${CMAKE_GENERATOR}"
          "-Dfuse-coupler_DIR=${FUSE_INSTALL_PREFIX}/lib/cmake/fuse-coupler"
          "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
          # <pkg>_DIR for every package this build's configure found, so the
          # consumer resolves the same ones (test/CMakeLists.txt, optional).
          ${FUSE_CONSUMER_DEP_ARGS}
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "configuring the ABI install consumer failed (${_fuse_rc}) -- "
                      "find_package(fuse-coupler) + fuse::abi does not work")
endif()

message(STATUS "abi.install_consumer_c: building the C consumer")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${CONSUMER_BUILD_DIR}"
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "building the ABI install consumer failed (${_fuse_rc})")
endif()

message(STATUS "abi.install_consumer_c: running the C consumer")
execute_process(
  COMMAND "${CONSUMER_BUILD_DIR}/fuse_abi_consumer"
  WORKING_DIRECTORY "${CONSUMER_BUILD_DIR}"
  OUTPUT_VARIABLE _c_out RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "the installed-and-linked C consumer exited ${_fuse_rc}\n${_c_out}")
endif()

message(STATUS "abi.install_consumer_c: running the in-process C++ reference")
execute_process(
  COMMAND "${FUSE_ABI_REFERENCE_EXE}"
  OUTPUT_VARIABLE _cxx_out RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "the in-process C++ reference exited ${_fuse_rc}\n${_cxx_out}")
endif()

if(NOT _c_out STREQUAL _cxx_out)
  message(FATAL_ERROR
    "the C consumer's output differs from the in-process C++ reference\n"
    "--- C (via fuse_coupler_abi) ---\n${_c_out}\n"
    "--- C++ (direct fuse::Runner) ---\n${_cxx_out}\n")
endif()
message(STATUS "abi.install_consumer_c: outputs match byte for byte:\n${_c_out}")
