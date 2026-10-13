# Driver for the "install.consumer_find_package" ctest entry (REVIEW.md
# R-25/BT-F9). ctest's add_test COMMAND runs one program; this is that
# program -- a `cmake -P` script, since the check is itself three separate
# cmake invocations (install, configure a second project, build it), not one
# executable.
#
# Required -D arguments: FUSE_BUILD_DIR, FUSE_INSTALL_PREFIX, CONSUMER_SRC_DIR,
# CONSUMER_BUILD_DIR, CMAKE_COMMAND, CMAKE_GENERATOR, CMAKE_PREFIX_PATH.
foreach(_fuse_required FUSE_BUILD_DIR FUSE_INSTALL_PREFIX CONSUMER_SRC_DIR
                        CONSUMER_BUILD_DIR CMAKE_COMMAND CMAKE_GENERATOR)
  if(NOT DEFINED ${_fuse_required})
    message(FATAL_ERROR "run-smoke-test.cmake: -D${_fuse_required}=... is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${FUSE_INSTALL_PREFIX}" "${CONSUMER_BUILD_DIR}")

message(STATUS "install.consumer_find_package: cmake --install ${FUSE_BUILD_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${FUSE_BUILD_DIR}" --prefix "${FUSE_INSTALL_PREFIX}"
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "cmake --install failed (${_fuse_rc})")
endif()

message(STATUS "install.consumer_find_package: configuring the consumer against the install prefix")
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
  message(FATAL_ERROR "configuring the install consumer failed (${_fuse_rc}) -- "
                      "find_package(fuse-coupler) does not work (REVIEW.md R-25)")
endif()

message(STATUS "install.consumer_find_package: building the consumer")
execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${CONSUMER_BUILD_DIR}"
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "building the install consumer failed (${_fuse_rc})")
endif()

message(STATUS "install.consumer_find_package: running the consumer")
execute_process(
  COMMAND "${CONSUMER_BUILD_DIR}/consumer"
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "the installed-and-linked consumer exited ${_fuse_rc}")
endif()
