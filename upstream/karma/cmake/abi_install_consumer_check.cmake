# Installs KARMA into a scratch prefix and builds the pure-C consumer against
# it. This is the ABI's install check: it proves the installed headers are
# self-contained, that one include path is enough, and that a C program needs
# nothing from KARMA's C++ surface.
#
# Invoked as: cmake -DKARMA_BUILD_DIR=<dir> -DKARMA_SOURCE_DIR=<dir>
#                   -DKARMA_SCRATCH=<dir> -P this

set(_prefix "${KARMA_SCRATCH}/prefix")
set(_build "${KARMA_SCRATCH}/consumer-build")
file(REMOVE_RECURSE "${_prefix}" "${_build}")

execute_process(COMMAND ${CMAKE_COMMAND} --install "${KARMA_BUILD_DIR}" --prefix "${_prefix}"
                OUTPUT_VARIABLE _out ERROR_VARIABLE _err RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "install failed: ${_err}")
endif()

foreach(_header karma_abi.h karma_component_abi.h)
  if(NOT EXISTS "${_prefix}/include/karma/abi/${_header}")
    message(FATAL_ERROR "installed header missing: karma/abi/${_header}")
  endif()
endforeach()

execute_process(COMMAND ${CMAKE_COMMAND}
                        -S "${KARMA_SOURCE_DIR}/test/abi_install_consumer"
                        -B "${_build}"
                        "-DKARMA_PREFIX=${_prefix}"
                        "-DCMAKE_BUILD_TYPE=Release"
                OUTPUT_VARIABLE _out ERROR_VARIABLE _err RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "consumer configure failed: ${_err}\n${_out}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} --build "${_build}"
                OUTPUT_VARIABLE _out ERROR_VARIABLE _err RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "consumer build failed: ${_err}\n${_out}")
endif()

execute_process(COMMAND "${_build}/karma_abi_consumer"
                OUTPUT_VARIABLE _out ERROR_VARIABLE _err RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "consumer run failed (${_rc}): ${_out}${_err}")
endif()
message(STATUS "installed consumer: ${_out}")
