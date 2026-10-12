# tools/check-mpi-dynamic-cache.cmake -- run as `cmake -P`: every branch of the
# decisions in cmake/FuseMpiDynamic.cmake, on plain values, with no MPI and no
# probe run. Exit status 0 when every row of every table agrees.
#
#     cmake -DFUSE_CHECK_TMP=<scratch dir> -P tools/check-mpi-dynamic-cache.cmake
#
# FUSE_CHECK_TMP is a directory the check may write fake "MPI" files into (the
# fingerprint hashes real files); it is created and removed here.

get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
include("${_root}/cmake/FuseMpiDynamic.cmake")
if(NOT FUSE_CHECK_TMP)
  set(FUSE_CHECK_TMP "$ENV{TMPDIR}/fuse-mpi-dynamic-check")
endif()

set(_failures 0)
macro(_expect label got want)
  if(NOT "${got}" STREQUAL "${want}")
    message(SEND_ERROR "${label}: got '${got}', expected '${want}'")
    math(EXPR _failures "${_failures} + 1")
  endif()
endmacro()

# ---- should_probe: (cached verdict, cached fingerprint, current fingerprint) -> probe?
#  1  no verdict yet                                  -> probe
#  2  a measurement on this MPI                       -> reuse (both polarities)
#  3  a measurement on a DIFFERENT MPI                -> probe (both polarities)
#  4  "not tested" is not a measurement               -> probe, even on the same MPI
set(_probe_rows
  "|||TRUE"
  "measured working|A|A|FALSE"
  "measured broken|A|A|FALSE"
  "measured working|A|B|TRUE"
  "measured broken|A|B|TRUE"
  "not tested|A|A|TRUE"
  "not tested|A|B|TRUE"
  "measured broken||A|TRUE")
foreach(_row IN LISTS _probe_rows)
  # a regex, not list(): list() would swallow the empty fields
  string(REGEX MATCH "^([^|]*)\\|([^|]*)\\|([^|]*)\\|(.*)$" _m "${_row}")
  fuse_mpi_dynamic_should_probe("${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" "${CMAKE_MATCH_3}" _got)
  _expect("should_probe(${_row})" "${_got}" "${CMAKE_MATCH_4}")
endforeach()

# ---- option_value: (previous value, previous auto, previous verdict, new default) -> value
#  first configure; follows the probe both ways; keeps an owner's choice; legacy caches
set(_option_rows
  "|||ON|ON"
  "OFF|OFF|measured broken|ON|ON"
  "ON|ON|measured working|OFF|OFF"
  "OFF|ON|measured working|ON|OFF"
  "ON|OFF|measured broken|OFF|ON"
  "ON||not tested|ON|ON"
  "OFF||measured broken|ON|ON"
  "ON||measured broken|ON|ON"
  "OFF||not tested|ON|OFF")
foreach(_row IN LISTS _option_rows)
  string(REGEX MATCH "^([^|]*)\\|([^|]*)\\|([^|]*)\\|([^|]*)\\|(.*)$" _m "${_row}")
  fuse_mpi_dynamic_option_value("${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" "${CMAKE_MATCH_3}"
                                "${CMAKE_MATCH_4}" _got)
  _expect("option_value(${_row})" "${_got}" "${CMAKE_MATCH_5}")
endforeach()

# ---- fingerprint: content, not path or time
file(REMOVE_RECURSE "${FUSE_CHECK_TMP}")
file(MAKE_DIRECTORY "${FUSE_CHECK_TMP}")
file(WRITE "${FUSE_CHECK_TMP}/libmpi.dylib" "build one")
file(WRITE "${FUSE_CHECK_TMP}/mpiexec" "launcher")
file(WRITE "${FUSE_CHECK_TMP}/mpicc" "wrapper")
set(MPI_C_LIBRARIES "${FUSE_CHECK_TMP}/does-not-exist.dylib;${FUSE_CHECK_TMP}/libmpi.dylib")
set(MPIEXEC_EXECUTABLE "${FUSE_CHECK_TMP}/mpiexec")
set(MPI_C_COMPILER "${FUSE_CHECK_TMP}/mpicc")
fuse_mpi_fingerprint(_fp_a)
fuse_mpi_fingerprint(_fp_again)
_expect("fingerprint is stable for an unchanged MPI" "${_fp_again}" "${_fp_a}")

file(WRITE "${FUSE_CHECK_TMP}/libmpi.dylib" "build two")
fuse_mpi_fingerprint(_fp_lib)
if("${_fp_lib}" STREQUAL "${_fp_a}")
  message(SEND_ERROR "fingerprint did not change when the MPI library's content changed")
  math(EXPR _failures "${_failures} + 1")
endif()
file(WRITE "${FUSE_CHECK_TMP}/libmpi.dylib" "build one")
fuse_mpi_fingerprint(_fp_back)
_expect("fingerprint returns when the content returns (a hash, not a timestamp)" "${_fp_back}" "${_fp_a}")

file(WRITE "${FUSE_CHECK_TMP}/mpiexec" "another launcher")
fuse_mpi_fingerprint(_fp_launcher)
if("${_fp_launcher}" STREQUAL "${_fp_a}")
  message(SEND_ERROR "fingerprint did not change when the launcher changed")
  math(EXPR _failures "${_failures} + 1")
endif()

set(MPI_C_LIBRARIES "")
set(MPIEXEC_EXECUTABLE "")
set(MPI_C_COMPILER "")
fuse_mpi_fingerprint(_fp_none)
_expect("a machine with no MPI gets the stable fingerprint" "${_fp_none}" "none;none;none;")

file(REMOVE_RECURSE "${FUSE_CHECK_TMP}")
if(_failures GREATER 0)
  message(FATAL_ERROR "check-mpi-dynamic-cache: ${_failures} failure(s)")
endif()
message(STATUS "check-mpi-dynamic-cache: all rows agree")
