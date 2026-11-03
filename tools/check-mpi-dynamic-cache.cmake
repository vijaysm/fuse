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
#  first configure; follows the probe both ways; keeps a user's choice; legacy caches
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

# ---- the per-layout record ---------------------------

# ---- layouts_missing: cached_layouts -> probe needed because of a missing record?
set(_layouts_missing_rows
  "|TRUE"
  "1+1:3/3|FALSE"
  "1+1:3/3\;2+1:3/3|FALSE")
foreach(_row IN LISTS _layouts_missing_rows)
  string(REGEX MATCH "^([^|]*)\\|(.*)$" _m "${_row}")
  fuse_mpi_dynamic_layouts_missing("${CMAKE_MATCH_1}" _got)
  _expect("layouts_missing(${_row})" "${_got}" "${CMAKE_MATCH_2}")
endforeach()

# ---- parse_probe_output: raw script stdout -> token list
# A regex MATCH, not list(): the fabricated "probe stdout" below carries its
# own embedded ';' and blank lines, which list() would read as list
# separators or swallow outright -- the same caution the fingerprint rows
# above are already built around.
set(_probe_stdout_clean "LAYOUT 1+1 PASSED 3 OF 3\nLAYOUT 2+1 PASSED 2 OF 3\n")
fuse_mpi_dynamic_parse_probe_output("${_probe_stdout_clean}" _got)
_expect("parse_probe_output(clean)" "${_got}" "1+1:3/3;2+1:2/3")

set(_probe_stdout_noisy
  "probe-mpi-dynamic: layout 3+2 run 2/3: accept exited 1, connect exited 1\n"
  "--- accepting job (3 ranks) ---\n"
  "    some; stray ; semicolons in a log line\n"
  "# example: LAYOUT 9+9 PASSED 9 OF 9 would be a token if this comment line were not anchored\n"
  "LAYOUT 3+2 PASSED 2 OF 3\n"
  "LAYOUT 1+2 PASSED 0 OF 3\n")
string(CONCAT _probe_stdout_noisy_joined ${_probe_stdout_noisy})
fuse_mpi_dynamic_parse_probe_output("${_probe_stdout_noisy_joined}" _got)
_expect("parse_probe_output(noisy)" "${_got}" "3+2:2/3;1+2:0/3")

fuse_mpi_dynamic_parse_probe_output("" _got)
_expect("parse_probe_output(empty)" "${_got}" "")

# ---- layout_record: cached_layouts + layout key -> "p/n" or ""
set(_fuse_test_layouts "1+1:3/3;2+1:3/3;3+2:2/3;2+3:1/3;1+2:1/3;1+3:0/3")
fuse_mpi_dynamic_layout_record("${_fuse_test_layouts}" "1+1" _got)
_expect("layout_record(1+1)" "${_got}" "3/3")
fuse_mpi_dynamic_layout_record("${_fuse_test_layouts}" "3+2" _got)
_expect("layout_record(3+2)" "${_got}" "2/3")
fuse_mpi_dynamic_layout_record("${_fuse_test_layouts}" "9+9" _got)
_expect("layout_record(absent)" "${_got}" "")
fuse_mpi_dynamic_layout_record("" "1+1" _got)
_expect("layout_record(empty cache)" "${_got}" "")

# ---- layout_working: "p/n" -> N-of-N?
set(_working_rows
  "3/3|TRUE"
  "0/3|FALSE"
  "2/3|FALSE"
  "0/0|TRUE"
  "|FALSE")
foreach(_row IN LISTS _working_rows)
  string(REGEX MATCH "^([^|]*)\\|(.*)$" _m "${_row}")
  fuse_mpi_dynamic_layout_working("${CMAKE_MATCH_1}" _got)
  _expect("layout_working(${_row})" "${_got}" "${CMAKE_MATCH_2}")
endforeach()

# ---- layouts_value: (previous_value, previous_auto, new_value) -> value
#  first configure; follows a fresh probe; keeps a user's hand-edited record
set(_layouts_value_rows
  "||1+1:3/3|1+1:3/3"
  "1+1:3/3|1+1:3/3|1+1:2/3|1+1:2/3"
  "1+1:3/3|1+1:0/3|1+1:2/3|1+1:3/3"
  "1+1:3/3||1+1:2/3|1+1:2/3")
foreach(_row IN LISTS _layouts_value_rows)
  string(REGEX MATCH "^([^|]*)\\|([^|]*)\\|([^|]*)\\|(.*)$" _m "${_row}")
  fuse_mpi_dynamic_layouts_value("${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" "${CMAKE_MATCH_3}" _got)
  _expect("layouts_value(${_row})" "${_got}" "${CMAKE_MATCH_4}")
endforeach()

# ---- end-to-end, against the layouts a real row needs: an MPI that
# measures N of N at 1 rank a side but not at the multi-rank layouts the
# accept/chain/participants rows actually use (a shape seen on a real
# host) disables every row except the
# plain 1+1 pair.
set(_fuse_real_layouts "1+1:3/3;2+1:3/3;3+2:2/3;2+3:3/3;1+2:1/3;1+3:1/3")
set(_row_layout_rows
  "connect.accept.1+1|1+1|TRUE"
  "connect.accept.3+2|3+2|FALSE"
  "connect.accept.2+3|2+3|TRUE"
  "participants.accept.3|2+1|TRUE"
  "participants.accept.5|3+2|FALSE"
  "connect.chain.3 (final step)|1+2|FALSE"
  "connect.chain.4 (final step)|1+3|FALSE")
foreach(_row IN LISTS _row_layout_rows)
  string(REGEX MATCH "^([^|]*)\\|([^|]*)\\|(.*)$" _m "${_row}")
  fuse_mpi_dynamic_layout_record("${_fuse_real_layouts}" "${CMAKE_MATCH_2}" _record)
  fuse_mpi_dynamic_layout_working("${_record}" _got)
  _expect("row_enabled(${CMAKE_MATCH_1})" "${_got}" "${CMAKE_MATCH_3}")
endforeach()

message(STATUS "check-mpi-dynamic-cache: per-layout record rows agree")

file(REMOVE_RECURSE "${FUSE_CHECK_TMP}")
if(_failures GREATER 0)
  message(FATAL_ERROR "check-mpi-dynamic-cache: ${_failures} failure(s)")
endif()
message(STATUS "check-mpi-dynamic-cache: all rows agree")
