# The decisions behind the accept/connect capability probe. They are functions of plain values, with no probe run and no MPI, so
# tools/check-mpi-dynamic-cache.cmake can test every branch on any machine.
#
# WHY THIS EXISTS. The probe's verdict used to be cached once and never looked
# at again. Two ways that goes wrong, both measured on a development machine:
#   - "measured broken" stayed broken after the MPI was reinstalled with a
#     device that works, and FUSE_ENABLE_CONNECT_ACCEPT, an option whose value
#     is cached at the first configure, stayed OFF with it;
#   - "not tested" -- a verdict that records that NO measurement happened --
#     was cached like a result, so a reconfigure on the same machine, after
#     the launcher had become usable, still did not measure.
# A verdict is only worth caching together with the MPI it was measured on.

# fuse_mpi_fingerprint(<out>)
#
# What identifies the MPI the probe measures: the content hash of the first MPI
# library FindMPI reported, of the launcher and of the compiler wrapper. A hash,
# not a timestamp or a version string: a reinstall over the same prefix changes
# the content, and two builds of one version (a different device) are different
# MPIs. A part that does not exist contributes "none", so a machine with no MPI
# gets a stable fingerprint rather than an error.
function(fuse_mpi_fingerprint out)
  set(_lib "")
  foreach(_candidate IN LISTS MPI_C_LIBRARIES)
    if(EXISTS "${_candidate}" AND NOT IS_DIRECTORY "${_candidate}")
      set(_lib "${_candidate}")
      break()
    endif()
  endforeach()
  set(_fp "")
  foreach(_part IN ITEMS "${_lib}" "${MPIEXEC_EXECUTABLE}" "${MPI_C_COMPILER}")
    if(NOT "${_part}" STREQUAL "" AND EXISTS "${_part}")
      file(REAL_PATH "${_part}" _real)
      file(SHA256 "${_real}" _hash)
      string(APPEND _fp "${_real}=${_hash};")
    else()
      string(APPEND _fp "none;")
    endif()
  endforeach()
  set(${out} "${_fp}" PARENT_SCOPE)
endfunction()

# fuse_mpi_dynamic_should_probe(<cached_verdict> <cached_fingerprint> <fingerprint> <out>)
#
# TRUE when the probe must run. A verdict is reused only if it is a measurement
# (so not "not tested") and was taken on the MPI that is configured now.
function(fuse_mpi_dynamic_should_probe cached_verdict cached_fingerprint fingerprint out)
  set(_probe TRUE)
  if(NOT "${cached_verdict}" STREQUAL ""
     AND NOT "${cached_verdict}" STREQUAL "not tested"
     AND "${cached_fingerprint}" STREQUAL "${fingerprint}")
    set(_probe FALSE)
  endif()
  set(${out} ${_probe} PARENT_SCOPE)
endfunction()

# fuse_mpi_dynamic_option_value(<previous_value> <previous_auto> <previous_verdict>
#                               <new_default> <out>)
#
# The value FUSE_ENABLE_CONNECT_ACCEPT should have after this configure. The
# option follows the probe, except where the user set it: a value that differs
# from what the probe itself set last time was the user's choice and is kept.
# A cache from before the probe recorded what it set has no such record; there
# the previous verdict says what the old code would have set (OFF for "measured
# broken", ON otherwise). A first configure has no previous value and takes the
# default.
function(fuse_mpi_dynamic_option_value previous_value previous_auto previous_verdict new_default out)
  if("${previous_value}" STREQUAL "")
    set(${out} ${new_default} PARENT_SCOPE)
    return()
  endif()
  set(_set_by_probe "${previous_auto}")
  if("${_set_by_probe}" STREQUAL "")
    if("${previous_verdict}" STREQUAL "measured broken")
      set(_set_by_probe OFF)
    else()
      set(_set_by_probe ON)
    endif()
  endif()
  if("${previous_value}" STREQUAL "${_set_by_probe}")
    set(${out} ${new_default} PARENT_SCOPE)
  else()
    set(${out} ${previous_value} PARENT_SCOPE)
  endif()
endfunction()

# ---- A single 1-rank-a-side verdict is not a
# measurement of a 3-rank row. The functions below turn the probe's raw,
# multi-layout output into a per-layout record, and read that record back
# for a specific row's own layout -- all pure functions of plain values,
# exactly like the three above, so tools/check-mpi-dynamic-cache.cmake can
# test every branch with no MPI and no probe run.

# fuse_mpi_dynamic_layouts_missing(<cached_layouts> <out>)
#
# TRUE when there is no per-layout record at all. A verdict cached before
# this record existed -- or one whose record was cleared -- is not a
# measurement of any layout, even when fuse_mpi_dynamic_should_probe()
# above would reuse its single legacy verdict (same fingerprint, same MPI):
# the single verdict was only ever a measurement of 1+1.
function(fuse_mpi_dynamic_layouts_missing cached_layouts out)
  if("${cached_layouts}" STREQUAL "")
    set(${out} TRUE PARENT_SCOPE)
  else()
    set(${out} FALSE PARENT_SCOPE)
  endif()
endfunction()

# fuse_mpi_dynamic_parse_probe_output(<probe_output> <out>)
#
# Turns tools/probe-mpi-dynamic.sh's own stdout (one "LAYOUT <na>+<nb>
# PASSED <p> OF <n>" line per layout it tried, in the order it tried them)
# into the cache's own token list, "<na>+<nb>:<p>/<n>" per layout. A line
# that does not match the expected shape (diagnostic text the script wrote
# to stdout by mistake, a truncated capture) is silently not a token --
# the caller sees a shorter list than it asked for, not a malformed one.
function(fuse_mpi_dynamic_parse_probe_output probe_output out)
  set(_tokens "")
  string(REGEX REPLACE "\r\n" "\n" _norm "${probe_output}")
  string(REPLACE "\n" ";" _lines "${_norm}")
  foreach(_line IN LISTS _lines)
    string(REGEX MATCH "^LAYOUT ([0-9]+\\+[0-9]+) PASSED ([0-9]+) OF ([0-9]+)$" _m "${_line}")
    if(_m)
      list(APPEND _tokens "${CMAKE_MATCH_1}:${CMAKE_MATCH_2}/${CMAKE_MATCH_3}")
    endif()
  endforeach()
  set(${out} "${_tokens}" PARENT_SCOPE)
endfunction()

# fuse_mpi_dynamic_layout_record(<cached_layouts> <layout> <out>)
#
# The "<passed>/<total>" string tools/probe-mpi-dynamic.sh recorded for one
# layout, or "" if that layout has no entry (never probed, or a cache from
# before this record existed). cached_layouts is the ';'-joined token list
# fuse_mpi_dynamic_parse_probe_output() produces -- an ordinary CMake list,
# so this reads it with foreach() rather than a single regex over the whole
# string (test/CMakeLists.txt and examples/CMakeLists.txt call this once per
# registered row, not once per configure, so it stays this simple).
function(fuse_mpi_dynamic_layout_record cached_layouts layout out)
  set(_found "")
  foreach(_tok IN LISTS cached_layouts)
    string(REGEX MATCH "^([^:]+):(.+)$" _m "${_tok}")
    if(_m AND CMAKE_MATCH_1 STREQUAL "${layout}")
      set(_found "${CMAKE_MATCH_2}")
    endif()
  endforeach()
  set(${out} "${_found}" PARENT_SCOPE)
endfunction()

# fuse_mpi_dynamic_layout_working(<record> <out>)
#
# TRUE iff <record> (fuse_mpi_dynamic_layout_record()'s own output) reads
# "N of N" -- every run of that layout passed. An empty record (never
# measured) and a "p/n" with p < n (measured, but not reliably) are both
# FALSE: the N-of-N rule, with no
# partial credit and no constant to widen after a row still fails.
function(fuse_mpi_dynamic_layout_working record out)
  set(_working FALSE)
  if(NOT "${record}" STREQUAL "")
    string(REGEX MATCH "^([0-9]+)/([0-9]+)$" _m "${record}")
    if(_m AND CMAKE_MATCH_1 STREQUAL "${CMAKE_MATCH_2}")
      set(_working TRUE)
    endif()
  endif()
  set(${out} ${_working} PARENT_SCOPE)
endfunction()

# fuse_mpi_dynamic_layouts_value(<previous_value> <previous_auto> <new_value> <out>)
#
# The value FUSE_MPI_DYNAMIC_LAYOUTS should have after a configure that DID
# re-probe. Same by-hand-override rule as fuse_mpi_dynamic_option_value()
# above, written separately because that function's
# migration branch is specific to a two-valued ON/OFF option and this one
# carries an opaque string record instead: the user's own edit is kept
# only when it differs from what the probe itself set last time; a first
# configure (no previous record at all) takes the fresh measurement
# unconditionally.
function(fuse_mpi_dynamic_layouts_value previous_value previous_auto new_value out)
  if("${previous_value}" STREQUAL "" OR "${previous_auto}" STREQUAL "")
    set(${out} "${new_value}" PARENT_SCOPE)
    return()
  endif()
  if("${previous_value}" STREQUAL "${previous_auto}")
    set(${out} "${new_value}" PARENT_SCOPE)
  else()
    set(${out} "${previous_value}" PARENT_SCOPE)
  endif()
endfunction()
