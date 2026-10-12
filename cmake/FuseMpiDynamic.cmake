# The decisions behind the accept/connect capability probe (KNOWN_ISSUES.md D23,
# D35). They are functions of plain values, with no probe run and no MPI, so
# tools/check-mpi-dynamic-cache.cmake can test every branch on any machine.
#
# WHY THIS EXISTS. The probe's verdict used to be cached once and never looked
# at again. Two ways that goes wrong, both measured on the owner's machine:
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
# option follows the probe, except where the owner set it: a value that differs
# from what the probe itself set last time was an owner's choice and is kept.
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
