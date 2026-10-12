# tools/check-host-cores.cmake -- run as `cmake -P`: the host core count that
# the omp legs are registered with must not depend on OMP_NUM_THREADS
# (cmake/FuseHostCores.cmake says why). Exit status 0 on agreement.
#
# WHY IT BUILDS ITS OWN TOOLS. ProcessorCount() asks `sysctl -n hw.ncpu` first
# and falls through to `nproc`, and only GNU nproc reads OMP_NUM_THREADS. On a
# Mac whose sysctl works the dependence never shows, so a check that merely
# called the helper there would pass whether or not the helper is correct (the
# first version of this file did, and passed on a copy with the fix removed).
# Here a `sysctl` that fails and an `nproc` that behaves like GNU's are put
# first on PATH, which makes the helper reach the environment-reading branch
# on every machine. find_program() searches PATH before its own PATHS, so the
# fakes win over /usr/sbin/sysctl.
#
#   cmake [-DFUSE_CHECK_TMP=<dir>] -P tools/check-host-cores.cmake
if(WIN32)
  message(STATUS "check-host-cores: skipped on Windows (ProcessorCount reads NUMBER_OF_PROCESSORS there)")
  return()
endif()
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
include(${_root}/cmake/FuseHostCores.cmake)

if(NOT FUSE_CHECK_TMP)
  set(FUSE_CHECK_TMP "/tmp/fuse-host-cores-probe")
endif()
file(REMOVE_RECURSE "${FUSE_CHECK_TMP}")
file(MAKE_DIRECTORY "${FUSE_CHECK_TMP}/bin")
# Fails the way a blocked or absent sysctl does.
file(WRITE "${FUSE_CHECK_TMP}/bin/sysctl" "#!/bin/sh\nexit 1\n")
# GNU coreutils' documented behaviour: OMP_NUM_THREADS, when set, is the answer.
# 7 is the "real" count, deliberately not a power of two or a likely OMP value.
file(WRITE "${FUSE_CHECK_TMP}/bin/nproc"
  "#!/bin/sh\nif [ -n \"$OMP_NUM_THREADS\" ]; then echo \"$OMP_NUM_THREADS\"; else echo 7; fi\n")
file(CHMOD "${FUSE_CHECK_TMP}/bin/sysctl" "${FUSE_CHECK_TMP}/bin/nproc"
     PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
set(ENV{PATH} "${FUSE_CHECK_TMP}/bin:$ENV{PATH}")

# The fakes must be what the helper reaches; otherwise the comparison below
# proves nothing, and it is reported as such rather than as a pass.
unset(ENV{OMP_NUM_THREADS})
fuse_host_core_count(_unset)
set(ENV{OMP_NUM_THREADS} 1)
fuse_host_core_count(_one)
set(ENV{OMP_NUM_THREADS} 3)
fuse_host_core_count(_three)
set(_after "$ENV{OMP_NUM_THREADS}")
file(REMOVE_RECURSE "${FUSE_CHECK_TMP}")

message(STATUS "host core count (fake nproc, failing sysctl): unset=${_unset} OMP_NUM_THREADS=1 -> ${_one} OMP_NUM_THREADS=3 -> ${_three}")
if(NOT _unset EQUAL 7)
  message(FATAL_ERROR
    "the probe did not reach the fake nproc (the unset count is ${_unset}, expected 7): "
    "this check is not exercising the code path it exists for")
endif()
if(NOT _after STREQUAL "3")
  message(FATAL_ERROR "fuse_host_core_count() did not restore OMP_NUM_THREADS (found '${_after}', expected 3)")
endif()
if(NOT _one EQUAL _unset OR NOT _three EQUAL _unset)
  message(FATAL_ERROR
    "the host core count depends on OMP_NUM_THREADS (unset: ${_unset}, =1: ${_one}, =3: ${_three}); a "
    "configure run under that variable would register the omp legs and the omp_identical rows with the "
    "wrong thread count")
endif()
