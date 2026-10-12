# fuse_host_core_count(<out>) -- how many processors this machine has,
# independent of OMP_NUM_THREADS.
#
# CMake's ProcessorCount() honours OMP_NUM_THREADS, so a configure run under
# `OMP_NUM_THREADS=1` (a build or ctest command that exports it for the whole
# line) sees ONE core. Two things then go wrong without a message:
#
#   - fuse_add_omp_legs() registers only the `.omp1` leg of each test and
#     drops the all-cores leg (13 rows on a 12-core machine, measured
#     2026-10-12: 193 rows configured with the variable unset, 180 with it set);
#   - every example.*.omp_identical row is registered with THREADS_HI=1, so it
#     compares one thread against one thread. The row exists and passes, and
#     can no longer detect anything: it is the slowest rows in the suite
#     (about 5 minutes for the Brusselator) running a vacuous comparison.
#
# The count is read with the variable removed from the environment of this
# call only, then restored. A genuine single-core machine still returns 1, and
# the omp legs are then not discriminating; test/CMakeLists.txt says so at
# configure time instead of letting that pass unnoticed.
function(fuse_host_core_count out)
  include(ProcessorCount)
  set(_had_omp FALSE)
  if(DEFINED ENV{OMP_NUM_THREADS})
    set(_had_omp TRUE)
    set(_saved_omp "$ENV{OMP_NUM_THREADS}")
    unset(ENV{OMP_NUM_THREADS})
  endif()
  ProcessorCount(_n)
  if(_had_omp)
    set(ENV{OMP_NUM_THREADS} "${_saved_omp}")
  endif()
  set(${out} ${_n} PARENT_SCOPE)
endfunction()
