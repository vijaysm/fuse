# Run BIN at OMP_NUM_THREADS=1 and =THREADS_HI and require the two standard
# outputs to be byte-identical, after dropping the lines Kokkos prints about
# the thread environment itself (which legitimately differ). Any other
# difference is a result that depends on the thread count.
#
# Invoked by CTest as `cmake -DBIN=... -DTHREADS_HI=... -DOUT=... -P this`.
if(NOT BIN OR NOT THREADS_HI OR NOT OUT)
  message(FATAL_ERROR "compare-runs.cmake: BIN, THREADS_HI and OUT are required")
endif()
set(_strip "Kokkos::OpenMP|OMP_PROC_BIND|OMP_PLACES|For best performance|For unit testing|In general, for best|oversubscribing|Detected:|Requested:|MPI detected|\\[  spaces  \\]|^[ \t]*$")
foreach(n 1 ${THREADS_HI})
  set(ENV{OMP_NUM_THREADS} ${n})
  set(ENV{OMP_PROC_BIND} false)
  # ARGS is an optional ;-list forwarded to the binary, so an example whose
  # registered test needs a command line can be compared at that same line.
  execute_process(COMMAND ${BIN} ${ARGS} OUTPUT_VARIABLE _out ERROR_VARIABLE _err RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "compare-runs.cmake: ${BIN} exited ${_rc} at OMP_NUM_THREADS=${n}\n${_err}")
  endif()
  string(REPLACE "\n" ";" _lines "${_out}")
  set(_kept "")
  foreach(l IN LISTS _lines)
    if(NOT l MATCHES "${_strip}")
      string(APPEND _kept "${l}\n")
    endif()
  endforeach()
  file(WRITE "${OUT}.${n}.txt" "${_kept}")
  set(_run_${n} "${_kept}")
endforeach()
if(NOT "${_run_1}" STREQUAL "${_run_${THREADS_HI}}")
  execute_process(COMMAND diff "${OUT}.1.txt" "${OUT}.${THREADS_HI}.txt" OUTPUT_VARIABLE _d)
  message(FATAL_ERROR "compare-runs.cmake: output differs between OMP_NUM_THREADS=1 and ${THREADS_HI}:\n${_d}")
endif()
string(LENGTH "${_run_1}" _len)
if(_len LESS 200)
  message(FATAL_ERROR "compare-runs.cmake: output is only ${_len} bytes; an empty run would compare equal, so this is refused")
endif()
message(STATUS "identical at OMP_NUM_THREADS=1 and ${THREADS_HI} (${_len} bytes compared)")
