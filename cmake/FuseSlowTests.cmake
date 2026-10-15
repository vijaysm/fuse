# The tests that are too slow to run on every edit.
#
# A row is listed here when its measured average wall time is at least
# 10 s (CTestCostData.txt from the development host's `all` build, 313 rows,
# summed 1566 s; these 17 rows are 93% of that sum, the other
# 296 rows total 103 s). They get the ctest label `slow`, so
#
#     ctest -LE slow          or          ctest --preset tier0-quick
#
# runs everything else. Every row still runs in a full run (`ctest`) -- the
# label selects, it never disables. tools/check-test-cost.py compares this
# list with a build directory's measured costs, so the quick tier cannot
# silently grow back into the slow one.
#
# What the quick tier cannot see: code compiled only under FUSE_HAVE_MPI
# (tier 0 never compiles it, and the multi-rank rows need a launcher), and the
# OpenMP determinism rows (`*.omp_identical`), which are slow because each
# runs a whole example at one thread and again at every core and compares the
# two outputs exactly (tools/compare-runs.cmake).
set(FUSE_SLOW_TESTS
  example.brusselator.newton_petsc   # 10.1 s average over 9 runs, build-all, 2026-10-15
  example.brusselator.omp_identical                    #  313.2 s
  petsc.brusselator_newton                             #  244.2 s average of 5 runs; a passing run took 1220.8 s (the other four were aborted early)
  verification.gray_scott_reference                    #  238.2 s
  verification.fsi_transient                           #  198.3 s
  verification.fsi_energy_diag                         #  123.9 s
  example.gray_scott.omp_identical                     #   51.1 s
  verification.gray_scott_mms                          #   49.5 s
  verification.brusselator_participants                #   46.8 s
  verification.brusselator_lowfive_harness             #   45.7 s
  verification.fsi_stokes_mms                          #   35.1 s
  verification.fsi_coupled                             #   30.6 s
  example.gray_scott                                   #   25.7 s
  example.dn_heat.omp_identical                        #   13.9 s
  example.fsi_channel.omp_identical                    #   13.4 s
  example.brusselator.newton                           #   12.7 s
  verification.brusselator_runner                      #   10.4 s
  verification.gray_scott_multirate                    #   10.0 s
)

# Appends the label to each listed row that exists in the CURRENT directory:
# a test property is directory-scoped, so this is called at the end of every
# directory that registers tests.
function(fuse_label_slow_tests)
  foreach(slow_test IN LISTS FUSE_SLOW_TESTS)
    if(TEST ${slow_test})
      set_property(TEST ${slow_test} APPEND PROPERTY LABELS slow)
    endif()
  endforeach()
endfunction()
