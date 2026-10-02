# Stands in for "abi.install_consumer_fortran" when TARGET fuse-abi does not
# exist at configure time: the task that produced this test track and the C
# ABI table (src/abi, build-evidence/abi-matrix.toml) run in parallel on one
# checkout, and this file is reached only when the other side has not
# landed yet. FAILS on purpose, with a message naming exactly why, rather
# than being silently absent (which would read like "nothing to test here")
# or SKIPPED (which many CI summaries render indistinguishably from passed).
# Once libfuse-abi exists, test/CMakeLists.txt's `if(TARGET fuse-abi)` takes
# the other branch on the next configure and this file stops being reached.
message(FATAL_ERROR
  "abi.install_consumer_fortran: awaits libfuse-abi -- TARGET fuse-abi did "
  "not exist in this configure (the other S6-phase-2 track's src/abi has "
  "not landed in this checkout yet). consumer.f90 already compiles against "
  "fuse-abi-fortran alone (ctest abi.fortran_consumer_compiles passes without "
  "libfuse-abi); linking an EXECUTABLE that actually calls into ffuse_* needs "
  "fuse_coupler_abi() resolved, which only libfuse-abi provides. Re-run cmake "
  "once that target exists -- this test activates automatically, no edit "
  "needed here.")
