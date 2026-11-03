# Stands in for "abi.install_consumer_fortran" when TARGET fuse-abi does not
# exist at configure time: this file is reached only when the C ABI table
# (src/abi, build-evidence/abi-matrix.toml) is not part of the checkout.
# FAILS on purpose, with a message naming exactly why, rather
# than being silently absent (which would read like "nothing to test here")
# or SKIPPED (which many CI summaries render indistinguishably from passed).
# Once libfuse-abi exists, test/CMakeLists.txt's `if(TARGET fuse-abi)` takes
# the other branch on the next configure and this file stops being reached.
message(FATAL_ERROR
  "abi.install_consumer_fortran: awaits libfuse-abi -- TARGET fuse-abi did "
  "not exist in this configure (the C ABI library under src/abi is "
  "not part of this checkout). consumer.f90 already compiles against "
  "fuse-abi-fortran alone (ctest abi.fortran_consumer_compiles passes without "
  "libfuse-abi); linking an EXECUTABLE that actually calls into ffuse_* needs "
  "fuse_coupler_abi() resolved, which only libfuse-abi provides. Re-run cmake "
  "once that target exists -- this test activates automatically, no edit "
  "needed here.")
