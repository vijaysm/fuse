# Driver for the "abi.fortran_sizes" ctest entry (ABI-FORTRAN-SIZES,
# build-evidence/abi-matrix.toml). ctest's add_test COMMAND runs one program
# with no shell redirection available; this `cmake -P` script runs
# sizes_test, captures its stdout, and diffs it against
# test/abi/struct_sizes.golden -- the same golden the C ABI's own
# AbiGolden.StructSizesMatchTheTable pins (ABI-GOLDEN-SIZES), so a struct
# that grows without the golden being updated fails on BOTH sides, not just
# the C one.
#
# Required -D arguments: EXE, GOLDEN.
foreach(_fuse_required EXE GOLDEN)
  if(NOT DEFINED ${_fuse_required})
    message(FATAL_ERROR "compare_sizes.cmake: -D${_fuse_required}=... is required")
  endif()
endforeach()

execute_process(
  COMMAND "${EXE}"
  OUTPUT_VARIABLE _fuse_actual
  RESULT_VARIABLE _fuse_rc)
if(NOT _fuse_rc EQUAL 0)
  message(FATAL_ERROR "abi.fortran_sizes: ${EXE} exited ${_fuse_rc}")
endif()

file(READ "${GOLDEN}" _fuse_expected)

# STRIP normalises only the leading/trailing whitespace a WRITE statement's
# newline convention can differ on from the committed file's; the claim
# under test (abi-design.md rule 3 / struct_sizes.golden) is the numbers on
# each line, not the file's exact trailing-newline byte count.
string(STRIP "${_fuse_actual}" _fuse_actual)
string(STRIP "${_fuse_expected}" _fuse_expected)

if(NOT _fuse_actual STREQUAL _fuse_expected)
  message(FATAL_ERROR "abi.fortran_sizes: ffuse's C_SIZEOF table differs from ${GOLDEN}:\n"
                      "--- golden ---\n${_fuse_expected}\n--- ffuse ---\n${_fuse_actual}")
endif()
message(STATUS "abi.fortran_sizes: ffuse's C_SIZEOF table matches ${GOLDEN}")
