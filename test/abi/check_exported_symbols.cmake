# ABI-GOLDEN-SYMBOLS (build-evidence/abi-matrix.toml): asserts libfuse-abi's
# built shared library exports exactly the one symbol
# test/abi/exported_symbols.golden names. An accidentally exported symbol
# -fvisibility=hidden should have caught is a compatibility promise nobody
# meant to make and this is the only place that would ever say so; ALSO
# fails if the library was built without -fvisibility=hidden, since in that
# case this list would show every public fuse:: symbol the library
# statically links, not just one.
#
# Invoked as: cmake -DFUSE_ABI_LIBRARY=<path> -DFUSE_ABI_GOLDEN=<path> -P this
# Adapted from upstream/karma/cmake/abi_exports_check.cmake, which solves the
# identical problem for libkarma_abi; see that file for the platform-nm
# reasoning this keeps unchanged.

if(NOT EXISTS "${FUSE_ABI_LIBRARY}")
  message(FATAL_ERROR "ABI library not found: ${FUSE_ABI_LIBRARY}")
endif()

file(STRINGS "${FUSE_ABI_GOLDEN}" _lines)
set(_expected "")
foreach(_line IN LISTS _lines)
  string(STRIP "${_line}" _line)
  if(_line STREQUAL "" OR _line MATCHES "^#")
    continue()
  endif()
  list(APPEND _expected "${_line}")
endforeach()
list(SORT _expected)

if(APPLE)
  execute_process(COMMAND nm -gU "${FUSE_ABI_LIBRARY}"
                  OUTPUT_VARIABLE _out RESULT_VARIABLE _rc ERROR_VARIABLE _err)
else()
  execute_process(COMMAND nm -D --defined-only "${FUSE_ABI_LIBRARY}"
                  OUTPUT_VARIABLE _out RESULT_VARIABLE _rc ERROR_VARIABLE _err)
endif()
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "nm failed: ${_err}")
endif()

string(REPLACE "\n" ";" _rows "${_out}")
set(_actual "")
foreach(_row IN LISTS _rows)
  # "<address> T _name" on Mach-O, "<address> T name" on ELF. Only an
  # uppercase T (a defined, exported TEXT/code symbol) counts -- data
  # symbols are not what "one exported function" claims.
  if(_row MATCHES "[ \t]+T[ \t]+([A-Za-z_][A-Za-z0-9_]*)")
    set(_name "${CMAKE_MATCH_1}")
    # The Mach-O list carries a leading underscore; compare bare names.
    string(REGEX REPLACE "^_" "" _name "${_name}")
    list(APPEND _actual "${_name}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _actual)
list(SORT _actual)

if(NOT _actual STREQUAL _expected)
  message(FATAL_ERROR
    "exported symbol set differs from ${FUSE_ABI_GOLDEN}\n"
    "  expected: ${_expected}\n"
    "  actual:   ${_actual}")
endif()
message(STATUS "fuse-abi exports match the golden list (${_actual})")
