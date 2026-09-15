# Asserts that the ABI library's exported symbol set equals the documented
# export list. An accidentally exported symbol is a compatibility promise
# nobody meant to make, and it will not show up in any other test.
#
# Invoked as: cmake -DKARMA_ABI_LIBRARY=<path> -DKARMA_ABI_EXPORT_LIST=<path> -P this

if(NOT EXISTS "${KARMA_ABI_LIBRARY}")
  message(FATAL_ERROR "ABI library not found: ${KARMA_ABI_LIBRARY}")
endif()

file(STRINGS "${KARMA_ABI_EXPORT_LIST}" _lines)
set(_expected "")
foreach(_line IN LISTS _lines)
  string(STRIP "${_line}" _line)
  if(_line STREQUAL "" OR _line MATCHES "^#")
    continue()
  endif()
  # The Mach-O list carries a leading underscore; compare bare names.
  string(REGEX REPLACE "^_" "" _line "${_line}")
  list(APPEND _expected "${_line}")
endforeach()
list(SORT _expected)

if(APPLE)
  execute_process(COMMAND nm -gU "${KARMA_ABI_LIBRARY}"
                  OUTPUT_VARIABLE _out RESULT_VARIABLE _rc ERROR_VARIABLE _err)
else()
  execute_process(COMMAND nm -D --defined-only "${KARMA_ABI_LIBRARY}"
                  OUTPUT_VARIABLE _out RESULT_VARIABLE _rc ERROR_VARIABLE _err)
endif()
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "nm failed: ${_err}")
endif()

string(REPLACE "\n" ";" _rows "${_out}")
set(_actual "")
foreach(_row IN LISTS _rows)
  # "<address> T _name" on Mach-O, "<address> T name" on ELF.
  if(_row MATCHES "[ \t]+[TtWwDdBb][ \t]+([A-Za-z_][A-Za-z0-9_]*)")
    set(_name "${CMAKE_MATCH_1}")
    string(REGEX REPLACE "^_" "" _name "${_name}")
    list(APPEND _actual "${_name}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _actual)
list(SORT _actual)

if(NOT _actual STREQUAL _expected)
  message(FATAL_ERROR
    "exported symbol set differs from the documented export list\n"
    "  documented: ${_expected}\n"
    "  exported:   ${_actual}")
endif()
message(STATUS "ABI exports match the documented list (${_actual})")
