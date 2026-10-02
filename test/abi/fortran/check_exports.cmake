# Driver for the "abi.fortran_exports" ctest entry. ctest's add_test COMMAND
# runs one program; a `cmake -P` script is that program here, the same
# pattern test/install_consumer/run-smoke-test.cmake uses for a check that is
# not itself a single executable.
#
# Pins abi-design.md's "Link names are ffuse_*, never fuse_*": nm -g
# --defined-only on the SHIM'S OWN COMPILED OBJECT (extracted from the
# archive with `ar x`, not the whole archive) must list exactly the
# ffuse_<entry> family and nothing matching ^fuse_ (which would collide
# with libfuse, the filesystem project, exactly the hazard the
# one-exported-symbol C ABI exists to avoid -- docs/dev/abi-design.md's
# "The constraint"). The whole archive also contains ffuse.o, whose
# gfortran-mangled module-internal symbols (___ffuse_MOD_..., the compiler's
# own derived-type copy/vtable/initialiser helpers) are Fortran-language
# plumbing no C caller ever references and no amount of -fvisibility
# hides from gfortran's own object format -- not the collision surface
# abi-design.md is about, and parsing them correctly is one more way for
# this check to be fooled by a compiler-specific naming quirk than
# checking the shim object alone.
#
# Required -D arguments: NM, AR, LIBRARY, MEMBER (the archive member to
# extract, e.g. fuse_fortran_shim.c.o), ENTRY_NAMES (semicolon-separated
# entry names from entries.def, e.g. "version;capabilities;...").
foreach(_fuse_required NM AR LIBRARY MEMBER ENTRY_NAMES)
  if(NOT DEFINED ${_fuse_required})
    message(FATAL_ERROR "check_exports.cmake: -D${_fuse_required}=... is required")
  endif()
endforeach()

if(NOT EXISTS "${LIBRARY}")
  message(FATAL_ERROR "check_exports.cmake: library not found: ${LIBRARY}")
endif()

set(_fuse_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/fortran_exports_check")
file(REMOVE_RECURSE "${_fuse_extract_dir}")
file(MAKE_DIRECTORY "${_fuse_extract_dir}")
execute_process(
  COMMAND "${AR}" x "${LIBRARY}" "${MEMBER}"
  WORKING_DIRECTORY "${_fuse_extract_dir}"
  RESULT_VARIABLE _fuse_ar_rc
)
if(NOT _fuse_ar_rc EQUAL 0)
  message(FATAL_ERROR "ar x ${LIBRARY} ${MEMBER} failed (${_fuse_ar_rc})")
endif()
set(_fuse_member_path "${_fuse_extract_dir}/${MEMBER}")
if(NOT EXISTS "${_fuse_member_path}")
  message(FATAL_ERROR "ar x did not produce ${_fuse_member_path}")
endif()

execute_process(
  COMMAND "${NM}" -g --defined-only "${_fuse_member_path}"
  OUTPUT_VARIABLE _fuse_nm_out
  RESULT_VARIABLE _fuse_nm_rc
)
if(NOT _fuse_nm_rc EQUAL 0)
  message(FATAL_ERROR "nm failed on ${_fuse_member_path} (${_fuse_nm_rc})")
endif()

# nm on an ARCHIVE (as opposed to one object file) prints a bare
# "<member>.o:" header line before each member's symbols -- e.g.
# "fuse_fortran_shim.c.o:" -- which a blanket "every identifier-looking
# token in the whole blob" extraction mistakes for a symbol named
# fuse_fortran_shim (caught running this exact check: it failed on that
# name, which nm never actually listed as a defined global). Line-by-line,
# keeping only lines shaped like a real nm symbol entry (hex address, one
# type letter, the name) and skipping the member-header lines (no leading
# address, end in ":"), is what distinguishes the two.
string(REPLACE "\n" ";" _fuse_nm_lines "${_fuse_nm_out}")

set(_fuse_bad "")
set(_fuse_seen_expected "")
foreach(_line IN LISTS _fuse_nm_lines)
  string(STRIP "${_line}" _line)
  if(_line STREQUAL "" OR _line MATCHES ":$")
    continue()
  endif()
  if(NOT _line MATCHES "^[0-9a-fA-F]+ +[A-Za-z] +(.+)$")
    continue()
  endif()
  set(_tok "${CMAKE_MATCH_1}")
  string(REGEX REPLACE "^_" "" _name "${_tok}")
  if(_name MATCHES "^fuse_")
    list(APPEND _fuse_bad "${_name}")
  elseif(_name MATCHES "^ffuse_")
    list(APPEND _fuse_seen_expected "${_name}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _fuse_bad)
list(REMOVE_DUPLICATES _fuse_seen_expected)

if(_fuse_bad)
  list(JOIN _fuse_bad ", " _fuse_bad_text)
  message(FATAL_ERROR "abi.fortran_exports: library exports a fuse_* symbol "
                      "(forbidden -- abi-design.md \"Link names\"): ${_fuse_bad_text}")
endif()

set(_fuse_missing "")
foreach(_entry IN LISTS ENTRY_NAMES)
  if(NOT "ffuse_${_entry}" IN_LIST _fuse_seen_expected)
    list(APPEND _fuse_missing "ffuse_${_entry}")
  endif()
endforeach()
if(_fuse_missing)
  list(JOIN _fuse_missing ", " _fuse_missing_text)
  message(FATAL_ERROR "abi.fortran_exports: expected exported symbol(s) missing "
                      "from ${LIBRARY}: ${_fuse_missing_text}")
endif()

message(STATUS "abi.fortran_exports: ${LIBRARY} exports exactly the ffuse_* family "
               "(${_fuse_seen_expected}), no fuse_* symbol")
