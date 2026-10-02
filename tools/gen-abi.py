#!/usr/bin/env python3
"""Generate the plain-C ABI headers (and, later, the Fortran module) from
include/fuse/abi/entries.def.

The .def file is the single source of truth; the headers a user reads carry no
macros (docs/dev/abi-design.md, point 5). So the headers are GENERATED here
and committed, and test/abi/entries_golden_test checks that regenerating them
reproduces the committed bytes exactly. A hand edit to a header therefore
fails the suite until the .def is changed to match.

    python3 tools/gen-abi.py --check        # exit 1 if any output differs
    python3 tools/gen-abi.py --write        # rewrite the outputs

Stdlib only; the .def is parsed with a small tokenizer, not the C
preprocessor, so the file's own `#ifndef` scaffolding is ignored.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEF = ROOT / "include/fuse/abi/entries.def"
OUT_TYPES = ROOT / "include/fuse/abi/types.h"
OUT_PARTICIPANT = ROOT / "include/fuse/abi/participant.h"
OUT_FORTRAN_MODULE = ROOT / "fortran/ffuse.f90"
OUT_FORTRAN_SHIM = ROOT / "fortran/fuse_fortran_shim.c"

# Entries whose IN struct (or, for last_error, OUT struct) carries a
# NUL-terminated string the Fortran side should be allowed to pass as an
# ordinary CHARACTER actual argument (docs/dev/abi-design.md, "Fortran:
# ffuse", "Strings"). Each maps to the names of the convenience procedure(s)
# generated alongside the plain struct-taking one; all are gathered under one
# PUBLIC generic interface named after the entry, so `fuse_<name>(...)` always
# dispatches to whichever specific procedure matches the actual arguments.
FORTRAN_STRING_ENTRIES = {"load_graph", "field_handle", "begin_step", "end_step", "last_error"}

# ---------------------------------------------------------------- parsing

_MACRO = re.compile(r"^(FUSE_ABI_(?:STRUCT_END|STRUCT|FIELD|ENTRY|ENUM))\((.*)\)\s*$")


def _split_args(s: str) -> list[str]:
    """Split a macro argument list on top-level commas, honouring quotes."""
    out, cur, depth, quote = [], [], 0, None
    for ch in s:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            cur.append(ch)
        elif ch in "([":
            depth += 1
            cur.append(ch)
        elif ch in ")]":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return out


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


class Spec:
    def __init__(self) -> None:
        self.enums: list[tuple[str, str, str]] = []
        self.structs: list[tuple[str, str, list[tuple[str, str, str, str]]]] = []
        self.entries: list[tuple[str, str, str, str]] = []


def parse(path: Path = DEF) -> Spec:
    spec = Spec()
    cur: tuple[str, str, list] | None = None
    for ln, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        m = _MACRO.match(line)
        if not m:
            continue
        kind, args = m.group(1), _split_args(m.group(2))
        if kind == "FUSE_ABI_ENUM":
            name, value, doc = args
            spec.enums.append((name, value, _unquote(doc)))
        elif kind == "FUSE_ABI_STRUCT":
            if cur is not None:
                sys.exit(f"{path}:{ln}: FUSE_ABI_STRUCT inside an open struct")
            name, doc = args
            cur = (name, _unquote(doc), [])
        elif kind == "FUSE_ABI_FIELD":
            if cur is None:
                sys.exit(f"{path}:{ln}: FUSE_ABI_FIELD outside a struct")
            ctype, ftype, name, doc = args
            cur[2].append((ctype, ftype, name, _unquote(doc)))
        elif kind == "FUSE_ABI_STRUCT_END":
            if cur is None or cur[0] != args[0]:
                sys.exit(f"{path}:{ln}: FUSE_ABI_STRUCT_END({args[0]}) does not close {cur and cur[0]}")
            spec.structs.append(cur)
            cur = None
        elif kind == "FUSE_ABI_ENTRY":
            name, inp, out, doc = args
            spec.entries.append((name, inp, out, _unquote(doc)))
    if cur is not None:
        sys.exit(f"{path}: struct {cur[0]} never closed")
    _validate(spec)
    return spec


def _validate(spec: Spec) -> None:
    names = {s[0] for s in spec.structs}
    for sname, _, fields in spec.structs:
        if not fields or fields[0][2] != "struct_size" or fields[0][0] != "uint32_t":
            sys.exit(f"{sname}: first member must be `uint32_t struct_size`")
        seen = set()
        for _, _, f, _ in fields:
            if f in seen:
                sys.exit(f"{sname}: duplicate member {f}")
            seen.add(f)
    for name, inp, out, _ in spec.entries:
        for t in (inp, out):
            if t not in names:
                sys.exit(f"entry {name}: {t} is not a declared struct")
    enum_names = [e[0] for e in spec.enums]
    if len(set(enum_names)) != len(enum_names):
        sys.exit("duplicate enumerator")


# ------------------------------------------------------------- generation

BANNER = """/* GENERATED by tools/gen-abi.py from include/fuse/abi/entries.def -- do not
 * edit; edit the .def and run `python3 tools/gen-abi.py --write`. The golden
 * test test/abi/entries_golden_test.cpp fails when this file and the .def
 * disagree. Plain C11, no macros beyond the include guard and extern "C". */
"""


def _wrap_doc(doc: str, indent: str) -> str:
    if not doc:
        return ""
    words, lines, cur = doc.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 76 - len(indent):
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "".join(f"{indent}/* {l} */\n" for l in lines)


def gen_types(spec: Spec) -> str:
    o = [BANNER, "#ifndef FUSE_ABI_TYPES_H", "#define FUSE_ABI_TYPES_H", "",
         "#include <stdint.h>", "", "#ifdef __cplusplus", 'extern "C" {', "#endif", ""]
    o.append("/* ---- status codes, enumerators and limits ---------------------------- */")
    o.append("/* Values are part of the ABI and never change. */")
    o.append("enum {")
    for i, (name, value, doc) in enumerate(spec.enums):
        comma = "," if i + 1 < len(spec.enums) else ""
        o.append(f"  {name} = {value}{comma}" + (f"  /* {doc} */" if doc else ""))
    o.append("};")
    o.append("")
    o.append("/* Every entry point returns one of the FUSE_OK / FUSE_ERR_* values. */")
    o.append("typedef int32_t fuse_status;")
    o.append("")
    o.append("/* ---- closed value types: no struct_size, and they never grow --------- */")
    o.append("")
    o.append("/* An exact rational time. Time never crosses this boundary as a double:")
    o.append(" * the coupler's schedule is exact, and a caller that rounds loses the")
    o.append(" * agreement on where a step begins and ends. */")
    o.append("typedef struct fuse_exact_time {")
    o.append("  int64_t numerator;")
    o.append("  int64_t denominator;  /* > 0 */")
    o.append("} fuse_exact_time;")
    o.append("")
    o.append("typedef struct fuse_semantic_version {")
    o.append("  uint32_t major;")
    o.append("  uint32_t minor;")
    o.append("  uint32_t patch;")
    o.append("} fuse_semantic_version;")
    o.append("")
    o.append("/* An opaque coupling context; create() makes one, destroy() ends it. */")
    o.append("typedef struct fuse_ctx fuse_ctx;")
    o.append("")
    o.append("/* ---- argument structs ------------------------------------------------- */")
    o.append("/* Every struct below begins with `uint32_t struct_size`, which the caller")
    o.append(" * sets to sizeof(the struct it was compiled against). Members are only ever")
    o.append(" * APPENDED; the callee reads those that struct_size says are present and")
    o.append(" * takes the documented default for the rest. A struct_size below the")
    o.append(" * oldest known layout, or above FUSE_ABI_MAX_STRUCT_SIZE, is")
    o.append(" * FUSE_ERR_VERSION. */")
    for sname, sdoc, fields in spec.structs:
        o.append("")
        o.append(_wrap_doc(sdoc, "").rstrip("\n"))
        o.append(f"typedef struct {sname} {{")
        width = max(len(f[0]) for f in fields) + 1
        for ctype, _, fname, fdoc in fields:
            decl = f"  {ctype:<{width}}{fname};"
            o.append(decl + (f"  /* {fdoc} */" if fdoc else ""))
        o.append(f"}} {sname};")
    o += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif /* FUSE_ABI_TYPES_H */", ""]
    return "\n".join(o)


def gen_participant(spec: Spec) -> str:
    o = [BANNER, "#ifndef FUSE_ABI_PARTICIPANT_H", "#define FUSE_ABI_PARTICIPANT_H", "",
         '#include "fuse/abi/types.h"', "", "#ifdef __cplusplus", 'extern "C" {', "#endif", ""]
    o.append("/* The participant ABI: a foreign code with its own time loop drives the")
    o.append(" * coupler through this table. Every member has the shape")
    o.append(" *     fuse_status f(fuse_ctx* ctx, const IN* in, OUT* out);")
    o.append(" * `in` and `out` are the argument structs of types.h; pass NULL where the")
    o.append(" * type is fuse_null. The table itself begins with struct_size: a caller")
    o.append(" * compares it with sizeof(fuse_participant_abi_v1) and uses only the")
    o.append(" * members that are present, so a v1 caller runs against a later library. */")
    o.append(f"typedef struct fuse_participant_abi_v{_abi_version(spec)} {{")
    o.append("  uint32_t struct_size;")
    o.append("  uint32_t abi_version;")
    for name, inp, out, doc in spec.entries:
        o.append("")
        if doc:
            o.append(_wrap_doc(doc, "  ").rstrip("\n"))
        o.append(f"  fuse_status (*{name})(fuse_ctx* ctx, const {inp}* in, {out}* out);")
    o.append(f"}} fuse_participant_abi_v{_abi_version(spec)};")
    o.append("")
    o.append("/* THE ONE EXPORTED SYMBOL. Returns the table for `abi_version`, or NULL")
    o.append(" * when this library does not speak that version. Static storage; never")
    o.append(" * freed. Thread-safe. (libfuse, the filesystem library, owns `fuse_*`; a")
    o.append(" * single unmistakable name is the whole collision surface -- see")
    o.append(" * docs/dev/abi-design.md.) */")
    o.append("const fuse_participant_abi_v1* fuse_coupler_abi(uint32_t abi_version);")
    o += ["", "#ifdef __cplusplus", "}", "#endif", "", "#endif /* FUSE_ABI_PARTICIPANT_H */", ""]
    return "\n".join(o)


def _abi_version(spec: Spec) -> str:
    for name, value, _ in spec.enums:
        if name == "FUSE_ABI_VERSION":
            return value
    sys.exit("FUSE_ABI_VERSION missing from entries.def")


# --------------------------------------------------------- Fortran: ffuse
#
# docs/dev/abi-design.md, "Fortran: `ffuse`", is the specification; the
# mapping table there is reproduced as FORTRAN_TYPE / FORTRAN_DEFAULT below so
# this file is the only place either is spelled out. Every member's `ftype`
# column in entries.def is already the Fortran kind the design calls for
# (c_int32_t, c_int64_t, c_ptr, fuse_semantic_version, fuse_exact_time); this generator
# only has to know how to spell and default-initialise each of those five.

FORTRAN_TYPE = {
    "c_int32_t": "integer(c_int32_t)",
    "c_int64_t": "integer(c_int64_t)",
    "c_ptr": "type(c_ptr)",
    "fuse_semantic_version": "type(fuse_semantic_version)",
    "fuse_exact_time": "type(fuse_exact_time)",
}
FORTRAN_DEFAULT = {
    "c_int32_t": "0_c_int32_t",
    "c_int64_t": "0_c_int64_t",
    "c_ptr": "c_null_ptr",
    "fuse_semantic_version": "fuse_semantic_version(0_c_int32_t, 0_c_int32_t, 0_c_int32_t)",
    "fuse_exact_time": "fuse_exact_time(0_c_int64_t, 1_c_int64_t)",
}

FORTRAN_BANNER = """! GENERATED by tools/gen-abi.py from include/fuse/abi/entries.def -- do not
! edit; edit the .def and run `python3 tools/gen-abi.py --write`. The sizes
! test test/abi/fortran/sizes_test.f90 fails when this module's C_SIZEOF of
! a type disagrees with test/abi/struct_sizes.golden. docs/dev/abi-design.md,
! "Fortran: ffuse", is the mapping this file implements.
"""

SHIM_BANNER = """/* GENERATED by tools/gen-abi.py from include/fuse/abi/entries.def -- do not
 * edit; edit the .def and run `python3 tools/gen-abi.py --write`. Forwards
 * each ffuse_<entry> to the participant ABI table (fuse_coupler_abi()), the
 * one deliberate exception to "one exported symbol" stated in
 * docs/dev/abi-design.md, "The costs": Fortran keeps flat bind(C) symbols
 * because a function-pointer struct is unergonomic there. Never exports a
 * fuse_* symbol -- only ffuse_*; test/abi/fortran/check_exports.cmake (ctest
 * abi.fortran_exports) greps the built library for that. */
"""


def _fortran_struct_decl(sname: str, fields: list[tuple[str, str, str, str]]) -> list[str]:
    o = [f"  type, bind(C), public :: {sname}"]
    for _, ftype, fname, _ in fields:
        o.append(f"    {FORTRAN_TYPE[ftype]} :: {fname} = {FORTRAN_DEFAULT[ftype]}")
    o.append(f"  end type {sname}")
    return o


def _fortran_entry_names(spec: Spec) -> dict[str, tuple[str, bool]]:
    """entry name -> (public/specific procedure name, whether it is also
    overloaded with a string convenience and so needs the `_struct` suffix).

    A Fortran module cannot declare a derived type and a procedure under the
    same name in one scope (unlike C's separate tag/ordinary namespaces), so
    an entry whose `fuse_<name>` collides with any type the module declares
    is REFUSED here: the fix is a different type name in entries.def (which
    is what happened on 2026-10-02 to fuse_capability_set, fuse_setup_options
    and fuse_semantic_version), never a mangled procedure name. See
    docs/dev/abi-design.md, "Fortran: ffuse".
    """
    struct_names = {s[0] for s in spec.structs} | {
        "fuse_semantic_version", "fuse_exact_time", "fuse_ctx"}
    out: dict[str, tuple[str, bool]] = {}
    for name, _, _, _ in spec.entries:
        base = f"fuse_{name}"
        if base in struct_names:
            sys.exit(f"entry '{name}': the Fortran procedure {base} would collide with the "
                     f"type {base}; rename the type in entries.def")
        overloaded = name in FORTRAN_STRING_ENTRIES
        out[name] = (f"{base}_struct" if overloaded else base, overloaded)
    return out


def _fortran_raw_interface(name: str, inp: str, out: str) -> list[str]:
    imports = sorted({"c_ptr", "c_int32_t", inp, out})
    o = [f"    function ffuse_{name}(ctx, in, out) result(status) bind(C, name=\"ffuse_{name}\")",
         f"      import :: {', '.join(imports)}",
         "      type(c_ptr), value :: ctx",
         f"      type({inp}), intent(in) :: in",
         f"      type({out}), intent(inout) :: out",
         "      integer(c_int32_t) :: status",
         f"    end function ffuse_{name}"]
    return o


def _fortran_struct_wrapper(name: str, inp: str, out: str, fn_name: str) -> list[str]:
    return [
        f"  function {fn_name}(ctx, in, out) result(status)",
        "    type(fuse_ctx), intent(in) :: ctx",
        f"    type({inp}), intent(inout) :: in",
        f"    type({out}), intent(inout) :: out",
        "    integer(c_int32_t) :: status",
        "",
        "    in%struct_size = int(c_sizeof(in), c_int32_t)",
        "    out%struct_size = int(c_sizeof(out), c_int32_t)",
        f"    status = ffuse_{name}(ctx%ptr, in, out)",
        f"  end function {fn_name}",
        "",
    ]


# Hand-specified convenience overloads for the five entries whose argument
# struct carries a NUL-terminated string (abi-design.md, "Strings"). Each
# entry is `(specific_name, extra_fortran_args, body_lines)`; `body_lines`
# builds the `in`/`out` struct used by the matching *_struct procedure.
def _fortran_convenience(name: str, inp: str, out: str) -> list[str]:
    o: list[str] = []
    if name == "load_graph":
        o += [
            "  function fuse_load_graph_path(ctx, yaml_path, out) result(status)",
            "    type(fuse_ctx), intent(in) :: ctx",
            "    character(len=*), intent(in) :: yaml_path",
            f"    type({out}), intent(inout) :: out",
            "    integer(c_int32_t) :: status",
            f"    type({inp}) :: in",
            "    character(kind=c_char, len=len_trim(yaml_path)+1), target :: path_buf",
            "",
            "    path_buf = trim(yaml_path) // c_null_char",
            "    in%yaml_path = c_loc(path_buf)",
            "    in%bytes = c_null_ptr",
            "    in%byte_count = 0_c_int64_t",
            "    status = fuse_load_graph_struct(ctx, in, out)",
            "  end function fuse_load_graph_path",
            "",
        ]
    elif name == "field_handle":
        o += [
            "  function fuse_field_handle_name(ctx, component, field, out) result(status)",
            "    type(fuse_ctx), intent(in) :: ctx",
            "    character(len=*), intent(in) :: component",
            "    character(len=*), intent(in) :: field",
            f"    type({out}), intent(inout) :: out",
            "    integer(c_int32_t) :: status",
            f"    type({inp}) :: in",
            "    character(kind=c_char, len=len_trim(component)+1), target :: component_buf",
            "    character(kind=c_char, len=len_trim(field)+1), target :: field_buf",
            "",
            "    component_buf = trim(component) // c_null_char",
            "    field_buf = trim(field) // c_null_char",
            "    in%component = c_loc(component_buf)",
            "    in%field = c_loc(field_buf)",
            "    status = fuse_field_handle_struct(ctx, in, out)",
            "  end function fuse_field_handle_name",
            "",
        ]
    elif name in ("begin_step", "end_step"):
        o += [
            f"  function fuse_{name}_name(ctx, component, out) result(status)",
            "    type(fuse_ctx), intent(in) :: ctx",
            "    character(len=*), intent(in) :: component",
            f"    type({out}), intent(inout) :: out",
            "    integer(c_int32_t) :: status",
            f"    type({inp}) :: in",
            "    character(kind=c_char, len=len_trim(component)+1), target :: component_buf",
            "",
            "    component_buf = trim(component) // c_null_char",
            "    in%component = c_loc(component_buf)",
            f"    status = fuse_{name}_struct(ctx, in, out)",
            f"  end function fuse_{name}_name",
            "",
        ]
    elif name == "last_error":
        o += [
            "  function fuse_last_error_message(ctx, message, truncated) result(status)",
            "    type(fuse_ctx), intent(in) :: ctx",
            "    character(len=*), intent(out) :: message",
            "    logical, intent(out), optional :: truncated",
            "    integer(c_int32_t) :: status",
            f"    type({inp}) :: in",
            f"    type({out}) :: out",
            "    character(kind=c_char, len=max(len(message),1)), target :: buf",
            "    integer(c_int64_t) :: n, i",
            "",
            "    buf = ' '",
            "    out%buffer = c_loc(buf)",
            "    out%capacity = int(len(message), c_int64_t)",
            "    status = fuse_last_error_struct(ctx, in, out)",
            "    n = min(out%length, out%capacity)",
            "    message = ' '",
            "    do i = 1, n",
            "      message(i:i) = buf(int(i):int(i))",
            "    end do",
            "    if (present(truncated)) truncated = (out%truncated /= 0)",
            "  end function fuse_last_error_message",
            "",
        ]
    return o


def gen_fortran_module(spec: Spec) -> str:
    struct_fields = {s[0]: s[2] for s in spec.structs}
    o = [FORTRAN_BANNER, "module ffuse", "  use, intrinsic :: iso_c_binding", "  implicit none",
         "  private", ""]

    o.append("  ! ---- status codes, enumerators and limits (entries.def FUSE_ABI_ENUM) ----")
    for enm_name, value, _ in spec.enums:
        o.append(f"  integer(c_int32_t), parameter, public :: {enm_name} = {value}_c_int32_t")
    o.append("")

    o.append("  ! ---- closed value types: no struct_size, and they never grow ----")
    o.append("  type, bind(C), public :: fuse_exact_time")
    o.append("    integer(c_int64_t) :: numerator = 0_c_int64_t")
    o.append("    integer(c_int64_t) :: denominator = 1_c_int64_t")
    o.append("  end type fuse_exact_time")
    o.append("")
    o.append("  type, bind(C), public :: fuse_semantic_version")
    o.append("    integer(c_int32_t) :: major = 0_c_int32_t")
    o.append("    integer(c_int32_t) :: minor = 0_c_int32_t")
    o.append("    integer(c_int32_t) :: patch = 0_c_int32_t")
    o.append("  end type fuse_semantic_version")
    o.append("")
    o.append("  ! An opaque coupling context; wraps the one C_PTR fuse_ctx* resolves to")
    o.append("  ! (abi-design.md, \"Fortran: ffuse\"). create() fills %ptr; destroy() ends it.")
    o.append("  type, public :: fuse_ctx")
    o.append("    type(c_ptr) :: ptr = c_null_ptr")
    o.append("  end type fuse_ctx")
    o.append("")

    o.append("  ! ---- argument structs (entries.def FUSE_ABI_STRUCT), members in the same")
    o.append("  ! order as include/fuse/abi/types.h; C_SIZEOF of each is pinned by")
    o.append("  ! test/abi/struct_sizes.golden via test/abi/fortran/sizes_test.f90.")
    for sname, _, fields in spec.structs:
        o += _fortran_struct_decl(sname, fields)
        o.append("")

    o.append("  ! ---- raw bind(C) entry points, exported by libfuse-abi-fortran as")
    o.append("  ! ffuse_<name>; never fuse_<name> (abi-design.md, \"Link names\").")
    o.append("  interface")
    for i, (name, inp, out, _) in enumerate(spec.entries):
        if i:
            o.append("")
        o += _fortran_raw_interface(name, inp, out)
    o.append("  end interface")
    o.append("")

    entry_names = _fortran_entry_names(spec)

    public_names = []
    for name, _, _, _ in spec.entries:
        fn_name, overloaded = entry_names[name]
        # The public-facing name is the generic interface fuse_<name> when
        # overloaded with a string convenience, otherwise fn_name itself
        # (which is fuse_<name>, or fuse_<name>_entry for the three entries
        # that collide with a struct of the same name -- see
        # _fortran_entry_names).
        public_names.append(f"fuse_{name}" if overloaded else fn_name)
    o.append("  public :: " + ", &\n            ".join(public_names))
    o.append("")

    for name, _, _, _ in spec.entries:
        fn_name, overloaded = entry_names[name]
        if overloaded:
            specifics = [fn_name]
            if name == "load_graph":
                specifics.append("fuse_load_graph_path")
            elif name == "field_handle":
                specifics.append("fuse_field_handle_name")
            elif name in ("begin_step", "end_step"):
                specifics.append(f"fuse_{name}_name")
            elif name == "last_error":
                specifics.append("fuse_last_error_message")
            o.append(f"  interface fuse_{name}")
            o.append("    module procedure " + ", ".join(specifics))
            o.append(f"  end interface fuse_{name}")
            o.append("")

    o.append("contains")
    o.append("")
    for name, inp, out, _ in spec.entries:
        fn_name, overloaded = entry_names[name]
        o += _fortran_struct_wrapper(name, inp, out, fn_name)
        if overloaded:
            o += _fortran_convenience(name, inp, out)

    o.append("end module ffuse")
    text = "\n".join(o)
    if not text.endswith("\n"):
        text += "\n"
    return text


def gen_fortran_shim(spec: Spec) -> str:
    o = [SHIM_BANNER, '#include "fuse/abi/participant.h"', "", "#include <pthread.h>", "",
         "#if defined(__GNUC__) || defined(__clang__)",
         '#define FUSE_FORTRAN_EXPORT __attribute__((visibility("default")))',
         "#else",
         "#define FUSE_FORTRAN_EXPORT",
         "#endif",
         "",
         "/* Thread-safe once-init (abi-design.md: \"one extra line to fetch the table",
         " * before first use\"); every ffuse_* entry point shares this one lookup. */",
         "static const fuse_participant_abi_v1* g_fuse_fortran_table;",
         "",
         "static void fuse_fortran_shim_init(void) {",
         "  g_fuse_fortran_table = fuse_coupler_abi(FUSE_ABI_VERSION);",
         "}",
         "",
         "static const fuse_participant_abi_v1* fuse_fortran_shim_table(void) {",
         "  static pthread_once_t once = PTHREAD_ONCE_INIT;",
         "  pthread_once(&once, fuse_fortran_shim_init);",
         "  return g_fuse_fortran_table;",
         "}",
         ""]
    for name, inp, out, _ in spec.entries:
        o.append(f"FUSE_FORTRAN_EXPORT fuse_status ffuse_{name}(fuse_ctx* ctx, const {inp}* in, {out}* out) {{")
        o.append("  const fuse_participant_abi_v1* table = fuse_fortran_shim_table();")
        o.append(f"  if (!table || !table->{name}) return FUSE_ERR_UNSUPPORTED;")
        o.append(f"  return table->{name}(ctx, in, out);")
        o.append("}")
        o.append("")
    text = "\n".join(o)
    if not text.endswith("\n"):
        text += "\n"
    return text


OUTPUTS = {
    OUT_TYPES: gen_types,
    OUT_PARTICIPANT: gen_participant,
    OUT_FORTRAN_MODULE: gen_fortran_module,
    OUT_FORTRAN_SHIM: gen_fortran_shim,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--write", action="store_true")
    g.add_argument("--print", choices=["types", "participant", "fortran-module", "fortran-shim"])
    a = ap.parse_args()
    spec = parse()
    if a.print:
        printers = {
            "types": gen_types,
            "participant": gen_participant,
            "fortran-module": gen_fortran_module,
            "fortran-shim": gen_fortran_shim,
        }
        sys.stdout.write(printers[a.print](spec))
        return 0
    stale = 0
    for path, gen in OUTPUTS.items():
        text = gen(spec)
        if a.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)}")
        elif not path.exists() or path.read_text() != text:
            print(f"STALE {path.relative_to(ROOT)}: differs from entries.def")
            stale += 1
    if a.check:
        print(f"gen-abi: {len(OUTPUTS) - stale} current, {stale} stale, of {len(OUTPUTS)}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
