#!/usr/bin/env python3
"""Enforce the invariants that a grep can check: I1, I6, and the output rule.

I1  no floating-point time crosses an ABI
I6  `assert` is never an invariant
--  `fmt` is the only formatter: no iostream, no printf in library code

Accumulates and reports everything, then exits non-zero.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ABI_TIME = re.compile(r"\b(double|float)\b[^;,)]*\b\w*(time|dt|step_size|interval|when)\w*\b",
                      re.IGNORECASE)
ASSERT_CALL = re.compile(r"^\s*assert\s*\(")
# A bare `throw std::…` bypasses the on-error policy and carries no backtrace.
BARE_THROW = re.compile(r"\bthrow\s+std::[a-z_]+\s*[({]")
IOSTREAM = re.compile(r"^\s*#\s*include\s*<(iostream|sstream|iomanip)>")
# A build-feature macro read without including the generated config header.
# config.hpp.in uses #cmakedefine01, so the macro is either 1, 0, or -- if the
# header is missing -- UNDEFINED, which the preprocessor reads as 0. The file
# then compiles as though the feature were off, in a build where it is on, and
# nothing warns. This has now happened twice: once to the exceptions option and
# once to MpiTransport, which vanished from an MPI build. FUSE_FP_REPRODUCIBLE
# (cmake/FuseFpPolicy.cmake) is the same kind of macro: a test reading it without
# the header would silently take the OFF branch in a build where it is ON.
# `//` to end of line, capturing the prefix so it can be preserved. Block
# comments are deliberately not handled: this project's prose is `///` lines,
# and a regex for /* */ that got it wrong could blank a real `#if`.
COMMENT_LINE = re.compile(r"(^|[^:])(//[^\n]*)", re.M)

# Only the macros config.hpp.in actually `#cmakedefine01`s belong here: this
# alternation also decides which files must include "fuse/base/config.hpp"
# before using the name (the check below), and FUSE_CONNECT_ACCEPT_ENABLED
# (set directly by target_compile_definitions, not by config.hpp.in) does not
# owe that header anything. Widening this one to a bare `ENABLE_[A-Z0-9_]+`
# was tried and reverted: error messages such as "-DFUSE_ENABLE_MPI=ON" name
# the CMake *option*, not a macro, in plain string literals, and a blanket
# ENABLE_ alternation reads those strings as a macro use and demands the
# config.hpp include that message has no reason to carry.
FEATURE_MACRO = re.compile(
    r"\bFUSE_(HAVE_[A-Z0-9_]+|ENABLE_EXCEPTIONS|LINALG_EIGEN|FP_REPRODUCIBLE)\b")
# Every feature macro is `#cmakedefine01`: ALWAYS defined, as 0 or 1. So a
# presence test on one is unconditionally true, and `#ifdef FUSE_HAVE_MOAB`
# compiles the MOAB branch into a build without MOAB -- or, when config.hpp is
# not included, compiles it OUT of every build. Both happened (REVIEW.md 2.3).
# Only `#if FUSE_HAVE_X` / `#if !FUSE_HAVE_X` is a valid test.
#
# This alternation is deliberately wider than FEATURE_MACRO above: it also
# covers FUSE_CONNECT_*_ENABLED (FUSE_CONNECT_ACCEPT_ENABLED today) and
# FUSE_CONNECT_MPMD, which reach the translation unit as an always-defined
# 0/1 the same way a `#cmakedefine01` macro does (here, via
# target_compile_definitions(... PUBLIC NAME=$<BOOL:...>)) -- a presence test
# on them is exactly as unconditionally-true-and-wrong. Checked against the
# tree (2026-10-11): no other `FUSE_ENABLE_*`/`FUSE_HAVE_*` name is ever
# tested with `#ifdef`/`#ifndef`/`defined()` outside the one exempted guard
# below, so broadening this (ifdef-misuse) alternation to a bare
# `ENABLE_[A-Z0-9_]+` carries none of FEATURE_MACRO's string-literal risk.
FEATURE_PRESENCE_TEST = re.compile(
    r"#\s*(ifdef|ifndef)\s+FUSE_(HAVE_[A-Z0-9_]+|ENABLE_[A-Z0-9_]+|CONNECT_[A-Z0-9_]+_ENABLED|CONNECT_MPMD|LINALG_EIGEN|FP_REPRODUCIBLE)\b"
    r"|\bdefined\s*\(?\s*FUSE_(HAVE_[A-Z0-9_]+|ENABLE_[A-Z0-9_]+|CONNECT_[A-Z0-9_]+_ENABLED|CONNECT_MPMD|LINALG_EIGEN|FP_REPRODUCIBLE)\b")
# `#if defined(X) && X` already tests the value, not just presence -- the one
# shape this gate must not flag, because it is not the bug the rule exists
# for (connect.hpp's kConnectAcceptEnabled, both call sites of it).
FEATURE_VALUE_GUARD = re.compile(r"#\s*if\s+defined\s*\(\s*(FUSE_\w+)\s*\)\s*&&\s*\1\b")
# A macro from either parent. The layering gate forbids parent NAMESPACES and
# includes; a parent MACRO read in an #if is just as live and was not caught
# (SOMA_ENABLE_LOWFIVE compiled the LowFive transport out of every build).
LEGACY_MACRO = re.compile(r"(?<![a-z0-9_])D?(SOMA|KARMA)_[A-Z][A-Z0-9_]*\b")
PRINTF = re.compile(r"\b(printf|fprintf|sprintf|snprintf|puts|fputs)\s*\(")

# Device code has no fmtlog and no iostream; Kokkos::printf is the sanctioned
# exception, and a literal `printf` inside a KOKKOS_ lambda is usually that.
DEVICE_HINT = re.compile(r"KOKKOS_(INLINE_)?FUNCTION|KOKKOS_LAMBDA")


# I-VERDICT. A deferred outcome must never be cleared. `MpiTransport` defers
# every local rejection into one `err`, reached by `fail()`, and turns it into a
# return only at a synchronisation point where every rank is present. Phase D used
# to clear `err` so it could reuse it as its own loop accumulator, which was
# harmless while every outcome before it had already been agreed -- and then
# silently destroyed the components agreement (R-30), an outcome recorded after
# that agreement and read only two phases later. The check ran, was right on
# every rank, and its answer was thrown away; `build_plan()` returned success
# on ranks that had just rejected, and it took an instrumented run to see it.
#
# Declaration and reset look almost identical, so the rule keys off the `;`
# terminator only being a reset when the line does not also declare the type.
VERDICT_RESET = re.compile(
    r"^\s*(?:err|verdict)\s*=\s*(?:Err|Verdict)::None\s*;")


def scan(root: Path) -> list[str]:
    problems: list[str] = []

    abi_dir = root / "include" / "fuse" / "abi"
    if abi_dir.is_dir():
        for header in sorted(abi_dir.rglob("*.h")):
            rel = header.relative_to(root)
            for lineno, line in enumerate(header.read_text().splitlines(), 1):
                if ABI_TIME.search(line):
                    problems.append(
                        f"{rel}:{lineno}: floating-point time in an ABI header (I1): {line.strip()}"
                    )

    # `include` and `src` carry every rule. `test` and `examples` carry only the
    # feature-macro rules: a test may legitimately throw a std:: type in a
    # fixture, but a test that reads FUSE_HAVE_MOAB without config.hpp compiles
    # to a skip stub in every configuration and reads like a pass (REVIEW.md
    # R-06: 38 MOAB cases, never once executed). Found 2026-09-19 -- the same
    # scope gap the layering gate closed on 09-17, in the other tool.
    #
    # `tools/src` joined this list 2026-10-12: its two generators link
    # fuse::model/fuse::num and could misread a feature macro exactly as a
    # test or example can, and nothing scanned them before. They are not
    # `library_code`, on purpose and symmetrically with test/examples: they
    # are one-shot golden/baseline writers invoked by hand or by a test
    # fixture, not code a consumer links against, and their existing
    # `std::fprintf`/`<iostream>` diagnostic output (usage messages, a
    # written-file confirmation) is the same kind of harness-facing text
    # test/examples are already exempt for -- rewriting it through fmt would
    # buy this gate nothing it does not already enforce in library code.
    for area in ("include", "src", "test", "examples", "tools/src"):
        library_code = area in ("include", "src")
        area_dir = root / area
        if not area_dir.is_dir():
            continue
        for source in sorted(area_dir.rglob("*")):
            if source.suffix not in {".hpp", ".h", ".cpp", ".cc", ".C"}:
                continue
            rel = source.relative_to(root)
            text = source.read_text()
            device_file = bool(DEVICE_HINT.search(text))
            all_lines = text.splitlines()
            for lineno, line in enumerate(all_lines, 1):
                code_only = COMMENT_LINE.sub(
                    lambda m: m.group(1) + " " * len(m.group(2)), line)
                # `#ifndef FUSE_X` followed by `#error` is the guard that the
                # macro IS defined (error.hpp does this for the exceptions
                # macro). That is the only presence test with a purpose.
                nxt = next((l for l in all_lines[lineno:lineno + 2] if l.strip()), "")
                is_guard = bool(re.match(r"\s*#\s*ifndef\b", code_only)) and nxt.lstrip().startswith("#error")
                # `#if defined(X) && X` reads as a presence test but is not
                # one: the `&& X` already makes the branch depend on the
                # VALUE, which is all this rule ever asks for (connect.hpp's
                # kConnectAcceptEnabled, both call sites).
                is_guard = is_guard or bool(FEATURE_VALUE_GUARD.search(code_only))
                if library_code and VERDICT_RESET.match(code_only):
                    problems.append(
                        f"{rel}:{lineno}: clears a deferred outcome (I-VERDICT); "
                        "it is one monotonic value per call, agreed at a "
                        "collective -- use a separate accumulator: "
                        f"{line.strip()}"
                    )
                if FEATURE_PRESENCE_TEST.search(code_only) and not is_guard:
                    problems.append(
                        f"{rel}:{lineno}: presence test on a feature macro (#ifdef / "
                        f"defined()); every FUSE_HAVE_* is #cmakedefine01 and always "
                        f"defined, so this is unconditionally true -- use `#if MACRO`")
                if LEGACY_MACRO.search(code_only) and not rel.as_posix().startswith("include/fuse/abi/"):
                    problems.append(
                        f"{rel}:{lineno}: parent macro in fuse:: code (I11): "
                        f"{LEGACY_MACRO.search(code_only).group(0)} -- nothing defines it here, "
                        f"so the guarded code is silently compiled out")
                if not library_code:
                    continue
                if ASSERT_CALL.match(line):
                    problems.append(
                        f"{rel}:{lineno}: `assert` is never an invariant (I6); "
                        f"throw or return a status"
                    )
                if BARE_THROW.search(line):
                    problems.append(f"{rel}:{lineno}: bare `throw std::…`; use FUSE_THROW so the "
                                    f"on-error policy runs and a backtrace is captured")
                if IOSTREAM.match(line):
                    # The rule's PURPOSE is that no library output depends on the
                    # host's locale: a stream built from the global locale prints
                    # "0,7" under de_DE and groups digits, and the IR and plan
                    # dumps are byte-compared reference-output artifacts.
                    #
                    # `fmt` gives that by construction, so it stays the default.
                    # A canonical-text writer that instead pins the classic locale
                    # on every stream it creates has the same property, and
                    # verifying the pin is a STRONGER check than banning the
                    # header -- banning it would have been satisfied by a file
                    # that formats through `std::to_string`, which reads the
                    # global locale and would have passed silently.
                    pinned = text.count("imbue(std::locale::classic())")
                    streams = len(re.findall(r"std::(o|i)stringstream\s+\w+", text))
                    if pinned == 0 or pinned < streams:
                        problems.append(
                            f"{rel}:{lineno}: iostream in library code; use fmt, or pin "
                            f"std::locale::classic() on every stream "
                            f"({streams} stream(s) declared, {pinned} pinned)")
                # The failure path may not throw or allocate, so it writes with fputs and
                # says so on the line; the marker is the review trail, not a loophole.
                raw_ok = "NOLINT(fuse-raw-io)" in line
                # Comments are blanked first, for the same reason the
                # feature-macro check below does it: naming `fprintf` in prose
                # is not a call to it. A comment explaining why a third-party
                # library's own fprintf cannot be redirected is exactly the
                # kind of note this project wants, and a gate that fires on it
                # pressures the author to delete the explanation rather than
                # fix any code. Found 2026-09-18, when the check fired on a
                # doc comment in the ported high-order backend while the
                # feature-macro check on the same file did not.
                code = COMMENT_LINE.sub(
                    lambda m: m.group(1) + " " * len(m.group(2)), line)
                if PRINTF.search(code) and "Kokkos::printf" not in code and not device_file and not raw_ok:
                    problems.append(f"{rel}:{lineno}: printf in library code; use fmt")

            # Whole-file check: reading a build-feature macro requires the
            # generated header that defines it.
            # Comments are blanked first, keeping the byte offsets so the
            # reported line numbers stay right. A build-feature macro NAMED IN
            # PROSE is not a read of it, and a header's leading doc comment
            # necessarily precedes its includes -- so without this the gate
            # fires on any file that documents the macro, and the pressure
            # becomes to reword the documentation rather than fix the code.
            # Only `//` line comments are stripped: that is where this
            # project's prose lives, and a narrow rule that is obviously
            # correct beats a general one that might blank a real directive.
            scanned = COMMENT_LINE.sub(
                lambda m: m.group(1) + " " * len(m.group(2)), text)
            first_use = FEATURE_MACRO.search(scanned)
            if first_use and not rel.as_posix().endswith("base/config.hpp.in"):
                names = sorted(set(m.group(0) for m in FEATURE_MACRO.finditer(scanned)))
                cfg = min([i for i in (scanned.find('#include "fuse/base/config.hpp"'),
                                       scanned.find('#include <fuse/base/config.hpp>')) if i >= 0],
                          default=-1)
                if cfg < 0:
                    problems.append(
                        f"{rel}: reads {', '.join(names)} without including "
                        f'"fuse/base/config.hpp"; the macro is then undefined, which '
                        f"the preprocessor reads as 0, so the file compiles as though "
                        f"the feature were off")
                elif cfg > first_use.start():
                    # Presence is not enough: an #if that reads the macro before
                    # the definition arrives is compiled out, and the include
                    # further down makes the file LOOK correct.
                    line = scanned.count("\n", 0, first_use.start()) + 1
                    problems.append(
                        f"{rel}:{line}: reads {first_use.group(0)} before including "
                        f'"fuse/base/config.hpp" (line '
                        f'{text.count(chr(10), 0, cfg) + 1}); move the include above '
                        f"the first use")
    return problems


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    problems = scan(root)
    for p in problems:
        print(f"error: {p}")
    print(f"check-invariants: {len(problems)} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
