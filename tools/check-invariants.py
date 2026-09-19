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
# once to MpiTransport, which vanished from an MPI build.
# `//` to end of line, capturing the prefix so it can be preserved. Block
# comments are deliberately not handled: this project's prose is `///` lines,
# and a regex for /* */ that got it wrong could blank a real `#if`.
COMMENT_LINE = re.compile(r"(^|[^:])(//[^\n]*)", re.M)

FEATURE_MACRO = re.compile(r"\bFUSE_(HAVE_[A-Z0-9_]+|ENABLE_EXCEPTIONS)\b")
PRINTF = re.compile(r"\b(printf|fprintf|sprintf|snprintf|puts|fputs)\s*\(")

# Device code has no fmtlog and no iostream; Kokkos::printf is the sanctioned
# exception, and a literal `printf` inside a KOKKOS_ lambda is usually that.
DEVICE_HINT = re.compile(r"KOKKOS_(INLINE_)?FUNCTION|KOKKOS_LAMBDA")


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

    for area in ("include", "src"):
        area_dir = root / area
        if not area_dir.is_dir():
            continue
        for source in sorted(area_dir.rglob("*")):
            if source.suffix not in {".hpp", ".h", ".cpp", ".cc", ".C"}:
                continue
            rel = source.relative_to(root)
            text = source.read_text()
            device_file = bool(DEVICE_HINT.search(text))
            for lineno, line in enumerate(text.splitlines(), 1):
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
                    # dumps are byte-compared golden artifacts.
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
                cfg = scanned.find('#include "fuse/base/config.hpp"')
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
