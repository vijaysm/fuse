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
                if PRINTF.search(line) and "Kokkos::printf" not in line and not device_file and not raw_ok:
                    problems.append(f"{rel}:{lineno}: printf in library code; use fmt")
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
