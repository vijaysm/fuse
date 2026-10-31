#!/usr/bin/env python3
"""Refuse a KARMA* type prefix, a karma_ file prefix or a .C extension under
physics-kit/.

docs/dev/porting.md's naming map states these three rules for ported code in
general ("The parents' ... type prefixes (KARMA*), their file prefixes
(karma_*, soma_*) and their source extension (.C) do not survive the port.")
and tools/check-layering.py's check_legacy() already enforces them under
include/fuse/, src/, test/ and examples/ -- fuse proper. physics-kit/ is
none of those four directories (docs/dev/layering.md: "physics-kit/ is a
separate optional component"), so nothing there was ever checked for this
until now, and physics-kit/ is exactly where a KARMA-derived file is going
to land first.

Not folded into tools/check-layering.py itself: that script's check_legacy()
is already scoped to the four fuse-proper directories by a loop over a fixed
tuple, and physics-kit is deliberately NOT a fifth member of that tuple --
check-layering.py's whole physics-kit section exists to check the OPPOSITE
direction (fuse proper must not reach INTO physics-kit); this gate checks
physics-kit's OWN naming, a separate concern reusing the same three
upstream-identity regexes verify-import.py's sibling gate would otherwise
have needed to grow a second, unrelated job to cover (verify-import.py's own
job is upstream/'s byte-identity against UPSTREAM.lock, nothing to do with
naming).

Run from the repository root. Exits non-zero with a report of every
violation, never on the first one -- the same accumulate-and-report
convention as tools/check-layering.py and tools/check-invariants.py.

Proof this fires: a probe header (KARMAProbeType, under physics-kit/include/
fuse_physics/karma_probe.h) and its copy at physics-kit/src/karma_probe.C
produced five violation lines (file prefix x2, type prefix x2, extension x1)
here, then were removed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# --- rules
# Same three regexes check-layering.py's check_legacy() applies to fuse
# proper (include/fuse/, src/, test/, examples/), applied here to
# physics-kit/ instead. KARMA_* macros and the soma/KARMA namespaces are
# NOT checked here: the task that created this gate names exactly "any
# KARMA type prefix, any karma_ file prefix and any .C extension" -- the
# three rows of docs/dev/porting.md's naming map a stray copy-paste from
# upstream/karma would trip first (a type name, a filename, a source
# extension), not a macro or namespace rename, which a compile already
# forces to be dealt with.
LEGACY_TYPE_PREFIX = re.compile(r"\bKARMA[A-Z][A-Za-z0-9]*\b")
LEGACY_FILE_PREFIX = re.compile(r"^karma_")
LEGACY_EXTENSION = {".C"}
# `//` to end of line, capturing the prefix so it can be preserved -- the
# same blanking check-layering.py's own COMMENT_LINE does, and for the same
# reason: a prose comment NAMING "KARMA*" (a port's rename changelog, or
# this gate's own probe-proof record) is not a USE of it.
COMMENT_LINE = re.compile(r"(^|[^:])(//[^\n]*)")
# --- end rules

CHECKED_SUFFIXES = {".hpp", ".h", ".cpp", ".cc", ".C", ".inl", ".ipp", ".txt", ".cmake"}


def check_physics_kit_naming(root: Path) -> list[str]:
    problems: list[str] = []
    physics_kit_dir = root / "physics-kit"
    if not physics_kit_dir.is_dir():
        print("check-physics-kit-naming: no physics-kit/ yet, nothing to check")
        return problems
    for f in sorted(physics_kit_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(root)
        if f.suffix in LEGACY_EXTENSION:
            problems.append(f"{rel}: '.C' extension; physics-kit sources are '.cpp' "
                             f"(docs/dev/porting.md)")
        if LEGACY_FILE_PREFIX.match(f.name):
            problems.append(f"{rel}: legacy 'karma_' file prefix; rename per "
                             f"docs/dev/porting.md")
        if f.suffix not in CHECKED_SUFFIXES:
            continue
        for lineno, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            code_only = COMMENT_LINE.sub(
                lambda m: m.group(1) + " " * len(m.group(2)), line)
            if LEGACY_TYPE_PREFIX.search(code_only):
                problems.append(f"{rel}:{lineno}: 'KARMA*' type prefix; the "
                                 f"fuse::physics namespace carries it (docs/dev/porting.md)")
    return problems


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    problems = check_physics_kit_naming(root)
    for p in problems:
        print(f"error: {p}")
    print(f"check-physics-kit-naming: {len(problems)} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
