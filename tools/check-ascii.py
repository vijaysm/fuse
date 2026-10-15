#!/usr/bin/env python3
"""Refuse any tracked source file that is not pure ASCII.

Owner's ruling (2026-10-15): no non-ASCII code point anywhere in code, comments,
docs, commit messages. Markdown prose is explicitly exempted from this gate by
the same ruling -- *.md is still allowed to carry Unicode (section signs, an em
dash, author names) and is reviewed by eye, not by this tool.

Walks `git ls-files` (or an explicit file list passed on the command line),
applies the exclusions below, and for every remaining file either:
  - decodes it as UTF-8 and reports every code point above U+007F as a
    violation, with file:line:col and the Unicode character name, or
  - if it is not valid UTF-8 at all, reports the decode failure as a
    violation UNLESS the path matches one of the documented binary-fixture
    globs below -- an undecodable file that is not on that list is far more
    likely to be a newly-added fixture nobody told this gate about than a
    deliberate binary artifact, so the default is to refuse, not to guess.

Exit 0 on a clean tree; exit 1 and print every violation otherwise (the same
accumulate-and-report convention as tools/check-layering.py and
tools/check-invariants.py).
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
import unicodedata
from pathlib import Path

# Each entry is (glob pattern against the repo-relative POSIX path, reason).
# fnmatch patterns: "*" does not cross "/" by default in fnmatch.fnmatch, so
# "upstream/*" would only match one level deep -- these use the directory
# PREFIX test below instead, and fnmatch only for filename-shaped globs.
EXCLUDE_DIR_PREFIXES = [
    ("upstream/", "vendored parent trees staged for removal (docs/dev/soma-import.md); "
                   "not edited"),
]

EXCLUDE_SUFFIXES = [
    (".md", "doc prose; owner's 2026-10-15 ruling excludes Markdown from this gate"),
]

# Binary fixtures: captured raw bytes (an IR accumulator snapshot, a packed
# descriptor), not text, so "not valid UTF-8" is their normal, correct state.
# Listed as globs against the repo-relative POSIX path. Tested: a .bin here is
# not reported; an undecodable file that matches none of these IS reported,
# so a newly added binary fixture must be added here deliberately.
BINARY_FIXTURE_GLOBS = [
    "test/golden/*.bin",
    "python/tests/golden/*.bin",
]


def excluded(rel_posix: str) -> bool:
    for prefix, _reason in EXCLUDE_DIR_PREFIXES:
        if rel_posix.startswith(prefix):
            return True
    for suffix, _reason in EXCLUDE_SUFFIXES:
        if rel_posix.endswith(suffix):
            return True
    return False


def is_declared_binary_fixture(rel_posix: str) -> bool:
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in BINARY_FIXTURE_GLOBS)


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=root, check=True,
                          stdout=subprocess.PIPE, text=True).stdout
    return [root / line for line in out.splitlines() if line]


def display_path(path: Path, root: Path) -> str:
    """Repo-relative POSIX path when possible, else the path as given.

    The default invocation (no arguments) always resolves under `root`
    because the candidates come from `git ls-files` run in `root`. The
    explicit-file-list form accepts paths outside the tree too (a gate test
    exercising a scratch fixture that is deliberately not part of the
    repository), so this falls back to the given path rather than raising.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def scan_file(path: Path, rel_posix: str) -> list[str]:
    problems: list[str] = []
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if is_declared_binary_fixture(rel_posix):
            return []
        problems.append(
            f"{rel_posix}: not valid UTF-8 (decode failed at byte {exc.start}: "
            f"{exc.reason}); not on the binary-fixture list in "
            f"tools/check-ascii.py -- add it there if it really is binary, "
            f"otherwise it is corrupted text that needs fixing")
        return problems
    line = 1
    col = 0
    for ch in text:
        col += 1
        if ch == "\n":
            line += 1
            col = 0
            continue
        if ord(ch) > 0x7F:
            name = unicodedata.name(ch, f"UNNAMED-U+{ord(ch):04X}")
            problems.append(
                f"{rel_posix}:{line}:{col}: non-ASCII code point U+{ord(ch):04X} ({name})")
    return problems


def scan(root: Path, files: list[Path] | None = None) -> list[str]:
    problems: list[str] = []
    candidates = files if files is not None else tracked_files(root)
    for path in sorted(candidates):
        if not path.is_file():
            continue
        rel_posix = display_path(path, root)
        if excluded(rel_posix):
            continue
        problems.extend(scan_file(path, rel_posix))
    return problems


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    if len(argv) > 1:
        files = [Path(a).resolve() for a in argv[1:]]
    else:
        files = None
    problems = scan(root, files)
    for p in problems:
        print(f"error: {p}")
    print(f"check-ascii: {len(problems)} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
