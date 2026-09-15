#!/usr/bin/env python3
"""Prove that `upstream/` is byte-identical to the tagged parent trees (invariant I8).

Two levels of check:

  manifest  sha256 of every imported file against upstream/MANIFEST.sha256.
            Needs nothing but this repository, so it runs in CI.
  parents   re-derive each tree with `git archive <tag>` from the parent repository
            and compare. This is the real proof; it runs where the parents exist.

  --write   regenerate the manifest (use only when performing an import).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / "upstream"
MANIFEST = UPSTREAM / "MANIFEST.sha256"
# Files that live under upstream/ but are OURS, not imported: the format and lint
# opt-outs, this directory's own readme, and the manifest itself. Matched against
# the path RELATIVE TO upstream/, so only these top-level entries are skipped --
# an imported tree's own README.md is several directories down and still checked.
SKIP = {".clang-format", ".clang-tidy", "MANIFEST.sha256", "README.md"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def imported_files() -> list[Path]:
    out = []
    for p in sorted(UPSTREAM.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        if str(p.relative_to(UPSTREAM)) in SKIP:   # top-level entries only
            continue
        out.append(p)
    return out


def read_lock() -> dict[str, dict[str, str]]:
    """Parse the [name] path/ref/tag/sha fields out of UPSTREAMS.lock."""
    text = (ROOT / "UPSTREAMS.lock").read_text()
    parents: dict[str, dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        sec = re.match(r"^\[([a-z0-9_]+)\]", line)
        if sec:
            current = sec.group(1)
            parents[current] = {}
            continue
        kv = re.match(r"^\s*(path|ref|sha|tag)\s*=\s*\"([^\"]+)\"", line)
        if kv and current:
            parents[current][kv.group(1)] = kv.group(2)
    return parents


def check_manifest() -> int:
    if not MANIFEST.exists():
        print("error: upstream/MANIFEST.sha256 is missing")
        return 1
    expected = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split("  ", 1)
            expected[rel] = h
    actual = {str(p.relative_to(ROOT)): digest(p) for p in imported_files()}

    problems = []
    for rel, h in sorted(expected.items()):
        if rel not in actual:
            problems.append(f"{rel}: in manifest, missing from tree")
        elif actual[rel] != h:
            problems.append(f"{rel}: content differs from manifest (I8 violation)")
    for rel in sorted(set(actual) - set(expected)):
        problems.append(f"{rel}: present in tree, absent from manifest")

    for p in problems:
        print(f"error: {p}")
    print(f"verify-import[manifest]: {len(expected)} files, {len(problems)} problem(s)")
    return 1 if problems else 0


def check_parents() -> int:
    problems, checked = [], 0
    for name, info in read_lock().items():
        src = Path(info.get("path", ""))
        tag = info.get("tag", "")
        dest = UPSTREAM / name
        if not (src / ".git").exists():
            print(f"verify-import[parents]: {name}: parent not reachable at {src}, skipped")
            continue
        try:
            blob = subprocess.run(["git", "-C", str(src), "archive", tag],
                                  capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            problems.append(f"{name}: `git archive {tag}` failed in {src}")
            continue
        import io
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                checked += 1
                here = dest / member.name
                if not here.exists():
                    problems.append(f"{name}/{member.name}: missing from the import")
                    continue
                want = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                if digest(here) != want:
                    problems.append(f"{name}/{member.name}: differs from {tag} (I8 violation)")
    for p in problems:
        print(f"error: {p}")
    print(f"verify-import[parents]: {checked} files compared, {len(problems)} problem(s)")
    return 1 if problems else 0


def write_manifest() -> int:
    lock = read_lock()
    lines = [
        "# sha256 of every file imported under upstream/, one per line.",
        "# Regenerate only when performing an import: tools/verify-import.py --write",
    ]
    for name, info in sorted(lock.items()):
        lines.append(f"# {name}: {info.get('ref','?')} @ {info.get('sha','?')} tag {info.get('tag','?')}")
    files = imported_files()
    for p in files:
        lines.append(f"{digest(p)}  {p.relative_to(ROOT)}")
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"verify-import: wrote manifest for {len(files)} files")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="regenerate the manifest")
    ap.add_argument("--parents", action="store_true", help="also compare against the parent repos")
    args = ap.parse_args()
    if args.write:
        return write_manifest()
    rc = check_manifest()
    if args.parents:
        rc |= check_parents()
    return rc


if __name__ == "__main__":
    sys.exit(main())
