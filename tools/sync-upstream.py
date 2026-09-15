#!/usr/bin/env python3
"""Classify what has changed in a frozen parent since its import (plan section 6).

Both parents are frozen, so this is not a merge tool: its job is to make a
post-freeze bug-fix backport visible instead of silent, and to catch a parent
that has quietly moved away from the SHA this repository was built from.

Every upstream commit since the imported tag is classified:

  absorb       touches only paths still byte-identical here -> can be applied as-is
  cherry-pick  touches a path that has diverged in fuse -> a human ports it and
               records the upstream SHA in UPSTREAMS.lock
  superseded   touches code fuse has replaced (listed in [<parent>].superseded)
  conflict     touches a diverged path AND a byte-identical one -> needs a ruling

Prints a table to commit alongside any lock bump. Read-only: it never writes to a
parent and never modifies this repository.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def lock() -> dict[str, dict[str, object]]:
    text = (ROOT / "UPSTREAMS.lock").read_text()
    out: dict[str, dict[str, object]] = {}
    cur = None
    for line in text.splitlines():
        sec = re.match(r"^\[([a-z0-9_]+)\]", line)
        if sec:
            cur = sec.group(1)
            out[cur] = {"diverged": [], "superseded": []}
            continue
        if not cur:
            continue
        kv = re.match(r"^\s*(path|ref|sha|tag)\s*=\s*\"([^\"]+)\"", line)
        if kv:
            out[cur][kv.group(1)] = kv.group(2)
        lst = re.match(r"^\s*(diverged|superseded)\s*=\s*\[(.*)\]\s*$", line)
        if lst:
            out[cur][lst.group(1)] = re.findall(r'"([^"]+)"', lst.group(2))
    return out


def git(path: str, *args: str) -> str:
    return subprocess.run(["git", "-C", path, *args],
                           capture_output=True, text=True, check=True).stdout.strip()


def classify(paths: list[str], diverged: list[str], superseded: list[str]) -> str:
    def matches(p: str, pats: list[str]) -> bool:
        return any(p == q or p.startswith(q.rstrip("*")) for q in pats)

    if paths and all(matches(p, superseded) for p in paths):
        return "superseded"
    hits_div = any(matches(p, diverged) for p in paths)
    hits_clean = any(not matches(p, diverged) and not matches(p, superseded) for p in paths)
    if hits_div and hits_clean:
        return "conflict"
    if hits_div:
        return "cherry-pick"
    return "absorb"


def main() -> int:
    rows, unreachable, moved = [], [], []
    for name, info in sorted(lock().items()):
        path = str(info.get("path", ""))
        tag = str(info.get("tag", ""))
        sha = str(info.get("sha", ""))
        if not (Path(path) / ".git").exists():
            unreachable.append(f"{name}: parent not reachable at {path}")
            continue
        try:
            head = git(path, "rev-parse", "--short", "HEAD")
            tagged = git(path, "rev-parse", "--short", tag)
        except subprocess.CalledProcessError:
            unreachable.append(f"{name}: tag '{tag}' not found in {path}")
            continue
        if not tagged.startswith(sha) and not sha.startswith(tagged):
            moved.append(f"{name}: tag {tag} is {tagged}, lock says {sha}")
        log = git(path, "log", "--format=%h%x00%s", f"{tag}..HEAD")
        if not log:
            print(f"{name}: frozen at {tag} ({tagged}); HEAD {head}; no commits since import")
            continue
        for line in log.splitlines():
            csha, subject = line.split("\0", 1)
            files = git(path, "show", "--name-only", "--format=", csha).splitlines()
            files = [f for f in files if f]
            rows.append((name, csha, classify(files, list(info["diverged"]),
                                              list(info["superseded"])), len(files), subject))

    for m in moved:
        print(f"error: {m}")
    for u in unreachable:
        print(f"warning: {u}")

    if rows:
        print(f"\n| parent | commit | class | files | subject |")
        print(f"|---|---|---|---|---|")
        for name, csha, kind, n, subject in rows:
            print(f"| {name} | `{csha}` | **{kind}** | {n} | {subject[:70]} |")
        need = [r for r in rows if r[2] in ("cherry-pick", "conflict")]
        print(f"\n{len(rows)} commit(s) since import; {len(need)} need a human ruling.")

    return 1 if moved else 0


if __name__ == "__main__":
    sys.exit(main())
