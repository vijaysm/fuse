#!/usr/bin/env python3
"""The mutation gate (docs/dev/testing.md): a ported area is accepted only when
each of its named mutations is KILLED -- the suite fails with it applied and
passes without it. A green suite that survives every mutation has proved
nothing about the code it claims to cover.

Mutations live in build-evidence/mutations.toml. Each names a file, an exact
`old` string that must occur exactly once, the `new` text, and the test target
expected to catch it. This script applies one mutation at a time to a scratch
copy of the tree, builds and runs that target, restores, and writes the log.

    tools/mutate.py --build-dir <dir> [--area base] [--write-log]

Never run against a dirty tree: the restore is a copy-back, and the gate refuses
to start if git status is not clean.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd: list[str], cwd: Path | None = None) -> int:
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True, type=Path)
    ap.add_argument("--area", default=None)
    ap.add_argument("--write-log", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    if subprocess.run(["git", "status", "--porcelain", "include", "src"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip():
        print("error: include/ or src/ has uncommitted changes; the gate restores by copy-back")
        return 2

    spec = tomllib.loads((ROOT / "build-evidence" / "mutations.toml").read_text())
    rows = [m for m in spec["mutation"] if args.area is None or m["area"] == args.area]
    results = []
    for m in rows:
        path = ROOT / m["file"]
        original = path.read_text()
        if original.count(m["old"]) != 1:
            results.append((m, "INVALID", f"`old` occurs {original.count(m['old'])} times"))
            continue
        try:
            path.write_text(original.replace(m["old"], m["new"], 1))
            build = sh(["cmake", "--build", str(args.build_dir), "-j", str(args.jobs), "--target", m["target"]])
            if build != 0:
                # A mutant that does not compile has proved nothing about the
                # tests. It is the MUTATION that is defective; rewrite it.
                verdict, note = "INCONCLUSIVE", "mutant does not compile; rewrite it"
            else:
                run = sh(["ctest", "--test-dir", str(args.build_dir), "-R", f"^{m['test']}$"])
                verdict, note = ("KILLED", "test fails") if run != 0 else ("SURVIVED", "suite still green")
        finally:
            path.write_text(original)
        results.append((m, verdict, note))
        print(f"{verdict:<9} {m['id']:<36} {note}")

    # restore the build so the next ordinary build is not a mutant
    sh(["cmake", "--build", str(args.build_dir), "-j", str(args.jobs)])

    killed = sum(1 for _, v, _ in results if v == "KILLED")
    survived = [(m, n) for m, v, n in results if v != "KILLED"]
    print(f"\nmutation gate: {killed} killed, {len(survived)} not killed, of {len(results)}")

    if args.write_log:
        area = args.area or "all"
        log = ROOT / "build-evidence" / f"{area}-mutations.md"
        lines = [f"# Mutation evidence — `{area}`", "",
                 f"Run {date.today().isoformat()} against build `{args.build_dir.name}`. "
                 "Regenerate with `tools/mutate.py --write-log`; do not edit.", "",
                 "| Mutation | File | Verdict | Killed by |", "|---|---|---|---|"]
        for m, v, n in results:
            lines.append(f"| `{m['id']}` — {m['describes']} | `{m['file']}` | **{v}** ({n}) | `{m['test']}` |")
        lines += ["", f"**{killed} killed / {len(results)} total.**"]
        log.write_text("\n".join(lines) + "\n")
        print(f"wrote {log.relative_to(ROOT)}")
    return 0 if not survived else 1


if __name__ == "__main__":
    sys.exit(main())
