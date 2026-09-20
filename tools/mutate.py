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
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd: list[str], cwd: Path | None = None) -> int:
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def build(build_dir: Path, target: str, jobs: int) -> tuple[int, str]:
    r = subprocess.run(["cmake", "--build", str(build_dir), "-j", str(jobs), "--target", target],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def load_manifest() -> list[dict]:
    """build-evidence/mutations.toml plus every build-evidence/mutations/*.toml.

    The directory form exists so that work on different layers can add
    mutations without editing one shared file; ids must be unique across all
    of them, and a duplicate is an error rather than a silent shadow."""
    rows: list[dict] = []
    files = [ROOT / "build-evidence" / "mutations.toml"]
    files += sorted((ROOT / "build-evidence" / "mutations").glob("*.toml"))
    for f in files:
        if f.is_file():
            rows += tomllib.loads(f.read_text()).get("mutation", [])
    seen: dict[str, int] = {}
    for m in rows:
        seen[m["id"]] = seen.get(m["id"], 0) + 1
    dups = sorted(k for k, v in seen.items() if v > 1)
    if dups:
        raise SystemExit(f"error: duplicate mutation id(s) across manifests: {dups}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True, type=Path)
    ap.add_argument("--area", default=None)
    ap.add_argument("--write-log", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--check-anchors", action="store_true",
                    help="verify every `old` occurs exactly once in its file; no build, "
                         "no git requirement. Run this in CI: an anchor that drifted is a "
                         "mutation that is silently no longer assessed (REVIEW.md BT-F5).")
    args = ap.parse_args()

    spec_rows = load_manifest()
    if args.check_anchors:
        bad = 0
        for m in spec_rows:
            path = ROOT / m["file"]
            n = path.read_text().count(m["old"]) if path.is_file() else -1
            if n != 1:
                bad += 1
                where = "file missing" if n < 0 else f"`old` occurs {n} times"
                print(f"STALE-ANCHOR  {m['id']}  {m['file']}: {where}")
        print(f"mutation anchors: {len(spec_rows) - bad} valid, {bad} stale, of {len(spec_rows)}")
        return 1 if bad else 0

    # A build directory that is not a configured CMake tree makes EVERY build fail,
    # which the verdict logic would read as "the mutant does not compile" -- 12
    # inconclusive results that look like a code problem and are a path typo.
    cache = args.build_dir / "CMakeCache.txt"
    if not cache.is_file():
        print(f"error: {args.build_dir} is not a configured CMake build directory")
        return 2

    # And it must be a build OF THIS TREE. A stale build directory belonging to
    # another source root builds fine and runs tests that never see the mutation,
    # so every mutant comes back inconclusive -- which reads as a code problem and
    # is a path problem. This happened once; hence the check.
    home = ""
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:"):
            home = line.split("=", 1)[1].strip()
            break
    if Path(home).resolve() != ROOT.resolve():
        print(f"error: {args.build_dir} was configured for {home}, not {ROOT}")
        return 2

    if subprocess.run(["git", "status", "--porcelain", "include", "src"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip():
        print("error: include/ or src/ has uncommitted changes; the gate restores by copy-back")
        return 2

    rows = [m for m in spec_rows if args.area is None or m["area"] == args.area]
    results = []
    # A mutation whose test cannot RUN here must not be reported killed: the
    # test exits non-zero whether or not the mutation had any effect, which is
    # a false kill and worse than no evidence at all. `requires = "mpi"` marks
    # those, and they are assessed on a host with a working launcher.
    mpi_available = os.environ.get("FUSE_MUTATE_MPI", "") == "1"
    for m in rows:
        if m.get("requires") == "mpi" and not mpi_available:
            results.append((m, "NOT-ASSESSED",
                            "needs a working MPI launcher; run with "
                            "FUSE_MUTATE_MPI=1 on a host that has one"))
            print(f"NOT-ASSESSED  {m['id']}  (needs MPI)")
            continue
        path = ROOT / m["file"]
        original = path.read_text()
        if original.count(m["old"]) != 1:
            results.append((m, "INVALID", f"`old` occurs {original.count(m['old'])} times"))
            continue
        try:
            path.write_text(original.replace(m["old"], m["new"], 1))
            rc, log = build(args.build_dir, m["target"], args.jobs)
            if rc != 0:
                # Two very different reasons a mutant fails to build. A
                # static_assert firing IN THE TEST FILE is a compile-time kill:
                # the test caught it before main(). Any other error means the
                # MUTATION is malformed and has proved nothing; rewrite it.
                test_file = f"/{m['test'].replace('.', '/')}_test."
                if "static assertion failed" in log and test_file in log:
                    verdict, note = "KILLED", "test static_assert fails"
                else:
                    verdict, note = "INCONCLUSIVE", "mutant does not compile; rewrite it"
            else:
                run = sh(["ctest", "--test-dir", str(args.build_dir), "-R", f"^{m['test']}$"])
                verdict, note = ("KILLED", "test fails") if run != 0 else ("SURVIVED", "suite still green")
                # A mutation may be declared EQUIVALENT: the mutated code has
                # the same observable behaviour as the original, so no test
                # can distinguish them. It is allowed only with a written
                # reason, and a surviving mutant WITHOUT one is still a gap.
                # If such a mutation is ever killed, the declaration was wrong
                # and that is reported too.
                if m.get("equivalent"):
                    if verdict == "SURVIVED":
                        verdict, note = "EQUIVALENT", m["equivalent"]
                    else:
                        verdict, note = ("WRONGLY-DECLARED-EQUIVALENT",
                                         "a test distinguishes it, so it is not equivalent: "
                                         + m["equivalent"])
        finally:
            path.write_text(original)
        results.append((m, verdict, note))
        print(f"{verdict:<9} {m['id']:<36} {note}")

    # restore the build so the next ordinary build is not a mutant
    sh(["cmake", "--build", str(args.build_dir), "-j", str(args.jobs)])

    killed = sum(1 for _, v, _ in results if v == "KILLED")
    equivalent = sum(1 for _, v, _ in results if v == "EQUIVALENT")
    not_assessed = sum(1 for _, v, _ in results if v == "NOT-ASSESSED")
    survived = [(m, n) for m, v, n in results
                if v not in ("KILLED", "EQUIVALENT", "NOT-ASSESSED")]
    print(f"\nmutation gate: {killed} killed, {equivalent} declared equivalent, "
          f"{not_assessed} not assessed here, {len(survived)} not killed, "
          f"of {len(results)}")

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
