#!/usr/bin/env python3
"""check-test-cost.py -- keep the quick test tier quick.

`ctest -LE slow` is the routine pre-commit run, and it is only as fast as the
`slow` list is complete. cmake/FuseSlowTests.cmake lists the rows whose
measured average wall time is at least THRESHOLD_S seconds; this script reads
a build directory's own cost record (Testing/Temporary/CTestCostData.txt,
which ctest updates on every run) and fails when a row at or above the
threshold is NOT in that list, so a row that has become slow cannot silently
grow the quick tier.

    python3 tools/check-test-cost.py [BUILD_DIR] [--threshold SECONDS]

Exit 0: every measured-slow row is listed. Exit 1: at least one is missing
(printed with its time). Exit 2: nothing to check (no build directory or no
cost record) -- "not measured" is reported as such and is never a pass.

Two limits, stated rather than hidden. A cost record only holds rows that have
run, so a row never run in this build directory is not checked. And an average
taken while the machine was loaded overstates a row's cost; the gate therefore
reports each offending row's run count, and a row listed here that has since
measured well under the threshold is printed as information (never a failure),
because a stale entry costs only a row that is skipped in the quick tier.
"""
import argparse
import re
import sys
from pathlib import Path

THRESHOLD_S = 10.0


def slow_list(root: Path):
    text = (root / "cmake" / "FuseSlowTests.cmake").read_text()
    m = re.search(r"set\(FUSE_SLOW_TESTS\n(.*?)\n\)", text, re.S)
    if not m:
        raise SystemExit("check-test-cost: cannot find set(FUSE_SLOW_TESTS ...) in "
                         "cmake/FuseSlowTests.cmake")
    names = []
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(line)
    return names


def cost_rows(build: Path):
    f = build / "Testing" / "Temporary" / "CTestCostData.txt"
    if not f.is_file():
        return None
    rows = {}
    for line in f.read_text().splitlines():
        if line.startswith("---"):
            break
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows[parts[0]] = (int(parts[1]), float(parts[2]))
            except ValueError:
                pass
    return rows


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir", nargs="?", default="build-all")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_S)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    listed = set(slow_list(root))
    build = Path(args.build_dir)
    if not build.is_absolute():
        build = root / build
    rows = cost_rows(build)
    if rows is None:
        print(f"check-test-cost: no cost record under {build} -- not measured")
        return 2
    missing = sorted(((t, n, r) for n, (r, t) in rows.items()
                      if t >= args.threshold and n not in listed), reverse=True)
    for t, n, r in missing:
        print(f"error: {n}: measured {t:.1f} s average over {r} run(s), at or above "
              f"{args.threshold:g} s, but not in FUSE_SLOW_TESTS "
              f"(cmake/FuseSlowTests.cmake)")
    stale = sorted(n for n in listed if n in rows and rows[n][1] < args.threshold / 4)
    for n in stale:
        print(f"info: {n}: listed as slow but last measured {rows[n][1]:.1f} s "
              f"(under a quarter of the threshold); consider unlisting after a quiet re-run")
    unseen = sorted(n for n in listed if n not in rows)
    if unseen:
        print(f"info: {len(unseen)} listed row(s) have no cost record in this build "
              f"directory (gated off here, or never run): not checked")
    print(f"check-test-cost: {len(missing)} slow row(s) missing from the list "
          f"({len(rows)} rows measured, threshold {args.threshold:g} s)")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
