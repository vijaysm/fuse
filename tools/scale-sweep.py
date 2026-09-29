#!/usr/bin/env python3
"""Sweep fuse_example_layout_driver over rank counts and processor-set shapes.

Runs the processor-layout driver (examples/layout_driver/main.cpp) once per
(rank count, shape) cell of the sweep, in either STRONG mode (a fixed total
entity count at every rank count) or WEAK mode (entity count scaled linearly
with rank count, so the work per rank stays constant), and concatenates every
run's --timing CSV into one output file.

Output schema (this script's own, distinct from the driver's per-run CSV,
which docs/user/layouts.md documents separately):

    # fuse/tools/scale-sweep.py output
    # schema: mode,ranks,layout,shape,n,phase,min_s,mean_s,max_s
    # mode   : "strong" (n fixed across ranks) or "weak" (n = entities_per_rank * ranks)
    # ranks  : MPI world size for this row
    # layout : "c1_<lo>-<hi>_c2_<lo>-<hi>", exactly as the driver's own CSV names it
    # shape  : "identical" | "disjoint" | "partial" -- the --shape this row came from
    # n      : total entities (--entities) for this row's run
    # phase  : "initialize" or "period_<k>", copied from the driver's own CSV
    # min_s/mean_s/max_s : copied verbatim from the driver's own CSV row

Each cell is one `mpiexec -n <ranks> fuse_example_layout_driver --comp1 ...
--comp2 ... --entities ... --steps ... --timing <tmp>.csv` invocation; ranges
are computed from --shape and the rank count exactly as
examples/CMakeLists.txt's registered layouts are (identical: both components
span the whole world; disjoint: the lower and upper halves; partial: needs
>= 3 ranks and shares every rank but the two ends).

--dry-run prints the mpiexec command line for every cell and exits without
running anything -- this machine has 12 cores (docs/user/layouts.md), so a
real sweep past 12 ranks needs a cluster; --dry-run is how the command lines
are checked without one.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def comp_ranges(ranks: int, shape: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """(comp1, comp2) inclusive ranges for `shape` at this rank count.

    Mirrors examples/CMakeLists.txt's registered layouts: identical spans the
    whole world twice, disjoint splits it in half, and partial (needs >= 3
    ranks, same as the driver itself) shares every rank but the two ends
    comp1 keeps at 0 and comp2 keeps at ranks-1.
    """
    if ranks < 1:
        raise ValueError(f"ranks must be >= 1, got {ranks}")
    if shape == "identical":
        return (0, ranks - 1), (0, ranks - 1)
    if shape == "disjoint":
        if ranks < 2:
            raise ValueError("disjoint needs >= 2 ranks")
        mid = ranks // 2
        return (0, mid - 1), (mid, ranks - 1)
    if shape == "partial":
        if ranks < 3:
            raise ValueError("partial overlap needs >= 3 ranks")
        return (0, ranks - 2), (1, ranks - 1)
    raise ValueError(f"unknown shape '{shape}'")


def layout_name(comp1: tuple[int, int], comp2: tuple[int, int]) -> str:
    return f"c1_{comp1[0]}-{comp1[1]}_c2_{comp2[0]}-{comp2[1]}"


def entities_for(mode: str, ranks: int, entities: int) -> int:
    if mode == "strong":
        return entities
    if mode == "weak":
        return entities * ranks
    raise ValueError(f"unknown mode '{mode}'")


def build_command(args: argparse.Namespace, ranks: int, comp1: tuple[int, int],
                  comp2: tuple[int, int], n: int, timing_path: Path) -> list[str]:
    cmd = [
        args.mpiexec, args.numproc_flag, str(ranks), args.driver,
        "--comp1", f"{comp1[0]}:{comp1[1]}",
        "--comp2", f"{comp2[0]}:{comp2[1]}",
        "--entities", str(n),
        "--steps", str(args.steps),
        "--timing", str(timing_path),
    ]
    if args.check:
        cmd.append("--check")
    return cmd


def read_driver_csv(path: Path) -> list[dict[str, str]]:
    """Data rows only -- the driver's own '#'-prefixed schema header is
    documented once by this script's own header, not repeated per row."""
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append(row)
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ranks", type=str, default="2,3,4",
                   help="comma-separated rank counts to sweep, e.g. 2,3,4")
    p.add_argument("--shapes", type=str, default="identical,disjoint,partial",
                   help="comma-separated shapes to sweep: identical, disjoint, partial")
    p.add_argument("--mode", choices=["strong", "weak"], default="strong",
                   help="strong: --entities is the fixed total; weak: --entities is "
                        "the per-rank count, scaled by the rank count")
    p.add_argument("--entities", type=int, default=100,
                   help="strong mode: total entities; weak mode: entities per rank")
    p.add_argument("--steps", type=int, default=5, help="--steps forwarded to the driver")
    p.add_argument("--check", action="store_true",
                   help="forward --check to every run (recommended; it is not the default "
                        "here because a sweep's purpose is timing, not correctness)")
    p.add_argument("--driver", type=str,
                   default="./build-t1/examples/fuse_example_layout_driver",
                   help="path to fuse_example_layout_driver")
    p.add_argument("--mpiexec", type=str, default="mpiexec", help="mpiexec launcher")
    p.add_argument("--numproc-flag", type=str, default="-n",
                   help="the launcher's rank-count flag, e.g. -n or -np")
    p.add_argument("--out", type=str, default="scale-sweep.csv",
                   help="output CSV path")
    p.add_argument("--dry-run", action="store_true",
                   help="print every mpiexec command line and exit; run nothing")
    args = p.parse_args()

    ranks_list = [int(r) for r in args.ranks.split(",") if r.strip()]
    shapes_list = [s.strip() for s in args.shapes.split(",") if s.strip()]

    cells: list[tuple[int, str, tuple[int, int], tuple[int, int], int]] = []
    skipped: list[str] = []
    for ranks in ranks_list:
        for shape in shapes_list:
            try:
                comp1, comp2 = comp_ranges(ranks, shape)
            except ValueError as e:
                skipped.append(f"ranks={ranks} shape={shape}: {e}")
                continue
            n = entities_for(args.mode, ranks, args.entities)
            cells.append((ranks, shape, comp1, comp2, n))

    if args.dry_run:
        for ranks, shape, comp1, comp2, n in cells:
            timing_path = Path(f"/tmp/scale-sweep.{ranks}.{shape}.csv")
            cmd = build_command(args, ranks, comp1, comp2, n, timing_path)
            print(" ".join(cmd))
        for s in skipped:
            print(f"# skipped: {s}", file=sys.stderr)
        return 0

    if not shutil.which(args.mpiexec) and not Path(args.mpiexec).is_file():
        print(f"error: mpiexec launcher '{args.mpiexec}' not found", file=sys.stderr)
        return 1
    if not Path(args.driver).is_file():
        print(f"error: driver '{args.driver}' not found -- build "
              f"fuse_example_layout_driver first (docs/user/layouts.md)", file=sys.stderr)
        return 1

    out_rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for ranks, shape, comp1, comp2, n in cells:
            timing_path = Path(tmp) / f"{ranks}.{shape}.csv"
            cmd = build_command(args, ranks, comp1, comp2, n, timing_path)
            print("+ " + " ".join(cmd), file=sys.stderr)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"error: run failed (exit {result.returncode}) for "
                      f"ranks={ranks} shape={shape}\n{result.stderr}", file=sys.stderr)
                return 1
            layout = layout_name(comp1, comp2)
            for row in read_driver_csv(timing_path):
                out_rows.append({
                    "mode": args.mode,
                    "ranks": row["ranks"],
                    "layout": layout,
                    "shape": shape,
                    "n": row["n"],
                    "phase": row["phase"],
                    "min_s": row["min_s"],
                    "mean_s": row["mean_s"],
                    "max_s": row["max_s"],
                })

    for s in skipped:
        print(f"# skipped: {s}", file=sys.stderr)

    with open(args.out, "w", newline="") as f:
        f.write("# fuse/tools/scale-sweep.py output\n")
        f.write("# schema: mode,ranks,layout,shape,n,phase,min_s,mean_s,max_s\n")
        f.write("# mode   : \"strong\" (n fixed across ranks) or \"weak\" "
                "(n = entities_per_rank * ranks)\n")
        f.write("# ranks  : MPI world size for this row\n")
        f.write("# layout : \"c1_<lo>-<hi>_c2_<lo>-<hi>\", as the driver's own CSV names it\n")
        f.write("# shape  : \"identical\" | \"disjoint\" | \"partial\"\n")
        f.write("# n      : total entities (--entities) for this row's run\n")
        f.write("# phase  : \"initialize\" or \"period_<k>\", copied from the driver's CSV\n")
        f.write("# min_s/mean_s/max_s : copied verbatim from the driver's own CSV row\n")
        writer = csv.DictWriter(
            f, fieldnames=["mode", "ranks", "layout", "shape", "n", "phase",
                          "min_s", "mean_s", "max_s"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} row(s) to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
