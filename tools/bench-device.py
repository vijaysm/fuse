#!/usr/bin/env python3
"""Timing harness for the scaled device benchmark (docs/dev/testing.md,
"Benchmarks"). Runs fuse_bench_scaled_device_benchmark host vs device N
times at each requested rank count, records /proc/loadavg before and
after every repetition, and prints and saves a Markdown table of the
median and minimum per-phase wall time (the "build", "initialize" and
"period_<k>" rows scaled_device_benchmark.cpp's own --timing CSV writes).

Usage:
    python3 tools/bench-device.py --binary <build>/bench/fuse_bench_scaled_device_benchmark \
        --ranks 1 2 3 --n 1000 --periods 5 --repeats 5 --out bench-report.md

A result is labelled preliminary in the printed/saved report unless the
1-minute load average was below --max-load (default 2) both before and
after EVERY repetition at EVERY rank count -- this script does not refuse
to run on a loaded host, it only refuses to call the numbers quiet.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def one_minute_load():
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except OSError:
        return None


def find_mpiexec():
    for name in ("mpiexec", "mpirun"):
        from shutil import which
        path = which(name)
        if path:
            return path
    return "mpiexec"


def run_once(binary, mpiexec, ranks, n, periods, fields, env_extra):
    import os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        timing_path = tmp.name
    cmd = [mpiexec, "-n", str(ranks), binary,
           "--n", str(n), "--periods", str(periods), "--fields", fields,
           "--timing", timing_path]
    env = os.environ.copy()
    env.update(env_extra)
    load_before = one_minute_load()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    load_after = one_minute_load()
    if proc.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd)} exited {proc.returncode}\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}")
    rows = {}
    with open(timing_path, newline="") as f:
        reader = csv.DictReader(filter(lambda l: not l.startswith("#"), f))
        for row in reader:
            rows[row["phase"]] = float(row["mean_s"])
    Path(timing_path).unlink(missing_ok=True)
    return rows, load_before, load_after


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="path to fuse_bench_scaled_device_benchmark")
    ap.add_argument("--mpiexec", default=None, help="mpiexec/mpirun path (default: found on PATH)")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--periods", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-load", type=float, default=2.0,
                    help="above this 1-minute load average, results are marked preliminary")
    ap.add_argument("--out", default=None, help="Markdown report path (also printed to stdout)")
    ap.add_argument("--json", default=None, help="optional raw-results JSON path")
    args = ap.parse_args()

    mpiexec = args.mpiexec or find_mpiexec()
    results = {}       # (ranks, space) -> {phase: [seconds, ...]}
    quiet = True
    raw = []

    for ranks in args.ranks:
        for space, fields, env_extra in (
            ("host", "host", {}),
            ("device", "device", {"FUSE_EXECUTION_SPACE": "device"}),
        ):
            samples = {}
            for rep in range(args.repeats):
                rows, load_before, load_after = run_once(
                    args.binary, mpiexec, ranks, args.n, args.periods, fields, env_extra)
                raw.append({"ranks": ranks, "space": space, "repeat": rep,
                           "rows": rows, "load_before": load_before, "load_after": load_after})
                for k in (load_before, load_after):
                    if k is None or k >= args.max_load:
                        quiet = False
                for phase, seconds in rows.items():
                    samples.setdefault(phase, []).append(seconds)
            results[(ranks, space)] = samples

    lines = []
    lines.append("# Scaled device benchmark: host vs device timing")
    lines.append("")
    lines.append(f"n={args.n}, periods={args.periods}, repeats={args.repeats}")
    lines.append("" if quiet else
                "**PRELIMINARY** -- the 1-minute load average was >= "
                f"{args.max_load} before or after at least one repetition; "
                "re-run on a quiet host for representative numbers.")
    lines.append("")
    phases = sorted({p for s in results.values() for p in s.keys()})
    header = "| ranks | space | " + " | ".join(f"{p} median (s)" for p in phases) + " | " + \
             " | ".join(f"{p} min (s)" for p in phases) + " |"
    sep = "|" + "---|" * (2 + 2 * len(phases))
    lines.append(header)
    lines.append(sep)
    for ranks in args.ranks:
        for space in ("host", "device"):
            samples = results.get((ranks, space), {})
            medians = [f"{statistics.median(samples[p]):.6g}" if p in samples else "n/a"
                      for p in phases]
            mins = [f"{min(samples[p]):.6g}" if p in samples else "n/a" for p in phases]
            lines.append(f"| {ranks} | {space} | " + " | ".join(medians) + " | " +
                        " | ".join(mins) + " |")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        Path(args.out).write_text(report)
    if args.json:
        Path(args.json).write_text(json.dumps(raw, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
