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

Never run against a dirty tree: the restore is a copy-back, and the gate rejects
starting if git status is not clean.
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

    # Every key this tool reads, checked here rather than where it is used. A
    # row missing one used to raise KeyError deep in the run loop: the area
    # ABANDONED every remaining mutation and printed a traceback instead of a
    # verdict line, so from the caller's side an area with one malformed row
    # was indistinguishable from an area with nothing to do -- and
    # --check-anchors still called all of them valid, because it only reads
    # `file` and `old`. Found 2026-09-21 by adding two rows without `target`.
    required = ("id", "area", "describes", "file", "old", "new", "target", "test")
    missing = [
        f"{m.get('id', '<no id>')}: missing {k}"
        for m in rows
        for k in required
        if k not in m
    ]
    if missing:
        raise SystemExit("error: malformed mutation row(s):\n  " + "\n  ".join(missing))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    # Not `required=True`: the --check-anchors branch below never reads
    # args.build_dir at all (it is a text-anchor check over
    # build-evidence/*.toml and the source tree, no build, no git
    # requirement), but argparse's own requiredness demanded one anyway --
    # which is exactly what kept this sub-second static check out of CI
    # (BC-3): a `gates` job leg has no configured build directory to hand
    # it. `type=Path` only applies when a value is actually given.
    ap.add_argument("--build-dir", default=None, type=Path)
    ap.add_argument("--area", default=None)
    ap.add_argument("--write-log", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--check-anchors", action="store_true",
                    help="verify every `old` occurs exactly once in its file; no build, "
                         "no git requirement. Run this in CI: an anchor that drifted is a "
                         "mutation that is silently no longer assessed (REVIEW.md BT-F5).")
    args = ap.parse_args()

    if args.build_dir is None and not args.check_anchors:
        ap.error("--build-dir is required unless --check-anchors is given")

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

    # Read from the build itself, not from an env var the caller might set to
    # mean something else: FUSE_ENABLE_MPI is what decides whether a target
    # like nodal_remap_distributed_nullcomm_test's `#if !FUSE_HAVE_MPI` body
    # GTEST_SKIPs. A skipped test exits 0 whichever way `old`/`new` reads, so
    # a `requires = "no-mpi"` row run against an MPI build is not evidence
    # either way -- reported NOT-ASSESSED, the same courtesy `requires =
    # "mpi"` gets on a build with none.
    build_has_mpi = any(
        line.startswith("FUSE_ENABLE_MPI:") and line.rstrip().endswith("=ON")
        for line in cache.read_text(errors="replace").splitlines())

    # Same courtesy, for `requires = "lowfive"`: LowFiveTransport's real
    # body (src/transport/lowfive_transport.cpp) is entirely behind `#if
    # FUSE_HAVE_LOWFIVE` -- a build configured with FUSE_ENABLE_LOWFIVE=OFF
    # (the default; tier 0's own configure does not turn it on) compiles
    # out that code path altogether, so its own test binary runs only the
    # `#else` stub suite (`LowFiveTransport.EveryEntryPointRefusesWithout
    # LowFive`), which passes identically with or without the mutation --
    # a false SURVIVED, not evidence the mutated line is unobservable.
    # Checked from the cache, like `build_has_mpi` above, because this is
    # a pure compile-time flag with no launcher to fail at runtime (unlike
    # "mpi"/"petsc", which also need a working process launcher).
    build_has_lowfive = any(
        line.startswith("FUSE_ENABLE_LOWFIVE:") and line.rstrip().endswith("=ON")
        for line in cache.read_text(errors="replace").splitlines())

    # FpPolicy track, KNOWN_ISSUES.md D30: a `requires = "release"` gate
    # (read from CMAKE_BUILD_TYPE, the same courtesy "mpi"/"petsc" get) was
    # added here for base.fp_policy's probe on the premise that Debug
    # (-O0) never contracts a*b+c. That premise is false -- confirmed by
    # disassembly (docs/dev/testing.md's "Floating-point contraction
    # policy" section) that -ffp-contract defaults to `on` at every
    # optimization level -- so the mutation is equally visible in Debug
    # and Release (hand-proved both ways, build-evidence/mutations/
    # fp-policy.toml), and the gate is removed rather than kept unused.

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

    # The SAME courtesy, for `requires = "petsc"`: the PetscToken/PetscSelect
    # tracks' own test binaries (test/petsc/*_test.cpp) call
    # fuse::petsc::ensure_initialized(), which calls PETSc's own
    # PetscInitialize() -- and PetscInitialize() calls MPI_Init() when MPI is
    # not already up (fuse/petsc/token.hpp's own ensure_initialized() comment).
    # A sandbox that cannot complete a bare MPI_Init() at all (KNOWN_ISSUES.md
    # D32's own finding, independent of PETSc) therefore cannot run ANY
    # petsc.* test, in ANY build configuration -- unlike `mpi`/`no-mpi`, this
    # is not a question of whether FUSE_ENABLE_MPI was turned on, so it is
    # gated the same way "mpi" is (an explicit env var a host with a working
    # MPI runtime sets), not by inspecting FUSE_ENABLE_PETSC in the cache.
    petsc_available = os.environ.get("FUSE_MUTATE_PETSC", "") == "1"

    # The SAME argument, one step further: a test that is ALREADY FAILING on
    # the unmutated tree also exits non-zero whether or not the mutation had
    # any effect. Reporting that as KILLED is a false kill, and the gate did
    # exactly that on 2026-09-21 -- three mutations whose test was red were
    # all reported killed, which is how the omission was found. So each
    # distinct test is run once on the clean tree first, and a mutation whose
    # test does not pass there is reported NO-BASELINE and proves nothing.
    baseline: dict[str, bool] = {}

    def baseline_ok(target: str, test: str) -> bool:
        if test not in baseline:
            rc, _ = build(args.build_dir, target, args.jobs)
            baseline[test] = rc == 0 and sh(
                ["ctest", "--test-dir", str(args.build_dir), "--no-tests=error",
                 "-R", f"^{test}$"]) == 0
        return baseline[test]

    for m in rows:
        if m.get("requires") == "mpi" and not mpi_available:
            results.append((m, "NOT-ASSESSED",
                            "needs a working MPI launcher; run with "
                            "FUSE_MUTATE_MPI=1 on a host that has one"))
            print(f"NOT-ASSESSED  {m['id']}  (needs MPI)")
            continue
        if m.get("requires") == "no-mpi" and build_has_mpi:
            results.append((m, "NOT-ASSESSED",
                            "its test's own #if !FUSE_HAVE_MPI body GTEST_SKIPs on an "
                            "MPI build, which exits 0 either way; assess it against a "
                            "build configured with -DFUSE_ENABLE_MPI=OFF"))
            print(f"NOT-ASSESSED  {m['id']}  (needs a build with NO MPI)")
            continue
        if m.get("requires") == "lowfive" and not build_has_lowfive:
            results.append((m, "NOT-ASSESSED",
                            "needs a build configured with -DFUSE_ENABLE_LOWFIVE=ON; "
                            "LowFiveTransport's real body is behind #if FUSE_HAVE_LOWFIVE "
                            "and does not exist to fail in a build without it"))
            print(f"NOT-ASSESSED  {m['id']}  (needs FUSE_ENABLE_LOWFIVE=ON)")
            continue
        if m.get("requires") == "petsc" and not petsc_available:
            results.append((m, "NOT-ASSESSED",
                            "needs a PETSc-enabled build whose MPI_Init() actually "
                            "completes (ensure_initialized() calls PetscInitialize(), "
                            "which calls MPI_Init()); run with FUSE_MUTATE_PETSC=1 on "
                            "a host that has one"))
            print(f"NOT-ASSESSED  {m['id']}  (needs PETSc)")
            continue
        if not baseline_ok(m["target"], m["test"]):
            results.append((m, "NO-BASELINE",
                            f"`{m['test']}` already fails without the mutation, "
                            "so its failure with it applied is not evidence"))
            print(f"NO-BASELINE   {m['id']:<36} {m['test']} is already red")
            continue
        path = ROOT / m["file"]
        original = path.read_text()
        if original.count(m["old"]) != 1:
            note = f"`old` occurs {original.count(m['old'])} times"
            results.append((m, "INVALID", note))
            print(f"INVALID   {m['id']:<36} {note}")
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
                run = sh(["ctest", "--test-dir", str(args.build_dir),
                          "--no-tests=error", "-R", f"^{m['test']}$"])
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
        # Confirm the restore put back the exact pre-mutation bytes. In one
        # sandboxed full sweep (2026-10-18, 363 rows, Linux overlay file
        # system) ten files were left mutated with no exception and an
        # ordinary verdict; not reproduced on the owner's machine, but a gate
        # whose contract is "the tree is byte-identical afterwards" must
        # check it rather than trust the write. A failure here is loud.
        restored = path.read_text()
        if restored != original:
            path.write_text(original)
            restored2 = path.read_text()
            if restored2 != original:
                verdict, note = ("RESTORE-FAILED",
                                  f"{m['file']} did not restore to its pre-mutation "
                                  "bytes even after a forced second write")
            else:
                note = note + " [RESTORE-NOTE: required a forced second write]"
        results.append((m, verdict, note))
        print(f"{verdict:<9} {m['id']:<36} {note}")

    # restore the build so the next ordinary build is not a mutant
    sh(["cmake", "--build", str(args.build_dir), "-j", str(args.jobs)])

    killed = sum(1 for _, v, _ in results if v == "KILLED")
    equivalent = sum(1 for _, v, _ in results if v == "EQUIVALENT")
    not_assessed = sum(1 for _, v, _ in results if v == "NOT-ASSESSED")
    no_baseline = sum(1 for _, v, _ in results if v == "NO-BASELINE")
    survived = [(m, n) for m, v, n in results
                if v not in ("KILLED", "EQUIVALENT", "NOT-ASSESSED", "NO-BASELINE")]
    print(f"\nmutation gate: {killed} killed, {equivalent} declared equivalent, "
          f"{not_assessed} not assessed here, {no_baseline} without a baseline, "
          f"{len(survived)} not killed, of {len(results)}")
    if no_baseline:
        print("  NO-BASELINE is not a pass: fix the failing test, then re-run "
              "to find out whether these mutations are really killed.")

    if args.write_log:
        area = args.area or "all"
        # An area may be a path ("examples/brusselator"); the log is one file.
        log = ROOT / "build-evidence" / f"{area.replace('/', '-')}-mutations.md"
        lines = [f"# Mutation evidence -- `{area}`", "",
                 f"Run {date.today().isoformat()} against build `{args.build_dir.name}`. "
                 "Regenerate with `tools/mutate.py --write-log`; do not edit.", "",
                 "| Mutation | File | Outcome | Killed by |", "|---|---|---|---|"]
        for m, v, n in results:
            lines.append(f"| `{m['id']}` -- {m['describes']} | `{m['file']}` | **{v}** ({n}) | `{m['test']}` |")
        lines += ["", f"**{killed} killed / {len(results)} total.**"]
        log.write_text("\n".join(lines) + "\n")
        print(f"wrote {log.relative_to(ROOT)}")
    return 0 if not survived and not no_baseline else 1


if __name__ == "__main__":
    sys.exit(main())
