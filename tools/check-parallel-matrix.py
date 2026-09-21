#!/usr/bin/env python3
"""Check that every cell of the parallel matrix has coverage that exists.

S5.5. The rank counts were already enumerated in test/CMakeLists.txt and the
CommGraph rows already had tests; what was missing was anything that asserted
the matrix was COVERED. A test can be deleted, renamed, or left unregistered
and nothing said the matrix had lost a row -- and that has happened in this
tree before: a whole parallel file compiled to nothing behind a stale guard
while the legs reported passing.

So this gate reads build-evidence/parallel-matrix.toml and, for each cell,
checks that what the cell CLAIMS is really there:

  test    = the TEST(Suite, Name) or TYPED_TEST(Suite, Name) must appear in
            the named file, and that file must be registered in
            test/CMakeLists.txt. A name that no longer matches is a stale
            claim, not a pass.
  refusal = the error code must appear in include/ or src/ -- and in some
            test. A refusal nothing makes fire is a string in a source file;
            this tree's own rule (docs/dev/testing.md) is that a deliberate
            refusal is verified only when a test makes it fire.
  ranks   = for an mpi cell, every rank count named must actually be
            registered for that test file, so a cell cannot claim 5 ranks
            while the build runs 2.
  gap     = counted and reported. The gate exits non-zero while any gap
            stands: a gap is a fact about the library, and the only honest
            way for this to go green is to close one.

Exit 0 only when every claim was verified and no gap remains.
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "build-evidence" / "parallel-matrix.toml"
TEST_CMAKE = ROOT / "test" / "CMakeLists.txt"


def registered_files(cmake: str) -> set[str]:
    """Test source files the build actually compiles.

    fuse_add_gtest(<layer> <name>) and fuse_add_mpi_test(<layer> <name>) both
    name test/<layer>/<name>_test.cpp, and the foreach() spellings expand to
    several names at once.
    """
    files: set[str] = set()
    for layer, names in re.findall(
        r"foreach\(t ([a-z_0-9 ]+)\)\s*\n\s*fuse_add_(?:gtest|mpi_test)\((\w+)",
        cmake,
    ):
        pass  # the two capture orders differ between spellings; handled below

    # foreach(t a b c) ... fuse_add_gtest(<layer> ${t} ...)
    for m in re.finditer(
        r"foreach\(t ([^)]+)\)\s*\n\s*fuse_add_(?:gtest|mpi_test)\((\w+)\s+\$\{t\}",
        cmake,
    ):
        for name in m.group(1).split():
            files.add(f"test/{m.group(2)}/{name}_test.cpp")
    # direct calls
    for m in re.finditer(r"fuse_add_(?:gtest|mpi_test)\((\w+)\s+(\w+)", cmake):
        if m.group(2) != "${t}":
            files.add(f"test/{m.group(1)}/{m.group(2)}_test.cpp")
    return files


def mpi_ranks_for(cmake: str) -> list[int]:
    """The rank counts fuse_add_mpi_test uses when a call names none."""
    # Comments may sit between the guard and the assignment -- they do, and
    # requiring the two to be adjacent made this return [] the moment one was
    # added, which failed every mpi cell at once for no real reason.
    # `[\s\S]*?` and not `[^)]*?`: the comment between the guard and the
    # assignment contains parentheses, so excluding them made this fail on the
    # real file while still passing the probe -- a check that fires for the
    # wrong reason, which is worse than one that does not fire.
    m = re.search(
        r"if\(NOT ARG_RANKS\)[\s\S]*?set\(ARG_RANKS ([0-9 ]+)\)", cmake
    )
    if not m:
        raise SystemExit(
            "error: cannot find fuse_add_mpi_test's default rank list in "
            "test/CMakeLists.txt; the matrix cannot be checked against a "
            "build whose rank counts it cannot read"
        )
    return [int(x) for x in m.group(1).split()]


def main() -> int:
    spec = tomllib.loads(MANIFEST.read_text())
    cmake = TEST_CMAKE.read_text()
    have_files = registered_files(cmake)
    default_ranks = mpi_ranks_for(cmake)

    # Every test body and every code mention, read once.
    test_text = "\n".join(
        p.read_text() for p in sorted((ROOT / "test").rglob("*.cpp"))
    )
    lib_text = "\n".join(
        p.read_text()
        for d in ("include", "src")
        for p in sorted((ROOT / d).rglob("*"))
        if p.is_file() and p.suffix in {".hpp", ".cpp", ".h", ".in"}
    )

    problems: list[str] = []
    gaps: list[tuple[str, str]] = []
    verified = 0

    for cell in spec.get("cell", []):
        cid = cell.get("id", "<no id>")
        for key in ("id", "axis"):
            if key not in cell:
                problems.append(f"{cid}: cell is missing `{key}`")

        if "gap" in cell:
            gaps.append((cid, " ".join(cell["gap"].split())))
            continue

        if not cell.get("test") and not cell.get("refusal") and cell.get(
            "kind"
        ) != "registration":
            problems.append(
                f"{cid}: claims neither a test nor a refusal, and is not a "
                f"registration cell -- an axis with no evidence is a gap and "
                f"must say so"
            )
            continue

        rel = cell.get("file")
        # A cell may span files: "a rank owning nothing is ordinary" is pinned
        # once for the transport and once for the graph, and they live apart.
        extra = cell.get("also_in", [])
        if rel and cell.get("kind") != "registration":
            for other in extra:
                if other not in have_files:
                    problems.append(
                        f"{cid}: {other} is not registered in "
                        f"test/CMakeLists.txt, so nothing in it runs")
            if rel not in have_files:
                problems.append(
                    f"{cid}: {rel} is not registered in test/CMakeLists.txt, "
                    f"so nothing in it runs"
                )
            path = ROOT / rel
            if not path.is_file():
                problems.append(f"{cid}: {rel} does not exist")
                continue
            body = path.read_text() + "\n" + "\n".join(
                (ROOT / o).read_text() for o in extra if (ROOT / o).is_file())
            for full in cell.get("test", []):
                suite, _, name = full.partition(".")
                pat = re.compile(
                    r"(?:TEST|TEST_F|TYPED_TEST)\(\s*" + re.escape(suite) +
                    r"\s*,\s*" + re.escape(name) + r"\s*\)"
                )
                if pat.search(body):
                    verified += 1
                else:
                    problems.append(
                        f"{cid}: {rel}{' (or ' + ', '.join(extra) + ')' if extra else ''}"
                        f" has no test {full} -- the claim names a "
                        f"test that is not there (renamed? deleted?)"
                    )

        for code in cell.get("refusal", []):
            if f'"{code}"' not in lib_text:
                problems.append(
                    f"{cid}: refusal {code} appears nowhere in include/ or "
                    f"src/, so the cell claims a refusal the library does not "
                    f"make"
                )
            elif code not in test_text:
                problems.append(
                    f"{cid}: refusal {code} exists but no test mentions it; a "
                    f"refusal counts as coverage only when a test makes it "
                    f"fire (docs/dev/testing.md)"
                )
            else:
                verified += 1

        if cell.get("kind") == "mpi":
            missing = [n for n in cell.get("ranks", []) if n not in default_ranks]
            if missing:
                problems.append(
                    f"{cid}: claims rank counts {missing} that the build does "
                    f"not run (fuse_add_mpi_test defaults to {default_ranks})"
                )
            else:
                verified += 1

        if cell.get("kind") == "registration":
            missing = [n for n in cell.get("ranks", []) if n not in default_ranks]
            if missing:
                problems.append(
                    f"{cid}: rank counts {missing} are named by the matrix but "
                    f"not registered; fuse_add_mpi_test runs {default_ranks}"
                )
            else:
                verified += 1

    for p in problems:
        print(f"MATRIX-BROKEN  {p}")
    for cid, why in gaps:
        print(f"MATRIX-GAP     {cid}: {why}")
    print(
        f"parallel matrix: {verified} claim(s) verified, {len(problems)} "
        f"broken, {len(gaps)} gap(s), of {len(spec.get('cell', []))} cells"
    )
    if gaps and not problems:
        print(
            "  A gap is not a pass. It is coverage this library does not have, "
            "and the count can only go down by writing the test."
        )
    return 1 if (problems or gaps) else 0


if __name__ == "__main__":
    sys.exit(main())
