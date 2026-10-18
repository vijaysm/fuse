#!/usr/bin/env python3
"""Every gtest source that asserts an exception is raised must be registered
with THROWS, so the no-exceptions CI leg (ctest -LE throws) skips it.

A test that calls EXPECT_THROW/ASSERT_THROW (or catches fuse::Refusal /
fuse::InvalidArgument by hand) cannot pass in a FUSE_ENABLE_EXCEPTIONS=OFF
build: the refusal aborts the process instead, and ctest reports
"Subprocess aborted". Three such tests sat unlabelled until the leg first ran
as a leg (2026-10-18: model.multiscale, model.component,
verification.brusselator_species_core_decomposition); this gate finds them
statically from test/CMakeLists.txt, both the direct fuse_add_gtest calls and
the foreach loops, so the leg never has to.

Exit 0 when every throwing test carries THROWS, 1 otherwise, listing each.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEST = ROOT / "test"
THROW_PATTERN = re.compile(
    r"EXPECT_THROW|ASSERT_THROW|EXPECT_ANY_THROW|ASSERT_ANY_THROW"
    r"|catch \(const fuse::(Refusal|InvalidArgument)")


def registrations(cmake_text: str) -> tuple[set[str], set[str]]:
    """(registered with THROWS, registered without) as layer/name_test.cpp."""
    throws: set[str] = set()
    plain: set[str] = set()
    for m in re.finditer(r"fuse_add_gtest\((\w+)\s+(\w+)([^)]*)\)", cmake_text):
        layer, name, rest = m.groups()
        if name.startswith("${"):
            continue
        (throws if "THROWS" in rest else plain).add(f"{layer}/{name}_test.cpp")
    for m in re.finditer(
            r"foreach\(t ([^)]*)\)(.*?)fuse_add_gtest\((\w+) \$\{t\}([^)]*)\)",
            cmake_text, re.S):
        names, body, layer, rest = m.groups()
        # A per-name THROWS inside the loop body (see the Brusselator
        # foreach) marks that one name only.
        per_name = set(re.findall(r'STREQUAL "(\w+)"', body)) if "THROWS" in body else set()
        for n in names.split():
            target = throws if ("THROWS" in rest or n in per_name) else plain
            target.add(f"{layer}/{n}_test.cpp")
    return throws, plain


def main() -> int:
    throws, plain = registrations((TEST / "CMakeLists.txt").read_text())
    bad = []
    for path in sorted(TEST.rglob("*_test.cpp")):
        rel = path.relative_to(TEST).as_posix()
        if rel in plain and rel not in throws and THROW_PATTERN.search(path.read_text()):
            bad.append(rel)
    for rel in bad:
        print(f"error: test/{rel} asserts an exception but is registered without THROWS")
    print(f"check-throws-labels: {len(bad)} unlabelled throwing test(s) of "
          f"{len(throws) + len(plain)} registered")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
