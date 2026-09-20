#!/usr/bin/env python3
"""Enforce invariant I7 and the layering rule: dependencies point down only.

Run from the repository root. Exits non-zero with a report of every violation,
never on the first one -- the same accumulate-and-report rule the library's own
diagnostics follow (design rule A2).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# --- rules
# Layers, lowest first. A layer may include from its own layer and any layer
# BELOW it, never above.
# Bottom to top. `backend` and `transport` sit BELOW `runtime`: the runtime
# drives mesh access, spatial transfer and data movement through their
# interfaces, so it depends on them and not the reverse. (They were the other
# way round until the runtime port, where the order said the Runner's use of
# MeshBackend -- which is how it resolves a remap at all -- was an inversion.
# `transport` was missing from this list entirely, which meant its headers were
# not checked at all.) `transport` is above `backend` because matching ranks
# across two rank sets needs the global cell ids, which come from the mesh
# interface.
# Every directory under include/fuse MUST appear here. Three did not until
# 2026-09-17 -- `field`, `linalg` and `abi` -- while this list named an
# `interface` layer that no directory matched. A path that maps to no layer is
# assigned no rank and no third-party allowance, so the checks below simply
# never ran on those three: an entire directory could have included <mpi.h> in
# a public header and this gate would have reported clean. The positions below
# are the ones the actual include directions already impose (verified, not
# assumed): `field` includes only `base` and is included by `backend` and
# `runtime`; `linalg` includes only `base` and is included by `num`.
LAYERS = [
    "base",
    "linalg",
    "model",
    "ir",
    "schedule",
    "num",
    "field",
    "backend",
    "transport",
    "runtime",
    "abi",
]

# Third-party headers that may appear in a PUBLIC header, per layer. Anything
# not listed is a violation wherever it appears under include/.
# Kokkos is permitted from `runtime` up because kernels are templated on an
# execution space in the public interface; `linalg` backends never are.
ALLOWED_PUBLIC_THIRD_PARTY = {
    "base": {"Kokkos_Core.hpp"},
    # The linalg facade exists so that a backend is swappable; naming one in
    # its public header would defeat that, so nothing third-party is allowed.
    "linalg": set(),
    "model": set(),
    "ir": set(),
    "schedule": set(),
    "num": {"Kokkos_Core.hpp"},
    # The lazy field pipeline's stages ARE Kokkos subviews -- that is the
    # substance of usability rule U3, not an implementation detail, so the view
    # type is in the public interface by design.
    "field": {"Kokkos_Core.hpp"},
    "runtime": {"Kokkos_Core.hpp"},
    "backend": {"Kokkos_Core.hpp", "Kokkos_Macros.hpp"},   # per-backend headers live under backend/<name>/
    "transport": set(),
    # The C ABI is the one surface that must compile for a consumer with none
    # of our dependencies installed.
    "abi": set(),
}

# Never in any public header, at any layer, outside its own backend directory.
FORBIDDEN_PUBLIC = re.compile(
    r"^\s*#\s*include\s*[<\"]("
    r"Eigen/|KokkosSparse|KokkosBatched|KokkosBlas"
    r"|mpi\.h|petsc|moab/|MBI|hdf5\.h|H5[A-Z]"
    r")"
)

# Invariant I11: fuse:: code never depends on upstream/ and never carries a
# parent's namespace, type prefix or file prefix. Applied to include/fuse/ AND
# src/ (not just public headers), because a legacy name in a .cpp is the same
# unfinished port. Once both parents are retired these rules cost nothing and
# stay as a guard against a stray copy-paste from the parents' repositories.
LEGACY_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](upstream/|soma/|coupling/karma_|karma_[a-z_]+\.h)')
LEGACY_NAMESPACE = re.compile(r'\bnamespace\s+(soma|KARMA)\b|\b(soma|KARMA)::')
LEGACY_TYPE_PREFIX = re.compile(r'\bKARMA[A-Z][A-Za-z0-9]*\b')
LEGACY_MACRO = re.compile(r'\b(KARMA|SOMA)_[A-Z_]+\b')
LEGACY_FILE_PREFIX = re.compile(r'^(karma_|soma_)')
LEGACY_EXTENSION = {".C"}
# `//` to end of line, capturing the prefix so it can be preserved -- the same
# blanking check-invariants.py's feature-macro check does, and for the same
# reason (REVIEW.md R-38 / BT-F12): a prose comment NAMING a legacy
# namespace/prefix/macro (e.g. a port's rename changelog, or porting.md's own
# examples) is not a USE of it, and a gate that fires on the name in prose
# pressures an author to reword the documentation rather than fix code. Block
# comments are deliberately not handled here either, for the same reason
# check-invariants.py gives: this project's prose is `///` lines, and a regex
# for `/* */` that got it wrong could blank real code.
COMMENT_LINE = re.compile(r"(^|[^:])(//[^\n]*)")
# --- end rules

INCLUDE_FUSE = re.compile(r"^\s*#\s*include\s*[<\"]fuse/([a-z_]+)/")
INCLUDE_ANY = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]")


def layer_of(path: Path) -> str | None:
    parts = path.parts
    if "fuse" not in parts:
        return None
    i = parts.index("fuse")
    return parts[i + 1] if i + 1 < len(parts) else None


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    include_dir = root / "include" / "fuse"
    if not include_dir.is_dir():
        print("check-layering: no include/fuse yet, nothing to check")
        return 0

    rank = {name: i for i, name in enumerate(LAYERS)}
    problems: list[str] = []

    for header in sorted(include_dir.rglob("*.h*")):
        layer = layer_of(header.relative_to(root))
        if layer is None or layer not in rank:
            continue
        rel = header.relative_to(root)
        for lineno, line in enumerate(header.read_text().splitlines(), 1):
            m = INCLUDE_FUSE.match(line)
            if m:
                dep = m.group(1)
                if dep in rank and rank[dep] > rank[layer]:
                    problems.append(
                        f"{rel}:{lineno}: layer '{layer}' includes from '{dep}', which is above it"
                    )
                continue
            if FORBIDDEN_PUBLIC.match(line):
                inc = INCLUDE_ANY.match(line).group(1)
                backend_scoped = layer == "backend" and len(rel.parts) > 3
                if not backend_scoped:
                    problems.append(
                        f"{rel}:{lineno}: third-party header '{inc}' in a public header (I7)"
                    )
                continue
            m = INCLUDE_ANY.match(line)
            if m and m.group(1).startswith("Kokkos"):
                if m.group(1) not in ALLOWED_PUBLIC_THIRD_PARTY.get(layer, set()):
                    problems.append(
                        f"{rel}:{lineno}: '{m.group(1)}' is not permitted in a public "
                        f"'{layer}' header"
                    )

    problems += check_legacy(root)

    for p in problems:
        print(f"error: {p}")
    print(f"check-layering: {len(problems)} violation(s)")
    return 1 if problems else 0


def check_legacy(root: Path) -> list[str]:
    """I11: no upstream include, no legacy namespace/prefix.

    test/ IS SCANNED, and was not until 2026-09-16. A stale
    `#include <soma/ir/lowering.hpp>` sat committed in test/parallel/ and
    compiled anyway, because building a parent subtree puts its include
    directory on the path -- so the test was quietly consuming a parent header
    while every gate reported clean, and it would have broken the moment the
    subtree retired. A test is fuse:: code for this purpose.

    examples/ is scanned from 2026-09-17 for the same reason, and it matters
    more there: an example is what a user copies. One that included a parent
    header or named a parent namespace would teach the wrong thing and would
    break when the subtree retires. The port of the two parent examples left
    exactly one such reference -- a doc comment naming
    `soma/runtime/component.hpp` -- which this gate could not see while
    examples/ was outside its scope.
    """
    problems: list[str] = []
    for area in (root / "include" / "fuse", root / "src", root / "test",
                 root / "examples"):
        if not area.is_dir():
            continue
        for f in sorted(area.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root)
            if f.suffix in LEGACY_EXTENSION:
                problems.append(f"{rel}: '.C' extension; FUSE sources are '.cpp' (I11)")
            if LEGACY_FILE_PREFIX.match(f.name):
                problems.append(f"{rel}: legacy file prefix; rename per docs/dev/porting.md (I11)")
            if f.suffix not in {".hpp", ".h", ".cpp", ".cc", ".C", ".inl", ".ipp"}:
                continue
            for lineno, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if LEGACY_INCLUDE.match(line):
                    problems.append(f"{rel}:{lineno}: fuse:: code includes from a parent tree (I11)")
                # Comments blanked (preserving column offsets) before the
                # namespace/prefix/macro checks: those three read as prose
                # naming the rename just as easily as a live use of it, and
                # LEGACY_INCLUDE above is anchored to a real #include and
                # cannot match inside a `//` comment's text.
                code_only = COMMENT_LINE.sub(
                    lambda m: m.group(1) + " " * len(m.group(2)), line)
                if LEGACY_NAMESPACE.search(code_only):
                    problems.append(f"{rel}:{lineno}: legacy namespace 'soma'/'KARMA' (I11)")
                if LEGACY_TYPE_PREFIX.search(code_only):
                    problems.append(f"{rel}:{lineno}: 'KARMA*' type prefix; the namespace carries it (I11)")
                if LEGACY_MACRO.search(code_only):
                    problems.append(f"{rel}:{lineno}: legacy macro prefix; use FUSE_* (I11)")
    return problems


if __name__ == "__main__":
    sys.exit(main())
