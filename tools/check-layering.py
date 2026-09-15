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
LAYERS = [
    "base",
    "model",
    "ir",
    "schedule",
    "num",
    "runtime",
    "backend",
    "interface",
]

# Third-party headers that may appear in a PUBLIC header, per layer. Anything
# not listed is a violation wherever it appears under include/.
# Kokkos is permitted from `runtime` up because kernels are templated on an
# execution space in the public interface; `linalg` backends never are.
ALLOWED_PUBLIC_THIRD_PARTY = {
    "base": {"Kokkos_Core.hpp"},
    "model": set(),
    "ir": set(),
    "schedule": set(),
    "num": {"Kokkos_Core.hpp"},
    "runtime": {"Kokkos_Core.hpp"},
    "backend": {"Kokkos_Core.hpp"},   # per-backend headers live under backend/<name>/
    "interface": set(),
}

# Never in any public header, at any layer, outside its own backend directory.
FORBIDDEN_PUBLIC = re.compile(
    r"^\s*#\s*include\s*[<\"]("
    r"Eigen/|KokkosSparse|KokkosBatched|KokkosBlas"
    r"|mpi\.h|petsc|moab/|MBI|hdf5\.h|H5[A-Z]"
    r")"
)
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

    for p in problems:
        print(f"error: {p}")
    print(f"check-layering: {len(problems)} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
