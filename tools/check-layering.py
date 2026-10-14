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
    # `search` (fuse::search, the ArborX-backed spatial index) sits directly
    # below `backend`, on purpose: the ArborX track's task explicitly places
    # it there so that `backend` CAN depend on it (the reverse of
    # `partition`'s own mesh_adapter.hpp, which reaches UP into `backend`
    # instead). `box_index.hpp` is still pimpl'd against ArborX itself (no
    # `#include <ArborX.hpp>` anywhere under include/) and `distributed.hpp`
    # never names mpi.h, but `box_index.hpp` DOES now name `Kokkos_Core.hpp`
    # (docs/dev/search.md's "Execution space" section, mesh-search track):
    # `BoxIndex::build`/`query_points`/`query_nearest` gained
    # `Kokkos::View<..., Kokkos::DefaultExecutionSpace::memory_space>`
    # overloads so a device-resident caller never round-trips through the
    # host, the same reason `field`/`num`/`runtime`/`partition` below allow
    # it -- a Kokkos-templated public API, not an implementation detail.
    # `mesh_adapter.hpp`'s adapter stays a template constrained by a
    # structural (duck-typed) concept over `fuse::MeshView`'s shape rather
    # than an `#include "fuse/backend/...")` of the real type -- the only way
    # to consume that shape from BELOW `backend` at all.
    "search",
    "backend",
    # `partition` (fuse::partition, the Kokkos-view RCB port) sits directly
    # above `backend`: its core (rcb.hpp/median.hpp/coord_transform.hpp)
    # depends only on `base`, and its MeshView adapter (mesh_adapter.hpp)
    # depends on `backend` for the mesh interface -- nothing above `backend`
    # needs it yet, so this is the lowest position that satisfies both.
    "partition",
    "transport",
    "runtime",
    "abi",
    # `petsc` (fuse::petsc, the S7 Newton seam's second substrate,
    # PetscToken track, 2026-10-11) sits at the TOP: it depends on
    # `runtime` (fuse::LinearizationToken, fuse/runtime/component.hpp) and
    # `num` (fuse::num::PreconditionerBlock/BlockJacobiPreconditioner,
    # reused rather than reimplemented for the block-Jacobi PCSHELL
    # composition) -- both already well below `abi` -- and nothing in the
    # library depends on it (no existing layer, including `abi`, names
    # `fuse/petsc/...`). Placing it after every existing layer, rather
    # than splicing it in just above `runtime`, needs no argument about
    # exactly where between `runtime` and `abi` it belongs (it depends on
    # layers well below either), and guarantees by construction that nothing
    # already-ranked could be made to depend on it by a future layer
    # insertion between `petsc` and whatever used to be the topmost entry.
    "petsc",
]

# Third-party headers that may appear in a PUBLIC header, per layer. Anything
# not listed is a violation wherever it appears under include/.
# Kokkos is permitted from `runtime` up because kernels are templated on an
# execution space in the public interface; `linalg` backends never are.
ALLOWED_PUBLIC_THIRD_PARTY = {
    # error.hpp's device-callable FUSE_DEBUG_ASSERT needs only the two light
    # headers (Kokkos_Macros for KOKKOS_IF_ON_*, Kokkos_Abort for
    # Kokkos::abort), not Kokkos_Core; both are allowed so the base layer does
    # not pay the full core include for one macro (pass-3 review, N-base-model-3).
    "base": {"Kokkos_Core.hpp", "Kokkos_Macros.hpp", "Kokkos_Abort.hpp"},
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
    # box_index.hpp is pimpl'd against ArborX (no <ArborX.hpp> under
    # include/) and distributed.hpp is mpi.h-free by construction (both
    # header comments explain why), but box_index.hpp's device-resident
    # Kokkos::View overloads (docs/dev/search.md, "Execution space") put
    # Kokkos_Core.hpp in this layer's public surface by design, the same
    # justification `field`/`num`/`runtime`/`partition` already carry.
    "search": {"Kokkos_Core.hpp"},
    "runtime": {"Kokkos_Core.hpp"},
    "backend": {"Kokkos_Core.hpp", "Kokkos_Macros.hpp"},   # per-backend headers live under backend/<name>/
    # rcb_partition/weighted_median/apply_coord_transform are all templated
    # on an execution space, in the public interface, the same reason `num`
    # and `field` need the core header.
    "partition": {"Kokkos_Core.hpp"},
    "transport": set(),
    # The C ABI is the one surface that must compile for a consumer with none
    # of our dependencies installed.
    "abi": set(),
    # fuse/petsc/*.hpp name no PETSc type at all (FORBIDDEN_PUBLIC's own
    # `petsc` pattern matches `#include <petsc...>`/`"petsc..."` at every
    # layer, no exception for this one) -- every PETSc call lives in
    # src/petsc/*.cpp, behind plain std::span/DistVector shapes. Nothing
    # third-party is named in this layer's public surface at all.
    "petsc": set(),
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


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    include_dir = root / "include" / "fuse"
    if not include_dir.is_dir():
        print("check-layering: no include/fuse yet, nothing to check")
        return 0

    rank = {name: i for i, name in enumerate(LAYERS)}
    problems: list[str] = []

    for header in sorted(include_dir.rglob("*.h*")):
        rel = header.relative_to(root)
        rel_in_fuse = header.relative_to(include_dir)
        if len(rel_in_fuse.parts) < 2:
            # A header directly under include/fuse/, in no layer directory at
            # all -- not what this loop checks (there is no "fuse/<layer>/"
            # to rank), and not the BC-9 gap: that gap is a directory this
            # list does not know, not the absence of one.
            continue
        layer = rel_in_fuse.parts[0]
        if layer not in rank:
            # This is the gap BC-9 named: LAYERS' own header comment says
            # "every directory under include/fuse MUST appear here", but
            # until now an unlisted one was silently skipped below instead
            # of failing loudly -- exactly how `field`, `linalg` and `abi`
            # went unchecked for a release each (see the LAYERS comment).
            # A `continue` here would repeat that: the gate must fail
            # instead, on the fact that a directory exists that it cannot
            # rank, before it gets a chance to miss a real violation in it.
            problems.append(
                f"{rel}: 'include/fuse/{layer}/' is not a directory LAYERS "
                f"knows about -- add it to tools/check-layering.py's LAYERS "
                f"(with its position argued, as every other entry's comment "
                f"does) before this header can be checked for anything (I7)"
            )
            continue
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

    # The include-direction check above only ever walked include/fuse's
    # PUBLIC headers; a src/*.cpp translation unit including a higher
    # layer's public header was invisible to both this gate and the link
    # graph (a .cpp does not appear in anyone else's dependency chain, so
    # nothing downstream would ever notice the inversion either). Same rank
    # table, same direction rule; the third-party-header and Kokkos-in-
    # public-header rules stay public-header-only, since a .cpp is never
    # part of a layer's public surface.
    src_dir = root / "src"
    if src_dir.is_dir():
        for tu in sorted(src_dir.rglob("*.cpp")):
            rel = tu.relative_to(root)
            rel_in_src = tu.relative_to(src_dir)
            if len(rel_in_src.parts) < 2:
                continue
            layer = rel_in_src.parts[0]
            if layer not in rank:
                continue
            for lineno, line in enumerate(tu.read_text(errors="replace").splitlines(), 1):
                m = INCLUDE_FUSE.match(line)
                if not m:
                    continue
                dep = m.group(1)
                if dep in rank and rank[dep] > rank[layer]:
                    problems.append(
                        f"{rel}:{lineno}: layer '{layer}' includes from '{dep}', which is above it"
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
