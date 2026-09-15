# Imported parent trees

Byte-identical copies of the two frozen parents, extracted with `git archive` from the
`fuse-import-0` tag in each. **Do not edit anything in this directory.**

| Directory | Parent | Tag | Files |
|---|---|---|---|
| `soma/` | the coupling compiler and runtime | `fuse-import-0` | 181 |
| `karma/` | the coupling spine, plus its finite-element stack | `fuse-import-0` | 512 |

The exact refs are in [`../UPSTREAMS.lock`](../UPSTREAMS.lock).

## Why they are here unmodified

Invariant I8: upstream code lands byte-identical with its own suite green **before** anything
changes, so that every later transformation can be checked against a known-good source. An import
that had to be patched to build is not an import, and its test results would no longer be evidence
about the code we took.

Consequences, all deliberate:

- `.clang-format` here sets `DisableFormat`, and `.clang-tidy` disables every check. Reformatting
  would break byte-identity, which is the one property this directory exists to have.
- The layering and invariant gates do not scan `upstream/`. These trees do not follow this
  repository's conventions and are not expected to.
- Each tree keeps its own `CMakeLists.txt` and is configured as a subproject.

## Verifying

```bash
tools/verify-import.py             # sha256 against upstream/MANIFEST.sha256
tools/verify-import.py --parents   # re-derive from `git archive` and compare
```

The second form is the real proof and needs the parent repositories present. Both run in CI; only
the first can run without the parents.

## How this directory empties

Each stage from S1 ports one area into `include/fuse/` and `src/`, and records the corresponding
upstream paths as `superseded` in `UPSTREAMS.lock`. When a parent's every path is superseded, its
subtree is removed and the lock records the final state. `upstream/` is scaffolding with a planned
end, not a vendor directory.
