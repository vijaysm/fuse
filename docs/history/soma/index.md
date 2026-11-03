# Soma design history

Historical design, planning and handoff documents for `soma::`, the coupler library imported
byte-identically as `upstream/soma` (parent repository `coupe`, commit `e52b981`, imported
2026-09-14 per `UPSTREAMS.lock`) and in the process of being retired as its behaviour is ported
natively into `fuse::`. Every page here is preserved from that commit with only the markup-only
edits listed under "Edits to the preserved text" below; none describes FUSE's current
architecture. See [](../../dev/soma-import.md) for the per-file and per-case record
of what was ported, superseded, or remains an open gap, and [](../../dev/porting.md) for the
general porting procedure.

```{toctree}
:maxdepth: 1

specs/2026-09-02-soma-design
specs/2026-09-11-comm-graph-design
plans/2026-09-02-soma-implementation
plans/soma-phase-0
plans/soma-phase-1
plans/soma-phase-2
plans/soma-phase-3
plans/soma-phase-4
plans/soma-phase-5
plans/soma-phase-6
plans/soma-phase-7
plans/soma-phase-8
BUILDING
HANDOFF
```

## Edits to the preserved text

Each page begins with a provenance note and is otherwise identical to its `upstream/soma/docs/`
original except for the hunks below, which exist only because the page now builds under
`sphinx -W` in a different directory: five links that pointed at sibling files of the retired tree
became inline code naming the upstream path, one anchor gained an explicit MyST target, and one
elision placeholder in a CMake block became a CMake comment so that the block is valid CMake.
No wording was changed. The listing is the output of `diff` between each original and the page
below its provenance note.

```diff
--- upstream/soma/docs/BUILDING.md
+++ docs/history/soma/BUILDING.md
@@ -37 +37 @@
-A [`CMakePresets.json`](../CMakePresets.json) at the repo root defines both
+A `CMakePresets.json` (upstream/soma/CMakePresets.json) at the repo root defines both
@@ -119 +119 @@
-[`cmake/SomaOptions.cmake`](../cmake/SomaOptions.cmake)), so the two
+`cmake/SomaOptions.cmake` (upstream/soma/cmake/SomaOptions.cmake)), so the two
@@ -140 +140 @@
-[`cmake/FindSomaMOAB.cmake`](../cmake/FindSomaMOAB.cmake) locates MOAB via
+`cmake/FindSomaMOAB.cmake` (upstream/soma/cmake/FindSomaMOAB.cmake) locates MOAB via
@@ -523 +523 @@
-[`cmake/FindSomaLowFive.cmake`](../cmake/FindSomaLowFive.cmake) locates the
+`cmake/FindSomaLowFive.cmake` (upstream/soma/cmake/FindSomaLowFive.cmake) locates the
```

```diff
--- upstream/soma/docs/HANDOFF.md
+++ docs/history/soma/HANDOFF.md
@@ -52 +52 @@
-| [`docs/plans/soma-phase-{0..8}.md`](plans/) | 58 tasks with literal code and tests. All complete. |
+| `docs/plans/soma-phase-{0..8}.md` (this directory's plans/ subfolder) | 58 tasks with literal code and tests. All complete. |
```

```diff
--- upstream/soma/docs/plans/soma-phase-5.md
+++ docs/history/soma/plans/soma-phase-5.md
@@ -833 +833 @@
-  ...
+  # ...
```

```diff
--- upstream/soma/docs/specs/2026-09-11-comm-graph-design.md
+++ docs/history/soma/specs/2026-09-11-comm-graph-design.md
@@ -386,0 +387 @@
+(test-13-correction)=
```
