# The MOAB backend's covering-set intersection

`MoabMeshBackend::build_remap_halo` (`docs/user/geometry.md`'s own "covering set" section)
and `fuse::backend::assemble` (`src/backend/intx_weights.cpp`) both sit on top of MOAB's own
`Intx2MeshOnSphere`/`Intx2MeshInPlane` intersection classes. This page is the MOAB-half
counterpart to `docs/dev/native-halo.md`: the native backend's remap halo is a box-box overlap
test with no MOAB in the build at all; the MOAB backend's own intersection runs entirely inside
MOAB's own `moab::Intx2Mesh` family, which this page describes only as far as `assemble()`'s own
call site needs.

## Two drivers over the same cell-pair geometry

MOAB ships two different DRIVERS that decide WHICH (source, target) cell pairs to test for
overlap, sharing the identical underlying intersection math
(`computeIntersectionBetweenTgtAndSrc`, `findNodes`, the same `SourceParent`/`TargetParent`/
`Counting` output tags):

- `Intx2Mesh::intersect_meshes` -- an ADVANCING FRONT. An outer loop re-seeds, once per
  disconnected island of unresolved target cells, via a KD-tree proximity search that finds one
  confirmed intersecting (source, target) pair; an inner pair of loops then walks outward from
  that seed through target adjacency (`tgtNeighTag`) and source adjacency (`srcNeighTag`),
  expanding only to cells ADJACENT -- sharing a side, via MOAB's own vertex-handle identity -- to
  a cell already confirmed to intersect something.
- `Intx2Mesh::intersect_meshes_kdtree` -- a single KD-tree built over the full covering set,
  queried directly, by spatial proximity, for every target cell in turn. No adjacency walk, so no
  dependency on vertex-handle identity between any two cells.

MOAB's own parallel remapper (`src/Remapping/TempestRemapper.cpp`) offers exactly this choice for
covering-set/target-set intersection on the sphere, naming the two "the advancing-front
propagation algorithm" and "the Kd-tree search algorithm"; its own sphere intersection test
(`test/intx_on_sphere_test.cpp`) uses the KD-tree path as its brute-force fallback.

## Why the front misses adopted cells on a partitioned mesh

MOAB's adjacency tags are built from shared VERTEX HANDLES, not from geometric coincidence. A
source cell adopted into a rank's local covering set through MOAB's parallel exchange is not
guaranteed to share vertex handles with a cell already resident on that rank, even when it sits at
the identical geometric location -- so the advancing front may never reach it, no matter where it
is seeded. This mechanism is inferred from the kernel's adjacency source and a probe
(cells present and geometrically covered on a rank, intersected nothing there); where the adjacency
tag is populated for adopted cells was not read to the end. Worse, a target cell is marked resolved the moment ANY intersecting source cell is
found for it, not when its full area is accounted for, so a target cell satisfied in part by a
front-reachable source cell is never revisited for the remaining, adjacency-unreachable, adopted
source cell that would complete its coverage.

This was diagnosed, not assumed: a probe
(`trace_remap_halo_probe`, `point_in_convex_spherical_polygon` -- a point-in-convex-spherical-
polygon test built with zero shared code path with MOAB's own intersection machinery, specifically
so a bug in that machinery could not make the diagnostic agree with itself) found, on the real
`wholeATM_T.h5m`/`recMeshOcn.h5m` sphere pair at 2 and 3 ranks, source cells that were present on
the rank whose own local target cells geometrically contained them, yet intersected with nothing
there.

`fuse::backend::assemble` therefore calls `Intx2Mesh::intersect_meshes_kdtree`, not
`intersect_meshes`, unconditionally. Both are inherited unchanged by `Intx2MeshOnSphere` from the
base `Intx2Mesh` (neither is overridden), dispatched correctly through the existing
`std::unique_ptr<moab::Intx2Mesh>` base pointer regardless of the concrete worker type, and need
nothing `assemble()` does not already provide (`FindMaxEdges`, `set_error_tolerance`); the kd-tree
driver builds its own tags internally rather than depending on `createTags()`. This is a
correctness choice, not a measured performance trade-off: FUSE's own standing preference is
conservation correctness over an efficiency difference that has not been measured either way.

## Diagnosing a coverage gap

`FUSE_REMAP_HALO_TRACE=1` enables, on every MOAB `assemble()` call, both
`trace_remap_halo_coverage` (per-rank before/after-fold zero-coverage counts, up to 5 uncovered
cells' centroid/global-id/target-box attribution, adopted-cell-never-intersected counts, a
coverage-ratio histogram) and `trace_remap_halo_probe` (a per-`(gid, owner_rank)` cross-rank
probe: whether the cell is present locally, what area MOAB's own intersection computed for it
there, and what the independent point-in-polygon test computes for it there). Both print to
`stderr`, prefixed `FUSE_REMAP_HALO_TRACE`, and both run collectively on every rank whenever the
communicator has more than one rank -- never conditioned on whether that specific rank has
something to report, so a traced run cannot hang on a mismatched collective the way a
rank-conditional one could.

`RemapWeights::min_src_coverage`/`min_tgt_coverage` (Allreduce-MIN over every owned cell on every
rank) is the production, always-on signal: a single zero-coverage cell anywhere drives it to 0,
which is what a conservative channel's own build already refuses on
(`E-REMAP-NOT-CONSERVATIVE`-shaped errors in `assemble()`); the trace above exists to localise
WHICH cell and why, not to replace that check.
