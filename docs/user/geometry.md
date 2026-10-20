# Geometry and the remap ladder

FUSE represents geometry natively, and what it can do with it depends on which
optional libraries the build found. The floor is deliberately useful on its
own: **arbitrary unstructured geometry with lowest-order conservative
projection, in the default build, with no optional dependency at all.**

| Build | Geometry | Conservative projection |
|---|---|---|
| default | FUSE's native simplicial mesh | lowest order, arbitrary geometry |
| `FUSE_ENABLE_MOAB=ON` | MOAB's mesh, used natively — no copy | as above, plus MOAB's search and partitioning |
| MOAB built with TempestRemap | as above | plus Earth-system remaps on the sphere, high order |

The lowest-order path is not a fallback that merely compiles. Because it is
always present, it is also the **differential oracle** for the other two: on a
problem all three can express, they are required to agree.

## The two dimensions

A mesh is characterised by two dimensions, not one:

- the **manifold dimension** is the intrinsic dimension of a cell — 2 for a
  triangle, 3 for a tetrahedron, 1 for an edge, 0 for a point;
- the **spatial dimension** is the dimension of the space the coordinates live
  in.

Their difference is the **codimension**, and it decides which operations are
meaningful rather than merely difficult:

| Codimension | Example | Normals |
|---|---|---|
| 0 | tetrahedra in 3D, triangles in 2D | none — the cells fill space |
| 1 | triangles in 3D, edges in 2D | exactly one per cell |
| 2 or more | edges in 3D | infinitely many |

Both dimensions are template parameters, so this is enforced when the code is
compiled rather than when it runs. Asking a volume mesh for a normal is not a
rejection you discover in a log; it does not build.

## Cells are simplices, and they remember their parent

Cells are triangles in 2D and tetrahedra in 3D. A hexahedral or polyhedral
mesh is decomposed, and **every simplex names the cell it came from**.

That parent map is load-bearing rather than bookkeeping. Conservation is
accounted per *user* cell: if a hexahedron becomes six tetrahedra, a
conservation sum must still balance over one cell, and a report that named
simplices would be answering a question nobody asked. `parent_cell_count()` is
the number to balance over; `simplex_count()` is an implementation detail of
how the geometry is stored.

Simplices were chosen over a canonical cell-type table for accuracy: clipping a
simplex assumes nothing about convexity, so a non-convex cell — or a hexahedron
whose quadrilateral faces are warped and therefore not planar — is handled
exactly, since a cell's intersection is exactly the sum of its simplices'
intersections.

## Measures are manifold measures

`measure()` returns the length, area or volume according to the *manifold*
dimension. A triangle in three-space has an area, and that is what you get. It
is computed from the Gram determinant of the edge matrix, which is the single
formula covering every dimension pair — a plain determinant of the 3D edge
matrix would report zero for that triangle.

A degenerate cell measures exactly zero, never a NaN. The distinction matters
because a NaN weight propagates through a conservation sum without announcing
itself, while a zero is visible as a cell that received nothing.

## What the mesh does not carry

Fields. The reference model this follows bundles point, cell and global data
into the mesh; FUSE already has a field layer carrying units, intent and
semantic shape, and a second home for a field would be two places to look for
one and two places for its units to disagree. The mesh carries geometry only.

## How the conservative map is built

For every target cell `j` and source cell `i`, the exact measure of their
intersection is computed by clipping simplices and taking analytic moments, and

```
W_ji = |T_j intersect S_i| / m_j
```

so a target value is the measure-weighted average of the source field over the
target cell. That one formula is both **constant-preserving** (rows sum to one
where the meshes overlap fully) and **conservative** (total mass is unchanged),
which is why there is no separate "conservative" switch: a first-order
supermesh map is conservative by construction. The audit fields on the weights
report how nearly it achieved both on the pair it was given, rather than
asserting it.

Measured on non-matching pairs over the unit domain, all residuals at rounding:

| Pair | Row defect (constants) | Column defect (mass) |
|---|---|---|
| 8 → 5 cells, 1D | < 1e-14 | < 1e-14 |
| 3×3 → 4×4 triangulated quads, 2D | < 1e-13 | < 1e-13 |
| 3×3×3 → 4×4×4 hexes, 3D | < 1e-13 | < 1e-13 |

Remapping a mesh onto itself returns exactly the identity: one entry per cell,
each equal to 1.

### Two things it rejects

**A surface pair.** The 2D kernel clips in the xy-plane. Handed a surface
embedded in three-space it would return the measure of its *shadow* — a
plausible wrong answer — so it rejects with `E-REMAP-NONPLANAR`. A surface pair
needs a parameterisation, or great-circle geometry on the sphere, which is what
the MOAB-enabled build configuration provides via TempestRemap.

**Anything above order 1.** Piecewise constant is what this path reconstructs.
Asking for high order without MOAB is rejected by name rather than quietly
served at order 1.

### Meshes at different radii

A spherical covering-set construction (MOAB's, above) needs its two meshes on
the SAME sphere. Two real files at genuinely different nominal radii -- an
atmosphere mesh at a unit radius and an ocean mesh at Earth radius in metres,
say -- are refused outright, `E-REMAP-GEOMETRY`, "not the same domain": the
library cannot know two different radii are meant to be the same sphere in
different units, so it never guesses.

**Rescaling is the driver's job, never the library's** (KNOWN_ISSUES D81).
`MoabMeshBackend::sphere_radius(MeshId[, ComponentId])` measures a mesh's
vertices as a sphere about the origin and reports the mean radius and the
largest single vertex's deviation from it, so a driver can decide what, if
anything, needs doing. `MoabMeshBackend::project_onto_unit_sphere(MeshId[,
ComponentId])` measures the same way, then -- unless the mesh is ALREADY on
the unit sphere to within its own tolerance -- rescales every vertex by
`1 / mean_radius` and reports what it did (`SphereProjectionReport`:
`mean_radius`, `applied_factor`, `already_unit`). Already-unit is a measured
no-op: no vertex coordinate is read back or rewritten.

Both come in two forms: a `MeshId`-only overload for the backend's default
decomposition, and a `(MeshId, ComponentId)` overload for a NAMED placement's
own, independently-decomposed copy (`docs/user/placement.md`'s own
per-placement model) -- `set_placements()` gives every declared processor set
its own Core with its own independently-read copy of every loaded mesh, so
rescaling one placement's copy does nothing to another's, and a driver
coupling two independently placed components over real files must call the
placement-aware overload on EACH placement whose copy it will use.

The library itself never calls either function: `build_remap_halo` keeps
refusing `E-REMAP-GEOMETRY` exactly as before on a mismatched pair, whether or
not a rescale was available. A driver loading real, differently-scaled
spherical files -- grep for `FUSE_SPHERE_ATM`/`FUSE_SPHERE_OCN` in this
tree's own test suite for a worked example -- queries or rescales explicitly,
after `load()` and before the first `build_remap_halo`/`build_weights` call,
collectively (every rank holding a piece of the mesh, or of that placement's
own share of it, in the same order).

### The covering set is intersected with a k-d-tree search, not an advancing front

MOAB offers two different algorithms for deciding which source and target cells in the covering
set actually overlap: an advancing front that walks cell adjacency outward from a small number of
seed pairs, and a k-d-tree search that tests every target cell against the covering set directly
by spatial proximity. FUSE's own `assemble()` always uses the k-d-tree search (KNOWN_ISSUES D82):
on a genuinely partitioned (multi-rank) mesh, the advancing front left source cells that were
present on a rank, and geometrically covered by that rank's own target cells, with no intersection
at all, while the same pair at one rank was covered completely; a target cell such a cell alone
would have covered is silently left short, with no error, only a lower delivered total than the
true one. The likely cause is adjacency: the front expands only through cells that share MOAB's
own vertex identity, which a cell adopted through the parallel exchange need not share with the
resident cells (inferred, see `docs/dev/moab-intersection.md`). The k-d-tree search walks no
adjacency and has no such dependency. This was measured directly on a real, partitioned sphere pair before the fix (the
delivered total low by roughly 28 times the derived tolerance) and is why a driver should always
trust `RemapWeights::min_src_coverage`/`min_tgt_coverage` -- not just whether `build_weights`
returned -- before relying on a conservative channel's own total. See `docs/dev/moab-intersection.md`
for the mechanism in full.

### A remap halo on a placement is adopted into THAT placement's own Core

`build_remap_halo`'s covering-set construction (above) runs inside whichever
`moab::Interface` the two meshes were resolved through -- the backend's
default Core when no placement is named, or a DECLARED placement's own,
independently-read Core (`set_placements()`, `docs/user/placement.md`)
otherwise. KNOWN_ISSUES D83 (owner decision 2026-10-19): the covering cells
that construction finds are adopted back into THAT SAME Core --
`MoabMeshBackend::adopt_remap_halo`, `has_remap_halo`, `remap_halo_size`,
`exchange_remap_halo` and `accumulate_remap_halo` all take an optional
trailing `ComponentId placement` (default: the unplaced decomposition,
matching every existing caller that does not pass one), and resolve every
MOAB call -- the GLOBAL_ID tag read, `add_entities`, the gather/scatter
exchange -- through that placement's own Core and communicator, never the
default Core's unconditionally. A `moab::EntityHandle` is only ever
meaningful within the ONE Core that allocated it; reading one back through
a DIFFERENT Core (what an earlier revision of this library did whenever the
delegated single-placement path happened to resolve through a NAMED
placement rather than the default) is undefined by MOAB's own contract, not
merely a bookkeeping slip, and is what the "no positive GLOBAL_ID" refusal
in KNOWN_ISSUES D81's own round 7/8/9 history was actually measuring.

The bookkeeping this adopts into (`has_remap_halo`'s own cache) is keyed by
`(source, target, dim, placement)`, not `(source, target, dim)` alone: the
default Core's own halo for a pair and a placement's own halo for the
IDENTICAL pair are two independent entries, and building one never makes
the other read as built. A driver (or the runtime's own weight builder)
that resolves a channel's mesh pair through a placement must pass that SAME
placement to `has_remap_halo`/`remap_halo_size` to find the halo
`build_remap_halo` actually adopted for it.

The collective agreement a halo adoption needs (every rank must agree
whether the covering set it found is trustworthy before any rank acts on
it) runs over the SAME communicator the covering-set construction itself
used -- a placement's own sub-communicator when one is named, never the
backend's base communicator on its behalf. A rank outside the named
placement has nothing to agree on with that placement's own members and
takes no part in it at all.

### A note on negligible overlaps

Two cells that merely share a face have an intersection of measure zero, but
clipping against a coplanar plane returns a rounding-level positive — measured
at up to 1.1e-16 of a cell, which on an identity pair produced 84 spurious
entries beside the 27 real ones. Contributions below 1e-12 of the target cell's
measure are dropped.

The reason is not accuracy: those entries could not move a double-precision
result. It is that **a matrix entry is a data dependency**. In parallel it makes
the target rank wait on whichever rank owns that source cell, for a
contribution of 1e-16. The floor is applied before the conservation audit, so
any mass it discards is reported in the residuals rather than hidden.

## What the conservative map is checked against

Conservation and constant preservation are necessary but **not sufficient**: a
map that sent mass to the wrong cells could still sum correctly along both
axes. The checks below are ordered by what they can catch that the previous one
cannot.

| Check | What it would catch | Result |
|---|---|---|
| Conservation and constant residuals | mass created or destroyed | < 1e-13 in 1D, 2D and 3D |
| Identity pair | any error at all in the clipper | exactly one entry per cell, each 1.0 |
| **Entrywise against closed form** | mass delivered to the *wrong* cell | worst entry error 2.8e-15 over 1331 entries on coprime 5→7 grids; no missing and no spurious entries |
| **Common affine map** | an axis-aligned assumption anywhere | weights change by 6.6e-15 under a rotation, shear and translation |
| **Refinement study** | a map that conserves but does not converge | observed order **1.05** over five refinements |
| Input winding | a dependence on vertex ordering, which the mesh contract does not fix | identical map from clockwise and counterclockwise input |
| **1e-10 slivers** | a hidden tolerance in the candidate search | slivers kept; smallest weight 4.0e-10 |

The closed-form check is the strongest available for the geometry: for two
axis-aligned grids the exact intersection volume is the product of the per-axis
overlaps, so every individual weight has a reference value rather than only the
row and column sums.

The refinement study is one-dimensional, deliberately. The candidate search is
currently a brute-force box test, and a five-level 3D study would take hours —
the 5×5×5 → 7×7×7 case alone is about 1.5 million clip calls. In 3D the kernel
is pinned against closed form instead, which says more about the geometry than
a convergence slope does. A 3D study becomes affordable when a spatial index
replaces the brute-force search.

## How a cell becomes simplices

Every backend presents its cells to the clipping kernel as simplices, so that
kernel has one geometric case instead of six. *Which* corners form which
simplex is not each backend's business: it is a fixed table in
`include/fuse/backend/cell_simplices.hpp`, covering edges, triangles, quads,
convex polygons, tetrahedra, pyramids, prisms and hexahedra.

The table is shared rather than duplicated for a reason that is easy to miss.
A hexahedron with **non-planar faces is not a polyhedron**: each choice of
diagonal triangulates those faces differently, so the two decompositions
enclose genuinely different solids with different volumes. Neither is wrong.
But it means "the volume of this cell" is not a property of the cell alone, and
two backends that decomposed the same mesh about different diagonals would
disagree about cell volumes — a conservation check between them would then
report a defect that is really a convention mismatch. One table removes that
possibility, and
`CellSimplices.ANonPlanarHexHasNoSingleVolumeAndThatIsWhyTheTableIsFixed`
records the fact.

The tables are `constexpr` and device-callable, so a kernel can decompose a
cell in place instead of asking the host for its corner list.
