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
refusal you discover in a log; it does not build.

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
