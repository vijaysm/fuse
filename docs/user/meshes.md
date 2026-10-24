# Meshes, backends and partitioning

How FUSE sees a mesh, which backends exist, every generated mesh you can ask for, how a region
becomes a buffer size, and how a native mesh is distributed, searched and repartitioned. Read it
when you choose a `MeshDecl::source`, size a field, or need a mesh on more than one rank.
[](geometry.md) covers what is done *with* the geometry (the remap ladder); this page covers where
the geometry comes from.

```cpp
#include <fuse/backend/mesh_view.hpp>        // MeshView, MeshBackend, make_region_sizes
#include <fuse/backend/serial_backend.hpp>   // SerialMeshBackend
#include <fuse/backend/native_backend.hpp>   // fuse::native::NativeMeshBackend
#include <fuse/backend/moab_backend.hpp>     // MoabMeshBackend (FUSE_ENABLE_MOAB)
```

## Loading generated meshes

```cpp
fuse::SerialMeshBackend serial;
for (const char* src : {"generated:line:8", "generated:box:2x3x4"}) {
  if (auto r = serial.load(fuse::MeshDecl{.name = src, .source = src}); !r.ok()) {
    std::cerr << r.error().code << ": " << r.error().message << "\n";
    return 1;
  }
  const fuse::MeshView& v = *serial.view(serial.mesh_id(src).value()).value();
  std::cout << src << ": dimension " << v.dimension();
  for (int d = 0; d <= 3; ++d) std::cout << "  dim" << d << "=" << v.entity_count(d);
  std::cout << "  measure(cell 0) = " << v.measure(v.dimension(), 0) << "\n";
}
// A named boundary set of the box: faces on one side, or the whole skin.
const fuse::MeshId box = serial.mesh_id("generated:box:2x3x4").value();
for (const char* set : {"left", "top", "wall"}) {
  auto n = serial.region_size(fuse::Region{.name = set, .mesh = box, .entity_set = set, .dimension = 2});
  std::cout << "box " << set << ": " << n.value() << " faces\n";
}
auto bad = serial.load(fuse::MeshDecl{.name = "bad", .source = "generated:quad:4x4"});
std::cout << bad.error().code << ": " << bad.error().message << "\n";
```

```text
generated:line:8: dimension 1  dim0=9  dim1=8  dim2=0  dim3=0  measure(cell 0) = 0.125
generated:box:2x3x4: dimension 3  dim0=60  dim1=0  dim2=52  dim3=24  measure(cell 0) = 0.0416667
box left: 12 faces
box top: 6 faces
box wall: 52 faces
E-MESH-SOURCE: unknown generated kind 'quad'
```

The native backend accepts a wider grammar and runs on one process with `(rank, num_ranks) = (0, 1)`:

```cpp
fuse::native::NativeMeshBackend native(/*rank=*/0, /*num_ranks=*/1);
for (const char* src : {"generated:line:8", "generated:tri:3x2", "generated:quad:3x2",
                        "generated:cubedsphere:2", "generated:hexbox:2x2x2",
                        "generated:tetbox:2x2x2", "generated:prismpyramidmix:2x2x2",
                        "generated:polyhedral:2x2x2"}) {
  if (auto r = native.load(fuse::MeshDecl{.name = src, .source = src}); !r.ok()) { /* ... */ }
  const fuse::MeshView& v = *native.view(native.mesh_id(src).value()).value();
  const int d = v.dimension();
  double total = 0.0;
  for (std::size_t i = 0; i < v.entity_count(d); ++i) {
    if (v.is_owned(d, i)) total += v.measure(d, i);      // owned only: ghosts would double count
  }
  std::cout << src << ": dim " << d << ", " << v.entity_count(d) << " cells ("
            << v.owned_count(d) << " owned), " << v.entity_count(0) << " vertices, measure "
            << total << "\n";
}
```

```text
generated:line:8: dim 1, 8 cells (8 owned), 9 vertices, measure 1
generated:tri:3x2: dim 2, 12 cells (12 owned), 12 vertices, measure 1
generated:quad:3x2: dim 2, 6 cells (6 owned), 12 vertices, measure 1
generated:cubedsphere:2: dim 2, 24 cells (24 owned), 54 vertices, measure 11.0557
generated:hexbox:2x2x2: dim 3, 8 cells (8 owned), 27 vertices, measure 1
generated:tetbox:2x2x2: dim 3, 48 cells (48 owned), 27 vertices, measure 1
generated:prismpyramidmix:2x2x2: dim 3, 32 cells (32 owned), 31 vertices, measure 1
generated:polyhedral:2x2x2: dim 3, 8 cells (8 owned), 27 vertices, measure 1
```

Expected output (exact digits can vary by platform).

## What a backend provides: `MeshView`

Every backend answers the same questions through `fuse::MeshView`. The core knows only counts,
ownership, ids and geometry; no third-party type appears in the interface.

| Method | Meaning |
|---|---|
| `dimension()` | the top topological dimension: 1 for a line, 2 for a surface, 3 for a volume |
| `entity_count(dim)` | entities of that dimension visible on this rank, owned plus ghost; 0 for a dimension the mesh does not represent |
| `owned_count(dim)`, `is_owned(dim, i)` | how many, and which, local entities this rank owns |
| `centroid(dim, i)` | a `Point{x, y, z}`; unused coordinates are 0 |
| `measure(dim, i)` | physical length, area or volume, unsigned, additive over a tiling |
| `global_id(dim, i)` | unique across ranks within one dimension; sparse, unordered, not an index |
| `simplex_count(dim, i)`, `simplex_points(dim, i, s, out)` | the cell as triangles or tetrahedra, for the clipping kernel |
| `cell_vertices(dim, i)` | the cell's shape and corner vertex indices in canonical order, for nodal interpolation; `E-MESH-NO-CELL-VERTICES` where a backend has none |

Four rules are easy to get wrong on a distributed mesh. `entity_count` is local, never global; a
global quantity is a reduction over owned entities. Owned entities are not a prefix of the local
numbering: test with `is_owned`, do not loop to `owned_count`. A global id must never subscript an
array. And `measure` varies with `i` on an unstructured mesh, so a conservation check must read it
per entity. `is_owned`, `centroid`, `measure` and `global_id` throw `std::out_of_range` on a bad
`dim` or `i`.

The backend itself, `fuse::MeshBackend`, offers `load(const MeshDecl&)`, `mesh_id(name)`,
`view(MeshId)`, `region_size(const Region&)`, the per-placement `placed_view(id, ComponentId)` and
`placed_region_size(r, ComponentId)` ([](placement.md)), and `name()`. The n-th successful `load`
defines `MeshId{n}`; always resolve by name with `mesh_id()` rather than assuming the graph's and the
backend's numbering line up (`E-MESH-NAME` for an unknown name, `E-MESH-NAME-AMBIGUOUS` when two
loaded meshes share one).

## Generated sources

`MeshDecl::source` is either `generated:<kind>:<dims>` or a file path. Dimensions are positive
decimal counts joined with `x`; a count that would overflow is `E-MESH-SIZE`, an unknown kind or a
malformed count is `E-MESH-SOURCE`.

### Serial and MOAB backends

| Source | Mesh | Dimensions represented |
|---|---|---|
| `generated:line:N` | unit interval, N segments | 1 (cells), 0 (vertices) |
| `generated:box:NxMxP` | unit cube, N·M·P hexahedra | 3 (cells), 2 (the boundary skin only), 0 |

`box` needs exactly three counts (`2x2x2x2` is refused). The serial backend accepts nothing but
`generated:` sources; a path is `E-MESH-SOURCE`. The MOAB backend uses the identical grammar for
generated sources and passes any other string to MOAB as a file.

### Native backend

| Source | Mesh |
|---|---|
| `generated:line:N` | unit interval, N segments |
| `generated:tri:NxM` | unit square, N·M quads each split into 2 triangles |
| `generated:quad:NxM` | unit square, N·M quads |
| `generated:cubedsphere:N` | 6 panels of N·N quads with vertices on the unit sphere |
| `generated:hexbox:NxMxP` | unit cube, N·M·P hexahedra |
| `generated:tetbox:NxMxP` | unit cube, 6 tetrahedra per hex (Kuhn split) |
| `generated:prismpyramidmix:NxMxP` | unit cube; a hex at even i+j+k becomes 2 prisms, at odd i+j+k 6 pyramids around an added centroid |
| `generated:polyhedral:NxMxP` | unit cube, N·M·P hexahedra stored as general polyhedra (explicit face lists) |

Generation is distributed: each rank builds its own slice, split along the outermost extent, and
global ids do not depend on the rank count. Any other source string is a file path read by the
native reader (below). `fuse::native::generate_native_mesh(spec, rank, num_ranks)` exposes the
generator directly (without the `generated:` prefix) and returns `MeshArrays` that
`NativeMeshView::create_from_host(arrays, rank)` turns into a view.

## Regions and buffer sizes

A `fuse::Region` names a mesh, an entity set and a dimension:

```cpp
fuse::Region{.name = "wall", .mesh = m, .entity_set = "wall", .dimension = 2};
fuse::Region{.name = "cells", .mesh = m};          // entity_set "" (whole mesh), dimension -1 (mesh dim)
```

`entity_set` empty or `"all"` means the whole mesh at the given dimension. The serial backend also
knows the six sides of a box, `left`/`right` (x), `front`/`back` (y), `bottom`/`top` (z), and
`wall` for the whole skin; all are dimension-2 face sets, a stated dimension other than 2 is
`E-MESH-DIM`, and asking a 1D line for one is `E-MESH-SET`. The native backend accepts only the
whole mesh (`E-MESH-SET` otherwise). The MOAB backend resolves a name against the file's `NAME`-tagged
sets, or the tagged-set spellings `sideset_<n>`, `neumann_<n>`, `nodeset_<n>`, `dirichlet_<n>`,
`block_<n>` and `material_<n>`; a name matching several sets is `E-MESH-SET`. Other refusals:
`E-MESH-ID` for a mesh id the backend never loaded and `E-MESH-DIM` for a dimension the mesh does
not represent.

`fuse::make_region_sizes(graph, dag, backend)` is the function every program passes to
`schedule()` ([](quickstart.md)). It returns a callback that, for each region and placement,
resolves the mesh by name, decides whether the region's fields are cell-sized or nodal
([](component-interface.md#nodal-and-cell-fields-and-what-count-must-be)), and asks
`placed_region_size()` or the placed view's vertex count. Before building that callback it
executes every remap-halo instruction in the DAG, collectively, so that a cross-rank weighted
channel's buffers are sized from the widened mesh. `RegionSizes` has no error channel, so a region
that cannot be sized throws `std::runtime_error` carrying the backend's code and message rather
than returning a plausible number.

## Native mesh files

`fuse/backend/native_mesh_io.hpp` reads and writes a mesh as a writer-defined subset of legacy
VTK ASCII `UNSTRUCTURED_GRID`, one file per rank:

```cpp
fuse::native::write_native_mesh_file(path, view, rank);          // Result<void>
auto arrays = fuse::native::read_native_mesh_file(path);         // Result<MeshArrays<HostSpace>>
```

Polyhedra use `VTK_POLYHEDRON` (type 42). Ownership and ids travel as `POINT_DATA` / `CELL_DATA`
scalars (`GlobalPointID`, `PointOwner`, `PointIsOwned`, `GlobalCellID`, `CellOwner`,
`CellIsOwned`), stored as doubles that are exact up to 2⁵³, and a `#` comment line records the
topological dimension. The reader accepts only files this writer produced; it is not a general VTK
reader. I/O errors are `E-MESH-IO`, and a file whose connectivity is inconsistent fails the
builder's own checks (`E-MESH-CONNECTIVITY`, `E-MESH-POLYGON-DEGENERATE`,
`E-MESH-POLYGON-FACE-MISMATCH`, `E-MESH-CELL-NOT-STAR-SHAPED`).

## The native distributed mesh

`fuse::native::NativeMeshBackend(rank, num_ranks)` is MPI-free in its public interface: a driver
with a communicator passes its rank and size as plain integers and, for operations that must
communicate, the Fortran handle through `set_communicator(CommHandle)`. Three things happen on a
multi-rank native mesh.

**Ownership.** Every local entity is owned by exactly one rank; the rest are ghosts whose values
are copies. `is_owned` is the test. Generated meshes assign vertex owners deterministically from the
cell split, so a global id names the same entity on every rank.

**Remap halo.** A conservative or nodal channel between two meshes that are partitioned differently
needs, on each target rank, the source cells that overlap its own. The native backend builds that
covering set itself: a box-overlap search across ranks, packing of the needed cells, and merging
them into the target rank's mesh as non-owned local entities. This runs inside
`make_region_sizes()` and needs nothing from you beyond an MPI build and an initialised MPI; a
communicator of more than one rank in a build without MPI is `E-MESH-NO-MPI`, and one before
`MPI_Init` is `E-MESH-MPIINIT`. A peer that failed its half of the exchange is reported as
`E-MESH-HALO-PEER-REFUSED` on the ranks that succeeded, and a merge failure as `E-MESH-HALO-MERGE`.
The halo's ghost pattern is available afterwards through `has_remap_halo`, `remap_halo_pattern`
and `remap_halo_ghost_offset`, which the Runner uses to accumulate contributions back to owners.

**Migration.** `fuse::native::migrate(view, dest, comm)` moves every owned cell to the rank named in
`dest` (one entry per owned cell, `E-MESH-MIGRATE-INCOMPLETE` otherwise; a rank outside the
communicator is `E-MESH-MIGRATE-RANGE`), re-resolves vertex ownership and returns the new
`NativeMeshView`. Global ids, cell count and total measure are conserved across a migration.

## Partitioning

`fuse/partition/` provides recursive coordinate bisection over any mesh's centroids, and a
one-call repartition for native meshes:

```cpp
auto arrays = fuse::native::generate_native_mesh("hexbox:3x3x3", /*rank=*/0, /*num_ranks=*/1);
auto view = fuse::native::NativeMeshView::create_from_host(arrays.value(), 0);
fuse::partition::RepartitionOptions opts;             // comm = kNullComm: one process
auto same = fuse::partition::repartition(*view.value(), opts);
std::cout << "repartition on one process: " << same.value()->owned_count(3) << " cells owned\n";
opts.rcb.num_parts = 2;
auto refused = fuse::partition::repartition(*view.value(), opts);
std::cout << refused.error().code << "\n";
```

```text
repartition on one process: 27 cells owned
E-PART-REPARTITION-NUM-PARTS
```

- `fuse::partition::centroids_of(const MeshView&, CommHandle = kNullComm)` extracts centroids and
  global ids (`E-PART-MIXED-DIM`, `E-PART-NO-ENTITIES`).
- `fuse::partition::rcb_partition<ExecutionSpace>(RcbOptions, centroids, weights, gids)` returns an
  `RcbResult{part_id, num_parts_used}`: an assignment only, no data movement. `RcbOptions` has
  `num_parts` (0 means the communicator size), `median_buckets`, `median_refine_iters`,
  `coord_transform` (`Identity` or a gnomonic projection with `gnomonic` options, for spherical
  meshes) and `comm`. The options are agreed across ranks first; a disagreement is `E-PART-OPTS`.
  Other refusals: `E-PART-NO-MPI`, `E-PART-NUM-PARTS`, `E-PART-SIZE-MISMATCH`, `E-PART-WEIGHT`.
- `fuse::partition::repartition(view, RepartitionOptions{rcb, comm})` chains centroids, RCB with
  uniform weights and `native::migrate`, always one part per rank; any other `num_parts` is
  `E-PART-REPARTITION-NUM-PARTS`.

`fuse_example_layout_driver --mesh <spec> [--migrate]` runs exactly this on real ranks: it builds
`comp1`'s mesh on its own processor set, optionally migrates it onto `comp2`'s by RCB, and
`--check` confirms that cell count, total measure and id uniqueness survived. Any of the native
generated kinds is accepted for `<spec>` (without the `generated:` prefix). `--migrate` requires
`--mesh`, and `--coupler` is refused together with `--mesh`. [](layout-driver.md) and
[](layouts.md) document the driver.

## The MOAB backend

`fuse::MoabMeshBackend` (`name()` returns `"moab"`) needs `-DFUSE_ENABLE_MOAB=ON`, which requires
MPI ([](build.md)). It adds:

- reading any file format the linked MOAB reads (`.h5m` and the rest of MOAB's reader set) through
  `moab::Core::load_file`; when both MOAB and FUSE were built with MPI, a partitioned file is
  read in parallel (`PARALLEL=READ_PART` with shared-entity resolution). A failed read is
  `E-MOAB-LOAD`;
- MOAB's own entity sets as regions (above), `owned_entities(...)` and `partition_rank_of(...)`;
- `set_communicator(int fortran_comm_handle)` and `set_placements(std::vector<RankSet>)`, which
  gives every declared processor set its own independently read copy of each mesh
  ([](placement.md));
- `sphere_radius()` and `project_onto_unit_sphere()` for meshes on the sphere, and, when MOAB was
  built with TempestRemap, the spherical and high-order remaps of [](geometry.md).

A remap halo between two placements that resolve to different MOAB instances is refused with
`E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED`.

## Search at a glance

`fuse/search/box_index.hpp` is the spatial index behind the covering-set construction, usable on
its own:

```cpp
using namespace fuse::search;
const Tolerance tol;                       // relative 1e-8, absolute floor 1e-12
std::vector<Box> boxes = {
    padded(Box{Point{0.0, 0.0, 0.0}, Point{1.0, 1.0, 0.0}}, tol),
    padded(Box{Point{1.0, 0.0, 0.0}, Point{2.0, 1.0, 0.0}}, tol)};
auto idx = BoxIndex::build(boxes);         // ExecutionSpace::Default
auto hits = idx.value().query_points({Point{1.0, 0.5, 0.0}, Point{5.0, 5.0, 0.0}});
const QueryResult& q = hits.value();        // CSR: offsets.size() == queries + 1
for (std::size_t p = 0; p + 1 < q.offsets.size(); ++p) {
  std::cout << "point " << p << " hits:";
  for (std::size_t k = q.offsets[p]; k < q.offsets[p + 1]; ++k) std::cout << " " << q.indices[k];
  std::cout << "\n";
}
auto nearest = idx.value().query_nearest({Point{5.0, 5.0, 0.0}}, /*k=*/1);
std::cout << "nearest box to (5,5): " << nearest.value().indices[0]
          << " (index on " << idx.value().space_name() << ")\n";
```

```text
point 0 hits: 0 1
point 1 hits:
nearest box to (5,5): 1 (index on Serial)
```

`BoxIndex` adds no tolerance of its own; pad boxes with `padded(box, Tolerance)` from
`fuse/search/geometry.hpp` first. An empty index is valid. `build(boxes, ExecutionSpace)` and the
`*_device_resident` variants choose where the index lives; `FUSE_EXECUTION_SPACE` sets the default
(`E-SEARCH-SPACE` for an unknown value). `query_points_distributed(comm, local_boxes, points)` in
`fuse/search/distributed.hpp` answers queries against boxes spread across ranks and returns
`PointHit{rank, index}` pairs (`E-SEARCH-NO-MPI` without MPI). Mixed box dimensions are
`E-SEARCH-MIXED-DIM`; `k < 1` is `E-SEARCH-K`.

## Error codes

| Code | Meaning |
|---|---|
| `E-MESH-SOURCE` | unknown generated kind, malformed counts, or a path given to the serial backend |
| `E-MESH-SIZE` | a generated mesh whose entity count would overflow |
| `E-MESH-MOVED` | a call on a moved-from backend |
| `E-MESH-NAME`, `E-MESH-NAME-AMBIGUOUS` | `mesh_id()` found no mesh, or more than one, with that name |
| `E-MESH-ID` | a `MeshId` the backend never loaded |
| `E-MESH-SET` | an entity set the mesh does not define, or one that names several sets |
| `E-MESH-DIM` | a dimension the mesh does not represent, or one inconsistent with the entity set |
| `E-MESH-PLACEMENT` | a placement the backend was never told about |
| `E-MESH-NO-CELL-VERTICES` | the view has no corner list for that cell (nodal interpolation needs one) |
| `E-MESH-IO` | the native file reader or writer failed |
| `E-MESH-CONNECTIVITY`, `E-MESH-POLYGON-DEGENERATE`, `E-MESH-POLYGON-FACE-MISMATCH`, `E-MESH-CELL-NOT-STAR-SHAPED` | the native builder rejected the cell or face data |
| `E-MESH-NO-MPI`, `E-MESH-MPIINIT` | a multi-rank operation in a build without MPI, or before `MPI_Init` |
| `E-MESH-HALO`, `E-MESH-HALO-GEOMETRY`, `E-MESH-HALO-MERGE`, `E-MESH-HALO-PEER-REFUSED` | the covering-set halo could not be built, exchanged or merged; `-GEOMETRY` names a mesh pair the available construction cannot cover |
| `E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED` | a MOAB halo between two separately placed copies of a mesh |
| `E-MESH-MIGRATE-INCOMPLETE`, `E-MESH-MIGRATE-RANGE`, `E-MESH-MIGRATE-PEER-REFUSED` | `migrate()` was given the wrong number of destinations, a destination outside the communicator, or a peer failed |
| `E-MESH-GEOMETRY` | a MOAB sphere query on a mesh whose vertices are not at one radius |
| `E-MESH-ARG`, `E-MESH-INDEX` | a null buffer or mismatched length, or a local index out of range |

`E-PART-*`, `E-SEARCH-*` and `E-MOAB-*` are listed in their sections above; `E-EXTENT-*` in
[](component-interface.md). [](troubleshooting.md) covers the general pattern.

## Limitations

- The serial backend represents no edges or faces of a line, and only the boundary skin's faces of
  a box; interior faces and edges are not enumerated.
- The native backend accepts only the whole mesh as a region; named entity subsets are
  `E-MESH-SET`.
- The native file reader reads only files written by `write_native_mesh_file`.
- Nodal fields size by the whole mesh's vertex count; a nodal field on a named subset is
  `E-EXTENT-REGION`.
- `SchurSlabSolver`, MLS remap and other per-feature limits are on their own pages.

## See also

- [](geometry.md): simplices, measures and the remap ladder built on these views.
- [](placement.md) and [](layouts.md): per-placement decompositions and processor sets.
- [](transfer-maps.md): persisting remap weights.
- [](component-interface.md): binding a field of the right size.
- [](layout-driver.md): the `--mesh` / `--migrate` driver.
