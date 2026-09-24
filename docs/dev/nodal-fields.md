# Nodal fields: sizing, and parallel vertex ownership

A field's discretization (`FieldSpec::disc`/`order`) decides not just which
remap operator applies to it (`connect.hpp`'s `to_semantics`,
`consistent_interp.hpp`'s `is_nodal`) but how many scalar values it needs --
and until this track, every buffer-sizing and conservation-measure call site
answered that second question the same, wrong way for a genuinely nodal
field: by the region's own (cell) dimension, unconditionally.

## The one definition

`include/fuse/backend/mesh_view.hpp` now carries the single definition,
`field_extent(disc, order, view, region_dim)`:

* FV, and `DiscontinuousGLL` order 0, read one value per `region_dim` entity
  -- a cell. This is `MeshBackend::region_size`'s existing behaviour,
  unchanged: `region_size`/`placed_region_size` still answer this question
  alone, with no field in view, exactly as they always have. Nothing calls
  them differently.
* `ContinuousGLL` order 1 is genuinely nodal -- `connect.hpp` already maps
  it to `SemanticLocation::Vertex`, and `consistent_interp.hpp`'s H/H^T
  already index `MeshView` dimension-0 local indices for it -- so its
  extent is the mesh's **vertex** count, `view.entity_count(0)`, regardless
  of `region_dim`. "The number of distinct vertices of the region's cells"
  means the WHOLE mesh: a region naming a named entity subset (anything but
  `""`/`"all"`) is refused, `E-EXTENT-REGION`, for the same reason
  `consistent_interp.hpp`'s `resolve_side` already refuses one for the
  nodal side -- `MeshView` has no way to enumerate a subset's vertices, so
  "the region's cells" cannot be answered for one.
* Everything else -- `ContinuousGLL` order != 1, `DiscontinuousGLL` order
  >= 1, `PointCloud` -- is a **named refusal**, `E-EXTENT-DISC`, not a
  guess. A spectral element needs as many DOFs per cell as `order` implies,
  which nothing here computes; a point cloud has no cell topology to size a
  field by at all.

A companion, `region_extent_kind(graph, region)`, answers the same question
at the GRAPH level, for a `RegionId` rather than a `(disc, order)` pair
directly: it scans every `FieldSpec` declared over that region and requires
they agree. A region with no field on it yet defaults to `Cell` (backward
compatible: nothing that never declares a nodal field sees any change at
all). A region carrying both a cell field and a nodal one refuses,
`E-EXTENT-MIXED` -- one region cannot be sized two different ways at once,
and silently picking either would size the other field wrong with no
diagnostic.

## Where it is wired in

Every call site that used to size or check a field's buffer by
`view.entity_count(region_dim)` unconditionally now goes through
`region_extent_kind` (to decide Cell vs. Vertex for the region in question)
and, once decided, reads the vertex count directly rather than re-deriving
it a second way:

* **`make_region_sizes`** (`src/backend/serial_backend.cpp`) -- the
  `RegionSizes` closure `plan_buffers`/`schedule()` call for every arc. For
  a `Vertex`-kind region it takes `backend.placed_view(...)->entity_count(0)`
  instead of `backend.placed_region_size(...)`; for `Cell` it is completely
  unchanged. This one function is generic over `MeshBackend`, so it covers
  the native and MOAB backends too via the same virtual calls
  `region_size`/`placed_view` already use -- no backend-specific sizing code
  was needed.
* **`Runner::Impl::weights_for`** (`src/runtime/runner.cpp`) -- the
  conservation-measure builder a `Constrain` node's pre-/post-remap integral
  check reads. See "The conservation measure" below.
* **`Runner::Impl::attach_comm_graphs`** (`src/runtime/runner.cpp`, Tier 1) --
  which dimension a `Transport` instruction matches entities at. See
  "Parallel: which dimension Transport matches at" below.

`schedule/buffers.cpp`'s `try_plan_buffers` itself needed **no change**: it
already calls `sizes(region, placement)` once per arc with no assumption
about what the answer means, and `RegionSizes`'s signature -- `(RegionId,
int) -> size_t` -- did not need to grow a `FieldSpec` parameter, because
`region_extent_kind` recovers the same information by reading the graph
`make_region_sizes` already closes over. This is why the fix stayed inside
`include/fuse/backend/**`/`src/backend/**` (this track's own files) rather
than needing a change to `ValueType`/`Arc` (`include/fuse/ir/op.hpp`) or to
`try_plan_buffers`'s signature (which would have required a change in
`src/schedule/scheduler.cpp`, outside this track's ownership).

## The conservation measure

A `Constrain` node's integral is `sum_i owned(i) * measure(i) * value(i)`
for a CELL field -- `weights_for` builds the `measure(i)` vector once,
zeroing every non-owned (ghost) entity so a distributed field's integral
does not double-count another rank's copy.

For a nodal `Extensive` field this is wrong twice over: `MeshView::measure(0,
i)` throws by contract (a vertex has no measure), and even if it did not,
multiplying would be the wrong operation. `consistent_interp.hpp`'s "DENSITY
VS LOAD" already establishes the convention this follows: the
`"consistent-transpose"` builder (`H^T D`) converts a density SOURCE into a
LOAD before the matrix is ever applied, so what reaches a nodal Extensive
field's buffer is already an amount, not a density -- its conserved integral
is the plain SUM of its (owned) values, weight 1, not a measure-weighted
sum. `weights_for` now branches on `region_extent_kind`: `Vertex` gets
weight `1.0` per owned vertex (0 for a ghost, same double-counting guard as
the cell path), dimension 0 instead of the region's own dimension for the
entity-count cross-check; `Cell` is unchanged.

This is why the acceptance test (`InterfaceTransfer.
ConsistentPairRunsThroughTheRunnerExactlyAndBalancesPower`) checks `sum(
beam_load)` against `sum(fluid_force * measure)` directly, with weight 1 on
the beam side, rather than reusing a single "measure-weighted" helper for
both sides -- they are genuinely different sums, not the same sum computed
two ways.

## Parallel: lowest-rank vertex ownership

In parallel, a vertex shared by several ranks has exactly one owner: the
LOWEST rank holding it. This is not a new rule invented for this track --
it is the native mesh's own rule, `resolve_vertex_ownership`
(`src/backend/native_migration.cpp`): every rank `Allgatherv`s its local
vertex global ids, and each vertex's owner is `min` over every rank
reporting that id. `NativeMeshView::is_owned(0, i)` already answers
correctly for any native-mesh-backed run; nothing in this track re-derives
or re-implements that rule -- it is reused, exactly as directed.

### Which dimension Transport matches at

`Runner::Impl::attach_comm_graphs` builds one `CommGraph` per `Transport`
instruction, matching entities between the two components' own
decompositions of the same region by `MeshView::global_id` at one
dimension, `dim`. That dimension used to be unconditionally the region's
own (cell) dimension; it is now `0` whenever `region_extent_kind` says the
region is `Vertex`-kind, the same verdict `make_region_sizes` and
`weights_for` already reach for it. This check is local (reads only the
graph, identical on every rank) and is made before `CommGraph::build()`,
the actual collective, exactly like the two existing local checks ((1)
region-preservation, (2) named-subset refusal) already in that loop.

Once `dim` is right, **no other change was needed** for a plain (non
-conservative) nodal `Transport` to move correctly: `comm_graph.cpp`'s
`gather_side` already treats the SOURCE side as owned-only and the
DESTINATION side as every local entity (owned + ghost) --

> "The source side contributes OWNED entities only [...] A destination's
> ghost slots [...] are part of the buffer that has to be filled [...] and
> several destinations wanting one id is a multicast, which the transport
> handles."

-- which is exactly "move the owned values, and fill every non-owned copy
from its owner" for ANY dimension, cell or vertex. This is why property (a)
in `test/parallel/nodal_field_test.cpp` (every copy of a shared vertex
equals its owner's value after a transfer) follows from the `dim` fix
alone, with no new transport logic: the machinery was already
dimension-generic, it was simply never asked about dimension 0 for a field
in view.

### H^T on a distributed interface: refused until it can be verified

A Remap into a nodal field on more than one rank is refused
(`E-RT-REMAP-NODAL-PARALLEL`, KNOWN_ISSUES D8).

The reduction such a target needs depends on how the weights were built. With
weights on a genuinely partitioned source, each holder of a shared vertex holds a
partial row: loads must be summed onto the owner once, and pointwise values taken
from the owner. With weights on a replicated source, which is every case reachable
today because weights are built on the default view (D5 item 3), each holder
already holds the complete row, and the sum multiplies it by the holder count.

The reduction was first wired in on the assumption that rows are partial, and that
assumption was never tested. It is now behind the refusal:
`reduce_nodal_remap_target()`, `accumulate_shared_rows` and `copy_owner_shared_rows`
remain, and the last two are unit-tested under MPI. The refusal is lifted once
per-placement weights exist and a partitioned, non-matching nodal Remap through the
Runner conserves force and balances power at 1 to 5 ranks.

## Mutation coverage

`build-evidence/mutations/nodal-fields.toml` (this track's own file) records
four rows: the ownership rule inverted to max-rank, a halo copy skipped, a
shared load summed on every holder rather than once, and a nodal field's
extent read as its region's cell count. The first three need a working MPI
launcher (`requires = "mpi"`) and are `NOT-ASSESSED` in this sandbox
(`docs/dev/testing.md`'s "Mutations that cannot be assessed locally"). The
fourth was applied by hand to `serial_backend.cpp` -- the `Vertex`-kind
branch's `view.value()->entity_count(0)` changed to
`view.value()->entity_count(resolved.dimension)`, i.e. the region's own
(cell) dimension -- rebuilt, and confirmed to turn FOUR cases in
`test/backend/field_extent_test.cpp` from passing to failing, not just
one: `MakeRegionSizesNodal.LineMeshVertexCountIsCellsPlusOne` reports `9`
where `10` (vertices) is expected on the 9-cell line mesh, and the box/
quad-triangle/hex-tet cases fail the same way. The mutated source was then
restored byte-for-byte (`cmp` against a pre-mutation backup) and the suite
rebuilt and reconfirmed passing (14/14) before moving on.
