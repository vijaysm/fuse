# The native distributed mesh

`fuse::native` is a general-cell, distributed, unstructured mesh on Kokkos views, implementing
`fuse::MeshView`/`fuse::MeshBackend` (`mesh_view.hpp`) without MOAB. It exists so mesh migration --
repartitioning a mesh's cells onto a different PE layout, driven by `fuse::partition::rcb_partition`
-- works against FUSE's own data structures, portable to GPUs through Kokkos memory spaces, not only
against MOAB's.

## Data model

`fuse::native::MeshArrays<MemorySpace>` (`include/fuse/backend/native_mesh.hpp`) is plain data: vertex
coordinates (`nv x 3`), a cell-to-vertex CSR, per-cell `CellType`, `int64` global ids for both cells and
vertices, per-entity owner rank and an owned/ghost flag, and -- for `CellType::Polyhedron` cells only --
a cell-to-face and a deduplicated face-to-vertex CSR (plus a per-cell-face orientation flag, since two
cells sharing a face store it in the direction that is outward FOR THEM, which are opposite directions).
Every array lives in `MemorySpace`; `to_memory_space<Target>` deep-copies to another space in either
direction, the one hook that makes the mesh GPU-resident.

`CellType` (`include/fuse/backend/native/cell_topology.hpp`) covers Point, Segment, Triangle, Quad,
Polygon (n >= 3, general), Tet, Pyramid, Prism, Hex, and Polyhedron (explicit face list). A mesh holds
cells of exactly one topological dimension; "mixed types" means mixed SHAPES at that one dimension (a
tri+quad 2D mesh, or a hex+tet+prism+pyramid+polyhedron 3D mesh), never mixed dimension in one mesh --
the same contract `MeshView::dimension()` already promises for every other backend.

`NativeMeshView` (`include/fuse/backend/native_mesh.hpp`/`src/backend/native_mesh.cpp`) is the concrete,
non-templated `MeshView`. It is built from a `MeshArrays<AnySpace>` by mirroring to the host ONCE
(`create_from_host` is what every `create<MemorySpace>` delegates to) and validating/decomposing
entirely there. `Result`/`Failure`/`fmt` are not device-callable in this codebase's own conventions, and
this is a one-time, per-rank, host-side cost -- the standing priority (robustness before efficiency)
made explicitly: a correct O(n) host validation pass beats a clever but unverifiable device kernel here.

`NativeMeshBuilder` is the host-side (`std::vector`-based) construction API every source (generators,
the file reader, migration's unpack) goes through: `add_vertex`, `add_cell`, `add_polyhedron_cell`, then
`build()` -> `Result<MeshArrays<Kokkos::HostSpace>>`. It deduplicates a polyhedron's faces by their
vertex GLOBAL id SET (matching either cyclic direction), recording per-cell-face orientation.

## The simplex-decomposition geometry contract

`MeshView`'s contract (`mesh_view.hpp`) requires every cell to present a simplex decomposition whose
measures sum exactly to the cell's own measure, and refuses -- named, never a silent wrong answer -- a
cell it cannot decompose correctly. The native mesh adds one more requirement of its own, because it is
the first backend with explicit shared face structure between neighbouring cells:

**The shared-face rule.** A face shared by two cells must be triangulated IDENTICALLY from both sides.
Two cells commonly number that face's corners differently (different starting corner, and MOAB's own
convention makes their cyclic directions opposite -- each cell's face is stored outward-facing FOR
THAT CELL). A triangulation keyed on local corner index -- fan from corner 0, say -- picks a DIFFERENT
diagonal on each side of the same physical face, and a conservative supermesh remap across that boundary
then sees a gap on one side and an overlap on the other, even though each cell's own measure is exact.

**The rule chosen: fan from the face's MINIMUM-GLOBAL-ID corner.** Implemented once, in
`fuse::native::canonical_fan_triangulation` (`include/fuse/backend/native/geometry.hpp`/
`src/backend/native_geometry.cpp`), and reused both for a 2D polygon CELL and for one FACE of a 3D cell.
Two records of the same face always agree on which vertex has the minimum global id, so rotating the fan
to start there needs no agreement on which cyclic direction to walk either: fanning forward from vertex
0 over `v1..v_{n-1}` gives triangles `(v0,v1,v2), (v0,v2,v3), ...`, and fanning backward gives the SAME
n-2 triangles as vertex sets, because the diagonal pattern `v0-v_i` for `i = 2..n-2` is symmetric under
`i -> n-i`. So the two cells' independent triangulations of the shared face are identical without either
one knowing anything about the other's local numbering or orientation -- only the global ids.

**Validity is checked, not assumed.** A fan is only a correct decomposition when the apex sees the whole
polygon (star-shaped about that point); a reflex corner can make the fan fold back on itself. The check
is exact, not a convexity heuristic: Newell's formula gives a polygon's signed vector area as an
algebraic IDENTITY with the signed sum of the SAME fan's own triangles, true for ANY apex, star-shaped or
not. Comparing the UNSIGNED triangle-area sum (what `simplex_points`/`measure` must report) against that
signed magnitude catches exactly the fold-back case: `unsigned_sum > |signed_total|` if and only if some
triangle disagrees in orientation with the whole. It is exact for a planar polygon and well defined
(though only a vector-area identity, not a differential-geometric one) for a NON-PLANAR face too -- a
cubed-sphere quad, embedded in 3D, passes this check without being flagged as a defect merely for having
curvature. A cell this test fails refuses with the named error `E-MESH-POLYGON-NOT-STAR-SHAPED`
(`E-MESH-CELL-NOT-STAR-SHAPED` for the analogous 3D check, which fans every face to the cell's own
centroid and cross-checks against the divergence-theorem enclosed volume from the same oriented faces).
Neither ever reports a wrong measure silently.

## Sources (M1)

Generators (`src/backend/native_generators.cpp`): `generated:line:N`, `generated:tri:NxM`,
`generated:quad:NxM`, `generated:cubedsphere:N` (six independently-numbered panels, not seam-welded --
each panel's own boundary vertices are distinct global ids from its neighbour's, a documented
simplification), `generated:hexbox:NxMxP`, `generated:tetbox:NxMxP` (Kuhn split of each hex, six tets,
reusing `cell_simplices.hpp`'s existing table), `generated:prismpyramidmix:NxMxP` (a hex at even
`i+j+k` becomes 2 prisms, odd becomes 6 pyramids about an inserted centroid vertex -- one mesh, two 3D
shapes), and `generated:polyhedral:NxMxP` (the same hex box, but every cell stored as a general
`Polyhedron` via explicit face CSR rather than the fixed `Hex` shape, exercising the polyhedron path on
a case whose true answer is already known). Every generator is DISTRIBUTED: each rank builds only its
own slice (`rank_slice`, an even split of the relevant index range), with global ids that depend on the
grid dimensions alone, never on the rank count -- the same property `fuse::partition`'s own fixtures
rely on to compare a distributed run against a serial oracle.

File I/O (`include/fuse/backend/native_mesh_io.hpp`/`src/backend/native_mesh_io.cpp`): a documented
SUBSET of legacy VTK ASCII `UNSTRUCTURED_GRID`. Every native cell type maps onto an existing VTK legacy
cell type (`VTK_HEXAHEDRON`, `VTK_WEDGE`, `VTK_PYRAMID`, ..., `VTK_POLYHEDRON` for a general
`Polyhedron`) with an IDENTICAL corner-order convention, so no reordering is needed. This is a
WRITER-DEFINED subset -- the reader here parses only what the writer produces -- not a general VTK
parser, but every token is standard legacy VTK syntax, so the geometry remains openable in a real
VTK/ParaView reader; global ids and owner/ghost flags ride as ordinary named `POINT_DATA`/`CELL_DATA`
scalars, plus one `#`-prefixed comment line (topological dimension) that a real reader skips. Ids are
written as `double` (exact for every integer up to 2^53, which covers any mesh size this is used at). A
round trip reproduces the mesh exactly, including a polyhedral mesh's face structure.

## Repartition migration (M2)

`fuse::native::migrate` (`include/fuse/backend/native_migration.hpp`/`src/backend/native_migration.cpp`)
moves owned cells, with their full closure, onto a new per-cell destination rank. Structured so the MPI
part is ONLY the exchange, matching `fuse::partition::detail::collectives.cpp`'s own isolation:

- `pack_for_migration(mesh, dest, num_ranks)` -- packs each owned cell (in cell-index order restricted
  to `is_owned`) whose destination is `dest[i]` into `result[dest[i]]`: the cell's own id and type, and
  its full closure (every vertex it touches, with coordinates and global id; for a `Polyhedron`, its
  face structure too, translated to positions within THIS packed cell's own closure). Refuses
  `E-MESH-MIGRATE-INCOMPLETE` when `dest.size()` is not the owned cell count, `E-MESH-MIGRATE-RANGE` when
  an entry names a destination outside `[0, num_ranks)`.
- `exchange_migration_buffers(send, comm)` -- the ONLY function that names an MPI type: an
  `MPI_Alltoall` for sizes then `MPI_Alltoallv` for the payload, always staged through HOST buffers (see
  "GPU-aware MPI" below). `comm == kNullComm` is the identity, touching no MPI symbol.
- `unpack_migrated_mesh(received, dim, my_rank)` -- rebuilds a mesh from every received buffer, deduping
  vertices and faces by global id (`NativeMeshBuilder`'s own rule). Every unpacked cell is OWNED by this
  rank (a cell migrates whole, never as a ghost); vertex ownership is left tentative.
- `resolve_vertex_ownership(arrays, comm)` -- the deterministic, cross-rank-agreed ownership rule: a
  vertex's new owner is the MINIMUM rank, over the whole communicator, among ranks that now hold a copy
  of it. Computed with one `MPI_Allgatherv` of every rank's own `(global id)` list (segment boundaries
  from a preceding `MPI_Allgather` of counts) and a local min-by-key scan -- an O(total vertex count
  across all ranks) pass, chosen deliberately over a comm-avoiding heuristic: any rule that tries to
  decide a shared boundary vertex's owner from one rank's own received buffers alone cannot see the
  copies that landed on OTHER ranks, so it cannot be made correct without SOME cross-rank exchange: this
  is the smallest one that is actually correct, not a shortcut around it. `comm == kNullComm` makes
  every vertex trivially owned by rank 0.
- `migrate(mesh, dest, comm)` composes all four, with a COLLECTIVE refusal: `pack_for_migration`'s
  refusal is local, but a caller must never let one rank refuse and return while its neighbours proceed
  into the alltoall expecting every rank to participate -- exactly the "divergent early return causing
  hangs" failure mode `rcb_moab`'s `MeshMigration.cpp` (see `HANDOFF.md`) met. So `migrate` reduces its
  own local verdict with every other rank via one `MPI_Allreduce(MIN)` BEFORE the exchange: if any rank
  is invalid, EVERY rank returns the refusal, none of them touching the alltoall.

**Emulated Tier-0 testing.** Because migration is only `pack` + `exchange` + `unpack` + ownership
agreement, and `exchange` is the identity at `kNullComm`, a Tier-0 test can emulate a P-rank migration in
one process: call `pack_for_migration` once per emulated source rank, hand each destination's
concatenated buffers directly to `unpack_migrated_mesh` (skipping `exchange_migration_buffers`
entirely), and check the recombined result -- exactly the way the partition track compares RCB against
an emulated-rank oracle. `test/backend/native_migration_test.cpp` does this at P = 1..5, and separately
checks a full round trip through two DIFFERENT partitions (a 1-rank mesh split three ways, then gathered
back onto one rank) reproduces the original global cell id set and total measure exactly.

**`fuse::partition::repartition`** (`include/fuse/partition/repartition.hpp`, header-only) is the
convenience the task asked for: `centroids_of` (already existing, `mesh_adapter.hpp`) + `rcb_partition`
(uniform per-cell weight) + `migrate`, sharing one communicator across all three. A caller wanting
load-imbalance-aware weights calls `rcb_partition`/`migrate` directly instead.

**GPU-aware MPI.** `cmake/FuseMpiGpuAware.cmake`'s `fuse_probe_mpi_gpu_aware` is a configure-time check
for the `MPIX_CUDA_AWARE_SUPPORT`/`MPIX_ROCM_AWARE_SUPPORT` extension macros (`<mpi-ext.h>`), recorded as
`FUSE_MPI_GPU_AWARE` in the generated `config.hpp` and read by `fuse::native::mpi_is_gpu_aware()`. The
result is INFORMATIONAL ONLY: `exchange_migration_buffers` always stages through host memory regardless
of what it reports, per the standing priority (robustness before efficiency) -- a device-resident
exchange path is future work, named in "what is deferred" below. (This sandbox links MPICH, which does
not implement either extension macro, so the probe reports 0 here; it is exercised as a real 0/1
decision only on an MPI build that does.)

## What is verified where

Tier 0 (no MPI; this sandbox): the geometry contract per cell type (`native_distributed_mesh_test.cpp`
-- simplex measures sum to the analytic value, the two-differently-numbered-hexes shared-face case, the
concave-polygon refusal, connectivity/dimension/corner-count refusals, owned-vs-ghost accounting), the
generators and file round trip (`native_generators_test.cpp`), and the emulated migration properties
above plus `migrate`'s own two refusals at `kNullComm` (`native_migration_test.cpp`), and
`repartition`'s identity at `kNullComm` (`test/partition/repartition_test.cpp`). All of these are
VERIFIED, in the sense `docs/dev/testing.md`'s "five states" define it: a test exists and passes.

Tier 1 (`test/backend/native_migration_mpi_test.cpp`, 2-5 ranks): real migration under a real `mpiexec`
-- each rank builds its OWN slice via `generate_native_mesh`, `fuse::partition::repartition` runs the
actual collective sequence (dimension agreement, RCB's own collectives, the refusal-agreement Allreduce,
the real `MPI_Alltoallv` exchange, the ownership Allgatherv), and the test checks global cell count
conservation, total measure conservation, global id uniqueness, a round trip through two different
partitions, and the collective refusal firing when only ONE rank's assignment is bad. This sandbox
cannot initialise MPI (`mpiexec` cannot bind a port, and a singleton `MPI_Init` aborts here), so these
are "compiled here, awaiting host run" -- `docs/dev/testing.md`'s "compiled, untested" state, not
verified, until run on the user's host. The same is true of `examples/layout_driver`'s
`layout.migrate.{2,3,4}.*` rows (below).

## The layout driver's `--mesh`/`--migrate`

`examples/layout_driver/main.cpp` gains `--mesh <generated-spec>` and `--migrate`. With `--mesh` alone,
comp1 and comp2 each get an INDEPENDENTLY generated slice of the same spec, sized to their own
placement -- the native-backend analogue of what the driver's existing synthetic
`PartitionedTestBackend` mode already does. With `--migrate` added, comp2's mesh is instead comp1's,
migrated onto comp2's placement: `centroids_of` + `rcb_partition` (targeting `num_parts = comp2`'s own
rank count) + a remap from RCB's `[0, num_parts)` part id to comp2's actual WORLD rank
(`comp2_ranks.ranks()[part_id]`) + `migrate`, all over `MPI_COMM_WORLD` -- a rank outside comp1
contributes an empty local mesh (zero owned cells, zero centroids), which `rcb_partition`'s own
documented "empty local input" case already handles as ordinary, not a special branch.

`--check` in this mode re-verifies MESH data transfer: global cell count and total measure conserved
across the migration (an `MPI_Allreduce` each, printed and compared on rank 0) -- a DIFFERENT claim from
the field-transfer `--check` the driver's other (non-`--mesh`) mode makes, and deliberately not the same
one. See "what is deferred" for why the two are not merged.

Registered at 2, 3 and 4 ranks (`layout.migrate.{2,3,4}.c1-<lo>-<hi>.c2-<lo>-<hi>`,
`build-evidence/parallel-matrix.toml`'s `A-LAYOUT` cell), on the same disjoint/partially-overlapping
placement shapes the existing `layout.*` rows already cover. Compiled and linked in this sandbox
(Tier 1); running them is a host action, same as every other MPI test here.

## What is deferred

- **Coverage migration (M3).** This migrates OWNED cells whole; it does not build or maintain GHOST
  layers across a repartition. A caller needing ghosts after migration re-derives them separately.
- **Per-placement remap weights.** `repartition`'s convenience wrapper uses a uniform per-cell weight;
  load-imbalance-aware weighting is available by calling `rcb_partition`/`migrate` directly, not through
  the wrapper.
- **Device-resident intersection / exchange.** `exchange_migration_buffers` always stages through host
  memory, regardless of what `mpi_is_gpu_aware()` reports (see "GPU-aware MPI" above); no kernel in this
  track runs the pack/unpack byte-layout logic on a device execution space.
- **Field-transfer integration for a migrated mesh.** The layout driver's `--migrate` mode verifies MESH
  conservation properties, not a full FIELD hand-off through the coupling graph's transport layer for an
  arbitrary post-migration entity ordering. Matching migrated entities back to `MpiTransport`'s exchange
  plan (so a component's FIELD DATA, not just its mesh, follows a repartition) is a materially larger
  feature -- closer in shape to a distributed remap operator than to migration itself -- and is not
  implemented here.
- **A cubed-sphere seam.** The six panels the `cubedsphere` generator builds are independently numbered;
  a seam-welded (shared-vertex) cubed sphere is not implemented.
