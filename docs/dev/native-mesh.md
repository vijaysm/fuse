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
  "GPU-aware MPI" below). `comm == kNullComm` is the identity, touching no MPI symbol. A Space-templated
  GPU-aware REPLACEMENT, `exchange_migration_buffers<Space>`, exists alongside it (Migration track,
  2026-10-15) but is not yet called by `migrate` -- see "GPU-aware MPI" below.
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

**GPU-aware MPI (Migration track, 2026-10-15 update).** `cmake/FuseMpiGpuAware.cmake`'s
`fuse_probe_mpi_gpu_aware` is a configure-time check for the `MPIX_CUDA_AWARE_SUPPORT`/
`MPIX_ROCM_AWARE_SUPPORT` extension macros (`<mpi-ext.h>`), recorded as `FUSE_MPI_GPU_AWARE` in the
generated `config.hpp`. The run-time probe that prefers `MPIX_Query_cuda_support()` over that
configure-time fallback now lives ONCE, as `fuse::linalg::mpi_is_gpu_aware()` (`include/fuse/linalg/
comm.hpp`, `src/linalg/mpi_gpu_aware.cpp`) -- `fuse::native::mpi_is_gpu_aware()` and
`fuse::transport::MpiTransport::is_gpu_aware()` both call it now rather than each keeping its own copy
(`linalg` sits below both `backend` and `transport` in the layering order, and `backend` already links
it). The `std::vector<std::byte>`-based `exchange_migration_buffers` is UNCHANGED and still what
`migrate`/`build_native_remap_halo` call: it always stages through host memory regardless of the probe.
A GPU-aware REPLACEMENT, `exchange_migration_buffers<Space>` (Kokkos::View<std::byte*, Space> buffers,
`MPI_Alltoallv` on the raw device pointer when `Space` is a device memory space and the probe says so,
host-staged otherwise), is added and verified byte-for-byte against the `std::vector` overload at 1-5
ranks and both `FUSE_TRANSPORT_HOST_STAGING` settings (`test/backend/native_migration_mpi_test.cpp`),
but is not yet called by `migrate`: wiring it through needs `unpack_migrated_mesh`/`merge_halo_cells`
to also stop requiring a host-resident buffer first (KNOWN_ISSUES.md D51's own remaining gap). (This
sandbox links MPICH, which does not implement either extension macro, so the configure-time fallback
reports 0 here; higgs links OpenMPI 4.1.5, which does, so the run-time probe is exercised as a real 0/1
decision there -- see the ctest rows above.)

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
verified, until run on the user's host (KNOWN_ISSUES.md records no host run of this file specifically).
`examples/layout_driver`'s own `layout.migrate.{2,3,4}.*` rows are a DIFFERENT claim: KNOWN_ISSUES.md's
D5 entry records all 18 `layout.*` tests, including the 3 `layout.migrate.*` ones, passing on the host
2026-09-23 -- this doc previously (and wrongly) lumped them in with the still-host-unverified migration
test above; they are VERIFIED, not merely compiled.

Tier 1, native halo (`test/parallel/native_halo_mpi_test.cpp`; docs/dev/native-halo.md has the full
breakdown): `parallel.native_halo_mpi.{2,3}` both passed on the host 2026-10-10 -- VERIFIED. This
sibling file was in the same "compiled, awaiting host run" state the migration row above is still in
until that run cleared it; an earlier revision of this table did not distinguish the two and left
native halo looking unverified after it no longer was.

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

## Execution space

`MeshArrays<MemorySpace>`'s template argument has always defaulted to
`Kokkos::DefaultExecutionSpace::memory_space` -- device memory on a CUDA/HIP/SYCL build. Through two
rounds of work this is now `NativeMeshView`'s own PRIMARY storage, not merely a type parameter nothing
read: `create_from_host` mirrors to the host ONCE to validate (below), but keeps nothing from that
pass -- what the view holds afterward is `device_arrays()` and the device-resident derived geometry,
with every host-serving query rebuilt from those lazily.

**Validation stays host-only, unconditionally.** Connectivity range checks, the polygon/polyhedron
fan triangulation (`canonical_fan_triangulation`), and the star-shapedness checks
(`canonical_cell_tetrahedra`) all return `Result`/use `fmt::format`, neither of which is
device-callable in this codebase's own convention (same standing priority -- robustness over
efficiency -- the file header above already states). `create_from_host` runs this once, over a
host-mirrored `MeshArrays<Kokkos::HostSpace>`, exactly as before.

**Device-primary storage, on-demand host mirrors.** What changed is what happens to that validation
pass's own host-side scratch afterward. `device_arrays_` -- `to_memory_space<DeviceMemSpace>` of the
constructor's input, built unconditionally at the TOP of `create_from_host` -- is this view's PRIMARY
storage, exposed as `device_arrays()`. The host arrays the validation pass computes as it goes
(`vertex_xyz_`/`cell_vertices_`/`cell_faces_`/`cell_simplices_`/`centroid_`/`measure_` -- everything
`measure(dim,i)`/`centroid(dim,i)`/`simplex_count`/`simplex_points`/`cell_vertices`/`vertex_x` serve)
are no longer retained once construction finishes: `create_from_host`'s last step releases them
(`std::vector<...>().swap(...)`, not merely `.clear()`, so the capacity is actually freed, not just the
size). `ensure_host_mirror()` rebuilds them the FIRST time any of those accessors is actually called --
a plain mirror of `device_arrays_` plus the device-resident measure/centroid/simplex CSR below, not a
second validation pass (that already succeeded once; this step cannot fail) -- and caches the result
for the view's lifetime after that. `host_mirrors_materialized()` is this count (0 immediately after
construction, 1 after the first host accessor call), the hook
`test/backend/native_distributed_mesh_test.cpp`'s `NoHostMirrorExistsUntilAHostAccessorIsCalled` checks
directly. Every *other* field (`vertex_gid_`/`vertex_owner_`/`vertex_owned_`/`cell_type_`/`cell_gid_`/
`cell_owner_`/`cell_owned_`) is cheap per-entity bookkeeping copied once from the constructor's host
argument, not part of this lazy group -- there is no fallible geometry behind any of them to defer.

**The device-resident derived geometry, unchanged in shape, now retained in full.**
`NativeMeshView::device_measure()`/`device_centroid()` are
`Kokkos::View<double*, DeviceMemSpace>`/`Kokkos::View<double* [3], DeviceMemSpace>`
(`DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space`), populated by two Kokkos kernels
dispatched on `Kokkos::DefaultExecutionSpace` at the end of `create_from_host`, after the host
validation pass has already run:

* **Centroid** is a per-cell vertex average (the SAME convention `centroid(dim,i)` has always used,
  not a simplex-weighted one), computed directly from `vertex_coords` and the cell-to-vertex CSR with
  no simplex decomposition involved at all, in the same ascending-corner order the host pass uses.
* **Measure** sums each cell's own simplices -- the SAME simplex corners
  `canonical_fan_triangulation`/`canonical_cell_tetrahedra` just validated, flattened into a device
  CSR -- in the same order the host pass pushed them, through device-callable counterparts of
  `geometry.hpp`'s `triangle_area`/`tetrahedron_volume` formulas (`native_mesh.cpp`'s anonymous
  namespace; independent of that header's host-only operator overloads, so geometry.hpp itself is
  untouched).

Both kernels reproduce the host pass's own accumulation order exactly, so on a host-only build
(`Kokkos::DefaultExecutionSpace` IS the host space there) the device arrays are bit-identical to a
materialized `measure_`/`centroid_`; on higgs's CUDA build they agree to 1e-14 relative (nvcc's FMA
contraction; KNOWN_ISSUES.md D45, the accepted contraction gap) -- and since `ensure_host_mirror()`
now serves `measure_`/`centroid_` BY MIRRORING `device_measure_`/`device_centroid_` rather than
recomputing them, that is no longer merely a cross-check this file's tests happen to make: it is the
actual relationship between what a host accessor returns and what the device computed.

**New: the device simplex CSR is retained, not discarded.** `device_simplex_offsets()`/
`device_simplex_corners()` -- previously flattened inside `create_from_host` purely to feed the
measure kernel, then thrown away (this section used to say so explicitly) -- are now persisted as
`Kokkos::View<int*, DeviceMemSpace>`/`Kokkos::View<double* [4][3], Kokkos::LayoutRight, DeviceMemSpace>`
members. Cell `c`'s simplices are rows `offsets(c)..offsets(c+1)` of `corners`; a row's first
`dimension() + 1` corner slots are the simplex's physical points, the same convention
`cell_simplices_`'s own `std::array<Point, 4>` has always used. `ensure_host_mirror()` unflattens this
CSR into `cell_simplices_` rather than recomputing it, the same "mirror, not re-derive" relationship
measure/centroid now have.

**What this makes possible: device-resident element boxes.**
`fuse::search::element_boxes_device_resident` (`include/fuse/search/mesh_adapter.hpp`,
`src/search/mesh_adapter_device.cpp`) takes a simplex CSR shaped exactly like
`device_simplex_offsets()`/`device_simplex_corners()` and returns `Kokkos::View<Box*, DeviceMemSpace>`
-- the device-resident counterpart of `element_boxes` (same file), built with a `Kokkos::parallel_scan`
compaction over an optional per-entity `owned` flag (the device analogue of `element_boxes`'s own
`is_owned` filter) rather than a sequential `push_back`. `fuse::search` cannot name
`fuse::backend::NativeMeshView` (this layer sits below backend, mesh_adapter.hpp's own header
comment), so this is not a `MeshLike`-constrained template the way `element_boxes` is: a caller that
CAN name both (`fuse::backend` itself, or a test linking it) passes the native mesh's own CSR views
directly, and feeds the result straight to `BoxIndex::build_device_resident` -- index construction for
a native mesh's elements never touches the host at all.
`test/search/mesh_adapter_test.cpp`'s `DeviceElementBoxesMatchHostElementBoxesForANativeMesh` checks
the device path reproduces `element_boxes`'s own boxes to 1e-13 and that the device view's
`host_mirrors_materialized()` never leaves 0 across the whole pipeline (CSR read, box build, index
build).

**What is still host-only.** `NativeMeshBuilder` (host `std::vector`-based construction), the VTK file
reader/writer, `fuse::native::migrate`'s pack/unpack, and `build_native_remap_halo`/
`build_supermesh_weights` (the MLS and conservative-intersection weight builders) all still go through
the host-only `MeshView`/`MeshArrays<Kokkos::HostSpace>` surface and are unaffected by this section --
see "What is deferred" below and docs/dev/device-resident-arcs.md's own Phase 2 ordering (search, then
this file, then halo and migration pack/unpack, then the weight builders, named there as this track's
siblings' scope, not this one's). The halo receive-side merge and migration's unpack (both a
dedup-by-global-id decision) still run on the host regardless of what space the mesh they operate on
is in -- unaffected by `NativeMeshView` becoming device-primary, since neither goes through
`NativeMeshView` at all (`MeshArrays<Space>`'s own CSR, directly).

## Device-capable packing (Phase 2, "weight-builders" track)

`include/fuse/backend/native/mesh_pack_device.hpp` templates migration's and the remap halo's byte-wire
PACK on the Kokkos memory space (docs/dev/native-halo.md's own "Device packing" section has the full
account): `MeshArrays<Space>`'s plain CSR connectivity is read directly, with no `NativeMeshView`
round trip, so packing runs as Kokkos kernels on `Space`. The receiving side's dedup-by-global-id merge
is not yet templated -- see "What is deferred" below and KNOWN_ISSUES.md.

## What is deferred

- **Coverage migration (M3).** This migrates OWNED cells whole; it does not build or maintain GHOST
  layers across a repartition. A caller needing ghosts after migration re-derives them separately.
- **Per-placement remap weights.** `repartition`'s convenience wrapper uses a uniform per-cell weight;
  load-imbalance-aware weighting is available by calling `rcb_partition`/`migrate` directly, not through
  the wrapper.
- **Device-resident migration/halo, CLOSED 2026-10-15 (Migration track, round 3).** `pack_for_migration`/
  `pack_halo_cells`, `exchange_migration_buffers<Space>`, and `unpack_migrated_mesh`/`merge_halo_cells`
  (via their device-templated `unpack_migrated_mesh_device`/`merge_halo_cells_device`) are wired end to
  end in `migrate`/`build_native_remap_halo`, with no host mirror materialized across the full round
  trip (KNOWN_ISSUES.md D51, docs/dev/native-halo.md's "Device merge decision"). Three narrow, declared
  exceptions remain (no cross-cell Polyhedron face dedup on device, one bounded host round trip for
  `resolve_vertex_ownership_device`'s vertex gids, and `E-MESH-CONNECTIVITY`'s corner-count check not
  reproduced) -- see D51's closing text for why none blocks this track's own mandate.
- **Field-transfer integration for a migrated mesh.** The layout driver's `--migrate` mode verifies MESH
  conservation properties, not a full FIELD hand-off through the coupling graph's transport layer for an
  arbitrary post-migration entity ordering. Matching migrated entities back to `MpiTransport`'s exchange
  plan (so a component's FIELD DATA, not just its mesh, follows a repartition) is a materially larger
  feature -- closer in shape to a distributed remap operator than to migration itself -- and is not
  implemented here.
- **A cubed-sphere seam.** The six panels the `cubedsphere` generator builds are independently numbered;
  a seam-welded (shared-vertex) cubed sphere is not implemented.
