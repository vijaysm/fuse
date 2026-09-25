# Spatial search

`fuse::search` is a facade over [ArborX](https://github.com/arborx/arborx)'s scalable BVH
(`ArborX::BoundingVolumeHierarchy`, `ArborX::DistributedTree`) -- the way `fuse::linalg` is a
facade over its dense solver. Nothing outside `src/search/*.cpp` names an ArborX type: not even
this layer's own public headers do, so a consumer higher up never has to see ArborX either. It
answers two questions over axis-aligned boxes -- "which boxes contain this point?" and "which `k`
boxes are nearest this point?" -- on the host, serially or distributed, and adapts a `fuse::MeshView`
into the boxes it indexes.

## Dependency: ArborX 2.1, pinned

`cmake/FuseDependencies.cmake`'s `fuse_resolve_arborx()` prefers an installed ArborX >= 2.1 on
`CMAKE_PREFIX_PATH` and otherwise always builds the pinned copy, like yaml-cpp and r3d, never gated
on `FUSE_ALLOW_FETCH`: ArborX is required, and a default configure must not fail for want of it. It
is hand-written rather than routed through `fuse_require()` because the fetch path needs
`ARBORX_ENABLE_MPI` set to `FUSE_ENABLE_MPI` *before* `FetchContent_MakeAvailable` runs. The pin is by commit
(`8f89be60bc3797670b9be8843da7dcdbc5f2333f`, exactly tag `v2.1`; verified by the lead;
`UPSTREAMS.lock`'s `[arborx]` entry), built against the SAME Kokkos target this build already
resolved -- ArborX's own `CMakeLists.txt` calls `find_package(Kokkos 4.5 REQUIRED CONFIG)`, which
reuses the existing imported targets once Kokkos is already resolved rather than fetching a second
copy.

ArborX v2.1 asks for Kokkos >= 4.5; `/opt/kokkos` here is 5.1.0, whose `AnyNewer`
`KokkosConfigVersion.cmake` satisfies that request at configure time. It had never been compiled
against 5.1.0 before this track, so per the task's own requirement that was proven FIRST, standalone,
before any `fuse::` code was written: a `BoundingVolumeHierarchy` built over 100 boxes, an
`intersects` (point-in-box) CRS query, and a `nearest` (k-NN) CRS query all compiled and gave the
expected indices.

**One real API-shape finding from that standalone proof, which shaped `box_index.cpp`'s
implementation:** constructing the tree directly over bare geometry (`BoundingVolumeHierarchy(space,
boxes)`, which ArborX deduces `Value = Box`) and querying with the convenience
`(indices, offsets)` overload does not compile against a plain integer output view --
`Details::DefaultCallback::operator()` calls `out(value)`, and `value` is the `Box` itself, not an
index, for both spatial and nearest predicates. The fix is not an ArborX defect to patch around: it
is `ArborX::Experimental::attach_indices` plus a small custom callback that reads `value.index` --
exactly the shape `examples/distributed_tree/distributed_knn.cpp` in ArborX's own tree uses for its
own reason (tagging a rank onto the result). `box_index.cpp`'s `ExtractIndex` and
`distributed.cpp`'s `TagWithRank` are that callback for the local and distributed trees
respectively.

## Layering: BELOW `fuse::backend`, not above it

`docs/dev/layering.md` / `tools/check-layering.py`'s `LAYERS` list places `search` directly below
`backend` (between `field` and `backend`) -- the *opposite* direction from `fuse::partition`'s own
`mesh_adapter.hpp`, which sits above `backend` and freely includes `fuse/backend/mesh_view.hpp`.
The task's own placement requirement is that `fuse::backend` must be able to depend on
`fuse::search` (the next phase's nodal remap, which lives at or above `backend`, needs to call into
it), which fixes the direction: `search`'s rank must be *less than* `backend`'s.

That creates a real tension `mesh_adapter.hpp` (this layer's, not partition's) has to resolve:
task item 2 explicitly wants "an adapter that builds element bounding boxes from any fuse
MeshView", and `fuse::MeshView` is defined one layer *above* `search`. An `#include
"fuse/backend/mesh_view.hpp"` here would be exactly the upward dependency the gate exists to
catch.

**Resolution: `element_boxes<M>` is a template constrained by a structural (duck-typed) `MeshLike`
concept**, over the exact shape `mesh_view.hpp` documents (`dimension()`, `entity_count()`,
`is_owned()`, `simplex_count()`, `simplex_points()`, `centroid()` -- used only to deduce the point
type via `decltype`, never named). A caller that already has a concrete `fuse::MeshView&` in scope
(`fuse::backend` itself, once the nodal-remap track wires this in; a test that links
`fuse::backend`) instantiates the template there. Ordinary C++ template duck typing makes it
compile against the real class with no cast and no new virtual dispatch -- `element_boxes` never
sees `fuse::MeshView`'s name, only its shape, at the point it is instantiated by someone who does.
`test/search/mesh_adapter_test.cpp` proves this against `fuse::SerialMeshBackend`'s real generated
meshes; `test/search/refusals_test.cpp`'s `FakeMesh` proves the concept is genuinely structural (not
just satisfied by inheriting from `fuse::MeshView`) by instantiating the SAME template against an
unrelated hand-written type.

This is a substantive, not incidental, design decision worth the lead's confirmation -- see
`open_questions` in this track's report.

## Tolerance and padding: where "shared face finds both cells" actually lives

`BoxIndex` itself is **tolerance-agnostic**: `BoxIndex::build` takes boxes and indexes exactly what
it is given, with no relative-tolerance concept of its own. The "a point exactly on a shared face
between two elements is found in both of their boxes" property the task asks for is a single,
explicit, separately-testable function instead: `geometry.hpp::padded(box, tol)`, which grows a box
by `max(tol.relative * diagonal(box), tol.absolute_floor)` -- but **only on the box's own ACTIVE
axes**, leaving a degenerate one (an embedded-lower-dimension box's unused coordinate, which
`Point`'s convention requires to be exactly zero) untouched. That carve-out is not cosmetic: the
first implementation padded every axis unconditionally, and `mesh_adapter_test.cpp`'s
`GeneratedLineYieldsOneBoxPerCellAtDimension1` caught it immediately -- a 1D line mesh's boxes came
back classified as 3D, because symmetric padding turned `lo.z == hi.z == 0` into `lo.z == -p,
hi.z == +p`, two different floating-point values, and `classify_dim`'s exact-equality test (by
design, per `Point`'s own convention) read that as "this box has a real z-extent."

A caller that wants the shared-face property calls `padded()` explicitly before `BoxIndex::build`;
`element_boxes` (the mesh adapter) does this internally, so its output boxes are already grown.
`BoxIndex`'s own containment test, `geometry.hpp::contains`, is exact and closed on both ends
(`>=`/`<=`) -- `test/search/box_index_test.cpp`'s
`UnpaddedSharedFaceFindsOnlyOneCellByFloatingPointLuckOrBoth` pins that an EXACTLY-representable
shared coordinate is already found in both unpadded boxes by the closed boundary alone; the
`PointJustOutsideTheExactBoxIsFoundOnlyThroughPadding` test is the one that actually exercises
`padded()`'s growth, with a point that is not on any exactly-representable coordinate.

**"The containment decision proper belongs to the caller"** (the task's own words): `BoxIndex`'s
point query is a broad-phase, tolerance-widened AABB test, not a final geometric verdict. A caller
that needs, e.g., true barycentric or natural-coordinate containment for a curved or non-convex
element makes that decision itself, using `BoxIndex`'s candidates as the set worth checking rather
than trusting the box test as ground truth.

## Determinism

Every `QueryResult`/`DistributedQueryResult` group is sorted ascending by index (`(rank, index)` for
the distributed case) before being returned, regardless of query kind -- including `query_nearest`,
which is **not** sorted by distance. This is a deliberate reading of "downstream code never depends
on traversal order": a hit's position in ArborX's own traversal carries no relevance ranking (there
is none for a broad-phase candidate set), so sorting by distance would imply an ordering guarantee
this facade does not want to commit to. A caller that wants nearest-first order computes distances
itself; `geometry.hpp` deliberately exposes none, so nothing here silently picks a distance metric
ArborX's own is not required to agree with. `test/search/fixtures.hpp`'s `is_valid_knn` is the
correspondingly tie-tolerant reference `k`-nearest tests check against: with axis-aligned boxes that
can overlap a query point, several boxes legitimately tie at distance zero, and no k-nearest
algorithm -- ArborX's or a from-scratch one -- owes a *specific* one of them a slot over another;
only "every strictly closer box is present, no strictly farther one is" is owed, and that is what it
checks.

## What is verified where

* **Compiled and run here (Tier 0, host)**: `BoxIndex` build/dimension/size/empty, point queries
  against brute force in 1D/2D/3D, shared face (2D) and shared vertex (3D) tolerance behaviour, the
  exact-boundary and padding-only-finds-it cases, `k`-nearest against the tie-tolerant brute-force
  reference in 1D/2D/3D, result determinism, every refusal (`E-SEARCH-MIXED-DIM`, `E-SEARCH-K`,
  `E-SEARCH-DIM`, `E-SEARCH-NO-GEOMETRY`, `E-SEARCH-NO-MPI`), and the mesh adapter against
  `fuse::SerialMeshBackend`'s real generated line (1D) and box (3D) meshes.
* **Compiled here, UNRUN (Tier 1, `FUSE_ENABLE_MPI=ON`)**: `src/search/distributed.cpp`'s
  `ArborX::DistributedTree` path, and `test/search/distributed_mpi_test.cpp` at 1 to 5 ranks
  (`build-evidence/parallel-matrix.toml`'s `A-SEARCH-DISTRIBUTED` cell) -- this sandbox cannot
  initialise MPI at all (confirmed identically for `fuse::partition`'s own pre-existing Tier 1
  tests, not a regression introduced here), so no MPI test in this repository has ever produced a
  pass/fail verdict from inside it. Its verdict comes from a host run.
* **NOT covered by this track**: anything the nodal-remap track builds on top of `BoxIndex`'s API
  (see `api_for_next_phase` in this track's report) -- true geometric (natural-coordinate)
  containment, and any device (GPU) execution space. `BoxIndex` and `element_boxes` are host-only by
  construction (`Kokkos::DefaultHostExecutionSpace`), matching `fuse::partition::centroids_of`'s own
  scoping.

## Files

`include/fuse/search/{types,geometry,box_index,mesh_adapter,distributed}.hpp`,
`src/search/{box_index,distributed}.cpp`, `test/search/*.cpp`,
`build-evidence/mutations/search.toml`, this file.
