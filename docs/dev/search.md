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


### An installed MPI-enabled ArborX pulls MPI into a Tier-0 build

Found 2026-09-25 against an installed ArborX 2.1.99 built with `ARBORX_ENABLE_MPI ON`
(`/opt/arborx/share/cmake/ArborX`). Two things follow, and both bit:

1. **Its config needs MPI to resolve at all.** `ArborXConfig.cmake` runs
   `find_dependency(MPI)`, so on a machine where MPI is not discoverable the installed ArborX
   simply does not resolve — even for a build with `FUSE_ENABLE_MPI=OFF`, which wants nothing
   from MPI. `fuse_resolve_arborx()` then falls back to the pinned copy, and its message says so
   rather than the misleading "not installed".
2. **The failed find leaks a variable.** `ArborXSettings.cmake` does a plain
   `set(ARBORX_ENABLE_MPI ON)`, which survives in the caller's scope even though the find as a
   whole failed. A normal variable shadows a cache entry, so FORCE-ing the cache to `OFF` was not
   enough: the fetched ArborX's own `option()` kept `ON`, ran `find_package(MPI REQUIRED)`, and
   the configure died with `Could NOT find MPI` pointing inside `_deps/arborx-src/CMakeLists.txt`.
   The variable is now cleared before the fetch, and — the part that actually fixed it — ArborX's
   `FetchContent_Declare` carries **no** `FIND_PACKAGE_ARGS`: `MakeAvailable`'s own internal
   `find_package` retry re-leaked the variable *after* the reset, and the retry is redundant
   because `fuse_resolve_arborx()` has already done that find itself.

Verified by configuring Tier 0 with `/opt/arborx` on `CMAKE_PREFIX_PATH` and no MPI on `PATH`:
before, a hard `Could NOT find MPI`; after, a clean configure, the pinned copy, and 101/101. With
MPI available the installed 2.1.99 is still preferred.

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

## Execution space

`BoxIndex`/`query_points_distributed` build and query on
`Kokkos::DefaultHostExecutionSpace`/`Kokkos::HostSpace` by default -- byte-identical to this
facade's behaviour before this section existed -- or on `Kokkos::DefaultExecutionSpace` and its
`memory_space` when `fuse::search::resolve_execution_space()` (`types.hpp`) resolves to `Device`.
That resolution mirrors `RunnerConfig::resolved_execution_space()`'s own convention exactly --
`FUSE_EXECUTION_SPACE=device` in the process environment, or an explicit
`fuse::search::ExecutionSpace::Device` argument on `BoxIndex::build`/`query_points_distributed` --
without `fuse::search` depending on `fuse::runtime`: this layer sits below `backend`, which sits
below `runtime` (`docs/dev/layering.md`), so the enum and the reader live here, not as an include
of `runner.hpp`. `Device` requested on a build with no CUDA/HIP/SYCL backend degrades silently to
`Host` (`Kokkos::DefaultExecutionSpace` IS `Kokkos::DefaultHostExecutionSpace` there) -- never a
refusal; `BoxIndex::executes_on_device()`/`space_name()` report what the index actually built on,
not what was asked for.

**`Value` (`ArborX::PairValueIndex<ArborBox, int>`) names no space**, so one callback
(`ExtractIndex`) and one `Value` type index a `HostSpace` tree or a `Kokkos::DefaultExecutionSpace`-
memory tree identically; only the `ArborX::BoundingVolumeHierarchy<MemorySpace, Value>`/
`ArborX::DistributedTree<MemorySpace, Value>` template argument differs. `BoxIndex::Impl` carries a
tree of each kind; which one is live is `BoxIndex::device_`.

**Inputs and outputs.** The pre-existing `std::vector<Box>`/`std::vector<Point>` overloads of
`build`/`query_points`/`query_nearest` keep working unchanged, copying to the device internally
when `Device` is resolved, and copying the result back to a `std::vector`-based `QueryResult`
(sorted on the host, where `std::sort`'s O(k log k) applies). A device-resident caller that wants to
skip that round trip calls `build_device_resident`/`query_points_device_resident`/
`query_nearest_device_resident` instead -- DISTINCT NAMES, not overloads of `build`/`query_points`/
`query_nearest`: `BoxIndex::build({})` (an empty braced-init-list, no other argument to break the
tie) is equally valid as an empty `std::vector<Box>` or a default-constructed `Kokkos::View`, so
overloading on parameter type alone would make that pre-existing call -- `box_index_test.cpp`'s
`EmptyIndexIsValidNotARefusal`, unchanged since before this section existed -- ambiguous and a
compile error; a `std::vector<Box>` ALREADY IN HAND is never itself ambiguous against a
`Kokkos::View` parameter (no converting constructor exists either way), so this is purely the bare
`{}` literal's hazard, and a distinct name sidesteps it entirely. `build_device_resident` takes a
`Kokkos::View<const Box*, Kokkos::DefaultExecutionSpace::memory_space>` and converts `Box` to
`ArborX::Box` directly on the device (no host copy at all); `query_points_device_resident`/
`query_nearest_device_resident` take a `Kokkos::View<const Point*, ...>` and return a
`QueryResultView` whose `offsets`/`indices` stay in `Kokkos::DefaultExecutionSpace::memory_space`
throughout, including the per-query determinism sort -- an in-place insertion sort run as one Kokkos
thread per query directly on device memory (`sort_segments_on_device`, box_index.cpp), accepted as
O(k^2) per (typically short) hit segment in exchange for never touching the host.

**Building the boxes `build_device_resident` indexes, without a host round trip.**
`element_boxes` (`mesh_adapter.hpp`) is the host-side box builder, templated over the `MeshLike`
duck-typed concept so this layer never names `fuse::backend`. `element_boxes_device_resident` (same
header, `src/search/mesh_adapter_device.cpp`) is its device-resident counterpart: not a template
(there is no device-resident "duck typed" mesh concept to constrain against), it takes a simplex CSR
already in `Kokkos::DefaultExecutionSpace::memory_space` -- offsets, corner coordinates, corners per
simplex, and an optional per-entity `owned` flag mirroring `element_boxes`'s own `is_owned` filter --
and returns `Kokkos::View<Box*, DeviceMemSpace>`, compacted over `owned` with a `Kokkos::parallel_scan`
rather than a sequential `push_back`. A caller that can name a concrete mesh type
(`fuse::backend::NativeMeshView::device_simplex_offsets()`/`device_simplex_corners()`,
docs/dev/native-mesh.md's "Execution space") feeds this straight into `build_device_resident`: box
construction AND index construction for that mesh's elements never touch the host.

`query_points_distributed` threads the same `ExecutionSpace` through to its local `BoxIndex`
(`comm == kNullComm`) and to the `ArborX::DistributedTree` it builds directly otherwise
(`run_distributed_query<Space>`, distributed.cpp), copying `local_boxes`/`points` to the resolved
space's memory the same way the local facade does, and copying the ranked `(rank, index)` hits back
to host memory to assemble the `std::vector`-based `DistributedQueryResult` (no device-resident
overload for the distributed path -- out of this track's scope).

## Rank agreement in the distributed query

`query_points_distributed`'s dimension check (`E-SEARCH-MIXED-DIM`) is rank-local in its
computation but never in its RETURN: a rank whose own `local_boxes` fail `BoxIndex::build`'s
per-box dimension check, or whose `local_boxes` agree with themselves but disagree in dimension
with another rank's, does not refuse alone. Both cases are reconciled with an `MPI_Allreduce`
across the whole communicator (`MIN` of a locally-computed validity flag, then `MAX`/`MIN` of the
agreed dimension) BEFORE `ArborX::DistributedTree` is constructed -- the same agree-before-collective
recipe `fuse::native::migrate` and `fuse::partition::centroids_of` use for their own per-rank
rejections. A rank with zero local boxes casts no vote in the cross-rank comparison. Found
2026-10-11: the check used to run before the comm == kNullComm branch and return immediately on a
rank-local failure, which left every OTHER rank in `comm` entering the distributed tree's
collective construction -- a hang, not a refusal -- and the cross-rank disagreement case was not
checked at all. `test/search/distributed_mpi_test.cpp`'s `RefusesWhenOnlyOneRankSubmitsMixed...`
and `...DisagreeAboutDimension...` are the firing tests this fix added.

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
* **Compiled and run here (Tier 0, device)**: `BoxIndex::build`/`query_points`/`query_nearest` on
  `Kokkos::DefaultExecutionSpace` (Cuda on higgs's build), compared value-for-value against the Host
  build in the same process (`box_index_test.cpp`'s `DeviceBuildMatchesHostBuildForPointAndNearestQueries`),
  plus the `Kokkos::View`-based build/query overloads end to end against the `std::vector`-based
  result on the same index (`ViewBasedBuildAndQueryMatchVectorBasedOnTheSameIndex`). Every existing
  Tier-0 property test above also re-runs under `FUSE_EXECUTION_SPACE=device`
  (`fuse_add_device_twin`, test/CMakeLists.txt's `search.{box_index,refusals,mesh_adapter}.device`).
* **Compiled here, UNRUN (Tier 1, `FUSE_ENABLE_MPI=ON`, device)**: `run_distributed_query<DeviceExecutionSpace>`
  (distributed.cpp) -- this sandbox cannot initialise MPI at all regardless of execution space (the
  same pre-existing Tier 1 constraint the Host path already carries, above), so the device-mode
  `ArborX::DistributedTree` path has never produced a pass/fail verdict from inside this sandbox
  either.
* **NOT covered by this track**: anything the nodal-remap track builds on top of `BoxIndex`'s API
  (see `api_for_next_phase` in this track's report) -- true geometric (natural-coordinate)
  containment. `element_boxes` (the mesh adapter) stays host-only by construction
  (`Kokkos::DefaultHostExecutionSpace`), matching `fuse::partition::centroids_of`'s own scoping: it
  reads a `fuse::MeshView`'s `simplex_points`, which is itself a host-only accessor (see
  docs/dev/native-mesh.md's own "Execution space" section for what moved on the mesh side and what
  did not).

## Files

`include/fuse/search/{types,geometry,box_index,mesh_adapter,distributed}.hpp`,
`src/search/{box_index,distributed}.cpp`, `test/search/*.cpp`,
`build-evidence/mutations/search.toml`, this file.
