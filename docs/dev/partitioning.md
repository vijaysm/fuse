# Partitioning

`fuse::partition` is a Kokkos-view port of the user's `rcb_moab` (a MOAB/MPI Recursive Coordinate
Bisection partitioner living outside this repository). It splits a mesh's elements into
weight-balanced parts by recursively bisecting element centroids along their longest global
extent, using a histogram-refined parallel weighted median at each split -- the same algorithm
`rcb_moab`'s `RCBPartitioner::single_level_rcb` implements, carried over arithmetic-for-arithmetic
rather than redesigned.

## Generality

The core entry point, `rcb_partition<ExecutionSpace>` in `include/fuse/partition/rcb.hpp`, takes
three Kokkos views -- centroids (`N x 3` doubles), `int64` weights, `int64` global ids -- and
nothing that names a cell type, a topological dimension, or a manifold. A 1D line, a 2D planar
mesh, a 2D manifold embedded in 3D (a sphere is one case of this, reached through
`CoordTransform::Gnomonic`), and a 3D volumetric mesh of any cell type (hex, tet, prism, pyramid,
polyhedral, or a mix) all reach `rcb_partition` through the identical call: the algorithm only
ever sees three doubles and two integers per element. `include/fuse/partition/mesh_adapter.hpp`'s
`centroids_of` is the one place a `fuse::MeshView` is read, for callers that have one; per-element
weights always come from the caller, never derived from the mesh.

## Determinism

`rcb_moab`'s reproducibility argument carries over unchanged because every piece of it survived
the port intact:

* **Axis selection** is the longest *global* extent (`MPI_MIN`/`MPI_MAX` Allreduce, or the
  identity when `comm == kNullComm`), with a `depth % 3` fallback when every axis is degenerate
  (all points coincide, or there is no input at all) -- so every rank picks the same axis without
  exchanging anything beyond the extent itself.
* **The weighted median** (`include/fuse/partition/median.hpp`) sums `int64` weights, and `int64`
  addition is associative: the histogram Allreduce is bit-exact regardless of communicator size or
  rank layout. The classification a caller performs afterwards uses the median's final *bucket
  bounds* (`bucket_lo`/`bucket_hi`), not strict comparison against a single `split_value` --
  `rcb_moab`'s own `ParallelMedian.hpp` documents why: on a structured grid many cells share a
  coordinate value within floating-point tolerance of the converged split, and strict `<`/`>`
  would classify that shared value inconsistently depending on which side of it roundoff placed
  `split_value`.
* **Ties** (elements inside the bucket-bound band) are sorted by global id and distributed by an
  ascending-weight walk until the left target weight is met -- the same rule, the same order of
  operations, including `std::llround` for the target weight -- so two runs with the same global
  input produce the same tie assignment regardless of how that input is split across ranks.
* **Bucket-index arithmetic is identical on every execution space by construction, not by compiler
  flag.** The one place a bucket index is computed (`detail::accumulate_histogram` in
  `median.hpp`) writes `(c - cur_lo) * inv_dx` as two separate operations -- a subtraction, then a
  multiplication -- with no addition folded into either. A fused multiply-add can only change the
  result of an `a*b+c` pattern, and there is no such pattern in that expression, so it needs no
  `#pragma STDC FP_CONTRACT OFF`: the guarantee comes from the expression's shape. The other place
  this port could have introduced an FMA-sensitive `a + b*c` pattern -- updating the bucket bounds
  from the chosen bucket, once per refine iteration -- is deliberately kept as host-only scalar
  code in `weighted_median`, where exactly one code path is ever compiled, so no host/device
  divergence is possible there either.
* **The collective sequence is unchanged.** `rcb_partition`'s recursion issues the same Allreduce
  calls in the same order at the same recursion depth that `rcb_moab`'s
  `recursive_bisect_elements` does; the base case is `part_hi - part_lo <= 1`, checked *before*
  anything else, never `indices.empty()` -- an empty local subset is not a reason to skip a
  collective every other rank still calls (`rcb_moab`'s `CLAUDE.md` "Global collective discipline"
  documents a hang this exact mistake caused there once).

This port is deliberately **not** restructured into level-synchronous batching or any other
efficiency change: robustness and accuracy come first, and a restructuring is deferred until
parity with the oracle below is verified end to end, including on the user's host with real MPI.

## The serial oracle

`test/partition/rcb_oracle.hpp` is a header-only, test-only re-implementation of `rcb_moab`'s
recursion and median, copied arithmetic-for-arithmetic (including `std::llround` and the exact
order of operations) from `RCBPartitioner.cpp` and `ParallelMedian.cpp`. It runs *P emulated
ranks* in one process: the caller partitions a global index space into P owner slices, and every
collective the real algorithm would issue over a communicator is instead an explicit reduction
over those P slices' local arrays. Summing `int64` weights this way computes exactly what
`MPI_SUM` would (no reassociation hazard), and `MIN`/`MAX` over doubles are order-independent, so
the emulation is not an approximation -- it is what a real Allreduce would answer, for any P, run
in a single process. This is what makes `rcb_moab`'s `RcbBfBAcrossRanks` property (partitioning
the same global input at different communicator sizes gives identical part assignments) testable
without a working MPI launcher.

## What was ported, and what is deferred

Ported: the single-level RCB recursion and its histogram-refined weighted median (the entire
algorithm above); per-element weights supplied by the caller; an optional pre-bisection coordinate
transform, with `CoordTransform::Identity` as the default and `CoordTransform::Gnomonic` -- the
per-face tangent-plane projection from `rcb_moab/CoordTransform.cpp`, offered for sphere-like
manifolds and never assumed -- as the only other option
(`include/fuse/partition/coord_transform.hpp`). Degenerate inputs are handled by the *same* code
path as the general case, not a special branch: all centroids identical, an empty local input, an
empty rank in a real communicator, zero total weight, and a single part (`num_parts == 1`) all
recurse and terminate normally rather than being special-cased.

Deferred, and absent rather than silently ignored:

* `rcb_moab`'s **coarse-projected RCB** path (`N_global > 1,000,000`, a `G^3` grid pre-pass) --
  the single-level path is the one the task asked to port faithfully first; the coarse path is a
  distinct algorithm, not a variant of this one, and adding it means a second oracle.
* **Pole-cap handling** and **face-aware RCB** (`rcb_moab`'s skew-triggered per-cube-face dispatch
  in `RCBPartitioner::partition()`) -- both are refinements *on top of* the gnomonic transform this
  port already carries; the transform is ported, the dispatch logic that chooses between plain
  RCB, pole-aware RCB and face-aware RCB per input is not.
* **The post-partition refinement pass** (`rcb_moab`'s `PartitionRefine`, a Kernighan-Lin-style
  local-swap improvement) and **partition scopes** (per-region independent partitioning) -- neither
  changes the recursion this task ported; both are separate features layered on top of it.

None of the above is called from anywhere in `fuse::partition`; there is no ignored option field
standing in for them.

## What is verified where

* **Bit-for-bit parity with the oracle**, at the library's Tier 0 path (`comm == kNullComm`,
  `num_parts` independent of the (single) participant count), across every fixture family named in
  the task: a 1D line, 2D planar (a jittered grid, standing in for a mixed tri/quad mesh), a
  cubed-sphere-like point set on the unit sphere (with `CoordTransform::Gnomonic`), a 3D structured
  hex box, and a 3D unstructured point cloud (standing in for a mixed tet/prism/pyramid/polyhedral
  mesh) -- `test/partition/rcb_partition_test.cpp`, typed over every enabled Kokkos execution
  space.
* **The oracle's own multi-rank properties** -- comm-size independence (`P = 1` vs `P = 2..5`
  emulated ranks give identical assignments), the named distribution shapes (uniform, skewed, all
  points identical, an empty emulated rank, zero total weight), and imbalance within a bound on a
  uniform fixture -- checked on the oracle alone, without Kokkos or MPI, in
  `test/partition/oracle_test.cpp`.
* **The mesh adapter's happy path** against `SerialMeshBackend`'s generated line and box meshes
  (the only mesh source that exists at the time of this port) in
  `test/partition/mesh_adapter_test.cpp`, including the case that motivates reading
  `MeshView::dimension()` rather than scanning every dimension for "any owned entity": a box mesh
  legitimately has owned entities at both its cell dimension and its boundary-skin dimension at
  once, and a scan-based design would misread that as mixed-dimension input on every ordinary
  volumetric mesh.
* **Every refusal named in the task** -- `E-PART-NUM-PARTS` (negative `num_parts`),
  `E-PART-SIZE-MISMATCH` (centroids/weights/gids length disagreement), and (in a Tier 0 build only)
  `E-PART-NO-MPI` (a non-null communicator handle with no MPI compiled in) -- fires under a test in
  `test/partition/refusals_test.cpp`, which is labelled `throws`.
* **`E-PART-MIXED-DIM`** is, by construction, a cross-rank property: `centroids_of` Allreduce-agrees
  `MeshView::dimension()` (both `MAX` and `MIN`, so disagreement is detectable and not just which
  value "wins") *before* reading a single entity, and `comm == kNullComm` (one participant) agrees
  with itself trivially. The disagreement this refuses -- two ranks loading meshes of different
  dimension -- can therefore only be produced with two or more real ranks. It is compiled, not run,
  in `test/partition/rcb_mpi_test.cpp`'s `MeshAdapterMpi.RefusesWhenRanksDisagreeAboutMeshDimension`
  (see below).
* **Everything that needs a real communicator** -- the collective sequence itself (the axis-extent
  Allreduce, the median's histogram Allreduce loop, the tie-weight Allreduce, the tie Allgatherv)
  under more than one *real* rank, and the mixed-dimension refusal above -- is compiled in
  `test/partition/rcb_mpi_test.cpp` and registered as an MPI test, but this sandbox cannot start an
  MPI launcher (`mpiexec` cannot bind a port here, and a singleton `MPI_Init` aborts). Those results
  are "compiled, awaiting host run", not verified, until run on the user's host.
