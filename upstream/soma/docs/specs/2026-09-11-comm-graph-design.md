# CommGraph — a rendezvous primitive for cross-communicator field transport

**Design specification**
**Date:** 2026-09-11
**Status:** §4 (the primitive), §6 (error handling) and §7 (the regression
matrix) **implemented and verified 2026-09-11**. §5 (data flow into the
Runner) **implemented 2026-09-11 for fully-disjoint rank sets**, in
`Runner::Impl::attach_comm_graphs` (`src/runtime/runner.cpp`) — see §9 for
exactly what ships, what is refused, and what is implemented but has no
caller yet. Two claims below were found to be wrong while building it and are
corrected in place, each marked **[CORRECTED 2026-09-11]**; `docs/HANDOFF.md`'s
R5 entry carries the measurements.
**Location:** `modules/coupler/`
**Namespace:** `soma::` (`soma::detail::CrystalRouter` for the internal transport primitive)
**Closes:** R5 (`docs/HANDOFF.md` Tier 1), for the fully-disjoint case. Does
**not** close R3's remaining item, R31 (partial overlap), R34, or R35 — see
"Explicitly out of scope" below and §9 for R31's status.

---

## 1. Problem statement

Spec §4.1 states that a component's ranks and the coupler's ranks may be
**disjoint**, requiring n→m redistribution — the `Transport` operator (§5.1).
`MpiTransport::build_plan()` (`src/transport/mpi_transport.cpp`) already
implements the *delivery* half of this correctly: given each rank's
`TransportSpec::global_ids` (the ids it holds, if a source, or wants, if a
destination), it matches by id — not position — and executes an `Alltoallv`.
This is verified by `mpi_transport_test.cpp`'s deliberately non-corresponding
source/destination partition pattern.

What is missing (R5's actual defect) is everything **upstream** of that:
`Runner::Impl::resolve_instructions()`'s `OpKind::Transport` case
(`src/runtime/runner.cpp:1104-1110`) populates only `src_rank_set`/
`dst_rank_set` (component ids, per `rules_channel.cpp:100`) and
`elements`/`components`. `TransportSpec::src_ranks`, `dst_ranks`, and
`global_ids` — the three fields `MpiTransport::build_plan` actually requires —
are never populated anywhere in `src/`. There is no mechanism today that
answers, for a real coupled graph, "which physical rank holds/wants this
data," whether by shared identity (coincident meshes) or by physical location
(non-coincident meshes).

This document specifies **`CommGraph`**, the primitive that answers that
question and feeds `TransportSpec`, closing R5.

## 2. Use case (confirmed against the spec, not assumed)

Spec §6 mode 1 — *"In-process C++... one executable, communicator split per
component"* — is the only integration mode v1 implements. Components are
disjoint (or partially/fully overlapping) subsets of ranks descending from one
`MPI_COMM_WORLD`. `Transport` moves field **values** between them, matched by
identity, not geometry — geometric interpolation is `Remap`'s job, and always
runs *after* `Transport` in v1 (`rules_channel.cpp:99`, "Transport before
Remap: v1 places remap on the target's ranks"). `CommGraph` is not mesh
migration (no geometry moves) and not itself a remap (it does not compute
interpolation weights) — it only determines *who talks to whom*.

Two matching regimes are both required (confirmed with the project owner):

1. **Global-ID matching** — source and destination already share an identity
   space for the coupled region (e.g. `test/data/pair.yaml`'s coincident
   `fluid_wall`/`solid_wall` mesh).
2. **Coordinate/nearest-neighbor matching** — source and destination do not
   share an identity space (non-coincident discretizations). The destination
   publishes coordinates; the source side resolves nearest-element.

Both must support **fully overlapping, fully disjoint, and partially
overlapping** component rank sets, and the resulting graph must be
**bidirectional** — usable in reverse for a channel's return leg (e.g. a
`Bidirectional`-intent FSI field) without rebuilding.

## 3. Explicitly out of scope

- **True separate-executable MPMD** (spec §6 mode 3: independent MPI
  universes, no common parent communicator, needing `MPI_Comm_connect`/
  `accept` and a rendezvous/port-publishing mechanism). Deferred by the
  project owner's explicit decision, matching the spec's own "design
  accommodates; not implemented in v1." *Launcher-style* MPMD (one `mpiexec`
  invocation running heterogeneous binaries under one shared
  `MPI_COMM_WORLD`) is **in scope** — it needs nothing beyond §5's
  union-communicator construction, since all ranks already share one world.
- **R3's remaining item** (`Runner` does not execute the parallel-remap halo
  exchange). Confirmed during design review that this is a *different*
  mechanism: `MoabMeshBackend::exchange_remap_halo`/`accumulate_remap_halo`
  (`src/mesh/moab_backend.cpp:2028-2068`) already work, built on MOAB's own
  `ParallelComm` machinery, which requires one shared `moab::Core` across all
  ranks — the opposite precondition from `CommGraph` (disjoint, potentially
  separate `moab::Core` instances). R3's remaining gap is a Runner
  instruction-scheduling and storage-ownership decision, not a matching
  problem, and gets its own follow-up spec.
- **R34** (Instant-chain slab-emission redesign, `src/schedule/expand.cpp`)
  and **R35** (remap doc caveat + diagnostic-tail-truncation fix,
  `src/remap/tempest_weights.cpp`) are unrelated to this design.
- Using MOAB's own `moab::gs_data`/`TupleList` Crystal Router
  (`/opt/moab/cmake-all/include/moab/gs.hpp`), despite it being a proven
  implementation of the same algorithm: it would make `CommGraph` (a
  transport-layer component) depend on the mesh-backend layer, breaking the
  "works for `SerialMeshBackend` too, no MOAB required" property and this
  project's existing practice of owning its core MPI primitives
  (`mpi_transport.cpp` does not depend on MOAB either).

## 4. Architecture

```
resolve_instructions()  →  CommGraph::build()  →  TransportSpec  →  MpiTransport::build_plan()
     (unchanged callsite)      (NEW, this doc)      (unchanged shape)   (UNCHANGED — verified correct)
```

`CommGraph`'s output, in **either** matching mode, is expressed in the same
currency `TransportSpec` already defines: "the list of source global ids this
rank holds/wants." In `GlobalId` mode that list comes from direct lookup; in
`NearestNeighbor` mode it comes from search. **`TransportSpec` and
`MpiTransport` require zero changes.**

New files: `include/soma/transport/comm_graph.hpp`,
`src/transport/comm_graph.cpp` (same tier as `mpi_transport.cpp` — compiled,
no MOAB dependency).

### 4.1 Union communicator

Each side supplies **its own communicator**, not an externally-imposed
`RankSet` sub-range (`RankSet{first,count}`, `component.hpp:13`, only
expresses a contiguous slice of one shared numbering — insufficient for
arbitrary partial overlap):

```cpp
struct CommGraphSide {
  const MeshView* mesh;         // this side's local view of the coupled region
  int             dim;          // topological dimension of the coupled entities
  int             comm_handle;  // Fortran MPI communicator handle, this component's own
};
```

`CommGraph::build()`:
1. `MPI_Comm_group` on each side's communicator.
2. `MPI_Group_union` the two groups.
3. `MPI_Comm_create_group(MPI_COMM_WORLD, union_group, tag)` to materialize
   the union communicator. `MPI_COMM_WORLD` is the necessary common ancestor
   — `MPI_Comm_create_group` requires a base communicator whose group is a
   superset of the target, and every rank is already in one MPI job under
   spec mode 1.
4. `MPI_Group_translate_ranks` maps each side's local rank numbers into the
   union's numbering.

This makes fully-overlapping, fully-disjoint, and partially-overlapping
communicators uniform data rather than special cases: the union rank of
"source rank 3" and "destination rank 3" are computed, never assumed equal or
assumed different.

### 4.2 CrystalRouter (new internal primitive)

A portable, dependency-free Crystal Router (Fox et al. 1988; the same
algorithm underlying `moab::gs_data`, reimplemented here to stay
MOAB-independent — see §3). Routes tagged messages to their destination in
O(log P) rounds via recursive dimension-doubling exchange.

```cpp
namespace soma::detail {

struct Envelope {
  int                    dest_rank;
  std::vector<std::byte> payload;
};

class CrystalRouter {
 public:
  explicit CrystalRouter(MPI_Comm comm);
  // Collective. Every rank in `comm` calls this, even with an empty
  // `outgoing`. Returns everything routed TO this rank.
  [[nodiscard]] Result<std::vector<Envelope>> route(std::vector<Envelope> outgoing);
};

}  // namespace soma::detail
```

`CrystalRouter` is a pure routing primitive: it does not know who owns what,
only how to deliver a message whose target is already known. Both matching
modes need a directory phase in front of it (§4.3, §4.4) to determine that
target.

### 4.3 GlobalId mode

Shard function: `bucket(global_id) = hash(global_id) % union_size`.

1. Every rank routes `"I own G"` (from its local ids) and `"I want G"` (from
   its wanted ids) to `bucket(G)`, via `CrystalRouter`.
2. The bucket rank reconciles both streams for each `G` it receives: it now
   knows the true owner's union rank and every requester's union rank.
   `owner_rank == requester_rank` → classified into `local_pairs()` directly,
   no further routing. Otherwise the owner's rank is routed back to the
   requester via a second `CrystalRouter` round.
3. A `G` requested but never owned anywhere → `E-COMMGRAPH-NOMATCH`.

No full allgather of ids ever occurs; each rank's traffic is bounded by its
own local id count plus whatever it receives as a bucket owner.

### 4.4 NearestNeighbor mode

Naive `hash(cell) % union_size` bucketing was considered and rejected: it
deliberately scrambles spatial locality, so non-uniform meshes (refinement,
boundary layers) can overload a handful of bucket ranks with both message
volume and a spatially-incoherent, expensively-large candidate set — flagged
explicitly by the project owner during design review.

**Adopted instead: Morton-code (Z-order) partition, load-balanced by
population** — the same principle behind space-filling-curve mesh
partitioners:

1. Compute the union's coarse bounding box: one `MPI_Allreduce`(MIN/MAX) over
   each side's local bounding box corners — O(1) data, not O(N).
2. Choose a cell size (`2 * nn_tolerance`) and bin every point (via
   `MeshView::centroid()`, already dimension-independent — `Point{x,y,z}`
   with unused axes zero, `mesh_view.hpp:13-18`) into a `CellKey{ix,iy,iz}`.
3. Compute each occupied cell's Morton code (bit-interleave `ix,iy,iz`;
   unused axes drop out for 1D/2D — the same dimension-independence as
   `Point` itself).
4. Each rank builds a **local histogram**: `(Morton code, local point
   count)` per occupied cell — sized by distinct occupied cells, not point
   count.
5. A lightweight exchange of these histograms selects `P-1` splitters along
   the Morton-ordered axis so cumulative point count is balanced across
   bucket ranks (pivot selection, as in parallel sample sort — cheap because
   it operates on the summary, not raw points).
6. Routing is a binary search into the splitter array, not a hash. Source
   points route to their cell's bucket; query points route to their cell's
   bucket **and** neighboring cells within `nn_tolerance` (`3^dim - 1`
   neighbors — 2 for 1D, 8 for 2D, 26 for 3D), all via `CrystalRouter`.
7. The bucket rank runs a local (brute-force is fine — population is bounded
   by construction) nearest-neighbor search among what it received, applies
   the tolerance check and a **lowest-global-id tie-break** (deterministic,
   independent of arrival order), and routes the answer home via
   `CrystalRouter`. `owner_rank == requester_rank` is resolved into
   `local_pairs()` at this same step, exactly as in GlobalId mode.
8. A query point with no candidate within `nn_tolerance` anywhere →
   `E-COMMGRAPH-NOMATCH`, naming the point and the nearest distance actually
   found.

Because Morton-adjacent cells land in the same or a neighboring bucket
(locality-preserving, modulo curve-seam edge cases — bounded and testable),
candidate sets stay spatially coherent and neighbor-bucket probing touches a
small constant number of buckets rather than being hash-scattered across the
whole communicator.

### 4.5 Public API

```cpp
enum class MatchMode { GlobalId, NearestNeighbor };

struct CommGraphOptions {
  MatchMode mode;
  double    nn_tolerance = 0.0;  // required (> 0) for NearestNeighbor; rejected pre-collective otherwise
};

struct LocalCopyPair {
  std::size_t source_local_index;
  std::size_t dest_local_index;
};

class CommGraph {
 public:
  [[nodiscard]] static Result<CommGraph> build(const CommGraphSide& src,
                                                const CommGraphSide& dst,
                                                const CommGraphOptions& opts);

  [[nodiscard]] TransportSpec source_spec() const;   // cross-rank subset only
  [[nodiscard]] TransportSpec dest_spec() const;      // cross-rank subset only
  [[nodiscard]] const std::vector<LocalCopyPair>& local_pairs() const;

  [[nodiscard]] CommGraph transpose() const;          // O(1), no collective
};
```

`transpose()` is a pure relabeling (swap which side is "source") — no
recomputation, no collective — because the matched-pair correspondence is
symmetric by construction in both modes. This is also what makes a
same-rank-copy pair a hard *requirement*, not merely an optimization:
`MpiTransport` documents (R24) that it refuses overlapping source/destination
rank sets by design (one `global_ids` list cannot describe a rank in both).
`CommGraph` splitting matched pairs into `local_pairs()` (never touching
`MpiTransport`) and a filtered remainder is what makes a
**partially**-overlapping component pair deliverable at all, not just faster.

**[CORRECTED 2026-09-11]** The sentence above originally said the remainder is
"a filtered, **genuinely-disjoint** remainder", and §4's claim that
"`TransportSpec` and `MpiTransport` require zero changes" rested on it. It is
false under partial overlap. The same-rank shortcut removes matches whose two
ends are on one rank; it cannot remove a rank that is a source for some
entities and a destination for others, which is ordinary: rank 2, in both
components, may own an entity rank 3 wants and want an entity rank 0 owns. No
fixed number of `MpiTransport` phases repairs it — two overlapping ranks
exchanging with each other is irreducible. The implementation therefore adds
`send_plan()`/`recv_plan()`, the complete per-peer answer, correct in all
three overlap shapes and directly executable as an `Alltoallv`, and keeps
`source_spec()`/`dest_spec()` for the case `specs_are_disjoint()` reports —
which is the fully-disjoint one `rules_channel.cpp` actually emits, and which
does go through `MpiTransport` unchanged.

Two further deviations from the sketch above, both deliberate:

- **`build()` takes the base communicator explicitly.** §4.1 assumes
  `MPI_COMM_WORLD`. A Fortran communicator handle has no portable "unset"
  sentinel, and an assumed world cannot be tested against a subset without
  spawning a job per case, so the handle is a parameter.
- **The union group is derived from a membership allgather, not from
  `MPI_Group_union` of the two sides' groups.** A rank in only the source
  component holds no destination communicator and so cannot call
  `MPI_Comm_group` on one; the two groups are not available anywhere to union.
  One `MPI_Allgather` of a membership byte over the base communicator gives
  the same set, on every rank, and additionally lets a rank in NEITHER
  component leave `build()` holding the same `src_ranks`/`dst_ranks` as
  everyone else — which it must, because it still has to enter
  `MpiTransport`'s collectives. The supplied communicators are still used:
  their `MPI_Comm_size` is checked against the reported membership.
- **`build()` frees the union communicator before returning.** Nothing in the
  result owns an MPI resource, so `CommGraph` is a copyable value and
  `transpose()` is free. The alternative — retaining it — would give
  `CommGraph` a collective destructor, which is exactly the hazard R24 records
  for `MpiTransport`. Consequently the rank sets are BASE-communicator ranks.

## 5. Data flow into the Runner

1. Lowering is unchanged — `rules_channel.cpp` still emits one `Transport`
   instruction per disjoint-rank channel direction.
2. `resolve_instructions()`'s `OpKind::Transport` case (`runner.cpp:1104`) is
   extended to build a `CommGraphSide` per end and call `CommGraph::build()`
   once, during `setup()` — never in the hot loop, matching
   `MpiTransport::build_plan()`'s existing "all cost up front" discipline.
   **New API surface required:** today exactly one shared communicator is
   threaded through `RunnerConfig`/the C ABI (R4); per-component
   communicators must be added to that surface. This is new work, not a
   reuse of an existing hook.
3. `Resolved::transport` gains the filtered `TransportSpec` and
   `local_pairs()`.
4. `execute()`'s `Transport` case does two things: `memcpy` every
   `LocalCopyPair` directly (no MPI), and hand the filtered `TransportSpec`
   to `MpiTransport::move()` exactly as today for the cross-rank remainder.
5. For a bidirectional channel's return leg, `resolve_instructions()`
   recognizes a second `Transport` instruction sharing the same `(region,
   unordered rank-pair)` within the same `setup()` call and reuses
   `.transpose()` instead of rebuilding.

## 6. Error handling

New codes, following the project's `E-<AREA>-<REASON>` convention:

- `E-COMMGRAPH-NOMATCH` — a wanted id (GlobalId) or point exceeding
  `nn_tolerance` (NearestNeighbor) has no source. Names the offending
  id/point and, in NN mode, the nearest distance actually found — mirroring
  `E-REMAP-COVERAGE`'s practice of naming the failure, not just reporting it.
- `E-COMMGRAPH-COMM` — union-communicator construction failed
  (`MPI_Group_union`/`MPI_Comm_create_group` returned an error).

Every collective step (union-comm construction, each `CrystalRouter` round,
splitter selection) follows the **same** discipline already documented at the
top of `mpi_transport.cpp`: local validation sets a code, `MPI_Allreduce`
(MAX) agrees it across every rank before any rank acts on the result. This is
not new philosophy — it is the existing discipline applied consistently to
new collective code, preserving `MpiTransport`'s no-partial-failure-deadlock
guarantee.

`nn_tolerance <= 0` under `NearestNeighbor` mode is rejected before any
collective work starts (a precondition check, not a runtime discovery).

Named degenerate cases:
- A rank contributing zero local entities on either side still calls
  `build()` and every collective inside it — same rule `MpiTransport`
  already follows ("a rank in neither rank set still enters
  `MPI_Alltoallv` with zero counts").
- Duplicate global ids within one side's own list are refused as a
  configuration error (ambiguous ownership), never silently resolved.

## 7. Testing

Every test in this list is mutation-checked against a deliberately broken
version before being trusted, per this project's standing discipline.

| # | Test | What it pins |
|---|---|---|
| 1–6 | Rank-overlap shape (fully-overlapping / fully-disjoint / partially-overlapping) × matching mode (GlobalId / NearestNeighbor) | The 6-way cross product; each combination is its own test |
| 7 | Same-rank copy never touches `MpiTransport` | Assert via the allocation/call-counting technique `mpi_transport_test.cpp` already uses; mutation-check by disabling the shortcut |
| 8 | Bidirectionality via `transpose()` | No second collective build fires on the reverse leg |
| 9–11 | Dimension independence | Identical NN fixture in 1D, 2D, 3D fixtures |
| 12 | NN tolerance boundary | Points exactly at / just inside / just outside `nn_tolerance` — pins the refusal boundary precisely |
| 13 | SFC load balance | A deliberately non-uniform (refined-patch) mesh at a rank count large enough to demonstrate the splitter partition balances bucket population (assert no bucket exceeds K× the mean) — ~~this is the test that would have caught the raw-hash problem flagged during design review~~ **[CORRECTED 2026-09-11: it is not.](#test-13-correction)** |
| 14 | Union-comm construction is exercised for real | Reuse `mpi_transport_test.cpp`'s deliberately non-corresponding contiguous/strided pattern, but drive it through two independently `MPI_Comm_split`-built communicators rather than one shared communicator with `src_ranks`/`dst_ranks` handed in directly |

<a name="test-13-correction"></a>

### 7.1 Test 13, corrected — and why the correction matters

**[CORRECTED 2026-09-11]** Test 13 as specified does **not** catch the raw
`hash(cell) % P` it was written to rule out. Checked by building both: on the
refined-patch fixture the hash produced buckets of 618 against a mean of 535
at two ranks and 409 against 338 at four — comfortably inside any sane K. A
hash balances *population* perfectly well whenever there are many cells of
comparable weight; it is only *cell counts* it randomises, and those are not
what the assertion watches. The test passes against the rejected design.

What a hash actually destroys is **locality**, and that is directly
observable: the number of distinct buckets a query must be probed against.
Under a Morton partition a query's `3^dim` neighbourhood is contiguous on the
curve and lands in one or two buckets; under a hash it scatters over
`min(3^dim, P)`. Measured on an 8³ lattice in 3D:

| ranks | Morton splitters | raw `hash(cell) % P` |
|---|---|---|
| 2 | 1.41 | 2.00 |
| 3 | 1.78 | 3.00 |
| 4 | 1.95 | 3.97 |
| 8 | 2.38 | 7.43 |

Both assertions are in the suite. The balance one is kept — it still catches a
degenerate splitter array — but the locality one is the load-bearing half, and
`CommGraphStats::query_probes`/`query_points` exist to make it measurable.

**Two further tests exist that the matrix did not call for**, both because a
mutation found the matrix silent:

- `CommGraph.AMatchOneCellAwayIsStillFound`. Every NearestNeighbor fixture in
  the matrix happens to place a query and its match in the *same* cell, so the
  entire `3^dim` neighbourhood probe — routing fan-out and bucket-side search
  alike — could be deleted with the suite still green. The new fixture places
  matches 0.02 apart with a cell size of 0.5 so that they are *always* one
  cell over.
- `CommGraph.AGhostedSourceEntityIsNotASecondOwner`, the positive control for
  the duplicate-ownership refusal. Without it, "refuse two owners" is
  satisfied by refusing every mesh that has ghosts.

## 8. Open items for the implementation plan

- ~~**STILL OPEN.** Exact plumbing for per-component communicators through
  `RunnerConfig`/the C ABI (§5.2) — a design decision in its own right,
  deferred to the `writing-plans` pass. This is what keeps §5 unimplemented
  and R5 open.~~ **CLOSED 2026-09-11.** `attach_comm_graphs()` splits
  `base_communicator` per component with `MPI_Comm_split`, entirely inside
  `Runner::Impl` — no new C-ABI surface was needed. See §9 for what actually
  shipped.
- ~~**STILL OPEN, and new.** Where the Runner keeps the gather/scatter buffers.
  `source_spec()`'s `global_ids` is a subset of a rank's entities, so
  `MpiTransport::move()` cannot be handed the field buffer directly; it needs
  a packed one, gathered through `source_indices()` and scattered through
  `dest_indices()`. Those must be sized in `setup()`, never in the loop, which
  is proved allocation-free (Task 4.6). `CommGraph` supplies the maps; the
  Runner has nowhere to put the buffers.~~ **CLOSED 2026-09-11.**
  `TransportPlan` carries `gather`/`scatter` index vectors and `send`/`recv`
  buffers, sized once in `setup()`. See §9 for what actually shipped.
- ~~**STILL OPEN, and new.** Which output shape a `Transport` instruction uses,
  given the §4.5 correction. `rules_channel.cpp` emits `Transport` only for
  disjoint-rank channels, so `source_spec()` covers what exists today — but
  R31 already records that the emission is decided by `RankSet::overlaps()`,
  which is coarser than disjointness, and that is the same seam.~~ **CLOSED
  2026-09-11, for what `rules_channel.cpp` emits today**: `source_spec()`/
  `dest_spec()` only; a non-disjoint plan is refused
  (`E-RT-TRANSPORT-OVERLAP`) rather than routed to `send_plan()`/
  `recv_plan()`. See §9 and the R31 addendum for what actually shipped and
  what remains.
- ~~Whether `CrystalRouter`'s `Envelope::payload` should be a raw byte buffer
  (as sketched) or a small templated helper~~ — **settled:** a raw byte
  buffer, with per-destination batching in `comm_graph.cpp`'s `Outbox` so the
  router carries one envelope per peer rather than one per record, and
  fixed-width padding-free wire structs `memcpy`'d in and out.
- ~~Morton-code bit width and cell-size heuristics beyond the `2 *
  nn_tolerance` default~~ — **settled:** 21 bits per axis (three axes in a
  63-bit code), and the cell size is `2 * nn_tolerance` COARSENED, never
  refined, when a tolerance tiny against the domain would overflow the axis
  resolution. Coarsening is the safe direction: the `3^dim` probe is
  sufficient for any cell size `>= 2 * tolerance`, because a source within
  `tolerance` of a query differs from it by at most `cell/2` on each axis and
  so cannot be more than one cell away.

## 9. §5 status, as implemented (2026-09-11)

`Runner::Impl::attach_comm_graphs()` (`src/runtime/runner.cpp`) is the §5
implementation. This section records what it ships, what it refuses, and
what it implements without a caller — superseding the "STILL OPEN" bullets
of §8, which described the plumbing gap before this work closed it for the
disjoint case.

**Ships:** fully-disjoint rank sets (`CommGraph::specs_are_disjoint()` must
hold); `MatchMode::GlobalId` matching only; whole-mesh regions (an
`entity_set` of `""`/`"all"`); exactly one CommGraph-backed `Transport` per
run. Per-component communicators are built once, in `setup()`, by
`MPI_Comm_split` over `RunnerConfig::base_communicator`, coloured by each
component's `RankSet` — every rank, including one in no component, takes
part in every split (`MPI_UNDEFINED` colour) so the operation stays
collective regardless of component membership.

**Refused, with the code and the reason.** This list is authoritative
against `src/runtime/runner.cpp`'s `E-RT-TRANSPORT*` string literals as of
commit `ae6c5b9` — broader than this document's original draft, which named
only four of the seven guards below:

- `E-RT-TRANSPORT-NOMPI` — `RunnerConfig::base_communicator` is set, but this
  is a non-MPI build (no `SOMA_HAVE_MPI`). There is no communicator to split.
- `E-RT-TRANSPORT-COMM` — communicator setup failed, at any of three points:
  `base_communicator` decodes to `MPI_COMM_NULL`; `MPI_Comm_rank` fails on the
  base communicator; or `MPI_Comm_split` fails while building one component's
  communicator.
- `E-RT-TRANSPORT` (this build wires one CommGraph-backed Transport per run)
  — the `ExecutionPlan` contains more than one `Transport` instruction. One
  `Transport` instance per instruction is named as the next step, not
  implemented here (see the follow-up below and `docs/HANDOFF.md`).
- `E-RT-TRANSPORT-XREGION` — the instruction's source and destination arcs
  name different regions. Transport preserves region by construction
  (`rules_channel.cpp` changes only `rank_set`), so this guard is what makes
  `MatchMode::GlobalId` provably correct: both ends are the same mesh, and
  global ids agree without a tolerance. Crossing regions would need
  `NearestNeighbor` with a per-channel tolerance the graph cannot express
  today.
- `E-RT-TRANSPORT-REGION` — the region names a non-whole entity set (not
  `""`/`"all"`). `MeshBackend` has no way to say whether a buffer for a named
  subset is indexed by set position or by mesh-local index — the same
  restriction, and the same reason, as `E-REMAP-REGION`.
- `E-RT-TRANSPORT-OVERLAP` (R31) — `CommGraph::build()` succeeded but the two
  components' rank sets are not disjoint. One `global_ids` list per rank
  cannot describe a rank that is simultaneously a source and a destination;
  closing this needs `CommGraph::send_plan()`/`recv_plan()` and a backend
  that consumes them, per R31 in `docs/HANDOFF.md`. The lowering does not
  emit `Transport` for this case today, so this guard is currently
  unreachable from `rules_channel.cpp`'s own output — it exists for the
  seam R31 already names (`RankSet::overlaps()` is coarser than exact
  disjointness).
- `E-RT-TRANSPORT` (a CommGraph-backed Transport plan requires the "mpi"
  transport backend) — reused a second time, for a different condition: a
  CommGraph-backed plan (`has_graph == true`) whose configured
  `RunnerConfig::transport` is not `MpiTransport` (e.g. `"copy"` while
  `base_communicator` is also set). Only `MpiTransport::build_plan()` knows
  how to turn the subset gather/scatter indices `CommGraph` produces into a
  real cross-rank exchange; any other backend given that spec would silently
  move the wrong data. Added during Task 4's fix round: refuse rather than
  execute wrong.

Not counted above: `execute()`'s `OpKind::Transport` case also returns
`E-RT-TRANSPORT` ("Transport instruction resolved to no plan",
`src/runtime/runner.cpp:1577`) if `Resolved::transport` is negative. That is
an internal-invariant guard against a `resolve_instructions()` bug, not a
documented scope refusal — it should be unreachable whenever `setup()`
succeeded, and is listed here only for completeness of the grep, not as
part of the refusal surface above.

**Implemented but uncalled:** `MatchMode::NearestNeighbor` is implemented
and tested in `CommGraph` (§4.4, §7 tests 9–13) but has no caller from the
Runner. `attach_comm_graphs()` always passes `opts.mode = MatchMode::GlobalId`
because `E-RT-TRANSPORT-XREGION` (above) already proves both sides are the
same region on the same mesh whenever it does not fire, which makes ids
match by construction. `NearestNeighbor` becomes reachable only if a future
lowering emits a `Transport` instruction whose source and destination arcs
name *different* regions, together with a per-channel tolerance value for
`CommGraphOptions::nn_tolerance` — neither exists in `rules_channel.cpp` or
`RunnerConfig` today.

**Residual risk carried from Task 4, not closed here** (see
`docs/HANDOFF.md`'s R5 entry for the full statement): the `mesh_id()`/`view()`
resolution inside `attach_comm_graphs()` sits below the loop's
`// From here down the loop is COLLECTIVE` marker and `break`s locally on
failure. A rank belonging to neither the source nor destination component,
whose mesh-loading scheme has it skip loading the coupled region's mesh,
would desync from member ranks that proceed into the collective
`CommGraph::build()`. Unreachable by every mesh backend and test topology
this plan exercises (every backend loads every mesh on every rank
regardless of component membership, and the test fixture partitions the
whole MPI world between exactly two components with no bystander rank), so
it is parked rather than fixed — the fix is a mesh-loading architecture
decision, out of this plan's scope.
