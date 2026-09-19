<!-- Detailed layer review, 2026-09-19, HEAD bdf4346. Consolidated in ../../REVIEW.md; unified ids there are R-nn, this file's ids are BK-<id>. -->

# FUSE backend review — HEAD bdf4346

Scope: `include/fuse/backend/**`, `src/backend/**`, `test/backend/**`. Method: full
line-by-line read of every in-scope header and source file; no build was attempted
(no working MPI in this sandbox, per prior context) and no file was compiled — every
finding below is from direct reading plus, where noted, `grep`/`find` confirmation of
absence (test files, path references). File:line citations are against HEAD
`bdf4346`.

## 1. Coverage

Read in full, line by line:

- Docs: `docs/dev/invariants.md`, `layering.md`, `conventions.md`, `testing.md`,
  `porting.md`, `rank-agreement.md`, `operators.md` (all 7, ~730 lines total).
- `tools/check-layering.py`, `include/fuse/base/config.hpp.in`.
- Headers (all 10, ~2,190 lines): `mesh_view.hpp`, `moab_backend.hpp`,
  `linear_operator.hpp`, `weights_operator.hpp`, `serial_backend.hpp`,
  `supermesh.hpp`, `remap_plan.hpp`, `native/mesh.hpp`, `cell_simplices.hpp`,
  `spacetime.hpp`, `src/backend/high_order_backend.hpp`.
- Sources (all 9, ~6,960 lines): `moab_backend.cpp` (2,214 lines, full),
  `intx_weights.cpp` (2,181 lines, full), `tempest_weights.cpp` (587 lines, full),
  `serial_backend.cpp` (649 lines, full), `supermesh.cpp` (406 lines, full),
  `linear_operator.cpp`, `weights_operator.cpp`, `spacetime.cpp` (full).
- Test file inventory for all 7 files under `test/backend/` (every `TEST(...)` name
  enumerated via `grep`, ~140 tests) to assess what is and is not exercised; did not
  read every test body line by line (effort was spent confirming *absence* of
  parallel/degenerate-cell coverage, not re-deriving what the titles already state).
- `build-evidence/backend-mutations.md` (mutation run, 2026-09-18, 15/16 killed).
- Cross-checked every `src/mesh/`, `src/remap/`, `test/unit/` path reference cited
  in scope-file comments against the actual tree (`find`), and one such file's
  content in `upstream/soma/` for comparison (see Finding 4).

Not read line-by-line: `test/backend/*.cpp` test *bodies* beyond what was needed to
confirm coverage gaps (titles + targeted `grep` were sufficient for the findings
below); `upstream/soma/` beyond the one file cross-checked in Finding 4.

## 2. Findings table

| id | severity | file:line | title |
|---|---|---|---|
| B1 | BLOCKER | `src/backend/intx_weights.cpp:420-474`, `1659-1662`, `1706-1741`, `1915-1918` | Per-rank data-dependent `resolve()`/`fit_surface()`/`orient()` failures precede the collectives `build_remap_halo`/`build_weights` are documented to reach on every rank |
| B2 | BLOCKER | `src/backend/moab_backend.cpp:1829-1835`, `1879-1898`, `1918`, `1927` | `plan_halo_exchange` can return per-rank on inconsistent global ids between two `MPI_Alltoall(v)` calls, hanging the surviving ranks |
| B3 | BLOCKER | `include/fuse/backend/native/mesh.hpp:99-103`, `144-182` | `Mesh<M,S>` has no execution/memory-space parameter; `finalize()` is host-only code that indexes the same views a GPU-default Kokkos build would place in device memory |
| M1 | MAJOR | `docs claim vs. reality`: `include/fuse/backend/moab_backend.hpp:31,65,141,242,251`; `include/fuse/backend/remap_plan.hpp:40,138,143,324,444-445`; `src/backend/*.cpp` (6 files, 23 refs) | Code comments cite `test/unit/remap_parallel_test.cpp` and four sibling files as the evidence for parallel/order/halo correctness; none of those files exist in FUSE, and the one upstream analog tests a refusal FUSE deliberately removed |
| M2 | MAJOR | `src/backend/supermesh.cpp` (`inward_plane`, in the `intersect` helper) | A degenerate clipping simplex (zero-length edge) produces an always-true half-plane instead of a zero-measure clip, inflating overlap for exactly the case `MeshView` says is expected |
| M3 | MAJOR | `src/backend/supermesh.cpp` (`build_supermesh_weights`) | No check that per-cell coverage stays ≤ 1 + tol, unlike the MOAB path's `over_src`/`over_tgt` audit (`intx_weights.cpp:1306-1345`) — the exact defect class M2 produces goes undetected here |
| M4 | MAJOR | `include/fuse/backend/native/mesh.hpp:189-233` vs. `238-241` | `measure()`, `centroid()`, `normal()` are not `KOKKOS_INLINE_FUNCTION` and call the host-only `FUSE_ASSERT`, contradicting the file's own claim (lines 57-62) that derived quantities are meant to be read from parallel kernels — only `vertex()` actually is |
| MIN1 | MINOR | `src/backend/spacetime.cpp` (`apply_spacetime`) | Zero-temporal-weight early return skips all spatial-operator bounds/null validation for that call, so a malformed spatial triplet array is only ever caught on calls where the temporal coefficient happens to be nonzero |
| MIN2 | MINOR / DEVIATION | `src/backend/serial_backend.cpp:595-636` vs. `include/fuse/backend/mesh_view.hpp:274` | `make_region_sizes`, declared as a backend-agnostic adapter in the generic `MeshView` contract header, is defined in `serial_backend.cpp` even though nothing in its body is serial-backend-specific |

## 3. Findings in full

### B1 — BLOCKER — divergent per-rank refusal before a documented collective

**Where.** `src/backend/intx_weights.cpp`, function `resolve()` (lines 353-485),
called from `build_remap_halo` (lines 1659-1662) and from `build_weights` (lines
1915-1918); `fit_surface()`/`orient()` calls inside `build_remap_halo` (lines
1706-1741); the first true MOAB collective reached afterwards is
`worker.FindMaxEdges` (line 1752), and `build_weights`'s own collective is
`reduce_audit`'s three `MPI_Allreduce` calls (lines 1161-1166, called from line
1332).

**What is wrong.** `remap_plan.hpp:506` documents `build_remap_halo` as
"**COLLECTIVE**." `moab_backend.hpp:74-81` states, as a *known limitation*, that
`MeshView::dimension()` "reports the highest dimension this RANK holds entities
of. A rank whose partition is empty ... therefore reports a smaller number than
its peers. Agreeing on one answer needs a collective reduction ... Callers that
need a mesh-wide dimension must reduce it themselves." `resolve()` uses exactly
this per-rank value as the fallback for `s.dim` (line 417: `s.dim = r.dimension
>= 0 ? r.dimension : s.view->dimension();`) and then refuses locally, with no
reduction, if it is out of `[1,3]` (lines 420-425). It refuses locally again if
this rank's partition holds zero cells at that dimension (`range.empty()`, lines
447-450), if the MOAB/`MeshView` local counts disagree (lines 439-446), or if
*this rank's* copy of a cell has non-positive measure (lines 468-474) — any one of
which can legitimately differ from what a peer rank sees for the same region, per
the backend's own documented limitation. `build_remap_halo` calls `resolve()` for
both sides (1659-1662) and, further down, `fit_surface()` (1706, 1708) and
`orient()` (1737, 1739) — both per-rank-local geometric fits with their own
early-return `Failure` paths — before ever calling `worker.FindMaxEdges` (1752),
`build_processor_euler_boxes` (1760) or `construct_covering_set` (1775), which are
MOAB-internal MPI collectives. `build_weights` calls the same `resolve()` (1915,
1917) before it can reach `reduce_audit`'s `MPI_Allreduce`s. So: if rank 0's local
partition is empty at the requested dimension (an ordinary consequence of
partition imbalance, not a bug) while rank 1's is not, rank 0 returns `Failure` at
`intx_weights.cpp:447-450` and never calls `MPI_Allreduce`/`FindMaxEdges`/etc.;
rank 1 proceeds into those calls and blocks forever waiting for rank 0. This is
precisely the failure mode `docs/dev/rank-agreement.md` and invariant `I2` exist
to rule out ("A rank that fails agrees the failure with its peers before
throwing or entering a later collective"), and it is not ruled out here: nothing
between `resolve()`/`fit_surface()`/`orient()` and the first collective performs
the reduction the header itself says a caller in this position must do.

I confirmed this is not merely theoretical by checking that the codebase already
knows the fix pattern and applies it elsewhere: `reduce_audit`
(`intx_weights.cpp:1146-1184`) exists *specifically* to reconcile a
possibly-per-rank-divergent audit before the function's own final refusal, with
the comment at 1265-1284 explaining why an unreconciled per-rank check would be
wrong. That reconciliation is applied to the audit outcome; it is not applied to
`resolve()`/`fit_surface()`/`orient()`, which run earlier and can diverge first.

**Fix.** In `build_remap_halo` and `build_weights`, immediately after (or inside)
`resolve()`/`fit_surface()`/`orient()`, when a communicator is present, perform an
`MPI_Allreduce(MPI_MIN)` (or equivalent) of a local success/failure flag across
`pcomm->comm()` before any rank proceeds to the next MOAB collective call; on any
rank reporting failure, every rank returns the same `Failure` (carrying the
failing rank's message, gathered with a second short collective or via rank 0
broadcasting it) rather than only the rank that hit the local condition. This is
the same pattern `reduce_audit` already implements two functions away.

**Test to add.** An MPI test (2+ ranks) that partitions a source mesh so one
rank's partition is empty at the remap dimension (or holds a degenerate cell)
while its peer's is not, then calls `build_remap_halo`/`build_weights` and
asserts every rank returns the *same* `Failure`, with a wall-clock timeout on the
test process so a regression reports as a failing assertion rather than a hung
CI job — following the same discipline `test/support/fuse_test_main` already
applies at the harness level for exactly this hazard (see `docs/dev/testing.md`
harness section).

### B2 — BLOCKER — divergent per-rank refusal inside `plan_halo_exchange`

**Where.** `src/backend/moab_backend.cpp`, `MoabMeshBackend::Impl::plan_halo_exchange`
(lines 1810-1941 read in full). Per-rank-only `Failure` returns at line 1829-1835
(`dd.gid_error` — this rank's own global-id scan), line 1879-1886 (a local
non-owned entity's global id is claimed by no rank), and line 1891-1896 (a local
entity is marked not-owned yet its global id maps to this rank). These sit
between the `MPI_Allgather`/`MPI_Allgatherv` pair (lines 1853-1865) and the
`MPI_Alltoall`/`MPI_Alltoallv` pair (lines 1918-1931).

**What is wrong.** The `MPI_Allgather` at 1853 and `MPI_Allgatherv` at 1861 are
reached unconditionally by every rank (nothing data-dependent gates them), so
those two are safe. But the loop at 1876-1898, which builds `want` per owning
rank, returns `Failure` on **this rank's own** local entity/global-id
inconsistency (1880-1885, 1891-1895) — a condition that depends on this rank's
partition and this rank's global-id numbering, and can therefore fire on one
rank without firing on its peers (e.g. a genuine numbering bug that happens to
only produce an orphaned id on one rank, or the `dd.gid_error` case the class
itself documents at `moab_backend.hpp` as arising because "PARALLEL_RESOLVE_
SHARED_ENTS creates interface edges with GLOBAL_ID left at MOAB's -1 default" for
some dimensions and not others — a condition inherently local to which entities a
given rank happens to hold). A rank hitting either check returns before line
1918's `MPI_Alltoall`; its peers, having no such local inconsistency, reach line
1918 and 1927 and block waiting for the failed rank.

**Fix.** Same shape as B1: before the `MPI_Alltoall` at line 1918 (or right after
building `want`), reduce a local ok/fail flag across `comm` and have every rank
return the same `Failure` if any rank's local check failed. The `dd.gid_error`
check at 1829 is simpler still — it can be reconciled at the top of the function
before the first collective (1853) rather than after, since it needs no
per-rank data beyond what is already local at function entry.

**Test to add.** An MPI test (2+ ranks) that manufactures a global-id
inconsistency local to one rank (e.g. by patching a non-owned entity's stored
global id after `load()` so `owner_of` cannot resolve it on that rank only), and
asserts every rank observes the same `Failure` rather than a subset hanging.

### B3 — BLOCKER — native `Mesh<M,S>` has no memory-space parameter; `finalize()` assumes host-visible views

**Where.** `include/fuse/backend/native/mesh.hpp`. View aliases at lines 99-103:
`Points = Kokkos::View<double* [SpatialDims]>` etc. — no explicit memory space,
so each defaults to `Kokkos::DefaultExecutionSpace::memory_space` at compile
time. `finalize()` (lines 144-182) is a plain (non-`KOKKOS_FUNCTION`) member
function that loops `for (std::size_t c = 0; c < simplex_count(); ++c)` and reads
`cells_(c, v)` (148), `points_(cells_(c,v), k)` (175), `parent_(c)` (157) directly
from host code — it also constructs `Failure{...}` with `std::to_string` (150-154,
158-163), which is unambiguously host-only.

**What is wrong.** The file's own header comment (lines 33-38) explains the
*dimensions* are template parameters purely for indexing efficiency in device
kernels, and the class exists, per lines 60-62, so that "the whole point of this
layer is to be used from parallel kernels." But nothing pins the *memory space*
of `points_`/`cells_`/`parent_`/`centroid_`/`measure_` to host memory. In any
Kokkos build configured with a discrete-memory default execution space (Cuda
without UVM, HIP, SYCL) — the exact configurations this "native Kokkos mesh"
scope item is asking to be checked against — these views allocate in device
memory, and `finalize()`'s host-side loop dereferencing them via `operator()` is
an illegal host access to device memory: undefined behaviour at best, a runtime
abort or silent corruption at worst, and not caught at compile time (Kokkos does
not disallow calling a plain host member function; the illegality is in what
that function does to the view's backing memory, checked only at runtime and
only when Kokkos's own access-space debug checks are compiled in).

**Fix.** Add a `MemorySpace` template parameter to `Mesh` (default
`Kokkos::HostSpace`), so `finalize()`'s host assembly is always valid regardless
of the build's default execution space: `Points = Kokkos::View<double*
[SpatialDims], MemorySpace>` etc. If a device-resident copy is needed for a
kernel, build it explicitly via a mirror type and `Kokkos::deep_copy` after
`finalize()`, rather than letting the class silently inherit whatever
`Kokkos::DefaultExecutionSpace` happens to be. This also removes the
awkwardness noted in M4 below, since a host-resident `Mesh` and a device-usable
read accessor become two clearly separate things instead of one class trying to
be both.

**Test to add.** A compile-time check (or a CI matrix leg) that builds this
header against a Kokkos configuration whose default execution space is a
discrete-memory device and confirms `finalize()` still type-checks/runs against
host-space views specifically — the existing `native_mesh_test.cpp` suite only
runs against whatever the default build's execution space is (almost certainly
host-only here), so it cannot see this by construction.

### M1 — MAJOR — documented parallel/halo/order test evidence does not exist in FUSE

**Where.** Comments citing `test/unit/remap_parallel_test.cpp`,
`test/unit/remap_halo_parallel_test.cpp`, `test/unit/remap_test.cpp`,
`test/unit/remap_order_test.cpp`, and `test/unit/moab_mesh_test.cpp` appear 23
times across `include/fuse/backend/moab_backend.hpp` (lines 31, 65, 141, 242,
251), `include/fuse/backend/remap_plan.hpp` (lines 40, 138, 143, 324, 444-445),
`src/backend/high_order_backend.hpp`, `src/backend/moab_backend.cpp`, and
`src/backend/intx_weights.cpp` (lines 203, 1380, 1644, 1947, plus the two shared
with the header list).

**What is wrong.** `find . -iname "*remap_parallel_test*"` etc. across the whole
repository returns these filenames **only** under `upstream/soma/test/unit/`;
`src/mesh/`, `src/remap/`, and `test/unit/` do not exist anywhere in the FUSE
tree (`src/backend/` and `test/backend/` are what exist). Per
`docs/dev/porting.md`, "Port the tests first. They must pass against the
renamed copy before any behavioural change. These are the oracle for everything
that follows" — that step was not done for this functionality: the tests were
never carried into `test/backend/`. Worse, the one upstream file I read
(`upstream/soma/test/unit/remap_parallel_test.cpp`) tests that `build_weights`
**refuses** a partitioned read outright with `E-REMAP-PARALLEL` on more than one
rank — behaviour `intx_weights.cpp:385-391` explicitly says was *removed* ("There
is no refusal here any more") in favour of the halo-based path this review
covers. So even a mechanical copy of that file would assert the wrong contract
against current FUSE; the comments citing it as "pinning" today's halo/order
behaviour are describing tests that neither exist nor, if resurrected verbatim,
would compile against what the code now does. Per `docs/dev/testing.md`'s five
states, the parallel-halo functionality these comments repeatedly invoke as
evidence is, at best, "implemented, untested" — and the divergent-collective
hazards in B1/B2 above sit exactly inside this untested area. This is also
compounded by `build-evidence/backend-mutations.md`, which has zero entries for
`moab_backend.cpp`, the MOAB branch of `intx_weights.cpp`, or `tempest_weights.cpp`
— only the native/serial/supermesh(-native-path)/operator files have mutation
coverage.

**Fix.** Either port and adapt real MPI test files for the halo/parallel path
under `test/backend/` (matching current semantics, not upstream's removed
refusal) and update the citing comments to point at them, or — if that is
deferred — change every one of the 23 comments to say plainly that this
behaviour is untested rather than naming a file that does not exist, so a
reader doing exactly what I just did does not have to rediscover it.

**Test to add.** See B1/B2's "test to add"; this finding is the reason those do
not already exist.

### M2 — MAJOR — degenerate clipping simplex silently admits area instead of clipping to zero

**Where.** `src/backend/supermesh.cpp`, the `inward_plane<Dim>` helper and its
use inside `intersect()` (the r2d/r3d clipping routine backing
`build_supermesh_weights`).

**What is wrong.** `inward_plane` computes a face normal from an edge/cross
product, tests its length, and only normalises (and only flips the sign to
point inward) `if (len > 0.0)`. When the clipping simplex has a zero-length
edge or degenerate face — which happens exactly when a backend hands the
supermesh a degenerate cell, the case `mesh_view.hpp` explicitly documents as
expected and "detectable" via a zero-measure simplex — `len` is 0, the branch is
skipped, and the plane is left as `n = (0,0,0)`, `d = 0`. The half-space test
`dot(n,x) + d >= 0` then reads `0 >= 0`, which is unconditionally true: that
face imposes **no constraint at all** on the clip, instead of correctly
constraining the intersection to nothing. Since `intersect()` only checks the
*target* polygon for degeneracy (a `twice_area == 0.0` early return), a
degenerate *source* (clipping) simplex is not caught, and the clip can retain
more of the target polygon's area than the true (zero) overlap — a correctness
defect, not a hang, and one that inflates conservation-relevant weights rather
than refusing.

I checked `build-evidence/backend-mutations.md` and `test/backend/
supermesh_test.cpp`: the existing killed mutation `supermesh-planes-point-
outward` exercises a *well-formed* triangle with reversed winding, not a
degenerate one, and no test in `supermesh_test.cpp` constructs a degenerate
source/target cell through the r2d/r3d path (the one degenerate-simplex test in
scope, `native_mesh_test.cpp:99` `ADegenerateSimplexHasZeroMeasureAndNotANaN`,
exercises `native::Mesh::measure()`, an unrelated code path). This defect is
untested by both the unit suite and the mutation suite.

**Fix.** In `inward_plane`, when `len <= 0.0`, do not silently return a
zero/always-true plane. Either detect the degenerate simplex before calling
`intersect()` (in `lift()`, alongside the existing `ns == 0` check, using the
already-computed simplex measure) and skip it — contributing zero to that
parent cell's coverage, consistent with "a degenerate cell shows up as a
degenerate simplex, which has zero measure" — or have `intersect()` itself
return 0 immediately when any face of either input is degenerate, mirroring
the existing `twice_area == 0.0` early return for the target.

**Test to add.** A `Supermesh` test pairing a source mesh containing one
genuinely degenerate cell (via a backend or hand-built `MeshView` returning a
zero-measure simplex with two coincident corners) against a well-formed target,
asserting the degenerate cell contributes exactly zero weight rather than a
plausible-looking nonzero one.

### M3 — MAJOR — native supermesh path has no over-coverage audit check

**Where.** `src/backend/supermesh.cpp`, `build_supermesh_weights`'s audit
construction (the loops computing `min_src_coverage`/`min_tgt_coverage` etc.).
Contrast with `src/backend/intx_weights.cpp:1301-1345`, the MOAB path's audit,
which explicitly checks `c > 1.0 + kCoverageTol` for both source and target
cells and refuses with `E-REMAP-INTX` when a cell is assigned more area than it
has.

**What is wrong.** The native (always-available) supermesh path computes
`min_src_coverage`/`min_tgt_coverage` (minimums) but never checks the
corresponding *maximum* against 1 + tolerance. A bug that inflates overlap —
such as M2 above, or any future regression in the r2d/r3d integration — would
show up in the MOAB path as a hard refusal (`E-REMAP-INTX`, "the intersection
or its parent attribution is wrong") but would pass silently through the native
path with a wrong, conservative-*looking* weight matrix. Given `supermesh.hpp`'s
own framing of this path as "the differential oracle for the optional paths,"
an oracle that cannot catch its own over-coverage is a weaker guarantee than
the path it is meant to check against.

**Fix.** Add the same `covered/measure > 1 + tol` check (source and target)
to `build_supermesh_weights`'s audit, returning a refusal analogous to
`E-REMAP-INTX` rather than only recording a minimum.

**Test to add.** Same fixture as M2's test, asserting that if the fix in M2 is
reverted (or under an injected-fault build), `build_supermesh_weights` refuses
rather than returning a matrix with `covered/measure > 1`.

### M4 — MAJOR — `measure()`/`centroid()`/`normal()` are not device-callable despite the class's own stated purpose

**Where.** `include/fuse/backend/native/mesh.hpp`, `measure()` (189-192),
`centroid()` (194-198), `normal()` (200-233) versus `vertex()` (238-241).

**What is wrong.** The file's introduction (lines 57-62) states: "the whole
point of this layer is to be used from parallel kernels. They [the derived
quantities computed by `finalize()`] are computed in one pass by `finalize()`
and are read-only thereafter" — directly implying `measure()`/`centroid()`/
`normal()` should be callable from a Kokkos device kernel once `finalize()` has
run. Only `vertex()` (238) carries `KOKKOS_INLINE_FUNCTION`. The other three
call `FUSE_ASSERT` (190, 196, 208, 220, 229), which `fuse/base/error.hpp`
documents as throwing, formatting via `fmt`, and capturing a `Backtrace` —
unconditionally host-only machinery; the same header explicitly names
`FUSE_DEBUG_ASSERT` as "the one debug-only check, for device kernels ... where a
throw is impossible or unaffordable," implying `FUSE_ASSERT` itself was never
meant to run on a device. Without a `KOKKOS_INLINE_FUNCTION`/`KOKKOS_FUNCTION`
marker, `measure()`/`centroid()`/`normal()` will not compile at all if called
from inside a Kokkos device lambda under CUDA/HIP — so a kernel author reaching
for a cell's precomputed measure (exactly the allocation-free, precomputed
per-step read this layer exists to make cheap, per the file's own invariant-3
rationale) cannot get it there, contradicting the class's documented purpose
for these three members.

**Fix.** Either mark `measure()`, `centroid()`, and `normal()`
`KOKKOS_INLINE_FUNCTION` and replace their internal `FUSE_ASSERT(finalized_,
...)` with `FUSE_DEBUG_ASSERT(finalized_)` (checking `finalized()` once, with a
real `FUSE_ASSERT`, at the kernel's launch boundary instead — the pattern
`error.hpp` already documents), or narrow the file's own claim at lines 57-62 to
say only `vertex()` is meant for device use and that the derived scalars are a
host-side convenience. As written, the code and the doc comment disagree.

**Test to add.** A device-build compile check that calls `mesh.measure(c)` (or
`centroid`/`normal`) from inside a `Kokkos::parallel_for` lambda and expects it
to compile — it should currently fail to compile under a CUDA/HIP leg, which
would make this gap visible in CI rather than only on inspection.

## 4. Not found

Checked for and *not* found, i.e. absent rather than merely unread:

- No `#ifdef` on a `#cmakedefine01` macro anywhere in the reviewed headers or
  sources (`FUSE_HAVE_MPI`/`_MOAB`/`_TEMPEST`/`_KOKKOS`/`_PETSC`/`_LOWFIVE`,
  `FUSE_LINALG_EIGEN`, `FUSE_ENABLE_EXCEPTIONS` are all correctly tested with
  `#if`). The locally-scoped `FUSE_MOAB_PARALLEL`/`FUSE_REMAP_PARALLEL` macros are
  plain `#define`s (not `#cmakedefine01`), and are correctly tested with
  `#ifdef`/`#ifndef` — this distinction is handled correctly everywhere I read.
- No `throw std::...` bare-throw in library code in scope; every raise goes
  through `FUSE_THROW`/`FUSE_REFUSE`/`FUSE_ASSERT` or returns a `Failure`.
- No `moab/` or `mpi.h` include in any public header in scope (confirmed by
  grep across `include/fuse/backend/**`); MOAB and MPI headers are confined to
  the two `.cpp` translation units the code's own comments say they are
  (`moab_backend.cpp`, `intx_weights.cpp`).
- No `iostream`/`printf` formatting in the reviewed sources; `fmt` is used
  throughout for user-facing text.
- No `soma::`/`KARMA::` namespace or `KARMA*`/`karma_*`/`soma_*` naming residue
  in any in-scope file (`tools/check-layering.py`'s own scan would also catch
  this, but I checked the scope files directly).
- No throw-and-catch used as *internal* control flow within a single function
  in the reviewed sources; the one `try`/`catch` I found that matters
  (`intx_weights.cpp` around `s.view->measure()`, and `spacetime.cpp`'s
  `build_temporal_projection` wrapping `build_impl`) converts a
  contractually-throwing dependency (`MeshView::measure`, `Rational`'s
  overflow-checked arithmetic) into a `Result`, which is the documented
  pattern, not the internal-control-flow anti-pattern `KNOWN_ISSUES.md` B1a
  warns about.
- No exact-time value crossing an ABI as `double`; `spacetime.cpp`/`.hpp` use
  `fuse::Rational` throughout for interval and slab arithmetic, converting to
  `double` only for the final numerical weight, never for a comparison that
  decides correctness.
- No test in `test/backend/` exercises a multi-rank scenario at all (confirmed
  by grepping every `TEST(...)` name in the directory — none reference ranks,
  `pcomm`, or halo exchange under `mpiexec`); this is consistent with, and is
  the direct cause named in, Finding M1.
- `MoabMeshBackend::Impl`'s move constructor/assignment (`= default` on a
  `std::unique_ptr<Impl>`) does not by itself explain any lifetime bug — views
  and native-mesh handles obtained before a move remain valid, exactly as
  `moab_backend.hpp:87-92` documents, and I found no code path that reads
  `impl_` after a move without first checking for null.
- `WeightsOperator`'s row/col-sum and composition math (`linear_operator.cpp`,
  `weights_operator.cpp`) checks out algebraically against the documented
  composition rule (`operators.md`); I verified the row-sum/col-sum
  composition formulas by hand and found no defect there, consistent with the
  mutation suite's 4/4 kill rate on that file.
- No allocation-inside-a-step pattern found in the Kokkos-adjacent code
  reviewed (`native/mesh.hpp`'s only per-step-shaped reads are `vertex()`,
  which allocates nothing); `cell_simplices.hpp`'s decomposition tables are
  `constexpr`/stack-only.

## 5. Suggested new tests / mutations

1. MPI (2+ rank) test: unbalanced partition (one rank empty at the remap
   dimension) through `build_remap_halo`/`build_weights` — asserts a matched
   refusal on every rank, not a hang. (B1)
2. MPI (2+ rank) test: inject a global-id inconsistency local to one rank,
   through `plan_halo_exchange`/`exchange_remap_halo` — asserts a matched
   refusal. (B2)
3. Device-build (or Kokkos device-space-forced host build) compile/run check
   for `native::Mesh::finalize()` and for calling `measure()`/`centroid()`/
   `normal()` from a device lambda. (B3, M4)
4. `Supermesh` unit test with one degenerate source cell against a well-formed
   target: assert zero contributed weight, not a plausible nonzero one. (M2)
5. `Supermesh` audit test: assert `build_supermesh_weights` refuses when a
   cell's covered fraction exceeds 1 + tolerance, mirroring
   `intx_weights.cpp`'s `E-REMAP-INTX` case. (M3)
6. `Spacetime` unit test: a shape-corrupted `SpatialWeightsView` (out-of-range
   row/col index) paired with a `(src_step, tgt_step)` combination whose
   temporal coefficient is exactly zero — assert `apply_spacetime` still
   reports the shape error rather than silently succeeding. (MIN1)
7. Mutation candidates for `tools/mutate.py`, none currently present for these
   files: a degenerate-plane mutation in `inward_plane` (kill target: the new
   test in #4); a "skip the per-rank reduction" mutation in `resolve()`/
   `plan_halo_exchange` marked `requires = "mpi"` (kill target: #1/#2).

## Summary

9 findings in scope: **3 BLOCKER**, **4 MAJOR**, **2 MINOR** (one of the four
MAJOR findings, M1, is a documentation/process finding rather than a code
defect, but it directly explains why B1 and B2 were never caught). The three
BLOCKERs are two independent divergent-collective hang hazards in the MOAB
halo/remap-halo path (B1, B2) and a Kokkos memory-space gap in the native mesh
that breaks on any GPU-default build (B3). The MAJOR findings cover a
silent-correctness gap in the always-available supermesh path for degenerate
cells (M2, compounded by a missing audit check, M3), a mismatch between the
native mesh's documented device-usability and what actually compiles for a
device (M4), and the discovery that the parallel/halo functionality's cited
test evidence does not exist in this repository (M1). No file in scope was
compiled or run; all findings are from direct reading, cross-checked against
the governing docs and, where a claim was checkable mechanically (file
existence, macro definitions, test titles), against the repository itself.
