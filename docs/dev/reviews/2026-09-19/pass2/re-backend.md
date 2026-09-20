<!-- Second-pass review, 2026-09-19/20, HEAD dbdc872. Verifies the backend fix pass
against the review at bdf4346 (docs/dev/reviews/2026-09-19/backend.md) and the fixer's
report (docs/dev/reviews/2026-09-19/fixes/fix-backend.md), then re-reviews the same scope
fresh: include/fuse/backend/**, src/backend/**, test/backend/**. -->

# FUSE backend second-pass review — HEAD dbdc872

## Method

Built `tier0` (Debug, Eigen linalg, MOAB/MPI both off) into a private tree with
`cmake --preset tier0 -S <repo> -B $PWD/b0` and `cmake --build $PWD/b0 -j4`; all 71
registered tests pass (`ctest -j4` -> `100% tests passed, 0 tests failed out of 71`; the
`backend` label alone: 6/6). Ran `tools/mutate.py --area backend`: 18/18 mutations
KILLED, including the two the fix pass added
(`apply-spacetime-skips-validation-at-zero-weight`, `supermesh-no-over-coverage-audit`).
Ran `tools/mutate.py --check-anchors` (whole repo): 86/86 anchors valid, 0 stale -- the
fixer's disclosed stale anchor (`mesh-accepts-a-parent-outside-the-cell-set`) has since
been corrected by someone else, as the fixer's report asked. Ran `tools/check-layering.py`
and `tools/check-invariants.py`: 0 violations each. Read every commit in `git log
main..HEAD` that touches `src/backend/`, `include/fuse/backend/`, `test/backend/`, or
`build-evidence/mutations/backend-*.toml` (6 commits: `f42ff21`, `b4edd98`, `6e4c220`,
`cc4a2cd`, `e65c4dc`, `743846c`) against the code at HEAD, not against the fixer's prose.
MOAB+MPI cannot link on this machine (recorded constraint: the installed MOAB was built
against a since-upgraded compiler), so the R-01/R-02 reconciliation logic itself could not
be compiled or run here either -- confirmed by reading the diff and the surrounding
control flow, not by building it, exactly as Part 1 asks when a fix is MPI-only.

## Verification

| id | verdict | evidence |
|---|---|---|
| R-01 (BK-B1) | **PARTIAL** | `resolve()`'s own checks are correctly wrapped in an IIFE and reconciled once via a new `reconcile<T>()` (`src/backend/intx_weights.cpp:374-429`, `579`); `fit_surface()`/the sphere-classification payload/`orient()` are each reconciled at their call sites in `build_remap_halo` (`intx_weights.cpp:1809,1811,1819-1844,1856,1858`). Code-read confirms `reconcile()` sits behind `#ifdef FUSE_REMAP_PARALLEL` (`intx_weights.cpp:38`, defined only when `MOAB_HAVE_MPI && FUSE_HAVE_MPI`), matches `reduce_audit`'s existing shape, and is a no-op on `nproc<=1`. All 71 tier0 tests pass and the `#ifndef` no-op path is unaffected (this code is MOAB-gated and does not compile on tier0). **What remains**: the fix reconciles resolve()'s pass/fail *verdict* but not the `Side::dim` *value* it carries forward, and `moab_backend.hpp:74-81`'s own documented limitation (`MeshView::dimension()` can legitimately differ by rank) still reaches two unreconciled comparisons on that value one call frame up -- see new finding N-BACKEND-1, which reproduces the identical hang class this fix targets. Cannot compile/run the reconciliation itself here (MOAB does not link on this machine); the fixer's syntax-only check against real MOAB+MPI headers is not independently reproducible in this sandbox and needs a host run to execute at 2+ ranks, exactly as `docs/dev/testing.md`'s "mutations that cannot be assessed locally" rule anticipates. |
| R-02 (BK-B2) | **CONFIRMED-FIXED** (for the three call sites the finding named) | `dd.gid_error` reconciled at `moab_backend.cpp:1897-1902`, strictly before the first collective (`MPI_Allgather` at `1922`); the two want-loop refusals no longer `return` from inside the loop -- they set `local_error` and `break` (`1954-1965`), reconciled once at `1975-1978`, strictly before `MPI_Alltoall` at `1998`. `reconcile_halo()` (`1825-1863`) has the same any-fail/lowest-rank/broadcast shape as `intx_weights.cpp`'s `reconcile<T>`. All 71 tier0 tests pass; `#ifndef FUSE_MOAB_PARALLEL` stub path unchanged. A related but distinct per-rank refusal the original finding did not examine (it is downstream of both collective pairs, not between them) remains -- see N-BACKEND-2 -- so this does not fully close every hang-shaped defect in this function, only the two the finding cited. |
| R-03 (BK-B3) | **CONFIRMED-FIXED** | `Mesh` gained a `MemorySpace` template parameter defaulting to `Kokkos::DefaultExecutionSpace::memory_space` (`include/fuse/backend/native/mesh.hpp:105-131`); `finalize()` now assembles through `Kokkos::create_mirror_view`/`deep_copy` (`181-231`) rather than touching `points_`/`cells_`/`parent_` directly, so the host-only refusal logic (`Result`, `std::to_string`) never dereferences a device view. `simplex_measure` is templated on the view type so it can run on either the live views or the host mirrors (`324-367`). Verified: `backend.native_mesh` passes (part of the 71/71 tier0 run); confirmed by grep that the only consumer, `test/backend/native_mesh_test.cpp`, names only the first two template arguments everywhere, so the new default leaves every existing call unchanged. The device-resident (non-host-accessible `MemorySpace`) path is, as the fixer states, unexercised on this machine -- no CUDA/HIP/SYCL Kokkos build is available here to check it; this is a genuine "needs a host run" limit, not a gap in the fix. |
| R-11 (BK-M1) | **CONFIRMED-FIXED** | Grepped `include/fuse/backend/`, `src/backend/`, `test/backend/` for `test/unit/`, `src/mesh/`, `src/remap/` after the fix: every remaining `test/unit/...` occurrence (9 left, `moab_backend.hpp:143`, `remap_plan.hpp:40,153,335,460-461`, `intx_weights.cpp:205,1740,2066`) is now inside a sentence stating the cited file **does not exist**, never cited as evidence that a claim is verified -- confirmed by reading each in context, not just grep-counting. `src/mesh/`/`src/remap/` path citations are gone (0 hits). The specific replacement tests named as what IS actually tested were checked against the tree and exist: `Supermesh.AConstantSurvivesAndTotalMassIsPreserved`, the three `*IsExactlyConservative` tests, `Supermesh.ObservedOrderOnANonMatchingPairIsAtLeastOne` (`test/backend/supermesh_test.cpp:58,84,120,259,564`), `MoabMesh.MixedElementMeasuresMatchClosedForms` (`test/backend/moab_mesh_test.cpp:344`). |
| R-12 (BK-M2) | **PARTIAL** | `inward_plane<Dim>` now computes `len` immediately after building the normal and returns `false` (a real, checked return value, previously void) before `d` is ever computed when `len<=0.0` (`src/backend/supermesh.cpp:104-135`); both call sites (`intersect<2>`, `intersect<3>`) now clip to `0.0` on a degenerate face instead of proceeding (`172,200`). This is the exact fix the finding proposed and reads correctly by inspection: a zero-length/zero-area face can no longer produce the vacuous `(0,0,0),0` half-plane. **What remains**: `test/backend/supermesh_test.cpp` is byte-for-byte unchanged between `main` and `HEAD` (`git diff main..HEAD -- test/backend/supermesh_test.cpp` is empty) -- no dedicated regression test asserting "a degenerate source cell contributes exactly zero weight" was added, which is what the original finding's "Test to add" asked for. The fixer's report says this directly ("A black-box regression test isn't practical... the fix itself was re-verified correct by inspection after reverting") and defers it under "Needs another owner." The only safety net against a future regression here is the mutation entry added for R-13 (below), which covers over-coverage, not the degenerate-face path itself -- there is still no mutation or test targeting `inward_plane` directly. |
| R-13 (BK-M3) | **PARTIAL** | `build_supermesh_weights`'s audit gained a `c > 1.0 + kSupermeshCoverageTol` refusal (`E-REMAP-INTX`) on both the source and target sides (`supermesh.cpp:370-406`), `kSupermeshCoverageTol = 1e-9` matching the MOAB path's `kCoverageTol` exactly (`intx_weights.cpp:215`). Independently re-ran the mutation this session (`tools/mutate.py --area backend`): `supermesh-no-over-coverage-audit` **KILLED**, confirming the audit is live, not decorative. **What remains**: same gap as R-12 -- no permanent GoogleTest case was added (`test/backend/supermesh_test.cpp` diff is empty); the only verification is the mutation TOML entry, and `docs/dev/testing.md` itself says the mutation gate is "run before a port is accepted... not [yet] a CI leg," so a `ctest` run alone would not catch a regression here. |
| R-14 (BK-M4) | **PARTIAL** | `measure()`, `centroid()`, `normal()` are now `KOKKOS_INLINE_FUNCTION` and use `FUSE_DEBUG_ASSERT` instead of the host-only `FUSE_ASSERT` (`mesh.hpp:237,252-254,259-263,271-304`), matching the finding's proposed fix exactly. All 71 tier0 tests pass. **What remains, found this pass**: `FUSE_DEBUG_ASSERT`'s non-`NDEBUG` branch calls `::fuse::detail::debug_abort(...)` (`include/fuse/base/error.hpp:206-210`), and that function's declaration (`error.hpp:214`) carries no `KOKKOS_FUNCTION`/`__device__` annotation. On this OpenMP/Serial-default build `KOKKOS_INLINE_FUNCTION` compiles to a plain host `inline`, so nothing here is exercised -- but on a real CUDA/HIP/SYCL-default Kokkos build in a Debug (non-`NDEBUG`) configuration, `measure()`/`centroid()`/`normal()` would try to call a host-only function from device code, which does not compile. That is exactly the scenario the original finding's own "Test to add" named ("a device-build compile check... calling `measure()`... from inside a `Kokkos::parallel_for` lambda and expects it to compile"). The fixer's report discloses this precisely, under "Needs another owner" ("`debug_abort` ... is not itself marked `KOKKOS_FUNCTION`... a future CUDA/HIP/SYCL... build would need `debug_abort` to be device-callable... to actually link on device") -- so the gap is known, not hidden, but it means the finding's device-callability claim is not yet true in every build configuration, only in the Release (`NDEBUG`) case where `FUSE_DEBUG_ASSERT` is a no-op. |
| R-33 (BK-MIN1) | **CONFIRMED-FIXED** | `apply_spacetime`'s early return moved from before the triplet-validation loop to inside it, gated per-triplet rather than for the whole call (`src/backend/spacetime.cpp:488-512`). Directly ran the new test: `fuse_backend_spacetime_test --gtest_filter=*ValidatesTripletsEvenWhenTheTemporalWeightIsZero* -> 1/1 PASSED`. Independently re-ran the mutation: `apply-spacetime-skips-validation-at-zero-weight` **KILLED**. Both directions of the test (malformed triplet caught at `w==0`; well-formed triplet still applies no contribution at `w==0`) are present and correct by reading. |

## New findings

| id | severity | file:line | title |
|---|---|---|---|
| N-BACKEND-1 | BLOCKER | `src/backend/intx_weights.cpp:1764`, `1794`, `2041` | The R-01 fix reconciles `resolve()`'s pass/fail verdict but not the rank-local `Side::dim` value it carries forward; three unreconciled `dim` comparisons downstream reproduce the identical divergent-refusal-before-a-collective hazard |
| N-BACKEND-2 | MAJOR | `src/backend/moab_backend.cpp:2017-2022` | A fourth per-rank-local refusal in `plan_halo_exchange`, outside the three R-02 reconciled, can still return a different verdict from different ranks for the same call |
| N-BACKEND-3 | MINOR / DEVIATION | `src/backend/moab_backend.cpp:1825-1829` vs. `src/backend/intx_weights.cpp:374-387` | `reconcile_halo()` silently ignores `MPI_Comm_size`/`MPI_Comm_rank` failure instead of refusing, unlike the sibling `reconcile<T>()` the fix pass describes it as mirroring |

## New findings in full

### N-BACKEND-1 -- BLOCKER -- reconciling resolve()'s verdict is not enough; the `dim` value it carries is still rank-divergent and unreconciled

**Where.** `src/backend/intx_weights.cpp`: `build_remap_halo`, `if (a.dim != b.dim)` at
line 1764 and `if (a.dim != 2)` at line 1794; `build_weights`, `if (a.dim != b.dim)` at
line 2041.

**What is wrong.** `resolve()` (fixed under R-01) now reconciles whether it *succeeded*
on every rank, but the *value* of `Side::dim` it computes is still purely local:
`s.dim = r.dimension >= 0 ? r.dimension : s.view->dimension();` (line 498). When a
region's dimension is left unspecified in the coupling graph (`Region::dimension`
defaults to `-1`, per `include/fuse/model/region.hpp:22`, meaning "mesh dim" -- the
ordinary, not the exceptional, configuration), `s.dim` falls back to
`s.view->dimension()`, which `moab_backend.hpp:74-81` documents, unprompted, as its own
**KNOWN LIMITATION**: `dimension()` "reports the highest dimension this RANK holds
entities of. A rank whose partition is empty, or which holds only faces of a volume
mesh, therefore reports a smaller number than its peers... Callers that need a
mesh-wide dimension must reduce it themselves." `resolve()` only returns `Failure` when
its OWN `s.dim` is outside `[1,3]` or its own local partition is empty at that
dimension -- neither check requires two ranks to have picked the SAME `s.dim`. So two
ranks can each individually succeed in `resolve()` (each finds a valid, non-empty local
partition at whatever dimension its own `dimension()` returned) while disagreeing on
the dimension itself -- e.g. rank 0's local partition of a region is non-empty at
dimension 2 and empty at dimension 1, so `dimension()` returns 2; rank 1's local
partition of the SAME region happens to hold only boundary edges locally, so
`dimension()` returns 1. Both ranks' `resolve()` calls reconcile to "success" (there is
nothing to disagree about at that level), but `a.dim` is `2` on rank 0 and `1` on rank
1. `build_remap_halo`'s `if (a.dim != b.dim)` (1764) or `if (a.dim != 2)` (1794) then
evaluate to different booleans on the two ranks with no `reconcile()` call guarding
either -- one rank returns `Failure` immediately, the other proceeds toward
`worker.FindMaxEdges` (1866, guarded further down) exactly as B1 originally described,
just one comparison earlier than the checks R-01 patched. The same shape recurs in
`build_weights`'s own `if (a.dim != b.dim)` at line 2041, ahead of `has_remap_halo`'s
decision and `reduce_audit`'s `MPI_Allreduce`s. `adopt_remap_halo`/`has_remap_halo`
(`moab_backend.cpp:2101-2189`) do purely local map bookkeeping -- no cross-rank check
exists anywhere in this call chain that would catch the disagreement before it reaches
a collective.

**Fix.** Either reconcile the `dim` value itself once, immediately after both `resolve()`
calls return (an `MPI_Allreduce(MPI_MAX)` or similar on `a.dim`/`b.dim`, refusing if the
reduced value disagrees with a rank's own before proceeding -- the same shape `resolve()`
itself uses for measure totals elsewhere), or wrap each of the three comparisons in
`reconcile()` the same way `fit_surface()`/`orient()` were wrapped, so a disagreement
becomes an identical `Failure` on every rank rather than a race to the next collective.

**Test to add.** Same shape as R-01's own recommended test: an MPI test (2+ ranks) that
partitions a region so `MeshView::dimension()` returns a *different* value on two ranks
(one rank's local partition non-empty only at dimension 2, another's non-empty only at
dimension 1, both otherwise valid) and asserts `build_remap_halo`/`build_weights` return
the same `Failure` (or the same success) on every rank rather than diverging.

### N-BACKEND-2 -- MAJOR -- a fourth per-rank refusal in `plan_halo_exchange` was outside R-02's scope and remains unreconciled

**Where.** `src/backend/moab_backend.cpp:2013-2022`, inside
`MoabMeshBackend::Impl::plan_halo_exchange`, after the `MPI_Alltoallv` at line 2010.

**What is wrong.** After both request/answer collectives complete, the function walks
`ans_gids` (what peers asked this rank for) and refuses if any requested global id is
not in `owned_by_gid`: `if (it == owned_by_gid.end()) { return Failure{"E-MESH-HALO", ...}; }`.
This is the same *shape* of defect R-02 fixed -- a conclusion drawn from this rank's own
local map -- but it sits after both collective pairs in this function (`MPI_Allgather`/
`MPI_Allgatherv` and `MPI_Alltoall`/`MPI_Alltoallv`), so it does not, by itself, hang
*this* function: the collectives it would have raced into have already returned on
every rank by the time this check runs. The risk moves up a level instead: it means
`plan_halo_exchange` -- and therefore `adopt_remap_halo` and `build_remap_halo`, which
propagate its `Result` unchanged -- can still return a *different* verdict (success on
some ranks, `E-MESH-HALO` on others) for the identical call, for a reason none of R-02's
reconciliation touches. Nothing downstream of `build_remap_halo` in this layer
reconciles that divergence either (`adopt_remap_halo`/`has_remap_halo` are pure local
bookkeeping, confirmed by reading `moab_backend.cpp:2101-2189`). Whether that produces an
actual hang depends on what the caller does with a non-uniform result, which is outside
this layer's scope to fix -- but the backend-layer contract this file otherwise upholds
("every rank returns the IDENTICAL `Failure`") is broken at this one site, and a caller
that assumes `build_remap_halo`'s result is rank-uniform (a reasonable assumption, since
R-01/R-02 just established it everywhere else in this file) would be wrong here.

**Fix.** Reconcile this check the same way as the other three in this function: record
whether the local id was found, and call `reconcile_halo` once after the loop (there is
no further collective in this function to place it "before," so placing it immediately
after the loop, before `return`, is sufficient to make the function's own result
rank-uniform).

**Test to add.** An MPI test (2+ ranks) that manufactures a request/answer mismatch
local to one rank (e.g., patch one rank's `owned_by_gid` after `load()` so a genuine
peer request resolves to "not found" on exactly that rank) and assert every rank's
`plan_halo_exchange` result agrees.

### N-BACKEND-3 -- MINOR / DEVIATION -- `reconcile_halo` ignores `MPI_Comm_size`/`MPI_Comm_rank` failure; its sibling does not

**Where.** `src/backend/moab_backend.cpp:1825-1829` (`reconcile_halo`) versus
`src/backend/intx_weights.cpp:374-387` (`reconcile<T>`).

**What is wrong.** The fix pass's own commit message for `reconcile_halo` says it
"[m]irrors the reconciliation `intx_weights.cpp` performs... same finding, same fix,
kept as a separate copy" (comment at `moab_backend.cpp:1818-1823`). It does not, in one
respect: `reconcile<T>` checks `MPI_Comm_size`'s return value and returns a `Failure`
if it is not `MPI_SUCCESS` (`intx_weights.cpp:380-384`), explicitly choosing to refuse
rather than silently treat an unsizeable communicator as "no peers to disagree with."
`reconcile_halo` calls `MPI_Comm_size(comm, &nproc)` and `MPI_Comm_rank(comm, &rank)`
(lines 1828-1829) without checking either return value. If `MPI_Comm_size` were ever to
fail on one rank and succeed on its peers (a corrupted or mismatched communicator, the
same class of defect `E-TRANSPORT-PEERMATCH` exists elsewhere in this codebase to catch),
`nproc` keeps its initializer value of `1`, and `reconcile_halo` takes the `nproc<=1`
branch -- silently returning the LOCAL outcome unreconciled, on exactly the rank whose
communicator call failed, which is the one most likely to also have failed for the
reason this whole mechanism exists to catch. This is a silent fallback contrary to
invariant I4 ("No silent fallback... A capability that cannot be verified is unavailable
and says why"), present in the twin implementation that was explicitly written to avoid
it.

**Fix.** Check both return values in `reconcile_halo` the same way `reconcile<T>` does,
returning `Failure{"E-MESH-HALO", ...}` on either failing, rather than falling through to
the `nproc<=1` no-op path.

**Test to add.** Not independently testable without a way to force `MPI_Comm_size` to
fail (a real defect, not a mutation candidate in the usual sense); a code-review-only
finding until a fault-injection harness for MPI calls exists here.

## Not found

Checked and *not* found, beyond what the original review already ruled out (Section 4
there, unchanged in this pass -- re-confirmed by re-running `tools/check-layering.py`
and `tools/check-invariants.py`, both 0 violations):

- No regression in the six commits outside their stated scope: `git show --stat` on
  each of the six confirms only the files the fix pass's own commit-unit table names
  were touched (`intx_weights.cpp`; `moab_backend.cpp`; the three doc-only headers;
  `supermesh.cpp` + its new mutation TOML; `spacetime.cpp` + its test + its mutation
  TOML; `native/mesh.hpp`).
- No stale mutation anchor anywhere in the repository as of HEAD (`--check-anchors`:
  86 valid, 0 stale) -- the fixer's disclosed action item (rename `parent_(c)` to
  `h_parent(c)` in the `mesh-accepts-a-parent-outside-the-cell-set` anchor) has been
  carried out by whoever owns `build-evidence/mutations.toml`.
- No case where a doc-comment fix (R-11) still cites a test file that does not exist as
  if it were evidence -- checked every remaining `test/unit/` string in scope (9 left)
  and confirmed each sits inside a sentence that says the file does not exist, not one
  claiming it as support.
- `test/backend/native_mesh_test.cpp` has no `Kokkos::parallel_for`/`KOKKOS_LAMBDA` call
  anywhere (grepped) -- R-14's device-callability claim is, as both the review and the
  fixer already say, untested here by construction, not silently claimed otherwise.
- `test/backend/supermesh_test.cpp` and `test/backend/spacetime_test.cpp` are the only
  test files touched by the fix pass; `git diff main..HEAD` on `supermesh_test.cpp` is
  empty (confirms no test was added for R-12/R-13, matching the fixer's own disclosure)
  while `spacetime_test.cpp` gained exactly the one test the R-33 fix describes.
- `Mesh<M,S>`'s new third template parameter has exactly one call site pattern in the
  tree (`Mesh<M,S>` with two arguments, defaulted); no code anywhere names a third
  argument, so nothing exercises a non-default `MemorySpace`, consistent with the
  fixer's own "Needs a host run" note.

## Summary

Verification: **8 findings checked** -- 4 CONFIRMED-FIXED (R-02, R-03, R-11, R-33),
4 PARTIAL (R-01, R-12, R-13, R-14), 0 NOT-FIXED, 0 REJECTION-UPHELD (the fixer rejected
nothing in this layer). Every PARTIAL is a real, working fix with a specific, evidenced
gap: R-01 and R-14 have residual defects/limitations found this pass; R-12 and R-13 are
correct fixes with no permanent regression test, only a mutation entry.

Fresh review: **3 new findings** -- 1 BLOCKER (N-BACKEND-1, the same hang-hazard class
as B1/R-01, one call frame higher, in the same two functions), 1 MAJOR (N-BACKEND-2, a
fourth unreconciled per-rank refusal R-02's scope did not examine), 1 MINOR/DEVIATION
(N-BACKEND-3, an inconsistency between two reconciliation helpers written in the same
fix pass for the same purpose). All tier0-buildable code changed by the fix pass passes
its full test suite (71/71) and its full mutation gate (18/18 killed); the layering and
invariant gates are clean. The MOAB/MPI-gated code the fix pass touched (R-01, R-02)
could not be compiled or executed in this sandbox and needs a host tier-2/tier-1 run,
as `docs/dev/testing.md` anticipates for this exact class of change -- this review
verified what can be verified without one (code shape, placement relative to the
collectives named in each finding, non-regression of everything that does build here)
and no more.
