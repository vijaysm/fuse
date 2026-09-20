# FUSE backend third-pass (final) review — HEAD ced1945

Scope: `include/fuse/backend/**`, `src/backend/**`, `test/backend/**`, plus R-41/BT-F15.
Verifies the second fix pass (`docs/dev/reviews/2026-09-19/pass2/fix2-backend.md`) against
the code at HEAD, then re-reviews the same paths fresh, with particular attention to the
files that pass changed: `src/backend/intx_weights.cpp`, `src/backend/moab_backend.cpp`,
`src/backend/supermesh.cpp`/`supermesh_detail.hpp`, `test/backend/supermesh_test.cpp`,
`test/CMakeLists.txt`.

## Method

Built `tier0` (Debug, Eigen linalg, MOAB/MPI both off) into a private tree with
`cmake --preset tier0 -S <repo> -B $PWD/b0` and `cmake --build $PWD/b0 -j4`: zero errors,
`ctest -j4` → **100% tests passed, 0 tests failed out of 71** (`backend` label 6/6, and the
`fuse_backend_supermesh_test` binary run directly shows all 23 individual GoogleTest cases —
17 `Supermesh.*` + 6 `SupermeshDetail.*`, including all seven whitebox cases the second fix
pass added — passing). Ran `tools/mutate.py --area backend`: **18/18 KILLED**.
`tools/mutate.py --check-anchors` (whole repo): **95 valid, 0 stale** — the anchor the second
fixer flagged as needing another owner (`supermesh-planes-point-outward`'s `file` field) has
been corrected. `tools/check-layering.py` and `tools/check-invariants.py`: **0 violations**
each.

Went further than a tier0-only check because the second fix pass's central claim — that
tier1/tier2 now CONFIGURE and BUILD cleanly on this machine, reversing an earlier recorded
limitation — is exactly the kind of dated observation `docs/dev/invariants.md` (`I10`) says
must be re-verified rather than carried forward: configured `tier2` fresh
(`cmake --preset tier2 -S <repo> -B $PWD/b2`) and built `fuse-coupler-backend`,
`fuse_backend_supermesh_test` and `fuse_backend_moab_mesh_test` against it. Confirmed
independently: **tier2 configures** (MOAB and MPI both found; the log reads `fuse: real-mesh
MOAB tests will use /opt/petsc/download-moab/share/examples/fortran/spherecube.h5m`) and
**builds with zero errors** against the real MOAB 5.x + MPICH 5.0 headers — this is new,
positive signal beyond what the second-pass review (`re-backend.md`) could establish, since
MOAB did not link on this machine when that review was written. Also confirmed, independently,
that **nothing linked against this machine's MPI can execute**: running
`fuse_backend_moab_mesh_test --gtest_filter="MoabMeshFile.*"` segfaults inside libmpi's
debugger hooks (`MPIR_Assert_fail`/`pthread_mutex_destroy: Invalid argument`,
`dbginit.c:189-190`), the identical crash signature the second fixer's report describes from
a standalone `MPI_Init` hello-world — so the fixer's disclosed environment correction (builds
now, still cannot execute) is accurate and is re-confirmed here independently, not merely
carried forward from their report.

Read every commit in `git log main..HEAD` touching `src/backend/`, `include/fuse/backend/`,
`test/backend/`, or `test/CMakeLists.txt`'s backend section since the second-pass review
(`35d4a0e`, `4547e88`, `8732afe`, `8b14f43`, `08848ec`) against the code at HEAD, not against
the fixer's prose, plus the base-layer commit (`4631a8c`, `include/fuse/base/error.hpp`) that
closed R-14's residual gap, since the second fix pass explicitly deferred that one and it
bears directly on a finding in this layer's scope.

## Verification

| id | verdict | evidence |
|---|---|---|
| R-01 / N-BACKEND-1 (pass2) | **CONFIRMED-FIXED** | `resolve()` now computes `s.dim` and immediately calls a new `reconcile_dim(s.pcomm, s.dim, s.label)` (`src/backend/intx_weights.cpp:576-584`) — an `MPI_Allreduce(MIN)`/`MPI_Allreduce(MAX)` pair that refuses `E-REMAP-DIM` identically on every rank when they disagree — **before** any of the partition-dependent local checks that follow (`s.dim < 1`, `range.empty()`, …) or the `a.dim != b.dim`/`a.dim != 2` comparisons in `build_remap_halo` (`intx_weights.cpp:1849,1879`) and `build_weights` (`:2139`) that the original finding named. Read the full function; the reconciled value is what every downstream comparison now sees, so a rank whose local partition reports a different `MeshView::dimension()` than its peer's (`moab_backend.hpp`'s own documented limitation) is refused identically on every rank rather than racing one comparison ahead of where R-01's original fix stopped. Compiles with zero errors against real MOAB+MPI headers at tier2 in this sandbox (new this pass — MOAB does not link the same way it did when `re-backend.md` was written). No MPI test is registered for this code (see New findings, N-BACKEND-1 below) — needs a host run to execute, as `docs/dev/testing.md` anticipates for MPI-only code, but the code path exists and reads correctly. |
| N-BACKEND-2 (pass2) | **CONFIRMED-FIXED** | The fourth per-rank refusal identified in `plan_halo_exchange`'s `send_index` loop (after `MPI_Alltoallv`, `moab_backend.cpp:2033-2046`) no longer returns from inside the loop; it records `send_error` and reconciles once via `reconcile_halo(comm, !send_error.empty(), send_error)` (`:2048-2051`) before returning, matching the shape of the two `want`-loop fixes R-02 already made in this function. Compiles with zero errors at tier2. No MPI test registered (same gap as above). |
| N-BACKEND-3 (pass2) | **PARTIAL** | The literal fix is applied correctly: `reconcile_halo` now checks both `MPI_Comm_size`/`MPI_Comm_rank` return values and refuses `E-MESH-HALO` rather than silently taking the `nproc<=1` path (`moab_backend.cpp:1835-1840`), matching its sibling `reconcile<T>` (`intx_weights.cpp:402-406`). **What remains, found this pass**: the fix does not close the gap it claims to under the codebase's own MPI usage — see new finding N-BACKEND-1 below. In short, nothing in `MoabMeshBackend::set_communicator` (`moab_backend.cpp:2279-2313`) ever changes the communicator's error handler away from the default `MPI_ERRORS_ARE_FATAL` (confirmed: no `MPI_Comm_set_errhandler`/`MPI_ERRORS_RETURN` anywhere in `src/backend/*.cpp` or `include/fuse/backend/*.hpp`, unlike `src/transport/mpi_transport.cpp:263`, which explicitly duplicates its communicator and sets `MPI_ERRORS_RETURN` for exactly this reason). Under the default handler, an `MPI_Comm_size`/`MPI_Comm_rank` failure aborts the process inside the MPI call itself; the `!= MPI_SUCCESS` branch this fix added is not reached by any path that returns control to it. The fix is real, matches its sibling, and is harmless — but it is unreachable under this library's actual communicator-handling practice, so it does not yet deliver the I4 compliance its own commit message claims. |
| R-12 (BK-M2) missing-test gap | **CONFIRMED-FIXED** | `inward_plane<Dim>` moved to the new private header `src/backend/supermesh_detail.hpp`, reachable from `test/backend/supermesh_test.cpp` via a `PRIVATE` include path on that one test target (mirroring `comm_graph_test`'s existing access to `CrystalRouter`). Three new `SupermeshDetail.*` tests construct a degenerate 2D edge and 3D face directly and assert the refusal (`InwardPlaneRefusesAZeroLengthEdgeIn2D`, `...ZeroAreaFaceIn3D`), plus one asserting correct orientation on a non-degenerate face. Ran all three directly: pass. This closes the gap the first fix pass (and pass2) left open (a mutation entry only, no permanent test). |
| R-13 (BK-M3) missing-test gap | **CONFIRMED-FIXED** | The coverage audit's refusal was extracted to `supermesh_detail::audit_coverage(covered, measure, which, idx, dim)` (`supermesh_detail.hpp:26-42`), called from both of `supermesh.cpp`'s audit loops (`:347,356`). Three new `SupermeshDetail.Audit*` tests exercise the boundary directly (accepted at/under tolerance including a case 5e-10 inside it; refused at 1.5× with `E-REMAP-INTX` and the cell index/ratio in the message; a zero-measure cell skipped rather than dividing by zero). Ran all three directly: pass. Independently re-ran the mutation `supermesh-no-over-coverage-audit`: KILLED. |
| R-14 (BK-M4) residual gap | **CONFIRMED-FIXED** | The second fix pass deferred this correctly to the base layer (`FUSE_DEBUG_ASSERT`'s `debug_abort` call had no `KOKKOS_FUNCTION` annotation, so a real GPU-default Debug build would fail to compile). Read `include/fuse/base/error.hpp:214-227` at HEAD: the macro's non-`NDEBUG` branch now uses `KOKKOS_IF_ON_HOST(...debug_abort(...)...)` / `KOKKOS_IF_ON_DEVICE(...Kokkos::abort(...)...)` — the standard Kokkos host/device split for exactly this situation — rather than calling `debug_abort` unconditionally. This closes the gap without needing `debug_abort` itself to be device-callable. Confirmed compiling cleanly at tier0 **Debug** (the non-`NDEBUG` branch is the one exercised there) with `backend.native_mesh` passing. Not this layer's own fix (`include/fuse/base/`, made by whoever owns that layer, dated 2026-09-20 in the code comment), but it resolves a finding that was open in this layer's scope. |
| R-41 / BT-F15 | **CONFIRMED-FIXED** | `test/CMakeLists.txt:190-201` now does a `find_file(FUSE_MOAB_TEST_MESH_FILE NAMES spherecube.h5m PATHS "${_fuse_moab_prefix}/share/examples/fortran" NO_DEFAULT_PATH)` and defines `FUSE_MOAB_TEST_MESH` only when found, with a `message(STATUS ...)` either way — never a test that reads as passing when it did not run. Reproduced the positive case independently: a fresh `tier2` configure in this sandbox prints `fuse: real-mesh MOAB tests will use /opt/petsc/download-moab/share/examples/fortran/spherecube.h5m`, and `fuse_backend_moab_mesh_test` links successfully with that definition. The four `MoabMeshFile.*` cases (`test/backend/moab_mesh_test.cpp:614` onward) are gated by `#ifdef FUSE_MOAB_TEST_MESH` and compile in. Cannot *execute* them here — no MPI-linked binary can run in this sandbox (see Method) — that is a host-run item, not a gap in the fix. |

## New findings

| id | severity | file:line | title |
|---|---|---|---|
| N-BACKEND-1 | MAJOR | `src/backend/moab_backend.cpp:1897-1898,2279-2313`, `src/backend/intx_weights.cpp:1328-1334` | The rank-agreement mechanism's own MPI-failure handling — three fix rounds deep, including this round's N-BACKEND-3 — is unreachable under this library's actual communicator practice, and is applied inconsistently even on its own terms |
| N-BACKEND-2 | MINOR / DEVIATION | `src/backend/intx_weights.cpp:200-205` | A comment in `build_remap_halo`'s no-MOAB stub claims `build_weights` refuses without MOAB with `E-REMAP-NOMOAB`; it has not since 2026-09-17, and this round's own N-BACKEND-4 fix edited the adjacent 60 lines without correcting it |

## New findings in full

### N-BACKEND-1 — MAJOR — the reconciliation mechanism's MPI-failure branches are unreachable under this library's own error-handler practice, and inconsistently applied even ignoring that

**Where.** `src/backend/intx_weights.cpp:396-407` (`reconcile<T>`), `:468-479` (`reconcile_dim`,
this round's own fix), `:1328-1334` (`reduce_audit`); `src/backend/moab_backend.cpp:1828-1840`
(`reconcile_halo`, this round's N-BACKEND-3 fix), `:1897-1898` (`plan_halo_exchange`'s own
entry); `MoabMeshBackend::set_communicator`, `moab_backend.cpp:2279-2313`.

**What is wrong.** Every reconciliation helper in this file family calls `MPI_Comm_size`
and/or `MPI_Comm_rank` on `pcomm->comm()`, and several of them (`reconcile<T>`,
`reconcile_dim`, and — after this round's N-BACKEND-3 fix — `reconcile_halo`) check the
return value and refuse with a `Failure` rather than silently proceeding as if there were no
peers, explicitly citing invariant I4 ("no silent fallback"). That refusal branch cannot be
reached in this codebase's actual deployment. `MoabMeshBackend::set_communicator`
(`moab_backend.cpp:2279-2313`) takes the communicator directly from
`MPI_Comm_f2c(fortran_comm_handle)` and hands it straight to `moab::ParallelComm`'s
constructor — it never calls `MPI_Comm_set_errhandler` on it, and neither does anything else
in `src/backend/` or `include/fuse/backend/` (grepped: zero occurrences of "errhandler" in
either tree). MOAB's own `ParallelComm` does not touch the error handler either (grepped
`/opt/petsc/download-moab/include/moab/ParallelComm.hpp`: no match). So `pcomm->comm()` keeps
whatever error handler its owning application set on it — which for `MPI_COMM_WORLD`, or any
communicator descended from it without an explicit override, is the MPI default,
`MPI_ERRORS_ARE_FATAL`. Under that handler, an `MPI_Comm_size`/`MPI_Comm_rank` (or, for that
matter, `MPI_Allreduce`/`MPI_Bcast`) failure does not return a non-`MPI_SUCCESS` code to be
checked — it calls `MPI_Abort` (or the moral equivalent) from *inside* the failing MPI call,
before control ever returns to the calling C++ code. The `if (... != MPI_SUCCESS) return
Failure{...}` branches this round's N-BACKEND-3 fix added (and the ones `reconcile<T>` and
`reconcile_dim` already had) are therefore dead in the sense that matters: reachable by the
compiler, unreachable by any execution that does not already involve a process abort the
`Failure` path was written to avoid. Contrast with `src/transport/mpi_transport.cpp:263`,
which explicitly `MPI_Comm_dup`s its own communicator and calls
`MPI_Comm_set_errhandler(dup, MPI_ERRORS_RETURN)` — documented at
`include/fuse/transport/transport.hpp:115` as being done "so a failure is a Result, not an
abort." Nothing analogous happens for the backend layer's `pcomm`.

Independent of that root cause, N-BACKEND-3's own stated goal — making `reconcile_halo`
consistent with the sibling it was said to mirror — is not fully realized even ignoring
reachability. `reduce_audit` (`intx_weights.cpp:1328-1334`), the *original* reconciliation
function that `reconcile<T>`'s own doc comment says it copies the shape of ("this is the same
shape as `reduce_audit` below"), calls `MPI_Comm_size(comm, &nproc)` **without** checking its
return value at all — the same omission N-BACKEND-3 just fixed in `reconcile_halo`, present
in the function all three of these helpers cite as their model. And `plan_halo_exchange`'s own
entry point (`moab_backend.cpp:1897-1898`) calls `MPI_Comm_size`/`MPI_Comm_rank` unchecked
before ever calling `reconcile_halo` — if a hypothetical future change did give this
communicator `MPI_ERRORS_RETURN` and one of these two calls then failed, `nproc`/`me` would
keep their initialiser values (1, 0) and every buffer sized from `np` downstream
(`counts(np, 0)`, the `MPI_Allgather`/`MPI_Allgatherv` pair at `:1936,:1944`) would be sized
for one rank while the real communicator has more — a buffer-size mismatch, not merely a
silent single-rank fallback, and a strictly worse failure mode than the one N-BACKEND-3 just
closed one call away.

None of this reintroduces the hang class R-01/R-02/N-BACKEND-1/N-BACKEND-2 exist to prevent —
under the actual `MPI_ERRORS_ARE_FATAL` handler, a genuinely corrupted communicator produces a
loud abort, not a silent hang, so the operational risk is lower than a BLOCKER. But it means
three review rounds' worth of "reconcile the per-rank outcome, checked, before the next
collective" work rests on an error-handling assumption (`MPI_Comm_size` can fail gracefully on
this communicator) that nothing in the codebase makes true, and the one fix in this round
built specifically to satisfy I4 does not, on inspection, satisfy it under real conditions.

**Fix.** Either (a) have `set_communicator` duplicate the incoming communicator and set
`MPI_ERRORS_RETURN` on the duplicate, mirroring `mpi_transport.cpp`'s already-documented
pattern exactly, so `reconcile<T>`/`reconcile_dim`/`reconcile_halo`'s `Failure`-returning
branches become genuinely reachable; or (b) if duplicating this particular communicator is
undesirable for another reason, say in each function's comment that these branches are
defensive against a future change in error-handling policy rather than reachable today, and
drop the I4 citation until (a) is done. Either way, apply the same checked-`MPI_Comm_size`
pattern to `reduce_audit` and to `plan_halo_exchange`'s own entry-point calls, so the family is
internally consistent regardless of which of (a)/(b) is chosen.

**Test to add.** Not mechanically testable without a way to force `MPI_Comm_size` to fail on a
live communicator (the same limitation the fixer noted for N-BACKEND-3 itself) — a
code-review-only finding until a fault-injection harness for MPI calls exists here, same as
N-BACKEND-3's own "Test to add" concluded. If (a) is done, add a test asserting
`MPI_Comm_size`/`MPI_Comm_rank`'s error handler on the communicator `set_communicator` installs
is `MPI_ERRORS_RETURN`, not the default — a property a `MPI_Comm_get_errhandler` call can check
directly without needing to trigger a real failure.

### N-BACKEND-2 — MINOR / DEVIATION — a stale comment in `build_remap_halo`'s no-MOAB stub describes an error contract `build_weights` no longer has

**Where.** `src/backend/intx_weights.cpp:200-205`, inside the `#if !FUSE_HAVE_MOAB` stub for
`build_remap_halo`.

**What is wrong.** The comment reads: "`build_weights` refuses this build anyway, with
`E-REMAP-NOMOAB`." That was true when this file was ported from
`upstream/soma/src/remap/intx_weights.cpp` (confirmed: line 82 there does
`return Error{"E-REMAP-NOMOAB", ...}`), but it has not been true in FUSE since commit
`d5235f7` ("Route the default build's remap through the supermesh", 2026-09-17) — well before
any of the three review rounds. The current no-MOAB `build_weights`
(`intx_weights.cpp:182-197`, a few lines above the stale comment) does not refuse at all: it
checks only the discretisation kind and then unconditionally delegates to
`build_weights_via_supermesh`, which computes real conservative FV weights via the r3d/r2d
supermesh path. `grep -rn "E-REMAP-NOMOAB"` across `src/`, `include/` and `test/` finds no
site in FUSE that still produces this code (the one live reference,
`test/runtime/runner_test.cpp:251`'s `kNoWeightsHere` constant, is itself dead: `git log -p`
shows commit `d5235f7` deleted every `EXPECT_EQ(r.error().code, kNoWeightsHere)` call that used
it, but left the constant and its explanatory comment block declared and unused — the same
"doc describes a contract the code doesn't have" defect class R-11/BK-M1 already fixed
elsewhere in this exact file, recurring in a spot none of the three rounds' scope happened to
touch, made more conspicuous this round because N-BACKEND-4's own new function
(`build_weights_via_supermesh`, `intx_weights.cpp:136-165`) and its correct, accurate comment
sit fewer than 40 lines above the stale one). `test/runtime/runner_test.cpp` is outside this
layer's owned paths (`test/runtime/`, not `test/backend/`), so it is noted here for whoever
owns that file rather than filed as a backend finding.

**Fix.** Replace the stale sentence with something like: "Without MOAB, `build_weights` computes
real weights via `build_weights_via_supermesh` above rather than refusing; there is no
`E-REMAP-NOMOAB` case left in this build. The one thing `build_remap_halo` still has no
dependency-free equivalent for is a partitioned/haloed read, which is why this stub can be a
no-op rather than needing to redirect anywhere." Separately, flag
`test/runtime/runner_test.cpp:250-254`'s unused `kNoWeightsHere` constant and its comment for
that file's owner.

**Test to add.** None needed — this is a comment-only defect with no behavioural fix. A
`clang-tidy` unused-symbol-adjacent check would not catch a constant that has one remaining
non-executing use in a comment, so this is a review-only finding, same class as R-11's own
citation audit.

## Not found

Checked and *not* found, beyond what the first and second reviews already ruled out (both
sections re-confirmed clean this pass via `tools/check-layering.py`/`tools/check-invariants.py`,
0 violations each):

- No regression in the five backend-touching commits since the second-pass review
  (`35d4a0e`, `4547e88`, `8732afe`, `8b14f43`, `08848ec`): `git show --stat` on each confirms
  only the files the second fix pass's own commit-unit table names were touched
  (`intx_weights.cpp`; `moab_backend.cpp`; `supermesh.cpp` + new `supermesh_detail.hpp`;
  `supermesh_test.cpp`; `test/CMakeLists.txt`).
- No stale mutation anchor anywhere in the repository at HEAD (`--check-anchors`: 95 valid, 0
  stale) — the one the second fixer flagged as needing another owner
  (`supermesh-planes-point-outward`'s `file` field) has been corrected to
  `src/backend/supermesh_detail.hpp`.
- No case where the R-11 doc-comment fix (from the first fix pass) has regressed: every
  remaining `test/unit/...` string in scope still sits inside a sentence stating the cited
  file does not exist, not one claiming it as evidence — re-checked all 9 occurrences at HEAD.
- `test/backend/native_mesh_test.cpp` still has no `Kokkos::parallel_for`/`KOKKOS_LAMBDA` call
  (grepped) — the device-callability claims for `measure()`/`centroid()`/`normal()` (R-14)
  remain compile-verified only, not executed on an actual device build, which both prior
  reviews already disclosed and this one does not newly discover.
- `Mesh<M,S>`'s `MemorySpace` template parameter still has exactly one call-site pattern in the
  tree (two arguments, defaulted) — no code names a non-default third argument, so the
  device-mirror path in `finalize()` remains unexercised here, consistent with "needs a host
  run with a CUDA/HIP/SYCL Kokkos build," already disclosed twice.
- Re-ran the full `fuse_backend_supermesh_test` binary directly (not just via `ctest`, which
  reports one pass/fail per binary): all 23 individual GoogleTest cases pass, including the
  seven added this round.
- No MOAB-gated backend MPI test exists anywhere in `test/backend/` or `test/CMakeLists.txt`
  (grepped for `pcomm`, `ParallelComm`, `mpiexec`, `FUSE_REQUIRE_RANKS` under `test/backend/`:
  zero hits) — this is the direct, mechanical confirmation for the "no test is registered"
  half of N-BACKEND-1's and N-BACKEND-2's (pass2) verification rows above: there is nothing to
  check for a `TIMEOUT` property on, because nothing was registered. `fuse_add_mpi_test`
  (`test/CMakeLists.txt:83-98`) sets `TIMEOUT 120` on every MPI test it registers elsewhere in
  the tree (e.g. `model.descriptor_mpi`, the `parallel.*` runtime tests) — the mechanism exists
  and is used correctly everywhere it is used; the backend layer has simply never used it, in
  any of three rounds, for the rank-divergence hazards R-01/R-02/N-BACKEND-1/N-BACKEND-2 were
  all about.
- `Supermesh.BuildWeightsReachesTheSupermeshPathForANonMoabBackend` (N-BACKEND-4's own new
  test, this round) is not vacuous: it asserts `build_weights`'s dispatch output is *exactly*
  equal (row, col, val, n_rows, n_cols) to calling `build_supermesh_weights` directly as an
  oracle, not merely that no error occurred.

## Summary

Verification: **7 rows checked** — 6 CONFIRMED-FIXED (R-01/N-BACKEND-1(pass2),
N-BACKEND-2(pass2), R-12, R-13, R-14, R-41/BT-F15), 1 PARTIAL (N-BACKEND-3(pass2): the
literal fix matches its sibling exactly, but is unreachable under this codebase's own
communicator error-handling practice — see new finding N-BACKEND-1). 0 NOT-FIXED, 0
REJECTION-UPHELD.

Fresh review: **2 new findings** — 1 MAJOR (N-BACKEND-1: the rank-agreement mechanism's own
MPI-failure branches, across three review rounds, are unreachable given how
`MoabMeshBackend::set_communicator` obtains its communicator, and are applied inconsistently
even ignoring that), 1 MINOR/DEVIATION (N-BACKEND-2: a stale error-contract comment in the
same file R-11 already audited once, in a spot this round's own fix edited around without
correcting).

**What remains in this layer at MAJOR or higher, with evidence:** none of the three original
BLOCKERs (B1/B2/B3) remain open — B1 and B2's hang-hazard chains are now fully reconciled
end-to-end by reading (`resolve()` → `reconcile_dim` → `a.dim`/`b.dim` comparisons for B1;
all four per-rank refusals in `plan_halo_exchange` for B2; the `MemorySpace` template
parameter for B3), and this is confirmed to compile with zero errors against real MOAB 5.x +
MPICH 5.0 headers at tier2 in this sandbox — new, positive signal this pass, since that
compile was not previously possible here. **What is not closed, and cannot be closed without
a host run, is execution**: after three review rounds, zero MPI tests exist anywhere in
`test/backend/` for the rank-divergence hazards that were this layer's entire BLOCKER history
(R-01, R-02, and this round's own N-BACKEND-1(pass2)/N-BACKEND-2(pass2) fixes) — every
verification of the actual reconciliation behaviour at 2+ ranks has been by reading the code
and confirming it compiles, never by running it, because nothing in this sandbox can execute
an MPI-linked binary (confirmed independently this pass, down to a standalone hello-world). A
host run that exercises `build_remap_halo`/`build_weights`/`plan_halo_exchange` at 2+ ranks
with a deliberately unbalanced partition, asserting a matched `Failure` on every rank rather
than a hang, is the one verification step this layer has needed since the very first review
and still does not have. The one newly-discovered MAJOR finding (N-BACKEND-1, this report) is
lower-risk than that gap — it affects what happens on a communicator failure, not on an
ordinary data-dependent per-rank refusal — but it is evidence that even the code paths written
specifically to satisfy invariant I4 have not yet been checked against how this codebase's own
communicators are actually configured, which is the same category of gap (a claim about
distributed behaviour that reading alone cannot fully verify) that has run through all three
rounds of this layer's review.
