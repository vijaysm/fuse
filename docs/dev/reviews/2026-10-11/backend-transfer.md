# Review 2026-10-11 — `backend`: transfer / remap / supermesh / space-time

Read-only review at HEAD `4a8a9d8`. Scope read in full: `intx_weights.cpp` (2519 ln), `supermesh.cpp`,
`supermesh_detail.hpp`, `spacetime.cpp`, `transfer_map_io.cpp`, `weights_operator.cpp`,
`mls_weights.cpp`, `tempest_weights.cpp`, `nodal_remap_distributed.cpp`, `serial_backend.cpp`,
`linear_operator.cpp`, `high_order_backend.hpp`, plus headers `remap_plan.hpp`,
`consistent_interp.hpp`, `general_interp.hpp`, `spacetime.hpp`, `nodal_remap_distributed.hpp`, and
the relevant 2/3 of `moab_backend.cpp`; layer docs and KNOWN_ISSUES A/D sections for dedupe.
Findings feed the ranked index in `index.md`; IDs here are referenced as `TF-*`.

## 1. Findings

| ID | Severity | file:line | Finding | KNOWN |
|---|---|---|---|---|
| TF-1 | **BLOCKER** | `src/backend/nodal_remap_distributed.cpp:244-286` | `gather_global_source`'s per-rank buffer-building loops fail rank-locally *ahead of* their own `MPI_Allgatherv` — the exact R-01/N-BACKEND-1 class the round-4 D8 fix audited for, but the audit only covered *post*-collective returns | not recorded (D8 fixed the sibling sites) |
| TF-2 | MAJOR | `src/backend/intx_weights.cpp:1226-1229, 2451-2500` | `build_weights`' first-order path makes rank-local geometry decisions and calls `orient()`/`fit_surface()` unreconciled, ahead of `assemble()`'s collectives | class KNOWN (R-01); these sites not recorded |
| TF-3 | MAJOR | `src/backend/moab_backend.cpp:2857-2902` | `adopt_remap_halo`'s per-rank rejections (gid tag, per-cell GLOBAL_ID, `add_entities`, `regather_with_halo`) are unreconciled ahead of `plan_halo_exchange`'s collectives — one call frame above the R-02/N-BACKEND-2 fixes | class KNOWN (R-02/N-BACKEND-2); these sites not recorded |
| TF-4 | MAJOR | `src/backend/supermesh.cpp:104-111` (vs `202-227`) | The 1D supermesh kernel clips intervals on **x only**, with no guard analogous to the 2D path's `E-REMAP-NONPLANAR` — a 1D mesh along y/z silently yields an empty (or shadow-measure) matrix returned as **success** | not recorded |
| TF-5 | MAJOR | `test/**` (absence) | No test anywhere fires `E-REMAP-COVERAGE` (the `RemapFlags::Conservative` truncation refusal) — I4's own enforcement clause says a refusal no test fires "does not count as implemented"; `coverage_advice()`'s partitioned advice is likewise untested | not recorded |
| TF-6 | MINOR/DEVIATION | `include/fuse/backend/nodal_remap_distributed.hpp:65-73` | Header states "NOT YET WIRED INTO Runner::Impl::resolve_remap … `resolve_remap` still rejects" — false since 2026-09-25 (`runner.cpp:749,1132,1155`; docs/dev/nodal-fields.md round 2) | contradicts recorded D8 state |
| TF-7 | MINOR | `include/fuse/backend/remap_plan.hpp:470-477`, `src/backend/intx_weights.cpp:1909-1912, 2283-2286`, `include/fuse/backend/moab_backend.hpp:203` | "implemented, untested" comments are now half-stale: `test/parallel/remap_halo_mpi_test.cpp` (D4, host-verified 2026-09-28) fires `E-REMAP-PARALLEL`, `E-MESH-HALO-GEOMETRY` and exact forward/reverse halo identities; only the end-to-end conservation *figures* remain diagnostic-only | partially KNOWN (R-11/D4) |
| TF-8 | MINOR | `src/backend/intx_weights.cpp:1417-1419`, `src/backend/moab_backend.cpp:2594-2596` | `reduce_audit` and `plan_halo_exchange` ignore `MPI_Comm_size`/`MPI_Comm_rank` return values — the exact N-BACKEND-3 defect class, fixed in `reconcile_halo` (moab_backend.cpp:2310-2316) and `reconcile<T>` (intx_weights.cpp:483-490) but left at these two older sites; on failure `nproc` stays 1 and the local outcome is returned unreconciled | class KNOWN (N-BACKEND-3); sites not recorded |
| TF-9 | MINOR/DEVIATION | `src/backend/mls_weights.cpp:538-549` | MLS emits each row's columns in k-nearest-neighbour order, violating `RemapWeights`' documented "col: source entity, ascending within a row" (`remap_plan.hpp:106`); harmless to every current consumer but breaks the stated COO contract and makes row-sum rounding k-order-dependent | not recorded |
| TF-10 | MINOR | `include/fuse/backend/general_interp.hpp:848-851` | `cell_vertices`' Result is not checked before `.value()` on the winning-cell re-query — a failure there throws `std::bad_variant_access` out of a function whose contract is "Rejects, never throws" (latent: the up-front loop at 649-687 validated the same call) | not recorded |
| TF-11 | MINOR | `src/backend/nodal_remap_distributed.cpp:188-190, 247, 277-284, 531, 604-605` | Global ids travel as `double` (53-bit mantissa): any gid > 2⁵³ silently loses precision in the Allgatherv records and in `agree_on_local_failure`'s 4-double record — wrong vertex matching with no diagnostic. Unreachable through MOAB (int32 GLOBAL_ID) but the API takes any `MeshView` | not recorded |
| TF-12 | MINOR | `src/backend/nodal_remap_distributed.cpp:537-542` | `apply_distributed_H` silently keeps the *last* record for a duplicated owned gid, while `gather_global_source` (296-310) rejects the same contract violation by name (`E-RT-NODAL-DIST-OWNER`) — inconsistent strictness for the identical hazard | not recorded |
| TF-13 | MINOR | `src/backend/nodal_remap_distributed.cpp:461-467` | Tier-(b) fallback uses strict `score <` with first-wins tie order = Allgatherv concatenation order, which depends on rank count — contradicting the header's bit-identical-across-rank-counts claim (nodal_remap_distributed.hpp:33-39) in the tied-score, no-tight-match case (the serial twin at general_interp.hpp:818 has deterministic box-index order) | not recorded |
| TF-14 | MINOR | `src/backend/intx_weights.cpp:1285-1296` | `intersect_surfaces`' `decode` accepts raw ∈ [0,n) as a 0-based position; a future/convention-different MOAB writing 1-based positions would be silently accepted off-by-one for every index except the last | not recorded |
| TF-15 | MINOR | `src/backend/supermesh.cpp:289` | `min_w = acc.empty() ? 0.0 : 1.0`: if every accumulated entry falls below `kNegligible` and is dropped, an **empty** matrix is returned reporting `min_weight == 1.0` (`RemapWeights` documents "Zero for an empty matrix"); `monotone` is still correctly false via the row defect | not recorded |
| TF-16 | MINOR | `include/fuse/backend/consistent_interp.hpp:346-348` vs `src/backend/weights_operator.cpp:36` | `H` stores the *unweighted* column sum in `max_col_defect`; on a coincident node-to-node pair (identity, col sums exactly 1) `WeightsOperator::claims()` certifies `conservative` — the very claim the header says H "never makes" (consistent_interp.hpp:48-53, 338-345) | not recorded |
| TF-17 | MINOR/IMPROVEMENT | `src/backend/tempest_weights.cpp:435, 494, 506` | Failure and order-degradation detection keys on the literal substrings `"EXCEPTION"` / `"dropping to 1st order"` from TempestRemap's print stream — an upstream wording change silently reverts E-REMAP-DEGRADED to accepting a mislabelled first-order map. Disclosed as the only failure channel; a version-guarded assertion of the exact string would convert drift into a loud error | not recorded |

### Evidence and why it matters — the top findings

**TF-1 (BLOCKER).** `gather_global_source` first Allgathervs vertices (`nodal_remap_distributed.cpp:252`),
then builds the cell buffer in a loop that can fail **per rank** — `cell_vertices` error return (262-264),
unsupported shape (265-270), wrong corner count (271-276) — and only then Allgathervs cells (286). A rank
owning a pyramid/polygon source cell (legal mesh content the *serial* builder rejects cleanly with
`E-INTERP-SHAPE`, general_interp.hpp:657-673) returns or throws while its peers enter the cells
`MPI_Allgatherv` and hang. The vertex loop (244-251) is worse: `src.global_id(0, i)` **throws**
(`moab_backend.cpp:450-455`) whenever dim-0 `gid_error` is set — which `gather()` sets exactly on
partitioned reads with unnumbered vertices (moab_backend.cpp:1577-1599) — so the documented "partitioned
MOAB read routinely manufactures … entities no one numbered" case throws out of a "Rejects, never throws …
identically on every rank" function (nodal_remap_distributed.hpp:129-137) *before the first collective*.
D8 round 4's audit explicitly covered "every other **post**-collective early return"; these **pre**-collective
ones were missed. Fix: record-and-stop inside both loops and run `agree_on_local_failure` (or at minimum
an all-to-all flag) *before* each `allgatherv_records`; convert the throwing `global_id`/`centroid` calls
to recorded failures. Test: 3-rank run with an unsupported-shape cell (or unnumbered vertices) owned by
one rank only, asserting every rank returns the same `Failure` rather than hanging.

**TF-2 (MAJOR).** R-01's fix wrapped `resolve()`, and `build_remap_halo`'s own `fit_surface`/`orient`
call sites (intx_weights.cpp:2001-2036, 2048-2051) in `reconcile<T>`. But `build_weights`' first-order
path re-does the same classification **unreconciled**: `try_boxes` (2451-2454, rank-local classification
including halo-extended source cells), `intersect_boxes`' geometry rejections (903-921), `fit_surface`
(2474-2477), the kind/radius/plane-z match (2478-2497), and — one frame down — `orient(src/tgt)` **inside
`intersect_surfaces`** (1226-1229), which the R-01 fix never wrapped. All of these precede `assemble()`'s
collectives (`accumulate_remap_halo` at 1551-1554, `reduce_audit`'s `MPI_Allreduce` at 1596-1599), so any
rank-divergent verdict (a per-cell MOAB connectivity error on one rank, a halo fringe that changes
box/surface classification) is a hang, not an error. Fix: wrap the classification payloads in `reconcile`
the way `build_remap_halo` already does, and reconcile `orient` inside `intersect_surfaces` (or at its
`build_weights` call site).

**TF-3 (MAJOR).** `build_remap_halo` is collective; every rank then calls `adopt_remap_halo`, whose own
rejections — `E-MESH-HALO-GID` (moab_backend.cpp:2857-2861), missing GLOBAL_ID tag (2864-2870), per-covering-cell
gid check (2878-2882), `add_entities` (2889-2895), `regather_with_halo` (2899) — depend on this rank's
*covering-set contents*, which are rank-local by construction, and are returned unreconciled *before*
`plan_halo_exchange`'s `reconcile_halo` Allreduce (2607) and Allgather/Alltoall (2632, 2640). N-BACKEND-2
fixed the fourth refusal *inside* `plan_halo_exchange`; the adopt-level frame above it has the same shape.
Fix: route adopt's local failures through `reconcile_halo` before `plan_halo_exchange` runs.

**TF-4 (MAJOR).** `intersect` for dim 1 (supermesh.cpp:104-111) reads only `.x`; `boxes_overlap` tests only
`k < 1` (74-79). There is no spread check on y/z — the exact guard the 2D kernel has (`E-REMAP-NONPLANAR`,
202-227) — and no automatic axis choice, unlike the sibling `build_consistent_interp_weights_1d`
(consistent_interp.hpp:190-205, "a beam laid out along y or z is handled"). Consequences for a y-aligned
1D pair: every box test passes (all x equal), every interval overlap is 0 → an **empty matrix returned as
success**, coverage 0, `apply()` leaves the target field at zero — indistinguishable from a transferred
zero (the transfer-maps doc's own definition of the unacceptable silence). A diagonal 1D curve instead
over-measures its x-shadow. Every 1D test uses `generated:line` (x-aligned) only, so nothing catches it.
Fix: replicate the dim-2 guard (reject, or pick the max-spread axis) and add a y-axis 1D test.

**TF-5 (MAJOR, I4 binding rule).** `E-REMAP-COVERAGE` (intx_weights.cpp:1624-1636, supermesh counterpart
via `min_src_coverage`, remap_plan.hpp:96-97) — the refusal that guards conservative remaps against
partial-overlap truncation, plus the entire `coverage_advice()` "DO NOT reach for the Conservative flag"
paragraph (1457-1471) — is fired by **no test in the tree** (grep over `test/`: only a comment in
comm_graph_test.cpp:924). Invariant I4: "a refusal that no test makes fire does not count as implemented."
A serial partial-overlap supermesh pair (trivially built in `supermesh_test.cpp`, which already constructs
partial-overlap fixtures at line 665) would close it. Note the *audit fields* are covered; it is the refusal
that is not.

## 2. Checked and found clean

- **R-12 (degenerate clipping)**: fixed and fully closed — `inward_plane` returns `false` on `len <= 0` (supermesh_detail.hpp:112), both callers clip to zero, and the whitebox regression tests the pass-2 review asked for now exist (`SupermeshDetail.InwardPlane*`).
- **R-13 (over-coverage audit on supermesh)**: landed — `audit_coverage` rejects `c > 1 + 1e-9` on both sides, tolerance identical to the MOAB path's `kCoverageTol`, and unit-tested at the boundary. Pass-2's "no permanent test" gap is closed.
- **R-33 (spacetime zero-weight skip)**: fixed — triplet validation runs per-triplet before the `w == 0` continue; `ApplySpacetimeValidatesTripletsEvenWhenTheTemporalWeightIsZero` fires it.
- **N-BACKEND-1 (`Side::dim` divergence)**: fixed — `reconcile_dim` agrees min/max inside `resolve()`.
- **N-BACKEND-2 / N-BACKEND-3**: both fixed — `send_error` reconciled; `reconcile_halo` now checks `MPI_Comm_size`/`rank`.
- **I4 refusals**: TempestRemap absent → `E-REMAP-NOTEMPEST`, not degradation; "dropping to 1st order" → `E-REMAP-DEGRADED`; MLS options never defaulted, rank-deficiency rejected with a measured estimate; `Monotone`+order>1 and `HighOrder`+order1 refused by name; spherical-excess area formula, box broad-phase window, `assemble`'s global audit reduction, GidGuard restore-on-unwind, `run_halo_exchange`'s accumulate-then-zero semantics — all correct as documented.
- **D8 round-4 fixes**: `agree_on_local_failure` wraps the per-entity loop and `apply_distributed_H`'s owner-missing check; `apply_distributed_HT`'s post-collective code is deterministic sort/sum. Both host-found defects verified fixed.
- **I5**: `apply_weights` allocates nothing and validates triplets; `WeightsOperator::apply/transpose` allocation-free; `compose` is init-time only.
- **Transfer-map IO**: magic/version/length/checksum each its own code, entry-count vs remaining-bytes guard, bit-pattern reals, locale-free writes, strict `E-MAP-DUPLICATE`, `localise` re-orders rows by ascending source gid exactly as transfer-maps.md promises; every refusal in the doc table has a firing test.
- **Conventions**: fmt-only, `FUSE_ASSERT/FUSE_THROW/FUSE_REFUSE` discipline, `#if FUSE_HAVE_*` (value-tested) everywhere in scope; no third-party type leaks past the backend dir; `spacetime.cpp`'s exact-rational validation and no-throw conversion as documented.

## 3. Not reviewed

- `include/fuse/backend/native/**` and `src/backend/native_*.cpp` (native-mesh side — see `backend-native-mesh.md`), including whether `NativeMeshView::dimension()` reproduces the N-BACKEND-1 hazard on the `build_weights_via_supermesh`/native-halo paths (intx_weights.cpp:169, 271 use unreconciled dims there too).
- `moab_backend.cpp`'s `resolve_region`/`find_named_set_in` beyond a correctness skim; `MeshLoader::read_file` option strings against MOAB's parallel-read semantics (trusted as host-verified per C1/D4).
- `test/backend/moab_mesh_test.cpp` (54 cases) and `test/backend/native_*` case-by-case; `mms_transfer_test.cpp` beyond its header/fixture.
- The TempestRemap-enabled build's actual behavior (no TempestRemap here): `tempest_weights.cpp`'s `FUSE_HAVE_TEMPEST` half is review-by-reading only, same posture as REVIEW.md §5.

## 4. Test-coverage assessment

**Strong.** R-12/R-13 whitebox tests landed since pass 2 (closing its PARTIAL verdicts); R-33; N-BACKEND-1's
boundary (both sides); `E-MESH-HALO-GEOMETRY`; exact forward/reverse halo round-trips by gid
(`remap_halo_mpi_test.cpp`, host-verified per D4); MLS's complete refusal matrix + ArborX cross-check;
transfer-map IO's per-corruption refusals; spacetime's per-error refusals; the distributed nodal builder's
null-comm suite. Rank-count bit-identity is asserted exactly where the docs claim it (transfer maps).

**Gaps.**
1. `E-REMAP-COVERAGE` fires nowhere (TF-5) — the one I4-listed refusal with zero coverage.
2. The MOAB path's `over_src`/`over_tgt` `E-REMAP-INTX` audit fires only via mutation, never a test — the supermesh path now shows the whitebox pattern (`audit_coverage`) that could be applied to the reduced audit too.
3. The end-to-end conservation numbers the headers cite (1.2e-15 at 2 ranks etc.) are printed, never gated — worth a gated case on a fixture whose overlap is analytically full.
4. The hang-class defects (TF-1, TF-2, TF-3) have no failure-agreement tests at all: nothing anywhere asserts that a *rank-local* mesh defect produces the same `Failure` on every rank. `TargetOutsideSourceDomainRefuses` is the closest, and TF-1 shows exactly the shape it misses.
5. 1D supermesh is tested only on x-aligned `generated:line` meshes (TF-4); one y-aligned case would close it.
