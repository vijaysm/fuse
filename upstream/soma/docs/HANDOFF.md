# SOMA — handoff

**As of:** 2026-09-10, branch `soma-v1-implementation`, tip `438b2b1`, 139 commits.
**State:** All 8 planned phases complete. All 3 mandatory review gates closed.
Both demos built, sabotage-gated, and independently verified. **Tier 1
(R1-R7) and Tier 2 (R8-R16) both fully closed.** A full validation pass is
next, per the standing directive, before Tier 3.
**Tests:** 770/770 in Debug, Release, and ASan+UBSan (765 before Tier 2: R8/R15
added 2, R11-R16 added 3) — re-measured 2026-09-11 and unchanged by the
`CommGraph`/Runner-wiring work, whose sources are behind `SOMA_HAVE_MPI`.
**MPI alone: 781/781** (770 plus the 4 pre-existing MPI-only entries —
`mpi_transport` at 2 and 4 ranks, `capi_parallel_test` at 1 and 2 — plus
`comm_graph_{2,3,4}rank`, plus the `runner_transport` suite added while
wiring `CommGraph` into `Runner` at 2/3/4/5 ranks; the "769" recorded here
previously was inconsistent with the 770 above and is superseded by a
measurement rather than reconciled).
MOAB+MPI: 872/872 (867 R1-R7 + 5 R9/R10 remap-hardening tests; R8/R11-R16
don't touch MOAB-gated code either). 87/87
Python (Task 8.3, verified in a clean environment);
`capi_test` + `fortran_capi_test` pass under `-DSOMA_ENABLE_FORTRAN=ON`. The
`lowfive` preset was 771/771 and has not been re-run since — none of the
merged Tier-1/Tier-2 items touch that transport; its count should move by
+14 once re-run.
**Next:** No phase work remains. What remains is the task list in
"Remaining work" below — 35 items total (R34 found while closing R2, R35
found while closing R9), of which **R1, R2, R3, R4, R5, R6, R7, R8, R9, R10,
R11, R12, R13, R14, R15, and R16 are now closed**, R5 **for the fully-disjoint
rank-set case** (`Runner::Impl::attach_comm_graphs`, `src/runtime/runner.cpp`,
2026-09-11 — see the R5 entry; the partial/overlapping case remains open
under R31) (R3 closed for its first-order conservative builder; runtime
wiring and the high-order path remain, see R3's entry; R1 closed except its
remap bullet, WONTFIX-documented rather than fixed; R13 closed for part 1,
part 2 documented rather than fixed since there's nothing to fix). All of
Tier 3 (R17-R33) plus R34 and R35 remain — 19 open, ranked, none
blocking, several of them real correctness or capability gaps.

SOMA is a C++20 multiphysics coupling library. It represents a coupled
simulation as a **network graph that flows in time**: nodes are components,
edges carry the semantics of how fields are accumulated in time, merged in
space, extrapolated to resolve nonlinearity, and iterated to reduce numerical
error. Independent per-component coupling intervals are the default; lockstep
is the degenerate case.

---

## Read these first

| Document | What it is |
|---|---|
| [`docs/specs/2026-09-02-soma-design.md`](specs/2026-09-02-soma-design.md) | **The vision and binding authority.** Architecture, data model, numerics, space-time remap, scheduling. Everything else argues from this. |
| [`docs/plans/2026-09-02-soma-implementation.md`](plans/2026-09-02-soma-implementation.md) | Master plan: global constraints, verified environment, task index. |
| [`docs/plans/soma-phase-{0..8}.md`](plans/) | 58 tasks with literal code and tests. All complete. |
| [`docs/BUILDING.md`](BUILDING.md) | Build commands, all four configurations, and why each exists. |
| `.superpowers/sdd/2026-09-02-soma-implementation/progress.md` | **The ledger** — every task's status, every ruling and why, every deferred item. ~175KB, chronological, git-ignored but authoritative. This document summarizes it; when they disagree, trust the ledger. |

---

## Build and test

```bash
cmake --preset debug   && cmake --build --preset debug   -j && ctest --preset debug
cmake --preset release && cmake --build --preset release -j && ctest --preset release
cmake --preset asan    && cmake --build --preset asan    -j && ctest --preset asan
```

All three must pass. The Release leg is not optional: invariants guarded only
by `assert` vanish under `-DNDEBUG`, and that defect class got past a
Debug-only suite three times in Phases 0–2. The ASan leg catches a third class
neither sees — UB whose symptom happens to be correct output on this target.

The core builds with **zero optional dependencies**.
`SOMA_ENABLE_{MPI,MOAB,PETSC,KOKKOS,LOWFIVE,PYTHON,FORTRAN}` all default OFF;
yaml-cpp, GoogleTest, and pybind11 are version-pinned and fetched if absent.
A fourth preset, `lowfive` (binary dir `build-lowfive`), builds the HDF5-VOL
transport against `/opt/lowfive-shared` + `/opt/hdf5-shared` — not one of the
three gating legs, but it must pass when that transport is touched.

`-DSOMA_ENABLE_FORTRAN=ON` adds `fortran_capi_test`; `-DSOMA_ENABLE_PYTHON=ON`
adds the 87-test Python suite. Neither is in the 758 count above.

---

## What exists — all 8 phases

```
include/soma/
  base/       ids, rational, result, time_level, units             (Phase 0)
  graph/      region, field_spec, component, channel,
              coupling_graph, validate, descriptor, yaml_io         (Phase 1)
  ir/         op, dag, dump, lowering, checkpoint                   (Phase 2, 7)
  sched/      clock, slab, expand, plan                             (Phase 3)
  runtime/    field_ref, runner, iteration, kernels                 (Phase 4, 7)
  mesh/       serial + MOAB backends                                (Phase 4, 5)
  remap/      RemapPlan, Intx weights, TempestRemap high-order      (Phase 5)
  transport/  copy, mpi, lowfive                                    (Phase 4, 5, 8)
  num/        residual, convergence, accelerators, QR filters       (Phase 7)
  soma_c.h    the C ABI narrow waist                                (Phase 8)
src/          mirrors include/
fortran/      ISO_C_BINDING module over the C ABI                   (Phase 8)
python/       pybind11 module with NumPy field views                (Phase 8)
test/unit/    C++, C, Fortran, and Python suites
examples/     dn_heat (Demo A), lowfive_pair (Demo B), mms_pair
```

- **Phases 0–2** — vocabulary (`Rational` exact time arithmetic, `Units`,
  `TimeLevel`, `Result`/`Diagnostics`), the authored `CouplingGraph` with
  13 accumulating validation rules and YAML round-trip, and the
  `OperatorDAG` + lowering registry with byte-compared golden IR.
- **Phase 3** — rational-clock slab decomposition, time-resolved expansion,
  **acyclicity proof (Gate 1, closed)**, buffer allocation, `ExecutionPlan`,
  lockstep fast path. `LaggedTwoWayCouplingIsAccepted` pins the subtle case:
  an unlagged A↔B cycle is rejected at init; the same topology with a lag is
  a legal explicit scheme.
- **Phase 4** — runtime and serial backend; first coupled run; MMS-verified
  order of accuracy.
- **Phase 5** — MOAB backend, conservative FV↔FV weights via `Intx2Mesh`,
  TempestRemap high-order/monotone path (orders 1–4), MPI n→m transport.
- **Phase 6** — `Waveform`, temporal bases, space-time projection wired into
  lowering and the runner. **Gate 2 (space-time conservation) closed**: the
  original row/column moment-identity proof was structurally blind to stencil
  *placement* — any consistent degree-p reconstruction satisfied the same
  identities, including a mirrored non-causal one — so it was replaced with
  hand-derived golden matrices at degrees 2/3/4 plus a placement check
  independent of any one matrix.
- **Phase 7** — convergence criteria, derived checkpoint sets, accelerators
  (Picard → Aitken → IQN-ILS → IMVJ), QR column filtering (QR1/QR2/QR3).
  **Gate 3 (accelerators) closed.** **Demo A** complete and sabotage-gated.
- **Phase 8** — shared HDF5 rebuild and LowFive revalidation (8.0); C ABI over
  the serialized-graph narrow waist (8.1); Fortran shim (8.2); pybind11
  bindings, 87 tests (8.3); LowFive HDF5-VOL transport with proven in-memory
  interception (8.4); **Demo B** — two unmodified HDF5 executables coupled by
  interception, zero bytes on disk (8.5), sabotage-gated. A post-completion
  fix closed the C ABI's convergence-early-exit gap.

Every phase-completion and gate-closure claim above was independently
re-verified — rebuilt, re-run, and in most cases with a deliberate mutation
reproduced to confirm the guarding test actually fails when the mechanism is
broken. Details and exact commands are in the ledger.

---

# Remaining work

35 items. All of Tier 1 (R1-R7) and Tier 2 (R8-R16) are now closed — R5
closed 2026-09-11 for the fully-disjoint rank-set case by the Runner-wiring
plan; the overlapping-rank-set case it does not cover is tracked under R31
instead, so it is not counted here as an open Tier-1 item — leaving 19 open
in Tier 3 plus R34 and R35. R34 was found while closing R2 and R35 while
closing R9; both are listed at the end, out of tier order, so their
provenance stays visible.
None block declaring the plan complete; each was found, understood, and
deliberately deferred rather than hidden. Grouped by tier and ranked within
tier by impact. Each carries its evidence source so nobody has to re-derive it.
A closed item keeps its number and its entry, struck through, so
cross-references from the ledger still land.

## Tier 1 — real correctness or capability gaps

**R1. PARTLY CLOSED (bullets 2 and 3) on `wt-capivalid`; bullet 1 decided as
WONTFIX-document.** `soma_initialize()` now runs its own iteration-region gate,
sharing `verify_iteration_declaration()` and `match_convergence_measures()`
(`src/ir/checkpoint.cpp`) with the Runner path so the two cannot drift: an
empty `ConvergenceCriterion` is refused with `E-ITER-NOCONVERGE`, an
unresolved or ambiguous measure — and a channel no measure names — with
`E-ITER-MEASURE`. A fourth state found while doing it, `max_iterations < 1`,
was refused too (`E-ITER-MAXITER`); it had been silently clamped to one sweep.
The remap bullet below is NOT closed: it is answered with documentation in
`soma_c.h` and at the point of decision in `soma_initialize()`, because this
ABI performs no remap on any path, so SOMA's inability to build weights says
nothing about a caller's ability to couple the graph — and refusing would
break `pair.yaml`, which this ABI drives correctly today. Only about four of
`resolve()`'s ~15 refusal paths are decidable without a mesh backend, so a
check here would also read as a guarantee it could not give. The original text
follows unchanged.

**The C ABI performs no graph validation and cannot reach conservative
remap.** `soma_initialize()` never constructs a `Runner` or `MeshBackend`
(0 occurrences in `src/capi/soma_c.cpp`). Three confirmed consequences:

  - a realistic wall-to-wall FSI coupling (`fluid_wall`→`solid_wall`,
    non-coincident meshes, this repo's own `test/data/pair.yaml`)
    "initializes" successfully while its conservative remap is unreachable —
    `build_weights` would refuse it with `E-REMAP-REGION` on the Runner path,
    but the C ABI never asks;
  - an iteration region with an **empty** `ConvergenceCriterion` is accepted
    (Runner path rejects with `E-ITER-NOCONVERGE`);
  - an **unresolved or ambiguous** measure→channel match silently leaves
    `rt.converges == false` instead of being rejected (`E-ITER-MEASURE`).

All three are one shape: the C ABI is deliberately thinner and caller-driven
(see `soma_c.h`'s "WHAT THIS ABI DOES NOT DO"), but it has *no* validation
step at all, so callers get silent safe-fallback instead of a loud refusal for
graphs the C++ path rejects outright. **Highest impact: this is the production
entry point for external C/Fortran codes.**
*Source: Task 5.5 and the C-ABI convergence-fix entry in the ledger.*

**R2. ~~A multi-step target never receives its non-final space-time
projections.~~ CLOSED.** With 2 target steps per period, the consumer's tick at
t=1/2 ran *before* the instructions stamped t=1/2, and target step 0's delivery
was overwritten by step 1's before the consumer advanced again. This and Gate
2's "claim 6" (the `n_tgt >= 2` overwrite in `src/schedule/expand.cpp`) were the
same defect seen from two angles — the scheduler's exchange timing was the
mechanism, the undelivered sub-step the symptom. Never a conservation bug (the
integral was folded into a later step's sum); it was the feature's actual
purpose, a per-target-step value, silently failing for one full step every
period. Pre-existing and not confined to space-time: a plain `Remap→Sink` chain
on a 1-against-1/2 schedule had it too.

*Fixed* by giving every timed instance a `TickPhase`
(`include/soma/schedule/expand.hpp`), the missing half of its timestamp: a time
alone cannot say whether a value delivered at `t` is seen by the component step
that begins at `t`. `expand()` and `schedule()` sort on it directly after
`time`, and `Runner::Impl::step_period` interleaves the tick stream on it, so
each instant is deliveries, then ticks, then `Sample`s.

Which deliveries may go first is **the causality contract**, stated on
`TickPhase` and enforced there: *a delivery may be consumed by the target step
beginning at its stamp only if the value it carries is attributed to an
interval ending at or before that stamp.* The two chain kinds attribute
differently, and that is the whole of it. A window-closing chain (accumulation
or space-time) carries a quantity attributed to the target step it closes,
whose end IS the stamp, so it always qualifies — that is what closes R2. An
Instant chain carries the source's state at `(src_step + 1) * src_dt`, so it
qualifies only when its slab ends where its source step does; a target-limited
slab is deferred, and loses nothing by waiting because the next slab carries
the identical `src_step`.

That second clause was **missing from the first attempt** (commit `3ccd0c4`)
and is F1 of the adversarial review: marking every chain instance `BeforeTicks`
made a plain `1 → 1/2` chain hand its target's second sub-step source state
from `[0, 1)` at `t = 1/2` — its own future — for no gain in resolution, since
both slab instances carry `src_step = 0`. Measured provenance, consumer values
by advance time, producer ramp `2, 4, 6`:

| `1 → 1/2` | t=0 | t=1/2 | t=1 | t=3/2 | t=2 | t=5/2 |
|---|---|---|---|---|---|---|
| original `9902f82` | 0 | 0 | 2 | 2 | 4 | 4 |
| first fix `3ccd0c4` | 0 | **2** | 2 | **4** | 4 | **6** |
| with the contract | 0 | 0 | 2 | 2 | 4 | 4 |

Enforcing the contract also fixed three schedules that were acausal in the
**original**: `1/2 → 1/3`, `1/3 → 1/2` and `1/3 → 1/4`. See R34 for the family
it does not fix.

Goldens: no computed value changed and no instruction moved, so the Gate 2
golden matrices, `ExecutionPlan.DumpIsByteExact` and `test/golden/*.ir` all
stood as they were — **no golden file was regenerated.** The plan dump did gain
one token: `after-ticks`, printed only on deferred instructions. It has to be,
because `phase` stopped being a function of the printed `kind` the moment a
non-`Sample` node could carry it, and a byte-compared artifact that cannot
distinguish two behaviours is worse than none.
`ExecutionPlan.DumpShowsWhichInstructionsRunAfterTheTicks` pins the new form
byte for byte; `DumpIsByteExact`'s own fixture has no deferred instruction and
is unchanged.

Pinned by `SpacetimeRunner.EveryTargetStepProjectionReachesTheTargetComponent`,
`SpacetimeRunner.EachProjectionIsConsumedByTheNextTargetAdvance`,
`PlainChainRunner.ASharedValueIsNotDeliveredAcrossTheTargetTick`,
`Runner.EachAccumulationWindowReachesTheTargetStepItClosesInto`,
`Runner.NoTargetStepIntegratesAgainstUnfinishedSourceState`,
`Runner.EveryCompletedSourceStepStillReachesTheTarget` and
`Runner.ASampleSharingAnInstantWithAReductionDoesNotLeakIntoItsWindow`, every
one of them mutation-checked against `3ccd0c4`, `9902f82` or both.

**A CORRECTION, because the wrong reason reached this document once already.**
An earlier revision of this entry claimed the fix could not change accumulation
window COMPOSITION, on the argument that a `Sample` at `j*src_dt` and a
reduction at `k*dst_dt` could only share an instant at or above
`lcm(src_dt, dst_dt)`, "which is the period". **That argument is false.** The
period is the lcm over ALL clocks, so with three or more components
`lcm(src_dt, dst_dt)` can be a proper divisor of it and shared instants exist
strictly inside a period — `A(1/2) → B(1/2)` with an inert `C(1/3)` puts a
`Sample` and a reduction both at `t = 1/2`, and the composition DID shift there
(delivered `0,0,6,6` before, `0,2,4,6` after; the new answer is the correct
one, since the old one leaked source step 1 into a window that had already
closed). The correct argument — the one in `expand()` all along — is that a
reduction closing target step k reads source steps j with `j*src_dt` STRICTLY
less than `(k+1)*dst_dt`, so it never depends on a `Sample` stamped at its own
instant and ordering the `Sample` after it is always safe.
`Runner.ASampleSharingAnInstantWithAReductionDoesNotLeakIntoItsWindow` is the
regression test the false argument talked me out of writing.
*Source: Task 6.3 flagged item + Gate 2 review claim 6 + Gate-4 review F1/F2.*

**R3. The remap path is serial only.** *CLOSED for the first-order
conservative builder, 2026-09-09, branch `wt-parremap`. Conservative remap is
now correct and conservation-tested on a partitioned read at 1, 2 and 4 ranks.
What remains is listed under "still open" below and is narrower than this item
originally was.*

**A correction to this entry's own earlier text, because it was wrong and the
wrongness was load bearing.** A first pass at R3 concluded that the parallel
path was blocked by two things: that `RemapWeights` columns are `MeshView`
local indices so "there is nowhere to put" an off-rank source column, and that
the halo gather is a `Transport` by spec section 5.1 and so belonged above the
remap layer. **Both are false.** `MeshView`'s local entity set is *already*
"owned AND ghost" (`mesh_view.hpp:31-34`), `entity_count` is *already* "the
size a per-entity field buffer over the whole mesh must have"
(`mesh_view.hpp:64-67`), and `MeshBackend::region_size` is implemented as
exactly `entity_count`. So a column space wider than the owned set is what the
contract already says, and the check at `runner.cpp:554` is a consistency
assertion between two derivations of the same number, not a constraint pinning
the matrix to owned cells. And a source halo is `ranksA -> ranksA` with
duplication, not an n->m redistribution between two rank sets -- SOMA's own
`transport.hpp:104-106` says an overlapping rank set "is a local exchange
anyway". Recorded because "blocked on a design question above the remap layer"
is the kind of sentence that buries a solvable problem for years.

**What now works.** Three pieces, none of which touches `RemapPlan`,
`RemapWeights`, `RemapFlags`, `apply_weights`, the DAG or the operator
taxonomy:

1. `build_remap_halo(plan, graph, backend)` (`remap_plan.hpp`, implemented in
   `src/remap/intx_weights.cpp`) asks MOAB for the source cells overlapping
   this rank's target cells -- `set_parallel_comm` +
   `Intx2MeshOnSphere::construct_covering_set` -- and hands them to
   `MoabMeshBackend::adopt_remap_halo`, which adds them to the source mesh's
   file set as ordinary NON-OWNED local entities. `entity_count` grows,
   `region_size` grows with it, and the columns exist.
2. Once they are local, **the existing serial supermesh code needs no change
   at all**: the ordinary intersection over the halo-extended source gives
   target coverage 1.0 to 1e-16. This was the key measurement.
3. `MoabMeshBackend::exchange_remap_halo` fills the halo slots from their
   owners (planned once with an all-to-all; `accumulate_remap_halo` is the
   reverse, used by the audit).

**Measured, on a deliberately misaligned pair** -- MOAB's 3456-cell cubed
sphere with its own 8 partition sets against a 1536-cell partner with 4,
written by `test/tools/make_partitioned_sphere.cpp` with **exactly**
`make_partitioned_sphere 16 7996.4553620 4 <out>.h5m block` (the parameters
matter: the `rr` mode gives a different, more adversarial distribution and
different loss figures, so the mode is pinned rather than left to chance):

| ranks | halo cells adopted | integral of a varying field | source coverage | column defect |
|---|---|---|---|---|
| 1 | 0 | conserved to 3.0e-15 | 1.0 | 7.7e-14 |
| 2 | 1236 / 1728 | conserved to **1.2e-15** | 1.0 | 7.7e-14 |
| 4 | 822 / 1109 / 1373 / 2080 | conserved to **7.4e-16** | 1.0 | 7.7e-14 |

**And the halo is load bearing, CI-enforced without mutating anything.**
`remap_halo_parallel_test` drops the off-rank COLUMNS from the very matrix it
just verified -- which is exactly the matrix a partition-local intersection
would have built -- and re-measures: **66.7% of the quantity lost at 2 ranks,
91.2% at 4** (for a constant field, 66.7% and 87.5%). Independently, disabling
`exchange_remap_halo` loses the same 66.7% at 2 ranks: the geometry and the
value exchange are each necessary and neither is sufficient.

**Still open, and each is a deliberate refusal rather than a gap:**

- **The runtime does not run the exchange.** `Runner` refuses with
  `E-RT-REMAP-HALO` when the source mesh carries a halo, rather than applying
  a matrix to unfilled slots. Two things must be settled first: where a
  COLLECTIVE call may sit in the instruction stream, and whether the halo slots
  of a COMPONENT-OWNED registered array are the coupler's to write.
  **This guard is defensive-only and UNEXERCISED** -- reaching it needs a full
  coupled run under MPI, which no harness here does. Said plainly, because an
  unexercised guard is what the rest of this entry is about. The library-level
  sequence (`build_remap_halo` -> `build_weights` -> `exchange_remap_halo` ->
  `apply_weights`) is complete and tested.
- **The high-order builder is refused on a partitioned read.** With a halo the
  surface it is handed is a piece of the domain plus a fringe, not a closed
  one. Checked rather than assumed: with the refusal removed and TempestRemap
  present, it fails outright at 2 ranks — `E-REMAP-INTX: TempestRemap could not
  build the overlap mesh ... (return code 0, 1024 overlap faces)` — so this is
  a real backend limit on fringed surfaces, not caution. Closing it needs
  either a closed covering (ghost layers on the target too) or a
  fringe-aware stencil.
- **Non-spherical partitioned pairs** are `E-MESH-HALO-GEOMETRY`:
  `construct_covering_set` is declared on the spherical worker only in MOAB
  5.6. Named as that axis, not as "no parallel remap"; a planar construction
  would slot in behind `build_remap_halo` unchanged.
- **`build_remap_halo` allgathers the owned global ids**, O(total cells) per
  rank -- the same ceiling `MpiTransport::build_plan` has (R24). Setup only.
- **Call order is a contract, not a check.** `build_remap_halo` changes
  `entity_count`, so it must run before `make_region_sizes` / `plan_buffers` /
  `schedule()` and before a component registers storage. Calling it late is
  not detected; it surfaces as `E-RT-REMAP-SIZE`.

**The rule this item establishes, for reconciliation with the C ABI (R4).**
Guard on the property that breaks correctness -- **the mesh being DIVIDED**,
i.e. `communicator_size() > 1` -- and never on "a communicator was set". A
one-rank communicator is an ordinary MPI job whose mesh happens to be
undivided: every cell is local and the serial construction is exactly right.
Refusing it would take remap away from every MPI run, and **no serial test can
see the difference**, which is why `remap_parallel_test` runs a one-rank leg as
a POSITIVE control. Any other size-blind refusal in the codebase is a bug by
this standard.

**Also fixed here:** `E-REMAP-COVERAGE` was not collective. A source cell
straddling a target partition boundary is covered partly on its owner and
partly on a rank holding a halo copy, so a per-rank audit reports a correct map
as a coverage failure -- and reports it on SOME ranks only, so those unwind
while the rest enter the next collective and the job hangs instead of failing.
The audit is now folded onto owners (`accumulate_remap_halo`) and reduced over
the communicator, so every rank reaches the same verdict. Its advice text no
longer says "clear the Conservative flag" on a partitioned read: there, that
advice converts a loud correct refusal into the silent 66.7% truncation.

*Source: Task 5.2 "Suspicious, not fixed" + Task 5.5 flagged item 4; closed
and measured 2026-09-09.*

**R4. CLOSED on `wt-capivalid`, but not as written below — the item named the
wrong hazard.** `soma_initialize()` now reads the communicator, for exactly one
purpose: the rank count (`MPI_Comm_size`, which needed no build-system change —
`soma_core` already links `MPI::MPI_CXX` and two sibling TUs already include
`mpi.h`). What it refuses is not "a communicator was set" but the condition
that is actually wrong, and which the item below missed entirely:
**`soma_end_step()`'s convergence early exit is rank-local.** It decides from
this rank's buffer and advances this rank's clock, reducing nothing. Measured
on two ranks with `pair_iterated.yaml` and **no communicator set at all**:
rank 0 reported the window `[1, 2)` and rank 1 `[0, 1)` after the same two
sweeps, with `SOMA_OK` from `soma_initialize()` on both. So the gate fires on
`ranks > 1 && any component converges`, and refuses with
`SOMA_ERR_UNSUPPORTED` naming the components. Size 1 is accepted
unconditionally — `MPI_COMM_SELF` is a legitimate serial scope, and an earlier
version of this fix that refused it turned working code into a hard failure.
`n > 1` with no iteration region is accepted with an advisory. Pinned by
`test/parallel/capi_parallel_test.c` at `-n 1` and `-n 2`; the `-n 1` leg is
the positive control.

**Also corrected: this item's claim that "`test/unit/capi_test.c:605` already
accepts OK-or-UNSUPPORTED, so tightening this breaks no existing test" is
false.** The OK-or-UNSUPPORTED assertion is on `set_communicator`; the
assertion below it — `set_communicator` after `initialize` must be
`SOMA_ERR_STATE` — depends on `initialize` *succeeding*. Compiled verbatim
against a build whose `initialize` refused, it fails with `got 0, expected -3`.
The original text follows unchanged.

**`soma_set_communicator` is still a write-only sink on the C ABI.**
`src/capi/soma_c.cpp` stores `c.communicator`/`c.communicator_set` and
nothing reads either. A C or Fortran caller passing `MPI_COMM_WORLD` gets
`SOMA_OK` and a serial run with no diagnostic; `fortran/soma_mod.f90` exposes
the same. **Partially addressed:** Task 8.5 added the
`RunnerConfig::transport_setup` hook that carries a communicator into a
transport and calls `build_plan`, and LowFive uses it — but the C ABI does
not, and Task 5.4's explicit requirement that `soma_initialize` **fail loudly
if a communicator was set and nothing consumed it** is still unmet.
`test/unit/capi_test.c:605` already accepts OK-or-UNSUPPORTED, so tightening
this breaks no existing test. *Source: Task 5.4 BLOCKING FINDING.*

**R5. RE-GRADED, and it is worse than a missing test: SPEC §4.1's n→m MPI
REDISTRIBUTION HAS NO WORKING PUBLIC PATH AT ALL.** The item used to read
"argued by symmetry with `LowFiveTransport`, not verified end to end — no test
wires it". A test now wires it
(`MpiTransportThroughRunner.*`, `test/parallel/mpi_transport_test.cpp`, `-n 2`
and `-n 4`) and **the symmetry does not hold.**

The hook itself is fine: `RunnerConfig::transport_setup` hands back the very
`MpiTransport` the Runner constructed, `set_communicator` reaches it, and a
`build_plan` refusal surfaces out of `Runner::initialize()` with
`MpiTransport`'s own code. The break is on the Runner's side.
`Runner::Impl::resolve_instructions()` (`src/runtime/runner.cpp`, the
`OpKind::Transport` case) fills only `src_rank_set`, `dst_rank_set`,
`elements` and `components`; **`TransportSpec::src_ranks`, `dst_ranks` and
`global_ids` are never populated anywhere in `src/`**. LowFive survives that
because its `build_plan` ignores those three vectors; `MpiTransport::build_plan`
requires non-empty rank sets and `global_ids.size() == elements`. So:

  - planning the Runner's own spec → `E-TRANSPORT-RANKS`;
  - planning a spec `MpiTransport` accepts → `initialize()` succeeds and the
    first `Transport` instruction fails `prepare()` with `E-TRANSPORT-RESPEC`,
    no byte moved.

There is no argument to `build_plan()` for which this path works.
`MpiTransport::build_plan` has **no caller in `src/` at all** — the only
non-test call in the tree is LowFive's, in `examples/`. So the correct
statement is a Tier-1 *capability* gap, "MPI coupling does not work end to
end", not a testing gap. Closing it means teaching
`resolve_instructions()` to derive a rank set and a global-id list from the
graph and the mesh backend, which is a design change, not a fix.
*Source: Task 8.5 flagged item; re-graded on `wt-capivalid` from the tests
listed above.*

**PRIMITIVE IMPLEMENTED 2026-09-11; RUNNER WIRING NOW ALSO IMPLEMENTED, FOR
THE FULLY-DISJOINT CASE — R5 CLOSED FOR THAT CASE.** `CommGraph` and the
`CrystalRouter` under it are written, compiled and tested
(`include/soma/transport/comm_graph.hpp`, `src/transport/comm_graph.cpp`,
`src/transport/crystal_router.{hpp,cpp}`, `test/parallel/comm_graph_test.cpp`).
§5 of the spec, the data flow into the Runner, is now implemented in
`Runner::Impl::attach_comm_graphs()` (`src/runtime/runner.cpp`): it builds one
`CommGraph` per component pair at `setup()`, over per-component communicators
split from `RunnerConfig::base_communicator`, and populates
`TransportSpec::src_ranks`/`dst_ranks`/`global_ids` from it before handing the
plan to `MpiTransport::build_plan()`. The end-to-end statement "MPI coupling
works through the Runner" is now **true for fully-disjoint component rank
sets, `GlobalId` matching, whole-mesh regions, one CommGraph-backed `Transport`
per run** — see `docs/specs/2026-09-11-comm-graph-design.md` §9 for the
complete ships/refused/uncalled breakdown, including the full,
grep-verified list of `E-RT-TRANSPORT*` refusal codes.

Two things keep R5 from closing completely:

- **Partially or fully overlapping component rank sets are refused**
  (`E-RT-TRANSPORT-OVERLAP`), not handled — this is R31, tracked separately
  below with what closing it actually requires.
- **A residual collective-safety hazard, parked, not fixed.** Inside
  `attach_comm_graphs()`, the `mesh_id()`/`view()` resolution sits below the
  `// From here down the loop is COLLECTIVE` marker and `break`s locally on
  failure. A rank that belongs to NEITHER the source nor destination
  component, and that has not loaded the Transport's region mesh, would
  `break` out locally while member ranks proceed into the collective
  `CommGraph::build()` — a real deadlock/desync hazard in general MPI usage.
  It is not reachable today: every mesh backend exercised
  (`SerialMeshBackend`, the test suite's `PartitionedTestBackend`) loads
  every mesh unconditionally on every rank regardless of component
  membership, and this plan's test fixture (`TwoComponentJob`) partitions
  the whole MPI world between its two components with no bystander rank.
  Same risk category as the "assumption most likely to be wrong" already
  named in this plan (destination-side mesh loading) — the fix is a
  mesh-loading architecture decision, out of scope here. *Source: Task 4
  review, ruled PARKED.*

The mechanism that was missing now exists and is exercised end to end
through a real `MpiTransport`, by both the test suite and `src/` itself.

**Tests: 781/781 on an MPI-enabled Debug build** (770 + the 4 pre-existing
MPI-only entries + `comm_graph_{2,3,4}rank` + the `runner_transport` suite
added while wiring `CommGraph` into `Runner`, at 2/3/4/5 ranks); 28 GoogleTest
cases in the `comm_graph` binary at 3+ ranks, also passing at 5 and 8 and
clean under ASan+UBSan with MPI. The three gating legs are unchanged at
770/770 each, because the new sources are behind `SOMA_HAVE_MPI` exactly as
`mpi_transport.cpp` is. *Counts re-measured 2026-09-11 at the end of the
Runner-wiring plan; source: task-6-report.md in
`.superpowers/sdd/2026-09-11-comm-graph-runner-wiring/`.*

**Nine mutations, each reproduced and each caught** (the discipline of
"Lessons paid for" item 2): removing the same-rank shortcut; removing the
per-peer ordering; making the tolerance boundary exclusive; accepting
ambiguous ownership; advertising ghosts as owned sources; stopping the router
one round early; suppressing `E-COMMGRAPH-NOMATCH`; and dropping each half of
the `3^dim` neighbourhood probe. The last of those found a real coverage gap —
every NearestNeighbor fixture happened to put a query and its match in the
SAME cell, so the whole neighbourhood probe could be deleted with the suite
still green. `CommGraph.AMatchOneCellAwayIsStillFound` exists because of that
and is built so it cannot happen again.

**TWO CORRECTIONS TO THE SPEC, both found by building it.**

1. **§4.5's "a filtered, genuinely-disjoint remainder" is false under partial
   overlap, and so is §4's "`TransportSpec` and `MpiTransport` require zero
   changes" for that case.** The same-rank shortcut removes matches whose two
   ends are on ONE rank. It cannot remove a rank that is a source for some
   entities and a destination for others, which is ordinary under partial
   overlap: rank 2, in both components, may own an entity rank 3 wants (a
   send) while wanting an entity rank 0 owns (a receive). One `global_ids`
   list per rank cannot say that, which is precisely why `MpiTransport`
   refuses overlapping rank sets (R24). No fixed number of `MpiTransport`
   phases fixes it either — two overlapping ranks exchanging with each other
   is irreducible. `CommGraph` therefore publishes `send_plan()`/`recv_plan()`
   (the complete per-peer answer, correct in all three overlap shapes, and
   directly executable as an `Alltoallv`) alongside `source_spec()`/
   `dest_spec()` (the `MpiTransport` shape, available when
   `specs_are_disjoint()`). The fully-disjoint case — the one
   `rules_channel.cpp` actually emits — is unaffected and goes through
   `MpiTransport` unchanged.
2. **Test 13 as specified does not catch the design it exists to rule out.**
   "Assert no bucket exceeds K× the mean" passes against the rejected raw
   `hash(cell) % P`: measured on the refined-patch fixture, the hash gives 618
   against a mean of 535 at two ranks and 409 against 338 at four, well inside
   any sane K. A hash balances population perfectly whenever there are many
   cells of comparable weight. What it destroys is LOCALITY, and that is
   directly measurable: buckets probed per query, measured at 1.41/1.78/1.95/
   2.38 for Morton at 2/3/4/8 ranks against 2.00/3.00/3.97/7.43 for the hash —
   `min(3^dim, P)`, as predicted. Both assertions are in the suite; the second
   is the load-bearing one.

**What §5 needed, and its current state (closed items updated 2026-09-11 by
the Runner-wiring plan; see `docs/specs/2026-09-11-comm-graph-design.md` §9
for the full breakdown):**
- ~~The per-component communicator plumbing through `RunnerConfig`/the C
  ABI.~~ **CLOSED.** `attach_comm_graphs()` splits `base_communicator` once
  per component via `MPI_Comm_split`, colouring by `RankSet` membership,
  entirely inside `Runner::Impl` — no new C-ABI surface was needed; R4's
  single shared communicator is still the only handle threaded through
  `RunnerConfig`, and every component communicator derives from it locally.
- ~~A decision on the gather/scatter buffers.~~ **CLOSED.** `TransportPlan`
  (`src/runtime/runner_impl.hpp`) carries `gather`/`scatter` index vectors and
  `send`/`recv` buffers, sized once in `setup()` from `CommGraph::
  source_indices()`/`dest_indices()`, never in the hot loop.
- ~~Which of the two output shapes a `Transport` instruction uses.~~
  **CLOSED for what `rules_channel.cpp` emits today**: `source_spec()`/
  `dest_spec()` only, because the lowering only ever emits disjoint-rank
  `Transport` instructions; a non-disjoint plan is refused
  (`E-RT-TRANSPORT-OVERLAP`) rather than routed to `send_plan()`/
  `recv_plan()`. R31, below, records what closing the overlap case for real
  requires.

**New follow-ups this plan's implementation surfaced, not present when the
list above was first written:**
- **One `Transport` instruction per run.** `attach_comm_graphs()` refuses
  (`E-RT-TRANSPORT`) an `ExecutionPlan` with more than one `Transport`
  instruction; only the single-Transport case is wired end to end. One
  `Transport` instance per instruction is the named next step.
- **A residual collective-safety hazard, parked.** See the bulleted item
  above this "Tests" paragraph ("A residual collective-safety hazard, parked,
  not fixed") — a rank in neither component that has not loaded the coupled
  region's mesh would desync at the `mesh_id()`/`view()` resolution inside
  `attach_comm_graphs()`'s collective section. Not reachable by any backend
  or fixture this plan exercises; fixing it is a mesh-loading architecture
  decision, out of scope here.

*Source: implemented against the spec, 2026-09-11; Runner-wiring plan
(`.superpowers/sdd/2026-09-11-comm-graph-runner-wiring/`) closed the above
2026-09-11.*

**The design, unchanged from the entry written before implementation.** The missing mechanism is
`CommGraph`, a rendezvous primitive specified in full at
`docs/specs/2026-09-11-comm-graph-design.md`. It covers both matching
regimes a real caller needs — global-ID matching for coincident meshes (e.g.
`pair.yaml`'s `fluid_wall`/`solid_wall`) and coordinate/nearest-neighbor
matching for non-coincident ones — across fully-overlapping, fully-disjoint,
and partially-overlapping component communicators, with a same-rank-copy
shortcut (which turns out to be load-bearing, not just an optimization: it is
what makes a *partially*-overlapping pair deliverable at all, since
`MpiTransport` refuses overlapping rank sets by design, R24) and an O(1)
bidirectional `transpose()` for two-way channels. `TransportSpec` and
`MpiTransport` need no changes — `CommGraph` only ever produces what they
already consume.

Two decisions worth flagging here because they reversed an initial instinct:
a portable, MOAB-independent Crystal Router was chosen over reusing
`moab::gs_data` (proven, but would make the transport layer depend on the
mesh-backend layer, breaking "works without MOAB" and this project's own
practice of owning its MPI primitives); and NearestNeighbor-mode bucket
routing uses a Morton-code partition with load-balanced splitters, not a raw
spatial hash, because a raw hash was found — before any code was written, in
design review — to risk both load imbalance and redundant candidate
comparisons on non-uniform meshes. True separate-executable MPMD (spec §6
mode 3) is explicitly out of scope, matching the spec's own "not implemented
in v1." *Source: design session, 2026-09-11; not yet an implementation task.*

**R6. Accelerators allocate in the hot loop. — CLOSED.**
`AitkenAccel::update()` and the quasi-Newton solve each allocated a fresh
`std::vector<double>` per call — measured exactly 56 allocations for 56 updates
over 8 periods, while the surrounding iteration loop measured **0** over 213
sweeps. Every scratch buffer is now a member of the accelerator, sized on the
first call for a given `n` and reused; `qn_least_squares` keeps its allocating
public signature for the direct tests and the iteration goes through
`qn_least_squares_into`, the same code with the caller's buffers. Measured
after: **0** allocations over the same 8 periods, and 0 over 8 steady-state
windows for all seven kinds (before: 309 Aitken, 320 Steffensen, 5320 Secant,
920 IQN-ILS, 6698 IQN-IMVJ). IMVJ recycles the block its chain evicts, so a
window boundary is allocation-free once the chain is at its cap.
`TheAcceleratorItselfStillAllocatesPerUpdate` asserted only
`allocations <= updates`, which both the defect and the fix satisfy, so it was
renamed `TheAcceleratorAllocatesNothingPerUpdate` and now asserts the zero;
`Accelerator.SteadyStateUpdateAllocatesNothing` covers every kind directly.
Because the buffers are now grown and never shrunk, a size can no longer stand
in for "is there history" — `have_prev_` does, and
`EveryKindHandlesAChangeOfVectorLengthBothWays` covers the shrink and the
regrow for every kind (the old test covered IQN-ILS only). Output verified
bit-identical over 153,457 hex-float iterates across 224 configurations.
*Source: Task 7.5 flagged item.*

**R7. IMVJ's retained Jacobian survives divergence. — CLOSED.** The QN diverged
path cleared `v_`, `w_`, `cols_` but not `blocks_`, so a blow-up originating in
the retained inverse Jacobian was reproduced by every later call, indefinitely.
Two rules now govern it. It is dropped when **it is what diverged**, decided by
recomputing the same step without it — the plain IQN-ILS step
`x_new + Wn alpha` — and asking whether that is finite; and it is dropped
anyway once it has been present for more than one divergence it was not blamed
for. Unconditional clearing was rejected: it discards every past window because
of one bad extrapolation in the current one, which is IMVJ's whole advantage
over IQN-ILS.

The second rule is the guarantee, and it exists because the first version of
this fix shipped a **false termination proof**, caught in review by execution.
That version argued the probe alone terminates: the diverged path clears the
window's columns, so a second divergence would be judged at `cols == 0`, where
the step without the chain is `x_new` itself and therefore finite. The argument
ignores that `have_prev_` survives the diverged return, so `append_column()`
regenerates a column at the top of the very next call, *before* the probe runs.
A caller alternating two window-overflowing pairs held a poisoned chain through
**11 consecutive `E-ACCEL-DIVERGED` calls**, with a fresh benign input still
diverging afterwards — the exact "can never recover" failure R7 exists to
remove. The related unsoundness is that a non-finite probe never proved the
chain healthy in the first place: it proves only that the window's own
`Wn alpha` would have overflowed too, and both can be at fault at once.
The bound the code now provides, stated as the code provides it: **any one
retained Jacobian is present for at most two divergences**, whatever the probe
says and whatever the caller does in between, because only folding a window in
— which makes it a different operator — restarts the count.
`ImvjDropsAPoisonedChainEvenWhenEveryProbeExoneratesIt` is that 11-call
sequence, and it fails against the unbudgeted version.

`have_prev_ = true` on the diverged return was **kept, deliberately**: both
endpoints were evaluated by the solvers and validated finite and only the
extrapolation over them failed, so the pair is genuine secant information —
now documented rather than left looking like an oversight. Clearing it was the
other candidate fix and would have made the original proof literally true; the
budget was preferred because it does not rest on an argument about what `cols`
will be on the next call, nor on the probe being sound in both directions,
neither of which survived contact with a determined caller. The header's "the
secant history is dropped" was replaced by a three-part statement of what
actually goes. Four new tests, each mutation-checked:
`ImvjDropsARetainedJacobianThatIsWhatDiverged` (fails against the old code, and
against dropping the probe),
`ImvjKeepsARetainedJacobianTheDivergenceDidNotComeFrom` (fails against
unconditional clearing),
`ImvjDropsAPoisonedChainEvenWhenEveryProbeExoneratesIt` (fails against dropping
the budget), and
`Iteration.ImvjConvergesToTheSameAnswerAndItsRetainedJacobianPays` — the
convergence-quality test IMVJ did not have, comparing against IQN-ILS rather
than against IMVJ's own first window, because the latter comparison passes with
retention disabled. *Source: carried hazards, Phase 7.*

## Tier 2 — robustness, reachability, and untested paths

~~**R8. `LoweringRegistry::apply_first_match` is a use-after-free under
reentrant registration.**~~ **done.** It range-iterated `rules_`
(`include/soma/ir/lowering.hpp:80`) while a rule callback could call `add()`
and reallocate the vector — ASan-confirmed. Unreachable in v1 because
builtins register once before any dispatch, but a real hazard for future
plugin-style registration. Fixed by snapshotting the rule callables before
iterating. New reentrancy regression test; ASan-confirmed the fix by
reverting it and reproducing a real use-after-free, then restoring and
confirming clean. *Closed on `wt-lowersafe`, merged.*

~~**R9. MOAB remap generality limits.**~~ **validated, tolerances stand.**
`kHighOrderCoverageTol = 1e-6` and `kMaxOrder = 4` were re-measured against a
new production-scale pair (144x96 lat-lon "f19"-class vs 6x30^2 cubed-sphere
"ne30"-class, 13,824 vs 5,400 cells) not in any prior fixture: coverage
defect ~1e-15, three decades tighter than the tolerance. The first-order vs
high-order lat-lon-poles gap is now a hard-pinned regression test rather than
a soft/conditional one. Order-4 degradation on lat-lon sources was shown (via
a controlled same-order cubed-sphere-only comparison) to be a structural
property of the gnomonic least-squares fit near polar rows, not a coarseness
artifact — see the new item **R35** for the generalization and a secondary
diagnostic-tail-truncation finding. *Closed on `wt-remaphard`, merged.*

~~**R10. The remap path is non-reentrant.**~~ **enforced, not just
documented.** `AnnounceSetOutputBuffer` is a process-wide `FILE*` — the
TempestRemap backend's ONLY error channel, since it catches its own
exceptions and returns success. Concurrent `build_high_order_weights()`
calls could corrupt each other's captured diagnostics, silently turning a
real failure into a false success or vice versa. Fixed with a `std::mutex`
held for the entire `AnnounceCapture` lifetime, covering every exit path via
reverse local-destruction order. Mutation-confirmed both ways (repeated
corruption with the lock removed, clean with it restored) by the
implementer, the reviewer, and independently by me. *Closed on
`wt-remaphard`, merged.*

~~**R11. Causality rejection is not pinned against stencil placement.**~~
**done.** Reproduced a genuine mirrored-stencil bug in `spacetime.cpp`'s
`stencil_start()` (swapping the odd-degree past/future lean, exactly the
class Gate 2's golden-matrix review was blind to) and confirmed the runner's
CAUSALITY check (the `E-RT-SPACETIME` "reconstruction stencil reaches
forward" refusal) is independently sensitive to it — one specific existing
test flips to refused, the `SpacetimeRefusal` suite is unaffected — while
weakening/disabling the check itself was separately confirmed to break the
existing causality tests. New pinning test:
`SpacetimeRefusal.TheDefaultCoarseningScheduleIsAcceptedAndWouldTripOnAMirroredLean`.
*Closed on `wt-runtimeverify`, merged.*

~~**R12. Space-time inside an iteration region is untested.**~~ **closed —
and the premise was wrong.** The combination is not merely untested, it is
currently UNCONSTRUCTIBLE: a space-time `Sample` node always ties for a
region's chronologically-first instruction (it runs `AfterTicks` at the same
instant as the tick that feeds it), which pins `enter_time` to that tick's
own time and is then refused by the SAME region-isolation guard this file
already enforces (`E-RT-REGION`) — before any idempotency question can even
arise. This disproved and corrected a false comment in `resolve_regions()`
("no graph reaches this today ... which E-SCHED-REGION refuses first").
New pinning test:
`Iteration.ASpacetimeChannelInAnIterationRegionIsRefusedNotSilentlyWrong`.
Framed carefully as "unconstructible by every attempt made," not as a
permanent structural impossibility. *Closed on `wt-runtimeverify`, merged.*

~~**R13. `check_region_isolation()` is unreachable through `schedule()`**~~
**part 1 done, part 2 documented.** Confirmed (by construction, not just
argument — three independent attempts found no counterexample) that no
graph reaches this backstop through the real `schedule()` path; closed with
a direct unit test that mutates a real scheduled `ExecutionPlan`'s
`region_exit` marker and feeds it straight to `Runner`, confirmed load-bearing
by disabling the check and watching the new test (and only that test) fail.
Relatedly, confirmed **`OpKind::Accelerate` is never emitted by any lowering
rule** (zero hits in `rules_channel.cpp`/`rules_derived.cpp`) — documented at
`IterationRegion::carried` (`include/soma/ir/dag.hpp`) that this is because
acceleration is a region-level `Accelerator`, not a per-node op, per Task
7.3, cross-checked against the design spec and phase plan. *Closed on
`wt-runtimeverify`, merged.*

~~**R14. `test/data/pair.yaml` specifies an unschedulable graph**~~ **done.**
Checked exhaustively against every caller in the repo (C, Fortran, Python,
capi tests) — none reaches `schedule()` on this fixture. Closed with a
comment explaining the `accumulate`+`spacetime` conflict and pointing to
`pair_accumulate_remap_relax.yaml` as the schedulable sibling, rather than
splitting the fixture. *Closed on `wt-runtimeverify`, merged.*

~~**R15. Bidirectional fields emit a degenerate self-copy.**~~
**documented, not a bug.** `LoweringContext`'s own doc comment already
explains this is deliberate (required by `prove_acyclic()`/`expand()`); added
a cross-reference comment at the step-3 site plus a regression test pinning
the self-copy's actual identity (the Sink's input arc equals the field's own
Source output arc), since the only prior coverage asserted acyclicity but
never inspected the DAG shape. No behavior change. *Closed on `wt-lowersafe`,
merged.*

~~**R16. The runtime `memcmp`s only *registered* storage when verifying
rollback.**~~ **documented.** Added caveats at the `memcmp` site
(`iteration.cpp`), `Component::save_state()`, and
`RunStats::checkpoint_verifications` all stating rollback verification
covers only registered field-registry storage, not a component's private
state. `soma_c.h`/`soma_mod.f90` checked and found not to overclaim. Two
further overclaim sites (`include/soma/ir/lowering.hpp`'s `CheckpointSet`
comment, the design spec §5.3) found and left for a later documentation
pass — flagged, not silently ignored. *Closed on `wt-runtimeverify`,
merged.*

## Tier 3 — engineering, packaging, and hygiene

**R17. There is no CI.** No `.github/`, `.gitlab-ci.yml`, or equivalent exists
at the repo root, and `docs/BUILDING.md` says so explicitly ("there is no
separate CI service config (this repo has none)"). The plan's global
constraints specify `-Werror` "in CI only" — so `-Werror` is enforced nowhere
today, and the three-leg discipline this project depends on is entirely
manual. **Highest-value item in this tier by a wide margin.**
*Verified 2026-09-09.*

**R18. There are no `install()` or `export()` rules.** Neither the top-level
nor `src/CMakeLists.txt` contains any. SOMA cannot be installed, packaged, or
consumed downstream via `find_package(SOMA)` — only built in-tree. For a
library whose entire premise is being embedded in other people's simulation
codes, this is a real gap. *Verified 2026-09-09.*

**R19. LowFive's working configuration is fragile.**
`/opt/lowfive-shared/lib/liblowfive.dylib` links
`/opt/mpich/lib/libmpi.0.dylib` by absolute path, and the exact
missing-library error already drifted once (`libhdf5.310.dylib` →
`libmpicxx.12.dylib`) between Tasks 8.0 and 8.4 as MPICH was updated on this
machine. **If MPICH is upgraded again, re-verify before assuming LowFive
loads.** Also: `atexit(H5close)` is heavy-handed (a library terminating HDF5
on the application's behalf); `H5Eset_auto2` is toggled globally in one test;
neither that toggle nor the VOL singleton's counter is thread-safe.
*Source: Task 8.4 flagged items.*

**R20. `examples/lowfive_pair/main.cpp` includes `<hdf5.h>` directly.**
`src/CMakeLists.txt` says `lowfive_transport.cpp` is meant to be "the only
translation unit in the project that may include `<hdf5.h>`", read as scoped
to `soma_core` rather than a project-wide ban. Reasonable, but write the
policy down. *Source: Task 8.5 flagged item.*

**R21. Demo B's `TransportSpec` rank sets are hardcoded to component
declaration order.** `src_rank_set`/`dst_rank_set` are `0`/`1` (component IDs,
not `-1`), determined by reading `rules_channel.cpp` rather than from any
documented contract. Reordering `build_graph()`'s `add_component()` calls
would silently break `prepare()`'s `E-TRANSPORT-RESPEC` check. Closing it
properly needs inspecting the resolved DAG before `Runner::initialize()`,
which the one-hook design does not expose. *Source: Task 8.5 flagged item.*

**R22. Do not pin quasi-Newton iteration counts across build types.** IQN-ILS
takes 211 sweeps in Debug/ASan and 219 in Release on identical source and
inputs — almost certainly FMA contraction in the pivoted-QR least squares
under `-O3`. Not a correctness bug (answers agree to 1e-14 K, and results are
bit-reproducible within a build), but a live trap for future tests. Task 7.4's
pinned counts are safe only because they are on an exactly-linear problem.
*Source: Task 7.6 flagged item.*

**R23. QR filter gaps.** `limit` near 1 is legal but destructive for QR1 (at
0.9 it empties the history on any window with ≥2 comparable columns) with no
per-kind validation. The reuse cap evicts *before* the filter runs, so the
filter cannot save an informative old column by dropping a redundant newer one
(preCICE has the same ordering). `QR1FILTER_ABS` exists upstream with no
`FilterKind` here — deliberate, since it is unit-dependent, but it means the
enum is not a superset of preCICE's. **Open product decision:**
`QR2 @ limit = 1e-2` is preCICE's default and now SOMA's, but every synthetic
problem in this suite is well-conditioned, where filtering can only cost;
whether 1e-2 is right *here* was deliberately left alone.
*Source: Task 7.4.*

**R24. `MpiTransport` scaling and lifetime hazards.** `build_plan` allgathers
every global id to every rank — O(total entities) per rank, fine at test
scale, will not scale. The destructor and a second `set_communicator` are
**collective** (they free the duplicated communicator), so asymmetric
destruction across ranks hangs. Overlapping source/destination rank sets are
refused by design (one `global_ids` list cannot express two distributions), so
`MpiTransport` needs ≥2 ranks. *Source: Task 5.4 carried notes.*

**R25. Diagnostic and dump inconsistencies.** The plan dump omits
`Instruction::src_step`/`dst_step` (deliberately, to keep `DumpIsByteExact`),
so `--soma-dump-plan` no longer fully describes a space-time plan. It does now
show `Instruction::phase`, as `after-ticks` — see R2. The
`Residual` node's shadow buffers print `persistent=yes` while the region line
prints `carried=[]` — they agree in outcome but say apparently contradictory
things about the same buffers. The IR dump prints SI base units (`kg·s^-3`)
while `yaml_io.cpp` prints the friendly form (`W/m^2`); unify on
`units_symbol()`. `dump_text` on a cyclic DAG emits only the banner,
indistinguishable from empty. *Source: Tasks 2.4–2.8, 6.3, 7.6.*

**R26. The `iqn+qr2` comparison row is informational and ungated** — an
over-filtered least squares producing `E-ACCEL-DIVERGED` is a documented
outcome, not a regression — so it could rot without failing the suite. It
prints on every run. *Source: Task 7.6 flagged item.*

**R27. MOAB is noisy and leaks bounded memory.** It writes ~15,000 lines to
stdout during the two sphere builds (ugly in an HPC job log; suppressing needs
`rdbuf` swapping around a library call). `build_weights` leaks unreferenced
vertices into the `moab::Core` on the sphere/plane paths — bounded per build,
once per plan via `RemapCache`. *Source: Task 5.2.*

**R28. `regions_coincide` is O(cells) and re-run per Remap instruction, not
memoized.** Setup-only and negligible now; wasteful on a large subcycled
channel. Relatedly, `E-RT-REMAP-SIZE` is now defensive-only, reachable only by
mutation. *Source: Task 5.5.*

**R29. Python binding rough edges.** `ForwardingComponent::name()` allocates
per call, so a Python peer is **not** allocation-free in the hot loop (the
`AllocatesNothingWhileRunning` guard does not run against the Python path, so
nothing fails today). `Session.registry` is writable after `initialize()`,
but the Runner has already resolved its pointers, so late bindings silently do
nothing. `SOMA_ENABLE_PYTHON` + `SOMA_ENABLE_MPI` is untested — the
communicator test skips rather than asserting wrongly.
*Source: Task 8.3 flagged items.*

**R30. Phase 0–2 deferred minors, still open.** `CouplingGraph::channel()`
bounds check is assert-only (caught later by `channel_mut()`);
`ChannelBuilder` holds a raw `CouplingGraph*` and dangles if the *graph* is
relocated or destroyed; `add_field`'s three `push_back`s have no rollback if
the second or third throws `bad_alloc`; `E-DESC-TRUNC` is reported for a
zero-denominator `Rational` (semantic corruption, not truncation — misleading
but not wrong); `deserialize` lacks an upfront "N records fit in the remaining
bytes" check, so a hostile header claiming 100M records takes ~2.8 s to reject
instead of microseconds. *Source: Tasks 1.1–2.3.*

**R31. `Transport` is decided by `RankSet::overlaps()`, coarser than exact
equality** — partially overlapping rank sets would skip `Transport` for the
non-overlapping ranks. Untested; flagged for revisit in Task 5.4 and not
closed there. *Source: Task 2.4–2.8.*

**Still open as of the Runner-wiring plan (2026-09-11), and now more
precisely bounded.** `Runner::Impl::attach_comm_graphs()`
(`src/runtime/runner.cpp`) now refuses a non-disjoint plan outright
(`E-RT-TRANSPORT-OVERLAP`) rather than mishandling it, so the coarseness
above is a refusal, not a silent wrong answer — but it is still a refusal,
not a fix. `CommGraph` already publishes the general answer:
`send_plan()`/`recv_plan()` (`include/soma/transport/comm_graph.hpp`) give
the complete per-peer send/receive lists, correct in all three overlap
shapes (fully-disjoint, fully-overlapping, partially-overlapping), not just
the disjoint one `source_spec()`/`dest_spec()` cover. Closing R31 for real
means one of two things: build a **third `Transport` backend** consuming
`send_plan()`/`recv_plan()` directly (the intended path — `CommGraph` was
designed to hand off exactly this shape), or hand-roll an open-coded
`Alltoallv` in the Runner itself, which duplicates logic `MpiTransport`
already owns and reopens the collective-safety and error-agreement
discipline `mpi_transport.cpp` and `comm_graph.cpp` both already implement
— explicitly the worse option. *Source: Runner-wiring plan Task 7, following
the design correction recorded in `docs/specs/2026-09-11-comm-graph-design.md`
§4.5.*

**R32. LeakSanitizer is not active on macOS.** The ASan leg proves
memory-safety and UB-freedom, **not** leak-freedom. Nothing in this project
has ever been checked for leaks by a tool. *Source: Task 5.4.*

**R33. Upstream MOAB/TempestRemap defects worth reporting.**
`mbtempest --type 5` segfaults loading an ICO-dual mesh generated by MOAB's
own `GenerateICOMesh --dual`. `Intx2Mesh::FindMaxEdgesInSet` uses a
non-recursive `get_entities_by_dimension`, so MOAB's `TempestRemapper` path
fails `MB_INVALID_SIZE` on an ExodusII file with nested element-block sets.
Neither is on SOMA's path; both are in the installed version.
*Source: Task 5.3.*

---

## Found while closing R2

**R34. An Instant chain whose source step spans more than two target steps
still delivers acausally.** Tier 1. `src_dt > 2*dst_dt` — e.g. `1 → 1/3`,
`1 → 1/4`, `1 → 2/5`, `1/2 → 1/5` — leaves the target integrating against
source state that was not finished when that target step began. **Pre-existing:
measured identical on `9902f82`, before any R2 work**, and pinned as such by
`Runner.ASourceStepSpanningMoreThanTwoTargetStepsIsStillDeliveredEarly` in
test/unit/runner_test.cpp, which passes against that commit too.

R2's causality contract cannot reach it, and the reason says what the fix has
to be. Deferring a target-limited delivery past the tick at its own stamp buys
exactly ONE target step of protection; the value is still WRITTEN into the
target's array, and the next tick, at `(k+2)*dst_dt`, reads it. That suffices
iff `src_dt <= 2*dst_dt`. Beyond that the value must not be written at all yet
— which means an Instant chain should emit once per SOURCE step, at that step's
end, instead of once per slab. That changes instance counts (`Expand.
NonNestingIntervalsProduceIntersectedInstances` pins 4 for `1/2` against `1/3`,
which would become 2), the buffer plan, region contiguity and every plan dump,
so it is a redesign of slab-based emission and wants its own design pass, not a
patch. `decompose()` would keep its job for the accumulating and space-time
paths, which are unaffected.

Note this is the one place where the "explicit, one-period-lagged exchange"
the MMS study measures is not what the schedule actually does, so
test/verification/mms_transfer_test.cpp's stated order argument should be
re-read when R34 lands. *Source: Gate-4 review of R2, remediation round.*

---

## Found while validating R9 (Tier 2)

**R35. The order-4 degradation R9 flagged on a small mesh GENERALIZES to
production scale, independent of resolution, and the diagnostic tail relayed
for it is not scale-invariant.** Tier 2, informational rather than a
correctness bug: `E-REMAP-DEGRADED` fires correctly in both cases below, so
nothing ships a wrong answer. Two distinct findings, both from the same new
test, `RemapHighOrder.ProductionScaleFvAndCubedSphereValidateCoverageTolAndOrderFourGeneralizes`
(`test/unit/remap_test.cpp`), built on a pair modeled on real climate-model
grids never previously in this suite's fixtures: a 144x96 latitude-longitude
grid (CESM/E3SM's "f19" class, 13,824 cells) against a 6x30^2 equiangular
cubed sphere ("ne30" class, 5,400 cells), both generated in-test with no
network dependency.

1. **`kMaxOrder = 4` (`src/remap/tempest_weights.cpp`) is confirmed as a
   correct upper bound on what may be REQUESTED, but order 4 is NOT safe to
   assume will SUCCEED whenever the source has a lat-lon (pole-triangle)
   topology, at any resolution.** Orders 1-3 build cleanly on the f19/ne30
   pair with source-coverage defect measured at ~1e-15 -- three decades
   tighter than the 1.7e-8 the small 32x4 fixture measured, and nine tighter
   than `kHighOrderCoverageTol = 1e-6` itself, so that tolerance is now
   evidenced at production scale with wide margin, not just asserted.
   Order 4, however, hits `E-REMAP-DEGRADED` on this pair exactly as it does
   on the small 32x4 one (`RemapHighOrder.SilentDegradationToFirstOrderIsCaughtAndRefused`).
   Since the small case was coarse (128 source cells) and this one is fine
   (13,824 source cells, ~72x the resolution) and BOTH degrade while a
   same-order cubed-sphere-only pair (`RemapHighOrder.BuildsAtOrdersTwoThreeAndFour`,
   24-vs-16 cubed sphere, no lat-lon anywhere) builds cleanly at order 4 with
   no warning at any resolution measured so far, the degradation is a
   structural property of the gnomonic-coordinate least-squares fit near a
   lat-lon mesh's polar rows, not a coarseness artifact that finer meshes
   grow out of. Practical consequence for a caller: a production atmosphere
   model regridding FROM a lat-lon grid at order 4 should expect (and will
   safely receive) a refusal, not a degraded-but-successful map; order <= 3
   is the realistic ceiling for a lat-lon source, while a cubed-sphere source
   can still be requested at order 4. `kMaxOrder` itself needs no change --
   it already correctly caps what can be ASKED for, and the runtime
   degradation check is what actually decides success per-mesh, which this
   test shows generalizes correctly to production scale.

2. **The diagnostic text relayed for `E-REMAP-DEGRADED` is not scale-invariant
   and can omit TempestRemap's actual warning entirely.** The error message
   is built from `tail(out.log, 6)` (`src/remap/tempest_weights.cpp`) -- the
   last 6 lines of TempestRemap's captured `Announce` stream. On the small
   32x4/6x6^2 fixture, the "WARNING: Poor conditioning in fit matrix ...;
   dropping to 1st order" line happens to land inside that 6-line window. On
   the 144x96/6x30^2 production pair it does NOT: TempestRemap prints
   progress markers every 1000 elements while building the map
   (`..Element N/13824`), and by the time the map finishes those trailing
   progress lines alone fill the last 6 lines, pushing the actual
   conditioning warning out of the relayed tail. Measured directly: the
   `E-REMAP-DEGRADED` error's `.message` on the production pair contains
   SOMA's own boilerplate ("...silently built a lower-order map...") but does
   NOT contain the string "Poor conditioning" anywhere in the relayed text --
   confirmed by the test's own (non-asserting, evidence-only) check. So a
   caller debugging a real production `E-REMAP-DEGRADED` failure may see no
   reference at all to WHICH cell degraded or why, just the boilerplate
   sentence and irrelevant progress spam. Not fixed here: the right fix (grep
   the full log for the actual warning line rather than relaying a fixed
   line count, or making the progress-print interval configurable/silenceable)
   touches TempestRemap's own verbosity settings and this backend's message-
   construction convention, which is more than this task's scope of
   "validate kMaxOrder and kHighOrderCoverageTol." Flagged here rather than
   silently left for the next person to rediscover at 2am reading a
   production log with no useful text in it.
*Source: R9 validation, 2026-09-10, against a pair not in any existing test
fixture.*

---

## Order of work (as directed 2026-09-09)

Tier 1 in full, then Tier 2 in full, then a full validation pass, then Tier 3.
Within Tier 1 the items group into four file-disjoint tasks:

| Task | Items | Primary files |
|---|---|---|
| A | ~~R1 + R4 + R5~~ — **done**, except R1's remap bullet (WONTFIX-document). R5 **closed for the fully-disjoint case**: `CommGraph`, the primitive that closes its upstream half, is **implemented and tested**, and the Runner wiring (§5 of `docs/specs/2026-09-11-comm-graph-design.md`) is now also implemented in `Runner::Impl::attach_comm_graphs` (`src/runtime/runner.cpp`, 2026-09-11 Runner-wiring plan). Overlapping rank sets remain refused rather than handled — tracked as R31, not as an open R5 item. See the R5 entry and the spec's §9 for the complete ships/refused breakdown | was `src/capi/soma_c.cpp`, `include/soma/soma_c.h`, `test/unit/capi_test.c`, `test/parallel/capi_parallel_test.c`; now also `include/soma/transport/comm_graph.hpp`, `src/transport/{comm_graph,crystal_router}.cpp`, `test/parallel/comm_graph_test.cpp`, and `src/runtime/runner.cpp` |
| B | ~~R2 — space-time sub-step delivery~~ **done** | `src/schedule/expand.cpp`, `src/runtime/runner.cpp` |
| C | ~~R3 — parallel remap~~ **done** for the first-order conservative builder: halo, exchange, collective audit, conservation tested at 1/2/4 ranks. What is left is the runtime wiring and the high-order path — see R3 | `src/mesh/`, `src/remap/`, `test/` |
| D | ~~R6 + R7 — accelerator allocations, IMVJ divergence~~ **done** | `src/num/accel_*.cpp` |

R6 and R7 are grouped because they live in the same sources; R1/R4/R5 because
they are one seam. Tier 2 begins with R8, which is a known use-after-free with
a one-line fix.

---

## Known environment (verified, not assumed)

| Dependency | State |
|---|---|
| **TempestRemap** | Present and in use since Task 5.0's MOAB rebuild. The "absent" blocker recorded in the original plan is resolved. |
| **LowFive** | Working since Task 8.0's shared-HDF5 rebuild, at `/opt/lowfive-shared` + `/opt/hdf5-shared`. Fragile — see R19. |
| **MPICH** | `/opt/mpich`, and **not stable across this project's lifetime**: it was upgraded mid-project and broke LowFive's link once already. |

---

## Corrections made along the way, worth knowing about

- **A "C ABI drift" claim was made, then partly retracted.** Three symptoms
  (write-only `soma_set_communicator`, ignored convergence, unreachable
  wall-to-wall remap) looked like one root cause needing a single
  reconciliation task. That was wrong about the framing: the C ABI is
  *deliberately* caller-driven and documented as such, and `soma_run`/
  `soma_run_period` — the only functions that would ever drive full-DAG
  execution — return `SOMA_ERR_UNSUPPORTED` always, by design, not by
  omission. The convergence gap was real and is now fixed. The other two are
  real and remain open as R1 and R4.
  **Retracted again, on `wt-capivalid`:** the framing was right about the remap
  symptom and wrong about the communicator one. `soma_set_communicator` being
  write-only was never the hazard — the hazard is that
  `soma_end_step()`'s convergence early exit is rank-local and needs no
  communicator to desynchronise a multi-rank run. Fixing what the symptom
  named, rather than what it was a symptom of, produced a first attempt that
  refused `MPI_COMM_SELF` (breaking a correct serial run), missed the real
  defect entirely, and shipped header prose advising callers to "simply not
  call `soma_set_communicator`" — which steers them straight into it. See R4.
- **Gate 2's claim 6 was re-classified mid-review** from a scoping call to a
  real deferred correctness bug, and is now recognized as the same defect as
  Task 6.3's undelivered-sub-step finding (jointly R2). Both were right, and
  both were one missing bit: a timestamp cannot order an exchange against the
  component advance that begins at the same instant. R2 is closed.
- **QR3 was nearly shipped as a no-op** with a comment asserting no standard
  definition exists. The implementing agent found the definition (Davis et
  al., Algorithm 3; preCICE `QR3FILTER`) and corrected itself before
  committing.

---

## Lessons paid for

The failure modes this codebase has actually hit.

1. **`assert` is not enforcement.** Release is `-O3 -DNDEBUG`. This got past
   review three times in Phases 0–2 alone (`CouplingGraph`, `OperatorDAG`,
   and `Rational` — the last *after* that file had already been through a fix
   round). The Release CTest leg exists because of it.
2. **A test that passes against the broken code proves nothing.** Three Phase
   0 "overflow tests" used magnitudes just under `INT64_MAX` and passed
   against the very bug they were named for. Always verify a regression test
   fails against the pre-fix code, or — for an already-merged fix — disable
   the guard and confirm the expected failure reappears *exactly*.
3. **Never hold a reference into a container that can reallocate.**
   `ChannelBuilder` held a `Channel&`; a reallocating `add_channel()` made it
   a heap-use-after-free. R8 is the same bug, still open in another file.
4. **Asymmetric serialization is silent data loss.** `to_yaml` never emitted
   fields `from_yaml` read; the round-trip test passed because the fixture
   used only defaults.
5. **Locale can corrupt byte-compared artifacts.** `std::to_string(double)`
   emits `0,7` under a comma-decimal locale — caught one task before it would
   have poisoned the golden files as an unreproducible CI-only diff.
6. **`FIND_PACKAGE_ARGS` consults the CMake user package registry** and once
   resolved an unrelated project's build tree. Both registries are now
   disabled and both fetched dependencies are version-pinned.
7. **A structural-identity proof can be blind to placement.** Gate 2's
   original conservation proof was satisfied by any consistent degree-p
   reconstruction, including a mirrored non-causal stencil. When a proof
   argues from an algebraic identity, check whether a wrong-but-structurally-
   similar answer satisfies it too before trusting it as the gate.
8. **Partial convergence measurement is false convergence.** A criterion
   covering one channel of a two-channel region let IQN-ILS drive that
   channel's residual to exactly zero on sweep 3 while the other stood at
   0.8 — reporting convergence at 2.25 against a true 2.5, a 10% error with a
   monotone residual history the whole way. A quasi-Newton step is a
   least-squares fit over the concatenated vector and can zero one block at
   another's expense. `E-ITER-MEASURE` now requires a criterion to cover
   every channel in the region.
9. **Implementer reports have asserted work that did not exist** — "the code
   is written and present" when a grep found nothing, more than once. Check
   claims by grepping or running, not by reading the report. Every closure
   claim in this document is backed by an independent re-run.

---

## How to resume

1. Read the spec, then the ledger's tail for the last `DONE`/`CLOSED` entry.
2. Confirm the tree is green in Debug, Release, and ASan.
3. Pick from "Order of work" above, or open a new task against the spec if the
   goal is new functionality rather than closing a listed gap.
4. Per task: implementer → independent adversarial reviewer → remediation of
   what the review found → your own independent re-verification of both the
   fix and the review's central claim, reproducing the mutation yourself
   rather than accepting a report. This discipline found every real defect in
   the ledger.
5. Record every ruling in the ledger. It is what survives a lost session.
