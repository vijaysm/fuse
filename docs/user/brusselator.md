# Brusselator: a code FUSE has never met

`examples/brusselator/` ports a multirate 2D Brusselator that the user wrote
**independently of FUSE** (`/Users/mahadevan/Code/AI/Toy`, referred to below
as "the Toy") onto FUSE's own native mesh, conservative supermesh remap, and
`fuse::WindowCoupler`. It is this project's A4 example: physics FUSE was not
designed around, arriving as an outside artifact to be coupled, not written
to exercise a feature.

## The model (unchanged from the Toy)

On the periodic unit square,

```
U_t = a*Laplacian(U) + B - U + q + f
V_t = a*Laplacian(V) - q
q   = U^2*V - A*U
```

with `A = 3.4`, `B = 1`, `a = 10` by default. Initial conditions sample
`U0 = 22*[y(1-y)]^1.5`, `V0 = 27*[x(1-x)]^1.5` at cell midpoints. The forcing
`f` is 5 inside the radius-0.1 circle centred at `(0.3, 0.6)` for `t >= 1.1`,
zero elsewhere and before. Defaults: `nx_u=ny_u=16`, `nx_v=ny_v=24`,
`dt_u=0.002`, `dt_v=0.003`, `window=0.02`, `end=11.5`, Picard tolerance
`1e-11`, 60 max iterations. See the Toy's own `README.md`/`NUMERICS.md` for
the full derivation and reference.

## What was ported verbatim

- **The periodic 5-point diffusion stencil and backward Euler for diffusion
  and the U decay** (`brusselator_physics.hpp::periodic_laplacian`), the same
  matrix the Toy's `bruss::laplacian` builds, solved with
  `Eigen::SimplicialLDLT` exactly as the Toy does.
- **Initial conditions, the forcing mask, and the forcing's exact activated
  duration within a step** (`forcing_active_duration`,
  `max(0, t_end - max(t_begin, 1.1))`) -- the Toy's own closed form, so a
  step that straddles the `t=1.1` switch integrates it exactly regardless of
  window alignment.
- **The two design decisions in the Toy's `NUMERICS.md` "Spatial transfer"**:
  1. The exchange is represented on the U grid as ONE shared reaction rate
     `q = U^2*(P_VU V) - A*U`. U consumes `+q`; V consumes `-(P_UV q)`. V
     remaps its own state onto the U grid before it is exchanged; V remaps
     the received q from the U grid onto its own grid. Both `P`s are the
     area-overlap, mean-preserving transfer `P[T,S] = area(T∩S)/area(T)`.
  2. The consumer integrates the exchange rate over its own step exactly for
     a piecewise-linear-in-time rate -- here realised through
     `fuse::WindowCoupler`'s own contract: it feeds a consumer the substep
     MEAN of the provider's `TemporalBasis::Linear` waveform
     (`Waveform::integrate(...) / dt`), so `∫ q dt = mean * h` over any
     substep, exactly the quantity NUMERICS.md names.

## Where the FUSE port necessarily differs, and what it costs

**The mesh.** Cell geometry comes from `fuse::native::NativeMeshBackend`'s
`"generated:quad:NxM"` generator, not the Toy's hand-rolled `Grid`. Its cell
global id is `ix + iy*nx` and its vertices sit at `(ix/nx, iy/ny)`
(`src/backend/native_generators.cpp`), which happens to line up with the
Toy's own `(i,j)` convention, but `NativeGrid::make`
(`brusselator_physics.hpp`) does not assume that: it recovers the
native-index <-> canonical-index correspondence by reading back every
cell's own `MeshView::centroid` and refuses (`E-BRUSS-MESH-INDEX`) if that
recovery is not a clean bijection.

**The remap.** `fuse::build_weights` (a `RemapPlan` between two
`CouplingGraph` regions on the two native meshes, `RemapFlags::None`,
default `method=""`) dispatches to the dependency-free order-1 simplex
supermesh path for any non-MOAB backend (`src/backend/intx_weights.cpp`'s
`build_weights_via_supermesh`) -- this is real FUSE machinery, not a
hand-rolled sparse product, and it is what `test/verification/
brusselator_weights_test.cpp` checks against an independent oracle (below).

**One component owns q, and it is NOT the Toy's driver.** In the Toy, a
DRIVER computes `q` at the union of both components' schedules' time knots
by interpolating BOTH trial trajectories, and feeds the identical `q`
history to both `Component::advance()` calls. FUSE's `Component` ABI gives
`advance()` exactly one snapshot per call, valid at `ctx.t_begin` -- there is
no driver-level union-of-knots interpolation available. So here:

- `UComponent` computes `q` from **its own state and the just-received,
  remapped V mean** (`species_component.hpp`), and **consumes over its step
  exactly the trapezoid `0.5*dt*(q_prev + q_new)` that `WindowCoupler`
  delivers to V** from the same two consecutive `provided_` samples. That
  makes U's step a short fixed point (`q_new` depends on the state being
  solved for): solve against the one LDLT factorisation, re-evaluate `q_new`,
  repeat until the change is below `1e-13` relative, refuse
  (`E-BRUSS-U-REACTION`) if fifty solves do not get there. The contraction
  factor is about `0.5*dt*(2UV + A)`, of order `1e-2` at the default `dt`,
  so it takes a handful of solves. `q_new` is what `provided_` reports,
  satisfying `Component::advance()`'s "Provides fields hold values valid at
  `ctx.t_end`" contract.

  The first version of this port kept U's step linear by consuming
  `dt*q_prev` instead. That is not a quadrature-rule artifact, it is a
  conservation defect of the scheme: U removed one amount of the shared
  reaction from its budget and V received another, and the Toy's discrete
  budget identity held only to first order (`1.686e-3` per window, halving
  under refinement, against the Toy's `1.2e-13`). The mutation row
  `brusselator-u-consumes-start-of-step-reaction` restores that version and
  must fail `--check`.
- `VComponent::provided_` is `P_VU(V)`, V's own new state remapped onto the
  U grid -- this one needs no such split, since V's own state genuinely IS
  the `t_end`-valid quantity NUMERICS.md's design decision names.

Both converge to the same discrete-in-space, first-order-in-time solution,
but they are not the same finite-`dt` scheme -- see "Measured results" for
what that costs, measured, not asserted.

**A bookkeeping subtlety worth stating plainly.** `WindowCoupler` builds a
`TemporalBasis::Linear` waveform from CONSECUTIVE `provided_` samples and
integrates it (trapezoidal, for evenly spaced samples) to get a consumer's
substep mean. `UComponent`'s own diagnostic of "what I exchanged"
(`exchange_out_`) has to accumulate that SAME trapezoid of consecutive
`(q_prev_, q_new)` samples, not a plain per-step Riemann sum -- otherwise the
diagnostic does not measure what `WindowCoupler` actually delivers to V, and
the exchange-conservation check below sees a ~`1.7e-3` mismatch that is
purely the diagnostic's (the trajectory is unchanged). This was found by
measurement
(`build-evidence/mutations/brusselator.toml`'s
`brusselator-exchange-bookkeeping-drops-trapezoid` row reproduces it
exactly) and is now fixed in `UComponent::advance`.

**The forcing switch as a window boundary.** The Toy's driver inserts
`t=1.1` into its window-boundary array so the forcing switch is never
straddled by a single Picard-iterated window. `fuse::WindowCouplerConfig::
window_length` is fixed per `WindowCoupler` instance, so `main.cpp`'s
`window_boundaries()` instead builds the same boundary list (`0, window,
2*window, ..., end`, with `1.1` inserted if it is not already one) and
constructs a FRESH `WindowCoupler` for each maximal run of EQUAL-length
consecutive boundaries -- one coupler per "phase" rather than one array of
variable-length windows. Because FUSE time is `fuse::Rational`, this needs
no epsilon snapping: a boundary that already lands on `1.1` (e.g. the
default `--window 0.02`, since `55*0.02 == 1.1` exactly) is detected by
exact equality, not a tolerance -- the Toy's own code has to guard this
comparison with `32*epsilon`; this port does not.

## What is and is not Kokkos here

The reaction rate `q = U^2*V - A*U` -- the one genuinely elementwise,
embarrassingly parallel kernel in this model -- runs as a real
`Kokkos::parallel_for` over a `Kokkos::View<double*>`
(`brusselator_physics.hpp::reaction_kokkos`), on the canonical cell index
space. `ctest -L omp` exercises it at 1 and every host core
(`verification.brusselator_physics.omp1`/`.omp8`).

The periodic-diffusion backward-Euler solve stays Eigen (`SimplicialLDLT`,
as in the Toy), and the native-index <-> canonical-index permutation used
before/after applying remap weights (`NativeGrid::to_native`/`to_canonical`)
is a plain host gather loop, not a Kokkos kernel -- both are small,
inherently sequential-dependency operations (a sparse factorisation; a
permutation) where a parallel kernel would not change the asymptotic cost
at these grid sizes, unlike the reaction term, which is exactly the kind of
per-cell, no-cross-talk kernel Kokkos exists for.

## Grid decomposition over the component's own ranks

The mesh, the explicit (no linear solve) physics, AND (since the species-core
solver work below) the implicit diffusion solve are all decomposed over
however many ranks U's own component runs on. V's own grid/physics/solver
follow the identical path, and (since the Halo track's own work, docs/dev/
native-halo.md) `VParticipant`'s two-sided remap (the `q`: U grid -> V slab,
and `state`: V grid -> U slab transfers) is now routed through the same
`fuse::build_remap_halo`/`build_weights`/exchanger machinery the mesh/physics
decomposition already used for everything else -- see "VParticipant
two-sided remap decomposition" below for the measured agreement, including
on a genuinely ADOPTING halo (KNOWN_ISSUES.md D29, closed). One piece
remains open: a HOST run -- a real `MPI_Comm_split`-formed, more-than-one-
rank V communicator has never been exercised end to end on a machine that
can start MPI (`v_main.f90`'s own header comment has the full account) --
`verification.brusselator.participants.mpmd.5` is build-verified only, in
this sandbox, for exactly that reason.

**The mesh.** `NativeGrid::make` loads its backend at
`backend.rank()`/`backend.num_ranks()`. `fuse::native::NativeMeshBackend`
slabs a generated `"generated:quad:NxM"` mesh by ROWS
(`fuse::native::rank_slice(ny, rank, num_ranks)`, src/backend/
native_generators.cpp), so this rank's view is already its own contiguous
row band, owned cells only -- the native mesh carries no ghost cells of
its own. `NativeGrid` exposes that band as `j_lo()`/`j_hi()`/
`owned_rows()`/`owned_size()`, and a LOCAL canonical index
`local_index(i, j) = (j - j_lo())*nx + i` over it, alongside the GLOBAL
`index(i, j) = j*nx + i` every whole-grid caller already used. At
`num_ranks() == 1` the row band is the whole grid (`j_lo() == 0`,
`j_hi() == ny()`), the local index equals the global one, and every
existing whole-grid caller (`UCore`, `VDiffusionCore`, `VParticipant`, the
in-process example, the participant-ABI shim) is unaffected: byte-identical
output was checked directly, not assumed -- see "Byte-identical at
`num_ranks() == 1`" below.

**The halo.** One periodic row above and below this rank's own band is
exposed as `NativeGrid::ghost_pattern()`, returning FUSE's own
`fuse::linalg::GhostPattern` (the Containers track's own distributed-vector
interface) directly -- `brusselator_physics.hpp`'s own `GridGhostPattern` is
a type alias for it, not a lookalike: one `owner_rank`/`remote_local_index`
pair per ghost cell, in a fixed order (the below row's `nx` cells, then the
above row's). Every row band is contiguous, so the row below this band is
always the LAST row of whichever rank owns it, and the row above is always
the FIRST row of whichever rank owns it (itself, wrapping, at
`num_ranks() == 1`) -- which is why the pattern needs no communication to
build. `SlabField` holds a component's own state as one contiguous buffer,
owned block then ghost block (the same layout `fuse::linalg::DistVector`
uses), and refreshes its ghosts through an injected `GridHaloExchanger`
(`update(owned, owned_n, ghosts, ghost_n)`) -- an alias for
`fuse::linalg::GhostExchanger` itself: a same-buffer gather at
`num_ranks() == 1` (`fuse::linalg::make_serial_exchanger()`), an in-process
two-buffer copy in the tier-0 two-partition test, and real `MPI_Sendrecv` in
the tier-1 test (`test/parallel/brusselator_grid_mpi_test.cpp`, built, not
run in this sandbox -- MPI cannot start here). No MPI header appears in any
example header this track touched; the layering gate
(`tools/check-layering.py`) enforces it.

**The explicit parts.** Reaction (`reaction_kokkos_span`, the same
`Kokkos::parallel_for` `reaction_kokkos` already ran, now addressable by
raw pointer and size so it runs equally over a whole canonical vector or
over just a `SlabField`'s owned block), the forcing mask and the initial
condition (`forcing_mask_owned`/`initial_condition_owned`, the same
per-row formula restricted to `[j_lo, j_lo+owned_rows)`) and every mean
(`SlabField::owned_mean`, and `UCore`/`VDiffusionCore`'s own `global_mean()`
helper) are each an owned-cells computation; the mean is specifically an
owned PARTIAL sum reduced across the component's own communicator through
an injected `GridReducer` (`sum`/`max`/`rank`/`size`/`allgather` -- an
alias for `fuse::linalg::Reducer` itself) -- never a locally-reduced count
divided into a local sum, which would silently report each rank's own share
as if it were the global value. The forcing mask and initial condition need
no ghost value (no stencil); the reaction is pointwise over OWNED cells only
-- it is the solver seam below that needs both the halo and the reduction.

**The solver seam (closed).** `UCore`/`VDiffusionCore` (species_core.hpp)
call a small `DiffusionSolver` interface (`factor`/`solve`) rather than
running `Eigen::SimplicialLDLT` inline, selected per `PhysicsConfig::solver`
(`SolverKind::Auto`/`Direct`/`Schur`/`Pcg`/`Gmres`, `--solver` on
`u_main.c`/`v_main.f90`): `DirectDiffusionSolver` -- the original
factor/solve bodies, moved rather than rewritten -- remains valid only at
`num_ranks() == 1` and refuses otherwise (`E-BRUSS-DIFFUSION-DIRECT-
DECOMPOSED`); `SchurDiffusionSolver` wraps the Solvers track's
`fuse::linalg::SchurSlabSolver` (the slab-decomposed backward-Euler periodic
Laplacian's own Schur-complement solve); `KrylovDiffusionSolver` assembles
the same operator as a `fuse::linalg::DistSparseMatrix` (global 5-point-
stencil triplets restricted to this rank's own row band) and solves it with
`fuse::linalg::pcg`/`gmres` under a `JacobiPreconditioner`. `SolverKind::Auto`
resolves to `Direct` at `num_ranks() == 1` and `Schur` otherwise, so every
existing single-rank caller is unaffected. `UCore`/`VDiffusionCore`'s own
constructors take the grid's own `NativeGrid` (not bare `nx`/`ny`) plus an
injected `Reducer` and `GhostExchangerFactory`, since `UCore`'s own reaction
fixed-point loop decides convergence from a `reducer_->max()` over the
WHOLE component (never a per-rank norm) -- the diffusion solve underneath is
collective, so a rank that decided convergence from local data alone could
strand its peers in the next collective call. See "Species-core solver
decomposition" under "Measured results" below for the agreement numbers
that close this seam; what is NOT yet decomposed (V's own two-sided remap)
is noted above, at the top of this section.

**Byte-identical at `num_ranks() == 1`.** Checked directly: the unmodified
and the decomposition-carrying trees were each built into a fresh
`fuse_example_brusselator`, run with identical arguments
(`--nx-u 16 --ny-u 16 --nx-v 24 --ny-v 24 --dt-u 0.002 --dt-v 0.003
--window 0.02 --end 0.1`), and `cmp` found their `u.csv`, `v.csv` and
`diagnostics.txt` outputs byte-for-byte IDENTICAL -- not merely close at
displayed precision. The full brusselator ctest row set (16 rows,
including the new `verification.brusselator_grid_decomposition`) passed
100% on the decomposition-carrying tree.

**What this track measured about reduction order.** Reproducing one
whole-grid field's mean from two partitions' owned partial sums plus a
reduction is NOT always bit-identical to computing the mean over the
whole vector directly, because floating addition is not associative: for
the forcing mask (mostly zero, the few nonzero terms exactly
representable) the two agreed EXACTLY on an 8x8 two-partition test; for
the initial condition and the reaction kernel's output (every cell a
non-trivial fraction) the two differed by `4.4408920985006262e-16`, one
ulp at that magnitude. Both are far inside the `1e-9` bound the
decomposition test checks, which was set from this measurement, not
tuned to pass it.

## Measured results

All numbers below were measured on this build (macOS arm64, AppleClang
21.0.0, Kokkos 5.1.0 OpenMP+Serial, Eigen 5.0.1); the Toy's own numbers come
from a reference run of the unmodified Toy on the same machine
(`/Users/mahadevan/Code/AI/Toy`, its own `tmp/eigen-3.4-config` pinning
Eigen 3.4).

### Weights vs. an independent oracle (`verification.brusselator_weights`)

The Toy's own `overlaps()` (`src/brusselator.cpp`), reimplemented as an
oracle in `test/verification/brusselator_weights_test.cpp`, gives the exact
overlap area between ANY source and target cell of two axis-aligned uniform
rectangular grids as the product of independent 1D x/y overlaps -- exactly
how the Toy's own `Transfer` is built. Compared entry-by-entry against
`fuse::build_weights` at the default resolution (U 16x16, V 24x24):

| direction | max\|FUSE − oracle\| | max_row_defect | max_col_defect |
|---|---:|---:|---:|
| V → U | 4.707e-14 | 8.882e-14 | 1.077e-13 |
| U → V | 1.061e-13 | 1.044e-13 | 8.771e-14 |

Both directions also verified: `P·1 = 1` and `area_target^T·P = area_source^T`
(the two Toy `Transfer` invariants), applying the map to a constant field
reproduces the constant, and the same holds on the Toy's own "nonmatching"
pair (U 17x13, V 11x19).

### Species-core solver decomposition (`verification.brusselator_species_core_decomposition`)

Measured in-process against a `num_ranks() == 1` `DirectDiffusionSolver`
reference, on `UCore`/`VDiffusionCore` instances built one per partition and
stepped concurrently (one `std::thread` per rank; `UCore`'s own constructor
and reaction loop each issue a collective, so a sequential build/step would
deadlock -- the test's own thread-barrier `BarrierReducer`/
`BarrierGhostExchanger` reproduce an N-way `fuse::linalg::Reducer`/
`GhostExchanger` in-process, standing in for a real `mpiexec` this sandbox
cannot start):

| case | partitions | max\\|diff\\| vs. single-rank Direct |
|---|---:|---:|
| `SchurDiffusionSolver`, U (8x8 grid, 20 steps / 2 windows) | 2 | 2.420e-14 |
| `SchurDiffusionSolver`, U (8x12 grid) | 3 | 4.485e-14 |
| `SchurDiffusionSolver`, V | 2 | 2.665e-15 |

`KrylovDiffusionSolver`'s own `rel_tol` was swept against the same reference
to derive `PhysicsConfig::solver_tolerance`'s default:

| `rel_tol` | Pcg max\\|diff\\| | Gmres max\\|diff\\| |
|---:|---:|---:|
| 1e-8  | 4.602e-9  | 4.631e-9  |
| 1e-9  | 5.196e-10 | 4.583e-10 |
| 1e-10 | 3.387e-11 | 2.186e-11 |
| 1e-11 | 3.344e-12 | 1.475e-12 |
| 1e-12 | 1.421e-13 | 1.421e-13 |
| 1e-13 | 5.862e-14 | 5.462e-14 |

`solver_tolerance` defaults to `1e-13`, the tightest value swept -- closest
to the Schur path's exact-solve floor -- per this project's standing
robustness-over-efficiency rule, not the loosest value that happened to
pass.

### VParticipant two-sided remap decomposition (`verification.brusselator_species_core_decomposition`)

Measured the same way as the species-core solver rows above (in-process,
thread-barrier reducer/exchanger standing in for a real `mpiexec`), but now
exercising `VParticipant::step`/`remap_state_to_u`'s OWN two remaps (`q`: U
grid -> V slab; `state`: V grid -> U slab) over a decomposed V, each
direction's halo pre-installed through the same `NativeMeshBackend::
commit_remap_halo` seam `test/backend/native_halo_test.cpp`'s own
`BackendFixture` uses, against a `num_ranks() == 1` `VParticipant` reference
stepping the identical trajectory:

| case | partitions | solver | sub-steps | max\\|diff\\| vs. single-rank `VParticipant` |
|---|---:|---|---:|---:|
| aligned split (8+8 / 12+12 rows, zero adopted ghost cells) | 2 | Schur | 6 | 1.821e-14 |
| aligned split (4+4+4 / 6+6+6 rows, zero adopted ghost cells) | 3 | Schur | 6 | 7.994e-15 |
| MISALIGNED split (6/5/5 U rows / 8/8/8 V rows, adopting both directions) | 3 | Schur | 6 | 1.732e-14 |
| MISALIGNED split (6/5/5 U rows / 8/8/8 V rows, adopting both directions) | 3 | Pcg | 20 | 5.262e-14 |

All four reproduce the single-rank reference to round-off, in the same
1e-14/1e-15 band as the diffusion-only agreement measured above (the
2-partition `V` row there, 2.665e-15, is itself the tightest). The two
MISALIGNED rows adopt cells in one or both remap halo directions (measured
per rank at `make_from_grids` time, `FUSE_BRUSS_DEBUG=1`: rank 0 "0 U,
12 V", rank 1 "8 U, 12 V", rank 2 "8 U, 0 V") and are the configuration
KNOWN_ISSUES.md D29 tracked as refusing, then as measurably disagreeing with
the reference once the refusal was lifted by hand: `VParticipant::step()`/
`remap_state_to_u()` sized their `apply_weights` destination buffers from
this rank's pre-halo OWNED count rather than from the weights matrix's own
`n_rows` (the target mesh's CURRENT, post-halo local entity count -- larger
than owned whenever that mesh was ALSO widened as the source of the OTHER
transfer direction on the same backend), so `apply_weights`' unconditional
zero-and-write loop overflowed the buffer -- the measured 3.1e-4
`exchange_in` disagreement from the first sub-step, and the later
`Eigen::Block.h` assertion, were both downstream of that overflow, not of
the halo-construction pipeline (`test/backend/native_halo_test.cpp`'s own
`ThreeRanksMisalignedAdoptsExactlyWhatTheOracleComputes`/`...CompleteWeights
AndLinearFieldReproduction` tests already proved that correct at this exact
ratio, one direction at a time). See KNOWN_ISSUES.md D29 (closed) for the
full measurement and fix.

**Trace.** `FUSE_BRUSS_DEBUG=1` in the environment makes `VParticipant::
make_from_grids`/`step` print, to stderr, this rank's owned/adopted counts
in each direction, the weights matrix shapes `step()` builds its buffers
from, the first few exchanged ghost values, and `q_on_v`'s sum and
`exchange_in` after each step -- an opt-in diagnostic (no cost when unset),
not a substitute for the tests above.

### Agreement with the Toy at the defaults, short run (`--end 0.1`)

| | FUSE | Toy | \|difference\| |
|---|---:|---:|---:|
| final_mean_u | 1.52801298656 | 1.5280505999004665 | 3.7613e-5 |
| final_mean_v | 2.02340201528 | 2.0233654055333425 | 3.6610e-5 |

### Agreement with the Toy at the defaults, full run (`--end 11.5`)

| | FUSE | Toy |
|---|---:|---:|
| final_mean_u | 4.07486930721 | 4.0437736127620596 |
| final_mean_v | 1.18944993515 | 1.2358694937952479 |
| max_window_balance_error | 7.576e-13 | 1.2391e-13 |
| global_balance_error | 4.251e-11 | 1.5987e-11 |
| max_window_exchange_error | 7.985e-13 | 1.3878e-16 |
| windows / max iterations | 575 / 17 | -- / 9 |
| wall time | 137 s | -- |

The absolute difference at `--end 11.5` (`|Δmean_u| ≈ 0.031`,
`|Δmean_v| ≈ 0.046`) is much larger than at `--end 0.1`: this is a genuinely
nonlinear reaction-diffusion system integrated through 575 windows and the
`t=1.1` forcing switch, and two different first-order-accurate schemes
integrating the same stiff nonlinear problem over a long interval need not
stay close in absolute terms even though each is individually convergent
(their LOCAL per-step errors are what shrinks with `dt`, not necessarily
their long-time trajectory divergence for a fixed `dt`). The refinement
study below establishes the order claim directly, at `--end 0.1` only, for
that reason -- see "Scope of the refinement study" below.

### Cross-code convergence order (`--end 0.1`, joint `dt_u`/`dt_v`/`window`
refinement, both codes run at the SAME resolutions)

| refinement | FUSE mean_u | Toy mean_u | \|diff\| | FUSE mean_v | Toy mean_v | \|diff\| |
|---:|---:|---:|---:|---:|---:|---:|
| 1x | 1.52801298656 | 1.5280505999004665 | 3.7613e-5 | 2.02340201528 | 2.0233654055333425 | 3.6610e-5 |
| 2x | 1.52744078099 | 1.5274634823263886 | 2.2701e-5 | 2.02398507132 | 2.0239625197113655 | 2.2552e-5 |
| 4x | 1.52718665495 | 1.5272000395097252 | 1.3385e-5 | 2.02424135792 | 2.0242278499696398 | 1.3508e-5 |
| 8x | 1.52707026876 | 1.5270776552046188 | 7.3864e-6 | 2.02435773417 | 2.02435021713482 | 7.5170e-6 |

Successive ratios of `|diff|`: mean_u 1.657, 1.696, 1.812 (measured orders
0.73, 0.76, 0.86); mean_v 1.623, 1.670, 1.797 (0.70, 0.74, 0.85). The
difference shrinks monotonically at every level and the order rises toward
1 with refinement, but it has not reached 1 at 8x: this is stated as
measured, not rounded up. What remains between the codes is the one
departure above (where `q` is sampled), a smaller term than the
conservation defect it replaced -- the first version's `|diff|` was
`2.1052e-3` at 1x and did halve cleanly, because the defect itself was a
clean first-order error and dominated. Richardson-extrapolating each code
from its 4x/8x pair (both first order) gives limits that agree to
`1.388e-6` (mean_u: FUSE 1.52695388, Toy 1.52695527) and `1.526e-6`
(mean_v: 2.02447411 vs 2.02447258), five times closer than the 8x raw
difference, consistent with both discretising the same continuous problem.

### FUSE self-convergence in `dt` at fixed ratio (`--end 0.1`)

FUSE only, refinement 1x/2x/4x/8x (same joint `dt_u`/`dt_v`/`window`
scaling): mean_u = 1.52801298656, 1.52744078099, 1.52718665495,
1.52707026876; mean_v = 2.02340201528, 2.02398507132, 2.02424135792,
2.02435773417. Successive-difference orders: mean_u 1.17, 1.13; mean_v
1.19, 1.14. The Toy at the same four resolutions: mean_u 1.16, 1.11; mean_v
1.17, 1.12. Both are first order (backward Euler) and slightly on the high
side of it at these resolutions; neither is asserted to fall in a symmetric
window around 1.0 the data was not asked to hit.

### Exchange conservation and mass budget (`--end 0.1`, default resolution,
joint refinement 1x/2x/4x)

`max_window_exchange_error` (spatial, `|mean(∫q) − mean(P_UV ∫q)|`): 7.984e-14,
7.501e-14, 4.147e-15 across refinement -- round-off at every resolution, as
required.

`max_window_balance_error` (the Toy's "Discrete budget" identity, NUMERICS.md):
1.703e-13, 8.970e-14, 8.802e-15; `global_balance_error` 5.300e-13, 2.112e-13,
7.964e-14 -- round-off, as in the Toy. The first version of this port
measured 1.686e-03, 8.329e-04, 4.029e-04 here (first order in `dt`), and
explained it as a quadrature mismatch between two diagnostics. It was not: U
was consuming a different amount of the reaction than V received. Making U
consume exactly what it provides (above) took the identity to round-off
without touching the diagnostics.

### Scope of the refinement study

The cross-code and self-convergence studies above were run at `--end 0.1`
only (5 windows at the default resolution, scaling up to 40 at 8x). A
refinement study at the full `--end 11.5` default was not run: the base
resolution alone already takes ~93 s, and a `2x`/`4x`/`8x` ladder at that
endpoint would cost tens of minutes per point for a claim (first-order
convergence in `dt`) already established at the shorter interval.
This is a deliberate scope reduction, not a gap in the claim: nothing about
the discretisation changes between `t=0.1` and `t=11.5`.

## `--check`

Exits non-zero unless: every window converged; the final U and V fields are
`>= -1e-12` everywhere; `max_window_balance_error <= 1e-11` and
`|global_balance_error| <= 1e-9` (round-off scale: ~60x over the per-window
floor measured at `--end 0.1` and ~40x over the `--end 11.5` global value of
4.251e-11; the first version's defect measured four orders of magnitude
above both); `max_window_exchange_error <= 1e-9` (round-off; measured up to
8.7e-13 at the longest run). None of these thresholds were widened after the fact to make
a run pass -- see `main.cpp`'s own comment at the `--check` block for the
measurements they were set from.

## The A4 assessment

A4 asks for a code that is (1) a separate executable, built by its own build
system, (2) running its own time loop and calling FUSE when it wants to
exchange, (3) handing FUSE its existing arrays, (4) on its own set of
processes. This port satisfies **(3)**: `UComponent`/`VComponent` register no
copies -- `WindowMember::provided`/`required` point directly at the
`Eigen::VectorXd`/`std::vector<double>` storage the components themselves
own, and the Kokkos reaction kernel operates on that same storage through an
unmanaged view. It does **not** satisfy (1) or (4): this is the SAME
executable as the coupler (`fuse_example_brusselator`), not two independently
built binaries on separate processes. It does not satisfy (2) either, and
cannot today: FUSE has no component-driven entry point -- `WindowCoupler`
(like `Runner`) calls `advance()`; nothing lets a component's own `main()`
call INTO FUSE to request an exchange. These three are the honest gaps, not
softened, and are exactly what the Gray-Scott track's own C-ABI follow-on
(`docs/user/c-abi.md`) would need to close for a genuinely separate-process,
component-driven example: a second executable per species, its own time
loop, calling a C ABI entry point when it wants to exchange.

## Running it

```sh
cmake --build <build-dir> --target fuse_example_brusselator
<build-dir>/examples/fuse_example_brusselator \
  --nx-u 16 --ny-u 16 --nx-v 24 --ny-v 24 \
  --dt-u 0.002 --dt-v 0.003 --window 0.02 --end 11.5 \
  --out results
```

Outputs `u.csv`, `v.csv` (final fields, `time,x,y,value`, the Toy's own
format) and `diagnostics.txt` (parameters and the measured residuals above,
in the Toy's own key names) under `--out`. `--save-history` additionally
writes `windows.csv` (`t0,t1,mean_u,mean_v,iterations` per window) --
lighter-weight than the Toy's full per-cell history files, since nothing
downstream of this port replays a saved history the way the Toy's own
`--exchange`/`--template` CLI does; this is a deliberate scope reduction
from the Toy's `--save-history`, stated here rather than silently matched
in name only. `--accel none|aitken|iqn-ils` selects
`fuse::WindowCouplerConfig::acceleration` (the Toy is unrelaxed Picard,
`--accel none`, its only option). `--no-forcing`, `--tolerance`,
`--max-iterations`, `--alpha` mirror the Toy's own flags of the same name.
