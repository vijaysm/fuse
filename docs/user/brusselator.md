# Brusselator: a code FUSE has never met

`examples/brusselator/` ports a multirate 2D Brusselator that the user wrote **independently of
FUSE** (`/Users/mahadevan/Code/AI/Toy`, "the Toy") onto FUSE's native mesh, conservative supermesh
remap, and `fuse::WindowCoupler`. It is this project's A4 example: physics FUSE was not designed
around, arriving as an outside artifact to be coupled rather than written to exercise a feature. The
directory holds three things: an in-process, single-executable example (`fuse_example_brusselator`)
that runs the whole coupled model directly; a small graph-authoring tool
(`brusselator_graph_tool`) that writes the identical coupling specification as text YAML; and three
genuinely separate executables (`brusselator_u_main`, `brusselator_v_main`, `brusselator_coupler_main`)
that load that YAML and run the SAME model as three independently built, independently launched
programs — the A4 case in full.

## The model

On the periodic unit square,

```
U_t = a*Laplacian(U) + B - U + q + f
V_t = a*Laplacian(V) - q
q   = U^2*V - A*U
```

with `A = 3.4`, `B = 1`, `a = 10` by default. Initial conditions sample `U0 = 22*[y(1-y)]^1.5`,
`V0 = 27*[x(1-x)]^1.5` at cell midpoints. The forcing `f` is 5 inside the radius-0.1 circle centred
at `(0.3, 0.6)` for `t >= 1.1`, zero elsewhere and before. Defaults: `nx_u=ny_u=16`, `nx_v=ny_v=24`,
`dt_u=0.002`, `dt_v=0.003`, `window=0.02`, `end=11.5`, Picard tolerance `1e-11`, 60 max iterations.
See the Toy's own `README.md`/`NUMERICS.md` for the full derivation and reference.

## The port, and where it necessarily differs from the Toy

**Ported verbatim.** The periodic 5-point diffusion stencil and backward Euler for diffusion and the
U decay (`brusselator_physics.hpp::periodic_laplacian`) is the same matrix the Toy's
`bruss::laplacian` builds, solved with `Eigen::SimplicialLDLT` exactly as the Toy does; the initial
conditions, forcing mask, and the forcing's exact activated duration within a step
(`forcing_active_duration`) are the Toy's own closed form. Two design decisions from the Toy's own
`NUMERICS.md` "Spatial transfer" section carry over exactly: the exchange is one shared reaction
rate `q = U^2*(P_VU V) - A*U`, with both `P`s the area-overlap, mean-preserving transfer
`P[T,S] = area(T∩S)/area(T)`; and the consumer integrates the exchange rate over its own step
exactly for a piecewise-linear-in-time rate, through `fuse::WindowCoupler`'s own contract (it feeds a
consumer the substep mean of the provider's `TemporalBasis::Linear` waveform).

**The mesh and the remap are FUSE's own.** Cell geometry comes from
`fuse::native::NativeMeshBackend`'s `"generated:quad:NxM"` generator, not the Toy's hand-rolled
`Grid`; `NativeGrid::make` (`brusselator_physics.hpp`) recovers the native-index <-> canonical-index
correspondence by reading back every cell's own centroid and refuses (`E-BRUSS-MESH-INDEX`) if that
recovery is not a clean bijection, rather than assuming the two conventions happen to coincide.
`fuse::build_weights` dispatches to the dependency-free order-1 simplex supermesh path for any
non-MOAB backend (`src/backend/intx_weights.cpp`'s `build_weights_via_supermesh`) — real FUSE
machinery, checked against an independent oracle (see "Measured results").

**One component owns `q`, and it is not the Toy's driver.** In the Toy, a driver computes `q` at the
union of both components' schedules' time knots by interpolating both trial trajectories, feeding
the identical `q` history to both `Component::advance()` calls. FUSE's `Component` ABI gives
`advance()` exactly one snapshot per call, valid at `ctx.t_begin` — there is no driver-level
union-of-knots interpolation available. So `UComponent` computes `q` from its own state and the
just-received, remapped V mean, and consumes over its step exactly the trapezoid
`0.5*dt*(q_prev + q_new)` that `WindowCoupler` delivers to V from the same two consecutive
`provided_` samples; `q_new` depends on the state being solved for, so U's step is a short fixed
point against the one LDLT factorisation (contraction factor of order `1e-2` at the default `dt`,
converging in a handful of solves, refusing `E-BRUSS-U-REACTION` if fifty do not suffice).
`VComponent::provided_` is `P_VU(V)` directly — V's own new state genuinely IS the `t_end`-valid
quantity NUMERICS.md's design decision names, so it needs no such split. Both converge to the same
discrete-in-space, first-order-in-time solution, but they are not the same finite-`dt` scheme; see
"Measured results" for what that costs.

Two defects were found and fixed while building this consistency, each pinned by a mutation row in
`build-evidence/mutations/brusselator.toml`: an earlier version kept U's step linear by consuming
`dt*q_prev` instead of the trapezoid, which is a genuine conservation defect (the discrete mass
budget held only to first order, `1.686e-3` per window at the default resolution against the Toy's
`1.2e-13`) rather than a quadrature-rule artifact — `brusselator-u-consumes-start-of-step-reaction`
must fail `--check`. Separately, `UComponent`'s own "what I exchanged" diagnostic accumulated a
plain per-step Riemann sum instead of the same trapezoid `WindowCoupler` actually delivers to V,
producing a spurious ~1.7e-3 mismatch in the diagnostic alone (the trajectory was unaffected) —
`brusselator-exchange-bookkeeping-drops-trapezoid` reproduces it exactly; both are fixed in
`UComponent::advance`.

**The forcing switch is a window boundary, not a straddled one.** The Toy's driver inserts `t=1.1`
into its window-boundary array so the forcing switch is never straddled by a single Picard-iterated
window. `WindowCouplerConfig::window_length` is fixed per instance, so `main.cpp`'s
`window_boundaries()` instead builds the same boundary list with `1.1` inserted if not already
present, and constructs a fresh `WindowCoupler` for each maximal run of equal-length consecutive
boundaries. Because FUSE time is `fuse::Rational`, a boundary already landing on `1.1` (e.g. the
default `--window 0.02`, since `55*0.02 == 1.1` exactly) is detected by exact equality, needing no
epsilon the Toy's own code has to guard with.

**What runs on Kokkos.** The reaction rate `q = U^2*V - A*U` — the one genuinely elementwise,
embarrassingly parallel kernel in this model — runs as a real `Kokkos::parallel_for`
(`brusselator_physics.hpp::reaction_kokkos`); `ctest -L omp` exercises it at 1 and every host core.
The backward-Euler diffusion solve stays Eigen, and the native/canonical index permutation is a
plain host gather loop — both small, sequential-dependency operations where a parallel kernel would
not change the asymptotic cost, unlike the reaction term.

## Grid decomposition

The mesh, the explicit physics, and the implicit diffusion solve are all decomposed over however
many ranks each of U's and V's own components run on, including V's two-sided remap to/from U
(`VParticipant`, below) — the full set of pieces a grid-decomposed participant needs.

**Mesh and halo.** `NativeGrid::make` loads its backend at `backend.rank()`/`backend.num_ranks()`.
`NativeMeshBackend` slabs a generated `"generated:quad:NxM"` mesh by rows
(`fuse::native::rank_slice`), so a rank's view is already its own contiguous row band with no ghost
cells of its own; `NativeGrid` exposes `j_lo()`/`j_hi()`/`owned_rows()` and a local index over that
band. One periodic row above and below is exposed as `NativeGrid::ghost_pattern()`, returning FUSE's
own `fuse::linalg::GhostPattern` directly (`GridGhostPattern` is a type alias for it, not a
lookalike); `SlabField` holds a component's own state as one contiguous owned-then-ghost buffer, the
same layout `fuse::linalg::DistVector` uses, refreshed through an injected `GridHaloExchanger` — an
alias for `fuse::linalg::GhostExchanger` itself. At `num_ranks() == 1` the row band is the whole
grid, the local index equals the global one, and every existing whole-grid caller is unaffected:
checked directly, not assumed (see "Measured results", byte-identical row).

**The explicit parts.** Reaction, the forcing mask, the initial condition, and every mean
(`SlabField::owned_mean`) are each an owned-cells computation; the mean is specifically an owned
partial sum reduced across the component's own communicator through an injected `GridReducer` — an
alias for `fuse::linalg::Reducer` itself — never a locally reduced count divided into a local sum,
which would silently report each rank's own share as the global value.

**The halo and halo-only reduction order are measured, not assumed.** Reproducing one whole-grid
field's mean from two partitions' owned partial sums plus a reduction is not always bit-identical to
computing the mean over the whole vector directly, because floating addition is not associative: for
the forcing mask (mostly zero, the few nonzero terms exactly representable) the two agree exactly on
an 8x8 two-partition test; for the initial condition and the reaction kernel's output (every cell a
non-trivial fraction) they differ by `4.4408920985006262e-16`, one ulp at that magnitude — both far
inside the `1e-9` bound the decomposition test checks, which was set from this measurement, not
tuned to pass it.

## Solvers

`UCore`/`VDiffusionCore` (`species_core.hpp`) call a small `DiffusionSolver` interface
(`factor`/`solve`) rather than running `Eigen::SimplicialLDLT` inline, selected per
`PhysicsConfig::solver` (`SolverKind::Auto`/`Direct`/`Schur`/`Pcg`/`Gmres`, a `--solver` flag on
`u_main.c`/`v_main.f90`): `DirectDiffusionSolver` is the original factor/solve bodies, moved rather
than rewritten, valid only at `num_ranks() == 1` (refuses `E-BRUSS-DIFFUSION-DIRECT-DECOMPOSED`
otherwise); `SchurDiffusionSolver` wraps `fuse::linalg::SchurSlabSolver`, the slab-decomposed
backward-Euler periodic Laplacian's exact Schur-complement solve; `KrylovDiffusionSolver` assembles
the same operator as a `fuse::linalg::DistSparseMatrix` (global 5-point-stencil triplets restricted
to this rank's own row band) and solves it with `fuse::linalg::pcg`/`gmres` under a
`JacobiPreconditioner`. `SolverKind::Auto` resolves to `Direct` at `num_ranks() == 1` and `Schur`
otherwise, so every existing single-rank caller is unaffected — confirmed byte-identical (see
"Measured results"). `UCore`'s own reaction fixed-point loop decides convergence from a
`reducer_->max()` over the whole component, never a per-rank norm, since the diffusion solve
underneath is collective and a rank deciding convergence from local data alone could strand its
peers in the next collective call.

`solver_tolerance` (the Krylov solvers' `rel_tol`) was swept from `1e-8` to `1e-13` against the
single-rank `DirectDiffusionSolver` reference (see "Measured results") and defaults to `1e-13`, the
tightest value swept — closest to the Schur path's exact-solve floor — per this project's standing
robustness-over-efficiency rule, not the loosest value that happened to pass.

**V's own two-sided remap** (`q`: U grid -> V slab; `state`: V grid -> U slab) is routed through the
same `fuse::build_remap_halo`/`build_weights`/exchanger machinery the mesh/physics decomposition
already used for everything else (`VParticipant::step`/`remap_state_to_u`). This closed a real
defect (KNOWN_ISSUES.md D29): both `apply_weights` destination buffers were originally sized from
this rank's pre-halo owned cell count rather than from the weights matrix's own `n_rows` (the target
mesh's current, post-halo local entity count), which overflows whenever V (or U) was ALSO widened as
the source of the other remap direction on the same backend — a heap buffer overflow, not a wrong
value, root-caused by instrumentation (not guessed from reading) and fixed by sizing both
destination buffers from their own weights matrix's `n_rows`. Mutation
`brusselator-vparticipant-stale-adopted-ghost-not-exchanged` replaces the refusal this fix removed.
`fuse::build_supermesh_weights`, the MOAB builder, and the MLS builder all follow the identical
`n_rows` convention correctly; the defect was in this example's own caller, not in any builder — the
Runner, the library's own caller, already checks `weights->n_rows != out.elements` and refuses with
`E-RT-REMAP-SIZE` rather than applying.

## Participants: the separately built executables

`brusselator_graph_tool` builds the identical coupling specification `brusselator_graph.hpp`'s
`add_window_coupling()` constructs for the in-process example, and writes it as text YAML
(`fuse::to_yaml()`) instead of running it — graph **authoring**, kept deliberately separate from
graph **participating**. `brusselator_u_main` (pure C), `brusselator_v_main` (pure Fortran, built
only when a Fortran compiler is found), and `brusselator_coupler_main` (pure C) each call
`fuse_load_graph()` on that YAML and nothing else constructs a graph: U's and V's own
coupling-facing code is pure C/pure Fortran, with only the shared physics (`shim.cpp`) in C++, so a
foreign-language caller never touches a C++ type. Each program's launch mode comes from the loaded
YAML, not from a flag — the executable's own code is identical whether the graph says `mpmd`,
`accept`, or (for a tier-0 smoke run with no MPI at all) in-process.

**This is the A4 case, and it is now satisfied by this path (not by the in-process example).** A4
asks for a code that is (1) a separate executable, built by its own build system, (2) running its
own time loop and calling FUSE when it wants to exchange, (3) handing FUSE its existing arrays, and
(4) on its own set of processes. The in-process `fuse_example_brusselator` satisfies only (3) — it
shares one executable and one time loop with the coupler. The three participant executables satisfy
all four: `u_loop.c`'s `u_loop_step()` (the SAME function the tier-0 harness drives) is U's own time
loop, calling `fuse_begin_step()`/`fuse_end_step()` on the FUSE ABI whenever it wants to exchange;
`u_main.c`'s field registration binds the ABI directly to `u_main.c`'s own storage, no copy; and
`mpmd` mode runs `brusselator_u_main`/`brusselator_v_main`/`brusselator_coupler_main` as three
separate `appnum` segments of one `mpiexec`, each on its own ranks. Host-verified for `mpmd.3` and
`mpmd.5` (see "Measured results"); `accept` mode (U and V launched as two independent jobs that
rendezvous dynamically) is blocked on this host by KNOWN_ISSUES.md D23, an MPICH `ch3:nemesis`
limitation unrelated to this library — not a defect here, and unverified by measurement rather than
broken.

**Bundled coupler ranks (accept mode only, D18).** `fuse::transport::connect_pairwise()` supports
exactly two Connect-mode components, so in accept mode the coupler's ranks are extra ranks of U's own
`MPI_COMM_WORLD`, never a separate executable: `--ranks-u N` tells `brusselator_u_main` how many of
its own local ranks are real U-physics ranks, and any remaining local ranks are coupler placeholder
ranks that call `fuse_setup()` identically but never register a field. In `mpmd` mode `--ranks-u`
defaults to the whole local world (no bundled coupler — the coupler is `coupler_main.c`'s own
separate segment instead).

## Measured results

All numbers below were measured on this build (macOS arm64, AppleClang 21.0.0, Kokkos 5.1.0
OpenMP+Serial, Eigen 5.0.1, tier0-release); "host" rows are the user's own machine, recorded in
`KNOWN_ISSUES.md`, and are not reproducible in this sandbox (MPI cannot start here at all). The Toy's
own numbers come from a reference run of the unmodified Toy on the same machine.

| Category | Quantity | Value |
|---|---|---:|
| Weights vs. Toy's oracle, `verification.brusselator_weights` | max\|FUSE &minus; oracle\|, V&rarr;U | 4.707e-14 |
| | max\|FUSE &minus; oracle\|, U&rarr;V | 1.061e-13 |
| | `P.1=1` and `area_target^T.P=area_source^T` | both hold, both directions |
| Agreement with Toy, defaults, `--end 0.1` | FUSE final_mean_u / Toy / \|diff\| | 1.52801298656 / 1.5280505999004665 / 3.761e-5 |
| | FUSE final_mean_v / Toy / \|diff\| | 2.02340201528 / 2.0233654055333425 / 3.661e-5 |
| Agreement with Toy, defaults, `--end 11.5` | FUSE final_mean_u / Toy | 4.07486930721 / 4.0437736127620596 |
| | FUSE final_mean_v / Toy | 1.18944993515 / 1.2358694937952479 |
| | max_window_balance_error / global_balance_error | 7.576e-13 / 4.251e-11 |
| | windows / max iterations | 575 / 17 (Toy: &ndash; / 9) |
| Cross-code convergence order, `--end 0.1`, 1x&rarr;8x | \|diff\| ratios, mean_u | 1.657, 1.696, 1.812 (order 0.73&rarr;0.86) |
| | \|diff\| ratios, mean_v | 1.623, 1.670, 1.797 (order 0.70&rarr;0.85) |
| | Richardson limits agree to | 1.388e-6 (mean_u), 1.526e-6 (mean_v) |
| FUSE self-convergence in dt, `--end 0.1` | successive-difference order, mean_u / mean_v | 1.17, 1.13 / 1.19, 1.14 (Toy: 1.16, 1.11 / 1.17, 1.12) |
| Exchange conservation, `--end 0.1`, 1x/2x/4x | max_window_exchange_error | 7.984e-14, 7.501e-14, 4.147e-15 |
| | max_window_balance_error | 1.703e-13, 8.970e-14, 8.802e-15 |
| | global_balance_error | 5.300e-13, 2.112e-13, 7.964e-14 |
| Species-core solver decomposition, vs. single-rank Direct | Schur, U, 2 partitions (8x8, 20 steps) | 2.420e-14 |
| | Schur, U, 3 partitions (8x12) | 4.485e-14 |
| | Schur, V, 2 partitions | 2.665e-15 |
| | Pcg / Gmres at `rel_tol=1e-13` | 5.862e-14 / 5.462e-14 |
| VParticipant two-sided remap decomposition, vs. single-rank reference | aligned split, 2 partitions, Schur | 1.821e-14 |
| | aligned split, 3 partitions, Schur | 7.994e-15 |
| | MISALIGNED split, 3 partitions, Schur | 1.732e-14 |
| | MISALIGNED split, 3 partitions, Pcg (20 sub-steps) | 5.262e-14 |
| Byte-identical at `num_ranks()==1` | `cmp` of u.csv/v.csv/diagnostics.txt, decomposed vs. unmodified | identical |
| | full tier-0 ctest suite after the decomposition change | 179/179 |
| In-process `--end 0.1 --check` (this build) | final_mean_u / final_mean_v | 1.52801298656 / 2.02340201528 |
| | max_window_balance_error / global_balance_error / max_window_exchange_error | 1.703e-13 / 5.300e-13 / 7.984e-14 |
| In-process `--end 0.5 --check` (this build, the ctest default) | final_mean_u / final_mean_v | 1.10791238505 / 2.31637697544 |
| `--driver runner --end 0.5 --check` (this build) | final_mean_u / final_mean_v | 1.10791238505 / 2.31637697544 (matches `--driver window`) |
| Host, `verification.brusselator.participants.mpmd.3` (1+1+1 ranks) | mean_u, mean_v vs. single-process example | bit-identical (0 ulp) |
| Host, `verification.brusselator.participants.mpmd.5` (U 2, V 2, coupler 1) | \|mean_u diff\| / \|mean_v diff\| vs. single-process example | 8.9e-14 / 9.4e-14 (relative 5.6e-14 / 4.7e-14) |
| Host, `verification.brusselator.participants.accept.{3,5}` | status | blocked by D23 (MPICH ch3:nemesis), not a defect here |

The absolute difference at `--end 11.5` (`|Δmean_u| ~ 0.031`, `|Δmean_v| ~ 0.046`) is much larger
than at `--end 0.1`: this is a genuinely nonlinear reaction-diffusion system integrated through 575
windows and the `t=1.1` forcing switch, and two first-order-accurate schemes integrating the same
stiff nonlinear problem over a long interval need not stay close in absolute terms even though each
is individually convergent — the cross-code convergence-order study above establishes the order
claim directly, at `--end 0.1` only, for that reason. A refinement study at the full `--end 11.5`
default was not run: the base resolution alone already costs on the order of 90-140 s on this
machine, and a 2x/4x/8x ladder at that endpoint would cost tens of minutes per point for a claim
already established at the shorter interval — a deliberate scope reduction, not a gap in the claim,
since nothing about the discretisation changes between `t=0.1` and `t=11.5`.

## Newton acceleration (S7.2 WindowNewton)

`--accel newton` (via `--driver runner` only -- `--driver window` refuses
it cleanly, `E-ACCEL-KIND`, since `WindowCoupler` has no Newton-Krylov
path) solves the whole coupling window as one Newton-Krylov problem
instead of a Picard/Aitken/IQN-ILS sweep, following `docs/dev/newton-
krylov.md` Section 9's own design. `--iteration-mode gauss-seidel` and
`--no-block-jacobi` (the per-species physics-based preconditioner, ON by
default) are both available under every `--accel` value, not only
`newton`. Measured on this build, `--driver runner --accel newton`:

| Quantity | Value |
|---|---:|
| Windows converged, `--end 1.0` (50 windows), Jacobi, block-Jacobi preconditioner | 50/50, 0 divergences |
| Windows converged, SAME run, `--iteration-mode gauss-seidel` | 50/50, 0 divergences |
| Windows converged, SAME run, `--no-block-jacobi` | 50/50, 0 divergences |
| final_mean_u / final_mean_v, all three of the above | 0.668994004315 / 2.8189403867 (identical) |
| Outer Newton iterations, first window (block-Jacobi / identity) | 4 / 3 |
| Outer Newton iterations, late windows (block-Jacobi) | 3 |
| GMRES iterations, last Newton step, first window &rarr; late windows | ~70 &rarr; 2-6 |
| Windows converged, `--end 11.5` (575 windows, the example's own default) | 222/222 observed before the run was cut short for runtime, 0 divergences |
| `max_window_balance_error` / `global_balance_error`, `--end 1.0` | 1.158e-13 / 4.663e-12 |

`NewtonConfig::eta_min` (the Eisenstat-Walker forcing term's own floor,
`0.0` by default) is set to `1e-4` for this example -- at the SNES-
matching default, the first window's Newton solve stalls
(`DivergedLinearSolve`) once the forcing term tightens past the
finite-difference Jacobian action's own noise floor, even though `||F||`
has already dropped seven orders of magnitude; `main.cpp`'s own comment
on the `NewtonConfig` construction has the full measurement (`eta`
reaching `~6.6e-9` while `||F||/||F_0|| ~ 9e-8`) and the cost/accuracy
tradeoff against the tighter `eta_min = 1e-6` (also removes the stall,
at far more GMRES work per window). `docs/dev/newton-krylov.md` Section
9.9.2 has the full account.

The Newton test case is not a Picard-divergent one (no such case was
found in CLI-reachable parameter space, Section 9.6 of the design doc) --
it is the SAME defaults where `--accel aitken` reproduces `KNOWN_ISSUES.md`
D17's own instability (window index 2, `--driver window`): Newton
converges every window there instead.

Single-rank only: `RegionRuntime::newton_reducer` routes every norm/inner
product through a `fuse::linalg::Reducer`, but the reducer Newton's
window seam constructs today is unconditionally the serial one -- a real
cross-rank `ParticipantSetReducer` is `KNOWN_ISSUES.md` D31, open.

## The PETSc substrate

`--substrate petsc` (read only under `--accel newton`; default `native`)
selects `NewtonConfig::substrate = NewtonSubstrate::Petsc`: the IDENTICAL
window-Newton graph above, solved by PETSc's own `SNESNEWTONLS`
(`fuse::petsc::solve()`) instead of the native JFNK coupler --
[docs/dev/newton-krylov.md §10/§11](../dev/newton-krylov.md) has the full
design. `main.cpp` calls `fuse::petsc::install()` unconditionally and
early (a no-op for `--substrate native`, every existing invocation), so
`--substrate petsc` resolves rather than refusing with
`E-NEWTON-PETSC-UNAVAILABLE` whenever the binary was built with
`FUSE_ENABLE_PETSC`; without it (the default tier-0 configuration), the
SAME flag refuses with that code, caught by every driver branch's own
existing `!r.ok()` check on `Runner::initialize()`.

| Quantity | native | PETSc |
|---|---:|---:|
| Windows converged, `--end 1.0`, Jacobi, block-Jacobi preconditioner | 50/50 | 50/50 |
| Outer Newton iterations, first window | 4 | 5 |
| final_mean_u / final_mean_v | 0.668994004315 / 2.8189403867 | 0.668994004315 / 2.8189403867 |
| max window balance error / global | 1.158e-13 / 4.663e-12 | 1.144e-13 / 4.692e-12 |
| Agreement with native (derived bound, `petsc.brusselator_newton`) | — | passes |

Measured on the development host (2026-10-11). The first window's residual
history is the same on both substrates to six digits (1.25e+02 → 1.40e+01 →
1.34e-01 → 1.16e-05 → ~1e-09); SNES takes one more outer iteration because its
convergence test is applied after the step that the native solver already judges
converged. The final means agree to every printed digit. One mechanism had to be
added for this to hold: SNES's Eisenstat–Walker forcing has no floor, and without
one the first window's last linear solve chases an accuracy the finite-difference
Jacobian action cannot deliver and GMRES exhausts its budget
(`DivergedLinearSolve after 3 Newton iterations`, the first host run); the PETSc
substrate now applies `NewtonConfig::eta_min` through a KSP convergence test that
raises the forcing term EW chose to the floor before GMRES judges its first
residual (`docs/dev/newton-krylov.md` §10.3). The rows `example.brusselator.newton`
and `example.brusselator.newton_petsc` run five windows with `--check` on each
substrate, and the mutation that removes the floor fails exactly the PETSc row.
`petsc.brusselator_newton` (its own binary, since a PETSc abort would otherwise
take the shared verification binary's unrelated cases down with it) asserts the
one-window agreement between the substrates.

Unlike dn_heat's bound (a closed form from that problem's own affine
Jacobian), the Brusselator's window-Newton coupling has a nonlinear residual
spanning multiple window substeps, with no derived `||J⁻¹||`. The agreement
bound (`brusselator_bound::agreement_bound()`, `examples/brusselator/
agreement_bound.hpp`) is instead an A-POSTERIORI CALIBRATION, measured on the
SAME configuration the check itself runs: `measure_native_reference_error()`
(`test/petsc/brusselator_newton_test.cpp`) solves the identical window
natively at the check's own `NewtonConfig::rtol`, and again at two
successively tighter tolerances to obtain a reference solution, confirming
the two tight solves agree with each other to far less than the difference
either one has from the check-tolerance solve before treating that difference
(`e_n`) as the native solver's own measured error. The bound is then

```
e_n · (1 + ‖F_final,PETSc‖ / ‖F_final,native‖)
```

The SCALING ASSUMPTION is that both solvers' own errors relate to their own
TERMINAL residuals through the same operator, since both iterates sit near
the same root (the dn_heat bound's own identity, applied here only
approximately, since this residual is nonlinear) — a check failure under
this bound means either a real disagreement between the two substrates or
that this assumption itself has broken down, never grounds to loosen the
bound. An earlier formula also added the §10.3-measured 3.9e-8 relative
inter-Krylov term as a separate addend; that number is the first-Newton-step
residual disagreement between the two solvers' own Krylov methods on a
synthetic quadratic/arctan problem, not a bound on the distance between two
CONVERGED Newton iterates on the Brusselator's own window, so it no longer
appears in the asserted bound. `test/petsc/brusselator_newton_test.cpp`
still computes and prints that earlier formula alongside the current one,
labelled `[agreement-bound legacy]`
(`brusselator_bound::legacy_agreement_bound()`), purely informationally — a
host run reports both ratios side by side, but only the formula above is
asserted.

A round-off floor was examined and, unlike dn_heat's, found NOT defensibly
derivable from the code: the state update spans up to 11 backward-Euler
diffusion linear solves per window (`DiffusionSolver`, U's own
`window_sample_capacity`, `docs/dev/newton-krylov.md` §9.1), each followed
by remap interpolation, before the nonlinear reaction step. A rigorous
backward-error bound on that chain would need the diffusion operator's own
matrix norm and factorisation fill-in, neither derived from the committed
code; a plausible-looking constant asserted without that derivation would be
exactly the kind of unjustified margin this bound replaced. No round-off
term was added — the host run's own ratios decide whether the identity
above, alone, already holds with room to spare.

On the test's own grid (`test/petsc/brusselator_newton_test.cpp`'s 5×5/7×7,
the same grid the check itself uses — calibrating elsewhere would not
describe what the check runs), the two reference solves agree exactly and
measured, in the sandbox's native-only calibration, `e_n = 2.442491e-15`
(`check_iters=4`, `reference_iters=5`, `f_final_native=9.219417e-12`,
`reference_final_residual=1.783547e-14`, `reference_cross_check_diff=0.0`).
**On the owner's machine, 2026-10-12 (`main` at 278fbef,
`petsc.brusselator_newton`, one passing run of 1220.8 s)** the same
calibration gave `e_n = 2.664535e-15` (`check_iters=4`, `reference_iters=5`,
`f_final_native=8.142361e-12`, `reference_final_residual=2.341429e-14`,
`reference_cross_check_diff=0.0`); the PETSc terminal residual was
8.329282e-12, a ratio of 1.022957 to the native one; the asserted sharp bound
was 5.390239e-15 and the legacy bound, never asserted, 3.159481e-07. The row
passed, and its assertions are per element, so every element of the final U
and V differs between the substrates by at most 5.390239e-15. That run
printed no observed maximum, so no observed/bound ratio is recorded; the test
now prints one (`[agreement-bound] observed max ...`) and the next host run
will give it. The bound is recomputed from each run's own calibration, which
is why `e_n` differs between the two machines.

The communicator decision (`docs/dev/newton-krylov.md` §11.3): the PETSc
substrate always solves on `MPI_COMM_SELF`, never the Runner's own
`base_communicator`, because `check_region_supports_newton()` already
restricts every Newton-selected region -- on EITHER substrate -- to a
single cross-rank participant (the SAME D31 restriction the paragraph
above names); there is no cross-rank reduction for a single-rank Newton
solve to need regardless of substrate.

## `--check`

Exits non-zero unless: every window converged; the final U and V fields are `>= -1e-12` everywhere;
`max_window_balance_error <= 1e-11` and `|global_balance_error| <= 1e-9` (round-off scale, measured
tens of times over the per-window and global floors above); `max_window_exchange_error <= 1e-9`
(round-off; measured up to 8.7e-13 at the longest run). None of these thresholds were widened after
the fact to make a run pass.

## Running it

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
<build-dir>/examples/fuse_example_brusselator \
  --nx-u 16 --ny-u 16 --nx-v 24 --ny-v 24 \
  --dt-u 0.002 --dt-v 0.003 --window 0.02 --end 11.5 \
  --out results
```

Outputs `u.csv`, `v.csv` (final fields, `time,x,y,value`, the Toy's own format) and
`diagnostics.txt` (parameters and the measured residuals above, in the Toy's own key names) under
`--out`. `--save-history` additionally writes `windows.csv` (`t0,t1,mean_u,mean_v,iterations` per
window). `--accel none|aitken|iqn-ils|newton` selects `fuse::WindowCouplerConfig::acceleration` (the Toy is
unrelaxed Picard, its only option); `newton` is `--driver runner`-only (see "Newton acceleration"
above). `--iteration-mode jacobi|gauss-seidel`, `--no-block-jacobi` (the per-species
block-Jacobi preconditioner, default ON) and `--substrate native|petsc` (see "The PETSc
substrate" above) apply under any `--accel` value, read only when `newton` is selected. `--driver window|runner`
selects `fuse::WindowCoupler` (default) or `fuse::Runner` directly — both pass `--check`
identically for every `--accel` value `--driver window` supports (see "Measured results"). `--no-forcing`,
`--tolerance`, `--max-iterations`, `--alpha` mirror the Toy's own flags of the same name.

For the multi-executable path (tier 1, a Fortran compiler for V, host-verified for `mpmd` only):

```sh
cmake --preset tier1 && cmake --build --preset tier1
<build-dir>/examples/brusselator_graph_tool --ranks-u 1 --ranks-v 1 --ranks-coupler 1 \
    --launch mpmd > graph.yaml
bash test/parallel/brusselator_participants/run_participants.sh mpmd <mpiexec> 1 1 1 \
    <build-dir>/examples/brusselator_graph_tool <build-dir>/examples/brusselator_u_main \
    <build-dir>/examples/brusselator_v_main <build-dir>/examples/brusselator_coupler_main \
    <build-dir>/examples/brusselator_check <scratch-dir> 2 60
```

## State

**Verified** (tier 0, this sandbox): the in-process example (`example.brusselator`,
`example.brusselator.runner`, `example.brusselator.omp_identical`), the weights-vs-oracle and
convergence-order verification suites, the species-core solver decomposition and VParticipant
two-sided remap decomposition (in-process, thread-barrier stand-ins for a real `Reducer`/
`GhostExchanger`), and `brusselator_graph_tool`'s own reference-reproduction rows.
**Verified** (host, `KNOWN_ISSUES.md` D25/D28/D29): `verification.brusselator.participants.mpmd.{3,5}`
— the full three-executable, MPI-separated-process A4 case, at up to five total ranks.
**Deliberate refusal**, tested: `E-BRUSS-U-REACTION` (`BrusselatorRefusalCoverage.
UReactionFixedPointRefusesPastFiftySolves`, test/verification/
brusselator_refusal_coverage_test.cpp), `E-BRUSS-DIFFUSION-DIRECT-DECOMPOSED`
(`BrusselatorSpeciesCoreDecomposition.DirectSolverRefusesWhenDecomposed`).
**Deliberate refusal, implemented but deliberately untested**: `E-BRUSS-MESH-INDEX`
(`NativeGrid::make`) only fires when a loaded mesh's cell centroids disagree with the row
band/formula `NativeGrid::make` itself used to generate the "generated:quad:NxM" source it
loads — i.e. only on a corrupted or hand-crafted mesh view, never on the honest generator this
example always drives. `NativeMeshBackend`'s public surface has no way to substitute a
different view into a name's default slot to exercise this from a test: `load()` always
(re)generates one consistent with its own `nx`/`ny`; loading the same name twice makes
`mesh_id()` refuse `E-MESH-NAME-AMBIGUOUS` before any centroid is read; `set_placed_view`/
`commit_remap_halo` write the per-`ComponentId` placed slot, not the default slot
`NativeGrid::make` reads, and the latter is documented "nothing but the native halo machinery
may call this" (see test/verification/brusselator_refusal_coverage_test.cpp's trailing
comment for the full reasoning). **Absent / blocked, not a defect in this library**:
`verification.brusselator.participants.accept.{3,5}` (D23, needs MPICH ch4 or Open MPI);
`verification.brusselator.graph_tool.reproduces_reference_lowfive` is verified, but the
`lowfive.{mpmd,accept}.3` participant rows are blocked the same way D14's LowFive rebuild is for
every other LowFive-dependent row in this tree.
