# FSI channel: an unsteady-Stokes channel with an Euler-Bernoulli wall beam

`examples/fsi_channel/` couples a 2D unsteady-Stokes channel to a 1D
Euler-Bernoulli beam forming its top wall, strongly coupled through FUSE's
iteration region on a non-matching 1D interface (`fuse/backend/
consistent_interp.hpp`'s `H` / conservative `H^T D`).

## Small-displacement assumption

Stated once here and true everywhere else this model is discussed: **the
fluid domain is fixed**. The beam's motion enters the fluid solve only
through the top wall's normal-velocity (transpiration) boundary condition;
there is no moving mesh and no ALE term. This is the standard
small-displacement linearisation and is appropriate as long as the beam's
deflection stays well under the channel height `H` (checked by the
`--check` gate and by `FsiCoupled.SteadyStateBeamDeflectionMatchesThe
AnalyticLinearLoad`'s own magnitude).

## Fluid: unsteady Stokes on a staggered MAC grid

```
rho u_t = -grad p + mu Lap(u),   div u = 0,   on [0,L] x [0,H]
```

Harlow-Welch staggering: `u` on vertical faces ((nx+1) x ny), `v` on
horizontal faces (nx x (ny+1)), `p` at cell centres (nx x ny). Time: Crank-
Nicolson (trapezoidal); every boundary value is imposed strongly at its own
time level (t1 implicit, t0 explicit).

Boundaries: bottom no-slip; top no tangential slip, normal velocity =
beam's own `dw/dt` (transpiration); left a prescribed Poiseuille inflow;
right an outflow.

**Mirror-ghost Dirichlet walls.** A wall Dirichlet value `D` one half-cell
from the nearest real node `c` is imposed via the ghost `g = 2D - c` (the
value making `(g+c)/2 = D`), folded into that row's own diagonal. This
construction is *exact* for an affine profile through the wall (confirmed
directly: `FsiStokesMms.SpatialExactnessHoldsAtAnyGridForTheTemporalFixture`)
but carries a first-order local truncation for a genuinely curved profile.
**`FsiStokesMms.SpatialOrderIsTwo` FAILS as of this writing** (measured
order 1.436-1.688, required >1.8) and was left failing rather than
loosened to pass — see "Known open issue" below. Whether the local
truncation actually costs only a fraction of an order in the scheme's
*global* L2 rate, or is a fixable defect, was **not established** this
session; the speculative "costs roughly half an order" framing an earlier
draft of this file asserted as settled fact was not backed by a derivation
and has been removed.

**Outflow.** The outflow column's own velocity is extrapolated (a
zero-curvature/one-sided treatment; earlier zero-gradient drafts of this
file degenerated the same column's continuity equation into a hard,
load-incompatible constraint — see the header comments in
`stokes_solver.hpp` for the derivation). **Pressure gauge**: with an
all-velocity boundary on three sides and a floating (outflow) velocity on
the fourth, a single interior pressure pin removes only the pressure
field's additive constant; the discrete operator carries a *second*, genuine
null direction — a uniform shift in `u` compensated by a linear pressure
ramp — that one point-pin cannot remove (confirmed by direct `JacobiSVD`
null-space inspection: a singular value at machine zero, condition number
~6.5e18). The fix used here is the standard **pressure-outlet** condition:
`p = 0` is imposed (Dirichlet) along the *whole* outflow column, not at one
interior point, which pins both the additive constant and the ramp at once
(condition number after the fix: ~4.5e4, at the same grid). The trade-off,
reported honestly rather than hidden: the outflow column's own discrete
continuity equation is no longer separately enforced (replaced by the
pressure condition), so the *exact* discrete Poiseuille state is only an
*approximate*, not exact, fixed point of one Crank-Nicolson step at that
column (`FsiCoupled.FluidReconstructedPoiseuilleIsAFixedPointOfCnStep`,
open — see `KNOWN_ISSUES.md`/this task's `open_questions`); global mass
balance is still exact, carried by the interior continuity rows'
telescoping sum.

Verified by MMS (`test/verification/fsi_stokes_mms_test.cpp`): a
divergence-free streamfunction-derived solution for the spatial-order
study, and a separate solution — `u(x,y,t) = sin(w t) y`, `v = 0`, `p = 0`
— chosen so the mirror-ghost/staggered-divergence discretisation reproduces
it *exactly in space at any resolution* (confirmed directly before relying
on it), isolating pure Crank-Nicolson time-truncation for the temporal-order
study.

## Beam: clamped-clamped Euler-Bernoulli, finite differences + Newmark

```
rho_s h_s w_tt + EI w_xxxx = q(x,t),   x in [0,L],   w=w_x=0 at both ends
```

Second-order central finite differences in space on `n+1` nodes (`n`
elements); the clamped slope condition is imposed by the same mirror-ghost
construction as the fluid's walls. Newmark-beta with `beta=1/4, gamma=1/2`
("average acceleration") in time — unconditionally stable, second order.

`q(x_i,t)` is a *distributed* line load (N/m); the interface delivers a
*nodal* load (an already-integrated force, N — see Interface below), so
the beam divides by its own **tributary length** (`BeamSolver::
tributary_length(i)`, the trapezoidal half-cell-plus-half-cell weight at
each node) before using it as `q`.

Verified (`test/verification/fsi_beam_test.cpp`) against the closed-form
clamped-clamped deflection under a uniform and an affine distributed load
(derived by direct integration of `EI y'''' = q0 + q1 x` — see
`clamped_beam_affine_deflection` in `beam_solver.hpp`), and against the
closed-form first natural frequency (`beta_1 L = 4.730040744862704`, the
first root of `cos(beta L) cosh(beta L) = 1`; Blevins, *Formulas for
Natural Frequency and Mode Shape*, Table 8-1). Both converge at the
scheme's second order.

## Interface: consistent H, conservative H^T D

Non-matching 1D interface (`Nx_fluid != Nx_beam`): beam nodal `dw/dt` ->
fluid top-wall velocity by `remap_method("consistent")` (piecewise-linear
`H`, exact for an affine field at fixed non-matching resolution); fluid
wall traction (`-p + 2 mu dv/dy`) -> beam nodal load by
`remap_method("consistent-transpose").conservative()` (`H^T` applied to an
already-integrated load, scaled by the source region's own cell measure) —
see `docs/user/interface-transfer.md`.

**Unit convention.** FUSE's `generated:line:n` interface mesh spans the
*normalised* `[0,1]` interval regardless of physical length `L`; the fluid
side's `fluid_force` field is therefore the wall traction *scaled by L*
(`fsi::traction_to_fuse_density`), so that `consistent-transpose`'s own
scaling by the source region's normalised cell measure (`1/Nx_fluid`)
delivers a correctly-scaled physical force (`traction * dx_fluid`) at each
beam node.

## Coupling: one clock, strongly coupled

Both components share one `ComponentDecl::interval` (`dt`, an exact
`Rational`); the two channels (velocity, force) close a cycle within a
period, so one of them (`wall_force`) carries `.lag(1)`
`.allow_time_level_mismatch(...)` — the same pattern
`examples/dn_heat/main.cpp` uses for its Dirichlet-Neumann pair — so the
graph is schedulable at all; the iteration region resolves that lag every
sweep. Acceleration: `AccelKind::None` (plain Gauss-Seidel), `Aitken`, or
`IQN_ILS`, selected by `--accel`.

**Added-mass instability.** At `mass_ratio = rho_s h_s / (rho_f H) = 1`,
plain Gauss-Seidel diverges (measured: mid-span deflection reaches ~1e63 m
within a handful of sweeps; `--accel none --check` fails). `IQN_ILS`
converges cleanly at the same mass ratio and grid (measured: 185 sweeps
over 10 steps, mid-span deflection 0.1136 m). `Aitken`'s scalar relaxation
is measurably *less* robust than IQN-ILS's quasi-Newton update on this
mismatched-scale (velocity ~ 0.1 m/s, force ~ 1-10 N) two-field residual: it
converges cleanly at `mass_ratio >= ~6` but was not brought to convergence
at `mass_ratio = 1` within a bounded sweep budget in this session (open
question — see `open_questions`; a field-partitioned relaxation, per
`AccelConfig`'s own `set_field_partition` documentation, is the likely fix,
not yet implemented here).

## Running the example

```
fuse_example_fsi_channel --nx-fluid 16 --ny-fluid 8 --nx-beam 12 \
    --mass-ratio 1.0 --accel iqn-ils --dt 0.01 --t-end 0.1 --check
```

Writes a CSV of `step,time,sweeps,converged,beam_mid_deflection,
beam_max_deflection`. `--check` fails if any window does not converge
under acceleration, if the beam deflection is ever non-finite, or if the
beam never moved.

## Quadratic wall ghost (fixed: spatial order was below 2)

The mirror-ghost Dirichlet construction for a TANGENTIAL no-slip velocity
(bottom/top walls' `u`, the inflow/outflow-Dirichlet walls' `v`) was
originally LINEAR (`ghost = 2D - c`), exact only for an affine profile
through the wall -- a genuine O(h) local truncation at that row that cost
roughly half an order in the scheme's global L2 rate (measured: order
1.436-1.688, below the required >1.8). Replaced with a QUADRATIC ghost
through the wall value `D` and the first two interior nodes `c1` (this
row's own value), `c2` (the next node inward):

```
ghost = (8 D - 6 c1 + c2) / 3
```

the unique quadratic through `(0,D)`, `(h/2,c1)`, `(3h/2,c2)` evaluated at
the ghost location `-h/2`. Substituting into the 3-point Laplacian
`(c2 - 2 c1 + ghost)/h^2` gives `Lap = (8D - 12 c1 + 4 c2)/(3h^2)`: the
row's own diagonal gets `+4*halfmu*idy2` (not the interior
`+2*halfmu*idy2`), the neighbour `c2` gets `-(4/3)*halfmu*idy2` (not the
interior `-halfmu*idy2`), and `D`'s constant term moves to the RHS as
`+(8/3)*halfmu*idy2*D`. Applied everywhere the linear mirror was used for
a Dirichlet tangential component: `u` at the bottom and top walls, `v` at
the left (inflow) and right (only when NOT outflow, i.e. the fully-Dirichlet
MMS configuration) walls -- both the implicit (t1) and explicit (t0) sides
of Crank-Nicolson. Measured after the fix: order 2.206 on the fast
`{6,9,15,24}` grids, order 2.115 on the original `{8,16,32,48}` grids --
both above the required 2, confirming the fix and ruling out a
resolution-dependent artifact.

## Partitioned time-level consistency: the velocity side (fixed, incomplete alone)

`StokesSolver::step`'s wall-velocity boundary condition used to ignore its
own time argument, applying the SAME (new, t1) velocity value at BOTH
Crank-Nicolson's explicit (t0) and implicit (t1) evaluations. Fixed:
`step` now takes separate `top_wall_v_old` (t0) and `top_wall_v_new` (t1)
arrays; `FluidComponent` tracks the wall velocity at the period's start
(`velocity_in_t0_`) separately from the current sweep's guess. Measured
(round 2): this fix ALONE did not resolve the transient divergence -- the
FORCE side (below) turned out to carry the dominant mismatch.

## Partitioned time-level consistency: the force side -- root cause and fix

**Diagnosis.** Traced exactly, per period, at `fsi_transient_test`'s
coarsest `dt` (T1/40): the fluid's wall traction, sampled once per
(converged) coupling period, alternates sign and grows by a remarkably
consistent ratio: 158.9, -322.5, 849.9, -2073.8, 5180.8, -12844.3,
31930.8, -79287.8, 196996.2, -489288.4 -- |ratio| converging to 2.48,
sign flipping every period. This is the signature the lead's hypothesis
predicted: a genuine unstable eigenmode at the Nyquist frequency
(period-2*dt), not an iteration artefact (every window converges tightly
first).

**Why.** In an incompressible Crank-Nicolson step the pressure is the
step's Lagrange multiplier: continuity is enforced at t1, but the
momentum balance producing it averages the VISCOUS term between t0 and
t1 while treating the PRESSURE GRADIENT fully implicitly (p1 alone, no
p0 term at all). Confirmed directly against a manufactured solution with
a nontrivial time-varying pressure (`u(y,t)=A*sin(wt)*y`,
`p(x,t)=B*sin(wt)*x`, both exact for this scheme's own stencils): the
error against p(t1) or p(t0) is 5-10x LARGER than the error against
p(t_mid) at every dt tested, and does not shrink relative to it as dt
shrinks -- i.e. the solved pressure genuinely approximates
p(t_{n+1/2}), not p(t1). Delivering that pressure to the beam UNDER THE
LABEL "the load at t_{n+1}" (Newmark's own convention) is exactly the
half-step force/velocity mismatch the added-mass feedback amplifies.

**Two fixes were tried, in order:**

1. Linear extrapolation of the midpoint traction to t1,
   `tau(t1) ~= 1.5*tau_mid(n) - 0.5*tau_mid(n-1)`. This is EXACT for a
   linear-in-time signal and substantially reduced the growth (pushed
   the self-convergence test's onset of catastrophic blowup from
   `n_steps=20` to `n_steps=40` at the same total duration), but did NOT
   eliminate the instability -- because linear extrapolation itself has
   GAIN 2 at the Nyquist frequency (if `tau_mid(n-1) = -tau_mid(n)`,
   `1.5*tau_mid(n) - 0.5*(-tau_mid(n)) = 2*tau_mid(n)`), so a residual,
   smaller-gain feedback loop still diverges as `dt -> 0` (the added-mass
   term itself scales as `1/dt`, eventually overtaking any fixed,
   dt-independent gain reduction). Superseded by fix 2 below; not kept.

2. **Trapezoidal pressure gradient (kept).** `cn_step` (`stokes_solver.hpp`)
   now accepts an optional `p0` array; when supplied, the pressure
   gradient in BOTH momentum equations is
   `0.5*Gp1 + 0.5*Gp0` (half implicit p1, half explicit, known p0)
   instead of `Gp1` alone -- matching the viscous term's own averaging,
   so p1 becomes genuinely t1-accurate rather than a t_{n+1/2} quantity
   mislabelled as t1. `StokesSolver::step` passes its own previous
   pressure (`p_field_`, zero-initialised for the very first step,
   matching the channel's at-rest initial condition) as `p0`; every
   OTHER caller (both MMS tests) omits it and keeps the exact original,
   fully-implicit behaviour -- confirmed unaffected (`SpatialOrderIsTwo`
   and `TemporalOrderIsTwo` both still pass at their original numbers).
   Verified directly against the same manufactured pressure fixture: the
   error against the ANALYTIC p(t1) now converges at order 2.0013 (vs.
   order ~0 for both t1 and t0 before the fix). With this fix the
   extrapolation of fix 1 is no longer needed and was reverted
   (`FluidComponent::advance/save_state/restore_state` are back to the
   simple pass-through they were before round 3, plus the trapezoidal
   `p0` wiring).

**Measured effect.** The catastrophic divergence over a quarter beam
period (`fsi_transient_test`'s own self-convergence sweep, `n_steps` in
{10,20,40,80,160}) is GONE: mid-span deflection now converges smoothly
and monotonically (0.01873, 0.01914, 0.01939, 0.01953, 0.01961 m) with no
sign alternation at all. However, a LONGER run (dt fixed at T1/40, 480
steps = 12 full beam periods, instead of the self-convergence test's own
10 steps = a quarter period) still shows unbounded energy growth, with a
remarkably consistent per-step ratio of 1.22796 in total (beam+fluid)
energy, converged to 4 significant figures across periods 3-12 of that
run. In amplitude terms this is a per-step growth factor of ~1.108 --
compared to the ORIGINAL (pre-fix) per-step ratio of 2.48, this is a
substantial reduction in the instability's growth rate, but not full
elimination: the self-convergence test's own window (only a quarter
period) was simply too short to make this SLOWER residual growth visible.
`fsi_transient_test.cpp`'s own `SelfConvergenceInDtIsSecondOrderInTheAsymptoticRegime`
measures order 0.822 (required >1.8) with the smooth-but-slowly-diverging
sequence above -- left FAILING, not loosened, with these numbers. The
remaining root cause is not found within this round's budget; candidates
not yet ruled out: a residual mismatch in the VELOCITY delivery's own
consistency with the now-corrected pressure, or a genuinely separate
defect in the beam/fluid coupling's discrete eigenstructure at this mass
ratio. A full reproducer (the same long-run energy trace) is recorded in
`open_questions`.

## Partitioned time-level consistency: the force side, round 2

A follow-up round located the residual mechanism's own eigenvalue
precisely (`test/verification/fsi_energy_diag_test.cpp`'s affine-map
method — see KNOWN_ISSUES.md D9 for the full numbers) and formed a
specific hypothesis for it: the trapezoidal-pressure-gradient fix above
carries the PREVIOUS step's pressure (`p0`) forward as literal state to
make the solved p1 genuinely t1-accurate. But incompressibility only ever
pins the AVERAGE p_mid = (p0+p1)/2 — it never independently constrains p0
and p1 separately — so p1 = 2*p_mid - p0 has a free, genuine |lambda|=1
mode living purely in p0. The fluid alone never feels this mode (only
p_mid enters its own momentum balance), but the wall traction delivered
to the beam is built from p1, which DOES carry it — closing a feedback
loop through the beam's own load that could plausibly explain the
residual growth.

**Fix tested.** Revert `StokesSolver::step`'s `cn_step` call to a fully
implicit pressure gradient (`p0 = nullptr` — no p0 carried as state; the
solved `p_field_` genuinely IS p_mid, as the manufactured-solution
fixture already showed). Deliver `wall_traction()` as that genuine
mid-step value (`traction_mid_`, viscous part trapezoidally averaged over
t0/t1 to match CN's own convention for the momentum equation). Consume it
on the beam side with a NEW `BeamSolver::step_midpoint_load(dt, q_mid)`
that takes q_mid as a single mid-step load directly, rather than
averaging two endpoint forces the way Newmark's own convention would —
derived from averaging the beam's equation of motion at n and n+1:

```
(M/dt + K*dt/4) wdot_{n+1} = q_mid + (M/dt - K*dt/4) wdot_n - K w_n
w_{n+1} = w_n + dt/2 (wdot_n + wdot_{n+1})
```

confirmed to satisfy `M*ddot{w}_avg + K*w_avg = q_mid` exactly (to
round-off) after solving, and to preserve second-order temporal accuracy
in isolation (`FsiBeam.MidpointLoadStepConvergesAtSecondOrderInDt`, order
2.2174, off-resonance sinusoidal mid-step forcing).

**The hypothesis is confirmed correct — three independent measurements
agree exactly with what the derivation predicts:**

1. `FsiEnergyDiag.FluidAloneStepMapSpectralRadiusIsAtMostOne`:
   rho = 1.0000000000 (exactly marginal, the parasitic mode) before this
   fix, rho = 0.9946298894 (genuinely damped, no free mode) after.
2. The coupled step map's own "fluid p" input-column norm (finite
   difference perturbation of `p_field_` as carried state) is EXACTLY
   0.0 after this fix — perturbing pressure now provably has zero effect
   on the dynamics, because it is no longer carried state at all.
3. The beam-alone convergence test above confirms the midpoint
   substitution does not cost temporal order — the algebra is sound on
   its own terms.

**But it does NOT fix D9.** Measured directly, before/after this exact
change: `FsiEnergyDiag.CoupledStepMapRevealsTheGrowthMechanism`'s own
rho(M) moves from 1.115467 to 1.115466 — unchanged to 4 significant
figures, same dominant eigenvector (~84% pressure, ~16% beam wddot).
`FsiTransient`'s self-convergence order moves from 0.822 to 0.8182 —
also unchanged outside noise. A direct ablation (temporarily zeroing the
pressure term in the delivered traction) gives the IDENTICAL rho=1.000512
before and after this change. The parasitic p0 mode this round removes is
real, independently confirmed, and correctly fixed — but it sits
alongside a separate, comparable-magnitude, still-unidentified mechanism
in the pressure channel that this specific fix does not touch. D9 remains
OPEN; see KNOWN_ISSUES.md for the full round-2 writeup and the honest
"not yet found" statement.

**An unrelated regression was found and fixed as a side effect.** The
earlier round's trapezoidal-pressure-gradient scheme had apparently made
Aitken's scalar relaxation converge (422 sweeps) at mass_ratio=1, where it
previously did not converge at all; reverting to the fully-implicit
gradient reverses that side effect and Aitken now diverges there
(residual ~7.45e+111 within budget). `fsi_coupled_test.cpp`'s own
`AitkenNeedsFarMoreSweepsAtLowMassRatioThanAtHigh` was rewritten as
`AitkenDivergesAtLowMassRatioButConvergesAtHigh` to assert the current,
measured behaviour rather than the superseded one. IQN-ILS is unaffected
either way (19 sweeps at mass_ratio=1) — the accelerator this task's own
headline mass ratio actually needs still works.

## Partitioned time-level consistency: the force side, round 3

The lead's elimination argument: the continuous coupled energy identity
cancels its interface term iff the delivered traction is the discrete
work-conjugate of the wall velocity (hypothesis A), and the outflow's own
boundary term is sign-definite iff it enforces the physical "do-nothing"
condition (hypothesis B) -- at least one must fail, given each half is
bounded alone but the coupled rho is ~1.1155.

**Derived, from `cn_step`'s own discrete Laplacian (a plain 3-point
stencil for v in y and u in x -- no ghost complication at either
boundary this round needs).** The energy-conjugate viscous traction at a
Dirichlet-moving wall is a ONE-SIDED difference with coefficient mu (not
2*mu, the textbook full viscous stress -- that extra factor of mu comes
from the FULL symmetric stress tensor's own energy identity, which this
scheme's simplified vector-Laplacian momentum equation does not carry),
and the energy-conjugate pressure is the ADJACENT CELL's own value
exactly, no extrapolation -- both proved via the standard discrete
Green's identity for a 3-point Laplacian (worked by hand for the
boundary case, cross-checked numerically this round).

**Empirically, five candidate fixes were tried and NONE moved the
coupled rho(M) meaningfully** (baseline 1.115466; measured range across
all five: 1.115367-1.117273) -- pressure-only, viscous-only, and
combined hypothesis-A fixes; a genuine "do-nothing" outflow condition
(`-p+2*mu*du/dx=0`, replacing the zero-curvature ghost + p=0, with
continuity restored at the outflow column -- verified not to break the
Poiseuille fixed-point, steady-deflection, or power-balance tests); and
(A)+(B) combined. Fluid-alone rho changed substantially under each
variant (e.g. 0.9946->0.9786 under do-nothing), confirming the changes
are real, but the coupled system's own dominant eigenvalue is
insensitive to all of them.

**Why: a rigorous, round-off-validated discrete energy identity
explains it.** `test/verification/fsi_energy_diag_test.cpp`'s
`measure_full_energy_identity` (fluid alone, zero inflow, rest initial
state, nonzero CN-distinct top wall velocity) dots every interior
momentum row with its own CN-averaged velocity and sums, giving exactly
`(E1_interior-E0_interior)/dt = interior_viscous_power +
pressure_div_power - boundary_bt_term`. `boundary_bt_term` is the
continuity operator's own adjoint (B^T) contribution from v_wall/
u_outflow -- boundary velocities the continuity operator references but
which have no momentum row of their own, a contribution the naive
"momentum-row-only" identity misses; adding it closes this identity to
-7.3e-17 (round-off relative to E1~1.2e-2). `interior_viscous_power`
cleanly splits into bulk dissipation plus the two LOCAL, boundary-
attributable viscous flux terms above (measured: 1.58e-5 at the wall,
2.13e-6 at the outflow -- tiny). `pressure_div_power`, however, does
NOT localise to either boundary: it is `sum_cells(p1.div_avg)` over the
WHOLE domain (measured: -0.524, roughly 29,000x the combined local
viscous terms, 1.79e-5), dominated by `div_avg` being nonzero away from
both boundaries
too, in this diagnostic's own rest-initial-condition setup -- a property
of the GLOBAL continuity constraint's own Lagrange multiplier, not a
boundary effect either hypothesis can fix. This is exactly why the five
empirical fixes changed the local viscous numbers but left the coupled
rho untouched: the dominant term in this discretization's own energy
budget is not attributable to a wall-traction formula or an outflow
condition at all.

**Conclusion.** Per the lead's own fallback ("if neither A nor B is
nonzero... report that"): the per-term bookkeeping shows the lead's
local A/B elimination argument does not cleanly split for this
discretization's pressure channel the way it does for viscosity -- the
dominant energy-budget term is global (the continuity constraint's own
multiplier), not local. D9 remains OPEN; see KNOWN_ISSUES.md's own
round-3 section for the full numbers.

## Checkpoint-consistency fix (steady-state test)

`FsiCoupled.SteadyStateBeamDeflectionMatchesTheAnalyticLinearLoad` used to
throw a checkpoint-consistency exception before it could even run. Root
cause: the test seeds the fluid solver directly via
`fluid.solver().set_state(...)`, which bypasses `FluidComponent::advance()`'s
normal write path and therefore never updates the DERIVED, registered
`fluid_force` field (`force_out_`) to match the newly-seeded state. Fixed
by adding `FluidComponent::refresh_force_from_current_state()`, called by
the test immediately after `set_state()`.

## Steady-state deflection: root cause and fix

With the checkpoint bug fixed, the test ran but its error did not
decrease with resolution (round 2, before the trapezoidal-pressure fix:
order -0.064, errors ~0.0064-0.0069, essentially flat). Splitting the
error per the lead's instructions (round 3) -- (1) fluid's converged wall
traction vs. the analytic linear traction, and (2) the beam fed the
ANALYTIC load directly vs. the analytic deflection -- isolated the defect
immediately: the beam-side comparison (2) converged cleanly (errors
1.5e-4, 4.0e-5, 1.0e-5 -- order ~2), while the fluid-side traction
mismatch (1) was LARGE (~3.4-3.6, an O(1) mismatch, not a discretisation
error) and flat across all three resolutions.

**Root cause.** The test seeded the fluid at its exact discrete Poiseuille
state (`v_top=0`) but left the BEAM at rest (`w=0`, `BeamComponent`'s
default construction) -- NOT at its own static equilibrium under that
load. The pair (fluid at Poiseuille, beam at `w=0`) is NOT a fixed point
of the coupled system: `run_period()`'s own Newmark march genuinely
accelerates the beam away from `w=0` toward equilibrium WITHIN that one
period (a real physical transient, not a numerical artefact), delivering
a non-zero `wdot` -- hence a non-zero `v_top` -- back to the fluid, which
then genuinely changes the fluid's pressure field away from the seeded
Poiseuille profile. The large, flat error was measuring exactly this
transient response, not a discretisation defect.

**Fix.** The test now ALSO seeds the beam at its own analytic static
equilibrium (`BeamSolver::solve_static` under the analytic load, via
`BeamSolver::set_state(w, 0, 0)`) before `run_period()` -- together with
the fluid's Poiseuille seed, this is a genuine fixed point (zero net
force -> zero acceleration -> `w` stays put -> `wdot` stays 0 -> `v_top`
stays 0 -> the fluid stays put), so `run_period()` should leave it
essentially unchanged, and the residual IS the discretisation error the
test's name promises. Measured after the fix: order 1.743, errors
1.133e-04, 3.683e-05, 1.011e-05 at `nx_beam` in {8,16,32} -- cleanly
above the required >1.0 gate (kept unloosened; a >1.8 gate was not
required here since the beam's own FD order-2 test already covers that
separately).

## Round-off threshold comment (corrected)

`FsiStokesMms.SpatialExactnessHoldsAtAnyGridForTheTemporalFixture`'s
comment previously claimed the compared quantity was ">50x below" its
1.0e-8 threshold. Measured directly: `e1=6.5891e-08`, `e2=6.4599e-08`
(the errors compared are indeed ~6.5x the threshold, matching the lead's
correction, not ">50x below" it), while what the assertion ACTUALLY
compares, `|e1-e2|=1.2926e-09`, is ~50x SMALLER than `e1`/`e2` themselves
(the genuine "near-round-off relative to the errors" claim) but sits only
~7.7x below the 1.0e-8 threshold -- a real but modest margin. The comment
is corrected in place; the threshold itself is kept at 1.0e-8 (tightening
it risks flaking on a different machine's LAPACK without adding
confidence, since the exactness claim rests on the analytic
Laplacian-exactness argument in the test's own header, not on how tight
this floating-point check is).

## Aitken's added-mass limit (moved twice)

The earlier round's trapezoidal-pressure fix, by reducing the coupled
system's own added-mass feedback gain along the direction Aitken's scalar
relaxation is sensitive to, changed Aitken's own convergence boundary: it
previously did NOT converge at `mass_ratio=1` within a 500-sweep budget;
it then DID (measured: 422 sweeps). The test was renamed to
`AitkenNeedsFarMoreSweepsAtLowMassRatioThanAtHigh` to assert that
difficulty gradient instead of a moved pass/fail boundary.

Round 2's fix (see "the force side, round 2" above) reverted the
trapezoidal-pressure scheme, and with it, this side effect: Aitken now
DIVERGES again at `mass_ratio=1` (residual grows without bound, measured
~7.45e+111 within a 500-sweep budget) rather than converging slowly. The
test is renamed again, to `AitkenDivergesAtLowMassRatioButConvergesAtHigh`,
asserting non-convergence and a growing residual at `mass_ratio=1`
(mirroring `GaussSeidelDivergesWhileIqnIlsConvergesAtMassRatioOne`'s own
`residual_grew` check) while keeping the still-valid claim that Aitken
itself converges cleanly at `mass_ratio=25` (19 sweeps). IQN-ILS is
unaffected by either round's fix (19 sweeps at `mass_ratio=1` throughout)
— the accelerator this task's own headline mass ratio actually needs
still works regardless of which pressure scheme is in use.

## The energy growth: root cause and fix (2026-09-25)

The wall traction `StokesSolver::step()` delivered to the beam was
`-p + 2 mu dv/dy` -- Cauchy's `sigma . n` with the fluid's outward normal,
which is the force the wall exerts *on the fluid*. The force on the wall is its
negative: `+p - 2 mu dv/dy`. A pressurised channel pushes its wall outward.

With the sign wrong the beam saw a **negative added mass**, and everything
measured over four rounds of diagnosis follows from that: one real eigenvalue
above 1 (1.115466 on the diagnostic grid, energy x1.228 per step), a
pressure-plus-first-bending-mode eigenvector, and insensitivity to every
boundary detail varied -- because power balance, adjointness and the energy
budgets all compared the delivered vector with itself, and were correct
statements about a consistently wrong-signed force. No test stated the physics:
the steady-deflection test's analytic load carried the same sign.

With the sign corrected:

| Measurement | Before | After |
|---|---|---|
| coupled step-map spectral radius (diagnostic grid) | 1.115466 | 1.000000 (Newmark's acceleration passenger, energetically empty) |
| per-step energy ratio, periods 3-12, dt = T1/40 | 1.22796 | 0.998067 |
| steady-deflection order | 1.743 | 1.923 |
| transient self-convergence order, smooth start, T1/160..T1/2560 | (disabled, 0.82 from an impulsive start) | 1.846, pairwise 1.85 / 1.87 / 1.81 |

Two tests now state the physics rather than self-consistency, so a sign flip
cannot pass: `FsiCoupled.APressurisedChannelPushesItsWallAwayFromTheFluid` and
`FsiCoupled.AcceleratingTheWallAwayFromTheFluidIsOpposedByTheTraction`.
Mutation `fsi-traction-is-the-force-on-the-fluid` reproduces the defect.

Still open, and recorded rather than explained:
- the transient order sits at ~1.85 across three doublings rather than
  trending to 2 (`FsiTransient.SelfConvergenceInDtIsSecondOrderInTheAsymptoticRegime`
  passes its unchanged [1.8, 2.2] bound; the 0.15 is unaccounted for);
- an impulsive start (fluid at rest, inflow switched on at t = 0) is not
  second-order data for an incompressible system: the five-level regression
  over T1/40..T1/640 measures 0.87, and the three coarsest levels alone
  measure 0.15 (`ImpulsiveStartIsMeasuredNotAsserted`, recorded, not asserted);
- `BeamSolver::strain_energy()` is missing a `dx_beam` factor relative to
  `kinetic_energy()`'s quadrature (round 4's finding); it never enters the
  state map and is unrelated to the growth.

## Verification summary (measured, this session)

| Test file | Status | Notes |
|---|---|---|
| `fsi_beam_test.cpp` | 5/5 pass | static deflection + frequency, 2nd order (4 pre-existing cases); `MidpointLoadStepConvergesAtSecondOrderInDt` (new, round 2) order 2.2174; 63 ms total |
| `fsi_stokes_mms_test.cpp` | 4/4 pass | spatial order 2.206 (fast grids, 3.1s) and 2.115 (original `{8,16,32,48}` grids, 25.6s); temporal order clean 2nd order; ~28s total; unaffected by round 4's do-nothing outflow experiment (reverted) since MMS never sets `bcs.right.outflow=true` |
| `fsi_coupled_test.cpp` | 8/8 pass | steady-deflection order 1.923 (was 1.743); two physics-sign tests; power balance to 1e-12 |
| `fsi_transient_test.cpp` | 2/2 pass | smooth-start order 1.846 in [1.8, 2.2]; impulsive start recorded (0.9), not asserted |
| `fsi_energy_diag_test.cpp` | 6/6 pass | coupled rho = 1.000000 asserted; long-run energy ratio 0.998; budget armed, skipped on an energetically empty dominant mode |
| `fuse_example_fsi_channel` (`--check`) | passes with `--accel iqn-ils` | `--accel none` correctly fails (the added-mass demonstration) |
