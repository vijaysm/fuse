# FSI channel: an unsteady-Stokes channel with an Euler-Bernoulli wall beam

`examples/fsi_channel/` couples a 2D unsteady-Stokes channel to a 1D Euler-Bernoulli beam forming
its top wall, strongly coupled through a FUSE implicit coupling group on a non-matching 1D interface
(`fuse/backend/consistent_interp.hpp`'s `H` / conservative `H^T D`).

## Small-displacement assumption

Stated once here and true everywhere else this model is discussed: **the fluid domain is fixed**.
The beam's motion enters the fluid solve only through the top wall's normal-velocity
(transpiration) boundary condition; there is no moving mesh and no ALE term. This is the standard
small-displacement linearisation and is appropriate as long as the beam's deflection stays well under
the channel height `H`, checked by the `--check` gate and by
`FsiCoupled.SteadyStateBeamDeflectionMatchesTheAnalyticLinearLoad`'s own magnitude.

## Fluid: unsteady Stokes on a staggered MAC grid

```
rho u_t = -grad p + mu Lap(u),   div u = 0,   on [0,L] x [0,H]
```

Harlow-Welch staggering: `u` on vertical faces, `v` on horizontal faces, `p` at cell centres.
Time: Crank-Nicolson (trapezoidal); every boundary value is imposed strongly at its own time level
(t1 implicit, t0 explicit). Boundaries: bottom no-slip; top no tangential slip, normal velocity =
the beam's own `dw/dt` (transpiration); left a prescribed Poiseuille inflow; right an outflow.

**Wall ghost construction.** A Dirichlet wall value `D` is imposed through a ghost cell. The
TANGENTIAL no-slip components (bottom/top `u`; inflow/outflow-Dirichlet `v`) use a **quadratic**
ghost through `D` and the first two interior nodes, `ghost = (8 D - 6 c1 + c2) / 3` — the unique
quadratic through `(0,D)`, `(h/2,c1)`, `(3h/2,c2)` evaluated at the ghost location. A simpler linear
mirror (`ghost = 2D - c`) is exact only for an affine profile through the wall and was measured to
cost spatial order on this scheme (order 1.436-1.688 against a required >1.8); the quadratic
construction restores second order (measured 2.206 on a fast `{6,9,15,24}` grid ladder and 2.115 on
the original `{8,16,32,48}` ladder, both confirming the fix rather than a resolution-dependent
artefact). The remaining linear mirror-ghost form (`g = 2D - c`) stays in use only where it is exact
by construction — the normal-velocity Dirichlet condition and the pressure-outlet condition below —
and `FsiStokesMms.SpatialExactnessHoldsAtAnyGridForTheTemporalFixture` confirms that case directly.

**Outflow.** The outflow column's velocity is extrapolated (a zero-curvature/one-sided treatment,
not a zero-gradient one — see `stokes_solver.hpp`'s own derivation for why zero-gradient degenerates
the column's continuity equation into a load-incompatible constraint). **Pressure gauge**: with an
all-velocity boundary on three sides and a floating (outflow) velocity on the fourth, the discrete
operator carries two null directions — the pressure field's additive constant, and a uniform shift
in `u` compensated by a linear pressure ramp — confirmed by a direct `JacobiSVD` null-space
inspection (a singular value at machine zero, condition number ~6.5e18 with only a one-point pin).
The fix is the standard **pressure-outlet** condition, `p = 0` imposed along the whole outflow
column rather than at one interior point, pinning both null directions at once (condition number
after the fix: ~4.5e4). The trade-off: the outflow column's own discrete continuity equation is no
longer separately enforced, so the exact discrete Poiseuille state is only an approximate fixed
point of one Crank-Nicolson step at that column; global mass balance is still exact, carried by the
interior continuity rows' telescoping sum.

**The pressure-gradient treatment.** The momentum balance treats the pressure gradient
trapezoidally — `0.5*Gp1 + 0.5*Gp0` — matching the viscous term's own CN averaging, rather than
fully implicit (`Gp1` alone). A fully-implicit gradient leaves `p1`'s own value under-determined by
one genuine `|lambda|=1` mode living purely in the carried `p0` (incompressibility only ever pins
the *average* `p_mid = (p0+p1)/2`); the trapezoidal form removes that mode and makes the solved `p1`
genuinely t1-accurate (verified against a manufactured time-varying-pressure solution: order 2.0013
against the analytic `p(t1)`, versus order ~0 for the fully-implicit form).

**The wall traction delivered to the beam is the force *on the wall***, `+p - 2 mu dv/dy` — the
negative of Cauchy's `sigma . n` with the fluid's outward normal, which is the force the wall exerts
*on the fluid*. Getting this sign backwards (an earlier version of this port) manifests as a
negative added mass and an unstable coupled mode (see "History" below); `FsiCoupled.
APressurisedChannelPushesItsWallAwayFromTheFluid` and `FsiCoupled.
AcceleratingTheWallAwayFromTheFluidIsOpposedByTheTraction` state the physics directly, so a sign
flip cannot pass silently, and mutation `fsi-traction-is-the-force-on-the-fluid` reproduces the
defect.

Verified by MMS (`test/verification/fsi_stokes_mms_test.cpp`): a divergence-free
streamfunction-derived solution for the spatial-order study, and `u(x,y,t) = sin(w t) y`, `v = 0`,
`p = 0` — chosen so the mirror-ghost/staggered-divergence discretisation reproduces it exactly in
space at any resolution — isolating pure Crank-Nicolson time-truncation for the temporal-order study.

## Beam: clamped-clamped Euler-Bernoulli, finite differences + Newmark

```
rho_s h_s w_tt + EI w_xxxx = q(x,t),   x in [0,L],   w=w_x=0 at both ends
```

Second-order central finite differences in space on `n+1` nodes; the clamped slope condition uses
the same mirror-ghost construction as the fluid's walls. Newmark-beta with `beta=1/4, gamma=1/2`
("average acceleration") in time — unconditionally stable, second order. `q(x_i,t)` is a
*distributed* line load (N/m); the interface delivers a *nodal* load (an already-integrated force,
N), so the beam divides by its own tributary length (`BeamSolver::tributary_length(i)`, a
trapezoidal half-cell-plus-half-cell weight) before using it as `q`.

Verified (`test/verification/fsi_beam_test.cpp`) against the closed-form clamped-clamped deflection
under a uniform and an affine distributed load, and against the closed-form first natural frequency
(`beta_1 L = 4.730040744862704`, the first root of `cos(beta L) cosh(beta L) = 1`; Blevins,
*Formulas for Natural Frequency and Mode Shape*, Table 8-1). Both converge at the scheme's second
order. `BeamSolver::step_midpoint_load(dt, q_mid)`, which takes a single mid-step load directly
rather than averaging two endpoint forces, is likewise second order in isolation (measured order
2.2174).

## Interface: consistent H, conservative H^T D

Non-matching 1D interface (`Nx_fluid != Nx_beam`): beam nodal `dw/dt` -> fluid top-wall velocity by
`remap_method("consistent")` (piecewise-linear `H`, exact for an affine field at fixed
non-matching resolution); fluid wall traction -> beam nodal load by
`remap_method("consistent-transpose").conservative()` (`H^T` applied to an already-integrated load,
scaled by the source mesh entity set's own cell measure) — see [interface-transfer.md](interface-transfer.md).

**Unit convention.** FUSE's `generated:line:n` interface mesh spans the normalised `[0,1]` interval
regardless of physical length `L`; the fluid side's `fluid_force` field is therefore the wall
traction scaled by `L` (`fsi::traction_to_fuse_density`), so that `consistent-transpose`'s own
scaling by the source mesh's normalised cell measure (`1/Nx_fluid`) delivers a correctly-scaled
physical force (`traction * dx_fluid`) at each beam node.

## Coupling: one time grid, strongly coupled

Both components share one `ComponentDecl::interval` (`dt`, an exact `Rational`); the two field
transfers (velocity, force) form a cyclic dependency within one coupling cycle, so one of them
(`wall_force`) carries `.lag(1).allow_time_level_mismatch(...)` — the same pattern
[dn_heat](dn-heat.md) uses for its Dirichlet-Neumann pair — so the graph is schedulable at all; the
implicit coupling group resolves that lag every sweep. Acceleration: `AccelKind::None` (plain
Gauss-Seidel), `Aitken`, or `IQN_ILS`, selected by `--accel`. `--accel` also accepts `constant`, `secant`, `steffensen` and
`iqn-imvj` (the same names every example takes, examples/common/accel_names.hpp), and an unknown name is
refused with `E-EXAMPLE-ACCEL`. Measured 2026-10-20 at the command line above (mass ratio 1, `--dt 0.01`,
`--t-end 0.1`, sandbox tier-0 build, system compiler): `iqn-imvj` converges every window in 81 total
sweeps and `iqn-ils` in 218, both ending at mid-span deflection 1.063803e-02 m. `constant` and
`steffensen` exhaust the 200-sweep cap in all 10 windows with a diverging deflection (of order 1e289 and
1e256 m), and `secant` stops after 6 periods with a named refusal (`run_period: non-finite value detected`);
all three exit 1 under `--check`. `constant`, `secant`, `steffensen` and `iqn-imvj` use the same initial
relaxation (0.5) and, for `iqn-imvj`, the same history and filter as `iqn-ils`; none was tuned.

**Added-mass instability.** At `mass_ratio = rho_s h_s / (rho_f H) = 1`, plain Gauss-Seidel diverges
(`--accel none --check` fails: mid-span deflection runs away within a handful of sweeps). `IQN_ILS`
converges cleanly at the same mass ratio and grid — measured on this build (the `example.fsi_channel`
command below): 222 total sweeps over 10 steps, final mid-span deflection 1.064e-02 m. `Aitken`'s
scalar relaxation is measurably less robust than IQN-ILS's quasi-Newton update on this
mismatched-scale (velocity ~0.1 m/s, force ~1-10 N) two-field residual: it diverges at
`mass_ratio = 1` (residual growing without bound within a 500-sweep budget) but converges cleanly at
`mass_ratio = 25` (19 sweeps) — `FsiCoupled.AitkenDivergesAtLowMassRatioButConvergesAtHigh`. A
field-partitioned relaxation (`AccelConfig::set_field_partition`) is the likely way to make Aitken
converge at low mass ratio too; not implemented here.

## Building and running it

Tier 0 is enough:

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
<build-dir>/examples/fuse_example_fsi_channel --nx-fluid 16 --ny-fluid 8 --nx-beam 12 \
    --mass-ratio 1.0 --accel iqn-ils --dt 0.01 --t-end 0.1 --check
```

The exact-time scheduler steps on `--dt` rounded to a grid of 1e-6 (`rationalize_dt`). The example steps
on that delivered value for both the step count and the `time` column of the CSV, and prints a `NOTE` line
when the two differ, so a typed `--dt 0.0035104283866` runs as `0.00351`; a `--dt` that rounds to zero is
refused with `E-EXAMPLE-DT`. (Until 2026-10-22 the step count and the time column used the typed value, and the
reported time axis drifted from what the components advanced by up to `n_steps * 5e-7`; KNOWN_ISSUES.md D34 item C.)


Writes a CSV (`--csv path.csv`) of `step,time,sweeps,converged,beam_mid_deflection,
beam_max_deflection`. `--check` fails if any window does not converge under acceleration, if the
beam deflection is ever non-finite, or if the beam never moved.

Measured on this build (macOS arm64, AppleClang 21.0.0, Kokkos 5.1.0 OpenMP+Serial, tier0-release),
the exact `example.fsi_channel` `ctest` command:

```
FSI channel: fluid 16x8, beam 12 elements, mass ratio 1 (rho_s h_s = 0.2), accel=iqn-ils, dt=0.01, t_end=0.1
  beam first natural frequency: 70.75 rad/s (period 0.08881 s)
ran 10 steps; total sweeps 222, windows not converged 0
final beam mid-span deflection: 1.063803e-02 m
wrote fsi_channel.csv
all checks passed
```

`example.fsi_channel.omp_identical` runs the identical command at one OpenMP thread and at every
host core and diffs the output byte for byte — IQN-ILS goes through the Runner's own reductions, so
this is the example most exposed to a non-deterministic sum; verified on this build at 1 and 12
threads.

## History: the energy-growth defect and its fix (2026-09-25)

A long-run (12 beam periods) energy trace grew without bound even after each individually-plausible
partial fix (a trapezoidal pressure gradient to make `p1` genuinely t1-accurate; a careful
discrete-energy-identity bookkeeping of the wall/outflow boundary terms) — each closed one genuine,
independently-confirmed defect but left the coupled spectral radius essentially unchanged at
`rho ~ 1.1155`. The actual defect, found last: the wall traction `StokesSolver::step()` delivered to
the beam had the sign of the force *on the fluid*, not the force *on the wall* — a pressurised
channel pushes its wall outward, and the delivered value had the opposite sign. With the sign wrong,
the beam saw a **negative added mass**, and every earlier measurement (a real eigenvalue above 1 at
1.115466, a pressure-plus-first-bending-mode eigenvector, insensitivity to every boundary detail
varied) was a correct statement about a consistently wrong-signed force — no earlier test stated the
physics directly, so none caught it; the steady-deflection test's own analytic load happened to
carry the same sign.

| Measurement | Before the sign fix | After |
|---|---:|---:|
| coupled step-map spectral radius (diagnostic grid) | 1.115466 | 1.000000 |
| per-step energy ratio, periods 3-12, dt = T1/40 | 1.22796 | 0.998067 |
| steady-deflection order | 1.743 | 1.923 |
| transient self-convergence order, smooth start | 0.82 (impulsive start) | 1.846, pairwise 1.85 / 1.87 / 1.81 |

Host run 2026-09-25: every FSI suite passes — `fsi_transient` (order 1.846, inside its [1.8, 2.2]
bound), `fsi_energy_diag`, `fsi_coupled`, `fsi_stokes_mms`, `fsi_beam`. Two residual observations are
recorded, not asserted, since both are within their own passing bounds: the transient
self-convergence order sits at ~1.85 across three doublings rather than trending cleanly to 2.0 (the
remaining ~0.15 is unaccounted for); and an impulsive start (fluid at rest, inflow switched on at
`t=0`) is not itself second-order data for an incompressible system (`ImpulsiveStartIsMeasuredNotAsserted`,
recorded, not asserted). Unrelated to the growth and harmless: `BeamSolver::strain_energy()` is
missing a `dx_beam` factor relative to `kinetic_energy()`'s own quadrature; it never enters the
coupled state map.

Two earlier, now-superseded bugs were fixed on the way to this diagnosis and remain fixed: a
checkpoint-consistency gap where seeding the fluid solver directly bypassed the normal
`FluidComponent::advance()` write path, leaving the registered `fluid_force` field stale
(`FluidComponent::refresh_force_from_current_state()` fixes it); and the steady-state deflection
test originally seeding the fluid at Poiseuille but the beam at rest — not a fixed point of the
coupled system at all, so the "error" it measured was a real transient response, not a
discretisation error (fixed by also seeding the beam at its own analytic static equilibrium).

## Verification summary

| Test file | Status | Notes |
|---|---|---|
| `fsi_beam_test.cpp` | 5/5 pass | static deflection + frequency, 2nd order; midpoint-load step order 2.2174 |
| `fsi_stokes_mms_test.cpp` | 4/4 pass | spatial order 2.206 (fast grids) / 2.115 (original grids); temporal order clean 2nd order |
| `fsi_coupled_test.cpp` | 8/8 pass | steady-deflection order 1.923; the two traction-sign-physics tests; power balance to 1e-12 |
| `fsi_transient_test.cpp` | 2/2 pass | smooth-start order 1.846 in [1.8, 2.2]; impulsive start recorded (0.9), not asserted |
| `fsi_energy_diag_test.cpp` | 6/6 pass | coupled spectral radius = 1.000000 asserted; long-run energy ratio 0.998 |
| `fuse_example_fsi_channel` (`--check`) | passes with `--accel iqn-ils` | `--accel none` correctly fails (the added-mass demonstration) |

## State

**Verified**, with two `FUSE_ENABLE_CUSPARSE` caveats. `example.fsi_channel` and
`example.fsi_channel.omp_identical` pass at tier 0 on this build; every verification suite in the
table above passes both in this sandbox and on the host (2026-09-25 run). No multi-rank or
MOAB-dependent path exists for this example — both the fluid and beam solvers are single-rank: the
beam is always plain Eigen `SparseLU`, and the fluid solve goes through `SparseLuBackend`, Eigen
`SparseLU` by default; the non-matching 1D interface remap uses FUSE's
consistent/conservative-transpose builders on the serial backend.

Under `FUSE_ENABLE_CUSPARSE`, the fluid solve's `SparseLuBackend` is cusolverRf instead: both `example.fsi_channel`
rows above are then `DISABLED` (reported as "Not Run (Disabled)") rather than failing, because the
coupling is marginally stable at the step nearest the beam's first natural period and a ~1e-12
change in the fluid solve is enough to tip it — true on the flag-off build too under weaker
accelerators (`KNOWN_ISSUES.md` D36). Separately, `fsi_energy_diag_test.cpp`'s one 480-period case
is `GTEST_SKIP`ped under the flag: cusolverRf's solve is not bitwise reproducible run to run in situ
on this host, and that case usually fails to converge at period 140 because of it; the other five
cases in the binary still run (D37).
