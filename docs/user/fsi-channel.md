# FSI channel: an unsteady-Stokes channel with an Euler-Bernoulli wall beam

`examples/fsi_channel/` couples a 2D unsteady-Stokes channel to a 1D Euler-Bernoulli beam forming
its top wall, strongly coupled through a FUSE implicit coupling group on a non-matching 1D interface.
Read it to see an implicit coupling that is unstable without acceleration, and a non-matching
interface using the consistent and conservative-transpose transfers.

## What it demonstrates

A fluid and a structure with comparable inertia (mass ratio near 1) exhibit the added-mass
instability: plain Gauss-Seidel iteration diverges, and a quasi-Newton accelerator (IQN-ILS) converges.
The fluid and beam meshes do not match, so the two field transfers use `remap_method("consistent")`
and `remap_method("consistent-transpose").conservative()` (see [](interface-transfer.md)).
**This example requires the Eigen linear-algebra backend** (`FUSE_LINALG_BACKEND=eigen`, the
default): both solvers factor through `Eigen::SparseLU`, so it is not built under
`FUSE_LINALG_BACKEND=builtin`.

## Build and run

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
# or, without presets:
#   cmake -S . -B build -DFUSE_ALLOW_FETCH=ON && cmake --build build

<build-dir>/examples/fuse_example_fsi_channel --nx-fluid 16 --ny-fluid 8 --nx-beam 12 \
    --mass-ratio 1.0 --accel iqn-ils --dt 0.01 --t-end 0.1 --check
```

See [](build.md) for the presets. Flags (defaults in parentheses):

| Flag | Meaning |
|---|---|
| `--nx-fluid N` (24), `--ny-fluid N` (12) | Fluid cells along and across the channel. |
| `--nx-beam N` (16) | Beam elements; differing from `--nx-fluid` gives the non-matching interface. |
| `--mass-ratio R` (1.0) | `rho_s h_s / (rho_f H)`. |
| `--accel none\|constant\|aitken\|secant\|steffensen\|iqn-ils\|iqn-imvj` ("iqn-ils") | Coupling acceleration. An unknown name is refused with `E-EXAMPLE-ACCEL`. |
| `--dt D` (0.01), `--t-end T` (1.0) | Time step and end time. |
| `--csv FILE` ("fsi_channel.csv") | CSV of `step,time,sweeps,converged,beam_mid_deflection,beam_max_deflection`. |
| `--check` | Exit non-zero on failure (see "What the example checks"). |

The exact-time scheduler steps on `--dt` rounded to a grid of 1e-6. The example uses that delivered
value for both the step count and the `time` column of the CSV, and prints a `NOTE` line when it
differs from what you typed: `--dt 0.0035104283866` runs as `0.00351`. A `--dt` that rounds to zero is
refused with `E-EXAMPLE-DT`.

## The problem

**Small-displacement assumption.** The fluid domain is fixed. The beam's motion enters the fluid solve
only through the top wall's normal-velocity (transpiration) boundary condition; there is no moving
mesh and no ALE term. This is the standard small-displacement linearisation and is appropriate while
the beam's deflection stays well under the channel height `H`.

**Fluid: unsteady Stokes on a staggered MAC grid.**

```
rho u_t = -grad p + mu Lap(u),   div u = 0,   on [0,L] x [0,H]
```

Harlow-Welch staggering: `u` on vertical faces, `v` on horizontal faces, `p` at cell centres. Time
discretisation is Crank-Nicolson (trapezoidal), with every boundary value imposed strongly at its own
time level. Boundaries: bottom no-slip; top no tangential slip with normal velocity equal to the
beam's `dw/dt`; left a prescribed Poiseuille inflow; right an outflow.

- *Wall ghosts.* A Dirichlet wall value `D` is imposed through a ghost cell. Tangential no-slip
  components use a quadratic ghost through `D` and the first two interior nodes,
  `ghost = (8 D - 6 c1 + c2) / 3`, which keeps second-order accuracy. A linear mirror (`ghost = 2D - c`)
  is used only where it is exact by construction: the normal-velocity condition and the pressure outlet.
- *Outflow and pressure gauge.* The outflow column's velocity is extrapolated, and the pressure is set
  to zero along the whole outflow column (a pressure-outlet condition). That pins both null
  directions of the discrete operator at once (the pressure's additive constant and a uniform
  velocity shift compensated by a linear pressure ramp). Global mass balance remains exact.
- *Pressure gradient.* It is treated trapezoidally, `0.5*Gp1 + 0.5*Gp0`, matching the viscous term's
  averaging, so the solved `p1` is genuinely accurate at the new time level.
- *Wall traction.* The traction delivered to the beam is the force on the wall, `+p - 2 mu dv/dy`: a
  pressurised channel pushes its wall outward. The sign matters; the opposite sign acts as a negative
  added mass and makes the coupled problem unstable.

**Beam: clamped-clamped Euler-Bernoulli, finite differences and Newmark.**

```
rho_s h_s w_tt + EI w_xxxx = q(x,t),   x in [0,L],   w=w_x=0 at both ends
```

Second-order central differences on `n+1` nodes, with the clamped slope condition using the same
ghost construction as the fluid walls, and Newmark-beta (`beta=1/4, gamma=1/2`, average acceleration)
in time: unconditionally stable and second order. `q` is a distributed load (N/m); the interface
delivers a nodal load (N), so the beam divides by its tributary length
(`BeamSolver::tributary_length(i)`).

**Interface.** Beam nodal `dw/dt` goes to the fluid top-wall velocity through
`remap_method("consistent")` (piecewise-linear `H`, exact for an affine field). Fluid wall traction
goes to the beam nodal load through `remap_method("consistent-transpose").conservative()` (`H^T`
applied to an already-integrated load).

:::{note}
FUSE's `generated:line:n` interface mesh spans the normalised `[0,1]` interval whatever the physical
length `L`. The fluid's `fluid_force` field is therefore the wall traction scaled by `L`
(`fsi::traction_to_fuse_density`), so that the transpose remap's scaling by the normalised cell
measure `1/Nx_fluid` delivers the physical force `traction * dx_fluid` at each beam node. Reproduce this
scaling in your own components when you use the same remap on a physical-length interface.
:::

## How the graph is authored

Both components share one `ComponentDecl::interval` (`dt`, an exact `Rational`). The two field
transfers (velocity, force) form a cycle within one coupling cycle, so `wall_force` carries
`.lag(1).allow_time_level_mismatch(...)`, the same pattern [dn_heat](dn-heat.md) uses, and the
implicit coupling group resolves that lag every sweep. The graph is built in `examples/fsi_channel/main.cpp`
and the components are in `fsi_components.hpp`; the fluid and beam solvers
(`stokes_solver.hpp`, `beam_solver.hpp`) are plain classes with no FUSE dependency.

## Acceleration and the added-mass instability

`--accel` selects `none` (plain Gauss-Seidel), `aitken`, `iqn-ils`, or the other names above. With
mass ratio 1, `--dt 0.01`, `--t-end 0.1`:

| `--accel` | Result |
|---|---|
| `none` | Diverges within a handful of sweeps; `--check` fails. |
| `aitken` | Diverges at mass ratio 1; converges at mass ratio 25 (19 sweeps). Its scalar relaxation is less robust on this mismatched-scale (velocity about 0.1 m/s, force about 1-10 N) two-field residual. |
| `iqn-ils` | Converges every window: about 220 sweeps over 10 steps, final mid-span deflection 1.064e-02 m. |
| `iqn-imvj` | Converges every window in 81 sweeps, same final deflection 1.063803e-02 m. |
| `constant`, `steffensen` | Exhaust the 200-sweep cap in every window with a diverging deflection; `--check` fails. |
| `secant` | Stops after 6 periods with a named refusal (`run_period: non-finite value detected`); `--check` fails. |

`constant`, `secant`, `steffensen` and `iqn-imvj` use the same initial relaxation (0.5) as the others
and, for `iqn-imvj`, the same history and filter as `iqn-ils`; none is tuned. For low mass ratios
use a quasi-Newton accelerator, not Aitken.

## Expected output

The `--check` command above, exact digits can vary by platform:

```
FSI channel: fluid 16x8, beam 12 elements, mass ratio 1 (rho_s h_s = 0.2), accel=iqn-ils, dt=0.01, t_end=0.1
  beam first natural frequency: 70.75 rad/s (period 0.08881 s)
ran 10 steps; total sweeps 222, windows not converged 0
final beam mid-span deflection: 1.063803e-02 m
wrote fsi_channel.csv
all checks passed
```

The output is byte-identical at any OpenMP thread count. IQN-ILS goes through the Runner's own
reductions, so this is the example most exposed to a non-deterministic sum, and both components
declare `DeterminismLevel::Bitwise`.

## What the example checks

`--check` exits non-zero if any window does not converge under acceleration, if the beam deflection is
ever non-finite, or if the beam never moved. `--accel none --check` therefore fails, which is the
added-mass demonstration.

## Verification behind the example

The automated test suite verifies the model independently of this program:

- *Beam* (`fsi_beam_test.cpp`): static deflection and first natural frequency against closed forms
  (`beta_1 L = 4.730040744862704`), second order; the midpoint-load step is second order (2.22).
- *Stokes* (`fsi_stokes_mms_test.cpp`): spatial order about 2.1 to 2.2 and clean second-order temporal
  convergence against manufactured solutions.
- *Coupled* (`fsi_coupled_test.cpp`): steady-state beam deflection against the analytic load (order
  1.92), the two traction-sign physics tests, and power balance to 1e-12.
- *Transient* (`fsi_transient_test.cpp`): smooth-start self-convergence order about 1.85 (bounded in
  [1.8, 2.2]). An impulsive start (fluid at rest, inflow switched on at `t = 0`) is not second-order
  data for an incompressible system and is recorded, not asserted.
- *Energy* (`fsi_energy_diag_test.cpp`): the coupled step map has spectral radius 1.000000 and the
  long-run energy ratio per step is 0.998.

## Limitations

- The fluid domain is fixed (small displacement); large wall deflections are outside the model.
- The example is single-rank. No MPI or MOAB is involved.
- With `FUSE_ENABLE_CUSPARSE` the fluid solve uses `cusolverRf`, and the two `example.fsi_channel`
  ctest rows are registered as disabled: the coupling is marginally stable at the step nearest the
  beam's first natural period, and the tiny difference in that solver is enough to tip it. One
  480-period case in `fsi_energy_diag_test.cpp` is skipped under the flag for the same reason (the
  solve is not bitwise reproducible run to run).

## What to try next

- Vary `--mass-ratio` with `--accel aitken`: at 25 it converges (19 sweeps over the window budget), at 1 it diverges.
- Vary `--accel` over the names in the table above and compare the `total sweeps` line.
- Make the interface more strongly non-matching (`--nx-beam 7` against `--nx-fluid 16`) to exercise the
  consistent and conservative-transpose transfers at a different resolution ratio.
- Read the CSV written by `--csv` to see the beam oscillation at the first natural frequency printed
  in the header.

## See also

- [](interface-transfer.md): consistent and conservative-transpose transfers on non-matching meshes.
- [](convergence.md): convergence criteria and the accelerators used here.
- [dn_heat](dn-heat.md): the same lag-and-iterate pattern on a simpler problem.
- [](coupling-windows.md): coupling components that run at different time steps.
