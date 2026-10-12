# dn_heat: implicit coupling, Dirichlet-Neumann, iterated to a fixed point

`examples/dn_heat/` is a classical Dirichlet-Neumann pair of heat-conduction domains inside one
`fuse` implicit coupling group (glossary: *implicit coupling group* — a set of components whose
coupling within a step is solved to convergence by fixed-point sub-iterations). It is the example
that exercises checkpoint/restore (rollback), convergence acceleration, and a convergence
criterion covering two field transfers together, in one run.

## The problem, and why its error is the coupler's alone

Two slabs meet at a common interface, `L1` thick on one side and `L2` on the other, with
conductivities `k1(y)`, `k2(y)` varying along `n = 32` stations in `y` (four conductivity strips —
see "Why the conductivity varies" below). Conduction is 1D in `x` at each station (the standard
thermal-resistance-network model of a composite wall), and the outer faces are held at prescribed,
slowly time-varying temperatures `T1(t)`, `T2(t)`. Forcing is slow compared with the diffusion time,
so each slab is in **quasi-steady state** at every instant: the temperature profile through each
slab is exactly a straight line, which both participants evaluate in closed form —

```
DirichletSlab:  q(y) = k1(y) (T1(t) - u(y)) / L1        [the flux it takes to hold u]
NeumannSlab:    u(y) = T2(t) + q(y) L2 / k2(y)           [the temperature that flux implies]
```

Neither participant has a mesh, a time integrator, or a linear solve, and therefore neither has a
discretisation error. Whatever the coupled answer misses the analytic two-layer-wall solution by
came from the coupling and only from the coupling — the point this example exists to make, after an
earlier demo (`examples/dn_heat/main.cpp`'s own header comment names it as Task 4.7) measured a
coupler against a component carrying its own `O(dx^2) + O(dt)` error, which measured the component,
not the coupler. Composing the two relations gives the fixed-point map
`u <- T2 + r(y) (T1 - u)` with Dirichlet-Neumann contraction factor `r(y) = k1(y) L2 / (k2(y) L1)`;
its unique fixed point is the series-resistance solution

```
u*(y,t) = (T2(t) + r(y) T1(t)) / (1 + r(y))
q*(y,t) = (T1(t) - T2(t)) / (L1/k1(y) + L2/k2)
```

— the exact solution of the whole two-slab problem with continuous temperature and flux across the
interface. Those two functions are the yardstick every run is measured against.

**Why the conductivity varies along the interface.** With one scalar `r` the whole coupled operator
is a scalar affine map and any secant method solves it in two sweeps, which demonstrates nothing
about a vector quasi-Newton scheme. Four conductivity strips make the operator a diagonal map with
four distinct eigenvalues (spread 0.50 to 0.96): Picard crawls at the slowest of the four, Aitken's
single scalar relaxation cannot annihilate four modes at once, and IQN-ILS (a least-squares fit over
the whole concatenated residual) can, in about as many sweeps as there are distinct modes.

**Why the sweep is Jacobi, not Gauss-Seidel, and why that changes the rate.** The textbook
Dirichlet-Neumann iteration solves Dirichlet, hands the fresh flux straight to Neumann, solves that,
repeats — Gauss-Seidel. A FUSE implicit coupling group restores every checkpointed participant and
re-advances them all from the same delivered iterate (`src/runtime/iteration.cpp`: deliver every
block, then run every tick), so both participants see the *previous* sweep's values — a Jacobi
iteration on the pair `(q, u)`. Writing `dq_k`, `du_k` for successive differences, one sweep gives
`dq_{k+1} = -r du_k` and `du_{k+1} = dq_k`, so a block's norm contracts by `r` only every *second*
sweep and the per-sweep rate is `sqrt(r)`, not `r` — why the printed Picard residual falls in
near-equal pairs, and why Picard costs roughly twice the sweeps a serial Dirichlet-Neumann iteration
would. This is a property of the implicit coupling group's concurrent execution model, not a defect.

## How it is authored

Graph authoring (`examples/dn_heat/main.cpp::build_graph`) declares two coincident 32-station
interface meshes, one component and one field transfer each way, a lag on one field transfer to make
the cyclic dependency schedulable, a convergence criterion covering *both* field transfers, and the
implicit coupling group itself:

```{literalinclude} ../../examples/dn_heat/main.cpp
:language: cpp
:start-at: "const auto flux = g.add_channel(\"flux\", q_out, q_in);"
:end-at: "return g;"
```

Deleting either `crit.add(...)` line is rejected at `initialize()` with `E-ITER-MEASURE` naming the
unmeasured field transfer, rather than silently converging to the wrong fixed point: a single-measure
criterion lets a quasi-Newton step zero the *measured* residual while the unmeasured one is still far
from converged, reporting convergence well short of the true fixed point.

## Building and running it

Tier 0 is enough:

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
<build-dir>/examples/fuse_example_dn_heat
<build-dir>/examples/fuse_example_dn_heat --dump
```

With no flag the binary runs the **same graph** in seven configurations — `explicit` (one sweep per
coupling cycle, the classical staggered scheme), `picard`, `aitken`, `iqn-ils`, `iqn+qr2` (IQN-ILS
with `AccelConfig`'s default QR2 column filter at its default `1e-2` limit — shown to be the wrong
setting for this problem), `severed` (the exchange happens and lands in a decoy buffer on both
sides), and `expl-long`/`iqn-long` (the explicit and IQN-ILS cases over 10x as many coupling cycles,
so the explicit comparison is also made asymptotically and not only during the initial transient) —
and exits non-zero unless every implicit run agrees with the others and with the analytic solution,
the implicit answer beats the explicit one by orders of magnitude both cold and asymptotically, it
beats the severed case by orders of magnitude (the measurement proving the headline number moves
when the coupling actually breaks), IQN-ILS costs far fewer sweeps than Picard, every rollback was
verified, and no two runs of the same configuration disagree.

## What it prints

Measured on this build (macOS arm64, AppleClang 21.0.0, Kokkos 5.1.0 OpenMP+Serial, tier0-release,
`fuse_example_dn_heat` with no arguments):

```
             periods   sweeps  !conv    L2 err u/K  max err u/K    L2 err q     max q
  explicit      20       20     20   1.193e+02  2.105e+02   7.751e+00  1.423e+01
  picard        20    14918      0   7.910e-07  1.582e-06   1.089e-06  2.178e-06
  aitken        20      938      0   4.184e-07  8.240e-07   2.852e-07  4.300e-07
  iqn-ils       20      219      0   4.019e-14  5.684e-14   5.356e-14  9.948e-14
  iqn+qr2       20     8261      4   1.443e-06  2.886e-06   1.443e-06  2.886e-06
  severed       20       40      0   3.035e+01  3.412e+01   2.409e+02  2.923e+02
  expl-long    200      200    200   2.312e+00  4.542e+00   4.963e+00  5.819e+00
  iqn-long     200     2039      0   5.924e-07  1.185e-06   5.924e-07  1.185e-06

implicit vs explicit, cold      : 1.193e+02 K -> 7.910e-07 K, factor 1.51e+08
implicit vs explicit, asymptotic: 2.312e+00 K -> 5.924e-07 K, factor 3.90e+06
implicit vs SEVERED exchange    : 3.035e+01 K -> 7.910e-07 K, factor 3.84e+07
sweeps: Picard 14918 | Aitken 938 (15.9x) | IQN-ILS 219 (68.1x) | IQN-ILS+QR2(1e-2) 8261 (1.8x)
rollback verifications 29796; advances surviving rollback 20 of 14918 sweeps over 20 periods
conservation checks 29836, violations 0, worst 0.000e+00
accelerators agree to: Picard-Aitken 2.406e-06 K, Picard-IQN 1.582e-06 K
reproducibility: bit-identical over 32 values, on a non-trivial state (||u||_2 = 300.415654 K, min 293.558 K, max 304.458 K)
all gates passed
```

| Quantity | Measured this build |
|---|---:|
| implicit vs explicit, cold start | 1.193e+02 K &rarr; 7.910e-07 K (1.51e+08&times;) |
| implicit vs explicit, asymptotic | 2.312e+00 K &rarr; 5.924e-07 K (3.90e+06&times;) |
| implicit vs severed exchange | 3.035e+01 K &rarr; 7.910e-07 K (3.84e+07&times;) |
| Picard sweeps | 14918 |
| Aitken sweeps | 938 (15.9&times; fewer) |
| IQN-ILS sweeps | 219 (68.1&times; fewer) |
| accelerators agree | Picard&ndash;Aitken 2.41e-06 K, Picard&ndash;IQN 1.58e-06 K |
| rollback verifications | 29796 |
| conservation violations | 0 of 29836 checks |
| reproducibility | bit-identical over 32 values |

`--dump` prints the IR/plan pair instead of running anything (see [mms_pair](mms-pair.md) for the
shared convention, asserted the same way by the `example.dn_heat.dump` ctest row).

## Newton

`AccelKind::Newton` (selected with `.newton(cfg)` on the iteration region
rather than `.accelerate(accel)`) solves the SAME graph with S7.2's native
Jacobian-free Newton-Krylov coupler instead of a quasi-Newton accelerator
wrapped around the Picard sweep -- the full design is
[docs/dev/newton-krylov.md](../dev/newton-krylov.md). `examples/dn_heat/
main.cpp`'s own `newton` row runs it on the IDENTICAL affine map every other
row above solves (`DirichletSlab`'s own `beta = 0.0` default, bit-identical
to today's map -- `Dn.NonlinearAtBetaZeroIsBitIdenticalToTheAffineMap`,
test/runtime/newton_region_test.cpp); a `newton-nl` row (informational, like
`iqn+qr2` above) solves a REAL nonlinearity (`beta = 0.1`, a temperature-
dependent wall conductivity) to measure the quadratic convergence rate
Newton's own method promises.

Measured on a tier-0 Debug scratch build (not the release build the table
above was measured on -- Newton's own sweep accounting differs enough from
the Picard-family rows that the two are not meant to be read side by side as
one table; see the NOTE line below):

```
  newton        20       20      0   4.159e-11  6.321e-11   3.631e-10  5.960e-10
  NOTE: 'newton' row's own sweeps column (20) counts one per coupling period,
  not one per Newton step -- a Newton-selected region bypasses the Picard
  sweep loop entirely and so does not accumulate `total_iterations` the way
  the rows above do. The final period alone took 4 Newton step(s).
  newton       4 sweeps: 5.68e+01 4.31e+00 1.77e-02 3.26e-09

  newton-nl    4 sweeps: 5.68e+01 4.31e+00 1.78e-02 1.45e-08
  newton-nl fitted order(s) from consecutive residual triples above the
  round-off floor: 2.13 2.55
```

Newton agrees with Picard/Aitken/IQN-ILS to within the same agreement gate
those three hold each other to, converges in 4 Newton steps in the period
measured (against Picard's hundreds of sweeps for the same period, though
the two counts are not directly comparable -- a Newton step costs a full
GMRES solve, typically several MFFD-perturbed residual evaluations, where a
Picard sweep costs exactly one), and is bit-for-bit reproducible across two
runs of the same configuration. The fitted order on the nonlinear variant,
2.13 and 2.55 at this point in the trajectory, is within the "order 2.0 +-
measured" claim the unit test's own independent measurement
(`NewtonRegion.QuadraticRateMeasuredOnTheNonlinearProblem`, docs/dev/
newton-krylov.md §8.3) makes more precisely (2.24, 2.01) at a different
point in the coupled system's own time history -- the two measurements
agree that the order is close to 2, not that they are measuring identical
numbers, since `newton-nl` runs the nonlinearity over 20 real coupling
periods (the state keeps evolving) where the unit test isolates a single
period in a hand-built fixture.

The line search is shown to engage and recover from a deliberately poor
initial guess in `NewtonRegion.LineSearchEngagesAndRecoversFromA
DeliberatelyPoorGuess` (test/runtime/newton_region_test.cpp) rather than in
this example binary -- reproducing that specific combination
(`beta = 2.0`, `NeumannSlab`'s own initial guess forced to 2000 K) inside
`main.cpp`'s own gated configuration matrix was judged not to add
information the dedicated unit test does not already carry, and keeping it
out of this binary's own gate list avoids adding a configuration whose own
correctness gates would need to tolerate the SAME sign-flip/stall regime the
unit test's own beta sweep found at larger nonlinearity (docs/dev/
newton-krylov.md §8.4).

## The PETSc substrate

`NewtonConfig::substrate = NewtonSubstrate::Petsc` solves the IDENTICAL
graphs the `newton`/`newton-nl` rows above do, through PETSc's own
`SNESNEWTONLS` instead of the native JFNK coupler (`fuse::petsc::solve()`,
[docs/dev/newton-krylov.md §10/§11](../dev/newton-krylov.md)) -- selected by
data (`NewtonConfig::substrate`), never a code fork. `examples/dn_heat/
main.cpp --newton-petsc` runs `newton-petsc`/`newton-nl-petsc`: a SEPARATE
invocation (not folded into the default table), compiled in at all only
under `FUSE_HAVE_PETSC`, calling `fuse::petsc::install()` first so the
substrate registry actually resolves `NewtonSubstrate::Petsc`. Without
either (a build without `FUSE_ENABLE_PETSC`, or a process that never calls
`install()`), a graph naming the PETSc substrate refuses at
`Runner::initialize()` with `E-NEWTON-PETSC-UNAVAILABLE` --
`NewtonRegion.ARegionSelectingAnUnregisteredSubstrateIsRefusedAtSetup`
(test/runtime/newton_region_test.cpp) fires this refusal tier-0, without
PETSc, on every build.

| row | substrate | Newton steps | final \|\|F\|\| (final period alone) | agrees with Picard | ctest row |
|---|---|---|---|---|---|
| newton | native | 4 (per period; 5.68e+01 → 4.31 → 1.77e-02 → 3.26e-09) | 4.159e-11 | 6.321e-11 K | `example.dn_heat` |
| newton-petsc | PETSc | 4 | 1.649e-10 | 2.717e-10 K | `example.dn_heat.newton_petsc` (gated `FUSE_ENABLE_PETSC`) |
| newton-nl | native | 4 (… → 1.45e-08); fitted order 2.13, 2.55 | — | — | informational (`example.dn_heat`) |
| newton-nl-petsc | PETSc | informational, β = 0.1, rtol 1e-4 | — | — | informational, under `--newton-petsc` |

Measured on the development host (2026-10-11, `/opt/petsc/download-moab`): the
PETSc row converges in the same four steps per period and agrees with Picard to
2.7e-10 K against the native path's 6.3e-11 K — the two differ in how closely
their GMRES implementations reach the shared Eisenstat–Walker target, not in the
test they apply. (These two numbers are each solver's own FINAL-PERIOD terminal
residual, the "final period alone" column above — the run itself covers
`kPeriods` = 20 periods, and the bound below now accounts for all of them, not
only the last.)

The agreement check `--newton-petsc` performs between the two substrates is an
EXACT bound (`examples/dn_heat/agreement_bound.hpp`,
`dn_heat::agreement_bound()`), not a margin on top of one: for dn_heat's affine
per-station Jacobian (`{{1, r_i}, {-1, 1}}`, station ratios in [0.50, 0.96]),

```
||x_n - x_p||_∞ ≤ ||J⁻¹||_∞ · Σ_{k=1}^{kPeriods} (||F_k||₂,native + ||F_k||₂,PETSc)
```

summed over EVERY period `k`, with `||J⁻¹||_∞ = 4/3` exactly (the maximum over
the four strips) and `||F_k||` each solver's own TERMINAL residual norm for
period `k` alone (`Outcome::per_period_final_residual`), not the last period's
alone and not the `rtol·||F₀||` stopping criterion.

The sum over all periods, not just the last, is the correction this bound
needed: a single period's own identity (`x - x* = J⁻¹F(x)`, exact for affine
`F`, given identical inputs) only directly compares two solves that started
from the IDENTICAL state, which is true of period 1 but not of periods 2
through `kPeriods` — each later period's own Newton solve starts from
whatever the PREVIOUS period committed, and native and PETSc commit slightly
different values. Writing `e_k` for the max-norm disagreement after period
`k` and `A` for the map carrying period `k-1`'s own committed state into
period `k`'s own Newton starting point, `dn_component.hpp`'s own `advance()`
formulas (`q_[i] = k1_eff(...)`, `u_[i] = T2 + q_in_[i]·kL2/kK2`, both plain
assignments from the CURRENT period's own inputs, nothing else touching
these arrays between periods) show `A` is the identity, so `||A||_∞ = 1`
and the recursion `e_k ≤ ||A||_∞ e_{k-1} + ||J⁻¹||_∞(F_{n,k}+F_{p,k})`
telescopes (`e_0 = 0`: both substrates start identically) to the sum above.
`examples/dn_heat/agreement_bound.hpp`'s own "PER-PERIOD RECURSION" section
has the full derivation, including why a tighter (zero-amplification) analysis
is plausible but not what this bound asserts.

Newton's own quadratic convergence means a solver's terminal residual
routinely lands several orders of magnitude below the level that triggered
its own stop (§8.2 of [docs/dev/newton-krylov.md](../dev/newton-krylov.md)
measures a 765.78 → 68.77 → 0.288 → 2.6e-8 sequence on a related problem); a
bound built from the stopping criterion rather than the achieved residual is
looser than the identity above by that same margin, and would still pass a
PETSc substrate wrong by many orders of magnitude. For the same reason, the
check does not add the §10.3-measured 3.9e-8 relative inter-Krylov term: that
number is the first-Newton-step residual disagreement between the two
solvers' own Krylov methods on a synthetic quadratic/arctan problem, a
property of a single linear solve, not a bound on the distance between two
CONVERGED Newton iterates on dn_heat's own graph, and mixing it additively
into a bound on converged states combined two unrelated quantities' units.
`main.cpp` still computes and prints that earlier (now informational only)
formula alongside the asserted one, labelled `[agreement-bound legacy]`, so a
host run reports both ratios side by side; only the sum above is asserted.

A separate round-off floor was examined and found negligible for a SINGLE
period of this example: the state update that actually produces `u_` from a
converged Newton residual (`NeumannSlab::advance`) is `u_[i] = T2 + q_in_[i] *
kL2 / kK2`, and since `kL2 == kK2 == 1.0` exactly, the division and
multiplication are IEEE-754 exact, leaving a single rounded addition. A
standard backward-error bound on that one operation
(`|fl(a+b) - (a+b)| ≤ u·(|a|+|b|)`, `u` the IEEE double unit round-off
≈1.11e-16) gives a floor of at most `1.11e-16 · (|T2| + |q_in|) ≈ 1.11e-16 ·
(295 + 730) ≈ 1.1e-13` K per period using this example's own temperature
range — `1e-10 / 1.1e-13 ≈ 909`, roughly THREE orders of magnitude below a
single period's own sharp-bound terms (order 1e-10 K above). It was NOT
added to the formula: both substrates execute the identical `advance()` code,
so a disagreement at this scale already originates in `x_native ≠ x_petsc`
upstream of the rounding, a difference the sharp bound's own terms already
cover, not a separate source of slack the bound needs to carry. The sum over
`kPeriods` periods only widens this margin relative to the floor, never
narrows it, since the sum is never smaller than any one period's own term.

The check passes as `example.dn_heat.newton_petsc`; the observed
disagreement, the sum-based sharp bound, the legacy bound, the per-period
residual sums (`Σ F_k` for each substrate), and their ratios from that run
are PENDING a host measurement (this exact binary cannot run
`--newton-petsc` in the sandbox that derived the formula, KNOWN_ISSUES.md
D32's own "cannot complete `MPI_Init()`" note). A failure of this check means
either a real disagreement between the two substrates or a breakdown of the
bound's own assumption (the per-station Jacobian stops being the operator
either solver actually converged under — for instance if a future change
makes the graph non-affine, or the state carried between periods stops being
a plain, unscaled assignment); the bound itself must never be loosened to
make a failing run pass.

## Measured against

`ctest` row `example.dn_heat` runs the configuration matrix above and gates exit status on exactly
the relationships it prints: implicit must beat explicit and severed by orders of magnitude (cold
and asymptotic alike), the accelerators must agree with each other and with the analytic solution,
IQN-ILS must cost far fewer sweeps than Picard, a rollback must have been verified at least once, and
every conservation check (one per field transfer per sweep) must report zero violation.
`example.dn_heat.omp_identical` (`tools/compare-runs.cmake`) runs the identical binary at one OpenMP
thread and at every host core and diffs the output byte for byte; verified on this build at 1 and 12
threads.

IQN-ILS over IQN-IMVJ was a deliberate choice, measured elsewhere in this project on a nonlinear
contraction: IMVJ was 1.3-3.6&times; slower with erratic per-window sweep counts against a flat count
for ILS, landed in a different valid fixed point in 5 of 8 measured windows, and has no restart mode
to refresh its retained Jacobian — not used to demonstrate convergence quality anywhere in this
example.

## State

**Verified.** `example.dn_heat`, `example.dn_heat.dump`, and `example.dn_heat.omp_identical` all pass
at tier 0 on this build. The implicit coupling group mechanics this example exercises — derived
checkpoint set, two-field-transfer convergence criterion, Aitken and IQN-ILS acceleration, rollback —
are each independently covered by `test/runtime/` unit tests; this example is the one place all of
them run together against an analytic solution rather than a synthetic fixture.
