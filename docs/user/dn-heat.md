# dn_heat: implicit coupling, Dirichlet-Neumann, iterated to a fixed point

`examples/dn_heat/` is a classical Dirichlet-Neumann pair of heat-conduction domains inside one
`fuse` implicit coupling group. Read it when you want to see convergence acceleration, rollback, a
convergence criterion covering two field transfers, and the Newton-Krylov coupler running together
against an analytic solution. Background: [](convergence.md).

## What it demonstrates

An implicit coupling group (glossary: *implicit coupling group*, a set of components whose coupling
within a step is solved to convergence by fixed-point sub-iterations) with two participants that have
no discretisation error of their own, so every bit of error in the answer comes from the coupling.
The same graph is run with explicit coupling, Picard, Aitken, IQN-ILS and Newton iteration, and each
is compared with the exact two-layer-wall solution. It needs only the base build; a PETSc-substrate
variant is available when the library is built with PETSc.

## Build and run

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
# or, without presets:
#   cmake -S . -B build -DFUSE_ALLOW_FETCH=ON && cmake --build build

<build-dir>/examples/fuse_example_dn_heat                 # the full configuration matrix
<build-dir>/examples/fuse_example_dn_heat --dump          # print the IR and plan, run nothing
<build-dir>/examples/fuse_example_dn_heat --newton-petsc  # PETSc substrate (needs FUSE_ENABLE_PETSC)
```

See [](build.md) for the presets and for building with PETSc.

With no flag the binary runs the **same graph** in several configurations:

- `explicit`: one sweep per coupling cycle, the classical staggered scheme.
- `picard`, `aitken`, `iqn-ils`: implicit coupling with the named accelerator.
- `iqn+qr2`: IQN-ILS with `AccelConfig`'s default QR2 column filter at its default `1e-2` limit,
  which is the wrong setting for this problem and is shown for contrast.
- `severed`: the exchange happens but lands in a decoy buffer on both sides.
- `expl-long`, `iqn-long`: the explicit and IQN-ILS cases over 10 times as many coupling cycles, so
  the explicit comparison is made asymptotically as well as during the initial transient.
- `newton`, `newton-nl`: the Newton-Krylov coupler (see "Newton" below).

`--dump` prints the IR/plan pair instead of running anything, as in [mms_pair](mms-pair.md).
`--newton-petsc` runs only the PETSc-substrate comparison. In a build without PETSc it is refused at
run time.

## The problem

Two slabs meet at a common interface, `L1` thick on one side and `L2` on the other, with
conductivities `k1(y)`, `k2(y)` varying along `n = 32` stations in `y` (four conductivity strips,
explained below). Conduction is 1D in `x` at each station (the thermal-resistance-network model of a
composite wall), and the outer faces are held at prescribed, slowly time-varying temperatures `T1(t)`,
`T2(t)`. Forcing is slow compared with the diffusion time, so each slab is in **quasi-steady state**
at every instant: the temperature profile through each slab is exactly a straight line, which both
participants evaluate in closed form.

```
DirichletSlab:  q(y) = k1(y) (T1(t) - u(y)) / L1        [the flux it takes to hold u]
NeumannSlab:    u(y) = T2(t) + q(y) L2 / k2(y)           [the temperature that flux implies]
```

Neither participant has a mesh, a time integrator, or a linear solve, so neither has a discretisation
error. Whatever the coupled answer misses the analytic solution by came from the coupling alone.
Composing the two relations gives the fixed-point map `u <- T2 + r(y) (T1 - u)` with
Dirichlet-Neumann contraction factor `r(y) = k1(y) L2 / (k2(y) L1)`. Its unique fixed point is the
series-resistance solution

```
u*(y,t) = (T2(t) + r(y) T1(t)) / (1 + r(y))
q*(y,t) = (T1(t) - T2(t)) / (L1/k1(y) + L2/k2)
```

which is the exact solution of the whole two-slab problem with continuous temperature and flux across
the interface. These two functions are the yardstick every run is measured against.

**Why the conductivity varies along the interface.** With one scalar `r` the coupled operator is a
scalar affine map and any secant method solves it in two sweeps, which says nothing about a vector
quasi-Newton scheme. Four conductivity strips make the operator a diagonal map with four distinct
eigenvalues (0.50 to 0.96). Picard crawls at the slowest of the four, Aitken's single scalar
relaxation cannot annihilate four modes at once, and IQN-ILS (a least-squares fit over the whole
concatenated residual) can, in about as many sweeps as there are distinct modes.

**Why the sweep is Jacobi, not Gauss-Seidel.** The textbook Dirichlet-Neumann iteration solves
Dirichlet, hands the fresh flux to Neumann, solves that, and repeats (Gauss-Seidel). A FUSE implicit
coupling group restores every checkpointed participant and re-advances them all from the same
delivered iterate, so both participants see the *previous* sweep's values: a Jacobi iteration on the
pair `(q, u)`. Writing `dq_k`, `du_k` for successive differences, one sweep gives `dq_{k+1} = -r du_k`
and `du_{k+1} = dq_k`. A block's norm contracts by `r` only every second sweep, so the per-sweep rate
is `sqrt(r)`, not `r`. That is why the printed Picard residual falls in near-equal pairs and why
Picard costs roughly twice the sweeps of a serial Dirichlet-Neumann iteration. It is a property of
the implicit group's concurrent execution model, not a defect.

## How the graph is authored

`build_graph` in `examples/dn_heat/main.cpp` declares two coincident 32-station interface meshes, one
component and one field transfer each way, a lag on one field transfer to make the cyclic dependency
schedulable, a convergence criterion covering *both* field transfers, and the implicit coupling group
itself:

```{literalinclude} ../../examples/dn_heat/main.cpp
:language: cpp
:start-at: "const auto flux = g.add_channel(\"flux\", q_out, q_in);"
:end-at: "return g;"
```

Both flux fields are `Extensive`, so `validate()` requires the `flux` channel to declare
`.conservative()` (`E-CONN-CONSERVATION` otherwise); between the two coincident meshes the remap is
the identity, which does conserve. The example calls `fuse::validate()` before `lower()` and the
graph passes it ([](quickstart.md)).

:::{warning}
A convergence criterion must measure every field transfer in the cycle. Deleting either `crit.add(...)`
line is rejected at `initialize()` with `E-ITER-MEASURE`, naming the unmeasured field transfer. A
single-measure criterion would let a quasi-Newton step zero the measured residual while the unmeasured
one is still far from converged, reporting convergence well short of the true fixed point.
:::

## Expected output

Exact digits can vary by platform. Columns are coupling periods, total sweeps, periods that did not
converge, and L2/max errors against the analytic solution for the interface temperature `u` and flux
`q`.

```
             periods   sweeps  !conv    L2 err u/K  max err u/K    L2 err q     max q
  explicit      20       20     20   1.193e+02  2.105e+02   7.751e+00  1.423e+01
  picard        20    14918      0   7.910e-07  1.582e-06   1.089e-06  2.178e-06
  aitken        20      938      0   4.184e-07  8.240e-07   2.852e-07  4.300e-07
  iqn-ils       20      201      0   4.923e-14  5.684e-14   3.627e-14  4.974e-14
  iqn+qr2       20     5716      2   1.443e-06  2.886e-06   1.443e-06  2.886e-06
  severed       20       40      0   3.035e+01  3.412e+01   2.409e+02  2.923e+02
  expl-long    200      200    200   2.312e+00  4.542e+00   4.963e+00  5.819e+00
  iqn-long     200     2016      0   8.541e-07  1.708e-06   8.541e-07  1.708e-06
  newton        20       20      0   8.660e-11  1.290e-10   1.260e-10  2.303e-10

implicit vs explicit, cold      : 1.193e+02 K -> 7.910e-07 K, factor 1.51e+08
implicit vs explicit, asymptotic: 2.312e+00 K -> 8.541e-07 K, factor 2.71e+06
implicit vs SEVERED exchange    : 3.035e+01 K -> 7.910e-07 K, factor 3.84e+07
sweeps: Picard 14918 | Aitken 938 (15.9x) | IQN-ILS 201 (74.2x) | IQN-ILS+QR2(1e-2) 5716 (2.6x)
rollback verifications 29796; advances surviving rollback 20 of 14918 sweeps over 20 periods
conservation checks 29836, violations 0, worst 0.000e+00
accelerators agree to: Picard-Aitken 2.406e-06 K, Picard-IQN 1.582e-06 K
reproducibility: bit-identical over 32 values, on a non-trivial state (||u||_2 = 300.415654 K, min 293.558 K, max 304.458 K)
all gates passed
```

Before the table the program prints the problem set-up and the checkpoint analysis for the iteration
region (which components must be restorable and that no coupler buffer is loop-carried). After it, it
prints the residual history of the final period for each accelerator.


| Quantity | Expected |
|---|---:|
| implicit vs explicit, cold start | 1.193e+02 K &rarr; 7.910e-07 K (1.51e+08&times;) |
| implicit vs explicit, asymptotic | 2.312e+00 K &rarr; 8.541e-07 K (2.71e+06&times;) |
| implicit vs severed exchange | 3.035e+01 K &rarr; 7.910e-07 K (3.84e+07&times;) |
| Picard sweeps | 14918 |
| Aitken sweeps | 938 (15.9&times; fewer) |
| IQN-ILS sweeps | 201 (74.2&times; fewer) |
| accelerators agree | Picard&ndash;Aitken 2.41e-06 K, Picard&ndash;IQN 1.58e-06 K |
| rollback verifications | 29796 |
| conservation violations | 0 of 29836 checks |
| reproducibility | bit-identical over 32 values |

The Newton rows add further lines, described below.

## What the example checks

The exit status is non-zero unless all of these hold:

- Every implicit run agrees with the other implicit runs and with the analytic solution.
- The implicit answer beats the explicit one by orders of magnitude, both cold and asymptotically.
- The implicit answer beats the severed case by orders of magnitude, which shows the headline number
  moves when the coupling actually breaks.
- IQN-ILS costs far fewer sweeps than Picard.
- At least one rollback was verified, and every verified rollback restored the checkpoint.
- Every conservation check (one per field transfer per sweep) reports zero violation.
- No two runs of the same configuration disagree (bit-for-bit reproducibility).

The output is also byte-identical at any OpenMP thread count.

## Newton

`AccelKind::Newton`, selected with `.newton(cfg)` on the iteration region instead of
`.accelerate(accel)`, solves the same graph with the native Jacobian-free Newton-Krylov coupler
instead of wrapping a quasi-Newton accelerator around the Picard sweep (see
[the Newton-Krylov design notes](../dev/newton-krylov.md)). The `newton` row runs it on the identical
affine map every other row solves. The `newton-nl` row (informational) solves a real nonlinearity
(`beta = 0.1`, a temperature-dependent wall conductivity) to measure Newton's quadratic convergence.

```
  newton        20       20      0   8.660e-11  1.290e-10   1.260e-10  2.303e-10
  NOTE: 'newton' row's own sweeps column (20) counts one per coupling period, not one per Newton
  step -- a Newton-selected region bypasses the Picard sweep loop entirely and so does not
  accumulate total_iterations the way the rows above do. The final period alone took 4 Newton step(s).

  newton       4 sweeps: 5.68e+01 4.31e+00 1.77e-02 1.48e-09
  newton-nl    4 sweeps: 5.68e+01 4.31e+00 1.78e-02 1.39e-08
  newton-nl fitted order(s) from consecutive residual triples above the round-off floor: 2.13 2.56
```

Newton agrees with Picard, Aitken and IQN-ILS to within the agreement gate those three hold each
other to, converges in 4 Newton steps in the period shown (against hundreds of Picard sweeps for the
same period), and is bit-for-bit reproducible. Do not compare the two counts directly: a Newton step
costs a full GMRES solve, typically several finite-difference residual evaluations, where a Picard
sweep costs exactly one. The fitted order on the nonlinear variant (2.13 and 2.56 here) is close to
the quadratic order Newton promises; it varies with the point in the trajectory because the state
keeps evolving over the 20 periods.

## The PETSc substrate

`NewtonConfig::substrate = NewtonSubstrate::Petsc` solves the identical graphs through PETSc's
`SNESNEWTONLS` instead of the native coupler. It is selected by data in the config, not by a code
fork. This path requires a build with PETSc (`FUSE_ENABLE_PETSC=ON`, see [](build.md)) and a call to
`fuse::petsc::install()` before running, which `--newton-petsc` makes. Without either, a graph naming
the PETSc substrate is refused at `Runner::initialize()` with `E-NEWTON-PETSC-UNAVAILABLE`.

`--newton-petsc` runs `newton`, `newton-petsc` and `newton-nl-petsc` as a separate invocation, not as
part of the default table:

| row | substrate | Newton steps per period | checked against |
|---|---|---|---|
| newton | native | 4 | the other implicit rows |
| newton-petsc | PETSc | 4 | native `newton`, with the agreement bound below |
| newton-nl | native | 4, fitted order about 2.1 to 2.6 | informational |
| newton-nl-petsc | PETSc | informational, `beta = 0.1`, `rtol` 1e-4 | informational |

The two substrates differ in how closely their GMRES implementations reach the shared Eisenstat-Walker
target, not in the test they apply.

**The agreement bound.** The check between the two substrates is an exact bound
(`dn_heat::agreement_bound()` in `examples/dn_heat/agreement_bound.hpp`), not a margin added on top of
one. For dn_heat's affine per-station Jacobian (`{{1, r_i}, {-1, 1}}`, station ratios in [0.50, 0.96]):

```
||x_n - x_p||_inf <= ||J^-1||_inf * sum_{k=1..kPeriods} (||F_k||_2,native + ||F_k||_2,PETSc)
```

with `||J^-1||_inf = 4/3` (the maximum over the four strips) and `||F_k||` each solver's own terminal
residual norm for period `k` (`Outcome::per_period_final_residual`). The sum runs over every period,
because each period's Newton solve starts from whatever the previous period committed and the two
substrates commit slightly different values. The state carried between periods is a plain assignment
from the current period's inputs, so the carry map has infinity-norm 1 and the per-period errors
telescope into the sum. `agreement_bound.hpp` has the full derivation.

The bound uses the achieved terminal residual, not the stopping criterion, because quadratic
convergence routinely lands the terminal residual several orders of magnitude below the level that
triggered the stop. On a 20-period run the observed disagreement is about three orders of magnitude below the asserted bound. The program also prints two informational
variants, `[agreement-bound last-period]` and `[agreement-bound legacy]`, which are not asserted.

:::{note}
A failure of this check means either a real disagreement between the two substrates or that the
bound's assumptions no longer hold (the per-station Jacobian is not the operator the solvers converged
under, or the state carried between periods is no longer a plain, unscaled assignment). The bound is
not meant to be loosened to make a run pass.
:::

### Two regions, two substrates, in one graph

`NewtonConfig::substrate` is chosen per region, so a single graph may name a different substrate for
each of its Newton regions. `--newton-petsc` also runs that case: a second, independent
Dirichlet-Neumann pair in the same graph, region `dirichlet_neumann_a` on the native substrate and
region `dirichlet_neumann_b` on PETSc, solved together through one Runner invocation. The
`[dual-substrate]` lines print both regions' residual histories for the final period and both regions'
converged flags, and the two converged interface temperatures are checked against each other with the
same per-period-summed bound. See [the Newton-Krylov design notes](../dev/newton-krylov.md).

## What to try next

- Change the `AccelConfig` passed to `run_case` in `main.cpp` from IQN-ILS to `aitken`, `picard` or `none`. The
  sweep counts in the table are what to expect: IQN-ILS needs about as many sweeps as there are
  distinct modes, Aitken several times that, Picard the most.
- Remove one `crit.add(...)` line in `build_graph` and rebuild. Initialization refuses with
  `E-ITER-MEASURE`.
- Compare the `iqn-ils` and `iqn+qr2` rows to see why the column-filter setting matters: with the
  default filter the run needs thousands of sweeps and some periods do not converge.
- Compare `severed` against the implicit rows. When the exchange is cut, the error is of order tens of
  kelvin, which shows the small errors elsewhere come from the coupling and not from the gates.

## See also

- [](convergence.md): convergence criteria and measures, including Newton's own tests.
- [](coupling-windows.md): coupling components that run at different time steps.
- [](interface-transfer.md): how field transfers between meshes are specified.
- [mms_pair](mms-pair.md): the explicit counterpart.
- [FSI channel](fsi-channel.md): the same lag-and-iterate pattern on a fluid-structure problem.
