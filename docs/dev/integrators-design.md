# Integrators: design for porting KARMA's time integrators into fuse::num

Scope: upstream/karma/{include,src}/numerics/temporal, pinned sha 136f520 (UPSTREAMS.lock [karma]).
Every claim below about the vendored source is a source read or a grep, not a build. The scope
rules that bear on the design: the FEM core is adapted to fuse::linalg rather than ported
literally; all KARMA time integrators are ported into fuse core; physics-kit is tier 0 with PETSc
optional; a KARMA test is retired only if a named fuse test covers the same behaviour.

## 0. Plain answers first

- **How many integrators.** 22 named Runge-Kutta/Rosenbrock methods (the `RKType` enum,
  `include/enums/runge_kutta_schemes.h:42-64`), driven through 7 integrator classes (one base,
  `TemporalIntegrator`, plus `ExplicitRKIntegrator`, `ImplicitDIRKIntegrator`,
  `EmbeddedDIRKIntegrator`, `DOPRI45Integrator`, `RosenbrockIntegrator`, `PseudoTimeIntegrator`,
  `PetscTemporalIntegrator`), across 66 files and 8,316 lines (MEASURED, `wc -l`).
- **Which families.** Explicit RK (forward Euler, Kutta 2(2)/4(4), Dormand-Prince 5(4)); diagonally
  implicit RK, both plain (backward Euler, Crank-Nicolson, 8 SDIRK variants) and embedded
  (TRBE, SDIRK332, SDIRK543, SDIRK643); Rosenbrock/ROW methods with Newton-type linearisation
  (ROS22, ROS3P, RODASP, GRK4A, GRK4T, GRK4Sn); one pseudo-transient-continuation driver
  (backward Euler specialised for steady-state marching via dynamical time scales); one pass-through
  to PETSc's own `TS` object.
- **What fuse::linalg lacks for this.** Nothing for the matrix-vector/Krylov layer: `DistSparseMatrix`
  and `MatrixFreeOperator` already satisfy the same `LinearOperator` concept gmres/pcg consume
  (`include/fuse/linalg/operator.hpp:69-90`), and `fuse::num::solve()` is already a complete
  Jacobian-free Newton-Krylov driver with a Native/PETSc substrate registry
  (`include/fuse/num/newton_krylov.hpp:360-504`) -- this is the architecture KARMA's own 2007-2011
  author was reaching for (`temporal_integrator.h:23`, `"replace by karma_nonlinear_solver.h"`) and
  never built. What is missing is a **Jacobian-assembly interface** (KARMA's `jacobian_matrix`/
  `mass_matrix_subtract`, `include/interface/karma_interface.h:165,177`, assume a matrix KARMA can
  assemble; `fuse::num` today only ever differences a residual) and a **general direct linear solve**
  (`SchurSlabSolver` is an exact direct solver for one specific periodic-Laplacian operator,
  `include/fuse/linalg/schur_slab.hpp:4`, not a general sparse factorisation).
- **The proposed interface, in brief** (full version in Section 4):

```cpp
template <class T> concept TemporalProblem = requires(T& p, double t, const /*State*/& y,
                                                        /*State*/& out) {
  { p.rhs(t, y, out) } -> std::same_as<Result<void>>;
};                                   // + optional mass(), jacobian_action(), has_mass, is_stiff

template <CoupledValue Value, TemporalProblem<Value> Problem>
class Integrator {
 public:
  // one accepted step; Value is double in production, dual<S,N> under a sensitivity sweep
  virtual Result<StepRecord<Value>> step(Problem&, Value t, Value h, State<Value>&) = 0;
  virtual const ButcherTableau& tableau() const = 0;      // stages, A, b, c, (b - b_hat)
  virtual int order() const = 0;
  virtual std::optional<int> embedded_order() const = 0;  // nullopt: no error estimate
};

template <CoupledValue Value>
class EmbeddedController {                                 // Hairer & Wanner's PI/I controllers
 public:
  Result<StepDecision<Value>> propose(Value err_norm, Value h, const ControllerState&) ;
  // clip_to_window(): the ONE place a sub-step is shortened to land on a schedule point
};
```

## 1. Inventory (66 files, 8,316 lines, MEASURED)

### 1.1 Base classes (3 files, not independently steppable)

| File | Lines | Role |
|---|---|---|
| `karma_ts_base.h`/`.C` | 68/39 | `KARMATSBase`: order + A/L/S-stability pure-virtual trio. |
| `rk.h`/`rk.C` | 195/178 | `RK : KARMATSBase`: owns the Butcher arrays `A,B,C,E` (and an unused `AInv`, see 6.3), the `RKType`/`RKCategory` tags, and the static `RK::build()`/`RKMethodTraits<N>` chain-of-responsibility factory (`rk.h:182-193`, specialised once per method, `build()` falls through `RKMethodTraits<N-1>` until it matches -- a runtime-enum-to-type dispatch). |
| `rosenbrock.h`/`.C` | 77/68 | `Rosenbrock : RK`: adds the `G` (linearised-stage, i.e. "Gamma") coefficient matrix on top of the Butcher arrays. |
| `temporal_integrator.h`/`.C` | 295/524 | `TemporalIntegrator : IKARMA<...>`: owns the PETSc `Vec`/`SNES`/`KSP`/`PC` handles, the step/stage-count counters, `set_adaptive_params()` (defaults `safe=0.9, facl=0.2, facr=4.0, hmax=(tend-tstart)/5`, `temporal_integrator.C:101-107`), and `TemporalIntegrator::build()`, the enum-to-integrator-class dispatch (`INTEGRATOR_BUILD_CASE` macro, `temporal_integrator.C:188-199`, one `case` per `RKType`, selecting `RKMethodTraits<N>::StaticIntegratorType` or `::AdaptiveIntegratorType` on a `-adaptive` command-line flag). `estimate_error()` is **declared** (`temporal_integrator.h:132`) and **never defined anywhere in the vendored tree** (grepped) -- dead declaration; every integrator that does error-based control computes its own, independently (Section 2). |

### 1.2 The 22 Butcher-tableau methods (44 files: one `.h` + one `.C` each)

Columns: stages = `_primary_stages`; order = `n_order()`; embedded = secondary order if the
category is one of the `*Embedded`/`ROSENBROCK(Embedded)` categories, else "--"; A/L/S/FSAL read
from each class's own `is_*_stable()`/`is_FSAL()` bodies (every one MEASURED by `awk` over the
inline method bodies, not inferred from the method name).

| Method | `RKType` | Stages | Order | Embedded order | Category | A | L | S | FSAL | Driven by |
|---|---|---|---|---|---|---|---|---|---|---|
| Forward Euler (FE) | 0 | 1 | 1 | -- | ERK | F | F | F | F | `ExplicitRKIntegrator` |
| Kutta 2(2) | 1 | 2 | 2 | -- | ERK | F | F | F | F | `ExplicitRKIntegrator` |
| Kutta 4(4) | 2 | 4 | 4 | -- | ERK | F | F | F | F | `ExplicitRKIntegrator` |
| Dormand-Prince 5(4) | 3 | 7 | 5 | 5 (metadata bug, see 6.1; true embedded order is 4) | ERKEmbedded | F | F | F | F | `DOPRI45Integrator` |
| Backward Euler (BE) | 4 | 1 | 1 | -- | FIRKNonSingular | T | T | F | T | `ImplicitDIRKIntegrator` |
| Crank-Nicolson (CN, trapezoidal form) | 5 | 2 | 2 | -- | FIRKSingular | T | F | F | T | `ImplicitDIRKIntegrator` |
| SDIRK22 | 6 | 2 | 2 | -- | SDIRK | T | T | T | T | `ImplicitDIRKIntegrator` |
| SDIRK23 | 7 | 2 | 3 | -- | SDIRK | T | F | T | F | `ImplicitDIRKIntegrator` |
| SDIRK34 | 8 | 3 | 4 | -- | SDIRK | T | F | T | F | `ImplicitDIRKIntegrator` |
| SDIRK332 | 9 | 3 | 3 | 2 | SDIRKEmbedded | T | T | T | F | `ImplicitDIRKIntegrator` (static) / `EmbeddedDIRKIntegrator` (adaptive) |
| SDIRK543 | 10 | 5 | 4 | 3 | SDIRKEmbedded | T | T | T | T | `ImplicitDIRKIntegrator` / `EmbeddedDIRKIntegrator` |
| SDIRK643 | 11 | 6 | 4 | 3 | SDIRKEmbedded | T | F | F | T | `ImplicitDIRKIntegrator` / `EmbeddedDIRKIntegrator` |
| TRBE (trapezoidal/BE embedded pair) | 12 | 2 | 2 | 1 | DIRKEmbedded | T | F | F | T | `ImplicitDIRKIntegrator` / `EmbeddedDIRKIntegrator` |
| SDIRK33 | 13 | 3 | 3 | -- | SDIRK | T | T | T | T | `ImplicitDIRKIntegrator` |
| SDIRK45 | 14 | 4 | 5 | -- | SDIRK | T | F | F | F | `ImplicitDIRKIntegrator` |
| ROS22 | 15 | 2 | 2 | none (d'=d by design, see 6.1) | ROSENBROCK | T | T | T | F | `RosenbrockIntegrator` |
| GRK4A | 16 | 4 | 4 | 3 (claimed; E is identically zero, see 6.1 -- a vendor defect) | ROSENBROCKEmbedded | T | F | F | F | `RosenbrockIntegrator` |
| GRK4T | 17 | 4 | 4 | 3 | ROSENBROCKEmbedded | T | T | F | F | `RosenbrockIntegrator` |
| GRK4Sn | 18 | 4 | 4 | 3 | ROSENBROCKEmbedded | T | T | F | F | `RosenbrockIntegrator` |
| ROS3P | 19 | 3 | 3 | 2 | ROSENBROCKEmbedded | T | T | T | F | `RosenbrockIntegrator` |
| RODASP | 20 | 6 | 4 | 3 | ROSENBROCKEmbedded | T | T | T | F | `RosenbrockIntegrator` |
| SDIRK44 | 21 | 4 | 4 | -- | SDIRK | T | T | T | T | `ImplicitDIRKIntegrator` |

All 22 are `RK::build()`-constructible; none needs a mass matrix or a Jacobian at the *tableau*
level -- those are the *integrator driver's* concern (Section 3), not the method's.

### 1.3 The 7 integrator drivers (exactly `temporal_integrator.{h,C}` plus 6 more pairs)

| Driver | Lines (.h/.C) | What it does | Needs from the problem |
|---|---|---|---|
| `ExplicitRKIntegrator` | 65/328 | Classical explicit RK loop (`explicit_rk_integrator.C:153-311`): stage loop, no linear or nonlinear solve. Its "adaptive" branch is **not** an embedded-error controller -- it is a **physics-specific dynamical-time-scale controller** (`interface->compute_dynamical_timescales`, line 205, feeding `hnew = safe*dynmax*pow(prev_dynmax/dynmax,0.2)*pow(h/hold,0.2)`, line 258): a different adaptivity mechanism from the Hairer-style one below, asserted only for `ERK`/`ERKEmbedded` categories (`explicit_rk_integrator.C:36`) even though it never reads `E`. |
| `DOPRI45Integrator` | 72/339 | The classical Hairer-Norsett-Wanner DOPRI5(4) step with a **PI (Gustafsson) step-size controller**: `expo1 = 0.2 - beta*0.75`, `beta=0.08` ("recommended by Hairer", `dopri45_integrator.C:46`), `fac11 = pow(err,expo1)`, `fac = fac11/pow(facold,beta)` (Lund stabilisation), step accepted iff `err <= 1.0` where `err` is the rtol/atol-scaled RMS norm of `h*sum(E_i k_i)` (lines 199-245). Stiffness-detection heuristic present (`hlamb`/`iasti`, lines 259-276), returns error code `-4` if triggered 15 times running. |
| `ImplicitDIRKIntegrator` | 63/540 | Per-stage Newton solve via PETSc `SNES` (`SNESSolve(_snes, Zi, _curr_sol)`) on the stage residual `Z_i - h*sum A_ij f(t+c_i h, y_n+Z_j) = 0`; no error control. |
| `EmbeddedDIRKIntegrator` | 63/463 | Same stage loop as above, plus an **I-controller** (not PI): `order_power = 1/(min(order,embedded_order)+1)` (`embedded_dirk_integrator.C:183`), `facerr = pow(LTE_tolerance/local_error, order_power)`, `newdelt = safe*h*facerr` (lines 352-355) -- `local_error` is `VecNorm(k[0], NORM_2, ...)` (a genuinely collective PETSc call), compared against a single scalar `LTE_tolerance` (default `1e-6`, line 174), not a per-component rtol/atol vector the way DOPRI45 does it. Two different embedded-error architectures coexist in this directory (Section 2). |
| `RosenbrockIntegrator` | 65/530 | One assembled Jacobian per step (`__karma_form_temporal_jacobian`, line 226, builds `J - h*gamma_ii*I`-type operators from `interface->jacobian_matrix`/`mass_matrix_subtract`), reused across stages scaled by each stage's own `gamma_ii`; a non-autonomous `dfdt` correction term (`RosenbrockIntegrator::dfdt`, a raw function pointer, `rosenbrock_integrator.h:41`) for `d/dt f(t,y)` when the problem is not autonomous -- classical Rosenbrock-method requirement (Hairer & Wanner, *Solving ODEs II*, IV.7). Stage solves are **linear** (one assembled-matrix solve per stage, not a Newton iteration) -- this is what distinguishes Rosenbrock from DIRK. |
| `PseudoTimeIntegrator` | 63/349 | A fixed backward-Euler stage hardwired (not parameterised by `_scheme`'s `A`/`B`/`C` beyond reading them), used for pseudo-transient continuation to steady state; adapts its step from the same dynamical-time-scale formula as `ExplicitRKIntegrator`, not an embedded error. |
| `PetscTemporalIntegrator` | 60/213 | Delegates entirely to PETSc's own `TS` object (`TSCreate`/`TSSetIFunction`/`TSSetIJacobian`/`TSSetType(_ts, _name)`, `petsc_temporal_integrator.C:101-164`) -- KARMA's *second* substrate for the identical mathematical methods, selected by name string rather than the `RKType` enum. This is the direct precedent for `fuse::num::NewtonSubstrate::{Native,Petsc}` (`newton_krylov.hpp:422`): the same pattern, one layer up, already exists in fuse for the nonlinear solve. |

### 1.4 Test coverage (MEASURED: grepped, not assumed)

**No test anywhere in `upstream/karma/test/` instantiates any class in `numerics/temporal`.**
Grepped every `.C`/`.h` under `include/`, `src/`, `examples/` outside `numerics/temporal` itself
for `ExplicitRKIntegrator|ImplicitDIRKIntegrator|EmbeddedDIRKIntegrator|RosenbrockIntegrator|
DOPRI45Integrator|PseudoTimeIntegrator|PetscTemporalIntegrator|TemporalIntegrator::build|RKType`:
zero hits. The one test whose name suggests coverage, `test/ode_system_test.C`, instantiates
`ODESystem`/`ode_mass_operator.h`/`ode_system_operator.h` (`include/physics/ode-system/`), which
wrap **PETSc's own `TS`** (`ode_system.h:8`, `#include <petscts.h>`) directly, not
`numerics/temporal`'s own RK hierarchy; it asserts a Crank-Nicolson-equivalent scheme converges at
rate 1.8-2.2 (`ode_system_test.C:90`), but that assertion exercises a sibling implementation of the
same mathematical method, never `crank_nicolson.{h,C}`. Under the rule that a KARMA test is retired only if a
named fuse test covers the same behaviour (otherwise it is ported), since no named test covers any
of the 22 methods or 7 drivers in this directory, **every one is a port, with no retirement
available on coverage grounds** -- `UPSTREAMS.lock`'s `retire_at` for the whole `[karma]`
subtree is consistent with that: this directory's code is ADAPTED and tested anew, not carried
forward as-is. `ode_system_test.C`'s CN rate-2 assertion is still useful evidence: it
pins what a correct Crank-Nicolson port must reproduce (order 2, not more, not less), even though it
cannot be reused as a bit-identical oracle for `crank_nicolson.{h,C}`'s own tableau.

KARMA's own build compiles the 22 tableau files plus `karma_ts_base.C`, `rk.C` and `rosenbrock.C`
(`KARMA_SOURCES` in `src/CMakeLists.txt`) but none of the drivers
(`dopri45_integrator.C`, `embedded_dirk_integrator.C`, `explicit_rk_integrator.C`,
`implicit_dirk_integrator.C`, `petsc_temporal_integrator.C`, `pseudo_time_integrator.C`,
`rosenbrock_integrator.C`, `temporal_integrator.C`); this is why no test reaches them. The
tableaus and the three base classes are the live part of the port; the drivers are a design
reference only, and the Section 4 design replaces them rather than adapting them. The rank-divergent
accept/reject in `dopri45_integrator.C:243,258` (Section 4.6) was therefore never executed in
KARMA; it is a defect the fuse port must not reproduce, not a defect in KARMA's shipped behaviour.

## 2. Adaptive stepping: three independent mechanisms, not one

KARMA never settled on a single step-size controller. Four live in this directory:

1. **Dynamical-time-scale control** (`ExplicitRKIntegrator::core_integrator`,
   `explicit_rk_integrator.C:213-296`; `PseudoTimeIntegrator::core_integrator`,
   `pseudo_time_integrator.C:243-` onward): `hnew = safe * dynmax * (prev_dynmax/dynmax)^0.2 *
   (h/hold)^0.2`, clipped to `[facl*h, facr*h]` and to `hmax`, where `dynmax` comes from
   `interface->compute_dynamical_timescales()` -- a **physics callback**, not a truncation-error
   estimate. This is a PI-shaped formula (two ratio exponents, 0.2 each) applied to a
   problem-supplied scale rather than to an error norm; it needs no embedded tableau at all (it
   runs even for plain `FE`/`BE`).
2. **DOPRI45's PI (Gustafsson) controller** (`dopri45_integrator.C:198-276`): the textbook
   Hairer/Wanner/Norsett `dopri5.f` algorithm -- `expo1=0.2-beta*0.75`, Lund-stabilised
   `fac = pow(err,expo1)/pow(facold,beta)`, `facold` carried step to step, hardcoded exponents
   rather than derived from `n_order()`/`n_secondary_order()` (Section 6.1 explains why that
   matters). `err` is an rtol/atol-scaled root-mean-square over the whole vector, `sqrt(sum((E.k /
   (rtol*|y|+atol))^2)/n)`.
3. **The generic embedded-DIRK I-controller** (`embedded_dirk_integrator.C:328-390`): a *different*
   formula, `order_power = 1/(min(order,secondary_order)+1)` **derived from the tableau's own
   metadata** (the one place in this directory that does what `estimate_error()` was presumably
   meant to centralise), `newdelt = safe*h*pow(LTE_tolerance/local_error, order_power)`, no
   Lund/PI memory term (the PI extension is commented out, line 351-352, with a citation to
   "Hairer: IV: Stiff problems - Page 124" left as a TODO). `local_error` is a single scalar
   (`VecNorm`, collective), not a per-component scaled vector.
4. **Rosenbrock's own controller**, inside `RosenbrockIntegrator::core_integrator` past line 320
   (structurally the same `E`-weighted local-truncation-error pattern
   as (3), reusing the assembled Jacobian already factored for the step).

None of the four derives its safety factor, its exponent, or its accept/reject threshold from a
single shared place; `set_adaptive_params()`'s defaults (`safe=0.9` for DOPRI45's own constructor,
`temporal_integrator.C:34-46` gives `safety=0.8`/`facr=4.0`/`facl=0.2` as the GENERIC default --
two different "default safety factor" values live in the same directory, 0.8 vs 0.9) are the only
shared state, and even those are re-overridden per integrator constructor. **Dense output**: none
of the four controllers produces one -- there is no continuous/interpolated solution between
accepted steps anywhere in this directory; `Function()` is only ever evaluated at stage points.
Min/max step: `hmax` only (from `set_adaptive_params`); there is no `hmin` anywhere -- the only
floor is the generic `0.1*fabs(h) <= fabs(time)*uround` exit check in DOPRI45
(`dopri45_integrator.C:151-155`, returns error code `-3`), which the other three controllers do not
have at all (an unbounded-shrinking step in `EmbeddedDIRKIntegrator` has no refusal path).

## 3. Dependency mapping: include graph to fuse::linalg/fuse::num

Grepped every `#include` in `numerics/temporal/{include,src}` (MEASURED, 49 distinct headers);
outside the directory itself, the non-PETSc dependencies are: `ikarma.h` (CRTP registration base),
`karma_vector.h` (`KARMAVector`, a `Vec` wrapper with `.close()`/`l2_norm()`/`vec()`), `karma.h`
(aggregator), `karma_context.h` (`KARMAContext`: `time`, `delt`, `rank`, a scratch `Work_2` vector),
`karma_interface.h` (`KARMAInterface`, a singleton -- see below), `karma_monitor.h`, `karma_utils.h`
(`KARMAUtils::sign`, `KARMA::min/max/pow` templates), `karma_wrappers.h`, `parallel.h`
(`Parallel::barrier()`), plus the PETSc headers `petscsnes.h`/`petscts.h`.
`temporal_integrator.h:23` carries the author's own note next to the `petscsnes.h` include:
`// replace by karma_nonlinear_solver.h` -- the 2007-2011 author already wanted the abstraction
`fuse::num::solve()` now provides.

**`KARMAInterface`'s problem-facing surface** (`include/interface/karma_interface.h`, grepped lines
70-177), the actual "Problem" contract every driver above calls through:

| KARMA method | Signature (abbrev.) | Used by |
|---|---|---|
| `step_preprocess`/`step_postprocess` | `void()` | all 5 steppable drivers, around every accepted step |
| `stage_preprocess`/`stage_postprocess` | `void()` | `ExplicitRKIntegrator`, `DOPRI45Integrator`, around every stage |
| `compute_dynamical_timescales` | `(old, curr, delt, out vector<Real>&)` | mechanism 1 above |
| `synchronize_physics_solution` | `()` / `(const KARMAVector&)` | `EmbeddedDIRKIntegrator` after a step |
| `mass_multiply` | `(const KARMAVector& X, KARMAVector& Y)` | `ImplicitDIRKIntegrator`, `RosenbrockIntegrator`, `PseudoTimeIntegrator` -- generalised-ODE (`M y' = f(y)`) support |
| `system_residual` | `(const KARMAVector& X, KARMAVector& R)` | the stage residual `F` |
| `jacobian_matrix` | `(const KARMAVector& X, KARMAMatrix& J)` | `RosenbrockIntegrator` only -- an **assembled** Jacobian |
| `mass_matrix_subtract` | `(KARMAMatrix& P, Real const_multiplier=1.0)` | `RosenbrockIntegrator`: forms `J - c*M` in place |

### 3.1 fuse::linalg / fuse::num: what exists today

- **Exists.** `DistVector`/`DistSparseMatrix` (CSR, row-partitioned, `apply()` = SpMV) and
  `MatrixFreeOperator`, both satisfying the SAME `LinearOperator` concept gmres/pcg take
  (`operator.hpp:69-90`) -- no "assembled vs. matrix-free" fork at the solver level. `gmres`/`pcg`
  Krylov solves with `Preconditioner`/`IdentityPreconditioner`/`JacobiPreconditioner`
  (`preconditioner.hpp`). `fuse::num::solve()`: a complete Jacobian-free Newton-Krylov driver
  (MFFD Jacobian action, GMRES, Eisenstat-Walker forcing, line search, named convergence codes,
  lagged block-Jacobi preconditioning) ALREADY dispatchable over a `NewtonSubstrate::{Native,Petsc}`
  registry (`newton_krylov.hpp:420-504`) -- this is exactly KARMA's `TemporalIntegrator`/
  `PetscTemporalIntegrator` split, one layer up, already built and measured on `dn_heat` (order 2.0 +/- 0.24) and the Brusselator window (50/50 convergence).
  `SchurSlabSolver`: an exact direct solver, but scoped to one operator family
  (`A = I - coefficient*L_h`, the periodic 5-point Laplacian), not general. The derivative gate
  (`derivative_gate.hpp`) already supplies exactly the verification primitives a ported
  Jacobian action needs: `taylor_remainder_order()` (checks a claimed `J v` against the residual's
  own Taylor remainder) and `matrix_free_agreement()` (checks an assembled vs. matrix-free `apply`
  against each other) -- both DERIVED bounds, not fitted constants, consistent with the project's
  rule that tolerances are derived.
- **Missing.** (a) A **Jacobian-assembly interface** -- nothing in `fuse::linalg`/`fuse::num` builds
  a `DistSparseMatrix` FROM a residual function; every existing Newton-Krylov path is matrix-free by
  construction. (b) A **general direct (LU/Cholesky) sparse solve** -- only `SchurSlabSolver`'s
  dense `LDLT` on one specific Schur complement exists; there is no general assembled-and-factored
  path. (c) A **mass-matrix / DAE-shaped residual convention** -- `fuse::num::ResidualFn` is
  `F(x) -> DistVector`, with no first-class notion of `M dy/dt = f(y)` the way KARMA's
  `mass_multiply` gives one. (d) A **non-autonomous Jacobian correction (`df/dt`)** hook for
  Rosenbrock methods -- nothing today.
- **Layering constraint (read `tools/check-layering.py`'s own `LAYERS` list, MEASURED):**
  `base < linalg < model < ir < schedule < num < field < ... `. `fuse::num` sits BELOW `fuse::field`
  (where a value-typed `BasicFieldRef`/`FieldRef` would live) and may depend on `model`/`linalg`
  but never on `field` or `runtime`. The `Integrator<Value, Problem>` template family therefore
  cannot name `fuse::FieldRef` or `fuse::Component` -- it must be generic over `Value`
  (`model/value_traits.hpp`'s `CoupledValue` concept, which DOES sit in `model`, below `num`, so
  `num` may depend on it) and over a `Problem` concept expressed in plain spans/`DistVector`, the
  same discipline `newton_krylov.hpp`'s own `ResidualFn` already uses.

### 3.2 How a Rosenbrock stage-solve maps onto the EXISTING (not a new) Newton-Krylov machinery

Rosenbrock's stage equation is **linear** in the stage unknown (`(I/(h*gamma_ii) - J) k_i = ...`),
with the SAME Jacobian `J` (evaluated once, at `t_n, y_n`) reused across every stage, scaled only by
each stage's own `gamma_ii`. This is exactly what `MffdJacobianAction::set_point()` +
`apply()` already gives for free: one directional-derivative action fixed for the whole step, scaled
per stage by a caller-side constant, solved by `gmres` over a `MatrixFreeOperator` wrapping
`v -> v - h*gamma_ii*(J v)`. No assembled matrix is needed for Rosenbrock **as long as the linear
solve is driven to a tight-enough tolerance that the order is not degraded** -- a genuine risk
(Section 6.2), not a free adaptation.

## 4. Design for fuse::num

### 4.1 `Integrator<Value, Problem>` and the `TemporalProblem` concept

```cpp
namespace fuse::num {

// Value: double in production; dual<S, N> (fuse/num/dual.hpp, not yet in the tree)
// under a tangent-mode sensitivity sweep. Constrained by model::CoupledValue
// (value_traits.hpp), which already lives at the "model" layer, below "num".
template <class T>
concept TemporalProblem = requires(T& p, double t, std::span<const double> y,
                                    std::span<double> out) {
  { p.rhs(t, y, out) } -> std::same_as<Result<void>>;
};

// Opt-in extensions, detected structurally (no inheritance, matching PreconditionerBlock's
// own duck-typed convention, newton_krylov.hpp:302):
//   p.mass(y_dot, out)              -- generalised ODE M y' = f(y); absent means M = I
//   p.jacobian_action(x, v, out)    -- J(x) v; absent means fall back to MFFD on rhs()
//   p.dfdt(t, y, out)               -- non-autonomous correction; absent means autonomous
//   p.is_stiff()                   -- a hint, not a refusal; the controller still measures

struct ButcherTableau {
  int stages;
  int order;
  std::optional<int> embedded_order;     // nullopt: FE, Kutta22/44, SDIRK22/23/33/34/44/45, ROS22
  std::vector<std::vector<double>> A;
  std::vector<double> B, C;
  std::optional<std::vector<double>> E;  // B - B_hat; present iff embedded_order is
  bool is_A_stable, is_L_stable, is_S_stable, is_FSAL;
};

template <CoupledValue Value>
struct StepRecord {
  Value t_new, h_used, h_next;
  std::optional<Value> error_norm;   // nullopt for a non-embedded method
  bool accepted;
  std::size_t function_evaluations, jacobian_evaluations, linear_iterations;
};

template <CoupledValue Value, TemporalProblem Problem>
class Integrator {
 public:
  virtual ~Integrator() = default;
  virtual Result<StepRecord<Value>> step(Problem&, Value t, Value h,
                                          std::span<Value> state) = 0;
  virtual const ButcherTableau& tableau() const = 0;
};

}  // namespace fuse::num
```

Explicit methods (FE, Kutta22/44, DOPRI45) implement `step()` directly over `rhs()`.
Diagonally-implicit methods (BE, CN, SDIRK*, TRBE, embedded DIRK) implement each stage's nonlinear
solve by calling `fuse::num::solve()` with a `ResidualFn` closed over `(t_n + c_i h, A, known stage
values)` -- REUSING the existing Newton-Krylov driver and its substrate registry, not a new
nonlinear-solve layer (this is "adapted, not ported" in concrete form: KARMA's own
`SNESSolve(_snes, Zi, _curr_sol)` call becomes one call to `fuse::num::solve()`). Rosenbrock
methods implement `step()` as one `MffdJacobianAction::set_point()` per step followed by `stages`
many `gmres()` calls over scaled `MatrixFreeOperator`s (Section 3.2) -- no Newton iteration at all,
consistent with Rosenbrock's own definition (linearise once, solve the resulting LINEAR stage
system).

### 4.2 The error controller as a separate, reusable component

```cpp
template <CoupledValue Value>
struct ControllerState { Value fac_old; int consecutive_rejections = 0; };

enum class ControllerKind { PI, I };   // Section 2's mechanisms (2) and (3), named, not conflated

template <CoupledValue Value>
class EmbeddedController {
 public:
  EmbeddedController(ControllerKind kind, int order, int embedded_order,
                      double safety, double fac_min, double fac_max, double beta /* PI only */);

  // err_norm: ALREADY the rtol/atol-scaled RMS the caller computed (Section 4.4) -- this
  // class never touches a solution vector, only the scalar norm and the controller's own
  // recurrence state, so it is Value-generic without needing to know the state layout.
  [[nodiscard]] Result<StepDecision<Value>> propose(Value err_norm, Value h,
                                                     ControllerState<Value>& state) const;
};

struct StepDecision { bool accept; double h_next; };
```

This is a NAMED split KARMA never made (Section 2): one controller type, parameterised by
`ControllerKind`, replaces DOPRI45's hand-inlined PI formula and `EmbeddedDIRKIntegrator`'s
hand-inlined I formula, with `order`/`embedded_order` read from the SAME `ButcherTableau` the step
function uses -- eliminating the DOPRI45 metadata bug (Section 6.1: its hardcoded `expo1=0.2`
happens to be right only because nobody ever asks `n_secondary_order()` for the real exponent).

### 4.3 Step size and error norm when `Value` is `dual<S, N>`

A sub-step inside a sensitivity sweep carries both a primal value and `N` tangent components
(`dual<S,N>`). Two quantities need a stated convention:

- **Step size `h`.** Always primal-only (`dual<S,N>::value()` / `S`, never a dual itself) -- a
  step size with a nonzero tangent would mean "the schedule point moves as a function of the
  sensitivity parameter," which is a DIFFERENT and much harder problem (shape/parameter-dependent
  meshing of time) that nothing in this design asks for. `h` stays an `S` (matching
  `model/value_traits.hpp`'s `value_traits<T>::scalar_type`) even when the state is `dual<S,N>`.
- **Error norm.** Computed over the PRIMAL components only, for the SAME reason DOPRI45's and
  EmbeddedDIRK's error estimate is a physical-accuracy measure, not a sensitivity measure: a
  tangent component blowing up is a conditioning/stability question for the forward sensitivity
  equation, not a question the step-size controller (which only ever sees `S`, per the point above)
  is equipped to answer. This is decision 1 in Section 7; it determines whether a stiff sensitivity direction can silently
  degrade accuracy without ever triggering a rejection.

### 4.4 Contract with the coupling window

`fuse::model::iteration.hpp`'s existing `DtPolicy` enum (`iteration.hpp:530-545`, READ, not
designed here) already states the two policies a component's sub-stepping can have relative to a
window end: `Fixed` (straddle the boundary, roll back and recompute against the partner's converged
data next window) and `Adjustable` (shorten the last sub-step to land exactly on the window end, "no
straddling step and no extrapolation ever needed", `iteration.hpp:535-538`). An adaptive sub-stepper
under `DtPolicy::Adjustable` therefore needs exactly one more operation beyond `step()`:

```cpp
// Called once the controller has proposed h_next; clips it to the window end WITHOUT
// re-running the controller. This is KARMA's own `if (time+hnew > time_end) hnew = time_end-time`
// (present, independently, in explicit_rk_integrator.C:264-265, embedded_dirk_integrator.C:
// 388-389, dopri45_integrator.C has the equivalent `last` flag mechanism, lines 149-152) --
// generalised into one named operation instead of three ad hoc copies.
[[nodiscard]] Value clip_to_window_end(Value h_next, Value t, Value window_end) const;
```

What the controller does AT the clip: nothing extra -- the clipped step is still taken and still
subject to the SAME accept/reject test as any other step (a clipped step that fails its error check
is rejected and retried smaller, exactly like an unclipped one; KARMA's own code already does this,
since the clip happens to `hnew`, the PROPOSED next step, before it is taken, never to a step already
in flight). `DtPolicy::Fixed` needs no clip at all -- its contract is the straddle-and-roll-back one
`iteration.hpp:531-534` already names, which is a SCHEDULING decision (whether the schedule even
offers early termination of a sub-step), not an integrator-controller one; `Integrator::step()`
itself does not need to know which `DtPolicy` its caller is using.

**Sub-step history for the waveform/temporal-basis layer** (read `include/fuse/num/waveform.hpp`,
`temporal_basis.hpp`): `Waveform::push(Rational t, const double* values)`
(`waveform.hpp:216`) wants exact-`Rational` sample times and flat `double*` rows -- it is NOT
templated on `Value` today (consistent with value typing not being in the tree: `FieldRef` defaults to
`double`). A component driving an adaptive sub-stepper under `Adjustable`
must therefore: (1) convert each accepted sub-step's primal state to `double` before `push()` (a
`dual<S,N>`'s tangent components have nowhere to go in a `Waveform` today -- decision 2 in Section 7); (2)
convert the sub-step's OWN time (an `S`/double from Section 4.3) to an exact `Rational` before
calling `push()`, the same "form the exact difference before going to double" discipline
`temporal_basis.hpp:19-22` already states for the OTHER direction (`Rational` to `double`) -- for a
sub-step time that is the sum of many floating `h`'s, this is not free: the integrator's own `time`
accumulator (KARMA carries this as a `PetscReal`, `temporal_integrator.h:144`) would need to become
a `Rational` running total itself, counting sub-steps as exact fractions of the window, or every
`push()` re-derives its `Rational` time from the window's own exact bounds plus an integer sub-step
count -- the latter only works for a FIXED sub-step count known in advance, which an adaptive
sub-stepper by definition does not have. Section 7 (decision 3) resolves this.

### 4.5 Refusals, named

Every one of these is a REFUSAL (a named, tested failure path), not a silent clamp or a crash:

| Name | Condition | KARMA precedent |
|---|---|---|
| `E-INTEGRATOR-STEP-TOO-SMALL` | proposed `h` below a floor derived from `uround`/`time` (DOPRI45's own `0.1*fabs(h) <= fabs(time)*uround`, `dopri45_integrator.C:151`) | returns `-3` |
| `E-INTEGRATOR-MAX-STEPS` | `nstep > nmax` | returns `-2` |
| `E-INTEGRATOR-NONFINITE-ERROR` | the embedded error norm is NaN/Inf -- **new**: KARMA never checks this; an Inf `err` compares `false` to `err<=1.0` in C++ and is silently treated as a REJECT by accident, not by design | none (gap) |
| `E-INTEGRATOR-NEWTON-FAILED` | the per-stage `fuse::num::solve()` call returns a non-`Converged` status | KARMA logs and continues (`embedded_dirk_integrator.C:251-256`, `newton_fail` is set but never acted on -- a genuine KARMA defect, Section 6.1) |
| `E-INTEGRATOR-STIFF-UNDETECTED-NONSTIFF-METHOD` | DOPRI45's own stiffness detector trips 15 times | returns `-4` |

### 4.6 Collective agreement on accept/reject

**This is the most important correctness requirement of the port** (Section 6.1 has the
measured defect it fixes). KARMA's `DOPRI45Integrator` computes its error norm by hand:
`VecGetArray` on every rank, a LOCAL sum `lnrm`, then `MPI_Reduce(&lnrm, &err, 1, MPI_DOUBLE,
MPI_SUM, 0, KARMA_COMM_WORLD)` -- **root only** (`dopri45_integrator.C:243`) -- followed by `err =
sqrt(err/n)` and the accept/reject branch `if (err <= 1.0)` (`dopri45_integrator.C:245,258`), which
every rank evaluates on its OWN copy of `err`. Non-root ranks never receive the reduced value (that
needs `MPI_Allreduce`, not `MPI_Reduce`); every rank but 0 evaluates the branch on whatever `err` held
before the call (`0.0`, set at line ~208), i.e. **every non-root rank always accepts**, independent of
the root's decision. This is exactly the hazard the project's collective-path rule names ("a collective path
must never return on a locally computed condition: agree first, then every rank refuses with the
same code") -- except here it is worse than a locally computed condition, it is an UNDEFINED one.
`EmbeddedDIRKIntegrator`'s own error norm (`VecNorm(k[0], NORM_2, &local_error)`) is SAFE by
accident: PETSc's `VecNorm` is internally collective and returns the identical value on every rank,
so that controller's `if (local_error > LTE_tolerance)` branch, though it looks like the same
"locally computed condition" pattern, is not, because the input to it is already rank-uniform.

**fuse's design closes this by construction, not by a review rule**: `DistVector::norm2()`/`dot()`
already reduce across every participant (`newton_krylov.hpp`'s own header comment, quoted in
Section 3.1) -- so the ONE reduction this design needs is: every embedded error norm is computed
through `DistVector::norm2()` (or a to-be-added weighted-RMS norm following `residual_norm.hpp`'s
existing `NormKind::MeasureWeightedRms` convention, Section 6.4), never through a hand-rolled
`MPI_Reduce`. The accept/reject `bool` and the proposed `h_next` are then PURE functions of that
already-agreed scalar, computed identically (same arithmetic, same inputs) on every rank -- the same
property `newton-krylov.md`'s own rank-agreement argument already proves for the Newton loop's
convergence tests. No new reduction is introduced; the existing `DistVector` reduction is reused for
one more purpose. This REPLACES, rather than wraps, KARMA's own `Parallel::barrier()`-after-every-step
pattern (`explicit_rk_integrator.C:292,307`, `dopri45_integrator.C`'s own loop tail) -- a barrier
enforces timing agreement, not DATA agreement, and does nothing to fix the DOPRI45 defect above (all
ranks reach the barrier, just with different `err` values already acted on).

## 5. Verification plan

Every tolerance below is DERIVED and printed next to the test that uses it; none is picked after a run and then widened.

1. **Per-method convergence test**, one per row of the Section 1.2 table: Dahlquist
   `y' = lambda*y` for a NONSTIFF `lambda` (complex, `Re(lambda)<0`, e.g. `lambda = -1 + 5i` to
   exercise the method off the real axis) with closed-form `y(t)=y0*exp(lambda*t)`, and a STIFF
   `lambda` (e.g. `-1e5`) for every method whose own `is_S_stable()`/`is_A_stable()` claims it should
   survive it. Measure the observed order by least-squares slope of `log(error)` vs `log(h)` over a
   decade of `h` ABOVE the round-off floor (the SAME floor-identification discipline
   `least_squares.hpp`'s own header comment already uses for the DG fit, Section 3/no new idea
   needed) -- the asserted bound is `|observed_order - tableau_order| <= tolerance`, with
   `tolerance` derived from the SAME asymptotic-error-constant argument `waveform.hpp`'s own
   `kMaxDgDegree` comment uses (the fit's conditioning sets a floor the slope cannot resolve below),
   not a number chosen because a first run came out at e.g. `0.3` and `0.3` looked small enough.
2. **A second, non-Dahlquist problem per order class**: one nonstiff nonlinear ODE with a known
   closed form (e.g. the logistic equation) and one stiff nonlinear problem (e.g. the Robertson
   problem, the field's standard stiff-ODE benchmark with a well-documented reference solution) --
   Dahlquist alone cannot distinguish a correctly-ported nonlinear Jacobian path from one that
   happens to work only because `lambda` is a scalar.
3. **The KARMA differential test**: NOT a live run of KARMA (per Section 1.4 no KARMA test exercises this directory) -- instead, the Butcher
   coefficients copied BYTE-FOR-BYTE from the vendored `.C` files (already extracted into Section
   1.2/this document, each with a file:line citation) are hardcoded into both an independent
   reference script and the `ButcherTableau` the fuse port builds; for EXPLICIT methods (FE,
   Kutta22/44, DOPRI45) the arithmetic has no solver-ordering freedom, so the fuse result must match
   the reference to the LAST BIT on the SAME problem and step sequence. For IMPLICIT methods
   (SDIRK*, TRBE, embedded DIRK) the adaptation licenses a BOUNDED difference, not bit-identity: KARMA solves
   each stage with `SNES`'s own line search and Krylov defaults, fuse solves it with
   `fuse::num::solve()`'s own (different) Eisenstat-Walker forcing and line search -- the two Newton
   iterations converge to the same root to WITHIN their respective `atol`/`rtol`, so the bound is
   `|y_fuse - y_karma_reference| <= rtol*|y| + atol` (both integrators' own stage-solve tolerances,
   summed), not zero. For Rosenbrock, the bound is the SAME shape but driven by the GMRES
   tolerance on the stage's linear solve rather than a Newton tolerance.
4. **Mutation rows** (one per tableau, run against the convergence test in (1) to confirm it would
   actually fail): perturb one `A`/`B`/`C` coefficient by `1e-3` (relative) and confirm the measured
   order DROPS below the tableau's own order minus the derived tolerance from (1) -- a mutation that
   does not move the measured order means the convergence test in (1) is not sensitive to that
   coefficient and is itself a gap, not a pass. Separately: halve the controller's `safety` factor
   and confirm the number of ACCEPTED steps changes (the controller is exercised, not dead code);
   force a rejection path (set an artificially tight `rtol`) and confirm `E-INTEGRATOR-STEP-TOO-SMALL`
   fires for at least one method per family -- a refusal counts as verified
   only when a test makes it fire.
5. **Jacobian-action verification** (Rosenbrock, DIRK): `taylor_remainder_order()` and
   `matrix_free_agreement()` from `derivative_gate.hpp` (already built, Section 3.1), run against
   every `Problem` the integrator test suite defines -- no new verification primitive needed, only
   new call sites.
6. **Tier 0 vs. needs-MPI**: every item above runs at tier 0, single rank, no PETSc required for the
   Native substrate. The PETSc substrate (`PetscTemporalIntegrator`'s analogue, if ported per Open
   question 4) and any multi-rank `DistVector`/collective-agreement check (Section 4.6) need MPI;
   they run on a host with a working MPI launcher, as `mpirun -np 2 <test-binary>` and
   `mpirun -np 3 <test-binary>`.

## 6. Risks

### 6.1 Defects found in the vendored source that a literal port would carry forward

These are MEASURED, not suspected, and none should be reproduced in the fuse port:

- **DOPRI45's rank-divergent accept/reject** (`dopri45_integrator.C:243,258`, detailed in Section
  4.6): `MPI_Reduce` to root only, then every rank branches on its own unreduced copy. Fixed by
  construction once the error norm goes through `DistVector::norm2()` (Section 4.6) -- but worth
  recording as a standalone finding, since it is exactly the hazard class the project's
  collective-path rule exists to prevent, found in exactly the kind of code (a 2007-2011 import) the
  rule anticipates.
- **GRK4A's embedded error estimate is identically zero**: `E[i] = B[i] - B[i]` for every stage
  (`grk4a.C:71`, all four terms literally subtract the SAME constant from itself) because the
  vendored "d prime" (lower-order) coefficients copied into the source comment
  (`grk4a.C:60-64`) are IDENTICAL to the "d" (higher-order) ones directly above them
  (`grk4a.C:55-59`) -- compare with `ROS22` (`ros22.C:47-52`), whose comment legitimately states
  "d prime" equals "d" because ROS22 is a genuinely non-embedded method (category `ROSENBROCK`, not
  `ROSENBROCKEmbedded`) and never defines `E` at all. GRK4A, by contrast, IS tagged
  `ROSENBROCKEmbedded` (`grk4a.C:22`) and DOES define `E`, just identically zero -- a transcription
  defect, not a design choice. The authoritative source is Kaps & Rentrop's original 1979 tableau
  (*Numerische Mathematik* 33), not this file's own "d prime" row; porting GRK4A correctly means
  re-deriving its embedded coefficients from the literature, not translating `grk4a.C` literally.
- **`EmbeddedDIRKIntegrator` detects but ignores Newton failure**: `newton_fail` is set when
  `SNESGetConvergedReason` is negative (`embedded_dirk_integrator.C:251-256`) and then never read
  again in the function -- the step proceeds to use `_curr_sol` regardless. The fuse port's
  `E-INTEGRATOR-NEWTON-FAILED` refusal (Section 4.5) is the fix, not a literal port of this path.
- **DOPRI45's secondary-order metadata is wrong** (Section 1.2 table): `RK(7, 5, ...)` leaves
  `_secondary_order = 5` (the 4-argument constructor sets it equal to the primary order,
  `rk.C:59-62`) even though the true embedded order is 4 (confirmed: the `E` coefficients
  `71/57600, 0, -71/16695, 71/1920, -17253/339200, 22/525, -1/40`, `dopri45.C:23-24`, are the
  literature's own DOPRI5(4) error-estimation coefficients). `DOPRI45Integrator`'s controller never
  asks `n_secondary_order()` (it hardcodes `expo1=0.2`, which IS correct for order 4 embedded in
  order 5 -- `1/(4+1)=0.2` -- by luck of the hardcoding, not because the metadata was consulted and
  right). A fuse `EmbeddedController` that reads `embedded_order` from the tableau MUST carry the
  CORRECT value (4), not the vendored `_secondary_order` (5), for DOPRI45; every other embedded
  method's `n_secondary_order()` is self-consistent and safe to read directly (spot-checked RODASP,
  SDIRK543/643, TRBE, SDIRK332 above -- all have a 3-argument `RK(...)` constructor call that sets a
  genuinely distinct secondary order).
- **`AInv` (the Butcher `A` inverse) is computed/declared but never consumed**: `RK::AInvcoeff()`
  exists (`rk.h:143-146`), is SET for BE/CN/SDIRK22/23/34 (grepped, Section 1 above), but no
  integrator driver ever calls `AInvcoeff()` (grepped `src/numerics/temporal/*.C`, zero hits outside
  the setters themselves) -- vestigial state; nothing in the fuse `ButcherTableau` needs an `AInv`
  field unless a NEW use is identified (none is, in this directory).
- **`crank_nicolson.{h,C}`'s `#ifdef IMPLICIT_MIDPOINT` branch is permanently dead** (the macro is
  never defined anywhere in the build, grepped `CMakeLists.txt`/`*.cmake`/the whole tree) and, if it
  were ever enabled, would leave stage-1 of its 2-stage `A`/`B`/`C` arrays uninitialised
  (`RK::init()` allocates with plain `new Real[n]`, which does not value-initialise,
  `rk.C:45-56`) -- a latent bug in dead code. The fuse port implements implicit midpoint, if wanted,
  as its OWN named 1-stage method, never as a compile-time variant of CN sharing its allocation.

### 6.2 Risk: inexact (Krylov) linear solves inside a Rosenbrock stage may degrade formal order

The ROS methods KARMA ports (as opposed to ROW/W-methods) are classically analysed assuming an
EXACT linear solve per stage (Hairer & Wanner, *Solving ODEs II*, IV.7); an inexact GMRES solve
driven to a loose tolerance can silently reduce the observed order below the tableau's own claim
without tripping any of KARMA's own refusals (which only ever check the EMBEDDED error estimate,
never the linear-solve residual itself). The verification plan's convergence test (Section 5.1)
will CATCH this (an order below the derived tolerance band fails the test), so the per-stage GMRES
tolerance is set RELATIVE to the step's own embedded-error tolerance (decision 5 in Section 7)
rather than discovered empirically per method.

### 6.3 Risk: template instantiation cost

`Integrator<Value, Problem>` instantiated over `Value in {double, dual<S,1>, dual<S,2>, ...}` and
every `Problem` type in `physics-kit` multiplies the number of instantiated `step()` bodies (each
containing a full stage loop, for the higher-stage-count methods such as RODASP's 6 stages or
DOPRI45's 7) by `|Value| x |Problem|`. This is the SAME shape of cost `fuse::num::solve()` already
accepts for its own `ResidualFn`/`PreconditionerApplyFn` type erasure (`std::function`-based, not
templated per problem) -- so `Integrator` follows that same
type-erasure discipline (decision 6 in Section 7): one compiled `step()` per `Value`, problems
type-erased through a `std::function`-based `TemporalProblem` adapter rather than a fully templated
`Problem`, trading a small amount of per-call indirection for a bounded instantiation count
independent of how many physics kits exist.

### 6.4 Risk: the exact-rational schedule and a sub-step time accumulator

Section 4.4 names this: an adaptive sub-stepper accumulates `time` in
`Value`'s own scalar type (floating point), the same way KARMA does
(`temporal_integrator.h:144-150`, `PetscReal time`), while `Waveform`/the schedule itself work in
exact `Rational` (`waveform.hpp:29-35`). Converting a floating accumulated time back to an exact
`Rational` for `push()` is not a well-defined operation in general (a `Rational` constructed from an
arbitrary `double` is exact FOR THAT DOUBLE, but that double is itself already an accumulation of
rounding error over many sub-steps) -- so the integrator's time is itself a `Rational`
accumulator (Section 7, decision 3) rather than a float converted at `push()`.

## 7. Decisions

1. **Error norm under `dual<S,N>`** (Section 4.3). The step-size controller reads only the primal
   component of the embedded error, never a tangent component. The alternative (control on a
   combined primal+tangent norm) conflates two different notions of "the step is accurate" and has
   no precedent in KARMA or the literature.
2. **`dual<S,N>` sub-step history into `Waveform`** (Section 4.4). `Waveform::push()` is
   `double`-only. A component running a sensitivity sweep writes only the primal `Waveform` and
   keeps tangent sub-step history in a separate, integrator-owned buffer the coupling layer never
   sees. Templating `Waveform` on `Value` waits for a concrete consumer: adaptive sub-steps happen
   inside a fixed coupling window only, so tangent values do not yet cross a window boundary.
3. **Time accumulation for adaptive sub-steps** (Sections 4.4 and 6.4). The integrator's time is a
   `Rational` accumulator with a recorded per-window quantum, and the last step is clipped exactly
   to the window end. This preserves the "time is exact" invariant `waveform.hpp:29` states
   unconditionally, at the cost of the integrator rounding its own floating controller output to
   the quantum.
4. **No PETSc `TS` substrate.** `PetscTemporalIntegrator` (Section 1.3) is a second implementation
   of every method via PETSc's own `TS`. physics-kit is tier 0 with PETSc optional, and `fuse::petsc`
   already exists for the nonlinear solve (Section 3.1), so a `TS`-substrate integrator would be a
   third PETSc integration point. It is out of scope until a specific physics-kit component needs
   PETSc's `TS`.
5. **Per-stage linear-solve tolerance for Rosenbrock** (Section 6.2). The stage GMRES `rtol` is the
   step `rtol` times a fixed factor, starting at `1e-2`; it is held to the per-method order test and
   tightened if that test demands it, never loosened.
6. **Type erasure for `Problem`** (Section 6.3). `Problem` is type-erased through a
   `std::function`-based `TemporalProblem` adapter, matching `fuse::num::solve()`'s `ResidualFn`
   convention, so the `Integrator<Value, Problem>` sketch in Section 4.1 is realised as
   `Integrator<Value>` over `TemporalProblem<Value>`.

## 8. What is implemented, and how it is verified

Header-only, under `include/fuse/num/`: `butcher_tableau.hpp`, `embedded_controller.hpp`,
`integrator.hpp`, `explicit_integrator.hpp`, `dirk_integrator.hpp`, `rosenbrock_integrator.hpp`,
`order_conditions.hpp`, `ssp_coefficient.hpp` and `continuous_extension.hpp`, with tests under
`test/num/` and mutation rows in `build-evidence/mutations/integrators.toml` and
`integrators-extended.toml`.

- `butcher_tableau.hpp`: the `ButcherTableau` struct (Section 4.1) plus the tableaus as factory
  functions, each coefficient cited to its vendored `file:line`.
- `embedded_controller.hpp`: `EmbeddedController` (Section 4.2), the PI (DOPRI45's Gustafsson
  controller) and I (embedded DIRK's controller) kinds behind one entry point, `propose()`, a pure
  function of an already-agreed scalar error norm.
- `integrator.hpp`: the `IntegratorValue` concept (a local mirror of the `CoupledValue` /
  `value_traits<T>::scalar_type` shape), the type-erased `TemporalProblem<Value>` adapter, the
  `Integrator<Value>` interface every family implements, and `SubStepClock` (the `Rational` time
  accumulator of decision 3, `kDefaultQuantumCount = 2^20`, derived in the class's doc comment).
- `explicit_integrator.hpp`: `ExplicitIntegrator<Value>`, one class over any `Erk`/`ErkEmbedded`
  tableau (FE, KUTTA22, KUTTA44, DOPRI45, the SSP methods).
- `dirk_integrator.hpp`, `rosenbrock_integrator.hpp`: the implicit families (below).

### 8.1 Corrections to the vendored tableaus

A literal port would carry these vendored defects forward; each is corrected and pinned by a test.

- **GRK4A's embedded estimate is identically zero** (`grk4a.C:71`). `grk4a_tableau()` sets
  `embedded_order = std::nullopt` and `ButcherTableau::has_embedded_estimate()` reports `false`
  (`ButcherTableau.Grk4aEmbeddedEstimateIsMeasuredZeroNotAssumed`). The "d prime" row is left
  absent until it can be transcribed from Kaps and Rentrop (1979) reliably.
- **DOPRI45's `is_FSAL()` is hardcoded `false`** although the tableau has the textbook FSAL
  structure (row 6 of `A` bit-identical to `B`, `C[6] == 1.0`); `dopri45_tableau()` carries `true`
  (`ButcherTableau.DeclaredFsalMatchesTheStructuralProperty`, run over every tableau). Its embedded
  order is 4, not the vendored 5.
- **SDIRK332's `C[1]` is a transcription error.** The vendored `24/108` (`sdirk332.C:29`) does not
  equal row 1 of `A` summed (`29/108`, `sdirk332.C:40`), a defect of about 0.046 against a
  round-off-level agreement (max `8.9e-16`) everywhere else, found mechanically by
  `butcher_row_sum_defect()`. Scheme (5.4) of Norsett and Thomsen, "Embedded SDIRK-methods of basic
  order three", BIT 24 (1984) 634-646 is the vendored method: c = (5/6, 29/108, 1/6), rows
  (-61/108, 5/6) and (-23/183, -33/61, 5/6); the shipped tableau carries `C[1] = 29/108`. `C`
  enters only a non-autonomous right-hand side, so the Dahlquist and logistic tests cannot see it.
- **SDIRK44's declared order is 3, not 4.** The order-4 condition `sum_i b_i c_i^3 = 1/4` fails by
  exactly `1/18` in exact rational arithmetic (`b.c^3 = 11/36`); every condition through order 3
  holds (`OrderConditions.Sdirk44sDeclaredOrderIsCorrectedFromFourToThree`).
- **SDIRK34 (gamma = 0.1289) is not A-stable** (max abs R(iy) = 9.82). The tableau ships with the
  A-stable root gamma = 1.0685790213016289 (R(-inf) = -0.630) under the same name; the vendored root
  is pinned as not A-stable by a test.
- **SDIRK45 is not shipped**: it is not A-stable (R unbounded) and not singly diagonal. 21 of the 22
  vendored tableaus ship, plus three added ones (ESDIRK324L2SA, SSPRK33, SSPRK104), 24 in all.

The stability flags are computed from each tableau's stability function R(z) rather than copied
from the vendored `is_A_stable()`/`is_L_stable()` bodies (exact rational arithmetic for the
classical families; 40-digit arithmetic for the Rosenbrock family, after recovering the stored
convention: the vendored Rosenbrock files hold Hairer and Wanner's implementation form scaled by
gamma, identified by recovering ROS3P's published b = (2/3, 0, 1/3)). The computed flags are pinned
by `ButcherTableau.DeclaredStabilityMatchesTheStabilityFunction`. They differ from the vendored
claims for:

| Method | Vendored claim | Computed |
|---|---|---|
| SDIRK332 | A, L | A-stable, R(-inf) = -91/125 |
| ROS3P | A, L | A-stable, R(-inf) = 1 - sqrt(3) (by design, Lang and Verwer 2001) |
| GRK4Sn | A, L | A-stable, R(-inf) = 1/3 |
| GRK4T | A, L | A(89.3 deg)-stable, max abs R(iy) = 1.028, R(-inf) = 0.454 |

The remaining fifteen agree with their vendored flags.

### 8.2 The order-condition checker

`order_conditions.hpp`'s `RootedTreeCatalog` enumerates Butcher's rooted trees through order 5 by the
standard recursion (`gamma(t)` the density product); `OrderConditions.CatalogSizesMatchOeisA000081`
and `.DensityMatchesThePublishedTable` pin the catalog against OEIS A000081 (1, 1, 1, 2, 4, 9 trees
of order 0-5) and Hairer, Norsett and Wanner's density table. `stage_weights()` computes each tree's
`Phi_i` bottom-up with one INDEPENDENTLY summed factor per child,
`Phi_i(t) = PRODUCT_k (sum_j A[i][j] * Phi_j(child_k))`; summing a single shared stage index across
children would silently collapse every multi-child tree to the value of a single-child tree of the
same order. `OrderConditions.PerChildProductMatchesTheLiteratureConditions` pins the named
conditions (`b.c = 1/2`, `b.c^2 = 1/3`, `b.A.c = 1/6`, the order-4 star `b.(Ac)*c*c = 1/8`, ...) on
classical RK4, and the mutation row `integrators-order-condition-product-form-regression` is killed
by it.

Run over every classical-family tableau (ERK/DIRK/SDIRK/ESDIRK), the checker shows each satisfies
its declared order to round-off (`order_tolerance(stages) = 1024 * stages^2 * eps`, 2.3e-13 for one
stage to 2.3e-11 for the 10 stages of SSPRK(10,4)) and fails the next order up by at least three
orders of magnitude above that tolerance; both halves are checked.

The classical conditions apply to the Rosenbrock family only through order 2: a Rosenbrock
method's Jacobian-dependent terms make its order conditions diverge from the classical RK ones from
order 3 (Hairer and Wanner II, IV.7). `rosenbrock_classical_form()`'s derived `(M, b)` passes every
order-2 classical condition for all six Rosenbrock tableaus and fails order 3
(`OrderConditions.RosenbrockOrderTwoConditionsHoldButHigherOrderClassicalConditionsDoNot`), so the
Rosenbrock order-3-and-above claims (GRK4A/GRK4T/GRK4Sn/RODASP order 4, ROS3P order 3) are verified
numerically, by measured convergence against closed-form solutions.

### 8.3 Explicit family and convergence measurement

Convergence order is measured by a log-log least-squares fit over a decade of `h`, with the fitted
order, its standard error, and eight standard errors as the acceptance band. The band bounds the
fit's statistical scatter, not the systematic deviation of the observed order from the tableau order
at finite `h`; the latter is controlled by the per-method `h_max`, chosen in a sweep before the fit.
A failure of one of these rows is read against the `h` range first, never against the multiplier.
Measured, explicit family on the Dahlquist equation: FE 1.00503 +/- 0.00082 (expected 1), KUTTA22
2.00160 +/- 0.00025 (2), KUTTA44 4.00623 +/- 0.00101 (4; 4.01112 +/- 0.00303 on the logistic
equation), DOPRI45 5.03938 +/- 0.00726 (5, fixed step). A perturbed KUTTA44 coefficient collapses
the fitted order outside the band (`APerturbedKutta44CoefficientFailsTheOrderFourBand`).

DOPRI45's adaptive loop, driven through `SubStepClock`, lands on the window end at exact `Rational`
equality (`Dopri45AdaptiveLandsExactlyOnTheWindowEndViaSubStepClock`), and rejections fire under an
oversized initial step with a tight tolerance. The explicit methods agree byte-for-byte with an
independent 40-line reference RK loop on the same tableau and step sequence. FSAL reuse (carrying
the last stage forward instead of recomputing stage 0) is not implemented, as a deliberate
accuracy-over-efficiency choice.

### 8.4 DIRK and ESDIRK

`DirkIntegrator<Value>` runs any `Dirk`/`DirkEmbedded` tableau stage by stage through
`fuse::num::solve()`. A zero-diagonal stage is treated as explicit (no solve), so ESDIRK methods
(SDIRK643 already is one) need no separate family. `TemporalProblem::mass` is honoured on an
implicit stage (`DirkIntegrator.MassMatrixIsHonouredOnAnImplicitStage`) and refused
(`E-INTEGRATOR-WRONG-FAMILY`) in combination with an explicit stage. `E-INTEGRATOR-NEWTON-FAILED`
fires on an unsolvable stiff step. `TemporalProblem::dfdt` is not called: a DIRK stage evaluates `f`
at its own `t_i`, with no use for the correction Rosenbrock's linearisation needs.

`fuse::num::solve()` has no analytic-Jacobian hook; it always estimates the Jacobian action by MFFD,
so `DirkIntegrator` cannot use a caller-supplied `TemporalProblem::jacobian_action`.

Measured convergence on a complex Dahlquist equation:

| Method | Fitted order | Expected |
|---|---|---|
| SDIRK22 | 2.00197 +/- 0.00061 | 2 |
| SDIRK23 | 2.96553 +/- 0.01021 | 3 |
| SDIRK332 | 2.96633 +/- 0.00999 | 3 |
| SDIRK34 | 3.83911 +/- 0.04788 | 4 |
| SDIRK643 | 4.00650 +/- 0.00448 | 4 |
| ESDIRK324L2SA | 2.98121 +/- 0.00551 | 3 |

SDIRK34's wider band reflects the conditioning of the A-stable root it ships with. On the
non-autonomous problem `y' = omega*cos(omega*t) + 3t^2`, SDIRK22 converges at 2.017 +/- 0.005, and
ESDIRK324L2SA at 2.155 +/- 0.420 (wide because its error sequence is non-monotone at the coarse end).

Prothero-Robinson order reduction (`eps = 1e-3`), measured against each tableau's stage order,
reproduces the textbook phenomenon quantitatively:

| Method | Classical order | Stage order | Measured observed order |
|---|---|---|---|
| SDIRK22 | 2 | 1 | 1.633 +/- 0.044 |
| SDIRK332 | 3 | 1 | 2.145 +/- 0.023 |
| SDIRK34 | 4 | 1 | 2.154 +/- 0.027 |
| ESDIRK324L2SA | 3 | 2 | 2.367 +/- 0.030 |

### 8.5 Rosenbrock family

`RosenbrockIntegrator<Value>` solves each stage's linear system directly with `fuse::linalg::gmres`
against a `MatrixFreeOperator`, not through `fuse::num::solve()`, which is Newton-iterated and
wasteful on an already-linear stage system. `jacobian_action()` is used directly when
`TemporalProblem::jacobian_action` is present, otherwise `MffdJacobianAction` (from
`newton_krylov.hpp`, frozen at the step's start point);
`RosenbrockIntegrator.JacobianActionIsActuallyUsedWhenSupplied` counts calls on each path.

The step runs the classical Hairer-Wanner ROW recurrence on the recovered `(alpha, Gamma, b)` triple
(`rosenbrock_classical_triple()`):

```
(I - h*gamma*J) k_i = h * [ f(t_i, y + sum_j alpha_ij k_j) + J * sum_j Gamma_ij k_j
                             + h * gamma_i * f_t ]
y_new = y + sum_i b_i k_i
```

Applying the stored `A`/`gamma`/`B` directly as a "stored form" recurrence,
`(I/(h*gamma) - J) k_i = f(y + sum a_ij k_j) + sum (c_ij/h) k_j`, gives step errors of order unity
independent of `h`, because it omits the `1/gamma` scaling of the stored off-diagonal entries. The
classical recurrence reproduces the stability-function prediction `R(z)*y0` to full double
precision (`0.9797594332596039` computed against `0.9797594333` predicted), and the stored-form map
in `butcher_tableau.hpp` and the classical form agree by construction (one is the inverse of the
other).

Measured convergence on the complex Dahlquist equation:

| Method | Fitted order | Expected |
|---|---|---|
| ROS22 | 2.00197 +/- 0.00061 | 2 |
| GRK4A | 4.13581 +/- 0.06397 | 4 |
| GRK4T | 4.02999 +/- 0.02397 | 4 |
| GRK4Sn | 3.88578 +/- 0.03482 | 4 |
| ROS3P | 2.95550 +/- 0.01338 | 3 |
| RODASP | 4.00947 +/- 0.00473 | 4 |

On the nonlinear logistic equation ROS22 holds cleanly (2.014 +/- 0.004); GRK4A (3.088 +/- 0.291) and
RODASP (2.133 +/- 0.469) show wide standard errors at the `h` range tuned for the linear problem.

The formal order is conditional on the Krylov solve's accuracy (Section 6.2). Measured for ROS3P at
three GMRES settings: `rel_tol` factor 1e-2 gives 2.95550 +/- 0.01338; factor 1e-4 gives the same
to five significant figures (already inside the asymptotic regime); `rel_tol` about 0.3 collapses
the order to 1.07183 +/- 0.02126.

`E-INTEGRATOR-STAGE-SOLVE-FAILED` is made to fire by
`RosenbrockIntegrator::set_stage_gmres_max_iterations()` and the test
`StageSolveFailureIsRefusedNotFoldedIntoTheStep`: a one-step GMRES budget on the Dahlquist rotation
(complex-conjugate eigenvalues, so no real one-dimensional Krylov space can reach the stage
tolerance) names the stage and the solver's own code and leaves the state unchanged; the same step
with the default budget succeeds.

### 8.6 SSP methods

`ssp_coefficient.hpp` computes each method's SSP coefficient from its Shu-Osher form
(`ssp_coefficient_from_shu_osher()`: the minimum, over every convex-combination term with a nonzero
right-hand-side weight, of `alpha/beta`), not from a published value. `ssprk33_shu_osher()` and
`ssprk104_shu_osher()` encode each method's published decomposition (Shu and Osher 1988; Ketcheson
2008's low-storage form); the computation gives `C = 1` for SSPRK33 and `C = 6` for SSPRK104,
matching both papers. SSPRK104's Butcher `A`/`B` were derived by exact `fractions.Fraction`
arithmetic from the Shu-Osher recursion and are verified by the order-condition checker
(`ExtendedTableaus.Ssprk104SatisfiesOrderFourConditions`) and by an independent Shu-Osher reference
loop (`ExplicitIntegrator.Ssprk104MatchesTheShuOsherRecursion`, `EXPECT_NEAR` at `1e-13`, since the
two sum terms in a different order).

Monotonicity is measured in both directions (total variation of periodic upwind advection of a step
function, `n = 20` cells, 10 steps):

| Method | `C` | `h` within (`0.9*C*h_FE`) | TV within | `h` beyond (`1.5*C*h_FE`) | TV beyond |
|---|---|---|---|---|---|
| SSPRK33 | 1 | 0.9 | 2.000000 (tv0 = 2.0) | 1.5 | 1884.224971 |
| SSPRK104 | 6 | 5.4 | 2.000000 (tv0 = 2.0) | 9.0 | 5.774e23 |

Within the SSP coefficient total variation is preserved exactly; beyond it both methods violate it
grossly.

### 8.7 Dense output

`continuous_extension.hpp`'s `CubicHermiteExtension<Value>` evaluates `y(t_n + theta*h)` for `theta`
in `[0,1]` with the cubic Hermite interpolant matching value and derivative at both endpoints of
one accepted step. It is not DOPRI5's own fourth-order interpolant (Hairer, Norsett and Wanner I,
Section II.6), which is not shipped until it can be entered from the primary source rather than a
secondary transcription. `ContinuousExtension.EndpointsAreExactToRoundoff` confirms
`evaluate(0)`/`evaluate(1)` reproduce the step's endpoints to `1e-14`;
`ContinuousExtension.CubicHermiteInteriorErrorMeasuredOrder` measures the interior order at
`theta = 0.5` over an 8-point `h` sweep at `3.9708 +/- 0.0075`, the textbook local order 4.

The seam with the coupling window needs no new type: `evaluate(theta)` is what a window's temporal
interpolant needs, and `fuse::num::Waveform::push()` accepts one exact `Rational` sample per call,
so a caller computes `theta` from `SubStepClock`'s `now()` and `quantum()` (both exact `Rational`s)
and pushes `evaluate(static_cast<Scalar>(theta))`.

### 8.8 Not implemented

- The Kennedy-Carpenter ESDIRK4(3)6L[2]SA and ESDIRK5(4) pairs, the Kvaerno (2004) ESDIRK
  pairs and the Conde-Fekete-Shadid (2018) embedded SSP pair. Each needs its coefficients entered
  from a primary source and proved by the order-condition checker, the row sums and the
  L-stability test before use; transcribing many-digit rationals without that cross-validation risks
  a silent defect of the SDIRK332 kind. The primary source for the Kennedy-Carpenter pairs is NASA
  TM-2001-211038 (the unabridged form of Appl. Numer. Math. 44 (2003)), Appendix D, which also gives
  their dense-output rows in exact rationals; the available PDF is scanned with OCR damage in its
  text layer, so entries are read from the page image. Kvaerno 2004 (BIT 44) is paywalled.
- DOPRI5's own quartic interpolant and the stiffly accurate ESDIRKs' `b(theta)` polynomials
  (Section 8.7).
- The dynamical-time-scale controller (Section 2, mechanism 1). It is a separate controller family
  that needs its own design: what "dynamical time scale" means for a `TemporalProblem` that may not
  expose one directly, most likely an eigenvalue estimate from `jacobian_action()` or the MFFD
  action.
- An analytic-Jacobian parameter on `fuse::num::solve()`, which would let DIRK stages skip the MFFD
  evaluation when a caller holds an exact Jacobian action; it would mirror the dispatch
  `RosenbrockIntegrator` already uses.
