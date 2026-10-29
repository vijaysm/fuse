# S8 physics-kit, track "Integrators": read-first design for porting KARMA's time integrators into fuse::num

Scope: upstream/karma/{include,src}/numerics/temporal, pinned sha 136f520 (UPSTREAMS.lock [karma]),
read-only. No code in this stage beyond the absence of any compile probe (none was needed: every
claim below is a source read or a grep, not a build). Owner decisions D-A..D-E (2026-10-27) are
taken as given and cited where they bear on a choice.

## 0. Plain answers first

- **How many integrators.** 22 named Runge-Kutta/Rosenbrock methods (the `RKType` enum,
  `include/enums/runge_kutta_schemes.h:42-64`), driven through 7 integrator classes (one base,
  `TemporalIntegrator`, plus `ExplicitRKIntegrator`, `ImplicitDIRKIntegrator`,
  `EmbeddedDIRKIntegrator`, `DOPRI45Integrator`, `RosenbrockIntegrator`, `PseudoTimeIntegrator`,
  `PetscTemporalIntegrator`), across 66 files and 8,316 lines (MEASURED, `wc -l`, matches the task
  statement exactly).
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
same mathematical method, never `crank_nicolson.{h,C}`. Per D-C ("a KARMA test is retired only if a
named fuse test covers the same behaviour, otherwise it is ported"): since no named test covers any
of the 22 methods or 7 drivers in this directory, **every one is a port, with no retirement
available on coverage grounds** -- `UPSTREAMS.lock`'s own `retire_at = "S8"` for the whole `[karma]`
subtree is consistent with that: this directory's code is intended to be ADAPTED and tested anew,
not carried forward as-is. `ode_system_test.C`'s CN rate-2 assertion is still useful evidence: it
pins what a correct Crank-Nicolson port must reproduce (order 2, not more, not less), even though it
cannot be reused as a bit-identical oracle for `crank_nicolson.{h,C}`'s own tableau.

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
   (not fully read in this pass; structurally the same `E`-weighted local-truncation-error pattern
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

### 3.1 fuse::linalg / fuse::num: what exists today (read `include/fuse/linalg`, `include/fuse/num`
first, per the task)

- **Exists.** `DistVector`/`DistSparseMatrix` (CSR, row-partitioned, `apply()` = SpMV) and
  `MatrixFreeOperator`, both satisfying the SAME `LinearOperator` concept gmres/pcg take
  (`operator.hpp:69-90`) -- no "assembled vs. matrix-free" fork at the solver level. `gmres`/`pcg`
  Krylov solves with `Preconditioner`/`IdentityPreconditioner`/`JacobiPreconditioner`
  (`preconditioner.hpp`). `fuse::num::solve()`: a complete Jacobian-free Newton-Krylov driver
  (MFFD Jacobian action, GMRES, Eisenstat-Walker forcing, line search, named convergence codes,
  lagged block-Jacobi preconditioning) ALREADY dispatchable over a `NewtonSubstrate::{Native,Petsc}`
  registry (`newton_krylov.hpp:420-504`) -- this is exactly KARMA's `TemporalIntegrator`/
  `PetscTemporalIntegrator` split, one layer up, already built and (per `docs/dev/roadmap.md`'s S7
  entry) measured on `dn_heat` (order 2.0 +/- 0.24) and the Brusselator window (50/50 convergence).
  `SchurSlabSolver`: an exact direct solver, but scoped to one operator family
  (`A = I - coefficient*L_h`, the periodic 5-point Laplacian), not general. The derivative gate
  (`derivative_gate.hpp`, S7.3) already supplies exactly the verification primitives a ported
  Jacobian action needs: `taylor_remainder_order()` (checks a claimed `J v` against the residual's
  own Taylor remainder) and `matrix_free_agreement()` (checks an assembled vs. matrix-free `apply`
  against each other) -- both DERIVED bounds, not fitted constants, consistent with the owner's
  tolerance rule.
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
  (where the numtypes `BasicFieldRef`/`FieldRef` will live) and may depend on `model`/`linalg`
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

// Value: double in production; dual<S, N> (fuse/num/dual.hpp, numtypes track, not yet landed)
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
nonlinear-solve layer (this is D-A's "adapted, not ported" in concrete form: KARMA's own
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
(`dual<S,N>`, numtypes handoff notes). Two quantities need a stated convention:

- **Step size `h`.** Always primal-only (`dual<S,N>::value()` / `S`, never a dual itself) -- a
  step size with a nonzero tangent would mean "the schedule point moves as a function of the
  sensitivity parameter," which is a DIFFERENT and much harder problem (shape/parameter-dependent
  meshing of time) that nothing in this track's scope asks for. `h` stays an `S` (matching
  `model/value_traits.hpp`'s `value_traits<T>::scalar_type`) even when the state is `dual<S,N>`.
- **Error norm.** Computed over the PRIMAL components only, for the SAME reason DOPRI45's and
  EmbeddedDIRK's error estimate is a physical-accuracy measure, not a sensitivity measure: a
  tangent component blowing up is a conditioning/stability question for the forward sensitivity
  equation, not a question the step-size controller (which only ever sees `S`, per the point above)
  is equipped to answer. Open question 1 (Section 7) asks the owner to confirm this rather than
  silently deciding it, since it determines whether a stiff sensitivity direction can silently
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
`temporal_basis.hpp`, per the task): `Waveform::push(Rational t, const double* values)`
(`waveform.hpp:216`) wants exact-`Rational` sample times and flat `double*` rows -- it is NOT
templated on `Value` today (consistent with numtypes not having landed: `FieldRef` defaults to
`double`, per the handoff notes). A component driving an adaptive sub-stepper under `Adjustable`
must therefore: (1) convert each accepted sub-step's primal state to `double` before `push()` (a
`dual<S,N>`'s tangent components have nowhere to go in a `Waveform` today -- Open question 2); (2)
convert the sub-step's OWN time (an `S`/double from Section 4.3) to an exact `Rational` before
calling `push()`, the same "form the exact difference before going to double" discipline
`temporal_basis.hpp:19-22` already states for the OTHER direction (`Rational` to `double`) -- for a
sub-step time that is the sum of many floating `h`'s, this is not free: the integrator's own `time`
accumulator (KARMA carries this as a `PetscReal`, `temporal_integrator.h:144`) would need to become
a `Rational` running total itself, counting sub-steps as exact fractions of the window, or every
`push()` re-derives its `Rational` time from the window's own exact bounds plus an integer sub-step
count -- the latter only works for a FIXED sub-step count known in advance, which an adaptive
sub-stepper by definition does not have. This is Open question 3.

### 4.5 Refusals, named

Every one of these is a REFUSAL (a named, tested failure path), not a silent clamp or a crash:

| Name | Condition | KARMA precedent |
|---|---|---|
| `E-INTEGRATOR-STEP-TOO-SMALL` | proposed `h` below a floor derived from `uround`/`time` (DOPRI45's own `0.1*fabs(h) <= fabs(time)*uround`, `dopri45_integrator.C:151`) | returns `-3` |
| `E-INTEGRATOR-MAX-STEPS` | `nstep > nmax` | returns `-2` |
| `E-INTEGRATOR-NONFINITE-ERROR` | the embedded error norm is NaN/Inf -- **new**: KARMA never checks this; an Inf `err` compares `false` to `err<=1.0` in C++ and is silently treated as a REJECT by accident, not by design | none (gap) |
| `E-INTEGRATOR-NEWTON-FAILED` | the per-stage `fuse::num::solve()` call returns a non-`Converged` status | KARMA logs and continues (`embedded_dirk_integrator.C:251-256`, `newton_fail` is set but never acted on -- a genuine KARMA defect, Section 6.1) |
| `E-INTEGRATOR-STIFF-UNDETECTED-NONSTIFF-METHOD` | DOPRI45's own stiffness detector trips 15 times | returns `-4` |

### 4.6 Collective agreement on accept/reject (the owner's standing rule, applied)

**This is the single most important correctness requirement this track found** (Section 6.1 has the
measured defect it fixes). KARMA's `DOPRI45Integrator` computes its error norm by hand:
`VecGetArray` on every rank, a LOCAL sum `lnrm`, then `MPI_Reduce(&lnrm, &err, 1, MPI_DOUBLE,
MPI_SUM, 0, KARMA_COMM_WORLD)` -- **root only** (`dopri45_integrator.C:243`) -- followed by `err =
sqrt(err/n)` and the accept/reject branch `if (err <= 1.0)` (`dopri45_integrator.C:245,258`), which
every rank evaluates on its OWN copy of `err`. Non-root ranks never receive the reduced value (that
needs `MPI_Allreduce`, not `MPI_Reduce`); every rank but 0 evaluates the branch on whatever `err` held
before the call (`0.0`, set at line ~208), i.e. **every non-root rank always accepts**, independent of
the root's decision. This is exactly the hazard the owner's standing rule names ("a collective path
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

Every tolerance below is DERIVED and printed next to the test that uses it, per the owner's standing
rule; none is picked after a run and then widened.

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
3. **The KARMA differential test**: NOT a live run of KARMA (MPI cannot start in this sandbox, and
   per Section 1.4 no KARMA test exercises this directory anyway) -- instead, the Butcher
   coefficients copied BYTE-FOR-BYTE from the vendored `.C` files (already extracted into Section
   1.2/this document, each with a file:line citation) are hardcoded into both an independent
   reference script and the `ButcherTableau` the fuse port builds; for EXPLICIT methods (FE,
   Kutta22/44, DOPRI45) the arithmetic has no solver-ordering freedom, so the fuse result must match
   the reference to the LAST BIT on the SAME problem and step sequence. For IMPLICIT methods
   (SDIRK*, TRBE, embedded DIRK) D-A licenses a BOUNDED difference, not bit-identity: KARMA solves
   each stage with `SNES`'s own line search and Krylov defaults, fuse solves it with
   `fuse::num::solve()`'s own (different) Eisenstat-Walker forcing and line search -- the two Newton
   iterations converge to the same root to WITHIN their respective `atol`/`rtol`, so the bound is
   `|y_fuse - y_karma_reference| <= rtol*|y| + atol` (both integrators' own stage-solve tolerances,
   summed), not zero. For Rosenbrock, D-A's bound is the SAME shape but driven by the GMRES
   tolerance on the stage's linear solve rather than a Newton tolerance.
4. **Mutation rows** (one per tableau, run against the convergence test in (1) to confirm it would
   actually fail): perturb one `A`/`B`/`C` coefficient by `1e-3` (relative) and confirm the measured
   order DROPS below the tableau's own order minus the derived tolerance from (1) -- a mutation that
   does not move the measured order means the convergence test in (1) is not sensitive to that
   coefficient and is itself a gap, not a pass. Separately: halve the controller's `safety` factor
   and confirm the number of ACCEPTED steps changes (the controller is exercised, not dead code);
   force a rejection path (set an artificially tight `rtol`) and confirm `E-INTEGRATOR-STEP-TOO-SMALL`
   fires for at least one method per family -- per the report contract, "a refusal counts as verified
   only when a test makes it fire."
5. **Jacobian-action verification** (Rosenbrock, DIRK): `taylor_remainder_order()` and
   `matrix_free_agreement()` from `derivative_gate.hpp` (already built, Section 3.1), run against
   every `Problem` this track's own test suite defines -- no new verification primitive needed, only
   new call sites.
6. **Tier 0 vs. needs-MPI**: every item above runs at tier 0, single rank, no PETSc required for the
   Native substrate. The PETSc substrate (`PetscTemporalIntegrator`'s analogue, if ported per Open
   question 4) and any multi-rank `DistVector`/collective-agreement check (Section 4.6) need MPI;
   per the standing rule, those are BUILT, never RUN, in this sandbox -- the exact host command for a
   2-rank and a 3-rank run (once a real build exists) is `mpirun -np 2 <test-binary>` /
   `mpirun -np 3 <test-binary>`, run on the owner's own machine, not here.

## 6. Risks

### 6.1 Defects found in the vendored source that a literal port would carry forward

These are MEASURED, not suspected, and none should be reproduced in the fuse port:

- **DOPRI45's rank-divergent accept/reject** (`dopri45_integrator.C:243,258`, detailed in Section
  4.6): `MPI_Reduce` to root only, then every rank branches on its own unreduced copy. Fixed by
  construction once the error norm goes through `DistVector::norm2()` (Section 4.6) -- but worth
  the owner's attention as a standalone finding, since it is exactly the hazard class the project's
  own standing rule exists to prevent, found in exactly the kind of code (a 2007-2011 import) the
  rule was written anticipating.
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
will CATCH this (an order below the derived tolerance band fails the test), but the owner should
decide up front how tight the per-stage GMRES tolerance needs to be RELATIVE to the step's own
embedded-error tolerance (Open question 5) rather than discovering the coupling empirically per
method.

### 6.3 Risk: template instantiation cost

`Integrator<Value, Problem>` instantiated over `Value in {double, dual<S,1>, dual<S,2>, ...}` and
every `Problem` type in `physics-kit` multiplies the number of instantiated `step()` bodies (each
containing a full stage loop, for the higher-stage-count methods such as RODASP's 6 stages or
DOPRI45's 7) by `|Value| x |Problem|`. This is the SAME shape of cost `fuse::num::solve()` already
accepts for its own `ResidualFn`/`PreconditionerApplyFn` type erasure (`std::function`-based, not
templated per problem) -- Open question 6 asks whether `Integrator` should follow that same
type-erasure discipline (one compiled `step()` per `Value`, problems type-erased through a
`std::function`-based `TemporalProblem` adapter) rather than a fully templated `Problem`, trading a
small amount of per-call indirection for a bounded instantiation count independent of how many
physics kits exist.

### 6.4 Risk: the exact-rational schedule and a sub-step time accumulator

Section 4.4 already names this as Open question 3: an adaptive sub-stepper accumulates `time` in
`Value`'s own scalar type (floating point), the same way KARMA does
(`temporal_integrator.h:144-150`, `PetscReal time`), while `Waveform`/the schedule itself work in
exact `Rational` (`waveform.hpp:29-35`). Converting a floating accumulated time back to an exact
`Rational` for `push()` is not a well-defined operation in general (a `Rational` constructed from an
arbitrary `double` is exact FOR THAT DOUBLE, but that double is itself already an accumulation of
rounding error over many sub-steps) -- this needs an owner decision on which side gives way
(Section 7, open question 3), not a default chosen here.

## 7. Open questions for the owner

1. **Error norm under `dual<S,N>`** (Section 4.3): confirm the step-size controller reads only the
   primal component of the embedded error, never a tangent component, so a growing sensitivity
   direction cannot silently escape step-size control. Recommendation: confirm as stated (primal
   only); the alternative (control on a combined primal+tangent norm) conflates two different
   notions of "the step is accurate" and has no precedent in either KARMA or the literature this
   track found.
2. **`dual<S,N>` sub-step history into `Waveform`**: `Waveform::push()` is `double`-only today
   (Section 4.4). Options: (a) a component running a sensitivity sweep writes ONLY the primal
   `Waveform` and keeps tangent sub-step history in a SEPARATE, integrator-owned buffer the
   coupling layer never sees (simplest, defers tangent coupling entirely to a later track); (b)
   `Waveform` is templated on `Value` now, ahead of any concrete consumer (larger, blocks on the
   numtypes track landing first since `Waveform` sits in `num`, same layer). Recommendation: (a),
   since nothing in this track's own scope (D-B: "adaptive sub-steps inside a fixed coupling window
   only") needs tangent values to cross a window boundary yet.
3. **Exact-rational time accumulation for adaptive sub-steps** (Section 4.4/6.4): does the
   integrator's own `time` accumulator become a `Rational` (every sub-step's `h` rounded to a
   rational before advancing `time`, matching the schedule's own exactness, at the cost of an
   integrator having to round its OWN floating controller output), or does `Waveform::push()` gain a
   documented tolerance for "this Rational is only as exact as its originating double" (cheaper,
   but weakens the "time is exact" invariant `waveform.hpp:29` currently states unconditionally)?
   No recommendation: this is an accuracy/usability trade-off the standing rule reserves for the
   owner.
4. **Is a PETSc-`TS`-substrate integrator in this track's scope at all?** `PetscTemporalIntegrator`
   (Section 1.3) is a complete second implementation of every method via PETSc's own `TS`; D-B says
   "physics-kit is tier 0 with PETSc optional" but does not say whether "all KARMA time integrators"
   (D-B) includes re-exposing PETSc's `TS` as a `NewtonSubstrate`-style alternate substrate for
   `fuse::num::Integrator`, or whether PETSc's `TS` is out of scope entirely (since `fuse::petsc`
   already exists for the NONLINEAR solve, Section 3.1, a `TS`-substrate integrator would be a THIRD
   PETSc integration point). Recommendation: out of scope for this track; revisit once PETSc's `TS`
   is needed for a specific physics-kit component, not speculatively.
5. **Per-stage linear-solve tolerance for Rosenbrock, relative to the step's own error tolerance**
   (Section 6.2): a concrete ratio (e.g. GMRES `rtol` two orders tighter than the step's `rtol`) is
   an accuracy/performance trade-off the standing rule reserves for the owner; no default is chosen
   here.
6. **Type erasure vs. full templating for `Problem`** (Section 6.3): trades compile cost against
   call-site indirection; Recommendation: type-erase (`std::function`-based `TemporalProblem`
   adapter), matching `fuse::num::solve()`'s own existing `ResidualFn` convention, but this changes
   the 30-line sketch in Section 4.1 enough that it should be confirmed before any code is written.

## Deviations

- Item 2 of the task ("how KARMA does adaptive stepping") turned out to be four independent
  mechanisms, not one controller with one set of parameters; Section 2 reports all four rather than
  picking the one the task's phrasing ("controller (I, PI, PID?)") seemed to expect, since reporting
  only the best-matching one would have hidden the `ExplicitRKIntegrator`/`PseudoTimeIntegrator`
  dynamical-time-scale mechanism and the DOPRI45-vs-EmbeddedDIRK architectural split, both load-bearing
  for the fuse design in Section 4.2.
- Item 1's "which KARMA tests cover it with which asserted orders" is answered as "none, directly"
  for every one of the 66 files (Section 1.4) -- a substantive, measured finding, not a shortfall in
  the search; the one order assertion found (`ode_system_test.C:90`, Crank-Nicolson rate 1.8-2.2)
  exercises a sibling PETSc-`TS`-based implementation, not this directory's own `crank_nicolson.{h,C}`,
  and is reported as such rather than credited to this directory's coverage.
- No compile probe was run: this track produced no code, so there is nothing a probe would have
  exercised beyond confirming the existing repository still configures, which is already recorded
  in `UPSTREAMS.lock`'s own import-time measurement (2026-09-15, Apple clang 21/CMake 4.3.2, 0
  errors, 418 ctest entries) and was not re-verified here.
- Section 3's "what fuse::linalg lacks" reads `include/fuse/linalg` and `include/fuse/num` in full
  file listing but not every line of every file (`gmres.hpp`, `pcg.hpp`, `comm.hpp`,
  `dist_vector.hpp`, `line_search.hpp`, `accelerator.hpp`, `space_kernels.hpp`, `kernels.hpp` were
  read by header/purpose comment and targeted grep for solve/factor keywords, not line-by-line) --
  sufficient to answer "what exists" and "what's missing" at the interface level the task asks for,
  but a from-scratch implementer should re-read `gmres.hpp`/`pcg.hpp`'s own option structs before
  writing the Rosenbrock stage-solve call site.

---

## Lead's corrections after review (2026-10-27)

1. **The seven integrator drivers are not compiled by KARMA.** MEASURED: `src/CMakeLists.txt`'s
   `KARMA_SOURCES` lists the 22 tableau files plus `karma_ts_base.C`, `rk.C` and `rosenbrock.C`, and
   none of `dopri45_integrator.C`, `embedded_dirk_integrator.C`, `explicit_rk_integrator.C`,
   `implicit_dirk_integrator.C`, `petsc_temporal_integrator.C`, `pseudo_time_integrator.C`,
   `rosenbrock_integrator.C` or `temporal_integrator.C`. This is why Section 1.4 finds no test: the
   drivers are unbuilt legacy. The rank-divergent accept/reject in `dopri45_integrator.C:243,258`
   (Section 4.6, lead-verified by reading those lines) was therefore never executed in KARMA; it stays
   a defect the fuse port must not reproduce, not a defect in KARMA's shipped behaviour. The
   tableaus and the three base classes are the live part of the port; the drivers are a design
   reference only, and the Section 4 design replaces them rather than adapting them.
2. The lead verified Section 1.4's coverage claim independently (every class name declared under
   `include/numerics/temporal` grepped outside that directory: hits are the `RKType` enum names in
   `include/enums/runge_kutta_schemes.h`, the unrelated `FE`/`CN` abbreviations in FE tests, and
   `TemporalIntegrator` in the excluded `interface/` layer). The GRK4A zero-error-estimate,
   ignored-Newton-failure and DOPRI45 metadata claims (Section 6.1) were NOT independently verified
   by the lead and carry the track's own citations.

Owner decisions of 2026-10-27 on the open questions: (1) primal-only error norm; (2) option (a);
(3) a Rational time accumulator with a recorded per-window quantum, last step clipped exactly;
(4) no PETSc-TS substrate in S8; (5) stage GMRES rtol = step rtol times a fixed factor starting at
1e-2, held to the per-method order test and tightened, never loosened; (6) type-erased Problem.


---

## Round 1 built (2026-10-27/28, Stage B)

Built in `include/fuse/num/{butcher_tableau,embedded_controller,integrator,
explicit_integrator}.hpp` (all four header-only: no `.cpp` turned out to be needed --
`ButcherTableau`'s factories are plain functions, `EmbeddedController`/`Integrator`/
`ExplicitIntegrator` are template classes whose bodies have to live in the header
regardless) plus `test/num/{butcher_tableau,embedded_controller,integrator,
explicit_integrator}_test.cpp` (37 new `TEST` cases, all passing) and
`build-evidence/mutations/integrators.toml` (7 new rows, all killed).

**What the four headers are.** `butcher_tableau.hpp`: the `ButcherTableau` struct
(Section 4.1) plus all 22 KARMA tableaus as factory functions, each coefficient cited
to its vendored `file:line`. `embedded_controller.hpp`: `EmbeddedController` (Section
4.2), the PI (DOPRI45's own Gustafsson controller) and I (EmbeddedDIRKIntegrator's own
controller) kinds behind one entry point, `propose()`, a pure function of an
already-agreed scalar error norm. `integrator.hpp`: the `IntegratorValue` concept (a
LOCAL mirror of the not-yet-landed `CoupledValue`/`value_traits<T>::scalar_type`
shape), the type-erased `TemporalProblem<Value>` adapter (owner decision 6), the
`Integrator<Value>` interface every family implements, and `SubStepClock` (owner
decision 3: a Rational time accumulator with a recorded per-window quantum,
`kDefaultQuantumCount = 2^20`, derived in the class's own doc comment).
`explicit_integrator.hpp`: `ExplicitIntegrator<Value>`, one class over any
`Erk`/`ErkEmbedded` tableau, covering FE/KUTTA22/KUTTA44 (fixed step) and DOPRI45
(embedded, with the corrected metadata below).

**Three findings this round**, all re-verified or newly found by reading/running
against the vendored source directly (not inherited from Stage A by reference):

1. RE-VERIFIED (task's own instruction): GRK4A's embedded estimate is still measured
   identically zero (`grk4a.C:71`) on a second, independent read; `grk4a_tableau()`
   sets `embedded_order = std::nullopt` accordingly (`butcher_tableau.hpp`'s own top
   comment, `ButcherTableau::has_embedded_estimate()` reports `false`), tested by
   `ButcherTableau.Grk4aEmbeddedEstimateIsMeasuredZeroNotAssumed`.
2. NEW: DOPRI45's own `is_FSAL()` is hardcoded `false` (`dopri45.h`) despite the
   tableau having the textbook FSAL structure (row 6 of `A` bit-identical to `B`,
   `C[6] == 1.0`) -- `dopri45_tableau()` carries the CORRECTED value, `true`, tested by
   `ButcherTableau.DeclaredFsalMatchesTheStructuralProperty` (run over all 22 tableaus,
   not just DOPRI45).
3. NEW, found MECHANICALLY by `butcher_row_sum_defect()` rather than by reading:
   SDIRK332's own `C[1]` (`24.0/108.0`, `sdirk332.C:29`) does not equal row 1 of `A`
   summed (`29/108`, `sdirk332.C:40`) -- a defect of `~0.046`, far above every other
   tableau's own round-off-level agreement (max observed `8.9e-16`). Kept
   byte-identical to the vendored (wrong) value pending the owner's own Open Question
   5 below; `ButcherTableau.ClassicalFamiliesAreRowSumConsistentExceptTheKnownSdirk332
   Defect` asserts the defect itself, so a future silent "fix" is caught.

**Verification, MEASURED:**
- `test/num/{butcher_tableau,embedded_controller,integrator,explicit_integrator}
  _test.cpp`: 37 new `TEST` cases (6 + 7 + 8 + 16), all passing
  (`ctest -R '^num\.'`: `num.butcher_tableau`, `num.embedded_controller`,
  `num.integrator`, `num.explicit_integrator`, each one ctest row per
  `fuse_add_gtest`'s own convention).
- Convergence order measured by a log-log least-squares fit over a decade of `h`
  (fitted order, its own standard error from the fit, and 8 standard errors as the
  acceptance band -- the multiplier and the `h`-range are both derived and documented
  at the call site, `explicit_integrator_test.cpp`'s own comments): FE 1.00503 +/-
  0.00082 (band 0.0065, expected 1), KUTTA22 2.00160 +/- 0.00025 (band 0.0020, expected
  2), KUTTA44 4.00623 +/- 0.00101 (band 0.0081, expected 4, Dahlquist; 4.01112 +/-
  0.00303 on the logistic equation, band 0.0242), DOPRI45 5.03938 +/- 0.00726 (band
  0.0581, expected 5, fixed step) -- every fitted order inside its own band by 20-50
  percent of margin, MEASURED by a standalone sweep of candidate `h`-ranges before this
  round's own `h_max` values were fixed (this is a choice of WHICH `h` values enter the
  fit, decided before relying on the result, not a widened acceptance band after a
  failure). A mutated KUTTA44 coefficient collapses the fitted order well outside the
  same band (`APerturbedKutta44CoefficientFailsTheOrderFourBand`), confirming the test
  is sensitive, not merely passing regardless.
- DOPRI45's adaptive loop, driven through `SubStepClock`, lands on the window end at
  EXACT `Rational` equality (`Dopri45AdaptiveLandsExactlyOnTheWindowEndViaSubStepClock`)
  -- not approximately: `clk.now() == Rational(1)` is asserted, not `EXPECT_NEAR`.
  Rejections are made to fire under a deliberately oversized initial step and a tight
  tolerance (`Dopri45RejectsAnOversizedInitialStepUnderATightTolerance`).
- Byte-identical agreement against an independent 40-line reference RK loop (same
  tableau, same fixed step sequence) for all four methods
  (`expect_bit_identical_to_reference`, `EXPECT_EQ` on every component of every step,
  not `EXPECT_NEAR`).
- `E-INTEGRATOR-NONFINITE-ERROR`, `E-INTEGRATOR-STEP-TOO-SMALL`,
  `E-INTEGRATOR-NO-RHS`, `E-INTEGRATOR-MASS-UNSUPPORTED`, `E-INTEGRATOR-WRONG-FAMILY`
  and `E-INTEGRATOR-DIMENSION-MISMATCH` are each made to fire by a test, per the
  report contract ("a refusal counts as verified only when a test makes it fire").
- `build-evidence/mutations/integrators.toml`: 7 rows (one coefficient perturbation per
  round-1 tableau -- FE, KUTTA22, KUTTA44, DOPRI45 -- a controller safety-factor row, a
  rejection-path row, and a `SubStepClock` clip row), all run through
  `tools/mutate.py --build-dir build-tier0 --area num` (Python 3.15,
  `/opt/homebrew/bin/python3`, for `tomllib`; the system `/usr/bin/python3` is 3.9 and
  lacks it) against this round's own `build-tier0`: **25 killed, 0 not-killed, 0
  inconclusive, 0 without a baseline** (18 pre-existing num-area rows plus the 7 new
  ones) -- every row's source file was BYTE-IDENTICAL to its pre-mutation content after
  the run (re-checked with `tools/mutate.py --check-anchors`, 455/455 valid, and by
  `grep` for leftover mutant text, none found).
- Full tier-0 `ctest -j3`: **244/244 tests passed** (`num` label: 14 tests, the 10
  pre-existing plus the 4 new binaries), run once after the mutation gate's own restore,
  confirming the restore left the tree in the SAME state the rest of the suite already
  depends on.
- Sphinx: `fuse-docs` conda environment's own interpreter
  (`/Users/mahadevan/.claude-science/conda/envs/fuse-docs`, Sphinx 9.1.0),
  `sphinx-build -W --keep-going -b html docs docs/_build/html`: the track reported
  "build succeeded, zero warnings" in its structured result; the lead's own clean-export
  docs build of this landing is recorded in the lead's note below.

**Round 1 deliberately did NOT build** (round 2, D-B): the DIRK family (`ImplicitDIRK`/
`EmbeddedDIRK`-equivalent, over `fuse::num::solve()`), the Rosenbrock family (over
`gmres`/`MatrixFreeOperator`), `TemporalProblem::mass`/`jacobian_action`/`dfdt` call
sites (declared now, Section 4.1, not yet called from anywhere), the dynamical-
time-scale controller (out of this track's scope, Section 4 top comment of
`explicit_integrator.hpp`), and the FSAL reuse optimisation (a deliberate accuracy-
over-efficiency choice, same file's own top comment).

### Open questions, updated

Open questions 1-6 from Stage A stand (DIRK/Rosenbrock/numtypes scope, round 2). One
NEW open question from this round:

7. **SDIRK332's own `C[1]` defect** (finding 3 above): is `24.0/108.0`
   (`sdirk332.C:29`) a mistyped `C[1]` that should read `29.0/108.0` (making the tableau
   row-sum consistent, the far more likely transcription slip given every other row of
   every other tableau agrees to round-off), or did the ORIGINAL derivation of this
   tableau's `A[1][0]`/`A[1][1]` go wrong instead, with `C[1]` being the one correct
   value? No test in this round's own suite can distinguish the two (both keep the
   tableau internally self-consistent); re-deriving SDIRK332 from its own cited source
   (Kvaerno, Norsett & Owren, "Runge-Kutta research in Trondheim") is round-2 scope, not
   decided here. Recommendation: treat `C[1] = 29.0/108.0` as the more likely fix
   (consistent with every other row's own agreement), but confirm against the cited
   reference before changing the shipped coefficient -- a silent "fix" without that
   confirmation would just trade one undocumented wrong number for another.

**Lead's verification of finding 3 (2026-10-28, read from `upstream/karma/src/numerics/temporal/sdirk332.C`):**
`gamma = 5.0/6.0` (line 25); row 2 of `A` is `-23/183 - 33/61 + gamma = 1/6 = C[2] = 1 - gamma`
(lines 29 and 42), so `gamma = 5/6` is forced by the tableau itself; row 1 of `A` is
`-61/108 + 5/6 = 29/108` (line 41) against the vendored `C[1] = 24/108` (line 29). The row-sum
condition holds for every other row of every other tableau to round-off, so `C[1]` is the
transcription error, as the track inferred. `C` enters only a non-autonomous right-hand side; the
Dahlquist and logistic rows cannot see it, which is why no test distinguishes the two readings. The
owner decides whether the shipped `sdirk332_tableau()` carries `29/108` (recommended) before the
DIRK family is built in round 2.

**Lead's note on the order bands (2026-10-28).** The acceptance band is eight standard errors of
the least-squares slope. That bounds the fit's statistical scatter, not the systematic deviation of
the observed order from the tableau order at finite `h` (the next term of the asymptotic expansion);
the latter is controlled by the per-method `h_max` chosen in a standalone sweep before the fit, and
the measured deviations (FE 0.005, Kutta22 0.0016, Kutta44 0.006, DOPRI45 0.039) are each inside
their band with margin. A future failure of one of these rows is read against the `h` range first,
never against the multiplier.
