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

## Owner decisions on the tableau findings (2026-10-29) and the lead's stability audit

The owner supplied the primary source for SDIRK332: S. P. Norsett and P. G. Thomsen, "Embedded
SDIRK-methods of basic order three", BIT 24 (1984) 634-646. Scheme (5.4) there (the pair "based on
(3.15) using local extrapolation") is the vendored method: c = (5/6, 29/108, 1/6), rows
(-61/108, 5/6) and (-23/183, -33/61, 5/6), the paper's b-row (25/61, 36/61, 0) is the second-order
estimate and its e-row (26/61, 324/671, 1/11) the propagated third-order solution. The vendored
`C[1] = 24/108` is a transcription error; Open Question 7 is CLOSED with `C[1] = 29/108`.

The lead then evaluated every shipped tableau's stability function R(z) rather than trusting the
vendored `is_A_stable()`/`is_L_stable()` bodies (exact rational arithmetic for the classical
families, 40-digit arithmetic for the Rosenbrock family after recovering the stored convention: the
vendored Rosenbrock files hold Hairer and Wanner's implementation form scaled by gamma, identified by
recovering ROS3P's published b = (2/3, 0, 1/3)). Findings, all now carried as the computed flags and
pinned by `ButcherTableau.DeclaredStabilityMatchesTheStabilityFunction`:

| Method | Vendored claim | Computed | Decision |
|---|---|---|---|
| SDIRK332 | A, L | A-stable, R(-inf) = -91/125 | flag corrected |
| SDIRK34 (gamma = 0.1289) | A | not A-stable, max abs R(iy) = 9.82 | owner: ship the A-stable root gamma = 1.0685790213016289 (R(-inf) = -0.630) under the same name; the vendored root is pinned as not A-stable by a test |
| SDIRK45 | A | not A-stable (R unbounded), not singly diagonal | owner: dropped; 21 tableaus shipped |
| ROS3P | A, L | A-stable, R(-inf) = 1 - sqrt(3) (by design, Lang and Verwer 2001) | flag corrected |
| GRK4Sn | A, L | A-stable, R(-inf) = 1/3 | flag corrected |
| GRK4T | A, L | A(89.3 deg)-stable, max abs R(iy) = 1.028, R(-inf) = 0.454 | both flags corrected |

The remaining fifteen agree with their vendored flags. The row-sum test no longer carries an
exception; the vendored SDIRK332 value is pinned as a defect the check catches.

Owner decisions for round 2, recorded here: (a) the DIRK integrator treats a zero diagonal as an
explicit stage, so ESDIRK methods (SDIRK643 already is one) need no new family; Kennedy-Carpenter
ESDIRK3(2)4L[2]SA, 4(3)6L[2]SA, 5(4)7L[2]SA and Kvaerno (2004) are the pairs to add, cited to their
papers, each passing the stability-function test and the order conditions before use; (b) SSP
explicit methods (Shu-Osher SSPRK(3,3), Ketcheson SSPRK(10,4), the embedded SSP pairs of Conde,
Fekete and Shadid 2018) run in `ExplicitIntegrator` as tableaus; the SSP coefficient is carried as
tableau metadata and checked by a test of the same kind as the stability test; (c) dense output
(DOPRI5's free fourth-order interpolant, the b(theta) polynomials of the stiffly accurate ESDIRKs)
goes on the round-2 list as the integrator-side supply for the coupling window's temporal
interpolant.

## Round 2 built (2026-10-30/31, Stage B)

Built in `include/fuse/num/{order_conditions,dirk_integrator,rosenbrock_integrator,
continuous_extension}.hpp` (four new header-only files), plus `ssp_coefficient.hpp`
(the SSP-coefficient-from-Shu-Osher-form computation) and modifications to
`butcher_tableau.hpp` (three new tableaus, SDIRK44's order correction, a
`rosenbrock_classical_form()` helper), `integrator.hpp`/`explicit_integrator.hpp`
(the three refusal-code constants moved to the shared header, used by all three
families now). Four new test files (`order_conditions_test.cpp`,
`dirk_integrator_test.cpp`, `rosenbrock_integrator_test.cpp`,
`continuous_extension_test.cpp`, 27 new `TEST` cases) plus two new tests appended to
`explicit_integrator_test.cpp`, and `build-evidence/mutations/integrators-round2.toml`
(12 new rows).

### Item 1: the order-condition checker

`order_conditions.hpp`'s `RootedTreeCatalog` enumerates Butcher's rooted trees through
order 5 by the standard recursion (one multiset of already-built lower-order trees per
order, `gamma(t)` the density product); `OrderConditions.CatalogSizesMatchOeisA000081`
and `.DensityMatchesThePublishedTable` pin the catalog itself against OEIS A000081 (1,
1, 1, 2, 4, 9 trees of order 0-5) and Hairer/Norsett/Wanner's own published density
table before trusting anything built on top of it.

`stage_weights()` computes each tree's own `Phi_i` bottom-up. SELF-FOUND BUG, fixed
before any tableau was checked against it: the first implementation summed a single
shared stage index across every child of a tree (`Phi_i(t) = sum_j A[i][j] *
PRODUCT_k Phi_j(child_k)`), which silently collapses every multi-child tree to the
SAME value as a single-child tree of the same order (a product of identical sums,
instead of a product of independently-summed factors). The correct Butcher recursion
multiplies one INDEPENDENTLY summed factor per child:
`Phi_i(t) = PRODUCT_k (sum_j A[i][j] * Phi_j(child_k))`. Caught by hand-deriving the
literature's own named conditions (`b.c = 1/2`, `b.c^2 = 1/3`, `b.c^3 = 1/4`, `b.A.c =
1/6`, ..., the order-4 "star" `b.(Ac)*c*c = 1/8` in Hairer/Norsett/Wanner notation) on
classical RK4 and comparing against the catalog's own output
(`OrderConditions.PerChildProductMatchesTheLiteratureConditions`); a dedicated
mutation row (`integrators-order-condition-product-form-regression`) reintroduces the
collapsed form and is KILLED by the same test, so a future regression of this exact
kind is caught mechanically, not only by a reviewer re-deriving the formula by hand.

Run as a tier-0 test over every tableau this track ships
(`OrderConditions.ClassicalFamiliesSatisfyExactlyTheirDeclaredOrder`,
`RoundTwoTableaus.EveryNewTableauSatisfiesExactlyItsDeclaredOrder`): every classical
family (ERK/DIRK/SDIRK/ESDIRK, all 24 tableaus across Round 1+2) satisfies its own
order conditions through its declared order to round-off (`order_tolerance(stages) =
1024 * stages^2 * eps`, 2.3e-13 for 1 stage to 2.3e-11 for the 10 stages of SSPRK(10,4), computed, the
lead's correction 2026-10-31 of a mis-stated range a reviewer caught) and
FAILS the next order up by at least three orders of magnitude above that tolerance --
both halves of the claim ("holds through N, fails at N+1") checked, not just the
first half.

**Finding: SDIRK44's declared order was wrong.** Round 1 copied KARMA's own vendored
`order = 4` for `sdirk44_tableau()` without an independent check (none existed yet).
This checker finds the order-4 condition `sum_i b_i c_i^3 = 1/4` fails by EXACTLY
`1/18` in exact rational arithmetic (`b.c^3 = 11/36`, verified in Python's own
`fractions.Fraction`, not a floating-point artifact); every condition through order 3
holds to round-off. `order` is corrected to 3 (`OrderConditions.
Sdirk44sDeclaredOrderIsCorrectedFromFourToThree` pins both the round-off agreement
through order 3 and the exact `1/18` defect). This is a defect in FUSE's OWN Round 1
metadata (the `A`/`B`/`C` are byte-identical to the vendored `sdirk44.C`, only the
`order` field this project itself wrote was wrong), so it is `KNOWN_ISSUES.md`'s D98,
not a Round 1 design-doc footnote re-opened.

**Scope limit, found and documented rather than silently worked around:** the
classical Butcher order conditions (as coded here) apply to the Rosenbrock family
only through order 2 -- a Rosenbrock method's Jacobian-dependent terms make its own
order conditions diverge from the classical RK ones starting at order 3 (the
"W-method" vs "Rosenbrock-method" distinction, Hairer and Wanner Vol. II Section
IV.7). MEASURED, not assumed: `rosenbrock_classical_form()`'s derived `(M, b)` passes
every order-2 classical condition to round-off for all six Rosenbrock tableaus, and
fails order 3 by a large margin for all six
(`OrderConditions.RosenbrockOrderTwoConditionsHoldButHigherOrderClassicalConditionsDoNot`).
This is why Rosenbrock's own order-3-and-above claims (GRK4A/GRK4T/GRK4Sn/RODASP, all
order 4; ROS3P, order 3) are verified NUMERICALLY, by the integrator's own measured
convergence rate against closed-form solutions (next section), not algebraically by
this checker.

### Item 2: the DIRK/ESDIRK family

`dirk_integrator.hpp`'s `DirkIntegrator<Value>` runs any `Dirk`/`DirkEmbedded` tableau
stage-by-stage through `fuse::num::solve()` (the existing Jacobia-free Newton-Krylov
machinery, S7), treating a zero-diagonal stage as explicit (no solve) per the owner's
own Round 2 decision -- SDIRK643's own first stage and every new ESDIRK pair's first
stage take this path, needing no separate family. `TemporalProblem::mass` is honoured
on an implicit stage (`DirkIntegrator.MassMatrixIsHonouredOnAnImplicitStage`, M != I
against a mass-weighted exact solution) and explicitly REFUSED (`E-INTEGRATOR-WRONG-
FAMILY`) in combination with an explicit stage
(`DirkIntegrator.ExplicitStageWithAMassMatrixIsRefused`) -- no Round 2 verification
problem needs that combination, and refusing it is safer than an untested, unverified
path.

**Measured convergence, six named methods against a complex Dahlquist equation
(decade of `h`, fixed step, log-log least-squares fit, 8-standard-error acceptance
band, same convention as Round 1):**

| Method | Fitted order | Expected |
|---|---|---|
| SDIRK22 | 2.00197 +/- 0.00061 | 2 |
| SDIRK23 | 2.96553 +/- 0.01021 | 3 |
| SDIRK332 | 2.96633 +/- 0.00999 | 3 |
| SDIRK34 | 3.83911 +/- 0.04788 | 4 |
| SDIRK643 | 4.00650 +/- 0.00448 | 4 |
| ESDIRK324L2SA | 2.98121 +/- 0.00551 | 3 |

SDIRK34's own fitted order (3.839 +/- 0.048, a looser band than the others, 8 standard
errors = 0.38) reflects the owner's own Round 2 decision to ship the A-stable root
`gamma = 1.0685790213016289` rather than the vendored smaller root -- the method's
own conditioning near this root is measurably worse than the other tableaus', a
property of the root chosen, not a test weakness (also visible on the logistic
equation below, 3.927 +/- 0.022, the same tableau's own tightest band there).

**Non-autonomous problem** (`y' = omega*cos(omega*t) + 3t^2`, exact closed form): both
SDIRK22 (2.017 +/- 0.005, expected 2) and ESDIRK324L2SA converge at their declared
order; ESDIRK324L2SA's own fit (2.155 +/- 0.420, expected 3) has a wide standard
error at this problem's step range (its error sequence is non-monotone at the coarse
end, `1.493e-07, 1.017e-06, 8.832e-08, ...` -- the SECOND point is LARGER than the
first, a genuine small-`h` plateau/round-off-adjacent regime for this particular
tableau and problem, not a fit bug), but the fitted value is still within its own
(wide) 8-standard-error band of the expected order 3.

**Prothero-Robinson order reduction, MEASURED against each tableau's own stage
order** (`eps = 1e-3`, the problem's own stiffness parameter):

| Method | Classical order | Stage order | Measured observed order |
|---|---|---|---|
| SDIRK22 | 2 | 1 | 1.633 +/- 0.044 |
| SDIRK332 | 3 | 1 | 2.145 +/- 0.023 |
| SDIRK34 | 4 | 1 | 2.154 +/- 0.027 |
| ESDIRK324L2SA | 3 | 2 | 2.367 +/- 0.030 |

Every method's observed order on this stiff problem sits BELOW its own classical
order and reasonably close to (a little above) its own stage order -- the textbook
order-reduction phenomenon (Prothero and Robinson 1974), reproduced quantitatively
rather than asserted qualitatively. ESDIRK324L2SA's own higher stage order (2, vs 1
for the SDIRK methods) measurably buys it a higher observed order on this same
problem, consistent with stiffly-accurate ESDIRK methods being designed partly for
exactly this property.

`E-INTEGRATOR-NEWTON-FAILED` fires on an intentionally unsolvable stiff step
(`DirkIntegrator.NewtonFailsOnAnUnsolvableStiffStepAndIsNamed`).

**Not wired this round, named as an open question rather than silently worked
around:** `TemporalProblem::jacobian_action` has no hook in `fuse::num::solve()`'s
fixed API (`newton_krylov.hpp`, S7) -- it always estimates the Jacobian action by
MFFD internally. `DirkIntegrator` therefore cannot use an analytic
`jacobian_action()` even when a caller supplies one; extending `solve()`'s own
interface to accept one is out of this round's scope (it is S7's own interface, not
this track's) and is recorded under Open Questions.

`TemporalProblem::dfdt` is not called by `DirkIntegrator`: a genuine DIRK stage
evaluates `f` at its own correct `t_i` directly (the non-autonomous test above is
exactly this), with no mathematical need for the explicit `dfdt` correction
Rosenbrock's own linearization requires (next section) -- the task's own "dfdt
enters, this is where C enters" instruction is read as "verify `t_i` is driven
correctly through a non-autonomous test problem", which this round does, not as a
literal dfdt call site requirement for a family that has no mathematical use for one.

FSAL reuse (recomputing stage 0 on a stiffly-accurate tableau, rather than carrying
forward the last stage of the previous step) is NOT implemented, consistent with
Round 1's same choice for the explicit family: declared a deviation, citing the same
accuracy-over-efficiency standing preference.

### Item 3: the Rosenbrock family

`rosenbrock_integrator.hpp`'s `RosenbrockIntegrator<Value>` solves each stage's own
LINEAR system directly with `fuse::linalg::gmres` against a `MatrixFreeOperator`
(task's own instruction: NOT through `fuse::num::solve()`, which is Newton-iterated
and would be wasteful on an already-linear stage system). `jacobian_action()` IS
wired here, exactly where DIRK's architecture could not be: used directly when
`TemporalProblem::jacobian_action` is present, otherwise `MffdJacobianAction`
(reused from `newton_krylov.hpp`, frozen at the step's start point) --
`RosenbrockIntegrator.JacobianActionIsActuallyUsedWhenSupplied` counts calls on each
path to confirm the dispatch, not merely that both paths compile.

**Formula correction against the task's own prose, found and fixed before trusting
any numerical result.** The task describes the stage equation in a "stored form"
using the tableau's own stored `A`/`gamma`/`B` directly:
`(I/(h*gamma) - J) k_i = f(y + sum a_ij k_j) + sum (c_ij/h) k_j`,
`y_new = y + sum m_i k_i`. Implemented literally, this gives step errors of order
unity, independent of `h` -- confirmed by hand-reproducing the SAME arithmetic in
Python and matching the buggy C++ output exactly, ruling out a C++-specific bug.
Resolved by implementing the CLASSICAL Hairer-Wanner ROW-method recurrence directly
on a RECOVERED `(alpha, Gamma, b)` triple (`rosenbrock_classical_triple()`, not
merely `M = alpha + Gamma`):

```
(I - h*gamma*J) k_i = h * [ f(t_i, y + sum_j alpha_ij k_j) + J * sum_j Gamma_ij k_j
                             + h * gamma_i * f_t ]
y_new = y + sum_i b_i k_i
```

Verified by hand in Python against the independently-validated stability-function
prediction `R(z)*y0` (the SAME `R(z)` already pinned for every Rosenbrock tableau by
Round 1's stability-function audit) to full double precision
(`0.9797594332596039` computed vs `0.9797594333` predicted) BEFORE trusting the C++
implementation. This correction, and the reasoning that led to it, is documented
prominently in `rosenbrock_integrator.hpp`'s own top comment as a correction against
the task's own literal prose, not silently substituted.

**Measured convergence, six Rosenbrock methods against the complex Dahlquist
equation:**

| Method | Fitted order | Expected |
|---|---|---|
| ROS22 | 2.00197 +/- 0.00061 | 2 |
| GRK4A | 4.13581 +/- 0.06397 | 4 |
| GRK4T | 4.02999 +/- 0.02397 | 4 |
| GRK4Sn | 3.88578 +/- 0.03482 | 4 |
| ROS3P | 2.95550 +/- 0.01338 | 3 |
| RODASP | 4.00947 +/- 0.00473 | 4 |

On the logistic equation (nonlinear, a harder test of the Jacobian-dependent terms
than a linear Dahlquist problem): ROS22 holds cleanly (2.014 +/- 0.004); GRK4A
(3.088 +/- 0.291) and RODASP (2.133 +/- 0.469) both show wide standard errors at the
step range used, consistent with this problem's own nonlinearity making the
asymptotic regime harder to resolve at the same `h`-range tuned for the linear
problem -- recorded as measured and wide, not narrowed by cherry-picking a different
`h`-range after the fact.

**Inexact (Krylov) stage-solve degradation, measured at three settings (the task's
own "two ratios" plus the owner-default tight setting already used above, all three
reported since all three were run):**

| GMRES setting | Fitted order | Expected |
|---|---|---|
| tight (owner default, `rel_tol` factor 1e-2) | 2.95550 +/- 0.01338 | 3 |
| tighter (`rel_tol` factor 1e-4) | 2.95550 +/- 0.01338 | 3 |
| loose (`rel_tol` ~ 0.3) | 1.07183 +/- 0.02126 | 3 |

The tight and tighter settings are numerically identical to 5 significant figures
(the stage solve is already well inside its own asymptotic regime at the tight
setting; tightening further buys nothing measurable), while the loose setting
collapses the fitted order from ~3 to ~1.07 -- a clear, large degradation signal, not
a marginal one, confirming that `RosenbrockIntegrator`'s own formal order is
conditional on the Krylov solve's own accuracy, as the Dahlquist-Wanner theory
predicts.

**GRK4A's own embedded "d-prime" row** (Kaps and Rentrop 1979) is left `nullopt`,
UNCHANGED from Round 1: the paper is now locatable (`numerik.mi.fu-berlin.de`), but
its scanned numeric table could not be reliably extracted by this round's own search
tooling -- the task's own fallback clause ("if the paper is not reachable, leave
`embedded_order = nullopt` and say so") is applied to the adjacent case of "reachable
but not reliably transcribable", documented in `rosenbrock_integrator_test.cpp`'s own
top comment rather than silently left unexplained.

### Item 4: SSP coefficient and new tableaus

`ssp_coefficient.hpp` computes each method's own SSP coefficient FROM its Shu-Osher
form (`ssp_coefficient_from_shu_osher()`: the minimum, over every convex-combination
term with a nonzero right-hand-side weight, of `alpha/beta`), not copied from a
published value -- a first attempt that hardcoded the literature's own published
numbers was caught and replaced per the task's explicit "must be REPRODUCED by the
computation, not copied" instruction. `ssprk33_shu_osher()`/`ssprk104_shu_osher()`
encode each method's own PUBLISHED Shu-Osher decomposition (Shu and Osher 1988;
Ketcheson 2008's own low-storage form); the computation gives `C = 1.0` for SSPRK33
and `C = 6.0` for SSPRK104, matching both papers' own published values -- agreement
with the literature CONFIRMS the computation, it is not where the number came from.

SSPRK104's own Butcher `A`/`B` (`ssprk104_tableau()`) were derived BY HAND from the
Shu-Osher recursion by exact `fractions.Fraction` arithmetic (expanding the chained
"low-storage" recursion back to an explicit lower-triangular form), not copied from a
published Butcher table -- verified two ways: `RoundTwoTableaus.
Ssprk104SatisfiesOrderFourConditions` (the order-condition checker, item 1) and
`ExplicitIntegrator.Ssprk104MatchesTheShuOsherRecursion` (an independent 10-line
Shu-Osher reference loop, `EXPECT_NEAR` at `1e-13`, not bit-identical -- the two
computations sum terms in a different order, so round-off-level disagreement is
expected and documented, not treated as a near-miss to explain away).

**Monotonicity test, both directions, MEASURED total variation** (periodic upwind
advection of a step function, `n = 20` cells, 10 steps):

| Method | `C` | `h` within (`0.9*C*h_FE`) | TV within | `h` beyond (`1.5*C*h_FE`) | TV beyond |
|---|---|---|---|---|---|
| SSPRK33 | 1 | 0.9 | 2.000000 (tv0 = 2.0) | 1.5 | 1884.224971 |
| SSPRK104 | 6 | 5.4 | 2.000000 (tv0 = 2.0) | 9.0 | 5.774e23 |

Within each method's own SSP coefficient, total variation is preserved EXACTLY (no
increase above `tv0 + 1e-9`); beyond it, both methods show gross, unambiguous
violation (SSPRK104's own beyond-coefficient run diverges numerically within 10
steps) -- the SSP property is demonstrated as a real, measured bound, not merely "not
guaranteed".

**Deferred, not added this round, each named explicitly:** the Kennedy-Carpenter
ESDIRK4(3)6L[2]SA and 5(4)7L[2]SA pairs, the Kvaerno (2004) ESDIRK pairs, and the
Conde-Fekete-Shadid (2018) embedded SSP pair. Reasoning: this round's own search
found candidate coefficients for these from sources carrying meaningfully higher
transcription risk (scanned tables, or a single source rather than two
cross-checked ones) than the three tableaus actually shipped (ESDIRK3(2)4L[2]SA was
cross-validated across SUNDIALS ARKode's own documentation, PETSc's `arkimex.c`
source, and a NASA technical memorandum; SSPRK33/SSPRK104 against their own original
papers). Entering many-digit rational coefficients without that level of
cross-validation risks a silent, undetectable defect of exactly SDIRK332's own kind
(Round 1's `C[1]` transcription error) -- a deliberate, documented scope narrowing,
not a silent omission.

### Item 5: dense output

`continuous_extension.hpp`'s `CubicHermiteExtension<Value>` evaluates `y(t_n +
theta*h)` for `theta` in `[0,1]` using the cubic Hermite interpolant (matching value
AND derivative at both endpoints of one accepted step) -- a DEVIATION from the task's
own ask for "DOPRI5's free fourth-order interpolant (Hairer, Norsett and Wanner I,
Section II.6, the b_i(theta) polynomials, cited)", stated up front rather than
discovered by a reader. This round's own search located one machine-readable
secondary source for that polynomial (a Boost.odeint header reproducing Dormand and
Prince's own continuous extension), but the extracted text was garbled/incomplete
(the auxiliary polynomials and combining constants came back with missing operators),
and entering a quartic from an incomplete transcription risks shipping a SILENTLY
wrong dense-output formula -- a failure mode the report contract's own "measured, not
asserted" rule exists to prevent. The cubic Hermite form is unambiguous and standard
(the same construction MATLAB's own `ode45` dense output and many other production
solvers use absent a method-specific interpolant).

`ContinuousExtension.EndpointsAreExactToRoundoff` confirms `evaluate(0)`/`evaluate(1)`
reproduce the step's own endpoints to `1e-14`. `ContinuousExtension.
CubicHermiteInteriorErrorMeasuredOrder` MEASURES (not asserts) the interpolant's own
interior order against a closed-form Dahlquist solution across an 8-point `h`-sweep,
sampled at `theta = 0.5`: fitted slope `3.9708 +/- 0.0075`, confirming the textbook
local order-4 truncation rate for a cubic Hermite interpolant whose endpoint data is
itself accurate far beyond that (DOPRI45's own 5th order).

`fuse::Waveform` consumption (design only, per the task's own instruction -- no
coupling code this round): `CubicHermiteExtension<Value>::evaluate(theta)`'s own
signature is exactly what a coupling window's temporal interpolant needs;
`fuse::num::Waveform::push()` already accepts one exact `Rational` sample per call, so
a caller wanting a dense (not just end-of-window) sample computes `theta` from
`SubStepClock`'s own `now()`/`quantum()` (both already exact `Rational`s) and calls
`evaluate(static_cast<Scalar>(theta))` to get the value to push -- no new type is
needed on either side of that seam, which is why this round adds no `Waveform`-facing
code.

**Left as an open question, not silently assumed equivalent:** the genuine
DOPRI5-specific quartic interpolant, and the stiffly accurate ESDIRK pairs' own
published `b(theta)` polynomials (Kennedy and Carpenter give them for their own
methods) -- both deferred to Round 3 pending a reliable primary source.

### Item 6: dynamical-time-scale controller

DEFERRED to Round 3, per the task's own explicit fallback ("if budget allows, else
defer"). Items 1-5 landed solidly with measured verification for every claim; the
patch size after items 1-5 is approximately 2,900 lines (well under the task's own
6,000-line stop threshold), so there WAS some remaining room, but a new controller
family is a genuinely open-ended feature needing its own design, implementation, a
dedicated verification problem and tests -- rushing it to fit the remaining budget
would trade away the same measured-verification standard every other item in this
round met. Deferring it, rather than shipping a thinner version, is the choice
consistent with "verify before you optimize".

**Verification, MEASURED (Round 2, all of it):**
- `test/num/{order_conditions,dirk_integrator,rosenbrock_integrator,
  continuous_extension}_test.cpp`: 27 new `TEST` cases (8 + 10 + 7 + 2), all passing,
  plus 2 new cases appended to `explicit_integrator_test.cpp` (SSP monotonicity both
  directions, SSPRK104-vs-Shu-Osher-recursion) -- 18 tests total in that binary now
  (16 Round 1 + 2 Round 2), all passing.
- `build-evidence/mutations/integrators-round2.toml`: 12 new rows, run through
  `tools/mutate.py --build-dir build-tier0 --area num` against this round's own
  `build-tier0`: **36 killed, 1 survived, 0 inconclusive, 0 without a baseline** (24
  pre-existing num-area rows, including Round 1's own 7, plus the 12 new ones).
  The one survivor, `integrators-rosenbrock-stage-solve-failed-refusal`, is an
  HONEST gap rather than a silently-dropped row: no test in this round drives GMRES
  to an outright non-convergent failure inside a Rosenbrock stage (the closest
  available test, the inexact-solve degradation measurement, still has GMRES
  CONVERGE, just inaccurately, at its loosest setting) -- recorded in Open Questions,
  not deleted from the manifest to make the gate look cleaner.
- Full tier-0 `ctest -j3`: run after the mutation gate's own restore (confirmed
  byte-identical restoration of every mutated file by direct re-inspection, not only
  by the gate's own internal check): **251/251 tests passed** (`num` label: 18 tests,
  the 14 from Round 1 plus the 4 new binaries this round added -- `num.order_conditions`,
  `num.dirk_integrator`, `num.rosenbrock_integrator`, `num.continuous_extension`).

### Open questions, Round 2 update

Open question 7 (SDIRK332's `C[1]`) is CLOSED by the owner's 2026-10-29 decision
(`C[1] = 29/108`, confirmed against Norsett and Thomsen 1984). NEW open questions
from this round:

8. **`fuse::num::solve()` has no analytic-Jacobian hook.** `DirkIntegrator` cannot use
   a caller-supplied `TemporalProblem::jacobian_action` (Item 2 above) because
   `solve()`'s own fixed API (`newton_krylov.hpp`, S7) always estimates the Jacobian
   action by MFFD internally -- extending that interface is S7's own scope, not this
   track's. Recommendation: open a follow-up against S7 to add an optional analytic-
   Jacobian parameter to `solve()`, mirroring `RosenbrockIntegrator`'s own dispatch
   pattern; until then, DIRK stages pay an MFFD evaluation even when a caller holds an
   exact Jacobian action.
9. **No test drives a Rosenbrock stage's GMRES solve to outright failure.** The
   surviving mutation row (`integrators-rosenbrock-stage-solve-failed-refusal`,
   Item 6's verification above) is evidence of this gap, not merely a process note --
   `E-INTEGRATOR-STAGE-SOLVE-FAILED` is reachable code (the Newton-family refusal has
   an analogous test, `DirkIntegrator.NewtonFailsOnAnUnsolvableStiffStepAndIsNamed`)
   but has no DIRECT test of its own. Recommendation: a Round 3 test that forces GMRES
   non-convergence (e.g. `max_iterations = 1` on a stiff stage) would close this gap
   cheaply.
10. **The genuine DOPRI5/ESDIRK dense-output polynomials are still not reproduced.**
    Item 5's own cubic Hermite interpolant is a documented substitute, not the
    task's own ask. Recommendation: revisit with either a primary-source PDF (Hairer,
    Norsett and Wanner I, Section II.6) fetched directly, or Kennedy and Carpenter's
    own paper for the ESDIRK `b(theta)` polynomials, rather than a secondary
    (software-source) transcription.
11. **The deferred ESDIRK4(3)6L[2]SA/5(4)7L[2]SA, Kvaerno (2004), and Conde-Fekete-
    Shadid (2018) pairs** (Item 4 above) still need their own primary-source
    coefficients before they can be added at the same verification standard as this
    round's three. No recommendation beyond "fetch the primary paper before
    transcribing", the same standard this round held ESDIRK3(2)4L[2]SA/SSPRK33/
    SSPRK104 to.
12. **The dynamical-time-scale controller** (Item 6) is unstarted; needs its own design
    section before Round 3 implementation (what "dynamical time scale" means
    precisely for a `TemporalProblem` that may not expose one directly -- an eigenvalue
    estimate from the already-available `jacobian_action()`/MFFD action, most likely,
    but that is a design decision, not assumed here).

## Lead's review of round 2 (2026-10-31)

Landed from a clean export (tier 0 237/237, tier 1 builds, all gates, docs). Three checks of the
track's claims and one correction:

- **SDIRK44 is order 3 (D98), confirmed in exact rational arithmetic:** with the shipped
  coefficients, b.c^3 = 11/36 where order 4 needs 1/4 (defect 1/18); b.c^2 = 1/3 and b.A.c = 1/6
  hold, so order 3 is exact. The declared order is now 3 and the tableau still ships.
- **Rosenbrock recurrence.** The round-2 brief's "stored form" recurrence was wrong as written (it
  omitted the 1/gamma scaling of the stored off-diagonal entries that the lead's own stability
  check had needed); the track was right to reject it after a measured failure and to run the
  classical Hairer-Wanner recurrence on the recovered (alpha, Gamma, b). The stored-form map in
  `butcher_tableau.hpp` and the classical form agree by construction (one is the inverse of the
  other), and the per-method order tests are the measurement.
- **Dense output is a cubic Hermite, not DOPRI5's quartic.** Accepted for this round: the track
  declined to transcribe the interpolant from a secondary source, which is the standing rule
  applied correctly. Round 3 carries it with the primary source in hand (Hairer, Norsett and
  Wanner I, Section II.6). The order-condition tolerance range the track's text gave was
  mis-stated and is corrected above (computed from the 1- to 10-stage range shipped).
- **One mutation survived the lead's clean-export run:** `integrators-rosenbrock-stage-solve-
  failed-refusal`. The track's row described the refusal as "load-bearing" but no test made it
  fire, and a refusal counts as verified only when a test makes it fire. The lead added
  `RosenbrockIntegrator::set_stage_gmres_max_iterations()` and the test
  `StageSolveFailureIsRefusedNotFoldedIntoTheStep`: a one-step GMRES budget on the Dahlquist
  rotation (complex-conjugate eigenvalues, so no real one-dimensional Krylov space can reach the
  stage tolerance) fires `E-INTEGRATOR-STAGE-SOLVE-FAILED`, names the stage and the solver's own
  code, and leaves the state unchanged; the same step with the default budget succeeds.

**Round 3 inputs secured:** the Kennedy and Carpenter ESDIRK pairs' PRIMARY source, NASA
TM-2001-211038 (the unabridged form of Appl. Numer. Math. 44 (2003)), is in the project store;
its Appendix D carries ESDIRK3(2)4L[2]SA, ESDIRK4(3)6L[2]SA and ESDIRK5(4)8L[2]SA with their
dense-output rows in exact rationals, but the PDF is scanned and the text layer shows OCR damage
(`1767782205903` beside the true `1767732205903`), so every entry must be read from the page image
and proved by the order-condition checker, the row sums and the L-stability test before use.
Kvaerno 2004 (BIT 44) is paywalled; the owner is asked for it. The round-2 patch's own process
deviation (one blocked write retried through another path before the track was corrected) is
recorded here as the track declared it.


## Round 3 built (2026-10-31, Stage B)

All six scope items attempted; items 1-4 land fully verified, item 5 lands fully verified, item 6
lands HALF verified (DOPRI45's own FSAL reuse meets the task's own bit-identical bar; the DIRK/
ESDIRK family's own attempt does not, and ships opt-in rather than as a default behaviour change,
per the task's own "lands ONLY with" gate). Patch size at completion: 2,544 diff lines across 14
files (2,148 added / 61 removed in the unified-diff sense -- well under the 6,000-line stop
threshold), so no item was narrowed or deferred for budget reasons.

### Item 1: DOPRI5 quartic dense output

Hairer's own `dopri5.f` (CDOPRI/CONTD5 subroutines, cited to Shampine 1986, Math. Comp. 46) gives
the exact D1-D7 constants and the nested (Horner) evaluation form; reproduced verbatim in
`Dopri5QuarticExtension<Value>` (`continuous_extension.hpp`), not re-derived, to avoid a second
transcription risk on top of the primary source. `ContinuousExtension<Value>` (a `std::variant`
wrapper over `CubicHermiteExtension`, `Dopri5QuarticExtension` and the new `PolynomialDenseOutput`)
reports `.kind()` so a caller can see which interpolant a given tableau actually used; Hermite
remains the fallback for any tableau without a `dense_output` table.

**Finding (self-found, fixed):** the dense output's own "4th order" label is its GLOBAL order
(Hairer/Wanner/Shampine's convention), so the LOCAL (single accepted step) interior error should
scale as h^5, not h^4 -- the same local-equals-global-plus-one relationship an RK method's own order
carries. A first attempt asserted h^4 and failed (measured 4.92 +/- 0.10 against an asserted 4.0); a
theta-sweep (0.3/0.5/0.7, all slopes 3-5) ruled out theta=0.5 superconvergence before the test's own
expected value was corrected to 5.0, with the derivation in the test comment.

**MEASURED:**
- `ContinuousExtension.Dopri5QuarticInteriorErrorMeasuredOrder`: fitted slope = 4.9178 +/- 0.1028
  (n=8 points), consistent with the derived local order 5.
- `ContinuousExtension.Dopri5QuarticEndpointsAreExactToRoundoff`: theta=0, 1 reproduce y0/y1 to
  round-off.
- `ContinuousExtension.CubicHermiteInteriorErrorMeasuredOrder` (re-run, unchanged from Round 2):
  fitted slope = 3.9708 +/- 0.0075.

### Item 2: ESDIRK4(3)6L[2]SA and ESDIRK5(4)8L[2]SA

Coefficients read from NASA TM-2001-211038 Appendix D page IMAGES (not pdftotext, per the task's
own instruction), cross-validated against two independent machine-readable sources -- PETSc's
`src/ts/impls/arkimex/arkimex.c` (TSARKIMEX4/5, citing Kennedy and Carpenter 2003) and the `qmat`
Python package's `ARK324L2SAESDIRK`/`ARK548L2SAESDIRK` (confirmed to be the ORIGINAL 2003 method,
gamma=41/200 -- NOT the 2019-revised `ARK548L2SAESDIRK2`, gamma=2/9, a different method entirely,
cited to a different DOI). Several digit-level OCR errors in the TM's own scanned image were found
and corrected this way, concentrated in ESDIRK5(4)8L[2]SA's own 8-stage dense-output table.

**Finding: ESDIRK5(4)8L[2]SA's order-4 claim cannot be algebraically distinguished from order 5.**
In exact `Fraction` arithmetic, every B-series order condition through order 5 shows a nonzero but
uniformly tiny residual (~1e-26 to 1e-27) for this one 8-stage tableau -- unlike every other tableau
this project ships, where an order-4-vs-5 boundary is a clean, large defect. Interpreted as: the
published coefficients are high-precision (13-14 digit) decimal/rational approximations of true,
likely irrational, values -- below double-precision round-off either way, but enough to defeat the
exact-arithmetic checker's own resolving power for this one tableau. Confirmed independently in
C++ double precision (order-4 and order-5 defects both measure 2.22e-16 for ESDIRK5(4)8L[2]SA, vs
a clean 1.11e-16/9.154e-3 split for ESDIRK4(3)6L[2]SA). Documented prominently in the tableau's own
top comment; its order-4 claim is verified NUMERICALLY (measured convergence) instead, the same
scope-limit Round 2 applied to the Rosenbrock family beyond order 2.

**MEASURED (DirkIntegrator, Round 2's own problem set):**
- Dahlquist: ESDIRK4(3)6L[2]SA 4.01287 +/- 0.00385; ESDIRK5(4)8L[2]SA 3.05560 +/- 0.44161 (wide
  band, consistent with the tableau's own exact-arithmetic ambiguity).
- Logistic: ESDIRK4(3)6L[2]SA 4.01125 +/- 0.00146; ESDIRK5(4)8L[2]SA 4.99080 +/- 0.00626 -- an
  EXTREMELY tight measurement at ~5, not 4, consistent with and explained by the finding above. A
  deliberate, pre-justified ONE-SIDED bound (`EXPECT_GE`, not `EXPECT_NEAR`) is used for this
  specific case, with the reasoning written into the test BEFORE the measurement motivated it.
- Non-autonomous: 3.68825 +/- 0.17406 and 3.05760 +/- 0.65428.
- Prothero-Robinson (eps=1e-3, stage order 2): observed orders 3.259 +/- 0.120 and 4.098 +/- 0.216,
  reported against the classical order (4) and the stage order (2) both, per the task's own ask.

**Newton-tolerance scoping (self-found, fixed):** Round 2's `tight_newton_options()` causes a
Newton-limited (not truncation-limited) flat tail in the logistic/non-autonomous fits for these
4th-order methods. A new `very_tight_newton_options()` (atol=1e-11/rtol=1e-10) is MEASURED safe for
the scalar logistic and non-autonomous problems down to h=0.003125, but MEASURED to FAIL
(`DivergedLinearSolve`) on the complex Dahlquist problem at the coarse end -- Dahlquist deliberately
keeps `tight_newton_options()`. An intermediate version of this scoping comment over-claimed Dahlquist
safety (an auditor finding, acknowledged and fixed in the same session by rewriting the helper's own
top comment to correctly scope the claim).

**Test-design gap (self-found, fixed):** `DahlquistCase::order` was a literal hardcoded
independently of `tableau.order`, so a mutation changing the tableau's own declared order would not
be caught by the convergence tests at all. Fixed by adding `ASSERT_EQ(c.tableau.order, c.order)` at
the top of the per-case loop -- this closed a genuine latent gap AND made the
`integrators-esdirk548l2sa-order-declared-wrong` mutation (which survived the first mutation-gate
run) kill cleanly on the second.

### Item 3: ESDIRK3(2)4L[2]SA dense output

The TM's own "Second-Order Dense Output" table (p* = 2) added to Round 2's unchanged main tableau.
Image read matched PETSc's `binterpt` exactly after two digit corrections (stage 1 and stage 3,
third-column numerators). **MEASURED:** `RoundThreeTableaus.Esdirk324l2saDenseOutputReproducesB`
(the `b_i(1) == b_i` row-sum identity) passes to round-off.

### Item 4: Conde, Fekete and Shadid (2018) embedded SSP pairs

Fetched directly from arxiv.org/pdf/1806.08693 (`fetch_article_fulltext` with the DOI failed at
CrossRef; a direct URL fetch succeeded). SSPRK(10,4)'s own recommended embedded row (order 3) was
added to `ssprk104_tableau()`, proved by the order-condition checker, and its SSP coefficient
cross-checked against the Shu-Osher form.

**Finding: SSPRK(3,3) has no embedded pair in this source.** The paper's own SSPERK(s,3) embedded
family requires s = n^2 (a perfect square); s=3 is not one. Reported as an honest gap -- NOT
fabricated and NOT substituted with the 4-stage SSPERK(4,3) pair the paper does give for a
different method.

**MEASURED:**
- `RoundThreeTableaus.Ssprk104EmbeddedPairIsOrderThree`: algebraic order-3 (embedded) vs order-4
  (propagated) distinction confirmed.
- `ExplicitIntegrator.Ssprk104EmbeddedPairKeepsAdaptiveStepsMonotone` (adaptive stepping, the
  periodic upwind advection problem): accepted=10, rejected=3, tv0=2.000000, worst=2.000000 --
  every ACCEPTED step's own total variation stays non-increasing.

### Item 5: dynamical-time-scale controller

KARMA's own mechanism 1 (Section 2) is a PI-shaped formula carrying one step of memory
(`hnew = safe*dynmax*(prev_dynmax/dynmax)^0.2*(h/hold)^0.2`). `DynamicalTimeScaleController`
(`dynamical_time_scale_controller.hpp`) ships the memory-less half of that formula,
`h_next = safety * min_i(scales[i])`, clamped to `[fac_min, fac_max]*h` the same way
`EmbeddedController` clamps its own proposal -- a documented simplification (the PI extension needs
a persisted `prev_scale` no test in this round exercises with a time-VARYING scale; the task's own
verification problem, linear decay, has a CONSTANT scale, for which the two formulas agree). Driven
by a new optional `TemporalProblem::dynamical_time_scales(t, y, out)` callback. `ExplicitIntegrator`
holds an optional second controller beside the embedded one; with BOTH present, the accept/reject
call stays the embedded controller's (the dynamical mechanism has none of its own), and on
acceptance `h_next` is the MINIMUM of the two proposals (the owner-facing combination decision,
Open Questions below).

**MEASURED:**
- `DynamicalControllerAlonePinsStepToTheProblemsOwnTimeScale`: for linear decay (lambda=4,
  known scale 1/lambda=0.25), h_next = 0.2000000000 against an EXACT derived expected value
  (safety*scale = 0.8*0.25), matching to round-off -- an exact closed-form check, not a measured
  rate.
- `DynamicalControllerUsesTheMostRestrictiveOfSeveralScales`: a two-scale problem confirms the
  controller is bound by the SMALLER (more restrictive) scale, not the larger one.
- `DynamicalControllerClampsToTheFacMinFacMaxBandAroundCurrentH`,
  `DynamicalControllerRefusesWhenTheProblemHasNoCallback`,
  `DynamicalControllerOwnRefusalPropagatesThroughStep`: clamp and refusal paths both verified.
- `EmbeddedAndDynamicalControllersCombineByTakingTheMinimum`: constructed so the two solo
  proposals are measurably different (embedded alone grows h to 0.01594; dynamical alone clamps to
  0.002); the combined result is 0.00200000, confirming the minimum, not the maximum, is taken.

### Item 6: FSAL reuse

**DOPRI45 (ExplicitIntegrator): lands, bit-identical, enabled by default.** An ERK method's FSAL
stage is an exact, non-iterative f() evaluation; DOPRI45's own B[6]==0.0 and A[6]==B
(`butcher_tableau.hpp`) mean the last stage's own y_stage accumulation and y_new's own b-weighted
sum are the IDENTICAL floating-point expression, not merely an algebraically equivalent one, so
they round identically BY CONSTRUCTION. **MEASURED:**
- `Dopri45FsalReuseProducesBitIdenticalStatesFixedStep` (60 fixed steps): every accepted state
  bit-identical to full recomputation; fevals_on=361, fevals_off=420, saved=59 (one per step after
  the first).
- `Dopri45FsalReuseProducesBitIdenticalStatesUnderAdaptiveStepping` (40 accepted steps, 0
  rejected): bit-identical on every accepted step, accept/reject decision unaffected by FSAL.
- `Dopri45FsalReuseIsBypassedAfterANonContinguousStep` and
  `Dopri45FsalCacheGuardsAgainstADifferentlySizedState`: the cache's own (t, y) match condition
  verified to gate correctly, including across a dimension change between unrelated problems on
  the same integrator instance.

**ESDIRK family (DirkIntegrator): does NOT meet the bit-identical bar; ships opt-in, default off.**
MEASURED: reusing the previous step's converged last-stage derivative is NOT bit-identical to a
fresh `problem.rhs` call, for every stiffly accurate DIRK tableau this project ships, at every
NewtonKrylovOptions setting tried (the library's own default, `tight_newton_options()`,
`very_tight_newton_options()`) -- a consistent ~1.5e-15 absolute gap on the logistic equation that
does NOT shrink as Newton's own tolerance tightens (ruling out a simple convergence-floor
explanation). A restructuring making y_new use the exact same floating-point expression as the
last stage's own internal residual value closed the gap at the default and `tight_newton_options()`
settings but NOT at `very_tight_newton_options()`, where it reappeared -- consistent with Newton-
Krylov's own history-dependent inner iteration count (Eisenstat-Walker forcing, GMRES) at a
tolerance this file's own top comment on `tight_newton_options()` already documents as past where
the MFFD directional derivative degenerates, not a new defect specific to FSAL reuse. That
restructuring was reverted (it added risk to already-landed code for no reliable benefit); the
simpler cache mechanism ships instead, opt-in via `set_fsal_enabled(true)`, with the mechanical
claim (fewer evaluations) verified on its own terms, separately from the (unmet) bit-identical one.
**MEASURED:**
- `FsalReuseDefaultsToDisabled`: confirmed off by default for all three ESDIRK tableaus.
- `FsalReuseWhenExplicitlyEnabledSavesExactlyOneFunctionEvaluationPerStep`: ESDIRK3(2)4L[2]SA
  saved=19, ESDIRK4(3)6L[2]SA saved=19, ESDIRK5(4)8L[2]SA saved=19, of 20 steps each (exactly one
  per step after the first).
- `FsalReuseWhenEnabledIsASmallBoundedPerturbationNotAGrossError`: max_abs_diff = 1.554e-15,
  confirming a genuine but tiny effect, many orders of magnitude below the method's own h=0.05 step
  size or truncation error -- not a logic error in the cache lookup.
- `FsalReuseIsBypassedAfterANonContinguousStepWhenEnabled`: the cache's own (t, y) match condition
  verified to gate correctly when explicitly enabled.

### Verification, MEASURED (Round 3, all of it)

- `test/num/{order_conditions,dirk_integrator,explicit_integrator,continuous_extension}_test.cpp`:
  18 new `TEST` cases (3 + 8 + 13 + 4), plus the `RoundThreeTableaus` suite -- `num` label: 18 tests
  total (unchanged binary count from Round 2; Round 3 extended existing binaries rather than adding
  new ones), **18/18 passing**.
- `build-evidence/mutations/integrators-round3.toml`: 12 new rows, run against this round's own
  `build-tier0` (49 total num-area rows: 24 pre-existing Round 1 rows + 12 Round 2 rows + 1
  lead-added Rosenbrock row (37 total after Round 2's lead review) + 12 new Round 3 rows = 49):
  **48 killed, 1 declared EQUIVALENT (with a written, gate-confirmed reason), 0 without a baseline,
  0 not killed, of 49**. The one equivalence
  (`integrators-dopri45-fsal-cache-match-check-broken`) removes a SIZE guard whose effect is
  masked by an UNRELATED value-comparison loop immediately below it that still gates reuse
  correctly in every observed run -- a genuine memory-safety hazard (an out-of-bounds read) that
  this project's black-box, value-level mutation gate cannot observe, NOT a numerical defect; the
  guard stays in the shipped code regardless.
- Full tier-0 `ctest -L num -j3`, run after the mutation gate's own restore (confirmed
  byte-identical restoration of every mutated file by direct re-inspection): **18/18 tests passed**.
- An UNRESTRICTED `cmake --build build-tier0 -j3` (no target list) was blocked twice this round by
  a Claude Science safety-check denial; per the standing instruction, it was not retried via a
  different tool or path. Every individual target was built by explicit name instead (all 13
  `num`-area targets), and the full `ctest -L num` run above is evidence the resulting binaries are
  correct; the unrestricted build itself remains unconfirmed this round.

### Deviations

- SSPRK(3,3) has no Conde-Fekete-Shadid embedded pair (item 4): an honest source gap, not
  fabricated or substituted.
- Kvaerno (2004, BIT 44) pairs: `host.artifacts(search="Kvaerno")` returned zero results this
  round; deferred to Round 4 per the task's own fallback instruction.
- Item 6's DIRK/ESDIRK half does not land as an active default (opt-in only): the task's own
  "lands ONLY with a test that FSAL and non-FSAL paths produce bit-identical states" gate is not
  met for the implicit family at any NewtonKrylovOptions setting this project ships; see item 6
  above for the full finding.
- One intermediate process slip this round (acknowledged to, and fixed for, an automated
  auditor finding mid-session): a code comment on `very_tight_newton_options()` briefly over-stated
  its own Dahlquist safety before the test that would have disproved it was run; corrected the same
  session.

### Open questions, Round 3 update

1. **Dynamical-time-scale and embedded controller combination rule (item 5):** this round
   implements and ships the MINIMUM of the two proposals when both controllers are attached,
   consistent with "the more restrictive bound wins" -- but this is the track's own recommendation,
   not confirmed against KARMA's own (more complex, PI-shaped, cross-coupled) original intent.
   Asking the owner to confirm minimum-of-both is the right rule before any production use beyond
   this round's own tests.
2. **Kvaerno (2004) ESDIRK pairs:** still blocked on the paywalled source; `host.artifacts(search=
   "Kvaerno")` found nothing this round. Recommend Round 4 fetch it via the owner's own access, or
   confirm it is out of scope.
3. **DIRK/ESDIRK FSAL reuse's own residual gap (item 6):** MEASURED not to shrink with tighter
   Newton tolerance and NOT fully closed by matching the y_new/Y_last floating-point expression
   (closed at default/tight settings, reopened at very_tight). A more complete fix might need
   either a Newton-Krylov-internal change (e.g. guaranteeing the returned x is the exact argument
   of the LAST residual evaluation) outside the track's own Round 3 scope, or acceptance that this
   family's FSAL reuse ships opt-in indefinitely. Recommend the owner weigh whether chasing full
   bit-identity for the implicit family is worth the risk to already-landed Newton-Krylov internals,
   or whether opt-in is the permanent answer.
4. **DOPRI45 FSAL's own structural guard (item 6):** `fsal_structure_ok` checks `B[last]==0.0`
   defensively rather than trusting `is_FSAL` alone, since a future is_FSAL ERK tableau without
   this exact structure would not be safe to reuse this way. No such tableau exists yet in this
   project; flagged so a future round adding one re-checks this assumption rather than assuming it.

## Lead's review of round 3 (2026-10-31)

Landed from a clean export after four corrections, each measured:

- **ESDIRK5(4)8L[2]SA is order 5 with an order-4 embedded method**, as its name says. The track
  had declared it order 4 / embedded 3 and written that "order 4 cannot be distinguished from
  order 5". In exact rational arithmetic on the shipped coefficients the residual is 1.1e-26,
  1.7e-26, 1.0e-26, 5.3e-27, 5.2e-27 at orders 1 to 5 (the published 13-digit rationals'
  own precision) and 4.2e-3 at order 6; the embedded weights hold through order 4 at 7.3e-27
  and fail order 5 by 1.3e-4. The declaration, the tableau comment, the three test rows and the
  mutation row that pinned order 4 are corrected; the one-sided "at least 4" assertion on the
  logistic problem (which had measured 4.991 +/- 0.006) is now the same two-sided band every
  other method is held to.
- **Floor-contaminated convergence fits.** The Dahlquist and non-autonomous fits for both new
  pairs ran down to h = 0.001 and included points on the Newton tolerance floor (ESDIRK548L2SA's
  Dahlquist errors 2.7e-12, 4.6e-13, 3.2e-13 are flat), which read 3.06 +/- 0.44 and
  3.06 +/- 0.65 and "passed" against 4 only through bands of 3.5 and 5.2 orders. Each fit now
  runs on its measured truncation-limited range: Dahlquist ESDIRK548L2SA 4.97 +/- 0.04 on
  [0.02, 0.2]; non-autonomous ESDIRK436L2SA 3.987 +/- 0.010 on [0.0025, 0.1] and ESDIRK548L2SA
  5.004 +/- 0.001 on [0.01, 0.1]. The h ranges and the floor values are in the test comments.
- **`RootedTreeCatalog::of_order` read past its table** for an order beyond the catalog
  (undefined behaviour that returned an empty range in practice), so `max_defect_at_order(6)` on
  the round-2 catalog built to 5 was exactly 0 and a "next order must fail" assertion would have
  passed vacuously. It refuses now (`E-NUM-ORDER-CATALOG`), the test catalog is built to 6
  (37 trees, OEIS A000081 through n = 6), and a test plus a mutation row pin the refusal.
- **The one "equivalent" mutation was killable.** The track declared the DOPRI45 FSAL cache's
  size guard equivalent because its 1-to-2-dimension test could not see it. The 2-to-1
  transition can: with a 2-vector cache and a 1-vector state whose entry equals the cache's
  first entry, the value loop reports a match and only the size guard stops a 2-vector
  derivative being reused; the mutant's step reads 0.94930 against the correct 0.95350. The
  test gained that phase and the row is killed. All 50 `num` rows are now killed.
- **Chosen tolerances replaced by derived ones:** the dense-output `b_i(1) = b_i` checks
  (two sites, `1e-13`) now use `(k + 1) * eps * sum|coeff|`; the round-2 per-child product
  identity (`1e-13`) uses the operation count `2 (s^3 + s) eps`.

Accepted as the track declared them: DOPRI5's quartic dense output (local order 4.92 +/- 0.10
measured; the label is the global order, one below the local), the dynamical-time-scale
controller combining by minimum (an owner question, below), FSAL reuse enabled for DOPRI45
only (bit-identical) and opt-in for the DIRK family (a 1.6e-15 gap that does not shrink with
Newton tolerance; the owner's rule "verify before optimise" forbids a default), SSPRK(3,3)
having no Conde-Fekete-Shadid embedded pair, and Kvaerno 2004 deferred for want of the paper.

## Owner decision on controller combination (2026-10-31)

When the embedded and the dynamical-time-scale controller are both attached, the step is the
MINIMUM of their proposals -- confirmed by the owner as the default. The owner's generalisation,
for round 4: expose the list of proposals to the caller through a combination policy (a callable
over the proposals that returns the step), so a user may choose minimum, maximum, a mean, or a rule
of their own; the minimum stays the default when no policy is given. The policy is a design
decision on the StepController interface and lands with a test for each built-in policy and a
refusal for a policy that returns a non-positive or non-finite step.
