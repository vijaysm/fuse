# Time integrators

`fuse::num` ships a set of Runge-Kutta and Rosenbrock time integrators that a component can use
inside its own `advance()`: explicit RK, diagonally implicit (DIRK and ESDIRK), Rosenbrock, and
strong-stability-preserving (SSP) methods, with embedded-pair step-size control and a cubic Hermite
continuous extension. They are plain C++ headers under `fuse/num/` that are installed with the
library. They advance a state vector that you own, one step at a time; they do not know about the
coupling graph, the schedule or the `Runner`, and they do not replace your component. Use them when
the physics inside a component is an ODE system `y' = f(t, y)` and you would rather pick a method
by name than write a stage loop.

The step sizes you pass to an integrator are yours to choose. A component that sub-steps inside one
coupling interval keeps its own clock; see [Sub-stepping inside a window](#sub-stepping-inside-a-window).

## What is available

| Header | Provides |
|---|---|
| `fuse/num/integrator.hpp` | `TemporalProblem<Value>` (the right-hand side and its optional hooks), `Integrator<Value>` (the interface every family implements), `StepRecord<Value>`, `SubStepClock`, `serial_rms_norm` |
| `fuse/num/butcher_tableau.hpp` | `ButcherTableau` and one factory function per method |
| `fuse/num/explicit_integrator.hpp` | `ExplicitIntegrator<Value>`: fixed step or adaptive, for the `Erk` and `ErkEmbedded` tableaus |
| `fuse/num/dirk_integrator.hpp` | `DirkIntegrator<Value>`: implicit stages solved with `fuse::num::solve()` (Newton-Krylov), for the `Dirk`, `DirkEmbedded`, `Sdirk` and `SdirkEmbedded` tableaus |
| `fuse/num/rosenbrock_integrator.hpp` | `RosenbrockIntegrator<Value>`: one Jacobian per step and a GMRES solve per stage, for the `Rosenbrock` and `RosenbrockEmbedded` tableaus |
| `fuse/num/embedded_controller.hpp` | `EmbeddedController`: PI or I step-size control from an embedded error estimate |
| `fuse/num/continuous_extension.hpp` | `CubicHermiteExtension<Value>`: dense output between two accepted steps |
| `fuse/num/ssp_coefficient.hpp` | The SSP coefficient of a method from its Shu-Osher form (`ssp_coefficient_from_shu_osher()`) |

`Value` is the element type of the state (`double` in all the examples here). The step size, the
time and the error norm are always the underlying scalar type.

### Methods

Each method is a `ButcherTableau` returned by a factory function in `fuse/num/butcher_tableau.hpp`.
The struct carries the name, the family, the number of stages, the nominal order, the embedded order
when the method has an error estimate (`has_embedded_estimate()`), the stability flags
(`is_A_stable`, `is_L_stable`) and, for SSP methods, `ssp_coefficient`.

| Family | Factories | Integrator |
|---|---|---|
| Explicit RK | `forward_euler_tableau()`, `kutta22_tableau()`, `kutta44_tableau()`, `dopri45_tableau()` (embedded, order 5(4)) | `ExplicitIntegrator` |
| SSP explicit RK | `ssprk33_tableau()` (SSP coefficient 1), `ssprk104_tableau()` (SSP coefficient 6) | `ExplicitIntegrator` |
| SDIRK | `sdirk22_tableau()`, `sdirk23_tableau()`, `sdirk33_tableau()`, `sdirk34_tableau()`, `sdirk44_tableau()`; embedded: `sdirk332_tableau()`, `sdirk543_tableau()`, `sdirk643_tableau()` | `DirkIntegrator` |
| DIRK / ESDIRK | embedded: `trbe_tableau()`, `esdirk324l2sa_tableau()` (explicit first stage, L-stable, order 3(2)) | `DirkIntegrator` |
| Rosenbrock | `ros22_tableau()`; embedded: `ros3p_tableau()`, `rodasp_tableau()`, `grk4t_tableau()`, `grk4sn_tableau()` | `RosenbrockIntegrator` |
| Fully implicit one-stage and two-stage | `backward_euler_tableau()`, `crank_nicolson_tableau()` | not handled by `DirkIntegrator`: a step with them is refused with `E-INTEGRATOR-WRONG-FAMILY` |

A fixed-step `RosenbrockIntegrator` is built from a tableau and a relative tolerance for its stage
solves (`RosenbrockIntegrator<double>(ros22_tableau(), 1e-10)`); the adaptive constructor takes a
controller, `rtol`, `atol` and an optional norm, like `ExplicitIntegrator`. `grk4a_tableau()` is
also available. It carries no usable error estimate
(`has_embedded_estimate()` is `false`), so use it as a fixed-step method only.
`all_karma_tableaus()` and `all_round2_tableaus()` return every tableau in a `std::vector` if you
want to loop over them. The factory functions return plain data; a method that is not in the list is
a `ButcherTableau` you fill in yourself, and the integrator for its family will run it.

## Describing the problem

A `TemporalProblem<Value>` holds `std::function` members. Only `rhs` is required.

| Member | Meaning | Used by |
|---|---|---|
| `rhs(t, y, out)` | `out <- f(t, y)` | every family |
| `mass(v, out)` | `out <- M v`; absent means `M = I` | DIRK, Rosenbrock; an explicit integrator refuses a problem that sets it (`E-INTEGRATOR-MASS-UNSUPPORTED`) |
| `jacobian_action(t, y, v, out)` | `out <- J(t, y) v`; absent means a finite-difference product of `rhs` | Rosenbrock |
| `dfdt(t, y, out)` | `out <- df/dt` | Rosenbrock, for non-autonomous problems |
| `dimension` | state size; `0` skips the check | every family |

Every callback returns a `fuse::Result<void>`; a failure from `rhs` is returned unchanged from
`step()`.

## Taking steps

`Integrator::step(problem, t, h, state)` attempts one step of size `h` from time `t` and updates
`state` in place if the step is accepted. It returns a `StepRecord`:

| Field | Meaning |
|---|---|
| `accepted` | `false` for a step an adaptive method rejected; `state` is then untouched |
| `t_new`, `h_used` | the time reached and the step actually taken (`t` and `0` after a rejection) |
| `h_next` | the controller's proposal for the next attempt (equal to `h` for a fixed-step integrator) |
| `error_norm` | the scaled embedded error, present for adaptive integrators |
| `function_evaluations` | right-hand-side evaluations used by this step |

A rejection is an ordinary outcome, not an error. A `Failure` (a named `E-INTEGRATOR-...` code) is
returned only for a refusal, listed [below](#refusals).

### Example: an adaptive explicit method and a stiff implicit one

The program below integrates `y' = -y` with Dormand-Prince 5(4) under a PI controller, then a stiff
problem `y' = -1000 (y - cos t)` with a step 50 times larger than the fast time scale using the
L-stable ESDIRK method, and finally evaluates the Hermite extension between two points. It is
compiled and run as part of preparing this page.

```cpp
#include <cmath>
#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>
#include <fuse/base/error.hpp>
#include <fuse/num/butcher_tableau.hpp>
#include <fuse/num/continuous_extension.hpp>
#include <fuse/num/dirk_integrator.hpp>
#include <fuse/num/embedded_controller.hpp>
#include <fuse/num/explicit_integrator.hpp>
#include <fuse/num/integrator.hpp>

int main(int argc, char** argv) {
  fuse::initialize();
  Kokkos::ScopeGuard kokkos_guard(argc, argv);
  using namespace fuse::num;

  // y' = -y, y(0) = 1: the exact answer at t = 1 is exp(-1).
  TemporalProblem<double> decay;
  decay.dimension = 1;
  decay.rhs = [](double, std::span<const double> y, std::span<double> out) {
    out[0] = -y[0];
    return fuse::Result<void>::success();
  };

  // Adaptive explicit: Dormand-Prince 5(4) under a PI step-size controller.
  ControllerOptions opt;
  opt.order = 5;
  opt.embedded_order = 4;
  ExplicitIntegrator<double> dopri(dopri45_tableau(), EmbeddedController(opt),
                                   /*rtol=*/1e-8, /*atol=*/1e-10);
  std::vector<double> y{1.0};
  double t = 0.0, h = 0.1;
  int rejected = 0;
  while (t < 1.0) {
    if (t + h > 1.0) h = 1.0 - t;               // land exactly on the end of the window
    const auto r = dopri.step(decay, t, h, y);
    if (!r.ok()) { std::cerr << r.error().code << ": " << r.error().message << "\n"; return 1; }
    if (r.value().accepted) t = r.value().t_new; else ++rejected;
    h = r.value().h_next;
  }
  std::cout << "dopri45  y(1) error " << std::fabs(y[0] - std::exp(-1.0))
            << "  rejected " << rejected << "\n";

  // Stiff problem, fixed step: y' = -1000 (y - cos t), L-stable ESDIRK, h far above 1/1000.
  TemporalProblem<double> stiff;
  stiff.dimension = 1;
  stiff.rhs = [](double tt, std::span<const double> yy, std::span<double> out) {
    out[0] = -1000.0 * (yy[0] - std::cos(tt));
    return fuse::Result<void>::success();
  };
  DirkIntegrator<double> esdirk(esdirk324l2sa_tableau());
  std::vector<double> z{0.0};
  double tz = 0.0;
  const double hz = 0.05;
  for (int i = 0; i < 20; ++i) {
    const auto r = esdirk.step(stiff, tz, hz, z);
    if (!r.ok()) { std::cerr << r.error().code << ": " << r.error().message << "\n"; return 1; }
    tz = r.value().t_new;
  }
  std::cout << "esdirk   y(1) - cos(1) = " << z[0] - std::cos(1.0) << "\n";

  // Dense output between two accepted steps.
  const std::vector<double> y0{1.0}, f0{-1.0};
  const std::vector<double> y1{std::exp(-0.1)}, f1{-std::exp(-0.1)};
  CubicHermiteExtension<double> ce(0.1, y0, f0, y1, f1);
  std::cout << "hermite  y(0.05) error " << std::fabs(ce.evaluate(0.5)[0] - std::exp(-0.05)) << "\n";
  return 0;
}
```

Output (exact digits can vary by platform):

```text
dopri45  y(1) error 2.68613e-10  rejected 0
esdirk   y(1) - cos(1) = 0.000841074
hermite  y(0.05) error 2.47757e-07
```

The ESDIRK error is the physical lag of the stiff solution behind `cos t`, about `sin(t) / 1000`
(`8.4e-4` at `t = 1`). The point is that the step is stable at `h = 0.05`; an explicit method needs a
step of a few thousandths for this problem.

Link `fuse::num` and `fuse::linalg` (the DIRK and Rosenbrock integrators call the Newton-Krylov and
GMRES solvers). Building against an installed FUSE looks like any other consumer
([](quickstart.md#includes-linking-and-kokkos)); Kokkos must be initialised, as in every FUSE
program.

## Adaptive step control

An integrator built with an `EmbeddedController` computes the embedded error estimate of its
tableau, scales it componentwise by `atol + rtol * |y|`, reduces it to one number with a norm
functor, and lets the controller decide.

- `ControllerOptions` picks the controller (`ControllerKind::PI` or `ControllerKind::I`), the method
  order and embedded order it assumes, the safety factor (default 0.9), the bounds on how much the
  step may shrink and grow (`fac_min` 0.2, `fac_max` 4.0), the PI gain `beta` and an optional `h_min`.
  Set `order` and `embedded_order` to those of the tableau you pass.
- The default norm is `serial_rms_norm`, the root mean square of the scaled components. The reduction
  is the caller's: pass your own `ScaledNormFn` to the constructor if the error vector is spread over
  several ranks, so that every rank takes the same accept/reject decision.
- A step is accepted when the scaled error norm is at most 1. A rejected step leaves `state`
  unchanged and `h_next` is smaller than `h`.
- The controller carries a little history between steps in a `ControllerState`; the integrator holds
  it for you (`controller_state()` exposes it so you can reset it between windows).
- A fixed-step integrator (no controller) always accepts and returns `h_next == h`.

Passing a controller with a tableau that has no embedded estimate (`has_embedded_estimate()` is
`false`) is refused with `E-INTEGRATOR-WRONG-FAMILY`.

## Sub-stepping inside a window

An adaptive method proposes `h` as a floating-point number, while FUSE times are exact rationals.
`SubStepClock` bridges the two: it divides the window `[begin, end)` once into an integer number of
equal exact quanta (`2^20` by default), turns each proposed step into a whole number of quanta with
`round_to_quantum()` (never past the end of the window) and keeps `now()` as an exact `Rational`.
The last sub-step therefore ends exactly on the window boundary, and a sample time handed to the
runtime is exact:

```cpp
fuse::num::SubStepClock clock(fuse::Rational{0}, fuse::Rational{1, 50});
while (!clock.at_window_end()) {
  const std::int64_t n = clock.round_to_quantum(h_proposed);   // clipped to the window
  // ... step the integrator by n * clock.quantum().to_double() ...
  clock.advance(n);
}
```

## Refusals

Failures are returned as a `Result` carrying one of these codes; none is thrown.

| Code | Meaning |
|---|---|
| `E-INTEGRATOR-NO-RHS` | `TemporalProblem::rhs` is not set |
| `E-INTEGRATOR-WRONG-FAMILY` | the tableau's family is not one this integrator handles, a Rosenbrock tableau has no stored `gamma`, or a controller was given for a tableau without an embedded estimate |
| `E-INTEGRATOR-DIMENSION-MISMATCH` | `problem.dimension` is non-zero and differs from the state size |
| `E-INTEGRATOR-MASS-UNSUPPORTED` | an explicit method was given a problem with a mass matrix; use DIRK or Rosenbrock |
| `E-INTEGRATOR-NEWTON-FAILED` | a DIRK stage's nonlinear solve did not converge |
| `E-INTEGRATOR-STAGE-SOLVE-FAILED` | a Rosenbrock stage's GMRES solve did not reach its tolerance |
| `E-INTEGRATOR-NONFINITE-ERROR` | the error norm is NaN or infinite |
| `E-INTEGRATOR-STEP-TOO-SMALL` | a rejected step's next proposal is below `h_min` |

## Limits to know about

- The integrators work on a `std::span<Value>` in the calling process. They do not exchange data
  between ranks themselves.
- `DirkIntegrator` always forms its Newton Jacobian products by differencing `rhs`;
  `jacobian_action` is read by the Rosenbrock integrator only.
- Each step recomputes the first stage even when a method could reuse the last stage of the
  previous step. The result is the same; one extra right-hand-side evaluation is spent per step.
- `ButcherTableau::order` is the method's nominal order, recorded as data. The order of a
  Rosenbrock method beyond two depends on Jacobian-dependent conditions that the generic
  Runge-Kutta order checker used in the test suite does not verify.

## See also

- [](linear-algebra.md) for the Newton-Krylov and GMRES solvers the implicit integrators use.
- [](convergence.md) for how the coupling itself iterates, which is separate from the time
  integration inside a component.
- [](component-interface.md) for where an integrator call fits in `advance()`.
