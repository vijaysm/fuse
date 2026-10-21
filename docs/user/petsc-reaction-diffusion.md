# petsc_reaction_diffusion: a real PETSc-object LinearizationToken

`examples/petsc_reaction_diffusion/` is the smallest real port of a `fuse::ImplicitComponent` onto
a PETSc-object-backed `fuse::LinearizationToken` -- a real, assembled `Mat`/`Vec`, with
`apply_jacobian`/`apply_jacobian_transpose` as `MatMult`/`MatMultTranspose`, never `MATMFFD` -- paired
with a native-substrate implementation of the SAME physics in one Newton-coupled region. It retitles
and closes the parent plan's own "first implicit participant" item smaller than originally scoped
(owner decision 2026-10-21): KARMA's FEM import (elements, quadrature, assembly, a PETSc Vec-backed
field store) stays entirely `docs/dev/roadmap.md`'s own S8 scope, untouched here.

## The problem

A single scalar field obeys a nonlinear reaction-diffusion equation,

```
u_t = D u_xx - k u^3 + f(x, t),      x in [0, L_left + L_right], t >= 0
```

on a 1-D domain split at `x = L_left` into two abutting half-domains, Dirichlet-Neumann coupled
exactly like [dn_heat](dn-heat.md)'s own `DirichletSlab`/`NeumannSlab` pair -- generalised from
dn_heat's quasi-steady, closed-form, affine slab to a genuinely time-dependent, discretised
(`m` interior backward-Euler grid points per half), nonlinear one:

```
DirichletHalf  (native):  receives u_interface, solves its own m-point backward-Euler system,
                          provides q_interface = -D (u_interface - u_m) / dx  [the flux it took]
NeumannHalf    (native OR petsc): receives q_interface (via a one-sided ghost value consistent
                          with the SAME flux formula), solves its own m-point system, provides
                          the interface value its own solution implies
```

`f(x,t)` is chosen so that `u(x,t) = u0 sin(pi x / L_total) exp(-t/tau)` is an EXACT solution (the
method of manufactured solutions): both outer ends are homogeneous Dirichlet by construction
(`sin(pi) = 0`), and `L_left != L_right` is chosen so the interface sits away from the sine's own
extremum (a nonzero, non-degenerate exchanged flux). `reaction_diffusion_physics.hpp` carries the
closed form and both substrates' shared residual/Jacobian stencil; neither substrate has a separate
mesh type or DOF map (`fuse::native::NativeMeshBackend`/`NativeGrid` -- the Brusselator's own,
heavier, multi-rank-capable machinery -- was considered and rejected for this single-variable,
single-rank demonstration: see `design_notes.md` Section 3).

## The two substrates

`NeumannHalfNative` (plain arrays, a hand-rolled Thomas-algorithm Newton solve) and
`NeumannHalfPetsc` (a real PETSc `Vec`/`Mat`/`KSP` Newton solve: `MatCreateAIJ` with this half's own
tridiagonal nonzero count, `KSPPREONLY`+`PCLU` direct factorisation -- robustness over efficiency,
this project's standing rule) solve the IDENTICAL discrete equations. The OUTER sensitivity
(`dG/d(interface datum)`, what `linearization()` returns to the Newton coupler) is obtained by
IMPLICIT DIFFERENTIATION of the converged inner solve -- one more linear solve against the SAME,
already-factored inner Jacobian -- wrapped in a 1x1 PETSc `Mat`
(`fuse::petsc::make_assembled_token()`, reused verbatim, [docs/dev/newton-krylov.md sec. 10.1]
(../dev/newton-krylov.md)) for the PETSc half. See
[docs/dev/newton-krylov.md sec. 14](../dev/newton-krylov.md) for the full derivation, the historical
Jacobian bug this track's own derivative-gate test now guards against permanently
(`UncorrectedGhostJacobianMeasuresOrderOneNotTwo`), and the `NewtonConfig::eta_min` finding this
problem needed (the SAME value, `1e-4`, the Brusselator's own window already established).

## Building and running it

Default build target `fuse_example_petsc_reaction_diffusion`; `fuse::petsc` and
`reaction_diffusion_component_petsc.hpp` are compiled in only under `FUSE_ENABLE_PETSC`
(`#if FUSE_HAVE_PETSC`, `fuse/base/config.hpp`), the same conditional `fuse_example_dn_heat` uses.

```
fuse_example_petsc_reaction_diffusion            # native twin only -- no PETSc, no MPI
fuse_example_petsc_reaction_diffusion --petsc    # additionally builds AND RUNS the PETSc half
```

| row | what it runs | ctest row |
|---|---|---|
| native twin | `DirichletHalf` + `NeumannHalfNative`, 20 periods, checked against the manufactured solution | `example.petsc_reaction_diffusion` |
| native+petsc | additionally `DirichletHalf` + `NeumannHalfPetsc`, checked against the native twin's own converged answer | `example.petsc_reaction_diffusion.petsc` (gated `FUSE_ENABLE_PETSC`) |

A dedicated structural test, `petsc.reaction_diffusion` (`test/petsc/reaction_diffusion_test.cpp`),
and a native-only component-contract test, `runtime.petsc_reaction_diffusion_component`
(purity, token shape/signature, save/restore round trip), are registered alongside.

## What is measured, and what is host-pending

**Measured, this sandbox** (`m = 4` interior points/half, `dt = 0.01`, 20 periods, `t_final = 0.2`):
the native-twin region converges in 3 Newton iterations per period; max|error| against the
manufactured solution is 7.31e-2 (Left) / 5.34e-2 (Right) / 1.18e-1 (interface) against a derived
bound of 2.37e-1 (2x the largest measured figure -- see `main.cpp`'s own `kAgreementBound` comment
for the derivation). `AdjointIdentityHoldsForTheInnerJacobian` and
`RightInnerJacobianMeasuresOrderTwo` (`test/linalg/reaction_diffusion_derivative_gate_test.cpp`)
pass; the two deliberately-wrong-Jacobian control cases measure order 1, not 2, confirming the gate
can fail.

**Host-pending** (KNOWN_ISSUES.md D32, the already-documented, already-measured constraint that this
sandbox cannot complete a bare `MPI_Init()`, reproduced directly before this track wrote any
component code): `NeumannHalfPetsc`'s own constructor calls `fuse::petsc::ensure_initialized()`, so
`--petsc`, `example.petsc_reaction_diffusion.petsc`, and `petsc.reaction_diffusion` are all BUILT and
ctest-REGISTERED here, never RUN here -- exactly the same host-pending status every other `petsc.*`
row and `example.dn_heat.newton_petsc` already carry. A host run is what measures the native-vs-PETSc
agreement `main.cpp` checks against its own derived bound (`kSubstrateAgreementBound = 100 * 1e-8`:
the Newton `rtol` default times a conditioning headroom of 100, since both substrates solve the
IDENTICAL system by a direct linear solve).

## State

**Verified** at tier 0, native-only: `example.petsc_reaction_diffusion`,
`linalg.reaction_diffusion_derivative_gate`, `runtime.petsc_reaction_diffusion_component`. **Built,
registered, host-pending**: `example.petsc_reaction_diffusion.petsc`, `petsc.reaction_diffusion`.
