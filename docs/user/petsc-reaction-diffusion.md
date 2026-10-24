# petsc_reaction_diffusion: a PETSc-object LinearizationToken

`examples/petsc_reaction_diffusion/` is the smallest example of a `fuse::ImplicitComponent` whose
linearization is backed by real PETSc objects, paired with a native implementation of the same physics
in one Newton-coupled region. Read it if you are writing an implicit component and want the Newton
coupler to use a PETSc `Mat` for its Jacobian.

## What it demonstrates

A `fuse::LinearizationToken` that wraps a real, assembled PETSc `Mat`/`Vec`, with
`apply_jacobian`/`apply_jacobian_transpose` implemented as `MatMult`/`MatMultTranspose` (never
`MATMFFD`). The same discrete equations are solved on two substrates, native and PETSc, and the
answers compared.

**The native half builds and runs with no PETSc and no MPI. The `--petsc` half requires a build with
`FUSE_ENABLE_PETSC=ON`** (see [](build.md)); in any other build `--petsc` prints an error and exits 1.

## Build and run

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
# or, without presets:
#   cmake -S . -B build -DFUSE_ALLOW_FETCH=ON && cmake --build build

<build-dir>/examples/fuse_example_petsc_reaction_diffusion            # native twin only
<build-dir>/examples/fuse_example_petsc_reaction_diffusion --petsc    # additionally builds and runs the PETSc half
```

For `--petsc`, configure with PETSc enabled (the `tier3` preset, or `-DFUSE_ENABLE_PETSC=ON` on top of
the lower tiers) and rebuild. `--petsc` is the only flag.

| Run | What it does | Passes when |
|---|---|---|
| native twin | `DirichletHalf` + `NeumannHalfNative`, 20 periods | error against the manufactured solution is below the agreement bound |
| `--petsc` | also runs `DirichletHalf` + `NeumannHalfPetsc` | the PETSc result matches the native twin's converged answer to within `kSubstrateAgreementBound` |

## The problem

A single scalar field obeys a nonlinear reaction-diffusion equation,

```
u_t = D u_xx - k u^3 + f(x, t),      x in [0, L_left + L_right], t >= 0
```

on a 1D domain split at `x = L_left` into two abutting half-domains, Dirichlet-Neumann coupled as in
[dn_heat](dn-heat.md), but with a genuinely time-dependent, discretised (`m` interior backward-Euler
grid points per half), nonlinear solve in place of dn_heat's closed-form slab:

```
DirichletHalf  (native):  receives u_interface, solves its own m-point backward-Euler system,
                          provides q_interface = -D (u_interface - u_m) / dx  [the flux it took]
NeumannHalf    (native or PETSc): receives q_interface (via a one-sided ghost value consistent
                          with the same flux formula), solves its own m-point system, provides
                          the interface value its own solution implies
```

`f(x,t)` is chosen so that `u(x,t) = u0 sin(pi x / L_total) exp(-t/tau)` is an exact solution (method
of manufactured solutions). Both outer ends are homogeneous Dirichlet (`sin(pi) = 0`), and
`L_left != L_right` puts the interface away from the sine's extremum so the exchanged flux is nonzero.
`reaction_diffusion_physics.hpp` carries the closed form and the residual/Jacobian stencil both
substrates share. Neither substrate has a separate mesh type or DOF map.

## The two substrates

`NeumannHalfNative` uses plain arrays and a hand-rolled Thomas-algorithm Newton solve.
`NeumannHalfPetsc` uses a real PETSc `Vec`/`Mat`/`KSP` Newton solve: `MatCreateAIJ` with the
tridiagonal nonzero count, and `KSPPREONLY` with `PCLU` (a direct factorisation). Its constructor
calls `fuse::petsc::ensure_initialized()`.

The outer sensitivity (`dG/d(interface datum)`, which `linearization()` returns to the Newton coupler)
is obtained by implicit differentiation of the converged inner solve: one more linear solve against
the same, already-factored inner Jacobian. For the PETSc half this is wrapped in a 1x1 PETSc `Mat`
with `fuse::petsc::make_assembled_token()`. See [the Newton-Krylov design notes](../dev/newton-krylov.md) for the
derivation.

## How the graph is authored

`main.cpp` builds the Dirichlet-Neumann pair as one Newton-coupled iteration region, as
[dn_heat](dn-heat.md) does for its Newton rows, with `NeumannHalfNative` or `NeumannHalfPetsc` as the
Neumann participant:

```{literalinclude} ../../examples/petsc_reaction_diffusion/main.cpp
:language: cpp
:start-at: "fuse::ConvergenceCriterion crit;"
:end-at: "return g;"
```

The region-level Newton-Krylov driver is always the native one (`NewtonConfig::substrate` is left at
its default). The PETSc objects live entirely inside `NeumannHalfPetsc::linearization()`; the driver
only sees a `LinearizationToken`. That is the point of the example: the token hides which substrate
produced the Jacobian. The `eta_min = 1e-4` floor keeps GMRES from stalling at the matrix-free
Jacobian action's noise floor, and is the same value the [Brusselator](brusselator.md) uses. The
component code in `reaction_diffusion_component.hpp` (native) and
`reaction_diffusion_component_petsc.hpp` (PETSc, compiled only when `FUSE_HAVE_PETSC` is set) shows
how `linearization()` is implemented.

## Expected output

Native twin (`m = 4` interior points per half, `dt = 0.01`, 20 periods, `t_final = 0.2`), exact digits
can vary by platform:

- The Newton region converges in 3 iterations per period.
- Maximum error against the manufactured solution: 7.31e-2 (left half), 5.34e-2 (right half), 1.18e-1
  (interface), against the bound `kAgreementBound = 2.37e-1`.

With `--petsc`, the PETSc half converges in the same way and agrees with the native twin to within
`kSubstrateAgreementBound = 100 * 1e-8`: the Newton `rtol` default times a conditioning headroom of
100, since both substrates solve the identical system with a direct linear solve.

## What the example checks

The exit status is non-zero unless:

- the native twin's maximum errors (left, right, interface) are all below `kAgreementBound`, and
- with `--petsc`, the PETSc half's answer differs from the native twin's by less than
  `kSubstrateAgreementBound`.

The automated test suite also includes a structural PETSc test (`petsc.reaction_diffusion`), a
native-only component-contract test (`runtime.petsc_reaction_diffusion_component`: purity, token
shape and signature, save/restore round trip) and a derivative gate
(`linalg.reaction_diffusion_derivative_gate`). The gate confirms the inner Jacobian is second-order
accurate and that two deliberately wrong Jacobians measure order 1, so the gate can fail.

## What to try next

- Run without `--petsc` first; it needs nothing beyond the base build.
- In a PETSc build, run with `--petsc` and confirm the two halves agree to the printed bound.
- Use the PETSc token pattern (`make_assembled_token()`) as a template for your own implicit
  component.

## Limitations

- The example is single-rank and single-variable. It has no finite-element assembly or distributed
  field store.
- `--petsc` is unavailable in a build without `FUSE_ENABLE_PETSC`.

## See also

- [dn_heat](dn-heat.md): the Newton coupler, and the PETSc substrate, on a closed-form problem.
- [](convergence.md): Newton's own convergence tests.
- [](build.md): building with PETSc.
