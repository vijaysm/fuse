# Physics Systems

## Mass Operator

The generic mass API assembles consistent mass matrices over variables and components in a `KARMASystem`, multiplies by them, and solves with either the consistent or row-sum-lumped form. Distributed assembly processes each element once and preserves cross-partition cG coupling.

## Diffusion-Reaction

The active diffusion-reaction system supports one scalar cG variable with spatially and materially varying scalar diffusion, nonlinear reaction and source callbacks, optional analytic derivatives, residual/Jacobian/mass/preconditioner paths, PETSc SNES solves, material-specific models, and Dirichlet, outward-flux Neumann, and flux-form Robin conditions.

The typed Robin convention is:

```text
a * (k grad(u) . n) + b * u = c
```

with `a != 0`. Interior dG diffusion fluxes are not implemented, so dG and SE diffusion variables are rejected.

## Advection-Reaction

The active advection-reaction system supports one scalar dG variable with material-dependent state-independent velocity, nonlinear reaction and source callbacks, inflow data, conservative volume terms, upwind boundary fluxes, two-sided Rusanov interior fluxes, and PETSc SNES lifecycle paths.

Verified production paths cover P0 and P1, including partitioned assembly and mixed tetrahedron/prism face traversal. cG/SE advection and state-dependent velocities are not supported by this API.

## Mesh-Free ODEs

`KARMAAlgebraicSystem` and `ODESystem` solve distributed systems of the form:

```text
M(t) udot - f(t, u) = 0
```

through PETSc TS. Models supply a right-hand side and may supply an analytic Jacobian and dense time-dependent mass matrix. If the Jacobian is omitted, KARMA uses a component-scaled central difference.

The active tested time-integration interface is PETSc TS. Historical Runge-Kutta tableau headers do not make the excluded legacy temporal-integrator stack a supported API.
