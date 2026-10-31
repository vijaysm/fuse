# Concepts

The ideas behind FUSE, in the order you meet them when you write a coupling. Read this once before
the [](cookbook.md); come back to [](glossary.md) when a word is unfamiliar.

## The pipeline: describe, compile, run

Most coupling libraries are a set of calls that move data. FUSE is a small compiler.

```text
CouplingGraph  --validate-->  Diagnostics          what you declared, checked as a whole
      |
    lower()                   OperatorDAG          the dataflow, as a small set of operator kinds
      |
   schedule()                 ExecutionPlan        a flat instruction list for one coupling cycle
      |
   Runner                     your components run  allocation-free, on Kokkos
```

You declare *what* is coupled: components, the fields they provide and require, and how each transfer
should treat time and space. FUSE derives *how*: unit conversions, time averaging, remapping,
transport between ranks, conservation checks, and the agreement points that keep ranks from
hanging. Every derived step is visible with `fuse::dump_text(dag)` and `plan.to_string()`, so the
program that runs is never hidden from you.

The payoff is that mistakes surface early and together. An unschedulable cycle, a unit mismatch, a
component on no ranks, or a channel into a field nobody provides is reported by `validate()` or
`schedule()` before the first time step, not as a wrong answer or a hang at step 10,000.

## What you declare

Component
: An independently written solver. It advances its own state in time on its own time step, and runs
  on an explicit set of MPI ranks. A *coupler* is a component that runs no physics, only remaps.

Mesh, region
: A mesh is where a component's discretisation lives (a file, or `generated:line:64`). A region is
  the set of mesh entities of one dimension (cells, faces, vertices) that a field is defined on.

Field
: A named array on a region, with units, a conservation kind (`Extensive`, `Intensive` or `None`)
  and an intent: `Provides` (others may read it), `Requires` (FUSE fills it before your step) or
  `Bidirectional`.

Channel
: One directed transfer from a provided field to a required one, plus its treatment:
  `.accumulate(AccumOp::Mean)`, `.conservative()`, `.remap(...)`, `.lag(1)`, `.relax(omega)`,
  `.extrapolate(order)`.

Iteration region
: A set of channels solved to convergence each step, for strongly coupled problems
  ([Iteration](#iteration)).

```cpp
const auto ch = g.add_channel("T_link", t_out, t_in);
g.channel(ch).conservative().accumulate(fuse::AccumOp::Mean);
```

Authoring calls never return a status and never throw mid-chain. A request FUSE cannot honour is
recorded and reported when you call `validate()`, which lists every defect at once. See
[](fluent-api.md#diagnostics-accumulate).

## Time

**Time is exact.** `fuse::Rational{1, 4000}` is a fraction, not a `double`. Component steps, windows
and every instruction in the plan use exact arithmetic, so a schedule is still in step after ten
million steps and the deadlock proof runs on exact values. Convert with `to_double()` only at the edge
of your own physics.

**Components may run at different rates.** The *coupling period* is the shortest time after which the
whole pattern of steps and transfers repeats (the least common multiple of the steps). Within it, a
faster component takes several steps per slower step. FUSE records the faster one's output at each
sub-step (`Sample`), then reduces it over the slower component's step (`Accumulate`) with a sum, mean,
minimum, maximum or integral. Use the mean for a state and the integral for a flux, so that what one
side loses the other gains over the same interval. The integral is the sum of the samples times
the faster component's step, so it carries units of the source times time; the plain sum does not
scale by the step.

**Cycles must be broken explicitly.** If two channels form a loop, one must be *lagged*: the target
reads the source's value from the previous coupling step. That is explicit (staggered) coupling.
If you want the loop resolved instead, put it in an iteration region.

**Components with incommensurate steps** (no usable common period) are coupled with *coupling
windows*: a time interval over which the group is iterated as a whole, each member advancing across
the window against its partners' trajectories from the previous iteration ([](coupling-windows.md)).

## Space

A mesh backend provides entity geometry and topology behind one interface: a serial structured mesh,
a native distributed unstructured mesh, or MOAB. When a channel connects regions on different meshes,
FUSE builds a sparse *remap* matrix:

| Kind | When to use it | Page |
|---|---|---|
| Conservative (supermesh) | Fluxes and anything whose integral must be preserved | [](geometry.md) |
| Consistent interpolation and its transpose | Fluid-structure interfaces: values one way, loads back conserving their sum | [](interface-transfer.md) |
| Moving least squares | Scattered points with no mesh | [](point-cloud-remap.md) |

Building weights is the expensive part, so they can be written to disk and reused
([](transfer-maps.md)).

## Parallel layout

Each component declares the ranks it runs on as an explicit set. Two components may share all, some
or none of their ranks, and FUSE chooses the cheapest correct movement for each case: no move for
identical sets, a local copy for the shared ranks plus a message for the rest, or a plain move for
disjoint sets. Remapping can run on a separate *coupler* processor set. See [](placement.md).

After every data movement FUSE inserts an `Epoch`: a point where ranks agree whether anything failed.
A rank that fails reconciles the failure with its peers before throwing, so the error is reported
everywhere rather than hanging the ranks that did not see it. This is why the library can promise
that a distributed failure is an error, not a deadlock.

A coupling can also be split across separately built executables, launched together with one
`mpiexec` or started independently and joined over MPI ([](multiple-executables.md)). The graph
records how each component is launched, and the executables load the same description.

## Iteration

Explicit coupling exchanges once per step and accepts the lag. Strongly coupled problems need the
exchange solved to a fixed point. An *iteration region* wraps the channels involved:

```cpp
g.iterate({flux, temp})
    .name("dirichlet_neumann")
    .until(std::move(crit))          // when to stop
    .max_iterations(200)
    .accelerate(accel);              // how to get there faster
```

- **Convergence** is declared with `fuse::ConvergenceCriterion`, one measure per exchanged field
  (`Converged::Relative(tol, "u_wall")`). Every channel in the region must be measured. Partially
  measured regions are refused, because an accelerator can zero the measured residual while leaving
  the unmeasured one large. [](convergence.md) covers the choice of norm.
- **Acceleration** (`fuse::AccelConfig`) is one of `None` (Picard), `Constant`, `Aitken`, `Secant`,
  `Steffensen`, `IQN_ILS`, `IQN_IMVJ`, or `Newton` for the Jacobian-free Newton-Krylov solver.
- **Rollback.** Each sweep re-advances every member from the same starting state, so components in
  an iteration region must implement `save_state()` / `restore_state()` and declare that capability.
  FUSE tells you which components need it.

## What a component is

A component is a class implementing `fuse::Component`:

| Method | You promise |
|---|---|
| `name()` | Equals the `ComponentDecl::name` in the graph. This string is the only link |
| `register_fields(registry, graph)` | Bind your arrays to the fields you declared, once. The addresses must never move |
| `advance(ctx)` | Integrate from `ctx.t_begin` to `ctx.t_end`, reading `Requires` fields and writing `Provides` fields in place |
| `save_state()` / `restore_state()` | Only for iteration regions: restore element-wise, never reassign the container |
| `capabilities()` | Declare rollback support, determinism level and hidden state |

The one rule that surprises people: register storage once and write in place. Swapping two vectors
silently makes the coupler read the old buffer. [](quickstart.md) shows a full component.

## Errors

Nothing fails silently. FUSE reports problems in three ways, and they are meant to be different:

- `validate()` returns `Diagnostics`, every defect at once, before anything runs.
- An operation that cannot be done *refuses* with a stable code (`E-PLACEMENT-EMPTY`,
  `E-ITER-MEASURE`, ...). Look it up in [](troubleshooting.md). There is no silent fallback to a
  weaker behaviour.
- Internal invariants are checked in every build type, including Release.

A program that has called `fuse::initialize()` (every shipped example does; your own driver should)
can be told with `FUSE_ON_ERROR=debugger` to open a debugger on the failing rank at the moment it
fails ([](environment.md), [](troubleshooting.md#debugging-a-failure)).

## Ways in

| You have | Use |
|---|---|
| A C++ code you control | The fluent API and `fuse::Component` ([](quickstart.md)) |
| A coupling you want to keep as a file | YAML ([](yaml.md)): `fuse::to_yaml(g)`, `fuse::from_yaml(text)`, `fuse::load_yaml_file(path)` |
| An existing C or Fortran code with its own time loop | The participant ABI ([](c-abi.md), [](fortran.md)) |
| A plugin FUSE should call | The provider ABI ([](providers.md)) |
| Python physics | `pfuse` ([](python.md)) |
| Two unmodified HDF5 codes | LowFive transport ([](lowfive.md)) |

All of them describe the same coupling. A graph written in C++ can be serialised, loaded from C, and
run in another executable, because every front end shares one versioned binary description.

## See also

- [](getting-started.md), [](cookbook.md), [](api-overview.md).
- [](graph-programming.md) for every declaration, [](ir.md) for what FUSE compiles them into, and
  [](runtime.md) for running the result.
- [](drivers.md) and [](environment.md) for command-line and environment control.
- [](mixing-languages.md) for which language entry points work together.
- [](power-users.md) for performance, scale and reproducibility.
