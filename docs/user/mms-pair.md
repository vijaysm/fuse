# mms_pair: explicit coupling, verified against a manufactured solution

`examples/mms_pair/` is the smallest complete coupling in the tree and a good first program to read after [](quickstart.md). It
builds a coupling specification, lowers it, schedules it, runs it, and checks the result against a
closed-form solution, through the same `CouplingGraph` -> `lower()` -> `schedule()` -> `fuse::Runner`
sequence any user program uses.

## What it demonstrates

Two in-process `fuse::Component`s, one field transfer, explicit (lagged) coupling. The exchange is
lagged by one coupling cycle, and the example measures how much error that lag costs: the coupled
answer is compared with the analytic solution and with a control run in which the exchange is cut.
It needs only the base build (no MPI, no MOAB, no Eigen).

## Build and run

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
# or, without presets:
#   cmake -S . -B build -DFUSE_ALLOW_FETCH=ON && cmake --build build

<build-dir>/examples/fuse_example_mms_pair          # run and check
<build-dir>/examples/fuse_example_mms_pair --dump   # print the compiled IR and plan, run nothing
```

See [](build.md) for the presets. The only flag is `--dump`, which prints the compiled dataflow graph
(`=== IR ===`) and the flat execution plan (`=== PLAN ===`) and exits 0: a chain of 10 nodes
(`BuildRemapHalo`, `Source`, `Sample`, `Accumulate`, `Transport`, `Epoch`, `Remap`, `Constrain`,
`Diagnose`, `Sink`) for the one channel, and a 16-instruction, 9-buffer plan. The four `left` substeps
appear as four `Source`/`Sample` pairs at `t = 0, 1/4000, 1/2000, 3/4000`, followed by the
`Accumulate` (`Mean`) and the transfer chain at `t = 1/1000`.

With no flag the binary runs 100 coupling cycles to `t = 0.1`, prints the errors and the conservation
count, and returns 0 on success and non-zero if a gate fails (see "What the example checks").

## The problem

`left` solves the 1D heat equation by explicit Euler on `[0,1]` with homogeneous Dirichlet ends,
exact for a sine initial condition:

```
T_t = alpha T_xx,   T(0,t) = T(1,t) = 0,   T(x,0) = sin(pi x)
T(x,t) = exp(-pi^2 alpha t) sin(pi x)       (alpha = 0.01)
```

`right` integrates a decay relaxing toward `left`'s boundary value in closed form:

```
dR/dt = kappa (T - R)       (kappa = 2.0)
```

`left` provides `T`; `right` requires `T_bc`. Both meshes are coincident 64-cell lines
(`generated:line:64`), so the one field transfer's remap is an identity copy, the only configuration
in which an identity is the correct remap operator. The coupling cycle is `H = 1/1000` and `left`
subcycles 4 times inside it (`dt_left = H/4`), so the field transfer's `Mean` reduction over the
cycle actually reduces something rather than copying a single sample. The exchange is **explicit and
lagged by one coupling cycle**: `right` receives `left`'s mean of the *previous* cycle. That lag is
the dominant term in `right`'s error and is what the example measures; it is not a discretisation
defect in either component.

## How the graph is authored

Graph authoring is the fluent `CouplingGraph` API directly in `build_graph()` (see
[](fluent-api.md)), with no coupler and a single channel:

```{literalinclude} ../../examples/mms_pair/main.cpp
:language: cpp
:start-at: "fuse::CouplingGraph build_graph(std::size_t cells) {"
:end-at: "return g;"
```

`fuse::Conservation::Extensive` on `T`/`T_bc` is what makes `lower()` emit the `Constrain` operator
in the dataflow graph. Without it the conservation-check count would be zero and the conservation
gate vacuous. Because the target is Extensive, `validate()` requires the channel to declare that
its transfer conserves the field (`.conservative()`, `E-CONN-CONSERVATION` otherwise); between
these two coincident meshes the remap is the identity, which does.
`g.channel(ch).accumulate(fuse::AccumOp::Mean)` is the field transfer's time treatment
(glossary: *time integration*): the mean of `left`'s waveform over `right`'s step. The example
calls `fuse::validate()` before `lower()`, as [](quickstart.md) does.

`left` and `right` are each an ordinary `fuse::Component`. `HeatComponent` (`heat_component.hpp`)
registers its state array `t_` as `Intent::Provides`, writes the next time level into a scratch
buffer and copies it back element-wise. It never swaps the two vectors, because the registered
address must not move. `RelaxComponent` requires `T_bc` and integrates the decay in closed form each
step.

## Expected output

Exact digits can vary by platform.

```
periods                   : 100
instructions              : 1600
conservation checks       : 100
violations                : 0
worst conservation error  : 0
t                         : 0.1
L2 error left  (own disc.): 1.30225e-06
L2 error right (coupled)  : 0.00114793
L2 error right (no exch.) : 0.127525
coupled / uncoupled       : 0.00900155
```

| Quantity | Expected |
|---|---:|
| L2 error, `left`'s own discretisation | 1.302e-06 |
| L2 error, `right`, coupled | 1.148e-03 |
| L2 error, `right`, exchange removed | 1.275e-01 |
| coupled / uncoupled | 0.00900 |
| conservation violations | 0 of 100 checks |

`left`'s error is its own discretisation error (`O(dx^2)` in space at `dx = 1/64`, `O(dt)` in time at
`dt = 1/4000`), not a property of the coupling. `right`'s coupled error is the one-cycle lag plus the
`Mean` reduction's truncation, both coupler effects. The "no exchange" control runs the identical
graph with `right`'s import bound to a decoy buffer, so the coupled error is about 111 times smaller
than no coupling at all would give: evidence that the headline number is the coupling doing
something. The `instructions` figure is `periods * instructions_per_period` (100 x 16), a sanity count.

## What the example checks

The exit status is non-zero unless all of these hold:

- `kHeatGate = 1.0e-4` on `left`'s L2 error.
- `kRelaxGate = 3.0e-3` on `right`'s coupled L2 error.
- `kCouplingEvidenceRatio = 0.05` on coupled/uncoupled.
- At least one conservation check ran, and none reported a violation.

The output is bit-identical at any OpenMP thread count; the components declare
`DeterminismLevel::Bitwise`.

## What to try next

- Run `--dump` and compare the 10-node IR with the channel declared in `build_graph()`. Removing the
  `Conservation::Extensive` declaration drops the `Constrain` node and the conservation count falls
  to zero, which the example then rejects.
- Reduce the coupling cycle `H` in `build_graph()`. The one-cycle lag is the dominant term in
  `right`'s error, so the coupled error should fall while `left`'s own error stays at its
  discretisation level. To remove the lag altogether, iterate within the cycle; [dn_heat](dn-heat.md)
  does that with an implicit coupling group.

## See also

- [](quickstart.md): the same sequence in a minimal program.
- [dn_heat](dn-heat.md): the implicit counterpart, iterated to a fixed point.
- [](convergence.md): how an iterated coupling decides it has converged.
- [](interface-transfer.md): remap operators, of which this example uses the identity.
