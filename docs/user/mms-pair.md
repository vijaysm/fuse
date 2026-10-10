# mms_pair: explicit coupling, verified against a manufactured solution

`examples/mms_pair/` is the smallest complete coupling in the tree: two in-process
`fuse::Component`s, one field transfer, explicit (lagged) coupling, checked against a closed-form
solution rather than merely run. It is the tree's Demo A at the Phase 4 milestone of the original
spec: build a coupling specification, lower it, schedule it, run it, through the same
`CouplingGraph` -> `lower()` -> `schedule()` -> `fuse::Runner` sequence any user program uses.

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
(`generated:line:64`), so the one field transfer's remap is an identity copy — the only
configuration in which an identity is the correct remap operator. The coupling cycle is `H = 1/1000`
and `left` subcycles 4 times inside it (`dt_left = H/4`), so the field transfer's `Mean` reduction
over the cycle actually reduces something rather than copying a single sample. The exchange is
**explicit and lagged by one coupling cycle**: `right` receives `left`'s mean of the *previous*
cycle, which is this example's whole point — the lag is the dominant term in `right`'s error, and
is what the example measures, not a discretisation defect in either component
(`examples/mms_pair/main.cpp`'s own header comment).

## How it is authored

Graph authoring is the fluent `CouplingGraph` API directly in `main.cpp`'s `build_graph()`, with no
coupler and a single channel:

```{literalinclude} ../../examples/mms_pair/main.cpp
:language: cpp
:start-at: "fuse::CouplingGraph build_graph(std::size_t cells) {"
:end-at: "return g;"
```

`fuse::Conservation::Extensive` on `T`/`T_bc` is what makes `lower()` emit the `Constrain` operator
in the dataflow graph — without it the printed conservation-check count would be zero and the
example's own conservation gate would be vacuous. `g.channel(ch).accumulate(fuse::AccumOp::Mean)` is
the field transfer's time treatment (glossary: *time integration*): the mean of `left`'s waveform
over `right`'s step.

`left` and `right` are each an ordinary `fuse::Component`: `HeatComponent` (`heat_component.hpp`)
registers its state array `t_` as `Intent::Provides`, writes the next time level into a scratch
buffer and copies it back element-wise (never swaps the two vectors — the registered address must
not move, `fuse::runtime::component.hpp`'s own contract). `RelaxComponent` requires `T_bc` and
integrates the decay in closed form each step.

## Building and running it

Tier 0 (no MPI, no MOAB) is enough — this is the base build configuration:

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
<build-dir>/examples/fuse_example_mms_pair
<build-dir>/examples/fuse_example_mms_pair --dump
```

`--dump` prints the compiled dataflow graph (`=== IR ===`) and the flat execution plan
(`=== PLAN ===`) without running anything, then exits 0 — one `Source`/`Sample`/`Accumulate`/
`Transport`/`Epoch`/`Remap`/`Constrain`/`Diagnose`/`Sink` chain of 9 nodes for the one channel, and a
15-instruction, 9-buffer plan. The two `ctest` rows `example.mms_pair.dump`/`example.dn_heat.dump`
assert on that printed text (`PASS_REGULAR_EXPRESSION "=== IR ===.*=== PLAN ==="`), because an empty
dump still exits 0 and would otherwise pass silently.

With no flag the binary runs 100 coupling cycles to `t = 0.1` and prints the measured errors and the
conservation count, then returns non-zero if either component's error gate fails, if the coupled
error is not decisively better than an exchange-free control, or if no conservation check ran.

## What it prints

Measured on this build (macOS arm64, AppleClang 21.0.0, Kokkos 5.1.0 OpenMP+Serial, tier0-release,
`fuse_example_mms_pair` with no arguments):

```
periods                   : 100
instructions              : 1500
conservation checks       : 100
violations                : 0
worst conservation error  : 0
t                         : 0.1
L2 error left  (own disc.): 1.30225e-06
L2 error right (coupled)  : 0.00114793
L2 error right (no exch.) : 0.127525
coupled / uncoupled       : 0.00900155
exit status               : 0
```

| Quantity | Measured |
|---|---:|
| L2 error, `left`'s own discretisation | 1.302e-06 |
| L2 error, `right`, coupled | 1.148e-03 |
| L2 error, `right`, exchange removed | 1.275e-01 |
| coupled / uncoupled | 0.00900 |
| conservation violations | 0 of 100 checks |

`left`'s own error is its discretisation error alone (`O(dx^2)` in space at `dx = 1/64`, `O(dt)` in
time at `dt = 1/4000`) — not a property of the coupling. `right`'s coupled error is the one-cycle lag
plus the `Mean` reduction's own truncation, both coupler effects; the "no exchange" control runs the
identical graph with `right`'s import bound to a decoy buffer, so the coupled error is ~111x smaller
than what no coupling at all would give — the control proving the headline number is the coupling
doing something, not an artifact of the gates. The `instructions` count is this run's own executed
instruction total (`periods * instructions_per_period`), printed as a sanity figure.

## Measured against

The analytic solutions above, by `ctest` row `example.mms_pair` (`--check`-equivalent — this binary
has no separate `--check` flag; its ordinary run already gates on the numbers above):

- `kHeatGate = 1.0e-4` on `left`'s L2 error (its own ~1e-6 discretisation error, with margin).
- `kRelaxGate = 3.0e-3` on `right`'s coupled L2 error (measured ~1.15e-3, the coupler's own lag
  term plus margin).
- `kCouplingEvidenceRatio = 0.05` on coupled/uncoupled (measured 0.009, well inside the gate).
- At least one conservation check must have run, and none may report a violation.

`example.mms_pair.omp_identical` (`tools/compare-runs.cmake`) runs the same binary at one OpenMP
thread and at every host core and diffs the output byte for byte — load-bearing since the Runner's
coupling hot loop runs on Kokkos' host execution space, and pinning the
`DeterminismLevel::Bitwise` declaration on both components. Verified on this build: identical at 1
and 12 threads.

## State

**Verified.** `example.mms_pair`, `example.mms_pair.dump`, and `example.mms_pair.omp_identical` all
pass at tier 0 on this build (see "Measured against" above); no multi-rank or MOAB-dependent path
exists for this example — it is deliberately the smallest complete coupling, with coincident meshes
and an in-process, single-executable topology.
