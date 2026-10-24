# Learned models and surrogate components

How to put a trained model (PyTorch, scikit-learn, or any callable over numpy arrays) into a FUSE
coupling as a component, what it has to promise to be trustworthy there, and how to check that it
is. The route is `pfuse` ([](python.md)): a model's inference call is exactly what a component's
`advance()` does, so a surrogate is an ordinary `pfuse.Component` whose `advance()` runs the
model.

## The idea

A component in FUSE is a black box with declared inputs (`Requires` fields), declared outputs
(`Provides` fields) and a time step. The runtime does not know or care whether the box solves a
PDE or evaluates a network. What it does care about is the component protocol: bind storage once,
write outputs in place, and support checkpoint/restore if the component is iterated. A learned
model that keeps those promises can replace a physics component in the same graph, with the same
channels, remaps, conservation checks and convergence tests around it, which is also what makes
it possible to compare the two fairly.

## A surrogate component

The class below runs any `model(features) -> prediction` on its bound input array every step.
`model` can be a PyTorch module, a scikit-learn regressor's `predict`, or a plain function; the
component does not know which.

```python
class SurrogateComponent(pfuse.Component):
    """Runs `model(features) -> prediction` on the bound input array each step.

    `model` is any callable taking a float64 (N, F) feature matrix and returning an (N,)
    prediction. The component owns no state beyond its bound arrays, so save/restore of the
    output is a complete checkpoint and the component may join an iteration region.
    """

    def __init__(self, name, model, features):
        super().__init__(name)
        self.model = model
        self.features = features          # (inputs, t) -> (N, F) float64
        self.out = np.zeros(N)
        self.inp = np.zeros(N)
        self.saved = np.zeros(N)

    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            provides = graph.field(fid).intent == pfuse.Intent.Provides
            registry.bind_numpy(fid, self.out if provides else self.inp)

    def advance(self, ctx):
        x = self.features(self.inp, float(ctx.t_end))
        self.out[:] = self.model(x)       # write back in place; never rebind self.out

    def save_state(self):
        self.saved[:] = self.out

    def restore_state(self):
        self.out[:] = self.saved
```

The points that matter are the ones the array contract in [](python.md#arrays) imposes on every
component:

- `self.inp` and `self.out` are `float64`, C-contiguous numpy arrays bound once in
  `register_fields`. The runtime writes the inputs into `self.inp` before each `advance` and reads
  the outputs from `self.out` after it.
- The prediction is written **into** `self.out` with slice assignment. `self.out = pred` would
  leave the runtime reading the old buffer.
- `save_state`/`restore_state` copy the output, so the component can be a member of an iteration
  region (see [Iteration regions and rollback](python.md#iteration-regions-and-rollback)).

### With PyTorch

`torch.from_numpy` shares memory with a `float64` array, so inference reads the bound input with
no copy, and the result is copied back in place:

```python
# Requires torch; not executed as part of the documentation build.
import torch


class TorchSurrogate(SurrogateComponent):
    def __init__(self, name, module, features):
        super().__init__(name, model=None, features=features)
        self.module = module.double().eval()      # float64 weights to match the bound arrays

    def advance(self, ctx):
        x = torch.from_numpy(self.features(self.inp, float(ctx.t_end)))   # zero-copy view
        with torch.no_grad():
            y = self.module(x).reshape(-1)
        self.out[:] = y.numpy()                   # back into the bound array, in place
```

Keep the module in `float64` (`module.double()`), or convert explicitly at the boundary; a
`float32` model fed `float64` inputs raises inside PyTorch, and converting the bound array itself
to `float32` would break the binding. If the model runs on a GPU, move `x` there and bring `y`
back with `.cpu().numpy()`; the bound arrays stay in host memory. For scikit-learn, `model` is
simply `estimator.predict`.

### A complete, runnable example

The program below replaces one side of the Dirichlet-Neumann coupling from [](dn-heat.md) with a
least-squares "model" fitted to samples of the physics, then runs the physics component and the
surrogate through the **same graph** and compares them. It needs only numpy.

```python
"""A learned stand-in for one side of a Dirichlet-Neumann coupling, run through pfuse."""
import math
import numpy as np
import pfuse

N = 32
H = pfuse.Rational(1, 20)
RATIO = np.repeat([0.50, 0.75, 0.90, 0.96], N // 4)


def T_outer_1(t):
    return 340.0 + 40.0 * math.sin(2.0 * math.pi * t)


def T_outer_2(t):
    return 280.0 + 15.0 * math.sin(math.pi * t + 0.7)


# --- the components -------------------------------------------------------------------

class Slab(pfuse.Component):
    def __init__(self, name, initial=0.0):
        super().__init__(name)
        self.out = np.full(N, initial)
        self.inp = np.zeros(N)
        self.saved = np.zeros(N)

    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            provides = graph.field(fid).intent == pfuse.Intent.Provides
            registry.bind_numpy(fid, self.out if provides else self.inp)

    def save_state(self):
        self.saved[:] = self.out

    def restore_state(self):
        self.out[:] = self.saved


class Dirichlet(Slab):
    """The physics: wall temperature in, wall heat flux out."""

    def advance(self, ctx):
        self.out[:] = RATIO * (T_outer_1(float(ctx.t_end)) - self.inp)


class Neumann(Slab):
    def advance(self, ctx):
        self.out[:] = T_outer_2(float(ctx.t_end)) + self.inp


class SurrogateComponent(pfuse.Component):
    """Runs `model(features) -> prediction` on the bound input array each step.

    `model` is any callable taking a float64 (N, F) feature matrix and returning an (N,)
    prediction. The component owns no state beyond its bound arrays, so save/restore of the
    output is a complete checkpoint and the component may join an iteration region.
    """

    def __init__(self, name, model, features):
        super().__init__(name)
        self.model = model
        self.features = features          # (inputs, t) -> (N, F) float64
        self.out = np.zeros(N)
        self.inp = np.zeros(N)
        self.saved = np.zeros(N)

    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            provides = graph.field(fid).intent == pfuse.Intent.Provides
            registry.bind_numpy(fid, self.out if provides else self.inp)

    def advance(self, ctx):
        x = self.features(self.inp, float(ctx.t_end))
        self.out[:] = self.model(x)       # write back in place; never rebind self.out

    def save_state(self):
        self.saved[:] = self.out

    def restore_state(self):
        self.out[:] = self.saved


# --- a "trained" model: least-squares fit to samples of the physics ------------------

def features(u, t):
    """Per station: a constant, the input, and the forcing, each split by conductivity strip."""
    strip = np.repeat(np.eye(4), N // 4, axis=0)                 # (N, 4) one-hot
    forcing = np.sin(2.0 * math.pi * t) * np.ones(N)
    return np.column_stack([strip, strip * u[:, None], strip * forcing[:, None]])


rng = np.random.default_rng(0)
X, y = [], []
physics = Dirichlet("dirichlet")
for _ in range(200):
    t = rng.uniform(0.0, 1.0)
    physics.inp[:] = rng.uniform(280.0, 340.0, N)
    physics.out[:] = RATIO * (T_outer_1(t) - physics.inp)
    X.append(features(physics.inp, t))
    y.append(physics.out.copy())
X = np.concatenate(X)
y = np.concatenate(y)
weights, *_ = np.linalg.lstsq(X, y, rcond=None)


class LinearModel:
    def __init__(self, w):
        self.w = w

    def __call__(self, x):
        return x @ self.w


# --- the same graph for both runs -----------------------------------------------------

def build_graph():
    g = pfuse.CouplingGraph()
    m1 = g.add_mesh("iface_1", "generated:line:%d" % N)
    m2 = g.add_mesh("iface_2", "generated:line:%d" % N)
    r1 = g.add_region("iface_1", m1, dimension=1)
    r2 = g.add_region("iface_2", m2, dimension=1)
    cd = g.add_component("dirichlet", ranks=pfuse.RankSet(0, 1), interval=H)
    cn = g.add_component("neumann", ranks=pfuse.RankSet(1, 1), interval=H)
    q_out = g.add_field(cd, "q_wall", r1, units=pfuse.units.W_per_m2,
                        conservation=pfuse.Conservation.Extensive, intent=pfuse.Intent.Provides)
    u_in = g.add_field(cd, "u_wall_in", r1, units=pfuse.units.K,
                       conservation=pfuse.Conservation.Intensive, intent=pfuse.Intent.Requires)
    q_in = g.add_field(cn, "q_wall_in", r2, units=pfuse.units.W_per_m2,
                       conservation=pfuse.Conservation.Extensive, intent=pfuse.Intent.Requires)
    u_out = g.add_field(cn, "u_wall", r2, units=pfuse.units.K,
                        conservation=pfuse.Conservation.Intensive, intent=pfuse.Intent.Provides)
    flux = g.add_channel("flux", q_out, q_in)
    temp = g.add_channel("wall_temperature", u_out, u_in)
    g.channel(flux).conservative()
    g.channel(temp).lag(1).allow_time_level_mismatch("resolved by the iteration region")
    crit = pfuse.Converged.Relative(1e-8, "u_wall") & pfuse.Converged.Relative(1e-8, "q_wall")
    (g.iterate([flux, temp]).name("dirichlet_neumann").until(crit).max_iterations(500)
       .accelerate(pfuse.AccelConfig(kind=pfuse.AccelKind.Aitken, initial_relaxation=0.5)))
    return g


def run(dirichlet_side, periods=20):
    session = pfuse.Session(build_graph())
    neumann = Neumann("neumann", initial=T_outer_2(0.0))
    session.add_component(dirichlet_side)
    session.add_component(neumann)
    session.initialize()
    session.run(periods)
    return session.stats, neumann.out.copy()


stats_ref, u_ref = run(Dirichlet("dirichlet"))
stats_sur, u_sur = run(SurrogateComponent("dirichlet", LinearModel(weights), features))

print("physics  : sweeps %5d  unconverged %d" % (stats_ref.total_iterations,
                                                 stats_ref.windows_not_converged))
print("surrogate: sweeps %5d  unconverged %d" % (stats_sur.total_iterations,
                                                 stats_sur.windows_not_converged))
print("max |u_surrogate - u_physics| = %.3e K" % np.abs(u_sur - u_ref).max())
```

Expected output (exact digits can vary by platform):

```text
physics  : sweeps   938  unconverged 0
surrogate: sweeps   938  unconverged 0
max |u_surrogate - u_physics| = 1.256e-11 K
```

The physics here is affine in the features chosen, so the fit is exact and the two runs agree to
round-off, sweep for sweep. The point of the program is the comparison itself: the same graph,
the same partner component, the same accelerator and tolerance, and one line changed. With a
feature set that cannot represent the physics (dropping the per-strip split, for example) the
same program reports a maximum temperature difference of several kelvin and a different sweep
count, which is the answer you need before trusting the model inside a coupled run.

## What a trustworthy surrogate must state

A model that replaces a physics component inherits that component's obligations. Four of them
are easy to overlook and are worth stating explicitly, in code and in the record of the run.

**Hidden state.** A recurrent model, a model with an internal filter, or anything that carries
information from one step to the next has state the runtime cannot see. Declare it with
`CAPABILITIES = pfuse.CAP_HIDDEN_STATE`, and include that state in `save_state`/`restore_state`;
otherwise a rejected sweep in an iteration region restores the outputs but not the memory, and
the next sweep starts from a state no physics ever produced:

```python
class RecurrentSurrogate(SurrogateComponent):
    """A model with a hidden state h carried from step to step (a GRU-like cell, here linear)."""

    CAPABILITIES = pfuse.CAP_HIDDEN_STATE

    def __init__(self, name, model, features):
        super().__init__(name, model, features)
        self.h = np.zeros(N)
        self.saved_h = np.zeros(N)

    def advance(self, ctx):
        x = self.features(self.inp, float(ctx.t_end))
        self.h[:] = 0.9 * self.h + 0.1 * self.model(x)     # the recurrence: h is not a field
        self.out[:] = self.model(x)

    def save_state(self):
        super().save_state()
        self.saved_h[:] = self.h                           # checkpoint the hidden state too

    def restore_state(self):
        super().restore_state()
        self.h[:] = self.saved_h
```

The runtime verifies the restored `Provides` storage after every `restore_state` and counts the
checks in `RunStats.checkpoint_verifications`; it cannot verify `h`, so the checkpoint of hidden
state is entirely the component's responsibility.

**Rollback.** Without `save_state`/`restore_state` the component is refused as a member of an
iteration region (`E-ITER-NOROLLBACK`). For a stateless model the pair is two lines, as above.

**Determinism.** A fixed-point iteration decides convergence by comparing successive iterates.
A model that is stochastic (dropout left on, sampling, a nondeterministic GPU kernel) produces
noise that the convergence test cannot tell from a residual: the region either fails to
converge or stops on a lucky draw. Inside an iteration region, make inference deterministic:
evaluation mode, fixed seeds, deterministic kernels. In an explicit coupling (no iteration
region) a stochastic model is usable, but each run is then one sample.

**Applicability.** A model is valid on the region of input space it was trained on. Nothing in
the runtime checks that the coupled run stays inside it; a coupled system drifts, and an
out-of-distribution input returns a confident wrong number instead of a solver failure. Build the
check into `advance()`: a distance to the training distribution, a bound check on the features,
or an ensemble disagreement, and raise an exception when it trips. A Python exception raised in
`advance()` stops the run and reaches your driver verbatim ([](python.md#errors-inside-a-component)).

:::{warning}
A surrogate cannot be a member of a Newton-solved region. Newton needs the residual and
linearisation calls of a native `fuse::Component`, which a Python component cannot supply:
claiming `CAP_RESIDUAL` or `CAP_LINEARIZATION` is refused at `add_component()`, and a Newton
region with a Python member is refused at `initialize()` with `E-NEWTON-CAPABILITY`. Use Picard,
Aitken, Secant, Steffensen or IQN acceleration for a region that contains a surrogate.
:::

## Validate a surrogate against the physics it replaces

The example above is the pattern:

1. Author one graph. Do not simplify it for the surrogate; the channels, remaps, lag and
   convergence criterion are part of what you are validating.
2. Run it with the physics component, keep the partner's final fields, `RunStats` and the
   residual histories.
3. Run it again with the surrogate in the physics component's place, and only that changed.
4. Compare the partner's fields (the quantity the rest of the simulation consumes), the sweep
   counts (a surrogate that is accurate but non-smooth can make the fixed point harder to reach),
   `windows_not_converged`, and `conservation_violations` if the surrogate provides an extensive
   field.
5. Repeat over the operating range: sweep the forcing, the initial state, the coupling period and
   the tolerance in an outer loop, as the accelerator sweep in
   [](python.md#a-parameter-sweep-the-dirichlet-neumann-accelerators) does.

Because both runs go through the same runtime, the comparison isolates the model. A difference is
the model's, not an artefact of two different coupling setups.

## The C++ contract

The C++ library declares what a substitutable surrogate must state, as a type you can fill in and
have reviewed today, in `include/fuse/model/multiscale.hpp`:

```cpp
struct SurrogateContract {
  std::string name;
  std::vector<FieldSemantics> applicability;   // where the model claims validity; empty = no claim
  std::string ood_detector;                    // how out-of-distribution input is detected at run time
  bool carries_hidden_state = false;
  DeterminismLevel determinism = DeterminismLevel::Nondeterministic;   // Bitwise, RunToRun, Nondeterministic
  DerivativeKind derivative = DerivativeKind::None;                    // None, Forward, Reverse, Both
};

[[nodiscard]] Diagnostics review_surrogate_contract(const SurrogateContract& contract);
[[nodiscard]] Result<void> substitute_surrogate(ComponentId component,
                                                const SurrogateContract& contract);
```

`review_surrogate_contract()` checks the declaration for completeness and reports every gap at
once:

| Code | Severity | Condition |
|---|---|---|
| `E-SURROGATE-UNNAMED` | error | `name` is empty |
| `E-SURROGATE-NO-APPLICABILITY` | error | `applicability` is empty: there is nothing against which an input can be judged in or out of distribution |
| `E-SURROGATE-NO-OOD` | error | `ood_detector` is empty: a surrogate that cannot report when it is extrapolating is not substitutable for a component that would have converged or failed |
| `W-SURROGATE-NONDETERMINISTIC` | warning | `determinism` is `Nondeterministic` (the default); such a model cannot be used inside an iteration region |
| `W-SURROGATE-HIDDEN-STATE` | warning | `carries_hidden_state` is `true`; an iteration region or a rollback needs an explicit checkpoint hook from the model |

```cpp
#include "fuse/model/multiscale.hpp"

fuse::SurrogateContract c;
c.name = "closure-emulator";
fuse::FieldSemantics input;
input.id = "strain";
c.applicability = {input};
c.ood_detector = "mahalanobis-on-strain-invariants";
c.determinism = fuse::DeterminismLevel::Bitwise;
c.derivative = fuse::DerivativeKind::Forward;

const fuse::Diagnostics review = fuse::review_surrogate_contract(c);
if (review.has_errors()) std::cerr << review.to_string();
```

The library does not substitute a surrogate for a declared component itself:
`substitute_surrogate()` reviews the contract and refuses, with `E-SURROGATE-CONTRACT` when the
review has errors and `E-SURROGATE-NOT-IMPLEMENTED` otherwise. The component route above is how a
surrogate runs in a coupling. The contract is still the right checklist for a Python surrogate:
name it, state the input region it is valid on, say how you detect leaving that region, declare
hidden state (`CAP_HIDDEN_STATE`) and determinism honestly, and know which derivative, if any,
the model can supply. `SurrogateContract` is not bound in `pfuse`.

The multiscale declarations that sit next to the contract (instance sets, closures, overlap
accounting, sample pools) are described in [](multiscale.md).

## See also

- [](python.md) for the component protocol, the array contract and the run statistics.
- [](multiscale.md) for the multiscale declarations and their status.
- [](dn-heat.md) for the problem the example above borrows and [](convergence.md) for the
  measures an iteration region uses.
- [](concepts.md#iteration) for what rollback is for.
