# Coupling from Python (`pfuse`)

`pfuse` is the Python module over FUSE's C++ model and runtime: the same `CouplingGraph`,
`validate()`, `lower()`, scheduler and `Runner` that a C++ program uses, driven from Python, with
components written as Python classes over numpy arrays. This page is for anyone who wants to
author or run a coupling from Python, whether as the final program or as a way to prototype one
that will later be driven from C++.

## When to use it

- **Python physics.** A solver written in Python (or wrapping a library through numpy) joins a
  coupled run as a full peer: it is advanced by the real `fuse::Runner`, its fields are remapped,
  accumulated and checked for conservation like any other component's.
- **Prototyping a coupling.** Author the graph, run it with simple stand-in components, look at
  the lowered dataflow and the execution plan, and change the treatment of a channel without
  recompiling anything.
- **Authoring descriptions for a C++ driver.** A graph built in Python serialises to the same YAML
  and binary descriptor every other front end reads, so Python can be the place where a coupling
  is written and a C++ executable the place where it runs.
- **Teaching and experiments.** Sweep accelerators, tolerances and time steps in a loop and
  collect `RunStats` and residual histories as plain Python data.
- **Learned components.** A model's inference call fits the component protocol directly; see
  [](surrogates.md).

`pfuse` is not a way to call into an existing coupled run from outside its process, and it is not
the participant ABI ([](c-abi.md), [](fortran.md)). It is in-process: the Python interpreter owns
the run.

## Build and install

`pfuse` is built from the FUSE build tree behind the `FUSE_ENABLE_PYTHON` option (default `OFF`,
see [](build.md)). It needs `pybind11` and `numpy` installed in the interpreter you will use, and
the same interpreter must be given to CMake:

```bash
python3 -m pip install pybind11 numpy pytest

cmake -S . -B build -DFUSE_ENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python3)" \
  -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake --build build --target _pfuse
```

The `_pfuse` target builds the extension module and stages it, together with `pfuse/__init__.py`,
under `<build>/python/pfuse/`. There is no install step: put `<build>/python` on `PYTHONPATH`.

```bash
export PYTHONPATH=/path/to/build/python
python3 -c "import pfuse; print(pfuse.version())"
```

Expected output: the FUSE version string, for example `0.1.0`.

The extension is compiled against one interpreter's ABI and imports only under that interpreter.
Importing `pfuse` initialises Kokkos once for the process and registers its finalisation with
`atexit`; you do not call either yourself.

With `FUSE_ENABLE_TESTS` also on, the Python tests are registered as `ctest` rows named
`python.buffers`, `python.graph`, `python.yaml`, `python.runtime`, `python.newton`,
`python.dn_heat_rows` and `python.analytic_problem`, all under the label `python`:

```bash
ctest --test-dir build -L python --output-on-failure
```

They need `pytest` and `numpy` in the build interpreter. The sources are under `python/tests/` and
are a good second reference for every call on this page.

## A first coupled run

Two Python components exchange a heat flux and a wall temperature through one iteration region
accelerated with Aitken relaxation. This is the shape of every coupled run in `pfuse`: declare
the graph, write a `Component` per declared component, hand both to a `Session`, run, read the
statistics.

```python
import numpy as np
import pfuse


class Upstream(pfuse.Component):
    """Produces a flux from the temperature it is handed."""

    def __init__(self):
        super().__init__("upstream")
        self.flux = np.zeros(1)
        self.t_in = np.zeros(1)
        self._saved = np.zeros(1)

    def register_fields(self, registry, graph):
        cid = graph.find_component(self.name)
        for f in graph.component(cid).fields:
            if graph.field(f).intent == pfuse.Intent.Provides:
                registry.bind_numpy(f, self.flux)
            else:
                registry.bind_numpy(f, self.t_in)

    def advance(self, ctx):
        self.flux[0] = 2.0 * (300.0 - self.t_in[0])

    def save_state(self):
        self._saved[:] = self.flux

    def restore_state(self):
        self.flux[:] = self._saved


class Downstream(pfuse.Component):
    """Relaxes its interface temperature toward the flux it receives."""

    def __init__(self):
        super().__init__("downstream")
        self.t = np.full(1, 280.0)
        self.flux_in = np.zeros(1)
        self._saved = np.zeros(1)

    def register_fields(self, registry, graph):
        cid = graph.find_component(self.name)
        for f in graph.component(cid).fields:
            if graph.field(f).intent == pfuse.Intent.Provides:
                registry.bind_numpy(f, self.t)
            else:
                registry.bind_numpy(f, self.flux_in)

    def advance(self, ctx):
        self.t[0] = self.t[0] + 0.1 * self.flux_in[0]

    def save_state(self):
        self._saved[:] = self.t

    def restore_state(self):
        self.t[:] = self._saved


g = pfuse.CouplingGraph()
mesh = g.add_mesh("iface", "generated:line:1")
region = g.add_region("iface", mesh, dimension=1)

up = g.add_component("upstream", ranks=pfuse.RankSet(0, 1), interval=1)
down = g.add_component("downstream", ranks=pfuse.RankSet(0, 1), interval=1)

flux_out = g.add_field(up, "flux", region, units=pfuse.units.W_per_m2,
                       conservation=pfuse.Conservation.Extensive, intent=pfuse.Intent.Provides)
t_in = g.add_field(up, "t_in", region, units=pfuse.units.K,
                   conservation=pfuse.Conservation.Intensive, intent=pfuse.Intent.Requires)
flux_in = g.add_field(down, "flux_in", region, units=pfuse.units.W_per_m2,
                      conservation=pfuse.Conservation.Extensive, intent=pfuse.Intent.Requires)
t_out = g.add_field(down, "t", region, units=pfuse.units.K,
                    conservation=pfuse.Conservation.Intensive, intent=pfuse.Intent.Provides)

flux_ch = g.add_channel("flux", flux_out, flux_in)
temp_ch = g.add_channel("temperature", t_out, t_in)
g.channel(flux_ch).conservative()
g.channel(temp_ch).lag(1).allow_time_level_mismatch("resolved inside the iteration region")

criterion = pfuse.Converged.Relative(1e-6, "t") & pfuse.Converged.Relative(1e-6, "flux")
g.iterate([flux_ch, temp_ch]).name("interface").until(criterion).max_iterations(200).accelerate(
    pfuse.AccelConfig(kind=pfuse.AccelKind.Aitken, initial_relaxation=0.5))

session = pfuse.Session(g)
session.add_component(Upstream())
session.add_component(Downstream())
session.initialize()
session.run(1)

stats = session.stats
print("sweeps:", stats.total_iterations, "unconverged windows:", stats.windows_not_converged)
print("converged:", session.iteration_stats[0].converged,
      "last residual: %.3e" % session.iteration_stats[0].residual_history[-1])
print("conservation checks:", stats.conservation_checks, "violations:", stats.conservation_violations)
```

Expected output (exact digits can vary by platform):

```text
sweeps: 31 unconverged windows: 0
converged: True last residual: 1.761e-05
conservation checks: 31 violations: 0
```

What happened: `Session(g)` took a copy of the graph; `initialize()` loaded the generated mesh,
ran `validate()`, `lower()` and the scheduler, called each component's `register_fields`, and set
up the runtime; `run(1)` executed one coupling period, during which the iteration region swept
both components 31 times, checkpointing and restoring them around each sweep, until both
measures fell below `1e-6`.

:::{note}
Both fields on the flux channel are declared `Extensive` and the channel is marked
`.conservative()`. A channel into an extensive target that is not marked conservative is refused
by `validate()` with `E-CONN-CONSERVATION`.
:::

## Author a graph in Python

Every declaration of the C++ fluent API ([](fluent-api.md)) has a Python spelling with the same
name. The graph below uses all of them; it validates, lowers, and runs with the components shown
in [Run the tour graph](#run-the-tour-graph).

```python
import pfuse

g = pfuse.CouplingGraph()

# Meshes and regions. The serial backend accepts "generated:line:N" and "generated:box:NxMxP".
fluid_mesh = g.add_mesh("fluid_mesh", "generated:line:32")
solid_mesh = g.add_mesh("solid_mesh", "generated:line:16")
fluid_wall = g.add_region("fluid_wall", fluid_mesh, dimension=1)
solid_wall = g.add_region("solid_wall", solid_mesh, dimension=1)

# Components: a name, the ranks it runs on, its time step, and its kind.
fluid = g.add_component("fluid", ranks=pfuse.RankSet(0, 1), interval="1/4")
solid = g.add_component("solid", ranks=pfuse.RankSet(1, 1), interval="1/4",
                        kind=pfuse.ComponentKind.Physics)
hub = g.add_coupler("hub", ranks=pfuse.RankSet([0, 1]))   # remaps only, owns no fields

# Fields: owner, name, region, discretisation, order, components, units, conservation, intent.
q_out = g.add_field(fluid, "heat_flux", fluid_wall,
                    disc=pfuse.Discretization.FV, order=0, components=1,
                    units=pfuse.units.W_per_m2,
                    conservation=pfuse.Conservation.Extensive,
                    intent=pfuse.Intent.Provides)
q_in = g.add_field(solid, "q_wall", solid_wall, units=pfuse.units.W_per_m2,
                   conservation=pfuse.Conservation.Extensive,
                   intent=pfuse.Intent.Requires)
t_out = g.add_field(solid, "T_wall", solid_wall, units=pfuse.units.K,
                    conservation=pfuse.Conservation.Intensive,
                    intent=pfuse.Intent.Provides)
t_in = g.add_field(fluid, "T_in", fluid_wall, units=pfuse.units.K,
                   conservation=pfuse.Conservation.Intensive,
                   intent=pfuse.Intent.Requires)

# Channels: one directed transfer each, then its treatment on the builder.
flux = g.add_channel("flux", q_out, q_in)
(g.channel(flux)
   .conservative()                                   # check the integral on the way
   .remap(pfuse.RemapFlags.Conservative | pfuse.RemapFlags.Monotone)
   .remap_method("mono3")
   .remap_on(hub)                                    # build and apply the weights on the coupler
   .relax(0.7))                                      # under-relax inside the iteration region
temp = g.add_channel("wall_temperature", t_out, t_in)
(g.channel(temp)
   .lag(1)                                           # break the flux/temperature cycle
   .allow_time_level_mismatch("resolved by the iteration region"))

# An iteration region over the cycle, with a criterion per exchanged field and an accelerator.
crit = pfuse.Converged.Relative(1e-8, "T_wall") & pfuse.Converged.Absolute(1e-6, "heat_flux")
region = (g.iterate([flux, temp])
            .name("interface")
            .until(crit)
            .max_iterations(100)
            .accelerate(pfuse.AccelConfig(kind=pfuse.AccelKind.IQN_ILS,
                                          initial_relaxation=0.5, reuse=8,
                                          filter=pfuse.FilterKind.QR2, limit=1e-2)))

diags = pfuse.validate(g)
print(diags.to_string() or "no diagnostics")
print(g.component_count, "components,", g.field_count, "fields,",
      g.channel_count, "channels,", g.iteration_region_count, "iteration region")
```

Expected output:

```text
warning W-MISMATCH-OK: time-level mismatch permitted: resolved by the iteration region [channel 'wall_temperature']

3 components, 4 fields, 2 channels, 1 iteration region
```

### Declarations

| Call | Returns | Notes |
|---|---|---|
| `add_mesh(name, source)` | `MeshId` | `source` is `"generated:line:N"` or `"generated:box:NxMxP"` |
| `add_region(name, mesh, entity_set="", dimension=-1)` | `RegionId` | the entities of one dimension a field lives on |
| `add_component(name, ranks=None, interval=1, kind=Physics)` | `ComponentId` | `ranks` is a `RankSet`; `interval` is the time step (see below) |
| `add_coupler(name, ranks=None)` | `ComponentId` | a component of kind `Coupler`: no fields, remaps only |
| `add_field(owner, name, region, disc=FV, order=0, components=1, units=Units(), conservation=None_, intent=Provides)` | `FieldId` | `components > 1` makes a vector field |
| `add_channel(name, source, target)` | `ChannelId` | from a `Provides` field to a `Requires` one |
| `channel(id)` | `ChannelBuilder` | chain the treatments below |
| `iterate([channel ids])` | `IterationBuilder` | chain `.name`, `.until`, `.max_iterations`, `.accelerate`, `.initial_relaxation`, `.newton`, `.requires_residual`, `.requires_linearization`; `.id()` gives the `IterationRegionIdx` |

`RankSet(first, count)` is a contiguous set, `RankSet([0, 2, 3])` an explicit one; it exposes
`ranks`, `size`, `min_rank`, `max_rank`, `is_contiguous`, `contains(r)`, `overlaps(other)` and
`same_as(other)`. In a `pfuse` run every rank number is nominal: the session is one process, and
the rank sets only steer what the plan derives (where a transport is inserted, for example).

Enumerations are spelled as in C++, with `None` written `None_` because `None` is a Python
keyword: `pfuse.Conservation.None_`, `pfuse.AccelKind.None_`, `pfuse.RemapFlags.None_`,
`pfuse.TemporalBasis.None_`, `pfuse.FilterKind.None_`.

### Channel treatments

| Builder call | Effect |
|---|---|
| `.accumulate(op, window=None)` | reduce a faster source over the target's step: `AccumOp.Sum`, `Mean`, `Min`, `Max`, `Integral` (`Instant` is the default, no reduction). `window` is a `Window`: `Window.coupling_period()`, `Window.from_component(cid)` or `Window.explicit_length("2/5")` |
| `.conservative()` | insert a conservation check; required when the target field is `Extensive` |
| `.remap(flags)` | request remap properties: `RemapFlags.Conservative`, `Monotone`, `HighOrder`, combined with `\|` |
| `.remap_method(name)` | pin a specific remap method, for example `"mono3"` |
| `.remap_on(coupler)` | build and apply the weights on that coupler component |
| `.spacetime(basis, degree, window=None)` | interpolate the source in time: `TemporalBasis.Constant`, `Linear`, `DG`, `Spline` |
| `.extrapolate(order)` | extrapolate the source forward in time (order 0, 1 or 2) |
| `.relax(omega)` | under-relax the delivered value by `omega` |
| `.lag(steps)` | deliver the value from `steps` coupling steps ago; breaks a cycle |
| `.allow_time_level_mismatch(reason)` | authorise the staggering a lag introduces; the reason is kept in the graph |

The chains return the builder, so one expression configures a channel completely. Builders refer
to the graph they came from and are meant to be used in the expression that created them.

Accumulation on a multirate channel, measured:

```python
import numpy as np
import pfuse


class Producer(pfuse.Component):
    def __init__(self, cells):
        super().__init__("producer")
        self.data = np.zeros(cells)
        self.steps = 0

    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            registry.bind_numpy(fid, self.data)

    def advance(self, ctx):
        self.steps += 1
        self.data[:] = 2.0 * self.steps          # 2, 4, 6, 8 over the four sub-steps


class Consumer(pfuse.Component):
    def __init__(self, cells):
        super().__init__("consumer")
        self.data = np.zeros(cells)
        self.seen = []

    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            registry.bind_numpy(fid, self.data)

    def advance(self, ctx):
        self.seen.append(float(self.data[0]))


def line_pair(op):
    g = pfuse.CouplingGraph()
    m1 = g.add_mesh("m1", "generated:line:8")
    m2 = g.add_mesh("m2", "generated:line:8")
    r1 = g.add_region("r1", m1, dimension=1)
    r2 = g.add_region("r2", m2, dimension=1)
    p = g.add_component("producer", ranks=pfuse.RankSet(0, 1), interval="1/4")
    c = g.add_component("consumer", ranks=pfuse.RankSet(1, 1), interval=1)
    s = g.add_field(p, "out", r1, units=pfuse.units.K, intent=pfuse.Intent.Provides)
    t = g.add_field(c, "inp", r2, units=pfuse.units.K, intent=pfuse.Intent.Requires)
    g.channel(g.add_channel("link", s, t)).accumulate(op)
    return g


for op in (pfuse.AccumOp.Mean, pfuse.AccumOp.Max):
    session = pfuse.Session(line_pair(op))
    consumer = Consumer(8)
    session.add_component(Producer(8))
    session.add_component(consumer)
    session.initialize()
    session.run(2)
    print(op, "period =", session.plan.period, "lockstep =", session.plan.lockstep,
          "consumer saw", consumer.seen)
```

Expected output:

```text
AccumOp.Mean period = 1 lockstep = False consumer saw [0.0, 5.0]
AccumOp.Max period = 1 lockstep = False consumer saw [0.0, 8.0]
```

The consumer's first step reads the initial zero: an explicit channel delivers what the previous
period produced. Its second step reads the mean (5) or maximum (8) of the four values the producer
wrote during the first period.

### Iteration regions

`g.iterate([...])` takes the channels that form the loop. Every exchanged field must be measured,
so `until()` takes a `ConvergenceCriterion` built with `&` from one measure per field:

```python
crit = pfuse.Converged.Relative(1e-8, "T_wall") & pfuse.Converged.Absolute(1e-6, "heat_flux")
```

`pfuse.Converged` has `Relative(tol, field)`, `Absolute(tol, field)`, `AbsoluteOrRelative(tol,
field)` and `ResidualRelative(tol, field)`; each is a `ConvergenceMeasure` with `kind`,
`tolerance` and `field`. `pfuse.ConvergenceCriterion()` with `.add(measure)` builds the same thing
incrementally; `len(crit)` and `crit.measures` read it back. [](convergence.md) explains the
measures.

`accelerate()` takes an `AccelConfig(kind=Constant, initial_relaxation=0.1, omega=0.5, reuse=8,
filter=QR2, limit=1e-2, scale_fields=True)`. `kind` is one of `AccelKind.None_` (plain Picard),
`Constant`, `Aitken`, `Secant`, `Steffensen`, `IQN_ILS`, `IQN_IMVJ` or `Newton`; `reuse`,
`filter` and `limit` matter for the IQN variants. [](dn-heat.md) compares them on one problem.

`.newton(config=None)` selects the Jacobian-free Newton-Krylov solver for the region and takes a
`NewtonConfig`. Its keyword arguments, with defaults: `line_search_kind=LineSearchKind.BT`,
`forcing_choice=EisenstatWalkerChoice.Choice2`, `forcing_gamma=0.9`, `forcing_alpha=2.0`,
`eta_initial=0.3`, `eta_max=0.9`, `eta_min=0.0`, `forcing_safeguard_threshold=0.1`,
`forcing_eta_constant=0.1`, `atol=1e-50`, `rtol=1e-8`, `stol=1e-8`, `divergence_tolerance=1e4`,
`max_nonlinear_iterations=50`, `max_function_evaluations=10000`, `armijo_alpha=1e-4`,
`lambda_min=1e-12`, `max_backtracks=40`, `secant_tol=1e-10`, `max_secant_steps=40`,
`damping=1.0`, `gmres_restart=30`, `gmres_max_iterations=1000`, `preconditioner_lag=1`,
`use_block_jacobi_preconditioner=False`, `substrate=NewtonSubstrate.Native`,
`norm=NewtonNorm.Plain`, `residual_path=NewtonResidualPath.Advance`. Selecting Newton sets the
region's `requires_residual_capability`; `use_block_jacobi_preconditioner=True` also sets
`requires_linearization_capability`. A Python session cannot run such a region; see
[Newton regions](#author-a-newton-region-run-it-from-c).

:::{warning}
The channels of one iteration region must be on one clock: every member component steps with the
same `interval`. A region whose channels mix clocks is refused by the scheduler with
`E-SCHED-REGION`. Time interpolation (`.spacetime`, `.extrapolate`) belongs on explicit channels,
not on channels inside a region.
:::

### Exact time: Rational

Time steps and window lengths are exact fractions. Every `interval` or length argument accepts a
`pfuse.Rational`, an `int`, a string such as `"1/3"`, or a `fractions.Fraction`; `pfuse.rational(x)`
applies the same rule explicitly. A `float` is refused with `InvalidArgument`, and so is a `bool`:

```python
from fractions import Fraction
import pfuse

print(pfuse.rational("1/3"), pfuse.rational(Fraction(2, 6)), pfuse.rational(4))
r = pfuse.Rational(1, 3)
print(r + pfuse.Rational(1, 6), r * pfuse.Rational(3), r < pfuse.Rational(1, 2), float(r))
print(r.num, r.den, r.is_integer(), pfuse.Rational(5, 10))     # reduced on construction
```

```text
1/3 1/3 4
1/2 1 True 0.3333333333333333
1 3 False 1/2
```

`Rational` supports `+ - * /` with other `Rational`s, comparisons, hashing, `str()` and
`float()`; `to_double()` is the explicit conversion for the edge of your physics. Mixed arithmetic
with a Python `int` is not defined: write `pfuse.Rational(periods) * H`, not `periods * H`.

### Units

`pfuse.units` provides `dimensionless`, `K`, `metre`, `kilogram`, `second`, `newton`, `joule`,
`watt`, `pascal` and `W_per_m2`. Units multiply and divide (`pfuse.units.metre /
pfuse.units.second`), and `pfuse.Units(length=0, mass=0, time=0, current=0, temperature=0,
amount=0, luminosity=0, scale=None, offset=None)` builds any other from base-dimension exponents.
`scale` and `offset` are exact rationals and follow the `Rational` rule above.

A channel between fields of the same dimension and different scale gets an exact conversion;
`validate()` reports it as an informational diagnostic:

```python
import numpy as np
import pfuse

kW_per_m2 = pfuse.Units(mass=1, time=-3, scale=1000)      # exponents of the SI base dimensions
print(kW_per_m2, kW_per_m2.same_dimensions(pfuse.units.W_per_m2), kW_per_m2 == pfuse.units.W_per_m2)

g = pfuse.CouplingGraph()
m = g.add_mesh("m", "generated:line:4")
r = g.add_region("r", m, dimension=1)
a = g.add_component("a", ranks=pfuse.RankSet(0, 1))
b = g.add_component("b", ranks=pfuse.RankSet(0, 1))
q_out = g.add_field(a, "q", r, units=pfuse.units.W_per_m2, intent=pfuse.Intent.Provides)
q_in = g.add_field(b, "q_in", r, units=kW_per_m2, intent=pfuse.Intent.Requires)
g.add_channel("q", q_out, q_in)
print(pfuse.validate(g).to_string())


class Source(pfuse.Component):
    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            registry.bind_numpy(fid, np.full(4, 1500.0))      # W/m^2

    def advance(self, ctx):
        pass


class Sink(pfuse.Component):
    def __init__(self):
        super().__init__("b")
        self.data = np.zeros(4)

    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            registry.bind_numpy(fid, self.data)

    def advance(self, ctx):
        pass


session = pfuse.Session(g)
session.add_component(Source("a"))
sink = Sink()
session.add_component(sink)
session.initialize()
session.run(1)
print(sink.data)                                           # kW/m^2
```

```text
1000*kg*s^-3 True False
warning I-UNIT-SCALE-INSERTED: inserting an exact unit scale of 1/1000 from 'q' to 'q_in' [channel 'q']

[1.5 1.5 1.5 1.5]
```

Fields of different dimensions on one channel are refused with `E-UNITS`. A conversion that needs
an offset (for example kelvin to a unit declared with `offset=`) is refused with `E-UNITS-SCALE`:
affine conversions have to be requested explicitly, and that request is not available from
Python, so declare both ends in the same absolute unit.

### Read the graph back

Every declaration can be read back, which is how a component finds its own fields and how a
script checks what it built:

| Reader | Returns |
|---|---|
| `g.mesh(id)` | `MeshDecl`: `name`, `source` |
| `g.region(id)` | `Region`: `name`, `mesh`, `entity_set`, `dimension` |
| `g.component(id)` | `ComponentDecl`: `name`, `ranks`, `interval`, `fields` (list of `FieldId`), `kind` |
| `g.field(id)` | `FieldSpec`: `name`, `region`, `disc`, `order`, `components`, `units`, `conservation`, `intent` |
| `g.channel_at(id)` | `Channel`: `name`, `source`, `target`, `accumulate`, `remap`, `spacetime`, `relax`, `extrapolate`, `lag`, `time_level_mismatch_reason` |
| `g.iteration_region(idx)` | `IterationRegionDecl`: `name`, `channels`, `max_iterations`, `convergence`, `acceleration`, `requires_residual_capability`, `requires_linearization_capability`, `newton_config` |
| `g.owner_of(field)` | the owning `ComponentId` |
| `g.find_component(name)`, `g.find_field(owner, name)` | the id; raise `InvalidArgument` when there is no such name |
| `g.mesh_count`, `region_count`, `component_count`, `field_count`, `channel_count`, `iteration_region_count` | counts |

The specs inside a `Channel` are `AccumulateSpec(op, window)`, `RemapSpec(flags, method_override,
remap_on)`, `SpacetimeSpec(basis, degree, window)`, `RelaxSpec(enabled, omega)` and
`ExtrapolateSpec(enabled, order)`. `pfuse.has(flags, bit)` tests one bit of a `RemapFlags` value.

Ids (`ComponentId`, `RegionId`, `FieldId`, `ChannelId`, `MeshId`, `IterationRegionIdx`) are small
value types with `.value`, `.valid`, equality, ordering and hashing, so they work as dictionary
keys.

### Validate

`pfuse.validate(g)` returns a `Diagnostics`: iterate it for `Diagnostic` objects with `severity`
(`Severity.Warning` or `Severity.Error`), `code`, `message` and `context`; `has_errors`, `empty`,
`len()`, `bool()` and `to_string()` summarise it. Every defect is reported together, so fix them
all before trying again. `Session.initialize()` runs the same validation and raises when it
finds errors (see [Errors](#errors)). `validate()` runs with the default connection policy; a
different policy cannot be passed from Python.

### Compare and copy

`pfuse.structurally_equal(a, b)` compares two graphs declaration by declaration, ignoring object
identity; it is what the round-trip checks below use. `copy.copy(g)` makes an independent graph.
`Session(g)` also copies, so changing `g` after constructing a session does not affect the
session (`session.graph` is the copy).

## Write a component

A component is a subclass of `pfuse.Component` whose constructor passes the name of a component
declared in the graph. The runtime calls four methods:

| Method | Required | What it must do |
|---|---|---|
| `register_fields(self, registry, graph)` | yes | call `registry.bind_numpy(field_id, array)` once for each field of this component, found through `graph.component(graph.find_component(self.name)).fields` |
| `advance(self, ctx)` | yes | integrate from `ctx.t_begin` to `ctx.t_end`, reading `Requires` arrays and writing `Provides` arrays in place |
| `save_state(self)` | for iteration regions | copy everything `advance` changes into private storage |
| `restore_state(self)` | for iteration regions | copy it back element-wise (`self.out[:] = self.saved`), never rebind the attribute |

The base class refuses to be used unmodified: calling `advance` or `register_fields` on a
`pfuse.Component` that does not override them raises `TypeError`, and an empty name raises
`InvalidArgument`. `graph` is passed by reference to the session's own copy; the `registry` object
may be kept and used after `register_fields` returns (it shares ownership of the session's
registry and stays valid even after the session is garbage-collected).

### Arrays

`bind_numpy` accepts exactly one kind of array: `numpy.float64` in native byte order,
C-contiguous, writeable, one-dimensional of shape `(count,)` or two-dimensional of shape
`(count, components)`. The runtime reads and writes that memory directly, without copies, so
anything else would alias the wrong bytes:

```python
import numpy as np
import pfuse

reg = pfuse.FieldRegistry()
f0 = pfuse.FieldId(0)

data = np.zeros(4)
reg.bind_numpy(f0, data)
view = reg.get_numpy(f0)          # a zero-copy view of `data`
view[1] = 7.5
print(data[1], reg.is_bound(f0), reg.bound_count)

# What is refused, and why
for label, bad in [
    ("float32", np.zeros(4, dtype=np.float32)),
    ("strided slice", np.zeros(8)[::2]),
    ("transpose", np.zeros((3, 5)).T),
    ("read-only", np.zeros(4)),
    ("3-D", np.zeros((2, 3, 4))),
    ("list", [1.0, 2.0]),
]:
    if label == "read-only":
        bad.setflags(write=False)
    try:
        pfuse.FieldRegistry().bind_numpy(f0, bad)
    except pfuse.InvalidArgument as e:
        print("%-14s refused" % label)

# Rebinding is refused; the first binding stays
try:
    reg.bind_numpy(f0, np.ones(4))
except pfuse.FuseError as e:
    print(e.code)

# A bound array cannot be resized in place
try:
    data.resize(8, refcheck=True)
except ValueError:
    print("resize refused by numpy")

# Two-dimensional fields are (count, components), C-contiguous
vec = np.zeros((5, 3))
reg.bind_numpy(pfuse.FieldId(1), vec)
print(reg.get_numpy(pfuse.FieldId(1)).shape)
```

```text
7.5 True 1
float32        refused
strided slice  refused
transpose      refused
read-only      refused
3-D            refused
list           refused
E-FIELD-REBIND
resize refused by numpy
(5, 3)
```

The rules that follow from this:

- **Write in place.** `self.out[:] = values` writes into the bound memory. `self.out = values`
  rebinds the Python name to a different array and the runtime keeps reading the old one.
  `np.ascontiguousarray(x.T)` is a copy, so binding it does not bind `x`.
- **Shape is checked against the declaration.** The element count must match the region's size
  from the mesh (`E-RT-SIZE` at `initialize()` otherwise) and the second dimension must match the
  field's `components` (an `InvalidArgument` naming both numbers).
- **Lifetime is handled for you.** The registry holds a strong reference to every bound array for
  as long as the session lives, so binding a temporary (`registry.bind_numpy(f, np.zeros(n))`) is
  safe. The flip side is that `ndarray.resize` refuses, because the array has more than one
  owner. If a field's size can change, build a new graph and session.
- **A field is bound once.** A second `bind_numpy` for the same id raises `FuseError` with code
  `E-FIELD-REBIND` and leaves the first binding in place; `get_numpy` of an unbound id raises
  `E-FIELD-UNBOUND`. A component that binds nothing is caught at `initialize()` with
  `E-RT-UNBOUND`.

### StepContext

`advance` receives a `StepContext` with `t_begin` and `t_end` (exact `Rational`s), `dt()`
(their difference), `step` (this component's step count since the start of the run) and
`iteration` (the sweep number inside an iteration region, `0` outside one). `float(ctx.t_end)`
is the usual way into your physics. The trace below shows what a member of an iteration region
sees:

```python
import numpy as np
import pfuse


class Trace(pfuse.Component):
    """Logs every call the runtime makes, with the StepContext it was handed."""

    def __init__(self, name, gain):
        super().__init__(name)
        self.gain = gain
        self.out = np.zeros(4)
        self.inp = np.zeros(4)
        self.saved = np.zeros(4)
        self.log = []

    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            provides = graph.field(fid).intent == pfuse.Intent.Provides
            registry.bind_numpy(fid, self.out if provides else self.inp)

    def advance(self, ctx):
        self.log.append("advance t=[%s, %s] dt=%s step=%d iteration=%d"
                        % (ctx.t_begin, ctx.t_end, ctx.dt(), ctx.step, ctx.iteration))
        self.out[:] = 1.0 + self.gain * self.inp

    def save_state(self):
        self.log.append("save_state")
        self.saved[:] = self.out

    def restore_state(self):
        self.log.append("restore_state")
        self.out[:] = self.saved


g = pfuse.CouplingGraph()
m = g.add_mesh("m", "generated:line:4")
r = g.add_region("r", m, dimension=1)
a = g.add_component("a", ranks=pfuse.RankSet(0, 1), interval="1/2")
b = g.add_component("b", ranks=pfuse.RankSet(0, 1), interval="1/2")
a_out = g.add_field(a, "a_out", r, units=pfuse.units.K, intent=pfuse.Intent.Provides)
a_in = g.add_field(a, "a_in", r, units=pfuse.units.K, intent=pfuse.Intent.Requires)
b_out = g.add_field(b, "b_out", r, units=pfuse.units.K, intent=pfuse.Intent.Provides)
b_in = g.add_field(b, "b_in", r, units=pfuse.units.K, intent=pfuse.Intent.Requires)
ab = g.add_channel("ab", a_out, b_in)
ba = g.add_channel("ba", b_out, a_in)
g.channel(ba).lag(1).allow_time_level_mismatch("resolved by the iteration region")
crit = pfuse.Converged.Absolute(1e-10, "a_out") & pfuse.Converged.Absolute(1e-10, "b_out")
g.iterate([ab, ba]).name("loop").until(crit).max_iterations(50)

session = pfuse.Session(g)
first = Trace("a", gain=0.5)
session.add_component(first)
session.add_component(Trace("b", gain=0.5))
session.initialize()
session.run_period()

print("\n".join(first.log[:7]))
print("...")
it = session.iteration_stats[0]
print("sweeps:", it.iterations, "converged:", it.converged)
print("residuals:", ["%.1e" % x for x in it.residual_history[:4]], "...")
print("checkpoint verifications:", session.stats.checkpoint_verifications)
```

Expected output (exact digits can vary by platform):

```text
save_state
advance t=[0, 1/2] dt=1/2 step=0 iteration=0
restore_state
advance t=[0, 1/2] dt=1/2 step=0 iteration=1
restore_state
advance t=[0, 1/2] dt=1/2 step=0 iteration=2
restore_state
...
sweeps: 36 converged: True
residuals: ['2.8e+00', '1.4e+00', '7.1e-01', '3.5e-01'] ...
checkpoint verifications: 70
```

### Iteration regions and rollback

Each sweep of an iteration region re-advances every member over the same interval, so a member
must be able to go back to the state it had before the sweep. The runtime calls `save_state()`
once at the start of the region's period and `restore_state()` before every re-advance, as the
trace above shows. A component that defines both methods is treated as rollback-capable
automatically; a component in an iteration region that lacks them is refused at `initialize()`
with `E-ITER-NOROLLBACK`.

`save_state`/`restore_state` must cover **everything** `advance` changes: the bound `Provides`
arrays, and any private state such as an internal time, a history buffer or a hidden model state.
After each `restore_state` the runtime compares the registered `Provides` storage against its own
checkpoint and counts the comparison in `RunStats.checkpoint_verifications`; private state is not
covered by that check, so it is your responsibility.

### Capabilities

A `CAPABILITIES` integer attribute (class or instance) declares what the component supports; the
runtime reads it at `add_component()`:

| Constant | Meaning |
|---|---|
| `pfuse.CAP_ROLLBACK` | has `save_state`/`restore_state`; added automatically when both methods exist |
| `pfuse.CAP_HIDDEN_STATE` | `advance` changes state that is not bound through `register_fields`; declare it so the record of the run says so, and make `save_state`/`restore_state` include that state |
| `pfuse.CAP_RESIDUAL`, `pfuse.CAP_LINEARIZATION` | cannot be claimed from Python: `add_component()` refuses with `InvalidArgument` |

Combine with `|`: `CAPABILITIES = pfuse.CAP_HIDDEN_STATE | pfuse.CAP_ROLLBACK`.

### Errors inside a component

An exception raised in `register_fields`, `advance`, `save_state` or `restore_state` unwinds the
runtime cleanly and reaches the caller of `initialize()`, `run_period()` or `run()` **verbatim**,
as the same Python exception object. Raise whatever fits (`RuntimeError`, your own class); do not
try to return error codes. The GIL is released while the runtime executes the plan and reacquired
for every callback, so a component may call back into `pfuse` or any other Python code while it
advances.

## Inspect what FUSE built

### Diagnostics, dataflow and plan

`pfuse.lower(g)` compiles a validated graph into the operator dataflow. The result has `ok`,
`diagnostics` and `dag`; `pfuse.dump_text(dag)` prints it and `pfuse.dump_graphviz(dag)` emits a
DOT graph for `dot`:

```python
import subprocess
import pfuse

# g is the tour graph from "Author a graph in Python"
result = pfuse.lower(g)
if not result.ok:
    raise SystemExit(result.diagnostics.to_string())

text = pfuse.dump_text(result.dag)
print(text.splitlines()[0])                 # "# fuse-ir v1"
print(len(text.splitlines()), "lines")
print([line for line in text.splitlines() if "Remap" in line][0])

with open("interface.dot", "w") as f:
    f.write(pfuse.dump_graphviz(result.dag))
subprocess.run(["dot", "-Tsvg", "interface.dot", "-o", "interface.svg"], check=True)
```

```text
# fuse-ir v1
23 lines
node 0   BuildRemapHalo              -> (none)  "flux/build_remap_halo"
```

Each line of the text dump is one operator (`Source`, `Sample`, `Accumulate`, `Transport`,
`Epoch`, `Remap`, `Constrain`, `Diagnose`, `Relax`, `Sink`, ...) with its typed input and output
values, so you can see every conversion, remap and check that was derived from your
declarations. Diffing two dumps shows exactly what a change to a channel did.

After `initialize()`, `session.plan` is the `ExecutionPlan`: `period` (a `Rational`, the
coupling period), `lockstep` (`True` when every component steps once per period), `region_count`,
and `to_string()`, the flat instruction list with its buffers:

```python
plan = session.plan
print(plan.period, plan.lockstep, plan.region_count)
print(plan.to_string())
```

```text
1/4 True 1
# fuse-plan v1
period=1/4 lockstep=yes instructions=21 buffers=19 coupler_elements=336 regions=1
buffer 0   %0  elements=32  components=1  region=r0  owner=component  persistent=no  "fluid.heat_flux"
...
  4  t=1/4  Transport  node=4  in=[%0]  out=%2  {0->2}  enter=0
  5  t=1/4  Epoch  node=5  in=[%2]  out=%3  {reason=after-transport}
  6  t=1/4  Remap  node=6  in=[%3]  out=%4  {remap=CM, method=mono3}
...
```

### Run statistics

`session.stats` is a `RunStats`, cumulative over every `run`/`run_period` call:

| Field | Meaning |
|---|---|
| `periods` | coupling periods executed |
| `instructions_executed` | plan instructions executed |
| `conservation_checks`, `conservation_violations` | checks made on `.conservative()` channels and how many exceeded `RunnerConfig.conservation_tolerance` |
| `worst_conservation_error` | the largest conservation error seen; `0.0` both when every check was exact and when no check ran |
| `total_iterations` | sweeps of iteration-region bodies, summed over regions and periods |
| `windows_not_converged` | region-periods that hit `max_iterations` without meeting the criterion; non-zero means some answer in the run is not converged |
| `remap_weight_builds`, `remap_weight_nonzeros` | conservative remap matrices built at `initialize()` and their total non-zeros; `0` when every remap was a proved identity between coincident regions |
| `checkpoint_verifications` | comparisons of a member's registered `Provides` storage against the runtime's checkpoint after `restore_state()` |
| `epochs_reconciled` | rank-agreement points; `0` in a one-process run |

`session.iteration_stats` is a list with one `IterationStats` per iteration region, in plan
order, describing the **most recent period** only: `iterations` (sweeps), `converged`, and
`residual_history`, one entry per sweep holding the 2-norm of the whole concatenated coupling
residual `x_new - x_old`. Both are plain Python values, so turning them into checks or plots is
ordinary code:

```python
it = session.iteration_stats[0]
assert it.converged and session.stats.windows_not_converged == 0
history = list(it.residual_history)                             # a plain list of floats
rate = history[-1] / history[-2]                                # last contraction factor
print("%d sweeps, final contraction %.3f" % (len(history), rate))

import matplotlib.pyplot as plt
plt.semilogy(history)
plt.xlabel("sweep")
plt.ylabel("coupling residual")
plt.savefig("residuals.png")
```

For the Aitken case of the sweep below this prints `46 sweeps, final contraction 0.704` (exact
digits can vary by platform).

### Run the tour graph

The components below drive the graph from [Author a graph in Python](#author-a-graph-in-python).
They bind every field of their component by name, so the same base class serves both sides:

```python
import numpy as np
import pfuse


class Slab(pfuse.Component):
    """Binds every field of its graph component by name and keeps them in a dict."""

    def __init__(self, name, cells):
        super().__init__(name)
        self.cells = cells
        self.arrays = {}
        self.saved = {}

    def register_fields(self, registry, graph):
        cid = graph.find_component(self.name)
        for fid in graph.component(cid).fields:
            spec = graph.field(fid)
            shape = (self.cells, spec.components) if spec.components > 1 else (self.cells,)
            self.arrays[spec.name] = np.zeros(shape)
            registry.bind_numpy(fid, self.arrays[spec.name])

    def save_state(self):
        self.saved = {k: v.copy() for k, v in self.arrays.items()}

    def restore_state(self):
        for k, v in self.saved.items():
            self.arrays[k][:] = v


class Fluid(Slab):
    def advance(self, ctx):
        a = self.arrays
        a["heat_flux"][:] = 10.0 * (300.0 - a["T_in"])


class Solid(Slab):
    def advance(self, ctx):
        a = self.arrays
        a["T_wall"][:] = 280.0 + 0.5 * a["q_wall"]


session = pfuse.Session(g)
session.add_component(Fluid("fluid", 32))
session.add_component(Solid("solid", 16))
session.initialize()
session.run(4)
print(session.stats)
print("remap weight builds:", session.stats.remap_weight_builds,
      "nonzeros:", session.stats.remap_weight_nonzeros)
it = session.iteration_stats[0]
print("last period:", it.iterations, "sweeps, converged =", it.converged)
```

Expected output (exact digits can vary by platform):

```text
RunStats(periods=4, total_iterations=7)
remap weight builds: 2 nonzeros: 64
last period: 1 sweeps, converged = True
```

The two remap matrices (32 to 16 cells and back) were built once at `initialize()`; the
stationary problem converges in one sweep after the first period.

## Interchange with other front ends

A graph is the same object whichever front end authored it. `pfuse` reads and writes both
serialised forms:

```python
import copy
import pfuse

# g is the tour graph from "Author a graph in Python"

# YAML, both ways
text = g.to_yaml()
restored = pfuse.from_yaml(text)
print(pfuse.structurally_equal(g, restored))                 # True
with open("interface.yaml", "w") as f:
    f.write(text)
print(pfuse.structurally_equal(g, pfuse.load_yaml_file("interface.yaml")))

# The binary descriptor, the form every front end shares
blob = g.to_bytes()
print(type(blob), len(blob) > 0, pfuse.structurally_equal(g, pfuse.from_bytes(blob)))
with open("interface.fuse", "wb") as f:
    f.write(blob)

# Copies are independent
h = copy.copy(g)
h.add_component("probe")
print(g.component_count, h.component_count, pfuse.structurally_equal(g, h))
```

```text
True
True
<class 'bytes'> True True
3 4 False
```

The YAML is the same document `fuse::to_yaml` writes (`version: 1`, then `meshes`, `regions`,
`components` with their fields, `channels`, iteration regions), so a C++ program loads it with
`fuse::load_yaml_file(path)` and a Python one with `pfuse.load_yaml_file(path)`. The binary form
is the descriptor the C++ `fuse::serialize`/`deserialize` pair produces; a graph authored in
Python serialises to the same bytes as the same graph authored in C++ or read from YAML.
Malformed input is refused with a `FuseError` carrying the code: `E-YAML-PARSE` for YAML that does
not parse, `E-DESC-MAGIC` for bytes that are not a descriptor.

### Author a Newton region, run it from C++

Python can declare, serialise and read back a Newton-solved region, but a `Session` cannot run
one. A member of a Newton region must answer the solver's residual and, with the block-Jacobi
preconditioner, linearisation calls, which only a native `fuse::Component` with an
`ImplicitComponent` can do; `Session.add_component` accepts only `pfuse.Component` instances. So
the division of labour is: author in Python, run in C++.

```python
import numpy as np
import pfuse

g = pfuse.CouplingGraph()
m1 = g.add_mesh("m1", "generated:line:8")
m2 = g.add_mesh("m2", "generated:line:8")
r1 = g.add_region("r1", m1, dimension=1)
r2 = g.add_region("r2", m2, dimension=1)
a = g.add_component("a", ranks=pfuse.RankSet(0, 1))
b = g.add_component("b", ranks=pfuse.RankSet(0, 1))
out = g.add_field(a, "out", r1, units=pfuse.units.K, intent=pfuse.Intent.Provides)
inp = g.add_field(b, "inp", r2, units=pfuse.units.K, intent=pfuse.Intent.Requires)
link = g.add_channel("link", out, inp)

cfg = pfuse.NewtonConfig(rtol=1e-9, max_nonlinear_iterations=30,
                         line_search_kind=pfuse.LineSearchKind.BT,
                         forcing_choice=pfuse.EisenstatWalkerChoice.Choice2,
                         substrate=pfuse.NewtonSubstrate.Native,
                         residual_path=pfuse.NewtonResidualPath.Advance)
region = g.iterate([link]).name("newton").until(pfuse.Converged.Relative(1e-8, "inp")).newton(cfg)

decl = g.iteration_region(region.id())
print(decl.acceleration.kind, decl.requires_residual_capability,
      decl.requires_linearization_capability, decl.newton_config.rtol)

with open("newton.yaml", "w") as f:
    f.write(g.to_yaml())                      # hand this to a C++ driver
with open("newton.fuse", "wb") as f:
    f.write(g.to_bytes())

restored = pfuse.from_yaml(g.to_yaml())
print(pfuse.structurally_equal(g, restored),
      restored.iteration_region(region.id()).newton_config.max_nonlinear_iterations)


class Plain(pfuse.Component):
    def __init__(self, name):
        super().__init__(name)
        self.data = np.zeros(8)
        self.saved = np.zeros(8)

    def register_fields(self, registry, graph):
        for fid in graph.component(graph.find_component(self.name)).fields:
            registry.bind_numpy(fid, self.data)

    def advance(self, ctx):
        pass

    def save_state(self):
        self.saved[:] = self.data

    def restore_state(self):
        self.data[:] = self.saved


session = pfuse.Session(g)
session.add_component(Plain("a"))
session.add_component(Plain("b"))
try:
    session.initialize()
except pfuse.FuseError as e:
    print(e.code)                              # E-NEWTON-CAPABILITY


class Claims(Plain):
    CAPABILITIES = pfuse.CAP_RESIDUAL


try:
    pfuse.Session(g).add_component(Claims("a"))
except pfuse.InvalidArgument as e:
    print(type(e).__name__, "kCapabilityResidual" in e.message)
```

```text
AccelKind.Newton True False 1e-09
True 30
E-NEWTON-CAPABILITY
InvalidArgument True
```

The two refusals are distinct on purpose. A Python component that declares nothing is honest
about lacking the residual capability, and the runtime's own capability check refuses the region
at `initialize()` with `E-NEWTON-CAPABILITY` (its message, written for C++ authors, says to
override `Component::implicit()`; from Python the answer is to run the region from C++). A Python
component that *claims* `CAP_RESIDUAL` or `CAP_LINEARIZATION` is refused immediately at
`add_component()`, because nothing behind it could answer the calls. Python components may be
members of regions accelerated with any other `AccelKind`.

On the C++ side, load the file and run with native components, as [](dn-heat.md) does for its
Newton rows: `auto g = fuse::load_yaml_file("newton.yaml");` and then the usual
`validate`/`lower`/`schedule`/`Runner` sequence of [](quickstart.md).

## Run and drive experiments

### RunnerConfig

`Session.initialize(config=None)` takes a `RunnerConfig`:

| Field | Default | Meaning |
|---|---|---|
| `transport` | `"copy"` | how arcs move between components; `"copy"` is the only transport available in `pfuse` |
| `check_conservation` | `True` | evaluate the checks `.conservative()` channels requested |
| `conservation_tolerance` | `1e-12` | relative error above which a check counts as a violation |
| `abort_on_violation` | `False` | fail the period on a violation instead of counting it |
| `abort_on_nonconvergence` | `False` | fail the period with `E-RT-NOCONVERGE` when an iteration region reaches `max_iterations` unconverged, instead of counting it in `windows_not_converged` and continuing |
| `force_distributed_nodal_remap` | `False` | take the distributed nodal-remap code path even on one rank; a testing switch, not a performance setting |

The fields are readable and writable attributes as well as keyword arguments:
`pfuse.RunnerConfig(abort_on_nonconvergence=True)`.

### Period by period

`run(periods)` executes that many coupling periods; `run_period()` executes one. Between calls
every component's arrays and the statistics are yours to read, so a driver can stop early, adapt,
or log:

```python
session.initialize(pfuse.RunnerConfig(abort_on_nonconvergence=True))
for _ in range(5):
    session.run_period()
    it = session.iteration_stats[0]
    t = session.plan.period * pfuse.Rational(session.stats.periods)
    print("period", session.stats.periods, "t =", t, "sweeps", it.iterations,
          "last residual %.2e" % it.residual_history[-1])
```

With the Dirichlet-Neumann graph of the next section and Aitken acceleration this prints
(exact digits can vary by platform):

```text
period 1 t = 1/20 sweeps 56 last residual 4.25e-06
period 2 t = 1/10 sweeps 48 last residual 1.79e-06
period 3 t = 3/20 sweeps 45 last residual 4.23e-06
period 4 t = 1/5 sweeps 45 last residual 2.72e-06
period 5 t = 1/4 sweeps 41 last residual 4.00e-06
```

A `Session` runs one graph with one set of components for its whole life: there is no reset. To
run again, build a new `Session`. Calling `run` or `run_period` before `initialize()`, or
`initialize()` twice, raises `InvalidArgument`.

### A parameter sweep: the Dirichlet-Neumann accelerators

The complete program below authors the [](dn-heat.md) problem in Python (two heat-conducting
slabs whose interface temperature and flux are iterated to a fixed point each coupling period),
runs it once per accelerator and prints the comparison table. Only the two slab components are
Python; the graph, accelerators, scheduler and runtime are the same C++ code the example binary
runs, and the sweep counts agree with that binary's.

```python
"""Dirichlet-Neumann coupling of two heat-conducting slabs, swept over accelerators."""
import math
import numpy as np
import pfuse

N = 32                       # interface stations
H = pfuse.Rational(1, 20)    # coupling period
TOL = 1e-8
RATIO = np.repeat([0.50, 0.75, 0.90, 0.96], N // 4)   # k1*L2/(k2*L1) per station


def T_outer_1(t):
    return 340.0 + 40.0 * math.sin(2.0 * math.pi * t)


def T_outer_2(t):
    return 280.0 + 15.0 * math.sin(math.pi * t + 0.7)


class Slab(pfuse.Component):
    """Shared plumbing: bind Provides to `out`, Requires to `inp`, checkpoint `out`."""

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
    """Given the wall temperature, returns the wall heat flux of slab 1."""

    def advance(self, ctx):
        self.out[:] = RATIO * (T_outer_1(float(ctx.t_end)) - self.inp)


class Neumann(Slab):
    """Given the wall heat flux, returns the wall temperature of slab 2."""

    def advance(self, ctx):
        self.out[:] = T_outer_2(float(ctx.t_end)) + self.inp


def build_graph(accel, max_iterations):
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
    crit = pfuse.Converged.Relative(TOL, "u_wall") & pfuse.Converged.Relative(TOL, "q_wall")
    (g.iterate([flux, temp]).name("dirichlet_neumann")
       .until(crit).max_iterations(max_iterations).accelerate(accel))
    return g


def run_case(accel, max_iterations=1500, periods=20):
    session = pfuse.Session(build_graph(accel, max_iterations))
    session.add_component(Dirichlet("dirichlet"))
    neumann = Neumann("neumann", initial=T_outer_2(0.0))
    session.add_component(neumann)
    session.initialize()
    session.run(periods)
    t = float(H * pfuse.Rational(periods))
    exact_u = (T_outer_2(t) + RATIO * T_outer_1(t)) / (1.0 + RATIO)
    return session.stats, session.iteration_stats[0], np.abs(neumann.out - exact_u).max()


CASES = {
    "explicit": (pfuse.AccelConfig(kind=pfuse.AccelKind.None_), 1),
    "picard":   (pfuse.AccelConfig(kind=pfuse.AccelKind.None_), 1500),
    "aitken":   (pfuse.AccelConfig(kind=pfuse.AccelKind.Aitken, initial_relaxation=0.5), 1500),
    "iqn-ils":  (pfuse.AccelConfig(kind=pfuse.AccelKind.IQN_ILS, initial_relaxation=0.5,
                                   reuse=8, filter=pfuse.FilterKind.None_), 1500),
    "iqn+qr2":  (pfuse.AccelConfig(kind=pfuse.AccelKind.IQN_ILS, initial_relaxation=0.5,
                                   reuse=8, filter=pfuse.FilterKind.QR2, limit=1e-2), 1500),
}

print("  %-9s %7s %6s %12s %12s" % ("", "sweeps", "!conv", "last resid", "max err u/K"))
for label, (accel, max_it) in CASES.items():
    stats, last, err = run_case(accel, max_it)
    print("  %-9s %7d %6d %12.3e %12.3e" % (label, stats.total_iterations,
                                             stats.windows_not_converged,
                                             last.residual_history[-1], err))
```

Expected output (exact digits can vary by platform):

```text
             sweeps  !conv   last resid  max err u/K
  explicit       20     20    9.582e+02    2.105e+02
  picard      14918      0    1.102e-05    1.582e-06
  aitken        938      0    4.139e-06    8.240e-07
  iqn-ils       201      0    3.482e-13    5.684e-14
  iqn+qr2      5716      2    1.667e-05    2.886e-06
```

Read it as [](dn-heat.md) reads its own table: explicit coupling never converges within a
period and is 200 K off; Picard converges in 14918 sweeps; Aitken needs 16 times fewer; IQN-ILS
74 times fewer and reaches round-off; the QR2 filter at `1e-2` is too aggressive for this problem
and leaves two windows unconverged. Changing a line in `CASES`, or sweeping `TOL`, `N` or `H` in
an outer loop, is the whole cost of a new experiment.

:::{tip}
`residual_history` describes the last period only. To keep every period's history, call
`run_period()` in a loop and copy `list(session.iteration_stats[0].residual_history)` after each
call.
:::

## Errors

Every failure reaches Python as an exception; nothing returns a status code.

| Exception | Also a | Raised for |
|---|---|---|
| `pfuse.FuseError` | `RuntimeError` | the common base; raised bare for every library `Result` failure, with `.code` set to the `E-...` code |
| `pfuse.InvariantViolation` | | an internal invariant failed: a library bug, not a usage error |
| `pfuse.InvalidArgument` | `ValueError` | a precondition you can fix: wrong argument type or shape, unknown name, call out of sequence |
| `pfuse.NullPointer` | `InvalidArgument` | a null handle reached the library |
| `pfuse.Refusal` | `ValueError` | an operation the library declines by design |
| `pfuse.NotImplementedFeature` | `NotImplementedError` | a declared but unavailable feature |
| `pfuse.IoError` | `OSError` | a file could not be read or written |

Attributes on every instance: `code` (the `E-...` code of a `Result` failure, or the C++ class
name such as `fuse::InvalidArgument` for a thrown error), `message`, `where` (source location of a
thrown error, else `None`), `rank`, `backtrace`, and `diagnostics` (a `Diagnostics` when
`Session.initialize()` failed in `validate()`, `lower()` or the scheduler, else `None`).

```python
import pfuse

# A Result failure: bare FuseError with a code.
try:
    pfuse.from_bytes(b"not a fuse descriptor")
except pfuse.FuseError as e:
    print(type(e).__name__, e.code, e.diagnostics)

# A thrown C++ error: one of the six subclasses; .code names the C++ class.
try:
    pfuse.CouplingGraph().find_component("nobody")
except pfuse.InvalidArgument as e:
    print(type(e).__name__, e.code, e.where is not None)

# Builtin bases still work for callers that never import pfuse's classes.
try:
    pfuse.Component("")
except ValueError as e:
    print("ValueError caught:", type(e).__name__)
```

```text
FuseError E-DESC-MAGIC None
InvalidArgument fuse::InvalidArgument True
ValueError caught: InvalidArgument
```

A graph that fails at `initialize()` raises a bare `FuseError` whose `code` is `None` and whose
`diagnostics` lists every problem:

```python
try:
    session.initialize()
except pfuse.FuseError as e:
    if e.diagnostics is not None:
        print(e.diagnostics.to_string())      # every defect at once
    else:
        print(e.code, e.message)              # a runtime refusal, for example E-RT-UNBOUND
```

:::{note}
A thrown C++ error also prints its own report (message, location and backtrace) on `stderr` at
the moment it is thrown, before it arrives as a Python exception. That output is not the
exception; catching the exception does not suppress it. `pfuse` does not read `FUSE_ON_ERROR`.
:::

Codes you are likely to meet, and what to do:

| Code | Raised by | Fix |
|---|---|---|
| `E-UNITS` | `validate` | the two ends of a channel have different dimensions |
| `E-UNITS-SCALE` | `validate` | the conversion needs an offset; declare both ends in the same absolute unit |
| `E-CONN-CONSERVATION` | `validate` | the target is `Extensive` and the channel lacks `.conservative()` |
| `E-LAG` | `validate` | a `.lag(n)` longer than the coupling window; shorten it, or make the two clocks match |
| `E-SCHED-CYCLE` | scheduler | a loop with no lag and no iteration region carrying a previous iterate; add `.lag(1)` to one channel, or `.relax(omega)` inside a region that covers the loop |
| `E-SCHED-REGION` | scheduler | an iteration region whose channels sit on different clocks |
| `E-ITER-NOROLLBACK` | `initialize` | a member of an iteration region has no `save_state`/`restore_state` |
| `E-FIELD-REBIND`, `E-FIELD-UNBOUND` | `FieldRegistry` | a field bound twice, or read before it was bound |
| `E-RT-UNBOUND` | `initialize` | a component bound nothing for one of its fields |
| `E-RT-SIZE` | `initialize` | a bound array's length does not match the region's size |
| `E-RT-REGION` | `initialize` | a member of an iteration region would also be advanced outside the region's loop, for example because a region channel carries `.spacetime` or `.extrapolate`; remove them from region channels |
| `E-REMAP-REGION` | `initialize` | a remap between regions the serial backend cannot relate; use `generated:line` regions without entity sets |
| `E-NEWTON-CAPABILITY` | `initialize` | a Newton region with a Python member; run the region from C++ |
| `E-RT-NOCONVERGE` | `run` | with `abort_on_nonconvergence=True`, a region reached `max_iterations` |
| `E-YAML-PARSE`, `E-DESC-MAGIC` | `from_yaml`, `from_bytes` | the input is not a graph |

The full code list is in [](troubleshooting.md).

## Limitations

- A `Session` runs in one process. MPI is not exposed: `RunnerConfig`'s communicator and
  transport-setup fields are not bound, and `"copy"` is the only transport. Rank sets in the
  graph are nominal.
- Newton-solved regions cannot run from Python (`E-NEWTON-CAPABILITY`); they can be authored and
  serialised. `CAP_RESIDUAL` and `CAP_LINEARIZATION` cannot be claimed (`add_component` refuses).
  Python components are always in-process peers; the participant and provider ABIs are not
  exposed.
- Only the serial mesh backend is available, so mesh sources are `generated:line:N` and
  `generated:box:NxMxP`. The native and MOAB backends, partitioning, transfer-map files and remap
  plans are not exposed. Remaps between regions declared with an entity set are refused
  (`E-REMAP-REGION`).
- Restart is not exposed: there is no `write_restart`/`read_restart`, and no HDF5 or LowFive
  support; `FieldSpec.hdf5_path` is not bound.
- `validate()` runs with the default connection policy; `merge`, `connect`, the compatibility
  and transform APIs, and semantic descriptors are not bound. Affine unit conversions cannot be
  requested (`E-UNITS-SCALE`).
- `schedule()` is not callable on its own; the plan is reached through `Session.initialize()`.
- Field arrays live in host memory; the device memory space, the linear-algebra and numerics
  namespaces, coupling-window runtime internals, `SurrogateContract` and the multiscale
  declarations are not bound.
- The GIL is released while the runtime executes and reacquired for each callback; components
  run on the calling thread, one at a time.
- Reproducibility: the automated tests reproduce the C++ `dn_heat` non-Newton rows from Python
  components digit for digit, but bit-identical agreement between a Python and a C++ component
  is a property of your arithmetic (evaluation order, library functions), not a guarantee of the
  binding.

## Maturity

Covered by the automated test suite (the `python` `ctest` label), including a reproduction of the
`dn_heat` example's non-Newton rows from a Python-authored graph and a second coupled problem
checked against its closed-form fixed point.

## See also

- [](surrogates.md) for using a learned model as a component through `pfuse`.
- [](build.md) for the build options and [](getting-started.md) for the C++ first run.
- [](concepts.md) for the model this page assumes and [](fluent-api.md) for the C++ authoring
  chains the Python calls mirror.
- [](dn-heat.md) for the example the sweep above reproduces and [](convergence.md) for the
  measures.
- [](troubleshooting.md) for every error code.
- [](../dev/python-surface.md) for the binding's design.
