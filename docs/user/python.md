# Coupling from Python (`pfuse`)

`pfuse` is a `pybind11` binding directly over `fuse::CouplingGraph`, `fuse::Component` and
`fuse::Runner`, the C++ model and runtime, not the participant ABI ([](c-abi.md), [](fortran.md)).
Use it when you want to author or drive a coupling *in* Python, for example a Python physics
component exchanging fields with other components through a real `fuse::Runner`. It is not a way to
call an existing coupled run's step functions from outside the process. The binding's design and
per-file layout are in [](../dev/python-surface.md); this page is the task-oriented version.

## Build it

`pfuse` is built from a FUSE build tree, behind the `FUSE_ENABLE_PYTHON` option (default `OFF`; see
[](build.md)). It requires `pybind11` to be discoverable by CMake's `find_package(pybind11)`:

```bash
cmake -S . -B build -DFUSE_ENABLE_PYTHON=ON \
  -DPython3_EXECUTABLE=/path/to/your/python3 \
  -Dpybind11_DIR="$(/path/to/your/python3 -m pybind11 --cmakedir)"
cmake --build build --target _pfuse

PYTHONPATH=build/python python3 -c "import pfuse; print(pfuse.version())"
```

The `_pfuse` target copies the built extension next to `pfuse/__init__.py` under
`<build>/python/pfuse/`, so `PYTHONPATH=<build>/python` is all a working tree needs; there is no
install step. The extension is built against one interpreter's ABI and imports only under that
interpreter, so use the same `python3` for the build and for `import pfuse`.

With `FUSE_ENABLE_TESTS` on, the Python tests are registered as `ctest` rows labelled `python`
(`ctest --test-dir build -L python`); they need `pytest` in that interpreter.

## A first coupled run

Two components exchanging a flux and a temperature through one Picard-accelerated iteration
region — the same shape every coupled run in `pfuse` has, reduced to the smallest case that
still shows a converging fixed point rather than a single one-way pass:

```python
import numpy as np
import pfuse

class Upstream(pfuse.Component):
    """Produces a flux into the interface; a stand-in for a real solver's boundary term."""

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
    """Relaxes its interface temperature toward the flux it was just handed."""

    def __init__(self):
        super().__init__("downstream")
        self.t = np.zeros(1)
        self.t[0] = 280.0
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

up = g.add_component("upstream", ranks=pfuse.RankSet(0, 1), interval=pfuse.Rational(1, 1))
down = g.add_component("downstream", ranks=pfuse.RankSet(0, 1), interval=pfuse.Rational(1, 1))

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
g.channel(temp_ch).lag(1).allow_time_level_mismatch("resolved within the iteration region")

criterion = pfuse.Converged.Relative(1e-6, "t") & pfuse.Converged.Relative(1e-6, "flux")
g.iterate([flux_ch, temp_ch]).name("interface").until(criterion).max_iterations(200).accelerate(
    pfuse.AccelConfig(kind=pfuse.AccelKind.Aitken, initial_relaxation=0.5))

session = pfuse.Session(g)
session.add_component(Upstream())
session.add_component(Downstream())
session.initialize()
session.run(1)

print(session.stats.total_iterations, session.stats.windows_not_converged == 0)
# Expected output (exact digits can vary by platform): 31 True
```

Every `Component` subclass implements the same four methods `fuse::Component`'s C++ interface
has: `register_fields` (bind each of this component's own fields to a 1-D or 2-D
`float64`, C-contiguous, writeable numpy array — see "Array lifetime" below), `advance` (the one
step/sweep of physics), and `save_state`/`restore_state` (a byte-exact checkpoint pair — see
"Capabilities" below for why both exist even for a component with no Newton ambitions).
`Session` owns a copy of the graph, the field registry, and the underlying `fuse::Runner` for as
long as the `Session` object is alive, mirroring `fuse::Runner`'s own requirement that it be
constructed once against a graph it then holds for its whole life.

## Array lifetime

`FieldRegistry.bind_numpy(field_id, array)` keeps a strong reference to `array` for as long as
the registry (or the owning `Session`) is alive — `reg.bind_numpy(f, np.zeros(4))`, binding a
temporary with no other name, is safe because of this, not despite it. One consequence that is
easy to trip over without reading this: **a bound array can never be resized in place.**
`numpy.ndarray.resize()` refuses whenever the array's reference count is not exactly 1, and the
registry's own kept reference is always one more than that. This is numpy's own allocator
refusing, not a check `pfuse` added — if a component's field count can change, bind a new array
and re-bind, rather than attempting to resize the bound one.

## Capabilities and errors

A `Component` that overrides `save_state`/`restore_state` is treated as rollback-capable
automatically — nothing else to declare. A component that needs to say more (claim a
determinism level, or claim hidden state outside its registered fields) sets a `CAPABILITIES`
class attribute, an `int` built from `pfuse.CAP_ROLLBACK`/`CAP_HIDDEN_STATE`. `pfuse.
CAP_RESIDUAL`/`CAP_LINEARIZATION` are also bound, but only so that *claiming* either from Python
is refused immediately, at `add_component()` time: a Newton-solved region's residual/Jacobian
calls need `fuse::ImplicitComponent`'s C++ hooks, which no Python subclass can provide — see [](../dev/python-surface.md)'s "Components implemented in Python" for the two
distinct refusals this produces (one for a component that is honest about lacking the
capability, one for a component that wrongly claims it).

Every failure crosses into Python as an exception, never a return code: `pfuse.FuseError` is
the common base, with six subclasses mirroring `fuse::Error`'s own C++ hierarchy
(`InvariantViolation`, `InvalidArgument`, `NullPointer`, `Refusal`, `NotImplementedFeature`,
`IoError`) — each also inherits the closest Python builtin, so `except ValueError:` works for a
caller who never imports `pfuse`'s own classes. A `fuse::Result` failure (most library errors)
raises a **bare** `FuseError` with `.code`/`.message` set and no further subclass — a deliberate
flat design; see [](../dev/python-surface.md#errors-two-sources-one-python-hierarchy) for why.
`Session.initialize()` additionally sets `.diagnostics` (a `fuse::Diagnostics`) when the failure
came from `validate()`/`lower()`/`schedule()`, and leaves it `None` otherwise.

## Limitations

- Newton execution is not available for Python components: a Newton-solved region's members must
  be native. A Python-authored graph can still configure and run Newton over native components.
- Python components cannot claim `CAP_RESIDUAL` or `CAP_LINEARIZATION`; `add_component()` refuses.
- MPI is not exposed (`RunnerConfig`'s communicator and transport-setup fields are not bound); a
  `Session` runs in one process with the default `"copy"` transport.
- Only `fuse::SerialMeshBackend` is bound: use `"generated:line:N"` or `"generated:box:NxMxP"` mesh
  sources. The native and MOAB backends are not available.
- `FieldSpec::hdf5_path` and LowFive launch modes are not exposed.
- `validate()` is bound with its default `ConnectPolicy`; a non-default policy cannot be passed.

The reasoning behind each item is in
[](../dev/python-surface.md#what-this-does-not-expose-and-why).

## Maturity

Covered by the automated test suite (the `python` `ctest` label), including a reproduction of the
`dn_heat` example's non-Newton rows from a Python-authored graph.

## See also

- [](build.md) for the build options.
- [](dn-heat.md) for the example the Python tests reproduce.
- [](fluent-api.md) for the C++ authoring chains the Python graph API mirrors.
- [](../dev/python-surface.md) for the binding's design.
