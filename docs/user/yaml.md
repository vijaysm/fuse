# YAML reference

The complete description of a coupling can live in one YAML file instead of in C++ code. This page
is for anyone who writes such a file by hand, generates one from code, or loads one from C, Fortran
or Python: it lists every key the parser accepts, every value it rejects, and what the file does and
does not carry. If you only want to see what a coupling looks like, read the
[annotated example](#a-complete-example) and come back to the tables when you need them.

## Load, validate, save

Three functions in `<fuse/model/yaml_io.hpp>` move a `fuse::CouplingGraph` to and from text. Each
returns a `fuse::Result`, which has no implicit conversion to its value: test `.ok()`, then read
`.value()` or `.error()`.

```cpp
#include <fuse/model/validate.hpp>
#include <fuse/model/yaml_io.hpp>

const auto loaded = fuse::load_yaml_file("coupling.yaml");   // or fuse::from_yaml(text)
if (!loaded.ok()) {
  std::cerr << "[" << loaded.error().code << "] " << loaded.error().message << "\n";
  return 1;
}
const fuse::CouplingGraph& g = loaded.value();

const auto diags = fuse::validate(g);        // loading does not validate
if (diags.has_errors()) { std::cerr << diags.to_string(); return 1; }

const auto text = fuse::to_yaml(g);          // Result<std::string>
if (text.ok()) std::cout << text.value();
```

| Function | Signature | Fails with |
|---|---|---|
| `fuse::to_yaml` | `Result<std::string> to_yaml(const CouplingGraph&)` | `E-YAML-REF`, `E-YAML-EMIT` |
| `fuse::from_yaml` | `Result<CouplingGraph> from_yaml(std::string_view text)` | any `E-YAML-*` except `E-YAML-IO` |
| `fuse::load_yaml_file` | `Result<CouplingGraph> load_yaml_file(const std::string& path)` | `E-YAML-IO`, then everything `from_yaml` reports |

`from_yaml` never throws: a syntax error, a wrongly typed scalar or any other exception inside the
parser comes back as `E-YAML-PARSE`.

:::{warning}
`from_yaml` and `load_yaml_file` check the *document*: the schema version, the key names, the enum
spellings, the names one entry uses to refer to another. They do not check the *graph*. Duplicate
component names, a non-positive `interval`, a channel into a field nobody provides, an illegal
`remap_on` and every other graph-wide rule are reported only by `fuse::validate(g)`. Always call it
after loading. The C ABI runs it for you inside `initialize`, and reports failures there as
`E-ABI-INIT-VALIDATE`.
:::

**From C.** `load_graph` takes a `fuse_graph_source` with exactly one of `yaml_path` or `bytes` set
(both or neither is `E-ABI-LOAD-SOURCE`). A parse failure returns `FUSE_ERR_INVALID`, and
`last_error` carries the `E-YAML-*` code and message.

```c
fuse_graph_source src = {.struct_size = sizeof(src)};
src.yaml_path = "coupling.yaml";
if (fuse->load_graph(ctx, &src, NULL) != FUSE_OK) {
  fuse_error err = {.struct_size = sizeof(err)};
  char buf[512];
  err.buffer = buf;
  err.capacity = sizeof(buf);
  fuse->last_error(ctx, NULL, &err);
  fprintf(stderr, "%s: %s\n", err.code, buf);     /* e.g. E-YAML-VERSION: expected schema version 1 */
}
```

**From Fortran.** The string overload hides the struct:

```fortran
status = fuse_load_graph(ctx, "coupling.yaml", nothing)
```

**From Python.** The same three operations are `pfuse.from_yaml(text)`, `pfuse.load_yaml_file(path)`
and the `CouplingGraph.to_yaml()` method. Failures raise `pfuse.FuseError` with `.code` and
`.message`:

```python
import pfuse

g = pfuse.load_yaml_file("coupling.yaml")
diags = pfuse.validate(g)
text = g.to_yaml()

try:
    pfuse.from_yaml("this: is: not: a: graph")
except pfuse.FuseError as e:
    print(e.code)        # E-YAML-PARSE
```

(a-complete-example)=
## A complete example

Two components on one clock exchange a heat flux conservatively and iterate it to convergence with
IQN-ILS. Every key is explained in the comments; the tables below give the full set.

```yaml
# Every FUSE document starts with the schema version. Only 1 is accepted.
version: 1

# Meshes: where each component's discretisation lives. `source` is the
# backend's mesh descriptor; "generated:..." asks for a built-in mesh.
meshes:
  - {name: mf, source: "generated:box:4x4x4"}
  - {name: ms, source: "generated:box:3x3x3"}

# Regions: the entities of one dimension on one mesh that a field lives on.
# `entity_set` selects a named subset (empty = whole mesh); `dimension`
# is the entity dimension (2 = faces here; -1 = the mesh dimension).
regions:
  - {name: rf, mesh: mf, entity_set: wall, dimension: 2}
  - {name: rs, mesh: ms, entity_set: wall, dimension: 2}

# Components: the solvers. Each has a rank set, a time step (`interval`,
# an exact rational) and the fields it provides or requires. No `launch`
# block means the component runs in this process.
components:
  - name: fluid
    ranks: {first: 0, count: 1}      # ranks 0..0; or a list: [0, 2, 4]
    interval: 1                      # time step, exact: "1", "1/2", "3/1000"; never a decimal
    fields:
      - {name: q, region: rf, disc: FV, components: 1, units: W/m^2,
         conservation: Extensive, intent: Provides}
  - name: solid
    ranks: {first: 1, count: 1}
    interval: 1
    fields:
      - {name: q_in, region: rs, disc: FV, components: 1, units: W/m^2,
         conservation: Extensive, intent: Requires}

# Channels: one directed transfer each, "component.field" at both ends.
channels:
  - name: flux
    source: fluid.q
    target: solid.q_in
    accumulate: {op: Mean}           # how a faster source is reduced over the target step
    remap: {flags: [Conservative, Monotone]}
    relax: 0.7                       # under-relaxation factor (omega)

# Iteration regions: channels solved to a fixed point each coupling step.
iteration_regions:
  - name: outer
    channels: [flux]
    max_iterations: 30
    convergence:                     # one measure per exchanged field
      - {kind: Relative, tolerance: 1e-6, field: q}
    acceleration: {kind: IQN_ILS, initial_relaxation: 0.1, reuse: 8,
                   filter: QR2, limit: 0.01}
```

This program loads the file, validates it, lowers it to the operator DAG and schedules one coupling
period. The `RegionSizes` callback tells the scheduler how many entities each region has on this
rank; a constant is enough to size the plan without a mesh backend.

```cpp
#include <Kokkos_Core.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/model/yaml_io.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/schedule/plan.hpp>

#include <iostream>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  const auto loaded = fuse::load_yaml_file("coupling.yaml");
  if (!loaded.ok()) {
    std::cerr << "[" << loaded.error().code << "] " << loaded.error().message << "\n";
    return 1;
  }
  const fuse::CouplingGraph& g = loaded.value();

  const auto diags = fuse::validate(g);          // from_yaml does not do this
  std::cout << "validate: " << (diags.has_errors() ? "errors" : "ok") << "\n"
            << diags.to_string();
  if (diags.has_errors()) return 1;

  const auto lowered = fuse::lower(g);
  std::cout << "lower: " << (lowered.ok() ? "ok" : "errors") << "\n"
            << lowered.diags.to_string();
  if (!lowered.ok()) return 1;

  const fuse::ClockSet clocks(g);
  const auto sched = fuse::schedule(g, lowered.dag, clocks,
                                    [](fuse::RegionId, int) { return std::size_t{16}; });
  std::cout << "schedule: " << (sched.ok() ? "ok" : "errors") << "\n"
            << sched.diags.to_string()
            << "period = " << sched.plan.period().to_string() << "\n"
            << "instructions = " << sched.plan.instructions().size() << "\n";
  return sched.ok() ? 0 : 1;
}
```

Expected output (exact digits can vary by platform):

```text
validate: ok
lower: ok
schedule: ok
period = 1
instructions = 12
```

Print `fuse::dump_text(lowered.dag)` or `sched.plan.to_string()` to see what the file compiled to
([](cookbook.md#see-what-fuse-built)).

## Key reference

General rules that apply everywhere:

- Every map is checked against an allow-list. A key the parser does not know is `E-YAML-UNKNOWN-KEY`
  (merge entries are the one exception; see [](#merges)).
- Enum values are the C++ enumerator names, spelled exactly and case-sensitively: `Provides`, not
  `provides`; `IQN_ILS`; `cGLL`. The three lowercase families are window kinds, launch modes and the
  `version`/key names themselves.
- Exact times (`interval`, window lengths, unit `scale`/`offset`) are rationals written `N` or `N/D`.
- Entries refer to each other by name, never by position: a region names its mesh, a field its
  region, a channel its two fields as `component.field`.
- Flow style (`{a: 1, b: 2}`) and block style are interchangeable; comments are allowed anywhere.

### Top level

| Key | Type | Required | Notes |
|---|---|---|---|
| `version` | int | yes, must be `1` | anything else, or absent, is `E-YAML-VERSION` |
| `meshes` | sequence | no | |
| `regions` | sequence | no | |
| `components` | sequence | no | physics components |
| `couplers` | sequence | no | remap-only placements |
| `channels` | sequence | no | |
| `merges` | sequence | no | several writers into one field |
| `iteration_regions` | sequence | no | including coupling windows |

There are no top-level `launch`, `connect`, `windows`, `step` or `restart` sections. Launch
information is per component or coupler ([](#launch)); windows are per channel (`accumulate.window`,
`spacetime.window`) or per iteration region (`window_scope`); restart state is not part of the graph.

### Meshes

| Key | Type | Required | Default |
|---|---|---|---|
| `name` | string | yes | |
| `source` | string | no | `""` |

`source` is the mesh backend's descriptor, for example `generated:line:64`, `generated:quad:16x16`,
`generated:box:8x8x8` or a file path ([](geometry.md)). A duplicate `name` is `E-YAML-DUP`.

### Regions

| Key | Type | Required | Default |
|---|---|---|---|
| `name` | string | yes | |
| `mesh` | mesh name | yes | unknown name is `E-YAML-REF` |
| `entity_set` | string | no | `""`, the whole mesh |
| `dimension` | int | no | `-1`, the mesh dimension |

A duplicate `name` is `E-YAML-DUP`.

### Components

| Key | Type | Required | Default |
|---|---|---|---|
| `name` | string | yes | |
| `ranks` | `{first: int, count: int}` or a list of ints | no | empty rank set (`validate()` refuses it) |
| `interval` | rational `"N"` or `"N/D"` | no | `1` |
| `launch` | map, see [](#launch) | no | absent: runs in this process |
| `fields` | sequence of [field](#fields) maps | no | none |

- `ranks: {first: 0, count: 4}` means ranks 0 to 3. The map form accepts no other keys and needs both.
  A list such as `[4, 0, 8]` is sorted and de-duplicated; the emitter writes a contiguous set back in
  the `{first, count}` form.
- `interval` is the component's own time step, exact. There is no separate `step` key.

:::{warning}
Write rationals as `N/D`, never as a decimal. `interval: 0.5` is accepted by the parser but reads
as `0` (the integer prefix of the text), and `validate()` then reports
`E-INTERVAL: component 'a' has non-positive interval 0`. `interval: 1/2` is what you meant.
Exponent notation is worse: `3e-3` reads as `3`, which validates and runs with the wrong step.
Write `3/1000`. Only text that is not a number at all (`abc`) is refused at parse time, with
`E-YAML-PARSE: bad interval for component 'a'`.
:::

Duplicate component names are not caught at parse time; `validate()` reports them as `E-DUP-COMP`.

(launch)=
### The `launch` block

A component or coupler that runs in a separately built executable says how it is started. The
block has one required key, `mode`, and reads the others according to the mode. Keys that do not
belong to the chosen mode are accepted and ignored.

| `mode` | Meaning | Keys read | `validate()` requires |
|---|---|---|---|
| `mpmd` | one `mpiexec` with several `:`-separated programs | `appnum` (int) | `appnum` ≥ 0, else `E-LAUNCH-APPNUM` |
| `connect` | independently launched executables joined over MPI | `is_acceptor` (bool, default `false`), `port_file` (string), `service_name` (string) | `port_file` or `service_name` set, else `E-LAUNCH-CONNECT-TARGET` |
| `plugin` | a shared object FUSE loads into the host | `plugin_path` (string) | `plugin_path` set, else `E-LAUNCH-PLUGIN-PATH` |
| `lowfive` | HDF5 interception between unmodified codes | none | the entry is a coupler with no fields, else `E-LOWFIVE-COMPONENT-KIND` |

There is no spelling for "in process": omit the block. A missing `mode` or an unknown one
(`mode: inprocess`, say) is `E-YAML-PARSE`. See [](#multi-executable-launch-in-yaml) for a
worked example of each mode.

### Couplers

| Key | Type | Required | Default |
|---|---|---|---|
| `name` | string | yes | |
| `ranks` | as for components | no | empty |
| `launch` | as for components | no | in process |
| `index` | uint | no | appended after all physics components |

A coupler is a component that owns no fields and no interval; its only job is to host remapping
(`remap.remap_on` names it) or, in `lowfive` mode, the interception layer. `index` is its position
in the combined component array; a coupler with an `index` is placed at that slot among the physics
components, one without is appended. The emitter always writes `index`, so a round trip preserves
the order.

(fields)=
### Fields

Fields are declared inside the component that owns them.

| Key | Type | Required | Default | Values |
|---|---|---|---|---|
| `name` | string | yes | | unique within the component |
| `region` | region name | yes | | unknown name is `E-YAML-REF` |
| `disc` | enum | no | `FV` | `FV`, `cGLL`, `dGLL`, `PointCloud` |
| `order` | int | no | `0` | polynomial order of the discretisation |
| `components` | int | no | `1` | values per entity: `1` scalar, `3` vector, `9` tensor |
| `units` | string or map | no | dimensionless | see below |
| `conservation` | enum | no | `None` | `None`, `Extensive`, `Intensive` |
| `intent` | enum | no | `Provides` | `Provides`, `Requires`, `Bidirectional` |
| `index` | uint | no | order of appearance | global field order across all components |
| `hdf5_path` | string `"<file>.h5/<dataset>"` | no | unset | LowFive binding ([](lowfive.md)) |

`components` is the only shape key. Element type, counts, memory space and metadata are
registration-time properties of your arrays (`register_fields` in C++, `fuse_field_view` in C),
not of the file.

**Units, scalar form.** A unit expression over these symbols:

```text
1  m  kg  s  A  K  mol  cd  rad  sr  Hz  N  Pa  J  W  C  V  Ohm  F  T  Wb
```

Each symbol may carry an exponent `^N` or `^-N`; terms are joined by `*`, `/` or a space (a space
means multiplication). `1` is dimensionless. Examples that parse: `W/m^2`, `m/s`, `m s^-1`,
`kg/m^3`, `W/m^2/K`, `J*kg^-1*K^-1`. There are no metric prefixes: `km` is
`E-YAML-UNITS: unknown unit symbol 'km'`; write `{length: 1, scale: 1000}` instead.

**Units, map form.** The seven base exponents plus an exact affine part:

```yaml
units: {length: 1, time: -1, scale: 1000/1}        # km/s, as a structural unit
units: {temperature: 1, offset: 5463/20}           # degC: K with an offset of 273.15
```

| Key | Type | Default |
|---|---|---|
| `length`, `mass`, `time`, `current`, `temperature`, `amount`, `luminosity` | int in [-128, 127] | `0` |
| `scale` | rational > 0 | `1` |
| `offset` | rational | `0` |

An exponent outside the range is `E-YAML-UNITS`; a `scale` that is not positive is
`E-YAML-UNITS-SCALE`. Compatible units on the two ends of a channel are converted automatically;
incompatible ones are refused by `validate()` ([](cookbook.md#describe-a-coupling)).

### Channels

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | string | yes | | duplicate is `E-YAML-DUP` |
| `source` | `"component.field"` | yes | | unknown is `E-YAML-REF` |
| `target` | `"component.field"` | yes | | unknown is `E-YAML-REF` |
| `accumulate` | `{op, window}` | no | `{op: Instant, window: coupling_period}` | always emitted |
| `remap` | `{flags, method, remap_on}` | no | none | |
| `spacetime` | `{basis, degree, window}` | no | none | `basis` required when present |
| `relax` | double ω | no | off | presence turns relaxation on |
| `extrapolate` | int order | no | off | presence turns extrapolation on |
| `lag` | int k | no | `0` | read the source k coupling steps back |
| `allow_time_level_mismatch` | string | no | `""` | a non-empty reason suppresses the time-level check |

**`accumulate`.** `op` is one of `Instant`, `Sum`, `Mean`, `Min`, `Max`, `Integral`; `window` is a
[window value](#window-values). Use `Mean` for a state and `Integral` for a flux
([](concepts.md#time)). `Integral` multiplies the sum of the samples by the source component's
interval, so the target field's units must be the source's times time (`E-UNITS` otherwise); `Sum`
does not scale.

**`remap`.**

| Sub-key | Type | Notes |
|---|---|---|
| `flags` | list of `Conservative`, `Monotone`, `HighOrder` | `Conservative` is what `ChannelBuilder::conservative()` sets; there is no separate `conservative:` key |
| `method` | string | overrides the method FUSE would derive from the discretisation pair and conservation kinds, for example `intx`. The parser does not check it; `validate()` refuses a name no weight builder recognises with `E-CHANNEL-REMAP-METHOD` |
| `remap_on` | component or coupler name | where the remap runs ([](placement.md)); unknown is `E-YAML-REF`, an illegal choice is `E-CHANNEL-REMAP-ON` from `validate()` |

There is no "remap kind" key: the kind follows from the two fields' `disc` and `conservation`
unless `method` overrides it.

**`spacetime`.** `basis` is one of `None`, `Constant`, `Linear`, `DG`, `Spline`; `degree` is an int,
default `0`; `window` is a [window value](#window-values). A channel with both `accumulate` and
`spacetime` parses but is refused by `schedule()` with `E-SCHED-SPACETIME`; choose one.

```yaml
channels:
  - name: flux
    source: fluid.heat_flux
    target: solid.q_wall
    accumulate: {op: Mean, window: {kind: from_component, component: fluid}}
    remap: {flags: [Conservative, Monotone], method: intx, remap_on: coupler}
    relax: 0.7
    lag: 1
```

(window-values)=
#### Window values

`accumulate.window` and `spacetime.window` take one of three forms; the default is the coupling
period.

```yaml
window: coupling_period                              # scalar, or {kind: coupling_period}
window: {kind: from_component, component: fluid}     # that component's own step
window: {kind: explicit, length: 2/5}                # an exact length
```

Any other scalar, a map without `kind`, an unknown kind, `from_component` without `component`, or
`explicit` without a rational `length` is `E-YAML-PARSE`; an unknown component name is
`E-YAML-REF`. These are *channel* windows, the reduction interval of one transfer. The coupling
window of an iteration region is [`window_scope`](#window-scope).

(merges)=
### Merges

Several components writing into one field ([](combining-writers.md)).

```yaml
merges:
  - {field: load, component: structure, rule: Sum}
```

| Key | Type | Required | Values |
|---|---|---|---|
| `field` | field name | yes | |
| `component` | the owning component | yes | |
| `rule` | enum | yes | `Sum`, `Mean`, `First` |

A missing key, an undeclared `component.field` or an unknown rule is `E-YAML-MERGE`. Merge entries
are the one map whose unknown keys are not rejected, so check spelling here yourself. The emitter
writes `merges` only when the graph has at least one.

### Iteration regions

| Key | Type | Required | Default |
|---|---|---|---|
| `name` | string | yes | |
| `channels` | list of channel names | no | `[]`; unknown name is `E-YAML-REF` |
| `max_iterations` | int ≥ 1 | no | `1`; smaller is `E-YAML-PARSE` |
| `convergence` | sequence of measures | no | empty, which the runtime refuses with `E-ITER-NOCONVERGE` |
| `acceleration` | map | no | `kind: None` |
| `requires_residual_capability` | bool | no | `false`; forced `true` for `Newton` |
| `requires_linearization_capability` | bool | no | `false`; forced `true` for `Newton` with `use_block_jacobi_preconditioner` |
| `newton_config` | map | no | the Newton defaults below |
| `window_scope` | map | no | absent: an ordinary single-clock region |

**`convergence[]`** entries have exactly three keys, all required:

| Key | Type | Values |
|---|---|---|
| `kind` | enum | `Relative`, `Absolute`, `AbsoluteOrRelative`, `ResidualRelative` |
| `tolerance` | double | finite and > 0 |
| `field` | field name | not resolved at parse time; a wrong name is `E-ITER-MEASURE` at run time |

Measure every field the region exchanges ([](convergence.md)).

**`acceleration`**

| Key | Type | Default | Values |
|---|---|---|---|
| `kind` | enum | `None` | `None`, `Constant`, `Aitken`, `Secant`, `Steffensen`, `IQN_ILS`, `IQN_IMVJ`, `Newton` |
| `initial_relaxation` | double | `0.1` | |
| `omega` | double | `0.5` | |
| `reuse` | int | `8` | |
| `filter` | enum | `QR2` | `None`, `QR1`, `QR2`, `QR3` |
| `limit` | double | `1e-2` | |
| `scale_fields` | bool | `true` | |

:::{note}
A region with no `acceleration` block, or one without `kind`, is a plain Picard iteration
(`kind: None`). This is the region's default and differs from the C++ `fuse::AccelConfig` default
constructor, whose kind is `Constant`. The emitter always writes all seven keys.
:::

**`newton_config`** is read whenever present and written only when `kind` is `Newton`.

| Key | Type | Default | Values |
|---|---|---|---|
| `line_search_kind` | enum | `BT` | `BT`, `CP`, `Basic` |
| `forcing_choice` | enum | `Choice2` | `Constant`, `Choice1`, `Choice2` |
| `forcing_gamma` | double | `0.9` | |
| `forcing_alpha` | double | `2.0` | |
| `eta_initial` | double | `0.3` | |
| `eta_max` | double | `0.9` | |
| `eta_min` | double | `0.0` | |
| `forcing_safeguard_threshold` | double | `0.1` | |
| `forcing_eta_constant` | double | `0.1` | |
| `atol` | double | `1e-50` | |
| `rtol` | double | `1e-8` | |
| `stol` | double | `1e-8` | |
| `divergence_tolerance` | double | `1e4` | |
| `max_nonlinear_iterations` | int | `50` | |
| `max_function_evaluations` | uint | `10000` | |
| `armijo_alpha` | double | `1e-4` | |
| `lambda_min` | double | `1e-12` | |
| `max_backtracks` | int | `40` | |
| `secant_tol` | double | `1e-10` | |
| `max_secant_steps` | int | `40` | |
| `damping` | double | `1.0` | |
| `gmres_restart` | int | `30` | |
| `gmres_max_iterations` | uint | `1000` | |
| `preconditioner_lag` | int | `1` | |
| `use_block_jacobi_preconditioner` | bool | `false` | |
| `substrate` | enum | `Native` | `Native`, `Petsc` |
| `norm` | enum | `Plain` | `Plain`, `MeasureWeighted` |
| `residual_path` | enum | `Advance` | `Advance`, `Direct` |

`acceleration: {kind: Newton}` alone is a complete declaration: the capability flag is set for you
and every other value takes its default.

(window-scope)=
**`window_scope`: a coupling window.** Components whose steps do not divide one another are
coupled over a window rather than a tick ([](coupling-windows.md)). In YAML a window is an
iteration region with a `window_scope` block.

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `window_component` | component name | one of these two | unset | the window is that component's own step |
| `explicit_interval` | rational `"N/D"` | one of these two | unset | the window length, stated directly |
| `order` | int ≥ 2 | no | `2` | interpolation and extrapolation order |
| `policies` | sequence of `{component, policy}` | no at parse time; one per member at schedule time | `[]` | `policy` is `Fixed` or `Adjustable` |
| `iteration` | enum | no | `Jacobi` | `Jacobi`, `GaussSeidel` |

The parser checks spelling and that each policy entry has both keys; it does not check that a
window length is given, that `window_component` and the policy components exist, or that every
member appears exactly once. Those are schedule-time checks (`validate_coupling_window`).

```yaml
version: 1
meshes:
  - {name: u, source: "generated:quad:16x16"}
regions:
  - {name: u_all, mesh: u, dimension: 2}
components:
  - name: U
    ranks: {first: 0, count: 1}
    interval: 1/500                    # U's own fine step inside the window
    fields:
      - {name: u_out, region: u_all, units: 1, conservation: Extensive, intent: Provides}
      - {name: u_in,  region: u_all, units: 1, conservation: Intensive, intent: Requires}
  - name: V
    ranks: {first: 0, count: 1}
    interval: 3/1000                   # incommensurate with 1/500
    fields:
      - {name: v_out, region: u_all, units: 1, conservation: Intensive, intent: Provides}
      - {name: v_in,  region: u_all, units: 1, conservation: Extensive, intent: Requires}
channels:
  - {name: u2v, source: U.u_out, target: V.v_in, remap: {flags: [Conservative]}}
  - {name: v2u, source: V.v_out, target: U.u_in, remap: {flags: [Conservative]}}
iteration_regions:
  - name: brusselator_window
    channels: [u2v, v2u]
    max_iterations: 60
    convergence:
      - {kind: Relative, tolerance: 1e-11, field: u_out}
      - {kind: Relative, tolerance: 1e-11, field: v_out}
    acceleration: {kind: None}
    window_scope:                      # this makes the region a coupling window
      explicit_interval: 1/50          # window length; or window_component: U
      order: 2
      policies:
        - {component: U, policy: Adjustable}
        - {component: V, policy: Adjustable}
      iteration: Jacobi
```

Run through the program above, this document validates, lowers and schedules with
`period = 1/50`: the window length, not a common multiple of the two steps. The multi-executable
form of the same coupling is `examples/brusselator/participants/reference_mpmd_3.yaml`
([](brusselator.md)).

(multi-executable-launch-in-yaml)=
## Multi-executable launch in YAML

Each separately built executable loads the *same* file and finds its own component by name; the
component's `launch` block tells `setup` how the executables find each other
([](multiple-executables.md)). A component without a `launch` block runs in the process that
loaded the file. All four examples below validate and schedule.

**`mpmd`: one `mpiexec`, several programs.** `appnum` is the position of the program in the
`mpiexec` command line, the `MPI_APPNUM` the executable is started as.

```yaml
components:
  - name: fluid
    ranks: {first: 0, count: 2}
    launch: {mode: mpmd, appnum: 0}
    fields:
      - {name: q, region: r, units: W/m^2, conservation: Extensive, intent: Provides}
  - name: solid
    ranks: {first: 2, count: 1}
    launch: {mode: mpmd, appnum: 1}
    fields:
      - {name: q_in, region: r, units: W/m^2, conservation: Extensive, intent: Requires}
couplers:
  - name: coupler
    ranks: {first: 3, count: 1}
    launch: {mode: mpmd, appnum: 2}
channels:
  - name: flux
    source: fluid.q
    target: solid.q_in
    remap: {flags: [Conservative], remap_on: coupler}
```

**`connect`: independent launches joined over MPI.** Exactly one side is the acceptor; both name
the same rendezvous, either a `port_file` the acceptor writes or a `service_name` published through
the MPI name service.

```yaml
components:
  - name: fluid
    ranks: {first: 0, count: 2}
    launch: {mode: connect, is_acceptor: true, port_file: coupling.port}
    fields: [...]
  - name: solid
    ranks: {first: 2, count: 1}
    launch: {mode: connect, is_acceptor: false, port_file: coupling.port}
    fields: [...]
```

**`plugin`: a provider FUSE loads into the host.** The host process loads the shared object named
by `plugin_path` and drives it through the provider ABI ([](providers.md)).

```yaml
components:
  - name: fluid
    ranks: {first: 0, count: 2}
    launch: {mode: plugin, plugin_path: libfluid_provider.so}
    fields:
      - {name: q, region: r, units: W/m^2, conservation: Extensive, intent: Provides}
  - name: solid                        # the host's own component: no launch block
    ranks: {first: 2, count: 1}
    fields:
      - {name: q_in, region: r, units: W/m^2, conservation: Extensive, intent: Requires}
```

**`lowfive`: unmodified HDF5 codes.** The two codes keep writing and reading their files; each
bound field names its dataset with `hdf5_path`, and a *coupler* entry with `mode: lowfive` marks the
interception layer ([](lowfive.md)). The physics components have no `launch` block of their own,
and the `lowfive` entry must own no fields.

```yaml
components:
  - name: fluid                        # an unmodified HDF5 writer: no launch block
    ranks: {first: 0, count: 1}
    fields:
      - {name: q, region: r, units: W/m^2, conservation: Extensive, intent: Provides,
         hdf5_path: fluid.h5/wall/q}   # the dataset the code writes
  - name: solid
    ranks: {first: 1, count: 1}
    fields:
      - {name: q_in, region: r, units: W/m^2, conservation: Extensive, intent: Requires,
         hdf5_path: solid.h5/wall/q_in}
couplers:
  - name: lowfive                      # the interception layer, a bare placement
    ranks: {first: 0, count: 2}
    launch: {mode: lowfive}
channels:
  - name: flux
    source: fluid.q
    target: solid.q_in
    remap: {flags: [Conservative]}
```

An `hdf5_path` without any `lowfive` entry in the graph is `E-LOWFIVE-NO-COMPONENT` from
`validate()`; two fields on the same exact path are `E-LOWFIVE-DUPPATH`.

## Round trips: what the file keeps

`from_yaml(to_yaml(g))` produces a graph whose canonical descriptor bytes equal those of `g`. The
structure is preserved exactly; the text is not.

**Preserved**

- Every mesh, region, component, coupler, field, channel, merge and iteration region, with all
  declared options, in the original order. Field order across components survives through the
  field `index`; a coupler's position through its `index`.
- Launch modes and their keys; `hdf5_path`.
- Units, whatever they were. A unit the emitter has a symbol for is written as that symbol; any
  other is written structurally, as the map form, and reads back as the same unit.
- Convergence measures, acceleration settings and (for `Newton`) the complete `newton_config`.
- `window_scope` with all five keys.

**Normalised**

- Comments, flow versus block style, indentation and key order are gone. The emitter writes keys in
  a fixed order (`version`, `meshes`, `regions`, `components`, `couplers`, `channels`, `merges` only
  if any, `iteration_regions`), and inside a field always `name`, `index`, `region`, `disc`,
  `order`, `components`, `units`, `conservation`, `intent`.
- Defaults you omitted become explicit: every field gets all of its keys, every channel an
  `accumulate` block, every acceleration block all seven keys, every coupler an `index`.
- A rank list that is contiguous is written as `{first, count}`.
- A unit spelled `m/s` or `Hz` comes back as the map form (`{length: 1, time: -1}`); only the ten
  symbols `1 m kg s K N J W Pa W/m^2` are written as text.
- Doubles are written at full precision: `1e-12` becomes `9.9999999999999994e-12`.
- `newton_config` is written only for a `Newton` region; a hand-edited block on a non-Newton region
  does not survive, which changes nothing at run time.
- `requires_*_capability: false` is omitted.
- `entity_set: ""` is written explicitly.

**Not in the file at all.** A `ComponentDecl`'s region list, field registration (element type,
memory space, counts), restart and checkpoint state, runtime configuration, and transport choices
beyond `launch` and `hdf5_path`. A graph that fails `validate()` still round-trips: neither
function validates.

### Produce a starting file from code

Build the graph with the fluent API, write it out once, and keep editing the file:

```cpp
fuse::CouplingGraph g;
// ... add_mesh, add_region, add_component, add_field, add_channel ...
const auto text = fuse::to_yaml(g);
if (!text.ok()) { std::cerr << text.error().message << "\n"; return 1; }
std::ofstream("coupling.yaml") << text.value();
```

The result has every default spelled out, which makes it a complete template. For a graph with one
component and one field in `m/s`, for example, the emitter writes:

```yaml
version: 1
meshes:
  - name: m
    source: generated:line:8
regions:
  - name: r
    mesh: m
    entity_set: ""
    dimension: 1
components:
  - name: a
    ranks: {first: 0, count: 1}
    interval: 1
    fields:
      - name: f
        index: 0
        region: r
        disc: FV
        order: 0
        components: 1
        units:
          length: 1
          time: -1
        conservation: None
        intent: Provides
couplers:
  []
channels:
  []
iteration_regions:
  []
```

### Author in one language, load in another

Every front end reads and writes the same document. A common arrangement is to author the graph in
Python or C++ where the fluent API is convenient, write it with `to_yaml`, and have the C or Fortran
participants load the file with `load_graph` ([](c-abi.md), [](fortran.md)). The
`brusselator_graph_tool` in `examples/brusselator` does exactly this: it builds the graph in C++,
emits the reference YAML, and the C participants load it. The binary descriptor
(`fuse::serialize`, `fuse_graph_source::bytes`, `CouplingGraph.to_bytes()`) is the same graph in a
form that needs no parser; use YAML when a person edits the file and the descriptor when a program
hands it to another program.

## Errors

Every failure of `from_yaml` or `load_yaml_file` is one `Failure` with a stable code and a message
that names the offending entry. Through the C ABI the status is `FUSE_ERR_INVALID` and
`last_error` carries the same code; in Python it is the `.code` of the `FuseError`.

| Code | Trigger | Fix |
|---|---|---|
| `E-YAML-PARSE` | YAML syntax error (`yaml-cpp: error at line 1, column 9: illegal map value`); a wrongly typed scalar; a required key missing (`component entry is missing required key 'name'`); an unknown enum spelling (`unknown intent`, `unknown accelerator kind 'aitken'`, `unknown launch mode 'inprocess'`); a bad rational in `interval`, `explicit_interval` or a window `length`; a malformed window value; `max_iterations` < 1; a convergence measure without `tolerance` or `field`, or with a tolerance that is not finite and positive; a `window_scope` policy without `component` | Read the message: it names the entry and the key. For enums copy the spelling from the tables above, including case. For times write `N/D`. |
| `E-YAML-VERSION` | `version` missing or not `1` (`expected schema version 1`) | Put `version: 1` first. |
| `E-YAML-UNKNOWN-KEY` | a key outside the allow-list of any map (`unknown key 'restart' in top-level document`, `unknown key 'step' in component entry`) | Check the key's spelling and the section it belongs in; the component step is `interval`, launch is per component. |
| `E-YAML-REF` | a name that does not resolve: a region's `mesh`, a field's `region`, a channel's `source` or `target`, `remap_on`, a window's `component`, an iteration region's channel (`unknown mesh 'nope'`). From `to_yaml`, a graph holding a dangling id | Declare the referent earlier in the file, in the section the key expects; `source`/`target` are `component.field`. |
| `E-YAML-DUP` | two meshes, regions or channels with one name (`duplicate mesh name 'm'`) | Rename one. Duplicate components, couplers and fields are reported by `validate()` instead. |
| `E-YAML-UNITS` | a unit expression the parser cannot read (`unknown unit symbol 'km'`), or a structural exponent outside [-128, 127] | Use the symbol list above with `^`, `*`, `/`; express prefixes with `scale` in the map form. |
| `E-YAML-UNITS-SCALE` | `scale` ≤ 0 in a unit map (`field 'f' has non-positive unit scale 0`) | Give a positive rational. |
| `E-YAML-MERGE` | a merge entry missing `field`, `component` or `rule`; naming an undeclared `component.field`; or a rule other than `Sum`, `Mean`, `First` | Declare the field on that component first; check the rule spelling. |
| `E-YAML-EMIT` | the emitter failed while writing (`to_yaml` only) | Report it with the graph that produced it; a well-formed graph never triggers this. |
| `E-YAML-IO` | the file cannot be opened or read (`cannot open '/nonexistent.yaml'`) | Check the path and permissions; `load_yaml_file` resolves it relative to the working directory. |

Codes that start `E-ABI-` come from the C ABI around the load: `E-ABI-LOAD-SOURCE` for a
`fuse_graph_source` with neither or both of `yaml_path` and `bytes`, `E-ABI-LOAD-STATE` for a
second load on one context, `E-ABI-INIT-VALIDATE` when `initialize` runs `validate()` and it fails.
Everything else about the loaded graph is in [](troubleshooting.md).

### Names the parser does not resolve

Four strings are stored as written and checked only later, so a typo in them does not produce an
`E-YAML-REF`:

| Key | Checked by | Error |
|---|---|---|
| `convergence[].field` | the runner, when the region is first set up | `E-ITER-MEASURE`, which lists the fields the region does exchange |
| `window_scope.window_component` | `validate_coupling_window` / `schedule()` | a window diagnostic naming the component |
| `window_scope.policies[].component` | `validate_coupling_window` / `schedule()` | a window diagnostic naming the component |
| `remap.method` | the remap builder, when weights are built | `E-REMAP-UNSUPPORTED` (conservative builder), which lists the known methods |

### Case and spelling

- Enum values are case-sensitive enumerator names: `Provides`/`Requires`/`Bidirectional`;
  `None`/`Extensive`/`Intensive`; `FV`/`cGLL`/`dGLL`/`PointCloud`; `Instant`/`Sum`/`Mean`/`Min`/
  `Max`/`Integral`; `Conservative`/`Monotone`/`HighOrder`; `Relative`/`Absolute`/
  `AbsoluteOrRelative`/`ResidualRelative`; `IQN_ILS` with an underscore; `QR2`; `Fixed`/`Adjustable`;
  `Jacobi`/`GaussSeidel`; `BT`/`CP`/`Basic`; `Native`/`Petsc`.
- Lowercase, by contrast: launch modes (`mpmd`, `connect`, `plugin`, `lowfive`) and window kinds
  (`coupling_period`, `from_component`, `explicit`).
- Unit symbols are case-sensitive too: `K` is kelvin, `k` is nothing; `Pa`, `Hz`, `Ohm`, `Wb`.
- Component, field, mesh, region and channel names are free strings and are compared exactly.
- YAML itself treats an unquoted `1/2` or `W/m^2` as a string, so quotes are optional for rationals
  and units. Quote a `source` such as `"generated:box:8x8x8"` if your editor's YAML linter
  complains about the colon; the parser accepts it either way.

## See also

- [](graph-programming.md): building the same graph in C++ and the fluent API
- [](coupling-windows.md): what `window_scope` means and how a window runs
- [](multiple-executables.md): launching the components a `launch` block describes
- [](c-abi.md), [](python.md): loading the file from C, Fortran and Python
- [](cookbook.md): short recipes, including keeping the description in a file
