# Building a coupling graph in code

How to describe a coupling as a `fuse::CouplingGraph` from C++: every declaration struct, every
builder call, what each one means physically, and what `validate()` says when they disagree. Read
this after [](quickstart.md) when you write your own graph, generate one from a configuration
file, or want to change what a coupling does without touching the solvers. The YAML form of the
same graph is `fuse::to_yaml()`; the compiled form is covered in [](ir.md).

## A complete graph, validated

The program below declares a fluid and a solid exchanging a heat flux and a wall temperature. The
fluid steps four times per solid step, the flux is remapped on a separate coupler processor set, and
the temperature channel is extrapolated to the fluid's time. It compiles, lowers and schedules the
graph and prints what `validate()` found.

```cpp
#include <Kokkos_Core.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/schedule/plan.hpp>

#include <iostream>

fuse::CouplingGraph build_graph() {
  fuse::CouplingGraph g;

  // Meshes and the regions fields live on.
  const auto fluid_mesh = g.add_mesh({.name = "fluid_mesh", .source = "generated:line:32"});
  const auto solid_mesh = g.add_mesh({.name = "solid_mesh", .source = "generated:line:32"});
  const auto fluid_wall = g.add_region({.name = "fluid_wall", .mesh = fluid_mesh, .dimension = 1});
  const auto solid_wall = g.add_region({.name = "solid_wall", .mesh = solid_mesh, .dimension = 1});

  // Components: name, processor set, time step. A coupler is a processor set only.
  const auto fluid = g.add_component({.name = "fluid", .ranks = fuse::RankSet{0, 2},
                                      .interval = fuse::Rational{1, 4}});
  const auto solid = g.add_component({.name = "solid", .ranks = fuse::RankSet{2, 1},
                                      .interval = fuse::Rational{1}});
  const auto hub = g.add_coupler({.name = "hub", .ranks = fuse::RankSet{3, 1}});

  // Fields: what each component provides and requires.
  const auto q_out = g.add_field(fluid, {.name = "q", .region = fluid_wall,
                                         .units = fuse::units::kHeatFluxDensity,
                                         .conservation = fuse::Conservation::Extensive,
                                         .intent = fuse::Intent::Provides});
  const auto t_in = g.add_field(fluid, {.name = "T_wall", .region = fluid_wall,
                                        .units = fuse::units::kKelvin,
                                        .conservation = fuse::Conservation::Intensive,
                                        .intent = fuse::Intent::Requires});
  const auto q_in = g.add_field(solid, {.name = "q_in", .region = solid_wall,
                                        .units = fuse::units::kHeatFluxDensity,
                                        .conservation = fuse::Conservation::Extensive,
                                        .intent = fuse::Intent::Requires});
  const auto t_out = g.add_field(solid, {.name = "T", .region = solid_wall,
                                         .units = fuse::units::kKelvin,
                                         .conservation = fuse::Conservation::Intensive,
                                         .intent = fuse::Intent::Provides});

  // Channels and their treatments.
  const auto flux = g.add_channel("heat_flux", q_out, q_in);
  g.channel(flux)
      .accumulate(fuse::AccumOp::Mean)   // four fluid steps -> one solid value
      .conservative()                    // the integral of q is preserved
      .remap_on(hub)                     // remap on the coupler's ranks
      .lag(1)                            // breaks the flux/temperature cycle
      .allow_time_level_mismatch("explicit staggered scheme");
  const auto temp = g.add_channel("wall_temperature", t_out, t_in);
  g.channel(temp).extrapolate(1);        // first-order guess at the fluid's time
  return g;
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos_guard(argc, argv);
  const fuse::CouplingGraph graph = build_graph();

  // Nothing calls validate() for you.
  const fuse::Diagnostics diags = fuse::validate(graph);
  std::cout << "validate(): " << diags.size() << " diagnostic(s)\n" << diags.to_string();
  if (diags.has_errors()) return 1;

  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) { std::cerr << r.error().message << "\n"; return 1; }
  }
  const fuse::LoweringResult lowered = fuse::lower(graph);
  if (!lowered.ok()) { std::cerr << lowered.diags.to_string(); return 1; }
  const fuse::ClockSet clocks{graph};
  const fuse::ScheduleResult sched = fuse::schedule(
      graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, backend));
  if (!sched.ok()) { std::cerr << sched.diags.to_string(); return 1; }
  std::cout << "=== IR ===\n" << fuse::dump_text(lowered.dag)
            << "=== PLAN ===\n" << sched.plan.to_string();
  return 0;
}
```

Expected output, before the `=== IR ===` block:

```text
validate(): 1 diagnostic(s)
warning W-MISMATCH-OK: time-level mismatch permitted: explicit staggered scheme [channel 'heat_flux']
```

The one warning is deliberate: it records that the lag on `heat_flux` was authorised. The IR and
plan this program prints are read line by line in [](ir.md).

Link against `fuse::model` for the graph alone, or `fuse::runtime` and `fuse::backend` for the
whole pipeline ([](quickstart.md#includes-linking-and-kokkos)). Every `add_*` call takes a
designated-initialiser struct and returns a strongly typed id; every builder call returns the
builder and never fails. Problems are reported together by `validate()`
([](fluent-api.md#diagnostics-accumulate)).

## The declaration model

A graph holds seven kinds of declaration. Each is appended with one call and referred to afterwards
by the id that call returned. There is no removal and no renaming: build the graph you mean, or
build another one.

| Declaration | Call | Returns | Struct |
|---|---|---|---|
| Mesh | `add_mesh(MeshDecl)` | `MeshId` | `fuse/model/region.hpp` |
| Region | `add_region(Region)` | `RegionId` | `fuse/model/region.hpp` |
| Component | `add_component(ComponentDecl)` | `ComponentId` | `fuse/model/component.hpp` |
| Coupler | `add_coupler(CouplerDecl)` | `ComponentId` | `fuse/model/component.hpp` |
| Field | `add_field(ComponentId owner, FieldSpec)` | `FieldId` | `fuse/model/field_spec.hpp` |
| Channel | `add_channel(name, FieldId source, FieldId target)` | `ChannelId` | `fuse/model/channel.hpp` |
| Merge rule | `merge(FieldId target, MergeOp)` | `CouplingGraph&` | `fuse/model/merge.hpp` |
| Iteration region | `iterate({channels})` or `add_iteration_region(IterationRegionDecl)` | `IterationBuilder` or `IterationRegionIdx` | `fuse/model/coupling_graph.hpp`, `fuse/model/iteration.hpp` |

Ids are `fuse::Id<Tag>` values with `value()` and `valid()`; a default-constructed id is invalid.
`add_field()` asserts that its owner exists; everything else is checked by `validate()`.

### Meshes and regions

```cpp
struct MeshDecl { std::string name; std::string source; };
struct Region   { std::string name; MeshId mesh; std::string entity_set; int dimension = -1; };
```

- `MeshDecl::source` is a file path for a mesh backend to read, or a generated spec. The serial
  backend understands `"generated:line:N"` (the unit interval, N cells) and `"generated:box:NxMxP"`
  (the unit cube). Other backends read files ([](geometry.md)).
- `Region::entity_set` selects a named subset of the mesh; empty means the whole mesh.
  `dimension` is the topological dimension of the coupled entities (1 for the cells of a line,
  2 for faces, 3 for volume cells); `-1` means the mesh's own dimension.

### Components and couplers

```cpp
struct ComponentDecl {
  std::string          name;
  RankSet              ranks;                  // default: rank 0
  Rational             interval{1};            // time step, exact
  std::vector<FieldId> fields;                 // filled by add_field()
  ComponentKind        kind = ComponentKind::Physics;
  std::vector<RegionId> regions;               // unused; leave empty
  LaunchSpec           launch;                 // default: in process
};
struct CouplerDecl { std::string name; RankSet ranks; LaunchSpec launch; };
```

`RankSet` has three forms:

```cpp
fuse::RankSet{};                               // rank 0
fuse::RankSet{4, 8};                           // first rank 4, count 8: ranks 4..11
fuse::RankSet{std::vector<int>{0, 2, 5}};      // any set; sorted and de-duplicated
fuse::RankSet{0, 0};                           // empty: the component is switched off
```

The two-integer form is *first, count*, not *first, last*. A count of zero gives an empty set,
which is legal for a component that has no channels ([](placement.md)). `rank_union()` and
`rank_intersection()` combine sets; `contains()`, `overlaps()`, `same_as()`, `is_contiguous()`
query them.

`interval` is the component's time step as a `fuse::Rational`. Write `fuse::Rational{1, 4000}`,
never a `double`: the scheduler's common period is an exact least common multiple of these values
([](concepts.md#time)).

`add_coupler()` is `add_component()` with `kind = ComponentKind::Coupler`: a processor set that
owns no fields and is never advanced, on which a channel's remap can run
([](placement.md#a-separately-placed-coupler)).

`LaunchSpec` says how the component's executable is started: `LaunchMode::InProcess` (the
default), `Mpmd` (needs `appnum`), `Connect` (needs `port_file` or `service_name`, and
`is_acceptor` on one side), `Plugin` (needs `plugin_path`) or `LowFive`. Leave it alone for a
single executable; [](multiple-executables.md), [](providers.md) and [](lowfive.md) cover the
others.

### Fields

```cpp
struct FieldSpec {
  std::string    name;
  RegionId       region;
  Discretization disc = Discretization::FV;    // FV, ContinuousGLL, DiscontinuousGLL, PointCloud
  int            order = 0;                    // polynomial order for GLL; ignored for FV
  int            components = 1;               // 1 scalar, 3 vector, 9 tensor
  Units          units{};                      // dimensionless unless you say otherwise
  Conservation   conservation = Conservation::None;   // None, Extensive, Intensive
  Intent         intent = Intent::Provides;    // Provides, Requires, Bidirectional
  std::optional<std::string> hdf5_path;        // LowFive-bound fields only
};
```

- `intent`: a `Provides` field is written by its component and may be read by channels; a
  `Requires` field is filled by FUSE before the component's step; `Bidirectional` is both.
- `conservation`: `Extensive` for a quantity whose integral over the region must be preserved by a
  transfer (a flux, a force, a mass); `Intensive` for a pointwise state (a temperature, a
  pressure); `None` when neither applies. Both ends of a channel must agree.
- `units`: SI dimension exponents plus an exact scale and offset. Constants in
  `fuse/base/units.hpp`: `units::kDimensionless`, `kMetre`, `kKilogram`, `kSecond`, `kKelvin`
  (`kTemperature`), `kNewton`, `kJoule`, `kWatt`, `kPascal`, `kHeatFluxDensity` (W/m²). Any
  other unit is a `fuse::Units` value:

```cpp
constexpr fuse::Units kKiloWattPerSquareMetre{.mass = 1, .time = -3,
                                              .scale = fuse::Rational{1000}};
constexpr fuse::Units kMetrePerSecond = fuse::units::kMetre / fuse::units::kSecond;
```

### Channels

`add_channel(name, source, target)` declares one directed transfer from a `Provides` (or
`Bidirectional`) field to a `Requires` (or `Bidirectional`) field of another component. The
channel's *treatment* is set afterwards through `g.channel(id)`, a `ChannelBuilder` whose calls
chain and record, never check:

```cpp
g.channel(id)
    .accumulate(fuse::AccumOp::Mean, fuse::Window::coupling_period())
    .remap(fuse::RemapFlags::Conservative | fuse::RemapFlags::Monotone)
    .remap_on(hub)
    .relax(0.7);
g.channel(id2)
    .spacetime(fuse::TemporalBasis::DG, 1, fuse::Window::from(solid))
    .conservative();
g.channel(id3).lag(2).extrapolate(1).allow_time_level_mismatch("deliberate leapfrog stagger");
```

| Call | What it means physically | Compiles to |
|---|---|---|
| `accumulate(AccumOp op, Window w = coupling_period())` | The source is faster than the target. Record the source at each of its steps and reduce the record over the target's step: `Mean` for a state, `Integral` or `Sum` for a flux or a rate, `Min`/`Max` for a bound. `Instant` (the default) delivers the latest value only. The window is the target's step, another component's step (`Window::from(c)`) or an explicit length (`Window::explicit_length(r)`). | `Sample` + `Accumulate` |
| `spacetime(TemporalBasis basis, int degree, Window w = coupling_period())` | Project the source's trajectory over the window onto a temporal basis (`Constant`, `Linear`, `DG`, `Spline`) of the given degree, so the target sees a time-continuous value instead of one sample. Not combinable with a non-`Instant` `accumulate` on the same channel. | `Sample` + `Interpolate` |
| `extrapolate(int order)` | The target runs ahead of the source's last known value. Extrapolate the source forward in time with a polynomial of this order instead of holding it. Reduces the first-order lag error of an explicit scheme. | `Extrapolate` |
| `lag(int steps)` | The target reads the value from `steps` source steps ago. Breaks a dependence cycle (an explicit, staggered scheme). The value's time level becomes `WindowEnd-steps`. | a time-level shift; `Interpolate{lag}` only when no `Remap` follows |
| `allow_time_level_mismatch(std::string reason)` | Authorises delivering a value whose time level differs from the one the target field expects, and records why. Required with `lag()`; without it lowering refuses the channel with `E-IR-TYPE`. | nothing; `validate()` reports `W-MISMATCH-OK` |
| `remap(RemapFlags flags)` | Demand properties of the spatial transfer: `Conservative` (the integral over the target equals the integral over the source), `Monotone` (no new extrema), `HighOrder`. Flags accumulate across calls. A `Remap` is emitted anyway when the two regions differ. | `Remap` (+ `Constrain`, `Diagnose`) |
| `conservative()` | `remap(RemapFlags::Conservative)`. Needed for every channel into an `Extensive` field. | as above |
| `remap_method(std::string)` | Override the remap method FUSE would derive from the two discretisations and the conservation kind ([](geometry.md), [](point-cloud-remap.md), [](interface-transfer.md)). | `Remap{method}` |
| `remap_on(ComponentId)` | Run the remap on this component's processor set instead of the target's: a coupler, or the channel's own source or target. | `Transport` to and from that set, with an `Epoch` after each |
| `relax(double omega)` | Under-relax the delivered value, `omega` new plus `1 - omega` previous. Inside an iteration region this is what makes the value loop-carried between sweeps. | `Relax` |

Unit conversion is not a builder call. Declare each field's unit and FUSE inserts the exact scale
itself when the dimensions agree and the conversion is a pure ratio (see
[Units](#units-declare-them-and-let-fuse-scale)).

### Several writers into one field

Two channels may target the same field only if the graph says how to combine them:

```cpp
g.merge(q_in, fuse::MergeOp::Sum);     // Sum, Mean or First
```

Every channel that targets the field takes part, in declaration order; the merged value carries the
newest writer's time level. [](combining-writers.md) explains the choice of operator and the
`E-MULTI-WRITER` / `W-MERGE-UNUSED` diagnostics.

### Iteration regions

`iterate()` wraps a set of channels into an implicit coupling group: each step, the group is swept
until every exchanged field has converged ([](concepts.md#iteration)).

```cpp
const fuse::ConvergenceCriterion crit =
    fuse::Converged::Relative(1e-6, "q") & fuse::Converged::Relative(1e-6, "T");
g.iterate({flux, temp})
    .name("interface")                 // default: "iterate<k>"
    .until(crit)                       // or .until(fuse::Converged::Relative(1e-6, "q"))
    .max_iterations(50)
    .accelerate(fuse::AccelConfig{.kind = fuse::AccelKind::IQN_ILS,
                                  .initial_relaxation = 0.5,
                                  .filter = fuse::FilterKind::None});
```

`IterationBuilder` has `until(ConvergenceCriterion)`, `until(ConvergenceMeasure)`,
`max_iterations(int)`, `accelerate(AccelConfig)`, `initial_relaxation(double)`, `name(std::string)`,
`newton(NewtonConfig = {})`, `requires_residual(bool = true)`, `requires_linearization(bool = true)`
and `id()`. The declaration it fills is `IterationRegionDecl` (`name`, `channels`,
`max_iterations = 1`, `convergence`, `acceleration`, `window_scope`,
`requires_residual_capability`, `requires_linearization_capability`, `newton_config`), which you
can also fill by hand and pass to `add_iteration_region()`.

**Convergence measures.** `Converged::Relative(tol, field)`, `Absolute`, `AbsoluteOrRelative` and
`ResidualRelative` each return a `ConvergenceMeasure`; `&` joins them into a `ConvergenceCriterion`.
`field` names a channel, its source field or its target field, and must match exactly one. Every
channel in the region must be measured, or the runner refuses the region with `E-ITER-MEASURE`. A
tolerance that is not finite and positive throws `InvalidArgument` at the call. [](convergence.md)
covers the norms.

**Acceleration.** `AccelKind` is a ladder from weakest to strongest:

| `AccelKind` | What it does | When to use it |
|---|---|---|
| `None` | Plain Picard: deliver the new value unchanged | A contraction that already converges fast, or a reference run |
| `Constant` | A fixed relaxation factor `omega` | A mildly unstable fixed point where one factor is known to work |
| `Aitken` | One dynamic relaxation factor for the whole vector | The first thing to try; cheap and robust for one dominant mode |
| `Steffensen` | One dynamic factor per entry (a diagonal secant) | Entries that converge at visibly different rates |
| `Secant` | A rank-1 vector least-squares update | A step up from Aitken with no history to tune |
| `IQN_ILS` | Rank-k interface quasi-Newton, history cleared each window | Several coupled modes, stiff interfaces, FSI; the recommended strong method |
| `IQN_IMVJ` | IQN-ILS plus an inverse Jacobian carried across windows | Only when IQN-ILS is measured to be too slow; its carried Jacobian is never refreshed |
| `Newton` | Jacobian-free Newton-Krylov replacing the sweep | A region whose members implement `ImplicitComponent`; see [](convergence.md) |

`AccelConfig` carries `kind`, `initial_relaxation` (used when no history exists), `omega` (for
`Constant`), `reuse` (secant pairs kept), `filter` (`FilterKind::None`, `QR1`, `QR2`, `QR3`),
`limit` (the filter threshold) and `scale_fields`. The default filter, `QR2` at `1e-2`, is tuned for
ill-conditioned fluid-structure problems; on a well-conditioned problem it discards useful secant
information, and `FilterKind::None` is usually the better choice ([](dn-heat.md) measures the
difference).

:::{warning}
`fuse::AccelConfig{}` defaults to `kind = AccelKind::Constant`, but a region declared without
`.accelerate()` defaults to `AccelKind::None`. Writing `.accelerate(fuse::AccelConfig{})` to mean
"no acceleration" selects constant relaxation at `omega = 0.5`. Set `kind` explicitly.
:::

**Newton.** `.newton(cfg)` selects `AccelKind::Newton` and marks the region as requiring the
residual capability from every member (`.accelerate({.kind = Newton})` does the same). The
highlights of `NewtonConfig`: `line_search_kind` (`BT`, `CP`, `Basic`), `forcing_choice`
(Eisenstat-Walker `Constant`, `Choice1`, `Choice2`), the stopping tolerances `atol`, `rtol`, `stol`,
`divergence_tolerance`, `max_nonlinear_iterations`, the Krylov budget `gmres_restart` and
`gmres_max_iterations`, `use_block_jacobi_preconditioner` (also requires the linearization
capability), `substrate` (`Native` or `Petsc`), `norm` (`Plain` or `MeasureWeighted`) and
`residual_path` (`Advance` or `Direct`). [](convergence.md) explains the choices.

A region's own consistency is not checked by `validate()`. `lower()` refuses a region that covers no
nodes (`E-IR-REGION-EMPTY`); `Runner::initialize()` refuses an empty criterion
(`E-ITER-NOCONVERGE`), `max_iterations < 1` (`E-ITER-MAXITER`), an unmeasured or ambiguously
measured channel (`E-ITER-MEASURE`), a member without rollback (`E-ITER-NOROLLBACK`) or with
undeclared determinism (`E-ITER-NONDETERMINISTIC`), and a Newton region whose members lack the
capability (`E-NEWTON-CAPABILITY`). `schedule()` refuses a region whose channels sit on two clocks
with no window (`E-SCHED-REGION`).

### Coupling windows

A region whose members do not share a step is iterated over a *window* rather than a tick. There is
no builder for this: fill `IterationRegionDecl::window_scope` and call `add_iteration_region()`.

```cpp
const auto fwd  = g.add_channel("a2b", a_out, b_in);
const auto back = g.add_channel("b2a", b_out, a_in);
g.channel(back).lag(1).allow_time_level_mismatch("resolved by the window's own sweep");

fuse::CouplingWindowScope scope;
scope.window_component = "a";          // the window is a's step; or scope.explicit_interval = ...
scope.order = 2;                        // interpolation and extrapolation order at the boundary
scope.policies = {{"a", fuse::DtPolicy::Adjustable}, {"b", fuse::DtPolicy::Fixed}};
scope.iteration = fuse::WaveformIteration::GaussSeidel;   // default Jacobi needs no lag

fuse::IterationRegionDecl decl;
decl.name = "window";
decl.channels = {fwd, back};
decl.max_iterations = 20;
decl.convergence = fuse::Converged::Relative(1e-6, "a2b") & fuse::Converged::Relative(1e-6, "b2a");
decl.window_scope = scope;
g.add_iteration_region(decl);
```

Set exactly one of `window_component` and `explicit_interval`. Every component that owns an end of
a region channel needs a `DtPolicy` in `policies`: `Adjustable` shortens its last substep to land on
the window end, `Fixed` keeps its own grid and straddles. `order` must be at least 2. Under
`WaveformIteration::Jacobi` every region channel is treated as lagged by one sweep whatever `lag()`
says; under `GaussSeidel` at least one channel must carry an explicit `lag(1)`. `schedule()` checks
all of this with the `E-SCHED-WINDOW-*` codes ([](ir.md#what-schedule-proves-and-refuses)).
[](coupling-windows.md) covers the semantics and the runtime.

### Reading the graph back

Every declaration is readable; all accessors are bounds-checked.

```cpp
const fuse::ComponentDecl& c = g.component(fluid);
const fuse::FieldSpec&     f = g.field(q_out);
const fuse::Channel&       ch = g.channel_at(flux);         // the recorded treatment
const fuse::ComponentId owner = g.owner_of(q_out);
const std::optional<fuse::ComponentId> by_name = g.find_component("solid");
const std::optional<fuse::FieldId> fid = g.find_field(*by_name, "T");
const fuse::MergeSpec* m = g.merge_for(q_in);               // nullptr if none
for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
  const fuse::Channel& c_i = g.channel_at(fuse::ChannelId{i});
  if (c_i.target == q_in) { /* a writer of q_in */ }
}
```

Also available: `mesh()`, `region()`, `iteration_region()`, the `*_count()` functions for every
kind, `merge_count()` and `merge_at(i)`. There is no `find_channel()`; loop over
`channel_count()` and compare `channel_at(id).name`. Two free functions answer questions the
compiler also asks: `effective_channel_lag(g, ch)` (the lag as compiled, forced to 1 inside a
Jacobi window) and `window_length(g, scope)`.

### Saving and loading

```cpp
#include <fuse/model/descriptor.hpp>
#include <fuse/model/yaml_io.hpp>

const std::vector<std::byte> bytes = fuse::serialize(g);      // the versioned binary descriptor
const fuse::Result<fuse::CouplingGraph> back = fuse::deserialize(bytes);
if (back.ok() && fuse::structurally_equal(g, back.value())) { /* round trip is exact */ }

const fuse::Result<std::string> yaml = fuse::to_yaml(g);
if (!yaml.ok()) { /* yaml.error().code, yaml.error().message */ }
const fuse::Result<fuse::CouplingGraph> loaded = fuse::from_yaml(yaml.value());
const fuse::Result<fuse::CouplingGraph> from_disk = fuse::load_yaml_file("coupling.yaml");
```

The descriptor is what every front end shares (C, Fortran, Python, separate executables);
`deserialize()` refuses a foreign or truncated buffer with `E-DESC-MAGIC`, `E-DESC-VERSION` or
`E-DESC-TRAIL`. `structurally_equal()` compares the two serialisations byte for byte. The YAML
writer nests fields inside their component, so declare a component's fields together if you want
field ids to survive a round trip unchanged.

## Validating what you declared

```cpp
#include <fuse/model/validate.hpp>

const fuse::ConnectPolicy policy{};                 // .allow_automatic_unit_scale = true
if (const fuse::Diagnostics d = fuse::validate(g, policy); d.has_errors()) {
  std::cerr << d.to_string();
  return 1;
}
const fuse::LoweringResult lowered = fuse::lower(g, policy);   // the same policy
```

:::{warning}
Nothing calls `validate()` for you. `lower()`, `schedule()` and the `Runner` each check what they
need to proceed, not what you meant. Call `validate(g, policy)` yourself, print `d.to_string()`,
and pass the same `ConnectPolicy` to `lower()`; otherwise the checker and the compiler can disagree
about which unit conversions are allowed.
:::

`Diagnostics` holds every finding at once: `has_errors()`, `size()`, `items()` (each a
`Diagnostic{severity, code, message, context}`) and `to_string()`, which prints one line per item
as `error CODE: message [context]` or `warning CODE: ...`. A warning does not stop you, but each one
names something FUSE did that you did not write, or a declaration that has no effect; read them.

### Codes by what you declared

| Code | Meaning | Fix |
|---|---|---|
| **References** | | |
| `E-BAD-REF` | A region names a missing mesh, a field a missing region, a channel a missing field, a `remap_on` or a `Window::from()` a missing component | Use the ids the `add_*` calls returned |
| **Components** | | |
| `E-DUP-COMP` | Two components share a name | Names are the only link to a `fuse::Component`; make them unique |
| `E-INTERVAL` | `interval <= 0` | Give every component a positive `Rational` step |
| `E-COUPLER-FIELDS` | A coupler owns fields | Move the fields to a physics component |
| `E-PLACEMENT-EMPTY` | A channel's source, target or `remap_on` component is on no rank | Give it ranks, or remove the channel for runs where it is switched off |
| `E-LAUNCH-APPNUM`, `E-LAUNCH-CONNECT-TARGET`, `E-LAUNCH-PLUGIN-PATH` | A launch mode is missing its parameter | Set `appnum`, `port_file`/`service_name` or `plugin_path` |
| `E-LOWFIVE-COMPONENT-KIND` | A `LaunchMode::LowFive` component is not a bare coupler | Declare it with `add_coupler()` |
| **Fields** | | |
| `E-DUP-FIELD` | Two fields of one component share a name | Rename one |
| `E-SHAPE` | `components < 1`, or the two ends of a channel differ in component count | Match the counts |
| `E-LOWFIVE-NO-COMPONENT`, `E-LOWFIVE-DUPPATH` | `hdf5_path` without a LowFive component, or two fields on one path | See [](lowfive.md) |
| `W-NO-CONSUMER` | A `Provides` field no channel reads | Harmless; remove the field or add the channel |
| **Channels** | | |
| `E-UNITS` | Different dimensions (K against W) | Fix the declaration; no conversion exists |
| `E-UNITS-UNDECLARED` | A unit could not be interpreted | Give both ends a `fuse::Units` value |
| `E-UNITS-SCALE` | The conversion needs an offset (degC to K) or an affine map, or scaling is forbidden by the policy | Declare both fields in the same unit, or convert in the component |
| `I-UNIT-SCALE-INSERTED` (a warning) | An exact scale was inserted | Expected when the units differ by a ratio; a `Scale` node appears in the IR |
| `E-CONSERV` | The two ends differ in `conservation` | Make them agree |
| `E-INTENT` | The source is `Requires` or the target is `Provides` | Swap the ends, or change the intents |
| `E-SELF` | Both ends belong to one component | A channel couples two components |
| `E-CONN-LOCATION` | The two fields are represented differently (location, continuity, degree) and the channel is a plain copy | Declare a `remap()` |
| `E-CONN-CONSERVATION` | The target is `Extensive` but the transfer does not conserve | Add `conservative()`, or make the field `Intensive` |
| `E-CHANNEL-REMAP-ON` | `remap_on` names neither a coupler nor an endpoint | Name a coupler, the source or the target |
| `W-CHANNEL-REMAP-ON-UNUSED` | No `Remap` would be emitted, so `remap_on` does nothing | Remove it, or add the flags that need it |
| `E-LAG` | `abs(lag) × source interval` exceeds the window the value must survive (the target's step, or the region's window) | Lag the faster channel, or lag fewer steps |
| `W-MISMATCH-OK` | A time-level mismatch was authorised | Expected with `lag()`; the message carries your reason |
| **Merges** | | |
| `E-MULTI-WRITER` | Several channels target one field and no `merge()` says how | Declare `merge(field, op)`, or retarget all but one |
| `E-MERGE-FIELD` | A merge names a field that does not exist | Use a valid `FieldId` |
| `W-MERGE-UNUSED` | A merge is declared for a field with fewer than two writers | Remove the merge, or the channel that went missing |

Iteration regions and windows are checked later, by `lower()`, `schedule()` and
`Runner::initialize()`; the codes are listed under [Iteration regions](#iteration-regions) and in
[](ir.md#what-schedule-proves-and-refuses).

### Pitfalls the checker will tell you about

**Lag runs the wrong way.** `E-LAG` measures `lag × source interval` against the *target's* step.
In the complete example above, lagging `wall_temperature` (a slow source, step 1) into the fluid
(step 1/4) is refused: `lag spans 1 but the coupling window is 1/4`. Lagging `heat_flux` (step 1/4
into step 1) is fine. Put the lag on the channel whose source is faster, or lag inside a window
region, where the window length is the limit.

**A lagged channel needs a reason.** `lag()` alone validates, but lowering refuses it:

```text
error E-IR-TYPE: cannot assign Instant<r0, K, [WindowEnd-1, k], ranks=0> to Instant<r0, K, [WindowEnd, k], ranks=0> [node 13 "a.t_in"]
```

Add `.allow_time_level_mismatch("why")`. The reason is kept in the graph and in the YAML, so the
next reader knows the stagger is deliberate. Inside a `WaveformIteration::Jacobi` window the
compiler applies the lag itself and no reason is needed.

**`accumulate` and `spacetime` on one channel.** Both need a sampler on the source clock, and the
schedule holds one. `schedule()` refuses the pair:

```text
error E-SCHED-SPACETIME: channel 'flux' asks for both accumulate(Mean) and the space-time basis Linear. Those are two different temporal operators on the same data and each needs its own sampler on the source clock, which cannot be scheduled; drop one. ...
```

A degree-0 `spacetime` onto one target step is already the window mean.

**A cycle with nothing to break it.** Two channels closing a loop with no `lag()` anywhere, or an
iteration region around them with no `lag()` and no `relax()`, is an algebraic loop. `schedule()`
refuses it with `E-SCHED-CYCLE` and prints the cycle, node by node, ending with the fix:

```text
error E-SCHED-CYCLE: E-SCHED-CYCLE: the expanded schedule for one coupling period (period = 1) contains a dependence cycle that no lag breaks, so no execution order exists and the coupled system is an algebraic loop.
  channels on the cycle: 'wall_temperature', 'heat_flux'
  cycle:
    solid.T (Source) @t=0
    -> wall_temperature/extrapolate (Extrapolate, channel 'wall_temperature') @t=1
    ...
  Fix: add .lag(1) to one channel on the cycle to make it an explicit scheme, together with .allow_time_level_mismatch("reason") to authorise the staggering.
```

## Improving coupling semantics

Each goal below is a change to the graph, not to a solver.

### Conserve a flux across the interface

```cpp
// Both fields Conservation::Extensive, then:
g.channel(flux).conservative();
```

The remap becomes a conservative one and a `Constrain` node checks the integral after every
transfer; `runner.stats()` counts the checks and any violation. Without `conservative()` a channel
into an `Extensive` field is refused with `E-CONN-CONSERVATION`.

### Reduce the lag error of an explicit scheme

```cpp
g.channel(temp).extrapolate(1);                 // predict instead of hold
// or resolve it:
g.iterate({flux, temp}).until(crit).max_iterations(50)
    .accelerate(fuse::AccelConfig{.kind = fuse::AccelKind::Aitken});
```

`extrapolate(order)` keeps the scheme explicit and removes the first-order hold error. An iteration
region removes the error altogether at the cost of several sweeps per step.

### Stabilise an iteration that oscillates or diverges

```cpp
g.channel(temp).relax(0.5);                                            // under-relax one channel
g.iterate({flux, temp}).until(crit).initial_relaxation(0.3)
    .accelerate(fuse::AccelConfig{.kind = fuse::AccelKind::IQN_ILS,
                                  .filter = fuse::FilterKind::None});   // or Aitken first
```

Start with `Aitken`; move to `IQN_ILS` when several modes are slow. If an IQN region stalls, try
`FilterKind::None` before anything else.

### Couple components whose steps have no useful common period

```cpp
fuse::CouplingWindowScope scope;
scope.explicit_interval = fuse::Rational{1, 10};
scope.policies = {{"ocean", fuse::DtPolicy::Fixed}, {"ice", fuse::DtPolicy::Adjustable}};
decl.window_scope = scope;                      // then g.add_iteration_region(decl)
```

A window replaces the tick as the unit of exchange; each member advances across it against its
partners' trajectories ([](coupling-windows.md)).

### Combine several sources into one field

```cpp
g.add_channel("flux_from_a", q_a, q_in);
g.add_channel("flux_from_b", q_b, q_in);
g.merge(q_in, fuse::MergeOp::Sum);              // fluxes add; states would use Mean
```

### Keep remapping off the solver's ranks

```cpp
const auto hub = g.add_coupler({.name = "hub", .ranks = fuse::RankSet{8, 4}});
g.channel(flux).conservative().remap_on(hub);
```

The value moves source → coupler → target, with a rank-agreement `Epoch` after each move
([](placement.md#a-separately-placed-coupler)).

### Units: declare them and let FUSE scale

```cpp
constexpr fuse::Units kKiloWattPerSquareMetre{.mass = 1, .time = -3, .scale = fuse::Rational{1000}};
// source field .units = fuse::units::kHeatFluxDensity, target field .units = kKiloWattPerSquareMetre
```

`validate()` reports `I-UNIT-SCALE-INSERTED: inserting an exact unit scale of 1/1000` and the IR
gains a `Scale` node. An offset (degC against K) is refused with `E-UNITS-SCALE`: moving the zero
of a scale changes the meaning of a mean or a sum, so convert it inside the component. Setting
`ConnectPolicy{.allow_automatic_unit_scale = false}` turns the inserted scale into `E-UNITS-SCALE`
as well, for projects that want every conversion explicit.

## Building a graph from your own configuration

The authoring calls are ordinary C++, so a graph can be assembled from any configuration structure.
Keep the ids in maps keyed by the names your configuration uses, look them up when a channel refers
to them, and let `validate()` catch what the loop could not.

```cpp
struct FieldCfg   { std::string name; std::string region; fuse::Units units;
                    fuse::Conservation conservation; fuse::Intent intent; };
struct CompCfg    { std::string name; int first_rank; int rank_count; fuse::Rational dt;
                    std::vector<FieldCfg> fields; };
struct ChannelCfg { std::string name, from_component, from_field, to_component, to_field;
                    bool conservative; int lag; };

fuse::Result<fuse::CouplingGraph> graph_from_config(
    const std::vector<std::pair<std::string, std::string>>& meshes,   // name, source
    const std::vector<CompCfg>& components, const std::vector<ChannelCfg>& channels) {
  fuse::CouplingGraph g;
  std::map<std::string, fuse::RegionId> regions;
  for (const auto& [name, source] : meshes) {
    const auto mesh = g.add_mesh({.name = name, .source = source});
    regions[name] = g.add_region({.name = name, .mesh = mesh});
  }
  for (const CompCfg& c : components) {
    const auto id = g.add_component({.name = c.name,
                                     .ranks = fuse::RankSet{c.first_rank, c.rank_count},
                                     .interval = c.dt});
    for (const FieldCfg& f : c.fields) {
      const auto r = regions.find(f.region);
      if (r == regions.end()) {
        return fuse::Failure{"E-CONFIG-REGION", c.name + "." + f.name +
                             " names unknown region '" + f.region + "'"};
      }
      g.add_field(id, {.name = f.name, .region = r->second, .units = f.units,
                       .conservation = f.conservation, .intent = f.intent});
    }
  }
  for (const ChannelCfg& ch : channels) {
    const auto src_c = g.find_component(ch.from_component);
    const auto dst_c = g.find_component(ch.to_component);
    const auto src_f = src_c ? g.find_field(*src_c, ch.from_field) : std::nullopt;
    const auto dst_f = dst_c ? g.find_field(*dst_c, ch.to_field) : std::nullopt;
    if (!src_f || !dst_f) {
      return fuse::Failure{"E-CONFIG-CHANNEL", "channel '" + ch.name +
                           "' names a field that was not declared"};
    }
    const auto id = g.add_channel(ch.name, *src_f, *dst_f);
    if (ch.conservative) g.channel(id).conservative();
    if (ch.lag != 0) {
      g.channel(id).lag(ch.lag).allow_time_level_mismatch("lag from configuration");
    }
  }
  return g;
}
```

Three habits make such code easy to debug:

- Resolve names to ids as you go and return a `fuse::Failure` with your own code for anything the
  configuration leaves dangling; `validate()` would report `E-BAD-REF` for an invalid id, but it
  cannot name the configuration entry.
- Add every field of a component before the next component, so YAML round trips keep the ids.
- Run `validate()` once at the end and print all of it. A configuration error usually shows up as
  several related diagnostics (`E-UNITS` on a channel plus `W-NO-CONSUMER` on the field it should
  have read).

If your configuration is already YAML, `fuse::load_yaml_file()` reads the FUSE schema directly
and `validate()` applies to the result in the same way.

## Modifying a loaded YAML graph in code

`from_yaml()` and `load_yaml_file()` return an ordinary `CouplingGraph`. Everything the builders can
set can be set again on it, and new declarations can be appended:

```cpp
fuse::Result<fuse::CouplingGraph> loaded = fuse::load_yaml_file("coupling.yaml");
if (!loaded.ok()) { std::cerr << loaded.error().message << "\n"; return 1; }
fuse::CouplingGraph g = std::move(loaded).value();

for (std::uint32_t i = 0; i < g.channel_count(); ++i) {       // find a channel by name
  const fuse::ChannelId id{i};
  if (g.channel_at(id).name == "heat_flux") {
    g.channel(id).relax(0.7).remap(fuse::RemapFlags::Monotone);   // flags accumulate
  }
}
if (const auto solid = g.find_component("solid")) {            // append to a component
  if (const auto t = g.find_field(*solid, "T")) {
    const auto p = g.add_field(*solid, {.name = "p_wall", .region = g.field(*t).region,
                                        .units = fuse::units::kPascal});
    // wire p with g.add_channel(...) as for any other field
  }
}
std::vector<fuse::ChannelId> all;
for (std::uint32_t i = 0; i < g.channel_count(); ++i) all.push_back(fuse::ChannelId{i});
g.iterate(all).name("everything").until(fuse::Converged::Relative(1e-5, "heat_flux"))
    .max_iterations(30);
```

What can change on a loaded graph:

- Any channel treatment, through `g.channel(id)`. `lag(0)` removes a lag and
  `accumulate(AccumOp::Instant)` restores the default time treatment; `relax()` and `extrapolate()`
  stay enabled once set (a `Relax` node is emitted even at `omega = 1.0`), and `remap()` flags only
  accumulate.
- Any iteration region field, through `fuse::IterationBuilder{g, idx}`; `accelerate()` replaces the
  whole `AccelConfig`.
- Merge rules, through `merge()` (an existing rule for the field is replaced).
- New meshes, regions, components, fields, channels and regions, appended with the usual calls.

What cannot change: there is no API to remove or rename a declaration, to retarget a channel, or to
edit a `ComponentDecl`, `FieldSpec` or `Region` in place. Build a new graph for that, or edit the
YAML. After any change, run `validate()` again and, if you keep the result, `to_yaml()` it: a graph
edited in memory and the file it came from are now different couplings.

## Limitations

- Declarations are append-only; removal, renaming and retargeting are not supported.
- `accumulate()` with any operator other than `Instant` and `spacetime()` cannot be combined on one
  channel; `schedule()` refuses the pair with `E-SCHED-SPACETIME`.
- Unit conversions with an offset or an affine map are not inserted; `validate()` refuses them with
  `E-UNITS-SCALE`.
- A single iteration region cannot span two clocks without a window scope; `schedule()` refuses it
  with `E-SCHED-REGION`.
- Component and channel treatments beyond the `ChannelBuilder` set (sign flips, component selection
  or reordering, clipping) are not expressible in the graph.

## See also

- [](fluent-api.md): why builders record and `validate()` reports.
- [](ir.md): what the graph compiles to, and how to read it.
- [](combining-writers.md), [](placement.md), [](coupling-windows.md), [](convergence.md): the
  pages behind `merge()`, `remap_on()`, `window_scope` and `iterate()`.
- [](cookbook.md) for the shortest version of each recipe, [](troubleshooting.md) for the error
  code families.
