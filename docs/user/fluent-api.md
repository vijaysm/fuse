# The fluent API and its rules

This page is for anyone writing against the C++ API for the first time. FUSE uses method chaining
in **two distinct forms** that never mix: an *authoring* chain that builds a coupling
description and reports every problem at once, and a *pipeline* chain over field data that does
no work until a value is read. Knowing which one you are holding tells you what it costs and how
it fails.

## Authoring chains build a description

A `fuse::CouplingGraph` (`include/fuse/model/coupling_graph.hpp`) is built by
`add_mesh`/`add_region`/`add_component`/`add_coupler`/`add_field`/`add_channel`/
`add_iteration_region`, each taking a designated-initializer declaration struct and returning an
id, and refined by two builders: `fuse::ChannelBuilder` (`g.channel(id)`,
`include/fuse/model/channel.hpp`) and `fuse::IterationBuilder` (`g.iterate({...})`). Builder
calls chain and commit nothing:

```cpp
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/validate.hpp>

fuse::CouplingGraph g;
const auto mesh   = g.add_mesh({.name = "m", .source = "generated:box"});
const auto region = g.add_region({.name = "wall", .mesh = mesh,
                                   .entity_set = "w", .dimension = 2});
const auto fluid = g.add_component({.name = "fluid", .ranks = fuse::RankSet{0, 1},
                                     .interval = fuse::Rational{1}});
const auto solid = g.add_component({.name = "solid", .ranks = fuse::RankSet{1, 1},
                                     .interval = fuse::Rational{1}});
const auto src = g.add_field(fluid, {.name = "q", .region = region,
                                      .units = fuse::units::kHeatFluxDensity,
                                      .conservation = fuse::Conservation::Extensive,
                                      .intent = fuse::Intent::Provides});
const auto dst = g.add_field(solid, {.name = "q_in", .region = region,
                                      .units = fuse::units::kHeatFluxDensity,
                                      .conservation = fuse::Conservation::Extensive,
                                      .intent = fuse::Intent::Requires});
const auto ch = g.add_channel("flux", src, dst);
g.channel(ch).conservative().accumulate(fuse::AccumOp::Mean);
```

`ChannelBuilder` methods (`conservative()`, `remap()`, `remap_method()`, `remap_on()`,
`accumulate()`, `spacetime()`, `relax()`, `extrapolate()`, `lag()`,
`allow_time_level_mismatch()`) and `IterationBuilder` methods (`until()`, `max_iterations()`,
`accelerate()`, `newton()`, `initial_relaxation()`, `name()`, `requires_residual()`,
`requires_linearization()`) return the builder by reference and never return a status. An
instruction FUSE cannot honour is *recorded* and surfaces when the graph is checked, never as an
exception mid-chain. The one exception to "nothing is checked while authoring" is an argument
that cannot be stored at all: a `RankSet` whose count overflows (`E-RANKSET-RANGE`), a
`Converged::*` tolerance that is not finite and positive (`fuse::InvalidArgument`), or an
out-of-range id handed to `g.channel(id)` (an assertion). Those are caller bugs, not graph
defects, and they fail at the call.

The graph has no remove or rename operations, and no mutable access to a channel or region except
through its builder. Read it back with `mesh()`, `region()`, `component()`, `field()`,
`channel_at()`, `iteration_region()`, the `*_count()` functions, `owner_of(FieldId)`,
`find_component(name)` and `find_field(component, name)`.

Two of them carry rules `validate()` enforces that are easy to miss:

- `accumulate(op)` reduces the source's samples over each target step. `Sum`, `Mean`, `Min` and
  `Max` deliver a value in the source's own units; `Integral` delivers the left-rectangle time
  integral, `(sum of samples) * dt` with `dt` the source component's interval, whose units are
  the source's times time -- so an `Integral` channel from a power (W) must land in an energy (J),
  and one declared with equal units on both ends is refused with `E-UNITS` (the message names the
  integrated units).
- `remap_method(name)` must name a method a weight builder recognises: `""` (the default
  conservative supermesh), `"intx"`/`"conservative"`, `"tempest"`, `"consistent"`,
  `"consistent-transpose"`, `"mls"`/`"mls:..."`. Any other string is `E-CHANNEL-REMAP-METHOD`, and
  the weight builders refuse it again at run time (`E-REMAP-UNSUPPORTED`) instead of silently
  building the default remap under the misspelt name.

(diagnostics-accumulate)=
## Diagnostics accumulate

An authoring chain has nowhere to put an error, so FUSE does not try. One call,
`fuse::validate(g)` (`include/fuse/model/validate.hpp`), reports everything the graph gets wrong at
once, as a `fuse::Diagnostics` (`include/fuse/base/result.hpp`):

```cpp
if (fuse::Diagnostics d = fuse::validate(g); d.has_errors()) {
  fmt::print("{}", d.to_string());   // every failure, deterministically ordered
  return 1;
}
```

First-failure reporting is deliberately absent: a coupling graph with ten problems reports ten
`Diagnostic` entries (`d.items()`, each with `severity`, `code`, `message` and `context`), not the
first one encountered. `d.has_errors()` is true if any entry is an error; warnings alone leave it
false. `Diagnostics::merge()` combines the output of several checks run in sequence without losing
any of them. `lower()` and `schedule()` report through the same type (`lowered.diags`,
`sched.diags`), and the C ABI's `fuse_last_error` applies the same accumulate-rather-than-
short-circuit discipline at its boundary.

### What a graph with several defects reports

This graph has an empty placement, a zero interval, a duplicate component name, a unit mismatch,
a field with two writers and no merge rule, and a `Provides` field nothing reads:

```cpp
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/validate.hpp>

#include <cstdio>

int main() {
  fuse::CouplingGraph g;
  const auto mesh   = g.add_mesh({.name = "m", .source = "generated:line:16"});
  const auto region = g.add_region({.name = "wall", .mesh = mesh, .dimension = 1});
  const auto fluid = g.add_component({.name = "fluid", .ranks = fuse::RankSet{0, 2},
                                      .interval = fuse::Rational{1}});
  const auto solid = g.add_component({.name = "solid", .ranks = fuse::RankSet{2, 0},   // defect: empty placement
                                      .interval = fuse::Rational{0}});                 // defect: zero interval
  const auto rad   = g.add_component({.name = "fluid"});                               // defect: duplicate name
  const auto q = g.add_field(fluid, {.name = "q", .region = region,
                                     .units = fuse::units::kHeatFluxDensity,
                                     .intent = fuse::Intent::Provides});
  const auto q_rad = g.add_field(rad, {.name = "q_rad", .region = region,
                                       .units = fuse::units::kHeatFluxDensity,
                                       .intent = fuse::Intent::Provides});
  const auto q_in = g.add_field(solid, {.name = "q_in", .region = region,
                                        .units = fuse::units::kKelvin,                  // defect: unit mismatch with q
                                        .intent = fuse::Intent::Requires});
  g.add_field(solid, {.name = "unused_out", .region = region,
                      .intent = fuse::Intent::Provides});                              // warning: no consumer
  g.add_channel("flux", q, q_in);
  g.add_channel("flux_rad", q_rad, q_in);                                               // defect: two writers, no merge

  const fuse::Diagnostics d = fuse::validate(g);
  std::printf("%zu diagnostics, has_errors=%d\n%s", d.size(), d.has_errors() ? 1 : 0,
              d.to_string().c_str());
  return d.has_errors() ? 1 : 0;
}
```

Expected output:

```
8 diagnostics, has_errors=1
error E-INTERVAL: component 'solid' has non-positive interval 0
error E-DUP-COMP: duplicate component name 'fluid'
error E-PLACEMENT-EMPTY: component 'solid' is placed on no rank, so this channel has nowhere to deliver; an empty placement is legal only for a component with no channels -- give it a placement, or remove the channel for runs where it is switched off [channel 'flux']
error E-UNITS: unit mismatch: source 'q' is kg*s^-3, target 'q_in' is K [channel 'flux']
error E-PLACEMENT-EMPTY: component 'solid' is placed on no rank, so this channel has nowhere to deliver; an empty placement is legal only for a component with no channels -- give it a placement, or remove the channel for runs where it is switched off [channel 'flux_rad']
error E-UNITS: unit mismatch: source 'q_rad' is kg*s^-3, target 'q_in' is K [channel 'flux_rad']
error E-MULTI-WRITER: field 'q_in' is written by 2 channels, and no rule says how to combine them. Summing two heat fluxes and averaging two temperatures are different physics, and nothing in the field descriptions chooses between them. Declare one with graph.merge(field, MergeOp::Sum|Mean|First), or retarget all but one channel.
warning W-NO-CONSUMER: field 'unused_out' is Provides but no channel consumes it
```

The order is fixed: component checks first, then each channel in declaration order (the
`[channel '...']` context names it), then graph-wide checks, then warnings. A rerun on the same
graph prints the same text, so the output can be diffed.

:::{note}
`validate()` checks the graph's declarations. Iteration-region rules that need the lowered
dataflow (`E-ITER-MEASURE`, `E-ITER-NOROLLBACK`, `E-NEWTON-CAPABILITY`, ...) are reported by
`lower()` and by `Runner::initialize()` instead; see [](convergence.md).
:::

## Pipeline chains are lazy

Chains over field *data* (`include/fuse/field/pipeline.hpp`) are a structurally different kind of
chain: every non-terminal call returns a `[[nodiscard]]` view and does no work until a terminal
reads a value. A `FieldSpan` wraps a bound `FieldRef`; its `slice`/`component`/`owned` stages are
plain strided `Kokkos::subview`s of the same memory:

```cpp
auto f = fuse::FieldSpan<>::from(field_ref);
const double total = f.owned(n_owned)                // view: leading n_owned elements
                       .slice({2, fuse::end}, fuse::all)  // view: element 2 to the last
                       .local_sum();                  // terminal: reads the data
```

No type exposes both an authoring method and a pipeline method. A four-stage chain over a field
allocates nothing, because every stage is a view over the same buffer; [](field-pipeline.md) has
the full operation list and a complete example.

## Slicing follows `mdspan`

There is no `2:end` in C++, so `fuse::Extent` (`include/fuse/field/extent.hpp`) uses the spelling
of `std::submdspan` rather than inventing one:

```cpp
f.slice(fuse::all, fuse::all);        // everything                  (cf. std::full_extent)
f.slice({2, 5}, fuse::all);           // elements 2,3,4; all components
f.slice({2, fuse::end}, fuse::all);   // element 2 to the last        (cf. std::strided_slice)
f.slice(7, fuse::all);                // element 7 only
```

Ranges are half-open, like every other range in C++. A reversed or out-of-range extent is a
`FUSE_REFUSE` (`E-FIELD-EXTENT-REVERSED` / `E-FIELD-EXTENT-RANGE`) carrying the offending bounds,
never a silent clamp: a clamp would turn a caller's arithmetic mistake into a shorter loop that
still produces numbers.

**There is no collective reduction on `FieldSpan`.** `FieldSpan::local_sum()` is named
`local_sum`, not `sum`, because a sum over a distributed field's owned entries is a collective, and
a method named `sum` that silently returned one rank's partial answer would be wrong without anyone
noticing. For a global sum, compose `local_sum()` with `fuse::linalg::Reducer`
(`include/fuse/linalg/comm.hpp`; see [](../dev/linalg-distributed.md)).

## Chains cannot dangle

```cpp
fuse::OwnedField<> make_field();

auto x = make_field().slice({1, fuse::end}, fuse::all);   // rvalue: the result OWNS its data
fuse::OwnedField<> owned("f", n, 3);
auto y = owned.slice({1, fuse::end}, fuse::all);          // lvalue: a view into owned, which outlives y
auto z = owned.span();                                    // a view over the whole field
// make_field().span();                                   // does not compile
```

`OwnedField::span()` and `slice()` are ref-qualified: called on an rvalue, `span()` is `= delete`d
and `slice()` returns an owning value (a managed `Kokkos::subview` shares the reference count, so
the buffer outlives the temporary with no copy); called on an lvalue they return a borrowed
`FieldSpan`. Taking a borrowed view out of a temporary is therefore a compile error rather than a
use-after-free, the same bug class `auto x = a + b` made famous in Eigen, removed by the overload
set rather than documented around.

## See also

- [](quickstart.md) for the authoring chain in a complete program.
- [](graph-programming.md) for every declaration type and builder call.
- [](field-pipeline.md) for field views in the coupling hot loop.
- [](geometry.md) and [](../dev/invariants.md) for the compatibility checks `validate()` runs.
