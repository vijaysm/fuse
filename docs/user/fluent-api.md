# The fluent API and its rules

FUSE uses method chaining as its idiom in **two distinct forms** that never mix. Knowing which one
you are holding tells you what it costs and how it fails — the distinction is enforced by a static
gate in the test suite, not left to convention.

## Authoring chains build a description

A `CouplingGraph` is built by `add_mesh`/`add_region`/`add_component`/`add_field`/`add_channel`,
each taking a designated-initializer declaration struct and returning an id, and refined by
`ChannelBuilder` (`g.channel(id)`, `include/fuse/model/channel.hpp`), whose calls chain and commit
nothing:

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
`allow_time_level_mismatch()`) never return a status — an instruction FUSE cannot honour is
*recorded* and surfaces when the graph is checked, never as an exception mid-chain. This pattern
(`test/model/validate_test.cpp`'s `Fixture`) is exercised by the whole of `test/model/`.

(diagnostics-accumulate)=
## Diagnostics accumulate

An authoring chain has nowhere to put an error, so FUSE does not try. One call,
`fuse::validate(g)` (`include/fuse/model/validate.hpp`), reports everything the graph gets wrong at
once, as a `Diagnostics` (`include/fuse/base/result.hpp`):

```cpp
if (fuse::Diagnostics d = fuse::validate(g); d.has_errors()) {
  fmt::print("{}", d.to_string());   // every failure, deterministically ordered
  return 1;
}
```

First-failure reporting is deliberately absent: a coupling graph with ten problems reports ten
`Diagnostic` entries (`d.items()`), not the first one encountered. `Diagnostics::merge()` combines
the output of several checks run in sequence without losing any of them — the same accumulate-rather-
than-short-circuit discipline the compatibility stages ([](geometry.md), [](../dev/invariants.md))
and the C ABI's own `fuse_last_error` apply at their own boundaries.

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

No type exposes both an authoring method and a pipeline method — a static assertion in the test
suite pins that separation. Laziness itself is pinned by an allocation-count fence
(`test/field/pipeline_test.cpp`): a four-stage chain over a field allocates nothing, because every
stage is a view over the same buffer, and the fence catches an accidentally-eager rewrite that would
still pass every functional test while running slower and copying memory.

## Slicing follows `mdspan`

There is no `2:end` in C++, so `fuse::Extent` (`include/fuse/field/extent.hpp`) uses the spelling
standard C++ is converging on rather than inventing one:

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

auto x = make_field().span().slice({1, fuse::end}, fuse::all);  // safe: the chain owns its data
fuse::OwnedField<> owned(...);
auto y = owned.slice({1, fuse::end}, fuse::all);                 // view into owned, which outlives y
```

`OwnedField::span()` and `slice()` are ref-qualified: called on an rvalue they are `= delete`d or
return an owning value (a managed `Kokkos::subview` shares the reference count, so the buffer
outlives the temporary with no copy); called on an lvalue they return a borrowed `FieldSpan`. Taking
a borrowed view out of a temporary is therefore a compile error (`test/field/compile_fail_dangling_
span.cpp`) rather than a use-after-free — the same bug class `auto x = a + b` made famous in Eigen,
removed by the overload set rather than documented around.

## See also

- [](quickstart.md) for the authoring chain in a complete program.
- [](field-pipeline.md) for field views in the coupling hot loop.
- [](geometry.md) and [](../dev/invariants.md) for the compatibility checks `validate()` runs.
