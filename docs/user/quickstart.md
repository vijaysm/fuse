# Quickstart

A complete coupling of two components that run at different time steps and exchange one field,
walked through stage by stage. The code on this page is pulled from `examples/mms_pair/`, which
`ctest` runs on every build, so it compiles and runs as shown. Run it first
([](getting-started.md)) and then read this page with the source open.

Every FUSE program goes through the same five stages:

1. **Describe** the coupling as a `fuse::CouplingGraph`.
2. **Load** the meshes the graph names into a backend.
3. **Compile**: check the graph, then lower it to a dataflow graph.
4. **Schedule**: turn the dataflow graph into a fixed instruction order.
5. **Run**: register each component's storage, initialise, advance.

## Includes, linking and Kokkos

```cpp
#include <Kokkos_Core.hpp>
#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>
```

```cmake
find_package(fuse-coupler CONFIG REQUIRED)
target_link_libraries(my_coupling PRIVATE fuse::runtime fuse::backend)
```

The coupling hot loop runs on Kokkos, so initialise it before touching a `Runner` and keep the guard
alive for the whole program:

```cpp
int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos_guard(argc, argv);
  // ...
}
```

## 1. Describe the coupling

```{literalinclude} ../../examples/mms_pair/main.cpp
:language: cpp
:start-at: "fuse::CouplingGraph build_graph(std::size_t cells) {"
:end-at: "return g;"
```

What this declares:

- Two meshes, each a generated 64-cell line, and a region on each (`.dimension = 1` means cells of a
  line).
- Two components. `left` takes steps of 1/4000 and runs on rank 0 (`.ranks = {0, 1}` is
  *first rank, count*). `right` takes steps of 1/1000 and runs on rank 1. Steps are
  `fuse::Rational`, an exact fraction.
- One field on each side: `left` *provides* `T`, `right` *requires* `T_bc`. Both are in kelvin, so no
  unit conversion is inserted.
- One channel linking them, with `Mean` as its time treatment: `right` receives the mean of `left`'s
  four sub-steps.
- `Conservation::Extensive` on both fields. That is what makes FUSE add a conservation check to the
  transfer. Without it the check would not exist and the example's "conservation checks" line would
  read zero.

Every `add_*` call takes a designated-initialiser struct and returns an id. `g.channel(ch)` returns a
builder you chain; none of those calls can fail. To catch defects early, check the whole graph before
compiling. Nothing calls `validate()` for you:

```cpp
#include <fuse/model/validate.hpp>

if (fuse::Diagnostics d = fuse::validate(graph); d.has_errors()) {
  std::cerr << d.to_string();      // every defect, in a deterministic order
  return 1;
}
```

:::{note}
`mms_pair` itself skips this call, and `validate()` does have something to say about its graph: both
fields are `Extensive` but the channel does not declare `.conservative()`, so it reports
`E-CONN-CONSERVATION`. Your own graphs should pass `validate()` cleanly: add `.conservative()` to a
channel whose transfer must conserve the field, or mark the fields `Intensive` if it need not.
[](graph-programming.md) lists every validation code and its fix.
:::

## 2. Write the components

A component implements `fuse::Component`. This is `left`'s heat solver, `HeatComponent`, from
`examples/mms_pair/heat_component.hpp`:

```{literalinclude} ../../examples/mms_pair/heat_component.hpp
:language: cpp
:start-at: "fuse::Result<void> register_fields("
:end-before: "fuse::Result<void> save_state() override {"
```

Three rules matter, and the interface comments in `fuse/runtime/component.hpp` spell them out:

- **Register storage once and write in place.** FUSE keeps the address `register_fields()` resolved.
  This component therefore writes the new level into a scratch buffer and copies it back
  element by element. Swapping two vectors would leave FUSE reading the old buffer with no error.
- **Bind only the right arrays.** A `Provides` field points at the state array. A `Requires` field
  points at a separate buffer, because FUSE overwrites it at every exchange.
- **Return a `Result`, do not `assert`.** The solver above refuses a time step that would be
  unstable. In a Release build an `assert` would vanish and an unstable scheme would produce growing
  numbers without complaint.

Components inside an iteration region must also implement `save_state()` and `restore_state()` so a
rejected iteration can be re-run. `save_state()` and `capabilities()` are shown in the same file.

## 3. Load, compile and schedule

```{literalinclude} ../../examples/mms_pair/main.cpp
:language: cpp
:start-at: "fuse::SerialMeshBackend backend;"
:end-before: "if (dump) {"
```

`lower()` returns a `LoweringResult`: `.dag` is the dataflow graph and `.diags` carries any errors.
`schedule()` returns the instruction stream in `.plan`, or errors in `.diags`. It refuses a coupling
it cannot prove free of deadlock, naming the cycle.

### Seeing what the compiler did

`fuse::dump_text(lowered.dag)` prints the dataflow graph and `sched.plan.to_string()` prints the
compiled schedule, with every operator FUSE inserted that you did not write: unit conversions, time
interpolation, remap, transport, rank agreement and conservation checks. This is the first thing to
read when a coupling behaves unexpectedly, because it is what actually runs. In `mms_pair` it is the
`--dump` flag ([](getting-started.md#3-look-at-what-fuse-built)).

## 4. Run

```{literalinclude} ../../examples/mms_pair/main.cpp
:language: cpp
:start-at: "fuse::ComponentSet components;"
:end-before: "const double heat_err"
```

`ComponentSet` owns the component objects. Each one binds its arrays into the `FieldRegistry`.
`Runner::initialize()` resolves everything it needs once (remap weights, transport plans, buffers),
so `run()` allocates nothing in its step loop. `runner.stats()` then reports the number of periods,
instructions executed, and the conservation checks and violations.

## Four things the example does not say aloud

**Time is exact.** `fuse::Rational{1, 4000}` is a fraction. Intervals, windows and every instruction
in the plan use exact arithmetic, so the schedule is still in step after ten million steps and the
deadlock proof runs on exact values ([](concepts.md#time)).

**Rank sets are explicit and may be disjoint.** `left` runs on rank 0 and `right` on rank 1, so under
two MPI ranks they execute concurrently instead of taking turns. In a single process the same graph
runs unchanged ([](placement.md)).

**`validate()` reports everything.** A graph with several unrelated defects yields several
diagnostics in one call ([](fluent-api.md#diagnostics-accumulate)).

**Compiling is a pipeline with an inspectable middle.** The dataflow graph and the plan are plain
data you can print, diff and keep as a golden file.

## Where next

- [](mms-pair.md): this example in full, with its problem statement, expected output and checks.
- [](graph-programming.md): every declaration and treatment; [](component-interface.md): the full
  component contract; [](drivers.md): turning this into an executable you control.
- [](cookbook.md): recipes for iteration, remapping, windows, restart and more.
- [](dn-heat.md): the same pipeline with an iteration region and an accelerator.
- [](concepts.md): the model behind the five stages.
