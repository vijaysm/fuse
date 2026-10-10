# Quickstart

A coupling of two components, running at different time steps and exchanging one field, in the five
stages every FUSE program goes through: build the graph, load its meshes, compile it, schedule it,
run it. This is a trimmed version of `examples/mms_pair/main.cpp` — a tier-0 example that ctest runs
on every build — with the error gates removed; see [](examples.md) for the full listing and what it
measures.

```cpp
#include <Kokkos_Core.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos_guard(argc, argv);   // the coupling hot loop runs on Kokkos

  // 1. Build the graph: meshes, regions, components, fields, one channel.
  fuse::CouplingGraph g;
  const auto m_left  = g.add_mesh({.name = "left_mesh",  .source = "generated:line:64"});
  const auto m_right = g.add_mesh({.name = "right_mesh", .source = "generated:line:64"});
  const auto r_left  = g.add_region({.name = "left",  .mesh = m_left,  .dimension = 1});
  const auto r_right = g.add_region({.name = "right", .mesh = m_right, .dimension = 1});

  const auto left  = g.add_component({.name = "left",  .ranks = {0, 1},
                                       .interval = fuse::Rational{1, 4000}});
  const auto right = g.add_component({.name = "right", .ranks = {1, 1},
                                       .interval = fuse::Rational{1, 1000}});

  const auto t_out = g.add_field(left,  {.name = "T",    .region = r_left,
                                          .units = fuse::units::kKelvin,
                                          .conservation = fuse::Conservation::Extensive,
                                          .intent = fuse::Intent::Provides});
  const auto t_in  = g.add_field(right, {.name = "T_bc", .region = r_right,
                                          .units = fuse::units::kKelvin,
                                          .conservation = fuse::Conservation::Extensive,
                                          .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("T_link", t_out, t_in);
  g.channel(ch).accumulate(fuse::AccumOp::Mean);

  // 2. Load the meshes the graph named, into a backend.
  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    if (const auto r = backend.load(g.mesh(fuse::MeshId{i})); !r.ok()) return 1;
  }

  // 3. Compile: validate, then lower into the operator graph.
  if (auto d = fuse::validate(g); d.has_errors()) {   // every failure, not the first
    fmt::print("{}", d.to_string());
    return 1;
  }
  const auto lowered = lower(g);
  if (!lowered.ok()) return 1;

  // 4. Schedule: a fixed linear instruction order for one coupling cycle.
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  if (!sched.ok()) return 1;

  // 5. Run: register each component's storage, initialise, advance.
  fuse::ComponentSet components;
  components.add(/* a fuse::Component for "left" bound to t_out */ nullptr);
  components.add(/* a fuse::Component for "right" bound to t_in */ nullptr);
  fuse::FieldRegistry registry;
  for (auto& c : components) { if (!c.register_fields(registry, g).ok()) return 1; }

  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  if (!runner.initialize(components, registry).ok()) return 1;
  return runner.run(components, /*periods=*/100).ok() ? 0 : 1;
}
```

(The `components.add(nullptr)` lines stand in for the two lines of real code in a working
program — each component is an object implementing `fuse::Component`'s `register_fields()` and
`advance()`; `examples/mms_pair/heat_component.hpp` has both, in full.)

Four things that example does not say out loud:

**Time is exact.** `fuse::Rational{1, 4000}` is a rational, not a `double`. Component intervals,
time windows and every instruction in the compiled plan use exact arithmetic, so a schedule is
still exactly in step after ten million steps, and `schedule()`'s own acyclicity proof
([](../dev/invariants.md)) runs on that exact arithmetic rather than on floating-point
approximations of it.

**Rank sets are explicit and may be disjoint.** `left` runs on rank 0 and `right` on rank 1 above
(`ComponentDecl::ranks`, [](placement.md)), so with two MPI ranks the two components execute
concurrently rather than taking turns on one.

**`validate()` reports everything.** A graph with several unrelated defects reports a `Diagnostics`
with several entries in one call, not the first one encountered. See
[](fluent-api.md#diagnostics-accumulate) for why, and for the authoring-chain methods
(`g.channel(ch).accumulate(...)` above is one) that build the description `validate()` checks.

**Compiling is a pipeline with an inspectable middle.** `lower()` turns the graph into the dataflow
graph (`fuse::OperatorDAG`); `schedule()` turns that into the flat instruction stream a `Runner`
executes. `fuse::dump_text(lowered.dag)` (`include/fuse/ir/dump.hpp`) prints the dataflow graph in a
fixed text format, and `sched.plan.to_string()` prints the compiled schedule — every instruction,
its time level, and every operator FUSE inserted that you did not write (unit conversions, time
interpolation, remap). If a coupling behaves unexpectedly, this is the first place to look, because
it is the thing that actually runs.
