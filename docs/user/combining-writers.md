# Several sources writing one field

This page is for anyone whose import field is fed by more than one channel: a wall temperature
seen by two fluid domains, a heat flux with a convective and a radiative contribution, a
background field overridden locally by a finer model. FUSE compiles that, but only once you say
how the contributions combine, with `CouplingGraph::merge()` (`include/fuse/model/merge.hpp`).

## Why you have to say

Summing two heat fluxes and averaging two temperatures are different physics,
and nothing in the field descriptions distinguishes them. Units, mesh, and
conservation class are identical in both cases. A coupler that picked one for
you would be guessing at the physics, and the guess would be invisible in the
results.

So a field with more than one writer and no rule is rejected by `validate()`:

```
error E-MULTI-WRITER: field 'wall_flux' is written by 2 channels, and no rule says how to
combine them. Summing two heat fluxes and averaging two temperatures are different physics,
and nothing in the field descriptions chooses between them. Declare one with
graph.merge(field, MergeOp::Sum|Mean|First), or retarget all but one channel.
```

## Declaring the rule

```cpp
g.add_channel("convective", q_conv, wall_flux);
g.add_channel("radiative",  q_rad,  wall_flux);
g.merge(wall_flux, fuse::MergeOp::Sum);
```

`merge(FieldId target, MergeOp op)` returns the graph, so it chains. Calling it again for the same
field replaces the rule. `merge_for(FieldId)` returns the `MergeSpec*` for a field (or `nullptr`),
and `merge_count()`/`merge_at(i)` enumerate the declared rules in field-id order.

In the text form:

```yaml
merges:
  - field: wall_flux
    component: solid
    rule: Sum
```

| `MergeOp` | Meaning | When it is right |
|---|---|---|
| `Sum` | add the contributions | an extensive quantity split across sources: fluxes into one surface add |
| `Mean` | unweighted arithmetic mean | an intensive quantity sampled by several sources covering comparable extents |
| `First` | take the first contributing field transfer in declaration order | an override: a fine-scale source shadowing a background field |

`Mean` is unweighted. If the sources cover very different extents, it is
probably not what you want; weighted merges are not supported, because the
weights would have to come from coverage information the model does not carry.

Every field transfer targeting the field participates, in declaration order. The rule does not
list them, because a list is something a later field transfer can be left out of, and a
contribution silently dropped is the failure this whole mechanism exists to prevent.

## A complete example

Two components each provide a heat flux into the same `wall_flux` field of a third. The program
below runs the same graph under each rule.

```cpp
#include <fuse/backend/serial_backend.hpp>
#include <fuse/base/units.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/runtime/component.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/plan.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kN = 4;

/// Writes a constant into its one Provides field.
class Emitter final : public fuse::Component {
 public:
  Emitter(std::string name, std::string field, double value)
      : name_(std::move(name)), field_(std::move(field)), value_(value), data_(kN, 0.0) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = *g.find_component(name_);
    return reg.bind(*g.find_field(cid, field_),
                    fuse::FieldRef{data_.data(), fuse::Layout{.stride = 1, .count = kN},
                                   fuse::MemorySpace::Host});
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    for (double& d : data_) d = value_;
    return fuse::Result<void>::success();
  }

 private:
  std::string name_, field_;
  double value_;
  std::vector<double> data_;
};

/// Holds whatever is delivered into its one Requires field.
class Wall final : public fuse::Component {
 public:
  Wall() : data_(kN, 0.0) {}
  [[nodiscard]] std::string name() const override { return "solid"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = *g.find_component("solid");
    return reg.bind(*g.find_field(cid, "wall_flux"),
                    fuse::FieldRef{data_.data(), fuse::Layout{.stride = 1, .count = kN},
                                   fuse::MemorySpace::Host});
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] double value() const { return data_[0]; }

 private:
  std::vector<double> data_;
};

int run(fuse::MergeOp op) {
  fuse::CouplingGraph g;
  const std::string src = "generated:line:" + std::to_string(kN);
  const auto m = g.add_mesh({.name = "m", .source = src});
  const auto r = g.add_region({.name = "wall", .mesh = m, .dimension = 1});
  const auto conv  = g.add_component({.name = "convection", .ranks = fuse::RankSet{0, 1}});
  const auto rad   = g.add_component({.name = "radiation",  .ranks = fuse::RankSet{0, 1}});
  const auto solid = g.add_component({.name = "solid",      .ranks = fuse::RankSet{0, 1}});
  const auto q_conv = g.add_field(conv, {.name = "q_conv", .region = r,
                                         .units = fuse::units::kHeatFluxDensity,
                                         .intent = fuse::Intent::Provides});
  const auto q_rad = g.add_field(rad, {.name = "q_rad", .region = r,
                                       .units = fuse::units::kHeatFluxDensity,
                                       .intent = fuse::Intent::Provides});
  const auto wall_flux = g.add_field(solid, {.name = "wall_flux", .region = r,
                                             .units = fuse::units::kHeatFluxDensity,
                                             .intent = fuse::Intent::Requires});

  g.add_channel("convective", q_conv, wall_flux);
  g.add_channel("radiative",  q_rad,  wall_flux);
  g.merge(wall_flux, op);          // without this line: E-MULTI-WRITER

  const fuse::Diagnostics d = fuse::validate(g);
  if (d.has_errors()) { std::cout << d.to_string(); return 1; }

  fuse::SerialMeshBackend backend;
  if (!backend.load(g.mesh(m)).ok()) return 1;
  const auto lowered = fuse::lower(g);
  if (!lowered.ok()) { std::cout << lowered.diags.to_string(); return 1; }
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks,
                              fuse::make_region_sizes(g, lowered.dag, backend));
  if (!sched.ok()) { std::cout << sched.diags.to_string(); return 1; }

  fuse::ComponentSet components;
  components.add(std::make_unique<Emitter>("convection", "q_conv", 300.0));
  components.add(std::make_unique<Emitter>("radiation",  "q_rad",  50.0));
  components.add(std::make_unique<Wall>());
  fuse::FieldRegistry registry;
  for (auto& c : components) {
    if (!c.register_fields(registry, g).ok()) return 1;
  }
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  if (auto i = runner.initialize(components, registry, fuse::RunnerConfig{}); !i.ok()) {
    std::cout << i.error().message << "\n"; return 1;
  }
  if (!runner.run_period(components).ok()) return 1;

  const auto* wall = dynamic_cast<const Wall*>(components.find("solid"));
  std::printf("%-5s: wall_flux = %g\n", std::string(fuse::merge_op_name(op)).c_str(), wall->value());
  if (op == fuse::MergeOp::Sum) {
    std::cout << sched.plan.to_string();
  }
  return 0;
}

}  // namespace

int main() {
  Kokkos::ScopeGuard kokkos;
  if (run(fuse::MergeOp::Sum)) return 1;
  if (run(fuse::MergeOp::Mean)) return 1;
  if (run(fuse::MergeOp::First)) return 1;
  return 0;
}
```

Expected output (exact digits can vary by platform):

```
Sum  : wall_flux = 350
# fuse-plan v1
period=1 lockstep=yes instructions=4 buffers=4 coupler_elements=4 regions=0
buffer 0   %0  elements=4  components=1  region=r0  owner=component  persistent=no  "convection.q_conv"
buffer 1   %1  elements=4  components=1  region=r0  owner=component  persistent=no  "radiation.q_rad"
buffer 2   %2  elements=4  components=1  region=r0  owner=coupler  persistent=no  "solid.wall_flux/merge"
buffer 3   %3  elements=4  components=1  region=r0  owner=component  persistent=no  "solid.wall_flux"
  0  t=0  Source  node=0  out=%0
  1  t=0  Source  node=1  out=%1
  2  t=1  Merge  node=2  in=[%0,%1]  out=%2  {reason=Sum}
  3  t=1  Sink  node=3  in=[%2]  out=%3
Mean : wall_flux = 175
First: wall_flux = 300
```

## What it compiles to

One `Merge` operator consuming every writer, placed where the writers meet and
before the field is delivered, as the plan above shows. With more than three writers the
lowering emits a left fold of pairwise `Merge` nodes with the same result.

The combined value is timed at its most recent contribution: if one field transfer is
lagged and another is not, the result is available at the unlagged field transfer's
step, and the dependence on that field transfer is preserved in the schedule. If the writers'
time anchors cannot be reconciled, lowering refuses with `E-IR-MERGE-ANCHOR`.

## Diagnostics

| Code | Meaning | What to do |
|---|---|---|
| `E-MULTI-WRITER` | a field has two or more writers and no `merge()` | declare the rule, or retarget all but one channel |
| `E-MERGE-FIELD` | `merge()` was called with a `FieldId` the graph does not have | pass the id `add_field()` returned |
| `W-MERGE-UNUSED` | a rule is declared for a field written by fewer than two channels | usually a channel was renamed or retargeted and the rule was left behind; remove it or fix the channel |
| `E-IR-MERGE-ANCHOR` | the writers' time levels are ambiguous | check the channels' `lag()` and accumulation windows |

The warning reads:

```
warning W-MERGE-UNUSED: field 'wall_flux' declares a Sum merge but is written by 1 channel(s);
the merge combines nothing
```

It is a warning, not an error: the graph still runs and still means what it says.

## See also

- [](fluent-api.md) for how `validate()` reports every defect at once
- [](transfer-maps.md) for field transfers and remap weights
- [](ir.md) for the `Merge` operator in the dataflow graph
