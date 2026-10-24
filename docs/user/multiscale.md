# Multiscale coupling

This page is for anyone coupling a macro-scale model to many micro-scale instances (representative
volume elements, subgrid columns, particles) or preparing a learned model to stand in for a
component. It explains what multiscale coupling means in FUSE, how to build such a coupling from the
features that exist, and what the declared multiscale contract in `fuse/model/multiscale.hpp` is
and is not.

The short version: FUSE has no dedicated multiscale runtime. The contract types are declared and
reviewable, but every entry point that would register an instance set, a closure, an overlap or a
sample pool, or substitute a surrogate, refuses with a named code. A multiscale problem is coupled
today with ordinary components, channels, multirate accumulation and iteration regions.

## What multiscale coupling means here

A macro model advances a coarse state on a coarse grid and a coarse time step. Each macro cell owns
N micro instances, each with its own state and its own, much shorter, time loop. The micro
instances need the macro state (a strain, a temperature, a boundary condition) distributed down to
them; the macro model needs a *closure* (a stress, a flux, an effective coefficient) reduced up from
them.

```text
            macro model            one state per macro cell, step dt_M
          /     |      \
   distribute   |     closure      down: macro state, held over the micro steps
       |        |        |         up:   reduce N instance results to one value
   +-------+ +-------+ +-------+
   | inst  | | inst  | | inst  |   N micro instances per cell, step dt_m << dt_M
   | 1..N  | | 1..N  | | 1..N  |   each with its own state and time loop
   +-------+ +-------+ +-------+
        \       |       /
         +-- sample pool --+       instance results kept for reuse: training
                                   and validating an emulator
```

Four things make this more than a pair of channels:

- **Instance sets.** The N instances share one description and run independently; the macro
  scale sees only their aggregate.
- **Closures and distribution.** The reduction up and the distribution down are a pair. A closure
  that is not the adjoint of its own distribution silently breaks conservation across scales.
- **Coverage.** Each instance accounts for some share of its macro cell. Shares that do not sum to
  one are the multiscale analogue of a non-conservative remap, so they have to be accounted for,
  not assumed.
- **Sample pools.** Instance results kept for reuse are what make an emulator possible: it is
  trained on, and validated against, the pool. A learned closure then needs a contract before it can
  replace the instances.

## Build a multiscale coupling with existing features

### N instances as one component

Represent the instance set as one component that owns N values per macro cell, either as a field
with `components = N` on the macro region or as a field on a finer region with N cells per macro
cell. The component advances every instance in its own loop and performs the closure itself, writing
one value per macro cell to a second provided field:

```cpp
constexpr int kInstances = 16;
const auto micro_state = g.add_field(micro, fuse::FieldSpec{
    .name = "rve_stress", .region = macro_cells,
    .components = kInstances,                        // one value per instance per cell
    .units = fuse::units::kPascal,
    .conservation = fuse::Conservation::Intensive,
    .intent = fuse::Intent::Provides});
// The component itself reduces the instances; the macro side reads one value per cell.
const auto closure_out = g.add_field(micro, fuse::FieldSpec{
    .name = "mean_stress", .region = macro_cells,
    .units = fuse::units::kPascal,
    .conservation = fuse::Conservation::Intensive,
    .intent = fuse::Intent::Provides});
```

```cpp
class RveSet final : public fuse::Component {
 public:
  explicit RveSet(std::size_t cells, int instances)
      : cells_(cells), n_(instances), state_(cells * instances, 0.0), mean_(cells, 0.0) {}
  std::string name() const override { return "rve"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg, const fuse::CouplingGraph& g) override {
    const auto id = g.find_component("rve");
    for (const fuse::FieldId f : g.component(*id).fields) {
      const fuse::FieldSpec& spec = g.field(f);
      fuse::Result<void> r = spec.name == "rve_stress"
          ? reg.bind(f, fuse::FieldRef{state_.data(),
                fuse::Layout{.stride = static_cast<std::size_t>(n_), .component_stride = 1,
                             .count = cells_, .components = n_},
                fuse::MemorySpace::Host})
          : reg.bind(f, fuse::FieldRef{mean_.data(), fuse::Layout{.stride = 1, .count = cells_},
                                       fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    for (std::size_t c = 0; c < cells_; ++c) {
      double* inst = &state_[c * n_];
      for (int i = 0; i < n_; ++i) inst[i] = micro_step(inst[i], ctx);   // each instance: its own loop
      mean_[c] = std::accumulate(inst, inst + n_, 0.0) / n_;             // the closure, in place
    }
    return fuse::Result<void>::success();
  }
 private:
  static double micro_step(double s, const fuse::StepContext& ctx) { return s + ctx.dt().to_double(); }
  std::size_t cells_; int n_;
  std::vector<double> state_, mean_;
};
```

The instance-level field is optional; it is useful when another component (a diagnostic, a
sample-pool writer) wants the individual results. Only the closure field needs a channel to the
macro model. Coverage weighting lives in the component's own reduction.

:::{tip}
Keep the instance count static. The compiled plan sizes every buffer once at `initialize()`, so a
component that changes N between steps cannot be scheduled. Size for the largest N and mask unused
instances inside the component.
:::

### Scale separation in time

Give the micro component its own short step and the macro component its long one
(`ComponentDecl::interval`), and let FUSE sample and reduce across the rate difference. Use
`AccumOp::Integral` for a flux-like closure and `AccumOp::Mean` for a state-like one; the macro
model then receives the quantity integrated or averaged over exactly its own step:

```cpp
// micro steps 1/1000, macro steps 1/10 (set on ComponentDecl::interval)
const auto up = g.add_channel("micro_to_macro", micro_flux, macro_flux_in);
g.channel(up).accumulate(fuse::AccumOp::Integral);   // a flux: integrate over the macro step
const auto down = g.add_channel("macro_to_micro", macro_state, micro_state_in);
g.channel(down).lag(1).allow_time_level_mismatch("macro state is held over the micro substeps");
```

The two channels form a cycle, so one direction is lagged (explicit coupling) or the pair goes into
an iteration region (implicit coupling), as for any two-way exchange ([](concepts.md#time)). When
the two steps are incommensurate, couple them over a window instead ([](coupling-windows.md)).

### A learned closure as a component

A surrogate that replaces the instance set is an ordinary component: it binds the same fields,
advances on the same step, and is subject to the same conservation checks and iteration-region
rules (rollback, determinism) as the physics it replaces. Swapping the physics component for the
surrogate changes nothing in the graph. [](surrogates.md) covers writing one, including Python
models.

### Conservation across scales

Declare the closure quantity `Extensive` on both ends and the channel conservative. `validate()`
refuses an `Extensive` target without a conservative transfer (`E-CONN-CONSERVATION`), and
`lower()` emits a `Constrain` operator that checks the transferred integral on every exchange
(`RunStats::conservation_checks`, `conservation_violations`, `worst_conservation_error`):

```cpp
const auto q_micro = g.add_field(micro, fuse::FieldSpec{
    .name = "q", .region = micro_cells, .units = fuse::units::kWatt,
    .conservation = fuse::Conservation::Extensive, .intent = fuse::Intent::Provides});
const auto q_macro = g.add_field(macro, fuse::FieldSpec{
    .name = "q_in", .region = macro_cells, .units = fuse::units::kWatt,
    .conservation = fuse::Conservation::Extensive, .intent = fuse::Intent::Requires});
const auto ch = g.add_channel("q_up", q_micro, q_macro);
g.channel(ch).conservative().accumulate(fuse::AccumOp::Integral);
```

This checks that the sum the macro model receives equals the sum the instances produced, over space
and over the macro step. It does not check coverage: the component's own reduction is responsible
for weighting each instance by its share of the cell.

## The declared multiscale contract

`include/fuse/model/multiscale.hpp` declares the types a dedicated multiscale runtime would consume.
They are reference material for designing a coupling, and one of them, `SurrogateContract`, has a
working reviewer you can use now.

### Types

`InstanceSetSpec`: N instances of one component description.

| Field | Meaning |
|---|---|
| `name` | The set's name |
| `component` | The `ComponentId` every instance is a copy of |
| `count` | The instance count, static: a dynamic count would make the compiled plan data-dependent |

`ClosureSpec`: the reduction from instance results to a macro quantity.

| Field | Meaning |
|---|---|
| `name` | The closure's name |
| `kind` | `ClosureKind::WeightedMean` (coverage-weighted mean), `Sum` (extensive quantities) or `Custom` (a caller-supplied reduction) |
| `instance_field` | The per-instance `FieldId` being reduced |
| `macro_field` | The macro `FieldId` receiving the result |

`OverlapSpec`: each instance's share of the macro cell.

| Field | Meaning |
|---|---|
| `coverage` | `std::vector<Rational>`, one share per instance. Exact rationals, so the sum can be checked for equality with one rather than compared to a tolerance |
| `total()` | The exact sum; refuses `E-BASE-RATIONAL-OVERFLOW` when it is not representable |

`SamplePoolSpec`: instance results kept for reuse.

| Field | Meaning |
|---|---|
| `name` | The pool's name |
| `capacity` | Entries to keep; 0 keeps everything, which is a memory decision to make explicitly rather than a default |

`SurrogateContract`: what a learned model states before it may stand in for a component.

| Field | Meaning |
|---|---|
| `name` | The surrogate's name |
| `applicability` | `std::vector<FieldSemantics>` describing the inputs over which the model claims validity. Empty means no claim, which is a reason to refuse the model |
| `ood_detector` | How out-of-distribution input is detected at run time. Empty means it is not detected |
| `carries_hidden_state` | A model with hidden state cannot be rolled back by restoring its inputs, so an iteration region needs a checkpoint hook from the model |
| `determinism` | `DeterminismLevel::Bitwise` (identical on identical hardware and across the rank counts the coupler may use; the only level usable inside a fixed-point iteration), `RunToRun` (reproducible on one configuration) or `Nondeterministic` (the default) |
| `derivative` | `DerivativeKind::None` (the default), `Forward` (a directional derivative, JVP), `Reverse` (an adjoint, VJP) or `Both` |

The same determinism levels and hidden-state flag appear in the capability bits a `Component`
returns (`fuse/model/capabilities.hpp`, `make_capabilities(rollback, determinism, hidden_state)`),
so a contract can be compared with what a component declares.

### Entry points that refuse

These functions exist so that a graph using them is rejected rather than run as if the feature
were inert. Every call returns a `Failure`:

| Function | Refuses with |
|---|---|
| `declare_instance_set(const InstanceSetSpec&)` | `E-MULTISCALE-NOT-IMPLEMENTED` |
| `declare_closure(const ClosureSpec&)` | `E-MULTISCALE-NOT-IMPLEMENTED` |
| `check_overlap(const OverlapSpec&)` | `E-MULTISCALE-NOT-IMPLEMENTED`, even for coverage that sums to one; the message states the instance count and the sum |
| `declare_sample_pool(const SamplePoolSpec&)` | `E-MULTISCALE-NOT-IMPLEMENTED` |
| `substitute_surrogate(ComponentId, const SurrogateContract&)` | `E-SURROGATE-CONTRACT` when the contract review has errors (the review text is in the message), otherwise `E-SURROGATE-NOT-IMPLEMENTED` |

The operator IR reserves `OpKind::Map`, `OpKind::Reduce` and `OpKind::Closure` for the same purpose;
`schedule()` refuses a DAG containing one of them with `E-SCHED-MULTISCALE`. `lower()` never emits
them from a graph.

### Review a surrogate contract

`fuse::review_surrogate_contract(const SurrogateContract&)` returns `Diagnostics` and does not
refuse. Use it while writing a contract: it reports every gap at once.

| Code | Severity | Raised when |
|---|---|---|
| `E-SURROGATE-UNNAMED` | error | `name` is empty |
| `E-SURROGATE-NO-APPLICABILITY` | error | `applicability` is empty |
| `E-SURROGATE-NO-OOD` | error | `ood_detector` is empty |
| `W-SURROGATE-NONDETERMINISTIC` | warning | `determinism == Nondeterministic`; such a model can serve an explicit coupling but not an iteration region, whose convergence test cannot tell noise from a residual |
| `W-SURROGATE-HIDDEN-STATE` | warning | `carries_hidden_state`; a rollback needs an explicit checkpoint hook from the model |

`derivative` is recorded, not checked. This program reviews an incomplete and a complete contract,
then shows the refusals:

```cpp
#include <fuse/model/multiscale.hpp>

#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  // An incomplete contract: nothing but a name.
  fuse::SurrogateContract draft;
  draft.name = "wall-flux-emulator";
  const fuse::Diagnostics first = fuse::review_surrogate_contract(draft);
  std::cout << "--- draft: errors=" << first.has_errors() << "\n" << first.to_string();

  // A complete contract.
  fuse::SurrogateContract ready;
  ready.name = "wall-flux-emulator";
  fuse::FieldSemantics input;
  input.id = "wall_temperature";
  input.units = fuse::units::kKelvin;
  ready.applicability = {input};
  ready.ood_detector = "mahalanobis-on-wall-temperature";
  ready.determinism = fuse::DeterminismLevel::Bitwise;
  ready.derivative = fuse::DerivativeKind::Forward;
  const fuse::Diagnostics second = fuse::review_surrogate_contract(ready);
  std::cout << "--- ready: errors=" << second.has_errors() << " size=" << second.size() << "\n"
            << second.to_string();

  // Substitution and overlap accounting are not available: both refuse.
  const auto sub = fuse::substitute_surrogate(fuse::ComponentId{0}, ready);
  std::cout << "--- substitute_surrogate: " << sub.error().code << "\n"
            << sub.error().message << "\n";
  fuse::OverlapSpec overlap;
  overlap.coverage = {fuse::Rational{1, 4}, fuse::Rational{1, 4}, fuse::Rational{1, 2}};
  const auto ov = fuse::check_overlap(overlap);
  std::cout << "--- check_overlap: " << ov.error().code << "\n" << ov.error().message << "\n";
  return 0;
}
```

Expected output:

```text
--- draft: errors=1
error E-SURROGATE-NO-APPLICABILITY: the contract states no applicability region, so there is nothing against which an input can be judged in or out of distribution [surrogate 'wall-flux-emulator']
error E-SURROGATE-NO-OOD: no out-of-distribution detector is named; a surrogate that cannot report when it is extrapolating is not substitutable for a component that would have converged or failed [surrogate 'wall-flux-emulator']
warning W-SURROGATE-NONDETERMINISTIC: a nondeterministic surrogate cannot be used inside an iteration region: the convergence test compares successive iterates, and noise in the model is indistinguishable from a residual [surrogate 'wall-flux-emulator']
--- ready: errors=0 size=0
--- substitute_surrogate: E-SURROGATE-NOT-IMPLEMENTED
surrogate substitution is declared but not implemented ('wall-flux-emulator'). The contract is fixed so the descriptor does not change when it lands; until then a graph using it is refused rather than run as if the feature were inert.
--- check_overlap: E-MULTISCALE-NOT-IMPLEMENTED
overlap accounting is declared but not implemented ('3 instances summing to 1'). The contract is fixed so the descriptor does not change when it lands; until then a graph using it is refused rather than run as if the feature were inert.
```

### Why the contract is declared ahead of an implementation

The graph descriptor and the C ABI are versioned binary interfaces shared by every front end
(C++, YAML, C, Fortran, Python). A type added to them later is a version change for every
consumer. Declaring the multiscale and surrogate shapes now, and refusing to act on them, keeps the
descriptor stable and makes a graph that depends on an unavailable feature fail by name instead of
running with the closure silently absent.

## Limitations

- Instance sets, closures, overlap accounting, sample pools and surrogate substitution are not
  supported; the entry points refuse with `E-MULTISCALE-NOT-IMPLEMENTED`, `E-SURROGATE-CONTRACT` or
  `E-SURROGATE-NOT-IMPLEMENTED` as listed above.
- `OpKind::Map`, `Reduce` and `Closure` cannot be scheduled: `E-SCHED-MULTISCALE`.
- The instance count of a component-based instance set is fixed for the run.

## See also

- [](surrogates.md): writing a learned model as a component.
- [](coupling-windows.md): coupling incommensurate time steps.
- [](concepts.md): multirate accumulation, lags and iteration regions.
- [](component-interface.md): the `Component` contract and capability bits.
- [](ir.md): the operator kinds, including the reserved ones.
