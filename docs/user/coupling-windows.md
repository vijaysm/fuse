# Coupling two clocks over a window, not a tick

An iteration region's default form couples components on one clock: every member advances by the
same interval, and the region's sweep loop rolls everyone back to the same instant. A *coupling
window* is the opt-in alternative for components whose steps do not divide one another: the group
is iterated over a time interval as a whole, each member sub-stepping on its own grid against its
partners' trajectories from the previous sweep. This page is for anyone coupling components whose
steps are not multiples of each other, and for anyone who wants waveform iteration between
components that are.

Two components rarely share a clock in practice. A structural solver stepping every 13 s and a
thermal solver stepping every 7 s do not coincide again until 91 s, and refusing to couple them
until then is not an option. A coupling window couples every 13 s (the slower component's own
step) and maps the faster component's steps onto that window exactly, never by waiting for a common
multiple.

## The smallest working example

Two in-process components, steps 1/2 and 1/3, iterated over a window of length 1 through
`schedule()` and the `Runner`. The complete program is at the end of this page; this is the part
that declares the window:

```cpp
fuse::CouplingWindowScope scope;
scope.explicit_interval = fuse::Rational{1};          // or: scope.window_component = "A";
scope.order = 2;
scope.policies = {{"A", fuse::DtPolicy::Adjustable}, {"B", fuse::DtPolicy::Adjustable}};
scope.iteration = fuse::WaveformIteration::Jacobi;

fuse::IterationRegionDecl decl;
decl.name = "ab_window";
decl.channels = {fwd, back};                          // a2b and b2a
decl.max_iterations = 50;
decl.convergence.add(fuse::Converged::Absolute(1e-12, "a_out"));
decl.convergence.add(fuse::Converged::Absolute(1e-12, "b_out"));
decl.window_scope = scope;
g.add_iteration_region(decl);
```

Expected output (exact digits can vary by platform):

```text
window 0: converged=1 sweeps=24 last residual=4.61432e-13
window 1: converged=1 sweeps=23 last residual=9.54735e-13
window 2: converged=1 sweeps=21 last residual=7.18831e-13
period length 1, A advanced 136 times, B 204 times
a_out=2.49956 b_out=2.99962 windows_not_converged=0
```

A takes two substeps per window and B three; 68 sweeps over three windows give 136 and 204
`advance()` calls. The coupling period of the plan is the window length, not the least common
multiple of the member steps.

## Declare a window

There is no builder method for windows. Fill a `fuse::CouplingWindowScope`
(`fuse/model/iteration.hpp`), set it on `IterationRegionDecl::window_scope`, and add the region with
`CouplingGraph::add_iteration_region(decl)`. `g.iterate(...)` returns an `IterationBuilder` that has
no window setter, so a window-scoped region is always declared through `IterationRegionDecl`.

| Field | Type | Meaning |
|---|---|---|
| `window_component` | `std::optional<std::string>` | The window is as long as this component's own `ComponentDecl::interval` |
| `explicit_interval` | `std::optional<Rational>` | The window length stated directly |
| `policies` | `std::vector<WindowMemberPolicy>` | One `{component, policy}` entry per member; `policy` is `DtPolicy::Fixed` or `DtPolicy::Adjustable` (default `Adjustable`) |
| `order` | `int`, default 2 | Order of the temporal interpolant and extrapolant at the window boundary; must be at least 2 |
| `iteration` | `WaveformIteration`, default `Jacobi` | Which waveform-relaxation sweep the window runs (`Jacobi` or `GaussSeidel`) |

Exactly one of `window_component` and `explicit_interval` must be set. `fuse::window_length(g,
scope)` resolves whichever is set and refuses `E-SCHED-WINDOW-AMBIGUOUS` when both or neither is.

The *members* of a window are the components that own either end of a channel in the region
(`fuse::window_region_members(g, region)`, `fuse/schedule/window.hpp`). Every member must appear in
`policies` exactly once. `scope.policy_for("A")` returns the policy for a named member, or
`std::nullopt`.

The declaration can be checked before anything is compiled:

```cpp
#include <fuse/schedule/window.hpp>
fuse::Diagnostics diags;
const auto ok = fuse::validate_coupling_window(g, region, scope, /*waveform_capacity=*/4096, &diags);
```

`schedule()` runs the same check on every window-scoped region. Without the opt-in, a region whose
channels span two clocks is refused with `E-SCHED-REGION`.

### Windows in YAML

A window is an iteration region with a `window_scope` block. This is the region from
`examples/brusselator/participants/reference_mpmd_3.yaml`, where U steps every 1/500 and V every
3/1000:

```{literalinclude} ../../examples/brusselator/participants/reference_mpmd_3.yaml
:language: yaml
:start-at: "iteration_regions:"
```

The keys mirror the struct: `window_component` or `explicit_interval` (a rational `"N/D"`),
`order`, `policies` (a sequence of `{component, policy}` with `policy` `Fixed` or `Adjustable`), and
`iteration` (`Jacobi` or `GaussSeidel`). The parser checks neither the exclusivity of the two length
keys nor that every member has a policy; `schedule()` does. See [](yaml.md).

## Fixed or Adjustable, per member

Each member declares how its own step relates to the window boundary. The policy is per component,
never per region, and a window may mix the two.

**Adjustable** shortens its last substep so that it lands exactly on the window end. A 7 s step
inside a 13 s window runs 7, then 6. This member never straddles and never needs extrapolation.

**Fixed** keeps its own step. A substep that would cross the window end is taken in full (the
*straddling step*), using the partner's trajectory extrapolated past the window end with the
window's `order`. The whole substep is provisional: at the start of the next window it is rolled
back and recomputed with the partner's converged data for the part inside the finished window and a
fresh extrapolation for the part beyond it. A Fixed step that divides the window never straddles.

:::{note}
A `Fixed` member must support rollback (`kCapabilityRollback`, a real
`save_state()`/`restore_state()` pair), because its straddling step is replayed.
`WindowCoupler::create()` refuses a Fixed member without it with `E-RT-WINDOW-ROLLBACK`; the Runner
reports the missing capability at `initialize()` as `E-ITER-NOROLLBACK`, as for any region member.
:::

## Jacobi or Gauss-Seidel sweeps

`CouplingWindowScope::iteration` selects the classical waveform-relaxation scheme. It is set per
region: every member of one window sweeps under the same scheme.

**Jacobi** (the default). Every transfer inside the window is loop-carried regardless of
`Channel::lag`: each member reads each partner's *previous* iterate for the whole sweep. No `.lag()`
is needed to make the region schedulable. `fuse::effective_channel_lag(g, ch)` reports 1 for every
channel of a Jacobi window, and the resulting time-level mismatch is authorized automatically
(`fuse::channel_time_mismatch_is_authorized`).

**GaussSeidel.** Only the channels you declared with `.lag(...)` are loop-carried; every other
channel must be delivered earlier in the same sweep. A two-way window with no lagged channel closes
an unbroken cycle and is refused: `E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG` at schedule time,
`E-RT-WINDOW-GAUSSSEIDEL-NOLAG` from `WindowCoupler::create()`. Lag one direction:

```cpp
g.channel(back).lag(1).allow_time_level_mismatch("resolved by the window's own sweep");
scope.iteration = fuse::WaveformIteration::GaussSeidel;
```

In the direct driver, a member that provides both a lagged and an unlagged outgoing channel is
refused with `E-RT-WINDOW-GAUSSSEIDEL-MIXED-LAG`, because the provider's delivered waveform cannot
be both this sweep's fresh value and last sweep's at once. A GaussSeidel window may use any
accelerator.

## What a partner receives

`Component::advance()` hands the implementer one value per `Requires` field, valid at
`ctx.t_begin`, not a trajectory. Inside a window that value is the sender's *mean* over exactly the
receiving substep (`Waveform::integrate(...) / dt`), not a point sample, the same convention an
`AccumOp::Mean` channel uses. The sender's trajectory over the window is represented with a linear
in-step interpolant, which `integrate()` handles exactly, so the substep means handed to different
consumers over one sender window sum to the sender's whole-window integral to round-off: the exchange
conserves the time integral by construction.

The interpolation and extrapolation order at the window boundary is `scope.order`
(`WindowCouplerConfig::order` in the direct driver; `WindowMember::order` overrides it per member
when non-zero). Extrapolation fits a polynomial of degree `order - 1` to the last `order` committed
boundary values, reducing the degree while fewer values exist.

## Subcycling and sample capacity

Each member steps on its own grid inside the window; the window never waits for a common multiple
of the member intervals. The number of trajectory samples a member needs per window is
`fuse::window_sample_capacity(window_length, interval, policy)`: `ceil(L / interval) + 1`, plus one
more for a `Fixed` member. A member whose interval is small enough that it would need more samples
than the waveform capacity is refused with `E-SCHED-WINDOW-CAPACITY` (the Runner path; capacity
4096 by default in `validate_coupling_window`) or `E-RT-WINDOW-CAPACITY` (the direct driver;
`WindowCouplerConfig::waveform_capacity`, default 64).

`fuse::window_sample_times(...)` and `fuse::window_sample_shift(...)` (`fuse/model/coupling_graph.hpp`)
reconstruct a provider's sample times from the graph alone, which is what a consumer on another rank
or in another executable uses to interpret a straddling provider's samples.

## Convergence over a window

Convergence and acceleration act on the whole window's exchanged trajectory, every substep sample
and not only the window-end value, exactly as [](convergence.md) describes for a single-clock
region. `MeasureKind::ResidualRelative` takes the window's first sweep as its reference.

On the Runner path, measures name fields (`"a_out"` above), and every channel of the region must be
measured, as for any iteration region. In the direct driver, measures name *channels*
(`WindowChannel::name`); a measure that names no channel is refused with
`E-RT-WINDOW-CONVERGENCE-FIELD`.

Across ranks, the residual and reference norms of each block are summed over the ranks that host it
before the test is applied, so every rank takes the same decision. A block that no rank hosts is
refused with `E-RT-WINDOW-UNMEASURED` rather than counted as converged.

When a window reaches `max_iterations` without meeting its criterion:

- the Runner counts it in `RunStats::windows_not_converged` and continues (the default), or fails
  the period with `E-RT-NOCONVERGE` when `RunnerConfig::abort_on_nonconvergence` is `true`;
- `WindowCoupler::run_window()` returns `E-RT-WINDOW-NOCONVERGE` (the default
  `WindowCouplerConfig::abort_on_nonconvergence = true`), or commits the window and records the
  outcome in `WindowIterationStats::converged` when it is `false`.

:::{warning}
The two drivers default differently: the Runner continues past an unconverged window and records
it; `WindowCoupler` aborts. A run that ignores `windows_not_converged` reports an unconverged answer
as a converged one.
:::

## Two ways to run a window

### Through `schedule()` and the Runner

Declare the region as above, then compile and run as for any graph (see [](runtime.md)).
`schedule()` recognizes a region whose `window_scope` is set, replaces each member's clock with the
window length, and emits one window block per period; the `Runner` sweeps it to convergence. Window
members, channels and the convergence criterion all come from the graph, so the same declaration
runs in one process, across MPI ranks, or across separately launched executables
([](multiple-executables.md)). The Brusselator example runs its window on this path when started
with `--driver runner`, and always when split into participants.

`runner.iteration_stats()` holds one `IterationStats{iterations, converged, residual_history}` per
region for the most recent period.

### With `WindowCoupler` directly

`fuse::WindowCoupler` (`fuse/runtime/window_coupler.hpp`) drives a window without a graph, a
schedule or a `Runner`. You hand it the components, raw pointers to their exchanged storage, and a
configuration:

```cpp
fuse::WindowMember mA;
mA.component = &A; mA.name = "A"; mA.interval = fuse::Rational{1, 2};
mA.policy = fuse::DtPolicy::Adjustable;
mA.provided = A.provided_ptr(); mA.required = A.required_ptr(); mA.length = 1;
fuse::WindowMember mB = mA;
mB.component = &B; mB.name = "B"; mB.interval = fuse::Rational{1, 3};
mB.provided = B.provided_ptr(); mB.required = B.required_ptr();

const std::vector<fuse::WindowChannel> channels = {
    {.provider = 0, .consumer = 1, .lagged = false, .name = "a2b"},
    {.provider = 1, .consumer = 0, .lagged = false, .name = "b2a"}};

fuse::WindowCouplerConfig cfg;
cfg.window_length = fuse::Rational{1};
cfg.max_iterations = 50;
cfg.convergence.add(fuse::Converged::Absolute(1e-12, "a2b"));   // measures name channels here
cfg.convergence.add(fuse::Converged::Absolute(1e-12, "b2a"));

auto created = fuse::WindowCoupler::create({mA, mB}, channels, cfg);
fuse::WindowCoupler& coupler = *created.value();
for (int w = 0; w < 3; ++w) {
  const auto r = coupler.run_window();            // one window, swept to convergence
  const fuse::WindowIterationStats& st = coupler.stats();
}
```

Expected output (exact digits can vary by platform), identical to the Runner run above:

```text
window 0: converged=1 sweeps=24 next start=1
window 1: converged=1 sweeps=23 next start=2
window 2: converged=1 sweeps=21 next start=3
a_out=2.49956 b_out=2.99962
```

`WindowMember` fields: `component`, `name`, `interval` (default 1), `policy` (default
`Adjustable`), `order` (0 means use the config's), `provided`/`required` (non-owning pointers to
`length` doubles that `advance()` writes and reads), `length` (default 1). The pointers must stay
valid and fixed for the coupler's lifetime. `WindowChannel` fields: `provider` and `consumer`
(indices into the member list), `lagged` (read only under `GaussSeidel`), `name` (what a
convergence measure refers to).

`WindowCouplerConfig`:

| Field | Default | Meaning |
|---|---|---|
| `window_length` | 1 | Length of every window |
| `max_iterations` | 50 | Sweeps per window before giving up |
| `convergence` | empty | Measures naming channels |
| `acceleration` | `{.kind = None}` | Any `AccelKind` |
| `iteration` | `Jacobi` | `Jacobi` or `GaussSeidel` |
| `order` | 2 | Default interpolation/extrapolation order |
| `waveform_capacity` | 64 | Samples per member per window |
| `channel_io` | null | A `WindowChannelIO` override for provider-to-consumer reads; null reads the provider's buffer directly |
| `abort_on_nonconvergence` | `true` | `run_window()` fails with `E-RT-WINDOW-NOCONVERGE` instead of recording the outcome |

`create()` returns `Result<std::unique_ptr<WindowCoupler>>`. `run_window()` advances every member
through the next window; windows must be run back to back, because a Fixed member's pending straddle
is resolved against the window that immediately precedes it, and there is no way to skip or replay
one. `stats()` returns the last window's `WindowIterationStats{iterations, converged,
residual_history}` and `next_window_start()` the start time of the next window.

The direct driver does no spatial remap and no transport: a consumer reads the provider's buffer
element for element (clamped to the provider's length). Use the Runner path when the two sides are
on different meshes or different ranks. `examples/gray_scott` and `examples/brusselator` (with the
default `--driver window`) use the direct driver.

## Straddling steps and limitations

- At most one Fixed member may straddle a given window. A second straddling member is refused with
  `E-RT-WINDOW-STRADDLE` on both paths.
- A Fixed step longer than the window (a single step that spans past the window end rather than
  merely failing to divide it) is refused by `WindowCoupler::create()` with `E-RT-WINDOW-LONGSTEP`.
  The Runner path accepts it and replays it like any other straddle.
- On the Runner path, a straddling Fixed provider whose channel to its consumer is not on the same
  ranks and the same region is refused with `E-RT-WINDOW-STRADDLE-REPUBLISH`. The replay publishes
  the provider's corrected value straight into the consumer's import storage, which is exact only
  when the channel moves no data.
- A component driven from outside the Runner ([](c-abi.md)) may be a window member but may not
  straddle: `E-RT-WINDOW-EXTERNAL-STRADDLE`.
- A window member's exported and imported fields must be contiguous with one value per element:
  `E-RT-WINDOW-LAYOUT` otherwise, and `E-RT-SIZE` when the registered storage does not match the
  plan's size for the field.
- A Newton region with `NewtonResidualPath::Direct` cannot be window-scoped:
  `E-NEWTON-DIRECT-WINDOW` at `initialize()`. The `Advance` path (the default) is supported; see the
  Brusselator example's `--accel newton`.
- A component or a channel may belong to only one window-scoped region: `E-SCHED-WINDOW-OVERLAP`.
- The Runner's field transfer through a window is an identity map between matching regions; a
  window whose two sides need a genuine spatial remap is not supported on either path.

## Error codes

Schedule time (`validate_coupling_window`, `window_length`, `schedule()`):

| Code | Meaning | Fix |
|---|---|---|
| `E-SCHED-WINDOW-EMPTY` | the region has no channels | add the channels the window exchanges |
| `E-SCHED-WINDOW-AMBIGUOUS` | both or neither of `window_component` and `explicit_interval` set | set exactly one |
| `E-SCHED-WINDOW-COMPONENT` | `window_component` or a policy names a component that does not exist or is not a member | use the names of components owning the region's channel endpoints |
| `E-SCHED-WINDOW-INTERVAL` | the window length is not positive | fix the interval |
| `E-SCHED-WINDOW-POLICY` | a member has no policy entry, or a policy names a non-member | one entry per member, no extras |
| `E-SCHED-WINDOW-ORDER` | `order < 2` | use 2 or more |
| `E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG` | `GaussSeidel` with no lagged channel | `.lag(1)` on one direction, or use `Jacobi` |
| `E-SCHED-WINDOW-SHORTEN` | an `Adjustable` member's interval is not positive | fix the member's `ComponentDecl::interval` |
| `E-SCHED-WINDOW-CAPACITY` | a member needs more samples per window than the waveform capacity | lengthen the member's step, shorten the window, or raise the capacity |
| `E-SCHED-WINDOW-OVERLAP` | a component or channel is in two window-scoped regions | merge the regions or remove the shared member |
| `E-SCHED-WINDOW-CAUSALITY` | a window member's import is not written at the sweep step it is read; a consistency check on the compiled plan | report it with the plan dump (`plan.to_string()`) |

`WindowCoupler::create()` and `run_window()`:

| Code | Meaning |
|---|---|
| `E-RT-WINDOW-EMPTY` | no members or no channels |
| `E-RT-WINDOW-NULL` | a member's `component`, `provided` or `required` is null with `length > 0` |
| `E-RT-WINDOW-INTERVAL` | a member's interval is not positive |
| `E-RT-WINDOW-ORDER` | `cfg.order < 2`, or a member's own non-zero `order < 2` |
| `E-RT-WINDOW-CHANNEL` | a channel index is outside the member list |
| `E-RT-WINDOW-ROLLBACK` | a `Fixed` member does not support rollback |
| `E-RT-WINDOW-GAUSSSEIDEL-NOLAG` | the unlagged channels form a cycle under `GaussSeidel` |
| `E-RT-WINDOW-GAUSSSEIDEL-MIXED-LAG` | one member provides both a lagged and an unlagged channel |
| `E-RT-WINDOW-LONGSTEP` | a `Fixed` member's step is longer than the window |
| `E-RT-WINDOW-STRADDLE` | more than one member would straddle |
| `E-RT-WINDOW-CONVERGENCE-FIELD` | a measure names no channel |
| `E-RT-WINDOW-NOCONVERGE` | `run_window()` exhausted `max_iterations` with `abort_on_nonconvergence` set |

Runner (`initialize()` and the period loop):

| Code | Meaning |
|---|---|
| `E-RT-WINDOW-STRADDLE`, `E-RT-WINDOW-STRADDLE-REPUBLISH`, `E-RT-WINDOW-EXTERNAL-STRADDLE` | the straddle rules above |
| `E-RT-WINDOW-LAYOUT`, `E-RT-SIZE` | a member's field layout or size does not match the window |
| `E-RT-WINDOW-CAPACITY` | a member needs more samples this window than the plan compiled for |
| `E-RT-WINDOW-UNMEASURED` | a window block is hosted by no rank, so its residual cannot be measured |
| `E-RT-WINDOW-COMM`, `E-RT-WINDOW-PEER` | an MPI call in the window's convergence agreement failed, or a peer rank reported a failure |
| `E-RT-NOCONVERGE` | a window did not converge and `RunnerConfig::abort_on_nonconvergence` is `true` |
| `E-RT-WINDOW-UNPLACED`, `E-RT-WINDOW-SWEEP`, `E-RT-WINDOW-UNWIRED`, `E-RT-WINDOW-EXTERNAL` | internal consistency checks on the compiled window; report them with the plan dump |

## Windows in the examples

- `examples/brusselator` ([](brusselator.md)): `--window L` sets the window length (default 0.02,
  with `--dt-u 0.002 --dt-v 0.003`), `--driver window|runner` selects the direct driver or the
  Runner path, `--accel` the accelerator, `--iteration-mode jacobi|gauss-seidel` the sweep.
  `--check` fails unless every window converged.
- `examples/gray_scott` ([](gray-scott.md)): `--ratio 1|m|p:q` sets U's interval (the window)
  relative to V's; `7:13` is the incommensurate case. `--policy fixed|adjustable` sets the
  `DtPolicy` of the finer member, so `--ratio 7:13 --policy fixed` exercises a straddling step.

## The complete Runner example

```cpp
// A coupling window between two in-process components with steps 1/2 and 1/3,
// run through schedule() and the Runner.
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/schedule/window.hpp>

#include <Kokkos_Core.hpp>
#include <iostream>

namespace {

// out = a + k * in, with the rollback pair a window member needs.
class LinearBlock final : public fuse::Component {
 public:
  LinearBlock(std::string name, double a, double k) : name_(std::move(name)), a_(a), k_(k) {}
  std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto id = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*id).fields) {
      double* p = g.field(f).intent == fuse::Intent::Provides ? &out_ : &in_;
      auto r = reg.bind(f, fuse::FieldRef{p, fuse::Layout{.stride = 1, .count = 1},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    ++advances;
    out_ = a_ + k_ * in_;   // in_ holds the partner's mean over this substep
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { s_out_ = out_; s_in_ = in_; return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { out_ = s_out_; in_ = s_in_; return fuse::Result<void>::success(); }
  std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }

  double out() const { return out_; }
  int advances = 0;

 private:
  std::string name_;
  double a_, k_;
  double out_ = 0.0, in_ = 0.0, s_out_ = 0.0, s_in_ = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  namespace u = fuse::units;

  fuse::CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m, .dimension = 1});
  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "A", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1, 2}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "B", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1, 3}});
  const auto a_out = g.add_field(a, fuse::FieldSpec{.name = "a_out", .region = r, .units = u::kKelvin,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(a, fuse::FieldSpec{.name = "a_in", .region = r, .units = u::kKelvin,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{.name = "b_out", .region = r, .units = u::kKelvin,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(b, fuse::FieldSpec{.name = "b_in", .region = r, .units = u::kKelvin,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Requires});
  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);

  // The window: one unit of time, both members Adjustable, Jacobi sweeps.
  fuse::CouplingWindowScope scope;
  scope.explicit_interval = fuse::Rational{1};
  scope.order = 2;
  scope.policies = {{"A", fuse::DtPolicy::Adjustable}, {"B", fuse::DtPolicy::Adjustable}};
  scope.iteration = fuse::WaveformIteration::Jacobi;

  fuse::IterationRegionDecl decl;
  decl.name = "ab_window";
  decl.channels = {fwd, back};
  decl.max_iterations = 50;
  decl.convergence.add(fuse::Converged::Absolute(1e-12, "a_out"));
  decl.convergence.add(fuse::Converged::Absolute(1e-12, "b_out"));
  decl.window_scope = scope;
  g.add_iteration_region(decl);

  if (const auto d = fuse::validate(g); d.has_errors()) { std::cerr << d.to_string(); return 1; }
  if (const auto ok = fuse::validate_coupling_window(g, g.iteration_region(fuse::IterationRegionIdx{0}), scope); !ok.ok()) {
    std::cerr << ok.error().code << ": " << ok.error().message << "\n"; return 1;
  }

  fuse::SerialMeshBackend backend;
  if (const auto s = backend.load(g.mesh(m)); !s.ok()) { std::cerr << s.error().message; return 1; }
  const auto lowered = fuse::lower(g);
  if (!lowered.ok()) { std::cerr << lowered.diags.to_string(); return 1; }
  const fuse::ClockSet clocks{g};
  const auto sched = fuse::schedule(g, lowered.dag, clocks,
                                    fuse::make_region_sizes(g, lowered.dag, backend));
  if (!sched.ok()) { std::cerr << sched.diags.to_string(); return 1; }

  fuse::ComponentSet components;
  components.add(std::make_unique<LinearBlock>("A", 1.0, 0.5));
  components.add(std::make_unique<LinearBlock>("B", 2.0, 0.4));
  fuse::FieldRegistry registry;
  for (auto& c : components) {
    if (const auto s = c.register_fields(registry, g); !s.ok()) { std::cerr << s.error().message; return 1; }
  }

  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  if (const auto s = runner.initialize(components, registry); !s.ok()) {
    std::cerr << s.error().code << ": " << s.error().message << "\n"; return 1;
  }
  for (int w = 0; w < 3; ++w) {
    if (const auto s = runner.run_period(components); !s.ok()) {
      std::cerr << s.error().code << ": " << s.error().message << "\n"; return 1;
    }
    const fuse::IterationStats& st = runner.iteration_stats().front();
    std::cout << "window " << w << ": converged=" << st.converged
              << " sweeps=" << st.iterations
              << " last residual=" << st.residual_history.back() << "\n";
  }
  const auto* A = dynamic_cast<LinearBlock*>(components.find("A"));
  const auto* B = dynamic_cast<LinearBlock*>(components.find("B"));
  std::cout << "period length " << sched.plan.period().to_string()
            << ", A advanced " << A->advances << " times, B " << B->advances << " times\n"
            << "a_out=" << A->out() << " b_out=" << B->out()
            << " windows_not_converged=" << runner.stats().windows_not_converged << "\n";
  return 0;
}
```

The fields are `Intensive` because the channels are plain copies; an `Extensive` target without
`.conservative()` is refused by `validate()` with `E-CONN-CONSERVATION`.

## See also

- [](convergence.md): convergence measures, norms and Newton over a window.
- [](runtime.md): the Runner, its configuration and statistics.
- [](multiple-executables.md): a windowed coupling split across executables.
- [](brusselator.md) and [](gray-scott.md): the worked window examples.
- [](yaml.md): the `window_scope` block.
