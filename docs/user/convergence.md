# Deciding when a coupled iteration has converged

This page is for anyone writing an iteration region: how to state the
convergence criterion, which residual measure to pick, how to read the residual
history a run leaves behind, and which accelerator to select. The Newton
variant is summarised at the end with a link to its own material.

An iteration region (an implicit coupling group) repeats the exchange between
its member components until the residual over the exchanged interface values
is small. Three things are yours to decide: what "small" means for each
exchanged field, how many sweeps you are willing to pay, and which accelerator
drives the fixed point. Every one of them is declared on the graph, next to
the channels the region iterates:

```cpp
g.iterate({flux, temp})
    .name("dirichlet_neumann")
    .until(fuse::Converged::Relative(1e-6, "u_wall") &
           fuse::Converged::Relative(1e-6, "q_wall"))
    .max_iterations(50)
    .accelerate({.kind = fuse::AccelKind::Aitken, .initial_relaxation = 0.5});
```

`fuse::Converged`, `ConvergenceMeasure`, `ConvergenceCriterion`, `AccelKind`,
`AccelConfig` and `NewtonConfig` all live in `include/fuse/model/iteration.hpp`.

## The four measure kinds

Each exchanged field is judged by one `ConvergenceMeasure`: a `MeasureKind`, a
tolerance and the name of what it measures. The four factories in
`fuse::Converged` build one each. In every formula `r` is the coupling
residual of that field over one sweep (`x_new - x_old`) and ‖·‖ is the
Euclidean norm over its entries.

| Factory | Test | Use it when |
|---|---|---|
| `Converged::Relative(tol, name)` | ‖r‖ / ‖x‖ ≤ tol, with ‖x‖ the norm of the current coupling value | the field's magnitude is not known in advance; the usual choice |
| `Converged::Absolute(tol, name)` | ‖r‖ ≤ tol | the field has a known scale, or passes through zero (a relative test divides by a vanishing ‖x‖) |
| `Converged::AbsoluteOrRelative(tol, name)` | either of the two above | a field that is sometimes near zero and sometimes not |
| `Converged::ResidualRelative(tol, name)` | ‖r_k‖ / ‖r_0‖ ≤ tol, with r_0 the first residual of the current window | you want a stated reduction of the residual, not a size relative to the solution |

```cpp
auto m1 = fuse::Converged::Relative(1e-6, "u_wall");
auto m2 = fuse::Converged::Absolute(1e-3, "q_wall");          // in q_wall's units
auto m3 = fuse::Converged::AbsoluteOrRelative(1e-6, "u_wall");
auto m4 = fuse::Converged::ResidualRelative(1e-4, "q_wall");  // four orders of magnitude
```

Two details of the arithmetic are worth knowing. A relative measure whose
reference norm is zero falls back to the absolute test, so a field that starts
at exactly zero does not divide by zero. A non-finite residual or reference
(a NaN or an infinity from a diverging component) reads as "not converged",
never as converged. Every factory throws `fuse::InvalidArgument` for a
tolerance that is not finite and positive.

### Composing measures

`operator&` joins measures into a `ConvergenceCriterion`; the region converges
only when **every** measure holds.

```cpp
fuse::ConvergenceCriterion crit =
    fuse::Converged::Relative(1e-6, "u_wall") &
    fuse::Converged::Absolute(1e-3, "q_wall");
crit = crit & fuse::Converged::ResidualRelative(1e-4, "wall_temperature");

// Equivalent, one at a time:
fuse::ConvergenceCriterion crit2;
crit2.add(fuse::Converged::Relative(1e-6, "u_wall"));
crit2.add(fuse::Converged::Relative(1e-6, "q_wall"));

g.iterate({flux, temp}).until(crit).max_iterations(30);
```

`IterationBuilder::until()` accepts either a `ConvergenceCriterion` or a single
`ConvergenceMeasure`, so a one-channel region can write
`.until(fuse::Converged::Absolute(1e-9, "flux"))` directly.

### Every channel must be measured

The `name` in a measure is matched against each of the region's channels by
the channel's name, its source field's name, or its target field's name. It
must match exactly one channel, and **every channel in the region must be
named by at least one measure**. Otherwise `Runner::initialize()` refuses with
`E-ITER-MEASURE`.

This region has two channels but measures only one:

```cpp
const auto fwd  = g.add_channel("a2b", a_out, b_in);
const auto back = g.add_channel("b2a", b_out, a_in);
g.channel(back).lag(1).allow_time_level_mismatch("resolved by the iteration region");

g.iterate({fwd, back})
    .name("pair")
    .until(fuse::Converged::Relative(1e-8, "a2b"))   // b2a is not measured
    .max_iterations(50);
```

`initialize()` reports:

```
E-ITER-MEASURE: iteration region 'pair' exchanges channel 'b2a' (b_out -> a_in),
but no convergence measure names it, so the region can stop while that channel
is still far from its fixed point. An accelerator will do exactly that: a
least-squares step over the concatenated residual can zero the measured block
and leave the unmeasured one alone. Add a measure naming 'b_out', 'a_in' or
'b2a', or take the channel out of the region
```

The fix names the second channel (by channel name here; `"b_out"` or `"a_in"`
would do the same):

```cpp
g.iterate({fwd, back})
    .name("pair")
    .until(fuse::Converged::Relative(1e-8, "a2b") &
           fuse::Converged::Relative(1e-8, "b2a"))
    .max_iterations(50);
```

The other declaration-time refusals are `E-ITER-NOCONVERGE` (the criterion is
empty: an empty criterion is trivially satisfied, so a region with no measures
would stop after one sweep) and `E-ITER-MAXITER` (`max_iterations < 1`).

## A complete example, with its residual history

Two in-place components, each an affine map on a four-cell line, coupled head
to tail and iterated with Aitken acceleration. `a` computes
`a_out = 0.6 a_in + 1` and `b` computes `b_out = 0.9 b_in + 2`, so the coupled
map has contraction factor 0.54 and the fixed point of `a_out` is
`(1 + 0.6 * 2) / (1 - 0.54) = 4.7826...`. Each component checkpoints itself
with `save_state()`/`restore_state()`, because a region re-advances its
members every sweep from the same starting state.

```cpp
#include <fuse/backend/serial_backend.hpp>
#include <fuse/base/units.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>
#include <fuse/model/iteration.hpp>
#include <fuse/runtime/component.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/plan.hpp>

#include <Kokkos_Core.hpp>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kN = 4;

/// out = gain * in + offset, on every cell.
class Slab final : public fuse::Component {
 public:
  Slab(std::string name, double gain, double offset)
      : name_(std::move(name)), gain_(gain), offset_(offset),
        in_(kN, 0.0), out_(kN, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = *g.find_component(name_);
    const fuse::Layout layout{.stride = 1, .count = kN};
    auto r = reg.bind(*g.find_field(cid, name_ + "_in"),
                      fuse::FieldRef{in_.data(), layout, fuse::MemorySpace::Host});
    if (!r.ok()) return r;
    return reg.bind(*g.find_field(cid, name_ + "_out"),
                    fuse::FieldRef{out_.data(), layout, fuse::MemorySpace::Host});
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    for (std::size_t i = 0; i < kN; ++i) out_[i] = gain_ * in_[i] + offset_;
    return fuse::Result<void>::success();
  }

  // The region re-advances its members every sweep, so it needs a checkpoint:
  // save everything advance() can change, restore it byte for byte.
  fuse::Result<void> save_state() override {
    saved_in_ = in_; saved_out_ = out_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    in_ = saved_in_; out_ = saved_out_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true,
                                   fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/false);
  }
  [[nodiscard]] double out0() const { return out_[0]; }

 private:
  std::string name_;
  double gain_, offset_;
  std::vector<double> in_, out_, saved_in_, saved_out_;
};

}  // namespace

int main() {
  Kokkos::ScopeGuard kokkos;

  fuse::CouplingGraph g;
  const std::string src = "generated:line:" + std::to_string(kN);
  const auto ma = g.add_mesh({.name = "ma", .source = src});
  const auto mb = g.add_mesh({.name = "mb", .source = src});
  const auto ra = g.add_region({.name = "ra", .mesh = ma, .dimension = 1});
  const auto rb = g.add_region({.name = "rb", .mesh = mb, .dimension = 1});
  const auto a = g.add_component({.name = "a", .ranks = fuse::RankSet{0, 1},
                                  .interval = fuse::Rational{1}});
  const auto b = g.add_component({.name = "b", .ranks = fuse::RankSet{0, 1},
                                  .interval = fuse::Rational{1}});
  const auto a_in  = g.add_field(a, {.name = "a_in",  .region = ra,
                                     .units = fuse::units::kKelvin,
                                     .intent = fuse::Intent::Requires});
  const auto a_out = g.add_field(a, {.name = "a_out", .region = ra,
                                     .units = fuse::units::kKelvin,
                                     .intent = fuse::Intent::Provides});
  const auto b_in  = g.add_field(b, {.name = "b_in",  .region = rb,
                                     .units = fuse::units::kKelvin,
                                     .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, {.name = "b_out", .region = rb,
                                     .units = fuse::units::kKelvin,
                                     .intent = fuse::Intent::Provides});

  const auto fwd  = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  // One direction must be lagged to break the cycle; the region resolves it.
  g.channel(back).lag(1).allow_time_level_mismatch("resolved by the iteration region");

  g.iterate({fwd, back})
      .name("pair")
      .until(fuse::Converged::Relative(1e-8, "a2b") &
             fuse::Converged::Relative(1e-8, "b2a"))
      .max_iterations(50)
      .accelerate({.kind = fuse::AccelKind::Aitken, .initial_relaxation = 0.5});

  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    if (auto r = backend.load(g.mesh(fuse::MeshId{i})); !r.ok()) return 1;
  }
  const auto lowered = fuse::lower(g);
  if (!lowered.ok()) { std::fputs(lowered.diags.to_string().c_str(), stderr); return 1; }
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks,
                              fuse::make_region_sizes(g, lowered.dag, backend));
  if (!sched.ok()) { std::fputs(sched.diags.to_string().c_str(), stderr); return 1; }

  fuse::ComponentSet components;
  components.add(std::make_unique<Slab>("a", 0.6, 1.0));   // a_out = 0.6 a_in + 1
  components.add(std::make_unique<Slab>("b", 0.9, 2.0));   // b_out = 0.9 b_in + 2
  fuse::FieldRegistry registry;
  for (auto& c : components) {
    if (auto r = c.register_fields(registry, g); !r.ok()) return 1;
  }

  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  if (auto r = runner.initialize(components, registry, fuse::RunnerConfig{}); !r.ok()) {
    std::fputs(r.error().message.c_str(), stderr); return 1;
  }
  if (auto r = runner.run_period(components); !r.ok()) {
    std::fputs(r.error().message.c_str(), stderr); return 1;
  }

  const fuse::IterationStats& st = runner.iteration_stats()[0];
  std::printf("converged=%d after %d sweeps\n", st.converged ? 1 : 0, st.iterations);
  for (std::size_t k = 0; k < st.residual_history.size(); ++k) {
    std::printf("  sweep %zu  |r| = %.3e\n", k, st.residual_history[k]);
  }
  const auto* sa = dynamic_cast<const Slab*>(components.find("a"));
  std::printf("a_out[0] = %.10f (exact %.10f)\n", sa->out0(), (1.0 + 0.6 * 2.0) / (1.0 - 0.54));
  return 0;
}
```

Expected output (exact digits can vary by platform):

```
converged=1 after 12 sweeps
  sweep 0  |r| = 4.472e+00
  sweep 1  |r| = 3.640e+00
  sweep 2  |r| = 2.045e+00
  sweep 3  |r| = 1.229e+00
  sweep 4  |r| = 4.460e+00
  sweep 5  |r| = 1.256e-01
  sweep 6  |r| = 3.138e-02
  sweep 7  |r| = 2.623e-02
  sweep 8  |r| = 2.275e-04
  sweep 9  |r| = 3.856e-04
  sweep 10  |r| = 4.479e-04
  sweep 11  |r| = 6.818e-08
a_out[0] = 4.7826087517 (exact 4.7826086957)
```

The same program with the accelerator changed and everything else fixed:

| `AccelConfig` | Sweeps to `1e-8` |
|---|---|
| `{.kind = AccelKind::None}` (plain Picard) | not converged in 50 |
| `{.kind = AccelKind::Constant, .omega = 0.5}` | not converged in 50 |
| `{.kind = AccelKind::Aitken, .initial_relaxation = 0.5}` | 12 |
| `{.kind = AccelKind::IQN_ILS, .initial_relaxation = 0.5, .filter = FilterKind::None}` | 4 |
| `{.kind = AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .filter = FilterKind::None}` | 4 |

On an affine map the quasi-Newton kinds recover the exact fixed point once
they hold enough secant pairs (the residual drops to round-off in one step);
Aitken has to re-learn its scalar each time the residual changes direction,
which is the jump at sweeps 4, 9 and 10 above.

## Reading the residual history

After each `run_period()`, `Runner::iteration_stats()` holds one
`IterationStats` per iteration region, in plan order:

```cpp
for (const fuse::IterationStats& st : runner.iteration_stats()) {
  std::printf("%s after %d sweeps\n", st.converged ? "converged" : "NOT converged",
              st.iterations);
  for (std::size_t k = 1; k < st.residual_history.size(); ++k) {
    std::printf("  sweep %zu: |r| = %.3e  ratio %.2f\n", k, st.residual_history[k],
                st.residual_history[k] / st.residual_history[k - 1]);
  }
}
if (runner.stats().windows_not_converged > 0) { /* some answer in this run is unconverged */ }
```

- `residual_history[k]` is the Euclidean 2-norm of the whole concatenated
  coupling residual `x_new - x_old` after sweep `k`, all measured channels
  stacked into one vector. It is the number to plot. The per-measure decision
  is made on each field's own block, so the history can keep falling after
  one field has already met its tolerance.
- `converged` is false when the region exhausted `max_iterations`. The state
  on exit is still well defined (the components hold the last iterate, and
  the instructions after the region deliver it), but `RunStats::
  windows_not_converged` counts the event. Set
  `RunnerConfig::abort_on_nonconvergence = true` to turn it into a failure
  (`E-RT-NOCONVERGE`) instead.
- A ratio between successive entries that settles near a constant below 1 is
  a linear fixed point; an accelerator should make it fall. A ratio hovering
  near 1 means the tolerance is below what the components can resolve, or the
  coupling is not a contraction and needs an accelerator or Newton. An IQN
  region that stalls on a problem you believe is well conditioned is worth
  rerunning with `FilterKind::None` (see the filter notes below).

## Choosing tolerances

- A relative tolerance of `1e-6` is a reasonable start for a field whose
  magnitude is set by the physics; `1e-8` is near the point where a
  component's own linear solver tolerance, not the coupling, limits the
  residual. Tightening beyond what the components resolve produces a region
  that never converges rather than a better answer.
- Use `Absolute` with a tolerance in the field's units for a field that
  changes sign or starts from zero (a displacement, a net flux).
- `ResidualRelative(1e-3, ...)` is the usual form when the time step itself
  bounds the error you care about: three orders of reduction per step is often
  enough for a second-order time integrator, and a tighter coupling tolerance
  is wasted on it.
- State one measure per exchanged field, in that field's own terms. The
  Euclidean norm the measures use depends on the number of entries, on cell
  size variation, on the partition and on units; the next section explains
  what to do when that matters.

## The measure-weighted norm

The usual residual measure is the Euclidean norm over the array of interface
values, `sqrt(sum r_i^2)` (`fuse::num::NormKind::Euclidean`,
`include/fuse/num/residual_norm.hpp`). It is cheap and it is what published
iteration counts are usually quoted with. But it depends on four things that
are not physics:

| It changes when | Because |
|---|---|
| you refine the interface | there are more entries to square and add, so the same physical error reads larger |
| cells vary in size | a sliver cell counts as much as a large one, so it can hold up an iteration converged everywhere else |
| you change partitions | a value owned by one rank and mirrored on two others is counted three times |
| fields have different units | a temperature norm and a heat-flux norm are added as though commensurable |

A tolerance tuned on a coarse serial mesh is therefore a *different* tolerance
on a fine partitioned one, and nothing tells you that.

`NormKind::MeasureWeightedRms` computes
`sqrt(sum_owned w_i r_i^2 / sum_owned w_i)`, where `w_i` is the measure
(length, area or volume) of the entity the value lives on and the sums run over
owned entities only. That is a root-mean-square error weighted by how much of
the interface each value represents. It has the units of the field, and it is
independent of resolution, of grading and of the partition, so a tolerance
means the same thing in all three.

```cpp
fuse::num::ResidualField f{
    .residual = r.data(), .measure = cell_area.data(),
    .owned = owned_mask.data(), .count = n, .units = fuse::units::kKelvin};
const auto norm = fuse::num::field_norm(
    fuse::num::NormKind::MeasureWeightedRms, f);
// norm.value() carries .value, .units, .total_measure and .owned_count
```

Two behaviours are deliberate:

**A rank owning no part of the interface returns `owned_count == 0`, not zero.**
Zero would read as "converged" inside a reduction across ranks. `combine()`
excludes such contributions rather than averaging them in.

**Combining norms over fields with different dimensions is an error**
(`E-NORM-UNITS`), not a sum. A number made by adding a temperature to a heat
flux has no meaning, and a tolerance on it cannot be reasoned about. If you
genuinely want one number over unlike fields, scale them to a common unit
yourself, so that the scaling is visible and reviewable.

Use the weighted norm for new work, especially if you will change resolution or
rank count. Use the plain norm when you are reproducing a published iteration
count that was obtained with it. For a Newton region the choice is
`NewtonConfig::norm` (below).

## Accelerators

`IterationBuilder::accelerate(AccelConfig)` selects how the next iterate is
built from the residual history. The kinds form a ladder from weakest to
strongest; every one sees only a concatenated vector of doubles and an
iteration index.

| `AccelKind` | What it does | When to use it | `AccelConfig` fields it reads |
|---|---|---|---|
| `None` | plain Picard, `x_out = x_new` | a coupling that is already a strong contraction; a baseline to compare against | none |
| `Constant` | fixed under-relaxation `x + omega (x_new - x)` | you know a stable `omega`; reproducing a scheme from the literature | `omega` |
| `Aitken` | one dynamic relaxation factor for the whole vector, re-estimated every sweep | the default first choice for a two-field coupling; robust, no tuning | `initial_relaxation` |
| `Steffensen` | one dynamic factor per entry (diagonal secant) | entries converge at very different rates and are nearly decoupled | `initial_relaxation` |
| `Secant` | rank-1 vector least squares (IQN with one column) | rarely; a cheap step up from Aitken when the history is unreliable | `initial_relaxation` |
| `IQN_ILS` | rank-`k` interface quasi-Newton, inverse least squares; history cleared at each window | strongly coupled problems (added-mass FSI, stiff Dirichlet-Neumann); the usual production choice | `initial_relaxation`, `reuse`, `filter`, `limit`, `scale_fields` |
| `IQN_IMVJ` | `IQN_ILS` plus an approximate inverse Jacobian carried across windows | the same problems when successive windows are similar, so the previous window's Jacobian is a good start | as `IQN_ILS`; `reuse` also caps the chained windows at `min(reuse, 8)` |
| `Newton` | replaces the whole sweep with a Jacobian-free Newton-Krylov solve | members can evaluate a residual (`kCapabilityResidual`); see below | `NewtonConfig`, not `AccelConfig` |

The fields, with their defaults from `AccelConfig`:

- `initial_relaxation = 0.1`: used whenever no usable history exists, which is
  the first sweep of every window for every kind and any later sweep whose
  secant information is degenerate. `0.5` is a common choice for a problem that
  is not strongly coupled.
- `omega = 0.5`: the fixed factor for `Constant` only.
- `reuse = 8`: the number of secant column pairs kept within a window, in
  `[1, 32]`. With a filter active, raising it above 8 changes nothing; tune
  `filter`/`limit` first.
- `filter = FilterKind::QR2`, `limit = 1e-2`: the column filter applied before
  the least-squares solve (`None`, `QR1`, `QR2`, `QR3`). The default is tuned
  for ill-conditioned FSI. On a well-conditioned problem it discards useful
  columns and can slow convergence by an order of magnitude; `FilterKind::None`
  is then the right choice, and a window that will not converge under the
  default is a reason to try it before suspecting the accelerator.
- `scale_fields = true`: scale each field block of the residual by the
  reciprocal of its own 2-norm before the least-squares solve, so a
  large-magnitude field cannot dominate the fit. Only matters with two or more
  fields.

:::{warning}
`AccelConfig{}` defaults to `kind = AccelKind::Constant` with `omega = 0.5`,
but a region's own default (`IterationRegionDecl::acceleration`) is
`AccelKind::None`. Writing `.acceleration = fuse::AccelConfig{}` in an
`IterationRegionDecl`, or `.accelerate({})`, therefore switches the region from
plain Picard to constant relaxation. Always name the kind:
`fuse::AccelConfig{.kind = fuse::AccelKind::None}`.
:::

```cpp
const fuse::AccelConfig iqn{.kind = fuse::AccelKind::IQN_ILS,
                            .initial_relaxation = 0.5,
                            .reuse = 8,
                            .filter = fuse::FilterKind::None};
g.iterate({flux, temp})
    .until(fuse::Converged::Relative(1e-6, "u_wall") &
           fuse::Converged::Relative(1e-6, "q_wall"))
    .accelerate(iqn);
```

Every member of an accelerated region must support rollback
(`kCapabilityRollback`, with working `save_state()`/`restore_state()`);
`initialize()` refuses a region otherwise with `E-ITER-NOROLLBACK`, and after
every restore it compares the component's registered `Provides` storage byte
for byte against the checkpoint (`E-ITER-ROLLBACK` if they differ). A member
declaring `DeterminismLevel::Nondeterministic` is refused with
`E-ITER-NONDETERMINISTIC`, because noise in its output is indistinguishable,
in the convergence test, from a residual that has not converged.

## Newton at a glance

A region selecting `AccelKind::Newton` (`.newton(cfg)` or
`.accelerate({.kind = AccelKind::Newton})`) replaces the Picard sweep with a
Jacobian-free Newton-Krylov solve of `F(x) = x - G(x) = 0` over the stacked
interface vector. It tests convergence on its own: `atol`, `rtol` and `stol`
in `NewtonConfig` apply to a norm over the whole stacked residual, and
`max_nonlinear_iterations` bounds the Newton steps. The per-field measures in
`until()` are still required (they define the blocks and, for the
measure-weighted norm, their scales).

```cpp
fuse::NewtonConfig cfg;
cfg.rtol = 1e-8;
cfg.norm = fuse::NewtonNorm::MeasureWeighted;   // default: fuse::NewtonNorm::Plain
g.iterate({flux, temp})
    .until(fuse::Converged::Relative(1e-6, "u_wall") &
           fuse::Converged::Relative(1e-6, "q_wall"))
    .newton(cfg);
```

Selecting Newton sets `requires_residual_capability` on the region: every
member must declare `kCapabilityResidual` (`E-NEWTON-CAPABILITY` otherwise),
and `use_block_jacobi_preconditioner = true` additionally requires
`kCapabilityLinearization`. `NewtonConfig::substrate` chooses `Native` (always
available) or `Petsc` (`E-NEWTON-PETSC-UNAVAILABLE` without PETSc installed
and `fuse::petsc::install()` called). The Eisenstat-Walker forcing, line
search and GMRES settings keep SNES's defaults.

### Which Newton norm

| `NewtonConfig::norm` | The `atol`/`rtol` test is applied to |
|---|---|
| `Plain` (default) | the Euclidean 2-norm of the stacked residual `F(x)` |
| `MeasureWeighted` | `D * F(x)`, where `D` is a fixed diagonal scaling built once per solve |

**`Plain`** has the four dependencies listed above. Because all exchanged
fields are concatenated into one vector, a region mixing fields of different
units (a wall temperature in kelvin and a heat flux in W/m^2) is tested by a
single number with no unit. The relative test (`||F_k|| <= rtol * ||F_0||`)
softens this, since a global rescaling of either field cancels, but a field
whose error dominates the concatenated norm can still mask slower convergence
in another.

**`MeasureWeighted`** puts every block on a common footing. Each block of the
stacked vector (one per exchanged field transfer) is scaled by
`D_i = sqrt(w_i / sum_block_w) / s_b`, where `w_i` is the measure of the
entity the entry lives on, normalised within its own block, and `s_b` is the
tolerance declared for that block by the region's `ConvergenceCriterion`. A
value of 1 in the scaled space means "at this block's own tolerance, in its
own units". Choose it when the region mixes fields with different units or
magnitudes, or when you will change resolution or partition. `stol` (the
step-length test) is not scaled under either choice.

:::{warning}
`MeasureWeighted` needs exactly one tolerance per block, taken from the
measure that names the block's channel, source field or target field. If no
measure names a block, or several name it with differing tolerances,
`initialize()` refuses with `E-NEWTON-NORM-SCALE`. Name each block in exactly
one measure, or in several that agree, or select `NewtonNorm::Plain`.
:::

A region that sets `use_block_jacobi_preconditioner` together with
`MeasureWeighted` is handled for you: the preconditioner is composed with
`D^-1`. For a window-scoped region ([](coupling-windows.md)),
`MeasureWeighted` repeats each element's spatial weight uniformly across every
sample in its window. When a Newton region's components span more than one
rank, the norm is agreed across exactly that region's participant ranks; rank
count changes who contributes to the norm, never which norm or tolerance
applies. The solver itself, its codes and its derivation are in
[](../dev/newton-krylov.md).

## See also

- [](coupling-windows.md) for convergence across a time window and waveform iteration
- [](dn-heat.md) and [](fsi-channel.md) for worked examples comparing Aitken, IQN-ILS and Newton
- [](../dev/newton-krylov.md) for the Newton-Krylov implementation
- [](troubleshooting.md) for the `E-ITER-*` and `E-NEWTON-*` codes
