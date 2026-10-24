# Running a coupling: Runner, components and statistics

This page is for anyone who has a compiled coupling (a `CouplingGraph`, its `OperatorDAG` and an
`ExecutionPlan`) and needs to run it: how to construct the `fuse::Runner`, wire component objects
and field storage to it, step it period by period, read its statistics, and understand what it
checks before and during a run. [](quickstart.md) shows the whole pipeline once; this page is the
reference for the last stage of it.

## The smallest working example

Two components on one line mesh. `source` steps every 1/4 and writes a temperature; `sink` steps
every 1/2 and receives the mean over its own step. The program runs three coupling periods and
prints the statistics.

```cpp
// Two components, one channel, three coupling periods, statistics.
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>

#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>

namespace {

// Owns one array and binds every field the graph gives it to that array.
class ArrayComponent final : public fuse::Component {
 public:
  ArrayComponent(std::string name, std::size_t n, bool produces)
      : name_(std::move(name)), data_(n, 0.0), produces_(produces) {}

  std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto id = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*id).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1, .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    if (produces_) {  // write the provided field in place, valid at t_end
      const double t = ctx.t_end.to_double();
      for (std::size_t i = 0; i < data_.size(); ++i) data_[i] = t + 0.01 * double(i);
    }
    return fuse::Result<void>::success();
  }

  double first() const { return data_.front(); }

 private:
  std::string name_;
  std::vector<double> data_;
  bool produces_;
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  // 1. Describe.
  fuse::CouplingGraph g;
  const auto mesh = g.add_mesh(fuse::MeshDecl{.name = "m", .source = "generated:line:8"});
  const auto cells = g.add_region(fuse::Region{.name = "cells", .mesh = mesh, .dimension = 1});
  const auto src = g.add_component(fuse::ComponentDecl{
      .name = "source", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1, 4}});
  const auto dst = g.add_component(fuse::ComponentDecl{
      .name = "sink", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1, 2}});
  const auto t_out = g.add_field(src, fuse::FieldSpec{
      .name = "T", .region = cells, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Provides});
  const auto t_in = g.add_field(dst, fuse::FieldSpec{
      .name = "T_in", .region = cells, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("T_link", t_out, t_in);
  g.channel(ch).accumulate(fuse::AccumOp::Mean);   // source steps twice per sink step
  if (const auto d = fuse::validate(g); d.has_errors()) { std::cerr << d.to_string(); return 1; }

  // 2. Compile.
  fuse::SerialMeshBackend backend;
  if (const auto r = backend.load(g.mesh(mesh)); !r.ok()) { std::cerr << r.error().message; return 1; }
  const auto lowered = fuse::lower(g);
  if (!lowered.ok()) { std::cerr << lowered.diags.to_string(); return 1; }
  const fuse::ClockSet clocks{g};
  const auto sched = fuse::schedule(g, lowered.dag, clocks,
                                    fuse::make_region_sizes(g, lowered.dag, backend));
  if (!sched.ok()) { std::cerr << sched.diags.to_string(); return 1; }

  // 3. Wire components and fields.
  fuse::ComponentSet components;
  components.add(std::make_unique<ArrayComponent>("source", 8, true));
  components.add(std::make_unique<ArrayComponent>("sink", 8, false));
  fuse::FieldRegistry registry;
  for (auto& c : components) {
    if (const auto r = c.register_fields(registry, g); !r.ok()) { std::cerr << r.error().message; return 1; }
  }

  // 4. Run.
  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  fuse::RunnerConfig cfg;              // transport "copy", conservation checks on
  if (const auto r = runner.initialize(components, registry, cfg); !r.ok()) {
    std::cerr << r.error().code << ": " << r.error().message << "\n"; return 1;
  }
  if (const auto r = runner.run(components, 3); !r.ok()) {
    std::cerr << r.error().code << ": " << r.error().message << "\n"; return 1;
  }

  // 5. Report.
  const fuse::RunStats& s = runner.stats();
  std::cout << "period length        : " << sched.plan.period().to_string() << "\n"
            << "periods              : " << s.periods << "\n"
            << "instructions executed: " << s.instructions_executed << "\n"
            << "conservation checks  : " << s.conservation_checks
            << " (violations " << s.conservation_violations << ")\n"
            << "total iterations     : " << s.total_iterations << "\n"
            << "windows not converged: " << s.windows_not_converged << "\n"
            << "epochs reconciled    : " << s.epochs_reconciled << "\n";
  const auto* sink = dynamic_cast<ArrayComponent*>(components.find("sink"));
  std::cout << "sink T_in[0] after 3 periods: " << sink->first() << "\n";
  return 0;
}
```

Expected output (exact digits can vary by platform):

```text
period length        : 1/2
periods              : 3
instructions executed: 18
conservation checks  : 0 (violations 0)
total iterations     : 0
windows not converged: 0
epochs reconciled    : 0
sink T_in[0] after 3 periods: 1.375
```

The coupling period is 1/2, the sink's step. After three periods the sink holds the mean of the
source's values at t = 1.25 and t = 1.5, which is 1.375. There are no conservation checks because
the fields are `Intensive`; an `Extensive` field with a conservative channel adds a `Constrain`
instruction per transfer. There are no epochs because the run has one rank.

## Construct the Runner

```cpp
fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
```

`Runner(const CouplingGraph&, const OperatorDAG&, ExecutionPlan plan, const MeshBackend&)` takes
the graph and DAG by reference (they must outlive the Runner) and the plan by value. The backend
is any `fuse::MeshBackend` (`SerialMeshBackend`, `native::NativeMeshBackend`, `MoabMeshBackend`)
with the graph's meshes already loaded; see [](geometry.md). A Runner is move-only.

Nothing runs until `initialize()`:

```cpp
fuse::Result<void> initialize(fuse::ComponentSet& components, fuse::FieldRegistry& registry,
                              const fuse::RunnerConfig& config = {});
```

### `RunnerConfig`

| Field | Default | Meaning |
|---|---|---|
| `transport` | `"copy"` | The transport backend used for every `Transport` instruction: `"copy"`, `"mpi"` or `"lowfive"` |
| `check_conservation` | `true` | Evaluate `Constrain` instructions |
| `conservation_tolerance` | `1e-12` | Relative tolerance of a conservation check |
| `abort_on_violation` | `false` | Fail the period on a violated conservation check instead of counting it |
| `abort_on_nonconvergence` | `false` | `false` counts an unconverged region-period in `RunStats::windows_not_converged` and continues; `true` fails the period with `E-RT-NOCONVERGE` |
| `force_distributed_nodal_remap` | `false` | Take the distributed nodal-remap path even on one rank (for testing that path; not a performance setting) |
| `base_communicator` | unset | `MPI_Comm_c2f()` of a communicator containing every rank of every component. Required for `"mpi"` transport and for any region spanning more than one rank |
| `force_this_process` | unset | Override the rank identity used for hosting decisions without a communicator (testing only) |
| `transport_setup` | empty | `std::function<Result<void>(Transport&)>` called once on the transport right after it is constructed, for backend-specific setup |
| `execution_space` | `Default` | `RunnerExecutionSpace::{Default, Host, Device}`; where elementwise kernels run |

`cfg.resolved_execution_space()` returns what `Default` resolves to in the current environment.

## Wire components and fields

### `ComponentSet`

`fuse::ComponentSet` owns the `Component` objects. `add(std::unique_ptr<Component>)` throws
`InvalidArgument` for a null pointer, an empty name or a duplicate name. `find(name)` returns a
`Component*` or null; `size()`, `empty()`, `at(i)` and range iteration (yielding `Component&`) are
also available.

`initialize()` matches the set against the graph by name: a graph component with no object is
refused with `E-RT-COMPONENT` ("would never be advanced"), and so is an object whose name is not in
the graph ("nothing schedules it").

### `FieldRegistry`

Each component binds its storage to the `FieldId`s the graph gave it, once, from
`register_fields()`:

```cpp
reg.bind(field_id, fuse::FieldRef{ptr, fuse::Layout{.stride = 1, .count = n}, fuse::MemorySpace::Host});
```

`FieldRef` is a non-owning view: pointer, `Layout{stride, component_stride, count, components}` and
`MemorySpace::Host` or `Device`. A field with `components > 1` is interleaved with
`stride = components, component_stride = 1`, or blocked with `stride = 1, component_stride = count`.
`bind()` refuses `E-FIELD-ID` (invalid id), `E-FIELD-NULL` (null pointer with `count > 0`; null with
`count == 0` is allowed for a rank that owns nothing), `E-FIELD-LAYOUT` (strides that alias or
overflow) and `E-FIELD-REBIND` (a field bound twice). On an error the registry is unchanged.

:::{warning}
The registry copies pointers, not data. The storage must stay at the same address for the whole
run, including through `restore_state()`; swapping two vectors makes the coupler read the old
buffer. `initialize()` resolves every arc to a raw pointer once, and the period loop allocates
nothing.
:::

### What `initialize()` checks

`initialize()` constructs the transport, resolves every buffer, builds remap weights and verifies
the run before the first step. It reports the first failure it finds:

| Code | Meaning |
|---|---|
| `E-RT-COMPONENT` | a graph component has no object in the set, or an object has no declaration |
| `E-RT-UNBOUND` | a component field was never bound (`register_fields()` was not called, or skipped the field) |
| `E-RT-SIZE` | a bound field has a different `count` or `components` than the plan sizes its region at, or an instruction's operands disagree in size |
| `E-RT-SPACE` | a field is registered in device memory under `Host` execution, or `FUSE_EXECUTION_SPACE` has an unknown value |
| `E-RT-DEVICE` | `Device` execution requested on a build whose Kokkos has no device backend |
| `E-RT-TRANSPORT-NOMPI` | `base_communicator` set, or a multi-rank region, in a build without MPI |
| `E-RT-TRANSPORT` | the plan needs the `"mpi"` transport but another one is configured |
| `E-ITER-NOCONVERGE` | an iteration region with `max_iterations > 1` has no convergence measure |
| `E-ITER-MAXITER` | `max_iterations < 1` |
| `E-ITER-MEASURE` | a measure names nothing (no channel, source field or target field of that name), or a channel of the region is measured by no measure |
| `E-ITER-NOROLLBACK` | a member of a region's derived checkpoint set does not declare `kCapabilityRollback` |
| `E-ITER-NONDETERMINISTIC` | a region member declares `DeterminismLevel::Nondeterministic` |
| `W-ITER-DETERMINISM-UNDECLARED` | a region member declares no determinism level (a warning) |
| `E-NEWTON-CAPABILITY` | a member of a Newton region lacks `kCapabilityResidual`, or `kCapabilityLinearization` when the region needs it |
| `E-NEWTON-NORM-SCALE`, `E-NEWTON-DIRECT-WINDOW`, `E-NEWTON-PETSC-UNAVAILABLE` | Newton configuration problems; see [](convergence.md) |

After `initialize()`, `runner.checkpoint_explanations()` holds one string per region explaining
which components the region checkpoints and why. Print it when a region refuses
`E-ITER-NOROLLBACK`.

## Step the run

```cpp
fuse::Result<void> run_period(fuse::ComponentSet&);            // exactly one coupling period
fuse::Result<void> run(fuse::ComponentSet&, std::int64_t periods);
```

The unit of progress is the *coupling period*, `plan.period()`: the shortest interval after which
the pattern of steps and transfers repeats. `run_period()` advances every instruction of exactly one
period and no further; `run(components, n)` calls it `n` times. The simulated time of a step is
`period_index * plan.period() + tick.time`, delivered to each component in `StepContext::t_begin`
and `t_end` as exact `Rational`s. `runner.current_period_index()` reads the period counter.

There is no "run to time T" call. To stop at a time, compute the number of periods:

```cpp
const fuse::Rational t_end{23, 2};
const fuse::Rational n = t_end / sched.plan.period();   // exact
if (!n.is_integer()) { /* t_end is not a period boundary: choose the window or step differently */ }
const std::int64_t periods = n.num();
```

The Runner never executes an instruction of a period the caller did not ask for. This matters when
ranks are driven by different programs: a rank that runs on into an unrequested period would wait
in a collective for peers that have already returned.

### Externally driven components

A component whose own code owns the time loop ([](c-abi.md), [](fortran.md)) is *externally
driven* (`Component::is_externally_driven()` returns `true`). Such a graph is stepped with

```cpp
fuse::Result<std::optional<fuse::ExternalAdvanceRequest>> resume_period(fuse::ComponentSet&);
```

which runs the instruction stream up to the next externally driven component's `advance()` point
and returns an `ExternalAdvanceRequest{component, ctx, requires_checkpoint, requires_restore,
is_delivery_due, delivery_instruction_index, delivery_field}` describing whose turn it is and over
which interval; `std::nullopt` means the period completed. The ABI layer calls this for you;
`run_period()` on such a graph refuses `E-RT-EXTERNAL-UNSUPPORTED`. Do not mix `resume_period()`
with `run_period()`/`run()` on one Runner. Two plan queries support an external driver without side
effects: `components_due_at_period_start()` and `next_real_advance_after(components, index)`.

## Read statistics

`runner.stats()` returns the accumulated `fuse::RunStats`:

| Field | Meaning |
|---|---|
| `periods` | periods completed |
| `instructions_executed` | plan instructions executed, summed over periods and sweeps |
| `conservation_checks` | `Constrain` evaluations |
| `conservation_violations` | checks that exceeded `conservation_tolerance` |
| `worst_conservation_error` | the largest relative error seen by a check |
| `total_iterations` | sweeps of iteration-region bodies, summed over regions and periods |
| `windows_not_converged` | region-periods that hit `max_iterations` without meeting their criterion; non-zero means some answer in this run is not converged |
| `remap_weight_builds` | conservative remap matrices built at `initialize()`, one per distinct remap plan (0 when every remap is a proved-coincident identity) |
| `remap_weight_nonzeros` | non-zeros summed over those matrices |
| `checkpoint_verifications` | times the runtime compared a component's registered storage against its checkpoint after `restore_state()` |
| `epochs_reconciled` | rank-agreement points this rank took part in (0 on one rank) |
| `device_dispatches` | instructions whose kernels ran on the device space (0 under `Host`) |
| `device_resident_arcs` | coupler-owned buffers allocated in device memory |
| `device_boundary_copies` | copies across the host/device boundary |
| `device_staged_copies` | per-instruction staging copies for host-side operands |
| `device_transport_moves` | `Transport` exchanges whose packed buffer stayed in device memory |

`runner.iteration_stats()` returns one `fuse::IterationStats` per iteration region of the plan, in
DAG region order, describing the most recent period:

| Field | Meaning |
|---|---|
| `iterations` | sweeps taken |
| `converged` | whether the criterion was met |
| `residual_history` | one entry per sweep: the 2-norm of the whole concatenated residual `x_new - x_old` |

Other queries: `buffer(ArcId)` returns a read pointer to a coupler-owned buffer (`E-RT-SPACE` if it
is device-resident); `field_window_view(FieldId)` returns `{data, count, window_samples}` for a
field that is part of a coupling window; `hosts_component(ComponentId)` tells whether this process
hosts a component.

## How an iteration region runs

An iteration region is declared on the graph ([](concepts.md#iteration)); the Runner executes it
without any extra call:

1. At the region's entry in each period, every component in the region's derived checkpoint set is
   saved once with `save_state()`.
2. The body runs: every member advances and every channel of the region is exchanged.
3. The residual `x_new - x_old` is computed per transferred block, and each measure of the
   `ConvergenceCriterion` is tested. If all are satisfied, the region exits.
4. Otherwise the accelerator (`AccelKind::None`, `Constant`, `Aitken`, `Secant`, `Steffensen`,
   `IQN_ILS`, `IQN_IMVJ`) produces the next iterate, every checkpointed member is restored with
   `restore_state()`, the runtime compares the member's registered storage against the checkpoint
   (`checkpoint_verifications`), and the body runs again, up to `max_iterations`.

`AccelKind::Newton` replaces the sweep loop with a Jacobian-free Newton-Krylov solve of
`x - G(x) = 0`, on the native solver or on PETSc per `NewtonConfig::substrate`; its stopping tests
are its own ([](convergence.md)). A region whose `window_scope` is set runs as a coupling window
([](coupling-windows.md)). Inside a region, `advance()` must be a pure function of the restored
state and its inputs; a component that keeps unregistered state must save and restore it too, and
declares `kCapabilityHiddenState` if it cannot.

## Choose the execution space

`RunnerConfig::execution_space` selects where the Runner's elementwise kernels (interpolation,
remap, sampling, accumulation, extrapolation, merge, scaling, relaxation, residuals) run:

- `Host` runs them on the host execution space.
- `Device` runs them on the device space of the Kokkos build. `initialize()` refuses `E-RT-DEVICE`
  when the build has no CUDA, HIP or SYCL backend; `Runner::device_execution_space_name()` reports
  the space `Device` would use, so a program can ask before configuring.
- `Default` reads `FUSE_EXECUTION_SPACE`: `device` selects `Device`; `host`, empty or unset selects
  `Host`; anything else is `E-RT-SPACE`.

Fields may be registered in device memory (`MemorySpace::Device`) only under `Device` execution. A
host-registered field is mirrored to the device before a kernel reads it; see [](power-users.md)
for the device counters above. Transport, epochs, sinks, accelerators and windows run on the host
in every mode.

## Choose a transport

`RunnerConfig::transport` names a backend from `fuse::TransportRegistry::instance().available()`:

| Name | Moves data | Needs |
|---|---|---|
| `"copy"` | within one process | nothing; the default |
| `"mpi"` | between ranks over MPI | `FUSE_ENABLE_MPI`, `base_communicator`, and a `transport_setup` that calls `MpiTransport::set_communicator(comm_f)` |
| `"lowfive"` | between processes through the LowFive HDF5 VOL | `FUSE_ENABLE_LOWFIVE` and a `transport_setup` that calls `LowFiveTransport::set_location(file, dataset)` |

`transport_setup` is the one hook for backend-specific setup. `initialize()` calls it exactly once,
immediately after the registry constructs the named transport and before any instruction is
resolved; cast to the concrete type inside it (`fuse::MpiTransport` exists only in a build with
`FUSE_ENABLE_MPI`):

```cpp
cfg.transport = "mpi";
cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
cfg.transport_setup = [](fuse::Transport& t) -> fuse::Result<void> {
  return static_cast<fuse::MpiTransport&>(t).set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
};
```

An empty `transport_setup` is a no-op, which is right for `"copy"`. A plan that needs
rank-to-rank redistribution with any transport other than `"mpi"` is refused with
`E-RT-TRANSPORT`. See [](placement.md) for how the plan chooses what to move and
[](multiple-executables.md) and [](lowfive.md) for the two non-local transports.

## What happens when a rank fails

After every data movement the plan contains an `Epoch`, a point where the participating ranks
agree whether anything failed since the last one. A rank whose instruction fails does not throw at
once: it carries the failure to the next epoch, takes part in the agreement, and only then returns
the error. The peers learn of the failure at the same epoch and return `E-RT-EPOCH-PEER` (or
`E-RT-WINDOW-PEER` inside a window's convergence agreement) instead of waiting for a message that
never comes. The failing rank's own error carries the original code. `RunStats::epochs_reconciled`
counts these agreement points; it is 0 on one rank. `E-RT-EPOCH-TAG` is refused at `initialize()`
when the MPI implementation's tag upper bound is below what the library uses.

`FUSE_COLLECTIVE_TRACE=1` prints one line per collective instruction per rank (region, period,
sweep, instruction, kind, communicator, ordinal), so two ranks' streams can be diffed when they
disagree.

## Restart

A Runner writes and restores its state at period boundaries through
`fuse::write_restart_point(runner, components, writer, time, period_index)` and
`fuse::restore_restart_point(runner, components, reader, allow_accelerator_reset, any_reset,
out_time, out_period_index)` (`fuse/runtime/restart.hpp`). The Runner-level pieces they use are
`write_restart_fields()`, `restore_restart_fields()`, `current_period_index()` and
`set_current_period_index()`; an externally driven component asks for a point through
`request_restart_point()`, which the host polls with `take_restart_point_request()`. [](restart.md)
has the file formats, component hooks and refusals.

## Limitations

- Progress is by whole coupling periods; there is no run-to-time call.
- A Runner steps either with `run_period()`/`run()` or with `resume_period()`, never both.
- The `Runner` constructor and `initialize()` must be called on every rank of every component;
  each rank runs the instructions it hosts.

## See also

- [](component-interface.md): the `Component` contract, `StepContext` and capability bits.
- [](ir.md) and [](graph-programming.md): the DAG and plan the Runner interprets.
- [](convergence.md) and [](coupling-windows.md): iteration regions and windows.
- [](restart.md): restart to disk.
- [](environment.md): `FUSE_EXECUTION_SPACE`, `FUSE_COLLECTIVE_TRACE` and the other variables.
- [](troubleshooting.md): looking up an `E-...` code.
