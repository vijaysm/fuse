# The operator IR and the execution plan

What a `CouplingGraph` compiles to, how to read the two text dumps, how to walk the compiled
structures from your own code, and what the scheduler proves before it hands a plan to the
`Runner`. For power users: anyone debugging an unexpected transfer, keeping compiled output as a
golden file, or extending the lowering. [](quickstart.md) shows where the dumps come from;
[](graph-programming.md) covers what goes in.

## The pipeline and its signatures

```cpp
#include <fuse/model/validate.hpp>     // validate
#include <fuse/ir/lowering.hpp>        // lower, LoweringResult, OperatorDAG
#include <fuse/ir/dump.hpp>            // dump_text, dump_graphviz
#include <fuse/schedule/plan.hpp>      // ClockSet, schedule, ExecutionPlan
#include <fuse/backend/mesh_view.hpp>  // make_region_sizes

Diagnostics     validate(const CouplingGraph& g, const ConnectPolicy& policy = {});
LoweringResult  lower(const CouplingGraph& g, ConnectPolicy policy = {});   // .dag, .diags, ok()
ClockSet        clocks{g};
RegionSizes     sizes = make_region_sizes(g, dag, backend);   // std::function<size_t(RegionId, int placement)>
ScheduleResult  schedule(const CouplingGraph& g, const OperatorDAG& dag, const ClockSet& clocks,
                         const RegionSizes& sizes, ExpandMode mode = ExpandMode::Auto);  // .plan, .diags, ok()
std::string     dump_text(const OperatorDAG&);
std::string     dump_graphviz(const OperatorDAG&);
// and, on the plan: std::string ExecutionPlan::to_string() const;
```

Each stage is a pure function of the previous one. `validate()` is yours to call; `lower()` does not
call it. `lower()` and `validate()` must receive the same `ConnectPolicy`, or the compiler can emit
a unit conversion the checker forbade. `RegionSizes` answers "how many entities of this region live
on this rank for this placement"; `make_region_sizes()` asks a loaded mesh backend, and a test can
pass a lambda returning a constant. The `Runner` then executes `sched.plan`
([](quickstart.md#4-run)).

## The operator DAG

`lower()` turns every channel into a chain of typed operator nodes connected by *arcs*. The DAG is
in static single-assignment form: every arc has exactly one producer, `add_node()` throws
`InvariantViolation` on a second, and nothing is ever removed or rewritten.

### Operator kinds

| `OpKind` | Meaning | Produced by | Collective |
|---|---|---|---|
| `Source` | A component's provided field as it leaves the component; its output aliases the registered array | every `Provides` or `Bidirectional` field | no |
| `Sink` | Delivery into a required field; its output aliases the registered array | every `Requires` or `Bidirectional` field a channel writes | no |
| `Sample` | Record the source value onto a `Waveform` at each source step | `accumulate()` with a non-`Instant` op, or `spacetime()` | no |
| `Accumulate` | Reduce the waveform over the target step (`attrs.accum`) | `accumulate()` | no |
| `Interpolate` | Project a waveform onto a temporal basis (`basis`, `degree`), or carry a lag shift when no `Remap` follows (`lag`) | `spacetime()`; `lag()` on a same-region channel | no |
| `Extrapolate` | Extrapolate forward in time (`order`) | `extrapolate()` | no |
| `Scale` | Multiply by an exact `Rational` (`scale`; `reason` is `inferred` or `requested`) | differing field units with a pure ratio | no |
| `Transport` | Move a value between two processor sets (`src_rank_set -> dst_rank_set`, both component indices) | endpoints or `remap_on` on different rank sets | yes |
| `Epoch` | A rank-agreement point: every rank involved learns whether any of them failed (`reason`) | after every `Transport`; before a distributed remap | yes |
| `Remap` | Spatial transfer between regions (`remap_flags` as `C`/`M`/`H`, `method`) | differing regions, any `remap()` flag, or a distributed remap | no |
| `Constrain` | Check the conservation of a remapped field | a `Remap` of a field whose `conservation` is not `None`, except for pointwise methods | yes |
| `Diagnose` | Record transfer diagnostics | every `Remap` | no |
| `Merge` | Combine writers of one field (`merge` op, shown as `reason`) | `merge()` with two or more writers | no |
| `Relax` | Under-relax (`omega`); its output is the loop-carried value of an iteration region | `relax()` | no |
| `Residual` | The difference between sweeps of one region channel | each channel of an iteration region | no |
| `WindowAdvance` | One window member's advance across the window; no inputs, `attrs.window_import` names the arc it reads as metadata | each member of a window-scoped region | no |
| `ExternalDelivery` | A transfer the application performs itself (LowFive); its input is an ordering edge only | a channel whose source field has `hdf5_path` | no |
| `BuildRemapHalo` | Setup-time exchange of mesh geometry for a distributed remap; no inputs, no output, stamped `t=0` | each channel that may remap across ranks | no |
| `Accelerate` | Declared, never emitted: acceleration is a property of the region, not a node | — | no |
| `Map`, `Reduce`, `Closure` | Reserved for multiscale coupling; `schedule()` refuses them with `E-SCHED-MULTISCALE` ([](multiscale.md)) | — | yes |

`fuse::is_collective(OpKind)` returns the last column; `fuse::to_cstring(OpKind)` the name.

### Values and time levels

Every arc carries a `ValueType`:

```text
Instant<r1, kg*s^-3, Extensive, [WindowEnd-1, k], ranks=2>
   |      |     |         |            |            +-- component index whose ranks hold it (-1 = unplaced)
   |      |     |         |            +-- TimeLevel: anchor, offset in steps, iterate
   |      |     |         +-- Conservation (omitted when None)
   |      |     +-- Units (the field's, with any scale)
   |      +-- region id, "x3" appended for 3 components
   +-- ValueKind: Instant (one value) or Waveform (the samples of a window)
```

`TimeLevel{anchor, offset, iterate}` says *which* value this is: `[WindowEnd, k]` is the current
step's end at the current sweep, `[WindowEnd-1, k]` one step back (a `lag(1)`), `[WindowEnd, k-1]`
the previous sweep. A `Sink` requires the time level its field expects, and `type_check()` refuses
a producer that differs unless the channel carries `allow_time_level_mismatch()` (`E-IR-TYPE`).
`assignable_from(producer, consumer, allow_time_mismatch)` is the rule, and it also demands equal
kind, region, component count, units and conservation.

### How a channel becomes a chain

The builtin rule, `channel.default`, lowers each channel in a fixed order. Only the steps the
channel asks for emit nodes:

1. `ExternalDelivery`, if the source field is LowFive-bound.
2. `Scale`, if the connection check authorised a unit scale (`E-IR-UNLOWERABLE` if it authorised a
   conversion the IR cannot emit, `E-IR-NOSRC` if the source field has no producer).
3. `Sample` + `Accumulate`, for `accumulate()`.
4. `Sample` + `Interpolate`, for `spacetime()`.
5. `Extrapolate`.
6. The lag shift: no node, the value is retyped to `[WindowEnd-k, k]`.
7. `Transport` + `Epoch{after-transport}` when the next placement differs, or
   `Epoch{before-distributed-remap}` when a distributed remap moves the data itself.
8. `Remap` when regions differ, flags are set or the remap is distributed; otherwise
   `Interpolate{lag}` if a lag has nowhere else to live.
9. `Transport` + `Epoch{after-coupler-transport}` from a `remap_on` placement to the target.
10. `Constrain` (conserved fields) and `Diagnose`, after any `Remap`.
11. `Relax`.

After all channels: one `Merge` per merged field (a left fold of pairwise merges when there are more
than three writers, because an instruction holds three inputs), the window import `Sink`s, the
iteration regions, the remaining `Sink`s, and the type check. The lowering codes are `E-IR-NORULE`,
`E-IR-NOSRC`, `E-IR-UNLOWERABLE`, `E-IR-MULTI-WRITER`, `E-IR-MERGE-ANCHOR`, `E-IR-TYPE` and
`E-IR-REGION-EMPTY`.

**Why `Epoch` and `Constrain` appear although you never asked.** After every data movement the
ranks involved must agree on whether anything failed, or a rank that threw would leave its partners
blocked in the next receive. `Epoch` is that agreement, and `schedule()` refuses a plan where a
move has none (`E-SCHED-EPOCH-MISSING`) or one covering different ranks (`E-SCHED-EPOCH-PARTIAL`).
`Constrain` is the conservation check a field's `Conservation` declaration promised; `Diagnose`
records what the remap did. All three are visible in the dump precisely so that the program that
runs is never hidden.

## Reading `dump_text`

The complete example in [](graph-programming.md#a-complete-graph-validated) prints this IR:

```text
# fuse-ir v1
node 0   BuildRemapHalo              -> (none)  "heat_flux/build_remap_halo"
node 1   BuildRemapHalo              -> (none)  "wall_temperature/build_remap_halo"
node 2   Source                    -> %0 : Instant<r0, kg*s^-3, Extensive, [WindowEnd, k], ranks=0>  "fluid.q"
node 3   Source                    -> %1 : Instant<r1, K, Intensive, [WindowEnd, k], ranks=1>  "solid.T"
node 4   Sample       %0           -> %2 : Waveform<r0, kg*s^-3, Extensive, [WindowEnd, k], ranks=0>  "heat_flux/sample"
node 5   Accumulate   %2           -> %3 : Instant<r0, kg*s^-3, Extensive, [WindowEnd, k], ranks=0>  "heat_flux/accumulate" {accum=Mean}
node 6   Transport    %3           -> %4 : Instant<r0, kg*s^-3, Extensive, [WindowEnd-1, k], ranks=2>  "heat_flux/transport" {0->2}
node 7   Epoch        %4           -> %5 : Instant<r0, kg*s^-3, Extensive, [WindowEnd-1, k], ranks=2>  "heat_flux/epoch" {reason=after-transport}
node 8   Remap        %5           -> %6 : Instant<r1, kg*s^-3, Extensive, [WindowEnd-1, k], ranks=2>  "heat_flux/remap" {remap=C}
node 9   Transport    %6           -> %7 : Instant<r1, kg*s^-3, Extensive, [WindowEnd-1, k], ranks=1>  "heat_flux/transport_to_target" {2->1}
node 10  Epoch        %7           -> %8 : Instant<r1, kg*s^-3, Extensive, [WindowEnd-1, k], ranks=1>  "heat_flux/epoch_to_target" {reason=after-coupler-transport}
node 11  Constrain    %8           -> %9 : Instant<r1, kg*s^-3, Extensive, [WindowEnd-1, k], ranks=1>  "heat_flux/constrain"
node 12  Diagnose     %9           -> %10 : Instant<r1, kg*s^-3, Extensive, [WindowEnd-1, k], ranks=1>  "heat_flux/diagnose"
node 13  Extrapolate  %1           -> %11 : Instant<r1, K, Intensive, [WindowEnd, k], ranks=1>  "wall_temperature/extrapolate" {order=1}
node 14  Transport    %11          -> %12 : Instant<r1, K, Intensive, [WindowEnd, k], ranks=0>  "wall_temperature/transport" {1->0}
node 15  Epoch        %12          -> %13 : Instant<r1, K, Intensive, [WindowEnd, k], ranks=0>  "wall_temperature/epoch" {reason=after-transport}
node 16  Remap        %13          -> %14 : Instant<r0, K, Intensive, [WindowEnd, k], ranks=0>  "wall_temperature/remap"
node 17  Constrain    %14          -> %15 : Instant<r0, K, Intensive, [WindowEnd, k], ranks=0>  "wall_temperature/constrain"
node 18  Diagnose     %15          -> %16 : Instant<r0, K, Intensive, [WindowEnd, k], ranks=0>  "wall_temperature/diagnose"
node 19  Sink         %16          -> %17 : Instant<r0, K, Intensive, [WindowEnd, k], ranks=0>  "fluid.T_wall"
node 20  Sink         %10          -> %18 : Instant<r1, kg*s^-3, Extensive, [WindowEnd, k], ranks=1>  "solid.q_in"
```

One line per node, in topological order:

```text
node N  Kind  %in, ...  -> %out : <ValueType>  "label"  {non-default attrs}
```

- `%k` is arc `k`. A node with no output prints `(none)`; a node with no inputs prints nothing
  before the arrow.
- The label is `component.field` for a boundary node and `channel/step` for everything a channel
  emitted, so every node can be traced back to a line of your graph. `Node::origin` holds the
  channel id itself.
- Attributes print only when they differ from the default: `{accum=Mean}`, `{order=1}`,
  `{remap=C}` (flags `C`onservative, `M`onotone, `H`ighOrder), `{scale=1/1000, reason=inferred}`,
  `{omega=0.5}`, `{lag=1}`, `{method=...}`, `{window_import=%13}`, and `{0->2}` on a `Transport`
  (source and destination component indices).

Reading the example: `fluid.q` (%0) is sampled four times into a waveform (%2) and reduced to a
mean (%3). The lag on `heat_flux` shows up as the type change at node 6: `[WindowEnd, k]` becomes
`[WindowEnd-1, k]` with no node of its own. The value moves to the coupler (`0->2`), agrees
(`Epoch`), is remapped conservatively onto region `r1`, moves to the solid (`2->1`), agrees again,
is checked and recorded, and lands in `solid.q_in`. `wall_temperature` is the simpler chain:
extrapolate, move, agree, remap, check, record, deliver. The final `Sink` for `solid.q_in` requires
`[WindowEnd, k]` and receives `[WindowEnd-1, k]`; the channel's
`allow_time_level_mismatch("explicit staggered scheme")` is what let that through.

An iteration region adds `Residual` nodes and a trailing region line (from `examples/dn_heat`):

```text
node 14  Residual     %6           -> %12 : Instant<r1, kg*s^-3, Extensive, [WindowEnd, k], ranks=1>  "flux/residual"
node 15  Residual     %11          -> %13 : Instant<r0, K, Intensive, [WindowEnd-1, k], ranks=0>  "wall_temperature/residual"
...
region 0 "dirichlet_neumann" body=[4,5,6,7,8,9,10,11,12,13] residuals=[%12,%13] carried=[] max_iter=1500
```

`body` lists the node ids re-executed every sweep, `residuals` the arcs the convergence criterion
measures, `carried` the arcs that survive from one sweep to the next (the output of each `Relax`;
empty here, where the `lag(1)` alone breaks the loop).

A merged field shows the writers as several inputs:

```text
node 20  Merge        %7, %16      -> %17 : Instant<r1, kg*s^-3, Extensive, [WindowEnd, k], ranks=1>  "b.q_in/merge" {reason=Sum}
```

A window-scoped region replaces each member's `Source` with a `WindowAdvance` whose value is the
member's whole trajectory over the window (`r0x2`: two samples per entity here), and whose
`window_import` points at the `Sink` it reads from the other member:

```text
node 14  WindowAdvance              -> %0 : Instant<r0x2, K, Extensive, [WindowEnd, k], ranks=0>  "a.a_out/window" {window_import=%13}
node 12  Sink         %6           -> %12 : Instant<r1x2, K, Extensive, [WindowEnd, k], ranks=1>  "b.b_in/window"
```

### Rendering the graph

`dump_graphviz(dag)` prints a `digraph fuse_ir` with one box per node (kind, label, attributes)
and one edge per arc, labelled `%k`:

```cpp
std::ofstream("coupling.dot") << fuse::dump_graphviz(lowered.dag);
```

```sh
dot -Tsvg coupling.dot -o coupling.svg
```

## Reading the plan

`schedule()` expands the DAG over one *coupling period* and flattens it into instructions.
`plan.to_string()` prints:

```text
# fuse-plan v1
period=1 lockstep=no instructions=48 buffers=19 coupler_elements=480 regions=0
buffer 0   %0  elements=32  components=1  region=r0  owner=component  persistent=no  "fluid.q"
buffer 2   %2  elements=32  components=1  region=r0  owner=coupler  persistent=yes  "heat_flux/sample"
buffer 6   %6  elements=32  components=1  region=r1  owner=coupler  persistent=yes  "heat_flux/remap"
  0  t=0  BuildRemapHalo  node=0
  2  t=0  Source  node=2  out=%0
  4  t=0  Sample  node=4  in=[%0]  out=%2  after-ticks
  5  t=1/4  Source  node=2  out=%0
  6  t=1/4  Sample  node=4  in=[%0]  out=%2  after-ticks
  7  t=1/4  Extrapolate  node=13  in=[%1]  out=%11  {order=1}  after-ticks
  ...
  32  t=1  Accumulate  node=5  in=[%2]  out=%3  {accum=Mean}
  33  t=1  Transport  node=6  in=[%3]  out=%4  {0->2}
  34  t=1  Epoch  node=7  in=[%4]  out=%5  {reason=after-transport}
  35  t=1  Remap  node=8  in=[%5]  out=%6  {remap=C}
  ...
  47  t=1  Sink  node=20  in=[%10]  out=%18
```

- **Header.** `period` is the exact LCM of every non-coupler step (here 1/4 and 1). `lockstep=yes`
  means all steps are equal, the fast path. `coupler_elements` is what the coupler itself allocates,
  in elements; `regions` the number of iteration regions.
- **Buffers.** One per arc, in arc order, so `buffer k` describes `%k`. `owner=component` marks
  the `Source` and `Sink` outputs that alias registered arrays (never allocated by FUSE);
  `persistent=yes` marks a buffer whose value crosses a period boundary (a waveform, or a lagged
  value) and so cannot be recycled within a period.
- **Instructions.** `i  t=<time>  Kind  node=N  in=[...]  out=%k  {attrs}  [after-ticks]
  [enter=r]  [exit=r]`. Instructions are sorted by time, then phase, then dependence order. Node 2
  (`Source fluid.q`) appears four times because the fluid steps four times per period; node 5
  (`Accumulate`) once, at the solid's step end. `after-ticks` is the phase: within one instant,
  every delivery that is causally available runs before the components tick, and every `Sample`
  of the new state runs after. `enter=r` / `exit=r` bracket one sweep of iteration region `r`.

The same data is available as `Instruction{kind, out, in[3], n_in, attrs, time, src_step,
dst_step, phase, region_enter, region_exit, origin_node}`; `src_step` and `dst_step` (the step index
of each side within the period) are not printed.

## Walking the DAG and the plan from code

The program below builds a two-component graph whose fields differ by a unit ratio, lowers and
schedules it with a constant `RegionSizes`, and reads the result both ways. It compiles against the
headers listed at the top of this page.

```cpp
const fuse::OperatorDAG& dag = lowered.dag;

// Downstream, in dependence order.
for (const fuse::NodeId id : dag.topological_order()) {
  const fuse::Node& n = dag.node(id);
  std::cout << fuse::to_cstring(n.kind) << " \"" << n.label << "\"";
  if (n.origin.valid()) std::cout << " from channel '" << g.channel_at(n.origin).name << "'";
  if (n.output.valid()) {
    std::cout << " -> " << dag.arc(n.output).type.to_string()
              << ", read by " << dag.consumers_of(n.output).size() << " node(s)";
  }
  std::cout << "\n";
}
// Upstream: from a Sink back to its producing Source.
for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
  const fuse::Node& n = dag.node(fuse::NodeId{i});
  if (n.kind != fuse::OpKind::Sink) continue;
  fuse::NodeId cur = n.id;
  int hops = 0;
  while (!dag.node(cur).inputs.empty()) {
    cur = dag.arc(dag.node(cur).inputs.front()).producer;
    ++hops;
  }
  std::cout << "Sink \"" << n.label << "\" is " << hops << " hops from \""
            << dag.node(cur).label << "\"\n";
}

// Schedule with a constant region size (a mesh backend supplies the real one).
const fuse::ClockSet clocks{g};
const fuse::RegionSizes sizes = [](fuse::RegionId, int) -> std::size_t { return 16; };
const fuse::ScheduleResult sched = fuse::schedule(g, dag, clocks, sizes);
if (!sched.ok()) { std::cerr << sched.diags.to_string(); return 1; }
const fuse::ExecutionPlan& plan = sched.plan;
std::cout << "period=" << plan.period().to_string()
          << " lockstep=" << (plan.lockstep() ? "yes" : "no")
          << " instructions=" << plan.instructions().size()
          << " regions=" << plan.region_count() << "\n";
std::size_t collective = 0;
for (const fuse::Instruction& ins : plan.instructions()) {
  if (fuse::is_collective(ins.kind)) ++collective;
  const fuse::Node& origin = dag.node(ins.origin_node);
  if (ins.kind == fuse::OpKind::Scale) {
    std::cout << "Scale at t=" << ins.time.to_string() << " by "
              << ins.attrs.scale.to_string() << " (" << ins.attrs.reason
              << "), node \"" << origin.label << "\"\n";
  }
}
std::cout << collective << " collective instruction(s) per period\n";
for (const fuse::BufferDecl& b : plan.buffers().buffers) {
  if (!b.is_component_owned) {
    std::cout << "coupler buffer %" << b.arc.value() << " " << b.elements << "x"
              << b.components << " \"" << b.label << "\"\n";
  }
}
for (const std::string& e : fuse::epoch_sequence(plan, g, /*rank=*/1)) {
  std::cout << "rank 1 epoch " << e << "\n";
}
```

Expected output (exact digits can vary by platform):

```text
BuildRemapHalo "flux/build_remap_halo" from channel 'flux'
Source "a.q" -> Instant<r0, kg*s^-3, Extensive, [WindowEnd, k], ranks=0>, read by 1 node(s)
Scale "flux/scale" from channel 'flux' -> Instant<r0, 1000*kg*s^-3, Extensive, [WindowEnd, k], ranks=0>, read by 1 node(s)
Transport "flux/transport" from channel 'flux' -> Instant<r0, 1000*kg*s^-3, Extensive, [WindowEnd, k], ranks=1>, read by 1 node(s)
Epoch "flux/epoch" from channel 'flux' -> Instant<r0, 1000*kg*s^-3, Extensive, [WindowEnd, k], ranks=1>, read by 1 node(s)
Remap "flux/remap" from channel 'flux' -> Instant<r1, 1000*kg*s^-3, Extensive, [WindowEnd, k], ranks=1>, read by 1 node(s)
Constrain "flux/constrain" from channel 'flux' -> Instant<r1, 1000*kg*s^-3, Extensive, [WindowEnd, k], ranks=1>, read by 1 node(s)
Diagnose "flux/diagnose" from channel 'flux' -> Instant<r1, 1000*kg*s^-3, Extensive, [WindowEnd, k], ranks=1>, read by 1 node(s)
Sink "b.q_in" from channel 'flux' -> Instant<r1, 1000*kg*s^-3, Extensive, [WindowEnd, k], ranks=1>, read by 0 node(s)
Sink "b.q_in" is 7 hops from "a.q"
period=1 lockstep=yes instructions=9 regions=0
Scale at t=1 by 1/1000 (inferred), node "flux/scale"
3 collective instruction(s) per period
coupler buffer %1 16x1 "flux/scale"
coupler buffer %2 16x1 "flux/transport"
coupler buffer %3 16x1 "flux/epoch"
coupler buffer %4 16x1 "flux/remap"
coupler buffer %5 16x1 "flux/constrain"
coupler buffer %6 16x1 "flux/diagnose"
rank 1 epoch 1:after-transport
```

The accessors: `node_count()`, `arc_count()`, `node(NodeId)`, `arc(ArcId)`,
`consumers_of(ArcId)`, `topological_order()` (Kahn's algorithm, ties broken by ascending id),
`is_acyclic()`, `region_count()` and `region_at(RegionIdx)` returning
`IterationRegion{name, body, residuals, carried, max_iterations}`. Upstream is
`arc(input).producer`; downstream is `consumers_of(output)`. References returned by `node()` and
`arc()` are invalidated by a later `add_node()`/`add_arc()`. `epoch_sequence(plan, g, rank)` and
`constrain_sequence(plan, g, rank)` list, as `"time:reason"` strings, the agreement and
conservation points a given rank takes part in; comparing them across ranks is how to confirm that
every rank runs the same collectives in the same order.

### Dumps as golden files

Both dumps are deterministic (classic locale, topological order with id tie-breaks, arcs in
creation order), so they make good golden files. The pattern the library's own tests use:

```cpp
const std::string actual = fuse::dump_text(lower(graph).dag);
if (std::getenv("UPDATE_GOLDEN")) { write_file(golden_path, actual); return; }
EXPECT_EQ(actual, read_file(golden_path)) << "IR changed; review the diff";
```

A dump changes only when the graph or the compiler changes, so a failing golden test tells you one
of those happened. To see what a graph edit did to the coupling, dump before and after and `diff`
the two files: a new `Scale` line means a unit changed, a new `Transport`/`Epoch` pair means a
placement changed, a new `Constrain` means a field became conserved, a changed `period=` means a
time step changed. Running an example with `--dump` prints both dumps
([](getting-started.md#3-look-at-what-fuse-built)).

## What `schedule()` proves and refuses

`schedule()` runs a sequence of gates and stops at the first failure; the plan is empty unless
every gate passed. In order: window-region overlap, each window scope's legality
(`validate_coupling_window()`), time expansion, the acyclicity proof, topological rank, the
multiscale refusal, instruction arity, buffer planning, sorting, region grouping, the dependence
order check, window causality, region marking, and the `Epoch` guard.

| Code | Meaning | Fix |
|---|---|---|
| `E-SCHED-PERIOD` | A step does not divide the common period, or a tick is not representable as an exact 64-bit `Rational` | Use steps with a small common multiple; check for a denominator typo |
| `E-SCHED-INTERVAL` | A step is not positive | Fix the component's `interval` |
| `E-SCHED-EXPLODE` | More than `kMaxTimedNodes` (10⁶) node instances in one period | Steps are too far apart (1 against 1/1000000); use a window region or coarser ratios |
| `E-SCHED-EXPAND` | The `ClockSet` was built from a different graph, a channel endpoint has no owner, or a node without a channel has no placement | Build `ClockSet{g}` from the same graph; `validate()` first |
| `E-SCHED-SPACETIME` | `accumulate` (non-`Instant`) and `spacetime` on one channel | Drop one of the two |
| `E-SCHED-CYCLE` | The expanded period has a dependence cycle no lag breaks (an algebraic loop); the message lists the cycle | `lag(1)` + `allow_time_level_mismatch()` on one channel, or an iteration region with a `relax()` or lag inside it |
| `E-SCHED-PLAN` | The DAG itself has a cycle, or the expansion names a node outside it | An internal inconsistency; report it with the dump |
| `E-SCHED-MULTISCALE` | A `Map`, `Reduce` or `Closure` node | Not supported; see [](multiscale.md) |
| `E-SCHED-ARITY` | A node with more than three inputs | Only a custom lowering rule can produce this; fold the inputs pairwise |
| `E-SCHED-BUFFERS` | No `RegionSizes` callback, an arc with no region or fewer than one component, or a buffer size that overflows | Pass `make_region_sizes()` or a lambda; check `FieldSpec::components` |
| `E-SCHED-REGION` | A region's body is not one contiguous run per sweep: its channels sit on two clocks without a window, two regions overlap, or an outside channel reads a value the region re-advances | One clock per region or a window scope; route the interleaving channel from outside the loop |
| `E-SCHED-ORDER` | An instruction reads an arc no earlier instruction wrote | An internal defect; report it with the dump |
| `E-SCHED-WINDOW-EMPTY` | A window-scoped region has no channels | Add the channels |
| `E-SCHED-WINDOW-AMBIGUOUS` | Both or neither of `window_component` and `explicit_interval` set | Set exactly one |
| `E-SCHED-WINDOW-COMPONENT` | `window_component` or a policy names a component that owns no end of a region channel | Name a member |
| `E-SCHED-WINDOW-INTERVAL` | The window length is not positive | Fix the interval |
| `E-SCHED-WINDOW-POLICY` | A member has no `DtPolicy` entry | Add it to `policies` |
| `E-SCHED-WINDOW-ORDER` | `order < 2` | Use 2 or more |
| `E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG` | Gauss-Seidel with no lagged channel | `lag(1)` one channel, or use Jacobi |
| `E-SCHED-WINDOW-SHORTEN` | An `Adjustable` member has a non-positive step | Fix the interval |
| `E-SCHED-WINDOW-CAPACITY` | A member needs more samples per window than the waveform holds (4096) | A longer step or a shorter window |
| `E-SCHED-WINDOW-OVERLAP` | A component or channel belongs to two window regions | One window region per component |
| `E-SCHED-WINDOW-CAUSALITY` | A `WindowAdvance` reads a value its window instance has not delivered | Lag the back channel, or use Jacobi |
| `E-SCHED-EPOCH-MISSING`, `E-SCHED-EPOCH-PARTIAL` | A data move with no following agreement point, or one over a different rank set | Only a custom lowering rule can produce this; emit `Epoch` after `Transport` |

**Period, instants and phases.** `ExecutionPlan::period()` is the exact rational LCM of every
non-coupler step; `ClockSet::checked_common_period()` refuses a period longer than
`kMaxPeriodTicks` (1 000 000) ticks of the finest step with `E-SCHED-PERIOD`. Every instruction
has an exact `time` in `[0, period]` and a `phase` (`BeforeTicks`, then `AfterTicks`) that orders it
within the instant. Each DAG node is instantiated once per occurrence in the period: a source-clock
node once per source step, a target-clock node once per target step, a window node once per window.

**Slabs.** For a channel between two clocks the period is cut into *slabs*, maximal intervals that
cross no step boundary on either side (`decompose(src_dt, dst_dt, period)`, at most
`kMaxSlabsPerPeriod` = 100 000). An instant chain is stamped at slab ends; a slab that ends on a
target step ahead of the source step it belongs to is delivered `AfterTicks` and consumed one
target step later. An accumulating or space-time chain is stamped at the end of the target step it
closes and is always `BeforeTicks`.

## Extending the lowering

The lowering is a registry of rules, tried in descending priority, and the first rule that returns
`true` owns the channel:

```cpp
using LoweringRule = std::function<bool(LoweringContext&, ChannelId)>;
class LoweringRegistry {
  void add(std::string name, int priority, LoweringRule rule);
  bool apply_first_match(LoweringContext& ctx, ChannelId c) const;
  std::size_t size() const;
  static LoweringRegistry& builtin();     // the registry lower() uses
};
```

The builtin rule is `channel.default` at priority 100. `lower()` always consults `builtin()`, so a
rule meant for production is added there, with a higher priority, before the first `lower()` call.
A rule that returns `false` declines and the next rule is tried; a channel no rule claims is
`E-IR-NORULE`.

A rule receives the `LoweringContext`: the graph, the DAG to append to, the `Diagnostics` to report
into, and the maps the rest of `lower()` relies on. It may emit any existing `OpKind` through
`ctx.dag.add_arc()` and `ctx.dag.add_node()`, and it owes the same bookkeeping the builtin rule
performs:

- Read the source through `ctx.source_arc[source_field]` (what the component produces), never
  `ctx.field_arc` (what was last delivered to it); report `E-IR-NOSRC` and return `true` if it is
  missing.
- Record `ctx.chain_entry_arc[channel] = <first arc of your chain>`.
- When done, append your last arc to `ctx.field_writers[target_field]`, set
  `ctx.writer_arc_channel[last_arc] = channel` and `ctx.field_arc[target_field] = last_arc`.
- Keep the value type honest: `type_check()` compares your last arc to the `Sink`'s type, and
  `schedule()` applies every gate above to what you emitted, including the `Epoch` guard after any
  `Transport`.

A minimal rule that claims channels named `tap:*` when both ends share a region and a processor
set, and lowers each to a single `Diagnose` of the source value:

```cpp
bool lower_tap_channel(fuse::LoweringContext& ctx, fuse::ChannelId cid) {
  const fuse::CouplingGraph& g = ctx.graph;
  const fuse::Channel& ch = g.channel_at(cid);
  if (ch.name.rfind("tap:", 0) != 0) return false;                 // not ours
  const fuse::FieldSpec& src = g.field(ch.source);
  const fuse::FieldSpec& dst = g.field(ch.target);
  const bool same_ranks = g.component(g.owner_of(ch.source)).ranks
                              .same_as(g.component(g.owner_of(ch.target)).ranks);
  if (src.region != dst.region || !same_ranks) return false;      // decline

  const auto it = ctx.source_arc.find(ch.source.value());
  if (it == ctx.source_arc.end()) {
    ctx.diags.add(fuse::Severity::Error, "E-IR-NOSRC",
                  "source field has no producing arc", "channel '" + ch.name + "'");
    return true;                                                   // claimed, failed
  }
  const fuse::ArcId in = it->second;
  ctx.chain_entry_arc[cid.value()] = in;

  fuse::ValueType type = ctx.dag.arc(in).type;
  type.rank_set = static_cast<int>(g.owner_of(ch.target).value());
  const fuse::ArcId out = ctx.dag.add_arc(type, ch.name + "/tap");
  ctx.dag.add_node(fuse::OpKind::Diagnose, {in}, out, fuse::OpAttrs{}, cid,
                   ch.name + "/tap");

  // The bookkeeping every rule owes the rest of lower().
  ctx.field_writers[ch.target.value()].push_back(out);
  ctx.writer_arc_channel[out.value()] = cid;
  ctx.field_arc[ch.target.value()] = out;
  return true;                                                     // claimed
}

// Before the first lower():
fuse::LoweringRegistry::builtin().add("channel.tap", 200, lower_tap_channel);
```

Lowering a graph with a channel `tap:pressure` between two components on the same ranks prints:

```text
# fuse-ir v1
node 0   Source                    -> %0 : Instant<r0, kg*m^-1*s^-2, [WindowEnd, k], ranks=0>  "a.p"
node 1   Diagnose     %0           -> %1 : Instant<r0, kg*m^-1*s^-2, [WindowEnd, k], ranks=1>  "tap:pressure/tap"
node 2   Sink         %1           -> %2 : Instant<r0, kg*m^-1*s^-2, [WindowEnd, k], ranks=1>  "b.p_in"
```

and the result schedules. A rule is the right tool for a new *composition* of the existing
operators (a different order, a different placement, a treatment the builder cannot express). It is
not a way to add a new operator: the interpreter in the `Runner` knows exactly the `OpKind` set
above.

**The Newton substrate.** `fuse::num::register_newton_substrate(NewtonSubstrate kind,
NewtonSolveFn fn)` (`fuse/num/newton_krylov.hpp`) replaces the solver behind
`NewtonSubstrate::Native` or `NewtonSubstrate::Petsc`; `fuse::petsc::install()` is the one
registration the library ships ([](convergence.md)). Only those two slots exist.

### What is not extensible

- The `OpKind` set, `MergeOp`, `AccumOp`, `TemporalBasis` and `RemapFlags`.
- The `AccelKind` set: accelerators are built by a closed switch over `AccelConfig::kind`.
- The interpreter, the schedule gates and the instruction arity of three inputs.
- The `CouplingGraph` structure (no subclass hooks) and the `OperatorDAG` (append-only; no
  mutation or removal).
- Graph or DAG transformation passes: there is no rewrite API between `lower()` and `schedule()`.
- Conversions beyond an exact unit scale: affine units, sign flips, component selection or
  reordering, clipping and conservation enforcement are recognised by the connection checker but
  not emitted by the IR; a channel that needs one is refused with `E-IR-UNLOWERABLE`.
- `Map`, `Reduce` and `Closure` are reserved kinds that `schedule()` refuses with
  `E-SCHED-MULTISCALE` ([](multiscale.md)).

## See also

- [](graph-programming.md): the declarations this page compiles.
- [](quickstart.md#seeing-what-the-compiler-did) and
  [](getting-started.md#3-look-at-what-fuse-built): the dumps in a complete program.
- [](concepts.md#parallel-layout) for why every move is followed by an `Epoch`;
  [](placement.md) for `Transport` and couplers.
- [](coupling-windows.md) for `WindowAdvance`, [](combining-writers.md) for `Merge`,
  [](convergence.md) for regions and residuals, [](lowfive.md) for `ExternalDelivery`.
- [](power-users.md#diagnostics) for runtime statistics once the plan runs.
