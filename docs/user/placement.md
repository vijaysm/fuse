# Where a component runs

This page is for anyone placing components on specific MPI ranks: how to declare a processor set,
what the library does with it, how to route remapping through a separate coupler, and what to put
in `RunnerConfig` so that a multi-rank run actually moves data.

A component declares the ranks it runs on as a `fuse::RankSet`
(`include/fuse/model/component.hpp`). That declaration is an explicit set, not a range, because
real coupled jobs are laid out in ways a range cannot describe.

## Declaring a processor set

Every form of `RankSet`, in C++:

```cpp
// Contiguous: ranks 0..31. {first, count} converts implicitly in a declaration.
g.add_component({.name = "fluid", .ranks = {0, 32}});
g.add_component({.name = "fluid", .ranks = fuse::RankSet{0, 32}});

// Any set. Order does not matter: it is sorted and de-duplicated, so you can pass
// ranks in whatever order your launcher reports them.
g.add_component({.name = "solid", .ranks = fuse::RankSet{{0, 8, 16, 24}}});

// Unplaced in this run (count <= 0). Legal only for a component with no channels.
g.add_component({.name = "probe", .ranks = fuse::RankSet{0, 0}});

// The default is rank 0.
g.add_component({.name = "serial"});   // .ranks == fuse::RankSet{0, 1}
```

A `count` above 2^24 or a `first + count` that overflows `int` is refused at construction with
`E-RANKSET-RANGE`.

In a configuration file, both forms parse:

```yaml
components:
  - name: fluid
    ranks: {first: 0, count: 32}
  - name: solid
    ranks: [0, 8, 16, 24]
```

Queries on a `RankSet`:

```cpp
const fuse::RankSet a{0, 4};                  // {0,1,2,3}
const fuse::RankSet b{{2, 3, 4, 5}};
a.size();            // 4
a.contains(3);       // true
a.overlaps(b);       // true: share 2 and 3
a.same_as(b);        // false
a.is_contiguous();   // true
a.index_of(2);       // 2: position in the set, which is what indexes per-rank data
fuse::rank_union(a, b);         // {0,1,2,3,4,5}: the ranks an exchange between them involves
fuse::rank_intersection(a, b);  // {2,3}
```

## Shared, identical and disjoint processor sets

Three cases, and the library treats them differently because the physics does:

| Processor sets | What a field transfer between them costs |
|---|---|
| identical | no data move: every value is already on the rank that wants it |
| partially shared | a move for the ranks that differ, a local copy for the ranks in common |
| disjoint | a move |

The distinction is `same_as` versus `overlaps`, and it matters more than it
looks. A condition that suppressed the move whenever the processor sets merely
*touched* would deliver nothing at all to the ranks they do not share.

Partially shared processor sets are delivered by a **peer-wise** plan: a rank that
is a source for some entities and a destination for others is described peer by
peer rather than by one list of ids, and the matches whose two ends land on the
same rank are copied locally instead of travelling. Identical and disjoint
processor sets use the matched plan; on a pattern both can express, the two
deliver identical values.

### Reading what the plan does

`fuse::schedule()` writes the placement decision into the instruction stream, so you can see
it before running anything (`sched.plan.to_string()`; see [](quickstart.md)). The same channel
`flux` (`fluid.q -> solid.q_in`, conservative remap) compiled under three placements:

Identical sets (`fluid` and `solid` both on `{0..3}`): no `Transport`, the remap runs in place.

```
  0  t=0  Source  node=0  out=%0
  1  t=1  Remap  node=1  in=[%0]  out=%1  {remap=C}
  2  t=1  Constrain  node=2  in=[%1]  out=%2
  3  t=1  Diagnose  node=3  in=[%2]  out=%3
  4  t=1  Sink  node=4  in=[%3]  out=%4
```

Disjoint sets (`fluid` on `{0..3}`, `solid` on `{4..7}`), and likewise partially shared sets
(`solid` on `{2..5}`): one `Transport` from component 0 to component 1, followed by an `Epoch`
(the rank-agreement point) before the remap runs on the target's own decomposition.

```
  0  t=0  BuildRemapHalo  node=0
  1  t=0  Source  node=1  out=%0
  2  t=1  Transport  node=2  in=[%0]  out=%1  {0->1}
  3  t=1  Epoch  node=3  in=[%1]  out=%2  {reason=after-transport}
  4  t=1  Remap  node=4  in=[%2]  out=%3  {remap=C}
  5  t=1  Constrain  node=5  in=[%3]  out=%4
  6  t=1  Diagnose  node=6  in=[%4]  out=%5
  7  t=1  Sink  node=7  in=[%5]  out=%6
```

The partially shared plan has the same shape as the disjoint one; the difference (which ranks
copy locally and which send) is decided inside the `Transport` from the two `RankSet`s, not by
the instruction stream.

## An empty processor set means "switched off", and takes its field transfers with it

A processor set of zero ranks is legal and says *this component does not run here,
this time*. It is the supported way to leave a component out of one run
without editing the rest of the graph. The graph is identical on every rank, so a component
that another rank hosts still appears in yours.

It is legal **only for a component with no field transfers**. A field transfer out of a
component placed on no rank has nothing to read from, and a field transfer into one
has nowhere to deliver; `validate()` rejects both with `E-PLACEMENT-EMPTY`,
naming the component and the field transfer:

```
error E-PLACEMENT-EMPTY: component 'solid' is placed on no rank, so this channel has nowhere
to deliver; an empty placement is legal only for a component with no channels -- give it a
placement, or remove the channel for runs where it is switched off [channel 'flux']
```

So when you switch a component off, remove its field transfers for that run too. The rejection is
a modelling check, not a transport limitation: it fires for every transport, before any
communicator exists, and appears even without MPI.

## What the library does with a processor set

- **Stepping the component**: `advance()` is called only on the processes of
  the component's own processor set. A process outside it still constructs the
  `Component` object (every process holds the same graph), but never steps it,
  and a component placed on no processes is never stepped at all. Without a
  base communicator all components run in the one process.
- **Deciding whether a move is needed**: identical processor sets, no move.
- **Forming the exchange's communicator**: the union of the two processor sets,
  built collectively over that union only, never over the whole job. Ranks in
  neither processor set take no part.
- **Agreeing on failure**: the same union, so a synchronisation point (`Epoch`) reduces
  over exactly the ranks of its own exchange. A shared rank counts once.
- **Indexing per-rank data**: by *position* in the processor set (`index_of`), not by rank
  number, since a strided set has no arithmetic relationship between the two.

## Each component has its own decomposition of a shared mesh

Two components that share a mesh do not share its partition. Each holds the
mesh laid out over **its own** ranks, so a rank placed in both holds two
pieces, one per component. This is what makes partially overlapping
processor sets work: with A on ranks 0–2 and B on ranks 1–3, each side covers the
whole mesh entity set over its own ranks, and the transport moves what differs and
copies locally what both hold.

Components with identical processor sets always get identical decompositions,
which is why no data move is needed between them.

Remap weights are built on the processor set's own decomposition: the plan
carries the component, every weight builder resolves its meshes through
`placed_view`, and two processor sets never share a cached matrix.

**The MOAB backend** builds a decomposition per component too, once told
about the processor sets: `MoabMeshBackend::set_placements(placements)`, one
`RankSet` per component, indexed the same way `graph.component_count()`
counts them (including a coupler), and declared before the first `load()`.

With a real communicator adopted (`set_communicator`), each declared
processor set gets its own `moab::Core` and `moab::ParallelComm`, split from the
base communicator over exactly that processor set's ranks, and `load()` reads the same source
into it independently: a genuine second partitioned read, not a relabelling of the
default one. A rank outside a processor set holds none of it: `placed_view`
answers an empty view there, never the default decomposition. Without a
communicator, "this rank" is rank 0 of an effective one-rank job, so a
processor set that names it is the default view and one that does not is empty.

Weight builders resolve every side through `placed_view`, so a weight build
issued against a MOAB processor set's own decomposition uses it automatically.

:::{note}
Limitation: assembling and comparing a conservative weight matrix between two
independently-partitioned MOAB processor sets is not covered by the test suite;
check the weights' coverage fields (see [](geometry.md)) before relying on one.
:::

## A separately placed coupler

Every field transfer's Remap has to run somewhere. By default it runs on the
target's own processor set: the value moves source -> target and is remapped
there. A **coupler** is a third processor set a field transfer can name instead, so the
value moves source -> coupler -> target, remapped on the coupler's own
decomposition of both meshes.

```cpp
const auto coupler = g.add_coupler({.name = "coupler", .ranks = {8, 4}});
g.channel(ch).remap(fuse::RemapFlags::Conservative).remap_on(coupler);
```

or in YAML:

```yaml
couplers:
  - name: coupler
    ranks: {first: 8, count: 4}
channels:
  - name: flux
    source: fluid.q
    target: solid.q_in
    remap: {flags: [Conservative], remap_on: coupler}
```

A coupler is a processor set, not a physics component: it declares no fields
(`E-COUPLER-FIELDS` if it does), has no time grid, and is never advanced. It gets a communicator
like any other processor set, so its ranks take part in the field transfer's data moves and
synchronisation points, but nothing ever calls `advance()` on it and no
`Component` implementation needs to exist for it: a `ComponentSet` with no entry for the
coupler initializes and runs.

The compiled plan for the disjoint example above with `remap_on(hub)`, `hub` on `{8, 9}`
(component 2): two moves, with the remap between them.

```
  0  t=0  BuildRemapHalo  node=0
  1  t=0  Source  node=1  out=%0
  2  t=1  Transport  node=2  in=[%0]  out=%1  {0->2}
  3  t=1  Epoch  node=3  in=[%1]  out=%2  {reason=after-transport}
  4  t=1  Remap  node=4  in=[%2]  out=%3  {remap=C}
  5  t=1  Transport  node=5  in=[%3]  out=%4  {2->1}
  6  t=1  Epoch  node=6  in=[%4]  out=%5  {reason=after-coupler-transport}
  7  t=1  Constrain  node=7  in=[%5]  out=%6
  8  t=1  Diagnose  node=8  in=[%6]  out=%7
  9  t=1  Sink  node=9  in=[%7]  out=%8
```

`remap_on` may also name the field transfer's own source or target component
directly: "remap on the source's decomposition, then transport to the
target" is the same rule with the coupler set to an existing processor set rather than a
new one. Naming any other physics component is refused with `E-CHANNEL-REMAP-ON`.

**When a coupler costs something.** The rule is one rule, not a special case:
source -> C -> target, with any hop between *identical* processor sets omitted,
exactly like the identical/disjoint table above.

| `remap_on` | What it costs over the default |
|---|---|
| unset, or the target's own component | nothing: the default |
| a coupler with the same ranks as the target | no extra move (the second hop is `same_as`), but the Remap runs under the coupler's own processor-set identity, not the target's |
| the field transfer's own source component | the second hop only: remap on the source's decomposition, then one move to the target |
| any other processor set | two moves: source -> C, then C -> target |

A `remap_on` that could never change anything (no remap flags, matching
mesh entity sets, and a method that does not itself move data between processor
sets) has no effect at all, and `validate()` reports it with
`W-CHANNEL-REMAP-ON-UNUSED` rather than silently routing data through an extra hop nothing needed:

```
warning W-CHANNEL-REMAP-ON-UNUSED: channel names a remap_on placement, but no Remap operator
would be emitted for it (no remap flags, matching regions, and no distributed-remap placement
move); the setting is unused [channel 'flux']
```

A coupler placed on no ranks is legal on its own, by the same rule as any
other empty processor set above, until a field transfer's `remap_on` actually routes
through it, at which point `E-PLACEMENT-EMPTY` applies exactly as it does for
a field transfer's source or target.

## Running on more than one rank

Placement is a property of the graph; moving data is a property of the `Runner`. Three
`RunnerConfig` fields (`include/fuse/runtime/runner.hpp`) turn a placed graph into a multi-rank
run:

| Field | Meaning |
|---|---|
| `transport` | the transport's registered name: `"copy"` (the default, one process, whole arcs copied), `"mpi"`, or `"lowfive"` |
| `base_communicator` | `MPI_Comm_c2f()` of a communicator whose group contains every rank of every component. Set it and `Transport` redistributes between processor sets; leave it unset and every component runs in the one process. A Fortran handle, so that `<mpi.h>` stays out of the public headers. |
| `transport_setup` | a callback `Result<void>(Transport&)` that `initialize()` calls exactly once, right after the transport is constructed, for the backend-specific setup the `Transport` interface cannot express (`MpiTransport::set_communicator`, `LowFiveTransport::set_location`). Empty for `"copy"`. |

The pattern, from `examples/layout_driver/main.cpp`:

```{literalinclude} ../../examples/layout_driver/main.cpp
:language: cpp
:start-at: "fuse::RunnerConfig cfg;"
:end-at: "  };"
```

`fuse::MpiTransport` is declared in `include/fuse/transport/transport.hpp`. The same graph with
`RunnerConfig{}` (transport `"copy"`, no communicator) runs every component in one process, which
is how the serial tests exercise a placed graph.

## Diagnostics

| Code | Raised by | Meaning |
|---|---|---|
| `E-RANKSET-RANGE` | `RankSet(first, count)` | `count` above 2^24, or `first + count` overflows `int` |
| `E-PLACEMENT-EMPTY` | `validate()` | a channel's source or target component, or the coupler its `remap_on` names, is placed on no rank |
| `E-COUPLER-FIELDS` | `validate()` | a coupler declares fields |
| `E-CHANNEL-REMAP-ON` | `validate()` | `remap_on` names a component that is neither a coupler nor the channel's own source or target |
| `W-CHANNEL-REMAP-ON-UNUSED` | `validate()` | `remap_on` is set but no Remap would ever run for the channel |
| `E-TRANSPORT-RANKS` | `MpiTransport` | a transport plan over an empty or inconsistent rank set; normally pre-empted by `E-PLACEMENT-EMPTY` |

## See also

- [](layouts.md) and [](layout-driver.md) for running the placement test driver
- [](geometry.md) for per-placement meshes and remap halos
- [](transfer-maps.md) for how weights are built and applied
- [](multiple-executables.md) for launching components as separate programs
- [](runtime.md) for the rest of `RunnerConfig`
