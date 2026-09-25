# Where a component runs

A component declares the ranks it runs on. That declaration is an explicit set,
not a range, because real coupled jobs are laid out in ways a range cannot
describe.

## Declaring a placement

A contiguous block is the common case and keeps the short form:

```cpp
g.add_component({.name = "fluid", .ranks = {0, 32}});   // ranks 0..31
```

Anything else is a list. Order does not matter — it is sorted and
de-duplicated, so you can pass ranks in whatever order your launcher reports
them:

```cpp
// One rank per socket.
g.add_component({.name = "solid", .ranks = fuse::RankSet{{0, 8, 16, 24}}});
```

In a configuration file, both forms parse:

```yaml
components:
  - name: fluid
    ranks: {first: 0, count: 32}
  - name: solid
    ranks: [0, 8, 16, 24]
```

## Shared, identical and disjoint placements

Three cases, and the library treats them differently because the physics does:

| Placements | What a channel between them costs |
|---|---|
| identical | no data move: every value is already on the rank that wants it |
| partially shared | a move for the ranks that differ, a local copy for the ranks in common |
| disjoint | a move |

The distinction is `same_as` versus `overlaps`, and it matters more than it
looks. A condition that suppressed the move whenever the placements merely
*touched* would deliver nothing at all to the ranks they do not share.

Partially shared placements are delivered by a **peer-wise** plan: a rank that
is a source for some entities and a destination for others is described peer by
peer rather than by one list of ids, and the matches whose two ends land on the
same rank are copied locally instead of travelling. Identical and disjoint
placements use the matched plan, which is the longer-verified path; on a pattern
both can express, the two are required to deliver identical values.

## An empty placement is legal

A component placed on no ranks in a particular run is not an error. The graph
is identical on every rank, so a component that another rank hosts still
appears in yours; declaring it with no ranks says "not here, this run".

## What the library does with a placement

- **Deciding whether a move is needed** — identical placements, no move.
- **Forming the exchange's communicator** — the union of the two placements,
  built collectively over that union only, never over the whole job. Ranks in
  neither placement take no part.
- **Agreeing on failure** — the same union, so a rank-agreement point reduces
  over exactly the ranks of its own exchange. A shared rank counts once.
- **Indexing per-rank data** — by *position* in the placement, not by rank
  number, since a strided set has no arithmetic relationship between the two.

## An empty placement means "switched off", and takes its channels with it

A placement of zero ranks is legal and says *this component does not run here,
this time*. It is the supported way to leave a component out of one run
without editing the rest of the graph.

It is legal **only for a component with no channels**. A channel out of a
component placed on no rank has nothing to read from, and a channel into one
has nowhere to deliver; `validate()` refuses both with `E-PLACEMENT-EMPTY`,
naming the component and the channel. So when you switch a component off,
remove its channels for that run too.

The refusal is a modelling check, not a transport limitation: it fires for
every transport, before any communicator exists. Previously the same mistake
surfaced two layers down as `MpiTransport::build_plan` refusing an empty rank
set — correct, but it told you about a rank set when the error was in the
graph, and only when you were running under MPI.

## Each component has its own decomposition of a shared mesh

Two components that share a mesh do not share its partition. Each holds the
mesh laid out over **its own** ranks, so a rank placed in both holds two
pieces, one per component. This is what makes partially overlapping
placements work: with A on ranks 0–2 and B on ranks 1–3, each side covers the
whole region over its own ranks, and the transport moves what differs and
copies locally what both hold.

Components with identical placements always get identical decompositions,
which is why no data move is needed between them.

Remap weights are built on the placement's own decomposition: the plan
carries the component, every weight builder resolves its meshes through
`placed_view`, and two placements never share a cached matrix. The one
current limit is the MOAB backend, which does not yet build a decomposition
per component (every component shares one), so on MOAB the per-placement
weights coincide with the default ones.
