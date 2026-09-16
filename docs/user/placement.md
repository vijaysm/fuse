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

:::{note}
**Partially shared placements are refused today** with
`E-RT-TRANSPORT-OVERLAP`, at setup rather than during a run. The rank-matching
graph describes an exchange with one list of global ids per rank, which cannot
describe a rank that is both a source and a destination of the same move;
supporting it needs separate send and receive plans. Identical and disjoint
placements both work and are tested at 2–5 ranks.
:::

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
