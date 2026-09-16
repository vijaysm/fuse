# Several sources writing one field

A field is often fed by more than one source: a wall temperature seen by two
fluid domains, a heat flux with a convective and a radiative contribution, a
background field overridden locally by a finer model. FUSE compiles that, but
only once you say how the contributions combine.

## Why you have to say

Summing two heat fluxes and averaging two temperatures are different physics,
and nothing in the field descriptions distinguishes them. Units, mesh, and
conservation class are identical in both cases. A coupler that picked one for
you would be guessing at the physics, and the guess would be invisible in the
results.

So a field with more than one writer and no rule is refused:

```
error E-MULTI-WRITER: field 'wall_flux' is written by 2 channels, and no rule
says how to combine them. ... Declare one with graph.merge(field,
MergeOp::Sum|Mean|First), or retarget all but one channel.
```

## Declaring the rule

```cpp
g.add_channel("convective", q_conv, wall_flux);
g.add_channel("radiative",  q_rad,  wall_flux);
g.merge(wall_flux, fuse::MergeOp::Sum);
```

In the text form:

```yaml
merges:
  - field: wall_flux
    component: solid
    rule: Sum
```

| Rule | Meaning | When it is right |
|---|---|---|
| `Sum` | add the contributions | an extensive quantity split across sources: fluxes into one surface add |
| `Mean` | unweighted arithmetic mean | an intensive quantity sampled by several sources covering comparable extents |
| `First` | take the first contributing channel in declaration order | an override: a fine-scale source shadowing a background field |

`Mean` is unweighted. If the sources cover very different extents, it is
probably not what you want; weighted merges are not implemented, and the
absence is deliberate — the weights would have to come from coverage
information the model does not yet carry.

Every channel targeting the field participates. The rule does not list them,
because a list is something a later channel can be left out of, and a
contribution silently dropped is the failure this whole mechanism exists to
prevent.

## A merge that combines nothing

```
warning W-MERGE-UNUSED: field 'wall_flux' declares a Sum merge but is written
by 1 channel(s); the merge combines nothing
```

Usually a channel was renamed or retargeted and the rule was left behind. It
is a warning, not an error: the graph still runs and still means what it says.

## What it compiles to

One `Merge` operator consuming every writer, placed where the writers meet and
before the field is delivered. You can see it in the plan:

```
  7  t=1  Merge  node=7  in=[%3, %6]  out=%7  {reason=Sum}
```

The combined value is timed at its most recent contribution: if one channel is
lagged and another is not, the result is available at the unlagged channel's
step, and the dependence on that channel is preserved in the schedule.
