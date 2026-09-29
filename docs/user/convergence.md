# Deciding when a coupled iteration has converged

An implicit coupling iterates until the residual over the exchanged interface
values is small. What "small" means is a choice, and the default choice most
couplers make is worth understanding before you rely on it.

## The plain norm and what it depends on

The usual measure is the Euclidean norm over the array of interface values:
`sqrt(sum r_i^2)`. It is cheap and it is what the inherited iteration-count
regressions in this repository were recorded with, so it remains available. But
it depends on four things that are not physics:

| It changes when | Because |
|---|---|
| you refine the interface | there are more entries to square and add, so the same physical error reads larger |
| cells vary in size | a sliver cell counts as much as a large one, so it can hold up an iteration converged everywhere else |
| you change partitions | a value owned by one rank and mirrored on two others is counted three times |
| fields have different units | a temperature norm and a heat-flux norm are added as though commensurable |

A tolerance tuned on a coarse serial mesh is therefore a *different* tolerance
on a fine partitioned one, and nothing tells you that.

## The measure-weighted norm

`NormKind::MeasureWeightedRms` computes
`sqrt(sum_owned w_i r_i^2 / sum_owned w_i)`, where `w_i` is the measure
(length, area or volume) of the entity the value lives on and the sums run over
owned entities only. That is a root-mean-square error weighted by how much of
the interface each value represents. It has the units of the field, and it is
independent of resolution, of grading, and of the partition — so a tolerance
means the same thing in all three.

```cpp
fuse::num::ResidualField f{
    .residual = r.data(), .measure = cell_area.data(),
    .owned = owned_mask.data(), .count = n, .units = fuse::units::kKelvin};
const auto norm = fuse::num::field_norm(
    fuse::num::NormKind::MeasureWeightedRms, f);
// norm.value() carries .value, .units, .total_measure and .owned_count
```

Two behaviours are deliberate and worth knowing:

**A rank owning no part of the interface returns `owned_count == 0`, not zero.**
Zero would read as "converged" inside a reduction across ranks. `combine()`
excludes such contributions rather than averaging them in.

**Combining norms over fields with different dimensions is an error**
(`E-NORM-UNITS`), not a sum. A number made by adding a temperature to a heat
flux has no meaning, and a tolerance on it cannot be reasoned about. If you
genuinely want one number over unlike fields, scale them to a common unit
yourself — so that the scaling is visible and reviewable.

## Which to use

Use the weighted norm for new work, especially if you will change resolution or
rank count. Use the plain norm when you are reproducing a published iteration
count that was obtained with it. The choice is declared per implicit coupling group, so a setup
records which one its tolerance refers to.
