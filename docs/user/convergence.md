# Deciding when a coupled iteration has converged

This page is for anyone setting a tolerance on an implicit coupling: it
explains the two residual norms FUSE offers, when to choose each, and how the
choice applies to Newton regions.

An implicit coupling iterates until the residual over the exchanged interface
values is small. What "small" means is a choice, and the default choice most
couplers make is worth understanding before you rely on it.

## The plain norm and what it depends on

The usual measure is the Euclidean norm over the array of interface values:
`sqrt(sum r_i^2)`. It is cheap, and it is what published iteration counts are usually quoted
with, so it remains available (`NormKind::Euclidean`). But
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
count that was obtained with it.

## Newton regions

A region selecting `AccelKind::Newton` tests convergence on its own: `atol`,
`rtol` and `stol` in `NewtonConfig` (`include/fuse/model/iteration.hpp`) are
applied to a norm over the region's whole stacked residual vector, which
concatenates every exchanged field. `NewtonConfig::norm` chooses that norm:

```cpp
fuse::NewtonConfig cfg;
cfg.rtol = 1e-8;
cfg.norm = fuse::NewtonNorm::MeasureWeighted;   // default: fuse::NewtonNorm::Plain
g.iterate({/* ... */}).until(/* per-field tolerances */).newton(cfg);
```

| `NewtonConfig::norm` | The `atol`/`rtol` test is applied to |
|---|---|
| `Plain` (default) | the Euclidean 2-norm of the stacked residual `F(x)` |
| `MeasureWeighted` | `D * F(x)`, where `D` is a fixed diagonal scaling built once per solve |

### When to choose which

**`Plain`** has the four dependencies listed at the top of this page. Because
all exchanged fields are concatenated into one vector, a region mixing fields
of different units (a wall temperature in kelvin and a heat flux in W/m^2) is
tested by a single number with no unit. The relative test
(`||F_k|| <= rtol * ||F_0||`) softens this, since a global rescaling of either
field cancels, but a field whose error dominates the concatenated norm can
still mask slower convergence in another.

**`MeasureWeighted`** puts every block on a common footing. Each block of the
stacked vector (one per exchanged field transfer) is scaled by
`D_i = sqrt(w_i / sum_block_w) / s_b`, where `w_i` is the measure (length, area
or volume) of the entity the entry lives on, normalised within its own block,
and `s_b` is the tolerance declared for that block by the region's
`ConvergenceCriterion`. A value of 1 in the scaled space therefore means "at
this block's own tolerance, in its own units", and a block that is simply larger
in magnitude no longer dominates. Choose it when the region mixes fields with
different units or magnitudes, or when you will change resolution or partition.

`stol` (the step-length test) is not scaled under either choice.

### Declaring per-block tolerances

`MeasureWeighted` needs exactly one tolerance per block. A block's tolerance is
taken from the convergence measure that names its channel, its provider's
source field, or its consumer's target field.

:::{warning}
If no measure names a block, or several measures name it with differing
tolerances, `initialize()` refuses with `E-NEWTON-NORM-SCALE`. Fix it by naming
the block (by channel, source field or target field) in exactly one measure, or
in several measures that agree on the tolerance; or select
`NewtonNorm::Plain`.
:::

### Preconditioner

A region that sets `use_block_jacobi_preconditioner` together with
`MeasureWeighted` is handled for you: the preconditioner is composed with
`D^-1` (`(D*J)^-1 = J^-1 * D^-1`). See [](../dev/newton-krylov.md) for the
derivation.

### Window-scoped regions

For a window-scoped region ([](coupling-windows.md)), `MeasureWeighted` repeats
each element's spatial weight uniformly across every sample in its window. The
window agreement rule itself is an unweighted sum of squared per-block samples,
with no weighting by substep, and `MeasureWeighted` does not add any.

### Substrate and ranks

`NewtonConfig::substrate` (`Native` by default, or `Petsc` when built with
PETSc) changes which code evaluates these tests, never the tests: both read the
same `atol`/`rtol`/`stol`/`divergence_tolerance`.

When a Newton region's components span more than one rank, the norm is agreed
across exactly that region's participant ranks: the outer convergence test,
every GMRES inner product that selects the iteration count, and the line
search's step-length test all use the same agreed value. Ranks outside the
region take no part. Rank count changes who contributes to the norm, never
which norm or tolerance applies.

## See also

- [](coupling-windows.md) for convergence across a window
- [](../dev/newton-krylov.md) for the Newton-Krylov implementation
