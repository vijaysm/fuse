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

## Newton's own convergence tests

A region selecting `AccelKind::Newton` does not go through
`ConvergenceCriterion`/`NormKind` at all: `fuse::num::solve()`'s own
`atol`/`rtol`/`stol` (`NewtonConfig`, the model-level mirror of `fuse::num::
NewtonKrylovOptions`) test a norm over the region's whole stacked residual
vector, chosen by `NewtonConfig::norm`:

| `NewtonConfig::norm` | tests |
|---|---|
| `Plain` (default) | the PLAIN Euclidean 2-norm, `DistVector::norm2()` -- unchanged from every region declared before this option existed |
| `MeasureWeighted` | `D * F(x)`, `F` scaled entry by entry before the SAME `atol`/`rtol`/`stol` test runs |

**`Plain`** has the identical four dependencies the table at the top of this
page names, with one further consequence worth naming explicitly: every
field transfer the region measures is concatenated into ONE vector before
Newton's own norm is taken, so a region mixing fields of different units
(dn_heat's own `u_wall`, kelvin, and `q_wall`, W/m^2) is tested by a single
number with no unit of its own, computed the same way `field_norm()`'s
`E-NORM-UNITS` check exists specifically to refuse. `rtol`'s own RELATIVE
form (`||F_k|| <= rtol * ||F_0||`) keeps this from being as unit-dependent
as it looks (both numerator and denominator mix the same two fields in the
same proportion, so a global rescaling of either field cancels), but it does
not make the two fields' own convergence independently legible the way two
separate `ConvergenceCriterion` measures do for a Picard-accelerated
region: a field whose own error happens to dominate the concatenated norm's
magnitude can mask slower convergence in the other.

**`MeasureWeighted`** is the fix: each block of the stacked vector (one per
exchanged field) is scaled by `D_i = sqrt(w_i / sum_block_w) / s_b` before
the stacked norm is taken -- `w_i` the entity measure (length, area or
volume) each entry lives on, normalised within its own block (the SAME
root-mean-square weighting `NormKind::MeasureWeightedRms` above uses,
applied per block rather than to one field in isolation), `s_b` the
tolerance that block's own `ConvergenceCriterion` measure declares (matched
by the channel's name, its provider's source field, or its consumer's
target field -- whichever one a measure actually names). `1` in this scaled
space means "at this block's own declared tolerance", in that block's own
units, regardless of how the OTHER block's own units or magnitude compare:
a block whose own physical magnitude is three orders larger than another's
no longer dominates the combined norm just by being larger, since each
block is normalised against its OWN tolerance before being combined with
anything else. `NewtonConfig::stol` (the step-length test) is left
unscaled either way -- it tests the Newton STEP, `x_{k+1} - x_k`, which has
no declared tolerance of its own to normalise against.

A block this track's own setup cannot derive a tolerance scale for refuses
eagerly, at `initialize()`, with `E-NEWTON-NORM-SCALE`: either no
convergence measure names that block's channel, source field, or target
field at all, or more than one does with DIFFERING tolerances (this
project's own `ConvergenceCriterion` allows a measure to name any of the
three, so two measures naming different ones of them for the same block,
with different tolerances, is a genuine ambiguity this option will not
silently resolve by picking one). Name the block with a single, consistent
tolerance, or select `NewtonConfig::norm: Plain`.

**The preconditioner.** A region additionally requesting
`NewtonConfig::use_block_jacobi_preconditioner` under `MeasureWeighted`
needs its own right preconditioner composed with `D^-1`
(`(D*J)^-1 = J^-1 * D^-1`), not applied bare -- `docs/dev/newton-krylov.md`'s
own new section on `NewtonNorm` has the derivation and the measured effect
of getting this wrong.

**Window-scoped regions.** `decide_window_convergence()`, the agreement
rule a window-scoped region's own Newton solve uses, is an UNWEIGHTED flat
sum of squared per-block samples -- no spatial measure weighting of its
own, and no weighting by which substep within the window a sample belongs
to either. `MeasureWeighted` on a window-scoped region repeats each
element's own spatial measure uniformly across every sample in its own
window, matching that existing lack of time weighting rather than inventing
one `decide_window_convergence()` itself does not use. Verified against
the real Brusselator example (not just a synthetic fixture): every window
converges under both norms, and the production diagonal matches this
formula entry for entry at the example's own real grid; see
`docs/dev/newton-krylov.md`'s own new section for the derivation and the
measured results, including a window-specific weight-derivation defect
this verification found and fixed (every window with more than one
sample per transfer was refused before the fix).

`fuse::num::solve()` remains a general Newton-Krylov solver over an
arbitrary stacked residual either way, not specific to a coupling region's
own named fields -- `MeasureWeighted` is this track's own answer to "scale
the stacked vector so a region's own fields read on a common footing", not
a change to `fuse::num::solve()`'s own algorithm.

`NewtonConfig::substrate` (`Native` by default, `Petsc` selecting PETSc's
own `SNESNEWTONLS` -- `docs/dev/newton-krylov.md` §10/§11) changes WHICH
code evaluates this same test, never the test itself: both substrates read
the identical `atol`/`rtol`/`stol`/`divergence_tolerance` from the SAME
`NewtonConfig`, so a region's own convergence behaviour under this section
is substrate-independent by construction -- the two can differ only in HOW
closely each one's own GMRES/line-search implementation reaches the shared
target (docs/dev/newton-krylov.md §10.3's own measured step-1 difference),
never in what counts as converged.
