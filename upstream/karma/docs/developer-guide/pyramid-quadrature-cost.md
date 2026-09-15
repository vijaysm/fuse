# Why pyramid elements cost more, and by how much

Prompted by the split test timings: `karma_fe_variants_prism_pyramid` at 72s
serial and `karma_fe_face_parent_variants_prism_pyramid` at 132s, against
much smaller numbers for the tet/hex and line/surface groups.

## The mechanism

`KARMAQuadrature` picks points per axis differently by topology
(`include/numerics/fe/karma_quadrature.h`):

- edge, quad, hex: `points_per_axis = (degree + 2) / 2`
- prism, pyramid (`create_product_quadrature`): `points_per_axis = (degree + 4) / 2`

For the pyramid that count is cubed, and the extra point is not free:

| requested degree | hex points | pyramid points | ratio |
|---|---|---|---|
| 2 | 8 | 27 | 3.4x |
| 4 | 27 | 64 | 2.4x |
| 6 | 64 | 125 | 2.0x |

The FE tests request degree 4, so every pyramid element in them integrates at
**64 points where a hex uses 27**. FE reinit work is essentially linear in the
point count, and the quadratic pyramid variants carry up to 14 basis
functions, so this is a plausible first-order explanation for the group being
the slowest.

## Why the extra point exists

It is not an accident, and the pyramid rule is not inaccurate. The pyramid
uses a collapsed (Duffy) map: `x = (2u-1)h`, `y = (2v-1)h`, `z`, with
`h = 1-z`, and the map's Jacobian contributes `h^2`, which the code carries
*numerically* in the weight:

```
weights[q] = 4.0 * height * height * line_weights[i] * line_weights[j] * line_weights[k];
```

Integrating a degree-`d` polynomial therefore presents Gauss-Legendre in `z`
with an integrand of degree `d + 2`, and `+4` is exactly `+2` (the usual
Gauss-Legendre sizing) plus `+2` for that Jacobian. Verified: both the current
rule and the alternative below are exact to machine precision (max error
~2e-16, normalized by the pyramid volume) on every monomial up to the
requested degree. The current rule is **wasteful, not wrong**.

## The standard remedy

Absorb `(1-z)^2` into the weight *function* rather than the integrand, using
Gauss-Jacobi with `alpha = 2, beta = 0` in the collapsed direction. The
Jacobian is then integrated exactly by construction, and `z` needs only
`(degree + 2) / 2` points - the same as every other axis.

Measured on all monomials of total degree <= d, error normalized by the
pyramid volume:

| d | current (Gauss-Legendre) | error | Gauss-Jacobi(2,0) in z | error | fewer points |
|---|---|---|---|---|---|
| 2 | 27 | 3.3e-16 | 8 | 1.7e-16 | 3.38x |
| 4 | 64 | 2.9e-16 | 27 | 1.7e-16 | 2.37x |
| 6 | 125 | 1.2e-16 | 64 | 2.1e-16 | 1.95x |

The reduced rule also retains a degree of headroom: sized for degree `d` it is
still exact at `d+1`. With it, a pyramid costs exactly what a hex costs.

## The prism is a separate, simpler finding

The prism shares `create_product_quadrature` and therefore the `+4`, but a
prism is a genuine triangle-by-line tensor product - its weights are
`triangle_weights[i] * line_weights[k]` with **no Jacobian factor**. Nothing
on that path needs the extra point. Confirmed on the line axis: `(d+2)/2`
points integrate degree `d` to 3e-17.

At degree 4 that is 24 points where 18 suffice, a flat 1.33x overhead with no
justification at all. This one is unambiguous.

## Expected effect

Per (order, continuity) combination, summed over the six variants in the
prism/pyramid group: 264 quadrature points today against 135 with both
changes, **1.96x less quadrature work**.

The production consequence is larger than the test one: every pyramid element
in every assembly currently integrates at 2.4x the necessary point count.

## Status

Both changes are implemented (`create_product_quadrature`), and
`karma_fe_quadrature_exactness` pins the property: it integrates the monomial
basis against the closed form at degrees 1-6 and also asserts the reduced
point count, since a regression to the old sizing would still be exact and so
would pass every other assertion in the suite.

## Gradient terms: the open question, now answered

The monomial argument above says nothing about gradients. Under the collapsed
map the chain rule gives `d/dx = (1/h) d/dxi` and
`d/dz = d/dzeta + (xi/h) d/dxi + (eta/h) d/deta`, so basis gradients carry
`1/h` factors and the stiffness integrand carries `1/h^2`. The worry was that
the point removed from each axis had been margin for exactly those.

It was not, and the reason is structural rather than lucky. The `h^2` in the
volume element cancels the `1/h^2` from the chain rule exactly, so what the
rule actually integrates is a polynomial in the collapsed coordinates - and a
lower-degree one than the mass term. **Gradient terms are easier for this rule
than the monomials it was sized on, not harder.**

Measured directly on the standard 5-node rational pyramid basis, against a
converged 40^3 reference, relative error in the full 5x5 stiffness matrix and
in `int grad N_i dV`:

| requested degree | old rule points | error | new rule points | error |
|---|---|---|---|---|
| 2 | 27 | 3.3e-16 | 8 | 1.7e-16 |
| 4 | 64 | 2.5e-16 | 27 | 2.1e-16 |
| 6 | 125 | 2.1e-16 | 64 | 1.7e-16 |
| 8 | 216 | 5.0e-16 | 125 | 2.1e-16 |

Both rules are at rounding at every degree, including the reduced rule at 8
points. The two also agree with each other to 6.7e-16 at 40^3, and the
identity `sum_j K_ij = 0` (which follows from `sum_j N_j = 1`) holds to
2.2e-16.

An attempt to generalise this offline to higher-order bases, by substituting a
generic degree-`p` polynomial in the collapsed coordinates, was **discarded as
invalid**: a generic such polynomial has a non-vanishing `d/dxi` at the apex
and therefore an unbounded gradient there, which no quadrature converges on
and which real pyramid bases are specifically constructed to avoid. Neither
rule converged for it - a property of the synthetic model, not of either
quadrature. Higher-order bases have to be checked against the code's own, not
a stand-in.

`karma_fe_pyramid_gradients` does that, in the suite, for Pyramid5, Pyramid13
and Pyramid14. It uses the divergence theorem,

    int_Omega grad N_i dV = int_dOmega N_i n dS,

which is an identity of calculus and so needs no closed form for the rational
basis. The two sides come from different machinery: the left from the volume
rule against the rational gradients where the `1/h` factors live, the right
from the face rules against basis *traces*, which pyramid bases keep
polynomial by construction so they conform to neighbouring tetrahedra and
hexahedra. The right side is therefore effectively exact and the gap measures
the volume rule alone. Prism6 is a control - no collapsed map, no `1/h` - so a
bad pyramid column can be told apart from a common regression. The test sweeps
quadrature degree and prints the gap, so the host run yields the table rather
than a bare pass.

## Confirmed on the host, 2026-09-10

`karma_fe_pyramid_gradients` passes for Pyramid5, Pyramid13 and Pyramid14 at
quadrature degrees 2 through 8, serially and at two ranks, with Prism6 as the
control - inside a clean 422/422 run. That closes the part the offline work
could not reach: the reduced rule resolves gradient terms for the code's own
higher-order rational bases, not merely for the 5-node basis reproduced by
hand. The assertion threshold is 1e-10, so what is established is that the
gap is below that; `ctest -V -R karma_fe_pyramid_gradients` prints the actual
magnitudes.

## What is still not established

Distorted pyramids - now the only open item. Everything above is on the
reference cell. A distorted
element makes the geometric Jacobian non-constant, and the integrand genuinely
rational in a way no rule integrates exactly - for the old sizing as much as
the new. The offline check covered a distorted cell only through the discarded
synthetic basis, so it carries no weight. The FE and MMS suites are what settle
it; if a pyramid or prism convergence rate or tolerance moves, revert the
commit that resized `create_product_quadrature` first.

The monomial evidence was produced two independent ways, neither needing
`PetscInitialize`: in NumPy from first principles, and in C against PETSc's own
`PetscDTGaussQuadrature` and `PetscDTGaussJacobiQuadrature` replicating the new
code path exactly (max error 1.7e-16 through degree 5, 1.7e-15 at degree 6).
PETSc's Gauss-Jacobi weight convention was measured rather than assumed: for
`alpha = 2` on [0,1] the weights sum to 4/3, confirming they already carry the
`4h^2` the old code applied by hand.
