# Consistent interpolation and conservative load transfer on 1D interfaces

A strongly-coupled interface between two different solvers -- a 2D Stokes
channel and a 1D Euler-Bernoulli beam forming its wall, say -- needs two
different things from its two directions, on meshes that do not share a
resolution:

* the fluid side needs the beam's velocity **at a point**, as accurately as
  the beam's own representation supports;
* the beam side needs the fluid's traction **totalled into a load**, with
  the total conserved exactly, so no work is created or destroyed at the
  interface.

`fuse/backend/consistent_interp.hpp` provides one operator, `H`, and its
transpose, for exactly this pair. `build_supermesh_weights` (the always-on
P0 path) already does something related and is conservative in the same
integral sense, but is only first-order for a pointwise field: it reproduces
a *constant* exactly, and a genuinely varying field only to first order, even
at fixed resolution. `H` is the second-order, exact-for-linear counterpart
for the pointwise direction, paired with a transpose that is exact-for-force
rather than approximate.

## Where the two matrices live

`H` interpolates piecewise-linearly from a set of source **vertices**
(`MeshView` dimension 0 -- a beam's velocity is naturally nodal) to a set of
target evaluation points, which are either the target's own vertices
(node-to-node) or its cell centroids (e.g. a finite-volume fluid wall's face
midpoints). Both meshes are treated as 1D curves; the embedding axis is
chosen automatically from the spread of the source vertices, so a beam laid
out along `y` or `z` needs no special handling.

`H`'s rows sum to exactly one -- the discrete partition of unity a
piecewise-linear hat-function basis always has -- which is the property
everything else here is built from:

* it makes `H` exact for an affine field, at fixed, genuinely non-matching
  resolution (not just in the limit of refinement);
* it makes `H`'s transpose, applied to a **load** (an already-integrated
  per-entity amount, not a density), deliver a total that equals the total
  sent, to round-off, for *any* input -- `sum_i (H^T f)_i = sum_j f_j`
  follows directly from every row of `H` summing to one;
* combined, it makes the discrete interface power exact: `(H v)^T (D rho) ==
  v^T (H^T D rho)`, where `D` is the point side's own cell measure -- the
  work the fluid does on the interface, evaluated at the fluid's own points,
  equals the work delivered to the beam's nodes, bit for bit.

That last point is why the two directions are not independently-invented
matrices that merely happen to agree on a constant: they are literally one
bilinear form, read in two directions.

## Density versus load

FUSE's extensive-field convention stores a **density**: a value that only
becomes an amount once multiplied by its own cell's measure (see
`remap_plan.hpp`'s "WHICH SUM IS WHICH"). A vertex has no measure of its own
-- `MeshView::measure(0, i)` throws by contract -- so `H`'s transpose cannot
divide a delivered load back into a density on the way out, and does not try
to. Instead, the method that selects the transpose direction,
`"consistent-transpose"`, multiplies the point side's density by *that
side's own* cell measure before contracting with `H^T`, so the matrix it
returns is `H^T D`, not a bare `H^T`. Callers apply it directly to the
density array the channel already carries; the density-to-load conversion
is inside the operator, not a step the caller must remember.

## Selecting it on a channel

```cpp
#include "fuse/backend/consistent_interp.hpp"

// Pointwise direction: beam velocity (nodal) -> fluid wall (FV).
// plan.src_disc = Discretization::ContinuousGLL, plan.src_order <= 1
g.channel(vel_channel).remap_method("consistent");

// Load direction: fluid traction (FV, density) -> beam (nodal).
// plan.tgt_disc = Discretization::ContinuousGLL, plan.tgt_order <= 1
g.channel(force_channel).remap_method("consistent-transpose").conservative();
```

`build_weights()` (`src/backend/intx_weights.cpp`) recognises both method
strings ahead of its usual FV-only check, in both the MOAB and non-MOAB
translation units, and dispatches to `build_consistent_weights_for_plan`,
which resolves the two regions through the graph exactly as the supermesh
path does and refuses `E-INTERP-DISC` if the side that must be nodal is not
`Discretization::ContinuousGLL` at order `<= 1`.

**Known gap, reported rather than worked around:** a channel between a
nodal field (`SemanticLocation::Vertex`) and an FV field
(`SemanticLocation::Element`) is refused during `lower()` today, by
`include/fuse/model/connect.hpp`'s `check_channel`, which unconditionally
declares every channel's transfer as `TransferKind::Copy` regardless of
`remap_method`. Fixing that crosses this track's file ownership (it lives in
`connect.hpp` and is called from `src/ir/rules_channel.cpp`); see the task
report's `open_questions` for the two-sided fix this needs. Until it lands,
`build_weights()`/`build_consistent_weights_for_plan` can be exercised
directly against a `RemapPlan` (as `test/verification/interface_transfer_
test.cpp`'s `ConsistentInterpPlan` cases do), but not yet through a
validated two-component `Runner` graph with genuinely nodal fields.

## Domain edges

The two interface meshes are not required to span the same physical extent.
By default, a target evaluation point outside the source vertices' span
refuses with `E-INTERP-DOMAIN`, naming the point and the span -- extrapolating
a linear reconstruction past the data that supports it can be arbitrarily
wrong for anything but a genuinely affine field, and there is no way for the
builder to know in advance whether the field it will later be applied to is
affine near the boundary. Passing `InterfaceDomainPolicy::ClampToBoundary`
instead extrapolates with a constant (the nearest boundary node's value),
which keeps every conservation and power-balance property above intact --
they only depend on every row summing to one, which a clamped row still
does -- at the cost of reverting to zeroth-order accuracy near the
uncovered edge. `ClampToBoundary` is not yet reachable through the channel
API; see `open_questions`.

## What this is not

This is a 1D-interface operator: node ordering comes from sorting by a
single coordinate, which is correct for a genuine 1D interface curve and
wrong for anything with more than one tangential direction (a 2D surface
coupling, say). It also does not attempt anything above piecewise-linear
order; a higher-order nodal basis is a different, not-yet-written kernel.
