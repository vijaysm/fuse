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

**Verified end to end.** A channel between a nodal field
(`SemanticLocation::Vertex`) and an FV field (`SemanticLocation::Element`)
lowers, schedules and runs through a real, two-component `Runner` graph:
`InterfaceTransfer.ConsistentPairRunsThroughTheRunnerExactlyAndBalancesPower`
(`test/verification/interface_transfer_test.cpp`) builds exactly this pair
-- a nodal beam velocity through `H` to an FV fluid wall, and the fluid's
per-cell force back through `H^T D` to a nodal beam load -- and checks `H`
exact for a linear field, the delivered force equal to the sent force to
round-off, and discrete interface power balanced to round-off. An earlier
draft of this page reported a `lower()`-time refusal here; it did not
reproduce, and the paragraph was wrong on this point.

What made that end-to-end run possible is a separate fix, not to this
operator: until now, every field's buffer was sized by its region's CELL
count regardless of discretization, so a genuinely nodal field's `n + 1`
values never fit the `n`-cell buffer `Runner::initialize` expected. See
[](../dev/nodal-fields.md) for where a field's extent is now decided from
`(disc, order, region)`, and for parallel nodal fields (lowest-rank vertex
ownership, and what remains open for a distributed `H^T` interface).

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

## Nodal fields through the Runner

A field declared `Discretization::ContinuousGLL, order = 1` is genuinely
nodal: `n` line elements carry `n + 1` values, not `n`. Every buffer-sizing
and conservation-measure call site in `include/fuse/backend/**`,
`src/backend/**` and `src/runtime/runner.cpp` now reads that from one place
-- `field_extent`/`region_extent_kind` in `mesh_view.hpp` -- rather than
assuming a field's extent equals its region's cell count:

```cpp
const auto vel = g.add_field(beam, fuse::FieldSpec{
    .name = "beam_velocity", .region = r_beam,
    .disc = fuse::Discretization::ContinuousGLL, .order = 1,
    .conservation = fuse::Conservation::Intensive,
    .intent = fuse::Intent::Provides});
```

registers a buffer of `n_beam + 1` values, checked against whatever the
component actually binds at `Runner::initialize` (`E-RT-SIZE` on a
mismatch), and its conservation integral (when `.conservation =
fuse::Conservation::Extensive`, `.conservative()` on the channel) is the
plain sum of its owned values, not a measure-weighted one -- a vertex has
no measure of its own. A discretization this codebase does not implement an
extent for (`ContinuousGLL` order != 1, `DiscontinuousGLL` order >= 1,
`PointCloud`) refuses by name, `E-EXTENT-DISC`, at the same point.

In parallel, a nodal field's distributed buffer follows the native mesh's
own rule for a shared vertex: the LOWEST rank holding it owns it, and
`Transport` moves owned values and fills every other copy from its owner.
An `H^T` channel whose nodal target is itself distributed across more than
one rank needs a further cross-rank accumulation this track specified but
did not wire into `Runner` -- see [](../dev/nodal-fields.md)'s "H^T on a
distributed interface" for exactly where it goes and why it is safe, and
this task's `open_questions` for the scope decision it is waiting on.

## Beyond 1D: any supported cell (the "nodal remap" track)

The description above is `H`'s original 1D-curve algorithm (node ordering by
a single coordinate), which `build_consistent_interp_weights` still uses,
unchanged, whenever the source's topological dimension is 1. When the source
is 2D or 3D — triangles, quadrilaterals (including a curved surface mesh
embedded in R^3, e.g. a cubed sphere), tetrahedra, hexahedra, or triangular
prisms — the SAME public entry point dispatches instead to
`build_general_interp_weights` (`include/fuse/backend/general_interp.hpp`):
for each target evaluation point, it locates the containing source cell with
`fuse::search`, inverts that cell's reference map (an exact affine solve for
a simplex; a damped Gauss-Newton iteration, with a stated iteration cap and
residual test, for a genuinely multilinear quad/hex/prism), and reads off
the standard P1/Q1 basis weights at the corners `MeshView::cell_vertices`
reports. It is exact for the shape's own affine/multilinear field, and every
property `H`'s row-sum-one identity buys above — the transpose's exact load
conservation, the discrete power balance — carries over unchanged, because
it rests on nothing more shape-specific than that identity.

A source cell's shape must have a STANDARD nodal basis: a pyramid's is
rational rather than polynomial and is out of scope (refused by name,
`E-INTERP-SHAPE`), and a general polygon or polyhedron has no standard nodal
basis at all (refused the same way). A surface target point is accepted only
within a stated distance of the source's own (possibly non-planar) cell
faceting — see `general_interp.hpp`'s `kSurfaceTol` comment for why that
tolerance is a genuine, physically-motivated one (the faceting gap between
two independent discretizations of the same curved surface) rather than a
rounding-scale check. `InterfaceDomainPolicy::ClampToBoundary` is supported
here exactly as it is for the 1D algorithm.

**Distributed.** `include/fuse/backend/nodal_remap_distributed.hpp` gives
the same general interpolation across ranks: each evaluation entity's row is
built only by the rank that owns it, against an assembled replica of the
source mesh (see that header for the explicit trade this makes and why the
result is bit-identical across rank counts). A channel using method
`"consistent"`/`"consistent-transpose"` now reaches this automatically
through the Runner for a nodal target on more than one declared rank — no
change needed on the caller's side beyond `Channel::remap_method(...)`,
exactly as for a single-rank nodal target. Any OTHER method into such a
target still refuses `E-RT-REMAP-NODAL-PARALLEL` (per-placement supermesh
weights remain unimplemented). `KNOWN_ISSUES.md` D8 records what is, and is
not yet, verified about this wiring under a genuinely partitioned
multi-process run — read it before relying on this at more than one real
rank in production.

## What this is not

Neither algorithm attempts anything above piecewise-linear (P1) or bilinear/
trilinear (Q1) order; a higher-order nodal basis is a different,
not-yet-written kernel. The 1D algorithm specifically still relies on
coordinate-sort node ordering, correct only for a genuine 1D interface
curve — the general algorithm above is what replaces it for anything with
more than one tangential direction.
