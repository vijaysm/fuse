# Consistent interpolation and conservative load transfer on 1D interfaces

This page is for coupling two solvers across an interface (fluid and beam,
say) where one side is nodal and the two meshes do not match: how to get
accurate pointwise interpolation one way and exactly conserved loads the other.

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
density array the field transfer already carries; the density-to-load conversion
is inside the operator, not a step the caller must remember.

## Selecting it on a field transfer

```cpp
#include "fuse/backend/consistent_interp.hpp"

// Pointwise direction: beam velocity (nodal) -> fluid wall (FV).
// plan.src_disc = Discretization::ContinuousGLL, plan.src_order <= 1
g.channel(vel_channel).remap_method("consistent");

// Load direction: fluid traction (FV, density) -> beam (nodal).
// plan.tgt_disc = Discretization::ContinuousGLL, plan.tgt_order <= 1
g.channel(force_channel).remap_method("consistent-transpose").conservative();
```

### The complete graph for the pair

The two channels below are the whole declaration: a nodal field is
`Discretization::ContinuousGLL` at `order = 1`; a cell field is
`Discretization::FV` (the default). The velocity is a density-like pointwise
quantity (`Intensive`); the force is an amount (`Extensive`), which is what
makes the lowering emit a conservation check on that channel. Because the two
channels run in opposite directions between the same two components, they
close a cycle inside one coupling period, and one of them must carry a lag
(`E-SCHED-CYCLE` otherwise); inside an implicit coupling group the sweep is
what resolves that lag ([](fsi-channel.md)).

```cpp
fuse::CouplingGraph g;
const auto m_beam  = g.add_mesh(fuse::MeshDecl{.name = "beam_mesh",  .source = "generated:line:7"});
const auto m_fluid = g.add_mesh(fuse::MeshDecl{.name = "fluid_mesh", .source = "generated:line:11"});
const auto r_beam  = g.add_region(fuse::Region{.name = "r_beam",  .mesh = m_beam,  .dimension = 1});
const auto r_fluid = g.add_region(fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});
const auto beam  = g.add_component(fuse::ComponentDecl{
    .name = "beam",  .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
const auto fluid = g.add_component(fuse::ComponentDecl{
    .name = "fluid", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});

constexpr fuse::Units kMetersPerSecond{.length = 1, .time = -1};

// Values one way: nodal velocity -> FV wall, pointwise (H).
const auto vel_out = g.add_field(beam, fuse::FieldSpec{
    .name = "beam_velocity", .region = r_beam,
    .disc = fuse::Discretization::ContinuousGLL, .order = 1,
    .units = kMetersPerSecond, .conservation = fuse::Conservation::Intensive,
    .intent = fuse::Intent::Provides});
const auto vel_in = g.add_field(fluid, fuse::FieldSpec{
    .name = "fluid_velocity_in", .region = r_fluid,
    .disc = fuse::Discretization::FV,
    .units = kMetersPerSecond, .conservation = fuse::Conservation::Intensive,
    .intent = fuse::Intent::Requires});
const auto vch = g.add_channel("wall_velocity", vel_out, vel_in);
g.channel(vch).remap_method("consistent");

// Loads back: FV force density -> nodal load, conservative (H^T D).
const auto force_out = g.add_field(fluid, fuse::FieldSpec{
    .name = "fluid_force", .region = r_fluid,
    .disc = fuse::Discretization::FV,
    .units = fuse::units::kNewton, .conservation = fuse::Conservation::Extensive,
    .intent = fuse::Intent::Provides});
const auto force_in = g.add_field(beam, fuse::FieldSpec{
    .name = "beam_load", .region = r_beam,
    .disc = fuse::Discretization::ContinuousGLL, .order = 1,
    .units = fuse::units::kNewton, .conservation = fuse::Conservation::Extensive,
    .intent = fuse::Intent::Requires});
const auto fch = g.add_channel("wall_force", force_out, force_in);
g.channel(fch).remap_method("consistent-transpose").conservative()
    .lag(1).allow_time_level_mismatch("explicit coupling: the load lags the velocity by one period");
```

What the lowering emits (`dump_text(lowered.dag)`): a `Remap` per channel
carrying the method string, a `Constrain` only after the conservative load
transfer, and the one-period lag on the force (`[WindowEnd-1, k]`). The
pointwise channel gets no `Constrain`: a vertex sum against a cell integral
is not a conservation statement, so none is checked.

```text
node 0   Source                    -> %0 : Instant<r0, m*s^-1, Intensive, [WindowEnd, k], ranks=0>  "beam.beam_velocity"
node 1   Source                    -> %1 : Instant<r1, kg*m*s^-2, Extensive, [WindowEnd, k], ranks=1>  "fluid.fluid_force"
node 2   Remap        %0           -> %2 : Instant<r1, m*s^-1, Intensive, [WindowEnd, k], ranks=0>  "wall_velocity/remap" {method=consistent}
node 3   Diagnose     %2           -> %3 : Instant<r1, m*s^-1, Intensive, [WindowEnd, k], ranks=0>  "wall_velocity/diagnose"
node 4   Remap        %1           -> %4 : Instant<r0, kg*m*s^-2, Extensive, [WindowEnd-1, k], ranks=1>  "wall_force/remap" {remap=C, method=consistent-transpose}
node 5   Constrain    %4           -> %5 : Instant<r0, kg*m*s^-2, Extensive, [WindowEnd-1, k], ranks=1>  "wall_force/constrain"
node 6   Diagnose     %5           -> %6 : Instant<r0, kg*m*s^-2, Extensive, [WindowEnd-1, k], ranks=1>  "wall_force/diagnose"
node 7   Sink         %3           -> %7 : Instant<r1, m*s^-1, Intensive, [WindowEnd, k], ranks=1>  "fluid.fluid_velocity_in"
node 8   Sink         %6           -> %8 : Instant<r0, kg*m*s^-2, Extensive, [WindowEnd, k], ranks=0>  "beam.beam_load"
```

### A compiled check

The graph above, loaded into a `SerialMeshBackend`, lowered, scheduled and
run for two periods with a beam writing the linear velocity `0.3 + 2x` at its
8 nodes and a fluid writing the traction `1 + sin(3x)` at its 11 cell
centroids. The component binds `n + 1` values for a `ContinuousGLL` order-1
field and `n` for an FV field:

```cpp
class Side final : public fuse::Component {
 public:
  Side(std::string name, std::function<void(Side&)> fill)
      : name_(std::move(name)), fill_(std::move(fill)) {}
  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    for (const fuse::FieldId f : g.component(*g.find_component(name_)).fields) {
      const fuse::FieldSpec& fs = g.field(f);
      // A ContinuousGLL order-1 field on n line cells holds n + 1 values.
      const std::size_t cells = g.region(fs.region).name == "r_beam" ? kBeamCells : kFluidCells;
      const std::size_t n = fs.disc == fuse::Discretization::ContinuousGLL ? cells + 1 : cells;
      auto& buf = bufs[fs.name];
      buf.assign(n, 0.0);
      auto r = reg.bind(f, fuse::FieldRef{buf.data(), fuse::Layout{.stride = 1, .count = n},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    fill_(*this);
    return fuse::Result<void>::success();
  }
  std::map<std::string, std::vector<double>> bufs;

 private:
  std::string                name_;
  std::function<void(Side&)> fill_;
};
```

After `runner.run(components, 2)`, the three properties are read straight
off the bound buffers:

```cpp
const double h_f = 1.0 / kFluidCells;
double worst = 0.0, sent = 0.0, got = 0.0, p_fluid = 0.0, p_beam = 0.0;
for (std::size_t j = 0; j < kFluidCells; ++j) {
  const double x = (static_cast<double>(j) + 0.5) / kFluidCells;
  worst = std::max(worst, std::abs(v_in[j] - (0.3 + 2.0 * x)));
  sent += force[j] * h_f;               // density times cell measure
  p_fluid += v_in[j] * force[j] * h_f;
}
for (std::size_t i = 0; i < kBeamCells + 1; ++i) {
  got += load[i];                       // a nodal load is already an amount
  p_beam += v_nodes[i] * load[i];
}
```

Expected output (exact digits can vary by platform):

```text
remap_weight_builds     : 2
remap_weight_nonzeros   : 44
conservation_checks     : 2
conservation_violations : 0
H worst linear error    : 4.44089e-16
force sent / received   : 1.66539 / 1.66539
power fluid / beam      : 2.19292 / 2.19292
```

Two matrices (22 entries each: at most two bracketing nodes per fluid cell),
`H` exact for the linear field, the total load conserved to round-off, and
the discrete interface power equal on both sides. The conservation check the
Runner runs on the force channel integrates the FV density against its cell
measures on the way in and sums the nodal loads on the way out, which is the
same identity.

The strongly coupled version of this pair, with the two channels inside an
implicit coupling group and an accelerator, is `examples/fsi_channel`
([](fsi-channel.md)); its channel declarations are:

```{literalinclude} ../../examples/fsi_channel/main.cpp
:language: cpp
:lines: 139-168
```

`build_weights()` (`src/backend/intx_weights.cpp`) recognises both method
strings ahead of its usual FV-only check, in both the MOAB and non-MOAB
translation units, and dispatches to `build_consistent_weights_for_plan`,
which resolves the two mesh entity sets through the graph exactly as the supermesh
path does and rejects with `E-INTERP-DISC` if the side that must be nodal is not
`Discretization::ContinuousGLL` at order `<= 1`.

**End to end.** A field transfer between a nodal field
(`SemanticLocation::Vertex`) and an FV field (`SemanticLocation::Element`)
compiles into the dataflow graph, schedules and runs through a real, two-component `Runner` graph:
`InterfaceTransfer.ConsistentPairRunsThroughTheRunnerExactlyAndBalancesPower`
(`test/verification/interface_transfer_test.cpp`) builds exactly this pair
-- a nodal beam velocity through `H` to an FV fluid wall, and the fluid's
per-cell force back through `H^T D` to a nodal beam load -- and checks `H`
exact for a linear field, the delivered force equal to the sent force to
round-off, and discrete interface power balanced to round-off.

A nodal field's buffer holds `n + 1` values, not the `n` of its mesh entity
set's cells; the extent is decided from `(disc, order, region)` (see
[](../dev/nodal-fields.md), which also covers parallel nodal fields and
lowest-rank vertex ownership).

## Domain edges

The two interface meshes are not required to span the same physical extent.
By default, a target evaluation point outside the source vertices' span
rejects with `E-INTERP-DOMAIN`, naming the point and the span -- extrapolating
a linear reconstruction past the data that supports it can be arbitrarily
wrong for anything but a genuinely affine field, and there is no way for the
builder to know in advance whether the field it will later be applied to is
affine near the boundary. Passing `InterfaceDomainPolicy::ClampToBoundary`
instead extrapolates with a constant (the nearest boundary node's value),
which keeps every conservation and power-balance property above intact --
they only depend on every row summing to one, which a clamped row still
does -- at the cost of reverting to zeroth-order accuracy near the
uncovered edge. `ClampToBoundary` is not selectable through the field-transfer
API; it is available when calling the weight builders directly.

## Nodal fields through the Runner

A field declared `Discretization::ContinuousGLL, order = 1` is genuinely
nodal: `n` line elements carry `n + 1` values, not `n`. Every buffer-sizing
and conservation-measure call site in `include/fuse/backend/**`,
`src/backend/**` and `src/runtime/runner.cpp` reads that from one place
-- `field_extent`/`region_extent_kind` in `mesh_view.hpp` -- rather than
assuming a field's extent equals its mesh entity set's cell count:

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
fuse::Conservation::Extensive`, `.conservative()` on the field transfer) is the
plain sum of its owned values, not a measure-weighted one -- a vertex has
no measure of its own. A discretization this codebase does not implement an
extent for (`ContinuousGLL` order != 1, `DiscontinuousGLL` order >= 1,
`PointCloud`) rejects by name, `E-EXTENT-DISC`, at the same point.

In parallel, a nodal field's distributed buffer follows the native mesh's
own rule for a shared vertex: the LOWEST rank holding it owns it, and
`Transport` moves owned values and fills every other copy from its owner.
An `H`/`H^T` field transfer whose nodal SIDE (source or target) is itself
distributed across more than one rank is wired into `Runner` for methods
`"consistent"`/`"consistent-transpose"` specifically -- see "Beyond 1D"
below and [](../dev/nodal-fields.md)'s "H^T on a distributed interface"
for how it works.

## Beyond 1D: any supported cell

The description above is `H`'s 1D-curve algorithm (node ordering by
a single coordinate), which `build_consistent_interp_weights` uses
whenever the source's topological dimension is 1. When the source
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
rational rather than polynomial and is out of scope (rejected by name,
`E-INTERP-SHAPE`), and a general polygon or polyhedron has no standard nodal
basis at all (rejected the same way). A surface target point is accepted only
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
result is bit-identical across rank counts). A field transfer using method
`"consistent"` (H, interpolating FROM a nodal source) or
`"consistent-transpose"` (H^T D, delivering a load TO a nodal target) reaches this automatically through the Runner whenever the NODAL side of
that orientation, or its evaluation partner, is on more than one declared
rank — no change needed on the caller's side beyond
`Channel::remap_method(...)`, exactly as for a single-rank nodal side.
A method paired with the WRONG orientation for its nodal side, any OTHER
method (supermesh weights built from a component's own partition of the mesh are
not supported), or a
field transfer where BOTH sides are nodal, all reject
`E-RT-REMAP-NODAL-PARALLEL` by name. `ClampToBoundary` is not supported on
the distributed path.

## Calling the builders directly

The Runner reaches these operators through `build_weights()`, but both are
ordinary functions in `fuse/backend/consistent_interp.hpp` (header-only) and
take `MeshView`s, so a driver can build `H` without a graph:

```cpp
// H: rows = target evaluation entities at `tgt_dim` (0 for node-to-node,
// the cell dimension for centroids), columns = source vertices.
Result<RemapWeights> build_consistent_interp_weights(
    const MeshView& src_nodes, const MeshView& tgt_points, int tgt_dim,
    InterfaceDomainPolicy policy = InterfaceDomainPolicy::Refuse);

// The build_weights() entry point for "consistent" / "consistent-transpose":
// returns H or H^T D according to plan.method and the two (disc, order) pairs.
Result<RemapWeights> build_consistent_weights_for_plan(
    const RemapPlan& plan, const CouplingGraph& graph, const MeshBackend& backend);
```

`build_consistent_interp_weights` dispatches on `src_nodes.dimension()`: 1
takes the coordinate-sorted curve algorithm (`build_consistent_interp_weights_1d`),
2 or 3 take `build_general_interp_weights` (`fuse/backend/general_interp.hpp`).
The result is a `RemapWeights` with `order = 2`, `monotone` true (every row is
a convex combination of at most two nodes), and
`col_defect_is_measure_weighted = false`, so `WeightsOperator::claims()` never
reports it as conservative in the measure-weighted sense.

## Errors

| Code | Raised when | Fix |
|---|---|---|
| `E-INTERP-DISC` | `"consistent"` with a source, or `"consistent-transpose"` with a target, that is not `ContinuousGLL` order ≤ 1 | declare the nodal side as `ContinuousGLL`, `order = 1`, or swap the method to match the orientation |
| `E-INTERP-REGION` | a region id is invalid, names no mesh, or names an `entity_set` other than empty or `"all"` | interpolate over whole meshes |
| `E-INTERP-NOGEOMETRY` | the source has no vertices, or the target has no entities at the evaluation dimension | check the mesh and the region dimension |
| `E-INTERP-DEGENERATE` | two source vertices coincide along the interface axis | remove the duplicate node |
| `E-INTERP-DOMAIN` | a target point lies outside the source span (policy `Refuse`) | extend the source, or pass `InterfaceDomainPolicy::ClampToBoundary` to the direct builder |
| `E-INTERP-SHAPE` | a 2D/3D source cell has no standard nodal basis (pyramid, general polygon or polyhedron) | use tri/quad/tet/hex/prism source cells |
| `E-INTERP-METHOD` | `build_consistent_weights_for_plan` was handed another method string | one of the two names |
| `E-MESH-NO-CELL-VERTICES` | a 2D/3D source whose view does not implement `MeshView::cell_vertices` (the base-class default), or a cell type the backend cannot express as a `CellShape` | the serial, native and MOAB backends all implement it for the standard shapes |
| `E-EXTENT-DISC` | a field whose `(disc, order)` has no extent (`ContinuousGLL` order ≠ 1, `DiscontinuousGLL` order ≥ 1, `PointCloud`) | the two supported shapes |
| `E-RT-SIZE` | the component bound a buffer of the wrong length for a nodal field | bind `n + 1` values for `n` line cells |
| `E-RT-REMAP-NODAL-PARALLEL` | a nodal side on more than one rank through any method but the matching orientation of `"consistent"` / `"consistent-transpose"`, or both sides nodal | see "Distributed" above |

## What this is not

Neither algorithm attempts anything above piecewise-linear (P1) or bilinear/
trilinear (Q1) order; a higher-order nodal basis is not supported. The 1D algorithm
relies on
coordinate-sort node ordering, correct only for a genuine 1D interface
curve — the general algorithm above is what replaces it for anything with
more than one tangential direction.

## See also

- [](geometry.md) for the supermesh geometry, the method strings and the remap error codes
- [](transfer-maps.md) for the conservative first-order maps and how to persist weights
- [](fsi-channel.md) for the strongly coupled example built on this pair
- [](point-cloud-remap.md) for the other pointwise method
- [](../dev/nodal-fields.md) for nodal field extents and ownership
