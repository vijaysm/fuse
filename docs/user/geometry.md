# Geometry and the remap ladder

This page is for anyone coupling two components whose fields live on
different meshes: how to declare the two meshes and a conservative channel,
how the remap kind is selected, what the run reports, and what the
conservative map is built from and checked against. Mesh backends and the
generated-mesh grammar are covered in [](meshes.md).

FUSE represents geometry natively, and what it can do with it depends on which
optional libraries the build found. The floor is deliberately useful on its
own: **arbitrary unstructured geometry with lowest-order conservative
projection, in the default build, with no optional dependency at all.**

| Build | Geometry | Conservative projection |
|---|---|---|
| default | FUSE's native simplicial mesh | lowest order, arbitrary geometry |
| `FUSE_ENABLE_MOAB=ON` | MOAB's mesh, used natively — no copy | as above, plus MOAB's search and partitioning |
| MOAB built with TempestRemap | as above | plus Earth-system remaps on the sphere, high order |

The lowest-order path is not a fallback that merely compiles. Because it is
always present, it is also the **differential oracle** for the other two: on a
problem all three can express, they are required to agree.

## Remap a field between two meshes

A remap is never requested directly. You declare two meshes, a region on
each, a field on each region, and a channel between the fields; the lowering
emits a `Remap` operator whenever the two regions differ or a remap flag is
set, and the Runner builds the weights at `initialize()`.

```cpp
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>

fuse::CouplingGraph g;
const auto m_src = g.add_mesh(fuse::MeshDecl{.name = "fine",   .source = "generated:line:8"});
const auto m_dst = g.add_mesh(fuse::MeshDecl{.name = "coarse", .source = "generated:line:5"});
const auto r_src = g.add_region(fuse::Region{.name = "fine_cells",   .mesh = m_src, .dimension = 1});
const auto r_dst = g.add_region(fuse::Region{.name = "coarse_cells", .mesh = m_dst, .dimension = 1});

const auto a = g.add_component(fuse::ComponentDecl{
    .name = "source", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
const auto b = g.add_component(fuse::ComponentDecl{
    .name = "target", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});

// One value per cell (FV is the default discretisation); the value is a density.
const auto f_out = g.add_field(a, fuse::FieldSpec{
    .name = "rho", .region = r_src, .disc = fuse::Discretization::FV,
    .units = fuse::units::kKelvin, .conservation = fuse::Conservation::Intensive,
    .intent = fuse::Intent::Provides});
const auto f_in = g.add_field(b, fuse::FieldSpec{
    .name = "rho_in", .region = r_dst, .disc = fuse::Discretization::FV,
    .units = fuse::units::kKelvin, .conservation = fuse::Conservation::Intensive,
    .intent = fuse::Intent::Requires});

const auto ch = g.add_channel("rho_link", f_out, f_in);
g.channel(ch).conservative();   // RemapFlags::Conservative; method left empty
```

Three things decide which map is built:

- the **flags**: `ChannelBuilder::remap(RemapFlags)` ORs bits into the
  channel; `conservative()` is `remap(RemapFlags::Conservative)`. The bits are
  `Conservative`, `Monotone` and `HighOrder`;
- the **discretisation pair** `(disc, order)` of the two fields, read from
  their `FieldSpec`s (`FV`, `ContinuousGLL`, `DiscontinuousGLL`, `PointCloud`);
- the **method string**, `ChannelBuilder::remap_method(std::string)`. Empty
  (the default) means "derive from the discretisation pair and the
  conservation kind".

`Conservation` on the two fields also chooses the normalisation of the
first-order supermesh map: `Intensive` (the value is a density) gives
`W_ji = |T_j ∩ S_i| / m_j` with every target row summing to one;
`Extensive` (the value itself is the amount) gives `W_ji = |T_j ∩ S_i| / n_i`
with every source column summing to one. `None` behaves as `Intensive`.
The two fields of a channel must agree (`validate()` reports `E-CONSERV`
otherwise).

### Method override strings

The strings `build_weights()` (`fuse/backend/remap_plan.hpp`) recognises:

| `remap_method(...)` | Builder | Needs |
|---|---|---|
| `""` (default) | supermesh at order 1; TempestRemap when the fields ask for an order above 1 or `HighOrder` is set | nothing; MOAB with TempestRemap for the high-order case |
| `"intx"`, `"conservative"` | the first-order conservative supermesh builder, by name | nothing |
| `"tempest"` | the high-order TempestRemap builder (orders 1 to 4) | `FUSE_ENABLE_MOAB=ON`, MOAB built with TempestRemap, a `MoabMeshBackend` |
| `"consistent"` | piecewise-linear interpolation `H` **from** a nodal source ([](interface-transfer.md)) | nothing |
| `"consistent-transpose"` | the load transfer `H^T D` **to** a nodal target ([](interface-transfer.md)) | nothing |
| `"mls:degree=<d>,k=<k>,radial=<wendlandN>"` | moving least squares ([](point-cloud-remap.md)) | nothing |

`"consistent"`, `"consistent-transpose"` and `"mls:..."` are dispatched on the
string alone, before any discretisation check, in every build. Everything
else must be an FV-to-FV pair (`E-REMAP-UNSUPPORTED` otherwise). A string not
in this table is refused on every backend: `validate()` reports it as
`E-CHANNEL-REMAP-METHOD`, naming the known methods, and `build_weights()`
refuses it again (`E-REMAP-UNSUPPORTED`) for a plan that never went through
`validate()`. A misspelt method is never built as the default remap.

### Where the remap runs

By default a channel's `Remap` runs on the target component's ranks. To run
it on a separately placed coupler component instead:

```cpp
const auto coupler = g.add_coupler({.name = "C", .ranks = {2, 1}});
g.channel(ch).conservative().remap_on(coupler);
```

`remap_on` must name a `ComponentKind::Coupler` component or the channel's
own source or target component; anything else is `E-CHANNEL-REMAP-ON` from
`validate()`. A `remap_on` on a channel that can never emit a `Remap` (no
flags, same region on both ends, no placement-moving method) is reported as
`W-CHANNEL-REMAP-ON-UNUSED`. See [](placement.md).

### Run it and read the counters

The rest is the ordinary load, lower, schedule, run sequence. `RunStats`
(`fuse/runtime/runner.hpp`) reports what the remap did:

- `remap_weight_builds`: how many weight matrices were built at
  `initialize()`. Two provably coincident regions take the identity path and
  build none; identical plans share one matrix through `RemapCache`.
- `remap_weight_nonzeros`: the total number of entries in those matrices.
- `conservation_checks` / `conservation_violations` /
  `worst_conservation_error`: the `Constrain` operator the lowering emits after
  a `Remap` whose fields declare a `Conservation`. It totals the field over each
  region before and after the remap (the plain sum of owned values for an
  `Extensive` field, the measure-weighted integral for an `Intensive` one) and
  compares the two (relative tolerance `RunnerConfig::conservation_tolerance`,
  default 1e-12; `abort_on_violation` turns a violation into `E-RT-CONSERVATION`).

The complete program below (serial build, no optional library) runs the graph
above for three periods with a source that writes a ramp each step:

```cpp
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>

#include <Kokkos_Core.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

// A component that owns one double per cell and binds it as one field.
class CellField : public fuse::Component {
 public:
  CellField(std::string name, std::size_t cells, bool producer)
      : name_(std::move(name)), data_(cells, 0.0), producer_(producer) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1, .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    if (producer_) {
      ++steps_;
      for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] = 1.0 + 0.25 * static_cast<double>(i) + static_cast<double>(steps_);
      }
    }
    return fuse::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::string         name_;
  std::vector<double> data_;
  bool                producer_;
  int                 steps_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  constexpr std::size_t kSrcCells = 8;
  constexpr std::size_t kDstCells = 5;

  // ---- 1. Two meshes, two regions, two components, one conservative channel.
  fuse::CouplingGraph g;
  const auto m_src = g.add_mesh(fuse::MeshDecl{.name = "fine", .source = "generated:line:8"});
  const auto m_dst = g.add_mesh(fuse::MeshDecl{.name = "coarse", .source = "generated:line:5"});
  const auto r_src = g.add_region(fuse::Region{.name = "fine_cells", .mesh = m_src, .dimension = 1});
  const auto r_dst = g.add_region(fuse::Region{.name = "coarse_cells", .mesh = m_dst, .dimension = 1});

  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "source", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "target", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});

  const auto f_out = g.add_field(a, fuse::FieldSpec{
      .name = "rho", .region = r_src,
      .disc = fuse::Discretization::FV,
      .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto f_in = g.add_field(b, fuse::FieldSpec{
      .name = "rho_in", .region = r_dst,
      .disc = fuse::Discretization::FV,
      .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});

  const auto ch = g.add_channel("rho_link", f_out, f_in);
  g.channel(ch).conservative();   // RemapFlags::Conservative; method "" = supermesh

  // ---- 2. Load the meshes, lower, schedule.
  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    if (const auto r = backend.load(g.mesh(fuse::MeshId{i})); !r.ok()) {
      std::cerr << r.error().code << ": " << r.error().message << "\n";
      return 1;
    }
  }
  const auto lowered = lower(g);
  if (!lowered.ok()) { std::cerr << lowered.diags.to_string(); return 1; }
  const fuse::ClockSet clocks{g};
  const auto sched = schedule(g, lowered.dag, clocks,
                              fuse::make_region_sizes(g, lowered.dag, backend));
  if (!sched.ok()) { std::cerr << sched.diags.to_string(); return 1; }

  // ---- 3. Components, registry, Runner.
  fuse::ComponentSet components;
  components.add(std::make_unique<CellField>("source", kSrcCells, true));
  components.add(std::make_unique<CellField>("target", kDstCells, false));
  fuse::FieldRegistry registry;
  for (auto& c : components) {
    if (const auto r = c.register_fields(registry, g); !r.ok()) {
      std::cerr << r.error().message << "\n";
      return 1;
    }
  }

  fuse::Runner runner{g, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry); !r.ok()) {
    std::cerr << r.error().code << ": " << r.error().message << "\n";
    return 1;
  }
  if (const auto r = runner.run(components, 3); !r.ok()) {
    std::cerr << r.error().code << ": " << r.error().message << "\n";
    return 1;
  }

  // ---- 4. What the run did.
  const fuse::RunStats& s = runner.stats();
  std::cout << "periods                  : " << s.periods << "\n"
            << "remap_weight_builds      : " << s.remap_weight_builds << "\n"
            << "remap_weight_nonzeros    : " << s.remap_weight_nonzeros << "\n"
            << "conservation_checks      : " << s.conservation_checks << "\n"
            << "conservation_violations  : " << s.conservation_violations << "\n"
            << "worst_conservation_error : " << s.worst_conservation_error << "\n";

  const auto* src = dynamic_cast<const CellField*>(components.find("source"));
  const auto* dst = dynamic_cast<const CellField*>(components.find("target"));
  double sum_src = 0.0, sum_dst = 0.0;
  for (double v : src->data()) sum_src += v;
  for (double v : dst->data()) sum_dst += v;
  std::cout << "sum over source cells    : " << sum_src << "\n"
            << "sum over target cells    : " << sum_dst << "\n"
            << "sum*measure source       : " << sum_src / kSrcCells << "\n"
            << "sum*measure target       : " << sum_dst / kDstCells << "\n";
  return s.conservation_violations == 0 ? 0 : 2;
}
```

Expected output (exact digits can vary by platform):

```text
periods                  : 3
remap_weight_builds      : 1
remap_weight_nonzeros    : 12
conservation_checks      : 3
conservation_violations  : 0
worst_conservation_error : 0
sum over source cells    : 39
sum over target cells    : 24.375
sum*measure source       : 4.875
sum*measure target       : 4.875
```

Eight cells onto five give 12 overlaps (each coarse cell overlaps two or
three fine cells), built once and applied three times. The
integral `sum_i f_i m_i` is the same on both sides; the plain sum of values is
not, because the values are densities.

What the lowering emitted for this channel (`dump_text(lowered.dag)`,
`fuse/ir/dump.hpp`; see [](graph-programming.md)):

```text
node 0   Source                    -> %0 : Instant<r0, K, Intensive, [WindowEnd, k], ranks=0>  "source.rho"
node 1   Remap        %0           -> %1 : Instant<r1, K, Intensive, [WindowEnd, k], ranks=0>  "rho_link/remap" {remap=C}
node 2   Constrain    %1           -> %2 : Instant<r1, K, Intensive, [WindowEnd, k], ranks=0>  "rho_link/constrain"
node 3   Diagnose     %2           -> %3 : Instant<r1, K, Intensive, [WindowEnd, k], ranks=0>  "rho_link/diagnose"
node 4   Sink         %3           -> %4 : Instant<r1, K, Intensive, [WindowEnd, k], ranks=1>  "target.rho_in"
```

`{remap=C}` is the `Conservative` flag (`M` and `H` are `Monotone` and
`HighOrder`); a method override prints as `{method=...}`.

:::{note}
The program declares a density (`Intensive`), so the conserved total is the measure-weighted
integral, `sum_i f_i m_i`, which is why the plain sums above differ and the integrals agree. For an
`Extensive` cell field the value is the amount itself (not a density): the supermesh map is
column-normalised so that the plain sum of the owned values is preserved, and the `Constrain`
check compares exactly that sum. The same program with `Conservation::Extensive` on both fields
prints 39 on both sides and `conservation_violations : 0`. The nodal load transfer in
[](interface-transfer.md) is the same convention applied to vertices.
:::

### The MOAB and TempestRemap path

Nothing in the graph changes. The flags, the discretisation pair and the
method string are the whole of the request; what changes is the **mesh
backend** handed to `Runner` and the **build**:

- configure with `-DFUSE_ENABLE_MOAB=ON` (which requires
  `-DFUSE_ENABLE_MPI=ON`) and load the meshes through
  `fuse::MoabMeshBackend` (`fuse/backend/moab_backend.hpp`) instead of
  `SerialMeshBackend` or `NativeMeshBackend`;
- `build_weights()` dispatches on the backend object it is handed, so a
  `SerialMeshBackend` pair takes the dependency-free supermesh path even in a
  MOAB-enabled build;
- the high-order builder is selected by `remap_method("tempest")`, by an
  `order` above 1 on the FV fields, or by `RemapFlags::HighOrder` (order 2
  when no order is given). It needs the MOAB that FUSE was built against to
  have been built with TempestRemap; otherwise `E-REMAP-NOTEMPEST`. Without
  MOAB at all the same request is `E-REMAP-NOHIGHORDER`;
- `Monotone` together with an order above 1 is `E-REMAP-UNSUPPORTED`: bound
  preservation and an order above 1 are not simultaneously available, and the
  first-order map is not returned under the high-order name.

### Errors at `initialize()` and what to do

The Runner forwards the builder's own code and prepends which two regions it
was trying to connect.

| Code | Meaning | Fix |
|---|---|---|
| `E-MESH-SOURCE` | the backend cannot parse `MeshDecl::source` (`SerialMeshBackend` accepts only `generated:line:N` and `generated:box:NxMxP`) | use a source the backend reads; see [](meshes.md) |
| `E-MESH-SIZE` | the generated entity counts overflow | smaller counts |
| `E-MESH-MOVED` | a moved-from backend was used | keep the backend alive for the Runner's lifetime |
| `E-MESH-PLACEMENT` | a remap asked for a processor set the backend was never told about | register the placement first ([](placement.md)) |
| `E-REMAP-REGION` | a region names an `entity_set` other than empty or `"all"` | remap whole meshes; a sub-set region has no index space for the matrix |
| `E-REMAP-DIM` | the two regions have different topological dimension | both sides must be cells of the same dimension |
| `E-REMAP-UNSUPPORTED` | a non-FV pair on an FV-only method, mismatched source and target orders, `Monotone` with order above 1, an unknown method string (on a plan that skipped `validate()`) | see the method table above |
| `E-REMAP-NOHIGHORDER` | an order above 1 or `HighOrder` without MOAB/TempestRemap | drop the request or build with MOAB |
| `E-REMAP-NOTEMPEST` | the high-order builder in a MOAB build whose MOAB lacks TempestRemap | rebuild MOAB with TempestRemap |
| `E-REMAP-NONPLANAR` | the dependency-free kernel was handed a surface in 3D (or a 1D curve varying along more than one axis) | a surface pair needs the TempestRemap tier |
| `E-REMAP-NOGEOMETRY` | a view reports no simplices for an entity | the mesh carries no geometry to intersect |
| `E-REMAP-COVERAGE` | `Conservative` was set and the target domain does not cover every source cell | make the domains coincide, or clear the flag and read `min_src_coverage` |
| `E-REMAP-MONOTONE` | `Monotone` was set and the built matrix is not bound preserving | order 1 only |
| `E-REMAP-PARALLEL` | a partitioned read with no remap halo built for the pair | call `build_remap_halo` after `load()` and before anything sizes a buffer |
| `E-REMAP-GEOMETRY` | the two spherical meshes are not on the same sphere | rescale (below) |
| `E-RT-REMAP-SIZE` | the buffer plan and the backend disagree about a region's size | the mesh changed after `schedule()`, typically a halo built too late |
| `E-RT-REMAP-HALO`, `E-RT-REMAP-NO-HALO` | a remap halo exists but the Runner does not exchange it, or a disjoint-placement native pair has none | drive the remap directly, or place both components on one rank |
| `E-RT-REMAP-NODAL-PARALLEL`, `E-RT-REMAP-MLS-PARALLEL` | a nodal or MLS remap on more than one rank outside the supported orientation | see [](interface-transfer.md) and [](point-cloud-remap.md) |

Applying weights by hand (`apply_weights` in `remap_plan.hpp`) adds
`E-REMAP-ARG` (null buffer or `components < 1`), `E-REMAP-COO` (triplet
arrays disagree or an index is out of range) and `E-REMAP-SIZE` (buffer
length overflow).

## The two dimensions

A mesh is characterised by two dimensions, not one:

- the **manifold dimension** is the intrinsic dimension of a cell — 2 for a
  triangle, 3 for a tetrahedron, 1 for an edge, 0 for a point;
- the **spatial dimension** is the dimension of the space the coordinates live
  in.

Their difference is the **codimension**, and it decides which operations are
meaningful rather than merely difficult:

| Codimension | Example | Normals |
|---|---|---|
| 0 | tetrahedra in 3D, triangles in 2D | none — the cells fill space |
| 1 | triangles in 3D, edges in 2D | exactly one per cell |
| 2 or more | edges in 3D | infinitely many |

Both dimensions are template parameters, so this is enforced when the code is
compiled rather than when it runs. Asking a volume mesh for a normal is not a
rejection you discover in a log; it does not build.

## Cells are simplices, and they remember their parent

Cells are triangles in 2D and tetrahedra in 3D. A hexahedral or polyhedral
mesh is decomposed, and **every simplex names the cell it came from**.

That parent map is load-bearing rather than bookkeeping. Conservation is
accounted per *user* cell: if a hexahedron becomes six tetrahedra, a
conservation sum must still balance over one cell, and a report that named
simplices would be answering a question nobody asked. `parent_cell_count()` is
the number to balance over; `simplex_count()` is an implementation detail of
how the geometry is stored.

Simplices were chosen over a canonical cell-type table for accuracy: clipping a
simplex assumes nothing about convexity, so a non-convex cell — or a hexahedron
whose quadrilateral faces are warped and therefore not planar — is handled
exactly, since a cell's intersection is exactly the sum of its simplices'
intersections.

## Measures are manifold measures

`measure()` returns the length, area or volume according to the *manifold*
dimension. A triangle in three-space has an area, and that is what you get. It
is computed from the Gram determinant of the edge matrix, which is the single
formula covering every dimension pair — a plain determinant of the 3D edge
matrix would report zero for that triangle.

A degenerate cell measures exactly zero, never a NaN. The distinction matters
because a NaN weight propagates through a conservation sum without announcing
itself, while a zero is visible as a cell that received nothing.

## What the mesh does not carry

Fields. Many mesh libraries bundle point, cell and global data into the
mesh; FUSE already has a field layer carrying units, intent and
semantic shape, and a second home for a field would be two places to look for
one and two places for its units to disagree. The mesh carries geometry only.

## How the conservative map is built

For every target cell `j` and source cell `i`, the exact measure of their
intersection is computed by clipping simplices and taking analytic moments, and

```
W_ji = |T_j intersect S_i| / m_j
```

so a target value is the measure-weighted average of the source field over the
target cell. That one formula is both **constant-preserving** (rows sum to one
where the meshes overlap fully) and **conservative** (total mass is unchanged),
which is why there is no separate "conservative" switch: a first-order
supermesh map is conservative by construction. The audit fields on the weights
report how nearly it achieved both on the pair it was given, rather than
asserting it.

Measured on non-matching pairs over the unit domain, all residuals at rounding:

| Pair | Row defect (constants) | Column defect (mass) |
|---|---|---|
| 8 → 5 cells, 1D | < 1e-14 | < 1e-14 |
| 3×3 → 4×4 triangulated quads, 2D | < 1e-13 | < 1e-13 |
| 3×3×3 → 4×4×4 hexes, 3D | < 1e-13 | < 1e-13 |

Remapping a mesh onto itself returns exactly the identity: one entry per cell,
each equal to 1.

### Two things it rejects

**A surface pair.** The 2D kernel clips in the xy-plane. Handed a surface
embedded in three-space it would return the measure of its *shadow* — a
plausible wrong answer — so it rejects with `E-REMAP-NONPLANAR`. A surface pair
needs a parameterisation, or great-circle geometry on the sphere, which is what
the MOAB-enabled build configuration provides via TempestRemap.

**Anything above order 1.** Piecewise constant is what this path reconstructs.
Asking for high order without MOAB is rejected by name rather than quietly
served at order 1.

### Meshes at different radii

A spherical covering-set construction (MOAB's, above) needs its two meshes on
the SAME sphere. Two real files at genuinely different nominal radii -- an
atmosphere mesh at a unit radius and an ocean mesh at Earth radius in metres,
say -- are refused outright, `E-REMAP-GEOMETRY`, "not the same domain": the
library cannot know two different radii are meant to be the same sphere in
different units, so it never guesses.

**Rescaling is the driver's job, never the library's.**
`MoabMeshBackend::sphere_radius(MeshId[, ComponentId])` measures a mesh's
vertices as a sphere about the origin and reports the mean radius and the
largest single vertex's deviation from it, so a driver can decide what, if
anything, needs doing. `MoabMeshBackend::project_onto_unit_sphere(MeshId[,
ComponentId])` measures the same way, then -- unless the mesh is ALREADY on
the unit sphere to within its own tolerance -- rescales every vertex by
`1 / mean_radius` and reports what it did (`SphereProjectionReport`:
`mean_radius`, `applied_factor`, `already_unit`). Already-unit is a measured
no-op: no vertex coordinate is read back or rewritten.

Both come in two forms: a `MeshId`-only overload for the backend's default
decomposition, and a `(MeshId, ComponentId)` overload for a NAMED placement's
own, independently-decomposed copy ([](placement.md)'s
per-placement model) -- `set_placements()` gives every declared processor set
its own Core with its own independently-read copy of every loaded mesh, so
rescaling one placement's copy does nothing to another's, and a driver
coupling two independently placed components over real files must call the
placement-aware overload on EACH placement whose copy it will use.

The library itself never calls either function: `build_remap_halo` refuses
`E-REMAP-GEOMETRY` on a mismatched pair, whether or not a rescale was available. A driver loading real, differently-scaled
spherical files -- see `FUSE_SPHERE_ATM`/`FUSE_SPHERE_OCN` in the
test suite for a worked example -- queries or rescales explicitly,
after `load()` and before the first `build_remap_halo`/`build_weights` call,
collectively (every rank holding a piece of the mesh, or of that placement's
own share of it, in the same order).

### The covering set is intersected with a k-d-tree search, not an advancing front

MOAB offers two different algorithms for deciding which source and target cells in the covering
set actually overlap: an advancing front that walks cell adjacency outward from a small number of
seed pairs, and a k-d-tree search that tests every target cell against the covering set directly
by spatial proximity. FUSE's own `assemble()` always uses the k-d-tree search. On a partitioned (multi-rank) mesh the
advancing front can leave source cells that are present on a rank, and geometrically covered by that
rank's own target cells, with no intersection at all; a target cell such a cell alone would have
covered is then silently left short, with no error, only a lower delivered total than the true one.
The likely cause is adjacency: the front expands only through cells that share MOAB's own vertex
identity, which a cell adopted through the parallel exchange need not share with the resident cells.
The k-d-tree search walks no adjacency and has no such dependency. Regardless of algorithm, a driver
should always check `RemapWeights::min_src_coverage`/`min_tgt_coverage` -- not just whether
`build_weights` returned -- before relying on a conservative channel's own total. See
`docs/dev/moab-intersection.md` for the mechanism in full.

### A remap halo on a placement is adopted into THAT placement's own Core

`build_remap_halo`'s covering-set construction (above) runs inside whichever
`moab::Interface` the two meshes were resolved through -- the backend's
default Core when no placement is named, or a DECLARED placement's own,
independently-read Core (`set_placements()`, `docs/user/placement.md`)
otherwise. The covering cells that construction finds are adopted back into
THAT SAME Core --
`MoabMeshBackend::adopt_remap_halo`, `has_remap_halo`, `remap_halo_size`,
`exchange_remap_halo` and `accumulate_remap_halo` all take an optional
trailing `ComponentId placement` (default: the unplaced decomposition,
matching every existing caller that does not pass one), and resolve every
MOAB call -- the GLOBAL_ID tag read, `add_entities`, the gather/scatter
exchange -- through that placement's own Core and communicator, never the
default Core's unconditionally. A `moab::EntityHandle` is only ever
meaningful within the ONE Core that allocated it; reading one back through
a DIFFERENT Core is undefined by MOAB's own contract, which is why every call
above is routed through the placement's own Core.

The bookkeeping this adopts into (`has_remap_halo`'s own cache) is keyed by
`(source, target, dim, placement)`, not `(source, target, dim)` alone: the
default Core's own halo for a pair and a placement's own halo for the
IDENTICAL pair are two independent entries, and building one never makes
the other read as built. A driver (or the runtime's own weight builder)
that resolves a channel's mesh pair through a placement must pass that SAME
placement to `has_remap_halo`/`remap_halo_size` to find the halo
`build_remap_halo` actually adopted for it.

The collective agreement a halo adoption needs (every rank must agree
whether the covering set it found is trustworthy before any rank acts on
it) runs over the SAME communicator the covering-set construction itself
used -- a placement's own sub-communicator when one is named, never the
backend's base communicator on its behalf. A rank outside the named
placement has nothing to agree on with that placement's own members and
takes no part in it at all.

### A note on negligible overlaps

Two cells that merely share a face have an intersection of measure zero, but
clipping against a coplanar plane returns a rounding-level positive — measured
at up to 1.1e-16 of a cell, which on an identity pair produced 84 spurious
entries beside the 27 real ones. Contributions below 1e-12 of the target cell's
measure are dropped.

The reason is not accuracy: those entries could not move a double-precision
result. It is that **a matrix entry is a data dependency**. In parallel it makes
the target rank wait on whichever rank owns that source cell, for a
contribution of 1e-16. The floor is applied before the conservation audit, so
any mass it discards is reported in the residuals rather than hidden.

## What the conservative map is checked against

Conservation and constant preservation are necessary but **not sufficient**: a
map that sent mass to the wrong cells could still sum correctly along both
axes. The checks below are ordered by what they can catch that the previous one
cannot.

| Check | What it would catch | Result |
|---|---|---|
| Conservation and constant residuals | mass created or destroyed | < 1e-13 in 1D, 2D and 3D |
| Identity pair | any error at all in the clipper | exactly one entry per cell, each 1.0 |
| **Entrywise against closed form** | mass delivered to the *wrong* cell | worst entry error 2.8e-15 over 1331 entries on coprime 5→7 grids; no missing and no spurious entries |
| **Common affine map** | an axis-aligned assumption anywhere | weights change by 6.6e-15 under a rotation, shear and translation |
| **Refinement study** | a map that conserves but does not converge | observed order **1.05** over five refinements |
| Input winding | a dependence on vertex ordering, which the mesh contract does not fix | identical map from clockwise and counterclockwise input |
| **1e-10 slivers** | a hidden tolerance in the candidate search | slivers kept; smallest weight 4.0e-10 |

The closed-form check is the strongest available for the geometry: for two
axis-aligned grids the exact intersection volume is the product of the per-axis
overlaps, so every individual weight has a reference value rather than only the
row and column sums.

The refinement study is one-dimensional, deliberately. The candidate search is
a brute-force box test, and a five-level 3D study would take hours —
the 5×5×5 → 7×7×7 case alone is about 1.5 million clip calls. In 3D the kernel
is pinned against closed form instead, which says more about the geometry than
a convergence slope does.

## How a cell becomes simplices

Every backend presents its cells to the clipping kernel as simplices, so that
kernel has one geometric case instead of six. *Which* corners form which
simplex is not each backend's business: it is a fixed table in
`include/fuse/backend/cell_simplices.hpp`, covering edges, triangles, quads,
convex polygons, tetrahedra, pyramids, prisms and hexahedra.

The table is shared rather than duplicated for a reason that is easy to miss.
A hexahedron with **non-planar faces is not a polyhedron**: each choice of
diagonal triangulates those faces differently, so the two decompositions
enclose genuinely different solids with different volumes. Neither is wrong.
But it means "the volume of this cell" is not a property of the cell alone, and
two backends that decomposed the same mesh about different diagonals would
disagree about cell volumes — a conservation check between them would then
report a defect that is really a convention mismatch. One table removes that
possibility, and
`CellSimplices.ANonPlanarHexHasNoSingleVolumeAndThatIsWhyTheTableIsFixed`
records the fact.

The tables are `constexpr` and device-callable, so a kernel can decompose a
cell in place instead of asking the host for its corner list.

## See also

- [](meshes.md) for the mesh backends, the generated-mesh grammar and partitioning
- [](placement.md) for per-placement meshes and Cores
- [](transfer-maps.md) and [](point-cloud-remap.md) for the maps built on this geometry
- [](interface-transfer.md) for the nodal interpolation and load-transfer pair
- [](runtime.md) for `RunStats` and `RunnerConfig`
