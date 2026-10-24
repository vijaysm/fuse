# MOAB-free remap for point clouds

This page is for remapping scattered points (gauges, buoys, a LIDAR return)
or sampled mesh values with a smooth local polynomial fit instead of a
cell-overlap map: how to declare the fit, what it guarantees, and where it
stops.

ArborX (already a required dependency behind `fuse::search`, for its BVH) also ships a
moving-least-squares point-cloud interpolant. `fuse::build_mls_weights`
(`fuse/backend/mls_weights.hpp`) is a `RemapWeights` builder around the same idea, usable without
MOAB, for `Discretization::PointCloud` fields and for mesh fields sampled at vertices or cell
centroids.

```cpp
#include "fuse/backend/mls_weights.hpp"

fuse::RemapPlan plan;
plan.src = ocean_buoys;       // Discretization::PointCloud
plan.tgt = forecast_grid_centroids;  // Discretization::FV
plan.src_disc = fuse::Discretization::PointCloud;
plan.tgt_disc = fuse::Discretization::FV;
plan.method = "mls:degree=2,k=12,radial=wendland2";

const auto w = fuse::build_mls_weights(plan, graph, backend);
if (!w.ok()) { /* E-MLS-OPTIONS, -REGION, -DISC, -NEIGHBORS, -RANK, -MONOTONE, -UNSUPPORTED */ }
```

The result is an ordinary `RemapWeights`: `apply_weights`, `WeightsOperator`, and every audit field
(`order`, `monotone`, `min_weight`, `max_row_defect`) work exactly as they do for the supermesh or
TempestRemap builders. Selecting method `"mls"` (or `"mls:..."`) on a field transfer routes it to this builder.

## Declaring a point cloud

A point cloud is a field whose discretisation is `Discretization::PointCloud`
on a region whose mesh exposes the points as dimension-0 entities
(`MeshView::centroid(0, i)` is the point, `global_id(0, i)` its id):

```cpp
const auto buoys = g.add_region(fuse::Region{.name = "buoys", .mesh = m_buoys, .dimension = 0});
const auto sst = g.add_field(ocean_obs, fuse::FieldSpec{
    .name = "sst_obs", .region = buoys,
    .disc = fuse::Discretization::PointCloud,
    .units = fuse::units::kKelvin,
    .intent = fuse::Intent::Provides});
```

Such a field is built into weights through `RemapPlan` and
`build_mls_weights` directly, as in the snippet above. It cannot be run
through the `Runner`: a field buffer is sized by `field_extent`, which has no
definition for `PointCloud` and refuses it with `E-EXTENT-DISC` at
`make_region_sizes`. Through a coupler, MLS is available for the mesh
discretisations in the table below (FV cell centroids, or `ContinuousGLL`
order-1 vertices), selected with `remap_method("mls:...")` on the channel.

## The API

```cpp
enum class MlsRadial { Wendland0, Wendland2, Wendland4, Wendland6 };

struct MlsOptions {
  int       degree = -1;              // total polynomial degree; -1 = undeclared
  int       k = 0;                    // neighbour count; 0 = undeclared
  MlsRadial radial = MlsRadial::Wendland2;
};

// Parses "mls:degree=<int>,k=<int>,radial=<name>". E-MLS-OPTIONS otherwise.
Result<MlsOptions> parse_mls_options(std::string_view method);

// C(dim + degree, dim): the basis size k must reach. 0 for dim outside 1..3 or degree < 0.
std::size_t mls_basis_size(int dim, int degree) noexcept;

// The declared CRBF at x = distance / support_radius, clamped to [0, 1].
double mls_crbf(MlsRadial r, double x) noexcept;

// The builder build_weights() dispatches to for method "mls" / "mls:...".
Result<RemapWeights> build_mls_weights(const RemapPlan& plan, const CouplingGraph& graph,
                                       const MeshBackend& backend);
```

## A complete example

Two generated line meshes, the fit built between their cell centroids
directly from a `RemapPlan`, applied to a linear field, and the refusals
provoked on purpose. Serial build, no optional library:

```cpp
#include <fuse/backend/mls_weights.hpp>
#include <fuse/backend/serial_backend.hpp>

#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  // 1. Two meshes and two regions in a graph; the serial backend loads them.
  fuse::CouplingGraph g;
  const auto m_src = g.add_mesh(fuse::MeshDecl{.name = "fine", .source = "generated:line:12"});
  const auto m_tgt = g.add_mesh(fuse::MeshDecl{.name = "coarse", .source = "generated:line:5"});
  const auto r_src = g.add_region(fuse::Region{.name = "fine_cells", .mesh = m_src, .dimension = 1});
  const auto r_tgt = g.add_region(fuse::Region{.name = "coarse_cells", .mesh = m_tgt, .dimension = 1});
  fuse::SerialMeshBackend backend;
  (void)backend.load(g.mesh(m_src));
  (void)backend.load(g.mesh(m_tgt));

  // 2. The plan: FV cell centroids on both sides, degree 1, three neighbours.
  fuse::RemapPlan plan;
  plan.src = r_src;
  plan.tgt = r_tgt;
  plan.src_disc = fuse::Discretization::FV;   // centroid of each cell
  plan.tgt_disc = fuse::Discretization::FV;
  plan.method = "mls:degree=1,k=3,radial=wendland2";

  const auto w = fuse::build_mls_weights(plan, g, backend);
  if (!w.ok()) { std::cerr << w.error().code << ": " << w.error().message << "\n"; return 1; }

  // 3. A linear field is reproduced exactly at the target centroids.
  const fuse::MeshView& src = *backend.view(backend.mesh_id("fine").value()).value();
  const fuse::MeshView& tgt = *backend.view(backend.mesh_id("coarse").value()).value();
  std::vector<double> f(src.entity_count(1)), out(tgt.entity_count(1));
  for (std::size_t i = 0; i < f.size(); ++i) f[i] = 3.0 - 2.0 * src.centroid(1, i).x;
  (void)fuse::apply_weights(w.value(), f.data(), out.data(), 1);
  double worst = 0.0;
  for (std::size_t j = 0; j < out.size(); ++j) {
    worst = std::max(worst, std::abs(out[j] - (3.0 - 2.0 * tgt.centroid(1, j).x)));
  }
  std::cout << "nnz              : " << w.value().nnz() << "\n"
            << "order            : " << w.value().order << "\n"
            << "max_row_defect   : " << w.value().max_row_defect << "\n"
            << "min_weight       : " << w.value().min_weight << "\n"
            << "monotone         : " << w.value().monotone << "\n"
            << "max_col_defect   : " << w.value().max_col_defect << "\n"
            << "worst linear err : " << worst << "\n";

  // 4. The options are never defaulted.
  for (const char* bad : {"mls", "mls:degree=1,k=3", "mls:degree=1,k=3,radial=gaussian"}) {
    plan.method = bad;
    const auto r = fuse::build_mls_weights(plan, g, backend);
    std::cout << bad << " -> " << r.error().code << "\n";
  }
  // 5. k below the basis size, and the Conservative flag.
  plan.method = "mls:degree=2,k=2,radial=wendland2";
  std::cout << "degree=2,k=2 -> " << fuse::build_mls_weights(plan, g, backend).error().code << "\n";
  plan.method = "mls:degree=1,k=3,radial=wendland2";
  plan.flags = fuse::RemapFlags::Conservative;
  std::cout << "Conservative -> " << fuse::build_mls_weights(plan, g, backend).error().code << "\n";
  return 0;
}
```

Expected output (exact digits can vary by platform):

```text
nnz              : 15
order            : 2
max_row_defect   : 4.44089e-16
min_weight       : -0.00121992
monotone         : 0
max_col_defect   : inf
worst linear err : 8.88178e-16
mls -> E-MLS-OPTIONS
mls:degree=1,k=3 -> E-MLS-OPTIONS
mls:degree=1,k=3,radial=gaussian -> E-MLS-OPTIONS
degree=2,k=2 -> E-MLS-NEIGHBORS
Conservative -> E-MLS-UNSUPPORTED
```

Five targets times three neighbours is 15 entries. `order` is `degree + 1`.
The row sums are 1 to rounding because every basis contains the constant;
the small negative weight is what a degree-1 fit puts on its far neighbour,
so `monotone` is false. `max_col_defect` is `+inf` on purpose: no
conservation was measured, so no conservation can be claimed.

The same fit on a channel between the two regions, through the coupler, is
one line and needs no `RemapPlan`:

```cpp
g.channel(ch).remap_method("mls:degree=1,k=3,radial=wendland2");
```

The lowering emits the `Remap` with `{method=mls:degree=1,k=3,radial=wendland2}`
and no `Constrain` after it (a pointwise interpolation has no conserved
integral to check), and `RunStats::remap_weight_builds` counts the build.
Run on the 8-cell to 5-cell pair of [](geometry.md) it reports
`remap_weight_nonzeros : 15` and `conservation_checks : 0`.

## Declaring the fit -- never a default

`RemapPlan::method` is the only lever a field transfer has, so the polynomial degree, the neighbour count
and the radial weight are encoded in it, and all three are **required**:

```
mls:degree=<int>,k=<int>,radial=<wendland0|wendland2|wendland4|wendland6>
```

`method == "mls"` alone, or any key missing, misspelled, or duplicated, is `E-MLS-OPTIONS`. There is
no fallback degree, no fallback neighbour count, and no fallback radial family -- a field-transfer author
who does not say what they want does not get a plausible guess.

`degree` is the TOTAL polynomial degree of the local fit, in the point cloud's own AMBIENT
dimension (1, 2 or 3, read from the data: an unused coordinate axis is exactly zero, the same
convention `fuse::search::classify_dim` already uses). A cloud scattered on a sphere is still
ambient-3D -- the fit is ordinary 3D monomials restricted to the sphere by the data, not a
manifold-aware fit, which is also what ArborX's own class does.

`k` must be at least the basis size `C(dim + degree, dim)` (`fuse::mls_basis_size`): fewer
neighbours than the fit has degrees of freedom is `E-MLS-NEIGHBORS`, checked once per build (as soon
as the ambient dimension is known) rather than discovered per target as a rank failure. A source
side with fewer than `k` points overall is the same rejection.

## Which discretisations have a point extraction

| `(disc, order)` | Points read from |
|---|---|
| `PointCloud`, any order | `MeshView::centroid(0, i)`, dimension 0 |
| `ContinuousGLL`, order 1 | `MeshView::centroid(0, i)`, dimension 0 (vertex -- `mesh_view.hpp`'s own nodal convention) |
| `FV` or `DiscontinuousGLL`, order 0 | `MeshView::centroid(region_dim, i)`, the cell centroid |

Anything else is `E-MLS-DISC`. `RemapFlags::Conservative` is rejected outright (`E-MLS-UNSUPPORTED`):
MLS fits a smooth local polynomial and does not balance a mass budget, and a point cloud need not
even define `MeshView::measure`. `RemapFlags::Monotone` is checked on the assembled matrix, the same
way the TempestRemap builder checks it -- rejected (`E-MLS-MONOTONE`) only if the check actually
fails, which is generic for `degree=0` (a normalised weighted average) and not for `degree >= 1`.

## Why the coefficients are computed here, not read out of ArborX

ArborX 2.1's `ArborX::Interpolation::MovingLeastSquares` does the same computation, on a GPU if
asked. Its coefficients (`_coeffs`, a `num_targets x num_neighbors` view) are a **private** member
with no accessor -- the only public operation is `interpolate()`, which multiplies through them for
one field and hands back values, never the matrix. A free function one namespace deeper,
`ArborX::Interpolation::Details::movingLeastSquaresCoefficients`, really does return that view
directly. It was read in full before writing this builder, and deliberately not used, for two
reasons:

1. `Details::` is an implementation-detail namespace ArborX makes no compatibility promise about --
   this codebase's own `fuse::search` had to reach into a different corner of `ArborX::Details` once
   already (`box_index.cpp`'s header comment), and confined that to one `.cpp`, not exposed. Wiring
   FUSE's audited, transposable weight matrix to that surface for a feature meant to work without
   MOAB at all would trade a `MOAB` dependency for an unstable-ArborX-internals one.
2. It would not buy back what this builder's rejections need. `movingLeastSquaresCoefficients`'s
   kernel inverts the local moment matrix with a symmetric SVD pseudo-inverse and, when that matrix
   is singular, silently zeroes the dependent rows/columns and substitutes a unit diagonal entry so a
   coefficient always comes out -- no rank, no condition number, no signal to the caller at all. This
   builder's contract is the opposite: reject, by name, with a measured condition estimate, when a
   target's local fit is rank-deficient.

So the coefficient matrix is computed in FUSE. `fuse::search::BoxIndex::query_nearest` supplies the
`k` nearest source points (each source point indexed as its own degenerate, zero-size box); a
compactly supported radial weight (Wendland, below) is evaluated per neighbour; and the local fit is
one call to `fuse::linalg::least_squares_builtin` per target.

### The normal-equations reduction

`fuse::linalg::least_squares` solves `min ||A a - b||` for one right-hand side per call. Extracting
all `k` per-neighbour weights that way would need `k` refactorisations of the same weighted
Vandermonde matrix (one canonical right-hand side each) -- correct, but wasteful, and this builder
instead solves the mathematically equivalent, much smaller system directly.

Let `V` be the `k x P` matrix whose rows are the polynomial basis (`P = ` `mls_basis_size(dim,
degree)`) evaluated at each neighbour, recentred so the target sits at the local origin, and `Phi`
the diagonal matrix of radial weights. The weighted least-squares fit `min ||sqrt(Phi)(V a - f)||`
solves `M a = V^T Phi f` where `M = V^T Phi V` (the `P x P` moment matrix). The reconstructed value at
the target is `p(0) = a_0` (the coefficient of the constant basis term, since every other monomial
vanishes at the recentred origin), and because `a_0` is linear in the source values `f`:

```
a_0 = e_1^T M^{-1} V^T Phi f = (M^{-1} e_1)^T V^T Phi f = sum_j (c . basis_j) * phi_j * f_j
```

with `c = M^{-1} e_1`. So `weight_j = phi_j * (c . basis_j)`, and `c` is the solution of the single
`P x P` system `M c = e_1` -- one call to `fuse::linalg::least_squares_builtin(M, P, P, e_1, c)`, run
through the same column-pivoted, rank-truncating QR the rest of FUSE's least-squares work uses. Its
returned rank and the diagonal of the factorisation it leaves in its working array are exactly this
builder's rank-deficiency and condition-estimate diagnostics -- no second factorisation needed.

Both `weight_j` are proportional to `phi_j`, so this formula is invariant to any positive overall
rescaling of the radial weight for one target's fit: `mls_crbf`'s Wendland polynomials are ArborX's
own tabulated ones, which are not all normalised to 1 at distance 0 (`Wendland<4>`'s constant term is
3), and that does not change a single weight.

**Cross-checked, not merely asserted equivalent**: `mls_weights_test.cpp` builds ArborX's own
`MovingLeastSquares` class directly -- the public one -- on the same source/target clouds with the
same declared degree, `k` and radial weight, calls its `interpolate()`, and compares against this
builder's `apply_weights` to round-off (`MatchesArborXsOwnInterpolateOnTheSameCloud2D`/`3D`).

## The radial weight

Four Wendland compactly supported radial basis functions, the same family ArborX's own
`CRBF::Wendland<N>` offers: `wendland0`, `wendland2`, `wendland4`, `wendland6` (higher `N` is smoother
at the support boundary and flatter near the target). The support radius is `1.1x` the farthest of
the `k` declared neighbours -- the same choice ArborX's own kernel makes, and for the same reason: a
CRBF is exactly zero at its support boundary, so scaling past the farthest neighbour keeps every
neighbour's weight strictly positive. A target whose `k` neighbours all coincide with it (radius
would be zero) is floored at `std::numeric_limits<double>::epsilon()` before the `1.1x` scaling, so
no divide-by-zero is possible -- checked directly by
`SourcePointExactlyAtTargetDoesNotDivideByZero`.

## Coincident points

A source point sitting exactly on a target, or several source points sitting exactly on each other,
must not crash this builder, and do not: the radial weight is finite (in fact maximal) at distance
zero, and duplicate points simply degrade the local moment matrix's rank like any other loss of
geometric information -- caught, if the degradation is severe enough, by the same `E-MLS-RANK`
rejection a collinear or coplanar neighbour set triggers, with a measured condition estimate in the
message. `AllCoincidentNeighboursIsRankDeficient` pins this for a fit whose basis needs more than
one independent direction; `DuplicateSourcePointsDoNotCrashAndDegreeZeroStillReproducesAConstant`
pins the case where duplicates are present but the fit (degree 0) still succeeds.

## Accuracy

- **Exactness.** A degree-`D` polynomial is reproduced to round-off in 1D, 2D, 3D, and on points
  scattered on a sphere (`ExactForDegree2Polynomial1D`/`ExactForDegree1Polynomial2D`/
  `ExactForDegree2Polynomial3D`/`ExactForDegree1PolynomialOnScatteredSpherePoints`). The 3D case uses
  a scattered, not a regular-grid, source cloud: a perfectly regular grid is a classic MLS
  degeneracy for an even-degree fit (measured on a 6x6x6 grid, `k=16`, `degree=2`: rank 9 of the 10
  basis functions needed, `E-MLS-RANK`) -- a property of a symmetric sampling, not a defect in this
  builder.
- **Convergence order.** `degree + 1` under refinement of a quasi-uniform 2D grid for a smooth,
  non-polynomial field (`ConvergenceOrderIsDegreePlusOneOnRefinement`); the k-nearest support radius
  scales with the grid spacing automatically, which is what gives the right asymptotic order without
  a separately declared radius.
- **Every named rejection** fires in its own test (`mls_weights_test.cpp`'s `MlsOptions.*` and
  `MlsRefusals.*` suites): malformed options, an invalid or sub-set mesh entity set, an unsupported
  `(disc, order)`, `k` below the basis size, a source side with too few points, a rank-deficient
  local fit (with the condition estimate present in the message), the `Conservative` flag, a
  detectably multi-rank backend, and the `Monotone` flag on a fit that is not bound preserving.

## Serial only

This builder searches for a target's `k` nearest neighbours only among the source entities this
rank's `MeshView` reports -- no halo. On a genuinely partitioned read, a target near a partition
boundary could have its true nearest neighbours on another rank, and this builder would never see
them: a silently locally-correct, globally-wrong answer, the same failure mode `build_remap_halo`
exists to close for the supermesh path. Distributed MLS is not supported. Where
this builder can detect the multi-rank case (a `fuse::native::NativeMeshBackend` reporting
`num_ranks() > 1`) it rejects the configuration with `E-MLS-UNSUPPORTED`, naming the reason; against an opaque `MeshBackend`
it cannot recognise, it has no way to detect the case at all, and returns an answer computed only
from what that rank can see.

Through the `Runner` the check is made from the graph rather than the
backend: a channel whose method is `"mls"` or `"mls:..."` and whose source or
target component is declared on more than one rank is refused at
`initialize()` with `E-RT-REMAP-MLS-PARALLEL`, whatever backend is in use.
Place both components on one rank, or use `"consistent"` for a nodal source
([](interface-transfer.md)).

## Selecting it on a field transfer

`build_mls_weights` needs no MOAB and no mesh geometry, so `build_weights()`
dispatches to it on the method string alone, in builds with and without MOAB, ahead of
the discretisation check that restricts other methods to finite-volume fields:

```cpp
g.channel(ch).remap_method("mls:degree=2,k=12,radial=wendland2");
```

## See also

- [](geometry.md) for the supermesh alternative, the method strings and a complete coupler run
- [](interface-transfer.md) for the nodal interpolation `H`, the other pointwise method
- [](transfer-maps.md) for persisting the resulting weights
- [](meshes.md) for the mesh backends that supply the points
