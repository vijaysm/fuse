#pragma once

/// Moving-least-squares (MLS) remap weights for point clouds, and for mesh
/// fields sampled at vertices or cell centroids when MOAB is absent.
///
/// ===========================================================================
/// WHY THIS FILE EXISTS
/// ===========================================================================
///
/// `intx_weights.cpp`'s two builders -- the exact-simplex supermesh
/// (dependency-free) and the MOAB/TempestRemap tier -- both need CELL
/// TOPOLOGY: a simplex decomposition to clip, or a mesh MOAB can intersect.
/// `Discretization::PointCloud` has none by definition (`mesh_view.hpp`'s
/// `field_extent` already refuses it, "a point cloud has no cell topology to
/// size a field by at all"), and a scattered observation network -- gauges,
/// buoys, a LIDAR return -- is exactly a point cloud. This builder answers a
/// different question: given a target point and a handful of nearby source
/// points, reconstruct a LOCAL POLYNOMIAL through them and evaluate it at the
/// target. That is moving least squares, and it needs nothing but point
/// positions, which `MeshView::centroid` gives for cells, vertices, and (via
/// `Discretization::PointCloud`, dimension 0) unstructured points alike.
///
/// ===========================================================================
/// WHERE THE COEFFICIENTS COME FROM -- ARBORX'S CLASS, OR OURS
/// ===========================================================================
///
/// ArborX 2.1 (the pinned commit in `UPSTREAMS.lock`) ships
/// `ArborX::Interpolation::MovingLeastSquares`, which does exactly this on a
/// GPU or host execution space. It was read before a line of this file was
/// written, specifically to answer: can this builder just call ArborX and
/// copy its coefficients into a `RemapWeights` triplet?
///
/// NO, not through the class's own public surface. `MovingLeastSquares`
/// stores `_coeffs` (a `Kokkos::View<FloatingCalculationType**, MemorySpace>`,
/// shape `num_targets x num_neighbors`) and `_indices` as PRIVATE members with
/// no accessor; the only public operation is `interpolate(space, source_values,
/// approx_values)`, which multiplies through them for one field at a time and
/// hands back values, not weights. There is no public way to get the matrix
/// out of an instance of the class.
///
/// The coefficients ARE computed by a free function one namespace deeper --
/// `ArborX::Interpolation::Details::movingLeastSquaresCoefficients` (in
/// `detail/ArborX_InterpDetailsMovingLeastSquaresCoefficients.hpp`), which
/// really does return the `num_targets x num_neighbors` Kokkos view directly,
/// callable independently of the class. That function was read in full
/// (`docs/user/point-cloud-remap.md` §"ArborX read" reproduces the header
/// comment's own account of it), and NOT used here, for two reasons that both
/// matter more than the convenience of not reimplementing MLS:
///
///   1. `Details::` is exactly what it says: an implementation-detail
///      namespace ArborX makes no compatibility promise about, one level
///      below the class it backs. `fuse::search` (this codebase's own ArborX
///      facade, one layer below this file) already had to reach into
///      `ArborX::Details` once, for an unrelated reason (`box_index.cpp`'s
///      own header comment; ArborX 2.1's `DefaultCallback` does not extract a
///      plain index), and that reach is confined to one `.cpp`, not exposed.
///      Building FUSE's audited, transposable weight matrix on a namespace
///      named `Details` -- reachable ONLY through a header path
///      (`detail/ArborX_Interp...`) that is not part of ArborX's own
///      `#include <ArborX_InterpolationMovingLeastSquares.hpp>` umbrella --
///      would wire this library's remap weights to an internal ArborX
///      artifact that can change shape or vanish in a point release with no
///      notice, for a feature (`fuse::backend`) that is supposed to work
///      without MOAB at all.
///   2. It would not have bought back the property this file's refusals
///      exist for. `movingLeastSquaresCoefficients`'s kernel inverts the local
///      moment matrix with a SYMMETRIC SVD PSEUDO-INVERSE
///      (`ArborX::Details::symmetricPseudoInverseSVDKernel`) and, when that
///      matrix is singular, ZEROES the dependent rows/columns and substitutes
///      a unit diagonal entry (row-echelon elimination inside the kernel) so
///      that a coefficient always comes out -- silently, with only a
///      `#ifndef NDEBUG` internal `KOKKOS_ASSERT` and no code path that
///      returns a rank, a condition number, or any signal to the caller at
///      all. This library's contract for this method is the opposite:
///      REFUSE, by name, with a measured condition estimate, when a target's
///      local fit is rank-deficient (below). Reaching into ArborX's kernel
///      would have to overrule that behaviour after the fact, in a kernel
///      not written to expose what it decided.
///
/// So: **the coefficient matrix is implemented HERE, in FUSE**, exactly as the
/// task anticipates for the case where ArborX's own is not reachable through
/// a stable surface. `fuse::search::BoxIndex::query_nearest` supplies the k
/// nearest source points (the ArborX-backed BVH, used exactly as the nodal
/// track's k-nearest work already established it), a compactly supported
/// radial weight (Wendland, §"radial weight" below) is evaluated in this file,
/// and the local polynomial fit is one call to
/// `fuse::linalg::least_squares_builtin` per target -- not on the k x P
/// weighted Vandermonde system directly (that facade takes one right-hand
/// side per call, so getting all k per-neighbour weights would need k
/// refactorisations of the same matrix), but on the mathematically equivalent
/// P x P NORMAL-EQUATIONS system `M c = e_1` where `M` is the local moment
/// matrix and `P` the polynomial basis size (derivation in
/// `docs/user/point-cloud-remap.md`). That system is exactly `P` unknowns,
/// solved once, and its own rank and its own pivoted-diagonal ratio -- read
/// straight off the factorisation `least_squares_builtin` leaves behind -- are
/// this file's rank-deficiency and condition-estimate diagnostics. This is
/// CROSS-CHECKED, not merely asserted equivalent: `mls_weights_test.cpp`
/// builds `ArborX::Interpolation::MovingLeastSquares` directly (the PUBLIC
/// class, ArborX's own supported entry point) on the same source/target
/// clouds and the same declared degree/k/radial weight, calls its
/// `interpolate()`, and compares the reconstructed values against this file's
/// `apply_weights` to round-off.
///
/// ===========================================================================
/// OPTIONS ARE DECLARED, NEVER DEFAULTED
/// ===========================================================================
///
/// `RemapPlan::method` is the one lever a channel has (`RemapSpec::
/// method_override`, `ChannelBuilder::remap_method`, threaded verbatim
/// through `OpAttrs::method` to here) -- there is no separate options struct
/// on the plan, so the polynomial degree, the neighbour count, and the radial
/// weight are ENCODED IN THE METHOD STRING and parsed by `parse_mls_options`
/// below. The dispatch predicate (owned by `intx_weights.cpp`, not this file):
/// `plan.method == "mls" || plan.method.rfind("mls:", 0) == 0`.
///
///   method = "mls:degree=<int>,k=<int>,radial=<name>"
///
/// All three keys are REQUIRED, in any order, comma-separated, no
/// whitespace. `method == "mls"` alone, or any key missing, unknown, or
/// unparsable, is `E-MLS-OPTIONS` -- this file never substitutes a default
/// degree, neighbour count, or radial weight, because a channel author who
/// leaves one out has not stated what map they want and a plausible guess
/// would give them one anyway, silently. Supported `radial` names:
/// `wendland0`, `wendland2`, `wendland4`, `wendland6` (Wendland's compactly
/// supported functions of matching smoothness, §CRBF below; the same family
/// ArborX's own `CRBF::Wendland<N>` offers, so a cross-check test can name the
/// identical function on both sides).
///
/// ===========================================================================
/// WHAT "DEGREE" AND "K" MEAN GEOMETRICALLY
/// ===========================================================================
///
/// The polynomial basis is TOTAL DEGREE in the AMBIENT coordinate dimension
/// (1, 2 or 3 -- read off the point cloud's own coordinates via
/// `fuse::search::classify_dim`, exactly the convention `Point`'s own
/// contract states: an unused axis is EXACTLY zero). Its size is `P =
/// C(dim + degree, degree)` (`mls_basis_size`), e.g. degree 2 in 3D is 10
/// (`{1, x, y, z, x^2, xy, xz, y^2, yz, z^2}`). A "point cloud on a sphere" is
/// still handled in the AMBIENT dimension (3), not on the manifold: the basis
/// is ordinary 3D monomials evaluated at the point's `(x, y, z)`, restricted
/// to whatever surface the points happen to sit on -- the same choice
/// ArborX's own class makes (it knows only `Point`s of a fixed dimension, not
/// a manifold structure).
///
/// `k` must be at least `P`: fewer neighbours than the basis has degrees of
/// freedom makes the local fit underdetermined before any arithmetic runs.
/// `P` depends on the ambient dimension, which is a property of the
/// resolved geometry rather than of the method string, so this check runs in
/// `build_mls_weights` immediately after the source/target sides and their
/// shared dimension are resolved -- ONCE PER BUILD, before any per-target
/// neighbour search -- and is `E-MLS-NEIGHBORS`, not discovered later, target
/// by target, as a rank deficiency. A SOURCE side with fewer than `k` points
/// overall is the same refusal, at the same place.
///
/// ===========================================================================
/// REFUSALS
/// ===========================================================================
///
/// Errors, never throws (the same contract `build_weights` documents):
///
///   * `E-MLS-OPTIONS`     -- `method`'s degree/k/radial are missing, malformed,
///                            unknown, or duplicated.
///   * `E-MLS-REGION`      -- a region id, entity set, or mesh id `build_weights`
///                            would also refuse (mirrors `supermesh_side`'s
///                            checks; this file resolves its own regions rather
///                            than reaching into `intx_weights.cpp`'s anonymous
///                            namespace).
///   * `E-MLS-DISC`        -- a `(disc, order)` pair this builder has no point
///                            extraction for. Supported: `PointCloud` (any
///                            order, dimension 0), `ContinuousGLL` order 1
///                            (vertex, dimension 0, matching `field_extent`'s
///                            own nodal convention), `FV` and
///                            `DiscontinuousGLL` order 0 (cell centroid,
///                            the region's own dimension).
///   * `E-MLS-NEIGHBORS`   -- `k` below the declared degree's basis size
///                            (§ above), or a source side with fewer than `k`
///                            points, discovered before any per-target work.
///   * `E-MLS-RANK`        -- a target's local `P x P` moment-matrix solve is
///                            numerically rank-deficient (coincident or
///                            collinear/coplanar neighbours for the declared
///                            degree). The message carries the target's local
///                            index, the rank found versus `P`, and a
///                            condition-number ESTIMATE -- the ratio of the
///                            largest to the smallest diagonal entry of the
///                            column-pivoted factorisation
///                            `least_squares_builtin` leaves in its working
///                            array, which is a cheap, standard proxy for
///                            conditioning (not a true SVD condition number)
///                            and exactly the number available without a
///                            second factorisation. Two or more neighbours
///                            occupying the SAME position is exactly one way
///                            to trigger this (the case the task calls out by
///                            name) -- it degrades the moment matrix's rank
///                            like any other loss of geometric information,
///                            rather than dividing by a coincident distance:
///                            the radial weight is a compactly supported RBF
///                            evaluated at distance, finite (in fact maximal)
///                            at distance zero, so a source point sitting
///                            exactly on the target does not raise a
///                            division-by-zero either.
///   * `E-MLS-MONOTONE`    -- `RemapFlags::Monotone` was requested and the
///                            assembled matrix is not bound preserving (some
///                            weight `< 0`, or a row sum off 1 by more than
///                            `1e-12`) -- checked on the actual matrix, the
///                            same posture `RemapWeights::monotone`'s own doc
///                            and the TempestRemap builder's `E-REMAP-MONOTONE`
///                            take, under this file's own code because this
///                            file built the matrix.
///   * `E-MLS-UNSUPPORTED` -- `RemapFlags::Conservative` was requested. MLS
///                            fits a smooth local polynomial; it does not
///                            balance a mass budget, has no measure concept on
///                            a point cloud at all, and this file does not
///                            fabricate a defect number to pretend otherwise
///                            (see `RemapWeights::max_col_defect` note below).
///                            Also raised for a backend this file can detect
///                            is multi-rank (`NativeMeshBackend::num_ranks() >
///                            1`) -- see "SERIAL ONLY" below.
///
/// ===========================================================================
/// WHAT THE AUDIT FIELDS MEAN FOR THIS BUILDER
/// ===========================================================================
///
/// `RemapWeights::order` is `degree + 1` (the same "exact for degree N-1"
/// convention the header of `remap_plan.hpp` states: order 1 is exact for
/// constants, order 2 for linears). `max_row_defect` and `min_weight` are
/// MEASURED from the assembled matrix, because MLS's constant-reproduction
/// property (any degree >= 0 basis contains the constant monomial) makes row
/// sums come out at 1 to round-off as a consequence of the algorithm, not by
/// a normalisation step -- so recording the true residual is both honest and
/// cheap. `monotone` is likewise computed, never assumed: it is true only
/// when that measured pair happens to satisfy the bound-preserving condition,
/// which is generic for degree 0 (a normalised weighted average is always a
/// convex combination) and NOT generic for degree >= 1 (a polynomial fit
/// admits negative weights on the far neighbours, same as every other
/// builder's high-order path).
///
/// `max_col_defect`, `max_col_defect_native`, `src_measure_total` and
/// `tgt_measure_total` are the conservation (COLUMN) audit, and this builder
/// makes NO conservation claim at all: a point cloud need not define
/// `MeshView::measure` for its entities (dimension 0 "has no geometric
/// measure" by contract) and even where a measure exists (the FV/centroid
/// case), MLS was never asked to balance one. Leaving `max_col_defect` at its
/// struct default of `0.0` would be actively WRONG here, because
/// `WeightsOperator::claims()` reads `max_col_defect <= kClaimTolerance` to
/// decide `OperatorClaims::conservative` -- a bare default of `0.0` would
/// silently CLAIM perfect conservation for a map that was never checked for
/// it. So this builder sets both column-defect fields to
/// `+std::numeric_limits<double>::infinity()`: an explicit "not measured, and
/// not applicable" that reads as failing the claim gate, rather than a
/// fabricated number that would read as passing it. `RemapFlags::Conservative`
/// is refused outright (`E-MLS-UNSUPPORTED` above) for the same reason, before
/// any matrix is built.
///
/// ===========================================================================
/// SERIAL ONLY
/// ===========================================================================
///
/// This builder does not build or consume a remap halo: every target's k
/// nearest neighbours are searched for among the LOCAL source entities this
/// rank's `MeshView` reports, exactly the entities `entity_count`/`centroid`
/// already expose. On a genuinely partitioned read, a target near a partition
/// boundary could have its true nearest neighbours on another rank and this
/// builder would never see them -- silently returning a locally-correct but
/// globally-wrong answer, the same failure mode `build_remap_halo` exists to
/// close for the FV supermesh path. DISTRIBUTED MLS (a halo built from a
/// covering set of source points, sized to the largest declared `k` and
/// support radius) IS A FOLLOW-ON, not attempted here. Where this file CAN
/// detect the multi-rank case (the backend is a `NativeMeshBackend` reporting
/// `num_ranks() > 1`) it refuses `E-MLS-UNSUPPORTED` rather than returning a
/// silently partial map; where it cannot (an opaque `MeshBackend`), the
/// limitation is stated here and in `docs/user/point-cloud-remap.md`, not
/// enforced, and a caller running this method under a real multi-rank backend
/// this file does not recognise gets an answer computed only from what that
/// rank can see.

#include <cstddef>
#include <string>
#include <string_view>

#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/result.hpp"

namespace fuse {

/// The compactly supported radial basis function (CRBF) family this builder
/// evaluates the local weight with, restricted to the four Wendland orders
/// ArborX 2.1's own `CRBF::Wendland<N>` offers (`N` = 0, 2, 4, 6): higher `N`
/// is smoother (`C^N` at the support boundary) and flatter near the target,
/// at the cost of a wider effective stencil for the same neighbour count.
enum class MlsRadial { Wendland0, Wendland2, Wendland4, Wendland6 };

[[nodiscard]] const char* to_cstring(MlsRadial r) noexcept;

/// Evaluates the declared CRBF at `x = distance / support_radius`, clamped to
/// `[0, 1]` exactly as ArborX's own `CRBF::evaluate` does (`Kokkos::min
/// (Kokkos::abs(y), T(1))`) so a caller need not pre-clamp. Strictly positive
/// on `[0, 1)` and zero only at exactly `x == 1` -- this builder's own support
/// radius (`build_mls_weights`) is always scaled `1.1x` past the farthest
/// declared neighbour for exactly this reason, so no neighbour that
/// contributed to the search ever receives a zero weight.
[[nodiscard]] double mls_crbf(MlsRadial r, double x) noexcept;

/// `C(dim + degree, dim)`: the number of total-degree-`<=degree` monomials in
/// `dim` variables (1, 2 or 3), i.e. the size of the local polynomial basis
/// `build_mls_weights` fits. `dim` outside 1..3 or `degree < 0` returns 0,
/// total, never throws -- a caller uses it to validate before committing to a
/// build, exactly as `parse_mls_options`'s own `E-MLS-NEIGHBORS` check does.
[[nodiscard]] std::size_t mls_basis_size(int dim, int degree) noexcept;

/// The three channel-declared MLS options, parsed from `RemapPlan::method`.
/// Never default-constructed into a valid state by this builder: every field
/// here corresponds to a REQUIRED key in the method string (see this header's
/// own "OPTIONS ARE DECLARED" section), and `parse_mls_options` is the only
/// way to produce one that `build_mls_weights` will accept.
struct MlsOptions {
  int       degree = -1;              ///< total polynomial degree; -1 = undeclared.
  int       k = 0;                    ///< neighbour count; 0 = undeclared.
  MlsRadial radial = MlsRadial::Wendland2;
};

/// Parses `method` (the full `RemapPlan::method` string, e.g.
/// `"mls:degree=2,k=12,radial=wendland2"`) into its three required options.
///
/// Errors, never throws: `E-MLS-OPTIONS` for anything other than exactly the
/// three keys `degree`, `k`, `radial`, each present exactly once, `degree`
/// a non-negative integer, `k` a positive integer, and `radial` one of
/// `wendland0`/`wendland2`/`wendland4`/`wendland6`. `method == "mls"` (no
/// `:` at all) is the same refusal, naming the missing keys.
[[nodiscard]] Result<MlsOptions> parse_mls_options(std::string_view method);

/// Moving-least-squares `RemapWeights` for `plan` (dispatch predicate and
/// full contract in this header's own comment above). `graph` and `backend`
/// resolve `plan.src`/`plan.tgt` exactly as `build_weights` does; this
/// function does not require MOAB and does not touch it.
[[nodiscard]] Result<RemapWeights> build_mls_weights(const RemapPlan& plan,
                                                     const CouplingGraph& graph,
                                                     const MeshBackend& backend);

}  // namespace fuse
