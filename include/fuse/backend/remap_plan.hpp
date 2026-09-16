#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/spacetime.hpp"  // SpatialWeightsView

namespace fuse {

/// Spec §5.4. Conservative finite-volume remap weights built from a supermesh
/// (Farrell, Piggott, Pain, Gorman & Wilson, CMAME 198 (2009) 2632).
///
/// ===========================================================================
/// WHICH SUM IS WHICH. Read this before writing a conservation check.
/// ===========================================================================
///
/// `W` is stored row = TARGET entity, col = SOURCE entity, so `g = W f` takes a
/// source field to a target field. Let `m^s_i` be `MeshView::measure` of source
/// entity `i` and `m^t_j` that of target entity `j`. The two properties are:
///
///   ROW property -- CONSTANT PRESERVATION.
///       `sum_i W_ji == 1`   for every target row `j`
///   Equivalently: a constant field transfers to the same constant. It is a
///   plain, unweighted sum along a row.
///
///   COLUMN property -- CONSERVATION.
///       `sum_j W_ji * m^t_j == m^s_i`   for every source column `i`
///   Equivalently: `sum_j g_j m^t_j == sum_i f_i m^s_i` for every field `f`.
///   It is a MEASURE-WEIGHTED sum down a column.
///
/// The two are easy to state backwards, and a matrix that is wrong one way
/// still looks right on a uniform field, so both are pinned by separate tests
/// (`RemapRow.*` and `RemapColumn.*` in `test/unit/remap_test.cpp`) and both
/// residuals are reported in `RemapWeights::max_row_defect` and
/// `max_col_defect`. This is the same orientation `TemporalProjection` uses
/// (`fuse/backend/spacetime.hpp`): row sums unity, column sums carrying the
/// integral. Task 6.4 composes the two and needs them to agree.
///
/// ===========================================================================
/// THE TWO PROPERTIES ARE NOT ALWAYS SIMULTANEOUSLY ATTAINABLE
/// ===========================================================================
///
/// Suppose both hold exactly and let `f == c` be constant. The row property
/// gives `g == c`; the column property then gives `c * sum_j m^t_j == c * sum_i
/// m^s_i`. So
///
///     BOTH properties can hold only if the two meshes report the SAME TOTAL
///     MEASURE.
///
/// That is a real restriction, not a technicality. Two cubed-sphere meshes of
/// the *same* sphere are two different polyhedral surfaces: `MeshView::measure`
/// integrates the bilinear patch through the cell's corners, which is inscribed
/// and therefore short of the spherical area by O(h^2). Measured on the meshes
/// this project ships against (radius 7996.4553620):
///
///     MOAB's spherecube.h5m (6x24^2)   total measure 802651780.5417746
///     an equiangular 6x16^2 partner    total measure 801859336.3405802
///     relative difference                            9.88e-4
///
/// so a remap between them cannot be both conservative and constant-preserving
/// at better than about 1e-3. This library resolves that in favour of
/// CONSERVATION, because that is what `RemapFlags::Conservative` asks for, and
/// reports the constant-preservation residual in `max_row_defect` rather than
/// hiding it. On meshes whose measures ARE additive over the supermesh -- any
/// axis-aligned structured pair, and any pair of identical meshes -- both
/// properties hold to machine precision and `max_row_defect` is ~1e-15.
///
/// ===========================================================================
/// PARTIAL OVERLAP: TRUNCATE, DO NOT EXTRAPOLATE
/// ===========================================================================
///
/// When the two meshes do not cover the same domain the weights are still built
/// from the intersection, and nothing is renormalised to hide the shortfall:
///
///   * A TARGET cell the source domain covers only partly receives only the
///     mass that is actually there. Its row sum is the covered fraction, below
///     one, so a constant does NOT survive on it -- the value is diluted toward
///     zero. That is the only choice that keeps the map conservative, and it is
///     reported as `min_tgt_coverage`. It is not an error: no mass is lost.
///   * A SOURCE cell the target domain covers only partly has the mass in its
///     uncovered part DROPPED, not redistributed over the covered part. Its
///     column sum falls short of `m^s_i`, so the map is genuinely
///     non-conservative. Reported as `min_src_coverage`, and with
///     `RemapFlags::Conservative` set it is `E-REMAP-COVERAGE` -- the caller
///     asked for a conservative map and would not have got one.
///
/// The alternative -- rescaling each column by `m^s_i / (sum_j overlap)` so the
/// column property holds by construction -- was rejected deliberately: it makes
/// every matrix report perfect conservation, including the ones that quietly
/// pile a boundary cell's entire mass onto whatever fraction of it happens to
/// be covered.
struct RemapWeights {
  std::vector<std::int32_t> row;  ///< target entity, ascending
  std::vector<std::int32_t> col;  ///< source entity, ascending within a row
  std::vector<double>       val;
  std::size_t               n_rows = 0;  ///< target entities (local)
  std::size_t               n_cols = 0;  ///< source entities (local)

  // --- conservation audit, filled by build_weights ---------------------------
  //
  // Carried on the matrix so a caller -- the space-time conservation gate of
  // spec §7.4 above all -- can check what it was handed without rebuilding the
  // supermesh or re-deriving the measures.

  /// `sum_i m^s_i` and `sum_j m^t_j` over the entities this rank holds. Equal,
  /// to rounding, exactly when the two meshes discretise the same domain with
  /// the same total measure; see the impossibility argument above.
  double src_measure_total = 0.0;
  double tgt_measure_total = 0.0;

  /// Fraction of each entity the supermesh actually accounted for, minimised
  /// over entities. Exactly 1 (to ~1e-13) when the domains coincide; below 1
  /// under partial overlap. Computed in the SUPERMESH's own measure, so it is
  /// a pure coverage number and does not inherit the bilinear-versus-spherical
  /// discrepancy discussed above.
  double min_src_coverage = 1.0;
  double min_tgt_coverage = 1.0;

  /// `max_j |sum_i W_ji - 1|` -- the constant-preservation (ROW) residual.
  double max_row_defect = 0.0;
  /// `max_i |sum_j W_ji m^t_j - m^s_i| / m^s_i` -- the conservation (COLUMN)
  /// residual, in `MeshView::measure`.
  double max_col_defect = 0.0;

  // --- what the map actually is ----------------------------------------------
  //
  // Added in Task 5.3, when a second builder appeared. Two builders that both
  // return a `RemapWeights` do NOT return the same kind of object, and the
  // difference is not visible in the triplets. These fields say which one you
  // were handed, and are asserted rather than described: see
  // `RemapOrder.*` and `RemapMonotone.*` in test/unit/.

  /// The polynomial order of the reconstruction the weights were built from:
  /// 1 = piecewise constant (exact for constants), 2 = exact for linears, and
  /// so on. NOT a promise about the observed convergence rate -- that is
  /// measured, in `test/unit/remap_order_test.cpp`.
  int order = 1;

  /// True only when the map is bound preserving BY CONSTRUCTION and that has
  /// been checked on this matrix: every weight `>= 0` and every row sum equal
  /// to 1 to `1e-12`. Together those two are exactly the condition for
  /// `min_i f_i <= g_j <= max_i f_i` over each row's support, for every input
  /// `f`. A map with a row sum of `1 - 1e-3` (which is what a first-order
  /// supermesh map on a cubed-sphere pair has, for the measure reason above)
  /// is NOT bound preserving -- it dilutes a constant downwards -- and this
  /// stays false for it.
  bool monotone = false;

  /// `min_k val[k]`. Zero for an empty matrix. Non-negative exactly when the
  /// map cannot manufacture a new extremum from a single source cell; the
  /// high-order maps are negative here (measured: -0.049 at order 2 on a
  /// cubed-sphere pair) and that is not a defect, it is what buys the order.
  double min_weight = 0.0;

  /// The conservation (COLUMN) residual in the measure THE BUILDER ITSELF
  /// USED, as opposed to `max_col_defect`, which is always in
  /// `MeshView::measure`.
  ///
  /// These two are the same number for every first-order supermesh path: it
  /// assembles the weights from `MeshView::measure` directly. They are NOT the
  /// same for the TempestRemap path, which is conservative in TempestRemap's
  /// own exact spherical-polygon areas (great-circle edges). On a cubed sphere
  /// those differ from `MeshView::measure`'s bilinear-patch areas by O(h^2) --
  /// the same ~1e-3 discrepancy the header discusses above -- so a TempestRemap
  /// map typically reports `max_col_defect_native ~ 1e-15` and
  /// `max_col_defect ~ 1e-3`, and neither number is wrong.
  ///
  /// The weights are deliberately NOT rescaled to make `max_col_defect` small:
  /// the rescaling that would do it, `W_ji * (m^s_i / A^s_i) * (A^t_j / m^t_j)`,
  /// perturbs every weight by a smooth O(h^2) factor and therefore caps the
  /// observed order of accuracy at 2. That was measured, not assumed, and it
  /// is why orders 3 and 4 exist here at all.
  double max_col_defect_native = 0.0;

  [[nodiscard]] std::size_t nnz() const noexcept { return val.size(); }
};

/// Adapt to the non-owning view `apply_spacetime` takes. `spacetime.hpp`
/// promised this would be "one aggregate initialisation" once `RemapWeights`
/// landed; it is spelled once, here, because the field order of
/// `SpatialWeightsView` puts `nnz` between the pointers and the shape and a
/// hand-written brace list gets it wrong silently.
[[nodiscard]] inline SpatialWeightsView spatial_view(
    const RemapWeights& w) noexcept {
  return SpatialWeightsView{w.row.data(), w.col.data(), w.val.data(),
                            w.val.size(), w.n_rows, w.n_cols};
}

/// Spec §5.4: (source region, target region, discretization pair, method) ->
/// weights, hash-keyed so identical channels share one build.
///
/// `temporal` and `temporal_degree` are unused until Phase 6 Task 6.3. They are
/// declared now so the struct shape does not change later, and they take part
/// in `key()` and `operator==` so a plan that differs only in its temporal half
/// is a different cache entry.
struct RemapPlan {
  RegionId       src, tgt;
  Discretization src_disc = Discretization::FV;
  Discretization tgt_disc = Discretization::FV;
  int            src_order = 0;
  int            tgt_order = 0;
  RemapFlags     flags = RemapFlags::None;
  std::string    method;
  TemporalBasis  temporal = TemporalBasis::None;
  int            temporal_degree = 0;

  friend bool operator==(const RemapPlan&, const RemapPlan&) = default;

  /// The cache key. Every field of the struct appears, so two plans with the
  /// same key compare equal -- `RemapCache` relies on that, and a key that
  /// dropped a field would silently serve one plan's weights to another.
  ///
  /// `method` is length-prefixed rather than just concatenated: it is a free
  /// string, and without the prefix `method = "a/None0"` and `method = "a"`
  /// with a different temporal half produce the same key.
  [[nodiscard]] std::string key() const {
    return std::to_string(src.value()) + ">" + std::to_string(tgt.value()) +
           "/" + to_cstring(src_disc) + std::to_string(src_order) + ">" +
           to_cstring(tgt_disc) + std::to_string(tgt_order) + "/" +
           std::to_string(static_cast<std::uint32_t>(flags)) + "/" +
           std::to_string(method.size()) + ":" + method + "/" +
           to_cstring(temporal) + std::to_string(temporal_degree);
  }
};

/// Sparse matvec `dst = W * src`, applied independently to each component.
///
/// `src` is `w.n_cols * components` doubles and `dst` is `w.n_rows *
/// components`, both entity-major with the component index fastest -- the same
/// layout `apply_spacetime` uses. `dst` is overwritten, not accumulated.
///
/// DEVIATION FROM THE BRIEF, which had this return `void`. Every precondition
/// below is a memory-safety precondition on raw pointers, and an unchecked one
/// vanishes under Release's `-O3 -DNDEBUG`; `components <= 0` in particular
/// would make `n_rows * components` wrap when widened to `std::size_t` and the
/// zeroing loop would run for ~2^64 iterations. `apply_spacetime` deviated the
/// same way and for the same reason. Errors, never throws:
///
///   * `E-REMAP-ARG`  -- null `src`/`dst`, or `components < 1`.
///   * `E-REMAP-COO`  -- the triplet arrays disagree in length, or an index
///                       falls outside `[0, n_rows)` / `[0, n_cols)`.
///   * `E-REMAP-SIZE` -- the buffer length would overflow `std::size_t`.
[[nodiscard]] inline Result<void> apply_weights(const RemapWeights& w,
                                                const double* src, double* dst,
                                                int components) {
  if (components < 1) {
    return Failure{"E-REMAP-ARG", "apply_weights: components must be >= 1, not " +
                                    std::to_string(components)};
  }
  if (src == nullptr || dst == nullptr) {
    return Failure{"E-REMAP-ARG",
                 "apply_weights: null source or destination buffer"};
  }
  if (w.row.size() != w.val.size() || w.col.size() != w.val.size()) {
    return Failure{"E-REMAP-COO",
                 "apply_weights: the triplet arrays disagree in length (row " +
                     std::to_string(w.row.size()) + ", col " +
                     std::to_string(w.col.size()) + ", val " +
                     std::to_string(w.val.size()) + ")"};
  }
  const std::size_t nc = static_cast<std::size_t>(components);
  constexpr std::size_t kMax = static_cast<std::size_t>(-1);
  if (w.n_rows != 0 && nc > kMax / w.n_rows) {
    return Failure{"E-REMAP-SIZE",
                 "apply_weights: n_rows * components overflows std::size_t"};
  }
  if (w.n_cols != 0 && nc > kMax / w.n_cols) {
    return Failure{"E-REMAP-SIZE",
                 "apply_weights: n_cols * components overflows std::size_t"};
  }

  const std::size_t n = w.n_rows * nc;
  for (std::size_t i = 0; i < n; ++i) dst[i] = 0.0;

  for (std::size_t k = 0; k < w.val.size(); ++k) {
    // Checked, not asserted: a hand-built or deserialised RemapWeights with a
    // stale index would otherwise write outside `dst` in a Release build.
    if (w.row[k] < 0 || static_cast<std::size_t>(w.row[k]) >= w.n_rows ||
        w.col[k] < 0 || static_cast<std::size_t>(w.col[k]) >= w.n_cols) {
      return Failure{"E-REMAP-COO",
                   "apply_weights: triplet " + std::to_string(k) + " is (" +
                       std::to_string(w.row[k]) + ", " +
                       std::to_string(w.col[k]) + "), outside the " +
                       std::to_string(w.n_rows) + "x" +
                       std::to_string(w.n_cols) + " matrix"};
    }
    const std::size_t r = static_cast<std::size_t>(w.row[k]) * nc;
    const std::size_t c = static_cast<std::size_t>(w.col[k]) * nc;
    for (std::size_t comp = 0; comp < nc; ++comp) {
      dst[r + comp] += w.val[k] * src[c + comp];
    }
  }
  return Result<void>::success();
}

/// Conservative FV<->FV remap weights.
///
/// TWO BUILDERS BEHIND ONE ENTRY POINT. The plan chooses; the geometry vetoes.
///
///   * `method` = "" (the default), "intx" or "conservative"
///     -- the SUPERMESH builder of Task 5.2. First order (exact for constants),
///        conservative in `MeshView::measure`, weights non-negative.
///   * `method` = "tempest", or `method` = "" with an order above 1
///     -- the TEMPESTREMAP builder of Task 5.3. Orders 1 to 4, or a monotone
///        first-order map. Conservative in TempestRemap's own exact spherical
///        areas; see `RemapWeights::max_col_defect_native`.
///
/// The order the plan asks for is `max(src_order, tgt_order)`, or 2 when
/// `RemapFlags::HighOrder` is set with no explicit order, or 1 otherwise. For
/// FV->FV the reconstruction is a property of the SOURCE, so `src_order` and
/// `tgt_order` must agree when both are given; a plan that sets them
/// differently is `E-REMAP-UNSUPPORTED` rather than having one of them quietly
/// ignored.
///
/// WHICH PROPERTIES EACH CONFIGURATION ACTUALLY GUARANTEES. Measured residuals
/// are for a 6x24^2 against 6x16^2 cubed-sphere pair; see the task report and
/// `test/unit/remap_test.cpp` for the exact numbers, which are asserted.
///
///   supermesh, order 1        conservative in `MeshView::measure` (1e-16),
///     (axis-aligned boxes)    constant-preserving (1e-16), bound-preserving.
///                             All three, exactly, because the two meshes have
///                             the same total measure.
///   supermesh, order 1        conservative in `MeshView::measure` (1e-15),
///     (cubed spheres)         constant-preserving only to ~1e-3, NOT bound
///                             preserving. The two meshes do not report the
///                             same total measure -- see the impossibility
///                             argument at the top of this header -- and this
///                             builder resolves that in favour of conservation.
///   TempestRemap, order 1     conservative in TempestRemap's spherical areas
///     + `Monotone`            (1e-16), constant-preserving (1e-16), bound
///                             preserving. Conservation in `MeshView::measure`
///                             is only ~1e-3 -- the same discrepancy, moved.
///   TempestRemap, order >= 2  conservative in TempestRemap's spherical areas
///                             (1e-15), constant-preserving (1e-15), and NOT
///                             bound preserving: measured minimum weights
///                             -0.049 (order 2) and -0.075 (order 4), and a
///                             step field overshoots by 3.2e-2 at order 2.
///
/// So: order and bound preservation are not simultaneously available, and this
/// is not an implementation gap that a future release closes. Bound
/// preservation for every input is equivalent to "all weights >= 0 and all row
/// sums 1"; a reconstruction that is exact for linears needs the stencil to
/// reach past the immediate neighbours and the least-squares fit puts negative
/// entries there. TempestRemap resolves it the same way: setting `fMonotone`
/// makes it call its first-order kernel, whatever order you asked for. This
/// library refuses `Monotone` together with an order above 1 rather than
/// returning the first-order map under the high-order name.
///
/// INDEX SPACE. `row` and `col` are `MeshView` LOCAL INDICES at the target and
/// source region's dimension -- the same `i` you pass to `measure(dim, i)` --
/// so `n_rows == tgt_view->entity_count(dim)` and `n_cols ==
/// src_view->entity_count(dim)`. Both regions must therefore designate the
/// whole mesh at that dimension (`entity_set` empty or "all"); a sub-set region
/// is `E-REMAP-REGION`, because `MeshBackend` offers no way to say whether a
/// buffer for it is indexed by set position or by mesh-local index, and
/// guessing would produce a matrix that is wrong by a permutation.
///
/// GEOMETRY, SUPERMESH BUILDER. Three constructions, chosen from the meshes
/// rather than from the plan, each exact in a measure that agrees with
/// `MeshView::measure`:
///
///   1. AXIS-ALIGNED CELLS, any dimension. Every cell of both meshes is an
///      axis-aligned box (an interval in 1D, a rectangle in 2D, a right
///      hexahedron in 3D), verified cell by cell -- not assumed. Overlaps are
///      products of interval overlaps, and `measure` of such a cell is exactly
///      that product, so both properties hold to rounding.
///   2. SURFACE ON A SPHERE, dimension 2. Every vertex of both meshes lies at
///      a common radius about the origin. `moab::Intx2MeshOnSphere`.
///   3. SURFACE IN A PLANE, dimension 2. Every vertex of both meshes lies in a
///      common plane `z = const`; `moab::Intx2MeshInPlane` works in the x-y
///      projection, so no other plane orientation is accepted.
///
/// General unstructured 3D supermeshing is NOT attempted, and neither is a
/// curved surface that is neither a sphere nor a plane: both are
/// `E-REMAP-UNSUPPORTED`.
///
/// GEOMETRY, TEMPESTREMAP BUILDER. Case 2 only -- a dimension-2 surface whose
/// vertices all lie at a common radius about the origin. TempestRemap is a
/// spherical remapper: its edges are great-circle arcs, its areas are spherical
/// excesses, and its reconstruction stencils are built in gnomonic coordinates.
/// A planar surface or a volume mesh is `E-REMAP-UNSUPPORTED` from this builder,
/// and the message says so in those terms: high-order remap on a plane or in a
/// volume is UNIMPLEMENTED here, not impossible. Nothing in `RemapPlan`,
/// `RemapWeights` or `RemapFlags` is spherical, and the internal backend
/// interface (`src/remap/high_order_backend.hpp`) takes a polygon-soup surface
/// plus a geometry classification, so a planar or volumetric high-order backend
/// can be added behind the same seam.
///
/// SIDE EFFECTS ON THE MOAB INSTANCE, stated because they are surprising. The
/// SUPERMESH builder's sphere and plane constructions call into MOAB's
/// intersection machinery, which requires positively oriented cells and
/// identifies parents through the `GLOBAL_ID` tag. `build_weights` therefore
/// may reverse a cell's connectivity and does temporarily rewrite `GLOBAL_ID`
/// on both meshes' cells, restoring the original values before returning.
/// Neither is visible through `MeshView`: the measure is unsigned, the centroid
/// is order-independent, and the global ids the view reports were cached at
/// load. The intersection polygons are deleted afterwards; the vertices MOAB
/// created for them are not, so a build leaves a bounded amount of
/// unreferenced geometry behind in the `moab::Core`. Use `RemapCache` and it
/// happens once per plan.
///
/// The TEMPESTREMAP builder has NO side effects on the `moab::Core` at all: it
/// copies the connectivity and coordinates out into its own polygon soup,
/// orients that copy, and works on it. It does, however, briefly redirect
/// TempestRemap's global `Announce` output buffer (a process-wide `FILE*`) so
/// that the diagnostics can be captured -- TempestRemap catches its own
/// exceptions and returns success, so the captured text is the ONLY signal that
/// a build failed or silently dropped to first order. That makes this path,
/// and the remap path generally, serial and non-reentrant -- and, as of R10,
/// that is now ENFORCED rather than merely documented: a process-wide
/// `std::mutex` in `tempest_weights.cpp` (`announce_mutex()`) is held for the
/// entire span from before the capturing object is constructed to after it is
/// destroyed and its text consumed, so two threads calling into the high-order
/// builder concurrently now SERIALIZE rather than silently corrupt each
/// other's diagnostics (one thread's captured text being stolen or clobbered
/// by another's, which is worse than a crash because printing is this
/// backend's only failure channel). The mutex does not make TempestRemap
/// itself reentrant -- only one build may be in flight process-wide at a
/// time, exactly as before -- it only makes the existing serial constraint
/// safe under concurrent use instead of merely being a hazard callers had to
/// avoid on their own. Nothing here uses `Intx2Mesh`'s parallel mode either,
/// so the supermesh builder was never a counterexample to the "serial" half
/// of this claim.
///
/// PARTITIONED READS need a halo, and it is not built implicitly.
///
/// The intersection runs over the cells the calling rank HOLDS. On a
/// partitioned read that is correct exactly when the source mesh carries a
/// remap halo -- the off-rank source cells overlapping this rank's target
/// cells, adopted as ordinary non-owned local entities by `build_remap_halo`
/// above. Without one, every overlap crossing a partition boundary is dropped:
/// measured on a deliberately misaligned pair, 66.7% of the transferred
/// quantity at two ranks and 91.2% at four, returned as SUCCESS unless
/// `RemapFlags::Conservative` happens to be set. So `build_weights` refuses a
/// partitioned pair for which no halo was built (`E-REMAP-PARALLEL`), and with
/// one it conserves to 1.2e-15 at two ranks and 7.4e-16 at four. Both halves
/// are pinned by `test/unit/remap_parallel_test.cpp` (the refusal) and
/// `test/unit/remap_halo_parallel_test.cpp` (the halo and the conservation).
///
/// The refusal is keyed on the communicator having MORE THAN ONE RANK, not on
/// a communicator being present: a one-rank communicator is an ordinary MPI
/// job whose mesh happens to be undivided, every cell is local, and the serial
/// construction is exactly right. The one-rank leg of the refusal test is a
/// POSITIVE control for that.
///
/// WHAT IS AND IS NOT COVERED IN PARALLEL:
///
///   * The FIRST-ORDER conservative supermesh builder: supported and
///     conservation-tested at 1, 2 and 4 ranks.
///   * The HIGH-ORDER builder: refused on a partitioned read. Not a discovered
///     impossibility -- with a halo the source surface it is handed is a piece
///     of the domain plus a fringe rather than a closed one, and its
///     reconstruction stencils near that fringe are unmeasured. Refused until
///     someone measures them.
///   * NON-SPHERICAL surfaces: the covering-set construction available here is
///     spherical only, so a partitioned planar or volume pair is
///     `E-MESH-HALO-GEOMETRY`, naming that axis. A planar covering
///     construction would slot in behind `build_remap_halo` and nothing above
///     it would change.
///   * The RUNTIME (`Runner`): does not yet run the halo exchange around the
///     `Remap` operator, and refuses (`E-RT-REMAP-HALO`) rather than applying
///     a matrix to unfilled halo slots. The library-level sequence
///     `build_remap_halo` -> `build_weights` -> `exchange_remap_halo` ->
///     `apply_weights` is complete.
///
/// THE AUDIT BELOW IS REDUCED OVER THE COMMUNICATOR. `min_src_coverage`,
/// `max_col_defect` and friends are global statements: a source cell
/// straddling a target partition boundary is covered partly here and partly
/// elsewhere, so a per-rank figure would report a correct map as a coverage
/// failure -- and, worse, would report it on some ranks only, so
/// `E-REMAP-COVERAGE` would fire unevenly and the job would hang in the next
/// collective instead of failing. Every rank gets the same numbers and the
/// same verdict.
///
/// Errors, never throws. `E-REMAP-NOMOAB`, `E-REMAP-NOTEMPEST`,
/// `E-REMAP-BACKEND`, `E-REMAP-PARALLEL`, `E-REMAP-REGION`, `E-REMAP-DIM`,
/// `E-REMAP-UNSUPPORTED`, `E-REMAP-GEOMETRY`, `E-REMAP-MEASURE`,
/// `E-REMAP-GID`, `E-REMAP-INTX`, `E-REMAP-COVERAGE`, `E-REMAP-SIZE`,
/// `E-REMAP-MONOTONE`, `E-REMAP-DEGRADED`.
///
///   * `E-REMAP-PARALLEL` -- the mesh backend's communicator holds more than
///     one rank and no remap halo has been built for this source/target pair
///     (call `build_remap_halo` first), or the plan asks for the high-order
///     builder, which is unverified on a partitioned read. See above.
///   * `E-REMAP-NOTEMPEST` -- the plan needs the TempestRemap builder and this
///     build's MOAB was configured without TempestRemap.
///   * `E-REMAP-MONOTONE`  -- `RemapFlags::Monotone` was asked for and the
///     matrix that came out is not bound preserving. Not a refusal to try: the
///     map was built and then checked, and the check failed.
///   * `E-REMAP-DEGRADED`  -- TempestRemap built a map of LOWER order than the
///     one requested (its least-squares fit was too ill-conditioned on this
///     mesh pair and it fell back to first order). It reports this only in its
///     `Announce` stream, and returns success; this build reads that stream and
///     refuses rather than handing back a first-order map labelled order N.
[[nodiscard]] Result<RemapWeights> build_weights(const RemapPlan& plan,
                                                 const CouplingGraph& graph,
                                                 const MeshBackend& backend);

/// COLLECTIVE. Give `plan`'s SOURCE mesh the halo it needs for `plan` to be
/// computable on a partitioned read, and return how many cells were adopted.
///
/// WHY THIS EXISTS AND WHY IT IS SEPARATE FROM `build_weights`. On a
/// partitioned read the source cells overlapping this rank's target cells are
/// spread over every rank. `MeshView`'s local entity set is already "owned AND
/// ghost", and `entity_count` is already "the size a per-entity field buffer
/// must have", so the halo needs no new vocabulary: an off-rank source cell
/// becomes an ordinary non-owned local entity, `region_size` widens with it,
/// and `RemapWeights` addresses it with an ordinary column index. Nothing in
/// `RemapPlan`, `RemapWeights`, `RemapFlags` or `apply_weights` changes.
///
/// It is a SEPARATE CALL because it changes `entity_count(dim)`, and therefore
/// every buffer sized from it. It must run after `load` and BEFORE
/// `make_region_sizes`, `plan_buffers`, `schedule()`, or a component
/// registering storage for a field on the source mesh. `build_weights` will
/// not do it implicitly: by the time the runtime calls that, the buffer plan
/// is already fixed, and silently widening the matrix under a plan that was
/// sized without the halo is exactly the mismatch `E-RT-REMAP-SIZE` exists to
/// catch.
///
/// Zero adopted cells is a legitimate result -- one rank, or two partitions
/// that already align -- and is NOT the same as "not built". `build_weights`
/// keys `E-REMAP-PARALLEL` on whether this ran, not on the count.
///
/// Idempotent per (source mesh, target mesh, dimension). Serial, or a
/// communicator of one rank, is a successful no-op, so a caller may make the
/// call unconditionally.
///
/// GEOMETRY. The covering-set construction available here handles SPHERICAL
/// surfaces only, so a partitioned pair that is not one is
/// `E-MESH-HALO-GEOMETRY`, naming that axis. That is a limit of the backend
/// doing the construction, not of the idea: a planar or volumetric covering
/// construction would slot in behind this same call and nothing above it would
/// change.
///
/// COST. One covering-set construction plus one allgather of the owned global
/// ids, which is O(total cells) per rank -- the same ceiling
/// `MpiTransport::build_plan` has. Setup only, and stated rather than
/// discovered.
///
/// Errors, never throws: everything `build_weights` can return while resolving
/// the two regions, plus `E-MESH-HALO*` from the mesh layer.
[[nodiscard]] Result<std::size_t> build_remap_halo(const RemapPlan& plan,
                                                   const CouplingGraph& graph,
                                                   MeshBackend& backend);

/// Spec §5.4: weights are built once at setup, shared by identical channels.
///
/// The returned pointer is stable for the life of the cache -- entries are held
/// through `unique_ptr`, so rehashing the map does not move the weights -- and
/// is owned by the cache. A failed build is not cached and counts as a miss.
class RemapCache {
 public:
  [[nodiscard]] Result<const RemapWeights*> get_or_build(
      const RemapPlan& plan, const CouplingGraph& graph,
      const MeshBackend& backend);

  [[nodiscard]] std::size_t size() const noexcept { return cache_.size(); }
  [[nodiscard]] std::size_t hits() const noexcept { return hits_; }
  [[nodiscard]] std::size_t misses() const noexcept { return misses_; }

 private:
  std::unordered_map<std::string, std::unique_ptr<RemapWeights>> cache_;
  std::size_t hits_ = 0;
  std::size_t misses_ = 0;
};

}  // namespace fuse
