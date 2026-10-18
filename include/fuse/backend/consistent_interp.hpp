#pragma once

/// Consistent (piecewise-linear) interpolation between non-matching 1D
/// interface meshes -- the H operator the FSI phase needs -- and its exact
/// transpose used as a conservative load transfer.
///
/// WHY THIS EXISTS. `supermesh.cpp` gives a P0 (piecewise-constant, cell
/// average) map: exact for constants, conservative, but only first order for
/// a genuinely varying field (a linear field is NOT reproduced pointwise at
/// fixed non-matching resolution -- `test/verification/interface_transfer_
/// test.cpp`'s file header measured the error directly). A strongly-coupled
/// FSI interface needs two DIFFERENT things from its two directions:
///
///   * the fluid wall needs the beam's velocity AT A POINT (a pointwise,
///     intensive quantity) with second-order accuracy -- that is `H` below;
///   * the beam needs the fluid's traction TOTALLED into a nodal load (an
///     extensive, conservative quantity) -- that is `H`'s transpose, used
///     directly, so the two are exact adjoints of one specific matrix rather
///     than two matrices that happen to agree in the constant case.
///
/// WHERE THE DOFS LIVE. `H` interpolates a field whose DOFs are the SOURCE
/// mesh's VERTICES (`MeshView` dimension 0) -- a beam's velocity/displacement
/// is naturally nodal -- to the TARGET mesh's evaluation points, which are
/// either its own vertices (dimension 0, node-to-node) or its cell centroids
/// (dimension `tgt_dim`, e.g. a P0 fluid wall's face midpoints). Both meshes
/// are treated as ONE-DIMENSIONAL CURVES embedded in R^3 with the embedding
/// axis taken from the source's own vertex spread (see `build_consistent_
/// interp_weights` below) -- there is no unstructured connectivity query on
/// `MeshView`, so the nodes are ordered by that coordinate, which is exactly
/// right for a 1D interface and wrong for anything with more than one
/// tangential direction. This header does not attempt the 2D (surface)
/// case; see `open_questions` in the task report that introduced this file.
///
/// THE MATRIX. For `n` source nodes sorted by coordinate `x_0 < x_1 < ... <
/// x_{n-1}` and a target point `x`, `H` has AT MOST TWO nonzeros per row: the
/// two nodes bracketing `x`, weighted by the standard linear (hat-function)
/// shape functions, so `sum_i H_ji == 1` for every row `j` EXACTLY (a
/// rounding residual only) -- the partition-of-unity property that makes
/// this operator exact for an affine field and, via the identity below,
/// makes its transpose conserve a total load exactly rather than
/// approximately.
///
/// ROW SUM 1 IS WHAT CONSERVATION RESTS ON, NOT A SEPARATE CONVENTION.
/// `sum_i (H^T f)_i = sum_j f_j * (sum_i H_ji) = sum_j f_j * 1 = sum_j f_j`
/// for ANY load vector `f`, exactly, as long as every row of `H` sums to
/// one -- which requires every target point to fall within the source
/// nodes' span; see `InterfaceDomainPolicy` for what happens when it does
/// not. This is why `H`'s claim is `preserves_constants` (a statement about
/// its OWN rows), never `conservative` in `OperatorClaims`' sense (that
/// axis is about column sums weighted by a TARGET measure, and a vertex
/// carries no measure to weight by -- `MeshView::measure(0, i)` throws by
/// contract). The physically conservative use of this matrix is via
/// `apply_transpose`, one call away on the exact same object.
///
/// DENSITY VS LOAD -- READ THIS BEFORE WIRING A FORCE FIELD TRANSFER. FUSE's
/// `Conservation::Extensive` convention (`remap_plan.hpp`'s "WHICH SUM IS
/// WHICH") stores a DENSITY: a cell's value integrates to an amount only
/// once multiplied by that cell's measure. `H^T` needs a LOAD (an already-
/// integrated per-entity amount) on its input, because a vertex has no
/// measure of its own to divide a delivered load by on the way out. So the
/// weights `build_consistent_transpose_weights_for_plan` returns for the
/// "consistent-transpose" method are not a bare transpose of `H`: each
/// column (one per source-of-the-field-transfer, i.e. the density side) is
/// pre-multiplied by THAT side's own cell measure, converting density to
/// load before the H^T contraction runs. Concretely, if `D` is the diagonal
/// of the point side's cell measures, the matrix returned is `(D H)^T =
/// H^T D`, applied directly to the density array the field transfer carries -- so
/// the caller never has to perform the density-to-load conversion itself,
/// and the total delivered is exactly the total sent (see the file's
/// mutation and test coverage for the identity `sum((H^T D) rho) ==
/// sum(D rho)`, which follows from `H`'s row-sum-one property alone).
///
/// DISCRETE POWER BALANCE. With `H` used forward for a pointwise velocity
/// `v` (beam -> fluid) and `H^T D` used forward for a fluid density `rho`
/// (fluid -> beam, `D` its cell measures), the two field transfers' delivered
/// quantities satisfy `(H v)^T (D rho) == v^T (H^T D rho)` EXACTLY -- not
/// approximately -- because the left- and right-hand sides are literally
/// the same bilinear form `v^T H^T D rho` associated differently. This is
/// the discrete statement that interface power balances: the work the
/// fluid does on the interface, evaluated at the fluid's own points, equals
/// the work delivered to the beam's nodes.
///
/// SELECTION. `RemapPlan::method == "consistent"` selects `H` (the SOURCE
/// field must be `Discretization::ContinuousGLL` order <= 1, i.e. genuinely
/// nodal -- `connect.hpp`'s `to_semantics` already maps that combination to
/// `SemanticLocation::Vertex`). `method == "consistent-transpose"` selects
/// `H^T D` (the TARGET field must be that same nodal discretization; the
/// SOURCE field may be `Discretization::FV`, density-convention, in which
/// case its own mesh entity set's cell measure supplies `D`, or may itself be nodal
/// order <= 1, in which case the caller's values are already a load and no
/// scaling is applied). Both dispatch out of `build_weights()` in
/// `src/backend/intx_weights.cpp` before its FV<->FV-only check, so neither
/// needs MOAB.
///
/// DOMAIN EDGES. The two interface meshes are not required to span the same
/// physical extent. See `InterfaceDomainPolicy`.

// HEADER-ONLY BY DESIGN, NOT BY DEFAULT. Every other file in this directory
// pairs a declaration here with a definition in the matching src/backend/
// *.cpp, registered in src/CMakeLists.txt's explicit `fuse-coupler-backend`
// source list. That list is a shared build file this task was not given
// ownership of (see the task's `open_questions`), so these functions are
// `inline` and defined below instead of in a new .cpp this task cannot wire
// into the library target. They are called only at plan-build time (once per
// distinct field transfer, cached by `RemapCache`), never in a per-entity hot loop,
// so the cost of a header-only definition is init-time only.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "fuse/backend/general_interp.hpp"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/ids.hpp"
#include "fuse/base/result.hpp"

namespace fuse {

// `InterfaceDomainPolicy` (what to do when a target point falls outside the
// source's coverage) is defined in general_interp.hpp now that TWO builders
// share it -- this file's own 1D curve algorithm below, and
// `build_general_interp_weights` (any 2D/3D cell, including a surface). The
// enum and its meaning are unchanged; only its physical location moved so
// general_interp.hpp does not have to include this file to name it.

/// Builds `H` for a source mesh whose topological dimension is 1 (a beam, a
/// wire, any 1D interface curve): row = target evaluation point (index into
/// `tgt_dim`), column = source vertex (index into `src_nodes` at dimension
/// 0), value = the piecewise-linear hat-function weight.
///
/// GEOMETRY. Both views are treated as 1D curves. The embedding axis is
/// chosen as whichever of x, y, z has the largest coordinate spread over
/// `src_nodes`' vertices (so a beam laid out along y or z, not just x, is
/// handled without the caller having to know which); `tgt_points` is
/// projected onto that same axis. Nodes are then ordered by that one
/// coordinate -- `MeshView` gives no connectivity query, and for a genuine
/// 1D interface, coordinate order and mesh order coincide.
///
/// REJECTS, never throws:
///   * `E-INTERP-NOGEOMETRY`  -- `src_nodes` has no dimension-0 entities, or
///     `tgt_points` has none at `tgt_dim`.
///   * `E-INTERP-DEGENERATE`  -- two source vertices coincide (within
///     rounding) along the chosen axis, so the piecewise-linear
///     reconstruction is singular there.
///   * `E-INTERP-DOMAIN`      -- (policy `Refuse`, the default) a target
///     point lies outside the source vertices' span along the chosen axis.
[[nodiscard]] inline Result<RemapWeights> build_consistent_interp_weights_1d(
    const MeshView& src_nodes, const MeshView& tgt_points, int tgt_dim,
    InterfaceDomainPolicy policy = InterfaceDomainPolicy::Refuse) {
  const std::size_t n_src = src_nodes.entity_count(0);
  if (n_src == 0) {
    return Failure{"E-INTERP-NOGEOMETRY",
                   "build_consistent_interp_weights: the source view reports "
                   "no dimension-0 (vertex) entities, so there are no nodal "
                   "DOFs to interpolate from"};
  }
  const std::size_t n_tgt = tgt_points.entity_count(tgt_dim);
  if (n_tgt == 0) {
    return Failure{"E-INTERP-NOGEOMETRY",
                   fmt::format("build_consistent_interp_weights: the target "
                               "view reports no dimension-{} entities to "
                               "evaluate the interpolant at",
                               tgt_dim)};
  }

  // One source vertex, reduced to the single coordinate the interface is
  // ordered by.
  struct AxisNode {
    double      x = 0.0;
    std::size_t idx = 0;  // local vertex index into src_nodes
  };
  const auto coord = [](const Point& p, int axis) {
    return axis == 0 ? p.x : (axis == 1 ? p.y : p.z);
  };

  // ---- 1. Order the source vertices along the interface's own axis --------
  std::vector<Point> src_pts(n_src);
  for (std::size_t i = 0; i < n_src; ++i) src_pts[i] = src_nodes.centroid(0, i);

  // Which of x/y/z has the largest spread over the source vertices: the
  // interface's own embedding axis, so a beam laid out along y or z is
  // handled without the caller having to know which.
  int axis = 0;
  {
    double lo[3], hi[3];
    for (int a = 0; a < 3; ++a) lo[a] = hi[a] = coord(src_pts.front(), a);
    for (const Point& p : src_pts) {
      for (int a = 0; a < 3; ++a) {
        lo[a] = std::min(lo[a], coord(p, a));
        hi[a] = std::max(hi[a], coord(p, a));
      }
    }
    double best_spread = hi[0] - lo[0];
    for (int a = 1; a < 3; ++a) {
      const double spread = hi[a] - lo[a];
      if (spread > best_spread) { axis = a; best_spread = spread; }
    }
  }

  std::vector<AxisNode> nodes(n_src);
  for (std::size_t i = 0; i < n_src; ++i) {
    nodes[i] = AxisNode{coord(src_pts[i], axis), i};
  }
  std::sort(nodes.begin(), nodes.end(),
            [](const AxisNode& a, const AxisNode& b) { return a.x < b.x; });

  const double span = nodes.back().x - nodes.front().x;
  // A tolerance scaled by the node spacing, not an absolute epsilon -- the
  // interface may be metres or millimetres. The degenerate (single-node, or
  // all-nodes-coincident) case falls back to an absolute floor so the
  // degeneracy check below still means something.
  const double tol = std::max(span, 1.0) * 1e-9;
  for (std::size_t k = 1; k < nodes.size(); ++k) {
    if (nodes[k].x - nodes[k - 1].x <= tol) {
      return Failure{
          "E-INTERP-DEGENERATE",
          fmt::format("build_consistent_interp_weights: source vertices {} "
                      "and {} coincide along the interface axis (x = {:.6g} "
                      "and {:.6g}, tolerance {:.3g}); the piecewise-linear "
                      "reconstruction is singular there",
                      nodes[k - 1].idx, nodes[k].idx, nodes[k - 1].x,
                      nodes[k].x, tol)};
    }
  }

  // ---- 2. One or two nonzeros per target row -------------------------------
  RemapWeights w;
  w.n_rows = n_tgt;
  w.n_cols = n_src;
  w.order = 2;  // exact for an affine field; see the header doc above.
  w.row.reserve(n_tgt * 2);
  w.col.reserve(n_tgt * 2);
  w.val.reserve(n_tgt * 2);

  std::vector<double> row_sum(n_tgt, 0.0);
  std::vector<double> col_sum(n_src, 0.0);
  double min_weight = 1.0;

  for (std::size_t j = 0; j < n_tgt; ++j) {
    const Point p = tgt_points.centroid(tgt_dim, j);
    const double x = coord(p, axis);

    // Locate the bracketing pair by the first node whose coordinate is not
    // less than x. std::lower_bound needs a Point-vs-double comparator.
    const auto it =
        std::lower_bound(nodes.begin(), nodes.end(), x,
                          [](const AxisNode& n, double v) { return n.x < v; });

    std::size_t col_a, col_b;
    double val_a, val_b;
    if (n_src == 1) {
      col_a = col_b = nodes[0].idx;
      val_a = 1.0;
      val_b = 0.0;
    } else if (it == nodes.begin() && x < nodes.front().x - tol) {
      // Below the span.
      if (policy == InterfaceDomainPolicy::Refuse) {
        return Failure{
            "E-INTERP-DOMAIN",
            fmt::format(
                "build_consistent_interp_weights: target point {} (x = "
                "{:.6g}) lies below the source vertices' span [{:.6g}, "
                "{:.6g}]; refusing rather than extrapolating a linear "
                "reconstruction past the data that supports it",
                j, x, nodes.front().x, nodes.back().x)};
      }
      col_a = col_b = nodes.front().idx;
      val_a = 1.0;
      val_b = 0.0;
    } else if (it == nodes.end()) {
      // lower_bound found no node with x_node >= x: x is greater than every
      // node, including the last. That includes the "just barely past the
      // last node, within rounding" case (lower_bound compares against x
      // exactly, not against x - tol), so a hit within `tol` of the last
      // node is treated as landing exactly ON it, same as the below-span
      // branch treats a near-front hit -- neither is a genuine extrapolation.
      if (x <= nodes.back().x + tol) {
        col_a = col_b = nodes.back().idx;
        val_a = 1.0;
        val_b = 0.0;
      } else if (policy == InterfaceDomainPolicy::Refuse) {
        return Failure{
            "E-INTERP-DOMAIN",
            fmt::format(
                "build_consistent_interp_weights: target point {} (x = "
                "{:.6g}) lies above the source vertices' span [{:.6g}, "
                "{:.6g}]; refusing rather than extrapolating a linear "
                "reconstruction past the data that supports it",
                j, x, nodes.front().x, nodes.back().x)};
      } else {
        col_a = col_b = nodes.back().idx;
        val_a = 1.0;
        val_b = 0.0;
      }
    } else if (std::abs(it->x - x) <= tol) {
      // Exact hit (within rounding) on a source vertex.
      col_a = col_b = it->idx;
      val_a = 1.0;
      val_b = 0.0;
    } else {
      // Genuine bracket: `it` is strictly greater than x (the equal and
      // below-span cases were handled above), so the pair is (it-1, it).
      //
      // KNOWN_ISSUES.md D67: this used to compute `t = (x - lo.x) /
      // (hi.x - lo.x)` directly, a DIFFERENT floating-point expression
      // from the one `build_distributed_interp_rows`/`build_general_
      // interp_weights` use for the SAME Edge-cell hat weight
      // (`general_interp_detail::invert_reference_map`'s Gauss-Newton
      // step) -- algebraically identical, not bit-identical, so the two
      // code paths could (and, measured on a real 7-cell/11-cell pair,
      // did) disagree by 1 ULP at several bracketing vertices. Routing
      // this bracket through the SAME shared predicate
      // (`general_interp_detail::edge_hat_weights`) both code paths now
      // call closes that gap: whichever path a caller's placement
      // selects computes the identical bits. `lo.x != hi.x` is already
      // guaranteed by the coincident-vertex rejection above (tolerance
      // `tol`), so the synthetic Edge cell this delegates to can never
      // hit a singular Jacobian; the `Result` is still checked rather
      // than `.value()`-d, matching this file's own "rejects, never
      // throws" contract.
      const AxisNode& lo = *(it - 1);
      const AxisNode& hi = *it;
      const auto hat = general_interp_detail::edge_hat_weights(lo.x, hi.x, x);
      if (!hat.ok()) {
        return Failure{hat.error().code,
                       fmt::format("build_consistent_interp_weights: target point {} (x = "
                                   "{:.6g}), bracket [{:.6g}, {:.6g}]: {}",
                                   j, x, lo.x, hi.x, hat.error().message)};
      }
      col_a = lo.idx;
      col_b = hi.idx;
      val_a = hat.value().first;
      val_b = hat.value().second;
    }

    w.row.push_back(static_cast<std::int32_t>(j));
    w.col.push_back(static_cast<std::int32_t>(col_a));
    w.val.push_back(val_a);
    row_sum[j] += val_a;
    col_sum[col_a] += val_a;
    min_weight = std::min(min_weight, val_a);
    if (col_b != col_a || val_b != 0.0) {
      w.row.push_back(static_cast<std::int32_t>(j));
      w.col.push_back(static_cast<std::int32_t>(col_b));
      w.val.push_back(val_b);
      row_sum[j] += val_b;
      col_sum[col_b] += val_b;
      min_weight = std::min(min_weight, val_b);
    }
  }

  for (const double s : row_sum) {
    w.max_row_defect = std::max(w.max_row_defect, std::abs(s - 1.0));
  }
  // NOT the RemapWeights-standard (measure-weighted) conservation residual:
  // a vertex has no measure to weight by (`MeshView::measure(0, i)` throws),
  // so there is nothing to divide by on this axis. This is the plain,
  // unweighted column-sum residual instead -- informative for diagnostics,
  // but deliberately NOT what makes `WeightsOperator::claims().conservative`
  // true, because it isn't the property that name promises. `H` is not, and
  // does not claim to be, conservative in that sense; see the header for
  // which property it actually has (row-sum-one) and what that buys.
  for (const double s : col_sum) {
    w.max_col_defect = std::max(w.max_col_defect, std::abs(s - 1.0));
  }
  // REVIEW.md TF-16: say so structurally, not just in the comment above --
  // `max_col_defect` here is the unweighted column-sum residual, not a
  // conservation residual, so `WeightsOperator::claims()` must not read it
  // as one (a coincident node-to-node pair's column sum is exactly 1 by
  // accident of geometry, which would otherwise certify `conservative`).
  w.col_defect_is_measure_weighted = false;
  w.min_weight = min_weight;
  // Bound preserving: every weight in [0, 1] (by construction: a convex
  // combination of two bracketing nodes, or a single weight of exactly 1) and
  // every row sums to 1. True regardless of `policy`, including under
  // ClampToBoundary -- a clamped row is still a single weight of 1.
  w.monotone = (w.min_weight >= -1e-14) && (w.max_row_defect <= 1e-10);
  w.src_measure_total = 0.0;  // undefined for a vertex set; left at the default.
  w.tgt_measure_total = 0.0;
  w.min_src_coverage = 1.0;
  w.min_tgt_coverage = 1.0;
  return w;
}

/// Builds `H` for a source of ANY supported cell shape -- the public entry
/// point every caller (this file's own `build_consistent_weights_for_plan`,
/// and any test that calls it directly the way
/// `test/backend/operator_conformance_test.cpp`'s `ConsistentInterp.*` cases
/// do) should use.
///
/// DISPATCH, NOT A REWRITE: `src_nodes.dimension() == 1` (a 1D interface
/// curve -- a beam, a wire) routes to `build_consistent_interp_weights_1d`
/// above, UNCHANGED, so every existing 1D test (including the hand-built
/// `ArbitraryLine`/`PartitionedTestMesh`-style views that carry no cell
/// connectivity at all, only a coordinate-ordered vertex list) keeps its
/// exact prior behaviour byte for byte. `dimension() == 2` or `3` (a
/// triangle/quad/sphere-surface or tet/hex/prism source) routes to
/// `build_general_interp_weights` (general_interp.hpp), which needs
/// `MeshView::cell_vertices` and rejects by name
/// (`E-MESH-NO-CELL-VERTICES`) when the concrete backend does not implement
/// it.
///
/// The two paths are kept separate rather than folded into one
/// connectivity-based algorithm because the 1D algorithm's coordinate-sort
/// is strictly more permissive (it needs no cell_vertices at all, only a
/// vertex position per source DOF) and is the one every existing 1D test
/// and the FSI beam model were written and verified against -- see
/// build_consistent_interp_weights_1d's own header for why coordinate order
/// and mesh order coincide for a genuine 1D interface. A mesh of literal
/// `CellShape::Edge` cells with real connectivity would also be handled
/// correctly by `build_general_interp_weights` (an edge is just a 2-corner,
/// 1-parameter affine cell, a degenerate case of `Triangle`'s machinery),
/// but nothing here relies on that: the dispatch is purely on
/// `dimension()`, so an Edge-shaped source never actually reaches
/// general_interp.hpp.
[[nodiscard]] inline Result<RemapWeights> build_consistent_interp_weights(
    const MeshView& src_nodes, const MeshView& tgt_points, int tgt_dim,
    InterfaceDomainPolicy policy = InterfaceDomainPolicy::Refuse) {
  if (src_nodes.dimension() == 1) {
    return build_consistent_interp_weights_1d(src_nodes, tgt_points, tgt_dim, policy);
  }
  return build_general_interp_weights(src_nodes, tgt_points, tgt_dim, policy);
}

namespace consistent_interp_detail {

/// Mirrors `intx_weights.cpp`'s anonymous-namespace `supermesh_side`
/// (duplicated rather than shared: that helper is file-local to a different
/// translation unit). Resolves a `RegionId` to its `MeshView` and entity
/// dimension, rejecting exactly the cases that helper rejects.
struct Side {
  const MeshView* view = nullptr;
  int             dim = 0;
};

[[nodiscard]] inline Result<Side> resolve_side(const CouplingGraph& graph,
                                               const MeshBackend& backend,
                                               RegionId id, ComponentId placement,
                                               const char* which) {
  if (!id.valid() || id.value() >= graph.region_count()) {
    return Failure{"E-INTERP-REGION",
                   fmt::format("{} region id {} is not one of the graph's {} "
                               "regions",
                               which, id.value(), graph.region_count())};
  }
  const Region& r = graph.region(id);
  if (!(r.entity_set.empty() || r.entity_set == "all")) {
    return Failure{
        "E-INTERP-REGION",
        fmt::format("{} region '{}' names entity set '{}'. The consistent-"
                    "interpolation builder indexes MeshView local indices "
                    "over the whole mesh, exactly like the supermesh "
                    "builder, so a sub-set region would give a matrix wrong "
                    "by a permutation",
                    which, r.name, r.entity_set)};
  }
  if (!r.mesh.valid() || r.mesh.value() >= graph.mesh_count()) {
    return Failure{"E-INTERP-REGION",
                   fmt::format("{} region '{}' names no valid mesh", which,
                               r.name)};
  }
  const auto mid = backend.mesh_id(graph.mesh(r.mesh).name);
  if (!mid.ok()) return mid.error();
  const auto view = backend.placed_view(mid.value(), placement);
  if (!view.ok()) return view.error();
  Side s;
  s.view = view.value();
  s.dim = r.dimension >= 0 ? r.dimension : s.view->dimension();
  return s;
}

/// True when `disc`/`order` denote a genuinely nodal (vertex) field: a
/// continuous representation at order <= 1, which `connect.hpp`'s
/// `to_semantics` already maps to `SemanticLocation::Vertex`.
[[nodiscard]] inline bool is_nodal(Discretization disc, int order) {
  return disc == Discretization::ContinuousGLL && order <= 1;
}

}  // namespace consistent_interp_detail

/// The `build_weights()` entry point for `RemapPlan::method` values
/// `"consistent"` and `"consistent-transpose"`. Resolves `plan.src`/
/// `plan.tgt` through `backend` exactly as the supermesh path does, decides
/// which side is the nodal (vertex) side from `plan.src_disc`/`plan.tgt_disc`
/// and `plan.src_order`/`plan.tgt_order`, and returns either `H` or `H^T D`
/// as documented above.
///
/// Additionally rejects `E-INTERP-METHOD` for any other method string --
/// callers reach this only after `build_weights()` has already recognised
/// the method as one of these two, so this is a defensive, not a primary,
/// check -- and `E-INTERP-DISC` when the side that must be nodal is not
/// `Discretization::ContinuousGLL` order <= 1.
[[nodiscard]] inline Result<RemapWeights> build_consistent_weights_for_plan(
    const RemapPlan& plan, const CouplingGraph& graph,
    const MeshBackend& backend) {
  using consistent_interp_detail::is_nodal;
  using consistent_interp_detail::resolve_side;

  auto src = resolve_side(graph, backend, plan.src, plan.placement, "source");
  if (!src.ok()) return src.error();
  auto tgt = resolve_side(graph, backend, plan.tgt, plan.placement, "target");
  if (!tgt.ok()) return tgt.error();

  if (plan.method == "consistent") {
    if (!is_nodal(plan.src_disc, plan.src_order)) {
      return Failure{
          "E-INTERP-DISC",
          fmt::format("method 'consistent' builds H, the piecewise-linear "
                      "interpolation FROM the source field's nodal DOFs; the "
                      "source field is {} order {}, not a continuous field "
                      "at order <= 1",
                      to_cstring(plan.src_disc), plan.src_order)};
    }
    // The target's evaluation points are its own vertices for a node-to-node
    // transfer, or its cells' centroids (e.g. a P0 fluid wall's face
    // midpoints) otherwise.
    const int tgt_eval_dim = is_nodal(plan.tgt_disc, plan.tgt_order)
                                 ? 0
                                 : tgt.value().dim;
    return build_consistent_interp_weights(*src.value().view,
                                           *tgt.value().view, tgt_eval_dim);
  }

  if (plan.method == "consistent-transpose") {
    if (!is_nodal(plan.tgt_disc, plan.tgt_order)) {
      return Failure{
          "E-INTERP-DISC",
          fmt::format(
              "method 'consistent-transpose' delivers H^T D, a conservative "
              "load transfer TO the target field's nodal DOFs; the target "
              "field is {} order {}, not a continuous field at order <= 1",
              to_cstring(plan.tgt_disc), plan.tgt_order)};
    }
    const bool src_is_load = is_nodal(plan.src_disc, plan.src_order);
    // Build H itself with the SAME orientation the paired forward field transfer
    // would use: nodal side = plan.tgt (here, the "beam"), point side =
    // plan.src (here, the "fluid"). The measure that converts the fluid's
    // density to a load, if one is needed, is therefore plan.src's own
    // mesh entity set dimension -- never plan.tgt's, which may be dimension 0 and
    // therefore have no measure at all.
    const int src_eval_dim = src_is_load ? 0 : src.value().dim;
    auto h = build_consistent_interp_weights(*tgt.value().view,
                                             *src.value().view, src_eval_dim);
    if (!h.ok()) return h.error();
    const RemapWeights& hw = h.value();
    // hw: n_rows = |plan.src entities| (the point side), n_cols = |plan.tgt
    // entities| (the nodal side). The returned matrix is `H^T D`: n_rows =
    // hw.n_cols, n_cols = hw.n_rows, value at (i, j) = hw(j, i) * D_j.
    RemapWeights out;
    out.n_rows = hw.n_cols;
    out.n_cols = hw.n_rows;
    out.order = hw.order;
    out.row.reserve(hw.val.size());
    out.col.reserve(hw.val.size());
    out.val.reserve(hw.val.size());
    double min_weight = hw.val.empty() ? 0.0 : 1.0;
    for (std::size_t k = 0; k < hw.val.size(); ++k) {
      const auto fluid_pt = static_cast<std::size_t>(hw.row[k]);
      const auto beam_dof = static_cast<std::size_t>(hw.col[k]);
      const double scale =
          src_is_load ? 1.0
                      : src.value().view->measure(src.value().dim, fluid_pt);
      const double v = hw.val[k] * scale;
      out.row.push_back(static_cast<std::int32_t>(beam_dof));
      out.col.push_back(static_cast<std::int32_t>(fluid_pt));
      out.val.push_back(v);
      min_weight = std::min(min_weight, v);
    }
    out.min_weight = min_weight;
    // This matrix is a LOAD TRANSFER, not an interpolation: its row sums are
    // measure-weighted sums of H's columns, not 1, so it neither preserves
    // constants nor is conservative in `OperatorClaims`' sense (that axis
    // means "column sums weighted by a TARGET measure equal the source
    // measure", which does not even type-check here -- the nodal side has no
    // measure to weight by). `max_row_defect`/`max_col_defect` are therefore
    // set to a fixed sentinel ABOVE `WeightsOperator`'s claim tolerance
    // (1e-10) rather than to the (differently-typed) numbers this builder
    // could compute, so as not to manufacture a false claim the conformance
    // fixture would then hold this operator to incorrectly. What this matrix
    // actually conserves -- the total, summed over all rows, equals the
    // total load sent -- is a distinct, genuine property, proven in the
    // header from H's row-sum-one identity and checked directly by
    // ConsistentInterp.TransposeConservesTotalLoad rather than through
    // OperatorClaims.
    out.max_row_defect = 1.0;
    out.max_col_defect = 1.0;
    out.monotone = false;
    out.src_measure_total = 0.0;
    out.tgt_measure_total = 0.0;
    return out;
  }

  return Failure{"E-INTERP-METHOD",
                 fmt::format("build_consistent_weights_for_plan: '{}' is "
                             "neither 'consistent' nor 'consistent-transpose'",
                             plan.method)};
}

}  // namespace fuse
