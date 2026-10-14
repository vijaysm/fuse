#include "fuse/backend/supermesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "fuse/base/error.hpp"

#include "supermesh_detail.hpp"

extern "C" {
#include "r2d.h"
#include "r3d.h"
}

namespace fuse {
namespace {

/// A simplex lifted out of a MeshView, with the cell it belongs to.
struct Simplex {
  std::array<Point, 4> p{};      ///< dim+1 of these are used
  std::size_t          parent = 0;
  double               lo[3] = {0, 0, 0};
  double               hi[3] = {0, 0, 0};
};

[[nodiscard]] double coord(const Point& p, int k) {
  return k == 0 ? p.x : (k == 1 ? p.y : p.z);
}

/// Every simplex of every entity at `dim`, with its bounding box. Rejects when
/// a view carries no geometry: that is legal for a view (the contract says so)
/// but it is not something to build a supermesh from.
[[nodiscard]] Result<std::vector<Simplex>> lift(const MeshView& m, int dim,
                                                const char* which) {
  const std::size_t n = m.entity_count(dim);
  std::vector<Simplex> out;
  out.reserve(n * (dim == 3 ? 6 : 2));
  for (std::size_t c = 0; c < n; ++c) {
    const std::size_t ns = m.simplex_count(dim, c);
    if (ns == 0) {
      return Failure{
          "E-REMAP-NOGEOMETRY",
          std::string{which} + " mesh reports no simplices for entity " +
              std::to_string(c) +
              ", so it carries no geometry to intersect. A supermesh built "
              "from it would be empty, which is indistinguishable from two "
              "meshes that genuinely do not overlap"};
    }
    for (std::size_t s = 0; s < ns; ++s) {
      Simplex sx;
      sx.parent = c;
      m.simplex_points(dim, c, s, sx.p.data());
      for (int k = 0; k < 3; ++k) {
        sx.lo[k] = coord(sx.p[0], k);
        sx.hi[k] = sx.lo[k];
      }
      for (int v = 1; v <= dim; ++v) {
        for (int k = 0; k < 3; ++k) {
          sx.lo[k] = std::min(sx.lo[k], coord(sx.p[v], k));
          sx.hi[k] = std::max(sx.hi[k], coord(sx.p[v], k));
        }
      }
      out.push_back(sx);
    }
  }
  return out;
}

/// `axis1d` is the coordinate axis `build_supermesh_weights` selected for a
/// 1D pair (REVIEW.md TF-4) -- `dim == 1` tests ONLY that axis, since the
/// other two are the pair's (near-)constant flat coordinates, not a second
/// dimension to be boxed in. Ignored for `dim >= 2`.
[[nodiscard]] bool boxes_overlap(const Simplex& a, const Simplex& b, int dim,
                                 int axis1d) {
  if (dim == 1) {
    return !(a.hi[axis1d] < b.lo[axis1d] || b.hi[axis1d] < a.lo[axis1d]);
  }
  for (int k = 0; k < dim; ++k) {
    if (a.hi[k] < b.lo[k] || b.hi[k] < a.lo[k]) return false;
  }
  return true;
}

// --- the geometric kernel --------------------------------------------------
//
// r3d and r2d clip a polytope against half-spaces, keeping the side where
// `dot(n, x) + d >= 0` (verified in their clip loops: a vertex with a negative
// signed distance is the one discarded). So the planes of the CLIPPING simplex
// must point INWARD.
//
// The planes are constructed here rather than with r3d_tet_faces_from_verts,
// and the inward direction is established by TESTING against the opposite
// vertex instead of relying on a vertex-ordering convention. The contract on
// MeshView deliberately does not require an orientation, so there is no
// ordering to rely on -- and an outward-pointing plane would not error, it
// would silently return the complement's measure.
//
// `inward_plane` itself lives in supermesh_detail.hpp, a private header
// (REVIEW.md R-12) rather than in this anonymous namespace, so
// test/backend/supermesh_test.cpp can call it directly with a manufactured
// degenerate face -- the regression test the original fix pass could not
// add through the public MeshView grammar alone.
using fuse::supermesh_detail::inward_plane;

/// The exact measure of the intersection of two simplices. `axis1d` is the
/// coordinate axis to read for `dim == 1` (REVIEW.md TF-4: hard-coding `.x`
/// silently measured the wrong thing -- an EMPTY overlap on every pair for
/// a y- or z-aligned line, since both "intervals" degenerate to the same
/// constant x); ignored for `dim >= 2`.
[[nodiscard]] double intersect(const Simplex& t, const Simplex& s, int dim,
                               int axis1d) {
  if (dim == 1) {
    // An overlap of intervals is a subtraction. Using a clipper here would add
    // rounding rather than remove it.
    const double tlo = std::min(coord(t.p[0], axis1d), coord(t.p[1], axis1d));
    const double thi = std::max(coord(t.p[0], axis1d), coord(t.p[1], axis1d));
    const double slo = std::min(coord(s.p[0], axis1d), coord(s.p[1], axis1d));
    const double shi = std::max(coord(s.p[0], axis1d), coord(s.p[1], axis1d));
    return std::max(0.0, std::min(thi, shi) - std::max(tlo, slo));
  }

  if (dim == 2) {
    r2d_poly poly;
    r2d_rvec2 verts[3];
    for (int v = 0; v < 3; ++v) {
      verts[v].xy[0] = t.p[v].x;
      verts[v].xy[1] = t.p[v].y;
    }
    // r2d_init_poly documents counterclockwise input; a clockwise triangle
    // yields inverted edges and a zero or nonsense area. The signed area says
    // which we were handed, and MeshView promises nothing about it.
    const double twice_area =
        (verts[1].xy[0] - verts[0].xy[0]) * (verts[2].xy[1] - verts[0].xy[1]) -
        (verts[2].xy[0] - verts[0].xy[0]) * (verts[1].xy[1] - verts[0].xy[1]);
    if (twice_area < 0.0) std::swap(verts[1], verts[2]);
    if (twice_area == 0.0) return 0.0;  // degenerate target: no area to give
    r2d_init_poly(&poly, verts, 3);

    r2d_plane planes[3];
    for (int f = 0; f < 3; ++f) {
      double n[3], d;
      // REVIEW.md R-12: a degenerate clipping edge has no half-plane to
      // give; clip to zero rather than silently impose no constraint.
      if (!inward_plane<2>(s.p, f, n, d)) return 0.0;
      planes[f].n.xy[0] = n[0];
      planes[f].n.xy[1] = n[1];
      planes[f].d = d;
    }
    r2d_clip(&poly, planes, 3);
    r2d_real moments[R2D_NUM_MOMENTS(0)];
    r2d_reduce(&poly, moments, 0);
    return moments[0] > 0.0 ? moments[0] : 0.0;
  }

  r3d_poly poly;
  r3d_rvec3 verts[4];
  for (int v = 0; v < 4; ++v) {
    verts[v].xyz[0] = t.p[v].x;
    verts[v].xyz[1] = t.p[v].y;
    verts[v].xyz[2] = t.p[v].z;
  }
  // Same reasoning as the 2D case: r3d_init_tet's moments come out negative
  // for a negatively oriented tetrahedron.
  if (r3d_orient(verts) < 0.0) std::swap(verts[1], verts[2]);
  r3d_init_tet(&poly, verts);

  r3d_plane planes[4];
  for (int f = 0; f < 4; ++f) {
    double n[3], d;
    // REVIEW.md R-12: a degenerate clipping face has no half-plane to give;
    // clip to zero rather than silently impose no constraint.
    if (!inward_plane<3>(s.p, f, n, d)) return 0.0;
    planes[f].n.xyz[0] = n[0];
    planes[f].n.xyz[1] = n[1];
    planes[f].n.xyz[2] = n[2];
    planes[f].d = d;
  }
  r3d_clip(&poly, planes, 4);
  r3d_real moments[R3D_NUM_MOMENTS(0)];
  r3d_reduce(&poly, moments, 0);
  return moments[0] > 0.0 ? moments[0] : 0.0;
}

}  // namespace

Result<RemapWeights> build_supermesh_weights(const MeshView& src,
                                             const MeshView& tgt, int dim) {
  if (dim < 1 || dim > 3) {
    return Failure{"E-REMAP-DIM",
                   "supermesh weights are defined for entity dimension 1, 2 "
                   "or 3, not " +
                       std::to_string(dim)};
  }

  auto s_lift = lift(src, dim, "source");
  if (!s_lift.ok()) return s_lift.error();
  auto t_lift = lift(tgt, dim, "target");
  if (!t_lift.ok()) return t_lift.error();
  const std::vector<Simplex>& ss = s_lift.value();
  const std::vector<Simplex>& ts = t_lift.value();

  // The 2D kernel is PLANAR: it clips in (x, y) and ignores z. Handed a
  // surface embedded in three-space -- a codimension-1 mesh, the box skin for
  // instance -- it would silently return the measure of the shadow on the
  // xy-plane, which is a plausible number and the wrong one. Reject instead.
  //
  // Supporting it properly needs a surface parameterisation (or great-circle
  // geometry for the spherical case, which is what the TempestRemap tier is
  // for), not a different clipper.
  if (dim == 2) {
    const auto out_of_plane = [](const std::vector<Simplex>& v) {
      double zlo = 0.0, zhi = 0.0;
      bool first = true;
      for (const Simplex& s : v) {
        for (int k = 0; k <= 2; ++k) {
          const double z = s.p[k].z;
          if (first) { zlo = zhi = z; first = false; }
          zlo = std::min(zlo, z);
          zhi = std::max(zhi, z);
        }
      }
      return zhi - zlo;
    };
    const double spread = std::max(out_of_plane(ss), out_of_plane(ts));
    if (spread > 0.0) {
      return Failure{
          "E-REMAP-NONPLANAR",
          "the 2D supermesh kernel clips in the xy-plane, and these surfaces "
          "span " + std::to_string(spread) +
              " in z. Projecting them would return the measure of their "
              "shadow, which is a plausible wrong answer; a surface pair needs "
              "a parameterisation (or great-circle geometry on the sphere, "
              "which the TempestRemap tier provides)"};
    }
  }

  // The 1D kernel (`intersect`'s `dim == 1` branch) is a single coordinate
  // subtraction, with no analogue of the check just above (REVIEW.md
  // TF-4). Reading `.x` unconditionally had two silent failure modes: a
  // line laid out along y or z has every pair of "intervals" degenerate to
  // the SAME constant x, so every overlap computes to exactly 0 -- an
  // EMPTY matrix returned as SUCCESS, indistinguishable from a target
  // field correctly left at zero; and a genuinely diagonal line would have
  // its x-SHADOW measured instead of its own length, a plausible but wrong
  // number for exactly the reason the dim == 2 check above refuses
  // projection rather than risk it.
  //
  // Both are closed the SAME way: find the single coordinate axis along
  // which these segments actually vary (not necessarily x -- a beam laid
  // out along y or z is handled, matching
  // `build_consistent_interp_weights_1d`'s own axis choice,
  // consistent_interp.hpp), and refuse outright, the SAME way the dim == 2
  // check does, when the segments are not aligned with any SINGLE
  // coordinate axis -- i.e. when projecting onto the chosen axis would
  // measure a shadow, not a length.
  int axis1d = 0;
  if (dim == 1) {
    double lo[3], hi[3];
    bool first = true;
    const auto fold = [&](const std::vector<Simplex>& v) {
      for (const Simplex& s : v) {
        for (int vtx = 0; vtx <= 1; ++vtx) {
          for (int k = 0; k < 3; ++k) {
            const double c = coord(s.p[vtx], k);
            if (first) { lo[k] = hi[k] = c; }
            lo[k] = std::min(lo[k], c);
            hi[k] = std::max(hi[k], c);
          }
          first = false;
        }
      }
    };
    fold(ss);
    fold(ts);
    double spread[3] = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    for (int k = 1; k < 3; ++k) {
      if (spread[k] > spread[axis1d]) axis1d = k;
    }
    const double diameter = spread[axis1d];
    const double tol = std::max(diameter, 1.0) * supermesh_detail::kSupermeshCoverageTol;
    int nonzero_axes = 0;
    for (int k = 0; k < 3; ++k) {
      if (spread[k] > tol) ++nonzero_axes;
    }
    if (nonzero_axes > 1) {
      return Failure{
          "E-REMAP-NONPLANAR",
          "the 1D supermesh kernel clips along a single coordinate axis, and "
          "these segments vary along more than one (spread x=" +
              std::to_string(spread[0]) + " y=" + std::to_string(spread[1]) +
              " z=" + std::to_string(spread[2]) +
              "). Projecting onto one axis would measure the segments' "
              "SHADOW, not their own length -- a plausible wrong answer; a "
              "non-axis-aligned 1D interface needs an arc-length "
              "parameterisation, not a coordinate subtraction"};
    }
  }

  RemapWeights w;
  w.n_rows = tgt.entity_count(dim);
  w.n_cols = src.entity_count(dim);
  w.order = 1;

  // Accumulated supermesh measure per (target cell, source cell). A map keyed
  // on the pair, not a dense matrix: the supermesh of two resolved meshes is
  // extremely sparse, and a dense n_rows x n_cols would be the dominant cost
  // and the dominant memory at any realistic size.
  std::map<std::pair<std::size_t, std::size_t>, double> acc;
  std::vector<double> src_covered(w.n_cols, 0.0);
  std::vector<double> tgt_covered(w.n_rows, 0.0);

  // --- the negligible-overlap floor ----------------------------------------
  //
  // Two simplices that merely SHARE A FACE have an intersection of measure
  // zero, but clipping one against the other's coplanar plane returns a
  // rounding-level positive: measured at up to 1.1e-16 of a cell on an
  // identity pair, where it produced 84 spurious off-diagonal entries beside
  // the 27 real ones.
  //
  // Those entries are not numerically harmful -- the diagonal was exactly 1
  // and both residuals were at rounding -- but they are not free either. An
  // entry in the matrix is a DATA DEPENDENCY: in parallel it makes the target
  // rank wait on the source rank that owns that cell, for a contribution of
  // 1e-16. Dropping them is about the communication graph, not about accuracy.
  //
  // The floor is RELATIVE to the target cell's measure, because that is what
  // the weight is divided by and because an absolute epsilon would mean
  // something different on every mesh scale. A contribution below it cannot
  // move a double-precision result. It is applied BEFORE the audit, so mass
  // lost to the floor is reported in `max_col_defect` rather than concealed by
  // computing the audit from entries that were then discarded.
  constexpr double kNegligible = 1e-12;
  // REVIEW.md R-13: the coverage tolerance itself, and the rejection it backs,
  // now live in supermesh_detail::audit_coverage -- the same relative
  // tolerance intx_weights.cpp's MOAB path uses for its over_src/over_tgt
  // audit (`kCoverageTol` there), so the two paths reject on the same
  // threshold, and unit-testable directly (test/backend/supermesh_test.cpp).

  // Brute force over bounding-box-compatible pairs. This is O(n_t * n_s) box
  // tests, which is the honest starting point: it is exact, it has no tuning
  // parameter, and it is the reference a spatial index must reproduce. The
  // index is a performance task, and replacing this before it is measured
  // would be replacing something correct with something unmeasured.
  for (const Simplex& t : ts) {
    for (const Simplex& s : ss) {
      if (!boxes_overlap(t, s, dim, axis1d)) continue;
      const double m = intersect(t, s, dim, axis1d);
      if (m <= 0.0) continue;
      acc[{t.parent, s.parent}] += m;
      src_covered[s.parent] += m;
      tgt_covered[t.parent] += m;
    }
  }

  // --- assemble, and audit while assembling --------------------------------
  w.row.reserve(acc.size());
  w.col.reserve(acc.size());
  w.val.reserve(acc.size());
  double min_w = 1.0;
  std::vector<double> row_sum(w.n_rows, 0.0);
  std::size_t dropped = 0;
  for (const auto& [key, measure] : acc) {
    const auto [j, i] = key;
    const double mj = tgt.measure(dim, j);
    if (!(mj > 0.0)) {
      return Failure{"E-REMAP-MEASURE",
                     "target entity " + std::to_string(j) +
                         " has measure " + std::to_string(mj) +
                         "; dividing by it would give an infinite weight"};
    }
    const double val = measure / mj;
    if (val < kNegligible) {
      ++dropped;
      continue;
    }
    w.row.push_back(static_cast<std::int32_t>(j));
    w.col.push_back(static_cast<std::int32_t>(i));
    w.val.push_back(val);
    row_sum[j] += val;
    min_w = std::min(min_w, val);
  }
  // REVIEW.md TF-15: `acc.empty()` (no candidate pair overlapped at all) is
  // not the only way `w.val` ends up empty -- every accumulated entry
  // falling below `kNegligible` and being dropped does too, and the loop
  // above only updates `min_w` for a KEPT entry, so an all-dropped `acc`
  // would otherwise leave `min_w` at its initial 1.0. `RemapWeights`
  // documents zero for an empty matrix (`min_weight`'s own contract); key
  // on `w.val`, the matrix actually returned, not on `acc`, the
  // pre-filter accumulator.
  w.min_weight = w.val.empty() ? 0.0 : min_w;

  for (std::size_t i = 0; i < w.n_cols; ++i) {
    w.src_measure_total += src.measure(dim, i);
  }
  for (std::size_t j = 0; j < w.n_rows; ++j) {
    w.tgt_measure_total += tgt.measure(dim, j);
  }

  // Coverage in the SUPERMESH's own measure, so it is a pure geometric
  // fraction and inherits nothing from either mesh's measure convention.
  // Coverage from the KEPT entries, so it describes the matrix returned.
  std::fill(src_covered.begin(), src_covered.end(), 0.0);
  std::fill(tgt_covered.begin(), tgt_covered.end(), 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    const auto j = static_cast<std::size_t>(w.row[k]);
    const auto i = static_cast<std::size_t>(w.col[k]);
    const double m = w.val[k] * tgt.measure(dim, j);
    src_covered[i] += m;
    tgt_covered[j] += m;
  }
  // REVIEW.md R-13: the MOAB path rejects E-REMAP-INTX when a cell is
  // assigned more area than it has (intx_weights.cpp's over_src/over_tgt
  // audit, kCoverageTol). This path tracked only the minimum, so the exact
  // defect class R-12 produces -- overlap inflated past the cell it came
  // from -- passed here as a wrong, conservative-looking weight matrix
  // instead of being caught. Checked against the SAME kept-entries coverage
  // the minimum already uses, so a degenerate cell fixed by R-12 (now
  // contributing zero) cannot trip this.
  for (std::size_t i = 0; i < w.n_cols; ++i) {
    const double m = src.measure(dim, i);
    if (m > 0.0) {
      w.min_src_coverage = std::min(w.min_src_coverage, src_covered[i] / m);
    }
    const auto audit =
        supermesh_detail::audit_coverage(src_covered[i], m, "source", i, dim);
    if (!audit.ok()) return audit.error();
  }
  for (std::size_t j = 0; j < w.n_rows; ++j) {
    const double m = tgt.measure(dim, j);
    if (m > 0.0) {
      w.min_tgt_coverage = std::min(w.min_tgt_coverage, tgt_covered[j] / m);
    }
    const auto audit =
        supermesh_detail::audit_coverage(tgt_covered[j], m, "target", j, dim);
    if (!audit.ok()) return audit.error();
  }

  // Constant preservation: the row residual.
  for (std::size_t j = 0; j < w.n_rows; ++j) {
    w.max_row_defect = std::max(w.max_row_defect, std::abs(row_sum[j] - 1.0));
  }

  // Conservation: the column residual. sum_j W_ji m_j is the mass the map
  // delivers from a unit field on source cell i, and it must equal m_i.
  std::vector<double> col_mass(w.n_cols, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    col_mass[static_cast<std::size_t>(w.col[k])] +=
        w.val[k] * tgt.measure(dim, static_cast<std::size_t>(w.row[k]));
  }
  for (std::size_t i = 0; i < w.n_cols; ++i) {
    const double mi = src.measure(dim, i);
    if (mi > 0.0) {
      w.max_col_defect =
          std::max(w.max_col_defect, std::abs(col_mass[i] - mi) / mi);
    }
  }
  // Identical for this builder: it assembles from MeshView::measure directly,
  // so there is no second measure convention for the two numbers to differ in.
  // They diverge only for the spherical path, which is conservative in its own
  // great-circle areas.
  w.max_col_defect_native = w.max_col_defect;
  (void)dropped;  // counted for the comment above; not a field on RemapWeights

  // Bound preserving only if actually checked, never by assertion: every
  // weight non-negative AND every row summing to one. A partially covered
  // target cell fails the second and dilutes a constant downwards, so it is
  // not monotone even though its weights are all positive.
  w.monotone = (w.min_weight >= 0.0) && (w.max_row_defect <= 1e-12);
  return w;
}

}  // namespace fuse
