#pragma once

/// General consistent (nodal) interpolation on ANY supported cell shape --
/// the counterpart of consistent_interp.hpp's original 1D-curve-only `H` for
/// a source mesh whose cells are triangles, quadrilaterals, tetrahedra,
/// prisms or hexahedra (including a curved SURFACE mesh embedded in R^3,
/// e.g. a cubed sphere). `consistent_interp.hpp`'s public
/// `build_consistent_interp_weights` dispatches here whenever the source
/// view's topological dimension is 2 or 3; its own 1D (segment) algorithm is
/// UNCHANGED and still used, unconditionally, whenever that dimension is 1 --
/// see that file's header for why the two are kept as separate code paths
/// rather than unified into one.
///
/// THE ALGORITHM, for one target evaluation point `x`:
///   1. broad phase: `fuse::search::BoxIndex` (built once per call over the
///      source mesh's cells, `fuse::search::element_boxes`) returns every
///      cell whose tolerance-padded bounding box contains `x`;
///   2. narrow phase: for each candidate, invert that cell's reference map
///      (`invert_reference_map` below) to get `x`'s NATURAL coordinates in
///      the cell, and test membership in the reference domain
///      (`in_reference_domain`);
///   3. among every candidate that genuinely contains `x` (within
///      tolerance), the ties are broken by the LOWEST source cell global id
///      (`MeshView::global_id`) -- deterministic regardless of which
///      candidate the spatial index happened to list first, and regardless
///      of rank count in the distributed builder (nodal_remap_distributed.hpp)
///      that reuses this same per-cell primitive;
///   4. the winning cell's `basis_weights` are scattered into the row at
///      that cell's corner `MeshView` dimension-0 local indices
///      (`MeshView::cell_vertices`).
///
/// REFERENCE MAPS AND THEIR INVERSE. A cell's physical position is an affine
/// (Triangle, Tet) or multilinear (Quad, Prism, Hex) combination of its
/// corners; `basis_weights`/`jacobian_columns` give the forward map and its
/// derivatives in closed form (no shape needs numerical differencing), and
/// `invert_reference_map` runs a damped Gauss-Newton iteration on the
/// physical residual (least-squares when the cell's manifold dimension is
/// less than 3, e.g. a triangle embedded in R^3 -- see `NPARAMS` below) to
/// find the natural coordinates. An AFFINE cell (Triangle, Tet) has a
/// CONSTANT Jacobian, so the very first Newton step lands exactly (to
/// round-off): the iteration is not special-cased for them, it simply
/// converges in one step.
///
/// SHAPES NOT HANDLED, REJECTED BY NAME (`E-INTERP-SHAPE`):
///   * `Pyramid` -- the standard nodal basis for a pyramid is RATIONAL (its
///     apex is a genuine basis singularity; see e.g. Bedrosian's or
///     Zgainski's rational pyramid element), not the polynomial family every
///     other shape here uses, and implementing and validating a rational
///     basis is out of this track's scope -- rejecting loudly beats a
///     plausible but unvalidated one.
///   * `Polygon` (n > 4) and `Polyhedron` -- no STANDARD nodal basis exists
///     for either (cell_simplices.hpp's `Polygon` decomposition is a fan
///     triangulation for the CLIPPING kernel, not a basis); a caller with a
///     polygonal or polyhedral source has no representation this builder
///     can interpolate against.
///   * `Point` -- a 0D cell has no interior to locate a target inside.
///
/// SURFACES EMBEDDED IN R^3 (a sphere's triangles or quads: manifold
/// dimension 2, embedding dimension 3). `invert_reference_map`'s Gauss-Newton
/// step minimises the SQUARED physical residual over the cell's 2 natural
/// parameters, so the point it converges to is the closest point ON the
/// (possibly non-planar) cell to the target -- exactly "project onto the
/// cell". The leftover residual after convergence is the projection (normal)
/// distance; `build_general_interp_weights` accepts a hit only when that
/// distance is within `kSurfaceTol` (relative to the cell's own diagonal) of
/// zero, and otherwise treats the point as not contained by that cell (same
/// domain-edge handling as a laterally-outside point) -- never a silent,
/// arbitrarily-far projection.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "fuse/backend/cell_simplices.hpp"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"
#include "fuse/search/box_index.hpp"
#include "fuse/search/geometry.hpp"
#include "fuse/search/mesh_adapter.hpp"

namespace fuse {

/// What `build_general_interp_weights` (and the 1D
/// `build_consistent_interp_weights`) do when a target evaluation point
/// falls outside the source's coverage -- laterally outside every cell's
/// reference domain, or (a surface source) too far off the surface. Shared
/// with consistent_interp.hpp's original 1D algorithm; defined here so
/// general_interp.hpp does not have to include consistent_interp.hpp (which
/// includes this file) to name it.
enum class InterfaceDomainPolicy {
  /// The default. Rejects with `E-INTERP-DOMAIN`, naming the offending point.
  Refuse,
  /// Nearest-cell extrapolation: the point is assigned the weights of
  /// whichever candidate cell comes closest to containing it, with its
  /// natural coordinates clamped into that cell's reference domain. Still
  /// row-sum-one (a clamped point is still a valid convex combination of
  /// that cell's corners), so every conservation/power-balance property
  /// continues to hold; what is lost is accuracy near the uncovered edge.
  ClampToBoundary,
};

namespace general_interp_detail {

/// Natural-coordinate parameter count for each supported shape: 1 for an
/// Edge (a segment, P1), 2 for a surface cell (Triangle, Quad), 3 for a
/// volume cell (Tet, Prism, Hex).
///
/// EDGE IS SUPPORTED HERE, BUT `consistent_interp.hpp`'s PUBLIC DISPATCH
/// NEVER REACHES IT: a source mesh of topological dimension 1 always routes
/// to `build_consistent_interp_weights_1d` (the original, coordinate-sort
/// algorithm), unconditionally, so this Edge case is dead code from that
/// entry point's perspective -- see that file's own dispatch comment for
/// why. It exists here because `nodal_remap_distributed.hpp`'s distributed
/// builder does NOT go through that dimension-based dispatch at all (it
/// calls `invert_reference_map`/`basis_weights` directly, on cells of
/// whatever dimension the source mesh actually has), and a 1D interface is
/// as legitimate a THING TO DISTRIBUTE as any other -- the distributed
/// algorithm is the same per-cell search regardless of dimension, which is
/// the whole reason it needs its own assembly step rather than reusing the
/// serial 1D algorithm's whole-view coordinate sort.
[[nodiscard]] constexpr int shape_nparams(CellShape s) {
  switch (s) {
    case CellShape::Edge:
      return 1;
    case CellShape::Triangle:
    case CellShape::Quad:
      return 2;
    case CellShape::Tet:
    case CellShape::Prism:
    case CellShape::Hex:
      return 3;
    default:
      return 0;  // Polygon, Pyramid, Point: not handled here.
  }
}

[[nodiscard]] constexpr std::size_t shape_corner_count(CellShape s) {
  switch (s) {
    case CellShape::Edge: return 2;
    case CellShape::Triangle: return 3;
    case CellShape::Quad: return 4;
    case CellShape::Tet: return 4;
    case CellShape::Prism: return 6;
    case CellShape::Hex: return 8;
    default: return 0;
  }
}

[[nodiscard]] inline bool shape_supported(CellShape s) {
  return shape_nparams(s) != 0;
}

struct Vec3 {
  double x = 0, y = 0, z = 0;
  Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
  Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
  Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};
[[nodiscard]] inline Vec3 to_vec(Point p) { return {p.x, p.y, p.z}; }
[[nodiscard]] inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
[[nodiscard]] inline double norm(Vec3 a) { return std::sqrt(dot(a, a)); }

/// Up to 8 corner physical positions, and the shape they belong to.
struct Cell {
  CellShape shape = CellShape::Edge;
  std::array<Vec3, 8> p{};
  std::size_t n = 0;
};

/// Forward map `X(u)` and its Jacobian columns `dX/du_k`, in closed form --
/// every shape here is affine or multilinear in its natural coordinates, so
/// no shape needs numerical differencing. `u` has `shape_nparams(shape)`
/// meaningful entries; the rest are ignored.
struct MapResult {
  Vec3 x{};
  std::array<Vec3, 3> jac{};  // jac[k] = dX/du_k, k < nparams
};

[[nodiscard]] inline MapResult eval_map(const Cell& c, const std::array<double, 3>& u) {
  const double r = u[0], s = u[1], t = u[2];
  MapResult m;
  switch (c.shape) {
    case CellShape::Edge: {
      m.x = c.p[0] * (1 - r) + c.p[1] * r;
      m.jac[0] = c.p[1] - c.p[0];
      return m;
    }
    case CellShape::Triangle: {
      const double L0 = 1 - r - s;
      m.x = c.p[0] * L0 + c.p[1] * r + c.p[2] * s;
      m.jac[0] = c.p[1] - c.p[0];
      m.jac[1] = c.p[2] - c.p[0];
      return m;
    }
    case CellShape::Tet: {
      const double L0 = 1 - r - s - t;
      m.x = c.p[0] * L0 + c.p[1] * r + c.p[2] * s + c.p[3] * t;
      m.jac[0] = c.p[1] - c.p[0];
      m.jac[1] = c.p[2] - c.p[0];
      m.jac[2] = c.p[3] - c.p[0];
      return m;
    }
    case CellShape::Quad: {
      // Corner order 0,1,2,3 walks the quad's PERIMETER (cell_simplices.hpp:
      // the 0-2 diagonal splits it), so the standard CCW bilinear
      // parametrisation applies: corner0=(0,0), corner1=(1,0), corner2=(1,1),
      // corner3=(0,1).
      const Vec3& P0 = c.p[0]; const Vec3& P1 = c.p[1];
      const Vec3& P2 = c.p[2]; const Vec3& P3 = c.p[3];
      m.x = P0 * ((1 - r) * (1 - s)) + P1 * (r * (1 - s)) + P2 * (r * s) + P3 * ((1 - r) * s);
      m.jac[0] = (P1 - P0) * (1 - s) + (P2 - P3) * s;
      m.jac[1] = (P3 - P0) * (1 - r) + (P2 - P1) * r;
      return m;
    }
    case CellShape::Prism: {
      // Triangle 0-1-2 at t=0, opposite triangle 3-4-5 at t=1
      // (cell_simplices.hpp).
      const double L0 = 1 - r - s;
      const Vec3 bot = c.p[0] * L0 + c.p[1] * r + c.p[2] * s;
      const Vec3 top = c.p[3] * L0 + c.p[4] * r + c.p[5] * s;
      m.x = bot * (1 - t) + top * t;
      m.jac[0] = (c.p[1] - c.p[0]) * (1 - t) + (c.p[4] - c.p[3]) * t;
      m.jac[1] = (c.p[2] - c.p[0]) * (1 - t) + (c.p[5] - c.p[3]) * t;
      m.jac[2] = top - bot;
      return m;
    }
    case CellShape::Hex: {
      // Corners 0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0) 4=(0,0,1) 5=(1,0,1)
      // 6=(1,1,1) 7=(0,1,1) -- "face 0-3, opposite face 4-7"
      // (cell_simplices.hpp), each face CCW as seen from outside.
      const Vec3& P0 = c.p[0]; const Vec3& P1 = c.p[1];
      const Vec3& P2 = c.p[2]; const Vec3& P3 = c.p[3];
      const Vec3& P4 = c.p[4]; const Vec3& P5 = c.p[5];
      const Vec3& P6 = c.p[6]; const Vec3& P7 = c.p[7];
      const double rm = 1 - r, sm = 1 - s, tm = 1 - t;
      m.x = P0 * (rm * sm * tm) + P1 * (r * sm * tm) + P2 * (r * s * tm) + P3 * (rm * s * tm) +
            P4 * (rm * sm * t) + P5 * (r * sm * t) + P6 * (r * s * t) + P7 * (rm * s * t);
      m.jac[0] = (P1 - P0) * (sm * tm) + (P2 - P3) * (s * tm) + (P5 - P4) * (sm * t) +
                (P6 - P7) * (s * t);
      m.jac[1] = (P3 - P0) * (rm * tm) + (P2 - P1) * (r * tm) + (P7 - P4) * (rm * t) +
                (P6 - P5) * (r * t);
      m.jac[2] = (P4 - P0) * (rm * sm) + (P5 - P1) * (r * sm) + (P6 - P2) * (r * s) +
                (P7 - P3) * (rm * s);
      return m;
    }
    default:
      return m;  // unreachable: shape_supported() gates every caller.
  }
}

/// Basis (shape function) values at natural coordinates `u`, one per corner,
/// in the same corner order `eval_map` uses -- always sums to 1 (a partition
/// of unity, affine or multilinear), which is what makes the resulting
/// interpolation exact for a constant and, at genuine corners, exact for the
/// field's own nodal values.
[[nodiscard]] inline std::array<double, 8> basis_weights(CellShape shape,
                                                          const std::array<double, 3>& u) {
  const double r = u[0], s = u[1], t = u[2];
  std::array<double, 8> w{};
  switch (shape) {
    case CellShape::Edge:
      w[0] = 1 - r; w[1] = r;
      return w;
    case CellShape::Triangle:
      w[0] = 1 - r - s; w[1] = r; w[2] = s;
      return w;
    case CellShape::Tet:
      w[0] = 1 - r - s - t; w[1] = r; w[2] = s; w[3] = t;
      return w;
    case CellShape::Quad:
      w[0] = (1 - r) * (1 - s); w[1] = r * (1 - s); w[2] = r * s; w[3] = (1 - r) * s;
      return w;
    case CellShape::Prism: {
      const double L0 = 1 - r - s, L1 = r, L2 = s;
      w[0] = L0 * (1 - t); w[1] = L1 * (1 - t); w[2] = L2 * (1 - t);
      w[3] = L0 * t;       w[4] = L1 * t;       w[5] = L2 * t;
      return w;
    }
    case CellShape::Hex: {
      const double rm = 1 - r, sm = 1 - s, tm = 1 - t;
      w[0] = rm * sm * tm; w[1] = r * sm * tm; w[2] = r * s * tm; w[3] = rm * s * tm;
      w[4] = rm * sm * t;  w[5] = r * sm * t;  w[6] = r * s * t;  w[7] = rm * s * t;
      return w;
    }
    default:
      return w;
  }
}

/// How far outside `shape`'s reference domain `u` lies, 0 if inside (within
/// `eps`) -- an L2-style violation measure used both to accept/reject a
/// candidate and to rank candidates for `ClampToBoundary`.
[[nodiscard]] inline double reference_violation(CellShape shape, const std::array<double, 3>& u,
                                                double eps) {
  const double r = u[0], s = u[1], t = u[2];
  const auto neg = [eps](double v) { return v < -eps ? -eps - v : 0.0; };
  double v = 0.0;
  switch (shape) {
    case CellShape::Edge: {
      const auto out01 = [eps](double x) {
        if (x < -eps) return (-eps - x);
        if (x > 1 + eps) return (x - 1 - eps);
        return 0.0;
      };
      const double a = out01(r);
      return a * a;
    }
    case CellShape::Triangle:
      v += neg(r) * neg(r) + neg(s) * neg(s);
      if (r + s > 1 + eps) v += (r + s - 1 - eps) * (r + s - 1 - eps);
      return v;
    case CellShape::Tet:
      v += neg(r) * neg(r) + neg(s) * neg(s) + neg(t) * neg(t);
      if (r + s + t > 1 + eps) {
        const double d = r + s + t - 1 - eps;
        v += d * d;
      }
      return v;
    case CellShape::Quad: {
      const auto out01 = [eps](double x) {
        if (x < -eps) return (-eps - x);
        if (x > 1 + eps) return (x - 1 - eps);
        return 0.0;
      };
      const double a = out01(r), b = out01(s);
      return a * a + b * b;
    }
    case CellShape::Prism: {
      v += neg(r) * neg(r) + neg(s) * neg(s);
      if (r + s > 1 + eps) v += (r + s - 1 - eps) * (r + s - 1 - eps);
      const double tc = t < -eps ? (-eps - t) : (t > 1 + eps ? (t - 1 - eps) : 0.0);
      return v + tc * tc;
    }
    case CellShape::Hex: {
      const auto out01 = [eps](double x) {
        if (x < -eps) return (-eps - x);
        if (x > 1 + eps) return (x - 1 - eps);
        return 0.0;
      };
      const double a = out01(r), b = out01(s), c = out01(t);
      return a * a + b * b + c * c;
    }
    default:
      return std::numeric_limits<double>::max();
  }
}

[[nodiscard]] inline bool in_reference_domain(CellShape shape, const std::array<double, 3>& u,
                                              double eps) {
  return reference_violation(shape, u, eps) <= eps;
}

/// Projects `u` onto `shape`'s reference domain by simple per-coordinate
/// clamping (tensor shapes) or barycentric renormalisation (simplex
/// shapes) -- used only under `InterfaceDomainPolicy::ClampToBoundary`.
[[nodiscard]] inline std::array<double, 3> clamp_to_domain(CellShape shape,
                                                           std::array<double, 3> u) {
  const auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };
  switch (shape) {
    case CellShape::Edge:
      return {clamp01(u[0]), 0.0, 0.0};
    case CellShape::Triangle: {
      double L0 = 1 - u[0] - u[1], L1 = u[0], L2 = u[1];
      L0 = std::max(0.0, L0); L1 = std::max(0.0, L1); L2 = std::max(0.0, L2);
      const double sum = L0 + L1 + L2;
      if (sum > 0) { L0 /= sum; L1 /= sum; L2 /= sum; }
      return {L1, L2, 0.0};
    }
    case CellShape::Tet: {
      double L0 = 1 - u[0] - u[1] - u[2], L1 = u[0], L2 = u[1], L3 = u[2];
      L0 = std::max(0.0, L0); L1 = std::max(0.0, L1);
      L2 = std::max(0.0, L2); L3 = std::max(0.0, L3);
      const double sum = L0 + L1 + L2 + L3;
      if (sum > 0) { L0 /= sum; L1 /= sum; L2 /= sum; L3 /= sum; }
      return {L1, L2, L3};
    }
    case CellShape::Quad:
      return {clamp01(u[0]), clamp01(u[1]), 0.0};
    case CellShape::Prism: {
      double L0 = 1 - u[0] - u[1], L1 = u[0], L2 = u[1];
      L0 = std::max(0.0, L0); L1 = std::max(0.0, L1); L2 = std::max(0.0, L2);
      const double sum = L0 + L1 + L2;
      if (sum > 0) { L0 /= sum; L1 /= sum; L2 /= sum; }
      return {L1, L2, clamp01(u[2])};
    }
    case CellShape::Hex:
      return {clamp01(u[0]), clamp01(u[1]), clamp01(u[2])};
    default:
      return u;
  }
}

/// Bounding-box diagonal of `c`'s own corners -- the length scale
/// `invert_reference_map`'s convergence tolerance and the surface-distance
/// check are both relative to.
[[nodiscard]] inline double cell_diameter(const Cell& c) {
  Vec3 lo = c.p[0], hi = c.p[0];
  for (std::size_t k = 1; k < c.n; ++k) {
    lo.x = std::min(lo.x, c.p[k].x); hi.x = std::max(hi.x, c.p[k].x);
    lo.y = std::min(lo.y, c.p[k].y); hi.y = std::max(hi.y, c.p[k].y);
    lo.z = std::min(lo.z, c.p[k].z); hi.z = std::max(hi.z, c.p[k].z);
  }
  return norm(hi - lo);
}

struct RefInverse {
  std::array<double, 3> u{};
  double residual = 0.0;       ///< |X(u) - target|, absolute
  int iterations = 0;
  bool converged = false;
};

/// `invert_reference_map`'s own step-convergence tolerance, in NATURAL
/// (reference) coordinate units -- the loop stops once one Newton step moves
/// `u` by less than this, so the returned `u` is itself accurate to about
/// this magnitude (better, in practice, for a well-conditioned cell: Newton
/// converges quadratically, so the step just before the one that trips this
/// test is typically already far smaller). This is the ONE number every
/// "how accurately do we know `u`" question in this file answers from,
/// rather than a second, independently chosen constant -- see
/// `kRefEpsTight`'s own comment for where it is reused.
inline constexpr double kNewtonTol = 1e-12;

/// Inverts `c`'s reference map for physical point `target` by damped
/// Gauss-Newton on the physical residual, starting from the reference
/// domain's own centre. For a 2-parameter (surface) shape the 3D residual is
/// minimised in a least-squares sense (normal equations, `J^T J` 2x2),
/// which is exactly "project onto the cell" for a point off a curved
/// surface; for a 3-parameter (volume) shape `J` is square and this is a
/// plain Newton step. An AFFINE shape (Triangle, Tet) has a Jacobian
/// independent of `u`, so it converges in exactly one iteration to
/// round-off.
///
/// Rejects `E-INTERP-DEGENERATE` if the (least-squares) Jacobian is singular
/// to working precision -- a genuinely degenerate (zero-measure) cell -- and
/// `E-INTERP-NEWTON` if `max_iter` is exhausted without the step norm or
/// residual falling under `tol` (relative to the cell's own diameter).
[[nodiscard]] inline Result<RefInverse> invert_reference_map(const Cell& c, Vec3 target,
                                                             int max_iter = 50,
                                                             double tol = kNewtonTol) {
  const int np = shape_nparams(c.shape);
  const double diam = std::max(cell_diameter(c), 1.0);
  const double abstol = tol * diam;

  std::array<double, 3> u{};
  switch (c.shape) {
    case CellShape::Triangle: u = {1.0 / 3, 1.0 / 3, 0.0}; break;
    case CellShape::Quad: u = {0.5, 0.5, 0.0}; break;
    case CellShape::Tet: u = {0.25, 0.25, 0.25}; break;
    case CellShape::Prism: u = {1.0 / 3, 1.0 / 3, 0.5}; break;
    case CellShape::Hex: u = {0.5, 0.5, 0.5}; break;
    default: break;
  }

  RefInverse out;
  for (int it = 0; it < max_iter; ++it) {
    const MapResult m = eval_map(c, u);
    const Vec3 res = target - m.x;
    const double res_norm = norm(res);
    out.u = u;
    out.residual = res_norm;
    out.iterations = it;
    if (res_norm <= abstol) { out.converged = true; return out; }

    // Normal-equations coefficients: A = J^T J (np x np, symmetric), rhs = J^T res.
    double A[3][3] = {};
    double rhs[3] = {};
    for (int a = 0; a < np; ++a) {
      rhs[a] = dot(m.jac[static_cast<std::size_t>(a)], res);
      for (int b = 0; b < np; ++b) {
        A[a][b] = dot(m.jac[static_cast<std::size_t>(a)], m.jac[static_cast<std::size_t>(b)]);
      }
    }
    // Gaussian elimination with partial pivoting, np <= 3.
    double du[3] = {};
    {
      double M[3][4];
      for (int a = 0; a < np; ++a) {
        for (int b = 0; b < np; ++b) M[a][b] = A[a][b];
        M[a][np] = rhs[a];
      }
      double max_diag = 0.0;
      for (int a = 0; a < np; ++a) max_diag = std::max(max_diag, std::fabs(A[a][a]));
      const double piv_tol = std::max(max_diag, 1.0) * 1e-14;
      bool singular = false;
      for (int col = 0; col < np; ++col) {
        int piv = col;
        for (int row = col + 1; row < np; ++row) {
          if (std::fabs(M[row][col]) > std::fabs(M[piv][col])) piv = row;
        }
        if (std::fabs(M[piv][col]) <= piv_tol) { singular = true; break; }
        if (piv != col) {
          for (int k = 0; k <= np; ++k) std::swap(M[col][k], M[piv][k]);
        }
        for (int row = 0; row < np; ++row) {
          if (row == col) continue;
          const double f = M[row][col] / M[col][col];
          for (int k = col; k <= np; ++k) M[row][k] -= f * M[col][k];
        }
      }
      if (singular) {
        return Failure{"E-INTERP-DEGENERATE",
                       "invert_reference_map: the cell's Jacobian is singular "
                       "to working precision (a zero-measure or degenerate "
                       "cell); the reference map cannot be inverted"};
      }
      for (int a = 0; a < np; ++a) du[a] = M[a][np] / M[a][a];
    }

    double step_norm2 = 0.0;
    for (int a = 0; a < np; ++a) { u[static_cast<std::size_t>(a)] += du[a]; step_norm2 += du[a] * du[a]; }
    if (std::sqrt(step_norm2) <= tol) {
      const MapResult m2 = eval_map(c, u);
      out.u = u;
      out.residual = norm(target - m2.x);
      out.converged = true;
      out.iterations = it + 1;
      return out;
    }
  }
  return Failure{"E-INTERP-NEWTON",
                 fmt::format("invert_reference_map: did not converge within {} "
                             "iterations (last residual {:.6g}, tolerance {:.3g})",
                             max_iter, out.residual, abstol)};
}

/// Relative-to-diameter tolerance a surface cell's (Triangle/Quad in a
/// mesh whose points are not exactly coplanar with it) projection residual
/// must fall within to be accepted as "on" the cell, rather than merely
/// nearest to it.
///
/// NOT A ROUNDING TOLERANCE -- a PHYSICAL one. Two independent meshes of the
/// SAME smooth curved surface (a cubed sphere at two different panel counts,
/// say) are each a piecewise-BILINEAR facet approximation of that surface,
/// and the two facetings do not share the same interior points: a target
/// vertex from one faceting sits, in general, slightly off the OTHER
/// faceting's nearby quad (the chordal deviation of a bilinear patch from
/// the true curved surface it approximates, second order in the cell's own
/// angular width -- at a coarse resolution, comparable to several percent of
/// the cell's own diameter, not to machine round-off). Rejecting a hit at a
/// rounding-scale tolerance here would reject most of a legitimately
/// contained coarse-mesh point as "off surface"; `0.1` accepts the genuine
/// faceting gap between two reasonable discretizations of one surface while
/// still rejecting a point that is actually far off the interface (the
/// `OffSurfaceRefusesByDefault` test below is off by several cell diameters,
/// not a fraction of one).
inline constexpr double kSurfaceTol = 0.1;

/// TIER (a): how far outside `shape`'s reference domain `u` may lie and
/// still count as a GENUINE containment -- one a cell is deterministically
/// preferred for over every other candidate, tie-broken by lowest source
/// global id only among cells that ALSO pass this tier.
///
/// DERIVED FROM `kNewtonTol`, NOT TUNED. `invert_reference_map` stops once a
/// Newton step moves `u` by less than `kNewtonTol`, so `u` for a point
/// GENUINELY inside the cell is itself known to about that precision. `1e3`
/// is a stated, fixed safety margin -- not fit against any test's outcome --
/// for two effects that are always present regardless of geometry: rounding
/// accumulated over up to `max_iter` (50) successive floating-point Newton
/// steps, and the conversion from "the LAST step was smaller than
/// `kNewtonTol`" to "the FINAL `u` differs from the true root by less than
/// `kNewtonTol`", which is exact only in the strict limit and picks up a
/// small, iteration-count-dependent factor in practice. Nothing about this
/// value depends on mesh resolution, curvature, or cell shape -- unlike
/// `kRefEpsLoose` below, it is a statement about FLOATING POINT, not about
/// geometry, which is exactly why a genuinely-outside point (even one just
/// past a shared mesh edge, per `kRefEpsLoose`'s own comment) reliably fails
/// it: that gap is geometric, five to six orders of magnitude larger than
/// floating-point round-off.
inline constexpr double kRefEpsTight = 1e3 * kNewtonTol;

/// TIER (b): the SECOND, and last, chance for a candidate that fails
/// `kRefEpsTight` -- accepted ONLY when NO candidate passes tier (a) at all,
/// and then only the LEAST-outside such candidate (never a global-id
/// tie-break: at this tier "outside" is a real geometric fact about every
/// candidate, not a tie between equally-valid ones, so the smallest
/// violation is preferred on its own terms).
///
/// EQUAL TO `kSurfaceTol`, DELIBERATELY, NOT AN INDEPENDENT CONSTANT: both
/// bound the SAME physical phenomenon -- the faceting gap between two
/// independent discretizations of one (possibly curved) interface -- along
/// two different axes of the same reference cell (`kSurfaceTol` is the
/// NORMAL/off-surface residual; this is the LATERAL/in-plane one). A
/// physical gap of relative size `kSurfaceTol` between two facetings
/// displaces a target point by comparably the same FRACTION of a cell's own
/// reference width in whichever direction it happens to fall, lateral or
/// normal, since reference coordinates are themselves a dimensionless
/// (0-to-1) rescaling of that same physical cell. Reusing the constant
/// rather than inventing a second one keeps that reasoning honest: changing
/// one changes both, and a reviewer checking "why 0.1" only has to read it
/// once.
inline constexpr double kRefEpsLoose = kSurfaceTol;

/// The piecewise-linear hat-function weight of the two nodes bracketing a
/// 1D point `target_x` (`lo_x` the LOW node's coordinate, `hi_x` the HIGH
/// node's, along whichever single axis a caller has already reduced its
/// geometry to) -- computed via `invert_reference_map`/`basis_weights` on a
/// synthetic, 2-corner `CellShape::Edge` cell collapsed onto that axis
/// (`y = z = 0` on both corners and the target), rather than the direct
/// ratio `(target_x - lo_x) / (hi_x - lo_x)`.
///
/// WHY THIS EXISTS (KNOWN_ISSUES.md D67). `consistent_interp.hpp`'s 1D
/// coordinate-sort algorithm (`build_consistent_interp_weights_1d`) used to
/// compute this exact ratio directly with that one division;
/// `build_distributed_interp_rows` (nodal_remap_distributed.hpp) and
/// `build_general_interp_weights` above compute the SAME physical quantity
/// for an Edge cell through `invert_reference_map`'s damped Gauss-Newton
/// instead -- one step from `u = 0`, landing at `du = dot(jac, res) /
/// dot(jac, jac)` with `jac = p1 - p0`, `res = target - p0` (exact to
/// round-off for an affine cell, per this file's own header comment on
/// why an affine shape converges in one step). The two expressions are
/// ALGEBRAICALLY identical (`du` reduces to the same ratio when `jac` is a
/// scalar `h = hi_x - lo_x` and `res` a scalar `dx = target_x - lo_x`:
/// `(h*dx)/(h*h)` versus `dx/h`) but NOT bit-identical: one extra multiply
/// before the divide rounds differently from a bare divide, in general. A
/// real host run measured exactly this 1-ULP gap at several bracketing
/// vertices of a 7-cell/11-cell pair (e.g. 0.06611570247933884 from the
/// direct ratio against 0.06611570247933886 from the Gauss-Newton step, at
/// the domain's own left edge) -- `NodalRemapDistributedNullComm.
/// D67SerialAndDistributedEdgeWeightsAgreeAtTheDomainBoundary`
/// (test/backend/nodal_remap_distributed_nullcomm_test.cpp) reproduces it
/// serially, with no MPI at all. Routing BOTH call sites through this one
/// function closes the gap: whichever code path a caller's placement
/// selects (`Runner::resolve_remap`'s `both_on_one_rank`, runner.cpp --
/// the SERIAL dispatch for a one-rank placement, or the DISTRIBUTED
/// `build_distributed_interp_rows` otherwise), the computed weight is now
/// the identical bit pattern, because it is now the identical computation.
///
/// Never fails in practice -- every caller rejects `hi_x == lo_x` (within
/// its own degeneracy tolerance) before reaching here, so the synthetic
/// Edge cell's Jacobian (`hi_x - lo_x`, nonzero) is never singular, and one
/// Newton step converges an affine map to round-off unconditionally -- but
/// still returns a `Result` and never calls `.value()` on an unchecked one
/// internally, matching this file's own "rejects, never throws" contract:
/// a future change to `invert_reference_map` that adds a genuine failure
/// mode here must surface as a `Failure` to the caller, not an abort.
[[nodiscard]] inline Result<std::pair<double, double>> edge_hat_weights(double lo_x, double hi_x,
                                                                        double target_x) {
  Cell c;
  c.shape = CellShape::Edge;
  c.n = 2;
  c.p[0] = Vec3{lo_x, 0.0, 0.0};
  c.p[1] = Vec3{hi_x, 0.0, 0.0};
  const auto inv = invert_reference_map(c, Vec3{target_x, 0.0, 0.0});
  if (!inv.ok()) {
    return Failure{inv.error().code, "edge_hat_weights: " + inv.error().message};
  }
  const std::array<double, 8> bw = basis_weights(CellShape::Edge, inv.value().u);
  return std::make_pair(bw[0], bw[1]);
}

}  // namespace general_interp_detail

/// Builds general consistent interpolation `H` for a source mesh whose
/// topological dimension is 2 or 3 -- see the file header for the full
/// algorithm. `src_nodes`'s dimension-0 entities are the DOFs (columns);
/// `tgt_points`'s dimension-`tgt_dim` entities are the evaluation points
/// (rows), exactly `consistent_interp.hpp`'s existing 1D builder's
/// convention.
///
/// Rejects, never throws:
///   * `E-INTERP-NOGEOMETRY` -- no source cells, or no target points.
///   * `E-MESH-NO-CELL-VERTICES` -- `src_nodes` (or the winning candidate
///     cell) has no cell-to-vertex connectivity (mesh_view.hpp's default).
///   * `E-INTERP-SHAPE` -- a source cell's shape has no standard nodal
///     basis (`Pyramid`, `Polygon`, `Polyhedron`, `Point`) -- checked for
///     EVERY source cell up front, not only ones a target point happens to
///     land near, so the rejection does not depend on the query set.
///   * `E-INTERP-DEGENERATE` / `E-INTERP-NEWTON` -- from
///     `invert_reference_map`.
///   * `E-INTERP-DOMAIN` -- (policy `Refuse`, the default) a target point is
///     not contained (or, for a surface source, not close enough to the
///     surface) by any candidate cell.
[[nodiscard]] inline Result<RemapWeights> build_general_interp_weights(
    const MeshView& src_nodes, const MeshView& tgt_points, int tgt_dim,
    InterfaceDomainPolicy policy = InterfaceDomainPolicy::Refuse) {
  using namespace general_interp_detail;

  const int src_dim = src_nodes.dimension();
  const std::size_t n_src_cells = src_nodes.entity_count(src_dim);
  const std::size_t n_src_verts = src_nodes.entity_count(0);
  if (n_src_cells == 0 || n_src_verts == 0) {
    return Failure{"E-INTERP-NOGEOMETRY",
                   "build_general_interp_weights: the source view reports no "
                   "cells or no vertices to interpolate from"};
  }
  const std::size_t n_tgt = tgt_points.entity_count(tgt_dim);
  if (n_tgt == 0) {
    return Failure{"E-INTERP-NOGEOMETRY",
                   fmt::format("build_general_interp_weights: the target view "
                               "reports no dimension-{} entities to evaluate "
                               "the interpolant at",
                               tgt_dim)};
  }

  // ---- Up front: every source cell's shape and corners, validated --------
  std::vector<Cell> cells(n_src_cells);
  for (std::size_t i = 0; i < n_src_cells; ++i) {
    const auto cv = src_nodes.cell_vertices(src_dim, i);
    if (!cv.ok()) {
      return Failure{cv.error().code,
                     fmt::format("build_general_interp_weights: source cell {} "
                                 "of dimension {}: {}",
                                 i, src_dim, cv.error().message)};
    }
    if (!shape_supported(cv.value().shape)) {
      const char* name = "unknown";
      switch (cv.value().shape) {
        case CellShape::Pyramid: name = "Pyramid"; break;
        case CellShape::Polygon: name = "Polygon"; break;
        case CellShape::Edge: name = "Edge"; break;
        default: break;
      }
      return Failure{
          "E-INTERP-SHAPE",
          fmt::format("build_general_interp_weights: source cell {} is a {}, "
                      "which has no standard nodal basis implemented here "
                      "(Pyramid: the standard basis is rational, not "
                      "polynomial, and is out of scope; Polygon/Polyhedron: "
                      "no standard nodal basis exists at all)",
                      i, name)};
    }
    Cell& c = cells[i];
    c.shape = cv.value().shape;
    c.n = cv.value().vertices.size();
    if (c.n != shape_corner_count(c.shape) || c.n > 8) {
      return Failure{"E-INTERP-SHAPE",
                     fmt::format("build_general_interp_weights: source cell {} "
                                 "reports {} corners for its shape, which needs "
                                 "{}",
                                 i, c.n, shape_corner_count(c.shape))};
    }
    for (std::size_t k = 0; k < c.n; ++k) {
      c.p[k] = to_vec(src_nodes.centroid(0, cv.value().vertices[k]));
    }
  }

  // ---- Broad phase: one box per source cell -------------------------------
  std::vector<std::size_t> owned_ids;
  // Padded by kRefEpsLoose, not the default (rounding-scale) tolerance: a
  // tier-(b) candidate's own reference-domain violation can be as large as
  // kRefEpsLoose, which -- for a well-shaped cell -- corresponds to roughly
  // the same FRACTION of physical displacement outside its box (reference
  // coordinates are a dimensionless rescaling of the cell's own physical
  // extent). A tighter box padding would silently drop a genuine tier-(b)
  // candidate from the broad-phase result before the narrow phase below ever
  // gets to classify it.
  auto boxes_r = search::element_boxes(src_nodes, owned_ids,
                                       search::Tolerance{.relative = kRefEpsLoose});
  if (!boxes_r.ok()) {
    return Failure{boxes_r.error().code,
                   "build_general_interp_weights: " + boxes_r.error().message};
  }
  auto idx_r = search::BoxIndex::build(boxes_r.value());
  if (!idx_r.ok()) return Failure{idx_r.error().code, idx_r.error().message};
  const search::BoxIndex& index = idx_r.value();

  std::vector<search::Point> qpts(n_tgt);
  for (std::size_t j = 0; j < n_tgt; ++j) {
    const Point p = tgt_points.centroid(tgt_dim, j);
    qpts[j] = search::Point{p.x, p.y, p.z};
  }
  auto hits_r = index.query_points(qpts);
  if (!hits_r.ok()) return Failure{hits_r.error().code, hits_r.error().message};
  const search::QueryResult& hits = hits_r.value();

  // A cheap fallback candidate set for ClampToBoundary when the padded-box
  // query finds nothing at all (the point is genuinely far outside every
  // cell): the k nearest cell boxes.
  const int k_nearest = static_cast<int>(std::min<std::size_t>(8, n_src_cells));

  RemapWeights w;
  w.n_rows = n_tgt;
  w.n_cols = n_src_verts;
  w.order = 2;
  std::vector<double> row_sum(n_tgt, 0.0);
  std::vector<double> col_sum(n_src_verts, 0.0);
  double min_weight = 1.0;

  for (std::size_t j = 0; j < n_tgt; ++j) {
    std::vector<std::size_t> candidates;
    for (std::size_t p = hits.offsets[j]; p < hits.offsets[j + 1]; ++p) {
      candidates.push_back(owned_ids[hits.indices[p]]);
    }
    if (candidates.empty() && policy == InterfaceDomainPolicy::ClampToBoundary) {
      auto nn = index.query_nearest({qpts[j]}, k_nearest);
      if (nn.ok()) {
        for (std::size_t p = nn.value().offsets[0]; p < nn.value().offsets[1]; ++p) {
          candidates.push_back(owned_ids[nn.value().indices[p]]);
        }
      }
    }

    const Vec3 target = to_vec(tgt_points.centroid(tgt_dim, j));

    // TWO-TIER SELECTION (see kRefEpsTight/kRefEpsLoose's own comments for
    // the derivation of each bound).
    //
    // Tier (a): a candidate that GENUINELY contains the point (within
    // floating-point round-off, kRefEpsTight) wins outright, tie-broken by
    // the lowest source global id ONLY among candidates that pass this tier
    // -- a true tie (a point exactly on a shared face/vertex), never a
    // contest between a genuine containment and an extrapolation. This is
    // load-bearing: the lowest-gid rule alone, applied at a loose tolerance,
    // would let a lower-gid NEIGHBOUR's cell win by EXTRAPOLATING past its
    // own boundary even when a higher-gid cell genuinely contains the point
    // -- `PointDeepInsideBIsNotStolenByALowerGidNeighbourExtrapolating`
    // below is exactly that failure mode, on a field that is not affine
    // across the shared edge (so extrapolation is visibly wrong, not
    // accidentally exact).
    //
    // Tier (b): only reached when NO candidate passes tier (a). The single
    // LEAST-outside candidate (by a raw, eps-independent violation score) is
    // accepted if it is within kRefEpsLoose/kSurfaceTol -- the genuine
    // faceting gap between two independent discretizations of a curved
    // interface, never a tie-break, since "outside" is a real geometric
    // fact about every tier-(b) candidate, not a contest between equals.
    //
    // Beyond both tiers: rejected (E-INTERP-DOMAIN) under the default
    // `Refuse` policy; under `ClampToBoundary`, the same least-outside
    // candidate is accepted anyway, with its natural coordinates clamped
    // into its own reference domain.
    bool have_tight = false;
    std::size_t tight_cell = 0;
    std::size_t tight_gid = 0;
    std::array<double, 3> tight_u{};

    bool have_fallback = false;
    std::size_t fallback_cell = 0;
    double fallback_score = std::numeric_limits<double>::max();
    std::array<double, 3> fallback_u{};
    bool fallback_passes_loose = false;

    for (const std::size_t ci : candidates) {
      const auto inv = invert_reference_map(cells[ci], target);
      if (!inv.ok()) {
        return Failure{inv.error().code,
                       fmt::format("build_general_interp_weights: target point "
                                   "{}, candidate cell {}: {}",
                                   j, ci, inv.error().message)};
      }
      const double diam = std::max(cell_diameter(cells[ci]), 1.0);
      const bool surface_ok = inv.value().residual <= kSurfaceTol * diam;
      const std::size_t gid = src_nodes.global_id(src_dim, ci);

      if (surface_ok && in_reference_domain(cells[ci].shape, inv.value().u, kRefEpsTight)) {
        if (!have_tight || gid < tight_gid) {
          have_tight = true;
          tight_cell = ci;
          tight_gid = gid;
          tight_u = inv.value().u;
        }
        continue;  // a tight hit is, by construction, never a worse fallback
                   // than itself, and tier (a) always wins when it exists.
      }

      // Raw (eps = 0), not `kRefEpsLoose`-relative: a genuine ranking needs
      // the ACTUAL distance outside the domain, not "0 once inside the loose
      // band" -- using the loose eps here would make every candidate that
      // merely CLEARS the loose bound read as equally "0 violation",
      // destroying the ranking `ClampToBoundary`'s own ordering needs.
      const double lateral = reference_violation(cells[ci].shape, inv.value().u, 0.0);
      const double normal_gap = surface_ok ? 0.0 : (inv.value().residual / diam);
      const double score = lateral + normal_gap * normal_gap;
      const bool passes_loose =
          surface_ok && in_reference_domain(cells[ci].shape, inv.value().u, kRefEpsLoose);
      if (!have_fallback || score < fallback_score) {
        have_fallback = true;
        fallback_cell = ci;
        fallback_score = score;
        fallback_u = inv.value().u;
        fallback_passes_loose = passes_loose;
      }
    }

    std::size_t use_cell;
    std::array<double, 3> use_u{};
    if (have_tight) {
      use_cell = tight_cell;
      use_u = tight_u;
    } else if (have_fallback && fallback_passes_loose) {
      use_cell = fallback_cell;
      use_u = fallback_u;
    } else if (policy == InterfaceDomainPolicy::ClampToBoundary && have_fallback) {
      use_cell = fallback_cell;
      use_u = clamp_to_domain(cells[fallback_cell].shape, fallback_u);
    } else {
      return Failure{
          "E-INTERP-DOMAIN",
          fmt::format("build_general_interp_weights: target point {} (at "
                      "({:.6g}, {:.6g}, {:.6g})) is not contained by any "
                      "source cell, even loosely (refusing rather than "
                      "extrapolating past the data that supports it)",
                      j, target.x, target.y, target.z)};
    }

    // REVIEW.md TF-10: the up-front loop above already validated this exact
    // call for every source cell, including `use_cell` -- but re-deriving
    // the same corner list here rather than keeping the up-front one is
    // the existing shape of this function, and `.value()` on an unchecked
    // Result is exactly the "Rejects, never throws" contract this function
    // documents forbidding. A latent re-query failure (this `MeshView`
    // returning a different, failing answer for an index it already
    // answered once) must still be a Failure, not a std::bad_variant_access
    // escaping uncaught.
    const auto cv = src_nodes.cell_vertices(src_dim, use_cell);
    if (!cv.ok()) {
      return Failure{cv.error().code,
                     fmt::format("build_general_interp_weights: re-querying "
                                 "source cell {}'s corners (already validated "
                                 "above) failed: {}",
                                 use_cell, cv.error().message)};
    }
    const std::array<double, 8> bw = basis_weights(cells[use_cell].shape, use_u);
    for (std::size_t k = 0; k < cells[use_cell].n; ++k) {
      const std::size_t col = cv.value().vertices[k];
      const double v = bw[k];
      w.row.push_back(static_cast<std::int32_t>(j));
      w.col.push_back(static_cast<std::int32_t>(col));
      w.val.push_back(v);
      row_sum[j] += v;
      col_sum[col] += v;
      min_weight = std::min(min_weight, v);
    }
  }

  for (const double s : row_sum) w.max_row_defect = std::max(w.max_row_defect, std::abs(s - 1.0));
  // Unweighted column-sum residual, diagnostic only -- see
  // consistent_interp.hpp's own comment on `max_col_defect` for why this is
  // not what makes `OperatorClaims::conservative` true (a vertex has no
  // measure to weight a "conservative" column sum by).
  for (const double s : col_sum) w.max_col_defect = std::max(w.max_col_defect, std::abs(s - 1.0));
  // REVIEW.md TF-16: the comment above was never enforced structurally --
  // `WeightsOperator::claims()` read `max_col_defect` directly regardless
  // of which builder produced it. Say so on the matrix itself now.
  w.col_defect_is_measure_weighted = false;
  w.min_weight = min_weight;
  w.monotone = (w.min_weight >= -1e-12) && (w.max_row_defect <= 1e-8);
  w.src_measure_total = 0.0;
  w.tgt_measure_total = 0.0;
  w.min_src_coverage = 1.0;
  w.min_tgt_coverage = 1.0;
  return w;
}

}  // namespace fuse
