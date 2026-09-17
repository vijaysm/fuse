// Lowest-order conservative remap on genuinely NON-MATCHING meshes -- the
// capability the default build did not have. Every expectation here is either
// a closed form or a conservation identity, so a failure names the defect
// rather than a tolerance.

#include "fuse/backend/supermesh.hpp"

#include <memory>
#include <string>
#include <cmath>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/serial_backend.hpp"

namespace {

struct Pair {
  fuse::SerialMeshBackend backend;
  const fuse::MeshView*   src = nullptr;
  const fuse::MeshView*   tgt = nullptr;
};

/// Two generated meshes over the same unit domain, at different resolutions.
[[nodiscard]] std::unique_ptr<Pair> pair_of(const std::string& s,
                                            const std::string& t) {
  auto p = std::make_unique<Pair>();
  if (!p->backend.load({.name = "s", .source = s}).ok()) return nullptr;
  if (!p->backend.load({.name = "t", .source = t}).ok()) return nullptr;
  const auto sid = p->backend.mesh_id("s");
  const auto tid = p->backend.mesh_id("t");
  if (!sid.ok() || !tid.ok()) return nullptr;
  const auto sv = p->backend.view(sid.value());
  const auto tv = p->backend.view(tid.value());
  if (!sv.ok() || !tv.ok()) return nullptr;
  p->src = sv.value();
  p->tgt = tv.value();
  return p;
}

/// Applies the map: g = W f.
[[nodiscard]] std::vector<double> apply_map(const fuse::RemapWeights& w,
                                       const std::vector<double>& f) {
  std::vector<double> g(w.n_rows, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    g[static_cast<std::size_t>(w.row[k])] +=
        w.val[k] * f[static_cast<std::size_t>(w.col[k])];
  }
  return g;
}

}  // namespace

// ---- 1D, non-matching --------------------------------------------------

TEST(Supermesh, OneDimensionalNonMatchingIsExactlyConservative) {
  // 8 source cells onto 5 target cells over [0,1]: no target boundary
  // coincides with a source boundary except at the ends, so nearly every
  // weight is a partial overlap.
  auto p = pair_of("generated:line:8", "generated:line:5");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 1);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  EXPECT_EQ(w.n_rows, 5u);
  EXPECT_EQ(w.n_cols, 8u);
  EXPECT_GT(w.nnz(), 5u) << "a non-matching pair must have split cells";

  // The domains coincide, so both coverages are 1 and both residuals vanish.
  EXPECT_NEAR(w.min_src_coverage, 1.0, 1e-14);
  EXPECT_NEAR(w.min_tgt_coverage, 1.0, 1e-14);
  EXPECT_LT(w.max_row_defect, 1e-14) << "constants are not preserved";
  EXPECT_LT(w.max_col_defect, 1e-14) << "mass is not conserved";
  EXPECT_NEAR(w.src_measure_total, 1.0, 1e-14);
  EXPECT_NEAR(w.tgt_measure_total, 1.0, 1e-14);
  EXPECT_GE(w.min_weight, 0.0);
  EXPECT_TRUE(w.monotone) << "all weights positive and rows summing to one";
  EXPECT_EQ(w.order, 1);
}

TEST(Supermesh, AConstantSurvivesAndTotalMassIsPreserved) {
  auto p = pair_of("generated:line:8", "generated:line:5");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 1);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  // A constant must come back as the same constant in every target cell. This
  // is the property a first-order map is exact for, so the tolerance is
  // rounding and not discretisation.
  const std::vector<double> f(w.n_cols, 3.75);
  const std::vector<double> g = apply_map(w, f);
  for (std::size_t j = 0; j < g.size(); ++j) {
    EXPECT_NEAR(g[j], 3.75, 1e-14) << "target cell " << j;
  }

  // A varying field is NOT reproduced pointwise at order 1, but its integral
  // must still be preserved exactly -- that is the difference between
  // accuracy and conservation, and only the second is claimed here.
  std::vector<double> ramp(w.n_cols);
  for (std::size_t i = 0; i < ramp.size(); ++i) {
    ramp[i] = 1.0 + 2.0 * static_cast<double>(i);
  }
  const std::vector<double> out = apply_map(w, ramp);
  double m_in = 0.0, m_out = 0.0;
  for (std::size_t i = 0; i < ramp.size(); ++i) {
    m_in += ramp[i] * p->src->measure(1, i);
  }
  for (std::size_t j = 0; j < out.size(); ++j) {
    m_out += out[j] * p->tgt->measure(1, j);
  }
  EXPECT_NEAR(m_out, m_in, 1e-13 * m_in) << "integral not conserved";
}

// ---- 3D, non-matching, through the clipping kernel ---------------------

TEST(Supermesh, ThreeDimensionalNonMatchingIsExactlyConservative) {
  // 3x3x3 onto 4x4x4 over the unit cube: 27 hexahedra as 162 tetrahedra
  // against 64 hexahedra as 384, and no interior face plane is shared.
  auto p = pair_of("generated:box:3x3x3", "generated:box:4x4x4");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 3);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  EXPECT_EQ(w.n_rows, 64u);
  EXPECT_EQ(w.n_cols, 27u);
  EXPECT_NEAR(w.src_measure_total, 1.0, 1e-13) << "the unit cube";
  EXPECT_NEAR(w.tgt_measure_total, 1.0, 1e-13);

  // The exactness claim: these are clipped polyhedra and analytic moments, so
  // the residuals are at rounding, not at a geometric tolerance.
  EXPECT_LT(w.max_row_defect, 1e-13) << "constants not preserved";
  EXPECT_LT(w.max_col_defect, 1e-13) << "mass not conserved";
  EXPECT_NEAR(w.min_src_coverage, 1.0, 1e-13);
  EXPECT_NEAR(w.min_tgt_coverage, 1.0, 1e-13);
  EXPECT_GE(w.min_weight, 0.0) << "a negative volume escaped the clipper";
  EXPECT_TRUE(w.monotone);

  // And a constant is reproduced in three dimensions too.
  const std::vector<double> f(w.n_cols, -2.5);
  for (const double g : apply_map(w, f)) EXPECT_NEAR(g, -2.5, 1e-13);
}

TEST(Supermesh, RemappingOntoTheSameMeshIsTheIdentity) {
  // The sharpest available check on the kernel: every target cell must draw
  // from exactly one source cell, with weight exactly 1. A clipper that were
  // slightly wrong would show up as extra entries with tiny weights.
  auto p = pair_of("generated:box:3x3x3", "generated:box:3x3x3");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 3);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  EXPECT_EQ(w.nnz(), 27u) << "an identity map has one entry per cell";
  for (std::size_t k = 0; k < w.nnz(); ++k) {
    EXPECT_EQ(w.row[k], w.col[k]) << "entry " << k << " is off-diagonal";
    EXPECT_NEAR(w.val[k], 1.0, 1e-13);
  }
}

// ---- refusals ----------------------------------------------------------

TEST(Supermesh, RefusesADimensionItDoesNotDefine) {
  auto p = pair_of("generated:line:4", "generated:line:4");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 0);
  ASSERT_FALSE(r.ok()) << "dimension 0 entities have no measure to divide by";
  EXPECT_EQ(r.error().code, "E-REMAP-DIM");
}

TEST(Supermesh, RefusesASurfacePairRatherThanProjectingIt) {
  // The boundary skin of a box is a 2-manifold in three-space. The planar
  // kernel would clip its xy shadow and return a plausible wrong number, so
  // it must refuse. This is the guard that keeps "always available" from
  // quietly meaning "always answers".
  auto p = pair_of("generated:box:2x2x2", "generated:box:3x3x3");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 2);
  ASSERT_FALSE(r.ok()) << "a projected shadow measure would look legitimate";
  EXPECT_EQ(r.error().code, "E-REMAP-NONPLANAR");
}

// ---- 2D, planar triangulations -----------------------------------------
//
// The mesh generators produce lines and boxes, so nothing above reaches the
// 2D kernel: the mutation that removes its winding normalisation survived,
// which is how the gap was found. A planar triangulation is a real case
// (2D CFD, planar FEM), so it gets a view here rather than the branch being
// deleted.

namespace {

/// An n-by-n grid of the unit square, each quad split into two triangles.
/// Cells are the triangles, so every cell is already a simplex.
///
/// `wind_clockwise` deliberately emits the corners in the opposite order. The
/// MeshView contract promises nothing about orientation, so the kernel must
/// normalise it; with this flag a test can prove it does rather than assume.
class PlanarTriangulation final : public fuse::MeshView {
 public:
  PlanarTriangulation(std::size_t n, bool wind_clockwise)
      : n_(n), cw_(wind_clockwise) {}

  [[nodiscard]] int dimension() const override { return 2; }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    return dim == 2 ? 2 * n_ * n_ : (dim == 0 ? (n_ + 1) * (n_ + 1) : 0);
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override {
    return entity_count(dim);
  }
  [[nodiscard]] bool is_owned(int, std::size_t) const override { return true; }
  [[nodiscard]] std::size_t global_id(int, std::size_t i) const override {
    return i;
  }
  [[nodiscard]] fuse::Point centroid(int dim, std::size_t i) const override {
    fuse::Point c[3];
    simplex_points(dim, i, 0, c);
    return fuse::Point{(c[0].x + c[1].x + c[2].x) / 3.0,
                       (c[0].y + c[1].y + c[2].y) / 3.0, 0.0};
  }
  [[nodiscard]] double measure(int dim, std::size_t i) const override {
    fuse::Point c[3];
    simplex_points(dim, i, 0, c);
    // Unsigned: half the magnitude of the cross product, so the measure does
    // not depend on the winding either.
    return 0.5 * std::abs((c[1].x - c[0].x) * (c[2].y - c[0].y) -
                          (c[2].x - c[0].x) * (c[1].y - c[0].y));
  }
  [[nodiscard]] std::size_t simplex_count(int dim, std::size_t) const override {
    return dim == 2 ? 1 : 0;
  }
  void simplex_points(int, std::size_t i, std::size_t,
                      fuse::Point* out) const override {
    const std::size_t quad = i / 2;
    const bool        upper = (i % 2) == 1;
    const double h = 1.0 / static_cast<double>(n_);
    const double x0 = static_cast<double>(quad % n_) * h;
    const double y0 = static_cast<double>(quad / n_) * h;
    // Lower triangle (x0,y0)-(x1,y0)-(x1,y1); upper (x0,y0)-(x1,y1)-(x0,y1).
    const double xs[3] = {x0, upper ? x0 + h : x0 + h, upper ? x0 : x0 + h};
    const double ys[3] = {y0, upper ? y0 + h : y0, upper ? y0 + h : y0 + h};
    for (int k = 0; k < 3; ++k) {
      const int src = cw_ ? (2 - k) : k;   // reversed order is clockwise
      out[k] = fuse::Point{xs[src], ys[src], 0.0};
    }
  }

 private:
  std::size_t n_;
  bool        cw_;
};

}  // namespace

TEST(Supermesh, TwoDimensionalNonMatchingIsExactlyConservative) {
  const PlanarTriangulation src(3, false);
  const PlanarTriangulation tgt(4, false);
  const auto r = fuse::build_supermesh_weights(src, tgt, 2);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  EXPECT_EQ(w.n_cols, 18u) << "3x3 quads, two triangles each";
  EXPECT_EQ(w.n_rows, 32u);
  EXPECT_NEAR(w.src_measure_total, 1.0, 1e-14) << "the unit square";
  EXPECT_NEAR(w.tgt_measure_total, 1.0, 1e-14);
  EXPECT_LT(w.max_row_defect, 1e-13) << "constants not preserved";
  EXPECT_LT(w.max_col_defect, 1e-13) << "mass not conserved";
  EXPECT_GE(w.min_weight, 0.0);
  EXPECT_TRUE(w.monotone);

  const std::vector<double> f(w.n_cols, 7.25);
  for (const double g : apply_map(w, f)) EXPECT_NEAR(g, 7.25, 1e-13);
}

TEST(Supermesh, WindingOrderOfTheInputDoesNotChangeTheAnswer) {
  // The same two meshes, but every triangle handed over clockwise. MeshView
  // requires no orientation, so the two maps must be identical -- and r2d
  // documents counterclockwise input, so something has to normalise it.
  const PlanarTriangulation ccw_s(3, false), ccw_t(4, false);
  const PlanarTriangulation cw_s(3, true),   cw_t(4, true);

  const auto a = fuse::build_supermesh_weights(ccw_s, ccw_t, 2);
  const auto b = fuse::build_supermesh_weights(cw_s, cw_t, 2);
  ASSERT_TRUE(a.ok()) << a.error().code << ": " << a.error().message;
  ASSERT_TRUE(b.ok()) << b.error().code << ": " << b.error().message;

  ASSERT_EQ(a.value().nnz(), b.value().nnz())
      << "winding changed the sparsity pattern";
  for (std::size_t k = 0; k < a.value().nnz(); ++k) {
    EXPECT_EQ(a.value().row[k], b.value().row[k]);
    EXPECT_EQ(a.value().col[k], b.value().col[k]);
    EXPECT_NEAR(a.value().val[k], b.value().val[k], 1e-14)
        << "entry " << k << " differs with the input winding";
  }
  EXPECT_LT(b.value().max_col_defect, 1e-13)
      << "a clockwise mesh lost mass, so its triangles clipped wrongly";
}

TEST(Supermesh, EqualCountEqualMeasureButDifferentArrangementIsNotTheIdentity) {
  // Eight cells of measure 0.125 on both sides, arranged completely
  // differently: a 2x2x2 block against a 8x1x1 column. Before the supermesh
  // existed this pair was REFUSED, because the only available map was an
  // identity and applying one here would have been silently wrong -- same
  // count, same measures, no correspondence.
  //
  // It is now computed rather than detected. The assertion that matters is
  // that the result is emphatically not an identity, and is conservative.
  auto p = pair_of("generated:box:2x2x2", "generated:box:8x1x1");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 3);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  ASSERT_EQ(w.n_rows, 8u);
  ASSERT_EQ(w.n_cols, 8u);
  EXPECT_GT(w.nnz(), 8u)
      << "an 8x1x1 column crosses several 2x2x2 cells, so rows must be split; "
         "exactly 8 entries would mean an identity had been applied";

  // Each target column cell spans x in [k/8, (k+1)/8] and the full y,z extent,
  // so it draws from the four source cells at its x, never from one.
  std::vector<std::size_t> per_row(w.n_rows, 0);
  for (std::size_t k = 0; k < w.nnz(); ++k) {
    ++per_row[static_cast<std::size_t>(w.row[k])];
  }
  for (std::size_t j = 0; j < w.n_rows; ++j) {
    EXPECT_EQ(per_row[j], 4u) << "target cell " << j
                              << " should straddle four source cells";
  }

  EXPECT_LT(w.max_row_defect, 1e-13);
  EXPECT_LT(w.max_col_defect, 1e-13);
  const std::vector<double> f(w.n_cols, 1.5);
  for (const double g : apply_map(w, f)) EXPECT_NEAR(g, 1.5, 1e-13);
}

// ---- the analytic oracle -----------------------------------------------
//
// Conservation and constant preservation are NECESSARY but not SUFFICIENT: a
// map that sent mass to the wrong cells could still sum correctly along both
// axes. Nothing above would notice. For two axis-aligned grids the exact
// intersection volume is a closed form -- the product of the per-axis overlaps
// -- so every individual weight can be checked against it rather than only
// the sums.

namespace {

/// Exact overlap of cell `a` of an `na`-cell uniform partition of [0,1] with
/// cell `b` of an `nb`-cell one.
[[nodiscard]] double axis_overlap(std::size_t a, std::size_t na, std::size_t b,
                                  std::size_t nb) {
  const double lo = std::max(static_cast<double>(a) / static_cast<double>(na),
                             static_cast<double>(b) / static_cast<double>(nb));
  const double hi = std::min((static_cast<double>(a) + 1.0) / static_cast<double>(na),
                             (static_cast<double>(b) + 1.0) / static_cast<double>(nb));
  return std::max(0.0, hi - lo);
}

/// The dense weight matrix a box-to-box remap must produce, from closed form.
[[nodiscard]] std::vector<double> analytic_box_weights(std::size_t ns,
                                                       std::size_t nt,
                                                       int dim) {
  const std::size_t n_src = dim == 1 ? ns : (dim == 3 ? ns * ns * ns : 0);
  const std::size_t n_tgt = dim == 1 ? nt : (dim == 3 ? nt * nt * nt : 0);
  std::vector<double> w(n_tgt * n_src, 0.0);
  const double tgt_measure =
      dim == 1 ? 1.0 / static_cast<double>(nt)
               : 1.0 / (static_cast<double>(nt) * static_cast<double>(nt) *
                        static_cast<double>(nt));
  for (std::size_t j = 0; j < n_tgt; ++j) {
    for (std::size_t i = 0; i < n_src; ++i) {
      double vol = 1.0;
      if (dim == 1) {
        vol = axis_overlap(i, ns, j, nt);
      } else {
        // The structured backend numbers cells x fastest, then y, then z.
        const std::size_t si[3] = {i % ns, (i / ns) % ns, i / (ns * ns)};
        const std::size_t tj[3] = {j % nt, (j / nt) % nt, j / (nt * nt)};
        for (int k = 0; k < 3; ++k) {
          vol *= axis_overlap(si[k], ns, tj[k], nt);
        }
      }
      w[j * n_src + i] = vol / tgt_measure;
    }
  }
  return w;
}

/// Compares a built map against the closed form, entry by entry, in BOTH
/// directions: every built entry must match, and every non-negligible
/// analytic entry must be present.
void expect_matches_analytic(const fuse::RemapWeights& w, std::size_t ns,
                             std::size_t nt, int dim, double tol) {
  const std::vector<double> ref = analytic_box_weights(ns, nt, dim);
  ASSERT_EQ(ref.size(), w.n_rows * w.n_cols);

  std::vector<double> built(w.n_rows * w.n_cols, 0.0);
  for (std::size_t k = 0; k < w.nnz(); ++k) {
    built[static_cast<std::size_t>(w.row[k]) * w.n_cols +
          static_cast<std::size_t>(w.col[k])] = w.val[k];
  }

  double worst = 0.0;
  std::size_t missing = 0, spurious = 0;
  for (std::size_t idx = 0; idx < ref.size(); ++idx) {
    worst = std::max(worst, std::abs(built[idx] - ref[idx]));
    if (ref[idx] > 1e-10 && built[idx] == 0.0) ++missing;
    if (ref[idx] == 0.0 && built[idx] > 0.0) ++spurious;
  }
  EXPECT_EQ(missing, 0u) << "the map omits overlaps the geometry has";
  EXPECT_EQ(spurious, 0u) << "the map invents overlaps the geometry does not "
                             "have -- mass is going to the wrong cells";
  EXPECT_LT(worst, tol) << "largest entrywise deviation from closed form";
  std::printf("[ oracle   ] dim=%d %zu->%zu: worst entry error %.3e over %zu "
              "entries\n", dim, ns, nt, worst, w.nnz());
}

}  // namespace

TEST(Supermesh, OneDimensionalWeightsMatchTheClosedForm) {
  // Coprime counts, so almost no target boundary lands on a source boundary
  // and nearly every entry is a partial overlap.
  auto p = pair_of("generated:line:7", "generated:line:11");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 1);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  expect_matches_analytic(r.value(), 7, 11, 1, 1e-14);
}

TEST(Supermesh, ThreeDimensionalWeightsMatchTheClosedForm) {
  // The strongest check on the clipping kernel available here: 27 hexahedra as
  // 162 tetrahedra against 64 as 384, and every one of the resulting weights
  // compared against the tensor product of the per-axis overlaps.
  auto p = pair_of("generated:box:3x3x3", "generated:box:4x4x4");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 3);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  expect_matches_analytic(r.value(), 3, 4, 3, 1e-13);
}

TEST(Supermesh, CoprimeThreeDimensionalGridsMatchTheClosedForm) {
  // 5 against 7 per axis: 125 against 343 cells, no shared interior planes at
  // all, and many thin slivers -- the regime where a clipper's tolerances
  // would show up if it had any.
  auto p = pair_of("generated:box:5x5x5", "generated:box:7x7x7");
  ASSERT_TRUE(p != nullptr);
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 3);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  expect_matches_analytic(r.value(), 5, 7, 3, 1e-13);
  EXPECT_LT(r.value().max_col_defect, 1e-13);
}

// ---- affine invariance --------------------------------------------------
//
// Every check above uses axis-aligned boxes, because that is where a closed
// form exists. An axis-aligned assumption anywhere in the kernel would
// therefore be invisible. This one has no closed form and does not need one:
// applying the SAME affine map to both meshes scales every intersection
// measure and every cell measure by the same |det|, so the weights -- which
// are ratios of the two -- must come out bit-for-bit unchanged.

namespace {

/// A view whose geometry is that of `base` after an affine map. Only the
/// coordinates change; connectivity, counts and ids are the base's.
class AffineView final : public fuse::MeshView {
 public:
  AffineView(const fuse::MeshView& base, const double m[3][3],
             const double t[3])
      : base_(base) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) m_[r][c] = m[r][c];
      t_[r] = t[r];
    }
    det_ = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  }

  [[nodiscard]] int dimension() const override { return base_.dimension(); }
  [[nodiscard]] std::size_t entity_count(int d) const override {
    return base_.entity_count(d);
  }
  [[nodiscard]] std::size_t owned_count(int d) const override {
    return base_.owned_count(d);
  }
  [[nodiscard]] bool is_owned(int d, std::size_t i) const override {
    return base_.is_owned(d, i);
  }
  [[nodiscard]] std::size_t global_id(int d, std::size_t i) const override {
    return base_.global_id(d, i);
  }
  [[nodiscard]] fuse::Point centroid(int d, std::size_t i) const override {
    return map(base_.centroid(d, i));
  }
  /// A volume scales by |det| under an affine map. Exact, not approximate.
  [[nodiscard]] double measure(int d, std::size_t i) const override {
    return base_.measure(d, i) * std::abs(det_);
  }
  [[nodiscard]] std::size_t simplex_count(int d, std::size_t i) const override {
    return base_.simplex_count(d, i);
  }
  void simplex_points(int d, std::size_t i, std::size_t s,
                      fuse::Point* out) const override {
    base_.simplex_points(d, i, s, out);
    for (int k = 0; k <= d; ++k) out[k] = map(out[k]);
  }

 private:
  [[nodiscard]] fuse::Point map(const fuse::Point& p) const {
    return fuse::Point{
        m_[0][0] * p.x + m_[0][1] * p.y + m_[0][2] * p.z + t_[0],
        m_[1][0] * p.x + m_[1][1] * p.y + m_[1][2] * p.z + t_[1],
        m_[2][0] * p.x + m_[2][1] * p.y + m_[2][2] * p.z + t_[2]};
  }
  const fuse::MeshView& base_;
  double                m_[3][3]{};
  double                t_[3]{};
  double                det_ = 1.0;
};

}  // namespace

TEST(Supermesh, WeightsAreUnchangedByACommonAffineMap) {
  auto p = pair_of("generated:box:3x3x3", "generated:box:4x4x4");
  ASSERT_TRUE(p != nullptr);
  const auto plain = fuse::build_supermesh_weights(*p->src, *p->tgt, 3);
  ASSERT_TRUE(plain.ok()) << plain.error().code << ": " << plain.error().message;

  // A rotation about z composed with a shear and a translation: no face of
  // either mesh is axis-aligned afterwards, and the two meshes are no longer
  // in the unit cube at all.
  const double c = std::cos(0.7), s = std::sin(0.7);
  const double m[3][3] = {{c, -s, 0.3}, {s, c, 0.0}, {0.0, 0.2, 1.1}};
  const double t[3] = {-2.5, 7.25, 0.125};
  const AffineView asrc(*p->src, m, t);
  const AffineView atgt(*p->tgt, m, t);

  const auto moved = fuse::build_supermesh_weights(asrc, atgt, 3);
  ASSERT_TRUE(moved.ok()) << moved.error().code << ": " << moved.error().message;

  ASSERT_EQ(plain.value().nnz(), moved.value().nnz())
      << "the affine map changed the sparsity pattern, so some test in the "
         "kernel depends on the coordinate axes";
  double worst = 0.0;
  for (std::size_t k = 0; k < plain.value().nnz(); ++k) {
    ASSERT_EQ(plain.value().row[k], moved.value().row[k]);
    ASSERT_EQ(plain.value().col[k], moved.value().col[k]);
    worst = std::max(worst, std::abs(plain.value().val[k] - moved.value().val[k]));
  }
  // Not bitwise: the coordinates themselves are different numbers, so the
  // arithmetic rounds differently. Rounding is all that may differ.
  EXPECT_LT(worst, 1e-13) << "weights moved under an affine map";
  std::printf("[ affine   ] worst weight change %.3e\n", worst);
  EXPECT_LT(moved.value().max_col_defect, 1e-13);
}

// ---- the observed order of accuracy ------------------------------------

TEST(Supermesh, ObservedOrderOnANonMatchingPairIsAtLeastOne) {
  // Remap the exact cell averages of a smooth field from an n-cell mesh onto a
  // non-matching one, and measure how the error falls under refinement. Order
  // 1 is what a piecewise-constant reconstruction claims; this measures what
  // it delivers, which is the only way the claim means anything.
  //
  // The ratio 11/7 is deliberately not an integer: with a nested pair every
  // target cell would be a union of whole source cells and the map would be
  // exact, measuring nothing.
  //
  // ONE DIMENSION ONLY, and deliberately. The candidate search is currently a
  // brute-force box test, so a 3D study over five refinements would be hours:
  // the 5x5x5 -> 7x7x7 case alone is 1.5 million clip calls. The 3D kernel is
  // instead pinned against the closed-form tensor product above, which is a
  // stronger statement about the geometry than a convergence slope is. A 3D
  // study becomes affordable once the spatial index lands.
  const auto exact_mean = [](double a, double b) {
    // Mean of sin(2 pi x) over [a, b], in closed form.
    constexpr double two_pi = 6.283185307179586476925287;
    return (std::cos(two_pi * a) - std::cos(two_pi * b)) / (two_pi * (b - a));
  };

  std::vector<double> hs, errs;
  for (const std::size_t ns : {14u, 28u, 56u, 112u, 224u}) {
    const std::size_t nt = ns * 11 / 7;
    auto p = pair_of("generated:line:" + std::to_string(ns),
                     "generated:line:" + std::to_string(nt));
    ASSERT_TRUE(p != nullptr);
    const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 1);
    ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
    const fuse::RemapWeights& w = r.value();

    std::vector<double> f(w.n_cols);
    for (std::size_t i = 0; i < w.n_cols; ++i) {
      f[i] = exact_mean(static_cast<double>(i) / static_cast<double>(ns),
                        static_cast<double>(i + 1) / static_cast<double>(ns));
    }
    const std::vector<double> g = apply_map(w, f);

    double l2 = 0.0;
    for (std::size_t j = 0; j < w.n_rows; ++j) {
      const double want =
          exact_mean(static_cast<double>(j) / static_cast<double>(nt),
                     static_cast<double>(j + 1) / static_cast<double>(nt));
      const double e = g[j] - want;
      l2 += e * e * p->tgt->measure(1, j);
    }
    hs.push_back(1.0 / static_cast<double>(ns));
    errs.push_back(std::sqrt(l2));
  }

  // Least-squares slope of log(error) against log(h).
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = static_cast<double>(hs.size());
  for (std::size_t k = 0; k < hs.size(); ++k) {
    const double x = std::log(hs[k]), y = std::log(errs[k]);
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const double order = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  std::printf("[ order    ] observed %.4f   finest L2 %.4e\n", order,
              errs.back());
  for (std::size_t k = 0; k < hs.size(); ++k) {
    std::printf("[ order    ]   h=%.5f  L2=%.6e\n", hs[k], errs[k]);
  }

  // At least first order, which is the claim. The upper bound is there because
  // a slope far above 2 would mean the errors are at rounding rather than at
  // discretisation, so the study would be measuring nothing.
  EXPECT_GT(order, 0.9) << "the map does not deliver the order it claims";
  EXPECT_LT(order, 3.0) << "suspiciously high: is the pair accidentally nested?";
}

// ---- partial overlap ----------------------------------------------------

TEST(Supermesh, PartialOverlapIsReportedRatherThanHidden) {
  // A target mesh covering twice the source domain in x. Half of every
  // straddling target cell has no source, so the map CANNOT preserve a
  // constant there -- and the honest behaviour is to say so in the audit
  // rather than rescale the rows to hide it.
  auto p = pair_of("generated:box:4x4x4", "generated:box:4x4x4");
  ASSERT_TRUE(p != nullptr);
  const double id[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const double shift[3] = {0.5, 0.0, 0.0};   // half a domain to the right
  const AffineView moved_tgt(*p->tgt, id, shift);

  const auto r = fuse::build_supermesh_weights(*p->src, moved_tgt, 3);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  // Coverage is well below 1 on both sides, and that is the number a caller
  // checks before trusting the map.
  EXPECT_LT(w.min_tgt_coverage, 0.01) << "the outermost target cells are "
                                          "entirely outside the source";
  EXPECT_LT(w.min_src_coverage, 0.01);
  // Constants are NOT preserved under partial overlap, so the row residual is
  // O(1) and `monotone` must be false: a constant is diluted downwards.
  EXPECT_GT(w.max_row_defect, 0.5);
  EXPECT_FALSE(w.monotone)
      << "a map that dilutes a constant is not bound preserving, however "
         "positive its weights are";
  std::printf("[ overlap  ] tgt coverage %.4f  row defect %.4f\n",
              w.min_tgt_coverage, w.max_row_defect);
}

TEST(Supermesh, ThinSliversAreClippedAndTheirMassIsKept) {
  // Every pair above has slivers no thinner than 1/35 of the domain, so a
  // candidate filter that quietly required, say, 1e-9 of overlap would pass
  // all of them. The mutation gate found exactly that gap.
  //
  // Offsetting the target by 1e-10 makes every target cell straddle two source
  // cells, one of them by ~1e-10 -- slivers eleven orders of magnitude thinner
  // than anything tested so far, and the regime where a remap implementation
  // with a hidden tolerance loses mass.
  auto p = pair_of("generated:box:4x4x4", "generated:box:4x4x4");
  ASSERT_TRUE(p != nullptr);
  const double id[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const double tiny[3] = {1e-10, 0.0, 0.0};
  const AffineView shifted(*p->tgt, id, tiny);

  const auto r = fuse::build_supermesh_weights(*p->src, shifted, 3);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  // 64 diagonal entries plus a sliver for each cell that now straddles: the
  // 48 cells not on the +x boundary each gain one. Anything close to 64 means
  // the slivers were filtered away.
  EXPECT_GT(w.nnz(), 64u + 40u)
      << "only " << w.nnz() << " entries: the 1e-10 slivers were dropped, so "
         "that much mass went missing without a diagnostic";

  // The sliver weights are ~1e-10/0.015625 = 6.4e-9, comfortably above the
  // negligible-overlap floor of 1e-12, so they must be KEPT rather than
  // discarded -- the floor exists for rounding-level contributions, not for
  // small ones.
  double smallest = 1.0;
  for (const double v : w.val) smallest = std::min(smallest, v);
  EXPECT_LT(smallest, 1e-7) << "no sliver-sized weight is present";
  EXPECT_GT(smallest, 1e-12) << "a kept weight below the floor";

  // Interior cells are fully covered, so conservation still holds at rounding
  // for them; the +x boundary column loses the 1e-10 strip that left the
  // domain, which is reported and not hidden.
  EXPECT_LT(w.max_col_defect, 1e-8)
      << "mass lost beyond the 1e-10 strip that genuinely left the domain";
  std::printf("[ slivers  ] nnz=%zu  smallest weight %.3e  col defect %.3e\n",
              w.nnz(), smallest, w.max_col_defect);
}
