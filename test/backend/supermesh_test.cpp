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
