// Lowest-order conservative remap on genuinely NON-MATCHING meshes -- the
// capability the default build did not have. Every expectation here is either
// a closed form or a conservation identity, so a failure names the defect
// rather than a tolerance.

#include "fuse/backend/supermesh.hpp"
#include "fuse/base/config.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <cmath>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/remap_plan.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/model/coupling_graph.hpp"

// PRIVATE header, reached the same way test/parallel/comm_graph_test.cpp
// reaches the transport layer's CrystalRouter -- see
// src/backend/supermesh_detail.hpp for why.
#include "backend/supermesh_detail.hpp"

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

// ---- 1D, axis orientation ------------------------------------------------
//
// Every 1D test above uses `generated:line`, which SerialMeshBackend lays
// out along x only -- so a kernel that silently hard-coded `.x` would pass
// every one of them. This section is the two cases that would have caught
// it: a line genuinely laid out along y (the kernel must still measure it,
// not return an empty matrix), and a line laid out diagonally (the kernel
// must refuse rather than measure its x-shadow).

namespace {

/// `n_cells` cells of equal length along the straight segment from `p0` to
/// `p1` -- `p0`/`p1` need not be axis-aligned, which is the point: unlike
/// `generated:line` (x only), this can place a 1D mesh along any direction.
class ParametricLineMesh final : public fuse::MeshView {
 public:
  ParametricLineMesh(std::size_t n_cells, fuse::Point p0, fuse::Point p1)
      : n_cells_(n_cells), p0_(p0), p1_(p1) {}

  [[nodiscard]] int dimension() const override { return 1; }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    if (dim == 0) return n_cells_ + 1;
    if (dim == 1) return n_cells_;
    return 0;
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override {
    return entity_count(dim);
  }
  [[nodiscard]] bool is_owned(int, std::size_t) const override { return true; }
  [[nodiscard]] std::size_t global_id(int, std::size_t i) const override { return i; }
  [[nodiscard]] fuse::Point centroid(int dim, std::size_t i) const override {
    const double n = static_cast<double>(n_cells_);
    return at(dim == 0 ? static_cast<double>(i) / n
                        : (static_cast<double>(i) + 0.5) / n);
  }
  [[nodiscard]] double measure(int dim, std::size_t) const override {
    if (dim != 1) throw std::domain_error("ParametricLineMesh: no measure at dimension 0");
    const double dx = p1_.x - p0_.x, dy = p1_.y - p0_.y, dz = p1_.z - p0_.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz) / static_cast<double>(n_cells_);
  }
  [[nodiscard]] std::size_t simplex_count(int dim, std::size_t) const override {
    return dim == 1 ? 1 : 0;
  }
  void simplex_points(int dim, std::size_t i, std::size_t,
                      fuse::Point* out) const override {
    out[0] = centroid(0, i);
    out[1] = centroid(0, i + 1);
  }

 private:
  [[nodiscard]] fuse::Point at(double t) const {
    return fuse::Point{p0_.x + t * (p1_.x - p0_.x), p0_.y + t * (p1_.y - p0_.y),
                       p0_.z + t * (p1_.z - p0_.z)};
  }
  std::size_t n_cells_;
  fuse::Point p0_, p1_;
};

}  // namespace

TEST(Supermesh, OneDimensionalYAlignedIsExactlyConservative) {
  // The exact mirror of OneDimensionalNonMatchingIsExactlyConservative
  // (generated:line:8 onto generated:line:5), laid out along y instead of
  // x. Pre-fix signature: every pair's x coordinate is identically 0, so
  // `intersect`'s dim==1 branch computes every overlap as exactly 0 -- an
  // EMPTY matrix returned as SUCCESS, not a refusal, indistinguishable
  // from a target field correctly left at zero.
  ParametricLineMesh src(8, fuse::Point{0, 0, 0}, fuse::Point{0, 1, 0});
  ParametricLineMesh tgt(5, fuse::Point{0, 0, 0}, fuse::Point{0, 1, 0});
  const auto r = fuse::build_supermesh_weights(src, tgt, 1);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  EXPECT_EQ(w.n_rows, 5u);
  EXPECT_EQ(w.n_cols, 8u);
  EXPECT_GT(w.nnz(), 5u) << "a non-matching pair must have split cells -- "
                            "an empty matrix is the pre-fix failure "
                            "signature, not a passing one";
  EXPECT_NEAR(w.min_src_coverage, 1.0, 1e-14);
  EXPECT_NEAR(w.min_tgt_coverage, 1.0, 1e-14);
  EXPECT_LT(w.max_row_defect, 1e-14) << "constants are not preserved";
  EXPECT_LT(w.max_col_defect, 1e-14) << "mass is not conserved";
  EXPECT_NEAR(w.src_measure_total, 1.0, 1e-14);
  EXPECT_NEAR(w.tgt_measure_total, 1.0, 1e-14);
  EXPECT_TRUE(w.monotone);
}

TEST(Supermesh, OneDimensionalDiagonalPairRefusesRatherThanProjecting) {
  // A line running diagonally in the xy-plane is aligned with NEITHER axis:
  // projecting onto either would measure its SHADOW, not its own length --
  // a plausible but wrong number, the exact reason
  // RefusesASurfacePairRatherThanProjectingIt refuses the 2D analogue
  // instead of guessing.
  ParametricLineMesh src(8, fuse::Point{0, 0, 0}, fuse::Point{1, 1, 0});
  ParametricLineMesh tgt(5, fuse::Point{0, 0, 0}, fuse::Point{1, 1, 0});
  const auto r = fuse::build_supermesh_weights(src, tgt, 1);
  ASSERT_FALSE(r.ok()) << "a diagonal 1D pair would have its shadow "
                          "measured, which looks like a legitimate answer";
  EXPECT_EQ(r.error().code, "E-REMAP-NONPLANAR");
}

// ---- rejections ----------------------------------------------------------

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
  // it must reject. This is the guard that keeps "always available" from
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
  // existed this pair was REJECTED, because the only available map was an
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

TEST(Supermesh, MinWeightIsZeroWhenEveryAccumulatedEntryIsDroppedAsNegligible) {
  // Shifting the target by a FULL unit domain (not a
  // fraction, as PartialOverlapIsReportedRatherThanHidden's 0.5 or
  // ThinSliversAreClippedAndTheirMassIsKept's 1e-10) makes it adjacent to,
  // not overlapping, the source: an interior target cell's box never
  // touches the source's at all (zero overlap, never added to the
  // accumulator), and only the cells sharing the x=1 boundary face produce
  // the clipper's own documented rounding-level spurious positive (this
  // file's kNegligible comment: "measured at up to 1.1e-16 of a cell").
  // `acc` therefore ends up NON-EMPTY (the touching-face entries) while
  // EVERY one of them is dropped by the kNegligible floor, leaving `w.val`
  // EMPTY -- the case the pre-fix min_w initialisation (keyed on
  // `acc.empty()`, not on `w.val.empty()`) got wrong.
  auto p = pair_of("generated:box:4x4x4", "generated:box:4x4x4");
  ASSERT_TRUE(p != nullptr);
  const double id[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  const double shift[3] = {1.0, 0.0, 0.0};  // a full domain: adjacent, not overlapping
  const AffineView moved_tgt(*p->tgt, id, shift);

  const auto r = fuse::build_supermesh_weights(*p->src, moved_tgt, 3);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  ASSERT_TRUE(w.val.empty())
      << "nnz=" << w.nnz() << ": this fixture must have no GENUINE overlap, "
         "only boundary-touching slivers the kNegligible floor drops -- "
         "otherwise it is not exercising the case this test names";
  EXPECT_EQ(w.min_weight, 0.0)
      << "RemapWeights::min_weight documents zero for an empty matrix";
}

// ---- RemapFlags::Conservative vs. partial coverage ----
//
// A refusal no test fires does not count as implemented.
// E-REMAP-COVERAGE guards a conservative remap against exactly the
// partial-overlap truncation PartialOverlapIsReportedRatherThanHidden above
// exercises. The MOAB path's own assemble() checks it;
// build_weights_via_supermesh, this backend-agnostic path's own entry
// point, must check plan.flags too, or RemapFlags::Conservative would be
// silently ignored for every non-MOAB backend.

namespace {

/// Wraps an inner SerialMeshBackend, substituting an AffineView-shifted
/// copy of ONE named mesh (by MeshId) for `placed_view` -- the SAME hook
/// `PerPlacementBackend` above uses, but shifting geometry rather than
/// renumbering it. This is what lets PartialOverlapIsReportedRatherThanHidden's
/// own shift-by-half-a-domain fixture be driven through `fuse::build_weights`
/// (the public entry point RemapFlags lives on) instead of
/// `build_supermesh_weights` directly (which takes raw MeshViews and never
/// sees a RemapPlan at all).
class ShiftedMeshBackend final : public fuse::MeshBackend {
 public:
  ShiftedMeshBackend(const fuse::SerialMeshBackend& inner, fuse::MeshId shifted,
                     double shift_x)
      : inner_(inner), shifted_id_(shifted) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) m_[r][c] = (r == c) ? 1.0 : 0.0;
    }
    t_[0] = shift_x;
    t_[1] = 0.0;
    t_[2] = 0.0;
  }
  fuse::Result<void> load(const fuse::MeshDecl&) override {
    return fuse::Failure{"E-TEST", "not loadable"};
  }
  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(std::string_view n) const override {
    return inner_.mesh_id(n);
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(fuse::MeshId id) const override {
    return inner_.view(id);
  }
  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& r) const override {
    return inner_.region_size(r);
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(
      fuse::MeshId id, fuse::ComponentId) const override {
    if (id.value() != shifted_id_.value()) return inner_.view(id);
    if (!shifted_view_) {
      const auto v = inner_.view(id);
      if (!v.ok()) return v.error();
      shifted_view_ = std::make_unique<AffineView>(*v.value(), m_, t_);
    }
    return shifted_view_.get();
  }
  [[nodiscard]] std::string name() const override { return "shifted-serial"; }

 private:
  const fuse::SerialMeshBackend& inner_;
  fuse::MeshId shifted_id_;
  double m_[3][3]{};
  double t_[3]{};
  mutable std::unique_ptr<AffineView> shifted_view_;
};

}  // namespace

TEST(Supermesh, BuildWeightsRefusesPartialCoverageUnderConservativeFlag) {
  fuse::SerialMeshBackend inner;
  ASSERT_TRUE(inner.load({.name = "s", .source = "generated:box:4x4x4"}).ok());
  ASSERT_TRUE(inner.load({.name = "t", .source = "generated:box:4x4x4"}).ok());
  const auto tmid = inner.mesh_id("t");
  ASSERT_TRUE(tmid.ok());
  // Half a domain to the right, exactly PartialOverlapIsReportedRatherThanHidden's
  // own shift: half of every straddling target cell has no source.
  ShiftedMeshBackend backend(inner, tmid.value(), /*shift_x=*/0.5);

  fuse::CouplingGraph graph;
  const auto smid = graph.add_mesh({.name = "s", .source = "generated:box:4x4x4"});
  const auto gtmid = graph.add_mesh({.name = "t", .source = "generated:box:4x4x4"});
  const auto srid = graph.add_region({.name = "s_all", .mesh = smid, .dimension = 3});
  const auto trid = graph.add_region({.name = "t_all", .mesh = gtmid, .dimension = 3});

  fuse::RemapPlan plan;
  plan.src = srid;
  plan.tgt = trid;

  // Pre-fix signature: RemapFlags::Conservative is silently ignored by
  // build_weights_via_supermesh, so this call SUCCEEDS with a truncated,
  // non-conservative map instead of refusing.
  plan.flags = fuse::RemapFlags::Conservative;
  const auto refused = fuse::build_weights(plan, graph, backend);
  ASSERT_FALSE(refused.ok()) << "RemapFlags::Conservative must refuse a "
                                "partial-overlap pair, not silently "
                                "truncate it";
  EXPECT_EQ(refused.error().code, "E-REMAP-COVERAGE");
  EXPECT_NE(refused.error().message.find("Conservative flag to accept a truncating map"),
           std::string::npos)
      << "the refusal must carry advice on what to do, not just its own "
         "name: \"" << refused.error().message << "\"";

  // Without the flag, the IDENTICAL pair must still succeed (truncating,
  // not refusing) -- the flag, not the geometry, is what decides.
  plan.flags = fuse::RemapFlags::None;
  const auto truncated = fuse::build_weights(plan, graph, backend);
  ASSERT_TRUE(truncated.ok()) << truncated.error().code << ": "
                              << truncated.error().message;
  EXPECT_LT(truncated.value().min_src_coverage, 0.6)
      << "the shifted pair must genuinely be a partial overlap, or this "
         "test is not exercising the boundary it claims to";
}

// ---- inward_plane whitebox regression ----------------------------------
//
// build_supermesh_weights has no source grammar that produces a degenerate
// clipping face (SerialMeshBackend only generates regular, non-degenerate
// lines/boxes), so a black-box regression test through the public API is not
// possible. These reach the code directly through the private
// supermesh_detail.hpp header instead.

TEST(SupermeshDetail, InwardPlaneRefusesAZeroLengthEdgeIn2D) {
  // Face 0 of the "triangle" v omits v[0] and is the edge v[1]-v[2]; make
  // that edge zero-length by placing v[1] and v[2] at the same point. Without the
  // check this would produce n=(0,0,0), d=0 -- a half-plane that reads `0 >= 0` for
  // every point, i.e. no constraint at all -- instead of rejecting.
  const std::array<fuse::Point, 4> v{{{0.0, 0.0, 0.0},
                                      {1.0, 1.0, 0.0},
                                      {1.0, 1.0, 0.0},   // == v[1]: degenerate
                                      {0.0, 0.0, 0.0}}};
  double n[3] = {0.0, 0.0, 0.0};
  double d = 0.0;
  EXPECT_FALSE((fuse::supermesh_detail::inward_plane<2>(v, 0, n, d)));
}

TEST(SupermeshDetail, InwardPlaneRefusesAZeroAreaFaceIn3D) {
  // Face 0 of the tetrahedron v omits v[0] and is the triangle v[1]-v[2]-v[3];
  // collapse it to a segment (v[1] == v[2]) so its cross product, and
  // therefore its normal, is exactly zero.
  const std::array<fuse::Point, 4> v{{{0.0, 0.0, 0.0},
                                      {1.0, 0.0, 0.0},
                                      {1.0, 0.0, 0.0},   // == v[1]: degenerate
                                      {0.0, 1.0, 0.0}}};
  double n[3] = {0.0, 0.0, 0.0};
  double d = 0.0;
  EXPECT_FALSE((fuse::supermesh_detail::inward_plane<3>(v, 0, n, d)));
}

TEST(SupermeshDetail, InwardPlaneAcceptsAndOrientsANonDegenerateFaceIn2D) {
  // The unit right triangle (0,0), (1,0), (0,1). Face 0 omits v[0] and is
  // the hypotenuse v[1]-v[2]; the omitted vertex v[0] is at the origin, on
  // the plane's positive side by construction (inward_plane orients toward
  // the omitted vertex), and the plane must actually pass through the edge.
  const std::array<fuse::Point, 4> v{
      {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}}};
  double n[3] = {0.0, 0.0, 0.0};
  double d = 0.0;
  ASSERT_TRUE((fuse::supermesh_detail::inward_plane<2>(v, 0, n, d)));
  // n is a unit normal.
  EXPECT_NEAR(n[0] * n[0] + n[1] * n[1] + n[2] * n[2], 1.0, 1e-12);
  // v[1] and v[2] sit ON the plane.
  EXPECT_NEAR(n[0] * v[1].x + n[1] * v[1].y + d, 0.0, 1e-12);
  EXPECT_NEAR(n[0] * v[2].x + n[1] * v[2].y + d, 0.0, 1e-12);
  // The omitted vertex v[0] (the origin) is strictly on the inward side.
  EXPECT_GT(n[0] * v[0].x + n[1] * v[0].y + d, 0.0);
}

// ---- audit_coverage's rejection boundary ----------------------------
//
// The defect class this guards against -- overlap inflated past the cell it
// came from -- cannot be produced through the public MeshView grammar by a
// well-formed mesh (that is why only a
// mutation, `supermesh-no-over-coverage-audit`, and not a permanent test).
// Testing the extracted boundary directly on a synthetic ratio pins the
// threshold and the error shape without manufacturing invalid geometry.

TEST(SupermeshDetail, AuditCoverageAcceptsExactAndSlightlyUnderCoverage) {
  EXPECT_TRUE(fuse::supermesh_detail::audit_coverage(1.0, 1.0, "source", 0, 2).ok());
  EXPECT_TRUE(fuse::supermesh_detail::audit_coverage(0.5, 1.0, "target", 3, 3).ok());
  // Within the 1e-9 relative tolerance: rounding, not a defect.
  EXPECT_TRUE(
      fuse::supermesh_detail::audit_coverage(1.0 + 5e-10, 1.0, "source", 0, 2)
          .ok());
}

TEST(SupermeshDetail, AuditCoverageRefusesOverCoverageWithTheDefectsErrorCode) {
  // 1.5x its own area: exactly the class the
  // `supermesh-no-over-coverage-audit` mutation targets.
  const auto r =
      fuse::supermesh_detail::audit_coverage(1.5, 1.0, "target", 7, 2);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-INTX");
  EXPECT_NE(r.error().message.find("target cell 7"), std::string::npos)
      << r.error().message;
  EXPECT_NE(r.error().message.find("1.5"), std::string::npos)
      << r.error().message;
}

TEST(SupermeshDetail, AuditCoverageSkipsAZeroMeasureCellRatherThanDivideByZero) {
  EXPECT_TRUE(fuse::supermesh_detail::audit_coverage(0.0, 0.0, "source", 2, 1)
                  .ok());
}

// ---- build_weights dispatches on the backend, not on FUSE_HAVE_MOAB -----
// With MOAB compiled in, resolve()'s dynamic_cast must not reject every
// SerialMeshBackend call with "build_weights needs the MOAB backend" (which
// would fail Runner.RemapsBetweenDifferentlySizedRegions and its siblings).
// This runs identically at the base build configuration (where it is the
// only path there is, so it is a plain sanity check) and at the MOAB-enabled
// build configuration (where `direct` must be the oracle's weights, not
// E-REMAP-BACKEND).

// An unrecognised method string is refused by `build_weights()` on EVERY
// backend, with the MOAB path's own code. The dependency-free path used to
// match only "consistent"/"consistent-transpose"/"mls" and then fall through
// to the conservative supermesh for ANY other string, so a misspelt method
// was silently computed as the default remap (the serial backend here is
// exactly that path, with or without MOAB compiled in). "tempest" names a
// backend this path does not have and is refused by name rather than
// quietly built at order 1.
TEST(Supermesh, BuildWeightsRefusesAnUnknownMethodOnTheSerialBackend) {
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load({.name = "s", .source = "generated:line:8"}).ok());
  ASSERT_TRUE(backend.load({.name = "t", .source = "generated:line:5"}).ok());
  fuse::CouplingGraph graph;
  const fuse::MeshId smid = graph.add_mesh({.name = "s", .source = "generated:line:8"});
  const fuse::MeshId tmid = graph.add_mesh({.name = "t", .source = "generated:line:5"});
  fuse::RemapPlan plan;
  plan.src = graph.add_region({.name = "s_all", .mesh = smid, .dimension = 1});
  plan.tgt = graph.add_region({.name = "t_all", .mesh = tmid, .dimension = 1});

  for (const char* bad : {"supermesh", "consistent_transpose", "Intx", "mls2"}) {
    plan.method = bad;
    const auto r = fuse::build_weights(plan, graph, backend);
    ASSERT_FALSE(r.ok()) << "method '" << bad << "' was built as if it were known";
    EXPECT_EQ(r.error().code, "E-REMAP-UNSUPPORTED") << r.error().message;
    EXPECT_NE(r.error().message.find(bad), std::string::npos) << r.error().message;
  }
  for (const char* good : {"", "intx", "conservative"}) {
    plan.method = good;
    const auto r = fuse::build_weights(plan, graph, backend);
    EXPECT_TRUE(r.ok()) << "method '" << good << "': " << r.error().code << ": "
                        << r.error().message;
  }
#if !FUSE_HAVE_MOAB
  plan.method = "tempest";
  const auto r = fuse::build_weights(plan, graph, backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-REMAP-NOHIGHORDER") << r.error().message;
#endif
}

TEST(Supermesh, BuildWeightsReachesTheSupermeshPathForANonMoabBackend) {
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load({.name = "s", .source = "generated:box:2x2x2"}).ok());
  ASSERT_TRUE(backend.load({.name = "t", .source = "generated:box:8x1x1"}).ok());

  fuse::CouplingGraph graph;
  const fuse::MeshId smid =
      graph.add_mesh({.name = "s", .source = "generated:box:2x2x2"});
  const fuse::MeshId tmid =
      graph.add_mesh({.name = "t", .source = "generated:box:8x1x1"});
  const fuse::RegionId srid =
      graph.add_region({.name = "s_all", .mesh = smid, .dimension = 3});
  const fuse::RegionId trid =
      graph.add_region({.name = "t_all", .mesh = tmid, .dimension = 3});

  fuse::RemapPlan plan;
  plan.src = srid;
  plan.tgt = trid;

  // The dispatch under test: the SAME public entry point every caller (the
  // Runner included) actually calls.
  const auto direct = fuse::build_weights(plan, graph, backend);
  ASSERT_TRUE(direct.ok()) << direct.error().code << ": " << direct.error().message
                           << " -- build_weights refused a plain serial mesh "
                              "pair; it must reach build_supermesh_weights "
                              "regardless of whether MOAB was compiled in";

  // The oracle: calling the supermesh builder directly, over the same views.
  const auto sid = backend.mesh_id("s");
  const auto tid = backend.mesh_id("t");
  ASSERT_TRUE(sid.ok());
  ASSERT_TRUE(tid.ok());
  const auto sview = backend.view(sid.value());
  const auto tview = backend.view(tid.value());
  ASSERT_TRUE(sview.ok());
  ASSERT_TRUE(tview.ok());
  const auto oracle = fuse::build_supermesh_weights(*sview.value(), *tview.value(), 3);
  ASSERT_TRUE(oracle.ok()) << oracle.error().code << ": " << oracle.error().message;

  // build_weights's dispatch must produce EXACTLY the supermesh builder's own
  // matrix, not a different (or rejected) one.
  EXPECT_EQ(direct.value().row, oracle.value().row);
  EXPECT_EQ(direct.value().col, oracle.value().col);
  EXPECT_EQ(direct.value().val, oracle.value().val);
  EXPECT_EQ(direct.value().n_rows, oracle.value().n_rows);
  EXPECT_EQ(direct.value().n_cols, oracle.value().n_cols);
}

// ---- weights follow the PROCESSOR SET's decomposition -------

namespace {

/// The same mesh with its local cell numbering reversed: cell i here is cell
/// n-1-i of the wrapped view. A different decomposition of identical
/// geometry, which is exactly what a per-placement backend hands out, and
/// the one for which "weights built on the default view" is measurably wrong:
/// they would put source cell 0's mass into target cell 0, and here target
/// cell 0 is the OTHER end of the line.
class ReversedView final : public fuse::MeshView {
 public:
  explicit ReversedView(const fuse::MeshView& inner) : inner_(inner) {}
  [[nodiscard]] int dimension() const override { return inner_.dimension(); }
  [[nodiscard]] std::size_t entity_count(int d) const override { return inner_.entity_count(d); }
  [[nodiscard]] std::size_t owned_count(int d) const override { return inner_.owned_count(d); }
  [[nodiscard]] bool is_owned(int d, std::size_t i) const override { return inner_.is_owned(d, r(d, i)); }
  [[nodiscard]] fuse::Point centroid(int d, std::size_t i) const override { return inner_.centroid(d, r(d, i)); }
  [[nodiscard]] double measure(int d, std::size_t i) const override { return inner_.measure(d, r(d, i)); }
  [[nodiscard]] std::size_t global_id(int d, std::size_t i) const override { return inner_.global_id(d, r(d, i)); }
  [[nodiscard]] std::size_t simplex_count(int d, std::size_t i) const override { return inner_.simplex_count(d, r(d, i)); }
  void simplex_points(int d, std::size_t i, std::size_t s, fuse::Point* out) const override {
    inner_.simplex_points(d, r(d, i), s, out);
  }
 private:
  [[nodiscard]] std::size_t r(int d, std::size_t i) const { return inner_.entity_count(d) - 1 - i; }
  const fuse::MeshView& inner_;
};

/// Default view for the default processor set; the reversed one for component 1.
class PerPlacementBackend final : public fuse::MeshBackend {
 public:
  PerPlacementBackend(const fuse::SerialMeshBackend& inner, fuse::MeshId target)
      : inner_(inner), target_(target) {}
  fuse::Result<void> load(const fuse::MeshDecl&) override { return fuse::Failure{"E-TEST", "not loadable"}; }
  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(std::string_view n) const override { return inner_.mesh_id(n); }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(fuse::MeshId id) const override { return inner_.view(id); }
  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& rg) const override { return inner_.region_size(rg); }
  /// Processor set 1 numbers the TARGET mesh backwards and the source as the
  /// default. Only one side on purpose: reversing both is a mirror symmetry
  /// of a uniform line pair and leaves the matrix unchanged (the first draft
  /// of this test did that, and its own "the two matrices coincide" guard
  /// caught it). A decomposition need not treat every mesh alike.
  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(fuse::MeshId id, fuse::ComponentId who) const override {
    if (!who.valid() || who.value() != 1 || id.value() != target_.value()) return inner_.view(id);
    auto it = reversed_.find(id.value());
    if (it == reversed_.end()) {
      const auto v = inner_.view(id);
      if (!v.ok()) return v.error();
      it = reversed_.emplace(id.value(), std::make_unique<ReversedView>(*v.value())).first;
    }
    return it->second.get();
  }
  [[nodiscard]] std::string name() const override { return "per-placement"; }
 private:
  const fuse::SerialMeshBackend& inner_;
  fuse::MeshId target_;
  mutable std::map<std::uint32_t, std::unique_ptr<ReversedView>> reversed_;
};

/// Dense form of a RemapWeights, for permutation comparisons.
std::vector<double> dense(const fuse::RemapWeights& w) {
  std::vector<double> m(w.n_rows * w.n_cols, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    m[static_cast<std::size_t>(w.row[k]) * w.n_cols + static_cast<std::size_t>(w.col[k])] += w.val[k];
  }
  return m;
}

}  // namespace

TEST(Supermesh, WeightsAreBuiltOnThePlacementsOwnDecomposition) {
  fuse::SerialMeshBackend inner;
  ASSERT_TRUE(inner.load({.name = "s", .source = "generated:line:8"}).ok());
  ASSERT_TRUE(inner.load({.name = "t", .source = "generated:line:3"}).ok());
  const auto tid = inner.mesh_id("t");
  ASSERT_TRUE(tid.ok());
  const PerPlacementBackend backend{inner, tid.value()};

  fuse::CouplingGraph graph;
  const auto smid = graph.add_mesh({.name = "s", .source = "generated:line:8"});
  const auto tmid = graph.add_mesh({.name = "t", .source = "generated:line:3"});
  const auto srid = graph.add_region({.name = "s_all", .mesh = smid, .dimension = 1});
  const auto trid = graph.add_region({.name = "t_all", .mesh = tmid, .dimension = 1});

  fuse::RemapPlan plan;
  plan.src = srid;
  plan.tgt = trid;
  const auto dflt = fuse::build_weights(plan, graph, backend);
  ASSERT_TRUE(dflt.ok()) << dflt.error().code << ": " << dflt.error().message;

  const std::string default_key = plan.key();
  plan.placement = fuse::ComponentId{1};
  EXPECT_NE(plan.key(), default_key) << "two processor sets must never share a cache entry";
  const auto placed = fuse::build_weights(plan, graph, backend);
  ASSERT_TRUE(placed.ok()) << placed.error().code << ": " << placed.error().message;

  // With the target numbered backwards, the placed matrix is the default one
  // with its ROWS reversed. 8 onto 3 is not a divisor pair, so the rows are
  // not palindromic and the row-reversed matrix differs from the original.
  const auto D = dense(dflt.value());
  const auto P = dense(placed.value());
  ASSERT_EQ(D.size(), P.size());
  const std::size_t R = dflt.value().n_rows, C = dflt.value().n_cols;
  double max_perm = 0.0, max_same = 0.0;
  for (std::size_t i = 0; i < R; ++i) {
    for (std::size_t j = 0; j < C; ++j) {
      max_perm = std::max(max_perm, std::abs(P[i * C + j] - D[(R - 1 - i) * C + j]));
      max_same = std::max(max_same, std::abs(P[i * C + j] - D[i * C + j]));
    }
  }
  // Exact: the two builds see the same simplices in a different order, and
  // every entry is one overlap length divided by one target measure.
  EXPECT_EQ(max_perm, 0.0) << "the placed weights are not the row-reversed default ones";
  EXPECT_GT(max_same, 0.1) << "the two matrices coincide, so the test cannot tell whether the "
                              "placement was honoured -- choose a less symmetric pair";
}


// ---------------------------------------------------------------------------
// Soma-import port (docs/dev/soma-import.md, RemapPartial.* rows). The
// diagnostic FIELDS this backend reports under partial coverage are already
// proved above (PartialOverlapIsReportedRatherThanHidden, ThinSliversAre-
// ClippedAndTheirMassIsKept); what those do not reach is `build_weights`'s
// own `RemapFlags::Conservative` REFUSAL (E-REMAP-COVERAGE, intx_weights.cpp)
// -- every other test in this tree that sets `RemapFlags::Conservative` does
// so on a fully-covering pair and never exercises the branch at all (grepped
// across test/{backend,parallel,schedule,ir,model,abi,runtime} -- the flag is
// set in a dozen places, the refusal it guards is asserted in none). Nor is
// the numerical TRUNCATION behaviour apply_weights produces when the flag is
// absent: build succeeds, but a source cell the target never reaches
// contributes nothing to any target value, rather than being redistributed.

namespace {

/// Wraps a SerialMeshBackend and returns an `AffineView` of ONE named mesh
/// instead of its raw view -- everything else (load, mesh_id, region_size)
/// delegates unchanged. Lets a plain "generated:box:NxNxN" source (fixed to
/// the unit cube by the generator grammar) stand in for a mesh on a
/// different sub-domain, the same technique `PartialOverlapIsReported-
/// RatherThanHidden` above uses through `build_supermesh_weights` directly --
/// this one goes through `build_weights(plan, graph, backend)` instead, the
/// entry point the Conservative refusal actually lives behind.
class RescaledBackend final : public fuse::MeshBackend {
 public:
  RescaledBackend(fuse::SerialMeshBackend& inner, fuse::MeshId rescale_id,
                   const double m[3][3], const double t[3])
      : inner_(inner), rescale_id_(rescale_id) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) m_[r][c] = m[r][c];
      t_[r] = t[r];
    }
  }
  fuse::Result<void> load(const fuse::MeshDecl&) override {
    return fuse::Failure{"E-TEST", "not loadable"};
  }
  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(std::string_view n) const override {
    return inner_.mesh_id(n);
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(fuse::MeshId id) const override {
    const auto v = inner_.view(id);
    if (!v.ok()) return v.error();
    if (id.value() != rescale_id_.value()) return v.value();
    if (!rescaled_) rescaled_ = std::make_unique<AffineView>(*v.value(), m_, t_);
    return rescaled_.get();
  }
  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& rg) const override {
    return inner_.region_size(rg);
  }
  [[nodiscard]] std::string name() const override { return "rescaled"; }

 private:
  fuse::SerialMeshBackend&      inner_;
  fuse::MeshId                  rescale_id_;
  double                        m_[3][3]{};
  double                        t_[3]{};
  mutable std::unique_ptr<AffineView> rescaled_;
};

/// Source: a 4x4x4 unit-cube grid (cell width 0.25 in every dimension).
/// Target: the SAME grid rescaled to occupy x in [0.25, 0.75] only -- exactly
/// the middle two of the source's four x-slabs, with the generator's own
/// cell boundaries (0, 0.25, 0.5, 0.75, 1) making the split exact rather than
/// approximate. Source slabs 0 and 3 (x in [0,0.25] and [0.75,1]) then have
/// NO target cell anywhere near them: half the source domain, by measure, is
/// unreachable by this target.
struct NarrowTargetFixture {
  fuse::SerialMeshBackend inner;
  fuse::CouplingGraph     graph;
  fuse::MeshId            smid, tmid;
  fuse::RegionId          srid, trid;
  std::unique_ptr<RescaledBackend> backend;

  NarrowTargetFixture() {
    if (!inner.load({.name = "s", .source = "generated:box:4x4x4"}).ok()) return;
    if (!inner.load({.name = "t", .source = "generated:box:4x4x4"}).ok()) return;
    smid = graph.add_mesh({.name = "s", .source = "generated:box:4x4x4"});
    tmid = graph.add_mesh({.name = "t", .source = "generated:box:4x4x4"});
    srid = graph.add_region({.name = "s_all", .mesh = smid, .dimension = 3});
    trid = graph.add_region({.name = "t_all", .mesh = tmid, .dimension = 3});
    const double half_x[3][3] = {{0.5, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const double shift[3]     = {0.25, 0.0, 0.0};
    const auto   tid          = inner.mesh_id("t");
    backend = std::make_unique<RescaledBackend>(inner, tid.value(), half_x, shift);
  }

  [[nodiscard]] fuse::RemapPlan plan(fuse::RemapFlags flags) const {
    return fuse::RemapPlan{.src = srid, .tgt = trid, .flags = flags};
  }
};

}  // namespace

TEST(RemapPartial, SourceWiderThanTargetIsRefusedWhenConservativeIsAsked) {
  NarrowTargetFixture fx;
  const auto r = fuse::build_weights(fx.plan(fuse::RemapFlags::Conservative),
                                      fx.graph, *fx.backend);
  ASSERT_FALSE(r.ok()) << "half the source mass has no target cell to reach; "
                           "RemapFlags::Conservative must refuse rather than "
                           "silently drop it";
  EXPECT_EQ(r.error().code, "E-REMAP-COVERAGE");
}

TEST(RemapPartial, SourceWiderThanTargetTruncatesWhenConservationIsNotAsked) {
  NarrowTargetFixture fx;
  const auto r = fuse::build_weights(fx.plan(fuse::RemapFlags::None), fx.graph,
                                      *fx.backend);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();

  // The target is entirely inside the source's domain (every target cell is
  // fully covered); the source's outer two slabs are entirely outside the
  // target's (zero coverage, not partial).
  EXPECT_NEAR(w.min_tgt_coverage, 1.0, 1e-12);
  EXPECT_NEAR(w.min_src_coverage, 0.0, 1e-12);

  const std::vector<double> src(w.n_cols, 2.0);
  std::vector<double> dst(w.n_rows, -1.0);
  ASSERT_TRUE(fuse::apply_weights(w, src.data(), dst.data(), 1).ok());

  // Every target value is exactly the source constant -- truncation drops
  // unreached mass, it does not dilute what DOES arrive.
  for (const double v : dst) EXPECT_NEAR(v, 2.0, 1e-12);

  // Exactly the covered half of the source domain's integral arrives: the
  // two reached slabs, not all four.
  const auto sv = fx.backend->view(fx.smid);
  const auto tv = fx.backend->view(fx.tmid);
  ASSERT_TRUE(sv.ok());
  ASSERT_TRUE(tv.ok());
  double src_integral = 0.0;
  for (std::size_t i = 0; i < w.n_cols; ++i) src_integral += src[i] * sv.value()->measure(3, i);
  double tgt_integral = 0.0;
  for (std::size_t j = 0; j < w.n_rows; ++j) tgt_integral += dst[j] * tv.value()->measure(3, j);
  EXPECT_NEAR(tgt_integral, 0.5 * src_integral, 1e-12 * src_integral)
      << "exactly half the source's constant-field integral should arrive; "
         "the other half had no target cell to receive it";
}

// ===========================================================================
// Device-resident arcs: the broad
// phase (fuse::device::candidate_pairs) under Device, standing in front of
// the UNCHANGED host clip. Host mode (no guard below) is
// covered by every test above; these rows set
// FUSE_EXECUTION_SPACE directly (build_supermesh_weights's own
// fuse::search::resolve_execution_space() reads it, mirroring
// RunnerConfig's convention -- see mls_weights.cpp's identical choice for
// the MLS weights) so they are a genuine comparison regardless of
// whether this binary's own ctest row also sets it externally.
// ===========================================================================

namespace {

class SupermeshExecutionSpaceEnvGuard {
 public:
  explicit SupermeshExecutionSpaceEnvGuard(const char* value) {
    const char* prev = std::getenv("FUSE_EXECUTION_SPACE");
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_value_ = prev;
    if (value == nullptr) {
      unsetenv("FUSE_EXECUTION_SPACE");
    } else {
      setenv("FUSE_EXECUTION_SPACE", value, /*overwrite=*/1);
    }
  }
  ~SupermeshExecutionSpaceEnvGuard() {
    if (had_prev_) {
      setenv("FUSE_EXECUTION_SPACE", prev_value_.c_str(), /*overwrite=*/1);
    } else {
      unsetenv("FUSE_EXECUTION_SPACE");
    }
  }
  SupermeshExecutionSpaceEnvGuard(const SupermeshExecutionSpaceEnvGuard&) = delete;
  SupermeshExecutionSpaceEnvGuard& operator=(const SupermeshExecutionSpaceEnvGuard&) = delete;

 private:
  bool had_prev_ = false;
  std::string prev_value_;
};

}  // namespace

TEST(SupermeshDevice, ThreeDimensionalDeviceBroadPhaseMatchesHostExactly) {
  // Same 3x3x3 -> 4x4x4 pair as ThreeDimensionalNonMatchingIsExactlyConservative
  // (above): no interior face plane shared, every weight from a genuine
  // clip. dim == 3 is exactly device_candidate_pairs's own box test, so the
  // candidate set -- and therefore every accumulated triplet, in the same
  // accumulation order -- is identical to the host brute force's, not
  // merely close.
  auto p = pair_of("generated:box:3x3x3", "generated:box:4x4x4");
  ASSERT_TRUE(p != nullptr);

  fuse::RemapWeights host_w;
  {
    SupermeshExecutionSpaceEnvGuard guard("host");
    const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 3);
    ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
    host_w = r.value();
  }
  fuse::RemapWeights device_w;
  {
    SupermeshExecutionSpaceEnvGuard guard("device");
    const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 3);
    ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
    device_w = r.value();
  }

  ASSERT_EQ(host_w.row.size(), device_w.row.size());
  EXPECT_EQ(host_w.row, device_w.row);
  EXPECT_EQ(host_w.col, device_w.col);
  ASSERT_EQ(host_w.val.size(), device_w.val.size());
  for (std::size_t k = 0; k < host_w.val.size(); ++k) {
    EXPECT_DOUBLE_EQ(host_w.val[k], device_w.val[k]) << "triplet " << k;
  }
  EXPECT_DOUBLE_EQ(host_w.max_row_defect, device_w.max_row_defect);
  EXPECT_DOUBLE_EQ(host_w.max_col_defect, device_w.max_col_defect);
  EXPECT_DOUBLE_EQ(host_w.min_src_coverage, device_w.min_src_coverage);
  EXPECT_DOUBLE_EQ(host_w.min_tgt_coverage, device_w.min_tgt_coverage);
  EXPECT_EQ(host_w.monotone, device_w.monotone);
}

TEST(SupermeshDevice, OneDimensionalStaysOnHostBroadPhaseUnderDeviceToo) {
  // dim == 1 never takes the device broad phase (device_candidate_pairs's
  // own comment: its box test has no axis1d parameter) -- this is the
  // regression check that Device mode's result for a 1D pair is still
  // produced by the SAME brute-force loop Host mode uses, not a silently
  // wrong device-filtered one.
  auto p = pair_of("generated:line:8", "generated:line:5");
  ASSERT_TRUE(p != nullptr);
  SupermeshExecutionSpaceEnvGuard guard("device");
  const auto r = fuse::build_supermesh_weights(*p->src, *p->tgt, 1);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  const fuse::RemapWeights& w = r.value();
  EXPECT_EQ(w.n_rows, 5u);
  EXPECT_EQ(w.n_cols, 8u);
  EXPECT_LT(w.max_row_defect, 1e-14);
  EXPECT_LT(w.max_col_defect, 1e-14);
}

// ===========================================================================
// The normalisation convention
// build_supermesh_weights applies is chosen by its own Conservation argument.
// Both tests below use a STRICT, convention-specific assertion (amount
// conserved for Extensive; the constant reproduced exactly for Intensive) --
// not a loose tolerance either one could pass by coincidence -- so the
// mutation row that swaps them (build-evidence/mutations/supermesh.toml) is
// killed by the WRONG assertion failing, not by a widened bound.
// ===========================================================================

TEST(Supermesh, ExtensiveConservesTheTotalAmountNotTheConstant) {
  // A non-matching resolution over the SAME domain (both box generators,
  // dim 3): Extensive and Intensive give DIFFERENT matrices on a pair like
  // this, so this test would pass under either convention's assertion only
  // by coincidence.
  auto p = pair_of("generated:box:2x2x2", "generated:box:3x3x3");
  ASSERT_TRUE(p != nullptr);
  const auto built =
      fuse::build_supermesh_weights(*p->src, *p->tgt, 3, fuse::Conservation::Extensive);
  ASSERT_TRUE(built.ok()) << built.error().code << ": " << built.error().message;
  const fuse::RemapWeights& w = built.value();

  constexpr double kFillValue = 7.0;
  const std::vector<double> f(w.n_cols, kFillValue);
  const std::vector<double> g = apply_map(w, f);

  const double source_sum = std::accumulate(f.begin(), f.end(), 0.0);
  const double target_sum = std::accumulate(g.begin(), g.end(), 0.0);
  EXPECT_NEAR(source_sum, target_sum, 1e-9)
      << "Extensive must conserve the TOTAL amount (source_sum == target_sum), not reproduce "
         "the constant on every target cell -- source_sum=" << source_sum
      << " target_sum=" << target_sum;

  // 7.0 * 8 source cells (2x2x2): pins the source side so the real
  // assertion above is checking the TARGET against a known number, not two
  // unknowns.
  EXPECT_NEAR(source_sum, 56.0, 1e-9);
}

TEST(Supermesh, IntensiveReproducesTheConstantNotJustTheTotal) {
  auto p = pair_of("generated:box:2x2x2", "generated:box:3x3x3");
  ASSERT_TRUE(p != nullptr);
  const auto built =
      fuse::build_supermesh_weights(*p->src, *p->tgt, 3, fuse::Conservation::Intensive);
  ASSERT_TRUE(built.ok()) << built.error().code << ": " << built.error().message;
  const fuse::RemapWeights& w = built.value();

  constexpr double kFillValue = 7.0;
  const std::vector<double> f(w.n_cols, kFillValue);
  const std::vector<double> g = apply_map(w, f);

  // The STRICT, convention-specific check: EVERY target cell reads back the
  // SAME constant -- not merely a total that happens to match (which a
  // swapped-normalisation mutation could still satisfy by the pair's own
  // total-measure coincidence; the per-cell check cannot).
  for (double v : g) {
    EXPECT_NEAR(v, kFillValue, 1e-12)
        << "Intensive (row-sum-to-1) must reproduce a constant field exactly on every "
           "target cell, regardless of how many source cells intersected it";
  }
}
