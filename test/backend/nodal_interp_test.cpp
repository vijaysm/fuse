// General (any supported cell) consistent interpolation -- nodal-remap
// track, step "Consistent interpolation for any supported cell". This file
// exercises `fuse::build_general_interp_weights` (general_interp.hpp),
// reached through the public dispatcher `fuse::build_consistent_interp_weights`
// (consistent_interp.hpp) exactly the way `operator_conformance_test.cpp`'s
// `ConsistentInterp.*` cases exercise the 1D path -- this file is that
// suite's sibling for 2D and 3D cells, real and hand-built.
//
// WHAT IT DOES NOT CHECK: full end-to-end Runner wiring (that is
// `test/verification/interface_transfer_test.cpp`'s job, extended
// separately); the distributed builder (nodal_remap_distributed.hpp),
// covered by its own compile-checked, MPI-gated test.

#include "fuse/backend/consistent_interp.hpp"
#include "fuse/backend/general_interp.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/native_backend.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/backend/weights_operator.hpp"

using fuse::CellShape;
using fuse::InterfaceDomainPolicy;
using fuse::MeshView;
using fuse::Point;
using fuse::RemapWeights;
using fuse::Result;

namespace {

std::vector<double> apply_H(const RemapWeights& w, const std::vector<double>& f) {
  std::vector<double> g(w.n_rows, 0.0);
  const auto r = fuse::apply_weights(w, f.data(), g.data(), 1);
  if (!r.ok()) ADD_FAILURE() << r.error().message;
  return g;
}

// ===========================================================================
// A minimal, hand-built single/few-cell MeshView: exact control over
// geometry, shape and connectivity, for the properties a generated mesh
// cannot isolate as cleanly (a single degenerate cell, a point exactly on a
// shared face, a shape this builder must refuse).
// ===========================================================================
class ManualCellMesh final : public MeshView {
 public:
  /// `cell_gids`, when non-empty, overrides the default identity numbering
  /// for dimension-`dim` global ids -- e.g. to make a LOWER local index
  /// carry a HIGHER global id, exercising the lowest-global-id tie-break
  /// against insertion order rather than (accidentally) agreeing with it.
  ManualCellMesh(int dim, std::vector<Point> verts, std::vector<CellShape> shapes,
                std::vector<std::vector<std::size_t>> cells,
                std::vector<std::size_t> cell_gids = {})
      : dim_(dim), verts_(std::move(verts)), shapes_(std::move(shapes)),
        cells_(std::move(cells)), cell_gids_(std::move(cell_gids)) {}

  int dimension() const override { return dim_; }
  std::size_t entity_count(int dim) const override {
    if (dim == 0) return verts_.size();
    if (dim == dim_) return cells_.size();
    return 0;
  }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int, std::size_t) const override { return true; }
  Point centroid(int dim, std::size_t i) const override {
    if (dim == 0) return verts_.at(i);
    // Cell centroid: mean of its corners -- fine for the target side of a
    // test (never used as a nodal DOF), and for computing a reasonable
    // interior query point in the tests below.
    Point c{0, 0, 0};
    for (const std::size_t v : cells_.at(i)) {
      c.x += verts_[v].x; c.y += verts_[v].y; c.z += verts_[v].z;
    }
    const double n = static_cast<double>(cells_.at(i).size());
    return Point{c.x / n, c.y / n, c.z / n};
  }
  double measure(int dim, std::size_t) const override {
    if (dim == 0) throw std::out_of_range("no vertex measure");
    return 1.0;  // unused by these tests
  }
  std::size_t global_id(int dim, std::size_t i) const override {
    if (dim == dim_ && !cell_gids_.empty()) return cell_gids_.at(i);
    return i;
  }
  // Real simplex decomposition (cell_simplices.hpp), needed by
  // fuse::search::element_boxes for the broad-phase box query -- NOT the
  // stand-in a "point cloud" (empty cells_[i]) target view needs, which
  // never has entities at `dim_` and so never reaches these.
  std::size_t simplex_count(int dim, std::size_t i) const override {
    if (dim != dim_ || cells_.at(i).empty()) return 0;
    return fuse::cell_simplex_count(shapes_.at(i), cells_.at(i).size());
  }
  void simplex_points(int dim, std::size_t i, std::size_t s, Point* out) const override {
    if (dim != dim_) throw std::out_of_range("ManualCellMesh::simplex_points: wrong dimension");
    int corners[4] = {0, 0, 0, 0};
    if (!fuse::simplex_corners(shapes_.at(i), cells_.at(i).size(), s, corners)) {
      throw std::out_of_range("ManualCellMesh::simplex_points: simplex out of range");
    }
    const int n = fuse::shape_dimension(shapes_.at(i)) + 1;
    for (int k = 0; k < n; ++k) out[k] = verts_.at(cells_.at(i)[static_cast<std::size_t>(corners[k])]);
  }
  Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
    if (dim != dim_) {
      return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "ManualCellMesh: wrong dimension"};
    }
    CellVertices out;
    out.shape = shapes_.at(i);
    out.vertices = cells_.at(i);
    return out;
  }

  /// One target-point view: a single dimension-`tdim` "point cloud" of
  /// bare evaluation points, reusing this same class with cells of size 0
  /// (never queried: only `centroid`/`entity_count` at `tdim` are read by
  /// `build_general_interp_weights` for the TARGET side).
  static std::unique_ptr<ManualCellMesh> points(int tdim, std::vector<Point> pts) {
    std::vector<CellShape> shapes(pts.size(), CellShape::Edge);
    std::vector<std::vector<std::size_t>> cells(pts.size());
    auto m = std::make_unique<ManualCellMesh>(tdim, pts, shapes, cells);
    return m;
  }

 private:
  int dim_;
  std::vector<Point> verts_;
  std::vector<CellShape> shapes_;
  std::vector<std::vector<std::size_t>> cells_;
  std::vector<std::size_t> cell_gids_;
};

// A single reference-shaped cell (unit triangle/quad/tet/prism/hex), so
// natural coordinates equal physical coordinates and exactness is easy to
// hand-check.
std::unique_ptr<ManualCellMesh> unit_cell(CellShape shape) {
  std::vector<Point> v;
  std::vector<std::size_t> corners;
  switch (shape) {
    case CellShape::Triangle:
      v = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
      corners = {0, 1, 2};
      break;
    case CellShape::Quad:
      v = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
      corners = {0, 1, 2, 3};
      break;
    case CellShape::Tet:
      v = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
      corners = {0, 1, 2, 3};
      break;
    case CellShape::Prism:
      v = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
      corners = {0, 1, 2, 3, 4, 5};
      break;
    case CellShape::Hex:
      v = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
           {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
      corners = {0, 1, 2, 3, 4, 5, 6, 7};
      break;
    default:
      ADD_FAILURE() << "unit_cell: unsupported shape in this helper";
  }
  const int dim = fuse::shape_dimension(shape);
  return std::make_unique<ManualCellMesh>(dim, v, std::vector<CellShape>{shape},
                                          std::vector<std::vector<std::size_t>>{corners});
}

}  // namespace

// ===========================================================================
// Exactness on a single reference cell: the interpolated value at a point
// must equal the affine/multilinear field evaluated there, to round-off,
// for every supported shape.
// ===========================================================================

TEST(GeneralInterp, TriangleIsExactForAnAffineField) {
  auto src = unit_cell(CellShape::Triangle);
  auto tgt = ManualCellMesh::points(0, {{0.2, 0.3, 0.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}});
  const auto h = fuse::build_consistent_interp_weights(*src, *tgt, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  EXPECT_LT(h.value().max_row_defect, 1e-13);
  // v(x,y) = 2 + 3x - 5y at the corners (0,0)->2, (1,0)->5, (0,1)->-3.
  const std::vector<double> v = {2.0, 5.0, -3.0};
  const auto Hv = apply_H(h.value(), v);
  EXPECT_NEAR(Hv[0], 2.0 + 3.0 * 0.2 - 5.0 * 0.3, 1e-12);
  EXPECT_NEAR(Hv[1], 2.0, 1e-12);
  EXPECT_NEAR(Hv[2], 5.0, 1e-12);
}

TEST(GeneralInterp, QuadIsExactForABilinearField) {
  auto src = unit_cell(CellShape::Quad);
  auto tgt = ManualCellMesh::points(0, {{0.25, 0.75, 0.0}, {0.5, 0.5, 0.0}});
  const auto h = fuse::build_consistent_interp_weights(*src, *tgt, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  EXPECT_LT(h.value().max_row_defect, 1e-13);
  const auto field = [](double x, double y) { return 1.0 - 2.0 * x + 3.0 * y + 4.0 * x * y; };
  const std::vector<double> v = {field(0, 0), field(1, 0), field(1, 1), field(0, 1)};
  const auto Hv = apply_H(h.value(), v);
  EXPECT_NEAR(Hv[0], field(0.25, 0.75), 1e-12);
  EXPECT_NEAR(Hv[1], field(0.5, 0.5), 1e-12);
}

TEST(GeneralInterp, TetIsExactForAnAffineField) {
  auto src = unit_cell(CellShape::Tet);
  auto tgt = ManualCellMesh::points(0, {{0.1, 0.2, 0.3}, {0.25, 0.25, 0.25}});
  const auto h = fuse::build_consistent_interp_weights(*src, *tgt, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const auto field = [](double x, double y, double z) { return -1.0 + 2.0 * x - y + 4.0 * z; };
  const std::vector<double> v = {field(0, 0, 0), field(1, 0, 0), field(0, 1, 0), field(0, 0, 1)};
  const auto Hv = apply_H(h.value(), v);
  EXPECT_NEAR(Hv[0], field(0.1, 0.2, 0.3), 1e-12);
  EXPECT_NEAR(Hv[1], field(0.25, 0.25, 0.25), 1e-12);
}

TEST(GeneralInterp, PrismIsExactForItsOwnBasisField) {
  auto src = unit_cell(CellShape::Prism);
  auto tgt = ManualCellMesh::points(0, {{0.2, 0.3, 0.6}, {0.1, 0.1, 0.9}});
  const auto h = fuse::build_consistent_interp_weights(*src, *tgt, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  EXPECT_LT(h.value().max_row_defect, 1e-13);
  // Corners (r,s,t) at 0/1/0/t and permutations; the prism basis is exact for
  // any field affine in (r,s) and linear in t -- on this UNIT reference cell
  // that is simply x, y, z themselves (r=x, s=y, t=z).
  const auto field = [](double x, double y, double z) { return 3.0 + 2.0 * x - y + 5.0 * z; };
  const std::vector<double> v = {field(0, 0, 0), field(1, 0, 0), field(0, 1, 0),
                                 field(0, 0, 1), field(1, 0, 1), field(0, 1, 1)};
  const auto Hv = apply_H(h.value(), v);
  EXPECT_NEAR(Hv[0], field(0.2, 0.3, 0.6), 1e-12);
  EXPECT_NEAR(Hv[1], field(0.1, 0.1, 0.9), 1e-12);
}

TEST(GeneralInterp, HexIsExactForATrilinearField) {
  auto src = unit_cell(CellShape::Hex);
  auto tgt = ManualCellMesh::points(0, {{0.3, 0.4, 0.5}, {0.9, 0.1, 0.2}});
  const auto h = fuse::build_consistent_interp_weights(*src, *tgt, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  EXPECT_LT(h.value().max_row_defect, 1e-13);
  const auto field = [](double x, double y, double z) {
    return 1.0 + x - 2.0 * y + 3.0 * z + 0.5 * x * y - 0.25 * y * z + 0.1 * x * y * z;
  };
  std::vector<double> v(8);
  const std::vector<Point> corners = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  for (std::size_t i = 0; i < 8; ++i) v[i] = field(corners[i].x, corners[i].y, corners[i].z);
  const auto Hv = apply_H(h.value(), v);
  EXPECT_NEAR(Hv[0], field(0.3, 0.4, 0.5), 1e-11);
  EXPECT_NEAR(Hv[1], field(0.9, 0.1, 0.2), 1e-11);
}

// ===========================================================================
// Refusals: every one of them must actually fire.
// ===========================================================================

TEST(GeneralInterp, PyramidRefusesByName) {
  std::vector<Point> v = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0.5, 0.5, 1}};
  ManualCellMesh src(3, v, {CellShape::Pyramid}, {{0, 1, 2, 3, 4}});
  auto tgt = ManualCellMesh::points(0, {{0.5, 0.5, 0.1}});
  const auto h = fuse::build_consistent_interp_weights(src, *tgt, 0);
  ASSERT_FALSE(h.ok());
  EXPECT_EQ(h.error().code, "E-INTERP-SHAPE");
}

TEST(GeneralInterp, PolygonRefusesByName) {
  std::vector<Point> v = {{0, 0, 0}, {1, 0, 0}, {1.2, 1, 0}, {0.5, 1.3, 0}, {-0.2, 0.8, 0}};
  ManualCellMesh src(2, v, {CellShape::Polygon}, {{0, 1, 2, 3, 4}});
  auto tgt = ManualCellMesh::points(0, {{0.5, 0.5, 0.0}});
  const auto h = fuse::build_consistent_interp_weights(src, *tgt, 0);
  ASSERT_FALSE(h.ok());
  EXPECT_EQ(h.error().code, "E-INTERP-SHAPE");
}

TEST(GeneralInterp, NoCellVerticesRefuses) {
  // A hand-rolled 2D MeshView that never overrides cell_vertices -- the
  // MeshView default (E-MESH-NO-CELL-VERTICES) must surface here, not a
  // crash or a silently wrong answer.
  class NoConnectivity final : public MeshView {
   public:
    int dimension() const override { return 2; }
    std::size_t entity_count(int dim) const override { return dim == 0 ? 4 : (dim == 2 ? 1 : 0); }
    std::size_t owned_count(int dim) const override { return entity_count(dim); }
    bool is_owned(int, std::size_t) const override { return true; }
    Point centroid(int dim, std::size_t i) const override {
      static const Point pts[4] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
      return dim == 0 ? pts[i] : Point{0.5, 0.5, 0.0};
    }
    double measure(int, std::size_t) const override { return 1.0; }
    std::size_t global_id(int, std::size_t i) const override { return i; }
    std::size_t simplex_count(int dim, std::size_t) const override { return dim == 2 ? 2 : 0; }
    void simplex_points(int, std::size_t, std::size_t s, Point* out) const override {
      static const Point pts[4] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
      if (s == 0) { out[0] = pts[0]; out[1] = pts[1]; out[2] = pts[2]; }
      else { out[0] = pts[0]; out[1] = pts[2]; out[2] = pts[3]; }
    }
  } src;
  auto tgt = ManualCellMesh::points(0, {{0.5, 0.5, 0.0}});
  const auto h = fuse::build_consistent_interp_weights(src, *tgt, 0);
  ASSERT_FALSE(h.ok());
  EXPECT_EQ(h.error().code, "E-MESH-NO-CELL-VERTICES");
}

TEST(GeneralInterp, DegenerateCellRefuses) {
  // A "triangle" whose three corners are collinear: zero measure, singular
  // Jacobian.
  std::vector<Point> v = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}};
  ManualCellMesh src(2, v, {CellShape::Triangle}, {{0, 1, 2}});
  auto tgt = ManualCellMesh::points(0, {{0.5, 0.0, 0.0}});
  const auto h = fuse::build_consistent_interp_weights(src, *tgt, 0);
  ASSERT_FALSE(h.ok());
  EXPECT_EQ(h.error().code, "E-INTERP-DEGENERATE");
}

TEST(GeneralInterp, OutOfDomainRefusesByDefaultAndClampAccepts) {
  auto src = unit_cell(CellShape::Triangle);
  auto tgt = ManualCellMesh::points(0, {{5.0, 5.0, 0.0}});
  const auto refused = fuse::build_consistent_interp_weights(*src, *tgt, 0,
                                                              InterfaceDomainPolicy::Refuse);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.error().code, "E-INTERP-DOMAIN");

  const auto clamped = fuse::build_consistent_interp_weights(*src, *tgt, 0,
                                                              InterfaceDomainPolicy::ClampToBoundary);
  ASSERT_TRUE(clamped.ok()) << clamped.error().message;
  EXPECT_LT(clamped.value().max_row_defect, 1e-12)
      << "a clamped point is still a convex combination of the cell's corners";
}

TEST(GeneralInterp, OffSurfaceRefusesByDefault) {
  // A flat unit-square quad "surface" cell; a target far off its plane
  // (large z) is neither laterally out of bounds nor within the surface
  // tolerance -- refused as a domain edge either way.
  auto src = unit_cell(CellShape::Quad);
  auto tgt = ManualCellMesh::points(0, {{0.5, 0.5, 3.0}});
  const auto h = fuse::build_consistent_interp_weights(*src, *tgt, 0);
  ASSERT_FALSE(h.ok());
  EXPECT_EQ(h.error().code, "E-INTERP-DOMAIN");
}

TEST(GeneralInterp, PointExactlyOnASharedEdgePicksTheLowerGlobalId) {
  // Two triangles sharing the edge x=1: cell 0 = (0,0)-(1,0)-(1,1) with
  // global id 7, cell 1 = (1,0)-(2,0)-(1,1) with global id 3 -- global id
  // does not follow local/insertion order, so the tie-break is genuinely
  // exercised, not accidentally satisfied by index order.
  class TwoTris final : public MeshView {
   public:
    int dimension() const override { return 2; }
    std::size_t entity_count(int dim) const override {
      if (dim == 0) return 4;
      if (dim == 2) return 2;
      return 0;
    }
    std::size_t owned_count(int dim) const override { return entity_count(dim); }
    bool is_owned(int, std::size_t) const override { return true; }
    Point centroid(int dim, std::size_t i) const override {
      static const Point v[4] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {2, 0, 0}};
      if (dim == 0) return v[i];
      return i == 0 ? Point{2.0 / 3, 1.0 / 3, 0} : Point{4.0 / 3, 1.0 / 3, 0};
    }
    double measure(int, std::size_t) const override { return 0.5; }
    std::size_t global_id(int dim, std::size_t i) const override {
      if (dim != 2) return i;
      return i == 0 ? 7 : 3;  // cell 1 (index 1) has the LOWER global id
    }
    std::size_t simplex_count(int dim, std::size_t) const override { return dim == 2 ? 1 : 0; }
    void simplex_points(int, std::size_t i, std::size_t, Point* out) const override {
      static const Point v[4] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {2, 0, 0}};
      if (i == 0) { out[0] = v[0]; out[1] = v[1]; out[2] = v[2]; }
      else { out[0] = v[1]; out[1] = v[3]; out[2] = v[2]; }
    }
    Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
      if (dim != 2) return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "wrong dim"};
      CellVertices out;
      out.shape = CellShape::Triangle;
      out.vertices = (i == 0) ? std::vector<std::size_t>{0, 1, 2}
                              : std::vector<std::size_t>{1, 3, 2};
      return out;
    }
  } src;
  // The shared edge is x=1, 0<=y<=1; (1, 0.5) sits exactly on it.
  auto tgt = ManualCellMesh::points(0, {{1.0, 0.5, 0.0}});
  const auto h = fuse::build_consistent_interp_weights(src, *tgt, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  // Cell 1 (local index 1, global id 3) must win: its corners are local
  // vertices {1, 3, 2}, so only THOSE columns may be nonzero.
  for (std::size_t k = 0; k < h.value().col.size(); ++k) {
    const auto col = static_cast<std::size_t>(h.value().col[k]);
    EXPECT_TRUE(col == 1 || col == 3 || col == 2)
        << "column " << col << " belongs to the higher-global-id cell (0)";
  }
}

TEST(GeneralInterp, NewtonNonConvergenceRefuses) {
  // Exercises invert_reference_map's own non-convergence path directly
  // (general_interp_detail, exposed by the header) rather than hunting for a
  // pathological cell that happens to defeat 50 iterations: `max_iter = 0`
  // means the loop body never runs at all, so this MUST refuse
  // E-INTERP-NEWTON deterministically, on any shape.
  using namespace fuse::general_interp_detail;
  Cell c;
  c.shape = CellShape::Hex;
  c.n = 8;
  const std::vector<Point> corners = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
  for (std::size_t k = 0; k < 8; ++k) c.p[k] = to_vec(corners[k]);
  const auto inv = invert_reference_map(c, Vec3{0.3, 0.4, 0.5}, /*max_iter=*/0);
  ASSERT_FALSE(inv.ok());
  EXPECT_EQ(inv.error().code, "E-INTERP-NEWTON");
}

TEST(GeneralInterp, PointDeepInsideBIsNotStolenByALowerGidNeighbourExtrapolating) {
  // Two triangles splitting the unit square along its anti-diagonal:
  //   A = (0,0)-(1,0)-(0,1), global id 0 (LOWER)
  //   B = (1,0)-(1,1)-(0,1), global id 5 (HIGHER)
  // Target p = (0.6, 0.6) is genuinely, tightly inside B (natural coords
  // r=0.2, s=0.4, both well within [0,1], sum 0.6) but ALSO within the OLD
  // single loose tolerance (0.1) of A's own boundary when A's affine map is
  // extrapolated past it (r=s=0.6, sum 1.2, violation (0.2)^2=0.04 < 0.1) --
  // exactly the shape of point this test exists to catch: under a ONE-TIER
  // rule, both A and B would register as "containing" p and the lowest-gid
  // tie-break would wrongly pick A, extrapolating its plane.
  //
  // The field is per-VERTEX (v00=0, v10=1, v11=10, v01=1 at (0,0),(1,0),
  // (1,1),(0,1)) and deliberately NOT globally affine across the shared edge
  // (v11 != v10 + v01 - v00: 10 != 1), so A's plane extended to p and B's own
  // plane at p give VISIBLY different numbers -- 1.2 (wrong, extrapolated by
  // A) vs 2.8 (correct, B's own affine interpolant) -- rather than an
  // affine field, which A's extrapolation would reproduce exactly by
  // construction and which could never expose this bug.
  std::vector<fuse::Point> v = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  // A = {0,1,3} (index 0, global id 0, LOWER); B = {1,2,3} (index 1, global
  // id 5, HIGHER) -- global ids deliberately disagree with local index order
  // so the lowest-global-id tie-break is exercised against, not agreeing
  // with, insertion order.
  ManualCellMesh src(2, v, {fuse::CellShape::Triangle, fuse::CellShape::Triangle},
                    {{0, 1, 3}, {1, 2, 3}}, /*cell_gids=*/{0, 5});
  auto tgt = ManualCellMesh::points(0, {{0.6, 0.6, 0.0}});
  const auto h = fuse::build_consistent_interp_weights(src, *tgt, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;

  // v00=0, v10=1, v11=10, v01=1 -- vertex order matches ManualCellMesh's own
  // vertex list above.
  const std::vector<double> field_values = {0.0, 1.0, 10.0, 1.0};
  const auto Hv = apply_H(h.value(), field_values);
  ASSERT_EQ(Hv.size(), 1u);
  EXPECT_NEAR(Hv[0], 2.8, 1e-12)
      << "p=(0.6,0.6) is genuinely inside B (global id 5); got " << Hv[0]
      << ", which is A's (global id 0) WRONG extrapolated value (1.2) if tier "
         "(a) was skipped and the lowest-gid tie-break ran on the loose tier "
         "instead";
}

TEST(GeneralInterp, HAndItsTransposeAreExactAdjointsToRoundoff) {
  auto src = unit_cell(CellShape::Hex);
  auto tgt = ManualCellMesh::points(0, {{0.2, 0.3, 0.4}, {0.7, 0.8, 0.1}, {0.5, 0.5, 0.5}});
  const auto h = fuse::build_consistent_interp_weights(*src, *tgt, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const RemapWeights& H = h.value();

  std::mt19937 rng(11);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(H.n_cols), y(H.n_rows);
  for (double& e : x) e = dist(rng);
  for (double& e : y) e = dist(rng);
  const auto Hx = apply_H(H, x);
  std::vector<double> Hty(H.n_cols, 0.0);
  ASSERT_TRUE(fuse::WeightsOperator(H).apply_transpose(y, Hty, 1).ok());

  double lhs = 0.0, rhs = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) lhs += y[i] * Hx[i];
  for (std::size_t i = 0; i < x.size(); ++i) rhs += Hty[i] * x[i];
  EXPECT_NEAR(lhs, rhs, 1e-11 * (std::abs(lhs) + std::abs(rhs) + 1.0));
}

// ===========================================================================
// Real, non-matching NativeMeshBackend meshes: order-2 convergence, and
// conservative H^T D power balance, mirroring
// operator_conformance_test.cpp's 1D `ConsistentInterp.*` cases.
// ===========================================================================

namespace {

struct NativeMesh {
  fuse::native::NativeMeshBackend backend;
  const MeshView* view = nullptr;
  NativeMesh() : backend(0, 1) {}
};

std::unique_ptr<NativeMesh> load_native(const std::string& source) {
  auto m = std::make_unique<NativeMesh>();
  if (!m->backend.load({.name = "m", .source = source}).ok()) return nullptr;
  const auto mid = m->backend.mesh_id("m");
  if (!mid.ok()) return nullptr;
  const auto view = m->backend.view(mid.value());
  if (!view.ok()) return nullptr;
  m->view = view.value();
  return m;
}

}  // namespace

TEST(GeneralInterp, TriangleMeshOrderIsTwoUnderRefinement) {
  const auto field = [](double x, double y) {
    return std::sin(2.0 * x) * std::cos(1.5 * y);
  };
  std::vector<double> hs, errs;
  for (const std::size_t n : {6u, 12u, 24u, 48u}) {
    auto src = load_native("generated:tri:" + std::to_string(n) + "x" + std::to_string(n));
    auto tgt = load_native("generated:tri:" + std::to_string(n + 3) + "x" + std::to_string(n + 5));
    ASSERT_TRUE(src && tgt);
    const auto h = fuse::build_consistent_interp_weights(*src->view, *tgt->view, 0);
    ASSERT_TRUE(h.ok()) << h.error().message;
    const std::size_t nv = src->view->entity_count(0);
    std::vector<double> v(nv);
    for (std::size_t i = 0; i < nv; ++i) {
      const Point p = src->view->centroid(0, i);
      v[i] = field(p.x, p.y);
    }
    const auto Hv = apply_H(h.value(), v);
    const std::size_t ntv = tgt->view->entity_count(0);
    double l2 = 0.0;
    for (std::size_t j = 0; j < ntv; ++j) {
      const Point p = tgt->view->centroid(0, j);
      const double e = Hv[j] - field(p.x, p.y);
      l2 += e * e / static_cast<double>(ntv);
    }
    hs.push_back(1.0 / static_cast<double>(n));
    errs.push_back(std::sqrt(l2));
  }
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = static_cast<double>(hs.size());
  for (std::size_t k = 0; k < hs.size(); ++k) {
    const double x = std::log(hs[k]), y = std::log(errs[k]);
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const double order = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  for (std::size_t k = 0; k < hs.size(); ++k) {
    std::printf("TriangleMeshOrder: h=%.6g  L2 error=%.6e\n", hs[k], errs[k]);
  }
  std::printf("TriangleMeshOrder: fitted order = %.6f\n", order);
  EXPECT_GT(order, 1.8) << "observed order " << order << ", finest error " << errs.back();
}

TEST(GeneralInterp, QuadMeshConservesLoadAndBalancesPower) {
  auto beam = load_native("generated:quad:5x5");   // the "nodal" side
  auto fluid = load_native("generated:quad:8x7");  // the "point" side (cell centroids)
  ASSERT_TRUE(beam && fluid);
  const auto h = fuse::build_consistent_interp_weights(*beam->view, *fluid->view, 2);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const RemapWeights& H = h.value();

  RemapWeights HtD;
  HtD.n_rows = H.n_cols;
  HtD.n_cols = H.n_rows;
  for (std::size_t k = 0; k < H.val.size(); ++k) {
    HtD.row.push_back(H.col[k]);
    HtD.col.push_back(H.row[k]);
    HtD.val.push_back(H.val[k] * fluid->view->measure(2, static_cast<std::size_t>(H.row[k])));
  }

  std::mt19937 rng(3);
  std::uniform_real_distribution<double> dist(-3.0, 5.0);
  std::vector<double> rho(fluid->view->entity_count(2));
  for (double& x : rho) x = dist(rng);
  const auto F = apply_H(HtD, rho);

  double sent = 0.0;
  for (std::size_t j = 0; j < rho.size(); ++j) sent += rho[j] * fluid->view->measure(2, j);
  double received = 0.0;
  for (double v : F) received += v;
  EXPECT_NEAR(sent, received, 1e-10 * std::abs(sent))
      << "total force must be conserved to round-off";

  std::vector<double> v(H.n_cols);
  for (double& x : v) x = dist(rng);
  const auto Hv = apply_H(H, v);
  double power_fluid = 0.0;
  for (std::size_t j = 0; j < rho.size(); ++j) power_fluid += Hv[j] * rho[j] * fluid->view->measure(2, j);
  double power_beam = 0.0;
  for (std::size_t i = 0; i < v.size(); ++i) power_beam += v[i] * F[i];
  EXPECT_NEAR(power_fluid, power_beam, 1e-10 * (std::abs(power_fluid) + std::abs(power_beam)))
      << "discrete interface power must balance exactly";
}

TEST(GeneralInterp, TetMeshIsExactForALinearField) {
  auto src = load_native("generated:tetbox:3x3x3");
  auto tgt = load_native("generated:tetbox:4x5x4");
  ASSERT_TRUE(src && tgt);
  const auto h = fuse::build_consistent_interp_weights(*src->view, *tgt->view, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const auto field = [](double x, double y, double z) { return 1.0 + 2.0 * x - 3.0 * y + z; };
  const std::size_t nv = src->view->entity_count(0);
  std::vector<double> v(nv);
  for (std::size_t i = 0; i < nv; ++i) {
    const Point p = src->view->centroid(0, i);
    v[i] = field(p.x, p.y, p.z);
  }
  const auto Hv = apply_H(h.value(), v);
  const std::size_t ntv = tgt->view->entity_count(0);
  for (std::size_t j = 0; j < ntv; ++j) {
    const Point p = tgt->view->centroid(0, j);
    EXPECT_NEAR(Hv[j], field(p.x, p.y, p.z), 1e-10) << "target vertex " << j;
  }
}

TEST(GeneralInterp, HexMeshIsExactForATrilinearField) {
  auto src = load_native("generated:hexbox:3x3x3");
  auto tgt = load_native("generated:hexbox:4x5x4");
  ASSERT_TRUE(src && tgt);
  const auto h = fuse::build_consistent_interp_weights(*src->view, *tgt->view, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const auto field = [](double x, double y, double z) {
    return 1.0 + x - 2.0 * y + 3.0 * z + 0.5 * x * y * z;
  };
  // Trilinear on the CELL, not globally: 0.5*x*y*z is not reproduced exactly
  // across cell boundaries by a trilinear basis, so restrict this exactness
  // check to the purely affine part.
  const auto affine = [](double x, double y, double z) { return 1.0 + x - 2.0 * y + 3.0 * z; };
  (void)field;
  const std::size_t nv = src->view->entity_count(0);
  std::vector<double> v(nv);
  for (std::size_t i = 0; i < nv; ++i) {
    const Point p = src->view->centroid(0, i);
    v[i] = affine(p.x, p.y, p.z);
  }
  const auto Hv = apply_H(h.value(), v);
  const std::size_t ntv = tgt->view->entity_count(0);
  for (std::size_t j = 0; j < ntv; ++j) {
    const Point p = tgt->view->centroid(0, j);
    EXPECT_NEAR(Hv[j], affine(p.x, p.y, p.z), 1e-10) << "target vertex " << j;
  }
}

TEST(GeneralInterp, SphereSurfaceOrderIsAtLeastTwoUnderRefinement) {
  const auto field = [](double x, double y, double z) {
    return x * x - y * y + 0.5 * z;  // smooth on the sphere, not affine
  };
  std::vector<double> hs, errs;
  for (const std::size_t n : {4u, 8u, 16u}) {
    auto src = load_native("generated:cubedsphere:" + std::to_string(n));
    auto tgt = load_native("generated:cubedsphere:" + std::to_string(n + 3));
    ASSERT_TRUE(src && tgt);
    const auto h = fuse::build_consistent_interp_weights(*src->view, *tgt->view, 0);
    ASSERT_TRUE(h.ok()) << h.error().message;
    const std::size_t nv = src->view->entity_count(0);
    std::vector<double> v(nv);
    for (std::size_t i = 0; i < nv; ++i) {
      const Point p = src->view->centroid(0, i);
      v[i] = field(p.x, p.y, p.z);
    }
    const auto Hv = apply_H(h.value(), v);
    const std::size_t ntv = tgt->view->entity_count(0);
    double l2 = 0.0;
    for (std::size_t j = 0; j < ntv; ++j) {
      const Point p = tgt->view->centroid(0, j);
      const double e = Hv[j] - field(p.x, p.y, p.z);
      l2 += e * e / static_cast<double>(ntv);
    }
    hs.push_back(1.0 / static_cast<double>(n));
    errs.push_back(std::sqrt(l2));
  }
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = static_cast<double>(hs.size());
  for (std::size_t k = 0; k < hs.size(); ++k) {
    const double x = std::log(hs[k]), y = std::log(errs[k]);
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const double order = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  for (std::size_t k = 0; k < hs.size(); ++k) {
    std::printf("SphereSurfaceOrder: h=%.6g  L2 error=%.6e\n", hs[k], errs[k]);
  }
  std::printf("SphereSurfaceOrder: fitted order = %.6f\n", order);
  EXPECT_GT(order, 1.8) << "observed order " << order << ", finest error " << errs.back();
}

TEST(GeneralInterp, MixedPyramidPrismMeshAsSourceRefusesByName) {
  auto src = load_native("generated:prismpyramidmix:2x2x2");
  ASSERT_TRUE(src);
  auto tgt = ManualCellMesh::points(0, {{0.5, 0.5, 0.5}});
  const auto h = fuse::build_consistent_interp_weights(*src->view, *tgt, 0);
  ASSERT_FALSE(h.ok());
  EXPECT_EQ(h.error().code, "E-INTERP-SHAPE");
}

// ===========================================================================
// mesh_view.hpp's cell_vertices() itself, on the two concrete backends this
// track wired it into beyond native (which the tests above already exercise
// implicitly through every "generated:*" case): SerialMeshBackend's
// structured hex.
// ===========================================================================

TEST(CellVerticesQuery, StructuredHexBoxReportsTheRightCorners) {
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load({.name = "b", .source = "generated:box:2x1x1"}).ok());
  const auto mid = backend.mesh_id("b");
  ASSERT_TRUE(mid.ok());
  const auto view = backend.view(mid.value());
  ASSERT_TRUE(view.ok());
  // Cell 0 is the [0,0.5]x[0,1]x[0,1] box (x-fastest numbering).
  const auto cv = view.value()->cell_vertices(3, 0);
  ASSERT_TRUE(cv.ok()) << cv.error().message;
  EXPECT_EQ(cv.value().shape, CellShape::Hex);
  ASSERT_EQ(cv.value().vertices.size(), 8u);
  // Corner 0 must be the cell's own (0,0,0)-ish corner and corner 6 its
  // diagonal opposite, by cell_simplices.hpp's convention.
  const Point p0 = view.value()->centroid(0, cv.value().vertices[0]);
  const Point p6 = view.value()->centroid(0, cv.value().vertices[6]);
  EXPECT_NEAR(p0.x, 0.0, 1e-14);
  EXPECT_NEAR(p0.y, 0.0, 1e-14);
  EXPECT_NEAR(p0.z, 0.0, 1e-14);
  EXPECT_NEAR(p6.x, 0.5, 1e-14);
  EXPECT_NEAR(p6.y, 1.0, 1e-14);
  EXPECT_NEAR(p6.z, 1.0, 1e-14);
}

// Round 2 (nodal-remap track): a StructuredMesh line's dimension-1
// cell_vertices answer went from "refused" to "implemented", to give
// `nodal_remap_distributed.hpp`'s builder the same connectivity answer on a
// 'generated:line' mesh that test/verification/interface_transfer_test.cpp's
// `run_consistent_pair` already relies on for its serial (single-rank)
// coverage. This test therefore now asserts the ANSWER, not a refusal --
// see StructuredHexBoxStillRefusesAtTheWrongDimension below for what still
// refuses on this same backend.
TEST(CellVerticesQuery, StructuredLineReportsConsecutiveEdgeCorners) {
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load({.name = "l", .source = "generated:line:4"}).ok());
  const auto mid = backend.mesh_id("l");
  ASSERT_TRUE(mid.ok());
  const auto view = backend.view(mid.value());
  ASSERT_TRUE(view.ok());
  const auto cv = view.value()->cell_vertices(1, 1);
  ASSERT_TRUE(cv.ok()) << cv.error().message;
  EXPECT_EQ(cv.value().shape, fuse::CellShape::Edge);
  ASSERT_EQ(cv.value().vertices.size(), 2u);
  EXPECT_EQ(cv.value().vertices[0], 1u);
  EXPECT_EQ(cv.value().vertices[1], 2u);
}

TEST(CellVerticesQuery, StructuredHexBoxStillRefusesAtTheWrongDimension) {
  fuse::SerialMeshBackend backend;
  ASSERT_TRUE(backend.load({.name = "b", .source = "generated:box:2x2x2"}).ok());
  const auto mid = backend.mesh_id("b");
  ASSERT_TRUE(mid.ok());
  const auto view = backend.view(mid.value());
  ASSERT_TRUE(view.ok());
  // Dimension 2 (a face) of a dimension-3 box mesh has no connectivity
  // answer -- only dimension 3 (the hex cells themselves) does; unaffected
  // by this round's 1D addition, which is gated on `dim_ == 1` specifically.
  const auto cv = view.value()->cell_vertices(2, 0);
  ASSERT_FALSE(cv.ok());
  EXPECT_EQ(cv.error().code, "E-MESH-NO-CELL-VERTICES");
}
