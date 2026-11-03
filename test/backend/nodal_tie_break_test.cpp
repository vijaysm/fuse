// The deterministic, global-id tie-break
// for a point exactly on a shared face/edge/vertex, exercised at EVERY
// boundary vertex of a structured mesh across several resolutions, cross-
// checked between the two code paths that must agree on it --
// `fuse::build_general_interp_weights` (general_interp.hpp, the serial
// builder) and `fuse::build_distributed_interp_rows`
// (nodal_remap_distributed.hpp, `comm_handle == kNullComm` here, exactly
// as a multi-rank run's own replicated-source assembly feeds it) -- plus a
// 1-ULP perturbation of the query point in each coordinate direction. No
// MPI symbol anywhere in this translation unit (mirrors
// nodal_remap_distributed_nullcomm_test.cpp's own convention).

#include "fuse/backend/general_interp.hpp"
#include "fuse/backend/nodal_remap_distributed.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/search/types.hpp"

namespace {

using fuse::CellShape;
using fuse::MeshView;
using fuse::Point;
using fuse::Result;

/// `n x n` unit Quad cells over `[0, n] x [0, n]`: `(n+1)^2` vertices,
/// corner order 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1) within each cell, matching
/// general_interp.hpp's own CCW bilinear convention. Cell global ids are
/// numbered in REVERSE row-major order (the LAST cell by insertion order
/// gets global id 0) so the lowest-global-id tie-break is genuinely
/// exercised against, never accidentally satisfied by, iteration order --
/// the same deliberate choice `nodal_interp_test.cpp`'s own
/// `PointExactlyOnASharedEdgePicksTheLowerGlobalId` makes.
class StructuredQuadMesh final : public MeshView {
 public:
  explicit StructuredQuadMesh(std::size_t n) : n_(n) {
    const std::size_t nv = (n_ + 1) * (n_ + 1);
    verts_.resize(nv);
    for (std::size_t j = 0; j <= n_; ++j) {
      for (std::size_t i = 0; i <= n_; ++i) {
        verts_[j * (n_ + 1) + i] = Point{static_cast<double>(i), static_cast<double>(j), 0.0};
      }
    }
    const std::size_t nc = n_ * n_;
    cells_.resize(nc);
    for (std::size_t j = 0; j < n_; ++j) {
      for (std::size_t i = 0; i < n_; ++i) {
        const std::size_t c = j * n_ + i;
        const std::size_t v00 = j * (n_ + 1) + i;
        const std::size_t v10 = v00 + 1;
        const std::size_t v11 = v10 + (n_ + 1);
        const std::size_t v01 = v00 + (n_ + 1);
        cells_[c] = {v00, v10, v11, v01};
      }
    }
  }

  int dimension() const override { return 2; }
  std::size_t entity_count(int dim) const override {
    if (dim == 0) return verts_.size();
    if (dim == 2) return cells_.size();
    return 0;
  }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int, std::size_t) const override { return true; }
  Point centroid(int dim, std::size_t i) const override { return verts_.at(i); }
  double measure(int dim, std::size_t) const override {
    if (dim == 0) throw std::out_of_range("StructuredQuadMesh: no vertex measure");
    return 1.0;
  }
  // Reverse row-major: the cell visited LAST (highest local index) gets
  // the LOWEST global id.
  std::size_t global_id(int dim, std::size_t i) const override {
    if (dim == 2) return cells_.size() - 1 - i;
    return i;
  }
  std::size_t simplex_count(int dim, std::size_t) const override { return dim == 2 ? 2 : 0; }
  void simplex_points(int dim, std::size_t i, std::size_t s, Point* out) const override {
    if (dim != 2) throw std::out_of_range("StructuredQuadMesh::simplex_points: wrong dimension");
    const auto& c = cells_.at(i);
    // Triangulate 0-1-2 / 0-2-3 (the 0-2 diagonal), matching
    // cell_simplices.hpp's own Quad convention.
    if (s == 0) { out[0] = verts_[c[0]]; out[1] = verts_[c[1]]; out[2] = verts_[c[2]]; }
    else { out[0] = verts_[c[0]]; out[1] = verts_[c[2]]; out[2] = verts_[c[3]]; }
  }
  Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
    if (dim != 2) return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "wrong dimension"};
    CellVertices out;
    out.shape = CellShape::Quad;
    out.vertices.assign(cells_.at(i).begin(), cells_.at(i).end());
    return out;
  }

  std::size_t n() const { return n_; }
  Point vertex_point(std::size_t i) const { return verts_.at(i); }

 private:
  std::size_t n_;
  std::vector<Point> verts_;
  std::vector<std::array<std::size_t, 4>> cells_;
};

/// A single bare evaluation point (dimension 0), reused for one query at a
/// time -- both builders under test only ever read `entity_count`/
/// `is_owned`/`centroid`/`global_id` at dimension 0 off this view.
class OnePoint final : public MeshView {
 public:
  explicit OnePoint(Point p) : p_(p) {}
  int dimension() const override { return 0; }
  std::size_t entity_count(int dim) const override { return dim == 0 ? 1 : 0; }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int, std::size_t) const override { return true; }
  Point centroid(int, std::size_t) const override { return p_; }
  double measure(int, std::size_t) const override {
    throw std::out_of_range("OnePoint: no measure");
  }
  std::size_t global_id(int, std::size_t i) const override { return i; }
  std::size_t simplex_count(int, std::size_t) const override { return 0; }
  void simplex_points(int, std::size_t, std::size_t, Point*) const override {}
  Result<CellVertices> cell_vertices(int, std::size_t) const override {
    return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "OnePoint: no cells"};
  }

 private:
  Point p_;
};

/// A hand-built, irregular Quad mesh (not a structured grid, unlike
/// `StructuredQuadMesh` above): used below to build ONE genuinely degenerate
/// cell (a rank-deficient Jacobian for every natural coordinate, collapsed
/// along an off-axis diagonal so its own bounding box still varies in both
/// x and y, exactly like `nodal_remap_distributed_nullcomm_test.cpp`'s own
/// `ManualQuadMesh`/decoy construction for the distributed-side
/// regression, `DegenerateDecoyCellDoesNotAbortAnAlreadyFoundMatch`) next to
/// two genuine unit-square cells. `StructuredQuadMesh` cannot express this:
/// every one of its own cells is axis-aligned and non-degenerate by
/// construction.
class IrregularQuadMesh final : public MeshView {
 public:
  IrregularQuadMesh(std::vector<Point> verts, std::vector<std::array<std::size_t, 4>> cells)
      : verts_(std::move(verts)), cells_(std::move(cells)) {}
  int dimension() const override { return 2; }
  std::size_t entity_count(int dim) const override {
    if (dim == 0) return verts_.size();
    if (dim == 2) return cells_.size();
    return 0;
  }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int, std::size_t) const override { return true; }
  Point centroid(int dim, std::size_t i) const override {
    if (dim == 0) return verts_.at(i);
    Point c{0, 0, 0};
    for (const std::size_t v : cells_.at(i)) { c.x += verts_[v].x; c.y += verts_[v].y; }
    return Point{c.x / 4.0, c.y / 4.0, 0.0};
  }
  double measure(int dim, std::size_t) const override {
    if (dim == 0) throw std::out_of_range("IrregularQuadMesh: no vertex measure");
    return 1.0;  // unused by this test
  }
  std::size_t global_id(int, std::size_t i) const override { return i; }
  std::size_t simplex_count(int dim, std::size_t) const override {
    return dim == 2 ? fuse::cell_simplex_count(CellShape::Quad, 4) : 0;
  }
  void simplex_points(int dim, std::size_t i, std::size_t s, Point* out) const override {
    if (dim != 2) throw std::out_of_range("IrregularQuadMesh: wrong dimension");
    int corners[4] = {0, 0, 0, 0};
    if (!fuse::simplex_corners(CellShape::Quad, 4, s, corners)) {
      throw std::out_of_range("IrregularQuadMesh::simplex_points: simplex out of range");
    }
    for (int k = 0; k < 3; ++k) out[k] = verts_.at(cells_.at(i)[static_cast<std::size_t>(corners[k])]);
  }
  Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
    if (dim != 2) return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "IrregularQuadMesh: wrong dimension"};
    CellVertices out;
    out.shape = CellShape::Quad;
    out.vertices.assign(cells_.at(i).begin(), cells_.at(i).end());
    return out;
  }

 private:
  std::vector<Point> verts_;
  std::vector<std::array<std::size_t, 4>> cells_;
};

/// The set of GLOBAL cell ids genuinely containing a point, by brute-force
/// geometric inspection (independent of either builder under test): a
/// quad `[i, i+1] x [j, j+1]` contains `(x, y)` within `eps` of its own
/// closed region.
std::set<std::size_t> genuine_owners(const StructuredQuadMesh& mesh, double x, double y,
                                     double eps) {
  std::set<std::size_t> owners;
  const std::size_t n = mesh.n();
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      const double lo_x = static_cast<double>(i), hi_x = static_cast<double>(i + 1);
      const double lo_y = static_cast<double>(j), hi_y = static_cast<double>(j + 1);
      if (x >= lo_x - eps && x <= hi_x + eps && y >= lo_y - eps && y <= hi_y + eps) {
        owners.insert(mesh.global_id(2, j * n + i));
      }
    }
  }
  return owners;
}

/// Winning cell's global id from each builder, for query point `(x, y)` --
/// ASSERT-fails (via ADD_FAILURE, returning a sentinel) if either builder
/// refuses, so a caller's own loop can keep reporting every vertex rather
/// than aborting the whole resolution on the first failure.
struct Winners {
  std::size_t serial_gid = 0;
  std::size_t distributed_gid = 0;
  bool ok = false;
};

/// Finds the mesh cell whose own corner-VERTEX-global-id set equals
/// `col_gids` exactly -- not "a cell containing one of these vertices"
/// (ambiguous: a shared corner/edge vertex belongs to more than one
/// cell), but the one cell whose FULL 4-corner set this row's own 4
/// (row, col, val) entries -- one per corner, including the three
/// zero-weight ones a point exactly on a vertex produces -- was built
/// from. `build_general_interp_weights`/`build_distributed_interp_rows`
/// both push every corner of the winning cell regardless of its weight's
/// magnitude (general_interp.hpp's own "for k < cells[use_cell].n" loop),
/// so this set always has exactly as many entries as the winning cell has
/// corners (4, for a Quad).
std::size_t cell_matching_corner_set(const StructuredQuadMesh& mesh,
                                     const std::set<std::size_t>& col_gids) {
  for (std::size_t c = 0; c < mesh.entity_count(2); ++c) {
    const auto cv = mesh.cell_vertices(2, c);
    std::set<std::size_t> this_cell_gids;
    for (const std::size_t v : cv.value().vertices) this_cell_gids.insert(mesh.global_id(0, v));
    if (this_cell_gids == col_gids) return mesh.global_id(2, c);
  }
  ADD_FAILURE() << "no cell's corner set matches the reported columns (count "
                << col_gids.size() << ")";
  return static_cast<std::size_t>(-1);
}

Winners locate(const StructuredQuadMesh& mesh, double x, double y) {
  Winners w;
  OnePoint pt(Point{x, y, 0.0});

  const auto serial_r = fuse::build_general_interp_weights(mesh, pt, /*tgt_dim=*/0);
  if (!serial_r.ok()) {
    ADD_FAILURE() << "serial builder refused at (" << x << ", " << y
                  << "): " << serial_r.error().message;
    return w;
  }
  EXPECT_FALSE(serial_r.value().col.empty());
  if (serial_r.value().col.empty()) return w;
  // The serial builder's columns are source VERTEX LOCAL indices; map to
  // global ids (identity on this mesh, but go through global_id() rather
  // than assume it, matching the distributed builder's own convention).
  std::set<std::size_t> serial_gids;
  for (const std::int32_t col : serial_r.value().col) {
    serial_gids.insert(mesh.global_id(0, static_cast<std::size_t>(col)));
  }
  w.serial_gid = cell_matching_corner_set(mesh, serial_gids);

  const auto dist_r = fuse::build_distributed_interp_rows(mesh, pt, /*eval_dim=*/0,
                                                           fuse::search::kNullComm);
  if (!dist_r.ok()) {
    ADD_FAILURE() << "distributed builder refused at (" << x << ", " << y
                  << "): " << dist_r.error().message;
    return w;
  }
  EXPECT_EQ(dist_r.value().rows.size(), 1u);
  if (dist_r.value().rows.size() != 1u) return w;
  // The distributed row's own columns are SOURCE VERTEX GLOBAL ids
  // directly already.
  const std::set<std::size_t> dist_gids(dist_r.value().rows[0].src_global_ids.begin(),
                                        dist_r.value().rows[0].src_global_ids.end());
  w.distributed_gid = cell_matching_corner_set(mesh, dist_gids);
  w.ok = true;
  return w;
}

}  // namespace

TEST(NodalTieBreak, EveryBoundaryVertexOfAStructuredMeshAtSeveralResolutionsIsSingleOwned) {
  for (const std::size_t n : {2u, 3u, 4u, 8u}) {
    StructuredQuadMesh mesh(n);
    const std::size_t nv1 = n + 1;
    for (std::size_t j = 0; j <= n; ++j) {
      for (std::size_t i = 0; i <= n; ++i) {
        const bool is_boundary = (i == 0 || i == n || j == 0 || j == n);
        if (!is_boundary) continue;  // interior vertices are covered separately below
        const double x = static_cast<double>(i), y = static_cast<double>(j);
        const Winners w = locate(mesh, x, y);
        ASSERT_TRUE(w.ok) << "n=" << n << " vertex (" << i << ", " << j << ")";

        // The brute-force geometric owner set: a corner vertex has exactly
        // one genuine owner, an edge (non-corner boundary) vertex exactly
        // two.
        const std::set<std::size_t> owners = genuine_owners(mesh, x, y, 1e-9);
        const bool corner = (i == 0 || i == n) && (j == 0 || j == n);
        ASSERT_EQ(owners.size(), corner ? 1u : 2u)
            << "n=" << n << " vertex (" << i << ", " << j << "): brute-force owner count";
        const std::size_t expected_gid = *owners.begin();  // lowest, since std::set is ordered
        EXPECT_EQ(w.serial_gid, expected_gid)
            << "n=" << n << " vertex (" << i << ", " << j << "): serial builder's winner";
        EXPECT_EQ(w.distributed_gid, expected_gid)
            << "n=" << n << " vertex (" << i << ", " << j << "): distributed builder's winner";
        EXPECT_EQ(w.serial_gid, w.distributed_gid)
            << "n=" << n << " vertex (" << i << ", " << j << "): the two builders disagree";
      }
    }
    (void)nv1;
  }
}

TEST(NodalTieBreak, TheMeshsOwnCornerGlobalIdZeroIsSingleOwnedAtEveryResolution) {
  // The corner vertex global id 0 specifically
  // -- StructuredQuadMesh numbers vertices identity (dim 0's global_id
  // returns `i`), so vertex local index 0 (physical corner (0, 0)) IS
  // global id 0 at every resolution.
  for (const std::size_t n : {2u, 3u, 4u, 8u}) {
    StructuredQuadMesh mesh(n);
    ASSERT_EQ(mesh.global_id(0, 0), 0u);
    const Winners w = locate(mesh, 0.0, 0.0);
    ASSERT_TRUE(w.ok) << "n=" << n;
    // The ONLY cell touching (0,0) is the one with local index 0 (reverse
    // numbering: global id n*n - 1, the HIGHEST id on this mesh) -- a
    // single candidate, no tie at all, but still must resolve the same
    // way on both paths.
    const std::size_t expected_gid = mesh.global_id(2, 0);
    EXPECT_EQ(w.serial_gid, expected_gid) << "n=" << n;
    EXPECT_EQ(w.distributed_gid, expected_gid) << "n=" << n;
  }
}

TEST(NodalTieBreak, OneUlpPerturbationOfASharedInteriorVertexStillResolvesConsistently) {
  // An INTERIOR shared vertex where FOUR cells genuinely meet (n=4, vertex
  // (2, 2), the mesh's own centre) -- the richest tie this mesh shape can
  // produce. Perturbing by 1 ULP in each of the four axis directions
  // either keeps the point within every tier's tolerance of all four
  // cells (still a four-way tie, same winner) or moves it just inside
  // exactly one neighbour -- either way, both builders must agree with
  // each other, and must not refuse or crash.
  StructuredQuadMesh mesh(4);
  const double cx = 2.0, cy = 2.0;
  const std::set<std::size_t> owners = genuine_owners(mesh, cx, cy, 1e-9);
  ASSERT_EQ(owners.size(), 4u) << "the mesh centre is not a genuine 4-way shared vertex";
  const std::size_t expected_gid = *owners.begin();

  const double up = std::nextafter(cx, cx + 1.0);
  const double down = std::nextafter(cx, cx - 1.0);
  const std::vector<std::pair<double, double>> perturbed = {
      {up, cy}, {down, cy}, {cx, up}, {cx, down}, {up, up}, {down, down}, {up, down}, {down, up}};

  for (const auto& [px, py] : perturbed) {
    const Winners w = locate(mesh, px, py);
    ASSERT_TRUE(w.ok) << "perturbed (" << px << ", " << py << ")";
    EXPECT_EQ(w.serial_gid, w.distributed_gid)
        << "perturbed (" << px << ", " << py << "): the two builders disagree by 1 ULP of "
                                                 "perturbation alone";
    // A 1-ULP nudge is many orders of magnitude below kRefEpsTight (a
    // fixed multiple of kNewtonTol, itself 1e-12): the winner must be
    // UNCHANGED from the unperturbed centre's own winner, not merely
    // "some" cell.
    EXPECT_EQ(w.serial_gid, expected_gid)
        << "perturbed (" << px << ", " << py << "): winner changed under a 1-ULP nudge";
  }
}


// A per-candidate invert_reference_map failure SKIPS just that candidate
// instead of aborting the whole evaluation point, once a broad-phase box
// query has narrowed the candidate set to the genuinely nearby cells
// (nodal_remap_distributed.cpp, build_distributed_interp_rows does this, and
// so does this file's other builder under test,
// fuse::build_general_interp_weights, general_interp.hpp): its per-candidate
// loop, just below its own IDENTICAL box-index broad phase, must not do
// `if (!inv.ok()) return Failure{...};` UNCONDITIONALLY -- a candidate that
// merely FAILS TO CONVERGE, however irrelevant, would abort the whole point
// even after a genuine tight match was already found earlier in the SAME
// candidate list. Here it is triggered by a NEARBY degenerate cell (one the
// broad phase cannot exclude, by construction -- see IrregularQuadMesh's own
// comment above).
//
// Without the skip, the builder refuses --
//   E-INTERP-DEGENERATE: build_general_interp_weights: target point 0,
//   candidate cell 2: invert_reference_map: the cell's Jacobian is
//   singular to working precision (a zero-measure or degenerate cell);
//   the reference map cannot be inverted
// even though candidate cell 0 (the genuine tight match, the target
// point's own centroid) is tested first and already satisfies tier (a).
//
// With the skip, the degenerate decoy is silently skipped, exactly as
// build_distributed_interp_rows does, and the genuine tight match (cell 0)
// decides the row.
TEST(NodalTieBreak, DegenerateNearbyCandidateAbortsAnAlreadyFoundTightMatch) {
  // The EXACT construction nodal_remap_distributed_nullcomm_test.cpp's own
  // DegenerateDecoyCellDoesNotAbortAnAlreadyFoundMatch uses on the
  // distributed builder, fed here to the serial one instead.
  const std::vector<Point> verts = {
      {0.0, 0.0, 0.0},  // v0
      {1.0, 0.0, 0.0},  // v1
      {1.0, 1.0, 0.0},  // v2
      {0.0, 1.0, 0.0},  // v3
      {2.0, 0.0, 0.0},  // v4
      {2.0, 1.0, 0.0},  // v5
  };
  const std::vector<std::array<std::size_t, 4>> cells = {
      {0, 1, 2, 3},  // gid 0: unit square [0, 1] x [0, 1] -- the genuine tight match below
      {1, 4, 5, 2},  // gid 1: unit square [1, 2] x [0, 1]
      {0, 5, 5, 0},  // gid 2: the degenerate decoy, collapsed along the (0,0)-(2,1) diagonal
  };
  IrregularQuadMesh src(verts, cells);
  OnePoint tgt(Point{0.5, 0.5, 0.0});  // the centroid of cell 0, a genuine tight match

  const auto rows_r = fuse::build_general_interp_weights(src, tgt, /*tgt_dim=*/0);
  ASSERT_TRUE(rows_r.ok()) << rows_r.error().message;
  ASSERT_EQ(rows_r.value().col.size(), 4u);
  // The winning cell must be the genuine tight match (cell 0's own four
  // corner vertices, gids 0-3), never the degenerate decoy (gids 0, 5) --
  // same assertion nodal_remap_distributed_nullcomm_test.cpp's own sibling
  // case makes for the distributed builder.
  std::set<std::int32_t> cols(rows_r.value().col.begin(), rows_r.value().col.end());
  EXPECT_EQ(cols, (std::set<std::int32_t>{0, 1, 2, 3}));
}
