// Single-rank/null-communicator proof, primitive level (the Runner-level half
// is a base-build-configuration Runner test): this file exercises
// `fuse::build_distributed_interp_rows`/`apply_distributed_H`/
// `apply_distributed_HT` (nodal_remap_distributed.hpp) directly, with
// `comm_handle == fuse::search::kNullComm`, in a PLAIN `fuse_add_gtest`
// target that links no MPI at all and RUNS in this sandbox (unlike
// test/parallel/nodal_remap_distributed_test.cpp, which `#include <mpi.h>`
// and can only be compiled, never executed, here -- see that file's own
// header). It proves `allgatherv_records`'s `comm_handle == kNullComm`
// branch (nodal_remap_distributed.cpp) is not just dead code that happens
// to compile: the exact same public functions the MPI test calls produce
// the mathematically correct answer here too, with zero MPI symbols
// touched anywhere in this translation unit.
//
// The mesh is a plain, local (no MPI, no RankSet) 1D line -- every entity
// trivially "owned" by the one and only process -- built by hand below
// rather than reusing test/support/partitioned_test_backend.hpp's
// GeometricLineMesh, which unconditionally `#include <mpi.h>` and so cannot
// be used from a target that has none.

#include "fuse/backend/nodal_remap_distributed.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/cell_simplices.hpp"
#include "fuse/backend/consistent_interp.hpp"
#include "fuse/backend/mesh_view.hpp"
#include "fuse/base/config.hpp"
#include "fuse/search/types.hpp"

namespace {

using fuse::CellShape;
using fuse::MeshView;
using fuse::Point;
using fuse::Result;

/// `n_cells` unit-spaced Edge cells over `[x_lo, x_hi]`: `n_cells + 1`
/// vertices, global ids equal to local index (the only rank there is owns
/// everything), exactly the shape `general_interp_detail`'s Edge basis
/// expects (`cell_vertices` returns 2 corners in ascending-x order).
class LocalLineMesh final : public MeshView {
 public:
  LocalLineMesh(std::size_t n_cells, double x_lo, double x_hi) : n_cells_(n_cells) {
    verts_.reserve(n_cells + 1);
    for (std::size_t i = 0; i <= n_cells; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(n_cells);
      verts_.push_back(Point{x_lo + t * (x_hi - x_lo), 0.0, 0.0});
    }
  }

  int dimension() const override { return 1; }
  std::size_t entity_count(int dim) const override {
    if (dim == 0) return verts_.size();
    if (dim == 1) return n_cells_;
    return 0;
  }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int, std::size_t) const override { return true; }  // one rank: owns everything
  Point centroid(int dim, std::size_t i) const override {
    if (dim == 0) return verts_.at(i);
    return Point{(verts_.at(i).x + verts_.at(i + 1).x) / 2.0, 0.0, 0.0};
  }
  double measure(int dim, std::size_t i) const override {
    if (dim == 0) throw std::out_of_range("LocalLineMesh: no vertex measure");
    return verts_.at(i + 1).x - verts_.at(i).x;
  }
  std::size_t global_id(int, std::size_t i) const override { return i; }
  std::size_t simplex_count(int dim, std::size_t) const override { return dim == 1 ? 1 : 0; }
  void simplex_points(int dim, std::size_t i, std::size_t, Point* out) const override {
    if (dim != 1) throw std::out_of_range("LocalLineMesh: wrong dimension");
    out[0] = verts_.at(i);
    out[1] = verts_.at(i + 1);
  }
  Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
    if (dim != 1) return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "LocalLineMesh: wrong dimension"};
    return CellVertices{CellShape::Edge, {i, i + 1}};
  }

 private:
  std::size_t n_cells_;
  std::vector<Point> verts_;
};

/// A minimal, hand-built 2D MeshView (Quad cells only) -- enough control
/// over geometry and connectivity to place a genuinely DEGENERATE decoy
/// cell next to well-formed ones, each with an honest (non-fabricated)
/// bounding box.
///
/// The failure mechanism, reproduced here without MPI at
/// all. The ORIGINAL (unfiltered, every-cell) eval loop in
/// `build_distributed_interp_rows` tested EVERY source cell against EVERY
/// evaluation point, in cell order, and aborted the WHOLE evaluation point
/// the instant ANY single candidate's `invert_reference_map` failed --
/// even one far outside that candidate's own bounding box, and even after
/// the genuine answer had ALREADY been found earlier in that very same
/// scan. The decoy cell below is collapsed along one axis (two pairs of
/// coincident corners): its Jacobian is identically rank-deficient
/// everywhere (the collapsed axis contributes nothing to the forward map,
/// for ANY natural coordinate), so `invert_reference_map` genuinely fails
/// with `E-INTERP-DEGENERATE` for any target -- a real, reproducible
/// failure, not a contrived one, and precisely the defect shape
/// `invert_reference_map`'s own singular-Jacobian check exists to catch.
class ManualQuadMesh final : public MeshView {
 public:
  ManualQuadMesh(std::vector<Point> verts, std::vector<std::array<std::size_t, 4>> cells)
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
    if (dim == 0) throw std::out_of_range("ManualQuadMesh: no vertex measure");
    return 1.0;  // unused by this test
  }
  std::size_t global_id(int, std::size_t i) const override { return i; }
  std::size_t simplex_count(int dim, std::size_t) const override {
    return dim == 2 ? fuse::cell_simplex_count(CellShape::Quad, 4) : 0;
  }
  void simplex_points(int dim, std::size_t i, std::size_t s, Point* out) const override {
    if (dim != 2) throw std::out_of_range("ManualQuadMesh: wrong dimension");
    int corners[4] = {0, 0, 0, 0};
    if (!fuse::simplex_corners(CellShape::Quad, 4, s, corners)) {
      throw std::out_of_range("ManualQuadMesh::simplex_points: simplex out of range");
    }
    for (int k = 0; k < 3; ++k) out[k] = verts_.at(cells_.at(i)[static_cast<std::size_t>(corners[k])]);
  }
  Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
    if (dim != 2) return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "ManualQuadMesh: wrong dimension"};
    CellVertices out;
    out.shape = CellShape::Quad;
    out.vertices.assign(cells_.at(i).begin(), cells_.at(i).end());
    return out;
  }

 private:
  std::vector<Point> verts_;
  std::vector<std::array<std::size_t, 4>> cells_;
};

/// A bare point cloud for the EVAL side: `build_distributed_interp_rows`
/// only ever reads `entity_count`/`is_owned`/`centroid`/`global_id` at
/// `eval_dim` off its second argument, so nothing else needs implementing.
class PointCloud final : public MeshView {
 public:
  explicit PointCloud(std::vector<Point> pts) : pts_(std::move(pts)) {}
  int dimension() const override { return 0; }
  std::size_t entity_count(int dim) const override { return dim == 0 ? pts_.size() : 0; }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int, std::size_t) const override { return true; }
  Point centroid(int, std::size_t i) const override { return pts_.at(i); }
  double measure(int, std::size_t) const override {
    throw std::out_of_range("PointCloud: no measure");
  }
  std::size_t global_id(int, std::size_t i) const override { return i; }
  std::size_t simplex_count(int, std::size_t) const override { return 0; }
  void simplex_points(int, std::size_t, std::size_t, Point*) const override {}
  Result<CellVertices> cell_vertices(int, std::size_t) const override {
    return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "PointCloud: no cells"};
  }

 private:
  std::vector<Point> pts_;
};

}  // namespace

TEST(NodalRemapDistributedNullComm, AffineFieldIsExactAndLoadIsConserved) {
  LocalLineMesh src(7, 0.0, 1.0);
  LocalLineMesh tgt(11, 0.0, 1.0);

  const auto rows_r = fuse::build_distributed_interp_rows(src, tgt, /*eval_dim=*/0,
                                                           fuse::search::kNullComm);
  ASSERT_TRUE(rows_r.ok()) << rows_r.error().message;
  const fuse::DistributedInterpRows& rows = rows_r.value();
  EXPECT_LT(rows.max_row_defect, 1e-10);
  ASSERT_EQ(rows.rows.size(), tgt.entity_count(0));  // every target vertex is "owned" (1 rank)

  const auto field = [](double x) { return 2.0 + 3.0 * x; };
  const std::size_t nv_src = src.entity_count(0);
  std::vector<double> src_vals(nv_src);
  for (std::size_t i = 0; i < nv_src; ++i) src_vals[i] = field(src.centroid(0, i).x);

  std::vector<double> Hv;
  ASSERT_TRUE(fuse::apply_distributed_H(rows, src, src_vals, 1, Hv, fuse::search::kNullComm).ok());
  ASSERT_EQ(Hv.size(), rows.rows.size());
  for (std::size_t r = 0; r < rows.rows.size(); ++r) {
    const double x = tgt.centroid(0, rows.rows[r].local_index).x;
    EXPECT_NEAR(Hv[r], field(x), 1e-10) << "target vertex local index " << rows.rows[r].local_index;
  }

  // H^T D: a unit load on every target vertex, scattered back to the
  // source. Total sent must equal total received -- no MPI reduction
  // needed with one process, unlike the MPI test's own version of this
  // check.
  std::vector<double> loads(rows.rows.size(), 1.0);
  std::vector<double> out_src;
  ASSERT_TRUE(fuse::apply_distributed_HT(rows, src, tgt, /*eval_dim=*/0, loads, 1, out_src,
                                         fuse::search::kNullComm)
                 .ok());
  ASSERT_EQ(out_src.size(), nv_src);
  const double total_sent = static_cast<double>(rows.rows.size());
  double total_received = 0.0;
  for (const double v : out_src) total_received += v;
  EXPECT_NEAR(total_sent, total_received, 1e-9 * total_sent)
      << "H^T must conserve the total load with a single process";
}

TEST(NodalRemapDistributedNullComm, TargetOutsideSourceDomainRefuses) {
  LocalLineMesh src(5, 0.0, 1.0);
  LocalLineMesh tgt(5, 0.0, 2.0);  // extends past the source
  const auto rows_r = fuse::build_distributed_interp_rows(src, tgt, /*eval_dim=*/0,
                                                           fuse::search::kNullComm);
  ASSERT_FALSE(rows_r.ok());
  EXPECT_EQ(rows_r.error().code, "E-INTERP-DOMAIN");
}

TEST(NodalRemapDistributedNullComm, RealCommunicatorWithoutMpiRefuses) {
#if !FUSE_HAVE_MPI
  LocalLineMesh src(3, 0.0, 1.0);
  LocalLineMesh tgt(3, 0.0, 1.0);
  // Any handle other than kNullComm names a "real" communicator; this
  // build has no MPI at all, so it must reject rather than silently treat
  // it as local (see nodal_remap_distributed.cpp's allgatherv_records).
  const auto rows_r = fuse::build_distributed_interp_rows(src, tgt, /*eval_dim=*/0,
                                                           fuse::search::kNullComm + 1);
  ASSERT_FALSE(rows_r.ok());
  EXPECT_EQ(rows_r.error().code, "E-RT-NODAL-DIST-NOMPI");
#else
  GTEST_SKIP() << "this build has MPI; the NOMPI refusal path is exercised only without it";
#endif
}

TEST(NodalRemapDistributedNullComm, DegenerateDecoyCellDoesNotAbortAnAlreadyFoundMatch) {
  // Two well-formed unit-square cells side by side ([0,1]x[0,1], gid 0;
  // [1,2]x[0,1], gid 1), plus a degenerate decoy (gid 2) collapsed onto
  // the DIAGONAL segment from (0,0) to (2,1) (corners v0, v5, v5, v0 --
  // two coincident pairs, chosen off-axis so the decoy's own bounding box
  // still varies in both x and y, exactly like the real cells' boxes --
  // a box index cannot mix box dimensionalities, so an axis-aligned
  // collapse, which flattens the box itself, is not a usable
  // reproduction here). Its Jacobian is rank-deficient along its own
  // collapsed direction for EVERY natural coordinate (the second pair
  // cancels identically in the forward map, the same algebra an
  // axis-aligned collapse uses), so `invert_reference_map` genuinely
  // fails (E-INTERP-DEGENERATE) for any target -- including one, like
  // this test's own, at the genuine corner cell 0, which this decoy's
  // own (uncollapsed) bounding box legitimately overlaps, so the fix's
  // broad phase cannot simply exclude it either.
  const std::vector<Point> verts = {
      {0.0, 0.0, 0.0},  // v0
      {1.0, 0.0, 0.0},  // v1
      {1.0, 1.0, 0.0},  // v2
      {0.0, 1.0, 0.0},  // v3
      {2.0, 0.0, 0.0},  // v4
      {2.0, 1.0, 0.0},  // v5
  };
  const std::vector<std::array<std::size_t, 4>> cells = {
      {0, 1, 2, 3},  // gid 0: unit square [0, 1] x [0, 1]
      {1, 4, 5, 2},  // gid 1: unit square [1, 2] x [0, 1]
      {0, 5, 5, 0},  // gid 2: the degenerate decoy, collapsed along the (0,0)-(2,1) diagonal
  };
  ManualQuadMesh src(verts, cells);
  PointCloud tgt({{0.5, 0.5, 0.0}});  // the centroid of cell 0

  const auto rows_r =
      fuse::build_distributed_interp_rows(src, tgt, /*eval_dim=*/0, fuse::search::kNullComm);
  ASSERT_TRUE(rows_r.ok()) << rows_r.error().message;
  const fuse::DistributedInterpRows& rows = rows_r.value();
  ASSERT_EQ(rows.rows.size(), 1u);
  EXPECT_LT(rows.max_row_defect, 1e-10);

  // The winning cell must be the genuine tight match (cell 0's own four
  // corner vertices, gids 0-3), never the degenerate decoy (which would
  // name only corner vertex gids 0 and 5) -- the fix must have silently
  // skipped it as just another non-winning candidate, exactly as it
  // would skip a real cell that merely does not contain the point.
  std::vector<std::size_t> cols;
  for (const auto& row : rows.rows) {
    for (const std::size_t gid : row.src_global_ids) cols.push_back(gid);
  }
  std::sort(cols.begin(), cols.end());
  EXPECT_EQ(cols, (std::vector<std::size_t>{0, 1, 2, 3}));
}

// ---------------------------------------------------------------------------
// Reproduces, WITHOUT any MPI symbol at all (both of
// this file's own primitives run serially, `comm_handle == kNullComm`), the
// 1-ULP disagreement measured on a real
// host run between `NodalRemapRunnerMpi.ConsistentTransposeIsBitIdenticalAcrossRankCounts`'s
// one-rank reference and its two/three-rank runs.
//
// THE MEASURED MECHANISM (not an earlier, incorrect guess).
// `Runner::resolve_remap` (runner.cpp) routes a nodal field transfer
// through `build_nodal_dist_remap`/`build_distributed_interp_rows` (this
// file's own `fuse::build_distributed_interp_rows`, which always inverts a
// cell's reference map via `general_interp_detail::invert_reference_map`'s
// damped Gauss-Newton, regardless of dimension) ONLY when the two
// components are NOT confined to a single shared rank (`both_on_one_rank`
// in runner.cpp, around "A single-rank placement on BOTH sides skips this
// whole path"). When they ARE (every one-rank run of a region placed on one
// rank, and `NodalRemapRunnerMpi.ConsistentTransposeIsBitIdenticalAcrossRankCounts`'s
// own `compute_reference()` helper, which calls the public dispatcher
// directly), the channel is instead built by the ORIGINAL, UNCHANGED
// `fuse::build_consistent_interp_weights` -> (`src_nodes.dimension() == 1`)
// -> `build_consistent_interp_weights_1d` (consistent_interp.hpp): a
// DIFFERENT, coordinate-sort implementation that computes the same
// bracketing hat-function weight by a DIFFERENT floating-point expression
// (`t = (x - lo.x) / (hi.x - lo.x)`) than `invert_reference_map`'s own
// Gauss-Newton normal-equation step for an Edge cell
// (`du = dot(jac, res) / dot(jac, jac)`, `jac = p1 - p0`, `res = target -
// p0`) -- algebraically the identical ratio, but NOT the identical sequence
// of rounding operations, so the two formulas can (and, at exactly this
// boundary vertex, for n_beam=7/n_fluid=11, DO) disagree by 1 ULP. This has
// NOTHING to do with `gather_global_source`'s null-comm-vs-Allgatherv
// assembly (ruled out below by `AssembledSourceCellGeometryIsBitIdenticalRegardlessOfAssemblyPath`)
// or with the lowest-global-id tie-break (there is only ONE candidate cell
// at this vertex, no tie at all) -- it is purely two different arithmetic
// expressions for the same ratio, invoked depending on which of the two
// code paths above a caller's placement happens to select.
TEST(NodalRemapDistributedNullComm, D67SerialAndDistributedEdgeWeightsAgreeAtTheDomainBoundary) {
  // n_beam=7 (source, nodal DOFs), n_fluid=11 (target, cell centroids) --
  // the exact resolution pair of the original host run.
  LocalLineMesh beam(7, 0.0, 1.0);
  LocalLineMesh fluid(11, 0.0, 1.0);

  // Path 1: the DISTRIBUTED primitive, serial (comm_handle == kNullComm) --
  // Gauss-Newton via invert_reference_map, exactly as a multi-rank run
  // takes (a host run confirmed ranks 2 and 3 agree
  // with this bit for bit).
  const auto dist_r = fuse::build_distributed_interp_rows(beam, fluid, /*eval_dim=*/1,
                                                           fuse::search::kNullComm);
  ASSERT_TRUE(dist_r.ok()) << dist_r.error().message;

  // Path 2: the SERIAL (both-components-on-one-rank) dispatcher --
  // build_consistent_interp_weights_1d's coordinate-sort, exactly what
  // compute_reference() and a genuinely one-rank Runner placement call.
  const auto serial_r = fuse::build_consistent_interp_weights(beam, fluid, /*tgt_dim=*/1);
  ASSERT_TRUE(serial_r.ok()) << serial_r.error().message;
  const fuse::RemapWeights& hw = serial_r.value();

  // Extract H[fluid_cell = row, beam_vertex_gid = col] from each path's own
  // storage shape and compare EVERY entry bit for bit -- not just the one
  // boundary vertex, so a regression anywhere in this bracket
  // is caught, not just at global id 0.
  auto bits = [](double v) {
    std::uint64_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
  };

  std::size_t mismatches = 0;
  std::size_t compared = 0;
  for (const fuse::DistributedInterpRow& row : dist_r.value().rows) {
    const std::size_t fluid_cell = row.local_index;
    for (std::size_t k = 0; k < row.src_global_ids.size(); ++k) {
      const std::size_t beam_gid = row.src_global_ids[k];
      const double dist_val = row.weights[k];
      // Find the matching (row, col) entry in the serial builder's COO.
      bool found = false;
      double serial_val = 0.0;
      for (std::size_t e = 0; e < hw.val.size(); ++e) {
        if (static_cast<std::size_t>(hw.row[e]) == fluid_cell &&
            static_cast<std::size_t>(hw.col[e]) == beam_gid) {
          found = true;
          serial_val = hw.val[e];
          break;
        }
      }
      ASSERT_TRUE(found) << "fluid cell " << fluid_cell << " beam vertex " << beam_gid
                         << ": the serial builder reports no weight at all for an entry "
                            "the distributed builder found";
      ++compared;
      if (bits(dist_val) != bits(serial_val)) {
        ++mismatches;
        ADD_FAILURE() << "fluid cell " << fluid_cell << ", beam vertex global id " << beam_gid
                      << ": distributed (Gauss-Newton) weight " << std::setprecision(20)
                      << dist_val << " (bits " << bits(dist_val)
                      << ") != serial (coordinate-sort) weight " << serial_val << " (bits "
                      << bits(serial_val) << ")";
      }
    }
  }
  ASSERT_GT(compared, 0u) << "the two builders produced no comparable entries at all";
  EXPECT_EQ(mismatches, 0u) << "the distributed and serial Edge weight "
                               "formulas disagree by rounding at " << mismatches << " of "
                            << compared << " compared entries";
}
