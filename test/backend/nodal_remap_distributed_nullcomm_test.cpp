// Nodal-remap track, round 2, item 3 ("prove the single-rank/null-
// communicator path via a tier-0 Runner test"): this file is the
// PRIMITIVE-level half of that proof -- it exercises
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
// trivially "owned" by the one and only participant -- built by hand below
// rather than reusing test/support/partitioned_test_backend.hpp's
// GeometricLineMesh, which unconditionally `#include <mpi.h>` and so cannot
// be used from a target that has none.

#include "fuse/backend/nodal_remap_distributed.hpp"

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/cell_simplices.hpp"
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
  // needed with one participant, unlike the MPI test's own version of this
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
      << "H^T must conserve the total load with a single participant";
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
  // build has no MPI at all, so it must refuse rather than silently treat
  // it as local (see nodal_remap_distributed.cpp's allgatherv_records).
  const auto rows_r = fuse::build_distributed_interp_rows(src, tgt, /*eval_dim=*/0,
                                                           fuse::search::kNullComm + 1);
  ASSERT_FALSE(rows_r.ok());
  EXPECT_EQ(rows_r.error().code, "E-RT-NODAL-DIST-NOMPI");
#else
  GTEST_SKIP() << "this build has MPI; the NOMPI refusal path is exercised only without it";
#endif
}
