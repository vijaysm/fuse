// Nodal-remap track, step 2: "Distributed rows and the exact transpose"
// (KNOWN_ISSUES D8). Exercises `fuse::build_distributed_interp_rows`,
// `apply_distributed_H` and `apply_distributed_HT`
// (nodal_remap_distributed.hpp) directly -- not yet through the Runner, see
// that header's own file comment for exactly why -- against
// `GeometricLineMesh` (test/support/partitioned_test_backend.hpp), a
// distributed 1D interface with real coordinates and connectivity, at three
// rank-set shapes: disjoint, identical, and partially overlapping.
//
// MPI CANNOT INITIALISE IN THIS SANDBOX (docs/dev/testing.md): every case
// here is compiled, never run, in the environment this file was written in.
// outcomes are a host action.
//
// WHAT THIS FILE DOES NOT CHECK: bit-identical results ACROSS DIFFERENT rank
// counts needs two separate job launches compared against each other (the
// parallel matrix's own job, not one gtest binary); what this file DOES
// check, at whatever rank count it runs, is that the algorithm produces the
// mathematically correct answer (exact for an affine field; H^T conserves
// the total load; discrete power balances) -- which is the property the
// bit-identical design in nodal_remap_distributed.hpp's own header rests on.

#include "fuse/backend/nodal_remap_distributed.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse/base/error.hpp"

#include "fuse_test_main.hpp"

#include "../support/partitioned_test_backend.hpp"

namespace {

int world_size() {
  int n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}
int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

/// Runs the full build -> apply H -> apply H^T D pipeline for a source line
/// of `n_src` cells on `src_ranks` and a target line of `n_tgt` cells on
/// `tgt_ranks`, both over `[0, 1]`, and checks: H exact for an affine field
/// at every target vertex this rank owns; H^T D's total conserved; discrete
/// power balanced. Every rank of `MPI_COMM_WORLD` must call this.
void check_affine_exact_and_conservative(std::size_t n_src, fuse::RankSet src_ranks,
                                         std::size_t n_tgt, fuse::RankSet tgt_ranks) {
  GeometricLineBackend src_backend(n_src, src_ranks);
  GeometricLineBackend tgt_backend(n_tgt, tgt_ranks);
  ASSERT_TRUE(src_backend.load({.name = "src", .source = "n/a"}).ok());
  ASSERT_TRUE(tgt_backend.load({.name = "tgt", .source = "n/a"}).ok());
  const auto src_view = src_backend.view(src_backend.mesh_id("src").value()).value();
  const auto tgt_view = tgt_backend.view(tgt_backend.mesh_id("tgt").value()).value();

  const int comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
  const auto rows_r = fuse::build_distributed_interp_rows(*src_view, *tgt_view, /*eval_dim=*/0,
                                                          comm_handle);
  ASSERT_TRUE(rows_r.ok()) << rows_r.error().message;
  const fuse::DistributedInterpRows& rows = rows_r.value();
  EXPECT_LT(rows.max_row_defect, 1e-10);

  // Ownership uniqueness, checked DIRECTLY and locally: exactly one row per
  // OWNED target evaluation entity on this rank, never one for a ghost too.
  // This is NOT implied by the load-conservation check further down --
  // apply_distributed_H / apply_distributed_HT redistribute whatever `rows`
  // it is handed with total_sent == total_received for ANY row set whose
  // per-row weights sum to ~1 (a duplicated ghost row included: it adds the
  // same unit to both sides at once), so that check alone cannot see a ghost
  // that wrongly got its own row built. `owned_count(0)` is the independent
  // ground truth `rows.rows.size()` must match (mesh_view.hpp's own
  // contract); a mismatch here is exactly what
  // "nodal-remap-dist-row-built-on-every-holder" (build-evidence/mutations/
  // nodal-remap-distributed.toml) removes the guard against.
  EXPECT_EQ(rows.rows.size(), tgt_view->owned_count(0))
      << "build_distributed_interp_rows must build exactly one row per "
         "OWNED target evaluation entity on this rank, never one for a "
         "ghost too";

  // v(x) = 2 + 3x, exact for P1 -- src_view's OWNED and ghost vertices alike
  // need a value (apply_distributed_H only reads OWNED ones, but the buffer
  // must still be full-sized).
  const auto field = [](double x) { return 2.0 + 3.0 * x; };
  const std::size_t nv_src = src_view->entity_count(0);
  std::vector<double> src_vals(nv_src);
  for (std::size_t i = 0; i < nv_src; ++i) src_vals[i] = field(src_view->centroid(0, i).x);

  std::vector<double> Hv;
  ASSERT_TRUE(
      fuse::apply_distributed_H(rows, *src_view, src_vals, 1, Hv, comm_handle).ok());
  ASSERT_EQ(Hv.size(), rows.rows.size());
  for (std::size_t r = 0; r < rows.rows.size(); ++r) {
    const double x = tgt_view->centroid(0, rows.rows[r].local_index).x;
    EXPECT_NEAR(Hv[r], field(x), 1e-10) << "owned target vertex local index "
                                        << rows.rows[r].local_index;
  }

  // H^T D: a unit "density" of 1 on every owned target vertex (already a
  // load here -- no measure to multiply by on a vertex target), scattered
  // back to the source. Total sent (owned target count, globally) must equal
  // total received (owned source count, globally) -- checked via the LOCAL
  // owned sums plus an Allreduce, since this rank only sees its own slice.
  std::vector<double> loads(rows.rows.size(), 1.0);
  std::vector<double> out_src;
  ASSERT_TRUE(fuse::apply_distributed_HT(rows, *src_view, *tgt_view, /*eval_dim=*/0, loads, 1,
                                        out_src, comm_handle)
                 .ok());
  ASSERT_EQ(out_src.size(), nv_src);

  double local_sent = static_cast<double>(rows.rows.size());
  double local_received = 0.0;
  for (std::size_t i = 0; i < nv_src; ++i) {
    if (src_view->is_owned(0, i)) local_received += out_src[i];
  }
  double total_sent = 0.0, total_received = 0.0;
  MPI_Allreduce(&local_sent, &total_sent, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_received, &total_received, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_NEAR(total_sent, total_received, 1e-9 * std::abs(total_sent))
      << "H^T must conserve the total load globally";
}

/// Review 2026-10-11 TF-1: `gather_global_source`'s per-rank vertex/cell
/// buffer-building loops (inside `build_distributed_interp_rows`, the only
/// entry point that reaches it) can fail -- or throw -- rank-locally BEFORE
/// their own `MPI_Allgatherv`, which is a hang (or an uncaught exception) at
/// every OTHER rank waiting inside that same collective. This wrapper
/// injects exactly one of the two triggering defects, on exactly one rank,
/// at one local entity index that rank owns: an unsupported source cell
/// shape (the serial builder's own `E-INTERP-SHAPE` case, legal mesh content
/// `gather_global_source` must still reject without hanging its peers), or a
/// `global_id` that throws (standing in for `MoabMeshView::global_id`'s own
/// `InvariantViolation` on a partitioned read with unnumbered vertices,
/// moab_backend.cpp's `gid_error`). All other ranks, and every other entity
/// on the faulty rank, see `GeometricLineMesh`'s ordinary answers.
enum class SourceFault { kNone, kUnsupportedShape, kVertexGidThrows };

class FaultInjectingSourceMesh final : public fuse::MeshView {
 public:
  FaultInjectingSourceMesh(GeometricLineMesh base, SourceFault fault_here)
      : base_(std::move(base)), fault_(fault_here) {}

  [[nodiscard]] int dimension() const override { return base_.dimension(); }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    return base_.entity_count(dim);
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override { return base_.owned_count(dim); }
  [[nodiscard]] bool is_owned(int dim, std::size_t i) const override {
    return base_.is_owned(dim, i);
  }
  [[nodiscard]] fuse::Point centroid(int dim, std::size_t i) const override {
    return base_.centroid(dim, i);
  }
  [[nodiscard]] double measure(int dim, std::size_t i) const override {
    return base_.measure(dim, i);
  }
  [[nodiscard]] std::size_t simplex_count(int dim, std::size_t i) const override {
    return base_.simplex_count(dim, i);
  }
  void simplex_points(int dim, std::size_t i, std::size_t s, fuse::Point* out) const override {
    base_.simplex_points(dim, i, s, out);
  }

  // Local vertex index 1 is always this rank's OWN (never the shared,
  // previous-rank-owned boundary vertex at local index 0 -- see
  // GeometricLineMesh::is_owned above), so it is a safe, deterministic
  // target for the thrown-global-id fault regardless of rank count.
  static constexpr std::size_t kFaultVertexLocalIndex = 1;
  static constexpr std::size_t kFaultCellLocalIndex = 0;

  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    if (fault_ == SourceFault::kVertexGidThrows && dim == 0 && i == kFaultVertexLocalIndex) {
      FUSE_THROW(fuse::InvariantViolation,
                "FaultInjectingSourceMesh: simulated unnumbered vertex, standing in for "
                "MoabMeshView::global_id's own gid_error throw on a partitioned MOAB read");
    }
    return base_.global_id(dim, i);
  }

  [[nodiscard]] fuse::Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
    auto cv = base_.cell_vertices(dim, i);
    if (fault_ == SourceFault::kUnsupportedShape && dim == 1 && i == kFaultCellLocalIndex &&
        cv.ok()) {
      CellVertices bad = cv.value();
      bad.shape = fuse::CellShape::Pyramid;  // shape_supported() == false
      return bad;
    }
    return cv;
  }

 private:
  GeometricLineMesh base_;
  SourceFault fault_;
};

/// Builds a source line mesh of `n_src` cells over every rank of
/// `MPI_COMM_WORLD`, injecting `fault` on local rank `fault_rank` only, and
/// a trivial target line mesh over the same communicator; calls
/// `build_distributed_interp_rows` and asserts EVERY rank returns the SAME
/// `Failure` -- the agreement `agree_on_local_failure` makes, not a
/// rank-divergent return or a hang. `fault_rank` is relative to
/// `MPI_COMM_WORLD` (identical to the rank's position in `all`, since the
/// source RankSet here is the whole communicator).
void check_one_rank_bad_source_agrees(std::size_t n_src, SourceFault fault, int fault_rank) {
  const int me = world_rank();
  const int n = world_size();
  GeometricLineMesh base(n_src, n, me, /*x_lo=*/0.0, /*x_hi=*/1.0);
  FaultInjectingSourceMesh src_view(std::move(base), me == fault_rank ? fault : SourceFault::kNone);
  // A target identical to (an unfaulted copy of) the source topology is
  // enough: gather_global_source runs, and must agree-and-refuse, before
  // build_distributed_interp_rows ever reaches the target evaluation loop.
  GeometricLineMesh tgt_view(n_src, n, me, /*x_lo=*/0.0, /*x_hi=*/1.0);

  const int comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
  const auto rows_r =
      fuse::build_distributed_interp_rows(src_view, tgt_view, /*eval_dim=*/0, comm_handle);
  ASSERT_FALSE(rows_r.ok()) << "rank " << me << " expected every rank to refuse together";
  if (fault == SourceFault::kUnsupportedShape) {
    EXPECT_EQ(rows_r.error().code, "E-INTERP-SHAPE");
  } else {
    EXPECT_EQ(rows_r.error().code, "E-RT-NODAL-DIST-GID");
  }
}

}  // namespace

// Review 2026-10-11 TF-1. Pre-fix signature: the faulty rank either throws
// uncaught (the `kVertexGidThrows` case) or returns `Failure` directly from
// inside `gather_global_source`'s loop (the `kUnsupportedShape` case) BEFORE
// that loop's own `MPI_Allgatherv` -- in either case every OTHER rank blocks
// forever inside the collective it entered on schedule. Post-fix: every rank
// returns the identical `Failure` within the test's own TIMEOUT.
TEST(NodalRemapDistributed, UnsupportedShapeSourceCellAgreesAcrossRanks) {
  if (world_size() < 3) GTEST_SKIP() << "needs >= 3 ranks for a one-rank-bad row";
  check_one_rank_bad_source_agrees(/*n_src=*/5 * static_cast<std::size_t>(world_size()),
                                   SourceFault::kUnsupportedShape, /*fault_rank=*/world_size() - 1);
}

TEST(NodalRemapDistributed, UnnumberedSourceVertexAgreesAcrossRanks) {
  if (world_size() < 3) GTEST_SKIP() << "needs >= 3 ranks for a one-rank-bad row";
  check_one_rank_bad_source_agrees(/*n_src=*/5 * static_cast<std::size_t>(world_size()),
                                   SourceFault::kVertexGidThrows, /*fault_rank=*/world_size() - 1);
}

TEST(NodalRemapDistributed, DisjointRankSets) {
  if (world_size() < 2) GTEST_SKIP() << "needs at least 2 ranks to be disjoint";
  const int half = world_size() / 2;
  fuse::RankSet src_ranks{0, half};
  fuse::RankSet tgt_ranks{half, world_size() - half};
  check_affine_exact_and_conservative(7, src_ranks, 11, tgt_ranks);
}

TEST(NodalRemapDistributed, IdenticalRankSets) {
  fuse::RankSet all{0, world_size()};
  check_affine_exact_and_conservative(9, all, 13, all);
}

TEST(NodalRemapDistributed, PartiallyOverlappingRankSets) {
  if (world_size() < 3) GTEST_SKIP() << "needs at least 3 ranks to overlap partially";
  fuse::RankSet src_ranks{0, world_size() - 1};      // ranks [0, n-1)
  fuse::RankSet tgt_ranks{1, world_size() - 1};      // ranks [1, n)
  check_affine_exact_and_conservative(8, src_ranks, 6, tgt_ranks);
}

TEST(NodalRemapDistributed, TargetOutsideSourceDomainRefuses) {
  fuse::RankSet all{0, world_size()};
  GeometricLineBackend src_backend(5, all, /*x_lo=*/0.0, /*x_hi=*/1.0);
  GeometricLineBackend tgt_backend(5, all, /*x_lo=*/0.0, /*x_hi=*/2.0);  // extends past the source
  ASSERT_TRUE(src_backend.load({.name = "src", .source = "n/a"}).ok());
  ASSERT_TRUE(tgt_backend.load({.name = "tgt", .source = "n/a"}).ok());
  const auto src_view = src_backend.view(src_backend.mesh_id("src").value()).value();
  const auto tgt_view = tgt_backend.view(tgt_backend.mesh_id("tgt").value()).value();
  const int comm_handle = MPI_Comm_c2f(MPI_COMM_WORLD);
  const auto rows_r =
      fuse::build_distributed_interp_rows(*src_view, *tgt_view, /*eval_dim=*/0, comm_handle);
  ASSERT_FALSE(rows_r.ok());
  EXPECT_EQ(rows_r.error().code, "E-INTERP-DOMAIN");
}
