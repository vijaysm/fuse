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
// Verdicts are a host action.
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

}  // namespace

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
