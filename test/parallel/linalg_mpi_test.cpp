// "Containers" track, tier 1: fuse::linalg::make_mpi_reducer/
// make_mpi_exchanger (src/linalg/comm_mpi.cpp) and the DistVector/
// DistSparseMatrix operations built on them, over a REAL MPI_COMM_WORLD.
//
// MPI CANNOT START IN THIS SANDBOX (docs/dev/testing.md): this file is
// compiled -- registered with fuse_add_mpi_test, RANKS 2 -- but never run
// here; outcomes are a host action.
//
// THE REFERENCE. Every quantity below is checked against a closed-form
// value every rank can compute WITHOUT any communication: `g(i) =
// sin(1.3*i + 0.2)` for a vector entry and the stencil applied to `g`
// directly for a matrix row. Because `g` is a pure function of the GLOBAL
// index, every rank reconstructs the IDENTICAL whole-domain answer a
// single rank computing the whole problem serially would get -- the "rank-0
// serial computation" the task names, done redundantly on every rank
// instead of gathered from one, which needs no extra collective of its
// own and cannot disagree with what rank 0 would have produced. This is
// the same comparison test/linalg/dist_vector_test.cpp's and
// dist_sparse_matrix_test.cpp's in-process two-partition fakes make
// against an independently-computed ground truth; here the peer is a real
// MPI rank instead of a second DistVector in the same process.

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

namespace {

using fuse::linalg::DistSparseMatrix;
using fuse::linalg::DistVector;
using fuse::linalg::GhostPattern;
using fuse::linalg::RowOwnership;
using fuse::linalg::Triplet;

/// The closed-form reference value at GLOBAL index `i` -- smooth, bounded,
/// and a pure function of `i`, so every rank's own call reproduces the
/// identical double bit pattern with no communication at all.
double g(std::int64_t i) { return std::sin(1.3 * static_cast<double>(i) + 0.2); }

/// This rank's half of `[0, global_n)`: rank 0 owns the first half (rounded
/// down), rank 1 the rest. `FUSE_REQUIRE_RANKS(2)` in every test below
/// makes "this rank" unambiguous.
std::pair<std::int64_t, std::int64_t> my_row_range(std::int64_t global_n) {
  const std::int64_t split = global_n / 2;
  return fuse::testing::rank() == 0 ? std::make_pair(std::int64_t{0}, split)
                                    : std::make_pair(split, global_n);
}

TEST(LinalgMpi, GhostUpdateMatchesTheClosedFormReferenceAcrossRanks) {
  FUSE_REQUIRE_RANKS(2);
  constexpr std::int64_t global_n = 10;
  const auto [row_lo, row_hi] = my_row_range(global_n);
  const auto owned_n = static_cast<std::size_t>(row_hi - row_lo);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  // Every ghost is the WHOLE other rank's owned range -- an interleaving
  // pattern deliberately unlike "ghost g is remote index g", so a wrong
  // unpack index (build-evidence/mutations/linalg.toml's
  // "ghost-unpack-index" row) cannot survive by accident.
  const int other = 1 - fuse::testing::rank();
  const auto [other_lo, other_hi] = fuse::testing::rank() == 0
                                        ? std::make_pair(global_n / 2, global_n)
                                        : std::make_pair(std::int64_t{0}, global_n / 2);
  GhostPattern pattern;
  for (std::int64_t c = other_hi - 1; c >= other_lo; --c) {  // reversed, not identity order
    pattern.owner_rank.push_back(other);
    pattern.remote_local_index.push_back(c - other_lo);
  }

  auto exchanger = fuse::linalg::make_mpi_exchanger(comm_f, pattern, owned_n);
  auto reducer = fuse::linalg::make_mpi_reducer(comm_f);
  DistVector v(owned_n, exchanger, reducer);
  for (std::size_t i = 0; i < owned_n; ++i) {
    v.data()[i] = g(row_lo + static_cast<std::int64_t>(i));
  }

  ASSERT_TRUE(v.update_ghosts().ok());
  for (std::size_t gi = 0; gi < v.ghost_size(); ++gi) {
    const std::int64_t global_idx = other_lo + pattern.remote_local_index[gi];
    EXPECT_DOUBLE_EQ(v.data()[v.owned_size() + gi], g(global_idx)) << "ghost " << gi;
  }
}

TEST(LinalgMpi, DotNorm2AndNormInfMatchTheWholeDomainClosedFormReference) {
  FUSE_REQUIRE_RANKS(2);
  constexpr std::int64_t global_n = 10;
  const auto [row_lo, row_hi] = my_row_range(global_n);
  const auto owned_n = static_cast<std::size_t>(row_hi - row_lo);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  // Neither dot/norm2/norm_inf needs a ghost at all (DistVector::dot/
  // norm_inf reduce only the OWNED block, dist_vector.hpp), so an empty
  // pattern is the correct one here, not a placeholder.
  auto exchanger = fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n);
  auto reducer = fuse::linalg::make_mpi_reducer(comm_f);
  DistVector v(owned_n, exchanger, reducer);
  for (std::size_t i = 0; i < owned_n; ++i) {
    v.data()[i] = g(row_lo + static_cast<std::int64_t>(i));
  }

  double expect_dot = 0.0, expect_max = 0.0, sum_abs_terms = 0.0;
  for (std::int64_t i = 0; i < global_n; ++i) {
    expect_dot += g(i) * g(i);
    sum_abs_terms += g(i) * g(i);  // every term is non-negative, so this equals expect_dot
    expect_max = std::max(expect_max, std::fabs(g(i)));
  }

  // Review 2026-10-11 NL-10: v.dot(v) sums a 5-term rank-0 partial and a
  // 5-term rank-1 partial via MPI_Allreduce; `expect_dot` above sums all
  // 10 terms in a single ascending-index loop. Floating-point addition is
  // not associative, so the two groupings can disagree in the last bit
  // even though both are "the same sum" mathematically -- a bit-tight
  // `EXPECT_DOUBLE_EQ` would pass today by luck of magnitude, not by
  // construction (the same fragility class KNOWN_ISSUES.md's D30
  // already records for contraction, not reassociation). Bounded by a
  // measured, term-count-scaled tolerance instead, as
  // dist_sparse_matrix_test.cpp's own `check_two_partition_apply` already
  // does for the analogous apply() split.
  const double tol = static_cast<double>(global_n) * std::numeric_limits<double>::epsilon() *
                     sum_abs_terms;
  EXPECT_NEAR(v.dot(v), expect_dot, tol);
  EXPECT_NEAR(v.norm2(), std::sqrt(expect_dot), tol);
  EXPECT_DOUBLE_EQ(v.norm_inf(), expect_max);  // max is exact under any grouping, no reassociation
}

TEST(LinalgMpi, ApplyMatchesTheWholeDomainClosedFormStencilReference) {
  FUSE_REQUIRE_RANKS(2);
  // A periodic 1D 3-point stencil (center 2, each neighbour -1) over
  // global_n rows -- the same periodic-wrap shape
  // dist_sparse_matrix_test.cpp's 2D 5-point Laplacian exercises, small
  // enough that building it independently on both ranks costs nothing.
  constexpr std::int64_t global_n = 16;
  const auto [row_lo, row_hi] = my_row_range(global_n);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  std::vector<Triplet> t;
  for (std::int64_t r = row_lo; r < row_hi; ++r) {
    t.push_back({r, r, 2.0});
    t.push_back({r, (r - 1 + global_n) % global_n, -1.0});
    t.push_back({r, (r + 1) % global_n, -1.0});
  }
  const RowOwnership owners{{0, global_n / 2}};
  auto m = DistSparseMatrix::from_triplets(row_lo, row_hi, global_n, t,
                                           fuse::linalg::make_mpi_reducer(comm_f), owners);
  ASSERT_TRUE(m.ok()) << m.error().message;

  const auto owned_n = static_cast<std::size_t>(row_hi - row_lo);
  auto x_exchanger = fuse::linalg::make_mpi_exchanger(comm_f, m.value().ghost_pattern(), owned_n);
  DistVector x(owned_n, x_exchanger, fuse::linalg::make_mpi_reducer(comm_f));
  for (std::size_t i = 0; i < owned_n; ++i) x.data()[i] = g(row_lo + static_cast<std::int64_t>(i));

  DistVector y(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n),
              fuse::linalg::make_mpi_reducer(comm_f));
  ASSERT_TRUE(m.value().apply(x, y).ok());

  for (std::int64_t r = row_lo; r < row_hi; ++r) {
    const double expect =
        2.0 * g(r) - g((r - 1 + global_n) % global_n) - g((r + 1) % global_n);
    EXPECT_DOUBLE_EQ(y.data()[static_cast<std::size_t>(r - row_lo)], expect) << "row " << r;
  }

  DistVector d(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n),
              fuse::linalg::make_mpi_reducer(comm_f));
  m.value().diagonal(d);
  for (std::size_t i = 0; i < owned_n; ++i) EXPECT_DOUBLE_EQ(d.data()[i], 2.0);
}

}  // namespace
