// DistSparseMatrix at tier 0. The periodic 5-point Laplacian and the
// random nonsymmetric matrix both run under an in-process "fake"
// two-partition split -- the same FakePeerExchanger device
// dist_vector_test.cpp uses, duplicated here rather than shared through a
// new header. MPI cannot start in this
// sandbox (docs/dev/testing.md): src/linalg/comm_mpi.cpp's real
// MpiExchanger/MpiReducer are built (test/parallel/linalg_mpi_test.cpp)
// but never run here.

#include "fuse/linalg/dist_sparse_matrix.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/base/error.hpp"
#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_vector.hpp"

namespace {

using fuse::Result;
using fuse::linalg::DistSparseMatrix;
using fuse::linalg::DistVector;
using fuse::linalg::GhostExchanger;
using fuse::linalg::GhostPattern;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::linalg::RowOwnership;
using fuse::linalg::Triplet;

/// See dist_vector_test.cpp's own copy for the full rationale: a
/// GhostExchanger that reads a peer DistVector held in the same process,
/// bound in a second phase because the peer does not exist yet when this
/// exchanger is handed to its own DistVector's constructor.
class FakePeerExchanger : public GhostExchanger {
 public:
  explicit FakePeerExchanger(GhostPattern pattern) : pattern_(std::move(pattern)) {}
  void bind_peers(std::array<const DistVector*, 2> peers) { peers_ = peers; }
  [[nodiscard]] std::size_t ghost_size() const override { return pattern_.owner_rank.size(); }
  [[nodiscard]] Result<void> update(const double* owned, std::size_t owned_n, double* ghosts,
                                    std::size_t ghost_n) const override {
    (void)owned;
    (void)owned_n;
    for (std::size_t g = 0; g < ghost_n; ++g) {
      const auto owner = static_cast<std::size_t>(pattern_.owner_rank[g]);
      ghosts[g] = peers_[owner]->data()[pattern_.remote_local_index[g]];
    }
    return Result<void>::success();
  }

 private:
  GhostPattern pattern_;
  std::array<const DistVector*, 2> peers_{nullptr, nullptr};
};

// NOTE ON A HAZARD DELIBERATELY AVOIDED: a tempting helper would build the
// two DistVectors, bind_peers() to each other's addresses, and RETURN a
// struct holding both -- but a struct return is free to be constructed via
// a move into the caller's storage rather than named-return-value
// optimisation (NRVO is permitted, never guaranteed), which would leave
// `peers_` pointing at the no-longer-existing local object's address, not
// the moved-to one. `check_two_partition_apply` below builds both
// DistVectors in its OWN frame and binds peers only once their addresses
// are final, never passing them through a return.

// ---------------------------------------------------------------------
// Requirement 3: the column split and ghost pattern are DERIVED, never
// supplied; a foreign row is refused; duplicates are summed.
// ---------------------------------------------------------------------

TEST(DistSparseMatrixFromTriplets, DerivesGhostPatternFromOffProcessColumnsAscending) {
  // global_n = 6, two participants: rank 0 owns rows/cols [0,3), rank 1
  // owns [3,6). Rank 0's own triplets name columns 5, 3 and 4 (in THAT
  // order in the input) -- the ghost pattern must come out sorted
  // ascending by global column regardless of input order.
  const RowOwnership owners{{0, 3}};
  const std::vector<Triplet> t{
      {0, 0, 1.0}, {0, 5, 2.0}, {1, 3, 3.0}, {2, 4, 4.0},
  };
  auto m = DistSparseMatrix::from_triplets(0, 3, 6, t, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok()) << m.error().message;
  const GhostPattern& g = m.value().ghost_pattern();
  ASSERT_EQ(g.owner_rank.size(), 3u);
  EXPECT_EQ(g.owner_rank[0], 1);
  EXPECT_EQ(g.owner_rank[1], 1);
  EXPECT_EQ(g.owner_rank[2], 1);
  // Ascending global column 3, 4, 5 -> local (on rank 1) 0, 1, 2.
  EXPECT_EQ(g.remote_local_index[0], 0);
  EXPECT_EQ(g.remote_local_index[1], 1);
  EXPECT_EQ(g.remote_local_index[2], 2);
}

TEST(DistSparseMatrixFromTriplets, RefusesATripletWhoseRowIsNotOwnedHere) {
  const RowOwnership owners{{0, 5}};
  const std::vector<Triplet> t{{0, 0, 1.0}, {7, 2, 1.0}};  // row 7 is rank 1's, not rank 0's
  auto m = DistSparseMatrix::from_triplets(0, 5, 10, t, make_serial_reducer(), owners);
  ASSERT_FALSE(m.ok());
  EXPECT_EQ(m.error().code, "E-LINALG-FOREIGN-ROW");
}

TEST(DistSparseMatrixFromTriplets, RefusesATripletWhoseColumnIsOutOfRange) {
  const RowOwnership owners{{0, 5}};
  const std::vector<Triplet> t{{0, 0, 1.0}, {1, 99, 1.0}};  // global_n is 10
  auto m = DistSparseMatrix::from_triplets(0, 5, 10, t, make_serial_reducer(), owners);
  ASSERT_FALSE(m.ok());
  EXPECT_EQ(m.error().code, "E-LINALG-COL-RANGE");
}

TEST(DistSparseMatrixFromTriplets, DuplicateEntriesAreSummed) {
  const RowOwnership owners{{0}};
  const std::vector<Triplet> t{{0, 0, 3.0}, {0, 0, 4.0}, {1, 1, 1.5}};
  auto m = DistSparseMatrix::from_triplets(0, 2, 2, t, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok()) << m.error().message;
  DistVector d(2, make_serial_exchanger(), make_serial_reducer());
  m.value().diagonal(d);
  EXPECT_DOUBLE_EQ(d.data()[0], 7.0);
  EXPECT_DOUBLE_EQ(d.data()[1], 1.5);
}

// ---------------------------------------------------------------------
// Requirement 4: A*x on two in-process partitions, against a serial
// ground truth computed directly from the SAME global triplet list (never
// through DistSparseMatrix) -- (a) the periodic 5-point Laplacian on a
// 16x16 grid, (b) a random nonsymmetric ~5 nnz/row matrix. diagonal()
// needs no communication, checked directly against the assembled triplets.
// ---------------------------------------------------------------------

/// Row-major iy*nx+ix, periodic in both directions: center 4, each of the
/// four nearest neighbours -1 -- the standard 5-point discrete Laplacian,
/// including the wrap rows (ix==0/nx-1, iy==0/ny-1).
std::vector<Triplet> periodic_laplacian_5pt(int nx, int ny) {
  std::vector<Triplet> t;
  t.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * 5);
  auto idx = [nx](int ix, int iy) { return static_cast<std::int64_t>(iy) * nx + ix; };
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      const std::int64_t r = idx(ix, iy);
      t.push_back({r, r, 4.0});
      t.push_back({r, idx((ix - 1 + nx) % nx, iy), -1.0});
      t.push_back({r, idx((ix + 1) % nx, iy), -1.0});
      t.push_back({r, idx(ix, (iy - 1 + ny) % ny), -1.0});
      t.push_back({r, idx(ix, (iy + 1) % ny), -1.0});
    }
  }
  return t;
}

/// y[r] = sum over (r, c, v) in t of v * x[c] -- the ground truth every
/// apply() result below is checked against, assembled with no reference
/// to DistSparseMatrix at all.
std::vector<double> serial_spmv(const std::vector<Triplet>& t, std::int64_t global_n,
                                const std::vector<double>& x) {
  std::vector<double> y(static_cast<std::size_t>(global_n), 0.0);
  for (const Triplet& e : t) y[static_cast<std::size_t>(e.row)] += e.value * x[static_cast<std::size_t>(e.col)];
  return y;
}

std::vector<Triplet> rows_in(const std::vector<Triplet>& t, std::int64_t lo, std::int64_t hi) {
  std::vector<Triplet> out;
  for (const Triplet& e : t) {
    if (e.row >= lo && e.row < hi) out.push_back(e);
  }
  return out;
}

/// Builds the two-partition DistSparseMatrix pair for `global_triplets`
/// split at `row_lo[1]`, applies A*x with a fake two-partition exchanger,
/// and checks both partitions' output against `serial_spmv` -- the single
/// routine every case below shares.
void check_two_partition_apply(const std::vector<Triplet>& global_triplets,
                               std::int64_t global_n, std::int64_t split,
                               const std::vector<double>& x_global, double tol) {
  const RowOwnership owners{{0, split}};
  auto m0 = DistSparseMatrix::from_triplets(0, split, global_n, rows_in(global_triplets, 0, split),
                                            make_serial_reducer(), owners);
  auto m1 = DistSparseMatrix::from_triplets(split, global_n, global_n,
                                            rows_in(global_triplets, split, global_n),
                                            make_serial_reducer(), owners);
  ASSERT_TRUE(m0.ok()) << m0.error().message;
  ASSERT_TRUE(m1.ok()) << m1.error().message;

  const auto n0 = static_cast<std::size_t>(split);
  const auto n1 = static_cast<std::size_t>(global_n - split);

  auto exch0 = std::make_shared<FakePeerExchanger>(m0.value().ghost_pattern());
  auto exch1 = std::make_shared<FakePeerExchanger>(m1.value().ghost_pattern());
  DistVector x0(n0, exch0, make_serial_reducer());
  DistVector x1(n1, exch1, make_serial_reducer());
  exch0->bind_peers({nullptr, &x1});
  exch1->bind_peers({&x0, nullptr});
  std::copy(x_global.begin(), x_global.begin() + static_cast<std::ptrdiff_t>(n0), x0.data());
  std::copy(x_global.begin() + static_cast<std::ptrdiff_t>(n0), x_global.end(), x1.data());

  DistVector y0(n0, make_serial_exchanger(), make_serial_reducer());
  DistVector y1(n1, make_serial_exchanger(), make_serial_reducer());

  ASSERT_TRUE(m0.value().apply(x0, y0).ok());
  ASSERT_TRUE(m1.value().apply(x1, y1).ok());

  const std::vector<double> y_truth = serial_spmv(global_triplets, global_n, x_global);
  for (std::size_t i = 0; i < n0; ++i) {
    EXPECT_NEAR(y0.data()[i], y_truth[i], tol) << "rank 0 row " << i;
  }
  for (std::size_t i = 0; i < n1; ++i) {
    EXPECT_NEAR(y1.data()[i], y_truth[n0 + i], tol) << "rank 1 row " << i;
  }
}

TEST(DistSparseMatrixApply, PeriodicLaplacian16x16SplitInTwoRowSlabs) {
  constexpr int nx = 16, ny = 16;
  constexpr std::int64_t global_n = nx * ny;  // 256
  const std::vector<Triplet> t = periodic_laplacian_5pt(nx, ny);

  std::mt19937 rng(7u);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> x(static_cast<std::size_t>(global_n));
  for (double& v : x) v = u(rng);

  // Measured: with coefficients {4, -1, -1, -1, -1} and |x_i| <= 1, five
  // terms summed in two different groupings (diag-then-offdiag here vs.
  // serial_spmv's single pass) differ only by double round-off re-ordered
  // across at most 5 terms -- 1e-12 is far above that and far below the
  // smallest discretisation feature this stencil could hide.
  check_two_partition_apply(t, global_n, /*split=*/128, x, /*tol=*/1e-12);
}

TEST(DistSparseMatrixApply, RandomNonsymmetricFiveNonzerosPerRow) {
  constexpr std::int64_t global_n = 50;
  std::mt19937 rng(41u);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  std::uniform_int_distribution<std::int64_t> col_pick(0, global_n - 1);

  std::vector<Triplet> t;
  for (std::int64_t r = 0; r < global_n; ++r) {
    std::set<std::int64_t> cols;
    while (cols.size() < 5) cols.insert(col_pick(rng));
    for (std::int64_t c : cols) t.push_back({r, c, u(rng)});
  }

  std::vector<double> x(static_cast<std::size_t>(global_n));
  for (double& v : x) v = u(rng);

  check_two_partition_apply(t, global_n, /*split=*/25, x, /*tol=*/1e-11);
}

TEST(DistSparseMatrixApply, DiagonalNeedsNoCommunicationAndMatchesTheAssembledTriplets) {
  const RowOwnership owners{{0, 3}};
  const std::vector<Triplet> t{
      {0, 0, 5.0}, {0, 4, -1.0}, {1, 1, 6.0}, {1, 5, -2.0}, {2, 3, -3.0},  // row 2 has NO diagonal
  };
  auto m = DistSparseMatrix::from_triplets(0, 3, 6, t, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok()) << m.error().message;
  DistVector d(3, make_serial_exchanger(), make_serial_reducer());
  m.value().diagonal(d);
  EXPECT_DOUBLE_EQ(d.data()[0], 5.0);
  EXPECT_DOUBLE_EQ(d.data()[1], 6.0);
  EXPECT_DOUBLE_EQ(d.data()[2], 0.0);  // structurally absent diagonal entry
}

// Row i's write to y overwrites x's owned block
// before every later row that reads the SAME entry as an off-diagonal
// input column (dist_sparse_matrix.hpp's own apply() doc states the
// rule) -- apply(v, v) is exactly the case the header forbids.
TEST(DistSparseMatrixApply, RefusesXAndYAliasing) {
  const RowOwnership owners{{0, 3}};
  const std::vector<Triplet> t{
      {0, 0, 5.0}, {0, 1, -1.0}, {1, 1, 6.0}, {1, 2, -2.0}, {2, 2, 7.0}, {2, 0, -3.0},
  };
  auto m = DistSparseMatrix::from_triplets(0, 3, 3, t, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok()) << m.error().message;
  DistVector v(3, make_serial_exchanger(), make_serial_reducer());
  v.data()[0] = 1.0;
  v.data()[1] = 2.0;
  v.data()[2] = 3.0;
  EXPECT_THROW((void)m.value().apply(v, v), fuse::InvariantViolation);
}

TEST(DistSparseMatrixBasics, LocalAndGlobalRowCounts) {
  const RowOwnership owners{{0, 3}};
  auto m = DistSparseMatrix::from_triplets(0, 3, 10, {}, make_serial_reducer(), owners);
  ASSERT_TRUE(m.ok());
  EXPECT_EQ(m.value().local_rows(), 3u);
  EXPECT_EQ(m.value().global_rows(), 10);
}

}  // namespace
