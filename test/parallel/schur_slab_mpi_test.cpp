// "Schur-slab" track, tier 1: fuse::linalg::SchurSlabSolver and
// Reducer::allgather's real MPI implementation (src/linalg/comm_mpi.cpp),
// over a REAL MPI_COMM_WORLD at 2 and 3 ranks.
//
// MPI CANNOT START IN THIS SANDBOX (docs/dev/testing.md): this file is
// compiled -- registered with fuse_add_mpi_test, RANKS 2 3 -- but never run
// here; outcomes are a host action. The properties measured here are
// exactly the ones test/linalg/schur_slab_test.cpp already measures with
// the in-process ring-allgather fake (requirement 2's 2- and 3-partition
// agreement with the whole-grid reference, and the periodic wrap); this
// file repeats them over the real collective instead of the fake one,
// using whatever rank count the ctest row invokes it at
// (`fuse::testing::size()`), not a hardcoded 2 or 3.

#include "fuse/linalg/schur_slab.hpp"

#include <Eigen/SparseCholesky>

#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"

#include "fuse_test_main.hpp"

#include "brusselator_physics.hpp"

namespace {

using fuse::linalg::DistVector;
using fuse::linalg::RowOwnership;
using fuse::linalg::SchurSlabSolver;

/// The same contiguous row-band split src/backend/native_generators.cpp's
/// own `rank_slice` computes -- see test/linalg/schur_slab_test.cpp's own
/// copy of this helper for why it is reproduced rather than linked.
std::pair<std::int64_t, std::int64_t> row_band(std::int64_t ny, int rank, int num_ranks) {
  if (num_ranks <= 1) return {0, ny};
  const std::int64_t base = ny / num_ranks;
  const std::int64_t extra = ny % num_ranks;
  const std::int64_t r = rank;
  const std::int64_t lo = (r < extra) ? r * (base + 1) : extra * (base + 1) + (r - extra) * base;
  const std::int64_t size = base + (r < extra ? 1 : 0);
  return {lo, lo + size};
}

RowOwnership row_ownership_for(std::int64_t ny, int num_ranks) {
  RowOwnership owners;
  owners.row_lo.resize(static_cast<std::size_t>(num_ranks));
  for (int r = 0; r < num_ranks; ++r) owners.row_lo[static_cast<std::size_t>(r)] = row_band(ny, r, num_ranks).first;
  return owners;
}

Eigen::VectorXd whole_grid_reference_solve(int nx, int ny, double coefficient,
                                           const Eigen::VectorXd& rhs) {
  brusselator::Sparse laplacian = brusselator::periodic_laplacian(nx, ny);
  Eigen::SparseMatrix<double> matrix = (-coefficient) * laplacian;
  const int n = nx * ny;
  for (int i = 0; i < n; ++i) matrix.coeffRef(i, i) += 1.0;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
  factor.compute(matrix);
  EXPECT_EQ(factor.info(), Eigen::Success);
  return factor.solve(rhs);
}

/// Deterministic, rank-independent (a pure function of global index), so
/// every rank reconstructs the SAME rhs with no communication -- the same
/// closed-form-reference discipline test/parallel/linalg_mpi_test.cpp's
/// own `g(i)` uses.
double g(std::int64_t i) { return std::sin(1.7 * static_cast<double>(i) + 0.4); }

TEST(SchurSlabSolverMpi, AgreesWithWholeGridReferenceAtTheInvokedRankCount) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  constexpr int nx = 6;
  constexpr std::int64_t ny = 18;
  constexpr double coefficient = 0.6;
  const int nranks = fuse::testing::size();
  const int rank = fuse::testing::rank();
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const RowOwnership owners = row_ownership_for(ny, nranks);
  const auto [j_lo, j_hi] = row_band(ny, rank, nranks);

  auto reducer = fuse::linalg::make_mpi_reducer(comm_f);
  auto solver = SchurSlabSolver::factor(nx, ny, j_lo, j_hi, coefficient, owners, reducer);
  ASSERT_TRUE(solver.ok()) << solver.error().message;

  Eigen::VectorXd rhs_global(nx * ny);
  for (std::int64_t i = 0; i < nx * ny; ++i) rhs_global(i) = g(i);

  const std::size_t owned_n = static_cast<std::size_t>((j_hi - j_lo) * nx);
  DistVector rhs_owned(owned_n, fuse::linalg::make_serial_exchanger(), reducer);
  for (std::size_t i = 0; i < owned_n; ++i) {
    rhs_owned.data()[i] = rhs_global(j_lo * nx + static_cast<std::int64_t>(i));
  }
  DistVector x_owned(owned_n, fuse::linalg::make_serial_exchanger(), reducer);
  ASSERT_TRUE(solver.value().solve(rhs_owned, x_owned).ok());

  const Eigen::VectorXd expect = whole_grid_reference_solve(nx, static_cast<int>(ny), coefficient, rhs_global);
  for (std::size_t i = 0; i < owned_n; ++i) {
    EXPECT_NEAR(x_owned.data()[i], expect(j_lo * nx + static_cast<std::int64_t>(i)), 1e-9)
        << "rank " << rank << " local index " << i;
  }
}

TEST(SchurSlabSolverMpi, PeriodicWrapForcingOnTheLastRankReachesRankZero) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  constexpr int nx = 6;
  constexpr std::int64_t ny = 18;
  constexpr double coefficient = 0.6;
  const int nranks = fuse::testing::size();
  const int rank = fuse::testing::rank();
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  const RowOwnership owners = row_ownership_for(ny, nranks);
  const auto [j_lo, j_hi] = row_band(ny, rank, nranks);

  auto reducer = fuse::linalg::make_mpi_reducer(comm_f);
  auto solver = SchurSlabSolver::factor(nx, ny, j_lo, j_hi, coefficient, owners, reducer);
  ASSERT_TRUE(solver.ok()) << solver.error().message;

  Eigen::VectorXd rhs_global = Eigen::VectorXd::Zero(nx * ny);
  rhs_global((ny - 1) * nx + 1) = 9.0;  // last row of the grid, one column

  const std::size_t owned_n = static_cast<std::size_t>((j_hi - j_lo) * nx);
  DistVector rhs_owned(owned_n, fuse::linalg::make_serial_exchanger(), reducer);
  for (std::size_t i = 0; i < owned_n; ++i) {
    rhs_owned.data()[i] = rhs_global(j_lo * nx + static_cast<std::int64_t>(i));
  }
  DistVector x_owned(owned_n, fuse::linalg::make_serial_exchanger(), reducer);
  ASSERT_TRUE(solver.value().solve(rhs_owned, x_owned).ok());

  const Eigen::VectorXd expect = whole_grid_reference_solve(nx, static_cast<int>(ny), coefficient, rhs_global);
  for (std::size_t i = 0; i < owned_n; ++i) {
    EXPECT_NEAR(x_owned.data()[i], expect(j_lo * nx + static_cast<std::int64_t>(i)), 1e-9)
        << "rank " << rank << " local index " << i;
  }
}

TEST(SchurSlabSolverMpi, OneRowRowBandRefuses) {
  // Deliberately uneven row_lo -- rank size()-1 owns exactly 1 row -- so
  // this fires regardless of which rank count (>=2) the ctest row uses.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  const int nranks = fuse::testing::size();
  const int rank = fuse::testing::rank();
  constexpr int nx = 4;
  const std::int64_t ny = nranks * 3 + 1;  // enough rows for everyone but the last to have >=2

  RowOwnership owners;
  owners.row_lo.resize(static_cast<std::size_t>(nranks));
  for (int r = 0; r < nranks; ++r) {
    owners.row_lo[static_cast<std::size_t>(r)] = (r < nranks - 1) ? r * 3 : ny - 1;
  }
  const std::int64_t j_lo = owners.row_lo[static_cast<std::size_t>(rank)];
  const std::int64_t j_hi = (rank + 1 < nranks) ? owners.row_lo[static_cast<std::size_t>(rank + 1)] : ny;

  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);
  auto reducer = fuse::linalg::make_mpi_reducer(comm_f);
  auto result = SchurSlabSolver::factor(nx, ny, j_lo, j_hi, 0.5, owners, reducer);

  // Review 2026-10-11 NL-3: under the fixed factor() (commit c11ac18,
  // src/linalg/schur_slab.cpp:221-247) every rank scans ALL bands from
  // `row_ownership` and refuses identically, with zero collectives --
  // NOT "this rank's own band happens to be fine, so only the undersized
  // rank refuses while its peers' own factor() legitimately succeeds and
  // carries on" (the comment this replaces, which described the PRE-FIX
  // behaviour that deadlocked on a real host: one rank refusing locally
  // while its peers entered the allgather below it never reaches). Every
  // rank here must see the SAME refusal, exactly as the tier-0 sibling
  // (`SchurSlabSolverRefusal.
  // OneRowBandRefusesOnEveryRankWithoutStrandingAnyoneInACollective`,
  // test/linalg/schur_slab_test.cpp:424-443) already asserts.
  ASSERT_FALSE(result.ok()) << "rank " << rank << ": every rank must refuse together, "
                                                   "regardless of its own band's size";
  EXPECT_EQ(result.error().code, "E-LINALG-SCHUR-SLAB");
}

TEST(LinalgComm, MpiReducerAllgatherConcatenatesInRankOrder) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  const int nranks = fuse::testing::size();
  const int rank = fuse::testing::rank();
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);
  auto reducer = fuse::linalg::make_mpi_reducer(comm_f);

  const std::vector<double> send{static_cast<double>(rank), static_cast<double>(rank) + 0.5};
  std::vector<double> recv(2 * static_cast<std::size_t>(nranks), -1.0);
  reducer->allgather(send.data(), send.size(), recv.data());

  for (int r = 0; r < nranks; ++r) {
    EXPECT_DOUBLE_EQ(recv[static_cast<std::size_t>(2 * r)], static_cast<double>(r));
    EXPECT_DOUBLE_EQ(recv[static_cast<std::size_t>(2 * r + 1)], static_cast<double>(r) + 0.5);
  }
}

}  // namespace
