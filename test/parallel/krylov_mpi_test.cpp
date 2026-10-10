// "Krylov" track, tier 1: fuse::linalg::pcg/gmres (both header-only
// templates -- include/fuse/linalg/{pcg,gmres}.hpp) over a REAL
// MPI_COMM_WORLD, exercising make_mpi_reducer/make_mpi_exchanger
// (src/linalg/comm_mpi.cpp) through every dot, norm and ghost update the
// solvers issue.
//
// MPI CANNOT START IN THIS SANDBOX (docs/dev/testing.md): this file is
// compiled -- registered with fuse_add_mpi_test, RANKS 2 -- but never run
// here; outcomes are a host action.
//
// THE REFERENCE. Unlike test/parallel/linalg_mpi_test.cpp's own
// closed-form checks (every entry a pure function of its global index, so
// every rank reconstructs an identical answer with no communication), a
// Krylov solve's iterate has no such closed form -- it depends on the
// WHOLE problem's spectrum, not on any one row in isolation. So each test
// below checks the property a caller actually depends on: `pcg`/`gmres`
// report `converged`, and an INDEPENDENTLY recomputed residual (a fresh
// `apply`, exactly test/linalg/{pcg,gmres}_test.cpp's own tier-0 checks)
// is within the requested tolerance -- the same "recompute, never trust
// the recurrence alone" discipline those solvers' own header comments
// describe, now checked across a real reduction instead of a one-rank or
// in-process-fake one.

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/gmres.hpp"
#include "fuse/linalg/pcg.hpp"
#include "fuse/linalg/preconditioner.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

namespace {

using fuse::Result;
using fuse::linalg::DistSparseMatrix;
using fuse::linalg::DistVector;
using fuse::linalg::GhostPattern;
using fuse::linalg::gmres;
using fuse::linalg::GmresOptions;
using fuse::linalg::IdentityPreconditioner;
using fuse::linalg::pcg;
using fuse::linalg::PcgOptions;
using fuse::linalg::RowOwnership;
using fuse::linalg::SolverReport;
using fuse::linalg::Triplet;

/// A smooth, bounded, pure function of the GLOBAL index -- the same
/// closed-form right-hand side test/parallel/linalg_mpi_test.cpp's own
/// `g` is, so every rank builds its own slice of `b` with no
/// communication at all.
double g(std::int64_t i) { return std::sin(1.3 * static_cast<double>(i) + 0.2); }

std::pair<std::int64_t, std::int64_t> my_row_range(std::int64_t global_n) {
  const std::int64_t split = global_n / 2;
  return fuse::testing::rank() == 0 ? std::make_pair(std::int64_t{0}, split)
                                    : std::make_pair(split, global_n);
}

TEST(KrylovMpi, PcgConvergesOnAPeriodicDiagonallyDominantSpdSystem) {
  FUSE_REQUIRE_RANKS(2);
  // Periodic 1D, three-point, diagonal 3 and neighbours -1: strictly
  // diagonally dominant (3 > 1 + 1) and symmetric, hence SPD by
  // Gershgorin -- unlike the bare graph Laplacian (diagonal 2), which is
  // singular (the constant vector is its null space) and cannot be a PCG
  // test matrix.
  constexpr std::int64_t global_n = 20;
  const auto [row_lo, row_hi] = my_row_range(global_n);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  std::vector<Triplet> t;
  for (std::int64_t r = row_lo; r < row_hi; ++r) {
    t.push_back({r, r, 3.0});
    t.push_back({r, (r - 1 + global_n) % global_n, -1.0});
    t.push_back({r, (r + 1) % global_n, -1.0});
  }
  const RowOwnership owners{{0, global_n / 2}};
  auto reducer = fuse::linalg::make_mpi_reducer(comm_f);
  auto m = DistSparseMatrix::from_triplets(row_lo, row_hi, global_n, t, reducer, owners);
  ASSERT_TRUE(m.ok()) << m.error().message;

  const auto owned_n = static_cast<std::size_t>(row_hi - row_lo);
  auto x_exchanger = fuse::linalg::make_mpi_exchanger(comm_f, m.value().ghost_pattern(), owned_n);
  DistVector b(owned_n, x_exchanger, reducer);
  for (std::size_t i = 0; i < owned_n; ++i) b.data()[i] = g(row_lo + static_cast<std::int64_t>(i));
  DistVector x(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, m.value().ghost_pattern(), owned_n),
              fuse::linalg::make_mpi_reducer(comm_f));

  PcgOptions opt;
  opt.rel_tol = 1e-10;
  opt.max_iterations = 500;
  Result<SolverReport> result = pcg(m.value(), IdentityPreconditioner{}, x, b, opt);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_TRUE(result.value().converged);

  DistVector ax(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n),
               fuse::linalg::make_mpi_reducer(comm_f));
  ASSERT_TRUE(m.value().apply(x, ax).ok());
  DistVector r(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n),
              fuse::linalg::make_mpi_reducer(comm_f));
  r.copy_from(b);
  r.axpy(-1.0, ax);
  EXPECT_LE(r.norm2() / b.norm2(), opt.rel_tol);
}

TEST(KrylovMpi, GmresConvergesOnAPeriodicNonsymmetricDiagonallyDominantSystem) {
  FUSE_REQUIRE_RANKS(2);
  // Periodic 1D, three-point, ASYMMETRIC neighbours (-1 and -1.5) with
  // diagonal 3: still strictly diagonally dominant (3 > 1 + 1.5), but not
  // symmetric, so PCG does not apply and this is a GMRES case.
  constexpr std::int64_t global_n = 20;
  const auto [row_lo, row_hi] = my_row_range(global_n);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);

  std::vector<Triplet> t;
  for (std::int64_t r = row_lo; r < row_hi; ++r) {
    t.push_back({r, r, 3.0});
    t.push_back({r, (r - 1 + global_n) % global_n, -1.0});
    t.push_back({r, (r + 1) % global_n, -1.5});
  }
  const RowOwnership owners{{0, global_n / 2}};
  auto reducer = fuse::linalg::make_mpi_reducer(comm_f);
  auto m = DistSparseMatrix::from_triplets(row_lo, row_hi, global_n, t, reducer, owners);
  ASSERT_TRUE(m.ok()) << m.error().message;

  const auto owned_n = static_cast<std::size_t>(row_hi - row_lo);
  auto x_exchanger = fuse::linalg::make_mpi_exchanger(comm_f, m.value().ghost_pattern(), owned_n);
  DistVector b(owned_n, x_exchanger, reducer);
  for (std::size_t i = 0; i < owned_n; ++i) b.data()[i] = g(row_lo + static_cast<std::int64_t>(i));
  DistVector x(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, m.value().ghost_pattern(), owned_n),
              fuse::linalg::make_mpi_reducer(comm_f));

  GmresOptions opt;
  opt.rel_tol = 1e-10;
  opt.max_iterations = 500;
  opt.restart = 20;
  Result<SolverReport> result = gmres(m.value(), IdentityPreconditioner{}, x, b, opt);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_TRUE(result.value().converged);

  DistVector ax(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n),
               fuse::linalg::make_mpi_reducer(comm_f));
  ASSERT_TRUE(m.value().apply(x, ax).ok());
  DistVector r(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n),
              fuse::linalg::make_mpi_reducer(comm_f));
  r.copy_from(b);
  r.axpy(-1.0, ax);
  EXPECT_LE(r.norm2() / b.norm2(), opt.rel_tol);
}

}  // namespace
