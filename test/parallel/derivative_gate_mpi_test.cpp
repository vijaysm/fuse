// "DerivativeGate" track, tier 1: the adjoint dot-product identity
// (include/fuse/linalg/derivative_gate.hpp) over a REAL MPI_COMM_WORLD
// fuse::linalg::DistSparseMatrix/DistVector, at RANKS 2 and 3.
//
// MPI CANNOT START IN THIS SANDBOX (docs/dev/testing.md): this file is
// compiled -- registered with fuse_add_mpi_test, RANKS 2 3 -- but never run
// here; outcomes are a host action.
//
// WHY THIS DOES NOT CALL fuse::linalg::adjoint_gate DIRECTLY.
// `adjoint_gate` (derivative_gate.hpp) takes the WHOLE vector as a host
// span, which is the right shape for the tier-0 adapters
// (test/linalg/derivative_gate_test.cpp) that hold every operator's full
// input/output in one process, but is the wrong shape for a genuinely
// distributed check: gathering the full vector onto every rank just to
// call a host-span function would defeat the point of testing the
// DISTRIBUTED path at all. Instead this file computes the identical
// SCALAR identity, `<A x, y> == <x, A^T y>`, using `DistVector::dot`
// directly -- which already performs the cross-rank reduction
// (`Reducer::sum`, dist_vector.hpp) test/parallel/linalg_mpi_test.cpp's own
// `ApplyMatchesTheWholeDomainClosedFormStencilReference` already exercises
// for this same matrix shape -- against a bound of the SAME derived kind
// `adjoint_gate` uses (`c * eps * ||A||_2 * ||x|| * ||y||`), with `||A||_2`
// known in closed form for this stencil rather than estimated: the
// periodic 1D "2, -1, -1" stencil (the same operator
// test/parallel/linalg_mpi_test.cpp already builds) is symmetric
// (A^T = A, so apply() stands in for apply_transpose() on both sides) with
// eigenvalues `2 - 2cos(2*pi*k/n)`, whose magnitude is maximised at
// `k = floor(n/2)`; for the even `global_n` every case below uses, that
// maximum is exactly `2 - 2cos(pi) = 4`.

#include "fuse/linalg/derivative_gate.hpp"

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

double g(std::int64_t i) { return std::sin(1.3 * static_cast<double>(i) + 0.2); }
double h(std::int64_t i) { return std::cos(0.7 * static_cast<double>(i) - 0.4); }

/// This rank's row range over `[0, global_n)`, split as evenly as possible
/// across `fuse::testing::size()` ranks -- the same contiguous
/// row-band convention `RowOwnership` uses throughout `fuse::linalg`.
std::pair<std::int64_t, std::int64_t> my_row_range(std::int64_t global_n) {
  const int rank = fuse::testing::rank();
  const int nranks = fuse::testing::size();
  const std::int64_t base = global_n / nranks;
  const std::int64_t extra = global_n % nranks;
  const std::int64_t lo =
      (rank < extra) ? rank * (base + 1) : extra * (base + 1) + (rank - extra) * base;
  const std::int64_t size = base + (rank < extra ? 1 : 0);
  return {lo, lo + size};
}

RowOwnership row_ownership_for(std::int64_t global_n, int nranks) {
  RowOwnership owners;
  owners.row_lo.resize(static_cast<std::size_t>(nranks));
  std::int64_t base = global_n / nranks, extra = global_n % nranks;
  for (int r = 0; r < nranks; ++r) {
    owners.row_lo[static_cast<std::size_t>(r)] =
        (r < extra) ? r * (base + 1) : extra * (base + 1) + (r - extra) * base;
  }
  return owners;
}

void check_adjoint_identity_over_mpi(std::int64_t global_n) {
  const auto [row_lo, row_hi] = my_row_range(global_n);
  const int comm_f = MPI_Comm_c2f(MPI_COMM_WORLD);
  const auto reducer = fuse::linalg::make_mpi_reducer(comm_f);
  const int nranks = fuse::testing::size();

  std::vector<Triplet> t;
  for (std::int64_t r = row_lo; r < row_hi; ++r) {
    t.push_back({r, r, 2.0});
    t.push_back({r, (r - 1 + global_n) % global_n, -1.0});
    t.push_back({r, (r + 1) % global_n, -1.0});
  }
  const RowOwnership owners = row_ownership_for(global_n, nranks);
  auto m = DistSparseMatrix::from_triplets(row_lo, row_hi, global_n, t, reducer, owners);
  ASSERT_TRUE(m.ok()) << m.error().message;

  const auto owned_n = static_cast<std::size_t>(row_hi - row_lo);
  auto x_exch = fuse::linalg::make_mpi_exchanger(comm_f, m.value().ghost_pattern(), owned_n);
  auto y_exch = fuse::linalg::make_mpi_exchanger(comm_f, m.value().ghost_pattern(), owned_n);
  DistVector x(owned_n, x_exch, reducer);
  DistVector y(owned_n, y_exch, reducer);
  for (std::size_t i = 0; i < owned_n; ++i) {
    x.data()[i] = g(row_lo + static_cast<std::int64_t>(i));
    y.data()[i] = h(row_lo + static_cast<std::int64_t>(i));
  }

  DistVector ax(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n),
               reducer);
  DistVector aty(owned_n, fuse::linalg::make_mpi_exchanger(comm_f, GhostPattern{}, owned_n),
                reducer);
  ASSERT_TRUE(m.value().apply(x, ax).ok());
  // A^T = A for this symmetric stencil: apply() stands in for
  // apply_transpose() on y too.
  ASSERT_TRUE(m.value().apply(y, aty).ok());

  const double lhs = y.dot(ax);    // <A x, y>, globally reduced by DistVector::dot
  const double rhs = x.dot(aty);   // <x, A^T y>, globally reduced
  const double x_norm = x.norm2();  // globally reduced
  const double y_norm = y.norm2();
  constexpr double norm_a = 4.0;  // closed form, this file's own header comment
  const double eps = std::numeric_limits<double>::epsilon();
  const double bound = fuse::linalg::kAdjointGateConstant * eps * norm_a * x_norm * y_norm;
  EXPECT_LT(std::abs(lhs - rhs), bound)
      << "lhs=" << lhs << " rhs=" << rhs << " bound=" << bound;
}

TEST(DerivativeGateMpi, AdjointIdentityHoldsOverARealCommWorld) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  check_adjoint_identity_over_mpi(/*global_n=*/32);
}

}  // namespace
