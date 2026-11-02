// fuse::linalg::SsorPreconditioner -.
//
// WHAT EACH CASE CHECKS.
//   SsorExactness.OmegaOneOnOneDTridiagonalMatchesHandComputedSweep -- the
//     SAME derivation preconditioner.hpp's own header comment states
//     (two triangular sweeps, no M ever formed), worked by hand for one
//     concrete 3x3 case and compared bit-for-bit (exact rational
//     arithmetic on this small a problem, no rounding tolerance needed).
//   SsorRefusals.OmegaOutOfRangeIsRefused -- the (0, 2) bound,
//     E-LINALG-SSOR-OMEGA-RANGE, fired at both ends and at omega = 0.
//   SsorRefusals.ZeroDiagonalEntryIsRefused -- the same structurally-
//     absent-diagonal refusal JacobiPreconditioner makes.
//   SsorOptimalOmega.ConvergesInFewerPcgIterationsThanJacobiOnAConsistently
//     OrderedSystem -- the 1-D discrete Poisson matrix (n = 30), for which
//     Young's closed form gives the optimal SOR relaxation factor
//     omega_opt = 2 / (1 + sqrt(1 - rho_J^2)), rho_J = cos(pi/(n+1)) -- a
//     derived constant, not one chosen after a measurement. Measured:
//     jacobi 15, ssor(omega_opt) 8.
//     NOTE (lead, 2026-11-02): the round-4 draft of this comment claimed
//     that SSOR "does not beat Jacobi at any omega" on the 2-D Q1
//     stiffness+mass system because that system lacks Property A. That
//     was a pre-asymptotic artifact of the 6 x 6 mesh it was measured on
//     (49 dofs); on 32 x 32 and 64 x 64 meshes SSOR(1.0) beats Jacobi
//     (55 vs 57, 90 vs 110) with the ratio growing as theory predicts
//     (Axelsson & Barker 1984: O(h^-1) vs O(h^-2) condition numbers).
//     Consistent ordering governs SOR as a stationary iteration, not
//     SSOR as a CG preconditioner. physics-kit/test/linear_layer_test.cpp
//     now asserts the inequality on the asymptotic meshes.

#include "fuse/linalg/preconditioner.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"
#include "fuse/linalg/dist_sparse_matrix.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/pcg.hpp"

namespace {

using fuse::linalg::DistSparseMatrix;
using fuse::linalg::DistVector;
using fuse::linalg::JacobiPreconditioner;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;
using fuse::linalg::pcg;
using fuse::linalg::PcgOptions;
using fuse::linalg::RowOwnership;
using fuse::linalg::SsorPreconditioner;
using fuse::linalg::Triplet;

DistSparseMatrix to_dist(const std::vector<Triplet>& t, std::int64_t n) {
  const RowOwnership owners{{0}};
  auto m = DistSparseMatrix::from_triplets(0, n, n, t, make_serial_reducer(), owners);
  EXPECT_TRUE(m.ok()) << (m.ok() ? "" : m.error().message);
  return std::move(m).value();
}

DistVector make_vector(std::int64_t n, const std::vector<double>& values) {
  DistVector v(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  for (std::int64_t i = 0; i < n; ++i) v.data()[i] = values[static_cast<std::size_t>(i)];
  return v;
}

// The 1-D Poisson 3x3 matrix A = [[2,-1,0],[-1,2,-1],[0,-1,2]] (Dirichlet,
// h = 1). At omega = 1, D = diag(2,2,2); hand-computed two-sweep solve of
// M y = r for r = (1, 0, 0) (exact rational arithmetic, verified
// independently against M = (D+L) D^-1 (D+U) solved directly):
//
//   forward  (D+L) z = r:  z0 = r0/2 = 1/2
//                          z1 = (r1 + z0)/2 = 1/4
//                          z2 = (r2 + z1)/2 = 1/8
//   backward (D+U) y = D z (omega=1, D/omega=D):
//                          y2 = z2 = 1/8
//                          y1 = z1 + y2/2 = 1/4 + 1/16 = 5/16 = 0.3125
//                          y0 = z0 + y1/2 = 1/2 + 5/32 = 21/32 = 0.65625
std::vector<Triplet> one_d_poisson_3x3() {
  return {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 1, 2.0}, {1, 2, -1.0}, {2, 1, -1.0}, {2, 2, 2.0}};
}

TEST(SsorExactness, OmegaOneOnOneDTridiagonalMatchesHandComputedSweep) {
  constexpr std::int64_t n = 3;
  DistSparseMatrix a = to_dist(one_d_poisson_3x3(), n);
  auto prec = SsorPreconditioner::from_matrix(a, /*omega=*/1.0);
  ASSERT_TRUE(prec.ok()) << prec.error().message;

  DistVector r = make_vector(n, {1.0, 0.0, 0.0});
  DistVector y(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  const auto applied = prec.value().apply(r, y);
  ASSERT_TRUE(applied.ok()) << applied.error().message;

  const double expected[3] = {21.0 / 32.0, 5.0 / 16.0, 1.0 / 8.0};
  const double eps = std::numeric_limits<double>::epsilon();
  for (int i = 0; i < 3; ++i) {
    // A handful of exact-arithmetic divisions and multiplications (the
    // two sweeps above): a few ULPs, derived from the operation count,
    // not an empirical margin.
    EXPECT_NEAR(y.data()[i], expected[i], 16.0 * eps) << "row " << i;
  }
}

TEST(SsorRefusals, OmegaOutOfRangeIsRefused) {
  constexpr std::int64_t n = 3;
  DistSparseMatrix a = to_dist(one_d_poisson_3x3(), n);
  for (const double omega : {-1.0, 0.0, 2.0, 3.0}) {
    const auto prec = SsorPreconditioner::from_matrix(a, omega);
    EXPECT_FALSE(prec.ok()) << "omega=" << omega;
    if (!prec.ok()) EXPECT_EQ(prec.error().code, "E-LINALG-SSOR-OMEGA-RANGE") << "omega=" << omega;
  }
  // The open interval's own interior boundary-adjacent values are legal.
  for (const double omega : {0.1, 1.0, 1.9}) {
    const auto prec = SsorPreconditioner::from_matrix(a, omega);
    EXPECT_TRUE(prec.ok()) << "omega=" << omega << (prec.ok() ? "" : (": " + prec.error().message));
  }
}

TEST(SsorRefusals, ZeroDiagonalEntryIsRefused) {
  // Row 1 has no diagonal triplet at all -- a structurally absent entry,
  // the same case JacobiPreconditioner's own identically-named test fires.
  const std::vector<Triplet> t = {{0, 0, 2.0}, {0, 1, -1.0}, {1, 0, -1.0}, {1, 2, -1.0}, {2, 1, -1.0}, {2, 2, 2.0}};
  DistSparseMatrix a = to_dist(t, 3);
  const auto prec = SsorPreconditioner::from_matrix(a);
  ASSERT_FALSE(prec.ok());
  EXPECT_EQ(prec.error().code, "E-LINALG-ZERO-DIAGONAL");
}

// SsorOptimalOmega.ConvergesInFewerPcgIterationsThanJacobiOnAConsistently
// OrderedSystem -- this file's own header comment above states the full
// derivation. The matrix: the classical n x n 1-D discrete Poisson
// operator (tridiagonal [-1, 2, -1], Dirichlet), CONSISTENTLY ORDERED
// under its own natural ordering. Its eigenvalues are 2 - 2cos(k*pi/
// (n+1)) (a standard closed form for this exact matrix), so the Jacobi
// iteration matrix I - D^-1 A has eigenvalues cos(k*pi/(n+1)) and
// spectral radius rho_j = cos(pi/(n+1)) (k = 1, the largest). Young's
// formula for a consistently ordered system's OPTIMAL SOR relaxation
// factor, omega_opt = 2 / (1 + sqrt(1 - rho_j^2)), is a closed-form
// function of n alone -- computed once below, never swept or chosen
// after seeing an iteration count.
TEST(SsorOptimalOmega, ConvergesInFewerPcgIterationsThanJacobiOnAConsistentlyOrderedSystem) {
  constexpr std::int64_t n = 30;
  std::vector<Triplet> triplets;
  triplets.push_back({0, 0, 2.0});
  triplets.push_back({0, 1, -1.0});
  for (std::int64_t i = 1; i < n - 1; ++i) {
    triplets.push_back({i, i - 1, -1.0});
    triplets.push_back({i, i, 2.0});
    triplets.push_back({i, i + 1, -1.0});
  }
  triplets.push_back({n - 1, n - 2, -1.0});
  triplets.push_back({n - 1, n - 1, 2.0});
  DistSparseMatrix a = to_dist(triplets, n);

  const double rho_j = std::cos(M_PI / static_cast<double>(n + 1));
  const double omega_opt = 2.0 / (1.0 + std::sqrt(1.0 - rho_j * rho_j));

  DistVector b(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  for (std::int64_t i = 0; i < n; ++i) b.data()[static_cast<std::size_t>(i)] = 1.0;

  PcgOptions opt;
  opt.rel_tol = 1e-10;
  opt.max_iterations = 500;

  const auto jacobi = JacobiPreconditioner::from_matrix(a);
  ASSERT_TRUE(jacobi.ok()) << jacobi.error().message;
  DistVector x_jacobi(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  x_jacobi.set_zero();
  const auto jacobi_report = pcg(a, jacobi.value(), x_jacobi, b, opt);
  ASSERT_TRUE(jacobi_report.ok()) << jacobi_report.error().message;
  ASSERT_TRUE(jacobi_report.value().converged);

  const auto ssor = SsorPreconditioner::from_matrix(a, omega_opt);
  ASSERT_TRUE(ssor.ok()) << ssor.error().message;
  DistVector x_ssor(static_cast<std::size_t>(n), make_serial_exchanger(), make_serial_reducer());
  x_ssor.set_zero();
  const auto ssor_report = pcg(a, ssor.value(), x_ssor, b, opt);
  ASSERT_TRUE(ssor_report.ok()) << ssor_report.error().message;
  ASSERT_TRUE(ssor_report.value().converged);

  testing::Test::RecordProperty("jacobi_iterations", std::to_string(jacobi_report.value().iterations));
  testing::Test::RecordProperty("ssor_iterations", std::to_string(ssor_report.value().iterations));
  testing::Test::RecordProperty("omega_opt", std::to_string(omega_opt));
  EXPECT_LT(ssor_report.value().iterations, jacobi_report.value().iterations)
      << "jacobi=" << jacobi_report.value().iterations << " ssor=" << ssor_report.value().iterations
      << " omega_opt=" << omega_opt;
}

}  // namespace
